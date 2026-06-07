"""S.1 gate: the ``dist_<NP>`` partition reader (:mod:`fesom_jax.partit`).

Reads the bit-identical CORE2 partitions (``dist_2``/``dist_4``) and asserts the
structural invariants that S.2 (sharded mesh) relies on: the node partition is
unique + complete, the element/edge partitions are redundant at the boundary,
the communicator index lists are 0-based and in range, and the per-file rank
headers / counts are self-consistent. A format-fixture read of a *different*
mesh (``nares4/dist_768``) confirms the reader is not over-fit to CORE2, and the
error paths (missing dir, ``npes`` mismatch) raise cleanly.

SKIPs cleanly when the partition files are absent so the suite stays green off
the Levante pool filesystem.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from fesom_jax import partit
from fesom_jax.partit import ComStruct, Partition

CORE2_DIR = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NARES4_DIR = Path("/work/ab0995/a270088/meshes/nares4")

# Known CORE2 global counts (the mesh-export meta.txt; pinned so a partition-vs-mesh
# mismatch is caught here).
CORE2 = dict(nod2D=126858, elem2D=244659, edge2D=371644)

core2_avail = pytest.mark.skipif(
    not (CORE2_DIR / "dist_2").is_dir(),
    reason=f"CORE2 dist partitions missing under {CORE2_DIR}",
)


@pytest.fixture(scope="module")
def part2() -> Partition:
    return partit.read_partition(CORE2_DIR, 2)


@pytest.fixture(scope="module")
def part4() -> Partition:
    return partit.read_partition(CORE2_DIR, 4)


# --------------------------------------------------------------------------
# Global counts + the ownership invariants
# --------------------------------------------------------------------------
@core2_avail
@pytest.mark.parametrize("npes", [2, 4])
def test_global_counts(npes):
    p = partit.read_partition(CORE2_DIR, npes)
    assert p.npes == npes
    assert (p.nod2D, p.elem2D, p.edge2D) == (CORE2["nod2D"], CORE2["elem2D"], CORE2["edge2D"])
    # per-rank count arrays are shape (npes,)
    for arr in (p.myDim_nod2D, p.eDim_nod2D, p.myDim_elem2D, p.eDim_elem2D,
                p.eXDim_elem2D, p.myDim_edge2D, p.eDim_edge2D):
        assert arr.shape == (npes,)
    # the id-list / communicator tuples are npes long
    assert len(p.myList_nod2D) == len(p.com_nod2D) == npes
    assert len(p.myList_elem2D) == len(p.com_elem2D_full) == npes


@core2_avail
@pytest.mark.parametrize("npes", [2, 4])
def test_nodes_unique_partition(npes):
    """Nodes are uniquely partitioned: Σ myDim == nod2D and the interior lists
    are disjoint and cover [0, nod2D)."""
    p = partit.read_partition(CORE2_DIR, npes)
    assert int(p.myDim_nod2D.sum()) == p.nod2D
    interiors = [p.myList_nod2D[r][: int(p.myDim_nod2D[r])] for r in range(npes)]
    union = np.concatenate(interiors)
    assert union.size == p.nod2D                       # disjoint (no double-count)
    assert np.array_equal(np.unique(union), np.arange(p.nod2D))   # complete + dense


@core2_avail
@pytest.mark.parametrize("npes", [2, 4])
def test_elem_edge_redundant_but_complete(npes):
    """Elements/edges are redundantly owned at the boundary (Σ myDim > count),
    yet the interior union still covers [0, count) densely."""
    p = partit.read_partition(CORE2_DIR, npes)
    for mydim, mylist, count in (
        (p.myDim_elem2D, p.myList_elem2D, p.elem2D),
        (p.myDim_edge2D, p.myList_edge2D, p.edge2D),
    ):
        interiors = [mylist[r][: int(mydim[r])] for r in range(npes)]
        union = np.concatenate(interiors)
        assert union.size > count                       # boundary redundancy
        assert np.array_equal(np.unique(union), np.arange(count))  # complete + dense


# --------------------------------------------------------------------------
# my_list: global-id lists are 0-based, sorted, in range
# --------------------------------------------------------------------------
@core2_avail
@pytest.mark.parametrize("npes", [2, 4])
def test_mylist_shifted_sorted_inrange(npes):
    p = partit.read_partition(CORE2_DIR, npes)
    for r in range(npes):
        for mylist, mydim, edim, count in (
            (p.myList_nod2D[r], p.myDim_nod2D[r], p.eDim_nod2D[r], p.nod2D),
            (p.myList_edge2D[r], p.myDim_edge2D[r], p.eDim_edge2D[r], p.edge2D),
        ):
            md, ed = int(mydim), int(edim)
            assert mylist.shape == (md + ed,)
            assert mylist.min() >= 0 and mylist.max() < count   # 0-based, in range
            assert np.unique(mylist).size == mylist.size        # unique within rank
            # interior owned gids are strictly ascending (FESOM numbers owned
            # entities in increasing global order — "monotone-consistent"). The
            # halo order is a partitioner detail (sorted only within a neighbour
            # segment for ≥3 ranks), so it is not asserted globally here.
            assert np.all(np.diff(mylist[:md]) > 0)
        # elem list also includes eXDim
        me = p.myList_elem2D[r]
        n_e = int(p.myDim_elem2D[r] + p.eDim_elem2D[r] + p.eXDim_elem2D[r])
        assert me.shape == (n_e,)
        assert me.min() >= 0 and me.max() < p.elem2D


@core2_avail
@pytest.mark.parametrize("npes", [2, 4])
def test_part_prefix_consistent(npes):
    """part is a 0-based node-count prefix: part[0]==0, part[-1]==nod2D, and the
    per-rank gaps equal myDim_nod2D (the rpart.out counts == my_list counts)."""
    p = partit.read_partition(CORE2_DIR, npes)
    assert p.part.shape == (npes + 1,)
    assert p.part[0] == 0 and p.part[-1] == p.nod2D
    np.testing.assert_array_equal(np.diff(p.part), p.myDim_nod2D)


# --------------------------------------------------------------------------
# com_info: the communicator index lists
# --------------------------------------------------------------------------
def _check_com(cs: ComStruct, *, rlist_size: int, myDim: int, eDim: int,
               npes: int, rank: int):
    # rPE / sPE are 0-based MPI ranks, never self
    for pe in (cs.rPE, cs.sPE):
        assert pe.min(initial=0) >= 0 and pe.max(initial=0) < npes
        assert rank not in pe.tolist()
    # rptr / sptr are 0-based cumulative offsets starting at 0, non-decreasing
    assert cs.rptr[0] == 0 and cs.sptr[0] == 0
    assert np.all(np.diff(cs.rptr) >= 0) and np.all(np.diff(cs.sptr) >= 0)
    assert cs.rptr.shape == (cs.rPE.size + 1,)
    assert cs.sptr.shape == (cs.sPE.size + 1,)
    # rlist length == eDim == rptr[-1]; receives into the LOCAL halo lanes
    assert cs.rlist.size == rlist_size
    assert int(cs.rptr[-1]) == rlist_size
    if cs.rlist.size:
        assert cs.rlist.min() >= myDim and cs.rlist.max() < myDim + eDim
    # slist length == sptr[-1]; sends our LOCAL interior lanes
    assert cs.slist.size == int(cs.sptr[-1])
    if cs.slist.size:
        assert cs.slist.min() >= 0 and cs.slist.max() < myDim


@core2_avail
@pytest.mark.parametrize("npes", [2, 4])
def test_com_structs_inrange(npes):
    p = partit.read_partition(CORE2_DIR, npes)
    for r in range(npes):
        _check_com(p.com_nod2D[r], rlist_size=int(p.eDim_nod2D[r]),
                   myDim=int(p.myDim_nod2D[r]), eDim=int(p.eDim_nod2D[r]),
                   npes=npes, rank=r)
        _check_com(p.com_elem2D[r], rlist_size=int(p.eDim_elem2D[r]),
                   myDim=int(p.myDim_elem2D[r]), eDim=int(p.eDim_elem2D[r]),
                   npes=npes, rank=r)
        # elem2D_full halo = eDim + eXDim
        _check_com(p.com_elem2D_full[r],
                   rlist_size=int(p.eDim_elem2D[r] + p.eXDim_elem2D[r]),
                   myDim=int(p.myDim_elem2D[r]),
                   eDim=int(p.eDim_elem2D[r] + p.eXDim_elem2D[r]),
                   npes=npes, rank=r)


@core2_avail
def test_halo_nodes_owned_elsewhere(part2):
    """Cross-rank consistency: every halo node gid is interior-owned by a
    DIFFERENT rank (the partition is globally consistent)."""
    p = part2
    owner = np.full(p.nod2D, -1, dtype=np.int64)
    for r in range(p.npes):
        owner[p.myList_nod2D[r][: int(p.myDim_nod2D[r])]] = r
    assert (owner >= 0).all()                       # every node owned exactly once
    for r in range(p.npes):
        md = int(p.myDim_nod2D[r])
        halo_gids = p.myList_nod2D[r][md:]
        assert (owner[halo_gids] != r).all()        # halo lives on another rank
        assert (owner[halo_gids] >= 0).all()


# --------------------------------------------------------------------------
# synth_serial — the npes==1 identity partit
# --------------------------------------------------------------------------
def test_synth_serial_identity():
    p = partit.synth_serial(nod2D=5, elem2D=4, edge2D=3)
    assert p.npes == 1
    assert (p.nod2D, p.elem2D, p.edge2D) == (5, 4, 3)
    assert p.myDim_nod2D.tolist() == [5] and p.eDim_nod2D.tolist() == [0]
    assert p.myDim_elem2D.tolist() == [4] and p.eXDim_elem2D.tolist() == [0]
    np.testing.assert_array_equal(p.myList_nod2D[0], np.arange(5))
    np.testing.assert_array_equal(p.myList_elem2D[0], np.arange(4))
    np.testing.assert_array_equal(p.myList_edge2D[0], np.arange(3))
    np.testing.assert_array_equal(p.part, np.array([0, 5]))
    # empty communicators: no neighbours, no halo, offsets are a single 0
    for cs in (p.com_nod2D[0], p.com_elem2D[0], p.com_elem2D_full[0]):
        assert cs.rPE.size == 0 and cs.sPE.size == 0
        assert cs.rlist.size == 0 and cs.slist.size == 0
        np.testing.assert_array_equal(cs.rptr, [0])
        np.testing.assert_array_equal(cs.sptr, [0])


# --------------------------------------------------------------------------
# pytree registration round-trips (Partition + nested ComStruct + tuples + meta)
# --------------------------------------------------------------------------
@core2_avail
def test_pytree_roundtrip(part2):
    leaves, treedef = jax.tree_util.tree_flatten(part2)
    assert all(isinstance(x, np.ndarray) for x in leaves)   # numpy leaves, host
    p2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert (p2.npes, p2.nod2D, p2.elem2D, p2.edge2D) == \
           (part2.npes, part2.nod2D, part2.elem2D, part2.edge2D)
    np.testing.assert_array_equal(p2.myDim_nod2D, part2.myDim_nod2D)
    np.testing.assert_array_equal(p2.myList_nod2D[1], part2.myList_nod2D[1])
    np.testing.assert_array_equal(p2.com_elem2D_full[0].rlist,
                                  part2.com_elem2D_full[0].rlist)


# --------------------------------------------------------------------------
# Format fixture — a DIFFERENT mesh / npes (not over-fit to CORE2 dist_2)
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not (NARES4_DIR / "dist_768" / "com_info00000.out").exists(),
    reason=f"nares4/dist_768 format fixture missing under {NARES4_DIR}",
)
def test_nares4_format_fixture():
    """Read rank 0 of nares4/dist_768 via the block readers — confirms the
    reader handles a different mesh + a large npes (768) format."""
    dist = NARES4_DIR / "dist_768"
    ml = partit._read_my_list(dist, 0)
    assert ml["myDim_nod2D"] > 0
    # 0-based gids in range; max gid bounded by nodes-per-rank × npes
    assert ml["myList_nod2D"].min() >= 0
    c_nod, c_elem, c_elem_full = partit._read_com_info(
        dist, 0, eDim_nod=ml["eDim_nod2D"],
        eDim_elem=ml["eDim_elem2D"], eXDim_elem=ml["eXDim_elem2D"])
    _check_com(c_nod, rlist_size=ml["eDim_nod2D"],
               myDim=ml["myDim_nod2D"], eDim=ml["eDim_nod2D"], npes=768, rank=0)
    _check_com(c_elem_full,
               rlist_size=ml["eDim_elem2D"] + ml["eXDim_elem2D"],
               myDim=ml["myDim_elem2D"],
               eDim=ml["eDim_elem2D"] + ml["eXDim_elem2D"], npes=768, rank=0)


# --------------------------------------------------------------------------
# Error paths
# --------------------------------------------------------------------------
def test_missing_dist_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="dist_99"):
        partit.read_partition(tmp_path, 99)


def test_npes_eq_1_directs_to_synth(tmp_path):
    with pytest.raises(ValueError, match="synth_serial"):
        partit.read_partition(tmp_path, 1)


def test_npes_mismatch_raises(tmp_path):
    """rpart.out whose header npes disagrees with the requested npes raises."""
    d = tmp_path / "dist_2"
    d.mkdir()
    (d / "rpart.out").write_text("3\n10 20 30\n")     # says 3, dir is dist_2
    with pytest.raises(ValueError, match="npes=3 does not match"):
        partit.read_partition(tmp_path, 2)


def test_missing_rpart_in_existing_dir_raises(tmp_path):
    (tmp_path / "dist_2").mkdir()                       # dir exists, rpart absent
    with pytest.raises(FileNotFoundError, match="rpart.out"):
        partit.read_partition(tmp_path, 2)
