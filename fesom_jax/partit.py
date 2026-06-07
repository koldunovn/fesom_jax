"""Domain-partition reader for the FESOM2 → JAX port (Phase 8, Task S.1).

Numpy port of the C MPI port's ASCII ``dist_<NP>`` partition reader
(``port2/fesom2_port/src/fesom_partit.c:68-208`` — ``read_rpart`` /
``read_my_list`` / ``read_com_info``). It loads the **bit-identical** FESOM
domain decomposition so an N-rank JAX run can be diffed per-substep against the
C N-rank dump (Phase-8 design, Locked decision 2). Nothing here is generated:
the canonical CORE2 partitions ship under
``/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/dist_{2,4,8,…}``.

The C reads ONE rank's files (the running ``mype``); the single-process
``shard_map`` model (Locked decision 3) instead needs **every** rank's data in
one process, so :func:`read_partition` reads ``rank = 0 .. npes-1`` and returns
a :class:`Partition` whose per-rank fields are stacked over a leading device
axis (rectangular counts) or held as ``npes``-long tuples (the ragged id lists
and communicators).

Index conventions (verified against ``fesom_halo.c``)
-----------------------------------------------------
``dist_<NP>`` is 1-based Fortran on disk. We shift to the 0-based convention the
rest of the JAX port uses, **but only the fields that are array indices or ids**:

* ``myList_*`` (global entity ids), ``rlist``/``slist`` (LOCAL field indices),
  ``rptr``/``sptr`` (cumulative offsets) → **subtract 1**.
* ``rPE``/``sPE`` are MPI **rank** ids, already 0-based on disk (the C feeds them
  straight to ``MPI_Isend``/``MPI_Irecv``, ``fesom_halo.c:166,182``) → **no shift**.

After the shift, for a com block: ``rlist[rptr[k]:rptr[k+1]]`` are the LOCAL halo
lanes (in ``[myDim, myDim+eDim)``) this rank **receives** from neighbour
``rPE[k]``; ``slist[sptr[k]:sptr[k+1]]`` are the LOCAL interior lanes (in
``[0, myDim)``) this rank **sends** to neighbour ``sPE[k]`` (``fesom_partit.h:42-52``).

The exchange is **broadcast-only** (owner value → halo copies, no additive
accumulate): a rank computes redundantly over its halo and the post-kernel
broadcast refreshes the halo copies. There is no additive exchange anywhere in
the C (``fesom_halo.c`` has only ``fesom_halo_exchange``).

Ownership asymmetry (load-bearing — verified on CORE2 ``dist_2``)
----------------------------------------------------------------
Only **nodes** are uniquely partitioned: ``Σ_d myDim_nod2D == nod2D`` and the
interior node lists are disjoint. **Elements and edges are NOT** — a boundary
element whose three vertices span two ranks is listed in the interior
(``myDim``) of *both* ranks (so it is computed redundantly), hence
``Σ_d myDim_elem2D > elem2D`` (the surplus = #boundary elements; 562 on
``dist_2``). Consequence for S.5: a reduction over elements/edges must pick a
unique owner per shared entity (do NOT naively sum over ``myDim`` — that
double-counts the boundary); node reductions are safe. The global counts here
are therefore taken as ``max(gid)+1`` (the id space is dense ``[0, count)``),
not ``Σ myDim``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from jax import tree_util


def _meta(static: bool = False) -> dataclasses.Field:
    """A dataclass field flagged as static pytree metadata when ``static``
    (mirrors :func:`fesom_jax.mesh._meta`)."""
    return dataclasses.field(metadata={"static": static})


# --------------------------------------------------------------------------
# Communicator block (one rank × one entity kind)
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ComStruct:
    """One rank's communicator for one entity kind — numpy mirror of the C
    ``fesom_com_struct`` (``fesom_partit.h:42-52``), all indices 0-based.

    Receive side (owner → halo): for neighbour rank ``rPE[k]`` this rank fills
    its LOCAL halo lanes ``rlist[rptr[k]:rptr[k+1]]``. Send side: for neighbour
    ``sPE[k]`` it ships its LOCAL interior lanes ``slist[sptr[k]:sptr[k+1]]``.

    Empty (``rPEnum==sPEnum==0``) for a single-rank synthesised partit.
    """

    rPE: np.ndarray    # (rPEnum,)   neighbour ranks we RECEIVE from (0-based)
    rptr: np.ndarray   # (rPEnum+1,) 0-based cumulative offsets into rlist
    rlist: np.ndarray  # (eDim,)     0-based LOCAL halo lanes we receive into
    sPE: np.ndarray    # (sPEnum,)   neighbour ranks we SEND to (0-based)
    sptr: np.ndarray   # (sPEnum+1,) 0-based cumulative offsets into slist
    slist: np.ndarray  # (sptr[-1],) 0-based LOCAL interior lanes we send


tree_util.register_dataclass(
    ComStruct,
    data_fields=["rPE", "rptr", "rlist", "sPE", "sptr", "slist"],
    meta_fields=[],
)


# --------------------------------------------------------------------------
# Partition (all ranks)
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Partition:
    """All ranks' partition + communicator state for an ``npes``-way mesh
    decomposition — numpy mirror of the C ``fesom_partit`` (``fesom_partit.h``),
    read for EVERY rank in one process (the single-process ``shard_map`` model).

    Registered as a JAX pytree (house convention): the rectangular per-rank count
    arrays, the ragged ``npes``-long id-list tuples, the per-kind communicator
    tuples and ``part`` are data leaves; the static global counts are metadata.
    It is **host metadata** consumed by ``shard_mesh`` (S.2) to build the padded
    per-device arrays — it is not itself sharded onto devices.
    """

    # --- per-rank interior/halo counts, each (npes,) ---
    myDim_nod2D: np.ndarray
    eDim_nod2D: np.ndarray
    myDim_elem2D: np.ndarray
    eDim_elem2D: np.ndarray
    eXDim_elem2D: np.ndarray         # extended halo (elements only)
    myDim_edge2D: np.ndarray
    eDim_edge2D: np.ndarray

    # --- per-rank global-id lists (0-based), ragged ⇒ npes-long tuples ---
    myList_nod2D: tuple              # each (myDim+eDim,)         node gids
    myList_elem2D: tuple             # each (myDim+eDim+eXDim,)   elem gids
    myList_edge2D: tuple             # each (myDim+eDim,)         edge gids

    # --- per-rank communicators (npes-long tuples of ComStruct) ---
    com_nod2D: tuple                 # nod2D halo
    com_elem2D: tuple                # elem2D small halo
    com_elem2D_full: tuple           # elem2D extended halo

    # --- rpart prefix vector (mirror C part[]; here 0-based) ---
    # part[k] = #nodes owned by ranks [0..k); part[-1] == nod2D. Vestigial in the
    # C loop (never indexed downstream) — kept for completeness + the cross-check
    # part[k+1]-part[k] == myDim_nod2D[k]. Global ids are NOT contiguous per rank
    # (myList_* is scattered), so part does NOT map id→owner.
    part: np.ndarray                 # (npes+1,)

    # --- static global counts (metadata) ---
    npes: int = _meta(static=True)
    nod2D: int = _meta(static=True)
    elem2D: int = _meta(static=True)
    edge2D: int = _meta(static=True)


_PART_META = [f.name for f in dataclasses.fields(Partition) if f.metadata.get("static")]
_PART_DATA = [f.name for f in dataclasses.fields(Partition) if not f.metadata.get("static")]
tree_util.register_dataclass(Partition, data_fields=_PART_DATA, meta_fields=_PART_META)


# --------------------------------------------------------------------------
# ASCII token stream — mirror of the C read_int / read_int_array
# --------------------------------------------------------------------------
class _IntStream:
    """Sequential whitespace-separated integer reader over an ASCII file —
    equivalent to the C ``fscanf(f, " %d")`` (``fesom_partit.c:29-41``), which
    treats spaces and newlines identically (Fortran free-format). The whole file
    is tokenised once; :meth:`read_array` slices the token list so even the big
    ``myList`` blocks parse in one ``np.array`` call.
    """

    def __init__(self, path: Path):
        self._path = path
        with open(path) as f:
            self._tok = f.read().split()
        self._i = 0

    def read_int(self) -> int:
        if self._i >= len(self._tok):
            raise ValueError(f"{self._path}: expected integer at token {self._i} (EOF)")
        v = int(self._tok[self._i])
        self._i += 1
        return v

    def read_array(self, n: int) -> np.ndarray:
        """Read ``n`` integers as an ``int32`` array (all FESOM ids/indices fit
        int32: gids ≤ ~3.7e5)."""
        j = self._i + n
        if j > len(self._tok):
            raise ValueError(
                f"{self._path}: expected {n} integers at token {self._i}, "
                f"only {len(self._tok) - self._i} remain"
            )
        out = np.array(self._tok[self._i:j], dtype=np.int32)
        self._i = j
        return out


def _rank_file(dist_dir: Path, base: str, rank: int) -> Path:
    """``dist_dir/<base><rank:05d>.out`` (mirror ``build_rank_file``,
    ``fesom_partit.c:53-58``)."""
    return dist_dir / f"{base}{rank:05d}.out"


# --------------------------------------------------------------------------
# Block readers (numpy ports of the C statics)
# --------------------------------------------------------------------------
def _read_rpart(dist_dir: Path, npes: int) -> np.ndarray:
    """Port of ``read_rpart`` (``fesom_partit.c:68-93``): validate the file's
    ``npes`` then read the ``npes`` per-rank node counts and return the 0-based
    prefix-sum ``part`` ``(npes+1,)``. (The rest of ``rpart.out`` — the per-node
    owner vector — is unused by the C and skipped here, as in the C.)"""
    path = dist_dir / "rpart.out"
    if not path.exists():
        raise FileNotFoundError(f"fesom_partit: cannot open {path}")
    s = _IntStream(path)
    file_npes = s.read_int()
    if file_npes != npes:
        raise ValueError(
            f"fesom_partit: rpart.out npes={file_npes} does not match "
            f"requested npes={npes} ({path})"
        )
    counts = s.read_array(npes)
    part = np.empty(npes + 1, dtype=np.int32)
    part[0] = 0
    part[1:] = np.cumsum(counts)
    return part


def _read_my_list(dist_dir: Path, rank: int) -> dict:
    """Port of ``read_my_list`` (``fesom_partit.c:111-145``): per-rank counts +
    global-id lists. Global ids are shifted 1-based → 0-based."""
    path = _rank_file(dist_dir, "my_list", rank)
    if not path.exists():
        raise FileNotFoundError(f"fesom_partit: cannot open {path}")
    s = _IntStream(path)
    hdr = s.read_int()
    if hdr != rank:
        raise ValueError(f"fesom_partit: {path} says rank={hdr}, expected {rank}")

    myDim_n = s.read_int()
    eDim_n = s.read_int()
    myList_n = s.read_array(myDim_n + eDim_n) - 1            # gid 1-based → 0-based

    myDim_e = s.read_int()
    eDim_e = s.read_int()
    eXDim_e = s.read_int()
    myList_e = s.read_array(myDim_e + eDim_e + eXDim_e) - 1

    myDim_ed = s.read_int()
    eDim_ed = s.read_int()
    myList_ed = s.read_array(myDim_ed + eDim_ed) - 1

    return dict(
        myDim_nod2D=myDim_n, eDim_nod2D=eDim_n, myList_nod2D=myList_n,
        myDim_elem2D=myDim_e, eDim_elem2D=eDim_e, eXDim_elem2D=eXDim_e,
        myList_elem2D=myList_e,
        myDim_edge2D=myDim_ed, eDim_edge2D=eDim_ed, myList_edge2D=myList_ed,
    )


def _read_com_block(s: _IntStream, rlist_size: int) -> ComStruct:
    """Port of ``read_com_block`` (``fesom_partit.c:154-181``). ``rlist_size`` is
    supplied by the caller (the file does not encode it): ``eDim`` for nod2D /
    elem2D, ``eDim+eXDim`` for elem2D_full. ``rPE``/``sPE`` kept 0-based;
    ``rptr``/``sptr``/``rlist``/``slist`` shifted to 0-based.

    Note: the C caps ``rPEnum``/``sPEnum`` at ``FESOM_MAX_NEIGHBOR_PARTITIONS=32``
    (static C arrays). The numpy reader sizes dynamically, so it is strictly more
    general — no cap needed, same result.
    """
    rPEnum = s.read_int()
    rPE = s.read_array(rPEnum)                  # 0-based MPI ranks — NO shift
    rptr = s.read_array(rPEnum + 1) - 1         # 1-based offsets → 0-based
    rlist = s.read_array(rlist_size) - 1        # 1-based local → 0-based

    sPEnum = s.read_int()
    sPE = s.read_array(sPEnum)                   # NO shift
    sptr = s.read_array(sPEnum + 1) - 1         # 0-based offsets
    slist_size = int(sptr[-1] - sptr[0])         # == C sptr[sPEnum]-sptr[0]
    slist = s.read_array(slist_size) - 1        # 1-based local → 0-based

    return ComStruct(rPE=rPE, rptr=rptr, rlist=rlist, sPE=sPE, sptr=sptr, slist=slist)


def _read_com_info(dist_dir: Path, rank: int,
                   eDim_nod: int, eDim_elem: int, eXDim_elem: int
                   ) -> tuple[ComStruct, ComStruct, ComStruct]:
    """Port of ``read_com_info`` (``fesom_partit.c:190-208``): rank header then 3
    com blocks (nod2D, elem2D, elem2D_full). The block ``rlist`` sizes follow the
    partition convention (``fesom_partit.c:203-205``)."""
    path = _rank_file(dist_dir, "com_info", rank)
    if not path.exists():
        raise FileNotFoundError(f"fesom_partit: cannot open {path}")
    s = _IntStream(path)
    hdr = s.read_int()
    if hdr != rank:
        raise ValueError(f"fesom_partit: {path} says rank={hdr}, expected {rank}")
    com_nod = _read_com_block(s, eDim_nod)
    com_elem = _read_com_block(s, eDim_elem)
    com_elem_full = _read_com_block(s, eDim_elem + eXDim_elem)
    return com_nod, com_elem, com_elem_full


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def read_partition(mesh_dir: str | Path, npes: int) -> Partition:
    """Read the ``mesh_dir/dist_<npes>/`` decomposition for ALL ``npes`` ranks.

    Mirror of ``fesom_partit_init`` (``fesom_partit.c:213-278``) but reads every
    rank in one process. For ``npes == 1`` there are no ``dist_`` files — use
    :func:`synth_serial` (the C synthesises the serial partit the same way,
    ``fesom_partit.c:280-308``).

    Raises ``FileNotFoundError`` if ``dist_<npes>/`` (or a rank file) is missing,
    ``ValueError`` on an ``npes`` mismatch or a wrong per-file rank header.
    """
    if npes < 1:
        raise ValueError(f"read_partition: npes must be >= 1, got {npes}")
    if npes == 1:
        raise ValueError(
            "read_partition: npes==1 has no dist_ files; build the identity "
            "partit with synth_serial(nod2D, elem2D, edge2D)"
        )

    mesh_dir = Path(mesh_dir)
    dist_dir = mesh_dir / f"dist_{npes}"
    if not dist_dir.is_dir():
        raise FileNotFoundError(
            f"fesom_partit: {dist_dir} missing — no partition for npes={npes} "
            f"under {mesh_dir}"
        )

    part = _read_rpart(dist_dir, npes)

    # accumulators (lists indexed by rank → stacked/tupled at the end)
    cnt: dict[str, list[int]] = {
        k: [] for k in (
            "myDim_nod2D", "eDim_nod2D", "myDim_elem2D", "eDim_elem2D",
            "eXDim_elem2D", "myDim_edge2D", "eDim_edge2D",
        )
    }
    myList = {"nod2D": [], "elem2D": [], "edge2D": []}
    com = {"nod2D": [], "elem2D": [], "elem2D_full": []}

    for rank in range(npes):
        ml = _read_my_list(dist_dir, rank)
        for k in cnt:
            cnt[k].append(int(ml[k]))
        myList["nod2D"].append(ml["myList_nod2D"])
        myList["elem2D"].append(ml["myList_elem2D"])
        myList["edge2D"].append(ml["myList_edge2D"])

        c_nod, c_elem, c_elem_full = _read_com_info(
            dist_dir, rank,
            eDim_nod=ml["eDim_nod2D"],
            eDim_elem=ml["eDim_elem2D"],
            eXDim_elem=ml["eXDim_elem2D"],
        )
        com["nod2D"].append(c_nod)
        com["elem2D"].append(c_elem)
        com["elem2D_full"].append(c_elem_full)

    myDim_nod2D = np.array(cnt["myDim_nod2D"], dtype=np.int32)
    myDim_elem2D = np.array(cnt["myDim_elem2D"], dtype=np.int32)
    myDim_edge2D = np.array(cnt["myDim_edge2D"], dtype=np.int32)

    # Global counts via max(gid)+1. The id space is dense [0, count). ⚠️ ONLY
    # nodes are uniquely partitioned (Σ myDim_nod2D == nod2D); ELEMENTS and EDGES
    # along a partition boundary are REDUNDANTLY owned — a boundary element whose
    # vertices span two ranks sits in BOTH ranks' interior myDim list (the
    # "redundant compute over the halo" model). So Σ myDim_elem2D > elem2D (the
    # overlap = #boundary entities). This is load-bearing for S.5: an element/edge
    # reduction must NOT naively sum over myDim (it would double-count the
    # boundary); only node reductions can. See docs/PORTING_LESSONS.md.
    def _global_count(mylists) -> int:
        return int(max(int(ml.max()) for ml in mylists)) + 1

    nod2D = _global_count(myList["nod2D"])
    elem2D = _global_count(myList["elem2D"])
    edge2D = _global_count(myList["edge2D"])

    return Partition(
        myDim_nod2D=myDim_nod2D,
        eDim_nod2D=np.array(cnt["eDim_nod2D"], dtype=np.int32),
        myDim_elem2D=myDim_elem2D,
        eDim_elem2D=np.array(cnt["eDim_elem2D"], dtype=np.int32),
        eXDim_elem2D=np.array(cnt["eXDim_elem2D"], dtype=np.int32),
        myDim_edge2D=myDim_edge2D,
        eDim_edge2D=np.array(cnt["eDim_edge2D"], dtype=np.int32),
        myList_nod2D=tuple(myList["nod2D"]),
        myList_elem2D=tuple(myList["elem2D"]),
        myList_edge2D=tuple(myList["edge2D"]),
        com_nod2D=tuple(com["nod2D"]),
        com_elem2D=tuple(com["elem2D"]),
        com_elem2D_full=tuple(com["elem2D_full"]),
        part=part,
        npes=npes,
        nod2D=nod2D,
        elem2D=elem2D,
        edge2D=edge2D,
    )


def synth_serial(nod2D: int, elem2D: int, edge2D: int) -> Partition:
    """The ``npes==1`` identity partition — numpy mirror of
    ``fesom_partit_set_global_counts_serial`` (``fesom_partit.c:280-308``): one
    rank owns everything, zero halo (``eDim==eXDim==0``), no neighbours, and
    ``myList_*`` is the identity ``arange``. Makes the sharded code path reduce
    to the dense single-device model (S.2's no-op invariant) so the ``npes==1``
    suite stays byte-identical to ``v1.0``.
    """
    def empty_com() -> ComStruct:
        z = np.zeros(0, dtype=np.int32)
        z1 = np.zeros(1, dtype=np.int32)   # rPEnum+1 == 1 offset entry for 0 PEs
        return ComStruct(rPE=z, rptr=z1, rlist=z, sPE=z, sptr=z1.copy(), slist=z)

    return Partition(
        myDim_nod2D=np.array([nod2D], dtype=np.int32),
        eDim_nod2D=np.array([0], dtype=np.int32),
        myDim_elem2D=np.array([elem2D], dtype=np.int32),
        eDim_elem2D=np.array([0], dtype=np.int32),
        eXDim_elem2D=np.array([0], dtype=np.int32),
        myDim_edge2D=np.array([edge2D], dtype=np.int32),
        eDim_edge2D=np.array([0], dtype=np.int32),
        myList_nod2D=(np.arange(nod2D, dtype=np.int32),),
        myList_elem2D=(np.arange(elem2D, dtype=np.int32),),
        myList_edge2D=(np.arange(edge2D, dtype=np.int32),),
        com_nod2D=(empty_com(),),
        com_elem2D=(empty_com(),),
        com_elem2D_full=(empty_com(),),
        part=np.array([0, nod2D], dtype=np.int32),
        npes=1,
        nod2D=nod2D,
        elem2D=elem2D,
        edge2D=edge2D,
    )
