"""Task 5.2 gate: the numpy PHC reader (``phc_ic.load_phc_ic``) reproduces the C
``fesom_phc.c`` initial condition on CORE2, verified against the C surface dumps
(``phc_dump_preextrap_rank0.txt`` / ``phc_dump_postload_rank0.txt``).

Three stages (cf. fesom_phc.c):
  1. bracket indices ``bilin_i``/``bilin_j`` — must be EXACT (integer search parity).
  2. pre-extrap surface T/S (bilinear) — map class (~1e-14); + dummy-mask parity.
  3. post-load surface T/S (full pipeline: sequential GS extrap + cleanup + ptheta) —
     map class (~1e-14). The sequential Gauss-Seidel order-dependence is the #1 risk.

SKIPS unless the CORE2 mesh export, the C PHC dump, and the PHC NetCDF all exist
(local artifacts). ⚠️ The C dump is **surface-only**; the vertical interp + deep
``ptheta`` are exercised indirectly (physical-range check here; the per-substep density
gate in Task 5.7). A full-column C dump can be added if 5.7 shows a depth mismatch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CORE2_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
PHC_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "phc_dump_core2"
PHC_NC = Path("/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc")

pytestmark = pytest.mark.skipif(
    not (CORE2_MESH_DIR.is_dir() and PHC_DUMP_DIR.is_dir() and PHC_NC.is_file()),
    reason="needs CORE2 mesh export + C PHC dump + PHC NetCDF (Task 5.2 artifacts)",
)

# Bit-exact since the bilinear association-order fix (2026-06-12): the C groups
# ((v·wx)·wy) (fesom_phc.c:215-218); grouping wx·wy first cost ~1 ulp at ~27k nodes.
# The %.17g dump round-trips doubles exactly, so equality is the right gate.


@pytest.fixture(scope="module")
def phc():
    from fesom_jax import mesh as meshmod, phc_ic
    m = meshmod.load_mesh(CORE2_MESH_DIR)
    res = phc_ic.load_phc_ic(m, str(PHC_NC))
    pre = np.loadtxt(PHC_DUMP_DIR / "phc_dump_preextrap_rank0.txt")
    post = np.loadtxt(PHC_DUMP_DIR / "phc_dump_postload_rank0.txt")
    pre = pre[np.argsort(pre[:, 0])]      # order by 1-based gid → node index = gid-1
    post = post[np.argsort(post[:, 0])]
    return dict(mesh=m, res=res, pre=pre, post=post)


def test_dump_is_full_mesh_in_order(phc):
    gid = phc["pre"][:, 0].astype(np.int64)
    assert np.array_equal(gid, np.arange(1, phc["mesh"].nod2D + 1))


def test_bracket_indices_exact(phc):
    res, pre = phc["res"], phc["pre"]
    assert np.array_equal(res.bilin_i, pre[:, 3].astype(np.int64))
    assert np.array_equal(res.bilin_j, pre[:, 4].astype(np.int64))


def test_preextrap_surface_and_dummy_mask(phc):
    res, pre = phc["res"], phc["pre"]
    assert np.array_equal(res.T_pre_surf, pre[:, 1])
    assert np.array_equal(res.S_pre_surf, pre[:, 2])


def test_postload_surface_matches_c(phc):
    """Full pipeline incl. the sequential GS extrapolation — the order-dependent #1 risk."""
    res, post = phc["res"], phc["post"]
    assert np.array_equal(res.T[:, 0], post[:, 1])
    assert np.array_equal(res.S[:, 0], post[:, 2])


def test_ic_field_physical(phc):
    """Full-column sanity (the surface dump can't verify depth)."""
    res, mesh = phc["res"], phc["mesh"]
    ml = np.asarray(mesh.node_layer_mask)
    T, S = res.T, res.S
    assert np.isfinite(T).all() and np.isfinite(S).all()
    assert -3.0 <= T[ml].min() and T[ml].max() <= 40.0
    assert 0.0 <= S[ml].min() and S[ml].max() <= 45.0
    assert (T[~ml] == 0.0).all() and (S[~ml] == 0.0).all()   # below-bottom zeroed


# ==========================================================================
# dist_16 partition-faithful extrapolation (the z2_cdump oracle partition)
# ==========================================================================
# The GS land fill is order-dependent ⇒ the C IC is PARTITION-DEPENDENT (a 1-rank and a
# 16-rank C run differ by up to 25.8 PSU at fill nodes — Baltic/Kara). The z2_cdump
# oracle ran on dist_16, so the cached CORE2 IC must replicate the 16-rank fill order.
# Oracle: the C 16-rank postload surface dump (mevp/cdump_16r, same partition as
# z2_cdump — gid lists verified identical). Rank node lists (C local order =
# myList_nod2D) come from the per-rank dump gid columns themselves.
DIST16_DIR = PHC_DUMP_DIR / "dist16"
IC_DIST16_DIR = Path(__file__).resolve().parents[2] / "data" / "ic_core2_dist16"
IC_SERIAL_DIR = Path(__file__).resolve().parents[2] / "data" / "ic_core2"

dist16_missing = pytest.mark.skipif(not DIST16_DIR.is_dir(),
                                    reason="needs the C dist_16 postload dumps")


@pytest.fixture(scope="module")
def phc16(phc):
    from fesom_jax import phc_ic
    files = sorted(DIST16_DIR.glob("phc_dump_postload_rank*.txt"),
                   key=lambda p: int(p.stem.rsplit("rank", 1)[1]))
    assert len(files) == 16
    dumps = [np.loadtxt(f) for f in files]
    rank_nodes = [d[:, 0].astype(np.int64) - 1 for d in dumps]   # C local order
    res = phc_ic.load_phc_ic(phc["mesh"], str(PHC_NC), rank_nodes=rank_nodes)
    return dict(res=res, rank_nodes=rank_nodes, dumps=dumps)


@dist16_missing
def test_dist16_partition_covers_mesh(phc, phc16):
    g = np.concatenate(phc16["rank_nodes"])
    assert g.size == phc["mesh"].nod2D and np.unique(g).size == g.size


@dist16_missing
def test_dist16_postload_surface_bitexact(phc16):
    """The partition-faithful extrap reproduces the C 16-rank IC EXACTLY (incl. the 488
    Baltic/Kara brackish fill nodes that the serial order gets wrong by up to 25.8 PSU)."""
    res = phc16["res"]
    for d, nodes in zip(phc16["dumps"], phc16["rank_nodes"]):
        assert np.array_equal(res.T[nodes, 0], d[:, 1])
        assert np.array_equal(res.S[nodes, 0], d[:, 2])


@dist16_missing
def test_dist16_ic_cache_is_current(phc16):
    """``data/ic_core2_dist16`` (what every z2_cdump-gated zstar test loads) is the
    dist_16 build — full-column equality catches a stale or serial-order cache."""
    res = phc16["res"]
    assert np.array_equal(np.load(IC_DIST16_DIR / "T_ic.npy"), res.T)
    assert np.array_equal(np.load(IC_DIST16_DIR / "S_ic.npy"), res.S)


def test_serial_ic_cache_is_current(phc):
    """``data/ic_core2`` (what the legacy 1-rank-oracle CORE2 tests load) is the SERIAL
    build — the two caches differ at GS-filled nodes (partition-dependent C IC)."""
    res = phc["res"]
    assert np.array_equal(np.load(IC_SERIAL_DIR / "T_ic.npy"), res.T)
    assert np.array_equal(np.load(IC_SERIAL_DIR / "S_ic.npy"), res.S)
