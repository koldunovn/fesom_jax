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

ATOL = 1e-12  # observed ~1e-14 (map class); C dump is %.17g full-precision


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
    cT, cS = pre[:, 1], pre[:, 2]
    assert np.max(np.abs(res.T_pre_surf - cT)) < ATOL
    assert np.max(np.abs(res.S_pre_surf - cS)) < ATOL
    # land/unfilled surface nodes are DUMMY in both, identically.
    assert np.array_equal(res.T_pre_surf > 0.99e10, cT > 0.99e10)


def test_postload_surface_matches_c(phc):
    """Full pipeline incl. the sequential GS extrapolation — the order-dependent #1 risk."""
    res, post = phc["res"], phc["post"]
    assert np.max(np.abs(res.T[:, 0] - post[:, 1])) < ATOL
    assert np.max(np.abs(res.S[:, 0] - post[:, 2])) < ATOL


def test_ic_field_physical(phc):
    """Full-column sanity (the surface dump can't verify depth)."""
    res, mesh = phc["res"], phc["mesh"]
    ml = np.asarray(mesh.node_layer_mask)
    T, S = res.T, res.S
    assert np.isfinite(T).all() and np.isfinite(S).all()
    assert -3.0 <= T[ml].min() and T[ml].max() <= 40.0
    assert 0.0 <= S[ml].min() and S[ml].max() <= 45.0
    assert (T[~ml] == 0.0).all() and (S[~ml] == 0.0).all()   # below-bottom zeroed
