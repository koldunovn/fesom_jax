"""Task G.3 gate — GM/Redi coefficient builder (init_redi_gm).

Feeds the C GM-ON dump's step-1 inputs (bvfreq, hnode_new, fer_tapfac) to
``gm.init_redi_gm`` with default ``Params`` (k_gm=redi_kmax=1000) and matches
``fer_K``/``Ki``/``fer_C``/``fer_scal`` all-node vs the C. Per-node maps (resolution
scaling + the depth-exp F2 + the Redi taper) → map class. Plus the ML-seam check
(d(fer_K)/d(k_gm) is the expected scaling) and AD finiteness.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import gm
from fesom_jax.io_dump import load_gm_dump
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params

CORE2_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
GM_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "gm_dump_core2"

pytestmark = pytest.mark.skipif(
    not (CORE2_MESH_DIR.is_dir() and (GM_DUMP_DIR / "gm_meta.txt").is_file()),
    reason=f"CORE2 mesh / GM dump missing ({GM_DUMP_DIR}); run jax_gm_dump_core2.sh",
)


@pytest.fixture(scope="module")
def setup():
    mesh = load_mesh(CORE2_MESH_DIR)
    fields, meta = load_gm_dump(GM_DUMP_DIR)
    cfg = gm.GMConfig()
    fer_K, Ki, fer_C, fer_scal = gm.init_redi_gm(
        mesh, jnp.asarray(fields["bvfreq"]), jnp.asarray(fields["hnode_new"]),
        jnp.asarray(fields["fer_tapfac"]), Params.defaults(), cfg)
    return mesh, fields, cfg, (np.asarray(fer_K), np.asarray(Ki),
                              np.asarray(fer_C), np.asarray(fer_scal))


def _isclose_viol(got, ref, mask, rtol, atol):
    d = np.abs(got[mask] - ref[mask])
    return float(np.max(d - (atol + rtol * np.abs(ref[mask])))), float(np.max(d))


def test_fer_K_matches_c_dump(setup):
    """fer_K = max(scaling·k_gm, K_GM_min)·zscaling on the iface range."""
    mesh, f, cfg, (fer_K, Ki, fer_C, fer_scal) = setup
    m = np.asarray(mesh.node_iface_mask)
    v, d = _isclose_viol(fer_K, f["fer_K"], m, rtol=1e-12, atol=1e-12)
    print(f"\nfer_K max|Δ|={d:.3e}  viol={v:.3e}  (scale {np.abs(f['fer_K'][m]).max():.1f})")
    assert v <= 0.0


def test_Ki_matches_c_dump(setup):
    """Ki = Ki_top·⟨zscaling⟩·√tapfac + Redi_Kmin·|√tapfac−1| on the layer range."""
    mesh, f, cfg, (fer_K, Ki, fer_C, fer_scal) = setup
    m = np.asarray(mesh.node_layer_mask)
    v, d = _isclose_viol(Ki, f["Ki"], m, rtol=1e-12, atol=1e-12)
    print(f"\nKi max|Δ|={d:.3e}  viol={v:.3e}  (scale {np.abs(f['Ki'][m]).max():.1f})")
    assert v <= 0.0


def test_fer_C_fer_scal_match_c_dump(setup):
    """fer_C = cm² (the wave-speed integral, conservative bounds); fer_scal = the
    resolution scaling. Per-node scalars over all nodes."""
    mesh, f, cfg, (fer_K, Ki, fer_C, fer_scal) = setup
    cref = f["fer_C"].reshape(-1)
    sref = f["fer_scal"].reshape(-1)
    dc = np.max(np.abs(fer_C - cref))
    ds = np.max(np.abs(fer_scal - sref))
    print(f"\nfer_C max|Δ|={dc:.3e} (scale {np.abs(cref).max():.3e});  "
          f"fer_scal max|Δ|={ds:.3e}")
    # cm_sum is a depth reduction (Σ over levels) → scatter/reduction class.
    assert dc < 1e-9 * max(np.abs(cref).max(), 1.0)
    assert ds < 1e-13


def test_ml_seam_k_gm(setup):
    """The 2nd ML-hook is live: d(Σfer_K)/d(k_gm) = Σ scaling·zscaling over the
    iface range where scaling·k_gm > K_GM_min (the max is not clamped)."""
    mesh, f, cfg, _ = setup
    bv = jnp.asarray(f["bvfreq"]); hn = jnp.asarray(f["hnode_new"])
    tf = jnp.asarray(f["fer_tapfac"])

    def loss(k_gm):
        p = Params(k_ver=jnp.asarray(1e-5), a_ver=jnp.asarray(1e-4),
                   k_gm=k_gm, redi_kmax=jnp.asarray(1000.0))
        fer_K, *_ = gm.init_redi_gm(mesh, bv, hn, tf, p, cfg)
        return jnp.sum(fer_K)

    g = float(jax.grad(loss)(jnp.asarray(1000.0)))
    print(f"\nd(Σfer_K)/d(k_gm) = {g:.6e}")
    assert np.isfinite(g) and g > 0.0      # fer_K increases with k_gm


def test_ad_finite(setup):
    """d(ΣKi)/d(bvfreq) + d(Σfer_C)/d(hnode_new) finite (safe-sqrt on bv, the
    taper √tapfac, the max clamps) — the masked-NaN baseline."""
    mesh, f, cfg, _ = setup
    bv0 = jnp.asarray(f["bvfreq"]); hn0 = jnp.asarray(f["hnode_new"])
    tf = jnp.asarray(f["fer_tapfac"])

    def lKi(bv):
        _, Ki, *_ = gm.init_redi_gm(mesh, bv, hn0, tf, Params.defaults(), cfg)
        return jnp.sum(Ki)

    def lC(hn):
        *_, fer_C, _ = gm.init_redi_gm(mesh, bv0, hn, tf, Params.defaults(), cfg)
        return jnp.sum(fer_C)

    gbv = np.asarray(jax.grad(lKi)(bv0))
    ghn = np.asarray(jax.grad(lC)(hn0))
    assert np.all(np.isfinite(gbv)) and np.all(np.isfinite(ghn))
