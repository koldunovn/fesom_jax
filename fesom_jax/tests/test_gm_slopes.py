"""Task G.2 gate — GM neutral slopes (compute_sigma_xy + compute_neutral_slope).

Feeds the C GM-ON dump's step-1 inputs to the JAX kernels and matches the GM
intermediates all-node on the CORE2 mesh:

* ``compute_sigma_xy`` — fed T/S/sw_alpha/sw_beta → match ``sigma_xy`` (an
  element→node area-weighted ∇T/∇S scatter → scatter class ~1e-12).
* ``compute_neutral_slope`` — fed the C's ``sigma_xy`` + ``bvfreq`` (isolating it
  from the scatter) → match ``neutral_slope``/``slope_tapered``/``fer_tapfac``
  (per-node map + tanh taper → map class ~1e-14).

Plus a chained path (T/S/bvfreq → both kernels) and AD finiteness (masked-NaN).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import eos, gm
from fesom_jax.io_dump import load_gm_dump
from fesom_jax.mesh import load_mesh

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
    return mesh, fields, cfg


def test_sigma_xy_matches_c_dump(setup):
    """compute_sigma_xy fed the C's T/S/sw_alpha/sw_beta → match ``sigma_xy``
    (element→node area-weighted gradient scatter → scatter class)."""
    mesh, f, cfg = setup
    T = jnp.asarray(f["T"]); S = jnp.asarray(f["S"])
    a = jnp.asarray(f["sw_alpha"]); b = jnp.asarray(f["sw_beta"])
    sig = np.asarray(gm.compute_sigma_xy(mesh, T, S, a, b, cfg))   # (N,nl,2)
    sig_c = f["sigma_xy"]                                          # (N,nl,2)
    m = np.asarray(mesh.node_layer_mask)[..., None]
    d = np.max(np.abs(sig[np.broadcast_to(m, sig.shape)]
                      - sig_c[np.broadcast_to(m, sig_c.shape)]))
    scale = np.max(np.abs(sig_c[np.broadcast_to(m, sig_c.shape)]))
    print(f"\nsigma_xy max|Δ|={d:.3e}  (scale {scale:.3e}, rel {d/scale:.2e})")
    assert d < 1e-9                                               # scatter/reduction class


def _max_isclose_violation(got, ref, mask, rtol, atol):
    """max over masked lanes of (|Δ| − (atol + rtol·|ref|)); ≤0 ⇒ all within tol.
    Returns also the raw max|Δ| for reporting."""
    mm = np.broadcast_to(mask[..., None], got.shape) if got.ndim == 3 else mask
    d = np.abs(got[mm] - ref[mm])
    tol = atol + rtol * np.abs(ref[mm])
    return float(np.max(d - tol)), float(np.max(d))


def test_neutral_slope_matches_c_dump(setup):
    """compute_neutral_slope fed the C's sigma_xy + bvfreq (isolated from the
    scatter). ``neutral_slope`` is UNTAPERED → huge dynamic range (slopes reach
    ~1e5-1e6 where N²→the eps² floor), so gate it RELATIVE (map class). It is
    eager-bit-exact vs the C; XLA FMA-contraction of |s|=√(sx²+sy²) shifts it
    ~1e-15 *relative* when fused (the density-lesson effect). ``slope_tapered``
    (the field actually consumed downstream) is bounded (taper→0 at huge slope)."""
    mesh, f, cfg = setup
    ns, st, tf = gm.compute_neutral_slope(
        mesh, jnp.asarray(f["sigma_xy"]), jnp.asarray(f["bvfreq"]), cfg)
    ns, st, tf = np.asarray(ns), np.asarray(st), np.asarray(tf)
    nl = mesh.nl
    smask = np.asarray(mesh.node_layer_mask)[:, : nl - 1]

    # neutral_slope: map-class RELATIVE (rtol) — abs |Δ| is meaningless at 1e5-1e6.
    v, d = _max_isclose_violation(ns, f["neutral_slope"], smask, rtol=1e-12, atol=1e-13)
    print(f"\nneutral_slope max|Δ|={d:.3e}  isclose-violation={v:.3e} (rtol 1e-12)")
    assert v <= 0.0
    # slope_tapered = ns·√c1: where the UNTAPERED slope is huge (~1e5) the taper→0,
    # so it's a huge×tiny product ≈ 0 carrying the huge factor's FMA noise (~4e-10
    # abs). Physically that lane IS ~zero slope (negligible Redi flux); gate isclose
    # with a near-zero absolute floor (eager is bit-exact; this is the FMA floor).
    v, d = _max_isclose_violation(st, f["slope_tapered"], smask, rtol=1e-12, atol=1e-9)
    print(f"slope_tapered max|Δ|={d:.3e}  isclose-violation={v:.3e} (atol 1e-9)")
    assert v <= 0.0
    # fer_tapfac (N,nl) ∈ [0,1]: tight absolute.
    mtf = np.asarray(mesh.node_layer_mask)
    dtf = np.max(np.abs(tf[mtf] - f["fer_tapfac"][mtf]))
    print(f"fer_tapfac max|Δ|={dtf:.3e}")
    assert dtf < 1e-14


def test_chained_slope_tapered_matches_c_dump(setup):
    """The realistic path T/S/bvfreq → compute_sigma_xy → compute_neutral_slope →
    ``slope_tapered`` matches the C (inherits the sigma_xy scatter floor)."""
    mesh, f, cfg = setup
    T = jnp.asarray(f["T"]); S = jnp.asarray(f["S"])
    a = jnp.asarray(f["sw_alpha"]); b = jnp.asarray(f["sw_beta"])
    bvfreq = jnp.asarray(f["bvfreq"])
    sig = gm.compute_sigma_xy(mesh, T, S, a, b, cfg)
    _, st, _ = gm.compute_neutral_slope(mesh, sig, bvfreq, cfg)
    st = np.asarray(st)
    smask = np.asarray(mesh.node_layer_mask)[:, : mesh.nl - 1]
    mm = np.broadcast_to(smask[..., None], st.shape)
    d = np.max(np.abs(st[mm] - f["slope_tapered"][mm]))
    print(f"\nchained slope_tapered max|Δ|={d:.3e}")
    assert d < 1e-9                                              # scatter floor


def test_ad_finite(setup):
    """d(Σslope_tapered)/d(T) finite everywhere incl. masked/weak-strat lanes
    (the safe-sqrt on |s| + √c1, the bv≤0 where-mask, the denom clamp)."""
    mesh, f, cfg = setup
    T0 = jnp.asarray(f["T"]); S = jnp.asarray(f["S"])
    a = jnp.asarray(f["sw_alpha"]); b = jnp.asarray(f["sw_beta"])
    bvfreq = jnp.asarray(f["bvfreq"])

    def loss(T):
        sig = gm.compute_sigma_xy(mesh, T, S, a, b, cfg)
        _, st, _ = gm.compute_neutral_slope(mesh, sig, bvfreq, cfg)
        return jnp.sum(st)

    gT = np.asarray(jax.grad(loss)(T0))
    assert np.all(np.isfinite(gT))
    assert np.any(gT != 0.0)
