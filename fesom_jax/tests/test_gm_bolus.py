"""Task G.4 gate — GM streamfunction + bolus velocity (fer_solve_gamma +
fer_gamma2vel).

Feeds the C GM-ON dump's step-1 inputs to the JAX kernels and matches all-node:

* ``fer_solve_gamma`` — fed sigma_xy/bvfreq/fer_K/fer_C → match ``fer_gamma`` (the
  per-node 2-component Thomas TDMA on the conservative range; TDMA class).
* ``fer_gamma2vel`` — fed the C's ``fer_gamma`` + static ``helem`` → match
  ``fer_uv`` (gather/difference/÷helem; gather class).

Plus the chained path and AD through the TDMA.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import gm, ops
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
    fields, _ = load_gm_dump(GM_DUMP_DIR)
    cfg = gm.GMConfig()
    # static helem = ⅓Σ_v hnode (linfs), masked to the element layer range.
    helem = ops.gather_nodes_to_elem(
        jnp.asarray(fields["hnode"]), mesh.elem_nodes).mean(axis=1)
    helem = jnp.where(mesh.elem_layer_mask, helem, 0.0)
    return mesh, fields, cfg, helem


def _isclose_viol(got, ref, mask, rtol, atol):
    d = np.abs(got[mask] - ref[mask])
    return float(np.max(d - (atol + rtol * np.abs(ref[mask])))), float(np.max(d))


def test_fer_gamma_matches_c_dump(setup):
    """fer_solve_gamma (2-component Thomas TDMA) vs the dumped fer_gamma."""
    mesh, f, cfg, _ = setup
    ferg = gm.fer_solve_gamma(
        mesh, jnp.asarray(f["sigma_xy"]), jnp.asarray(f["bvfreq"]),
        jnp.asarray(f["fer_K"]), jnp.asarray(f["fer_C"]).reshape(-1), cfg)
    ferg = np.asarray(ferg)
    ref = f["fer_gamma"]                                   # (N,nl,2)
    m = np.broadcast_to(np.asarray(mesh.node_layer_mask)[..., None], ferg.shape)
    v, d = _isclose_viol(ferg, ref, m, rtol=1e-9, atol=1e-13)
    print(f"\nfer_gamma max|Δ|={d:.3e}  viol={v:.3e}  (scale {np.abs(ref[m]).max():.3e})")
    assert v <= 0.0


def test_fer_uv_matches_c_dump(setup):
    """fer_gamma2vel fed the C's fer_gamma + static helem → match fer_uv (elem)."""
    mesh, f, cfg, helem = setup
    feruv = np.asarray(gm.fer_gamma2vel(mesh, jnp.asarray(f["fer_gamma"]), helem))
    ref = f["fer_uv"]                                      # (E,nl,2)
    m = np.broadcast_to(np.asarray(mesh.elem_layer_mask)[..., None], feruv.shape)
    v, d = _isclose_viol(feruv, ref, m, rtol=1e-10, atol=1e-15)
    print(f"\nfer_uv max|Δ|={d:.3e}  viol={v:.3e}  (scale {np.abs(ref[m]).max():.3e})")
    assert v <= 0.0


def test_chained_fer_uv_matches_c_dump(setup):
    """The full path sigma_xy/bvfreq/fer_K/fer_C → Γ → fer_uv vs the C."""
    mesh, f, cfg, helem = setup
    ferg = gm.fer_solve_gamma(
        mesh, jnp.asarray(f["sigma_xy"]), jnp.asarray(f["bvfreq"]),
        jnp.asarray(f["fer_K"]), jnp.asarray(f["fer_C"]).reshape(-1), cfg)
    feruv = np.asarray(gm.fer_gamma2vel(mesh, ferg, helem))
    ref = f["fer_uv"]
    m = np.broadcast_to(np.asarray(mesh.elem_layer_mask)[..., None], feruv.shape)
    v, d = _isclose_viol(feruv, ref, m, rtol=1e-9, atol=1e-15)
    print(f"\nchained fer_uv max|Δ|={d:.3e}  viol={v:.3e}")
    assert v <= 0.0


def test_ad_through_tdma(setup):
    """d(Σfer_uv²)/d(sigma_xy) finite through the streamfunction TDMA (the
    grad-verified ops.tdma) + the bolus reconstruction; nonzero."""
    mesh, f, cfg, helem = setup
    sig0 = jnp.asarray(f["sigma_xy"])
    bv = jnp.asarray(f["bvfreq"]); fK = jnp.asarray(f["fer_K"])
    fC = jnp.asarray(f["fer_C"]).reshape(-1)

    def loss(sig):
        ferg = gm.fer_solve_gamma(mesh, sig, bv, fK, fC, cfg)
        feruv = gm.fer_gamma2vel(mesh, ferg, helem)
        return jnp.sum(feruv * feruv)

    g = np.asarray(jax.grad(loss)(sig0))
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)
