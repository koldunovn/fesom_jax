"""Task G.5 gate — the GM driver + the bolus vertical velocity.

* ``gm_diagnostics`` — the full GM chain from the ocean state (sw_alpha_beta →
  sigma_xy → neutral_slope → init_redi_gm → fer_solve_gamma → fer_gamma2vel). Fed
  the C's step-1 T/S/bvfreq/hnode_new, its ``fer_uv`` must match the dump end-to-end
  (composing the individually-verified G.1-G.4 kernels).
* ``fer_w`` — the bolus vertical velocity. The C computes it with the SAME
  edge→node transport-divergence scatter + reverse-cumsum + ÷area as ``w``, just
  driven by ``fer_uv`` (``fesom_ale.c:124-152,166-186``), so ``fer_w =
  ale.compute_w(fer_uv)`` — a pure reuse of the dump-verified ``compute_w``. Checked
  for the no-flux BC + finiteness + activity (nonzero).

The bolus-augmented advection's tight effect on T/S is folded into the G.7 full-GM
assembled-step gate (the C dump has bolus + Redi together; isolating bolus-only T/S
would need the FESOM_GM_BOLUS_ONLY knob).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ale, gm, ops
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
    f, _ = load_gm_dump(GM_DUMP_DIR)
    cfg = gm.GMConfig()
    helem = ops.gather_nodes_to_elem(jnp.asarray(f["hnode"]), mesh.elem_nodes).mean(axis=1)
    helem = jnp.where(mesh.elem_layer_mask, helem, 0.0)
    return mesh, f, cfg, helem


def test_gm_diagnostics_fer_uv_matches_c_dump(setup):
    """The whole GM chain from the C's state reproduces fer_uv end-to-end."""
    mesh, f, cfg, helem = setup
    diag = gm.gm_diagnostics(
        mesh, jnp.asarray(f["T"]), jnp.asarray(f["S"]), jnp.asarray(f["bvfreq"]),
        jnp.asarray(f["hnode_new"]), helem, Params.defaults(), cfg)
    feruv = np.asarray(diag.fer_uv)
    ref = f["fer_uv"]
    m = np.broadcast_to(np.asarray(mesh.elem_layer_mask)[..., None], feruv.shape)
    d = np.max(np.abs(feruv[m] - ref[m]))
    print(f"\ngm_diagnostics fer_uv max|Δ|={d:.3e} (scale {np.abs(ref[m]).max():.3e})")
    assert d < 1e-9                      # composes verified kernels; gather/TDMA floor
    # slope_tapered / Ki also flow out for G.6:
    assert np.all(np.isfinite(np.asarray(diag.slope_tapered)))
    assert np.all(np.isfinite(np.asarray(diag.Ki)))


def test_fer_w_is_compute_w_of_fer_uv(setup):
    """fer_w = compute_w(fer_uv): no-flux BC (w[nzmax]=0), finite, and ACTIVE."""
    mesh, f, cfg, helem = setup
    fer_uv = jnp.asarray(f["fer_uv"])
    fer_w = np.asarray(ale.compute_w(mesh, fer_uv, helem))
    assert np.all(np.isfinite(fer_w))
    # no-flux bottom BC: at each node's bottom interface (nlevels-1) fer_w == 0.
    nlev = np.asarray(mesh.nlevels_nod2D)
    bot = fer_w[np.arange(fer_w.shape[0]), nlev - 1]
    assert np.max(np.abs(bot)) < 1e-20
    # the bolus is genuinely active somewhere (nonzero vertical velocity).
    assert np.max(np.abs(fer_w)) > 0.0
    print(f"\nfer_w max|w|={np.max(np.abs(fer_w)):.3e}  bottom max|w|={np.max(np.abs(bot)):.3e}")


def test_ad_through_gm_diagnostics(setup):
    """d(Σfer_uv²)/d(T) through the whole driver finite + nonzero (the full chain:
    sw_alpha_beta + the sigma_xy scatter + the slope safe-sqrts + the TDMA)."""
    mesh, f, cfg, helem = setup
    T0 = jnp.asarray(f["T"])
    S = jnp.asarray(f["S"]); bv = jnp.asarray(f["bvfreq"]); hn = jnp.asarray(f["hnode_new"])

    def loss(T):
        diag = gm.gm_diagnostics(mesh, T, S, bv, hn, helem, Params.defaults(), cfg)
        return jnp.sum(diag.fer_uv * diag.fer_uv)

    g = np.asarray(jax.grad(loss)(T0))
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)
