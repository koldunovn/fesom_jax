"""TKE derivative cut points and the column-input assembly on the assembled ``mixing_tke``,
pi mesh (``unit`` group).

pi's own step runs PP mixing, so the TKE driver is called directly with a perturbed velocity,
a seeded tke and a synthetic nodal stress (as in ``test_adjoint_config_tke.py``).

* no cuts ⇒ the identical program: the jaxpr text of ``mixing_tke`` hashes to the value
  pinned from the parent commit (cd2ba05, before the cut points and the column-input
  refactor), with and without an explicit empty set;
* every ``tke.*`` cut leaves the ``mixing_tke`` forward bitwise;
* ``tke_column_inputs`` + the column core reproduce ``mixing_tke``'s ``tke_new`` bitwise.

The column-core tests are in ``test_tke_cuts.py`` (``ci``), the CORE2 step in
``test_tke_cuts_core2.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import cvmix_tke
from fesom_jax import tke as tkemod
from fesom_jax.tests.test_tke_cuts import TKE_CUTS, _sha, check_pin

# sha256 of the jaxpr text of ``mixing_tke_jaxpr_sha()`` without the cut points, jax 0.10.1, CPU, x64: first from
# the parent commit cd2ba05 (26c34b68), re-pinned from perf/core2-gpu 0378994 (see ``test_tke_cuts.PINNED``).
PINNED_PI = {
    "mixing_tke_pi": "e4c31c6448f7e0867c9977a25414b001b8bc5accaf5717a5961197f607054041",
}


def _pi_driver():
    """``mixing_tke`` on the pi mesh from a perturbed state with a seeded tke and a synthetic
    nodal stress; returns ``(mesh, args)``."""
    from fesom_jax import ale
    from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
    from fesom_jax.state import State
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    rng = np.random.default_rng(3)
    mesh = load_mesh()
    st = State.rest(mesh)
    imask = np.asarray(mesh.node_iface_mask)
    tke = jnp.asarray(np.where(imask, 1e-4 * (1.0 + 0.5 * rng.random(imask.shape)), 0.0))
    stress = jnp.asarray(rng.standard_normal((mesh.nod2D, 2)) * 0.05)
    uv = jnp.asarray(np.asarray(st.uv) + 0.05 * rng.standard_normal(np.shape(st.uv)))
    bv = jnp.asarray(np.where(imask, 1e-5 * (1.0 + rng.standard_normal(imask.shape)), 0.0))
    _, Z3d = ale.live_geometry(mesh, st.hnode)
    return mesh, dict(uv=uv, bvfreq=bv, tke=tke, stress_node_surf=stress, hnode=st.hnode,
                      Z3d=Z3d)


def _mixing_tke(mesh, a, cuts=None):
    from fesom_jax.params import Params
    extra = {} if cuts is None else {"cuts": cuts}
    return tkemod.mixing_tke(mesh, a["uv"], a["bvfreq"], a["tke"], a["stress_node_surf"],
                             a["hnode"], tkemod.TkeConfig(), Params.defaults(), dt=1800.0,
                             Z3d=a["Z3d"], **extra)


def mixing_tke_jaxpr_sha(cuts=None) -> str:
    mesh, a = _pi_driver()
    return _sha(str(jax.make_jaxpr(lambda a: _mixing_tke(mesh, a, cuts))(a)))


def test_no_cuts_traces_the_pinned_mixing_tke_program():
    ref = mixing_tke_jaxpr_sha()
    assert mixing_tke_jaxpr_sha(cuts=frozenset()) == ref
    check_pin(PINNED_PI, "mixing_tke_pi", ref)
    assert mixing_tke_jaxpr_sha(cuts=frozenset({"tke.surf"})) != ref


def test_all_cuts_leave_mixing_tke_forward_bitwise():
    mesh, a = _pi_driver()
    ref = jax.jit(lambda a: _mixing_tke(mesh, a))(a)
    got = jax.jit(lambda a: _mixing_tke(mesh, a, frozenset(TKE_CUTS)))(a)
    for i, (u, v) in enumerate(zip(ref, got)):
        assert np.array_equal(np.asarray(u), np.asarray(v)), f"mixing_tke output {i} moved"
    assert float(jnp.max(ref[3])) > 0.0


def test_column_inputs_reproduce_mixing_tke_bitwise():
    from fesom_jax.params import Params
    mesh, a = _pi_driver()
    p = Params.defaults()
    cfg = tkemod.TkeConfig()

    def via_inputs(a):
        d = tkemod.tke_column_inputs(mesh, a["uv"], a["bvfreq"], a["tke"],
                                     a["stress_node_surf"], a["hnode"], Z3d=a["Z3d"])
        return cvmix_tke.integrate_tke_column(
            *(d[k] for k in tkemod.COLUMN_INPUTS), c_k=p.tke_c_k, c_eps=p.tke_c_eps,
            cd=p.tke_cd, alpha_tke=p.tke_alpha, mxl_min=cfg.mxl_min, tke_min=cfg.tke_min,
            kappaM_max=cfg.kappaM_max, dt=1800.0)

    tke_new_model = jax.jit(lambda a: _mixing_tke(mesh, a)[3])(a)
    tke_new_cols, _, _ = jax.jit(via_inputs)(a)
    assert np.array_equal(np.asarray(tke_new_model), np.asarray(tke_new_cols))
    assert float(jnp.max(tke_new_model)) > 0.0
