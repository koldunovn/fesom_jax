"""Task 1.2: the State pytree is correctly shaped and behaves under JAX
transforms (tree_map, scan, grad) — the structural half of GATE 1."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, tree_util

from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.state import State

pytestmark = pytest.mark.skipif(
    not DEFAULT_PI_MESH_DIR.is_dir(),
    reason=f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}",
)


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(DEFAULT_PI_MESH_DIR)


def _expected_shapes(mesh):
    n, e, nl = mesh.nod2D, mesh.elem2D, mesh.nl
    return {
        "T": (n, nl), "S": (n, nl), "T_old": (n, nl), "S_old": (n, nl),
        "del_ttf": (n, nl),
        "uv": (e, nl, 2), "uv_rhs": (e, nl, 2), "uv_rhsAB": (e, nl, 2),
        "uvnode": (n, nl, 2), "uvnode_rhs": (n, nl, 2),
        "w": (n, nl), "w_e": (n, nl), "w_i": (n, nl), "cfl_z": (n, nl),
        "eta_n": (n,), "d_eta": (n,), "ssh_rhs": (n,), "ssh_rhs_old": (n,),
        "hnode": (n, nl), "hnode_new": (n, nl), "helem": (e, nl),
        "hbar": (n,), "hbar_old": (n,),
        "density": (n, nl), "hpressure": (n, nl), "bvfreq": (n, nl), "Kv": (n, nl),
        "Av": (e, nl), "pgf_x": (e, nl), "pgf_y": (e, nl),
        # sea ice (Phase 6, surface-only 2-D)
        "a_ice": (n,), "m_ice": (n,), "m_snow": (n,), "u_ice": (n,), "v_ice": (n,),
        "t_skin": (n,), "sigma11": (e,), "sigma12": (e,), "sigma22": (e,),
    }


def test_zeros_shapes_and_dtype(mesh):
    st = State.zeros(mesh)
    exp = _expected_shapes(mesh)
    names = {f.name for f in dataclasses.fields(State)}
    assert names == set(exp), "State fields drifted from the expected inventory"
    for name, shape in exp.items():
        arr = getattr(st, name)
        assert arr.shape == shape, f"{name}: {arr.shape} != {shape}"
        assert arr.dtype == jnp.float64, f"{name}: {arr.dtype}"
        assert np.all(np.asarray(arr) == 0.0)


def test_rest_state(mesh):
    st = State.rest(mesh, T0=10.0, S0=35.0)
    assert np.all(np.asarray(st.T) == 10.0)
    assert np.all(np.asarray(st.S) == 35.0)
    np.testing.assert_array_equal(np.asarray(st.T_old), np.asarray(st.T))
    # flow & SSH at rest
    for fld in ("uv", "w", "eta_n", "d_eta", "hbar"):
        assert np.all(np.asarray(getattr(st, fld)) == 0.0)
    # thickness is positive on valid layers, zero below bottom, sums to depth
    hnode = np.asarray(st.hnode)
    mask = np.asarray(mesh.node_layer_mask)
    assert np.all(hnode[mask] > 0.0)
    assert np.all(hnode[~mask] == 0.0)
    # column thickness sum == depth of deepest valid interface (per-node zbar_3d_n)
    z_n = np.asarray(mesh.zbar_3d_n)
    nln = np.asarray(mesh.nlevels_nod2D)
    col_depth = -z_n[np.arange(mesh.nod2D), nln - 1]            # ulevels=1 ⇒ top at 0
    np.testing.assert_allclose(hnode.sum(axis=1), col_depth, rtol=1e-12, atol=1e-6)


def test_pytree_roundtrip_and_treemap(mesh):
    st = State.rest(mesh)
    leaves, treedef = tree_util.tree_flatten(st)
    assert len(leaves) == len(dataclasses.fields(State))
    st2 = tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(np.asarray(st2.T), np.asarray(st.T))
    st3 = tree_util.tree_map(lambda x: x + 1.0, st)
    np.testing.assert_array_equal(np.asarray(st3.eta_n), np.asarray(st.eta_n) + 1.0)


def test_state_flows_through_scan_and_grad(mesh):
    """A trivial differentiable 'step' over State proves the pytree is usable in
    lax.scan + jax.grad (the Phase-3 integration pattern, smoke-tested early)."""
    st0 = State.zeros(mesh)

    def step(state, k):
        # touch one node and one element field so both entity sizes participate
        new = dataclasses.replace(
            state,
            T=state.T + 0.5,
            uv=state.uv + 1.0,
        )
        return new, jnp.sum(new.T[0])

    final, ys = lax.scan(step, st0, jnp.arange(3))
    assert ys.shape == (3,)
    np.testing.assert_allclose(np.asarray(final.T[0]), 1.5, rtol=0, atol=0)

    def loss(scale):
        s0 = dataclasses.replace(st0, T=st0.T + scale)
        sf, _ = lax.scan(step, s0, jnp.arange(4))
        return jnp.sum(sf.T)

    g = jax.grad(loss)(2.0)
    # dloss/dscale = number of T entries (scale added once to every T element)
    assert float(g) == pytest.approx(mesh.nod2D * mesh.nl)
