"""Data-free checks of the surface-flux adjustment seam (``step._apply_surface_adj``,
``Params.surface_adj``): the arithmetic and sign convention on a four-node toy mesh, and the
guards. The CORE2 behaviour is in ``test_surface_adj.py``."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import step as stepmod
from fesom_jax.config import VCPW
from fesom_jax.params import Params

DT = 1800.0


def _params(adj):
    return dataclasses.replace(Params.defaults(), surface_adj=adj)


def test_apply_surface_adj_arithmetic_and_guards():
    mesh = SimpleNamespace(elem_nodes=jnp.asarray([[0, 1, 2], [1, 2, 3]]))
    n = 4
    bc_T, hf = jnp.zeros(n), jnp.ones(n)
    ss, sns = jnp.zeros((2, 2)), jnp.zeros((n, 2))
    heat = jnp.asarray([10.0, 0.0, 0.0, -5.0])
    sx = jnp.asarray([3.0, 0.0, 0.0, 0.0])
    b2, h2, ss2, sns2 = stepmod._apply_surface_adj(
        mesh, {"heat": heat, "stress_x": sx}, bc_T=bc_T, heat_flux=hf, stress_surf=ss,
        stress_node_surf=sns, dt=DT, halo_ctx=None)
    np.testing.assert_allclose(np.asarray(b2), DT * np.asarray(heat) / VCPW, rtol=1e-15)
    np.testing.assert_allclose(np.asarray(h2), 1.0 - np.asarray(heat), rtol=1e-15)
    np.testing.assert_allclose(np.asarray(sns2[:, 0]), np.asarray(sx))
    np.testing.assert_array_equal(np.asarray(sns2[:, 1]), np.zeros(n))
    np.testing.assert_allclose(np.asarray(ss2), [[1.0, 0.0], [0.0, 0.0]], rtol=1e-15)
    kw = dict(bc_T=bc_T, heat_flux=hf, stress_surf=ss, stress_node_surf=sns, dt=DT)
    with pytest.raises(NotImplementedError):
        stepmod._apply_surface_adj(mesh, {"heat": heat}, halo_ctx=object(), **kw)
    with pytest.raises(ValueError, match="unknown keys"):
        stepmod._apply_surface_adj(mesh, {"heatt": heat}, halo_ctx=None, **kw)
    with pytest.raises(TypeError):
        stepmod._apply_surface_adj(mesh, heat, halo_ctx=None, **kw)
    with pytest.raises(ValueError, match="forced path"):
        stepmod._apply_surface_adj(mesh, {"heat": heat}, halo_ctx=None,
                                   **{**kw, "bc_T": None, "stress_node_surf": None})


def test_params_leaf_default_has_no_leaves():
    p = Params.defaults()
    assert p.surface_adj is None
    assert len(jax.tree_util.tree_leaves(p)) == 8
    q = _params({"heat": jnp.zeros(3)})
    assert len(jax.tree_util.tree_leaves(q)) == 9
