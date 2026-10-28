"""The atmospheric-forcing seam (``Params.forcing_adj``, ported from the ice session's e0381dc): the step adds the
anomalies to the step forcing (``step._apply_forcing_adj``); unset, nothing is traced. Toy arrays only; the CORE2
gates (a zero anomaly is the unperturbed forward, FD, sharded == dense, the monthly maps) are in
``scripts/tests/test_core2_adj_fast.py`` and ``scripts/tests/test_core2_adj_sharded.py``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import step as stepmod
from fesom_jax.params import Params
from fesom_jax.surface_forcing import StepForcing

jax.config.update("jax_enable_x64", True)


def _sf(n=5):
    rng = np.random.default_rng(0)
    return StepForcing(*(jnp.asarray(rng.standard_normal(n)) for _ in StepForcing._fields))


def test_apply_forcing_adj_adds_and_checks_keys():
    sf = _sf()
    d = jnp.arange(5.0)
    out = stepmod._apply_forcing_adj(sf, {"Tair": d, "u_air": -d})
    np.testing.assert_array_equal(np.asarray(out.Tair), np.asarray(sf.Tair + d))
    np.testing.assert_array_equal(np.asarray(out.u_air), np.asarray(sf.u_air - d))
    for f in StepForcing._fields:
        if f not in ("Tair", "u_air"):
            assert getattr(out, f) is getattr(sf, f)
    assert set(stepmod.FORCING_ADJ_KEYS) <= set(StepForcing._fields)
    with pytest.raises(ValueError, match="unknown keys"):
        stepmod._apply_forcing_adj(sf, {"tair": d})
    with pytest.raises(TypeError):
        stepmod._apply_forcing_adj(sf, d)


def test_unset_has_no_leaves():
    p = Params.defaults()
    assert p.forcing_adj is None
    n0 = len(jax.tree.leaves(p))
    p1 = jax.tree_util.tree_map(lambda x: x, p)
    assert len(jax.tree.leaves(p1)) == n0
    import dataclasses
    p2 = dataclasses.replace(p, forcing_adj={"Tair": jnp.zeros(3)})
    assert len(jax.tree.leaves(p2)) == n0 + 1
