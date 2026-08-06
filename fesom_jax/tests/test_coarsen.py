"""Tests for the mesh-Laplacian scale filter (NN-GM Task 5).

Runs on the packaged `pi` mesh — small, real FESOM geometry, CPU-fast. The properties
checked here are the ones the training data depends on: a filter that shifted the mean
or leaked across a coastline would put a bias into every target the network learns from.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import coarsen
from fesom_jax.mesh import load_mesh

PI_MESH_DIR = Path(__file__).resolve().parents[1] / "data" / "mesh_pi"   # packaged with the model


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(PI_MESH_DIR)


def _area(mesh):
    return np.asarray(mesh.area[:, 0] if mesh.area.ndim == 2 else mesh.area)


def test_constant_field_is_preserved_exactly(mesh):
    """A constant must come through untouched — the Laplacian annihilates it."""
    phi = jnp.full((mesh.nod2D,), 3.25)
    lap = coarsen.laplacian(mesh, phi)
    assert float(jnp.max(jnp.abs(lap))) < 1e-9, "Laplacian of a constant must vanish"
    out = coarsen.filter_field(mesh, phi, n_steps=25, kappa_dt=1.0e6)
    assert np.allclose(np.asarray(out), 3.25, rtol=0, atol=1e-9)


def test_area_weighted_mean_is_conserved(mesh):
    """The filter may move a field around, but not change its area-weighted mean."""
    rng = np.random.default_rng(0)
    phi = jnp.asarray(rng.normal(10.0, 2.0, mesh.nod2D))
    a = _area(mesh)
    m0 = float((a * np.asarray(phi)).sum() / a.sum())
    out = coarsen.filter_field(mesh, phi, n_steps=20, kappa_dt=5.0e5)
    m1 = float((a * np.asarray(out)).sum() / a.sum())
    assert abs(m1 - m0) / max(abs(m0), 1e-30) < 1e-12, (m0, m1)


def test_filter_reduces_variance_monotonically(mesh):
    """Smoothing must remove variance, and more steps must remove more."""
    rng = np.random.default_rng(1)
    phi = jnp.asarray(rng.normal(0.0, 1.0, mesh.nod2D))
    v = [float(jnp.var(coarsen.filter_field(mesh, phi, n_steps=n, kappa_dt=2.0e5)))
         for n in (0, 5, 20, 60)]
    assert v[0] > v[1] > v[2] > v[3], v


def test_small_scales_are_damped_more_than_large(mesh):
    """The point of a scale filter: grid-scale structure must lose far more amplitude
    than domain-scale structure.

    The filter scale is set RELATIVE TO THE MESH (3x the median node spacing) via the
    helpers — a filter scale chosen without reference to the local spacing is simply a
    no-op, which is exactly how the first version of this test fooled itself.
    Grid-scale structure is represented by white noise (all its variance sits at the
    mesh scale); domain-scale structure by a single wave across the domain.
    """
    n, kdt = coarsen.plan_filter(mesh, factor=3.0)

    lon = np.asarray(mesh.geo_coord_nod2D[:, 0])
    large = jnp.asarray(np.sin(lon))                        # one wave over the domain
    small = jnp.asarray(np.random.default_rng(7).normal(0.0, 1.0, mesh.nod2D))
    kw = dict(n_steps=n, kappa_dt=kdt)
    r_large = float(jnp.std(coarsen.filter_field(mesh, large, **kw)) / jnp.std(large))
    r_small = float(jnp.std(coarsen.filter_field(mesh, small, **kw)) / jnp.std(small))
    assert r_small < r_large, (r_small, r_large)
    assert r_large > 0.8, f"the domain-scale wave should mostly survive, kept {r_large:.3f}"
    assert r_small < 0.5, f"grid-scale noise should be strongly damped, kept {r_small:.3f}"


def test_mask_blocks_transport_across_it(mesh):
    """A masked (dry) region must neither donate nor receive — the no-flux condition."""
    n = mesh.nod2D
    m = np.ones(n)
    lon = np.asarray(mesh.geo_coord_nod2D[:, 0])
    dry = lon > np.median(lon)
    m[dry] = 0.0
    phi = np.zeros(n)
    phi[~dry] = 1.0                                        # unit field only in the wet half
    out = np.asarray(coarsen.filter_field(mesh, jnp.asarray(phi), n_steps=15,
                                          kappa_dt=4.0e5, node_mask=jnp.asarray(m)))
    assert np.allclose(out[dry], 0.0), "dry nodes must stay exactly zero"
    a = _area(mesh)[~dry]
    m0 = (a * phi[~dry]).sum() / a.sum()
    m1 = (a * out[~dry]).sum() / a.sum()
    assert abs(m1 - m0) < 1e-12, "mass must be conserved within the wet region"


def test_masked_mean_conserved_with_realistic_mask(mesh):
    rng = np.random.default_rng(2)
    n = mesh.nod2D
    m = np.ones(n)
    m[rng.random(n) < 0.15] = 0.0                          # scattered dry nodes
    phi = rng.normal(5.0, 1.0, n) * m
    a = _area(mesh) * m
    out = np.asarray(coarsen.filter_field(mesh, jnp.asarray(phi), n_steps=12,
                                          kappa_dt=3.0e5, node_mask=jnp.asarray(m)))
    m0 = (a * phi).sum() / a.sum()
    m1 = (a * out).sum() / a.sum()
    assert abs(m1 - m0) / abs(m0) < 1e-12


def test_three_dimensional_field_filters_per_level(mesh):
    """A (nod2D, nl) field must filter level by level, without mixing levels."""
    rng = np.random.default_rng(3)
    phi = np.zeros((mesh.nod2D, 3))
    phi[:, 0] = rng.normal(0, 1, mesh.nod2D)
    phi[:, 2] = 7.0                                        # constant level
    out = np.asarray(coarsen.filter_field(mesh, jnp.asarray(phi), n_steps=10, kappa_dt=2.0e5))
    assert np.allclose(out[:, 1], 0.0), "an empty level must stay empty"
    assert np.allclose(out[:, 2], 7.0, atol=1e-9), "a constant level must be preserved"
    assert np.var(out[:, 0]) < np.var(phi[:, 0]), "the noisy level must be smoothed"


def test_plan_filter_is_stable_on_a_variable_mesh(mesh):
    """plan_filter must satisfy the stability bound at EVERY node, including the finest.

    Regression: choosing the step count from the median spacing (230 km on the pi mesh)
    while the finest node is 35 km overshoots the limit by ~44x, and the filter amplifies
    instead of smoothing — the field reached 1e47.
    """
    n, kdt = coarsen.plan_filter(mesh, factor=3.0)
    limit = coarsen.STABILITY_C * np.asarray(coarsen.node_resolution(mesh)) ** 2
    assert (np.asarray(kdt) <= limit * (1 + 1e-12)).all()
    phi = jnp.asarray(np.random.default_rng(11).normal(0, 1, mesh.nod2D))
    out = np.asarray(coarsen.filter_field(mesh, phi, n_steps=n, kappa_dt=kdt))
    assert np.isfinite(out).all() and np.abs(out).max() < np.abs(np.asarray(phi)).max()


def test_unstable_kappa_dt_is_refused(mesh):
    """The stability guard must fire rather than silently returning garbage."""
    bad = 10.0 * coarsen.STABILITY_C * float(np.asarray(coarsen.node_resolution(mesh)).min()) ** 2 * 100
    with pytest.raises(ValueError, match="stability limit"):
        coarsen.filter_field(mesh, jnp.zeros(mesh.nod2D), n_steps=2, kappa_dt=bad)


def test_scale_and_step_helpers_are_consistent():
    dx = 1.0e5
    target = 2.0e5
    n = coarsen.steps_for_scale(target, dx)
    kdt = coarsen.kappa_dt_for_scale(target, n)
    assert kdt <= coarsen.STABILITY_C * dx ** 2 * (1 + 1e-12), "must respect the stability bound"
    assert coarsen.scale_of(n, kdt) == pytest.approx(target, rel=1e-12)


def test_step_count_grows_quadratically_with_scale_separation():
    """Documented cost scaling — the reason the pipeline aggregates before filtering."""
    dx = 5.0e3
    n_small = coarsen.steps_for_scale(2 * dx, dx)
    n_big = coarsen.steps_for_scale(20 * dx, dx)
    assert n_big / n_small == pytest.approx(100.0, rel=0.05)


def test_node_resolution_is_positive(mesh):
    r = np.asarray(coarsen.node_resolution(mesh))
    assert (r > 0).all() and np.isfinite(r).all()
