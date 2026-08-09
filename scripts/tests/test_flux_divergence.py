"""``scripts/nn_eddy/flux_divergence.py`` — the M3b divergence operator.

The operator re-derives median-dual geometry that already exists inside
``gm_redi.diff_part_hor_redi``, because that file is dump-verified against the C, sits inside
the differentiated step and carries every production run, so it is not to be refactored to
expose it. A re-derived copy of delicate geometry is only worth having if it proves itself, so
the central test here feeds it the Redi flux and demands the reference's own answer back.

The trap being guarded is the partial-cell branching: an edge's two adjacent elements need not
both reach a given layer, ``CX``/``CY``/``dz`` differ in all three cases, and a wrong copy
fails ONLY near topography. So ``test_all_three_partial_cell_cases_occur`` asserts the fixture
actually exercises all three — without it the equivalence test could pass while never touching
the branch that matters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "nn_eddy"))

import fesom_jax  # noqa: E402,F401
import jax.numpy as jnp  # noqa: E402

from fesom_jax.mesh import load_mesh  # noqa: E402

import flux_divergence as fd  # noqa: E402

PI_MESH_DIR = _ROOT / "fesom_jax" / "data" / "mesh_pi"
DT = 1800.0


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(PI_MESH_DIR)


@pytest.fixture(scope="module")
def fields(mesh):
    """Random but correctly masked inputs. The equivalence is an algebraic identity, so random
    fields test it better than smooth ones: they leave no room for two different formulas to
    agree by accident on a special case."""
    rng = np.random.default_rng(20260809)
    N, E, nl = int(mesh.nod2D), int(mesh.elem2D), int(mesh.nl)
    nmask = np.asarray(mesh.node_layer_mask)
    emask = np.asarray(mesh.elem_layer_mask)
    T = jnp.asarray(np.where(nmask, rng.normal(size=(N, nl)) * 3.0 + 5.0, 0.0))
    Ki = jnp.asarray(np.where(nmask, rng.uniform(200.0, 900.0, size=(N, nl)), 0.0))
    slope = jnp.asarray(rng.normal(size=(N, nl - 1, 3)) * 1e-3)
    hnode = jnp.asarray(np.where(nmask, rng.uniform(8.0, 60.0, size=(N, nl)), 0.0))
    hnode_new = jnp.asarray(np.where(nmask, rng.uniform(8.0, 60.0, size=(N, nl)), 0.0))
    helem = jnp.asarray(np.where(emask, rng.uniform(8.0, 60.0, size=(E, nl)), 0.0))
    return dict(T=T, Ki=Ki, slope=slope, hnode=hnode, hnode_new=hnode_new, helem=helem)


def test_all_three_partial_cell_cases_occur(mesh):
    """The fixture must exercise el1-only, el2-only AND both — otherwise the equivalence test
    below is vacuous exactly where a fresh copy of this geometry goes wrong."""
    el1, el2 = np.asarray(mesh.edge_tri[:, 0]), np.asarray(mesh.edge_tri[:, 1])
    has1, has2 = el1 >= 0, el2 >= 0
    emask = np.asarray(mesh.elem_layer_mask)[:, : int(mesh.nl) - 1]
    in1 = emask[np.where(has1, el1, 0)] & has1[:, None]
    in2 = emask[np.where(has2, el2, 0)] & has2[:, None]
    both, only1, only2 = (in1 & in2).sum(), (in1 & ~in2).sum(), (in2 & ~in1).sum()
    assert both > 0 and only1 > 0 and only2 > 0, (
        f"partial-cell coverage is incomplete on this mesh: both={both}, only1={only1}, "
        f"only2={only2} — the equivalence test would not touch the branching")


def test_matches_diff_part_hor_redi(mesh, fields):
    """THE test: fed the Redi flux, the operator must return the model's own answer."""
    worst, scale = fd.check_equivalence(
        mesh, fields["T"], fields["slope"], fields["Ki"],
        fields["hnode"], fields["hnode_new"], fields["helem"], dt=DT)
    assert scale > 0.0, "the reference tendency is identically zero — the fixture proves nothing"
    # EXACTLY zero, not a tolerance. Both sides run the same operations in the same order, so
    # any difference at all is a difference of formula — which is the house precedent
    # (freeze_equivalence.py, max|difference| = 0). An earlier draft normalised first and
    # un-normalised afterwards; that reassociation alone cost 6.9e-18 on a field of scale
    # 8.4e-02 and would have hidden a real discrepancy of the same size behind a round-off bar.
    assert worst == 0.0, (
        f"max|difference| {worst:.3e} against a reference of scale {scale:.3e}; the "
        f"partial-cell branching differs from gm_redi.diff_part_hor_redi")


def test_divergence_conserves_globally(mesh, fields):
    """The edge scatter is antisymmetric (+c to one endpoint, -c to the other), so the
    volume-weighted integral of the divergence is zero by construction for ANY flux. This is
    what makes a kappa fitted against the divergence meaningful: the operator can redistribute
    tracer but never create it."""
    rng = np.random.default_rng(7)
    E, nl = int(mesh.elem2D), int(mesh.nl)
    emask = np.asarray(mesh.elem_layer_mask)
    Fx = jnp.asarray(np.where(emask, rng.normal(size=(E, nl)), 0.0))
    Fy = jnp.asarray(np.where(emask, rng.normal(size=(E, nl)), 0.0))
    div = fd.divergence_of_elem_flux(mesh, Fx, Fy, fields["hnode"], fields["helem"])
    vol = jnp.asarray(mesh.areasvol) * fields["hnode"]
    total = float(jnp.sum(div * vol))
    gross = float(jnp.sum(jnp.abs(div) * vol))
    assert gross > 0.0, "the test flux produced no divergence at all"
    assert abs(total) <= 1e-12 * gross, (
        f"volume-weighted divergence sums to {total:.3e} against a gross {gross:.3e} — the "
        f"edge scatter is not antisymmetric")


def test_uniform_flux_reproduces_the_spherical_metric_term(mesh, fields):
    """A uniform flux is NOT divergence-free on a sphere, and pinning that down is what fixes
    the operator's SIGN — which the equivalence test structurally cannot see, because it runs
    through ``divergence_integral``, common to both sides.

    For constant ``(Fx, Fy)`` in the local Cartesian frame,
    ``∇·F = (1/(R cosφ))[∂Fx/∂λ + ∂(Fy cosφ)/∂φ] = -Fy tanφ / R``.
    So this is a quantitative check of the continuous operator, not just of closure: it
    convicts a flipped sign, a swapped ``CX``/``CY``, and a wrong radius alike. The first draft
    of this file returned ``-∇·F`` and was caught here at correlation 0.99998, slope -0.9975.

    Two conditions, both about the argument rather than convenience. The dual cell must be
    COMPLETE at the layer — every incident edge having two elements that both reach it — or the
    cell is bounded by coastline and the analytic form does not apply. And the layer thickness
    must be UNIFORM, because each boundary segment enters weighted by its own ``dz``.
    """
    R_EARTH = 6367500.0                     # fesom_jax.config.R_EARTH
    FX, FY = 1.0, -0.5
    nl = int(mesh.nl)
    emask = np.asarray(mesh.elem_layer_mask)
    nmask = np.asarray(mesh.node_layer_mask)
    Fx = jnp.asarray(np.where(emask, FX, 0.0))
    Fy = jnp.asarray(np.where(emask, FY, 0.0))
    helem = jnp.asarray(np.where(emask, 25.0, 0.0))
    hnode = jnp.asarray(np.where(nmask, 25.0, 0.0))
    div = np.asarray(fd.divergence_of_elem_flux(mesh, Fx, Fy, hnode, helem))

    edges = np.asarray(mesh.edges)
    el1, el2 = np.asarray(mesh.edge_tri[:, 0]), np.asarray(mesh.edge_tri[:, 1])
    has1, has2 = el1 >= 0, el2 >= 0
    in1 = emask[np.where(has1, el1, 0)] & has1[:, None]
    in2 = emask[np.where(has2, el2, 0)] & has2[:, None]
    full = in1 & in2
    complete = np.ones((int(mesh.nod2D), nl), dtype=bool)
    np.logical_and.at(complete, edges[:, 0], full)
    np.logical_and.at(complete, edges[:, 1], full)
    sel = complete & nmask
    assert sel.sum() > 100, f"only {sel.sum()} complete dual cells — too few to test with"

    lat = np.asarray(mesh.coord_nod2D)[:, 1]
    pred = np.broadcast_to((-FY * np.tan(lat) / R_EARTH)[:, None], div.shape)
    d, p = div[sel], pred[sel]
    assert np.sqrt((p ** 2).mean()) > 0.0, "the analytic metric term is zero — nothing tested"
    corr = float(np.corrcoef(d, p)[0, 1])
    slope = float(np.dot(d, p) / np.dot(p, p))
    assert corr > 0.999, (
        f"correlation with the analytic metric term is only {corr:.6f} on {sel.sum()} complete "
        f"dual cells — the operator is not computing a spherical divergence")
    assert abs(slope - 1.0) < 0.02, (
        f"slope against the analytic metric term is {slope:+.6f}, not +1. A slope near -1 means "
        f"the SIGN is flipped; a slope near +1 but off by more than the discretisation error "
        f"means CX/CY or the radius are wrong")


def test_node_flux_path_matches_the_same_metric_term(mesh):
    """The NODE entry point is what the M3b fit actually calls — the coarse teacher products
    carry ``uT_eddy``/``vT_eddy`` on ``nod2``, not on elements — so it needs its own conviction,
    not inheritance from the element path. It uses a different edge rule (mean of the two
    endpoints, versus combination by level membership), and swapping the two rules is a silent
    error that stays conservative and therefore survives the other tests here.
    """
    R_EARTH = 6367500.0
    FX, FY = 1.0, -0.5
    nl = int(mesh.nl)
    emask = np.asarray(mesh.elem_layer_mask)
    nmask = np.asarray(mesh.node_layer_mask)
    Fxn = jnp.asarray(np.where(nmask, FX, 0.0))
    Fyn = jnp.asarray(np.where(nmask, FY, 0.0))
    helem = jnp.asarray(np.where(emask, 25.0, 0.0))
    hnode = jnp.asarray(np.where(nmask, 25.0, 0.0))
    div = np.asarray(fd.divergence_of_node_flux(mesh, Fxn, Fyn, hnode, helem))

    edges = np.asarray(mesh.edges)
    el1, el2 = np.asarray(mesh.edge_tri[:, 0]), np.asarray(mesh.edge_tri[:, 1])
    has1, has2 = el1 >= 0, el2 >= 0
    full = (emask[np.where(has1, el1, 0)] & has1[:, None]) & \
           (emask[np.where(has2, el2, 0)] & has2[:, None])
    complete = np.ones((int(mesh.nod2D), nl), dtype=bool)
    np.logical_and.at(complete, edges[:, 0], full)
    np.logical_and.at(complete, edges[:, 1], full)
    sel = complete & nmask
    assert sel.sum() > 100, f"only {sel.sum()} complete dual cells — too few to test with"

    lat = np.asarray(mesh.coord_nod2D)[:, 1]
    pred = np.broadcast_to((-FY * np.tan(lat) / R_EARTH)[:, None], div.shape)
    d, p = div[sel], pred[sel]
    corr = float(np.corrcoef(d, p)[0, 1])
    slope = float(np.dot(d, p) / np.dot(p, p))
    assert corr > 0.999 and abs(slope - 1.0) < 0.02, (
        f"node-flux path: correlation {corr:.6f}, slope {slope:+.6f} against the analytic "
        f"metric term — the endpoint-average edge rule is wrong")
