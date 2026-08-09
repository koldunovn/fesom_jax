"""``scripts/nn_eddy/divergence_fit.py`` — the M3b estimator, and above all its OWN error.

The module reports a negative result on real data: fitting κ to ``∇·F`` does not rescue the
GM projection. A negative result is only worth anything if the instrument that produced it is
characterised, and this one has a known approximation baked in. It solves

    κ̂ = -(∇·F) / ∇²σ

which is exact only for a spatially CONSTANT κ, because in general
``∇·(-κ∇σ) = -κ∇²σ - ∇κ·∇σ`` and the second term is dropped. Real κ varies by orders of
magnitude across the domain, so the dropped term is not obviously small, and if it alone can
manufacture a 43 % up-gradient population then the real-data numbers say nothing about the
ocean. These tests measure that, on analytic fields where the true κ is known.

(The *other* M3b measurement — the divergent-fraction ratio η_teacher/η_downgradient — does not
use this estimator at all: it pushes two flux fields through the same operator and compares. It
is unaffected by anything measured here, which is why it, not κ̂_div, carries the conclusion.)
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

from fesom_jax import ale  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.state import State  # noqa: E402

import flux_divergence as fd  # noqa: E402

PI_MESH_DIR = _ROOT / "fesom_jax" / "data" / "mesh_pi"
R_EARTH = 6367500.0


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(PI_MESH_DIR)


@pytest.fixture(scope="module")
def geom(mesh):
    hnode = State.rest(mesh).hnode
    _, helem = ale.commit_thickness(mesh, hnode)
    return hnode, helem


def _analytic(mesh, kappa_varies: bool, amp: float = 0.5):
    """A smooth σ with its exact gradient, a κ field, and the flux ``F = -κ∇σ``.

    Analytic gradients rather than discrete ones on purpose: then the only errors in κ̂ are the
    operator's discretisation and the dropped ``∇κ·∇σ``, which is exactly what is being sized.
    """
    nl1 = int(mesh.nl) - 1
    lon = np.asarray(mesh.coord_nod2D)[:, 0][:, None]
    lat = np.asarray(mesh.coord_nod2D)[:, 1][:, None]
    k, m = 3.0, 2.0
    coslat = np.maximum(np.cos(lat), 1e-6)
    sx = (k * np.cos(k * lon) * np.cos(m * lat)) / (R_EARTH * coslat)
    sy = (-m * np.sin(k * lon) * np.sin(m * lat)) / R_EARTH
    grad = np.broadcast_to(np.stack([sx, sy], -1), (lon.shape[0], nl1, 2)).copy()
    kap = np.full((lon.shape[0], nl1), 500.0)
    if kappa_varies:
        # exp form, so `amp` is a LOG range: amp=0.5 is roughly +-50 %, amp=2.3 a full decade.
        kap = kap * np.exp(amp * np.sin(lon) * np.sin(2.0 * lat))
    F = -kap[..., None] * grad
    return grad, kap, F


def _kappa_hat(mesh, geom, grad, F):
    hnode, helem = geom
    nl1 = int(mesh.nl) - 1
    div = lambda v: np.asarray(fd.divergence_of_node_flux(  # noqa: E731
        mesh, jnp.asarray(v[..., 0]), jnp.asarray(v[..., 1]), hnode, helem))[:, :nl1]
    return -div(F), div(grad)


def test_constant_kappa_is_recovered_exactly(mesh, geom):
    """With κ constant the estimator is exact by linearity of the divergence, so this is the
    test of the SIGN and the wiring rather than of the physics. It would fail on a flipped sign,
    on the wrong edge rule, or on numerator and denominator using different operators."""
    grad, kap, F = _analytic(mesh, kappa_varies=False)
    num, den = _kappa_hat(mesh, geom, grad, F)
    good = np.abs(den) > np.percentile(np.abs(den), 50)
    khat = num[good] / den[good]
    assert good.sum() > 1000, "too few cells with a usable denominator to test with"
    rel = np.abs(khat - kap[good]) / kap[good]
    assert np.max(rel) < 1e-9, (
        f"constant-κ recovery is not exact: worst relative error {np.max(rel):.3e}")
    assert np.all(khat > 0), "recovered κ must be positive for a down-gradient flux"


def test_varying_kappa_bias_is_measured_not_assumed(mesh, geom):
    """κ varying by ±50 % over the domain — mild against the real fer_K range. This records how
    much of the real-data spread the estimator can produce ON ITS OWN.

    The numbers are asserted loosely and PRINTED: the point is the record, not a bar. If a
    future change to the operator moves them, the real-data conclusion has to be re-read."""
    grad, kap, F = _analytic(mesh, kappa_varies=True)
    num, den = _kappa_hat(mesh, geom, grad, F)
    good = np.abs(den) > np.percentile(np.abs(den), 50)
    r = (num[good] / den[good]) / kap[good]
    med, neg = float(np.median(r)), float(np.mean(r < 0))
    inb = float(np.mean((r >= 1 / 3) & (r <= 3)))
    print(f"\n[estimator bias, ±50 % κ] median κ̂/κ {med:.3f}, up-gradient {neg:.1%}, "
          f"inside [1/3,3] {inb:.1%}, p10 {np.percentile(r, 10):+.3f}, "
          f"p90 {np.percentile(r, 90):+.3f}")
    # The dropped ∇κ·∇σ must not, by itself, reproduce the real-data failure. On real products
    # the divergence fit gave median 0.05 with 43 % up-gradient; if the estimator manufactures
    # that from a mild κ gradient alone, the real numbers are about the instrument.
    assert 0.5 < med < 2.0, f"median κ̂/κ = {med:.3f}: the estimator is biased on its own"
    assert neg < 0.20, (
        f"a ±50 % κ variation alone produces {neg:.1%} up-gradient cells — comparable to the "
        f"43 % measured on real products, so κ̂_div cannot support a conclusion about the ocean")


def test_bias_at_a_realistic_kappa_range(mesh, geom):
    """The honest stress test. Real ``fer_K`` spans orders of magnitude, not ±50 %, so the mild
    case above is not enough to clear the estimator. Sweep the κ contrast up to two decades and
    record where — if anywhere — the dropped ``∇κ·∇σ`` starts manufacturing up-gradient cells."""
    print("\n[estimator bias vs κ contrast]")
    worst_neg = 0.0
    for amp, label in ((0.5, "±50 %"), (1.15, "×10 range"), (2.3, "×100 range")):
        grad, kap, F = _analytic(mesh, kappa_varies=True, amp=amp)
        num, den = _kappa_hat(mesh, geom, grad, F)
        good = np.abs(den) > np.percentile(np.abs(den), 50)
        r = (num[good] / den[good]) / kap[good]
        med, neg = float(np.median(r)), float(np.mean(r < 0))
        worst_neg = max(worst_neg, neg)
        print(f"  κ contrast {label:11s} median κ̂/κ {med:6.3f}  up-gradient {neg:6.2%}  "
              f"inside [1/3,3] {np.mean((r >= 1/3) & (r <= 3)):6.1%}")
    assert worst_neg < 0.20, (
        f"at a realistic κ contrast the estimator alone yields {worst_neg:.1%} up-gradient "
        f"cells, comparable to the 43 % measured on real products — κ̂_div then says nothing "
        f"about the ocean and only the divergent-fraction ratio may be quoted")


def test_a_purely_rotational_flux_is_annihilated(mesh, geom):
    """The property the whole M3b idea rests on: ``∇·F = 0`` for a rotational flux, so fitting
    against the divergence removes it exactly rather than approximately."""
    nl1 = int(mesh.nl) - 1
    lon = np.asarray(mesh.coord_nod2D)[:, 0][:, None]
    lat = np.asarray(mesh.coord_nod2D)[:, 1][:, None]
    # F = k x grad(psi): components (-d psi/dy, +d psi/dx), divergence-free in the continuum
    coslat = np.maximum(np.cos(lat), 1e-6)
    px = (2.0 * np.cos(2.0 * lon) * np.cos(lat)) / (R_EARTH * coslat)
    py = (-np.sin(2.0 * lon) * np.sin(lat)) / R_EARTH
    rot = np.broadcast_to(np.stack([-py, px], -1), (lon.shape[0], nl1, 2)).copy()
    grad = np.broadcast_to(np.stack([px, py], -1), (lon.shape[0], nl1, 2)).copy()
    hnode, helem = geom
    div = lambda v: np.asarray(fd.divergence_of_node_flux(  # noqa: E731
        mesh, jnp.asarray(v[..., 0]), jnp.asarray(v[..., 1]), hnode, helem))[:, :nl1]
    d_rot, d_grad = np.abs(div(rot)), np.abs(div(grad))
    # Discrete, so not exactly zero — but it must be far below the divergence of the gradient
    # field built from the SAME potential, which is the only fair yardstick.
    ratio = float(np.median(d_rot[d_grad > 0]) / np.median(d_grad[d_grad > 0]))
    print(f"\n[rotational annihilation] median |∇·F_rot| / |∇·F_grad| = {ratio:.4f}")
    assert ratio < 0.25, (
        f"a rotational flux keeps {ratio:.3f} of a gradient flux's divergence — the operator "
        f"does not annihilate it well enough for the M3b argument to rest on it")
