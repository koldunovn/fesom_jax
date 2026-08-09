"""Horizontal divergence of a tracer flux on the FESOM median-dual mesh (plan M3b).

**Why this exists.** The offline kappa projection fits an eddy flux ``F`` against ``-K∇σ``,
and on real coarse-grained NG5 products it keeps only 12.5 % of columns, puts the median
kappa-hat at 0.070 of the formula value, and leaves 40.7 % of what it keeps pointing
UP-gradient. The standing explanation is a large rotational component in the diagnosed
flux — a part of ``F`` that moves no tracer at all. That explanation is INFERRED, never
measured.

A rotational flux has ``∇·F ≡ 0``. So fitting ``kappa`` to the DIVERGENCE instead of to the
flux removes that component exactly rather than approximately, and the same fit is
simultaneously the test of the inference: if the residual collapses and the reachable
fraction rises, the rotational reading was right and the bounded-multiplier arm survives;
if it does not, the ansatz itself is what fails, which is evidence for the unbounded arm.

⚠️ **This is a PARTIAL constraint and must be reported as one.** Only the horizontal
divergence exists here. The coarse-grained products carry no ``w'T'``, so the vertical eddy
flux convergence is missing from ``∇·F`` and any kappa fitted against it absorbs that
omission. Say so wherever the numbers appear.

⚠️ **Average first, then differentiate.** Differentiating amplifies noise, and the six 5-day
marks in a month are effectively independent (5 days is much longer than the eddy
decorrelation time), so averaging them cuts the random rotational part by about sqrt(6) at no
cost in signal. Fit ``<F>`` against ``<∇σ>`` — never the mean of per-snapshot kappa-hats,
which is a different and worse estimator.

**Why it lives here and not in the library.** ``gm_redi.diff_part_hor_redi`` computes exactly
this divergence for the Redi flux, and it is verified against the C dumps, sits inside the
differentiated step and carries every production run. It is not to be refactored to expose the
geometry (an earlier split was made, tested green and reverted on that instruction). So the
geometry is re-derived here, and :func:`check_equivalence` proves the copy agrees with the
original to the last bit before any science is done with it — the same discipline as
``freeze_equivalence.py``.

⚠️ **The part a fresh copy gets wrong is the partial-cell branching.** An edge's two adjacent
elements need not both reach layer ``nz``. The C's five cases collapse to three by
level-membership — el1 only, el2 only, both — and ``CX``, ``CY`` and ``dz`` are each DIFFERENT
in the three. Get it wrong and nothing fails in the open ocean; it fails only near topography,
which is where the kappa guard already rejects nearly everything, so the damage would read as
more of the same rather than as a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fesom_jax  # noqa: E402,F401  (enables x64; must precede the jax imports)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

if ROOT not in Path(fesom_jax.__file__).resolve().parents:
    raise SystemExit(f"FATAL: fesom_jax from {fesom_jax.__file__}, not {ROOT}")

from fesom_jax import ops  # noqa: E402
from fesom_jax.mesh import Mesh  # noqa: E402

__all__ = ["edge_geometry", "elem_to_edge", "node_to_edge", "divergence_integral",
           "divergence_from_edge_flux", "divergence_of_elem_flux",
           "divergence_of_node_flux", "check_equivalence"]


def edge_geometry(mesh: Mesh, helem):
    """The median-dual control-volume geometry per (edge, layer) — the delicate part.

    The two vectors in ``mesh.edge_cross_dxdy`` run from the edge midpoint to the two adjacent
    element centroids, so they are the two boundary segments of the median-dual control volume
    around each endpoint. The flux through them is ``c = (CX·Fx + CY·Fy)·dz``, and ``+c`` goes
    to one endpoint while ``-c`` goes to the other, which makes the operator antisymmetric and
    therefore globally conservative by construction, not by cancellation.

    Returns ``(CX, CY, dz, live)``, each ``(edge2D, nl-1)``; ``live`` is the mask of
    (edge, layer) pairs that carry a flux at all.
    """
    nl, nl1 = mesh.nl, mesh.nl - 1
    el1, el2 = mesh.edge_tri[:, 0], mesh.edge_tri[:, 1]
    has1, has2 = el1 >= 0, el2 >= 0
    el1s, el2s = jnp.where(has1, el1, 0), jnp.where(has2, el2, 0)

    cross = mesh.edge_cross_dxdy
    dxL, dyL = cross[:, 0:1], cross[:, 1:2]
    dxR = jnp.where(has2[:, None], cross[:, 2:3], 0.0)
    dyR = jnp.where(has2[:, None], cross[:, 3:4], 0.0)

    h1, h2 = helem[el1s][:, :nl1], helem[el2s][:, :nl1]
    in1 = mesh.elem_layer_mask[el1s][:, :nl1] & has1[:, None]
    in2 = mesh.elem_layer_mask[el2s][:, :nl1] & has2[:, None]
    both, only1, only2 = in1 & in2, in1 & ~in2, in2 & ~in1

    CX = jnp.where(both, dyL - dyR, jnp.where(only1, dyL, -dyR))
    CY = jnp.where(both, dxR - dxL, jnp.where(only1, -dxL, dxR))
    dz = jnp.where(both, 0.5 * (h1 + h2), jnp.where(only1, h1, h2))
    live = has1[:, None] & (both | only1 | only2)
    return CX, CY, dz, live


def elem_to_edge(mesh: Mesh, fe):
    """Combine an element field ``(E, nl, …)`` onto edges ``(edge2D, nl-1, …)``.

    The SAME three-case rule the geometry uses: the mean of the two adjacent elements where
    both reach the layer, otherwise whichever one does. Applying a different rule here than in
    :func:`edge_geometry` is the way to get a plausible, conservative, wrong answer.
    """
    nl1 = mesh.nl - 1
    el1, el2 = mesh.edge_tri[:, 0], mesh.edge_tri[:, 1]
    has1, has2 = el1 >= 0, el2 >= 0
    el1s, el2s = jnp.where(has1, el1, 0), jnp.where(has2, el2, 0)
    f1, f2 = fe[el1s][:, :nl1], fe[el2s][:, :nl1]
    in1 = mesh.elem_layer_mask[el1s][:, :nl1] & has1[:, None]
    in2 = mesh.elem_layer_mask[el2s][:, :nl1] & has2[:, None]
    both, only1 = in1 & in2, in1 & ~in2
    if f1.ndim == 3:                      # trailing component axis
        both, only1 = both[..., None], only1[..., None]
    return jnp.where(both, 0.5 * (f1 + f2), jnp.where(only1, f1, f2))


def divergence_integral(mesh: Mesh, Fx_edge, Fy_edge, helem):
    """``∫∇·F dV`` over each node's dual cell, un-normalised ``(N, nl)``.

    This is the whole geometric content of the operator; everything after it is a division by
    the control volume. Keeping it separate is what lets :func:`check_equivalence` compare
    against ``gm_redi.diff_part_hor_redi`` through the SAME final arithmetic and demand exact
    equality — normalising first and un-normalising afterwards reassociates one multiply and a
    divide, which alone costs a few units in the last place and would force a round-off bar
    where none is needed.
    """
    CX, CY, dz, live = edge_geometry(mesh, helem)
    c = jnp.where(live, (CX * Fx_edge + CY * Fy_edge) * dz, 0.0)
    out = ops.scatter_add(jnp.stack([c, -c], axis=1), mesh.edges, mesh.nod2D)
    return jnp.pad(out, ((0, 0), (0, 1)))


def divergence_from_edge_flux(mesh: Mesh, Fx_edge, Fy_edge, hnode, helem):
    """``∇·F`` per node and layer ``(N, nl)`` from an EDGE-based horizontal flux.

    ``Fx_edge``/``Fy_edge`` are ``(edge2D, nl-1)``. Positive means net OUTFLOW from the node's
    dual cell, the ordinary mathematical sign.

    ⚠️ **The sign was wrong here at first, and the equivalence test could not see it.** The
    scattered edge sum is already ``+∫∇·F dV``, not the convergence, so negating it returned
    ``-∇·F``. It reads like a convergence because ``diff_part_hor_redi`` adds its result to the
    tracer — but the flux that routine builds is ``Kh(∇T + S ∂_zT)``, i.e. **plus** ``K∇T``, the
    negative of the physical down-gradient flux, so its tendency really is ``+∇·F``. And
    :func:`check_equivalence` runs through :func:`divergence_integral`, which is common to both
    sides, so it passed either way. What caught it was the constant-flux case measured against
    the analytic spherical metric term: correlation 0.99998 with slope **-0.9975**.
    """
    out = divergence_integral(mesh, Fx_edge, Fy_edge, helem)
    denom = mesh.areasvol * hnode
    safe = jnp.where(denom > 0.0, denom, 1.0)
    return jnp.where(mesh.node_layer_mask, out / safe, 0.0)


def divergence_of_elem_flux(mesh: Mesh, Fx_elem, Fy_elem, hnode, helem):
    """``∇·F`` from an ELEMENT-based flux ``(E, nl)``."""
    return divergence_from_edge_flux(mesh,
                                     elem_to_edge(mesh, Fx_elem),
                                     elem_to_edge(mesh, Fy_elem),
                                     hnode, helem)


def node_to_edge(mesh: Mesh, fn):
    """Average a NODE field ``(N, …)`` onto edges ``(edge2D, nl-1, …)``.

    The control-volume boundary passes through the edge midpoint, and the mean of the two
    endpoints is the natural estimate there — it is exactly what ``diff_part_hor_redi`` does
    with the nodal diffusivity, ``Kh = 0.5*(Ki[e1] + Ki[e2])``. Note this is a DIFFERENT rule
    from :func:`elem_to_edge`, which combines the two adjacent elements by level membership;
    using one where the other belongs produces a plausible, conservative, wrong answer.
    """
    nl1 = mesh.nl - 1
    e1, e2 = mesh.edges[:, 0], mesh.edges[:, 1]
    return 0.5 * (fn[e1][:, :nl1] + fn[e2][:, :nl1])


def divergence_of_node_flux(mesh: Mesh, Fx_node, Fy_node, hnode, helem):
    """``∇·F`` from a NODE-based flux — the form the coarse-grained teacher products take.

    ``coarse_ng5_YYYY_MM.nc`` carries ``uT_eddy``/``vT_eddy`` (and the salt pair) on ``nod2``
    with ``nz1 = nl-1`` layers, because the coarse-graining lands everything on nodes. So this,
    not :func:`divergence_of_elem_flux`, is the entry point for the M3b fit.
    """
    return divergence_from_edge_flux(mesh,
                                     node_to_edge(mesh, Fx_node),
                                     node_to_edge(mesh, Fy_node),
                                     hnode, helem)


def check_equivalence(mesh: Mesh, T_old, slope_tapered, Ki, hnode, hnode_new, helem,
                      *, dt: float):
    """Feed this operator the Redi flux and demand it reproduces ``diff_part_hor_redi``.

    The reference is dump-verified against the C and carries every production run, so it is the
    right oracle — and the point of the exercise is that a re-derived copy of delicate geometry
    must prove itself rather than be believed. Returns ``(max_abs_difference, scale)``; the
    house precedent (``freeze_equivalence.py``) is a max difference of exactly 0, and anything
    else means the partial-cell branching differs.
    """
    from fesom_jax import gm_redi

    nl, nl1 = mesh.nl, mesh.nl - 1
    txy = gm_redi.tr_xy_elem(mesh, T_old)

    # ∂T/∂z at interfaces on the OLD mesh, exactly as the reference builds it.
    dz_node = 0.5 * (gm_redi._shift_down(hnode) + hnode)
    safe_dz = jnp.where(dz_node > 0.0, dz_node, 1.0)
    k = jnp.arange(nl)[None, :]
    trz_mask = (k >= mesh.ulevels_nod2D[:, None]) & (k < (mesh.nlevels_nod2D - 1)[:, None])
    trz = jnp.where(trz_mask, (gm_redi._shift_down(T_old) - T_old) / safe_dz, 0.0)

    e1, e2 = mesh.edges[:, 0], mesh.edges[:, 1]
    Kh = 0.5 * (Ki[e1][:, :nl1] + Ki[e2][:, :nl1])
    Tz1 = 0.5 * (trz[e1][:, :nl1] + trz[e1][:, 1:nl])
    Tz2 = 0.5 * (trz[e2][:, :nl1] + trz[e2][:, 1:nl])
    st1, st2 = slope_tapered[e1], slope_tapered[e2]
    SxTz = 0.5 * (Tz1 * st1[:, :, 0] + Tz2 * st2[:, :, 0])
    SyTz = 0.5 * (Tz1 * st1[:, :, 1] + Tz2 * st2[:, :, 1])

    # The element gradient reaches the edge through the SAME three-case rule, which is what
    # `elem_to_edge` implements — so this call is also a test of that helper, not just of the
    # geometry. Kh and the slope terms are already edge quantities.
    Tx_e = elem_to_edge(mesh, txy[:, :, 0:1])[:, :, 0]
    Ty_e = elem_to_edge(mesh, txy[:, :, 1:2])[:, :, 0]
    Fx, Fy = Kh * (Tx_e + SxTz), Kh * (Ty_e + SyTz)

    # Rebuild the reference's OWN last line from this operator's convergence — `rhs * dt / safe`,
    # in that order. Going through `divergence_from_edge_flux` and multiplying back by dt would
    # be mathematically the same and bit-wise different (measured: 6.9e-18 on a field of scale
    # 8.4e-02, i.e. a third of an ulp), which would turn an exact identity into a judgement call
    # about round-off. The geometry is what is under test; the normalisation is not.
    ref = gm_redi.diff_part_hor_redi(mesh, T_old, slope_tapered, Ki, hnode, hnode_new,
                                     helem, dt=dt)
    integ = divergence_integral(mesh, Fx, Fy, helem)
    denom = mesh.areasvol * hnode_new
    safe = jnp.where(denom > 0.0, denom, 1.0)
    mine = jnp.where(mesh.node_layer_mask, integ * dt / safe, 0.0)
    return float(jnp.max(jnp.abs(ref - mine))), float(jnp.max(jnp.abs(ref)))
