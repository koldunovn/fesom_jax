"""Upwind tracer advection — substep 15 (advection part), Task 2.10.

Literal vectorized port of the **upwind (non-FCT)** tracer-advection path of
``fesom_tracer_adv.c`` for the Phase-2 pi config (linfs ALE, ``use_wsplit=0`` ⇒
``w_e = w``, single MPI rank). Drives one tracer per call, mirroring
``fesom_tracer_advect_one`` (``fesom_tracer_adv.c:1269``):

1. :func:`tracer_ab` — AB2 extrapolation ``ttfAB = -(0.5+ε)·valuesold +
   (1.5+ε)·values`` and the ``valuesold := values`` save (``init_tracers_AB_one``,
   ``:174``; ε=0.1). The fluxes use ``ttfAB``; the ALE reconstruction uses
   ``values``. At **step 1** ``valuesold == values`` ⇒ ``ttfAB == values``.
2. :func:`adv_flux_hor` — horizontal upwind edge flux (``adv_tra_hor_upw1``,
   ``:212``): the volume flux ``vflux`` across each edge (summed over the two
   adjacent cells, each masked to its own layer range — the C's 5 level "zones"
   collapse to one masked sum) drives the upwind face value
   ``-½(T₁(vflux+|vflux|) + T₂(vflux−|vflux|))``.
3. :func:`adv_flux_ver` — vertical upwind flux (``adv_tra_ver_upw1``, ``:701``)
   using ``w_e``: ``-½(T[nz](w+|w|) + T[nz−1](w−|w|))·area``; the edge-replicated
   ``T[nz−1]`` reproduces the C's surface term ``-w·T·area`` at ``nzmin`` and the
   no-flux bottom ``flux_v[nzmax]=0`` falls out of the layer mask.
4. :func:`flux2dtracer` — divergence assembly (``flux2dtracer_upwind``, ``:740``):
   vertical ``(flux_v[nz]−flux_v[nz+1])·dt/areasvol`` + the antisymmetric edge→node
   horizontal scatter ``±flux_h·dt/areasvol``.
5. :func:`ale_reconstruct` — ``del_ttf += T·(hnode−hnode_new)`` (≡0 in linfs) then
   ``T += del_ttf/hnode_new`` (``ale_reconstruct``, ``:792``).

⚠️ The C dump runs **FCT**, this runs **upwind**. ``S=35`` is horizontally constant
⇒ advects trivially (upwind == FCT == the dump, a clean step-1 gate). ``T`` has the
Gaussian blob ⇒ upwind ≠ FCT in the curved region, so the step-1 ``T`` dump match is
a Phase-4 (FCT) gate; here ``T`` is verified against an independent numpy upwind
reference + a constant-tracer property. (See ``docs/REFERENCE_RUNS.md``.)
"""

from __future__ import annotations

import jax.numpy as jnp

from . import ops
from .config import DT_DEFAULT
from .mesh import Mesh

# AB2 stabilization offset (ε=0.1, oce_modules.F90:92 / fesom_tracer_adv.c:185-187).
_EPS_AB = 0.1
_AB_OLD = -(0.5 + _EPS_AB)   # valuesold coefficient
_AB_NEW = (1.5 + _EPS_AB)    # values   coefficient


def _shift_down(x):
    """``out[...,k] = x[...,k-1]``, edge-replicated at k=0 (surface)."""
    return jnp.concatenate([x[..., :1], x[..., :-1]], axis=-1)


def _shift_up(x):
    """``out[...,k] = x[...,k+1]``, zero-padded at the last level."""
    return jnp.concatenate([x[..., 1:], jnp.zeros_like(x[..., :1])], axis=-1)


def tracer_ab(values, valuesold):
    """AB2 extrapolation + the ``valuesold := values`` save (``init_tracers_AB_one``,
    ``fesom_tracer_adv.c:174``). Returns ``(ttfAB, valuesold_new)`` where
    ``ttfAB = -(0.5+ε)·valuesold + (1.5+ε)·values`` and ``valuesold_new = values``.
    At step 1 ``valuesold == values`` (``ic.initial_state`` sets ``T_old=T``) so
    ``ttfAB == values``."""
    ttfAB = _AB_OLD * valuesold + _AB_NEW * values
    return ttfAB, values


def adv_flux_hor(mesh: Mesh, uv, helem, ttf):
    """Horizontal upwind flux ``[edge2D, nl]`` (``adv_tra_hor_upw1``,
    ``fesom_tracer_adv.c:212``).

    Per edge the volume flux is the sum of each adjacent cell's contribution,
    masked to that cell's layer range — the C's 5 explicit level-zones (el1-only
    above / el2-only above / both / el1-only below / el2-only below) are exactly
    ``vflux = mask₁·flux₁ + mask₂·flux₂`` per level. ``el1`` uses
    ``(u·dy₁ − v·dx₁)·h`` and ``el2`` uses ``(v·dx₂ − u·dy₂)·h`` (the C sign
    convention). The upwind face value is ``-½(T₁(vflux+|vflux|) +
    T₂(vflux−|vflux|))`` (T₁ if ``vflux>0`` flows n1→n2, else T₂)."""
    et = mesh.edge_tri
    el1, el2 = et[:, 0], et[:, 1]
    has1, has2 = el1 >= 0, el2 >= 0
    el1s = jnp.where(has1, el1, 0)
    el2s = jnp.where(has2, el2, 0)
    cross = mesh.edge_cross_dxdy
    dx1, dy1 = cross[:, 0:1], cross[:, 1:2]
    dx2, dy2 = cross[:, 2:3], cross[:, 3:4]
    U, V = uv[:, :, 0], uv[:, :, 1]
    lm = mesh.elem_layer_mask
    t1 = jnp.where(lm[el1s] & has1[:, None],
                   (U[el1s] * dy1 - V[el1s] * dx1) * helem[el1s], 0.0)
    t2 = jnp.where(lm[el2s] & has2[:, None],
                   (V[el2s] * dx2 - U[el2s] * dy2) * helem[el2s], 0.0)
    vflux = t1 + t2                                       # (edge, nl)
    T1 = ttf[mesh.edges[:, 0]]                            # (edge, nl)
    T2 = ttf[mesh.edges[:, 1]]
    av = jnp.abs(vflux)
    return -0.5 * (T1 * (vflux + av) + T2 * (vflux - av))


def adv_flux_ver(mesh: Mesh, w_e, ttf):
    """Vertical upwind flux ``[nod2D, nl]`` at interfaces (``adv_tra_ver_upw1``,
    ``fesom_tracer_adv.c:701``), using ``W = w_e``. The unified interior formula
    ``-½(T[nz](w+|w|) + T[nz−1](w−|w|))·area`` with the edge-replicated ``T[nz−1]``
    reproduces the C's surface flux ``-W·T·area`` at ``nzmin`` (½·2w·T = w·T); the
    bottom interface ``flux_v[nzmax]=0`` falls out of the layer mask (support
    ``[nzmin, nzmax)``)."""
    w = w_e
    aw = jnp.abs(w)
    T_below = ttf                                         # T[nz]
    T_above = _shift_down(ttf)                            # T[nz-1] (→ T[nzmin] at top)
    flux = -0.5 * (T_below * (w + aw) + T_above * (w - aw)) * mesh.area
    return ops.mask_below_bottom(flux, mesh.node_layer_mask)


def flux2dtracer(mesh: Mesh, flux_h, flux_v, *, dt: float = DT_DEFAULT):
    """Assemble ``del_ttf`` ``[nod2D, nl]`` from the horizontal + vertical fluxes
    (``flux2dtracer_upwind``, ``fesom_tracer_adv.c:740``).

    Vertical divergence ``(flux_v[nz] − flux_v[nz+1])·dt/areasvol[nz]``; horizontal
    antisymmetric edge→node scatter ``flux_h → +n1 / −n2`` then ``·dt/areasvol``
    (the per-receiver ``÷areasvol`` factors out of the edge scatter since it depends
    only on the node+level). The C's ``if(area>0)`` guard → a safe divide + the
    layer mask."""
    areasvol = mesh.areasvol
    safe = jnp.where(areasvol > 0.0, areasvol, 1.0)
    lm = mesh.node_layer_mask
    dttf_v = jnp.where(lm, (flux_v - _shift_up(flux_v)) * dt / safe, 0.0)
    vals = jnp.stack([flux_h, -flux_h], axis=1)          # (edge, 2, nl)
    raw = ops.scatter_add(vals, mesh.edges, mesh.nod2D)  # (nod2D, nl)
    dttf_h = jnp.where(lm, raw * dt / safe, 0.0)
    return dttf_h + dttf_v


def ale_reconstruct(mesh: Mesh, T, del_ttf, hnode, hnode_new):
    """ALE tracer reconstruction (``ale_reconstruct``, ``fesom_tracer_adv.c:792``):
    ``del_ttf += T·(hnode − hnode_new)`` then ``T += del_ttf/hnode_new`` on valid
    layers with ``hnode_new>0``. In **linfs** ``hnode == hnode_new`` ⇒ the thickness
    term vanishes and ``T_new = T + del_ttf/hnode``."""
    del_ttf = del_ttf + T * (hnode - hnode_new)
    safe_hn = jnp.where(hnode_new > 0.0, hnode_new, 1.0)
    upd = mesh.node_layer_mask & (hnode_new > 0.0)
    dT = jnp.where(upd, del_ttf / safe_hn, 0.0)
    return T + dT


def advect_one(mesh: Mesh, uv, w_e, helem, hnode, hnode_new, T, T_old,
               *, dt: float = DT_DEFAULT):
    """One tracer's upwind advection + ALE reconstruction
    (``fesom_tracer_advect_one``, ``fesom_tracer_adv.c:1269``). Returns
    ``(T_new, T_old_new)`` with ``T_old_new = T`` (the AB2 ``valuesold`` save). The
    fluxes use the AB2-extrapolated ``ttfAB``; the reconstruction updates ``T``."""
    ttfAB, T_old_new = tracer_ab(T, T_old)
    flux_h = adv_flux_hor(mesh, uv, helem, ttfAB)
    flux_v = adv_flux_ver(mesh, w_e, ttfAB)
    del_ttf = flux2dtracer(mesh, flux_h, flux_v, dt=dt)
    T_new = ale_reconstruct(mesh, T, del_ttf, hnode, hnode_new)
    return T_new, T_old_new
