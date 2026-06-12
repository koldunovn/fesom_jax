"""Implicit vertical tracer diffusion — substep 15 (diffusion part), Task 2.10.

Literal vectorized port of ``diff_ver_part_impl_ale`` (``fesom_tracer_diff.c:85``)
for the Phase-2 pi config. A per-**node** implicit tridiagonal solve (1 unknown per
node-column, vs :func:`~fesom_jax.momentum.impl_vert_visc`'s per-element 2-unknown),
using the node vertical diffusivity ``Kv`` from substep 4.

Phase-2 simplifications (all verified against the dump config):

* **No GM/Redi** (``gm=NULL``) ⇒ the isoneutral ``K33`` augmentations ``Ty/Ty1`` ≡ 0.
* **No implicit vertical advection** (``do_wimpl=0`` — the C sets it false for FCT
  *and* ``use_wsplit=0``).
* **Full-cell linfs** ⇒ ``zbar_n=zbar``, ``Z_n=Z`` (static mid-layer depths), exactly
  as :func:`~fesom_jax.momentum.impl_vert_visc`.

Surface boundary condition (Phase 5, Task 5.6). The C adds a surface forcing
increment to the RHS at ``tr[nzmin]`` (``bc_surface``, ``fesom_tracer_diff.c:44-72``)
and, for the temperature tracer, a shortwave-penetration flux divergence per layer
(``:298-308``):

* ``bc_surf`` — a per-node ``[nod2D]`` increment added to the surface layer's RHS.
  For linfs: ``bc_T = −dt·heat_flux/vcpw`` (T) and ``bc_S = dt·(virtual_salt +
  relax_salt)`` (S); built by :mod:`fesom_jax.core2_forcing`. ``None`` ⇒ 0 (the pi
  analytical path: zero heat/water/virtual-salt/relax-salt flux).
* ``sw_3d`` — the per-node, per-layer shortwave flux ``[nod2D, nl]``
  (:func:`fesom_jax.forcing.cal_shortwave_rad`), consumed by the **T** tracer only as
  the divergence ``(sw[nz] − sw[nz+1]·area[nz+1]/areasvol[nz])·dt`` added to every
  valid layer's RHS. ``None`` ⇒ skip (the pi path: ``USE_SW_PENE`` is gated on
  ``use_jra``, off under analytical forcing ⇒ ``sw_3d=0``).

With ``bc_surf=None`` and ``sw_3d=None`` the result is **bit-identical** to the
Phase-2 path (the 313 pi gates must not move). ``sw_3d`` is a per-step forcing
*constant* (it depends only on the JRA shortwave + chl climatology + geometry, not on
the model state), so it introduces no new AD path; ``bc_surf`` carries the
differentiable SST→heat_flux / SST·S_top→virtual/relax_salt seam into the TDMA.

The tridiagonal (per node, layer ``nz``):

    a[nz] = -Kv[nz]   ·dt·area[nz]  /(areasvol[nz]·(Z[nz-1]-Z[nz]))   (0 at surface)
    c[nz] = -Kv[nz+1] ·dt·area[nz+1]/(areasvol[nz]·(Z[nz]-Z[nz+1]))   (0 at bottom)
    b[nz] = -a[nz] - c[nz] + hnode_new[nz]                            (mass + implicit)
    rhs[nz] = a[nz]·(T[nz]-T[nz-1]) + c[nz]·(T[nz]-T[nz+1])           (explicit op)

solved with :func:`~fesom_jax.ops.tdma`; the increment is added to ``T``. Padded
below-bottom with ``(b=1, a=c=rhs=0)`` so those rows solve to 0. A constant-in-z
tracer (e.g. ``S=35``) has zero ``rhs`` ⇒ stays put (verified).
"""

from __future__ import annotations

import jax.numpy as jnp

from . import ops
from .config import DT_DEFAULT
from .mesh import Mesh


def _shift_down(x):
    """``out[...,k] = x[...,k-1]``, edge-replicated at k=0."""
    return jnp.concatenate([x[..., :1], x[..., :-1]], axis=-1)


def _shift_up(x):
    """``out[...,k] = x[...,k+1]``, zero-padded at the last level."""
    return jnp.concatenate([x[..., 1:], jnp.zeros_like(x[..., :1])], axis=-1)


def impl_vert_diff_one(mesh: Mesh, T, Kv, hnode_new, *, dt: float = DT_DEFAULT,
                       bc_surf=None, sw_3d=None, Z3d=None):
    """Per-node implicit vertical diffusion of one tracer. Returns the updated
    tracer ``[nod2D, nl]`` (``diff_ver_part_impl_ale``, ``fesom_tracer_diff.c:85``).

    ``bc_surf`` (``[nod2D]`` or ``None``) is the surface forcing increment added to the
    surface layer's RHS (``bc_surface``, ``:290``). ``sw_3d`` (``[nod2D, nl]`` or
    ``None``) is the shortwave flux whose per-layer divergence is added to every valid
    layer's RHS (T tracer only, ``:298-308``). Both ``None`` ⇒ the Phase-2 path."""
    nl = mesh.nl
    # Layer-center spacings dZ. Static column-uniform ``mesh.Z`` or — under zstar (JZ.6) —
    # the per-node live mid-depths ``Z3d`` (the C builds zbar_n/Z_n from hnode_new,
    # ``tracer_diff.c:148-158``). ``Z3d=None`` ⇒ static, byte-identical.
    # The padded tail (and the unused k=0 of dZ_up) is exactly 0; replace those zeros with 1
    # so the masked-off a/c lanes are FINITE — a bare 0 denominator yields inf/NaN that the
    # `where` masks forward but POISONS the backward pass (0·inf = NaN in d/d(Kv)).
    if Z3d is None:
        Zp = jnp.concatenate([mesh.Z, mesh.Z[-1:]])       # (nl,) static mid-depths
        dZ_up = (_shift_down(Zp) - Zp)[None, :]           # Z[nz-1]-Z[nz]; [0]=0 unused (a=0)
        dZ_dn = (Zp - _shift_up(Zp))[None, :]             # Z[nz]-Z[nz+1]; tail=0 unused
    else:
        dZ_up = _shift_down(Z3d) - Z3d                    # (nod2D,nl) per-node live spacing
        dZ_dn = Z3d - _shift_up(Z3d)
    dZ_up = jnp.where(dZ_up == 0.0, 1.0, dZ_up)
    dZ_dn = jnp.where(dZ_dn == 0.0, 1.0, dZ_dn)

    area = mesh.area
    safe_av = jnp.where(mesh.areasvol > 0.0, mesh.areasvol, 1.0)

    a_full = -Kv * dt / dZ_up * (area / safe_av)
    c_full = -_shift_up(Kv) * dt / dZ_dn * (_shift_up(area) / safe_av)

    k = jnp.arange(nl)[None, :]
    nzmin = (mesh.ulevels_nod2D - 1)[:, None]
    nzmax = (mesh.nlevels_nod2D - 1)[:, None]
    valid = mesh.node_layer_mask
    surf = k == nzmin
    bot = k == (nzmax - 1)

    a = jnp.where(surf, 0.0, a_full)
    c = jnp.where(bot, 0.0, c_full)
    b = jnp.where(valid, -a - c + hnode_new, 1.0)
    a = jnp.where(valid, a, 0.0)
    c = jnp.where(valid, c, 0.0)

    # RHS = the explicit vertical-diffusion operator on T (the C's tr[nz]).
    rhs = jnp.where(valid, a * (T - _shift_down(T)) + c * (T - _shift_up(T)), 0.0)

    # Surface BC: add bc_surf to the surface layer's RHS (only on non-degenerate
    # columns — `surf & valid` excludes a single-interface node the C `continue`s).
    if bc_surf is not None:
        rhs = rhs + jnp.where(surf & valid, bc_surf[:, None], 0.0)

    # Shortwave penetration (T only): the per-layer flux divergence
    # (sw[nz] − sw[nz+1]·area[nz+1]/areasvol[nz])·dt, added to every valid layer.
    if sw_3d is not None:
        sw_div = (sw_3d - _shift_up(sw_3d) * (_shift_up(area) / safe_av)) * dt
        rhs = rhs + jnp.where(valid, sw_div, 0.0)

    dT = ops.tdma(a, b, c, rhs)
    return T + ops.mask_below_bottom(dT, valid)


def impl_vert_diff(mesh: Mesh, T, S, Kv, hnode_new, *, dt: float = DT_DEFAULT,
                   bc_T=None, bc_S=None, sw_3d=None, Z3d=None):
    """Diffuse both tracers with the same ``Kv`` (``fesom_impl_vert_diff_tracers``,
    ``fesom_tracer_diff.c:338``). Returns ``(T_new, S_new)``.

    ``bc_T``/``bc_S`` are the per-tracer surface BC increments (``None`` ⇒ 0); ``sw_3d``
    is the shortwave flux applied to **T only** (``None`` ⇒ off). All ``None`` ⇒ the
    Phase-2 pi path (bit-identical). ``Z3d`` (zstar live mid-depths from ``hnode_new``;
    ``None`` ⇒ static, byte-identical) re-points the layer-center spacings (JZ.6)."""
    return (impl_vert_diff_one(mesh, T, Kv, hnode_new, dt=dt, bc_surf=bc_T, sw_3d=sw_3d, Z3d=Z3d),
            impl_vert_diff_one(mesh, S, Kv, hnode_new, dt=dt, bc_surf=bc_S, sw_3d=None, Z3d=Z3d))
