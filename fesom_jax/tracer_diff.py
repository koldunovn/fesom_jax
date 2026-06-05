"""Implicit vertical tracer diffusion — substep 15 (diffusion part), Task 2.10.

Literal vectorized port of ``diff_ver_part_impl_ale`` (``fesom_tracer_diff.c:85``)
for the Phase-2 pi config. A per-**node** implicit tridiagonal solve (1 unknown per
node-column, vs :func:`~fesom_jax.momentum.impl_vert_visc`'s per-element 2-unknown),
using the node vertical diffusivity ``Kv`` from substep 4.

Phase-2 simplifications (all verified against the dump config):

* **No GM/Redi** (``gm=NULL``) ⇒ the isoneutral ``K33`` augmentations ``Ty/Ty1`` ≡ 0.
* **No implicit vertical advection** (``do_wimpl=0`` — the C sets it false for FCT
  *and* ``use_wsplit=0``).
* **No surface flux** (``bc_surface=0``): analytical forcing has zero
  heat/water/virtual-salt/relax-salt flux.
* **No shortwave penetration**: ``USE_SW_PENE`` is gated on ``use_jra`` and the dump
  uses analytical forcing ⇒ ``sw_3d=0``.
* **Full-cell linfs** ⇒ ``zbar_n=zbar``, ``Z_n=Z`` (static mid-layer depths), exactly
  as :func:`~fesom_jax.momentum.impl_vert_visc`.

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


def impl_vert_diff_one(mesh: Mesh, T, Kv, hnode_new, *, dt: float = DT_DEFAULT):
    """Per-node implicit vertical diffusion of one tracer. Returns the updated
    tracer ``[nod2D, nl]`` (``diff_ver_part_impl_ale``, ``fesom_tracer_diff.c:85``)."""
    nl = mesh.nl
    Zp = jnp.concatenate([mesh.Z, mesh.Z[-1:]])           # (nl,) static mid-depths
    # Layer-center spacings. The padded tail (and the unused k=0 of dZ_up) is exactly
    # 0; replace those zeros with 1 so the masked-off a/c lanes are FINITE — a bare 0
    # denominator yields inf/NaN that the `where` masks forward but that POISONS the
    # backward pass (0·inf = NaN in d/d(Kv); cf. the eos unused-N²-level lesson).
    dZ_up = _shift_down(Zp) - Zp                          # Z[nz-1]-Z[nz]; [0]=0 unused (a=0)
    dZ_dn = Zp - _shift_up(Zp)                            # Z[nz]-Z[nz+1]; tail=0 unused
    dZ_up = jnp.where(dZ_up == 0.0, 1.0, dZ_up)
    dZ_dn = jnp.where(dZ_dn == 0.0, 1.0, dZ_dn)

    area = mesh.area
    safe_av = jnp.where(mesh.areasvol > 0.0, mesh.areasvol, 1.0)

    a_full = -Kv * dt / dZ_up[None, :] * (area / safe_av)
    c_full = -_shift_up(Kv) * dt / dZ_dn[None, :] * (_shift_up(area) / safe_av)

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

    dT = ops.tdma(a, b, c, rhs)
    return T + ops.mask_below_bottom(dT, valid)


def impl_vert_diff(mesh: Mesh, T, S, Kv, hnode_new, *, dt: float = DT_DEFAULT):
    """Diffuse both tracers with the same ``Kv`` (``fesom_impl_vert_diff_tracers``,
    ``fesom_tracer_diff.c:338``). Returns ``(T_new, S_new)``."""
    return (impl_vert_diff_one(mesh, T, Kv, hnode_new, dt=dt),
            impl_vert_diff_one(mesh, S, Kv, hnode_new, dt=dt))
