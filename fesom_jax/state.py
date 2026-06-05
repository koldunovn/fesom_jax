"""Evolving model state for the FESOM2 → JAX port (Task 1.2).

:class:`State` is a flat, frozen dataclass registered as a JAX **pytree**, so it
flows through ``jax.lax.scan`` / ``jax.grad`` as a single structured value. Every
field is a dense array sized to the global mesh (``nl=48`` for pi); ragged
vertical extent is handled by the masks in :mod:`fesom_jax.mesh`, not by ragged
storage.

Layout mirrors the C port's ``fesom_dyn`` (``fesom_dyn.h``) and ``fesom_aux``
(``fesom_aux.h``) structs and the thickness fields on ``fesom_mesh``. Each field
notes its C owner, entity, and whether it is a **layer** quantity (valid
``[ulevels-1, nlevels-1)``) or an **interface** quantity (valid
``[ulevels-1, nlevels-1]``). Velocity vectors are ``[..., 2]`` (u,v), matching
C macro ``FESOM_ELEMVEC``.

This phase only *defines and sizes* the state and provides constructors; the
individual fields are populated as the substep kernels are ported in Phase 2.
The exact bit-level rest initialization is a Phase-2 (Task 2.11) gate; here
:meth:`State.rest` gives a clean, physically-consistent starting point.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import tree_util

from .mesh import Mesh


@dataclasses.dataclass(frozen=True)
class State:
    # --- tracers (node, layer) — fesom_tracers ---
    T: jax.Array            # (nod2D,nl) potential temperature [°C]
    S: jax.Array            # (nod2D,nl) salinity [psu]
    T_old: jax.Array        # (nod2D,nl) previous-step T (AB2 single slot, valuesold)
    S_old: jax.Array        # (nod2D,nl) previous-step S
    del_ttf: jax.Array      # (nod2D,nl) per-step tracer increment accumulator

    # --- momentum (element layer, + node interpolation) — fesom_dyn ---
    uv: jax.Array           # (elem2D,nl,2) horizontal velocity (u,v)
    uv_rhs: jax.Array       # (elem2D,nl,2) momentum RHS
    uv_rhsAB: jax.Array     # (elem2D,nl,2) AB2 RHS history
    uvnode: jax.Array       # (nod2D,nl,2) u,v interpolated to nodes (PP mixing)
    uvnode_rhs: jax.Array   # (nod2D,nl,2) momentum-advection RHS at nodes (momadv_opt=2)

    # --- vertical velocity (node, interface) — fesom_dyn ---
    w: jax.Array            # (nod2D,nl) vertical velocity at interfaces
    w_e: jax.Array          # (nod2D,nl) explicit part (tracer advection)
    w_i: jax.Array          # (nod2D,nl) implicit part (impl_vert_visc)
    cfl_z: jax.Array        # (nod2D,nl) vertical CFL at interfaces

    # --- SSH (node, 2D) — fesom_dyn ---
    eta_n: jax.Array        # (nod2D,) surface elevation η
    d_eta: jax.Array        # (nod2D,) SSH increment (CG solve output)
    ssh_rhs: jax.Array      # (nod2D,) SSH equation RHS
    ssh_rhs_old: jax.Array  # (nod2D,) previous-step ssh_rhs

    # --- ALE thickness — fesom_mesh ---
    hnode: jax.Array        # (nod2D,nl) layer thickness at vertices (layer)
    hnode_new: jax.Array    # (nod2D,nl) thickness after SSH update
    helem: jax.Array        # (elem2D,nl) layer thickness at cells (layer)
    hbar: jax.Array         # (nod2D,) thickness-weighted SSH proxy
    hbar_old: jax.Array     # (nod2D,) previous-step hbar

    # --- EOS / pressure / mixing — fesom_aux ---
    density: jax.Array      # (nod2D,nl) in-situ density − ρ0 (density_m_rho0, layer)
    hpressure: jax.Array    # (nod2D,nl) hydrostatic pressure (layer)
    bvfreq: jax.Array       # (nod2D,nl) Brunt-Väisälä N² (interface)
    Kv: jax.Array           # (nod2D,nl) tracer vertical diffusivity (interface)
    Av: jax.Array           # (elem2D,nl) momentum vertical viscosity (elem interface)
    pgf_x: jax.Array        # (elem2D,nl) pressure-gradient force x (layer)
    pgf_y: jax.Array        # (elem2D,nl) pressure-gradient force y (layer)

    # ---- constructors -----------------------------------------------------
    @classmethod
    def zeros(cls, mesh: Mesh) -> "State":
        """All-zero state with every field correctly shaped & float64."""
        n, e, nl = mesh.nod2D, mesh.elem2D, mesh.nl
        f = jnp.float64

        def Z(*shape):
            return jnp.zeros(shape, f)

        return cls(
            T=Z(n, nl), S=Z(n, nl), T_old=Z(n, nl), S_old=Z(n, nl), del_ttf=Z(n, nl),
            uv=Z(e, nl, 2), uv_rhs=Z(e, nl, 2), uv_rhsAB=Z(e, nl, 2),
            uvnode=Z(n, nl, 2), uvnode_rhs=Z(n, nl, 2),
            w=Z(n, nl), w_e=Z(n, nl), w_i=Z(n, nl), cfl_z=Z(n, nl),
            eta_n=Z(n), d_eta=Z(n), ssh_rhs=Z(n), ssh_rhs_old=Z(n),
            hnode=Z(n, nl), hnode_new=Z(n, nl), helem=Z(e, nl), hbar=Z(n), hbar_old=Z(n),
            density=Z(n, nl), hpressure=Z(n, nl), bvfreq=Z(n, nl), Kv=Z(n, nl),
            Av=Z(e, nl), pgf_x=Z(e, nl), pgf_y=Z(e, nl),
        )

    @classmethod
    def rest(cls, mesh: Mesh, T0: float = 10.0, S0: float = 35.0) -> "State":
        """Rest state: zero flow/SSH, constant ``T=T0``/``S=S0``, reference layer
        thicknesses (``zbar`` differences). The exact C-matching rest init is a
        Phase-2 gate; this is a clean physical starting point."""
        st = cls.zeros(mesh)
        n, e, nl = mesh.nod2D, mesh.elem2D, mesh.nl
        T = jnp.full((n, nl), float(T0), jnp.float64)
        S = jnp.full((n, nl), float(S0), jnp.float64)

        # reference layer thickness h[k] = z(k) - z(k+1) > 0, masked to valid layers
        z_n = mesh.zbar_3d_n                                   # (n, nl)
        dz_n = jnp.zeros((n, nl)).at[:, :-1].set(z_n[:, :-1] - z_n[:, 1:])
        hnode = jnp.where(mesh.node_layer_mask, dz_n, 0.0)

        zbar = mesh.zbar                                       # (nl,)
        dz = jnp.zeros((nl,)).at[:-1].set(zbar[:-1] - zbar[1:])
        helem = jnp.where(mesh.elem_layer_mask, dz[None, :], 0.0)

        return dataclasses.replace(
            st, T=T, S=S, T_old=T, S_old=S, hnode=hnode, hnode_new=hnode, helem=helem
        )


# Register State as a pytree: every field is a data leaf (no static metadata).
_STATE_FIELDS = [f.name for f in dataclasses.fields(State)]
tree_util.register_dataclass(State, data_fields=_STATE_FIELDS, meta_fields=[])
