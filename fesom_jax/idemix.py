"""IDEMIX internal-wave mixing driver — node-placed, over the ``idemix_jax`` column core.

The analogue of FESOM2's ``gen_modules_cvmix_idemix.F90`` driver (``calc_cvmix_idemix``):
the IDEMIX 1.0 closure (Olbers & Eden 2013) advances the internal-wave energy ``iwe`` per
column with an implicit vertical solve, propagates it laterally between columns, and hands
the wave dissipation ``iw_diss`` (+ ``alpha_c``/``E_iw``) to the TKE closure
(:func:`fesom_jax.tke.mixing_tke` ``idemix=``), or — IDEMIX-only mixing, FESOM
``mix_scheme_nmb==6`` — its Osborn-shortcut ``KappaH``/``KappaM`` to ``Kv``/``Av``
(:func:`osborn_kv_av`). The physics lives in the external, separately verified
``idemix_jax`` package (``idemix_step_columns``, ``horizontal_propagation``,
``neighbors_from_edges``, ``masks_from_levels``); this module is only the host wiring
(design: idemix-jax ``docs/SPEC_STAGE3.md`` §6.3 step seam, §6.5 Osborn path, §6.6
connectivity).

**Three rulings** (deliberate, documented deviations from the Fortran driver):

1. **Node placement.** IDEMIX runs on NODES, where TKE lives — not on elements followed by
   elem→node averaging as the Fortran does (``F90:365-455`` element loop, ``:462-491``
   averaging). The column core is placement-agnostic; node placement removes two averaging
   passes and a halo exchange, and the Fortran's own TKE consumes node-averaged fields
   anyway (SPEC_STAGE3 §6.3). Per-node inputs: the wet interfaces
   ``ulevels_nod2D-1 .. nlevels_nod2D-1`` (0-based), ``dzw = hnode[:, :nl-1]``,
   ``dzt = tke.interface_spacing`` (the TKE's ``dz_trr``), ``Nsqr = bvfreq``,
   ``coriolis = mesh.coriolis_node``.
2. **Propagation after the column solve.** As in the Fortran (which propagates in place on
   ``iwe_n`` after ``integrate_idemix``), the lateral step acts on the column-step energy
   ``E_iw``: ``iwe_new = E_iw + dt * tend``, where ``tend`` is
   ``horizontal_propagation``'s return value — a tendency in m²/s³ defined as
   ``(E_after - E_iw) / dt`` — so the composition recovers ``E_after`` (to roundoff).
   ``iw_diss``/``alpha_c``/``KappaH``/``KappaM``/``E_iw`` are the COLUMN-step fields (the
   Fortran's TKE consumes ``iwe_Tdis`` from ``integrate_idemix``, before propagation).
3. **v0 CFL cap.** The lateral sub-stepping is explicit; the horizontal group velocity is
   capped at ``v0 <= sqrt(cap * res**2 / (tau_h * dt / n_hor_iter))`` (pyOM2
   ``idemix.f90:85-99`` with a 2x margin at the default ``cap = 0.1``), ``res`` the per-node
   ``mesh.mesh_resolution`` (Voronoi diameter, m). ``lateral_cfl_cap = 0`` disables it.

Lateral operator (:func:`build_lateral_operator`, host-side NumPy, once per mesh): the
two-point flux weight of edge ``(a, b)`` is ``w = cross_len / node_dist`` — the cross-edge
length (the two ``edge_cross_dxdy`` segments, m; one is zero on a boundary edge) over the
great-circle distance between the edge's nodes (``geo_coord_nod2D``, radius
:data:`fesom_jax.config.R_EARTH`). Cell volumes are ``area_if * dzt`` with the PER-INTERFACE
area :func:`interface_area` built from the per-LAYER ``mesh.areasvol`` exactly as the Fortran
``vol_wcelli`` (``gen_modules_cvmix_idemix.F90:524-530``): the surface interface ``uln`` takes
the top wet layer's area ``areasvol(uln)``, every deeper wet interface ``nz`` — interior and
the wet bottom interface ``nln+1`` — the layer ABOVE it, ``areasvol(nz-1)`` (1-based). 0-based:
``area_if[:, k] = areasvol[:, max(k-1, uln0)]`` on the wet range, 0 elsewhere. So the wet
bottom interface, where the bottom tidal forcing is deposited, has volume and exchanges energy
laterally. The neighbor arrays index the GLOBAL mesh, so the lateral step is dense/1-device
only (``sharded=True`` raises). The sharded runners take the per-device form
(:func:`partition_idemix_inputs`, neighbors mapped to local lanes), which is exact on ONE shard
and refused on more (halo lanes would go stale between the lateral sub-iterations).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from idemix_jax.forcing import IdemixForcing
from idemix_jax.mesh import (horizontal_propagation, idemix_step_columns, masks_from_levels,
                             neighbors_from_edges)
from idemix_jax.params import IdemixParams

from . import pp, tke
from .config import R_EARTH


class IdemixConfig(NamedTuple):
    """Static IDEMIX config (the :class:`~fesom_jax.tke.TkeConfig` precedent): hashable,
    no differentiable leaves — the tunables ``tau_v``/``tau_h``/``gamma``/``jstar``/``mu0``
    are an :class:`idemix_jax.IdemixParams`."""

    n_hor_iter: int = 5             # lateral explicit sub-steps (idemix_n_hor_iwe_prop_iter)
    lateral: bool = True            # horizontal propagation on/off
    lateral_cfl_cap: float = 0.1    # v0 <= sqrt(cap*res^2/(tau_h*dt/n_hor_iter)); 0.0 disables

    def validate(self) -> "IdemixConfig":
        """Raise on an invalid combination; returns ``self``."""
        if not isinstance(self.n_hor_iter, int) or self.n_hor_iter < 1:
            raise ValueError(f"IdemixConfig.n_hor_iter must be an int >= 1, got {self.n_hor_iter!r}")
        if not self.lateral_cfl_cap >= 0.0:
            raise ValueError(
                f"IdemixConfig.lateral_cfl_cap must be >= 0 (0 disables), got {self.lateral_cfl_cap!r}")
        return self


class LateralOperator(NamedTuple):
    """Static per-mesh connectivity of the lateral propagation (a pytree of arrays)."""

    nbr_idx: jax.Array      # [N, max_deg] int32; pad slots hold the own index
    nbr_weight: jax.Array   # [N, max_deg] float64, symmetric, >= 0; 0 on pad slots
    resolution: jax.Array   # [N] m (mesh.mesh_resolution), for the v0 CFL cap


class IdemixOut(NamedTuple):
    """Outputs of :func:`mixing_idemix`; every field ``[nod2D, nl]``, 0 outside the wet
    interfaces."""

    iwe_new: jax.Array      # wave energy after column step + lateral [m^2/s^2] (the new state)
    iw_diss: jax.Array      # wave dissipation (>= 0 for physical states) [m^2/s^3] -> TKE
    alpha_c: jax.Array      # dissipation coefficient [s/m^2] -> TKE
    E_iw: jax.Array         # column-step energy [m^2/s^2] (what TKE's Rinum cap reads)
    KappaH: jax.Array       # Osborn-shortcut diffusivity [m^2/s]
    KappaM: jax.Array       # Osborn-shortcut viscosity [m^2/s]


def _great_circle(lon1, lat1, lon2, lat2):
    """Haversine central angle [rad] between points given in radians."""
    s = (np.sin(0.5 * (lat2 - lat1)) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * (lon2 - lon1)) ** 2)
    return 2.0 * np.arcsin(np.sqrt(np.clip(s, 0.0, 1.0)))


def build_lateral_operator(mesh) -> LateralOperator:
    """The lateral-propagation connectivity of ``mesh`` (host-side NumPy, once per mesh).

    Edge weight ``w = cross_len / node_dist`` (dimensionless, unclipped): ``cross_len`` =
    ``hypot(cross[:,0], cross[:,1]) + hypot(cross[:,2], cross[:,3])`` from
    ``mesh.edge_cross_dxdy`` (the two edge-midpoint→element-centroid segments, m; one is
    zero on a boundary edge), ``node_dist`` = the great-circle distance between the edge's
    two nodes (``geo_coord_nod2D``, radius ``R_EARTH``). ``resolution`` =
    ``mesh.mesh_resolution``."""
    edges = np.asarray(mesh.edges)
    cross = np.asarray(mesh.edge_cross_dxdy, dtype=np.float64)
    cross_len = np.hypot(cross[:, 0], cross[:, 1]) + np.hypot(cross[:, 2], cross[:, 3])
    geo = np.asarray(mesh.geo_coord_nod2D, dtype=np.float64)          # (lon, lat) radians
    a, b = edges[:, 0], edges[:, 1]
    node_dist = R_EARTH * _great_circle(geo[a, 0], geo[a, 1], geo[b, 0], geo[b, 1])
    if np.any(node_dist <= 0.0):
        raise ValueError("build_lateral_operator: an edge joins two coincident nodes")
    w = cross_len / node_dist
    nbr_idx, nbr_weight = neighbors_from_edges(int(mesh.nod2D), edges, w)
    return LateralOperator(nbr_idx=jnp.asarray(nbr_idx, dtype=jnp.int32),
                           nbr_weight=jnp.asarray(nbr_weight, dtype=jnp.float64),
                           resolution=jnp.asarray(mesh.mesh_resolution, dtype=jnp.float64))


def _wet_interfaces(mesh):
    """``[nod2D, nl]`` bool: the wet interfaces ``ulevels_nod2D-1 .. nlevels_nod2D-1``."""
    return masks_from_levels(mesh.ulevels_nod2D - 1, mesh.nlevels_nod2D - 1, mesh.nl - 1)


def interface_area(mesh):
    """Per-interface cell area ``[nod2D, nl]`` [m^2] — the Fortran ``vol_wcelli`` area
    (``gen_modules_cvmix_idemix.F90:524-530``) from the per-LAYER ``mesh.areasvol``: interface
    ``k`` (0-based) takes layer ``max(k-1, uln0)`` (surface → top wet layer; interior and wet
    bottom interface → the layer above), 0 outside the wet interfaces."""
    k = jnp.arange(mesh.nl)[None, :]
    uln0 = (mesh.ulevels_nod2D - 1)[:, None]
    layer = jnp.minimum(jnp.maximum(k - 1, uln0), mesh.nl - 1)
    area_if = jnp.take_along_axis(mesh.areasvol, layer, axis=1)
    return jnp.where(_wet_interfaces(mesh), area_if, 0.0)


def mixing_idemix(mesh, iwe, bvfreq, hnode, forcing: IdemixForcing,
                  params_idemix: IdemixParams, cfg: IdemixConfig,
                  lateral_op: LateralOperator | None, *, dt: float, Z3d=None,
                  sharded: bool = False) -> IdemixOut:
    """One IDEMIX step on the nodes: the masked implicit column step, then (``cfg.lateral``)
    the explicit lateral propagation of the column-step energy (module docstring, rulings
    1-3). ``iwe``/``bvfreq``/``hnode`` are the ``[nod2D, nl]`` host fields; ``forcing`` the
    per-node surface/bottom energy fluxes [m^3/s^3]; ``Z3d`` the live zstar layer centers
    (``None`` ⇒ static mesh, as :func:`fesom_jax.tke.interface_spacing`). ``sharded=True``
    with ``cfg.lateral`` raises: the neighbor arrays index the global mesh."""
    cfg.validate()
    if cfg.lateral and sharded:
        raise NotImplementedError(
            "IDEMIX lateral propagation is dense/1-device only: neighbor arrays index the "
            "global mesh")
    if cfg.lateral and lateral_op is None:
        raise ValueError("IdemixConfig.lateral=True needs a LateralOperator "
                         "(build_lateral_operator(mesh))")
    nl = mesh.nl
    is_wet = _wet_interfaces(mesh)
    dzw = hnode[:, :nl - 1]
    dzt = tke.interface_spacing(mesh, hnode, Z3d)
    res = idemix_step_columns(
        iwe, bvfreq, dzw, dzt, mesh.coriolis_node,
        jnp.asarray(forcing.fsrf), jnp.asarray(forcing.fbot), dt, params_idemix,
        forc=None, is_wet=is_wet)

    if cfg.lateral:
        v0 = res.v0
        if cfg.lateral_cfl_cap > 0.0:
            sub_dt = dt / cfg.n_hor_iter
            v0_cap = jnp.sqrt(cfg.lateral_cfl_cap * lateral_op.resolution[:, None] ** 2
                              / (params_idemix.tau_h * sub_dt))
            v0 = jnp.minimum(v0, v0_cap)
        tend = horizontal_propagation(
            res.E_iw, v0, dzt, interface_area(mesh), lateral_op.nbr_idx, lateral_op.nbr_weight,
            dt, params_idemix, n_iter=cfg.n_hor_iter, is_wet=is_wet)   # [m^2/s^3]
        iwe_new = res.E_iw + dt * tend
    else:
        iwe_new = res.E_iw

    return IdemixOut(iwe_new=iwe_new, iw_diss=res.iw_diss, alpha_c=res.alpha_c,
                     E_iw=res.E_iw, KappaH=res.KappaH, KappaM=res.KappaM)


def osborn_kv_av(mesh, out: IdemixOut, bvfreq, *, exch=None, freeze_convection=False):
    """IDEMIX-only mixing (FESOM mix_scheme_nmb==6): KappaH->Kv, KappaM->Av via
    tke._wire_kv_av + pp.mo_convect (SPEC_STAGE3 §6.5). Returns (Kv, Av).

    ``exch`` is the node halo exchange required before the node→elem ``Av`` gather
    (``None`` ⇒ identity, the dense/1-device path); ``freeze_convection`` is
    ``pp.mo_convect``'s ``freeze_increment``."""
    k = jnp.arange(mesh.nl)[None, :]
    is_surf = k == (mesh.ulevels_nod2D - 1)[:, None]
    is_bot = k == (mesh.nlevels_nod2D - 1)[:, None]
    Kv, Av = tke._wire_kv_av(mesh, out.KappaM, out.KappaH, is_surf, is_bot, exch)
    return pp.mo_convect(mesh, Kv, Av, bvfreq, freeze_increment=freeze_convection)


def partition_idemix_inputs(forcing: IdemixForcing | None, lateral: LateralOperator | None,
                            part):
    """Gather the global IDEMIX forcing / lateral operator to the per-device padded
    ``[P, Lmax_nod, …]`` host form the sharded runners fold (the
    ``shard_mesh.partition_forcing_static`` precedent; pad lanes get the safe float pad 1.0).

    The lateral neighbor indices are mapped from global node ids to each device's LOCAL lanes
    (``myList_nod2D`` order); a neighbor absent from the device — possible only for halo nodes
    when ``P > 1`` — and every pad lane point at the lane itself with weight 0. On one device
    (``synth_serial``: identity list, no pad) the result equals the global operator. Either
    argument may be ``None`` (returned as ``None``)."""
    from .shard_mesh import _default_pad, _shard_along_axis, local_sizes

    _, Lmax = local_sizes(part)
    L = Lmax["nod"]
    lists = [np.asarray(ml) for ml in part.myList_nod2D]

    def _nodes(x):
        a = np.asarray(x, dtype=np.float64)
        return _shard_along_axis(a, lists, L, 0, _default_pad(a.dtype))

    forcing_p = None if forcing is None else IdemixForcing(*(_nodes(x) for x in forcing))
    lateral_p = None
    if lateral is not None:
        nbr = np.asarray(lateral.nbr_idx)
        w = np.asarray(lateral.nbr_weight, dtype=np.float64)
        deg = nbr.shape[1]
        idx_p = np.tile(np.arange(L, dtype=np.int32)[None, :, None], (len(lists), 1, deg))
        w_p = np.zeros((len(lists), L, deg))
        for d, ml in enumerate(lists):
            g2l = np.full(int(part.nod2D), -1, dtype=np.int64)
            g2l[ml] = np.arange(ml.size)
            loc = g2l[nbr[ml]]                                    # [n_local, deg]
            ok = loc >= 0
            idx_p[d, :ml.size] = np.where(ok, loc, np.arange(ml.size)[:, None])
            w_p[d, :ml.size] = np.where(ok, w[ml], 0.0)
        lateral_p = LateralOperator(nbr_idx=idx_p, nbr_weight=w_p,
                                    resolution=_nodes(lateral.resolution))
    return forcing_p, lateral_p
