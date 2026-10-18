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
:data:`fesom_jax.config.R_EARTH`). **Depth-masked face** (patch 0011, ``IdemixConfig.
lateral_face_mask``, default on): at interface ``k`` only the segments of the elements WET at
``k`` count (element interfaces ``ulevels-1 .. nlevels-1``, 0-based — the Fortran's loops
C/D/E, and for the no-cavity case the elements of the layer ``k-1`` whose ``areasvol`` is
interface ``k``'s area, see below), :func:`lateral_face_weight`. Without it every level of an
edge used the full face while the cell volume shrinks with depth to the wet elements' share; on
the deep interfaces of a node whose elements end at different depths that inflated the explicit
sub-step's Jacobi diagonal past the stability limit the ``v0`` cap (ruling 3) assumes, and the
CORE2 EKI members with large ``tau_h * gamma^2 / jstar^2`` blew up (idemix-jax
``examples/output/lateral_face_mask_report.txt``). ``lateral_face_mask=False`` is the full face,
bitwise patches 0001-0010. :func:`lateral_stability` is the matching preflight (``s * mu_max``
of the sub-step, <= 2 needed). Cell volumes are ``area_if * dzt`` with the PER-INTERFACE
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

import numbers
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from idemix_jax.core import idemix_step
from idemix_jax.forcing import IdemixForcing
from idemix_jax.mesh import (horizontal_propagation, idemix_step_columns,
                             lateral_step_stability, masks_from_levels, neighbors_from_edges,
                             validate_lateral_inputs)
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
    lateral_face_mask: bool = True  # face at k = wet elements' segments only (patch 0011);
    #                                 False = full face at every level (patches 0001-0010)

    def validate(self) -> "IdemixConfig":
        """Raise on an invalid combination; returns ``self``. ``n_hor_iter`` may be any
        integral number (Python or NumPy int, not bool)."""
        if (not isinstance(self.n_hor_iter, numbers.Integral) or isinstance(self.n_hor_iter, bool)
                or self.n_hor_iter < 1):
            raise ValueError(f"IdemixConfig.n_hor_iter must be an int >= 1, got {self.n_hor_iter!r}")
        if not self.lateral_cfl_cap >= 0.0:
            raise ValueError(
                f"IdemixConfig.lateral_cfl_cap must be >= 0 (0 disables), got {self.lateral_cfl_cap!r}")
        if not isinstance(self.lateral_face_mask, bool):
            raise ValueError(
                f"IdemixConfig.lateral_face_mask must be a bool, got {self.lateral_face_mask!r}")
        return self


class LateralOperator(NamedTuple):
    """Static per-mesh connectivity of the lateral propagation (a pytree of arrays).

    ``face_seg``/``face_top``/``face_bot`` describe the dual face of each slot's edge per
    adjacent element (``[..., 0]`` = ``edge_tri[:, 0]``, ``[..., 1]`` = ``edge_tri[:, 1]``):
    the element's cross segment over the node distance and its first/last wet interface
    (0-based; the empty range ``nl``/``-1`` for a missing element and on pad slots). They are
    the compact form of the per-level weight :func:`lateral_face_weight` expands in the step
    (``[N, max_deg, nl]`` float64 would be ~0.4 GB on CORE2)."""

    nbr_idx: jax.Array      # [N, max_deg] int32; pad slots hold the own index
    nbr_weight: jax.Array   # [N, max_deg] float64, symmetric, >= 0; 0 on pad slots (full face)
    resolution: jax.Array   # [N] m (mesh.mesh_resolution), for the v0 CFL cap
    face_seg: jax.Array     # [N, max_deg, 2] float64, element cross segment / node distance
    face_top: jax.Array     # [N, max_deg, 2] int32, element's first wet interface (0-based)
    face_bot: jax.Array     # [N, max_deg, 2] int32, element's last wet interface (0-based)


class IdemixParamsField(NamedTuple):
    """Per-node LOG multipliers of IDEMIX constants (``Params.idemix_field``; idemix-jax patch
    0010): node ``i`` runs with ``mu0 * exp(log_mu0[i])`` and the bottom energy flux
    ``fbot[i] * exp(log_fbot[i])``. Each entry is a ``[nod2D]`` float array or ``None`` (that
    constant stays the scalar / the forcing as given). The gradient w.r.t. a field at 0 is the
    sensitivity map ``dJ/d log mu0(x)`` (``dJ/d log fbot(x)``), one reverse pass for every node;
    its sum over the nodes is the gradient w.r.t. the scalar ``log mu0`` (``log`` of a uniform
    scaling of ``fbot``). A pytree: ``None`` entries carry no leaves. Dense (unsharded) runs
    only — the arrays index the global mesh."""

    log_mu0: jax.Array | None = None    # [nod2D] log multiplier of params.mu0
    log_fbot: jax.Array | None = None   # [nod2D] log multiplier of forcing.fbot


def _field_active(params_field) -> bool:
    return params_field is not None and (params_field.log_mu0 is not None
                                         or params_field.log_fbot is not None)


def _check_field(mesh, params_field) -> None:
    """Static shape check (trace time) of an active :class:`IdemixParamsField`."""
    n = int(mesh.nod2D)
    for name in IdemixParamsField._fields:
        x = getattr(params_field, name)
        if x is not None and tuple(jnp.shape(x)) != (n,):
            raise ValueError(f"IdemixParamsField.{name} must have shape ({n},) (one value per "
                             f"node), got {tuple(jnp.shape(x))}")


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


def check_no_cavity_mesh(mesh) -> None:
    """Raise ``ValueError`` if ``mesh`` has ice-shelf cavities (a node with
    ``ulevels_nod2D > 1``), whatever the run's ``cavity`` config says: upstream accepts a cavity
    mesh with ``cavity: null``, and IDEMIX is wired and verified for meshes without cavities only
    (no IDEMIX + cavity reference, and the TKE cavity column shift does not move the IDEMIX
    fields). Host-side NumPy on the static mesh levels, wherever the mesh is concrete on the
    IDEMIX path: :func:`build_lateral_operator`, the run config's ``idemix_inputs``, the run
    entry, and the column step when its mesh is not traced (:func:`_is_traced`)."""
    n = int(np.count_nonzero(np.asarray(mesh.ulevels_nod2D) > 1))
    if n:
        raise ValueError(
            f"IDEMIX is not supported on a mesh with ice-shelf cavities: {n} node(s) have "
            "ulevels_nod2D > 1 (IDEMIX is wired and verified for meshes without cavities only)")


def _is_traced(mesh) -> bool:
    """Whether ``mesh``'s levels or areas are tracers (``mesh`` an argument of a jitted function
    or a ``shard_map`` body), so that no host-side check can read them."""
    return any(isinstance(x, jax.core.Tracer)
               for x in (mesh.areasvol, mesh.ulevels_nod2D, mesh.nlevels_nod2D))


def check_wet_columns(mesh) -> None:
    """Raise ``ValueError`` unless every node column has at least one wet layer
    (``nlevels_nod2D - ulevels_nod2D >= 1``, i.e. >= 2 wet interfaces). IDEMIX's column solve
    and its end-cap interface spacing assume it; a column without a wet layer would be all
    "dry" interfaces, its forcing silently dropped. Host-side NumPy on the static mesh
    levels, run once per mesh (:func:`build_lateral_operator`, the run config's
    ``idemix_inputs`` and the run entry) -- never inside a traced step."""
    n_layers = np.asarray(mesh.nlevels_nod2D) - np.asarray(mesh.ulevels_nod2D)
    bad = np.flatnonzero(n_layers < 1)
    if bad.size:
        raise ValueError(
            f"IDEMIX needs >= 1 wet layer in every node column; {bad.size} node(s) have none "
            f"(nlevels_nod2D - ulevels_nod2D < 1), e.g. node(s) {bad[:5].tolist()}")


def build_lateral_operator(mesh) -> LateralOperator:
    """The lateral-propagation connectivity of ``mesh`` (host-side NumPy, once per mesh).

    Edge weight ``w = cross_len / node_dist`` (dimensionless, unclipped): ``cross_len`` =
    ``hypot(cross[:,0], cross[:,1]) + hypot(cross[:,2], cross[:,3])`` from
    ``mesh.edge_cross_dxdy`` (the two edge-midpoint→element-centroid segments, m; one is
    zero on a boundary edge), ``node_dist`` = the great-circle distance between the edge's
    two nodes (``geo_coord_nod2D``, radius ``R_EARTH``). ``resolution`` =
    ``mesh.mesh_resolution``. Raises on a cavity mesh (:func:`check_no_cavity_mesh`), if a node
    column has no wet layer (:func:`check_wet_columns`), or if the per-interface area
    :func:`interface_area` the lateral step will use is not finite and > 0 on every wet interface
    (``idemix_jax.mesh.validate_lateral_inputs``: ``horizontal_propagation`` silently
    exchanges nothing at such an interface, which hid the per-layer ``areasvol`` bug of
    patch 0003 for a whole campaign). All checks are host-side NumPy and change no number.

    The face fields (patch 0011): per slot and adjacent element (``mesh.edge_tri``, ``-1`` =
    none) the segment ``hypot(cross[2e], cross[2e+1]) / node_dist`` and the element's wet
    interfaces ``ulevels-1 .. nlevels-1`` (0-based), from which :func:`lateral_face_weight`
    builds the depth-masked face. The 2-D weight is checked to be symmetric, finite, >= 0 and
    0 on pads (``validate_lateral_inputs``)."""
    check_no_cavity_mesh(mesh)
    check_wet_columns(mesh)
    wet = np.asarray(_wet_interfaces(mesh))
    area_if = np.asarray(interface_area(mesh))
    validate_lateral_inputs(area_if, wet)
    nl = int(mesh.nl)
    edges = np.asarray(mesh.edges)
    cross = np.asarray(mesh.edge_cross_dxdy, dtype=np.float64)
    seg1 = np.hypot(cross[:, 0], cross[:, 1])
    seg2 = np.hypot(cross[:, 2], cross[:, 3])
    cross_len = seg1 + seg2
    geo = np.asarray(mesh.geo_coord_nod2D, dtype=np.float64)          # (lon, lat) radians
    a, b = edges[:, 0], edges[:, 1]
    node_dist = R_EARTH * _great_circle(geo[a, 0], geo[a, 1], geo[b, 0], geo[b, 1])
    if np.any(node_dist <= 0.0):
        raise ValueError("build_lateral_operator: an edge joins two coincident nodes")
    w = cross_len / node_dist
    nbr_idx, nbr_weight = neighbors_from_edges(int(mesh.nod2D), edges, w)
    # per-element face data, slot-aligned exactly like the weight (same edge loop)
    et = np.asarray(mesh.edge_tri, dtype=np.int64)
    if et.shape != edges.shape or np.any(et[:, 0] < 0):
        raise ValueError("build_lateral_operator: mesh.edge_tri must be [edge2D, 2] with an "
                         "el1 on every edge (-1 only for a missing el2)")
    has = et >= 0
    el = np.where(has, et, 0)
    top = np.where(has, np.asarray(mesh.ulevels)[el] - 1, nl)
    bot = np.where(has, np.asarray(mesh.nlevels)[el] - 1, -1)
    seg = np.stack([seg1, seg2], axis=1) / node_dist[:, None]
    _, packed = neighbors_from_edges(int(mesh.nod2D), edges,
                                     np.concatenate([seg, top, bot], axis=1))
    packed = np.asarray(packed)
    nbr_np = np.asarray(nbr_idx)
    pad = (nbr_np == np.arange(nbr_np.shape[0])[:, None])[:, :, None]
    face_top = np.where(pad, nl, packed[:, :, 2:4]).astype(np.int32)
    face_bot = np.where(pad, -1, packed[:, :, 4:6]).astype(np.int32)
    validate_lateral_inputs(area_if, wet, nbr_idx=nbr_np, nbr_weight=np.asarray(nbr_weight))
    return LateralOperator(nbr_idx=jnp.asarray(nbr_idx, dtype=jnp.int32),
                           nbr_weight=jnp.asarray(nbr_weight, dtype=jnp.float64),
                           resolution=jnp.asarray(mesh.mesh_resolution, dtype=jnp.float64),
                           face_seg=jnp.asarray(packed[:, :, 0:2], dtype=jnp.float64),
                           face_top=jnp.asarray(face_top), face_bot=jnp.asarray(face_bot))


def lateral_face_weight(lateral_op: LateralOperator, nl: int, cfg: IdemixConfig):
    """The face weight the lateral step passes to ``horizontal_propagation``.

    ``cfg.lateral_face_mask=False``: ``lateral_op.nbr_weight`` itself (``[N, max_deg]``, the
    full face at every level — bitwise patches 0001-0010). ``True`` (default): the per-level
    ``[N, max_deg, nl]`` weight of the elements wet at each interface ``k``: the full
    ``nbr_weight`` (bitwise) where both adjacent elements are wet, one element's
    ``face_seg`` where only it is, 0 where neither is. Symmetric (both slots of an edge carry
    the same face data). Traceable (``jnp``); built inside the step from the compact fields."""
    if not cfg.lateral_face_mask:
        return lateral_op.nbr_weight
    k = jnp.arange(nl)
    wet_e = ((k >= jnp.asarray(lateral_op.face_top)[..., None])
             & (k <= jnp.asarray(lateral_op.face_bot)[..., None]))      # [N, D, 2, nl]
    m1, m2 = wet_e[:, :, 0], wet_e[:, :, 1]
    seg = jnp.asarray(lateral_op.face_seg)
    return jnp.where(m1 & m2, jnp.asarray(lateral_op.nbr_weight)[:, :, None],
                     jnp.where(m1, seg[:, :, 0, None],
                               jnp.where(m2, seg[:, :, 1, None], 0.0)))


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


def _v0_cap(params_idemix: IdemixParams, cfg: IdemixConfig, lateral_op: LateralOperator,
            dt: float):
    """``[nod2D, 1]`` cap ``sqrt(cap * res**2 / (tau_h * dt / n_hor_iter))`` (ruling 3)."""
    sub_dt = dt / cfg.n_hor_iter
    return jnp.sqrt(cfg.lateral_cfl_cap * lateral_op.resolution[:, None] ** 2
                    / (params_idemix.tau_h * sub_dt))


def _lateral_v0(v0, params_idemix: IdemixParams, cfg: IdemixConfig,
                lateral_op: LateralOperator, dt: float):
    """The group velocity the lateral step uses: the column step's ``v0``, capped (ruling 3)."""
    if cfg.lateral_cfl_cap > 0.0:
        v0 = jnp.minimum(v0, _v0_cap(params_idemix, cfg, lateral_op, dt))
    return v0


def _columns_per_node_mu0(iwe, bvfreq, dzw, dzt, coriolis, fsrf, fbot, dt, params_idemix,
                          mu0_node, is_wet):
    """``idemix_step_columns``'s masked column step with ``mu0`` per column (``mu0_node``
    ``[nod2D]``): ``idemix_jax.core.idemix_step`` vmapped over the columns with the parameters'
    ``mu0`` batched and the other four shared (the library's column map closes over one
    ``IdemixParams``)."""
    def one(E, N2, dw, dz, f, fs, fb, w, m):
        return idemix_step(E, N2, dw, dz, f, fs, fb, dt, params=params_idemix._replace(mu0=m),
                           forc=jnp.zeros_like(E), is_wet=w)

    return jax.vmap(one)(iwe, bvfreq, dzw, dzt, coriolis, fsrf, fbot,
                         jnp.asarray(is_wet, dtype=bool), mu0_node)


def _column_step(mesh, iwe, bvfreq, hnode, forcing, params_idemix, dt, Z3d, params_field=None):
    """The step's column evaluation: ``(res, dzt, is_wet)`` (:func:`_idemix_core`)."""
    nl = mesh.nl
    is_wet = _wet_interfaces(mesh)
    dzw = hnode[:, :nl - 1]
    dzt = tke.interface_spacing(mesh, hnode, Z3d)
    if not _field_active(params_field):
        res = idemix_step_columns(
            iwe, bvfreq, dzw, dzt, mesh.coriolis_node,
            jnp.asarray(forcing.fsrf), jnp.asarray(forcing.fbot), dt, params_idemix,
            forc=None, is_wet=is_wet)
    else:
        _check_field(mesh, params_field)
        fbot = jnp.asarray(forcing.fbot)
        if params_field.log_fbot is not None:
            fbot = fbot * jnp.exp(params_field.log_fbot)
        if params_field.log_mu0 is None:
            res = idemix_step_columns(
                iwe, bvfreq, dzw, dzt, mesh.coriolis_node, jnp.asarray(forcing.fsrf), fbot,
                dt, params_idemix, forc=None, is_wet=is_wet)
        else:
            mu0_node = params_idemix.mu0 * jnp.exp(params_field.log_mu0)
            res = _columns_per_node_mu0(iwe, bvfreq, dzw, dzt, mesh.coriolis_node,
                                        jnp.asarray(forcing.fsrf), fbot, dt, params_idemix,
                                        mu0_node, is_wet)
    return res, dzt, is_wet


def _idemix_core(mesh, iwe, bvfreq, hnode, forcing, params_idemix, cfg, lateral_op, dt, Z3d,
                 params_field=None):
    """The step's IDEMIX evaluation: ``(res, iwe_new, dzt, is_wet)`` — the column result
    (:class:`idemix_jax.mesh` ``idemix_step_columns``) and the energy after the lateral step.
    Shared by :func:`mixing_idemix` and :func:`budget_terms` so the budget re-evaluates exactly
    the step's arithmetic. ``params_field`` (:class:`IdemixParamsField`): per-node log
    multipliers of ``mu0`` and of the bottom forcing; ``None`` (or all entries ``None``) is the
    scalar path, unchanged. The lateral face is :func:`lateral_face_weight` (``cfg.
    lateral_face_mask``)."""
    res, dzt, is_wet = _column_step(mesh, iwe, bvfreq, hnode, forcing, params_idemix, dt, Z3d,
                                    params_field)
    if cfg.lateral:
        v0 = _lateral_v0(res.v0, params_idemix, cfg, lateral_op, dt)
        tend = horizontal_propagation(
            res.E_iw, v0, dzt, interface_area(mesh), lateral_op.nbr_idx,
            lateral_face_weight(lateral_op, mesh.nl, cfg),
            dt, params_idemix, n_iter=int(cfg.n_hor_iter), is_wet=is_wet)   # [m^2/s^3]
        iwe_new = res.E_iw + dt * tend
    else:
        iwe_new = res.E_iw
    return res, iwe_new, dzt, is_wet


def lateral_stability(mesh, iwe, bvfreq, hnode, forcing: IdemixForcing,
                      params_idemix: IdemixParams, cfg: IdemixConfig,
                      lateral_op: LateralOperator, *, dt: float, Z3d=None,
                      params_field: IdemixParamsField | None = None, **kw):
    """Preflight: ``s * mu_max`` of the step's explicit lateral sub-step (host-side, not
    traced) — ``idemix_jax.mesh.lateral_step_stability`` on exactly what
    :func:`mixing_idemix` would pass to ``horizontal_propagation`` for these inputs: the
    column step's ``v0`` with the cap (ruling 3), the face :func:`lateral_face_weight`, the
    per-interface area, ``dzt`` and ``s = dt * tau_h / n_hor_iter``. A value above 2 means the
    sub-step amplifies a lateral mode (the column step that follows can damp a little of it;
    well above 2 is an instability). For screening parameter sets (e.g. ensemble members) on
    the state they start from. ``kw`` goes to ``lateral_step_stability`` (``dense_max``,
    ``tol``). Returns its ``LateralStability`` (``s_mu_max``, ``level``, ``column``,
    ``by_level``, ``s_diag_max``, ...). A concrete cavity mesh raises
    (:func:`check_no_cavity_mesh`, as in :func:`mixing_idemix`); a traced one cannot be read."""
    cfg.validate()
    if not cfg.lateral or lateral_op is None:
        raise ValueError("lateral_stability needs cfg.lateral=True and a LateralOperator")
    if not _is_traced(mesh):
        check_no_cavity_mesh(mesh)
    res, dzt, is_wet = _column_step(mesh, iwe, bvfreq, hnode, forcing, params_idemix, dt, Z3d,
                                    params_field)
    v0 = _lateral_v0(res.v0, params_idemix, cfg, lateral_op, dt)
    w = lateral_face_weight(lateral_op, mesh.nl, cfg)
    return lateral_step_stability(
        np.asarray(v0), np.asarray(dzt), np.asarray(interface_area(mesh)),
        np.asarray(lateral_op.nbr_idx), np.asarray(w), dt, params_idemix,
        n_iter=int(cfg.n_hor_iter), is_wet=np.asarray(is_wet), **kw)


def mixing_idemix(mesh, iwe, bvfreq, hnode, forcing: IdemixForcing,
                  params_idemix: IdemixParams, cfg: IdemixConfig,
                  lateral_op: LateralOperator | None, *, dt: float, Z3d=None,
                  sharded: bool = False,
                  params_field: IdemixParamsField | None = None) -> IdemixOut:
    """One IDEMIX step on the nodes: the masked implicit column step, then (``cfg.lateral``)
    the explicit lateral propagation of the column-step energy (module docstring, rulings
    1-3). ``iwe``/``bvfreq``/``hnode`` are the ``[nod2D, nl]`` host fields; ``forcing`` the
    per-node surface/bottom energy fluxes [m^3/s^3]; ``Z3d`` the live zstar layer centers
    (``None`` ⇒ static mesh, as :func:`fesom_jax.tke.interface_spacing`). ``sharded=True``
    with ``cfg.lateral`` raises: the neighbor arrays index the global mesh.
    ``params_field`` (:class:`IdemixParamsField`, ``Params.idemix_field``): per-node log
    multipliers of ``mu0`` and of the bottom forcing ``fbot``; ``None`` (default) ⇒ the scalar
    path, bitwise unchanged; refused with ``sharded=True``. A concrete cavity mesh raises
    (:func:`check_no_cavity_mesh`); a traced one cannot be read."""
    cfg.validate()
    if not _is_traced(mesh):
        check_no_cavity_mesh(mesh)
    if sharded and _field_active(params_field):
        raise NotImplementedError(
            "Params.idemix_field (per-node IDEMIX multipliers) is dense/1-device only: its "
            "arrays index the global mesh")
    if cfg.lateral and sharded:
        raise NotImplementedError(
            "IDEMIX lateral propagation is dense/1-device only: neighbor arrays index the "
            "global mesh")
    if cfg.lateral and lateral_op is None:
        raise ValueError("IdemixConfig.lateral=True needs a LateralOperator "
                         "(build_lateral_operator(mesh))")
    res, iwe_new, _, _ = _idemix_core(mesh, iwe, bvfreq, hnode, forcing, params_idemix, cfg,
                                      lateral_op, dt, Z3d, params_field)
    return IdemixOut(iwe_new=iwe_new, iw_diss=res.iw_diss, alpha_c=res.alpha_c,
                     E_iw=res.E_iw, KappaH=res.KappaH, KappaM=res.KappaM)


#: keys of :func:`budget_terms` (the run entry's ``idemix_budget_out`` files)
BUDGET_TERMS = ("dEdt", "P_surf", "P_bot", "D", "Dif", "Lat", "E", "abs_resid",
                "n_wet", "n_capped", "cfl_max")


def budget_terms(mesh, s0, s1, forcing: IdemixForcing, params_idemix: IdemixParams,
                 cfg: IdemixConfig, lateral_op: LateralOperator | None, *, dt: float,
                 Z3d=None, params_field: IdemixParamsField | None = None) -> dict:
    """Per-node (column-summed) wave-energy budget of the model step ``s0 -> s1`` [m^5/s^3]
    (multiply by ``rho0`` for W). A diagnostic only — the step never calls it.

    Re-evaluates the step's IDEMIX call (:func:`_idemix_core`, the code
    :func:`mixing_idemix` runs) on the step's exact inputs: ``s0.iwe``, ``s1.bvfreq`` (the
    step computes N^2 from the start-of-step T/S and carries it), ``s0.hnode`` and ``Z3d`` =
    the live zstar layer centres of ``s0.hnode`` (``None`` = static), and the step's
    ``params_field`` (``Params.idemix_field``; ``None`` = the scalar path). Volumes
    ``V = interface_area * dzt``. Each entry is a ``[nod2D]`` sum over the column's wet
    interfaces (sum over nodes for the global budget):

    * ``dEdt = sum V (s1.iwe - s0.iwe) / dt`` — from the CARRIED new energy;
    * ``P_surf``/``P_bot`` = ``sum V Tsur`` / ``sum V Tbot`` (forcing power);
    * ``D = sum V iw_diss`` (dissipation, >= 0 for physical states);
    * ``Dif = sum V Tdif`` (vertical diffusion: zero per column in the ``dzt`` metric, not in
      ``V``, because the node area changes with depth);
    * ``Lat = sum V (iwe_new - E_iw) / dt`` (lateral propagation; sums to ~0 over the mesh);
    * ``E = sum V s1.iwe``; ``abs_resid = |dEdt - (P_surf + P_bot - D + Dif + Lat)|``;
    * ``n_wet`` wet interfaces; ``n_capped`` those where the v0 CFL cap binds
      (``v0 > v0_cap``; 0 without ``cfg.lateral`` or with ``lateral_cfl_cap = 0``);
      ``cfl_max`` the column max of the UNCAPPED lateral CFL number
      ``v0^2 tau_h (dt / n_hor_iter) / res^2`` (0 without ``cfg.lateral``) — the cap's
      res-based number, not the sub-step's stability (that depends on the face and the volume,
      :func:`lateral_stability`).

    If ``s1`` is the step's output, ``dEdt = P_surf + P_bot - D + Dif + Lat`` per node and
    globally, to roundoff (the column solve's budget identity plus the lateral composition).
    A concrete cavity mesh raises (:func:`check_no_cavity_mesh`, as in :func:`mixing_idemix`);
    a traced one cannot be read."""
    if not _is_traced(mesh):
        check_no_cavity_mesh(mesh)
    res, iwe_new, dzt, is_wet = _idemix_core(mesh, s0.iwe, s1.bvfreq, s0.hnode, forcing,
                                             params_idemix, cfg, lateral_op, dt, Z3d,
                                             params_field)
    V = interface_area(mesh) * dzt

    def col(x):
        return jnp.sum(jnp.where(is_wet, x, 0.0), axis=1)

    t = dict(dEdt=col(V * (s1.iwe - s0.iwe)) / dt, P_surf=col(V * res.Tsur),
             P_bot=col(V * res.Tbot), D=col(V * res.iw_diss), Dif=col(V * res.Tdif),
             Lat=col(V * (iwe_new - res.E_iw)) / dt, E=col(V * s1.iwe))
    t["abs_resid"] = jnp.abs(t["dEdt"] - (t["P_surf"] + t["P_bot"] - t["D"] + t["Dif"]
                                          + t["Lat"]))
    t["n_wet"] = jnp.sum(is_wet, axis=1).astype(s0.iwe.dtype)
    zero = jnp.zeros_like(t["n_wet"])
    if cfg.lateral:
        r2 = lateral_op.resolution[:, None] ** 2
        cfl = res.v0 ** 2 * params_idemix.tau_h * (dt / cfg.n_hor_iter) / r2
        t["cfl_max"] = jnp.max(jnp.where(is_wet, cfl, 0.0), axis=1)
        if cfg.lateral_cfl_cap > 0.0:
            capped = is_wet & (res.v0 > _v0_cap(params_idemix, cfg, lateral_op, dt))
            t["n_capped"] = jnp.sum(capped, axis=1).astype(s0.iwe.dtype)
        else:
            t["n_capped"] = zero
    else:
        t["n_capped"] = zero
        t["cfl_max"] = zero
    return {k: t[k] for k in BUDGET_TERMS}


def osborn_kv_av(mesh, out: IdemixOut, bvfreq, *, exch=None, freeze_convection=False,
                 conv_surrogate=None):
    """IDEMIX-only mixing (FESOM mix_scheme_nmb==6): KappaH->Kv, KappaM->Av via
    tke._wire_kv_av + pp.mo_convect (SPEC_STAGE3 §6.5). Returns (Kv, Av).

    ``exch`` is the node halo exchange required before the node→elem ``Av`` gather
    (``None`` ⇒ identity, the dense/1-device path); ``freeze_convection`` is
    ``pp.mo_convect``'s ``freeze_increment``, ``conv_surrogate`` its ``surrogate_dn2``
    (``AdjointConfig.conv_surrogate``, the backward-only ramp, as on the PP/KPP/TKE paths).
    The convective adjustment is FESOM2's switch (``TkeConfig.convection`` is TKE-only)."""
    k = jnp.arange(mesh.nl)[None, :]
    is_surf = k == (mesh.ulevels_nod2D - 1)[:, None]
    is_bot = k == (mesh.nlevels_nod2D - 1)[:, None]
    Kv, Av = tke._wire_kv_av(mesh, out.KappaM, out.KappaH, is_surf, is_bot, exch)
    return pp.mo_convect(mesh, Kv, Av, bvfreq, freeze_increment=freeze_convection,
                         surrogate_dn2=conv_surrogate)


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
        seg = np.asarray(lateral.face_seg, dtype=np.float64)
        top = np.asarray(lateral.face_top, dtype=np.int32)
        bot = np.asarray(lateral.face_bot, dtype=np.int32)
        empty_top = int(top.max()) if top.size else 0       # >= nl on pads / missing elements
        deg = nbr.shape[1]
        P = len(lists)
        idx_p = np.tile(np.arange(L, dtype=np.int32)[None, :, None], (P, 1, deg))
        w_p = np.zeros((P, L, deg))
        seg_p = np.zeros((P, L, deg, 2))
        top_p = np.full((P, L, deg, 2), empty_top, dtype=np.int32)
        bot_p = np.full((P, L, deg, 2), -1, dtype=np.int32)
        for d, ml in enumerate(lists):
            g2l = np.full(int(part.nod2D), -1, dtype=np.int64)
            g2l[ml] = np.arange(ml.size)
            loc = g2l[nbr[ml]]                                    # [n_local, deg]
            ok = loc >= 0
            idx_p[d, :ml.size] = np.where(ok, loc, np.arange(ml.size)[:, None])
            w_p[d, :ml.size] = np.where(ok, w[ml], 0.0)
            seg_p[d, :ml.size] = np.where(ok[:, :, None], seg[ml], 0.0)
            top_p[d, :ml.size] = np.where(ok[:, :, None], top[ml], empty_top)
            bot_p[d, :ml.size] = np.where(ok[:, :, None], bot[ml], -1)
        lateral_p = LateralOperator(nbr_idx=idx_p, nbr_weight=w_p,
                                    resolution=_nodes(lateral.resolution), face_seg=seg_p,
                                    face_top=top_p, face_bot=bot_p)
    return forcing_p, lateral_p
