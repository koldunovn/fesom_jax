"""Per-substep halo-exchange schedule for the FESOM2 → JAX port (Phase 8, Task S.4).

The single source of truth S.7 wires in: which field is halo-exchanged, of which
kind, at which substep, and whether the exchange is **post-kernel** (insert after a
kernel returns — easy) or **intra-kernel** (fires MID-kernel ⇒ the JAX kernel that
currently fuses both stages must be SPLIT). Ported from the C
``port2/.../docs/MPI_PORT_REPORT.md`` "Halo exchanges per timestep" table (ocean)
plus the ``fesom_exchange_nod2D`` call sites in the ice C
(``fesom_ice_{evp,fct,coupling,thermo}.c``).

Substep labels match ``step.py`` (the ocean step) / ``ice_step.py`` (the ice step).
``kind`` is the :class:`~fesom_jax.shard_mesh.ShardedMesh` exchange-map key
(``'nod'`` / ``'elem'``); the C distinguishes ``elem2D`` (small ``eDim`` halo) from
``elem2D_full`` (``eDim+eXDim``), but the S.2 ``all_gather`` map refreshes the FULL
local extent (a superset) which is correct for the N-vs-1 gate, so both map to
``'elem'`` here. (If the per-substep C-N dump diff (S.9c) ever needs the exact
``eDim``-only intermediate, restrict the refresh to ``[myDim:myDim+eDim]``.)

Loop-bound rule (``PORTING_LESSONS §4``, verified for the JAX sharding on dist_4):
every owned node has ALL its incident edges + elements local, and every owned
element has all its contributing edges local. So a kernel's **local** scatter
(``segment_sum`` over local owned+halo entities) gives each **owned** entity its
COMPLETE sum; the post-kernel broadcast only refreshes the (incomplete) HALO copies
for the next kernel. No kernel scatter-loop needs a special halo bound beyond using
the local connectivity + the post-kernel exchange.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Exch:
    """One scheduled halo exchange."""
    after: str        # JAX substep / kernel label it follows
    field: str        # the variable (step.py local or State field) exchanged
    kind: str         # 'nod' | 'elem' (ShardedMesh.exchange key)
    placement: str    # 'post' (after the kernel) | 'intra' (mid-kernel ⇒ split)
    cref: str         # C reference (MPI_PORT_REPORT row or fesom_*.c:line)


_KINDS = ("nod", "elem")
_PLACEMENTS = ("post", "intra")


# --------------------------------------------------------------------------
# Ocean step (step.py) — MPI_PORT_REPORT "Halo exchanges per timestep"
# --------------------------------------------------------------------------
OCEAN_SCHEDULE: tuple[Exch, ...] = (
    Exch("1 eos.compute_pressure_bv", "density",   "nod", "post", "pressure_bv row"),
    Exch("1 eos.compute_pressure_bv", "hpressure", "nod", "post", "pressure_bv row"),
    Exch("1 eos.compute_pressure_bv", "bvfreq",    "nod", "post", "pressure_bv row"),
    # KPP path also needs sw_alpha/sw_beta (computed substep 4); the C exchanges them
    # with the pressure_bv block. They feed only owned-node KPP work ⇒ exchange post-EOS.
    Exch("3 pgf.pressure_force_linfs", "pgf_x", "elem", "post", "pressure_force row"),
    Exch("3 pgf.pressure_force_linfs", "pgf_y", "elem", "post", "pressure_force row"),
    Exch("4 mixing (pp/kpp)", "uvnode", "nod",  "post", "compute_vel_nodes row (oce_dyn.F90:225)"),
    Exch("4 mixing (pp/kpp)", "Kv",     "nod",  "post", "pp_mixing/mo_convect row"),
    Exch("4 mixing (pp/kpp)", "Av",     "elem", "post", "pp_mixing/mo_convect row"),
    Exch("5 momentum.compute_vel_rhs", "uv_rhs",   "elem", "post", "compute_vel_rhs row"),
    Exch("5 momentum.compute_vel_rhs", "uv_rhsAB", "elem", "post", "compute_vel_rhs row"),
    # 6 visc_filt_bidiff is a FUSED bilaplacian: the C exchanges u_b/v_b mid-kernel,
    # then u_c/v_c, then uv_rhs at the end (oce_dyn.F90:367,395). ⇒ SPLIT (S.7).
    Exch("6 momentum.visc_filt_bidiff [stage1]", "u_b", "elem", "intra", "visc_filt_bcksct (oce_dyn.F90:367)"),
    Exch("6 momentum.visc_filt_bidiff [stage1]", "v_b", "elem", "intra", "visc_filt_bcksct (oce_dyn.F90:367)"),
    Exch("6 momentum.visc_filt_bidiff [stage2]", "u_c", "nod",  "intra", "visc_filt_bcksct (oce_dyn.F90:395)"),
    Exch("6 momentum.visc_filt_bidiff [stage2]", "v_c", "nod",  "intra", "visc_filt_bcksct (oce_dyn.F90:395)"),
    Exch("6 momentum.visc_filt_bidiff", "uv_rhs", "elem", "post", "visc_filt external row"),
    Exch("7 momentum.impl_vert_visc", "uv_rhs", "elem", "post", "impl_vert_visc row"),
    Exch("8 ssh.compute_ssh_rhs", "ssh_rhs", "nod", "post", "compute_ssh_rhs_linfs (oce_ale.F90:1954)"),
    # 9 CG (_pcg): pp each iter before SpMV, rr after residual, X final ⇒ INTRA (S.6).
    Exch("9 ssh.solve_ssh [CG iter]", "pp", "nod", "intra", "inside CG (solver.F90)"),
    Exch("9 ssh.solve_ssh [CG iter]", "rr", "nod", "intra", "inside CG (solver.F90)"),
    Exch("9 ssh.solve_ssh", "d_eta", "nod", "post", "after CG (oce_ale.F90:3117)"),
    Exch("10 momentum.update_vel", "uv", "elem", "post", "update_vel row (oce_dyn.F90:172)"),
    Exch("11 ssh.compute_hbar", "ssh_rhs_old", "nod", "post", "compute_hbar (oce_ale.F90:2078)"),
    Exch("11 ssh.compute_hbar", "hbar",        "nod", "post", "compute_hbar (oce_ale.F90:2102)"),
    Exch("13 ale.compute_w", "w", "nod", "post", "ALE vert_vel (oce_ale.F90:2679)"),
    Exch("13 ale.compute_cfl_z", "cfl_z", "nod", "post", "ALE compute_cflz row"),
    Exch("13 ale.compute_wvel_split", "w_e", "nod", "post", "ALE wvel_split row"),
    Exch("13 ale.compute_wvel_split", "w_i", "nod", "post", "ALE wvel_split row"),
    # 15 advect_one_fct is a FUSED Zalesak FCT: exchange fct_LO after the low-order
    # solve, then fct_plus/fct_minus before the limiter (oce_adv_tra_*.F90). ⇒ SPLIT.
    Exch("15 tracer_adv.advect_one_fct [LO]",     "fct_LO",    "nod", "intra", "oce_adv_tra_driver.F90:294"),
    Exch("15 tracer_adv.advect_one_fct [limiter]", "fct_plus",  "nod", "intra", "oce_adv_tra_fct.F90:401"),
    Exch("15 tracer_adv.advect_one_fct [limiter]", "fct_minus", "nod", "intra", "oce_adv_tra_fct.F90:401"),
    Exch("15 tracer_adv.advect_one_fct", "T", "nod", "post", "after tracer FCT (oce_ale_tracer.F90:268)"),
    Exch("15 tracer_adv.advect_one_fct", "S", "nod", "post", "after tracer FCT"),
    Exch("15 tracer_diff.impl_vert_diff", "T", "nod", "post", "after impl_vert_diff"),
    Exch("15 tracer_diff.impl_vert_diff", "S", "nod", "post", "after impl_vert_diff"),
    Exch("16 ale.commit_thickness", "hnode", "nod",  "post", "commit_thickness (oce_ale.F90:1027)"),
    Exch("16 ale.commit_thickness", "helem", "elem", "post", "commit_thickness (oce_ale.F90:1249)"),
)


# --------------------------------------------------------------------------
# Sea-ice step (ice_step.py) — fesom_ice_*.c fesom_exchange_nod2D call sites
# --------------------------------------------------------------------------
ICE_SCHEDULE: tuple[Exch, ...] = (
    # ocean2ice recomputes + exchanges the node surface current (fesom_ice_coupling.c:113);
    # in JAX it reads the carried state.uvnode (already exchanged by the ocean step).
    Exch("ice ocean2ice", "uvnode", "nod", "post", "fesom_ice_coupling.c:113-114"),
    # EVP momentum subcycle: u_ice/v_ice exchanged INSIDE the 120-step subcycle scan
    # (fesom_ice_evp.c:446) ⇒ a collective inside lax.scan under shard_map.
    Exch("ice EVP subcycle [in scan]", "u_ice", "nod", "intra", "fesom_ice_evp.c:446"),
    Exch("ice EVP subcycle [in scan]", "v_ice", "nod", "intra", "fesom_ice_evp.c:447"),
    # ice FCT advection — a fused Zalesak FCT like the ocean tracer FCT (fesom_ice_fct.c):
    # low-order solve, antidiffusive flux loop, then the limiter ⇒ SPLIT.
    Exch("ice FCT [LO]",      "m_ice_lo", "nod", "intra", "fesom_ice_fct.c:268-270"),
    Exch("ice FCT [LO]",      "a_ice_lo", "nod", "intra", "fesom_ice_fct.c:268-270"),
    Exch("ice FCT [LO]",      "m_snow_lo", "nod", "intra", "fesom_ice_fct.c:268-270"),
    Exch("ice FCT [limiter]", "icepplus",  "nod", "intra", "fesom_ice_fct.c:474"),
    Exch("ice FCT [limiter]", "icepminus", "nod", "intra", "fesom_ice_fct.c:475"),
    Exch("ice FCT", "m_ice",  "nod", "post", "fesom_ice_fct.c:517 (final m_ice,a_ice,m_snow)"),
    Exch("ice FCT", "a_ice",  "nod", "post", "fesom_ice_fct.c:517"),
    Exch("ice FCT", "m_snow", "nod", "post", "fesom_ice_fct.c:517"),
    # thermodynamics emits ustar (fesom_ice_thermo.c:534); ice-ocean flux balances.
    Exch("ice thermo", "ustar", "nod", "post", "fesom_ice_thermo.c:534"),
    Exch("ice oce_fluxes", "virtual_salt", "nod", "post", "fesom_ice_coupling.c:160"),
    Exch("ice oce_fluxes", "relax_salt",   "nod", "post", "fesom_ice_coupling.c:174"),
    Exch("ice oce_fluxes", "heat_flux",    "nod", "post", "fesom_ice_coupling.c:177"),
    Exch("ice oce_fluxes", "water_flux",   "nod", "post", "fesom_ice_coupling.c:178"),
)


# --------------------------------------------------------------------------
# Fused JAX kernels that must be SPLIT to host an intra-kernel exchange (S.7)
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FusedSplit:
    kernel: str          # the JAX function to split
    seam: str            # where the halo exchange goes
    cref: str


FUSED_KERNELS_NEEDING_SPLIT: tuple[FusedSplit, ...] = (
    FusedSplit("momentum.visc_filt_bidiff",
               "after stage-1 u_b/v_b (elem) ‖ before stage-2 u_c/v_c (nod) ‖ exchange each",
               "oce_dyn.F90:367,395"),
    FusedSplit("tracer_adv.advect_one_fct (ocean)",
               "after fct_LO (low-order) ‖ before the Zalesak plus/minus limiter",
               "oce_adv_tra_driver.F90:294 / oce_adv_tra_fct.F90:401"),
    FusedSplit("ssh._pcg (CG)",
               "exchange pp before each SpMV, rr after each residual update (S.6)",
               "solver.F90 parallel CG"),
    FusedSplit("ice_adv FCT",
               "after the low-order solve ‖ before the limiter (like the ocean FCT)",
               "fesom_ice_fct.c:268,474"),
    FusedSplit("ice_evp EVP subcycle",
               "exchange u_ice/v_ice INSIDE the 120-step subcycle lax.scan",
               "fesom_ice_evp.c:446"),
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def post_exchanges(schedule: tuple[Exch, ...]) -> tuple[Exch, ...]:
    """The post-kernel exchanges (simple inserts) of a schedule."""
    return tuple(e for e in schedule if e.placement == "post")


def intra_exchanges(schedule: tuple[Exch, ...]) -> tuple[Exch, ...]:
    """The intra-kernel exchanges (need a kernel split) of a schedule."""
    return tuple(e for e in schedule if e.placement == "intra")


def validate(schedule: tuple[Exch, ...]) -> None:
    """Sanity-check a schedule's kinds/placements (raises on a typo)."""
    for e in schedule:
        if e.kind not in _KINDS:
            raise ValueError(f"{e.after}/{e.field}: bad kind {e.kind!r}")
        if e.placement not in _PLACEMENTS:
            raise ValueError(f"{e.after}/{e.field}: bad placement {e.placement!r}")


validate(OCEAN_SCHEDULE)
validate(ICE_SCHEDULE)
