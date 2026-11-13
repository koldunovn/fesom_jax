"""Mode F: FESOM2's coupled (OpenIFS, ``__oasis`` + ``__oifs``) surface path for fesom_jax.

What FESOM2 does in AWI-ESM3 v3.5.1 (fesom2_upstream dbf28153; namelists of the AWI-ESM3 run: use_atm_ice_tskin,
use_meltponds, ice_ave_steps 1), and where it is here:

* receive (gen_forcing_couple.F90 update_atm_forcing, ``__oasis``/``__oifs``): every field is taken as OASIS
  delivers it (``force_flux_consv`` and ``net_rec_from_atm`` return at once under ``__oifs``: the conservation is
  OASIS's), held until the next exchange; the four stresses are rotated from geographic to the model frame
  (vector_g2r). -> :func:`flux_step_forcing`.
* ice_thermo_cpl.F90 ``thermodynamics`` (Dorn et al. 2009, the coupled-slab convention): the surface temperature is
  the atmosphere's ice-tile skin ``tsk_ico`` (clamped to 273.15 K), the growth budget is driven by ``-heat_ico``
  over ice and ``-(heat_oce + heat_swo)`` over open water, evaporation/sublimation are weighted by the open-water /
  ice fraction, the albedo is ``ice_albedo`` with melt ponds (ice_meltponds.F90) -> :func:`thermo_cpl`.
* ice_oce_coupling.F90 ``oce_fluxes`` coupled branches: the zstar freshwater balance adds the residual ice
  freshwater flux and scales the atmospheric water by rhofwt/rhowat -> :func:`ice_surface_step_cpl`.
* ``oce_fluxes_mom``: the received ocean stress ``taux_oce`` blended with the ice-ocean stress; the EVP takes the
  received ice stress ``taux_ico``; ``cal_shortwave_rad`` penetrates ``0.54 (1 - albw) heat_swo`` (albw = 0.1,
  the coupled build's constant) -> :func:`ice_surface_step_cpl`.

Differences from the Fortran (also in docs/coupling/awicm.md):

* ice_temp is not advected with the ice (FESOM2 advects ice_temp*a_ice as a 4th FCT tracer). With
  use_atm_ice_tskin the thermodynamics overwrite it wherever a_ice > 0.001, so the advected value is read only on
  nodes with a_ice <= 0.001 (the albedo of a trace of ice, the snow-melt gate and the pond lid there).
* Runoff (FluxConfig.runoff, init_ocean's ``cpl_runoff``): ``"climatology"`` (default, plan decision 1 of the
  CORE2 coupling) uses the CORE2 runoff climatology and ignores hydr_oce/calv_oce, so enthalpyoffuse = 0;
  ``"coupled"`` takes the runoff mapper's hydr_oce/calv_oce as FESOM main 0aaa2d73 does under ``__oasis``/``__oifs``
  (gen_forcing_couple.F90 update_atm_forcing, receive 12 and 13): ``runoff = hydr_oce`` (liquid), ``runoff_solid =
  calv_oce``, ``enthalpyoffuse = -min(calv_oce * 333.55e6, 1000)`` [W m-2, the heat to melt the calved ice],
  ``runoff = runoff_liquid + runoff_solid``; ice_thermo_cpl.F90 then adds enthalpyoffuse to the open-water heat
  flux ``a2ohf`` and ``runo = runoff`` to ``PmEocn``, and oce_fluxes adds ``runoff`` to the coupled freshwater
  balance (force_flux_consv returns at once under ``__oifs``). u10w/v10w are accepted and unused (FESOM2 reads them
  only for icebergs).
* fesom_jax's ocean is the CORE2 configuration (zstar, TKE, mEVP, GM/Redi); AWI-ESM3's FESOM2 runs linfs, KPP,
  standard EVP. The coupled branches above are those of zstar (use_virt_salt false).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .. import ice_adv, ice_coupling, ice_evp, ice_mevp, ice_thermo
from .. import forcing as _forcing
from ..config import VCPW

#: State fields mode F carries (None otherwise): FESOM2's ice%data(4) ice_temp [K], ice%atmcoupl%ice_alb, and the
#: melt-pond area fraction / depth [m] / lid thickness [m] (ice%thermo%apnd/hpnd/ipnd)
CPL_FIELDS = ("cpl_ice_temp", "cpl_ice_alb", "cpl_apnd", "cpl_hpnd", "cpl_ipnd")
# FESOM2's initial values (ice_init.F90: ice_temp 265.15 K, ice_alb 0.6, ponds 0)
CPL_INIT = {"cpl_ice_temp": 265.15, "cpl_ice_alb": 0.6, "cpl_apnd": 0.0, "cpl_hpnd": 0.0, "cpl_ipnd": 0.0}

AIMIN = 0.001          # ice_thermo_cpl.F90 Aimin
HIMIN = 0.005          # himin
HCUTOFF = 1.0e-6
BIGVAL = 1.0e10
GAMMA_T = 10.0 / 86400.0
H0MAX = 1.5
TMELT = 273.15


class MeltPondParams(NamedTuple):
    """ice_meltponds.F90 &meltpond (the AWI-ESM3 namelist.ice values = the module defaults) + its constants."""
    hi_min: float = 0.10
    hs1: float = 0.03
    rfracmin: float = 0.15
    rfracmax: float = 1.00
    pndaspect: float = 0.80
    albpnd: float = 0.20
    albpnd_frz: float = 0.36
    puny: float = 1.0e-11
    rhoice_p: float = 917.0
    rhosno_p: float = 330.0
    rhowat_p: float = 1025.0
    rhofresh: float = 1000.0
    kice: float = 2.03
    lfresh: float = 3.34e5


@dataclasses.dataclass(frozen=True)
class FluxConfig:
    """The coupled-thermodynamics switches of AWI-ESM3 v3.5.1 (namelist.config/ice)."""
    use_meltponds: bool = True        # namelist.ice use_meltponds
    snowmelt_tgate: bool = True       # MOD_ICE default
    use_atm_ice_tskin: bool = True    # namelist.config
    ponds: MeltPondParams = MeltPondParams()
    runoff: str = "climatology"       # "climatology" (CORE2 runoff file) or "coupled" (hydr_oce + calv_oce)


RUNOFF_MODES = ("climatology", "coupled")
# gen_forcing_couple.F90: enthalpyoffuse = runoff_solid*333.55*1000000.0 [W m-2], evaluated left to right; the
# literals are default (4-byte) reals, so 333.55 is float32(333.55) in either working precision
LATENT_FUSION = float(np.float32(333.55))
ENTHALPY_CAP = 1000.0                      # enthalpyoffuse = -min(enthalpyoffuse, 1000.0)


class CplForcing(NamedTuple):
    """Per-step mode F forcing of the ocean step (FESOM2's received arrays, model frame). Each ``[nod2D]``."""
    taux_oce: jnp.ndarray     # stress_atmoce_x [N m-2], rotated to the model frame
    tauy_oce: jnp.ndarray
    taux_ico: jnp.ndarray     # stress_atmice_x
    tauy_ico: jnp.ndarray
    prec_rain: jnp.ndarray    # prec_oce [m s-1 fresh water, + down]
    prec_snow: jnp.ndarray    # snow_oce
    evap: jnp.ndarray         # evap_oce = evap_no_ifrac [m s-1, + down, per open-water area]
    subl: jnp.ndarray         # subl_oce = sublimation [m s-1, + down, per ice area]
    heat_oce: jnp.ndarray     # oce_heat_flux [W m-2, + down, non-solar over open water]
    heat_ico: jnp.ndarray     # ice_heat_flux = a2ihf [W m-2, + down, over ice]
    heat_swo: jnp.ndarray     # shortwave [W m-2, net into open water]
    tsk_ico: jnp.ndarray      # ist_ref [K], the atmosphere's ice-tile skin temperature
    Ssurf_month: jnp.ndarray  # SSS restoring target (rate 0 in the coupled configuration)
    chl: jnp.ndarray          # chlorophyll (shortwave penetration)
    # FluxConfig.runoff = "coupled" only (None: the static CORE2 climatology, the default's program unchanged)
    runoff: jnp.ndarray = None      # runoff = hydr_oce + calv_oce [m s-1 fresh water, + into the ocean]
    enthalpy: jnp.ndarray = None    # enthalpyoffuse = -min(calv_oce * 333.55e6, 1000) [W m-2]
    COUPLED_FLUX = True       # the step's dispatch mark (fesom_jax.step: the coupled ice step)


def flux_config(ice_cfg) -> FluxConfig:
    """The :class:`FluxConfig` of a run from its ``ice:`` block (``fesom_jax.ice.IceConfig``: ``cpl_meltponds``,
    ``snowmelt_tgate``, ``pnd_*``). The IceConfig defaults give ``FluxConfig()``, the AWI-ESM3 v3.5.1 setting."""
    if ice_cfg is None:
        return FluxConfig()
    pp = MeltPondParams(hi_min=ice_cfg.pnd_hi_min, hs1=ice_cfg.pnd_hs1, rfracmin=ice_cfg.pnd_rfracmin,
                        rfracmax=ice_cfg.pnd_rfracmax, pndaspect=ice_cfg.pnd_aspect, albpnd=ice_cfg.pnd_albpnd,
                        albpnd_frz=ice_cfg.pnd_albpnd_frz)
    return FluxConfig(use_meltponds=bool(ice_cfg.cpl_meltponds), snowmelt_tgate=bool(ice_cfg.snowmelt_tgate),
                      ponds=pp)


# ---------------------------------------------------------------------------------------------------------------
# init / forcing
# ---------------------------------------------------------------------------------------------------------------
def init_flux(state, static, runoff: str = "climatology"):
    """Give ``state`` the :data:`CPL_FIELDS` (from a mode F restart they are kept; else FESOM2's initial values, in
    the working precision) and ``static`` the mode F configuration (``runoff``: :data:`RUNOFF_MODES`)."""
    if runoff not in RUNOFF_MODES:
        raise ValueError(f"runoff must be one of {RUNOFF_MODES}, got {runoff!r}")
    if getattr(state, "cpl_ice_temp", None) is None:
        dt = state.a_ice.dtype
        state = dataclasses.replace(state, **{k: jnp.full_like(state.a_ice, v, dtype=dt) for k, v in CPL_INIT.items()})
    static.flux = dataclasses.replace(flux_config(static.cfg.ice), runoff=runoff)
    return state, static


def runoff_enthalpy(hydr, calv):
    """FESOM main 0aaa2d73 gen_forcing_couple.F90 (``__oifs``, receive 13): (runoff_liquid + runoff_solid,
    enthalpyoffuse) from hydr_oce and calv_oce, in their precision."""
    dt = calv.dtype
    enth = -jnp.minimum((calv * jnp.asarray(LATENT_FUSION, dt)) * jnp.asarray(1000000.0, dt),
                        jnp.asarray(ENTHALPY_CAP, dt))
    return hydr + calv, enth


def flux_step_forcing(state, forcing: dict, static, dates):
    """Mode F: the :class:`CplForcing` stack ``[nsteps, nod2D]`` of one interval from the fields as OASIS delivers
    them (``MODE_F_FIELDS``, geographic stresses), each ``(nod2D,)`` held over the interval or ``(nsteps, nod2D)``.
    Missing ``tsk_ico`` = FESOM2's ist_ref before the first receive (0 K, which thermo_cpl replaces by 271.35 K).
    Returns ``(state, seq)`` (the State unchanged)."""
    from .ocean import _rows, g2r
    from .. import jra55
    from .. import precision as PR
    n, T = static.nod2D, len(dates)
    need = ("taux_oce", "tauy_oce", "taux_ico", "tauy_ico", "prec_oce", "snow_oce", "evap_oce", "subl_oce",
            "heat_oce", "heat_ico", "heat_swo")
    missing = [k for k in need if k not in forcing]
    if missing:
        raise KeyError(f"mode F forcing lacks {missing}")
    r = lambda k: _rows(forcing, k, T, n)                        # noqa: E731
    tox, toy, tix, tiy = r("taux_oce"), r("tauy_oce"), r("taux_ico"), r("tauy_ico")
    if jra55.FORCE_ROTATION:                                      # rotated_grid: both pairs received -> g2r
        tox, toy = g2r(tox, toy, static.trig, static.M)
        tix, tiy = g2r(tix, tiy, static.trig, static.M)
    tsk = r("tsk_ico") if "tsk_ico" in forcing else jnp.zeros((T, n), tox.dtype)
    months = np.array([d[3] for d in dates]) - 1
    extra = {}
    if getattr(static.flux, "runoff", "climatology") == "coupled":
        miss = [k for k in ("hydr_oce", "calv_oce") if k not in forcing]
        if miss:
            raise KeyError(f"runoff 'coupled' needs {miss} (the runoff mapper's fields)")
        hydr, calv = r("hydr_oce"), r("calv_oce")
        if static.precision == "single":             # FESOM2's WP arithmetic of the received fields
            hydr, calv = hydr.astype(jnp.float32), calv.astype(jnp.float32)
        extra["runoff"], extra["enthalpy"] = runoff_enthalpy(hydr, calv)
    seq = CplForcing(taux_oce=tox, tauy_oce=toy, taux_ico=tix, tauy_ico=tiy,
                     prec_rain=r("prec_oce"), prec_snow=r("snow_oce"), evap=r("evap_oce"), subl=r("subl_oce"),
                     heat_oce=r("heat_oce"), heat_ico=r("heat_ico"), heat_swo=r("heat_swo"), tsk_ico=tsk,
                     Ssurf_month=jnp.asarray(static.sss_clim[months]), chl=jnp.asarray(static.chl_clim[months]),
                     **extra)
    return state, (PR.cast_tree(seq) if static.precision == "single" else seq)



# ---------------------------------------------------------------------------------------------------------------
# the same forcing build as ONE compiled program (gpu.cpl_forcing_jit, FESOM_CPL_FORCING_JIT)
# ---------------------------------------------------------------------------------------------------------------
JIT_ENV = "FESOM_CPL_FORCING_JIT"


def forcing_jit_enabled() -> bool:
    """``gpu.cpl_forcing_jit`` (``FESOM_CPL_FORCING_JIT=1``): :func:`flux_step_forcing` as 8 compiled programs instead
    of ~100 eager device ops a coupling interval (each op a Python dispatch: ~20 ms of GPU idle an interval on GH200
    under nsys, more while the atmosphere's thread holds the GIL). Values bitwise the eager build (see
    :func:`_stages`)."""
    import os
    return os.environ.get(JIT_ENV, "0").strip() not in ("", "0")


def _stages():
    """The compiled stages of :func:`_flux_seq_staged`, built once. Each stage holds only multiplications (and
    negations) or only additions (and subtractions, min, casts, gathers): XLA contracts ``a*b + c`` into one FMA
    when a product and its sum share a fusion (one rounding instead of two; an optimization_barrier does not stop
    it, the barrier is gone before fusion), so a product and the sum it feeds are never in the same program."""
    g = globals()
    if "_STAGES" in g:
        return g["_STAGES"]

    def m1(pairs, trig):              # vector_g2r's first products (fesom_jax.jra55._vector_g2r)
        sgl, cgl, sgo, cgo = trig[:4]
        return [((-v) * sgl * cgo, u * sgo, (-v) * sgl * sgo, u * cgo, v * cgl) for u, v in pairs]

    def a1(prods):
        return [(p[0] - p[1], p[2] + p[3], p[4]) for p in prods]

    def m2(vecs, M):
        return [tuple(M[3 * i + j] * vec[j] for i in range(3) for j in range(3)) for vec in vecs]

    def a2(prods):
        return [tuple(p[3 * i] + p[3 * i + 1] + p[3 * i + 2] for i in range(3)) for p in prods]

    def m3(vecs, trig):
        srl, crl, sro, cro = trig[4:]
        return [((-srl) * cro * x, srl * sro * y, crl * z, (-sro) * x, cro * y) for x, y, z in vecs]

    def a3(prods):
        return [(p[3] + p[4], p[0] - p[1] + p[2]) for p in prods]            # (u_rot, v_rot)

    def enth_m(calv, lf, mega):       # (calv * L) * 1e6 as written (the factors as arguments: nothing folded)
        return (calv * lf) * mega

    g["_STAGES"] = tuple(jax.jit(f) for f in (m1, a1, m2, a2, m3, a3, enth_m))
    return g["_STAGES"]


def _flux_lean_call(fields, static, dates, cache, key, kw):
    """The lean build (:func:`forcing_lean_enabled`): constants, scalars, month rows and (dist_N) the lane maps
    placed once per placement of the forcing."""
    import functools
    from .. import jra55
    leaf = next(iter(fields.values()))
    place = leaf.sharding
    lanes_here = static.npes > 1
    if lanes_here:                   # the forcing replicated over the ocean's devices (as the coupler puts it)
        from jax.sharding import NamedSharding, PartitionSpec
        mesh = static.device_mesh()
        if set(place.device_set) != set(mesh.devices.flat):
            place = NamedSharding(mesh, PartitionSpec())
            fields = jax.device_put(fields, place)
    if lanes_here and forcing_lanes_first_enabled():
        return _lanes_first_call(fields, static, dates, cache, kw)
    lk = ("lean", key, lanes_here)
    if lk not in cache:
        out_sh = None
        if lanes_here:
            from jax.sharding import NamedSharding, PartitionSpec
            out_sh = NamedSharding(static.device_mesh(), PartitionSpec(None, "p"))
        asm = functools.partial(_assemble_lanes, **kw)
        cache[lk] = functools.partial(_flux_seq_lean, rotate=bool(jra55.FORCE_ROTATION),
                                      assemble=jax.jit(asm) if out_sh is None else jax.jit(asm, out_shardings=out_sh),
                                      **kw)
    pk = ("lean_place", place)
    if pk not in cache:
        lanes = None
        if lanes_here:
            from ..zarr_output import _folded_gid_owned
            gid, _ = _folded_gid_owned(static.part, static.sm, "nod")
            lanes = (np.where(gid >= 0, gid, 0).astype(np.int32), gid < 0)
        sc = (np.asarray(LATENT_FUSION, np.float64), np.asarray(1000000.0, np.float64))
        if "clim" not in cache:
            cache["clim"] = (jnp.asarray(static.sss_clim), jnp.asarray(static.chl_clim))
        cache[pk] = jax.device_put((static.trig, static.M, cache["clim"], sc, lanes), place)
    trig, M, clim, sc, lanes = cache[pk]
    mk = ("lean_months", place, tuple(int(d[3]) for d in dates))
    if mk not in cache:
        cache[mk] = jax.device_put(np.array([d[3] for d in dates], np.int32) - 1, place)
    return cache[lk](fields, cache[mk], trig, M, *clim, sc, lanes)


def forcing_lanes_first_enabled() -> bool:
    """``gpu.cpl_forcing_lanes`` (``FESOM_CPL_FORCING_LANES=1``, dist_N with ``cpl_forcing_lean``; lane
    tco319_ng5_flux): the received fields gathered onto each device's lanes FIRST, then broadcast over the interval,
    rotated and assembled there. The lean build works on replicated global ``[nsteps, nod2D]`` arrays and gathers at
    the end, so every device holds the whole interval's forcing: at NG5 (7.4 M nodes, 20 steps, float64 rotation
    products) that is ~20 GB per device and more than 1 TB on 32 CPU devices (job 28945544). Every operation is
    element-wise per node and step, and the same stage programs run (products and sums apart), so the values are
    the lean build's to the bit (pad lanes 1.0 at the end, as there)."""
    import os
    return os.environ.get("FESOM_CPL_FORCING_LANES", "0").strip() not in ("", "0")


def _assemble_pad(fields, rot, enth, months, sss_clim, chl_clim, pad, **kw):
    """:func:`_assemble` on the lanes, then the pad lanes set to 1.0 (:func:`_assemble_lanes`' values)."""
    seq = _assemble(fields, rot, enth, months, sss_clim, chl_clim, **kw)
    return jax.tree.map(lambda a: jnp.where(pad, jnp.asarray(1, a.dtype), a), seq)


def _lanes_first_call(fields, static, dates, cache, kw):
    """:func:`forcing_lanes_first_enabled`: the lanes' tables (rotation trig, SSS / chlorophyll climatologies, pad
    mask) placed once, sharded over the lanes; per interval one gather of the received ``(nod2D,)`` fields onto the
    lanes, then :func:`_flux_seq_lean`'s stages on ``[nsteps, P*Lmax]`` sharded arrays."""
    import functools
    from jax.sharding import NamedSharding, PartitionSpec
    from .. import jra55
    mesh = static.device_mesh()
    sh_l, sh_tl = NamedSharding(mesh, PartitionSpec("p")), NamedSharding(mesh, PartitionSpec(None, "p"))
    rep = NamedSharding(mesh, PartitionSpec())
    if "lanes_first" not in cache:
        from ..zarr_output import _folded_gid_owned
        gid, _ = _folded_gid_owned(static.part, static.sm, "nod")
        idx = np.where(gid >= 0, gid, 0).astype(np.int64)
        tabs = (jax.device_put(idx.astype(np.int32), sh_l), jax.device_put(gid < 0, sh_l),
                tuple(jax.device_put(np.asarray(t)[idx], sh_l) for t in static.trig),
                jax.device_put(np.asarray(static.M), rep),
                jax.device_put((np.asarray(LATENT_FUSION, np.float64), np.asarray(1000000.0, np.float64)), rep),
                jax.device_put(np.asarray(static.sss_clim)[:, idx], sh_tl),
                jax.device_put(np.asarray(static.chl_clim)[:, idx], sh_tl),
                jax.jit(lambda f, i: jax.tree.map(lambda a: a[i], f), out_shardings=sh_l))
        cache["lanes_first"] = tabs
    idx_d, pad, trig_l, M, sc, sss_l, chl_l, gather = cache["lanes_first"]
    kw2 = dict(kw, n=int(idx_d.shape[0]))
    ak = ("lanes_first_asm",) + tuple(sorted(kw2.items()))
    if ak not in cache:
        cache[ak] = jax.jit(functools.partial(_assemble_pad, **kw2), out_shardings=sh_tl)
    mk = ("lanes_first_months", tuple(int(d[3]) for d in dates))
    if mk not in cache:
        cache[mk] = jax.device_put(np.array([d[3] for d in dates], np.int32) - 1, rep)
    fl = gather(fields, idx_d)
    return _flux_seq_lean(fl, cache[mk], trig_l, M, sss_l, chl_l, sc, pad, rotate=bool(jra55.FORCE_ROTATION),
                          assemble=cache[ak], **kw2)


def _flux_seq_staged(fields, months, trig, M, sss_clim, chl_clim, *, T, n, runoff, single, rotate, has_tsk,
                     assemble):
    """:func:`flux_step_forcing`'s operations in a few compiled programs: the rotation in the product / sum stages
    of :func:`_stages`, the rest (broadcasts, the runoff sum, the enthalpy cap, the month rows, the casts: no
    product) in ``assemble``."""
    m1, a1, m2, a2, m3, a3, enth_m = _stages()

    def r(a):
        return jnp.broadcast_to(a, (T, n)) if a.shape == (n,) else a

    rot = None
    if rotate:
        pairs = [(r(fields["taux_oce"]), r(fields["tauy_oce"])), (r(fields["taux_ico"]), r(fields["tauy_ico"]))]
        rot = a3(m3(a2(m2(a1(m1(pairs, trig)), M)), trig))
    enth = None
    if runoff == "coupled":
        calv = r(fields["calv_oce"])
        calv = calv.astype(jnp.float32) if single else calv
        enth = enth_m(calv, jnp.asarray(LATENT_FUSION, calv.dtype), jnp.asarray(1000000.0, calv.dtype))
    return assemble(fields, rot, enth, months, sss_clim, chl_clim)


def _assemble(fields, rot, enth, months, sss_clim, chl_clim, *, T, n, runoff, single, has_tsk):
    from .. import precision as PR

    def r(k):
        a = fields[k]
        return jnp.broadcast_to(a, (T, n)) if a.shape == (n,) else a

    if rot is None:
        tox, toy, tix, tiy = r("taux_oce"), r("tauy_oce"), r("taux_ico"), r("tauy_ico")
    else:
        (tox, toy), (tix, tiy) = rot
    tsk = r("tsk_ico") if has_tsk else jnp.zeros((T, n), tox.dtype)
    extra = {}
    if runoff == "coupled":
        hydr, calv = r("hydr_oce"), r("calv_oce")
        if single:
            hydr, calv = hydr.astype(jnp.float32), calv.astype(jnp.float32)
        extra["runoff"] = hydr + calv
        extra["enthalpy"] = -jnp.minimum(enth, jnp.asarray(ENTHALPY_CAP, calv.dtype))
    seq = CplForcing(taux_oce=tox, tauy_oce=toy, taux_ico=tix, tauy_ico=tiy,
                     prec_rain=r("prec_oce"), prec_snow=r("snow_oce"), evap=r("evap_oce"), subl=r("subl_oce"),
                     heat_oce=r("heat_oce"), heat_ico=r("heat_ico"), heat_swo=r("heat_swo"), tsk_ico=tsk,
                     Ssurf_month=sss_clim[months], chl=chl_clim[months], **extra)
    return PR.cast_tree(seq) if single else seq


def flux_step_forcing_jit(state, forcing: dict, static, dates):
    """:func:`flux_step_forcing` in a few compiled programs (:func:`_flux_seq_staged`, one set per interval length
    and field set); the climatologies stay on the device. Returns ``(state, seq)``, bitwise the eager build
    (checked on the GPU by ``scripts/core3/ocean_speed.py run --check-forcing``)."""
    import functools
    from .. import jra55
    n, T = static.nod2D, len(dates)
    need = ("taux_oce", "tauy_oce", "taux_ico", "tauy_ico", "prec_oce", "snow_oce", "evap_oce", "subl_oce",
            "heat_oce", "heat_ico", "heat_swo")
    missing = [k for k in need if k not in forcing]
    if missing:
        raise KeyError(f"mode F forcing lacks {missing}")
    runoff = getattr(static.flux, "runoff", "climatology")
    names = need + (("tsk_ico",) if "tsk_ico" in forcing else ()) + \
        (("hydr_oce", "calv_oce") if runoff == "coupled" else ())
    miss = [k for k in names if k not in forcing]
    if miss:
        raise KeyError(f"runoff 'coupled' needs {miss} (the runoff mapper's fields)")
    fields = {}
    for k in names:
        a = forcing[k]
        a = a if isinstance(a, jax.Array) else jnp.asarray(np.asarray(a))
        if a.shape not in ((n,), (T, n)):
            raise ValueError(f"forcing[{k!r}] has shape {a.shape}; need ({n},) or ({T}, {n})")
        fields[k] = a
    cache = static._fjit
    if "clim" not in cache:
        cache["clim"] = (jnp.asarray(static.sss_clim), jnp.asarray(static.chl_clim))
    key = (T, tuple((k, fields[k].shape, str(fields[k].dtype)) for k in names), runoff)
    kw = dict(T=T, n=n, runoff=runoff, single=static.precision == "single", has_tsk="tsk_ico" in forcing)
    if key not in cache:
        cache[key] = functools.partial(_flux_seq_staged, rotate=bool(jra55.FORCE_ROTATION),
                                       assemble=jax.jit(functools.partial(_assemble, **kw)), **kw)
    if forcing_lean_enabled():
        return state, _flux_lean_call(fields, static, dates, cache, key, kw)
    months = jnp.asarray(np.array([d[3] for d in dates]) - 1, jnp.int32)
    return state, cache[key](fields, months, static.trig, static.M, *cache["clim"])

def forcing_lean_enabled() -> bool:
    """``gpu.cpl_forcing_lean`` (``FESOM_CPL_FORCING_LEAN=1``, lane ocean_speed3; with ``cpl_forcing_jit``): the
    interval's forcing build without eager device ops and host-to-device copies: the broadcasts and casts inside the
    stage programs, the scalars and month rows copied to the device once, and (dist_N) the gather onto the lanes in
    the assembly program (:func:`_flux_seq_lean`). Eight compiled programs an interval and nothing else: less work
    for the ocean's Python thread, which shares the interpreter with the atmosphere's. The same operations on the
    same values (broadcasts, casts, gathers and selects moved between programs; no product meets a sum)."""
    import os
    return os.environ.get("FESOM_CPL_FORCING_LEAN", "0").strip() not in ("", "0")


def _lean_stages():
    """:func:`_stages`' m1 and enth_m with the broadcast to ``[T, n]`` (and the cast to float32) inside."""
    g = globals()
    if "_LEAN" in g:
        return g["_LEAN"]
    m1, a1, m2, a2, m3, a3, enth_m = _stages()

    @functools.partial(jax.jit, static_argnames=("T",))
    def m1b(pairs, trig, T):
        def r(a):
            return jnp.broadcast_to(a, (T, a.shape[-1])) if a.ndim == 1 else a
        sgl, cgl, sgo, cgo = trig[:4]
        pairs = [(r(u), r(v)) for u, v in pairs]
        return [((-v) * sgl * cgo, u * sgo, (-v) * sgl * sgo, u * cgo, v * cgl) for u, v in pairs]

    @functools.partial(jax.jit, static_argnames=("T", "single"))
    def enth_mb(calv, lf, mega, T, single):
        calv = jnp.broadcast_to(calv, (T, calv.shape[-1])) if calv.ndim == 1 else calv
        calv = calv.astype(jnp.float32) if single else calv
        return (calv * lf.astype(calv.dtype)) * mega.astype(calv.dtype)

    g["_LEAN"] = (m1b, a1, m2, a2, m3, a3, enth_mb)
    return g["_LEAN"]


def _flux_seq_lean(fields, months, trig, M, sss_clim, chl_clim, scalars, lanes, *, T, n, runoff, single, rotate,
                   has_tsk, assemble):
    """:func:`_flux_seq_staged` with no eager op: see :func:`forcing_lean_enabled`. ``lanes``: None, or
    ``(idx, pad)`` of the dist_N lanes (gathered in ``assemble``)."""
    m1b, a1, m2, a2, m3, a3, enth_mb = _lean_stages()
    rot = None
    if rotate:
        pairs = [(fields["taux_oce"], fields["tauy_oce"]), (fields["taux_ico"], fields["tauy_ico"])]
        rot = a3(m3(a2(m2(a1(m1b(pairs, trig, T=T)), M)), trig))
    enth = None
    if runoff == "coupled":
        lf, mega = scalars
        enth = enth_mb(fields["calv_oce"], lf, mega, T=T, single=single)
    return assemble(fields, rot, enth, months, sss_clim, chl_clim, lanes)


def _assemble_lanes(fields, rot, enth, months, sss_clim, chl_clim, lanes, **kw):
    """:func:`_assemble`, then (``lanes`` given) each leaf gathered onto the dist_N lanes as
    :func:`fesom_jax.coupled.ocean.to_lanes` does (halo lanes their node's value, pad lanes 1.0)."""
    seq = _assemble(fields, rot, enth, months, sss_clim, chl_clim, **kw)
    if lanes is None:
        return seq
    idx, pad = lanes
    return jax.tree.map(lambda a: jnp.where(pad, jnp.asarray(1, a.dtype), a[:, idx]), seq)


#: namcouple (AWI-ESM3 v3.5.1): the atmosphere's OASIS name -> the FESOM2 name it arrives under (cpl_driver cpl_recv)
OASIS_TO_FESOM = {"A_Qns_oce": "heat_oce", "A_Qs_all": "heat_swo", "A_Precip_liquid": "prec_oce",
                  "A_Precip_solid": "snow_oce", "A_Evap": "evap_oce", "A_Subl": "subl_oce", "A_Q_ice": "heat_ico",
                  "A_Ice_tskin": "tsk_ico", "A_TauX_oce": "taux_oce", "A_TauY_oce": "tauy_oce",
                  "A_TauX_ice": "taux_ico", "A_TauY_ice": "tauy_ico", "A_WindX": "u10w_oce", "A_WindY": "v10w_oce",
                  "A_Runoff": "hydr_oce", "A_Calving": "calv_oce", "R_Runoff_oce": "hydr_oce",
                  "R_Calving_oce": "calv_oce"}


def oasis_to_fesom(fields: dict) -> dict:
    """The receive side of FESOM2's coupling (cpl_driver.F90 cpl_oasis3mct_recv + gen_forcing_couple.F90
    update_atm_forcing, ``__oasis``/``__oifs``) up to the arrays the ocean step reads, for fields given by their
    namcouple OASIS names (``A_Qns_oce`` ...) or their FESOM2 names (``heat_oce`` ...); returns mode F's keys.

    What FESOM2 does on receive, and so does this function:

    * renames (cpl_recv order; the namcouple pairs in :data:`OASIS_TO_FESOM`);
    * NO unit or sign change: ``prec_rain(:) = exchange(:)`` etc. FESOM2 uses every field as delivered. The
      atmosphere's put side (fork ``ece_fesom_set_ocean_fluxes.F90``) already sends the water fluxes in m s-1
      (``PFPLCL/RHOH2O``, ``PEVAPTI/RHOH2O``, runoff ``/1000``), heat in W m-2 positive down, stresses in N m-2
      geographic. A field put in kg m-2 s-1 is a put-side bug, not something FESOM2 converts;
    * no ice/ocean fraction weighting on receive (FESOM2 weights evap by 1 - A and subl by A inside
      ice_thermo_cpl, i.e. in :func:`thermo_cpl`), no conservation correction (``force_flux_consv`` returns at
      once under ``__oifs``: OASIS's GLBPOS/GSSMAR do it);
    * the stress rotation geographic -> model frame happens later in update_atm_forcing, here in
      :func:`flux_step_forcing` (``ocean_interval``), so the returned stresses stay geographic.

    Unknown names raise; ``hydr_oce``/``calv_oce``/``u10w_oce``/``v10w_oce`` are passed through (accepted and not
    used by the ocean step, see the module docstring)."""
    known = set(OASIS_TO_FESOM.values())
    out = {}
    for k, v in fields.items():
        name = OASIS_TO_FESOM.get(k, k)
        if name not in known:
            raise KeyError(f"{k!r} is neither a namcouple OASIS name nor a FESOM2 receive name")
        if name in out:
            raise KeyError(f"{name!r} given twice (as {k!r})")
        out[name] = v
    return out


def dummy_seq(nsteps: int, n: int, npes: int = 1, runoff: str = "climatology"):
    """A host :class:`CplForcing` of zeros ``[npes, nsteps, n]`` (the partitioned shape the runner folds; ``n`` =
    the lanes per device) to build the interval program (``runoff = "coupled"``: with the runoff fields)."""
    z = np.zeros((npes, nsteps, n), np.float64)
    nf = len(CplForcing._fields) - (0 if runoff == "coupled" else 2)
    return CplForcing(*([z] * nf))


def ice_temp_alb(state, static):
    """ice_temp [K] and ice_alb as the last coupled thermodynamics left them (what FESOM2 sends)."""
    return state.cpl_ice_temp, state.cpl_ice_alb


# ---------------------------------------------------------------------------------------------------------------
# ice_thermo_cpl.F90
# ---------------------------------------------------------------------------------------------------------------
def _tfrez(S):
    return -0.0575 * S + 1.7105e-3 * jnp.sqrt(jnp.maximum(S, 0.0) ** 3) - 2.155e-4 * (S * S)


def _pond_albedo(hsn, apnd, ipnd, alb_dry, pp: MeltPondParams):
    """meltpond_albedo(aice = 1, hsno, apnd, ipnd, t, alb_dry, alb_dry) of ice_thermo_cpl's ice_albedo."""
    albw = 0.066
    albedo_ice = alb_dry                      # albsn_dry = albi_dry = alb_noponds
    albedo_pond = jnp.where(ipnd > 0.01, pp.albpnd_frz, pp.albpnd)
    frac_pond = jnp.minimum(apnd, 1.0)
    frac_ice = 1.0 - frac_pond
    return jnp.maximum(albw, jnp.minimum(0.9, frac_ice * albedo_ice + frac_pond * albedo_pond))


def _sqrt0(x):
    """``sqrt(x)`` for ``x >= 0`` with a finite (zero) derivative at ``x == 0``: the double where. The primal is
    bitwise ``sqrt(x)``; plain ``sqrt`` stores 0.5/sqrt(0) = Inf, which a discarded branch's zero cotangent turns
    into NaN in the reverse pass (lane coupled_grad2: 94,446 nodes of the coupled CORE2 step, _pond_area's a_new)."""
    pos = x > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0 * x)


def _pond_area(aice, hice, hsno, meltt, melts, frain, tsfc_k, fsurfn, dt, apnd, hpnd, ipnd, pp: MeltPondParams):
    """ice_meltponds.F90 ``meltpond_area`` (element-wise, every branch as the Fortran)."""
    puny = pp.puny
    skip = (aice < puny) | (hice < pp.hi_min) | (hsno > pp.hs1)
    hi = hice / jnp.maximum(aice, puny)
    hs = hsno / jnp.maximum(aice, puny)
    armor = jnp.minimum(1.0, hi / 0.50)
    rfrac = pp.rfracmin + (pp.rfracmax - pp.rfracmin) * armor
    volpn_init = apnd * hpnd
    hlid = ipnd
    dvn = rfrac / pp.rhofresh * (meltt * pp.rhoice_p + melts * pp.rhosno_p + frain * pp.rhofresh) * dt * aice
    ts = tsfc_k - 273.15
    # dvn <= 0: the lid freezes when Ts < 0
    bdt = -2.0 * ts * pp.kice * dt / (pp.rhoice_p * pp.lfresh)
    d_open = 0.5 * _sqrt0(jnp.maximum(bdt, 0.0))
    d_cond = 0.5 * bdt / jnp.where(hlid > 0.0, hlid, 1.0)
    d_frz = jnp.where(hlid > d_open, d_cond, d_open)
    d_frz = jnp.minimum(d_frz, hpnd * pp.rhofresh / pp.rhoice_p)
    d_frz = jnp.where(ts < 0.0, d_frz, 0.0)
    # dvn > 0: the lid melts
    d_mlt = jnp.maximum(fsurfn * dt / (pp.rhoice_p * pp.lfresh), 0.0)
    d_mlt = -jnp.minimum(d_mlt, hlid)
    grow = dvn <= 0.0
    dhlid = jnp.where(grow, d_frz, d_mlt)
    hlid = jnp.where(grow, hlid + dhlid, jnp.maximum(hlid + dhlid, 0.0))
    dvn = dvn - dhlid * apnd * pp.rhoice_p / pp.rhofresh
    volpn = volpn_init + dvn
    gone = volpn <= 0.0
    has_p = apnd > puny
    a_ex = jnp.maximum(0.0, jnp.minimum(aice * 0.9, apnd + 0.5 * dvn * aice / (pp.pndaspect * jnp.where(has_p, apnd, 1.0))))
    h_ex = jnp.where(a_ex > puny, volpn / jnp.where(a_ex > puny, a_ex, 1.0), 0.0)
    a_new = jnp.maximum(jnp.minimum(_sqrt0(jnp.maximum(volpn, 0.0) * aice / pp.pndaspect), aice * 0.9), puny)
    h_new = volpn / jnp.maximum(a_new, puny)
    big = aice > 10.0 * puny
    apondn = jnp.where(gone, 0.0, jnp.where(has_p, a_ex, jnp.where(big, a_new, 0.0)))
    hpondn = jnp.where(gone, 0.0, jnp.where(has_p, h_ex, jnp.where(big, h_new, 0.0)))
    hlid = jnp.where(gone, 0.0, hlid)
    fbcap = ((pp.rhowat_p - pp.rhoice_p) * hi - pp.rhosno_p * hs) / pp.rhofresh
    keep = apondn > puny
    hpondn = jnp.where(keep, jnp.minimum(hpondn, jnp.maximum(fbcap, 0.0)), 0.0)
    apondn = jnp.where(keep, apondn, 0.0)
    hlid = jnp.where(keep, hlid, 0.0)
    hlid = jnp.minimum(hlid, hpondn)
    z = jnp.zeros_like(apnd)
    return jnp.where(skip, z, apondn), jnp.where(skip, z, hpondn), jnp.where(skip, z, hlid)


class CplThermoOut(NamedTuple):
    a_ice: jnp.ndarray
    m_ice: jnp.ndarray
    m_snow: jnp.ndarray
    flx_h: jnp.ndarray           # ehf (+ down) = net_heat_flux
    flx_fw: jnp.ndarray          # fw [m s-1] = fresh_wa_flux
    real_salt_flux: jnp.ndarray
    thdgr: jnp.ndarray
    thdgrsn: jnp.ndarray
    evaporation: jnp.ndarray     # evap*(1-A) + subli*A
    ice_sublimation: jnp.ndarray  # subli*A
    resid: jnp.ndarray           # residualifwflx
    ice_temp: jnp.ndarray
    ice_alb: jnp.ndarray
    apnd: jnp.ndarray
    hpnd: jnp.ndarray
    ipnd: jnp.ndarray


def thermo_cpl(cfg, fc: FluxConfig, *, A, h, hsn, ice_temp_old, apnd, hpnd, ipnd, T_oc, S_oc, cf: CplForcing,
               runo, geo_lat, non_cavity, dt, use_virt_salt: bool) -> CplThermoOut:
    """ice_thermo_cpl.F90 ``thermodynamics`` (``__oifs``) over all nodes, element-wise; the cavity skip as the
    Fortran (``ulevels_nod2D > 1``: unchanged, zero fluxes)."""
    pp = fc.ponds
    rhoice, rhosno, rhowat, rhofwt = cfg.rhoice, cfg.rhosno, cfg.rhowat, cfg.rhofwt
    con, consn, Sice = cfg.con, cfg.consn, cfg.Sice
    cl, cc = cfg.cl, cfg.cc
    h0min = jnp.where(geo_lat > 0.0, cfg.h0, cfg.h0_s)
    a2ohf = cf.heat_oce + cf.heat_swo            # + enthalpyoffuse (runoff "coupled"; else 0, plan decision 1)
    if cf.enthalpy is not None:
        a2ohf = a2ohf + cf.enthalpy
    a2ihf = cf.heat_ico
    evap, subli, rain, snow = cf.evap, cf.subl, cf.prec_rain, cf.prec_snow
    rsss = S_oc                                  # ref_sss_local
    A_in, h_in, hsn_in = A, h, hsn

    # --- surface temperature (ice_surftemp, use_atm_ice_tskin) and albedo, before the growth ---
    tref = cf.tsk_ico
    tref = jnp.where((tref < 173.15) | (tref > 400.0), 271.35, tref)
    if not fc.use_atm_ice_tskin:
        raise NotImplementedError("only use_atm_ice_tskin (the AWI-ESM3 v3.5.1 setting) is ported")
    t_ice = jnp.where(tref > 273.15, 273.15, tref)
    on = A > AIMIN
    t = jnp.where(on, t_ice, ice_temp_old)        # A <= Aimin: t keeps the old ice_temp (Fortran's local t)
    ice_temp = jnp.where(on, t_ice, _tfrez(S_oc) + TMELT)
    alb_tramp = float(getattr(cfg, "alb_tramp", 0.0))
    h_snowscale = float(getattr(cfg, "h_snowscale", 0.0))
    if alb_tramp > 0.0:
        # FESOM main ice_albedo: linear frozen -> melting over alb_tramp K below the melting point
        fmelt = jnp.minimum(1.0, jnp.maximum(0.0, (t - 273.15 + alb_tramp) / alb_tramp))
        alb_snow = cfg.albsn + fmelt * (cfg.albsnm - cfg.albsn)
        alb_bare = cfg.albi + fmelt * (cfg.albim - cfg.albi)
    else:
        frozen = t < 273.15
        alb_snow = jnp.where(frozen, cfg.albsn, cfg.albsnm)
        alb_bare = jnp.where(frozen, cfg.albi, cfg.albim)
    if h_snowscale > 0.0:
        # FESOM main ice_albedo: smooth snow/bare blend, alpha_snow = tanh(hsn/h_snowscale)
        alpha_snow = jnp.tanh(hsn / h_snowscale)
        alb_np = alpha_snow * alb_snow + (1.0 - alpha_snow) * alb_bare
    else:
        alb_np = jnp.where(hsn > 0.001, alb_snow, alb_bare)
    alb0 = jnp.where(h > 0.0, alb_np, 0.066)
    if fc.use_meltponds:
        alb = jnp.where(h > 0.0, _pond_albedo(hsn, apnd, ipnd, alb0, pp), alb0)
    else:
        alb = alb0

    # --- ice_growth ---
    Tfrezs = _tfrez(S_oc)
    heff = (h + hsn * con / consn) / jnp.maximum(A, AIMIN)
    Qicecon = (Tfrezs - 0.0) * con / jnp.maximum(heff, HIMIN)
    Qatmice = -a2ihf
    Qatmocn = -a2ohf
    Qocnice = (T_oc - Tfrezs) * GAMMA_T * cc
    Qocnatm = jnp.minimum(Qocnice, Qatmocn)
    ahf = A * Qatmice + (1.0 - A) * Qatmocn
    Qicecon = Qicecon * dt / cl
    Qatmice = Qatmice * dt / cl
    Qatmocn = Qatmocn * dt / cl
    Qocnice = Qocnice * dt / cl
    Qocnatm = Qocnatm * dt / cl
    subli = A * subli
    evap = (1.0 - A) * evap
    PmEice = (A * snow + subli) * dt
    PmEocn = (evap + rain + (1.0 - A) * snow + runo) * dt
    hsn = hsn + PmEice * rhofwt / rhosno
    neg = hsn < 0.0
    PmEice = jnp.where(neg, hsn * rhosno / rhofwt, 0.0)
    hsn = jnp.where(neg, 0.0, hsn)
    h = h + PmEice * rhofwt / rhoice
    neg = h < 0.0
    PmEice = jnp.where(neg, h * rhoice / rhofwt, 0.0)
    h = jnp.where(neg, 0.0, h)
    resid = PmEice / dt
    PmEocn = PmEocn + PmEice
    hsnold, hold = hsn, h
    if fc.snowmelt_tgate:
        gate = t > 273.0
    else:
        gate = jnp.ones_like(t, dtype=bool)
    dsnow = jnp.maximum(A * jnp.minimum(Qatmice - Qicecon, 0.0) * rhoice / rhosno, -hsn)
    dsnow = jnp.where(gate, dsnow, 0.0)
    hsn = hsn + dsnow
    dhice = A * (Qatmice - Qocnice) - dsnow * rhosno / rhoice
    dhiow = (1.0 - A) * jnp.maximum(Qatmocn - Qocnatm, 0.0)
    htmp = h + dhice + dhiow
    allm = htmp < 0.0
    hsn = jnp.where(allm, hsn + jnp.maximum(htmp * rhoice / rhosno, -hsn), hsn)
    h = jnp.where(allm, 0.0, htmp)
    h = jnp.where(h < HCUTOFF, 0.0, h)
    htmp = jnp.maximum(hold, HCUTOFF)
    dcice = 0.5 * A * jnp.minimum(dhice, 0.0) / htmp
    dslat = jnp.maximum(jnp.minimum(dcice * (hsnold / jnp.maximum(A, AIMIN)) - dsnow, 0.0), -hsn)
    dslat = jnp.where(A <= 0.0, -hsn, dslat)
    hsn = hsn + dslat
    h0cur = jnp.maximum(h0min, jnp.minimum(H0MAX, hold))
    dciow = jnp.maximum(dhiow, 0.0) / h0cur
    Anew = A + dcice + dciow
    Anew = jnp.minimum(Anew, h * BIGVAL)
    Anew = jnp.maximum(0.0, jnp.minimum(1.0, Anew))
    dhsngrowth = (hsn - hsnold) / dt
    dhgrowth = (h - hold) / dt
    PmEocn = PmEocn / dt
    if not use_virt_salt:
        fwice = -dhgrowth * rhoice
        fwsnw = -dhsngrowth * rhosno
        fw = PmEocn * rhofwt + fwice + fwsnw
        rsf = fwice * Sice / rhowat
    else:
        fwice = -dhgrowth * rhoice * (rsss - Sice) / rsss
        fwsnw = -dhsngrowth * rhosno
        fw = PmEocn * rhofwt + fwice + fwsnw
        rsf = jnp.zeros_like(fw)
    ehf = -ahf + cl * (dhgrowth + dhsngrowth * rhosno / rhoice)
    # melt ponds (new A, h, hsn before flooding)
    if fc.use_meltponds:
        meltt = jnp.maximum(0.0, -dhgrowth)
        melts = jnp.maximum(0.0, -dhsngrowth)
        pa, ph, pi = _pond_area(Anew, h, hsn, meltt, melts, rain, t, a2ihf, dt, apnd, hpnd, ipnd, pp)
        pond_on = Anew > AIMIN
        z = jnp.zeros_like(apnd)
        apnd_n, hpnd_n, ipnd_n = (jnp.where(pond_on, pa, z), jnp.where(pond_on, ph, z), jnp.where(pond_on, pi, z))
    else:
        apnd_n = hpnd_n = ipnd_n = jnp.zeros_like(apnd)
    # flooding
    htmp = h
    hdraft = (h * rhoice + hsn * rhosno) / rhowat
    hflood = hdraft - jnp.minimum(h, hdraft)
    h = h + hflood
    hsn = hsn - hflood * rhoice / rhosno
    dhflice = (h - htmp) / dt
    if not use_virt_salt:
        rsf = rsf - dhflice * rhoice * Sice / rhowat
    else:
        fw = fw + dhflice * rhoice * Sice / rsss
    fw = fw / rhowat

    keep = non_cavity
    sel = lambda new, old: jnp.where(keep, new, old)       # noqa: E731
    zero = lambda x: jnp.where(keep, x, 0.0)               # noqa: E731
    return CplThermoOut(
        a_ice=sel(Anew, A_in), m_ice=sel(h, h_in), m_snow=sel(hsn, hsn_in),
        flx_h=zero(ehf), flx_fw=zero(fw), real_salt_flux=zero(rsf), thdgr=zero(dhgrowth), thdgrsn=zero(dhsngrowth),
        evaporation=zero(evap + subli), ice_sublimation=zero(subli), resid=zero(resid),
        ice_temp=sel(ice_temp, ice_temp_old), ice_alb=alb, apnd=apnd_n, hpnd=hpnd_n, ipnd=ipnd_n)


# ---------------------------------------------------------------------------------------------------------------
# cavities: what FESOM main 0aaa2d73 does with the received fields at ice-shelf cavity nodes (coupled, __oifs)
# ---------------------------------------------------------------------------------------------------------------
#: the switch (trace time): "1" when the coupled ocean runs with cavities (init_ocean's ``cpl_cavity``, set from the
#: ocean's cavity switch); unset or "0" = today's program
CAVITY_ENV = "FESOM_CPL_CAVITY"
#: lane L-OCN's three-equation melt (fesom_jax.cavity.cavity_heat_water_fluxes_3eq), called where FESOM2's oce_fluxes
#: calls it: fn(state, mesh, srf, heat_flux, water_flux) -> (heat_flux, water_flux), overwriting the cavity nodes.
#: None (default): no melt here (the hook point for the wiring; the rest of the cavity branches below do not need it)
CAVITY_MELT = None


def cavity_on() -> bool:
    import os
    return os.environ.get(CAVITY_ENV, "0") == "1"


def top_wet(x, ulevels_nod2D):
    """x[n, ulevels_nod2D(n) - 1]: a (nod2D, nl) field at each node's top wet level (cavity nodes: under the shelf)."""
    return jnp.take_along_axis(x, (jnp.asarray(ulevels_nod2D) - 1)[:, None].astype(jnp.int32), axis=1)[:, 0]


def cavity_virtual_salt(water_flux, S_topwet, areasvol_top, ocean_area, open_water, *, use_virt_salt,
                        owned_mask=None, axis_name=None):
    """oce_fluxes' virtual salt with cavities (ice_oce_coupling.F90:375-438): rsss = S at the top wet level
    (ref_sss_local), integrate_nod weights areasvol at ulevels_nod2D, the balance subtracted on open-ocean nodes
    only (mesh%ocean_area = the open ocean). linfs (use_virt_salt): rsss*water_flux everywhere; zstar: only under the
    shelves (rigid lid, nothing moves there), counter-balanced over the open ocean."""
    from ..sss_runoff import _area_mean
    vs = S_topwet * water_flux
    if not use_virt_salt:
        vs = jnp.where(open_water, 0.0, vs)
    net = _area_mean(vs, areasvol_top, ocean_area, owned_mask=owned_mask, axis_name=axis_name)
    return jnp.where(open_water, vs - net, vs)


def cavity_fw_balance_zstar(water_flux, flux, areasvol_top, ocean_area, open_water, *, owned_mask=None,
                            axis_name=None):
    """oce_fluxes' freshwater balance with cavities and zstar (ice_oce_coupling.F90:566-598): no flux under the
    shelves (flux = 0 there), integrate_nod with areasvol at ulevels_nod2D, net/ocean_area added on open-ocean nodes
    only. (linfs: water_flux is not read after oce_fluxes in fesom_jax, so its balance is not needed.)"""
    from ..sss_runoff import _area_mean
    flux = jnp.where(open_water, flux, 0.0)
    net = _area_mean(flux, areasvol_top, ocean_area, owned_mask=owned_mask, axis_name=axis_name)
    return jnp.where(open_water, water_flux + net, water_flux)


def cavity_stress(stress_surf, stress_node_surf, open_water, elem_open):
    """oce_fluxes_mom (ice_oce_coupling.F90:72-117): cavity nodes and cavity elements (ulevels > 1) take no
    atmosphere or ice-ocean stress: 0 here, left to cavity_momentum_fluxes (lane L-OCN), which FESOM2 calls right
    after. An open element (ulevels = 1) never has a cavity node (ulevels_nod2D is the minimum over the node's
    elements), so the open elements' averages are unchanged."""
    sns = jnp.where(open_water[:, None], stress_node_surf, 0.0)
    ss = jnp.where(elem_open[:, None], stress_surf, 0.0)
    return ss, sns


# ---------------------------------------------------------------------------------------------------------------
# the coupled ice step (the twin of fesom_jax.ice_step.ice_surface_step, ice_ave_steps = 1)
# ---------------------------------------------------------------------------------------------------------------
def ice_surface_step_cpl(cfg, mesh, state, cf: CplForcing, fs, *, dt: float, boundary_node=None,
                         use_virt_salt: bool = True, owned_mask=None, axis_name=None, exch=None, zbar3=None,
                         params=None, mevp_wide=None, ice_update=None, legacy_sw_depth: bool = False,
                         cavity_cfg=None):
    """FESOM2's coupled ice block for one ocean step: ocean2ice -> EVP/mEVP with the received ice stress -> FCT ->
    cut_off -> :func:`thermo_cpl` -> oce_fluxes (coupled) -> oce_fluxes_mom with the received ocean stress ->
    cal_shortwave_rad of ``heat_swo``. Returns an :class:`fesom_jax.ice_step.IceStepOut` whose ``cpl`` carries the
    :data:`CPL_FIELDS`.

    ``cavity_cfg`` (lane L-OCN, ``fesom_jax.cavity``; passed by ``step`` when the run config has ``cavity:``): turns
    the cavity branches on (as :data:`CAVITY_ENV`), clears the sea ice under the shelves after the EVP and after the
    thermodynamics (ice_timestep), makes the shelf front an ice coast, calls the three-equation melt at the
    :data:`CAVITY_MELT` point (unless a hook is set there) and applies the shelf-base drag after :func:`cavity_stress`
    (oce_fluxes_mom -> cavity_momentum_fluxes). ``None`` = the program as before."""
    from ..ice_step import IceStepOut, _with_ice_dt
    if int(cfg.ice_ave_steps) != 1:
        raise NotImplementedError("mode F runs the sea ice every ocean step (ice_ave_steps = 1)")
    cfg = _with_ice_dt(cfg, dt)
    fc = flux_config(cfg)
    open_water = fs.open_water
    geo_lat = jnp.asarray(mesh.geo_coord_nod2D)[:, 1]
    srf = ice_coupling.ocean2ice(state)
    if cavity_cfg is not None:
        from .. import cavity as _cav
        if boundary_node is None:                                 # drivers pass the global mask, already extended
            boundary_node = _cav.cavity_boundary_node_mask(mesh, ice_evp.boundary_node_mask(mesh))
    _evp = ice_mevp.mevp_dynamics if cfg.whichEVP == 1 else ice_evp.evp_dynamics
    kw = {} if mevp_wide is None else {"wide": mevp_wide}
    _exch = (lambda f, k: f) if exch is None else exch          # noqa: E731
    # IceConfig.adjoint_mode in the coupled ice block, as fesom_jax.ice_step.ice_surface_step reads it (the same
    # switches, backward only; "exact" with the default adjoint_sweeps = -1 is the unchanged program):
    #   "rheology"    the mEVP subcycles are not unrolled in the backward (adjoint_subcycles 0): the ice velocity
    #                 carries no tangent, thermodynamics / advection / cut-off / ocean-to-ice stay differentiated;
    #   "prescribed"  the ice trajectory is given: the nine prognostic ice fields (and the coupled ice_temp, albedo,
    #                 ponds) carry no tangent into the step nor out of it, every flux stays differentiable;
    #   adjoint_sweeps >= 0 (with "exact"): only the last N subcycles unrolled.
    # "frozen" is handled by fesom_jax.step for either ice block; "dynamics" and adjoint_cuts are not wired here.
    mode = getattr(cfg, "adjoint_mode", "exact")
    if mode == "dynamics" or getattr(cfg, "adjoint_cuts", ()):
        raise NotImplementedError("the coupled ice block: adjoint_mode 'dynamics' / adjoint_cuts are not wired")
    _sg = jax.lax.stop_gradient
    prescribed = mode == "prescribed"
    if prescribed:
        from ..ice_step import _ICE_PROGNOSTIC
        state = dataclasses.replace(state, **{f: _sg(getattr(state, f)) for f in _ICE_PROGNOSTIC + CPL_FIELDS
                                              if getattr(state, f, None) is not None})
    if mode in ("rheology", "prescribed"):
        kw["adjoint_subcycles"] = 0
    elif getattr(cfg, "adjoint_sweeps", -1) >= 0:
        kw["adjoint_subcycles"] = int(cfg.adjoint_sweeps)
    u_ice, v_ice, s11, s12, s22 = _evp(
        cfg, mesh, a_ice=state.a_ice, m_ice=state.m_ice, m_snow=state.m_snow, u_ice=state.u_ice,
        v_ice=state.v_ice, sigma11=state.sigma11, sigma12=state.sigma12, sigma22=state.sigma22,
        srfoce_u=srf.u, srfoce_v=srf.v, elevation=srf.ssh, stress_ax=cf.taux_ico, stress_ay=cf.tauy_ico,
        boundary_node=boundary_node, exch=exch, **kw)
    if cavity_cfg is not None:                                    # cavity_ice_clean_vel after the EVP
        u_ice, v_ice = _cav.cavity_ice_clean_vel(mesh, u_ice, v_ice)
    a_adv, m_adv, ms_adv = ice_adv.fct_solve(cfg, mesh, state.a_ice, state.m_ice, state.m_snow, u_ice, v_ice,
                                             exch=exch)
    a_co, m_co, ms_co = ice_thermo.cut_off(a_adv, m_adv, ms_adv)
    runoff = fs.runoff_node if cf.runoff is None else cf.runoff     # the climatology, or hydr_oce + calv_oce
    th = thermo_cpl(cfg, fc, A=a_co, h=m_co, hsn=ms_co, ice_temp_old=state.cpl_ice_temp, apnd=state.cpl_apnd,
                    hpnd=state.cpl_hpnd, ipnd=state.cpl_ipnd, T_oc=srf.temp, S_oc=srf.salt, cf=cf,
                    runo=runoff, geo_lat=geo_lat, non_cavity=open_water, dt=cfg.ice_dt,
                    use_virt_salt=use_virt_salt)
    a_out, m_out, ms_out = _exch(th.a_ice, "nod"), _exch(th.m_ice, "nod"), _exch(th.m_snow, "nod")
    if prescribed:                                # the exit half: the updated ice leaves without a tangent
        a_out, m_out, ms_out, u_ice, v_ice, s11, s12, s22 = (_sg(x) for x in (a_out, m_out, ms_out, u_ice, v_ice,
                                                                               s11, s12, s22))
    if cavity_cfg is not None:                                    # cavity_ice_clean_ma at the end of ice_timestep
        a_out, m_out, ms_out = _cav.cavity_ice_clean_ma(mesh, a_out, m_out, ms_out)

    # oce_fluxes: heat_flux = -ehf, water_flux = -fw; virtual salt / SSS restoring as the standalone helper
    icef = ice_coupling.ice_oce_fluxes(srf.salt, th.flx_fw, th.flx_h, cf.Ssurf_month, runoff,
                                       fs.areasvol_surf, fs.ocean_area, open_water, owned_mask=owned_mask,
                                       axis_name=axis_name, use_virt_salt=use_virt_salt,
                                       real_salt_flux=th.real_salt_flux)
    water_flux = icef.water_flux
    cav = cavity_on() or cavity_cfg is not None
    heat_flux_in, virtual_salt, relax_salt = icef.heat_flux, icef.virtual_salt, icef.relax_salt
    if cav:
        # the received fields at cavity nodes (FESOM main 0aaa2d73, __oifs): thermo_cpl skips the nodes (heat, water,
        # runoff, enthalpy and the ice fluxes give 0 there: heat_flux = -ehf, water_flux = -fw), the melt overwrites
        # them (CAVITY_MELT), then the salt and water balances take the cavity branches
        ul = mesh.ulevels_nod2D
        # areasvol at the top wet level: the forcing's own surface areas on open nodes (fs.areasvol_surf, the
        # balances' weights without cavities), the level under the shelf at cavity nodes
        areasvol_top = jnp.where(open_water, fs.areasvol_surf,
                                 top_wet(jnp.asarray(mesh.areasvol), ul).astype(fs.areasvol_surf.dtype))
        S_topwet = top_wet(state.S, ul)
        _melt = CAVITY_MELT if CAVITY_MELT is not None else (
            None if cavity_cfg is None else _cav.melt_fn(cavity_cfg))
        if _melt is not None:
            heat_flux_in, water_flux = _melt(state, mesh, srf, heat_flux_in, water_flux)
        virtual_salt = cavity_virtual_salt(water_flux, S_topwet, areasvol_top, fs.ocean_area, open_water,
                                           use_virt_salt=use_virt_salt, owned_mask=owned_mask, axis_name=axis_name)
        relax_salt = jnp.where(open_water, relax_salt, 0.0)       # no SSS restoring under the shelves (rate 0 here)
    if not use_virt_salt and cav:
        flux = (th.evaporation - th.ice_sublimation + cf.prec_rain + cf.prec_snow * (1.0 - a_co) + th.resid
                + runoff) * (cfg.rhofwt / cfg.rhowat)
        flux = flux - th.thdgr * cfg.rhoice / cfg.rhowat - th.thdgrsn * cfg.rhosno / cfg.rhowat
        water_flux = cavity_fw_balance_zstar(water_flux, flux, areasvol_top, fs.ocean_area, open_water,
                                             owned_mask=owned_mask, axis_name=axis_name)
    elif not use_virt_salt:
        # the coupled zstar balance (ice_oce_coupling.F90 oce_fluxes, __oasis): + residualifwflx, x rhofwt/rhowat
        flux = (th.evaporation - th.ice_sublimation + cf.prec_rain + cf.prec_snow * (1.0 - a_co) + th.resid
                + runoff) * (cfg.rhofwt / cfg.rhowat)
        flux = flux - th.thdgr * cfg.rhoice / cfg.rhowat - th.thdgrsn * cfg.rhosno / cfg.rhowat
        from ..sss_runoff import _area_mean
        water_flux = water_flux + _area_mean(flux, fs.areasvol_surf, fs.ocean_area, owned_mask=owned_mask,
                                             axis_name=axis_name)
    stress_atm = jnp.stack([cf.taux_oce, cf.tauy_oce], axis=-1)
    stress_surf, stress_node_surf = ice_coupling.ice_oce_fluxes_mom(
        mesh, a_out, u_ice, v_ice, srf.u, srf.v, stress_atm, open_water, cfg)
    if cav:
        stress_surf, stress_node_surf = cavity_stress(stress_surf, stress_node_surf, open_water,
                                                      jnp.asarray(mesh.ulevels) <= 1)
        if cavity_cfg is not None:                                # cavity_momentum_fluxes: the shelf-base drag
            from ..config import C_D as _C_D, DENSITY_0 as _RHO0
            stress_surf, stress_node_surf = _cav.cavity_momentum_fluxes(
                mesh, state.uv, state.uvnode, stress_surf, stress_node_surf, _RHO0, _C_D)
    pene_open = open_water & (a_out <= 0.0)
    heat_flux, sw_3d = _forcing.cal_shortwave_rad(mesh, heat_flux_in, cf.heat_swo, cf.chl, pene_open,
                                                  zbar3=zbar3, legacy_depth=legacy_sw_depth)
    is_nonlinfs = 0.0 if use_virt_salt else 1.0
    bc_T = -dt * heat_flux / VCPW
    bc_S = dt * (virtual_salt + relax_salt + icef.real_salt_flux * is_nonlinfs)
    cpl = {"cpl_ice_temp": th.ice_temp, "cpl_ice_alb": th.ice_alb, "cpl_apnd": th.apnd, "cpl_hpnd": th.hpnd,
           "cpl_ipnd": th.ipnd}
    t_skin_out = th.ice_temp - TMELT
    if prescribed:                                # (exit half) the coupled ice fields and the skin as well
        cpl = {k: _sg(v) for k, v in cpl.items()}
        t_skin_out = _sg(t_skin_out)
    if getattr(state, "cpl_fw", None) is not None:          # the monthly `fw` output (ocean.py monthly_extra)
        cpl["cpl_fw"] = water_flux.astype(state.cpl_fw.dtype)
    return IceStepOut(
        stress_surf=stress_surf, bc_T=bc_T, bc_S=bc_S, sw_3d=sw_3d, stress_node_surf=stress_node_surf,
        heat_flux=heat_flux, water_flux=water_flux, virtual_salt=virtual_salt, relax_salt=relax_salt,
        real_salt_flux=icef.real_salt_flux, a_ice=a_out, m_ice=m_out, m_snow=ms_out, u_ice=u_ice, v_ice=v_ice,
        t_skin=t_skin_out, sigma11=s11, sigma12=s12, sigma22=s22, cpl=cpl)
