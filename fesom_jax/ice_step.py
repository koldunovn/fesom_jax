"""Assembled sea-ice step (Phase 6, Task 6.6) — composes the five ice kernels in the C
runtime order and produces the surface BCs the ocean step consumes.

Mirrors the per-iteration flow of ``fesom_main.c`` (the ice block before the ocean step):

    bulk_compute (atm-ocean fluxes + Ch/Ce + stress_atmice)
      → fesom_ice_step:  ocean2ice → EVP → FCT → cut_off → thermo → oce_fluxes
      → oce_fluxes_mom   (ice-ocean stress blend, prognostic u_ice/a_ice)
      → cal_shortwave_rad (penetration; open-water-only gate on the prognostic a_ice)
      → bc_T = −dt·heat_flux/vcpw,  bc_S = dt·(virtual_salt + relax_salt)

Returns :class:`IceStepOut` — the surface fluxes (``stress_surf``/``bc_T``/``bc_S``/``sw_3d``,
+ diagnostics) AND the updated prognostic ice state (a/m/snow, u/v_ice, t_skin, σ) to carry.
Every kernel is individually dump-gated (Tasks 6.2–6.5); this is the wiring + the runtime
order, gated end-to-end by the config-C per-substep T/S dump (Task 6.6).

The two Phase-5 ``a_ice`` couplings (the shortwave gate + the momentum stress blend) now read
the **prognostic** post-thermo ``a_ice`` and the EVP ``u_ice`` (Phase 5 used the static mask +
``u_ice=0``). Differentiable end-to-end (the SST→flux / current→stress / runoff seams).
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import forcing as _forcing
from . import ice_adv, ice_coupling, ice_evp, ice_mevp, ice_thermo
from .config import VCPW
from .ice import IceConfig
from .mesh import Mesh
from .state import State


# The prognostic ice state a step carries (State's sea-ice block, state.py) — what
# adjoint_mode="prescribed" cuts at entry and exit. Kept next to IceStepOut so the two lists
# cannot drift apart unnoticed: every name here is also an IceStepOut field.
_ICE_PROGNOSTIC = ("a_ice", "m_ice", "m_snow", "u_ice", "v_ice", "t_skin",
                   "sigma11", "sigma12", "sigma22")
# The ice STATE that adjoint_mode="dynamics" cuts at entry and (carried) exit; the rest of
# _ICE_PROGNOSTIC (u_ice, v_ice, σ) stays live.
_ICE_STATE = ("a_ice", "m_ice", "m_snow", "t_skin")

ADJOINT_MODES = ("exact", "rheology", "prescribed", "frozen", "dynamics")
ADJOINT_DAMPINGS = ("none", "local", "frozen_sweep")
#: IceConfig.adjoint_cuts names ("exact" and "dynamics"; see IceConfig)
ADJOINT_CUTS = ("advect", "fct", "ssh", "ocean_vel", "ice2ocean", "fct_limiter", "fct_flux",
                "strength_conc")


def validate_adjoint_settings(cfg: IceConfig) -> None:
    """Trace-time check of the ice adjoint fields. ``IceConfig._replace`` (the path every driver
    takes) skips the validating ``__new__``, so ``ice_surface_step`` calls this on every trace.
    Raises ``ValueError`` for an unknown damping form, κ < 0 (or not finite), "frozen_sweep" with
    κ0 ≤ 0, θ outside [0, 1] or adjoint_sweeps ≠ −1, ``adjoint_sweeps``
    < −1, "dynamics" / damping / ``adjoint_sweeps`` with standard EVP (``whichEVP=0``: the hooks
    are mEVP's), and damping / ``adjoint_sweeps`` under a mode that does not unroll the sweeps
    ("rheology", "prescribed", "frozen"), where they would be silently ignored; likewise a
    negative ``adjoint_thin_mass``, or a positive one outside mEVP under "dynamics"/"exact"."""
    mode = cfg.adjoint_mode
    damping = getattr(cfg, "adjoint_damping", "none")
    kappa = getattr(cfg, "adjoint_kappa", 0.25)
    sweeps = getattr(cfg, "adjoint_sweeps", -1)
    if damping not in ADJOINT_DAMPINGS:
        raise ValueError(f"IceConfig.adjoint_damping={damping!r}: use one of {ADJOINT_DAMPINGS}")
    try:
        kappa_f = float(kappa)
    except (TypeError, ValueError):
        raise ValueError(f"IceConfig.adjoint_kappa={kappa!r} is not a number") from None
    if not (kappa_f >= 0.0 and kappa_f < float("inf")):
        raise ValueError(f"IceConfig.adjoint_kappa={kappa!r}: need a finite κ ≥ 0")
    if isinstance(sweeps, bool) or int(sweeps) != sweeps or sweeps < -1:
        raise ValueError(f"IceConfig.adjoint_sweeps={sweeps!r}: need −1 (all) or an integer ≥ 0")
    if damping == "frozen_sweep":
        k0 = getattr(cfg, "adjoint_alpha_kappa", 1.0)
        th = getattr(cfg, "adjoint_theta", 0.75)
        try:
            k0_f, th_f = float(k0), float(th)
        except (TypeError, ValueError):
            raise ValueError(f"IceConfig.adjoint_alpha_kappa={k0!r} / adjoint_theta={th!r} "
                             "must be numbers") from None
        if not (0.0 < k0_f < float("inf")):
            raise ValueError(f"IceConfig.adjoint_alpha_kappa={k0!r}: need a finite κ0 > 0")
        if not (0.0 <= th_f <= 1.0):
            raise ValueError(f"IceConfig.adjoint_theta={th!r}: need θ in [0, 1]")
        if sweeps != -1:
            raise ValueError("IceConfig.adjoint_damping='frozen_sweep' differentiates the whole "
                             f"solve; adjoint_sweeps={sweeps!r} must stay −1")
    cuts = getattr(cfg, "adjoint_cuts", ())
    if isinstance(cuts, str) or not isinstance(cuts, tuple):
        raise ValueError(f"IceConfig.adjoint_cuts={cuts!r}: need a tuple of names from "
                         f"{ADJOINT_CUTS}")
    bad = [c for c in cuts if c not in ADJOINT_CUTS]
    if bad:
        raise ValueError(f"IceConfig.adjoint_cuts: unknown name(s) {bad}; use {ADJOINT_CUTS}")
    if cuts and mode not in ("dynamics", "exact"):
        raise ValueError(f"IceConfig.adjoint_cuts={cuts!r} act under adjoint_mode 'dynamics' "
                         f"or 'exact' only, not {mode!r}")
    if "strength_conc" in cuts and cfg.whichEVP != 1:
        raise ValueError("IceConfig.adjoint_cuts 'strength_conc' acts on the mEVP strength "
                         f"(whichEVP=1), not whichEVP={cfg.whichEVP}")
    guard = getattr(cfg, "adjoint_sweep_guard", 0.0)
    try:
        guard_f = float(guard)
    except (TypeError, ValueError):
        raise ValueError(f"IceConfig.adjoint_sweep_guard={guard!r} is not a number") from None
    if not (0.0 <= guard_f < float("inf")):
        raise ValueError(f"IceConfig.adjoint_sweep_guard={guard!r}: need a finite G ≥ 0")
    if guard_f > 0.0 and damping != "frozen_sweep":
        raise ValueError(f"IceConfig.adjoint_sweep_guard={guard!r} acts on the frozen_sweep "
                         f"backward only, not adjoint_damping={damping!r}")
    thin = getattr(cfg, "adjoint_thin_mass", 0.0)
    try:
        thin_f = float(thin)
    except (TypeError, ValueError):
        raise ValueError(f"IceConfig.adjoint_thin_mass={thin!r} is not a number") from None
    if not (0.0 <= thin_f < float("inf")):
        raise ValueError(f"IceConfig.adjoint_thin_mass={thin!r}: need a finite load ≥ 0 [kg/m²]")
    if thin_f > 0.0 and (mode not in ("dynamics", "exact") or cfg.whichEVP != 1):
        raise ValueError(f"IceConfig.adjoint_thin_mass={thin!r} acts on the mEVP momentum solve "
                         f"(whichEVP=1) under adjoint_mode 'dynamics' or 'exact' only, not "
                         f"{mode!r} with whichEVP={cfg.whichEVP}")
    tuned = damping != "none" or sweeps != -1
    if cfg.whichEVP != 1 and (mode == "dynamics" or tuned):
        raise ValueError(
            f"IceConfig adjoint_mode={mode!r}, adjoint_damping={damping!r}, "
            f"adjoint_sweeps={sweeps!r} need mEVP (whichEVP=1); whichEVP={cfg.whichEVP}")
    if tuned and mode not in ("exact", "dynamics"):
        raise ValueError(
            f"IceConfig.adjoint_damping / adjoint_sweeps act on the unrolled mEVP sweeps, which "
            f"adjoint_mode={mode!r} does not differentiate; use 'exact' or 'dynamics'")


class IceStepOut(NamedTuple):
    # surface BCs the ocean step consumes
    stress_surf: jnp.ndarray        # (elem2D,2)
    bc_T: jnp.ndarray               # (nod2D,)
    bc_S: jnp.ndarray               # (nod2D,)
    sw_3d: jnp.ndarray              # (nod2D,nl)
    stress_node_surf: jnp.ndarray   # (nod2D,2) ice-blended NODE wind stress (KPP ustar)
    # diagnostics (for the dump gate)
    heat_flux: jnp.ndarray
    water_flux: jnp.ndarray
    virtual_salt: jnp.ndarray
    relax_salt: jnp.ndarray
    real_salt_flux: jnp.ndarray     # zstar real-salt producer [PSU·m/s] (0 under linfs)
    # updated prognostic ice state (to thread into State)
    a_ice: jnp.ndarray
    m_ice: jnp.ndarray
    m_snow: jnp.ndarray
    u_ice: jnp.ndarray
    v_ice: jnp.ndarray
    t_skin: jnp.ndarray
    sigma11: jnp.ndarray
    sigma12: jnp.ndarray
    sigma22: jnp.ndarray
    # ice_ave_steps > 1 only (None otherwise): the State.ICE_AVG_FIELDS to carry
    ice_steps_since_upd: jnp.ndarray | None = None
    ice_update: jnp.ndarray | None = None
    srfoce_temp: jnp.ndarray | None = None
    srfoce_salt: jnp.ndarray | None = None
    srfoce_ssh: jnp.ndarray | None = None
    srfoce_u: jnp.ndarray | None = None
    srfoce_v: jnp.ndarray | None = None
    ice_flx_h: jnp.ndarray | None = None
    ice_flx_fw: jnp.ndarray | None = None
    ice_evaporation: jnp.ndarray | None = None
    ice_sublimation: jnp.ndarray | None = None
    ice_thdgr: jnp.ndarray | None = None
    ice_thdgrsn: jnp.ndarray | None = None
    ice_real_salt_flux: jnp.ndarray | None = None
    ice_a_old: jnp.ndarray | None = None


def _with_ice_dt(cfg: IceConfig, dt: float) -> IceConfig:
    """``cfg`` with ``ice_dt = ice_ave_steps * dt`` (``ice_setup_step.F90:56``)."""
    if int(cfg.ice_ave_steps) < 1:
        raise ValueError(f"IceConfig.ice_ave_steps must be >= 1, got {cfg.ice_ave_steps}")
    return cfg._replace(ice_dt=cfg.ice_ave_steps * dt)


def update_decision(k_prev, nave: int):
    """FESOM2's ice update decision (``fesom_module.F90:1129-1135``): ``ice_update`` when the
    counter ``ice_steps_since_upd`` has reached ``ice_ave_steps - 1`` (then it restarts at 0),
    else no update and the counter advances. Returns ``(ice_update, ice_steps_since_upd)``."""
    upd = k_prev >= nave - 1
    return upd, jnp.where(upd, 0, k_prev + 1)


def init_ice_avg(state: State, cfg, *, xp=None) -> State:
    """Give ``state`` the ``ice_ave_steps > 1`` fields (``State.ICE_AVG_FIELDS``) at the upstream
    init values, or strip them when they are not used.

    FESOM2 sets ``ice_steps_since_upd = ice_ave_steps - 1`` and ``ice_update = .true.`` after
    ``ice_setup`` (``fesom_module.F90:600-601``), so the first step is an ice update and
    ``ocean2ice`` takes the current surface values; the running means and the held thermo
    outputs are overwritten before anything reads them, so they start at 0.

    * ``cfg`` None or ``ice_ave_steps == 1`` ⇒ the fields are set to None (a restart written by an
      ``ice_ave_steps > 1`` run loads into an ``ice_ave_steps == 1`` run with the program it always
      had).
    * fields already present (a restart written by an ``ice_ave_steps > 1`` run, which carries the
      counter, the flag, the means and the held fluxes, as FESOM2's derived-type restart does,
      ``MOD_ICE.F90:468-469, 484-490``) ⇒ unchanged, so a run resumes mid-cycle.
    * fields absent (a cold start, or a restart written before Task 12 or by an
      ``ice_ave_steps == 1`` run) ⇒ the upstream init values: the run starts a new cycle, as a
      FESOM2 run restarted from its netCDF restart does (that restart has no counter,
      ``io_restart.F90:343-360``).

    Works on a dense State, a partitioned host ``[P, Lmax]`` State and a folded device-sharded
    ``[P*Lmax]`` State (the new leaves take ``a_ice``'s shape and sharding). ``xp=np`` builds host
    numpy leaves (the host cold-start pipeline)."""
    import numpy as np
    from .state import ICE_AVG_FIELDS

    nave = 1 if cfg is None else int(cfg.ice_ave_steps)
    if nave == 1:
        if all(getattr(state, f) is None for f in ICE_AVG_FIELDS):
            return state
        return dataclasses.replace(state, **{f: None for f in ICE_AVG_FIELDS})
    if state.ice_steps_since_upd is not None:
        missing = [f for f in ICE_AVG_FIELDS if getattr(state, f) is None]
        if missing:
            raise ValueError(f"State has ice_steps_since_upd but not {missing}")
        return state
    ref = state.a_ice
    shape = tuple(ref.shape)
    host = {f: np.zeros(shape, np.float64) for f in ICE_AVG_FIELDS}
    host["ice_steps_since_upd"] = np.full(shape, nave - 1, np.int32)
    host["ice_update"] = np.ones(shape, np.int32)
    if xp is np or isinstance(ref, np.ndarray):
        new = host
    elif (isinstance(ref, jax.Array) and not isinstance(ref, jax.core.Tracer)
          and len(ref.sharding.device_set) > 1):
        new = {f: jax.make_array_from_callback(shape, ref.sharding, lambda idx, a=a: a[idx])
               for f, a in host.items()}
    else:
        new = {f: jnp.asarray(a) for f, a in host.items()}
    return dataclasses.replace(state, **new)


def ice_surface_step(cfg: IceConfig, mesh: Mesh, state: State, sf, fs, *,
                     dt: float, boundary_node=None, use_virt_salt: bool = True,
                     owned_mask=None, axis_name=None, exch=None, zbar3=None,
                     params=None, mevp_wide=None, ice_update=None) -> IceStepOut:
    """One assembled ice step → the surface fluxes + the new ice state. ``sf`` is this step's
    :class:`surface_forcing.StepForcing` (atmosphere + month SSS/chl); ``fs`` the
    :class:`surface_forcing.ForcingStatic` (runoff/areas/open_water). ``boundary_node`` (the
    coastal mask) is precomputed once if given, else derived from ``mesh``.

    ``use_virt_salt`` (static; ``True`` for linfs, ``False`` for zstar = ``ale_cfg`` present):
    selects the freshwater/salt path — the thermo real-salt split + the ``real_salt_flux``
    producer, the global water-flux balancing, ``virtual_salt≡0``, and the ``bc_S`` real-salt
    term. ``True`` ⇒ byte-identical to the pre-Phase-9 ice step.

    ``params`` (a :class:`fesom_jax.params.Params`, or ``None``): its optional ice controls
    ``ice_pstar`` / ``ice_cd`` replace ``cfg.pstar`` / ``cfg.cd_oce_ice`` for this step, as traced
    values. Unset (``None``, the default) ⇒ the configured constants, nothing traced.

    ``mevp_wide`` (a :class:`fesom_jax.ice_wide.MevpWide`, sharded mEVP only; ``None`` ⇒
    unchanged): the mEVP subcycles run on K-ring ice lanes with one exchange every K subcycles.

    **Sea ice every N ocean steps** (``cfg.ice_ave_steps = N > 1``, opt-in; plan Task 12),
    FESOM2's scheme (``fesom_module.F90:1100-1166``, e96a2b13; the same in 4ccdd35a, the R2
    binary's source, ``:657-692``):

    * ``ocean2ice`` runs every step BEFORE the update decision, with the previous step's
      ``ice_update`` and ``ice_steps_since_upd``: current surface values after an update step,
      running means since the last update otherwise (:func:`ice_coupling.ocean2ice_avg`, which
      documents the mirrored upstream ``u_w``/``v_w`` quirk: the ice and the ocean stress see
      ``uw/(k+1)``, not a mean). The bulk formulae read these ``srfoce`` fields every step.
    * the decision (:func:`update_decision`); the ice update (EVP/mEVP + FCT + cut-off +
      thermodynamics) runs under ``lax.cond`` only on update steps, at ``ice_dt = N*dt``. The
      counter is the same on every device, so all devices take the same branch.
    * ``oce_fluxes_mom`` and ``oce_fluxes`` run EVERY step from the last thermo outputs, which the
      State holds (``ICE_AVG_FIELDS``) with the counter, the flag and the means; the restart
      carries them (:func:`init_ice_avg` for a State or restart without them).
    * ``N == 1`` is the program this function always traced (the new State fields are None).
    * ``ice_update`` (static ``True`` / ``False``; ``None`` ⇒ the decision above under ``lax.cond``): the caller
      knows the cycle's phase (``gpu.ice_static``, :func:`integrate_sharded.run_steps_sharded_forced`'s
      ``ice_phase``) and this step is (not) an update step, so the branch is chosen at trace time. A GPU
      ``lax.cond`` copies its predicate to the host and syncs every step; the static branch does not. The caller
      must pass FESOM2's decision (``ice_steps_since_upd == ice_ave_steps - 1``); the counter and the flag are
      then written from it."""
    # ⚠️ The ice timestep MUST track the ocean dt (C ``fesom_ice_setup``: ``ice_dt =
    # ice_ave_steps*dt``, fesom_ice.c:231). ``IceConfig.ice_dt`` is only a build-time default
    # (500 s) — if the run's ocean dt differs (e.g. the dt=1800 climate run) and ice_dt is left
    # at 500, EVERY ice rate (thermo growth/melt ×ice_dt, FCT transport ×ice_dt, EVP dte/Tevp_inv)
    # runs at the wrong timestep ⇒ the ice evolves 3.6× too slowly ⇒ a high-lat climate bias.
    # Deriving it from dt here makes the desync impossible regardless of the passed cfg.
    cfg = _with_ice_dt(cfg, dt)
    validate_adjoint_settings(cfg)
    # Ice controls (Params.ice_pstar / ice_cd): the traced value takes the config field's place.
    # Safe because no consumer of these two fields needs a static value: pstar is read only by
    # the pressure (ice_mevp.mevp_setup, ice_evp's strength) and cd_oce_ice only by the four
    # drag sites (ice_mevp / ice_evp momentum solves, ice_coupling.ice_oce_fluxes_mom,
    # ice_thermo.compute_ustar), all plain arithmetic; nothing below jits with cfg static.
    # Unset ⇒ dead branch: the jaxpr is that of the configured constants.
    _pstar = getattr(params, "ice_pstar", None)
    _cd = getattr(params, "ice_cd", None)
    if _pstar is not None:
        cfg = cfg._replace(pstar=_pstar)
    if _cd is not None:
        cfg = cfg._replace(cd_oce_ice=_cd)

    # IceConfig.adjoint_mode="prescribed": the ice TRAJECTORY is taken as given. The nine
    # prognostic ice fields carry no tangent into this step (below) and none out of it (at the
    # exit, before the stress blend and the shortwave gate), so the ice-state recurrence — whose
    # gain at an advancing ice edge is unbounded (a x462 jump in the a_ice tangent in one day
    # at the Okhotsk edge) — is cut, while every flux the ocean receives stays
    # differentiable in the ocean state and the atmosphere: the open-water bulk formulae, the
    # under-ice basal flux, the ice-ocean stress at the prescribed ice velocity, the shortwave
    # gate at the prescribed cover. This is the sea-ice treatment of the ECCO adjoint when the
    # ice package's adjoint is disabled: the ice MODEL is not differentiated, the fluxes are.
    # Contrast "frozen", which stop_gradients the whole IceStepOut — fluxes included — and so
    # zeroes dJ/d(atmosphere) identically and removes the SST -> flux damping (measured: the
    # forcing gradient 2.4e-03 -> 0.0 exactly, the T gradient +16 % over 30 d). The forward is
    # untouched: stop_gradient is the identity.
    prescribed = cfg.adjoint_mode == "prescribed"
    _sg = jax.lax.stop_gradient
    if prescribed:
        state = dataclasses.replace(
            state, **{f: _sg(getattr(state, f)) for f in _ICE_PROGNOSTIC})
    # IceConfig.adjoint_mode="dynamics" (stage 1 of the ice-dynamics adjoint): the ice STATE
    # (a, m, m_snow, t_skin) enters without a tangent; u_ice, v_ice and σ stay live. The exit
    # half is at the return below, AFTER this step's own uses of the post-thermo state.
    dynamics = cfg.adjoint_mode == "dynamics"
    if dynamics:
        state = dataclasses.replace(state, **{f: _sg(getattr(state, f)) for f in _ICE_STATE})
    # Derivative cuts (IceConfig.adjoint_cuts, "dynamics" and "exact"; empty ⇒ nothing traced).
    _cuts = (frozenset(getattr(cfg, "adjoint_cuts", ()))
             if cfg.adjoint_mode in ("dynamics", "exact") else frozenset())

    def _cut(names, *xs):
        if _cuts & set(names):
            xs = tuple(_sg(x) for x in xs)
        return xs if len(xs) > 1 else xs[0]
    open_water = fs.open_water
    geo_lat = jnp.asarray(mesh.geo_coord_nod2D)[:, 1]

    # --- ice_ave_steps > 1 (FESOM2 fesom_module.F90:1100-1166): ocean2ice runs EVERY step and,
    # with the PREVIOUS step's flag and counter, keeps the running means since the last update;
    # the update decision follows; the ice step runs only on an update step, at ice_dt = N*dt.
    nave = int(cfg.ice_ave_steps)
    avg = nave > 1
    if avg:
        if state.ice_steps_since_upd is None:
            raise ValueError(
                f"ice_ave_steps={nave} needs the State.ICE_AVG_FIELDS; build the State with "
                "ice_step.init_ice_avg(state, ice_cfg) first (the drivers do)")
        k_prev = state.ice_steps_since_upd[0]          # the same value on every lane
        upd_prev = state.ice_update[0] != 0
        srf = ice_coupling.ocean2ice_avg(state, k_prev, upd_prev)
        # update_atm_forcing (gen_forcing_couple.F90:718-738, gen_bulk_formulae.F90:175-221)
        # reads ice%srfoce_temp/u/v, i.e. the running means, quirk included
        u_w0, v_w0, sst_bulk = srf.u, srf.v, srf.temp
    else:
        # --- bulk: atm-ocean fluxes + exchange coeffs + atm-ice wind stress ---
        # state.uvnode is refreshed from state.uv at the top of step() (drift fix 1), so this
        # tap and ocean2ice below both see the current surface velocity, as in the Fortran.
        u_w0 = state.uvnode[:, 0, 0]
        v_w0 = state.uvnode[:, 0, 1]
        sst_bulk = state.T[:, 0]
    bulk = _forcing.bulk_surface_fluxes(
        mesh, sf.u_air, sf.v_air, sf.shum, sf.shortwave, sf.longwave, sf.Tair,
        sf.prec_rain, sf.prec_snow, sst_bulk, u_w0, v_w0)
    stress_ax, stress_ay = _forcing.atm_ice_stress(
        sf.u_air, sf.v_air, state.u_ice, state.v_ice)         # prev-step u_ice (bulk runs first)

    # --- fesom_ice_step: ocean2ice → EVP → FCT → cut_off → thermo → oce_fluxes ---
    if not avg:
        srf = ice_coupling.ocean2ice(state)     # slices of already-exchanged ocean state
    # mEVP dispatch (Phase 9c): a static Python branch — whichEVP is a config constant, not a
    # traced value, so this is a trace-time selection (no lax.cond). whichEVP=0 ⇒ standard EVP
    # (byte-identical); =1 ⇒ mEVP (ice_mevp.mevp_dynamics, same signature + outputs).
    _evp = ice_mevp.mevp_dynamics if cfg.whichEVP == 1 else ice_evp.evp_dynamics
    # IceConfig.adjoint_mode="rheology": run all the subcycles in the forward but unroll NONE
    # of them in the backward, so ice thermodynamics, advection, cut-off and the ocean-to-ice
    # coupling stay differentiated while only the ice VELOCITY is dark. Measured: this alone is
    # what tames the ice adjoint (docs/ADJOINT_HORIZON.md).
    # "prescribed" takes the same cut: its u_ice/v_ice leave the step without a tangent anyway,
    # so unrolling the subcycles in the backward would only cost memory.
    # "exact" / "dynamics" unroll all the sweeps, or the last `adjoint_sweeps` (−1 = all ⇒ no
    # keyword, the unchanged program); the damping is read from cfg inside mevp_dynamics.
    if cfg.adjoint_mode in ("rheology", "prescribed"):
        _evp_kw = {"adjoint_subcycles": 0}
    elif getattr(cfg, "adjoint_sweeps", -1) >= 0:
        _evp_kw = {"adjoint_subcycles": int(cfg.adjoint_sweeps)}
    else:
        _evp_kw = {}
    if mevp_wide is not None:
        if cfg.whichEVP != 1:
            raise ValueError("mevp_wide (the mEVP wide halo) needs whichEVP=1 (mEVP); the "
                             f"standard EVP has no wide path, got whichEVP={cfg.whichEVP}")
        _evp_kw["wide"] = mevp_wide
    _exch = (lambda f, k: f) if exch is None else exch       # noqa: E731

    def _ice_update():
        """ice_timestep (ice_setup_step.F90): EVP/mEVP → FCT → cut_off → thermo. Returns the new
        prognostic ice state and the thermo outputs oce_fluxes reads."""
        _evp_u, _evp_v = _cut(("ocean_vel",), srf.u, srf.v)
        u_ice, v_ice, s11, s12, s22 = _evp(
            cfg, mesh, a_ice=state.a_ice, m_ice=state.m_ice, m_snow=state.m_snow,
            u_ice=state.u_ice, v_ice=state.v_ice,
            sigma11=state.sigma11, sigma12=state.sigma12, sigma22=state.sigma22,
            srfoce_u=_evp_u, srfoce_v=_evp_v, elevation=_cut(("ssh",), srf.ssh),
            stress_ax=stress_ax, stress_ay=stress_ay, boundary_node=boundary_node, exch=exch,
            **_evp_kw)

        _fct_u, _fct_v = _cut(("advect", "fct"), u_ice, v_ice)
        a_adv, m_adv, ms_adv = ice_adv.fct_solve(
            cfg, mesh, state.a_ice, state.m_ice, state.m_snow, _fct_u, _fct_v, exch=exch,
            freeze_limiter="fct_limiter" in _cuts, freeze_limited_flux="fct_flux" in _cuts)
        _th_u, _th_v = _cut(("advect",), u_ice, v_ice)
        a_co, m_co, ms_co = ice_thermo.cut_off(a_adv, m_adv, ms_adv)

        th = ice_thermo.thermodynamics(
            cfg, m_ice=m_co, m_snow=ms_co, a_ice=a_co, u_ice=_th_u, v_ice=_th_v,
            t_skin=state.t_skin, srfoce_temp=srf.temp, srfoce_salt=srf.salt,
            srfoce_u=srf.u, srfoce_v=srf.v,
            fsh=sf.shortwave, flo=sf.longwave, Tair=sf.Tair, qa=sf.shum,
            u_wind=sf.u_air, v_wind=sf.v_air, rain=sf.prec_rain, snow=sf.prec_snow,
            runo=fs.runoff_node, ch=bulk.ch, ce=bulk.ce, geo_lat=geo_lat,
            non_cavity=open_water, use_virt_salt=use_virt_salt)

        # Refresh the prognostic ice TRACER halos RIGHT AFTER thermo — BEFORE they are consumed
        # (the FCT scatter→cut_off→thermo chain leaves them incomplete on the halo). ⚠️ This must
        # precede ice_oce_fluxes_mom: its `stress_surf` is a node→elem GATHER of the blended node
        # stress `sns`, which reads `a_ice` at the element's HALO vertices (a boundary OWNED
        # element) — incomplete-halo a_ice there gives a wrong OWNED `stress_surf`, hence a wrong
        # ocean `uv` (the bug the npes==2 gate caught: uv≈7e-4 while every ICE field was
        # bit-exact). The same complete halos feed the next step's EVP (reads a_ice/m_ice at
        # element vertices). u_ice/v_ice are already halo-complete (refreshed inside the EVP
        # subcycle scan); sigma/t_skin are per-element/per-node maps of complete inputs ⇒
        # auto-complete. exch=None ⇒ no-op.
        a_out = _exch(th.a_ice, "nod")
        m_out = _exch(th.m_ice, "nod")
        ms_out = _exch(th.m_snow, "nod")
        return (a_out, m_out, ms_out, u_ice, v_ice, th.t_skin, s11, s12, s22,
                th.flx_h, th.flx_fw, th.evaporation, th.ice_sublimation, th.thdgr,
                th.thdgrsn, th.real_salt_flux, a_co)

    if avg:
        # the update decision (fesom_module.F90:1129-1135), AFTER ocean2ice
        upd_now, k_now = update_decision(k_prev, nave)

        def _hold():
            """No ice step: the ice state and the last thermo outputs as they are."""
            return (state.a_ice, state.m_ice, state.m_snow, state.u_ice, state.v_ice,
                    state.t_skin, state.sigma11, state.sigma12, state.sigma22,
                    state.ice_flx_h, state.ice_flx_fw, state.ice_evaporation,
                    state.ice_sublimation, state.ice_thdgr, state.ice_thdgrsn,
                    state.ice_real_salt_flux, state.ice_a_old)

        if ice_update is None:
            # The counter is replicated, so every device takes the same branch and the halo
            # exchanges / reductions inside the update branch stay matched across devices.
            upd_out = jax.lax.cond(upd_now, _ice_update, _hold)
        else:
            upd_now = jnp.asarray(bool(ice_update))
            k_now = jnp.zeros_like(k_prev) if ice_update else k_prev + 1
            upd_out = _ice_update() if ice_update else _hold()
    else:
        upd_out = _ice_update()
    (a_out, m_out, ms_out, u_ice, v_ice, t_skin_out, s11, s12, s22,
     flx_h, flx_fw, evaporation, ice_sublimation, thdgr, thdgrsn, rsf, a_old) = upd_out
    if prescribed:
        # The exit half of "prescribed": the updated ice state leaves without a tangent, so the
        # stress blend and the shortwave gate below see the prescribed cover, and the next step's
        # entry cut has nothing to cut. th's FLUX outputs (flx_h, flx_fw, evaporation, ...) are
        # deliberately not touched — they are what "the fluxes stay differentiable" means.
        a_out, m_out, ms_out, u_ice, v_ice, t_skin_out, s11, s12, s22 = (
            _sg(x) for x in (a_out, m_out, ms_out, u_ice, v_ice, t_skin_out, s11, s12, s22))

    # oce_fluxes (ice_oce_coupling.F90:198-): the virtual-salt / SSS-relaxation terms read the
    # CURRENT surface salinity (salt(1,n)), not the ice's running mean srfoce_salt. With
    # ice_ave_steps == 1 the two are the same array.
    S_top = state.S[:, 0] if avg else srf.salt
    icef = ice_coupling.ice_oce_fluxes(
        S_top, flx_fw, flx_h, sf.Ssurf_month, fs.runoff_node,
        fs.areasvol_surf, fs.ocean_area, open_water,
        owned_mask=owned_mask, axis_name=axis_name,
        use_virt_salt=use_virt_salt, real_salt_flux=rsf)

    # zstar (Phase 9a): the global freshwater-flux balancing on water_flux (the C's
    # !use_virt_salt block, fesom_ice_coupling.c:193-216). a_ice_old = the THERMO-ENTRY
    # (post-advection/cut_off) concentration `a_co`: the C saves values_old inside the thermo
    # loop AFTER fct_solve/cut_off and BEFORE overwriting with thermo's outputs
    # (fesom_ice_thermo.c:497-506), and that is what the balance consumes
    # (fesom_ice_coupling.c:197-207). With a_co the balance's prec_snow·(1−a_old) cancels
    # thermo's snow·(1−A) in fw exactly ⇒ post-balance ⟨water_flux⟩ ≡ 0 (the conservation
    # gate in test_ice_step.py); passing state.a_ice instead leaks
    # ⟨prec_snow·(A_entry−A_state)⟩ into the volume budget every step (2026-07-03 review).
    # Under linfs this is skipped (dead branch ⇒ byte-identical). With ice_ave_steps > 1 the
    # thermo outputs and a_old are the held ones on a step without an ice update, while
    # prec_rain/prec_snow/runoff are this step's forcing (oce_fluxes runs every step,
    # fesom_module.F90:1165-1166), so the balance is exact only on the update steps.
    water_flux = icef.water_flux
    if not use_virt_salt:
        water_flux = ice_coupling.fresh_water_balance_zstar(
            water_flux, evaporation, ice_sublimation, sf.prec_rain, sf.prec_snow,
            a_old, fs.runoff_node, thdgr, thdgrsn,
            fs.areasvol_surf, fs.ocean_area, cfg,
            owned_mask=owned_mask, axis_name=axis_name)

    # --- oce_fluxes_mom: ice-ocean stress blend (prognostic a_ice/u_ice; halo-complete a_ice) ---
    _mom_u, _mom_v = _cut(("ice2ocean",), u_ice, v_ice)
    # ice_ave_steps > 1: srfoce_u/v are the ice's u_w/v_w (quirk included), as the Fortran reads
    # ice%srfoce_u/v here every step (ice_oce_coupling.F90:55-56, 81-83).
    stress_surf, stress_node_surf = ice_coupling.ice_oce_fluxes_mom(
        mesh, a_out, _mom_u, _mom_v, srf.u, srf.v, bulk.stress_node_surf, open_water, cfg)

    # --- shortwave penetration (open-water-only gate on the prognostic a_ice) ---
    pene_open = open_water & (a_out <= 0.0)
    heat_flux, sw_3d = _forcing.cal_shortwave_rad(
        mesh, icef.heat_flux, sf.shortwave, sf.chl, pene_open, zbar3=zbar3)

    # --- surface BCs (fesom_tracer_diff.c:43-75 / bc_surface). bc_T's base; the zstar
    #     −dt·sval·water_flux·is_nonlinfs term needs the POST-ADVECTION surface T (the C's
    #     sval = trarr[surface] at the diffusion), so it is added in step.py before
    #     impl_vert_diff (NOT here). bc_S is fully resolved here: under zstar virtual_salt≡0
    #     and the real_salt_flux term turns on (is_nonlinfs=1); linfs ⇒ rsf=0 ⇒ unchanged.
    is_nonlinfs = 0.0 if use_virt_salt else 1.0
    bc_T = -dt * heat_flux / VCPW
    bc_S = dt * (icef.virtual_salt + icef.relax_salt + icef.real_salt_flux * is_nonlinfs)

    if dynamics:
        # The exit half of "dynamics": the CARRIED ice state leaves without a tangent. Placed
        # after the stress blend and the shortwave gate on purpose: inside this step the
        # post-thermo a_out (moved by the live ice velocity through advection) feeds both with
        # its perturbed value, exactly as in a forward whose ice state is overwritten by a base
        # trajectory after every step (ice_cbar's replay), so this mode's derivative is that
        # replay's. "prescribed" cuts before the blend instead (its ice velocity is given too).
        a_out, m_out, ms_out, t_skin_out = (_sg(x) for x in (a_out, m_out, ms_out, t_skin_out))

    extra = {}
    if avg:
        extra = dict(
            ice_steps_since_upd=jnp.full_like(state.ice_steps_since_upd, k_now),
            ice_update=jnp.full_like(state.ice_update, upd_now.astype(jnp.int32)),
            srfoce_temp=srf.temp, srfoce_salt=srf.salt, srfoce_ssh=srf.ssh,
            srfoce_u=srf.u, srfoce_v=srf.v,
            ice_flx_h=flx_h, ice_flx_fw=flx_fw, ice_evaporation=evaporation,
            ice_sublimation=ice_sublimation, ice_thdgr=thdgr, ice_thdgrsn=thdgrsn,
            ice_real_salt_flux=rsf, ice_a_old=a_old)
    return IceStepOut(
        stress_surf=stress_surf, bc_T=bc_T, bc_S=bc_S, sw_3d=sw_3d,
        stress_node_surf=stress_node_surf,
        heat_flux=heat_flux, water_flux=water_flux,
        virtual_salt=icef.virtual_salt, relax_salt=icef.relax_salt,
        real_salt_flux=icef.real_salt_flux,
        a_ice=a_out, m_ice=m_out, m_snow=ms_out,
        u_ice=u_ice, v_ice=v_ice, t_skin=t_skin_out,
        sigma11=s11, sigma12=s12, sigma22=s22, **extra)
