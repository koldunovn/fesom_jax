"""Full REcoM in the model step and the run (parent plan Tasks 10/12/13 wiring, lane W).

What :func:`fesom_jax.step.step` calls in full mode (``RecomConfig.mode == "full"``) and what the run driver needs
around it:

* :func:`with_recom_static` -- the per-run node constants stage 3 needs and the step cannot build from the traced
  mesh fields (lane T's SAAR columns, interpolated on the host from the node positions; the seafloor area under each
  layer, an element scatter): optional fields of :class:`~fesom_jax.surface_forcing.ForcingStatic`, built once on
  the host. ``ForcingStatic`` is already partitioned, folded and device-placed (once per run, the placed-constants
  cache) on every forced path, and a ``None`` field adds no leaf, so the REcoM-off programs are unchanged. The rest
  of what stage 1 and stage 3 read comes from the mesh (geographic latitude, ``area``, ``areasvol``, level counts,
  ``ocean_area``) and from the step.
* :func:`with_bgc_state` / :func:`start_segment` -- the full-REcoM State fields (``State.BGC_FULL_FIELDS``) from a
  cold-start :class:`~fesom_jax.recom.ic.BgcIC`, and FESOM2's run-segment start (``mstep`` back to 1, the carbonate
  cache refilled at that step).
* :func:`full_bgc_step` -- one BGC step: stage 1 (``recom()``, :func:`fesom_jax.recom.stage1.recom_sources_sinks`)
  then stage 3 (the BGC part of the tracer loop, :func:`fesom_jax.recom.transport.recom_stage3`), plus the 2-D
  output diagnostics and the optional budget sums.
* :func:`output_fields` -- the BGC fields of the run's mean-output streams.

**Where stage 1 runs.** FESOM2 calls ``recom()`` after the forcing update, the ice step and the ocean flux coupling
and before ``oce_timestep_ale`` (``fesom_module.F90:1203``). Nothing in the physics reads a BGC tracer, so the step
evaluates stage 1 at the tracer stage, from the values it had at that point: T and S of the step start, ``hnode``
and the zstar geometry of the step start, ``a_ice`` after the ice step, this step's forcing. The values are those
of the Fortran order; only the place in the program differs (and, under ``bgc_every = k``, stage 1 can sit inside
the BGC step's ``lax.cond``).

**Reductions.** ``relax_alk``'s global mean (``bio_fluxes``) is the one global sum: ``jnp.sum`` dense, the owned-node
sum + ``psum`` sharded (:func:`fesom_jax.reductions.global_sum`, so the halo copies do not count twice).

**BGC every k steps** (``RecomConfig.bgc_every = k > 1``, a stated deviation, plan Task 2b): stage 1 and stage 3 run on
the k-th step with ``dt_bgc = k dt`` on the interval's mean transport, from the interval-start thickness; stage 1
then reads T/S at the start of that k-th step and its hnode is the interval-start one (the thickness the held BGC
content refers to); ``mstep`` counts BGC steps, so the carbonate refresh periods (``int(86400/dt_bgc * 7)``, ``*30``)
are in BGC steps (stated deviation, as in lane S).

**Output diagnostics** (``State.bgc_diag``, :data:`DIAG2D`): the air-sea CO2 flux ``GloCO2flux`` [mmol C m-2 d-1,
positive into the ocean], ``GloPCO2surf`` [uatm], the column NPP ``NPPn + NPPd`` [mmol C m-2 d-1], the POC export
through the interface nearest 100 m (:func:`export_level`) [mmol C m-2 d-1], the surface pH ``GloHplus`` and the O2
flux ``GloO2flux`` [mmol O2 m-2 d-1]. The export is a fesom-jax diagnostic (REcoM 692f804 has no such output): the
downward sinking flux of the organic carbon tracers (:data:`POC_IDS`: PhyC, DetC, DiaC, DetZ2C; phytoplankton and
diatoms do not sink in the water column with ``VPhy = VDia = 0``) through that interface, ``-sum_{k < k100}
(values_mod - X1) hnode / dt`` (``ver_sinking_recom`` has no surface flux and the bottom loss ``str_bf`` is only in
the bottom layer, so the sum is exactly the flux through the interface), zero where the column ends above it.

**Budget** (``RecomConfig.budget``, ``State.bgc_budget``, :data:`BUDGET_TERMS`): per column and tracer, summed over
the BGC steps, the content [tracer units m3] ``sum_k (.) hnode areasvol`` that entered through the surface BC, stage 1
(sources and sinks, the clip and the DICremin surface reset), REcoM's tracer-loop terms (benthic return flux,
sinking, deposit to the seafloor), and the clip alone. The transport conserves ``sum_k X hnode areasvol``, so the
inventory change of every tracer is ``surface + stage1 + loop``.
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from . import transport
from .sinking import SaarColumns, load_saar_table, saar_columns, seafloor_area, sinking_geometry
from .sms import CARB_FIELDS, CarbCache
from .stage1 import BgcStepInputs, RecomGeometry, recom_sources_sinks
from .tracers import NTR, SLOT

RAD = math.pi / 180.0                      # FESOM2 rad (oce_modules.F90:34-35)
SECONDS_PER_DAY = 86400.0
DIAG2D = ("co2flux", "pco2surf", "npp", "export100", "ph", "o2flux")
BUDGET_TERMS = ("surface", "stage1", "loop", "clip")
POC_IDS = (1005, 1008, 1014, 1026)         # PhyC, DetC, DiaC, DetZ2C
EXPORT_DEPTH = 100.0                       # [m]


# ---------------------------------------------------------------------------------------------------------------
# per-run constants
# ---------------------------------------------------------------------------------------------------------------
def recom_static_arrays(mesh) -> dict:
    """The ``ForcingStatic.bgc_*`` arrays of ``mesh`` (host numpy, global node order)."""
    nod, nl = int(mesh.nod2D), int(mesh.nl)
    sf = np.asarray(seafloor_area(np.asarray(mesh.elem_nodes), np.asarray(mesh.nlevels), np.asarray(mesh.elem_area),
                                  nod, nl))
    geo = np.asarray(mesh.geo_coord_nod2D, np.float64)
    cols = saar_columns(geo[:, 0] / RAD, geo[:, 1] / RAD)
    return dict(bgc_seafloor=sf, bgc_saar_h=np.asarray(cols.h), bgc_saar_pcap=np.asarray(cols.p_cap),
                bgc_saar_data=np.asarray(cols.has_data), bgc_saar_baltic=np.asarray(cols.baltic))


def with_recom_static(fs, mesh):
    """``fs`` (a :class:`~fesom_jax.surface_forcing.ForcingStatic`, global) with full REcoM's node constants."""
    return fs._replace(**{k: (jnp.asarray(v) if isinstance(fs.runoff_node, jax.Array) else v)
                          for k, v in recom_static_arrays(mesh).items()})


def _saar(fs) -> SaarColumns:
    return SaarColumns(h=fs.bgc_saar_h, p_cap=fs.bgc_saar_pcap, has_data=fs.bgc_saar_data,
                       baltic=fs.bgc_saar_baltic, p_ref=jnp.asarray(load_saar_table().p_ref))


def export_level(mesh):
    """The interface index nearest ``-EXPORT_DEPTH`` in the nominal ``zbar`` (the export diagnostic's depth; a traced
    scalar inside the step, where the mesh is an argument)."""
    return jnp.argmin(jnp.abs(jnp.asarray(mesh.zbar) + EXPORT_DEPTH))


# ---------------------------------------------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------------------------------------------
def cold_ic(mesh, rcfg):
    """FESOM2 + REcoM's cold-start BGC state for ``rcfg`` (:func:`fesom_jax.recom.ic.bgc_initial_state`)."""
    from . import ic
    sw, p = rcfg.load()
    files = {"pi": ic.PI_IC_FILES, "core2": ic.CORE2_IC_FILES, "core2_present": ic.CORE2_PRESENT_IC_FILES}[rcfg.ic_set]
    if rcfg.ic_dir is None:
        raise ValueError("full REcoM's cold start needs recom.ic_dir (FESOM's ClimateDataPath of the BGC IC files)")
    return ic.bgc_initial_state(mesh, files, sw, p, ic_dir=rcfg.ic_dir)


def with_bgc_state(state, ic, rcfg):
    """``state`` with every full-REcoM field from ``ic`` (a :class:`~fesom_jax.recom.ic.BgcIC`, FESOM's ``nl-1``
    layers; the tracers padded with a zero bottom level), the counters, cache and diagnostics zero. Backend-agnostic
    like :func:`fesom_jax.recom.transport.init_passive` (host numpy for the run driver's cold start)."""
    import dataclasses
    xp = np if isinstance(state.S, np.ndarray) else jnp
    out = {k: xp.asarray(v) for k, v in ic_fields(ic).items()}
    out.update(segment_fields(ic.bgc.shape[0], ic.bgc.shape[1], rcfg, xp))
    return dataclasses.replace(state, **out)


def ic_fields(ic) -> dict:
    """The restarted full-REcoM fields of a :class:`~fesom_jax.recom.ic.BgcIC` (host numpy, global): the tracers
    padded from FESOM's ``nl-1`` layers to ``nl`` levels with a zero bottom level, ``Benthos``, ``Alk_surf``."""
    n = ic.bgc.shape[0]
    pad = lambda a: np.concatenate([np.asarray(a), np.zeros((n, 1, a.shape[2]))], axis=1)  # noqa: E731
    return dict(bgc=pad(ic.bgc), bgc_old=pad(ic.bgc_old), bgc_benthos=np.asarray(ic.benthos),
                bgc_alk_surf=np.asarray(ic.alk_surf))


def segment_fields(n: int, nl1: int, rcfg, xp=jnp) -> dict:
    """The fields that restart with every run segment: ``bgc_mstep`` 0, the carbonate cache, the diagnostics and the
    budget sums zero."""
    out = dict(bgc_carb=xp.zeros((n, nl1, len(CARB_FIELDS))), bgc_mstep=xp.zeros((n,), np.int32),
               bgc_diag=xp.zeros((n, len(DIAG2D))))
    if rcfg.budget:
        out["bgc_budget"] = xp.zeros((n, len(BUDGET_TERMS), NTR))
    return out


def start_segment(state, rcfg):
    """FESOM2's run-segment start for a State that carries the BGC tracers (a restart): ``mstep`` restarts at 1 and
    the carbonate cache is refilled at that step (``fesom_module.F90:849,983,1049``; the cache is not in the Fortran
    restart), so both are reset here; ``Benthos`` and ``Alk_surf`` come from the restart. Backend-agnostic (``x * 0``
    keeps a sharded leaf's sharding)."""
    import dataclasses
    T = state.T
    xp = np if isinstance(T, np.ndarray) else jnp
    if state.bgc_benthos is None or state.bgc_alk_surf is None:
        raise ValueError("start_segment needs a full-REcoM State (bgc_benthos, bgc_alk_surf): a restart written by "
                         "the passive mode cannot continue as full REcoM")
    z2 = T[..., 0] * 0                                                # (nod,) in the State's layout ([P, Lmax] too)
    z3 = T[..., :-1] * 0
    out = dict(bgc_mstep=z2.astype(np.int32),
               bgc_carb=xp.stack([z3] * len(CARB_FIELDS), axis=-1),
               bgc_diag=xp.stack([z2] * len(DIAG2D), axis=-1))
    if rcfg.budget:
        out["bgc_budget"] = xp.stack([xp.stack([z2] * NTR, axis=-1)] * len(BUDGET_TERMS), axis=-2)
    else:
        out["bgc_budget"] = None
    return dataclasses.replace(state, **out)


def place_global_fields(fields: dict, part, devices=None) -> dict:
    """Global host node fields ``{name: (nod2D, ...)}`` partitioned to ``part`` and placed folded ``[P*Lmax, ...]``
    on the device mesh, as :func:`fesom_jax.zarr_output.read_restart` places a restart field (a physics-only restart
    into a REcoM run: the BGC IC joins the restart's folded State)."""
    from .. import halo
    from ..integrate_sharded import _fold
    from ..shard_mesh import _default_pad, _shard_along_axis, local_sizes
    from jax.sharding import NamedSharding, PartitionSpec
    _, Lmax = local_sizes(part)
    devs = jax.devices()[:part.npes] if devices is None else list(devices)
    sharding = NamedSharding(halo.device_mesh("p", devices=devs), PartitionSpec("p"))
    out = {}
    for name, g in fields.items():
        g = np.asarray(g)
        folded = _fold(_shard_along_axis(g, part.myList_nod2D, Lmax["nod"], 0, _default_pad(g.dtype)))
        out[name] = jax.make_array_from_callback(folded.shape, sharding, lambda idx, a=folded: a[idx])
    return out


# ---------------------------------------------------------------------------------------------------------------
# the BGC step
# ---------------------------------------------------------------------------------------------------------------
def recom_params(rcfg, params):
    """``(switches, params)``: the namelist's, with ``params.recom`` (traced leaves) replacing the parameters when
    given."""
    sw, p = rcfg.load()
    pr = getattr(params, "recom", None) if params is not None else None
    return sw, (p if pr is None else pr)


def full_bgc_step(mesh, st, rcfg, *, params, h0, zbar0, Z0, helem, helem_flux, uv_adv, w_e_adv, w_i, hnode_new,
                  Z3d_new, gm_diag, Kv_eff, T0, S0, T_fin, S_fin, a_ice, sf, fs, dt_bgc, reduce_sum, exch, ckpt,
                  adj) -> dict:
    """One full REcoM BGC step (stage 1 + stage 3) from ``st``; returns the updated ``State`` BGC fields
    (:data:`fesom_jax.recom.transport.FULL_STEP_FIELDS`, + ``bgc_budget`` when carried).

    ``h0``/``zbar0``/``Z0``/``helem`` the thickness and zstar geometry the BGC content refers to (the step start, or
    the interval start under ``bgc_every``); the transport arguments as :func:`fesom_jax.recom.transport.recom_stage3`;
    ``T0``/``S0`` T and S at the step start (stage 1), ``T_fin``/``S_fin`` after the tracer step (S before the
    salinity floor; the sinking viscosity and density); ``a_ice`` after the ice step; ``sf`` this step's
    :class:`~fesom_jax.surface_forcing.StepForcing` (with ``press_air``, ``fe_dust``, ``daynew``, ``ndpyr``);
    ``fs`` the ForcingStatic with :func:`with_recom_static`'s fields.

    Single precision (an opt-in deviation, plan ``20261026-recom-lane-sp``): REcoM's code (stage 1, mocsy, the sinking
    geometry, the loop terms, the diagnostics and budget sums) runs in float64, REcoM's own ``WP = 8``: every WP
    model field it reads is widened exactly here (``PR.full``, the identity in double precision) and its constants
    are explicit float64 (``test_recom_sp`` checks that no float32 operand meets a float64 one implicitly); stage 3
    is FESOM2 #1207's scheme with ``State.bgc_carry``, which joins the returned fields."""
    from .. import precision as PR
    _sp = PR.sp()
    if _sp and st.bgc_carry is None:
        raise ValueError("single precision + full REcoM needs State.bgc_carry (precision.cast_state adds it)")
    sw, p = recom_params(rcfg, params)
    nl1 = mesh.nl - 1
    f64 = jnp.float64
    nlev = (jnp.asarray(mesh.nlevels_nod2D) - 1).astype(jnp.int32)
    lat_rad = jnp.asarray(mesh.geo_coord_nod2D[:, 1], f64)
    X, X_old = transport.hold_major(st.bgc), st.bgc_old          # stage 1 reads each tracer's columns contiguously
    h = h0[:, :nl1]
    full = PR.full                         # the identity in double precision: the double-precision trace is unchanged
    areasvol, area = full(mesh.areasvol), full(mesh.area)
    zbar0_f, Z0_f = full(zbar0), full(Z0)

    # ---- stage 1: recom() ----
    geom1 = RecomGeometry(hnode=h, zbar=zbar0_f, nlev=nlev, lat_rad=lat_rad, area_top=areasvol[:, 0],
                          ocean_area=full(mesh.ocean_area))
    mstep = st.bgc_mstep[0] + 1
    fe_dust = full(sf.fe_dust)
    if not sw.constant_CO2 and sf.co2_atm is None:
        raise ValueError("constant_CO2 is off but the step forcing has no co2_atm: build the forcing with bgc_co2 "
                         "(RunConfig.bgc_co2)")
    inputs = BgcStepInputs(daynew=sf.daynew, ndpyr=sf.ndpyr, mstep=mstep, fe_dust=fe_dust,
                           n_dust=jnp.zeros_like(fe_dust),
                           co2_atm=p.CO2_for_spinup if sf.co2_atm is None else jnp.asarray(sf.co2_atm, f64),
                           press_air=full(sf.press_air),
                           u_wind=full(sf.u_air), v_wind=full(sf.v_air), shortwave=full(sf.shortwave),
                           a_ice=full(a_ice))
    carb = CarbCache(*(st.bgc_carb[..., i] for i in range(len(CARB_FIELDS))))
    r1 = ckpt(lambda: recom_sources_sinks(full(X[:, :nl1]), carb, st.bgc_benthos, st.bgc_alk_surf,
                                          full(T0[:, :nl1]), full(S0[:, :nl1]), geom1, inputs, dt_bgc, p,
                                          reduce_sum=reduce_sum))()
    if _sp:      # stage 1's float64 result; stage 3 rounds it once, the residual into the carry (R1)
        X1 = jnp.concatenate([r1.bgc, full(X[:, nl1:])], axis=1)
    else:
        X1 = jnp.concatenate([r1.bgc.astype(X.dtype), X[:, nl1:]], axis=1)

    # ---- stage 3: the BGC tracers in FESOM2's tracer loop ----
    geom_sink = sinking_geometry(h, hnode_new[:, :nl1], zbar0_f, Z0_f[:, :nl1], area, areasvol,
                                 full(fs.bgc_seafloor), nlev, lat_rad / RAD, PR.full_tree(_saar(fs)))
    out3 = transport.recom_stage3(
        mesh, X1, X_old, uv_adv=uv_adv, w_e_adv=w_e_adv, w_i=w_i, helem=helem, hnode=h0, hnode_new=hnode_new,
        zbar3_live=zbar0, Z3d_live=Z0, Z3d_new=Z3d_new, gm_diag=gm_diag, Kv_eff=Kv_eff, temp=T_fin, salt=S_fin,
        sfc=r1.sfc, benthos=r1.benthos, geom_sink=geom_sink, params=p, switches=sw, dt=dt_bgc, exch=exch, ckpt=ckpt,
        adj=adj, helem_flux=helem_flux, return_terms=True, **({"carry": st.bgc_carry} if _sp else {}))
    X_new, X_old_new, benthos_new, res = out3[:4]

    # ---- 2-D diagnostics ----
    k100 = export_level(mesh)
    d2 = r1.sms.diag2d
    poc = jnp.array([SLOT[i] for i in POC_IDS])
    dsink = (res.values_mod[..., poc] - r1.bgc[..., poc]).sum(axis=-1) * h               # (nod, nl-1) content/area
    above = jnp.arange(nl1)[None, :] < k100
    export = jnp.where(nlev > k100, -jnp.sum(jnp.where(above, dsink, 0.0), axis=1) / dt_bgc * SECONDS_PER_DAY,
                       0.0)
    s = r1.sfc
    diag = jnp.stack([s.GloCO2flux, s.GloPCO2surf, d2["NPPn"] + d2["NPPd"], export, s.GloHplus, s.GloO2flux],
                     axis=-1)
    out = dict(bgc=X_new, bgc_old=X_old_new, bgc_benthos=benthos_new,
               bgc_carb=jnp.stack([getattr(r1.carb, f) for f in CarbCache._fields], axis=-1),
               bgc_mstep=st.bgc_mstep + 1, bgc_diag=diag)
    if _sp:
        out["bgc_carry"] = out3[4]

    # ---- budget sums ----
    if st.bgc_budget is not None:
        wet = (jnp.arange(nl1)[None, :] < nlev[:, None])[..., None]
        hv = (h * areasvol[:, :nl1])[..., None]
        content = lambda a: jnp.sum(jnp.where(wet, a * hv, 0.0), axis=1)                # noqa: E731
        terms = jnp.stack([transport.surface_bc(s, dt_bgc) * areasvol[:, :1],
                           content(r1.bgc - full(X[:, :nl1])),
                           content(res.values_mod - r1.bgc),
                           content(r1.sms.clip)], axis=1)
        out["bgc_budget"] = st.bgc_budget + terms
    return out


def inventory(X, hnode, mesh, owned=None, carry=None):
    """``sum X hnode areasvol`` over the wet layers (and the owned nodes, if a mask is given), per tracer
    (``X`` (nod, nl, ntr) or (nod, nl-1, ntr)): the quantity the transport conserves. ``carry`` (single precision,
    ``State.bgc_carry``): its content ``sum carry areasvol`` is added, as FESOM2 counts the carry
    (``fesom_conservation.F90:197``). Float64 throughout."""
    nl1 = mesh.nl - 1
    nlev = np.asarray(mesh.nlevels_nod2D) - 1
    wet = (np.arange(nl1)[None, :] < nlev[:, None])[..., None]
    av = np.asarray(mesh.areasvol, np.float64)[:, :nl1]
    cell = np.asarray(X, np.float64)[:, :nl1] * (np.asarray(hnode, np.float64)[:, :nl1] * av)[..., None]
    if carry is not None:
        cell = cell + np.asarray(carry, np.float64)[:, :nl1] * av[..., None]
    c = np.where(wet, cell, 0.0).sum(axis=1)
    if owned is not None:
        c = c[np.asarray(owned)]
    return c.sum(axis=0)


# ---------------------------------------------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------------------------------------------
_TRACER_OUT = {"DIC": 1002, "Alk": 1003, "DIN": 1001, "DSi": 1018, "DFe": 1019, "O2": 1022}


def output_fields(s, rcfg) -> dict:
    """The BGC fields ``rcfg.output`` names, from a State ``s`` (inside the chunk scan's per-step sum): 3-D tracers
    ``(nod, nl)`` (``Chl`` = PhyChl + DiaChl) and the 2-D diagnostics of ``State.bgc_diag``."""
    out = {}
    for name in rcfg.output:
        if name in _TRACER_OUT:
            out[name] = s.bgc[..., SLOT[_TRACER_OUT[name]]]
        elif name == "Chl":
            out[name] = s.bgc[..., SLOT[1006]] + s.bgc[..., SLOT[1015]]
        else:
            out[name] = s.bgc_diag[..., DIAG2D.index(name)]
    return out


def check_step_inputs(sf, fs, ale_cfg, sp: bool, bgc_every: int = 1) -> None:
    """The step's trace-time guards for full REcoM (raise with the reason). Single precision is an opt-in deviation
    (FESOM2 refuses USE_SINGLE_PRECISION with RECOM_COUPLED; plan 20261026-recom-lane-sp) for ``bgc_every = 1``
    only; the step has already refused SP outside the CORE2 production physics."""
    if sp and int(bgc_every) != 1:
        raise NotImplementedError("single precision + full REcoM needs recom.bgc_every: 1 (the BGC-every-k interval "
                                  "sums and the k-step stage have no single-precision scheme)")
    if ale_cfg is None:
        raise NotImplementedError("full REcoM needs zstar (ale: {}), the configuration it is ported for")
    if sf is None or sf.press_air is None or sf.fe_dust is None or sf.daynew is None:
        raise ValueError("full REcoM needs the forced path with forcing.mslp (press_air) and REcoM's per-step inputs "
                         "(build the forcing with bgc_dust_path=cfg.bgc_dust_path())")
    if fs is None or fs.bgc_seafloor is None:
        raise ValueError("full REcoM needs ForcingStatic with its node constants: "
                         "forcing_static = fesom_jax.recom.wiring.with_recom_static(forcing.static, mesh)")
