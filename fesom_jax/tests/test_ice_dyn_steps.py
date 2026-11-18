"""``IceConfig.ice_dyn_steps`` (not FESOM2; fesom3_jax's option and its tests/test_ice_dyn_steps.py): the sea-ice
dynamics and advection (EVP/mEVP, TG/FCT, cut_off) every N steps with ice_dt = N*dt, the velocities and stresses held
on the steps between, the thermodynamics, the ice-ocean stress and the fluxes every step with dt
(:func:`fesom_jax.ice_step.ice_dyn_cfgs`). In the coupled ocean's mode F (:mod:`fesom_jax.coupled`, the TCo319-NG5
path) on the packaged pi mesh from its cold start (PHC + FESOM's initial sea ice), a synthetic atmosphere held over
the intervals:

* the configuration: the default is 1 (both time steps the same object), N = 2 gives ice_dt 2*dt to the dynamics and
  dt to the thermodynamics, not with ice_ave_steps > 1, an interval without the static phase is refused;
* a step between two dynamics steps holds u_ice, v_ice and the stresses bit for bit and runs the thermodynamics;
* intervals of 1, 1 and 2 steps (phases 0, 1, 0: the phase comes from the absolute step) are the one interval of 4
  steps up to the programs' other fusion (1e-15), and a restart store written after the first (odd) step resumes the
  cycle to the same State bit for bit;
* dist_2 (two fake CPU devices, padded halo, the mEVP wide halo) gives the serial run's State, as close as it does
  with the dynamics every step.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=2`` (or more) for the dist_2 test; the SSS and runoff
climatologies from ``$FESOM_SSS_PATH`` / ``$FESOM_RUNOFF_PATH`` (or the Levante defaults)."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PI = ROOT / "fesom_jax" / "data" / "mesh_pi"
IC = ROOT / "fesom_jax" / "data" / "ic_pi"
DT = 900.0
ICE = ("u_ice", "v_ice", "sigma11", "sigma12", "sigma22")
FIELDS = ICE + ("a_ice", "m_ice", "m_snow", "T", "S", "uv", "eta_n", "cpl_ice_temp")


def _have_inputs() -> bool:
    from fesom_jax import paths
    try:
        paths.require("sss_path")
        paths.require("runoff_path")
    except FileNotFoundError:
        return False
    return PI.is_dir() and IC.is_dir()


need = pytest.mark.skipif(not _have_inputs(), reason="needs the packaged pi mesh + IC and the SSS / runoff "
                                                     "climatologies ($FESOM_SSS_PATH, $FESOM_RUNOFF_PATH)")


def _cfg_file(tmp_path, ndyn: int, name: str) -> str:
    """configs/ng5_cpl_flux.yaml (mode F's physics: zstar, TKE, mEVP, melt ponds) on pi, double precision, CG SSH."""
    d = yaml.safe_load((ROOT / "configs" / "ng5_cpl_flux.yaml").read_text())
    d.update(mesh=str(PI), partition="serial", restart_in=None, output_dir=None, dt=DT,
             gpu={"halo_transport": "padded"})
    d["ice"] = dict(d["ice"], ice_dyn_steps=ndyn)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return str(p)


def _init(tmp_path, ndyn, name, partition=None, restart_in=None):
    from fesom_jax import coupled as C
    return C.init_ocean(_cfg_file(tmp_path, ndyn, name), mesh_dir=str(PI), restart_in=restart_in, ic_dir=str(IC),
                        start=(1990, 1, 1), year0=1990, nsteps=4, mode="flux",
                        partition=partition, dist_dir=str(PI), log=lambda *a, **k: None)


def _forcing(n):
    """A cold, windy synthetic atmosphere (OASIS names): the ice moves and grows."""
    from fesom_jax import coupled as C
    rng = np.random.default_rng(7)
    f = {k: jnp.asarray(s * rng.standard_normal(n).astype(np.float32)) for k, s in
         (("A_TauX_oce", 0.1), ("A_TauY_oce", 0.1), ("A_TauX_ice", 0.2), ("A_TauY_ice", 0.2))}
    f.update({k: jnp.asarray(np.full(n, v, np.float32)) for k, v in
              (("A_Qns_oce", -200.0), ("A_Qs_all", 20.0), ("A_Q_ice", -80.0))})
    f.update({k: jnp.asarray(1e-8 * np.abs(rng.standard_normal(n)).astype(np.float32)) for k in
              ("A_Precip_liquid", "A_Precip_solid", "A_Evap", "A_Subl")})
    f["A_Ice_tskin"] = jnp.asarray((255.0 + rng.standard_normal(n)).astype(np.float32))
    f["hydr_oce"] = jnp.zeros((n,), jnp.float32)
    f["calv_oce"] = jnp.zeros((n,), jnp.float32)
    return C.oasis_to_fesom(f)


def _view(state, static, names=FIELDS):
    from fesom_jax import coupled as C
    return {k: np.asarray(v) for k, v in C.global_view(state, static, names).items()}


def _rel(a, b):
    return float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(a))), 1e-30)


def test_config():
    from fesom_jax.ice import IceConfig
    from fesom_jax.ice_step import ice_dyn_cfgs, ice_dyn_steps
    from fesom_jax.run_config import RunConfig
    one = IceConfig(whichEVP=1)
    assert one.ice_dyn_steps == 1 and ice_dyn_steps(None) == 1
    d, t = ice_dyn_cfgs(one, 360.0)
    assert d is t and d.ice_dt == 360.0                       # the default: one config, the program as before
    d, t = ice_dyn_cfgs(one._replace(ice_dyn_steps=2), 360.0)
    assert d.ice_dt == 720.0 and t.ice_dt == 360.0 and d.Tevp_inv == 3.0 / 720.0
    with pytest.raises(ValueError, match="choose one"):
        ice_dyn_cfgs(one._replace(ice_dyn_steps=2, ice_ave_steps=2), 360.0)
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="ice_dyn_steps"):
            RunConfig(ice=one._replace(ice_dyn_steps=bad)).validate()
    RunConfig(ice=one._replace(ice_dyn_steps=3)).validate()


@need
def test_cpl_phase_held_steps_and_restart(tmp_path):
    """Intervals 1 + 1 + 2 = one interval of 4 (up to the programs' other fusion, as gpu.ice_static's test); the held
    step bit for bit; a restart after the odd first step = the 1 + 1 + 2 run bit for bit (the same programs)."""
    from fesom_jax import coupled as C
    from fesom_jax.integrate_sharded import run_steps_sharded_forced
    state, st = _init(tmp_path, 2, "a")
    assert st.ice_phase() == 0
    fo = _forcing(st.nod2D)
    s0 = _view(state, st)
    assert float(np.max(s0["a_ice"])) > 0.5                   # the pi IC has sea ice to move
    # one interval of 4 steps: the bootstrap step (dynamics), a cycle (held, dynamics), the inline held step
    ref = C.ocean_interval(state, st, fo, nsteps=4)
    assert st.step == 4 and sorted(st._exe) == [(4, True, 0)]
    ref = _view(ref, st)

    state, st = _init(tmp_path, 2, "b")
    s = C.ocean_interval(state, st, fo, nsteps=1)              # step 0: dynamics
    a1 = _view(s, st)
    for f in ICE:
        assert not np.array_equal(a1[f], s0[f]), f"step 0 (dynamics): {f} did not move"
    rst = C.write_ocean_restart(s, st, str(tmp_path / "r1"))   # after step 0: a restart in the middle of the cycle
    assert st.ice_phase() == 1
    s = C.ocean_interval(s, st, fo, nsteps=1)                  # step 1: held
    a2 = _view(s, st)
    for f in ICE:
        np.testing.assert_array_equal(a2[f], a1[f], err_msg=f"step 1 (held): {f}")
    assert not np.array_equal(a2["m_ice"], a1["m_ice"]), "step 1 (held): no thermodynamics"
    s = C.ocean_interval(s, st, fo, nsteps=2)                  # steps 2 (dynamics), 3 (held)
    assert sorted(st._exe) == [(1, False, 1), (1, True, 0), (2, False, 0)]
    b = _view(s, st)
    rel = {f: _rel(ref[f], b[f]) for f in FIELDS}
    print("\n[ice_dyn_steps 1+1+2 vs 4 steps] max|d|/max|a|:", {f: f"{v:.1e}" for f, v in rel.items() if v > 0})
    bad = {f: v for f, v in rel.items() if v > 1e-10}
    assert not bad, bad

    state, st = _init(tmp_path, 2, "c", restart_in=rst)       # the restart: step 1, phase 1, no AB2 bootstrap
    assert st.step == 1 and st.ice_phase() == 1 and not st.bootstrap
    s = C.ocean_interval(state, st, fo, nsteps=1)
    s = C.ocean_interval(s, st, fo, nsteps=2)
    got = _view(s, st)
    for f in FIELDS:
        np.testing.assert_array_equal(got[f], b[f], err_msg=f"restart after step 0: {f}")

    # the static phase is required (a lax.cond on the device phase would sync the host every step)
    with pytest.raises(ValueError, match="needs ice_phase"):
        run_steps_sharded_forced(st.sm, s, st.sop, st.stress_p, None, st.fs_p, 2, dt=st.dt, npes=1,
                                 state_is_folded=True, **st.phys)


@need
def test_cpl_dist2_is_serial(tmp_path, monkeypatch):
    """dist_2 (padded halo, mEVP wide halo K=4) against serial, 5 steps, ice dynamics every step and every 2nd step:
    the held steps read halo values no exchange refreshed in the step, so their halos must be the owners'."""
    if len(jax.devices()) < 2:
        pytest.skip("needs 2 (fake) devices: XLA_FLAGS=--xla_force_host_platform_device_count=2")
    from fesom_jax import coupled as C
    monkeypatch.setenv("FESOM_MEVP_WIDE", "4")
    rel = {}
    for ndyn in (1, 2):
        out = {}
        for part in ("serial", "dist_2"):
            state, st = _init(tmp_path, ndyn, f"d{ndyn}_{part}", partition=part)
            assert (st.mevp_wide is not None) == (part == "dist_2")
            fo = _forcing(st.nod2D)
            state = C.ocean_interval(state, st, fo, nsteps=2)
            state = C.ocean_interval(state, st, fo, nsteps=3)
            out[part] = _view(state, st)
        rel[ndyn] = {f: _rel(out["serial"][f], out["dist_2"][f]) for f in FIELDS}
        for f in FIELDS:
            assert np.all(np.isfinite(out["dist_2"][f])), f
    print("\n[ice_dyn_steps dist_2 vs serial] max|d|/max|a|:",
          {n: {f: f"{v:.1e}" for f, v in r.items() if v > 0} for n, r in rel.items()})
    # the sharded sums' round-off, grown over 5 steps; N = 2 no further from serial than N = 1 (plus a margin)
    for f in FIELDS:
        assert rel[2][f] <= max(1e-9, 10.0 * rel[1][f]), (f, rel[1][f], rel[2][f])


def test_standalone_driver_chunks(tmp_path):
    """The file-forced driver (scripts/run_from_config.py, the core2_gpu.yaml options on pi, 4 fake CPU devices, the
    mEVP wide halo): ``--ice-dyn-steps 2`` over 6 steps in one chunk and in chunks of 3 (the second starts mid-cycle,
    phase 1) agree up to the chunks' other fusion (as gpu.ice_static's test), and differ from the dynamics every
    step."""
    from fesom_jax.tests.test_precision_shard import _cfg, _load, _run
    cfg = _cfg(tmp_path)
    got = {}
    for name, extra in (("one", ("--ice-dyn-steps", "2", "--chunk-steps", "6")),
                        ("three", ("--ice-dyn-steps", "2", "--chunk-steps", "3")),
                        ("every", ("--chunk-steps", "6"))):
        out = _run(cfg, "--steps", "6", *extra, "--restart-out", str(tmp_path / name))
        assert ("ice_dyn_steps 2" in out) == (name != "every"), name
        got[name] = _load(tmp_path / name)
    rel = {k: _rel(a, got["three"][k]) for k, a in got["one"].items() if np.issubdtype(a.dtype, np.floating)}
    print("\n[ice_dyn_steps driver chunks 6 vs 3+3] max|d|/max|a|:",
          {k: f"{v:.1e}" for k, v in sorted(rel.items(), key=lambda kv: -kv[1]) if v > 0})
    bad = {k: v for k, v in rel.items() if v > (1e-8 if k in ("Kv", "Av", "tke") else 1e-10)}
    assert not bad, bad
    assert not np.array_equal(got["one"]["u_ice"], got["every"]["u_ice"])
