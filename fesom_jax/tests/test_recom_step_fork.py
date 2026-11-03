"""Lane W of the spectral port (docs/plans/20261028-recom-spectral-port.md), gate G4b for the fork: one full JAX
step on the fork's pi bundle from the physics-matched twin's step-0 state against the twin's step 1, for every BGC
tracer of the fork's layout (37 with CDOM on the spectral twin, 36 on the broadband twin), ``Benthos`` and stage 1's
surface diagnostics, judged against the physics floor of the same pair and step (T and S, which reach the BGC
tracers through the transport). A copy of ``test_recom_step.py::test_g4b_one_step_vs_twin`` (REcoM 692f804) with
the fork's reference, layout, light and the three host-model legacy settings of its FESOM2 base.

The twins (``scripts/albedo/run_pi_recom_fork.sbatch``, one day, dumps at steps 1, 2, 3, 17, 32) run the namelist
sets ``scripts/recom/fortran/namelists_albedo_spectral_jaxphys/`` and ``..._nowb_jaxphys/``: the user's REcoM setup
with the physics brought as close to fesom-jax as the fork's FESOM2 base allows (their README: GM/Redi off on both
sides since the fork's ``init_Redi_GM`` is the FESOM 1.4-style builder fesom-jax does not have, cvmix TKE, mEVP,
``ice_diff`` 10, no Monin-Obukhov mixing). The JAX side is ``configs/albedo/pi_recom_spectral_g4b.yaml`` (``gm:
null``, ``spectral_co2lim: false``) plus the three legacy settings of the fork's FESOM2 base as step kwargs:
``legacy_fct_bounds`` (the FCT limiter bounds without the thickness), ``legacy_sw_depth`` (the absolute shortwave
attenuation depth before FESOM2 #1116) and ``legacy_k33_weights`` (dead with ``gm: null``, kept for the record).
"""

from __future__ import annotations

import dataclasses
import os
from functools import lru_cache
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fesom_jax import ssh, surface_forcing  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax.io_dump import load_recom_dump  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.recom import reference as R  # noqa: E402
from fesom_jax.recom import wiring  # noqa: E402
from fesom_jax.run_config import load_yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REF = Path(os.environ.get("FESOM_RECOM_FORK_REF", "/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/recom_ref"))
CFG = REPO / "configs" / "albedo" / "pi_recom_spectral_g4b.yaml"
RUN_CFG = REPO / "configs" / "albedo" / "pi_recom_spectral.yaml"      # the production spectral config (gm on, no legacy)
NML_NOWB = REPO / "scripts" / "recom" / "fortran" / "namelists_albedo_spectral_nowb_jaxphys" / "namelist.recom"
# the physics-matched twins: the spectral build (37 tracers, the spectral light) and the broadband build (36)
# The same REcoM on FESOM 2.8 (sveta-loza/fesom2 fesom2.8_recom_spectral,
# docs/plans/completed/20261102-fortran-spectral-on-fesom28.md): its own config (GM/Redi on, mesh_pi_fork) and none
# of the legacy settings.
CFG28 = REPO / "configs" / "albedo" / "pi_recom_spectral_g4b_fesom28.yaml"
FESOM28_RUNS = ("fesom28_spectral_pi_1d_dumps",)
RUNS = [r for r in os.environ.get("FESOM_RECOM_FORK_G4B_RUNS",
                                  "pi_1d_spectral_jaxphys,pi_1d_nowb_jaxphys," + ",".join(FESOM28_RUNS)).split(",")
        if (REF / r / "dumps" / "step_000001").is_dir()]
HAVE = bool(RUNS) and CFG.is_file()
need_twin = pytest.mark.skipif(not HAVE, reason=f"the fork's physics-matched twin dumps or the gate config are absent "
                                                f"({REF}, {CFG})")
# A BGC tracer's step-1 error, relative to its max, may be at most G4B_FACTOR times the larger relative error of T
# and S after the same step (the physics floor of the pair), as the 692f804 gate; and the floor itself must stay at
# the measured floor of this pair, else the factor would admit anything. Measured (jobs 48606105/48606106, the twins
# with upstream 66658357 cherry-picked: without it the first forcing window's winds stay unrotated and the floor is
# 3.5e-3): T 5.4e-5, S 1.6e-5, a_ice 3.2e-4; the fork's FESOM2 base is a year older than the physics fesom-jax
# follows (the 692f804 pair: 4.1e-8). FLOOR_MAX is twice that; a floor above it means the physics match was lost.
G4B_FACTOR = 100.0
FLOOR_MAX = 1.1e-4
# The same REcoM on FESOM 2.8 (FESOM28_RUNS) with GM/Redi on and none of the legacy settings: floor 4.09e-8 (T; a_ice
# 2.7e-7; job 48698151 on the twin of job 48698141), the 692f804 pair's 4.1e-8. FLOOR_MAX_28 is twice that.
FLOOR_MAX_28 = 8.2e-8
EXTRA = ("recom_out.NPPc", "recom_out.NPPp", "recom_out.GloO2flux", "recom_in.a_ice")


def _config(run: str, yaml=None):
    cfg = load_yaml(yaml or (CFG28 if run in FESOM28_RUNS else CFG))
    if "spectral" not in run:                 # the broadband build: the 36-tracer namelist, no spectral light
        cfg = dataclasses.replace(cfg, recom=cfg.recom._replace(namelist=str(NML_NOWB), spectral=False))
    return cfg


@lru_cache(maxsize=4)
def _setup(run: str, yaml=None):
    """The twin ``run``'s step-0 state, forcing and operators under the config ``yaml`` (the gate's by default; the
    sharded and gradient gates pass the production spectral config)."""
    cfg = _config(run, yaml)
    phys = cfg.physics_kwargs()
    lay = cfg.recom.layout
    mesh = load_mesh(cfg.mesh) if cfg.mesh else load_mesh()      # mesh: null = the packaged pi mesh
    d1 = load_recom_dump(REF / run / "dumps", 1, names=R.dump_names(layout=lay) + list(EXTRA))[0]
    st = R.twin_step0_state(mesh, d1, phys["ale_cfg"], cfg.recom, layout=lay)
    cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=np.asarray(st.T[:, 0]), dt=cfg.dt, mslp=True,
                                               bgc_dust_path=cfg.bgc_dust_path(), bgc_oasim=cfg.bgc_oasim(),
                                               **cfg.forcing_paths())
    fs = wiring.with_recom_static(cf.static, mesh, cfg.recom)
    op = ssh.build_ssh_operator(mesh, dt=cfg.dt)
    return dict(cfg=cfg, phys=phys, lay=lay, mesh=mesh, d1=d1, st=st, cf=cf, fs=fs, op=op,
                legacy=run not in FESOM28_RUNS)


# the fork's FESOM2 base (cda35117, 2025-10): the FCT limiter bounds without the thickness (before FESOM2
# 18346243 / 9afdae7b), the absolute shortwave attenuation depth (before #1116, 4aab0efd) and the pre-#944 K33
# weights (dead with gm null); step.py's docstring: "the REcoM spectral fork needs all three". Twin gates only.
LEGACY = dict(legacy_fct_bounds=True, legacy_sw_depth=True, legacy_k33_weights=True)


def _run(fx, n: int, legacy: bool = True):
    """``n`` dense steps from the twin's step 0; the states after each step (index 0: the start)."""
    mesh = fx["mesh"]
    dates = surface_forcing.dates_for_steps(1958, fx["cfg"].dt, n)
    st = jax.tree.map(jnp.asarray, fx["st"])
    states = [st]
    for i in range(n):
        sf = fx["cf"].step_forcing(*dates[i])
        st = stepmod.step_jit(st, mesh, fx["op"], jnp.zeros((mesh.elem2D, 2)), dt=fx["cfg"].dt,
                              is_first_step=(i == 0), step_forcing=sf, forcing_static=fx["fs"],
                              **(LEGACY if legacy else {}), **fx["phys"])
        states.append(st)
    return states


def _one_step(fx):
    return jax.device_get(_run(fx, 1, legacy=fx["legacy"])[1])


RESTART = os.environ.get("FESOM_G6_FORK_RESTART",
                         "/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/pi_recom_spectral/restart")


def warm_start(fx, restart=RESTART):
    """``(state, forcing, start day)`` for the gates that want a grown ecosystem: the 30-day spectral run's restart
    (``restart``; the lane-W run of job 48603201: a bloom with grazers and detritus above REcoM's ``tiny`` floor)
    as a new run segment (mstep 1), else the twin's step 0 (day 1)."""
    mesh, cfg = fx["mesh"], fx["cfg"]
    if not Path(restart, "zarr.json").exists() and not Path(restart, ".zgroup").exists():
        return jax.tree.map(jnp.asarray, fx["st"]), fx["cf"], 1
    from fesom_jax import partit
    from fesom_jax import zarr_output as zo
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    st, meta = zo.read_restart(restart, mesh, ser)
    st = wiring.start_segment(jax.tree.map(lambda a: jnp.asarray(np.asarray(a)), st), cfg.recom)
    assert st.T.shape == fx["st"].T.shape and st.bgc.shape == fx["st"].bgc.shape
    day = 1 + int(meta["step"]) * int(cfg.dt) // 86400
    cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=np.asarray(fx["st"].T[:, 0]), dt=cfg.dt,
                                               mslp=True, bgc_dust_path=cfg.bgc_dust_path(),
                                               bgc_oasim=cfg.bgc_oasim(), **cfg.forcing_paths())
    return st, cf, day


def _rel(a, b, mask):
    a, b = np.asarray(a)[mask], np.asarray(b)[mask]
    den = float(np.max(np.abs(b)))
    return (float(np.max(np.abs(a - b))) / den if den > 0 else float(np.max(np.abs(a)))), den


def g4b_errors(run: str):
    """``{name: (rel. error, max |fortran|)}`` after one JAX step from the twin's step 0: every tracer of the
    layout, Benthos, T, S, the ice concentration and stage 1's surface diagnostics."""
    fx = _setup(run)
    mesh, lay, f = fx["mesh"], fx["lay"], fx["d1"]
    nl1 = mesh.nl - 1
    st1 = _one_step(fx)
    nlev = np.asarray(f["mesh.nlevels_nod2D"]).astype(int) - 1
    wet = np.arange(nl1)[None, :] < nlev[:, None]
    allnod = np.ones(mesh.nod2D, bool)
    e = {"T": _rel(st1.T[:, :nl1], f["step_end.tr1"], wet), "S": _rel(st1.S[:, :nl1], f["step_end.tr2"], wet),
         "a_ice": _rel(st1.a_ice, f["recom_in.a_ice"], allnod)}       # recom_in(1) holds the ice after step 1
    for s, tid in enumerate(lay.ids):
        e[f"{tid}_{lay.names[s]}"] = _rel(st1.bgc[:, :nl1, s], f[f"step_end.tr{tid}"], wet)
    for i in range(4):
        e[f"Benthos{i + 1}"] = _rel(st1.bgc_benthos[:, i], f["step_end.Benthos"][i], allnod)
    dg = dict(zip(wiring.DIAG2D, np.moveaxis(np.asarray(st1.bgc_diag), -1, 0)))
    e["co2flux"] = _rel(dg["co2flux"], f["recom_out.GloCO2flux"], allnod)
    e["o2flux"] = _rel(dg["o2flux"], f["recom_out.GloO2flux"], allnod)
    e["pco2surf"] = _rel(dg["pco2surf"], f["recom_out.GloPCO2surf"], allnod)
    e["ph"] = _rel(dg["ph"], f["recom_out.GloHplus"], allnod)
    npp = f["recom_out.NPPn"] + f["recom_out.NPPd"] + f["recom_out.NPPc"] + f["recom_out.NPPp"]
    e["npp"] = _rel(dg["npp"], npp, allnod)
    return e


@need_twin
@pytest.mark.parametrize("run", RUNS)
def test_g4b_fork_one_step_vs_twin(run):
    e = g4b_errors(run)
    floor = max(e["T"][0], e["S"][0])
    for k, v in sorted(e.items(), key=lambda kv: -kv[1][0]):
        print(f"G4B-fork {run} {k:>18}: rel {v[0]:.2e} (max |F| {v[1]:.3e})")
    print(f"G4B-fork {run} physics floor (max rel of T, S) {floor:.2e}, a_ice {e['a_ice'][0]:.2e}")
    assert floor < (FLOOR_MAX_28 if run in FESOM28_RUNS else FLOOR_MAX), \
        f"the pair's physics is not matched (T/S floor {floor:.2e})"
    bad = {k: v for k, v in e.items() if k not in ("T", "S", "a_ice") and v[0] > max(G4B_FACTOR * floor, 1e-12)}
    assert not bad, (floor, bad)
