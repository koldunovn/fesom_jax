"""REcoM plan Task 13 (lane W): full REcoM's restart and run-segment semantics.

FESOM2-REcoM restarts the BGC tracers (with their AB2 history) and ``Benthos``; ``Alk_surf`` is rebuilt from the IC at
every run start (fesom-jax keeps it in the restart: the same values); the carbonate cache and ``mstep`` are not in
the restart: ``mstep`` restarts at 1 in every run segment and the cache is refilled at that step
(``fesom_module.F90:849,983,1049``). So a split run is not the unsplit run (the twin's split pair shows it, lane F);
fesom-jax mirrors that. ``GloHplus`` (restart ``HPlus``) is output only and is not restarted here (``bgc_diag``).
"""

from __future__ import annotations

import dataclasses

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fesom_jax import halo, partit, shard_mesh, ssh, surface_forcing  # noqa: E402
from fesom_jax import integrate_sharded as ish  # noqa: E402
from fesom_jax import zarr_output as zo  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.recom import wiring  # noqa: E402
from fesom_jax.recom.config import RecomConfig  # noqa: E402
from fesom_jax.recom.ic import BgcIC  # noqa: E402
from fesom_jax.recom.tracers import NTR  # noqa: E402
from fesom_jax.state import BGC_FULL_FIELDS, RESTART_SKIP, State  # noqa: E402
from fesom_jax.tests import test_recom_step as TS  # noqa: E402


def _fake_ic(mesh, rng):
    n, nl1 = mesh.nod2D, mesh.nl - 1
    return BgcIC(bgc=rng.random((n, nl1, NTR)), bgc_old=rng.random((n, nl1, NTR)), alk_surf=rng.random(n),
                 benthos=rng.random((n, 4)), hplus=np.zeros(n))


def test_full_fields_restart_roundtrip(tmp_path):
    mesh = load_mesh()
    rng = np.random.default_rng(1)
    rcfg = RecomConfig(namelist="x", budget=True)
    g = wiring.with_bgc_state(jax.tree.map(np.asarray, State.rest(mesh)), _fake_ic(mesh, rng), rcfg)
    g = dataclasses.replace(g, bgc_mstep=g.bgc_mstep + 7, bgc_carb=g.bgc_carb + 1.0, bgc_diag=g.bgc_diag + 2.0)
    assert g.bgc_carb.shape == (mesh.nod2D, mesh.nl - 1, 8) and g.bgc_mstep.dtype == np.int32
    assert g.bgc_budget.shape == (mesh.nod2D, len(wiring.BUDGET_TERMS), NTR)
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, ser)
    fs, spec = ish.folded_state(shard_mesh.partition_state(g, ser))
    placed = ish._to_global_sharded(fs, spec, halo.device_mesh("p", devices=jax.devices()[:1]))
    zo.write_restart(tmp_path / "r", placed, sm, ser, step=5, calendar_date="1958-doy001", dt_stage=2700.0)
    import zarr
    keys = set(zarr.open_group(str(tmp_path / "r"), mode="r").array_keys())
    assert {"bgc", "bgc_old", "bgc_benthos", "bgc_alk_surf"} <= keys
    assert not (set(RESTART_SKIP) & keys)
    loaded, _ = zo.read_restart(tmp_path / "r", mesh, ser)
    for f in RESTART_SKIP:
        assert getattr(loaded, f) is None, f
    for f in ("bgc", "bgc_old", "bgc_benthos", "bgc_alk_surf"):
        np.testing.assert_array_equal(np.asarray(getattr(loaded, f))[: mesh.nod2D], getattr(g, f), err_msg=f)
    seg = wiring.start_segment(loaded, rcfg)
    assert int(np.asarray(seg.bgc_mstep).max()) == 0 and np.asarray(seg.bgc_mstep).dtype == np.int32
    assert float(np.abs(np.asarray(seg.bgc_carb)).max()) == 0.0
    assert np.asarray(seg.bgc_carb).shape[1:] == (mesh.nl - 1, 8)
    assert np.asarray(seg.bgc_budget).shape[1:] == (len(wiring.BUDGET_TERMS), NTR)
    # host [P, Lmax] layout (the run driver's cold start) keeps the lane axes
    hp = wiring.start_segment(shard_mesh.partition_state(g, ser), rcfg)
    assert hp.bgc_carb.shape[:2] == hp.T.shape[:2] and hp.bgc_diag.shape == hp.T.shape[:2] + (len(wiring.DIAG2D),)
    with pytest.raises(ValueError):
        wiring.start_segment(dataclasses.replace(loaded, bgc_benthos=None), rcfg)
    assert set(BGC_FULL_FIELDS) >= set(RESTART_SKIP)


@TS.need_twin
def test_split_run_segment_semantics(tmp_path):
    """3 steps through run_from_config, once continuous and once as 2 + restart + 1: the physics is bitwise the same
    (fesom-jax's restart is continuous), the BGC run segment restarts as FESOM2's: mstep 1 at step 3, the carbonate
    cache refilled there (so it differs from the continuous run's, refreshed last at step 1)."""
    from fesom_jax.run import run_from_config
    fx = TS._setup()
    cfg, mesh = fx["cfg"], fx["mesh"]
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, ser)
    sop = ssh.partition_ssh_operator(fx["op"], ser)

    def forcing():           # one per run segment (BgcForcing's mstep == 1 is the segment's first step)
        return surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=np.asarray(fx["st"].T[:, 0]), dt=cfg.dt,
                                                     mslp=True, bgc_dust_path=cfg.bgc_dust_path(),
                                                     **cfg.forcing_paths())

    def run(c, state0, out):
        # chunks of 2: the continuous run is chunk [0, 2) + a continuation chunk [2, 3), the same two programs as
        # the split run (a differently chunked program may round differently, by an ulp)
        return run_from_config(c, mesh=mesh, part=ser, sm=sm, sop=sop, forcing=forcing(), state0=state0, year=1958,
                               chunk_steps=2, out_dir=str(out))

    cont = run(dataclasses.replace(cfg, n_steps=3, duration=None), fx["st"], tmp_path / "cont")
    run(dataclasses.replace(cfg, n_steps=2, duration=None), fx["st"], tmp_path / "seg1")
    split = run(dataclasses.replace(cfg, n_steps=1, duration=None, restart_in=str(tmp_path / "seg1")), None,
                tmp_path / "seg2")
    a, b = jax.device_get(cont.state_p), jax.device_get(split.state_p)
    n = mesh.nod2D
    np.testing.assert_array_equal(np.asarray(a.T)[:n], np.asarray(b.T)[:n])
    np.testing.assert_array_equal(np.asarray(a.uv), np.asarray(b.uv))
    assert int(np.asarray(a.bgc_mstep)[0]) == 3 and int(np.asarray(b.bgc_mstep)[0]) == 1
    wet = np.arange(mesh.nl - 1)[None, :] < (np.asarray(mesh.nlevels_nod2D) - 1)[:, None]
    ca, cb = np.asarray(a.bgc_carb)[:n, :, 0], np.asarray(b.bgc_carb)[:n, :, 0]
    frac = float(np.mean(ca[wet] != cb[wet]))
    dX = float(np.max(np.abs(np.asarray(a.bgc)[:n] - np.asarray(b.bgc)[:n])))
    print(f"SPLIT: carbonate cache differs in {frac:.3f} of the wet cells; max |dBGC| {dX:.3e}")
    assert frac > 0.99


def test_full_recom_configs():
    """configs/pi_recom_bgc.yaml and configs/core2_recom.yaml load as full REcoM with hashable configs; full REcoM
    refuses linfs, single precision with bgc_every > 1 (single precision with bgc_every 1 is the opt-in deviation of
    plan 20261026-recom-lane-sp, tested in test_recom_sp.py), and an unknown output field."""
    from pathlib import Path
    from fesom_jax.run_config import RunConfig, load_yaml
    repo = Path(__file__).resolve().parents[2]
    for name in ("pi_recom_bgc.yaml", "core2_recom.yaml"):
        cfg = load_yaml(repo / "configs" / name)
        assert cfg.recom.mode == "full" and isinstance(cfg.recom.output, tuple)
        hash(cfg.recom)
        assert cfg.physics_kwargs()["recom_cfg"] is cfg.recom
    d = load_yaml(repo / "configs" / "pi_recom_bgc.yaml").to_dict()
    assert RunConfig.from_dict(d).recom == load_yaml(repo / "configs" / "pi_recom_bgc.yaml").recom
    with pytest.raises(NotImplementedError):
        RunConfig.from_dict(dict(d, ale=None)).validate()
    with pytest.raises(NotImplementedError):
        RunConfig.from_dict(dict(d, precision="single", recom=dict(d["recom"], bgc_every=2))).validate()
    with pytest.raises(ValueError):
        RunConfig.from_dict(dict(d, recom=dict(d["recom"], output=["DIC", "nope"]))).validate()
