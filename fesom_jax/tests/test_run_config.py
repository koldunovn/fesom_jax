"""A3 gate: unified :class:`~fesom_jax.run_config.RunConfig` + single-YAML (``RUN_CONFIG_OK``).

The regression-guard invariant: **`RunConfig.defaults()` ⇒ a byte-identical step** vs a bare
:func:`fesom_jax.step.step` (config promotion must not move a single bit). Plus: the γ promotion
actually changes the dynamics (NG5's ``gamma1=0.2`` is not silently ignored); YAML round-trips;
the shipped ``configs/{ng5,core2_full}.yaml`` load + validate (ng5 → ``gamma1=0.2``, the
implemented tracer path); unknown keys raise; the tracer selector rejects an unimplemented scheme;
the dt-ramp logic.

  PY -m pytest fesom_jax/tests/test_run_config.py -x
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import jax.numpy as jnp

from fesom_jax import forcing, ic, momentum, ssh
from fesom_jax import step as stepmod
from fesom_jax.config import DT_DEFAULT, TracerConfig, ViscConfig
from fesom_jax.gm import GMConfig
from fesom_jax.kpp import KppConfig
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.run_config import DtRamp, RunConfig, load_yaml
from fesom_jax.state import State

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DT = 100.0


def _leaves_equal(a: State, b: State) -> bool:
    for f in dataclasses.fields(State):
        if getattr(a, f.name) is None:   # unused optional field
            continue
        if not np.array_equal(np.asarray(getattr(a, f.name)), np.asarray(getattr(b, f.name))):
            return False
    return True


# --------------------------------------------------------------------------
# 1. THE regression guard: RunConfig.defaults() ⇒ bit-identical step
# --------------------------------------------------------------------------
def test_defaults_bit_identical_step():
    mesh = load_mesh(DEFAULT_PI_MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress = forcing.surface_stress(mesh)
    st0 = ic.initial_state(mesh)
    cfg = RunConfig.defaults()

    # 5 steps so uv (and thus the viscosity filter) is genuinely exercised.
    bare = stepmod.run(st0, mesh, op, stress, 5, dt=DT)
    via = stepmod.run(st0, mesh, op, stress, 5, dt=DT, **cfg.physics_kwargs())
    assert _leaves_equal(bare, via), "RunConfig.defaults() step is NOT bit-identical to bare step()"


# --------------------------------------------------------------------------
# 2. The γ promotion is real — gamma1=0.2 changes the biharmonic viscosity output
# --------------------------------------------------------------------------
def test_visc_gamma_changes_dynamics():
    # The flow-aware γ1 term only bites when gamma1·|Δu| exceeds the gamma0=0.003 floor AND
    # gamma2·|Δu|² (so |Δu| ∈ ~(0.03, 0.35)) — pi's tiny velocities sit under the floor, so
    # exercise the kernel directly with a velocity field in that band.
    mesh = load_mesh(DEFAULT_PI_MESH_DIR)
    rng = np.random.default_rng(0)
    uv = jnp.asarray(0.1 * rng.standard_normal((mesh.elem2D, mesh.nl, 2)))
    uv_rhs = jnp.zeros((mesh.elem2D, mesh.nl, 2))

    default = momentum.visc_filt_bidiff(mesh, uv, uv_rhs, dt=DT)
    default2 = momentum.visc_filt_bidiff(mesh, uv, uv_rhs, dt=DT, visc_cfg=ViscConfig())
    ng5 = momentum.visc_filt_bidiff(mesh, uv, uv_rhs, dt=DT, visc_cfg=ViscConfig(gamma1=0.2))
    # visc_cfg=None and visc_cfg=ViscConfig() are byte-identical (the bit-identity invariant)
    assert np.array_equal(np.asarray(default), np.asarray(default2))
    # gamma1=0.2 genuinely changes the result (the threading is live, not silently ignored)
    assert not np.array_equal(np.asarray(default), np.asarray(ng5)), (
        "visc_cfg.gamma1 did not change visc_filt_bidiff — the γ is being silently ignored")


# --------------------------------------------------------------------------
# 3. YAML round-trip (overrides + spec) reconstructs an identical RunConfig
# --------------------------------------------------------------------------
def test_yaml_roundtrip(tmp_path):
    cfg = RunConfig(
        ale=None, gm=GMConfig(), kpp=KppConfig(), tke=None,
        visc=ViscConfig(gamma1=0.2), dt=180.0, dt_ramp=DtRamp(after_step=480, dt=240.0),
        mesh="/work/x/ng5", partition="dist_64",
        forcing={"kind": "core2", "start_year": 1958},
        output_dir="/work/x/out", snapshot_every=480, checkpoint_every=2400,
        n_steps=1440, duration="2yr")
    p = tmp_path / "rc.yaml"
    cfg.to_yaml(p)
    back = load_yaml(p)
    assert back == cfg, f"YAML round-trip differs:\n{cfg}\n!=\n{back}"


# --------------------------------------------------------------------------
# 4. The shipped configs load + validate; ng5 has the pinned values
# --------------------------------------------------------------------------
def test_ng5_config():
    cfg = load_yaml(CONFIGS / "ng5.yaml")
    assert cfg.visc.gamma1 == 0.2                  # the pinned NG5 viscosity
    assert cfg.gm is None                           # GM OFF
    assert cfg.tracer == TracerConfig()             # the implemented (0,1) path
    cfg.tracer.validate()                           # must not raise
    assert cfg.dt == 180.0 and cfg.dt_ramp.dt == 240.0


def test_core2_config_loads():
    cfg = load_yaml(CONFIGS / "core2_full.yaml")    # validates (incl. tke xor kpp)
    assert cfg.dt == 1800.0
    assert cfg.gm is not None                        # GM on for CORE2


def test_core2_gpu_config():
    """``configs/core2_gpu.yaml`` (CORE2 GPU speed plan, Task 3) = ``core2_full.yaml`` + the two
    certified speed options (Chebyshev CG preconditioner degree 3 / kappa 30, on-device forcing)
    and no default restart target. ``core2_full.yaml`` keeps both OFF: it is the adjoint's
    reference configuration (``scripts/adjoint/core2_adj.py`` refuses the preconditioner)."""
    full = load_yaml(CONFIGS / "core2_full.yaml")
    gpu = load_yaml(CONFIGS / "core2_gpu.yaml")
    assert gpu.ssh_cheb() == (3, 30.0)
    assert gpu.forcing["on_device"] is True
    assert gpu.restart_out is None
    assert gpu.gpu_env() == {"FESOM_HALO_COMPACT": "1", "FESOM_COLUMN_UNROLL": "1",
                             "FESOM_MEVP_WIDE": "12", "FESOM_SSH_ELL": "1", "FESOM_CG_BATCH": "4",
                             "FESOM_MEVP_UNROLL": "12", "FESOM_COLUMN_KERNEL": "1", "FESOM_SSH_INC_ELL": "1",
                             "FESOM_OUTPUT_ASYNC": "1", "FESOM_RESTART_ASYNC": "1", "FESOM_FORCING_AHEAD": "2"}
    assert gpu.gpu["lane_order"] == "rcm"           # applied by the driver, not an environment switch
    assert gpu.ssh_gathered() and not full.ssh_gathered()
    assert full.ssh_cheb()[0] == 0 and not full.forcing.get("on_device")
    assert full.gpu is None and full.gpu_env() == {}
    assert dataclasses.replace(gpu, ssh=full.ssh, forcing=full.forcing, gpu=full.gpu,
                               restart_out=full.restart_out) == full


def test_gpu_block_validation():
    with pytest.raises(KeyError, match="halo_compactt"):
        RunConfig.from_dict({"gpu": {"halo_compactt": True}}).validate()
    with pytest.raises(TypeError, match="gpu.halo_compact"):
        RunConfig.from_dict({"gpu": {"halo_compact": 1}}).validate()
    assert RunConfig.from_dict({"gpu": {"halo_compact": False}}).gpu_env() == {}


def test_gpu_mevp_wide_key(monkeypatch):
    """``gpu.mevp_wide`` is an integer K >= 0 (not a bool) and becomes ``FESOM_MEVP_WIDE=K``;
    0 is off. ``core2_gpu.yaml`` turns it on at K = 12 (user 2026-10-20)."""
    from fesom_jax import ice_wide
    for bad in (True, -1, 2.0, "8"):
        with pytest.raises(TypeError, match="gpu.mevp_wide"):
            RunConfig.from_dict({"gpu": {"mevp_wide": bad}}).validate()
    cfg = RunConfig.from_dict({"gpu": {"mevp_wide": 8, "halo_compact": True}})
    cfg.validate()
    assert cfg.gpu_env() == {"FESOM_MEVP_WIDE": "8", "FESOM_HALO_COMPACT": "1"}
    assert RunConfig.from_dict({"gpu": {"mevp_wide": 0}}).gpu_env() == {}
    # a string switch is passed through as written (it used to become "1")
    assert RunConfig.from_dict({"gpu": {"bgc_layout": "loop"}}).gpu_env() == {"FESOM_BGC_LAYOUT": "loop"}
    assert RunConfig.from_dict({"gpu": {"ssh_ell": True}}).gpu_env() == {"FESOM_SSH_ELL": "1"}
    assert RunConfig.from_dict({"gpu": {"cg_batch": 4}}).gpu_env() == {"FESOM_CG_BATCH": "4"}
    assert RunConfig.from_dict({"gpu": {"step_unroll": 2}}).gpu_env() == {"FESOM_STEP_UNROLL": "2"}
    assert RunConfig.from_dict({"gpu": {"ice_static": True}}).gpu_env() == {"FESOM_ICE_STATIC": "1"}
    assert RunConfig.from_dict({"gpu": {"chunk_tail": 32}}).gpu_env() == {"FESOM_CHUNK_TAIL": "32"}
    assert RunConfig.from_dict({"gpu": {"mevp_unroll": 12}}).gpu_env() == {"FESOM_MEVP_UNROLL": "12"}
    assert load_yaml(CONFIGS / "core2_gpu.yaml").gpu_env()["FESOM_MEVP_WIDE"] == "12"   # on since 8980bf8
    monkeypatch.delenv("FESOM_MEVP_WIDE", raising=False)
    assert ice_wide.mevp_wide_from_env() == 0
    monkeypatch.setenv("FESOM_MEVP_WIDE", "8")
    assert ice_wide.mevp_wide_from_env() == 8
    monkeypatch.setenv("FESOM_MEVP_WIDE", "x")
    with pytest.raises(ValueError, match="FESOM_MEVP_WIDE"):
        ice_wide.mevp_wide_from_env()


def test_gpu_halo_transport_and_chunk_steps():
    """``gpu.halo_transport`` and ``gpu.chunk_steps`` are driver keys (no environment switch); a transport other than
    all_gather refuses ``gpu.halo_compact``; bad values are refused."""
    cfg = RunConfig.from_dict({"gpu": {"halo_transport": "padded", "chunk_steps": 96, "column_unroll": True}})
    cfg.validate()
    assert cfg.halo_transport() == "padded" and cfg.chunk_steps() == 96
    assert cfg.gpu_env() == {"FESOM_COLUMN_UNROLL": "1"}
    assert RunConfig.from_dict({}).halo_transport() == "all_gather" and RunConfig.from_dict({}).chunk_steps() is None
    with pytest.raises(ValueError, match="halo_compact"):
        RunConfig.from_dict({"gpu": {"halo_transport": "padded", "halo_compact": True}}).validate()
    with pytest.raises(ValueError, match="halo_transport"):
        RunConfig.from_dict({"gpu": {"halo_transport": "pigeon"}}).validate()
    with pytest.raises(TypeError, match="chunk_steps"):
        RunConfig.from_dict({"gpu": {"chunk_steps": 0}}).validate()


def test_split_tails():
    """``gpu.chunk_tail``: a short chunk of a multiple of q steps becomes q-chunks; the bootstrap flag stays on the
    first; full chunks and other lengths are untouched; 0 is off."""
    from fesom_jax.run import Chunk, split_tails
    ch = [Chunk(0, 64, 2700.0, True), Chunk(64, 96, 2700.0, False), Chunk(160, 64, 2700.0, False),
          Chunk(224, 32, 2700.0, False), Chunk(256, 48, 2700.0, False)]
    got = split_tails(ch, 96, 32)
    assert [(c.start, c.count, c.bootstrap_ab2) for c in got] == [
        (0, 32, True), (32, 32, False), (64, 96, False), (160, 32, False), (192, 32, False), (224, 32, False),
        (256, 48, False)]
    assert split_tails(ch, 96, 0) == ch


def test_gpu_ordered_sums_key(monkeypatch):
    """``gpu.ordered_sums`` (a bool) becomes ``FESOM_ORDERED_SUMS=1``, read by ``ops.ordered_sums_enabled``;
    off unless the config turns it on (``core2_gpu.yaml`` does not, yet)."""
    from fesom_jax import ops
    with pytest.raises(TypeError, match="gpu.ordered_sums"):
        RunConfig.from_dict({"gpu": {"ordered_sums": 1}}).validate()
    cfg = RunConfig.from_dict({"gpu": {"ordered_sums": True}})
    cfg.validate()
    assert cfg.gpu_env() == {"FESOM_ORDERED_SUMS": "1"}
    assert RunConfig.from_dict({"gpu": {"ordered_sums": False}}).gpu_env() == {}
    assert "FESOM_ORDERED_SUMS" not in load_yaml(CONFIGS / "core2_gpu.yaml").gpu_env()
    monkeypatch.delenv("FESOM_ORDERED_SUMS", raising=False)
    assert not ops.ordered_sums_enabled()
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    assert ops.ordered_sums_enabled()
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "0")
    assert not ops.ordered_sums_enabled()


def test_run_from_config_gpu_env_is_scoped(monkeypatch):
    """``run_from_config`` sets the ``gpu:`` switches for the run only, and an explicit
    environment setting wins over the config."""
    import os
    from fesom_jax import run
    seen = {}

    @run._with_gpu_env
    def fake(cfg):
        seen.update({k: os.environ.get(k) for k in ("FESOM_HALO_COMPACT", "FESOM_COLUMN_UNROLL")})
        raise RuntimeError("boom")

    cfg = RunConfig.from_dict({"gpu": {"halo_compact": True, "column_unroll": True}})
    monkeypatch.delenv("FESOM_HALO_COMPACT", raising=False)
    monkeypatch.setenv("FESOM_COLUMN_UNROLL", "0")
    with pytest.raises(RuntimeError, match="boom"):
        fake(cfg)
    assert seen == {"FESOM_HALO_COMPACT": "1", "FESOM_COLUMN_UNROLL": "0"}
    assert "FESOM_HALO_COMPACT" not in os.environ
    assert os.environ["FESOM_COLUMN_UNROLL"] == "0"


def test_ice_ave_steps_config():
    """``ice.ice_ave_steps`` (speed plan Task 12): N >= 1 validates and ``ice_dt = N*dt``
    (``ice_setup_step.F90:56``); 0, negatives and non-integers are refused. Until Task 12 this
    was a guard refusing anything but 1 (the ice ran every step while ice_dt doubled)."""
    from fesom_jax.ice import IceConfig
    from fesom_jax.ice_step import _with_ice_dt
    for n in (1, 2, 3):
        RunConfig.from_dict({"ice": {"ice_ave_steps": n}}).validate()
        assert _with_ice_dt(IceConfig(ice_ave_steps=n), 1800.0).ice_dt == n * 1800.0
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="ice_ave_steps"):
            RunConfig.from_dict({"ice": {"ice_ave_steps": bad}}).validate()
    with pytest.raises(ValueError, match="ice_ave_steps"):
        _with_ice_dt(IceConfig(ice_ave_steps=0), 1800.0)
    assert _with_ice_dt(IceConfig(), 1800.0).ice_dt == 1800.0


# --------------------------------------------------------------------------
# 5. Unknown keys raise (no silent typos) — top level and sub-config
# --------------------------------------------------------------------------
def test_unknown_key_raises():
    with pytest.raises(KeyError):
        RunConfig.from_dict({"nonsense": 1})
    with pytest.raises(KeyError):
        RunConfig.from_dict({"visc": {"gamma9": 1.0}})        # bad γ name
    with pytest.raises(KeyError):
        RunConfig.from_dict({"gm": {"not_a_gm_field": 1.0}})  # bad sub-config key


# --------------------------------------------------------------------------
# 6. The tracer selector rejects an unimplemented scheme (a kernel change)
# --------------------------------------------------------------------------
def test_tracer_validation():
    TracerConfig().validate()                                  # the implemented path: OK
    with pytest.raises(NotImplementedError):
        TracerConfig(num_ord_hor=1).validate()                 # different num_ord ⇒ kernel work
    with pytest.raises(NotImplementedError):
        RunConfig(tracer=TracerConfig(hor_scheme="MUSCL")).validate()


# --------------------------------------------------------------------------
# 7. Mutually-exclusive mixing + dt-ramp logic
# --------------------------------------------------------------------------
def test_validate_and_dt_ramp():
    from fesom_jax.tke import TkeConfig
    with pytest.raises(ValueError):
        RunConfig(kpp=KppConfig(), tke=TkeConfig()).validate()  # both mixing schemes

    cfg = RunConfig(dt=180.0, dt_ramp=DtRamp(after_step=100, dt=240.0))
    assert cfg.dt_at(0) == 180.0 and cfg.dt_at(99) == 180.0
    assert cfg.dt_at(100) == 240.0 and cfg.dt_at(500) == 240.0
    assert cfg.is_ramp_step(100) and not cfg.is_ramp_step(101)


# --------------------------------------------------------------------------
# 7b. Archival restart config: validation + YAML round-trip
# --------------------------------------------------------------------------
def test_restart_archive_config():
    RunConfig().validate()                                        # off by default: never raises
    RunConfig(restart_archive_out="/x", restart_archive_period="year").validate()   # OK
    RunConfig(restart_archive_out="/x", restart_archive_period="month",
             restart_archive_length=3).validate()                 # OK, general N
    with pytest.raises(ValueError):
        RunConfig(restart_archive_out="/x", restart_archive_period="fortnight").validate()
    with pytest.raises(ValueError):
        RunConfig(restart_archive_out="/x", restart_archive_period="year",
                 restart_archive_length=0).validate()
    with pytest.raises(ValueError):
        RunConfig(restart_archive_out="/x", restart_archive_period=None).validate()  # out set, period not


def test_restart_archive_yaml_roundtrip(tmp_path):
    cfg = RunConfig(restart_archive_out="/work/x/archive", restart_archive_period="month",
                    restart_archive_length=2)
    p = tmp_path / "rc.yaml"
    cfg.to_yaml(p)
    back = load_yaml(p)
    assert back == cfg
    assert back.restart_archive_period == "month" and back.restart_archive_length == 2


# --------------------------------------------------------------------------
# 8. Aggregate gate
# --------------------------------------------------------------------------
def test_run_config_ok():
    assert RunConfig.defaults() == RunConfig()
    print("RUN_CONFIG_OK")


def test_executable_cache_key_follows_the_trace_time_switches(monkeypatch):
    """The forced runner's executable cache keys on the FESOM_* switches read at trace time, so a second run in
    one process with other `gpu:` options cannot reuse an executable traced under the first run's options."""
    from fesom_jax import integrate_sharded as ish
    monkeypatch.delenv("FESOM_HALO_COMPACT", raising=False)
    a = ish._trace_env()
    monkeypatch.setenv("FESOM_HALO_COMPACT", "1")
    b = ish._trace_env()
    assert a != b and ("FESOM_HALO_COMPACT", "1") in b
