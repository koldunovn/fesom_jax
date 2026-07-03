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
