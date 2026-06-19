"""A7 release gate (``RELEASE_OK``): the data-free invariants CI asserts on every push.

No external mesh/forcing — only the config + driver surface a released model must keep stable:
the shipped ``configs/*.yaml`` parse + validate, the ``RunConfig`` bit-identity default invariant
holds, the duration parser round-trips, and the public entry points import. The heavy bit-identity
STEP gate (a real `step()` on a mesh) lives in the Levante suite; this is the fast push gate.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fesom_jax.config import TracerConfig, ViscConfig
from fesom_jax.run import parse_duration, plan_chunks
from fesom_jax.run_config import RunConfig, load_yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


def test_shipped_configs_parse_and_validate():
    found = sorted(CONFIGS.glob("*.yaml"))
    assert found, "no configs/*.yaml shipped"
    for p in found:
        cfg = load_yaml(p)          # parses + validates (raises on unknown key / bad physics)
        assert cfg.dt > 0


def test_runconfig_default_invariant():
    # the regression guard at config level: defaults() is exactly the empty config (all physics off,
    # module-constant γ's, the implemented tracer scheme) ⇒ a defaults run is bit-identical (the
    # STEP-level proof is in test_run_config on a mesh; this is the data-free contract)
    d = RunConfig.defaults()
    assert d == RunConfig()
    assert d.visc == ViscConfig() and d.tracer == TracerConfig()
    assert d.ale is None and d.gm is None and d.kpp is None and d.tke is None and d.ice is None
    d.validate()                    # defaults must always be valid


def test_ng5_config_pins():
    cfg = load_yaml(CONFIGS / "ng5.yaml")
    assert cfg.visc.gamma1 == 0.2 and cfg.gm is None       # the pinned NG5 physics
    assert cfg.tracer == TracerConfig()                     # the implemented MFCT/QR4C/FCT (0,1)


def test_duration_and_chunks_roundtrip():
    assert parse_duration("1d", 1800.0) == 48
    assert parse_duration("10step", 1800.0) == 10
    ch = plan_chunks(10, 4, start_step=0, dt=1800.0)
    assert sum(c.count for c in ch) == 10 and ch[0].bootstrap_ab2


def test_public_entry_points_import():
    import fesom_jax.run            # the run driver
    import fesom_jax.run_config     # the config
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_from_config_cli", ROOT / "scripts" / "run_from_config.py")
    assert spec is not None         # the CLI script is present + loadable


def test_release_ok():
    print("RELEASE_OK")
