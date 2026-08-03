"""The documented Quick-start command must actually run — DATA-FREE (usability review P1/P2).

``configs/pi_demo.yaml`` is the only configuration advertised as needing no external data, and
the README calls the config-driven run "the release path"; before this test nothing exercised
it (review P1: ``mesh: null`` defeated, the CORE2 IC hardcoded, the forcing build
unconditional). This runs the real script the way the docs tell a new user to, in an empty
cwd with a scrubbed environment — the closest thing to a fresh machine we can get in-repo.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("FESOM_")}
    # THIS checkout must win over any pip-installed (editable) fesom_jax — the test
    # exercises the repo it lives in, not whatever the env happens to have installed.
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_pi_demo_config_runs(tmp_path):
    env = _clean_env()
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_from_config.py"),
         str(REPO / "configs" / "pi_demo.yaml"), "--steps", "3"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, (
        f"the documented Quick-start run failed\n--- stdout:\n{out.stdout}"
        f"\n--- stderr:\n{out.stderr}")
    assert "DONE step=3" in out.stdout, f"no DONE line in:\n{out.stdout}"


def test_forcing_null_never_uses_default_paths(tmp_path, monkeypatch):
    """`forcing: null` + a booby-trapped $FESOM_JRA_DIR: the run must NOT read it (review P2 —
    the old driver reinterpreted null as 'no path override' and fell through to the defaults)."""
    env = _clean_env()
    env["FESOM_JRA_DIR"] = str(tmp_path / "must_never_be_opened")
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_from_config.py"),
         str(REPO / "configs" / "pi_demo.yaml"), "--steps", "1"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, (
        f"unforced run failed (did it try to build forcing?)\n--- stdout:\n{out.stdout}"
        f"\n--- stderr:\n{out.stderr}")
    assert "must_never_be_opened" not in out.stdout + out.stderr
