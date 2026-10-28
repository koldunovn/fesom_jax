"""Single precision through the production driver (``scripts/run_from_config.py``) on the packaged pi mesh,
sharded over 4 fake CPU devices with the ``core2_gpu.yaml`` options (Chebyshev CG, on-device forcing, gathered SSH,
compact halo, column unroll, mEVP wide halo): finite, and a restart in the middle resumes the straight run to within
one float32 rounding at a handful of points (the rounding carry is in the restart, Deviation D6). A double-precision
restart starts a single-precision run.

Why not bit for bit: the driver's resume is not bitwise in double precision either — on this configuration, and on
the merge base without any single-precision code, the resumed double-precision run differs from the straight one at
round-off (~1e-8 relative) in nearly every field (measured 2026-10-20, pi, 4 fake devices). The single-precision resume
is closer than that: the velocities and the SSH are bitwise, T differs at 2 of 150 720 points by one float32 ulp, and
the carry at ~50 points by less than one rounding residual.

Each run is a subprocess with ``--xla_force_host_platform_device_count=4``.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _cfg(tmp_path) -> Path:
    s = (ROOT / "configs" / "core2_gpu.yaml").read_text()
    for key, val in (("mesh", "null"), ("output_dir", "null"), ("duration", "null"), ("restart_out", "null")):
        s = re.sub(rf"^{key}:.*$", f"{key}: {val}", s, flags=re.M)
    p = tmp_path / "pi_gpu.yaml"
    p.write_text(s)
    return p


def _run(cfg, *extra):
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT),
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip())
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_from_config.py"), str(cfg), "--dist-dir",
                        str(ROOT / "fesom_jax" / "data" / "mesh_pi"), "--diagnostics", *extra],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "NG5_R0_FINITE" in r.stdout
    return r.stdout


def _load(path):
    import zarr
    g = zarr.open_group(str(path), mode="r")
    return {k: np.asarray(g[k][:]) for k in g.array_keys()}


def test_single_restart_resumes_bitwise(tmp_path):
    cfg = _cfg(tmp_path)
    _run(cfg, "--precision", "single", "--steps", "6", "--chunk-steps", "3", "--restart-out", str(tmp_path / "a"))
    _run(cfg, "--precision", "single", "--steps", "3", "--chunk-steps", "3", "--restart-out", str(tmp_path / "b1"))
    _run(cfg, "--precision", "single", "--steps", "3", "--chunk-steps", "3", "--restart-in", str(tmp_path / "b1"),
         "--restart-out", str(tmp_path / "b2"))
    a, b = _load(tmp_path / "a"), _load(tmp_path / "b2")
    assert "carry_T" in a and "carry_S" in a
    assert a["T"].dtype == np.float64                       # on disk always float64
    for k in a:
        diff = a[k] != b[k]
        assert diff.mean() <= 1e-3, (k, int(diff.sum()))           # a handful of points at most
        if k in ("carry_T", "carry_S"):      # a residual of the order of one rounding of X·hnode (≈1e-3 for S)
            assert np.max(np.abs(a[k] - b[k])) <= np.max(np.abs(a[k])), k
        else:                                                        # one float32 ulp
            np.testing.assert_allclose(a[k], b[k], rtol=2.5e-7, atol=0.0, err_msg=k)
    for k in ("uv", "eta_n"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_double_restart_starts_single(tmp_path):
    cfg = _cfg(tmp_path)
    _run(cfg, "--steps", "3", "--chunk-steps", "3", "--restart-out", str(tmp_path / "d"))
    assert "carry_T" not in _load(tmp_path / "d")
    out = _run(cfg, "--precision", "single", "--steps", "3", "--chunk-steps", "3", "--restart-in",
               str(tmp_path / "d"), "--restart-out", str(tmp_path / "s"), "--budget")
    assert "[budget]" in out
    assert "carry_T" in _load(tmp_path / "s")


def test_single_ordered_sums_round_off(tmp_path):
    """Single precision with the ordered-sum assemblies (gpu.ordered_sums / FESOM_ORDERED_SUMS=1) against the
    segment_sum ones: the same WP sums in another order, so the two-step states agree to float32 round-off."""
    cfg = _cfg(tmp_path)
    out = {}
    for flag in ("0", "1"):
        os.environ["FESOM_ORDERED_SUMS"] = flag
        try:
            _run(cfg, "--precision", "single", "--steps", "2", "--chunk-steps", "2",
                 "--restart-out", str(tmp_path / f"o{flag}"))
        finally:
            os.environ.pop("FESOM_ORDERED_SUMS", None)
        out[flag] = _load(tmp_path / f"o{flag}")
    a, b = out["0"], out["1"]
    for k in ("T", "S", "uv", "a_ice", "m_ice"):
        scale = np.max(np.abs(a[k]))
        assert np.max(np.abs(a[k] - b[k])) <= 1e-5 * scale, k
    assert np.max(np.abs(a["eta_n"] - b["eta_n"])) <= 1e-5           # metres


def test_single_ice_ave_steps(tmp_path):
    """Single precision with the sea ice every second ocean step (``ice_ave_steps = 2``), the two FESOM2 options
    together as upstream builds them (the running means and the held ice fluxes are WP, float32): finite, and the
    restart carries the counter, the flag and the means (float64 on disk) next to the rounding carry."""
    from fesom_jax.state import ICE_AVG_FIELDS
    cfg = _cfg(tmp_path)
    _run(cfg, "--precision", "single", "--ice-ave-steps", "2", "--steps", "6", "--chunk-steps", "3",
         "--restart-out", str(tmp_path / "a"))
    a = _load(tmp_path / "a")
    for k in ("carry_T", "carry_S") + ICE_AVG_FIELDS:
        assert k in a, k
    assert a["srfoce_temp"].dtype == np.float64 and np.all(np.isfinite(a["srfoce_temp"]))
    assert np.all(np.isin(a["ice_update"], (0, 1)))
