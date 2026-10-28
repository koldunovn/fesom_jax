"""``gpu.ice_static`` (``FESOM_ICE_STATIC``): with sea ice every N steps the forced scan fixes each step's ice branch at
trace time instead of a ``lax.cond`` on the device counter. Through ``run_from_config`` on pi (4 fake CPU devices, the
``core2_gpu.yaml`` options, a cold start) the restart and the daily means equal the ``lax.cond`` run's up to the order
of operations, for N = 2 and 3 and chunks whose lengths are not multiples of N (the bootstrap step, whole cycles, the
inline remainder, and a phase that changes from chunk to chunk)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import zarr


def _daily(root):
    out = {}
    for d in sorted(p for p in root.rglob("*") if (p / ".zgroup").exists() or (p / "zarr.json").exists()):
        g = zarr.open_group(str(d), mode="r")
        out.update({f"{d.name}:{k}": np.asarray(g[k][:]) for k in g.array_keys()})
    return out


@pytest.mark.parametrize("nave,steps,chunk", [(2, 9, 4), (3, 10, 4)])
def test_ice_static_matches_cond(tmp_path, nave, steps, chunk):
    from fesom_jax.tests.test_precision_shard import _cfg, _load, _run
    cfg = _cfg(tmp_path)
    got = {}
    for flag in ("0", "1"):
        os.environ["FESOM_ICE_STATIC"] = flag
        try:
            out = _run(cfg, "--ice-ave-steps", str(nave), "--steps", str(steps), "--chunk-steps", str(chunk),
                       "--restart-out", str(tmp_path / f"r{flag}"), "--daily-out", str(tmp_path / f"d{flag}"))
        finally:
            os.environ.pop("FESOM_ICE_STATIC", None)
        assert ("ice_static:" in out) == (flag == "1")
        got[flag] = {**_load(tmp_path / f"r{flag}"), **_daily(tmp_path / f"d{flag}")}
    assert got["0"].keys() == got["1"].keys()
    assert "ice_steps_since_upd" in got["0"]
    rel = {}
    for k, a in got["0"].items():
        b = got["1"][k]
        if not np.issubdtype(a.dtype, np.floating):
            np.testing.assert_array_equal(a, b, err_msg=k)
            continue
        rel[k] = float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(a))), 1e-30)
    print(f"\n[ice_static N={nave}] fields that differ (max|d|/max|a|):",
          {k: f"{v:.1e}" for k, v in sorted(rel.items(), key=lambda kv: -kv[1]) if v > 0})
    # round-off of the inline steps' other fusion, grown over a few steps; the mixing coefficients (TKE) amplify it
    bad = {k: v for k, v in rel.items() if v > (1e-8 if k in ("Kv", "Av", "tke") else 1e-10)}
    assert not bad, bad
