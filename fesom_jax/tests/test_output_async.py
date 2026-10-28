"""``gpu.output_async`` (``FESOM_OUTPUT_ASYNC``): the daily and monthly means written on a background thread are the
same stores as the ones written in the run loop (pi, 4 fake CPU devices, the ``core2_gpu.yaml`` options, two days
in half-day chunks: two daily stores written while the run goes on, the monthly one at the end); the same for the
archival restarts of ``gpu.restart_async``."""

from __future__ import annotations

import os

import numpy as np
import zarr


def _stores(root):
    out = {}
    for d in sorted(p for p in root.rglob("*") if (p / ".zgroup").exists() or (p / "zarr.json").exists()):
        g = zarr.open_group(str(d), mode="r")
        out[str(d.relative_to(root))] = {k: np.asarray(g[k][:]) for k in g.array_keys()}
    return out


def test_output_async_same_stores(tmp_path):
    from fesom_jax.tests.test_precision_shard import _cfg, _run
    cfg = _cfg(tmp_path)
    got = {}
    for flag in ("0", "1"):
        os.environ["FESOM_OUTPUT_ASYNC"] = flag
        try:
            _run(cfg, "--steps", "96", "--chunk-steps", "24", "--daily-out", str(tmp_path / f"d{flag}"),
                 "--monthly-out", str(tmp_path / f"m{flag}"))
        finally:
            os.environ.pop("FESOM_OUTPUT_ASYNC", None)
        got[flag] = {**{f"d/{k}": v for k, v in _stores(tmp_path / f"d{flag}").items()},
                     **{f"m/{k}": v for k, v in _stores(tmp_path / f"m{flag}").items()}}
    assert len([k for k in got["0"] if k.startswith("d/")]) == 2 and any(k.startswith("m/") for k in got["0"])
    assert got["0"].keys() == got["1"].keys()
    for store, arrays in got["0"].items():
        assert arrays.keys() == got["1"][store].keys(), store
        for k, v in arrays.items():
            np.testing.assert_array_equal(v, got["1"][store][k], err_msg=f"{store}:{k}")


def test_restart_async_same_archive(tmp_path):
    """``gpu.restart_async`` (``FESOM_RESTART_ASYNC``): the calendar archival restart written on the background thread
    while the next chunks run is the store the run loop writes, and ``restart.latest`` names the end-of-run one
    (single precision, so the float64 widening runs on the thread's host copy; two days in half-day chunks)."""
    from fesom_jax.tests.test_precision_shard import _cfg, _run
    cfg = _cfg(tmp_path)
    got, latest = {}, {}
    for flag in ("0", "1"):
        os.environ["FESOM_RESTART_ASYNC"] = flag
        try:
            out = _run(cfg, "--precision", "single", "--steps", "96", "--chunk-steps", "24", "--progress",
                       "--restart-archive-out", str(tmp_path / f"a{flag}"), "--restart-archive-period", "day")
        finally:
            os.environ.pop("FESOM_RESTART_ASYNC", None)
        assert ("background)" in out) == (flag == "1")
        got[flag] = _stores(tmp_path / f"a{flag}")
        latest[flag] = (tmp_path / f"a{flag}" / "restart.latest").read_text()
    assert len(got["0"]) == 2 and got["0"].keys() == got["1"].keys()
    assert latest["0"] == latest["1"] == max(got["0"])
    for store, arrays in got["0"].items():
        assert arrays.keys() == got["1"][store].keys(), store
        for k, v in arrays.items():
            np.testing.assert_array_equal(v, got["1"][store][k], err_msg=f"{store}:{k}")


def test_forcing_two_chunks_ahead_bitwise(tmp_path):
    """``gpu.forcing_ahead`` (``FESOM_FORCING_AHEAD``) 2 against 1: the same forcing builds in the same order on the
    one prefetch worker, so the restart is bit for bit the same (on-device forcing tables, 4-step chunks)."""
    from fesom_jax.tests.test_precision_shard import _cfg, _load, _run
    cfg = _cfg(tmp_path)
    out = {}
    for n in ("1", "2"):
        os.environ["FESOM_FORCING_AHEAD"] = n
        try:
            _run(cfg, "--steps", "16", "--chunk-steps", "4", "--restart-out", str(tmp_path / f"r{n}"))
        finally:
            os.environ.pop("FESOM_FORCING_AHEAD", None)
        out[n] = _load(tmp_path / f"r{n}")
    for k in out["1"]:
        np.testing.assert_array_equal(out["1"][k], out["2"][k], err_msg=k)


def test_step_unroll_round_off(tmp_path):
    """``gpu.step_unroll`` (``FESOM_STEP_UNROLL``) 2 against 1: 7 steps in one chunk (one eager step, then 6 scanned
    as 3 trips of two) end in the same restart up to round-off; integer fields bitwise."""
    from fesom_jax.tests.test_precision_shard import _cfg, _load, _run
    cfg = _cfg(tmp_path)
    out = {}
    for n in ("1", "2"):
        os.environ["FESOM_STEP_UNROLL"] = n
        try:
            _run(cfg, "--steps", "7", "--chunk-steps", "7", "--restart-out", str(tmp_path / f"r{n}"))
        finally:
            os.environ.pop("FESOM_STEP_UNROLL", None)
        out[n] = _load(tmp_path / f"r{n}")
    assert out["1"].keys() == out["2"].keys()
    for k, a in out["1"].items():
        b = out["2"][k]
        if not np.issubdtype(a.dtype, np.floating):
            np.testing.assert_array_equal(a, b, err_msg=k)
            continue
        scale = max(float(np.max(np.abs(a))), 1e-30)
        assert np.max(np.abs(a - b)) <= 1e-10 * scale, (k, float(np.max(np.abs(a - b))), scale)
