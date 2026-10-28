"""The halo transports through ``run_from_config`` with the ``core2_gpu.yaml`` options (pi, 4 fake CPU devices, ice every
second step, rcm lanes, the mEVP wide halo): ``--padded`` and ``--coloured`` (``gpu.halo_compact`` off; they refuse
it) end in the restart of the compact all_gather run, up to the order of the sums."""

from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.mark.parametrize("transport", ["--padded", "--coloured"])
def test_transport_matches_compact(tmp_path, transport):
    from fesom_jax.tests.test_precision_shard import _cfg, _load, _run
    cfg = _cfg(tmp_path)
    common = ("--ice-ave-steps", "2", "--steps", "6", "--chunk-steps", "3")
    _run(cfg, *common, "--restart-out", str(tmp_path / "compact"))
    os.environ["FESOM_HALO_COMPACT"] = "0"
    try:
        _run(cfg, *common, transport, "--restart-out", str(tmp_path / "other"))
    finally:
        os.environ.pop("FESOM_HALO_COMPACT", None)
    a, b = _load(tmp_path / "compact"), _load(tmp_path / "other")
    assert a.keys() == b.keys()
    rel = {}
    for k in a:
        if not np.issubdtype(a[k].dtype, np.floating):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
            continue
        rel[k] = float(np.max(np.abs(a[k] - b[k]))) / max(float(np.max(np.abs(a[k]))), 1e-30)
    print(f"\n[{transport}] fields that differ (max|d|/max|a|):",
          {k: f"{v:.1e}" for k, v in sorted(rel.items(), key=lambda kv: -kv[1]) if v > 0})
    bad = {k: v for k, v in rel.items() if v > (1e-8 if k in ("Kv", "Av", "tke") else 1e-10)}
    assert not bad, bad
