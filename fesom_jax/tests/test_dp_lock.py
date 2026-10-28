"""DP lock of the single-precision work (docs/plans/20261020-core2-single-precision.md, Task 2).

The double-precision programs — the production forced chunk with every speed option off and with the
``core2_gpu.yaml`` options, and the CORE2 adjoint's reverse program — must lower to exactly the StableHLO they
lowered to at the branch base (``fesom_jax/tests/data/dp_hlo_hashes.json``). The precision switch keeps every
helper the identity in DP, so any difference here is a DP change. The baseline is regenerated only in a commit that
says it changes DP on purpose (e.g. a merge from perf/core2-gpu).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).with_name("data") / "dp_hlo_hashes.json"

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "mesh_core2").exists(), reason="CORE2 mesh not staged")


def test_dp_programs_unchanged(tmp_path):
    out = tmp_path / "now.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "perf" / "dp_hlo_hash.py"), "--write", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    base = json.loads(BASE.read_text())["programs"]
    now = json.loads(out.read_text())["programs"]
    assert set(now) == set(base)
    changed = {k: (base[k], now[k]) for k in base if base[k]["sha256"] != now[k]["sha256"]}
    assert not changed, f"DP programs changed: {changed}"
    for k, v in now.items():          # a DP program holds no float32 (fesom3-jax measured 0 for its chunk)
        assert v["f32_types"] == base[k]["f32_types"], k
