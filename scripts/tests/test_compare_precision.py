"""scripts/perf/compare_precision.py on synthetic states and budget logs."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("compare_precision", ROOT / "scripts" / "perf" / "compare_precision.py")
CP = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = CP
spec.loader.exec_module(CP)


def _state(n=50, nl=4, S=35.0, T=4.0, eta=0.0):
    return {"hnode": np.full((n, nl), 10.0), "T": np.full((n, nl), T), "S": np.full((n, nl), S),
            "eta_n": np.full(n, eta), "a_ice": np.zeros(n), "m_ice": np.zeros(n)}


def test_state_stats_and_carry():
    av = np.ones((50, 4))
    lat = np.linspace(-1, 1, 50)
    s = _state()
    st = CP.state_stats(s, av, lat)
    assert st["S_vol_mean"] == 35.0 and st["T_vol_mean"] == 4.0 and st["volume"] == 2000.0
    s2 = dict(s, carry_S=np.full((50, 4), 1e-3))           # carry content 0.2 psu m3 over 2000 m3
    assert abs(CP.state_stats(s2, av, lat)["S_vol_mean"] - (35.0 + 1e-4)) < 1e-12
    pw = CP.pointwise(s, _state(T=4.5), av)
    assert abs(pw["SST"]["rms"] - 0.5) < 1e-12 and pw["SSS"]["max"] == 0.0


def test_budget_parse_and_verdict(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("[run] x\n[budget] step 96 V=1.0e+18 S_oce=3.5e+19 T_oce=4.0e+18 S_ice=1.0e+14 ice_vol=1e13 "
                   "eta_mean=0.0e+00\n[budget] step 192 V=1.0000000001e+18 S_oce=3.5e+19 T_oce=4.0e+18 "
                   "S_ice=1.0e+14 ice_vol=1e13 eta_mean=1e-9\n")
    rows = CP.parse_budget(log)
    assert [r["step"] for r in rows] == [96, 192]
    b = CP.budget_drift(rows)
    assert b["steps"] == 96 and abs(b["dV"] - 1e8) < 1e3 and b["deta_mean"] == 1e-9
    res = {"pairs": {"sp-dp": {"delta": {"S_vol_mean": 1e-8, "T_vol_mean": 1e-7, "ssh_global_mean": 0.0},
                               "pointwise": {}}},
           "budget": {"sp": {"dS_total": 0.0, "dV": 0.0}, "dp": {"dS_total": 0.0, "dV": 0.0}}}
    v = CP.verdict(res, days=30)
    assert v["S_vol_mean"]["pass"] and v["T_vol_mean"]["pass"] and v["budget_salt"]["pass"]
    res["pairs"]["sp-dp"]["delta"]["S_vol_mean"] = 1e-6
    assert not CP.verdict(res, days=30)["S_vol_mean"]["pass"]
