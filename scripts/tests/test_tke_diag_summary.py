"""Tables and figures of the TKE-adjoint diagnosis (``scripts/adjoint/tke_diag_summary.py``,
``scripts/adjoint/plot_tke_diag.py``) on synthetic result rows: no model, no device."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import tke_diag_summary as S  # noqa: E402


def _screen_row(trace, setting="none", repeat=0, start="2013-01-01T00:00:00", k=6):
    ft = [{"boundary": i, "tke": x, "T": x / 2, "uv": x / 3, "eta_n": 0.0, "S": 0.0}
          for i, x in enumerate(trace)]
    return {"action": "tke_screen", "start": start, "n_steps": k * (len(trace) - 1),
            "chunk_steps": k, "setting": setting, "repeat": repeat, "trace": list(trace),
            "field_trace": ft, "status": "ok", "slurm_job": "1"}


def test_trace_stats_overall_and_worst_stretches():
    # 16 chunks of 6 steps: flat for 8 chunks, then 4 chunks at 2x per step, then flat
    rates = [1.0] * 8 + [2.0] * 4 + [1.0] * 4
    tr = [1.0]
    for r in rates:
        tr.append(tr[-1] * r ** 6)
    s = S.trace_stats(tr, 6)
    assert s["n_steps"] == 96
    assert math.isclose(s["overall"], 2.0 ** (24 / 96), rel_tol=1e-12)
    assert math.isclose(s["worst4"], 2.0, rel_tol=1e-12)          # the burst itself
    assert math.isclose(s["median"], 1.0, rel_tol=1e-12)          # the median hides it
    assert math.isclose(s["max_chunk"], 2.0, rel_tol=1e-12)
    # worst day (8 chunks of 3 h): the day holding the 4 burst chunks
    assert math.isclose(s["worst_day"], 2.0 ** 0.5, rel_tol=1e-12)
    assert len(s["per_day"]) == 2


def test_trace_stats_skips_a_zero_seed_field():
    s = S.trace_stats([0.0, 0.0, 1.0, 4.0], 1)
    assert s["first_nonzero"] == 2 and math.isclose(s["overall"], 4.0)


def test_divergence_lag():
    a = [1.0, 1.0, 2.0, 4.0]
    b = [1.0, 1.0, 2.0, 8.0]
    assert S.divergence_lag([a, b], 6) == 3 * 6 * S.DT / 86400.0
    assert S.divergence_lag([a, a], 6) == float("inf")


def test_screen_table_groups_repeats_and_reports_the_spread():
    rows = [_screen_row([1, 2, 4, 8], repeat=0), _screen_row([1, 2, 4, 16], repeat=1),
            _screen_row([1, 1, 1, 1], setting="freeze_mixing")]
    text, recs = S.screen_table(rows)
    assert len(recs) == 2 and {r["setting"] for r in recs} == {"exact", "freeze_mixing"}
    ex = [r for r in recs if r["setting"] == "exact"][0]
    assert ex["n"] == 2 and "[" in text             # median [min, max] over the repeats
    assert "| 2013-01-01 | 0.375 | exact | 2 |" in text


def test_cuts_and_site_tables_and_json(tmp_path):
    base = {"start": "2013-01-01T00:00:00", "weights": "tke", "box": [0, 1, 2, 3, 0, 1000],
            "mld_box": {"median": 20.0, "max": 40.0}, "history": [{"sigma": 1.0}],
            "top_v": [{"field": "tke", "lon": 1.0, "lat": 2.0, "level": 1, "regime": "free",
                       "tke": 1e-6, "F": 0.5, "Ri": 7.0}],
            "census_v": {"norm2": 1.0, "regime": {"free": 1.0}, "level": {"1": 1.0}}}
    rows = [dict(base, action="tke_site", site="site1", n_steps=1, sigma=10.0),
            dict(base, action="tke_cuts", site="site1", n_steps=1, sigma=10.0,
                 setting="none", repeat=1, ratio=1.0),
            dict(base, action="tke_cuts", site="site1", n_steps=1, sigma=0.5,
                 setting="tke.km_e", repeat=0, ratio=0.05)]
    t = S.cuts_table(rows)
    assert "tke.km_e" in t and "0.05" in t and "none (repeat 1)" in t
    st = S.site_table(rows)
    assert "site1" in st and "free 1.00" in st
    d = tmp_path / "run"
    d.mkdir()
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "s.md"
    assert S.main([str(d), "--md", str(out), "--json", str(tmp_path / "s.json")]) == 0
    assert "## Cuts at the top site" in out.read_text()


def test_plots_render(tmp_path):
    pytest.importorskip("matplotlib")
    import plot_tke_diag as P
    rows = [_screen_row([1, 2, 4, 8]), _screen_row([1, 1, 1, 1], setting="freeze_mixing")]
    rows += [{"action": "tke_fdlin", "slurm_job": "2", "label": "fd", "start": "2013-01-01",
              "n_steps": 1, "weights": "tke", "h": h, "rel_err": e, "noise_rel": 1e-9,
              "noise": "roughness", "predicted_h": 1e-6, "platform": "cpu",
              "ssh_forward_tol": 1e-12} for h, e in [(1e-2, 0.5), (1e-4, 1e-3), (1e-6, 1e-5)]]
    made = P.plot_screens(rows, tmp_path) + P.plot_fdlin(rows, tmp_path)
    assert len(made) == 2 and all(p.exists() and p.stat().st_size > 0 for p in made)
    assert np.isfinite(S.trace_stats(rows[0]["trace"], 6)["overall"])


def test_nonfinite_strings_read_back_as_floats(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "results.jsonl").write_text(json.dumps({"action": "x", "h": "inf", "v": ["nan", 1.0],
                                                 "name": "tke"}) + "\n")
    r = S.load_rows([d])[0]
    assert r["h"] == float("inf") and math.isnan(r["v"][0]) and r["name"] == "tke"
