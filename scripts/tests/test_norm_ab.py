"""The control-transform A/B driver ``scripts/calib/norm_ab.py`` (model-free parts).

Checked: the three transforms (log = the calibration's own u, linear p0·(1 + σv), raw p = v)
round-trip p ↔ v; the reparameterised gradient equals a central finite difference of the
reparameterised cost; the log arm is the base function itself (the same objects, so iterates are
bitwise those of the plain driver); linear at v = 0 evaluates the base at exactly u = 0 and returns
its gradient unchanged; a point with p ≤ 0 is rejected without running the model; the
model-evaluation budget counts model runs only; every added aux value is plain JSON.

Cheap (no mesh, no model): numpy and the optimiser only. Lives in the ``scripts`` group.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "calib"))
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
import norm_ab as N  # noqa: E402

P0 = np.array([0.1, 0.7, 3.75, 30.0])
SIG = np.full(4, math.log(2.0))
KINDS = ("log", "linear", "raw")

# A cost in physical units with the raw-unit scales of the TKE problem (diagonal of the measured
# 12-hour Gauss-Newton Hessian at the twin truth, ~3e4 … 1e-2) plus coupling and a cubic term, so
# the transforms see a non-quadratic, badly scaled function.
P_STAR = np.array([0.14, 0.5, 3.0, 25.0])
_D = np.array([3.2e4, 1.6e2, 2.0, 8.5e-3])
_C = np.array([[1.0, 0.3, -0.2, 0.1],
               [0.3, 1.0, 0.5, -0.1],
               [-0.2, 0.5, 1.0, 0.2],
               [0.1, -0.1, 0.2, 1.0]])
H_P = np.sqrt(_D)[:, None] * _C * np.sqrt(_D)[None, :]


def _J_p(p):
    d = np.asarray(p, np.float64) - P_STAR
    f = 0.5 * d @ H_P @ d + 0.1 * float(np.sum(_D * d ** 3))
    g = H_P @ d + 0.3 * _D * d ** 2
    return float(f), g


class _Base:
    """The calibration's ``fun_and_grad`` contract in log-u: ``(J, ∇_u J, aux)``, aux plain JSON."""

    def __init__(self):
        self.calls = 0

    def __call__(self, u):
        self.calls += 1
        u = np.asarray(u, np.float64)
        p = P0 * np.exp(SIG * u)
        f, gp = _J_p(p)
        return f, SIG * p * gp, {"J_data": f, "p": p.tolist()}


def _tr(kind):
    return N.Transform(kind, tuple(P0), tuple(SIG))


@pytest.mark.parametrize("kind", KINDS)
def test_round_trip(kind):
    tr = _tr(kind)
    p = np.array([0.03, 1.9, 7.5, 12.0])
    v = tr.to_v(p)
    assert v.dtype == np.float64
    np.testing.assert_allclose(tr.to_p(v), p, rtol=1e-15)
    assert np.all(tr.to_p(tr.to_v(P0)) == P0)


@pytest.mark.parametrize("kind", KINDS)
def test_to_v_refuses_nonpositive_p(kind):
    with pytest.raises(ValueError):
        _tr(kind).to_v(np.array([0.1, 0.0, 3.0, 25.0]))


def test_transforms_are_the_documented_maps():
    v = np.array([0.4, -0.3, 1.1, -0.9])
    np.testing.assert_allclose(_tr("log").to_p(v), P0 * np.exp(SIG * v), rtol=1e-15)
    np.testing.assert_allclose(_tr("linear").to_p(v), P0 * (1.0 + SIG * v), rtol=1e-15)
    np.testing.assert_array_equal(_tr("raw").to_p(v), v)
    np.testing.assert_allclose(_tr("log").dp_dv(v), SIG * P0 * np.exp(SIG * v), rtol=1e-15)
    np.testing.assert_allclose(_tr("linear").dp_dv(v), SIG * P0, rtol=1e-15)
    np.testing.assert_array_equal(_tr("raw").dp_dv(v), np.ones(4))


def test_transform_from_control_set_matches_the_calibration_controls():
    tr = N.transform_for(N.tke_control_set(), "linear")
    np.testing.assert_array_equal(np.array(tr.p0), P0)
    np.testing.assert_array_equal(np.array(tr.sigma), SIG)


@pytest.mark.parametrize("kind", KINDS)
def test_chain_rule_matches_central_fd(kind):
    tr = _tr(kind)
    fg = N.reparam(_Base(), tr)
    for p in (P0, np.array([0.12, 0.55, 3.3, 26.0]), np.array([0.2, 0.4, 5.0, 20.0])):
        v = tr.to_v(p)
        f, g, _ = fg(v)
        for i in range(4):
            h = 1e-6 * max(1.0, abs(v[i]))
            e = np.zeros(4)
            e[i] = h
            fd = (fg(v + e)[0] - fg(v - e)[0]) / (2.0 * h)
            assert abs(fd - g[i]) <= 1e-7 * max(abs(g[i]), 1e-3 * np.max(np.abs(g))), \
                (kind, i, fd, g[i])


def test_log_is_the_base_function_itself():
    base = _Base()
    fg = N.reparam(base, _tr("log"))
    for u in (np.zeros(4), np.array([0.5, -0.5, 0.5, -0.5]), np.array([1.585, -1.585, 1.0, -1.0])):
        mine = fg(u)
        ref = _Base()(u)
        assert mine[0] == ref[0]
        assert mine[1].tobytes() == ref[1].tobytes()
        assert mine[2] == ref[2]
    out = []
    fg_spy = N.reparam(lambda u: out.append(u) or ("F", "G", "A"), _tr("log"))
    inp = np.ones(4)
    assert fg_spy(inp) == ("F", "G", "A")
    assert out[0] is inp                  # no copy, no arithmetic on the way in


def test_linear_at_zero_is_log_at_zero_bitwise():
    seen = []

    def base(u):
        seen.append(np.asarray(u, np.float64).copy())
        return _Base()(u)

    fl, gl, _ = N.reparam(base, _tr("linear"))(np.zeros(4))
    fo, go, _ = N.reparam(base, _tr("log"))(np.zeros(4))
    assert seen[0].tobytes() == np.zeros(4).tobytes()
    assert fl == fo
    assert gl.tobytes() == go.tobytes()


@pytest.mark.parametrize("kind,v", [("linear", [0.0, -1.5, 0.0, 0.0]),
                                    ("linear", [0.0, 0.0, -2.0, 0.0]),
                                    ("raw", [0.1, 0.7, 0.0, 30.0]),
                                    ("raw", [-0.1, 0.7, 3.75, 30.0])])
def test_out_of_domain_is_rejected_without_running_the_model(kind, v):
    base = _Base()
    f, g, aux = N.reparam(base, _tr(kind))(np.array(v, np.float64))
    assert base.calls == 0
    assert math.isnan(f) and g.shape == (4,) and np.all(np.isnan(g))
    assert aux["out_of_domain"] is True and aux["transform"] == kind
    assert len(aux["p"]) == 4 and len(aux["v"]) == 4
    json.dumps(aux)                       # minimize writes aux with plain json.dumps


@pytest.mark.parametrize("kind", ("linear", "raw"))
def test_added_aux_is_plain_json(kind):
    tr = _tr(kind)
    _, _, aux = N.reparam(_Base(), tr)(tr.to_v(P0 * 1.1))
    s = json.dumps(aux)
    back = json.loads(s)
    assert back["transform"] == kind and back["out_of_domain"] is False
    assert back["J_data"] == aux["J_data"]


def test_model_budget_counts_model_runs_only():
    base = _Base()
    tr = _tr("linear")
    fg = N.reparam(base, tr, max_model_evals=2)
    bad = np.array([0.0, -5.0, 0.0, 0.0])
    fg(np.zeros(4))
    fg(bad)
    fg(bad)
    fg(np.full(4, 0.1))
    assert base.calls == 2
    fg(bad)                               # a rejection is still free at the budget
    with pytest.raises(N.ModelBudget):
        fg(np.full(4, 0.2))
    assert base.calls == 2


def test_model_budget_applies_to_the_log_arm_too():
    base = _Base()
    fg = N.reparam(base, _tr("log"), max_model_evals=1)
    fg(np.zeros(4))
    with pytest.raises(N.ModelBudget):
        fg(np.zeros(4))


# ---------------------------------------------------------------------------
# S0: conditioning of the Gauss-Newton Hessian under each transform

IDENT = Path("/work/ab0995/a270088/fesom_jax_dev/runs/calib")


def _spd(n, seed):
    a = np.random.default_rng(seed).standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def test_hessian_transforms_closed_form():
    H_u = _spd(4, 1)
    p = np.array([0.14, 0.5, 3.0, 25.0])
    hs = N.transform_hessians(H_u, p, P0, SIG)
    H_p = H_u / np.outer(SIG * p, SIG * p)
    np.testing.assert_allclose(hs["raw"], H_p, rtol=1e-14)
    np.testing.assert_allclose(hs["linear"], H_p * np.outer(SIG * P0, SIG * P0), rtol=1e-14)
    np.testing.assert_array_equal(hs["log"], H_u)
    # at the first guess the linear and log Hessians are the same matrix (dp/dv = σp0 in both)
    np.testing.assert_allclose(N.transform_hessians(H_u, P0, P0, SIG)["linear"], H_u, rtol=1e-14)


def test_kappa_with_and_without_the_prior():
    H = np.diag([4.0, 1.0, 0.5, 0.01])
    assert N.kappa(H) == pytest.approx(400.0, rel=1e-14)
    assert N.kappa(H, prior=1.0) == pytest.approx(5.0 / 1.01, rel=1e-14)


def test_cond_rows_report_no_unit_prior_for_raw():
    rows = N.cond_rows_from_records([{"point": "p0", "n_steps": 12, "p": P0.tolist(),
                                      "H_data": _spd(4, 2).tolist(),
                                      "calib": {"p0": P0.tolist(), "sigma": SIG.tolist()}}])
    r = rows[0]
    assert set(r["kappa"]) == {"raw", "linear", "log"}
    assert set(r["kappa_unit_prior"]) == {"linear", "log"}
    assert r["kappa"]["linear"] == pytest.approx(r["kappa"]["log"], rel=1e-12)


@pytest.mark.skipif(not (IDENT / "twin_w12" / "ident.jsonl").is_file()
                    or not (IDENT / "twin_w24" / "ident.jsonl").is_file(),
                    reason="track F's ident.jsonl files are absent")
def test_cond_regression_on_track_f_hessians():
    want = {(12, "p0"): (9.663e6, 236.8, 236.8), (12, "truth"): (8.233e6, 1009, 1079),
            (24, "p0"): (3.323e6, 933.0, 933.0), (24, "truth"): (1.508e7, 236.4, 686.2)}
    got = {}
    for f in ("twin_w12", "twin_w24"):
        for r in N.cond_rows(IDENT / f / "ident.jsonl"):
            got[(r["n_steps"], r["point"])] = r["kappa"]
    assert set(got) == set(want)
    for k, (raw, lin, log) in want.items():
        assert got[k]["raw"] == pytest.approx(raw, rel=1e-3), k
        assert got[k]["linear"] == pytest.approx(lin, rel=1e-3), k
        assert got[k]["log"] == pytest.approx(log, rel=1e-3), k


# ---------------------------------------------------------------------------
# S1: the surrogate optimiser A/B


def test_milestone_counts():
    f = [10.0, 5.0, 0.2, 0.05, 0.009, 0.02]
    assert N.evals_to(f, 1e-2) == 4          # f/f[0] = 0.005 at evaluation 4
    assert N.evals_to(f, 1e-3) == 5          # 0.0009 at evaluation 5
    assert N.evals_to(f, 1e-4) is None
    assert N.evals_to([1.0, math.nan, 0.5], 0.6) == 3
    errs = [[5.0, 3.0], [0.9, 2.0], [0.5, 0.8], [2.0, 0.1]]
    assert N.evals_to_error(errs, 1.0) == 3   # every identifiable direction within 1 %


def test_surrogate_arms_share_the_physical_start():
    sur = N.Surrogate(H_P, P_STAR, P0, SIG, V_ident=np.eye(4)[:, :2])
    p_start = P0 * np.array([2 ** 0.5, 2 ** -0.5, 2 ** 0.5, 2 ** -0.5])
    first = {}
    for kind in KINDS:
        tr = _tr(kind)
        f, g = sur.fg(tr)(tr.to_v(p_start))[:2]
        first[kind] = f
    assert first["raw"] == pytest.approx(first["linear"], rel=1e-14)
    assert first["raw"] == pytest.approx(first["log"], rel=1e-14)


@pytest.mark.parametrize("kind", ("raw", "linear"))
def test_surrogate_gradient_is_the_chain_rule(kind):
    sur = N.Surrogate(H_P, P_STAR, P0, SIG, V_ident=np.eye(4)[:, :2])
    tr = _tr(kind)
    v = tr.to_v(P0 * 1.2)
    f, g, _ = sur.fg(tr)(v)
    for i in range(4):
        h = 1e-6 * max(1.0, abs(v[i]))
        e = np.zeros(4)
        e[i] = h
        fd = (sur.fg(tr)(v + e)[0] - sur.fg(tr)(v - e)[0]) / (2 * h)
        assert abs(fd - g[i]) <= 1e-6 * max(abs(g[i]), 1e-3 * np.max(np.abs(g)))


def test_surrogate_lbfgs_smoke(tmp_path):
    sur = N.Surrogate(np.diag([4.0, 1.0, 0.5, 2.0]), P_STAR, P0, SIG, V_ident=np.eye(4))
    run = N.run_surrogate_arm(sur, _tr("linear"), P0, "lbfgs", tmp_path / "a", maxeval=30)
    assert len(run["f"]) <= 30 and min(run["f"]) < run["f"][0]
    assert (tmp_path / "a" / "iterations.jsonl").is_file()
    with pytest.raises(FileExistsError):
        N.run_surrogate_arm(sur, _tr("linear"), P0, "lbfgs", tmp_path / "a", maxeval=30)


# ---------------------------------------------------------------------------
# the real-model actions on a duck-typed calibration (what gate/profile/twin touch)

T = N.T
TRUTH = "0.3,0.2333333333333333,7.5,15"


class _FakeCalib:
    def __init__(self):
        self.cs = N.tke_control_set()
        self.n_evals = 0
        self.fun_calls = 0
        self.u_truth = self.cs.to_u(P_STAR)

    def fun_and_grad(self, u):
        self.n_evals += 1
        return _Base()(u)

    def fun(self, u):
        self.fun_calls += 1
        f, _, aux = _Base()(u)
        return f, {"J_data": f}

    def physical(self, u):
        return (P0 * np.exp(SIG * np.asarray(u, np.float64))).tolist()

    def config(self):
        return {"fake": True}


@pytest.fixture
def no_provenance(monkeypatch):
    monkeypatch.setattr(T, "_run_info", lambda calib, action, **kw: {"fake": True})


def test_gate_verdict_uses_the_repeat_spread():
    g1 = np.array([1.0, -2.0, 3e-3, 4.0])
    g2 = g1 * (1.0 + 1e-5 * np.array([1.0, -1.0, 1.0, -1.0]))
    assert N.gate_verdict((5.0, g1), (5.0, g2), (5.0, g1 + 2.0 * (g2 - g1)))["pass"]
    bad = g1 + 1e-3 * np.max(np.abs(g1))           # 10x the 1e-4 floor, far beyond the spread
    assert not N.gate_verdict((5.0, g1), (5.0, g2), (5.0, bad))["pass"]
    assert not N.gate_verdict((5.0, g1), (5.0, g1), (5.0 + 1e-6, g1))["pass"]


def test_run_gate_on_a_deterministic_calibration(tmp_path):
    fake = _FakeCalib()
    out = N.run_gate(fake, fake.cs, tmp_path)
    assert out["gate"]["pass"] and not out["fallback_needed"]
    assert out["gate"]["diff_lin_g"] == [0.0] * 4        # CPU: linear at v = 0 IS log at u = 0
    assert fake.n_evals == 3 and fake.fun_calls == 1
    with pytest.raises(FileExistsError):
        N.run_gate(fake, fake.cs, tmp_path)


def test_build_calib_constructs_exactly_what_tke_lbfgs_main_does(tmp_path, monkeypatch):
    rec = {"bcm": [], "cal": []}

    class Stop(Exception):
        pass

    class Model:
        s0 = "S0"

    def bcm(*a, **k):
        rec["bcm"].append((a, k))
        return Model()

    def cal(*a, **k):
        rec["cal"].append((a[1:], {kk: v for kk, v in k.items()}))
        raise Stop

    monkeypatch.setattr(T, "build_calib_model", bcm)
    monkeypatch.setattr(T, "Calibration", cal)
    argv = ["--mode", "twin", "--prior", "off", "--months", "1,4,7,10", "--window-steps", "12",
            "--truth", TRUTH, "--states-dir", str(tmp_path), "--out", str(tmp_path / "o")]
    with pytest.raises(Stop):
        T.main(["calib"] + argv)
    with pytest.raises(Stop):
        N.build_calib(argv)
    (a1, k1), (a2, k2) = rec["bcm"]
    assert repr(a1) == repr(a2) and repr(sorted(k1.items())) == repr(sorted(k2.items()))
    (a1, k1), (a2, k2) = rec["cal"]
    assert repr(a1) == repr(a2) and repr(sorted(k1.items())) == repr(sorted(k2.items()))
    assert np.array_equal(k2["u_truth"], N.tke_control_set().to_u([0.3, 0.7 / 3, 7.5, 15.0]))


def test_profile_points_and_metrics():
    pts = N.profile_points(P_STAR)
    assert len(pts) == 33 and pts[0][0] == -1
    fac = np.array([2.0 ** (k / 2.0) for k in N.PROFILE_K])
    m_log = N.profile_metrics(fac, np.log(fac) ** 2)
    m_lin = N.profile_metrics(fac, (fac - 1.0) ** 2)
    assert m_log["ratio_2"] == pytest.approx(1.0) and m_log["ratio_4"] == pytest.approx(1.0)
    assert m_lin["ratio_2"] == pytest.approx(4.0) and m_lin["ratio_4"] == pytest.approx(16.0)
    assert m_log["r2_log_x4"] == pytest.approx(1.0) and m_log["r2_lin_x4"] < 0.99
    assert m_lin["r2_lin_x4"] == pytest.approx(1.0) and m_lin["r2_log_x4"] < 0.99


def test_run_profile_resumes(tmp_path):
    fake = _FakeCalib()
    full = N.run_profile(fake, fake.cs, tmp_path / "a")
    assert fake.fun_calls == 33
    lines = (tmp_path / "a" / "profile.jsonl").read_text().splitlines()
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "profile.jsonl").write_text("\n".join(lines[:5]) + "\n")
    fake2 = _FakeCalib()
    part = N.run_profile(fake2, fake2.cs, tmp_path / "b")
    assert fake2.fun_calls == 28 and part["metrics"] == full["metrics"]


def test_twin_log_arm_is_the_plain_driver_bitwise(tmp_path, no_provenance):
    from fesom_jax import optim_lbfgs
    fake = _FakeCalib()
    N.run_twin(fake, fake.cs, tmp_path / "log", kind="log", start="p0", model_budget=100,
               maxeval=12)
    T.run_lbfgs(_FakeCalib(), tmp_path / "plain", u0=np.zeros(4), maxiter=12, maxeval=12,
                options=optim_lbfgs.LBFGSOptions(expand=T.LBFGS_EXPAND))

    def evals(d):
        return [json.loads(x) for x in (d / "iterations.jsonl").read_text().splitlines()
                if json.loads(x)["type"] == "eval"]
    a, b = evals(tmp_path / "log"), evals(tmp_path / "plain")
    assert len(a) == len(b) == 12
    assert [(x["u"], x["f"], x["g"]) for x in a] == [(x["u"], x["f"], x["g"]) for x in b]


def test_twin_meta_must_match_and_the_model_budget_survives_a_resume(tmp_path, no_provenance):
    fake = _FakeCalib()
    out = tmp_path / "lin"
    r = N.run_twin(fake, fake.cs, out, kind="linear", start="plus", model_budget=3, maxeval=40)
    assert r["status"] == "model_budget" and N._model_evals_logged(out / "iterations.jsonl") == 3
    fake2 = _FakeCalib()
    r2 = N.run_twin(fake2, fake2.cs, out, kind="linear", start="plus", model_budget=3, maxeval=40)
    assert r2["status"] == "model_budget" and fake2.n_evals == 0
    with pytest.raises(ValueError):
        N.run_twin(fake2, fake2.cs, out, kind="log", start="plus", model_budget=3, maxeval=40)
    meta = json.loads((out / "norm_ab.json").read_text())
    assert meta["transform"] == "linear" and meta["start"] == "plus"
    np.testing.assert_allclose(N.Transform("linear", tuple(P0), tuple(SIG)).to_p(meta["v0"]),
                               P0 * N.STARTS["plus"], rtol=1e-15)


# ---------------------------------------------------------------------------
# Task 7: the decision rule and the run reader


def _m(log, lin, rep):
    """milestone evaluation counts: {start: (log, linear)} and the replicate's log-p0 count."""
    return {"arms": {s: {"log": a, "linear": b} for s, (a, b) in zip(("p0", "plus", "minus"),
                                                                    zip(log, lin))},
            "replicate": rep}


def test_decision_rule_outcomes():
    no_stall = {}
    assert N.decide(_m((5, 6, 5), (10, 12, 9), 5), no_stall, budget=25)["verdict"] == "log better"
    assert N.decide(_m((10, 12, 9), (5, 6, 5), 10), no_stall, budget=25)["verdict"] == \
        "linear better"
    assert N.decide(_m((5, 6, 5), (6, 7, 5), 5), no_stall, budget=25)["verdict"] == \
        "no practical difference"
    # a 4-evaluation gap is not enough when the replicate itself moved by 3 (2·noise = 6)
    assert N.decide(_m((5, 6, 5), (9, 10, 9), 8), no_stall, budget=25)["verdict"] == \
        "no practical difference"
    assert N.decide(_m((None, None, 5), (None, None, 6), None), no_stall, budget=25)["verdict"] \
        == "inconclusive"
    # never reaching counts as budget + 1
    assert N.decide(_m((5, 6, 5), (None, None, None), 5), no_stall, budget=25)["verdict"] == \
        "log better"


def test_decision_rule_linear_stall_at_the_edge():
    stall = {"minus": True}
    r = N.decide(_m((5, 6, 9), (5, 6, None), 5), stall, budget=25)
    assert r["verdict"] == "log better" and r["reason"] == "linear stalled at p = 0"


def test_read_run_counts_model_evaluations_only(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "norm_ab.json").write_text(json.dumps({"transform": "linear", "start": "minus",
                                                "replicate": 0, "model_budget": 25}))
    rows = [{"type": "header"},
            {"type": "eval", "f": 10.0, "aux": {"J_data": 10.0, "p": [0.1, 0.7, 3.75, 30.0]}},
            {"type": "eval", "f": float("nan"), "aux": {"out_of_domain": True,
                                                        "p": [0.1, -0.1, 3.75, 30.0]}},
            {"type": "eval", "f": 0.05, "aux": {"J_data": 0.05, "p": [0.2, 0.3, 7.0, 16.0]}},
            {"type": "stop", "status": "line_search_stalled"}]
    (d / "iterations.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    r = N.read_run(d)
    assert r["f"] == [10.0, 0.05] and r["n_rejected"] == 1
    assert r["status"] == "line_search_stalled" and len(r["p"]) == 2


def test_cli_forwards_the_tke_options_untouched():
    argv = ["twin", "--transform", "linear", "--start", "minus", "--mode", "twin", "--prior", "off",
            "--maxeval", "40", "--out", "x"]
    args, rest = N._parse(argv)
    assert args.model_budget == 25 and args.transform == "linear" and args.start == "minus"
    assert rest == ["--mode", "twin", "--prior", "off", "--maxeval", "40", "--out", "x"]
    args, rest = N._parse(["twin", "--transform", "log", "--start", "p0", "--model-budget", "7",
                           "--mode", "twin"])
    assert args.model_budget == 7 and rest == ["--mode", "twin"]
