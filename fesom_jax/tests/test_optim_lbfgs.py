"""Host-side L-BFGS (fesom_jax.optim_lbfgs): convergence, diagonal scaling, pair rules, resume, stall.

Unit tests on analytic problems (no mesh, no JAX). The quadratics have their minimum value at 0
(``½·eᵀAe``), so that ``f`` keeps full relative precision near the optimum; a quadratic whose
minimum value is O(1) cannot resolve an Armijo decrease below ~1e-16·|f| and stalls short of a
1e-8 gradient for reasons that have nothing to do with the optimiser.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fesom_jax.optim_lbfgs import LBFGS, LBFGSOptions, Limits, minimize

N_QUAD = 4
M_DEFAULT = 5


def _quadratic(seed, rotated):
    rng = np.random.default_rng(seed)
    h = np.logspace(0.0, 3.0, N_QUAD)                    # condition number 1e3
    Q = np.linalg.qr(rng.normal(size=(N_QUAD, N_QUAD)))[0] if rotated else np.eye(N_QUAD)
    A = Q @ np.diag(h) @ Q.T
    u_star = rng.normal(size=N_QUAD)

    def fg(u):
        e = u - u_star
        return 0.5 * e @ A @ e, A @ e
    return fg, u_star


def _rosenbrock(x):
    f = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
    g = np.array([-400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
                  200.0 * (x[1] - x[0] ** 2)])
    return f, g


@pytest.mark.parametrize("seed", range(10))
def test_diagonal_quadratic_converges_within_3n_plus_10(seed):
    fg, u_star = _quadratic(seed, rotated=False)
    r = minimize(fg, np.zeros(N_QUAD), maxiter=3 * N_QUAD + 10, maxeval=200, gtol=1e-8,
                 options=LBFGSOptions(m=M_DEFAULT))
    assert r.status == "gtol", (r.status, r.n_iter)
    assert np.linalg.norm(r.g) < 1e-8 and r.n_iter <= 22
    np.testing.assert_allclose(r.u, u_star, atol=1e-8)


@pytest.mark.parametrize("seed", range(10))
def test_rotated_quadratic_converges(seed):
    # A dense (rotated) Hessian with condition number 1e3 is not reachable in 3n+10 = 22
    # iterations by limited-memory BFGS without an exact line search: scipy's L-BFGS-B
    # (maxcor 5) needs a median 27 / max 44 iterations over 50 such problems, this one a median
    # 24.5 / max 36 (measured 2026-09-27). The bound here is 40.
    fg, u_star = _quadratic(seed, rotated=True)
    r = minimize(fg, np.zeros(N_QUAD), maxiter=40, maxeval=200, gtol=1e-8)
    assert r.status == "gtol", (r.status, r.n_iter)
    np.testing.assert_allclose(r.u, u_star, atol=1e-7)


def test_rosenbrock():
    r = minimize(_rosenbrock, np.array([-1.2, 1.0]), maxiter=200, maxeval=1000, gtol=1e-6)
    assert r.status == "gtol" and r.n_iter <= 200, (r.status, r.n_iter)
    np.testing.assert_allclose(r.u, [1.0, 1.0], atol=1e-5)


def test_diagonal_scaling_approaches_inverse_hessian():
    h = np.logspace(-1.0, 2.0, 10)                        # a 1000x spread of curvatures

    def fg(u):
        return 0.5 * np.sum(h * u * u), h * u
    # m = 1: the direction leans on D most, and D gets the most updates before convergence
    r = minimize(fg, np.ones(10), maxiter=60, maxeval=400, gtol=1e-12, options=LBFGSOptions(m=1))
    Dh = r.machine.D * h
    assert np.all(np.abs(Dh - 1.0) < 0.15), Dh            # scalar gamma*I would be off by 1000x
    r5 = minimize(fg, np.ones(10), maxiter=60, maxeval=400, gtol=1e-12)
    Dh5 = r5.machine.D * h
    assert np.all((Dh5 > 0.5) & (Dh5 < 2.0)), Dh5


def _first_trial(fg=None):
    """A machine after its initial evaluation at u0 = (1, 0): f = 1, g = (1, 0)."""
    m = LBFGS(np.array([1.0, 0.0]))
    m.ask()
    m.tell(1.0, np.array([1.0, 0.0]))
    return m


def test_negative_curvature_pair_is_skipped_not_stored():
    m = _first_trial()
    u = m.ask()
    assert u[0] < 1.0                                     # moved along -g
    info = m.tell(0.5, np.array([2.0, 0.0]))              # f decreased, g grew: s'y < 0
    assert info["accepted"] and not info["pair_stored"] and info["sy"] < 0.0
    assert len(m.S) == 0 and m.D is None
    np.testing.assert_array_equal(m.u, u)


def test_wolfe_curvature_failure_accepts_without_storing():
    m = _first_trial()
    m.ask()
    # the slope along d at the trial is still (almost) the initial one: g0'd < g+'d < c2*g0'd,
    # so s'y > 0 and only the Wolfe curvature test rejects the pair
    info = m.tell(0.99, np.array([0.999995, 0.0]))
    assert info["accepted"] and not info["wolfe"] and not info["pair_stored"]
    assert info["sy"] > 1e-10 * 0.004
    assert len(m.S) == 0


def test_positive_curvature_pair_is_stored_and_sets_the_diagonal():
    m = _first_trial()
    m.ask()
    info = m.tell(0.9, np.array([0.5, 0.0]))
    assert info["accepted"] and info["wolfe"] and info["pair_stored"]
    assert len(m.S) == 1 and m.D is not None and np.all(m.D > 0.0)


def test_armijo_failure_shortens_then_accepts_best_decrease():
    m = _first_trial()
    u1 = m.ask()
    t1 = m.t
    info = m.tell(2.0, np.array([1.0, 0.0]))              # increase: shorten
    assert not info["accepted"] and m.t <= 0.5 * t1 and m.t >= 0.01 * t1
    m.ask()
    m.tell(1.0 + 1e-300, np.array([1.0, 0.0]))            # no decrease: shorten again
    m.ask()
    info = m.tell(1.0, np.array([1.0, 0.0]))              # third trial, no decrease: failure
    assert info["reset"] == "line_search_failed" and m.status == "running"
    np.testing.assert_array_equal(m.u, [1.0, 0.0])
    assert not np.array_equal(m.ask(), u1)


def test_wrong_sign_gradient_stalls_within_the_bound():
    fg, _ = _quadratic(0, rotated=True)

    def wrong(u):
        f, g = fg(u)
        return f, -g
    opts = LBFGSOptions()
    r = minimize(wrong, np.zeros(N_QUAD), maxiter=100, maxeval=100, gtol=1e-8, options=opts)
    assert r.status == "line_search_stalled"
    assert r.n_eval <= 1 + 2 * (1 + opts.max_halvings) == 7
    assert r.n_iter == 2


def test_nonfinite_trial_is_a_failed_armijo_test():
    m = _first_trial()
    t = m.t
    m.ask()
    info = m.tell(np.nan, np.array([np.nan, 0.0]))
    assert not info["accepted"] and m.t == pytest.approx(0.1 * t)


def test_stop_reasons_maxiter_maxeval_ftol_are_logged(tmp_path):
    for kw, status in ((dict(maxiter=3, maxeval=100), "maxiter"),
                       (dict(maxiter=100, maxeval=4), "maxeval"),
                       (dict(maxiter=100, maxeval=100, ftol=0.5, ftol_iters=1), "ftol")):
        log = tmp_path / f"{status}.jsonl"
        r = minimize(_rosenbrock, np.array([-1.2, 1.0]), gtol=1e-12, log_path=log, **kw)
        assert r.status == status
        recs = [json.loads(ln) for ln in log.read_text().splitlines()]
        assert recs[0]["type"] == "header" and recs[-1] == {**recs[-1], "type": "stop",
                                                              "status": status}
        assert recs[-2]["status"] in (status, "running")


def _eval_rows(path):
    rows = []
    for ln in path.read_text().splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:                      # the torn line an interruption left
            continue
        if r.get("type") == "eval":
            rows.append((r["u"], r["f"], r["g"], r["accepted"], r["pair_stored"], r["status"]))
    return rows


def test_resume_from_a_truncated_log_is_bitwise(tmp_path):
    full = tmp_path / "full.jsonl"
    r_full = minimize(_rosenbrock, np.array([-1.2, 1.0]), maxiter=200, maxeval=1000, gtol=1e-6,
                      log_path=full)
    lines = full.read_text().splitlines(keepends=True)
    assert len(lines) > 30
    # interrupted run: the header, 20 evaluations, and half of the 21st line (torn write)
    cut = tmp_path / "cut.jsonl"
    cut.write_text("".join(lines[:21]) + lines[21][: len(lines[21]) // 2])
    calls = []

    def counted(u):
        calls.append(1)
        return _rosenbrock(u)
    r_res = minimize(counted, np.array([-1.2, 1.0]), maxiter=200, maxeval=1000, gtol=1e-6,
                     log_path=cut, resume=True)
    assert r_res.u.tobytes() == r_full.u.tobytes() and r_res.f == r_full.f
    assert r_res.status == r_full.status and r_res.n_eval == r_full.n_eval
    assert len(calls) == r_full.n_eval - 20                # the 20 logged ones were not re-run
    assert _eval_rows(cut) == _eval_rows(full)
    # a run stopped early by maxeval, then extended, is the same run
    early = tmp_path / "early.jsonl"
    minimize(_rosenbrock, np.array([-1.2, 1.0]), maxiter=200, maxeval=15, gtol=1e-6,
             log_path=early)
    r_ext = minimize(_rosenbrock, np.array([-1.2, 1.0]), maxiter=200, maxeval=1000, gtol=1e-6,
                     log_path=early)
    assert r_ext.u.tobytes() == r_full.u.tobytes()
    assert _eval_rows(early) == _eval_rows(full)


def test_resume_refuses_a_foreign_or_existing_log(tmp_path):
    log = tmp_path / "a.jsonl"
    minimize(_rosenbrock, np.array([-1.2, 1.0]), maxiter=5, maxeval=10, log_path=log)
    with pytest.raises(FileExistsError):
        minimize(_rosenbrock, np.array([-1.2, 1.0]), maxiter=5, log_path=log, resume=False)
    with pytest.raises(ValueError, match="u0"):
        minimize(_rosenbrock, np.array([-1.0, 1.0]), maxiter=5, log_path=log)
    with pytest.raises(ValueError, match="options"):
        minimize(_rosenbrock, np.array([-1.2, 1.0]), maxiter=5, log_path=log,
                 options=LBFGSOptions(m=3))
    with pytest.raises(ValueError, match="does not match|diverged"):
        minimize(lambda u: (_rosenbrock(u)[0] * 2.0, _rosenbrock(u)[1] * 2.0),
                 np.array([-1.2, 1.0]), maxiter=50, maxeval=100,
                 log_path=_tamper(log, tmp_path))


def _tamper(log, tmp_path):
    """A copy of ``log`` whose 3rd evaluation reports a different gradient."""
    out = tmp_path / "tampered.jsonl"
    rows = log.read_text().splitlines()
    k = [i for i, ln in enumerate(rows) if json.loads(ln).get("type") == "eval"][2]
    rec = json.loads(rows[k])
    rec["g"] = [x * 3.0 for x in rec["g"]]
    rows[k] = json.dumps(rec)
    out.write_text("\n".join(rows) + "\n")
    return out


def test_accept_without_any_pair_cold_starts_with_the_last_step_length():
    m = _first_trial()
    u1 = m.ask()
    m.tell(0.5, np.array([2.0, 0.0]))                     # accepted, pair skipped (s'y < 0)
    assert len(m.S) == 0
    d = m.ask() - m.u
    # not the unscaled -g (length 2): the reset rule, length = the last accepted step
    assert np.linalg.norm(d) == pytest.approx(np.linalg.norm(u1 - np.array([1.0, 0.0])),
                                              rel=1e-15)
    assert d[0] < 0.0


def test_expand_grows_the_cold_start_after_a_dropped_pair():
    """With ``expand`` an accepted step whose pair was dropped (memory still empty) is followed by
    a cold start ``expand`` times longer; the default (1) keeps the length (test above)."""
    m = LBFGS(np.array([1.0, 0.0]), LBFGSOptions(expand=4.0))
    m.ask()
    m.tell(1.0, np.array([1.0, 0.0]))                    # initial evaluation
    u1 = m.ask()
    L1 = np.linalg.norm(u1 - np.array([1.0, 0.0]))
    m.tell(0.5, np.array([2.0, 0.0]))                     # accepted, s'y < 0: dropped
    u2 = m.ask()
    assert np.linalg.norm(u2 - u1) == pytest.approx(4.0 * L1, rel=1e-15)
    m.tell(0.2, np.array([3.0, 0.0]))                     # dropped again: 4x again
    u3 = m.ask()
    assert np.linalg.norm(u3 - u2) == pytest.approx(16.0 * L1, rel=1e-15)
    with pytest.raises(ValueError):
        LBFGSOptions(expand=0.5)


def _concave_start(x):
    # f = -x²/2 + x⁴/400: concave for x < 5.77, minimum at x = 10; from x = 0.1 every early step
    # sees the gradient grow (s'y < 0), so no pair is stored until the step reaches the bowl
    f = -0.5 * x[0] ** 2 + x[0] ** 4 / 400.0
    return f, np.array([-x[0] + x[0] ** 3 / 100.0])


def test_expand_escapes_the_crawl_of_dropped_pairs(tmp_path):
    r1 = minimize(_concave_start, np.array([0.1]), maxiter=40, maxeval=40, gtol=1e-5)
    assert r1.u[0] < 1.0 and r1.status == "maxeval"      # the first step's length, 40 times
    log = tmp_path / "e.jsonl"
    r4 = minimize(_concave_start, np.array([0.1]), maxiter=40, maxeval=40, gtol=1e-5,
                  options=LBFGSOptions(expand=4.0), log_path=log)
    assert r4.status == "gtol" and abs(r4.u[0] - 10.0) < 1e-5 and r4.n_eval < 40
    # the expanded machine replays bitwise from its own truncated log
    lines = log.read_text().splitlines()
    part = tmp_path / "p.jsonl"
    part.write_text("\n".join(lines[:6]) + "\n")
    r4b = minimize(_concave_start, np.array([0.1]), maxiter=40, maxeval=40, gtol=1e-5,
                   options=LBFGSOptions(expand=4.0), log_path=part)
    assert r4b.u.tobytes() == r4.u.tobytes() and r4b.n_eval == r4.n_eval
