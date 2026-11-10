"""The L-BFGS chain of the sea-ice calibration (``scripts/icecal/icecal.py chain-init`` /
``chain-step`` and ``scripts/icecal/job_chain_gh200.sh``), model-free: a synthetic quadratic window
evaluator stands in for the model, and ``sbatch`` / ``squeue`` are fake executables on ``PATH``
(the job script runs for real; nothing reaches Slurm). A small emulator runs every submitted
chain step in-process, in submission order.

What is pinned:

* a chain driven through ``chain-step`` reproduces ``optim_lbfgs.minimize`` on the same function
  BITWISE (every evaluated u, f, g and the stop status), with the prior on, with the prior off
  and a target outside the physical domain (those points are failed evaluations recorded by the
  coordinator, the evaluator never sees them), with a NaN gradient at evaluation 3 (a failed
  point; the chain goes on), and with two windows;
* ``maxeval`` is respected; one job is submitted per posted evaluation after the first, and each
  submission is in ``jobs.txt`` and the ledger;
* ``STOP`` (present at the start, or created during the evaluation) prevents submission; the job
  script refuses at STOP and, by hand, at >= 4 dolpung GPUs held or eligible (running jobs plus
  pending jobs not waiting on a dependency; chain-step then exits 3, the window file kept); inside
  a dolpung job it submits the next step afterany on itself, without the cap (the GPU is handed on);
* ``--free`` optimises a subset (short or full names; the others stay at u0 in every posted point)
  and the chain reproduces ``minimize`` on the restricted function; ``--fmin-frac`` reaches the
  options (the cold-start length);
* an existing window file is never overwritten (``publish_json`` refuses; chain-step does not
  re-evaluate);
* a raising evaluation writes no window file and submits nothing;
* two windows of one evaluation run the coordinator once (``coord.lock``) and submit the next
  round once;
* ``chain-init`` writes ``run_config.json`` once and validates its arguments.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "icecal"))
import icecal as I  # noqa: E402

T = I._lbfgs()
optim_lbfgs = I.optim_lbfgs
CS = I.ice_thermo_control_set()
N = CS.n
H = np.array([4.0, 3.0, 2.0, 1.0, 10.0, 5.0, 2.0, 2.0, 1.0, 3.0])
#: twin truth of the plan's table (in the physical domain)
A_IN = np.array([0.5, -1.0, 0.5, -1.0, -0.3, 0.5, 0.5, -0.5, 0.5, 0.7])
#: albsn target u = 6 (p = 1.17 > 0.95): outside the domain, so the line search probes there
A_OUT = A_IN.copy()
A_OUT[0] = 6.0


def quad(u, a, h=H, w=1.0):
    r = np.asarray(u, np.float64) - a
    return w * 0.5 * float(np.sum(h * r * r)), w * h * r


# ---------------------------------------------------------------------------
# fixtures: fake Slurm, a run directory


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    """Fake ``sbatch``/``squeue`` first on PATH; the ledger and the wrapper in tmp."""
    b = tmp_path / "bin"
    b.mkdir()
    log = tmp_path / "sbatch.log"
    (b / "sbatch").write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> {log}\n'
        f'n=$(wc -l < {log})\n'
        'echo "$((900000 + n));levante"\n')
    (b / "squeue").write_text(
        "#!/bin/bash\n"
        'for i in $(seq 1 ${FAKE_SQUEUE_N:-0}); do echo "RUNNING None"; done\n'
        'for i in $(seq 1 ${FAKE_SQUEUE_DEP:-0}); do echo "PENDING Dependency"; done\n'
        'for i in $(seq 1 ${FAKE_SQUEUE_PRI:-0}); do echo "PENDING Priority"; done\n')
    for f in ("sbatch", "squeue"):
        (b / f).chmod(0o755)
    ledger = tmp_path / "ledger.txt"
    monkeypatch.setenv("PATH", f"{b}:{os.environ['PATH']}")
    monkeypatch.setenv("ICECAL_LEDGER", str(ledger))
    monkeypatch.setenv("ICECAL_WRAPPER", str(tmp_path / "wrapper.sbatch"))
    for v in ("FAKE_SQUEUE_N", "FAKE_SQUEUE_DEP", "FAKE_SQUEUE_PRI", "SLURM_JOB_ID",
              "SLURM_JOB_PARTITION"):
        monkeypatch.delenv(v, raising=False)

    class S:
        pass
    s = S()
    s.log, s.ledger = log, ledger

    def calls():
        return log.read_text().splitlines() if log.exists() else []
    s.calls = calls
    return s


def init(tmp_path, windows=(2013,), *, prior=True, maxeval=12, mode="real", truth=None,
         free=None, fmin_frac=None):
    files = tmp_path / "inputs"
    files.mkdir(parents=True, exist_ok=True)
    states, obs = {}, {}
    for y in windows:
        (files / f"fesom.{y}.001.00000").mkdir(exist_ok=True)
        (files / f"obs_{y}.npz").write_bytes(f"obs {y}".encode())
        states[y], obs[y] = files / f"fesom.{y}.001.00000", files / f"obs_{y}.npz"
    R = tmp_path / "chain"
    I.chain_init(R, list(windows), mode=mode, prior=prior, maxeval=maxeval, truth=truth,
                 states=states, obs=obs, free=free, fmin_frac=fmin_frac)
    return R


def make_eval(targets: dict, nan_k=None, seen=None, raise_k=None, stop_at_k=None, R=None):
    """Quadratic window evaluator: window w -> 0.5*w_w*sum h (u - a_w)^2 (weights split J)."""
    def ev(cfg, window, k, u, out_dir):
        if seen is not None:
            seen.append((int(k), int(window), np.asarray(u).copy()))
        assert CS.in_domain(u), f"the evaluator got an out-of-domain point {u}"
        if raise_k is not None and k == raise_k:
            raise RuntimeError("synthetic evaluation crash")
        a, w = targets[int(window)]
        J, g = quad(u, a, w=w)
        if nan_k is not None and k == nan_k:
            g = np.full(N, np.nan)
        if stop_at_k is not None and k == stop_at_k:
            (Path(R) / "STOP").write_text("stop\n")
        return {"J": J, "g": g.tolist(), "platform": "cpu", "device": "cpu",
                "compile_s": 0.0, "run_s": 0.0}
    return ev


def drive(R, slurm, ev, windows=(2013,), limit=200):
    """Start one chain step per window by hand (as the orchestrator does), then run every
    submitted step in submission order until nothing is pending. Returns the exit codes."""
    pending = list(windows)
    seen, rcs = 0, []
    while pending:
        w = pending.pop(0)
        rcs.append(I.chain_step(R, w, evaluate=ev, wait_s=0.0))
        calls = slurm.calls()
        for ln in calls[seen:]:
            args = shlex.split(ln)
            assert "--run-dir" in args and args[args.index("--run-dir") + 1] == str(R)
            pending.append(int(args[args.index("--window") + 1]))
        seen = len(calls)
        assert len(rcs) < limit
    return rcs


def reference(tmp_path, fwin, *, prior, maxeval, nan_k=None, name="ref", free=None,
              fmin_frac=None):
    """``optim_lbfgs.minimize`` on the same function, with the coordinator's host arithmetic
    (fsum over windows, the prior added, out-of-domain points failed); ``free``: on the
    restricted function (``tke_lbfgs.restrict``, the others at 0)."""
    k = [0]
    dcs = I.DomainControlSet(CS.controls)

    def fg(u):
        k[0] += 1
        u = np.asarray(u, np.float64)
        try:
            dcs.check(u)
        except ValueError:
            return float("nan"), np.full(N, np.nan)
        parts = [f(u) for f in fwin]
        Js = [p[0] for p in parts]
        gs = [np.full(N, np.nan) if (nan_k is not None and k[0] == nan_k) else p[1]
              for p in parts]

        def s(v):
            v = [float(x) for x in v]
            return math.fsum(v) if all(math.isfinite(x) for x in v) else float("nan")
        jd = s(Js)
        gd = np.array([s(c) for c in zip(*gs)], np.float64)
        return jd + (T._prior(u) if prior else 0.0), gd + (u if prior else 0.0)
    log = tmp_path / f"{name}.jsonl"
    idx = T.free_indices(CS, free)
    f0 = fg if len(idx) == N else T.restrict(fg, np.zeros(N), idx)
    r = optim_lbfgs.minimize(f0, np.zeros(len(idx)), maxiter=maxeval, maxeval=maxeval, gtol=0.0,
                             ftol=0.0, log_path=log, options=I.chain_options(fmin_frac))
    return r, optim_lbfgs._read_log(log)[1]


def evals_of(R):
    return optim_lbfgs._read_log(Path(R) / "iterations.jsonl")[1]


def same_rows(a, b):
    key = lambda r: json.dumps([r["u"], r["f"], r["g"], r.get("accepted")])  # noqa: E731
    return [key(r) for r in a] == [key(r) for r in b]


def last_stop(R):
    rows = [json.loads(x) for x in (Path(R) / "iterations.jsonl").read_text().splitlines()]
    return rows[-1] if rows[-1].get("type") == "stop" else None


# ---------------------------------------------------------------------------
# chain-init


def test_chain_init_writes_the_config_once(tmp_path):
    R = init(tmp_path, maxeval=7)
    cfg = json.loads((R / "run_config.json").read_text())
    assert cfg["options"] == json.loads(json.dumps(vars(optim_lbfgs.LBFGSOptions(expand=4))))
    assert cfg["calib"]["maxeval"] == 7 and cfg["calib"]["controls"] == list(CS.names)
    assert cfg["free"] == list(CS.names) and cfg["calib"]["mode"] == "real"
    w = cfg["calib"]["window_files"]["2013"]
    assert w["obs_sha256"] == I.sha256_file(w["obs"]) and w["start"] == "2013-01-01T00:00:00"
    with pytest.raises(FileExistsError):
        I.chain_init(R, [2013], mode="real", prior=True, states={2013: w["state"]},
                     obs={2013: w["obs"]})
    with pytest.raises(ValueError, match="truth"):
        I.chain_init(tmp_path / "t", [2013], mode="twin", prior=False,
                     states={2013: w["state"]}, obs={2013: w["obs"]})
    with pytest.raises(FileNotFoundError):
        I.chain_init(tmp_path / "m", [2014], mode="real", prior=True, states_dir=tmp_path,
                     obs_dir=tmp_path)
    Rt = init(tmp_path / "tw", mode="twin", prior=False, truth=json.dumps(A_IN.tolist()))
    c = json.loads((Rt / "run_config.json").read_text())["calib"]
    assert c["u_truth"] == A_IN.tolist() and c["prior"] is False
    # the coordinator settings agree with run_config.json (run_mjstep compares them)
    _, coord = I.load_chain(R)
    assert coord["months"] == [2013] and isinstance(coord["cs"], I.DomainControlSet)


# ---------------------------------------------------------------------------
# the chain against optim_lbfgs.minimize


@pytest.mark.parametrize("case", ["prior_in_domain", "prior_off_out_of_domain", "nan_grad"])
def test_chain_reproduces_minimize_bitwise(case, tmp_path, slurm):
    prior, a, nan_k, maxeval = {"prior_in_domain": (True, A_IN, None, 12),
                                "prior_off_out_of_domain": (False, A_OUT, None, 12),
                                "nan_grad": (True, A_IN, 3, 10)}[case]
    R = init(tmp_path, prior=prior, maxeval=maxeval)
    seen = []
    rcs = drive(R, slurm, make_eval({2013: (a, 1.0)}, nan_k=nan_k, seen=seen))
    assert set(rcs) == {0}
    ref, ref_rows = reference(tmp_path, [lambda u: quad(u, a)], prior=prior, maxeval=maxeval,
                              nan_k=nan_k)
    rows = evals_of(R)
    print(f"[chain] {case}: {len(rows)} evaluations, stop {last_stop(R)['status']}, "
          f"f {[r['f'] for r in rows]}")
    assert same_rows(rows, ref_rows), "the chain's iterates differ from minimize's"
    assert last_stop(R)["status"] == ref.status
    assert len(rows) <= maxeval
    # one job per posted evaluation after the first; jobs.txt and the ledger record each
    posted = sorted(int(p.parent.name.split("_")[1]) for p in R.glob("eval_*/u.json"))
    assert len(slurm.calls()) == len(posted) - 1
    jobs = (R / "jobs.txt").read_text().split() if (R / "jobs.txt").exists() else []
    assert len(jobs) == len(posted) - 1
    led = slurm.ledger.read_text().splitlines() if slurm.ledger.exists() else []
    assert [ln.split("\t")[1] for ln in led] == jobs
    assert all(len(ln.split("\t")) == 5 for ln in led)
    for ln in slurm.calls():
        assert "--time=07:50:00" in ln and str(I.ROOT) in ln and "chain-step" in ln
    assert [s[0] for s in seen] == posted          # each posted point evaluated once
    if case == "prior_off_out_of_domain":
        ood = [r for r in rows if "out_of_domain" in (r.get("aux") or {})]
        assert ood, "the line search never left the domain; the case tests nothing"
        assert len(posted) == len(rows) - len(ood)
    if case == "nan_grad":
        r3 = rows[2]
        assert r3["accepted"] is False and len(rows) > 3
        assert not np.all(np.isfinite(np.array(r3["g"], np.float64)))


def test_maxeval_is_respected(tmp_path, slurm):
    R = init(tmp_path, maxeval=4)
    drive(R, slurm, make_eval({2013: (A_IN, 1.0)}))
    rows = evals_of(R)
    assert len(rows) == 4 and last_stop(R)["status"] == "maxeval"
    assert len(slurm.calls()) == 3
    # a further (stray) step does nothing
    assert I.chain_step(R, 2013, evaluate=make_eval({2013: (A_IN, 1.0)}), wait_s=0.0) == 0
    assert len(evals_of(R)) == 4 and len(slurm.calls()) == 3


def test_two_windows_coordinate_once_and_reproduce_minimize(tmp_path, slurm):
    wins = (2013, 2014)
    tg = {2013: (A_IN, 0.6), 2014: (A_IN[::-1].copy(), 0.4)}
    R = init(tmp_path, windows=wins, maxeval=8)
    ev = make_eval(tg)
    # evaluation 1: window 2013 first; the coordinator waits for 2014
    assert I.chain_step(R, 2013, evaluate=ev, wait_s=0.0) == 0
    assert slurm.calls() == [] and len(evals_of(R)) == 0
    assert I.chain_step(R, 2014, evaluate=ev, wait_s=0.0) == 0
    calls = slurm.calls()
    assert sorted(int(shlex.split(c)[shlex.split(c).index("--window") + 1]) for c in calls) \
        == [2013, 2014]
    assert (R / "eval_002" / "coord.lock").is_file() and len(evals_of(R)) == 1
    _, coord = I.load_chain(R)
    assert T._mj_try_coordinate(R, 1, coord) is None        # a second finisher: no-op
    assert len(evals_of(R)) == 1 and len(slurm.calls()) == 2
    # the rest of the chain, emulated
    pending, seen = [], len(calls)
    for c in calls:
        a = shlex.split(c)
        pending.append(int(a[a.index("--window") + 1]))
    while pending:
        I.chain_step(R, pending.pop(0), evaluate=ev, wait_s=0.0)
        new = slurm.calls()[seen:]
        seen += len(new)
        for c in new:
            a = shlex.split(c)
            pending.append(int(a[a.index("--window") + 1]))
    fw = [lambda u, w=w: quad(u, tg[w][0], w=tg[w][1]) for w in wins]
    ref, ref_rows = reference(tmp_path, fw, prior=True, maxeval=8)
    assert same_rows(evals_of(R), ref_rows) and last_stop(R)["status"] == ref.status
    for p in R.glob("eval_*/u.json"):
        k = int(p.parent.name.split("_")[1])
        assert (R / f"eval_{k:03d}" / "win_2013.json").is_file()
        assert (R / f"eval_{k:03d}" / "win_2014.json").is_file()


# ---------------------------------------------------------------------------
# STOP, the GPU cap, crashes, no overwrite


def test_stop_prevents_submission(tmp_path, slurm):
    R = init(tmp_path)
    (R / "STOP").write_text("")
    calls = []
    assert I.chain_step(R, 2013, evaluate=make_eval({2013: (A_IN, 1.0)}, seen=calls),
                        wait_s=0.0) == 0
    assert calls == [] and slurm.calls() == [] and not (R / "eval_001").exists()
    p = subprocess.run([str(I.JOB_SCRIPT), str(R), "2013"], capture_output=True, text=True)
    assert p.returncode == 3 and slurm.calls() == []
    # STOP created while evaluation 2 runs: its window file is written, nothing is submitted
    R2 = init(tmp_path / "b")
    rcs = drive(R2, slurm, make_eval({2013: (A_IN, 1.0)}, stop_at_k=2, R=R2))
    assert set(rcs) == {0}
    assert (R2 / "eval_002" / "win_2013.json").is_file()
    assert len(slurm.calls()) == 1 and (R2 / "eval_003" / "u.json").is_file()
    assert not list(R2.glob("eval_003/win_*.json"))


def test_gpu_cap_refuses_and_the_chain_resumes(tmp_path, slurm, monkeypatch):
    R = init(tmp_path)
    monkeypatch.setenv("FAKE_SQUEUE_N", "4")
    p = subprocess.run([str(I.JOB_SCRIPT), str(R), "2013"], capture_output=True, text=True)
    assert p.returncode == 5 and "cap 4" in p.stdout and slurm.calls() == []
    ev = make_eval({2013: (A_IN, 1.0)})
    assert I.chain_step(R, 2013, evaluate=ev, wait_s=0.0) == 3     # evaluated, not submitted
    assert (R / "eval_001" / "win_2013.json").is_file() and slurm.calls() == []
    rows = [json.loads(x) for x in (R / "chain.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "submit_failed" and "exit 5" in rows[-1]["error"]
    monkeypatch.setenv("FAKE_SQUEUE_N", "3")
    p = subprocess.run([str(I.JOB_SCRIPT), str(R), "2013"], capture_output=True, text=True,
                       env={**os.environ, "WT": str(I.ROOT)})
    assert p.returncode == 0 and len(slurm.calls()) == 1
    assert p.stdout.strip().splitlines()[-1] == "900001"
    assert (R / "jobs.txt").read_text().split() == ["900001"]
    # the resumed step evaluates the posted point 2
    assert I.chain_step(R, 2013, evaluate=ev, wait_s=0.0) == 0
    assert (R / "eval_002" / "win_2013.json").is_file()


def test_cap_counts_held_gpus_and_a_chain_step_hands_its_gpu_on(tmp_path, slurm, monkeypatch):
    R = init(tmp_path)

    def job():
        return subprocess.run([str(I.JOB_SCRIPT), str(R), "2013"], capture_output=True,
                              text=True, env={**os.environ, "WT": str(I.ROOT)})
    # pending jobs waiting on a dependency take over a predecessor's GPU: not counted
    monkeypatch.setenv("FAKE_SQUEUE_N", "3")
    monkeypatch.setenv("FAKE_SQUEUE_DEP", "3")
    p = job()
    assert p.returncode == 0 and len(slurm.calls()) == 1
    assert "--dependency" not in slurm.calls()[0]
    # an eligible pending job is about to take a GPU: counted
    monkeypatch.setenv("FAKE_SQUEUE_PRI", "1")
    p = job()
    assert p.returncode == 5 and "cap 4" in p.stdout and len(slurm.calls()) == 1
    # a chain step inside a dolpung job: afterany on itself, no cap
    monkeypatch.setenv("FAKE_SQUEUE_N", "4")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "dolpung")
    p = job()
    assert p.returncode == 0 and len(slurm.calls()) == 2
    assert "--dependency=afterany:4242" in shlex.split(slurm.calls()[1])
    # inside a job on another partition (a CPU test job) the cap applies
    monkeypatch.setenv("SLURM_JOB_PARTITION", "compute")
    p = job()
    assert p.returncode == 5 and len(slurm.calls()) == 2


def test_free_subset_and_fmin_frac(tmp_path, slurm):
    free = ["con", "ice.c_melt", "albim", "h0"]
    R = init(tmp_path, free=free, fmin_frac=0.05, maxeval=10)
    cfg = json.loads((R / "run_config.json").read_text())
    idx = sorted(CS.names.index(c if c.startswith("ice.") else "ice." + c) for c in free)
    assert cfg["free"] == [CS.names[i] for i in idx]          # index order, full names
    assert cfg["options"]["fmin_frac"] == 0.05
    assert cfg["options"] == json.loads(json.dumps(vars(I.chain_options(0.05))))
    with pytest.raises(ValueError, match="unknown"):
        init(tmp_path / "bad", free=["con", "nosuch"])
    seen = []
    rcs = drive(R, slurm, make_eval({2013: (A_IN, 1.0)}, seen=seen))
    assert set(rcs) == {0}
    fixed = [i for i in range(N) if i not in idx]
    assert seen and all(np.all(u[fixed] == 0.0) for _, _, u in seen)
    assert any(np.any(u[idx] != 0.0) for _, _, u in seen)
    ref, ref_rows = reference(tmp_path, [lambda u: quad(u, A_IN)], prior=True, maxeval=10,
                              free=cfg["free"], fmin_frac=0.05)
    rows = evals_of(R)
    assert same_rows(rows, ref_rows), "the chain's iterates differ from minimize's"
    assert last_stop(R)["status"] == ref.status
    # the cold start: |d| = 2 * fmin_frac * f / |g| on the free components
    r1, r2 = rows[0], rows[1]
    g1 = np.asarray(r1["g"])
    step = np.linalg.norm(np.asarray(r2["u"]) - np.asarray(r1["u"]))
    assert step == pytest.approx(2 * 0.05 * abs(r1["f"]) / np.linalg.norm(g1), rel=1e-12)


def test_a_raising_evaluation_submits_nothing(tmp_path, slurm):
    R = init(tmp_path)
    ev = make_eval({2013: (A_IN, 1.0)}, raise_k=2)
    assert I.chain_step(R, 2013, evaluate=ev, wait_s=0.0) == 0
    assert len(slurm.calls()) == 1
    with pytest.raises(RuntimeError, match="synthetic evaluation crash"):
        I.chain_step(R, 2013, evaluate=ev, wait_s=0.0)
    assert not (R / "eval_002" / "win_2013.json").exists()
    assert len(slurm.calls()) == 1 and len(evals_of(R)) == 1
    assert not (R / "eval_003").exists()


def test_window_file_is_never_overwritten(tmp_path, slurm):
    p = tmp_path / "w" / "win_2013.json"
    p.parent.mkdir()
    I.publish_json(p, {"J_w": 1.0})
    before = p.read_bytes()
    with pytest.raises(FileExistsError, match="never overwritten"):
        I.publish_json(p, {"J_w": 2.0})
    assert p.read_bytes() == before
    # chain-step finds the window file of the posted evaluation 2 and does not evaluate it
    # again: it coordinates with the file as it is (and goes on to evaluation 3)
    R = init(tmp_path)
    ev = make_eval({2013: (A_IN, 1.0)})
    assert I.chain_step(R, 2013, evaluate=ev, wait_s=0.0) == 0
    u2 = json.loads((R / "eval_002" / "u.json").read_text())["u"]
    J, g = quad(u2, A_IN)
    wp = R / "eval_002" / "win_2013.json"
    I.publish_json(wp, {"eval": 2, "month": 2013, "u": u2, "J_w": J, "g_w": g.tolist(),
                        "platform": "cpu", "marker": "written before the step"})
    before = wp.read_bytes()
    calls = []
    assert I.chain_step(R, 2013, evaluate=make_eval({2013: (A_IN, 1.0)}, seen=calls),
                        wait_s=0.0) == 0
    assert [c[0] for c in calls] == [3] and wp.read_bytes() == before
    rows = evals_of(R)
    assert rows[1]["f"] == J + T._prior(np.asarray(u2)) and rows[1]["aux"]["windows"][0]["J"] == J
    assert (R / "eval_003" / "win_2013.json").is_file()
