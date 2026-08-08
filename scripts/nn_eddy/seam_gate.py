"""NN-GM child-plan M1 GATE — the ``gm_nn`` seam in the assembled CORE2 model, on a GPU.

Two claims, both required before the seam is used for anything:

**(a) STABILITY.** 10 days from a developed student state with a RANDOM-perturbed network
(``zero_last=False``, so multipliers range over the full [1/3, 3] bound) must stay physical:
no NaN, ``T ∈ [-3, 40] °C``, ``S ∈ [0, 45]``, ``|η| < 10 m`` at every DAILY mark. Reported
against a ``gm_nn=None`` control run over the same window: peak device memory and s/step, so
the cost of the seam is a measured number rather than an expectation.

**(b) GRADIENT.** ``d(loss)/dα`` for α a uniform scaling of the network's last layer, at the
5-day mark, AD against central differences over an h-sweep: **≤10% agreement**, the bar the
horizon scan met at 4.2–5.4%. Deliberately NOT "finite and nonzero" — this project has a
recorded case of a gradient that was finite, nonzero, and structurally wrong (a dead path),
so the gate is quantitative or it is nothing.

The gate must be reproduced across TWO separate jobs to 3 significant figures (submit the
sbatch twice; the results JSONL appends). Config and driver mirror ``horizon_scan.py`` /
``student_year.py`` — the same model, the same forcing handling, the same remat discipline —
with the plan's LOCKED adjoint switches: ``AdjointConfig(freeze_mixing=True)`` plus
``IceConfig(whichEVP=1, adjoint_mode="rheology")``. Without them the long-window gradient is
inf, which is a property of the model's ice rheology and mixing closure, not of this seam.

Usage (A100, one GPU):
  python scripts/nn_eddy/seam_gate.py --snap /work/.../student_free_1958/snap_day150.pkl \\
      --mode stability
  python scripts/nn_eddy/seam_gate.py --snap ... --mode gradient
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# ⚠️ MUST come before `import fesom_jax`. `python scripts/foo.py` puts the SCRIPT's directory on
# sys.path and never the cwd, so a bare import resolves through the editable install — which
# points at the MAIN checkout. This script's whole subject (`gm_nn`, the seam in `gm.py`, the
# `gm_nn` Params leaf) exists only in this worktree, so without these two lines the gate either
# dies on an ImportError (as it did, job 26798423) or, worse for a script that happened to touch
# only shared modules, measures the wrong code and reports it as this branch's number. The
# assert below is the part that makes it impossible to get wrong quietly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fesom_jax  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

_got = Path(fesom_jax.__file__).resolve()
if ROOT not in _got.parents:
    raise SystemExit(f"FATAL: fesom_jax imported from {_got}, not from {ROOT}")

from fesom_jax import gm_nn, ssh, surface_forcing  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402
from fesom_jax.ale import AleConfig  # noqa: E402
from fesom_jax.gm import GMConfig  # noqa: E402
from fesom_jax.ice import IceConfig  # noqa: E402
from fesom_jax.integrate import _run_steps  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.params import Params  # noqa: E402
from fesom_jax.phc_ic import phc_initial_state  # noqa: E402
from fesom_jax.step import step  # noqa: E402
from fesom_jax.tke import TkeConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from horizon_scan import level_weights, stage1_misfit  # noqa: E402  (loss forms, unit-tested)
import nn_data as _nd  # noqa: E402

MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist864"
DT = 1800.0
STEPS_PER_DAY = 48
STAB_DAYS = 10
GRAD_DAYS = 5
H_SWEEP = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
FD_BAR = 0.10                      # the plan's ≤10% AD/FD agreement bar

# Physical-bounds gate (plan M1). Wider than the model's healthy range on purpose: this asks
# "did the perturbed closure destabilise the model", not "is this a good simulation".
T_LO, T_HI = -3.0, 40.0
S_LO, S_HI = 0.0, 45.0
ETA_MAX = 10.0

ALE_CFG = AleConfig()
TKE_CFG = TkeConfig()
GM_CFG = GMConfig()
# LOCKED (parent Task 2): rheology-mode ice + frozen mixing in the BACKWARD only. The forward
# is bit-identical either way — stop_gradient is the identity forward — so the stability half
# of this gate is unaffected by these, and the gradient half is impossible without them.
ICE_CFG = IceConfig(whichEVP=1, adjoint_mode="rheology")
ADJ_CFG = AdjointConfig(freeze_mixing=True)
OUT_DEFAULT = _nd.data_path("seam_gate")


def seed_date(snap_path: str, year: int):
    """(month, day) of a ``snap_dayNNN.pkl`` / ``snap_stepNNN.pkl`` seed within ``year``."""
    name = Path(snap_path).name
    m = re.search(r"day(\d+)", name)
    if m:
        d = datetime.date(year, 1, 1) + datetime.timedelta(days=int(m.group(1)))
        return d.month, d.day
    m = re.search(r"step(\d+)", name)
    secs = int(m.group(1)) * DT if m else 0
    d = datetime.date(year, 1, 1) + datetime.timedelta(seconds=secs)
    return d.month, d.day


def peak_gb():
    pk = 0.0
    for d in jax.devices():
        try:
            pk = max(pk, ((d.memory_stats() or {}).get("peak_bytes_in_use") or 0) / 1e9)
        except Exception:
            pass
    return pk


def _free(tree):
    jax.tree.map(lambda x: x.delete() if hasattr(x, "delete") else None, tree)


def scaled_last_layer(nn, alpha):
    """``nn`` with its last layer uniformly scaled by ``alpha`` — the gate's scalar probe.

    A scalar probe rather than a random weight direction because the FD comparison needs a
    reproducible direction across two jobs, and because α=0 recovers the ``zero_last`` network
    exactly, so the probe axis passes through the deployment fallback."""
    return gm_nn.GmNN(Ws=nn.Ws[:-1] + (nn.Ws[-1] * alpha,),
                      bs=nn.bs[:-1] + (nn.bs[-1] * alpha,),
                      log_m_max=nn.log_m_max)


def build(snap: Path, year: int, n_steps: int):
    """Mesh, operator, seed state, stacked forcing for ``n_steps``, and the step body factory."""
    t0 = time.time()
    mesh = load_mesh(MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    with open(snap, "rb") as f:
        st0 = jax.device_put(pickle.load(f))
    mo, dy = seed_date(str(snap), year)
    sst0 = np.asarray(phc_initial_state(mesh, IC_DIR).T[:, 0])
    cf = surface_forcing.build_surface_forcing(mesh, year, sst_ic=sst0)
    dates = surface_forcing.dates_for_steps(year, DT, n_steps, start_month=mo, start_day=dy)
    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()} "
          f"seed={snap.name} date={year}-{mo:02d}-{dy:02d} built in {time.time()-t0:.1f}s",
          flush=True)
    return mesh, op, st0, cf, dates


def body_for(mesh, op, fs, params, remat):
    def body(carry, sf):
        nxt = step(carry, mesh, op, None, params, dt=DT, is_first_step=False, step_forcing=sf,
                   forcing_static=fs, ice_cfg=ICE_CFG, gm_cfg=GM_CFG, tke_cfg=TKE_CFG,
                   ale_cfg=ALE_CFG, adjoint_cfg=ADJ_CFG, remat_blocks=remat)
        return nxt, None
    return body


# ---------------------------------------------------------------------------
# (a) stability
# ---------------------------------------------------------------------------
def check_bounds(st, day, tag):
    """The M1 physical-bounds gate at one daily mark. Returns a record; never raises, so a
    failing run still writes its numbers (a NaN at day 7 is a result, not a crash)."""
    T, S, eta = jax.device_get((st.T, st.S, st.eta_n))
    rec = dict(day=day, tag=tag,
               finite=bool(np.isfinite(T).all() and np.isfinite(S).all() and np.isfinite(eta).all()),
               T_min=float(np.nanmin(T)), T_max=float(np.nanmax(T)),
               S_min=float(np.nanmin(S)), S_max=float(np.nanmax(S)),
               eta_absmax=float(np.nanmax(np.abs(eta))))
    rec["ok"] = bool(rec["finite"] and T_LO <= rec["T_min"] and rec["T_max"] <= T_HI
                     and S_LO <= rec["S_min"] and rec["S_max"] <= S_HI
                     and rec["eta_absmax"] < ETA_MAX)
    print(f"  [{tag}] day {day:2d}  T [{rec['T_min']:7.2f},{rec['T_max']:6.2f}]  "
          f"S [{rec['S_min']:6.2f},{rec['S_max']:6.2f}]  |eta|max {rec['eta_absmax']:5.2f}  "
          f"{'OK' if rec['ok'] else 'OUT OF BOUNDS'}", flush=True)
    return rec


def run_stability(args):
    n = STAB_DAYS * STEPS_PER_DAY
    mesh, op, st0, cf, dates = build(args.snap, args.year, n)
    fs = cf.static
    nn = gm_nn.init_gm_nn(jax.random.PRNGKey(args.seed), zero_last=False, w_scale=args.w_scale)
    out = {"mode": "stability", "seed": args.seed, "w_scale": args.w_scale,
           "days": STAB_DAYS, "snap": args.snap.name, "year": args.year, "arms": {}}

    final = {}
    for arm in ("control", "nn"):
        params = Params.defaults() if arm == "control" else \
            dataclasses.replace(Params.defaults(), gm_nn=nn)
        body = body_for(mesh, op, fs, params, remat=False)

        @jax.jit
        def one_day(state0, sfs_day, _body=body):
            return _run_steps(_body, state0, xs=sfs_day, checkpoint=False)

        marks, st = [], st0
        t0 = time.time()
        for d in range(STAB_DAYS):
            sfs = cf.stack(dates[d * STEPS_PER_DAY:(d + 1) * STEPS_PER_DAY])
            st = one_day(st, sfs)
            _free(sfs)
            marks.append(check_bounds(st, d + 1, arm))
            if not marks[-1]["ok"]:
                break
        wall = time.time() - t0
        n_done = len(marks) * STEPS_PER_DAY
        final[arm] = jax.device_get((st.T, st.S, st.eta_n))
        out["arms"][arm] = dict(marks=marks, ok=all(m["ok"] for m in marks),
                                days_completed=len(marks), wall_s=wall,
                                s_per_step=wall / max(n_done, 1), peak_gb=peak_gb())
        print(f"[{arm}] {len(marks)}/{STAB_DAYS} d  {wall:.1f} s  "
              f"{wall / max(n_done, 1):.4f} s/step  peak {peak_gb():.2f} GB", flush=True)
        if arm == "nn":
            _free(st)

    # ⚠️ "in bounds" is NOT evidence that the network did anything. On the first run of this gate
    # the two arms' daily global extrema agreed to 0.01 K on 8 of 10 days — which is what a
    # bounded kappa multiplier SHOULD look like in a global min/max (those extrema sit at the
    # warmest tropical surface point and the coldest polar water, neither of which GM controls),
    # but it is indistinguishable from a seam that never reached the step. So measure the
    # trajectory difference directly and put a number on it, against the measured
    # GPU-nondeterminism floor rather than against zero.
    if len(final) == 2 and out["arms"]["nn"]["days_completed"] == STAB_DAYS \
            and out["arms"]["control"]["days_completed"] == STAB_DAYS:
        wet = np.asarray(mesh.node_layer_mask, bool)
        area = np.asarray(mesh.areasvol) * wet
        diff = {}
        for i, name in enumerate(("T", "S")):
            d = np.abs(final["nn"][i] - final["control"][i])
            rms = float(np.sqrt((area * d ** 2).sum() / max(area.sum(), 1e-300)))
            diff[name] = dict(max=float(d[wet].max()), area_rms=rms)
        d_eta = np.abs(final["nn"][2] - final["control"][2])
        diff["eta"] = dict(max=float(d_eta.max()))
        out["trajectory_diff"] = diff
        out["nn_is_active"] = bool(diff["T"]["max"] > 0.0 and diff["S"]["max"] > 0.0)
        print(f"[active] day-{STAB_DAYS} nn-vs-control: "
              f"max|dT|={diff['T']['max']:.3e} K (area-rms {diff['T']['area_rms']:.3e}), "
              f"max|dS|={diff['S']['max']:.3e}, max|d eta|={diff['eta']['max']:.3e} m", flush=True)
        if not out["nn_is_active"]:
            print("[active] ⚠️ THE TWO ARMS ARE BIT-IDENTICAL — the perturbed network is not "
                  "reaching the step. A 'stable' verdict here would be meaningless.", flush=True)

    c, m = out["arms"]["control"], out["arms"]["nn"]
    out["stable"] = bool(m["ok"] and m["days_completed"] == STAB_DAYS
                         and out.get("nn_is_active", True))
    out["s_per_step_overhead"] = m["s_per_step"] / c["s_per_step"] - 1.0 if c["s_per_step"] else None
    out["peak_gb_delta"] = m["peak_gb"] - c["peak_gb"]
    print(f"\n[cost] s/step {c['s_per_step']:.4f} (control) -> {m['s_per_step']:.4f} (nn), "
          f"{100 * out['s_per_step_overhead']:+.1f}%; peak {c['peak_gb']:.2f} -> "
          f"{m['peak_gb']:.2f} GB", flush=True)
    return out


# ---------------------------------------------------------------------------
# (b) gradient
# ---------------------------------------------------------------------------
def run_gradient(args):
    """AD vs FD for d(5-day Stage-1 loss)/dα, α scaling the last layer.

    The target is the trajectory of the SAME network at α=1: loss(α) compares the α-run's
    5-day state against the α=1 run's, so loss(1) == 0 exactly and the minimum sits at the
    probe's own value. That is the horizon-scan harness self-check reused — a nonzero
    ``loss_at_probe`` here means the plumbing is wrong, before any gradient is believed. It is
    exact rather than approximate only because the target comes through the IDENTICAL graph
    (same remat, same checkpoint segmentation) as the loss; a differently-segmented forward
    reassociates and would leave a small nonzero residual that hides real plumbing bugs.

    ⚠️ **Big arrays are ARGUMENTS, never closures.** The first version of this function closed
    over the 240-step forcing stack, the target state, the masks and the weights, and jitted
    three separate functions (forward, loss, grad). Each lowering then embedded those arrays as
    compile-time constants — measured 4.58 GB and 3.77 GB in two of them — and the run died with
    ``RESOURCE_EXHAUSTED: 49.53GiB`` on an 80 GB card while merely evaluating a FORWARD
    (jobs 26798462/3). The child plan's M6 already states this rule for the training driver
    ("windows enter the jitted value_and_grad as ARGUMENTS — do NOT close over the batch"); it
    is written there as a compile-count argument and it is a memory argument too."""
    n = GRAD_DAYS * STEPS_PER_DAY
    mesh, op, st0, cf, dates = build(args.snap, args.year, n)
    fs = cf.static
    nn = gm_nn.init_gm_nn(jax.random.PRNGKey(args.seed), zero_last=False, w_scale=args.w_scale)
    wet3d = jnp.asarray(mesh.node_layer_mask, jnp.float64)
    area3d = jnp.asarray(np.asarray(mesh.areasvol), jnp.float64) * wet3d
    sfs = cf.stack(dates[:n])
    # segments: √N is the standard O(√N)-memory choice; --seg-mult > 1 stores fewer carries and
    # recomputes more, the knob horizon_scan exposes as $HORIZON_SEG_MULT.
    segments = max(2, int(round(args.seg_mult * math.sqrt(n))))

    def forward(alpha, state0, xs):
        p = dataclasses.replace(Params.defaults(), gm_nn=scaled_last_layer(nn, alpha))
        return _run_steps(body_for(mesh, op, fs, p, remat=True), state0, xs=xs,
                          checkpoint=True, segments=segments)

    a1 = jnp.asarray(1.0, jnp.float64)
    t0 = time.time()
    tgt = jax.jit(forward)(a1, st0, sfs)
    # Pull the three target fields to HOST and drop the device State: keeping a whole 1.8 GB
    # State alive for three arrays is most of a jitted graph's headroom.
    T_hat, S_hat, e_hat = (jnp.asarray(x) for x in jax.device_get((tgt.T, tgt.S, tgt.eta_n)))
    _free(tgt)
    del tgt
    print(f"[target] 5-day reference trajectory in {time.time()-t0:.1f} s", flush=True)

    # Per-level inverse-variance weights. A single-mark loss has no temporal variance of its
    # own, so they come from the daily marks of the SAME α=1 run — a second, cheap, non-remat
    # forward. Weights need only be fixed and sensible, not bit-reproducible, so its different
    # graph structure does not matter here (it would if it fed the target).
    p1 = dataclasses.replace(Params.defaults(), gm_nn=nn)
    day_body = body_for(mesh, op, fs, p1, remat=False)
    one_day = jax.jit(lambda s, xs: _run_steps(day_body, s, xs=xs, checkpoint=False))
    daily, st = [], st0
    for d in range(GRAD_DAYS):
        st = one_day(st, jax.tree.map(lambda x: x[d * STEPS_PER_DAY:(d + 1) * STEPS_PER_DAY],
                                      sfs))
        daily.append(jax.device_get((st.T, st.S, st.eta_n)))
    w_T, w_S, w_eta = level_weights(
        jnp.asarray(np.stack([m[0] for m in daily])),
        jnp.asarray(np.stack([m[1] for m in daily])),
        jnp.asarray(np.stack([m[2] for m in daily])), wet3d, area3d)
    w_T, w_S, w_eta = (jax.block_until_ready(x) for x in (w_T, w_S, w_eta))
    del daily
    _free(st)
    print(f"[weights] w_T surf={float(w_T[0]):.3e}  w_S surf={float(w_S[0]):.3e}  "
          f"w_eta={float(w_eta):.3e}", flush=True)

    def loss_of(alpha, state0, xs, Th, Sh, eh, wT, wS, we, wet, area):
        """The 5-day misfit. EVERY large array is a parameter — see the docstring's warning."""
        s = forward(alpha, state0, xs)
        return stage1_misfit(s.T, s.S, s.eta_n, Th, Sh, eh, wT, wS, we, wet, area)

    _big = (st0, sfs, T_hat, S_hat, e_hat, w_T, w_S, w_eta, wet3d, area3d)
    _loss_jit = jax.jit(loss_of)
    _grad_jit = jax.jit(jax.grad(loss_of, argnums=0))
    loss_jit = lambda a: _loss_jit(a, *_big)              # noqa: E731
    grad_jit = lambda a: _grad_jit(a, *_big)              # noqa: E731
    l_at_probe = float(loss_jit(a1))                      # harness self-check: must be 0
    t1 = time.time()
    ad = float(grad_jit(a1))
    bwd_s = time.time() - t1
    pk = peak_gb()
    # α=1 is the loss minimum, so dL/dα there is 0 by construction and a ratio test is
    # meaningless. Probe OFF the minimum, where both AD and FD carry signal.
    a0 = jnp.asarray(args.alpha0, jnp.float64)
    l0 = float(loss_jit(a0))
    ad0 = float(grad_jit(a0))
    rows = []
    for h in H_SWEEP:
        fd = float((loss_jit(a0 * (1 + h)) - loss_jit(a0 * (1 - h))) / (2 * h * float(a0)))
        rel = abs(ad0 - fd) / max(abs(fd), 1e-300)
        rows.append(dict(h=h, fd=fd, rel=rel))
        print(f"    h={h:.0e}  FD={fd:+.8e}  rel={rel:.3e}", flush=True)
    plateau = min(r["rel"] for r in rows)
    best = min(rows, key=lambda r: r["rel"])
    out = dict(mode="gradient", seed=args.seed, w_scale=args.w_scale, days=GRAD_DAYS,
               snap=args.snap.name, year=args.year, alpha0=float(args.alpha0),
               loss_at_probe=l_at_probe, grad_at_probe=ad, bwd_s=bwd_s, peak_gb=pk,
               loss_at_alpha0=l0, grad_at_alpha0=ad0, fd_rows=rows, plateau=plateau,
               best_h=best["h"], best_fd=best["fd"],
               sign_agree=bool(np.sign(ad0) == np.sign(best["fd"])),
               fd_pass=bool(np.isfinite(ad0) and plateau <= FD_BAR),
               segments=segments)
    print(f"\n[grad] loss(alpha=1) = {l_at_probe:.3e} (expect 0 — harness self-check)")
    print(f"[grad] alpha0={float(a0):g}: loss {l0:.6e}  AD {ad0:+.8e}  "
          f"plateau {100 * plateau:.2f}% (bar {100 * FD_BAR:.0f}%)  "
          f"sign_agree={out['sign_agree']}")
    print(f"[grad] backward {bwd_s:.1f} s  peak {pk:.2f} GB  segments={segments}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", type=Path, required=True, help="developed student State pickle")
    ap.add_argument("--mode", choices=("stability", "gradient"), required=True)
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--seed", type=int, default=17, help="PRNG seed for the perturbed network")
    ap.add_argument("--w-scale", type=float, default=0.5,
                    help="init weight scale — larger = a more aggressive multiplier field")
    ap.add_argument("--alpha0", type=float, default=0.5,
                    help="probe point for the FD check (off the loss minimum at alpha=1)")
    ap.add_argument("--seg-mult", type=float, default=1.0,
                    help="checkpoint segments = seg_mult*sqrt(n); >1 trades recompute for memory")
    ap.add_argument("--results", type=Path, default=OUT_DEFAULT / "seam_gate.jsonl")
    args = ap.parse_args()

    rec = run_stability(args) if args.mode == "stability" else run_gradient(args)
    rec["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip()
    rec["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:                    # append: two jobs, one file
        f.write(json.dumps(rec) + "\n")
    if args.mode == "stability":
        tok = "STABLE" if rec["stable"] else "UNSTABLE"
    else:
        tok = "FD_PASS" if rec["fd_pass"] else "FD_FAIL"
    print(f"\nM1_GATE_{args.mode.upper()}_{tok}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
