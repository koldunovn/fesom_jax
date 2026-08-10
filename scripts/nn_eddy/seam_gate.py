"""NN-GM child-plan M1 GATE — the ``gm_nn`` seam in the assembled CORE2 model, on a GPU.

Two claims, both required before the seam is used for anything:

**(a) STABILITY.** 10 days from a developed student state with a RANDOM-perturbed network
(``zero_last=False``, so multipliers range over the full [1/3, 3] bound) must stay physical:
no NaN, ``T ∈ [-3, 40] °C``, ``S ∈ [0, 45]``, ``|η| < 10 m`` at every DAILY mark. Reported
against a ``gm_nn=None`` control run over the same window: peak device memory and s/step, so
the cost of the seam is a measured number rather than an expectation.

**(b) GRADIENT — the routine check is ``--mode dotprod``, at the 5-day training window.**
``⟨J·v, w⟩ = ⟨v, Jᵀ·w⟩`` for ``J`` the Jacobian of the window map ``α ↦ s_fin``: forward-mode
AD on the left, this file's hand-rolled chunked reverse accumulation on the right, and they
must agree to **1%** for every ``w``. Deliberately NOT "finite and nonzero" — this project has
a recorded case of a gradient that was finite, nonzero, and structurally wrong (a dead path),
so the gate is quantitative or it is nothing.

**Why not finite differences here.** A central difference needs a step size ``h``, and ``h`` is
squeezed from both ends. Too large and the difference quotient picks up curvature as well as
slope — truncation error, growing like ``h²``, measured at 35.8% for ``h=1e-2``. Too small and
``L(α(1+h))`` and ``L(α(1−h))`` are nearly equal, so their difference drowns in the fact that
each ``L`` is itself only reproducible to 1.7e-6 (atomic scatter-adds in the GPU finite-element
assembly sum in a nondeterministic order). Normally the two bounds leave a plateau in between.
**At 5 days they do not overlap at all**: the largest attainable FD numerator is 1.4e-6 against
a 1.7e-6 noise floor, so the signal is under the noise even at the most favourable ``h``.
Lengthening the window does fix it — the trajectories diverge more, the noise floor does not
move, and the horizon ladder measured agreement 4.2% at 5 d, 5.4% at 10, 2.4% at 20, **0.50%
at 30** — which is why ``--mode gradient`` defaults its verdict to a named ``h=1e-3`` and
refuses to report an agreement number whose numerator has not cleared the loss noise it
measured in the same process. But it costs four extra full forwards, i.e. 2.5–3 h of an A100 at
30 days, against ~15–20 min for the dot-product test at 5. **Run FD once for the record; run
the dot-product test after every change.**

The dot-product test is also the stronger claim. Forward and reverse mode are independent code
paths in JAX, and the chunked reverse loop is hand-written — host-parked boundaries, per-chunk
VJPs, a cotangent carried backwards — none of which forward mode touches. FD only ever checked
the derivative of ONE scalar functional; testing several ``w`` checks the Jacobian in several
directions, so a defect confined to a state component the loss happens to be insensitive to
still shows up.

⚠️ **The ``w`` must be structured, and white noise is not.** The first run of this test
(26808456) used a normalised random cotangent and reported a 40% disagreement that was pure
noise: in a state space of ~1e8 components a random direction is very nearly orthogonal to
``ds/dalpha`` — measured alignment 2.4e-5 — so the inner product was 1.35 out of a possible
5.6e4, and the model's own 8.7e-4 forward spread put ~49 of noise against it. That is the
finite-difference disease in new clothes: a tiny difference of large numbers. The probes are
therefore the cotangents of MEANINGFUL SCALARS (global mean SST, volume-mean T and S), whose
alignment with the sensitivity direction is O(1), and every row must clear ``DOT_SNR_MIN``
before it is allowed to judge anything.

**Which seed** (also 2026-08-08). The gradient half MUST start from
``foundation_snaps/snap_step037440.pkl``. The adjoint's per-step amplification is a property
of the seed STATE: 1.0008/step from the foundation snapshot against **1.0539/step** from
``student_free_1958/snap_day150.pkl``, measured with the network entirely absent in both
(26803145 vs 26801647), which over 240 steps is 1.2x against 3e5. Seeded from the student
snapshot this gate produced a positive AD against a negative FD and did not reproduce; from
the foundation snapshot the same probe reproduces to five significant figures with the right
sign. The screen that catches this (:func:`fesom_jax.adjoint_stab.amplification`) now runs
here, on the trace the reverse sweep already produces, so a bad seed is a REJECT verdict
rather than a number somebody quotes. The STABILITY half is a pure forward run, so its
recorded numbers from the student snapshot stand and do not need redoing.

The gate must be reproduced to 3 significant figures across two passes (``--repeat 2`` in one
process, or the sbatch submitted twice; the results JSONL appends). Config and driver mirror
``horizon_scan.py`` / ``student_year.py`` — the same model, the same forcing handling, the
same remat discipline — with the plan's LOCKED adjoint switches:
``AdjointConfig(freeze_mixing=True)`` plus ``IceConfig(whichEVP=1, adjoint_mode="rheology")``.
Without them the long-window gradient is inf, which is a property of the model's ice rheology
and mixing closure, not of this seam.

Usage (A100, one GPU):
  # stability — a forward run, so the student snapshot is fine and its numbers stand
  python scripts/nn_eddy/seam_gate.py --snap /work/.../student_free_1958/snap_day150.pkl \\
      --mode stability
  # the routine gradient check, ~15-20 min
  python scripts/nn_eddy/seam_gate.py \\
      --snap /work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl \\
      --year 1960 --mode dotprod --days 5
  # the one-off finite-difference corroboration, 2.5-3 h — do not put this in a loop
  python scripts/nn_eddy/seam_gate.py --snap ... --year 1960 \\
      --mode gradient --days 30 --repeat 2 --h-sweep 1e-2,1e-3
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

from fesom_jax import adjoint_stab, gm_nn, ssh, surface_forcing  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402
from fesom_jax.ale import AleConfig  # noqa: E402
from fesom_jax.gm import GMConfig  # noqa: E402
from fesom_jax.ice import IceConfig  # noqa: E402
from fesom_jax.adjoint import scan_steps  # noqa: E402
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
# 5 — the SAME window the training driver will use. It stays 5 because the fix for the
# 5-day FD problem is to stop using finite differences, not to lengthen the window: see
# `--mode dotprod`, which has no step size and no noise floor and costs ~15 min against the
# 30-day FD run's 2.5-3 h.
GRAD_DAYS = 5
H_SWEEP = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
FD_BAR = 0.10                      # the plan's ≤10% AD/FD agreement bar
# The verdict's step size is chosen by a RULE fixed in advance, not by a number fixed in
# advance and not by `min(rel)`.
#
# `min(rel)` is not a statistic: it picks whichever h happens to flatter the AD, and it
# reported 19% and 84% agreement from the same situation on 2026-08-08. But a hardcoded h is
# not right either — `1e-3` was taken from the horizon ladder, where the 60- and 90-day signals
# are large, and at 30 days with this probe it sits UNDER the noise: measured 2026-08-09
# (26806052), the FD numerator at h=1e-3 was 4.78e-06 against a 1.09e-06 loss spread, SNR 4.4,
# while h=1e-2 gave 3.73e-05, SNR 34. Reading the verdict at 1e-3 would have called a passing
# gate a 28% failure. The giveaway that 1e-3 was noise and not truncation: disagreement GREW
# as h shrank, and truncation does the opposite.
#
# So: **the smallest h whose signal-to-noise clears FD_SNR_MIN.** Smallest, because truncation
# error falls with h; SNR-gated, because that keeps the choice independent of the AD value —
# unlike `min(rel)`, the signal-to-noise ratio does not depend on the answer being tested, so
# this cannot flatter it. If no h clears the floor, there is no verdict, and the run says so.
FD_PRIMARY_H = None
# The dot-product test's bar. Both sides are EXACT derivatives, so the only thing separating
# them is that they are two different compilations running on a GPU whose finite-element
# assembly uses atomic scatter-adds: this model reproduces to ~6e-4 on the loss and ~3e-5 on
# the gradient between identical runs. 1% therefore sits more than an order of magnitude above
# the floor and orders of magnitude below anything a real defect produces. **It is deliberately
# NOT a round-off bar** — a 1e-9 bar was set here once and no correct result could have met it.
DOT_BAR = 0.01
# ⚠️ `snr` below is a DIAGNOSTIC, not a gate, and the story it was built on did not survive
# contact with the data. It models the noise on <J.v, w> as
# (forward relative spread) x ||ds|| x ||w||, i.e. as if the perturbation lay entirely along
# w. It does not: the perturbation is a small addition to ds, and its projection onto w is
# suppressed by the SAME random-direction factor that suppresses the signal's, so the two
# largely cancel — and both sides are computed from nearly the same trajectory, so their
# errors are correlated rather than independent. Measured (26808947): this model predicted
# SNR ~0.003 for every row while the rows actually agreed to 2.6e-4, 1.4e-4 and 6.1e-5.
#
# It also fails to explain what it was built to explain. A white-noise cotangent disagreed by
# 40% on the first run (26808456) and I attributed that to cancellation at an alignment of
# 2.4e-5 — but `sst` has alignment 1.9e-6, an order of magnitude SMALLER, and agrees to 2.6e-4.
# **The white-noise discrepancy is not explained.** Use structured cotangents (the gradients of
# scalar diagnostics); do not read `random` as a verdict either way until someone finds out why.
DOT_SNR_MIN = 0.0
# How far the FD numerator must clear the loss's own run-to-run spread before an agreement
# number means anything. 10x is not conservative — at 5 days the ratio was BELOW 1.
FD_SNR_MIN = 10.0

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
    def one_step(s, sf):
        return step(s, mesh, op, None, params, dt=DT, is_first_step=False, step_forcing=sf,
                    forcing_static=fs, ice_cfg=ICE_CFG, gm_cfg=GM_CFG, tke_cfg=TKE_CFG,
                    ale_cfg=ALE_CFG, adjoint_cfg=ADJ_CFG, remat_blocks=remat)
    return one_step


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
            return scan_steps(_body, state0, sfs_day, rematerialize=False)

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
class ChunkedWindow:
    """The chunked forward/backward/tangent plumbing for one window — built ONCE and shared by
    every gradient mode, so the FD check and the dot-product check cannot end up exercising
    different code. Everything in here was learned the expensive way; the comments say how."""

    def __init__(self, args, mesh, op, st0, cf, dates, n_chunks, K):
        self.mesh, self.op, self.st0, self.cf, self.dates = mesh, op, st0, cf, dates
        self.n_chunks, self.K = n_chunks, K
        fs = cf.static
        self.nn = gm_nn.init_gm_nn(jax.random.PRNGKey(args.seed), zero_last=False,
                                   w_scale=args.w_scale)
        self.wet3d = jnp.asarray(mesh.node_layer_mask, jnp.float64)
        self.area3d = jnp.asarray(np.asarray(mesh.areasvol), jnp.float64) * self.wet3d
        # segments INSIDE one chunk — √K, the standard O(√K)-memory choice at chunk scale.
        # This is the knob's correct scope; it was never the window-length knob. Recorded in the
        # results row, not passed to anything: `scan_steps` derives √K itself and takes no
        # `segments` argument, because S + K/S is minimised at S=√K and BOTH directions cost more
        # memory — a parameter whose only settings are worse than its default is a trap, not a
        # dial. `--seg-mult` is gone with it.
        self.segments = max(2, int(round(math.sqrt(K))))
        nn = self.nn

        def run_chunk(state, sfs, alpha):
            p = dataclasses.replace(Params.defaults(), gm_nn=scaled_last_layer(nn, alpha))
            return scan_steps(body_for(mesh, op, fs, p, remat=True), state, sfs,
                              rematerialize=True)

        def run_chunk_nockpt(state, sfs, alpha):
            """Same chunk WITHOUT the checkpointing scheme — for forward mode only.

            ``jax.checkpoint`` exists to trade compute for memory in the REVERSE pass, and a
            JVP has no reverse pass and stores no tape, so under forward mode it is pure
            overhead: semantically the identity, but it doubles the nesting XLA has to chew
            through and the measured cost was minutes of constant-folding scatter-adds inside
            ``jit(jvp_chunk)/jvp()/while/body/closed_call/checkpoint/checkpoint``. Values are
            unchanged — this is a compile-time change, not a numerical one."""
            p = dataclasses.replace(Params.defaults(), gm_nn=scaled_last_layer(nn, alpha))
            return scan_steps(body_for(mesh, op, fs, p, remat=True), state, sfs,
                              rematerialize=False)

        self.run_chunk = run_chunk
        self.fwd_chunk = jax.jit(run_chunk)

        @jax.jit
        def bwd_chunk(state, sfs, alpha, cot):
            """VJP of ONE chunk → (cotangent w.r.t. the chunk's input state, dα share)."""
            _, vjp = jax.vjp(lambda s, a: run_chunk(s, sfs, a), state, alpha)
            return vjp(cot)

        @jax.jit
        def jvp_chunk(state, tan_state, sfs, alpha):
            """JVP of ONE chunk → (output state, its tangent) for a unit tangent on α.

            Forward mode stores no tape, so this costs ~2x a forward in memory and needs no
            checkpointing scheme at all — which is exactly why it is the honest independent
            check on the hand-rolled reverse sweep above, and why it runs the UNcheckpointed
            chunk (see :func:`run_chunk_nockpt`; identical values, far less to compile)."""
            return jax.jvp(lambda s, a: run_chunk_nockpt(s, sfs, a), (state, alpha),
                           (tan_state, jnp.asarray(1.0, jnp.float64)))

        self.bwd_chunk, self.jvp_chunk = bwd_chunk, jvp_chunk

    def chunk_sfs(self, c):
        """Forcing stacked PER CHUNK ON DEMAND and freed after use. Holding every chunk's stack
        on the device is ~0.5 GB per simulated day of pure idle residency — the defect that made
        adjoint_horizon's device peak grow with window length (56.9 → 43.2 GB at 30 days once
        fixed). Harmless at 5 chunks, ~15 GB of waste at 30, fatal beyond."""
        return self.cf.stack(self.dates[c * self.K:(c + 1) * self.K])

    def forward(self, alpha, keep_boundaries=False):
        """Run the window chunk by chunk. Boundaries come back on the HOST — that is what makes
        the device peak independent of window length (~1.9 GB of host per day)."""
        s = self.st0
        kept = [jax.device_get(s)] if keep_boundaries else None
        for c in range(self.n_chunks):
            # block before freeing the input: see the ORDER MATTERS note in loss_and_grad
            sfs_c = self.chunk_sfs(c)
            nxt = jax.block_until_ready(self.fwd_chunk(s, sfs_c, alpha))
            _free(sfs_c)
            if s is not self.st0:
                _free(s)
            s = nxt
            if keep_boundaries:
                kept.append(jax.device_get(s))
        return s, kept

    def forward_tangent(self, alpha, verbose=False):
        """``(s_fin, ds_fin/dα)`` by FORWARD-mode AD across the whole window.

        Chunk by chunk with both the primal and the tangent carried on the device: no tape, no
        stored boundaries, no host traffic. The tangent of the first chunk is seeded from a
        zero State (every ``State`` leaf is float64 — asserted, because a non-float leaf would
        need a ``float0`` tangent and silently break this construction)."""
        leaves = jax.tree.leaves(self.st0)
        bad = [str(x.dtype) for x in leaves if not np.issubdtype(np.dtype(x.dtype), np.floating)]
        if bad:
            raise SystemExit(f"FATAL: State carries non-float leaves {sorted(set(bad))}; the "
                             f"JVP seed below would need float0 tangents")
        s = self.st0
        ds = jax.tree.map(jnp.zeros_like, self.st0)
        for c in range(self.n_chunks):
            sfs_c = self.chunk_sfs(c)
            s_n, ds_n = self.jvp_chunk(s, ds, sfs_c, alpha)
            s_n, ds_n = jax.block_until_ready((s_n, ds_n))
            _free(sfs_c)
            if s is not self.st0:
                _free(s)
                _free(ds)
            s, ds = s_n, ds_n
            if verbose:
                print(f"    [jvp] chunk {c:3d}  |ds|={float(adjoint_stab.cotangent_norm(ds)):.6e}",
                      flush=True)
        return s, ds

    def reverse(self, alpha, cot, boundaries, verbose=False):
        """``⟨cot, ds_fin/dα⟩`` by the chunked reverse sweep, plus the boundary-norm trace.

        The SAME sweep the FD mode uses; only the seed cotangent differs. Seeding it with an
        arbitrary ``cot`` rather than dL/ds is what makes the dot-product identity a test of
        the whole Jacobian instead of one scalar functional's derivative."""
        g = jnp.asarray(0.0, jnp.float64)
        tr = [float(adjoint_stab.cotangent_norm(cot))]
        for c in reversed(range(self.n_chunks)):
            s_in = jax.device_put(boundaries[c])
            sfs_c = self.chunk_sfs(c)
            g_s, g_a = self.bwd_chunk(s_in, sfs_c, alpha, cot)
            g = g + g_a
            # ⚠️ ORDER MATTERS: s_in, sfs_c and the OLD cot are live inputs to the VJP until it
            # completes. Block on its output first, then free.
            g_s = jax.block_until_ready(g_s)
            _free(cot)
            _free(sfs_c)
            _free(s_in)
            cot = g_s
            tr.append(float(adjoint_stab.cotangent_norm(cot)))
            if verbose:
                print(f"    [bwd] chunk {c:3d}  |cot|={tr[-1]:.6e}  "
                      f"<cot, ds/da> so far={float(g):+.8e}", flush=True)
        return float(g), tr


def chunked_window(args, mesh, op, st0, cf, dates, n_chunks, K) -> ChunkedWindow:
    return ChunkedWindow(args, mesh, op, st0, cf, dates, n_chunks, K)


def run_dotprod(args):
    """**The routine gradient check: ⟨J·v, w⟩ = ⟨v, Jᵀ·w⟩.** No step size, no noise floor.

    ``J`` is the Jacobian of the whole window map ``α ↦ s_fin``. The forward side runs JAX's
    forward-mode AD (``J·v`` for the unit tangent ``v = 1``); the reverse side runs OUR
    hand-rolled chunked reverse accumulation (``Jᵀ·w``, contracted with ``v``). The two must
    agree for any ``w``.

    **Why this replaces the finite-difference check rather than supplementing it.** At the
    5-day window the training driver actually uses, FD has no usable step size: the loss moves
    1.7e-6 between two identical passes (GPU atomic scatter-adds in the FE assembly) while the
    FD numerator at the best h is 1.4e-6 — under the noise — and at larger h truncation error
    takes over (35.8 % at h=1e-2). Lengthening the window to 30 days does fix that (agreement
    0.50 %), and that run is worth having once, but it costs 2.5-3 h of an A100 because FD
    needs four extra full forwards. This costs one forward-mode sweep plus the reverse sweep
    that is being tested, ~15-20 min at 5 days, so it can be run after every change instead of
    once a campaign.

    **And it is a stronger claim.** Forward-mode and reverse-mode are independent code paths in
    JAX, and the chunked reverse loop is hand-written — boundaries parked on the host,
    per-chunk VJPs, a cotangent carried backwards — none of which forward mode touches. FD only
    ever checked the derivative of ONE scalar functional; a random ``w`` here checks the whole
    Jacobian, so a bug confined to a state component the loss happens to be insensitive to
    still shows up.

    ⚠️ **The bar is 1 %, and it is NOT a round-off bar.** Two different compilations of the
    same mathematics on a GPU with atomic scatter-adds do not agree to machine precision: this
    model reproduces to ~6e-4 on the loss and ~3e-5 on the gradient between identical runs. A
    1e-9 bar was set once here and no correct result could have met it."""
    days = int(args.days)
    n = days * STEPS_PER_DAY
    K = args.chunk_steps
    if n % K:
        raise SystemExit(f"--chunk-steps {K} must divide the {n}-step window")
    n_chunks = n // K
    if ADJ_CFG.freeze_mixing is not True:
        raise SystemExit(f"FATAL: freeze_mixing={ADJ_CFG.freeze_mixing}, must be True")
    if ICE_CFG.adjoint_mode != "rheology" or ICE_CFG.whichEVP != 1:
        raise SystemExit(f"FATAL: ice adjoint {ICE_CFG.adjoint_mode!r}, must be 'rheology'")
    print(f"[adjoint] LOCKED switches ON: freeze_mixing={ADJ_CFG.freeze_mixing}, "
          f"ice adjoint_mode={ICE_CFG.adjoint_mode!r}; "
          f"freeze_nn_features={ADJ_CFG.freeze_nn_features}", flush=True)

    mesh, op, st0, cf, dates = build(args.snap, args.year, n)
    W = chunked_window(args, mesh, op, st0, cf, dates, n_chunks, K)
    a0 = jnp.asarray(args.alpha0, jnp.float64)

    t0 = time.time()
    s_fin_f, ds = W.forward_tangent(a0)
    fwd_s = time.time() - t0
    print(f"[jvp] forward-mode sweep in {fwd_s:.1f} s  |ds/dalpha| = "
          f"{float(adjoint_stab.cotangent_norm(ds)):.6e}", flush=True)

    # The reverse side needs the window's chunk boundaries, which the tangent sweep does not
    # keep (forward mode has no tape). One extra forward, and it doubles as the reproducibility
    # sample: |s_fin| from the two sweeps must agree to the model's own ~6e-4.
    s_fin_r, bnds = W.forward(a0, keep_boundaries=True)
    host_gb = sum(a.nbytes for a in jax.tree.leaves(bnds)) / 1e9
    dsn = float(adjoint_stab.cotangent_norm(jax.tree.map(lambda a, b: a - b, s_fin_f, s_fin_r)))
    ref = float(adjoint_stab.cotangent_norm(s_fin_r))
    print(f"[fwd] boundaries on host: {host_gb:.1f} GB; the two forwards differ by "
          f"{dsn/max(ref,1e-300):.3e} relative (the model's own nondeterminism)", flush=True)

    # --- the cotangents w -----------------------------------------------------
    # Each is the gradient of a SCALAR DIAGNOSTIC of the final state, so it points somewhere
    # the sensitivity actually lives: alignment with ds/dalpha is O(1) rather than the ~1e-5 a
    # white-noise direction gets in this many dimensions. 'loss' is the one training uses, so
    # its row is directly comparable to --mode gradient's dL/dalpha; the others probe
    # directions the loss need not be sensitive to.
    wet = jnp.asarray(mesh.node_layer_mask, jnp.float64)
    area = jnp.asarray(np.asarray(mesh.areasvol), jnp.float64) * wet
    a_surf = area[:, 0]

    def _diag(name):
        if name == "sst":        # area-weighted global mean surface temperature
            return lambda s: jnp.sum(a_surf * s.T[:, 0]) / jnp.sum(a_surf)
        if name == "heat":       # volume-weighted global mean temperature
            return lambda s: jnp.sum(area * s.T) / jnp.sum(area)
        if name == "salt":       # volume-weighted global mean salinity
            return lambda s: jnp.sum(area * s.S) / jnp.sum(area)
        return None

    cots = {}
    want = [c.strip() for c in args.dot_cotangents.split(",") if c.strip()]
    for nm in want:
        f = _diag(nm)
        if f is not None:
            cots[nm] = jax.block_until_ready(jax.vjp(f, s_fin_r)[1](1.0)[0])
    if "random" in want:
        # kept only so the measurement that condemned it can be reproduced; it is noise
        keys = jax.random.split(jax.random.PRNGKey(args.seed + 101),
                                len(jax.tree.leaves(s_fin_r)))
        kt = jax.tree.unflatten(jax.tree.structure(s_fin_r), list(keys))
        w_rand = jax.tree.map(lambda x, k: jax.random.normal(k, x.shape, x.dtype), s_fin_r, kt)
        nrm = float(adjoint_stab.cotangent_norm(w_rand))
        cots["random"] = jax.tree.map(lambda x: x / nrm, w_rand)
    if "loss" in want:
        marks = bnds[1:] if K == STEPS_PER_DAY else [bnds[-1]]
        w_T, w_S, w_eta = level_weights(
            jnp.asarray(np.stack([m.T for m in marks])),
            jnp.asarray(np.stack([m.S for m in marks])),
            jnp.asarray(np.stack([m.eta_n for m in marks])), wet, area)
        s_ref, _ = W.forward(jnp.asarray(1.0, jnp.float64))
        T_hat, S_hat, e_hat = (jnp.asarray(x) for x in
                               jax.device_get((s_ref.T, s_ref.S, s_ref.eta_n)))
        _free(s_ref)

        def misfit_of(s):
            return stage1_misfit(s.T, s.S, s.eta_n, T_hat, S_hat, e_hat,
                                 w_T, w_S, w_eta, wet, area)
        _, vjp_L = jax.vjp(misfit_of, s_fin_r)
        cots["loss"] = jax.block_until_ready(vjp_L(1.0)[0])
    if not cots:
        raise SystemExit("--dot-cotangents selected nothing (use 'loss', 'random', or both)")

    # ‖J·v‖ and ‖w‖ set the largest the inner product COULD be. How far below that it lands is
    # how much cancellation the row carries, and cancellation multiplies the model's own
    # forward nondeterminism straight into the answer.
    ds_norm = float(adjoint_stab.cotangent_norm(ds))
    fwd_rel = dsn / max(ref, 1e-300)

    rows = []
    for name, w in cots.items():
        w_norm = float(adjoint_stab.cotangent_norm(w))
        lhs = float(sum(jnp.sum(a * b) for a, b in
                        zip(jax.tree.leaves(ds), jax.tree.leaves(w))))     # <J.v, w>
        t1 = time.time()
        rhs, trace = W.reverse(a0, w, bnds, verbose=(name == "loss"))      # <v, J^T.w>
        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
        # --- does this row carry signal at all? ---------------------------------------
        # ⚠️ THE SAME TRAP AS FINITE DIFFERENCES, IN NEW CLOTHES, and it caught this test on
        # its first run (26808456). A white-noise `w` over a State whose 40 leaves carry
        # wildly different physical units is very nearly ORTHOGONAL to ds/dalpha: measured
        # alignment 2.4e-5, so the inner product was 1.35 out of a possible 5.6e4 — a tiny
        # difference of huge numbers. The two forwards in that same run differed by 8.7e-4
        # relative, which puts ~49 of noise against 1.35 of signal, i.e. SNR 0.03. The 40%
        # "disagreement" it reported was noise, not a defect. So each row must prove it has
        # signal before it is allowed to judge anything.
        align = abs(lhs) / max(ds_norm * w_norm, 1e-300)
        noise = fwd_rel * ds_norm * w_norm
        snr = abs(lhs) / max(noise, 1e-300)
        usable = True          # judged on agreement; see DOT_SNR_MIN for why not on snr
        # The amplification screen's bars were calibrated on LOSS-seeded sweeps. An arbitrary
        # seed has a large component in directions the adjoint annihilates on the first chunk,
        # which reads as "growth concentrated in one chunk" and is nothing of the kind — the
        # random row measured 0.9967/step (SHRINKING) with the spike in the very first backward
        # chunk. So the screen is only meaningful on the loss row, and only that row reports it.
        amp = adjoint_stab.amplification(trace, K) if name == "loss" else None
        rows.append(dict(cotangent=name, lhs=lhs, rhs=rhs, rel=rel, bwd_s=time.time() - t1,
                         alignment=align, snr=snr, usable=usable, w_norm=w_norm,
                         amp_median=(None if amp is None else amp.median),
                         amp_log_spread=(None if amp is None else amp.log_spread),
                         amp_clean=(None if amp is None else bool(amp.clean)),
                         amp_reason=(None if amp is None else amp.reason), cot_trace=trace))
        print(f"[dot] w={name:6s}  <J.v,w> = {lhs:+.10e}   <v,J^T.w> = {rhs:+.10e}   "
              f"rel = {rel:.3e}  (bar {DOT_BAR:.0e})", flush=True)
        print(f"      alignment {align:.3e} of the maximum; the forward's own spread is "
              f"{fwd_rel:.1e} (diagnostics — the verdict is the agreement above)", flush=True)
        if amp is not None:
            print(f"      seed amplification {amp.median:.4f}/step, log-spread "
                  f"{amp.log_spread:.3f} -> "
                  f"{'USABLE' if amp.clean else 'REJECT: ' + amp.reason}", flush=True)

    judged = [r for r in rows if r["usable"]]
    worst = max(r["rel"] for r in judged)
    all_clean = all(r["amp_clean"] for r in rows if r["amp_clean"] is not None)
    out = dict(mode="dotprod", seed=args.seed, w_scale=args.w_scale, days=days,
               snap=args.snap.name, year=args.year, alpha0=float(a0),
               chunk_steps=K, n_chunks=n_chunks, segments=W.segments, host_gb=host_gb,
               fwd_tangent_s=fwd_s, peak_gb=peak_gb(), rows=rows,
               forward_repro_rel=fwd_rel, ds_norm=ds_norm,
               n_rows_judged=len(judged), n_rows_below_snr=len(rows) - len(judged),
               worst_rel=worst, bar=DOT_BAR, snr_min=DOT_SNR_MIN,
               dot_pass=bool(worst <= DOT_BAR), amp_clean=all_clean,
               gate_pass=bool(worst <= DOT_BAR and all_clean))
    skipped = [r["cotangent"] for r in rows if not r["usable"]]
    print(f"\n[dot] worst disagreement {100*worst:.4f}% over {len(judged)} row(s) that carry "
          f"signal, against a {100*DOT_BAR:.0f}% bar; amplification screen "
          f"{'PASS' if all_clean else 'REJECT'}  -> gate_pass={out['gate_pass']}")
    if skipped:
        print(f"[dot] NOT JUDGED (below the {DOT_SNR_MIN:.0f}x signal-to-noise floor, so their "
              f"numbers mean nothing either way): {', '.join(skipped)}")
    print(f"[dot] peak device {out['peak_gb']:.2f} GB  host {host_gb:.1f} GB  "
          f"{n_chunks} x {K} steps", flush=True)
    return out


def run_gradient(args):
    """AD vs FD for d(5-day Stage-1 loss)/dα, α scaling the last layer, by CHUNKED REVERSE
    ACCUMULATION.

    ⚠️ **This is the EXPENSIVE, one-off check; `--mode dotprod` is the routine one.** FD needs
    two extra full forwards per step size and has no usable step size at 5 days at all (the
    numerator sits under the loss's own run-to-run spread), so a meaningful FD verdict costs a
    30-day window and 2.5-3 h of an A100. Run it once for the record; use the dot-product test
    after every change.

    The target is the trajectory of the SAME network at α=1: loss(α) compares the α-run's
    5-day state against the α=1 run's, so loss(1) == 0 exactly and the minimum sits at the
    probe's own value. That is the horizon-scan harness self-check reused — a nonzero
    ``loss_at_probe`` here means the plumbing is wrong, before any gradient is believed. It is
    exact rather than approximate only because the target comes through the IDENTICAL graph
    (same chunking, same remat) as the loss; a differently-structured forward reassociates and
    would leave a small nonzero residual that hides real plumbing bugs.

    ⚠️ **Why this is chunked and not one ``jax.grad`` over the window.** A monolithic
    whole-window reverse pass does not fit an 80 GB A100 at 5 days FOR ANY ``--seg-mult``, and
    three jobs were spent discovering that the hard way (26798462/3 at 49.53 GiB, 26798871 at
    69.24 GiB). ``_run_steps(segments=S)`` stores ``S + N/S`` carries, minimised at ``S=√N``;
    a CORE2 ``State`` is **1.93 GB**, so N=240 costs ≥ 31 × 1.93 ≈ 60 GB of carries plus ~18 GB
    of per-step reverse working set ≈ 78 GB against ~78 GB usable. Raising ``segments`` adds
    outer carries, lowering it adds inner ones — √N is already the floor, there is nothing to
    tune. Chunked reverse accumulation instead caps the device at ONE chunk regardless of
    window length: measured 43.1 GB at 1 day and 56.9 GB at 30 days in
    ``adjoint_horizon.py``, whose ``fwd_chunk``/``bwd_chunk`` structure this reuses. Exact —
    coarse-grained manual checkpointing, not an approximation. See the parent plan's
    Technical Details ("Adjoint memory") and ``docs/ADJOINT_HORIZON.md``.

    ⚠️ **Big arrays are still ARGUMENTS, never closures**, inside each jitted chunk: the first
    version of this function closed over the forcing stack, the target state, the masks and the
    weights, and each lowering embedded them as compile-time constants (4.58 GB and 3.77 GB
    measured). That fix is kept — it is worth ~11 GB and it is the child plan's M6 rule."""
    days = int(args.days)
    n = days * STEPS_PER_DAY
    K = args.chunk_steps
    if n % K:
        raise SystemExit(f"--chunk-steps {K} must divide the {n}-step window")
    n_chunks = n // K
    # The two LOCKED backward switches (parent Task 2, GATE HORIZON_OK). Without them the
    # 30-day gradient is +1.64e+154 with the cotangent already inf at the start of the
    # backward — they are worth 154 orders of magnitude, and a gradient reported without
    # them is not a gradient. Asserted rather than assumed, and echoed so every log carries
    # the proof; `stop_gradient` is the identity forward, so the trajectory is unaffected.
    if ADJ_CFG.freeze_mixing is not True:
        raise SystemExit(f"FATAL: freeze_mixing={ADJ_CFG.freeze_mixing}, must be True")
    if ICE_CFG.adjoint_mode != "rheology" or ICE_CFG.whichEVP != 1:
        raise SystemExit(f"FATAL: ice adjoint {ICE_CFG.adjoint_mode!r}/whichEVP="
                         f"{ICE_CFG.whichEVP}, must be 'rheology'/1")
    print(f"[adjoint] LOCKED switches ON: freeze_mixing={ADJ_CFG.freeze_mixing}, "
          f"ice adjoint_mode={ICE_CFG.adjoint_mode!r} (whichEVP={ICE_CFG.whichEVP}); "
          f"freeze_nn_features={ADJ_CFG.freeze_nn_features}", flush=True)
    mesh, op, st0, cf, dates = build(args.snap, args.year, n)
    W = chunked_window(args, mesh, op, st0, cf, dates, n_chunks, K)
    nn, chunk_sfs, forward = W.nn, W.chunk_sfs, W.forward
    fwd_chunk, bwd_chunk, segments = W.fwd_chunk, W.bwd_chunk, W.segments
    wet3d, area3d = W.wet3d, W.area3d

    a1 = jnp.asarray(1.0, jnp.float64)
    t0 = time.time()
    s_ref, boundaries = forward(a1, keep_boundaries=True)
    T_hat, S_hat, e_hat = (jnp.asarray(x) for x in jax.device_get((s_ref.T, s_ref.S,
                                                                  s_ref.eta_n)))
    _free(s_ref)
    del s_ref
    host_gb = sum(a.nbytes for a in jax.tree.leaves(boundaries)) / 1e9
    print(f"[target] {days}-day reference trajectory in {time.time()-t0:.1f} s "
          f"({n_chunks} x {K}-step chunks; boundaries on host: {host_gb:.1f} GB)", flush=True)

    # Per-level inverse-variance weights. A single-mark loss has no temporal variance of its
    # own, so they come from the daily marks of the SAME α=1 run — which, at K=48, ARE the
    # chunk boundaries we just saved. The previous version ran a second 5-day forward for
    # these; chunking makes it free, and it now shares the target's exact graph.
    marks = boundaries[1:] if K == STEPS_PER_DAY else None
    if marks is None:                       # non-daily chunking: fall back to the end mark
        marks = [boundaries[-1]]
    w_T, w_S, w_eta = level_weights(
        jnp.asarray(np.stack([m.T for m in marks])),
        jnp.asarray(np.stack([m.S for m in marks])),
        jnp.asarray(np.stack([m.eta_n for m in marks])), wet3d, area3d)
    w_T, w_S, w_eta = (jax.block_until_ready(x) for x in (w_T, w_S, w_eta))
    print(f"[weights] w_T surf={float(w_T[0]):.3e}  w_S surf={float(w_S[0]):.3e}  "
          f"w_eta={float(w_eta):.3e}", flush=True)
    del marks, boundaries          # ~12 GB of host-side States, not needed past this point

    def misfit_of(s):
        return stage1_misfit(s.T, s.S, s.eta_n, T_hat, S_hat, e_hat,
                             w_T, w_S, w_eta, wet3d, area3d)

    def loss_only(alpha):
        s, _ = forward(alpha)
        v = float(misfit_of(s))
        _free(s)
        return v

    def loss_and_grad(alpha, verbose=False):
        """One forward (boundaries to host) + one chunked reverse sweep. Exact dL/dα."""
        s_fin, bnds = forward(alpha, keep_boundaries=True)
        l = float(misfit_of(s_fin))
        # seed the sweep with dL/d(final state), then drop the final State before the
        # backward allocates: 1.93 GB of headroom for free.
        _, vjp_L = jax.vjp(misfit_of, s_fin)
        cot = jax.block_until_ready(vjp_L(1.0)[0])   # s_fin is live until this completes
        _free(s_fin)
        del s_fin
        t1 = time.time()
        g = jnp.asarray(0.0, jnp.float64)
        norms = [float(adjoint_stab.cotangent_norm(cot))]      # seed, before any chunk
        field_norms = []
        for c in reversed(range(n_chunks)):
            s_in = jax.device_put(bnds[c])
            sfs_c = chunk_sfs(c)
            g_s, g_a = bwd_chunk(s_in, sfs_c, alpha, cot)
            g = g + g_a
            # ORDER: JAX dispatch is ASYNC, so `s_in` and the OLD `cot` are still inputs to an
            # in-flight VJP when the frees below run. PJRT is believed to defer the actual
            # deallocation until pending uses complete, so this may be safe either way — but
            # it costs nothing here (the loop is serialized anyway) and it removes the
            # question. `adjoint_horizon.py` blocks first; these two scripts must not differ.
            g_s = jax.block_until_ready(g_s)
            _free(cot)
            _free(sfs_c)
            _free(s_in)
            del s_in
            cot = g_s
            # |cotangent| per chunk — the diagnostic that separates genuine backward
            # amplification from a broken chunk. With freeze_mixing + rheology the measured
            # rate is 1.0008/step => ~1.04 per 48-step chunk, so a chunk that multiplies the
            # norm by orders of magnitude is a defect, not the Lyapunov exponent. Job
            # 26800197 showed chunk 0 contributing 60x the rest of the window and flipping
            # the sign, with no norm trace to say which it was.
            cot_norm = float(adjoint_stab.cotangent_norm(cot))
            # PER-FIELD norms. A single scalar says "chunk 0 explodes"; it cannot say WHICH
            # state variable carries it, and that is the whole question. The cotangent is a
            # State pytree, so the breakdown is free — and a field that is flat for four
            # chunks then jumps four orders in one names the term to look at, instead of
            # leaving it to be guessed.
            fn = {f.name: float(jnp.linalg.norm(getattr(cot, f.name)))
                  for f in dataclasses.fields(cot)
                  if hasattr(getattr(cot, f.name), "shape")}
            field_norms.append(fn)
            norms.append(cot_norm)
            if verbose:
                print(f"    [bwd] chunk {c:2d}  t=+{c * K * DT / 86400:4.2f} d  "
                      f"|cot|={cot_norm:.6e}  dL/dalpha so far={float(g):+.8e}", flush=True)
        _free(cot)
        if verbose and field_norms:
            # the five fields that GREW most across the final chunk — the culprit, by name
            prev, last = (field_norms[-2] if len(field_norms) > 1 else {}), field_norms[-1]
            grow = sorted(((last[k] / max(prev.get(k, 0.0), 1e-300), k) for k in last),
                          reverse=True)[:5]
            print("    [bwd] biggest per-field GROWTH across the last chunk: " +
                  ", ".join(f"{k} x{r:.3e}" for r, k in grow), flush=True)
            print("    [bwd] largest per-field cotangent norms after the last chunk: " +
                  ", ".join(f"{k}={last[k]:.3e}"
                            for k in sorted(last, key=last.get, reverse=True)[:5]), flush=True)
        return l, float(g), time.time() - t1, norms

    l_at_probe, ad, bwd_s, _ = loss_and_grad(a1)     # harness self-check: loss must be 0
    pk = peak_gb()
    # α=1 is the loss minimum, so dL/dα there is 0 by construction and a ratio test is
    # meaningless. Probe OFF the minimum, where both AD and FD carry signal.
    a0 = jnp.asarray(args.alpha0, jnp.float64)
    l0, ad0, _, cot_trace = loss_and_grad(a0, verbose=True)

    # --- is the gradient a function of its inputs? ------------------------------------
    # Identical passes, SAME process, same compiled executables, same inputs. Any spread is
    # the model's own GPU nondeterminism (atomic scatter-adds in the FE assembly) amplified
    # by the backward. docs/ADJOINT_HORIZON.md 1.2 measured 21x-33x spread WITH SIGN FLIPS
    # for a differentiated state-dependent closure, and 6 significant figures once it was
    # frozen — that is the contrast this reproduces for the learned closure. A verdict read
    # off ONE sample of a quantity that moves by orders of magnitude is not a measurement.
    rep_loss, rep_grad = [l0], [ad0]
    for r in range(1, max(1, args.repeat)):
        lr, gr, _, _ = loss_and_grad(a0)
        rep_loss.append(lr)
        rep_grad.append(gr)
        print(f"[repeat {r}] loss={lr:.9e}  dL/dalpha={gr:+.9e}", flush=True)
    grad_spread = (max(abs(g) for g in rep_grad) / max(min(abs(g) for g in rep_grad), 1e-300)
                   if len(rep_grad) > 1 else 1.0)
    loss_spread = ((max(rep_loss) - min(rep_loss)) / max(abs(np.mean(rep_loss)), 1e-300)
                   if len(rep_loss) > 1 else 0.0)
    sign_stable = len({np.sign(g) for g in rep_grad}) == 1
    if len(rep_grad) > 1:
        print(f"[repeat] {len(rep_grad)} passes: dL/dalpha spread {grad_spread:.4g}x, "
              f"sign_stable={sign_stable}, loss spread {loss_spread:.3e} relative", flush=True)

    # The absolute size of the loss's own run-to-run wobble, in the same units as the FD
    # numerator it is about to be compared against. With one pass there is no measurement of
    # it, so the SNR test below cannot run and says so rather than passing by default.
    loss_noise = (float(max(rep_loss) - min(rep_loss)) if len(rep_loss) > 1 else float("nan"))

    rows = []
    hs = tuple(float(x) for x in args.h_sweep.split(',') if x.strip()) or H_SWEEP
    for h in hs:
        lp, lm = loss_only(a0 * (1 + h)), loss_only(a0 * (1 - h))
        num = float(lp - lm)                     # the FD NUMERATOR — what must clear the noise
        fd = num / (2 * h * float(a0))
        rel = abs(ad0 - fd) / max(abs(fd), 1e-300)
        snr = abs(num) / loss_noise if loss_noise > 0 else float("nan")
        rows.append(dict(h=h, fd=fd, rel=rel, numerator=num, snr=snr))
        print(f"    h={h:.0e}  FD={fd:+.8e}  rel={rel:.3e}  "
              f"numerator={num:+.3e}  SNR={snr:.1f}x", flush=True)

    # --- the verdict, at the smallest h that clears the noise floor --------------------
    with_signal = [r for r in rows if np.isfinite(r["snr"]) and r["snr"] >= FD_SNR_MIN]
    if with_signal:
        prim = min(with_signal, key=lambda r: r["h"])
    else:
        prim = max(rows, key=lambda r: r["snr"] if np.isfinite(r["snr"]) else -1.0)
    fd_meaningful = bool(np.isfinite(prim["snr"]) and prim["snr"] >= FD_SNR_MIN)
    fd_agrees = bool(np.isfinite(ad0) and prim["rel"] <= FD_BAR
                     and np.sign(ad0) == np.sign(prim["fd"]))
    # `plateau` is kept ONLY so old records stay comparable. It is not the verdict: taking the
    # minimum over a sweep picks whichever h flatters the AD.
    plateau = min(r["rel"] for r in rows)
    best = min(rows, key=lambda r: r["rel"])

    # --- does this seed have a usable adjoint at all? ----------------------------------
    # Same screen M6 runs on every training window, from the library, on the trace the
    # reverse sweep already produced. A blown seed is why this gate failed for a day.
    amp = adjoint_stab.amplification(cot_trace, K)
    print(f"[screen] seed amplification: median {amp.median:.4f}/step, worst 3-chunk "
          f"{amp.sustained:.4f} (chunk {amp.worst_chunk}), single chunk {amp.max:.4f}, "
          f"log-spread {amp.log_spread:.3f} -> "
          f"{'USABLE' if amp.clean else 'REJECT: ' + amp.reason}", flush=True)

    out = dict(mode="gradient", seed=args.seed, w_scale=args.w_scale, days=days,
               snap=args.snap.name, year=args.year, alpha0=float(args.alpha0),
               loss_at_probe=l_at_probe, grad_at_probe=ad, bwd_s=bwd_s, peak_gb=pk,
               host_gb=host_gb, chunk_steps=K, n_chunks=n_chunks,
               loss_at_alpha0=l0, grad_at_alpha0=ad0, fd_rows=rows, plateau=plateau,
               best_h=best["h"], best_fd=best["fd"],
               primary_h=prim["h"], primary_fd=prim["fd"], primary_rel=prim["rel"],
               primary_snr=prim["snr"], loss_noise=loss_noise,
               fd_meaningful=fd_meaningful, fd_agrees=fd_agrees,
               sign_agree=bool(np.sign(ad0) == np.sign(prim["fd"])),
               fd_pass=bool(fd_meaningful and fd_agrees),
               amp_median=amp.median, amp_max=amp.max, amp_log_spread=amp.log_spread,
               amp_worst_chunk=amp.worst_chunk, amp_clean=bool(amp.clean),
               amp_reason=amp.reason,
               segments=segments, cot_trace=cot_trace,
               freeze_nn_features=bool(args.freeze_nn_features), repeat=int(args.repeat),
               rep_grad=rep_grad, rep_loss=rep_loss, grad_spread=grad_spread,
               loss_spread=loss_spread, sign_stable=bool(sign_stable),
               # per-chunk cotangent growth: with the LOCKED switches the measured rate is
               # 1.0008/step => ~1.04 per 48-step chunk. Orders of magnitude in one chunk is
               # a defect, not the Lyapunov exponent.
               cot_growth=[cot_trace[i + 1] / cot_trace[i] if cot_trace[i] > 0 else None
                           for i in range(len(cot_trace) - 1)])
    out["gate_pass"] = bool(out["fd_pass"] and amp.clean and sign_stable
                            and (grad_spread < 1.01 if len(rep_grad) > 1 else False))
    print(f"\n[grad] loss(alpha=1) = {l_at_probe:.3e} (expect 0 — harness self-check)")
    print(f"[grad] alpha0={float(a0):g}: loss {l0:.6e}  AD {ad0:+.8e}")
    if not fd_meaningful:
        print(f"[grad] ⚠️ FD IS NOT A MEASUREMENT at h={prim['h']:.0e}: numerator "
              f"{prim['numerator']:+.3e} against a loss that itself moves {loss_noise:.3e} "
              f"between identical passes (SNR {prim['snr']:.1f}x, need {FD_SNR_MIN:.0f}x). "
              f"The agreement number below is noise. Lengthen the window — the AD signal "
              f"grows with it and this floor does not.", flush=True)
    print(f"[grad] VERDICT at h={prim['h']:.0e} — the SMALLEST step size whose signal cleared "
          f"the noise ({len(with_signal)} of {len(rows)} did): rel {100 * prim['rel']:.2f}% "
          f"(bar {100 * FD_BAR:.0f}%)  sign_agree={out['sign_agree']}  "
          f"SNR {prim['snr']:.1f}x  -> gate_pass={out['gate_pass']}")
    print(f"[grad]   (for comparison only, NOT the verdict: min over the sweep is "
          f"{100 * plateau:.2f}% at h={best['h']:.0e})")
    print(f"[grad] backward {bwd_s:.1f} s  peak device {pk:.2f} GB  host {host_gb:.1f} GB  "
          f"{n_chunks} x {K} steps  segments/chunk={segments}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", type=Path, required=True, help="developed student State pickle")
    ap.add_argument("--mode", choices=("stability", "gradient", "dotprod"), required=True,
                    help="dotprod is the ROUTINE gradient check (<J.v,w> = <v,J^T.w>): no "
                         "step size, no noise floor, ~15-20 min at 5 days. gradient is the "
                         "one-off finite-difference check and needs a 30-day window and "
                         "2.5-3 h to say anything at all.")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--seed", type=int, default=17, help="PRNG seed for the perturbed network")
    ap.add_argument("--w-scale", type=float, default=0.5,
                    help="init weight scale — larger = a more aggressive multiplier field")
    ap.add_argument("--alpha0", type=float, default=0.5,
                    help="probe point for the FD check (off the loss minimum at alpha=1)")
    ap.add_argument("--days", type=int, default=GRAD_DAYS,
                    help="gradient window in days. Default 5 — the window the training driver "
                         "uses, and --mode dotprod works there because it has no step size. "
                         "--mode gradient needs 30 to say anything: its FD signal grows with "
                         "window length while the noise floor does not (agreement 4.2%% at 5 d, "
                         "5.4%% at 10, 2.4%% at 20, 0.50%% at 30).")
    ap.add_argument("--dot-cotangents", default="loss,sst,heat",
                    help="which cotangents w to test the identity with, comma-separated. "
                         "'loss' is the one training uses, so its row is comparable with "
                         "--mode gradient; 'sst'/'heat'/'salt' are the gradients of global "
                         "mean SST / volume-mean T / volume-mean S, which probe directions "
                         "the loss need not be sensitive to. 'random' is available but is "
                         "NOISE — a white-noise direction is ~orthogonal to ds/dalpha in this "
                         "many dimensions (measured alignment 2.4e-5) and its row never "
                         "clears the signal-to-noise floor. Each row costs one reverse sweep; "
                         "the forward-mode sweep is shared between them.")
    ap.add_argument("--h-sweep", default="",
                    help="comma-separated FD step sizes; empty = the built-in sweep. At long "
                         "windows only the largest one or two carry signal — the rest cost a "
                         "full forward each and return noise.")
    ap.add_argument("--freeze-nn-features", action="store_true",
                    help="AdjointConfig.freeze_nn_features: treat the network's INPUTS as "
                         "constants in the BACKWARD (forward bit-identical). Cuts the state "
                         "feedback that amplifies GPU nondeterminism; KEEPS dL/d(weights).")
    ap.add_argument("--repeat", type=int, default=1,
                    help="identical gradient passes in ONE process — the reproducibility "
                         "measurement of docs/ADJOINT_HORIZON.md 1.2. Use 2 or more.")
    ap.add_argument("--chunk-steps", type=int, default=STEPS_PER_DAY,
                    help="reverse-accumulation chunk length in steps (default 48 = 1 day). "
                         "Device peak is ONE chunk regardless of window length; host cost is "
                         "~1.9 GB per saved boundary. Smaller = lower device peak, more host.")
    ap.add_argument("--results", type=Path, default=OUT_DEFAULT / "seam_gate.jsonl")
    args = ap.parse_args()

    global ADJ_CFG
    ADJ_CFG = AdjointConfig(freeze_mixing=True,
                            freeze_nn_features=bool(args.freeze_nn_features))
    rec = {"stability": run_stability, "gradient": run_gradient,
           "dotprod": run_dotprod}[args.mode](args)
    rec["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip()
    rec["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:                    # append: two jobs, one file
        f.write(json.dumps(rec) + "\n")
    if args.mode == "stability":
        tok = "STABLE" if rec["stable"] else "UNSTABLE"
    elif args.mode == "dotprod":
        tok = "DOT_PASS" if rec["gate_pass"] else "DOT_FAIL"
    else:
        tok = "FD_PASS" if rec["fd_pass"] else "FD_FAIL"
    print(f"\nM1_GATE_{args.mode.upper()}_{tok}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
