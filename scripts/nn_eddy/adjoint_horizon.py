"""NN-GM Task 2 follow-on — WHY the adjoint blows up, and what buys horizon back.

`horizon_scan.py` established the fact: gradient clean at a 1-day window, blown by 2 days,
~5 orders of magnitude of amplification per extra day. This script asks the two questions
that fact leaves open.

**1. Is it chaos, or is it a term?**  Five orders per day is ~1.3x per 1800 s step, an
e-folding time of ~80 minutes. Nothing in a 1-degree ocean grows that fast: resolved
baroclinic instability e-folds in days, and the static instabilities that *are* fast are
removed by the model's own convective adjustment every step. So the working hypothesis is
that this is not the ocean's Lyapunov exponent but a stiff parameterization feedback in the
tangent-linear model — the failure mode MITgcm/ECCO hit and documented. `--freeze` tests
that hypothesis directly: cut one term's state-dependence out of the backward (the forward
stays bit-identical, `stop_gradient` being the identity) and see whether the growth goes
away. If one knob collapses |dL/dm| from 1e7 back toward O(1), that term is the amplifier.
If no knob helps and the growth is smooth in time, it really is chaos and only the
stabilizers below (or a shorter window) can help.

**2. How far can the backward actually reach?**  The old scan OOMed at 7 and 10 days
because the whole window's reverse pass had to fit on one card at once. It doesn't have to.
This script accumulates the reverse pass **chunk by chunk**: the forward stores the state at
every chunk boundary on the HOST, then the backward walks chunks in reverse, running one
`jax.vjp` per chunk and carrying only the cotangent on the device. Peak device memory is
that of a single chunk regardless of window length, so a 10-day window costs the same
memory as a 1-day window and 10x the time. The gradient is exact — this is manual
checkpointing at chunk granularity, not an approximation.

The chunked walk also hands us the diagnostic that settles question 1 for free: the norm of
the cotangent at every chunk boundary. **Smooth exponential growth in that trace means
chaos; growth concentrated in a few chunks means a switch is firing.** The trace is printed
and written to the results file.

Stabilizers (`fesom_jax.adjoint_stab`, all off by default): `--gamma` discounts the
cotangent per step, `--filter-factor` applies the exact transpose of the model's own
mesh-Laplacian filter (adjoint diffusion — the ensemble-mean-adjoint closure), `--clip`
caps the per-step cotangent norm (nonlinear, diagnostic only).

Everything is measured on the same perfect-model GM-kappa twin and the same Stage-1 loss as
`horizon_scan.py` — the pure loss pieces and the cached 10-day truth are imported from it,
so numbers here are directly comparable to `horizon_results.jsonl`.

Usage (A100):
  python scripts/nn_eddy/adjoint_horizon.py --snap $WORK/port_jax/foundation_snaps/snap_step037440.pkl \\
      --season winterNH --n-days 3 --freeze mix
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

# ⚠️ Import path, and it is not boilerplate. `python scripts/nn_eddy/foo.py` puts the
# SCRIPT's directory on sys.path, never the cwd — so from this worktree `import fesom_jax`
# resolves through the editable install to the MAIN checkout (~/port_jax), and any
# worktree-only library change is silently absent while the run still looks correct. Put the
# worktree root first and then verify we actually got it, because the failure is invisible:
# the model runs, the numbers come out, they are just the wrong code's numbers.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fesom_jax  # noqa: E402  (enables x64; must follow the sys.path fix above)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

if Path(fesom_jax.__file__).resolve().parents[1] != _ROOT:
    raise SystemExit(f"fesom_jax resolved to {fesom_jax.__file__}, not the checkout this "
                     f"script lives in ({_ROOT}). Refusing to report results attributed to "
                     f"code that did not run.")

from fesom_jax import (adjoint_stab, eos, gm, gm_redi, momentum, pgf, pp, ssh,  # noqa: E402
                       surface_forcing, tke, tracer_adv)
from fesom_jax.calibrate import build_params
from fesom_jax.integrate import _run_steps
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.step import step
from fesom_jax import ice_mevp  # noqa: E402
from fesom_jax.ice import IceConfig  # noqa: E402
from fesom_jax.kpp import KppConfig  # noqa: E402
from horizon_scan import (ALE_CFG, BLOW, DT, GM_CFG, ICE_FROZEN, IC_DIR, MESH_DIR,
                          NDAYS_TRUTH, STEPS_PER_DAY, TKE_CFG, level_weights, peak_gb,
                          seed_date, stage1_misfit)
import nn_data as _nd  # noqa: E402  (the one DATA_ROOT constant; ab0995 /work is at 98%)
import adjoint_ckpt as _ckpt  # noqa: E402  (resume a reverse sweep across jobs)

# ---------------------------------------------------------------------------
# the freeze knobs: which term's state-dependence to cut out of the backward
#
# Every one of these is applied by MONKEYPATCHING the module attribute the step calls, not
# by editing library code. `step.py` resolves `tke.mixing_tke`, `pp.mo_convect`, ... at call
# time, so a patched attribute is picked up with no change to any shipped file and no risk
# of a stray knob surviving into a production run. Each wrapper bumps a counter, and
# `patched()` refuses to return a result if a requested knob never fired — a monkeypatch
# that silently misses would report "not the culprit" for a term never actually tested.

FREEZE_HELP = {
    "mix":     "vertical mixing coefficients Kv, Av — the documented ECCO practice: use the "
               "coefficients the closure computes but omit their dependence on the ocean state",
    "tkestate": "the prognostic TKE carried across steps (cuts the closure's own memory loop)",
    "convect": "ONLY the convective-adjustment bump (the N^2<0 switch), leaving the closure live",
    "visc":    "Av where it enters the implicit vertical momentum viscosity",
    "slope":   "N^2 inside the GM block — the 1/N^2 neutral-slope amplifier; k_gm and T,S stay live",
    "k33":     "the GM/Redi K33 isoneutral vertical augmentation",
    "bv":      "N^2 everywhere at its source (mixing + convection + GM slope in one cut) — the "
               "big hammer: if this tames it, the amplifier is the stratification feedback",
    "fct":     "the limited antidiffusive tracer flux, so the BACKWARD sees pure monotone upwind "
               "advection while the forward keeps full FCT. This is the NWP incremental-4D-Var "
               "practice of running a simplified, more diffusive tangent-linear/adjoint model "
               "than the nonlinear forward. NOT the same as the FESOM_FCT_STOPGRAD_AE hatch, "
               "which froze only the limiter factor `ae` and left the antidiffusive flux live — "
               "that removed the limiter's own STABILIZING negative feedback (as the tracer "
               "field steepens, ae falls) and measured harmful, which is exactly what dropping a "
               "stabilizing term should do. This knob drops the whole correction instead",
    "icevisc": "the sea-ice rheology's viscosity dependence on the strain rate — the Toyoda "
               "et al. (2019) approximation used by Lyu et al. (2023) to build a usable adjoint "
               "of viscous-plastic rheology. Only meaningful with --ice exact. The mEVP "
               "viscosity is P/(Delta+delta_min) with delta_min=1e-11, so its strain derivative "
               "is ~1e22 in rigid ice and ~1e16 in ordinary deforming ice; float64 overflows "
               "after 14-19 of the 120 subcycles in ONE ice step, which is why the exact ice "
               "adjoint returns NaN immediately rather than degrading",
    "pgf":     "the baroclinic pressure-gradient force — cuts the T,S -> momentum loop entirely. "
               "Diagnostic ONLY; the resulting gradient is not usable for training",
}
_FIRED: dict[str, int] = {}


def _bump(name):
    _FIRED[name] = _FIRED.get(name, 0) + 1


@contextlib.contextmanager
def patched(knobs):
    """Install the requested freeze knobs for the duration of the block."""
    unknown = set(knobs) - set(FREEZE_HELP)
    if unknown:
        raise SystemExit(f"unknown --freeze target(s): {sorted(unknown)}; "
                         f"choose from {sorted(FREEZE_HELP)}")
    # `check_fired` claims "fired during the forward", so the counters have to be per-block.
    # They are module-level, so without this a knob that fired in an EARLIER block still reads as
    # fired — the guard would pass on a run where the patch never actually ran. One process, one
    # rung today, so it has never bitten; it is exactly the shape that bites later.
    for k in knobs:
        _FIRED.pop(k, None)
    saved = []

    def install(mod, attr, make):
        orig = getattr(mod, attr)
        saved.append((mod, attr, orig))
        setattr(mod, attr, make(orig))

    if "mix" in knobs or "tkestate" in knobs:
        def mk_tke(orig):
            def w(*a, **k):
                Kv, Av, uvnode, tke_new = orig(*a, **k)
                if "mix" in knobs:
                    Kv, Av = adjoint_stab.freeze(Kv), adjoint_stab.freeze(Av)
                    _bump("mix")
                if "tkestate" in knobs:
                    tke_new = adjoint_stab.freeze(tke_new)
                    _bump("tkestate")
                return Kv, Av, uvnode, tke_new
            return w
        install(tke, "mixing_tke", mk_tke)

    if "convect" in knobs:
        # ⚠️ **, NOT a fixed signature.** This wrapper was written when `mo_convect` took exactly
        # (mesh, Kv, Av, bvfreq). The library has since grown its OWN convection freeze —
        # `mo_convect(..., freeze_increment=)`, driven by `AdjointConfig.freeze_convection` — and
        # tke.py/kpp.py/pp.py now all pass that keyword. The wrapper rejected it and every
        # `--freeze convect` run died in 27 s with a TypeError (job 26822509), a knob that had
        # been silently unusable since the library gained the flag.
        #
        # This is the "two implementations of freeze mixing" trap in its second instance, and the
        # standing decision is to RETIRE the monkeypatches in favour of AdjointConfig rather than
        # keep both. Not done here on purpose: every recorded freeze number in
        # adjoint_horizon_results.jsonl was produced through the monkeypatch, so switching
        # mechanisms mid-campaign would make tonight's bisection incomparable with the record.
        # The two are the same operation — `freeze_increment(base, full)` against the library's
        # `base + stop_gradient(full - base)` — so the migration is safe, but it needs its own
        # bit-identity gate, not a quiet swap.
        def mk_conv(orig):
            def w(mesh, Kv, Av, bvfreq, **kw):
                out = orig(mesh, Kv, Av, bvfreq, **kw)
                _bump("convect")
                return adjoint_stab.freeze_increment((Kv, Av), out)
            return w
        install(pp, "mo_convect", mk_conv)

    if "visc" in knobs:
        def mk_visc(orig):
            def w(mesh, uv, uv_rhs, Av, *a, **k):
                _bump("visc")
                return orig(mesh, uv, uv_rhs, adjoint_stab.freeze(Av), *a, **k)
            return w
        install(momentum, "impl_vert_visc", mk_visc)

    if "slope" in knobs:
        def mk_gm(orig):
            def w(mesh, T, S, bvfreq, *a, **k):
                _bump("slope")
                return orig(mesh, T, S, adjoint_stab.freeze(bvfreq), *a, **k)
            return w
        install(gm, "gm_diagnostics", mk_gm)

    if "k33" in knobs:
        def mk_k33(orig):
            def w(*a, **k):
                _bump("k33")
                return adjoint_stab.freeze(orig(*a, **k))
            return w
        install(gm_redi, "k33_augmentation", mk_k33)

    if "bv" in knobs:
        def mk_bv(orig):
            def w(*a, **k):
                out = orig(*a, **k)
                if not (isinstance(out, tuple) and len(out) == 3):
                    raise RuntimeError("eos.compute_pressure_bv no longer returns "
                                       "(density, hpressure, bvfreq) — the 'bv' knob needs updating")
                _bump("bv")
                return out[0], out[1], adjoint_stab.freeze(out[2])
            return w
        install(eos, "compute_pressure_bv", mk_bv)

    if "fct" in knobs:
        def mk_fct(orig):
            def w(*a, **k):
                _bump("fct")
                return adjoint_stab.freeze(orig(*a, **k))
            return w
        install(tracer_adv, "zalesak_limit", mk_fct)

    if "icevisc" in knobs:
        def mk_icevisc(orig):
            def w(*a, **k):
                _bump("icevisc")
                k["freeze_viscosity"] = True
                return orig(*a, **k)
            return w
        install(ice_mevp, "mevp_iterate", mk_icevisc)

    if "pgf" in knobs:
        def mk_pgf(orig):
            def w(*a, **k):
                _bump("pgf")
                return adjoint_stab.freeze(orig(*a, **k))
            return w
        install(pgf, "pressure_force_shchepetkin", mk_pgf)

    try:
        yield
    finally:
        for mod, attr, orig in saved:
            setattr(mod, attr, orig)


def check_fired(knobs):
    missed = [k for k in knobs if not _FIRED.get(k)]
    if missed:
        raise SystemExit(
            f"--freeze {missed} never fired during the forward: the patched call site is not on "
            f"this configuration's code path (or its signature changed). Refusing to report a "
            f"result that would read as 'this term is not the amplifier' without having tested it.")


# ---------------------------------------------------------------------------

def segments_for(length, seg_mult=1.0):
    return max(2, int(round(seg_mult * math.sqrt(length))))


def _rss_gb():
    """Peak resident set size of this process, GB. Measured, because the 60/90-day truth build
    was OOM-killed at 240 GB (26801648) and no arithmetic I could write reached that number:
    the marks are ~98 MB per truth-day, so 90 days is under 10 GB. Print it per iteration
    rather than infer it."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def load_or_build_truth(cache, one_day, st0, stack_steps, wet3d, area3d, m_true,
                        recompute, nd_truth=NDAYS_TRUTH, want_mark=None, weights_from=None):
    """The truth mark(s) + per-level weights.

    ``want_mark`` (0-based) returns ONLY that mark instead of all ``nd_truth`` of them. The
    caller uses exactly one — ``truth_marks[ndays-1]`` — and materialising the rest costs
    ~98 MB per truth-day on the host: 8.8 GB at 90 days, 36 GB at a year, for nothing.

    ``weights_from`` loads ``w_T``/``w_S``/``w_eta`` from a DIFFERENT cache (e.g. the existing
    30-day one) instead of deriving them here. Two reasons, and the first is scientific:
    the weights come from the truth marks, so a different ``--truth-days`` is a different loss
    function and |dL/dm| is then NOT comparable across window lengths — the flag's own help
    says so. Sharing one weight set makes the ladder rungs measure the same thing. The second
    is that it removes the per-day mark accumulation from the build entirely, which is the only
    version that scales to a year."""
    w = None
    if weights_from is not None:
        zw = np.load(weights_from)
        w = (jnp.asarray(zw["w_T"]), jnp.asarray(zw["w_S"]), jnp.asarray(zw["w_eta"]))
        print(f"[truth] weights reused from {weights_from} (shared loss across rungs)",
              flush=True)

    if cache.exists() and not recompute:
        z = np.load(cache)
        if want_mark is not None:
            d = want_mark
            marks = {d: (z["T"][d], z["S"][d], z["eta"][d])}
        else:
            marks = [(z["T"][d], z["S"][d], z["eta"][d]) for d in range(nd_truth)]
        if w is None:
            w = (jnp.asarray(z["w_T"]), jnp.asarray(z["w_S"]), jnp.asarray(z["w_eta"]))
        return marks, w[0], w[1], w[2]

    keep_all = w is None            # weights need every mark; a supplied set needs none
    # ⚠️ CACHE THE ONE MARK WE KEEP, even when the weights came from elsewhere. Until
    # 2026-08-09 `--weights-from` meant no cache was written AT ALL, so every run rebuilt the
    # truth from scratch: 1.5 h for a year, 3 h for two, 7.4 h for five — per leg of a resumed
    # chain, which makes a multi-year window unreachable no matter how good the checkpoint is.
    # A single mark is ~98 MB against the all-marks cache's 98 MB PER DAY.
    single_cache = None
    if not keep_all and want_mark is not None:
        single_cache = cache.with_name(cache.stem + f"_mark{want_mark}.npz")
        if single_cache.exists() and not recompute:
            z = np.load(single_cache)
            print(f"[truth] mark {want_mark} reused from {single_cache} "
                  f"(no {nd_truth}-day rebuild)", flush=True)
            return {want_mark: (z["T"], z["S"], z["eta"])}, w[0], w[1], w[2]
    marks, s = ([] if keep_all else {}), st0
    for d in range(nd_truth):
        sfs_day = stack_steps(d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY)
        s = one_day(s, sfs_day, jnp.asarray(m_true, jnp.float64))
        if keep_all:
            marks.append(jax.device_get((s.T, s.S, s.eta_n)))
        elif want_mark is not None and d == want_mark:
            marks[d] = jax.device_get((s.T, s.S, s.eta_n))
        jax.tree.map(lambda x: x.delete() if hasattr(x, "delete") else None, sfs_day)
        if d % 10 == 0 or d == nd_truth - 1:
            print(f"[truth] day {d+1}/{nd_truth}  host RSS {_rss_gb():.1f} GB", flush=True)
    del s
    if keep_all:
        T_st = jnp.asarray(np.stack([m[0] for m in marks]))
        S_st = jnp.asarray(np.stack([m[1] for m in marks]))
        e_st = jnp.asarray(np.stack([m[2] for m in marks]))
        w_T, w_S, w_eta = level_weights(T_st, S_st, e_st, wet3d, area3d)
        w_T, w_S, w_eta = (jax.block_until_ready(x) for x in (w_T, w_S, w_eta))
        for a in (T_st, S_st, e_st):
            a.delete()
        cache.parent.mkdir(parents=True, exist_ok=True)
        # The rename is atomic, so a READER is always safe. The temp path must carry the pid or
        # two jobs sharing a seed — routine now that ice-on/ice-off and freeze-knob contrasts
        # run as concurrent siblings — interleave their writes into one file and then rename
        # the corruption into place. The forward is identical across those contrasts, which is
        # precisely why they collide on the same truth.
        tmp = cache.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(tmp, T=np.stack([m[0] for m in marks]), S=np.stack([m[1] for m in marks]),
                 eta=np.stack([m[2] for m in marks]), w_T=np.asarray(w_T),
                 w_S=np.asarray(w_S), w_eta=np.asarray(w_eta))
        tmp.rename(cache)
        if want_mark is not None:
            marks = {want_mark: marks[want_mark]}
        return marks, w_T, w_S, w_eta
    if single_cache is not None:
        single_cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = single_cache.with_suffix(f".tmp{os.getpid()}.npz")
        m = marks[want_mark]
        np.savez(tmp, T=np.asarray(m[0]), S=np.asarray(m[1]), eta=np.asarray(m[2]))
        tmp.rename(single_cache)
        print(f"[truth] mark {want_mark} cached -> {single_cache} "
              f"({single_cache.stat().st_size/1e6:.0f} MB); later legs skip the "
              f"{nd_truth}-day rebuild", flush=True)
    print(f"[truth] built mark {want_mark} only; peak host RSS {_rss_gb():.1f} GB", flush=True)
    return marks, w[0], w[1], w[2]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="freeze targets:\n" + "\n".join(f"  {k:9s} {v}" for k, v in FREEZE_HELP.items()))
    ap.add_argument("--snap", required=True)
    ap.add_argument("--season", default="seed")
    ap.add_argument("--n-days", type=int, required=True, help="window length in days (<= --truth-days)")
    ap.add_argument("--boundary-stride", type=int, default=1,
                    help="SECOND level of checkpointing: keep only every Nth chunk boundary on "
                         "the host and recompute the spans between them in the backward. Host "
                         "cost becomes (n_chunks/N + N) states instead of n_chunks, minimised "
                         "near sqrt(n_chunks); the price is one extra forward pass and the "
                         "gradient is unchanged. 1 = keep all (default). A YEAR needs this: "
                         "366 x 1.93 GB = 706 GB against a 515 GB node.")
    ap.add_argument("--ckpt-dir", type=Path, default=None,
                    help="make the reverse sweep RESUMABLE across jobs: persist the forward's "
                         "chunk boundaries once and the reverse cotangent every --ckpt-every "
                         "chunks into this directory. Window length then costs jobs instead of "
                         "a bigger slot — a year is ~10 h against the 12 h GPU limit and two "
                         "years fit no slot at all. Absent = today's behaviour, unchanged.")
    ap.add_argument("--resume", action="store_true",
                    help="continue the checkpoint in --ckpt-dir. REFUSES if the run's identity "
                         "(seed, window, chunking, stride, freezes, truth settings) differs "
                         "from the one that wrote it — resuming across a config change would "
                         "give a gradient belonging to neither run.")
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="reverse chunks between cotangent checkpoints. Each write is 1.93 GB "
                         "into one of TWO alternating slots (never accumulating, never "
                         "deleting), and bounds how much is redone after an unclean kill.")
    ap.add_argument("--stop-after-min", type=float, default=None,
                    help="checkpoint and exit 42 after this many minutes. Set it BELOW the "
                         "job's wall clock: a scheduler kill mid-write is what the two-slot "
                         "scheme survives, but a clean stop costs nothing and loses nothing.")
    ap.add_argument("--weights-from", type=Path, default=None,
                    help="npz to take w_T/w_S/w_eta from (e.g. the 30-day truth cache). Makes "
                         "every ladder rung share ONE loss function, so |dL/dm| is comparable "
                         "across window lengths, and removes the per-day mark accumulation.")
    ap.add_argument("--allow-unfrozen", action="store_true",
                    help="permit a >2-day window with no --freeze knobs. Refused by default: "
                         "that configuration is measured at +1.64e+154 over 30 days.")
    ap.add_argument("--allow-frozen-ice", action="store_true",
                    help="permit a >2-day window with --ice frozen, i.e. NO ice in the "
                         "backward. Refused by default: the training configuration is "
                         "--ice rheology, it costs ~2 %% more, and it moves the 30-day "
                         "gradient by 1.07 %%. Use only for a deliberate ice-on/ice-off "
                         "contrast.")
    ap.add_argument("--chunk-steps", type=int, default=STEPS_PER_DAY,
                    help="reverse-accumulation chunk length in steps (default 48 = 1 day). "
                         "Smaller = finer cotangent-norm trace and lower device peak, but more "
                         "host memory for the stored boundary states (~1 GB each).")
    ap.add_argument("--truth-mult", type=float, default=1.25)
    ap.add_argument("--truth-days", type=int, default=NDAYS_TRUTH,
                    help="length of the reference trajectory in days (default 10). It caps the "
                         "window, so a 30-day window needs 30 here. ⚠️ The per-level variance "
                         "weights in the loss are computed from the spread of the reference "
                         "marks, so runs with DIFFERENT --truth-days use different weights and "
                         "their loss and gradient VALUES are not comparable. The growth rate "
                         "per step and the reproducibility are. Each length gets its own cache "
                         "file, so nothing is silently reused across the two.")
    ap.add_argument("--run-start-year", type=int, default=1958)
    ap.add_argument("--mixing", choices=("tke", "kpp"), default="tke",
                    help="which vertical mixing closure to run. Both the MITgcm/ECCO community "
                         "and this study point the same way: the TKE closure (MITgcm's GGL90, "
                         "same Gaspar 1990 family) differentiates better than KPP, because its "
                         "diffusivity comes from a PROGNOSTIC energy equation advanced by a "
                         "smooth implicit solve, while KPP's comes from a DIAGNOSTIC profile "
                         "hung off a boundary-layer depth located by searching for a critical "
                         "Richardson number — a discrete index, discontinuous in the state. "
                         "fesom_jax/kpp.py has to stop_gradient that index and several other "
                         "discrete selections; fesom_jax/tke.py has none. This flag measures "
                         "the difference in our own model rather than taking it on faith.")
    ap.add_argument("--ice", choices=("frozen", "rheology", "exact"), default="frozen",
                    help="how the BACKWARD treats sea ice; the forward runs full mEVP in all "
                         "three. 'frozen' (the default so far) skips the whole ice update, so no "
                         "gradient flows through ice at all. 'exact' differentiates everything "
                         "and returns NaN — the mEVP momentum solve amplifies by 1.1795 per "
                         "relaxation sweep, 4e8 per step over its 120 sweeps. 'rheology' is the "
                         "middle option and the one we want: differentiate ice thermodynamics, "
                         "advection, cut-off and the ocean-to-ice coupling, but unroll zero mEVP "
                         "sweeps in the backward. Over 8 steps that alone tames the ice adjoint "
                         "(3.4e-05 against 2.0e+39) while every other ice component is harmless, "
                         "and it recovers the pathway that matters here: eddy transport changes "
                         "SST, which changes ice growth, which changes the surface heat and salt "
                         "fluxes back into the ocean. Only the ice velocity stays dark.")
    # --ice-damping (Newtonian damping of the mEVP adjoint) was removed 2026-08-07 along with
    # ice_mevp.mevp_dynamics(adjoint_damping=...). Measured and rejected: the usable window is
    # annihilated at T_N=6, blown at T_N=7, cancelling exactly at 6.57, so no single global
    # constant works. Numbers kept in docs/ADJOINT_HORIZON.md; use --ice rheology instead.
    ap.add_argument("--freeze", default="", help="comma-separated freeze targets (see below)")
    ap.add_argument("--gamma", type=float, default=1.0, help="per-step cotangent discount")
    ap.add_argument("--filter-factor", type=float, default=0.0,
                    help="adjoint diffusion scale in multiples of local node spacing")
    ap.add_argument("--clip", type=float, default=0.0, help="per-step cotangent norm cap (nonlinear)")
    ap.add_argument("--fd-h", default="1e-2,1e-3,1e-4",
                    help="finite-difference step sizes for the sign check; empty to skip "
                         "(each value costs two extra full forwards)")
    ap.add_argument("--truth-cache", type=Path, default=None)
    ap.add_argument("--recompute-truth", action="store_true")
    ap.add_argument("--repeat", type=int, default=1,
                    help="recompute loss and gradient this many times IN THE SAME PROCESS and "
                         "report the spread. The executables are already compiled, so repeats "
                         "are nearly free, and they answer the question that has to be settled "
                         "before any horizon number means anything: is this gradient even a "
                         "function of its inputs? On GPU it need not be — the unstructured "
                         "scatter-adds this model is built from use atomics, whose summation "
                         "order varies run to run.")
    ap.add_argument("--tag", default="", help="free-text label written into the results row")
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parent / "adjoint_horizon_results.jsonl")
    args = ap.parse_args()

    ndays, nd_truth = args.n_days, args.truth_days
    assert 1 <= ndays <= nd_truth, f"--n-days must be 1..{nd_truth} (raise --truth-days)"
    n = ndays * STEPS_PER_DAY
    K = int(args.chunk_steps)
    assert K >= 1 and n % K == 0, f"--chunk-steps must divide the window ({n} steps)"
    n_chunks = n // K
    knobs = [k for k in args.freeze.split(",") if k]
    # ⚠️ A long window with NOTHING frozen is a documented blow-up, not an experiment worth
    # 40 minutes of A100. Every clean record in adjoint_horizon_results.jsonl carries
    # freeze:['mix']; the one 30-day run with freeze:[] is tagged "month-unfrozen-contrast"
    # and returned +1.64e+154. Two jobs were spent rediscovering that (26800812/3, cancelled
    # at |cot| = 1.8e57 by day 20) because a NEW sbatch simply forgot the flag. Short windows
    # stay unguarded: unfrozen 1-2 day contrasts are this script's attribution job.
    if ndays > 2 and not knobs and not args.allow_unfrozen:
        raise SystemExit(
            f"--n-days {ndays} with --freeze '' is the configuration measured at +1.64e+154 "
            f"over 30 days. Pass '--freeze mix' (what every clean long-window record used), "
            f"or '--allow-unfrozen' if an unfrozen long contrast is genuinely what you want.")
    # ⚠️ Same trap, second flag. `--ice frozen` is this script's DEFAULT and it means "the
    # backward skips the ice update entirely", not "the rheology is approximated". The mode we
    # actually train in is `rheology`: ice thermodynamics, advection, cut-off and the
    # ocean-to-ice coupling all differentiated, ZERO mEVP sweeps unrolled. The name refers to
    # the component being approximated, not to what is switched on, and that reading cost the
    # whole long-window record: every rung at 60, 90, 180, 365 and 731 days ran with no ice in
    # the backward at all, in a campaign whose blow-ups all land in February.
    # It is not even expensive. Paired runs from the same seed, freeze:['mix']:
    #   5 d  -5.587557e-02 (ice) vs -5.599449e-02 (none)  -> 0.21 %, backward 482 s vs 439 s
    #  10 d  -1.601827e-01       vs -1.613639e-01         -> 0.73 %, backward 808 s vs 769 s
    #  30 d  -3.382626e-01       vs -3.419135e-01         -> 1.07 %, backward 2121 s vs 2078 s
    # Both clean, amplification identical to four digits, +0.6 GB on the card. The difference
    # grows with window length, so it is a real accumulating pathway, not noise: those runs
    # reproduce their gradient exactly across repeated passes.
    if ndays > 2 and args.ice == "frozen" and not args.allow_frozen_ice:
        raise SystemExit(
            f"--n-days {ndays} with --ice frozen puts NO ice in the backward, while the "
            f"training configuration is '--ice rheology' (ice differentiated, mEVP sweeps not "
            f"unrolled). It costs ~2 % more backward time and moves the 30-day gradient by "
            f"1.07 %. Pass '--ice rheology', or '--allow-frozen-ice' for a deliberate contrast.")
    mix_kw = ({"tke_cfg": TKE_CFG} if args.mixing == "tke" else {"kpp_cfg": KppConfig()})
    ice_cfg = (ICE_FROZEN if args.ice == "frozen"
               else IceConfig(whichEVP=ICE_FROZEN.whichEVP, adjoint_mode="exact"))
    if args.ice == "rheology":
        # Differentiate the ice, but unroll ZERO mEVP relaxation sweeps in the backward. The
        # forward still runs all 120, so the trajectory is untouched. Installed for the whole
        # process (every jit trace below happens after this point).
        _mevp_orig = ice_mevp.mevp_dynamics

        def _mevp_no_grad(*a, **k):
            k["adjoint_subcycles"] = 0
            return _mevp_orig(*a, **k)

        ice_mevp.mevp_dynamics = _mevp_no_grad
    stab_cfg = adjoint_stab.StabConfig(gamma=args.gamma, filter_factor=args.filter_factor,
                                       clip=args.clip)

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh = load_mesh(MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    with open(args.snap, "rb") as f:
        st0 = jax.device_put(pickle.load(f))
    yr, mo, dy = seed_date(args.snap, args.run_start_year)
    sst0 = np.asarray(phc_initial_state(mesh, IC_DIR).T[:, 0])
    cf = surface_forcing.build_surface_forcing(mesh, yr, sst_ic=sst0)
    dates_all = surface_forcing.dates_for_steps(yr, DT, nd_truth * STEPS_PER_DAY,
                                                start_month=mo, start_day=dy)
    fs = cf.static

    def stack_steps(lo, hi):
        return cf.stack(dates_all[lo:hi])

    wet3d = jnp.asarray(mesh.node_layer_mask, jnp.float64)
    area3d = jnp.asarray(np.asarray(mesh.areasvol), jnp.float64) * wet3d
    kgm0 = float(Params.defaults().k_gm)

    def params_for(m):
        return build_params({"k_gm": kgm0 * m, "redi_kmax": kgm0 * m})

    stabilize = adjoint_stab.make_state_stabilizer(mesh, stab_cfg)
    if stabilize is not None and stabilize.plan is not None:
        ns, kdt = stabilize.plan
        print(f"[setup] adjoint diffusion: {ns} Laplacian steps at {args.filter_factor:g}x local "
              f"spacing (kappa_dt median {float(jnp.median(jnp.asarray(kdt))):.3e} m^2)", flush=True)

    def run_chunk(state, sfs, m):
        p = params_for(m)

        def body(carry, sf):
            nxt = step(carry, mesh, op, None, p, dt=DT, is_first_step=False, step_forcing=sf,
                       forcing_static=fs, ice_cfg=ice_cfg, gm_cfg=GM_CFG, ale_cfg=ALE_CFG,
                       remat_blocks=True, **mix_kw)
            if stabilize is not None:
                nxt = stabilize(nxt)
            return nxt, None

        return _run_steps(body, state, xs=sfs, checkpoint=True, segments=segments_for(K))

    @jax.jit
    def fwd_chunk(state, sfs, m):
        return run_chunk(state, sfs, m)

    @jax.jit
    def bwd_chunk(state, sfs, m, cot):
        _, vjp = jax.vjp(lambda s, mm: run_chunk(s, sfs, mm), state, m)
        return vjp(cot)

    # the truth run always uses the UNPATCHED model at one-day granularity — the twin's
    # reference trajectory must not depend on which backward knob is being tested.
    @jax.jit
    def one_day(state, sfs_day, m):
        segs = segments_for(STEPS_PER_DAY)

        def body(carry, sf):
            nxt = step(carry, mesh, op, None, params_for(m), dt=DT, is_first_step=False,
                       step_forcing=sf, forcing_static=fs, ice_cfg=ice_cfg, gm_cfg=GM_CFG,
                       ale_cfg=ALE_CFG, remat_blocks=True, **mix_kw)
            return nxt, None

        return _run_steps(body, state, xs=sfs_day, checkpoint=True, segments=segs)

    print(f"[setup] built in {time.time()-t0:.1f}s; seed={args.season} date={yr}-{mo:02d}-{dy:02d} "
          f"N={ndays} d ({n} steps) in {n_chunks} x {K}-step chunks  m_true={args.truth_mult}  "
          f"freeze={knobs or 'none'}  mixing={args.mixing}  ice={args.ice}  "
          f"stab={stab_cfg.describe()}", flush=True)

    # ⚠️ was hardcoded to /work/ab0995, which is at 98% (58 T free of 2.6 P). The truth cache
    # is ~98 MB per truth-day, so a 90-day ladder rung writes ~9 GB there and a year ~36 GB.
    # `_nd.data_path` is the single migrated-root constant (ba1583, 116 T free).
    cache = args.truth_cache or (_nd.data_path("horizon_truth") /
                                 f"{Path(args.snap).stem}_m{args.truth_mult:g}_{nd_truth}d.npz")
    t1 = time.time()
    truth_marks, w_T, w_S, w_eta = load_or_build_truth(
        cache, one_day, st0, stack_steps, wet3d, area3d, args.truth_mult,
        args.recompute_truth, nd_truth, want_mark=ndays - 1,
        weights_from=args.weights_from)
    print(f"[truth] {nd_truth}-day truth ready in {time.time()-t1:.1f}s ({cache})", flush=True)
    _m = truth_marks[ndays - 1] if isinstance(truth_marks, dict) else truth_marks[ndays - 1]
    T_hat, S_hat, e_hat = (jnp.asarray(x) for x in _m)
    del truth_marks

    def misfit_of(s):
        return stage1_misfit(s.T, s.S, s.eta_n, T_hat, S_hat, e_hat,
                             w_T, w_S, w_eta, wet3d, area3d)

    def _free(tree):
        jax.tree.map(lambda x: x.delete() if hasattr(x, "delete") else None, tree)

    def chunk_sfs(c):
        """One chunk's forcing stack, built ON DEMAND and freed by the caller.

        ⚠️ This used to be `sfs_chunks = [stack_steps(...) for c in range(n_chunks)]` — every
        chunk's stack built up front and held on DEVICE for the whole run, though only one is
        ever in flight. That idle residency is ~0.5 GB per simulated day and it is the entire
        reason the device peak was NOT flat in window length: 43.1 GB at 1 d rising to 56.9 GB
        at 30 d. Extrapolated it OOMs an 80 GB card around 90 days, which would have looked
        like "the adjoint does not scale" when it is only a buffer lifetime. Same values, same
        order of operations, bit-identical results — only when the arrays exist changes. The
        re-stack costs ~0.1-0.3 s per chunk per pass against ~85 s per simulated day.
        (Same lesson as horizon_scan.py's "⚠️ MEMORY" note about the whole-window stack.)"""
        return stack_steps(c * K, (c + 1) * K)

    def forward(m, keep_boundaries=False, stride=1, ck=None):
        """Run the window; optionally return chunk-boundary states, on the HOST.

        ``stride > 1`` keeps only every ``stride``-th boundary (plus the window start), which
        is the SECOND level of checkpointing — see :func:`rebuild_span`. Returned as a dict
        keyed by chunk index, because it is no longer dense.

        ``ck`` streams each kept boundary to disk AS IT IS PRODUCED and honours the leg's time
        budget, which is what makes the forward itself resumable. Without it the forward is
        all-or-nothing: a leg killed part way redoes the whole thing, which is 1.5 h at a
        one-year window and is what put a 5-year rung out of reach (leg 1 would need 7.4 h of
        truth plus 7.4 h of forward before the reverse could start, against a 12 h slot).
        Only the MAIN forward passes ``ck`` — the FD evaluations and :func:`rebuild_span` must
        not, or they would overwrite the checkpoint with their own trajectory."""
        s, c0 = st0, 0
        kept = {0: jax.device_get(s)} if keep_boundaries else None
        if ck is not None and keep_boundaries:
            st_r, c0 = ck.forward_resume(like=st0)
            if st_r is not None:
                kept = ck.load_boundaries(like=st0)
                s = jax.device_put(st_r)
            elif ck.put_boundary(0, kept[0]):
                raise SystemExit(_ckpt.EXIT_INCOMPLETE)
        for c in range(c0, n_chunks):
            sfs_c = chunk_sfs(c)
            nxt = jax.block_until_ready(fwd_chunk(s, sfs_c, m))
            _free(sfs_c)                     # after block_until_ready: never free a live input
            if s is not st0:
                _free(s)
            s = nxt
            if keep_boundaries and ((c + 1) % stride == 0 or c == n_chunks - 1):
                kept[c + 1] = jax.device_get(s)
                if ck is not None and ck.put_boundary(c + 1, kept[c + 1]):
                    raise SystemExit(_ckpt.EXIT_INCOMPLETE)
        return s, kept

    def rebuild_span(stored, lo, hi, m):
        """Re-run chunks ``lo..hi-1`` from the stored boundary at ``lo``, returning their input
        states on the HOST, keyed by chunk index.

        This is the second level of checkpointing, and the reason a YEAR is reachable at all.
        Level one already trades device memory for recompute inside each chunk. Level two does
        the same to the HOST store: a CORE2 ``State`` is 1.93 GB, so keeping one boundary per
        day costs 706 GB over 365 days against a 515 GB node — measured, not estimated (60 days
        came in at 117.8 GB against 61 x 1.93 = 118). Keeping every ``stride``-th boundary and
        recomputing the span in between costs ``n/stride + stride`` states, minimised near
        ``stride = sqrt(n_chunks)``: **~74 GB for a year** at stride 19.

        The price is exactly ONE extra forward pass over the window, because every chunk is
        recomputed at most once across the whole reverse sweep. The gradient is UNCHANGED — the
        recomputed states are the same states, so this is checkpointing, not approximation, and
        `--boundary-stride 2` against `1` must agree to round-off (that is the validation run)."""
        s = jax.device_put(stored[lo])
        local = {lo: stored[lo]}
        for c in range(lo, hi - 1):
            sfs_c = chunk_sfs(c)
            nxt = jax.block_until_ready(fwd_chunk(s, sfs_c, m))
            _free(sfs_c)
            _free(s)
            s = nxt
            local[c + 1] = jax.device_get(s)
        _free(s)
        return local

    m0 = jnp.asarray(1.0, jnp.float64)
    rec = {"season": args.season, "snap": Path(args.snap).name, "date": f"{yr}-{mo:02d}-{dy:02d}",
           "n_days": ndays, "n_steps": n, "dt": DT, "chunk_steps": K, "n_chunks": n_chunks,
           "marks": "end", "m_true": args.truth_mult, "kgm0": kgm0,
           "config": f"zstar+{args.mixing.upper()}+mEVP+GM", "adjoint": f"{args.ice}-ice",
           "freeze": knobs, "gamma": args.gamma, "filter_factor": args.filter_factor,
           "clip": args.clip, "stab": stab_cfg.describe(), "tag": args.tag,
           # Provenance that used to live ONLY in the checkpoint identity dict, which is built
           # only under --ckpt-dir. Without these a row cannot be reproduced from the row: the
           # 180-day ladder record did not say it ran at boundary-stride 2, and the year rung
           # did not say it ran at 19, which is exactly the confound that cost two 5-hour jobs
           # to resolve. `ice` duplicates `adjoint` under the name a reader actually greps for.
           "ice": args.ice, "mixing": args.mixing,
           "boundary_stride": max(1, int(args.boundary_stride)),
           "truth_days": nd_truth, "truth_mult": args.truth_mult,
           "weights_from": (None if args.weights_from is None else str(args.weights_from))}

    # --- resumable reverse sweep (optional) ----------------------------------
    # The identity below is what a --resume is checked against. Every field is something that,
    # if it differed between legs, would make the accumulated gradient belong to neither run:
    # the seed and its date, the window and its chunking, the recompute stride, which terms are
    # frozen in the backward, and where the loss's truth and weights came from. Add to this
    # list rather than trimming it — a silently-resumed configuration change is exactly the
    # class of bug this project has twice paid for.
    ck = None
    if args.ckpt_dir is not None:
        if args.repeat > 1:
            raise SystemExit("--ckpt-dir with --repeat > 1 is refused: the repeats measure "
                             "reproducibility within ONE process, which a resumed run cannot "
                             "provide. Checkpoint a single pass, or drop --ckpt-dir.")
        ident = dict(snap=Path(args.snap).name, date=f"{yr}-{mo:02d}-{dy:02d}",
                     season=args.season, n_days=ndays, n_chunks=n_chunks, chunk_steps=K,
                     dt=DT, boundary_stride=max(1, int(args.boundary_stride)),
                     freeze=sorted(knobs), mixing=args.mixing, ice=args.ice,
                     truth_days=nd_truth, truth_mult=args.truth_mult,
                     weights_from=(None if args.weights_from is None
                                   else str(args.weights_from)),
                     gamma=args.gamma, filter_factor=args.filter_factor, clip=args.clip,
                     run_start_year=args.run_start_year)
        ck = _ckpt.Checkpoint.open(args.ckpt_dir, ident, resume=args.resume,
                                   ckpt_every=args.ckpt_every,
                                   stop_after=(None if args.stop_after_min is None
                                               else 60.0 * args.stop_after_min))
    try:
        with patched(knobs):
            def one_pass(verbose, ck=None):
                """One full forward + chunked reverse accumulation. Returns (loss, dL/dm,
                cotangent-norm trace, host GB, fwd seconds, bwd seconds).

                ``ck`` (a :class:`adjoint_ckpt.Checkpoint`) makes the sweep RESUMABLE across
                jobs. The forward's boundaries are written once and the reverse cotangent every
                few chunks, so a window longer than the queue's wall clock stops being
                impossible: a year is ~10 h against a 12 h limit and two years fit no slot at
                all, but the carried state of the reverse loop is one 1.93 GB cotangent, a
                scalar and an index. With ``ck`` the run stops on its own time budget, exits
                ``EXIT_INCOMPLETE``, and the next job continues from the same chunk."""
                ta = time.time()
                stride = max(1, int(args.boundary_stride))
                resuming_reverse = ck is not None and ck.state["phase"] == "reverse"
                if ck is not None and ck.have_boundaries:
                    boundaries = ck.load_boundaries(like=st0)
                    loss_v, fwd_t = float(ck.loss), 0.0
                    s_fin = None
                else:
                    s_fin, boundaries = forward(m0, keep_boundaries=True, stride=stride,
                                                ck=ck)
                    loss_v = float(misfit_of(s_fin))
                    fwd_t = time.time() - ta
                    if ck is not None:
                        ck.finish_forward(loss_v)
                hgb = sum(a.nbytes for a in jax.tree.leaves(boundaries)) / 1e9
                if verbose:
                    print(f"[fwd] loss(m=1)={loss_v:.6e}  {fwd_t:.1f}s  "
                          f"boundaries on host: {hgb:.1f} GB", flush=True)

                # The loss cotangent seeds the sweep. `forward` always keeps the LAST boundary
                # whatever the stride, so it is available on a resume too and the final state
                # never has to be recomputed.
                if s_fin is None:
                    s_fin = jax.device_put(boundaries[n_chunks])
                _, vjp_L = jax.vjp(misfit_of, s_fin)
                seed_cot = jax.block_until_ready(vjp_L(1.0)[0])
                _free(s_fin)
                del s_fin

                if ck is None:
                    c_t, g0, c_start = seed_cot, 0.0, n_chunks - 1
                    tr = [float(adjoint_stab.cotangent_norm(c_t))]
                else:
                    c_t, g0, c_start, tr = ck.reverse_start(seed_cot, n_chunks, like=st0)
                    if not resuming_reverse:
                        tr = [float(adjoint_stab.cotangent_norm(c_t))]
                    else:
                        _free(seed_cot)
                        c_t = jax.device_put(c_t)
                tb = time.time()
                g = jnp.asarray(g0, jnp.float64)
                for sup_lo in reversed(range(0, n_chunks, stride)):
                    sup_hi = min(sup_lo + stride, n_chunks)
                    if sup_lo > c_start:          # this superchunk was done in an earlier leg
                        continue
                    # boundaries[] holds only every stride-th state; rebuild the span in
                    # between from the nearest stored one. stride=1 => `span` IS boundaries.
                    span = (boundaries if stride == 1
                            else rebuild_span(boundaries, sup_lo, sup_hi, m0))
                    for c in reversed(range(sup_lo, min(sup_hi, c_start + 1))):
                        s_in = jax.device_put(span[c])
                        sfs_c = chunk_sfs(c)             # on demand; see chunk_sfs's docstring
                        g_s, g_c = bwd_chunk(s_in, sfs_c, m0, c_t)
                        g = g + g_c
                        # ⚠️ ORDER MATTERS: `s_in`, `sfs_c` and the OLD `c_t` are live inputs
                        # to the VJP until it completes. Block on its output first, then free.
                        g_s = jax.block_until_ready(g_s)
                        _free(c_t)
                        _free(sfs_c)
                        _free(s_in)
                        del s_in
                        c_t = g_s
                        tr.append(float(adjoint_stab.cotangent_norm(c_t)))
                        if verbose:
                            print(f"    [bwd] chunk {c:3d}  t=+{c*K*DT/86400:5.2f} d  "
                                  f"|cot|={tr[-1]:.6e}  dL/dm so far={float(g):+.6e}", flush=True)
                        # After chunk c the carried cotangent belongs to chunk c-1; -1 means
                        # the sweep is complete. This is the only place a leg may end.
                        if ck is not None and ck.tick(c - 1, c_t, float(g), tr):
                            raise SystemExit(_ckpt.EXIT_INCOMPLETE)
                    if stride > 1:
                        span = None
                return loss_v, float(g), tr, hgb, fwd_t, time.time() - tb

            l0, ad, trace, host_gb, fwd_s, bwd_s = one_pass(verbose=True, ck=ck)
            check_fired(knobs)
            pk = peak_gb()

            # --- reproducibility: is this gradient a function of its inputs? ---------
            # Same process, same compiled executables, same inputs. Any spread here is the
            # model's own numerical nondeterminism (atomic scatter-adds on the unstructured
            # mesh) amplified by the backward. A horizon verdict read off a single sample of
            # a quantity that moves by orders of magnitude between identical runs is not a
            # measurement, so this is checked BEFORE any of it is reported.
            rep_loss, rep_grad = [l0], [ad]
            for r in range(1, max(1, args.repeat)):
                lr, gr, _, _, _, _ = one_pass(verbose=False)
                rep_loss.append(lr)
                rep_grad.append(gr)
                print(f"[repeat {r}] loss={lr:.9e}  dL/dm={gr:+.6e}", flush=True)

            # --- per-step amplification, the chaos-vs-switch discriminator -----------
            # ONE implementation, in the library, shared with seam_gate and M6's window
            # screen. It used to live here inline; a second copy elsewhere is how this
            # project once ended up with two "freeze mixing" implementations and no test
            # that they agreed.
            _amp = adjoint_stab.amplification(trace, K)
            amp = list(_amp.per_step)
            amp_med, amp_max, amp_spread, worst = (_amp.median, _amp.max,
                                                   _amp.log_spread, _amp.worst_chunk)
            amp_sus = _amp.sustained

            # --- FD sign check -------------------------------------------------------
            rows = []
            for h in [float(x) for x in args.fd_h.split(",") if x.strip()]:
                lp, _ = forward(m0 * (1 + h))
                lm, _ = forward(m0 * (1 - h))
                rows.append((h, float((misfit_of(lp) - misfit_of(lm)) / (2 * h))))
            if rows:
                best_fd = min(rows, key=lambda r: abs(ad - r[1]) / max(abs(r[1]), 1e-300))[1]
                plateau = min(abs(ad - fd) / max(abs(fd), 1e-300) for _, fd in rows)
                fd_sign_agree = bool(np.sign(ad) == np.sign(best_fd))
            else:
                best_fd, plateau, fd_sign_agree = None, None, None

        informative = bool(ad < 0.0)
        blown = bool((not np.isfinite(ad)) or abs(ad) > BLOW)
        clean = bool(np.isfinite(ad) and ad != 0.0 and not blown and informative
                     and (fd_sign_agree is not False))
        # spread across repeats, as a ratio — the only scale-free way to say "orders of magnitude"
        grad_spread = (max(abs(g) for g in rep_grad) / max(min(abs(g) for g in rep_grad), 1e-300)
                       if len(rep_grad) > 1 else 1.0)
        loss_spread = ((max(rep_loss) - min(rep_loss)) / max(abs(np.mean(rep_loss)), 1e-300)
                       if len(rep_loss) > 1 else 0.0)
        reproducible = bool(len(rep_grad) > 1 and grad_spread < 1.01
                            and len(set(np.sign(rep_grad))) == 1)
        rec.update(dict(loss=l0, grad=ad, abs_grad=abs(ad), best_fd=best_fd, plateau=plateau,
                        fd_sign_agree=fd_sign_agree, informative=informative, blown=blown,
                        clean=clean, peak_gb=pk, host_gb=host_gb, fwd_s=fwd_s, bwd_s=bwd_s,
                        cot_trace=trace, amp_per_step=amp, amp_median=amp_med, amp_max=amp_max,
                        amp_log_spread=amp_spread, amp_sustained=amp_sus,
                        worst_chunk=worst, oom=False,
                        amp_clean=bool(_amp.clean), amp_reason=_amp.reason,
                        repeat=len(rep_grad), rep_grad=rep_grad, rep_loss=rep_loss,
                        grad_spread=grad_spread, loss_rel_spread=loss_spread,
                        reproducible=reproducible))
        print(f"\n  N={ndays} d  freeze={knobs or 'none'}  stab={stab_cfg.describe()}", flush=True)
        print(f"  dL/dm = {ad:+.6e}   |grad|={abs(ad):.3e}   informative(<0)={informative}", flush=True)
        for h, fd in rows:
            print(f"      h={h:.0e}  FD={fd:+.6e}  rel={abs(ad-fd)/max(abs(fd),1e-300):.2e}")
        if len(rep_grad) > 1:
            print(f"  REPRODUCIBILITY over {len(rep_grad)} identical passes: "
                  f"|grad| spans {grad_spread:.3g}x, signs {sorted(set(int(np.sign(g)) for g in rep_grad))}, "
                  f"loss relative spread {loss_spread:.2e}", flush=True)
            print(f"      {'reproducible' if reproducible else 'NOT REPRODUCIBLE — a single sample of this gradient is not a measurement'}",
                  flush=True)
        print(f"  cotangent amplification: median {amp_med:.4f}/step, max {amp_max:.4f}/step "
              f"(chunk {worst}), log-spread {amp_spread:.3f}", flush=True)
        # The screen M6 runs on every training window, applied here too so the ladder and the
        # driver can never disagree about whether a seed's adjoint was usable.
        print(f"  SCREEN: {'PASS — this window is trainable' if _amp.clean else 'REJECT — ' + _amp.reason}"
              f"  (bars: median <= {adjoint_stab.AMP_CLEAN_MAX:.3f}/step, "
              f"log-spread <= {adjoint_stab.AMP_SPREAD_MAX:.3f})", flush=True)
        print(f"  peak device {pk:.1f} GB   host {host_gb:.1f} GB   fwd {fwd_s:.0f}s  bwd {bwd_s:.0f}s",
              flush=True)
    except Exception as e:
        msg = type(e).__name__ + ": " + str(e)[:300]
        is_oom = ("RESOURCE" in str(e).upper() or "memory" in str(e).lower())
        rec.update(dict(loss=None, grad=None, peak_gb=peak_gb(), oom=is_oom, clean=False, error=msg))
        print(f"\n  N={ndays} d: FAILED ({msg})", flush=True)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    tok = "OOM" if rec.get("oom") else ("CLEAN" if rec.get("clean") else "BLOWN")
    label = "-".join(knobs) if knobs else "base"
    print(f"ADJHORIZON_{args.season.upper()}_N{ndays}d_{label.upper()}_{tok}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
