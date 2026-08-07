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
        def mk_conv(orig):
            def w(mesh, Kv, Av, bvfreq):
                out = orig(mesh, Kv, Av, bvfreq)
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


def load_or_build_truth(cache, one_day, st0, stack_steps, wet3d, area3d, m_true,
                        recompute, nd_truth=NDAYS_TRUTH):
    """The 10-day truth marks + per-level weights, in horizon_scan's exact format/location
    so the two scripts share one cache and produce comparable losses."""
    if cache.exists() and not recompute:
        z = np.load(cache)
        marks = [(z["T"][d], z["S"][d], z["eta"][d]) for d in range(nd_truth)]
        return marks, jnp.asarray(z["w_T"]), jnp.asarray(z["w_S"]), jnp.asarray(z["w_eta"])
    marks, s = [], st0
    for d in range(nd_truth):
        sfs_day = stack_steps(d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY)
        s = one_day(s, sfs_day, jnp.asarray(m_true, jnp.float64))
        marks.append(jax.device_get((s.T, s.S, s.eta_n)))
        jax.tree.map(lambda x: x.delete() if hasattr(x, "delete") else None, sfs_day)
    del s
    T_st = jnp.asarray(np.stack([m[0] for m in marks]))
    S_st = jnp.asarray(np.stack([m[1] for m in marks]))
    e_st = jnp.asarray(np.stack([m[2] for m in marks]))
    w_T, w_S, w_eta = level_weights(T_st, S_st, e_st, wet3d, area3d)
    w_T, w_S, w_eta = (jax.block_until_ready(x) for x in (w_T, w_S, w_eta))
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez(tmp, T=np.stack([m[0] for m in marks]), S=np.stack([m[1] for m in marks]),
             eta=np.stack([m[2] for m in marks]), w_T=np.asarray(w_T), w_S=np.asarray(w_S),
             w_eta=np.asarray(w_eta))
    tmp.rename(cache)
    for a in (T_st, S_st, e_st):
        a.delete()
    return marks, w_T, w_S, w_eta


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="freeze targets:\n" + "\n".join(f"  {k:9s} {v}" for k, v in FREEZE_HELP.items()))
    ap.add_argument("--snap", required=True)
    ap.add_argument("--season", default="seed")
    ap.add_argument("--n-days", type=int, required=True, help="window length in days (<= --truth-days)")
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

    cache = args.truth_cache or (Path("/work/ab0995/a270088/nn_eddy/horizon_truth") /
                                 f"{Path(args.snap).stem}_m{args.truth_mult:g}_{nd_truth}d.npz")
    t1 = time.time()
    truth_marks, w_T, w_S, w_eta = load_or_build_truth(
        cache, one_day, st0, stack_steps, wet3d, area3d, args.truth_mult,
        args.recompute_truth, nd_truth)
    print(f"[truth] {nd_truth}-day truth ready in {time.time()-t1:.1f}s ({cache})", flush=True)
    T_hat, S_hat, e_hat = (jnp.asarray(x) for x in truth_marks[ndays - 1])
    del truth_marks

    def misfit_of(s):
        return stage1_misfit(s.T, s.S, s.eta_n, T_hat, S_hat, e_hat,
                             w_T, w_S, w_eta, wet3d, area3d)

    sfs_chunks = [stack_steps(c * K, (c + 1) * K) for c in range(n_chunks)]

    def forward(m, keep_boundaries=False):
        """Run the window; optionally return the chunk-boundary states, on the HOST."""
        s = st0
        kept = [jax.device_get(s)] if keep_boundaries else None
        for c in range(n_chunks):
            s = fwd_chunk(s, sfs_chunks[c], m)
            if keep_boundaries:
                kept.append(jax.device_get(s))
        return s, kept

    m0 = jnp.asarray(1.0, jnp.float64)
    rec = {"season": args.season, "snap": Path(args.snap).name, "date": f"{yr}-{mo:02d}-{dy:02d}",
           "n_days": ndays, "n_steps": n, "dt": DT, "chunk_steps": K, "n_chunks": n_chunks,
           "marks": "end", "m_true": args.truth_mult, "kgm0": kgm0,
           "config": f"zstar+{args.mixing.upper()}+mEVP+GM", "adjoint": f"{args.ice}-ice",
           "freeze": knobs, "gamma": args.gamma, "filter_factor": args.filter_factor,
           "clip": args.clip, "stab": stab_cfg.describe(), "tag": args.tag}
    try:
        with patched(knobs):
            def one_pass(verbose):
                """One full forward + chunked reverse accumulation. Returns (loss, dL/dm,
                cotangent-norm trace, host GB, fwd seconds, bwd seconds)."""
                ta = time.time()
                s_fin, boundaries = forward(m0, keep_boundaries=True)
                loss_v = float(misfit_of(s_fin))
                fwd_t = time.time() - ta
                hgb = sum(a.nbytes for a in jax.tree.leaves(boundaries)) / 1e9
                if verbose:
                    print(f"[fwd] loss(m=1)={loss_v:.6e}  {fwd_t:.1f}s  "
                          f"boundaries on host: {hgb:.1f} GB", flush=True)
                _, vjp_L = jax.vjp(misfit_of, s_fin)
                c_t = vjp_L(1.0)[0]
                del s_fin
                tr = [float(adjoint_stab.cotangent_norm(c_t))]
                tb = time.time()
                g = jnp.asarray(0.0, jnp.float64)
                for c in reversed(range(n_chunks)):
                    s_in = jax.device_put(boundaries[c])
                    g_s, g_c = bwd_chunk(s_in, sfs_chunks[c], m0, c_t)
                    g = g + g_c
                    c_t = g_s
                    del s_in
                    tr.append(float(adjoint_stab.cotangent_norm(c_t)))
                    if verbose:
                        print(f"    [bwd] chunk {c:3d}  t=+{c*K*DT/86400:5.2f} d  "
                              f"|cot|={tr[-1]:.6e}  dL/dm so far={float(g):+.6e}", flush=True)
                return loss_v, float(g), tr, hgb, fwd_t, time.time() - tb

            l0, ad, trace, host_gb, fwd_s, bwd_s = one_pass(verbose=True)
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
            amp = [(trace[i + 1] / trace[i]) ** (1.0 / K) if trace[i] > 0 else float("nan")
                   for i in range(len(trace) - 1)]
            amp_fin = [a for a in amp if np.isfinite(a)]
            amp_med = float(np.median(amp_fin)) if amp_fin else float("nan")
            amp_max = float(np.max(amp_fin)) if amp_fin else float("nan")
            # spread of log-amplification across chunks: near 0 => smooth exponential growth
            # (chaos); large => the growth lives in a few chunks (a switch is firing)
            amp_spread = (float(np.std(np.log(amp_fin))) if len(amp_fin) > 1 else 0.0)
            worst = int(n_chunks - 1 - int(np.argmax(amp))) if amp_fin else -1

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
                        amp_log_spread=amp_spread, worst_chunk=worst, oom=False,
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
        print(f"  {'CONCENTRATED — a switch is firing, not chaos' if amp_spread > 0.35 else 'SMOOTH — consistent with a genuine Lyapunov exponent'}", flush=True)
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
