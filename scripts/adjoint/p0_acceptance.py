#!/usr/bin/env python
"""GPU acceptance measurements of the CORE2 adjoint toolkit (run once per new compiled program;
the numbers are recorded in ``docs/ADJOINT_API.md``).

    python scripts/adjoint/p0_acceptance.py dot    --adjoint exact --chunks 1,6,48
    python scripts/adjoint/p0_acceptance.py guard  --adjoint exact --days 1,10
    python scripts/adjoint/p0_acceptance.py timing --save-names ssh_delta --days 1,10

Run through ``scripts/adjoint/job_gpu.sbatch`` (it sets ``PYTHONPATH`` to the checkout). Every
result is one JSONL row with the driver's provenance (:func:`core2_adj.provenance`) in
``--out/results.jsonl``.

``dot``
    The pinned-state dot test of a window map ``M`` (start carry and parameters -> end carry)
    for several chunk lengths of the same window. The chunk-entry States come from one run of
    the forward chunk program; the tangent sweep (the forward-mode chunk program, compiled
    without remat) restarts every chunk from them, and so does the reverse sweep (the chunk
    VJP, which re-runs the chunk forward inside itself). Within a chunk each of the three
    programs follows its OWN primal trajectory: they are different compiled programs, and on a
    GPU even one program is not bitwise reproducible (scatter-add order). So the mismatch of
    the dot test measures the transposition AND the difference between the trajectories the
    two sweeps linearise about; the second shrinks with the chunk length. The action records,
    per chunk length: the dot test per amplitude and repeat (``rel_norm`` =
    ``|<y,Mx> − <Mᵀy,x>| / (‖Mx‖·‖y‖)``), the repeat spread of each side on the same pinned
    States, and where the tangent program's primal departs most from the forward program's
    (field, node, level, both values), next to the same statistic for two runs of the forward
    program itself.
``guard``
    :func:`chunked_grad` with ``check_linear="full"`` over 1- and 10-day windows: the measured
    repeat spread ``s`` and the scaling error of the first reverse chunk.
``timing``
    Forward and reverse seconds per step (second repeat, compile excluded), compile seconds and
    the device's peak memory, with or without a remat policy (``--save-names``). Run one policy
    per process: the device peak is a process-wide maximum.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core2_adj as D  # noqa: E402  (runs the import guard)

import jax  # noqa: E402
import numpy as np  # noqa: E402

from fesom_jax import adjoint, adjoint_check, zarr_output  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402

FIELDS = tuple(zarr_output._all_state_fields())


def _peak_gb() -> float | None:
    try:
        st = jax.devices()[0].memory_stats() or {}
    except Exception:          # noqa: BLE001 — not every backend reports it
        return None
    v = st.get("peak_bytes_in_use")
    return None if v is None else v / 1e9


def _cfg(name: str) -> AdjointConfig:
    return (AdjointConfig.long_window() if name == "long_window" else AdjointConfig.exact())


def _norm(tree) -> float:
    return math.sqrt(max(adjoint_check.tree_vdot(tree, tree), 0.0))


def _abs_dot(a, b) -> float:
    """``Σ |a_i·b_i|`` over the floating leaves: the scale of the cancellation in ``<a, b>``.
    ``‖a‖·‖b‖`` (Cauchy-Schwarz) is far larger on a State whose fields carry different units
    (``ssh_rhs`` against T), which makes ``rel_norm`` small for any mismatch."""
    tot = 0.0
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        if adjoint_check._is_real(x) and adjoint_check._is_real(y):
            tot += float(np.sum(np.abs(np.asarray(x) * np.asarray(y))))
    return tot


def cot_field_stats(a, b=None) -> dict:
    """Per field of a State cotangent ``a``: its norm, and with ``b`` (a repeat) the norm of
    ``a − b`` and where ``|a − b|`` is largest."""
    out = {}
    for f in FIELDS:
        x = np.asarray(getattr(a, f))
        if x.dtype.kind != "f":
            continue
        e = {"norm": float(np.linalg.norm(x.ravel())), "max_abs": float(np.max(np.abs(x)))
             if x.size else 0.0}
        if b is not None:
            d = x - np.asarray(getattr(b, f))
            e["diff_norm"] = float(np.linalg.norm(d.ravel()))
            if d.size and np.any(d != 0):
                i = np.unravel_index(int(np.argmax(np.abs(d))), d.shape)
                e.update(diff_max=float(abs(d[i])), diff_index=[int(v) for v in i],
                         a=float(x[i]), b=float(np.asarray(getattr(b, f))[i]))
        out[f] = e
    return out


def _sub(a, b):
    return jax.tree.map(lambda x, y: np.asarray(x) - np.asarray(y) if np.asarray(x).dtype
                        != jax.dtypes.float0 else x, a, b)


def field_diff(a, b) -> dict:
    """Per State field: max |a − b|, its location, both values, the field's max |a|, and the
    number of differing entries (exact ``!=``) and of entries differing by more than 1e-6 of
    the field's max |a|."""
    out = {}
    for f in FIELDS:
        x, y = np.asarray(getattr(a, f)), np.asarray(getattr(b, f))
        if x.dtype.kind != "f":
            continue
        d = np.abs(x - y)
        scale = float(np.max(np.abs(x))) if x.size else 0.0
        if not d.size or not np.any(d > 0):
            out[f] = {"max": 0.0, "n_diff": 0, "scale": scale}
            continue
        i = np.unravel_index(int(np.argmax(d)), d.shape)
        out[f] = {"max": float(d[i]), "index": [int(v) for v in i], "a": float(x[i]),
                  "b": float(y[i]), "scale": scale, "rel_scale": float(d[i]) / max(scale, 1e-300),
                  "n_diff": int(np.count_nonzero(d)),
                  "n_big": int(np.count_nonzero(d > 1e-6 * scale))}
    return out


def merge_max(acc: dict, new: dict, chunk: int) -> dict:
    """Keep, per field, the entry with the largest ``max`` over chunks (tagged with its chunk)."""
    for f, v in new.items():
        if f not in acc or v["max"] > acc[f]["max"]:
            acc[f] = {**v, "chunk": chunk}
    return acc


def at_node(states: dict, node: int) -> dict:
    """Surface/ice values at one node in several States (for the largest t_skin difference)."""
    out = {}
    for name, st in states.items():
        out[name] = {f: float(np.asarray(getattr(st, f))[node]) for f in
                     ("a_ice", "m_ice", "m_snow", "t_skin", "eta_n", "d_eta")}
        out[name]["T0"] = float(np.asarray(st.T)[node, 0])
        out[name]["S0"] = float(np.asarray(st.S)[node, 0])
    return out


def top_fields(diff: dict, n: int = 6) -> list:
    return sorted(((f, v) for f, v in diff.items() if v["max"] > 0),
                  key=lambda fv: -fv[1].get("rel_scale", 0.0))[:n]


def _forward_pins(model, params, nc, start, *, compare=None):
    """Chunk-entry States (host) from the forward chunk program; with ``compare`` (an earlier
    list) also the per-field difference to it, entry by entry (the old list is released as it
    goes, so the host holds about one list)."""
    fwd = adjoint._jitted_forward(model.chunk_fn)
    xs_fn = model.xs_fn(start)
    carry = model.init_carry()
    pins = [jax.device_get(carry)]
    diff = {}
    for c in range(nc):
        xs = xs_fn(c)
        nxt = jax.block_until_ready(fwd(carry, xs, params))
        adjoint._free(xs)
        adjoint._free(carry)
        carry = nxt
        h = jax.device_get(nxt)
        if compare is not None:
            merge_max(diff, field_diff(model.state_of(compare[c + 1]), model.state_of(h)), c)
            compare[c + 1] = None
        pins.append(h)
    adjoint._free(carry)
    return pins, diff


def _tl_sweep(model, params, pins, nc, start, t_carry, t_par, *, primal_diff: bool):
    """The tangent sweep on pinned States (the program of :func:`adjoint_check.window_jvp`),
    optionally with the per-chunk difference of the tangent program's primal to the pins."""
    tl = adjoint_check._jitted_tangent(model.chunk_fn_tl,
                                       tuple(sorted(D.TL_COMPILER_OPTIONS.items())))
    xs_fn = model.xs_fn(start)
    t = adjoint_check._tangent_like(pins[0], t_carry)
    diff, worst = {}, None
    t0 = time.time()
    for c in range(nc):
        s = jax.device_put(pins[c])
        xs = xs_fn(c)
        s_next, t_next = jax.block_until_ready(tl(s, xs, params, t, t_par))
        adjoint._free(xs)
        adjoint._free(s)
        if primal_diff:
            h = jax.device_get(s_next)
            fd = field_diff(model.state_of(pins[c + 1]), model.state_of(h))
            merge_max(diff, fd, c)
            if worst is None or fd.get("t_skin", {}).get("max", 0) > worst[0]:
                worst = (fd.get("t_skin", {}).get("max", 0), c, h)
        adjoint._free(s_next)
        t = t_next
    tan = jax.device_get(t)
    extra = {}
    if primal_diff and worst is not None and worst[0] > 0:
        c = worst[1]
        node = diff["t_skin"]["index"][0]
        extra = {"t_skin_worst_chunk": c,
                 "t_skin_worst_node": at_node({"forward": model.state_of(pins[c + 1]),
                                               "tangent_program": model.state_of(worst[2]),
                                               "chunk_entry": model.state_of(pins[c])}, node)}
    return tan, diff, extra, time.time() - t0


def action_dot(args) -> None:
    jpath = Path(args.out) / "results.jsonl"
    start = D._parse_date(args.start)
    amps = [float(a) for a in args.amps.split(",")]
    for K in [int(k) for k in args.chunks.split(",")]:
        model = D.build_model(start, args.steps, K, None, _cfg(args.adjoint))
        nc = model.chunks_for(args.steps)
        params = model.params0
        loss = D.cost_fn(model, args.cost, args.box, args.steps)
        dirs = [D.direction(model, d, args.box) for d in args.dirs.split(",")]
        pins_prev = None
        for rep in range(args.repeats):
            with D._Timer() as tf:
                pins, fwd_diff = _forward_pins(model, params, nc, start, compare=pins_prev)
            pins_prev = None
            y = jax.device_get(jax.vjp(loss, jax.device_put(pins[nc]))[1](1.0)[0])
            yn = _norm(y)
            for dn in dirs:
                xs_fn = model.xs_fn(start)
                with D._Timer() as tv:
                    r1 = adjoint_check.window_vjp(model.chunk_fn, params, model.init_carry(), y,
                                                  nc, xs_fn, pinned_states=pins, chunk_steps=K,
                                                  check_linear=True, field_trace=False)
                    rhs1 = adjoint_check.tree_vdot((r1.state_cot, jax.device_get(r1.params_cot)),
                                                   (dn.carry, dn.params))
                    r2 = adjoint_check.window_vjp(model.chunk_fn, params, model.init_carry(), y,
                                                  nc, model.xs_fn(start), pinned_states=pins,
                                                  chunk_steps=K, check_linear=False,
                                                  field_trace=False)
                    rhs2 = adjoint_check.tree_vdot((r2.state_cot, jax.device_get(r2.params_cot)),
                                                   (dn.carry, dn.params))
                with D._Timer() as tt:
                    tan, pdiff, extra, tl_s = _tl_sweep(model, params, pins, nc, start, dn.carry,
                                                        dn.params, primal_diff=True)
                    tan_rep, _, _, _ = _tl_sweep(model, params, pins, nc, start, dn.carry,
                                                 dn.params, primal_diff=False)
                mx = _norm(tan)
                lhs1 = adjoint_check.tree_vdot(y, tan)
                lhs1r = adjoint_check.tree_vdot(y, tan_rep)
                tl_rep = _norm(_sub(tan, tan_rep)) / max(mx, 1e-300)
                rows_amp = []
                for a in amps:
                    if a == 1.0:
                        t_a, lhs = tan, lhs1
                    else:
                        sc = adjoint_check._scale
                        t_a, _, _, _ = _tl_sweep(model, params, pins, nc, start,
                                                 sc(dn.carry, a), sc(dn.params, a),
                                                 primal_diff=False)
                        lhs = adjoint_check.tree_vdot(y, t_a)
                    mxa = _norm(t_a)
                    rhs = a * rhs1
                    d = abs(lhs - rhs)
                    rows_amp.append({"amp": a, "lhs": lhs, "rhs": rhs,
                                     "rel": d / max(abs(lhs), abs(rhs), 1e-300),
                                     "rel_norm": d / max(mxa * yn, 1e-300),
                                     "rel_abs": d / max(_abs_dot(y, t_a), 1e-300)})
                row = D.provenance(
                    model, "p0_dot", n_steps=args.steps, start=start, direction=dn.name,
                    cost=args.cost, box=args.box, repeat=rep, direction_info=dn.info,
                    amps=rows_amp, norm_Mx=mx, norm_y=yn,
                    tl_repeat_spread_norm=abs(lhs1 - lhs1r) / max(mx * yn, 1e-300),
                    tl_repeat_rel_tangent=tl_rep,
                    adj_repeat_spread_norm=abs(rhs1 - rhs2) / max(mx * yn, 1e-300),
                    adj_repeat_spread_rel=abs(rhs1 - rhs2) / max(abs(rhs1), 1e-300),
                    adj_repeat_fields=(cot_field_stats(model.state_of(r1.state_cot),
                                                       model.state_of(r2.state_cot))
                                       if dn is dirs[0] else None),
                    adj_linear_check=r1.result.linear_check,
                    primal_tl_vs_fwd_top=top_fields(pdiff), primal_tl_vs_fwd=pdiff,
                    primal_fwd_repeat_top=top_fields(fwd_diff) if rep else None,
                    primal_fwd_repeat=fwd_diff if rep else None, **extra,
                    forward_wall_s=tf.wall, adj_wall_s=tv.wall, adj_compile_s=tv.compile_s,
                    tl_wall_s=tt.wall, tl_compile_s=tt.compile_s, tl_sweep_s=tl_s,
                    peak_gb=_peak_gb())
                D.append_row(jpath, row)
                print(f"[dot] K={K} rep={rep} dir={dn.name} "
                      + " ".join(f"amp={r['amp']:.0e}:rel_norm={r['rel_norm']:.3e}"
                                 for r in rows_amp)
                      + f" tl_rep={row['tl_repeat_spread_norm']:.2e}"
                      f" adj_rep={row['adj_repeat_spread_norm']:.2e}"
                      f" top_primal={[(f, v['max']) for f, v in row['primal_tl_vs_fwd_top'][:3]]}",
                      flush=True)
            pins_prev = pins
            del pins
        pins_prev = None
        if K == int(args.steps) and args.library_dot:
            # the driver's own path (run_dot: window_jvp / window_vjp) at the longest chunk
            for dn in args.dirs.split(","):
                for r in D.run_dot(model, args.cost, dn, box=args.box, amps=amps):
                    r["peak_gb"] = _peak_gb()
                    D.append_row(jpath, r)
                    print(f"[run_dot] K={K} dir={dn} amp={r['amp']:.0e} "
                          f"rel_norm={r['rel_norm']:.3e}", flush=True)
            if args.grad_screen_tl:
                for r in D.run_grad(model, (args.cost,), box=args.box, check_linear="full"):
                    r.pop("_result", None)
                    r["peak_gb"] = _peak_gb()
                    D.append_row(jpath, r)
                    print(f"[grad] loss={r['loss']:.10g} guard={r['linear_check']} "
                          f"fwd_s={r['forward_seconds']:.1f} rev_s={r['reverse_seconds']:.1f}",
                          flush=True)
                r = D.run_screen(model, args.cost, box=args.box)
                r["peak_gb"] = _peak_gb()
                D.append_row(jpath, r)
                print(f"[screen] status={r['status']} trace={r['trace'][:3]}...{r['trace'][-1]}",
                      flush=True)
                r = D.run_tl(model, args.cost, "T_inside", box=args.box)
                r.pop("_pinned", None)
                r["peak_gb"] = _peak_gb()
                D.append_row(jpath, r)
                print(f"[tl] dJ={r['dJ']:.6e} trace_end={r['trace'][-1]:.3e}", flush=True)
        del model


def action_vjprep(args) -> None:
    """Run-to-run reproducibility of the pieces, per State field, on a one-chunk window of
    each chunk length: the forward chunk program twice, then the chunk VJP three times on the
    same inputs (and once with the doubled cotangent). On a GPU, compare with a run under
    ``XLA_FLAGS=--xla_gpu_deterministic_ops=true``."""
    jpath = Path(args.out) / "results.jsonl"
    start = D._parse_date(args.start)
    for mode in args.modes.split(","):
        for K in [int(k) for k in args.chunks.split(",")]:
            model = D.build_model(start, K, K, None, _cfg(mode))      # a one-chunk window
            nc = 1
            params = model.params0
            loss = D.cost_fn(model, args.cost, args.box, K)
            pins, _ = _forward_pins(model, params, nc, start)
            pins2, fdiff = _forward_pins(model, params, nc, start, compare=list(pins))
            del pins2
            y = jax.device_get(jax.vjp(loss, jax.device_put(pins[nc]))[1](1.0)[0])
            bwd = adjoint._jitted_backward(model.chunk_fn)
            xs_fn = model.xs_fn(start)
            outs = []
            for rep in range(args.repeats + 1):
                a = 2.0 if rep == args.repeats else 1.0
                xs = xs_fn(nc - 1)
                g_s, g_p = jax.block_until_ready(bwd(jax.device_put(pins[nc - 1]), xs, params,
                                                     jax.device_put(adjoint_check._scale(y, a))))
                adjoint._free(xs)
                outs.append((jax.device_get(g_s), jax.device_get(g_p)))
            base_s, base_p = outs[0]
            n_s, n_p = _norm(base_s), _norm(base_p)
            reps = [{"state_rel": _norm(_sub(o[0], base_s)) / max(n_s, 1e-300),
                     "params_rel": _norm(_sub(o[1], base_p)) / max(n_p, 1e-300)}
                    for o in outs[1:args.repeats]]
            o2 = outs[args.repeats]
            lin = {"state_rel": _norm(_sub(o2[0], adjoint_check._scale(base_s, 2.0)))
                   / max(2 * n_s, 1e-300),
                   "params_rel": _norm(_sub(o2[1], adjoint_check._scale(base_p, 2.0)))
                   / max(2 * n_p, 1e-300)}
            row = D.provenance(
                model, "p0_vjprep", n_steps=K, start=start, cost=args.cost,
                box=args.box, mode=mode, forward_repeat_top=top_fields(fdiff),
                forward_repeat=fdiff, vjp_repeats=reps, vjp_scaling=lin,
                cot_fields=cot_field_stats(model.state_of(base_s),
                                           model.state_of(outs[1][0]) if args.repeats > 1
                                           else None),
                params_grads=[D._small(o[1]) for o in outs], peak_gb=_peak_gb())
            D.append_row(jpath, row)
            top = sorted(((f, v.get("diff_norm", 0.0) / max(v["norm"], 1e-300), v["norm"])
                          for f, v in row["cot_fields"].items()), key=lambda t: -t[1])[:4]
            print(f"[vjprep] mode={mode} K={K} fwd_top={[(f, v['max']) for f, v in row['forward_repeat_top'][:3]]} "
                  f"vjp_reps={reps} scaling={lin} cot_top={top}", flush=True)
            del model, pins


def action_guard(args) -> None:
    jpath = Path(args.out) / "results.jsonl"
    start = D._parse_date(args.start)
    model = D.build_model(start, 48, args.chunk_steps, None, _cfg(args.adjoint))
    for days in [int(d) for d in args.days.split(",")]:
        n = int(round(days * 86400.0 / D.DT))
        for rep in range(args.repeats):
            try:
                rows = D.run_grad(model, (args.cost,), box=args.box, n_steps=n,
                                  check_linear="full")
            except RuntimeError as exc:          # the guard itself failed: record it
                D.append_row(jpath, D.provenance(model, "p0_guard", n_steps=n, start=start,
                                                 status="guard_failed", error=str(exc)))
                print(f"[guard] days={days} rep={rep} FAILED: {exc}", flush=True)
                continue
            for r in rows:
                r.pop("_result", None)
                r["action"], r["repeat"], r["peak_gb"] = "p0_guard", rep, _peak_gb()
                D.append_row(jpath, r)
                lc = r["linear_check"]
                print(f"[guard] days={days} rep={rep} loss={r['loss']:.12g} zero_max="
                      f"{lc['zero_max']} spread={lc.get('spread')} err={lc.get('err')} "
                      f"guard_s={lc.get('seconds', 0):.1f}", flush=True)


def action_timing(args) -> None:
    jpath = Path(args.out) / "results.jsonl"
    start = D._parse_date(args.start)
    names = tuple(s for s in args.save_names.split(",") if s)
    with D._Timer() as tb:
        model = D.build_model(start, 48, args.chunk_steps, None, _cfg(args.adjoint), "prescribed",
                              names)
    for days in [int(d) for d in args.days.split(",")]:
        n = int(round(days * 86400.0 / D.DT))
        for rep in range(args.repeats):
            with D._Timer() as t:
                r = D.run_grad(model, (args.cost,), box=args.box, n_steps=n,
                               check_linear=False)[0]
            r.pop("_result", None)
            r.update(action="p0_timing", repeat=rep, peak_gb=_peak_gb(),
                     forward_s_per_step=r["forward_seconds"] / n,
                     reverse_s_per_step=r["reverse_seconds"] / n, build_s=tb.wall,
                     outer_compile_s=t.compile_s, outer_n_compiles=t.n_compiles)
            D.append_row(jpath, r)
            print(f"[timing] save_names={names} days={days} rep={rep} "
                  f"fwd/step={r['forward_s_per_step']:.3f} rev/step={r['reverse_s_per_step']:.3f}"
                  f" compile_s={t.compile_s:.1f} n_compiles={t.n_compiles} "
                  f"peak_gb={r['peak_gb']} loss={r['loss']:.12g}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["dot", "guard", "timing", "vjprep"])
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--adjoint", choices=["exact", "long_window"], default="exact")
    ap.add_argument("--cost", default="box_T_vw")
    ap.add_argument("--box", default=D.DEFAULT_BOX)
    ap.add_argument("--steps", type=int, default=48, help="dot: window length in steps")
    ap.add_argument("--chunks", default="1,6,48", help="dot: chunk lengths")
    ap.add_argument("--dirs", default="T_inside,ln_tke_c_k,ln_k_gm", help="dot: directions")
    ap.add_argument("--amps", default="1,1e-8")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--library-dot", action="store_true",
                    help="dot: also run the driver's run_dot at the longest chunk")
    ap.add_argument("--grad-screen-tl", action="store_true",
                    help="dot: also run grad (check_linear='full'), screen and tl at the "
                         "longest chunk")
    ap.add_argument("--days", default="1,10", help="guard/timing: window lengths in days")
    ap.add_argument("--chunk-steps", type=int, default=48, help="guard/timing")
    ap.add_argument("--save-names", default="", help="timing: remat policy names")
    ap.add_argument("--modes", default="exact,long_window", help="vjprep: adjoint presets")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    tag = os.environ.get("SLURM_JOB_ID") or datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    args.out = args.out or str(D.RUNS_DIR / f"p0_{args.action}_{tag}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {"dot": action_dot, "guard": action_guard, "timing": action_timing,
     "vjprep": action_vjprep}[args.action](args)
    print(f"[done] results in {args.out}/results.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
