"""Single-precision sub-plan, Task 1: does XLA keep the float32 roundings a single-precision step relies on?

fesom3-jax found (job 28274339, GH200) that XLA:GPU with its default ``--xla_allow_excess_precision=true`` folds a
convert pair f64 -> f32 -> f64 into the float64 value, so FESOM2's tracer carry ``acc - f64(f32(acc/h))*h`` became 0
and every float32 rounding of T and S went uncompensated. This script measures, on the backend it runs on, which
mechanisms keep the rounding.

Kernels (inputs chosen so every product is exact in float64 — T float32, h and h_new float32-representable — so FMA
contraction cannot change a bit and numpy is an exact reference):

    carry   acc = T*h + d;  Tn = f32(acc / h_new);  carry = f32(acc - f64(Tn)*h_new)        (FESOM2 #1206)
    feed    x = T + dT;  y = f64(f32(x))*2 - x*2      (a rounded value read by later float64 code; folded => 0)
    scan    the carry kernel 8 times in a lax.scan, Tn and the carry carried (the in-model shape)

Variants:
    a  default flags
    b  jax.jit(..., compiler_options={"xla_allow_excess_precision": False})            (the planned mechanism)
    c  XLA_FLAGS=--xla_allow_excess_precision=false for the whole process               (child process)
    d  default flags, rounding by lax.reduce_precision(x, 8, 23) on the float64 value (no convert pair)
    e  default flags, lax.optimization_barrier on the float32 value before widening

Per variant and kernel: bitwise equality with numpy, the number of zero carries / residuals (the fold's signature;
the exact reference has none), and the convert / reduce-precision instructions left in the optimised HLO. A control
checks that a float64-only program compiles to the same optimised HLO with and without the option of variant b.

    python scripts/perf/sp_convert_pairs.py --out <dir>          # runs a, b, d, e here and c in a child process
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

N = 4096
SCAN_STEPS = 8
FLAG = "--xla_allow_excess_precision=false"


def _inputs(np):
    rng = np.random.default_rng(20261020)
    T = rng.uniform(-2.0, 30.0, N).astype(np.float32).astype(np.float64)
    h = rng.uniform(1.0, 250.0, N).astype(np.float32).astype(np.float64)
    h_new = (h * rng.uniform(0.999, 1.001, N)).astype(np.float32).astype(np.float64)
    d = rng.uniform(-1e-2, 1e-2, N)                     # a full-precision float64 tendency
    dT = rng.uniform(-1e-3, 1e-3, N)
    return T, h, h_new, d, dT


def _reference(np, T, h, h_new, d, dT):
    f32 = np.float32
    acc = T * h + d
    Tn = (acc / h_new).astype(f32)
    carry = (acc - Tn.astype(np.float64) * h_new).astype(f32)
    x = T + dT
    y = x.astype(f32).astype(np.float64) * 2.0 - x * 2.0
    Ts, cs = T.astype(f32), np.zeros(N, f32)
    for _ in range(SCAN_STEPS):
        acc_s = Ts.astype(np.float64) * h + d + cs.astype(np.float64)
        Tn_s = (acc_s / h_new).astype(f32)
        cs = (acc_s - Tn_s.astype(np.float64) * h_new).astype(f32)
        Ts = Tn_s
    return {"carry_T": Tn, "carry_c": carry, "feed_y": y, "scan_T": Ts, "scan_c": cs}


def _kernels(jnp, lax, how):
    """``how``: 'pair' (convert pair), 'rp' (reduce_precision), 'barrier' (optimization_barrier on the f32 value)."""
    f32, f64 = jnp.float32, jnp.float64

    def rnd(x):                                       # float64 in -> float64 holding the float32-rounded value
        if how == "rp":
            return lax.reduce_precision(x, exponent_bits=8, mantissa_bits=23)
        y = x.astype(f32)
        if how == "barrier":
            y = lax.optimization_barrier(y)
        return y.astype(f64)

    def carry_step(Tin, h, h_new, d, c):
        acc = Tin * h + d + c
        Tn = rnd(acc / h_new)
        carry = rnd(acc - Tn * h_new)
        return Tn, carry

    def carry(T, h, h_new, d):
        Tn, c = carry_step(T, h, h_new, d, jnp.zeros_like(T))
        return Tn.astype(f32), c.astype(f32)

    def feed(T, dT):
        x = T + dT
        return rnd(x) * 2.0 - x * 2.0

    def scan(T, h, h_new, d):
        def body(st, _):
            Ts, cs = st
            Tn, c = carry_step(Ts, h, h_new, d, cs)
            return (Tn, c), None
        (Ts, cs), _ = lax.scan(body, (T, jnp.zeros_like(T)), None, length=SCAN_STEPS)
        return Ts.astype(f32), cs.astype(f32)

    return carry, feed, scan


def _hlo_counts(text):
    convert = len(re.findall(r"\bconvert\(", text))
    types = dict(re.findall(r"%([\w.\-]+) = (\w+)\[", text))     # instruction name -> element type
    f32_to_f64 = sum(1 for src in re.findall(r"= f64\[[^\]]*\](?:\{[^}]*\})? convert\(%([\w.\-]+)\)", text)
                     if types.get(src) == "f32")
    rp = len(re.findall(r"reduce-precision\(", text))
    return {"convert": convert, "convert_f32_to_f64": f32_to_f64, "reduce_precision": rp}


def run_variants(names):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from jax import lax

    T, h, h_new, d, dT = _inputs(np)
    ref = _reference(np, T, h, h_new, d, dT)
    dev = jax.devices()[0]
    out = {"backend": jax.default_backend(), "device": str(dev), "jax": jax.__version__,
           "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""), "variants": {}}
    opt = {"xla_allow_excess_precision": False}
    spec = {"a": ("pair", None), "b": ("pair", opt), "c": ("pair", None), "d": ("rp", None),
            "e": ("barrier", None)}
    args = {k: jnp.asarray(v) for k, v in dict(T=T, h=h, h_new=h_new, d=d, dT=dT).items()}
    for name in names:
        how, copts = spec[name]
        carry, feed, scan = _kernels(jnp, lax, how)
        rec = {}
        for kname, fn, fargs, keys in (
                ("carry", carry, ("T", "h", "h_new", "d"), ("carry_T", "carry_c")),
                ("feed", feed, ("T", "dT"), ("feed_y",)),
                ("scan", scan, ("T", "h", "h_new", "d"), ("scan_T", "scan_c"))):
            jf = jax.jit(fn, compiler_options=copts) if copts else jax.jit(fn)
            a = [args[k] for k in fargs]
            compiled = jf.lower(*a).compile()
            res = compiled(*a)
            res = res if isinstance(res, tuple) else (res,)
            r = {"hlo": _hlo_counts(compiled.as_text())}
            for key, val in zip(keys, res):
                got, want = np.asarray(val), ref[key]
                r[key] = {"bitwise": bool(np.array_equal(got.view(np.uint8), want.view(np.uint8))),
                          "zeros": int(np.sum(got == 0)), "ref_zeros": int(np.sum(want == 0)),
                          "max_abs_diff": float(np.max(np.abs(got.astype(np.float64)
                                                               - want.astype(np.float64))))}
            rec[kname] = r
        out["variants"][name] = rec

    # control: a float64-only program compiles to the same optimised HLO with and without variant b's option
    def f64only(T, h, h_new, d):
        acc = T * h + d
        return acc / h_new - jnp.sqrt(jnp.abs(acc)) * 1e-3
    a = [args[k] for k in ("T", "h", "h_new", "d")]
    t0 = jax.jit(f64only).lower(*a).compile().as_text()
    t1 = jax.jit(f64only, compiler_options=opt).lower(*a).compile().as_text()
    strip = lambda s: re.sub(r"metadata=\{[^}]*\}", "", s)  # noqa: E731
    out["control_f64_only_hlo_identical"] = strip(t0) == strip(t1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.child:                                       # variant c: the flag is in XLA_FLAGS for this process
        print(json.dumps(run_variants(["c"])))
        return
    res = run_variants(["a", "b", "d", "e"])
    env = dict(os.environ)
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " " + FLAG).strip()
    child = subprocess.run([sys.executable, os.path.abspath(__file__), "--out", a.out, "--child"],
                           env=env, capture_output=True, text=True)
    if child.returncode != 0:
        print(child.stderr, file=sys.stderr)
        raise SystemExit(child.returncode)
    cres = json.loads(child.stdout.strip().splitlines()[-1])
    res["variants"]["c"] = cres["variants"]["c"]
    res["XLA_FLAGS_child"] = cres["XLA_FLAGS"]
    path = os.path.join(a.out, f"sp_convert_pairs_{res['backend']}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"backend {res['backend']} ({res['device']}), jax {res['jax']}; f64-only HLO identical with option: "
          f"{res['control_f64_only_hlo_identical']}")
    print(f"{'var':3} {'kernel':6} {'field':8} {'bitwise':7} {'zeros':>6} {'ref0':>5} {'max|d|':>10}  hlo convert / "
          f"f32->f64 / reduce-precision")
    for v in "abcde":
        for k, r in res["variants"][v].items():
            hc = r["hlo"]
            for key, s in r.items():
                if key == "hlo":
                    continue
                print(f"{v:3} {k:6} {key:8} {str(s['bitwise']):7} {s['zeros']:6d} {s['ref_zeros']:5d} "
                      f"{s['max_abs_diff']:10.3e}  {hc['convert']} / {hc['convert_f32_to_f64']} / "
                      f"{hc['reduce_precision']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
