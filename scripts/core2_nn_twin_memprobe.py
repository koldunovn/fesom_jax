"""Compile-only memory probe for the §3 NN-twin backward (E1 OOM diagnosis).

The NN-twin grad OOMs on an 80 GB A100 even at N=6 (a 3-h window). The OOM-vs-N data
decomposes the peak as  ~0.85 GiB/step (the stored scan carries) + ~34 GiB fixed
(ONE step's reverse-sweep working set) — so trajectory length is NOT the wall; the
single-step VJP tape is. This probe MEASURES that, cheaply: it builds the exact twin
loss (with a DUMMY zero truth — no 375s forward) and reads ``Compiled.memory_analysis()``
for jit(loss) [forward] vs jit(grad(loss)) [backward]. ``memory_analysis`` is the
XLA-computed peak from buffer assignment — NO execution, so a compile-only GPU slot
(or even CPU, for the forward/backward RATIO + which-block) suffices.

``--remat`` toggles the nested in-step checkpointing fix (env STEP_REMAT) so the fix
can be validated by re-measuring the backward peak BEFORE paying for a full GPU run.

Usage:  python scripts/core2_nn_twin_memprobe.py --config tkegm --n 6 [--remat blocks]
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def gib(b):
    return b / (1024.0 ** 3)


def analyze(label, fn, *args):
    t0 = time.time()
    lowered = fn.lower(*args)
    compiled = lowered.compile()
    m = compiled.memory_analysis()
    dt = time.time() - t0
    temp = getattr(m, "temp_size_in_bytes", 0)
    arg = getattr(m, "argument_size_in_bytes", 0)
    out = getattr(m, "output_size_in_bytes", 0)
    alias = getattr(m, "alias_size_in_bytes", 0)
    gen = getattr(m, "generated_code_size_in_bytes", 0)
    peak = temp + arg + out - alias
    print(f"[{label}] peak≈{gib(peak):6.2f} GiB  (temp={gib(temp):.2f}  arg={gib(arg):.2f}  "
          f"out={gib(out):.2f}  alias={gib(alias):.2f}  code={gib(gen):.2f})  compiled in {dt:.1f}s",
          flush=True)
    return peak, temp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tkegm", choices=["all3", "tkegm"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--remat", default="off",
                    help="off | blocks (nested in-step checkpointing; sets STEP_REMAT)")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--dump", default="", help="dir for XLA HLO dump → report largest buffers")
    args = ap.parse_args()
    remat_blocks = (args.remat != "off")

    import jax
    import jax.numpy as jnp
    import fesom_jax  # noqa: F401  (x64)
    from fesom_jax import calibrate, tke_nn
    from fesom_jax.integrate import integrate
    import core2_paper_nn_twin as twin               # reuse build() + the exact loss recipe

    print(f"=== memprobe config={args.config} N={args.n} remat={args.remat} "
          f"platform={jax.devices()[0].platform} ===", flush=True)

    DT = twin.DT
    mesh, state, op, fs, sfs, sf_last, cfgs = twin.build(args.year, args.n, args.config)
    nl = mesh.nl
    node_mask = jnp.asarray(mesh.node_layer_mask)
    w3d = jnp.asarray(mesh.area[:, 0])[:, None] * node_mask
    print(f"[mesh] nod2D={mesh.nod2D} nl={nl}  3D-nodal field={gib(mesh.nod2D*nl*8):.3f} GiB (f64)",
          flush=True)

    def model_ts(nn):
        p = calibrate.build_params({"tke_nn": nn})
        fin = integrate(state, mesh, op, None, n_steps=args.n, dt=DT, step_forcings=sfs,
                        forcing_static=fs, params=p, remat_blocks=remat_blocks, **cfgs)
        return fin.T, fin.S

    # DUMMY truth (zeros) — memory is independent of the constant target, so skip the forward.
    truth_T = jnp.zeros_like(state.T)
    truth_S = jnp.zeros_like(state.S)

    def wmse(a, b):
        d = a - b
        return jnp.sum(w3d * d * d) / jnp.sum(w3d)

    def loss(d):
        T, S = model_ts(d["tke_nn"])
        return wmse(T, truth_T) + wmse(S, truth_S)

    trainee0 = tke_nn.init_tke_nn(jax.random.PRNGKey(123), zero_last=True)
    init = {"tke_nn": trainee0}

    analyze("forward ", jax.jit(loss), init)
    peak, _ = analyze("backward", jax.jit(jax.grad(loss)), init)

    if args.dump:
        # Largest individual buffers in the backward (the "what" behind the peak). XLA writes a
        # buffer-assignment text dump; we surface its top entries. Recompile under the dump flag.
        import glob
        import os
        import re
        ddir = args.dump
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                                   f" --xla_dump_to={ddir} --xla_dump_hlo_as_text").strip()
        # the env flag must be set before the backend reparses; do it via a fresh compile of a
        # trivially-different fn so XLA re-emits. Simplest: just report from the optimized HLO text.
        compiled = jax.jit(jax.grad(loss)).lower(init).compile()
        txt = compiled.as_text()
        # collect array shapes that appear, size them, print the largest unique buffers
        shapes = re.findall(r"f64\[([0-9,]+)\]", txt)
        sizes = {}
        for s in set(shapes):
            dims = [int(x) for x in s.split(",") if x]
            nel = 1
            for d in dims:
                nel *= d
            sizes[s] = nel * 8
        top = sorted(sizes.items(), key=lambda kv: -kv[1])[:15]
        print(f"\n[dump] top f64 buffer shapes in backward HLO (peak≈{gib(peak):.1f} GiB):", flush=True)
        for s, b in top:
            print(f"   f64[{s}]  {gib(b)*1024:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
