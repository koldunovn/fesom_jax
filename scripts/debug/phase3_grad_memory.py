"""Task 3.1 GPU gate — the N=200 backward-pass **memory sanity**.

Reverse-mode through an N-step time loop must store enough to run the backward pass.
Without rematerialization that is O(N · all per-step intermediates) → it blows up with
N. With ``jax.checkpoint`` on the scan body it is O(N · carry) (the ``State`` pytree)
+ O(1) working set (each step's internals are recomputed) — which fits an A100.

This script runs ``jax.grad`` of mean-SST-after-N-steps w.r.t. the PP ``k_ver`` through
:func:`fesom_jax.integrate.integrate` and reports the peak device memory, for
``checkpoint`` on (the gate: must fit + finite) and — best effort — off (expected to
use far more / OOM, demonstrating checkpointing is load-bearing).

Usage (one fresh process per mode so peak-memory stats are clean):
    python scripts/debug/phase3_grad_memory.py --n 200 --checkpoint
    python scripts/debug/phase3_grad_memory.py --n 200 --no-checkpoint
Submit on a GPU node via scripts/debug/phase3_grad_memory.sbatch.
"""

import argparse
import time

import fesom_jax  # enables x64 + proves the install
import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import forcing, ic, ssh
from fesom_jax.integrate import integrate
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params

DT = 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--checkpoint", dest="checkpoint", action="store_true", default=True)
    ap.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    args = ap.parse_args()

    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(f"N={args.n} checkpoint={args.checkpoint} dt={DT}")

    mesh = load_mesh()
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress = forcing.surface_stress(mesh)
    st0 = ic.initial_state(mesh)
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
    nwet0 = jnp.sum(wet0)

    def loss(kver):
        p = Params(k_ver=kver, a_ver=jnp.asarray(1e-4, jnp.float64))
        fin = integrate(st0, mesh, op, stress, n_steps=args.n, params=p, dt=DT,
                        checkpoint=args.checkpoint)
        return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / nwet0

    grad_fn = jax.jit(jax.grad(loss))
    k0 = jnp.asarray(1e-5, jnp.float64)

    try:
        t0 = time.time()
        g = grad_fn(k0)
        g.block_until_ready()
        dt_s = time.time() - t0
        gval = float(g)
        print(f"RESULT grad={gval:.8e} finite={np.isfinite(gval)} "
              f"compile+run={dt_s:.1f}s")
        ok = np.isfinite(gval) and gval != 0.0
    except Exception as exc:  # OOM (RESOURCE_EXHAUSTED) etc.
        print(f"RESULT FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        ok = False

    for d in jax.devices():
        try:
            st = d.memory_stats() or {}
            pk = st.get("peak_bytes_in_use")
            lim = st.get("bytes_limit")
            if pk:
                msg = f"  {d}: peak {pk / 1e9:.2f} GB"
                if lim:
                    msg += f" / limit {lim / 1e9:.1f} GB ({100 * pk / lim:.0f}%)"
                print(msg)
        except Exception:
            pass

    print("MEMORY_GATE_OK" if ok else "MEMORY_GATE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
