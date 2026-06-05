# fesom-jax — a differentiable FESOM2 ocean model in JAX

Port of the [FESOM2](https://fesom.de) unstructured-grid finite-volume ocean
model to **JAX**, producing a **differentiable** forward model whose purpose is
**hybrid ML**: embedding trainable neural-network parameterizations (vertical
mixing, mesoscale eddy fluxes) and training them end-to-end through the dynamics.

## Status

Early development. See the roadmap and per-task verification gates in
[`docs/plans/20260605-fesom-jax-port.md`](docs/plans/20260605-fesom-jax-port.md)
— the authoritative source of truth for this work.

## Sources of truth

| What | Where | Role |
|------|-------|------|
| C port | `/home/a/a270088/port2/fesom2_port/src/` | **algorithmic** source — mirror it kernel-by-kernel |
| Fortran model | `/home/a/a270088/port2/fesom2/src/` | **numerical** reference — produces per-substep dumps |
| Physics/algorithm | `/home/a/a270088/port2/FRESH_START.md` | timestep sequence, ALE, EOS, mixing, SSH, FCT |
| Kokkos port | `/home/a/a270088/port_kokkos/` | parallelization + fidelity lessons + compare scripts |

## Key principles

- **Golden Rule (JAX-adapted):** preserve the *exact* computation — the math and
  the load-bearing association order — but express it as vectorized array ops.
  No literal loop-by-loop translation; no physics simplification.
- **Fidelity target:** climate-close, not bit-identical (scatters/reductions
  reassociate FP sums): ~1e-15 for map/gather kernels, ~1e-12 for
  scatter/reduction kernels. This does not hurt AD.
- **AD-safe by construction:** pure functional JAX, float64 everywhere
  (`jax_enable_x64`), `lax.scan`/`cond`/`while` instead of Python control flow on
  traced values; an early end-to-end gradient check, re-run at every gate.

## Setup

See [`docs/ENV.md`](docs/ENV.md) for the exact environment (JAX install, GPU
backend, recorded versions).

## Tests / verification

The verification ladder (see the plan) is reference-comparison-driven against the
Fortran per-substep dumps:

```bash
pytest fesom_jax/tests/            # all gates
pytest fesom_jax/tests/ -k verify  # per-substep probe-column diffs
pytest fesom_jax/tests/ -k gradient  # AD checks (Phase 3+)
```
