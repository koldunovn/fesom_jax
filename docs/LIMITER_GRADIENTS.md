# Limiter gradients — the Zalesak FCT AD decision (Task 4.1)

The FCT (flux-corrected transport) tracer advection adds a high-order (HO)
antidiffusive flux on top of the monotone low-order (LO) upwind flux, then runs the
**Zalesak limiter** to clip the antidiffusive part so the update introduces no new
local extrema. The limiter is the new **AD-hard** kernel of Phase 4 — the same risk
class as the CG `custom_linear_solve` (now retired): it is built from non-smooth
`min`/`max`/sign-select operations, and we must decide how reverse-mode AD treats
those kinks. This file records the decision, the reasoning, and the empirical check.

It is the analogue of the per-substep gates: a kernel that the smoke test
(`test_gradient.py`) must keep **finite and FD-consistent where smooth**.

## Where the kinks are (`zalesak_limit`, `fesom_tracer_adv.py`)

Every non-smooth op in the limiter, by stage (`oce_tra_adv_fct`,
`fesom_tracer_adv.c:851`):

| stage | op | what it does |
|---|---|---|
| a1 | `max(LO,T)`, `min(LO,T)` | per-node admissible window |
| a2 | `max`/`min` over 3 vertices | per-element bounds |
| a3 | `segment_max`/`segment_min` over a node's cells | per-node cluster bounds |
| a4 | `max`/`min` over 3 vertical neighbours | admissible increment vs LO |
| b1 | `max(0,·)` / `min(0,·)` (ReLU-like) | split antidiffusive flux into ± sums |
| b2 | `min(1, ratio)` | per-node limiter factor in [0,1] |
| b3 | `min(a,b)`, `f≥0 ? · : ·` (sign-select) | combine factors at faces |

All are piecewise-linear / selection ops with a **well-defined subgradient almost
everywhere** (the kink set has measure zero). `jnp.maximum`/`minimum`/`where` and
`jax.ops.segment_max`/`segment_min` all carry the standard VJP that routes the
cotangent to the selected branch / arg-extremum.

## The options (from the plan)

- **(a) subgradient as-is** — let JAX differentiate the limiter as written. The VJP is
  the generalized (sub)gradient; at a kink JAX picks one side (a measure-zero event).
- **(b) smooth min/max relaxation** — replace `min`/`max` with soft (e.g.
  log-sum-exp) surrogates, with a relaxation temperature `β`.
- **(c) `stop_gradient` on the limiter coefficients** — treat the per-node factors
  `fct_plus`/`fct_minus` (the `ae` applied to each flux) as **constants** in the
  backward pass: `adf_limited = stop_gradient(ae) · adf`. Gradient flows through the
  flux magnitude but not through the limiter's field-dependence.

## Decision: **(a) subgradient as-is**

Rationale, in priority order:

1. **It satisfies the gate.** The plan requires the gradient be *finite and
   FD-consistent where smooth*. Away from a kink the limiter is locally affine, so
   AD == FD exactly — option (a) is the only choice that is FD-consistent *by
   construction* wherever the function is smooth. (b) changes the forward value
   (below) and (c) deliberately drops a real term, so neither is FD-consistent where
   the limiter is active and field-dependent.

2. **It is NaN-safe here** — the reason the CG-class risk does **not** materialize.
   The only division in the limiter is the b2 ratio
   `fct_ttf_max / (fct_plus·dt/areasvol/hnode_new + flux_eps)`. The C's
   `flux_eps = 1e-16` (kept verbatim) floors the denominator strictly positive
   (`fct_plus ≥ 0`), and the negative branch is floored strictly negative
   (`− flux_eps`, `fct_minus ≤ 0`). So **every ratio is finite**, and when the
   antidiffusive flux vanishes the ratio is large-but-finite and `min(1, ratio)`
   selects the constant `1` with a **0 cotangent on a finite value** — no `0·inf`.
   The `±bignumber = ±1e3` padding in a2 is finite, so the `max`/`min`/`segment_*`
   reductions never see an inf. This is the same masked-finite discipline as the eos
   `bvfreq` and `tracer_diff` `1/zdiff` traps, here satisfied for free by `flux_eps`.
   *(Contrast the CG: there the gradient genuinely needed a separate tight
   `transpose_solve`. The limiter needs no such machinery — the forward `flux_eps` is
   the whole fix.)*

3. **It is faithful** — the forward stays bit-for-bit the dump-matching C FCT (the
   ≤1e-12 gate), because (a) touches only the backward pass.

4. **It is idiomatic** — exactly what `jax.grad` produces with no extra code, no
   hand-written `custom_vjp`, no free parameter to tune.

### Why not (b)

A smooth relaxation changes the **forward** value (it is no longer the Zalesak
limiter), breaking the ≤1e-12 forward dump gate — unless implemented as a
backward-only `custom_vjp`, which then introduces an arbitrary temperature `β` (a
bias/variance knob with no physical meaning) for no benefit: the kinks here are not a
NaN source (point 2), so there is nothing to smooth away. Rejected.

### Why not (c)

`stop_gradient(ae)` yields a **biased** gradient: it discards
`∂(ae)/∂(field)`, the sensitivity of *how much* antidiffusion is applied to the
tracer/velocity field. Wherever the limiter is active and field-dependent this makes
AD disagree with FD even in locally-smooth regions — directly failing the
"FD-consistent where smooth" gate. It also throws away real signal that hybrid-ML
training may need (an NN parameterization's effect on tracers is partly mediated by
its effect on the limiting). Rejected as the default.

**Fallback note.** (c) remains a one-line escape hatch
(`adf = jax.lax.stop_gradient(ae) * adf` inside `zalesak_limit`) if a future training
run shows the limiter's piecewise gradient injects harmful high-frequency noise into
the parameter gradient. That is a *training-stability* call to make with evidence, not
a default; the per-step gradient is correct as a subgradient, and the smooth-regime
restriction already in `test_gradient.py` keeps the smoke test off the kinks.

## Empirical check (`test_gradient.py`, FCT active)

The permanent AD gate is re-run with FCT wired into `step` (so the limiter runs every
step of the N-step window):

- `d(mean SST)/d(k_ver)` AD↔FD step-size sweep — **plateau 5.7e-07** at the
  signal-lifted `k_ver=1e-4` (≪ 1e-4 gate); finite, correct sign at the physical
  `k_ver=1e-5`. (Essentially unchanged from the Phase-3 upwind plateau 5.9e-7 — the
  limiter is inactive in the smooth blob region, so the subgradient matches FD there.)
- `d(loss)/d(a_ver)` flows through the CG **and** the FCT limiter — finite, FD-consistent.
- `d(loss)/d(T₀)` (the IC field, the masked-NaN probe) — **finite everywhere**, 0 on
  below-bottom lanes, nonzero on wet (no NaN from the limiter or the qr4c Z-stencil
  safe-divides).
- smooth-regime certification holds (blob column stratified, `S=35`).

The blob probe column stays smooth, so the FD plateau is undamaged by the limiter
(its active set does not flip under the FD perturbation at a generic point). This
confirms (a) is finite and FD-consistent where smooth on the assembled model — the
Task-4.1 AD-hard item is retired the same way the CG was.
