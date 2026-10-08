# The public adjoint API — design

Task 3 of `docs/plans/20260810-split-nn-eddy-and-promote-adjoint.md`. What `fesom_jax.adjoint`
must offer so the research code can stop importing `fesom_jax.integrate._run_steps`, and why the
surface is shaped the way it is. No code in this task; Task 4 implements it and carries the tests.

(The additions of 2026-09 — exact tangent-linear SSH solve, linearity guard, multi-cost sweep,
named cuts, `adjoint_check`, `adjoint_costs`, the CORE2 driver and what they measured on the
A100 — are in the second part of this document, "Additions of 2026-09".)

The mechanism itself is not in question — it is measured and written up in `ADJOINT_HORIZON.md`.
What is in question is where the seam goes, because right now there is none: chunked reverse
accumulation exists only as six closures inside `main()` of a 949-line script, closing over the
mesh, the SSH operator, the forcing builder, the freeze knobs, the stabilizer and the multiplier.

## The surface

Three functions and one protocol. Everything else stays where it is.

```
fesom_jax.adjoint
    scan_steps(step_fn, state, xs, *, rematerialize=True) -> state
    chunked_grad(chunk_fn, params, *, n_chunks, xs_fn, loss_fn, init_state,
                 chunk_steps, boundary_stride=1, checkpoint=None) -> ChunkedGrad
    ChunkedGrad(loss, grad, trace, host_gb, forward_seconds, reverse_seconds, ...)
    ReverseCheckpoint            # a Protocol, not an implementation
```

### `scan_steps` — the in-chunk runner

`step_fn(state, x) -> state`, over `xs` stacked on a leading axis. Returns the final state.

Two things distinguish it from `_run_steps`:

**The body contract is `(state, x) -> state`, not `(carry, x) -> (carry, None)`.** The `None` is
an artefact of `lax.scan`'s signature and has no business in an API. Ten call sites currently
write the adapter by hand; the library writes it once.

**There is no `segments` parameter.** `rematerialize=True` selects two-level √N rematerialization
with the segment count derived internally from `len(xs)` — the same `max(2, round(√N))` that all
five checkpointing call sites compute for themselves today, one of them by copying the formula
into a comment that calls it "the knob's correct scope". `rematerialize=False` is a plain scan,
which is what the five forward-only and forward-mode call sites pass as `checkpoint=False`.

### `chunked_grad` — the mechanism worth publishing

The caller supplies:

| what | why it is the caller's |
|---|---|
| `chunk_fn(state, xs, params) -> state` | everything model-specific — mesh, SSH operator, config, freeze knobs, stabilizer, how `params` becomes `Params` — lives in this closure and nothing else needs to know about it |
| `xs_fn(c) -> xs` | per-chunk forcing, built ON DEMAND. Not a list. See "why this is an argument and not a list" below |
| `loss_fn(final_state) -> scalar` | the library differentiates it to seed the sweep and never looks inside |
| `init_state`, `params`, `n_chunks`, `chunk_steps` | the window |
| `boundary_stride` | host-memory / recompute trade, default 1 |
| `checkpoint` | optional `ReverseCheckpoint`; `None` runs to completion in one process |

The library supplies:

- the forward over `n_chunks`, with chunk-boundary states parked on the **host** and the device
  peak equal to one chunk regardless of window length;
- second-level checkpointing: keep every `boundary_stride`-th boundary, recompute the spans
  between from the nearest kept one, for exactly one extra forward pass over the window;
- the reverse walk — a jitted `jax.vjp` over one chunk at a time, carrying the state cotangent and
  accumulating the parameter cotangent with `jax.tree.map`;
- the **buffer lifetimes**, which are a correctness hazard rather than an optimisation: the VJP's
  output must be blocked on before its inputs are freed, because `s_in`, `xs_c` and the previous
  cotangent are all live until it completes. The reference code carries a ⚠️ on that line. A
  second implementation would get it wrong, and freeing a live donated buffer does not fail
  loudly;
- the boundary-norm trace, ready for `adjoint_stab.amplification`.

`params` is a pytree, not a scalar. The reference implementation accumulates `g = g + g_c` on a
float64 scalar multiplier; the library must `jax.tree.map(add)` so a network's parameters work.
That is the whole point of the promotion — the NN-GM training loop differentiates a `gm_nn`
pytree, not a κ multiplier.

**Why `xs_fn` is a function and not a list.** Because building all the chunks' forcing up front
and holding it on device is what made the device peak grow with window length: 43.1 GB at 1 day
rising to 56.9 GB at 30, about 0.5 GB per simulated day of pure idle residency, extrapolating to
an OOM on an 80 GB card near 90 days. Building per chunk and freeing after `block_until_ready`
made the peak flat — 43.1 / 43.2 / 43.1 GB at 1 / 30 / 60 days — with bit-identical results,
because only when the arrays exist changed. An API that accepts a list invites the version that
does not scale, and it reads as the more natural one. It is not offered.

### `ChunkedGrad` — what comes back

`loss`, `grad` (pytree matching `params`), `trace`, `host_gb`, `forward_seconds`,
`reverse_seconds`, plus the window description (`n_chunks`, `chunk_steps`, `boundary_stride`) so a
result row can be reproduced from the row. The last point is not decoration: the 180-day ladder
record did not say it ran at boundary-stride 2 and the year rung did not say it ran at 19, and
resolving that confound cost two five-hour jobs.

**`trace` is returned, not screened.** It is the chunk-boundary cotangent norm, seeded with the
loss cotangent, so `len(trace) == n_chunks + 1`, and it is exactly what
`adjoint_stab.amplification(trace, chunk_steps)` consumes. The library produces it because the
reverse sweep produces it anyway; it does not decide what passes, because the bars (1.010/step
median, 0.020 log-spread) are the research project's, not the library's.

### `ReverseCheckpoint` — a protocol the library calls and does not implement

A year-long gradient is ~10 h of one A100 against a 12 h queue limit, and two years fit no slot at
all. What makes that tractable is that the reverse loop's entire carried state is one cotangent
`State` (1.93 GB), a scalar and an index — so the sweep can stop and resume. Without this in the
library, `adjoint_horizon.py` cannot move onto `chunked_grad` (Task 5) without losing the
capability that makes long windows reachable.

But the implementation knows about `/work` paths, a SLURM exit code, and a wall-clock budget.
None of that belongs in a library. So the library defines the protocol and calls it; the research
repository keeps `adjoint_ckpt.Checkpoint`, which already has exactly these methods:

```
have_boundaries -> bool          loss -> float
forward_resume(like)             -> (state | None, first_chunk)
put_boundary(c, state)           -> bool   # True => stop now, leg incomplete
load_boundaries(like)            -> dict
finish_forward(loss)             -> None
reverse_start(seed_cot, n_chunks, like) -> (cot, grad0, start_chunk, trace)
tick(next_chunk, cot, grad, trace)      -> bool   # True => stop now
```

Two signature changes are needed and both follow from `params` becoming a pytree:
`reverse_start` returns `grad0` and `tick` takes `grad`, and both are typed `float` today.

⚠️ **This is scope the plan's Task 4 does not list.** Recorded here rather than discovered at
Task 5.

## `_run_steps` stays private, and this is the reasoning

Not promoted as-is, for two independent reasons.

**Its contract would become public API.** `body: (carry, x) -> (carry, None)` over a `State`
pytree publishes both the scan artefact and the pytree's role as a scan carry.

**Its headline parameter cannot do what it looks like it does.** `segments` reads as the knob you
reach for when a window does not fit, and reaching for it makes things worse. The scheme stores
`S + N/S` carries, minimised at `S=√N`; a CORE2 `State` is 1.93 GB, so a 5-day window (N=240)
costs ≥ 2√240 × 1.93 ≈ 60 GB of carries plus ~18 GB of reverse working set against ~78 GB usable
on an 80 GB A100. Measured: `S=31` died at 69.24 GiB (job 26798871). A public function whose main
parameter is a trap is a hazard, not an API. `scan_steps` keeps the mechanism and drops the knob.

`_run_steps` therefore stays in `fesom_jax/integrate.py`, private, with a comment naming
`scan_steps` and `chunked_grad`. `integrate()` keeps using it.

⚠️ Its docstring currently quotes "43.1 GB at 1 day and 56.9 GB at 30 days" for chunked reverse
accumulation. That is the PRE-fix measurement, from before the per-chunk forcing lifetime was
fixed; the figure is now flat at 43.1 / 43.2 / 43.1. Fix it when the comment is added.

## The call sites, checked against this surface

Ten `_run_steps` calls move with the research code. Five of them do not need the adjoint at all.

| call site | today | becomes |
|---|---|---|
| `adjoint_horizon.py:592` (`run_chunk`) | `checkpoint=True, segments=√K` | `chunk_fn` body via `scan_steps` |
| `adjoint_horizon.py:615` (`one_day`, truth) | `checkpoint=True, segments=√48` | `scan_steps` |
| `horizon_scan.py:236` (`one_day`, truth) | `checkpoint=True, segments=√48` | `scan_steps` |
| `horizon_scan.py:305` (monolithic loss) | `checkpoint=True, segments=√n` | `scan_steps` — this is the superseded path, kept working, not fixed |
| `horizon_scan.py:317` (in-graph day loop) | `checkpoint=True, segments=√48` | `scan_steps` |
| `seam_gate.py:307` (`one_day`, stability arm) | `checkpoint=False` | `scan_steps(rematerialize=False)` |
| `seam_gate.py:390` (`run_chunk`) | `checkpoint=True, segments=√K` | `chunk_fn` body via `scan_steps` |
| `seam_gate.py:403` (`run_chunk_nockpt`, JVP) | `checkpoint=False` | `scan_steps(rematerialize=False)` |
| `student_year.py:146` (`chunk`) | `checkpoint=False` | `scan_steps(rematerialize=False)` |
| `student_year.py:156` (`first_chunk`) | `checkpoint=False` | `scan_steps(rematerialize=False)` |

Two observations that shaped the surface.

**`seam_gate.py:403` is not an adjoint use and neither is `:307`.** 403 is forward mode: a JVP has
no reverse pass and stores no tape, so `jax.checkpoint` there is pure overhead — semantically the
identity, measured as minutes of constant-folding inside
`jit(jvp_chunk)/jvp()/while/body/closed_call/checkpoint/checkpoint`. This is why `rematerialize`
is a real parameter with a real `False` case and not an internal default.

**Half the call sites want `rematerialize=False`.** If `scan_steps` had been designed only for the
adjoint, five of ten sites would have kept a hand-rolled `lax.scan` and the private import would
have survived in spirit.

Four further calls stay in `fesom_jax` and are a separate tidy-up: `determinism_probe.py:83`,
`sharded_grad_ab.py:158`, `scripts/paper/core2_lw_avgadj.py:261`,
`scripts/paper/core2_lw_signflip.py:160`.

## The numerical invariant

`dL/dm` from `chunked_grad` must equal `dL/dm` from the existing closures **to 3e-5** on the same
window, seed and flags. That number is not chosen for comfort: it is the agreement already
measured across boundary strides 1 / 2 / 3, and it is tighter than the model's own forward
reproducibility of 6e-4. If the extraction misses it, the extraction is wrong. **Do not adjust the
tolerance.**

Task 4 additionally asserts, on the pi mesh:

- the chunked gradient **equals** a monolithic `jax.grad` over a window short enough for both to
  fit — exact equality, since this is checkpointing and not approximation;
- the result is independent of chunk length (1, 2, 4 chunks over the same window);
- `boundary_stride > 1` changes nothing beyond 3e-5;
- `adjoint_stab.amplification(result.trace, chunk_steps)` accepts the trace and its length is
  `n_chunks + 1`.

## Names considered and rejected

`chunked_vjp` — it seeds with 1.0 on a scalar loss, so it is a gradient; the general VJP form is
available by passing your own `loss_fn`. `value_and_grad_chunked` — accurate and unusable at a
call site. `reverse_accumulate` — names the implementation rather than the result.

---

# Additions of 2026-09: an exact tangent-linear model and the trust toolkit

The surface above is unchanged; everything below is added to it, and every model default (no
new argument, `AdjointConfig()`, `Params` without `surface_adj`, `save_names=()`) traces the same
program as before (jaxpr text equality in the tests) with a bit-identical forward. The one
deliberate exception is the SSH solve, whose tangent changed (the forward did not). One default
of `chunked_grad` changed behaviour: `check_linear=True` runs one extra chunk VJP per call (the
zero-cotangent guard below); `check_linear=False` restores the old sweep. GPU numbers are from one A100-80GB
(CORE2 all-on reference physics: zstar + TKE + mEVP + GM, prescribed sea-ice adjoint, dt = 1800 s,
2013-01-01, jobs listed with each number); CPU numbers from a Levante compute node.

## The SSH solve: exact tangent-linear model (`fesom_jax/ssh.py`)

`solve_ssh` used to differentiate its conjugate-gradient solve through
`lax.custom_linear_solve(..., solve, transpose_solve)`. The transpose solve (reverse mode) is
tight and was correct. Forward mode, however, re-used the FORWARD solve, which stops at an
ABSOLUTE threshold derived from the primal right-hand side (`SOLTOL·‖b‖`): a small tangent gets
zero iterations. A one-step dot test (tangent-linear against adjoint) therefore failed exactly
where it matters, at small amplitude: pi mesh 7.6e-3, CORE2 all-on 3.4e-2 relative at amplitude
1e-8.

The solve is now a `jax.custom_jvp` (`_ssh_implicit`): the primal is the unchanged early-stopped
PCG (the forward is bitwise the old one), and the tangent is the implicit derivative of the
solution, `dδ = A⁻¹(db − dA·δ)`, computed by a tight solve in both directions
(`custom_linear_solve` with a relative-tolerance PCG, `GRAD_SOLTOL`), so the tangent and the
adjoint are exact transposes. Under zstar the operator depends on `hbar` and `dA·δ` is the
direct bilinear form `stiff_increment_matvec(mesh, dhbar, δ)`. Every array enters the rule as an
explicit argument: arrays held in `nondiff_argnums` become stale tracers when an already jitted
solve is differentiated or when a sharded program is lowered.

| one-step dot test, amplitude 1e-8 | before | after |
|---|---|---|
| pi (linfs) | 7.6e-3 | 4.3e-14 |
| CORE2 all-on (zstar), 3rd step | 3.4e-2 | 3.1e-12 |

Cost: the tangent solve takes about 345 CG iterations on CORE2 against 131 for the forward
(the transpose 351; `MAXITER` 500). The residual inconsistency between "the derivative of the
converged solve" and "the derivative of the early-stopped iterate" is a known few-1e-6 effect
and is unchanged. Sharded: the derivative solve returns zero off the owned lanes; before that
fix the reverse halo exchange scatter-added PCG partial sums from halo lanes onto the owners and
owner cotangents next to partition boundaries were wrong by 48-63 % for one solve. Parameter
gradients had stayed inside the old 5e-3 sharded-vs-dense tolerance. Sharded multi-GPU
per-node sensitivities computed before that fix should be re-validated.

The solve's output is named `"ssh_delta"` (`jax.ad_checkpoint.checkpoint_name`) INSIDE the
custom_jvp (primal and rule): a name on the caller's copy of the output does not reach the
residual the reverse program stores under zstar.

## `save_names`: a remat policy that keeps the SSH solution

`scan_steps(..., save_names=("ssh_delta",))` (also `integrate`/`integrate_jit`) gives the
per-step `jax.checkpoint` the policy `save_only_these_names(*save_names)`, so the reverse program
keeps one `[nod2D]` float64 per step instead of re-running the forward CG in the reverse pass
(CG `while_loop`s in the jaxpr of the reverse program: 4 → 3 for one step, 9 → 7 for an 8-step
pi window through `scan_steps`). `()` (default) is the old program.
It is not bitwise with the default (a different program; the early-stopped CG amplifies
round-off): the policy moves a pi 8-step gradient by less than turning remat on/off does.

A100, CORE2, 48-step chunks, `AdjointConfig.exact()`, second repeat, compile excluded
(jobs 27737801 without, 27737802 with the policy):

| `save_names` | window | forward s/step | reverse s/step | device peak | first-call compile |
|---|---|---|---|---|---|
| `()` | 1 day | 0.292 | 1.209 | 49.68 GB | 283 s |
| `()` | 10 days | 0.271 | 1.209 | 49.69 GB | - |
| `("ssh_delta",)` | 1 day | 0.284 | 1.193 | 49.63 GB | 281 s |
| `("ssh_delta",)` | 10 days | 0.270 | 1.185 | 49.65 GB | - |

The policy saves 1.3 % (1 day) to 2.0 % (10 days) of the reverse time at no extra device memory;
two repeats of the same setting differ by about 1 %. That is below the 5 % bar set for making it
a recommendation, so the default stays `()`. (The forward CG is 40-130 iterations of a sparse
matvec, a small part of a production step on the A100.)

## `chunked_grad`: linearity guard, field traces, `on_cot`, `state_cot`

- `check_linear` (default `True` ≡ `"zero"`): before the first reverse chunk of each call (and of
  each resumed leg) the chunk VJP is evaluated once more with a ZERO cotangent; every state and
  params output must be exactly 0 (exact on any platform: a linear map sends 0 to 0 whatever the
  rounding), else `RuntimeError`. `"full"` adds two more VJPs of the same chunk, `cot` again (the
  repeat spread `s`) and `2·cot`, and requires `‖vjp(2c) − 2·vjp(c)‖ ≤ max(1e-9, 10·s)·‖2·vjp(c)‖`
  for the state and the params parts. If either part's measured spread exceeds
  `INFORMATIVE_SPREAD` = 1e-6 (relative to `‖vjp(c)‖`), that bound admits a non-linear VJP too:
  `linear_check["informative"]` is then `False` and a `RuntimeWarning` says the check proved
  nothing (the pass/fail rule is unchanged). `False` skips it. The cost is in
  `reverse_seconds`; what it measured is in `ChunkedGrad.linear_check`. See "What the guard can and cannot see on a GPU"
  below before relying on `"full"`.
- `field_trace=True`: `ChunkedGrad.field_trace` holds, per chunk boundary (order of `trace`),
  `{state leaf: norm}` of the state cotangent; `adjoint_stab.amplification_by_field` turns it into
  per-field growth rates.
- `on_cot(c, cot_state)`: called at every boundary, the seed boundary `c = n_chunks` first, then
  after each reverse chunk; read-only.
- `ChunkedGrad.state_cot`: d loss / d `init_state` (host), i.e. the initial-state sensitivity.

## `chunked_grad_multi`: several costs, one forward

`chunked_grad_multi(chunk_fn, params, *, init_state, n_chunks, xs_fn, loss_fns, chunk_steps,
boundary_stride=1, check_linear=True, on_forward=None, on_chunk=None, on_cot=None)` runs ONE
forward to host boundaries and K reverse sweeps interleaved chunk by chunk; per chunk the cached
chunk VJP is applied K times to the same inputs. Between applications the K−1 inactive
cotangents and gradients are parked on the host, so the device holds about one cost's working
set plus one State; reverse cost ≈ K × a single sweep. Returns `ChunkedGradMulti(losses, grads,
traces, field_traces, state_cots, ...)`, equal bitwise (CPU) to K single-cost sweeps. Hook
contract: `on_cot(k, c, cot_state)` once per cost per boundary, boundaries in reverse order from
the seed boundary `c = n_chunks`, `k` ascending within a boundary; `on_chunk(k, c, norm,
grad_so_far)`. No `ReverseCheckpoint` (raises).

## `AdjointConfig`: named cuts, `validate()`, `FREEZE_FLAGS`, presets

- `diag_cuts: frozenset = frozenset()`: named DIAGNOSTIC derivative cuts from the registry
  `KNOWN_CUTS` (`{name: location}`; Phase 0 has `mix.kv_out` and `mix.av_out`, Kv and Av after
  the mixing closure and its halo exchange). At a cut point `dcut(cfg, name, x)` applies
  `stop_gradient` to every floating leaf when the name is in the set; otherwise it returns `x`
  itself, so an empty set traces the unchanged program. The set is a STATIC field on purpose: each
  cut set is part of the `chunk_fn` closure, so the cached executables of `chunked_grad` can never
  silently reuse a program traced without the cut. `{mix.kv_out, mix.av_out}` equals
  `freeze_mixing` bitwise (CPU, one-step VJP, state and params).
- `validate()` returns the config with `diag_cuts` coerced to a `frozenset`; unknown names raise
  `ValueError`, a bare string `TypeError`. `step()` calls it.
- `FREEZE_FLAGS`: the boolean `freeze_*` fields; iterate over these, not `_fields`.
- Presets: `AdjointConfig.exact()` (everything differentiated) and `AdjointConfig.long_window()`
  (`freeze_mixing + freeze_gm_slope`, today's recommendation beyond about a day; see the
  guard numbers below for what `exact()` does over 10 days).

## `differentiate_at(fn, args, alt)`

A `jax.custom_jvp` returning `fn(*args)` whose tangent (and so its VJP) is the Jacobian of `fn`
at `alt` instead of at `args` (same pytree; `alt` gets no tangent of its own; `fn` module-level,
all arrays through `args`). `alt == args` gives the plain VJP bitwise; works inside
`jax.checkpoint` and `lax.scan`; forward mode keeps working.

## `Params.surface_adj`: a surface-flux control

An optional `Params` leaf (default `None` = nothing traced, identical jaxpr, bitwise forward): a
dict with optional `[nod2D]` entries, applied at ONE place in `step()` after the ice and no-ice
surface branches join:

- `heat`: W/m², **POSITIVE = INTO the ocean**. The model's `heat_flux` is positive for ocean heat
  LOSS and temperature is forced only through `bc_T = −dt·heat_flux/VCPW`, so the adjustment is
  `bc_T += dt·heat/VCPW` and `heat_flux −= heat` (what KPP reads). Applied under sea ice too.
- `stress_x`, `stress_y`: N/m² in the model (rotated) frame, added to the nodal surface stress
  (read by TKE and KPP) and, as the mean of each element's three nodes, to the element stress.

Dense only (raises under a halo context); needs the forced path; unknown keys raise. An all-zero
dict is NOT bitwise with `None` on CPU (a different compiled program: XLA regroups upstream FMAs;
T within 0.5 ulp of its maximum, tke 0.7 ulp); the tests hold it to 16 ulp of each field's
maximum.

## `fesom_jax.adjoint_check`

- `dot_test(f, x, v, w, amps=(1, 1e-4, 1e-8), *, jvp=None, vjp=None)` → rows `(amp, lhs, rhs,
  rel, rel_norm)`, `rel_norm = |<w,Mv> − <Mᵀw,v>| / (‖Mv‖·‖w‖)`. Several amplitudes: a tangent
  model with an early-stopped solve inside passes at 1 and fails at 1e-8. ⚠️ On a CORE2 State
  `‖Mv‖` is dominated by fields with large units (`ssh_rhs`, EVP stresses), so `rel_norm` is
  1e-18-1e-20 for any mismatch that matters; read the plain `rel` next to it (and, in
  `scripts/adjoint/p0_acceptance.py`, `rel_abs = |lhs − rhs| / Σ|w_i·(Mv)_i|`).
- `fd_sweep(J, x, d, hs, ad, repeats=1)` → rows `(h, fd, ad, rel_err, noise_over_h)`;
  `plateau(rows, rtol)`.
- Chunked window maps on PINNED states: `forward_states(chunk_fn, params, s0, n_chunks, xs_fn)`
  (host list of chunk-entry States), `window_jvp(chunk_fn_tl, params, s0, t0, dparams, n_chunks,
  xs_fn, *, pinned_states=None, compiler_options=None)` (state and params tangent; the chunk
  function built WITHOUT remat) and `window_vjp(chunk_fn, params, s0, w, n_chunks, xs_fn, *,
  pinned_states=None, ...)` (reverse through `chunked_grad` over the same stored boundaries).
  `NO_CONSTANT_FOLDING` as `compiler_options` compiles the CORE2 tangent chunk in 27 s instead of
  761 s (CPU; XLA folds constant scatter-adds of zero tangents for minutes).
- `power_iteration(apply_M, apply_MT, v0, *, weights, mask, iters, tol)` and `volume_weights`:
  the leading singular value of a window map in a weighted norm (use `uv`, not `uvnode`: the
  cotangent of the incoming `uvnode` is zero below the surface; at the surface it is small but
  NOT zero, because the bulk formulae and the ice-ocean stress read the previous step's
  `uvnode[:, 0]` before the mixing scheme recomputes it); `fdlin(J_map, x, v, hs, tl, ...)`.

Pinning fixes the chunk ENTRY states only. Within a chunk the forward program, the tangent
program and the VJP program (which re-runs the chunk forward inside itself) each follow their own
trajectory, and on the all-on CORE2 model two different compiled programs are not round-off apart
even after one step (see "Measured on the A100" below).

## `fesom_jax.adjoint_costs`

`box_volume_weights` / `box_mean_T_vw` (weights `hnode·areasvol` with the LIVE zstar thickness:
an unweighted mean is not conserved by transport and creates spurious velocity sensitivities),
`sst_area_mean`, `ssh_area_mean`, `mld_field(state, mesh, invalid="column"|"zero")` (density
threshold on the live `Z`; a column mixed to the bottom gets, by default, the depth of its deepest
wet MID-layer, `deepest_wet_mid_depth`: the deepest value a valid crossing can take and its limit
as the crossing at the deepest wet level disappears, so the field is continuous at the
valid→invalid switch on that path — not the bottom-interface depth `column_bottom_depth`, which
lies half a layer deeper and would make the MLD jump by `hnode/2`; `"zero"` keeps the threshold
routine's 0, bit-identical to `obs_compare.mld_density_threshold` and so compatible with the
earlier TKE→MLD calibration D2a), `with_running_costs(step_fn, accumulators)` (a step on the carry `(State, acc)`;
accumulators may be scalars or `[nod2D]` fields, e.g. a window-mean MLD; pass the wrapped step to
`scan_steps` inside the chunk function), `terminal(cost)`, `running_mean(name, n_steps)`.

## The shared CORE2 driver `scripts/adjoint/core2_adj.py`

One model and ONE compiled chunk program per process (`build_model(start, n_steps, chunk_steps,
restart, adjoint_cfg, ice_mode="prescribed", save_names=(), accumulators=None,
rematerialize=True, *, params0, to_params, surface_adj, config, state)`); everything that changes
between windows goes through `init_state` / `xs_fn` / `loss_fn`, and a second window compiles
nothing. Named boxes (`BOXES`), costs (`COSTS`: `box_T_vw`, `sst_aw`, `ssh_aw`, `mld_box`,
`mld_mean`, or a callable), directions (`DIRECTIONS`: T points, `ln_k_gm`, box `heat` /
`stress_x`, `ln_tke_c_k`, `ln_tke_c_eps`, `ln_tke_cd`, `ln_tke_alpha`) and the actions `fwd grad
screen fd tl dot states restartcheck`, each appending JSONL rows with provenance. Run it through
`scripts/adjoint/job_gpu.sbatch` / `job_cpu.sbatch` (they set `PYTHONPATH` to the checkout; the
driver refuses any other `fesom_jax`). `scripts/adjoint/p0_acceptance.py` holds the GPU
acceptance measurements below (`dot`, `guard`, `timing`, `vjprep`).

**Shared 2013 States.** One forward of 2013 from the hindcast restart `fesom.2013.001.00000`
(`core2_adj.py states`, A100 job 27737765, 4500 s for 365 days, 0.25 s/step including the daily
diagnostics) wrote full States on the 1st of every month, 2013-01-01 to 2014-01-01 (13 zarr
restarts `fesom.<YYYY>.<DOY>.00000`, 0.98 GB each on disk, loadable by
`zarr_output.read_restart` / `core2_adj.load_state`), 365 daily 2-D diagnostic files
(`daily/diag_<YYYY>_<DOY>_00000.npz`: MLD and its valid flag, SST, surface speed, sea-ice
concentration, column-minimum N², tke at interface 1, the daily fractions of steps with tke at
its floor and with convection) and a `README.json` with provenance and the field list, 16 GB in
all, in `/work/ab0995/a270088/fesom_jax_dev/states_2013/`. These daily files were written
before the `mld_field` fill changed (git 2cd03f1): where `mld_valid` is 0 their `mld` holds the
bottom-INTERFACE depth, half a layer below today's fill; mask with `mld_valid` or recompute.
`build_model(start=...)` finds these States before the hindcast archive. Restart check (`core2_adj.py restartcheck`, 2013-07-01, one
day): the day restarted from the saved 2013-07-01 State differs
from the continuous run by as much as repeated restarted days differ among themselves, within a
factor 1.03 for every daily field (3 repeats, job 27737934: MLD 0.42 m against a spread of 2.27
m, SST 3.2e-3 / 4.0e-3 K, a_ice 4.3e-4 / 5.7e-4, surface speed 2.75873e-2 / 2.75869e-2 m/s; 5
repeats, job 27739111: MLD 2.28 / 2.83 m, SST 5.3e-3 / 6.4e-3 K, column-minimum N² 3.17e-6 /
3.07e-6 s⁻²). With a GPU the one-day spread of a restarted day is itself a sample (a maximum
over a few repeats), so an equally good restart lands just above it about as often as just
below. The driver therefore passes a field when its difference is at most `tie_margin` × the
spread, `RESTART_TIE_MARGIN` = 1.05 by default (`--tie-margin`), and records `tie_margin` and
the fields over it (`over_margin`) in the row. The strict rule of the first version flagged the
two near-ties above (surface speed ratio 1.000015, job 27737934; N² ratio 1.032, job 27739111,
which therefore exited 1) as `pass: false`; under the margin both checks pass (largest ratio
1.032). A zero spread (CPU) still requires a zero difference. The restart path is bitwise on a CPU (`scripts/tests/test_core2_adj.py`).

## Measured on the A100

All on one A100-80GB, CORE2 all-on reference physics, prescribed sea-ice adjoint, window from
2013-01-01 00:00, cost `box_T_vw` on the default box (`natl_subpolar`, 50-60°N, 40-20°W,
0-500 m), measurement script `scripts/adjoint/p0_acceptance.py`, rows with provenance in
`/work/ab0995/a270088/fesom_jax_dev/runs/p0/acc_*/results.jsonl`.

### Two runs are not the same trajectory, even of one program

GPU scatter-adds are not reproducible from run to run, and on the all-on model the round-off
differences flip discrete switches within a few steps:

- The tangent program's primal against the forward chunk program, ONE step from the same pinned
  State: the ice skin temperature `t_skin` (a warm-start state of the ice surface-temperature
  solve) differs by up to 4.4 K at 8-20 nodes of almost closed ice (e.g. node 101619, a_ice 0.99:
  −5.85 against −1.43 °C, with `a_ice` equal to 4e-6), the EVP stresses by up to 7e2 at tens to
  hundreds of elements, `d_eta` by 3e-5 m at about 2000 nodes, T by 3e-5 K and S by 1.6e-3 psu at
  single nodes; Kv and Av by ≤ 1.2e-14 (no switch yet).
- After 6 steps the convective-mixing switch has flipped: Kv or Av differ by 0.1 m²/s at 600-2600
  points. After 48 steps (one day): at 2500-8700 points, T by up to 8e-3 K, S by 2e-2 psu, `uv`
  by 1.4e-2 m/s, `a_ice` by 7e-4.
- Two runs of the SAME forward program over one day differ just as much (Av 0.1 at 7600 points,
  Kv 0.125, `t_skin` 3.9 K, `uv_rhs` 0.064; jobs 27737797, 27737798). So the tangent program is
  no further from the forward than the forward is from itself.

Pinning the chunk-entry States removes the drift ACROSS chunks; within a chunk each sweep follows
its own trajectory.

### Dot test against chunk length

1-day window, pinned States, two repeats each with a new forward, amplitudes 1 and 1e-8
(jobs 27737797, 27737798, 27738387, 27738388). Plain relative mismatch
`|<y,Mx> − <Mᵀy,x>| / max(|<y,Mx>|, |<Mᵀy,x>|)`, range over repeats and amplitudes:

| preset | direction | 1-step chunks | 6-step chunks | 48-step chunks |
|---|---|---|---|---|
| `exact()` | T point in the box | 6.7e-13 - 3.1e-12 | 4.7e-13 - 8.6e-12 | 1.5e-8 - 2.2e-5 |
| `exact()` | `ln_tke_c_k` | 1.8e-10 - 5.8e-9 | 3.6e-6 - 1.6e-5 | 5.7e-2 - 1.1 |
| `long_window()` | T point in the box | 2.6e-13 - 6.9e-12 | 3.9e-13 - 9.7e-12 | 7.0e-12 - 3.4e-11 |
| `long_window()` | `ln_tke_c_k` | 0 = 0 | 0 = 0 | 0 = 0 |

(`freeze_mixing` cuts every derivative path of the TKE constants, so both sides are exactly 0.)
`rel_norm` is at most 2.3e-18 with 1-step chunks, 1e-15 with 6 and 3.3e-10 with 48; it is small
because `‖Mx‖` is dominated by `ssh_rhs` and the EVP stresses (see `dot_test`).

With every step's entry pinned the tangent and the adjoint agree to ≤ 5.8e-9 along both
directions and both presets: there is no transposition error (one would be O(1e-1) or larger at
any chunk length). The growth with the chunk length is the in-chunk trajectories, and it is as
large as each sweep's own run-to-run variation ON THE SAME PINS: the tangent program run twice
along `ln_tke_c_k` (`exact()`) differs by 4e-9 - 1.4e-8 of its norm with 1-step chunks, 3e-5 -
8e-4 with 6-step and 0.95 - 50 with 48-step chunks; the adjoint run twice (48-step, `exact()`)
differs by 2.2e-5 (T point) and 2.8e-2 (`ln_tke_c_k`) in the product with the direction, and its
state cotangent (S, T, tke) by 101 % of its norm. Under `long_window()` the same repeats differ
by ≤ 6e-8 (tangent, T point) and ≤ 3.5e-11 (adjoint product), and the mismatch stays at 1e-11.

**The exact adjoint over one day is not reproducible on the GPU.** `dJ/d ln_tke_c_k` of five
1-day `exact()` sweeps: +4.4e-4, −7.1e-5, −6.9e-5, +1.0e-4, −6.5e-5 (sign flips); `dJ/d ln_k_gm`
3.8e-4 - 5.6e-4; the state-cotangent norm after one day 0.57 - 4.0 (from 6.6e-3 at the seed).
Under `long_window()` the same gradient is 5.3932e-4 in two sweeps (agreeing to 2e-10) and the
cotangent norm 9.33e-3. Over 10 days the `exact()` sweep overflows (6.5e51 at the window start;
`long_window()`: 3.6e-2).

### Linearity guard on the GPU

`"zero"` (the default) found `vjp(0) = 0` exactly in every sweep of these runs (`zero_max` = 0).
`"full"` (first reverse chunk, 48-step chunks):

| preset | window | repeat spread s (state / params) | scaling error (state / params) | result | job |
|---|---|---|---|---|---|
| `exact()` | 1 day | 1.16 / 0.93 | 2.96 / 2.38 | pass | 27737799 |
| `exact()` | 1 day | 6.1e-2 / 8.3e-3 | 8.5e-2 / 1.1e-2 | pass | 27737797 |
| `exact()` | 1 day | 4.48 / 0.96 | 1.76 / 0.38 | pass | 27738387 |
| `exact()` | 10 days | 3.37 / 3.2e-2 | 0.74 / 3.9e-3 | pass | 27737799 |
| `long_window()` | 1 day | 1.8e-7 / 6.6e-10 | 4.5e-6 / 6.3e-8 | **RuntimeError** | 27737800 |
| `long_window()` | 1 day | 3.5e-6 / 6.8e-8 | 1.7e-7 / 1.6e-9 | pass | 27737798 |
| `long_window()` | 10 days | 8.8e-7 / 1.0e-9 | 8.6e-7 / 1.7e-9 | pass | 27737800 |

The same measurement on a single chunk (job 27737935): with 1-step chunks the chunk VJP repeats to
2.8e-12 / 3.8e-12 (`exact()`) and 3.4e-12 / 1.0e-12 (`long_window()`), scaling error 8.2e-12 /
4.6e-12; with 48-step chunks `exact()` repeats to 0.84 / 0.11 (scaling 2.8e-2 / 4.5e-3) and
`long_window()` to 4.5e-6 / 6.1e-8 (scaling 4.5e-6 / 6.2e-8). The guard itself costs 170 s on a
48-step chunk (three extra chunk VJPs; `"zero"` 55 s).

So with production chunks `"full"` cannot tell a non-linear VJP from GPU non-determinism: the
run-to-run spread of one chunk VJP is itself not stable (1.8e-7 in one pair of calls, 4.5e-6 in
another, same program and inputs), one measured spread does not bound the next, and the 1-day
`long_window()` run raised a false alarm; under `exact()` the spread is O(1) and the test is
void — since the 2026-09 follow-up `"full"` reports that itself (`informative: False` and a
`RuntimeWarning` whenever the spread exceeds 1e-6). Use `"zero"` on the GPU; run `"full"` with 1-step chunks (spread ~3e-12) or on a CPU
(spread 0, where it is exact) once per new compiled program.

Deterministic GPU kernels (`XLA_FLAGS=--xla_gpu_deterministic_ops=true`) would make repeats
exact, but the CORE2 forward chunk program had not finished compiling after 2 h under that flag
(job 27737936, timed out); they are not a practical fix at this model size.

