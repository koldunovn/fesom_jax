# FESOM2 → JAX Port — Phase 7a: Differentiable Parameter Tuning / Calibration (sub-plan)

**Parent plan:** `docs/plans/20260605-fesom-jax-port.md` (Phase 7a section).
**Predecessor:** `docs/plans/20260607-fesom-jax-gmredi.md` (GM/Redi, GATE 6B — the `k_gm`/`redi_kmax`
2nd ML-hook seam this calibrates) — and the §"Forward pointer" there.
**Created:** 2026-06-07. **Status:** ⏸️ DEFERRED — **do Phase 6C (KPP, `…-kpp.md`) first** (the user's
2026-06-07 decision: finish the full functioning model before the tuning on-ramp). This plan is
**design-complete and ready to execute** when KPP lands; the early scoping below is preserved verbatim.

**Goal:** prove the differentiable port can **calibrate physics parameters** against a target and push
the optimum back into the operational Fortran model — the *same* `params.py` ML-hook seam, but the
trainable leaves are the existing physical constants, not NN weights. A **small extension, not a
restructure**: the `Params` pytree, the checkpointed differentiable `integrate`, and FD-verified
gradients (`d/d(k_gm)` plateau 3.5e-6) are all already in place. What's missing is only an objective +
an optimizer loop + a target.

---

## 0. What was already verified this session (2026-06-07, before the KPP pivot)

These de-risk 7a.1 — they are done, do not re-check:

1. **`optax` is installed, clean.** `optax 0.2.8` + `absl-py 2.4.0` into the `fesom-jax` env; the
   pip dry-run confirmed **jax/jaxlib 0.10.1 untouched** (only those two pkgs added). Verified an x64
   Adam step works: `optax.adam(50.0)` from `k_gm=800` with grad `−1` → `849.9999995` (Adam normalizes
   the gradient magnitude, so the **step size ≈ lr in parameter units** → ~14 iters cover 800→1500).
2. **The `k_gm` twin is well-posed (no upper clamp).** `gm.py:198` is `k_top = max(scaling·k_gm,
   k_gm_min=2.0)` — **only a lower floor**, no upper clamp at `k_gm_max=1000`. So injecting `k_gm=1500`
   is unclamped and `fer_K ∝ k_gm` **linearly** (through `k_top·zscaling`) → the misfit bowl is smooth
   and well-conditioned across [800,1500]. (`k_gm_max=1000` in `GMConfig` is just the default value,
   not a clamp.)
3. **The seam parameterization.** Match the GM grad-gate (`core2_gm_grad_gate.py:107-108`) and the C
   auto-sync (`Redi_Kmax = K_GM_max`): tune a **single scalar θ driving BOTH** `k_gm=θ` and
   `redi_kmax=θ`. `k_ver`/`a_ver` held at defaults (they cancel in the perfect-model twin).

---

## 1. The calibration seam (factor once, reuse for every tunable)

New module `fesom_jax/calibrate.py` — the optimizer half of the ML-hook seam, generic over a **pytree
of tunable leaves** (a `dict`, e.g. `{'k_gm': θ}`; expand to `{'k_gm':…, 'k_ver':…}` or a `[nod2D]`
field leaf with **zero** structural change — array leaves already differentiate):

```python
def optimize(loss_fn, init, optimizer, *, n_iters, on_step=None, stop_fn=None):
    """Minimize loss_fn(tunables)->scalar over the pytree `init` with an optax optimizer.
    jit's value_and_grad once; host-side loop (logging / early-stop / the work is in the
    jitted forward+backward). Returns (final_tunables, history)."""
    vg = jax.jit(jax.value_and_grad(loss_fn))
    params, opt_state = init, optimizer.init(init)
    history = []
    for it in range(1, n_iters + 1):
        loss, grads = vg(params)
        gnorm = optax.global_norm(grads)
        history.append({'it': it, 'loss': float(loss), 'gnorm': float(gnorm),
                        'params': jax.device_get(params)})
        if on_step: on_step(history[-1])
        if stop_fn and stop_fn(history[-1]): break
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    return params, history

def grid_scan(loss_fn, base, leaf, values):
    """Forward-only 1-D sweep of base[leaf] over `values` → [(v, loss)]. The misfit-bowl
    probe: CONFIRM the minimum sits at the injected value before trusting the descent."""
    f = jax.jit(loss_fn)
    return [(float(v), float(f({**base, leaf: jnp.asarray(v, jnp.float64)}))) for v in values]
```

**The expansion pattern (7a.3):** the experiment script owns `build_params(tun) -> Params` (maps the
tunable dict into a full `Params`); to add a 2nd tunable, add its key to `init` and one line to
`build_params`. Everything else is unchanged. `optimize` is optimizer-agnostic — Adam for scalars,
swap `optax.lbfgs` (line-search) later if needed.

**Unit test** `fesom_jax/tests/test_calibrate.py` (fast, no CORE2 — keeps the suite green + covers the
machinery): `optimize` recovers the min of an analytic `loss({'a':a,'b':b})=(a−3)²+(b+1)²` from
`(0,0)` → `(≈3,≈−1)` (exercises the multi-leaf pytree); `grid_scan` returns the right bowl.

---

## 2. Task 7a.1 — the perfect-model `k_gm` twin (the first experiment, ~1 day)

Driver `scripts/core2_param_tune.py` (model on `core2_gm_grad_gate.py` — same CORE2 build, GM ON / ice
OFF for a deterministic, EVP-floor-free, GM-isolated signal):

1. **Forward window.** `forward_sst(tun) = integrate(st0, mesh, op, None, n_steps=N, params=build_params(tun), dt=500, step_forcings=sfs, forcing_static=fs, gm_cfg=GMConfig()).T[:,0]` over a
   **short** window `N` (default ~24–48 steps = 3.3–6.7 h; the GM grad-gate already shows clean signal
   at N=1). Forcing **stacked** (`cf.stack`) — fine for small N (don't stack for long runs → OOM).
2. **Twin target (detached).** `sst_target = device_get(forward_sst({'k_gm': 1500.}))` → a frozen
   constant (no grad leaks to it).
3. **Misfit.** `misfit(tun) = Σ_wet (forward_sst(tun) − sst_target)² / n_wet` (wet-surface masked, like
   `mean_sst` in the grad gate). Primary field = **SST** (the observable, per the user); add `--field
   {sst,t3d}` so a subsurface-T misfit (a *much* stronger GM signal) is the fallback if SST is too flat.
4. **Grid scan FIRST (cheap, forward-only — run on the login node).** `grid_scan(misfit, {'k_gm':1500},
   'k_gm', linspace(625,1675,11) ∪ {1500})` → print the bowl, confirm **argmin = 1500**. This validates
   well-posedness before any backward (and runs without a SLURM job).
5. **Recover (GPU job — backward HANGS on the login node).** From `k_gm=800`, Adam with a **cosine-decay
   lr** (`optax.adam(optax.cosine_decay_schedule(init_value=60, decay_steps=iters, alpha=0.02))` — big
   early steps, fine late steps → tight convergence; plain `lr=50` oscillates ±lr near the min). ~60–80
   iters; `stop_fn` when `|θ−1500|/1500 < 2%` and `misfit < 1e-2·misfit0`. Report θ + misfit trajectory.
6. **PASS token `TWIN_RECOVER_OK`**: recovered θ within ~2 % of 1500 AND final misfit ≪ initial.

**Compute:** GPU job `scripts/core2_param_tune.sbatch` (mirror `core2_gm_grad_gate.sbatch`,
`-A ab0995_gpu -p gpu --gres=gpu:1`, `--time=02:00:00`). Per `value_and_grad`: N-step GM forward+backward
(~0.4 s/step on A100 ⇒ N=24 ≈ 9 s/iter, N=48 ≈ 17 s/iter); 60–80 iters ⇒ ~10–25 min + compile. Fits
A100-80GB at N≤48. `jax.clear_caches()` between the grid scan and the backward loop.

---

## 3. Task ladder (from the parent plan) + Fortran transfer

- **7a.1** perfect-model `k_gm` twin (above) — proves the optax/loss loop end-to-end.
- **7a.2** the `Params`-expansion pattern for N tunables + a misfit (reference-run first, then real obs
  SST/SSS/ice). Add `k_ver`/`a_ver` as a 2nd/3rd leaf via `build_params`; a `[nod2D]` field leaf demo.
- **7a.3** short-window-adjoint-**at-equilibrium** tuning + a **gradient-free EKI** baseline for the slow
  mean (Ensemble Kalman Inversion, forward runs only, `vmap`-parallel — no adjoint, no chaos).
- **7a.4** export the optimum to `namelist.oce` (`K_GM_max`/`Redi_Kmax`) and confirm the **Fortran** run
  reproduces the JAX-predicted change. **Scalar → namelist = ZERO Fortran code** (the killer app). The
  three transfer tiers: scalar/profile→namelist (zero code) · 2D/3D field→netCDF (tens of lines) ·
  NN→Fortran inference (Phase 7).

**⚠️ Decades-spin-up is LOCKED OUT of the adjoint** (memory: a decade ≈ 630k steps, even O(√N) → >150
GB; AND chaotic gradient blow-up past the Lyapunov time). For *equilibrated* tuning: **spin up FORWARD,
no AD** (the stability run / Fortran) → **short-window adjoint anchored at the equilibrated state** →
**EKI for the slow mean**. The perfect-model twin (7a.1) only needs the short-window adjoint.

**Well-conditioned targets** (clean gradients shown): `k_gm`/`redi_kmax` (GM eddy), `k_ver`/`a_ver`
(mixing), GM depth/resolution scalings; **+ KPP's `Ricr`/`visc_sh_limit`/backgrounds** once Phase 6C
lands (mixing-seam tunables — note the KPP `Ricr` path runs through the discrete `kbl`, so prefer the
additive `visc_sh_limit`/`K_bg` for a clean gradient; use EKI for `Ricr`). ⚠️ EVP **rheology** is stiff
(`1/delta_min ~1e16`) — `stop_gradient` / EKI it.

---

## 4. GATE 7a (acceptance)

A tuned scalar (e.g. `k_gm`) **measurably reduces a defined misfit in JAX** AND, written to the
namelist, **in Fortran**; the **perfect-model twin recovers the injected value** (`TWIN_RECOVER_OK`);
masked-NaN clean; suite green (the `calibrate.py` unit test + `params=None` bit-identical).

---

## Revision Log

- **2026-06-07 — Created (deferred).** Scoped at the start of the post-GATE-6B session, then the user
  chose to finish the full functioning model (Phase 6C KPP) first. Design preserved: the `calibrate.py`
  seam (`optimize`/`grid_scan`, optax-backed, dict-pytree tunables, the `build_params` expansion
  pattern), the perfect-model `k_gm` twin recipe (grid-scan bowl → cosine-decay Adam recovery, GM-on /
  ice-off, SST misfit), and the verified de-risking facts (§0: optax installed clean; `k_gm` unclamped
  ⇒ twin well-posed; θ→(k_gm,redi_kmax) parameterization). Resume here after GATE 6C.
