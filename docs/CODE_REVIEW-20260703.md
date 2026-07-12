# fesom_jax — comprehensive code review (2026-07-03)

**Scope.** The full `fesom_jax` package (~16.5 k lines, 55 modules), the operational
`scripts/` (~12.8 k lines), and the test suite (~17 k lines, 79 files), reviewed at the
working tree of commit `a492244` **including the uncommitted changes** (the 2026-07-02
archival-restart / canonical-output switchover: `run.py`, `run_config.py`,
`zarr_output.py`, `ushow_output.py`, `canonical_redist.py`, both hindcast sbatch chains).

**Method.** Seven independent subsystem reviews (core dynamics; tracers & mixing; sea ice;
forcing/IC/obs; distributed infrastructure; run driver & I/O; tests & scripts), each
reading its files in full and cross-checking suspected defects against the C/Fortran
references (`/home/a/a270088/port2/fesom2_port{,_zstar}/src`), the docs, and the test
suite before flagging. All critical and major findings were then independently
re-verified against the source (and, for the operational ones, against the live SLURM
state and job logs) by the orchestrating review. **No code was changed.**

**Severity scale.** `critical` = data loss / production derailment / wrong science in a
live path; `major` = real defect or silent-failure hazard in a reachable path; `minor` =
latent bug, fidelity landmine, or robustness gap; `nit` = polish. Every finding carries a
confidence label (`confirmed` = re-checked against source/reference/logs; `likely`;
`speculative`).

---

## 1. Executive summary

**The model itself is in excellent shape; the operational shell around it is where the
risk is.** Across ~10 k lines of numerics (dynamics, tracers, mixing, ice, forcing) the
review found **zero critical and three major** defects — a remarkable defect density for
a hand-ported model of this complexity, attributable to the dump-gated porting
discipline. The three critical findings all live in the SLURM chain scripts, and two of
them are in the *uncommitted* 2026-07-02 archival-restart switchover that is running
production right now.

### ⚠️ Urgent operational findings (live today)

1. **The pending `core2_hind` job 26025085 is already broken.** Both hindcast sbatch
   scripts clobber the run-tag variable: `TAG=$(cat "$ARCHIVE/restart.latest")`
   (`run_core2_hindcast.sbatch:102`, `run_forca20_hindcast.sbatch:79`) overwrites the
   chain's `TAG` (e.g. `core2_hindcast_v2`) with the restart directory name (e.g.
   `fesom.1958.033.00000`), and `do_resubmit` exports the clobbered value. Job#8
   (26017875) completed cleanly and already fired this resubmit — the pending job will
   compute `OUT=…/runs/fesom.1958.033.00000` (empty), find no archive, and **cold-start a
   junk canary chain while the real run (currently at ~1971 per `restart.latest`) silently
   stalls**. The junk chain then self-perpetuates: each cold start lands in a fresh bogus
   `OUT`, so the `.jobcount` runaway guard (which lives inside `$OUT`) **never fires**.
   Fix is one line per script (read the pointer into a differently named variable, e.g.
   `LTAG`). The pending `forca20_hind` job 26024433 will inherit the same bug on *its*
   first resubmit.
2. **The silent cold-start fallback that caused the 2026-07-03 data-loss incident is
   still in both hindcast chains.** `CUR=$($PY -c "…json…" 2>/dev/null || echo 0)` plus a
   missing pointer both yield `CUR=0` → cold start from 1958 — and a cold start
   *overwrites* existing data (`ushow_output.py:94,177` open period stores with
   `mode="w"`; `zarr_output.py` rewrites `$OUT/restart` in place). The incident is visible
   in the logs (core2 job#6 at step 728640 → job#7 `CUR=0 / restart=none`). A transient
   parallel-FS read failure of `.zattrs` is enough to re-trigger it. There is no
   "cold start requested but `$OUT`/`$ARCHIVE` non-empty ⇒ abort" guard.
3. **`run_dars_3yr.sbatch:63` still contains `rm -rf $OUT` inside the cold-start branch
   of a self-resubmitting chain** — a direct violation of the project's all-caps
   no-deletion rule, reachable from the same `|| echo 0` failure class. The chain is
   superseded but the script remains committed and submittable.

### Top code findings (non-operational)

| # | Sev | Where | What |
|---|-----|-------|------|
| 1 | major | `ice_step.py:146` | zstar freshwater balance gets start-of-step `a_ice` instead of the post-advection `a_co` the C uses → residual global freshwater leak every step (same class as the just-fixed sublimation leak, smaller) |
| 2 | major | `integrate_sharded.py:454-490` | nothing blocks `return_grad_fn=True, use_ragged=True` → silently wrong gradients through the known-broken `ragged_all_to_all` transpose |
| 3 | major | `step.py:250` | Shchepetkin PGF fed `ρ−2ρ0` (double `ρ0` subtraction); analytically offset-invariant today (dump gate passes at 1e-14) but coarsens stencil-difference precision ~30× and is a trap for any non-invariant future consumer |
| 4 | major | `zarr_output.py:94-206` + `run.py` | rolling `--restart-out`/`checkpoint_every` writes are in-place, non-atomic, attrs-before-data → a mid-write kill leaves a mixed-step restart that `read_restart` accepts silently |
| 5 | major | `run.py:433` + `zarr_output.py:246-275` | "immutable" archival restarts are silently rewritten by any re-run of the same calendar span (crash-resume re-crossing a boundary; branching experiment with the same archive root) |
| 6 | major | `run_config.py` vs `run.py` | several validated YAML keys are dead: `snapshot_every`, `checkpoint_every`, `restart_archive_*`, `forcing`, `output_dir` are accepted but never read by the driver (only CLI flags are wired) — the unknown-key check gives false confidence |
| 7 | major | `run_forca20_hindcast.sbatch:64` | `TOTAL=438000` omits the 1960 leap day (1959–60 = 731 days, not 730); the "3-year" chain ends 1960-12-31 and declares DONE a day early |

The model-physics majors (#1–#3) are one-line fixes; #1 is worth fixing before long
zstar production analysis, and #3's fix will perturb zstar results at the rounding floor
(any bit-stability expectation on existing chains breaks).

**Test suite: genuinely strong** — 610 tests in three principled tiers (C-dump gates with
per-field tolerances and anti-triviality guards; analytic/structural invariants;
FD-vs-AD checks), exactly one xfail (the documented upstream JAX ragged bug, still
current). The recurring gap pattern: the *assembled invariants* are under-tested relative
to the kernels — no ⟨water_flux⟩≈0 conservation gate on the assembled zstar ice step
(would have caught both freshwater-budget bugs), no crash/partial-write restart test, no
test that YAML keys take effect, and the sbatch chain logic is entirely untested (the TAG
clobber and the leap-day TOTAL are exactly the class a tiny bash harness would catch).

**Overall:** the porting methodology (C line-number citations, dump gates per kernel,
masked-NaN AD discipline, dead-branch config gating, PORTING_LESSONS journal) is the
strongest reviewer-visible asset of this codebase and is what kept the numerics nearly
defect-free. The weak flank is everything that runs *unattended*: chain scripts, restart
atomicity, and cold-start semantics. The recommendations in §10 are ordered accordingly.

---

## 2. Cross-cutting observations

**Package hygiene is unusually good for a research code.**

- Zero `TODO`/`FIXME`/`HACK` markers in the package; zero bare/broad `except` clauses in
  `fesom_jax/` (all 68 broad excepts live in one-off `scripts/core2_paper_*` drivers).
- `jax_enable_x64` is configured in exactly one place (`fesom_jax/config.py`); exactly one
  env-var switch in the package (`FESOM_REUSE_EXE`), documented.
- Traceability is exceptional: nearly every routine cites the C/Fortran lines it ports,
  deviations are explicit and dated, and `docs/PORTING_LESSONS.md` (≈4.9 k lines) is a
  real engineering journal with test anchors per lesson.

**Repo/git hygiene needs a sweep.**

- A ~640-line uncommitted diff across 13 files — including the archival-restart mechanism
  and both production chain scripts — has been running production for days. Commit it
  (after fixing the TAG clobber) so the running configuration is reproducible.
- 39 untracked paths, including stray logs (`scripts/_offpath.log`, `cache_ic_forca20.log`,
  `prepare_forca20.log`, `restart_ushow_forca20.log`, `fullsuite.log`) and a large set of
  untracked sbatch scripts and handoff docs — commit or prune.
- `scripts/` holds 670 files of which only 205 are tracked; the 4 live production chains
  are distinguishable from ~60 one-off experiment drivers only by reading headers. A
  `scripts/prod/` vs `scripts/exp/` split (or a naming convention) would prevent the
  wrong-script-submitted class of error.
- `pyproject.toml` floor pins (`jax>=0.4.34`) are stale relative to the recorded working
  set (JAX 0.10.1); harmless but meaningless as compatibility statements.
- `configs/forca20_tke_mevp_prod.yaml` still says `partition: dist_32` + "all de-risked"
  while dist_32 is known to compile-hang; production works only because the sbatch
  overrides `PART=dist_16`. Running the YAML directly picks the hanging partition.

---

## 3. Core ocean dynamics (`step.py`, `momentum.py`, `ssh.py`, `ale.py`, `pgf.py`, `eos.py`, `ops.py`, `state.py`)

### Assessment

`step.py` is a functional re-orchestration of the C driver: one `step()` threading a
frozen `State` pytree through 16 substeps, with every optional physics package gated by a
static config object whose `None` produces a genuinely dead (untraced) branch — the
mechanism by which the baseline path stays byte-identical as features accrete. The SSH
solve is the AD centerpiece: a static COO stiffness operator + MITgcm preconditioner, a
PCG replicating the C's loose early-stop semantics, wrapped in
`jax.lax.custom_linear_solve` so the reverse-mode cotangent is a tight (1e-13) implicit
solve; zstar recomputes the stiffness increment inside the matvec so the operator itself
is differentiable state. zstar geometry is reconstructed live each step (no carried
geometry), with deliberate nominal below-bottom spacing for AD safety.

### Findings

- **[major] [confirmed — independently re-verified] `step.py:250-251`** — the Shchepetkin
  PGF is fed `ρ−2ρ0`, not the `ρ−ρ0` its contract and the C specify.
  `eos.compute_pressure_bv` already returns `density = _insitu(…) − rho_ref`
  (`eos.py:128`, matching the C's `density_m_rho0`), and `step.py` subtracts `DENSITY_0`
  again; `pgf.py:79-82` documents the argument as "ρ − ρ0". The kernel is analytically
  invariant to a constant offset (constants cancel in the stencil differences), which is
  why the zstar dump gate still passes at <1e-14 — but the shift coarsens the anomaly's
  stencil-difference ulp ~30× (values ~30 → ~−1000) and is a latent trap for any future
  non-offset-invariant consumer. One-line fix; note it moves zstar results at the
  rounding floor.
- **[minor] [likely] `eos.py:63`** — unguarded `jnp.sqrt(S)` in `jm_components`
  generates NaN cotangents wherever S==0. The CORE2 IC zero-fills below-bottom S
  (`phc_ic.py:398-399`) and nothing re-fills those lanes, so S stays exactly 0 there for
  the whole run; the sqrt VJP is `0·inf = NaN` in any d(loss)/d(S-field) adjoint —
  confined to dry lanes but violating the module's own masked-NaN discipline (cf. the
  `zdiff` guard at `eos.py:150-158`). Standard double-`where` fix.
- **[minor] [confirmed] `state.py:41,48`** — `del_ttf` and `uvnode_rhs` are dead `State`
  fields: `step()` never reads or writes them, so they ride every `lax.scan` carry as
  constant zeros. Under per-step `jax.checkpoint` that is ≈145 MB of dead residual per
  checkpointed step at CORE2 (float64) — material given the documented adjoint OOM
  battles. Dropping them is free.
- **[minor] [confirmed] `pgf.py:137`** — comment says "bottom wins single-layer overlap";
  the code makes the forward (surface) stencil win (`bwd_m & ~fwd_m` then selecting
  `dz_fwd` on overlap). The C's behavior for a single-mid-layer element is documented at
  `pgf.py:97-98` as C UB and the numpy-ref test skips `nlevels<3` elements — a
  comment/code inconsistency plus an untested divergence on degenerate elements, not live
  wrongness on current meshes.
- **[minor] [likely] `ssh.py:574-583`** — forward-mode JVP through `solve_ssh` can
  under-converge or return 0: the tangent solve reuses the absolute threshold
  `rtol_fwd = soltol·RMS(ssh_rhs)` derived from the *primal* rhs, which is scale-wrong
  for a (typically far smaller) tangent rhs. Reverse mode is unaffected; but `jax.jvp` /
  forward-over-reverse HVPs through the step (the README advertises TLM mode) would be
  silently inaccurate.
- **[minor] [confirmed] `integrate.py:126,143`** — `integrate()` unconditionally runs its
  first step with `is_first_step=True`, so resuming a warm state through it silently
  re-bootstraps AB2 while consuming the restored AB slot — neither continuation-faithful
  nor a clean cold start. `run.py` handles this correctly via explicit `bootstrap_ab2`
  flags; `integrate()` and `step.run` (the experiment/adjoint drivers) offer no option.
- **[nit] [confirmed] `ssh.py:458`** — the zero-rhs short-circuit tests the deflated
  `b_eff`, not `ssh_rhs` as the cited C line does; with a warm start and an exactly-zero
  rhs the loop would run all 500 iterations. Unreachable in practice.
- **[nit] [confirmed] `eos.py:143` / `step.py:465`** — under zstar the JAX computes and
  carries `hpressure` every step while the C computes none; documented "unused" but
  still materialized and stored (wasted compute + an incomparable diagnostic).

### Verified clean

The reversed `eta_n` init blend weights (documented lesson); the residual measured
against original ‖ssh_rhs‖ (warm-start fidelity); `area` vs `areasvol` in `compute_w`;
the surface-interface edge replication in vertical momentum advection (exact for the
no-cavity scope); horizontal momentum advection carrying no thickness weight (matches the
C); raw `segment_sum` on `mesh.edges` (no −1 sentinels there — only `edge_tri` has them,
and it goes through the masked scatter).

### Test-coverage gaps

(1) The active w-split path in `impl_vert_visc` has only identity-at-zero + finiteness
tests (acceptable — w_split is production-off); (2) degenerate `nlevels<3` PGF elements
untested (C UB); (3) no gradient test evaluates `jm_components` at S=0; (4)
`test_step_core2.py` holds only a rest-state test — the real CORE2 forced gates live in
other files, making the name misleading.

---

## 4. Tracers & vertical mixing (`tracer_adv.py`, `tracer_diff.py`, `kpp.py`, `pp.py`, `tke.py`, `cvmix_tke.py`, `tke_nn.py`, `gm.py`, `gm_redi.py`)

### Assessment

**No critical or major defects found** — the subsystem is unusually well-oracled
(per-kernel C dumps + a fully independent 200-line loop-based numpy FCT reference +
property gates like bit-exact constant-tracer preservation). The FCT path was
cross-checked line-by-line against the independent numpy reference (flux signs, level
bounds, Zalesak limiter, assembly all agree), and the conservation structure was verified
(antisymmetric edge scatters, telescoping vertical divergences, symmetric diffusion
coupling). Discontinuity handling is principled: discrete indices (`kbl`, table bins)
stop-gradiented while continuous weights stay differentiable, with a dedicated design doc
for limiter subgradients. `tke_nn.py`'s structural guarantees are real (bounded,
positive-definite for any weights; zero-last-layer ⇒ bit-exact identity, tested).

### Findings

- **[minor] [speculative] `kpp.py:627`** — the `kn = min(kbl − caseA, nzmax − 1)` clamp
  may diverge from the C for bottomed-out unstable columns (`kbl == nzmax`, `caseA == 0`):
  matching-level gathers get `dthick[nzmax−1]` instead of `dthick[nzmax]`. The all-node
  step-1 Kv/Av dump gate passes, so either the C clamps identically or no CORE2 column
  exercises the case; needs a C cross-read (`fesom_kpp.c:449-579`) to close.
- **[minor] [confirmed] `tracer_adv.py:118`, `tke.py:170`, `tke_nn.py:112`,
  `cvmix_tke.py:195`** — several no-cavity (`ulevels ≡ 1`) assumptions are silent:
  surface handling hard-codes column index 0. Unlike other un-ported branches (ddmix,
  IDEMIX) nothing raises — a cavity mesh would run and be silently wrong at the surface
  rather than fail loudly.
- **[minor] [speculative] `cvmix_tke.py:233-235`, `tke.py:171`** — index gathers on
  sharded padding nodes (`nlevels_nod2D = 0`) rely on negative-index wrap semantics of
  `take_along_axis` (gathering at −2 yields finite garbage that downstream masks zero
  out). Safe today, but this exact pad-node class already produced a masked-NaN gradient
  bug once (fixed at `cvmix_tke.py:219-227`); an explicit `maximum(nlev, 1)` clamp would
  be robust.
- **[minor] [likely] `tke.py:113-115`** — code-vs-comment mismatch on which bottom
  interface `_wire_kv_av` zeroes (docstring cites the C's below-bottom index; code zeroes
  the bottom interface itself). Downstream-invisible and dump-gated, so the code is almost
  certainly the faithful one — fix the comment before someone "corrects" the code.
- **[nit] [confirmed] `cvmix_tke.py:41`** — `_MXL_BIG = 1.0e30` is dead, and its comment
  misdescribes the current mixing-length mechanism.
- **[nit] [confirmed] `tracer_adv.py:471-474`** — `flux2dtracer_fct` docstring overstates
  the `dt/areasvol` scaling (the LO transition term enters unscaled — the code is right).
- **[nit] [confirmed] `gm_redi.py:103`** — `vd_dn` built by rotation
  (`concatenate([vd[:,1:], vd[:,:1]])`) rather than the zero-pad `_shift_up` idiom used
  everywhere else; safe for two independent masking reasons, but invites copy-paste error.
- **[nit] [confirmed] `kpp.py:760,800`** — `assemble_mixing` returns `diffKt` twice in a
  6-tuple (as `Kv` and as `diffKt`); positional-unpacking hazard.

### Verified clean

Zalesak limiter (bounds, ±bignumber pads, `flux_eps` signs, horizontal
`min(plus[n1],minus[n2])` pairing) exactly matches the numpy C-reference; `adv_flux_ver`'s
unified surface formula; `tracer_diff` increment-form Backward-Euler conserves
`Σ areasvol·h·T` up to BCs; `fill_up_dn_grad` node-based bounds are faithful (an initially
suspected triangle-vs-node mismatch was disproved); the `sw_3d` divergence asymmetry is
the C's exact form; PP `factor³`/`mean(factor²)` algebra; GM `fer_solve_gamma` is
diagonally dominant (no TDMA pivot risk); K33-via-`Kv+=aug` reproduces the C exactly given
`impl_vert_diff`'s coefficient layout.

### Test-coverage gaps

(1) `k33_augmentation` has no isolated dump gate (sanity-only, acknowledged); (2)
`adv_tra_vert_impl` (w_split) has no numeric oracle for the active tridiagonal
(production-off, mitigated); (3) KPP gradient quality near the `kbl` crossing is tested
for finiteness only — inherent to the subgradient design but undocumented as a limit; (4)
nothing exercises cavity meshes (consistent with the no-cavity scope).

---

## 5. Sea ice (`ice*.py`)

### Assessment

Line-faithful port orchestrated in the C runtime order. Both rheologies are 120-iteration
`lax.scan`s with `jax.checkpoint` bodies; genuinely shared blocks are factored into
`ice_evp.py` with 14 documented C-fidelity traps covering intentional EVP/mEVP
differences. The recent sublimation fix (5a61bb0) was re-verified term-for-term against
the C and is **correct and self-consistent**.

### Findings

- **[major] [confirmed — independently re-verified against the C] `ice_step.py:139-147`**
  — the zstar freshwater balance passes the wrong `a_ice_old`: start-of-step
  `state.a_ice` instead of the thermo-entry (post-advection/cut_off) concentration
  `a_co`. The C saves `values_old` inside the thermo loop *after* advection/cut_off and
  *before* overwriting with thermo's outputs (`fesom_ice_thermo.c:497-506`), and that is
  what the balance consumes (`fesom_ice_coupling.c:197-207`). The JAX comment
  ("a_ice_old = the PREVIOUS step's concentration") misreads this. Consequence: with the
  correct value, `prec_snow·(1−a_old)` cancels thermo's `snow·(1−A)` exactly and the
  post-balance global ⟨water_flux⟩ ≡ 0; with `state.a_ice`, a residual
  `⟨prec_snow·(A_entry − A_state)⟩` leaks into the volume budget every step — the same
  failure class as the sublimation leak, smaller and sign-fluctuating. One-line fix (pass
  `a_co`). The existing balance test (`test_ale_zstar.py:380`) re-derives the
  implementation's own formula and cannot catch it.
- **[minor] [confirmed] `ice_thermo.py:314`** — latent `AttributeError`: `cfg.ref_sss`
  does not exist on `IceConfig` (only `ref_sss_local`). Dead under the default; any
  config flipping `ref_sss_local=0` crashes at trace time.
- **[minor] [confirmed] `sss_runoff.py:295-298`** (via `ice_coupling.ice_oce_fluxes`) —
  the relax_salt global-mean subtraction is cavity-masked in JAX but not in the C
  (`fesom_ice_coupling.c:171-174`); the C masks only virtual_salt. No-op on current
  meshes; a fidelity landmine for any future cavity mesh, invisible to current gates.
- **[minor] [likely] `ice_step.py:118-131`** — the halo-refresh comment claims `t_skin`
  is "auto-complete", but it is computed from FCT-scatter outputs that are
  halo-incomplete, and it is a recurrent Newton warm-start state — halo `t_skin` (and
  halo bc_T/bc_S/flux diagnostics) permanently diverge from owner values under sharding.
  Benign today; the "halo == owner" invariant for a persistent State field is silently
  broken.
- **[nit] [confirmed] `ice_thermo.py:258-260`** — `ThermoOut.evap` and
  `ThermoOut.evaporation` are identical duplicates post-5a61bb0; the leak happened
  precisely because a consumer picked the field with the wrong historical semantics.
- **[nit] [confirmed] `ice_evp.py:225-229`** — std-EVP `velocity_update` zeroes cavity
  nodes where the C preserves old velocity; unreachable in the declared no-cavity scope
  (mEVP handles it correctly).

### Verified clean

The sublimation fix's full term-for-term match; the bc_S sign chain incl. flooding
correction; the deferred zstar bc_T; the odd-looking 0<a≤0.001 stress blend (exactly the
C); FCT `um = ΣU` not mean; mEVP iteration line-for-line vs `fesom_ice_maevp.c:224-322`;
the EVP 2×2 solve; no accidental EVP/mEVP drift; systematic AD guards; the `ice_dt`
force-derivation closing the historic desync by construction.

### Test-coverage gaps

(1) **No conservation regression on the assembled zstar ice step** — a ⟨water_flux⟩≈0
post-balance assertion on a real state would have caught both budget bugs; (2) the zstar
coupling path has no C-dump gate (both budget bugs lived exactly there); (3) `cut_off`
has no direct unit test; (4) the tiny-concentration stress-blend band and the
`ref_sss_local=0` branch are unexercised (the latter would crash); (5) no test that a
non-default dt propagates into EVP `dte`/thermo rates.

---

## 6. Forcing, initial conditions, observations (`forcing*.py`, `jra55.py`, `surface_forcing.py`, `sss_runoff.py`, `ic.py`, `phc_ic.py`, `obs_*.py`)

### Assessment

Cleanly split into host-side numpy readers and device-side AD-safe per-step math;
`forcing_local.py` (the per-process sub-mesh reader) is verified bit-identical to the
global build. Calendar handling is proleptic-Gregorian via `datetime`, exact across leap
years and dt-ramps. **No major findings** — the highest-risk areas (year-boundary record
bracketing, lon wrap in all three distinct interpolation routines, unit conversions,
L&Y09 bulk coefficients, leap years, the diurnal-SST fix) were explicitly checked and
cleared.

### Findings

- **[minor] [confirmed] `forcing.py:263`** — the no-ice
  `water_flux = evap − prec_rain − prec_snow` carries evaporation with sign opposite to
  the ice-on path (`obudget`'s evap is negative under evaporation), so a no-ice linfs run
  gets an inverted evaporative virtual-salt flux; the "+up" docstring labels are
  internally inconsistent. Bit-verified C-faithful, and production (ice on) never uses
  it — but it silently affects ocean-only gradient/dev configs. Deserves a documented
  decision rather than a silent inherit.
- **[minor] [confirmed] `obs_compare.py:100-106`** — `build_h_map` excludes nodes by
  latitude only; a *regional* (lon-limited) obs grid would silently absorb
  wrong-longitude nodes into its edge columns. Correct for the documented global grids;
  no guard.
- **[minor] [confirmed] `jra55.py:509-516`** — the coefficient cache degenerates to a
  per-step re-read+re-interpolation of 8 × (320×640) slices during the ~1.5–2.25 h
  windows at each year edge (values stay correct; only caching degenerates).
- **[minor] [confirmed] `sss_runoff.py:143,171`** — cells never reached by the 30-cell
  expanding fill keep the −1e30/−99 sentinel and are interpolated unguarded; safe on
  current meshes, silent O(1e30) restoring target on a future mesh with an enclosed
  basin. A post-interp assert would fail loudly.
- **[minor] [likely] `forcing_local.py:83-87`** — `stack_partitioned` allocates the full
  `[npes, n_steps, Lmax]` host array per field though only local rows are used (~16× the
  needed host memory at NG5 scale).
- **[nit] [confirmed] `jra55.py:466-467`** — the only out-of-bounds guard on bilinear
  corner indices is a bare `assert` (stripped under `python -O`).
- **[nit] [confirmed] `run.py:144` vs `surface_forcing.py:289`** — the two date generators
  disagree on sub-second handling; moot for integer dt.
- **[nit] [confirmed] `jra55.py:330-334,504`** — files checked for a shared grid but not
  a shared `calendar` attribute.
- **[nit] [speculative] `forcing_local.py:113-115`** — a process with zero local
  partitions dies with an opaque ValueError (cannot happen with the current mapping).
- **[nit] [confirmed] `sss_runoff.py:302-305`** — the standalone water balance mixes sign
  conventions (`⟨water_flux + runoff⟩`, +up vs +down); coherent only because the balanced
  value is inert under linfs and bypassed by the ice-on path. Dump-verified C fidelity —
  flagged so nobody "fixes" it or starts consuming it in a new path.

### Verified clean

JRA record bracketing at year boundaries and mid-interval shift semantics; leap years
(incl. 1960 in `test_run_entry.py`); lon wrap in all three interpolation routines; month
indexing; unit conversions; restoring sign/strength (10 m / 60 days piston); L&Y09 bulk
coefficients incl. documented deliberate quirks; the host/device dtype boundary; the
`LocalForcing` lane-layout assumption (bit-identity tested); the static-`a_ice` gating
decision; the diurnal-SST fix (period-aligned chunks + every-step `_MeanStream`);
`reopen_year` grid-invariance guard and resume wiring.

### Test-coverage gaps

(1) No JRA year-edge behavior test, nor a Dec 31→Jan 1 hand-off through `reopen_year`
against an oracle; (2) no leap-year JRA read test (Feb 29 / record-366 indexing); (3) no
sentinel-free assertion on the filled climatologies; (4) `ic.py` covered only indirectly;
(5) no negative test for regional obs grids.

---

## 7. Distributed infrastructure (`mesh.py`, `partit.py`, `shard_mesh.py`, `halo*.py`, `integrate*.py`, `canonical_redist.py`, `reductions.py`)

### Assessment

The sharding model mirrors FESOM's MPI decomposition exactly: `dist_N` partitions read
into per-device padded bundles with lane order `[interior | halo | pad]`, two halo
transports (all_gather and point-to-point `ragged_all_to_all`) built from a common owner
map (hence byte-comparable), and a single `shard_map` running the unmodified `step()`
with exchanges at the C's exchange points (`halo_points.py` is the schedule-as-data).
The dead-branch discipline (`halo_ctx=None` ⇒ byte-identical dense graph) keeps the
single-device suite as a permanent regression oracle. The CG solve's psum'd residual
makes the data-dependent trip count device-identical (no deadlock).

### Findings

- **[major] [confirmed — independently re-verified] `integrate_sharded.py:454-490`
  (also `halo.py:91`, `ssh.py:317`)** — nothing enforces forward-only use of the ragged
  halo despite the documented broken `ragged_all_to_all` autodiff transpose (gradient
  over-counts ~`axis_size`×, GPU-only). `run_steps_sharded(return_grad_fn=True,
  use_ragged=True)` builds `bodyg` with the ragged context wired in and returns a
  differentiable `run(p)` — silently wrong gradients. Grads are only ever *tested* with
  `use_ragged=False`; a one-line raise (or a transpose-blocking `custom_vjp` on
  `halo_exchange_ragged`) closes it.
- **[minor] [likely] `integrate_sharded.py:663-667`, `canonical_redist.py:272-282`** —
  `id()`-keyed executable/maps caches hold no reference to the keyed objects; after GC,
  address reuse can return a stale executable/maps for a rebuilt setup. Usually a
  shape-mismatch crash; equal-Lmax setups would corrupt silently. Weakref-based keying
  removes the hazard.
- **[minor] [confirmed] `ice_evp.py:254-255`** — sharded ice silently falls back to a
  local-mesh boundary mask if `boundary_node` is omitted; under `shard_map` the local
  mesh carries the *global* `edge2D_in` against a local edge ordering, so partition cuts
  become artificial coasts. All current callers pass the mask; a loud guard in `step.py`
  would prevent silent misuse.
- **[minor] [confirmed] `shard_mesh.py:182-212,252-265`** — no build-time assertion that
  every halo lane has an interior owner (`owner == -1` corrupts silently in both
  transports) nor that owned `elem_nodes[:myDim]` are sentinel-free. FESOM partitions
  guarantee both; a new partition-generator bug would pass unchecked.
- **[minor] [confirmed] `halo.py:84-85` vs `shard_mesh.py:211`** — pad lanes are *not*
  "unchanged" under the all_gather exchange (they receive own lane-0's value; the ragged
  path genuinely preserves them). Forward-safe, but the vjp scatter-adds pad cotangents
  onto interior lane 0 — correct only while pad cotangents are exactly zero (empirically
  asserted, not structurally guaranteed). Identity-mapping pads makes the transpose
  unconditionally safe.
- **[minor] [confirmed] `integrate_sharded.py:675-676`, `canonical_redist.py:179-186`** —
  with `FESOM_REUSE_EXE` the executable is reused but all constant inputs (folded mesh,
  halo maps, SSH operator, static forcing) are re-placed on device every chunk;
  `redistribute_fields` re-folds + re-places its six offset maps per output write.
  Cacheable under the same key.
- **[minor] [confirmed] `shard_mesh.py:501-527`** — the sharded-mesh export/load
  round-trip silently drops the ragged exchange maps; a `use_ragged=True` run on a
  reloaded bundle dies with an opaque `TypeError`.
- **[nit] [confirmed] `canonical_redist.py:65-120`, `shard_mesh.py:252-303`** — map
  builders are per-entity pure-Python loops (minutes of host time at NG5 scale;
  vectorizable).
- **[nit] [confirmed] `halo_points.py:54-55` vs `step.py:285-286`** — schedule/code drift
  on `sw_alpha`/`sw_beta` (the "single source of truth" says exchange; the wiring
  correctly doesn't).
- **[nit] [confirmed] `integrate_sharded.py` (×4 sites)** — four near-identical
  HaloCtx/step-loop assembly blocks; an exchange-context fix must be applied in four
  places.
- **[nit] [speculative] `canonical_redist.py:253`** — `int(shard.index[0].start)` assumes
  a concrete integer slice start; degenerate shardings would raise TypeError.

### Verified clean

`step.py`'s exchange wiring matches `OCEAN_SCHEDULE` row-for-row; `reductions.global_sum`
cannot double-count; `partit.py` index shifts match the C consumer; `canonical_redist`
send/recv alignment, tail clipping, and pad-slot drops are correct and the fixed chunk
grid gives genuinely partition-independent bytes (tested P=2≡P=4); the
per-chunk-recompile fix and multi-GPU roundoff non-determinism are documented and handled.

### Test-coverage gaps

(1) No test that grad-through-ragged is blocked (it isn't); (2) `redistribute` is
GPU-only so CPU CI never runs it, and the true multi-host disjoint-write split is
untested (acknowledged); (3) no property tests of the partition invariants the build
relies on; (4) `_FORCED_JIT_CACHE` keying has no unit test; (5) the sharded-gradient
group is known-slow post-Phase-8b, reducing how often the heaviest backward gates run.

---

## 8. Run driver, config, I/O (`run.py`, `run_config.py`, `io_dump.py`, `zarr_output.py`, `ushow_output.py`, `canonical_redist.py`, chain sbatch)

### Assessment

One YAML → `RunConfig` (frozen dataclass, unknown-key rejection) → chunked sharded scan →
restart → exit; campaigns are self-resubmitting sbatch chains resuming from an archival
pointer. The restart *design fundamentals* are right: every `State` leaf written
(dataclass-derived, locked by `test_no_leaf_dropped`), partition-independent canonical
layout (byte-identity across device counts, tested), and the `restart.latest` pointer
written via temp-then-rename *after* the data barrier — correct ordering. The failure
modes are in what happens around that happy path.

### Findings

- **[critical] [confirmed — verified against live SLURM state] `run_core2_hindcast.sbatch:102,123`
  and `run_forca20_hindcast.sbatch:79,97-99`** — the `TAG` clobber described in §1. As of
  this review the broken resubmit has already fired: pending job 26025085 will cold-start
  a junk canary chain in `runs/fesom.1958.033.00000` while the real chain (at ~1971)
  stalls, and the junk chain self-perpetuates because `.jobcount` lives inside the
  ever-changing `$OUT`. The pending forca20 job 26024433 inherits the bug on its first
  resubmit.
- **[critical] [confirmed] `run_*_hindcast.sbatch:104-117`** — silent cold-start
  fallback: any failure to resolve `CUR` (missing pointer, transient `.zattrs` read
  error) becomes `CUR=0` → cold start from 1958 that **overwrites** existing stores
  (`ushow_output.py:94,177` `mode="w"`; same-tag archival dirs rewritten in place). This
  is the exact mechanism of the 2026-07-03 incident (log evidence: core2 job#6 at
  728640 → job#7 `CUR=0 / restart=none`). Recommend: exit 1 (no resubmit) when
  `restart.latest` exists but `CUR` can't be read, and refuse cold start when
  `$ARCHIVE` is non-empty.
- **[critical] [confirmed] `run_dars_3yr.sbatch:63`** — `rm -rf $OUT` inside the
  cold-start branch of a self-resubmitting chain (reachable via the same `|| echo 0`
  class). Direct violation of the no-deletion rule; superseded but still committed and
  submittable.
- **[major] [confirmed] `zarr_output.py:94-123,164-206` + `run.py:424-428,617-625`** —
  rolling restart / `checkpoint_every` writes are in-place and non-atomic, with attrs
  written *before* data: a kill mid-write leaves a store whose `.zattrs` claims step=N
  but whose chunks mix step-N and previous-step data; `read_restart` has no integrity
  check. Mitigated in the current chains only because they resume from the archival
  stream (whose pointer ordering is correct); anyone resuming from `--restart-out
  $OUT/restart` carries the corruption mode.
- **[major] [confirmed] `run.py:433-439` + `zarr_output.py:246-275`** — "immutable, never
  overwritten" archival restarts ARE overwritten on a re-run of the same calendar span
  (tag is purely calendar-derived; `open_group(mode="a")` + full data rewrite). A
  crash-resume that re-crosses a boundary, or a branching experiment resumed from an old
  archive dir while keeping the same archive root (exactly what the docstrings
  encourage), silently rewrites the original chain's downstream archival dirs. Nothing
  enforces or documents that branching runs must change the archive root.
- **[major] [confirmed] `run_config.py:81-84,104-112` vs `run.py` / CLI** — dead YAML
  keys: `snapshot_every`, `checkpoint_every`, `restart_archive_out/period/length`,
  `forcing`, `output_dir` are validated but never read by the driver (only the CLI flags
  are wired; `configs/core2_full.yaml` sets `snapshot_every: 48` and
  `checkpoint_every: 1440`, both silently ignored). Worst case: a user sets
  `restart_archive_out` in YAML, believes archival restarts are on, and gets none.
- **[major] [confirmed] `run_forca20_hindcast.sbatch:64`** — `TOTAL=438000` omits the
  1960 leap day (1959–60 = 731 days × 360 steps/day = 263160, not 262800); the chain
  declares `FORCA20_HINDCAST_DONE` one day early (correct TOTAL = 438360). The model
  calendar itself is leap-correct (CORE2's TOTAL includes its 16 leap days).
- **[major] [confirmed] operational** — the forca20 chain is currently halted (job#5,
  26018697, exit=134 at ~99 GB MaxRSS after the incident reset it to step 0) and the
  CORE2 chain is re-running years already computed pre-incident; a `forca20_hind` job is
  pending (26024433) — verify the TAG it was submitted with before it starts.
- **[minor] [confirmed] `run.py:465-466,667-668`** — time-mean accumulators don't
  survive job boundaries: the second segment's rollover re-write (`mode="w"`) destroys
  the first segment's partial flush of the boundary-straddling period, keeping only the
  post-boundary fragment. Documented as accepted, but the overwrite (not merge) isn't
  stated; with FORCA20's monthly straddles this affects ~1 month per segment.
- **[minor] [confirmed] `configs/forca20_tke_mevp_prod.yaml`** — `partition: dist_32` +
  "all de-risked" is stale (dist_32 compile-hangs); production works only via the sbatch
  `PART=dist_16` override.
- **[minor] [confirmed] `ushow_output.py:94,130`, `canonical_redist.py:203`** —
  period-output stores are non-atomic (`mode="w"`, consolidate-at-end): a kill mid-flush
  leaves a truncated store with no completeness marker; backfill/consolidation scripts
  scanning the output dir can race an in-progress write.
- **[minor] [confirmed] `scripts/run_from_config.py:91-93,106-112`** — stale help text
  (`--daily-out` claims 5 fields, 12 are written; `--output-layout` help describes the
  old restart auto-fold).
- **[minor] [likely] `run_*_hindcast.sbatch:137-146`** — the compile-hang guard greps
  `"chunk 1/"`, which prints only after chunk 1 *completes*; a healthy-but-slow first
  compile+chunk gets killed and re-queued (observed firing on forca20 job#4). Bounded by
  the 3-strike hangcount; not a data-loss risk.
- **[minor] [likely] chains** — no singleton/lock guard: a manual submit while the auto
  chain is live gives two jobs concurrently writing the same `$OUT/restart` and racing on
  unlocked `.jobcount`/`.hangcount`; the hang guard also greps a hardcoded `-o` path and
  would kill healthy jobs submitted with a custom `-o`.
- **[nit] [confirmed]** — `_barrier` duplicated (`ushow_output.py:290`,
  `canonical_redist.py:341`); disjoint chunk-range writer math duplicated
  (`zarr_output.py:196-202`, `canonical_redist.py:303-307`);
  `write_snapshot`/`snapshot_due`/`OnlineStats` unreachable from the production CLI;
  `FESOM_REUSE_EXE` documented only in comments/sbatch.
- **[nit] [confirmed] `run.py:107`** — `plan_chunks` bootstraps AB2 only at `s==0` or the
  ramp step; a cold `state0` injected with `start_step>0` (API-only path) would skip the
  bootstrap. Unreachable from the CLI.

### Strengths

Restart design fundamentals (above); YAML unknown-key rejection + `defaults()`
bit-identity regression guard; the ramp-aware piecewise calendar is exact incl. leap
years and well tested; time metadata consistent across all three writers (int64 CF time,
leading time dim, consolidated metadata — the recent ushow fixes landed everywhere);
chain safety rails that do exist (non-zero exit stops the chain; `--diagnostics` keeps
NaNs out of restarts; hang/runaway guards).

### Test-coverage gaps

(1) No crash/partial-write test (resuming from a half-written restart; detecting a
truncated output store); (2) no idempotence test for re-running a calendar span; (3) no
test that YAML `checkpoint_every`/`snapshot_every`/`restart_archive_*` take effect —
would have caught the dead-config drift; (4) time dtype/CF attrs asserted in no pytest
(only sbatch smoke jobs outside CI); (5) no leap-day-crossing output/boundary test; (6)
the sbatch chain logic (pointer read, CUR resolution, resubmit env) is entirely
untested — the TAG clobber and the leap-day TOTAL are exactly the class a tiny bash test
harness would catch.

---

## 9. Tests & scripts

### Test suite

**610 test functions across 77 files.** Exactly **1 xfail** (`test_halo.py:202`, the
upstream JAX 0.10.1 `ragged_all_to_all` transpose bug — verified still current in the
production env; `strict=False`, so a fixing JAX upgrade will silently XPASS instead of
flagging that the planned `custom_vjp` workaround is obsolete). ~57 `skipif` + 50 inline
skips, **all environment/data availability** (missing mesh exports, C-dump fixtures,
device counts, GPU-only collectives) — none mask product bugs; `conftest.py` centralizes
the skip-if-dump-missing pattern.

**Assertion quality is strong**, in three principled tiers: golden C-dump gates with
calibrated per-field tolerances asserting zero violating nodes over all 126 858
(`test_ice_thermo.py:37-81`); analytic/structural invariants (rest-state <1e-12 with
bit-exact tracer preservation; `RunConfig.defaults()` bit-identity per State leaf; exact
integer calendar arithmetic; exact unit Jacobians); and explicit anti-triviality guards
(asserting the dump has >5000 iced AND >5000 open-water nodes before gating; asserting
KPP genuinely differs from PP before equivalence claims). All `rtol` ≤ 1e-9; the loosest
atols are documented physics floors (FCT upwind-flip divergence), not laziness. No
tautological asserts found.

**Hermeticity:** `data/` is a symlink to /work, so most of the suite is Levante-bound by
design. A real hermetic subset exists and runs in CI (`.github/workflows/ci.yml`, 5
data-free files per push), and the full suite runs via `scripts/runs/run_suite.sbatch` (3
groups, 1:45 wall). Weakness: both are **hand-listed filenames, no pytest markers
registered at all** — a new hermetic test file is silently excluded from CI until someone
edits ci.yml.

Suite-level nits: `test_canonical_redist.py:85,95,109` uses the
`assert_array_equal(...), m` trailing-tuple pattern (the failure-context message is
discarded; the assert itself still works); three files hardcode `/work/ab0995/...`
absolute oracle paths instead of routing through the `data/` symlink.

### Scripts

Deletion-safety audit (all `rm -rf`/`rmtree`/`mode="w"` hits classified by reachability):
the three chain-reachable findings are in §8 (dars chain `rm -rf`, hindcast silent
cold-start overwrite, ushow `mode="w"`). Additionally:

- **[minor] [confirmed] `run_ng5.sbatch:54`** — `rm -rf "$OUT"` fires on the default
  (no-`RIN`) invocation; manual script, but an accidental bare resubmit deletes prior
  output.
- One-shot bench/smoke scripts deleting their own scratch dirs (attended, acceptable):
  `bench_*_scaling_io.sbatch`, `run_core2_e2e.sbatch`, `run_perf.sbatch`, and Python
  `shutil.rmtree` in four bench/validate scripts.
- Chain-logic positives worth recording: a non-zero srun **stops** both hindcast chains
  (no resubmit-on-failure loop); a walltime hard-kill safely stalls (documented); the
  hang guard has a 3-strike cap.
- Rot: 670 files in `scripts/`, only 205 tracked; ~450 untracked job logs / result
  artifacts / figures sit beside the 4 live production chains with no naming or directory
  separation. Import spot-check found **zero dead module references** — no API rot.
- sbatch hygiene: the cheap-job convention (`-p compute`, ≤30 min) is followed
  consistently; paths and the interpreter are hardcoded to this user's Levante layout in
  every script (consistent, non-portable).

---

## 10. Recommended actions, ranked

**Now (before the pending chain jobs run):**
1. Fix the `TAG` clobber in both hindcast sbatch scripts (rename the pointer-read
   variable); cancel/resubmit the already-poisoned pending core2 job 26025085 with the
   correct `TAG`, and check what `TAG` the pending forca20 job 26024433 carries.
2. Make cold-start explicit: in both chains, exit 1 (no resubmit) when `restart.latest`
   exists but `CUR` cannot be read, and refuse to cold-start when `$OUT`/`$ARCHIVE` is
   non-empty. Remove `rm -rf $OUT` from `run_dars_3yr.sbatch`.
3. Fix `TOTAL=438360` in the forca20 chain (leap day).
4. Commit the working tree — production is running uncommitted code.

**Soon (correctness):**
5. `ice_step.py:146`: pass `a_co` as `a_ice_old`; add the assembled-step
   ⟨water_flux⟩≈0 conservation gate that would have caught both budget bugs.
6. Guard the ragged-halo gradient path (raise in `bodyg`/`solve_ssh` when
   `use_ragged=True` under differentiation, or `custom_vjp` that errors on transpose).
7. `step.py:250`: drop the double `DENSITY_0` subtraction (accepting the rounding-floor
   perturbation of zstar chains), or document why it stays.
8. Wire the dead `RunConfig` keys to the driver (or delete them from the schema); add a
   test that YAML keys take effect.
9. Make restart writes atomic (write to temp dir + rename; or write a completion marker
   `read_restart` requires) and enforce/document the archival-immutability contract for
   branching runs.

**When convenient (robustness/perf):**
10. `eos.py:63` sqrt(S) double-`where`; drop the dead `State` fields (`del_ttf`,
    `uvnode_rhs`, ~145 MB/step of dead adjoint residual); `IceConfig.ref_sss` field;
    build-time partition invariant asserts in `shard_mesh.py`; weakref the id()-keyed
    caches; cache the re-placed constants under `FESOM_REUSE_EXE`; fix the JRA year-edge
    cache degeneration.
11. Register pytest markers (hermetic/slow/data) and drive CI + run_suite off markers
    instead of hand-listed filenames; make the ragged xfail `strict=True` pinned to the
    JAX version; add the missing oracle tests listed per subsystem (JRA year-edge +
    leap-day read, crash/partial-write restart, chain-script bash harness).
12. Repo sweep: separate production chains from experiment scripts, gitignore/remove
    stray logs, fix the stale `dist_32` claim in `forca20_tke_mevp_prod.yaml`, refresh
    stale CLI help text.

---

*Review conducted 2026-07-03 by seven parallel subsystem reviews plus orchestrator
re-verification of all critical/major findings. No code was modified.*
