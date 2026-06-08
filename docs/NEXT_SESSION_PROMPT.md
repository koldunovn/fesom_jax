# Next-session prompt — FESOM2 → JAX port: PHASE 8 (multi-GPU / multi-core sharding)

> **⚡ STATE (2026-06-08): S.1→S.7 are DONE and COMMITTED on `main` (the HEAD commit — `git log --oneline -5`).
> The ENTIRE `shard_map`
> wiring is N-vs-1 correct on owned: the OCEAN exchanges, the forced-path exchanges (+GM, +KPP, +ice), the
> distributed reductions, the multi-step `lax.scan`, AND the in-scan collectives (the CG `while_loop` + the
> ice EVP 120-subcycle scan). The PRIMARY assembled KPP+GM+ice gate PASSED (every clean ocean field
> machine-precision, `uv` 1.9e-16; T/S the climate-close FCT floor; ice fields bit-exact). Single-device suite
> byte-identical `v1.0`. NEXT: S.8 (the AD gradient gate), then S.9 (the real 2–4×A100 GATE), S.10 (docs/tag
> `v1.1-multi-gpu`).**
>
> Commits: `80f6a71` S.1–S.5 foundation · `497e802` S.6 CG · `a770000`+`990da45` S.7 part 1 · `9bc9c39`
> S.7 part 2 · **HEAD = S.7 part 3** (GM/KPP/ice/reductions/multistep + the PRIMARY assembled gate). Plan
> Revision Log #2→#12 records every decision/discovery (#11+#12 = S.7 part 3).

## ⚠️ READ FIRST — two hard rules this session learned
1. **Run every multi-minute `shard_map`/model compile via `sbatch` on `-p compute`, NEVER on the login
   node** (`levante0`). The full assembled step under `shard_map` is a ~2-min XLA compile + GBs of RAM;
   running it directly (the Bash tool runs on login) is antisocial + killable. Only lightweight host-side
   numpy checks (mesh/stencil/connectivity audits, seconds) may run on login. Pattern:
   `scripts/test_step_sharded.sbatch` (`-p compute -A ab0995 --mem=80G --time=00:25:00`,
   `XLA_FLAGS=--xla_force_host_platform_device_count=N`); submit + watch with a background `while squeue…`
   loop. `hostname`→`levante0.*` + no `$SLURM_JOB_ID` = login. See `[[hpc-job-file-conventions]]`.
2. **Use the reference ports' exchange maps — they ARE the checklist.** `port_kokkos/docs/SYNC_MAP.md` is
   the authoritative per-substep internal-exchange (`D21`) list; it caught two scatter-result exchanges in
   part 2 (`un_u`, `tr_xy`) that the `MPI_PORT_REPORT` table folds inside a kernel. Read the relevant substep
   row BEFORE wiring, not after debugging.

## START HERE — the plan is the source of truth; do NOT re-plan
1. **READ `docs/plans/20260607-fesom-jax-phase8-sharding.md`** — the full Phase-8 sub-plan. Revision Log
   #2→#12 records what each task built; **S.1–S.7 are ALL ticked; S.8 onward open** (S.8 = the AD gate).
2. **READ the Phase-8 lessons** in `docs/PORTING_LESSONS.md` (every "Phase 8 — sharding (Task S.x)" section).
   The load-bearing ones for **S.8 (AD)**: the **masked-NaN rule across the device axis** (padded/halo lanes
   must compute a FINITE value + 0 cotangent — the Phase 3/5/6 discipline, now on the device-pad axis); 🎯
   **a collective lowers inside `jax.checkpoint`→`lax.scan`→`shard_map`** (the ice EVP scan — its FORWARD is
   verified; the BACKWARD recompute re-runs the `all_gather`, whose transpose is the reverse exchange); the
   CG `custom_linear_solve` `transpose_solve` already runs sharded (S.6). General part-3 context: 🎯 **JAX
   needs FEWER exchanges than the C** (only SCATTER results read at the halo need an exchange); 🎯 the **FCT
   upwind-flip floor / free-running multi-step chaos** is climate-close (Decision 4), so AD gates must also be
   FIELD-APPROPRIATE / teacher-forced if a free-running grad is noisy; the `check_vma=False` gating.
3. **Memory:** `[[fesom-jax-port]]`, `[[hpc-job-file-conventions]]`, `[[porting-lessons-log]]`.

## WHAT'S BUILT (S.1→S.7 ALL DONE — do NOT re-litigate)
- **`partit.py`** (S.1) reads `dist_<NP>`; **`shard_mesh.py`** (S.2/2b) builds the per-device padded
  `ShardedMesh` + `partition_state`/`partition_forcing_*`; **`halo.py`** (S.3) = `halo_exchange`
  (`all_gather`+gather) + the `HaloCtx` (S.7); **`halo_points.py`** (S.4) = `OCEAN_SCHEDULE`/`ICE_SCHEDULE`
  + `FUSED_KERNELS_NEEDING_SPLIT`; **`reductions.py`** (S.5) = `global_sum`/`global_dot` (owned-mask +`psum`).
- **`ssh.py`** (S.6) — `partition_ssh_operator` + `SSHHalo`; halo folded into `ssh_matvec`/`ssh_precond`;
  `_pcg`/`solve_ssh` gated on `halo=None`. Real CORE2 CG = 127/130 iters, N==1 deterministic, machine-precision
  `d_eta`; lowers on real 4×A100. Fixture: `scripts/capture_core2_ssh_rhs.py`→`data/ssh_rhs_core2/`.
- **`integrate_sharded.py`** (S.7 p1) — folds the `ShardedMesh`/`State`/`SSHOperator` to `[P*Lmax]`
  `PartitionSpec('p')`, reconstructs the per-device LOCAL `Mesh`/`State`/`SSHOperator` (Lmax static sizes;
  CSR dummy) inside `shard_map`, runs the step. **`check_vma=False`** (constant-carry scans). `run_step_sharded`
  builds + threads the `HaloCtx`+`SSHHalo`.
- **`step.py` / `momentum.py` / `tracer_adv.py`** (S.7 p2) — the OCEAN halo exchanges (one `_exch(field,kind)`
  closure; `halo_ctx=None` dead branch ⇒ byte-identical) + the fused-kernel splits (`visc_filt_bidiff` exch
  `Uc/Vc`; `momentum_adv_scalar` exch `un_u/un_v`; `advect_one_fct`/`zalesak_limit` exch `fct_LO`+`tr_xy`+
  `fct_plus/minus`). Gate `test_step_sharded.py` (npes==1 byte-identical; npes==2 owned field-appropriate).

- **`gm.py` / `eos.py` / `kpp.py` / `ice_*.py` / `sss_runoff.py` / `core2_forcing.py` / `integrate_sharded.py`**
  (S.7 p3) — the FORCED-PATH exchanges, ALL N-vs-1 correct on owned (gated, sbatch on compute):
  **GM** (`gm_diagnostics` exch `fer_gamma` INTRA + `fer_uv`/`slope_tapered`/`Ki` + `step.py` `fer_w`; the Redi
  `tr_xy/tr_z` auto-complete in JAX); **KPP** (`smooth_nod3D` per-sweep refresh + `viscA` before the `Av`
  gather); **reductions** (`_area_mean`→`owned_mask`/`psum`, + `_fold_forcing` folds the forcing to sharded
  inputs); **ice** (`evp_dynamics` `u_ice/v_ice` exch inside the 120-subcycle `lax.scan`; `fct_solve`
  low/high-order/limiter splits; the GLOBAL `boundary_node` partitioned in; ⚠️ `a_ice` exchanged BEFORE the
  `ice_oce_fluxes_mom` `stress_surf` gather, not at step-end); **multi-step** (`run_steps_sharded` = step-1
  eager + `lax.scan` rest). Gates: per-kernel `run_gm_diag_sharded` (bit-exact), `Kv/Av` ~1e-14, ice fields
  bit-exact, the PRIMARY assembled KPP+GM+ice gate (`uv` 1.9e-16). The few-step gate is TEACHER-FORCED (a
  free-running compare decorrelates chaotically — Decision 4).

## THE LADDER — start at S.8 (S.1→S.7 DONE)
- **S.8 AD gate** — `jax.grad` of the sharded run == single-device gradient; masked/halo lanes finite +
  0-cotangent (masked-NaN across the device axis). `custom_linear_solve` transpose already runs sharded (S.6).
  ⚠️ The ice EVP scan is `jax.checkpoint`'d with an in-scan `all_gather` — the FORWARD lowers (verified); the
  BACKWARD recompute re-runs the collective (its transpose = the reverse exchange = `psum`/reduce-scatter).
  The multi-step `run_steps_sharded` checkpoints its scan body (the AD-window memory cap). Start small (2-step
  OCEAN grad N-vs-1) then +forcing; reuse the teacher-forcing insight if a free-running grad is noisy.
- **S.9 the GATE** — (a) per-substep N==1 on 2/4×A100 (`-A ab0995_gpu -p gpu --gres=gpu:4 --nodes=1`),
  (b) gradient, (c) BONUS JAX-N ↔ C-N dump diff on `dist_2/4` (looser cross-runtime budget). Update
  `[[fesom-jax-port]]` memory at the gate.
- **S.10 docs** — tag `v1.1-multi-gpu`, move the plan to `completed/`.

## REPRODUCE / TOOLS
- **Env python (ALL python/pytest):** `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`.
- **Sharding tests → COMPUTE node sbatch** (see Rule 1): `scripts/test_step_sharded.sbatch` runs
  `test_step_sharded.py` on 4 fake-devices; `scripts/run_suite.sbatch` runs the full suite (ocean / ice /
  SHARDING) — "ALL GREEN" iff all three rc=0. The collective tests SKIP cleanly at 1 device.
- **S.7 part 3 gate scripts** (focused, per-increment — the proof; each a separate `-k` selection on
  `test_step_sharded.py`): `test_gm_red_sharded.sbatch` (GM + reductions), `test_kpp_sharded.sbatch` (KPP),
  `test_ice_ms.sbatch` (ice + multistep), `test_assembled.sbatch` (the PRIMARY KPP+GM+ice). ⚠️ A forced step
  under `shard_map` is a **~10–30 min compile** (the assembled is the heaviest, the most collectives) — budget
  `--time` generously. ⚠️ **`run_suite.sbatch`'s SHARDING group auto-collects `test_*_sharded.py`, so it now
  pulls the heavy forced tests ⇒ ~1.5–2 hr** — a FOLLOW-UP should add a `@pytest.mark.slow` to the forced
  `test_step_sharded.py` tests and `-m "not slow"` the SHARDING group (keep the focused sbatch jobs as the
  forced-test gates).
- **CORE2 partitions:** `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/dist_{2,4,…}`. Dense mesh
  `data/mesh_core2/`; IC `data/ic_core2/`; forcing via `core2_forcing.build_core_forcing` (JRA on `/pool`).
  `scripts/capture_core2_ssh_rhs.py` shows assembling the forced KPP+GM+ice step (`dt=1800`).
- **Reference ports (READ, MIRROR):** **Kokkos** `port_kokkos/docs/SYNC_MAP.md` (the authoritative
  per-substep internal-exchange checklist) + `SCATTER_STRATEGY.md` + `KOKKOS_PORTING_LESSONS.md`. **C MPI**
  `port2/fesom2_port/` — `fesom_partit.c`, `fesom_halo.c`, `MPI_PORT_REPORT.md`, the C-N dump for S.9c
  (per-rank `%05d`). **C edits → port2 `jax-mesh-export` branch, NEVER main.**
- I (Claude) drive SLURM (acct ab0995 / ab0995_gpu).

## WORKFLOW (standing rules)
- **Tests-first per task, gate on CPU fake-devices (sbatch on compute)**; every task ends green AND the
  single-device suite green. **Append a lesson per task** to `docs/PORTING_LESSONS.md`; tick the plan
  checkboxes + add a Revision-Log entry. Commit per-task on `main` ONLY when the user asks.
- ⚠️ **Recurring gotchas:** `halo_ctx=None`/`exch=None`/`kpp_cfg=None` must trace the EXACT `v1.0` graph
  (dead branch — the single-device suite is the proof); the masked-NaN AD rule (padded/halo lanes finite,
  across the device axis); `check_vma=False` for the step's constant-carry scans; the per-substep gate is
  FIELD-APPROPRIATE (don't demand 1e-12 of the FCT tracers / cancelling SSH divergences — that's the
  climate-close floor, Decision 4); float64 throughout.

Confirm you've read the plan + Revision Log #2→#12 + the Phase-8 lessons; then start **S.8 — the AD gradient
gate**. Concretely: `jax.grad` of a scalar loss of a sharded run (start with a **2-step OCEAN** `run_steps_sharded`,
no forcing) wrt `params` (e.g. `k_gm`) and the initial `state` (`T0`) == the single-device gradient (the `v1.0`
baseline) to tight tol; then masked-NaN check (padded/halo lanes finite + 0 cotangent across the device axis);
then +forcing (KPP+GM+ice). The forward already lowers (incl. the in-scan collectives + `custom_linear_solve`);
S.8 gates the BACKWARD. New tests go in a `test_gradient_sharded.py`; gate via a focused sbatch on compute
(the grad compile is heavier than the forward — budget `--time` generously). If a free-running multi-step grad
is noisy (chaos, Decision 4), gate the gradient teacher-forced or at 1–2 steps. S.7 part 3 is committed (the
HEAD commit; `git log --oneline -5` to confirm); the working tree is clean at the start of the S.8 session.
