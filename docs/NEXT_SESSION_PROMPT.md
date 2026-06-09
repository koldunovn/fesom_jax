# Next-session prompt — FESOM2 → JAX port: PHASE 8b (scaling) — the data-loading REWRITE

> **⚡ STATE (2026-06-09): Phase 8 DONE (tag `v1.1-multi-gpu`). Phase 8b (scaling) is well underway — the
> IMMEDIATE NEXT TASK is the data-loading rewrite, fully scoped in
> `docs/plans/20260608-fesom-jax-phase8b-scaling.md` → the "B.3 REWRITE PREP" section (read it first).**
>
> What's done in 8b (plan Revision Log #1–#6): B.0 = `ragged_all_to_all` halo (FORWARD validated on A100;
> JAX's autodiff transpose of it is broken → `custom_vjp` is B.0d, forward-only-safe for now). The
> **headline result is trustworthy**: on the FULL CORE2 model (real JRA1958 + PHC IC + prognostic ice,
> 4×A100), **JAX = 92.6 ms/step vs Kokkos-CUDA full-step = 117 ms — COMPARABLE / slightly faster** (the
> earlier "10× slower" was a compile-in-timing bug, fixed: compile-once + warm-call, 1 config/job).
> Multi-node bring-up (B.3 STEP 1-2) is VALIDATED: `jax.distributed` across 2 nodes (8 A100), cross-process
> `psum`/`all_gather` in `shard_map` correct; per-mesh PHC IC + real forcing are mesh-agnostic
> (`load_phc_ic` + `build_core_forcing`); farc/dars IC cached on `/work`, NG5 IC caching (job 25445934).
>
> **THE BLOCKER (the rewrite):** dars/NG5 full-model OOMs the GPU in SETUP (identical `1.34 GiB jit__where`
> for BOTH halos) because the data pipeline builds the FULL GLOBAL arrays as `jnp` on GPU 0 before sharding
> (`State.rest`/`phc_state` + `_fold`/`folded_*`/`_halo_arrays`). **Fix = host-build (numpy) the whole
> pipeline + always `device_put`-sharded** (contained, NOT per-subdomain) — also unblocks single-node
> dars-4. Verify dars-4 single → dars-8 multi → dist_16/32 → NG5 vs Kokkos `SCALING_M524.md`. The full
> change-list + risks are in the REWRITE PREP section. ⚠️ Honest scaling truth so far: all_gather is
> O(P·N_local) (OOMs at scale), ragged is the multi-node answer; JAX is per-step competitive with Kokkos.
>
> Commits: …`b28f621` S.8 · `ddc8fff` S.9 · `7b16d27` S.10/tag · then Phase 8b `4acb9a2` plan · `6ec4234`
> B.0a · `ba342f6` B.0b/c · `85d39b3` SSH-ragged · `b123726` farc bench · `51adf5c` full-model bench ·
> `633770d` (HEAD) multi-node + 1-compile timing. The 8b plan Revision Log #1→#6 records every step.
> Plan Revision Log #2→#14 records every decision/discovery (#14 = S.9).

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
   #2→#14 records what each task built; **S.1–S.9 are ALL ticked; S.10 (docs/tag) is all that's left.**
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

## THE LADDER — start at S.10 (S.1→S.9 DONE)
- **S.8 AD gate** ✅ **DONE** — `jax.grad` of the sharded run == single-device, field-appropriate; masked/
  halo/pad lanes finite + 0-cotangent; the ice-EVP checkpointed-scan backward runs. Found+fixed 8 backward
  bugs (7 masked-NaN guards + `jax.jit`-around-`shard_map` for the checkpointed-scan grad), all
  forward-byte-identical. See Revision Log #13 + the S.8 lessons.
- **S.9 the GATE** ✅ **DONE (real A100s)** — (a) the assembled N==1 forward on 2×A100 == single-device on
  every prognostic field; (b) the OCEAN gradient over NCCL == single-device. GPU-aware tolerances baked in
  (`_BYTE_ID_ATOL`, `_DIAG_FIELDS` σ-exclusion). Deferred to follow-ups: npes=4, the C-N dump diff (8b), and
  the forced-grad-on-GPU (EVP-scan memory). See Revision Log #14 + the S.9 lesson.
- **S.10 docs** ✅ **DONE** — tagged `v1.1-multi-gpu`; plan moved to `docs/plans/completed/`. **PHASE 8 is
  CLOSED.** ⬅️ **START HERE = Phase 8b** (the scaling fork below).

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

Confirm you've read the plan + Revision Log #2→#14 + the Phase-8 lessons (esp. the S.8 masked-NaN /
`jax.jit`-around-`shard_map` + the S.9 byte-id-is-a-CPU-property / EVP-σ-VP-kink lessons). **S.1→S.9 are
DONE** — the model is N-vs-1 forward- AND gradient-correct, VALIDATED ON REAL A100s. **Next:**
- **S.10 — close Phase 8** (small): tag `v1.1-multi-gpu`, move the plan to `docs/plans/completed/`. Then the
  fork below. (Deferred S.9 follow-ups, do only if the user asks: npes=4 forward; the JAX-N↔C-N dump diff on
  `dist_2/4`; the forced-grad-on-GPU once the EVP-scan backward fits GPU memory — `jax.checkpoint` the EVP
  subcycle scan or shrink subcycles for the gate.)
- **Phase 8b — scaling (the user's NG5 goal)**: the realistic ladder is **farc (638k, 1 A100) → dars (3.16M,
  4 GPU/1 node) → NG5 (7.4M×70, multi-node `jax.distributed`)**, validated against the Kokkos
  `port_kokkos/docs/SCALING_{NG5,FARC,DARS}.md` numbers. ⚠️ **STEP 0 is mandatory: replace the `all_gather`
  halo exchange (`halo.py`) with `ragged_all_to_all`** (point-to-point neighbour exchange — the `com_struct`
  slist/rlist is already in `partit.py`). The current `all_gather` is O(P·N_local) and gets WORSE with node
  count → scaling numbers on it are meaningless. The S.8 gate (`test_gradient_sharded.py`) + the S.7 forward
  gate (`test_step_sharded.py`) are the correctness ORACLES for that rewrite. NG5 needs the deep-mesh fixes
  the Kokkos `SCALING_NG5.md` notes (nl=70 > a hardcoded cap; the step-0 global-gather OOM).

The working tree is clean at the start of the next session (S.8 = the HEAD commit; `git log --oneline -6`).
