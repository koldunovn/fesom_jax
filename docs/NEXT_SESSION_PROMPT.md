# Next-session prompt — FESOM2 → JAX port: PHASE 8 (multi-GPU / multi-core sharding)

> **⚡ STATE (2026-06-08): S.1→S.9 are DONE on `main`. The model is VALIDATED CORRECT ON REAL A100s** (S.9,
> `scripts/phase8_s9_gpu.sbatch`, 4×A100 job 25430592). The ENTIRE `shard_map` wiring is N-vs-1 correct on
> owned (S.7), gradient-correct (S.8), and now confirmed on real GPUs: the assembled CORE2 step (KPP+GM+ice)
> sharded across 2 A100s == single-device on every PROGNOSTIC field — ocean dynamics clean (uv 1.1e-9, d_eta
> 2.6e-11, w 2.3e-13, Kv/Av 1e-14), FCT tracers + prognostic ice climate-close (T/S 1e-2…1e-3, ice
> u/v/m/a/snow 1e-7…1e-9) — and the OCEAN gradient matches over NCCL (`jax.grad`-thru-`shard_map`, d/d(k_ver)
> rel 3.75e-8). **Two GPU truths (neither a bug, both now baked into the gate):** byte-identity is a CPU
> property (GPU reassociation floor ~8e-9 → platform-aware `_BYTE_ID_ATOL`); the EVP stress σ is a
> non-prognostic VP-kink diagnostic (O(0.5) branch-flip, but the u_ice/v_ice it drives is correct to 1e-7) →
> EXCLUDED from the gate (`_DIAG_FIELDS`). The forced gradient OOM'd on GPU (memory, not correctness) →
> deferred. NEXT: **S.10** (tag `v1.1-multi-gpu`, move plan to `completed/`). THEN → Phase 8b (scaling:
> farc→dars→NG5) — but FIRST replace the `all_gather` halo exchange with `ragged_all_to_all` (the `all_gather`
> is O(P·N_local), fine for the 2–4-dev correctness gate but NON-scaling; the point-to-point `com_struct`
> slist/rlist is already in `partit.py`).**
>
> Commits: `80f6a71` S.1–S.5 · `497e802` S.6 CG · `a770000`+`990da45` S.7 part 1 · `9bc9c39` S.7 part 2 ·
> `5214cf0` S.7 part 3 · `b28f621` S.8 AD gate · **HEAD = S.9** (the real-A100 gate + GPU-aware tolerances).
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
