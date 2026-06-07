# Next-session prompt — FESOM2 → JAX port: PHASE 8 (multi-GPU / multi-core mesh sharding) — BUILD

> **⚡ STATE (2026-06-07): the full differentiable FESOM2 CORE2 model (KPP + GM/Redi + prognostic ice)
> runs 2 yr stable on ONE A100 and MATCHES the C-port-KPP climate to ~0.01 °C at all latitudes
> (tag `v1.0-single-gpu`). Phase 8 is PLANNED — the sub-plan is drafted, plan-reviewed, and committed.
> NOTHING sharded yet. NEXT: execute the plan, starting at task S.1 (the partition reader).**

## START HERE — the plan is the source of truth; do NOT re-plan
1. **READ `docs/plans/20260607-fesom-jax-phase8-sharding.md` FIRST** (committed `f2b4056`). It is the
   full Phase-8 sub-plan: Scope, locked Decisions, C-reference tables (file:line), the task ladder
   **S.1 → S.2 → S.2b → S.3 → … → S.10**, the GATE, and Revision Log (#1 = the plan-review pass). Work it
   task-by-task; tick `[x]` and log each task in the Revision Log as you go.
2. **What works (don't re-litigate):** Phases 0–6 + 6B + 6C COMPLETE; the single-device model
   (`step.py`/`integrate.py`, KPP+GM+ice behind static `kpp_cfg`/`gm_cfg`/`ice_cfg`) is differentiable
   end-to-end, 2-yr stable, climate-matches the C. Suite = 483 green. **Phase 8 must keep the `npes==1`
   path byte-identical to `v1.0` (suite stays green) — sharding is additive, gated behind a static arg.**
3. **Memory:** `[[fesom-jax-port]]` (locked Phase-8 decisions + the load-bearing gotchas),
   `[[hpc-job-file-conventions]]`, `[[porting-lessons-log]]`.

## THE DESIGN (locked this session, user-confirmed — see the plan for the full rationale)
- **Mirror the C MPI port EXACTLY:** broadcast-only halo exchange (owner→halo) + redundant
  `myDim+eDim(+eXDim)` compute + `psum` reductions. (Research confirmed BOTH the C `port2/.../fesom_halo.c`
  AND the Kokkos port use broadcast+redundant, **no additive exchange** — the *old* handoff's "additive
  `scatter_add`" framing was wrong.)
- **Read the bit-identical FESOM `dist_<NP>`** (port `fesom_partit.c`'s ASCII reader to numpy). CORE2
  partitions: `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/dist_{2,4,8,16,…,864}`. Do NOT generate
  (pymetis isn't installed; generating forfeits the C cross-check).
- **Single-process `shard_map`** over a 1-D device mesh, pad-to-`Lmax` + mask. Scale CPU-fake-devices →
  4×A100 (one node). Multi-node (`jax.distributed`) is deferred (Phase 8b).
- **GATE (this phase's deliverable): 2–4 device per-substep correctness.** N-device == 1-device JAX to
  ~1e-12/substep + the gradient gate. Bonus: per-substep JAX-N ↔ C-N dump diff on `dist_2/4`. **The 2-yr
  multi-GPU climate run is an explicit FOLLOW-UP (Phase 8b), NOT this phase's gate.**

## THE FOUNDATION (do these first, in order — each independently gateable on CPU fake-devices)
- **S.1** partition reader (`partit.py`, numpy port of `fesom_partit.c:68-208`) → `Partition` pytree.
- **S.2** sharded-mesh build (`shard_mesh.py`): local-index remap of ALL connectivity, pad-to-`Lmax`+masks,
  exchange index maps, npy export. ⚠️ **gather-on-sentinel safety** (`ops.gather` at `-1` returns the last
  lane, not 0 — clamp+mask).
- **S.2b** partition `State` + per-step forcing + IC builders (host-build → `partition_state`). Needed
  before S.4/S.6.
- **S.3** the broadcast `halo.py` primitive + the ported identity gate (`fesom_halo_identity_test`). ONE
  simple collective (`all_gather`/`all_to_all`); `ragged_all_to_all` is a perf follow-up.
- **S.4** exchange schedule (ocean **+ ice/EVP**, from `MPI_PORT_REPORT.md`) + the 2-device scatter gate;
  classify post- vs **intra-kernel** exchanges (the fused `visc_filt`/FCT/CG kernels must be split).
- Then S.5 `psum` reductions → S.6 distributed CG (🔴 **iter-count determinism is LOAD-BEARING** — loose
  `soltol` early-stop, a 1-iter drift = ~1e-5, verify on the CORE2 KPP+GM+ice config) → S.7 wire
  `shard_map` → S.8 AD gate → S.9 the GATE → S.10 docs.

## REPRODUCE / TOOLS
- **The correctness gate vehicle (no GPU needed until S.9):** CPU fake-devices —
  `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=N <env-py> -m pytest …` (N=2,4).
  This makes "N-device vs 1-device" a same-process, same-code comparison.
- **JAX 0.10.1** has `shard_map`, `ppermute`, `psum`, `all_to_all`, `ragged_all_to_all`, `all_gather`
  (all confirmed available).
- 1-GPU climate baseline (the reference to beat, for Phase 8b): `sbatch scripts/core2_kpp_climate_gpu.sh
  [years] [dt] [outdir]`; bias map: `HDF5_USE_FILE_LOCKING=FALSE <env-py> scripts/kpp_bias_map.py
  --year 1958 --jax-dir <dir>`.
- Multi-GPU (S.9): `-A ab0995_gpu -p gpu --gres=gpu:4 --nodes=1` (4×A100-80GB/node).

## KEY PATHS / COMPUTE
- JAX repo `/home/a/a270088/port_jax` (git `main`, tag `v1.0-single-gpu`). **Env python (ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`.
- **C MPI port = the parallelization SoT** (read, mirror — do NOT change unless dumping):
  `/home/a/a270088/port2/fesom2_port/` — `src/fesom_partit.c` (the `dist_<NP>` reader + `com_struct`),
  `src/fesom_halo.c` (broadcast `fesom_halo_exchange` + `fesom_halo_identity_test`),
  `docs/MPI_PORT_REPORT.md` ("Halo exchanges per timestep" = the exchange schedule), `docs/PORTING_LESSONS.md §4`
  (the `myDim+eDim` halo-bound rule). **C edits (e.g. to add an N-rank dump for the S.9 bonus) → port2
  `jax-mesh-export` branch, NEVER main.** Kokkos port `/home/a/a270088/port_kokkos/` = GPU-fidelity ref.
- **CORE2 `dist_<NP>` partitions:** `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/dist_{2,4,8,…}` (NOT
  under `/work/.../meshes/` — those are *other* meshes; `nares4/dist_768` etc. are extra format fixtures).
  The single-device mesh bundle (dense, no partition) is `data/mesh_core2/*.npy`; S.2 EXTENDS it into a
  per-`NP` sharded bundle. Large files → `/work`.
- I (Claude) drive SLURM (acct ab0995 / ab0995_gpu).

## WORKFLOW (standing rules)
- **Tests-first per task, gate on CPU fake-devices**; every task ends with new tests green AND the
  single-device suite (483) still green (run via `scripts/run_suite.sbatch` in two chunks on the compute
  node — the full set in one process exceeds login-node RAM).
- **Append a lesson per task** to `docs/PORTING_LESSONS.md`. Commit per-task on `main` when asked. Tick
  the plan checkboxes + add a Revision-Log entry per task. Update `[[fesom-jax-port]]` memory at the GATE.
- ⚠️ **Recurring cross-phase gotchas** (full detail in `docs/PORTING_LESSONS.md`): the masked-NaN AD rule
  (masked/padded lanes must compute a FINITE value, now across the device axis); `npes==1` must trace to
  the exact `v1.0` graph (static-arg dead branch, like `kpp_cfg=None`); CG iteration-count must not drift
  across partitions; float64 throughout (`jax_enable_x64`).

Confirm you've read the committed plan; then start **S.1** (the `dist_<NP>` partition reader) — the
foundation everything else builds on. Build S.1→S.2→S.2b→S.3 before wiring anything into the step.
