# Next-session prompt — FESOM2 → JAX port: PHASE 8 (multi-GPU / multi-core sharding)

> **⚡ STATE (2026-06-08): S.1→S.7-part-2 are DONE and COMMITTED on `main`. The sharding foundation, the
> distributed CG (gated on real 4×A100), the device-mesh placement, and the whole OCEAN step's halo
> exchanges all work; the single-device suite is byte-identical `v1.0` (ALL GREEN 475+47+36). NEXT: S.7
> PART 3 — the FORCED-PATH exchanges (+GM, +KPP, +ice) + the multi-step scan + the `io_dump` per-substep
> gate. Then S.8 (AD), S.9 (the GPU GATE), S.10 (docs/tag).**
>
> Commits this far: `80f6a71` S.1–S.5 foundation · `497e802` S.6 CG · `a770000`+`990da45` S.7 part 1 ·
> `9bc9c39` S.7 part 2. Plan Revision Log #2→#10 records every decision/discovery.

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
   #2→#10 records what each task built; S.1–S.7(parts 1&2) are ticked, S.7-part-3 onward open.
2. **READ the Phase-8 lessons** in `docs/PORTING_LESSONS.md` (every "Phase 8 — sharding (Task S.x)" section).
   The load-bearing ones for part 3: 🎯 **JAX needs FEWER exchanges than the C** (per-node intermediates are
   auto-complete on the halo — only SCATTER results read at the halo need an exchange); 🎯 the **FCT
   upwind-flip floor** is climate-close non-determinism, not a bug; the `check_vma=False` + `halo_ctx=None`
   gating; the operator/loop-bound-by-value rules.
3. **Memory:** `[[fesom-jax-port]]`, `[[hpc-job-file-conventions]]`, `[[porting-lessons-log]]`.

## WHAT'S BUILT (S.1→S.7-part-2 — do NOT re-litigate)
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

## THE LADDER — start at S.7 PART 3
- **S.7 PART 3 — the FORCED-PATH exchanges + the multi-step gate.** Build up incrementally, each gated on
  CPU fake-devices (sbatch on compute) + single-device suite green:
  1. **+GM/Redi** (`gm_cfg=GMConfig()`): the Redi diffusion (`gm_redi.diff_part_hor_redi`) has internal
     `tr_xy` (elem) + `tr_z` (nod) scatter-result exchanges (Kokkos SYNC_MAP row 13-Redi); thread `exch` in.
     The GM coefficient chain (`gm.gm_diagnostics`) is per-node/elem maps → auto-complete on the halo (verify;
     likely only the post-kernel `fer_uv`/`slope_tapered`/`Ki` exchanges needed).
  2. **+KPP** (`kpp_cfg=KppConfig()`, needs CORE2 forcing): `kpp.mixing_kpp` has 2 internal exchanges
     (Kokkos: `smooth_blmc` = blmc×3 + the 3-sweep `eos.smooth_nod3D`; + the pre-elem-average). ⚠️ apply the
     redundant-compute lens: a per-node field feeding `smooth_nod3D` is auto-complete on the halo (no pre-smooth
     exchange) UNLESS it is itself a scatter/OBL-search result incomplete on the halo — verify per field.
     Also `sw_alpha`/`sw_beta` (EOS, KPP path).
  3. **+ice** (`ice_cfg=IceConfig()`): the `ICE_SCHEDULE` (halo_points.py) — the EVP subcycle `u_ice/v_ice`
     exchange is INSIDE the 120-step `lax.scan` (the hard one: a collective in `scan` under `shard_map`,
     `check_vma=False`); the ice FCT (`m_ice_lo`/`a_ice_lo`/`m_snow_lo` + `icepplus`/`icepminus`) splits like
     the ocean FCT; thermo `ustar`; oce_fluxes.
  4. **Reductions:** route `sss_runoff._area_mean` + `ice_coupling.ice_oce_fluxes` (virtual-salt / relax-salt
     / water-flux balances) through `owned_mask`/`axis_name` (the helpers exist; `owned_mask=None` ⇒ byte-id).
     `ocean_area` is the replicated static constant.
  5. **Multi-step `lax.scan`** under `shard_map` (the `integrate` body unchanged; `run_step_sharded` is the
     single-step template) — step-1-eager + scan-rest, for a few-step run.
  6. **PRIMARY gate:** a few-step assembled CORE2 step (KPP+GM+ice) sharded == single-device, **per-substep**,
     **field-appropriate** budget (momentum/SSH/ALE/EOS <1e-7; FCT tracers + cancelling SSH divergences to the
     climate-close upwind-flip floor — NOT 1e-12 for those; see the lesson). Reuse `io_dump` against a
     single-device CORE2 dump (capture it like `capture_core2_ssh_rhs.py` assembles the forced step).
- **S.8 AD gate** — `jax.grad` of the sharded run == single-device gradient; masked/halo lanes finite +
  0-cotangent (masked-NaN across the device axis). `custom_linear_solve` transpose already runs sharded (S.6).
- **S.9 the GATE** — (a) per-substep N==1 on 2/4×A100 (`-A ab0995_gpu -p gpu --gres=gpu:4 --nodes=1`),
  (b) gradient, (c) BONUS JAX-N ↔ C-N dump diff on `dist_2/4` (looser cross-runtime budget). Update
  `[[fesom-jax-port]]` memory at the gate.
- **S.10 docs** — tag `v1.1-multi-gpu`, move the plan to `completed/`.

## REPRODUCE / TOOLS
- **Env python (ALL python/pytest):** `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`.
- **Sharding tests → COMPUTE node sbatch** (see Rule 1): `scripts/test_step_sharded.sbatch` runs
  `test_step_sharded.py` on 4 fake-devices; `scripts/run_suite.sbatch` runs the full suite (ocean / ice /
  SHARDING) — "ALL GREEN" iff all three rc=0. The collective tests SKIP cleanly at 1 device.
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

Confirm you've read the committed plan + Revision Log #2→#10 + the Phase-8 lessons; then start **S.7 PART 3**
(+GM Redi exchanges first — the smallest forced increment — gated npes==2 vs single-device, sbatch on compute).
