# Next-session prompt — FESOM2 → JAX port: PHASE 8 (multi-GPU / multi-core sharding) — INTEGRATION

> **⚡ STATE (2026-06-07): the sharding FOUNDATION + the distributed CG are COMPLETE — tasks S.1→S.6 done,
> all gated on CPU fake-devices AND (S.6) a real 4×A100 run, single-device suite still green (`v1.0` path
> byte-identical: full suite ALL GREEN 475+47+33). What remains: S.7 (wire `shard_map` around the full step
> + split 5 fused kernels — the big one), S.8 (AD gate), S.9 (the 2–4 device GATE on GPU), S.10 (docs).
> NEXT: start at S.7.**
>
> **S.6 result (Revision Log #8):** distributed SSH CG in `ssh.py` (`partition_ssh_operator` + `SSHHalo` +
> folded-exchange matvec/precond; `_pcg`/`solve_ssh` gated on optional `halo`). Two discoveries: (1) the SSH
> stencil EXCEEDS the node halo but every overshoot entry is EXACTLY zero ⇒ local matvec exact on owned rows
> (asserted by value, not topology); (2) the real CORE2 CG takes **127 iters (cold) / 130 (warm)**, NOT pi's
> ≈3 — but the count is robustly deterministic (margin ~10 orders over the ~1e-15 reassociation), N==1 count
> identical, owned `d_eta` matches to **machine precision**. The `all_gather`+`psum`-in-`while_loop`-in-
> `custom_linear_solve`-in-`shard_map` lowers on both CPU fake-devices and real NCCL. ssh_rhs fixture +
> margin report: `scripts/capture_core2_ssh_rhs.py` → `data/ssh_rhs_core2/`.

## START HERE — the plan is the source of truth; do NOT re-plan
1. **READ `docs/plans/20260607-fesom-jax-phase8-sharding.md` FIRST.** It is the full Phase-8 sub-plan;
   the **Revision Log #2→#7** records exactly what S.1–S.5 built + every decision/discovery, and the
   task checkboxes show S.1–S.5 ticked, S.6–S.10 open. Work S.6 task-by-task; tick `[x]` + add a
   Revision-Log entry as you go.
2. **Read the Phase-8 lessons** in `docs/PORTING_LESSONS.md` (sections "Phase 8 — sharding (Task S.1…S.5)")
   — the load-bearing discoveries (ownership asymmetry, the all_gather exchange, the loop-bound proof,
   node-only reductions).
3. **Memory:** `[[fesom-jax-port]]`, `[[hpc-job-file-conventions]]`, `[[porting-lessons-log]]`.

## WHAT'S BUILT (S.1–S.5 — the exchange-ready foundation; do NOT re-litigate)
- **`fesom_jax/partit.py`** (S.1) — reads the bit-identical FESOM `dist_<NP>` (`Partition`/`ComStruct`
  pytrees, all ranks in one process, 0-based shift). 🎯 **Nodes are uniquely owned; elements/edges are
  REDUNDANTLY owned at the boundary** (`Σ myDim_elem > elem2D`) — load-bearing.
- **`fesom_jax/shard_mesh.py`** (S.2 + S.2b) — `build_sharded_mesh` (gather by `myList`, remap
  connectivity, pad-to-`Lmax`, masks, the `(src_dev, src_lane)` exchange map), `export`/`load`, and
  `partition_state`/`partition_forcing_*`. Serial `npes==1` is array-equal to the dense `Mesh`/`State`.
  `nod_in_elem2D` CSR omitted (IC-only). `scripts/export_dist_mesh.py` writes the per-`NP` bundle.
- **`fesom_jax/halo.py`** (S.3) — `halo_exchange` = `all_gather` + per-lane gather (owner→halo, interior
  identity). **Sharding convention: fold the device axis into the leading dim — `[P*Lmax,…]`,
  `PartitionSpec('p')` ⇒ each device sees `[Lmax,…]` (the step body is unchanged).** Ported
  `fesom_halo_identity_test`; AD = reverse exchange.
- **`fesom_jax/halo_points.py`** (S.4) — the per-substep exchange SCHEDULE (`OCEAN_SCHEDULE` +
  `ICE_SCHEDULE`, each `Exch` tagged **post/intra**) + `FUSED_KERNELS_NEEDING_SPLIT` (the 5 kernels S.7
  must split). The §4 loop-bound rule is VERIFIED (owned entities' incident edges/elements are local ⇒
  local scatter is complete on owned; broadcast only refreshes halo).
- **`fesom_jax/reductions.py`** (S.5) — `global_sum`/`global_dot` (owned-mask sum + `psum`). ALL per-step
  reductions are node-based (the S.1 elem/edge caveat does NOT bite). `_area_mean` routed through it
  (gated `owned_mask=None` ⇒ byte-identical).

## THE INTEGRATION LADDER (S.6 → S.10) — start at S.7
- **S.6 distributed CG** (`ssh.py`). ✅ **DONE** (Revision Log #8). The static `SSHOperator` is partitioned
  by node ownership (`partition_ssh_operator`); the halo-exchange is folded INTO `ssh_matvec`/`ssh_precond`
  (`pp` before the stiffness SpMV, `rr` before the precond SpMV), CG dots → `global_dot`. The operator
  loop-bound holds **by value** (overshoot columns are exactly zero). Iteration-count determinism VERIFIED
  on the real CORE2 KPP+GM+ice config (127/130 iters, N==1 identical, margin ~10 orders over reassociation;
  no pinning needed). 9 tests pass on CPU fake-devices + real 4×A100. The `halo=None` path is byte-identical
  `v1.0`. The exchange convention/global_dot/SSHHalo are the templates S.7 reuses for the whole step.
- **S.7 wire `shard_map`** (`integrate_sharded.py` + `step.py`). Device-mesh placement (reshape the S.2
  `[P,Lmax,…]` arrays to `[P*Lmax,…]`, `PartitionSpec('p')`); **split the 5 fused kernels**
  (`FUSED_KERNELS_NEEDING_SPLIT`); interleave `halo_exchange` at the `halo_points` post-exchange points;
  thread `owned_mask`/`axis_name` into the reduction sites. **Gate behind a STATIC arg** (like
  `kpp_cfg=None`) so `None` ⇒ the `v1.0` graph untraced (byte-identical). PRIMARY gate: a few-step CORE2
  step sharded == single-device `v1.0` per-substep ~1e-12.
- **S.8 AD gate** — `jax.grad` of the sharded run == single-device gradient; masked/halo lanes finite +
  0-cotangent (the masked-NaN rule across the device axis).
- **S.9 the GATE** — (a) per-substep N==1 on 2/4×A100 (`-A ab0995_gpu -p gpu --gres=gpu:4 --nodes=1`),
  (b) gradient, (c) BONUS JAX-N ↔ C-N dump diff on `dist_2/4` (looser cross-runtime budget). Update
  `[[fesom-jax-port]]` memory at the gate.
- **S.10 docs** — tag `v1.1-multi-gpu`, move the plan to `completed/`.

## REPRODUCE / TOOLS
- **Env python (ALL python/pytest):** `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`.
- **CPU fake-device gate (the sharding gate vehicle — no GPU until S.9):**
  `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 <env-py> -m pytest
  fesom_jax/tests/test_halo.py fesom_jax/tests/test_*_sharded.py`. These SKIP cleanly at 1 device.
- **Full suite (compute node, ~22 min): `sbatch scripts/run_suite.sbatch`** — now runs THREE groups:
  ocean / ice / **SHARDING (4 fake-devices)**. "ALL GREEN" iff all three rc=0. The host-side foundation
  tests (`test_partit`/`test_shard_mesh`/`test_partition_state`) run in the ocean group (no devices
  needed); the collective tests (`test_halo`/`test_*_sharded`) in the sharding group.
- **CORE2 partitions:** `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/dist_{2,4,8,…}`. The dense mesh
  is `data/mesh_core2/`. `JAX_PLATFORMS=cpu <env-py> -m pytest fesom_jax/tests/test_partit.py` reads them.
- **C MPI port = the parallelization SoT** (read, mirror): `/home/a/a270088/port2/fesom2_port/` —
  `src/fesom_partit.c`, `src/fesom_halo.c`, `docs/MPI_PORT_REPORT.md` (the exchange table), the C-N dump
  for S.9c (per-rank `%05d` suffix). **C edits → port2 `jax-mesh-export` branch, NEVER main.**
- I (Claude) drive SLURM (acct ab0995 / ab0995_gpu).

## WORKFLOW (standing rules)
- **Tests-first per task, gate on CPU fake-devices**; every task ends green AND the single-device suite
  green (`run_suite.sbatch`). **Append a lesson per task** to `docs/PORTING_LESSONS.md`. Tick the plan
  checkboxes + add a Revision-Log entry per task. Commit per-task on `main` ONLY when the user asks.
- ⚠️ **Recurring gotchas:** the masked-NaN AD rule (padded/halo lanes must compute a FINITE value — now
  across the device axis); `npes==1`/`partition=None` must trace to the exact `v1.0` graph (static-arg
  dead branch, like `kpp_cfg=None`); the CG iteration count must NOT drift across partitions (S.6);
  float64 throughout.

Confirm you've read the committed plan + the S.1–S.6 Revision-Log entries; then start **S.7** (wire
`shard_map` around `step`/`integrate` — split the 5 fused kernels in `FUSED_KERNELS_NEEDING_SPLIT`,
interleave the `halo_points` post-exchanges at the C's exchange points, gate the whole thing behind a
static `partition`/`exch` arg (like `kpp_cfg=None`) so `None` ⇒ the byte-identical `v1.0` graph; PRIMARY
gate: a few-step assembled CORE2 step sharded == single-device `v1.0` per-substep ~1e-12). S.6 left the
exchange primitive, `global_dot`, the `SSHHalo` context, and the per-device operator partition as the
templates to reuse.
