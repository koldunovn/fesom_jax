# Next-session prompt — FESOM2 → JAX port: PHASE 8 (multi-GPU / multi-core via mesh sharding)

> **⚡ STATE (2026-06-07): the full differentiable FESOM2 CORE2 model (KPP + GM/Redi + prognostic ice)
> runs 2 yr stable on ONE A100 at dt=1800 AND now MATCHES the C-port-KPP climate to the bit-faithful
> inter-reference budget at ALL latitudes.** The high-lat sea-ice bias is RESOLVED (it was a config
> desync — `IceConfig()` default `ice_dt=500` vs ocean `dt=1800`; fix = derive `ice_dt=ice_ave_steps·dt`
> in `ice_surface_step`). Verified re-run: SST RMS **0.490→0.0107 °C** (1958) / **0.0071** (1959), polar
> bands 0.71–0.77→0.003–0.017, m_ice RMS 0.196→0.001 — inside the 0.005–0.014 °C budget everywhere.
> This single-GPU version is **tagged `v1.0-single-gpu`**. **NEXT: scale it out — multi-GPU / multi-core
> via mesh sharding (`shard_map` + `ppermute` halos), the committed Phase 8 milestone.**

## START HERE
1. **What works (don't re-litigate):** Phases 0–6 + 6B + 6C COMPLETE; the full CORE2 model
   (`step.py`/`integrate.py`, KPP+GM+ice behind static `kpp_cfg`/`gm_cfg`/`ice_cfg`) is per-substep
   dump-gated, differentiable end-to-end (checkpointed `lax.scan` + CG `custom_linear_solve`), 2-yr
   stable, and **climate-matches the C** (tag `v1.0-single-gpu`, commit on `main`). Suite 483 green.
2. **The sea-ice close-out** (for context, now CLOSED): `docs/plans/20260607-fesom-jax-seaice-climate-bias.md`
   (Revision Log #2) + `docs/PORTING_LESSONS.md` (the `ice_dt` ROOT-CAUSE section). **Lesson that matters
   for Phase 8: a from-scratch vectorized port CAN match C to ~0.01 °C climate — that bar is the Phase-8
   acceptance gate too (an N-device run must match the 1-device run).**
3. **Memory:** `[[fesom-jax-port]]` (locked decisions — note **decision (5): sharding = `shard_map` +
   `ppermute`, the committed later milestone**), `[[hpc-job-file-conventions]]`, `[[porting-lessons-log]]`.

## PHASE 8 GOAL — shard the unstructured mesh across devices, preserve fidelity + AD
The model is currently **single-device**: every field is a dense `[nod2D|elem2D|edge2D, …]` array over the
WHOLE mesh, scatters are `ops.scatter_add` (= `jax.ops.segment_sum`) over global indices, and global
reductions are plain `jnp.sum`. To run on **N GPUs (Levante: up to 4×A100/node, multi-node via
`jax.distributed`) or many CPU cores**, partition the mesh and add halo exchange — mirroring the C MPI
port's domain decomposition. **This is the parallelization the whole port was designed for (locked
decision 5); the physics does NOT change.**

### The pieces (suggested order — each independently gateable)
1. **Mesh partition + halo infrastructure (the foundation).** Decide owned-vs-halo per device for
   nodes/elements/edges + build the `ppermute` send/recv index maps. **Easiest faithful route: READ the
   existing FESOM2 partition** the C/Fortran MPI runs already use (the `dist_<NP>/` dir: `rpart.out`,
   `my_list*`, `com_*` — the C `partit` carries `myList_nod2D`, `eDim_nod2D` halo, the `com_nod2D`
   send/recv lists). That gives a *bit-identical* decomposition to the C, so an N-device JAX run can be
   diffed against the C N-rank run. Alt: partition in Python (`pymetis`). Extend `Mesh` with per-device
   owned/halo counts + the exchange index maps (today `mesh.py` has NO halo/partition fields — greenfield).
2. **Halo-aware scatter — the core primitive.** `ops.scatter_add` must become "accumulate locally to
   owned+halo, then `ppermute`-exchange halo contributions to owners and sum, then broadcast owner values
   back to halos" — exactly the C `fesom_exchange_nod2D` after each scatter. Get this ONE primitive right
   and most kernels (TG-RHS, EVP stress2rhs, momentum, ssh_rhs, w, FCT) follow, since they all route
   through it. The C annotates every write-loop's halo bound (`feedback_write_loops_halo.md`) — match it.
3. **Distributed global reductions → `jax.lax.psum`.** The CG solver (`ssh.py:237-258`: `Σb²`, `Σr·z`,
   `Σp·Ap`, `Σr²` dot-products + norms), the area-mean balances (`sss_runoff._area_mean`,
   `ice_coupling.integrate_nod_2D` for virtual/relax salt) all sum over ALL nodes → need `psum` across the
   device mesh. ⚠️ The CG **early-stop iteration count** (the dump-matching loose `soltol=1e-5`, 3 iters on
   pi) must stay deterministic + identical across partitions — the residual norm is now a `psum`; verify
   the count doesn't drift (the C's huge residual margin makes it robust, but check).
4. **Distributed CG matvec.** The SSH stiffness matvec is a `segment_sum` over edges (`ssh.py:168-215`);
   under sharding it's a local matvec + a halo exchange of the partial sums each iteration. The static
   operator (host scipy COO→CSR) must be partitioned consistently with the node ownership.
5. **AD through the collectives.** `ppermute` transpose = the reverse permutation; `psum` transpose =
   `psum` — so reverse-mode SHOULD carry through, but **gate it**: a sharded `d(loss)/d(param)` must match
   the single-device gradient (the masked-NaN + FD↔AD discipline from Phases 3/5/6, now across devices).

### THE GATE (Phase 8 acceptance)
**An N-device run == the 1-device run to the scatter-reassociation budget (~1e-12/step), and the climate
matches** (same ~0.01 °C bar the ice fix just hit). Two-level: (a) **per-step** N-device vs 1-device on a
few steps (tight, ~1e-12 — pure halo/reduction correctness, no physics change); (b) **climate** N-device
2-yr vs the C N-rank run (the existing `kpp_bias_map.py` / inter-ref budget). If `dist_<NP>` matches the
C decomposition, you also get a **direct JAX-N-rank ↔ C-N-rank** per-substep dump diff (the strongest gate).

## REPRODUCE / TOOLS (all committed; tag `v1.0-single-gpu`)
- 1-GPU climate (the reference to beat): `sbatch scripts/core2_kpp_climate_gpu.sh [years] [dt] [outdir]`
  (A100 ~70 min/2 yr) → `data/kpp_climate_2yr_fix/<var>.fesom.<yr>.monthly.nc`.
- Bias map / compare: `HDF5_USE_FILE_LOCKING=FALSE <env-py> scripts/kpp_bias_map.py --year 1958
  --jax-dir <dir>` (lat-band + hotspot + depth stats vs the C-port-KPP ref). ushow live files with
  `HDF5_USE_FILE_LOCKING=FALSE`.
- Multi-GPU on Levante: `-A ab0995_gpu -p gpu --gres=gpu:4 --nodes=1` (4×A100-80GB/node); multi-node via
  `jax.distributed.initialize`. Multi-core: CPU process-per-core or `jax.devices('cpu')` with
  `XLA_FLAGS=--xla_force_host_platform_device_count=N`.

## KEY PATHS / COMPUTE
- JAX repo `/home/a/a270088/port_jax` (git `main`, tag `v1.0-single-gpu`). **Env python (ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python` → `JAX_PLATFORMS=cpu … -m pytest`.
- **C MPI port = the parallelization SoT:** `/home/a/a270088/port2/fesom2_port/src/` — `fesom_partit.c`
  (the partition/halo struct), `fesom_halo.c` (`fesom_exchange_nod2D`), and every kernel's halo-bound
  annotations. **C edits → port2 `jax-mesh-export`, NEVER main.** Kokkos port
  `/home/a/a270088/port_kokkos/` = the GPU-fidelity reference.
- Mesh: the single-GPU run loads `data/mesh_core2/*.npy` (the C `jax-mesh-export` bundle — single-rank,
  dense arrays, **NO partition/halo info**). FESOM `dist_<NP>/` partitions (`rpart`/`my_list`/`com_*`,
  the format to read) exist on-system for OTHER meshes under `/work/ab0995/a270088/meshes/`
  (FORCA05/tropotest_icon/nares4) — so the partitioner + format are available; the CORE2 `dist_<NP>`
  must be obtained from the original FESOM CORE2 mesh dir (the Fortran runs') or generated (FESOM
  partitioner / `pymetis`) and then EXPORTED into the npy bundle (extend the mesh-export step). Large
  files → `/work`.
- I (Claude) drive SLURM (acct ab0995 / ab0995_gpu).

## WORKFLOW
- This is a big phase — **make a Phase-8 sub-plan** (`/planning:make`) before coding: partition source,
  the halo-scatter primitive design, the reduction/CG distribution, the gate ladder. Tick/log per task.
- **STANDING RULE: append a lesson per task** to `docs/PORTING_LESSONS.md`. Commit per-task on `main`
  when asked. Update `[[fesom-jax-port]]` memory at the phase gate.
- Phase 7a (differentiable param tuning, `docs/plans/20260607-fesom-jax-paramtune.md`) is ALSO ready
  (the forward climate now matches C) — but the user's chosen next step is **Phase 8 sharding**. 7a can
  follow, and benefits from sharding (bigger ensembles / longer adjoint windows).

Confirm you've absorbed this; then start the **Phase 8 sub-plan** — partition source + the halo-aware
`scatter_add` primitive first (it's the foundation; the rest is reductions + the gate).
