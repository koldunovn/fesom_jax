# FESOM2 → JAX Port — Phase 8: multi-GPU / multi-core via mesh sharding (sub-plan)

**Parent plan:** `docs/plans/20260605-fesom-jax-port.md` (roadmap — sharding = locked decision 5).
**Predecessor:** the full differentiable CORE2 model (KPP + GM/Redi + prognostic ice), tag
`v1.0-single-gpu` (commit `67303d4`) — runs 2 yr stable on ONE A100 and **matches the C-port-KPP
climate to ~0.01 °C** at all latitudes. That single-device run is **the reference to beat**.
**Created:** 2026-06-07. **Status:** 🚧 **IN PROGRESS** — foundation first (S.1–S.4), then reductions
+ CG + AD + the gate ladder.

---

## 0. Scope (READ FIRST)

### What Phase 8 is, and is NOT

The model is **single-device today**: every field is a dense `[nod2D|elem2D|edge2D, …]` array over the
WHOLE mesh, scatters are `jax.ops.segment_sum` over global indices, global reductions are plain
`jnp.sum`. Phase 8 **partitions the unstructured mesh across N devices** (one device ↔ one MPI rank
equivalent) and adds **halo exchange + distributed reductions**, mirroring the C MPI port's domain
decomposition. **The physics does NOT change** — this is purely the parallelization the whole port was
built for (Locked decision 5).

**Deliverable scope (user-confirmed 2026-06-07): correctness on 2–4 devices.** The committed GATE is a
few-step **N-device == 1-device** match (per-substep ~1e-12) on CPU fake-devices then 2–4×A100, plus the
gradient gate. **The full 2-yr multi-GPU climate run vs the C is an explicit FOLLOW-UP** (Post-Completion),
*not* this phase's gate — at N ranks the long climate diverges chaotically from reduction-order
non-determinism (the C port sees the same thing: *"Allreduce summation-order non-determinism amplified by
chaos"*, `port2/.../docs/MPI_PORT_REPORT.md`), so per-substep correctness, not 2-yr climate, is the right
acceptance bar for this phase.

### The model to mirror (user-confirmed 2026-06-07): **mirror the C MPI port EXACTLY**

Both reference ports (the C MPI port `port2/fesom2_port/` and the Kokkos port `port_kokkos/`) use the
**same** decomposition model, and we replicate it 1:1:

* **Broadcast-only halo exchange** (owner value → halo copies). There is **no additive exchange**
  anywhere in either reference (`fesom_halo.c` has only `fesom_halo_exchange`; verified). Owned entities
  get their COMPLETE scatter-sum because each rank **computes redundantly over a wide-enough halo**
  (`eDim`/`eXDim` elements) so every element/edge incident to an owned node is local; the broadcast then
  refreshes the (incomplete) halo copies for the next kernel.
* **Per-kernel halo loop bounds** — a producing loop must run over `myDim+eDim(+eXDim)` for any field a
  downstream kernel reads at the halo (`port2/.../docs/PORTING_LESSONS.md §4`, the authoritative rule).
* **Reductions** = sum over OWNED entities → `MPI_Allreduce(SUM)` (→ `jax.lax.psum`).

**Why mirror-C (vs a from-scratch additive model):** the C decomposition is read **bit-identically**
from the FESOM `dist_<NP>` files, so an N-rank JAX run can be diffed **per-substep against the C N-rank
dump** (the C already writes per-rank dumps, `fesom_dump.c:107` `%05d` mype suffix) — reusing the entire
existing dump-gate at N ranks. This is the strongest fidelity check available and it is only unlocked by
matching the C exactly.

### Partition SOURCE — read the bit-identical FESOM `dist_<NP>` (do NOT generate)

The canonical CORE2 mesh ships precomputed partitions:
`/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/dist_{2,4,8,16,32,64,128,…,864}`. `dist_2/4/8` line up
exactly with 2/4/8 CPU-fake-devices and the Levante 4×A100 node. **Read them** (port the
`fesom_partit.c` ASCII reader to numpy) — `pymetis` is not installed and generating a partition would
forfeit the C cross-check. `dist_8` partition correctness is already C-validated
(`MPI_PORT_REPORT.md:67`). Real `dist_` bundles for *other* meshes (e.g. `…/meshes/nares4/dist_768/`)
exist as extra format fixtures.

### JAX mechanics (locked): single-process `shard_map` over a 1-D device mesh

We scale **within one process** from N CPU fake-devices (`XLA_FLAGS=--xla_force_host_platform_device_count=N`)
to a single 4×A100 node. Each device holds its rank's **padded** local arrays
(`Lmax = max_d(myDim_d+eDim_d)`, pad + mask — the established masked-lane discipline) and the existing
`step`/`integrate` run **unchanged** under `jax.experimental.shard_map`, with halo exchanges interleaved
at the C's exchange points and `jnp.sum`→`psum`. Multi-NODE (`jax.distributed.initialize`, >4 devices)
is a Post-Completion follow-up; single-process keeps the **N-device-vs-1-device gate trivial** (same
process, same code path) — the whole point of "correctness first".

---

## Decisions (locked)

1. **Mirror the C MPI port exactly** (user-confirmed): broadcast-only halo + per-kernel
   `myDim/+eDim/+eXDim` redundant compute + `psum` reductions. The algorithmic SoT is `fesom_partit.c`,
   `fesom_halo.c`, the `MPI_PORT_REPORT.md` exchange table, and `PORTING_LESSONS.md §4`.
2. **Partition = read FESOM `dist_<NP>`** (bit-identical to C), not generated. Port the
   `fesom_partit.c` reader to numpy; export an extended per-`NP` npy mesh bundle.
3. **Single-process `shard_map` over a 1-D device mesh**, pad-to-`Lmax` + mask for even JAX sharding.
   CPU fake-devices → 2–4×A100. Multi-node deferred.
4. **GATE = 2–4 device per-substep correctness + gradient** (not the 2-yr climate). N-device == 1-device
   JAX to the scatter/reduction reassociation budget (~1e-12/step); bonus per-substep JAX-N ↔ C-N dump
   diff on `dist_2/dist_4`.
5. **No physics change, no new `State`/kernel math** — only: a `Partition`/sharded-`Mesh`
   representation, a `halo.py` exchange primitive, `jnp.sum`→`psum` at the reduction sites, and the
   `shard_map` wrapper. `kpp_cfg`/`gm_cfg`/`ice_cfg` and every existing gate stay green single-device.
6. **AD through the collectives is gated, not assumed**: `ppermute`/`all_to_all` transpose = the reverse
   exchange, `psum` transpose = `psum`; a sharded gradient must equal the single-device gradient.

---

## C reference (file:line)

### `dist_<NP>` file format — `fesom_partit.c`

| file | layout | C reader |
|------|--------|----------|
| `rpart.out` | line1 `npes`; line2 `npes` per-rank node counts → prefix-sum `part[]` | `read_rpart` `:68-93` |
| `my_list<R>.out` | `rank`; `myDim_nod2D`,`eDim_nod2D`,`myList_nod2D[myDim+eDim]`; `myDim_elem2D`,`eDim_elem2D`,`eXDim_elem2D`,`myList_elem2D[…]`; `myDim_edge2D`,`eDim_edge2D`,`myList_edge2D[…]` | `read_my_list` `:111-145` |
| `com_info<R>.out` | `rank`; 3 com blocks (nod2D, elem2D, **elem2D_full**), each `rPEnum,rPE[],rptr[],rlist[eDim]; sPEnum,sPE[],sptr[],slist[]` | `read_com_info`/`read_com_block` `:154-208` |

`com_struct` semantics (`fesom_partit.h:42-52`): **rlist** = LOCAL halo indices `(myDim..myDim+eDim]`
this rank **receives** into; **slist** = LOCAL interior indices `[1..myDim]` this rank **sends** to
neighbours' halos. Indices are **1-based** on disk (subtract 1). `myList_*` are global IDs. The serial
`npes==1` case = identity (`fesom_partit.c:280-308`).

### Broadcast exchange — `fesom_halo.c:135-201`

Pack `field[slist]` per sender-PE → Isend; Irecv → unpack into `field[rlist]` per receiver-PE. Pure
owner→halo overwrite (`memcpy`, no `+=`). The identity gate (`fesom_halo_identity_test:212-284`):
set `f[i]=gid`, exchange, assert `f[halo]==myList[halo]` (+ corruption-recovery) — **port this test**.

### Per-substep exchange points + reductions — `MPI_PORT_REPORT.md` ("Halo exchanges per timestep")

The fields exchanged, in execution order (the spec S.7 wires in): after `pressure_bv`
(`density,hpressure,bvfreq,sw_alpha,sw_beta` nod3D), after `compute_vel_nodes` (`uvnode`), after
`pp/kpp + mo_convect` (`Kv` nod3D, `Av` elem3D), after `compute_vel_rhs`/`visc_filt`/`impl_vert_visc`
(`uv_rhs` elem3D-vec, `u_b/v_b` elem3D, `u_c/v_c` nod3D), after `compute_ssh_rhs_linfs` (`ssh_rhs`),
**inside CG** (`pp` each iter before SpMV, `rr` after residual, `X` final), after CG (`d_eta`), after
`update_vel` (`uv`), after `compute_hbar` (`ssh_rhs_old,hbar`), after ALE `vert_vel` (`w`), after each
tracer FCT (`T,S`; inside: `fct_LO`, `fct_plus/minus`), after `impl_vert_diff` (`T,S`), after
`commit_thickness` (`hnode,helem`). **Per-timestep `MPI_Allreduce(SUM)`**: CG `pp·App`, `rr·zz`,
`rr·rr` per iter; `integrate_nod_2D` for virtual-salt / relax-salt / water-flux balances. One-time
startup exchanges: `elem_area`(elem2D+full), `elem_cos`,`metric_factor`,`coriolis`,`elem_center_x/y`;
`Allreduce` `ocean_area`.

### Halo-bound rule — `PORTING_LESSONS.md §4`

> for every array, ask "who reads this, and do they loop into the halo?" If yes, the producing loop must
> cover `myDim+eDim` and the array must be allocated `myDim+eDim(+eXDim)`.

---

## JAX surface to change

* **`ssh.py`** — CG dots `Σb²,Σr·z,Σp·Ap,Σr²` (`:237-258`) → `psum`; matvec `ssh_matvec`/`ssh_precond`
  (`:168-177`, `compute_ssh_rhs` scatter `:215`) → local segment_sum + `pp`/`rr` exchange per iter; the
  static operator (`build_ssh_operator`, host scipy COO→CSR `:80-163`) partitioned by node ownership.
* **Reductions**: `sss_runoff._area_mean`, `ice_coupling.integrate_nod_2D` → owned-sum + `psum`.
* **Scatters**: 35 `segment_sum`/`scatter_add` sites across 14 modules route through `ops.scatter_add`
  (+ direct `jax.ops.segment_sum` in `ssh.py`); under sharding they become **local** segment_sum into
  `[Lmax]` arrays — correctness comes from the loop bounds + the post-kernel broadcast, not the scatter.
* **`mesh.py`** — gains the partition/halo fields (today it has **none** — greenfield).
* **`step.py`/`integrate.py`** — wrapped by a `shard_map` entry point; bodies unchanged.

---

## Development Approach

* **Tests-first per task, gated on CPU fake-devices** (`XLA_FLAGS=--xla_force_host_platform_device_count=2`)
  — cheap, no GPU, runs on the compute node. Promote to 2–4×A100 only at S.9.
* **Every task ends green**: new tests pass AND the single-device suite (483) stays green (sharding is
  additive; the `npes==1` path must stay bit-identical to `v1.0-single-gpu`).
* **STANDING RULE: append a lesson per task** to `docs/PORTING_LESSONS.md`. Commit per-task on `main`
  when asked. Update `[[fesom-jax-port]]` memory at the gate.
* Env python (ALL python/pytest): `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`.
  CPU gates: `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=N … -m pytest`.

## Progress Tracking
`[x]` done · ➕ newly discovered · ⚠️ blocker. Keep this file in sync; log each task in the Revision Log.

---

## Implementation Steps

### S.1: Partition reader (`dist_<NP>` → `Partition` pytree)

**Files:**
- Create: `fesom_jax/partit.py`
- Create: `fesom_jax/tests/test_partit.py`

- [x] `Partition` frozen dataclass (registered pytree): per-device `myDim/eDim/eXDim` for nod/elem/edge,
      `myList_*` (global IDs, 0-based after the −1 shift), and the three `com_struct`s
      (`rPE,rptr,rlist,sPE,sptr,slist`) for nod2D/elem2D/elem2D_full. Static counts as metadata.
      ➕ `ComStruct` registered too (nested pytree); ragged per-rank data held as `npes`-long tuples.
- [x] `read_partition(mesh_dir, npes)` — numpy port of `read_rpart`/`read_my_list`/`read_com_info`
      (`fesom_partit.c:68-208`); store global IDs 0-based, com indices 0-based, keep the 3 com blocks.
      ➕ `rPE`/`sPE` kept 0-based (no shift — they are MPI ranks, per `fesom_halo.c:166,182`).
      ➕ global counts = `max(gid)+1` (NOT `Σ myDim`: elem/edge are redundantly owned — see Rev #2).
- [x] `synth_serial(nod2D,elem2D,edge2D)` — the `npes==1` identity partit (`:280-308`) so the same code
      path covers single-device (zero halo, no neighbours).
- [x] write tests: read CORE2 `dist_2` + `dist_4`; assert `Σ_d myDim_nod2D == nod2D`,
      `rlist` lengths == `eDim`, `slist` length == `sptr[-1]`, `myList` monotone-consistent, rank header
      matches; format-fixture read of `…/meshes/nares4/dist_768/com_info00000.out`.
      ➕ interior-only monotone (halo is per-segment-sorted for ≥3 ranks); + cross-rank owner consistency.
- [x] write error tests: missing `dist_<NP>` raises; `npes` mismatch in `rpart.out` raises.
- [x] run tests — must pass before S.2. Append lesson. **20 passed; lesson appended.**

### S.2: Sharded-mesh build + export (local remap, pad-to-`Lmax`, exchange index maps)

**Files:**
- Create: `fesom_jax/shard_mesh.py`
- Modify: `fesom_jax/mesh.py` (optional halo/partition fields on `Mesh`, or a parallel `ShardedMesh`)
- Create: `scripts/tools/export_dist_mesh.py`
- Create: `fesom_jax/tests/test_shard_mesh.py`

- [x] `build_sharded_mesh(mesh, partition)` → per-device local arrays: gather each global `Mesh` field
      by `myList_*`; **remap connectivity** `elem_nodes`/`edges`/`edge_tri`/`edge_up_dn_tri` global→local
      (a global→local lookup per device; `-1` sentinels preserved; outer-halo unmappable vertices → `-1`).
      ➕ `nod_in_elem2D` (CSR) OMITTED — IC-only (`phc_ic.py`), host-built then partitioned in S.2b.
      ➕ parallel `ShardedMesh` dataclass (mesh.py untouched ⇒ single-device byte-identical).
- [x] **pad-to-`Lmax`** each entity kind to a common per-kind max over devices; build `owned_mask`
      (`[P,Lmax]` true on `[0:myDim_d]`) and `valid_mask` (true on `[0:myDim+eDim(+eXDim)]`) — padded
      lanes carry a safe value (float→`1.0` for denominators / `-1` index) and are masked.
- [x] ⚠️ **gather-on-sentinel safety** (review #3): connectivity carries `-1`; the build clamps `-1`→0
      before the value-remap gather, then restores `-1`. **Proven safe**: owned elements have 0 `-1`
      vertices on dist_4 (`elem_nodes[:myDim]` all local) ⇒ no owned output depends on a sentinel gather;
      only halo/eXDim lanes carry `-1`. Kernel-side clamp+mask deferred to S.7 (when wiring the kernels).
- [x] build the **exchange index maps** for the JAX collective: `(src_dev, src_lane)` `[P,Lmax]` per kind
      (the `all_gather` form — interior identity, halo←owner's interior). The `slist`/`rlist`/segment form
      stays in the `Partition` `ComStruct` for the `ragged_all_to_all` perf follow-up.
- [x] `scripts/tools/export_dist_mesh.py` → write the padded local arrays + masks + exchange maps as `.npy` +
      `meta.txt`; default output on `/work`. `load_sharded_mesh` reloads it.
- [x] write tests: every local `elem_nodes` index `< Lmax_nod` or `==-1`; halo lane gids equal their
      owner's; `export→reload` round-trips; the **serial `NP=1`** sharded mesh is array-equal to the dense
      `Mesh` (the no-op invariant). ➕ connectivity-remap invertibility; interior-identity exchange.
- [x] run tests — must pass before S.2b. Append lesson. **11 passed; lesson appended.**

### S.2b: Partition `State` + per-step forcing + IC builders (load-bearing — review #2/#6)

**Files:**
- Modify: `fesom_jax/shard_mesh.py` (add `partition_state` / `partition_forcing`)
- Create: `fesom_jax/tests/test_partition_state.py`

> Promoted out of S.4/S.7 — S.4's scatter gate and S.6's CG test cannot produce sharded inputs without
> this, so it must land **before** S.3's consumers. A real, testable deliverable, not a clause.

- [x] `partition_state(global_state, partition)` — gather each `State` field by `myList_{nod,elem}`
      (leading dim **detected**, not hardcoded), pad to `Lmax` (shared `local_sizes` with the mesh),
      return a per-device padded `State` (padded lanes float→1.0, masked-safe).
- [x] `partition_forcing(...)` — `partition_forcing_static` (5 node fields + replicated scalar
      `ocean_area`) + `partition_step_forcing` (10 fields; handles single `[nod2D]` AND scanned
      `[n_steps,nod2D]` → `[P,n_steps,Lmax_nod]`, node axis detected).
- [x] cover the **IC builders**: `State.rest` tested (node+elem fields populated); PHC IC / ice
      cold-start follow the **same host-build → `partition_state`** pattern (sidesteps the C's PHC
      `extrap_nod3D` per-sweep exchange — noted in the lesson).
- [x] write tests: serial-`NP=1` `partition_state`/`partition_forcing` == the dense `State`/forcing
      (the no-op invariant); 2/4-device gather places each owned field on the right device; padded lanes
      coincide with the mesh's masked lanes ⇒ inert (+ finite, masked-NaN rule).
- [x] run tests — must pass before S.3. Append lesson. **8 passed; lesson appended.**

### S.3: Broadcast halo-exchange primitive (`halo.py`) + identity gate

**Files:**
- Create: `fesom_jax/halo.py`
- Create: `fesom_jax/tests/test_halo.py`

- [x] `halo_exchange(field, src_dev, src_lane, axis_name)` under `shard_map`: `all_gather` then per-lane
      gather `g[src_dev, src_lane]` (owner→halo overwrite, interior identity). Handles `[Lmax]`,
      `[Lmax,nl]`, `[Lmax,nl,2]` (the fancy index hits the leading two axes; trailing ride along). **ONE
      primitive** (`all_gather`); `ragged_all_to_all` = perf follow-up. ➕ sharding convention =
      fold device axis into leading dim (`[P*Lmax,…]`, `PartitionSpec('p')`) so the body sees `[Lmax,…]`.
- [x] **identity gate** (port `fesom_halo_identity_test`): set `f[owned]=global_id`, halo=sentinel,
      exchange on CPU fake-devices, assert every halo lane carries its owner's gid; corruption-recovery
      (clobber one halo, re-exchange, restored).
- [x] cover all three kinds (nod2D, elem2D, elem2D_full) × {2,4} devices + a multi-level field.
- [x] write AD test: `halo_exchange` is linear; its vjp = the **reverse** exchange — linearity check +
      FD grad-check on interior AND halo lanes.
- [x] run tests — must pass before S.4. Append lesson. **9 passed (4 fake-devices); skip cleanly at
      1 device; `run_suite.sbatch` gains a SHARDING group.**

### S.4: Exchange-point map + the 2-device scatter correctness gate

**Files:**
- Create: `fesom_jax/halo_points.py` (the per-substep exchange schedule, ported from `MPI_PORT_REPORT`)
- Modify: `fesom_jax/ops.py` (document the local-scatter + loop-bound contract; no math change)
- Create: `fesom_jax/tests/test_scatter_sharded.py`

- [x] encode the **exchange schedule** as data (`halo_points.py` — `Exch(after, field, kind, placement,
      cref)`), from the `MPI_PORT_REPORT` table — the single source S.7 iterates.
- [x] include the **sea-ice exchanges** (`ICE_SCHEDULE`): enumerated the C `fesom_ice_{evp,fct,coupling,
      thermo}.c` `fesom_exchange_nod2D` calls. ⚠️ the EVP `u_ice/v_ice` exchange is **inside the
      120-subcycle `lax.scan`** — tagged `intra` + listed in `FUSED_KERNELS_NEEDING_SPLIT` (lowering
      verified for `all_gather` in S.3; the in-scan case is exercised in S.7).
- [x] audit scatter loop bounds vs `PORTING_LESSONS §4`: **verified on dist_4** every owned node's
      incident edges+elements are local, and every owned element's contributing edges are local (0
      violations) ⇒ the local scatter is complete on owned entities, NO kernel needs a special halo bound.
- [x] ⚠️ **classify post-kernel vs intra-kernel** (review #5): each `Exch` tagged `post`/`intra`;
      `FUSED_KERNELS_NEEDING_SPLIT` enumerates the 5 — `momentum.visc_filt_bidiff`, ocean +
      `ice_adv` FCT, `ssh._pcg` (CG), `ice_evp` subcycle — with the seam each split exposes.
- [x] **scatter gate**: representative edge→node (compute_ssh_rhs shape) sharded on 2/4 CPU devices ==
      single-device on owned nodes to 1e-11 (BEFORE broadcast — owned is complete); broadcast then makes
      halo match too.
- [x] edge→element scatter (biharmonic, `edge_tri` -1 boundary) gate too — owned elems (incl. redundant
      boundary) complete on every owning device.
- [x] run tests — must pass before S.5. Append lesson. **7 passed; lesson appended.**

### S.5: Distributed reductions (`jnp.sum` → `jax.lax.psum`)

**Files:**
- Modify: `fesom_jax/sss_runoff.py` (`_area_mean`), `fesom_jax/ice_coupling.py` (`integrate_nod_2D`)
- Create: `fesom_jax/reductions.py` (a `global_sum(owned_vals, owned_mask)` helper = owned-sum + `psum`)
- Create: `fesom_jax/tests/test_reductions_sharded.py`

- [x] `global_sum`/`global_dot`: sum over **owned, unpadded** lanes only (mask halo+pad → 0), then
      `jax.lax.psum` over the device axis. `axis_name=None` ⇒ plain masked sum (single-device path).
- [x] route `_area_mean` through it (gated: `owned_mask=None` ⇒ byte-identical `jnp.sum`; sharded path
      threads `owned_mask`/`axis_name` in S.7). ➕ found: `ice_oce_fluxes` routes through `_area_mean`
      (one primitive); ALL per-step reductions are node-based ⇒ the S.1 elem/edge caveat does not apply.
      `ocean_area` is the static global constant (replicated).
- [x] write tests: 2/4-device global_sum + global_dot == single-device to ~1e-12; owned-mask excludes
      halo (corrupted halo value unchanged); the `axis_name=None` helper == plain `jnp.sum` (byte-id).
- [x] run tests — must pass before S.6. Append lesson. **8 passed; sss tests stay green; lesson
      appended.**

### S.6: Distributed CG solve (`ssh.py`)

**Files:**
- Modify: `fesom_jax/ssh.py` (`ssh_matvec`, `ssh_precond`, `_pcg`, `solve_ssh` + `SSHHalo`,
  `ShardedSSHOperator`, `partition_ssh_operator`, `local_ssh_operator`)
- Create: `fesom_jax/tests/test_ssh_sharded.py`
- Create: `scripts/debug/capture_core2_ssh_rhs.py` + `.sbatch` (the realistic-rhs fixture + margin report)
- Create: `scripts/debug/phase8_ssh_sharded_gpu.sbatch` (early real-4×A100 confirmation)

- [x] partition the **static operator**: `partition_ssh_operator` filters the global operator to each
      device's (row-local ∧ col-local) entries, remaps to local node lanes, pads `nnz`. 🎯 **The SSH
      stencil EXCEEDS the node halo** (11664 owned-row cols outside the halo on dist_2) — but every such
      entry is **exactly zero** (the topological pattern keeps numeric zeros), so the local matvec is EXACT
      on owned rows; the build **asserts no NONZERO owned-row entry is dropped** (loud guard).
- [x] `ssh_matvec`/`ssh_precond` → **fold the halo-exchange INTO the matvec/precond** (exchange the input
      before each SpMV: `pp` before the stiffness SpMV, `rr` before the precond SpMV = exactly after the
      residual update), so `_pcg`'s body is unchanged; the CG dots → `global_dot` (`psum`), `n` → global.
- [x] 🔴 **iteration-count determinism — VERIFIED ROBUST** (review #1): the real CORE2 CG takes **127
      iters (cold) / 130 (warm)** (NOT pi's ≈3 — the docstring config was pi). Captured the rhs from a real
      KPP+GM+ice `step()` (dt=1800) + recorded the margin: consecutive residuals cross the `soltol` stop by
      a factor **~1.09** (tightest: last-above iterate 0.93 % over `rtol`) — ~10 orders above the ~1e-15
      `psum` reassociation ⇒ the count cannot drift. N==1 count identical (2 & 4 dev); no pinning needed.
- [x] ⚠️ **collective inside the CG `while_loop`** (review #4): `all_gather` + `psum` inside the
      `lax.while_loop` inside `custom_linear_solve` inside `shard_map` **lowers and runs** (CPU fake-dev +
      real A100). The `psum`'d residual is device-identical ⇒ the data-dependent trip count is identical on
      all devices (no deadlock). Reverse-mode is `transpose_solve`, NOT AD-through-the-`while_loop`.
- [x] keep `custom_linear_solve` (forward early-stop / transpose tight) — the transpose solve also runs
      sharded; the symmetric matvec represents the global `S` on owned lanes ⇒ the implicit-diff gradient
      is structurally intact (checked in S.8).
- [x] write tests: 2/4-device `solve_ssh` `d_eta` == single-device on owned nodes (matches to **~3e-16
      abs, 1e-15 rel** — machine precision, far below 1e-12); iteration count equal (127/130); zero-rhs
      short-circuit holds; serial `NP=1` sharded == dense (no-op invariant); operator loop-bound guard.
- [x] run tests — must pass before S.7. **9 passed (4 CPU fake-devices, ~54 s); 43 single-device ssh tests
      green (byte-identical `v1.0`). Lesson appended.**

### S.7: Wire `shard_map` around `step`/`integrate`

**Files:**
- Create: `fesom_jax/integrate_sharded.py` (the `shard_map` entry point + device-mesh placement)
- Modify: `fesom_jax/step.py` (insert the `halo_points` exchanges at the C's exchange points)
- Create: `fesom_jax/tests/test_step_sharded.py`

- [x] device-mesh placement: `integrate_sharded.py` folds the sharded `Mesh`/`SSHOperator` (S.2/S.6) +
      `State` (S.2b) to `[P*Lmax]` `PartitionSpec('p')` and reconstructs the per-device LOCAL `Mesh`/`State`/
      `SSHOperator` (LOCAL `Lmax` static sizes; CSR dummy) inside `shard_map`, running the UNMODIFIED `step`.
      🎯 needs `check_vma=False` (the tridiagonal/FCT `scan`s have constant initial carries). **npes==1
      whole step == dense byte-identically; npes==2 lowers + 58% deep-interior owned nodes match (no
      exchanges) ⇒ the local kernels are correct on real shards.**
- [x] **split the fused kernels** (OCEAN, via an `exch` arg, identity-when-None ⇒ no-op single-device):
      `momentum.visc_filt_bidiff` (exch `Uc/Vc` between its 2 scatter stages); `momentum_adv_scalar`
      (exch `un_u/un_v` before the vertex→element gather — ➕ a scatter-result site the Kokkos `SYNC_MAP`
      flagged that the `MPI_PORT_REPORT` folds into substep 4); `tracer_adv.advect_one_fct`/`zalesak_limit`
      (exch `fct_LO`, ➕ `tr_xy`, `fct_plus/minus`). **[part 3 ✅]** GM/Redi (`fer_gamma` INTRA + `fer_uv`/
      `slope_tapered`/`Ki`/`fer_w`; `tr_xy/tr_z` auto-complete in JAX) + KPP (`smooth_blmc` per-sweep refresh +
      `viscA` before the `Av` gather) DONE & gated; ice EVP/FCT splits CODED (gating).
- [x] interleave `halo_exchange` at each `OCEAN_SCHEDULE` exchange point inside `step` via one
      `_exch(field,kind)` closure. **Gated by `halo_ctx=None`** (a pytree-or-None arg — the treedef makes
      `None` a trace-time dead branch, the `kpp_cfg=None` discipline) ⇒ byte-identical `v1.0` (the 483-test
      single-device suite stays GREEN). 🎯 **JAX needs FEWER exchanges than the C**: per-node intermediates
      are auto-complete on the halo (computed over the full local extent), so only SCATTER results need an
      exchange (the C's pre-smooth `bvfreq` exchange is unnecessary).
- [x] `run_step_sharded` (`integrate_sharded.py`) wraps the body under one `shard_map` (`check_vma=False`).
      **[part 3 ✅]** `run_steps_sharded` = step-1-eager + `lax.scan` of the rest under ONE `shard_map`
      (the collective-in-scan LOWERS, even with `jax.checkpoint`); a FREE-running multi-step compare
      decorrelates chaotically (Decision 4 at 2 steps) so the per-step gate is TEACHER-FORCED.
- [x] **PRIMARY per-substep gate** (CPU fake-devices, N=2): the assembled OCEAN step sharded == single-device
      on OWNED to a **field-appropriate** budget — momentum/SSH/ALE/EOS to <1e-7 (the wiring proof), FCT
      tracers + cancelling SSH divergences to the **climate-close upwind-flip floor** (~1e-3 on a sharp test
      bump; NOT a missing exchange — all FCT inputs match to 1e-9, `S` matches when constant, owned==halo).
      **[part 3 ✅ COMPLETE]** GM gate: clean fields machine-precision + per-kernel `gm_diagnostics` bit-exact
      (0.0) on owned ⇒ T/S spread is the FCT flip floor. KPP gate: `Kv`/`Av` machine-precision. Ice gate: ALL
      ice prognostic fields bit-exact (0.0) on owned, `uv` machine-precision (after the `a_ice`-before-the-
      `stress_surf`-gather fix). **PRIMARY assembled gate (KPP+GM+ice+forcing): `uv` 1.9e-16, all clean fields
      machine-precision, T/S the climate-close floor — PASSED.** The few-step per-substep compare is
      teacher-forced (a State-field N-vs-1 compare IS per-substep — every substep output is a State field;
      `io_dump` is reserved for the S.9c cross-runtime C-N diff).
- [x] write tests: `test_step_sharded.py` (reconstruction + npes==1 no-op byte-identity + npes==2 owned
      field-appropriate gate). **[part 3 ✅]** + GM (`gm_diagnostics` per-kernel + full-step) + KPP (forced
      npes==1 byte-id + npes==2 owned) + reductions (`sss_runoff_fluxes`) + ice (serial + owned) + multistep
      (scan-lowers + teacher-forced) + the PRIMARY assembled KPP+GM+ice gate.
- [x] run tests — **3 passed (CPU fake-devices); single-device suite ALL GREEN (475+47+36)**. **[part 3 ✅]**
      GM 2 passed + reductions 9 passed + KPP 2 passed (`Kv`/`Av` ~1e-14) + ice 2 passed (npes==1 byte-id +
      npes==2 `uv` 1.8e-16) + 18 single-device ice + teacher-forced multistep + the PRIMARY assembled gate
      (`uv` 1.9e-16). **S.7 (the `shard_map` wiring) is COMPLETE.**

### S.8: AD through the collectives (gradient gate)

**Files:**
- Create: `fesom_jax/tests/test_gradient_sharded.py` ✅
- Create: `scripts/debug/test_grad_ocean.sbatch` + `scripts/debug/test_grad_forced.sbatch` ✅ (the focused-gate
  sbatch convention from S.7p3, in place of a single `phase8_grad_gate.py`)

- [x] `jax.grad` of a scalar loss (owned-masked mean SST) of the sharded run wrt `params` (`k_ver`/`a_ver`/
      `k_gm`/`redi_kmax`, closed over the `shard_map` ⇒ their cotangent is `psum`'d — Decision 6 verified:
      `d/d(k_ver)` matches single-device to **3.75e-8**) and wrt initial `state` (`T0`). The CG
      `custom_linear_solve` transpose backward runs sharded (isolated probe clean); reverse-mode carries
      through. ➕ **8 AD bug-fixes the sharded BACKWARD exposed** (the dense XLA folds the `0·inf`; `shard_map`
      keeps it): 7 masked-NaN guards (pp `dz_inv`, momentum `dZ_up/dZ_dn`, ocean+ice FCT `segment_max/min`
      `±inf`, kpp `bldepth`+`blmix`×2) + `jax.jit`-around-the-`shard_map` for the `jax.checkpoint`'d
      multi-step scan grad. All forward-byte-identical (211 single-device tests green, incl. the v1.0 dump).
- [x] **masked-NaN across devices**: `d/d(T0)` FINITE everywhere (halo/pad/below-bottom), exactly **0**
      cotangent on dry/pad lanes, nonzero on owned-wet — OCEAN and FORCED (the ice-EVP backward).
- [x] **gradient gate**: sharded `d(loss)/d(param)` == single-device, **field-appropriate** (the gradient
      analog of Decision 4): CLEAN paths (k_ver→diffusion) machine-floor; FCT-advection paths (a_ver/k_gm/T0)
      the upwind-flip floor. T0-field reconstruction `Bᵀ(g_p)` (the `all_gather` transpose = scatter-add) ==
      dense to **max 7e-8** (OCEAN) / **3.6e-5** (FORCED). FD spot-check on the well-conditioned `k_ver`
      (plateau **4.2e-5** < 1e-4) — k_ver chosen over the plan's `k_gm` (k_gm → FCT-advection is noisy /
      needs GM-on; the FORCED gate covers `k_gm` finite + field-appropriate).
- [x] write tests: 2-device grad == 1-device grad (param + T0-reconstruction); masked-lane cotangent == 0;
      no NaN; the multi-step scan-backward + the FORCED assembled (KPP+GM+ice, the **ice-EVP 120-subcycle
      checkpointed-scan backward**) all finite. `test_gradient_sharded.py` (5 tests).
- [x] run tests — **PASS** (OCEAN param/fd/ic/multistep + FORCED assembled, CPU fake-devices); single-device
      211 green (byte-identical v1.0). Lessons appended. **S.8 COMPLETE.**

### S.9: GATE — 2–4 device correctness (CPU + A100) + bonus C-N-rank diff

**Files:**
- Create: `scripts/phase8_sharded_gate_gpu.sh`, `scripts/phase8_sharded_gate_run.py`
- Create: `scripts/phase8_cn_dump_diff.py` (JAX-N-rank ↔ C-N-rank per-substep compare)

- [x] **(a) per-substep N==1** on real devices: full assembled CORE2 step (KPP+GM+ice) on 2×A100
      (`scripts/debug/phase8_s9_gpu.sbatch`, job 25430592) == single-device on every PROGNOSTIC field (ocean
      dynamics clean 1e-9…1e-18; FCT tracers + prognostic ice climate-close 1e-2…1e-9). GPU findings:
      byte-id is a CPU property (serial-collapse worst 7.66e-9 → platform-aware `_BYTE_ID_ATOL`); EVP stress
      σ rides the VP yield kink (O(0.5) on a non-prognostic diagnostic, driven u_ice/v_ice correct to 1e-7)
      → σ EXCLUDED from the gate (`_DIAG_FIELDS`). npes=4 + the C-N diff deferred (Phase 8b).
- [x] **(b) gradient** (S.8) on GPU: OCEAN param grad PASSED on real A100 (`jax.grad`-thru-`shard_map` over
      NCCL, d/d(k_ver) rel 3.75e-8). FORCED grad (EVP-scan backward) OOM'd (memory-bound, not correctness) →
      deferred (needs EVP-scan checkpointing / fewer subcycles to fit GPU memory).
- [ ] **(c) BONUS — JAX-N ↔ C-N** [DEFERRED to Phase 8b]: build/run the C MPI port at 2/4 ranks on
      `dist_2`/`dist_4` with the per-rank dump, diff each substep against the JAX N-rank run. ⚠️ tolerance =
      the **cross-runtime reassociation budget** (review #9): XLA `psum` tree-order ≠ MPI `Allreduce`
      tree-order even at equal rank count, so this is **looser than the N-vs-1 JAX gate** (which shares the
      XLA runtime) — frame it as a strong cross-check, not a 1e-12 equality. C edits, if any, on the port2
      `jax-mesh-export` branch — **never `main`**.
- [x] verify the single-device suite still green and `npes==1` is byte-identical to `v1.0` (re-calibration
      only loosens the GPU branch + drops a diagnostic; CPU gate unchanged).
- [x] document results in the Revision Log (#14); update `[[fesom-jax-port]]` memory at the gate.

### S.10: [Final] Docs + handoff

- [x] update `docs/NEXT_SESSION_PROMPT.md` (Phase 8 done → Phase 8b scaling fork).
- [x] append the final Phase-8 lessons to `docs/PORTING_LESSONS.md` (the S.9 lesson); tag `v1.1-multi-gpu`.
- [x] move this plan to `docs/plans/completed/`.

---

## Technical Details

* **Padded sharding**: global sharded array shape `[P, Lmax, …]`, sharded `PartitionSpec('p', None, …)`;
  inside `shard_map` each device sees `[Lmax, …]`. `Lmax` per entity kind. Masks: `owned_mask` (reductions
  / loss), `valid_mask` (computation / gather safety). FESOM `myDim` is uneven across ranks → padding is
  required for even JAX sharding; padded lanes are inert by mask.
* **Exchange collective**: the C pack(`slist`)→Isend / Irecv→unpack(`rlist`) maps to, per device, a local
  `gather(send_idx)` → `all_to_all`/`ragged_all_to_all` (ragged per-neighbour sizes) → `scatter(recv_idx)`.
  `all_gather`-of-boundary is the simple verifiable reference; `ragged_all_to_all` the scalable form.
* **Index conventions**: `dist_<NP>` is 1-based on disk → 0-based in JAX (consistent with the existing
  0-based mesh export). `-1` sentinels (`edge_tri` boundary, outer-halo `elem_nodes`) preserved and masked.
* **Determinism**: per-device local sums reassociate vs the global sum (~1e-12); `psum` reassociates
  across devices (~1e-12). Neither hurts AD (Key fact: bit-identity to C was never the target —
  climate-close is). The CG iteration count must NOT drift (S.6).

## Post-Completion
*Follow-ups requiring more compute / external systems — informational, not this phase's gate.*

* **Full 2-yr multi-GPU climate run vs C** (the handoff's original end-state): 4×A100, the
  `kpp_bias_map.py` / inter-ref budget vs the C N-rank climate. Expect chaotic divergence in the long
  mean from `psum`-order non-determinism (the C sees the same) — assess against the inter-ref budget, not
  bit-for-bit. This is the natural "Phase 8b".
* **Multi-node** (`jax.distributed.initialize`, >4 devices / process-per-GPU): scale past one node; reuse
  the larger `dist_<NP>` (dist_8…dist_864). The single-process `shard_map` design ports directly.
* **Performance**: profile the exchange collective; `ragged_all_to_all` vs `all_gather`; overlap halo
  exchange with compute; revisit sparse SSH operator (Locked decision 4's "later optimization").
* **Phase 7a** (differentiable param tuning, `docs/plans/20260607-fesom-jax-paramtune.md`) benefits from
  sharding (bigger ensembles / longer adjoint windows) and can follow.

---

## Revision Log

### #0 — Plan created (2026-06-07)
Phase 8 sub-plan drafted after research: confirmed (via two reference ports + file:line) that the C MPI
port and the Kokkos port both use **broadcast-only halo + redundant `myDim+eDim(+eXDim)` compute +
`psum`-style reductions**, and that the handoff's "additive" description was a mischaracterization of the
C (which has no additive exchange). User chose (a) **mirror the C exactly** (unlocks the per-substep
JAX-N ↔ C-N dump diff via the bit-identical `dist_<NP>` decomposition) and (b) deliverable = **2–4 device
correctness first** (climate run = follow-up). Found the CORE2 `dist_{2,4,8,…}` partitions at
`/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/` (the bit-identical source) and confirmed the C
multi-rank port is functional (`MPI_PORT_REPORT.md`: "dist_8 partition correctness is solid"; remaining
long-run drift = chaotic Allreduce-order, not a per-substep bug → validates the per-substep gate choice).
Task ladder S.1→S.10: reader → sharded-mesh → exchange primitive → scatter gate → reductions → CG → wire
`shard_map` → AD gate → the 2–4 device GATE → docs.

### #15 — S.10 COMPLETE: Phase 8 closed, tagged `v1.1-multi-gpu` (2026-06-08)
Phase 8 (multi-GPU/multi-core via `shard_map` + halos) is **DONE**: the FESOM2→JAX model is N-vs-1
forward- and gradient-correct, validated on real A100s. Tagged **`v1.1-multi-gpu`** (the multi-GPU baseline,
the sharding analog of `v1.0-single-gpu`). Final Phase-8 lessons in `PORTING_LESSONS.md`;
`NEXT_SESSION_PROMPT.md` pivoted to the **Phase 8b** scaling fork. This plan moves to
`docs/plans/completed/`. ⚠️ The headline caveat carried into 8b: the `all_gather` halo (`halo.py`) is
**O(P·N_local)** — correct but non-scaling; **Phase 8b STEP 0 = replace it with `ragged_all_to_all`**
(point-to-point neighbour exchange; the `com_struct` slist/rlist is already in `partit.py`), gated by the
S.7/S.8 oracles, before any farc→dars→NG5 scaling numbers mean anything.

### #14 — S.9 COMPLETE: the model runs CORRECTLY on real A100s (2026-06-08)
First time the sharded model touched real GPUs (NCCL, not CPU fake-devices). `scripts/debug/phase8_s9_gpu.sbatch`
on a **4×A100 node** (job 25430592), each gate in its own fresh process (the S.8 OOM lesson). **Verdict:
the model is validated correct on real A100s.** 🎯 The assembled CORE2 step (KPP + GM/Redi + prognostic ice
+ bulk forcing) sharded across 2 A100s == single-device on every PROGNOSTIC field: ocean dynamics at the
**clean floor** (uv 1.1e-9, d_eta/eta/hbar 2.6e-11, w 2.3e-13, Kv/Av/bvfreq 4e-14…3e-18), FCT tracers T/S
climate-close (9.7e-3/6.0e-3), and the **prognostic ice** (u_ice 1.1e-7, v_ice/m_ice/a_ice/m_snow
2.9e-8…5.9e-9). The **OCEAN gradient gate passed on real A100s** (`jax.grad`-through-`shard_map` over NCCL:
d/d(k_ver) rel 3.75e-8, d/d(a_ver) rel 4.1e-4 == single-device) — the S.8 masked-NaN backward fixes hold on
GPU. **Two GPU-specific findings, neither a bug:** (1) **byte-identity is a CPU property** — GPU XLA
fuses/reorders the same arithmetic differently, so the 1-device serial-collapse worst across all State fields
was **7.66e-9** (CPU is ~0); the CPU-calibrated `< 1e-9` byte-id asserts were physically too tight for GPU.
Fix: `_BYTE_ID_ATOL = 1e-9 if cpu else 1e-7` (platform-aware; CPU gate unchanged). (2) the **EVP internal
stress σ11/22/12 jumped O(0.5)** at the viscous-plastic **yield-curve kink** (σ = ζ·ε, ζ = ice_strength/Δ,
Δ = max(√radicand, Δ_min): near-rigid ice rides Δ≈Δ_min so a ~1e-15 reassociation wiggle × huge viscosity →
a branch flip in the raw stress on a handful of near-kink elements). **Decisively not wrong physics: the
u_ice/v_ice σ DRIVES matches to 1e-7** — the net stress divergence (the force) is correct, only the
per-element stress branch flips. **User decision (S.9):** σ is a non-prognostic VP-kink diagnostic →
**EXCLUDED from the N-vs-1 gate** (`_DIAG_FIELDS`, still printed so the floor stays visible); we gate the
prognostic velocity it drives, not the stress. The **forced gradient** (the EVP-scan backward) **OOM'd**
(RESOURCE_EXHAUSTED, 249 KiB after 2.5 h) — memory-bound, NOT correctness; the OCEAN-grad pass already
validates AD-through-`shard_map` on the hardware. **User decision: close S.9 on the current run** (no more
GPU time now); forced-grad-on-GPU (needs EVP-scan memory work — checkpointing / fewer subcycles for the
gate) deferred to its own task. CPU single-device gate stays green (the re-calibration only loosens the GPU
branch + drops a diagnostic). NEXT: S.10 (tag `v1.1-multi-gpu`, move plan to completed/) → then **Phase 8b**
(the user's NG5 goal): STEP 0 = replace the O(P·N_local) `all_gather` halo with `ragged_all_to_all`, then
farc→dars→NG5 scaling.

### #13 — S.8 COMPLETE: the AD gradient gate (2026-06-08)
Built `fesom_jax/tests/test_gradient_sharded.py` (5 tests) + `scripts/test_grad_{ocean,forced}.sbatch`.
The whole differentiable sharded model is **gradient-correct on CPU fake-devices** (the BACKWARD through
every collective: the `all_gather`-exchange transpose = scatter-add, the `psum` transpose = `psum`, the CG
`custom_linear_solve` `transpose_solve` sharded, and the **ice-EVP 120-subcycle `jax.checkpoint`'d-scan
backward**). 🎯 **The gate earned its keep — the sharded REVERSE pass exposed 8 bugs the dense
single-device XLA silently folds** (the `0·inf`/`0·(±inf)` masked-NaN on the device-pad axis the plan
flagged): **7 masked-NaN guards** (`pp.py` dz_inv; `momentum.py` dZ_up/dZ_dn; `tracer_adv.py`+`ice_adv.py`
FCT `segment_max/min` `±inf`-identity on empty pad-node segments; `kpp.py` bldepth+blmix ×2 OBL
interpolations) + **`jax.jit`-around-the-`shard_map`** for the `jax.checkpoint`'d multi-step scan grad
(`closed_call`-under-`shard_map` backward, JAX 0.10). ALL forward-byte-identical — **211 single-device tests
green** (incl. `test_reference_dump` = the v1.0 byte-identity). Method: a focused `jax_debug_nans` probe
(scalar `d/d(a_ver)`, ~1 min) pinpoints each trap by source line; a `d/d(T0)` probe reaches every kernel;
the FORCED `d/d(T0)` probe pre-cleared the ice path before the ~25-min assembled gate. The gate is
**FIELD-APPROPRIATE** (the gradient analog of Decision 4): k_ver→clean-diffusion matches single-device to
3.75e-8; a_ver/k_gm/T0→FCT-advection inherit the upwind-flip floor (a_ver rel 4e-4, k_gm tiny+noisy); the
T0-field reconstruction `Bᵀ(g_p)` == dense to max 7e-8 (OCEAN) / 3.6e-5 (FORCED). With KPP active the PP
backgrounds k_ver/a_ver are 0 (unused) — the eddy seam is GM's k_gm/redi_kmax. NEXT: S.9 (the real
2–4×A100 gate) → S.10 (tag `v1.1-multi-gpu`). ⚠️ The `all_gather` halo exchange is O(P·N_local) — fine for
the 2–4-device correctness gate but NON-scaling; **Phase 8b** (scaling: farc→dars→NG5) must FIRST replace
it with `ragged_all_to_all` (point-to-point, the `com_struct` slist/rlist already in `partit.py`).

### #12 — S.7 part 3 COMPLETE: ice + multi-step scan + the PRIMARY assembled gate (2026-06-08)
Closed S.7. **Ice** (`ice_evp.evp_dynamics` exch `u_ice/v_ice` INSIDE the 120-subcycle `lax.scan`;
`ice_adv.fct_solve`/`_solve_high_order`/`_fem_fct` exch the low-order `a_l/m_l/ms_l` + high-order `dvalues`
+ limiter `icepplus/icepminus`; the GLOBAL `boundary_node` partitioned through `run_step_sharded(
boundary_node_p)` → `step(boundary_node)`): 🎯 **an `all_gather` lowers inside `jax.checkpoint` inside
`lax.scan` inside `shard_map`** (the hardest collective placement — the ice npes==1 byte-id passed). Every
ICE prognostic field (`a/m/snow/u/v_ice/sigma`) matched single-device to **0.0 bit-exact** on owned. 🐛 The
npes==2 gate caught `uv`≈7e-4 (a CLEAN field) while every ice field was bit-exact — `ice_oce_fluxes_mom`'s
`stress_surf` is a node→elem GATHER reading `a_ice` at HALO vertices, and I'd exchanged `a_ice` at step-END
(too late); moving it BEFORE the gather fixed `uv` to **1.8e-16** (the per-field ordering — ice bit-exact,
ocean fields descending by coupling depth — localized it to an ice→ocean OUTPUT instantly). **Multi-step**
(`run_steps_sharded` = step-1 eager + `lax.scan` rest under one `shard_map`): the collective-in-scan lowers
+ runs FINITE; a FREE-running 2-step compare decorrelates chaotically (the step-1 FCT flip floor →
density→uv→PP/`mo_convect` binary flips→ssh_rhs, ordered by coupling depth — Decision 4 at 2 steps), so the
per-step gate is TEACHER-FORCED (each sharded step reads the single-device's previous state ⇒ only the
within-step reassociation differs; a threading bug would show a CLEAN field diverging). **PRIMARY assembled
gate** (KPP + GM/Redi + prognostic ice + bulk/SSS forcing, npes==2): every clean ocean field MACHINE
PRECISION (`uv` 1.9e-16, `d_eta` 3.3e-16), T/S the climate-close FCT floor, ice fields bit-exact —
**PASSED**. The whole forced-path `shard_map` wiring (all exchanges + reductions + fused-kernel splits +
the in-scan collectives) is now N-vs-1 correct on owned. NEXT: S.8 (the AD gradient gate) → S.9 (the real
4×A100 GATE) → S.10 (docs/tag).

### #11 — S.7 part 3: GM/Redi + reductions + KPP forced-path exchanges DONE (2026-06-08)
Wired three of the four forced-path increments + the distributed reductions, each gated npes==2 on CPU
fake-devices (sbatch on compute) with the single-device suite green. **GM/Redi** (`gm.gm_diagnostics` exch:
`fer_gamma` nod INTRA before `fer_gamma2vel`, then `fer_uv` elem + `slope_tapered`/`Ki` nod; `step.py` 13a
`fer_w` nod): the Kokkos `SYNC_MAP` row 1b caught `fer_gamma` (the plan's per-field guess missed it — it is
gathered at HALO vertices of owned boundary elements, S.1 redundant ownership); the Redi `tr_xy`/`tr_z` need
NO exchange in JAX (recomputed from halo-complete `T_old`). A NEW **per-kernel GM gate** (`run_gm_diag_sharded`,
the S.4 scatter-gate analogue) matched `fer_uv`/`slope_tapered`/`Ki` on owned to **0.0 bit-exact** ⇒ the
exchanges are proven, and the full GM step's T/S spread (T≈8.6e-3, S≈3.9e-3 on PHC IC; ALL clean fields
machine-precision) is the FCT upwind-flip floor, NOT a missing exchange. **Reductions** (`_area_mean` →
`owned_mask`/`psum`, a 5-file `None`-default thread, byte-identical) + `_fold_forcing` (folds the
`StepForcing`/`ForcingStatic` NamedTuples to sharded `shard_map` inputs — the forced path needs the forcing
SHARDED, not closed-over/replicated). **KPP** (`kpp.mixing_kpp`/`assemble_mixing`/`eos.smooth_nod3D` exch:
the 3-sweep `blmc` smoother refreshes the halo PER SWEEP — each sweep is an element→node scatter — + `viscA`
before the node→elem `Av` gather; the per-node-COLUMN kernels `ri_iwmix`/`prestep`/`bldepth`/`blmix` +
`sw_alpha/β`/`dbsfc` are auto-complete ⇒ no exchange): `Kv`≈2.4e-14 / `Av`≈9.1e-15 on owned (machine
precision) ⇒ the smoother + `viscA` exchanges are correct; the forced npes==1 byte-id proves the whole forced
machinery (forcing fold + reductions + KPP exch) collapses to `v1.0` (a ~17 min compile — the most collectives
yet). Gates: GM 2 passed + reductions 9 passed + single-device sss 9 passed + KPP 2 passed. ⚠️ The ICE
increment is CODED (EVP in-scan `u_ice/v_ice` exchange inside the 120-subcycle `lax.scan`; the ice FCT
low-order/high-order-`dvalues`/`icepplus/icepminus` splits — same per-sweep idiom as KPP; the GLOBAL
`boundary_node` partitioned in, since the local-mesh recompute mis-flags partition-boundary nodes as coastal)
and GATING. NEXT: confirm ice + the multi-step `lax.scan` (S.7p3.5) + the PRIMARY KPP+GM+ice io_dump gate
(S.7p3.6).

### #10 — S.7 part 2: OCEAN halo exchanges + fused-kernel splits DONE (2026-06-08)
Wired the OCEAN step's halo exchanges + fused-kernel splits in `step.py` (one `_exch(field,kind)` closure +
`halo_ctx=None` dead-branch gating), `momentum.py` (`visc_filt_bidiff` exch `Uc/Vc`; `momentum_adv_scalar`/
`compute_vel_rhs` exch `un_u/un_v`), `tracer_adv.py` (`advect_one_fct`/`zalesak_limit` exch `fct_LO`+`tr_xy`+
`fct_plus/minus`), `halo.py` (`HaloCtx`), `integrate_sharded.py` (`run_step_sharded` builds + threads the
`HaloCtx`+`SSHHalo`) + the `test_step_sharded.py` npes==2 owned gate. **Used the reference ports' lessons
(per the user)** — `port_kokkos/docs/SYNC_MAP.md` is the authoritative internal-exchange (`D21`) checklist and
caught TWO scatter-result exchanges the `MPI_PORT_REPORT` folds into a kernel and I'd missed: `momentum_adv_scalar`'s
`un_u/un_v` (gathered back to cells) and the FCT `tr_xy` (read by `fill_up_dn_grad`). 🎯 **Key model insight:**
the JAX redundant-compute model needs FEWER exchanges than the C — per-node intermediates are auto-complete on
the halo (computed over the full local extent), so only SCATTER results need an exchange (the C's pre-smooth
`bvfreq` exchange is unnecessary here). **Result:** npes==1 whole step byte-identical (`max|Δ|=0`), single-device
suite ALL GREEN (475+47+36), and the npes==2 OCEAN step matches single-device on owned to <1e-9 on EVERY field
EXCEPT the FCT tracers (T,S) and cancelling SSH divergences — which match to the documented **climate-close
upwind-flip / cancellation floor** (~1e-3 on a sharp bump), NOT a missing exchange (proven: all FCT inputs match
to 1e-9, `S` matches when constant, owned==halo). The per-substep gate is therefore field-appropriate (Decision
4: per-substep correctness, not bit-identity). ⚠️ **Meta-lesson (user-flagged): run the multi-minute `shard_map`
compiles via `sbatch` on COMPUTE, never the login node.** NEXT (S.7 part 3): the forced-path exchanges (ice
EVP/FCT splits, KPP `smooth_blmc`, GM/Redi `tr_xy/tr_z`) + the multi-step `lax.scan` + the `io_dump` per-substep
gate vs the single-device dump on the KPP+GM+ice config.

### #9 — S.7 part 1: device-mesh placement + local reconstruction DONE (2026-06-07)
Built `fesom_jax/integrate_sharded.py` (fold `ShardedMesh`/`State`/`ShardedSSHOperator` → `[P*Lmax]`
`PartitionSpec('p')`; reconstruct the per-device LOCAL `Mesh`/`State`/`SSHOperator` inside `shard_map` with
LOCAL `Lmax` static sizes + a step-unused CSR dummy; run the UNMODIFIED `step`) + `tests/test_step_sharded.py`
(**3 passed**: reconstruction + npes==1 no-op + npes==2 interior-match). **Two discoveries.** (1) The kernels
use `mesh.nod2D/elem2D/edge2D` only as `num_segments`/shape bounds (`myDim_edge2D` is operator-build-only), so
a local `Mesh` with `Lmax` static sizes runs the kernels per-shard with ZERO code change; the **npes==1 whole
step under `shard_map` == dense byte-identically** (`max|Δ|=0`). (2) 🎯 `shard_map(check_vma=False)` is
required — the tridiagonal-solve / FCT `lax.scan`s carry CONSTANT initial carries (non-"varying") that JAX
0.10's varying-manual-axes typing rejects against the varying body output; relaxing it lowers the unmodified
kernels (contrast S.6's CG, which lowered with the default because its carries all derive from the sharded
`b`). **npes==2 lowers and 58% of owned nodes (deep interior) match single-device with NO exchanges** — the
local kernels are proven correct on real shards; the boundary 42% is the halo footprint the exchanges refresh.
Committed (a770000). NEXT (S.7 part 2): the ~13 ocean halo exchanges + the 5 fused-kernel splits
(`visc_filt_bidiff` exch `Uc/Vc`; `advect_one_fct`/`zalesak_limit` exch `fct_LO`+`fct_plus/minus`) +
reduction routing in `step.py`, gated behind a static arg (`None` ⇒ byte-identical), then the per-substep
CORE2 N-vs-1 gate (ocean → +KPP → +GM → +ice).

### #8 — S.6 distributed CG solve DONE (2026-06-07)
Built the distributed SSH CG in `ssh.py` (`partition_ssh_operator` + `SSHHalo` + `ShardedSSHOperator`;
folded `halo_exchange` into `ssh_matvec`/`ssh_precond`; gated `_pcg`/`solve_ssh` on an optional `halo`) +
`tests/test_ssh_sharded.py` (**9 passed**, 4 CPU fake-devices, ~54 s) + `scripts/debug/capture_core2_ssh_rhs.py`
(the realistic-rhs fixture + margin report). **Two load-bearing discoveries.** (1) 🎯 **The SSH operator
stencil EXCEEDS the node halo** — owned rows have columns outside the local node list (11664 on dist_2,
20466 on dist_4) — **but every such entry is EXACTLY zero** (the operator keeps the topological pattern
incl. numeric zeros; the far "wing" columns are all zeros, precond `∝S` zero too). So a local matvec over
(row-local ∧ col-local) entries is EXACT on owned rows; `partition_ssh_operator` asserts no NONZERO
owned-row entry is dropped. The loop-bound had to be checked on VALUES, not topology. (2) 🔴 **The real
CORE2 CG takes 127 iters (cold) / 130 (warm)**, not pi's ≈3 (the `ssh.py` docstring config was pi) — but
the count is **robustly deterministic**: captured the residual-vs-threshold margin on the real KPP+GM+ice
rhs and consecutive residuals cross the `soltol` stop by only ~1.09× (tightest 0.93 % over `rtol`), ~10
orders above the ~1e-15 `psum` reassociation. Verified N==1 iteration count (127/130 on 2 & 4 devices) and
owned `d_eta` matching to **~3e-16 (1e-15 rel)** — machine precision, far below the 1e-12 budget (each owned
row's local `segment_sum` is the same nonzero terms in the same order; the contracting CG damps the dot
reassociation). `all_gather`+`psum` inside the `while_loop` inside `custom_linear_solve` inside `shard_map`
**lowers and runs** (review #4 resolved); the `psum`'d residual makes the trip count device-identical (no
deadlock). `halo=None` traces the **exact `v1.0` graph** (43 single-device ssh tests green). Submitted an
early real-4×A100 confirmation (`scripts/debug/phase8_ssh_sharded_gpu.sbatch`; the formal multi-GPU gate is S.9).
NEXT: S.7 (wire `shard_map` around `step`/`integrate` — split the 5 fused kernels, interleave the
`halo_points` post-exchanges, gate behind a static arg; PRIMARY per-substep CORE2 N==1 gate).

### #7 — S.5 distributed reductions DONE (2026-06-07)
Built `fesom_jax/reductions.py` (`global_sum`/`global_dot` = owned-mask sum + `psum`) +
`tests/test_reductions_sharded.py` (**8 passed**, 4 fake-devices) and routed `sss_runoff._area_mean`
through it (gated `owned_mask=None` ⇒ byte-identical `jnp.sum` — the 9 sss tests stay green). **Key
finding: ALL per-step reductions are node-based** (`_area_mean` for the salt/water balances; CG dots;
`ocean_area`), and `ice_oce_fluxes` routes through the SAME `_area_mean` — so there is one reduction
primitive and the S.1 element/edge redundant-ownership caveat does NOT apply (nodes are uniquely owned,
`owned_mask` is the unique-owner mask). 2/4-device owned-sum + `psum` matches the single-device global
sum to 1e-12; a corrupted halo value leaves the result unchanged (mask correctness). `psum`'d scalars are
replicated ⇒ `out_specs=PartitionSpec()`. NEXT: S.6 (distributed CG — 🔴 iteration-count determinism is
LOAD-BEARING: a 1-iter drift = ~1e-5; verify on the real CORE2 KPP+GM+ice config, not pi).

### #6 — S.4 exchange schedule + scatter gate DONE (2026-06-07)
Built `fesom_jax/halo_points.py` (the `OCEAN_SCHEDULE` + `ICE_SCHEDULE` exchange data, ported from
`MPI_PORT_REPORT` + the ice C `fesom_exchange_nod2D` sites, each `Exch` tagged post/intra +
`FUSED_KERNELS_NEEDING_SPLIT`) + `tests/test_scatter_sharded.py` (**7 passed**, 4 fake-devices).
**Verified the `PORTING_LESSONS §4` loop-bound rule for the JAX sharding** (dist_4: owned nodes' incident
edges+elements local; owned elements' contributing edges local — 0 violations) ⇒ a LOCAL `segment_sum`
gives each OWNED entity its complete sum; the broadcast only refreshes halo. The scatter gate confirms
this: owned edge→node + edge→element scatters match the global to 1e-11 *before* broadcast, halo matches
after. Classified post (simple insert) vs intra (kernel split) exchanges; the 5 fused kernels to split in
S.7 are `visc_filt_bidiff`, ocean + `ice_adv` FCT, `ssh._pcg` (CG, S.6), and the EVP subcycle (collective
inside the 120-step `lax.scan`). The `all_gather` map refreshes the full elem extent (superset ⇒ serves
elem2D + elem2D_full). NEXT: S.5 (distributed reductions: `_area_mean`/`integrate_nod_2D` → owned-sum +
`psum`; ⚠️ element/edge reductions need the unique-owner mask from S.1).

### #5 — S.3 broadcast halo primitive + identity gate DONE (2026-06-07)
Built `fesom_jax/halo.py` (`halo_exchange` + `device_mesh` + `run_halo_exchange`) +
`tests/test_halo.py` (**9 passed** on 4 CPU fake-devices; **9 skipped** at 1 device ⇒ suite stays
green). The exchange is `all_gather('p')` then a per-lane gather `g[src_dev, src_lane]` consuming the
S.2 exchange map — owner→halo overwrite, interior identity, exactly the C `fesom_halo_exchange`.
**Sharding convention locked**: fold the device axis into the leading dim (`[P*Lmax,…]`,
`PartitionSpec('p')`) so each device sees `[Lmax,…]` (no stray size-1 axis) and the S.7 step body is
unchanged. Ported `fesom_halo_identity_test` (gid round-trip + corruption recovery) for all 3 kinds ×
{2,4} devices + a multi-level field; AD verified (linear; vjp = reverse exchange; FD-checked on interior
+ halo lanes). `src_lane≥0` always ⇒ the collective never gathers a sentinel. `run_suite.sbatch` now has
a SHARDING group (4 fake-devices) so multi-device collectives are gated in the regular suite. **The
foundation S.1→S.2→S.2b→S.3 is COMPLETE** — exchange-ready. NEXT: S.4 (exchange-point schedule from
`MPI_PORT_REPORT` + the 2-device scatter gate; classify post- vs intra-kernel exchanges).

### #4 — S.2b partition State / forcing / IC DONE (2026-06-07)
Added `partition_state` / `partition_forcing_static` / `partition_step_forcing` to `shard_mesh.py` +
`tests/test_partition_state.py` (**8 passed**, ~14 s CPU). Each pytree field is sharded by **detecting**
its entity axis (size == nod2D/elem2D), so one helper handles node/elem `State` fields AND single
(`[nod2D]`) vs scanned (`[n_steps,nod2D]`) forcing. State + mesh pad to the **same `Lmax`** (factored
`local_sizes`) so padded state lanes coincide with masked mesh lanes (provably inert). Serial `npes==1`
== dense for `State.rest` + both forcing tuples (no-op invariant). IC strategy = host-build then
partition (sidesteps the C's PHC `extrap_nod3D` per-sweep exchange). **Full suite re-run (job 25409012):
ocean group 456 passed** (= prior 417 + the 39 new S.1/S.2/S.2b tests) — ice group confirming. The
foundation (S.1→S.2→S.2b) is complete and the single-device path is intact. NEXT: S.3 (the broadcast
`halo.py` primitive + ported `fesom_halo_identity_test`).

### #3 — S.2 sharded-mesh build + export DONE (2026-06-07)
Built `fesom_jax/shard_mesh.py` (`ShardedMesh` + `build_sharded_mesh` + `export`/`load`) +
`scripts/tools/export_dist_mesh.py` + `tests/test_shard_mesh.py` (**11 passed**, ~4 s CPU). Per-device gather
by `myList`, connectivity remap (`elem_nodes`/`edges`/`edge_tri`/`edge_up_dn_tri`) global→local with
`-1` sentinels, pad-to-`Lmax` + `owned`/`valid` masks. **Serial `npes==1` sharded mesh is array-equal
to the dense `Mesh`** (no-op invariant — every non-static field; `mesh.py` untouched so the
single-device suite is structurally unaffected). **Exchange map = `(src_dev, src_lane)` per kind** for
the `all_gather` primitive (interior identity, halo←owner interior), built from a min-id-interior-owner
map — verified gid-consistent; the `com_struct` slist/rlist form is retained in `Partition` for the
`ragged_all_to_all` perf follow-up. **`nod_in_elem2D` (CSR) omitted** — IC-only (host-built in S.2b),
so the one awkward ragged field never needs sharding. **gather-on-sentinel safety proven**: owned
elements have 0 `-1` vertices (all local), so no owned output depends on a sentinel gather; pad floats
→ `1.0` (finite denominators, the masked-NaN rule on the device axis). Lesson appended. NEXT: S.2b
(`partition_state`/`partition_forcing` + host-built ICs).

### #2 — S.1 partition reader DONE (2026-06-07)
Built `fesom_jax/partit.py` (numpy port of `fesom_partit.c:68-208`) + `tests/test_partit.py` (**20
passed**, ~2 s on CPU). `Partition` + nested `ComStruct` frozen dataclasses, both registered pytrees;
all `npes` ranks read in one process (vs the C's per-rank `mype`), ragged per-rank `myList`/`com` held
as `npes`-long tuples, rectangular counts as `[npes]` arrays, static global counts as meta.
`synth_serial` gives the `npes==1` identity. **Key discovery (load-bearing for S.5):** only NODES are
uniquely partitioned (`Σ myDim_nod2D == nod2D`, disjoint interiors); ELEMENTS/EDGES are *redundantly
owned* at partition boundaries (`Σ myDim_elem2D = 245221 > 244659`, overlap 562; edges overlap 581) —
the redundant-compute model — so element/edge reductions need a unique-owner rule (S.5 must not sum
over `myDim`), and global counts are `max(gid)+1` not `Σ myDim`. **Index conventions** (verified vs
`fesom_halo.c`, not just the file): `myList`/`rlist`/`slist`/`rptr`/`sptr` shift 1-based→0-based;
`rPE`/`sPE` do NOT (MPI rank ids). Tests cover dist_2/dist_4 counts + node-unique/elem-redundant
partition + 0-based com index ranges + cross-rank halo-owner consistency + pytree round-trip + the
nares4/dist_768 format fixture + 4 error paths. Lesson appended to `PORTING_LESSONS.md`. No existing
file touched (additive) ⇒ the 483 single-device suite is unaffected (running via `run_suite.sbatch`
to confirm). NEXT: S.2 (sharded-mesh build: local remap, pad-to-`Lmax`, exchange index maps).

### #1 — plan-review pass, light revision (2026-06-07)
Ran the `plan-review` agent (verdict: NEEDS REVISION, light — approach + conventions sound, all C
file:line citations verified accurate). Applied all 9 findings: **(critical)** elevated S.6 CG
iteration-count determinism to LOAD-BEARING (a 1-iter drift = ~1e-5, 7 orders over the gate) and moved
its verification onto the CORE2 KPP+GM+ice config, not pi; **(important)** promoted sharding of
`State`/forcing/IC into a new **S.2b** (was buried in an S.7 clause; sequenced before S.4/S.6 which
consume it), added gather-on-sentinel safety to S.2, flagged the collective-inside-CG-`while_loop`
lowering in S.6, and called out the fused-kernel splits (`visc_filt_bidiff`, FCT, CG matvec) needed to
host the intra-kernel exchanges in S.4/S.7; **(minor)** committed S.3 to ONE exchange primitive
(`ragged_all_to_all` → perf follow-up), specified the `npes==1` guard as a **static arg** (not a runtime
check, mirroring `kpp_cfg=None`) in S.7, and reframed the S.9c JAX-N↔C-N tolerance as the looser
cross-runtime reassociation budget. Ladder is now S.1→S.2→S.2b→S.3→…→S.10.
