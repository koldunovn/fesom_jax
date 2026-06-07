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
- Create: `scripts/export_dist_mesh.py`
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
- [x] `scripts/export_dist_mesh.py` → write the padded local arrays + masks + exchange maps as `.npy` +
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
- Create: `scripts/capture_core2_ssh_rhs.py` + `.sbatch` (the realistic-rhs fixture + margin report)
- Create: `scripts/phase8_ssh_sharded_gpu.sbatch` (early real-4×A100 confirmation)

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
- [ ] **split the fused kernels** flagged in S.4 (review #5) so the intra-kernel exchange seam is exposed:
      `momentum.visc_filt_bidiff` (`u_b/v_b` ‖ then `u_c/v_c`), ocean + `ice_adv` FCT (`fct_LO` ‖ then
      `fct_plus/minus`); the CG split is in S.6. Each split must be a no-op single-device.
- [ ] interleave `halo_exchange` at each S.4 exchange point inside `step`. **Gate via a static arg**
      (review #8): thread `partition`/`exch` through `step`/`integrate` in `static_argnames` (mirroring
      `kpp_cfg`/`gm_cfg`/`ice_cfg=None`, `step.py:51-54`) — `None` ⇒ the branch is not traced ⇒
      byte-identical to `v1.0`, NOT a runtime `if npes==1`. Reductions already routed (S.5/S.6).
- [ ] `integrate_sharded` = the checkpointed `lax.scan` (unchanged body) under `shard_map`; step-1-eager
      + scan-rest structure preserved.
- [ ] **PRIMARY per-step gate** (CPU fake-devices, N=2 then 4): a few-step assembled CORE2 step
      (KPP+GM+ice) sharded == the single-device `v1.0` run, per-substep ~1e-12 (reuse `io_dump`/
      per-substep compare against the single-device dump).
- [ ] write tests: 2-device 3-step run == 1-device to ~1e-12 (ocean-only first, then +ice/+KPP/+GM).
- [ ] run tests — must pass before S.8. Append lesson.

### S.8: AD through the collectives (gradient gate)

**Files:**
- Create: `scripts/phase8_grad_gate.py` + `.sbatch`
- Create: `fesom_jax/tests/test_gradient_sharded.py`

- [ ] `jax.grad` of a scalar loss of the sharded few-step run wrt `params` (the mixing/eddy seams) and
      wrt initial `state` (`T0`) — `ppermute`/`all_to_all` transpose = reverse exchange, `psum`
      transpose = `psum`; confirm reverse-mode carries through.
- [ ] **masked-NaN across devices**: padded/halo lanes must compute a FINITE value and contribute 0 to
      the cotangent (the Phase 3/5/6 discipline, now across the device axis).
- [ ] **gradient gate**: sharded `d(loss)/d(param)` == single-device gradient (the `v1.0` baseline) to
      tight tol; an FD spot-check on one well-conditioned seam (`k_gm`).
- [ ] write tests: small-mesh 2-device grad == 1-device grad; masked-lane cotangent == 0; no NaN.
- [ ] run tests — must pass before S.9. Append lesson.

### S.9: GATE — 2–4 device correctness (CPU + A100) + bonus C-N-rank diff

**Files:**
- Create: `scripts/phase8_sharded_gate_gpu.sh`, `scripts/phase8_sharded_gate_run.py`
- Create: `scripts/phase8_cn_dump_diff.py` (JAX-N-rank ↔ C-N-rank per-substep compare)

- [ ] **(a) per-substep N==1** on real devices: full assembled CORE2 step (KPP+GM+ice) a few steps on
      2 then 4×A100 (`-A ab0995_gpu -p gpu --gres=gpu:4 --nodes=1`) == single-device `v1.0`, ~1e-12/substep.
- [ ] **(b) gradient** (S.8) re-run on GPU at N=2/4.
- [ ] **(c) BONUS — JAX-N ↔ C-N**: build/run the C MPI port at 2/4 ranks on `dist_2`/`dist_4` with the
      per-rank dump, diff each substep against the JAX N-rank run. ⚠️ tolerance = the **cross-runtime
      reassociation budget** (review #9): XLA `psum` tree-order ≠ MPI `Allreduce` tree-order even at equal
      rank count, so this is **looser than the N-vs-1 JAX gate** (which shares the XLA runtime) — frame it
      as a strong cross-check, not a 1e-12 equality. (`dist_2/4` aren't in `MPI_PORT_REPORT`'s step-1
      table but are very likely clean — fewer ranks than the validated `dist_8`.) C edits, if any, on the
      port2 `jax-mesh-export` branch — **never `main`**.
- [ ] verify the single-device suite (483) still green and `npes==1` is byte-identical to `v1.0`.
- [ ] document results in the Revision Log; update `[[fesom-jax-port]]` memory at the gate.

### S.10: [Final] Docs + handoff

- [ ] update `docs/NEXT_SESSION_PROMPT.md` (Phase 8 done → the multi-node / climate follow-up, or Phase 7a).
- [ ] append the final Phase-8 lessons to `docs/PORTING_LESSONS.md`; tag `v1.1-multi-gpu`.
- [ ] move this plan to `docs/plans/completed/`.

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
`tests/test_ssh_sharded.py` (**9 passed**, 4 CPU fake-devices, ~54 s) + `scripts/capture_core2_ssh_rhs.py`
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
early real-4×A100 confirmation (`scripts/phase8_ssh_sharded_gpu.sbatch`; the formal multi-GPU gate is S.9).
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
`scripts/export_dist_mesh.py` + `tests/test_shard_mesh.py` (**11 passed**, ~4 s CPU). Per-device gather
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
