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

- [ ] `Partition` frozen dataclass (registered pytree): per-device `myDim/eDim/eXDim` for nod/elem/edge,
      `myList_*` (global IDs, 0-based after the −1 shift), and the three `com_struct`s
      (`rPE,rptr,rlist,sPE,sptr,slist`) for nod2D/elem2D/elem2D_full. Static counts as metadata.
- [ ] `read_partition(mesh_dir, npes)` — numpy port of `read_rpart`/`read_my_list`/`read_com_info`
      (`fesom_partit.c:68-208`); store global IDs 0-based, com indices 0-based, keep the 3 com blocks.
- [ ] `synth_serial(nod2D,elem2D,edge2D)` — the `npes==1` identity partit (`:280-308`) so the same code
      path covers single-device (zero halo, no neighbours).
- [ ] write tests: read CORE2 `dist_2` + `dist_4`; assert `Σ_d myDim_nod2D == nod2D`,
      `rlist` lengths == `eDim`, `slist` length == `sptr[-1]`, `myList` monotone-consistent, rank header
      matches; format-fixture read of `…/meshes/nares4/dist_768/com_info00000.out`.
- [ ] write error tests: missing `dist_<NP>` raises; `npes` mismatch in `rpart.out` raises.
- [ ] run tests — must pass before S.2. Append lesson.

### S.2: Sharded-mesh build + export (local remap, pad-to-`Lmax`, exchange index maps)

**Files:**
- Create: `fesom_jax/shard_mesh.py`
- Modify: `fesom_jax/mesh.py` (optional halo/partition fields on `Mesh`, or a parallel `ShardedMesh`)
- Create: `scripts/export_dist_mesh.py`
- Create: `fesom_jax/tests/test_shard_mesh.py`

- [ ] `build_sharded_mesh(mesh, partition)` → per-device local arrays: gather each global `Mesh` field
      by `myList_*`; **remap connectivity** `elem_nodes`/`edges`/`edge_tri`/`edge_up_dn_tri`/
      `nod_in_elem2D` global→local (a global→local lookup per device; `-1` sentinels preserved; outer-halo
      unmappable vertices → `-1`, mirroring `MPI_PORT_REPORT` `elem_nodes` halo note).
- [ ] **pad-to-`Lmax`** each entity kind to a common per-kind max over devices; build `owned_mask`
      (`[P,Lmax]` true on `[0:myDim_d]`) and `valid_mask` (true on `[0:myDim+eDim]`) — padded lanes carry
      a safe value (0 / `-1` index) and are masked (the masked-lane rule).
- [ ] ⚠️ **gather-on-sentinel safety** (review #3): `ops.gather` is a plain `field[idx]` (`ops.py:33`) —
      a JAX gather at `-1`/pad returns the LAST lane (garbage), unlike `scatter_add` which already masks
      `seg<0` (`ops.py:77-81`). Clamp sentinel/pad indices to 0 before every connectivity gather **and**
      mask the result; prove no **owned-entity** output depends on a sentinel gather (the `eDim` element
      ring has all 3 vertices valid; only `eXDim`/`elem2D_full` lanes may be `-1`, feeding only the
      `elem2D_full` exchange, not node scatters). Consider hardening `ops.gather` itself.
- [ ] build the **exchange index maps** for the JAX collective: per device, flat `send_idx` (local
      indices = `slist`), `recv_idx` (local indices = `rlist`), and per-neighbour segment offsets +
      target-device ids — the data `halo.py` consumes. One set per com kind (nod2D/elem2D/elem2D_full).
- [ ] `scripts/export_dist_mesh.py` → write `data/mesh_core2_dist<NP>/` (the padded local arrays + masks
      + exchange maps as `.npy` + `meta.txt`); large files on `/work`. Loadable like `load_mesh`.
- [ ] write tests: every local `elem_nodes` index `< Lmax_nod` or `==-1`; halo node global-ids equal
      their owner's; `build→export→reload` round-trips; the **serial `NP=1`** sharded mesh is array-equal
      to the dense `Mesh` (the no-op invariant).
- [ ] run tests — must pass before S.2b. Append lesson.

### S.2b: Partition `State` + per-step forcing + IC builders (load-bearing — review #2/#6)

**Files:**
- Modify: `fesom_jax/shard_mesh.py` (add `partition_state` / `partition_forcing`)
- Create: `fesom_jax/tests/test_partition_state.py`

> Promoted out of S.4/S.7 — S.4's scatter gate and S.6's CG test cannot produce sharded inputs without
> this, so it must land **before** S.3's consumers. A real, testable deliverable, not a clause.

- [ ] `partition_state(global_state, partition)` — gather each `State` field (40 fields over
      `nod2D`/`elem2D`, `state.py`) by `myList_{nod,elem,edge}`, pad to `Lmax`, return a per-device padded
      `State` (same pytree; padded lanes masked-safe).
- [ ] `partition_forcing(...)` — same for `StepForcing` (10 `[nod2D]` fields) + `ForcingStatic` (5
      `[nod2D]` fields, `core2_forcing.py:63-85`); the per-step `[n_steps,nod2D,…]` stack shards by the
      node partition.
- [ ] cover the **IC builders**: PHC IC, ice cold-start IC, `State.rest` — build the global IC on host
      then `partition_state`. (Bonus: this **sidesteps the C's PHC `extrap_nod3D` per-sweep exchange**,
      `MPI_PORT_REPORT.md` startup — same host-build trick as the SSH operator.)
- [ ] write tests: serial-`NP=1` `partition_state`/`partition_forcing` == the dense `State`/forcing
      (the no-op round-trip, mirroring S.2's mesh invariant); 2-device gather places each owned field on
      the right device; padded lanes are inert.
- [ ] run tests — must pass before S.3. Append lesson.

### S.3: Broadcast halo-exchange primitive (`halo.py`) + identity gate

**Files:**
- Create: `fesom_jax/halo.py`
- Create: `fesom_jax/tests/test_halo.py`

- [ ] `halo_exchange(field, exch, kind)` under `shard_map`: pack `field[send_idx]` → device-to-device →
      scatter into `field[recv_idx]` (owner→halo overwrite). Handles `[Lmax]`, `[Lmax,nl]`, `[Lmax,nl,2]`
      (the `n_levels·n_components` stride, `fesom_halo.h:18-28`). **Use ONE simple primitive for the gate**
      (review #7): `all_gather` of boundary values, or `all_to_all` over `Lmax`-padded buffers — whichever
      lowers cleanest. `ragged_all_to_all` (ragged per-neighbour sizes) is a **Post-Completion perf**
      item, not needed for 2–4 device correctness.
- [ ] **identity gate** (port `fesom_halo_identity_test`): set `f[local]=global_id`, exchange on 2 CPU
      devices, assert every halo lane carries its owner's gid; corruption-recovery (clobber one halo,
      re-exchange, restored).
- [ ] cover all three kinds (nod2D, elem2D, elem2D_full) + a multi-level field.
- [ ] write AD test: `halo_exchange` is linear; its vjp = the **reverse** exchange (halo→owner additive
      gather) — `jax.linear_transpose`/FD check on a tiny mesh.
- [ ] run tests — must pass before S.4. Append lesson.

### S.4: Exchange-point map + the 2-device scatter correctness gate

**Files:**
- Create: `fesom_jax/halo_points.py` (the per-substep exchange schedule, ported from `MPI_PORT_REPORT`)
- Modify: `fesom_jax/ops.py` (document the local-scatter + loop-bound contract; no math change)
- Create: `fesom_jax/tests/test_scatter_sharded.py`

- [ ] encode the **exchange schedule** (which field, which kind, at which substep) as data, from the
      `MPI_PORT_REPORT` table — the single source S.7 iterates; cross-check field-by-field against the C
      `fesom_exchange_*` call sites.
- [ ] include the **sea-ice exchanges** (the table above is the ocean step): enumerate the C
      `fesom_ice_evp.c` / `fesom_ice_fct.c` / `fesom_ice_coupling.c` `fesom_exchange_*` calls. ⚠️ the EVP
      stress/velocity exchange happens **inside the 120-subcycle `lax.scan`** (`ice_evp.py`) — a collective
      inside `scan` under `shard_map`; verify it lowers/transposes (it should; flag if not) and consider
      its 120×/step cost for the perf follow-up.
- [ ] audit the JAX kernels' scatter loop bounds vs `PORTING_LESSONS §4`: confirm that, computed over the
      local (owned+halo) entities + a post-kernel broadcast, each owned node receives its complete sum.
      Record any kernel whose loop must explicitly include halo entities.
- [ ] ⚠️ **classify post-kernel vs intra-kernel exchanges** (review #5): several scheduled exchanges fire
      *inside* a kernel that the JAX port currently **fuses** — `visc_filt_bcksct` exchanges `u_b/v_b`
      then `u_c/v_c` mid-bilaplacian (`MPI_PORT_REPORT:127`), tracer FCT exchanges `fct_LO` then
      `fct_plus/minus` mid-limiter (`:140`), CG exchanges `pp/rr` mid-iteration. Tag each exchange
      post/intra; **enumerate the kernels needing a split** (`momentum.visc_filt_bidiff`, ocean +
      `ice_adv` FCT, the CG matvec) so S.7 budgets the refactor — a fused 2nd stage reads a halo-stale 1st
      stage → owned-node boundary error (the C's hardest surface).
- [ ] **2-device scatter gate**: take one representative edge→node scatter (`compute_ssh_rhs`,
      `ssh.py:183-220`) — run it sharded on 2 CPU devices (local segment_sum + `ssh_rhs` broadcast) and
      assert it equals the single-device `compute_ssh_rhs` on owned nodes to ~1e-12.
- [ ] write tests for an edge→element scatter (biharmonic visc path) too — exercises `com_elem2D`.
- [ ] run tests — must pass before S.5. Append lesson.

### S.5: Distributed reductions (`jnp.sum` → `jax.lax.psum`)

**Files:**
- Modify: `fesom_jax/sss_runoff.py` (`_area_mean`), `fesom_jax/ice_coupling.py` (`integrate_nod_2D`)
- Create: `fesom_jax/reductions.py` (a `global_sum(owned_vals, owned_mask)` helper = owned-sum + `psum`)
- Create: `fesom_jax/tests/test_reductions_sharded.py`

- [ ] `global_sum`/`global_dot`: sum over **owned, unpadded** lanes only (mask halo+pad → 0 to avoid
      double-count), then `jax.lax.psum` over the device axis. (Mirrors `Kokkos parallel_reduce(0,myDim)` +
      `MPI_Allreduce`.)
- [ ] route `_area_mean` and `integrate_nod_2D` (virtual-salt / relax-salt / water-flux balances) through
      it; the `ocean_area` constant becomes a `psum` at mesh build (or precomputed, since it is static).
- [ ] write tests: 2-device area-mean / integrate == single-device to ~1e-13; the owned-mask correctly
      excludes halo (a deliberately-corrupted halo value does not change the result).
- [ ] run tests — must pass before S.6. Append lesson.

### S.6: Distributed CG solve (`ssh.py`)

**Files:**
- Modify: `fesom_jax/ssh.py` (`build_ssh_operator`, `ssh_matvec`, `ssh_precond`, `_pcg`, `solve_ssh`)
- Create: `fesom_jax/tests/test_ssh_sharded.py`

- [ ] partition the **static operator**: `build_ssh_operator` emits per-device local `rows/cols`
      (node-ownership-consistent; columns may reference halo nodes) so the matvec is a local
      `segment_sum` into `[Lmax_nod]`.
- [ ] `ssh_matvec`/`ssh_precond` → local segment_sum **+ halo-exchange** the result (`pp` before SpMV,
      `rr` after the residual update, `X` final — the `MPI_PORT_REPORT` "inside CG" schedule); the CG
      dot-products → `global_dot` (`psum`).
- [ ] 🔴 **iteration-count determinism — LOAD-BEARING for the whole 1e-12 gate** (review #1): `_pcg`
      early-stops at the loose `soltol=1e-5` (`ssh.py:226-269`), so a **single**-iteration divergence
      moves `d_eta` by `~soltol·‖b‖ ≈ 1e-5·‖b‖` — *seven orders above* the per-substep budget. The
      residual norm is now a `psum`, so the count CAN drift. Assert the count is identical N-device vs
      1-device **on the actual CORE2 KPP+GM+ice reference** (NOT pi — the cited "≈3 iters, cond≈800" is the
      wrong config); record the residual-vs-threshold margin on that config. If it is ever marginal, pin
      the forward to a fixed iteration count for the sharded path.
- [ ] ⚠️ **collective inside the CG `while_loop`** (review #4): `_pcg` is a `lax.while_loop` with a
      data-dependent trip count (`:246-265`) now carrying a per-iter `pp` `halo_exchange` **and** `psum`
      dots. Verify it lowers under `shard_map`; the `psum`'d residual makes the trip count device-identical
      (all devices see the same value → no deadlock). Reverse-mode is already safe — `custom_linear_solve`
      (`:314`) means the backward calls `transpose_solve`, NOT AD-through-the-`while_loop` (state this so
      it is not re-litigated).
- [ ] keep `custom_linear_solve` (forward early-stop / transpose tight) — the transpose solve also runs
      sharded; the implicit-diff gradient must survive (checked in S.8).
- [ ] write tests: 2-device `solve_ssh` `d_eta` == single-device to ~1e-12 on owned nodes; iteration
      count equal; zero-rhs short-circuit still holds.
- [ ] run tests — must pass before S.7. Append lesson.

### S.7: Wire `shard_map` around `step`/`integrate`

**Files:**
- Create: `fesom_jax/integrate_sharded.py` (the `shard_map` entry point + device-mesh placement)
- Modify: `fesom_jax/step.py` (insert the `halo_points` exchanges at the C's exchange points)
- Create: `fesom_jax/tests/test_step_sharded.py`

- [ ] device-mesh placement: put the sharded `Mesh`/`SSHOperator` (S.2) + `State`/`step_forcings` (S.2b)
      on a 1-D `jax.sharding.Mesh('p')` with `PartitionSpec('p')` on the leading device axis; `shard_map`
      the body.
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
