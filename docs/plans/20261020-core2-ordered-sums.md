# Ordered sums: element/edge -> node/element assemblies without atomic scatter-adds (CORE2 GPU speed, Tier A)

## Overview
- `ops.scatter_add` (`fesom_jax/ops.py:58-83`, a masked `jax.ops.segment_sum`) is the largest kernel item left in
  the CORE2 step: 5.4 ms of 28.3 ms on 4 GH200 (job 28291187, `lines.txt`: `ops.py:83 scatter_add`, 52
  launches/step, mostly 3-D `[Lmax, 48(, k)]` assemblies), more on A100 (11.6 ms of 74 ms in job 28284314). On a
  GPU it lowers to an atomic scatter-add: slow for f64 on 3-D fields and non-deterministic (the GPU noise floors).
- fesom3-jax assembles the same sums as ORDERED gathers ("slot sums": each target row adds its incoming entries in
  a fixed order) and spends ~2.7 ms on all its assemblies on GH200 (`~/fesom3_jax/docs/PERFORMANCE.md`).
- Goal: an opt-in ordered-sum path for `ops.scatter_add` (and its two edge wrappers) driven by precomputed inverse
  maps, deterministic on the GPU, the same arithmetic in the same order as the CPU `segment_sum` (round-off at
  most: XLA may fuse the producer of the values into the add chain and contract into FMAs — the `gpu.ssh_ell` and
  `gpu.column_unroll` precedents, which are round-off, not bitwise). The project rule: fesom-jax is the shadow of
  FESOM2 — this is a performance change that must not change values beyond round-off.
- Opt-in: `gpu.ordered_sums: true` (env `FESOM_ORDERED_SUMS=1`) in `configs/core2_gpu.yaml` after measurement; off
  by default everywhere; `core2_full.yaml` and the adjoint driver untouched.

## Design
- **Inverse maps** (host numpy, built once per mesh): for an index array `seg` (`[*S]`, flattened to `n` entries,
  `-1` = boundary sentinel) and `num_segments` rows, `inv` is `[num_segments, D]` int32 of flattened entry
  positions, each row's entries in ASCENDING position (= the CPU scatter's update order), padded with a sentinel
  position `n` that reads an appended zero row. D = the largest row count.
- **Ordered sum:** `vals_ext = concat([safe_vals_flat, zeros(1, *rest)])`; `g = vals_ext[inv]` (`[N, D, *rest]`);
  `out = g[:, 0] + g[:, 1] + ... + g[:, D-1]` as an explicit left-to-right chain (no `jnp.sum`, which XLA may
  reassociate). Consider a `custom_vjp` whose backward is the plain gather `cot_vals[i] = cot_out[seg[i]]` (masked)
  so the adjoint needs no scatter either; at minimum the default AD must be correct.
- **Which index arrays:** inventory every `ops.scatter_add` / `scatter_add_edges_to_nodes` /
  `scatter_add_edges_to_elems` call site (`grep -rn scatter_add fesom_jax/*.py`): the common `seg`s are
  `mesh.elem_nodes` (E×3 → nodes), `mesh.edges` (Ed×2 → nodes), `mesh.edge_tri` (Ed×2 → elements), plus a few
  local ones (ice advection, coarsen, obs operators, `ice_wide.py` already has its own ordered gathers). Start with
  the three mesh ones; leave the rest on `segment_sum` unless the profile says otherwise.
- **Plumbing:** call sites pass the inverse map explicitly (`ops.scatter_add(vals, mesh.edges, mesh.nod2D,
  inv=mesh.inv_edges)`); `scatter_add` uses it only when `ordered_sums_enabled()` and `inv is not None`, otherwise
  today's `segment_sum` (byte-identical default). The Mesh dataclass (`fesom_jax/mesh.py`) gains optional fields
  `inv_elem_nodes` (nod × D1), `inv_edges` (nod × D2), `inv_edge_tri` (elem × D3), `None` by default, built by a
  helper `mesh.with_inverse_maps(mesh)` on the dense mesh; on the sharded mesh they must be built PER DEVICE from
  the LOCAL connectivity after `shard_mesh.build_sharded_mesh` remaps it (`CONN_FIELDS`, `shard_mesh.py:74`), padded
  to a common D, and folded/sharded like the other node/elem fields (`NODE_FIELDS` / `ELEM_FIELDS`,
  `integrate_sharded.local_mesh`). Built only when the option is on, so the default program is unchanged.
- **Where it applies:** the forward model on the sharded GPU path (`run_steps_sharded_forced` via `run.py`), dense
  path optional. Gradient path: allowed only if the AD test passes.

## Tasks (gate-first; CPU tests with 4 fake devices; never on the login node)

### Task 1: inverse maps and the ordered sum
**Files:** Modify `fesom_jax/ops.py`, `fesom_jax/mesh.py`; tests in `fesom_jax/tests/test_ops.py` (unit, no mesh)
and a new `fesom_jax/tests/test_ordered_sums.py` (CORE2 mesh; list it in `scripts/runs/suite_groups.txt`).
- [x] `ops.inverse_map(seg, num_segments)`, `ops.ordered_sums_enabled()`, `scatter_add(..., inv=None)`
- [x] tests: small synthetic seg with sentinels (exact against a numpy loop in index order); CORE2 elem_nodes /
  edges / edge_tri with 2-D and 3-D values against `segment_sum` (CPU: bitwise or ≤ 1 ulp — record which);
  `jax.grad` against the `segment_sum` gradient (rel 1e-13)
- [x] run tests

### Task 2: call sites and the sharded mesh
**Files:** Modify the call sites (tracer advection, momentum, GM/Redi, PP/KPP, EOS/ALE if they scatter, sea ice),
`fesom_jax/shard_mesh.py`, `fesom_jax/integrate_sharded.py`, `fesom_jax/run.py`, `fesom_jax/run_config.py`
(`gpu.ordered_sums`); tests in `fesom_jax/tests/test_ordered_sums.py`, `fesom_jax/tests/test_run_config.py`.
- [x] per-device inverse maps on the sharded mesh, folded like the other fields, built only with the option on
- [x] tests: the full sharded ocean step (`test_step_sharded.py` helpers) with the option on vs off at round-off
  (1e-11 of each field's scale, the unroll test's bar), and 3 forced CORE2 steps with real forcing (the
  `test_mevp_wide_sharded.py` pattern); option off ⇒ byte-identical program (same lowered HLO hash)
- [x] run tests, plus `test_halo.py`, `test_mevp_wide_sharded.py`, `test_ssh_ell.py`

### Task 3: GPU measurement
- [x] 4 A100 (`scripts/perf/gpu_profile.sbatch`, the full `core2_gpu.yaml` options via env:
  `FESOM_HALO_COMPACT=1 FESOM_COLUMN_UNROLL=1 FESOM_MEVP_WIDE=12 FESOM_SSH_ELL=1` and `--cheb-degree 3
  --forcing-on-device --ssh-gathered`), off vs on; steady ms/step and the `ops.py` scatter rows in `lines.txt`
- [x] fidelity on the GPU: `NOISE=4` for both, then `scripts/perf/noise_ab.py` (owned-lane pair test). Expect the
  ON program to be deterministic (its own within pairs ~0 where atomics were the only noise source) — report it
- [x] ledger rows (`docs/PERFORMANCE_CORE2_GPU.md`), a lesson (`docs/PORTING_LESSONS.md`), tick this plan
- [x] report; the default stays off until the main session decides

## Results (2026-10-21)
- Implemented as planned, plus the SSH rhs and stiff increment matvec (also over `mesh.edges`, also on the
  gathered solve's global geometry). Option: `gpu.ordered_sums: true` / `FESOM_ORDERED_SUMS=1` (ordered sums with
  an optimization barrier on each sum; `=2` without the barrier, `=3` also on the values, A/B only). Opt-in, off
  in `core2_gpu.yaml`.
- 4 A100, core2_gpu options: 45.89 (off) -> 39.62 / 39.71 ms/step (-13.7 %); deterministic on the GPU; pair
  test round-off only. Without the barrier 43.19 (one 8.6 ms merged kernel).
- CPU: each assembly bitwise in isolation; full sharded forced step a few ulp at step 1 (XLA's rewrite of the
  default program); option off ⇒ the lowered program hashes the same as on the base commit.
- Gradient: default AD of the gathers, matches the `segment_sum` gradient to 1e-13; no `custom_vjp`.
- Ledger rows and the Tier A bullet: `docs/PERFORMANCE_CORE2_GPU.md`; lesson: `docs/PORTING_LESSONS.md`.
