# Investigation D: the sharded forced step differs from the dense one on pi (user-approved 2026-10-25)

Context: `docs/plans/completed/20261025-recom-port.md` Task 2b (⚠️ finding). Branch `physics/sharded-dense` off `recom/port`,
worktree `/work/ab0995/a270088/fesom_jax_dev/wt_phys_shd`. Time box: 5 hours wall. Effort: high.

## The symptom

`fesom_jax/tests/test_recom_transport.py::test_every_k_sharded_equals_dense` prints it: on pi with FESOM2's own
`dist_4` partition (4 host devices), forced JRA55-do 1958-01-01, dt 2700, a synthetic T/S cold start, physics
`zstar + w_split + mEVP (whichEVP 1) + classical TKE + GM/Redi`, after 4 steps the SHARDED physics differs from the
DENSE one by max |ΔS| = 0.030 psu. Sharding should change only the summation order (≤ ~1e-12 here); after the FCT
limiter-factor fix (af5429a, `docs/PERFORMANCE_CORE2_GPU.md` "The sharded FCT seam bug") the sharded CORE2 step was
bitwise the serial one — but that check did not use w_split, and maybe not this option set.

## Steps

- [x] D1 reproduce with a small script (dense `step_jit` loop vs `integrate_sharded.run_steps_sharded_forced` on
      `partit.read_partition("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/pi", 4)`; reuse the fixture code of
      `test_recom_transport.py`): per State field max |Δ| on OWNED nodes/elements after steps 1, 2, 3, 4
- [x] D2 bisect the options, one off at a time: w_split, GM/Redi, TKE (→ PP), mEVP (→ EVP), the halo transport
      (default all_gather vs padded), ice_ave_steps; and the first step at which a field leaves round-off. Name the
      first field and, inside the step, the first substep (reuse the dump/compare helpers you find in
      `fesom_jax/io_dump.py` and the tests, or temporary prints — nothing committed that slows the step)
- [x] D3 find the cause in the code (a missing or misplaced halo exchange is the usual one: compare with FESOM2's
      exchange points in `/work/ab0995/a270088/fesom2_sp_zstar/src`, e.g. `oce_ale.F90` around w_split, and with
      `fesom_jax/halo_points.py`); fix it with a regression test (sharded == dense to round-off for the option set
      that failed) on branch `physics/sharded-dense`
- [x] D4 run `fesom_jax/tests/test_dp_lock.py` and the sharded suites touched by the fix on a compute node; say which
      DP programs the fix changes (a fix inside a path the production programs use changes their hash: report it,
      do not regenerate the lock baseline yourself)
- [x] D5 lesson in `docs/PORTING_LESSONS.md`; commits

## Rules

- **Nothing heavy on the login node** (the user's rule after a crash): every pytest, XLA compile or model run on a
  compute node (`-A ab0995 -p compute --hint=nomultithread --time=30:00`, `PYTHONPATH=<worktree> JAX_PLATFORMS=cpu
  XLA_FLAGS=--xla_force_host_platform_device_count=4`) or an interactive allocation; login node: git, grep, reading,
  sacct only. Wait on Slurm with a background shell `until` loop on `sacct`.
- Change only what the fix needs; never delete, scancel, push, or touch other worktrees/branches; English commit
  messages, no attribution lines, files staged one by one.

## Deliverable

The first diverging field/substep, the cause with file:line, the fix + test (commits), DP-lock impact, node-hours.

## Result (2026-10-25)

First divergence at step 1, in the sea ice only (sigma O(1), u_ice 0.065 m/s; all ocean fields follow), with every
option bisected (w_split, GM/Redi, TKE, mEVP/EVP, padded): none matters. Cause: with no `boundary_node_p` the runners let
the ice step rebuild the coastal mask from each device's local mesh (`ice_evp.py:270`, `ice_mevp.py:306`), which is
empty there (local edge lanes < `edge2D_in`), so the 455 coastal nodes of pi had no ice boundary condition. Production
(`run.py`, `adjoint_sharded.py`) passes the global mask and was not affected. Fix: `ShardedMesh.boundary_node` +
`integrate_sharded._coast_mask`; sharded = dense to <= 2e-11 relative on every field (S 5e-13). DP lock unchanged.
Lesson in `docs/PORTING_LESSONS.md` (2026-10-25, investigation D).
