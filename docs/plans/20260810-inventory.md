# Inventory for the nn_eddy split — every path under `scripts/nn_eddy/` and `scripts/tests/`

Task 1 of `20260810-split-nn-eddy-and-promote-adjoint.md`. Generated from
`git ls-tree -r --name-only HEAD scripts/nn_eddy scripts/tests` at `d56a947` against an explicit
classification, with coverage asserted rather than eyeballed: a path in no bucket, a bucket naming
a path that does not exist, or a test file missing from the target map is an error that stops the
generator.

**86 tracked paths: 69 MOVE, 15 STAY, 2 BOTH.** `scripts/nn_eddy/` is 27 `.py` + 37 `.sbatch` +
1 `.sh`; `scripts/tests/` is 19 `test_*.py` + `conftest.py` + `__init__.py`. Nothing is deleted.

## Three corrections to the plan, all found by reading the tree rather than the directory names

**1. `scripts/tests/` is not a research directory, and `--path scripts/tests` would take three
library test files with it.** Their targets are not research code and are not moving:

| test | target | target's home |
|---|---|---|
| `test_lw_forward_seam.py` | `scripts/archive/core2_kpp_climate_run.py` | fesom_jax |
| `test_lw_foundation_check.py` | `scripts/paper/core2_lw_foundation_check.py` | fesom_jax |
| `test_write_namelist.py` | `scripts/write_namelist.py` | fesom_jax |

Consequences: Task 7's filter must name the 16 moving test files individually, not the directory;
and Task 8 must NOT "remove the `scripts` group from the manifest and from `VALID_GROUPS`" — the
group survives with 3 files plus its `conftest.py`, which has to stop putting `scripts/nn_eddy/`
on `sys.path` because that directory will not exist.

**2. `_run_steps` is called 14 times in 8 files, not 8 times in 3, and half the files are
library-side.** Counted as AST `Call` nodes, not `grep` lines:

| file | calls | at | side |
|---|---|---|---|
| `scripts/nn_eddy/horizon_scan.py` | 3 | 236, 305, 317 | research → MOVE |
| `scripts/nn_eddy/seam_gate.py` | 3 | 307, 390, 403 | research → MOVE |
| `scripts/nn_eddy/adjoint_horizon.py` | 2 | 592, 615 | research → MOVE |
| `scripts/nn_eddy/student_year.py` | 2 | 146, 156 | research → MOVE |
| `scripts/nn_eddy/determinism_probe.py` | 1 | 83 | library → STAY |
| `scripts/nn_eddy/sharded_grad_ab.py` | 1 | 158 | library → STAY |
| `scripts/paper/core2_lw_avgadj.py` | 1 | 261 | library → STAY |
| `scripts/paper/core2_lw_signflip.py` | 1 | 160 | library → STAY |

The plan's Task 6 named `adjoint_horizon.py`, `horizon_scan.py` and `seam_gate.py` and missed
`student_year.py` plus four library-side callers. Its acceptance check — `grep -r "_run_steps"
scripts/` returns nothing outside the library — therefore needs the four STAY files moved onto the
public API too, or it cannot pass. Only the four MOVE files (10 calls) are load-bearing for the
split; the other four are a library tidy-up that can follow.

There is a second private import the plan does not mention: `kappa_hat.py:145` uses
`fesom_jax.eos._shift_down` (2 call sites). It moves to the new repository, so it becomes a
cross-repo private import — the thing the split is supposed to end. It needs a public
alternative or a local copy.

**3. Task 6's proposed test — "no file under `scripts/` imports a private `fesom_jax` symbol" —
cannot pass and should not.** An AST census (`ImportFrom` from a `fesom_jax.*` module plus
`fesom_jax.mod._attr` access, so aliases like `import gm as _gm` do not count) finds **23 files**:

| where | files | symbols |
|---|---|---|
| `scripts/bench/` | 7 | `_folded_gid_owned_nod`, `_cached_maps`, `_shard_along_axis`, `_owner_map`, `_ragged_exchange_map`, `_colour_edges`, `_KINDS`, `_chunk_dates`, `_SubMesh` |
| `scripts/debug/` | 5 | `_chunk_dates`, `_all_state_fields`, `_shard_along_axis` |
| `scripts/paper/` | 4 | `_run_steps`, `_layer_center_Z`, `_safe_sqrt`, `_shift_down`, `_shard_along_axis` |
| `scripts/nn_eddy/` | 7 | `_run_steps` (6), `_shift_down` (1) |

Sixteen of the 23 are in-repo tools of the library with nothing to do with this split. Of the
seven under `scripts/nn_eddy/`, two STAY, so **five files carry a private import across the
boundary**: `adjoint_horizon.py`, `horizon_scan.py`, `seam_gate.py`, `student_year.py` (all
`_run_steps`) and `kappa_hat.py` (`_shift_down`). Those five are the whole of Task 6's real scope.
The test that keeps the split from leaking back belongs in the NEW repository and asserts the
boundary that matters: nothing in it imports a private symbol from the INSTALLED `fesom_jax`.

## Two things that decide where a borderline file goes

**A file under `scripts/nn_eddy/` whose subject is the library stays in the library.** Three `.py`
qualify — `determinism_probe.py` (is the model bit-reproducible), `freeze_equivalence.py` (do the
library's freeze flag and the script's monkeypatch agree), `sharded_grad_ab.py` (does sharding the
adjoint pay) — and library docs and tests already cite two of them by path. `seam_gate.py` reads
the same way and is not: it is the NN-GM plan's own M1 gate, which happens to exercise a library
seam.

**Nine `.sbatch` under `scripts/nn_eddy/` invoke `fesom_jax/tests/` files and nothing else, or
drive one of the three scripts above.** They would reference test files the new repository does not
contain. `gm_nn_seam_only.sbatch` is the one mixed case: 4 library test files and 2
`scripts/tests/` files that move, so those two lines drop when it stays.

## Non-code artefacts

- `docs/PORTING_LESSONS.md` STAYS in `fesom_jax`, whole. It is one continuous document and its
  entries cross-reference each other by number.
- `docs/ADJOINT_HORIZON.md` STAYS: after Tasks 3–5 it documents a library module. It cites
  `determinism_probe.py` for a published number, which is why that file stays too.
- `docs/HANDOFF-*.md` stay untracked on disk, gitignored by `/docs/HANDOFF-*.md` (`.gitignore:73`).
- `docs/plans/` stay in the main checkout; the copies here are the worktree's.
- `/work/ba1583/…` and `/work/ab0995/…` data products are referenced by path from moved scripts
  and are not in either repository.

## The table

`MOVE` → new repository. `STAY` → `fesom_jax`. `BOTH` → each tree needs its own copy.
The `test` column is the test file that covers a script, or for a test file, what it imports.

| path | destination | test | note |
|---|---|---|---|
| `scripts/nn_eddy/adjoint_ckpt.py` | **MOVE** | `test_adjoint_ckpt.py` | checkpoint/resume for the chunked reverse sweep — revisit at Task 3, it may belong with the promoted adjoint |
| `scripts/nn_eddy/adjoint_exact_ice.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_fd_long.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_horizon.py` | **MOVE** | `test_freeze_knobs_install.py` |  |
| `scripts/nn_eddy/adjoint_horizon.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_horizon_ladder.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_ice_rheology.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_ice_toyoda.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_kpp_vs_tke.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_long_chain.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_mixfrozen.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_month.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_prod_flags.sbatch` | **STAY** |  | runs fesom_jax/tests/ only |
| `scripts/nn_eddy/adjoint_repro.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_repro_det.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_resume_check.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/adjoint_tests.sbatch` | **STAY** |  | runs fesom_jax/tests/ only |
| `scripts/nn_eddy/adjoint_tests_gpu.sbatch` | **STAY** |  | runs fesom_jax/tests/ only |
| `scripts/nn_eddy/adjoint_tests_verify.sbatch` | **STAY** |  | runs fesom_jax/tests/ only |
| `scripts/nn_eddy/adjoint_tests_verify_gpu.sbatch` | **STAY** |  | runs fesom_jax/tests/ only |
| `scripts/nn_eddy/analyse_teacher_rate.py` | **MOVE** |  |  |
| `scripts/nn_eddy/apriori_eval.py` | **MOVE** | `test_apriori_eval.py` |  |
| `scripts/nn_eddy/apriori_eval.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/armb_fit.py` | **MOVE** | `test_armb_fit.py` |  |
| `scripts/nn_eddy/armb_fit.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/averaging_scan.py` | **MOVE** |  |  |
| `scripts/nn_eddy/check_coarse_product.py` | **MOVE** | `test_check_coarse_product.py` |  |
| `scripts/nn_eddy/check_teacher_snapshots.py` | **MOVE** |  |  |
| `scripts/nn_eddy/coarsen_eta.py` | **MOVE** | `test_coarsen_eta.py` |  |
| `scripts/nn_eddy/coarsen_eta.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/coarsen_stream.py` | **MOVE** | `test_coarsen_stream.py` |  |
| `scripts/nn_eddy/coarsen_stream.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/compare_moc_vs_pyfesom2.py` | **MOVE** |  |  |
| `scripts/nn_eddy/determinism_probe.py` | **STAY** |  | the model's own bit-reproducibility; ADJOINT_HORIZON.md cites it as the source of a published number |
| `scripts/nn_eddy/determinism_probe.sbatch` | **STAY** |  | driver for a STAY script |
| `scripts/nn_eddy/divergence_fit.py` | **MOVE** | `test_divergence_fit.py` |  |
| `scripts/nn_eddy/divergence_fit.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/features_from_product.py` | **MOVE** | `test_features_from_product.py` |  |
| `scripts/nn_eddy/flux_divergence.py` | **MOVE** | `test_flux_divergence.py` |  |
| `scripts/nn_eddy/freeze_equivalence.py` | **STAY** |  | asserts AdjointConfig(freeze_mixing=True) equals the script-side monkeypatch — a statement about the library |
| `scripts/nn_eddy/gm_nn_seam_only.sbatch` | **STAY** |  | runs 4 fesom_jax/tests/ files; its 2 scripts/tests/ files move, so those two lines drop |
| `scripts/nn_eddy/gm_nn_seam_tests.sbatch` | **STAY** |  | runs 15 fesom_jax/tests/ files |
| `scripts/nn_eddy/horizon_scan.py` | **MOVE** | `test_horizon_scan.py`, `test_student_year.py` | superseded for its numbers, imported by 4 live scripts, so edited rather than deleted |
| `scripts/nn_eddy/horizon_scan.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/horizon_scan_stopgrad.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/kappa_hat.py` | **MOVE** | `test_kappa_hat.py` |  |
| `scripts/nn_eddy/mesh_diag.py` | **MOVE** | `test_mesh_diag.py`, `test_section_transport.py` |  |
| `scripts/nn_eddy/migrate_data_root.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/nn_data.py` | **MOVE** | `test_nn_data.py` |  |
| `scripts/nn_eddy/offline_fit.py` | **MOVE** |  |  |
| `scripts/nn_eddy/offline_fit.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/pyfesom2_ref_moc.py` | **MOVE** |  |  |
| `scripts/nn_eddy/sample_teacher_rate.sh` | **MOVE** |  |  |
| `scripts/nn_eddy/seam_gate.py` | **MOVE** |  | NN-GM child-plan M1 gate, so research despite gating a library seam |
| `scripts/nn_eddy/seam_gate.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/seam_gate_dotprod.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/seam_gate_freeze_ab.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/seam_gate_grad.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/sharded_grad_ab.py` | **STAY** |  | whether sharding the adjoint pays; a property of integrate_sharded, cited by fesom_jax/tests/test_gradient_sharded.py |
| `scripts/nn_eddy/sharded_grad_ab.sbatch` | **STAY** |  | driver for a STAY script |
| `scripts/nn_eddy/signal_eval.py` | **MOVE** | `test_nn_data.py`, `test_student_year.py` |  |
| `scripts/nn_eddy/signal_eval.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/student_year.py` | **MOVE** | `test_student_year.py` |  |
| `scripts/nn_eddy/student_year.sbatch` | **MOVE** |  |  |
| `scripts/nn_eddy/verify_output.py` | **MOVE** |  |  |
| `scripts/tests/__init__.py` | **BOTH** |  | package marker; both trees need one |
| `scripts/tests/conftest.py` | **BOTH** |  | the fesom_jax copy must drop scripts/nn_eddy from sys.path (that directory goes away); the new repo gets its own |
| `scripts/tests/test_adjoint_ckpt.py` | **MOVE** | → `adjoint_ckpt.py` |  |
| `scripts/tests/test_apriori_eval.py` | **MOVE** | → `apriori_eval.py` |  |
| `scripts/tests/test_armb_fit.py` | **MOVE** | → `armb_fit.py` |  |
| `scripts/tests/test_check_coarse_product.py` | **MOVE** | → `check_coarse_product.py` |  |
| `scripts/tests/test_coarsen_eta.py` | **MOVE** | → `coarsen_eta.py` |  |
| `scripts/tests/test_coarsen_stream.py` | **MOVE** | → `coarsen_stream.py` |  |
| `scripts/tests/test_divergence_fit.py` | **MOVE** | → `divergence_fit.py` |  |
| `scripts/tests/test_features_from_product.py` | **MOVE** | → `features_from_product.py` |  |
| `scripts/tests/test_flux_divergence.py` | **MOVE** | → `flux_divergence.py` |  |
| `scripts/tests/test_freeze_knobs_install.py` | **MOVE** | → `adjoint_horizon.py` |  |
| `scripts/tests/test_horizon_scan.py` | **MOVE** | → `horizon_scan.py` |  |
| `scripts/tests/test_kappa_hat.py` | **MOVE** | → `kappa_hat.py` |  |
| `scripts/tests/test_lw_forward_seam.py` | **STAY** | → `scripts/archive/core2_kpp_climate_run.py` | tests scripts/archive/core2_kpp_climate_run.py |
| `scripts/tests/test_lw_foundation_check.py` | **STAY** | → `scripts/paper/core2_lw_foundation_check.py` | tests scripts/paper/core2_lw_foundation_check.py |
| `scripts/tests/test_mesh_diag.py` | **MOVE** | → `mesh_diag.py` |  |
| `scripts/tests/test_nn_data.py` | **MOVE** | → `nn_data.py + signal_eval.py` |  |
| `scripts/tests/test_section_transport.py` | **MOVE** | → `mesh_diag.py` |  |
| `scripts/tests/test_student_year.py` | **MOVE** | → `student_year.py + horizon_scan.py + signal_eval.py` |  |
| `scripts/tests/test_write_namelist.py` | **STAY** | → `scripts/write_namelist.py` | tests scripts/write_namelist.py |
