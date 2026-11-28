# Partial bottom cells: validation against FESOM2

fesom-jax supports FESOM2's partial bottom cells (`use_partial_cell`, `partial_cell_thresh`) with the zstar and
linfs vertical coordinates (issue #3, plan `docs/plans/completed/20261125-fesom-jax-partial-cells.md`). With partial cells,
the bottom layer of each column ends at the topographic depth instead of at the next model level. This page shows
that the port reproduces FESOM2. It has three parts: the geometry, the response of the model to partial cells,
and a one-year stability check. FESOM2's own namelist describes the option as "not recommended"; the port keeps
it optional, and the default (full cells) is unchanged.

## What was compared

- **Fortran reference.** FESOM2 at `fesom2_icefw_base` 4413769a (upstream, no local changes, build
  `build_base`), CORE2 mesh, run on 4 nodes (`dist_512`). The setup is `A_main_ias2_year`, matched to the
  fesom-jax standard physics: GM/Redi, cvmix TKE, mEVP sea ice with `ice_ave_steps` 2, `opt_visc` 7, dt 1800 s,
  JRA55-do forcing from 1 January 1958, PHC3 winter initial state. Each partial-cell run is a copy of its
  full-cell twin with `use_partial_cell = .true.` and nothing else changed (`partial_cell_thresh` 0, FESOM2's
  default PGF `shchepetkin`).
- **fesom-jax.** The same physics, configs `experiments/partial_cells/core2_{zstar,linfs}_{fc,pc}.yaml`; each
  pair differs only in `use_partial_cell`. The initial state is a PHC cache built for the Fortran partition
  (`dist_512`), because FESOM2's IC extrapolation depends on the partition; the partial-cell run uses its own
  cache, in which each node's bottom layer is interpolated at that node's own depth
  (`scripts/partial_cells/build_ic.py`).
- **Mask.** The Fortran runs read the CORE2 levels on `/pool`, which were regenerated on 2026-07-03. They
  differ from ours at 4 elements and 2 nodes in the Ross Sea. These points and the nodes of those elements
  (6 nodes of 126,858) are left out of every comparison.

## G1: geometry

`scripts/partial_cells/g1_geometry.py` compares our element and node bottom depths (`zbar_e_bot`,
`zbar_n_bot`) with `zbar_e_bottom` / `zbar_n_bottom` in the Fortran run's `fesom.mesh.diag.nc`.

- Partial cells move 94.5 % of the CORE2 element bottoms, by 49 m on average.
- 65,112 of 244,655 element bottoms and 27,641 of 126,852 node bottoms differ by exactly one ulp (at most
  9.1e-13 m); every other value is bit-identical. The one-ulp values occur only where the element depth (the
  mean of the three node depths) is the selected value. The element node order is identical in both models,
  so the cause is not the summation order. The Intel build evaluates `sum(depth)/3.0` as `sum * (1/3)`;
  recomputed that way, every element and node is bit-identical. fesom-jax keeps `/3.0`, as written in the
  Fortran source.

Result: **G1_GEOMETRY_OK** (job 29278291).

## G2: response to partial cells

`scripts/partial_cells/g2_compare.py` computes, for each model X, the response Δ_X = X_partial − X_full of
three fields: the bottom-layer temperature, the bottom-layer speed (|u|, independent of the rotation of the
Fortran output to geographic axes) and SSH. Every statistic is weighted by the area of the node's control
volume. The pass bars were fixed in the plan before any run:

- **direct** (day 1 and the January mean): rms(JAX_pc − F_pc) ≤ 1.2 · rms(JAX_fc − F_fc), and both
  ≤ 0.5 · rms(Δ_F). This bar asks that partial cells do not make fesom-jax drift further from FESOM2 than it
  already does with full cells, and that the difference between the models is small compared with the
  response;
- **response** (day 1 and the 1958 mean): corr(Δ_JAX, Δ_F) ≥ 0.9 at day 1 and ≥ 0.7 for the year;
  std(Δ_JAX) / std(Δ_F) between 0.7 and 1.4.

| case | field | rms Δ_F | rms Δ_JAX | corr | amp | rms(J_pc − F_pc) | rms(J_fc − F_fc) |
|---|---|---|---|---|---|---|---|
| zstar, day 1 | bottom T (°C) | 4.55e-2 | 4.54e-2 | 1.000 | 1.00 | 8.4e-4 | 8.8e-4 |
| | bottom speed (m/s) | 1.62e-3 | 1.62e-3 | 1.000 | 1.00 | 3.2e-5 | 3.3e-5 |
| | SSH (m) | 1.65e-3 | 1.65e-3 | 1.000 | 1.00 | 2.7e-5 | 2.7e-5 |
| zstar, January | bottom T | 4.78e-2 | 4.78e-2 | 1.000 | 1.00 | 4.8e-4 | 4.5e-4 |
| | bottom speed | 5.98e-3 | 5.98e-3 | 1.000 | 1.00 | 2.5e-5 | 2.4e-5 |
| | SSH | 5.10e-3 | 5.10e-3 | 1.000 | 1.00 | 3.0e-5 | 2.8e-5 |
| zstar, 1958 mean | bottom T | 5.99e-2 | 5.99e-2 | 0.999 | 1.00 | 1.5e-3 | 2.1e-3 |
| | bottom speed | 6.51e-3 | 6.51e-3 | 1.000 | 1.00 | 8.4e-5 | 9.0e-5 |
| | SSH | 9.80e-3 | 9.80e-3 | 1.000 | 1.00 | 1.1e-4 | 1.3e-4 |
| linfs, day 1 | bottom T | 4.55e-2 | 4.54e-2 | 1.000 | 1.00 | 8.4e-4 | 8.9e-4 |
| | bottom speed | 1.62e-3 | 1.62e-3 | 1.000 | 1.00 | 3.1e-5 | 3.4e-5 |
| | SSH | 1.64e-3 | 1.64e-3 | 1.000 | 1.00 | 6.6e-5 | 6.6e-5 |
| linfs, January | bottom T | 4.78e-2 | 4.78e-2 | 1.000 | 1.00 | 4.8e-4 | 4.4e-4 |
| | bottom speed | 5.96e-3 | 5.96e-3 | 1.000 | 1.00 | 2.5e-5 | 2.4e-5 |
| | SSH | 5.06e-3 | 5.06e-3 | 1.000 | 1.00 | 6.4e-5 | 6.3e-5 |

Result: **G2_RESPONSE_OK**, every bar in every case (job 29279308, report `g2_report.json` in the runs
directory). Over the first month, the difference between fesom-jax and FESOM2 is 1–2 % of the response, and
partial cells leave it the same size as with full cells. Over the year the two models drift apart slowly, with
full cells as with partial cells: between the full-cell runs, the rms difference of the monthly bottom
temperature grows from 4.5e-4 °C in January to 7.8e-3 °C in December, and that of SSH from 2.8e-5 m to
9.8e-4 m. The year-mean response is still reproduced with a correlation of 0.999 or higher.

An earlier run of the comparison (job 29279243) reported a year-mean bottom-speed correlation of 0.94 with a
model difference as large as the response. That was an error in the script, not in the model: for fesom-jax
it averaged the speeds of the 12 monthly means, for FESOM2 it took the speed of the year-mean velocity. Both
now take the speed of the mean velocity.

![Response to partial cells](figures/partial_cells_response.png)

*(a) Node bottom depth with partial cells minus with full cells. (b–g) The response (partial minus full
cells) in FESOM2 (left) and fesom-jax (right): bottom-layer temperature on day 1 and as the 1958 mean, and
the 1958-mean SSH. Colour scales are shared within each row. Made by `scripts/partial_cells/fig_response.py`
from the fields that `g2_compare.py` writes.*

## Stability over one year

The fesom-jax zstar partial-cell run completed 1958 (17,520 steps) on one A100.
`scripts/partial_cells/check_year_restart.py` reads its end-of-year restart: the thinnest wet layer is 4.56 m
(nodes) and 4.56 m (elements), T, S and SSH are finite on every wet point, and the bottom layer of every node
and element still equals FESOM2's bottom thickness exactly, as it must, because the zstar update never
reaches the bottom layer. Result: **YEAR_RESTART_OK** (job 29279249).

## Unit tests

`fesom_jax/tests/test_partial_cells.py` (41 tests) checks each kernel against a literal loop transcribed from
the Fortran routine (`init_bottom_elem_thickness`, `init_bottom_node_thickness`, the zstar and linfs
thickness initialisation, `pressure_force_4_zxxxx_shchepetkin` and `pressure_force_4_linfs_shchepetkin`, the
SSH stiffness depth). It also covers: every branch of the bottom depth, the depth clamp, bounds, the static geometry,
restarts (reset of the bottom thicknesses and refusal of a partial-cell restart in a full-cell run), the
initial state, the guards, a finite gradient through a partial-cell step, and the degenerate case:
`partial_cell_thresh` above every layer thickness gives the full-cell zstar step to 1e-12.
`test_partial_cells_sharded.py` (5 tests) checks that a sharded partial-cell step on 1 and 2 devices agrees with
the single-device step, for zstar and linfs.

With partial cells off, the code lowers to the same programs as before the port: the three double-precision
programs of `test_dp_lock` (the CORE2 forward chunk with the standard and the GPU options, and the CORE2
adjoint) hash equal to those of main d7e4e5d (job 29327883). The full CPU suite (array 29281090, with the
end of the CORE2 group rerun in 29327277 after its 8 h limit) passes except five tests that fail on main as well,
with the same numbers (job 29327279): `test_dp_lock` (main no longer matches its lock file),
`test_ssh_gathered` (a config on main) and three finite-difference gradient checks in `scripts/tests`.

## Jobs and cost

| what | jobs | cost |
|---|---|---|
| Fortran, 1 month daily: zstar fc / pc, linfs fc / pc | 29278124, 29278125, 29278199, 29278200 | 4 × 4 nodes × ≈ 5.5 min |
| Fortran, zstar pc 1958 (monthly) | 29278204 | 4 nodes × 20 min |
| fesom-jax, day 1 (zstar fc / pc, linfs fc / pc) | 29278405, 29278408, 29278410, 29278412 | 1 A100 × 1–2.5 min each |
| fesom-jax, zstar fc / pc 1958 | 29278406, 29278409 | 1 A100 × 78 min each |
| fesom-jax, linfs fc / pc January | 29278411, 29278413 | 1 A100 × 8 min each |
| G1, G2, year check, figure (CPU) | 29278291, 29279308, 29279249, 29279275 | a few minutes each |

In total, 2.8 node-hours of Fortran and 3.0 A100-hours of fesom-jax. The full-cell Fortran year was an
existing run (`A_main_ias2_year`, 998 s on 4 nodes).

All fesom-jax runs used the model code of commit 55604bc. Three of them (29278405, 29278406, 29278408) were
started before that commit was made, from the same working tree; the last change to a model file was made at
23:42 on 25 November, before the first run started at 00:11.

## Not ported

Refused with an error when combined with partial cells: ice cavities (`use_cavity_partial_cell`), REcoM and
single precision. Also not ported: `use_depthonelem` and the linfs PGF schemes other than FESOM2's default
`shchepetkin` (`nemo`, `cubicspline`, `easypgf`).
