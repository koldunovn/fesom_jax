# Split-explicit SSH for CORE2 on the GPU (CORE2 GPU speed plan, Task 14 / B4)

## Overview
- **Why.** At 817 SYPD (4 GH200, SP, dt 2700, ice-2; profile job 28303935, `docs/PERFORMANCE_CORE2_GPU.md` last
  section) the gathered CG SSH solve is the largest single area: 1.75 ms of 9.07 ms/step (19.5 %, past Task 14's
  15 % mark), and the GPUs leave its host-driven while loop at different times, so the NCCL waits after it add
  ~0.26 ms/step (nsys). fesom3-jax reaches 7.26-7.4 ms/step with a split-explicit (SE) barotropic solver; the Kokkos
  port measured -7 to -11 % on CORE2 GPU with SE. Estimate for fesom-jax (not a measurement): -1.0 to -1.3 ms/step,
  i.e. ~920-950 SYPD.
- **What.** An opt-in barotropic subcycling that replaces steps 8-12 of `fesom_jax/step.py` (SSH rhs, CG, velocity
  update, hbar, eta blend) by: barotropic forcing from the full 3-D predictor -> M explicit substeps of the 2-D
  (eta, U-bar) system -> trim of the 3-D velocity to the mean transport -> eta = hbar = eta^M. Steps 1-7 (momentum
  predictor) and the zstar tail (step 13 onward) are reused. Off by default; `core2_full.yaml` and every adjoint
  configuration keep the CG.
- **This is a physics change (Tier B), not a speed option.** It changes the solution; it is judged like ice-2:
  invariants at round-off, a bitwise CPU gate against an oracle where one exists, stability screens, and a twin
  against the CG run.
- Recon (read only, 2026-10-21): `scratchpad se_recon.md` of this session, summarised in Context below.

**Status 2026-10-22 morning.** Tasks 2-8 done on `perf/core2-se` (ledger sections "Split-explicit SSH: ..." and the
final stack): the twin is the CG's within the SP - DP spread, the final stack runs 7.39 ms/step steady (1001 SYPD) and
its 10-year run is bitwise the first SE twin. Open: the Task 8 STOP (SE into `core2_gpu_fast.yaml`), the se0 oracle
runs (Task 1, a command for Nikolay), Task 9 (lessons written; the plan to `completed/` after the STOP).

## Decisions for Nikolay (before any code)

User 2026-10-21: "SE as proposed" — D1 Sergey's SM AB3-AM4 on the FESOM2 path, with the barotropic drag of his
2026-10-18 answers (a switch: off in the bitwise gate against `se0`, which has no drag; on/off both in the Task 7
screens), D3 port_kokkos's viscosity, D4 from the screens, D5 SE in WP. D2 (who runs `se0`) still open.

**D1 — which scheme.** Three variants exist in his projects (details in Context):

| | FESOM2 2.7 SE (upstream) | Sergey's SM AB3-AM4 on the FESOM2 path (port_kokkos M12) | the same + Sergey's 2026-10-18 answers (FESOM_F / fesom3-jax) |
|---|---|---|---|
| scheme | dissipative forward-backward (Demange 2019), theta 0.14, semi-implicit Coriolis | Shchepetkin-McWilliams AB3-AM4, chi-dissipation, beta 0.281105 | as M12, + barotropic bottom drag in the substeps |
| coupling to 3-D | the whole momentum path in transports (`se_uvh`): 4 new transport kernels, own zstar thickness and w | velocity predictor reused (R = sum h u_rhs), trim of the 3-D velocity, zstar tail reused | as M12 |
| mesh / ALE | FESOM2, linfs + zstar | FESOM2 CORE2 mesh, zstar only | FESOM3 vertex-bottom mesh (FESOM_F); fesom3-jax linfs only |
| oracle | the upstream Fortran (namelist switch), no dumps | **port_kokkos frozen serial binary `se0` with `FESOM_SE_DUMP`** (C++ port of FESOM2, the C oracle's lineage) | none on a FESOM2 mesh |
| evidence | never run in production upstream (no enabled namelist, NetCDF restart drops the SE state, `sfirst` uninitialised until 3c75f68c) | CORE2 dt 1800: invariants at round-off, 3000-step screens, 1-year twin SE-SI SSH rms 0.2 mm | FESOM_F CORE2 zstar year clean at chi 0.1 with the drag |
| Sergey | "the variant to be replaced" (`subcycling.tex:52,157`) | his notes | his answers |

Recommendation: **Sergey's SM AB3-AM4 on the FESOM2 path as port_kokkos M12 built it (bitwise against its `se0`
binary on CPU), plus the barotropic bottom drag of Sergey's answer 3 as a separate switch** (FESOM_F: with the drag
chi 0.1 is stable, without it chi 0.1 blew up there; port_kokkos ran chi 0.1 without drag on CORE2 dt 1800). This is
a STATED deviation from upstream FESOM2 (upstream has the FB scheme); the user's direction (2026-10-21): "we did
split explicit not exactly as FESOM2 has it, Sergey gave additional instructions".

**D2 — the oracle run.** port_kokkos jobs belong to the Kokkos session. The bitwise gate needs a few short CPU runs
of the frozen `/work/ab0995/a270088/port2/m12/bin/fesom_port_serial_se0` (sha in `SHA256.se0`) with
`FESOM_SSH_MODE=se FESOM_SE_DUMP=1` on the gate mesh: either the Kokkos session makes them, or Nikolay allows this
session to run that frozen binary (CPU, < 1 node-h, its own output directory). Without it the gate is invariants +
the twin only.

**D3 — barotropic viscosity.** port_kokkos: live two-term flow-dependent gamma0/gamma1 (Zenodo 10 / 2750);
FESOM_F: linear A_h = 0.01 m/s x edge length on U^m - U^n. Recommendation: port_kokkos's (it is the oracle's);
FESOM_F's linear form only if the screens need it.

**D4 — chi and M at dt 2700.** chi 0.1 (port_kokkos, FESOM_F with drag) vs 0.2 (Sergey: his setups). M from a CFL
guard by power iteration on the discrete 2-D operator (FESOM_F; port_kokkos's resolution estimate under-predicts
omega_max 3.3x on CORE2 and M 35 blew up). FESOM_F at dt 1800: M_min 40 at chi 0.1, so ~60 at dt 2700 (fesom3-jax
runs 60 at 45 min). Decided by the Task 7 screens; the user picks the production pair.

**D5 — single precision.** FESOM2 #1207 has no `WP_full` in its SE file; fesom3-jax runs SE float32 throughout (an
float64 mean-transport accumulator changed nothing beyond chaos). Recommendation: SE in WP, with `hnode`/`hnode_new`
float64 as #1207 everywhere; a new row in the SP policy table.

## Context (from discovery)
- fesom-jax plug-in: `fesom_jax/step.py:387-470` (5-7 momentum predictor; 8 `ssh.compute_ssh_rhs`; 9
  `ssh.solve_ssh` / `solve_ssh_gathered`; 10 `momentum.update_vel`; 11-12 `ssh.compute_hbar`, `ssh.eta_n_update`;
  13 `ale.compute_w` + `ale.vert_vel_zstar_distribute` from `hbar - hbar_old`). The CG's implicit-derivative
  `custom_jvp` (`ssh.py:1013-1076`) is not needed by SE (explicit substeps differentiate directly).
- port_kokkos M12 (`~/port_kokkos/src/fesom_ssh_se.cpp` @ 29e443b, `docs/SSH_SE_M12.md`,
  `docs/report/M12_SERGEY_PACKET.md`, plan `docs/plans/20260813-m12-split-explicit.md`): R = sum_k h_k uv_rhs_k / tau
  after the full predictor; r-term removes the n-level eta gradient with H_e = sum_k helem_k (exact) and the Coriolis
  with a reconstructed AB2 sum (D4 there); AB3 U in the eta equation, AM4 eta in the U equation; no filter; both rings
  carried across steps (flat at cold start); trim with h^n weights; zstar tail reused incrementally (their D2); two
  2-D exchanges per substep; gather-CSR sums, no atomics; cold start only (no restart of the rings); invariants
  eta-compatibility <= 2e-14 m, trim <= 3.4e-13 m2/s, layer sum 1.5e-11 m over 1000 CORE2 steps.
- FESOM_F SE (`/work/ab0995/a270088/fesom3_se`, branch `split-explicit` fb92d87, `docs/se_question_packet.md`):
  Sergey's answers of 2026-10-18 (section 6): eta^M for the ice, no averaging filter, barotropic drag in the substeps,
  chi by experiment (his setups 0.2); measured: stored-sum Coriolis + drag is stable at chi 0.1-0.25, the
  first-substep Coriolis consistency blows up at every chi; rings in the checkpoint, exact resume; CFL guard.
- fesom3-jax (`~/fesom3_jax/fesom3_jax/oce/ssh_se.py`, `shard/se_wide.py`, `ops/se_kernel.py`, `ops/remat.py`):
  the JAX design to copy (not import): SE rings in the State and the checkpoint, the CFL guard (`ssh_se.py:462-552`),
  wide halo with K+1 rings (an element reads its edge neighbours through the viscosity; K 10: -6 %), inner substeps
  unrolled (`lax.scan(unroll=length)`), Pallas pair (+1-2 %), gradient tests; 4 GH200 SE vs gathered CG -11 % (DP,
  dt 1800); SE slower than CG on one GPU.
- fesom-jax wide-halo machinery: `fesom_jax/ice_wide.py` (mEVP rings R_K/E_K, lane order, maps, compact refresh);
  SE needs one ring more than the mEVP.
- No SE in fesom-jax, the C port (`~/port2/fesom2_port_zstar/src/fesom_momentum.h:25`: deferred) or any FESOM2 run.
  `momadv_opt` = 2 in fesom-jax (`momentum.py:5,103`), as FESOM2 requires.

## Development Approach
- Branch `perf/core2-se` from `perf/core2-combo` (after the FCT-fix merge), worktree
  `/work/ab0995/a270088/fesom_jax_dev/wt_se`; GPU jobs only from frozen `wt_se_<sha>` snapshots.
- Gate-first: each task's tests are written with it; CPU tests with 4 fake devices on compute nodes, never on the
  login node. SE off ⇒ every existing program byte-identical (`scripts/perf/dp_hlo_hash.py`, the existing gates).
- Every deviation from the oracle is a named switch, off in the oracle-gate configuration.
- Update this plan when scope changes; ➕ for new tasks, ⚠️ for blockers.

## Implementation Steps

### Task 1: oracle data (needs D1, D2)
➕ Done 2026-10-22: Nikolay ran se0 (job 28333062, `runs/se_oracle/README.md`); se0 SE-off is bitwise z2_cdump at
step 1 (all 12 tags; z2_cdump's steps 2-3 predate the C's Z3-Z6); fesom-jax SE vs se0 SE: step 1 rel 7e-7 (the CG's
step-1 floor), steps 2-3 6.3 / 6.8 mm against the CG's own 7.6 / 7.7 mm from the C, the SE - SI increment to 0.6-0.7 %
(job 28333516, test_ssh_se.py).
**Files:** Create `fesom_jax/tests/data/se_m12/` (dumps, or a pointer file if too large -> /work); Modify
`docs/oracle_data.md` if present, else this plan.
- [x] existing se0 dumps (`port2/m12/t8_*/sedump.r*.bin`, read only): `FESOM_SE_DUMP=<prefix>` writes per rank, every
  step OVERWRITING, `int step, eta^{n+1}[myDim_nod2D], Ubt^{n+1}[2*myDim_elem2D]` in local lane order (29e443b
  `fesom_ssh_se.cpp:1103-1121`); all four are step 10 of a 10-step CORE2 run on 8 ranks (`dist_8`, dt 1800, KPP+GM+EVP,
  zstar, 1958). Not usable as an oracle: one step (10) only, after 10 coupled steps on a partition with no fesom-jax
  IC (the C IC is partition-dependent; fesom-jax has dist_16 and dist_864 ICs), and no eta per step.
- [x] D2 (user, 2026-10-21): this session may run the frozen serial se0. The runs needed (prepared, NOT run — ⚠️ the
  auto-mode classifier denied writing the job script, see below): the z2_cdump configuration = CORE2 private mesh
  `port2/mesh/core2` (2026-05-28 levels = fesom-jax's export, verified identical), 16 ranks `dist_16` (identical to
  /pool's), dt 1800, PHC winter, JRA55 1958, default env (KPP+GM+EVP), `FESOM_ALE=zstar`,
  `FESOM_ALE_DUMP_DIR=<run>/dump FESOM_ALE_DUMP_STEPS=3`, 3 steps: `off3` (`FESOM_SSH_MODE=si`) and `se3`
  (`FESOM_SSH_MODE=se FESOM_SE_M=50 FESOM_SE_CHI=0.1 FESOM_SE_VISC=1`, gammas 10/2750, `FESOM_SE_DUMP=<run>/sedump`,
  snapshots every step for uv). The ALE dump rail fires in SE mode too (`hbar` tag = hbar, hbar_old, ssh_rhs_old,
  eta_n per step, gid-keyed text), so no new reader is needed. Cost < 0.1 node-h.
- [ ] check that port_kokkos SE-off equals the C oracle (`test_se0_ssh_off_matches_the_c_oracle`: every tag of
  steps 1-3 bitwise vs z2_cdump; skips until `off3` exists); ⚠️ if not, gate SE as (SE-on - SE-off) increments
- [x] reader + gate: `io_dump.load_ale_dump` reads the tags as they are; `test_se_core2_steps123_vs_se0` (eta per step
  vs `se3`, skips until it exists). Expected class: near-bitwise, bounded by the JZ.7 predictor agreement (forcing
  round-off), not by the CG tolerance (SE has none)

### Task 2: dense SE step (CPU, bitwise against port_kokkos)
**Files:** Create `fesom_jax/ssh_se.py`, `fesom_jax/tests/test_ssh_se.py`; Modify `fesom_jax/step.py`,
`fesom_jax/state.py`, `fesom_jax/run_config.py`.
- [x] `SeConfig` (M, chi, `visc` two_term/linear/off, gammas, `visc_ah`, `bt_drag`, `m_force`) riding on the step as
  `AleConfig.split_explicit` (the `use_wsplit` precedent: both FESOM2 `dynamics_general` switches, so every runner
  carries it unchanged); YAML `ssh: {split_explicit: {...}}` (`RunConfig.ale_cfg()`), zstar required (2078039, ad7ee2c)
- [x] forcing / substeps / trim / eta_n = hbar = eta^M as the Kokkos kernels (operation order kept; the divergence as
  the Kokkos gather CSR, rows in element-id order, pre-divided by areasvol; viscosity neighbours in edge-id order);
  ➕ D3 linear viscosity (FESOM_F `se_kappa_build`) and `bt_drag` as switches; Mesh tables optional (`se_*`)
- [x] State `se_eta (nod,2)`, `se_ubt (elem,3,2)`, `se_sbar_old (elem,2)`, None on the CG path (`SE_FIELDS`)
- [x] tests (job 28310231 + 28312092): block algebra on random inputs, every switch (eta-compat <= 1e-12 of the
  change, trim <= 1e-13); 100 pi steps: eta-compat 3.0e-16 m, layer sum 3.7e-12 m, volume drift rel 4.7e-19; rest
  stays at rest; SE vs CG eta within a few % on pi; SP block float32. SE off ⇒ DP programs byte-identical to the
  branch base 4395978 (`dp_hlo_hash.py`, jobs 28311862/28311865; ⚠️ the stored `dp_hlo_hashes.json` is stale on the
  base itself: the base fails `test_dp_lock` too). Oracle gate: skips (Task 1)
- [x] gradients: dot test (rel 1e-12), FD probe through 3 SE steps (rel 1e-6), finite cold-start gradient (drag on)

### Task 3: restart of the rings and the CG/SE guard
**Files:** Modify `fesom_jax/state.py`, `fesom_jax/zarr_output.py` (restart), `scripts/run_from_config.py`;
Modify `fesom_jax/tests/test_ssh_se.py`.
- [x] rings + last-step sum optional restart fields (f64 on disk); exact resume: CORE2 KPP+mEVP+zstar, 26 steps in
  chunks of 13 == 13 -> restart -> 13 BITWISE on every owned lane of every field (job 28310231)
- [x] `ssh_se.init_state` in `run_from_config`: a CG restart into SE starts from flat rings and says so; an SE
  restart into a CG run is refused; a cold start gets flat rings silently
- [x] tests: ring round trip through write/read_restart, the guards (pi and through the driver on CORE2)

### Task 4: CFL guard
**Files:** Modify `fesom_jax/ssh_se.py`, tests.
- [x] omega_max by 100 power iterations on the substeps' own operator (host numpy, once per run in run_from_config,
  for every dt of the run), M_min = ceil(tau / (0.9 a_crit(chi) / omega_max)), raises below unless `m_force`
- [x] CORE2 (job 28310231): omega_max = 3.1518e-2 1/s (= FESOM_F's 3.15e-2), a_crit 1.5788 (chi 0.1) / 1.3988 (0.2):
  M_min dt 1800: 40 (chi 0.1) / 46 (0.2); dt 2700: 60 / 68. M 50 is below M_min at dt 2700

### Task 5: sharded SE (per-substep exchanges, then the wide halo)
**Files:** Create `fesom_jax/se_wide.py`, `fesom_jax/tests/test_se_sharded.py`; Modify `fesom_jax/step.py`,
`fesom_jax/halo.py` / `integrate_sharded.py` plumbing as `ice_wide.py` did.
- [x] per-substep exchanges (+ one H0_e exchange per step): CORE2 dist_4, 4 fake devices, BITWISE the dense block on
  every owned lane, two_term / drag / linear (job 28311585). ⚠️ the dense reference must take the mesh as a jit
  ARGUMENT: closed over, XLA:CPU compiles the column sums 2.8e-14 off
- [x] wide halo `se_wide.py`: K+1 rings, compact all_gather at entry / per K substeps / exit, BITWISE the per-substep
  run on every valid lane for K = 2, 3, 5, 7, 10, 12, 50 (no 1-substep block: XLA inlines a trip-count-1 scan),
  negative control, gradient rel < 1e-12; through `run_steps_sharded_forced` (3 forced CORE2 steps) bitwise
- [x] `gpu.se_wide: K` (`FESOM_SE_WIDE`, off by default; needs `ssh.split_explicit`)

### Task 6: GPU speed
**Files:** Modify `fesom_jax/ssh_se.py`, `configs/core2_gpu_fast.yaml` (only after STOP below), ledger.
- [x] inner substeps unrolled inside a wide-halo window (`lax.scan(unroll=K)`), rings renamed not copied —
  `gpu.se_unroll` (edc2c71), round-off
- [x] A/B on one dolpung node: CG vs SE vs SE + wide K in {5, 10}, the 817-SYPD stack, steady + production loop —
  K 10/20/30 (jobs 28313318, 28314647, 28315580): CG 9.07 ms/step, SE w10 u10 8.4-8.7 (855-884 SYPD); production
  loop not yet measured
- [x] nsys profile of the SE stack; Pallas pair only if the substeps stay > ~10 % of the step — job 28314373: SE
  kernels 0.69 ms/step (8.9 % of the kernels) ⇒ no Pallas; the step is host-bound at the launch boundaries
  (~1 ms/step idle), ledger "Split-explicit SSH: speed"
- ➕ [x] `gpu.step_unroll` (combo 4e4a512): two steps per loop trip make both stacks slower (job 28315580); stays off

### Task 7: stability screens at dt 2700 (needs D4)
- [x] from the v3 1990-01-01 restart and from the 1958 cold start: chi 0.1 / 0.2, with and without `bt_drag`, M from
  the guard and 1.5x; 3000 steps each, max|eta|, the Antarctic/Bellingshausen shelf noise growth rate (FESOM_F's
  diagnostic), w CFL; SP and DP — SP done (job 28315629, ledger "stability screens"): without the drag dt 2700 blows
  up in 3 of 4 settings within a month, with the drag all hold; the cold start at dt 1800 holds in all 8. The shelf
  noise metric was not computed (max|eta|, max|uv|, cfl_z per day instead); DP not run
- [x] ⛔ STOP: table to Nikolay, he picks (chi, drag, M) — user 2026-10-21: chi 0.1 + bt_drag + M 60

### Task 8: twin and acceptance
- [x] 3 years 1990-1992 at dt 2700, SE vs CG, same stack (SP + ice-2): `scripts/perf/compare_monthly.py` (SSH rms and
  max, SST, ice area/volume, volume-mean T/S, budgets) against the CG-vs-CG spread of the existing twins — run as 10
  years 1990-1999 (job 28318591, 25 min): within the SP - DP spread after a few months, all finite, SE conserves the
  volume better than the CG (ledger "the 10-year twin")
- [ ] ⛔ STOP: Nikolay decides whether SE enters `core2_gpu_fast.yaml`

### Task 9: docs
- [ ] ledger rows, `docs/PORTING_LESSONS.md` (one lesson per task), SP policy table row (D5), this plan to
  `completed/`

## Technical Details
- Substep body (port_kokkos M12 / Sergey's notes): eta^{m+1} = eta^m - dtbt div(U^{AB3}); U^{m+1} = U^m + dtbt (F
  - g H grad eta^{AM4} + f x U^{AB3} + visc); weights from chi with beta 0.281105; <<U>> = sum U^{AB3} / M; with
  `bt_drag`: - (C_d |u_b| / H^n)(U^m - U^n) per substep, the 3-D drag left in F (FESOM_F Task 5.4).
- Collectives: per-substep 2M per step (120 at M 60); wide K 10: ~2M/K + entry/exit.
- Memory: rings are 2-D (nod x 2, elem x 3 x 2), small; `jax.checkpoint` on the substep window for AD.

## Costs (estimates)
- Implementation: one executing agent at high effort, ~1-1.5 days of work, ~3-5 M tokens.
- Compute: oracle runs < 1 CPU node-h; CPU tests ~10 node-h; GPU ~20-25 GPU-h (A/Bs ~3, screens ~8-10, the 3-year
  twin ~6, profiles ~1); dolpung at most two nodes with the fesom3 jobs counted.

## Post-Completion
- Sergey: the deviations list (as port_kokkos's packet) if SE goes into production.
- Upstream FESOM2: nothing (opt-in option of the port).
