# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. **Phases 0–4 COMPLETE (GATEs 0/1/2/3/4).
Phase 5 (CORE2 single-device) IN PROGRESS — sub-plan created; Tasks 5.1–5.6 DONE;
Task 5.7 (matched C dump run + CORE2 stability, 1-day + multi-day) is NEXT.** Full suite
**370 passing** (pi 313 + CORE2 additions: `test_surface_bc.py` +7, `test_core2_step.py` +5,
and the earlier 5.1–5.5 tests).

The pi model is fully ported, dump-gated, jitted, differentiable end-to-end, 1000-step
stable. Phase 5 runs that same physics (PP + **linfs** + FCT + opt_visc7, **no GM/KPP/ice**)
on the **CORE2 mesh** with PHC initial conditions and real JRA55+SSS+runoff forcing,
verified per-substep against a CORE2 C-port dump (**Path A**). DONE: CORE2 mesh exported +
loads zero-code (CW-orientation guard); **PHC IC** ported (numpy, ~1e-14); **JRA55-do
reader** ported (numpy, **bit-exact**); **L&Y09 bulk** ported (AD-safe, ~1e-13);
**SSS-restoring + runoff** ported (numpy + AD-safe); and **Task 5.6 — the assembled CORE2
step**: surface BCs (`bc_T`/`bc_S`/`sw_3d`) + the bulk/SSS/shortwave/ice-mask forcing wired
into `step`/`integrate`, verified vs a new per-substep CORE2 C dump at **step-1 post-step T
7.1e-15 / S 2.1e-14 (bit-exact)**, steps 2-3 ~1e-9. **The full single-device CORE2 model now
runs and matches the C per-substep.** Task 5.7 = generate the matched multi-step dump + run
1-day/multi-day for stability; Task 5.8 = GATE 5 (gradient on a CORE2 slice).

**JAX committed on `main`** through Task 5.6 (`7ca598f` Task 5.6, `d906750` 5.5, `c1aa677`
5.4, `5ab28af` 5.3, `d4fcdb2` 5.2, `4b34f6a` 5.1). C-side dump additions on the port2
`jax-mesh-export` branch through Task 5.6 (`cfb4b5e` Task 5.6 — `fesom_dump.c` env probes,
`fesom_step.c` surf dump, `fesom_bulk.c` `FESOM_BULK_FIXED_ITERS` in `fesom_bulk_compute`;
`78b1df1` 5.5) — ⚠️ **keep all C commits on that branch, never on port2 main**
(user); port2 is otherwise the user's — no housekeeping. **CORE2 data artifacts live on
`/work`** — `port_jax/data` is a **symlink** to `/work/ab0995/a270088/port_jax/data`
(gitignored; user rule "large files on /work not /home"). New C job scripts live **untracked**
in `port2/fesom2_port/jobs/` (`jax_*_core2.sh`, incl. `jax_step_dump_core2.sh`). The C dump
hooks (`dump_jra_fields`, `fesom_bulk_dump`, `fesom_sss_runoff_dump`, the per-substep
`fesom_dump.c` + the new surface-forcing records in `fesom_step.c`) are dump-only, env-gated.
⚠️ **Cheap C jobs: use `-p compute --time=00:30:00` (fast debug QOS), not `-p shared`** (~16 s
to start vs minutes).

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean model for
hybrid ML (trainable NN parameterizations for vertical mixing + mesoscale eddy fluxes,
trained end-to-end). Multi-session effort. Work from `/home/a/a270088/port_jax`. Max effort.

## START HERE, in order
1. **Phase-5 sub-plan (source of truth for Phase 5):**
   `docs/plans/20260606-fesom-jax-core2.md` — the scope correction (§0), Path A, the task
   ladder 5.1–5.8 with per-task gates, the 5 module research briefs, risks. **Tasks 5.1–5.6
   are ticked DONE; 5.7 is next.**
2. **Main plan:** `docs/plans/20260605-fesom-jax-port.md` — decisions, the verification
   ladder, the Revision Log. Phase 5 there is the outline; the sub-plan supersedes it.
3. **Lessons (every session):** `docs/PORTING_LESSONS.md` — esp. the **Phase 5** entries:
   the linfs-not-zlevel scope rule, the **pi↔CORE2 orientation trap**, the netCDF4 env note,
   the PHC sequential-GS extrap, the JRA cancellation/bit-exact-gather trap, the bulk
   fixed-5≠early-break finding, the SSS **Jacobi-fill-vectorizes** + **÷ocean_area crushes the
   global-mean** lessons, the "no invented modeling choice" rule, and the **Task 5.6** entries:
   the **static-ice-mask** ("no ice" ≠ ice-free), the **`T_old`-is-base** trap, the
   **`fesom_bulk_compute` fixed-iters** flag, and the **runoff-inert-in-no-ice** finding +
   the Phase-6 runoff activation spec. **STANDING RULE: append a lesson per task.**
4. **Project memory:** `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.
5. For 5.7: the **assembled CORE2 model is `fesom_jax/core2_forcing.py`** (the per-step driver:
   `build_core_forcing(mesh, year, sst_ic=…)` → `CoreForcing`; `compute_surface_fluxes`) +
   `step(..., step_forcing, forcing_static)` / `integrate(..., step_forcings, forcing_static)`.
   Build the IC with `phc_ic.core2_initial_state` (⚠️ **`T_old`=const base 10/35, not PHC**).
   `dates_for_steps(1958, 500, N)` gives the per-step `(year,day,sec,month)`. The step-1 gate
   lives in `tests/test_core2_step.py`; the dump is `data/step_dump_core2/core2_cdump.00000`
   (regen via `port2/jobs/jax_step_dump_core2.sh`). For 5.7: run **JITTED** `run`/`integrate`
   (eager CORE2 step ~32 s) for 1-day (~172 steps) + multi-day; watch the Aleutian Trench
   (node 94122). The C `fesom_dump.c` now reads `FESOM_DUMP_PROBES` (re-pin per run) and dumps
   the surface forcing at `DUMP_SUB_INIT`.

## STATUS
- **Phases 0–4 (GATEs 0–4):** full single-step pi model (`step.py`, substeps 1–16),
  checkpointed `lax.scan` (`integrate.py`), CG `custom_linear_solve`, FCT + opt_visc7,
  pi 1000-step stable, `test_gradient.py` plateau 5.70e-7. All committed.
- **Phase 5 — scope (user-confirmed, do NOT re-litigate):** the C port (`fesom2_port`) is a
  **deliberately simplified** FESOM — **linfs-only, full-cell, no cavities**. So Phase 5 =
  pi physics on the CORE2 mesh + PHC IC + JRA55/SSS/runoff. **NOT zlevel/zstar/partial-cells/
  w_i** (absent from the C port). **zstar is future work** — keep a `which_ale` design seam,
  don't port it now. Reference = **Path A** (per-substep CORE2 C dump at the matched config:
  `FESOM_MIX_SCHEME=PP FESOM_NO_GMREDI=1 FESOM_NO_ICE_*=1`, dt=500).
- **Task 5.1 DONE:** CORE2 mesh exported (`data/mesh_core2/`, job 25386129) — `load_mesh`
  works **zero-code**. **Orientation guard added** (`mesh.check_cw_orientation`): the C
  `orient_cw` swapped 244654/244659 CORE2 elements CCW→CW; the guard raises on any non-CW mesh.
  `test_mesh_core2.py` (12) + `test_step_core2.py` rest-state (max|uv|=1.8e-14).
- **Task 5.2 DONE:** `fesom_jax/phc_ic.py` — faithful numpy port of `fesom_phc.c` (bilinear +
  **sequential-GS** `extrap_nod3D` + `ptheta`); matches the C surface dump to **~1e-14**. Cache
  `data/ic_core2/{T,S}_ic.npy`; `core2_initial_state` builds the State. `test_phc_ic.py` (5).
- **Task 5.3 DONE:** `fesom_jax/jra55.py` — faithful numpy port of `fesom_jra55.c` (julday +
  mid-interval-shift time grid + shared bilinear stencil + wind g2r rotation). Verified vs a C
  all-node dump (job 25388630): **6 scalar fields bit-exact**, wind ~3.5e-15. `test_jra55.py` (5).
  ⚠️ **#1 trap:** the time-interp cancels two ~2.4e6 Julian-day numbers → a folded-weight gather's
  ~1e-13 reassociation blew to ~6e-8; fixed by a **bit-identical `(s·dx)·dy` divide-at-end gather**.
- **Task 5.4 DONE:** `fesom_jax/forcing.py` — AD-safe port of `fesom_bulk.c`
  (`ncar_ocean_fluxes_mode` fixed-5 unrolled + `obudget` + `bulk_surface_fluxes`, node→elem
  mean-of-3). Verified vs a C `bulk_dump_*` dump (job 25389451): **cd/ce/ch ~1e-17, heat_flux
  ~6e-13, stress ~5e-16**. `test_forcing.py` (10). ⚠️ **fixed-5 ≠ early-break at calm nodes**
  (`ch` up to 88%, but heat_flux ≤7 W/m² at ~4 nodes); JAX runs fixed-5 vs a **fixed-5** C dump
  (`FESOM_BULK_FIXED_ITERS=1`). `USE_SW_PENE=1` ⇒ shortwave penetration deferred to **5.6**
  (`heat_flux = qns−qsr`, pre-penetration).
- **Task 5.5 DONE:** `fesom_jax/sss_runoff.py` — numpy readers (`interp_2d_field` **lat-clamp /
  lon-cyclic-wrap** bilinear + the 30-cell **Jacobi** missing-fill via
  `scipy.ndimage.uniform_filter`) → `Ssurf_clim[12,nod2D]` + `runoff_node[nod2D]` (/1000); +
  AD-safe `sss_runoff_fluxes` (`virtual_salt=S_top·water_flux`, `relax_salt=
  surf_relax_S·(Ssurf−S_top)`, each minus its **area-weighted global mean**; then
  `water_flux += ⟨water_flux+runoff⟩`). Verified vs a new C `sss_dump_*` all-node dump (job
  25390216, year 1958, `data/sss_dump_core2/`) at 2 months — m1 (Jan/day1) + m4 (Apr/day100
  **month crossing**): **runoff bit-exact** (max|Δ|=0), **Ssurf bit-exact at 105148/126858**
  ocean-bracket nodes (95p ~3.6e-14, max 2.8e-12 at ~35 coastal fill nodes),
  **virtual_salt/relax_salt/water_flux ~1e-20** (fed the C's own inputs, the ÷`ocean_area`=3.6e14
  crushes the global-mean reduction). Both AD seams flow. `test_sss_runoff.py` (9).
  `ref_sss_local=1`, `surf_relax_S=1.929e-6`; **no legacy month +1**. New C `fesom_sss_runoff_dump`
  + `jax_sss_dump_core2.sh`.
- **Task 5.6 DONE:** the **assembled CORE2 step**. `fesom_jax/core2_forcing.py` (per-step driver:
  bulk → sss_runoff → **ice stress blend** → `cal_shortwave_rad` → `bc_T`/`bc_S`/`sw_3d`/
  `stress_surf`); `tracer_diff` gained `bc_T`/`bc_S`/`sw_3d`; `forcing` gained `cal_shortwave_rad`;
  `step`/`integrate` thread `step_forcing`/`forcing_static` (pi path `None` ⇒ **bit-identical**).
  Verified vs a new per-substep CORE2 C dump (job 25391647, 7 probes incl. Aleutian 94122,
  `FESOM_BULK_FIXED_ITERS=1`): **step-1 post-step T 7.1e-15 / S 2.1e-14** (bit-exact — the
  comprehensive gate), surface forcing 1e-13..1e-22, dynamics density 2.3e-13 / uv 1e-10 /
  d_eta 2e-9, steps 2-3 T/S ~1e-9. `test_surface_bc.py` (7) + `test_core2_step.py` (5).
  **chl = Sweeney monthly** (C default; constant-0.1 seam kept). ⚠️ **THREE bugs found+fixed**
  (full detail in PORTING_LESSONS.md): (1) the C "no-ice" run keeps a **static `a_ice=0.9` mask**
  (IC SST<0) gating cal_shortwave penetration + the momentum stress blend — **user decision:
  match it** (`ice_ic_aice`); (2) step-1 `T_old`=const base 10/35, NOT PHC; (3) `fesom_bulk_compute`
  didn't honor `FESOM_BULK_FIXED_ITERS`. **Finding: runoff is inert in linfs** (the balanced
  `water_flux` feeds only non-linfs paths).

## IMMEDIATE WORK — Task 5.7 onward (the sub-plan has the full ladder + gates)
**5.6 is DONE** (assembled step + surface forcing, step-1 bit-exact vs the C dump). Remaining:
- **5.7 matched C dump run + CORE2 stability:**
  - **Per-substep dump already exists** (`data/step_dump_core2/core2_cdump.00000`, 3 steps,
    7 probes incl. Aleutian 94122). For more steps / more probes: edit `FESOM_DUMP_PROBES` +
    `FESOM_DUMP_MAXSTEPS` + `NSTEPS` in `port2/jobs/jax_step_dump_core2.sh` and resubmit
    (`-p compute --time=00:30:00`). Extend `test_core2_step.py`'s per-substep dynamics gates
    (uv/d_eta/w with calibrated tolerances — d_eta ~2e-9 since CORE2 CG takes ~43 iters).
  - **Assemble + run CORE2 1-day + multi-day:** `integrate`/`run` with stacked `step_forcings`
    (`CoreForcing.stack(dates_for_steps(1958, 500, N))`) + `forcing_static`. ⚠️ **Use the
    JITTED `run`/`integrate`** — eager CORE2 step ~32 s/step. 1-day = ~172 steps; assert stable
    per FRESH_START §15 (no NaN, SST∈[−2,35], |SSH|<5 m, max|vel|<3 m/s). **Watch the Aleutian
    Trench (node 94122).** GPU sbatch if CPU is too slow.
  - ⚠️ **Stability risk:** PHC IC + JRA55 + the static ice mask, no ice physics, dt=500. The
    matched C run is the arbiter — if the C itself blows up, that's a finding (the static
    `a_ice` mask damps high-lat wind stress, so it's somewhat protective; see the 5.6 lessons).
  - **Gate:** step-1 dump-tight (done); 1-day + multi-day stable. **Lesson:** append.
- **5.8 GATE 5 — gradient on a CORE2 slice:** re-run `test_gradient.py` on a small CORE2 slice
  (`d(SST)/d(k_ver)` AD↔FD plateau through the CG; `d(loss)/d(T₀)` finite incl. masked lanes) +
  the **NEW** Phase-5 feedbacks `d(heat_flux)/d(SST)` and `d(stress)/d(current)` (the bulk +
  the ice-ocean drag seams — AD-safe by construction, see the 5.6 AD lesson). Confirm the
  checkpointed backward fits the A100 at CORE2 scale (drop N or O(√N) nesting; pi N=200 was
  4.23 GB). GPU sbatch. **Lesson:** append.

**First moves for 5.7:** (1) decide dump scope (more steps/probes? edit the job + resubmit).
(2) Build the multi-step forcing: `cf = build_core_forcing(mesh, 1958, sst_ic=state.T[:,0])`,
`sfs = cf.stack(dates_for_steps(1958, 500, N))`, then `integrate(state, mesh, op, None, N,
step_forcings=sfs, forcing_static=cf.static)` — but **jit it** (wrap in `integrate_jit` /
`run`). (3) 1-day stability assertions + the Aleutian watch. (4) Extend `test_core2_step.py`
with the per-substep dynamics gates (calibrated). The pattern is proven; 5.7 is mostly
runtime/stability + tolerance calibration, not new physics.

## THE PROVEN VERIFICATION RECIPE (still applies)
Per-substep dump at pinned probes (or all-node), truncate to `nlevels`, `verify.assert_close(col,
rec, kind=…)` (`map`/`gather` 1e-15, `scatter`/`reduction` 1e-12; calibrate `atol`). When a
kernel is fed the C's own upstream values (the bulk's `T_oc`, the SSS dump's `S_top`/`water_flux`),
the gate isolates that kernel → MAP-class. AD: any divide/sqrt whose denominator/arg can vanish in
a masked lane must compute a FINITE value (`where(d==0,1,d)` / double-`where` safe-sqrt) — a
forward `where` does NOT stop a 0·inf NaN backward. Re-run `test_gradient.py` at GATE 5. New
Phase-5 invariants: **mesh CW-orientation is guarded at load**; **the C port is the spec — port it
1:1 and verify by dump, don't invent a modeling choice.**

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`
- **Env python (ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q` (~8 min; CORE2 tests skip if their
  `data/` artifacts are absent). **netCDF4 + scipy installed.** GPU via SLURM (`-A ab0995_gpu
  -p gpu`/`gpu-devel`).
- CORE2 mesh: `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`; PHC IC:
  `/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc`; JRA55:
  `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/`; SSS `…/PHC2_salx.nc`, runoff
  `…/CORE2_runoff.nc`; **chl (5.6): `/pool/data/AWICM/FESOM2/FORCING/Sweeney/Sweeney_2005.nc`**.
- CORE2 data (gitignored): **`data/` is a symlink → `/work/ab0995/a270088/port_jax/data`**.
  Holds `mesh_core2/`, `ic_core2/`, `phc_dump_core2/`, `jra_dump_core2/`, `bulk_dump_core2/`,
  `sss_dump_core2/`. C-side jobs (untracked in port2): `port2/fesom2_port/jobs/jax_*.sh`.
- C port (algorithmic SoT): `/home/a/a270088/port2/fesom2_port/src/`; built binary
  `…/build/fesom_port`. C run-arg order: `<mesh> <out> <dt> <nsteps> <snap> <phc> <jra>`.
  Build: `bash -lc 'source env.sh && make -C build fesom_port'` (incremental, ~30 s).
- I (Claude) drive the C SLURM jobs (`sbatch`, acct ab0995): use **`-p compute --time=00:30:00`**
  (fast debug QOS, ~16 s to start) — NOT `-p shared`. Dumps run ~20–25 s. See memory
  [[hpc-job-file-conventions]].

## LOCKED DECISIONS (do NOT re-litigate)
1. Use case = hybrid ML params (swap points: vertical mixing PP/KPP, eddy flux GM/Redi);
   seam = `fesom_jax/params.py`. 2. Full-fidelity, bottom-up, match the C port 1:1. 3. AD-safe
   by construction + gradient re-run at every gate. 4. Mesh = index gather/scatter over
   `ops.py`. 5. Single-device + data-parallel now; mesh sharding Phase 8. 6. **Phase 5 = linfs
   on CORE2, Path A; zstar/partial-cells/GM/KPP/ice-model are later.** 7. **netCDF4 + scipy in
   the env.** 8. **(2026-06-06, user) Phase 5 keeps the C's static `a_ice` mask** (=0.9 where
   IC SST<0; gates shortwave penetration + the momentum stress blend) — i.e. **match the C's
   "no-ice" run, NOT a truly ice-free ocean.** chl = **Sweeney monthly** (the C default).

## CRITICAL GOTCHAS (full list in PORTING_LESSONS.md)
- **The C port is linfs-only / full-cell / no-cavity — match THAT, not real-FESOM.** zlevel
  lives only in the Fortran; the parent outline's zlevel/partial-cells/w_i are out of scope.
- **⚠️ "No ice" ≠ ice-free:** the C keeps a **static `a_ice=0.9` mask** (IC SST<0, 37089 nodes)
  that (a) skips shortwave penetration and (b) blends the wind stress (`ice_drag·a+atm·(1−a)`,
  `u_ice=0`). Replicated in `core2_forcing` (`ice_ic_aice`). Missing it = 122 W/m² heat_flux
  error at Antarctic nodes.
- **⚠️ CORE2 step-1 `T_old`/`S_old` = const base 10/35, NOT PHC** (the C sanity-advect saves
  `valuesold=base` before PHC overwrites `values`). `core2_initial_state` handles it; don't
  "fix" it to PHC.
- **⚠️ `FESOM_BULK_FIXED_ITERS` must be honored by `fesom_bulk_compute`** (not just the dump fn)
  — now wired. The per-substep reference run sets it; JAX runs fixed-5.
- **pi↔CORE2 orientation:** CORE2 raw mesh is ~all CCW; `load_mesh` asserts CW (Aleutian guard).
- **AD masked-NaN rule (bit us 4×):** make masked lanes finite; a forward `where` doesn't stop
  a backward 0·inf. (`cal_shortwave_rad`'s `sw_3d` is a constant ⇒ no AD path; the bc_T/bc_S +
  ice-ocean drag ARE differentiable seams, all safe-sqrt-guarded.)
- **Eager CORE2 `step()` ≈ 32 s/step on CPU** → use jitted `run`/`integrate` or GPU for 5.7.
- **Runoff is INERT in the no-ice Phase 5 — by the C's design, not a bug** (the C routes
  runoff through ice thermo → `flx_fw` → `water_flux` → `virtual_salt`; ice off ⇒ that door is
  shut; the balanced `water_flux` feeds only non-linfs paths). It works fully in the ice-on run
  + the reader/balance are done. **User decision: keep matching the no-ice C run**; runoff
  activates for free in Phase 6 (the seam is pure-in-`water_flux`). **Full spec: sub-plan
  "Runoff handoff to Phase 6"** (READ before touching runoff). Phase-5 SST/SSS forcing = bc_T
  (heat) + bc_S (virtual_salt=rsss·(evap−prec) + relax_salt).
- **netCDF4 import prints a benign `ndarray size changed` ABI warning** — harmless.
- **CORE2 stability risk** (dt=500, PHC+JRA, no ice physics): the matched C run is the arbiter
  (Task 5.7). The static `a_ice` mask damps high-lat wind stress (somewhat protective). Watch
  the Aleutian Trench (global elem 194724; vertex node gid 94122).
- **config = the pi reference physics on CORE2:** linfs, PP, FCT, opt_visc=7, `use_wsplit=0`,
  CG SSH (α=1), dt=500, PHC IC, JRA55+SSS+runoff + the static ice mask.

## WORKFLOW NOTES
- The sub-plan is authoritative for Phase 5; tick `[x]`, keep its Revision Log + the lessons
  current. **Commit only when asked** (per-task commits on `main`). **C edits → port2 branch
  `jax-mesh-export`, NEVER port2 main** (user); job scripts kept untracked there; otherwise leave
  the port2 repo to the user (no housekeeping). **Large generated files → `/work`** (the `data`
  symlink handles it). Cheap C jobs → `-p compute --time=00:30:00`. All python/pytest via the env
  python. See memory [[hpc-job-file-conventions]].

Confirm you've absorbed this, tell me which task you're starting (5.7), then proceed.
