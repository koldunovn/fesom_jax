# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. **Phases 0–4 COMPLETE (GATEs 0/1/2/3/4).
Phase 5 (CORE2 single-device) IN PROGRESS — sub-plan created; Tasks 5.1 + 5.2 DONE.**
Full suite **334 passing**.

The pi model is fully ported, dump-gated, jitted, differentiable end-to-end, 1000-step
stable. Phase 5 runs that same physics (PP + **linfs** + FCT + opt_visc7, **no GM/KPP/ice**)
on the **CORE2 mesh** with PHC initial conditions and real JRA55+SSS+runoff forcing,
verified per-substep against a CORE2 C-port dump (**Path A**). So far: the CORE2 mesh is
exported + loads zero-code (with a new **clockwise-orientation guard**), and the **PHC IC**
is ported (numpy) and matches the C to ~1e-14.

**Everything committed on `main`** (`d4fcdb2` Task 5.2, `4b34f6a` Task 5.1). CORE2 data
artifacts are **gitignored** (`data/mesh_core2/`, `data/ic_core2/`, `data/phc_dump_core2/`).
New C-side job scripts live **untracked** in `port2/fesom2_port/jobs/`
(`jax_mesh_export_core2.sh`, `jax_phc_dump_core2.sh`).

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean model for
hybrid ML (trainable NN parameterizations for vertical mixing + mesoscale eddy fluxes,
trained end-to-end). Multi-session effort. Work from `/home/a/a270088/port_jax`. Max effort.

## START HERE, in order
1. **Phase-5 sub-plan (source of truth for Phase 5):**
   `docs/plans/20260606-fesom-jax-core2.md` — the scope correction (§0), Path A, the task
   ladder 5.1–5.8 with per-task gates, the 5 module research briefs, risks. **Tasks 5.1 +
   5.2 are ticked DONE.**
2. **Main plan:** `docs/plans/20260605-fesom-jax-port.md` — decisions, the verification
   ladder, the Revision Log. Phase 5 there is the outline; the sub-plan supersedes it.
3. **Lessons (every session):** `docs/PORTING_LESSONS.md` — esp. the **Phase 5** entries:
   the linfs-not-zlevel scope rule, the **pi↔CORE2 orientation trap**, the netCDF4 env note,
   the PHC sequential-GS extrap, the "no invented modeling choice" rule. **STANDING RULE:
   append a lesson per task.**
4. **Project memory:** `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.
5. For 5.3–5.5: read `port2/fesom2_port/src/fesom_jra55.c`, `fesom_bulk.c` (5.3/5.4) and
   `fesom_sss_runoff.c` (5.5). The sub-plan's task bodies already hold detailed per-module
   briefs (functions, formulas, gotchas, AD-safe guards, suggested gate).

## STATUS
- **Phases 0–4 (GATEs 0–4):** full single-step pi model (`step.py`, substeps 1–16),
  checkpointed `lax.scan` (`integrate.py`), CG `custom_linear_solve`, FCT + opt_visc7,
  pi 1000-step stable, `test_gradient.py` plateau 5.70e-7. All committed.
- **Phase 5 — scope (user-confirmed, do NOT re-litigate):** the C port (`fesom2_port`) is a
  **deliberately simplified** FESOM — **linfs-only, full-cell, no cavities**. So Phase 5 =
  pi physics on the CORE2 mesh + PHC IC + JRA55/SSS/runoff. **NOT zlevel/zstar/partial-cells/
  w_i** (absent from the C port; the parent outline was wrong to list them). **zstar is future
  work** — keep a `which_ale` design seam, don't port it now. **SSS/runoff = faithful 1:1 C
  port** (no §9-shorthand alternative). Reference = **Path A** (per-substep CORE2 C dump at the
  matched config: `FESOM_MIX_SCHEME=PP FESOM_NO_GMREDI=1 FESOM_NO_ICE_*=1`, dt=500).
- **Task 5.1 DONE:** CORE2 mesh exported (`data/mesh_core2/`, job 25386129) — `load_mesh`
  works **zero-code** (full-cell ⇒ global zbar/Z valid; ragged masks handle per-node depth).
  **Orientation guard added** (`mesh.check_cw_orientation` runs in `load_mesh`): the C
  `orient_cw` swapped **244654/244659** CORE2 elements CCW→CW (export log) before geometry;
  the guard raises on any non-CW mesh (guards the historical Aleutian wrong-sign blow-up).
  `test_mesh_core2.py` (12) + `test_step_core2.py` rest-state (max|uv|=1.8e-14).
- **Task 5.2 DONE:** `fesom_jax/phc_ic.py` — faithful numpy port of `fesom_phc.c` (bilinear
  interp + **sequential-GS** `extrap_nod3D` + `ptheta`); matches the C surface dump to
  **~1e-14** (brackets exact). Cache `data/ic_core2/{T,S}_ic.npy`; `core2_initial_state`
  builds the State. `test_phc_ic.py` (5). **netCDF4 pip-installed** into the env.

## IMMEDIATE WORK — Task 5.3 onward (the sub-plan has the full ladder + gates)
- **5.3 JRA55 reader** (host numpy, netCDF4): files
  `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/{var}.{YEAR}.nc`, field order
  **uas,vas,huss,rsds,rlds,tas,prra,prsn** (`fesom_jra55.h:50-59`); cyclic-lon +2, lat-flip;
  bilinear stencil built ONCE on **geographic** coords; 3-hourly time-interp; **wind g2r
  rotation** (Euler 50/15/−90); units (Tair K→°C, prec /1000). Gate: dump the 8 jra fields vs
  C `fesom_jra55_step`.
- **5.4 L&Y09 bulk** (AD-safe JAX): `ncar_ocean_fluxes_mode` (**fixed 5 iters, unrolled** — drop
  the data-dependent break) + `obudget`; `heat_flux/water_flux/stress`; node→elem simple
  mean-of-3; albw=0.1; relative-wind in coeffs but absolute `ug` in obudget (preserve). The
  **differentiable SST→flux / current→stress feedback** lives here. Safe-sqrt / `where` guards.
- **5.5 SSS restoring + runoff** (numpy readers + AD-safe JAX): match `fesom_sss_runoff.c`
  exactly (virtual_salt + relax_salt into the S surface BC; runoff via the global-mean term in
  the no-ice path); `ref_sss_local=1`, `surf_relax_S=1.929e-6`; month index no legacy `+1`.
- **5.6 wire surface BCs** into `tracer_diff` (`bc_T=−dt·heat_flux/vcpw`,
  `bc_S=dt·(virtual_salt+relax_salt)`; currently `bc_surface=0`) + bulk `stress_surf`→momentum.
- **5.7 matched C dump + stability:** generate the CORE2 per-substep dump (re-pin
  `PROBE_GIDS` in `fesom_dump.c` incl. an Aleutian node, e.g. 94122); assemble + run CORE2
  1-day (~172 steps, dt=500) + multi-day; assert stable. ⚠️ **Use the JITTED `run`/`integrate`
  (or GPU)** — an eager CORE2 step is ~32 s on CPU. Finalize `T_old`/`S_old` (step-1 AB2)
  against the dump.
- **5.8 GATE 5:** gradient check on a CORE2 slice (+ the new d(heat_flux)/d(SST),
  d(stress)/d(current) feedbacks); confirm checkpointed backward fits the GPU at CORE2 scale.

**First moves for 5.3:** read `fesom_jra55.c`/`fesom_bulk.c`; stand up a JRA55 forcing-dump
job (clone `jax_phc_dump_core2.sh`, set a JRA55 year, dump `jra->{8 fields}` + bulk
heat/water/stress) to verify against.

## THE PROVEN VERIFICATION RECIPE (still applies)
Per-substep dump at pinned probes, truncate to `nlevels`, `verify.assert_close(col, rec,
kind=…)` (`map`/`gather` 1e-15, `scatter`/`reduction` 1e-12; calibrate `atol`). AD: any
divide/sqrt whose denominator/arg can vanish in a masked lane must compute a FINITE value
(`where(d==0,1,d)` / double-`where` safe-sqrt) — a forward `where` does NOT stop a 0·inf NaN
backward. Re-run `test_gradient.py` at GATE 5. New Phase-5 invariants: **mesh CW-orientation
is guarded at load**; **the C port is the spec — port it 1:1 and verify by dump, don't invent
a modeling choice** (FRESH_START is a description, not an alternative).

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`
- **Env python (ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q` (~7 min; CORE2 tests skip if their
  `data/` artifacts are absent). **netCDF4 is now installed** (Phase 5). GPU via SLURM
  (`-A ab0995_gpu -p gpu`/`gpu-devel`).
- CORE2 mesh: `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`; PHC IC:
  `/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc`; JRA55:
  `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/`; SSS `…/PHC2_salx.nc`, runoff
  `…/CORE2_runoff.nc`.
- Exported CORE2 mesh / IC / C dumps (gitignored): `data/mesh_core2/`, `data/ic_core2/`,
  `data/phc_dump_core2/`. C-side jobs (untracked in port2): `port2/fesom2_port/jobs/jax_*.sh`.
- C port (algorithmic SoT): `/home/a/a270088/port2/fesom2_port/src/`; built binary
  `…/build/fesom_port` (NL read from the mesh, handles pi+CORE2). C run-arg order:
  `<mesh> <out> <dt> <nsteps> <snap> <phc> <jra>` (phc=argv6 triggers PHC IC, t_insitu=1).
- I (Claude) drive the C SLURM jobs (`sbatch`, acct ab0995, `shared`); they're cheap (mesh
  export 17 s, PHC dump 16 s).

## LOCKED DECISIONS (do NOT re-litigate)
1. Use case = hybrid ML params (swap points: vertical mixing PP/KPP, eddy flux GM/Redi);
   seam = `fesom_jax/params.py`. 2. Full-fidelity, bottom-up, match the C port 1:1. 3. AD-safe
   by construction + gradient re-run at every gate. 4. Mesh = index gather/scatter over
   `ops.py`. 5. Single-device + data-parallel now; mesh sharding Phase 8. 6. **Phase 5 = linfs
   on CORE2, Path A; zstar/partial-cells/GM/KPP/ice are later.** 7. **netCDF4 in the env** for
   NetCDF-4 readers.

## CRITICAL GOTCHAS (full list in PORTING_LESSONS.md)
- **The C port is linfs-only / full-cell / no-cavity — match THAT, not real-FESOM.** zlevel
  lives only in the Fortran; the parent outline's zlevel/partial-cells/w_i are out of scope.
- **pi↔CORE2 orientation:** CORE2 raw mesh is ~all CCW; the C `orient_cw` normalizes to CW
  before geometry. `load_mesh` now asserts CW (guards the Aleutian wrong-sign blow-up).
- **AD masked-NaN rule (bit us 4×):** make masked lanes finite; a forward `where` doesn't stop
  a backward 0·inf.
- **Eager CORE2 `step()` ≈ 32 s/step on CPU** (~40× pi nodes, super-linear) → use jitted
  `run`/`integrate` or GPU for 5.7; `build_ssh_operator` itself is cheap (0.3 s).
- **PHC dump is surface-only** → vertical interp/deep-ptheta verified indirectly (5.7 density
  gate); `T_old`/`S_old` step-1 AB2 history finalized in 5.7.
- **CORE2-without-ice stability risk** (dt=500, PHC+JRA, no ice): the matched C run is the
  arbiter — if the C port itself blows up without ice, that's a finding (ice would move into
  Phase 5). Watch the Aleutian Trench (global elem 194724; vertex node gid 94122).
- **netCDF4 import prints a benign `ndarray size changed` ABI warning** — harmless.
- **config = the pi reference physics on CORE2:** linfs, PP, FCT, opt_visc=7, `use_wsplit=0`,
  CG SSH (α=1), dt=500, PHC IC, JRA55+SSS+runoff.

## WORKFLOW NOTES
- The sub-plan is authoritative for Phase 5; tick `[x]`, keep its Revision Log + the lessons
  current. Commit only when asked (per-task commits on `main`). C edits → port2 (branch
  `jax-mesh-export`); job scripts kept untracked there. All python/pytest via the env python.

Confirm you've absorbed this, tell me which task you're starting (5.3), then proceed.
