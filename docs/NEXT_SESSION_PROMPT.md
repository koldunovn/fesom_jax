# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. **Phases 0–4 COMPLETE (GATEs 0/1/2/3/4).
Phase 5 (CORE2 single-device) IN PROGRESS — sub-plan created; Tasks 5.1 + 5.2 + 5.3 + 5.4 DONE;
Task 5.5 (SSS restoring + runoff) is NEXT.** Full suite **349 passing** (pi 313 + CORE2 additions,
incl. `test_forcing.py` +10).

The pi model is fully ported, dump-gated, jitted, differentiable end-to-end, 1000-step
stable. Phase 5 runs that same physics (PP + **linfs** + FCT + opt_visc7, **no GM/KPP/ice**)
on the **CORE2 mesh** with PHC initial conditions and real JRA55+SSS+runoff forcing,
verified per-substep against a CORE2 C-port dump (**Path A**). So far: the CORE2 mesh is
exported + loads zero-code (with a new **clockwise-orientation guard**), the **PHC IC** is
ported (numpy, ~1e-14), the **JRA55-do reader** is ported (numpy) and matches a C all-node
dump **bit-exact**, and the **L&Y09 bulk** is ported (AD-safe JAX) and matches a C all-node
bulk dump to **~1e-17 (coeffs) / ~6e-13 (heat) / ~5e-16 (stress)** — the differentiable
SST→flux / current→stress seam.

**Everything committed on `main`** (`c1aa677` Task 5.4, `5ab28af` Task 5.3, `d4fcdb2` Task 5.2,
`4b34f6a` Task 5.1). The C-side dump additions are committed on the **port2 `jax-mesh-export`
branch** (`4ffc542` Task 5.4) — ⚠️ **keep all C commits on that branch, never on port2 main**
(user); the port2 repo is otherwise the user's — don't do housekeeping there. **CORE2 data
artifacts now live on `/work`** — `port_jax/data` is a **symlink** to
`/work/ab0995/a270088/port_jax/data` (gitignored via both `/data` and `/data/`; user rule "large
files on /work not /home"). New C-side job scripts live **untracked** in `port2/fesom2_port/jobs/`
(`jax_mesh_export_core2.sh`, `jax_phc_dump_core2.sh`, `jax_jra_dump_core2.sh`,
`jax_bulk_dump_core2.sh`). The C `dump_jra_fields` + `fesom_bulk_dump` + the
`ncar_ocean_fluxes_mode` `fixed_iters` param (in `fesom_bulk.c`/`fesom_main.c`) are dump-only
additions (gated on `FESOM_JRA_DUMP_DIR`/`FESOM_BULK_DUMP_DIR`). ⚠️ **Cheap C jobs: use
`-p compute --time=00:30:00` (fast debug QOS), not `-p shared`** (~16 s to start vs minutes).

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean model for
hybrid ML (trainable NN parameterizations for vertical mixing + mesoscale eddy fluxes,
trained end-to-end). Multi-session effort. Work from `/home/a/a270088/port_jax`. Max effort.

## START HERE, in order
1. **Phase-5 sub-plan (source of truth for Phase 5):**
   `docs/plans/20260606-fesom-jax-core2.md` — the scope correction (§0), Path A, the task
   ladder 5.1–5.8 with per-task gates, the 5 module research briefs, risks. **Tasks 5.1 +
   5.2 + 5.3 + 5.4 are ticked DONE; 5.5 is next.**
2. **Main plan:** `docs/plans/20260605-fesom-jax-port.md` — decisions, the verification
   ladder, the Revision Log. Phase 5 there is the outline; the sub-plan supersedes it.
3. **Lessons (every session):** `docs/PORTING_LESSONS.md` — esp. the **Phase 5** entries:
   the linfs-not-zlevel scope rule, the **pi↔CORE2 orientation trap**, the netCDF4 env note,
   the PHC sequential-GS extrap, the "no invented modeling choice" rule. **STANDING RULE:
   append a lesson per task.**
4. **Project memory:** `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.
5. For 5.5: read `port2/fesom2_port/src/fesom_sss_runoff.c` (`fesom_sss_runoff_init` readers +
   `fesom_sss_runoff_step` flux math). The sub-plan's task body holds the detailed per-module
   brief (functions, formulas, gotchas, AD-safe guards, suggested gate). `fesom_bulk.c` (5.4) is
   done — `fesom_jax/forcing.py` (`bulk_surface_fluxes`) is the AD-safe bulk producing the
   `water_flux` the SSS/runoff math consumes.

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
- **Task 5.3 DONE:** `fesom_jax/jra55.py` — faithful numpy port of `fesom_jra55.c` (julday +
  per-field time-grid transform with the **mid-interval shift** + shared bilinear stencil +
  per-field `getcoeffld` cache + **wind g2r rotation**, Euler 50/15/−90). Verified vs a new C
  all-node dump (job 25388630, year 1958, `data/jra_dump_core2/`) at two dates
  (day1/sec0 + day100/12:00): **6 scalar fields bit-exact** (max|diff|=0, all 126858 nodes),
  wind **~3.5e-15**. `test_jra55.py` (5). ⚠️ **#1 trap:** the C time-interp
  `field=rdate·coef_a+coef_b` cancels two ~2.4e6 Julian-day numbers → a folded-`1/denom` gather's
  ~1e-13 reassociation blew to ~6e-8; fixed by a **bit-identical `(s·dx)·dy` + divide-at-end
  gather**. Field order uas,vas,huss,rsds,rlds,**tas**,prra,prsn; interp on geographic coords,
  rotate wind after; `flip_lat=0`. New `dump_jra_fields` in `fesom_main.c` + `jax_jra_dump_core2.sh`.
- **Task 5.4 DONE:** `fesom_jax/forcing.py` — AD-safe port of `fesom_bulk.c`
  (`ncar_ocean_fluxes_mode` fixed-5 unrolled + `obudget` + `bulk_surface_fluxes`, node→elem
  mean-of-3). Verified vs a C `bulk_dump_*` dump (job 25389451, `data/bulk_dump_core2/`) at
  3 configs (zero + synthetic current, day1 + day100/noon): **cd/ce/ch ~1e-17, heat_flux ~6e-13,
  stress ~5e-16** over all 126858 nodes. `test_forcing.py` (10). ⚠️ **FINDING:** the sub-plan's
  "drop the early break ⇒ identical result" claim was WRONG — the M-O loop is non-convergent at
  calm nodes (`ch` up to **88%** fixed-5-vs-early-break) but the **physical** impact is bounded
  (heat_flux ≤7.2 W/m² at ~4 nodes). JAX runs fixed-5 (AD-safe) vs a **fixed-5** C dump via a new
  `FESOM_BULK_FIXED_ITERS` env gate. AD-safe `x2=sqrt(max(|1−16ζ|,1))` + double-`where` safe-sqrt
  (`u`/`mag`) + literal `jnp.copysign` switches; relative-vs-absolute wind mismatch preserved
  (synthetic-current dump mode). `USE_SW_PENE=1` ⇒ shortwave penetration deferred to 5.6
  (heat_flux = pre-pene `qns−qsr`). New `fesom_bulk_dump` + `jax_bulk_dump_core2.sh`.

## IMMEDIATE WORK — Task 5.5 onward (the sub-plan has the full ladder + gates)
- **5.5 SSS restoring + runoff** (numpy readers + AD-safe JAX): match `fesom_sss_runoff.c`
  exactly (virtual_salt + relax_salt into the S surface BC; runoff via the global-mean term in
  the no-ice path); `ref_sss_local=1`, `surf_relax_S=1.929e-6`; month index no legacy `+1`.
- **5.6 wire surface BCs** into `tracer_diff` (`bc_T=−dt·heat_flux/vcpw`,
  `bc_S=dt·(virtual_salt+relax_salt)`; currently `bc_surface=0`) + bulk `stress_surf`→momentum.
- **5.7 matched C dump + stability:** generate the CORE2 per-substep dump (re-pin
  `PROBE_GIDS` in `fesom_dump.c` incl. an Aleutian node, e.g. 94122); assemble + run CORE2
  1-day (~172 steps, dt=500) + multi-day; assert stable. ⚠️ **Use the JITTED `run`/`integrate`
  (or GPU)** — an eager CORE2 step is ~32 s on CPU. Finalize `T_old`/`S_old` (step-1 AB2)
  against the dump. ⚠️ **Set `FESOM_BULK_FIXED_ITERS=1` on the reference dump run** — else the
  bulk coefficients won't match JAX at calm nodes (the early-break finding from 5.4).
- **5.8 GATE 5:** gradient check on a CORE2 slice (+ the new d(heat_flux)/d(SST),
  d(stress)/d(current) feedbacks); confirm checkpointed backward fits the GPU at CORE2 scale.

**First moves for 5.5:** read `fesom_sss_runoff.c` (`fesom_sss_runoff_init` readers +
`fesom_sss_runoff_step` flux math); SSS `PHC2_salx.nc` (12-month, 30-cell expanding-neighbour
fill) → `Ssurf_clim[12,nod2D]`, runoff `CORE2_runoff.nc` (single record, /1000 → m/s) →
`runoff_node[nod2D]`, ported as host-numpy readers (like `phc_ic`/`jra55`). The AD-safe JAX flux
math (`virtual_salt=rsss·water_flux` with `rsss=S_top`/`ref_sss_local=1`; `relax_salt=
surf_relax_S·(Ssurf−S_top)`; subtract the area-weighted global mean; `water_flux += mean(...)`)
consumes the bulk `water_flux` (Task 5.4) + `S[:,0]`. Gate vs a new C `sss_runoff` dump (extend
the `FESOM_BULK_DUMP_DIR` pattern; dump after `fesom_sss_runoff_step`). Then **5.6** wires
`bc_T`/`bc_S`/`stress_surf` into the step. The bulk `forcing.bulk_surface_fluxes` →
`BulkFluxes(heat_flux, water_flux, stress_surf, …)` is the 5.6 input.

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
- CORE2 mesh / IC / C dumps (gitignored): **`data/` is a symlink → `/work/ab0995/a270088/port_jax/data`**
  (large files on /work, not /home — user rule). Holds `mesh_core2/`, `ic_core2/`,
  `phc_dump_core2/`, `jra_dump_core2/`, `bulk_dump_core2/`. C-side jobs (untracked in port2):
  `port2/fesom2_port/jobs/jax_*.sh`.
- C port (algorithmic SoT): `/home/a/a270088/port2/fesom2_port/src/`; built binary
  `…/build/fesom_port` (NL read from the mesh, handles pi+CORE2). C run-arg order:
  `<mesh> <out> <dt> <nsteps> <snap> <phc> <jra>` (phc=argv6 triggers PHC IC, t_insitu=1).
  Build: `bash -l configure.sh` (or `cd build && make` after `source env.sh`).
- I (Claude) drive the C SLURM jobs (`sbatch`, acct ab0995): use **`-p compute --time=00:30:00`**
  (fast debug QOS, ~16 s to start) — NOT `-p shared`. They're cheap (mesh export 17 s, dumps ~20 s).
  See memory [[hpc-job-file-conventions]].

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
- **L&Y09 bulk: fixed-5 ≠ early-break at calm nodes** (the M-O loop is non-convergent; `ch` up
  to 88%, but heat_flux ≤7 W/m² at ~4 nodes — bounded). JAX uses **fixed-5** (AD-safe); **set
  `FESOM_BULK_FIXED_ITERS=1` on any C bulk reference** (Task 5.7). `heat_flux = qns−qsr` is
  pre-shortwave-penetration (Task 5.6 removes the 0.54-visible band + builds `sw_3d`).
- **config = the pi reference physics on CORE2:** linfs, PP, FCT, opt_visc=7, `use_wsplit=0`,
  CG SSH (α=1), dt=500, PHC IC, JRA55+SSS+runoff.

## WORKFLOW NOTES
- The sub-plan is authoritative for Phase 5; tick `[x]`, keep its Revision Log + the lessons
  current. **Commit only when asked** (per-task commits on `main`). **C edits → port2 branch
  `jax-mesh-export`, NEVER port2 main** (user); job scripts kept untracked there; otherwise leave
  the port2 repo to the user (no housekeeping). **Large generated files → `/work`** (the `data`
  symlink handles it). Cheap C jobs → `-p compute --time=00:30:00`. All python/pytest via the env
  python. See memory [[hpc-job-file-conventions]].

Confirm you've absorbed this, tell me which task you're starting (5.5), then proceed.
