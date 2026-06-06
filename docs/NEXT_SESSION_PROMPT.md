# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. **Phases 0–4 COMPLETE (GATEs 0/1/2/3/4).
Phase 5 (CORE2 single-device) IN PROGRESS — sub-plan created; Tasks 5.1–5.5 DONE;
Task 5.6 (wire surface BCs into the step + assemble CORE2 forcing) is NEXT.** Full suite
**358 passing** (pi 313 + CORE2 additions, incl. `test_sss_runoff.py` +9).

The pi model is fully ported, dump-gated, jitted, differentiable end-to-end, 1000-step
stable. Phase 5 runs that same physics (PP + **linfs** + FCT + opt_visc7, **no GM/KPP/ice**)
on the **CORE2 mesh** with PHC initial conditions and real JRA55+SSS+runoff forcing,
verified per-substep against a CORE2 C-port dump (**Path A**). So far: the CORE2 mesh is
exported + loads zero-code (with a **clockwise-orientation guard**), the **PHC IC** is
ported (numpy, ~1e-14), the **JRA55-do reader** is ported (numpy, **bit-exact**), the
**L&Y09 bulk** is ported (AD-safe JAX, ~1e-13 — the SST→flux / current→stress seam), and
the **SSS-restoring + runoff balance** is ported (numpy readers + AD-safe JAX) and matches a
C all-node dump — **runoff bit-exact, SSS bit-exact at 105148/126858 nodes, the salt/water
balance ~1e-20** (the differentiable `water_flux→virtual_salt` / `S_top→relax_salt` seam).
**The bulk + SSS/runoff produce the `heat_flux`/`water_flux`/`virtual_salt`/`relax_salt`/
`stress_surf` that Task 5.6 wires into the step.**

**Everything committed on `main`** (`d906750` Task 5.5, `c1aa677` Task 5.4, `5ab28af` Task 5.3,
`d4fcdb2` Task 5.2, `4b34f6a` Task 5.1). The C-side dump additions are committed on the port2
`jax-mesh-export` branch (`78b1df1` Task 5.5) — ⚠️ **keep all C commits on that branch, never on
port2 main** (user); the port2 repo is otherwise the user's — don't do housekeeping there. **CORE2
data artifacts live on `/work`** — `port_jax/data` is a **symlink** to
`/work/ab0995/a270088/port_jax/data` (gitignored via both `/data` and `/data/`; user rule "large
files on /work not /home"). New C-side job scripts live **untracked** in `port2/fesom2_port/jobs/`
(`jax_mesh_export_core2.sh`, `jax_phc_dump_core2.sh`, `jax_jra_dump_core2.sh`,
`jax_bulk_dump_core2.sh`, `jax_sss_dump_core2.sh`). The C `dump_jra_fields` + `fesom_bulk_dump` +
`fesom_sss_runoff_dump` + the `ncar_ocean_fluxes_mode` `fixed_iters` param are dump-only additions
(gated on `FESOM_JRA_DUMP_DIR`/`FESOM_BULK_DUMP_DIR`/`FESOM_SSS_DUMP_DIR`). ⚠️ **Cheap C jobs: use
`-p compute --time=00:30:00` (fast debug QOS), not `-p shared`** (~16 s to start vs minutes).

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean model for
hybrid ML (trainable NN parameterizations for vertical mixing + mesoscale eddy fluxes,
trained end-to-end). Multi-session effort. Work from `/home/a/a270088/port_jax`. Max effort.

## START HERE, in order
1. **Phase-5 sub-plan (source of truth for Phase 5):**
   `docs/plans/20260606-fesom-jax-core2.md` — the scope correction (§0), Path A, the task
   ladder 5.1–5.8 with per-task gates, the 5 module research briefs, risks. **Tasks 5.1–5.5
   are ticked DONE; 5.6 is next.**
2. **Main plan:** `docs/plans/20260605-fesom-jax-port.md` — decisions, the verification
   ladder, the Revision Log. Phase 5 there is the outline; the sub-plan supersedes it.
3. **Lessons (every session):** `docs/PORTING_LESSONS.md` — esp. the **Phase 5** entries:
   the linfs-not-zlevel scope rule, the **pi↔CORE2 orientation trap**, the netCDF4 env note,
   the PHC sequential-GS extrap, the JRA cancellation/bit-exact-gather trap, the bulk
   fixed-5≠early-break finding, the SSS **Jacobi-fill-vectorizes** + **÷ocean_area crushes the
   global-mean** lessons, the "no invented modeling choice" rule. **STANDING RULE: append a
   lesson per task.**
4. **Project memory:** `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.
5. For 5.6: read `port2/fesom2_port/src/fesom_tracer_diff.c` — `bc_surface` (**@44-72**:
   `bc_T=−dt·heat_flux/vcpw`, `bc_S=dt·(virtual_salt+relax_salt)` for linfs; the `sw_3d`
   divergence added to the T tracer **@295-299**) — and `fesom_cal_shortwave_rad`
   (`fesom_bulk.c:362-415`, the deferred 5.4 sub-item — builds `sw_3d`, needs `chl`). JAX side:
   `fesom_jax/tracer_diff.py` (currently **`bc_surface=0`**), `fesom_jax/step.py` (`stress_surf`
   is already a `step()` arg), `fesom_jax/forcing.py` (`bulk_surface_fluxes → BulkFluxes`),
   `fesom_jax/sss_runoff.py` (`sss_runoff_fluxes → SSSFluxes`).

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

## IMMEDIATE WORK — Task 5.6 onward (the sub-plan has the full ladder + gates)
- **5.6 wire surface BCs into the step + assemble CORE2 forcing:**
  - **bc_T/bc_S into `tracer_diff`** (currently `bc_surface=0`, Phase 2): `bc_T =
    −dt·heat_flux/vcpw` (`fesom_tracer_diff.c:56`; linfs ⇒ the `sval·water_flux·is_nonlinfs`
    term is 0), `bc_S = dt·(virtual_salt + relax_salt)` (`:67-69`; `real_salt_flux=0`, sea-ice
    off). Add `bc_T`/`bc_S` args to `impl_vert_diff`/`impl_vert_diff_one` (the surface-layer
    forcing increment added to `tr[nzmin]`). `vcpw=4.2e6`.
  - **Shortwave penetration** (the deferred 5.4 sub-item, `USE_SW_PENE=1`):
    `fesom_cal_shortwave_rad` (`fesom_bulk.c:362-415`) removes the **0.54-visible** band from
    `heat_flux` (`heat_flux += 0.54·(1−albw)·shortwave`) + builds the per-column `sw_3d`
    (Sweeney-2005 two-band exponential), which the **T** diffusion consumes as a flux divergence
    (`fesom_tracer_diff.c:295-299`). ⚠️ **Needs `chl`** — decide constant
    (`FESOM_PHASE1_CHL_CONST`) vs Sweeney monthly climatology (`Sweeney_2005.nc`); check what the
    matched config uses (`fesom_main.c:1080-1097`, env `FESOM_CHL_SRC`/`FESOM_CHL_FILE`). This is
    the **hardest** sub-piece (chl reader + 3-D `sw_3d` build + the T-tracer divergence term).
  - **stress_surf (bulk) into momentum:** `step()` already takes `stress_surf` — pass the bulk
    `BulkFluxes.stress_surf` (Task 5.4, node→elem mean-of-3) instead of pi's analytical wind.
  - **Thread the forcing through `step`/`integrate`:** the bulk runs each step consuming the SST
    tap `T[:,0]` + surface-current tap `uvnode[:,0,:]`; the per-step jra atmosphere (host-numpy
    `JRAFields` → device constant) + the month index are loop-carried/closed-over; the SSS/runoff
    flux math consumes the bulk `water_flux`. Keep `params=None ⇒ defaults` transparency intact
    (the 313 pi tests must stay bit-identical — CORE2 forcing must not perturb the pi gates).
  - **Gate:** per-substep CORE2 dump at step 1 (tight, all kernels via `step()`); SST/SSS
    evolution vs C over a few steps. **Lesson:** append.
- **5.7 matched C dump + stability:** generate the CORE2 per-substep dump (re-pin `PROBE_GIDS`
  in `fesom_dump.c` incl. an Aleutian node, e.g. 94122); assemble + run CORE2 1-day (~172 steps,
  dt=500) + multi-day; assert stable. ⚠️ **Use the JITTED `run`/`integrate` (or GPU)** — eager
  CORE2 step ~32 s on CPU. Finalize `T_old`/`S_old` (step-1 AB2) against the dump. ⚠️ **Set
  `FESOM_BULK_FIXED_ITERS=1` on the reference dump run** (the calm-node bulk finding from 5.4).
- **5.8 GATE 5:** gradient check on a CORE2 slice (+ the new d(heat_flux)/d(SST),
  d(stress)/d(current) feedbacks); confirm checkpointed backward fits the GPU at CORE2 scale.

**First moves for 5.6:** read `fesom_tracer_diff.c` `bc_surface` (@44-72) + the `sw_3d`
divergence in `diff_ver_part_impl_ale` (@~290-310), and `fesom_cal_shortwave_rad`
(`fesom_bulk.c:362-415`); resolve the `chl` source (constant vs Sweeney — read
`fesom_main.c:1080-1130`). Then in JAX: (1) add `bc_T`/`bc_S` (+ the `sw_3d` T-divergence) to
`tracer_diff.py`; (2) build `cal_shortwave_rad` (chl reader → `sw_3d` 3-D, AD-safe — the
`heat_flux` change is on the SST→flux gradient path); (3) swap `step()`'s `stress_surf` to the
bulk; (4) write the per-step forcing closure that runs the bulk + SSS/runoff each step (jra atmo
+ month index loop-carried) and feeds `tracer_diff`/momentum, preserving `params=None`
transparency. Gate: a step-1 per-substep CORE2 dump (extend the dump pattern; dump `bc_T`/`bc_S`/
`sw_3d`/post-step `T`/`S`).

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
   on CORE2, Path A; zstar/partial-cells/GM/KPP/ice are later.** 7. **netCDF4 + scipy in the env.**

## CRITICAL GOTCHAS (full list in PORTING_LESSONS.md)
- **The C port is linfs-only / full-cell / no-cavity — match THAT, not real-FESOM.** zlevel
  lives only in the Fortran; the parent outline's zlevel/partial-cells/w_i are out of scope.
- **pi↔CORE2 orientation:** CORE2 raw mesh is ~all CCW; the C `orient_cw` normalizes to CW
  before geometry. `load_mesh` now asserts CW (guards the Aleutian wrong-sign blow-up).
- **AD masked-NaN rule (bit us 4×):** make masked lanes finite; a forward `where` doesn't stop
  a backward 0·inf. (Watch this in 5.6's `cal_shortwave_rad` — `log10(chl)`, `exp(z/sc)`.)
- **Eager CORE2 `step()` ≈ 32 s/step on CPU** → use jitted `run`/`integrate` or GPU for 5.7.
- **CORE2-without-ice stability risk** (dt=500, PHC+JRA, no ice): the matched C run is the
  arbiter — if the C port itself blows up without ice, that's a finding (ice would move into
  Phase 5). Watch the Aleutian Trench (global elem 194724; vertex node gid 94122).
- **netCDF4 import prints a benign `ndarray size changed` ABI warning** — harmless.
- **L&Y09 bulk: fixed-5 ≠ early-break at calm nodes** — JAX uses fixed-5; **set
  `FESOM_BULK_FIXED_ITERS=1` on any C bulk/per-substep reference** (Task 5.7).
- **heat_flux = qns−qsr is PRE-shortwave-penetration** — **Task 5.6** removes the 0.54-visible
  band + builds `sw_3d` (the active sub-item now). The T-tracer eqn consumes `heat_flux` (via
  `bc_T`) **and** `sw_3d` (a per-layer divergence) — both must be wired together.
- **SSS/runoff DONE (Task 5.5):** the bulk `water_flux` → `virtual_salt`/`relax_salt`/balanced
  `water_flux` chain is verified; 5.6 feeds `virtual_salt`/`relax_salt` into `bc_S`.
- **config = the pi reference physics on CORE2:** linfs, PP, FCT, opt_visc=7, `use_wsplit=0`,
  CG SSH (α=1), dt=500, PHC IC, JRA55+SSS+runoff.

## WORKFLOW NOTES
- The sub-plan is authoritative for Phase 5; tick `[x]`, keep its Revision Log + the lessons
  current. **Commit only when asked** (per-task commits on `main`). **C edits → port2 branch
  `jax-mesh-export`, NEVER port2 main** (user); job scripts kept untracked there; otherwise leave
  the port2 repo to the user (no housekeeping). **Large generated files → `/work`** (the `data`
  symlink handles it). Cheap C jobs → `-p compute --time=00:30:00`. All python/pytest via the env
  python. See memory [[hpc-job-file-conventions]].

Confirm you've absorbed this, tell me which task you're starting (5.6), then proceed.
