# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. **Phases 0–5 COMPLETE (GATEs 0/1/2/3/4/5).
Phase 5 (CORE2 single-device) is DONE — the model runs the pi physics on the CORE2 mesh
with PHC IC + real JRA55/SSS/runoff forcing, matches the C per-substep dump at step 1, is
numerically stable for ~a week, and is differentiable end-to-end (gradient-gated). Phase 6
(sea ice + GM/Redi + KPP) is NEXT and needs a sub-plan first.** Full suite **376 passing**
(pi 313 + CORE2: `test_mesh_core2` 12, `test_phc_ic` 5, `test_jra55` 5, `test_forcing` 12,
`test_sss_runoff` 9, `test_surface_bc` 7, `test_core2_step` 6, `test_gradient_core2` 3).

Phase 5 delivered, in order (sub-plan Tasks 5.1–5.8): CORE2 mesh export (zero-code load +
CW-orientation guard), PHC IC (numpy, ~1e-14), JRA55-do reader (numpy, bit-exact), L&Y09
bulk (AD-safe, ~1e-13), SSS-restoring + runoff (numpy + AD-safe), the assembled CORE2 step
(surface BCs wired, step-1 T/S **bit-exact** vs the C dump), the matched-C dump run +
multi-day stability, and **GATE 5 — the gradient gate on a CORE2 slice** (the new bulk
SST→flux / current→stress feedbacks AD↔FD ~1e-11; the assembled-model masked-NaN probe finite
everywhere; the CG implicit-diff transpose residual 8.8e-14 on the CORE2 operator; the N=20
checkpointed backward fits the A100 at 37.8 GB).

**JAX committed on `main`** through Task 5.8. **CORE2 data artifacts live on `/work`** —
`port_jax/data` is a **symlink** to `/work/ab0995/a270088/port_jax/data` (gitignored), holding
`mesh_core2/`, `ic_core2/`, and the `{phc,jra,bulk,sss,step}_dump_core2/` C dumps. C-side dump
additions are on the port2 `jax-mesh-export` branch; **C job scripts stay untracked** per user
(port2 is otherwise the user's — no housekeeping). ⚠️ **Cheap C jobs: `-p compute
--time=00:30:00` (fast debug QOS); long C runs / JAX GPU: `-A ab0995_gpu -p gpu --gres=gpu:1`
(or `-p gpu-devel`, 30 min).** SLURM job `.out` logs are gitignored (`scripts/*.out`).

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean model for
hybrid ML (trainable NN parameterizations for vertical mixing + mesoscale eddy fluxes,
trained end-to-end). Multi-session effort. Work from `/home/a/a270088/port_jax`. Max effort.

## START HERE, in order
1. **Parent plan (source of truth across phases):**
   `docs/plans/20260605-fesom-jax-port.md` — decisions, the verification ladder, the
   Revision Log, and the **Phase 6 outline**.
2. **Phase-5 sub-plan (now COMPLETE — read for the CORE2 design + the runoff handoff):**
   `docs/plans/20260606-fesom-jax-core2.md` — Tasks 5.1–5.8 all `[x]`. **Read especially
   "Runoff handoff to Phase 6"** (the locked plan for activating runoff via the ice
   freshwater budget — the reader + balance are already done and pure-in-`water_flux`).
3. **Lessons (every session):** `docs/PORTING_LESSONS.md` — esp. the **Phase 5** entries
   through **Task 5.8** (the no-ice-supercooling finding, the multi-step-non-smooth gradient
   finding, the CG-residual / `d_eta⊥a_ver` insight, the bulk-seam smooth-subset FD, the
   backward-memory numbers). **STANDING RULE: append a lesson per task.**
4. **Project memory:** `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.

## STATUS
- **Phases 0–4 (GATEs 0–4):** full single-step pi model, checkpointed `lax.scan`, CG
  `custom_linear_solve`, FCT + opt_visc7, pi 1000-step stable, `test_gradient.py` plateau
  5.70e-7. All committed.
- **Phase 5 (GATE 5) COMPLETE:** the pi physics (PP + **linfs** + FCT + opt_visc7, **no
  GM/KPP/ice**) on the **CORE2 mesh** + PHC IC + JRA55/SSS/runoff, verified per-substep vs a
  CORE2 C-port dump (Path A), numerically stable days 1–7, differentiable end-to-end and
  gradient-gated (the new bulk feedbacks + the assembled-model masked-NaN + the CG residual +
  the backward memory). Reusable forward driver `scripts/core2_stability_run.py` (A100
  ~0.06 s/step); gradient gate `scripts/core2_grad_gate.py` (+`.sbatch`).
- **Two standing Phase-5 findings Phase 6 resolves:** (1) ⚠️ **no-ice high-lat
  supercooling** — without sea ice the SST supercools without bound (−22 °C by day 8) and the
  sub-EOS-valid density destabilizes the dynamics at model day ~8; the C does the same through
  the verified window, so it's PHYSICAL, capped by the ice in Phase 6. (2) **runoff is inert**
  in the no-ice linfs run (it enters via ice thermo, which is off) — activates for free once
  Phase 6 ports the ice freshwater budget (handoff plan locked in the sub-plan).

## IMMEDIATE WORK — Phase 6 (sea ice → GM/Redi → KPP). NEEDS A SUB-PLAN FIRST.
Phase 6 is the next big phase; like Phase 5 it should start with a **sub-plan**
(`docs/plans/<date>-fesom-jax-phase6.md`) scoped by **reading the C ice port** (the
algorithmic source of truth), not the parent outline. Suggested order (confirm with the user):

1. **Sea ice** (the high-value piece — caps supercooling, activates runoff, and is the path
   to a physically realistic CORE2 run):
   - **Thermodynamics** `fesom_ice_thermo.c` (the ice `obudget` / growth-melt) + the
     ice-ocean freshwater + heat coupling `fesom_ice_coupling.c` (`flx_fw`, `water_flux =
     −flx_fw` incl. runoff, the heat-flux ice blend). **This is what activates runoff** — see
     the sub-plan handoff: feed `water_flux = −flx_fw` into the EXISTING
     `sss_runoff.sss_runoff_fluxes` (pure in `water_flux`); `runoff_node` is already plumbed.
   - **Dynamics** (the ice EVP momentum solver, if present in the C port) + advection.
   - The static `a_ice` mask (Phase 5's `core2_forcing.ice_ic_aice`) becomes a **dynamic**
     prognostic; the two `a_ice` couplings (shortwave-penetration gate + the momentum stress
     blend) already exist in `core2_forcing.compute_surface_fluxes` — wire them to the
     prognostic `a_ice`. **AD-safe by construction** (the ice thermo has freezing-point
     `min`/`max` kinks + the EVP — guard every masked divide / safe-sqrt, re-run the gradient
     gate).
   - **Verify** with the proven recipe: a per-substep C dump at the **ice-ON** config (drop
     `FESOM_NO_ICE_*`), gate JAX vs C at pinned probes incl. river-mouth nodes (runoff
     freshening) and a high-lat ice node (supercooling now capped); multi-day stability with
     the supercooling resolved.
2. **GM/Redi** (mesoscale eddy fluxes — the SECOND ML-hook seam): port `fesom_gm*`/Redi from
   the C; the `params.py` seam already anticipates the eddy-flux swap point.
3. **KPP** (the other vertical-mixing scheme — the FIRST ML-hook's alternative): port from the
   C; swap point is `pp.py` ↔ a `kpp.py` behind the `params.py` mixing seam.

**First moves for Phase 6:** (1) read the C ice sources in `port2/fesom2_port/src/fesom_ice*.c`
to scope what's actually there (the C port is the spec — match THAT, like Phase 5's linfs
discovery). (2) Draft the Phase-6 sub-plan with a task ladder + per-task gates. (3) Decide the
order with the user (ice first is the recommendation — it unblocks realistic SST + runoff).

## THE PROVEN VERIFICATION RECIPE (still applies)
Per-substep dump at pinned probes, truncate to `nlevels`, `verify.assert_close(col, rec,
kind=…)` (`map`/`gather` 1e-15, `scatter`/`reduction` 1e-12; calibrate `atol`; **relative**
for big intermediate fields like `ssh_rhs`~1e5 / `pressure`~5e5). A per-substep gate is only
"tight" at **step 1** (shared inputs); downstream of the chaotic CG it diverges to ~1e-6 by
step 3 — gate the multi-step trajectory on **robust global reductions** (SST-range, max-speed)
vs a matched **C arbiter**. AD: any divide/sqrt whose denominator/arg can vanish in a masked
lane must compute a FINITE value (`where(d==0,1,d)` / double-`where` safe-sqrt) — a forward
`where` does NOT stop a 0·inf NaN backward. **Re-run the gradient gate at every GATE.**
⚠️ **Phase-5 gradient lesson:** a clean FD↔AD plateau is only well-posed in **smooth regimes**;
the multi-step *forced* trajectory is genuinely non-smooth in the physics params (active flux
limiter + convective adjustment — sea ice adds freezing-point kinks), so validate FD↔AD at
N=1 / on isolated seams / by the linear-solve residual, and lean on the (sub)gradient +
masked-NaN finiteness for the full model.

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`
- **Env python (ALL python/pytest):**
  `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q` (~12 min; CORE2 tests skip if their
  `data/` artifacts are absent). **netCDF4 + scipy + jax-cuda12 (GPU) installed.**
- GPU via SLURM: `-A ab0995_gpu -p gpu --gres=gpu:1` (12 h) or `-p gpu-devel` (30 min). The
  jitted CORE2 step is **~0.06 s on an A100**; the N=20 checkpointed gradient backward is
  ~37.8 GB / ~230 s. ⚠️ a heavy backward (full T-field grad) **OOMs/CPU-kills on the login
  node** — use the GPU (`scripts/core2_grad_gate.sbatch`).
- CORE2 mesh: `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`; PHC IC:
  `/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc`; JRA55:
  `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/`; SSS `…/PHC2_salx.nc`, runoff
  `…/CORE2_runoff.nc`; chl `…/Sweeney/Sweeney_2005.nc`.
- CORE2 data (gitignored): **`data/` is a symlink → `/work/ab0995/a270088/port_jax/data`**.
- C port (algorithmic SoT): `/home/a/a270088/port2/fesom2_port/src/`; built binary
  `…/build/fesom_port`. C run-arg order: `<mesh> <out> <dt> <nsteps> <snap> <phc> <jra>`.
  Build: `bash -lc 'source env.sh && make -C build fesom_port'` (incremental, ~30 s). **Ice
  sources to read for Phase 6: `fesom_ice*.c` (thermo / coupling / dynamics).**
- I (Claude) drive SLURM (acct ab0995 / ab0995_gpu). Cheap C dumps → `-p compute
  --time=00:30:00` (debug). Long C / JAX-GPU → a real QOS. See memory
  [[hpc-job-file-conventions]].

## LOCKED DECISIONS (do NOT re-litigate)
1. Use case = hybrid ML params (swap points: vertical mixing PP/KPP, eddy flux GM/Redi);
   seam = `fesom_jax/params.py`. 2. Full-fidelity, bottom-up, match the C port 1:1. 3. AD-safe
   by construction + gradient re-run at every gate. 4. Mesh = index gather/scatter over
   `ops.py`. 5. Single-device + data-parallel now; mesh sharding Phase 8. 6. **Phase 5 = linfs
   on CORE2, Path A; zstar/partial-cells are later C-side work (keep the `which_ale` seam).**
   7. netCDF4 + scipy + jax-cuda12 in the env. 8. Phase 5 kept the C's **static `a_ice` mask**
   (=0.9 where IC SST<0); **Phase 6 makes `a_ice` prognostic** (the couplings already exist in
   `core2_forcing`). 9. **(Task 5.7) the no-ice high-lat supercooling is an accepted PHYSICAL
   limitation** capped by Phase-6 sea ice. 10. **(Task 5.8) the gradient gate validates FD↔AD
   in smooth regimes only** (the forced multi-step model is non-smooth in the physics params;
   the AD is a valid sub-gradient — see the lesson).

## CRITICAL GOTCHAS (full list in PORTING_LESSONS.md)
- **The C port is the spec — port it 1:1 and verify by dump.** Phase 5's scope came from
  reading `fesom_ale.c` (linfs-only), not the parent outline; do the same for Phase-6 ice.
- **⚠️ "No ice" ≠ ice-free AND ≠ stable forever:** the static `a_ice=0.9` mask gates shortwave
  + blends stress; the no-ice SST supercools without bound (−22 by day 8 → max|vel|>3 ~day 8).
  Phase-6 sea ice caps it.
- **⚠️ Runoff is INERT in the no-ice Phase-5 run by the C's design** (its local term lives in
  ice thermo, off in Phase 5) — **Phase 6 activates it** via the locked handoff (feed
  `water_flux=−flx_fw` incl. runoff into the existing `sss_runoff_fluxes`).
- **⚠️ CORE2 step-1 `T_old`/`S_old` = const base 10/35, NOT PHC** (`core2_initial_state`).
- **⚠️ `FESOM_BULK_FIXED_ITERS=1`** on the C reference (the M-O loop is non-convergent at calm
  nodes; JAX runs fixed-5).
- **AD masked-NaN rule (bit us 4×):** make masked lanes finite; a forward `where` doesn't stop
  a backward 0·inf. The Phase-6 ice thermo (freezing-point min/max, EVP) is the next AD target.
- **Eager CORE2 `step()` ≈ 32 s/step; jitted ≈ 3 s CPU / 0.06 s A100** → always jit; GPU for
  any backward.
- **netCDF4 import prints a benign `ndarray size changed` ABI warning** — harmless.
- **config = the pi reference physics on CORE2:** linfs, PP, FCT, opt_visc=7, `use_wsplit=0`,
  CG SSH (α=1), dt=500, PHC IC, JRA55+SSS+runoff + (Phase 5) the static ice mask.

## WORKFLOW NOTES
- Phase 6 needs its own sub-plan (authoritative for the phase); tick `[x]`, keep its Revision
  Log + the lessons current. **Commit only when asked** (per-task commits on `main`). **C edits
  → port2 branch `jax-mesh-export`, NEVER port2 main** (user); job scripts kept untracked
  there; otherwise leave the port2 repo to the user (no housekeeping). **Large generated files
  → `/work`** (the `data` symlink). Cheap C jobs → `-p compute --time=00:30:00`; long C /
  JAX-GPU → a real QOS. All python/pytest via the env python. See memory
  [[hpc-job-file-conventions]].

Confirm you've absorbed this, propose the Phase-6 sub-plan scope (read the C ice sources
first), then proceed.
