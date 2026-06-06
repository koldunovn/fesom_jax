# Next-session prompt — FESOM2 → JAX port

Paste the block below to start the next session. **Phases 0–5 COMPLETE (GATEs 0–5) + Phase 6
SEA ICE COMPLETE (GATE 6) — all committed on `main` (Phase 6 = 7 per-task commits 6.1–6.7).** The
prognostic-ice CORE2 model (thermo + EVP + FCT + coupling on top of the Phase-5 ocean) runs
**10 days stable with the high-lat supercooling CAPPED at −1.91 °C and runoff active** — the
two standing Phase-5 findings RESOLVED. Each ice kernel is bit-exact vs an ice-ON C dump; the
assembled step matches the C at step 1; gradient-gated (FD↔AD plateau 4.5e-10, masked-NaN clean
at scale). **Next big phases: GM/Redi (6B, the 2nd ML-hook) → KPP (6C), each a new sub-plan.**

Suite = **ocean 376 + ice 47** (`test_ice_{ic,thermo,coupling,evp,adv,step}.py`). ⚠️ Run the
ice tests as a SEPARATE group — the full 423-in-one-process exceeds the login-node RAM (the
heavy assembled `test_ice_step` + the jit-cache of 400 prior tests; a pytest+JAX pattern, NOT a
bug). Each file passes standalone; ocean-only (`--ignore` the 6 ice files) = 376; ice group = 47.

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean model for
hybrid ML (trainable NN parameterizations for vertical mixing + mesoscale eddy fluxes, trained
end-to-end). Multi-session effort. Work from `/home/a/a270088/port_jax`. Max effort.

## Phase 6 sea ice is committed (7 per-task commits on `main`). C-side dump hooks live on the
port2 `jax-mesh-export` branch (the user's to commit). The repo is clean except the stray
untracked `scripts/fullsuite.log` (pre-existing, not ours).

## START HERE, in order
1. **Parent plan (source of truth across phases):** `docs/plans/20260605-fesom-jax-port.md`
   — decisions, the verification ladder, the Revision Log, the **Phase 6 outline** (sea ice
   ✅; GM/Redi 6B = NEXT; KPP 6C).
2. **Phase-6 sea-ice sub-plan (COMPLETE — read for the ice design + the AD findings):**
   `docs/plans/20260606-fesom-jax-phase6-seaice.md` — Tasks 6.1–6.7 all `[x]`, GATE 6 met.
3. **Phase-5 CORE2 sub-plan:** `docs/plans/20260606-fesom-jax-core2.md` (the CORE2 model the
   ice sits on).
4. **Lessons (every session):** `docs/PORTING_LESSONS.md` — esp. the **Phase 6** entries
   (the supercooling-cap result, the EVP-stiff-gradient, the element-based-CSR-free FCT, the
   masked-NaN-in-both-loops rule, the GPU memory traps). **STANDING RULE: append a lesson per task.**
5. **Project memory:** `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.

## STATUS
- **Phases 0–5 (GATEs 0–5):** full pi model + CORE2 single-device (PP/linfs/FCT/opt_visc7 +
  PHC IC + JRA55/SSS/runoff), committed on `main`.
- **Phase 6 SEA ICE (GATE 6) COMPLETE, uncommitted.** New modules: `ice.py` (IceConfig +
  cold-start IC), `ice_thermo.py` (5-iter Newton + freezing/albedo kinks), `ice_coupling.py`
  (ocean2ice + oce_fluxes runoff handoff + stress blend), `ice_evp.py` (120-subcycle `lax.scan`),
  `ice_adv.py` (2-D Zalesak FCT, **CSR-free / element-based**), `ice_step.py` (the assembled
  step). `State` +9 ice fields (a/m/m_snow/u_ice/v_ice/t_skin + σ11/12/22, σ = EVP elastic
  memory). `step.py`/`integrate.py` gained a static `ice_cfg` arg (`None` ⇒ pi/Phase-5
  bit-identical; an `IceConfig` ⇒ the ice step runs before the ocean substeps). Reusable runs:
  `scripts/core2_ice_stability_run.py` (+`_gpu.sh`, A100 ~0.08 s/step), `scripts/core2_ice_grad_gate.py`
  (+`.sbatch`).
- **The two Phase-5 findings RESOLVED:** supercooling capped (sea ice), runoff active (handoff).

## IMMEDIATE WORK — Phase 6B (GM/Redi). NEEDS A SUB-PLAN FIRST.
GM/Redi is the **second ML-hook seam** (mesoscale eddy fluxes — the bolus velocity + Redi
neutral diffusion, substep 14). Like ice, start by **reading the C** (`fesom_gm.c` ~49 KB, the
algorithmic spec) to scope what's actually there, then draft `docs/plans/<date>-fesom-jax-gmredi.md`
with a task ladder + per-task dump gates. Suggested first moves:
1. Read `fesom_gm.c` (+ `fesom_gm.h`, FRESH_START GM section) — neutral slopes, tapering
   (Large et al.), the bolus velocity, the Redi rotation, how it enters substep 14
   (`gm_bolus`) + the tracer advection/diffusion. Note the AD hazards (slope clipping/tapering
   `min`/`max`, the safe divides) + the GM dump fields the C can emit.
2. Decide whether GM needs new C dump hooks (probably yes — a per-node bolus/slope dump,
   `fesom_bulk_dump`-style) or reuses the per-substep dump.
3. Draft the sub-plan; gate each kernel bit-exact vs the C dump; re-run the gradient gate.
The `params.py` seam already anticipates the eddy-flux swap point (Phase 7 puts an NN there).

## THE PROVEN VERIFICATION RECIPE (still applies — used for all of Phase 6)
Per-kernel: add a small all-node C dump hook (`fesom_*_dump`, env-gated, re-runs the kernel on
copies → inputs+outputs) on the port2 `jax-mesh-export` branch; clone a `-p compute
--time=00:30:00` dump job; feed JAX the C inputs → match outputs MAP/scatter-class
(map/gather ~1e-15, scatter/reduction ~1e-12; per-node maps often bit-exact). **The C port is
the spec — port 1:1.** AD: every divide/sqrt that can vanish in a masked lane must be FINITE
(`where(d==0,1,d)` / double-`where` safe-sqrt). Re-run the gradient gate at the GATE: FD↔AD
only in SMOOTH regimes (N=1 / isolated seams), lean on masked-NaN finiteness for the assembled
model. **A C `if(cond) skip` over a traced cond → run unconditionally + `where`-mask the output
+ guard every divide for the masked lane** (and mask in BOTH loops if the C skips in two).

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`
- **Env python (ALL python/pytest):** `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest fesom_jax/tests/ -q`. ⚠️ Run the **ice tests as a separate
  group** (don't run all 423 in one process — login-node RAM). ⚠️ Only ONE CPU-JAX process at a
  time on the login node (two → `pthread_create` OOM crash).
- GPU via SLURM: `-A ab0995_gpu -p gpu --gres=gpu:1` (A100-40, ~31.8 GB usable) or `-p
  gpu-devel`. The assembled ice step is ~0.06–0.08 s/step jitted on the A100. ⚠️ GPU memory
  traps: don't `cf.stack()` a long forcing trajectory (stream per-step); run ONE N-step backward
  per process (`jax.clear_caches()` between, or one probe per job).
- CORE2 mesh `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`; PHC `…/INITIAL/phc3.0/phc3.0_winter.nc`;
  JRA55 `…/FORCING/JRA55-do-v1.4.0/`; SSS `…/PHC2_salx.nc`; runoff `…/CORE2_runoff.nc`; chl
  `…/Sweeney/Sweeney_2005.nc`. CORE2 data (gitignored): `data/` is a symlink → `/work/.../port_jax/data`
  (holds `mesh_core2/`, `ic_core2/`, and the `{phc,jra,bulk,sss,step,ice_thermo,ice_evp,ice_full}_dump_core2/`).
- C port (algorithmic SoT): `/home/a/a270088/port2/fesom2_port/src/`. Build:
  `bash -lc 'cd …/port2/fesom2_port && source env.sh && make -C build fesom_port'` (~30 s). C
  run args: `<mesh> <out> <dt> <nsteps> <snap> <phc> <jra(year)>`. **GM source to read for 6B:
  `fesom_gm.c`/`fesom_gm.h`.** Ice env knobs (now all OFF = the C default ice-ON): `FESOM_NO_GMREDI=1`
  is the GM-off knob (for ice-only dumps); for GM dumps, drop it.
- I (Claude) drive SLURM (acct ab0995 / ab0995_gpu). Cheap C dumps → `-p compute --time=00:30:00`.

## LOCKED DECISIONS (do NOT re-litigate)
1. Use case = hybrid ML params (swap points: vertical mixing PP/KPP, eddy flux GM/Redi); seam =
   `fesom_jax/params.py`. 2. Full-fidelity, bottom-up, match the C port 1:1. 3. AD-safe by
   construction + gradient re-run at every gate. 4. Mesh = index gather/scatter over `ops.py`.
   5. Single-device now; mesh sharding Phase 8. 6. linfs on CORE2 (zstar later, C-side first).
   7. netCDF4 + scipy + jax-cuda12 in the env. 8. Phase 6 made `a_ice` PROGNOSTIC (the two
   couplings — shortwave gate + stress blend — read it). 9. (Phase 6) the EVP IC-gradient is
   stiff-but-finite (`1/delta_min`) — keep `delta_min=1e-11` (match C); trainable gradients use
   the mixing seam, not the EVP. 10. (Phase 6) the assembled multi-kernel step is climate-close
   (~1e-6, the EVP floor propagates) — gate KERNELS tight, the ASSEMBLY climate-close.

## CRITICAL GOTCHAS (full list in PORTING_LESSONS.md)
- **The C port is the spec — port it 1:1 and verify by dump.** Phase-6 scope came from reading
  `fesom_ice*.c`; do the same for `fesom_gm.c`.
- **AD masked-NaN rule (bit us 5×):** make masked lanes finite; a forward `where` doesn't stop a
  backward `0·inf`. GM slope tapering/clipping is the next AD target (safe divides + min/max).
- **A finite-but-huge gradient through a plastic/iterative solver = conditioning, not a bug** —
  gate on finiteness, confirm the trainable path (mixing seam) is well-conditioned separately.
- **`ice_cfg=None` keeps the pi/Phase-5 path bit-identical** — a new subsystem behind one static
  config arg defaulting to None is a dead compile-time branch when off.
- **Element-based beats CSR for P1-triangle FE kernels** (the ice FCT mass-matvec + cluster
  min/max went CSR-free via element gather/scatter) — check this for any GM FE assembly.
- **netCDF4 import prints a benign `ndarray size changed` ABI warning** — harmless.
- **config = the pi+CORE2 reference physics:** linfs, PP, FCT, opt_visc=7, `use_wsplit=0`,
  CG SSH (α=1), dt=500, PHC IC, JRA55+SSS+runoff, **prognostic sea ice** (Phase 6).

## WORKFLOW NOTES
- Phase 6B (GM/Redi) needs its own sub-plan (authoritative); tick `[x]`, keep its Revision Log +
  the lessons current. **Commit only when asked** (per-task commits on `main`; Phase 6 ice is
  pending a commit — ask first). **C edits → port2 branch `jax-mesh-export`, NEVER port2 main**;
  job scripts kept untracked there. **Large generated files → `/work`** (the `data` symlink).
  Cheap C jobs → `-p compute --time=00:30:00`; long C / JAX-GPU → a real QOS. All python/pytest
  via the env python; ice tests as a separate group; one CPU-JAX at a time on the login node.
  See memory [[hpc-job-file-conventions]].

Confirm you've absorbed this; ask the user whether to commit Phase 6 first; then propose the
Phase-6B (GM/Redi) sub-plan scope (read `fesom_gm.c` first), and proceed.
