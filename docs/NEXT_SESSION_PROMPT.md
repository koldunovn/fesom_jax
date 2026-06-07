# Next-session prompt — FESOM2 → JAX port (Phase 6B GM/Redi → Task G.7, GATE 6B)

Paste the block below to start the next session. **Phases 0–6 COMPLETE (GATEs 0–6) + Phase 6B
GM/Redi Tasks G.1–G.6 COMPLETE — the ENTIRE GM/Redi physics is ported and dump-verified, all
committed on `main` (4 commits).** Only **Task G.7** (the assembly + GATE 6B) remains: wire the GM
pipeline into the live `step.py`/`integrate.py` behind a static `gm_cfg=None`, gate the assembled
full-GM post-step T/S vs the C dump, run multi-day GM-ON stability (GPU), and re-run the gradient
gate **with the new `d/d(k_gm)` 2nd-ML-hook target**.

Every GM kernel matches the C dump at FP-noise level: sw_alpha_beta (bit-exact), neutral slopes
(map-class), coefficients (`d/d(k_gm)=2.03e6` flows), the streamfunction TDMA (8.9e-15) + bolus
velocity (1.1e-16), the GM driver end-to-end (`fer_uv` 2.2e-16), and the Redi terms — G7a
(1.78e-15), **G7b's 5-branch edge loop (1.07e-14)**, K33 ("just augment Kv"). The GM modules are
**standalone** (step.py is still untouched ⇒ ocean physics bit-identical, no regression).

---

We are porting the FESOM2 ocean model to JAX to build a **differentiable** ocean model for hybrid
ML (trainable NN parameterizations for vertical mixing + mesoscale eddy fluxes, trained
end-to-end). Multi-session effort. Work from `/home/a/a270088/port_jax`. Max effort.

## START HERE, in order
1. **Parent plan (source of truth across phases):** `docs/plans/20260605-fesom-jax-port.md` — the
   Revision Log (the GM/Redi sub-plan is DRAFTED there); Phase 6 outline (sea ice ✅; GM/Redi 6B
   in progress; KPP 6C next).
2. **GM/Redi sub-plan (G.1–G.6 ✅, G.7 = NEXT):** `docs/plans/20260607-fesom-jax-gmredi.md` — read
   the §0 scope (the 6 integration points), the G.7 task block, and each `✅ DONE` note (the GM
   design + the dump-gate results). **G.7 is the only `[ ]` task.**
3. **Lessons (every session):** `docs/PORTING_LESSONS.md` — esp. the **Phase 6B** entries (the
   2nd-ML-hook seam, the huge-dynamic-range slope gate, the static-geometry TDMA, the 5→3-case
   G7b collapse, the K33-augment-Kv trick, the Redi `valuesold`-vs-`values` threading). **STANDING
   RULE: append a lesson per task.**
4. **Sea-ice sub-plan (the `ice_cfg` precedent for the G.7 wiring):**
   `docs/plans/20260606-fesom-jax-phase6-seaice.md` (Task 6.6 — how `ice_cfg=None` keeps the path
   bit-identical; mirror it for `gm_cfg`).
5. **Project memory:** `/home/a/a270088/.claude/projects/-home-a-a270088-port-jax/memory/`.

## STATUS
- **Phases 0–6 (GATEs 0–6):** full pi + CORE2 + sea ice, committed on `main`.
- **Phase 6B GM/Redi G.1–G.6 COMPLETE, committed** (`c587b83` G.1-3, `2e43ed9` G.4, `c3886e9` G.5,
  `dbf40ca` G.6). Modules: `eos.compute_sw_alpha_beta`; `gm.py` (`GMConfig`, `compute_sigma_xy`,
  `compute_neutral_slope`, `init_redi_gm`, `fer_solve_gamma`, `fer_gamma2vel`, **`gm_diagnostics`**
  the driver); `gm_redi.py` (`tr_xy_elem`, `diff_ver_part_redi_expl` G7a, `diff_part_hor_redi` G7b,
  `k33_augmentation`). `params.py` += `k_gm`/`redi_kmax` (the 2nd ML-hook, `default_factory`
  defaults ⇒ backward-compatible). 5 test files (test_sw_alpha_beta/test_gm_slopes/test_gm_coeffs/
  test_gm_bolus/test_gm_bolus_adv/test_gm_redi), all green standalone.
- **C dump hooks (on port2 `jax-mesh-export`, the user's to commit):** `fesom_gm_dump`
  (all-node GM fields) + `fesom_redi_blob` (T pre/post each Redi piece) in `fesom_step.c`. Job
  `jobs/jax_gm_dump_core2.sh` (GM-ON, PP, ice-OFF; `FESOM_GM_DUMP_DIR`+`FESOM_REDI_DUMP_DIR`).
- **Dumps cached** (gitignored, on `/work` via the `data` symlink): `data/gm_dump_core2/`
  (`gm_*.f64` all GM fields incl. inputs T/S/bvfreq/hnode/hnode_new; `redi_*.f64`
  T_old/T_pre/T_g7a/T_g7b/tr_xy/tr_z; `gm_meta.txt`/`redi_meta.txt`; reader `io_dump.load_gm_dump`).
  `data/gm_step_dump_core2/core2_cdump.00000` (the per-substep dump, GM-ON, at the 7 probes — **the
  post-step T/S for the G.7 assembled gate**).

## IMMEDIATE WORK — Task G.7 (assemble + stability + gradient = GATE 6B)
Wire the GM pipeline into the live step behind a static `gm_cfg=None` arg (mirror `ice_cfg`):

1. **Add `gm_cfg=None` to `step.py` (`step`/`step_jit` `static_argnames`) + `integrate.py`.** When
   `None` ⇒ bit-identical (a dead branch; the 423-test suite unchanged). When a `GMConfig`:
   - After EOS+bvfreq+smooth: `sw_alpha,sw_beta = eos.compute_sw_alpha_beta(mesh, T, S)` (already
     ported, just CALL it in the step), then `diag = gm.gm_diagnostics(mesh, T, S, bvfreq,
     hnode_new, helem, params, gm_cfg)` → `fer_uv, slope_tapered, Ki, fer_K, fer_C`.
   - `fer_w = ale.compute_w(mesh, diag.fer_uv, helem)` (pure reuse).
   - **Bolus wrap:** pass `uv + diag.fer_uv` and `w_e + fer_w` into `tracer_adv.advect_one_fct`
     (functional — the carried `uv`/`w_e` are untouched, so the C's subtract-back is automatic).
     The FCT vertical uses `w_e` (not `w`).
   - **Redi explicit (per T AND S):** `T_adv += gm_redi.diff_ver_part_redi_expl(mesh, T_old,
     slope_tapered, Ki, hnode_new, dt)` then `+= gm_redi.diff_part_hor_redi(mesh, T_old,
     slope_tapered, Ki, hnode, hnode_new, helem, dt)`. ⚠️ **G7a/G7b read `T_old`/`S_old`** (the AB2
     pre-step tracer = the C `valuesold` = `st.T_old`/`st.S_old`), NOT the advected T. Apply the
     deltas to the post-advection `T_adv`/`S_adv`.
   - **K33 in diffusion:** `Kv_eff = Kv + gm_redi.k33_augmentation(mesh, slope_tapered, Ki)`, pass
     `Kv_eff` to `tracer_diff.impl_vert_diff` (no diffusion-kernel change — `a∝Kv[nz]`/`c∝Kv[nz+1]`).
   - `dt` for GM/Redi = the model `dt` (`fesom_phase1_dt` = 500 on CORE2).
2. **Assembled gate (closes K33's tight gate):** the full-GM post-step T/S vs
   `data/gm_step_dump_core2/core2_cdump.00000` substep-15 T/S at the 7 probes. GM is deterministic
   (no EVP-like floor), so expect TIGHT (~1e-9..1e-12, not the ice ~1e-6). `gm_cfg=None` ⇒ pi +
   Phase-5 + ice paths bit-identical (re-run the full suite on the **compute node**).
3. **Multi-day GM-ON stability (GPU):** new `scripts/core2_gm_stability_run.py` (+ `_gpu.sh`) —
   the full production CORE2 (PP + GM/Redi + sea ice) multi-day stable; GM should REDUCE spurious
   convection / smooth fronts (a physical sanity sign). Bounded vel/SSH/T/S, no NaN.
4. **Gradient gate = GATE 6B:** re-run `test_gradient*` with GM live; **add `d(SST)/d(k_gm)`** the
   2nd-ML-hook FD↔AD plateau in a smooth regime (the eddy-flux gradient path, like `d/d(k_ver)`
   proved mixing); masked-NaN `d/d(T₀)` finite + 0 on masked. Measure the backward memory (GM adds
   a per-step TDMA + the Redi scatters).

**GATE 6B (acceptance):** the CORE2 model (PP/linfs/FCT/opt_visc7 + PHC IC + JRA55/SSS/runoff +
sea ice + **GM/Redi**) reproduces the C GM-ON dump (kernels bit-exact G.1-G.6; assembled step
tight/climate-close); multi-day stable with GM doing physical work; gradient gate passes with the
new `d/d(k_gm)` target + masked-NaN clean; full suite green. **Then Phase 6C = KPP** (read
`fesom_kpp.c` first; `pp.py` ↔ a new `kpp.py` behind the mixing seam).

## THE PROVEN VERIFICATION RECIPE (used for all of G.1–G.6)
Per-kernel: feed the JAX kernel the C dump's all-node inputs (`io_dump.load_gm_dump`) → match
outputs (map ~1e-15 / scatter ~1e-12; **gate high-dynamic-range fields RELATIVE**, e.g. the
untapered slope reaches ~1e5). AD: every divide/sqrt that can vanish in a masked lane must be
FINITE (`where(d==0,1,d)` / double-`where` safe-sqrt). For the assembled model, gate kernels tight,
the assembly climate-close, and lean on masked-NaN finiteness for the gradient.

## KEY PATHS
- Working repo (git `main`, local-only, no remote): `/home/a/a270088/port_jax`. 4 GM commits on
  `main` (`c587b83`/`2e43ed9`/`c3886e9`/`dbf40ca`). **Commit only when asked** (per-task).
- **Env python (ALL python/pytest):** `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`
  → `JAX_PLATFORMS=cpu … -m pytest`. ⚠️ **Heavy / full-suite runs → a COMPUTE NODE** via
  `sbatch scripts/run_suite.sbatch` (128 physical cores `--hint=nomultithread`, 257 GB) — the
  CORE2 backprop tests (`test_gradient_core2`) **HANG on the login node** (RAM thrash). The login
  node runs ONE CPU-JAX process at a time (two → pthread OOM); use it for quick interactive
  dump-gates while the compute node runs the full suite. Ice tests as a separate group.
- GPU via SLURM: `-A ab0995_gpu -p gpu --gres=gpu:1` (A100-40, ~31.8 GB). ⚠️ stream the forcing
  per step (don't `cf.stack` the trajectory); one N-step backward per process (`jax.clear_caches()`).
- C port (algorithmic SoT): `/home/a/a270088/port2/fesom2_port/src/`. Build:
  `bash -lc 'cd …/port2/fesom2_port && source env.sh && make -C build fesom_port'` (~30 s). The GM
  dump job: `sbatch jobs/jax_gm_dump_core2.sh` (`-p compute --time=00:30:00`, 31 s).
- CORE2 mesh `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`; PHC `…/phc3.0_winter.nc`; JRA55
  `…/JRA55-do-v1.4.0/`; data symlink `data/` → `/work/.../port_jax/data` (mesh_core2/, ic_core2/,
  the `*_dump_core2/` incl. `gm_dump_core2/` + `gm_step_dump_core2/`).
- I (Claude) drive SLURM (acct ab0995 / ab0995_gpu). **C edits → port2 `jax-mesh-export`, NEVER
  port2 main** (the user's strict rule); job scripts untracked there; the C dump hooks are the
  user's to commit.

## LOCKED DECISIONS (do NOT re-litigate)
1. Hybrid-ML use case; seam = `params.py`. **GM/Redi IS the 2nd ML-hook** — `k_gm`/`redi_kmax`
   threaded NOW (default=config ⇒ bit-identical; `d/d(k_gm)=2.03e6` proven). 2. Full-fidelity,
   match the C 1:1, dump-gate. 3. AD-safe by construction + gradient re-run at every gate.
   4. `gm_cfg=None` ⇒ bit-identical (the `ice_cfg` precedent). 5. **GM/Redi is STATELESS** —
   recomputed each step from T/S/N² (no new State fields). 6. **Full-cell linfs ⇒ static vertical
   geometry** (`zbar_n=zbar`, `Z_n=Z`; verified `hnode_new==zbar thickness` bit-exact) — the
   gamma TDMA + G7a + K33 all use it. 7. Active GM namelist = a small subset (ODM95 taper, GMzexp
   depth scaling, resolution scaling, Redi=GM sync, Redi_Ktaper); the rest is dead code (§0).
   8. GM is mixing-independent (dump on PP, ice-OFF; KPP is 6C). 9. `dt` for GM/Redi =
   `fesom_phase1_dt` = the runtime 500.

## CRITICAL GOTCHAS (full list in PORTING_LESSONS.md, Phase 6B section)
- **AD masked-NaN rule:** make masked lanes finite (safe-sqrt on slope `|s|`/√bv/√tapfac/√c1;
  guard `1/(3·areasvol)`, `1/mid`, `1/helem`, `1/(areasvol·hnode_new)`).
- **`neutral_slope` has huge dynamic range (~1e5-1e6 where N²→the eps² floor) — gate it RELATIVE.**
  The tapered slope (consumed downstream) is bounded; gate it isclose with a near-zero atol floor
  (the `huge×taper→0` lane carries FMA noise ~4e-10).
- **XLA FMA / eager-vs-fused noise:** eager is bit-exact vs the C; a fused path shifts ~machine-ε
  RELATIVE — gate map-class, not bit-exact-absolute on high-dynamic-range fields.
- **G7b's 5 branches → 3 cases** `(in1,in2)` (el1-only A/D, el2-only B/E, both C). **K33 = augment
  Kv** (`Ty(nz)==Ty1(nz-1)`, pass `Kv+K33_aug` to the unchanged `impl_vert_diff`).
- **G7a/G7b read `T_old`/`S_old`** (the AB2 pre-step tracer) for their gradients, apply to the
  post-advection T. The bolus wrap = pass `uv+fer_uv`, `w_e+fer_w` (functional, no subtract).
- **config = pi+CORE2 reference physics:** linfs, PP, FCT, opt_visc=7, use_wsplit=0, CG α=1,
  dt=500, PHC IC, JRA55+SSS+runoff, prognostic sea ice, **+ GM/Redi (Phase 6B)**.

## WORKFLOW NOTES
- G.7 is the only remaining GM/Redi task → then GATE 6B → then Phase 6C (KPP, own sub-plan, read
  `fesom_kpp.c` first). Tick `[x]`, keep the Revision Log + lessons current. **Commit per-task on
  `main` when asked.** Heavy/full suite → `sbatch scripts/run_suite.sbatch` (compute node);
  interactive dump-gates on the login node (one at a time). Cheap C jobs → `-p compute
  --time=00:30:00`; GPU stability/gradient → a GPU QOS. **C edits → port2 `jax-mesh-export` ONLY.**
  See memory [[hpc-job-file-conventions]].

Confirm you've absorbed this; then proceed with Task G.7 (the assembly + GATE 6B): wire `gm_cfg`
into `step.py`/`integrate.py`, gate the assembled post-step T/S vs `gm_step_dump_core2`, run the
multi-day GM-ON GPU stability + the gradient gate (with `d/d(k_gm)`), full suite on the compute node.
