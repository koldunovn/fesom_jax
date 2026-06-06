# Reference runs — the per-substep verification oracle

## What is the oracle (decision: Path A)

The JAX port verifies each ported kernel against a **per-substep reference dump
produced by the C port** (`fesom2_port`), which is the JAX port's *algorithmic*
source of truth. Because JAX mirrors the C algorithm, JAX↔C per-substep diffs are
pure floating-point reassociation (~1e-15 map/gather, ~1e-12 scatter) — the
tightest, most diagnostic gate.

Verification chain: **JAX ↔ C** (this dump → ports the algorithm correctly) +
**C ↔ Fortran** (validated previously, climate-close, realistic configs → the
physics is correct) ⟹ JAX is correct.

The C-port dump writer is `fesom2_port/src/fesom_dump.c` (branch
`jax-mesh-export`), env-gated and modelled on the Fortran `fesom_dump_shim.F90`
(same binary record layout, read by `fesom_jax/io_dump.py`). Unlike the Fortran
shim (node fields only) it also dumps the **element** fields (pgf, uv_rhs, uv, Av).

## How to regenerate the pi reference dump

```bash
# build (module toolchain — NOT conda)
source /home/a/a270088/port2/fesom2_port/env.sh
cmake --build /home/a/a270088/port2/fesom2_port/build -j 8

# run: pi, npes=1 (global mesh), PP + no GM + linfs + opt_visc7, dt=100, 10 steps
sbatch /home/a/a270088/port2/fesom2_port/jobs/jax_cdump_pi.sh
# → port_jax/fesom_jax/tests/fixtures/pi_cdump.00000
```

Config (must match the JAX target exactly):

| knob | value | how |
|---|---|---|
| mesh | pi (`port2/fesom2/test/meshes/pi`), nl=48 | CLI arg |
| ranks | npes=1 → global mesh | `srun -n 1` |
| ALE | linfs | C compile-time default |
| mixing | PP (Pacanowski-Philander) | `FESOM_MIX_SCHEME=PP` |
| GM/Redi | off | `FESOM_NO_GMREDI=1` |
| horizontal visc | opt_visc=7 (biharmonic) | C default (`visc_filt_bidiff`) |
| IC | constant T=10, S=35 **+ a Gaussian T-blob** (see below) | C default when no PHC path |
| forcing | analytical wind (`fesom_forcing_analytical.c`), no heat/water flux | C default |
| dt | 100 s | CLI arg |
| steps dumped | 1..10 | `FESOM_DUMP_MAXSTEPS=10` |
| dump enable | `FESOM_DUMP_FILE=<prefix>` | → `<prefix>.00000` |

### ⚠️ The IC is constant **plus a T-blob** (not bare constant)

`fesom_main.c:744-753`, when given no PHC path (the dump run gives none), applies
`fesom_ic_tracer_T_blob` *on top of* the constant T=10/S=35 IC: a Gaussian +5 °C
temperature anomaly, centre `(lon0,lat0)=(−45°,40°)` geographic, horizontal
`σ_h=10°` (with a `cos(lat0)` small-circle correction and a **4σ cutoff**:
`if r²_h > 16: continue`), vertical `σ_z=300 m` about `z=Z[nz]` (negative
downward), added additively to T on every wet layer; **S stays 35**
(`fesom_ic.c:82-129`). So node 1001 sits inside the blob (stratified → bvfreq≠0)
and node 3000 outside (T=10 → bvfreq=0). **Any kernel verified against this dump
whose result depends on T/S must reproduce the blob**, not the bare constant. T/S
are effectively frozen over the 10 dumped steps (weak wind-only flow), so the
substep-1 EOS fields are step-independent in this config.

## Probes

Node probes (pinned, global 1-based ids): **1001, 1500, 2000, 2500, 3000**
(`fesom_dump.c`, same as the Fortran shim). Element probes = the first cell
incident to each node probe — on this pi mesh that is **1757, 2656, 3688, 4604,
5575**. Element records carry the *element* gid in `probe_global_id`; node
records carry the node gid. Always truncate a JAX column to the record's
`nlevels` before diffing.

## Which substeps are usable when

All substeps are dumped at all 10 steps, but the JAX→C match holds over different
windows depending on the kernel:

- **Substeps 1–13, 16 (EOS, PGF, mixing, momentum, SSH, vel, hbar, eta, ALE,
  thickness):** upstream of tracer advection, so **match at every step** —
  independent of the upwind-vs-FCT choice.
- **Substep 15 (T, S):** the C port runs **FCT**; JAX Phase 2 runs **upwind**.
  ⚠️ **CORRECTION (Task 2.10):** the IC is constant + a Gaussian **T-blob**, so `T`
  is **not** horizontally constant — only `S=35` is. Consequences at **step 1** (uv
  is the wind-driven velocity, so advection is active):
  - **`S`** advects trivially (constant ⇒ the transport divergence cancels by
    discrete continuity, and diffusion sees no gradient) ⇒ upwind == FCT == the dump
    **bit-for-bit**. This is the clean step-1 tracer gate.
  - **`T`** (the blob has horizontal+vertical gradients) ⇒ upwind ≠ FCT; the dump's
    FCT `T` differs from upwind `T` by the limited antidiffusive flux (~3e-7 at
    step 1). So the **tight `T` match is a Phase-4 (FCT) gate**; Phase-2 upwind `T` is
    verified against an independent numpy upwind loop reference + the constant-tracer
    property, and only *bounded* (`<1e-5`) against the dump.
  From step 2 even `S` diverges at the scheme level. (`test_tracers.py`.)

## CORE2 reference runs (Phase 5)

The same Path-A recipe on the **CORE2** mesh (`/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2`,
nod2D=126858, nl=48) with **PHC IC + JRA55 + SSS-restoring + runoff** forcing. Two artifacts:

**(1) Per-substep dump** (`data/step_dump_core2/core2_cdump.00000`, gitignored on `/work`),
regen via `port2/fesom2_port/jobs/jax_step_dump_core2.sh` (matched config: `FESOM_MIX_SCHEME=PP
FESOM_NO_GMREDI=1 FESOM_NO_ICE_DYN/ADV/THERMO=1 FESOM_BULK_FIXED_ITERS=1`, dt=500, 3 steps).
Probes: 7 node gids `{1001,33778,43828,61202,66921,79663,94122}` (incl. the **Aleutian-Trench
watch node 94122**) + their 7 incident-element gids `{307,747,25954,61526,99096,110065,154575}`
(element fields land here). Gated by `tests/test_core2_step.py`:
- **step-1 T/S** post-step — *bit-exact class* (~7e-15 / 2e-14): the comprehensive whole-step gate.
- **step-1 per-substep dynamics** (`test_step1_dynamics_per_substep`): pressure / PGF / Av /
  uv_rhs / ssh_rhs / d_eta / uv / hbar / eta_n / w / hnode — bit-exact class (JAX & C share the
  identical PHC IC). Large intermediates `ssh_rhs` (~1e5) / `pressure` (~5e5) match ~1e-11 *relative*.
- **steps 2-3 evolution** (`test_evolution_steps23`): T/S/uv/d_eta stay ~1e-6 — the discrete CG
  iteration count + FCT amplify the step-1 ~1e-15; bounded, confirming the loop-carried forcing
  + AB2 history + warm-started CG thread correctly.

**(2) Stability arbiter run** (no dump — the per-step monitor only), regen via
`port2/fesom2_port/jobs/jax_core2_stability.sh` (same config + `FESOM_PRINT_EVERY=36`; give it
a non-debug QOS — single-rank C is ~4.6 s/step). The C prints per-step max|uv|/max|eta|/T-range/
fluxes. **Used to cross-validate the JAX multi-day trajectory:** `scripts/core2_stability_run.py`
(JITTED, A100 ~0.06 s/step; `core2_stability_gpu.sh`) runs the assembled JAX model with the same
forcing and JAX **tracks the C to ~3 sig figs** on SST_min / max|uv| / max|eta| (step 216:
−6.60=−6.60, 1.389≈1.39, 2.715≈2.71) — robust global reductions track even though per-element
values diverge chaotically (the step-1 bit-exact match degrades to ~1e-6 by step 3).

**Stability verdict (Task 5.7):** numerically stable **days 1–7** (no NaN, max|vel|<3, |SSH|<5);
the no-ice run then **supercools without bound** (JAX SST −1.9 IC → −22.8 day 8; the matched C
tracks this to 3 sig figs through the **verified window ~day 2.3 / step 396**, same mechanism —
the longer C run was cancelled, so the day-8 value is JAX's), and past the EOS-valid range
(~−20 °C) the spurious density field destabilizes the dynamics at day ~8. A *physical* no-ice
limitation (sea ice, Phase 6, caps it), **not** a numerical blow-up and **not** the "C blows up
⇒ move ice to Phase 5" trigger (the C is stable + tracks JAX through the verified window).

## Secondary cross-check: the Fortran dump (NOT per-substep comparable)

The existing Fortran dump `port2/fesom2/work_pi/pi_fesom_dump.0000{0,1}` uses a
**realistic stratified IC** (PHC-like; bvfreq ~1e-4 = real stratification) and
**KPP + opt_visc=5** — a different IC *and* different physics. So it is **not**
comparable per-substep to the constant-IC C dump (density@step1 differs by O(1),
purely from the IC, not a bug). It remains a *climate-level* physics cross-check
only. The C↔Fortran agreement (realistic configs) was established during the C
port's own validation.
