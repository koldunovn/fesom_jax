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
| IC | constant T=10, S=35 | C Phase-1 default |
| forcing | analytical wind (`fesom_forcing_analytical.c`), no heat/water flux | C default |
| dt | 100 s | CLI arg |
| steps dumped | 1..10 | `FESOM_DUMP_MAXSTEPS=10` |
| dump enable | `FESOM_DUMP_FILE=<prefix>` | → `<prefix>.00000` |

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
  At **step 1** the tracer field is horizontally constant so advection does
  nothing → upwind == FCT == the dump (clean). From step 2 they diverge at the
  scheme level; the **full multi-step T/S match is a Phase-4 (FCT) gate**. Phase-2
  upwind is otherwise checked by the constant-tracer and pure-diffusion tests.

## Secondary cross-check: the Fortran dump (NOT per-substep comparable)

The existing Fortran dump `port2/fesom2/work_pi/pi_fesom_dump.0000{0,1}` uses a
**realistic stratified IC** (PHC-like; bvfreq ~1e-4 = real stratification) and
**KPP + opt_visc=5** — a different IC *and* different physics. So it is **not**
comparable per-substep to the constant-IC C dump (density@step1 differs by O(1),
purely from the IC, not a bug). It remains a *climate-level* physics cross-check
only. The C↔Fortran agreement (realistic configs) was established during the C
port's own validation.
