# fesom-jax — a differentiable FESOM2 ocean model in JAX

A port of the [FESOM2](https://fesom.de) unstructured-grid, finite-volume ocean model to
**JAX** — a fully **differentiable**, **multi-GPU / multi-node** forward model. Its purpose is
**hybrid ML**: embedding trainable neural-network parameterizations (vertical mixing, mesoscale
eddy fluxes, …) and training them end-to-end *through the ocean dynamics* with `jax.grad`.

The complete model runs: ALE thickness, EOS + hydrostatic pressure, pressure-gradient force,
FCT tracer advection, the SSH conjugate-gradient solve, KPP vertical mixing, GM/Redi eddy
fluxes, and prognostic EVP sea ice — forced by real JRA55 atmosphere + PHC initial conditions —
in **float64**, and it is differentiable end to end (including the CG solve and the 120-subcycle
sea-ice EVP scan).

> **Status:** multi-GPU + multi-node validated (`v1.2-multinode-scaling`). Forward is N-vs-1
> correct vs single-device; gradients validated (1-step tight, multi-step mechanism). Runs
> CORE2 / farc / dars / NG5 (up to 7.4 M nodes, 64 GPU) within ~4–16 % of hand-tuned
> Kokkos-CUDA where compute-bound. See **[Performance](#performance)** and
> **[Limitations](#limitations--what-it-can-and-cant-do)**.

---

## Highlights

- **Differentiable**: `jax.grad` of any scalar loss flows back through the whole timestep to
  the physics parameters *and* the initial state / forcing — e.g. `d(sea-ice)/d(air-temperature)`.
- **Scales**: `shard_map` over a 1-D device mesh, **halo-only point-to-point exchange**
  (`ragged_all_to_all`), `jax.distributed` across nodes. Per-step communication is GPU↔GPU
  halo + reductions only; the host↔GPU transfer is one-time.
- **Competitive**: within a few-to-~16 % of Kokkos-CUDA in SYPD across the mesh ladder
  (see `docs/figures/jax_vs_kokkos_sypd.png`).
- **Gather-free output**: each GPU writes its own shard to Zarr in parallel — no rank-0 gather.

---

## Installation

The model needs JAX with float64 enabled. On **Levante** there is a ready env; elsewhere create one.

```bash
# new conda/mamba env (Python 3.12)
mamba create -y -n fesom-jax python=3.12 pip
mamba activate fesom-jax

cd /home/a/a270088/port_jax           # the repo
pip install -e ".[cuda,dev,viz]"      # GPU JAX (CUDA12 wheels) + pytest + matplotlib
# CPU-only (no GPU wheels):  pip install -e ".[dev,viz]"
```

This pulls `jax[cuda12]`, numpy, scipy, netCDF4, **zarr** (sharded output), and (viz) matplotlib.
No system CUDA module is needed — CUDA 12 + cuDNN ship as pip wheels. **Recorded working set:**
Python 3.12.13, JAX/jaxlib **0.10.1**, CUDA 12.9 wheels, numpy 2.4, netCDF4 1.7, zarr 2.18,
matplotlib 3.10. Full detail + GPU verify in **[`docs/ENV.md`](docs/ENV.md)**.

**On Levante you can skip the install** and use the existing interpreter directly:

```bash
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
$PY -c "import jax, fesom_jax; print(jax.__version__, jax.devices())"
```

> ⚠️ **float64 is mandatory.** Every entry point sets `jax.config.update("jax_enable_x64", True)`
> (`fesom_jax/config.py`). The default JAX fp32 would be ~2× faster but wrong for ocean dynamics.

---

## Repository layout

```
fesom_jax/
  mesh.py partit.py            # dense Mesh loader; FESOM dist_<NP> partition reader
  state.py params.py config.py # State pytree; physics params (the ML hooks); constants
  step.py integrate.py         # one ocean timestep; checkpointed multi-step scan (single device)
  ale.py eos.py pgf.py pp.py   # thickness / equation-of-state / pressure-gradient / PP mixing
  momentum.py tracer_adv.py tracer_diff.py   # momentum; FCT tracer advection; diffusion
  ssh.py                       # SSH stiffness operator + CG solve (custom_linear_solve, AD-safe)
  kpp.py gm.py gm_redi.py      # KPP vertical mixing; GM bolus + Redi eddy fluxes
  cvmix_tke.py tke.py          # CVMix classical-TKE prognostic mixing (opt-in tke_cfg; Phase 9b)
  ice*.py                      # prognostic sea ice: EVP + mEVP (ice_mevp.py) dynamics, FCT, thermo
  # Phase-9 differentiable options, each config-gated (None/0 ⇒ byte-identical):
  #   zstar moving coordinate (ale_cfg), classical-TKE mixing (tke_cfg), mEVP rheology (whichEVP=1)
  forcing.py core2_forcing.py jra55.py phc_ic.py sss_runoff.py   # bulk fluxes; JRA55; PHC IC; SSS
  halo.py halo_points.py reductions.py        # halo exchange (all_gather + ragged); psum reductions
  shard_mesh.py integrate_sharded.py          # per-device sharded mesh/state; shard_map runners
  zarr_output.py               # sharded, gather-free model output to Zarr
  tests/                       # ~55 verification + gradient + sharding gates
scripts/                       # benchmarks + SLURM sbatch (forward scaling, gates, plots)
docs/                          # ENV, plans, porting lessons, scaling, the ragged-bug record
data/  mesh_core2/ ic_core2/   # CORE2 dense mesh + cached PHC IC (small, in-repo)
```

---

## Quick start

### 1. Forward — single device (CORE2, ocean-only)

```python
import jax, jax.numpy as jnp
from fesom_jax.mesh import load_mesh
from fesom_jax.ssh import build_ssh_operator
from fesom_jax.phc_ic import core2_initial_state
from fesom_jax.integrate import integrate

mesh   = load_mesh("data/mesh_core2")               # dense Mesh pytree (float64)
op     = build_ssh_operator(mesh, dt=1800.0)        # static SSH stiffness (built once)
state0 = core2_initial_state(mesh, "data/ic_core2") # PHC winter IC (cached)
stress = jnp.zeros((mesh.elem2D, 2))                # element wind stress

state_N = integrate(state0, mesh, op, stress, n_steps=10, dt=1800.0)
print(state_N.T[:, 0].mean())                        # surface temperature after 10 steps
```

**Full model** (KPP + GM/Redi + prognostic ice + real JRA55) — add the configs + forcing
(`step()` takes one step's forcing; for forced *multi*-step use `run_steps_sharded`, or pass
`integrate(step_forcings=...)` a per-step stack):

```python
from fesom_jax.step import step
from fesom_jax import core2_forcing, ice
from fesom_jax.kpp import KppConfig; from fesom_jax.gm import GMConfig; from fesom_jax.ice import IceConfig
sst0   = state0.T[:, 0]
state0 = ice.seed_ice(state0, mesh, sst0)            # cold-start ice
cf     = core2_forcing.build_core_forcing(mesh, 1958, sst_ic=sst0)             # JRA55 1958
sf, fs = cf.step_forcing(*core2_forcing.dates_for_steps(1958, 1800.0, 1)[0]), cf.static
state_1 = step(state0, mesh, op, stress, dt=1800.0, is_first_step=True,
               step_forcing=sf, forcing_static=fs,
               kpp_cfg=KppConfig(), gm_cfg=GMConfig(), ice_cfg=IceConfig())
```

### 2. Backward — gradients (the whole point)

```python
import jax
from fesom_jax.params import Params

def loss(params):                                    # any scalar functional of the state
    sN = integrate(state0, mesh, op, stress, n_steps=1, params=params, dt=1800.0)
    wet = mesh.node_layer_mask[:, 0]
    return jnp.sum(jnp.where(wet, sN.T[:, 0], 0.0)) / jnp.sum(wet)   # mean SST

g = jax.grad(loss)(Params.defaults())                # d(mean SST) / d(every physics param)
print(g.k_ver, g.a_ver, g.k_gm, g.redi_kmax)         # mixing / eddy sensitivities
```

`jax.grad` works w.r.t. **params** (the ML hooks), the **initial state**, and the **forcing** (e.g.
`d(a_ice)/d(air-temperature)`) — flowing through the CG solve and the EVP scan. See
`fesom_jax/tests/test_gradient_sharded.py` for validated examples. (Read
**[Limitations](#limitations--what-it-can-and-cant-do)** on the useful time horizon.)

### 3. Multi-GPU / multi-node forward

Python API (shard a global state across `npes` devices and step):

```python
import numpy as np
from fesom_jax import partit, shard_mesh, ssh
from fesom_jax import integrate_sharded as ish

part    = partit.read_partition("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars", 8)  # dist_8
sm      = shard_mesh.build_sharded_mesh(mesh, part)
state_p = shard_mesh.partition_state(state0, part)
sop     = ssh.partition_ssh_operator(op, part)
stress_p = np.zeros((8, sm.Lmax["elem"], 2))
final = ish.run_steps_sharded(sm, state_p, sop, stress_p, n_steps=25, dt=180.0,
                              npes=8, use_ragged=True)   # ragged halo (forward); see limitations
```

Multi-node uses `jax.distributed` (1 process/node × 4 GPU) — set `JDIST=1` and launch with `srun`;
the runner places each process's shards via `make_array_from_callback` (no global array on one GPU).
The benchmark driver handles all of this:

```bash
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
# single node, CORE2 full model, ragged, timed:
$PY scripts/bench_forward_scaling.py --mesh-dir data/mesh_core2 \
    --dist-dir /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 --npes 4 --steps 25 \
    --dt 1800 --full 1 --ic-dir data/ic_core2 --ragged 1 --out-zarr /work/.../core2_out.zarr
# multi-node: submit the prebuilt sbatch (sets JDIST, srun, paths) — see below.
```

---

## Data & paths on Levante

| What | Path |
|------|------|
| Env interpreter | `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python` |
| GPU partition / account | `-p gpu -A ab0995_gpu` (A100-80GB ×4/node) |
| CORE2 dense mesh + IC | `data/mesh_core2/`, `data/ic_core2/{T,S}_ic.npy` (in repo) |
| farc/dars/NG5 mesh bundles | `/work/ab0995/a270088/fesom_jax_meshes/mesh_{farc,dars,ng5}/` |
| farc/dars/NG5 cached PHC IC | `/work/.../mesh_{dars,ng5}/{T,S}_ic.npy` |
| dist_`<NP>` partitions | `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/{core2,farc,dars,ng5}/dist_<NP>` |
| PHC winter IC (source nc) | `/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc` |
| JRA55-do forcing | `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0` |

Meshes are dense `.npy` bundles exported from the C port (`docs/MESH_EXPORT_LAYOUT.md`). Cache a
mesh's PHC IC once with `scripts/cache_phc_ic.py --mesh-dir <M> --out-dir <M>` (slow at NG5 scale).

⚠️ The C IC is **partition-dependent** (the `extrap_nod3D` Gauss-Seidel land fill is
order-dependent and runs per-rank), so an IC meant to match a C dump oracle bit-for-bit must be
built with that run's partition. Two CORE2 caches coexist: `data/ic_core2` = the **serial**
(1-rank) order (`cache_phc_ic.py`; the legacy core2/kpp/gm/ice oracles were 1-rank runs) and
`data/ic_core2_dist16` = the **dist_16** order (`scripts/rebuild_ic_dist16.py`; matches the
16-rank `z2_cdump` zstar oracle).

**Mesh ladder + per-mesh timestep** (CFL — `dt` is mesh-specific):

| mesh  | nodes  | levels | dt (cold) | min nodes (JAX) |
|-------|--------|--------|-----------|-----------------|
| CORE2 | 127 k  | 48     | 1800 s    | 1 node (4 GPU)  |
| farc  | 638 k  | 48     | 900 s     | 1 node          |
| dars  | 3.16 M | 57     | 180 s     | **2 nodes**     |
| NG5   | 7.4 M  | 70     | 180 s     | **8 nodes**     |

---

## Reproducing the benchmarks

Prebuilt SLURM scripts in `scripts/` (each loops the relevant `--npes` / halo / mesh):

```bash
sbatch scripts/bench_core2_2node.sbatch     # CORE2 2 nodes (ragged + allgather)
sbatch scripts/bench_dars_dist16.sbatch     # dars 4 nodes
sbatch scripts/bench_ng5_dist32.sbatch      # NG5 8 nodes (+ sharded zarr output)
sbatch scripts/bench_ng5_dist64.sbatch      # NG5 16 nodes
```

Conventions: GPU jobs use `-p gpu -A ab0995_gpu`; **keep `--time` ≤ 30 min** so short jobs backfill
(16-node allocations can wait hours otherwise). Each job compiles once and times a warm reuse
(XLA compile excluded). The full-model XLA compile is ~30 s (CORE2) to ~2 min (NG5).

Regenerate the SYPD comparison plot:

```bash
$PY scripts/plot_sypd.py     # -> docs/figures/jax_vs_kokkos_sypd.png
```

---

## Performance

Full model (real JRA+PHC+ice), ragged halo, A100, vs **Kokkos M5.24 CUDA** (`SCALING_M524.md`):

| mesh / scale | JAX s/step | Kokkos s/step | gap | JAX SYPD (prod. dt) |
|---|---|---|---|---|
| CORE2  1N (4 GPU)  | 0.087 | 0.117 | **JAX faster** | 56.9 |
| CORE2  2N (8 GPU)  | 0.123 | 0.095 | +29 % | 40.2 |
| dars   2N (8 GPU)  | 0.934 | 0.814 | +15 % | 0.68 |
| dars   4N (16 GPU) | 0.513 | 0.475 | +8 %  | 1.24 |
| NG5    8N (32 GPU) | 0.840 | 0.810 | +4 %  | 0.76 |
| NG5    16N (64 GPU)| 0.584 | 0.492 | +16 % | 1.09 |

`SYPD = dt / (365 · s_per_step)` at the production dt (CORE2 1800 / farc 900 / dars,NG5 240 with a
×1.03 cold-start correction — the Kokkos convention). **Plot:** `docs/figures/jax_vs_kokkos_sypd.png`.

Pattern: **compute-bound → JAX competitive-to-faster; comms-bound → Kokkos's hand-tuned MPI wins**
(the gap widens as the per-GPU shard shrinks). The ragged halo is *necessary* multi-node — at
dars-8 / NG5 the `all_gather` halo OOMs while ragged fits.

---

## Testing / verification

Sharding tests need CPU "fake devices" (`--xla_force_host_platform_device_count=N`) and a compute
node (not the login node). `ragged_all_to_all` is **GPU-only** (see limitations), so ragged gates
run only on real GPUs.

```bash
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
$PY -m pytest fesom_jax/tests/                                  # single-device gates
$PY -m pytest fesom_jax/tests/ -k "verify"                      # per-substep vs Fortran dumps
$PY -m pytest fesom_jax/tests/ -k "gradient"                    # AD checks
# sharding (CPU fake-devices, compute node):
XLA_FLAGS=--xla_force_host_platform_device_count=4 $PY -m pytest fesom_jax/tests/test_step_sharded.py
sbatch scripts/run_suite.sbatch                                 # full suite on a compute node
```

---

## Limitations — what it can and can't do

### Can
- **Full forward ocean** (ALE/EOS/PGF/FCT/SSH-CG/KPP/GM-Redi/EVP-ice) with real JRA55 + PHC,
  float64; N-vs-1 correct vs single-device (field-appropriate; see below).
- **End-to-end gradients** w.r.t. physics params, initial state, and forcing — through the CG
  solve and the EVP scan. 1-step gradient validated tight; multi-step mechanism validated.
- **Multi-GPU + multi-node** (CORE2→NG5, up to 64 GPU), halo-only ragged comms, sharded Zarr output.

### Can't (yet) / caveats
1. **Ragged-halo gradients are broken** — JAX's `lax.ragged_all_to_all` has an incorrect
   reverse-mode *transpose* (forward is byte-exact; the adjoint over-counts ~`axis_size`×). So
   `use_ragged=True` is **forward-only**. For gradients use `use_ragged=False` (the `all_gather`
   halo — correct AD, but O(P·N_local) communication). Full record + minimal repro + the planned
   `custom_vjp` fix: **[`docs/JAX_RAGGED_A2A_BUG.md`](docs/JAX_RAGGED_A2A_BUG.md)**.
2. **Gradient time horizon ≈ predictability window.** The model is chaotic: gradients are clean and
   useful over **hours→weeks** (forcing→fast-variable sensitivities — ice, SST, mixed layer), but a
   free-running trajectory's gradient **explodes exponentially** after the Lyapunov/predictability
   time (months→years). Long-window training needs shadowing-type methods (the differentiable-chaos
   frontier), not just more memory.
3. **Big meshes need a minimum node count** (memory). dars **≥ 2 nodes**, NG5 **≥ 8 nodes** — the
   compiled full-step working set (XLA rematerialization floor) is ~4× heavier per node-level than
   Kokkos's hand-managed memory, so JAX needs *more* nodes than Kokkos for the biggest mesh
   (single-node dars/NG5 OOM; dars-1-node and NG5-2-node do not fit).
4. **`ragged_all_to_all` is GPU-only** — unimplemented on XLA:CPU. CPU correctness gates use the
   `all_gather` halo; all ragged scaling is GPU/NCCL.
5. **Not bit-identical.** "Climate-close, not bit-exact": FCT upwind-flip and reduction-order
   reassociation make N-vs-1 and free-running multi-step diverge at the ~1e-12 floor (same as the
   C/Kokkos ports). Forward gates are field-appropriate, not bit-exact.
6. **Comms-bound at high node count / tiny mesh.** XLA/NCCL collective overhead doesn't overlap as
   tightly as hand-tuned MPI; the gap to Kokkos widens as the per-GPU shard shrinks (CORE2 doesn't
   scale past ~1–2 nodes).
7. **No long bit-reproducible climate run.** A multi-GPU 2-yr run diverges by reduction order
   (chaotic) — same as the C; a separate follow-up, out of scope here.
8. **Forced gradient at large scale** is memory-bounded by the EVP-scan backward (validated on CPU
   fake-devices + GPU ocean grad; the full forced grad on many GPUs is deferred).

---

## Documentation map

| Doc | What |
|-----|------|
| [`docs/plans/20260605-fesom-jax-port.md`](docs/plans/20260605-fesom-jax-port.md) | roadmap + locked design decisions (source of truth) |
| [`docs/plans/20260608-fesom-jax-phase8b-scaling.md`](docs/plans/20260608-fesom-jax-phase8b-scaling.md) | multi-GPU/multi-node scaling phase (the scaling work above) |
| [`docs/PORTING_LESSONS.md`](docs/PORTING_LESSONS.md) | per-task gotchas & hard-won facts (AD seams, sharding, fidelity) |
| [`docs/ENV.md`](docs/ENV.md) | exact environment + GPU verification |
| [`docs/JAX_RAGGED_A2A_BUG.md`](docs/JAX_RAGGED_A2A_BUG.md) | the ragged AD bug: record, repro, workaround |
| [`docs/MESH_EXPORT_LAYOUT.md`](docs/MESH_EXPORT_LAYOUT.md) | the dense mesh `.npy` bundle format |

### Reference ports (algorithmic / numerical sources of truth)
| What | Where |
|------|-------|
| C MPI port | `/home/a/a270088/port2/fesom2_port/src/` (kernel-by-kernel source) |
| Fortran FESOM2 | `/home/a/a270088/port2/fesom2/src/` (per-substep numerical dumps) |
| Kokkos port | `/home/a/a270088/port_kokkos/` (parallelization + the `SCALING_*.md` numbers) |

## Key principles

- **Golden rule:** preserve the *exact* computation (math + load-bearing association order),
  expressed as vectorized array ops — no loop-by-loop translation, no physics simplification.
- **AD-safe by construction:** pure functional JAX, float64 everywhere, `lax.scan/cond/while`
  instead of Python control flow on traced values; gradient checks re-run at every gate.
- **Fidelity target:** climate-close, not bit-identical (FP reassociation in scatters/reductions);
  ~1e-15 for map/gather kernels, ~1e-12 for scatter/reduction kernels — which does not hurt AD.
