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
- **Calibratable & learnable**: that gradient drives adjoint sensitivity maps, perfect-model +
  real-obs parameter calibration, and a learned (NN) TKE closure — see
  [Differentiable capabilities](#differentiable-capabilities).
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
  longwindow.py                # ensemble-averaged (climate-timescale) adjoint seam: seed-spread + streaming mean+SE
  tests/                       # ~55 verification + gradient + sharding gates
scripts/                       # benchmarks + SLURM sbatch + differentiable-capability drivers (core2_paper_*, core2_lw_*)
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

**Full model** (KPP + GM/Redi + prognostic ice + real JRA55) — add the configs + forcing.
`step()` takes one step's forcing; for a forced *multi*-step run use `run_steps_sharded` (per-step
forcing) or `integrate(step_forcings=...)` (a pre-stacked stack — simplest, but it holds all forcing
in memory and so caps at **~weeks**). For **multi-year** forwards use the per-year-chunked driver
`scripts/core2_kpp_climate_run.py` (`--load-state` restart, `is_first_step=False`) — the tool that
produced the 5-yr spin-up + 10-yr climate reference:

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

### 2b. Reverse-mode (adjoint) vs forward-mode (tangent-linear)

`jax.grad` above is **reverse-mode** (the adjoint): one backward pass gives `d(one scalar)/d(all
inputs)` — the right tool for **few outputs, many inputs**, e.g. a `[nod2D]` parameter *field* → one
global metric (`∂(mean MLD)/∂c_k(x)`). For the **mirror case** — **one scalar input → a field of
outputs**, the spatial *response* to a single global knob — use **forward-mode** (`jax.jvp`, the
tangent-linear model): one forward sweep carries the tangent alongside the primal, no tape.

```python
ck0 = jnp.float64(Params.defaults().tke_c_k)          # one global scalar knob
mld_field, dmld_dck = jax.jvp(                         # forward-mode: d(MLD field)/d(c_k)
    lambda ck: window_mean_mld_field(ck, state0), (ck0,), (1.0,))
```

**Rule of thumb — pay for the small side:** reverse mode costs about one extra model run per
*output*, forward mode about one per *input* — so pick the one whose count is smaller (many params →
one number: reverse; one knob → a whole map: forward). They are two views of the same underlying
calculation, so they agree where they overlap (checked to **0.7 %**) and run into the same
short-window limit (see Limitations). The long-window driver exposes both via `--mode {adjoint,tlm}`
(`scripts/core2_lw_avgadj.py`).

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

## Differentiable capabilities

What the end-to-end gradient is *for*. Three capabilities — built on `jax.grad` of the global
model, plus a gradient-free **EKI** (ensemble Kalman inversion) for the slow targets the adjoint
can't reach — each proven **perfect-model-first, then applied to real OMIP-style obs** (MLD vs de
Boyer Montégut / WOA, T/S vs WOA, sea-ice vs OSI-SAF). Drivers: `scripts/core2_paper_*.py`;
headline figures: `scripts/fig_{sensitivity,calibration,hybridml}.png`.

- **Sensitivity maps** (`fig_sensitivity.png`). One backward pass promotes a scalar physics
  parameter to a `[nod2D]` field leaf → an adjoint sensitivity map: `∂(mean MLD)/∂(c_k field)`
  peaks in the Weddell/Labrador deep-convection regions, `∂(upper-ocean T)/∂(k_gm field)` in the
  polar eddy band. Adjoint==FD to ~7 digits, and cross-checked against an EKI forward-ensemble
  estimate of the same `k_gm` sensitivity (agree to 6.6 %). Honestly labelled as the
  *fast/instantaneous* (~6-h-window) sensitivity, not the equilibrium.

- **Calibration** (`fig_calibration.png`). Perfect-model twins recover injected parameters through
  the global adjoint — `k_gm` 800→1500 (0.075 %), `tke_c_k` (0.18 %). On real obs, the TKE
  `c_k`→WOA-MLD calibration reduces the MLD misfit and *generalizes*: a random held-out fold
  improves as much as train (not overfitting), while a blocked-region split exposes the honest
  limit that one global scalar can't fix a spatially-structured bias. The **slow** GM→T/S target
  uses EKI (forward-only ⇒ immune to the adjoint's chaos/memory ceiling *and* to the
  sea-ice-rheology adjoint instability) — the EKI twin recovers `k_gm` to 0.034 %. Adjoint and EKI
  agree on the shared scalar, so each is used where it is correct. Calibrated scalars transfer to
  the operational Fortran `namelist.oce` (`scripts/write_namelist.py`).

- **Hybrid-ML — a learned TKE closure** (`fig_hybridml.png`). A small structure-preserving MLP maps
  local column features → a bounded multiplier on `c_k/c_eps/c_d` (bounded ⇒ positive-definite
  diffusivities; **NN→0 ⇒ default TKE bit-identical**, the deployment fallback). Trained end-to-end
  through the global adjoint, it recovers a known NN-twin's evolution + bulk mixing field, and
  reduces *real held-out* MLD misfit (−2.1 %, = train ⇒ no overfit). **Honest finding (the
  offline/online gap):** the short-window optimum deploys *stably* on a 90-day forward (a
  trust-region regulariser keeps drift ≈ default) but its obs benefit does **not** persist — the
  short-window adjoint optimises the fast MLD response, misaligned with the slow deployed
  equilibrium (the same adjoint↔EKI boundary). A held-out short-window obs reduction is necessary
  but not sufficient for a deployable closure, so the long-forward drift+persisted-benefit gate is
  essential.

- **Climate-timescale sensitivity — the ensemble-averaged adjoint** (`scripts/core2_lw_avgadj.py`,
  `fig_avgadj.py`). A single burst's gradient is clean only for hours→days (chaos, below); to reach
  the **10-yr-mean** response, *don't* backprop a long window — **average many short frozen-ice
  adjoint bursts** seeded along a 10-yr reference trajectory (Lea/Allen/Haine): the chaotic part
  cancels, the slow climate signal survives. Gives `d(10-yr-mean MLD)/d(c_k) = +1.46 ± 0.50 m` (more
  mixing deepens the climate-mean MLD — the calibration sign, now at the climate horizon), with the
  full `[nod2D]` map, across-burst SE, and a **MAD robust filter** that drops the few summer-convection
  bursts that blow up before the others. Runs in both `--mode {adjoint, tlm}` (the *where-to-tune* map
  and the *spatial-fingerprint* response). The slow GM→interior-T target (`d(mean 0–100 m T)/d(k_gm)`)
  is reachable but small at a 1-day window — the adjoint↔EKI boundary again, pending finite-difference
  validation against two 10-yr `k_gm ± δ` forwards.

**Run them** — each driver has an `.sbatch` sibling with the Levante GPU directives, and `--help` lists
its knobs. Heavy state goes to `/work`; results are small `scripts/*.jsonl` + `*_map.npz` (gitignored):

```bash
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
$PY scripts/core2_paper_sensitivity.py  --help     # Fig 2 — instantaneous adjoint sensitivity maps
$PY scripts/core2_paper_calib_twin.py    --help     # perfect-model calibration (k_gm / c_k recovery)
$PY scripts/core2_paper_nn_twin.py       --help     # learned-TKE hybrid-ML twin
# climate-timescale sensitivity (needs a 10-yr reference of state snapshots), reverse- OR forward-mode:
$PY scripts/core2_lw_avgadj.py --mode adjoint --target mld_ck \
    --snap-dir /work/ab0995/a270088/port_jax/longwindow/ref10_snaps --K 200 --n 48
$PY scripts/fig_avgadj.py --maps scripts/lw_avgadj_mld_ck_adjoint_map.npz   # render the climate map
```

> These are research capabilities on the CORE2 (127 k-node) configuration, not a turnkey DA
> product. The honest limits — the chaotic gradient horizon, the adjoint↔EKI split, the
> offline/online closure gap — are reported as first-class findings, not hidden.

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

**Forward model — what runs, what doesn't, and why** (the forward integration; the gradient modes are
the next table):

| forward configuration | runs? | what it means in plain terms |
|---|---|---|
| Single-device, ocean-only **or full** model (KPP+GM/Redi+ice+JRA55), float64 | ✅ | the normal way to run it on one GPU — see Quick start |
| Forced multi-step, **pre-stacked** (`integrate(step_forcings=…)`) | ✅ but **≤ ~weeks** | the simple path loads every timestep's atmospheric forcing into memory up front, so it runs out of room after a few weeks of simulated time |
| **Multi-year** forced forward | ✅ | a separate driver (`scripts/core2_kpp_climate_run.py`) feeds the forcing one year at a time and saves/reloads the model state, so it can run for years — this is how we made the 5-yr spin-up and the 10-yr reference |
| Multi-GPU / multi-node, **ragged** halo | ✅ **GPU only** | the fast, lean way the GPUs swap their shared edges only exists on GPUs, not CPUs (#4) |
| Multi-GPU, **all_gather** halo | ✅ CPU **and** GPU | works anywhere (CPU too), but every GPU has to hold a copy of all the others' edge data, so the biggest meshes run out of memory (#1) |
| Big mesh on too few nodes | ❌ out of memory | the model needs more memory per node than the original C/Fortran, so the largest meshes only fit if spread over enough nodes: dars needs **≥ 2 nodes**, NG5 **≥ 8** (#3) |
| Long **bit-for-bit repeatable** run | ❌ | the ocean is chaotic and tiny rounding differences (from adding numbers in a different order across GPUs) grow over time, so two runs never end up identical — exactly like the original models (#5, #7) |

**Differentiation modes — status, limits, and the reason** (the forward model runs at every scale
above; these are the constraints on the *gradient* modes):

| mode | status | what it means in plain terms |
|---|---|---|
| Reverse-mode **adjoint**, single-GPU | ✅ used everywhere | the gradient is only trustworthy over short windows because the ocean is chaotic (#2); the sea-ice part of the gradient blows up soonest, so we switch it off in the gradient beyond ~1 day (#9) |
| Reverse-mode adjoint, **sharded** (multi-GPU) | ❌ wrong gradient | a bug in JAX's multi-GPU data exchange makes the multi-GPU gradient wrong (the forward is fine), so every gradient runs on **one** GPU — we get scale by launching many separate jobs instead (#1) |
| Forward-mode **TLM** (`jax.jvp`), single-GPU | ✅ | the same calculation run forwards instead of backwards; handy when you turn **one** knob and want the whole map of its effects. It hits the same short-window limit as the adjoint (it's the same gradient, the other way round) |
| **Ensemble-averaged** climate adjoint | ✅ research | a single long gradient would blow up, so we average many short ones taken along a long run (the chaos cancels out); we freeze sea ice in the gradient and throw out the few short runs that still blew up early |
| **EKI** (gradient-free) | ✅ | for slow effects a short gradient can't see, an ensemble method that needs no gradient at all — more expensive, and it gives a single number rather than a full map (#10) |

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
2. **Gradient time horizon ≈ predictability window (both adjoint AND tangent-linear).** The model is
   chaotic: a single burst's gradient is clean over **hours→days** (forcing→fast-variable
   sensitivities — ice, SST, mixed layer) but **explodes exponentially** past the Lyapunov time (for
   the all-on CORE2 config the window-mean-MLD gradient is clean to ~1 day and blows up by ~2 days).
   Forward-mode (TLM) does **not** escape this — it shares the adjoint's linearization (same singular
   values). To get a **climate-mean** sensitivity anyway, *don't* backprop a long window: **average
   many short frozen-ice bursts along a reference trajectory** (the ensemble-averaged adjoint — the
   chaotic part cancels, the slow signal survives). Long-window *training* still needs shadowing-type
   methods (the differentiable-chaos frontier), not just more memory.
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
9. **Sea-ice rheology adjoint is unstable → the "frozen-ice" adjoint for multi-day windows.** The
   (m)EVP rheology's iterative pseudo-time solve has a backward that amplifies far faster than the
   ocean's, so any gradient window beyond ~1 day runs the **frozen-ice adjoint** — the ice forward
   runs in full; only its *backward* is `stop_gradient`'d (`IceConfig(adjoint_mode="frozen")`). Ice
   *thermodynamic* sensitivities over hours→days are fine; the *rheology* gradient over long windows
   is not. (Every climate-sensitivity burst uses this.)
10. **Slow targets need EKI, not the short-window adjoint (the adjoint↔EKI boundary).** Sensitivities
    that develop over months — GM/Redi → interior T/S, the deployed-closure equilibrium — are *small
    and under-resolved* in a clean short adjoint window, so the short-window adjoint optimum can be the
    wrong target for a long deployment (the offline/online closure gap). Those use the forward-only
    **EKI** ensemble (immune to the chaos/memory ceiling and the ice-adjoint instability); adjoint and
    EKI agree on the shared scalar and are each used where correct.

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
