# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`fesom-jax`: a port of the FESOM2 unstructured-grid ocean model to JAX, fully differentiable and
multi-GPU/multi-node via `shard_map`. This checkout (`fesom_jax_recom`, branch `main`, upstream
`koldunovn/fesom_jax`) carries the **REcoM biogeochemistry** port (`fesom_jax/recom/`, 31 tracers,
`bin_2p3z2d` configuration). The README is long and authoritative; `docs/` holds the per-topic
detail. Read `docs/PORTING_LESSONS.md` "Cross-cutting" before touching numerics.

## Environment

- **This machine is AWI's Albedo, not DKRZ's Levante.** Every absolute path in the repo
  (`/work/ab0995/...`, `/pool/data/AWICM/...`, the `PY=` interpreter in sbatch scripts, `-A ab0995_gpu`)
  is a Levante default and does not exist here. The REcoM CORE2 inputs on Albedo live under
  `/albedo/work/projects/p_pool_recom/`. Albedo's `gpu` partition has PCIe A100s: the README notes
  CORE2 runs 7.6x slower there than on NVLink and anti-scales past 1 GPU, so prefer `partition: null`.
- **The JAX env is `/albedo/work/user/sloza/fesom-jax-env/fesom-jax/bin/python`** (python 3.12, jax 0.10.1
  with CUDA 12 wheels, editable install of this checkout; built by `create_env.sh` next to it). Home is over
  quota, so envs, data and runs live on `/albedo/work`. Albedo-specific scripts/configs/docs:
  `scripts/albedo/`, `configs/albedo/`, `docs/recom/RUNNING_ALBEDO.md`; the CORE2 data package is at
  `/albedo/work/projects/p_specbic/sloza/fesom_jax_data`, runs under `.../fesom_jax_runs`.
- The login node (256 cores, `ulimit -u 2048`) kills XLA CPU compiles unless you run under `taskset -c 0-15`.
  SLURM: `-A clidyn.clidyn`; CPU `-p mpp`; GPU `-p gpu --gpus=a100:1` (PCIe A100s, use one GPU).
- Input data resolve as `explicit arg / YAML forcing: key > $FESOM_* env var > Levante default`
  (`fesom_jax/paths.py`). Off Levante, export `FESOM_JRA_DIR`, `FESOM_PHC_PATH`, `FESOM_SSS_PATH`,
  `FESOM_RUNOFF_PATH`, `FESOM_CHL_PATH`, `FESOM_MESH_ROOT`, or set the keys in the run YAML.
  `scripts/fetch_data.py --dest DIR --print-env` fetches the CORE2 one-year package and prints them.
- The packaged `pi` mesh (3140 nodes) and its PHC IC ship in `fesom_jax/data/`, so pi runs and
  pi tests need no data. The REcoM Fortran twin (`$FESOM_RECOM_TWIN`) and the REcoM IC/dust
  inputs are not in the repo; tests needing them skip.
- `import fesom_jax` flips `jax_enable_x64` as a side effect. Import it before creating any JAX
  array. float64 is mandatory; `precision: single` is an opt-in deviation handled by `precision.py`.

## Commands

```bash
# install (editable)
pip install -e ".[dev]"                  # add cuda for GPU wheels, viz for notebooks

# data-free tests that run anywhere (what CI runs): the `ci` group from the manifest
FILES=$(awk 'NF && $1 !~ /^#/ && $1=="ci" {print $2}' scripts/runs/suite_groups.txt)
python -m pytest -q -p no:cacheprovider $FILES

# one file / one test
python -m pytest fesom_jax/tests/test_recom_sms.py -q
python -m pytest fesom_jax/tests/test_recom_step.py::test_g4b_one_step_vs_twin -q

# sharding tests need CPU fake devices (and a compute node, not a login node)
XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest fesom_jax/tests/test_recom_transport.py -q

# tests under scripts/tests/ need the repo root on sys.path (their conftest does it); run from repo root
python -m pytest scripts/tests/test_write_namelist.py -q

# full suite = a SLURM job on a compute node (Levante scripts; adapt partition/account/PY for Albedo)
sbatch scripts/runs/run_suite.sbatch
sbatch scripts/runs/run_suite_parallel.sbatch     # one array task per cost group

# config-driven run: one YAML -> load restart or cold IC -> N steps -> portable restart
python scripts/run_from_config.py configs/pi_recom_bgc.yaml --restart-out runs/pi_recom/seg0
python scripts/run_from_config.py configs/core2_gpu_recom.yaml --restart-in DIR --steps 11680 \
    --restart-out OUT/restart --monthly-out OUT/monthly
```

There is no linter or formatter configured. CI (`.github/workflows/ci.yml`) runs only the pi-mesh
gates plus the `ci` cost group on a 2-core runner.

### Test-suite rules

- `scripts/runs/suite_groups.txt` is the manifest: **every** `test_*.py` under `fesom_jax/tests/`
  and `scripts/tests/` must appear exactly once, grouped by cost (`ci`, `unit`, `core2`, `ice`,
  `sharding`, `scripts`). `test_suite_groups.py` fails otherwise. New file: `core2` if it loads
  the CORE2 mesh/IC/forcing, `sharding` if it needs fake devices, `scripts` if under
  `scripts/tests/`, else `unit` (or `ci` if it runs in under 30 s on 2 cores).
- Most verification gates compare against Fortran/C reference dumps in `fesom_jax/tests/fixtures/`
  or on `/work`; without them they **skip**, and the terminal summary says how many. Green off
  Levante does not mean the numerics were verified.
- The suite is bounded by process memory *mappings*, not RAM: `conftest.py` clears JAX caches per
  module. "LLVM compilation error: Cannot allocate memory" is that ceiling, not a numerics bug.
- `@pytest.mark.gpu` tests skip unless the default backend is GPU.

## Architecture

**One step, many configs.** `fesom_jax/step.py::step(state, mesh, op, stress, params, ...)` is the
single ocean timestep, mirroring the FESOM2/C substep order (EOS, PGF, mixing, momentum, SSH CG
solve, ALE thickness, FCT tracers). Physics options are static, hashable config dataclasses passed
as keyword args (`ale_cfg`, `gm_cfg`, `kpp_cfg`, `tke_cfg`, `ice_cfg`, `recom_cfg`, ...). `None`
means off and must leave the traced program **byte-identical**: an option is a dead branch, never
a runtime switch. This is the regression guard everything relies on (`test_dp_lock.py`,
`test_run_config.py`).

**State and params are pytrees.** `state.py::State` is a flat frozen dataclass, every field dense
`[entity, nl, ...]` with axis 0 the mesh entity (sharding folds the device axis into axis 0).
Optional field groups (ice, SE, `BGC_FIELDS`) are `None` unless enabled so old restarts load.
`params.py::Params` holds the differentiable tunables (the ML-hook seam: PP backgrounds, GM/Redi
ceilings, TKE constants, NN weights). Ragged vertical extent is handled by masks from `mesh.py`,
never ragged storage; layer vs interface validity differs by one level.

**Time loops.** `integrate.py::integrate` is the single-device `lax.scan` with per-step
`jax.checkpoint` (`remat_segments=-1` for O(sqrt N) memory in gradients). `integrate_sharded.py`
wraps the same `step` in `shard_map` over a 1-D device mesh: `shard_mesh.py` builds per-device
padded local meshes from a FESOM `dist_<N>` partition (`partit.py`), lane order
`[owned | halo | pad]`, and `halo.py`/`halo_points.py`/`halo_grouped.py` provide the four halo
transports (all_gather default, ragged = GPU-only and forward-only because its AD transpose is
broken, padded, coloured, grouped). `reductions.py` routes global sums so halo copies are not
double-counted. `ops.py` (gather, masked scatter-add, TDMA) is the vocabulary every kernel is
written in.

**Run driver.** `run_config.py::RunConfig` composes the sub-configs plus orchestration keys from one
YAML (`configs/*.yaml`; `RunConfig.defaults()` is bit-identical to a bare `step`). `run.py::
run_from_config` plans time chunks (forcing is interpolated at runtime, never pre-stacked for a
year), runs the sharded forced scan, writes a portable restart (`zarr_output.py`, gather-free,
device-count independent). `scripts/run_from_config.py` wires mesh/partition/IC/forcing to it;
campaigns are SLURM dependency chains (`scripts/chain_submit.sh`), no in-process orchestration.
`surface_forcing.py` + `jra55.py` + `phc_ic.py` + `sss_runoff.py` build the forced path;
`gpu.*` YAML keys (`configs/core2_gpu*.yaml`) are the speed options, all off by default.

**REcoM (`fesom_jax/recom/`).** Off unless `RunConfig.recom` / `recom_cfg` is set. `wiring.py` is
the entry: `full_bgc_step` = stage 1 (`stage1.py`, FESOM2's `recom()` sources and sinks, column
local except the `relax_alk` global mean) then stage 3 (`transport.py`, the BGC tracers through the
same advection/Redi/vertical-diffusion path as T and S plus REcoM's benthic return, sinking and
benthos terms). `sms.py` holds the ecosystem terms and the cached carbonate state, `carbonate.py`
the mocsy chemistry, `sinking.py` ballasting, `tracers.py` the tracer-ID slot table, `params.py`
reads `namelist.recom` as FESOM2 does and raises `NotImplementedError` for unported switches
(`SUPPORTED`), `ic.py` the cold-start IC sets, `forcing.py` dust/CO2, `reference.py` the Fortran
twin loader. Nothing in the physics reads a BGC tracer. REcoM requires zstar, the forced path with
`forcing.mslp: true` (sea-level pressure feeds gas solubility), and `with_recom_static` constants.
`bgc_every > 1` and single precision are stated deviations from FESOM2, not faithful modes.

**The spectral fork (branch `recom-spectral`).** The port of `sveta-loza/fesom2`
`fesom2.6_recom_tra_diags_spectral` (4 phytoplankton groups, CDOM, 13-band OASIM light) lives next to the 692f804
code and is selected by the namelist: a `namelist.recom` with an `&spectral` group is the fork flavour
(`params.RecomSwitches.spectral_fork`, `layout.layout_of` gives its 22-37-tracer layouts), `wiring.full_bgc_step`
then runs `sms_fork.py`/`stage1.recom_sources_sinks_fork`, the layout-aware `sinking.py`, and with
`RecomConfig.spectral` the light of `spectral.py` from the OASIM fields `oasim.py` reads (`forcing.oasim_dir`).
Configs `configs/albedo/pi_recom_spectral.yaml`, `core2_recom_spectral.yaml`; gates `test_recom_*_fork.py`,
`test_recom_spectral_*.py` against the fork's own twins (`docs/recom/VALIDATION.md`, the plan
`docs/plans/20261028-recom-spectral-port.md`). The fork's FESOM2 base needs `legacy_k33_weights`,
`legacy_fct_bounds`, `legacy_sw_depth` and its own level files in a twin gate; its GM/Redi is not matchable (the
physics-matched one-day twins run GM/Redi off). The same REcoM on current FESOM2 (`sveta-loza/fesom2`
`fesom2.8_recom_spectral`, in-tree `src/int_recom/`, no REcoM submodule) needs none of that: GM/Redi on, no
`legacy_*`, T floor 4.1e-8 (`FORTRAN_JAX_DIFFERENCES.md` "The fork on FESOM 2.8"). Stated deviation: the fork's whole-array ballast reset reduces over
the whole mesh, not the rank. The 692f804 programs stay byte-identical.

**Adjoint toolkit.** `adjoint.py`, `adjoint_sharded.py`, `adjoint_check.py`, `adjoint_costs.py`,
`longwindow.py`, `calibrate.py`, `eki.py`, `tke_nn.py`, `gm_nn.py`; drivers under
`scripts/adjoint/` and `scripts/paper/`. Gradients are trustworthy only over hours to days (chaos);
multi-day windows use the frozen-ice adjoint (`IceConfig(adjoint_mode="frozen")`). Sharded
gradients need `use_padded` or `use_coloured`, never ragged.

## Working conventions

- **Golden rule:** preserve the exact FESOM2 math and load-bearing association order, expressed as
  vectorized `ops.py` array ops. No loop-by-loop translation, no physics simplification. Cite the
  Fortran/C file:line the way existing code does.
- **Fidelity classes, not bit-identity:** ~1e-15 for map/gather kernels, ~1e-12 for
  scatter/reduction kernels. Do not chase a scatter discrepancy below 1e-12.
- **AD-safe by construction:** pure functional JAX, `lax.scan/cond/while` instead of Python
  control flow on traced values, fixed-iteration solvers, no hidden host state.
- Use the truncated `PI = 3.14159265358979` from `config.py`, never `jnp.pi`.
- Every new switch defaults off and must leave the default jaxpr unchanged; add the test.
- Docs are living records: `docs/plans/` (dated plans, `completed/` when done), `docs/PORTING_LESSONS.md`
  (append a numbered lesson for a non-obvious finding), `docs/recom/VALIDATION.md` (REcoM gate numbers).
  Plans cite SLURM job ids as evidence; keep doing that.
- Large artifacts (`*.npy`, `*.npz`, `*.nc`, `data/`, `*.out`, `*.jsonl`) are gitignored except
  `fesom_jax/data/**` and `fesom_jax/tests/fixtures/**`.
