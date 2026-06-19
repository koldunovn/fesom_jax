# fesom-jax — User Guide

How to configure and run the JAX FESOM2 ocean model from a **single YAML file**: cold-start or
resume from a portable restart, run an arbitrary length, stream output, and chain multi-job campaigns
with SLURM. (For the API/internals see `README.md` and `docs/PORTING_LESSONS.md`.)

---

## 1. Install

```bash
mamba create -n fesom-jax python=3.12 -y && conda activate fesom-jax
pip install -e ".[dev]"          # CPU; add ".[cuda]" on a GPU node for CUDA-12 wheels
```

Everything is **float64** (`jax_enable_x64`, set at import). The model runs single-device (CPU/1 GPU)
or sharded across many GPUs / nodes via `jax.distributed` — the *same* code, selected by the partition.

---

## 2. One YAML → one run

A run is driven by one [`RunConfig`](../fesom_jax/run_config.py) YAML. The driver does exactly one
thing: **load restart (or cold IC) → run N steps (or a duration) → write a portable restart → exit.**
There is no in-model orchestration; multi-job campaigns are a SLURM `--dependency` chain (§6).

```bash
# one segment (cold start from PHC IC if restart_in is null):
python scripts/run_from_config.py configs/core2_full.yaml --steps 480 --restart-out runs/core2/seg0

# resume that restart onto ANY device count and run more:
python scripts/run_from_config.py configs/core2_full.yaml --restart-in runs/core2/seg0 \
       --steps 480 --restart-out runs/core2/seg1

# on a GPU node (or multi-node with JDIST=1):
sbatch scripts/run_from_config.sbatch configs/dars.yaml --steps 4800 --restart-out runs/dars/seg0
```

### The YAML schema

Every key is optional; an absent key takes the **bit-identical default** (today's `config.py`). The
schema is sparse: `null` = that physics OFF, `{}` = ON with defaults, a mapping = ON with overrides.

```yaml
# physics (selectable sub-configs)
ale:   {}            # {} = zstar ALE on;  null = linfs
gm:    null          # null = GM off;  {} or {k_gm_min: …} = GM on (+ overrides)
kpp:   {}            # vertical mixing: KPP …                  (kpp XOR tke — exactly one)
tke:   null          #                 … or classical-TKE
ice:   {}            # mEVP sea ice (needed for the NG5 LKF figures)

# promoted run-dependent scalars
visc:   {gamma1: 0.2}   # horizontal-viscosity γ's (only the NON-default ones; NG5 wants γ1=0.2)
tracer: {}              # MFCT/QR4C/FCT, num_ord (0,1) — the implemented path (a different
                        #   scheme RAISES: num_ord is hard-coded in the kernels, not a toggle)
dt: 180.0
dt_ramp: {after_step: 175200, dt: 240.0}   # cold 180 → prod 240 (a dt CHANGE across a restart)

# run orchestration (consumed by the driver, not the step kernel)
mesh:  /work/.../ng5
partition: dist_64       # "dist_<N>" (a FESOM partition) or "serial"
forcing: {kind: core2, start_year: 1958}
output_dir: /work/.../ng5_out
snapshot_every:   480    # steps; time-subsampled instantaneous snapshots (0 = off)
checkpoint_every: 2400   # steps; portable-restart cadence (0 = off)
restart_in:  null        # null ⇒ cold PHC IC; else a prior restart store (any device count)
restart_out: /work/.../ng5_out/restart
n_steps: 1440            # OR a duration string (below)
duration: 2yr            # "2yr" / "3mo" / "5d" / "12h" / "100step" / an int (steps)
```

**Unknown keys raise** (no silent typos). `dt_ramp` is a dt change at a step boundary; the driver
re-bootstraps the AB2 history there (a dt change invalidates the history formed at the old dt) and
rebuilds the dt-dependent SSH operator for the new dt.

---

## 3. Run lengths

`n_steps` (exact) or `duration` (a string the driver converts with the configured `dt`):
`step` (exact count), `s`/`h`/`d` (seconds/hours/days), `mo` (30-day), `yr` (365-day). A single
step (`n_steps: 1` / `"1step"`) is valid — useful for smoke tests.

---

## 4. Restart — portable across device counts

The restart writes the **FULL** prognostic State (every leaf, incl. the history/carry slots
`T_old`/`S_old`/`uv_rhsAB`/`sigma*`/`tke`) gid-keyed and gather-free, so it reloads onto **any** device
count: **save on 64 GPU → resume on 8** (or the reverse). A chained run is bit-identical to a continuous
one (the restart round-trip is faithful and AB2 continues across the seam). See
`fesom_jax/zarr_output.py` (`write_restart`/`read_restart`).

---

## 5. Output — streaming mean/variance + snapshots

Output is gather-free (each device writes its own shard). The streaming accumulator
(`fesom_jax.zarr_output.OnlineStats`, Welford) gives the **time mean AND variance** — hence the EKE map
(`eke_from_stats` = ½⟨u'²+v'²⟩) — without storing every step. Periodic instantaneous snapshots
(`snapshot_every`) capture the visual fields / KE spectra / LKF deformation. Read a field back to a
dense global array for analysis with `zarr_output.reconstruct_global`.

---

## 6. Multi-job campaigns — SLURM chains (no in-model orchestration)

A multi-year run is a **series of dependent SLURM jobs**, each one config-driven segment that resumes
the previous restart. The model never loops jobs internally.

```bash
# 6 segments of 4800 steps each, chained with --dependency=afterok; a failing rung stops the chain:
scripts/chain_submit.sh configs/dars.yaml 6 4800 /work/.../runs/dars
```

Segment 0 cold-starts; segment *k* resumes `seg<k-1>` and writes `seg<k>`. This is the ladder-gated
pattern for the long NG5 spin-up — **de-risk on the smaller meshes (farc/dars) first**, then climb NG5.

---

## 7. Mesh / dt ladder

| mesh  | nodes  | levels | dt (s)        | nodes (GPU) | role |
|-------|--------|--------|---------------|-------------|------|
| CORE2 | 127 k  | 48     | 1800          | 1           | mean-state fidelity (reuse the 5+10 yr) |
| farc  | 638 k  | 48     | 900           | 1           | machinery de-risk + ladder |
| dars  | 3.16 M | 57     | 180           | ≥2          | machinery de-risk + scaling |
| NG5   | 7.4 M  | 70     | 180→240 (ramp)| ≥8          | eddy-resolving flagship + LKFs |

**Preparing a mesh** (one-time, offline, no C build): `load_mesh` reads a directory of `.npy` arrays
(the derived geometry — gradients, areas, edges, …). Produce it from the raw FESOM ASCII with
`python scripts/prepare_mesh.py RAW_MESH_DIR OUT_DIR` (a pure-numpy port of FESOM's mesh setup, verified
byte-faithful vs the original C export; `docs/MESH_EXPORT_LAYOUT.md`). Partitions (`dist_<N>`) ship with
the mesh and are read directly.

Forcing (JRA55 + Large-Yeager bulk + SSS-restoring/runoff/chl) is interpolated **at runtime** (like
FESOM — the bilinear weights build once at setup, then each step reads + interpolates); nothing is
pre-staged. Per-step forcing is fed to the sharded scan in fine (≈ few-day) time-chunks (`--chunk-steps`)
so a multi-year run never pre-stacks its forcing.

---

## 8. Testing

```bash
# fast, data-free gates (what CI runs anywhere):
pytest fesom_jax/tests/test_release.py fesom_jax/tests/test_run_config.py \
       fesom_jax/tests/test_stream_output.py fesom_jax/tests/test_run_entry.py
# full regression on Levante (needs the data symlink; ~1:45):
sbatch scripts/run_suite.sbatch
```

The standing invariant: `RunConfig.defaults()` / config-off / `params=None` ⇒ **bit-identical** to the
bare model — the regression guard, asserted in `test_run_config.py`.
