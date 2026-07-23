# Running FESOM2-JAX strong-scaling on JUPITER (GH200)

Clone-and-run instructions to reproduce the Levante strong-scaling campaign on the JSC
exascale system **JUPITER** (GH200 Grace–Hopper nodes). Levante tops out at 16 nodes /
64 GPUs; JUPITER is where NG5/dars scale past that.

> **Status.** The *code* and the *launch pattern* below are the exact ones the Levante
> campaign used (verified). Everything JUPITER-specific — module/container names,
> partitions, account, filesystem paths, NCCL fabric settings — is marked **⚠️VERIFY**
> and must be checked on the machine; they were not tested here (no JUPITER access from
> the porting host). The Kokkos twin's JUPITER run is the reference for the machine
> facts: `port_kokkos/docs/plans/20260722-m7-JUPITER-scaling-PLAN.md` and
> `…-FINDINGS.md` (GH200 booster).

Machine (from the Kokkos campaign): **quad-GH200 nodes — 4 GH200 superchips/node, each a
72-core Grace ARM CPU + H100-class GPU (sm_90), NVLink-C2C 900 GB/s host↔device, HBM3;
NDR InfiniBand between nodes.** So: **aarch64**, 4 GPUs/node, `MaxNodes=UNLIMITED`, fast
queue (128–256-node jobs started within minutes; 28 concurrent jobs, no AssocMaxJobs
friction).

---

## 0. The one thing that is different from Kokkos

Kokkos on JUPITER just needs a C++/CUDA compiler. **FESOM2-JAX needs a working
`jaxlib` with CUDA support on aarch64/Hopper** — that is the whole risk of this port on
JUPITER, and §2 is about getting it. Two more consequences:

- **JAX collectives use NCCL, not MPI.** The Kokkos "prove the MPI is CUDA-aware or it
  segfaults on device pointers" lesson (their L77) does **not** transfer literally, but
  its analog does: **prove NCCL uses the InfiniBand fabric and that an inter-node
  all-reduce works before believing any multi-node number** (§3, gate J2).
- **`jax.distributed.initialize()` auto-detects SLURM** (coordinator, process count,
  process id from `SLURM_*`). No launcher glue needed — one process per node, each
  owning its 4 local GPUs (`local_device_ids=range(GPUS_PER_NODE)`), exactly as on
  Levante.

---

## 1. Get the code and the data onto JUPITER

### Code
```bash
git clone https://github.com/koldunovn/fesom_jax.git
cd fesom_jax
# these JUPITER files live under docs/ and scripts/bench/jupiter/
```
`data/mesh_core2` and `data/ic_core2` ship **in the repo** — so CORE2 (the bootstrap
mesh, §4a) needs no mesh transfer, only forcing.

### Data to transfer from Levante (rsync over JUDAC ⚠️VERIFY `judac.fz-juelich.de`)
JAX reads **two** kinds of mesh input — keep them straight:

| what | Levante path | size | needed for | env / flag |
|---|---|---|---|---|
| **JAX mesh export** (`.npy` bundle, ~35 files, device-independent) | `/work/ab0995/a270088/fesom_jax_meshes/mesh_<name>` | ng5 **24 GB**, dars 8.4 GB, farc/forca20 less | `--mesh-dir` (per mesh) | — |
| **FESOM partition files** (`dist_<N>/`) | `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/<name>/dist_<N>` (dars/forca20: `/work/ab0995/a270088/meshes/FORCA20`, `…/dist_<N>`) | small each | `--dist-dir` | — |
| **JRA55-do 1958 forcing** | `/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/` (uas,vas,huss,rsds,rlds,tas,prra,prsn `.1958.nc`) | ~12 GB | all runs (real forcing) | `FESOM_JRA_DIR` |
| SSS / runoff / chl / PHC IC | see `fesom_jax/paths.py` `LEVANTE_*` defaults | small | all runs | `FESOM_SSS_PATH`, `FESOM_RUNOFF_PATH`, `FESOM_CHL_PATH`, `FESOM_PHC_PATH` |

**Shortcut:** the `dist_<N>` partition files are the *same* FESOM format the Kokkos twin
already staged on JUPITER (`/e/scratch/.../meshes/<name>/dist_<N>` per their findings) —
point `--dist-dir` there and skip re-transferring them. Only the **`.npy` JAX mesh
exports** and the **forcing** are genuinely JAX-specific transfers. Partitions exist for
every planned scale (ng5 to 8192, dars to 4096) — no partitioner runs on JUPITER.

Set the data env once (put in `env_jupiter.sh`, §2):
```bash
export FESOM_JRA_DIR=/path/on/jupiter/JRA55-do-v1.4.0
export FESOM_SSS_PATH=... FESOM_RUNOFF_PATH=... FESOM_CHL_PATH=... FESOM_PHC_PATH=...
```

---

## 2. The environment (aarch64 + CUDA jaxlib) — do this first, it is the crux

Pick the first of these that gives you working GPU JAX. **Gate every path on J0 below.**

**(a) NGC JAX container via Apptainer (most reliable on GH200, recommended).**
JSC supports Apptainer. The NVIDIA JAX image bundles a Hopper/aarch64 `jaxlib` + CUDA +
NCCL tuned for GH200:
```bash
module load Apptainer            # ⚠️VERIFY module name
apptainer pull jax.sif docker://nvcr.io/nvidia/jax:25.04-py3   # ⚠️VERIFY latest tag
# run the repo inside it, binding the data + repo:
apptainer exec --nv \
  -B $PWD:$PWD -B /path/to/data:/path/to/data \
  jax.sif python -c "import jax; print(jax.devices())"
```
Then everywhere below, `PY="apptainer exec --nv -B ... jax.sif python"`.

**(b) pip aarch64 CUDA wheels (simplest if the wheels exist).**
Recent JAX ships `manylinux aarch64` CUDA wheels:
```bash
module load Python CUDA           # ⚠️VERIFY a CUDA 12.x module
python -m venv ~/jaxenv && source ~/jaxenv/bin/activate
pip install -U pip
pip install "jax[cuda12]"         # ⚠️VERIFY aarch64+cuda12 wheels resolve
pip install -e .                  # the fesom_jax package + deps (numpy, scipy, zarr, pyyaml, netCDF4)
```

**(c) mamba env** (fallback): a fresh aarch64 env with `jaxlib=*=cuda*` from conda-forge
if it carries aarch64+CUDA (⚠️often it does not — prefer a/b).

### J0 — the environment gate (run on a single GH200, interactively or 1-GPU sbatch)
```bash
python -c "import jax; print(jax.devices()); \
           import jax.numpy as jnp; x=jnp.ones((1000,1000)); \
           print(float((x@x).sum()))"
```
Must print **4 `CudaDevice`s** (or 1 in a 1-GPU alloc) and a finite number. If it prints
CPU, the CUDA plugin is not loaded — fix §2 before anything else. Then run the repo's
own smoke: `XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/bench/bench_forward_scaling.py \
  --name core2 --mesh-dir data/mesh_core2 --dist-dir <core2-dist> --npes 1 --steps 25 --full 1`
— expect a `[bench-finite] … nonfinite_T=0 nonfinite_uv=0` line and a `[bench] … per_step=` line.

---

## 3. Sanity gates before scaling (cheap, catch the machine-specific failures)

- **J1 — single-node 4-GPU** (`--npes 4`, `JDIST=1 GPUS_PER_NODE=4`, one node): confirms
  intra-node NVLink collectives + the padded/coloured transports run. bench-finite CLEAN.
- **J2 — TWO-node all-reduce over IB** (the load-bearing one — see §0): `--npes 8` across
  2 nodes. If it hangs or the numbers are absurd, NCCL is not on the IB fabric. Set (JSC
  documents the exact names ⚠️VERIFY):
  ```bash
  export NCCL_SOCKET_IFNAME=ib0        # ⚠️VERIFY the IB iface
  export NCCL_IB_HCA=mlx5              # ⚠️VERIFY the HCA prefix
  # first debug run only: export NCCL_DEBUG=INFO   (shows the transport it picked)
  ```
- **J3 — bench-finite is the gate on every row, always.** A blown-up ocean times a fake
  step (the hard-won Levante rule). Read the `[bench-finite]` line before trusting any
  `per_step`. dars cold-starts at **dt=120** for stability (per-step cost is
  dt-independent; SYPD uses the production dt — dars 240, farc 1200, ng5/forca20 240,
  core2 1800).

---

## 4. The scaling runs

Same harness as Levante: `scripts/bench/bench_forward_scaling.py`, best transport per
point (**padded ≤16 GPUs, coloured beyond** — coloured's cost is set by the mesh colouring,
not the device count, and is what lets dars/NG5 keep scaling past 64 GPUs), 150 steps,
≥2 reps, `.out` logs. The launch is portable; only SLURM headers change. Template:
`scripts/bench/jupiter/bench_scaling_jupiter.sbatch`.

**(a) BOOTSTRAP — CORE2, in-repo data, validates the whole pipeline (1→8 GPUs).** Small,
no mesh transfer. Confirms env + launch + reduce end-to-end before spending big-mesh
allocations.

**(b) THE TARGET — dars and NG5 to 128+ GPUs (32+ nodes).** This is why JUPITER: on GH200
(96 GB HBM3, NVLink-C2C) more fits per GPU than A100-80, and the coloured transport should
carry the multi-million-node curves further than Levante's 128-GPU top. Suggested rungs
(dists exist well beyond): **NG5 32/64/128/256, dars 16/32/64/128.** Per the Kokkos twin,
NG5 peaked at 128 and dars had its knee at 64→128 on *their* stack — the JAX numbers are
the open question.

Each point:
```bash
export JDIST=1 GPUS_PER_NODE=4          # multi-node; unset for a single node
srun -N<nodes> -n<nodes> --ntasks-per-node=1 --cpu-bind=none \
  $PY scripts/bench/bench_forward_scaling.py \
  --name ng5 --mesh-dir $MESH/mesh_ng5 --dist-dir $DIST/ng5 --ic-dir $MESH/mesh_ng5 \
  --npes <total_gpus> --steps 150 --dt 180 --full 1 --halo coloured --year 1958
```
(`--npes` = total GPU count = nodes×4. `--full 1` = the production physics
`ice+mevp+tke+zstar`; for the paper's per-mesh production model add `--mevp 1 --tke 1
--kpp 0 --gm 0 --zstar 1`, GM on for core2 only.)

---

## 5. Reduce + plot

`scripts/logs/bench_*.out` feeds the paper reducer unchanged:
`paper_jax/scripts/reduce/reduce_scaling.py` (best-per-ngpu, exact model-string select) →
`fig_scaling.py`. Or just grep the `[bench]` lines and compute
`SYPD = dt_prod / (sstep · 365.25)`. Keep JUPITER rows in their own log dir so a Levante
`make data` never mixes machines.

---

## 6. Gotchas carried from the Kokkos JUPITER run (adapted for JAX)

- **Memory-borderline legs at the fattest per-GPU loading** — their ng5_g2/farc_g2 got
  intermittent "task Killed". For JAX the analog is a single GPU that can't hold a
  huge mesh's per-device shard; start NG5 at enough GPUs (≥16), and set
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **min-of-2 reps** — some corners are noisy; ≥2 reps per point (the sbatch loops REP).
- **No partitioner needed** — dists pre-exist for every scale; copy, never regenerate.
- **Compile is excluded from timings** (2nd warm call is timed) but is NOT free wall-clock
  on a big mesh (~1–3 min at NG5). Do NOT enable a persistent XLA cache (project rule:
  staleness risk > the wall-clock win; timings already exclude compile).
- **dars at very large per-device partitions has an XLA compile pathology** (the zstar
  per-CG-iteration stiffness assembly blows up compile >3 h at ~435k lanes/GPU on A100;
  the dars-8 production point was excluded on Levante). On GH200 this may differ — if a
  dars point never leaves "compile", that's this pathology, not a hang; drop that point
  or raise the node count so the shard shrinks.
- **Cross-arch drift is expected** — GH200 vs A100 will differ at roundoff and grow over
  steps (the Kokkos twin documented ~1e-18 at step 0, ≤1.5e-4 T over 10 steps). Scaling
  numbers don't need bit-identity; if you run a fidelity leg, gate against a rerun-noise
  floor, not against zero.

---

## 7. What "done" looks like

Strong-scaling curves (s/step + SYPD@production-dt + parallel efficiency) for CORE2 (1→8),
farc, dars, NG5 out to the point each turns over on JUPITER's fabric, best transport per
point, every row bench-finite, ≥2 reps — the GH200 counterpart of `paper_jax` fig10, and
the node-for-node GH200-vs-A100 comparison at matched rank counts.
