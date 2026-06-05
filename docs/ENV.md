# Environment (Levante)

Reproducible JAX environment for `fesom-jax`. **Exact resolved versions are
recorded at the bottom after install** (Task 0.1 deliverable).

## Machine

- Host: `levante.dkrz.de` (login nodes `levanteN` are **CPU-only**).
- GPUs: SLURM `gpu` partition — 59× **NVIDIA A100 80GB** (`gpu:a100_80:4`, 12 h
  limit), plus A100-40 nodes; `gpu-devel` (A100-40, 30 min) for quick checks.
- GPU account: `ab0995_gpu`.
- Python base: mambaforge at `/work/ab0995/a270088/mambaforge`.
- No standalone `cuda`/`cudnn` spack modules → CUDA comes from JAX's pip wheels.

## Create the environment

CUDA 12 + cuDNN ship as pip wheels (`jax[cuda12]`), so **no system CUDA module is
needed**. The login node is CPU-only; the CUDA build still imports there and runs
on CPU (with a "no GPU" warning) — GPU is verified on a compute node.

```bash
# 1. dedicated env (Python 3.12)
mamba create -y -n fesom-jax python=3.12 pip
mamba activate fesom-jax

# 2. install the package + GPU JAX + dev tools
cd /home/a/a270088/port_jax
pip install -e ".[cuda,dev]"
```

CPU-only variant (no GPU wheels): `pip install -e ".[dev]"`.

## Verify (CPU, on login node)

```bash
python -c "import jax, jax.numpy as jnp; \
  print('jax', jax.__version__); \
  print('x64', jnp.ones(1).dtype); \
  print('devices', jax.devices())"
pytest fesom_jax/tests/test_config.py -q
```

## Verify GPU (compute node)

```bash
srun -A ab0995_gpu -p gpu-devel --gres=gpu:1 -t 5 \
  python -c "import jax, jax.numpy as jnp; \
  print(jax.devices()); \
  x=jnp.ones((1024,1024)); print((x@x).sum(), (x@x).dtype)"
```
Expect a `CudaDevice` in the device list and `float64` output.

## Recorded versions

Installed 2026-06-05 into mamba env `fesom-jax` (`pip install -e ".[cuda,dev]"`).

- Python: **3.12.13**
- JAX / jaxlib: **0.10.1**
- jax-cuda12-plugin / jax-cuda12-pjrt: **0.10.1**
- CUDA (pip wheels): cuda-runtime **12.9.79**, cuBLAS 12.9.2.10, cuDNN **9.23.0.39**, nvcc 12.9.86
- numpy **2.4.6**, scipy 1.17.1, pytest 9.0.3
- **CPU verified** (login node): `backend=cpu`, x64 `float64`, jitted float64 matmul OK;
  `pytest fesom_jax/tests/test_config.py` → 4 passed.
- **GPU verified** (job 25374974, gpu-devel → `vader1`): `backend=gpu`,
  `[CudaDevice(id=0)]`, x64 float64 jitted matmul OK on **A100-PCIE-40GB**,
  ~31.8 GB usable (JAX default mem fraction of 40 GB). Production target is the
  `gpu` partition's **A100-80GB**.

> On a CPU-only login node, importing JAX prints a benign
> `Jax plugin configuration error ... cuInit(0) failed: ... CUDA error 303`
> and falls back to CPU — expected (no GPU/driver on login nodes). On a GPU
> compute node the CUDA plugin initializes normally.
