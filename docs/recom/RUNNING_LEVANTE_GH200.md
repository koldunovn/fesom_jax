# fesom-jax REcoM on the Levante GH200s (dolpung)

The CORE2 gradient does not fit one Albedo A100 (plan `docs/plans/20261105-recom-parameter-estimation.md`, 2026-11-12:
a 1-day window needs 89-187 GiB), so the CORE2 adjoint work moves to the dolpung partition (aarch64 nodes with four
GH200 each, account `mh1571`, decision D3). Everything below is for the Levante user `a270044`; the paths are set in
`scripts/levante/gh200.conf` and can be overridden through the environment.

| What | Where |
|---|---|
| data pool (44 GiB, from Albedo) | `/work/bk1415/a270044/fesom_jax_pool` |
| checkout | `/work/bk1415/a270044/fesom_jax_recom` |
| venv (aarch64, python 3.12, JAX 0.10.1) | `$POOL/envs/fesom-jax-gh200` |
| run output, slurm logs | `$POOL/runs`, `$POOL/runs/slurm` |

## Once

1. **Data**, from Albedo: `python scripts/albedo/levante_pool.py rsync --user a270044 --run` (prints the commands without
   `--run`; rsync over ssh with `--partial`, so a broken copy resumes). The Levante configs in `configs/levante/` are
   written by `python scripts/albedo/levante_pool.py configs --user a270044` from the Albedo ones; edit those and rerun.
2. **Code**: push `recom-spectral` to the fork from Albedo, then on a Levante login node
   `git clone -b recom-spectral <fork> /work/bk1415/a270044/fesom_jax_recom`.
3. **Wheels**, on a Levante login node (x86, internet): `bash scripts/levante/fetch_wheels.sh`. It downloads the aarch64
   / CPython 3.12 wheels of `scripts/levante/requirements_gh200.txt` (the Albedo env's versions) and creates the slurm
   log directory. numcodecs 0.15.2 (zarr 2 needs < 0.16, which has no aarch64 wheel) and asciitree come as source.
4. **Venv**, on a GH200 node: `sbatch scripts/levante/make_env_gh200.sbatch` (from the checkout). It builds the venv
   from the ARM spack tree's python 3.12, installs offline (numcodecs is compiled there), installs the checkout
   editable and checks that JAX sees four GPUs.

## Jobs

- **Gradient horizon** (`scripts/recom/bgc_gradient_horizon.py`), CORE2 from the `w28sl1` restart of 2000-07-01:
  `sbatch scripts/levante/run_horizon_gh200.sbatch --days 1 --cost chl --repeats 2 --remat-blocks --out $POOL/runs/horizon/core2_smoke`.
  `CONFIG`, `RESTART`, `YEAR` select another config or start state. The driver uses one of the four GPUs.
- **Forward runs** (`scripts/run_from_config.py`): `sbatch scripts/levante/run_config_gh200.sbatch <config> [flags]`.

The jobs strip the x86 spack tree that srun exports from the login environment (as `scripts/perf/env_dolpung.sh`),
switch XLA command buffers off and use `/scratch/a/a270044/fesom_jax_tmp` for temporary files. Repeat the gates there
before trusting gradients: the noise of the forward differs by machine (D3).
