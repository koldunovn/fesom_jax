# env_dolpung.sh — run environment of the JAX port on the DKRZ dolpung GH200 partition (aarch64 nodes, 4x GH200
# per node, 72-core Grace + H100-class 120 GB each; account mh1571). Source it inside a dolpung job; logins are x86.
# Venv: python 3.12.13 (ARM spack tree) + the constraints.txt wheels, built in the mitgcm-jax project (its
# scripts/dolpung/{fetch_wheels,make_env}.sbatch); this file is a copy of its scripts/dolpung/env_dolpung.sh.
# No module loads: srun exports the x86 login env, so strip the x86 spack tree (port_kokkos env_dolpung.sh lesson).
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/sw/spack-levante/' | grep -v mambaforge | paste -sd:)
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/sw/spack-levante/' | grep -v mambaforge | paste -sd:)
export PATH LD_LIBRARY_PATH
export MITJAX_ARM_VENV=${MITJAX_ARM_VENV:-${MITJAX_WORK:-/work/ab0995/a270088/MIT}/envs/mitgcm-jax-arm}
export PY=$MITJAX_ARM_VENV/bin/python
unset PYTHONPATH PYTHONHOME CONDA_PREFIX
