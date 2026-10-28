#!/bin/bash
# Fast-adjoint plan, the GPU-options step: an A/B of FESOM_* GPU options on ONE adjoint chunk (adj_profile.py time),
# one whole GH200 node, one process per variant (the options are read at trace time). Each variant's forward and
# reverse ms/step and its chunk control gradient against the first variant's (the options change round-off only).
#   sbatch --export=ALL,WT=<checkout>,VARIANTS="base: seu:FESOM_SE_UNROLL=10+FESOM_MEVP_UNROLL=12 ..." scripts/perf/adj_ab_gh200.sbatch [adj_profile flags]
# A variant is name:ENV=V+ENV2=V2 (empty after the colon = no option; NOT commas: sbatch --export splits on them and
# exports the pieces job-wide — job 28351354 ran every variant with four options set). dolpung footprint: one node.
#SBATCH --job-name=adj_ab
#SBATCH --partition=dolpung
#SBATCH --account=mh1571
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /work/ab0995/a270088/fesom_jax_dev/runs/perf/adj_ab.%j.out
set -u
WT=${WT:?WT (the checkout) must be set}
VARIANTS=${VARIANTS:?VARIANTS="name:ENV=V+... ..."}
if env | grep -q "^FESOM_"; then echo "FATAL: FESOM_* set job-wide (a comma in VARIANTS?):"; env | grep "^FESOM_"; exit 2; fi
source /home/a/a270088/MIT/scripts/dolpung/env_dolpung.sh          # sets PY; unsets PYTHONPATH
export PYTHONPATH=$WT:/work/ab0995/a270088/fesom_jax_dev/envs/arm_extra/site
cd "$WT" || exit 2
export TMPDIR=/scratch/a/a270088/fesom_jax_dev_tmp CUDA_CACHE_PATH=/scratch/a/a270088/fesom_jax_dev_tmp/nv_gh200
if [ -n "${JAX_COMPILATION_CACHE_DIR+x}" ]; then echo "FATAL: no persistent XLA cache"; exit 2; fi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
"$PY" -c "import sys, fesom_jax; sys.exit(0 if fesom_jax.__file__.startswith('$WT/') else 4)" || { echo "FATAL: import"; exit 4; }
OUT=/work/ab0995/a270088/fesom_jax_dev/runs/perf/${SLURM_JOB_ID:-local}-adjab
mkdir -p "$OUT"
echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local} WT=$WT out=$OUT flags: $* ==="
names=()
for v in $VARIANTS; do
    name=${v%%:*}; vars=${v#*:}; names+=("$name")
    echo "--- $name [${vars}] $(date +%T)"
    env $(echo "$vars" | tr '+' ' ') "$PY" scripts/perf/adj_profile.py time "$OUT/$name" "$@" 2>&1 | grep -v cuda_vmm | tail -n 2 | cut -c1-300
done
"$PY" - "$OUT" "${names[@]}" <<'PY'
import json, sys
out, names = sys.argv[1], sys.argv[2:]
rec = {n: json.load(open(f"{out}/{n}/time.json")) for n in names}
b = rec[names[0]]
for n in names:
    r = rec[n]
    dg = max(abs(r["grad"][k] - b["grad"][k]) / max(abs(b["grad"][k]), 1e-300) for k in b["grad"])
    print(f"{n:12s} fwd {r['fwd_ms_per_step']:7.2f} ms/step  bwd {r['bwd_ms_per_step']:7.2f} ms/step  "
          f"sum {r['fwd_ms_per_step'] + r['bwd_ms_per_step']:7.2f}  vs {names[0]} "
          f"{100 * (r['fwd_ms_per_step'] + r['bwd_ms_per_step']) / (b['fwd_ms_per_step'] + b['bwd_ms_per_step']) - 100:+5.1f} %  "
          f"grad max rel {dg:.1e}  compile fwd {r['first_fwd_s']:.0f} s bwd {r['first_bwd_s']:.0f} s  peak {max(r['peak_gb'])} GB")
PY
echo "=== done $(date) ==="
