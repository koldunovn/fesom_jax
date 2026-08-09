#!/bin/bash
# NN-GM — how far does the adjoint actually reach? The long-window ladder.
#
# Chunked reverse accumulation makes the DEVICE cost flat in window length, so what limits a
# long gradient is (a) host RAM for the chunk boundaries, ~1.93 GB per saved day, (b) wall
# time, ~85 s per simulated day for one loss+gradient, and (c) the physics: the cotangent
# amplification measured at 1.0008/step predicts exp(0.0384*N_days) growth.
#
#   exp(0.0384 N):  30 d -> 3.2x    60 d -> 10x    90 d -> 32x   180 d -> 1e3x   365 d -> 1.2e6x
#
# Carrying |dL/dm| = 0.342 at 30 d forward, that crosses the BLOW=1e4 guard near 300 days.
# The ladder tests that prediction, made in advance, at 10x and 32x first.
#
# ⚠️ ALL RUNGS IN ONE BATCH MUST SHARE `--truth-days`. The per-level inverse-variance weights
# are computed from the truth marks, so a different --truth-days is a different loss function
# and the |grad| values are not comparable across rungs (the flag's own help says so). The
# amplification RATE is computed within a run and is comparable either way.
#
# ⚠️ 80 GB CARDS ONLY. Four nodes in this partition carry a100_40; `--gres=gpu:a100_80:1` is
# what keeps us off them. Never relax it — 40 GB does not fit even a 1-day reverse chunk.
#
# ⚠️ HOST MEMORY IS ALLOCATED PER CPU HERE (DefMemPerCPU=1940, MaxMemPerCPU=3940 MB), so the
# boundary store must be paid for with --cpus-per-task. 90 d = 91 x 1.93 = 176 GB.
#
# Usage:
#   NDAYS_LIST=30 TRUTH=30 sbatch scripts/nn_eddy/adjoint_horizon_ladder.sbatch   # validation
#   NDAYS_LIST="60 90" TRUTH=90 sbatch scripts/nn_eddy/adjoint_horizon_ladder.sbatch
#SBATCH --job-name=nn_adj_ladder
#SBATCH -A ab0995_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH -o scripts/nn_eddy/adjladder.%j.out
#SBATCH -e scripts/nn_eddy/adjladder.%j.out
source /sw/etc/profile.levante
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
cd "${SLURM_SUBMIT_DIR:-/home/a/a270088/nn_eddy/port_jax}" || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
SNAP=${SNAP:-/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl}
SEASON=${SEASON:-winterNH}
NDAYS_LIST=${NDAYS_LIST:-"60 90"}
TRUTH=${TRUTH:-90}
CHUNK=${CHUNK:-48}
# ⚠️ `${FDH-...}`, NOT `${FDH:-...}`. The colon form substitutes the default when the variable
# is empty as well as when it is unset, so `FDH=""` — the documented way to skip the
# finite-difference check — silently ran it anyway. Each h costs TWO extra full forwards: 26
# minutes on a 60-day window (job 26819415, which looked hung for exactly that long), 79 on a
# 180-day one. Without the colon, empty means empty and unset still gets the default.
FDH=${FDH-1e-3}
# ⚠️ THREE THINGS ARE FROZEN IN THE BACKWARD, AND ONLY TWO OF THEM ARE IN THIS VARIABLE.
# The third is the sea-ice DYNAMICS, which lives in `ICE` below and is spelled `rheology` — a
# name that reads as "switch the rheology on" and means "do not differentiate it". Do not
# conclude from this block that the freezing is complete; see the ICE block too.
#
# THE FREEZES ARE THE EXPERIMENT, not a tuning choice. `--freeze` takes a COMMA LIST of
# independent knobs; the default here is TWO of them, and each answers a different failure.
#
#   mix    vertical mixing coefficients Kv, Av from the TKE closure — use the values the closure
#          computes, omit their dependence on the ocean state (the documented ECCO practice).
#          WITHOUT IT nothing beyond ~2 days runs at all: every clean long-window record in
#          adjoint_horizon_results.jsonl carries it, the one 30-day run with freeze:[] is tagged
#          "month-unfrozen-contrast" and returned +1.64e+154, and omitting it cost two jobs
#          (26800812/3, cancelled at |cot| = 1.8e57 by day 20).
#   slope  N^2 inside the GM block only — the 1/N^2 neutral-slope amplifier. k_gm (the parameter
#          we differentiate, and the NN's output) and T,S stay LIVE. ADDED 2026-08-09: `mix`
#          alone still blows on any window ending in February, because a destratifying winter
#          column drives d(sigma)/dz toward zero and the slope's state-sensitivity with it. On
#          the 60-day probe ending 1961-02-14 the cotangent goes x1406 in the single chunk of
#          1961-02-06 (cumulative x4.363e+07 over nine chunks); adding `slope` takes that to
#          x1.591 and returns dL/dm = -9.413328e-01 against the full model's finite difference
#          of -9.803512e-01, a 3.98 % residual. Also the documented ECCO recipe — it blocked
#          GM/Redi in the MITgcm adjoint until 2008 (Forget et al. 2015; docs/ADJOINT_HORIZON.md
#          section 2b).
#
# `adjoint_horizon.py`'s OWN default is still no knobs at all, because its other job is
# attribution, where unfrozen 1-2 day contrasts are the point. Pass FREEZE=mix explicitly here
# for a deliberate contrast against the pre-2026-08-09 baseline.
FREEZE=${FREEZE-mix,slope}
# ⚠️ THIS IS THE THIRD FREEZE — the sea-ice DYNAMICS — and it is not in `FREEZE` above.
# `rheology` means WITH ice, WITHOUT the rheology sweeps: ice thermodynamics, advection,
# cut-off and the ocean-to-ice coupling are all differentiated, and ZERO mEVP relaxation sweeps
# are unrolled in the backward (the forward still runs all 120, so the trajectory is the
# validated one). The option names the component being APPROXIMATED, not the one switched on,
# and misreading it is why every rung at 60, 90, 180, 365 and 731 days was measured with no ice
# in the backward at all — in a campaign whose blow-ups all land in February, peak NH ice.
# The script now REFUSES `--ice frozen` past 2 days without --allow-frozen-ice. Use
# ICE=frozen EXTRA_ARGS=--allow-frozen-ice only for a deliberate ice-on/ice-off contrast.
# Mechanically it is `adjoint_subcycles=0` (ice_step.py:102): the forward runs all 120 mEVP
# subcycles, the backward unrolls none, so the ice velocity has no gradient path. The one
# published alternative that keeps EVP rheology in the adjoint — Newtonian damping of the
# subcycle carry — was implemented, measured and rejected here (1.1795 amplification per sweep
# swings the 120-sweep result from 1e-44 at T_N=6 to 1e+27 at T_N=7). MITgcm's free-drift
# adjoint was considered and declined: it overstates ice-velocity sensitivity by O(10^2) in the
# central Arctic (Toyoda et al. 2019 via Lyu et al. 2023). docs/ADJOINT_HORIZON.md section 2c.
ICE=${ICE:-rheology}
RES=${RES:-scripts/nn_eddy/adjoint_horizon_results.jsonl}
# Share ONE weight set across every rung, from the existing 30-day truth cache. Weights are
# derived from the truth marks, so a per-rung --truth-days is a per-rung LOSS FUNCTION and
# |dL/dm| would not be comparable between window lengths. It also removes the per-day mark
# accumulation that OOM-killed the 60/90 job at 240 GB (26801648).
# Same colon trap as FDH above: with `:-`, WEIGHTS="" would silently fall back to the shared
# weight set, so there would be no way to ask for per-rung weights. The `${WEIGHTS:+...}` at the
# call site already handles empty correctly; this is the half that did not.
WEIGHTS=${WEIGHTS-/work/ba1583/a270088/nn_eddy/horizon_truth/snap_step037440_m1.25_30d.npz}
echo "=== $(hostname) $(date) commit $(git rev-parse --short HEAD) job=$SLURM_JOB_ID ==="
echo "=== ladder N=[$NDAYS_LIST] d, truth=$TRUTH d, chunk=$CHUNK steps ==="
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
free -g | head -2
# Two-level checkpointing on the HOST. Boundaries cost 1.93 GB per simulated day, so a year is
# 706 GB against a 515 GB node and does NOT fit at stride 1; keeping every STRIDE-th boundary
# and recomputing the spans between costs exactly ONE extra forward pass over the window and
# takes a year to ~74 GB at the sqrt(N) optimum (stride 19). Validated at strides 1/2/3:
# dL/dm agreed to 3e-5, better than the model's own 6e-4 forward reproducibility.
STRIDE=${STRIDE:-1}
for N in $NDAYS_LIST; do
  echo "----- N=$N d (truth $TRUTH d, boundary-stride $STRIDE) -----"
  "$PY" scripts/nn_eddy/adjoint_horizon.py --snap "$SNAP" --season "$SEASON" \
      --n-days "$N" --truth-days "$TRUTH" --chunk-steps "$CHUNK" --fd-h "$FDH" \
      --boundary-stride "$STRIDE" \
      --freeze "$FREEZE" --ice "$ICE" ${WEIGHTS:+--weights-from "$WEIGHTS"} \
      --run-start-year 1958 --results "$RES" --tag "${TAG:-ladder}-N${N}" ${EXTRA_ARGS:-}
  echo "  (N=$N exit $?)"
done
echo "=== done $(date) ==="
