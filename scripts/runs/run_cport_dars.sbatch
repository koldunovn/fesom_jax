#!/bin/bash
# Run the C PORT (zstar) on dars at dt=180 to confirm the documented multi-rank halo blowup
# (MPI_PORT_REPORT: eta/w diverge at partition boundaries -> SSH CG NaN). 128 ranks (dist_128).
#SBATCH --job-name=cport_dars
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --time=00:30:00
#SBATCH -o /home/a/a270088/port_jax/scripts/run_cport_dars.%j.out
#SBATCH -e /home/a/a270088/port_jax/scripts/run_cport_dars.%j.out
source /sw/etc/profile.levante 2>/dev/null
source /home/a/a270088/port2/fesom2_port_zstar/env.sh 2>/dev/null
ulimit -s 204800
BIN=/home/a/a270088/port2/fesom2_port_zstar/build/fesom_port
MESH=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/dars
PHC=/home/a/a270088/FESOM_port/fesom2/tests/data/INITIAL/phc3.0/phc3.0_winter.nc
OUT=/work/ab0995/a270088/port/dars_dt180_cport; mkdir -p "$OUT"
export FESOM_MIX_SCHEME=KPP FESOM_PRINT_EVERY=10
echo "=== $(date) C-port dars dt=180, 128 ranks, 200 steps ==="
# mesh out dt nsteps snap phc jra_year
srun -l --ntasks=128 "$BIN" "$MESH" "$OUT" 180 200 100 "$PHC" 1958 2>&1 | tail -40
echo "cport exit=$?"
