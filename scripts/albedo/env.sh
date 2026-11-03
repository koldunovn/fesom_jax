# Albedo: the fesom-jax input-data environment (docs/recom/RUNNING_ALBEDO.md). `source scripts/albedo/env.sh`.
# Resolution order in the model: explicit arg / run-YAML key > these variables > Levante defaults.
DATA=/albedo/work/projects/p_specbic/sloza/fesom_jax_data
export FESOM_JRA_DIR=$DATA/forcing_1958_mslp                     # 8 Zenodo 1958 fields + psl.1958 (JRA55-do v1.3.1)
export FESOM_SSS_PATH=$DATA/core2_mesh_ic/PHC2_salx.nc
export FESOM_RUNOFF_PATH=$DATA/core2_mesh_ic/CORE2_runoff.nc
export FESOM_CHL_PATH=$DATA/core2_mesh_ic/Sweeney_2005.nc
export FESOM_PHC_PATH=$DATA/core2_mesh_ic/phc3.0_winter.nc
export FESOM_IC_DIR=$DATA/core2_mesh_ic/ic_core2                 # the cached PHC T/S on the CORE2 nodes
export FESOM_MESH_DIR=$DATA/core2_mesh_ic/mesh_core2
export FESOM_DIST_DIR=$DATA/core2_partitions
# REcoM's TEOS-10 SAAR table (fesom_jax.recom.sinking.load_saar_table): copied from the FESOM2-REcoM fork's
# src/int_recom/recom/gsw_mod_saar_data.F90 (the GSW data table; the REcoM 692f804 tree itself is not on Albedo)
export FESOM_RECOM_SRC=$DATA/recom_src
