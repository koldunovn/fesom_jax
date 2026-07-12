#!/bin/bash
# Stage the observational datasets for the paper's obs-application experiments (Task A1).
#
# OMIP-style, OBS-BASED targets only (NOT data-assimilating reanalyses — see docs/OBS_DATASETS.md
# for the rationale): WOA T/S, de Boyer Montégut MLD, EN4 T/S, OSI-SAF sea-ice. Downloads go to
# /work (large files; NOT /home); existing Levante obs collections are symlinked, not re-fetched.
# Idempotent: existing non-empty files are skipped.
#
# ⚠️ The login-node conda profile sets CURL_CA_BUNDLE to a missing mambaforge cert → every https
# download fails with "error setting certificate file". Point it at the system bundle (below).
#
# Usage:  bash scripts/tools/stage_obs.sh            # keystone set (WOA18 annual T/S, dBM, symlinks)
#         bash scripts/tools/stage_obs.sh --full     # + WOA18 monthly, WOA23, (EN4 raw years TODO)
set -u
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt     # THE fix (system CA, not the conda one)

OBS=${OBS_DIR:-/work/ab0995/a270088/port_jax/obs}
FULL=0; [ "${1:-}" = "--full" ] && FULL=1
mkdir -p "$OBS"/{woa18,woa23,mld_dbm,en4_cmpitool,osisaf_cmpitool}
echo "obs dir: $OBS  (full=$FULL)"

get() {  # get URL OUTFILE  — skip if already present & non-empty
    local url="$1" out="$2"
    if [ -s "$out" ]; then echo "  skip (exists): $(basename "$out")"; return 0; fi
    echo "  fetching: $(basename "$out")"
    curl -sSL --retry 3 --max-time 1800 -o "$out" "$url" \
        && [ -s "$out" ] && echo "    ok ($(du -h "$out" | cut -f1))" \
        || { echo "    FAILED: $url"; rm -f "$out"; return 1; }
}

# --- WOA18 (1°, decav climatology) — annual T/S (the §0/§2 WOA target + SST source) ---
WB=https://www.ncei.noaa.gov/data/oceans/woa/WOA18/DATA
echo "[WOA18] annual T/S (1°)"
get "$WB/temperature/netcdf/decav/1.00/woa18_decav_t00_01.nc" "$OBS/woa18/woa18_decav_t00_01.nc"
get "$WB/salinity/netcdf/decav/1.00/woa18_decav_s00_01.nc"    "$OBS/woa18/woa18_decav_s00_01.nc"
if [ "$FULL" = 1 ]; then
    echo "[WOA18] monthly T/S (1°) — t01..t12 / s01..s12"
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
        get "$WB/temperature/netcdf/decav/1.00/woa18_decav_t${m}_01.nc" "$OBS/woa18/woa18_decav_t${m}_01.nc"
        get "$WB/salinity/netcdf/decav/1.00/woa18_decav_s${m}_01.nc"    "$OBS/woa18/woa18_decav_s${m}_01.nc"
    done
    echo "[WOA23] annual T/S (1°)"
    W23=https://www.ncei.noaa.gov/thredds-ocean/fileServer/woa23/DATA
    get "$W23/temperature/netcdf/decav/1.00/woa23_decav_t00_01.nc" "$OBS/woa23/woa23_decav_t00_01.nc"
    get "$W23/salinity/netcdf/decav/1.00/woa23_decav_s00_01.nc"    "$OBS/woa23/woa23_decav_s00_01.nc"
fi

# --- de Boyer Montégut MLD climatology (DR003 = 0.03 kg/m³ density threshold) ---
# ⚠️ The IFREMER cerweb server is SLOW/flaky — retries + a long timeout; may need re-running.
echo "[dBM] MLD DR003 (0.03 kg/m³ density threshold)"
get "https://www.ifremer.fr/cerweb/deboyer/data/mld_DR003_c1m_reg2.0.nc" "$OBS/mld_dbm/mld_DR003_c1m_reg2.0.nc"

# --- symlink existing Levante obs (do NOT re-download) ---
# EN4 T/S + OSI-SAF sea-ice already live, seasonally averaged, in a colleague's cmpitool collection.
CMPI=/work/ab0995/a270301/cmpitool/obs
echo "[symlink] EN4 T/S + OSI-SAF sea-ice from $CMPI"
if [ -d "$CMPI" ]; then
    ln -sf "$CMPI"/thetao_EN4_*.nc "$CMPI"/so_EN4_*.nc "$OBS/en4_cmpitool/" 2>/dev/null
    ln -sf "$CMPI"/siconc_OSISAF_*.nc                  "$OBS/osisaf_cmpitool/" 2>/dev/null
    echo "    en4: $(ls "$OBS/en4_cmpitool" | wc -l) files; osisaf: $(ls "$OBS/osisaf_cmpitool" | wc -l) files"
else
    echo "    (cmpitool obs not found — see docs/OBS_DATASETS.md for EN4/OSI-SAF raw sources)"
fi

echo ""
echo "=== staged obs (see docs/OBS_DATASETS.md) ==="
du -sh "$OBS"/* 2>/dev/null
echo "OBS_STAGE_DONE"
