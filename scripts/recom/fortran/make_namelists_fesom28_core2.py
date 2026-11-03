"""Build the CORE2 warm-1998 namelist set of the spectral REcoM on FESOM 2.8 (sveta-loza/fesom2
fesom2.8_recom_spectral, docs/plans/completed/20261102-fortran-spectral-on-fesom28.md, step 5.5).

The physics, REcoM setup and file format are those of ``namelists_albedo_fesom28_spectral/`` (the pi set whose twin
passes G4b at 4.1e-8: GM/Redi on, cvmix TKE, mEVP, ...); the CORE2 inputs, calendar and restart are those of the
user's physics-matched warm run ``/albedo/work/user/sloza/SPECTRAL/tmp_run_kdind15wjp`` (2.6 format), key by key:

    python scripts/recom/fortran/make_namelists_fesom28_core2.py [OUT_DIR]

``OUT_DIR`` defaults to ``scripts/recom/fortran/namelists_albedo_fesom28_core2_warm1998``. ``@RESULT@`` stays a
placeholder (the job script fills it); the other placeholders of the pi set are resolved here.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "namelists_albedo_fesom28_spectral"
OLD = Path("/albedo/work/user/sloza/SPECTRAL/tmp_run_kdind15wjp")      # the 2.6-format partner run
MARESYS = "/albedo/work/user/anbros001/Maresys/"
POOL = "/albedo/work/projects/p_pool_fesom2/forcing/CORE2/"

# file -> group -> {key: value}; a key absent from the group is added before its closing '/'
SET = {
    "namelist.config": {
        "timestep": {"run_length": "12", "run_length_unit": "'m'"},
        "clockinit": {"yearnew": "1997"},           # the clock (fesom.clock) seeds 1 Jan 1998: a warm start
        "paths": {"MeshPath": "'/albedo/pool/fesom2/core2/'", "ClimateDataPath": f"'{MARESYS}'"},
        "restart_log": {"logfile_outfreq": "960"},
        "calendar": {"include_fleapyear": ".false."},
        "machine": {"n_levels": "2", "n_part": "12, 36"},
    },
    "namelist.forcing": {
        "nam_sbc": {**{f"nm_{v}_file": f"'{MARESYS}{f}.clim61'" for v, f in [
            ("xwind", "uas"), ("ywind", "vas"), ("xstre", "uas"), ("ystre", "vas"), ("humi", "huss"), ("qsr", "rsds"),
            ("qlw", "rlds"), ("tair", "tas"), ("prec", "prra"), ("snow", "prsn"), ("mslp", "psl")]},
            "nm_nc_iyear": "1900", "y_perpetual": ".true.",
            "nm_runoff_file": f"'{POOL}runoff.nc'", "runoff_data_source": "'CORE2'",
            "nm_sss_data_file": f"'{POOL}PHC2_salx.nc'",
            "chl_data_source": "'None'", "chl_const": "0.1"},
    },
    "namelist.recom": {
        "nam_rsbc": {"nm_fe_data_file": f"'{MARESYS}DustClimMonthlyAlbani.nc'",
                     "nm_co2_data_file": f"'{MARESYS}MonthlyAtmCO2_gcb2024.nc'"},
        "pavariables": {"REcoM_restart": ".true.", "REcoMDataPath": f"'{MARESYS}'"},
    },
}
# whole groups taken from the partner run (2.6 format, identical keys)
GROUPS_FROM_OLD = {"namelist.tra": ["tracer_init3d"], "namelist.io": ["nml_list"]}


def groups(text):
    return {m.group(1).lower(): m for m in re.finditer(r"^&(\w+)\s*(?:!.*)?$.*?^/\s*$", text, re.M | re.S)}


def set_key(text, group, key, value):
    g = groups(text).get(group.lower())
    if g is None:
        raise KeyError(f"group &{group} not found")
    body = g.group(0)
    pat = re.compile(rf"^(\s*){key}\s*=.*$", re.M | re.I)
    if pat.search(body):
        body = pat.sub(lambda m: f"{m.group(1)}{key} = {value}", body, count=1)
    else:
        body = re.sub(r"^/\s*$", f"{key} = {value}\n/", body, count=1, flags=re.M)
    return text[:g.start()] + body + text[g.end():]


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob("namelist.*"):
        shutil.copy(f, out / f.name)
    for fname, grps in SET.items():
        t = (out / fname).read_text()
        for grp, kv in grps.items():
            for k, v in kv.items():
                t = set_key(t, grp, k, v)
        (out / fname).write_text(t)
    for fname, grps in GROUPS_FROM_OLD.items():
        t, o = (out / fname).read_text(), (OLD / fname).read_text()
        for grp in grps:
            gn, go = groups(t)[grp], groups(o)[grp]
            t = t[:gn.start()] + go.group(0) + t[gn.end():]
        (out / fname).write_text(t)
    left = sorted({m for f in out.glob("namelist.*") for m in re.findall(r"@[A-Z_]+@", f.read_text())})
    assert left == ["@RESULT@"], left
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "namelists_albedo_fesom28_core2_warm1998")
