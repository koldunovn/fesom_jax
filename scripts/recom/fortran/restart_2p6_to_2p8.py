"""A FESOM 2.8 view of a FESOM 2.6 netCDF restart (docs/plans/completed/20261102-fortran-spectral-on-fesom28.md, step 5.5).

FESOM 2.6's io_restart named a tracer by its POSITION j, except that positions 6, 11, 12, 14 and 39 got the
transient-tracer names sf6, cfc11, cfc12, r14c, r39ar (the SELECT CASE ran on j). FESOM 2.8 selects on the tracer
ID, so in a REcoM run every BGC tracer is tra<j:04d>. This writes DST/fesom.<year>.{oce,ice,bio}.restart where
the oce files of those positions (value and _M1) are copied with file and variable renamed to tra<j:04d>, and
everything else is a symlink to SRC. _AB files are linked too (2.8 does not read the tracer _AB fields).

    python restart_2p6_to_2p8.py SRC_DIR DST_DIR [YEAR]          # e.g. restart_1997_4pft restart_1997_4pft_fesom28 1997

Only valid when positions 6/11/12/14/39 hold BGC tracers (no transient tracers in the run), as in the REcoM runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import netCDF4

RENAME = {"sf6": 6, "cfc11": 11, "cfc12": 12, "r14c": 14, "r39ar": 39}


def copy_renamed(src: Path, dst: Path, old: str, new: str):
    with netCDF4.Dataset(src) as a, netCDF4.Dataset(dst, "w", format=a.data_model) as b:
        b.setncatts({k: a.getncattr(k) for k in a.ncattrs()})
        for name, dim in a.dimensions.items():
            b.createDimension(name, None if dim.isunlimited() else len(dim))
        for name, v in a.variables.items():
            out = new if name == old else name
            w = b.createVariable(out, v.datatype, v.dimensions,
                                 fill_value=v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else None)
            w.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
            w[:] = v[:]


def main(src: Path, dst: Path, year: str):
    dst.mkdir(parents=True, exist_ok=False)
    for comp in ("ice", "bio"):
        (dst / f"fesom.{year}.{comp}.restart").symlink_to((src / f"fesom.{year}.{comp}.restart").resolve())
    so, do = src / f"fesom.{year}.oce.restart", dst / f"fesom.{year}.oce.restart"
    do.mkdir()
    renamed = []
    for f in sorted(so.iterdir()):
        stem, suffix = f.stem, ""
        for s in ("_M1", "_AB"):
            if stem.endswith(s):
                stem, suffix = stem[: -len(s)], s
        if stem in RENAME and suffix != "_AB":
            new = f"tra{RENAME[stem]:04d}{suffix}"
            copy_renamed(f, do / f"{new}.nc", stem + suffix, new)
            renamed.append(f"{f.name} -> {new}.nc")
        else:
            (do / f.name).symlink_to(f.resolve())
    (dst / "README.txt").write_text(
        f"FESOM 2.8 view of {src} (restart_2p6_to_2p8.py, fesom_jax_recom scripts/recom/fortran):\n"
        "2.6 named positions 6/11/12/14/39 sf6/cfc11/cfc12/r14c/r39ar, 2.8 names every BGC tracer tra<pos>.\n"
        "Copied with file and variable renamed:\n  " + "\n  ".join(renamed) + "\nEverything else: symlinks to the source.\n")
    print("\n".join(renamed))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "1997")
