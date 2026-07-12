#!/usr/bin/env python
"""Build the Zenodo data package: everything needed to run CORE2 for one year.

Two archives (see docs/DATA.md):

  core2_mesh_ic.zip       ~370 MB  mesh (raw + dense), cached PHC initial state,
                                   SSS restoring, river runoff, chlorophyll
  core2_forcing_1958.zip  ~10.4 GB the 8 JRA55-do fields for 1958

MESH PROVENANCE — the load-bearing detail. The CORE2 level files on /pool
(``nlvls.out`` / ``elvls.out``) were regenerated on 2026-07-03, AFTER this project's
mesh export (2026-06-06). They differ at 2 nodes and 4 elements in the Ross Sea
(~154 W, 77 S), where the newer files are shallower. Every fesom-jax result was
produced on the PRE-fix mesh, so that is what we ship — and since /pool overwrote
those files, this project's dense .npy export is the only surviving copy. We therefore
reconstruct the raw pre-fix ``nlvls.out``/``elvls.out`` FROM the export rather than
copying /pool's (now different) ones. Everything else (nod2d/elem2d/aux3d/edges) is
untouched since 2021/2026-05 and is copied verbatim.

Usage:  python scripts/build_zenodo_package.py --out /work/.../zenodo [--skip-forcing]
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
POOL_MESH = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
POOL_JRA = Path("/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0")
POOL_CHL = Path("/pool/data/AWICM/FESOM2/FORCING/Sweeney/Sweeney_2005.nc")
POOL_PHC = Path("/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc")

JRA_VARS = ("uas", "vas", "huss", "rsds", "rlds", "tas", "prra", "prsn")
YEAR = 1958

# Topology / geometry files that the 2026-07-03 level fix did NOT touch.
RAW_VERBATIM = ("nod2d.out", "elem2d.out", "aux3d.out",
                "edges.out", "edge_tri.out", "edgenum.out")


def sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(root: Path) -> None:
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "CHECKSUMS.sha256":
            lines.append(f"{sha256(p)}  {p.relative_to(root)}")
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    print(f"    checksums: {len(lines)} files")


def build_mesh_ic(stage: Path) -> None:
    d = stage / "core2_mesh_ic"
    d.mkdir(parents=True, exist_ok=True)

    # -- 1. raw mesh, with the PRE-FIX levels reconstructed from our export -----
    raw = d / "mesh_core2_raw"
    raw.mkdir(exist_ok=True)
    for name in RAW_VERBATIM:
        shutil.copy2(POOL_MESH / name, raw / name)
        print(f"    raw  {name}")

    nlv = np.load(REPO / "data/mesh_core2/nlevels_nod2D.npy").ravel().astype(int)
    elv = np.load(REPO / "data/mesh_core2/nlevels.npy").ravel().astype(int)
    for name, arr in (("nlvls.out", nlv), ("elvls.out", elv)):
        np.savetxt(raw / name, arr, fmt="%12d")
        back = np.loadtxt(raw / name, dtype=int).ravel()
        assert np.array_equal(back, arr), f"{name} did not round-trip!"
        print(f"    raw  {name}  (PRE-FIX, rebuilt from the .npy export, round-trip OK)")

    # sanity: we really are shipping the pre-fix levels, i.e. they differ from /pool's
    pool_nlv = np.loadtxt(POOL_MESH / "nlvls.out", dtype=int).ravel()
    n_diff = int((nlv != pool_nlv).sum())
    assert n_diff == 2, f"expected the known 2-node delta vs /pool, got {n_diff}"
    print(f"    -> confirmed PRE-FIX: differs from current /pool at {n_diff} nodes (expected 2)")

    # -- 2. the dense .npy bundle fesom-jax loads directly ----------------------
    shutil.copytree(REPO / "data/mesh_core2", d / "mesh_core2", dirs_exist_ok=True)
    print("    dense mesh_core2/ (.npy bundle)")

    # -- 3. cached PHC initial state + the source fields ------------------------
    shutil.copytree(REPO / "data/ic_core2", d / "ic_core2", dirs_exist_ok=True)
    print("    ic_core2/ (cached PHC IC)")
    for src in (POOL_PHC, POOL_JRA / "PHC2_salx.nc", POOL_JRA / "CORE2_runoff.nc", POOL_CHL):
        shutil.copy2(src, d / src.name)
        print(f"    {src.name}")

    (d / "README.md").write_text(MESH_README)
    write_checksums(d)


def build_forcing(stage: Path) -> None:
    d = stage / "core2_forcing_1958"
    d.mkdir(parents=True, exist_ok=True)
    for v in JRA_VARS:
        src = POOL_JRA / f"{v}.{YEAR}.nc"
        dst = d / src.name
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            print(f"    {src.name} (already staged)")
            continue
        print(f"    {src.name}  ({src.stat().st_size / 2**30:.2f} GB) ...", flush=True)
        shutil.copy2(src, dst)
    (d / "README.md").write_text(FORCING_README)
    write_checksums(d)


def zip_dir(stage: Path, name: str) -> None:
    """Zip with -0 (store): the netCDFs are already deflated, so recompressing them
    would burn hours for ~nothing."""
    out = stage / f"{name}.zip"
    if out.exists():
        out.unlink()
    print(f"  zipping {name} ...", flush=True)
    subprocess.run(["zip", "-0", "-r", "-q", out.name, name], cwd=stage, check=True)
    print(f"  -> {out.name}  {out.stat().st_size / 2**30:.2f} GB")


MESH_README = """\
# fesom-jax — CORE2 mesh + initial state (1 of 2)

Everything needed to set up the CORE2 (~1 degree, 126,858 node) global ocean, except the
atmospheric forcing, which is the companion archive `core2_forcing_1958.zip`.

## Contents

| Path | What |
|---|---|
| `mesh_core2/` | the mesh as a dense `.npy` bundle — what `fesom-jax` loads directly |
| `mesh_core2_raw/` | the same mesh in standard FESOM2 text format (`nod2d.out`, ...) |
| `ic_core2/` | cached PHC 3.0 initial temperature/salinity on the mesh (`T_ic.npy`, `S_ic.npy`) |
| `phc3.0_winter.nc` | the source PHC 3.0 winter climatology (to regenerate the IC yourself) |
| `PHC2_salx.nc` | sea-surface-salinity restoring field |
| `CORE2_runoff.nc` | river runoff |
| `Sweeney_2005.nc` | chlorophyll climatology (shortwave penetration) |

## !! Mesh version — please read

The CORE2 vertical level files (`nlvls.out`, `elvls.out`) distributed with FESOM2 were
regenerated on 2026-07-03. The version here is the **earlier** one, which is what every
fesom-jax result was produced with. The two differ at exactly **2 nodes and 4 elements**,
all in the Ross Sea (~154 W, 77 S), where this version is deeper (at the largest
difference: 580 m here vs 280 m in the newer files).

Both versions are structurally valid (no element is deeper than its shallowest node). We
ship this one so that published fesom-jax results reproduce exactly. If you need to match
a current FESOM2/Kokkos run instead, take `nlvls.out`/`elvls.out` from the upstream mesh
(https://gitlab.awi.de/fesom/core2) and rebuild the dense bundle with
`scripts/prepare_mesh.py`.

## Mesh citation

Wang, Q., Danilov, S., Sidorenko, D., Timmermann, R., Wekerle, C., Wang, X., Jung, T., and
Schroeter, J.: The Finite Element Sea Ice-Ocean Model (FESOM) v.1.4: formulation of an ocean
general circulation model, Geosci. Model Dev., 7, 663-693,
https://doi.org/10.5194/gmd-7-663-2014, 2014.

Verify the download with `sha256sum -c CHECKSUMS.sha256`.
"""

FORCING_README = """\
# fesom-jax — JRA55-do atmospheric forcing, 1958 (2 of 2)

One year (1958) of JRA55-do v1.4.0 surface forcing: the eight fields `fesom-jax` reads.

| File | Field |
|---|---|
| `uas.1958.nc`, `vas.1958.nc` | 10 m wind components |
| `tas.1958.nc` | 2 m air temperature |
| `huss.1958.nc` | 2 m specific humidity |
| `rsds.1958.nc` | downwelling shortwave radiation |
| `rlds.1958.nc` | downwelling longwave radiation |
| `prra.1958.nc`, `prsn.1958.nc` | rainfall, snowfall |

3-hourly, on the native JRA55 640 x 320 grid. Pair with `core2_mesh_ic.zip`.

## Running more than one year

This archive is one year, to keep the download manageable. For a longer run, fetch the
remaining years of JRA55-do v1.4.0 directly from the source — see `docs/DATA.md` in the
fesom-jax repository for instructions. The file naming (`{var}.{year}.nc`) and the eight
variables above are all `fesom-jax` needs; point `$FESOM_JRA_DIR` at the directory holding
them.

## Forcing citation

Tsujino, H., et al.: JRA-55 based surface dataset for driving ocean-sea-ice models
(JRA55-do), Ocean Modelling, 130, 79-139, https://doi.org/10.1016/j.ocemod.2018.07.002, 2018.

Verify the download with `sha256sum -c CHECKSUMS.sha256`.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="staging directory (needs ~22 GB)")
    ap.add_argument("--skip-forcing", action="store_true",
                    help="build only the small mesh/IC archive")
    ap.add_argument("--no-zip", action="store_true", help="stage the trees but do not zip")
    args = ap.parse_args()

    stage = Path(args.out)
    stage.mkdir(parents=True, exist_ok=True)

    print("[1/2] mesh + initial state")
    build_mesh_ic(stage)
    if not args.no_zip:
        zip_dir(stage, "core2_mesh_ic")

    if not args.skip_forcing:
        print("[2/2] JRA55-do forcing 1958 (~10.4 GB, slow)")
        build_forcing(stage)
        if not args.no_zip:
            zip_dir(stage, "core2_forcing_1958")
    else:
        print("[2/2] skipped (--skip-forcing)")

    print(f"\ndone -> {stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
