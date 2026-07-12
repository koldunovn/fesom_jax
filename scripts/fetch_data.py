#!/usr/bin/env python
"""Download the CORE2 one-year data package from Zenodo.

    python scripts/fetch_data.py --dest ~/fesom-data            # mesh + IC + 1958 forcing
    python scripts/fetch_data.py --dest ~/fesom-data --mesh-only  # skip the 10.4 GB forcing

Then point the model at it (add to your shell profile / job script):

    export FESOM_JRA_DIR=~/fesom-data/core2_forcing_1958
    export FESOM_SSS_PATH=~/fesom-data/core2_mesh_ic/PHC2_salx.nc
    export FESOM_RUNOFF_PATH=~/fesom-data/core2_mesh_ic/CORE2_runoff.nc
    export FESOM_CHL_PATH=~/fesom-data/core2_mesh_ic/Sweeney_2005.nc
    export FESOM_PHC_PATH=~/fesom-data/core2_mesh_ic/phc3.0_winter.nc
    export FESOM_IC_DIR=~/fesom-data/core2_mesh_ic/ic_core2

Or run `--print-env` to have those printed for you.

NOTE — the **pi** mesh needs none of this; it ships inside the package. This script is only
for the realistic CORE2 setup (examples/02_core2_realistic.ipynb). See docs/DATA.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# The Zenodo record holding the package. Override with $FESOM_ZENODO_RECORD or --record.
ZENODO_RECORD = os.environ.get("FESOM_ZENODO_RECORD", "")
ZENODO_API = "https://zenodo.org/api/records/{record}"

MESH_ZIP = "core2_mesh_ic.zip"
FORCING_ZIP = "core2_forcing_1958.zip"


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def record_files(record: str) -> dict:
    """{filename: {'url':…, 'size':…, 'md5':…}} for a Zenodo record."""
    url = ZENODO_API.format(record=record)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            meta = json.load(r)
    except Exception as e:  # noqa: BLE001
        _die(f"could not reach Zenodo record {record}: {e}")
    out = {}
    for f in meta.get("files", []):
        # the API shape has drifted over the years; accept both
        name = f.get("key") or f.get("filename")
        link = (f.get("links", {}) or {}).get("self") or (f.get("links", {}) or {}).get("download")
        out[name] = {"url": link, "size": f.get("size", 0),
                     "md5": (f.get("checksum") or "").replace("md5:", "")}
    return out


def download(url: str, dest: Path, size: int = 0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and size and dest.stat().st_size == size:
        print(f"  {dest.name}: already downloaded")
        return
    print(f"  {dest.name}: downloading ({size / 2**30:.2f} GB)..." if size
          else f"  {dest.name}: downloading...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if size:
                pct = 100 * done / size
                print(f"\r    {pct:5.1f}%  {done / 2**30:.2f}/{size / 2**30:.2f} GB",
                      end="", flush=True)
        if size:
            print()


def md5sum(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 — Zenodo publishes md5; integrity, not security
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def unzip(path: Path, dest: Path) -> None:
    print(f"  {path.name}: unpacking...", flush=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(dest)


def print_env(dest: Path) -> None:
    m = dest / "core2_mesh_ic"
    print("\n# add these to your shell profile or job script:")
    print(f"export FESOM_JRA_DIR={dest / 'core2_forcing_1958'}")
    print(f"export FESOM_SSS_PATH={m / 'PHC2_salx.nc'}")
    print(f"export FESOM_RUNOFF_PATH={m / 'CORE2_runoff.nc'}")
    print(f"export FESOM_CHL_PATH={m / 'Sweeney_2005.nc'}")
    print(f"export FESOM_PHC_PATH={m / 'phc3.0_winter.nc'}")
    print(f"export FESOM_IC_DIR={m / 'ic_core2'}")
    print(f"export FESOM_MESH_DIR={m / 'mesh_core2'}      # for the notebooks")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, type=Path, help="where to put the data")
    ap.add_argument("--record", default=ZENODO_RECORD, help="Zenodo record id")
    ap.add_argument("--mesh-only", action="store_true",
                    help="fetch only the ~370 MB mesh/IC archive, not the 10.4 GB forcing")
    ap.add_argument("--keep-zips", action="store_true", help="do not delete the zips after unpacking")
    ap.add_argument("--print-env", action="store_true",
                    help="just print the export lines for an existing download and exit")
    args = ap.parse_args()

    dest = args.dest.expanduser().resolve()
    if args.print_env:
        print_env(dest)
        return 0

    if not args.record:
        _die("no Zenodo record id. Pass --record ID, or set $FESOM_ZENODO_RECORD.\n"
             "       (The record id is the number at the end of the Zenodo URL.)")

    files = record_files(args.record)
    want = [MESH_ZIP] if args.mesh_only else [MESH_ZIP, FORCING_ZIP]
    missing = [w for w in want if w not in files]
    if missing:
        _die(f"record {args.record} does not contain {missing}. It has: {sorted(files)}")

    dest.mkdir(parents=True, exist_ok=True)
    for name in want:
        info = files[name]
        zpath = dest / name
        download(info["url"], zpath, info["size"])
        if info["md5"]:
            print(f"  {name}: verifying...", flush=True)
            got = md5sum(zpath)
            if got != info["md5"]:
                _die(f"{name} checksum mismatch (got {got}, expected {info['md5']}) — "
                     "delete it and re-run")
        unzip(zpath, dest)
        if not args.keep_zips:
            zpath.unlink()

    print(f"\ndone -> {dest}")
    print_env(dest)
    if args.mesh_only:
        print("\n(--mesh-only: the JRA55 forcing was NOT fetched, so a CORE2 run will not "
              "start yet. Re-run without --mesh-only.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
