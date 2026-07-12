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

# The Zenodo record holding the package: https://doi.org/10.5281/zenodo.21324319
# (concept DOI 10.5281/zenodo.21324318 always resolves to the newest version).
# Override with $FESOM_ZENODO_RECORD or --record.
ZENODO_RECORD = os.environ.get("FESOM_ZENODO_RECORD", "21324319")
ZENODO_API = "https://zenodo.org/api/records/{record}"

# Zenodo's edge returns an HTML 403 to any request with no User-Agent. urllib sets one by default,
# but say it explicitly -- this exact trap cost a day on the upload side.
UA = {"User-Agent": "fesom-jax-fetch/1.0 (+https://github.com/koldunovn/fesom_jax)"}

MESH_ZIP = "core2_mesh_ic.zip"
FORCING_ZIP = "core2_forcing_1958.zip"


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def record_files(record: str) -> dict:
    """{filename: {'url':…, 'size':…, 'md5':…}} for a Zenodo record."""
    url = ZENODO_API.format(record=record)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
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
    # A \r progress bar is fine on a terminal but turns a redirected log into thousands of lines,
    # so off a tty report sparsely (every 10%) instead.
    tty = sys.stdout.isatty()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        done, next_mark = 0, 10
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if not size:
                continue
            pct = 100 * done / size
            if tty:
                print(f"\r    {pct:5.1f}%  {done / 2**30:.2f}/{size / 2**30:.2f} GB",
                      end="", flush=True)
            elif pct >= next_mark:
                print(f"    {next_mark:3d}%  {done / 2**30:.2f}/{size / 2**30:.2f} GB", flush=True)
                next_mark += 10
        if size and tty:
            print()


def md5sum(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 — Zenodo publishes md5; integrity, not security
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def unzip(path: Path, dest: Path) -> None:
    print(f"  {path.name}: unpacking...", flush=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(dest)


def reassemble(archive: dict, files: dict, dest: Path) -> Path:
    """Download an archive's .partNNN files and cat them back into one file.

    The archives are split on Zenodo because Zenodo has no resumable upload and drops long PUTs
    (see scripts/upload_to_zenodo.py). That is an upload-side workaround; nobody downloading should
    have to care, so we hide it here -- download the parts, concatenate, verify the whole-file
    sha256 from MANIFEST.json.
    """
    name, parts = archive["name"], archive["parts"]
    out = dest / name
    if out.exists() and out.stat().st_size == archive["size"]:
        print(f"  {name}: already assembled")
        return out

    # Not split at all: the archive's only "part" IS the archive. Just download it (concatenating
    # would mean reading and writing the same path).
    if parts == [name]:
        if name not in files:
            _die(f"record is missing {name}")
        download(files[name]["url"], out, files[name]["size"])
        print(f"  {name}: verifying sha256...", flush=True)
        got = sha256sum(out)
        if got != archive["sha256"]:
            _die(f"{name}: sha256 mismatch (got {got}, expected {archive['sha256']}) — "
                 "delete it and re-run")
        print(f"  {name}: verified")
        return out

    print(f"  {name}: {len(parts)} part(s), {archive['size'] / 2**30:.2f} GB")
    part_paths = []
    for i, p in enumerate(parts, 1):
        if p not in files:
            _die(f"record is missing part {p} of {name}")
        pp = dest / p
        print(f"    part {i}/{len(parts)}", end=" ", flush=True)
        download(files[p]["url"], pp, files[p]["size"])
        part_paths.append(pp)

    print(f"  {name}: joining {len(part_paths)} parts...", flush=True)
    with open(out, "wb") as w:
        for pp in part_paths:
            with open(pp, "rb") as r:
                while chunk := r.read(8 << 20):
                    w.write(chunk)
    for pp in part_paths:
        pp.unlink()

    print(f"  {name}: verifying sha256...", flush=True)
    got = sha256sum(out)
    if got != archive["sha256"]:
        _die(f"{name}: sha256 mismatch after reassembly (got {got}, "
             f"expected {archive['sha256']}) -- delete it and re-run")
    print(f"  {name}: verified")
    return out


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
    dest.mkdir(parents=True, exist_ok=True)

    # The record may carry each archive either whole, or split into .partNNN files with a
    # MANIFEST.json saying how to rejoin them (Zenodo drops long uploads, so big archives are
    # uploaded in pieces). Handle both; the split is invisible from here.
    manifest = None
    if "MANIFEST.json" in files:
        mpath = dest / "MANIFEST.json"
        download(files["MANIFEST.json"]["url"], mpath, files["MANIFEST.json"]["size"])
        manifest = json.loads(mpath.read_text())

    for name in want:
        if manifest and any(a["name"] == name for a in manifest["archives"]):
            archive = next(a for a in manifest["archives"] if a["name"] == name)
            zpath = reassemble(archive, files, dest)
        elif name in files:
            info = files[name]
            zpath = dest / name
            download(info["url"], zpath, info["size"])
            if info["md5"]:
                print(f"  {name}: verifying...", flush=True)
                got = md5sum(zpath)
                if got != info["md5"]:
                    _die(f"{name} checksum mismatch (got {got}, expected {info['md5']}) — "
                         "delete it and re-run")
        else:
            _die(f"record {args.record} has neither {name} nor its parts. It has: {sorted(files)}")

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
