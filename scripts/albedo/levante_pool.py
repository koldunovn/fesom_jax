#!/usr/bin/env python3
"""The BGC data pool for fesom-jax on Levante (GH200), built from Albedo: one path map, two jobs.

    python scripts/albedo/levante_pool.py sizes                     # what would be copied, with sizes (Albedo)
    python scripts/albedo/levante_pool.py rsync --user b123456      # print the rsync commands (run them on Albedo)
    python scripts/albedo/levante_pool.py rsync --user b123456 --run
    python scripts/albedo/levante_pool.py configs --user b123456    # write configs/levante/*.yaml for that pool

The pool sits at ``/work/bk1415/<user>/fesom_jax_pool`` (``--root`` to change). Its layout is :data:`POOL`: each entry
maps an Albedo source (a file or a directory, symlinks followed) to a path under the root. The Levante configs are the
Albedo ones with every Albedo path rewritten through the same map, so the two cannot drift. Plan
``docs/plans/20261105-recom-parameter-estimation.md`` (the CORE2 gradient needs the GH200s, decision D3).

The copy runs from Albedo to Levante over ssh (``levante.dkrz.de``); it needs the user's Levante login, which is why
this script only prints the commands unless ``--run`` is given.
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOST = "levante.dkrz.de"
P = "/albedo/work/projects/p_specbic/sloza"
MARESYS = "/albedo/work/user/anbros001/Maresys"
JRA_VARS = ("huss", "prra", "prsn", "psl", "rlds", "rsds", "tas", "uas", "vas")

# (Albedo source, pool path, group). Groups: core2, pi. Files keep their names; directories are copied as a whole.
POOL = [
    *[(f"{MARESYS}/{v}.clim61.nc", f"forcing/jra55_clim61/{v}.clim61.nc", "core2") for v in JRA_VARS],
    *[(f"{MARESYS}/{f}", f"forcing/recom_maresys/{f}", "core2") for f in
      ("DustClimMonthlyAlbani.nc", "MonthlyAtmCO2_gcb2024.nc", "AeolianNitrogenDep.nc", "RiverineInput.nc",
       "ErosionInput.nc")],
    ("/albedo/work/projects/p_pool_fesom2/forcing/CORE2/PHC2_salx.nc", "forcing/core2_pool/PHC2_salx.nc", "core2"),
    ("/albedo/work/projects/p_pool_fesom2/forcing/CORE2/runoff.nc", "forcing/core2_pool/runoff.nc", "core2"),
    ("/albedo/pool/FESOM2/recom", "recom_pool", "core2"),
    (f"{P}/forcing_oasim_v2_rsds", "forcing/oasim_v2_rsds", "core2"),
    (f"{P}/forcing_oasim_v2", "forcing/oasim_v2", "pi"),
    (f"{P}/fesom_jax_data/core2_mesh_ic/mesh_core2", "fesom_jax_data/core2_mesh_ic/mesh_core2", "core2"),
    (f"{P}/fesom_jax_data/core2_mesh_ic", "fesom_jax_data/core2_mesh_ic", "core2"),
    (f"{P}/fesom_jax_data/recom_pi", "fesom_jax_data/recom_pi", "core2"),        # the optics tables live here too
    (f"{P}/fesom_jax_runs/restart_1997_4pft_jax", "restarts/restart_1997_4pft_jax", "core2"),
    *[(f"{P}/fesom_jax_runs/core2_recom_spectral_w28sl1/seg{k}/restart", f"restarts/w28sl1/seg{k}", "core2")
      for k in (3, 4, 5)],
    (f"{P}/fesom_jax_data/core2_mesh_ic/mesh_core2_part", "fesom_jax_data/core2_mesh_ic/mesh_core2_part", "core2"),  # dist_4
    (f"{P}/fesom_jax_data/mesh_pi_part", "fesom_jax_data/mesh_pi_part", "pi"),                                     # dist_4
    (f"{P}/fesom_jax_data/mesh_pi_fork", "fesom_jax_data/mesh_pi_fork", "pi"),
    (f"{P}/fesom_jax_data/ic_pi_fork", "fesom_jax_data/ic_pi_fork", "pi"),
    (f"{P}/fesom_jax_data/forcing_1958_mslp", "fesom_jax_data/forcing_1958_mslp", "pi"),
    (f"{P}/fesom_jax_runs/pi_recom_spectral_spin5/restart", "restarts/pi_spin5", "pi"),
]
# Albedo prefixes rewritten in the configs (longest first), beyond the per-entry map.
PREFIXES = [(f"{MARESYS}/", "forcing/jra55_clim61/"), (f"{P}/fesom_jax_runs/", "runs/")]
CONFIGS = ["configs/albedo/core2_recom_spectral_w28sl1.yaml", "configs/albedo/pi_recom_spectral_spin.yaml"]


def root_of(a) -> str:
    return a.root or f"/work/bk1415/{a.user}/fesom_jax_pool"


def entries(groups):
    return [e for e in POOL if e[2] in groups]


def to_pool(path: str, root: str) -> str:
    """An Albedo path rewritten into the pool, by the longest matching source in :data:`POOL`, then
    :data:`PREFIXES`; the Maresys REcoM files map to forcing/recom_maresys."""
    if path.rstrip("/") == MARESYS:                     # the forcing.jra_dir of the clim61 runs
        return f"{root}/forcing/jra55_clim61"
    best = None
    for src, dst, _ in POOL:
        if path == src or path.startswith(src.rstrip("/") + "/"):
            if best is None or len(src) > len(best[0]):
                best = (src, dst)
    if best:
        return f"{root}/{best[1]}{path[len(best[0]):]}"
    for src, dst in PREFIXES:
        if path.startswith(src):
            rest = path[len(src):]
            if src == f"{MARESYS}/" and not rest.endswith(".clim61.nc"):
                dst = "forcing/recom_maresys/"
            return f"{root}/{dst}{rest}"
    return path


def cmd_sizes(a):
    total = 0
    for src, dst, g in entries(a.groups):
        out = subprocess.run(["du", "-sbL", src], capture_output=True, text=True).stdout.split()
        n = int(out[0]) if out else 0
        total += n
        print(f"{n / 2**30:8.2f} GiB  {g:5s}  {src}  ->  {dst}")
    print(f"{total / 2**30:8.2f} GiB  total")


def cmd_rsync(a):
    root = root_of(a)
    dirs = sorted({str(Path(root, d).parent) for _, d, _ in entries(a.groups)} | {root})
    cmds = [["ssh", f"{a.user}@{HOST}", "mkdir -p " + " ".join(shlex.quote(d) for d in dirs)]]
    for src, dst, _ in entries(a.groups):
        is_dir = Path(src).is_dir()
        s = src.rstrip("/") + ("/" if is_dir else "")
        d = f"{a.user}@{HOST}:{root}/{dst}" + ("/" if is_dir else "")
        cmds.append(["rsync", "-aL", "--partial", "--info=progress2", s, d])
    for c in cmds:
        print(shlex.join(c))
        if a.run:
            r = subprocess.run(c)
            if r.returncode:
                sys.exit(f"failed ({r.returncode}): {shlex.join(c)}")


def cmd_configs(a):
    root = root_of(a)
    out = REPO / "configs" / "levante"
    out.mkdir(exist_ok=True)
    pat = re.compile(r"/albedo/[^\s'\"#]+")
    for rel in CONFIGS:
        text = (REPO / rel).read_text()
        lines = []
        for line in text.splitlines(keepends=True):     # rewrite the values, leave the comments as they are
            code, hash_, comment = line.partition("#")
            lines.append(pat.sub(lambda m: to_pool(m.group(0), root), code) + hash_ + comment)
        body = "".join(lines)
        if "core2" in rel:   # REcoM's CO2 series: the namelist names it on Albedo, the config must name it here
            body = body.replace("  oasim_dir:", f"  co2_file: {root}/forcing/recom_maresys/MonthlyAtmCO2_gcb2024.nc\n"
                                                "  oasim_dir:", 1)
        left = [m for line in body.splitlines() for m in pat.findall(line.partition("#")[0])]
        if left:
            sys.exit(f"{rel}: Albedo paths left unmapped: {left}")
        head = (f"# Levante (GH200) version of {rel}, written by scripts/albedo/levante_pool.py for the pool {root}.\n"
                f"# Edit the Albedo config and rerun the script instead of editing this file.\n")
        dst = out / Path(rel).name
        dst.write_text(head + body)
        print(f"wrote {dst.relative_to(REPO)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("sizes", "rsync", "configs"))
    ap.add_argument("--user", default="<levante-user>", help="the Levante login (e.g. b123456)")
    ap.add_argument("--root", default=None, help="the pool root on Levante (default /work/bk1415/<user>/fesom_jax_pool)")
    ap.add_argument("--groups", nargs="+", default=["core2", "pi"], choices=("core2", "pi"))
    ap.add_argument("--run", action="store_true", help="rsync: run the commands, not only print them")
    a = ap.parse_args()
    {"sizes": cmd_sizes, "rsync": cmd_rsync, "configs": cmd_configs}[a.what](a)


if __name__ == "__main__":
    main()
