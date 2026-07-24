#!/usr/bin/env python
"""Reduce the JUPITER (GH200) strong-scaling logs to per-mesh tables.

    python scripts/bench/jupiter/reduce_jupiter.py [logdir] [--md]

Scans `[bench]` / `[bench-finite]` pairs in `scripts/logs/jupiter/*.out` and emits, per mesh:
ms/step (best of all reps x transports), the winning transport, SYPD at the PRODUCTION dt,
and parallel efficiency against the smallest measured rung.

**J3 is enforced here, not by eye**: a row whose `[bench-finite]` reported any non-finite T/uv
is DROPPED, not reported — "a blown-up ocean times a fake step" (the Levante rule).  Rows with
no `[bench-finite]` at all are dropped too (a killed/hung leg leaves a `[bench]`-less log).

SYPD = dt_prod / (s_step * 365.25).  The production dt is read from the `####` header the
sbatch writes; DT_PROD below is only the fallback for hand-run legs.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DT_PROD = {"core2": 1800.0, "farc": 1200.0, "dars": 240.0, "ng5": 240.0, "forca20": 240.0}

# --------------------------------------------------------------------------------------
# Levante A100-80 reference — the fig10 v2 PRODUCTION-PHYSICS campaign, ms/step, best
# transport per point.  Directly comparable to the JUPITER rows: same code, same jax
# 0.10.1, same protocol (150 steps, >=2 reps, bench-finite gate, per-point banked-best
# transport) and the SAME per-mesh physics/dt (scripts/bench/fig10prod/README.md — verified
# flag-for-flag against scripts/bench/jupiter/bench_scaling_jupiter.sbatch).
# Provenance = the commits that banked each point:
#   core2 1/2/4/8   229/146/88/74   0526a69 (Levante job 26331912)
#   farc  4/8/16    283/199/153     92bb090 (job 26331916)
#   dars  16/32     450/272         012c6a9    dars 64  227  60472a9
#   dars  128       190 (coloured)  9d5e793  — ragged was 292/297 (ce1a51c); best-per-point
#   ng5   32/64     755/438         6d33019 / bf9664a
# NOTE these are NOT the `docs/PARALLELISM.md` "53-57 SYPD on 4xA100" figure: that one is
# the LEGACY ice+kpp+gm model and must never be divided into a production-physics row.
# scripts/bench/jupiter/calib_vs_levante.sbatch measures the legacy config on GH200 for
# exactly that comparison instead.
A100_MS = {
    ("core2", 1): 229.0, ("core2", 2): 146.0, ("core2", 4): 88.0, ("core2", 8): 74.0,
    ("farc", 4): 283.0, ("farc", 8): 199.0, ("farc", 16): 153.0,
    ("dars", 16): 450.0, ("dars", 32): 272.0, ("dars", 64): 227.0, ("dars", 128): 190.0,
    ("ng5", 32): 755.0, ("ng5", 64): 438.0,
}

RE_HDR = re.compile(r"^####\s+(\S+)\s+(\d+)GPU\s+\((\d+)N\)\s+halo=(\S+)\s+rep=(\d+)"
                    r"(?:\s+dt_prod=(\d+))?")
RE_FIN = re.compile(r"\[bench-finite\]\s+(\S+)\s+npes=(\d+).*?nonfinite_T=(\d+)\s+"
                    r"nonfinite_uv=(\d+)\s+max_uv=([\d.eE+-]+)")
RE_RUN = re.compile(r"\[bench\]\s+mesh=(\S+)\s+nod2D=(\d+)\s+nl=(\d+)\s+npes=(\d+)\s+"
                    r"halo=(\S+)\s+model=(\S+)\s+steps=(\d+)\s+per_step=\s*([\d.]+)\s*ms"
                    r".*?peak_gpu=([\d.]+)")


def parse(logdir: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(logdir.glob("*.out")):
        fin = None                      # the most recent [bench-finite]
        dtp = None                      # the most recent #### dt_prod
        for line in f.read_text(errors="replace").splitlines():
            m = RE_HDR.search(line)
            if m and m.group(6):
                dtp = float(m.group(6)); continue
            m = RE_FIN.search(line)
            if m:
                fin = dict(name=m.group(1), npes=int(m.group(2)),
                           bad_T=int(m.group(3)), bad_uv=int(m.group(4)),
                           max_uv=float(m.group(5)))
                continue
            m = RE_RUN.search(line)
            if not m:
                continue
            name, npes, halo = m.group(1), int(m.group(4)), m.group(5)
            # J3: pair with the [bench-finite] that immediately preceded this row.
            ok = fin is not None and fin["name"] == name and fin["npes"] == npes
            rows.append(dict(
                mesh=name, nod2D=int(m.group(2)), nl=int(m.group(3)), npes=npes, halo=halo,
                model=m.group(6), steps=int(m.group(7)), ms=float(m.group(8)),
                peak_gb=float(m.group(9)), log=f.name,
                finite=bool(ok and fin["bad_T"] == 0 and fin["bad_uv"] == 0),
                has_finite=bool(ok), max_uv=fin["max_uv"] if ok else float("nan"),
                dt_prod=dtp or DT_PROD.get(name, 1800.0)))
            fin = None                  # consume it — never reuse for a second row
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", nargs="?", default=str(ROOT / "scripts" / "logs" / "jupiter"))
    ap.add_argument("--md", action="store_true", help="markdown tables")
    ap.add_argument("--all", action="store_true", help="list every rep, not just the best")
    args = ap.parse_args()

    rows = parse(Path(args.logdir))
    if not rows:
        print(f"no [bench] rows under {args.logdir}"); return 1

    dropped = [r for r in rows if not r["finite"]]
    good = [r for r in rows if r["finite"]]
    print(f"# FESOM2-JAX strong scaling on JUPITER (GH200)\n")
    print(f"{len(rows)} timed rows, {len(good)} bench-finite CLEAN, {len(dropped)} DROPPED\n")
    for r in dropped:
        why = "no [bench-finite]" if not r["has_finite"] else f"nonfinite (max_uv={r['max_uv']})"
        print(f"  ! DROPPED {r['mesh']} npes={r['npes']} halo={r['halo']}: {why}  [{r['log']}]")
    if dropped:
        print()

    if args.all:
        print(f"{'mesh':8s} {'npes':>5s} {'halo':>9s} {'ms/step':>9s} {'SYPD':>8s} "
              f"{'peakGB':>7s}  log")
        for r in sorted(good, key=lambda r: (r["mesh"], r["npes"], r["halo"], r["ms"])):
            sypd = r["dt_prod"] / (r["ms"] / 1e3 * 365.25)
            print(f"{r['mesh']:8s} {r['npes']:5d} {r['halo']:>9s} {r['ms']:9.2f} "
                  f"{sypd:8.2f} {r['peak_gb']:7.2f}  {r['log']}")
        print()

    # Best (min ms) per (mesh, MODEL, npes) over transports AND reps.  The model string is
    # part of the key on purpose: a legacy `FULL(ice+kpp+gm+JRA1958)` calibration row and a
    # production `FULL(mevp+tke+gm+zstar+JRA1958)` row are DIFFERENT MODELS and must never
    # min() against each other (fig10prod/README.md: "reducer/figure selection must switch
    # from the ice+kpp+gm filter to these").  Mixing them silently reports a physics change
    # as a speedup.
    best: dict[tuple[str, str, int], dict] = {}
    per_halo: dict[tuple[str, str, int, str], float] = {}
    for r in good:
        k = (r["mesh"], r["model"], r["npes"])
        if k not in best or r["ms"] < best[k]["ms"]:
            best[k] = r
        kh = (r["mesh"], r["model"], r["npes"], r["halo"])
        if kh not in per_halo or r["ms"] < per_halo[kh]:
            per_halo[kh] = r["ms"]

    by_mesh: dict[tuple[str, str], list[int]] = defaultdict(list)
    for (mesh, model, npes) in best:
        by_mesh[(mesh, model)].append(npes)

    for (mesh, model) in sorted(by_mesh, key=lambda k: (best[(k[0], k[1], min(by_mesh[k]))]["nod2D"],
                                                        k[1])):
        npes_list = sorted(by_mesh[(mesh, model)])
        r0 = best[(mesh, model, npes_list[0])]
        n0, ms0 = npes_list[0], r0["ms"]
        print(f"\n## {mesh}  (nod2D={r0['nod2D']:,} nl={r0['nl']}, "
              f"dt_prod={r0['dt_prod']:g}s, {r0['model']})\n")
        hdr = (f"| GPUs | nodes | best halo | ms/step | SYPD | speedup | par.eff | "
               f"padded | coloured | peak GB | A100 ms | GH200/A100 |")
        sep = "|---:|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        if not args.md:
            hdr = hdr.replace("|", " "); sep = "-" * 96
        print(hdr); print(sep)
        # The A100 reference is the fig10 v2 PRODUCTION campaign — only comparable to the
        # production model string.  A legacy/calibration table gets no ratio column values.
        is_prod = "mevp" in model and "zstar" in model
        for np_ in npes_list:
            r = best[(mesh, model, np_)]
            sypd = r["dt_prod"] / (r["ms"] / 1e3 * 365.25)
            speed = ms0 / r["ms"]
            eff = speed / (np_ / n0) * 100.0
            pad = per_halo.get((mesh, model, np_, "padded"))
            col = per_halo.get((mesh, model, np_, "coloured"))
            f = lambda v: f"{v:.1f}" if v is not None else "—"
            nodes = max(1, -(-np_ // 4))
            a100 = A100_MS.get((mesh, np_)) if is_prod else None
            ratio = f"{a100 / r['ms']:.2f}x" if a100 else "—"
            line = (f"| {np_} | {nodes} | {r['halo']} | {r['ms']:.1f} | {sypd:.2f} | "
                    f"{speed:.2f}x | {eff:.0f}% | {f(pad)} | {f(col)} | {r['peak_gb']:.1f} | "
                    f"{f(a100)} | {ratio} |")
            if not args.md:
                line = line.replace("|", " ")
            print(line)
        print(f"\n(speedup + parallel efficiency are relative to the {n0}-GPU rung)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
