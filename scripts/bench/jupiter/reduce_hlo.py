"""Reduce XLA HLO dump dirs (hlo_dump_ng5.sbatch) to comparable structure tables.

For each dump dir: pick the LARGEST ``*.after_optimizations.txt`` (the jit_body step
module), count instructions by opcode — total, and separately inside every ``while``
body computation (the CG loops live there; the ng5 x P=128 cliff is zstar-CG-shaped).
Also count fusion kinds and async collective pairs, and report the largest few
computations by instruction count.

    python scripts/bench/jupiter/reduce_hlo.py $ESTA/hlo_dumps/<jobid>/*  [--full]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# computation header: `%name (args) -> type {` or `ENTRY %name (args) -> type {`
_COMP = re.compile(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s*\(.*\)\s*->\s*.*\{\s*$")
_ASSIGN = re.compile(r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*(.*)$")
_OPCODE = re.compile(r"\s*([\w\-]+)\(")
_FUSKIND = re.compile(r"kind=k?(\w+)")


def _opcode_of(rhs: str) -> str | None:
    """Opcode from an instruction RHS. The type may be a tuple with spaces:
    `(f64[3]{0}, s32[]) tuple(...)` — skip a leading paren-balanced type first."""
    rhs = rhs.lstrip()
    if rhs.startswith("("):
        depth = 0
        for i, ch in enumerate(rhs):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    rhs = rhs[i + 1:]
                    break
        else:
            return None
    else:                                   # plain type: `f64[59637]{0} fusion(...)`
        parts = rhs.split(None, 1)
        if len(parts) < 2:
            return None
        rhs = parts[1]
    m = _OPCODE.match(rhs)
    return m.group(1) if m else None


_WHILEREF = re.compile(r"(?:condition|body)=%?([\w.\-]+)")
_RESNAME = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=")
_OPERAND = re.compile(r"\(%?([\w.\-]+)")


def parse(path: Path):
    """Return (opcode Counter total, {computation: Counter}, fusion-kind Counter,
    set of while body/cond computation names, overlap stats).

    Overlap stats: for every async collective pair (`X-start` ... `X-done`), the number
    of scheduled instructions between them in the SAME computation — the compile-time
    overlap window the latency-hiding scheduler produced.  A serialized schedule has
    distance ~1 (done immediately follows start); a hidden collective has a large one."""
    total, per_comp, fus, wnames = Counter(), {}, Counter(), set()
    overlap = {}                # opcode-base -> list of distances
    inflight = {}               # comp -> max concurrent open async collectives
    comp, pos, start_pos, live = None, 0, {}, 0
    with open(path, errors="replace") as f:
        for line in f:
            m = _COMP.match(line)
            if m:
                comp = m.group(1)
                per_comp.setdefault(comp, Counter())
                pos, start_pos, live = 0, {}, 0
                continue
            m = _ASSIGN.match(line)
            if m:
                op = _opcode_of(m.group(1))
                if not op:
                    continue
                pos += 1
                total[op] += 1
                if comp:
                    per_comp[comp][op] += 1
                if op == "fusion":
                    k = _FUSKIND.search(line)
                    fus[k.group(1) if k else "?"] += 1
                elif op == "while":
                    wnames.update(_WHILEREF.findall(line))
                elif op.endswith("-start"):
                    nm = _RESNAME.match(line)
                    if nm:
                        start_pos[nm.group(1)] = (op[:-6], pos)
                        live += 1
                        if comp:
                            inflight[comp] = max(inflight.get(comp, 0), live)
                elif op.endswith("-done"):
                    om = _OPERAND.search(line.split("=", 1)[1])
                    if om and om.group(1) in start_pos:
                        base, p0 = start_pos.pop(om.group(1))
                        overlap.setdefault(base, []).append(pos - p0)
                        live -= 1
    return total, per_comp, fus, wnames, overlap, inflight


def biggest_after_opt(d: Path) -> Path | None:
    cands = sorted(d.glob("*after_optimizations.txt"), key=lambda p: p.stat().st_size)
    return cands[-1] if cands else None


KEY_OPS = ["fusion", "collective-permute-start", "collective-permute", "all-reduce-start",
           "all-reduce", "all-gather-start", "all-gather", "all-to-all", "while",
           "copy", "copy-start", "transpose", "bitcast", "pad", "reshape", "dynamic-slice",
           "dynamic-update-slice", "scatter", "gather", "reduce", "custom-call",
           "conditional", "select", "iota", "convert", "broadcast", "compare", "constant"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--full", action="store_true", help="print every opcode, not just KEY_OPS")
    args = ap.parse_args()

    results = {}
    for dd in args.dirs:
        d = Path(dd)
        f = biggest_after_opt(d)
        if f is None:
            print(f"[reduce_hlo] {d.name}: NO after_optimizations dump", file=sys.stderr)
            continue
        total, per_comp, fus, wnames, overlap, inflight = parse(f)
        while_bodies = {c: cnt for c, cnt in per_comp.items() if c in wnames}
        results[d.name] = dict(file=f.name, size=f.stat().st_size, total=total,
                               per_comp=per_comp, fus=fus, wb=while_bodies,
                               overlap=overlap, inflight=inflight)

    names = list(results)
    print(f"{'':34s} " + " ".join(f"{n:>14s}" for n in names))
    print(f"{'dump file size (MB)':34s} " + " ".join(
        f"{results[n]['size']/1e6:14.1f}" for n in names))
    print(f"{'TOTAL instructions':34s} " + " ".join(
        f"{sum(results[n]['total'].values()):14d}" for n in names))
    print(f"{'computations':34s} " + " ".join(
        f"{len(results[n]['per_comp']):14d}" for n in names))
    allops = sorted(set().union(*(r["total"] for r in results.values())))
    ops = allops if args.full else [o for o in KEY_OPS if any(
        results[n]["total"][o] for n in names)]
    for op in ops:
        print(f"{op:34s} " + " ".join(f"{results[n]['total'][op]:14d}" for n in names))
    for kind in sorted(set().union(*(r["fus"] for r in results.values()))):
        print(f"{'fusion kind=' + kind:34s} " + " ".join(
            f"{results[n]['fus'][kind]:14d}" for n in names))
    print(f"{'instrs in while-ish computations':34s} " + " ".join(
        f"{sum(sum(c.values()) for c in results[n]['wb'].values()):14d}" for n in names))
    for base in sorted(set().union(*(r["overlap"] for r in results.values()))):
        def fmt(n):
            ds = results[n]["overlap"].get(base, [])
            if not ds:
                return f"{'-':>14s}"
            ds = sorted(ds)
            return f"{ds[len(ds)//2]:5d}/{sum(ds)/len(ds):7.1f}"
        print(f"{'overlap med/mean: ' + base:34s} " + " ".join(fmt(n) for n in names))
    print()
    for n in names:
        top = sorted(results[n]["per_comp"].items(), key=lambda kv: -sum(kv[1].values()))[:8]
        print(f"-- {n}: top computations by instruction count")
        for c, cnt in top:
            keys = ", ".join(f"{o}:{v}" for o, v in cnt.most_common(6))
            print(f"   {sum(cnt.values()):7d}  {c[:80]:80s} {keys}")
        print(f"-- {n}: while bodies (instrs / cp-start / ar-start / ag-start / max-inflight)")
        for c, cnt in sorted(results[n]["wb"].items(),
                             key=lambda kv: -sum(kv[1].values())):
            tot = sum(cnt.values())
            if tot < 30:
                continue
            print(f"   {tot:7d}  cp:{cnt['collective-permute-start']:4d} "
                  f"ar:{cnt['all-reduce-start']:3d} ag:{cnt['all-gather-start']:3d} "
                  f"dus:{cnt['dynamic-update-slice']:3d} "
                  f"maxinflight:{results[n]['inflight'].get(c, 0):3d}  {c[:60]}")


if __name__ == "__main__":
    main()
