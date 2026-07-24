#!/usr/bin/env python
"""Reduce the ng5-128 remeasure (remeasure_ng5_128.sbatch) to its verdict table.

    python scripts/bench/jupiter/reduce_ng5_128.py [logfile]

Pairs each `#### ... [label] ####` header with the `[bench]` row that follows it, and prints
the legs grouped so the discriminators are readable at a glance:

  * anchors pre/post  -> is this allocation comparable to the campaign, and did it drift?
  * p128 coloured     -> does the anomaly reproduce on FRESH nodes?
  * p128 padded/ragged-> is the anomaly P-SPECIFIC (all transports slow) or COLOURED-SPECIFIC?

`per_step` is the measurement. `compile` is reported in a separate column and clearly marked
DIAGNOSTIC: it is excluded from per_step (the bench times the 2nd warm call on an
already-compiled executable), and is shown only because it is host-side work that a slow
fabric cannot inflate — so if it moves WITH per_step, the cause is the graph, not the network.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Campaign reference (job 1030306), ng5 coloured, for the same-mesh comparison.
CAMPAIGN = {(64, "coloured"): [243.41, 261.10], (128, "coloured"): [644.04, 638.52],
            (256, "coloured"): [257.20, 260.38], (32, "coloured"): [319.35, 316.83]}
CAMPAIGN_COMPILE = {32: "75-87", 64: "76", 128: "155-166", 256: "69-75"}

RE_HDR = re.compile(r"^####\s+ng5\s+(\d+)GPU\s+\((\d+)N\)\s+halo=(\S+).*?\[([^\]]+)\]")
RE_RUN = re.compile(r"\[bench\]\s+mesh=ng5\s+.*?npes=(\d+)\s+halo=(\S+)\s+model=(\S+)\s+"
                    r"steps=(\d+)\s+per_step=\s*([\d.]+)\s*ms.*?compile=\s*([\d.]+)s")
RE_FIN = re.compile(r"\[bench-finite\]\s+ng5\s+npes=(\d+).*?nonfinite_T=(\d+)\s+nonfinite_uv=(\d+)")
RE_FAIL = re.compile(r"####\s*\[([^\]]+)\]\s*rc=(\d+)")


def main() -> int:
    if len(sys.argv) > 1:
        logs = [Path(sys.argv[1])]
    else:
        logs = sorted((ROOT / "scripts" / "logs" / "jupiter").glob("ng5128_*.out"))
    if not logs:
        print("no ng5128_*.out found"); return 1

    legs, label, fin_ok = [], None, None
    for f in logs:
        for line in f.read_text(errors="replace").splitlines():
            m = RE_HDR.search(line)
            if m:
                label = m.group(4); fin_ok = None; continue
            m = RE_FIN.search(line)
            if m:
                fin_ok = (m.group(2) == "0" and m.group(3) == "0"); continue
            m = RE_FAIL.search(line)
            if m:
                legs.append(dict(label=m.group(1), npes=None, halo=None, ms=None,
                                 compile=None, finite=None, rc=int(m.group(2)))); continue
            m = RE_RUN.search(line)
            if m:
                legs.append(dict(label=label or "?", npes=int(m.group(1)), halo=m.group(2),
                                 ms=float(m.group(5)), compile=float(m.group(6)),
                                 finite=fin_ok, rc=0))
                label = None; fin_ok = None

    if not legs:
        print(f"no completed legs yet in {logs[-1].name}"); return 0

    print(f"# ng5-128 REMEASURE — {logs[-1].name}\n")
    print("per_step = THE MEASUREMENT (compile excluded from it).")
    print("compile  = DIAGNOSTIC ONLY (host-side; a slow fabric cannot inflate it).\n")
    print(f"{'leg':22s} {'GPUs':>5s} {'halo':>9s} {'per_step ms':>12s} "
          f"{'campaign ms':>22s} {'compile s':>10s} {'campaign':>9s} {'finite':>7s}")
    print("-" * 104)
    for L in legs:
        if L["rc"]:
            print(f"{L['label']:22s} {'':>5s} {'':>9s} {'FAILED rc=' + str(L['rc']):>12s}")
            continue
        ref = CAMPAIGN.get((L["npes"], L["halo"]))
        refs = "/".join(f"{v:.1f}" for v in ref) if ref else "—"
        cref = CAMPAIGN_COMPILE.get(L["npes"], "—")
        fin = "CLEAN" if L["finite"] else ("BAD" if L["finite"] is False else "?")
        print(f"{L['label']:22s} {L['npes']:5d} {L['halo']:>9s} {L['ms']:12.1f} "
              f"{refs:>22s} {L['compile']:10.1f} {cref:>9s} {fin:>7s}")

    # ---- verdict ----
    got = lambda pred: [L for L in legs if L["rc"] == 0 and pred(L)]
    p128c = got(lambda L: L["npes"] == 128 and L["halo"] == "coloured")
    p128o = got(lambda L: L["npes"] == 128 and L["halo"] != "coloured")
    anch = got(lambda L: L["npes"] in (64, 256))
    print("\n## verdict")
    if anch:
        drift = [f"{L['label']}={L['ms']:.1f}" for L in anch]
        print(f"  anchors: {', '.join(drift)}")
        print("           (campaign: 64 -> 243.4/261.1, 256 -> 257.2/260.4)")
    if p128c:
        v = [L["ms"] for L in p128c]
        lo, hi = min(v), max(v)
        print(f"  P=128 coloured: {', '.join(f'{x:.1f}' for x in v)}  "
              f"(spread {100*(hi-lo)/lo:.0f}%)")
        if hi < 400:
            print("  => the 638-644 ms campaign value did NOT reproduce on fresh nodes:")
            print("     it was an ALLOCATION/ENVIRONMENT artefact, and the ng5 curve should")
            print("     be corrected with these numbers.")
        elif lo > 500:
            print("  => REPRODUCED on fresh nodes: the anomaly is a real property of this")
            print("     (mesh, P) configuration, not the original allocation.")
        else:
            print("  => INCONCLUSIVE / bimodal — the point is unstable; more reps needed.")
    if p128c and p128o:
        cm = min(L["ms"] for L in p128c); om = min(L["ms"] for L in p128o)
        print(f"  P=128 other transports: min {om:.1f} ms vs coloured min {cm:.1f} ms")
        if om > 500 and cm > 500:
            print("     -> ALL transports slow at P=128 => P-SPECIFIC (graph/lowering), not")
            print("        a transport bug. Compile column should corroborate.")
        elif om < 400 < cm:
            print("     -> only coloured is slow at P=128 => COLOURED-SPECIFIC at this shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
