"""Sign + |grad| vs adjoint window — the Task-C1 sign-flip figure + the Part-D/E decision.

Reads the JSONL the sign-flip sweep (:mod:`scripts.core2_lw_signflip`) appended (one line per
(seed, N)) and, per seed, finds:
  * ``N*``     — the smallest window whose ``d(window-mean MLD)/d(c_k)`` has the SLOW/RIGHT (+)
                 sign (D2a's "more mixing deepens MLD"); below it the gradient is fast/wrong (−).
  * ``N_blow`` — the smallest window where the gradient BLOWS UP (``|grad| > BLOW`` or non-finite,
                 the chaotic-adjoint horizon — cf the all3 adjoint |g| 1e4→1e15).

Decision (the make-or-break routing): if **N* < N_blow** for any seed → a clean window reaches the
slow sign ⇒ **simple ensemble-averaging is viable (Part D)** → ``SIGNFLIP_HORIZON_OK``. Else (no
slow-signed window below blow-up) → escalate WITHIN the adjoint family (Part E) — NOT EKI.

The figure (2 panels: signed gradient vs N; |grad| + peak memory vs N, log) degrades cleanly if
matplotlib is absent — the decision + token still print. The blow-up criterion is |grad| magnitude
+ best-h FD sign agreement, NOT plateau<1% (the MLD density-threshold crossing makes the secant FD
noisy across h — the SIGN is the trustworthy quantity).

Usage:  python scripts/fig_signflip.py --results scripts/lw_signflip_results.jsonl --out scripts/fig_signflip.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BLOW = 1.0e2     # |d(MLD)/d(c_k)| beyond this ⇒ the adjoint has blown up


def load(results: Path):
    rows = []
    for line in results.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def is_blown(r) -> bool:
    if r.get("oom"):
        return True
    g = r.get("grad")
    if g is None or not math.isfinite(g):
        return True
    if "blown" in r:                       # the refined-label runs carry it directly
        return bool(r["blown"])
    return abs(g) > BLOW                    # prelim runs: recompute from |grad|


def per_seed(rows):
    seeds = {}
    for r in rows:
        seeds.setdefault(r.get("season", "seed"), []).append(r)
    out = {}
    for s, rs in seeds.items():
        rs = sorted(rs, key=lambda r: r["N"])
        Nstar = next((r["N"] for r in rs if (r.get("grad") is not None) and (not is_blown(r))
                      and (r["grad"] > 0.0)), None)
        Nblow = next((r["N"] for r in rs if is_blown(r)), None)
        out[s] = dict(rows=rs, Nstar=Nstar, Nblow=Nblow,
                      clean_slow=bool(Nstar is not None and (Nblow is None or Nstar < Nblow)))
    return out


def make_figure(seeds, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] matplotlib unavailable ({e}); skipping the plot", flush=True)
        return False
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 7.2), sharex=True)
    colors = ["C0", "C1", "C2", "C3"]
    for i, (s, d) in enumerate(sorted(seeds.items())):
        rs = [r for r in d["rows"] if r.get("grad") is not None]
        Ns = [r["N"] for r in rs]
        g = [r["grad"] for r in rs]
        ag = [abs(r["grad"]) for r in rs]
        c = colors[i % len(colors)]
        a1.plot(Ns, g, "o-", color=c, label=f"{s}  (N*={d['Nstar']}, N_blow={d['Nblow']})")
        a2.semilogy(Ns, ag, "o-", color=c, label=s)
        if d["Nblow"]:
            a1.axvline(d["Nblow"], color=c, ls=":", lw=1, alpha=0.6)
    a1.axhline(0.0, color="0.5", ls="--", lw=1)
    a1.set_ylabel("d(window-mean MLD)/d(c_k)\n(+ = slow/right; − = fast/wrong)")
    a1.set_title("C1 sign-flip: gradient sign + magnitude vs adjoint window\n"
                 "full config (zstar+TKE+mEVP+GM), frozen-ice adjoint")
    a1.legend(loc="best", fontsize=8); a1.grid(True, alpha=0.3)
    a2.axhline(BLOW, color="0.5", ls="--", lw=1, label=f"blow-up |grad|>{BLOW:g}")
    a2.set_ylabel("|gradient|"); a2.set_xlabel("adjoint window N (steps)")
    a2.legend(loc="best", fontsize=8); a2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"  [fig] wrote {out}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("scripts/lw_signflip_results.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("scripts/fig_signflip.png"))
    args = ap.parse_args()
    if not args.results.exists():
        print(f"no results at {args.results}"); return 1
    rows = load(args.results)
    seeds = per_seed(rows)

    print("\n=== C1 sign-flip horizon (per seed) ===")
    for s, d in sorted(seeds.items()):
        print(f"\n  seed={s}:")
        for r in d["rows"]:
            g = r.get("grad")
            tag = "OOM" if r.get("oom") else ("BLOWN" if is_blown(r) else
                                              ("+slow/right" if (g and g > 0) else "−fast/wrong"))
            gtxt = "  none" if g is None else f"{g:+.4e}"
            print(f"    N={r['N']:>4} ({r['days']:.2f} d)  grad={gtxt}  |g|="
                  f"{(abs(g) if g is not None else float('nan')):.3e}  "
                  f"peak={r.get('peak_gb', float('nan')):.1f}GB  → {tag}")
        print(f"    ⇒ N*={d['Nstar']}  N_blow={d['Nblow']}  clean-slow-window={d['clean_slow']}")

    any_clean = any(d["clean_slow"] for d in seeds.values())
    make_figure(seeds, args.out)
    print("\n" + ("SIGNFLIP_HORIZON_OK  (a clean window reaches the slow sign ⇒ Part D viable)"
                  if any_clean else
                  "SIGNFLIP_HORIZON_ESCALATE  (no slow-signed clean window ⇒ Part E, adjoint-only)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
