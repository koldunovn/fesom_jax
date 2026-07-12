"""Compare the JAX-TKE 1-yr climate to the C ``c_tke_2yr`` oracle — Phase 9b, JT.5.

THE arbiter of whether the step-1 forcing-gap (the 7e-4 stress diff at ~10% low-wind nodes)
matters at climate scale or washes out as a transient. Computes the annual-mean (mean of the
12 monthly means, per node) surface SST/SSS for:

  A  = RMS(jax_tke − c_tke)        — the JAX-vs-C-port error (the headline)
  C0 = RMS(c_tke − fortran_tke)    — the C-port-vs-Fortran reference (the achievable floor;
                                     the C plan measured SST/SSS ~0.0049/0.0028 °C/psu yr 1)

Gate (the K.8 / zstar discriminating style): if A ≈ C0 (a few×), the JAX TKE climate is as
faithful as the C is to Fortran ⇒ the forcing-gap is a step-1 transient, TKE is climate-validated.
If A ≫ C0 (≳ the TKE↔KPP scheme contrast, the stability run measured 0.43 °C over 2.78 days),
the forcing/bulk difference is real and propagates.

Usage:  python scripts/archive/core2_tke_climate_compare.py [--year 1958] [--jax-dir data/tke_climate_1yr_dist864]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
ORACLE = Path("/work/ab0995/a270088/port/tke")


def annual_surface(path, var):
    """Annual-mean surface field ``(nod2,)`` — mean over the 12 monthly records. The 2-D
    ``sst``/``sss`` are ``(time, nod2)``; land is NaN/0 in the C output."""
    with Dataset(path) as d:
        a = np.asarray(d.variables[var][:])           # (time, nod2)
    return np.nanmean(a, axis=0)


def rms(a, b, mask):
    d = (a - b)[mask]
    return float(np.sqrt(np.mean(d * d)))


def bias(a, b, mask):
    return float(np.mean((a - b)[mask]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--jax-dir", type=str, default=str(ROOT / "data" / "tke_climate_1yr_dist864"))
    args = ap.parse_args()
    y = args.year
    jax_dir = Path(args.jax_dir)

    srcs = {
        "jax_tke":     jax_dir / f"{{v}}.fesom.{y}.monthly.nc",
        "c_tke":       ORACLE / "c_tke_2yr" / f"{{v}}.fesom.{y}.monthly.nc",
        "fortran_tke": ORACLE / "fortran_linfs_tke" / f"{{v}}.fesom.{y}.monthly.nc",
    }
    out = {}
    for vshort, vlong in (("sst", "sst"), ("sss", "sss")):
        fields = {}
        for name, tmpl in srcs.items():
            p = Path(str(tmpl).format(v=vshort))
            fields[name] = annual_surface(p, vlong) if p.is_file() else None
        if fields["jax_tke"] is None or fields["c_tke"] is None:
            print(f"[{vshort}] MISSING jax_tke or c_tke ({srcs['jax_tke']}); skip", flush=True)
            continue
        # wet mask: finite + non-zero in both jax and C (land is 0/NaN)
        jt, ct = fields["jax_tke"], fields["c_tke"]
        mask = np.isfinite(jt) & np.isfinite(ct) & (np.abs(ct) > 1e-6)
        A = rms(jt, ct, mask)
        bA = bias(jt, ct, mask)
        row = {"A": A, "biasA": bA, "n": int(mask.sum())}
        if fields["fortran_tke"] is not None:
            ft = fields["fortran_tke"]
            m2 = mask & np.isfinite(ft)
            row["C0"] = rms(ct, ft, m2)
        out[vshort] = row
        c0s = f"  C0(C↔Fortran)={row['C0']:.4e}  A/C0={A/max(row.get('C0',np.nan),1e-30):.1f}×" \
              if "C0" in row else ""
        print(f"[{vshort}] A(JAX↔C-TKE)={A:.4e}  bias={bA:+.4e}  (n={row['n']}){c0s}", flush=True)

    # verdict: A within a few× of C0 (or ≪ the 0.43 °C TKE↔KPP contrast) ⇒ forcing-gap is a transient
    ok = True
    for v, r in out.items():
        floor = r.get("C0", 0.0)
        thresh = max(10 * floor, 0.05)          # a few× the C-Fortran floor, or 0.05 °C/psu
        passed = r["A"] < thresh
        ok = ok and passed
        print(f"  [{v}] A={r['A']:.4e} {'<' if passed else '≥'} thresh={thresh:.4e}  "
              f"{'OK' if passed else 'DIVERGES'}", flush=True)
    print("TKE_CLIMATE_OK" if ok else "TKE_CLIMATE_DIVERGES", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
