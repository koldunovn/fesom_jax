#!/usr/bin/env python
"""Phase 9a JZ.8 (GATE 9a) — the discriminating year-1 zstar climate comparison.

The C zstar port is validated vs Fortran-zstar at annual-mean surface RMS **0.0038/0.0014 °C/psu
(SST/SSS, 1958)** with a **3–9× contrast to linfs** — that contrast is what "resolves the
coordinate". The JAX gate (K.9 style): **JAX-zstar ↔ C-zstar** must be ``≪`` that zstar↔linfs
contrast (ideally ~ the C↔Fortran climate-close level), proving the JAX port reproduces the C's
zstar climate rather than drifting toward linfs.

Computes the annual-mean (mean of the 12 monthly means, per node) surface SST/SSS for four runs
and the wet-node RMS of their differences:

  * **A  = RMS(JAX-zstar, C-zstar)**        — the JAX port fidelity (should be small).
  * **A′ = RMS(JAX-zstar, Fortran-zstar)**  — JAX vs the Fortran ground truth (~ C↔Fortran level).
  * **B  = RMS(Fortran-zstar, Fortran-linfs)** — the coordinate contrast (the 3–9× signal).
  * **C₀ = RMS(C-zstar, Fortran-zstar)**    — the C port's own fidelity (the 0.0038/0.0014 ref).

Gate: ``A ≪ B`` (the port error is far below the coordinate contrast) — i.e. ``B / A ≫ 1``.

Usage:  python scripts/archive/core2_zstar_climate_compare.py [--year 1958]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import netCDF4

ROOT = Path(__file__).resolve().parents[2]
ZORACLE = Path("/work/ab0995/a270088/port/zstar")
JAX_DIR = ROOT / "data" / "zstar_climate_1yr"


def annual_surface(path, var):
    """Annual-mean surface field (mean over the 12 monthly records), ``(nod2,)``. The C/Fortran
    ``sst``/``sss`` are already surface (time, nod2); average over time."""
    with netCDF4.Dataset(path) as d:
        a = np.asarray(d.variables[var][:], dtype=np.float64)        # (12, nod2)
    a = np.ma.filled(a, np.nan) if np.ma.isMaskedArray(a) else a
    return np.nanmean(a, axis=0)                                     # (nod2,)


def rms(a, b, mask):
    d = (a - b)[mask]
    return float(np.sqrt(np.mean(d * d)))


def bias(a, b, mask):
    return float(np.mean((a - b)[mask]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--jax-dir", type=str, default=str(JAX_DIR),
                    help="JAX-zstar monthly output dir (e.g. data/zstar_climate_1yr_dist864)")
    args = ap.parse_args()
    y = args.year
    jax_dir = Path(args.jax_dir)

    runs = {
        "jax_zstar":     jax_dir / f"{{v}}.fesom.{y}.monthly.nc",
        "c_zstar":       ZORACLE / "c_zstar_2yr" / f"{{v}}.fesom.{y}.monthly.nc",
        "fortran_zstar": ZORACLE / "fortran_zstar_2yr" / f"{{v}}.fesom.{y}.nc",
        "fortran_linfs": ZORACLE / "fortran_linfs_2yr_b" / f"{{v}}.fesom.{y}.nc",
    }
    for name, p in runs.items():
        f = Path(str(p).format(v="sst"))
        if not f.is_file():
            raise SystemExit(f"MISSING {name}: {f}  (run the climate job first?)")

    print(f"=== JZ.8 zstar climate comparison (annual mean, year {y}) ===")
    out = {}
    for var in ("sst", "sss"):
        fields = {n: annual_surface(Path(str(p).format(v=var)), var) for n, p in runs.items()}
        # common wet mask: finite in ALL four annual fields.
        mask = np.ones_like(fields["jax_zstar"], dtype=bool)
        for a in fields.values():
            mask &= np.isfinite(a)
        n = int(mask.sum())
        A = rms(fields["jax_zstar"], fields["c_zstar"], mask)
        Ap = rms(fields["jax_zstar"], fields["fortran_zstar"], mask)
        B = rms(fields["fortran_zstar"], fields["fortran_linfs"], mask)
        C0 = rms(fields["c_zstar"], fields["fortran_zstar"], mask)
        biasA = bias(fields["jax_zstar"], fields["c_zstar"], mask)
        u = "°C" if var == "sst" else "psu"
        print(f"\n[{var}] wet nodes={n}")
        print(f"  A  = RMS(JAX-zstar , C-zstar)       = {A:.4e} {u}   bias={biasA:+.3e}")
        print(f"  A′ = RMS(JAX-zstar , Fortran-zstar) = {Ap:.4e} {u}")
        print(f"  C₀ = RMS(C-zstar  , Fortran-zstar)  = {C0:.4e} {u}   (the C↔Fortran ref ~0.0038/0.0014)")
        print(f"  B  = RMS(Fortran-zstar, Fortran-linfs) = {B:.4e} {u}   (the coordinate contrast)")
        print(f"  ⇒ contrast ratio B/A = {B/max(A,1e-30):.1f}   B/A′ = {B/max(Ap,1e-30):.1f}")
        out[var] = dict(A=A, Ap=Ap, B=B, C0=C0)

    # Gate: the JAX port error (A) must be ≪ the coordinate contrast (B) for BOTH SST/SSS — at
    # least the 3× lower bound of the C's measured 3–9× contrast (B/A > 3 ⇒ the JAX reproduces the
    # C zstar climate, not a linfs-ward drift). Calibrated from the prints (house style).
    ok = all(out[v]["A"] < out[v]["B"] / 3.0 for v in ("sst", "sss"))
    print(f"\n{'PASS' if ok else 'FAIL'}: JAX-zstar↔C-zstar ≪ the zstar↔linfs contrast "
          f"(B/A = {out['sst']['B']/max(out['sst']['A'],1e-30):.1f}× SST, "
          f"{out['sss']['B']/max(out['sss']['A'],1e-30):.1f}× SSS) — "
          f"{'the JAX port reproduces the C zstar climate ✓' if ok else 'port error too close to the contrast ✗'}")
    print("ZSTAR_CLIMATE_COMPARE_" + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
