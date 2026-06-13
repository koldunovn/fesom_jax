#!/usr/bin/env python
"""mEVP diff-of-diffs LIVENESS + ice-metric sanity — Phase 9c, JM.3.

Proves the mEVP option is **live and faithful** (not a silently-dead knob — absolute agreement
alone can't catch that), using the C plan's diff-of-diffs methodology:

    (JAX-mEVP − JAX-EVP)  must spatially pattern-correlate with  (C-mEVP − C-EVP)

on the scalar surface fields (SST, a_ice, m_ice — frame-independent, so no vector rotation
needed; the rotation-class concern is already ruled out by the bit-faithful NATIVE-frame
per-iterate velocity dump gates, JM.2). The JAX fields are a day-``N`` snapshot
(``mevp_liveness_fields.npz`` from ``core2_mevp_stability.py``); the C legs are the January-1958
monthly mean of ``c_mevp_2yr`` / ``c_evp_2yr`` (the comparable early period). A clearly POSITIVE
correlation ⇒ mEVP perturbs the ice the same way the C's mEVP does.

Also reports ice EXTENT (Σ area where a_ice>0.15) and VOLUME (Σ m_ice·area) for all four legs —
the JAX-mEVP metrics should sit in the C-mEVP ballpark (a day-N snapshot vs a monthly mean differ
by sampling, so "sane", not the C's ≤0.3%/0.7% bit-gate).

Usage:  python scripts/core2_mevp_climate_compare.py [--npz scripts/mevp_liveness_fields.npz]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "data" / "mesh_core2"
ORACLE = Path("/work/ab0995/a270088/port/mevp")


def load_c(oracle, var, month=0, year=1958):
    import netCDF4 as nc
    f = nc.Dataset(oracle / f"{var}.fesom.{year}.monthly.nc")
    a = np.asarray(f.variables[var][month])              # (nod2D,) — global gid order = JAX index
    f.close()
    return a


def pattern_corr(djax, dc, mask):
    """Centered spatial correlation of two difference fields over ``mask``."""
    x = djax[mask].astype(np.float64); y = dc[mask].astype(np.float64)
    x = x - x.mean(); y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom > 0 else 0.0


def ice_metrics(aice, mice, area, wet):
    ext = float(np.sum(area * wet * (aice > 0.15)))      # m² ice extent (15% threshold)
    vol = float(np.sum(area * wet * mice))               # m³ ice volume
    return ext, vol


def _annual_surface(path, var):
    """Annual-mean surface field (mean over the 12 monthly records), ``(nod2,)``."""
    import netCDF4
    with netCDF4.Dataset(path) as d:
        a = np.asarray(d.variables[var][:], dtype=np.float64)        # (12, nod2)
    a = np.ma.filled(a, np.nan) if np.ma.isMaskedArray(a) else a
    return np.nanmean(a, axis=0)


def annual_compare(jax_dir, year):
    """Year-scale C comparison (the treatment zstar/TKE got, which mEVP lacked): the annual-mean
    surface SST/SSS RMS, 4-way (mirrors `core2_zstar_climate_compare.py`):
      A  = RMS(JAX-mEVP , C-mEVP)        — the JAX port fidelity (should be small).
      A′ = RMS(JAX-mEVP , Fortran-mEVP)  — JAX vs the Fortran ground truth.
      C₀ = RMS(C-mEVP   , Fortran-mEVP)  — the C port's own fidelity (the climate-close floor).
      B  = RMS(C-mEVP   , C-EVP)         — the mEVP↔EVP RHEOLOGY contrast (the signal).
    Gate: A is climate-close — A ≲ max(B, 3·C₀), i.e. the port adds no more than the rheology
    signal / a few × the C↔Fortran floor (a rheology's SST/SSS contrast B is modest, unlike a
    coordinate change — so A≈C₀ is the binding statement, like TKE's SST 4.68e-3 ≈ the floor)."""
    y = year
    jd = Path(jax_dir)
    runs = {
        "jax_mevp":     jd / f"{{v}}.fesom.{y}.monthly.nc",
        "c_mevp":       ORACLE / "c_mevp_2yr" / f"{{v}}.fesom.{y}.monthly.nc",
        "c_evp":        ORACLE / "c_evp_2yr" / f"{{v}}.fesom.{y}.monthly.nc",
        "fortran_mevp": ORACLE / "fortran_mevp_2yr" / f"{{v}}.fesom.{y}.nc",
    }
    for name, p in runs.items():
        f = Path(str(p).format(v="sst"))
        if not f.is_file():
            raise SystemExit(f"MISSING {name}: {f}  (run the climate job first?)")
    print(f"=== mEVP year-{y} climate comparison (annual mean) ===")
    out = {}
    for var in ("sst", "sss"):
        fld = {n: _annual_surface(Path(str(p).format(v=var)), var) for n, p in runs.items()}
        mask = np.ones_like(fld["jax_mevp"], dtype=bool)
        for a in fld.values():
            mask &= np.isfinite(a)

        def rms(a, b):
            d = (fld[a] - fld[b])[mask]
            return float(np.sqrt(np.mean(d * d)))
        A = rms("jax_mevp", "c_mevp"); Ap = rms("jax_mevp", "fortran_mevp")
        C0 = rms("c_mevp", "fortran_mevp"); B = rms("c_mevp", "c_evp")
        u = "°C" if var == "sst" else "psu"
        print(f"\n[{var}] wet nodes={int(mask.sum())}")
        print(f"  A  = RMS(JAX-mEVP , C-mEVP)        = {A:.4e} {u}")
        print(f"  A′ = RMS(JAX-mEVP , Fortran-mEVP)  = {Ap:.4e} {u}")
        print(f"  C₀ = RMS(C-mEVP   , Fortran-mEVP)  = {C0:.4e} {u}   (the C↔Fortran floor)")
        print(f"  B  = RMS(C-mEVP   , C-EVP)         = {B:.4e} {u}   (the mEVP↔EVP rheology contrast)")
        print(f"  ⇒ A/C₀ = {A/max(C0,1e-30):.2f}   B/A = {B/max(A,1e-30):.1f}")
        out[var] = dict(A=A, Ap=Ap, B=B, C0=C0)
    ok = all(out[v]["A"] < max(out[v]["B"], 3.0 * out[v]["C0"]) for v in ("sst", "sss"))
    print(f"\n{'PASS' if ok else 'FAIL'}: JAX-mEVP reproduces the C-mEVP climate "
          f"(SST A={out['sst']['A']:.3e} vs C₀={out['sst']['C0']:.3e}; "
          f"SSS A={out['sss']['A']:.3e} vs C₀={out['sss']['C0']:.3e})")
    print("MEVP_CLIMATE_COMPARE_" + ("OK" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


def all3_compare(jax_dir, c_dir, year):
    """All-3 (zstar+TKE+mEVP) port fidelity: annual-mean surface SST/SSS RMS of JAX-all-3 vs the
    C-all-3 oracle (`c_dir`). ⚠️ The combination is unvalidated vs Fortran (no all-3 Fortran
    ground truth) — this proves **JAX reproduces the C port for the combination** (port fidelity),
    NOT physical correctness. Reference: the single-option C↔Fortran floor is ~5.3e-3/2.5e-3
    (SST/psu); a climate-close A_all3 in that ballpark ⇒ the JAX all-3 == the C all-3."""
    y = year
    jd, cd = Path(jax_dir), Path(c_dir)
    out = {}
    for var in ("sst", "sss"):
        jf = jd / f"{var}.fesom.{y}.monthly.nc"
        cf = cd / f"{var}.fesom.{y}.monthly.nc"
        for f in (jf, cf):
            if not f.is_file():
                raise SystemExit(f"MISSING {f}  (both JAX-all-3 and C-all-3 runs must finish first)")
        a = _annual_surface(jf, var); b = _annual_surface(cf, var)
        mask = np.isfinite(a) & np.isfinite(b)
        d = (a - b)[mask]
        A = float(np.sqrt(np.mean(d * d))); bi = float(np.mean(d))
        u = "°C" if var == "sst" else "psu"
        print(f"[{var}] wet nodes={int(mask.sum())}  "
              f"A_all3 = RMS(JAX-all3, C-all3) = {A:.4e} {u}  bias={bi:+.3e}  "
              f"(single-option C↔Fortran floor ~{5.29e-3 if var=='sst' else 2.49e-3:.2e})")
        out[var] = A
    # climate-close ⇒ JAX reproduces the C all-3 (allow a few× the single-option floor — the
    # combination compounds three options' reassociation, but should stay well under ~2e-2).
    ok = out["sst"] < 2e-2 and out["sss"] < 1e-2
    print(f"\n{'PASS' if ok else 'FAIL'}: JAX-all-3 {'reproduces' if ok else 'DIVERGES from'} "
          f"the C-all-3 (SST {out['sst']:.3e}, SSS {out['sss']:.3e}) — port fidelity "
          f"(NOT a physical-validation, the combination is unvalidated vs Fortran)")
    print("ALL3_CLIMATE_COMPARE_" + ("OK" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(ROOT / "scripts" / "mevp_liveness_fields.npz"))
    ap.add_argument("--month", type=int, default=0)      # January 1958 = the early period
    ap.add_argument("--annual", action="store_true",
                    help="year-scale 4-way annual-mean RMS vs c_mevp_2yr/fortran_mevp_2yr "
                         "(the C comparison; needs the year-1 climate run first)")
    ap.add_argument("--jax-dir", default=str(ROOT / "data" / "mevp_climate_1yr"))
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--all3", action="store_true",
                    help="all-3 (zstar+TKE+mEVP) JAX-vs-C-oracle port-fidelity RMS")
    ap.add_argument("--c-dir", default="/work/ab0995/a270088/port/mevp/c_all3_1yr")
    args = ap.parse_args()

    if args.all3:                                         # JAX-all-3 ↔ C-all-3 (port fidelity)
        return all3_compare(args.jax_dir, args.c_dir, args.year)
    if args.annual:                                       # the year-scale mEVP C comparison
        return annual_compare(args.jax_dir, args.year)

    from fesom_jax.mesh import load_mesh
    mesh = load_mesh(MESH_DIR)
    area = np.asarray(mesh.areasvol)[:, 0]
    wet = np.asarray(mesh.node_layer_mask)[:, 0].astype(bool)

    j = np.load(args.npz)
    print(f"[jax] day-{float(j['steps'])*float(j['dt'])/86400:.1f} snapshot from {args.npz}")

    # ---- diff-of-diffs liveness (scalars) ----
    print("\n=== diff-of-diffs liveness: corr( JAX(mEVP−EVP), C(mEVP−EVP) ) ===")
    legs = {"sst": ("sst_mevp", "sst_evp", "sst"),
            "a_ice": ("aice_mevp", "aice_evp", "a_ice"),
            "m_ice": ("mice_mevp", "mice_evp", "m_ice")}
    all_live = True
    for name, (km, ke, cvar) in legs.items():
        djax = np.asarray(j[km]) - np.asarray(j[ke])
        dc = load_c(ORACLE / "c_mevp_2yr", cvar, args.month) - load_c(ORACLE / "c_evp_2yr", cvar, args.month)
        # focus on the ice-active / cold domain (where the difference lives)
        active = wet & ((np.abs(djax) > 0) | (np.abs(dc) > 0))
        corr = pattern_corr(djax, dc, active)
        jrng = float(np.abs(djax[wet]).max()); crng = float(np.abs(dc[wet]).max())
        live = corr > 0.3 and jrng > 0
        all_live = all_live and live
        print(f"  {name:6s} corr={corr:+.3f}  |Δ|max JAX={jrng:.3e} C={crng:.3e}  "
              f"{'LIVE ✓' if live else 'weak/✗'}")

    # ---- ice-metric sanity ----
    print("\n=== ice metrics (extent Mkm², volume kkm³) — JAX day-N vs C Jan-mean ===")
    legs_m = [("JAX-mEVP", j["aice_mevp"], j["mice_mevp"]),
              ("JAX-EVP ", j["aice_evp"], j["mice_evp"])]
    for nm, a, m in legs_m:
        ext, vol = ice_metrics(np.asarray(a), np.asarray(m), area, wet)
        print(f"  {nm}: extent={ext/1e12:.3f} Mkm²  volume={vol/1e12:.3f} kkm³")
    for nm, od in [("C-mEVP  ", "c_mevp_2yr"), ("C-EVP   ", "c_evp_2yr")]:
        a = load_c(ORACLE / od, "a_ice", args.month); m = load_c(ORACLE / od, "m_ice", args.month)
        ext, vol = ice_metrics(a, m, area, wet)
        print(f"  {nm}: extent={ext/1e12:.3f} Mkm²  volume={vol/1e12:.3f} kkm³  (Jan-mean)")

    print(f"\nMEVP_LIVENESS_{'OK' if all_live else 'WEAK'}", flush=True)
    return 0 if all_live else 1


if __name__ == "__main__":
    raise SystemExit(main())
