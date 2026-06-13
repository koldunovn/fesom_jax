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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(ROOT / "scripts" / "mevp_liveness_fields.npz"))
    ap.add_argument("--month", type=int, default=0)      # January 1958 = the early period
    args = ap.parse_args()

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
