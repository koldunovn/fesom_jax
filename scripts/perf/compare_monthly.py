#!/usr/bin/env python
"""CORE2 GPU speed plan, Tier B: compare two runs' monthly-mean ushow stores month by month (e.g. dt 2700
against the dt 1800 hindcast over the same years).

    compare_monthly.py <run A monthly dir> <run B monthly dir> --mesh data/mesh_core2 [--months 1990_01 ...]
                       [--json out.json]

Per month and run: volume-weighted mean T and S (static layer thickness from zbar_3d_n, upper-edge areas),
area-weighted surface means (SST, SSS, SSH), NH/SH sea-ice area and volume, the largest surface speed and the
extremes of T and S over wet points; per month the area-weighted rms of A - B for SST, SSS and SSH. Values
that are not finite are counted and reported (a blow-up shows there first).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _read(store, name):
    import zarr
    return np.asarray(zarr.open(os.path.join(store, name), mode="r")[0], dtype=np.float64)


def month_stats(store, w3, wet3, a_surf, nh):
    T, S = _read(store, "temp"), _read(store, "salt")            # [48, nod]
    u, v = _read(store, "u"), _read(store, "v")
    ssh, ai, mi = _read(store, "ssh"), _read(store, "a_ice"), _read(store, "m_ice")
    out = {"nonfinite": int(sum(np.count_nonzero(~np.isfinite(x[wet3] if x.ndim == 2 else x))
                                for x in (T, S, u, v, ssh, ai, mi)))}
    W = w3.sum()
    out["T_mean"] = float((T * w3)[wet3].sum() / W)
    out["S_mean"] = float((S * w3)[wet3].sum() / W)
    A = a_surf.sum()
    out["SST"] = float((T[0] * a_surf).sum() / A)
    out["SSS"] = float((S[0] * a_surf).sum() / A)
    out["SSH_mean"] = float((ssh * a_surf).sum() / A)
    out["SSH_min"], out["SSH_max"] = float(ssh.min()), float(ssh.max())
    for hemi, m in (("NH", nh), ("SH", ~nh)):
        out[f"ice_area_{hemi}"] = float((ai * a_surf)[m].sum() / 1e12)          # 1e12 m2
        out[f"ice_vol_{hemi}"] = float((mi * a_surf)[m].sum() / 1e12)           # 1e12 m3
    out["speed_max_surf"] = float(np.sqrt(u[0] ** 2 + v[0] ** 2).max())
    out["speed_max"] = float(np.sqrt(u ** 2 + v ** 2)[wet3].max())
    out["T_min"], out["T_max"] = float(T[wet3].min()), float(T[wet3].max())
    out["S_min"], out["S_max"] = float(S[wet3].min()), float(S[wet3].max())
    return out, {"SST": T[0], "SSS": S[0], "SSH": ssh}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--mesh", default="data/mesh_core2")
    ap.add_argument("--months", nargs="*")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh(a.mesh)
    area = np.asarray(mesh.area, dtype=np.float64)                          # [nod, nl]
    zb = np.asarray(mesh.zbar_3d_n, dtype=np.float64)                        # [nod, nl] interfaces
    nlev = np.asarray(mesh.nlevels_nod2D)
    nl = area.shape[1]
    k = np.arange(nl)[:, None]
    wet3 = k < (nlev[None, :] - 1)                                           # layers 0 .. nlev-2
    dz = np.zeros((nl, area.shape[0]))
    dz[:-1] = (zb[:, :-1] - zb[:, 1:]).T
    w3 = np.where(wet3, area.T * np.abs(dz), 0.0)
    a_surf = area[:, 0]
    nh = np.asarray(mesh.geo_coord_nod2D)[:, 1] > 0
    months = a.months or sorted(set(os.listdir(a.a)) & set(os.listdir(a.b)))
    rows = []
    for mth in months:
        sa, fa = month_stats(os.path.join(a.a, mth), w3, wet3, a_surf, nh)
        sb, fb = month_stats(os.path.join(a.b, mth), w3, wet3, a_surf, nh)
        rms = {f"rms_{k}": float(np.sqrt(((fa[k] - fb[k]) ** 2 * a_surf).sum() / a_surf.sum())) for k in fa}
        rows.append({"month": mth, "a": sa, "b": sb, **rms})
        print(f"{mth}  T {sa['T_mean']:.5f}/{sb['T_mean']:.5f}  S {sa['S_mean']:.6f}/{sb['S_mean']:.6f}  "
              f"SST {sa['SST']:.3f}/{sb['SST']:.3f}  iceNH {sa['ice_area_NH']:.2f}/{sb['ice_area_NH']:.2f}  "
              f"iceSH {sa['ice_area_SH']:.2f}/{sb['ice_area_SH']:.2f}  SSH [{sa['SSH_min']:.2f},{sa['SSH_max']:.2f}]"
              f"/[{sb['SSH_min']:.2f},{sb['SSH_max']:.2f}]  |u|max {sa['speed_max']:.2f}/{sb['speed_max']:.2f}  "
              f"rms SST {rms['rms_SST']:.3f} SSS {rms['rms_SSS']:.3f} SSH {rms['rms_SSH']:.3f}  "
              f"nonfinite {sa['nonfinite']}/{sb['nonfinite']}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
