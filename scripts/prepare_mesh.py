#!/usr/bin/env python
"""Standalone mesh preparation (Task A8): raw FESOM ASCII → the C-export `.npy` layout.

    python scripts/prepare_mesh.py RAW_MESH_DIR OUT_DIR
    python scripts/prepare_mesh.py RAW_MESH_DIR x --verify C_EXPORT_DIR   # diff vs the C oracle

Reads the raw FESOM mesh files and writes EXACTLY the `.npy` + `meta.txt` layout that `load_mesh`
consumes (`docs/MESH_EXPORT_LAYOUT.md`) — so the JAX model ships **without a C-FESOM build
dependency**. The heavy mesh-setup derivation runs **once, offline** here (NOT online every run).

This is a numpy port of `fesom_mesh.c`'s `fesom_mesh_compute_metrics`, verified array-by-array
against the C exports (`data/mesh_core2`) — the same byte-identical C→JAX porting the project does.

**Byte-identity notes (from `fesom_mesh.c`):** the truncated `PI=3.14159265358979` is used everywhere
EXCEPT `mesh_resolution` (full-precision pi); `orient_cw` runs before all geometry; CSR per-node
element lists are element-ascending and the area/mesh_resolution sums follow that order; two cyclic
conventions (`elem_center` min-anchored `>=/<` vs the simple `>/<` trim elsewhere); `edge_dxdy` is in
radians, `edge_cross_dxdy` in meters (`lon*elem_cos*R`), `gradient_sca` lon-diffs `*elem_cos`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PI = 3.14159265358979                 # FESOM truncated pi (RAD/OMEGA/CYCLIC)
PI_FULL = 3.14159265358979323846      # full-precision pi — ONLY for mesh_resolution
RAD = PI / 180.0
R_EARTH = 6367500.0
OMEGA = 2.0 * PI / 86400.0
CYC = 2.0 * PI                        # cyclic length (radians)
HALF_CYC = 0.5 * CYC
ALPHA, BETA, GAMMA = 50.0, 15.0, -90.0   # Euler angles (deg)


# ==========================================================================
# Raw FESOM ASCII readers
# ==========================================================================
def read_fesom_ascii(mesh_dir) -> dict:
    """Read the primitive FESOM mesh + the prebuilt level/edge files."""
    d = Path(mesh_dir)
    with open(d / "nod2d.out") as f:
        nod2D = int(f.readline().split()[0])
        nod = np.loadtxt(f, max_rows=nod2D)
    nodes_deg = nod[:, 1:3].astype(np.float64)
    coast = nod[:, 3].astype(np.int32)

    with open(d / "elem2d.out") as f:
        elem2D = int(f.readline().split()[0])
        elem = np.loadtxt(f, max_rows=elem2D, dtype=np.int64)
    elem_nodes = (elem[:, :3] - 1).astype(np.int32)            # 1→0-based (raw orientation)

    with open(d / "aux3d.out") as f:
        nl = int(f.readline().split()[0])
        vals = np.loadtxt(f)
    zbar = vals[:nl].astype(np.float64)
    if zbar[1] > 0.0:                                          # negate ALL only if zbar[1] > 0
        zbar = -zbar
    depth = vals[nl: nl + nod2D].astype(np.float64)
    depth = np.where(depth > 0.0, -depth, depth)              # force ≤ 0

    nlevels_nod2D = np.loadtxt(d / "nlvls.out", dtype=np.int64).astype(np.int32)  # already max-over-cells
    nlevels = np.loadtxt(d / "elvls.out", dtype=np.int64).astype(np.int32)
    edges = (np.loadtxt(d / "edges.out", dtype=np.int64) - 1).astype(np.int32)    # 1→0-based pairs
    et = np.loadtxt(d / "edge_tri.out", dtype=np.int64)
    edge_tri = np.where(et > 0, et - 1, -1).astype(np.int32)  # -999/≤0 → -1

    return dict(nodes_deg=nodes_deg, coast=coast, elem_nodes=elem_nodes, nl=nl, zbar=zbar,
                depth=depth, nlevels_nod2D=nlevels_nod2D, nlevels=nlevels, edges=edges,
                edge_tri=edge_tri, nod2D=nod2D, elem2D=elem2D)


# ==========================================================================
# Rotation (g2r / r2g) — Euler angles, truncated pi
# ==========================================================================
def _rot_matrix():
    al, be, ga = ALPHA * RAD, BETA * RAD, GAMMA * RAD
    ca, sa, cb, sb, cg, sg = (np.cos(al), np.sin(al), np.cos(be),
                              np.sin(be), np.cos(ga), np.sin(ga))
    return np.array([
        cg * ca - sg * cb * sa, cg * sa + sg * cb * ca, sg * sb,
        -sg * ca - cg * cb * sa, -sg * sa + cg * cb * ca, cg * sb,
        sb * sa, -sb * ca, cb])


def _g2r(M, glon, glat):
    xg, yg, zg = np.cos(glat) * np.cos(glon), np.cos(glat) * np.sin(glon), np.sin(glat)
    xr = M[0] * xg + M[1] * yg + M[2] * zg
    yr = M[3] * xg + M[4] * yg + M[5] * zg
    zr = M[6] * xg + M[7] * yg + M[8] * zg
    rlat = np.arcsin(zr)
    rlon = np.where((yr == 0.0) & (xr == 0.0), 0.0, np.arctan2(yr, xr))
    return rlon, rlat


def _r2g(M, rlon, rlat):
    xr, yr, zr = np.cos(rlat) * np.cos(rlon), np.cos(rlat) * np.sin(rlon), np.sin(rlat)
    xg = M[0] * xr + M[3] * yr + M[6] * zr
    yg = M[1] * xr + M[4] * yr + M[7] * zr
    zg = M[2] * xr + M[5] * yr + M[8] * zr
    glat = np.arcsin(zg)
    glon = np.where((yg == 0.0) & (xg == 0.0), 0.0, np.arctan2(yg, xg))
    return glon, glat


def _trim(dx):
    """Simple cyclic trim about ±HALF_CYC (the >/< convention used everywhere except elem_center)."""
    dx = np.where(dx > HALF_CYC, dx - CYC, dx)
    dx = np.where(dx < -HALF_CYC, dx + CYC, dx)
    return dx


# ==========================================================================
# The full derivation (pipeline order of fesom_mesh_compute_metrics)
# ==========================================================================
def derive(raw: dict) -> dict:
    N, E, nl = raw["nod2D"], raw["elem2D"], raw["nl"]
    M = _rot_matrix()
    glon, glat = raw["nodes_deg"][:, 0] * RAD, raw["nodes_deg"][:, 1] * RAD
    geo = np.stack([glon, glat], axis=1)
    rlon, rlat = _g2r(M, glon, glat)
    coord = np.stack([rlon, rlat], axis=1)                     # rotated radians

    # --- orient_cw (BEFORE geometry): swap cols 1,2 where the rotated cross > 0 ---
    en = raw["elem_nodes"].copy()
    ax, ay = coord[en[:, 0], 0], coord[en[:, 0], 1]
    bx = _trim(coord[en[:, 1], 0] - ax); by = coord[en[:, 1], 1] - ay
    cx = _trim(coord[en[:, 2], 0] - ax); cy = coord[en[:, 2], 1] - ay
    ccw = (bx * cy - by * cx) > 0.0
    en[ccw, 1], en[ccw, 2] = raw["elem_nodes"][ccw, 2], raw["elem_nodes"][ccw, 1]
    n0, n1, n2 = en[:, 0], en[:, 1], en[:, 2]

    # --- CSR nod_in_elem2D (element-ascending order per node) ---
    pair_node = en.reshape(-1)                                 # (3E,) in element order
    pair_elem = np.repeat(np.arange(E, dtype=np.int32), 3)
    order = np.argsort(pair_node, kind="stable")               # group by node, keep elem order
    nie = pair_elem[order].astype(np.int32)
    counts = np.bincount(pair_node, minlength=N)
    offsets = np.zeros(N + 1, np.int32); offsets[1:] = np.cumsum(counts)

    # --- levels ---
    nlevels = raw["nlevels"]; nlevels_nod2D = raw["nlevels_nod2D"]
    ulevels = np.ones(E, np.int32); ulevels_nod2D = np.ones(N, np.int32)
    nlevels_nod2D_min = np.empty(N, np.int32)
    ul_max = np.ones(N, np.int32)
    np.minimum.at(_init_full(nlevels_nod2D_min, np.iinfo(np.int32).max), pair_node, nlevels[pair_elem])
    nlevels_nod2D_min = np.minimum(nlevels_nod2D_min, nlevels_nod2D)  # serial: no-empty guard is a no-op
    # ul_max = MAX of ulevels(=1) over cells ⇒ all 1 (mirror the reduction trivially)

    # --- elem_area (m²) ---  ay = cos(mean of 3 vertex rotated lats) == elem_cos
    lat0, lat1, lat2 = rlat[n0], rlat[n1], rlat[n2]
    lon0, lon1, lon2 = rlon[n0], rlon[n1], rlon[n2]
    ay = np.cos((lat0 + lat1 + lat2) / 3.0)
    aX = _trim(lon1 - lon0) * ay; aly = lat1 - lat0
    bX = _trim(lon2 - lon0) * ay; bly = lat2 - lat0
    elem_area = 0.5 * np.abs(aX * bly - bX * aly) * (R_EARTH ** 2)

    # --- node areas (CSR-order accumulation via element-ascending loop) ---
    area = np.zeros((N, nl)); third = elem_area / 3.0
    for e in range(E):                                         # element-ascending ⇒ CSR order
        nlev = int(nlevels[e]) - 1                             # C loops nz in [0, nlevels[e]-1)
        t = third[e]
        area[en[e, 0], :nlev] += t; area[en[e, 1], :nlev] += t; area[en[e, 2], :nlev] += t
    areasvol = np.zeros((N, nl))
    for n in range(N):
        k = int(nlevels_nod2D[n]) - 1                          # ulevels_nod2D-1 = 0 .. nlevels_nod2D-2
        areasvol[n, :k] = area[n, :k]
    ocean_area = float(areasvol[ulevels_nod2D <= 1, 0].sum())

    # --- elem centers (min-anchored cyclic), cos, metric, coriolis ---
    axk = np.stack([lon0, lon1, lon2], axis=1)
    amin = axk.min(axis=1, keepdims=True)
    axk = np.where(axk - amin >= HALF_CYC, axk - CYC, axk)     # second branch (< -HALF) never fires
    elem_center_x = axk.sum(axis=1) / 3.0
    elem_center_y = (lat0 + lat1 + lat2) / 3.0
    elem_cos = np.cos(elem_center_y)
    metric_factor = np.tan(elem_center_y) / R_EARTH
    cglon, cglat = _r2g(M, elem_center_x, elem_center_y)
    coriolis = 2.0 * OMEGA * np.sin(cglat)
    coriolis_node = 2.0 * OMEGA * np.sin(glat)

    # --- gradient_sca (lon diffs *elem_cos, lat diffs not; -0.5R/area) ---
    dX31 = _trim(lon2 - lon0) * elem_cos; dX21 = _trim(lon1 - lon0) * elem_cos
    dY31 = lat2 - lat0; dY21 = lat1 - lat0
    df = -0.5 * R_EARTH / elem_area
    gradient_sca = np.stack([(-dY31 + dY21) * df, (dY31) * df, (-dY21) * df,
                             (dX31 - dX21) * df, (-dX31) * df, (dX21) * df], axis=1)

    # --- edge geometry (dxdy radians; cross_dxdy meters) ---
    e0, e1 = raw["edges"][:, 0], raw["edges"][:, 1]
    edge_dxdy = np.stack([_trim(rlon[e1] - rlon[e0]), rlat[e1] - rlat[e0]], axis=1)
    # edge midpoint (asymmetric cyclic: adjust ax in one branch, bx in the other)
    eax, ebx = rlon[e0].copy(), rlon[e1].copy()
    diff = rlon[e0] - rlon[e1]
    eax = np.where(diff > HALF_CYC, eax - CYC, eax)
    ebx = np.where(diff < -HALF_CYC, ebx - CYC, ebx)
    emx = 0.5 * (eax + ebx); emy = 0.5 * (rlat[e0] + rlat[e1])
    et2 = raw["edge_tri"]; el1, el2 = et2[:, 0], et2[:, 1]
    cross = np.zeros((edges_len := e0.shape[0], 4))
    bx = _trim(elem_center_x[el1] - emx); by = elem_center_y[el1] - emy
    cross[:, 0] = bx * elem_cos[el1] * R_EARTH; cross[:, 1] = by * R_EARTH
    has2 = el2 >= 0
    el2s = np.where(has2, el2, 0)
    bx2 = _trim(elem_center_x[el2s] - emx); by2 = elem_center_y[el2s] - emy
    cross[:, 2] = np.where(has2, bx2 * elem_cos[el2s] * R_EARTH, 0.0)
    cross[:, 3] = np.where(has2, by2 * R_EARTH, 0.0)

    # --- mesh_resolution (full-precision pi; 3-pass Jacobi over CSR) ---
    inv_pi = 1.0 / PI_FULL
    mres = 2.0 * np.sqrt(areasvol[:, 0] * inv_pi)              # ulevels_nod2D-1 = 0
    for _ in range(3):
        mean3 = (mres[n0] + mres[n1] + mres[n2]) / 3.0
        acc = np.zeros(N); vol = np.zeros(N)
        # CSR-order accumulation: element-ascending pairs ⇒ np.add.at adds in that order
        np.add.at(acc, pair_node, np.repeat(mean3 * elem_area, 3))
        np.add.at(vol, pair_node, np.repeat(elem_area, 3))
        mres = np.where(vol > 0.0, acc / vol, 0.0)             # Jacobi copy-back

    # --- zbar_3d_n ---
    zbar = raw["zbar"]; zbar_3d_n = np.zeros((N, nl))
    for n in range(N):
        zbar_3d_n[n, :int(nlevels_nod2D[n])] = zbar[:int(nlevels_nod2D[n])]

    # --- edge_up_dn_tri (MFCT scan) ---
    edge_up_dn_tri = _build_edge_up_dn_tri(raw["edges"], en, rlon, rlat, nie, offsets)

    return {
        "coord_nod2D": coord, "geo_coord_nod2D": geo, "coast_flag": raw["coast"],
        "nlevels_nod2D": nlevels_nod2D, "nlevels_nod2D_min": nlevels_nod2D_min,
        "ulevels_nod2D": ulevels_nod2D, "ulevels_nod2D_max": ul_max, "depth": raw["depth"],
        "mesh_resolution": mres, "coriolis_node": coriolis_node, "area": area,
        "areasvol": areasvol, "zbar_3d_n": zbar_3d_n, "nod_in_elem2D_offsets": offsets,
        "elem_nodes": en, "nlevels": nlevels, "ulevels": ulevels, "elem_area": elem_area,
        "elem_cos": elem_cos, "metric_factor": metric_factor, "coriolis": coriolis,
        "elem_center_x": elem_center_x, "elem_center_y": elem_center_y, "gradient_sca": gradient_sca,
        "edges": raw["edges"], "edge_tri": raw["edge_tri"], "edge_dxdy": edge_dxdy,
        "edge_cross_dxdy": cross, "edge_up_dn_tri": edge_up_dn_tri,
        "zbar": zbar, "Z": 0.5 * (zbar[:-1] + zbar[1:]), "nod_in_elem2D": nie,
        "_meta": dict(nod2D=N, elem2D=E, edge2D=int(raw["edges"].shape[0]), nl=nl,
                      edge2D_in=int(np.sum((et2[:, 0] >= 0) & (et2[:, 1] >= 0))),
                      myDim_edge2D=int(raw["edges"].shape[0]), npes=1, ocean_area=ocean_area),
    }


def _init_full(arr, val):
    arr[:] = val
    return arr


def _edge_x_between_bc(bx, by, cx, cy, xx, xy):
    cr = cx * cx + cy * cy
    if cr == 0.0:
        return 0
    bxp = (bx * cx + by * cy) / cr; byp = (-bx * cy + by * cx) / cr
    xxp = (xx * cx + xy * cy) / cr; xyp = (-xx * cy + xy * cx) / cr
    import math
    ab = math.atan2(byp, bxp); axg = math.atan2(xyp, xxp)
    if ab > 0.0 and axg > 0.0 and axg < ab:
        return 1
    if ab < 0.0 and axg < 0.0 and axg > ab:
        return 1
    if ab == axg or axg == 0.0:
        return 1
    return 0


def _build_edge_up_dn_tri(edges, en, rlon, rlat, nie, offsets):
    import math
    ne = edges.shape[0]
    out = np.full((ne, 2), -1, np.int32)

    def scan(node, xx, xy):
        for idx in range(offsets[node], offsets[node + 1]):
            el = nie[idx]
            tri = en[el]
            js = 0 if tri[0] == node else (1 if tri[1] == node else (2 if tri[2] == node else -1))
            if js < 0:
                continue
            jb = 1 if js == 0 else 0
            jc = 1 if js == 2 else 2
            nb, nc = tri[jb], tri[jc]
            bx = rlon[nb] - rlon[node]; by = rlat[nb] - rlat[node]
            cx = rlon[nc] - rlon[node]; cy = rlat[nc] - rlat[node]
            if bx > HALF_CYC: bx -= CYC
            elif bx < -HALF_CYC: bx += CYC
            if cx > HALF_CYC: cx -= CYC
            elif cx < -HALF_CYC: cx += CYC
            if _edge_x_between_bc(bx, by, cx, cy, xx, xy):
                return el
        return -1

    for ed in range(ne):
        n1, n2 = int(edges[ed, 0]), int(edges[ed, 1])
        x0 = rlon[n2] - rlon[n1]; x1 = rlat[n2] - rlat[n1]
        if x0 > HALF_CYC: x0 -= CYC
        elif x0 < -HALF_CYC: x0 += CYC
        out[ed, 0] = scan(n1, -x0, -x1)
        out[ed, 1] = scan(n2, x0, x1)
    return out


# ==========================================================================
# Verification + write
# ==========================================================================
def verify_against(out: dict, ref_dir):
    """Diff each array vs the C export. Ints must be EXACT; floats are gated on a RELATIVE scale
    (the arrays span ~1e10 m² down to ~1e-20, so an absolute floor is meaningless — float64
    rounding from a different order-of-ops is rel ~1e-14). Returns {name: (abs, rel, is_int)}."""
    ref_dir = Path(ref_dir)
    diffs = {}
    for name, a in out.items():
        if name == "_meta":
            continue
        ref = np.load(ref_dir / f"{name}.npy")
        a = np.asarray(a)
        assert a.shape == ref.shape, f"{name}: shape {a.shape} != C-export {ref.shape}"
        if np.issubdtype(ref.dtype, np.integer):
            ad = int(np.max(np.abs(a.astype(np.int64) - ref.astype(np.int64)))) if a.size else 0
            diffs[name] = (ad, ad, True)
        else:
            ab = float(np.max(np.abs(a - ref))) if a.size else 0.0
            scale = float(np.max(np.abs(ref))) if a.size else 1.0
            diffs[name] = (ab, ab / scale if scale > 0 else ab, False)
    return diffs


def write_mesh(out: dict, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name, a in out.items():
        if name == "_meta":
            continue
        a = np.asarray(a)
        np.save(out_dir / f"{name}.npy", a.astype("<f8") if a.dtype.kind == "f" else a.astype("<i4"))
    m = out["_meta"]
    with open(out_dir / "meta.txt", "w") as f:
        for k in ("nod2D", "elem2D", "edge2D", "nl", "edge2D_in", "myDim_edge2D", "npes"):
            f.write(f"{k} {m[k]}\n")
        f.write(f"ocean_area {m['ocean_area']:.17g}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw_mesh_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--verify", help="a C-export dir to diff against")
    args = ap.parse_args()
    raw = read_fesom_ascii(args.raw_mesh_dir)
    print(f"[prepare_mesh] raw: nod2D={raw['nod2D']} elem2D={raw['elem2D']} nl={raw['nl']} "
          f"edge2D={raw['edges'].shape[0]}")
    out = derive(raw)
    if args.verify:
        diffs = verify_against(out, args.verify)
        worst_rel = max((rel for _, rel, isint in diffs.values() if not isint), default=0.0)
        worst_int = max((ad for ad, _, isint in diffs.values() if isint), default=0)
        for k in sorted(diffs, key=lambda k: -diffs[k][1]):
            ab, rel, isint = diffs[k]
            print(f"   {k:20s} abs={ab:.3e}" + ("" if isint else f"  rel={rel:.3e}"))
        print(f"[prepare_mesh] {len(diffs)} arrays; worst rel(float)={worst_rel:.3e} int={worst_int}")
        # float64 order-of-ops rounding floor ⇒ relative ~1e-12; ints exact
        print("MESH_PREP_OK" if (worst_rel < 1e-11 and worst_int == 0) else "MESH_PREP_MISMATCH")
    else:
        write_mesh(out, args.out_dir)
        print(f"[prepare_mesh] wrote mesh to {args.out_dir}")


if __name__ == "__main__":
    main()
