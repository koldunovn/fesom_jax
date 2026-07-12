#!/usr/bin/env python3
"""Task 0.3 gate: load the C-exported pi mesh and assert counts / shapes / ranges.

Mirrors FRESH_START.md §20's "test immediately" list. Pure numpy (no JAX).

Usage: python scripts/debug/verify_mesh_export.py [export_dir]   (default: data/mesh_pi)
"""
import pathlib
import sys

import numpy as np

# pi expected counts. NOTE: this pi mesh is 48-level globally — FRESH_START's
# "nl~23" is the PER-NODE level count (e.g. node 0 has nlev=23); the global nl
# (len(zbar), down to -6250 m) is 48, with nlevels_nod2D in [5, 46].
EXPECT_NOD2D = 3140
EXPECT_ELEM2D = 5839
EXPECT_NL = 48


def load_meta(d: pathlib.Path) -> dict:
    meta = {}
    for line in (d / "meta.txt").read_text().splitlines():
        k, v = line.split()
        meta[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
    return meta


def main() -> int:
    d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/mesh_pi")
    meta = load_meta(d)
    A = {p.stem: np.load(p) for p in sorted(d.glob("*.npy"))}

    print(f"export: {d}")
    print(f"meta:   {meta}")
    print("arrays:")
    for name in sorted(A):
        a = A[name]
        rng = f"[{np.min(a):.5g}, {np.max(a):.5g}]" if a.size else "[]"
        print(f"  {name:24s} {str(a.shape):16s} {str(a.dtype):5s} {rng}")

    nod, elem, edge, nl = meta["nod2D"], meta["elem2D"], meta["edge2D"], meta["nl"]

    # --- counts ---
    assert nod == EXPECT_NOD2D, f"nod2D {nod} != {EXPECT_NOD2D}"
    assert elem == EXPECT_ELEM2D, f"elem2D {elem} != {EXPECT_ELEM2D}"
    assert nl == EXPECT_NL, f"nl {nl} != {EXPECT_NL}"
    assert nl == A["zbar"].shape[0], "nl != len(zbar)"
    assert A["nlevels_nod2D"].max() <= nl, "nlevels_nod2D exceeds nl"

    # --- shapes (catches packing/layout bugs) ---
    assert A["coord_nod2D"].shape == (nod, 2)
    assert A["geo_coord_nod2D"].shape == (nod, 2)
    assert A["elem_nodes"].shape == (elem, 3)
    assert A["edges"].shape == (edge, 2)
    assert A["edge_tri"].shape == (edge, 2)
    assert A["gradient_sca"].shape == (elem, 6)
    assert A["edge_cross_dxdy"].shape == (edge, 4)
    assert A["edge_dxdy"].shape == (edge, 2)
    assert A["area"].shape == (nod, nl)
    assert A["areasvol"].shape == (nod, nl)
    assert A["zbar"].shape == (nl,)
    assert A["Z"].shape == (nl - 1,)
    assert A["nod_in_elem2D_offsets"].shape == (nod + 1,)
    assert A["edge_up_dn_tri"].shape == (meta["myDim_edge2D"], 2)

    # --- index ranges (0-based; FRESH_START §20) ---
    assert A["elem_nodes"].min() >= 0 and A["elem_nodes"].max() < nod, "elem_nodes range"
    assert A["edges"].min() >= 0 and A["edges"].max() < nod, "edges range"
    et = A["edge_tri"]
    assert et.min() >= -1 and et.max() < elem, "edge_tri range"
    # at most 2 adjacent elements; every edge has ≥1; boundary edges have a -1.
    nvalid = (et >= 0).sum(axis=1)
    assert nvalid.min() >= 1 and nvalid.max() <= 2, "edge_tri adjacency count"
    assert (nvalid == 1).any(), "expected some boundary edges (one -1)"
    updn = A["edge_up_dn_tri"]
    assert updn.min() >= -1 and updn.max() < elem, "edge_up_dn_tri range"

    # --- CSR node→elem ---
    off = A["nod_in_elem2D_offsets"]
    flat = A["nod_in_elem2D"]
    assert off[0] == 0 and np.all(np.diff(off) >= 1), "CSR offsets non-monotone"
    assert off[-1] == flat.shape[0], "CSR flat length mismatch"
    assert flat.min() >= 0 and flat.max() < elem, "CSR elem ids range"

    # --- value ranges (sanity) ---
    assert np.all(np.isfinite(A["gradient_sca"])), "gradient_sca non-finite"
    assert A["elem_area"].min() > 0, "non-positive elem_area"
    assert A["area"][:, 0].min() > 0, "non-positive surface CV area"
    assert np.all(np.abs(A["coord_nod2D"][:, 1]) <= np.pi / 2 + 1e-9), "rot lat out of range"
    # zbar: 0 at surface, strictly deepening (negative downward)
    assert A["zbar"][0] <= 1e-9, "zbar[0] not ~surface"
    assert np.all(np.diff(A["zbar"]) < 0), "zbar not monotonically downward"
    assert A["coriolis"].shape == (elem,) and np.all(np.isfinite(A["coriolis"]))

    print(f"\nelem_area  min/mean/max = {A['elem_area'].min():.3e} / "
          f"{A['elem_area'].mean():.3e} / {A['elem_area'].max():.3e}  m^2")
    print(f"zbar[0..3] = {A['zbar'][:4]}")
    print("\nTASK 0.3 GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
