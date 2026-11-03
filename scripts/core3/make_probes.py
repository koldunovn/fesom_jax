#!/usr/bin/env python
"""Pick probe nodes/elements for the CORE3 cavity oracle dumps (FESOM_DUMP_PROBES / _ELEM).

    python scripts/core3/make_probes.py MESH_JAX OUT_DIR

Writes OUT_DIR/probes_nod.txt and probes_elem.txt (1-based global ids, one per line) and probes.txt (a table:
kind, gid, lon, lat, ulevels, nlevels). Picks, deterministically: deep and shallow cavity nodes, open nodes at
the cavity front (ulevels_nod2D == 1 < ulevels_nod2D_max), Southern Ocean and Arctic sea-ice nodes, open-ocean
nodes; plus, per probe node, its first element, and cavity / cavity-front elements.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main():
    mesh, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    L = lambda n: np.load(mesh / f"{n}.npy")
    uln, nln, ulmax = L("ulevels_nod2D"), L("nlevels_nod2D"), L("ulevels_nod2D_max")
    ul, nle, en = L("ulevels"), L("nlevels"), L("elem_nodes")
    geo = np.degrees(L("geo_coord_nod2D")); lon, lat = geo[:, 0], geo[:, 1]
    off, nie = L("nod_in_elem2D_offsets"), L("nod_in_elem2D")
    rng = np.random.default_rng(20261103)

    def pick(mask, k):
        idx = np.flatnonzero(mask)
        return list(rng.choice(idx, size=min(k, idx.size), replace=False)) if idx.size else []

    kinds = [
        ("cavity_deep", (uln >= 20), 3),
        ("cavity_mid", (uln >= 8) & (uln < 20), 3),
        ("cavity_shallow", (uln >= 2) & (uln < 8), 3),
        ("cavity_front", (uln == 1) & (ulmax > 1), 4),
        ("south_ice", (uln == 1) & (ulmax == 1) & (lat < -62) & (nln > 30), 3),
        ("arctic", (lat > 78) & (nln > 30), 2),
        ("open", (np.abs(lat) < 45) & (nln > 40), 3),
    ]
    rows, nodes = [], []
    for name, m, k in kinds:
        for n in pick(m, k):
            nodes.append(int(n)); rows.append((name, int(n) + 1))
    elems = [int(nie[off[n]]) for n in nodes]
    erows = [("of_node_" + r[0], e + 1) for r, e in zip(rows, elems)]
    for name, m, k in (("elem_cavity", ul > 1, 4), ("elem_front", (ul == 1) & (ulmax[en].max(axis=1) > 1), 4)):
        for e in pick(m, k):
            elems.append(int(e)); erows.append((name, int(e) + 1))
    (out / "probes_nod.txt").write_text("".join(f"{g}\n" for _, g in rows))
    (out / "probes_elem.txt").write_text("".join(f"{g}\n" for _, g in erows))
    with open(out / "probes.txt", "w") as f:
        f.write("# kind gid lon lat ulevels nlevels  (nodes, then elements: ulevels/nlevels of the element)\n")
        for (k, g) in rows:
            n = g - 1
            f.write(f"N {k:16s} {g:7d} {lon[n]:9.3f} {lat[n]:8.3f} {uln[n]:3d} {nln[n]:3d}\n")
        for (k, g) in erows:
            e = g - 1; c = en[e]
            f.write(f"E {k:22s} {g:7d} {lon[c].mean():9.3f} {lat[c].mean():8.3f} {ul[e]:3d} {nle[e]:3d}\n")
    print((out / "probes.txt").read_text())


if __name__ == "__main__":
    main()
