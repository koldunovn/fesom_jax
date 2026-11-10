#!/usr/bin/env python
"""Whole-step check: a fesom_jax restart written after N steps vs the oracle's full dumps of step N.

    python scripts/core3/compare_step.py JAX_RESTART MESH_JAX DUMP_DIR --step 1

Compares the end-of-step fields (T, S [15], uv [10], eta_n [12], hbar [11], hnode [16], a_ice / m_ice [18]) split
into cavity, cavity-front and open nodes/elements, as max |diff| / max |field| and the 99th percentile. This mixes
every substep, including the ones whose jaxcpl settings fesom_jax does not have yet (opt_visc 5, GM/Redi
scalings, the tracer scheme: lane L-SFC), so it is a sanity rung, not a port check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fesom_jax import io_dump, zarr_output  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("restart"); ap.add_argument("mesh_dir"); ap.add_argument("dump_dir")
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    L = lambda n: np.load(Path(a.mesh_dir) / f"{n}.npy")
    uln, nln, ule, nle, ulmax, en = (L(x) for x in ("ulevels_nod2D", "nlevels_nod2D", "ulevels", "nlevels",
                                                    "ulevels_nod2D_max", "elem_nodes"))
    N, E = uln.size, ule.size
    k = np.arange(48)[None, :]
    lay_n = (k >= uln[:, None] - 1) & (k < nln[:, None] - 1)
    lay_e = (k >= ule[:, None] - 1) & (k < nle[:, None] - 1)
    gn = {"cavity": uln > 1, "front": (uln == 1) & (ulmax > 1), "open": ulmax == 1}
    ef = (ule == 1) & (ulmax[en].max(axis=1) > 1)
    ge = {"cavity": ule > 1, "front": ef, "open": (ule == 1) & ~ef}
    R = lambda f: np.asarray(zarr_output.reconstruct_global(a.restart, f), np.float64)
    D = lambda sub, name, n=N: io_dump.load_full_field(a.dump_dir, a.step, sub, name, n)
    rows = [("T", R("T"), D(15, "T"), lay_n, gn), ("S", R("S"), D(15, "S"), lay_n, gn),
            ("u", R("uv")[..., 0], D(10, "uv_1", E), lay_e, ge), ("v", R("uv")[..., 1], D(10, "uv_2", E), lay_e, ge),
            ("eta_n", R("eta_n"), D(12, "eta_n")[:, 0], None, gn), ("hbar", R("hbar"), D(11, "hbar")[:, 0], None, gn),
            ("a_ice", R("a_ice"), D(18, "a_ice")[:, 0], None, gn), ("m_ice", R("m_ice"), D(18, "m_ice")[:, 0], None, gn),
            ("tke", R("tke"), None, None, gn)]
    for name, j, o, m, g in rows:
        if o is None:
            continue
        if m is not None:
            nk = o.shape[1]; j = j[:, :nk]; m = m[:, :nk]
        else:
            m = np.ones(j.shape, bool)
        sc = np.abs(o[m]).max() or 1.0
        line = f"{name:6s} scale {sc:.3e} |"
        for gname, gm in g.items():
            mm = m & (gm[:, None] if j.ndim == 2 else gm)
            d = np.abs(j[mm] - o[mm]) / sc
            line += f" {gname}: max {d.max():.2e} p99 {np.percentile(d, 99):.2e} |"
        print(line)


if __name__ == "__main__":
    main()
