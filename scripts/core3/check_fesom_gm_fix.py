#!/usr/bin/env python
"""Local test of the FESOM2 init_Redi_GM fix (lane L-SFC, nothing upstream): a double-precision build of FESOM main
12ebf60b with and without the fix (both with the oracle dumps plus fer_k / Ki at the GM bolus stage), 6 steps.

1. Unfixed vs fixed, every full-dump field of every step: bitwise or not, and where (the nodes with
   ulevels_nod2D /= ulevels_nod2D_max, their neighbours, elsewhere).
2. CORE3 only: the fixed run's fer_k and Ki against fesom_jax's correct GM coefficients (gm.gm_diagnostics, the
   bug switch off) on the fixed run's own inputs of the same step, at the affected nodes and everywhere.

    python scripts/core3/check_fesom_gm_fix.py RUN_UNFIXED RUN_FIXED MESH_JAX [--jax] [--steps 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fesom_jax import io_dump  # noqa: E402


def names(run):
    out = set()
    for f in (Path(run) / "dumps" / "full").glob("s*_*_*.00000"):
        out.add(f.name[:-6])
    return sorted(out, key=lambda x: (int(x.split("_")[0][1:]), int(x.split("_")[1]), x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unfixed"); ap.add_argument("fixed"); ap.add_argument("mesh_dir")
    ap.add_argument("--jax", action="store_true"); ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--config", default=str(ROOT / "configs" / "core3_cavity_jaxcpl.yaml"))
    a = ap.parse_args()
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh(a.mesh_dir)
    N, E = int(mesh.nod2D), int(mesh.elem2D)
    uln, ulmax = np.asarray(mesh.ulevels_nod2D), np.asarray(mesh.ulevels_nod2D_max)
    en = np.asarray(mesh.elem_nodes)
    bad = uln != ulmax
    near = np.zeros(N, bool)
    near[en[bad[en].any(axis=1)].ravel()] = True
    near &= ~bad
    print(f"[fix] {int(bad.sum())} nodes with ulevels_nod2D /= ulevels_nod2D_max; {int(near.sum())} neighbours", flush=True)
    ebad = bad[en].any(axis=1)
    n_bitwise, n_diff = 0, 0
    for nm in names(a.fixed):
        st, sub, field = nm.split("_", 2)
        st, sub = int(st[1:]), int(sub)
        files = list((Path(a.fixed) / "dumps" / "full").glob(nm + ".00000"))
        b = files[0].read_bytes()
        n_own = int(np.frombuffer(b, "<i4", count=1)[0])
        gid = np.frombuffer(b, "<i4", count=n_own, offset=8)
        n = N if gid.max() <= N and field not in ("uv_1", "uv_2", "uv_rhs_1", "uv_rhs_2", "pgf_x", "pgf_y", "Av",
                                                    "fer_uv_1", "fer_uv_2", "helem", "stress_surf_x",
                                                    "stress_surf_y", "sigma11", "sigma12", "sigma22",
                                                    "elem_area") else E
        x = io_dump.load_full_field(str(Path(a.unfixed) / "dumps"), st, sub, field, n)
        y = io_dump.load_full_field(str(Path(a.fixed) / "dumps"), st, sub, field, n)
        if np.array_equal(np.nan_to_num(x, nan=-9e99).view(np.uint64), np.nan_to_num(y, nan=-9e99).view(np.uint64)):
            n_bitwise += 1
            continue
        n_diff += 1
        d = np.nanmax(np.abs(x - y), axis=1)
        sc = np.nanmax(np.abs(x)) or 1.0
        if not bad.any():
            where = f"max {np.nanmax(d) / sc:.1e} at {int((d > 0).sum())} points"
        elif n == N:
            where = (f"affected {np.nanmax(d[bad]) / sc:.1e} neighbours {np.nanmax(d[near]) / sc:.1e} "
                     f"elsewhere {np.nanmax(d[~bad & ~near]) / sc:.1e} ({int((d[~bad & ~near] > 0).sum())} nodes)")
        else:
            where = (f"elements at affected nodes {np.nanmax(d[ebad]) / sc:.1e} elsewhere {np.nanmax(d[~ebad]) / sc:.1e}"
                     f" ({int((d[~ebad] > 0).sum())} elements)")
        print(f"[fix] differs: step {st} substep {sub:2d} {field:22s} {where}", flush=True)
    print(f"[fix] {n_bitwise} field dumps bitwise, {n_diff} differ", flush=True)

    if a.jax:
        import jax
        import jax.numpy as jnp
        from fesom_jax import gm
        from fesom_jax.params import Params
        from fesom_jax.run_config import load_yaml
        gc = load_yaml(a.config).gm._replace(reproduce_fesom_bug_kgm_surface_level=False)
        p = Params.defaults(gc)
        nl = int(mesh.nl)
        pad = lambda x: np.pad(x, ((0, 0), (0, nl - x.shape[1]))) if x.shape[1] < nl else x  # noqa: E731
        ifc, lay = np.asarray(mesh.node_iface_mask), np.asarray(mesh.node_layer_mask)
        fn = jax.jit(lambda T, S, b, h, e: gm.gm_diagnostics(mesh, T, S, b, h, e, p, gc))
        for s in range(1, a.steps + 1):
            D = lambda sub, name, n=N: pad(io_dump.load_full_field(str(Path(a.fixed) / "dumps"), s, sub, name, n))  # noqa: E731
            out = fn(*(jnp.asarray(np.nan_to_num(v)) for v in (D(20, "T"), D(20, "S"), D(1, "bvfreq"), D(20, "hnode"),
                                                                 D(16, "helem", E))))
            for lab, mine, ref, m in (("fer_k", out.fer_K, D(14, "fer_k"), ifc), ("Ki", out.Ki, D(14, "Ki"), lay)):
                mine = np.asarray(mine)
                m = m[:, : ref.shape[1]] & np.isfinite(ref[:, : m.shape[1]])
                mine, ref = mine[:, : m.shape[1]], ref[:, : m.shape[1]]
                sc = np.abs(ref[m]).max()
                db = np.abs(mine - ref)[m & bad[:, None]]
                da = np.abs(mine - ref)[m]
                print(f"[fix] step {s} {lab:5s} fixed FESOM vs fesom_jax (correct): affected nodes "
                      f"{db.max() / sc:.2e}, all nodes {da.max() / sc:.2e}", flush=True)
            if s == 1:
                ref = D(14, "fer_k")
                m = ifc[:, : ref.shape[1]] & bad[:, None]
                print(f"[fix] step 1 fer_k at affected nodes: fixed min/max {np.nanmin(ref[m]):.1f}/{np.nanmax(ref[m]):.1f}",
                      flush=True)


if __name__ == "__main__":
    main()
