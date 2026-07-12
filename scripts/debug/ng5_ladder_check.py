#!/usr/bin/env python
"""Per-rung NG5 stability check (Task B2) — POST-HOC, reads a written restart store.

The R0→R3 cold-spin-up ladder gates each rung on the next: a rung must be **finite / CFL-stable**
before the next is launched. The live gate is the in-job ``--diagnostics`` flag of
``run_from_config.py`` (gather-free reductions on the resident final State); THIS script is the
**offline** companion — point it at a saved restart store and it reports the same health verdict
**plus** the masked **min layer thickness > 0** check (the zstar-ALE stability indicator the in-job
path omits, since that one needs the mesh layer mask and the in-job State is padded/folded).

It streams each State leaf via :func:`fesom_jax.zarr_output.reconstruct_global` — **one global
field resident at a time** (an NG5 3-D field is ≈ 4 GB; never the whole ~40-leaf State at once) —
so it runs on a single ordinary node (no device mesh, no GPU), e.g. ``-p compute``. Usage:

    python scripts/debug/ng5_ladder_check.py <restart_store> --mesh-dir <MESH> [--label R0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fesom_jax.diagnostics import format_diagnostics, verdict
from fesom_jax.mesh import load_mesh
from fesom_jax.zarr_output import _all_state_fields, reconstruct_global


def streaming_diagnostics(store, mesh=None) -> dict:
    """Build the :func:`fesom_jax.diagnostics.state_diagnostics` dict (+ ``min_hnode_wet`` when a
    ``mesh`` is given) by streaming each leaf's global reconstruction on the HOST, one field
    resident at a time (bounded peak ≈ one global field)."""
    store = Path(store)
    nonfinite_by_field, field_maxabs = {}, {}
    n_nonfinite = 0
    d = {"max_abs_uv": None, "max_abs_uvnode": None, "max_abs_w": None, "max_abs_eta": None,
         "T_min": None, "T_max": None, "S_min": None, "S_max": None,
         "a_ice_max": None, "m_ice_max": None, "min_hnode_wet": None}

    for name in _all_state_fields():
        g = np.asarray(reconstruct_global(store, name))
        nf = int(np.sum(~np.isfinite(g)))
        n_nonfinite += nf
        if nf:
            nonfinite_by_field[name] = nf
        field_maxabs[name] = float(np.max(np.abs(g)))
        if name == "uv":
            d["max_abs_uv"] = float(np.max(np.abs(g)))
        elif name == "uvnode":
            d["max_abs_uvnode"] = float(np.max(np.abs(g)))
        elif name == "w":
            d["max_abs_w"] = float(np.max(np.abs(g)))
        elif name == "eta_n":
            d["max_abs_eta"] = float(np.max(np.abs(g)))
        elif name == "T":
            d["T_min"], d["T_max"] = float(np.min(g)), float(np.max(g))
        elif name == "S":
            d["S_min"], d["S_max"] = float(np.min(g)), float(np.max(g))
        elif name == "a_ice":
            d["a_ice_max"] = float(np.max(g))
        elif name == "m_ice":
            d["m_ice_max"] = float(np.max(g))
        elif name == "hnode" and mesh is not None:
            mask = np.asarray(mesh.node_layer_mask)        # [nod2D, nl] wet-layer mask
            d["min_hnode_wet"] = float(g[mask].min()) if mask.any() else float("nan")
        del g

    d.update(n_nonfinite=n_nonfinite, nonfinite_by_field=nonfinite_by_field,
             field_maxabs=field_maxabs)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("store", help="restart store path (a write_restart output)")
    ap.add_argument("--mesh-dir", help="mesh dir (enables the masked min-layer-thickness check)")
    ap.add_argument("--label", default="", help="rung label for the report (e.g. R0)")
    args = ap.parse_args()

    mesh = load_mesh(args.mesh_dir) if args.mesh_dir else None
    d = streaming_diagnostics(args.store, mesh=mesh)
    print(format_diagnostics(d, label=args.label))
    if d.get("min_hnode_wet") is not None:
        mh = d["min_hnode_wet"]
        flag = "" if (mh == mh and mh > 0.0) else "  <-- NONPOSITIVE (zstar instability)"
        print(f"  min wet layer thickness = {mh:.4g} m{flag}")

    ok, _ = verdict(d)
    thick_ok = d.get("min_hnode_wet") is None or (d["min_hnode_wet"] == d["min_hnode_wet"]
                                                  and d["min_hnode_wet"] > 0.0)
    tag = f" [{args.label}]" if args.label else ""
    print(f"NG5_STABLE_OK{tag}" if (ok and thick_ok) else f"NG5_UNSTABLE{tag}")
    raise SystemExit(0 if (ok and thick_ok) else 1)


if __name__ == "__main__":
    main()
