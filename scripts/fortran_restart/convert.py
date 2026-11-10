"""Convert a Fortran FESOM2 restart into a fesom-jax restart directory.

    python scripts/fortran_restart/convert.py --oce <fesom.YYYY.oce.restart[.nc]> \
        --ice <fesom.YYYY.ice.restart[.nc]> --mesh-dir <fesom-jax mesh> --ic-dir <dir with T_ic.npy> \
        --date 1958-02-01 --start-year 1958 --dt 1800 --out <new restart dir>

The result loads with ``run_from_config.py --restart-in`` (and ``core2_adj``): a canonical-global
store with ``step`` = steps from ``--start-year``-01-01 at ``--dt`` (pass the same ``--year`` to
the run) and ``calendar_date`` = ``--date``. ``--ic-dir`` gives the reference cold start that fills
the dry lanes and the fields FESOM does not restart (:mod:`fesom_jax.fortran_restart`). Nothing is
overwritten: an existing ``--out`` is an error.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--oce", required=True)
    ap.add_argument("--ice", default=None)
    ap.add_argument("--mesh-dir", required=True)
    ap.add_argument("--ic-dir", default=None, help="T_ic.npy/S_ic.npy of the mesh (reference)")
    ap.add_argument("--date", required=True, help="the restart's date, YYYY-MM-DD[THH:MM]")
    ap.add_argument("--start-year", type=int, required=True, help="the run's forcing start year")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; restarts are never overwritten")

    import jax
    import numpy as np
    from fesom_jax import fortran_restart as FR
    from fesom_jax import partit, shard_mesh, zarr_output
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import cold_start_state

    when = datetime.datetime.fromisoformat(args.date)
    step = FR.restart_step(when, args.start_year, args.dt)
    mesh = load_mesh(args.mesh_dir)
    ref = cold_start_state(mesh, args.ic_dir, xp=np)
    oce = FR.FortranRestart(args.oce)
    ice = FR.FortranRestart(args.ice) if args.ice else None
    state, rep = FR.state_from_fortran(mesh, oce, ice, ref)
    bad = [k for k, v in vars(state).items() if not np.all(np.isfinite(np.asarray(v)))]
    if bad:
        raise SystemExit(f"non-finite fields after the conversion: {bad}")

    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    sp = shard_mesh.partition_state(jax.device_get(state), part)
    folded = jax.tree.map(lambda a: np.asarray(a).reshape((-1,) + np.shape(a)[2:]), sp)
    doy = (when - datetime.datetime(when.year, 1, 1)).days + 1
    attrs = {"source": "fortran restart (scripts/fortran_restart/convert.py)",
             "fortran_oce": str(Path(args.oce).resolve()),
             "fortran_ice": str(Path(args.ice).resolve()) if args.ice else "",
             "fortran_vars_oce": ",".join(oce.variables),
             "conversion_report": json.dumps(rep)}
    zarr_output.write_restart(out, folded, sm, part, step=step,
                              calendar_date=f"{when.year:04d}-doy{doy:03d}", dt_stage=args.dt,
                              attrs=attrs)
    print(json.dumps({"out": str(out), "step": step, "report": rep}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
