"""Gate G5 (REcoM plan Task 14, lane W): the 30-day pi run with full REcoM against the physics-matched Fortran twin.

JAX runs ``configs/pi_recom_bgc.yaml`` (with ``recom.budget``) through the production driver
(:func:`fesom_jax.run.run_from_config`, one device, chunks of one day, the daily mean stream with the BGC fields)
from the twin's step-0 state (:func:`fesom_jax.recom.reference.twin_step0_state`). Compared with the twin
(``recom_ref/pi_30d_jan_jaxphys``):

* the daily inventories ``sum mean_day(X) dz_ref areasvol`` of DIC, Alk, DIN, DSi, DFe, O2 (and T, S for the
  physics' own divergence), the global air-sea CO2 flux and NPP (``sum mean_day(.) areasvol(1)``), both sides from
  their daily means with the same nominal-thickness weighting;
* the end state (step 960) against the twin's ``step_end`` dump: every BGC tracer, Benthos, T and S;
* the closed JAX budget over the 30 days (``State.bgc_budget``): every tracer's inventory change against surface BC +
  stage 1 + REcoM's tracer-loop terms.

Run on a compute node: ``python scripts/recom/g5_pi_30d.py --out <dir> [--days 30]``. Prints ``G5_DONE``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from fesom_jax import partit, shard_mesh, ssh, surface_forcing  # noqa: E402
from fesom_jax.io_dump import load_recom_dump  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.recom import reference as R  # noqa: E402
from fesom_jax.recom import tracers as T  # noqa: E402
from fesom_jax.recom import wiring  # noqa: E402
from fesom_jax.run import run_from_config  # noqa: E402
from fesom_jax.run_config import load_yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INV = {"DIC": "DIC", "Alk": "Alk", "DIN": "DIN", "DSi": "DSi", "DFe": "DFe", "O2": "O2"}
SURF = {"co2flux": "CO2f"}


def _read_zarr_day(path, name):
    import zarr
    g = zarr.open_group(str(path), mode="r")
    a = np.asarray(g[name])[0]                     # (nz, nod2) or (nod2,): the ushow layout, time leading
    return a.T if a.ndim == 2 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "pi_recom_bgc.yaml")
    ap.add_argument("--analyze", action="store_true", help="skip the run; analyse <out>/final.npz and <out>/daily")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.config)
    n_steps = 32 * args.days
    cfg = dataclasses.replace(cfg, recom=cfg.recom._replace(budget=True), n_steps=n_steps, duration=None,
                              restart_out=str(args.out / "restart"))
    mesh = load_mesh()
    nl1 = mesh.nl - 1
    phys = cfg.physics_kwargs()
    d1 = load_recom_dump(R.TWIN / "dumps", 1, names=R.dump_names(step_end=False))[0]
    st0 = R.twin_step0_state(mesh, d1, phys["ale_cfg"], cfg.recom)
    daily = args.out / "daily"
    if not args.analyze:
        part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
        sm = shard_mesh.build_sharded_mesh(mesh, part)
        sop = ssh.partition_ssh_operator(ssh.build_ssh_operator(mesh, dt=cfg.dt), part)
        forcing = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=np.asarray(st0.T[:, 0]), dt=cfg.dt,
                                                        mslp=True, bgc_dust_path=cfg.bgc_dust_path(),
                                                        **cfg.forcing_paths())
        res = run_from_config(cfg, mesh=mesh, part=part, sm=sm, sop=sop, forcing=forcing, state0=st0, year=1958,
                              chunk_steps=32, out_dir=cfg.restart_out, daily_out=str(daily), progress=True)
        fin = jax.device_get(res.state_p)
        # one device, serial partition: lanes [0, nod2D) are the global nodes in order
        np.savez(args.out / "final.npz", **{k: np.asarray(getattr(fin, k))[: mesh.nod2D] for k in
                                           ("T", "S", "hnode", "bgc", "bgc_benthos", "bgc_budget", "bgc_diag")})
    fin = dict(np.load(args.out / "final.npz"))
    take = lambda k: fin[k]                                                            # noqa: E731
    report = {}

    # ---- closed budget ----
    I0 = wiring.inventory(st0.bgc, st0.hnode, mesh)
    IN = wiring.inventory(take("bgc"), take("hnode"), mesh)
    b = take("bgc_budget").sum(axis=0)
    # DICremin starts at zero (and is reset at the surface every step): scale by the larger inventory or the terms
    scale = np.maximum.reduce([np.abs(I0), np.abs(IN), np.abs(b[:3]).sum(axis=0)])
    rel = np.abs(IN - I0 - (b[0] + b[1] + b[2])) / scale
    report["budget_terms_total"] = {T.NAMES[s]: dict(dI=float(IN[s] - I0[s]), **{t: float(b[i, s]) for i, t in
                                                                                 enumerate(wiring.BUDGET_TERMS)})
                                    for s in range(T.NTR)}
    report["budget_rel"] = dict(zip(T.NAMES, rel.tolist()))
    print(f"G5 BUDGET {args.days} d: worst {T.NAMES[int(np.argmax(rel))]} {rel.max():.2e}, median {np.median(rel):.2e}")

    # ---- daily inventories ----
    import netCDF4
    dz = np.diff(-np.asarray(mesh.zbar))[:nl1]                                        # nominal layer thickness
    av = np.asarray(mesh.areasvol)[:, :nl1]
    nlev = np.asarray(mesh.nlevels_nod2D) - 1
    wet = np.arange(nl1)[None, :] < nlev[:, None]
    w3 = np.where(wet, av * dz[None, :], 0.0)                                          # (nod, nl-1)
    a2 = av[:, 0]
    rows = []
    for day in range(1, args.days + 1):
        p = daily / f"day_1958_{day:03d}"
        row = {"day": day}
        for jn, fn in INV.items():
            ff = netCDF4.Dataset(R.TWIN / f"{fn}.fesom.1958.nc")[fn][day - 1]          # (nl-1, nod)
            j = _read_zarr_day(p, jn)[:, :nl1]
            row[jn] = (float(np.sum(j * w3)), float(np.sum(np.asarray(ff).T * w3)))
        for jn, fn in SURF.items():
            ff = np.asarray(netCDF4.Dataset(R.TWIN / f"{fn}.fesom.1958.nc")[fn][day - 1])
            row[jn] = (float(np.sum(_read_zarr_day(p, jn) * a2)), float(np.sum(ff * a2)))
        ffn = sum(np.asarray(netCDF4.Dataset(R.TWIN / f"{v}.fesom.1958.nc")[v][day - 1]) for v in ("NPPn", "NPPd"))
        row["npp"] = (float(np.sum(_read_zarr_day(p, "npp") * a2)), float(np.sum(ffn * a2)))
        rows.append(row)
    print(f"\n{'day':>4} " + " ".join(f"{k:>18}" for k in rows[0] if k != "day"))
    for r in rows:
        print(f"{r['day']:>4} " + " ".join(f"{(v[0] - v[1]) / abs(v[1]):18.3e}" for k, v in r.items() if k != "day"))
    report["daily"] = rows

    # ---- end state vs the twin's step_end ----
    if args.days == 30:
        f = load_recom_dump(R.TWIN / "dumps", 960, names=["step_end.Benthos"] +
                            [f"step_end.tr{i}" for i in (1, 2) + T.BGC_IDS])[0]
        e = {}
        rel2 = lambda a, b: float(np.max(np.abs(a[wet] - b[wet]))) / float(np.max(np.abs(b[wet])))  # noqa: E731
        e["T"] = rel2(take("T")[:, :nl1], f["step_end.tr1"])
        e["S"] = rel2(take("S")[:, :nl1], f["step_end.tr2"])
        for s, tid in enumerate(T.BGC_IDS):
            e[T.NAMES[s]] = rel2(take("bgc")[:, :nl1, s], f[f"step_end.tr{tid}"])
        ben = np.asarray(f["step_end.Benthos"]).T
        e["Benthos"] = float(np.max(np.abs(take("bgc_benthos") - ben))) / float(np.max(np.abs(ben)))
        report["end_rel"] = e
        print("\nG5 END STATE (step 960) max|J-F|/max|F| over wet cells:")
        for k, v in sorted(e.items(), key=lambda kv: -kv[1]):
            print(f"  {k:>10} {v:.3e}")
    (args.out / "g5_report.json").write_text(json.dumps(report, indent=1))
    print("G5_DONE")


if __name__ == "__main__":
    main()
