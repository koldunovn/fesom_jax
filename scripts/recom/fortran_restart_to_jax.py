"""A FESOM2-REcoM Fortran restart -> a fesom-jax restart store (the warm start of the spectral REcoM runs, lane W of
docs/plans/completed/20261028-recom-spectral-port.md).

FESOM2 writes ``<ResultPath>/fesom.<year>.{oce,ice,bio}.restart/``: one NetCDF file per variable (``io_restart.F90``),
global node / element order, FESOM's ``nl-1`` layers (``nz_1``) or ``nl`` interfaces (``nz``), one record with
``time`` (seconds into the year) and ``iter``. The tracers are named by POSITION in ``namelist.tra``: ``tra00NN``, except
the positions FESOM2 spells as transient tracers (6 ``sf6``, 12 ``cfc12``, 14 ``r14c``, 39 ``r39ar``); positions 1 and 2
are ``temp`` and ``salt``, ``_M1`` is the previous step (``valuesold``, fesom-jax's ``T_old`` / ``bgc_old``) and ``_AB``
the AB-extrapolated tracer, which fesom-jax recomputes.

What goes where (:class:`fesom_jax.state.State`):

* ``temp``, ``salt``, ``*_M1`` -> ``T``, ``S``, ``T_old``, ``S_old``; the BGC positions in the run config's layout order
  (``RecomConfig.layout.ids``, position = slot + 3) -> ``bgc``, ``bgc_old``;
* ``u``, ``v``, ``urhs_AB``, ``vrhs_AB`` -> ``uv``, ``uv_rhsAB``; ``w``, ``w_expl``, ``w_impl`` -> ``w``, ``w_e``, ``w_i``;
  ``ssh``, ``hbar``, ``ssh_rhs_old`` -> ``eta_n``, ``hbar`` (``hbar_old`` = ``hbar``: FESOM2 does not restart it),
  ``ssh_rhs_old``;
* ``area``, ``hice``, ``hsnow``, ``uice``, ``vice`` -> ``a_ice``, ``m_ice``, ``m_snow``, ``u_ice``, ``v_ice``;
* ``BenN``, ``BenC``, ``BenSi``, ``BenCalc`` -> ``bgc_benthos``; ``HPlus`` -> the pH first guess; ``Alk_surf`` (the
  alkalinity restoring target, not in the Fortran restart: REcoM sets it from the IC's surface alkalinity at setup)
  from the run config's IC set, as the cold start does (``--alk-surf restart`` takes the restart's surface Alk).

The layer thicknesses are NOT taken from the restart: fesom-jax has no partial cells, so ``hnode``/``helem`` are
rebuilt from ``hbar`` with its zstar rule (:func:`fesom_jax.ale.init_thickness_zstar`); a partial-cell bottom value
then fills the full nominal cell (a small inventory change at the bottom, stated). Cells the fesom-jax levels have
below the Fortran bottom (the level files of a partition differ from the repo's at a few nodes) take the column's
deepest Fortran value. Everything else (mixing coefficients, TKE, the carbonate cache, the counters) starts as a
fresh segment. The store is written for one device with the serial partition and reloads on any device count.

Run on a compute node (the CORE2 State is several GB)::

    python scripts/recom/fortran_restart_to_jax.py --restart /albedo/work/projects/p_specbic/sloza/restart_1997_4pft \\
        --year 1997 --config configs/albedo/core2_recom_spectral_warm1998.yaml --out <dir> --date 1998-doy001
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

from fesom_jax import ale, halo, partit, shard_mesh  # noqa: E402
from fesom_jax import integrate_sharded as ish  # noqa: E402
from fesom_jax import zarr_output as zo  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.recom import wiring  # noqa: E402
from fesom_jax.recom.ic import BgcIC  # noqa: E402
from fesom_jax.run_config import load_yaml  # noqa: E402
from fesom_jax.state import State  # noqa: E402

TRANSIENT_STEMS = {6: "sf6", 12: "cfc12", 14: "r14c", 39: "r39ar"}     # io_restart.F90's names for these positions


def tracer_stem(position: int) -> str:
    """The file stem of tracer ``position`` (1-based, namelist.tra order): ``temp``, ``salt``, ``tra00NN`` or the
    transient-tracer names FESOM2 gives positions 6, 12, 14 and 39."""
    if position == 1:
        return "temp"
    if position == 2:
        return "salt"
    return TRANSIENT_STEMS.get(position, f"tra{position:04d}")


def _read(path: Path) -> np.ndarray:
    with netCDF4.Dataset(path) as d:
        names = [v for v in d.variables if v not in ("time", "iter")]
        assert len(names) == 1, (path, names)
        v = d.variables[names[0]]
        v.set_auto_maskandscale(False)
        a = np.asarray(v[0], dtype=np.float64)                     # drop the record axis
        t, it = float(d.variables["time"][0]), int(d.variables["iter"][0])
    return a, (t, it)


def _fill_down(a: np.ndarray, src_wet: np.ndarray, dst_wet: np.ndarray) -> np.ndarray:
    """``a`` (n, nl) on the Fortran levels ``src_wet``: masked to ``dst_wet`` (the fesom-jax levels), with the cells
    ``dst_wet & ~src_wet`` (deeper than the Fortran bottom) filled by the column's deepest Fortran value."""
    n, nl = a.shape
    deepest = np.maximum(src_wet.sum(axis=1) - 1, 0)                 # (n,) index of the deepest Fortran wet layer
    fill = a[np.arange(n), deepest][:, None]
    out = np.where(src_wet, a, fill)
    return np.where(dst_wet, out, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--restart", required=True, type=Path, help="the directory holding fesom.<year>.{oce,ice,bio}.restart")
    ap.add_argument("--year", required=True, type=int, help="the restart's year (the file names)")
    ap.add_argument("--config", required=True, type=Path, help="the run YAML (mesh, REcoM namelist/layout, IC set, dt)")
    ap.add_argument("--out", required=True, type=Path, help="the fesom-jax restart store to write (new)")
    ap.add_argument("--date", default=None, help="calendar_date of the store, 'YYYY-doyDDD' (default: Jan 1 of year+1)")
    ap.add_argument("--step", type=int, default=0, help="the store's absolute step (default 0: the run starts there)")
    ap.add_argument("--alk-surf", choices=("ic", "restart"), default="ic",
                    help="Alk_surf from the config's IC set (REcoM's rule) or the restart's surface alkalinity")
    args = ap.parse_args()
    t0 = time.time()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    cfg = load_yaml(args.config)
    if cfg.recom is None or cfg.recom.mode != "full":
        raise SystemExit("the config needs a full REcoM block (recom.namelist)")
    mesh = load_mesh(cfg.mesh)
    lay = cfg.recom.layout
    nod, el, nl = int(mesh.nod2D), int(mesh.elem2D), int(mesh.nl)
    oce = args.restart / f"fesom.{args.year}.oce.restart"
    ice = args.restart / f"fesom.{args.year}.ice.restart"
    bio = args.restart / f"fesom.{args.year}.bio.restart"
    for d in (oce, ice, bio):
        if not d.is_dir():
            raise SystemExit(f"missing {d}")
    stamps = {}

    def rd(d, name):
        a, st = _read(d / f"{name}.nc")
        stamps[name] = st
        return a

    # --- levels: the Fortran restart's wet layers (hnode > 0) against the fesom-jax bundle's ---
    h_f = rd(oce, "hnode").T                                        # (nod, nl-1)
    src_wet = np.concatenate([h_f > 0.0, np.zeros((nod, 1), bool)], axis=1)        # (nod, nl) layers
    nlev = np.asarray(mesh.nlevels_nod2D)
    dst_wet = np.arange(nl)[None, :] < (nlev - 1)[:, None]
    nle = np.asarray(mesh.nlevels)
    dst_wet_e = np.arange(nl)[None, :] < (nle - 1)[:, None]
    assert h_f.shape == (nod, nl - 1), h_f.shape
    n_diff = int((src_wet.sum(1) != dst_wet.sum(1)).sum())
    n_fill = int((dst_wet & ~src_wet).sum())
    print(f"[convert] mesh {nod} nodes, {el} elements, {nl} levels; Fortran and fesom-jax level counts differ at "
          f"{n_diff} nodes ({n_fill} cells filled downward)", flush=True)

    def node_layers(name):                                       # (nz_1, node) -> (nod, nl), filled and masked
        a = np.zeros((nod, nl))
        a[:, : nl - 1] = rd(oce, name).T
        return _fill_down(a, src_wet, dst_wet)

    def elem_layers(name):                                       # (nz_1, elem) -> (elem, nl), masked (0 below)
        a = np.zeros((el, nl))
        a[:, : nl - 1] = rd(oce, name).T
        return np.where(dst_wet_e, a, 0.0)

    def node_interfaces(name):                                   # (nz, node) -> (nod, nl)
        a = rd(oce, name).T
        assert a.shape == (nod, nl), a.shape
        return np.where(np.arange(nl)[None, :] < nlev[:, None], a, 0.0)

    # --- the physics ---
    T, S = node_layers("temp"), node_layers("salt")
    T_old, S_old = node_layers("temp_M1"), node_layers("salt_M1")
    uv = np.stack([elem_layers("u"), elem_layers("v")], axis=-1)
    uv_rhsAB = np.stack([elem_layers("urhs_AB"), elem_layers("vrhs_AB")], axis=-1)
    w, w_e, w_i = node_interfaces("w"), node_interfaces("w_expl"), node_interfaces("w_impl")
    hbar, ssh, ssh_rhs_old = rd(oce, "hbar"), rd(oce, "ssh"), rd(oce, "ssh_rhs_old")
    a_ice, m_ice, m_snow = rd(ice, "area"), rd(ice, "hice"), rd(ice, "hsnow")
    u_ice, v_ice = rd(ice, "uice"), rd(ice, "vice")
    # --- the BGC tracers by position ---
    bgc = np.zeros((nod, nl - 1, lay.ntr))
    bgc_old = np.zeros((nod, nl - 1, lay.ntr))
    for s, tid in enumerate(lay.ids):
        stem = tracer_stem(s + 3)
        bgc[..., s] = _fill_down(np.pad(rd(oce, stem).T, ((0, 0), (0, 1))), src_wet, dst_wet)[:, : nl - 1]
        bgc_old[..., s] = _fill_down(np.pad(rd(oce, f"{stem}_M1").T, ((0, 0), (0, 1))), src_wet, dst_wet)[:, : nl - 1]
        print(f"[convert] slot {s:2d} id {tid} {lay.names[s]:>10} <- {stem}: max {np.max(np.abs(bgc[..., s])):.4e}",
              flush=True)
    benthos = np.stack([rd(bio, n) for n in ("BenN", "BenC", "BenSi", "BenCalc")], axis=-1)
    hplus = rd(bio, "HPlus")
    times = {v[0] for v in stamps.values()}
    iters = {v[1] for v in stamps.values()}
    print(f"[convert] restart time {sorted(times)} s, iter {sorted(iters)} ({len(stamps)} files)", flush=True)
    if len(times) != 1 or len(iters) != 1:
        raise SystemExit("the restart files do not share one time/iter stamp")
    if args.alk_surf == "ic":
        print("[convert] Alk_surf from the config's IC set (REcoM's rule at setup) ...", flush=True)
        alk_surf = np.asarray(wiring.cold_ic(mesh, cfg.recom).alk_surf)
    else:
        alk_surf = bgc[:, 0, lay.slot(1003)].copy()

    # --- the State: a fresh rest State (mixing, TKE, counters) with the restart's prognostic fields ---
    st = jax.tree.map(np.asarray, State.rest(mesh, ale_cfg=cfg.ale_cfg()))
    hnode, helem, eta_n, _ = (np.asarray(x) for x in ale.init_thickness_zstar(mesh, jnp.asarray(hbar), jnp.asarray(hbar)))
    print(f"[convert] zstar thickness from hbar: max |eta_n - ssh| {np.max(np.abs(eta_n - ssh)):.3e}; "
          f"Fortran hnode (partial cells) vs rebuilt: max |diff| {np.max(np.abs(h_f - hnode[:, : nl - 1])):.3f} m",
          flush=True)
    st = dataclasses.replace(
        st, T=T, S=S, T_old=T_old, S_old=S_old, uv=uv, uv_rhsAB=uv_rhsAB, w=w, w_e=w_e, w_i=w_i,
        eta_n=ssh, ssh_rhs_old=ssh_rhs_old, hbar=hbar, hbar_old=hbar.copy(), hnode=hnode, hnode_new=hnode.copy(),
        helem=helem, a_ice=a_ice, m_ice=m_ice, m_snow=m_snow, u_ice=u_ice, v_ice=v_ice)
    st = wiring.with_bgc_state(st, BgcIC(bgc=bgc, bgc_old=bgc_old, alk_surf=alk_surf, benthos=benthos, hplus=hplus),
                               cfg.recom)
    wet3 = dst_wet
    print(f"[convert] T [{T[wet3].min():.3f}, {T[wet3].max():.3f}] S [{S[wet3].min():.3f}, {S[wet3].max():.3f}] "
          f"ssh [{ssh.min():.3f}, {ssh.max():.3f}] a_ice max {a_ice.max():.3f} |uv| max {np.abs(uv).max():.3f} "
          f"Benthos max {benthos.max(axis=0)}", flush=True)
    assert np.all(S[wet3] > 0.0), "zero salinity in a wet cell"

    # --- write (one device, the serial partition; reloads on any device count) ---
    date = args.date or f"{args.year + 1}-doy001"
    ser = partit.synth_serial(nod, el, int(mesh.edge2D))
    sm = shard_mesh.build_sharded_mesh(mesh, ser)
    fs, spec = ish.folded_state(shard_mesh.partition_state(st, ser))
    placed = ish._to_global_sharded(fs, spec, halo.device_mesh("p", devices=jax.devices()[:1]))
    zo.write_restart(args.out, placed, sm, ser, step=int(args.step), calendar_date=date, dt_stage=float(cfg.dt))
    (args.out / "PROVENANCE.json").write_text(json.dumps(dict(
        source=str(args.restart.resolve()), year=args.year, config=str(args.config), mesh=str(cfg.mesh),
        layout_ids=list(map(int, lay.ids)), fortran_time=sorted(times)[0], fortran_iter=sorted(iters)[0],
        calendar_date=date, step=int(args.step), dt=float(cfg.dt), alk_surf=args.alk_surf,
        level_mismatch_nodes=n_diff, filled_cells=n_fill,
        note="hnode/helem rebuilt from hbar (no partial cells in fesom-jax); _AB fields not carried"), indent=1))
    print(f"[convert] wrote {args.out} ({date}, step {args.step}) in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
