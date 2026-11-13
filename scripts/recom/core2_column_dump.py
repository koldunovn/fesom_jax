#!/usr/bin/env python
"""Every-step column dump of the CORE2 REcoM forward around one node: what the state looks like where the reverse trace of
``scripts/recom/core2_bgc_adj.py`` puts a burst (plan ``docs/plans/20261105-recom-parameter-estimation.md``, "CORE2
horizon" and "CORE2 near-zero guard"; the Kuroshio dump 49015775 was the first of these).

    python scripts/recom/core2_column_dump.py --config CONFIG --year 1998 --restart RESTART --dist-dir DIR \
        --node 88529 --radius-deg 1.5 --days 30 --out OUT

The forward is the gradient driver's: the same sharded step (``adjoint_sharded.build``, padded halo), the same config,
restart, forcing and physics kwargs, the default parameters. Only the forward runs, one step per chunk, so the state is
seen after every step. The near-zero guard (``FESOM_RECOM_GRAD_FREEZE_*``) acts on the reverse pass only and does not
change this forward. The chunking (1 step, no accumulators) differs from the gradient run's (2 steps), so values at
round-off (1e-16) may differ in their last bits; the occupancy of a cell does not.

Output OUT/columns.npz: ``nodes`` (global ids, the given nodes first), ``lon``/``lat`` [deg], ``depth`` [m], ``nlev``
(``nlevels_nod2D``), ``site`` (which given node a column
surrounds), ``dist_deg``, ``Z`` (mid-layer depths), ``tracers`` (names), ``step``, ``day``; per step
``bgc [n_steps, n_nodes, nl, n_tracers]``, ``T``, ``S``, ``speed`` (|uvnode|), ``Kv``, ``w``, ``hnode`` ``[n_steps, n_nodes,
nl]``, ``a_ice`` ``[n_steps, n_nodes]``; step 0 is the restart state. Written again at the end of every model day, so a
time-out keeps the days done.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fesom_jax  # noqa: E402,F401  (float64 on)
from fesom_jax import adjoint, adjoint_sharded, partit, ssh, surface_forcing  # noqa: E402
from fesom_jax import zarr_output as zo  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.params import Params  # noqa: E402
from fesom_jax.recom import wiring  # noqa: E402
from fesom_jax.run_config import load_yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core2_bgc_adj import dates, memory  # noqa: E402


WATCH = ("DIN", "DSi", "DiaN", "DiaC", "DetSi", "DetZ2N")    # printed once a model day


def great_circle_deg(lon0, lat0, lon, lat):
    """Angular distance [deg] from (lon0, lat0) to arrays (lon, lat), all in radians."""
    c = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--year", type=int, default=1998, help="the start year of the run --restart comes from")
    ap.add_argument("--restart", required=True)
    ap.add_argument("--dist-dir", required=True, help="the mesh directory holding dist_<npes>")
    ap.add_argument("--npes", type=int, default=4)
    ap.add_argument("--transport", default="padded", choices=adjoint_sharded.TRANSPORTS)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--node", type=int, nargs="+", required=True,
                    help="global node id(s), 0-based as the reverse trace prints them; the first is the centre")
    ap.add_argument("--radius-deg", type=float, default=1.5, help="also dump every node within this distance of the centre")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    cfg = load_yaml(a.config)
    dt = float(cfg.dt)
    spd = int(round(86400 / dt))
    n = a.days * spd
    if len(jax.devices()) < a.npes:
        raise SystemExit(f"{a.npes} devices wanted, {len(jax.devices())} visible")

    mesh = load_mesh(cfg.mesh)
    part = partit.read_partition(a.dist_dir, a.npes)
    st, meta = zo.read_restart(a.restart, mesh, partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D))
    st = wiring.start_segment(jax.tree.map(lambda x: np.asarray(x), st), cfg.recom)
    start = datetime.datetime(a.year, 1, 1) + datetime.timedelta(seconds=int(meta["step"]) * dt)
    cf = surface_forcing.build_surface_forcing(mesh, a.year, sst_ic=np.asarray(st.T[:, 0]), dt=dt,
                                               mslp=cfg.forcing_mslp(), bgc_dust_path=cfg.bgc_dust_path(),
                                               dt_ramp=cfg.dt_ramp, bgc_co2=cfg.bgc_co2(), bgc_oasim=cfg.bgc_oasim(),
                                               **cfg.forcing_paths())
    fs = wiring.with_recom_static(cf.static, mesh, cfg.recom)
    op = ssh.build_ssh_operator(mesh, dt=dt)
    phys = dict(cfg.physics_kwargs())
    names = list(cfg.recom.layout.names)

    # the columns: the given nodes, then around each the other nodes within the radius, nearest first; ``site`` is the
    # index of the given node a column belongs to, ``dist_deg`` its distance from that node
    geo = np.asarray(mesh.geo_coord_nod2D)
    lon, lat = geo[:, 0], geo[:, 1]
    nodes, site, dist = list(a.node), list(range(len(a.node))), [0.0] * len(a.node)
    for j, c in enumerate(a.node):
        dc = great_circle_deg(lon[c], lat[c], lon, lat)
        for i in np.argsort(dc):
            if dc[i] > a.radius_deg:
                break
            if int(i) not in nodes:
                nodes.append(int(i)); site.append(j); dist.append(float(dc[i]))
        print(f"site {j}: node {c} at {np.degrees(lon[c]):.2f} E {np.degrees(lat[c]):.2f} N, depth "
              f"{float(mesh.depth[c]):.0f} m, nlevels {int(mesh.nlevels_nod2D[c])}, {site.count(j)} columns within "
              f"{a.radius_deg} deg", flush=True)
    nodes, site, dist = np.array(nodes, np.int64), np.array(site, np.int32), np.array(dist)
    depth = np.asarray(mesh.depth)
    print(f"start {start.isoformat()}, {a.days} d = {n} steps, {nodes.size} columns", flush=True)

    t = time.time()
    chunk = adjoint_sharded.build(mesh, part, op, fs, phys=phys, transport=a.transport)
    print(f"sharded layout: Lmax {chunk.sm.Lmax}, {time.time() - t:.0f} s", flush=True)
    _, p0 = cfg.recom.load()
    params = dataclasses.replace(Params.defaults(), recom=p0)
    chunk_fn = chunk.make_chunk_fn(dt=dt, phys=phys, adjoint_cfg=None, to_params=lambda u: params,
                                   rematerialize=False)
    fwd = adjoint._jitted_forward(chunk_fn)

    # each column's owned lane in the folded carry
    gid, owned = np.asarray(chunk.gid["nod"]), np.asarray(chunk.owned["nod"])
    lane_of = {int(g): i for i, g in enumerate(gid) if owned[i]}
    lanes = jnp.asarray([lane_of[int(g)] for g in nodes], jnp.int32)

    @jax.jit
    def take(s):
        uvn = s.uvnode[lanes]
        return dict(bgc=s.bgc[lanes], T=s.T[lanes], S=s.S[lanes], Kv=s.Kv[lanes], w=s.w[lanes], hnode=s.hnode[lanes],
                    speed=jnp.sqrt(uvn[..., 0] ** 2 + uvn[..., 1] ** 2),
                    a_ice=s.a_ice[lanes] if s.a_ice is not None else jnp.zeros(lanes.shape))

    rec = {k: [] for k in ("bgc", "T", "S", "Kv", "w", "hnode", "speed", "a_ice")}

    def keep(s):
        for k, v in jax.device_get(take(s)).items():
            rec[k].append(np.asarray(v))

    def save():
        k_done = len(rec["bgc"])
        np.savez(a.out / "columns.npz", nodes=nodes, lon=np.degrees(lon[nodes]), lat=np.degrees(lat[nodes]),
                 depth=depth[nodes], nlev=np.asarray(mesh.nlevels_nod2D)[nodes], site=site, dist_deg=dist,
                 Z=np.asarray(mesh.Z), tracers=np.array(names), start=start.isoformat(),
                 step=np.arange(k_done), day=np.arange(k_done) / spd,
                 **{k: np.stack(v) for k, v in rec.items()})

    s = chunk.place(chunk.fold_state(st))
    keep(s)
    u0 = {}
    t = time.time()
    for k in range(n):
        xs = chunk.forcing_xs(cf.stack(dates(start, dt, k, 1), xp=np))
        nxt = jax.block_until_ready(fwd(s, xs, u0))
        adjoint._free(xs[0])
        adjoint._free(s)
        s = nxt
        keep(s)
        if (k + 1) % spd == 0:
            save()
            msg = []
            for j, c in enumerate(a.node):                       # each given node, top 3 levels
                b = rec["bgc"][-1][j]
                msg.append(f"{c}: " + " ".join(f"{nm} {b[:3, names.index(nm)].min():.1e}..{b[:3, names.index(nm)].max():.1e}"
                                               for nm in WATCH if nm in names))
            print(f"   day {(k + 1) // spd:3d} ({time.time() - t:.0f} s): " + " | ".join(msg), flush=True)
    save()
    print(f"done in {time.time() - t_all:.0f} s; {n} steps, device peaks {memory()} GiB -> {a.out / 'columns.npz'}",
          flush=True)


if __name__ == "__main__":
    main()
