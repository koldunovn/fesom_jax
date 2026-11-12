#!/usr/bin/env python
"""The REcoM gradient on a mesh too large for one GPU (CORE2): sharded over the devices of one node, one chunk of the
window on the devices and the chunk boundaries on the host (plan ``docs/plans/20261105-recom-parameter-estimation.md``:
a 1-day CORE2 window needs 89-187 GiB, more than one A100 80GB).

    python scripts/recom/core2_bgc_adj.py --config CONFIG --year 1998 --restart RESTART --dist-dir DIR \
        --days 1 --chunk-steps 8 --out OUT

It joins three existing pieces: :func:`fesom_jax.adjoint_sharded.build` (the step under ``shard_map`` over ``--npes``
devices, padded halo, AD-correct), :func:`fesom_jax.adjoint.chunked_grad_multi` (one forward, boundaries on the host,
one reverse walk per cost) and the costs of ``scripts/recom/bgc_gradient_horizon.py``: the area-weighted surface mean
of total chlorophyll (the four Chl tracers) and of DIN over the last day of the window, as per-node running sums with a
per-step weight (1/32 on the last day's steps, 0 before). The controls are log multipliers of the six REcoM
parameters of the horizon driver (``dJ/dlog p`` directly); the gradient with respect to the initial BGC state comes
from the cost's state cotangent. The check is the horizon driver's: a central difference along ``sign(dJ/dlog p)``
against the adjoint's ``sum |dJ/dlog p|``, both costs from the same pair of forward runs.

Output: one JSON line per cost in OUT/core2_bgc_adj.jsonl, and the log. Same fields as the horizon driver where they
mean the same.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
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

NAMES = ("expon_phy", "graz_max", "rho_N", "deg_Chl", "cdomcoeff", "k_Fe_d")    # as bgc_gradient_horizon.py
CHL = ("PhyChl", "DiaChl", "CoccoChl", "PhaeoChl")
COSTS = ("chl", "din")


def dates(start, dt, n0, n):
    out = []
    for k in range(n0, n0 + n):
        d = start + datetime.timedelta(seconds=k * float(dt))
        out.append((d.year, (d - datetime.datetime(d.year, 1, 1)).days + 1,
                    d.hour * 3600.0 + d.minute * 60.0 + d.second, d.month))
    return out


def memory():
    out = []
    for d in jax.local_devices():
        try:
            m = d.memory_stats() or {}
            out.append(round(m.get("peak_bytes_in_use", 0) / 2**30, 1))
        except Exception:
            out.append(None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--year", type=int, default=1998, help="the start year of the run --restart comes from")
    ap.add_argument("--restart", required=True)
    ap.add_argument("--dist-dir", required=True, help="the mesh directory holding dist_<npes>")
    ap.add_argument("--npes", type=int, default=4)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--chunk-steps", type=int, default=8, help="steps per chunk (32 a day); divides the window")
    ap.add_argument("--boundary-stride", type=int, default=1,
                    help="keep every k-th chunk boundary on the host and rebuild the others in the reverse (about 6.4 GB "
                         "per CORE2 boundary: a 15-day window of 2-step chunks needs k = 8 to stay near 200 GB)")
    ap.add_argument("--transport", default="padded", choices=adjoint_sharded.TRANSPORTS)
    ap.add_argument("--remat-blocks", action="store_true", help="step(remat_blocks=True), as the horizon driver")
    ap.add_argument("--h", type=float, default=1e-2, help="the FD step in log p")
    ap.add_argument("--no-fd", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    cfg = load_yaml(a.config)
    dt = float(cfg.dt)
    spd = int(round(86400 / dt))
    n = a.days * spd
    K = a.chunk_steps
    if n % K:
        raise SystemExit(f"--chunk-steps {K} does not divide the window of {n} steps")
    n_chunks = n // K
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
    if a.remat_blocks:
        phys["remat_blocks"] = True
    lay = cfg.recom.layout
    chl_ids = jnp.array([lay.names.index(c) for c in CHL])
    din_id = lay.names.index("DIN")
    print(f"start {start.isoformat()}, {a.days} d = {n} steps in {n_chunks} chunks of {K}, {a.npes} devices "
          f"{[d.device_kind for d in jax.devices()[:a.npes]]}, bgc {st.bgc.shape}", flush=True)

    t = time.time()
    chunk = adjoint_sharded.build(mesh, part, op, fs, phys=phys, transport=a.transport)
    print(f"sharded layout: Lmax {chunk.sm.Lmax}, {time.time() - t:.0f} s", flush=True)
    _, p0 = cfg.recom.load()

    def to_params(u):
        return dataclasses.replace(Params.defaults(), recom=p0._replace(
            **{k: getattr(p0, k) * jnp.exp(u[k]) for k in NAMES}))

    accs = {"chl": lambda fm: (lambda s: jnp.sum(s.bgc[:, 0, chl_ids], axis=-1)),
            "din": lambda fm: (lambda s: s.bgc[:, 0, din_id])}
    chunk_fn = chunk.make_chunk_fn(dt=dt, phys=phys, adjoint_cfg=None, to_params=to_params, accumulators=accs,
                                   rematerialize=True)
    area = np.asarray(mesh.areasvol)[:, 0]
    a_f = chunk.place(chunk.fold_nodes(area))                     # zero on halo and pad lanes: each node once
    A = float(area.sum())
    loss_fns = [lambda c, k=k: jnp.sum(c[1][k] * a_f) / A for k in COSTS]
    w_all = np.where(np.arange(n) >= n - spd, 1.0 / spd, 0.0)

    clock = {"last": None, "build": 0.0, "gap": 0.0, "calls": 0}

    def xs_fn(c):                                                # fresh on every call (the library frees it)
        t0 = time.time()
        if clock["last"] is not None:                            # the time since the previous chunk's inputs were
            clock["gap"] += t0 - clock["last"]                   # handed over: the device work of that chunk
        sf = cf.stack(dates(start, dt, c * K, K), xp=np)
        sched = {"month": np.zeros(K, np.int32), "cost_w": w_all[c * K:(c + 1) * K].astype(np.float64)}
        out = chunk.forcing_xs(sf, sched)
        t1 = time.time()
        clock["build"] += t1 - t0
        clock["calls"] += 1
        clock["last"] = t1
        if clock["calls"] % 16 == 1 or n_chunks <= 32:
            print(f"   xs chunk {c}: built in {t1 - t0:.1f} s (calls {clock['calls']}, inputs {clock['build']:.0f} s, "
                  f"between calls {clock['gap']:.0f} s)", flush=True)
        return out

    host_carry = (chunk.fold_state(st), adjoint_sharded.init_accumulators(chunk, {k: (mesh.nod2D,) for k in COSTS}))
    carry0 = lambda: chunk.place(host_carry)                     # noqa: E731  (a fresh device copy per use)
    u0 = {k: jnp.asarray(0.0, jnp.float64) for k in NAMES}

    # The reverse trace: at every chunk boundary, per cost, the norm of the BGC cotangent (raw and times the state
    # there) and where its largest entry sits (global node, level, tracer, the tracer's value). The boundary states
    # come from the forward's host copies; the cotangent is read on the device (read-only, the library's on_cot).
    gid = np.asarray(chunk.gid["nod"])
    names = list(lay.names)
    trace_path = a.out / f"trace_{start.strftime('%Y%m%d')}_{a.days}d.jsonl"

    @jax.jit
    def _peak(b):
        a_ = jnp.abs(b)
        i = jnp.argmax(a_)
        return jnp.sqrt(jnp.sum(b * b)), i, a_.reshape(-1)[i], jnp.sqrt(jnp.sum(b * b, axis=(0, 1)))

    def on_cot(k, c, cot):
        b = cot[0].bgc if isinstance(cot, tuple) else cot.bgc
        norm, i, peak, per_tr = jax.device_get(_peak(b))
        lane, lev, tr = np.unravel_index(int(i), b.shape)
        top = np.argsort(-per_tr)[:3]
        rec = dict(cost=COSTS[k], boundary=int(c), day=c * K / spd, norm=float(norm), peak=float(peak),
                   node=int(gid[lane]), level=int(lev), tracer=names[tr],
                   top_tracers=[(names[t], float(per_tr[t])) for t in top])
        with open(trace_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if c % (n_chunks // 8 or 1) == 0 or c == n_chunks:
            print(f"   trace {COSTS[k]} boundary {c:4d} (day {rec['day']:5.2f}): |cot bgc| {norm:.3e}  peak {peak:.2e} "
                  f"at node {rec['node']} level {lev} {names[tr]}", flush=True)

    t = time.time()
    res = adjoint.chunked_grad_multi(chunk_fn, u0, init_state=carry0(), n_chunks=n_chunks, xs_fn=xs_fn,
                                     loss_fns=loss_fns, chunk_steps=K, boundary_stride=a.boundary_stride,
                                     check_linear=False, field_trace=False, on_cot=on_cot)
    t_grad = time.time() - t
    print(f"gradient: forward {res.forward_seconds:.0f} s, reverse {res.reverse_seconds:.0f} s, host "
          f"{res.host_gb:.1f} GB, device peaks {memory()} GiB", flush=True)

    x0 = np.asarray(st.bgc)
    rows = []
    for k, name in enumerate(COSTS):
        g = {p: float(np.asarray(res.grads[k][p])) for p in NAMES}
        cot = res.state_cots[k]
        gx = chunk.unfold_state(cot[0]).bgc if cot is not None else None
        row = dict(cost=name, start=start.isoformat(), days=a.days, J=float(res.losses[k]), dJ_dlogp=g,
                   x0_norm=None if gx is None else float(np.linalg.norm(gx)),
                   x0_scaled_norm=None if gx is None else float(np.linalg.norm(x0 * gx)),
                   npes=a.npes, chunk_steps=K, boundary_stride=a.boundary_stride, t_grad=t_grad, host_gb=res.host_gb, device_peak_gib=memory())
        print(f"{a.days} d {name}: J {row['J']:.7e}  |dJ/dx0| {row['x0_norm']}  scaled {row['x0_scaled_norm']}", flush=True)
        for p in NAMES:
            print(f"   {p:10s} dJ/dlog p {g[p]: .6e}", flush=True)
        rows.append(row)

    if not a.no_fd:                                              # central differences along sign(g), per cost
        for k, (name, row) in enumerate(zip(COSTS, rows)):
            d = {p: float(np.sign(row["dJ_dlogp"][p])) for p in NAMES}
            J = []
            for s in (+1, -1):
                u = {p: jnp.asarray(s * a.h * d[p], jnp.float64) for p in NAMES}
                t = time.time()
                fin = adjoint.chunked_forward(chunk_fn, u, init_state=carry0(), n_chunks=n_chunks, xs_fn=xs_fn)
                J.append(float(loss_fns[k](fin)))
                adjoint._free(fin)
                print(f"   FD {name} {'+' if s > 0 else '-'}h: J {J[-1]:.9e} ({time.time() - t:.0f} s)", flush=True)
            fd = (J[0] - J[1]) / (2 * a.h)
            ad = sum(abs(v) for v in row["dJ_dlogp"].values())
            row.update(fd_dir=fd, adjoint_dir=ad, h=a.h)
            print(f"{name}: along sign(g): adjoint {ad:.6e}  FD {fd:.6e}  ratio {ad / fd if fd else float('nan'):.4f}",
                  flush=True)

    with open(a.out / "core2_bgc_adj.jsonl", "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"done in {time.time() - t_all:.0f} s; device peaks {memory()} GiB", flush=True)


if __name__ == "__main__":
    main()
