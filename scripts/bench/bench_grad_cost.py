#!/usr/bin/env python
"""bench_grad_cost.py — paper §5: the cost of a gradient (forward vs reverse-mode).

Times the complete CORE2 model (ice+KPP+GM, real per-step JRA forcing, PHC IC — the
same configuration as bench_forward_scaling's FULL) through the dense single-device
:func:`fesom_jax.integrate.integrate` (the production adjoint path is single-device):

  forward :  jit(loss)                     loss = area-mean(SST_N^2) after N steps
  reverse :  jit(value_and_grad(loss))     gradient wrt the full initial T field

Compile-once protocol (2nd call timed; compile reported separately). One FORWARD and
ONE gradient variant per process invocation, so `memory_stats` cumulative peaks are
attributable (`--seg 0` = per-step checkpointing, `--seg -1` = O(sqrt N) two-level).

    python scripts/bench/bench_grad_cost.py --mesh-dir data/mesh_core2 --ic-dir data/ic_core2 \
        --n 48 --seg 0
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
import fesom_jax  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp


def peak_gb():
    try:
        s = jax.local_devices()[0].memory_stats() or {}
        return s.get("peak_bytes_in_use", -1) / 2**30
    except Exception:
        return -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-dir", required=True)
    ap.add_argument("--ic-dir", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--dt", type=float, default=1800.0)
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--seg", type=int, default=0,
                    help="remat_segments: 0 = per-step checkpointing, -1 = O(sqrt N)")
    ap.add_argument("--remat-blocks", type=int, default=1,
                    help="nested in-step jax.checkpoint (the production adjoint setting)")
    ap.add_argument("--ice", type=int, default=1)
    ap.add_argument("--kpp", type=int, default=1)
    ap.add_argument("--gm", type=int, default=1)
    args = ap.parse_args()

    from fesom_jax import surface_forcing, ice, ssh
    from fesom_jax.gm import GMConfig
    from fesom_jax.ice import IceConfig
    from fesom_jax.integrate import integrate
    from fesom_jax.kpp import KppConfig
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import core2_initial_state

    mesh = load_mesh(args.mesh_dir)
    st0 = core2_initial_state(mesh, args.ic_dir)
    sst0 = np.asarray(st0.T[:, 0])
    if args.ice:
        st0 = ice.seed_ice(st0, mesh, sst0)
    op = ssh.build_ssh_operator(mesh, dt=args.dt)
    cf = surface_forcing.build_surface_forcing(mesh, args.year, sst_ic=sst0)
    sfs = cf.stack(surface_forcing.dates_for_steps(args.year, args.dt, args.n))
    fs = cf.static
    cfgs = dict(ice_cfg=IceConfig() if args.ice else None,
                kpp_cfg=KppConfig() if args.kpp else None,
                gm_cfg=GMConfig() if args.gm else None)
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
    nwet = jnp.sum(wet0)
    plat = jax.devices()[0].platform
    print(f"[gradbench] setup done: nod2D={mesh.nod2D} n={args.n} seg={args.seg} "
          f"plat={plat}", flush=True)

    def loss(T0):
        s = dataclasses.replace(st0, T=T0)
        fin = integrate(s, mesh, op, None, n_steps=args.n, dt=args.dt,
                        step_forcings=sfs, forcing_static=fs,
                        checkpoint=True, remat_segments=args.seg,
                        remat_blocks=bool(args.remat_blocks), **cfgs)
        sst = jnp.where(wet0, fin.T[:, 0], 0.0)
        return jnp.sum(sst * sst) / nwet

    T0 = jnp.asarray(st0.T)

    def timed(fn, tag):
        t = time.perf_counter(); jax.block_until_ready(fn(T0)); comp = time.perf_counter() - t
        t = time.perf_counter(); jax.block_until_ready(fn(T0)); run = time.perf_counter() - t
        print(f"[gradbench-part] {tag}: run={run:.2f}s compile={comp:.1f}s "
              f"peak={peak_gb():.2f}GiB", flush=True)
        return run, comp

    fwd_run, fwd_comp = timed(jax.jit(loss), "forward")
    fwd_peak = peak_gb()
    fwd_ms = fwd_run / args.n * 1e3

    g_run, g_comp = timed(jax.jit(jax.value_and_grad(loss)), f"value_and_grad seg={args.seg}")
    g_peak = peak_gb()
    g_ms = g_run / args.n * 1e3

    print(f"[gradbench] mesh=core2 nod2D={mesh.nod2D} n={args.n} seg={args.seg} "
          f"fwd_ms_step={fwd_ms:8.2f} grad_ms_step={g_ms:8.2f} ratio={g_ms / fwd_ms:5.2f} "
          f"compile_fwd={fwd_comp:6.1f}s compile_grad={g_comp:6.1f}s "
          f"peak_fwd={fwd_peak:.2f}GiB peak_grad={g_peak:.2f}GiB plat={plat}", flush=True)


if __name__ == "__main__":
    main()
