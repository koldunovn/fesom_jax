#!/usr/bin/env python
"""Replay rung for the cavity melt: fesom_jax.cavity vs the FESOM2 main SP oracle, step by step.

    python scripts/core3/replay_cavity_melt.py DUMP_DIR MESH_JAX [--steps 1 2] [--gamma-scale 0.6]

Feeds :func:`fesom_jax.cavity.cavity_heat_water_fluxes_3eq` the oracle's own inputs (the ``pre_fluxes`` full dumps:
T, S, uvnode, density, density_ref, hnode, Z_3d_n of the same step) and compares its heat/water flux at the cavity
nodes with the oracle's ``oce_fluxes`` dump (after ``oce_fluxes``: at cavity nodes these are exactly the 3-equation
fluxes, FESOM adds the freshwater balance only off the cavities). Runs twice: in single precision (the oracle's
build; expect float32 ulps) and in double precision (the same inputs; expect float32-sized differences, the
oracle's own rounding). Also checks ``cavity_momentum_fluxes`` against the oracle's ``stress_surf`` /
``stress_node_surf`` at cavity elements/nodes and that the oracle has no sea ice under the shelves.
Prints ``CAVITY_MELT_REPLAY_OK`` when the single-precision replay is within 1e-6 of the largest |flux| at every cavity
node.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fesom_jax import cavity, io_dump, precision  # noqa: E402


class _M:  # the two mesh arrays the kernels read
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("mesh_dir")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--gamma-scale", type=float, default=0.6)
    a = ap.parse_args()
    import jax.numpy as jnp

    mdir = Path(a.mesh_dir)
    uln = np.load(mdir / "ulevels_nod2D.npy"); ule = np.load(mdir / "ulevels.npy")
    N, E = uln.size, ule.size
    cav = uln > 1; cave = ule > 1
    ok = True
    for step in a.steps:
        F = lambda sub, name, n=N: io_dump.load_full_field(a.dump_dir, step, sub, name, n)
        inp = {k: F(20, k) for k in ("T", "S", "uvnode_1", "uvnode_2", "density", "density_ref", "hnode", "Z_3d_n")}
        uv_e = np.stack([F(20, "uv_1", E), F(20, "uv_2", E)], axis=-1)
        hf_o = F(19, "heat_flux")[:, 0]; wf_o = F(19, "water_flux")[:, 0]
        ss_o = np.stack([F(19, "stress_surf_x", E)[:, 0], F(19, "stress_surf_y", E)[:, 0]], -1)
        sn_o = np.stack([F(19, "stress_node_surf_x")[:, 0], F(19, "stress_node_surf_y")[:, 0]], -1)
        for prec in ("single", "double"):
            with precision.scope(prec):
                wp = precision.wp_dtype()
                m = _M(); m.ulevels_nod2D = jnp.asarray(uln); m.ulevels = jnp.asarray(ule)
                X = {k: jnp.asarray(v, wp) for k, v in inp.items()}
                X["hnode"] = jnp.asarray(inp["hnode"], jnp.float64)            # WP_full in both builds
                uvn = jnp.stack([X["uvnode_1"], X["uvnode_2"]], axis=-1)
                rhow = X["density"] + X["density_ref"]                          # WP sum, as Fortran
                zero = jnp.zeros(N, wp)
                hf, wf, _ = cavity.cavity_heat_water_fluxes_3eq(
                    m, X["T"], X["S"], uvn, X["hnode"], X["Z_3d_n"], rhow, zero, zero, a.gamma_scale)
                ss, sn = cavity.cavity_momentum_fluxes(m, jnp.asarray(uv_e, wp), uvn, jnp.zeros((E, 2), wp),
                                                       jnp.zeros((N, 2), wp), 1030.0, 0.0025)
                hf, wf, ss, sn = (np.asarray(x, np.float64) for x in (hf, wf, ss, sn))

            def rel(x, ref, mask):
                # max |diff| scaled by the field's max |value| over the cavity (pointwise relative errors blow up
                # where the melt changes sign: tin - tf cancels), and the share of bitwise-equal values
                d = np.abs(x[mask] - ref[mask])
                return float(d.max() / np.abs(ref[mask]).max()), float(np.mean(d == 0)), float(d.max())

            r_hf, r_wf = rel(hf, hf_o, cav), rel(wf, wf_o, cav)
            r_ss = rel(ss, ss_o, np.repeat(cave[:, None], 2, 1)); r_sn = rel(sn, sn_o, np.repeat(cav[:, None], 2, 1))
            print(f"step {step} {prec:6s}: heat_flux max {r_hf[0]:.2e} (abs {r_hf[2]:.1e} W/m2) bitwise {r_hf[1]:.3f} | "
                  f"water_flux max {r_wf[0]:.2e} bitwise {r_wf[1]:.3f} | stress_surf max {r_ss[0]:.2e} "
                  f"bitwise {r_ss[1]:.3f} | stress_node_surf max {r_sn[0]:.2e} bitwise {r_sn[1]:.3f}")
            if prec == "single":
                ok &= max(r_hf[0], r_wf[0]) < 1e-6
        for name in ("a_ice", "m_ice", "m_snow", "uice", "vice"):
            v = F(18, name)[:, 0]
            print(f"   ice under shelves: max|{name}| = {np.abs(v[cav]).max():.1e}")
            ok &= bool(np.abs(v[cav]).max() == 0.0)
        area = np.load(mdir / "areasvol.npy")[np.arange(N), uln - 1]
        print(f"   oracle basal melt step {step}: {-(wf_o[cav] * area[cav]).sum() * 1e3 * 86400 * 365 / 1e12:.1f} Gt/yr")
    print("CAVITY_MELT_REPLAY_OK" if ok else "CAVITY_MELT_REPLAY_FAIL")


if __name__ == "__main__":
    main()
