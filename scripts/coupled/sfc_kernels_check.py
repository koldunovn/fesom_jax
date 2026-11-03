#!/usr/bin/env python
"""Lane L-SFC (plan ~/awicm_jax/docs/plans/20261103-jaxcpl-core3.md): the new FESOM main 0aaa2d73 switches against
loop transcriptions of the Fortran, on the pi mesh, double precision, CPU. One check per kernel, no test zoo:

* visc_filt_bcksct (opt_visc 5, oce_dyn.F90:318-460): edge loop, node smoothing over nod_in_elem2D, back to elements;
* the hemispheric GM ceiling (oce_fer_gm.F90:295-306): fer_K at the top level;
* the coupled ice albedo with alb_tramp / h_snowscale (ice_thermo_cpl.F90 ice_albedo).

    python scripts/coupled/sfc_kernels_check.py
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from fesom_jax import gm, momentum
from fesom_jax.config import RAD, ViscConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params


def bcksct_fortran(mesh, uv, uv_rhs, dt, vc):
    """oce_dyn.F90 visc_filt_bcksct (use_ssh_se_subcycl false), loop by loop (Fortran 1-based levels)."""
    et = np.asarray(mesh.edge_tri)
    area = np.asarray(mesh.elem_area, np.float64)
    ulev, nlev = np.asarray(mesh.ulevels), np.asarray(mesh.nlevels)
    en = np.asarray(mesh.elem_nodes)
    nE, nl = area.size, int(mesh.nl)
    U_b = np.zeros((nE, nl))
    V_b = np.zeros((nE, nl))
    for ed in range(et.shape[0]):
        e1, e2 = int(et[ed, 0]), int(et[ed, 1])
        if e1 < 0 or e2 < 0:
            continue                                   # myList_edge2D(ed) > edge2D_in: boundary edge
        ln = np.sqrt(area[e1] + area[e2])
        nzmax, nzmin = min(nlev[e1], nlev[e2]), max(ulev[e1], ulev[e2])
        for nz in range(nzmin - 1, nzmax - 1):
            u1 = uv[e1, nz, 0] - uv[e2, nz, 0]
            v1 = uv[e1, nz, 1] - uv[e2, nz, 1]
            vi = dt * max(vc.gamma0, max(vc.gamma1 * np.sqrt(u1 * u1 + v1 * v1), vc.gamma2 * (u1 * u1 + v1 * v1))) * ln
            U_b[e1, nz] -= u1 * vi / area[e1]
            V_b[e1, nz] -= v1 * vi / area[e1]
            U_b[e2, nz] += u1 * vi / area[e2]
            V_b[e2, nz] += v1 * vi / area[e2]
    nN = int(mesh.nod2D)
    nod_in = [[] for _ in range(nN)]
    for e in range(nE):
        for k in range(3):
            nod_in[en[e, k]].append(e)
    ulev_n, nlev_n = np.asarray(mesh.ulevels_nod2D), np.asarray(mesh.nlevels_nod2D)
    U_c = np.zeros((nN, nl))
    V_c = np.zeros((nN, nl))
    for n in range(nN):
        els = nod_in[n]
        for nz in range(ulev_n[n] - 1, nlev_n[n] - 1):
            vi = sum(area[e] for e in els)
            U_c[n, nz] = sum(U_b[e, nz] * area[e] for e in els) / vi
            V_c[n, nz] = sum(V_b[e, nz] * area[e] for e in els) / vi
    out = np.array(uv_rhs, np.float64)
    for e in range(nE):
        for nz in range(ulev[e] - 1, nlev[e] - 1):
            out[e, nz, 0] += U_b[e, nz] - vc.easybsreturn * U_c[en[e], nz].sum() / 3.0
            out[e, nz, 1] += V_b[e, nz] - vc.easybsreturn * V_c[en[e], nz].sum() / 3.0
    return out


def main():
    mesh = load_mesh()
    rng = np.random.default_rng(1)
    nE, nl = int(mesh.elem2D), int(mesh.nl)
    lm = np.asarray(mesh.elem_layer_mask)[..., None]
    uv = np.where(lm, rng.normal(0.0, 0.3, (nE, nl, 2)), 0.0)
    uv_rhs = np.where(lm, rng.normal(0.0, 1e-3, (nE, nl, 2)), 0.0)
    dt = 1800.0
    ok = True

    # --- visc_filt_bcksct ---
    vc = ViscConfig(opt_visc=5, easybsreturn=1.5)
    got = np.asarray(momentum.horizontal_viscosity(mesh, jnp.asarray(uv), jnp.asarray(uv_rhs), dt=dt, visc_cfg=vc))
    ref = bcksct_fortran(mesh, uv, uv_rhs, dt, vc)
    inc = ref - uv_rhs
    rel = np.abs(got - ref).max() / max(np.abs(inc).max(), 1e-300)
    print(f"[bcksct] max|JAX-Fortran loops| / max|viscous term| = {rel:.2e} (max term {np.abs(inc).max():.3e})")
    ok &= rel < 1e-12
    d7 = np.asarray(momentum.horizontal_viscosity(mesh, jnp.asarray(uv), jnp.asarray(uv_rhs), dt=dt, visc_cfg=None))
    d7b = np.asarray(momentum.visc_filt_bidiff(mesh, jnp.asarray(uv), jnp.asarray(uv_rhs), dt=dt))
    print(f"[visc] opt_visc 7 through the dispatcher bitwise visc_filt_bidiff: {np.array_equal(d7, d7b)}")
    ok &= bool(np.array_equal(d7, d7b))

    # --- hemispheric GM ceiling: fer_K at the top level ---
    N = int(mesh.nod2D)
    bv = np.where(np.asarray(mesh.node_layer_mask), np.abs(rng.normal(1e-5, 5e-6, (N, nl))), 0.0)
    zb = np.asarray(mesh.zbar, np.float64)
    hn = np.where(np.asarray(mesh.node_layer_mask), np.append(zb[:-1] - zb[1:], 0.0)[None, :], 0.0)
    tap = np.ones((N, nl))
    cfg0 = gm.GMConfig(k_gm_max=2500.0, redi_kmax_default=1000.0)
    cfgh = cfg0._replace(k_gm_max_nh=1000.0, k_gm_hemi_trans=10.0)
    p = Params.defaults(cfg0)
    fk0, ki0, _, scal = gm.init_redi_gm(mesh, jnp.asarray(bv), jnp.asarray(hn), jnp.asarray(tap), p, cfg0)
    fkh, kih, _, _ = gm.init_redi_gm(mesh, jnp.asarray(bv), jnp.asarray(hn), jnp.asarray(tap), p, cfgh)
    latd = np.asarray(mesh.geo_coord_nod2D)[:, 1] / RAD
    w = np.clip((latd + 5.0) / 10.0, 0.0, 1.0)
    kmax = (1.0 - w) * 2500.0 + w * 1000.0
    top = np.maximum(np.asarray(scal) * kmax, 2.0)
    zs = 0.6 + 0.4 * np.exp(-np.abs(np.asarray(mesh.zbar_3d_n)[:, 0]) / 500.0)
    ref_top = top * np.clip(zs, 0.6, 1.0)
    e = np.abs(np.asarray(fkh)[:, 0] - ref_top).max() / ref_top.max()
    print(f"[gm hemi] fer_K top vs the Fortran formula: rel {e:.2e}; lat range {latd.min():.1f}..{latd.max():.1f}, "
          f"K_GM top (no hemi) {float(np.asarray(fk0)[:, 0].max()):.1f}, (hemi) min {float(np.asarray(fkh)[:, 0].min()):.1f}"
          f" max {float(np.asarray(fkh)[:, 0].max()):.1f}; Ki unchanged {np.array_equal(np.asarray(ki0), np.asarray(kih))}")
    ok &= e < 1e-14 and bool(np.array_equal(np.asarray(ki0), np.asarray(kih)))

    # --- coupled albedo with alb_tramp / h_snowscale ---
    from fesom_jax.coupled import flux
    from fesom_jax.ice import IceConfig
    t = np.linspace(271.0, 273.15, 50)
    h = np.full(50, 1.0)
    hsn = np.linspace(0.0, 0.2, 50)
    ic = IceConfig(albsn=0.82, albsnm=0.65, albi=0.70, albim=0.68, alb_tramp=1.0, h_snowscale=0.05, cpl_meltponds=False)
    fm = np.clip((t - 273.15 + 1.0) / 1.0, 0.0, 1.0)
    a_s = 0.82 + fm * (0.65 - 0.82)
    a_b = 0.70 + fm * (0.68 - 0.70)
    al = np.tanh(hsn / 0.05)
    ref_alb = al * a_s + (1 - al) * a_b
    z = jnp.zeros(50)
    cf = flux.CplForcing(*([z] * 12), Ssurf_month=z, chl=z)._replace(tsk_ico=jnp.asarray(t))
    out = flux.thermo_cpl(ic, flux.flux_config(ic), A=jnp.full(50, 0.9), h=jnp.asarray(h), hsn=jnp.asarray(hsn),
                          ice_temp_old=jnp.asarray(t), apnd=z, hpnd=z, ipnd=z, T_oc=z - 1.8, S_oc=z + 34.0, cf=cf,
                          runo=z, geo_lat=z + 1.0, non_cavity=jnp.ones(50, bool), dt=1800.0, use_virt_salt=True)
    e = np.abs(np.asarray(out.ice_alb) - ref_alb).max()
    print(f"[albedo] alb_tramp 1 K + h_snowscale 0.05 m vs ice_albedo: max abs {e:.2e}")
    ok &= e < 1e-14
    print("SFC KERNELS OK" if ok else "SFC KERNELS FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
