#!/usr/bin/env python
"""Field-level check of the coupled flux's cavity branches (fesom_jax/coupled/flux.py: cavity_virtual_salt,
cavity_fw_balance_zstar, cavity_stress, top_wet) against a line-by-line numpy transcription of FESOM main 0aaa2d73's
loops (ice_oce_coupling.F90 oce_fluxes :375-438 and :566-598, oce_fluxes_mom :72-117; gen_support.F90
integrate_nod_2D), on lane L-OCN's CORE3 mesh with synthetic fields (water flux, salinity, stresses everywhere,
including the cavity nodes), double precision.

    JAX_PLATFORMS=cpu python scripts/coupled/check_cavity_flux.py [--mesh DIR]"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", default="/work/ba1583/a270088/awicm/core3/ocn/mesh_core3")
    a = p.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from fesom_jax.coupled import flux as FX
    M = a.mesh
    ul = np.load(f"{M}/ulevels_nod2D.npy")
    ule = np.load(f"{M}/ulevels.npy")
    av = np.load(f"{M}/areasvol.npy")                       # (nod, nl)
    en = np.load(f"{M}/elem_nodes.npy")
    meta = dict(l.split() for l in open(f"{M}/meta.txt"))
    ocean_area = float(meta["ocean_area"])
    n, nl = av.shape
    rng = np.random.default_rng(1)
    wf = rng.normal(0, 1e-7, n)                              # incl. cavity nodes (the melt's place)
    S = 34 + rng.normal(0, 0.5, (n, nl))
    flux = rng.normal(0, 1e-7, n)
    sx, sy = rng.normal(0, 0.1, n), rng.normal(0, 0.1, n)
    cav = ul > 1
    res, ok = {}, True

    # --- FESOM, transcribed
    def integrate_nod(x):
        return sum(x[r] * av[r, ul[r] - 1] for r in range(n))
    for use_virt_salt in (True, False):
        vs = np.zeros(n)
        if use_virt_salt:
            for i in range(n):
                vs[i] = S[i, ul[i] - 1] * wf[i]
        else:
            for i in range(n):
                vs[i] = 0.0
                if ul[i] == 1:
                    continue
                vs[i] = S[i, ul[i] - 1] * wf[i]
        net = integrate_nod(vs) / ocean_area
        for i in range(n):
            if ul[i] > 1:
                continue
            vs[i] = vs[i] - net
        mine = np.asarray(FX.cavity_virtual_salt(jnp.asarray(wf), FX.top_wet(jnp.asarray(S), ul), FX.top_wet(
            jnp.asarray(av), ul), ocean_area, jnp.asarray(~cav), use_virt_salt=use_virt_salt))
        d = float(np.abs(mine - vs).max() / np.abs(vs).max())
        res[f"virtual_salt use_virt_salt={use_virt_salt}"] = d
        ok &= d < 1e-12
    # zstar freshwater balance
    f2 = flux.copy()
    f2[cav] = 0.0
    net = integrate_nod(f2) / ocean_area
    ref = wf.copy()
    for i in range(n):
        if ul[i] > 1:
            continue
        ref[i] = ref[i] + net
    mine = np.asarray(FX.cavity_fw_balance_zstar(jnp.asarray(wf), jnp.asarray(flux), FX.top_wet(jnp.asarray(av), ul),
                                                 ocean_area, jnp.asarray(~cav)))
    d = float(np.abs(mine - ref).max() / np.abs(ref).max())
    res["fw_balance_zstar"] = d
    ok &= d < 1e-12
    # stresses (the atmosphere's stress blended on open nodes; cavity nodes/elements untouched = 0 before the drag)
    sns = np.zeros((n, 2))
    for i in range(n):
        if ul[i] > 1:
            continue
        sns[i] = (sx[i], sy[i])
    ss = np.zeros((en.shape[0], 2))
    for e in range(en.shape[0]):
        if ule[e] > 1:
            continue
        ss[e] = sns[en[e]].sum(axis=0) / 3.0
    sns_in = np.c_[sx, sy]
    ss_in = (sns_in[en[:, 0]] + sns_in[en[:, 1]] + sns_in[en[:, 2]]) / 3.0      # fesom_jax's blend (all nodes)
    ss_m, sns_m = FX.cavity_stress(jnp.asarray(ss_in), jnp.asarray(sns_in), jnp.asarray(~cav), jnp.asarray(ule <= 1))
    res["stress_node_surf"] = float(np.abs(np.asarray(sns_m) - sns).max())
    front = (ule <= 1) & cav[en].any(axis=1)
    dd = np.abs(np.asarray(ss_m) - ss).max(axis=1)
    res["stress_surf_non_front_elems"] = float(dd[~front].max())
    # an open element (ulevels = 1) never has a cavity node (ulevels_nod2D = the min over its elements' ulevels)
    res["open_elems_with_a_cavity_node"] = int(front.sum())
    ok &= front.sum() == 0
    ok &= res["stress_node_surf"] == 0.0 and res["stress_surf_non_front_elems"] < 1e-15
    res["cavity_nodes"], res["cavity_elems"] = int(cav.sum()), int((ule > 1).sum())
    res["ok"] = bool(ok)
    import json
    print(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
