#!/usr/bin/env python
"""Lane L-SFC rungs of the CORE3 cavity ladder (the companion of lane L-OCN's ``scripts/core3/replay_ladder.py``:
the same loaders, the same cavity / shelf-front / open split, the same max |diff| / max |field| measure).

    python scripts/core3/replay_ladder_sfc.py DUMP_DIR MESH_JAX [--evp-dump DIR] [--step 1] [--rungs 6 14 18]

Rungs (each kernel on the oracle's inputs of the same step, output against the oracle's dump of that substep):
  6   momentum.horizontal_viscosity(opt_visc 5, easybsreturn 1.5)(uv[20], uv_rhs[5])          -> uv_rhs [6]
  14  gm.gm_diagnostics(T, S [20], bvfreq [1], hnode [20], helem [16]; jaxcpl's GM block)     -> fer_uv [14],
      steps 1 .. --gm-steps (dumps of --gm-dump), with GMConfig.reproduce_fesom_bug_kgm_surface_level (a FESOM2 bug
      reproduced, not physics) carrying its stale slot from step to step, and with the switch off
  18  sea ice from the instrumented oracle copy (``--evp-dump``, substeps 21/22/23 of
      ``oracle_core2/fesom-evpdump``): mEVP (metric_factor_zero, mass floor, the shelf front as a coast) on the
      EVPdynamics_m inputs [21] -> u/v_ice, sigma [22]; ice FCT + cut_off + cavity cleaning on [21] and the mEVP
      output [22] -> a/m_ice, m_snow [23] (the forced thermodynamics after [23] are not replayed here).
Controls: opt_visc 7; no hemispheric ceiling; the metric terms; the mEVP without the shelf-front boundary.
``DUMP_DIR`` for rungs 6/14 is lane L-OCN's ``run_cav_dp/dumps``; ``--evp-dump`` is this lane's
``oracle_core2/run_cav_dp_evp/dumps`` (the same namelists, restart and binary source plus the ice dumps).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fesom_jax import io_dump  # noqa: E402


def groups(mesh):
    """lane L-OCN's split (replay_ladder.py): cavity, shelf front (open, touching a cavity cell), open."""
    uln = np.asarray(mesh.ulevels_nod2D); ule = np.asarray(mesh.ulevels)
    ulmax = np.asarray(mesh.ulevels_nod2D_max); en = np.asarray(mesh.elem_nodes)
    g_nod = {"cavity": uln > 1, "front": (uln == 1) & (ulmax > 1), "open": ulmax == 1}
    efront = (ule == 1) & (ulmax[en].max(axis=1) > 1)
    g_elem = {"cavity": ule > 1, "front": efront, "open": (ule == 1) & ~efront}
    return g_nod, g_elem


def report(rung, label, mine, ref, mask, grp):
    mine, ref = np.asarray(mine, np.float64), np.asarray(ref, np.float64)
    if mine.ndim == 3:
        mine, ref = mine.reshape(mine.shape[0], -1), ref.reshape(ref.shape[0], -1)
        mask = np.repeat(mask, 2, axis=1) if mask is not None else None
    if mask is None:
        mask = np.ones(mine.shape, bool)
    mask = mask & np.isfinite(ref)
    scale = np.abs(ref[mask]).max() or 1.0
    line = f"rung {rung:2d} {label:30s} scale {scale:.3e} |"
    out = {}
    for g, gm in grp.items():
        mm = mask & (gm[:, None] if mine.ndim == 2 else gm)
        if not mm.any():
            continue
        d = np.abs(mine[mm] - ref[mm])
        out[g] = (d.max() / scale, np.percentile(d, 99) / scale)
        line += f" {g}: {out[g][0]:.2e} (p99 {out[g][1]:.1e}) bitwise {np.mean(d == 0):.3f} |"
    print(line, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir"); ap.add_argument("mesh_dir")
    ap.add_argument("--evp-dump", default=None)
    ap.add_argument("--gm-dump", default=None, help="dump dir with full fields for steps 1..--gm-steps (rung 14)")
    ap.add_argument("--gm-steps", type=int, default=1)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--rungs", type=int, nargs="+", default=[6, 14, 18])
    ap.add_argument("--config", default=str(ROOT / "configs" / "core2_cpl_jaxcpl.yaml"))
    ap.add_argument("--dt", type=float, default=1800.0)
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from fesom_jax import cavity, gm, ice_adv, ice_evp, ice_mevp, ice_thermo, momentum
    from fesom_jax.config import ViscConfig
    from fesom_jax.ice_step import _with_ice_dt
    from fesom_jax.mesh import load_mesh
    from fesom_jax.params import Params
    from fesom_jax.run_config import load_yaml

    cfg = load_yaml(a.config)
    mesh = load_mesh(a.mesh_dir)
    N, E, nl = int(mesh.nod2D), int(mesh.elem2D), int(mesh.nl)
    g_nod, g_elem = groups(mesh)
    lay_e = np.asarray(mesh.elem_layer_mask)
    st = a.step
    pad = lambda x: np.pad(x, ((0, 0), (0, nl - x.shape[1]))) if x.ndim == 2 and x.shape[1] < nl else x  # noqa

    def D(sub, name, n=N, two_d=False, d=None):
        x = io_dump.load_full_field(d or a.dump_dir, st, sub, name, n)
        return x[:, 0] if two_d else pad(x)

    def Duv(sub, name, d=None):
        return np.stack([D(sub, name + "_1", E, d=d), D(sub, name + "_2", E, d=d)], axis=-1)

    J = lambda x: jnp.asarray(np.nan_to_num(np.asarray(x, np.float64)))   # noqa: E731
    if 6 in a.rungs:
        uv0, r5, r6 = Duv(20, "uv"), Duv(5, "uv_rhs"), Duv(6, "uv_rhs")
        for vc, tag in ((cfg.visc, "visc opt5 (jaxcpl)"), (ViscConfig(), "visc opt7 (control)")):
            got = np.asarray(jax.jit(lambda u, r, vc=vc: momentum.horizontal_viscosity(
                mesh, u, r, dt=a.dt, visc_cfg=vc))(J(uv0), J(r5)))
            report(6, tag + " Δuv_rhs", got - r5, r6 - r5, lay_e, g_elem)
    if 14 in a.rungs:
        # GM/Redi bolus velocity at steps 1 .. a.gm_steps, each on the oracle's inputs of that step. With
        # reproduce_fesom_bug_kgm_surface_level (a FESOM2 bug reproduced, not physics) the stale fer_k slot is carried
        # from this replay's own previous call, starting at 500 m2/s at step 1 as a run start does.
        he = D(16, "helem", E)
        stale = np.full(N, gm.FESOM_FER_K_INIT)
        variants = (("GM, FESOM bug reproduced", cfg.gm._replace(reproduce_fesom_bug_kgm_surface_level=True)),
                    ("GM, correct (switch off)", cfg.gm._replace(reproduce_fesom_bug_kgm_surface_level=False)),
                    ("GM, no hemispheric ceiling (control)",
                     cfg.gm._replace(reproduce_fesom_bug_kgm_surface_level=True, k_gm_max_nh=-1.0)))
        bad = np.asarray(mesh.ulevels_nod2D) != np.asarray(mesh.ulevels_nod2D_max)
        print(f"   rung 14: {int(bad.sum())} nodes with ulevels_nod2D /= ulevels_nod2D_max "
              f"({int((bad & (np.asarray(mesh.ulevels_nod2D) == 1)).sum())} at the shelf front)", flush=True)
        stales = {tag: stale.copy() for tag, _ in variants}
        for s_ in range(1, a.gm_steps + 1):
            Ds = lambda sub, name, n=N: pad(io_dump.load_full_field(a.gm_dump or a.dump_dir, s_, sub, name, n))  # noqa
            T, S, bv, hn = Ds(20, "T"), Ds(20, "S"), Ds(1, "bvfreq"), Ds(20, "hnode")
            ref = np.stack([Ds(14, "fer_uv_1", E), Ds(14, "fer_uv_2", E)], -1)
            for tag, gc in variants:
                p = Params.defaults(gc)
                kw = {"kgm_stale": J(stales[tag])} if gc.reproduce_fesom_bug_kgm_surface_level else {}
                out = jax.jit(lambda T, S, b, h, e, gc=gc, p=p, kw=kw: gm.gm_diagnostics(
                    mesh, T, S, b, h, e, p, gc, **kw))(J(T), J(S), J(bv), J(hn), J(he))
                if out.kgm_stale is not None:
                    stales[tag] = np.asarray(out.kgm_stale)
                report(14, f"step {s_} {tag}", np.asarray(out.fer_uv), ref, lay_e, g_elem)
    if 18 in a.rungs and a.evp_dump:
        dd = a.evp_dump
        x = {k: D(21, k, two_d=True, d=dd) for k in ("a_ice", "m_ice", "m_snow", "uice", "vice", "srfoce_u",
                                                     "srfoce_v", "srfoce_ssh", "stress_atmice_x", "stress_atmice_y")}
        sg = [D(21, k, E, two_d=True, d=dd) for k in ("sigma11", "sigma12", "sigma22")]
        ref_u = np.stack([D(22, "uice", two_d=True, d=dd), D(22, "vice", two_d=True, d=dd)], -1)
        ref_s = np.stack([D(22, k, E, two_d=True, d=dd) for k in ("sigma11", "sigma12", "sigma22")], -1)
        icfg = _with_ice_dt(cfg.ice, a.dt)
        bn0 = ice_evp.boundary_node_mask(mesh)
        bn_front = cavity.cavity_boundary_node_mask(mesh, bn0)
        icem = x["a_ice"] >= 0.01
        res = {}
        for c, bn, tag in ((icfg, bn_front, "mEVP (jaxcpl, front = coast)"),
                           (icfg._replace(metric_factor_zero=False), bn_front, "mEVP with metric (control)"),
                           (icfg, bn0, "mEVP front not coast (control)")):
            r = jax.jit(lambda *v, c=c, bn=bn: ice_mevp.mevp_dynamics(
                c, mesh, a_ice=v[0], m_ice=v[1], m_snow=v[2], u_ice=v[3], v_ice=v[4], sigma11=v[10],
                sigma12=v[11], sigma22=v[12], srfoce_u=v[5], srfoce_v=v[6], elevation=v[7], stress_ax=v[8],
                stress_ay=v[9], boundary_node=jnp.asarray(bn)))(*(J(v) for v in list(x.values()) + sg))
            r = (cavity.cavity_ice_clean_vel(mesh, r[0], r[1]) + tuple(r[2:]))
            gu = np.stack([np.asarray(r[0]), np.asarray(r[1])], -1)
            report(18, tag + " u/v_ice", gu, ref_u, np.broadcast_to(icem[:, None], gu.shape), g_nod)
            res[tag] = gu
        # the ice FCT + cut_off + cavity cleaning, on the oracle's own pre-EVP state and post-EVP velocity
        adv = jax.jit(lambda a_, m_, s_, u_, v_: ice_adv.fct_solve(icfg, mesh, a_, m_, s_, u_, v_))(
            J(x["a_ice"]), J(x["m_ice"]), J(x["m_snow"]), J(ref_u[:, 0]), J(ref_u[:, 1]))
        co = ice_thermo.cut_off(*adv)
        co = cavity.cavity_ice_clean_ma(mesh, *co)
        for k, v in zip(("a_ice", "m_ice", "m_snow"), co):
            report(18, f"FCT+cut_off {k} [23]", v, D(23, k, two_d=True, d=dd), None, g_nod)


if __name__ == "__main__":
    main()
