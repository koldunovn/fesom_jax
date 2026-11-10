#!/usr/bin/env python
"""Lane L-SFC rung 2: the jaxcpl switches against FESOM main 0aaa2d73 in single precision (the oracle lane L-OCN built,
run on CORE2 without cavities with jaxcpl's namelists by ``/work/ba1583/a270088/awicm/core3/sfc/oracle_core2``).
Each check replays one block on the oracle's own inputs of the same step (full-field dumps, or the Fortran restart
the oracle started from) in fesom_jax single precision, and compares the block's output with the oracle's dump:

* viscosity (opt_visc 5, visc_filt_bcksct, easybsreturn 1.5): in = uv at the step start (step 1: the restart's u/v;
  step 2: s1_10 uv) and s<n>_5 uv_rhs; out = s<n>_6 uv_rhs. Control: opt_visc 7 on the same inputs;
* GM/Redi bolus velocity (K_GM_max 2500 / K_GM_max_NH 1000, Redi_Kmax 1000, scaling_*, ODM95): in = T, S at the step
  start (step 2: s1_15), bvfreq s2_1, hnode/helem s1_16; out = s2_14 fer_uv. Control: no hemispheric ceiling;
* mEVP (alpha/beta 300, 120 sweeps, the mass floor, metric_factor_zero): in = the ice state at the step start (step 1:
  the restart, stresses 0 as FESOM starts them; step 2: s1_18 and this replay's own step-1 stresses), uvnode/hbar of
  s<n>_20, the atmosphere-ice stress from s<n>_17's wind; out = s<n>_18 uice/vice. Controls: the port base's mass
  factor / the metric terms (jaxcpl: metric_factor_zero).

For each: max|Δ| / max|ref| over the wet points, the area-weighted rms ratio, and the probe columns through
fesom_jax.verify.compare_column (single-precision tolerances).

    python scripts/coupled/sfc_oracle_replay.py --oracle <run dir> --restart <Fortran restart dir prefix>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import gm, ice_evp, ice_mevp, momentum
from fesom_jax import precision as PR
from fesom_jax import forcing as F
from fesom_jax.ale import live_geometry
from fesom_jax.config import ViscConfig
from fesom_jax.io_dump import DumpRecord
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.run_config import load_yaml
from fesom_jax.verify import compare_column

ROOT = Path(__file__).resolve().parents[2]


def full(run, step, sub, name, n):
    """One full-field dump of the oracle (format of fesom_dump_core3.F90; reader as lane L-OCN's io_dump)."""
    files = sorted((Path(run) / "dumps" / "full").glob(f"s{step}_{sub}_{name}.*"))
    if not files:
        raise LookupError(f"s{step}_{sub}_{name}")
    out = None
    for f in files:
        b = f.read_bytes()
        n_own, ncol = np.frombuffer(b, "<i4", count=2)
        gid = np.frombuffer(b, "<i4", count=n_own, offset=8)
        v = np.frombuffer(b, "<f8", count=n_own * ncol, offset=8 + 4 * n_own).reshape(n_own, ncol)
        if out is None:
            out = np.full((n, ncol), np.nan)
        out[gid - 1] = v
    return out


def pad(a, nl):
    """(n, k) -> (n, nl) with zeros below (fesom_jax keeps nl slots)."""
    a = np.nan_to_num(np.asarray(a, np.float64))
    if a.ndim == 1:
        return a
    o = np.zeros((a.shape[0], nl))
    k = min(nl, a.shape[1])
    o[:, :k] = a[:, :k]
    return o


def report(tag, got, ref, mask, probes=None, kind="scatter"):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    m = np.broadcast_to(mask, ref.shape) & np.isfinite(ref)
    d = np.abs(got - ref)[m]
    r = np.abs(ref[m])
    rel = d.max() / max(r.max(), 1e-300)
    rms = np.sqrt(np.mean(d ** 2)) / max(np.sqrt(np.mean(r ** 2)), 1e-300)
    line = f"[{tag}] max|Δ|/max|ref| {rel:.2e}  rms ratio {rms:.2e}  (max|ref| {r.max():.3e}, {m.sum()} points)"
    worst = []
    if probes is not None:
        for gid in probes:
            i = gid - 1
            mm = m[i]
            col, gcol = np.atleast_1d(ref[i][mm] if ref.ndim > 1 else ref[i:i + 1][mm]), \
                np.atleast_1d(got[i][mm] if got.ndim > 1 else got[i:i + 1][mm])
            nlev = int(col.size)
            if nlev == 0:
                continue
            rec = DumpRecord(step=0, substep=0, probe_gid=gid, nlevels=nlev, field=tag, values=col)
            res = compare_column(gcol, rec, kind=kind, rtol=1e-5, atol=1e-6 * max(r.max(), 1e-30))
            worst.append(res)
        npass = sum(bool(x) for x in worst)
        line += f"  probes {npass}/{len(worst)} within 1e-5 rel + 1e-6·max"
    print(line, flush=True)
    return rel, rms, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="/work/ba1583/a270088/awicm/core3/sfc/oracle_core2/run1")
    ap.add_argument("--restart", default="/work/ab0995/a270088/fesom_jax_dev/fortran/core2_rst/jan/fesom.1958")
    ap.add_argument("--converter", default="/work/ba1583/a270088/awicm/core3/sfc/snap/fortran-restart_c95e476")
    ap.add_argument("--config", default=str(ROOT / "configs" / "core2_cpl_jaxcpl.yaml"))
    ap.add_argument("--mesh", default="/work/ab0995/a270088/port_jax/data/mesh_core2")
    ap.add_argument("--precision", choices=("single", "double"), default="single",
                    help="single against the SP oracle (build_sp), double against the DP oracle (build_dp)")
    a = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("fortran_restart", f"{a.converter}/fesom_jax/fortran_restart.py")
    FR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FR)

    cfg = load_yaml(a.config)
    mesh = load_mesh(a.mesh)
    N, E, nl = int(mesh.nod2D), int(mesh.elem2D), int(mesh.nl)
    run = a.oracle
    pn = [int(x) for x in open(Path(run).parent / "probes" / "probes_nod.txt").read().split()]
    pe = [int(x) for x in open(Path(run).parent / "probes" / "probes_elem.txt").read().split()]
    oce, ice = FR.FortranRestart(a.restart + ".oce.restart"), FR.FortranRestart(a.restart + ".ice.restart")
    elm = np.asarray(mesh.elem_layer_mask)
    ndm = np.asarray(mesh.node_layer_mask)
    dt = float(cfg.dt)
    out = {}

    sp = a.precision == "single"
    with PR.scope(a.precision):
        m32 = PR.cast_tree(mesh) if sp else mesh
        f32 = (lambda x: jnp.asarray(np.asarray(x), jnp.float32)) if sp else \
            (lambda x: jnp.asarray(np.asarray(x), jnp.float64))          # noqa: E731

        # ---------------- viscosity ----------------
        uv1 = np.stack([pad(oce.read("u", E), nl), pad(oce.read("v", E), nl)], axis=-1)
        for step, uv_in in ((1, uv1), (2, None)):
            if uv_in is None:
                uv_in = np.stack([pad(full(run, 1, 10, "uv_1", E), nl), pad(full(run, 1, 10, "uv_2", E), nl)], -1)
            r5 = np.stack([pad(full(run, step, 5, "uv_rhs_1", E), nl), pad(full(run, step, 5, "uv_rhs_2", E), nl)], -1)
            r6 = np.stack([pad(full(run, step, 6, "uv_rhs_1", E), nl), pad(full(run, step, 6, "uv_rhs_2", E), nl)], -1)
            for opt, tag in ((cfg.visc, "visc opt5"), (ViscConfig(), "visc opt7 (control)")):
                fn = jax.jit(lambda u, r, vc=opt: momentum.horizontal_viscosity(m32, u, r, dt=dt, visc_cfg=vc))
                got = np.asarray(fn(f32(uv_in), f32(r5)), np.float64)
                # the viscous increment itself (uv_rhs after - before), both components
                res = report(f"step {step} {tag} Δuv_rhs", got - r5, r6 - r5, elm[..., None], probes=pe)
                out[(step, tag)] = res[:2]

        # ---------------- GM bolus velocity (step 2) ----------------
        T = pad(full(run, 1, 15, "T", N), nl)
        S = pad(full(run, 1, 15, "S", N), nl)
        bv = pad(full(run, 2, 1, "bvfreq", N), nl)
        hn = pad(full(run, 1, 16, "hnode", N), nl)
        he = pad(full(run, 1, 16, "helem", E), nl)
        fer_ref = np.stack([pad(full(run, 2, 14, "fer_uv_1", E), nl), pad(full(run, 2, 14, "fer_uv_2", E), nl)], -1)
        hn64 = jnp.asarray(hn)
        zbar3, Z3d = live_geometry(m32, hn64)
        for gcfg, tag in ((cfg.gm, "GM fer_uv (hemispheric)"),
                          (cfg.gm._replace(k_gm_max_nh=-1.0), "GM fer_uv no-hemi (control)")):
            p = PR.cast_tree(Params.defaults(gcfg)) if sp else Params.defaults(gcfg)
            fn = jax.jit(lambda T, S, bv, h, he, g=gcfg, p=p: gm.gm_diagnostics(
                m32, T, S, bv, h, he, p, g, Z3d=Z3d, zbar3=zbar3).fer_uv)
            got = np.asarray(fn(f32(T), f32(S), f32(bv), f32(hn), f32(he)), np.float64)
            res = report(f"step 2 {tag}", got, fer_ref, elm[..., None], probes=pe)
            out[(2, tag)] = res[:2]

        # ---------------- mEVP (steps 1, 2) ----------------
        bn = jnp.asarray(ice_evp.boundary_node_mask(mesh))
        from fesom_jax.ice_step import _with_ice_dt
        icfg = _with_ice_dt(cfg.ice, dt)
        st_in = {"a": ice.read("area", N), "m": ice.read("hice", N), "ms": ice.read("hsnow", N),
                 "u": ice.read("uice", N), "v": ice.read("vice", N)}
        sig = {}
        for variant, tag in (({}, "mEVP jaxcpl"),
                             ({"mevp_mass_floor": False}, "mEVP base mass (control)"),
                             ({"metric_factor_zero": False}, "mEVP with metric terms (control)")):
            c = icfg._replace(**variant)
            s11 = s12 = s22 = np.zeros(E)
            for step in (1, 2):
                if step == 1:
                    a_, m_, ms_, u_, v_ = st_in["a"], st_in["m"], st_in["ms"], st_in["u"], st_in["v"]
                else:
                    a_, m_, ms_ = (full(run, 1, 18, k, N)[:, 0] for k in ("a_ice", "m_ice", "m_snow"))
                    u_, v_ = full(run, 1, 18, "uice", N)[:, 0], full(run, 1, 18, "vice", N)[:, 0]
                uw = full(run, step, 20, "uvnode_1", N)[:, 0]
                vw = full(run, step, 20, "uvnode_2", N)[:, 0]
                hb = full(run, step, 20, "hbar", N)[:, 0]
                wx, wy = full(run, step, 17, "u_wind", N)[:, 0], full(run, step, 17, "v_wind", N)[:, 0]
                tx, ty = F.atm_ice_stress(f32(wx), f32(wy), f32(u_), f32(v_))

                def evp(a_, m_, ms_, u_, v_, s11, s12, s22, uw, vw, hb, tx, ty, c=c):
                    return ice_mevp.mevp_dynamics(c, m32, a_ice=a_, m_ice=m_, m_snow=ms_, u_ice=u_, v_ice=v_,
                                                  sigma11=s11, sigma12=s12, sigma22=s22, srfoce_u=uw,
                                                  srfoce_v=vw, elevation=hb, stress_ax=tx, stress_ay=ty,
                                                  boundary_node=bn)
                uo, vo, s11, s12, s22 = jax.jit(evp)(*(f32(x) for x in (a_, m_, ms_, u_, v_, s11, s12, s22,
                                                                         uw, vw, hb)), tx, ty)
                ref = np.stack([full(run, step, 18, "uice", N)[:, 0], full(run, step, 18, "vice", N)[:, 0]], -1)
                got = np.stack([np.asarray(uo, np.float64), np.asarray(vo, np.float64)], -1)
                icem = (np.asarray(a_) >= 0.01)[:, None]
                res = report(f"step {step} {tag} u/v_ice", got, ref, icem, probes=pn)
                out[(step, tag)] = res[:2]
    print("[replay] summary (max rel, rms ratio):", flush=True)
    for k, v in out.items():
        print(f"   step {k[0]} {k[1]:36s} {v[0]:.2e} {v[1]:.2e}", flush=True)


if __name__ == "__main__":
    main()
