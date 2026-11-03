#!/usr/bin/env python
"""Replay ladder on CORE3 with cavities: one fesom_jax kernel per substep, fed the oracle's own inputs.

    python scripts/core3/replay_ladder.py DUMP_DIR MESH_JAX RESTART_JAX [--step 1] [--rungs 3 5 7 8 9 10 11 12 13 16]
                                          [--precision double] [--dt 1800]

Each rung takes the inputs of one substep from the oracle's FULL dumps of the SAME step (earlier substeps, the
``pre_fluxes`` state, ``oce_fluxes``) and, for what FESOM keeps from the previous step and does not dump (the AB2
momentum slot, ``w_expl``, ``ssh_rhs_old``, the CG warm start ``d_eta``), from the converted restart -- so only
``--step 1`` is fully determined (later steps need those carried fields from the previous step). It runs the
fesom_jax kernel and compares its output with the oracle's dump of that substep, split into cavity nodes /
elements, the cavity front (open, touching a cavity cell) and the rest, as max |diff| / max |field| and the share of
bitwise-equal values. Run it against the DOUBLE-precision oracle (``run_cav_dp``) with ``--precision double``.

Rungs (jaxcpl: linfs, no partial cells):
  2  sw_ab        eos.compute_sw_alpha_beta(T, S)                                        -> sw_alpha, sw_beta [2]
  3  pgf          pgf.pressure_force_linfs(hpressure[1])                                 -> pgf_x, pgf_y [3]
  4  mixing       tke.mixing_tke(uv, bvfreq[1], tke*, stress_node_surf[19], hnode)        -> Kv, Av [4]
  5  vel_rhs      momentum.compute_vel_rhs(uv, uv_rhsAB*, eta_n, pgf[3], w_e*, hnode)    -> uv_rhs [5]
  7  impl_visc    momentum.impl_vert_visc(uv, uv_rhs[6], Av[4], stress_surf[19])         -> du (uv_rhs [7])
  8  ssh_rhs      ssh.compute_ssh_rhs(uv, du[7], helem, ssh_rhs_old*)                    -> ssh_rhs [8]
  9  ssh_solve    ssh.solve_ssh(op, ssh_rhs[8], x0 = d_eta*)                              -> d_eta [9] (CG tolerance)
 10  update_vel   momentum.update_vel(uv, du[7], d_eta[9])                               -> uv [10]
 11  hbar         ssh.compute_hbar(uv[10], helem, hbar)                                  -> hbar, ssh_rhs_old [11]
 12  eta_n        ssh.eta_n_update(eta_n, hbar[11], hbar)                                 -> eta_n [12]
 13  w            ale.compute_w(uv[10], helem)                                            -> w [13]
 16  thickness    linfs: hnode, helem unchanged                                           -> hnode, helem [16]
Rung 15 (tracers: FCT advection + implicit vertical diffusion, the melt as the BC at the shelf base) runs without
GM/Redi: use it against the oracle run with fer_gm = redi = .false. (``fesom_oracle/run_cav_dp_nogm``).
(* = from the converted restart.) Rungs 6 (opt_visc 5), 14 (GM) are not here (lane L-SFC's): their
fesom_jax kernels need jaxcpl's L-SFC settings first; the ladder feeds the oracle's outputs past them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fesom_jax import io_dump, zarr_output  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir"); ap.add_argument("mesh_dir"); ap.add_argument("restart")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--rungs", type=int, nargs="+", default=[2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 16])
    ap.add_argument("--precision", default="double", choices=("double", "single"))
    ap.add_argument("--dt", type=float, default=1800.0)
    a = ap.parse_args()
    import jax.numpy as jnp
    from fesom_jax import ale, eos, momentum, pgf, precision, ssh, tke, tracer_adv, tracer_diff
    from fesom_jax.config import VCPW
    from fesom_jax.params import Params
    from fesom_jax.mesh import load_mesh

    mesh = load_mesh(a.mesh_dir)
    N, E, nl = int(mesh.nod2D), int(mesh.elem2D), int(mesh.nl)
    uln = np.asarray(mesh.ulevels_nod2D); ule = np.asarray(mesh.ulevels)
    ulmax = np.asarray(mesh.ulevels_nod2D_max); en = np.asarray(mesh.elem_nodes)
    g_nod = {"cavity": uln > 1, "front": (uln == 1) & (ulmax > 1), "open": ulmax == 1}
    efront = (ule == 1) & (ulmax[en].max(axis=1) > 1)
    g_elem = {"cavity": ule > 1, "front": efront, "open": (ule == 1) & ~efront}
    lay_n, ifc_n = np.asarray(mesh.node_layer_mask), np.asarray(mesh.node_iface_mask)
    lay_e = np.asarray(mesh.elem_layer_mask)

    st = a.step
    pad = lambda x: np.pad(x, ((0, 0), (0, nl - x.shape[1]))) if x.ndim == 2 and x.shape[1] < nl else x

    def D(sub, name, n=N, two_d=False):
        x = io_dump.load_full_field(a.dump_dir, st, sub, name, n)
        return x[:, 0] if two_d else pad(x)

    def Duv(sub, name):
        return np.stack([D(sub, name + "_1", E), D(sub, name + "_2", E)], axis=-1)

    R = lambda f: np.asarray(zarr_output.reconstruct_global(a.restart, f), np.float64)

    def report(rung, label, mine, ref, mask, groups):
        mine = np.asarray(mine, np.float64)
        if mine.ndim == 3:                        # (e, nl, 2): both components
            mine, ref = mine.reshape(mine.shape[0], -1), ref.reshape(ref.shape[0], -1)
            mask = np.repeat(mask, 2, axis=1) if mask is not None else None
        if mask is None:
            mask = np.ones(mine.shape, bool)
        elif mine.ndim == 1:
            mask = mask.reshape(-1)
        scale = np.abs(ref[mask]).max() or 1.0
        line = f"rung {rung:2d} {label:12s} scale {scale:.3e} |"
        for g, gm in groups.items():
            mm = mask & (gm[:, None] if mine.ndim == 2 else gm)
            if not mm.any():
                continue
            d = np.abs(mine[mm] - ref[mm])
            line += f" {g}: {d.max() / scale:.2e} bitwise {np.mean(d == 0):.3f} |"
        print(line, flush=True)

    with precision.scope(a.precision):
        J = lambda x: jnp.asarray(x)
        uv0 = Duv(20, "uv"); hnode = D(20, "hnode"); eta0 = D(20, "eta_n", two_d=True)
        hbar0 = D(20, "hbar", two_d=True)
        helem = D(16, "helem", E)
        if 2 in a.rungs:
            sa, sb = eos.compute_sw_alpha_beta(mesh, J(D(20, "T")), J(D(20, "S")))
            report(2, "sw_alpha", sa, D(2, "sw_alpha"), lay_n, g_nod)
            report(2, "sw_beta", sb, D(2, "sw_beta"), lay_n, g_nod)
        if 3 in a.rungs:
            px, py = pgf.pressure_force_linfs(mesh, J(D(1, "pressure")))
            report(3, "pgf_x", px, D(3, "pgf_x", E), lay_e, g_elem)
            report(3, "pgf_y", py, D(3, "pgf_y", E), lay_e, g_elem)
        if 4 in a.rungs:      # cvmix TKE (fesom_jax TkeConfig defaults = jaxcpl's namelist.cvmix values) + mo_convect
            sns = np.stack([D(19, "stress_node_surf_x", two_d=True), D(19, "stress_node_surf_y", two_d=True)], -1)
            Kv, Av, _, _ = tke.mixing_tke(mesh, J(uv0), J(D(1, "bvfreq")), J(R("tke")), J(sns), J(hnode),
                                          tke.TkeConfig(), Params.defaults(), dt=a.dt, cavity=True)
            report(4, "Kv", Kv, D(4, "Kv"), ifc_n, g_nod)
            report(4, "Av", Av, D(4, "Av", E), np.asarray(mesh.elem_iface_mask), g_elem)
        if 5 in a.rungs:
            rhsAB = R("uv_rhsAB"); w_e = R("w_e")
            uv_rhs, _ = momentum.compute_vel_rhs(mesh, J(uv0), J(rhsAB), J(eta0), J(D(3, "pgf_x", E)),
                                                 J(D(3, "pgf_y", E)), J(w_e), J(hnode), is_first_step=False,
                                                 dt=a.dt, cavity=True)
            report(5, "uv_rhs", uv_rhs, Duv(5, "uv_rhs"), lay_e, g_elem)
        du_o = Duv(7, "uv_rhs")
        if 7 in a.rungs:
            ss = np.stack([D(19, "stress_surf_x", E, True), D(19, "stress_surf_y", E, True)], -1)
            du = momentum.impl_vert_visc(mesh, J(uv0), J(Duv(6, "uv_rhs")), J(D(4, "Av", E)), J(ss), dt=a.dt)
            report(7, "du", du, du_o, lay_e, g_elem)
        if 8 in a.rungs:
            rhs = ssh.compute_ssh_rhs(mesh, J(uv0), J(du_o), J(helem), ssh_rhs_old=J(R("ssh_rhs_old")))
            report(8, "ssh_rhs", rhs, D(8, "ssh_rhs", two_d=True), None, g_nod)
        if 9 in a.rungs:
            op = ssh.build_ssh_operator(mesh, dt=a.dt)
            sol = ssh.solve_ssh(op, J(D(8, "ssh_rhs", two_d=True)), x0=J(R("d_eta")))
            sol = sol[0] if isinstance(sol, tuple) else sol
            report(9, "d_eta", sol, D(9, "d_eta", two_d=True), None, g_nod)
            # is it the operator or the solver? the residual of each solution under fesom_jax's operator
            b = D(8, "ssh_rhs", two_d=True)
            for lab, x in (("oracle d_eta", D(9, "d_eta", two_d=True)), ("fesom_jax d_eta", np.asarray(sol))):
                r = np.asarray(ssh.ssh_matvec(op, J(x))) - b
                print(f"   rung  9 residual |A x - b| / |b| of the {lab:15s} under fesom_jax's A: "
                      f"{np.linalg.norm(r) / np.linalg.norm(b):.3e}  (cavity rows {np.linalg.norm(r[g_nod['cavity']]) / np.linalg.norm(b):.3e})")
        if 10 in a.rungs:
            uv1 = momentum.update_vel(mesh, J(uv0), J(du_o), J(D(9, "d_eta", two_d=True)), dt=a.dt)
            report(10, "uv", uv1, Duv(10, "uv"), lay_e, g_elem)
        uv1_o = Duv(10, "uv")
        if 11 in a.rungs:
            sro, hb = ssh.compute_hbar(mesh, J(uv1_o), J(helem), J(hbar0), dt=a.dt)
            report(11, "hbar", hb, D(11, "hbar", two_d=True), None, g_nod)
            report(11, "ssh_rhs_old", sro, D(11, "ssh_rhs_old", two_d=True), None, g_nod)
        if 12 in a.rungs:
            eta = ssh.eta_n_update(mesh, J(eta0), J(D(11, "hbar", two_d=True)), J(hbar0))
            report(12, "eta_n", eta, D(12, "eta_n", two_d=True), None, g_nod)
        if 13 in a.rungs:
            w = ale.compute_w(mesh, J(uv1_o), J(helem))
            report(13, "w", w, D(13, "w"), ifc_n, g_nod)
        if 15 in a.rungs:
            # FCT advection (T then S) + implicit vertical diffusion with the surface BC at the top wet layer (the
            # melt under the shelves) and the shortwave penetration. GM/Redi are NOT in this rung: run it against an
            # oracle with fer_gm = redi = .false. (fesom_oracle/run_cav_dp_nogm), where uv / w are the tracer
            # transports as dumped.
            uv_adv, w_adv = J(uv1_o), J(D(13, "w"))
            hn = J(hnode)
            T_adv, _ = tracer_adv.advect_one_fct(mesh, uv_adv, w_adv, J(helem), hn, hn, J(D(20, "T")),
                                                 J(R("T_old")), dt=a.dt, cavity=True)
            S_adv, _ = tracer_adv.advect_one_fct(mesh, uv_adv, w_adv, J(helem), hn, hn, J(D(20, "S")),
                                                 J(R("S_old")), dt=a.dt, cavity=True)
            # the advection tendency alone (FESOM's del_ttf, dumped as substep 21 by the oracle since a20dd2d)
            try:
                for lab, X in (("T", "T"), ("S", "S")):
                    dtt, _ = tracer_adv.advect_one_fct(mesh, uv_adv, w_adv, J(helem), hn, hn, J(D(20, X)),
                                                       J(R(X + "_old")), dt=a.dt, content=True, cavity=True)
                    report(15, "del_ttf_" + lab, dtt, D(21, "del_ttf_" + lab), lay_n, g_nod)
            except LookupError as e:
                print(f"   rung 15: no advection-tendency dump ({e})")
            bc_T = -a.dt * D(19, "heat_flux", two_d=True) / VCPW
            bc_S = a.dt * (D(19, "virtual_salt", two_d=True) + D(19, "relax_salt", two_d=True))
            T1, S1 = tracer_diff.impl_vert_diff(mesh, T_adv, S_adv, J(D(15, "Kv_after")), hn, dt=a.dt,
                                                bc_T=J(bc_T), bc_S=J(bc_S), sw_3d=J(D(19, "sw_3d")))
            report(15, "T", T1, D(15, "T"), lay_n, g_nod)
            report(15, "S", S1, D(15, "S"), lay_n, g_nod)
            top = (uln - 1)[:, None]
            for lab, X, Xo in (("T", T1, D(15, "T")), ("S", S1, D(15, "S"))):
                d = np.abs(np.take_along_axis(np.asarray(X), top, 1) - np.take_along_axis(Xo, top, 1))[:, 0]
                print(f"   rung 15 {lab} at the top wet layer (the melt's BC) under the shelves: max |diff| "
                      f"{d[g_nod['cavity']].max():.3e}, open {d[g_nod['open']].max():.3e}")
        if 16 in a.rungs:
            report(16, "hnode", hnode, D(16, "hnode"), lay_n, g_nod)
            report(16, "helem", R("helem"), helem, lay_e, g_elem)


if __name__ == "__main__":
    main()
