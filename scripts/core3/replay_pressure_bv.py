#!/usr/bin/env python
"""Replay rung for substep 1 with cavities: ``eos.pressure_bv`` (``density_ref``, the cavity branch) vs the oracle.

    python scripts/core3/replay_pressure_bv.py DUMP_DIR MESH_JAX --precision single|double [--steps 1 2]

Feeds ``eos.reference_density`` + ``eos.compute_pressure_bv(..., density_ref=, cavity=True)`` the oracle's own T, S,
hnode of the step (the ``pre_fluxes`` full dump: nothing between it and ``pressure_bv`` changes them) and compares
``density_ref`` (pre_fluxes), ``density``, ``pressure`` (= hpressure) and ``bvfreq`` (substep 1) over every node, split
into cavity nodes, cavity-front nodes (open, touching a cavity cell) and the rest. Use ``--precision single`` with the
single-precision oracle and ``double`` with the double-precision one. Reports max |diff| / max |field| per group and
the share of bitwise-equal values.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fesom_jax import eos, io_dump, precision  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("mesh_dir")
    ap.add_argument("--precision", default="single", choices=("single", "double"))
    ap.add_argument("--steps", type=int, nargs="+", default=[1])
    ap.add_argument("--no-smooth", action="store_true", help="compare the raw (unsmoothed) N^2")
    a = ap.parse_args()
    import jax.numpy as jnp

    mesh64 = load_mesh(a.mesh_dir)
    uln = np.asarray(mesh64.ulevels_nod2D); nln = np.asarray(mesh64.nlevels_nod2D)
    ulmax = np.asarray(mesh64.ulevels_nod2D_max)
    N, nl = uln.size, int(mesh64.nl)
    k = np.arange(nl)[None, :]
    layer = (k >= uln[:, None] - 1) & (k < nln[:, None] - 1)
    iface = (k >= uln[:, None] - 1) & (k < nln[:, None])
    cav_lev = k < uln[:, None] - 1
    groups = {"cavity": uln > 1, "front": (uln == 1) & (ulmax > 1), "open": ulmax == 1}
    for step in a.steps:
        F = lambda sub, name: io_dump.load_full_field(a.dump_dir, step, sub, name, N)
        pad = lambda x: np.pad(x, ((0, 0), (0, nl - x.shape[1])))   # FESOM layer arrays are nl-1 deep
        T, S, h = pad(F(20, "T")), pad(F(20, "S")), pad(F(20, "hnode"))
        with precision.scope(a.precision):
            mesh = precision.cast_tree(mesh64) if a.precision == "single" else mesh64
            wp = precision.wp_dtype()
            dref = eos.reference_density(mesh)
            fn = eos.pressure_bv if a.no_smooth else eos.compute_pressure_bv
            out = fn(mesh, jnp.asarray(T, wp), jnp.asarray(S, wp), precision.full(jnp.asarray(h)),
                     density_ref=dref, cavity=True)
            mine = {"density_ref": dref, "density": out[0], "pressure": out[1], "bvfreq": out[2]}
            mine = {k_: np.asarray(v, np.float64) for k_, v in mine.items()}
        ref = {"density_ref": F(20, "density_ref"), "density": F(1, "density"), "pressure": F(1, "pressure"),
               "bvfreq": F(1, "bvfreq")}
        masks = {"density_ref": (k < nln[:, None] - 1), "density": layer | cav_lev, "pressure": layer,
                 "bvfreq": iface}
        for name in ("density_ref", "density", "pressure", "bvfreq"):
            o = ref[name]; ncol = o.shape[1]                     # nl or nl-1 stored columns
            j = mine[name][:, :ncol]; m = masks[name][:, :ncol]
            scale = np.abs(o[m]).max()
            line = f"step {step} {a.precision} {name:11s} scale {scale:.3e} |"
            for g, gm in groups.items():
                mm = m & gm[:, None]
                d = np.abs(o[mm] - j[mm])
                line += f" {g}: max {d.max() / scale:.2e} bitwise {np.mean(d == 0):.3f} |"
            print(line)
        # where the N^2 differences sit: per level relative to the node's top/bottom interface
        o, j = ref["bvfreq"], mine["bvfreq"][:, :ref["bvfreq"].shape[1]]
        m = masks["bvfreq"][:, :o.shape[1]]
        d = np.where(m, np.abs(o - j), 0.0)
        rel_top = k[:, :o.shape[1]] - (uln[:, None] - 1); rel_bot = (nln[:, None] - 1) - k[:, :o.shape[1]]
        for lab, sel in (("top interface", rel_top == 0), ("2nd interface", rel_top == 1),
                         ("bottom interface", rel_bot == 0), ("one above bottom", rel_bot == 1),
                         ("interior", (rel_top > 1) & (rel_bot > 1))):
            sm = m & sel
            print(f"   bvfreq diff at {lab:17s}: max {d[sm].max():.3e} share nonzero {np.mean(d[sm] > 1e-12 * np.abs(o[m]).max()):.3f}")
        # the cavity fill itself (levels above ulevels at cavity nodes)
        mm = (cav_lev & groups["cavity"][:, None])[:, :ref["density"].shape[1]]
        d = np.abs(ref["density"][mm] - mine["density"][:, :ref["density"].shape[1]][mm])
        print(f"   density in the shelf levels ({mm.sum()} values): max |diff| {d.max():.3e} kg/m3")


if __name__ == "__main__":
    main()
