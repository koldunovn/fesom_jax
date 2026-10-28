#!/usr/bin/env python
"""CORE2 GPU speed plan: is a change round-off only? Compare the repeat runs of two programs (A and B, each with
several runs from the cold start, `gpu_profile.py noise`) on the OWNED lanes of a dist_<P> partition.

    noise_ab.py <A noise dir> <B noise dir> [--npes 4] [--dist /pool/.../core2]

Per field: the relative distance (max |x - y| / max |x|) of every within-A, within-B and A-B pair; the median of the
cross pairs over the median of the within pairs, and how many cross pairs exceed the largest within pair. A change
that only reorders round-off has cross pairs distributed like within pairs (median ratio ~1, about as many cross
pairs above the within maximum as chance gives); a real change puts every cross pair above the within ones. The
single-draw max-floor judge cannot tell these apart when the noise contains rare discrete events (a switch flipped
at one node), because one such event in one draw is enough to fail it.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np


def owned_masks(dist, npes):
    from fesom_jax import partit
    from fesom_jax.shard_mesh import local_sizes
    part = partit.read_partition(Path(dist), npes)
    _, Lmax = local_sizes(part)
    masks = {}
    for kind, myd in (("nod", part.myDim_nod2D), ("elem", part.myDim_elem2D)):
        m = np.zeros((npes, Lmax[kind]), bool)
        for d in range(npes):
            m[d, : int(myd[d])] = True
        masks[npes * Lmax[kind]] = m.reshape(-1)
    return masks


def load_owned(path, masks):
    with np.load(path) as z:
        return {k: z[k][masks[z[k].shape[0]]].astype(np.float64) for k in z.files
                if z[k].ndim and z[k].shape[0] in masks}


def rel(a, b):
    out = {}
    for f in a:
        d = np.abs(a[f] - b[f])
        s = float(np.abs(a[f]).max()) if a[f].size else 0.0
        out[f] = float(d.max()) / (s or 1.0) if d.size else 0.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--npes", type=int, default=4)
    ap.add_argument("--dist", default="/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
    a = ap.parse_args(argv)
    masks = owned_masks(a.dist, a.npes)
    A = [load_owned(p, masks) for p in sorted(Path(a.a).glob("state_*.npz"))]
    B = [load_owned(p, masks) for p in sorted(Path(a.b).glob("state_*.npz"))]
    within = [rel(x, y) for x, y in itertools.combinations(A, 2)] + \
             [rel(x, y) for x, y in itertools.combinations(B, 2)]
    cross = [rel(x, y) for x in A for y in B]
    print(f"A: {len(A)} runs ({a.a}); B: {len(B)} runs ({a.b}); {len(within)} within pairs, "
          f"{len(cross)} cross pairs; owned lanes only")
    print(f"{'field':14s} {'med within':>11s} {'med cross':>11s} {'ratio':>6s} {'max within':>11s} "
          f"{'max cross':>11s}  cross>max(within)")
    rows = []
    for f in sorted(A[0]):
        w = np.array([r[f] for r in within])
        c = np.array([r[f] for r in cross])
        if w.max() == 0 and c.max() == 0:
            continue
        mw, mc = float(np.median(w)), float(np.median(c))
        rows.append((mc / mw if mw else float("inf"), f, mw, mc, float(w.max()), float(c.max()),
                     int(np.count_nonzero(c > w.max()))))
    for ratio, f, mw, mc, xw, xc, n in sorted(rows, reverse=True):
        print(f"{f:14s} {mw:11.3e} {mc:11.3e} {ratio:6.2f} {xw:11.3e} {xc:11.3e}  {n}/{len(cross)}")
    worst = max(r[0] for r in rows)
    print(f"largest median ratio {worst:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
