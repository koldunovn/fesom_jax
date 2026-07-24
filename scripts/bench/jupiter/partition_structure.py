"""Partition-STRUCTURE probe for the ng5 x P=128 cliff (HANDOFF-20260724, follow-up to
eliminated hypotheses 3-7): recompute, for each P, exactly the exchange structure the model
builds — `_ragged_exchange_map` + `_colour_edges` per kind (nod/elem/edge) — straight from
the `dist_<P>` files.  No mesh export load (global counts come from the id lists), no GPU,
no job: runs on a login node in minutes.

Reports per (mesh, P, kind): Lmax, halo volume, neighbour-graph degrees, colouring rounds K,
per-round slot widths, packed coloured buffer `total`, padded `pad_slot` (x P = dense-a2a
width), ragged send/recv_max.  The campaign checked K and buffer volume for NOD only; the
model also exchanges ELEM and EDGE kinds — a pathology there would have been invisible.

Also writes the exact per-round ppermute `perms` to an .npz for experiment (3) (replaying
the real permutation pattern in collective_scan.py instead of the ring).

    JAX_PLATFORMS=cpu python scripts/bench/jupiter/partition_structure.py \
        --dist-dir $RAW_ng5 --npes 32 64 128 256 --name ng5 --out /tmp/ng5_struct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fesom_jax import partit
from fesom_jax.shard_mesh import (_owner_map, _ragged_exchange_map, _colour_edges,
                                  local_sizes, _KINDS)


def analyze(dist_dir: Path, npes: int, name: str):
    part = partit.read_partition(dist_dir, npes)
    n_local, Lmax = local_sizes(part)
    mylist = {"nod": part.myList_nod2D, "elem": part.myList_elem2D,
              "edge": part.myList_edge2D}
    mydim = {"nod": part.myDim_nod2D, "elem": part.myDim_elem2D,
             "edge": part.myDim_edge2D}
    out = {"mesh": name, "P": npes}
    perms_npz = {}
    for k in _KINDS:
        # global count: max id + 1 (every entity appears in some rank's list)
        gcount = int(max(int(ml.max()) for ml in mylist[k])) + 1
        owner, owner_local = _owner_map(mylist[k], mydim[k], gcount)
        rex = _ragged_exchange_map(mylist[k], mydim[k], Lmax[k], owner, owner_local)
        ss = rex.send_sizes
        classes = _colour_edges(ss, npes)
        slots = [max(n for _, _, n in cls) for cls in classes]
        pairs = [len(cls) for cls in classes]
        outdeg = (ss > 0).sum(axis=1)
        indeg = (ss > 0).sum(axis=0)
        halo = np.asarray(n_local[k]) - np.asarray(mydim[k])
        r = {
            "Lmax": int(Lmax[k]),
            "padfac": round(float(Lmax[k] / np.mean(n_local[k])), 4),
            "halo_sum": int(halo.sum()), "halo_max": int(halo.max()),
            "graph_edges": int((ss > 0).sum()),
            "deg_out_max": int(outdeg.max()), "deg_out_mean": round(float(outdeg.mean()), 2),
            "deg_in_max": int(indeg.max()),
            "K": len(classes),
            "slots": slots,
            "round_pairs": pairs,
            "coloured_total": int(sum(slots)),
            "pad_slot": int(max(int(rex.recv_sizes.max()), 1)),
            "padded_width": int(max(int(rex.recv_sizes.max()), 1)) * npes,
            "send_max": int(rex.send_max), "recv_max": int(rex.recv_max),
            "chunk_max": int(ss.max()),
            "chunk_mean_nz": round(float(ss[ss > 0].mean()), 1) if (ss > 0).any() else 0.0,
        }
        out[k] = r
        perms_npz[f"{k}_send_sizes"] = ss
        for i, cls in enumerate(classes):
            perms_npz[f"{k}_round{i:02d}"] = np.array([(d, e, n) for d, e, n in cls],
                                                      dtype=np.int64)
    return out, perms_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist-dir", required=True)
    ap.add_argument("--npes", type=int, nargs="+", required=True)
    ap.add_argument("--name", default="mesh")
    ap.add_argument("--out", default=None, help="prefix for .json/.npz outputs")
    args = ap.parse_args()
    for P in args.npes:
        res, perms = analyze(Path(args.dist_dir), P, args.name)
        print(json.dumps(res), flush=True)
        if args.out:
            np.savez_compressed(f"{args.out}_{args.name}_p{P}_perms.npz", **perms)


if __name__ == "__main__":
    main()
