#!/usr/bin/env python
"""Why does the padded halo lose at large P? Compute its wire volume straight from the
partition files (no mesh, no GPU, seconds on a login node).

The padded transport ships ``P * pad_slot`` lanes from every device to every other,
where ``pad_slot = max`` over ALL (d,e) pairs of the d->e chunk. The ragged transport
ships only the real chunks (``send_max = max_d sum_e chunk(d,e)`` lanes). So

    pad factor = P * pad_slot / send_max

is the byte-for-byte cost of AD-correctness, and it grows with P because the number of
slots grows while the real neighbour count of a spatial partition does not. This script
tabulates it for the meshes in the halo A/B, so the crossover can be predicted rather
than measured 16 nodes at a time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fesom_jax import partit                                    # noqa: E402
from fesom_jax.shard_mesh import _owner_map, local_sizes        # noqa: E402

POOL = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1")
FORCA = Path("/work/ab0995/a270088/meshes/FORCA20")
# (name, dist-dir, npes, nl) — the halo A/B ladder + the two production chains.
CONFIGS = [
    ("core2", POOL / "core2", 4, 48),           # CORE2 production chain (dist_4)
    ("core2", POOL / "core2", 8, 48),
    ("dars", POOL / "dars", 16, 48),
    ("dars", POOL / "dars", 32, 48),
    ("forca20", FORCA, 16, 80),                 # FORCA20 production chain (dist_16, live)
    ("forca20", FORCA, 32, 80),                 # what the config claims (compile-hangs)
    ("ng5", POOL / "ng5", 64, 70),
    ("ng5", POOL / "ng5", 128, 70),
]


def colour_rounds(send_sizes: np.ndarray, P: int):
    """Greedy bipartite edge-colouring of the neighbour graph (left = senders, right =
    receivers). Each colour class is a partial permutation — at most one send and one recv
    per device — i.e. exactly one legal ``lax.ppermute`` round. Konig: Delta colours suffice
    for a bipartite graph; greedy may use a few more, so we report what it ACHIEVES.

    Returns (K, per_round_slot) where per_round_slot[k] = the max chunk in colour class k:
    the coloured transport ships ``sum_k per_round_slot[k]`` lanes instead of padded's
    ``P * pad_slot``."""
    edges = [(d, e, int(send_sizes[d, e]))
             for d in range(P) for e in range(P) if send_sizes[d, e] > 0]
    edges.sort(key=lambda t: -t[2])              # big chunks first: better packing
    used_send = [set() for _ in range(P)]
    used_recv = [set() for _ in range(P)]
    rounds: dict[int, int] = {}                  # colour -> max chunk in that colour
    for d, e, n in edges:
        c = 0
        while c in used_send[d] or c in used_recv[e]:
            c += 1
        used_send[d].add(c)
        used_recv[e].add(c)
        rounds[c] = max(rounds.get(c, 0), n)
    return len(rounds), sorted(rounds.values(), reverse=True)


def kind_stats(mylists, mydim, P: int) -> dict:
    """recv_sizes[e, d] = #halo lanes on e owned by d — vectorised (the shard_mesh
    builder's own double Python loop is too slow to probe interactively)."""
    gcount = int(max(int(ml.max()) for ml in mylists)) + 1
    owner, _ = _owner_map(mylists, mydim, gcount)
    recv_sizes = np.zeros((P, P), dtype=np.int64)
    for e in range(P):
        halo = mylists[e][int(mydim[e]):]                       # halo lanes on e
        if halo.size:
            recv_sizes[e] = np.bincount(owner[halo], minlength=P)
    send_sizes = recv_sizes.T                                   # d sends to e what e recvs from d
    pad_slot = max(int(recv_sizes.max()), 1)
    send_max = int(send_sizes.sum(axis=1).max())                # ragged: real lanes shipped
    nbr = [int((recv_sizes[e] > 0).sum()) for e in range(P)]    # real neighbours per device
    K, slots = colour_rounds(send_sizes, P)                     # coloured-ppermute proposal
    col_lanes = int(sum(slots))                                 # sum_k max-chunk-in-round-k
    return dict(pad_slot=pad_slot, pad_lanes=P * pad_slot, send_max=send_max,
                pad_factor=(P * pad_slot) / max(send_max, 1),
                nbr_mean=float(np.mean(nbr)), nbr_max=int(np.max(nbr)),
                occupancy=float(recv_sizes.sum()) / (P * P * pad_slot),
                K=K, col_lanes=col_lanes,
                col_factor=col_lanes / max(send_max, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on '<name>-<npes>'")
    args = ap.parse_args()

    print("ALL VOLUMES ARE PER DEVICE, PER EXCHANGE, of ONE [Lmax, nl] f64 field.")
    print("'local field' = this device's own array (Lmax*nl*8) — it SHRINKS as ~N/P while the")
    print("padded buffer GROWS as ~P*pad_slot. Where they cross, the halo exchange puts more")
    print("bytes on the wire than the local array it is refreshing. (all_gather = global field.)\n")
    print("Last block = the PROPOSED coloured-ppermute transport (K bipartite-edge-colouring")
    print("rounds of lax.ppermute, per-round slot) — what padded would cost if it stopped")
    print("sending zero-filled slots to its ~P-7 non-neighbours.\n")
    print(f"{'config':>12} {'kind':>5} {'Lmax':>9} {'pad_slot':>9} {'pad lanes':>10} "
          f"{'ragged lanes':>13} {'pad factor':>11} {'nbrs(mean/max)':>15} {'occup':>7} "
          f"|{'ragged':>8}{'padded':>9}{'local fld':>10}{'all_gather':>11}  (MB) "
          f"|{'K':>4}{'col lanes':>10}{'col factor':>11}{'col MB':>8}")
    for name, dist, npes, nl in CONFIGS:
        tag = f"{name}-{npes}"
        if args.only and args.only not in tag:
            continue
        if not (dist / f"dist_{npes}").is_dir():
            print(f"{tag:>12}  -- no dist_{npes} under {dist}")
            continue
        part = partit.read_partition(dist, npes)
        _, Lmax = local_sizes(part)
        nod2D = int(sum(int(m) for m in part.myDim_nod2D))       # nodes are uniquely owned
        for kind, mylists, mydim in (
            ("nod", part.myList_nod2D, part.myDim_nod2D),
            ("elem", part.myList_elem2D, part.myDim_elem2D),
        ):
            s = kind_stats(mylists, mydim, npes)
            mb = lambda lanes: lanes * nl * 8 / 1e6              # noqa: E731  [lanes, nl] f64
            # padded ships its WHOLE slotted buffer (a pad_slot-sized message to every rank,
            # zeros to the ~P-7 non-neighbours); ragged ships only the real chunks; all_gather
            # ships the global field to everyone (the reference both are trying to beat).
            print(f"{tag:>12} {kind:>5} {Lmax[kind]:>9,} {s['pad_slot']:>9,} "
                  f"{s['pad_lanes']:>10,} {s['send_max']:>13,} {s['pad_factor']:>10.1f}x "
                  f"{s['nbr_mean']:>7.1f}/{s['nbr_max']:<7d} {s['occupancy']:>6.1%} "
                  f"|{mb(s['send_max']):>8.2f}{mb(s['pad_lanes']):>9.1f}"
                  f"{mb(Lmax[kind]):>10.1f}{mb(nod2D):>11.0f} "
                  f"|{s['K']:>4}{s['col_lanes']:>10,}{s['col_factor']:>10.1f}x"
                  f"{mb(s['col_lanes']):>8.2f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
