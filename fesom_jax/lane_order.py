"""The order of the lanes inside the shards (``gpu.lane_order``, CORE2 GPU speed plan).

A FESOM partition lists every rank's nodes, elements and edges by global id, and the stored CORE2 numbering is
practically random: on a device an edge's two nodes lie thousands of lanes apart (fesom3-jax measured a median of
5,319 of 32,517 lanes on dist_4), so the gathers of the edge and element loops miss the cache. fesom3-jax renumbered
the mesh itself (Hilbert or reverse Cuthill-McKee, -3.6 % on 4 GH200). Here the global numbering stays: the mesh, the
partition files, the restarts and the output are unchanged, and only the order of each rank's lists, and with it the
order of the per-device arrays, changes. :func:`reorder_lanes` keeps every rank's sets and its blocks (owned, halo,
and for elements the extended halo) and orders each block: nodes by reverse Cuthill-McKee on the rank's edge graph,
elements by their smallest node lane, edges by their two node lanes. Every per-device structure is built from the
lists (``shard_mesh.build_sharded_mesh``, the state, forcing, SSH operator and restart partitioning), so the model is
the same up to the order of its sums (round-off).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .partit import ComStruct, Partition

METHODS = ("rcm",)


def _rcm(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reverse Cuthill-McKee order of the ``n``-node graph with the edges ``(a, b)`` (local lanes)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    g = coo_matrix((np.ones(2 * a.size, np.int8), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    return np.asarray(reverse_cuthill_mckee(g, symmetric_mode=True), dtype=np.int64)


def _by_blocks(key_order: np.ndarray, bounds) -> np.ndarray:
    """``key_order`` (a permutation of the lanes) split into the blocks ``bounds`` (each block keeps the lanes it had,
    in the order they appear in ``key_order``)."""
    out = []
    for lo, hi in bounds:
        sel = key_order[(key_order >= lo) & (key_order < hi)]
        out.append(sel)
    return np.concatenate(out)


def _remap_com(c: ComStruct, new_pos: np.ndarray) -> ComStruct:
    return dataclasses.replace(c, rlist=new_pos[np.asarray(c.rlist)].astype(np.asarray(c.rlist).dtype),
                               slist=new_pos[np.asarray(c.slist)].astype(np.asarray(c.slist).dtype))


def reorder_lanes(part: Partition, mesh, method: str = "rcm") -> Partition:
    """``part`` with every rank's lists in the cache-friendly order ``method`` (see the module docstring); the
    communicators' local lanes are remapped with them."""
    if method not in METHODS:
        raise ValueError(f"lane_order must be one of {METHODS}, got {method!r}")
    if part.npes == 1:
        return part
    edges_g = np.asarray(mesh.edges)
    elem_nodes_g = np.asarray(mesh.elem_nodes)
    nod, elem, edge = [], [], []
    com_nod, com_elem, com_elem_full = [], [], []
    for d in range(part.npes):
        ln = np.asarray(part.myList_nod2D[d])
        md = int(part.myDim_nod2D[d])
        g2l = np.full(mesh.nod2D, -1, np.int64)
        g2l[ln] = np.arange(ln.size)
        # nodes: RCM on the rank's edges (both ends on the rank), owned block first, then the halo
        le = np.asarray(part.myList_edge2D[d])
        ab = g2l[edges_g[le]]
        ok = (ab >= 0).all(axis=1)
        order = _rcm(ln.size, ab[ok, 0], ab[ok, 1])
        new_n = _by_blocks(order, ((0, md), (md, ln.size)))
        pos_n = np.empty(ln.size, np.int64)
        pos_n[new_n] = np.arange(ln.size)                 # old lane -> new lane
        nod.append(ln[new_n])
        g2new = np.full(mesh.nod2D, np.iinfo(np.int64).max, np.int64)
        g2new[ln] = pos_n
        # elements: by their smallest new node lane, within the owned / halo / extended-halo blocks
        lel = np.asarray(part.myList_elem2D[d])
        mde, ede = int(part.myDim_elem2D[d]), int(part.eDim_elem2D[d])
        key = g2new[elem_nodes_g[lel]].min(axis=1)
        order_e = np.argsort(key, kind="stable")
        new_e = _by_blocks(order_e, ((0, mde), (mde, mde + ede), (mde + ede, lel.size)))
        pos_e = np.empty(lel.size, np.int64)
        pos_e[new_e] = np.arange(lel.size)
        elem.append(lel[new_e])
        # edges: by (smaller, larger) new node lane, within the owned / halo blocks
        mdd = int(part.myDim_edge2D[d])
        ends = np.sort(g2new[edges_g[le]], axis=1)
        order_d = np.lexsort((ends[:, 1], ends[:, 0]))
        new_d = _by_blocks(order_d, ((0, mdd), (mdd, le.size)))
        edge.append(le[new_d])
        com_nod.append(_remap_com(part.com_nod2D[d], pos_n))
        com_elem.append(_remap_com(part.com_elem2D[d], pos_e))
        com_elem_full.append(_remap_com(part.com_elem2D_full[d], pos_e))
    return dataclasses.replace(part, myList_nod2D=tuple(nod), myList_elem2D=tuple(elem), myList_edge2D=tuple(edge),
                               com_nod2D=tuple(com_nod), com_elem2D=tuple(com_elem),
                               com_elem2D_full=tuple(com_elem_full))


def edge_lane_distance(part: Partition, mesh) -> np.ndarray:
    """Per rank, the median |lane(a) - lane(b)| over the rank's edges with both nodes on it (the locality measure)."""
    edges_g = np.asarray(mesh.edges)
    out = []
    for d in range(part.npes):
        ln = np.asarray(part.myList_nod2D[d])
        g2l = np.full(mesh.nod2D, -1, np.int64)
        g2l[ln] = np.arange(ln.size)
        ab = g2l[edges_g[np.asarray(part.myList_edge2D[d])]]
        ab = ab[(ab >= 0).all(axis=1)]
        out.append(np.median(np.abs(ab[:, 0] - ab[:, 1])))
    return np.asarray(out)
