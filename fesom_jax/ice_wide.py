"""A wide halo for the sharded mEVP subcycles: one exchange every K subcycles instead of one per
subcycle (CORE2 GPU speed plan, Tier A; docs/plans/20261020-core2-mevp-wide-halo.md).

The sharded mEVP (:func:`fesom_jax.ice_mevp.mevp_dynamics`) refreshes the ``u_aux``/``v_aux`` node
halo after every one of its 120 subcycles. A subcycle updates an element's stresses from its
vertices' velocities and a node's velocity from its elements' stresses, so the region a shard
updates correctly shrinks by one ring per subcycle. With K rings of "ice lanes" a shard runs K
subcycles between refreshes and recomputes the outer rings itself (communication avoiding; the
fesom3-jax ``shard/evp_wide.py`` and the Kokkos port's EVPWIDE).

Rings of device r: R_0 = its owned nodes; E_j = the elements with a vertex in R_{j-1};
R_j = R_{j-1} plus the vertices of E_j. R_1 is the standard local node set (owned + halo), E_1 the
standard interior elements. After j subcycles from a refresh the stresses are current on
E_{K-j+1} and the velocities on R_{K-j}; after K, on E_1 and R_0.

Lane order. The ice lanes of device r are its STANDARD local lanes first, in the standard order,
then the extra ring lanes in ascending global id, then pads; so standard lane i is ice lane i and an
owner's lane index for an owned entity is the same in both layouts.

Node sums (load-bearing for bitwise owned lanes). The standard mEVP sums a node's element terms
with ``ops.scatter_add``, i.e. in the order of the owner's local element lanes. An outer-ring node
on the ice lanes must get the bits its owner computes, so the wide path does not scatter: every ice
node lane carries the list of its elements' flattened ``[elem, vertex]`` slots in its OWNER's
standard lane order (``csr``), and :meth:`MevpWide.node_sum` adds them one by one from zero, as
the owner's segment-sum does.

The mEVP precompute (``mevp_setup``) runs on the standard lanes, unchanged; its outputs move to
the ice lanes with the inputs, so the ssh-tilt scatter is never recomputed in another order (as
fesom3-jax moves its precompute). The geometry the subcycle reads on the ice lanes
(``elem_nodes``, ``gradient_sca``, ``metric_factor``, ``elem_area``, ``coriolis_node``) is the
GLOBAL mesh's, gathered by global id.

Transport: the compact ``all_gather`` (:func:`fesom_jax.halo.compact_exchange_maps`), node and
element fields in ONE collective (entry, every refresh, exit).
"""

from __future__ import annotations

import dataclasses
import os
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax

from .halo import compact_exchange_maps, halo_none
from .shard_mesh import _owner_map


# --------------------------------------------------------------------------
# Host side (numpy)
# --------------------------------------------------------------------------
def mevp_wide_from_env() -> int:
    """``FESOM_MEVP_WIDE`` (read on the host by :func:`fesom_jax.run.run_from_config`; the
    ``gpu.mevp_wide`` config key sets it): the wide-halo width K, ``0`` when unset. ``K < 2`` ⇒
    the per-subcycle exchange."""
    v = os.environ.get("FESOM_MEVP_WIDE", "").strip()
    if v in ("", "0"):
        return 0
    try:
        k = int(v)
    except ValueError:
        raise ValueError(f"FESOM_MEVP_WIDE must be an integer >= 0, got {v!r}") from None
    if k < 0:
        raise ValueError(f"FESOM_MEVP_WIDE must be an integer >= 0, got {v!r}")
    return k


_N2E = [None, None]          # (elem_nodes array, its node -> element CSR): rings() of one builder reuse it


def _node_elem_csr(en: np.ndarray, nod2D: int):
    """Every node's elements (ascending), as ``(ptr [nod2D+1], elem ids)``; cached for the last ``en`` object."""
    if _N2E[0] is en and _N2E[1][0].size == nod2D + 1:
        return _N2E[1]
    flat = en.reshape(-1)
    order = np.argsort(flat, kind="stable")
    ptr = np.zeros(nod2D + 1, np.int64)
    np.cumsum(np.bincount(flat, minlength=nod2D), out=ptr[1:])
    _N2E[0], _N2E[1] = en, (ptr, (order // en.shape[1]).astype(np.int64))
    return _N2E[1]


def rings(elem_nodes, owned_nodes: np.ndarray, K: int, nod2D: int | None = None):
    """Ring sets of one device: ``(R_K node ids, E_K elem ids, E_1 mask over elements,
    R_{K-1} mask over nodes)``. ``owned_nodes`` = the device's owned global node ids (R_0).

    R_j = R_{j-1} + the vertices of E_j, E_j = the elements with a vertex in R_{j-1}, grown from the
    frontier (E_j = E_{j-1} + the elements of R_{j-1} - R_{j-2}): the work is the rings' size, not
    K sweeps over the whole mesh per device (dist_256 spent > 20 min here). Same sets as the sweep."""
    en = np.asarray(elem_nodes)
    nod2D = int(en.max()) + 1 if nod2D is None else int(nod2D)
    if K < 1 or en.size == 0 or int(en.min()) < 0:
        return _rings_sweep(en, owned_nodes, K, nod2D)
    ptr, n2e = _node_elem_csr(en, nod2D)
    nodes = np.zeros(nod2D, dtype=bool)
    front = np.unique(np.asarray(owned_nodes))
    nodes[front] = True
    elems = np.zeros(en.shape[0], dtype=bool)
    e1 = prev = None
    # FESOM_RINGS_FAST (awicm_jax lane c1279_start, default on): the new elements / nodes of a ring made unique
    # with a scratch index (last occurrence kept) instead of np.unique's sort (NG5 dist_128: ~29 s of every ocean
    # process's start in the 2 x 128 calls, job 3078725). Only the sets matter here (the masks; the id lists come
    # from them at the end): the same rings.
    fast = os.environ.get("FESOM_RINGS_FAST", "1") != "0"
    if fast:
        sc_e, sc_n = _scratch("e", en.shape[0]), _scratch("n", nod2D)
    for j in range(K):
        if j == K - 1:
            prev = nodes.copy()
        cnt = ptr[front + 1] - ptr[front]
        tot = int(cnt.sum())
        if tot:
            idx = np.repeat(ptr[front] - np.cumsum(cnt) + cnt, cnt) + np.arange(tot)
            if fast:
                ne = n2e[idx]
                ne = _first_unique(ne[~elems[ne]], sc_e)
            else:
                ne = np.unique(n2e[idx])
                ne = ne[~elems[ne]]
        else:
            ne = np.zeros(0, np.int64)
        elems[ne] = True
        if j == 0:
            e1 = elems.copy()
        if fast:
            v = en[ne].ravel()
            front = _first_unique(v[~nodes[v]], sc_n)
        else:
            v = np.unique(en[ne].ravel())
            front = v[~nodes[v]]
        nodes[front] = True
    return np.nonzero(nodes)[0], np.nonzero(elems)[0], e1, prev


_SCRATCH = {}


def _scratch(kind, n):
    a = _SCRATCH.get(kind)
    if a is None or a.size < n:
        a = _SCRATCH[kind] = np.empty(n, np.int64)
    return a


def _first_unique(a, scratch):
    """The distinct values of the int array `a` (non-negative, < scratch.size), in no particular order."""
    if a.size < 2:
        return a
    pos = np.arange(a.size)
    scratch[a] = pos
    return a[scratch[a] == pos]


def _rings_sweep(en, owned_nodes, K: int, nod2D: int):
    """:func:`rings` by K sweeps over every element (the reference; negative vertex ids)."""
    nodes = np.zeros(nod2D, dtype=bool)
    nodes[np.asarray(owned_nodes)] = True
    e1 = None
    prev = nodes
    for _ in range(K):
        prev = nodes
        elems = nodes[en].any(axis=1)
        e1 = elems if e1 is None else e1
        nodes = nodes.copy()
        nodes[en[elems].ravel()] = True
    return np.nonzero(nodes)[0], np.nonzero(elems)[0], e1, prev


def _owner_order(sm, mylist_nod, mylist_elem, nod2D: int):
    """Per global node, its elements' ``(global elem id, vertex slot)`` in the order the node's
    OWNER adds them in ``ops.scatter_add`` (= its standard local element lane order, the flattened
    ``[elem, vertex]`` update order of the segment-sum). Returns ``(order_e, order_k)``
    ``[nod2D, D]`` (``-1`` pad)."""
    P = sm.P
    myd = sm.counts["myDim_nod"]
    gs, ps, es, ks = [], [], [], []
    for d in range(P):
        en = np.asarray(sm.fields["elem_nodes"][d])          # [Lmax_e, 3] local node lanes
        flat = en.reshape(-1)
        p = np.arange(flat.size)
        m = (flat >= 0) & (flat < int(myd[d]))               # contributions to OWNED nodes
        gs.append(np.asarray(mylist_nod[d])[flat[m]])
        ps.append(p[m])
        es.append(np.asarray(mylist_elem[d])[p[m] // 3])
        ks.append(p[m] % 3)
    g = np.concatenate(gs)
    p = np.concatenate(ps)
    e = np.concatenate(es)
    k = np.concatenate(ks)
    # nodes are uniquely owned ⇒ each node's entries come from one device; order by (g, p)
    o = np.lexsort((p, g))
    g, e, k = g[o], e[o], k[o]
    deg = np.bincount(g, minlength=nod2D)
    D = int(deg.max())
    start = np.concatenate([[0], np.cumsum(deg)[:-1]])
    slot = np.arange(g.size) - start[g]
    order_e = np.full((nod2D, D), -1, dtype=np.int64)
    order_k = np.full((nod2D, D), -1, dtype=np.int64)
    order_e[g, slot] = e
    order_k[g, slot] = k
    return order_e, order_k


@dataclasses.dataclass(frozen=True)
class MevpWideHost:
    """The host-built ice-lane arrays of every device (``[P, …]`` numpy) + the width ``K``.

    ``arrays``: ``en`` [P,Le,3] ice-node lanes of each ice element's vertices (pads: 0); ``gs``
    [P,Le,6], ``mfac`` [P,Le], ``earea`` [P,Le], ``cor`` [P,Ln] (global values by id; pads copy
    the global entity 0); ``csr`` [P,Ln,D] flattened ``elem*3+vertex`` slots in the owner's
    order (``-1`` pad / missing element); the compact maps ``sn``/``ren``/``rrn`` (node send,
    entry read, refresh read), ``se``/``ree``/``rre`` (element), ``xsn``/``xrn``/``xse``/``xre``
    (the standard exit exchange), ``own_n``/``own_e`` (standard owned masks). ``rings`` keeps
    the per-device id lists for tests (``nod``/``elem`` global ids in ice-lane order, ``n_std``,
    ``r_km1`` the R_{K-1} mask over the ice node lanes)."""

    K: int
    arrays: dict
    rings: tuple
    gmeta: tuple = None      # static ((meta, Ln_out) entry, refresh, exit) of the point-to-point maps in ``arrays``


def build_mevp_wide(mesh, partition, sm, K: int) -> MevpWideHost | None:
    """Build the :class:`MevpWideHost` for width ``K`` (``None`` for one device or ``K < 2``:
    the per-subcycle exchange stays). ``mesh`` is the GLOBAL :class:`~fesom_jax.mesh.Mesh`,
    ``partition`` its :class:`~fesom_jax.partit.Partition`, ``sm`` the standard
    :class:`~fesom_jax.shard_mesh.ShardedMesh` built from them."""
    K = int(K)
    P = int(partition.npes)
    if P < 2 or K < 2:
        return None
    en_g = np.asarray(mesh.elem_nodes)
    nod2D, elem2D = int(mesh.nod2D), int(mesh.elem2D)
    ml_n = [np.asarray(x) for x in partition.myList_nod2D]
    ml_e = [np.asarray(x) for x in partition.myList_elem2D]
    myd_n = np.asarray(partition.myDim_nod2D)
    myd_e = np.asarray(partition.myDim_elem2D)
    Lsn, Lse = sm.Lmax["nod"], sm.Lmax["elem"]

    nods, elems, km1s = [], [], []
    for r in range(P):
        rn, re_, _e1, prev = rings(en_g, ml_n[r][: myd_n[r]], K, nod2D)
        std_n, std_e = ml_n[r], ml_e[r]
        in_n = np.zeros(nod2D, bool)
        in_n[rn] = True
        in_e = np.zeros(elem2D, bool)
        in_e[re_] = True
        if not in_n[std_n].all() or not in_e[std_e].all():
            raise ValueError(f"mevp_wide: device {r}'s standard lanes are not inside its K={K} "
                             "rings")
        mark_n = np.zeros(nod2D, bool)
        mark_n[std_n] = True
        mark_e = np.zeros(elem2D, bool)
        mark_e[std_e] = True
        nod = np.concatenate([std_n, rn[~mark_n[rn]]])        # standard first, then ascending
        elem = np.concatenate([std_e, re_[~mark_e[re_]]])
        nods.append(nod)
        elems.append(elem)
        km1s.append(prev[nod])
    Ln = max(Lsn, max(x.size for x in nods))
    Le = max(Lse, max(x.size for x in elems))

    order_e, order_k = _owner_order(sm, ml_n, ml_e, nod2D)
    D = order_e.shape[1]
    own_n, ol_n = _owner_map(ml_n, myd_n, nod2D)
    own_e, ol_e = _owner_map(ml_e, myd_e, elem2D)

    gs = np.asarray(mesh.gradient_sca)
    mfac = np.asarray(mesh.metric_factor)
    earea = np.asarray(mesh.elem_area)
    cor = np.asarray(mesh.coriolis_node)

    A = {name: [] for name in ("en", "gs", "mfac", "earea", "cor", "csr")}
    sd_n = np.zeros((P, Ln), np.int32)
    sl_n = np.zeros((P, Ln), np.int32)
    sd_e = np.zeros((P, Le), np.int32)
    sl_e = np.zeros((P, Le), np.int32)
    for r in range(P):
        nod, elem = nods[r], elems[r]
        g2l_n = np.full(nod2D, -1, np.int64)
        g2l_n[nod] = np.arange(nod.size)
        g2l_e = np.full(elem2D, -1, np.int64)
        g2l_e[elem] = np.arange(elem.size)
        # geometry by global id; pad lanes copy entity 0 (finite) and point at node lane 0
        ide = np.zeros(Le, np.int64)
        ide[: elem.size] = elem
        idn = np.zeros(Ln, np.int64)
        idn[: nod.size] = nod
        en = g2l_n[en_g[ide]]
        if (en[: elem.size] < 0).any():
            raise ValueError(f"mevp_wide: device {r} lacks a vertex of one of its ice elements")
        en[elem.size:] = 0
        A["en"].append(en.astype(np.int32))
        A["gs"].append(gs[ide])
        A["mfac"].append(mfac[ide])
        A["earea"].append(earea[ide])
        A["cor"].append(cor[idn])
        oe = order_e[idn]
        lane = np.where(oe >= 0, g2l_e[np.maximum(oe, 0)], -1)
        csr = np.where(lane >= 0, lane * 3 + order_k[idn], -1)
        csr[nod.size:] = -1
        A["csr"].append(csr.astype(np.int32))
        # exchange maps on the ice lanes: interior (standard owned/interior) lanes read
        # themselves, every other ice lane its owner's interior lane (the _exchange_map rule)
        sd_n[r] = r
        sd_e[r] = r
        sl_n[r, : myd_n[r]] = np.arange(myd_n[r])
        sl_e[r, : myd_e[r]] = np.arange(myd_e[r])
        hn = nod[myd_n[r]:]
        he = elem[myd_e[r]:]
        sd_n[r, myd_n[r]: nod.size] = own_n[hn]
        sl_n[r, myd_n[r]: nod.size] = ol_n[hn]
        sd_e[r, myd_e[r]: elem.size] = own_e[he]
        sl_e[r, myd_e[r]: elem.size] = ol_e[he]
    arrays = {k: np.stack(v) for k, v in A.items()}
    arrays["sn"], arrays["ren"] = compact_exchange_maps(sd_n, sl_n, field_len=Lsn)
    _, arrays["rrn"] = compact_exchange_maps(sd_n, sl_n, field_len=Ln)
    arrays["se"], arrays["ree"] = compact_exchange_maps(sd_e, sl_e, field_len=Lse)
    _, arrays["rre"] = compact_exchange_maps(sd_e, sl_e, field_len=Le)
    arrays["xsn"], arrays["xrn"] = compact_exchange_maps(*sm.exchange["nod"])
    arrays["xse"], arrays["xre"] = compact_exchange_maps(*sm.exchange["elem"])
    arrays["own_n"] = np.asarray(sm.owned_mask["nod"])
    arrays["own_e"] = np.asarray(sm.owned_mask["elem"])
    assert D == arrays["csr"].shape[2]
    # the same three exchanges point to point (wide_grouped_wanted picks them when traced)
    gmeta = (add_grouped_maps(arrays, "ent", sd_n, sl_n, sd_e, sl_e, Lsn, Ln),
             add_grouped_maps(arrays, "ref", sd_n, sl_n, sd_e, sl_e, Ln, Ln),
             add_grouped_maps(arrays, "ext", *sm.exchange["nod"], *sm.exchange["elem"], Lsn, Lsn))
    rg = tuple(dict(nod=nods[r], elem=elems[r], n_std=int(ml_n[r].size),
                    e_std=int(ml_e[r].size), r_km1=km1s[r]) for r in range(P))
    return MevpWideHost(K=K, arrays=arrays, rings=rg, gmeta=gmeta)


# --------------------------------------------------------------------------
# Device side (inside shard_map)
# --------------------------------------------------------------------------
class IceLaneMesh(NamedTuple):
    """The mesh fields the mEVP subcycle reads, on the ice lanes (duck-types
    :class:`~fesom_jax.mesh.Mesh` for :func:`~fesom_jax.ice_mevp.mevp_iterate`)."""

    nod2D: int
    elem_nodes: jnp.ndarray
    gradient_sca: jnp.ndarray
    metric_factor: jnp.ndarray
    elem_area: jnp.ndarray
    coriolis_node: jnp.ndarray


def _xfer(nod, elem, sn, rn, se, re_, axis_name):
    """One compact broadcast exchange of a node field ``[Lfn, cn]`` and an element field
    ``[Lfe, ce]`` in ONE ``all_gather``: each device ships its sent node lanes and sent element
    lanes in one buffer (channels zero-padded to the wider one); the results are read out of
    ``[field ; gathered]`` with ``rn``/``re_`` (:func:`fesom_jax.halo.compact_exchange_maps`
    with the matching ``field_len``)."""
    cn, ce = nod.shape[1], elem.shape[1]
    C = max(cn, ce)
    Sn, Se = sn.shape[0], se.shape[0]
    bn = nod[sn]
    be = elem[se]
    if cn < C:
        bn = jnp.concatenate([bn, jnp.zeros((Sn, C - cn), bn.dtype)], axis=1)
    if ce < C:
        be = jnp.concatenate([be, jnp.zeros((Se, C - ce), be.dtype)], axis=1)
    buf = jnp.concatenate([bn, be], axis=0)
    if halo_none():                                   # timing only: the own lanes stand in for every device's
        g = jnp.broadcast_to(buf[None], (lax.psum(1, axis_name),) + buf.shape)
    else:
        g = lax.all_gather(buf, axis_name, axis=0, tiled=False)
    P = g.shape[0]
    gn = g[:, :Sn, :cn].reshape(P * Sn, cn)
    ge = g[:, Sn:, :ce].reshape(P * Se, ce)
    return (jnp.concatenate([nod, gn], axis=0)[rn],
            jnp.concatenate([elem, ge], axis=0)[re_])


# --------------------------------------------------------------------------
# The point-to-point transport of the wide exchanges (fesom3_jax --halo-wide grouped)
# --------------------------------------------------------------------------
#: The keys of one wide exchange's point-to-point maps (:func:`grouped_wide_maps`), per device.
GROUPED_FIELDS = ("si", "sv", "cp", "hm", "self")


def wide_grouped_wanted() -> bool:
    """The wide exchanges go point to point when the run's halo transport is grouped (``FESOM_HALO_GROUPED=1``),
    unless ``FESOM_WIDE_GROUPED=0`` keeps the compact ``all_gather`` (A/B timing). Read when traced."""
    from . import halo_grouped
    return halo_grouped.enabled() and os.environ.get("FESOM_WIDE_GROUPED", "1").strip() not in ("", "0")


def grouped_wide_maps(sd_n, sl_n, sd_e, sl_e, Lfn: int):
    """The point-to-point maps of one wide exchange (entry, refresh or exit) from its all_gather maps
    ``(src_dev, src_lane)`` of the node lanes ``[P, Ln]`` and the element lanes ``[P, Le]`` (the ones
    :func:`fesom_jax.halo.compact_exchange_maps` takes); ``Lfn`` = the node lane count of the SENT field (the
    standard ``Lmax["nod"]`` for the entry and the exit, the wide ``Ln`` for a refresh).

    The node and the element field travel as ONE field of ``Lfn + Lfe`` rows (elements after the nodes, channels
    zero-padded to the wider one, as :func:`_xfer` packs them), so every neighbour gets one message an exchange
    (fesom3_jax's wide halos on the grouped transport, ``--halo-wide grouped``). The pairs (sender d, receiver e)
    are edge-coloured into rounds (:func:`fesom_jax.shard_mesh._colour_edges`) and the rounds packed into one buffer
    for :func:`fesom_jax.halo_grouped.exchange` (one NCCL group, only the real rows). Within a pair the rows go in
    the receiver's lane order. The compact all_gather instead delivers every device's sent lanes to every device:
    P times its send set, growing with P.

    Returns ``(maps, meta)``: ``maps`` = ``{si, sv, cp, hm, self}`` ``[P, …]`` (the send rows of the packed buffer
    and their validity, each lane's position in the received buffer, the lanes that receive, the row an own lane
    reads: itself, its standard lane, or lane 0 for a pad — the compact exchange's rule), ``meta`` = ``(perms,
    slots, offs, sendn)`` (STATIC). Every lane gets exactly the value the all_gather exchange gives it."""
    from . import halo_grouped
    from .shard_mesh import _colour_edges
    sd = np.concatenate([np.asarray(sd_n), np.asarray(sd_e)], axis=1).astype(np.int64)
    sl = np.concatenate([np.asarray(sl_n).astype(np.int64), np.asarray(sl_e).astype(np.int64) + int(Lfn)], axis=1)
    P, L = sd.shape
    remote = sd != np.arange(P)[:, None]
    e_i, l_i = np.nonzero(remote)                        # receiver, its lane
    d_i, s_i = sd[e_i, l_i], sl[e_i, l_i]                # sender, the row it sends
    order = np.lexsort((l_i, e_i, d_i))                  # by sender, receiver, receiver lane
    e_i, l_i, d_i, s_i = e_i[order], l_i[order], d_i[order], s_i[order]
    sizes = np.bincount(d_i * P + e_i, minlength=P * P).reshape(P, P)
    start = np.concatenate([[0], np.cumsum(sizes.ravel())])
    classes = _colour_edges(sizes, P)
    slots = tuple(max(n for _, _, n in cls) for cls in classes)
    offs = tuple(int(x) for x in np.cumsum((0,) + slots)[:-1])
    total = max(int(sum(slots)), 1)
    si = np.zeros((P, total), np.int32)
    sv = np.zeros((P, total), bool)
    cp = np.zeros((P, L), np.int32)
    for r, cls in enumerate(classes):
        base = offs[r]
        for d, e, n in cls:
            a = int(start[d * P + e])
            si[d, base:base + n] = s_i[a:a + n]
            sv[d, base:base + n] = True
            cp[e, l_i[a:a + n]] = base + np.arange(n, dtype=np.int32)
    maps = {"si": si, "sv": sv, "cp": cp, "hm": remote, "self": np.where(remote, 0, sl).astype(np.int32)}
    perms = tuple(tuple((d, e) for d, e, _ in cls) for cls in classes)
    return maps, (perms, slots, offs, halo_grouped.send_rows(sv, slots, offs))


def add_grouped_maps(arrays: dict, prefix: str, sd_n, sl_n, sd_e, sl_e, Lfn: int, Ln_out: int):
    """:func:`grouped_wide_maps` into ``arrays`` as ``g{prefix}_{field}``; returns the static ``(meta, Ln_out)``
    (``Ln_out`` = the node lane count of the result: where its element lanes start)."""
    maps, meta = grouped_wide_maps(sd_n, sl_n, sd_e, sl_e, Lfn)
    for k, v in maps.items():
        arrays[f"g{prefix}_{k}"] = v
    return meta, int(Ln_out)


def grouped_keys(prefixes=("ent", "ref", "ext")) -> tuple:
    """The fold keys of the point-to-point maps of the given exchanges."""
    return tuple(f"g{p}_{k}" for p in prefixes for k in GROUPED_FIELDS)


def _xfer_grouped(nod, elem, a: dict, prefix: str, gmeta, axis_name):
    """:func:`_xfer`'s exchange (the same values on every lane) point to point: one message per neighbour with its
    node and element rows, every round of the exchange in one NCCL group (:func:`fesom_jax.halo_grouped.exchange`;
    ``lax.ppermute`` rounds off the grouped transport, e.g. on the CPU). ``gmeta`` = ``(meta, Ln_out)``."""
    from . import halo_grouped
    meta, Ln = gmeta
    cn, ce = nod.shape[1], elem.shape[1]
    C = max(cn, ce)
    if cn < C:
        nod = jnp.concatenate([nod, jnp.zeros((nod.shape[0], C - cn), nod.dtype)], axis=1)
    if ce < C:
        elem = jnp.concatenate([elem, jnp.zeros((elem.shape[0], C - ce), elem.dtype)], axis=1)
    x = jnp.concatenate([nod, elem], axis=0)
    g = {k: a[f"g{prefix}_{k}"] for k in GROUPED_FIELDS}
    buf = jnp.where(g["sv"][:, None], x[g["si"]], jnp.zeros((), x.dtype))
    perms, slots, offs = meta[:3]
    if halo_none() or not perms:
        recv = buf                                       # timing only (FESOM_HALO_NONE): no transfer, wrong values
    elif halo_grouped.enabled():
        recv = halo_grouped.exchange(buf, meta)
    else:
        rounds = [lax.ppermute(buf[o:o + s], axis_name, perm=list(p)) for p, s, o in zip(perms, slots, offs)]
        recv = jnp.concatenate(rounds, axis=0) if len(rounds) > 1 else rounds[0]
    out = jnp.where(g["hm"][:, None], recv[g["cp"]], x[g["self"]])
    return out[:Ln, :cn], out[Ln:, :ce]


@dataclasses.dataclass(frozen=True)
class MevpWide:
    """Per-device wide-halo context (built inside ``shard_map`` by :func:`local_mevp_wide`),
    carried on :class:`~fesom_jax.halo.HaloCtx` as ``mevp_wide``. ``skip_refresh`` (test-only
    negative control, default ``-1`` = off) drops the refresh before block ``skip_refresh+1``."""

    K: int
    a: dict
    axis_name: str
    skip_refresh: int = -1
    gmeta: tuple = None      # ((meta, Ln_out) of the entry, the refresh, the exit); their maps are in ``a``

    def _grouped(self, i):
        return None if self.gmeta is None or not wide_grouped_wanted() else self.gmeta[i]

    def entry(self, nod, elem):
        """Standard-layout ``[Lsn, cn]`` / ``[Lse, ce]`` → ice lanes, every lane its owner's value."""
        a = self.a
        if (g := self._grouped(0)) is not None:
            return _xfer_grouped(nod, elem, a, "ent", g, self.axis_name)
        return _xfer(nod, elem, a["sn"], a["ren"], a["se"], a["ree"], self.axis_name)

    def refresh(self, nod, elem):
        """Ice lanes → ice lanes: every non-interior lane gets its owner's (current) value."""
        a = self.a
        if (g := self._grouped(1)) is not None:
            return _xfer_grouped(nod, elem, a, "ref", g, self.axis_name)
        return _xfer(nod, elem, a["sn"], a["rrn"], a["se"], a["rre"], self.axis_name)

    def exit(self, nod, elem):
        """The standard halo exchange of a node and an element field, in one collective."""
        a = self.a
        if (g := self._grouped(2)) is not None:
            return _xfer_grouped(nod, elem, a, "ext", g, self.axis_name)
        return _xfer(nod, elem, a["xsn"], a["xrn"], a["xse"], a["xre"], self.axis_name)

    def lane_mesh(self) -> IceLaneMesh:
        a = self.a
        return IceLaneMesh(nod2D=a["cor"].shape[0], elem_nodes=a["en"], gradient_sca=a["gs"],
                           metric_factor=a["mfac"], elem_area=a["earea"], coriolis_node=a["cor"])

    def node_sum(self, c):
        """Element→node sum of ``c`` ``[Le, 3]`` (term of element e for its vertex k), each node's
        terms added from zero in its owner's ``scatter_add`` order — the bits the owner's
        segment-sum gives (a trailing ``+0.0`` for a pad slot leaves a sum unchanged)."""
        csr = self.a["csr"]
        # gpu.mevp_wide_barrier (FESOM_MEVP_WIDE_BARRIER, lane ocean_speed3): the element terms computed once
        # (behind a barrier) instead of once per gathered slot inside the sum's fusion; the same additions
        if os.environ.get("FESOM_MEVP_WIDE_BARRIER", "0").strip() not in ("", "0"):
            c = lax.optimization_barrier(c)
        flat = c.reshape(-1)
        out = jnp.zeros(csr.shape[:1], c.dtype)
        for j in range(csr.shape[1]):
            idx = csr[:, j]
            out = out + jnp.where(idx >= 0, flat[jnp.maximum(idx, 0)], 0.0)
        return out


WIDE_KEYS = ("en", "gs", "mfac", "earea", "cor", "csr", "sn", "ren", "rrn", "se", "ree", "rre",
             "xsn", "xrn", "xse", "xre", "own_n", "own_e")


def fold_arrays(host: MevpWideHost) -> dict:
    """``[P, X, …]`` → ``[P*X, …]`` (the ``PartitionSpec('p')`` shard_map inputs)."""
    keys = WIDE_KEYS + (grouped_keys() if host.gmeta is not None else ())
    return {k: np.ascontiguousarray(host.arrays[k]).reshape(
        (-1,) + host.arrays[k].shape[2:]) for k in keys}


def local_mevp_wide(a: dict, K: int, axis_name: str = "p", gmeta: tuple = None) -> MevpWide:
    """The per-device :class:`MevpWide` from this shard's slice of :func:`fold_arrays`; ``gmeta`` = the host's
    (:attr:`MevpWideHost.gmeta`) for the point-to-point exchanges (None: the compact all_gather)."""
    return MevpWide(K=int(K), a=dict(a), axis_name=axis_name, gmeta=gmeta)
