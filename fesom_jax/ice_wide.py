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


def rings(elem_nodes, owned_nodes: np.ndarray, K: int, nod2D: int | None = None):
    """Ring sets of one device: ``(R_K node ids, E_K elem ids, E_1 mask over elements,
    R_{K-1} mask over nodes)``. ``owned_nodes`` = the device's owned global node ids (R_0)."""
    en = np.asarray(elem_nodes)
    nod2D = int(en.max()) + 1 if nod2D is None else int(nod2D)
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
    rg = tuple(dict(nod=nods[r], elem=elems[r], n_std=int(ml_n[r].size),
                    e_std=int(ml_e[r].size), r_km1=km1s[r]) for r in range(P))
    return MevpWideHost(K=K, arrays=arrays, rings=rg)


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


@dataclasses.dataclass(frozen=True)
class MevpWide:
    """Per-device wide-halo context (built inside ``shard_map`` by :func:`local_mevp_wide`),
    carried on :class:`~fesom_jax.halo.HaloCtx` as ``mevp_wide``. ``skip_refresh`` (test-only
    negative control, default ``-1`` = off) drops the refresh before block ``skip_refresh+1``."""

    K: int
    a: dict
    axis_name: str
    skip_refresh: int = -1

    def entry(self, nod, elem):
        """Standard-layout ``[Lsn, cn]`` / ``[Lse, ce]`` → ice lanes, every lane its owner's value."""
        a = self.a
        return _xfer(nod, elem, a["sn"], a["ren"], a["se"], a["ree"], self.axis_name)

    def refresh(self, nod, elem):
        """Ice lanes → ice lanes: every non-interior lane gets its owner's (current) value."""
        a = self.a
        return _xfer(nod, elem, a["sn"], a["rrn"], a["se"], a["rre"], self.axis_name)

    def exit(self, nod, elem):
        """The standard halo exchange of a node and an element field, in one collective."""
        a = self.a
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
    return {k: np.ascontiguousarray(host.arrays[k]).reshape(
        (-1,) + host.arrays[k].shape[2:]) for k in WIDE_KEYS}


def local_mevp_wide(a: dict, K: int, axis_name: str = "p") -> MevpWide:
    """The per-device :class:`MevpWide` from this shard's slice of :func:`fold_arrays`."""
    return MevpWide(K=int(K), a=dict(a), axis_name=axis_name)
