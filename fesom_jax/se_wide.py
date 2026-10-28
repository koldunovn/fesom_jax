"""A wide halo for the sharded split-explicit substeps: one exchange every K substeps instead of two per substep
(plan docs/plans/20261021-core2-split-explicit.md, Task 5; ``gpu.se_wide: K``, ``FESOM_SE_WIDE``).

The per-substep sharded SE (:func:`fesom_jax.ssh_se.se_block` with the step's exchange) refreshes eta^{m+1} and
U^{m+1} after each of its M substeps. A substep updates a node's eta from its elements' U* and an element's U from
its vertices' eta and, through the viscosity, its edge neighbours' U; so the region a shard updates correctly shrinks
by one ring per substep, and an element reads one ring further than in the mEVP. With K+1 rings of "SE lanes" a shard
runs K substeps between refreshes (fesom3-jax ``shard/se_wide.py``: "the EVP's rings plus one"; the Kokkos port's
M12b plan; Sergey's notes "a wide halo as in the mEVP").

Rings and lane order as :mod:`fesom_jax.ice_wide` (R_0 = the owned nodes, E_j = the elements with a vertex in
R_{j-1}, R_j = R_{j-1} plus the vertices of E_j), with K+1 instead of K: the STANDARD local lanes first, in the
standard order, then the extra ring lanes in ascending global id, then pads. So the standard layout is a prefix of
the SE lanes and leaving the window is a slice plus the standard exchange.

Bitwise owned lanes: the substeps on the SE lanes are :func:`fesom_jax.ssh_se._substep` itself (exchanges the
identity), and the SE tables on the SE lanes are the global ones mapped to the lanes (a node's divergence row in
global element order, an element's neighbours in ascending global edge id), so an owned value is computed from the
same values in the same order as on the standard lanes. The per-step constants (F, H0_e, H_e, the drag, the water
flux) are computed on the standard lanes and moved with the rings (one collective at the entry), <<U>> is
accumulated per lane (exact on the owned lanes, made valid on the halo lanes by the exit exchange).

Transport: the compact ``all_gather`` of ice_wide (node and element fields in ONE collective: entry, every refresh,
exit) — ceil(M/K) + 1 collectives per step instead of 2M.
"""

from __future__ import annotations

import dataclasses
import os
import types

import jax.numpy as jnp
import numpy as np
from jax import lax

from . import precision as PR
from .halo import compact_exchange_maps
from .ice_wide import _xfer, rings
from .shard_mesh import _owner_map


def se_wide_from_env() -> int:
    """``FESOM_SE_WIDE`` (the ``gpu.se_wide`` config key sets it): the width K, ``0`` when unset (the per-substep
    exchange)."""
    v = os.environ.get("FESOM_SE_WIDE", "").strip()
    if v in ("", "0"):
        return 0
    try:
        k = int(v)
    except ValueError:
        raise ValueError(f"FESOM_SE_WIDE must be an integer >= 0, got {v!r}") from None
    if k < 0:
        raise ValueError(f"FESOM_SE_WIDE must be an integer >= 0, got {v!r}")
    return k


@dataclasses.dataclass(frozen=True)
class SeWideHost:
    """The host-built SE-lane arrays of every device (``[P, …]`` numpy) + the width ``K``: geometry by global id
    (``en`` SE-node lanes of each SE element's vertices, ``gs``, ``cor``, ``earea``, ``ule``, ``uln``), the SE tables
    on the lanes (``div_elem``/``div_c``, ``nb``, ``kappa``), the compact maps (``sn``/``ren``/``rrn`` node send,
    entry read, refresh read; ``se``/``ree``/``rre`` element; ``xsn``/``xrn``/``xse``/``xre`` the standard exit
    exchange) and ``Lsn``/``Lse`` (the standard widths). ``rings`` keeps the per-device id lists for tests."""

    K: int
    arrays: dict
    rings: tuple


def build_se_wide(mesh, partition, sm, K: int) -> SeWideHost | None:
    """Build the :class:`SeWideHost` for width ``K`` (``None`` for one device or ``K < 2``). ``mesh`` = the GLOBAL
    mesh, ``partition`` its Partition, ``sm`` the standard ShardedMesh."""
    from .ssh_se import build_tables
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
    nods, elems = [], []
    for r in range(P):
        rn, re_, _e1, _prev = rings(en_g, ml_n[r][: myd_n[r]], K + 1, nod2D)
        in_n = np.zeros(nod2D, bool)
        in_n[rn] = True
        in_e = np.zeros(elem2D, bool)
        in_e[re_] = True
        if not in_n[ml_n[r]].all() or not in_e[ml_e[r]].all():
            raise ValueError(f"se_wide: device {r}'s standard lanes are not inside its K+1={K + 1} rings")
        mark_n = np.zeros(nod2D, bool)
        mark_n[ml_n[r]] = True
        mark_e = np.zeros(elem2D, bool)
        mark_e[ml_e[r]] = True
        nods.append(np.concatenate([ml_n[r], rn[~mark_n[rn]]]))
        elems.append(np.concatenate([ml_e[r], re_[~mark_e[re_]]]))
    Ln = max(Lsn, max(x.size for x in nods))
    Le = max(Lse, max(x.size for x in elems))
    t = build_tables(mesh)
    own_n, ol_n = _owner_map(ml_n, myd_n, nod2D)
    own_e, ol_e = _owner_map(ml_e, myd_e, elem2D)
    gs, cor, earea = (np.asarray(mesh.gradient_sca), np.asarray(mesh.coriolis), np.asarray(mesh.elem_area))
    ule, uln = np.asarray(mesh.ulevels), np.asarray(mesh.ulevels_nod2D)
    D = t["se_div_elem"].shape[1]
    A = {name: [] for name in ("en", "gs", "cor", "earea", "ule", "uln", "div_elem", "div_c", "nb", "kappa")}
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
        ide = np.zeros(Le, np.int64)
        ide[: elem.size] = elem
        idn = np.zeros(Ln, np.int64)
        idn[: nod.size] = nod
        en = g2l_n[en_g[ide]]
        if (en[: elem.size] < 0).any():
            raise ValueError(f"se_wide: device {r} lacks a vertex of one of its SE elements")
        en[elem.size:] = 0
        A["en"].append(en.astype(np.int32))
        A["gs"].append(gs[ide])
        A["cor"].append(cor[ide])
        A["earea"].append(earea[ide])
        A["ule"].append(ule[ide].astype(np.int32))
        A["uln"].append(uln[idn].astype(np.int32))
        el, c = t["se_div_elem"][idn], t["se_div_c"][idn]
        loc = g2l_e[el]
        c = np.where((loc < 0)[..., None], 0.0, c)
        c[nod.size:] = 0.0
        A["div_elem"].append(np.where(loc >= 0, loc, 0).astype(np.int32))
        A["div_c"].append(c)
        nb = t["se_elem_nb"][ide]
        lnb = np.where(nb >= 0, g2l_e[np.maximum(nb, 0)], -1)
        lnb[elem.size:] = -1
        A["nb"].append(lnb.astype(np.int32))
        A["kappa"].append(np.where(lnb >= 0, t["se_kappa"][ide], 0.0))
        # exchange maps on the SE lanes (the ice_wide rule): interior lanes read themselves, every other lane its
        # owner's interior lane
        sd_n[r] = r
        sd_e[r] = r
        sl_n[r, : myd_n[r]] = np.arange(myd_n[r])
        sl_e[r, : myd_e[r]] = np.arange(myd_e[r])
        hn, he = nod[myd_n[r]:], elem[myd_e[r]:]
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
    assert arrays["div_elem"].shape[2] == D
    rg = tuple(dict(nod=nods[r], elem=elems[r]) for r in range(P))
    return SeWideHost(K=K, arrays=arrays, rings=rg)


WIDE_KEYS = ("en", "gs", "cor", "earea", "ule", "uln", "div_elem", "div_c", "nb", "kappa",
             "sn", "ren", "rrn", "se", "ree", "rre", "xsn", "xrn", "xse", "xre")


def fold_arrays(host: SeWideHost) -> dict:
    """``[P, X, …]`` → ``[P*X, …]`` (the ``PartitionSpec('p')`` shard_map inputs)."""
    return {k: np.ascontiguousarray(host.arrays[k]).reshape((-1,) + host.arrays[k].shape[2:]) for k in WIDE_KEYS}


@dataclasses.dataclass(frozen=True)
class SeWide:
    """Per-device wide-halo context (built inside ``shard_map`` by :func:`local_se_wide`), carried on
    :class:`~fesom_jax.halo.HaloCtx` as ``se_wide``. ``skip_refresh`` (test-only negative control, default ``-1`` =
    off) leaves the rings unrefreshed before window ``skip_refresh + 1``."""

    K: int
    a: dict
    axis_name: str
    skip_refresh: int = -1

    def lane_mesh(self):
        """The mesh fields :func:`fesom_jax.ssh_se._substep` reads, on the SE lanes."""
        a = self.a
        return types.SimpleNamespace(elem_nodes=a["en"], gradient_sca=PR.wp(a["gs"]), coriolis=PR.wp(a["cor"]),
                                     elem_area=PR.wp(a["earea"]), ulevels=a["ule"], ulevels_nod2D=a["uln"],
                                     se_div_elem=a["div_elem"], se_div_c=PR.wp(a["div_c"]), se_elem_nb=a["nb"],
                                     se_kappa=PR.wp(a["kappa"]))

    def _x(self, nod, elem, s_n, r_n, s_e, r_e):
        return _xfer(nod, elem, self.a[s_n], self.a[r_n], self.a[s_e], self.a[r_e], self.axis_name)

    def subcycle(self, init, consts, cfg, tauM, invM, w):
        """The ``cfg.M`` substeps from ``init`` (standard layout: eta^n, the rings, <<U>> = 0) with the per-step
        ``consts = (Fx, Fy, H0e, He, kdrag | None, wf)`` (standard layout, owned lanes valid): entry to the SE
        lanes, windows of K substeps with a refresh of the rings before each window after the first, exit (slice
        to the standard prefix + the standard exchange). Returns the final carry in the standard layout."""
        from .ssh_se import _Consts, _scan
        Fx, Fy, H0e, He, kdrag, wf = consts
        etan, eta1, eta2, U0, U1, U2, _um = init
        Lsn, Lse = etan.shape[0], U0.shape[0]
        ecols = [U0, U1, U2, Fx[:, None], Fy[:, None], H0e[:, None], He[:, None]]
        if kdrag is not None:
            ecols.append(kdrag[:, None])
        xn, xe = self._x(jnp.stack([etan, eta1, eta2, wf], axis=-1), jnp.concatenate(ecols, axis=-1),
                         "sn", "ren", "se", "ree")
        k = _Consts(xe[:, 6], xe[:, 7], xe[:, 8], xe[:, 9], xe[:, 0:2], xe[:, 10] if kdrag is not None else None,
                    xn[:, 3])
        lmesh = self.lane_mesh()
        ident = lambda f, kind: f                       # noqa: E731
        K, M = self.K, int(cfg.M)

        def refresh(c):
            n, e = self._x(jnp.stack(c[0:3], axis=-1), jnp.concatenate(c[3:6], axis=-1), "sn", "rrn", "se", "rre")
            return (n[:, 0], n[:, 1], n[:, 2], e[:, 0:2], e[:, 2:4], e[:, 4:6], c[6])

        c = (xn[:, 0], xn[:, 1], xn[:, 2], xe[:, 0:2], xe[:, 2:4], xe[:, 4:6], jnp.zeros_like(xe[:, 0:2]))
        # Block lengths: first, nwin x K, rest — none longer than K. ⚠️ No block of ONE substep: XLA removes a
        # trip-count-1 loop and inlines its body, which then rounds differently (measured: K = 7, M = 50, rest 1 ⇒
        # 1-ulp differences on 2.3 % of the owned U ring; the mEVP wide halo found the same). A remainder of 1
        # shortens the first block by one instead.
        first = min(K, M)
        nwin = (M - first) // K
        rest = M - first - nwin * K
        if rest == 1 and first > 2:
            first -= 1
            nwin = (M - first) // K
            rest = M - first - nwin * K
        if M > 1 and 1 in (first, rest):
            raise NotImplementedError(f"se_wide: K={K} with M={M} substeps leaves a 1-substep block, which XLA inlines "
                                      "(no longer bitwise the per-substep path); pick another K")
        c = _scan(c, first, k, lmesh, cfg, tauM, invM, w, ident)
        skip = self.skip_refresh

        def window(cj, j):
            r = refresh(cj)
            if skip >= 0:
                r = tuple(jnp.where(j == skip, a, b) for a, b in zip(cj, r))
            return _scan(r, K, k, lmesh, cfg, tauM, invM, w, ident), None
        if nwin > 0:
            c, _ = lax.scan(window, c, jnp.arange(nwin))
        if rest > 0:
            c = _scan(refresh(c), rest, k, lmesh, cfg, tauM, invM, w, ident)
        n, e = self._x(jnp.stack([x[:Lsn] for x in c[0:3]], axis=-1),
                       jnp.concatenate([x[:Lse] for x in c[3:7]], axis=-1), "xsn", "xrn", "xse", "xre")
        return (n[:, 0], n[:, 1], n[:, 2], e[:, 0:2], e[:, 2:4], e[:, 4:6], e[:, 6:8])


def local_se_wide(a: dict, K: int, axis_name: str = "p", skip_refresh: int = -1) -> SeWide:
    """The per-device :class:`SeWide` from this shard's slice of :func:`fold_arrays`."""
    return SeWide(K=int(K), a=dict(a), axis_name=axis_name, skip_refresh=int(skip_refresh))
