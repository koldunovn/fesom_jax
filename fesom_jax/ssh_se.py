"""Split-explicit (SE) barotropic solver: Sergey Danilov's Shchepetkin-McWilliams AB3-AM4 subcycling on the FESOM2
path, as the Kokkos port built it (``port_kokkos/src/fesom_ssh_se.cpp`` @ 29e443b, ``docs/SSH_SE_M12.md``; Sergey's
notes ``ssh_sergey/subcycling.tex``; Banerjee et al. 2024, GMD 17, 7051). Plan:
``docs/plans/20261021-core2-split-explicit.md``.

**A stated deviation from upstream FESOM2.** FESOM2 2.7 has a different split-explicit scheme (the dissipative
forward-backward one of Demange et al. 2019 on a transport momentum path, ``oce_ale_ssh_splitexpl_subcycl.F90``); the
user chose Sergey's (2026-10-21). Opt-in physics: ``AleConfig.split_explicit = SeConfig(...)`` replaces substeps 8-12
of :func:`fesom_jax.step.step` (SSH rhs, CG, update_vel, hbar, eta blend); ``None`` (the default) is the CG path,
byte-identical. zstar only (the Kokkos port's ``se`` requires ``FESOM_ALE=zstar``).

Per baroclinic step tau = dt, after the full momentum predictor (substeps 5-7 unchanged; ``du`` = their increment):

    R    = sum_k h_k du_k / tau,  U^n_now = sum_k h_k u_k,  H_e = sum_k h_k                  (layer order, from 0)
    U_AB2 = w_new U^n_now + w_old U^{n-1}_now   (compute_vel_rhs's AB2 Coriolis weights; w_old = 0 on the first step)
    F    = R - f e_z x U_AB2 + g H_e grad(eta^n)           (the r-term: the n-level Coriolis and SSH gradient out)
    H0_e = H_e - mean_3(eta^n)
    M substeps, dtbt = tau/M, m = 0..M-1:
      U*      = (3/2+beta) U^m - (1/2+2 beta) U^{m-1} + beta U^{m-2}                                (AB3)
      <<U>>  += U* (1/M)
      eta^{m+1} = eta^m + dtbt (T(U*) - W)            (T: the divergence as a per-node gather, W the water flux)
      eta^A   = delta eta^{m+1} + (1-delta-gamma-zeta) eta^m + gamma eta^{m-1} + zeta eta^{m-2}     (AM4)
      U^{m+1} = U^m + dtbt (f e_z x U* ... = (f V*, -f U*) - g H^A grad(eta^A) + F + V[U^m] - V[U^n])
                H^A = H0_e + mean_3(eta^A),  V the live two-term harmonic viscosity of the Zenodo closure
    trim:  u_k = u_k + du_k - (sum_j h_j (u_j + du_j) - <<U>>)/H_e    (h^n weights: sum_k h_k u_k = <<U>> exactly)
    eta_n = hbar = eta^M,  hbar_old = hbar of the step start;  the zstar tail (substep 13 onward) is unchanged

``beta = 0.281105`` and ``zeta, gamma, delta`` from the dissipation parameter ``chi`` (the notes' eq. 4_7). Both rings
carry across steps (``State.se_eta``, ``se_ubt``, ``se_sbar_old`` = U^{n-1}_now), flat at a cold start
(:func:`cold_start`). Every expression keeps the Kokkos kernels' operation order; the per-node divergence is the
Kokkos port's gather CSR (rows sorted by global element id, the coefficient of a (node, element) pair the sum of its
two edge terms, pre-multiplied by 1/areasvol), and the viscosity visits an element's edge neighbours in ascending
global edge id (the Kokkos port's single-rank order). So the substeps are partition-independent: a shard's owned
lanes get the dense run's bits.

Switches (each a stated deviation from the Kokkos oracle, off in the oracle-gate configuration):

* ``bt_drag`` — Sergey's barotropic bottom drag in the substeps (his answer of 2026-10-18; FESOM_F
  ``oce_ssh_se.F90``): ``- (C_d |u_b| / H_e) (U^m - U^n)``, ``u_b`` the carried bottom-layer velocity; the 3-D drag
  stays in ``F``. The oracle has no drag.
* ``visc = "linear"`` — FESOM_F's linear viscosity ``A_h = visc_ah * l`` on ``U^m - U^n`` (``se_kappa_build``,
  ``se_visc``) instead of the two-term Zenodo closure; ``"off"`` drops the viscosity.

Precision: the working precision throughout (float32 under ``precision: single``), as upstream's SE file has no
``WP_full`` (FESOM2 #1207); ``hnode``/``hnode_new`` stay float64 in the zstar tail as everywhere.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import precision as PR
from .config import C_D, CYCLIC_LENGTH_RAD, G, R_EARTH

BETA = 0.281105                       # subcycling.tex eq. 4_7 (the Kokkos port's wbeta)
_EPS_AB = 0.1                         # compute_vel_rhs's AB2 offset (momentum._EPS_AB)
VISC_MODES = ("two_term", "linear", "off")


class SeConfig(NamedTuple):
    """Static split-explicit configuration (hashable; rides inside :class:`~fesom_jax.ale.AleConfig` as
    ``split_explicit``, the ``use_wsplit`` precedent: both are FESOM2 ``dynamics_general`` switches).

    ``M`` substeps per baroclinic step (the Kokkos port's ``FESOM_SE_M``, default 50) and the dissipation ``chi``
    (``FESOM_SE_CHI``, 0.1: the Kokkos port's CORE2 measurement, 0.05 unstable). ``visc``: ``"two_term"`` = the Kokkos
    port's live V[U^m] - V[U^n] with ``sqrt(max(gamma0, gamma1 |dU|/H) sqrt(A_e + A_nb))`` per edge
    (``FESOM_SE_VISC_GAMMA0/1``, Zenodo 10 / 2750), ``"linear"`` = FESOM_F's ``A_h = visc_ah * l`` on U^m - U^n,
    ``"off"`` = none. ``bt_drag`` = Sergey's barotropic bottom drag (off in the oracle). ``m_force`` runs below the
    CFL guard's ``M_min`` (``FESOM_SE_M_FORCE``)."""

    M: int = 50
    chi: float = 0.1
    visc: str = "two_term"
    gamma0: float = 10.0
    gamma1: float = 2750.0
    visc_ah: float = 0.01
    bt_drag: bool = False
    m_force: bool = False

    def validate(self) -> "SeConfig":
        """The Kokkos port's knob checks (``fesom_se_m``/``fesom_se_chi``: an out-of-range value aborts)."""
        if isinstance(self.M, bool) or not isinstance(self.M, int) or not 1 <= self.M <= 100000:
            raise ValueError(f"split_explicit.M must be an integer in [1, 100000], got {self.M!r}")
        if not 0.0 < float(self.chi) < 0.5:
            raise ValueError(f"split_explicit.chi must be in (0, 0.5), got {self.chi!r}")
        if self.visc not in VISC_MODES:
            raise ValueError(f"split_explicit.visc must be one of {VISC_MODES}, got {self.visc!r}")
        if self.gamma0 < 0.0 or self.gamma1 < 0.0 or self.visc_ah < 0.0:
            raise ValueError("split_explicit.gamma0, gamma1 and visc_ah must be >= 0")
        return self


def sm_coeffs(chi):
    """``(beta, zeta, gamma, delta)`` of the AB3-AM4 substeps for the dissipation ``chi`` (subcycling.tex eq. 4_7;
    the Kokkos port's ``fesom_se_step``)."""
    zeta = 0.00976186 - 0.13451357 * chi
    gamma = 0.083445 - 0.513584 * chi
    delta = 0.5 + 2.0 * zeta + gamma + 2.0 * chi
    return BETA, zeta, gamma, delta


def _weights(chi):
    """AB3 ``(a0, a1, a2)`` on (U^m, U^{m-1}, U^{m-2}) and AM4 ``(d3, d0, d1, d2)`` on (eta^{m+1}, eta^m, eta^{m-1},
    eta^{m-2})."""
    beta, zeta, gamma, delta = sm_coeffs(chi)
    return ((1.5 + beta, -(0.5 + 2.0 * beta), beta),
            (delta, 1.0 - delta - gamma - zeta, gamma, zeta))


# --------------------------------------------------------------------------
# Host tables (numpy, once per mesh): the divergence CSR, the edge neighbours, the linear-viscosity kappa
# --------------------------------------------------------------------------
TABLE_FIELDS = ("se_div_elem", "se_div_c", "se_elem_nb", "se_kappa", "se_div_inv")


def _trim_cyclic(x):
    half = 0.5 * CYCLIC_LENGTH_RAD
    x = np.where(x > half, x - CYCLIC_LENGTH_RAD, x)
    return np.where(x < -half, x + CYCLIC_LENGTH_RAD, x)


def build_tables(mesh) -> dict:
    """The global SE tables (host numpy) of a dense :class:`~fesom_jax.mesh.Mesh`:

    * ``se_div_elem`` ``[nod2D, D]`` int32 / ``se_div_c`` ``[nod2D, D, 2]``: node n's divergence row
      ``T_n = sum_k c_k . U[el_k]`` (``se_build_operators``): every edge (n1, n2; el1, el2; cross vectors
      (dx1, dy1), (dx2, dy2)) adds (-dy1, dx1) to (n1, el1), (dy1, -dx1) to (n2, el1), (dy2, -dx2) to (n1, el2) and
      (-dy2, dx2) to (n2, el2); a (node, element) pair gets exactly two such terms; the row is sorted by element id
      and multiplied by 1/areasvol[n, top]; cavity rows empty. Pads: element 0, coefficient 0.
    * ``se_elem_nb`` ``[elem2D, 3]`` int32: an element's edge neighbours in ascending edge id (``-1`` = coast).
    * ``se_kappa`` ``[elem2D, 3]``: FESOM_F's ``l*l/(d*A_e)`` per neighbour slot (``se_kappa_build`` without the
      ``visc_ah`` factor; ``l`` the edge length, ``d`` the centroid distance, the longitude metric the mean
      ``elem_cos`` of the two elements; 0 on a coast slot).
    * ``se_div_inv`` ``[elem2D, Dinv]`` int32: the divergence rows' inverse (:func:`div_inverse`), for the explicit
      transpose of :func:`divergence` (``FESOM_SE_DIV_VJP``)."""
    edges = np.asarray(mesh.edges).astype(np.int64)
    et = np.asarray(mesh.edge_tri).astype(np.int64)
    cr = np.asarray(mesh.edge_cross_dxdy).astype(np.float64)
    nod2D, elem2D = int(mesh.nod2D), int(mesh.elem2D)
    ulev_n = np.asarray(mesh.ulevels_nod2D)
    n1, n2, el1, el2 = edges[:, 0], edges[:, 1], et[:, 0], et[:, 1]
    dx1, dy1, dx2, dy2 = cr[:, 0], cr[:, 1], cr[:, 2], cr[:, 3]
    # the four adds per edge, in the Kokkos port's order (add() skips a missing element)
    nn = np.concatenate([n1, n2, n1, n2])
    ee = np.concatenate([el1, el1, el2, el2])
    cx = np.concatenate([-dy1, dy1, dy2, -dy2])
    cy = np.concatenate([dx1, -dx1, -dx2, dx2])
    ok = ee >= 0
    nn, ee, cx, cy = nn[ok], ee[ok], cx[ok], cy[ok]
    key = nn * elem2D + ee
    uk, inv = np.unique(key, return_inverse=True)        # sorted by (node, element id)
    sx = np.zeros(uk.size)
    sy = np.zeros(uk.size)
    np.add.at(sx, inv, cx)                               # 0 + a + b = a + b: the two terms, order-free
    np.add.at(sy, inv, cy)
    cnt = np.bincount(inv, minlength=uk.size)
    if (cnt != 2).any():
        raise ValueError(f"ssh_se tables: {int((cnt != 2).sum())} (node, element) pairs without exactly two edge "
                         "terms (a node of a triangle lies on two of its edges)")
    rn, re_ = uk // elem2D, uk % elem2D
    keep = ulev_n[rn] == 1                               # cavity rows empty (T = 0, the hbar guard)
    rn, re_, sx, sy = rn[keep], re_[keep], sx[keep], sy[keep]
    top = ulev_n - 1
    av = np.asarray(mesh.areasvol)[np.arange(nod2D), top].astype(np.float64)
    inv_a = 1.0 / av[rn]
    deg = np.bincount(rn, minlength=nod2D)
    D = max(int(deg.max()), 1)
    start = np.cumsum(deg) - deg
    slot = np.arange(rn.size) - start[rn]
    div_elem = np.zeros((nod2D, D), np.int32)
    div_c = np.zeros((nod2D, D, 2), np.float64)
    div_elem[rn, slot] = re_
    div_c[rn, slot, 0] = sx * inv_a
    div_c[rn, slot, 1] = sy * inv_a

    # edge neighbours: each element's three edges in ascending edge id
    eid = np.arange(edges.shape[0])
    own = np.concatenate([el1, el2])
    oth = np.concatenate([el2, el1])
    ed = np.concatenate([eid, eid])
    ok = own >= 0
    own, oth, ed = own[ok], oth[ok], ed[ok]
    o = np.lexsort((ed, own))
    own, oth, ed = own[o], oth[o], ed[o]
    ne = np.bincount(own, minlength=elem2D)
    if (ne != 3).any():
        raise ValueError(f"ssh_se tables: {int((ne != 3).sum())} elements without three edges")
    elem_nb = oth.reshape(elem2D, 3).astype(np.int32)
    elem_ed = ed.reshape(elem2D, 3)

    # FESOM_F's linear-viscosity geometry, per slot (without the visc_ah factor)
    nb_safe = np.maximum(elem_nb, 0)
    ecos = np.asarray(mesh.elem_cos).astype(np.float64)
    cxe = np.asarray(mesh.elem_center_x).astype(np.float64)
    cye = np.asarray(mesh.elem_center_y).astype(np.float64)
    coord = np.asarray(mesh.coord_nod2D).astype(np.float64)
    e_idx = np.arange(elem2D)[:, None]
    cosm = 0.5 * (ecos[e_idx] + ecos[nb_safe])
    a1 = _trim_cyclic(coord[edges[elem_ed, 1], 0] - coord[edges[elem_ed, 0], 0])
    a2 = coord[edges[elem_ed, 1], 1] - coord[edges[elem_ed, 0], 1]
    ln = np.sqrt((a1 * cosm * R_EARTH) ** 2 + (a2 * R_EARTH) ** 2)
    b1 = _trim_cyclic(cxe[nb_safe] - cxe[e_idx])
    b2 = cye[nb_safe] - cye[e_idx]
    d = np.sqrt((b1 * cosm * R_EARTH) ** 2 + (b2 * R_EARTH) ** 2)
    area = np.asarray(mesh.elem_area).astype(np.float64)[:, None]
    coast = elem_nb < 0
    kappa = np.where(coast, 0.0, ln * ln / (np.where(coast, 1.0, d) * area))
    return dict(se_div_elem=div_elem, se_div_c=div_c, se_elem_nb=elem_nb, se_kappa=kappa,
                se_div_inv=div_inverse(div_elem, div_c, elem2D))


def div_inverse(div_elem, div_c, n_elem: int, width: int | None = None):
    """The inverse of the divergence rows (host numpy, ``ops.inverse_map``): row ``e`` lists the flattened
    ``(node, slot)`` positions ``n*D + k`` whose entry is element ``e`` with a nonzero coefficient, ascending — at
    most the element's three vertices; the pad slots (element 0, coefficient 0) are left out."""
    from .ops import inverse_map
    el = np.asarray(div_elem)
    live = (np.asarray(div_c) != 0.0).any(axis=-1)
    return inverse_map(np.where(live, el, -1), int(n_elem), width)


def with_tables(mesh):
    """``mesh`` with the SE tables (:data:`TABLE_FIELDS`, :func:`build_tables`) as device arrays — needed by
    :func:`se_block`; the sharded mesh builds them per device (:func:`shard_tables`). Present ⇒ ``mesh``."""
    if getattr(mesh, "se_div_elem", None) is not None:
        return mesh
    t = build_tables(mesh)
    return dataclasses.replace(mesh, **{k: jnp.asarray(v) for k, v in t.items()})


def shard_tables(mesh, partition, Lmax: dict) -> dict:
    """Per-device SE tables ``{name: [P, Lmax_kind, ...]}`` on the shards' LOCAL lanes (for
    :func:`fesom_jax.shard_mesh.with_se_tables`): a node row keeps the elements present on the device (complete for
    an owned node), an element's neighbours map to local lanes (``-1`` when absent: only halo lanes, which the
    per-substep exchange overwrites; an owned element with an absent neighbour raises). Same global order as the
    dense tables, so owned lanes compute the dense bits."""
    g = build_tables(mesh)
    P = int(partition.npes)
    ml_n = [np.asarray(x) for x in partition.myList_nod2D]
    ml_e = [np.asarray(x) for x in partition.myList_elem2D]
    myd_n = np.asarray(partition.myDim_nod2D)
    myd_e = np.asarray(partition.myDim_elem2D)
    D = g["se_div_elem"].shape[1]
    out = {"se_div_elem": np.zeros((P, Lmax["nod"], D), np.int32),
           "se_div_c": np.zeros((P, Lmax["nod"], D, 2), g["se_div_c"].dtype),
           "se_elem_nb": np.full((P, Lmax["elem"], 3), -1, np.int32),
           "se_kappa": np.zeros((P, Lmax["elem"], 3), g["se_kappa"].dtype)}
    for d in range(P):
        g2l = np.full(int(mesh.elem2D), -1, np.int64)
        g2l[ml_e[d]] = np.arange(ml_e[d].size)
        el = g["se_div_elem"][ml_n[d]]
        c = g["se_div_c"][ml_n[d]]
        loc = g2l[el]
        live = (c != 0.0).any(axis=-1)
        miss = live & (loc < 0)
        if miss[: myd_n[d]].any():
            raise ValueError(f"ssh_se shard tables: device {d} lacks an element of an owned node's row")
        c = np.where(miss[..., None], 0.0, c)
        out["se_div_elem"][d, : ml_n[d].size] = np.where(loc >= 0, loc, 0)
        out["se_div_c"][d, : ml_n[d].size] = c
        nb = g["se_elem_nb"][ml_e[d]]
        lnb = np.where(nb >= 0, g2l[np.maximum(nb, 0)], -1)
        if ((nb >= 0) & (lnb < 0))[: myd_e[d]].any():
            raise ValueError(f"ssh_se shard tables: device {d} lacks an edge neighbour of an owned element")
        out["se_elem_nb"][d, : ml_e[d].size] = lnb
        out["se_kappa"][d, : ml_e[d].size] = np.where(lnb >= 0, g["se_kappa"][ml_e[d]], 0.0)
    # the local rows' inverse, one width over the devices (a shard's pad rows and absent elements are left out)
    invs = [div_inverse(out["se_div_elem"][d], out["se_div_c"][d], Lmax["elem"]) for d in range(P)]
    W = max(i.shape[1] for i in invs)
    out["se_div_inv"] = np.stack([div_inverse(out["se_div_elem"][d], out["se_div_c"][d], Lmax["elem"], W)
                                  for d in range(P)])
    return out


# --------------------------------------------------------------------------
# Device side
# --------------------------------------------------------------------------
STATE_FIELDS = ("se_eta", "se_ubt", "se_sbar_old")    # State order


def _column_sum(x, h, mask):
    """``sum_k h_k x_k`` over the layers in ``mask``, left to right from 0 (the Kokkos ``nz`` loop): ``[E, nl]`` →
    ``[E]``; a masked level adds +0.0."""
    s = jnp.zeros(x.shape[:1], x.dtype)
    for k in range(x.shape[1]):
        s = s + jnp.where(mask[:, k], h[:, k] * x[:, k], 0.0)
    return s


def transport(uv, helem, mesh):
    """``sum_k helem_k uv_k`` over each element's layers, ``[elem, 2]`` (the Kokkos ``Ubt_now``)."""
    lm = mesh.elem_layer_mask
    return jnp.stack([_column_sum(uv[:, :, c], helem, lm) for c in (0, 1)], axis=-1)


def cold_start(state, mesh):
    """The flat rings of a cold start (or of a CG state entering SE): ``se_eta = (eta_n, eta_n)``, the U ring three
    times ``sum_k helem uv``, ``se_sbar_old`` = the same sum (unused on a first step, where w_old = 0). At the rest IC
    (eta = u = 0) this is the Kokkos port's zero initialisation."""
    ub = transport(state.uv, state.helem, mesh)
    return dataclasses.replace(state, se_eta=jnp.stack([state.eta_n, state.eta_n], axis=-1),
                               se_ubt=jnp.stack([ub, ub, ub], axis=1), se_sbar_old=ub)


def flat_rings(eta_n, uv, helem, layer_mask):
    """:func:`cold_start`'s rings for arrays of any leading layout (host numpy ``[P, L, ...]`` or a folded sharded
    State's ``[P*L, ...]``): ``(se_eta, se_ubt, se_sbar_old)`` with the transport summed over the level axis left to
    right. numpy in ⇒ numpy out."""
    xp = np if isinstance(uv, np.ndarray) else jnp
    s = xp.zeros(uv.shape[:-2] + (2,), uv.dtype)
    for k in range(uv.shape[-2]):
        s = s + xp.where(layer_mask[..., k, None], helem[..., k, None] * uv[..., k, :], 0.0)
    return xp.stack([eta_n, eta_n], axis=-1), xp.stack([s, s, s], axis=-2), s


def init_state(state, layer_mask, cfg, *, restart: bool, log=print):
    """The driver's CG/SE guard and ring set-up (plan Task 3). ``cfg`` = the run's :class:`SeConfig` or None (the
    CG); ``layer_mask`` the element layer mask in ``state``'s layout.

    * SE run, rings in the State (a cold start that made them, or an SE restart): unchanged — the exact resume.
    * SE run, no rings (a cold start, or a restart written by a CG run): flat rings (:func:`flat_rings`); for a
      restart this is announced, because the substeps then start without their AB3/AM4 history.
    * CG run with rings (an SE restart): refused — the carried state is the SE scheme's (eta^n = eta^M, no CG
      warm start, the rings' meaning), and silently dropping it would switch the free-surface scheme mid-run."""
    has = getattr(state, "se_eta", None) is not None
    if cfg is None:
        if has:
            raise ValueError("this restart was written by a split-explicit run (it carries se_eta/se_ubt/"
                             "se_sbar_old); resuming it with the CG would switch the free-surface scheme mid-run. "
                             "Set ssh.split_explicit to continue it, or start the CG run from a CG restart.")
        return state
    if has:
        return state
    if restart:
        log("[run] split-explicit: the restart carries no SE rings (a CG restart) -> starting the substeps from "
            "flat rings (eta ring = eta_n, U ring = sum_k h u); the first step has no AB3/AM4 history")
    se_eta, se_ubt, se_sbar_old = flat_rings(state.eta_n, state.uv, state.helem, layer_mask)
    return dataclasses.replace(state, se_eta=se_eta, se_ubt=se_ubt, se_sbar_old=se_sbar_old)


def _grad3(gs, v0, v1, v2):
    """``(gs0 v0 + gs1 v1) + gs2 v2`` and its y twin: the Kokkos port's per-element gradient."""
    gx = gs[:, 0] * v0 + gs[:, 1] * v1 + gs[:, 2] * v2
    gy = gs[:, 3] * v0 + gs[:, 4] * v1 + gs[:, 5] * v2
    return gx, gy


def div_vjp_enabled() -> bool:
    """``FESOM_SE_DIV_VJP=1`` (read at trace time): :func:`divergence` with its explicit transpose."""
    return os.environ.get("FESOM_SE_DIV_VJP", "0").strip() not in ("", "0")


def divergence(U, mesh):
    """``T_n = sum_k (c_k,x U_x[el_k] + c_k,y U_y[el_k])`` in row order from 0 (``se_div_gather``): ``[elem, 2]`` →
    ``[nod]``, ``T = +d eta/dt``. With :func:`div_vjp_enabled` and the inverse table (``se_div_inv``; the wide-halo
    lanes have none) the reverse pass is :func:`_divergence_t` — ordered gathers through the inverse — instead of
    the default transpose of the gathers, a scatter-add (atomic on a GPU, the largest compute item of the
    sharded adjoint's reverse pass: 24 ms of 229 per step, job 28357325). The forward is the same."""
    el, c = mesh.se_div_elem, PR.wp(mesh.se_div_c)
    inv = getattr(mesh, "se_div_inv", None)
    if inv is not None and div_vjp_enabled():
        return _divergence_vjp(U, el, c, inv)
    return _divergence(U, el, c)


def _divergence(U, el, c):
    Ux, Uy = U[:, 0], U[:, 1]
    s = jnp.zeros(el.shape[:1], U.dtype)
    for k in range(el.shape[1]):
        s = s + (c[:, k, 0] * Ux[el[:, k]] + c[:, k, 1] * Uy[el[:, k]])
    return s


def _divergence_t(tbar, c, inv):
    """The transpose of :func:`_divergence` (it is linear in U): ``Ubar_e = sum over the rows' entries (n, k) of
    element e of c[n, k] tbar[n]``, as ``ops.ordered_sum`` over the inverse (ascending positions; no atomics)."""
    from .ops import ordered_sum
    vals = (c * tbar[:, None, None]).reshape((-1, 2))
    return ordered_sum(vals, inv)


@jax.custom_vjp
def _divergence_vjp(U, el, c, inv):
    return _divergence(U, el, c)


def _divergence_vjp_fwd(U, el, c, inv):
    return _divergence(U, el, c), (c, inv)


def _divergence_vjp_bwd(res, tbar):
    c, inv = res
    return _divergence_t(tbar, c, inv), None, None, None


_divergence_vjp.defvjp(_divergence_vjp_fwd, _divergence_vjp_bwd)


def _sqrt0(x):
    """``sqrt(x)``, finite gradient at 0 (momentum._safe_sqrt's double where)."""
    safe = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.sqrt(safe), 0.0)


def _visc_two_term(U0, Un, H0e, mesh, cfg):
    """The Kokkos port's live two-term harmonic viscosity ``V[U^m] - V[U^n]`` per element, divided by its area: per
    edge neighbour (ascending edge id) ``len = sqrt(A_e + A_nb)``, ``hh = (H0_e + H0_nb)/2``, ``spd = |dU|/hh``,
    ``vi = sqrt(max(gamma0, gamma1 spd) len)``; ``v -= dU^m vi^m; v += dU^n vi^n``; a coast slot adds nothing.

    A dead slot (a coast, a halo lane's absent neighbour, every slot of a sharded mesh's pad lane) has its velocity
    difference set to 0 before the coefficient, not only its term masked after: it reads lane 0 as a stand-in
    neighbour, and a pad lane's own velocity is not bounded (the pad Coriolis 1.0 s^-1 makes the substep unstable
    there, harmlessly in the forward), so its coefficient overflows within two steps and the reverse pass's
    zero cotangent times that infinite partial put NaN into lane 0 and from there everywhere (the sharded adjoint,
    docs/plans/20261022-core2-fast-adjoint.md Task 5). The live terms, hence the forward, are unchanged."""
    nb_all, area = mesh.se_elem_nb, mesh.elem_area
    vx = jnp.zeros_like(H0e)
    vy = jnp.zeros_like(H0e)
    for s in range(3):
        nb = nb_all[:, s]
        live = nb >= 0
        nbs = jnp.where(live, nb, 0)
        ln = _sqrt0(area + area[nbs])
        ihh = 1.0 / jnp.where(live, 0.5 * (H0e + H0e[nbs]), 1.0)
        for X, sign in ((U0, -1.0), (Un, 1.0)):
            dux = jnp.where(live, X[:, 0] - X[nbs, 0], 0.0)
            duy = jnp.where(live, X[:, 1] - X[nbs, 1], 0.0)
            spd = _sqrt0(dux * dux + duy * duy) * ihh
            vi = _sqrt0(jnp.maximum(cfg.gamma0, cfg.gamma1 * spd) * ln)
            if sign < 0:
                vx = vx - jnp.where(live, dux * vi, 0.0)
                vy = vy - jnp.where(live, duy * vi, 0.0)
            else:
                vx = vx + jnp.where(live, dux * vi, 0.0)
                vy = vy + jnp.where(live, duy * vi, 0.0)
    ia = 1.0 / area
    return vx * ia, vy * ia


def _visc_linear(dU, He, mesh, cfg):
    """FESOM_F's ``se_visc``: ``V_e = sum_k kappa_k (H_e + H_nb)/2 (dU_nb/H_nb - dU_e/H_e)``, ``kappa = visc_ah
    l^2/(d A_e)``, ``dU = U^m - U^n``, over the neighbour slots in order from 0. ``dU/H`` is 0 where ``H = 0`` (a
    pad lane of a sharded mesh; a wet element has ``H > 0``), so no 0/0 partial meets the reverse pass's zero
    cotangent there (see :func:`_visc_two_term`)."""
    nb_all, kap = mesh.se_elem_nb, PR.wp(mesh.se_kappa)
    wet = He > 0.0
    Hs = jnp.where(wet, He, 1.0)
    out = []
    for c in (0, 1):
        q = jnp.where(wet, dU[:, c] / Hs, 0.0)
        v = jnp.zeros_like(He)
        for s in range(3):
            nb = nb_all[:, s]
            live = nb >= 0
            nbs = jnp.where(live, nb, 0)
            kh = (cfg.visc_ah * kap[:, s]) * (0.5 * (He + He[nbs]))
            v = v + jnp.where(live, kh * (q[nbs] - q), 0.0)
        out.append(v)
    return out[0], out[1]


class SeOut(NamedTuple):
    """What :func:`se_block` returns: the trimmed velocity, the new SSH levels and rings, and the diagnostics the
    invariant tests read (``umean`` = <<U>>, ``unow`` = sum_k h^n u^n, ``he`` = H_e)."""
    uv: object
    eta_n: object
    hbar: object
    hbar_old: object
    se_eta: object
    se_ubt: object
    se_sbar_old: object
    umean: object
    he: object


class _Consts(NamedTuple):
    """The per-step constants of the substeps: element ``Fx, Fy, H0e, He, Un (elem, 2), kdrag`` (None without the
    drag) and the node water flux ``wf``."""
    Fx: object
    Fy: object
    H0e: object
    He: object
    Un: object
    kdrag: object
    wf: object


def _substep(c, k, lmesh, cfg, tauM, invM, w, exch):
    """One substep m -> m+1 (the k1 / k2 / k3 kernels of the Kokkos ``fesom_se_step``) on the lanes of ``lmesh``:
    carry ``(eta^m, eta^{m-1}, eta^{m-2}, U^m, U^{m-1}, U^{m-2}, <<U>>)``."""
    (a0, a1, a2), (d3, d0, d1, d2) = w
    en, gs, f = lmesh.elem_nodes, lmesh.gradient_sca, lmesh.coriolis
    nocav_n = lmesh.ulevels_nod2D == 1
    etam, eta1, eta2, U0, U1, U2, um = c
    ustar = a0 * U0 + a1 * U1 + a2 * U2                                    # k1: AB3 + the mean
    um = um + ustar * invM
    etanew = jnp.where(nocav_n, etam + tauM * (divergence(ustar, lmesh) - k.wf), etam)     # k2
    etanew = exch(etanew, "nod")
    h0 = d3 * etanew[en[:, 0]] + d0 * etam[en[:, 0]] + d1 * eta1[en[:, 0]] + d2 * eta2[en[:, 0]]   # k3: AM4
    h1 = d3 * etanew[en[:, 1]] + d0 * etam[en[:, 1]] + d1 * eta1[en[:, 1]] + d2 * eta2[en[:, 1]]
    h2 = d3 * etanew[en[:, 2]] + d0 * etam[en[:, 2]] + d1 * eta1[en[:, 2]] + d2 * eta2[en[:, 2]]
    hx, hy = _grad3(gs, h0, h1, h2)
    Ham = k.H0e + (h0 + h1 + h2) / 3.0
    if cfg.visc == "two_term":
        vx, vy = _visc_two_term(U0, k.Un, k.H0e, lmesh, cfg)
    elif cfg.visc == "linear":
        vx, vy = _visc_linear(U0 - k.Un, k.He, lmesh, cfg)
    else:
        vx = vy = jnp.zeros_like(k.H0e)
    if cfg.bt_drag:
        vx = vx - k.kdrag * (U0[:, 0] - k.Un[:, 0])
        vy = vy - k.kdrag * (U0[:, 1] - k.Un[:, 1])
    unx = U0[:, 0] + tauM * (f * ustar[:, 1] - G * Ham * hx + k.Fx + vx)
    uny = U0[:, 1] + tauM * (-f * ustar[:, 0] - G * Ham * hy + k.Fy + vy)
    # a sharded mesh's pad lane (ulevels 0; every element of a mesh has ulevels >= 1) keeps its U: with the pad
    # Coriolis 1.0 s^-1 its substep is unstable and its U overflowed within days (NaN on every pad lane of the U ring
    # after 30 days, job 28344098) — invisible in the forward, but NaN in the reverse pass's state cotangent
    live_e = (lmesh.ulevels >= 1)[:, None]
    unew = exch(jnp.where(live_e, jnp.stack([unx, uny], axis=-1), U0), "elem")
    return (etanew, etam, eta1, unew, U0, U1, um)


def se_unroll() -> int:
    """``FESOM_SE_UNROLL=u`` (``gpu.se_unroll``, read at trace time; default 1): the substep scans unroll ``u``
    substeps per loop trip (fewer trips and launches on a GPU; XLA may fuse across substeps, so the values move by
    round-off). fesom3-jax unrolls the whole wide-halo window, as ``ice_mevp.mevp_unroll`` the mEVP subcycles."""
    u = int(os.environ.get("FESOM_SE_UNROLL", "1") or 1)
    if u < 1:
        raise ValueError(f"FESOM_SE_UNROLL must be >= 1, got {u}")
    return u


def _scan(c, n, k, lmesh, cfg, tauM, invM, w, exch):
    """``n`` substeps (static) from the carry ``c``; ``jax.checkpoint`` per substep (the reverse pass stores the
    2-D carries only); ``se_unroll()`` substeps per loop trip."""
    if n == 0:
        return c

    def body(c, _):
        return _substep(c, k, lmesh, cfg, tauM, invM, w, exch), None
    return lax.scan(jax.checkpoint(body), c, None, length=n, unroll=min(se_unroll(), n))[0]


def se_block(mesh, st, du, water_flux, cfg: SeConfig, *, dt, is_first_step, exch=None, wide=None) -> SeOut:
    """Substeps 8-12 of the step in SE mode: the barotropic forcing, the ``cfg.M`` substeps, the trim, and
    ``eta_n = hbar = eta^M``. ``st`` is the step's carried state (uv = u^n, eta_n = eta^n, helem = h^n, the rings),
    ``du`` the momentum predictor's increment (substep 7), ``water_flux`` the node freshwater flux (zstar; ``None`` ⇒
    0). ``exch(field, kind)`` is the step's halo exchange (identity on one device): eta^{m+1} and U^{m+1} after every
    substep (the Kokkos port's two 2-D exchanges), H0_e once (an edge neighbour's depth), uv after the trim.
    ``wide`` (a :class:`fesom_jax.se_wide.SeWide`, sharded only): the substeps run on K+1-ring lanes with one refresh
    every K substeps instead (owned lanes: the same bits)."""
    if getattr(mesh, "se_div_elem", None) is None:
        raise ValueError("split-explicit SSH needs the SE mesh tables: mesh = ssh_se.with_tables(mesh) (dense) or "
                         "shard_mesh.with_se_tables(sm, mesh, partition) (sharded)")
    _exch = exch if exch is not None else (lambda f, kind: f)
    lm = mesh.elem_layer_mask
    helem = st.helem
    uv = st.uv
    tau = dt
    M = int(cfg.M)
    tauM = tau / M
    invM = 1.0 / M
    w = _weights(float(cfg.chi))
    wold = 0.0 if is_first_step else -(0.5 + _EPS_AB)
    wnew = 1.0 if is_first_step else (1.5 + _EPS_AB)
    en = mesh.elem_nodes
    gs = mesh.gradient_sca
    f = mesh.coriolis
    nocav_e = mesh.ulevels == 1
    nocav_n = mesh.ulevels_nod2D == 1

    # ---- forcing (se_forcing): R, U^n_now, H_e, U_AB2, F, H0_e (every lane; no communication) ----
    zero = jnp.zeros(uv.shape[:1], uv.dtype)
    rx = jnp.where(nocav_e, _column_sum(du[:, :, 0], helem, lm), zero)
    ry = jnp.where(nocav_e, _column_sum(du[:, :, 1], helem, lm), zero)
    ux = jnp.where(nocav_e, _column_sum(uv[:, :, 0], helem, lm), zero)
    uy = jnp.where(nocav_e, _column_sum(uv[:, :, 1], helem, lm), zero)
    He = jnp.where(nocav_e, _column_sum(jnp.ones_like(helem), helem, lm), zero)
    Rbx, Rby = rx / tau, ry / tau
    uab = wnew * ux + wold * st.se_sbar_old[:, 0]
    vab = wnew * uy + wold * st.se_sbar_old[:, 1]
    e0, e1, e2 = st.eta_n[en[:, 0]], st.eta_n[en[:, 1]], st.eta_n[en[:, 2]]
    gx, gy = _grad3(gs, e0, e1, e2)
    Fx = Rbx - f * vab + G * He * gx
    Fy = Rby + f * uab + G * He * gy
    H0e = He - (e0 + e1 + e2) / 3.0
    kdrag = None
    if cfg.bt_drag:
        kb = jnp.maximum(mesh.nlevels - 2, 0)
        ubx = jnp.take_along_axis(uv[:, :, 0], kb[:, None], axis=1)[:, 0]
        uby = jnp.take_along_axis(uv[:, :, 1], kb[:, None], axis=1)[:, 0]
        kdrag = C_D * _sqrt0(ubx * ubx + uby * uby) / jnp.where(He > 0.0, He, 1.0)
    wf = jnp.zeros_like(st.eta_n) if water_flux is None else water_flux
    init = (st.eta_n, st.se_eta[:, 0], st.se_eta[:, 1], st.se_ubt[:, 0], st.se_ubt[:, 1], st.se_ubt[:, 2],
            jnp.zeros_like(st.se_ubt[:, 0]))

    # ---- the M substeps ----
    if wide is None:
        k = _Consts(Fx, Fy, _exch(H0e, "elem"), He, st.se_ubt[:, 0], kdrag, wf)
        etaM, eta1, eta2, U0, U1, U2, umean = _scan(init, M, k, mesh, cfg, tauM, invM, w, _exch)
    else:
        etaM, eta1, eta2, U0, U1, U2, umean = wide.subcycle(init, (Fx, Fy, H0e, He, kdrag, wf), cfg, tauM, invM, w)

    # ---- finalize: hbar_old = hbar, hbar = eta_n = eta^M ----
    hbar_old = st.hbar
    hbar = jnp.where(nocav_n, etaM, st.hbar)
    eta_n = jnp.where(nocav_n, etaM, st.eta_n)

    # ---- trim (se_trim, h^n weights): u_k = u_k + du_k - (sum_j h_j (u+du)_j - <<U>>)/H_e ----
    ust = uv + du
    sx = _column_sum(ust[:, :, 0], helem, lm)
    sy = _column_sum(ust[:, :, 1], helem, lm)
    ok = nocav_e & (He > 0.0)
    Hs = jnp.where(ok, He, 1.0)
    cx = jnp.where(ok, (sx - umean[:, 0]) / Hs, 0.0)
    cy = jnp.where(ok, (sy - umean[:, 1]) / Hs, 0.0)
    lm3 = lm[:, :, None]
    uv_new = jnp.where(lm3, ust - jnp.stack([cx, cy], axis=-1)[:, None, :], uv)
    uv_new = _exch(uv_new, "elem")
    return SeOut(uv=uv_new, eta_n=eta_n, hbar=hbar, hbar_old=hbar_old,
                 se_eta=jnp.stack([eta1, eta2], axis=-1), se_ubt=jnp.stack([U0, U1, U2], axis=1),
                 se_sbar_old=jnp.stack([ux, uy], axis=-1), umean=umean, he=He)


# --------------------------------------------------------------------------
# CFL guard (host numpy, once per run): FESOM_F's se_init guard, the fesom3-jax ssh_se.cfl_* design
# --------------------------------------------------------------------------
CFL_SAFETY = 0.9
CFL_NIT = 100


def cfl_poly(a, chi):
    """Coefficients c(0..6) of ``z^4 (z-1)^2 + a^2 (w0 z^2 + w1 z + w2)(d3 z^3 + d0 z^2 + d1 z + d2)``: one gravity
    mode of the substeps with ``a = omega dtbt``."""
    (w0, w1, w2), (d3, d0, d1, d2) = _weights(chi)
    a2 = a * a
    return [a2 * (w2 * d2), a2 * (w1 * d2 + w2 * d1), a2 * (w0 * d2 + w1 * d1 + w2 * d0),
            a2 * (w0 * d1 + w1 * d0 + w2 * d3), a2 * (w0 * d0 + w1 * d3) + 1.0, a2 * (w0 * d3) - 2.0, 1.0]


def cfl_stable(a, chi, r=1.0):
    """Every root of the characteristic polynomial inside ``|z| < r`` (the Schur-Cohn recursion)."""
    c = cfl_poly(a, chi)
    rk = 1.0
    for k in range(7):
        c[k] = c[k] * rk
        rk = rk * r
    for n in range(6, 0, -1):
        if not abs(c[0]) < abs(c[n]):
            return False
        q = [c[n] * c[k + 1] - c[0] * c[n - 1 - k] for k in range(n)]
        c = [x / q[n - 1] for x in q]
    return True


def cfl_acrit(chi):
    """The first ``a = omega dtbt`` at which the substeps turn unstable (a scan from 0.5 in steps of 0.005, then 60
    bisections; 4 when none below 4)."""
    a0, da, amax = 0.5, 0.005, 4.0
    hi = amax
    for k in range(1, round((amax - a0) / da) + 1):
        if not cfl_stable(a0 + k * da, chi):
            hi = a0 + k * da
            break
    lo = amax if hi >= amax else hi - da
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cfl_stable(mid, chi):
            lo = mid
        else:
            hi = mid
    return lo


def cfl_omega_max(mesh, helem=None):
    """``omega_max = sqrt(lambda_max(A))`` of the substeps' gravity-wave operator ``A v = -T(H (-g grad v))``
    (``T`` the substeps' divergence, ``H = sum_k helem``, the rest depth when ``helem`` is None) by ``CFL_NIT`` power
    iterations from Knuth's hash of the 1-based node id, in the areasvol-weighted norm (a lower bound of omega_max;
    the safety factor covers it). Needs the SE tables (:func:`build_tables` is called if absent). Returns
    ``(omega, the last relative change of lambda)``."""
    t = build_tables(mesh) if getattr(mesh, "se_div_elem", None) is None else \
        {k: np.asarray(getattr(mesh, k)) for k in TABLE_FIELDS}
    en = np.asarray(mesh.elem_nodes)
    gs = np.asarray(mesh.gradient_sca).astype(np.float64)
    if helem is None:
        H = -np.asarray(mesh.zbar).astype(np.float64)[np.asarray(mesh.nlevels) - 1]
    else:
        lm = np.asarray(mesh.elem_layer_mask)
        H = np.where(lm, np.asarray(helem, np.float64), 0.0).sum(axis=1)
    H = np.where(np.asarray(mesh.ulevels) == 1, H, 0.0)
    el, c = t["se_div_elem"], t["se_div_c"].astype(np.float64)
    av = np.asarray(mesh.areasvol)[:, 0].astype(np.float64)
    gid = np.arange(1, av.shape[0] + 1, dtype=np.int64)
    v = ((gid * 2654435761) % 4294967296).astype(np.float64) / 4294967296.0 - 0.5
    v = v / np.sqrt(np.sum(av * v * v))

    def apply_a(v):
        pre = -(G * v[en])
        U = np.stack([H * (gs[:, 0:3] * pre).sum(axis=1), H * (gs[:, 3:6] * pre).sum(axis=1)], axis=-1)
        T = (c[:, :, 0] * U[el, 0] + c[:, :, 1] * U[el, 1]).sum(axis=1)
        return -T

    lam = lam_old = 0.0
    for _ in range(CFL_NIT):
        Av = apply_a(v)
        lam_old, lam = lam, float(np.sqrt(np.sum(av * Av * Av)))
        if not lam > 0.0:
            break
        v = Av / lam
    return float(np.sqrt(lam if lam > 0.0 else 0.0)), (abs(lam - lam_old) / lam if lam > 0.0 else 0.0)


def cfl_guard(mesh, dt, cfg: SeConfig, helem=None):
    """FESOM_F's CFL guard: ``M_min = ceil(tau / dtbt_max)``, ``dtbt_max = CFL_SAFETY a_crit(chi) / omega_max``.
    Returns ``(M_min, omega_max, a_crit, the guard line)``; raises when ``cfg.M < M_min`` unless ``cfg.m_force``."""
    omega, rel = cfl_omega_max(mesh, helem)
    acrit = cfl_acrit(float(cfg.chi))
    lim = CFL_SAFETY * acrit / omega if omega > 0.0 else math.inf
    m_min = 1 if math.isinf(lim) else max(1, math.ceil(dt / lim))
    line = (f"split-explicit CFL guard: omega_max = {omega:.4E} 1/s ({CFL_NIT} power iterations, last change "
            f"{rel:.1E}), a_crit(chi={cfg.chi}) = {acrit:.4f}, s = {CFL_SAFETY:.2f} -> dtbt <= {lim:.3f} s -> "
            f"M_min = {m_min}, M = {cfg.M} at dt = {dt:g} s")
    if cfg.M < m_min and not cfg.m_force:
        raise ValueError(line + f": M below M_min (set split_explicit.M >= {m_min}, or m_force: true)")
    return m_min, omega, acrit, line
