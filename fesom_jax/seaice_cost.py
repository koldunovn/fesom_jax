"""Monthly-mean sea-ice accumulators and the observation cost of the ice-parameter calibration.

The cost compares calendar-month means of the post-step ``a_ice`` and ``m_ice`` with the
super-observations of :mod:`fesom_jax.obs_seaice` (``[12, nod2D]`` arrays, NaN where there is no
observation)::

    J_C,hem = 1/2 sum_m sum_{n in hem, obs valid} w_C[m,n] (Cbar[m,n] - C_obs[m,n])^2 / sC[m,n]^2
    J_H,NH  = 1/2 sum_m sum_{n in NH,  obs valid} w_H[m,n] (Hbar[m,n] - H_obs[m,n])^2 / sH[m,n]^2

with the observed-area weights ``w = n_cells * 625 km^2 / (100 km)^2`` (:func:`obs_seaice.obs_weight`)
and ``J_data = J_C,NH + J_C,SH + J_H,NH``. Months the model window does not reach (no step in the
month) are skipped, as are nodes without an observation. The prior 1/2 u^T u is added on the host.

Monthly means inside a chunked scan: the accumulators of :func:`adjoint_costs.with_running_costs`
see only the new State, with no step index and no date. Every chunk therefore carries ONE month
index in its xs (a traced int32 scalar), and :func:`monthly_accumulators` builds the accumulator
functions as closures over it. That is exact only if no chunk straddles a month boundary;
:func:`chunk_months` computes the per-chunk month indices of a window and refuses a chunk length or
start time for which a chunk would straddle one. A step belongs to the month of its MID-step time
(the step that ends at 00:00 on the 1st belongs to the month before, as in FESOM's monthly output).

The weight sums ``N_*`` (all scored observations) and ``Nact_*`` (the ACTIVE ones: observation
> 0 or model > 0) are returned for the chi^2/N_eff scaling, which counts only active ones.

``cost_terms`` (jnp, differentiable w.r.t. the accumulators) scores JAX output; ``cost_terms_np``
(numpy) scores plain ``[12, nod2D]`` monthly means, e.g. a Fortran FESOM2 run's monthly
``a_ice``/``m_ice`` in mesh node order. Both evaluate the same formula (:func:`_terms`).
"""

from __future__ import annotations

import datetime
from typing import Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np

from .obs_seaice import check_obs, obs_weight

TERMS = ("J_C_NH", "J_C_SH", "J_H_NH")


# --- accumulators ----------------------------------------------------------------------------
def monthly_accumulators(month):
    """Accumulators for ``with_running_costs``: ``a_mon``/``m_mon`` [12, nod2D] add the new
    State's ``a_ice``/``m_ice`` into row ``month``; ``n_mon`` [12] counts the steps per month.

    ``month`` is the chunk's month index 0-11 (a traced int32 scalar from the chunk's xs, or a
    Python int). The one-hot row keeps the increment linear in the state, so its cotangent is
    the plain row of the accumulator's cotangent."""
    def onehot():
        return (jnp.arange(12) == month).astype(jnp.float64)

    return {
        "a_mon": lambda st: onehot()[:, None] * st.a_ice[None, :],
        "m_mon": lambda st: onehot()[:, None] * st.m_ice[None, :],
        "n_mon": lambda st: onehot(),
    }


def chunk_months(start, dt: float, chunk_steps: int, n_chunks: int) -> np.ndarray:
    """Month index (0-11) of every chunk of a window, as int32 [n_chunks].

    Step ``i`` (0-based) runs from ``start + i*dt`` to ``start + (i+1)*dt``; its month is that of
    its mid-step time. Raises ``ValueError`` if any chunk holds steps of two months (a chunk
    straddling a month boundary) or if the window leaves the calendar year it starts in (the
    month index would alias two years)."""
    if isinstance(start, str):
        start = datetime.datetime.fromisoformat(start)
    if chunk_steps < 1 or n_chunks < 1 or dt <= 0:
        raise ValueError("chunk_months: chunk_steps, n_chunks and dt must be positive")
    out = np.empty(n_chunks, dtype=np.int32)
    for c in range(n_chunks):
        first = start + datetime.timedelta(seconds=(c * chunk_steps + 0.5) * dt)
        last = start + datetime.timedelta(seconds=((c + 1) * chunk_steps - 0.5) * dt)
        if (first.year, first.month) != (last.year, last.month):
            raise ValueError(
                f"chunk {c} straddles a month boundary ({first:%Y-%m-%d %H:%M} .. "
                f"{last:%Y-%m-%d %H:%M}); choose a chunk length and start time that divide the "
                "months (e.g. 24 steps of 1800 s from 00:00 or 12:00)")
        if first.year != start.year:
            raise ValueError(f"chunk {c} lies in {first.year}, the window started in {start.year}; "
                             "monthly accumulators hold one calendar year")
        out[c] = first.month - 1
    return out


def monthly_means(acc: Mapping, xp=jnp):
    """``(Cbar, Hbar, have)`` from the accumulators: the sums divided by the step counts, and
    the months that hold at least one step (zero means elsewhere)."""
    n = acc["n_mon"]
    have = n > 0
    n_safe = xp.where(have, n, 1.0)[:, None]
    return acc["a_mon"] / n_safe, acc["m_mon"] / n_safe, have


# --- observations as cost arrays -------------------------------------------------------------
class CostObs(NamedTuple):
    """Dense cost arrays (no NaN): values, weights ``w / sigma^2`` (0 where invalid) and masks."""
    C: np.ndarray          # [12, nod2D]
    WC: np.ndarray         # [12, nod2D]  w_C / sC^2, 0 where no observation
    validC: np.ndarray     # [12, nod2D] bool
    H: np.ndarray
    WH: np.ndarray
    validH: np.ndarray
    nh: np.ndarray         # [nod2D] bool, northern nodes
    wC: np.ndarray         # [12, nod2D] observed-area weights (for N_eff)
    wH: np.ndarray


def cost_obs(obs: Mapping[str, np.ndarray]) -> CostObs:
    """Turn an ``obs_<YYYY>.npz`` dict (``C_obs, sC, nC, H_obs, sH, nH, nh``) into
    :class:`CostObs`. The thickness term is restricted to northern nodes (CCI is Arctic)."""
    check_obs(obs)
    nh = np.asarray(obs["nh"], dtype=bool)
    if nh.shape != (np.shape(obs["C_obs"])[1],):
        raise ValueError(f"nh has shape {nh.shape}, expected ({np.shape(obs['C_obs'])[1]},)")

    def one(v, s, n, mask=None):
        valid = np.asarray(n) > 0
        if mask is not None:
            valid = valid & mask[None, :]
        w = np.where(valid, obs_weight(n), 0.0)
        sig = np.where(valid, s, 1.0)
        return (np.where(valid, v, 0.0), w / sig ** 2, valid, w)

    C, WC, vC, wC = one(obs["C_obs"], obs["sC"], obs["nC"])
    H, WH, vH, wH = one(obs["H_obs"], obs["sH"], obs["nH"], nh)
    return CostObs(C=C, WC=WC, validC=vC, H=H, WH=WH, validH=vH, nh=nh, wC=wC, wH=wH)


# --- the cost --------------------------------------------------------------------------------
def _terms(xp, Cbar, Hbar, have, ob: CostObs) -> dict:
    shp = tuple(ob.C.shape)
    for name, a in (("C mean", Cbar), ("H mean", Hbar)):
        if tuple(a.shape) != shp:
            raise ValueError(f"{name} has shape {tuple(a.shape)}, expected {shp}")
    if tuple(have.shape) != (12,):
        raise ValueError(f"month mask has shape {tuple(have.shape)}, expected (12,)")
    nh = xp.asarray(ob.nh)[None, :]
    hv = xp.asarray(have)[:, None]

    def term(mod, o, W, valid, mask):
        use = xp.asarray(valid) & hv & mask
        r = xp.where(use, mod - xp.asarray(o), 0.0)
        per_month = 0.5 * xp.sum(xp.asarray(W) * r * r, axis=1)
        return per_month

    out = {}
    out["J_C_NH_month"] = term(Cbar, ob.C, ob.WC, ob.validC, nh)
    out["J_C_SH_month"] = term(Cbar, ob.C, ob.WC, ob.validC, ~nh)
    out["J_H_NH_month"] = term(Hbar, ob.H, ob.WH, ob.validH, nh)
    for t in TERMS:
        out[t] = xp.sum(out[t + "_month"])
    out["J_data"] = out["J_C_NH"] + out["J_C_SH"] + out["J_H_NH"]
    # observed-area weight sums of the scored observations: all of them (N_*), and the ACTIVE
    # ones (Nact_*: observation > 0 or model > 0), which N_eff of the chi^2/N_eff scaling counts
    # (decided 2026-10-10: the far-field open-water zeros stay in J but not in N_eff)
    for t, w, valid, mask, o, mod in (("C_NH", ob.wC, ob.validC, nh, ob.C, Cbar),
                                      ("C_SH", ob.wC, ob.validC, ~nh, ob.C, Cbar),
                                      ("H_NH", ob.wH, ob.validH, nh, ob.H, Hbar)):
        use = xp.asarray(valid) & hv & mask
        act = use & ((xp.asarray(o) > 0) | (mod > 0))
        out["N_" + t] = xp.sum(xp.where(use, xp.asarray(w), 0.0))
        out["Nact_" + t] = xp.sum(xp.where(act, xp.asarray(w), 0.0))
    return out


def cost_terms(acc: Mapping, ob: CostObs) -> dict:
    """The cost terms (jnp) from the running accumulators ``a_mon``, ``m_mon``, ``n_mon``.
    ``J_data`` is the scalar to differentiate; the per-term and per-month values (and the weight
    sums ``N_*``) are for reports."""
    Cbar, Hbar, have = monthly_means(acc, jnp)
    return _terms(jnp, Cbar, Hbar, have, ob)


def cost_terms_np(C_mean, H_mean, ob: CostObs, months=None) -> dict:
    """The same cost (numpy, float) from plain ``[12, nod2D]`` monthly means, e.g. a Fortran
    run's monthly ``a_ice``/``m_ice``. ``months`` selects the months to score (0-11; default all
    twelve); rows of other months are ignored and may be NaN."""
    C_mean = np.asarray(C_mean, dtype=np.float64)
    H_mean = np.asarray(H_mean, dtype=np.float64)
    have = np.zeros(12, dtype=bool)
    have[list(range(12)) if months is None else list(months)] = True
    out = _terms(np, C_mean, H_mean, have, ob)
    return {k: (float(v) if np.ndim(v) == 0 else np.asarray(v)) for k, v in out.items()}
