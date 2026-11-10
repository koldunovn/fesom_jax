#!/usr/bin/env python
"""Calibrate the four TKE constants against mixed-layer depth with conditioned controls, on the
shared CORE2 adjoint driver.

    python scripts/calib/tke_lbfgs.py calib --mode twin|real --optimizer lbfgs|adam [options]
    python scripts/calib/tke_lbfgs.py fd    [options]     # FD vs adjoint of dJ/du, per constant
    python scripts/calib/tke_lbfgs.py fwd   [options]     # J and per-window misfits at one point

Run it from a checkout with ``PYTHONPATH=<checkout>`` (``scripts/calib/tke_lbfgs.sbatch``,
``scripts/adjoint/job_*.sbatch``); the driver refuses a ``fesom_jax`` from another tree.

Controls
    ``u`` (4,) = non-dimensional controls of ``(tke_c_k, tke_c_eps, tke_cd, tke_alpha)`` via
    :func:`fesom_jax.controls.tke_control_set`: ``p = p0·exp(sigma·u)``, p0 = (0.1, 0.7, 3.75,
    30), sigma = ln 2 by default. ``u`` enters every compiled program as a strong float64 array
    of shape (4,), so no evaluation recompiles.

Cost (Bayesian form; the defaults are written into the run config and every JSONL row)
    J(u) = (N_eff / 2) · Σ_w misfit_w / σ_obs²  +  ½ uᵀu       (the prior term with --prior on)

    ``misfit_w`` is the area-weighted mean squared error (:func:`fesom_jax.obs_compare.misfit`)
    of the WINDOW-MEAN model MLD against the target MLD of the window's month, on the 1°
    WOA grid. The model MLD is the density-threshold MLD on the live geometry, summed after every
    step by a running accumulator (:func:`fesom_jax.adjoint_costs.with_running_costs`), divided
    by the number of steps and mapped to the grid with
    :func:`fesom_jax.obs_compare.to_obs_surface` (area-weighted cell mean of the wet surface
    nodes). Columns mixed to the bottom enter with the depth of the column's deepest wet
    mid-layer (``--mld-invalid depth``, default; :func:`fesom_jax.adjoint_costs.
    deepest_wet_mid_depth`), the deepest value a valid crossing can take: the cost is then
    continuous when the last crossing, at the deepest wet level, disappears. ``--mld-invalid
    zero`` enters them as 0 (the earlier WOA calibration's convention).
    Cell weights are that calibration's: ``cell_area · WOA-MLD-valid(month) · (WOA SST(month) >
    −1 °C) · (cell has a wet model node)``.
    σ_obs = 20 m (roughly the uncertainty of a monthly-climatology MLD) and N_eff = 500 (roughly
    the number of ~700 km decorrelation boxes in the WOA-valid ocean) are CHOICES: they set the
    weight of the data against the standard-normal prior on u, whose Hessian is the identity —
    a constant is "identifiable" when its data-Hessian eigenvalue is ≥ 10.

Modes
    ``real``: targets = WOA monthly MLD (``--woa``, read-only). ``twin``: targets = the model's
    own window-mean MLD at ``--truth`` (physical constants), mapped through the same operator
    and weighted with the same weights; computed once per run directory and kept in
    ``twin_targets.npz`` (so a resumed run uses the same targets).

Windows
    One window per month in ``--months`` of ``--year``, starting on the 1st at 00:00 from the
    saved State ``fesom.<YYYY>.<DOY>.00000`` (``--states-dir``, else the hindcast restart
    archive), ``--window-steps`` long (12 = 6 h at dt = 1800 s), in chunks of ``--chunk-steps``
    (default: the whole window). ONE chunk program per process: the window's State, dates,
    forcing, target and weights go in through ``init_state`` / ``xs_fn`` / ``loss_fn``; its
    gradient comes from :func:`fesom_jax.adjoint.chunked_grad` (via ``core2_adj.run_grad``) and
    the windows are summed on the host. The linearity guard runs as ``"full"`` on the FIRST
    evaluation of each window in the process and is off afterwards (a 6-h window is a single
    chunk, so the default guard would double every evaluation); each window row records it.

Optimisers
    ``lbfgs``: :func:`fesom_jax.optim_lbfgs.minimize` (m1qn3/lsopt style, about one evaluation
    per iteration; resumable from its log) with ``expand = --lbfgs-expand`` (default 4, see
    :data:`LBFGS_EXPAND`); ``--free`` optimises a subset of the controls (the rest stay at the
    start), ``--holdout-months`` adds the misfit of other months at the start and the result. ``adam``: ``optax.adam(optax.cosine_decay_schedule(
    lr, n_evals))`` run on the host against the SAME ``fun_and_grad(u)`` for a fixed budget of
    ``--n-evals`` evaluations (default 30); resumable by replaying its log. ``--lr`` is in u
    units (one u = one prior standard deviation); the default 0.1 is provisional until the
    first calibration's short probe fixes it.

Outputs (``--out``, default ``/work/ab0995/a270088/fesom_jax_dev/runs/calib/<action>_<mode>_<job>``)
    ``run_config.json``   cost, controls, windows, optimiser settings and provenance.
    ``iterations.jsonl``  the optimiser log: a ``header`` row (options, limits, u0, run_info),
                          one ``eval`` row per evaluation (u, f, g, accepted/pair_stored/status
                          for L-BFGS, lr for Adam, eval_s, and ``aux``: J_data, J_prior, p,
                          per-window J/misfit, compile_s, run_s), a ``stop`` row.
    ``windows.jsonl``     one row per window evaluation with the ``core2_adj`` provenance keys
                          (git HEAD, job, device, XLA flags, config, adjoint mode, ice mode,
                          window, chunk, compile_s separately from run_s) + eval, month, u, p,
                          J_w, misfit_m2, g_w, check_linear, linear_check.
    ``fd.jsonl`` / ``fwd.jsonl``  rows of the ``fd`` / ``fwd`` actions.
Nothing is deleted or overwritten: logs are appended, and an existing ``twin_targets.npz`` or
``run_config.json`` that disagrees with the arguments is an error.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import core2_adj as C  # noqa: E402  (its import guard runs here)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fesom_jax import adjoint, adjoint_check, obs_compare, optim_lbfgs  # noqa: E402
from fesom_jax import adjoint_costs as AC  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402
from fesom_jax.controls import ControlSet, tke_control_set  # noqa: E402

WOA_NPZ = Path("/home/a/a270088/port_jax/scripts/woa_targets.npz")
RUNS_DIR = Path("/work/ab0995/a270088/fesom_jax_dev/runs/calib")
_T_START = time.time()        #: process start, for --wall-budget-s
SIGMA_OBS = 20.0          #: m
N_EFF = 500.0             #: independent observations per window
ICE_SST = -1.0            #: °C; WOA cells at or below it are treated as ice covered
MLD_INVALID = {"depth": "column", "zero": "zero"}
TRUTH_DEFAULT = (0.14, 0.5, 3.0, 25.0)
ADAM_LR = 0.1
ADAM_EVALS = 30
#: L-BFGS cold-start growth after a dropped pair (:class:`fesom_jax.optim_lbfgs.LBFGSOptions`):
#: over the short first steps the window-mean MLD gradient switches between smooth pieces of the
#: cost (thousands of nodes change level per 1e-3 of u), s'y < 0, the pairs are dropped, and with
#: the plain rule (1) the optimiser crawls at the 0.2 % first step until a pair happens to pass
#: (twin, 12-h windows, job 27742868: six evaluations at f −0.5 % and constants +0.1 % each).
LBFGS_EXPAND = 4.0

__all__ = ["WOA_NPZ", "SIGMA_OBS", "N_EFF", "Window", "load_woa_month", "woa_weights",
           "mld_accumulators", "build_calib_model", "Calibration", "run_lbfgs", "run_adam",
           "run_fd", "run_fwd", "main"]


# ---------------------------------------------------------------------------
# targets and weights


def load_woa_month(path, month: int) -> dict:
    """WOA MLD and SST of ``month`` (1-12) as flat ``[n_cells]`` float64 (lat-major), with their
    validity masks. Land/invalid cells are set to 0 (0·NaN would poison the weighted sums even
    at weight 0); the masks drop them from the misfit."""
    with np.load(path) as d:
        lat, lon = np.asarray(d["lat"]), np.asarray(d["lon"])
        mi = int(month) - 1
        mld = np.asarray(d["mld_monthly"][mi], np.float64).reshape(-1)
        mld_ok = np.asarray(d["mld_valid_monthly"][mi]).reshape(-1).astype(bool)
        sst = np.asarray(d["sst_monthly"][mi], np.float64).reshape(-1)
        sst_ok = np.asarray(d["sst_valid"]).reshape(-1).astype(bool)
    return {"lat": lat, "lon": lon,
            "mld": np.where(np.isfinite(mld) & mld_ok, mld, 0.0), "mld_ok": mld_ok,
            "sst": np.where(np.isfinite(sst) & sst_ok, sst, 0.0), "sst_ok": sst_ok}


def woa_weights(hmap, woa: dict, ice_sst: float = ICE_SST) -> np.ndarray:
    """``[n_cells]`` misfit weights: ``cell_area · mld_ok · (sst > ice_sst) · cell_valid`` with
    ``cell_valid`` = the cell contains a wet model surface node."""
    nod2D = int(np.asarray(hmap.node_cell).shape[0])
    _, cell_valid = obs_compare.to_obs_surface(jnp.ones((nod2D,), jnp.float64), hmap)
    w = (np.asarray(hmap.cell_area) * woa["mld_ok"] * (woa["sst"] > ice_sst)
         * np.asarray(cell_valid))
    return np.asarray(w, np.float64)


def mld_accumulators(invalid: str = "depth"):
    """The accumulator factory for :func:`core2_adj.build_model`: ``{"mld": MLD field}`` with
    columns mixed to the bottom at the deepest wet mid-layer depth (``"depth"``) or 0
    (``"zero"``)."""
    mode = MLD_INVALID[invalid]

    def factory(mesh):
        return {"mld": lambda s: AC.mld_field(s, mesh, invalid=mode)}
    return factory


def build_calib_model(start, window_steps: int, chunk_steps: int | None = None, *,
                      cs: ControlSet | None = None, mld_invalid: str = "depth",
                      adjoint_cfg: AdjointConfig | None = None, ice_mode: str = "prescribed",
                      save_names: tuple = (), rematerialize: bool = True, restart=None,
                      state=None) -> "C.Model":
    """The CORE2 reference model with the u-vector controls and the MLD accumulator."""
    cs = cs or tke_control_set()
    return C.build_model(start, window_steps, chunk_steps or window_steps, restart, adjoint_cfg,
                         ice_mode, tuple(save_names), mld_accumulators(mld_invalid),
                         rematerialize, params0=np.zeros(cs.n, np.float64),
                         to_params=cs.to_params, state=state)


# ---------------------------------------------------------------------------
# the calibration problem


class Window(NamedTuple):
    start: datetime.datetime
    month: int
    state: Any                  #: host State at ``start``
    target: jax.Array           #: [n_cells] target MLD [m]
    weight: jax.Array           #: [n_cells]


def _prior(u: np.ndarray) -> float:
    return 0.5 * math.fsum((u * u).tolist())


class Calibration:
    """``J(u)``, ``∇J(u)`` over a set of windows (see the module docstring).

    ``starts``: window start datetimes (the WOA month is the start's month). ``states``: optional
    ``{start: host State}`` overrides (default: :func:`core2_adj.load_state` of the saved State
    at each start). ``mode="twin"`` computes its targets at ``u_truth`` (or loads them from
    ``out_dir/twin_targets.npz``)."""

    def __init__(self, model, starts: Sequence, *, cs: ControlSet | None = None,
                 mode: str = "real", woa=WOA_NPZ, sigma_obs: float = SIGMA_OBS,
                 n_eff: float = N_EFF, prior: bool = True, u_truth=None,
                 ice_sst: float = ICE_SST, states: dict | None = None, states_dir=None,
                 out_dir=None, mld_invalid: str = "depth", hmap=None,
                 first_check: str = "auto", targets_from=None):
        if mode not in ("real", "twin"):
            raise ValueError(f"mode must be 'real' or 'twin', got {mode!r}")
        if "mld" not in model.accumulators:
            raise ValueError("the model needs the 'mld' accumulator (build_calib_model)")
        self.model, self.cs, self.mode = model, cs or tke_control_set(), mode
        self.woa_path, self.sigma_obs, self.n_eff = Path(woa), float(sigma_obs), float(n_eff)
        self.prior, self.ice_sst, self.mld_invalid = bool(prior), float(ice_sst), mld_invalid
        self.scale = self.n_eff / (2.0 * self.sigma_obs ** 2)
        self.n_steps = model.n_steps
        self.out_dir = None if out_dir is None else Path(out_dir)
        self.states_dir = C.STATES_DIR if states_dir is None else Path(states_dir)
        self.u_truth = None if u_truth is None else self.cs.check(u_truth)
        if mode == "twin" and self.u_truth is None:
            raise ValueError("twin mode needs u_truth")
        woa0 = load_woa_month(self.woa_path, 1)
        self.hmap = hmap if hmap is not None else obs_compare.build_h_map(
            model.mesh, obs_compare.ObsGrid(lat=woa0["lat"], lon=woa0["lon"],
                                            depth=np.array([10.0, 100.0])))
        H = self.hmap
        # ONE jitted mapping and ONE jitted misfit per process; per-window inputs are arguments.
        self._cell_map = jax.jit(lambda s, n: obs_compare.to_obs_surface(s / n, H)[0])
        self._misfit = jax.jit(obs_compare.misfit)
        self._n = jnp.asarray(float(self.n_steps), jnp.float64)
        self._linear_done: set = set()
        self.first_check = self.resolve_check(first_check, model)
        self.targets_from = None if targets_from is None else Path(targets_from)
        self.n_evals = 0
        starts = [C._parse_date(s) for s in starts]
        states = dict(states or {})
        self.windows = []
        for s in starts:
            st = states[s] if s in states else self._load_state(s)
            woa = load_woa_month(self.woa_path, s.month)
            w = woa_weights(H, woa, self.ice_sst)
            self.windows.append(Window(s, s.month, st, jnp.asarray(woa["mld"]), jnp.asarray(w)))
        if mode == "twin":
            self._twin_targets()

    # -- setup ---------------------------------------------------------------

    @staticmethod
    def resolve_check(mode, model) -> str | bool:
        """The linearity guard of a window's FIRST evaluation. ``"auto"``: ``"full"`` on a CPU
        backend or with 1-step chunks, else ``"zero"``. On a GPU a multi-step chunk VJP re-runs
        its rematerialised forward, whose round-off flips discrete switches (the convective
        mixing switch, sea-ice thermodynamics) differently in every call, so the ``"full"``
        guard's repeat and 2·cot VJPs differ by far more than round-off and its test is noise;
        ``"zero"`` (vjp(0) == 0) stays exact on any platform. ``"off"`` = no guard."""
        if mode == "auto":
            return "full" if (jax.default_backend() == "cpu" or model.chunk_steps == 1) else "zero"
        if mode == "off":
            return False
        if mode not in ("full", "zero"):
            raise ValueError(f"first_check must be auto, full, zero or off, got {mode!r}")
        return mode

    def _load_state(self, start):
        cand = self.states_dir / C.restart_tag(start)
        path = cand if cand.is_dir() else C._resolve_restart(start)
        st, meta = C.load_state(path)
        C._check_restart_date(path, meta, start)
        return st

    def _twin_targets(self):
        """The twin targets: loaded from ``targets_from`` (another run's ``twin_targets.npz``,
        so that runs compared with each other share one truth forward — on a GPU two forwards
        at the truth differ) or from ``out_dir/twin_targets.npz``, else computed at
        ``u_truth`` and written there."""
        path = None if self.out_dir is None else self.out_dir / "twin_targets.npz"
        keys = [w.start.isoformat() for w in self.windows]
        if self.targets_from is not None:
            if not self.targets_from.is_file():
                raise FileNotFoundError(f"twin targets {self.targets_from} not found")
            path = self.targets_from
        if path is not None and path.exists():
            with np.load(path) as z:
                fkeys = [str(k) for k in z["starts"]]
                if (not np.array_equal(z["u_truth"], self.u_truth)
                        or not set(keys) <= set(fkeys) or int(z["n_steps"]) != self.n_steps
                        or str(z["mld_invalid"]) != self.mld_invalid):
                    raise ValueError(f"{path} was made for another truth/window set")
                # a subset of its windows is fine: each target is looked up by its start
                tg = [jnp.asarray(z[f"target_{fkeys.index(k)}"]) for k in keys]
        else:
            tg = [jnp.asarray(self.cell_mld(self.u_truth, w)) for w in self.windows]
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "xb") as f:
                    np.savez(f, u_truth=self.u_truth, starts=np.array(keys),
                             n_steps=self.n_steps, mld_invalid=self.mld_invalid,
                             **{f"target_{i}": np.asarray(t) for i, t in enumerate(tg)})
        self.windows = [w._replace(target=t) for w, t in zip(self.windows, tg)]

    # -- pieces --------------------------------------------------------------

    def cell_of(self, carry):
        """The window-mean model MLD on the grid ``[n_cells]`` from a window-end carry."""
        return self._cell_map(carry[1]["mld"], self._n)

    def loss_fn(self, w: Window) -> Callable:
        """``carry -> (N_eff/2)·misfit/σ_obs²`` for window ``w`` (the data term)."""
        def loss(carry):
            return self.scale * self._misfit(self.cell_of(carry), w.target, w.weight)
        loss.__name__ = f"mld_misfit_{w.start:%Y%m%d}"
        return loss

    def _forward(self, u, w: Window):
        m = self.model
        return adjoint.chunked_forward(m.chunk_fn, np.asarray(u, np.float64),
                                       init_state=m.init_carry(w.state),
                                       n_chunks=m.chunks_for(self.n_steps),
                                       xs_fn=m.xs_fn(w.start))

    def cell_mld(self, u, w: Window) -> np.ndarray:
        """Forward only: the window-mean model MLD on the grid at ``u``."""
        fin = self._forward(u, w)
        out = np.asarray(self.cell_of(fin))
        C._free(fin)
        return out

    def config(self) -> dict:
        return {"cost": "(N_eff/2)*sum_w misfit_w/sigma_obs^2 + prior*0.5*u.u",
                "misfit": "obs_compare.misfit of to_obs_surface(window-mean MLD) vs target",
                "sigma_obs_m": self.sigma_obs, "n_eff": self.n_eff, "prior": self.prior,
                "scale": self.scale, "mode": self.mode, "woa": str(self.woa_path),
                "ice_sst": self.ice_sst, "mld_invalid": self.mld_invalid,
                "controls": list(self.cs.names), "p0": self.cs.p0.tolist(),
                "sigma": self.cs.sigma.tolist(),
                "u_truth": None if self.u_truth is None else self.u_truth.tolist(),
                "p_truth": (None if self.u_truth is None
                            else np.asarray(self.cs.to_physical(self.u_truth)).tolist()),
                "windows": [w.start.isoformat() for w in self.windows],
                "months": [w.month for w in self.windows], "n_steps": self.n_steps,
                "chunk_steps": self.model.chunk_steps, "first_check": str(self.first_check),
                "twin_targets_from": (None if self.targets_from is None
                                      else str(self.targets_from))}

    def physical(self, u) -> list:
        u = np.asarray(u, np.float64)
        return (self.cs.p0 * np.exp(self.cs.sigma * u)).tolist()

    def _row(self, path_name: str, row: dict):
        if self.out_dir is not None:
            C.append_row(self.out_dir / path_name, row)

    # -- evaluations ---------------------------------------------------------

    def fun(self, u) -> tuple[float, dict]:
        """Forward only: ``J(u)`` and ``{"J_data", "J_prior", "windows": [...]}``."""
        u = self.cs.check(u)
        per = []
        for w in self.windows:
            fin = self._forward(u, w)
            jw = float(self.loss_fn(w)(fin))
            C._free(fin)
            per.append({"start": w.start.isoformat(), "month": w.month, "J": jw,
                        "misfit_m2": jw / self.scale})
        jd = math.fsum(r["J"] for r in per)
        jp = _prior(u) if self.prior else 0.0
        return jd + jp, {"J_data": jd, "J_prior": jp, "windows": per}

    def forward_fields(self, u, w: Window, extra: Callable | None = None) -> dict:
        """Forward only at ``u`` over window ``w``: the data term ``J`` and the node field
        ``mld_sum`` (host ``[nod2D]``, the accumulator = Σ over the window's steps), the
        window-end top-layer temperature ``sst_field`` (host ``[nod2D]``), plus
        ``extra(final carry)`` as a float if given."""
        fin = self._forward(np.asarray(u, np.float64), w)
        out = {"J": float(self.loss_fn(w)(fin)),
               "mld_sum": np.asarray(jax.device_get(fin[1]["mld"]), np.float64),
               "sst_field": np.asarray(jax.device_get(fin[0].T[:, 0]), np.float64)}
        if extra is not None:
            out["extra"] = float(extra(fin))
        C._free(fin)
        return out

    def masked_J(self, w: Window, mld_sum, mask, ref_sum) -> float:
        """The data term of window ``w`` from a node field, with the ``mask`` nodes replaced by
        ``ref_sum`` (held fixed): the cost :meth:`masked_loss_fn` differentiates."""
        s = np.where(np.asarray(mask, bool), ref_sum, mld_sum)
        return float(self.scale * self._misfit(self._cell_map(jnp.asarray(s), self._n),
                                               w.target, w.weight))

    def masked_loss_fn(self, w: Window, mask) -> Callable:
        """``carry -> data term`` with the ``mask`` nodes of the MLD sum held fixed
        (``stop_gradient``): same value as :meth:`loss_fn`, gradient without those nodes."""
        m = jnp.asarray(np.asarray(mask, bool))

        def loss(carry):
            s = carry[1]["mld"]
            s = jnp.where(m, jax.lax.stop_gradient(s), s)
            return self.scale * self._misfit(self._cell_map(s, self._n), w.target, w.weight)
        loss.__name__ = f"mld_misfit_masked_{w.start:%Y%m%d}"
        return loss

    def grad_window(self, u, w: Window, loss: Callable | None = None, *,
                    check_linear=False) -> tuple[float, np.ndarray, dict]:
        """``(loss, d loss/du, provenance row)`` of ONE window (default loss: its data term)."""
        row = C.run_grad(self.model, (loss or self.loss_fn(w),),
                         params=np.asarray(u, np.float64).copy(), start=w.start, init=w.state,
                         check_linear=check_linear, action="calib_grad")[0]
        if row.get("status") != "ok":
            raise RuntimeError(f"window {w.start}: gradient status {row.get('status')}")
        g = np.asarray(row.pop("_result")["grad"], np.float64).reshape(-1)
        return float(row["loss"]), g, row

    def cell_jacobian(self, u, w: Window) -> tuple[np.ndarray, np.ndarray, float]:
        """``(Jac [n_cells, n], cell MLD [n_cells], seconds)``: d(window-mean cell MLD)/du by
        forward mode (:func:`fesom_jax.adjoint_check.window_jvp`, one params tangent per
        control, on pinned chunk-entry states when the window has several chunks), and the
        cell field at ``u`` from the same tangent program's primal. The tangent program is the
        derivative of the smooth pieces only; a level switch of the MLD pick contributes
        nothing (like the adjoint)."""
        m = self.model
        u = np.asarray(u, np.float64)
        nc = m.chunks_for(self.n_steps)
        carry = m.init_carry(w.state)
        pins = (adjoint_check.forward_states(m.chunk_fn, u, carry, nc, m.xs_fn(w.start))
                if nc > 1 else None)
        t0 = time.time()
        cols, cell = [], None
        for i in range(self.cs.n):
            dp = np.zeros(self.cs.n, np.float64)
            dp[i] = 1.0
            r = adjoint_check.window_jvp(m.chunk_fn_tl, u, carry, None, dp, nc,
                                         m.xs_fn(w.start), pinned_states=pins,
                                         compiler_options=C.TL_COMPILER_OPTIONS)
            cols.append(np.asarray(self._cell_map(jnp.asarray(r.tangent[1]["mld"]), self._n)))
            if cell is None:
                cell = np.asarray(self._cell_map(jnp.asarray(r.state[1]["mld"]), self._n))
        C._free(carry)
        return np.stack(cols, 1), cell, time.time() - t0

    def fun_and_grad(self, u):
        """``(J, ∇J, aux)`` at ``u``: every window's forward + adjoint, summed on the host."""
        u = self.cs.check(u)
        self.n_evals += 1
        per, grads = [], []
        for w in self.windows:
            key = w.start.isoformat()
            lin = False if key in self._linear_done else self.first_check
            row = C.run_grad(self.model, (self.loss_fn(w),), params=u.copy(), start=w.start,
                             init=w.state, check_linear=lin, action="calib_grad")[0]
            if row.get("status") != "ok":
                raise RuntimeError(f"window {key}: gradient status {row.get('status')}")
            self._linear_done.add(key)
            g = np.asarray(row.pop("_result")["grad"], np.float64).reshape(-1)
            grads.append(g)
            jw = float(row["loss"])
            info = {"start": key, "month": w.month, "J": jw, "misfit_m2": jw / self.scale,
                    "g": g.tolist(), "check_linear": str(lin),
                    "linear_check": row.get("linear_check"), "compile_s": row["compile_s"],
                    "run_s": row["run_s"], "n_compiles": row["n_compiles"],
                    "forward_seconds": row["forward_seconds"],
                    "reverse_seconds": row["reverse_seconds"]}
            per.append(info)
            row.update({"eval": self.n_evals, "u": u.tolist(), "p": self.physical(u),
                        "window": key, "month": w.month, "J_w": jw,
                        "misfit_m2": jw / self.scale, "g_w": g.tolist(),
                        "calib": self.config()})
            self._row("windows.jsonl", row)
        jd = math.fsum(r["J"] for r in per)
        gd = np.array([math.fsum(col) for col in np.stack(grads, 0).T.tolist()], np.float64)
        jp = _prior(u) if self.prior else 0.0
        g = gd + (u if self.prior else 0.0)
        aux = {"J_data": jd, "J_prior": jp, "p": self.physical(u), "g_data": gd.tolist(),
               "compile_s": sum(r["compile_s"] for r in per),
               "run_s": sum(r["run_s"] for r in per), "windows": per}
        return jd + jp, g, C._jsonable(aux)


# ---------------------------------------------------------------------------
# optimisers


def _run_info(calib: Calibration, action: str, *, timer=None, **extra) -> dict:
    """The ``core2_adj`` provenance keys (compile/run seconds from ``timer``), the cost
    settings under ``calib``, and ``extra`` (which may set ``direction`` / ``h``)."""
    return C._jsonable({**C.provenance(calib.model, action, timer=timer),
                        "calib": calib.config(), **extra})


def _write_config(out_dir: Path, cfg: dict) -> None:
    """``run_config.json``: written once; a later run in the same directory must agree on the
    cost, windows and controls (a resume)."""
    path = out_dir / "run_config.json"
    keys = ("calib", "optimizer", "options", "free")
    if path.exists():
        old = json.loads(path.read_text())
        for k in keys:
            if json.dumps(old.get(k), sort_keys=True) != json.dumps(cfg.get(k), sort_keys=True):
                raise ValueError(f"{path}: '{k}' differs from this run's (resume with the same "
                                 f"arguments, or use a new --out)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        json.dump(cfg, f, indent=1)


def restrict(fun_and_grad, u_full0, free) -> Callable:
    """``fun_and_grad`` on the ``free`` components of u only (indices), the others held at
    ``u_full0``: ``v -> (f, g[free], aux)`` with ``aux["u_full"]`` / ``aux["g_full"]``."""
    u_full0 = np.array(u_full0, np.float64).reshape(-1)
    free = [int(i) for i in free]

    def fg(v):
        u = u_full0.copy()
        u[free] = np.asarray(v, np.float64).reshape(-1)
        out = fun_and_grad(u)
        g = np.asarray(out[1], np.float64).reshape(-1)
        aux = dict(out[2]) if len(out) > 2 else {}
        aux.update({"u_full": u.tolist(), "g_full": g.tolist(), "free": free})
        return out[0], g[free], aux
    return fg


def free_indices(cs: ControlSet, free) -> list:
    """Indices of the controls named in ``free`` (names or indices; ``None`` = all)."""
    if free is None:
        return list(range(cs.n))
    names = list(cs.names)
    return sorted({names.index(c) if isinstance(c, str) else int(c) for c in free})


def run_lbfgs(calib: Calibration, out_dir, *, u0=None, maxiter: int = 30, maxeval: int = 30,
              gtol: float = 0.0, ftol: float = 0.0, ftol_iters: int = 3,
              options: optim_lbfgs.LBFGSOptions | None = None,
              free=None, wrap: Callable | None = None) -> optim_lbfgs.Result:
    """L-BFGS from ``u0`` (default 0 = the first guess); log ``out_dir/iterations.jsonl``
    (resumed if it exists). ``free``: optimise only these controls (names or indices), the
    others stay at ``u0``; the log then holds the free components (``aux.u_full`` the whole
    vector) and ``Result.u`` is the free part."""
    out_dir = Path(out_dir)
    opts = options or optim_lbfgs.LBFGSOptions()
    idx = free_indices(calib.cs, free)
    info = _run_info(calib, "calib", optimizer="lbfgs", free=[calib.cs.names[i] for i in idx])
    _write_config(out_dir, {"calib": calib.config(), "optimizer": "lbfgs",
                            "options": C._jsonable(vars(opts)),
                            "free": [calib.cs.names[i] for i in idx], "provenance": info})
    u0 = np.zeros(calib.cs.n) if u0 is None else np.asarray(u0, np.float64)
    base = calib.fun_and_grad if wrap is None else wrap(calib.fun_and_grad)
    fg = base if len(idx) == calib.cs.n else restrict(base, u0, idx)
    return optim_lbfgs.minimize(fg, u0[idx], maxiter=maxiter, maxeval=maxeval,
                                gtol=gtol, ftol=ftol, ftol_iters=ftol_iters,
                                log_path=out_dir / "iterations.jsonl", resume=True,
                                options=opts, run_info=info)


def run_holdout(calib: Calibration, points: dict, out_dir) -> list:
    """Forward-only misfits of ``calib``'s windows (the held-out months) at each
    ``{label: u}``: one ``holdout.jsonl`` row per point with J_data, the per-window misfits
    [m²] and their sum; the last row compares every point with the first."""
    rows = []
    for label, u in points.items():
        with C._Timer() as t:
            J, aux = calib.fun(u)
        rows.append({**_run_info(calib, "holdout", timer=t), "point": label,
                     "u": np.asarray(u, np.float64).tolist(), "p": calib.physical(u),
                     "J_data": aux["J_data"], "windows": aux["windows"],
                     "misfit_sum_m2": math.fsum(w["misfit_m2"] for w in aux["windows"])})
    ref = rows[0]
    for r in rows[1:]:
        r["rel_change_vs_" + ref["point"]] = r["misfit_sum_m2"] / ref["misfit_sum_m2"] - 1.0
        r["rel_change_per_window"] = [a["misfit_m2"] / b["misfit_m2"] - 1.0
                                      for a, b in zip(r["windows"], ref["windows"])]
    for r in rows:
        C.append_row(Path(out_dir) / "holdout.jsonl", C._jsonable(r))
        print(json.dumps(C._jsonable({k: r.get(k) for k in (
            "point", "p", "J_data", "misfit_sum_m2", "rel_change_vs_" + ref["point"])})),
            flush=True)
    return rows


def _host_device():
    try:
        return jax.devices("cpu")[0]
    except RuntimeError:                         # no CPU backend registered: stay on default
        return None


def adam_minimize(fun_and_grad, u0, *, n_evals: int = ADAM_EVALS, lr: float = ADAM_LR,
                  log_path=None, run_info: dict | None = None,
                  pause_after: int | None = None) -> dict:
    """``optax.adam(optax.cosine_decay_schedule(lr, n_evals))`` for a fixed budget of
    ``n_evals`` evaluations of ``fun_and_grad(u) -> (f, g[, aux])``, on the host in float64.

    Logs like :func:`fesom_jax.optim_lbfgs.minimize` (``header``, ``eval``, ``stop`` rows). An
    existing log is replayed (each logged ``u`` must equal the replayed iterate bitwise) and the
    run continues. Returns ``{"u_last", "f_last", "u_best", "f_best", "n_eval"}`` over the
    EVALUATED iterates. ``pause_after``: stop this invocation once the log holds that many
    evaluations (a ``pause`` row, not ``stop``); a later call with the same settings resumes on
    the same schedule (the learning-rate probe runs this way)."""
    import optax
    dev = _host_device()

    def host():
        """The optimiser's own arithmetic on the host CPU (bitwise on any machine). ONLY that:
        ``fun_and_grad`` runs outside it, on the default device (a GPU when there is one)."""
        return jax.default_device(dev) if dev is not None else _nullctx()

    u = np.array(u0, np.float64).reshape(-1)
    header = {"type": "header", "optimizer": "adam", "lr": float(lr), "n_evals": int(n_evals),
              "schedule": "cosine_decay", "u0": u.tolist(), "n": int(u.size),
              "time": time.time(), "run_info": run_info or {}}
    logged = []
    fh = None
    with host():
        opt = optax.adam(optax.cosine_decay_schedule(float(lr), int(n_evals)))
        state = opt.init(jnp.asarray(u, jnp.float64))
        sched = optax.cosine_decay_schedule(float(lr), int(n_evals))

    def update(u, g, state):
        with host():
            upd, state = opt.update(jnp.asarray(g, jnp.float64), state,
                                    jnp.asarray(u, jnp.float64))
            return np.asarray(optax.apply_updates(jnp.asarray(u, jnp.float64), upd),
                              np.float64), state

    def lr_at(k):
        with host():
            return float(sched(k))

    if log_path is not None:
        log_path = Path(log_path)
        if log_path.exists():
            old, logged = optim_lbfgs._read_log(log_path)
            if old is None or (old.get("optimizer"), old.get("lr"), old.get("n_evals"),
                               old.get("u0")) != ("adam", header["lr"], header["n_evals"],
                                                  header["u0"]):
                raise ValueError(f"{log_path}: not an Adam log with these settings")
            for rec in logged:
                if np.array(rec["u"], np.float64).tobytes() != u.tobytes():
                    raise ValueError(f"{log_path}: evaluation {rec['eval']} does not match "
                                     f"the replayed iterate")
                u, state = update(u, rec["g"], state)
            fh = open(log_path, "a")
            if logged:
                optim_lbfgs._append(fh, {"type": "resume", "after_eval": len(logged),
                                         "time": time.time()})
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log_path, "x")
            optim_lbfgs._append(fh, header)
    hist = [(np.array(r["u"], np.float64), float(r["f"])) for r in logged]
    try:
        end = int(n_evals) if pause_after is None else min(int(n_evals), int(pause_after))
        for k in range(len(logged), end):
            t0 = time.time()
            out = fun_and_grad(u.copy())
            dt = time.time() - t0
            f, g = float(out[0]), np.array(out[1], np.float64).reshape(-1)
            rec = {"type": "eval", "eval": k + 1, "u": u.tolist(), "f": f, "g": g.tolist(),
                   "lr": lr_at(k), "eval_s": dt}
            if len(out) > 2:
                rec["aux"] = out[2]
            if fh is not None:
                optim_lbfgs._append(fh, rec)
            hist.append((u.copy(), f))
            u, state = update(u, g, state)
        ib = int(np.argmin([h[1] for h in hist])) if hist else -1
        res = {"u_last": hist[-1][0].tolist() if hist else None,
               "f_last": hist[-1][1] if hist else None,
               "u_best": hist[ib][0].tolist() if hist else None,
               "f_best": hist[ib][1] if hist else None, "n_eval": len(hist),
               "u_next": u.tolist()}
        if fh is not None:
            optim_lbfgs._append(fh, {"type": "stop" if end == int(n_evals) else "pause",
                                     "status": "budget" if end == int(n_evals)
                                     else "paused", **res, "time": time.time()})
    finally:
        if fh is not None:
            fh.close()
    return res


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run_adam(calib: Calibration, out_dir, *, u0=None, n_evals: int = ADAM_EVALS,
             lr: float = ADAM_LR, pause_after: int | None = None,
             wrap: Callable | None = None) -> dict:
    out_dir = Path(out_dir)
    info = _run_info(calib, "calib", optimizer="adam")
    _write_config(out_dir, {"calib": calib.config(), "optimizer": "adam",
                            "options": {"lr": float(lr), "n_evals": int(n_evals),
                                        "schedule": "cosine_decay"}, "provenance": info})
    u0 = np.zeros(calib.cs.n) if u0 is None else np.asarray(u0, np.float64)
    fg = calib.fun_and_grad if wrap is None else wrap(calib.fun_and_grad)
    return adam_minimize(fg, u0, n_evals=n_evals, lr=lr,
                         log_path=out_dir / "iterations.jsonl", run_info=info,
                         pause_after=pause_after)


# ---------------------------------------------------------------------------
# checks


def run_fd(calib: Calibration, u=None, *, components=None, hs=(1e-2, 1e-3, 1e-4),
           repeats: int = 1, plateau_rtol: float = 1e-2) -> list:
    """Central differences of ``J`` along each unit vector ``e_i`` of u (``components``: names or
    indices, default all) against the adjoint ``∂J/∂u_i``. One row per (component, h) plus one
    ``plateau`` row per component."""
    u = calib.cs.check(np.zeros(calib.cs.n) if u is None else u)
    names = list(calib.cs.names)
    idx = ([names.index(c) if isinstance(c, str) else int(c) for c in components]
           if components else list(range(calib.cs.n)))
    with C._Timer() as tg:
        J0, g, aux = calib.fun_and_grad(u)
    rows = []
    for i in idx:
        e = np.zeros(calib.cs.n)
        e[i] = 1.0
        with C._Timer() as t:
            fr = adjoint_check.fd_sweep(lambda x: calib.fun(x)[0], u, e, list(hs), float(g[i]),
                                        repeats=repeats)
        base = _run_info(calib, "fd", timer=t, direction=f"u_{names[i]}", u=u.tolist(), J=J0,
                         ad_eval=tg.fields(), fd_eval=t.fields())
        for r in fr:
            rows.append({**base, "row": "h", "h": r.h, "fd": r.fd, "ad": r.ad,
                         "rel_err": r.rel_err, "noise_over_h": r.noise_over_h})
        pl = adjoint_check.plateau(fr, plateau_rtol)
        rows.append({**base, "row": "plateau", "plateau_rtol": plateau_rtol,
                     "plateau": None if pl is None else {k: v for k, v in pl.items()
                                                         if k != "rows"}})
    for r in rows:
        calib._row("fd.jsonl", r)
    return rows


JUMP_RATIO = 0.5          #: a node jumps when |second difference| > JUMP_RATIO·|central difference|
JUMP_ATOL = 1e-6          #: m (window-mean MLD); smaller second differences are never a jump
FD_RULE_RTOL = 0.10       #: adjoint vs FD plateau, MLD cost (full or masked)
FD_RULE_SST_RTOL = 1e-3   #: adjoint vs FD plateau, the smooth control functional
PLATEAU_RTOL = 0.02       #: FD values of a plateau agree with its median to this
PLATEAU_SST_RTOL = 1e-3   #: = FD_RULE_SST_RTOL: a plateau is resolved at the tolerance it is judged at
SST_JUMP_ATOL = 1e-9      #: K; smaller second differences of the terminal SST are never a jump
PLATEAU_MIN = 3           #: consecutive step sizes in a plateau
SPREAD_RTOL = 0.10        #: repeat spread of the adjoint gradient / |mean| above this: unconstrained


def jumping_nodes(m_minus, m0, m_plus, *, ratio: float = JUMP_RATIO,
                  atol: float = JUMP_ATOL) -> np.ndarray:
    """Nodes whose window-mean MLD is not smooth on ``[u − h, u + h]``: the second difference
    ``m₊ − 2m₀ + m₋`` exceeds ``ratio`` times the central difference ``m₊ − m₋`` (and ``atol``).

    A smooth node has ``|m₊ − 2m₀ + m₋| ≈ h²|m''|`` against ``|m₊ − m₋| ≈ 2h|m'|``, so the ratio
    is ``O(h)``; a jump of size ``d`` on one side gives ``≈ d`` against ``≈ d`` (ratio ≈ 1). A
    node whose MLD does not depend on u at all has both differences at round-off; ``atol``
    keeps it out. Kinks (a clipped interpolation weight) can be flagged too — harmless, the
    masked cost holds a flagged node fixed in the FD and in the adjoint alike."""
    m_minus, m0, m_plus = (np.asarray(a, np.float64) for a in (m_minus, m0, m_plus))
    d1 = np.abs(m_plus - m_minus)
    d2 = np.abs(m_plus - 2.0 * m0 + m_minus)
    return d2 > np.maximum(ratio * d1, atol)


def _plateau(hs, fds, ad, rtol, nmin=PLATEAU_MIN):
    rows = [adjoint_check.FDRow(h, fd, ad, abs(fd - ad) / max(abs(ad), 1e-300), 0.0)
            for h, fd in zip(hs, fds)]
    pl = adjoint_check.plateau(rows, rtol)
    if pl is None or pl["n"] < nmin:
        return None
    return {k: v for k, v in pl.items() if k != "rows"}


def masked_sst_loss(mesh, mask) -> Callable:
    """``carry -> area-mean window-end SST`` (:func:`fesom_jax.adjoint_costs.sst_area_mean`) with
    the ``mask`` nodes held fixed (``stop_gradient``): same value, gradient without them."""
    m = jnp.asarray(np.asarray(mask, bool))
    w = AC._surface_weights(mesh, None)

    def loss(carry):
        t0 = carry[0].T[:, 0]
        t0 = jnp.where(m, jax.lax.stop_gradient(t0), t0)
        return jnp.sum(w * t0) / jnp.sum(w)
    loss.__name__ = "sst_area_mean_masked"
    return loss


def run_fd_rule(calib: Calibration, u=None, *, components=None,
                hs=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4), repeats: int = 2,
                grad_repeats: int = 3) -> list:
    """The window-length FD check, per window and control (``fd_rule.jsonl``).

    For each window of ``calib`` separately (prior off; the prior's gradient is exact):
    (a) central FDs of the full data term ``J`` against its adjoint; (b) the same with the nodes
    that JUMP anywhere in the h-sweep (:func:`jumping_nodes`, union over h for that control) held
    fixed at their value at ``u`` — in the FD and, through :meth:`Calibration.masked_loss_fn`,
    in the adjoint; (c) central FDs of a smooth functional of the same window and controls, the
    terminal area-mean SST, against its adjoint. A plateau = at least ``PLATEAU_MIN``
    consecutive step sizes agreeing with their median to ``PLATEAU_RTOL`` (SST:
    ``PLATEAU_SST_RTOL``). Every adjoint gradient is computed ``grad_repeats`` times at the same
    u, FIRST, and compared by its mean; the repeat spread (max − min) / |mean| is reported per
    control (a GPU recomputes the window non-deterministically, and round-off flips discrete
    switches within a few steps). (c') is (c) with the nodes whose window-end SST jumps in the
    sweep held fixed, like (b): over a 6-hour window the perturbation itself flips discrete
    switches (convective mixing) that move the terminal SST, so (c) is not smooth either.
    Verdicts, in this order: ``unconstrained_spread`` if the spread of the full data-term
    gradient exceeds ``SPREAD_RTOL``; ``unconstrained_noise`` if every |FD| of (a) is below its
    noise floor (spread of ``repeats`` forwards at u, divided by h); ``pass`` if (a) or (b) has
    a plateau within ``FD_RULE_RTOL`` of the adjoint AND (c) or (c') has one within
    ``FD_RULE_SST_RTOL``; else ``fail``. ``verdict_rule`` is the same with (c) alone. Rows: ``spread`` (per window: the repeated
    gradients), ``h`` (one per window, control and h, with the number of jumping nodes),
    ``summary`` (per window and control, the verdict)."""
    u = calib.cs.check(np.zeros(calib.cs.n) if u is None else u)
    names = list(calib.cs.names)
    idx = ([names.index(c) if isinstance(c, str) else int(c) for c in components]
           if components else list(range(calib.cs.n)))
    mesh = calib.model.mesh
    sst = AC.terminal(lambda st: AC.sst_area_mean(st, mesh))
    ws = np.asarray(AC._surface_weights(mesh, None), np.float64)
    wet = ws > 0.0

    def sst_masked(field, mask, ref):
        return math.fsum((ws * np.where(mask, ref, field)).tolist()) / math.fsum(ws.tolist())
    n = float(calib.n_steps)
    hs = [float(h) for h in hs]
    out = []
    for w in calib.windows:
        key = w.start.isoformat()
        with C._Timer() as t0:
            c0 = [calib.forward_fields(u, w, sst) for _ in range(max(1, int(repeats)))]
        m0 = c0[0]["mld_sum"] / n
        J0, S0 = c0[0]["J"], c0[0]["extra"]
        noise_J = max(c["J"] for c in c0) - min(c["J"] for c in c0)
        noise_S = max(c["extra"] for c in c0) - min(c["extra"] for c in c0)
        dnode = [np.abs(c["mld_sum"] - c0[0]["mld_sum"]) / n for c in c0[1:]]
        noise_node = float(max(d.max() for d in dnode)) if dnode else 0.0
        n_node_diff = int(max((d > JUMP_ATOL).sum() for d in dnode)) if dnode else 0
        n_node_diff_1m = int(max((d > 1.0).sum() for d in dnode)) if dnode else 0
        nrep = max(1, int(grad_repeats))
        with C._Timer() as ta:
            ga_r, gs_r, Ja_r, lin = [], [], [], None
            for k in range(nrep):
                Ja_k, g_k, row_k = calib.grad_window(
                    u, w, check_linear=calib.first_check if k == 0 else False)
                ga_r.append(g_k)
                Ja_r.append(Ja_k)
                lin = row_k.get("linear_check") if k == 0 else lin
                gs_r.append(calib.grad_window(u, w, sst)[1])
        ga_r, gs_r = np.stack(ga_r), np.stack(gs_r)
        ga, gs = ga_r.mean(0), gs_r.mean(0)
        Ja = float(np.mean(Ja_r))
        sp_a = (ga_r.max(0) - ga_r.min(0)) / np.maximum(np.abs(ga), 1e-300)
        sp_c = (gs_r.max(0) - gs_r.min(0)) / np.maximum(np.abs(gs), 1e-300)
        base = {**_run_info(calib, "fd_rule"), "window": key, "month": w.month,
                "u": u.tolist(), "p": calib.physical(u), "J": J0, "J_adjoint_program": Ja,
                "J_adjoint_repeats": Ja_r, "sst": S0, "repeats": int(repeats),
                "grad_repeats": nrep, "noise_J": noise_J, "noise_sst": noise_S,
                "noise_node_mld_m": noise_node, "n_node_diff_repeat": n_node_diff,
                "n_node_diff_repeat_gt_1m": n_node_diff_1m, "linear_check": lin,
                "first_check": str(calib.first_check), "center_s": t0.fields(),
                "adjoint_s": ta.fields()}
        sprow = {**base, "row": "spread", "g_a_repeats": ga_r.tolist(),
                 "g_c_repeats": gs_r.tolist(), "g_a_mean": ga.tolist(), "g_c_mean": gs.tolist(),
                 "rel_spread_a": sp_a.tolist(), "rel_spread_c": sp_c.tolist(),
                 "controls": names}
        out.append(sprow)
        calib._row("fd_rule.jsonl", sprow)
        print(json.dumps(C._jsonable({"window": key, "g_a_repeats": ga_r, "rel_spread_a": sp_a,
                                      "rel_spread_c": sp_c, "noise_J": noise_J,
                                      "n_node_diff_repeat": n_node_diff,
                                      "n_node_diff_repeat_gt_1m": n_node_diff_1m})),
              flush=True)
        for i in idx:
            e = np.zeros(calib.cs.n)
            e[i] = 1.0
            fw = {}
            with C._Timer() as tf:
                for h in hs:
                    fw[h] = (calib.forward_fields(u + h * e, w, sst),
                             calib.forward_fields(u - h * e, w, sst))
            masks = {h: jumping_nodes(fw[h][1]["mld_sum"] / n, m0, fw[h][0]["mld_sum"] / n)
                     for h in hs}
            union = np.zeros_like(m0, bool)
            for h in hs:
                union |= masks[h]
            smasks = {h: jumping_nodes(fw[h][1]["sst_field"], c0[0]["sst_field"],
                                       fw[h][0]["sst_field"], atol=SST_JUMP_ATOL) & wet
                      for h in hs}
            union_s = np.zeros_like(m0, bool)
            for h in hs:
                union_s |= smasks[h]
            with C._Timer() as tm:
                mloss = calib.masked_loss_fn(w, union)
                sloss = masked_sst_loss(mesh, union_s)
                gb_r, gc2_r = [], []
                for _k in range(nrep):
                    Jb, gb_k, _ = calib.grad_window(u, w, mloss)
                    gb_r.append(gb_k)
                    gc2_r.append(calib.grad_window(u, w, sloss)[1])
            gb_r = np.stack(gb_r)
            gc2_r = np.stack(gc2_r)
            gc2 = gc2_r.mean(0)
            sp_c2 = float((gc2_r[:, i].max() - gc2_r[:, i].min())
                          / max(abs(gc2[i]), 1e-300))
            sref = c0[0]["sst_field"]
            gb_all = gb_r.mean(0)
            sp_b = float((gb_r[:, i].max() - gb_r[:, i].min())
                         / max(abs(gb_all[i]), 1e-300))
            ref = c0[0]["mld_sum"]
            rec = {"a": [], "b": [], "c": [], "c2": []}
            dname = f"u_{names[i]}"
            for h in hs:
                p_, m_ = fw[h]
                fa = (p_["J"] - m_["J"]) / (2 * h)
                fb = (calib.masked_J(w, p_["mld_sum"], union, ref)
                      - calib.masked_J(w, m_["mld_sum"], union, ref)) / (2 * h)
                fc = (p_["extra"] - m_["extra"]) / (2 * h)
                fc2 = (sst_masked(p_["sst_field"], union_s, sref)
                       - sst_masked(m_["sst_field"], union_s, sref)) / (2 * h)
                d = np.abs(p_["mld_sum"] - m_["mld_sum"]) / n
                rec["a"].append(fa)
                rec["b"].append(fb)
                rec["c"].append(fc)
                rec["c2"].append(fc2)
                row = {**base, "row": "h", "direction": dname, "h": h,
                       "fd_a": fa, "ad_a": float(ga[i]),
                       "rel_a": abs(fa - ga[i]) / max(abs(ga[i]), 1e-300),
                       "fd_b": fb, "ad_b": float(gb_all[i]),
                       "rel_b": abs(fb - gb_all[i]) / max(abs(gb_all[i]), 1e-300),
                       "fd_c": fc, "ad_c": float(gs[i]),
                       "rel_c": abs(fc - gs[i]) / max(abs(gs[i]), 1e-300),
                       "fd_c2": fc2, "ad_c2": float(gc2[i]),
                       "rel_c2": abs(fc2 - gc2[i]) / max(abs(gc2[i]), 1e-300),
                       "n_jump_sst": int(smasks[h].sum()), "n_jump_sst_union": int(union_s.sum()),
                       "noise_over_h_a": noise_J / h, "noise_over_h_c": noise_S / h,
                       "n_jump": int(masks[h].sum()), "n_jump_union": int(union.sum()),
                       "n_jump_gt_1m": int((d > 1.0).sum()),
                       "max_node_diff_m": float(d.max())}
                out.append(row)
                calib._row("fd_rule.jsonl", row)
            pa = _plateau(hs, rec["a"], float(ga[i]), PLATEAU_RTOL)
            pb = _plateau(hs, rec["b"], float(gb_all[i]), PLATEAU_RTOL)
            pc = _plateau(hs, rec["c"], float(gs[i]), PLATEAU_SST_RTOL)
            pc2 = _plateau(hs, rec["c2"], float(gc2[i]), PLATEAU_SST_RTOL)
            ok_a = pa is not None and pa["rel_err"] <= FD_RULE_RTOL
            ok_b = pb is not None and pb["rel_err"] <= FD_RULE_RTOL
            ok_c = pc is not None and pc["rel_err"] <= FD_RULE_SST_RTOL
            ok_c2 = pc2 is not None and pc2["rel_err"] <= FD_RULE_SST_RTOL
            unc = all(abs(f) <= noise_J / h for f, h in zip(rec["a"], hs))
            pre = ("unconstrained_spread" if sp_a[i] > SPREAD_RTOL else
                   "unconstrained_noise" if unc else None)
            verdict_rule = pre or ("pass" if (ok_a or ok_b) and ok_c else "fail")
            verdict = pre or ("pass" if (ok_a or ok_b) and (ok_c or ok_c2) else "fail")
            summ = {**base, "row": "summary", "direction": dname, "hs": hs,
                    "rel_spread_a": float(sp_a[i]), "rel_spread_b": sp_b,
                    "rel_spread_c": float(sp_c[i]), "g_b_repeats": gb_r[:, i].tolist(),
                    "rel_spread_c2": sp_c2, "g_c2_repeats": gc2_r[:, i].tolist(),
                    "plateau_a": pa, "plateau_b": pb, "plateau_c": pc, "plateau_c2": pc2,
                    "ok_a": ok_a, "ok_b": ok_b, "ok_c": ok_c, "ok_c2": ok_c2,
                    "verdict": verdict, "verdict_rule": verdict_rule,
                    "ad_c2": float(gc2[i]), "n_jump_sst_union": int(union_s.sum()),
                    "n_jump_sst_per_h": {f"{h:g}": int(smasks[h].sum()) for h in hs},
                    "ad_a": float(ga[i]), "ad_b": float(gb_all[i]), "ad_c": float(gs[i]),
                    "J_masked_adjoint_program": Jb,
                    "n_jump_per_h": {f"{h:g}": int(masks[h].sum()) for h in hs},
                    "n_jump_union": int(union.sum()), "fd_forwards_s": tf.fields(),
                    "masked_adjoint_s": tm.fields(),
                    "rules": {"fd_rtol": FD_RULE_RTOL, "sst_rtol": FD_RULE_SST_RTOL,
                              "plateau_rtol": PLATEAU_RTOL, "plateau_sst_rtol": PLATEAU_SST_RTOL,
                              "plateau_min": PLATEAU_MIN, "jump_ratio": JUMP_RATIO,
                              "jump_atol_m": JUMP_ATOL, "spread_rtol": SPREAD_RTOL,
                              "sst_jump_atol_K": SST_JUMP_ATOL}}
            out.append(summ)
            calib._row("fd_rule.jsonl", summ)
            print(json.dumps(C._jsonable({
                "window": key, "direction": dname, "verdict": verdict,
                "verdict_rule": verdict_rule, "ad_a": ga[i],
                "spread_a": sp_a[i], "spread_b": sp_b, "spread_c": sp_c[i], "spread_c2": sp_c2,
                "pa": pa, "pb": pb, "pc": pc, "pc2": pc2, "n_jump": summ["n_jump_per_h"],
                "n_jump_sst": summ["n_jump_sst_per_h"],
                "noise_J": noise_J})), flush=True)
    return out


def run_ident(calib: Calibration, points: dict) -> dict:
    """Identifiability (``ident.jsonl``): at each ``{label: u}`` the Gauss-Newton data Hessian
    ``H = (N_eff/σ_obs²)·Σ_w J_wᵀ diag(w_w/Σw_w) J_w`` with ``J_w`` = d(window-mean cell
    MLD)/du from forward-mode JVPs (:meth:`Calibration.cell_jacobian`); eigenvalues and
    eigenvectors (the prior adds I; "identifiable" = eigenvalue ≥ 10). Machinery check (no
    finite differences, so no jumps): each window's data gradient from the tangents,
    ``(N_eff/σ_obs²)·J_wᵀ (w/Σw)(m − o)``, against the ADJOINT gradient of the same window at
    the same u (``rel_tl_adj`` per control; at the twin truth both are round-off)."""
    res = {}
    k = calib.n_eff / calib.sigma_obs ** 2
    for label, u in points.items():
        u = calib.cs.check(u)
        H = np.zeros((calib.cs.n, calib.cs.n))
        g_tl = np.zeros(calib.cs.n)
        g_adj = np.zeros(calib.cs.n)
        per = []
        for w in calib.windows:
            Jac, cell, secs = calib.cell_jacobian(u, w)
            wt = np.asarray(w.weight, np.float64)
            wn = wt / wt.sum()
            Hw = k * Jac.T @ (wn[:, None] * Jac)
            gw = k * Jac.T @ (wn * (cell - np.asarray(w.target, np.float64)))
            H += Hw
            g_tl += gw
            _, ga_w, _ = calib.grad_window(u, w)
            g_adj += ga_w
            per.append({"window": w.start.isoformat(), "month": w.month, "H_w": Hw.tolist(),
                        "g_tl_w": gw.tolist(), "g_adj_w": ga_w.tolist(),
                        "rel_tl_adj_w": (np.abs(gw - ga_w)
                                         / np.maximum(np.abs(ga_w), 1e-300)).tolist(),
                        "jvp_s": secs,
                        "col_rms_m_per_u": np.sqrt((wn[:, None] * Jac ** 2).sum(0)).tolist()})
        ev, V = np.linalg.eigh(H)
        with C._Timer() as tj:
            Jfwd, auxf = calib.fun(u)          # forward-only J there (twin truth: the noise floor)
        row = {**_run_info(calib, "ident"), "point": label, "u": u.tolist(),
               "J_data_fwd": auxf["J_data"], "J_windows_fwd": auxf["windows"],
               "fwd_s": tj.fields(),
               "p": calib.physical(u), "H_data": H.tolist(), "eigvals": ev.tolist(),
               "eigvecs": V.T.tolist(), "identifiable": (ev >= 10.0).tolist(),
               "diag_H": np.diag(H).tolist(), "g_data_tl": g_tl.tolist(),
               "g_data_adj": g_adj.tolist(),
               "rel_tl_adj": (np.abs(g_tl - g_adj) / np.maximum(np.abs(g_adj), 1e-300)).tolist(),
               "controls": list(calib.cs.names), "windows": per,
               "posterior_sd_u": np.sqrt(np.diag(np.linalg.inv(H + np.eye(calib.cs.n)))).tolist()}
        calib._row("ident.jsonl", row)
        print(json.dumps(C._jsonable({"point": label, "eigvals": ev, "diag_H": np.diag(H),
                                      "eigvecs": V.T, "g_data_tl": g_tl, "g_data_adj": g_adj,
                                      "J_data_fwd": auxf["J_data"]})), flush=True)
        res[label] = row
    return res


def run_adam_probe(calib: Calibration, out_dir, lrs, *, u0=None, n_evals: int = ADAM_EVALS,
                   probe_evals: int = 8) -> dict:
    """The Adam learning-rate probe: for each ``lr`` the first ``probe_evals`` evaluations of the
    ``n_evals``-evaluation cosine schedule (``out_dir/lr_<lr>/iterations.jsonl``, resumable),
    all in one process (one compilation). The chosen ``lr`` has the lowest cost at the LAST
    probe evaluation — progress and stability together (an lr that overshoots and oscillates
    ends high even if it passed low); ties go to the smaller lr. ``adam_probe.json`` records
    the J trajectories and the choice."""
    out_dir = Path(out_dir)
    traj = {}
    for lr in lrs:
        sub = out_dir / f"lr_{lr:g}"
        run_adam(calib, sub, u0=u0, n_evals=n_evals, lr=lr, pause_after=probe_evals)
        _, evs = optim_lbfgs._read_log(sub / "iterations.jsonl")
        traj[f"{lr:g}"] = {"lr": float(lr), "f": [float(e["f"]) for e in evs],
                           "u": [e["u"] for e in evs],
                           "p": [calib.physical(e["u"]) for e in evs],
                           "eval_s": [float(e.get("eval_s", 0.0)) for e in evs]}
    chosen = choose_lr(traj)
    rec = {**_run_info(calib, "adam_probe"), "probe_evals": int(probe_evals),
           "n_evals": int(n_evals), "criterion": "lowest J at the last probe evaluation",
           "trajectories": traj, "chosen_lr": chosen}
    k = 0
    while (out_dir / f"adam_probe{'' if k == 0 else f'.{k}'}.json").exists():
        k += 1                                   # a resumed probe never overwrites
    with open(out_dir / f"adam_probe{'' if k == 0 else f'.{k}'}.json", "x") as f:
        json.dump(C._jsonable(rec), f, indent=1)
    print(json.dumps({"chosen_lr": chosen,
                      "f_last": {k: t["f"][-1] for k, t in traj.items()}}), flush=True)
    return rec


def lr_from_probe(path) -> float:
    """The learning rate an :func:`run_adam_probe` record chose (``adam_probe.json``)."""
    rec = json.loads(Path(path).read_text())
    lr = float(rec["chosen_lr"])
    if not (math.isfinite(lr) and lr > 0.0):
        raise ValueError(f"{path}: chosen_lr {lr!r} is not a positive number")
    return lr


def choose_lr(traj: dict) -> float:
    """The probe's choice from ``{name: {"lr", "f": [J per evaluation]}}``: the lowest J at the
    last evaluation, ties to the smaller lr; runs with a non-finite last J never win."""
    ok = [t for t in traj.values() if t["f"] and math.isfinite(t["f"][-1])]
    if not ok:
        raise ValueError("adam probe: no run has a finite cost")
    return float(min(ok, key=lambda t: (t["f"][-1], t["lr"]))["lr"])


# ---------------------------------------------------------------------------
# multi-job evaluations: one job per window, a coordinator that owns the optimiser


class WallBudget(Exception):
    """Raised before an evaluation that would end after the process's wall budget; the
    optimiser logs stay consistent (no row is written for it) and a resumed run continues."""


def wall_budgeted(fun_and_grad, budget_s: float, first_s: float) -> Callable:
    """``fun_and_grad`` that raises :class:`WallBudget` instead of starting an evaluation
    expected to end after ``budget_s`` seconds since the process started; the expected length
    is the previous evaluation's (``first_s`` for the first, which includes compilation)."""
    last = [None]

    def fg(u):
        est = last[0] if last[0] is not None else first_s
        if time.time() - _T_START + est > budget_s:
            raise WallBudget(f"next evaluation (~{est:.0f} s) would end after the wall budget "
                             f"{budget_s:.0f} s")
        t0 = time.time()
        out = fun_and_grad(u)
        last[0] = time.time() - t0
        return out
    return fg


class NeedEvaluation(Exception):
    """Raised by the file-backed ``fun_and_grad`` when evaluation ``k`` is not complete: the
    windows in ``missing`` have no result file yet (their jobs must run first)."""

    def __init__(self, k: int, u, missing):
        self.k, self.u, self.missing = int(k), np.asarray(u, np.float64), list(missing)
        super().__init__(f"evaluation {k}: windows {self.missing} missing")


def mj_dir(out_dir, k: int) -> Path:
    return Path(out_dir) / f"eval_{int(k):03d}"


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "x") as f:
        json.dump(C._jsonable(obj), f)
    os.replace(tmp, path)          # a reader sees the whole file or none


def mj_fun_and_grad(out_dir, months, *, prior: bool, cs: ControlSet, scale: float) -> Callable:
    """``fun_and_grad(u)`` backed by per-window result files: evaluation ``k`` (1 + the number
    of evaluations in ``out_dir/iterations.jsonl``) lives in ``eval_<k>/``: ``u.json`` (written
    here, the point) and one ``win_<MM>.json`` per month (written by :func:`run_winjob`). With
    every window present it returns ``(J, ∇J, aux)`` summed on the host exactly as
    :meth:`Calibration.fun_and_grad` does (``math.fsum`` in month order, prior added); otherwise
    it raises :class:`NeedEvaluation`. A window file for another ``u`` or a mixture of
    platforms is an error (a calibration runs on one platform). Non-finite window values
    (stored as strings) are read as floats; a point outside the controls' domain
    (:meth:`ControlSet.check`: non-finite or ``|u| > 50``) is returned as a failed evaluation
    (``f = g = nan``) WITHOUT posting it — the line search then shortens the step, as it does
    for a non-finite cost — and it counts as an evaluation."""
    log = Path(out_dir) / "iterations.jsonl"
    months = [int(m) for m in months]

    def fg(u):
        u = np.asarray(u, np.float64).reshape(-1)
        try:
            cs.check(u)
        except ValueError as exc:
            return float("nan"), np.full(u.size, np.nan), {
                "out_of_domain": str(exc), "J_data": float("nan"), "J_prior": float("nan"),
                "p": (cs.p0 * np.exp(np.clip(cs.sigma * u, -700.0, 700.0))).tolist(),
                "windows": []}
        k = 1 + (len(optim_lbfgs._read_log(log)[1]) if log.exists() else 0)
        d = mj_dir(out_dir, k)
        up = d / "u.json"
        if up.exists():
            got = np.array(json.loads(up.read_text())["u"], np.float64)
            if got.tobytes() != u.tobytes():
                raise ValueError(f"{up}: holds another point than the optimiser's proposal")
        else:
            _write_json_atomic(up, {"eval": k, "u": u.tolist(), "p": (cs.p0 * np.exp(
                cs.sigma * u)).tolist(), "time": time.time()})
        rows, missing = {}, []
        for m in months:
            wp = d / f"win_{m:02d}.json"
            if wp.exists():
                rows[m] = json.loads(wp.read_text())
            else:
                missing.append(m)
        if missing:
            raise NeedEvaluation(k, u, missing)
        per = [rows[m] for m in months]
        for r in per:
            if np.array(r["u"], np.float64).tobytes() != u.tobytes():
                raise ValueError(f"eval {k}, month {r['month']}: window file for another u")
        plats = sorted({str(r.get("platform")) for r in per})
        if len(plats) != 1:
            raise ValueError(f"eval {k}: windows from several platforms {plats}")
        def _sum(vals):                        # fsum raises on inf + (-inf): non-finite -> nan
            vals = [float(x) for x in vals]
            return math.fsum(vals) if all(math.isfinite(x) for x in vals) else float("nan")
        jd = _sum(r["J_w"] for r in per)
        gd = np.array([_sum(c) for c in zip(*[r["g_w"] for r in per])], np.float64)
        jp = _prior(u) if prior else 0.0
        g = gd + (u if prior else 0.0)
        aux = {"J_data": jd, "J_prior": jp, "p": (cs.p0 * np.exp(cs.sigma * u)).tolist(),
               "g_data": gd.tolist(), "platform": plats[0],
               "compile_s": sum(float(r.get("compile_s", 0.0)) for r in per),
               "run_s": sum(float(r.get("run_s", 0.0)) for r in per),
               "windows": [{"start": r.get("window"), "month": r["month"], "J": r["J_w"],
                            "misfit_m2": float(r["J_w"]) / scale, "g": r["g_w"],
                            "slurm_job": r.get("slurm_job"), "device": r.get("device"),
                            "run_s": r.get("run_s")} for r in per]}
        return jd + jp, g, C._jsonable(aux)
    return fg


def run_mjstep(out_dir, *, months, cs: ControlSet, prior: bool, scale: float, u0,
               maxiter: int, maxeval: int, gtol: float, ftol: float,
               options: optim_lbfgs.LBFGSOptions, free=None, config: dict | None = None,
               optimizer: str = "lbfgs", lr: float = ADAM_LR,
               n_evals: int = ADAM_EVALS) -> dict:
    """One coordinator step of a multi-job calibration (L-BFGS, or Adam with ``optimizer =
    "adam"``): replay the log, and either finish or name the evaluation whose windows must be
    computed next. Returns ``{"status": "need", "eval", "missing", "u"}`` or the final state.
    No model is built (host arithmetic)."""
    out_dir = Path(out_dir)
    idx = free_indices(cs, free)
    if optimizer == "adam":
        if len(idx) != cs.n:
            raise ValueError("--free is implemented for --optimizer lbfgs only")
        opt_cfg = {"optimizer": "adam", "options": {"lr": float(lr), "n_evals": int(n_evals),
                                                    "schedule": "cosine_decay"}}
    else:
        opt_cfg = {"optimizer": "lbfgs", "options": C._jsonable(vars(options))}
    if config is not None:
        _write_config(out_dir, {**config, **opt_cfg, "free": [cs.names[i] for i in idx]})
    u0 = np.zeros(cs.n) if u0 is None else np.asarray(u0, np.float64)
    fg = mj_fun_and_grad(out_dir, months, prior=prior, cs=cs, scale=scale)
    log = out_dir / "iterations.jsonl"
    try:
        if optimizer == "adam":
            r = adam_minimize(fg, u0, n_evals=n_evals, lr=lr, log_path=log, run_info=config)
            return {"status": "budget", "n_eval": r["n_eval"], "f": r["f_last"],
                    "u": r["u_last"], "u_best": r["u_best"], "f_best": r["f_best"],
                    "p": (cs.p0 * np.exp(cs.sigma * np.asarray(r["u_last"]))).tolist()}
        if len(idx) != cs.n:
            fg = restrict(fg, u0, idx)
        r = optim_lbfgs.minimize(fg, u0[idx], maxiter=maxiter, maxeval=maxeval, gtol=gtol,
                                 ftol=ftol, log_path=log, resume=True, options=options,
                                 run_info=config)
    except NeedEvaluation as need:
        return {"status": "need", "eval": need.k, "missing": need.missing,
                "u": need.u.tolist()}
    u_fin = u0.copy()
    u_fin[idx] = r.u
    return {"status": r.status, "n_iter": r.n_iter, "n_eval": r.n_eval, "f": r.f,
            "u": u_fin.tolist(), "p": (cs.p0 * np.exp(cs.sigma * u_fin)).tolist()}


def mj_finished(out_dir):
    """The last row of the run's log if it is a ``stop`` row (the calibration is over)."""
    log = Path(out_dir) / "iterations.jsonl"
    if not log.exists():
        return None
    for line in reversed([x for x in log.read_text().splitlines() if x.strip()]):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        return rec if rec.get("type") == "stop" else None
    return None


def _mj_try_coordinate(out_dir, k: int, coord: dict):
    """Run the coordinator once after evaluation ``k`` is complete (``k = 0``: before the
    first), unless evaluation ``k+1`` is already posted or the run is over. The first process to
    create ``eval_<k+1>/coord.lock`` (O_EXCL) does it; the others return."""
    out_dir = Path(out_dir)
    if k >= 1 and not all((mj_dir(out_dir, k) / f"win_{m:02d}.json").exists()
                          for m in coord["months"]):
        return None
    nd = mj_dir(out_dir, k + 1)
    if (nd / "u.json").exists() or mj_finished(out_dir) is not None:
        return None
    nd.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(nd / "coord.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, f"{os.environ.get('SLURM_JOB_ID', 'local')} {time.time()}\n".encode())
    os.close(fd)
    res = run_mjstep(out_dir, **coord)
    print(f"[tke_lbfgs] coordinator after evaluation {k}: {json.dumps(C._jsonable(res))}",
          flush=True)
    return res


def run_winloop(calib: Calibration, out_dir, coord: dict, *, poll_s: float = 15.0,
                max_wait_s: float = 4 * 3600.0) -> None:
    """A persistent window worker of a multi-job calibration: compile once, then for k = 1, 2, …
    wait for ``eval_<k>/u.json``, write ``eval_<k>/win_<MM>.json`` (:func:`run_winjob`), and
    if that completed the evaluation run the coordinator (:func:`_mj_try_coordinate`), which
    posts the next point. Returns when the run's log ends with ``stop``; a restarted worker
    skips the evaluations whose file it already wrote."""
    out_dir = Path(out_dir)
    w = calib.windows[0]
    mine = f"win_{w.month:02d}.json"
    k, waited = 1, 0.0
    while True:
        d = mj_dir(out_dir, k)
        if (d / mine).exists():
            _mj_try_coordinate(out_dir, k, coord)
            k += 1
            continue
        if mj_finished(out_dir) is not None:
            print(f"[tke_lbfgs] month {w.month}: the run has stopped", flush=True)
            return
        if not (d / "u.json").exists():
            _mj_try_coordinate(out_dir, k - 1, coord)
            if not (d / "u.json").exists():
                time.sleep(poll_s)
                waited += poll_s
                if waited > max_wait_s:
                    raise TimeoutError(f"month {w.month}: no point for evaluation {k} after "
                                       f"{waited:.0f} s")
                continue
        waited = 0.0
        run_winjob(calib, out_dir, k)
        _mj_try_coordinate(out_dir, k, coord)
        k += 1


def run_winjob(calib: Calibration, out_dir, k: int) -> dict:
    """One window of evaluation ``k`` of a multi-job calibration: the data term and its
    gradient at ``eval_<k>/u.json``, written to ``eval_<k>/win_<MM>.json`` (atomically; an
    existing file is kept and nothing is recomputed)."""
    if len(calib.windows) != 1:
        raise ValueError("a window job evaluates exactly one window")
    w = calib.windows[0]
    d = mj_dir(out_dir, k)
    wp = d / f"win_{w.month:02d}.json"
    if wp.exists():
        print(f"[tke_lbfgs] {wp} exists; nothing to do", flush=True)
        return json.loads(wp.read_text())
    u = np.array(json.loads((d / "u.json").read_text())["u"], np.float64)
    lin = calib.first_check if int(k) == 1 else False
    J, g, row = calib.grad_window(u, w, check_linear=lin)
    rec = {**row, "eval": int(k), "month": w.month, "window": w.start.isoformat(),
           "u": u.tolist(), "p": calib.physical(u), "J_w": J, "misfit_m2": J / calib.scale,
           "g_w": g.tolist(), "check_linear": str(lin), "calib": calib.config()}
    _write_json_atomic(wp, rec)
    print(json.dumps(C._jsonable({"eval": k, "month": w.month, "J_w": J, "g_w": g,
                                  "run_s": row.get("run_s")})), flush=True)
    return rec


def run_gradrep(calib: Calibration, u=None, *, repeats: int = 3) -> list:
    """The repeat spread of each window's adjoint data gradient at ``u`` (``gradrep.jsonl``):
    ``repeats`` gradients per window, (max − min)/|mean| per control. On a GPU the window is
    recomputed non-deterministically (switch flips), so this is the noise level against which
    any other gradient of the same window (another program, a tangent sweep) is compared."""
    u = calib.cs.check(np.zeros(calib.cs.n) if u is None else u)
    rows = []
    for w in calib.windows:
        with C._Timer() as t:
            gs = np.stack([calib.grad_window(u, w)[1] for _ in range(max(1, int(repeats)))])
        mean = gs.mean(0)
        row = {**_run_info(calib, "gradrep", timer=t), "window": w.start.isoformat(),
               "month": w.month, "u": u.tolist(), "g_repeats": gs.tolist(),
               "g_mean": mean.tolist(),
               "abs_spread": (gs.max(0) - gs.min(0)).tolist(),
               "rel_spread": ((gs.max(0) - gs.min(0))
                              / np.maximum(np.abs(mean), 1e-300)).tolist()}
        calib._row("gradrep.jsonl", row)
        print(json.dumps(C._jsonable({k: row[k] for k in ("window", "g_mean", "abs_spread",
                                                          "rel_spread")})), flush=True)
        rows.append(row)
    return rows


def run_fwd(calib: Calibration, u=None) -> dict:
    u = calib.cs.check(np.zeros(calib.cs.n) if u is None else u)
    with C._Timer() as t:
        J, aux = calib.fun(u)
    row = _run_info(calib, "fwd", timer=t, u=u.tolist(), p=calib.physical(u), J=J, **aux)
    calib._row("fwd.jsonl", row)
    return row


# ---------------------------------------------------------------------------
# CLI


def _floats(s):
    return [float(x) for x in str(s).split(",") if x.strip()]


def _parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["calib", "fd", "fdrule", "ident", "adamprobe", "gradrep",
                                       "mjstep", "winjob", "winloop", "fwd"])
    ap.add_argument("--mj-months", default="",
                    help="winloop: all months of the calibration (--months = this worker's)")
    ap.add_argument("--eval", type=int, default=None,
                    help="winjob: the evaluation index k (reads eval_<k>/u.json in --out)")
    ap.add_argument("--mode", choices=["real", "twin"], default="real")
    ap.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    ap.add_argument("--year", type=int, default=2013)
    ap.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--window-steps", type=int, default=12, help="window length (12 = 6 h)")
    ap.add_argument("--chunk-steps", type=int, default=None, help="default: the window")
    ap.add_argument("--sigma-obs", type=float, default=SIGMA_OBS)
    ap.add_argument("--n-eff", type=float, default=N_EFF)
    ap.add_argument("--prior", choices=["on", "off"], default="on")
    ap.add_argument("--sigma", default=None, help="control sigma(s), default ln 2 each")
    ap.add_argument("--twin-targets", default=None,
                    help="twin: load the targets from this twin_targets.npz (shared truth)")
    ap.add_argument("--truth", default=",".join(map(str, TRUTH_DEFAULT)),
                    help="twin: physical truth (c_k, c_eps, cd, alpha)")
    ap.add_argument("--u", default=None, help="start / evaluation point in u (4 values)")
    ap.add_argument("--p", default=None, help="start / evaluation point in physical values")
    ap.add_argument("--mld-invalid", choices=sorted(MLD_INVALID), default="depth")
    ap.add_argument("--woa", default=str(WOA_NPZ))
    ap.add_argument("--ice-sst", type=float, default=ICE_SST)
    ap.add_argument("--states-dir", default=str(C.STATES_DIR))
    ap.add_argument("--maxiter", type=int, default=30)
    ap.add_argument("--maxeval", type=int, default=30)
    ap.add_argument("--gtol", type=float, default=0.0)
    ap.add_argument("--ftol", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=ADAM_LR)
    ap.add_argument("--n-evals", type=int, default=ADAM_EVALS)
    ap.add_argument("--free", default="", help="lbfgs: optimise only these controls (names)")
    ap.add_argument("--lbfgs-expand", type=float, default=LBFGS_EXPAND,
                    help="lbfgs: cold-start length factor after an accepted step whose pair was "
                         "dropped (1 = the plain rule, which crawls on this cost)")
    ap.add_argument("--holdout-months", default="",
                    help="calib: after the fit, the misfit of these months at the start and "
                         "at the result (forward only)")
    ap.add_argument("--wall-budget-s", type=float, default=None,
                    help="calib: stop cleanly (resumable) before an evaluation that would end "
                         "later than this many seconds after the process started")
    ap.add_argument("--first-eval-s", type=float, default=900.0,
                    help="calib: expected length of the first evaluation (with compilation)")
    ap.add_argument("--lr-from", default=None,
                    help="adam: take lr from this adam_probe.json (its chosen_lr)")
    ap.add_argument("--lrs", default="0.03,0.1,0.3", help="adamprobe: learning rates in u")
    ap.add_argument("--probe-evals", type=int, default=8)
    ap.add_argument("--pause-after", type=int, default=None,
                    help="adam: stop this job after this many logged evaluations (resumable)")
    ap.add_argument("--ident-points", default="p0,truth",
                    help="ident: comma list of p0 | truth | start (= --u/--p)")
    ap.add_argument("--hs", default="1e-2,1e-3,1e-4")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--grad-repeats", type=int, default=3,
                    help="fdrule: adjoint gradients per window at the same u (repeat spread)")
    ap.add_argument("--first-check", choices=["auto", "full", "zero", "off"], default="auto",
                    help="linearity guard of each window's first gradient (auto: full on CPU "
                         "or 1-step chunks, else zero)")
    ap.add_argument("--components", default="", help="fd: comma list of control names")
    ap.add_argument("--adjoint", choices=["exact", "long_window"], default="exact")
    ap.add_argument("--freeze", action="append", default=[])
    ap.add_argument("--cut", action="append", default=[])
    ap.add_argument("--ice", choices=C.ICE_MODES, default="prescribed")
    ap.add_argument("--save-names", default="")
    ap.add_argument("--no-remat", action="store_true")
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def _mj_coord(args, cs: ControlSet, months) -> dict:
    """The coordinator settings of a multi-job calibration from the command line (the same for
    `mjstep` and every `winloop` worker, which must all get the same options)."""
    starts = [datetime.datetime(args.year, int(m), 1) for m in months]
    scale = args.n_eff / (2.0 * args.sigma_obs ** 2)
    u = (cs.to_u(_floats(args.p)) if args.p else
         np.array(_floats(args.u), np.float64) if args.u else None)
    u_truth = cs.to_u(_floats(args.truth)) if args.mode == "twin" else None
    cfg = {"calib": {
        "cost": "(N_eff/2)*sum_w misfit_w/sigma_obs^2 + prior*0.5*u.u",
        "misfit": "obs_compare.misfit of to_obs_surface(window-mean MLD) vs target",
        "sigma_obs_m": args.sigma_obs, "n_eff": args.n_eff, "prior": args.prior == "on",
        "scale": scale, "mode": args.mode, "woa": str(args.woa), "ice_sst": args.ice_sst,
        "mld_invalid": args.mld_invalid, "controls": list(cs.names), "p0": cs.p0.tolist(),
        "sigma": cs.sigma.tolist(),
        "u_truth": None if u_truth is None else u_truth.tolist(),
        "p_truth": None if u_truth is None else _floats(args.truth),
        "twin_targets_from": args.twin_targets,
        "windows": [s.isoformat() for s in starts], "months": [s.month for s in starts],
        "n_steps": args.window_steps, "chunk_steps": args.chunk_steps or args.window_steps,
        "first_check": args.first_check, "multijob": True}}
    return dict(months=[s.month for s in starts], cs=cs, prior=args.prior == "on",
                scale=scale, u0=u, maxiter=args.maxiter, maxeval=args.maxeval, gtol=args.gtol,
                ftol=args.ftol, options=optim_lbfgs.LBFGSOptions(expand=args.lbfgs_expand),
                free=[c for c in args.free.split(",") if c] or None, config=cfg,
                optimizer=args.optimizer, lr=args.lr, n_evals=args.n_evals)


def main(argv=None) -> int:
    args = _parse(argv)
    if args.lr_from:
        args.lr = lr_from_probe(args.lr_from)
        print(f"[tke_lbfgs] lr {args.lr} from {args.lr_from}", flush=True)
    sig = _floats(args.sigma) if args.sigma else None
    cs = tke_control_set() if sig is None else tke_control_set(sigma=sig if len(sig) > 1
                                                               else sig[0])
    starts = [datetime.datetime(args.year, int(m), 1) for m in args.months.split(",")]
    tag = os.environ.get("SLURM_JOB_ID") or datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out = (Path(args.out) if args.out
           else RUNS_DIR / f"{args.action}_{args.mode}_{args.optimizer}_{tag}")
    if args.action == "calib" and mj_finished(out) is not None:
        print(f"[tke_lbfgs] {out}: the run has stopped ({mj_finished(out).get('status')}); "
              f"nothing to do", flush=True)
        return 0
    if args.action == "mjstep":                  # the coordinator: no model, host arithmetic
        res = run_mjstep(out, **_mj_coord(args, cs, [s.month for s in starts]))
        print(json.dumps(C._jsonable(res)), flush=True)
        return 0
    if args.action == "winjob" and (len(starts) != 1 or args.eval is None):
        raise ValueError("winjob needs exactly one --months value and --eval")
    if args.action == "winloop" and (len(starts) != 1 or not args.mj_months):
        raise ValueError("winloop needs exactly one --months value and --mj-months")
    states_dir = Path(args.states_dir)
    first = states_dir / C.restart_tag(starts[0])
    model = build_calib_model(starts[0], args.window_steps, args.chunk_steps, cs=cs,
                              mld_invalid=args.mld_invalid, adjoint_cfg=C._adjoint_cfg(args),
                              ice_mode=args.ice,
                              save_names=tuple(s for s in args.save_names.split(",") if s),
                              rematerialize=not args.no_remat,
                              restart=first if first.is_dir() else None)
    u_truth = cs.to_u(_floats(args.truth)) if args.mode == "twin" else None
    calib = Calibration(model, starts, cs=cs, mode=args.mode, woa=args.woa,
                        sigma_obs=args.sigma_obs, n_eff=args.n_eff, prior=args.prior == "on",
                        u_truth=u_truth, ice_sst=args.ice_sst, states={starts[0]: model.s0},
                        states_dir=states_dir, out_dir=out, mld_invalid=args.mld_invalid,
                        first_check=args.first_check, targets_from=args.twin_targets)
    u = (cs.to_u(_floats(args.p)) if args.p else
         np.array(_floats(args.u), np.float64) if args.u else None)
    print(f"[tke_lbfgs] {args.action} mode={args.mode} windows={len(starts)} x "
          f"{args.window_steps} steps -> {out}", flush=True)
    if args.action == "calib":
        free = [c for c in args.free.split(",") if c] or None
        u0 = np.zeros(cs.n) if u is None else np.asarray(u, np.float64)
        wrap = (None if args.wall_budget_s is None else
                (lambda f: wall_budgeted(f, args.wall_budget_s, args.first_eval_s)))
        if args.optimizer != "lbfgs" and free:
            raise ValueError("--free is implemented for --optimizer lbfgs only")
        try:
            if args.optimizer == "lbfgs":
                r = run_lbfgs(calib, out, u0=u0, maxiter=args.maxiter, maxeval=args.maxeval,
                              gtol=args.gtol, ftol=args.ftol, free=free, wrap=wrap,
                              options=optim_lbfgs.LBFGSOptions(expand=args.lbfgs_expand))
                u_fin = u0.copy()
                u_fin[free_indices(cs, free)] = r.u
                summary = {"status": r.status, "n_iter": r.n_iter, "n_eval": r.n_eval,
                           "f": r.f, "u": u_fin.tolist(), "p": calib.physical(u_fin)}
            else:
                summary = run_adam(calib, out, u0=u0, n_evals=args.n_evals, lr=args.lr,
                                   pause_after=args.pause_after, wrap=wrap)
                u_fin = np.asarray(summary["u_last"], np.float64)
        except WallBudget as wb:
            print(f"[tke_lbfgs] paused: {wb}; resubmit with the same arguments to continue",
                  flush=True)
            return 0
        print(json.dumps(C._jsonable(summary)), flush=True)
        hold = [datetime.datetime(args.year, int(m), 1)
                for m in args.holdout_months.split(",") if m.strip()]
        if hold:
            if args.mode != "real":
                raise ValueError("--holdout-months needs --mode real")
            hcal = Calibration(model, hold, cs=cs, mode="real", woa=args.woa,
                               sigma_obs=args.sigma_obs, n_eff=args.n_eff,
                               prior=args.prior == "on", ice_sst=args.ice_sst,
                               states_dir=states_dir, out_dir=None,
                               mld_invalid=args.mld_invalid, hmap=calib.hmap)
            run_holdout(hcal, {"start": u0, "result": u_fin}, out)
    elif args.action == "fd":
        comps = [c for c in args.components.split(",") if c] or None
        rows = run_fd(calib, u, components=comps, hs=_floats(args.hs), repeats=args.repeats)
        for r in rows:
            if r["row"] == "plateau":
                print(json.dumps(C._jsonable({"direction": r["direction"],
                                              "plateau": r["plateau"]})), flush=True)
    elif args.action == "fdrule":
        comps = [c for c in args.components.split(",") if c] or None
        run_fd_rule(calib, u, components=comps, hs=_floats(args.hs), repeats=args.repeats,
                    grad_repeats=args.grad_repeats)
    elif args.action == "winjob":
        run_winjob(calib, out, args.eval)
    elif args.action == "winloop":
        run_winloop(calib, out, _mj_coord(args, cs, [int(m) for m in args.mj_months.split(",")]))
    elif args.action == "gradrep":
        run_gradrep(calib, u, repeats=args.grad_repeats)
    elif args.action == "adamprobe":
        run_adam_probe(calib, out, _floats(args.lrs), u0=u, n_evals=args.n_evals,
                       probe_evals=args.probe_evals)
    elif args.action == "ident":
        pts = {}
        for name in args.ident_points.split(","):
            if name == "p0":
                pts["p0"] = np.zeros(cs.n)
            elif name == "truth":
                pts["truth"] = cs.to_u(_floats(args.truth))
            elif name == "start":
                pts["start"] = u
            else:
                raise ValueError(f"unknown ident point {name!r}")
        run_ident(calib, pts)
    else:
        row = run_fwd(calib, u)
        print(json.dumps(C._jsonable({"J": row["J"], "J_data": row["J_data"],
                                      "windows": row["windows"]})), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
