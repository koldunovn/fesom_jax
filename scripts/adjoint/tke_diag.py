#!/usr/bin/env python
"""Diagnosis of the classical-TKE adjoint on the CORE2 reference model, built on the shared driver
``scripts/adjoint/core2_adj.py`` (same model, same compiled chunk programs, same JSONL rows).

    python scripts/adjoint/tke_diag.py ACTION [options]

Run it from a checkout with ``PYTHONPATH=<checkout>`` (``scripts/adjoint/job_{cpu,gpu}.sbatch``);
the driver's import guard refuses any other copy of ``fesom_jax``.

The model's own column inputs
    :func:`step_tke_inputs` returns the exact inputs the step hands to the TKE column core
    (:func:`fesom_jax.cvmix_tke.integrate_tke_column`) — after the surface/sea-ice fluxes, the
    surface-adjustment seam, the equation of state on the live zstar geometry and the halo
    exchange — by running :func:`fesom_jax.step.step` itself with a ``mixing_tke`` that records
    its inputs. Nothing of the assembly is re-implemented: :func:`column_core` on these inputs
    gives that step's ``tke_new`` bit for bit (CPU; tested). The regime logger, the column zoom
    and the linear-range estimate of ``fdlin`` all read these inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core2_adj as C  # noqa: E402  (runs the import guard)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fesom_jax import cvmix_tke  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax import tke as tkemod  # noqa: E402

#: The keyword inputs of the column core that :func:`step_tke_inputs` returns next to
#: :data:`fesom_jax.tke.COLUMN_INPUTS`.
COLUMN_KW = ("c_k", "c_eps", "cd", "alpha_tke", "mxl_min", "tke_min", "kappaM_max", "dt")


# ---------------------------------------------------------------------------
# the step's own column inputs


class _StashingTke:
    """Stands in for the ``tke`` module inside ``fesom_jax.step`` while one step is traced: every
    attribute is the real module's; ``mixing_tke`` additionally records the column inputs,
    assembled by the real :func:`fesom_jax.tke.tke_column_inputs` from exactly the arguments
    the step passes, and the core's keyword inputs, then runs the real ``mixing_tke``."""

    def __init__(self):
        self.stash = []

    def __getattr__(self, name):
        return getattr(tkemod, name)

    def mixing_tke(self, mesh, uv, bvfreq, tke, stress_node_surf, hnode, cfg, params, **kw):
        if getattr(params, "tke_nn", None) is not None:
            raise NotImplementedError("step_tke_inputs: params.tke_nn (per-column constants) "
                                      "is not supported")
        inp = tkemod.tke_column_inputs(mesh, uv, bvfreq, tke, stress_node_surf, hnode,
                                       Z3d=kw.get("Z3d"), bvfreq_raw=kw.get("bvfreq_raw"))
        inp.update(bvfreq=bvfreq, stress_node_surf=stress_node_surf,
                   c_k=params.tke_c_k, c_eps=params.tke_c_eps, cd=params.tke_cd,
                   alpha_tke=params.tke_alpha,
                   mxl_min=jnp.asarray(cfg.mxl_min, jnp.float64),
                   tke_min=jnp.asarray(cfg.tke_min, jnp.float64),
                   kappaM_max=jnp.asarray(cfg.kappaM_max, jnp.float64),
                   dt=jnp.asarray(kw["dt"], jnp.float64))
        self.stash.append(inp)
        return tkemod.mixing_tke(mesh, uv, bvfreq, tke, stress_node_surf, hnode, cfg, params,
                                 **kw)


_INPUT_PROGRAMS: dict = {}


def _inputs_program(model):
    """The jitted ``(state, x, params) -> inputs`` of ``model`` (one per model object) and the
    stand-in ``tke`` module its trace needs."""
    key = id(model)
    if key not in _INPUT_PROGRAMS:
        mesh, op, fs, phys = model.mesh, model.op, model.forcing.static, model.phys
        adj, dt, to_params = model.adjoint_cfg, model.dt, model.to_params
        proxy = _StashingTke()

        def fn(state, x, u):
            proxy.stash.clear()
            new = stepmod.step(state, mesh, op, None, to_params(u), dt=dt, is_first_step=False,
                               step_forcing=x, forcing_static=fs, adjoint_cfg=adj, **phys)
            if len(proxy.stash) != 1:
                raise RuntimeError("step_tke_inputs: the step called tke.mixing_tke "
                                   f"{len(proxy.stash)} times (tke_cfg missing?)")
            # the whole new State is an output, so the program is the full step (nothing after
            # the mixing call is dead code) plus the recorded inputs
            return dict(proxy.stash[0], new_state=new)

        _INPUT_PROGRAMS[key] = (model, proxy, jax.jit(fn))
    return _INPUT_PROGRAMS[key][1:]


def step_forcing(model, start, n: int):
    """The forcing of step ``n`` (0-based) of a window starting at ``start`` (one step, no
    leading axis), from the model's own forcing reader."""
    xs = model.stack_forcing([model.date(C._parse_date(start), int(n))])
    return jax.tree.map(lambda a: a[0], xs)


def step_tke_inputs(model, state, x, params=None) -> dict:
    """The TKE column-core inputs of one model step from ``state`` with step forcing ``x``.

    Runs one jitted :func:`fesom_jax.step.step` with the model's physics, adjoint settings and
    parameters (default ``model.params0``) in which ``tke.mixing_tke`` records its inputs.
    Returns a dict (device arrays): the :data:`fesom_jax.tke.COLUMN_INPUTS` (``tke_old``,
    ``Ssqr``, ``Nsqr``, ``dzw``, ``dzt``, ``forc_tke_surf``, ``nlev``), the core's keyword inputs
    :data:`COLUMN_KW` (0-d arrays), ``uvnode``, ``is_surf``, ``is_bot``, ``is_interior``,
    ``bvfreq`` (unmasked, what ``mo_convect`` reads), ``stress_node_surf``, and ``new_state``
    (that step's output State).

    :func:`column_core` on these inputs reproduces ``new_state.tke`` bit for bit (CPU). The
    driver's chunk program (the step inside a rematerialised scan) is a different compiled
    program: its ``tke_new`` agrees with this step's to round-off (measured on the 2013-01-01
    restart: about 0.2 % of the interfaces differ, by at most 5e-16), the spread of any two
    equivalent compilations of the step.
    """
    params = model.params0 if params is None else params
    proxy, prog = _inputs_program(model)
    with mock.patch.object(stepmod, "tke", proxy):
        return prog(jax.device_put(state), jax.device_put(x), jax.device_put(params))


def column_args(inp: dict):
    """``(args, kw)`` of :func:`fesom_jax.cvmix_tke.integrate_tke_column` from a
    :func:`step_tke_inputs` dict."""
    args = tuple(inp[k] for k in tkemod.COLUMN_INPUTS)
    kw = {k: inp[k] for k in COLUMN_KW}
    kw["mxl_min"], kw["tke_min"] = float(kw["mxl_min"]), float(kw["tke_min"])
    kw["kappaM_max"], kw["dt"] = float(kw["kappaM_max"]), float(kw["dt"])
    return args, kw


_CORE = {}


def column_core(inp: dict, cuts=frozenset()):
    """``(tke_new, KappaM, KappaH)`` of the column core on a :func:`step_tke_inputs` dict."""
    args, kw = column_args(inp)
    static = tuple(sorted((k, kw.pop(k)) for k in ("mxl_min", "tke_min", "kappaM_max", "dt")))
    key = (static, frozenset(cuts))
    if key not in _CORE:
        st = dict(static)

        def fn(args, kw):
            return cvmix_tke.integrate_tke_column(*args, **kw, **st, with_diags=False,
                                                  cuts=frozenset(cuts))
        _CORE[key] = jax.jit(fn)
    return _CORE[key](args, kw)


def regimes(inp: dict) -> dict:
    """:func:`fesom_jax.cvmix_tke.tke_regimes` on a :func:`step_tke_inputs` dict (host arrays)."""
    args, kw = column_args(inp)
    return jax.device_get(cvmix_tke.tke_regimes(*args, **kw))


# ---------------------------------------------------------------------------
# the SSH forward tolerance (fdlin)

from fesom_jax import ssh as sshmod  # noqa: E402

_SOLVE_SSH = sshmod.solve_ssh


def set_ssh_forward_tol(tol: float | None, maxiter: int | None = None):
    """Make every SSH solve traced from now on stop at the relative residual ``tol`` instead of
    the model's ``SOLTOL`` (``None`` restores the library function). A DIFFERENT forward model
    (the production solve stops early, so its iterate — and the trajectory — depends on the
    iteration count); for ``fdlin`` only, where the early stop makes the forward non-smooth at
    small perturbations. It must be set before a model's programs are traced, and applies to the
    FD forwards and the tangent's primal alike. ``maxiter`` (default: the library's, or 2000
    with a ``tol`` below 1e-9) caps the iterations."""
    if tol is None:
        sshmod.solve_ssh = _SOLVE_SSH
        return
    tol = float(tol)
    cap = int(maxiter) if maxiter is not None else (2000 if tol < 1e-9 else None)

    def solve_ssh(*a, **kw):
        kw.setdefault("forward_tol", tol)
        if cap is not None:
            kw.setdefault("maxiter", cap)
        return _SOLVE_SSH(*a, **kw)

    solve_ssh.forward_tol, solve_ssh.maxiter = tol, cap
    sshmod.solve_ssh = solve_ssh


def ssh_forward_tol():
    """``(forward_tol, maxiter)`` the SSH solves are traced with now."""
    f = sshmod.solve_ssh
    if f is _SOLVE_SSH:
        from fesom_jax.config import MAXITER, SOLTOL
        return SOLTOL, MAXITER
    return f.forward_tol, f.maxiter


# ---------------------------------------------------------------------------
# models, settings, norms and seed regions

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

from fesom_jax import adjoint, adjoint_check, adjoint_stab  # noqa: E402
from fesom_jax.adjoint_stab import (FREEZE_FLAGS, KNOWN_CUTS, TKE_CANDIDATE_FIELDS,  # noqa: E402
                                    AdjointConfig)

RUNS_DIR = Path("/work/ab0995/a270088/fesom_jax_dev/runs/tke")
#: The State leaves the screen maps and the ``prog`` norm use (never ``uvnode``: the step
#: overwrites it before reading it, so its incoming cotangent is identically zero).
MAP_FIELDS = ("tke", "T", "S", "uv", "eta_n")
WEIGHTS = ("tke", "prog", "ts")


def with_adjoint(model, cfg: AdjointConfig):
    """A copy of ``model`` with other derivative-only settings: same mesh, forcing reader, start
    State and parameters; new chunk programs (a different adjoint setting is a different
    program; the forward values are the same)."""
    new = dataclasses.replace(model, adjoint_cfg=cfg.validate(), chunk_fn=None,
                              _chunk_fn_tl=None)
    new.chunk_fn = C._make_chunk_fn(new, rematerialize=new.rematerialize)
    return new


def parse_setting(spec: str, base: AdjointConfig | None = None) -> AdjointConfig:
    """``"tke.km_e+freeze_convection"`` -> ``base`` plus those cuts (names of
    :data:`~fesom_jax.adjoint_stab.KNOWN_CUTS`) and freeze flags (:data:`FREEZE_FLAGS`, with or
    without the ``freeze_`` prefix). ``""`` or ``"none"`` -> ``base``. The non-boolean
    candidate fields (:data:`~fesom_jax.adjoint_stab.TKE_CANDIDATE_FIELDS`) are given as
    ``field=value``: a number (``inf`` allowed; write ``1e-8``, not ``1e+08``) or a word, e.g.
    ``"tke_pr_cut=1e-8+tke_pr_mode=at"``, ``"mixing_state+tke_kme_tke=3e-6"``."""
    base = AdjointConfig.exact() if base is None else base
    cuts, flags = set(base.diag_cuts), {}
    for tok in [t for t in spec.split("+") if t and t != "none"]:
        name, eq, val = tok.partition("=")
        if eq:
            if name not in TKE_CANDIDATE_FIELDS:
                raise ValueError(f"unknown setting {name!r} (known: {TKE_CANDIDATE_FIELDS})")
            try:
                flags[name] = float(val)
            except ValueError:
                flags[name] = val
        elif tok in KNOWN_CUTS:
            cuts.add(tok)
        elif tok in FREEZE_FLAGS or f"freeze_{tok}" in FREEZE_FLAGS:
            flags[tok if tok.startswith("freeze_") else f"freeze_{tok}"] = True
        else:
            raise ValueError(f"unknown cut or flag {tok!r}")
    return base._replace(diag_cuts=frozenset(cuts), **flags).validate()


def parse_also(text: str, default: int = 1) -> list:
    """``"long_window,freeze_convection*2"`` -> ``[("long_window", default),
    ("freeze_convection", 2)]`` (the screen's further settings and their repeats)."""
    out = []
    for item in [a.strip() for a in text.split(",") if a.strip()]:
        spec, _, k = item.partition("*")
        out.append((spec, int(k) if k else int(default)))
    return out


def norm_weights(model, state, kind: str) -> dict:
    """Weights per State leaf for :func:`fesom_jax.adjoint_check.power_iteration`: ``tke`` =
    Euclidean over the wet interfaces' tke; ``prog`` = volume-weighted T, S, ``uv``, ``eta_n``
    in units of 1 K, 0.1 psu, 0.1 m/s, 0.1 m (:func:`~fesom_jax.adjoint_check.volume_weights`);
    ``ts`` = the T and S part of ``prog`` alone (no fast barotropic ``uv``/``eta_n`` response,
    which dominates ``prog`` over a step even where nothing grows)."""
    if kind == "tke":
        return {"tke": np.asarray(model.mesh.node_iface_mask, np.float64)}
    if kind in ("prog", "ts"):
        w = {k: np.asarray(v) for k, v in
             adjoint_check.volume_weights(jax.device_put(state), model.mesh).items()}
        return w if kind == "prog" else {k: w[k] for k in ("T", "S")}
    raise ValueError(f"weights must be one of {WEIGHTS}")


def box_seed_mask(model, box) -> dict:
    """The seed region of a box: nodes inside the lon/lat bounds on the levels of its depth
    range (tke, T, S; ``eta_n`` on the nodes), elements whose three nodes are inside (``uv``)."""
    node, level = C.box_masks(model.mesh, box)
    en = np.asarray(model.mesh.elem_nodes)
    elem = node[en].all(axis=1)
    return {"tke": node[:, None] & level[None, :], "T": node[:, None] & level[None, :],
            "S": node[:, None] & level[None, :], "eta_n": node,
            "uv": (elem[:, None] & level[None, :])[..., None]}


def column_seed_mask(model, nodes) -> dict:
    """Seed region = whole columns ``nodes`` (tke only)."""
    m = np.zeros(int(model.mesh.nod2D), bool)
    m[np.asarray(nodes, int)] = True
    return {"tke": np.broadcast_to(m[:, None], (m.size, int(model.mesh.nl)))}


def zeros_like_state(state):
    return jax.tree.map(lambda a: np.zeros(np.shape(a), np.asarray(a).dtype), state)


def random_state_vector(state, weights, seed: int = 0):
    """A State-shaped host vector with standard-normal entries on the weighted leaves."""
    rng = np.random.default_rng(seed)
    out = zeros_like_state(state)
    reps = {}
    for f in weights:
        reps[f] = rng.standard_normal(np.shape(getattr(state, f)))
    return dataclasses.replace(out, **reps)


def _leaf_dict(state, names) -> dict:
    return {f: np.asarray(getattr(state, f)) for f in names}


def _state_from_leaves(like, leaves: dict):
    out = zeros_like_state(like)
    return dataclasses.replace(out, **{k: np.asarray(v) for k, v in leaves.items()})


# ---------------------------------------------------------------------------
# window maps on pinned states

class Window:
    """The window map of ``model`` from ``start_state``: stored chunk-entry States (one forward)
    and the pinned tangent / adjoint maps. The window starts at ``start`` (default the model's
    start; ``start_state`` must be the State at that time) and is ``n_chunks`` chunks long
    (default ``model.n_chunks``), so one model — one set of compiled chunk programs — serves
    windows at several times."""

    def __init__(self, model, start_state=None, pins=None, tl_options=C.TL_COMPILER_OPTIONS,
                 *, start=None, n_chunks: int | None = None):
        if start is not None and start_state is None and C._parse_date(start) != model.start:
            raise ValueError("Window: a start other than the model's needs its start_state")
        self.model = model
        self.start = model.start if start is None else C._parse_date(start)
        self.nc = model.n_chunks if n_chunks is None else int(n_chunks)
        self.n_steps = self.nc * model.chunk_steps
        self.params = model.params0
        self.carry0 = model.init_carry(start_state)
        self.xs = model.xs_fn(self.start)
        self._reg = {}
        self.tl_options = tl_options
        with C._Timer() as t:
            self.pins = (adjoint_check.forward_states(model.chunk_fn, self.params, self.carry0,
                                                      self.nc, self.xs)
                         if pins is None else pins)
        self.forward_timer = t

    def M(self, v, *, n_chunks=None, start_chunk: int = 0):
        """Tangent at the end of ``n_chunks`` chunks from chunk ``start_chunk`` (pinned)."""
        n = self.nc - start_chunk if n_chunks is None else int(n_chunks)
        s0 = (self.carry0 if start_chunk == 0
              else jax.device_put(self.pins[start_chunk]))
        pins = self.pins[start_chunk:]
        xs = (lambda c: self.xs(c + start_chunk))
        return adjoint_check.window_jvp(self.model.chunk_fn_tl, self.params, s0, v, None, n, xs,
                                        pinned_states=pins,
                                        compiler_options=self.tl_options).tangent

    def MT(self, w, **kw):
        kw.setdefault("check_linear", False)
        return adjoint_check.window_vjp(self.model.chunk_fn, self.params, self.carry0, w,
                                        self.nc, self.xs, pinned_states=self.pins,
                                        chunk_steps=self.model.chunk_steps, field_trace=False,
                                        **kw).state_cot

    def regimes(self, which: int) -> dict:
        """:func:`regimes` of the window's first step (``which=0``) or of the step after the
        window (``which=1``), from the pinned States (cached; the forward values do not depend
        on the adjoint setting)."""
        if which not in self._reg:
            m = self.model
            n = 0 if which == 0 else self.n_steps
            st = m.state_of(self.pins[0 if which == 0 else self.nc])
            self._reg[which] = regimes(step_tke_inputs(m, st, step_forcing(m, self.start, n)))
        return self._reg[which]

    def forward(self, carry):
        """The nonlinear window map (chunked forward) from a host carry; the end State (host)."""
        fin = adjoint.chunked_forward(self.model.chunk_fn, self.params,
                                      init_state=jax.device_put(carry), n_chunks=self.nc,
                                      xs_fn=self.xs)
        out = jax.device_get(fin)
        adjoint._free(fin)
        return out


def _top_cells(arr, k: int):
    """``[(index..., value)]`` of the ``k`` entries of largest ``|arr|`` (host)."""
    a = np.asarray(arr)
    flat = np.abs(a).ravel()
    k = min(int(k), flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    return [tuple(int(i) for i in np.unravel_index(j, a.shape)) + (float(a.ravel()[j]),)
            for j in idx]


def cell_info(model, reg: dict, node: int, k: int) -> dict:
    """One interface's regime record (``reg`` from :func:`regimes`) with its position."""
    lon, lat = C._lonlat_deg(model.mesh)
    out = {"node": int(node), "level": int(k), "lon": float(lon[node]), "lat": float(lat[node])}
    r = int(reg["regime"][node, k])
    out["regime"] = cvmix_tke.REGIMES[r] if r >= 0 else "dry"
    for key in ("tke", "tke_excess", "floor_hit", "mxl", "mxl0", "at_cap", "dKm_dtke", "Ssqr",
                "Nsqr", "Ri", "Pr", "F"):
        if key in reg:
            out[key] = float(np.asarray(reg[key])[node, k])
    return out


def log_top_cells(model, vec, reg: dict, weights: dict, top: int) -> list:
    """Top cells of a State vector per weighted leaf, with the tke regime of their column at
    that interface (node leaves) — element leaves (``uv``) report the element."""
    out = []
    for f, w in weights.items():
        a = np.asarray(getattr(vec, f)) * np.sqrt(np.broadcast_to(np.asarray(w),
                                                                  np.shape(getattr(vec, f))))
        if f == "uv":
            a = np.sqrt(np.sum(a * a, axis=-1))
        for cell in _top_cells(a, top):
            if a.ndim == 1:
                rec = {"field": f, "node": cell[0], "value": cell[-1]}
            elif f == "uv":
                rec = {"field": f, "elem": cell[0], "level": cell[1], "value": cell[-1]}
            else:
                rec = {"field": f, **cell_info(model, reg, cell[0], cell[1]), "value": cell[-1]}
            out.append(rec)
    return out


def vector_census(vec, reg: dict, weights: dict | None = None) -> dict:
    """Where a State vector lives: shares of its weighted squared norm over its node-level
    leaves (``tke`` at interfaces; ``T``, ``S`` at layers, counted at the interface of the same
    index) by mixing-length regime, level, floor contact, stratification, Prandtl branch and
    stiffness — the classification behind the hypothesis verdicts (forward values from
    :func:`regimes` of the matching step). ``weights`` default: the tke leaf alone."""
    weights = {"tke": 1.0} if weights is None else weights
    a = None
    for f in ("tke", "T", "S"):
        if f in weights:
            x = np.asarray(getattr(vec, f), float) ** 2 * np.broadcast_to(
                np.asarray(weights[f], float), np.shape(getattr(vec, f)))
            a = x if a is None else a + x
    r = np.asarray(reg["regime"])
    tot = float(a[r >= 0].sum())
    if tot <= 0:
        return {"norm2": 0.0}
    k = np.broadcast_to(np.arange(a.shape[1])[None, :], a.shape)

    def sh(m):
        return float(a[np.asarray(m) & (r >= 0)].sum() / tot)
    n2, pr, F = np.asarray(reg["Nsqr"]), np.asarray(reg["Pr"]), np.asarray(reg["F"])
    ex = np.asarray(reg["tke_excess"])
    out = {"norm2": tot,
           "regime": {name: sh(r == i) for i, name in enumerate(cvmix_tke.REGIMES)},
           "level": {"0": sh(k == 0), "1": sh(k == 1), "2": sh(k == 2), "3-5": sh((k >= 3) & (k <= 5)),
                     "6-10": sh((k >= 6) & (k <= 10)), ">10": sh(k > 10)},
           "floor_hit": sh(np.asarray(reg["floor_hit"])), "at_cap": sh(np.asarray(reg["at_cap"])),
           "tke_excess<1e-6": sh(ex < 1e-6), "tke_excess<1e-5": sh(ex < 1e-5),
           "tke_excess<1e-4": sh(ex < 1e-4),
           "N2<=0": sh(n2 <= 0.0), "0<N2<1e-8": sh((n2 > 0.0) & (n2 < 1e-8)),
           "0<N2<1e-6": sh((n2 > 0.0) & (n2 < 1e-6)), "0<N2<1e-5": sh((n2 > 0.0) & (n2 < 1e-5)),
           "Pr==1": sh(pr <= 1.0), "1<Pr<10": sh((pr > 1.0) & (pr < 10.0)), "Pr==10": sh(pr >= 10.0),
           "F>1": sh(F > 1.0), "F>10": sh(F > 10.0), "F>100": sh(F > 100.0)}
    return out


# ---------------------------------------------------------------------------
# action: screen

def _column_norm_maps(cot) -> dict:
    """``{field: [nod2D] or [elem2D]}`` column L2 norms of a State cotangent (host)."""
    out = {}
    for f in MAP_FIELDS:
        a = np.asarray(getattr(cot, f))
        out[f] = np.abs(a) if a.ndim == 1 else np.sqrt(np.sum(a.reshape(a.shape[0], -1) ** 2,
                                                             axis=1))
    return out


def growth_stats(trace, steps_per_chunk: int, n_steps: int) -> dict:
    """Per-step growth of a reverse-pass norm trace (seed first): the median / spread / worst
    stretch of the per-chunk rates (:func:`~fesom_jax.adjoint_stab.amplification`) and the
    overall rate ``(trace[-1] / trace[0]) ** (1 / n_steps)``."""
    amp = adjoint_stab.amplification(trace, steps_per_chunk)
    tr = [float(x) for x in trace]
    overall = ((tr[-1] / tr[0]) ** (1.0 / n_steps)
               if len(tr) > 1 and tr[0] > 0 and math.isfinite(tr[-1]) and tr[-1] > 0
               else float("nan"))
    return {"median": amp.median, "log_spread": amp.log_spread, "sustained": amp.sustained,
            "max": amp.max, "overall": overall, "total": (tr[-1] / tr[0]) if tr[0] > 0 else None,
            "clean": amp.clean, "reason": amp.reason, "per_chunk": list(amp.per_step)}


def run_screen(model, out_dir: Path, *, cost="box_T_vw", box=C.DEFAULT_BOX, repeats: int = 1,
               map_every: int | None = None, abort_at: float = C.ABORT_AT,
               check_linear=True, label: str = "screen", stride: int = 1) -> list:
    """Terminal-seed sweeps (``repeats`` of the same window, same compiled programs) with
    per-field traces, per-step growth statistics, and column-norm maps of the cotangent of
    ``tke``, ``T``, ``S``, ``uv`` (elements) and ``eta_n`` every ``map_every`` steps (default one
    day) saved to an npz per repeat. Overflow aborts as in the driver. ``stride`` > 1 stores
    every ``stride``-th chunk boundary in the forward and recomputes the others in the reverse
    sweep (host memory for long windows; the gradient is the same program's)."""
    steps_day = int(round(86400.0 / model.dt))
    map_every = steps_day if map_every is None else int(map_every)
    rows = []
    for r in range(int(repeats)):
        maps, bounds = {f: [] for f in MAP_FIELDS}, []

        def on_cot(c, cot):
            if (c * model.chunk_steps) % map_every == 0:
                for f, a in _column_norm_maps(cot).items():
                    maps[f].append(np.asarray(a, np.float64))   # float32 overflows at 3.4e38
                bounds.append(c)

        row = C.run_screen(model, cost, box=box, abort_at=abort_at, check_linear=check_linear,
                           on_cot=on_cot, stride=int(stride))
        steps_done = (len(row["trace"]) - 1) * model.chunk_steps
        g = growth_stats(row["trace"], model.chunk_steps, max(steps_done, 1))
        fields = {}
        ft = [{k: v for k, v in f.items() if k != "boundary"} for f in row.get("field_trace", [])]
        if len(ft) > 1:
            for name, a in adjoint_stab.amplification_by_field(ft, model.chunk_steps).items():
                fields[name] = {"median": a.median, "log_spread": a.log_spread}
        npz = out_dir / f"{label}_maps_r{r}.npz"
        path = C._save_npz(npz, {"boundary": np.asarray(bounds, np.int32),
                                 "day": np.asarray(bounds, np.float64) * model.chunk_steps
                                 * model.dt / 86400.0,
                                 **{f: np.stack(v) if v else np.zeros((0,), np.float32)
                                    for f, v in maps.items()}})
        row.update(action="tke_screen", repeat=r, growth=g, field_growth=fields, maps=path,
                   maps_bytes=os.path.getsize(path))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# action: svd (power iteration of the window map on pinned states)

def run_svd(model, win: Window, out_dir: Path, *, weights: str = "tke", box=C.DEFAULT_BOX,
            mask=None, iters: int = 20, tol: float = 1e-3, seed: int = 0, top: int = 20,
            label: str = "svd", save: bool = True, v0=None) -> dict:
    """Leading singular value / vectors of the window map in the ``weights`` norm, seeded in
    ``box`` (or an explicit ``mask``), on the pinned States of ``win`` (which may belong to
    another setting of the same model: the forward is the same). Logs σ history and the top
    cells of the input vector ``v`` (with the tke regimes of the window's first step) and of the
    output vector ``u`` (regimes of the step after the window); saves ``v``/``u`` (weighted
    leaves) to an npz. ``v0`` (a State vector, e.g. the leading ``v`` of another setting on the
    same window) replaces the random start vector."""
    st0 = model.state_of(win.pins[0])
    w = norm_weights(model, st0, weights)
    mask = box_seed_mask(model, box) if mask is None else mask
    mask = {k: v for k, v in mask.items() if k in w}
    v0 = random_state_vector(st0, w, seed) if v0 is None else v0
    hist = []
    status = "ok"
    with C._Timer() as t:
        try:
            res = adjoint_check.power_iteration(
                lambda v: _map_M(model, win, v), lambda u: _map_MT(model, win, u), v0,
                weights=w, mask=mask, iters=iters, tol=tol,
                on_iter=lambda i, s: hist.append((i, s)))
        except ValueError as exc:            # M v = 0 (e.g. the setting cuts the whole map)
            if not hist or hist[-1][1] != 0.0:
                raise
            status = f"zero map: {exc}"
            res = adjoint_check.PowerResult(sigma=0.0, v=v0, u=zeros_like_state(st0),
                                            history=[{"sigma": s, "rel_change": float("nan")}
                                                     for _, s in hist], converged=False)
    n = win.n_steps
    reg0, reg1 = win.regimes(0), win.regimes(1)
    row = C.provenance(model, "tke_svd", timer=t, n_steps=n, start=win.start, weights=weights,
                       status=status,
                       box=box if isinstance(box, str) else list(box), iters=iters, tol=tol,
                       seed=seed, sigma=res.sigma, sigma_per_step=res.sigma ** (1.0 / n),
                       converged=res.converged, history=res.history,
                       top_v=log_top_cells(model, res.v, reg0, w, top),
                       top_u=log_top_cells(model, res.u, reg1, w, top), label=label)
    if any(f in w for f in ("tke", "T", "S")):
        row.update(census_v=vector_census(res.v, reg0, w), census_u=vector_census(res.u, reg1, w))
    if save:
        p = C._save_npz(out_dir / f"{label}_vectors.npz",
                        {"v": _leaf_dict(res.v, w), "u": _leaf_dict(res.u, w)})
        row.update(vectors=p, vectors_bytes=os.path.getsize(p))
    row["_v"], row["_u"] = res.v, res.u
    return row


def _map_M(model, win: Window, v):
    if model is win.model:
        return win.M(v)
    return adjoint_check.window_jvp(model.chunk_fn_tl, win.params, win.carry0, v, None, win.nc,
                                    win.xs, pinned_states=win.pins,
                                    compiler_options=win.tl_options).tangent


def _map_MT(model, win: Window, w):
    if model is win.model:
        return win.MT(w)
    return adjoint_check.window_vjp(model.chunk_fn, win.params, win.carry0, w, win.nc, win.xs,
                                    pinned_states=win.pins, chunk_steps=model.chunk_steps,
                                    check_linear=False, field_trace=False).state_cot


# ---------------------------------------------------------------------------
# action: cuts (svd with single cuts, then pairs among the strongest)

def run_cuts(model, win: Window, out_dir: Path, singles, *, pairs: bool = True,
             max_pair_terms: int = 4, min_effect: float = 0.10, jpath=None,
             row_extra: dict | None = None, **svd_kw) -> list:
    """σ₁ of the window map with no cut, with each single setting of ``singles`` (cut names or
    freeze flags, see :func:`parse_setting`), then — if ``pairs`` — with every pair among the
    ≤ ``max_pair_terms`` singles whose σ₁ moved by more than ``min_effect`` (relative).
    All on the SAME pinned States (``win``, from the model's own forward). Rows carry
    ``setting``, ``sigma`` and ``ratio`` = σ / σ(no cut); with ``jpath`` each row is appended
    as soon as it exists (plus ``row_extra``), so a job that runs out of time keeps them."""
    base = model.adjoint_cfg
    rows = []

    def one(spec):
        m = model if spec == "none" else with_adjoint(model, parse_setting(spec, base))
        r = run_svd(m, win, out_dir, label=f"cut_{spec.replace('+', '__')}", **svd_kw)
        r.pop("_v"), r.pop("_u")
        r.update(row_extra or {})
        r.update(action="tke_cuts", setting=spec,
                 ratio=(r["sigma"] / rows[0]["sigma"]) if rows else 1.0)
        rows.append(r)
        if jpath is not None:
            C.append_row(jpath, r)
        return r["sigma"]

    s0 = one("none")
    eff = {}
    for spec in singles:
        eff[spec] = one(spec) / s0
    if pairs:
        live = sorted([s for s in eff if abs(eff[s] - 1.0) > min_effect],
                      key=lambda s: -abs(eff[s] - 1.0))[:int(max_pair_terms)]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                one(f"{live[i]}+{live[j]}")
    return rows


# ---------------------------------------------------------------------------
# action: column (method control: full-model one-step column block vs the column core)

def core_column_jacobians(inp: dict, nodes, cuts=frozenset()):
    """``[len(nodes), nl, nl]``: ∂tke_new/∂tke_old of the column core, per column, on the
    step's own inputs (``jax.jacfwd`` of the core restricted to those columns — the core is
    independent per column)."""
    nodes = np.asarray(nodes, int)
    args, kw = column_args(inp)
    sub = tuple(jnp.asarray(a)[nodes] for a in args)
    static = {k: kw.pop(k) for k in ("mxl_min", "tke_min", "kappaM_max", "dt")}

    def f(t):
        return cvmix_tke.integrate_tke_column(t, *sub[1:], **kw, **static, with_diags=False,
                                              cuts=frozenset(cuts))[0]

    J = np.asarray(jax.jacfwd(f)(sub[0]))                        # [nc, nl, nc, nl]
    return np.stack([J[i, :, i, :] for i in range(nodes.size)])


def full_column_blocks(win: Window, nodes, n_levels: int):
    """``[len(nodes), nl, n_levels]``: columns ``k < n_levels`` of the one-step tke → tke
    block of the FULL model's tangent map at the columns ``nodes`` — one tangent per level,
    seeded at every listed column at once (the one-step tke → tke_new map is block-diagonal
    by column: tke enters a step only through its column core)."""
    nodes = np.asarray(nodes, int)
    st0 = win.model.state_of(win.pins[0])
    nl = int(np.shape(st0.tke)[1])
    out = np.zeros((nodes.size, nl, int(n_levels)))
    for k in range(int(n_levels)):
        t = np.zeros(np.shape(st0.tke))
        t[nodes, k] = 1.0
        v = dataclasses.replace(zeros_like_state(st0), tke=t)
        mv = win.M(v, n_chunks=1)
        out[:, :, k] = np.asarray(mv.tke)[nodes, :]
    return out


def run_column(model, win: Window, nodes, *, power_iters: int = 0, tol: float = 1e-13) -> dict:
    """Method control on the one-step map (``model.chunk_steps`` must be 1): the full model's
    one-step tke block at each column in ``nodes`` (tangent-linear, pinned) against
    ``jax.jacfwd`` of the column core on :func:`step_tke_inputs`, max |difference| relative to
    max |J| per column, σ₁ of both; the diagonal and the ±1 neighbour terms of the block. With
    ``power_iters`` > 0, also the driver's power iteration restricted to the first column."""
    if model.chunk_steps != 1:
        raise ValueError("column: build the model with chunk_steps=1 (a one-step map)")
    nodes = sorted(set(int(n) for n in nodes))
    st0 = model.state_of(win.pins[0])
    inp = step_tke_inputs(model, st0, step_forcing(model, win.start, 0))
    nlev = np.asarray(inp["nlev"])
    with C._Timer() as t:
        Jc = core_column_jacobians(inp, nodes, cuts=model.adjoint_cfg.diag_cuts)
        K = int(nlev[nodes].max()) + 1
        Jf = full_column_blocks(win, nodes, K)
    lon, lat = C._lonlat_deg(model.mesh)
    cols = []
    for i, n in enumerate(nodes):
        k = int(nlev[n]) + 1
        a, b = Jf[i, :k, :k], Jc[i, :k, :k]
        scale = max(np.max(np.abs(b)), 1e-300)
        s_full = float(np.linalg.svd(a, compute_uv=False)[0])
        s_core = float(np.linalg.svd(b, compute_uv=False)[0])
        cols.append({"node": n, "lon": float(lon[n]), "lat": float(lat[n]), "nlev": int(nlev[n]),
                     "rel_max_diff": float(np.max(np.abs(a - b)) / scale),
                     "outside_block_max": float(np.max(np.abs(Jf[i, k:, :]))) if k < Jf.shape[1]
                     else 0.0,
                     "sigma_full": s_full, "sigma_core": s_core,
                     "sigma_rel": abs(s_full - s_core) / max(s_core, 1e-300),
                     "diag": np.diag(b).tolist(), "sub": np.diag(b, -1).tolist(),
                     "super": np.diag(b, 1).tolist()})
    row = C.provenance(model, "tke_column", timer=t, n_steps=win.n_steps, start=win.start,
                       nodes=nodes, columns=cols,
                       max_rel_diff=max(c["rel_max_diff"] for c in cols),
                       max_sigma_rel=max(c["sigma_rel"] for c in cols))
    if power_iters:
        w = {"tke": np.asarray(column_seed_mask(model, nodes[:1])["tke"], np.float64)
             * np.asarray(model.mesh.node_iface_mask)}
        res = adjoint_check.power_iteration(win.M, win.MT, random_state_vector(st0, w, 0),
                                            weights=w, iters=power_iters, tol=tol)
        row.update(power_sigma=res.sigma, power_converged=res.converged,
                   power_history=res.history,
                   power_rel=abs(res.sigma - cols[0]["sigma_core"]) / cols[0]["sigma_core"])
    return row


# ---------------------------------------------------------------------------
# action: fdlin (FD of the nonlinear window map vs the tangent-linear response)

def switch_distances(model, state, x, dtke, dbvfreq=None) -> dict:
    """Distances (in units of the tangent ``dtke``) from ``state``'s step to the tke-closure
    switches, per wet interior interface where ``dtke != 0``: the floor (``tke − tke_min``),
    the KappaM cap, and the mixing-length switch (wall-bound ⇄ free: the tke at which the
    unconstrained length ``mxl0 ∝ √tke`` meets the bound; for a free length the bound is the
    smaller of the neighbour-plus-thickness limits). With ``dbvfreq`` (the tangent of the
    State's ``bvfreq``), also the convective switch of ``mo_convect`` (``bvfreq < 0``):
    ``|bvfreq| / |dbvfreq|`` on wet interior interfaces — an estimate, since the step reads the
    ``bvfreq`` it computes from the entry State, one step later than the State's own.
    Returns the minimum and the cell of each, ``min_h`` over all of them and ``min_h_tke``
    over the tke-closure switches only."""
    inp = step_tke_inputs(model, state, x)
    reg = regimes(inp)
    tke = np.asarray(inp["tke_old"])
    dzw = np.asarray(inp["dzw"])
    tmin, kmax = float(inp["tke_min"]), float(inp["kappaM_max"])
    ck = float(np.asarray(inp["c_k"]))
    live = np.asarray(inp["is_interior"]) & (np.abs(np.asarray(dtke)) > 0)
    d = np.where(live, np.abs(np.asarray(dtke)), np.nan)
    mxl, mxl0 = reg["mxl"], reg["mxl0"]
    out = {}
    # floor
    dist = np.where(live & (tke > tmin), (tke - tmin), np.nan)
    out["floor"] = dist / d
    # KappaM cap: c_k·mxl·√tke = kappaM_max ⇒ tke* = (kmax/(c_k·mxl))²
    with np.errstate(divide="ignore", invalid="ignore"):
        tcap = (kmax / (ck * mxl)) ** 2
        out["cap"] = np.where(live & (mxl > 0), np.abs(tcap - tke), np.nan) / d
        # mixing length: wall-bound (mxl < mxl0): switch when mxl0 falls to mxl ⇒ tke·(mxl/mxl0)²
        wall = live & (mxl < mxl0) & (mxl0 > 0)
        dist_w = np.where(wall, tke * np.abs(1.0 - (mxl / mxl0) ** 2), np.nan)
        # free (mxl == mxl0): switch when mxl0 rises to the neighbour bound
        up = np.concatenate([np.full_like(mxl[:, :1], np.inf), mxl[:, :-1] + dzw[:, :-1]], axis=1)
        dn = np.concatenate([mxl[:, 1:] + dzw[:, :-1], np.full_like(mxl[:, :1], np.inf)], axis=1)
        bound = np.minimum(up, dn)
        free = live & (reg["regime"] == 0) & (mxl0 > 0)
        dist_f = np.where(free, tke * np.abs((bound / mxl0) ** 2 - 1.0), np.nan)
        out["mixing_length"] = np.fmin(dist_w, dist_f) / d
        if dbvfreq is not None:
            db = np.abs(np.asarray(dbvfreq))
            lc = np.asarray(inp["is_interior"]) & (db > 0)
            out["convective"] = np.where(lc, np.abs(np.asarray(state.bvfreq)), np.nan) / np.where(
                lc, db, np.nan)
    res = {}
    for k, a in out.items():
        if np.all(np.isnan(a)):
            res[k] = {"h": math.inf}
            continue
        j = int(np.nanargmin(a))
        n, lev = np.unravel_index(j, a.shape)
        res[k] = {"h": float(a.ravel()[j]), "node": int(n), "level": int(lev)}
    res["min_h_tke"] = min(res[k]["h"] for k in ("floor", "cap", "mixing_length"))
    res["min_h"] = min(v["h"] for k, v in res.items() if isinstance(v, dict))
    return res


def predicted_range(model, win: Window, v) -> dict:
    """The largest ``h`` for which the tangent-linear map along ``v`` should hold: the minimum
    over the chunk-entry States of the window of :func:`switch_distances` for the tangent
    carried to that State (pinned). Evaluated at every chunk entry (not at every step)."""
    t = v
    per = []
    for c in range(win.nc):
        st = model.state_of(win.pins[c]) if c else model.state_of(win.pins[0])
        tt = model.state_of(t)
        d = switch_distances(model, st, step_forcing(model, win.start, c * model.chunk_steps),
                             tt.tke, dbvfreq=tt.bvfreq)
        per.append({"chunk": c, **d})
        if c + 1 < win.nc:
            t = win.M(t, n_chunks=1, start_chunk=c)
    h = min(p["min_h"] for p in per)
    return {"h": h, "h_tke": min(p["min_h_tke"] for p in per), "per_chunk": per}


def run_fdlin(model, win: Window, v, *, weights="tke", hs=None, noise="auto",
              label: str = "fdlin") -> list:
    """FD of the nonlinear window map along ``v`` against the tangent-linear response, for each
    ``h`` (default 1e-1 … 1e-12), in the output norm ``weights`` (``"tke"``, ``"prog"``, or a
    weights dict as in :func:`norm_weights`); the predicted linear range from
    :func:`predicted_range`; and a forward noise floor, rows it explains marked
    "noise-limited":

    * ``"repeat"`` (the default on a GPU): the difference of two runs of the unperturbed forward
      (non-deterministic scatter-adds);
    * ``"roughness"`` (the default on a CPU, where repeats are bitwise): the part of the central
      difference at ``h₀`` that the tangent does not explain,
      ``‖(J(x+h₀v) − J(x−h₀v))/2 − h₀·TL‖``, the larger of ``h₀ = min(hs)/100`` and
      ``min(hs)/10`` — the forward is deterministic but not smooth at that scale (the SSH solve
      stops at a relative residual, so its iteration count, and the solution, jump; clips and
      switches), and one sample of a jump can land low (2026-10-28: 30× below the base branch's);
    * ``None``: no floor.
    """
    hs = [10.0 ** -e for e in range(1, 13)] if hs is None else [float(h) for h in hs]
    st0 = model.state_of(win.pins[0])
    w = norm_weights(model, st0, weights) if isinstance(weights, str) else dict(weights)
    if noise == "auto":
        noise = "repeat" if jax.devices()[0].platform != "cpu" else "roughness"

    def wnorm(a, b, scale=1.0, minus=None, h=0.0):
        d = jax.tree.map(lambda x, y: (np.asarray(x) - np.asarray(y)) * scale
                         if adjoint_check._is_real(x) else x, a, b)
        if minus is not None:
            d = jax.tree.map(lambda x, y: x - h * np.asarray(y)
                             if adjoint_check._is_real(x) else x, d, minus)
        return math.sqrt(max(adjoint_check._wnorm2(d, w), 0.0))

    with C._Timer() as t:
        tl = win.M(v)
        pr = predicted_range(model, win, v)
        floor = None
        if noise == "repeat":
            floor = wnorm(win.forward(win.pins[0]), win.forward(win.pins[0]))
        elif noise == "roughness":
            def rough(h0):
                shift = (lambda s: jax.tree.map(lambda a, b: a + s * h0 * b
                                                if adjoint_check._is_real(a) else a,
                                                win.pins[0], v))
                return wnorm(win.forward(shift(1.0)), win.forward(shift(-1.0)), 0.5, tl, h0)
            floor = max(rough(min(hs) / 100.0), rough(min(hs) / 10.0))
        elif noise is not None:
            raise ValueError(f"noise must be 'auto', 'repeat', 'roughness' or None: {noise!r}")
        rows = adjoint_check.fdlin(win.forward, win.pins[0], v, hs, tl, weights=w,
                                   predicted_range=pr["h"], noise_floor=floor)
    base = C.provenance(model, "tke_fdlin", timer=t, n_steps=win.n_steps, start=win.start,
                        weights=weights if isinstance(weights, str) else sorted(weights),
                        label=label, predicted_h=pr["h"], predicted_h_tke=pr["h_tke"],
                        predicted=pr["per_chunk"],
                        noise=noise, noise_floor=floor,
                        tl_norm=math.sqrt(max(adjoint_check._wnorm2(tl, w), 0.0)))
    return [dict(base, h=r.h, rel_err=r.rel_err, per_field=r.per_field, fd_norm=r.fd_norm,
                 noise_rel=r.noise_rel, regime=r.regime) for r in rows]


# ---------------------------------------------------------------------------
# action: sites (power iteration and cuts at several windows / seed boxes, one set of programs)

def advance(model, state, t_from, t_to):
    """The State at ``t_to`` by a chunked forward from ``state`` at ``t_from`` (host)."""
    t0, t1 = C._parse_date(t_from), C._parse_date(t_to)
    n = int(round((t1 - t0).total_seconds() / model.dt))
    if n == 0:
        return state
    nc = model.chunks_for(n)
    fin = adjoint.chunked_forward(model.chunk_fn, model.params0,
                                  init_state=model.init_carry(state), n_chunks=nc,
                                  xs_fn=model.xs_fn(t0))
    out = model.state_of(jax.device_get(fin))
    adjoint._free(fin)
    return out


def _spec_box(b):
    return b if isinstance(b, str) else C.Box(*[float(x) for x in b])


def run_sites(model, spec: dict, out_dir: Path, jpath: Path, row_extra: dict) -> list:
    """Power iteration (and optionally cuts) for a list of windows, in time order, from the
    model's start State. ``spec``::

        {"windows": [{"name": str, "at": ISO time, "steps": int, "box": name or
                      [lat_min, lat_max, lon_min, lon_max, z_min, z_max],
                      "weights": ["tke", "prog"], "cuts": bool, "save_state": bool}, ...],
         "singles": [settings for the cut windows], "pairs": bool, "max_pair_terms": 4,
         "iters": 20, "tol": 1e-3, "seed": 0, "top": 20, "baseline_repeats": 1}

    One model per setting (built once, so its programs compile once and serve every window);
    the pinned States of each window come from the exact model's forward and are shared by
    every setting. With ``cuts``, the window gets σ₁ under ``none`` (``baseline_repeats``
    times: the repeat spread), each single, then the pairs among the ≤ ``max_pair_terms``
    singles that moved σ₁ by > 10 % (tke norm). Rows are appended to ``jpath`` as they finish."""
    wins = sorted(spec["windows"], key=lambda w: C._parse_date(w["at"]))
    kw = dict(iters=int(spec.get("iters", 20)), tol=float(spec.get("tol", 1e-3)),
              seed=int(spec.get("seed", 0)), top=int(spec.get("top", 20)))
    variants = {"none": model}

    def variant(sp):
        if sp not in variants:
            variants[sp] = with_adjoint(model, parse_setting(sp, model.adjoint_cfg))
        return variants[sp]

    rows = []
    state, t_state = model.s0, model.start
    for w in wins:
        t_at = C._parse_date(w["at"])
        with C._Timer() as tf:
            state = advance(model, state, t_state, t_at)
        t_state = t_at
        extra = dict(row_extra, site=w["name"], box=w["box"], advance_s=tf.wall)
        nb, _ = C.box_masks(model.mesh, _spec_box(w["box"]))
        mld = np.asarray(C.AC.mld_field(jax.device_put(state), model.mesh, invalid="column"))[nb]
        extra["mld_box"] = {"mean": float(mld.mean()), "median": float(np.median(mld)),
                            "max": float(mld.max()), "n_nodes": int(nb.sum())}
        if w.get("save_state"):
            d = out_dir / f"state_{C.restart_tag(t_at)}"
            if not d.exists():
                C._write_state(d, state, model.mesh, step=0, when=t_at,
                               attrs={"from": str(model.restart)})
            extra.update(saved_state=str(d), saved_state_bytes=_du_bytes(d))
        steps = int(w["steps"])
        win = Window(model, start_state=state, start=t_at, n_chunks=model.chunks_for(steps))
        extra["forward_s"] = win.forward_timer.wall
        box = _spec_box(w["box"])
        tag = f"{w['name']}_{steps}"

        def one(sp, weights, rep=0):
            m = variant(sp)
            label = f"{tag}_{weights}_{sp.replace('+', '__')}" + (f"_r{rep}" if rep else "")
            r = run_svd(m, win, out_dir, weights=weights, box=box, label=label,
                        save=(sp == "none"), **kw)
            r.pop("_v"), r.pop("_u")
            r = dict(extra, **r)
            r.update(action="tke_sites", setting=sp, weights=weights, repeat=rep,
                     box=w["box"] if isinstance(w["box"], str) else list(w["box"]))
            C.append_row(jpath, r)
            rows.append(r)
            return r["sigma"]

        for weights in w.get("weights", ["tke"]):
            s0 = one("none", weights)
            if not (w.get("cuts") and weights == "tke"):
                continue
            for rep in range(1, int(spec.get("baseline_repeats", 1))):
                one("none", weights, rep)
            eff = {sp: one(sp, weights) / s0 if s0 else float("nan")
                   for sp in spec.get("singles", [])}
            if spec.get("pairs", True):
                live = sorted([sp for sp, e in eff.items() if abs(e - 1.0) > 0.10],
                              key=lambda sp: -abs(math.log(max(eff[sp], 1e-300))))
                live = live[:int(spec.get("max_pair_terms", 4))]
                for i in range(len(live)):
                    for j in range(i + 1, len(live)):
                        one(f"{live[i]}+{live[j]}", weights)
        del win
    return rows


# ---------------------------------------------------------------------------
# action: locate (scan for growth sites, power iteration at the sites, cuts at the top site)

GLOBAL_BOX = C.Box(-90.0, 90.0, -180.0, 180.0, 0.0, 7000.0)


def _tag_time(t) -> str:
    return C._parse_date(t).strftime("%Y%m%dT%H%M")


def angular_distance(lon1, lat1, lon2, lat2) -> float:
    """Great-circle distance in degrees."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def site_box(lon: float, lat: float, half_lat: float = 2.0, half_lon: float = 2.0,
             z=(0.0, 1000.0)) -> list:
    """``[lat_min, lat_max, lon_min, lon_max, z_min, z_max]`` around a point; the longitude
    half-width is ``half_lon / cos(lat)`` (capped at 30°), so the box is roughly square."""
    hl = min(half_lon / max(math.cos(math.radians(lat)), 1e-3), 30.0)

    def wrap(x):
        return ((x + 180.0) % 360.0) - 180.0
    return [max(-90.0, lat - half_lat), min(90.0, lat + half_lat), wrap(lon - hl),
            wrap(lon + hl), float(z[0]), float(z[1])]


def realisation_diff(model, a, b, box=None) -> dict:
    """How two States of the same time from two runs of the same program differ: counts of
    wet interfaces where the convective switch (``bvfreq < 0``, what ``mo_convect`` reads), the
    tke floor contact (``tke ≤ tke_min``) or ``Kv ≥ 0.1`` (the convective value) differ, sea-ice
    skin temperatures differing by more than 0.1 K, and the largest differences of T, tke and
    eta_n — globally and inside ``box``."""
    mesh = model.mesh
    wet = np.asarray(mesh.node_iface_mask)
    lay = np.asarray(mesh.node_layer_mask)
    tmin = float(model.phys["tke_cfg"].tke_min)
    out = {}
    regions = {"global": np.ones(wet.shape[0], bool)}
    if box is not None:
        regions["box"] = C.box_masks(mesh, _spec_box(box))[0]
    for name, nb in regions.items():
        wi, li = wet & nb[:, None], lay & nb[:, None]

        def cnt(x, y, m):
            return int(np.count_nonzero((np.asarray(x) != np.asarray(y)) & m))
        d = {"conv_flips": cnt(np.asarray(a.bvfreq) < 0, np.asarray(b.bvfreq) < 0, wi),
             "tke_floor_flips": cnt(np.asarray(a.tke) <= tmin, np.asarray(b.tke) <= tmin, wi),
             "kv_convective_flips": cnt(np.asarray(a.Kv) >= 0.1, np.asarray(b.Kv) >= 0.1, wi)}
        dts = np.abs(np.asarray(a.t_skin) - np.asarray(b.t_skin))[nb]
        d["t_skin_gt_0.1K"] = int(np.count_nonzero(dts > 0.1))
        d["t_skin_max"] = float(dts.max()) if dts.size else 0.0
        d["T_max"] = float(np.max(np.abs(np.asarray(a.T) - np.asarray(b.T))[li], initial=0.0))
        d["tke_max"] = float(np.max(np.abs(np.asarray(a.tke) - np.asarray(b.tke))[wi],
                                    initial=0.0))
        d["eta_max"] = float(np.max(np.abs(np.asarray(a.eta_n) - np.asarray(b.eta_n))[nb],
                                    initial=0.0))
        out[name] = d
    return out


def run_locate(model, spec: dict, out_dir: Path, jpath: Path, row_extra: dict) -> dict:
    """Where the TKE adjoint grows, and which terms carry it, in one job (one set of compiled
    programs; ``model.chunk_steps`` = 1 recommended: the tangent and adjoint maps then linearise
    about the same pinned State at every step). ``spec``::

        {"scan_times": [ISO, ...], "scan_steps": 12, "scan_iters": 12,
         "n_sites": 2, "min_sep_deg": 15, "half_lat": 2, "half_lon": 2, "site_z": [0, 1000],
         "quiet": {"name": "quiet", "box": [lat0, lat1, lon0, lon1, z0, z1]},
         "site_windows": [1, 12, 48], "site_weights": ["tke", "prog"], "site_iters": 10,
         "save_states": true,
         "sites": [{"name": ..., "at": ISO, "lon": ..., "lat": ... (or "box"),
                    "windows": {"1": ["tke", "prog"], "48": ["tke"]}}, ...],   # optional
         "cuts": {"windows": [1, 12], "singles": [...], "pairs": true, "max_pair_terms": 4,
                  "min_effect": 0.10, "baseline_repeats": 2, "new_pins": true, "iters": 10,
                  "tol": 5e-3},
         "tol": 3e-3, "top": 30}

    ``scan_weights`` and ``cuts.weights`` choose the norm of the scan and of the cuts
    (``"tke"``, the default, or ``"prog"``).

    1. **scan** — power iteration of the ``scan_steps`` window map in the tke norm, seeded
       everywhere (:data:`GLOBAL_BOX`), at each scan time (States advanced from the model start
       in time order and kept on the host);
    2. **sites** — the ``n_sites`` scan windows with the largest σ whose leading input vectors
       peak ≥ ``min_sep_deg`` apart; a box around each peak (:func:`site_box`) and the
       ``quiet`` box at the top site's time: power iteration for each window length and norm
       (vectors saved); the site States saved (``save_states``) and ``sites.json`` written;
    3. **cuts** — at the top site: σ₁ with no cut (``baseline_repeats`` on the same pins, and
       with ``new_pins`` once more on a second forward from the same State, with the
       :func:`realisation_diff` of the two forwards per step), each single, then the pairs among
       the ≤ ``max_pair_terms`` singles that moved σ₁ of the longest cut window by more than
       ``min_effect``; settings in the outer loop (one compile each), each power iteration
       started from the no-cut leading vector of its window.

    With ``sites`` in the spec the scan is skipped and those sites are used as given (their
    States advanced from the model start). ``site_windows`` (and a site's own ``windows``) may
    be a list of window lengths (each with every ``site_weights`` norm) or a mapping
    ``{length: [norms]}``.

    Every row is appended to ``jpath`` as it is made. Returns the sites record."""
    import gc
    tol = float(spec.get("tol", 3e-3))
    top = int(spec.get("top", 30))
    times = sorted(C._parse_date(t) for t in spec.get("scan_times", []))
    if times and times[0] < model.start:
        raise ValueError("locate: scan times before the model start")
    sstep = int(spec.get("scan_steps", 12))
    scan_w = spec.get("scan_weights", "tke")
    cut_w = (spec.get("cuts") or {}).get("weights", "tke")

    def emit(r, **kw):
        r = {k: v for k, v in r.items() if not str(k).startswith("_")}
        r = dict(row_extra, **r)
        r.update(kw)
        C.append_row(jpath, r)
        return r

    def mld_of(state, box):
        nb, _ = C.box_masks(model.mesh, _spec_box(box))
        mld = np.asarray(C.AC.mld_field(jax.device_put(state), model.mesh, invalid="column"))[nb]
        return {"mean": float(mld.mean()), "median": float(np.median(mld)),
                "max": float(mld.max()), "n_nodes": int(nb.sum())}

    def plan(windows):
        if isinstance(windows, dict):
            return [(int(n), list(w)) for n, w in windows.items()]
        return [(int(n), list(spec.get("site_weights", ["tke", "prog"]))) for n in windows]

    # 1. scan (or the given sites)
    states, scan = {}, []
    state, t_state = model.s0, model.start
    for t in ([] if spec.get("sites") else times):
        with C._Timer() as tf:
            state = advance(model, state, t_state, t)
        states[t] = state
        win = Window(model, start_state=state, start=t, n_chunks=model.chunks_for(sstep))
        r = run_svd(model, win, out_dir, weights=scan_w, box=GLOBAL_BOX,
                    iters=int(spec.get("scan_iters", 12)), tol=tol, top=top,
                    label=f"scan_{_tag_time(t)}", save=False)
        r = emit(r, action="tke_scan", site="global", box="global", setting="none",
                 advance_s=tf.wall, forward_s=win.forward_timer.wall)
        scan.append(r)
        state = model.state_of(win.pins[win.nc])
        t_state = t + datetime.timedelta(seconds=sstep * model.dt)
        del win

    # 2. sites
    sites = []
    for i, x in enumerate(spec.get("sites") or []):
        x = dict(x)
        x.setdefault("name", f"site{i + 1}")
        x["at"] = C._parse_date(x["at"]).isoformat()
        if "box" not in x:
            x["box"] = site_box(x["lon"], x["lat"], float(spec.get("half_lat", 2.0)),
                                float(spec.get("half_lon", 2.0)), spec.get("site_z", (0.0, 1000.0)))
        sites.append(x)
    for t in sorted({C._parse_date(x["at"]) for x in sites}):
        if t < t_state:
            raise ValueError("locate: site times before the model start")
        state = advance(model, state, t_state, t)
        states[t], t_state = state, t
    for r in sorted(scan, key=lambda r: -r["sigma"]):
        # the peak of the leading input vector: its largest weighted cell with a position
        cells = sorted([c for c in r["top_v"] if "lon" in c], key=lambda c: -abs(c["value"]))
        if not cells:
            continue
        c = cells[0]
        if all(angular_distance(c["lon"], c["lat"], s["lon"], s["lat"])
               >= float(spec.get("min_sep_deg", 15.0)) for s in sites):
            sites.append({"name": f"site{len(sites) + 1}", "at": r["start"], "lon": c["lon"],
                          "lat": c["lat"], "node": c["node"], "level": c["level"],
                          "field": c.get("field"), "scan_weights": scan_w,
                          "regime": c.get("regime"), "tke": c.get("tke"), "F": c.get("F"),
                          "sigma_scan": r["sigma"], "scan_steps": sstep,
                          "box": site_box(c["lon"], c["lat"], float(spec.get("half_lat", 2.0)),
                                          float(spec.get("half_lon", 2.0)),
                                          spec.get("site_z", (0.0, 1000.0)))})
        if len(sites) == int(spec.get("n_sites", 2)):
            break
    all_sites = list(sites)
    if spec.get("quiet") and sites:
        q = dict(spec["quiet"])
        all_sites.append({"name": q.get("name", "quiet"), "at": sites[0]["at"], "box": q["box"],
                          "quiet": True, **({"windows": q["windows"]} if "windows" in q else {})})
    for t in [t for t in states if t not in {C._parse_date(x["at"]) for x in all_sites}]:
        del states[t]                          # host memory: keep the site times only
    cs = spec.get("cuts") or {}
    cut_windows = [int(x) for x in cs.get("windows", [1, 12])]
    kept = {}
    # the top site last: its kept cut windows then never coexist with another long window
    for s in all_sites[::-1]:
        t = C._parse_date(s["at"])
        st = states[t]
        s["mld_box"] = mld_of(st, s["box"])
        if spec.get("save_states", True) and not s.get("quiet"):
            d = out_dir / f"state_{C.restart_tag(t)}"
            if not d.exists():
                C._write_state(d, st, model.mesh, step=0, when=t,
                               attrs={"from": str(model.restart), "note": "tke_diag locate"})
            s["state"], s["state_bytes"] = str(d), _du_bytes(d)
        s["sigma"], s["vectors"] = {}, {}
        top_site = s is sites[0]
        for n, wlist in plan(s.get("windows") or spec.get("site_windows", [1, 12, 48])):
            win = Window(model, start_state=st, start=t, n_chunks=model.chunks_for(n))
            for wts in wlist:
                r = run_svd(model, win, out_dir, weights=wts, box=_spec_box(s["box"]),
                            iters=int(spec.get("site_iters", 10)), tol=tol, top=top,
                            label=f"{s['name']}_{n}_{wts}", save=True)
                v = r["_v"]
                r = emit(r, action="tke_site", site=s["name"], box=s["box"], setting="none",
                         mld_box=s["mld_box"], quiet=bool(s.get("quiet")),
                         forward_s=win.forward_timer.wall)
                s["sigma"][f"{n}_{wts}"] = r["sigma"]
                s["vectors"][f"{n}_{wts}"] = r.get("vectors")
                if top_site and wts == cut_w and n in cut_windows:
                    kept[n] = (win, v, r["sigma"])
            del win
        (out_dir / "sites.json").write_text(json.dumps(C._jsonable(
            {"scan": [{"at": r["start"], "sigma": r["sigma"], "top_v": r["top_v"][:3]}
                      for r in scan], "sites": all_sites}), indent=1))
    rec = {"scan": scan, "sites": all_sites}
    if not cs or not sites:
        return rec
    missing = [n for n in cut_windows if n not in kept]
    if missing:
        raise ValueError(f"locate: cut windows {missing} are not site windows (with the "
                         f"{cut_w} norm)")

    # 3. cuts at the top site (settings outer, windows inner)
    s = sites[0]
    t = C._parse_date(s["at"])
    box = _spec_box(s["box"])
    ckw = dict(iters=int(cs.get("iters", 10)), tol=float(cs.get("tol", 5e-3)), top=top)
    base_sigma = {n: kept[n][2] for n in kept}

    def one(sp, m, n, win, v0=None, rep=0, **kw):
        label = f"cut_{s['name']}_{n}_{sp.replace('+', '__')}" + (f"_r{rep}" if rep else "")
        r = run_svd(m, win, out_dir, weights=cut_w, box=box, label=label, save=False, v0=v0,
                    **ckw)
        r = emit(r, action="tke_cuts", site=s["name"], box=s["box"], setting=sp, repeat=rep,
                 mld_box=s["mld_box"], warm_start=v0 is not None,
                 ratio=r["sigma"] / base_sigma[n] if base_sigma[n] else None, **kw)
        return r["sigma"]

    for n in sorted(kept):
        win, v, _ = kept[n]
        for rep in range(1, int(cs.get("baseline_repeats", 2))):
            one("none", model, n, win, rep=rep)
        if cs.get("new_pins", True) and n > 1:
            win2 = Window(model, start_state=states[t], start=t, n_chunks=win.nc)
            diffs = [realisation_diff(model, model.state_of(win.pins[c]),
                                      model.state_of(win2.pins[c]), s["box"])
                     for c in range(1, win.nc + 1)]
            one("none", model, n, win2, rep="pins2", realisation_diff=diffs)
            del win2
    eff = {}
    nmax = max(kept)

    def run_setting(sp):
        m = with_adjoint(model, parse_setting(sp, model.adjoint_cfg))
        out = {}
        for n in sorted(kept):
            out[n] = one(sp, m, n, kept[n][0], v0=kept[n][1])
        del m
        gc.collect()
        jax.clear_caches()
        return out

    for sp in cs.get("singles", []):
        sig = run_setting(sp)
        eff[sp] = sig[nmax] / base_sigma[nmax] if base_sigma[nmax] else float("nan")
    if cs.get("pairs", True):
        live = sorted([sp for sp, e in eff.items()
                       if math.isfinite(e) and abs(e - 1.0) > float(cs.get("min_effect", 0.10))],
                      key=lambda sp: -abs(math.log(max(eff[sp], 1e-300))))
        live = live[:int(cs.get("max_pair_terms", 4))]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                run_setting(f"{live[i]}+{live[j]}")
    return rec


# ---------------------------------------------------------------------------
# action: colcensus (the TKE-only recursion of every column, with and without cuts)

def column_jacobians_all(inp: dict, cuts=frozenset(), adj=None) -> np.ndarray:
    """``[N, nl, nl]``: ``∂tke_new/∂tke_old`` of the column core in every column, on the step's
    own inputs, everything else (S², N², the geometry, the surface flux, the constants) held
    fixed — the TKE-only recursion. The core is independent per column, so one tangent with a
    unit entry at level ``k`` of EVERY column gives column ``k`` of all the blocks at once
    (``nl`` tangents in all)."""
    args, kw = column_args(inp)
    static = {k: kw.pop(k) for k in ("mxl_min", "tke_min", "kappaM_max", "dt")}
    rest = args[1:]

    @jax.jit
    def tangent(t0, e):
        return jax.jvp(lambda t: cvmix_tke.integrate_tke_column(
            t, *rest, **kw, **static, with_diags=False, cuts=frozenset(cuts),
            adj=adj)[0], (t0,), (e,))[1]

    t0 = jnp.asarray(args[0])
    n, nl = t0.shape
    out = np.zeros((n, nl, nl))
    for k in range(nl):
        e = jnp.zeros_like(t0).at[:, k].set(1.0)
        out[:, :, k] = np.asarray(tangent(t0, e))
    return out


def recursion_census(J: np.ndarray, nlev, lon, lat, reg: dict, top: int = 20) -> dict:
    """Spectral radius ρ and σ₁ of each wet column's TKE recursion ``J`` (from
    :func:`column_jacobians_all`), the share of wet columns with ρ above 1, 1.1, 1.5 and 2,
    the same by latitude band, and the ``top`` columns by ρ with the regime of their first
    interior interface."""
    nlev = np.asarray(nlev).astype(int)
    wet = nlev > 0
    idx = np.nonzero(wet)[0]
    rho = np.zeros(J.shape[0])
    sig = np.zeros(J.shape[0])
    for m in sorted(set(nlev[idx].tolist())):
        cols = idx[nlev[idx] == m]
        blk = J[cols][:, :m + 1, :m + 1]
        rho[cols] = np.max(np.abs(np.linalg.eigvals(blk)), axis=-1)
        sig[cols] = np.linalg.norm(blk, ord=2, axis=(1, 2))
    r = rho[idx]
    out = {"n_wet": int(idx.size), "rho_max": float(r.max()), "sigma_max": float(sig[idx].max()),
           "share_rho_gt": {str(x): float(np.mean(r > x)) for x in (1.0, 1.1, 1.5, 2.0)},
           "n_rho_gt_1": int(np.sum(r > 1.0))}
    bands = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
    la = np.asarray(lat)[idx]
    out["n_rho_gt_1_by_lat"] = {f"{a}..{b}": int(np.sum((r > 1.0) & (la >= a) & (la < b)))
                                for a, b in bands}
    order = idx[np.argsort(-rho[idx])][:int(top)]
    cells = []
    for c in order:
        k1 = 1
        rg = int(np.asarray(reg["regime"])[c, k1])
        cells.append({"node": int(c), "lon": float(lon[c]), "lat": float(lat[c]),
                      "rho": float(rho[c]), "sigma": float(sig[c]), "nlev": int(nlev[c]),
                      "regime_k1": cvmix_tke.REGIMES[rg] if rg >= 0 else "dry",
                      "tke_k0": float(np.asarray(reg["tke"])[c, 0]),
                      "tke_k1": float(np.asarray(reg["tke"])[c, k1]),
                      "N2_k1": float(np.asarray(reg["Nsqr"])[c, k1]),
                      "Ri_k1": float(np.asarray(reg["Ri"])[c, k1]),
                      "F_k1": float(np.asarray(reg["F"])[c, k1]),
                      "J00": float(J[c, 0, 0]), "J11": float(J[c, 1, 1]),
                      "J01": float(J[c, 0, 1]), "J10": float(J[c, 1, 0])})
    out["top"] = cells
    # where the unstable columns sit: regime / floor contact at the first interior interface
    unst = idx[rho[idx] > 1.0]
    if unst.size:
        rk = np.asarray(reg["regime"])[unst, 1]
        out["unstable_k1_regime"] = {name: int(np.sum(rk == i))
                                     for i, name in enumerate(cvmix_tke.REGIMES)}
        ex = np.asarray(reg["tke_excess"])[unst, 1]
        out["unstable_k1_tke_excess_lt_1e-5"] = float(np.mean(ex < 1e-5))
        out["unstable_tke_k0_median"] = float(np.median(np.asarray(reg["tke"])[unst, 0]))
    return out, rho


def two_step_rho(J0: np.ndarray, J1: np.ndarray, nlev) -> np.ndarray:
    """Per-step spectral radius of the TWO-step TKE-only recursion ``J1·J0`` of every wet column
    (``ρ(J1·J0)^(1/2)``; 0 on dry columns)."""
    nlev = np.asarray(nlev).astype(int)
    idx = np.nonzero(nlev > 0)[0]
    out = np.zeros(J0.shape[0])
    for m in sorted(set(nlev[idx].tolist())):
        cols = idx[nlev[idx] == m]
        P = np.matmul(J1[cols][:, :m + 1, :m + 1], J0[cols][:, :m + 1, :m + 1])
        out[cols] = np.sqrt(np.max(np.abs(np.linalg.eigvals(P)), axis=-1))
    return out


def _setting_cuts(sp: str):
    """``(cuts, TkeAdjoint | None)`` of a colcensus setting: ``none``, ``tke.*`` cut names
    and/or candidate fields (``tke_kme_tke=3e-6+tke_kme_scope=iface1``), '+'-joined."""
    if sp == "none":
        return frozenset(), None
    cfg = parse_setting(sp)
    return cfg.diag_cuts, cfg.tke_options()


def run_colcensus(model, state, when, singles, out_dir: Path, label: str = "colcensus",
                  top: int = 20, two_step: bool = False) -> list:
    """The TKE-only recursion of every column at ``when`` (``state`` is the State there):
    census of its spectral radius without cuts and with each setting of ``singles`` (``tke.*``
    cut names or candidate settings, see :func:`_setting_cuts`), plus the share of the no-cut
    unstable columns each setting stabilises and, for a near-floor km_e setting, the number of
    wet columns it flags. With ``two_step`` also the two-step recursion ``J(t1)·J(t0)`` (the
    next step's inputs from the model's own step): per-step ρ of the product and the shares on
    it. Saves the no-cut ρ map (npz)."""
    inp = step_tke_inputs(model, state, step_forcing(model, when, 0))
    inp1 = (step_tke_inputs(model, inp["new_state"], step_forcing(model, when, 1))
            if two_step else None)
    reg = regimes(inp)
    lon, lat = C._lonlat_deg(model.mesh)
    nlev = np.asarray(inp["nlev"]).astype(int)
    wet = nlev > 0
    tke1 = np.asarray(inp["tke_old"])[:, 1]
    rows = []
    base_rho = base_rho2 = None
    for sp in ["none"] + [s for s in singles if s != "none"]:
        cuts, adj = _setting_cuts(sp)
        with C._Timer() as t:
            J = column_jacobians_all(inp, cuts, adj)
            cen, rho = recursion_census(J, inp["nlev"], lon, lat, reg, top=top)
            if two_step:
                rho2 = two_step_rho(J, column_jacobians_all(inp1, cuts, adj), inp["nlev"])
                r2 = rho2[wet]
                cen["two_step"] = {"rho_max": float(r2.max()), "n_rho_gt_1": int(np.sum(r2 > 1.0)),
                                   "share_rho_gt": {str(x): float(np.mean(r2 > x))
                                                    for x in (1.0, 1.1, 1.5)}}
        if adj is not None and adj.kme_tke is not None:
            flag = wet & (tke1 < adj.kme_tke)
            cen["n_flagged"] = int(flag.sum())
            cen["share_flagged"] = float(flag.sum() / max(1, wet.sum()))
        if base_rho is None:
            base_rho = rho
            p = C._save_npz(out_dir / f"{label}_rho.npz", {"rho": rho.astype(np.float32),
                                                         "lon": lon, "lat": lat})
            cen["rho_map"] = p
            if two_step:
                base_rho2 = rho2
            unst = wet & (rho > 1.0)
            if unst.any():
                cen["unstable_tke_k1_quantiles"] = {
                    str(q): float(np.quantile(tke1[unst], q))
                    for q in (0.5, 0.9, 0.95, 0.99, 1.0)}
        else:
            unst = wet & (base_rho > 1.0)
            cen["stabilised_share"] = (float(np.mean(rho[unst] <= 1.0)) if unst.any()
                                       else None)
            cen["new_unstable"] = int(np.sum(wet & (base_rho <= 1.0) & (rho > 1.0)))
            if two_step:
                u2 = wet & (base_rho2 > 1.0)
                cen["two_step"]["stabilised_share"] = (float(np.mean(rho2[u2] <= 1.0))
                                                       if u2.any() else None)
        del J
        rows.append(C.provenance(model, "tke_colcensus", timer=t, n_steps=1, start=when,
                                 setting=sp, label=label, **cen))
    return rows


PR_BAND = (1.0, 10.0)


def pr_census(inp: dict, n2_stars, cuts=frozenset()) -> dict:
    """Where the Prandtl number's derivative lives, on one step's column inputs: the wet
    interior interfaces with ``1 < 6.6·Ri < 10`` (Pr unclipped, so dPr/dN² = 6.6/S² ≠ 0) and
    their KappaH sensitivity ``G = |∂KappaH/∂N²|_Pr = KappaM·6.6/(Pr²·max(S², 1e-12))``. For
    each ``N²*``: the share of these cells and of their summed G below ``N²*`` (what the cut
    "only where N² < N²*" removes), and the largest G left above it; plus G-weighted N²
    quantiles. Host arrays."""
    _, KappaM, _ = column_core(inp, cuts)
    n2, s2 = np.asarray(inp["Nsqr"]), np.asarray(inp["Ssqr"])
    km = np.asarray(KappaM)
    interior = np.asarray(inp["is_interior"])
    ri = n2 / np.maximum(s2, 1e-12)
    x = 6.6 * ri
    live = interior & (x > PR_BAND[0]) & (x < PR_BAND[1])
    pr = np.clip(x, *PR_BAND)
    G = np.where(live, km * 6.6 / (pr ** 2 * np.maximum(s2, 1e-12)), 0.0)
    n2l, gl = n2[live], G[live]
    order = np.argsort(n2l)
    cg = np.cumsum(gl[order]) / max(gl.sum(), 1e-300)
    out = {"n_interior": int(interior.sum()), "n_live": int(live.sum()),
           "G_total": float(gl.sum()), "G_max": float(gl.max()) if gl.size else 0.0,
           "n2_live_quantiles": {str(q): float(np.quantile(n2l, q))
                                 for q in (0.01, 0.1, 0.5, 0.9, 0.99)} if gl.size else {},
           "n2_at_G_share": {str(q): float(n2l[order][np.searchsorted(cg, q)])
                             for q in (0.5, 0.9, 0.99, 0.999)} if gl.size else {},
           "per_n2_star": []}
    for t in n2_stars:
        below = n2l < t
        out["per_n2_star"].append({
            "n2_star": float(t), "share_cells": float(below.mean()) if gl.size else 0.0,
            "share_G": float(gl[below].sum() / max(gl.sum(), 1e-300)),
            "G_max_above": float(gl[~below].max()) if (~below).any() else 0.0})
    return out


def top_nodes(svd_row: dict, n: int) -> list:
    """The first ``n`` distinct nodes among the top tke cells of an svd row's input vector."""
    out = []
    for c in svd_row.get("top_v") or []:
        if c.get("field") == "tke" and c["node"] not in out:
            out.append(int(c["node"]))
        if len(out) == n:
            break
    return out


def load_vector(path, like):
    """A State vector from an svd npz (``v.<leaf>`` entries) on the shape of ``like``."""
    z = np.load(path)
    leaves = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith("v.")}
    if not leaves:
        raise ValueError(f"{path} has no v.* arrays")
    return _state_from_leaves(like, leaves)


# ---------------------------------------------------------------------------
# command line

def _du_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def start_state(args, out_dir: Path):
    """``(state or None, info)``: with ``--spinup-from`` a forward from the saved State of that
    date to ``--start`` (no bulk State store: recomputed on demand; ``--save-state`` writes the
    one State this run uses and records its footprint)."""
    if not args.spinup_from:
        return None, {}
    t0, t1 = C._parse_date(args.spinup_from), C._parse_date(args.start)
    n = int(round((t1 - t0).total_seconds() / C.DT))
    if n <= 0:
        raise ValueError("--spinup-from must be before --start")
    ck = math.gcd(n, 48)
    m0 = C.build_model(t0, n_steps=n, chunk_steps=ck, restart=args.spinup_restart,
                       ice_mode=args.ice)
    r = C.run_fwd(m0, ("box_T_vw",))
    info = {"spinup_from": t0.isoformat(), "spinup_steps": n, "spinup_restart": str(m0.restart),
            "spinup_wall_s": r["wall_s"]}
    if args.save_state:
        d = out_dir / f"state_{C.restart_tag(t1)}"
        C._write_state(d, r["_final"], m0.mesh, step=n, when=t1,
                       attrs={"spinup_from": str(m0.restart)})
        info.update(saved_state=str(d), saved_state_bytes=_du_bytes(d))
    return r["_final"], info


def _parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["screen", "svd", "cuts", "column", "fdlin", "sites",
                                       "locate", "colcensus", "prcensus"])
    ap.add_argument("--two-step", action="store_true",
                    help="colcensus: also the two-step recursion J(t1)·J(t0)")
    ap.add_argument("--n2-stars", default="1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5",
                    help="prcensus: N²* values")
    ap.add_argument("--stride", type=int, default=1,
                    help="screen: store every stride-th chunk boundary (host memory)")
    ap.add_argument("--spec", default=None,
                    help="sites / locate: JSON file (see run_sites / run_locate)")
    ap.add_argument("--ssh-tol", type=float, default=None,
                    help="relative SSH forward tolerance for EVERY program of this run "
                         "(a different forward model; for fdlin; default: the model's SOLTOL)")
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--restart", default=None)
    ap.add_argument("--spinup-from", default=None, help="date of a saved State to run from")
    ap.add_argument("--spinup-restart", default=None)
    ap.add_argument("--save-state", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--steps", type=int, default=None)
    g.add_argument("--days", type=float, default=None)
    ap.add_argument("--chunk-steps", type=int, default=6)
    ap.add_argument("--adjoint", choices=["exact", "long_window"], default="exact")
    ap.add_argument("--setting", default="none",
                    help="extra cuts/flags on top of --adjoint, '+'-joined (e.g. tke.km_e)")
    ap.add_argument("--ice", choices=C.ICE_MODES, default="prescribed")
    ap.add_argument("--cost", default="box_T_vw")
    ap.add_argument("--box", default=C.DEFAULT_BOX)
    ap.add_argument("--box-bounds", default=None,
                    help="lat_min,lat_max,lon_min,lon_max,z_min,z_max (registers --box)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--map-every", type=int, default=None, help="screen: map interval (steps)")
    ap.add_argument("--also", default="",
                    help="screen: comma list of further settings on the same model "
                         "('long_window', or '+'-joined cuts/flags on top of --setting; a "
                         "leading '=' puts them on top of --adjoint alone); 'setting*k' runs "
                         "that one k times (default --also-repeats)")
    ap.add_argument("--also-repeats", type=int, default=1,
                    help="screen: repeats of each --also setting (same window, same programs)")
    ap.add_argument("--check-linear", choices=["zero", "full", "off"], default="zero")
    ap.add_argument("--weights", choices=WEIGHTS, default="tke")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--singles", default=",".join(sorted(KNOWN_CUTS)),
                    help="cuts: comma list of single settings")
    ap.add_argument("--no-pairs", action="store_true")
    ap.add_argument("--nodes", default="",
                    help="column: comma list of node ids, or 'svd' (the top tke cells of the "
                         "window's leading vector, from a power iteration in this run)")
    ap.add_argument("--n-nodes", type=int, default=6, help="column --nodes svd: how many")
    ap.add_argument("--power-iters", type=int, default=0)
    ap.add_argument("--vector", default=None,
                    help="fdlin: svd npz with the direction v, or 'svd' (power iteration of "
                         "this window first, in this run; --weights/--box/--iters apply)")
    ap.add_argument("--hs", default=None, help="fdlin: comma list (default 1e-1..1e-12)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    job = os.environ.get("SLURM_JOB_ID") or datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out = (Path(args.out) if args.out
           else RUNS_DIR / f"{args.action}_{args.tag or 'run'}_{job}")
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / "results.jsonl"
    if args.box_bounds:
        C.BOXES[args.box] = C.Box(*[float(x) for x in args.box_bounds.split(",")])
    if args.ssh_tol is not None:
        set_ssh_forward_tol(args.ssh_tol)          # before any program is traced
    n_steps = (args.steps if args.steps is not None
               else int(round((args.days if args.days is not None else 1.0) * 86400.0 / C.DT)))
    base = (AdjointConfig.long_window() if args.adjoint == "long_window"
            else AdjointConfig.exact())
    cfg = parse_setting(args.setting, base)
    state, sinfo = start_state(args, out)
    model = C.build_model(args.start, n_steps, args.chunk_steps, args.restart, cfg, args.ice,
                          state=state)
    lin = {"zero": True, "full": "full", "off": False}[args.check_linear]
    tol, cap = ssh_forward_tol()
    extra = dict(sinfo, setting=args.setting, tag=args.tag, ssh_forward_tol=tol,
                 ssh_maxiter=cap)
    if args.action == "screen":
        # rows are written as each setting finishes (a long job keeps what it has done)
        def write(rs):
            for r in rs:
                C.append_row(jpath, dict(extra, **{k: v for k, v in r.items()
                                                   if not str(k).startswith("_")}))
        write(run_screen(model, out, cost=args.cost, box=args.box, repeats=args.repeats,
                         map_every=args.map_every, check_linear=lin, stride=args.stride))
        for spec, nrep in parse_also(args.also, args.also_repeats):
            if spec.startswith("="):            # on top of --adjoint alone, not --setting
                spec = spec[1:]
                cfg2 = parse_setting(spec, base)
            else:
                cfg2 = (AdjointConfig.long_window() if spec == "long_window"
                        else parse_setting(spec, cfg))
            rs = run_screen(with_adjoint(model, cfg2), out, cost=args.cost, box=args.box,
                            repeats=nrep, map_every=args.map_every,
                            check_linear=lin, stride=args.stride,
                            label=f"screen_{spec.replace('+', '__')}")
            for r in rs:
                r["setting"] = spec
            write(rs)
        rows = []
    elif args.action == "sites":
        spec = json.loads(Path(args.spec).read_text())
        extra["spec"] = str(args.spec)
        rows = run_sites(model, spec, out, jpath, extra)
    elif args.action == "colcensus":
        singles = [x for x in args.singles.split(",") if x]
        st0 = model.s0 if state is None else state
        rows = run_colcensus(model, st0, model.start, singles, out, top=args.top,
                             two_step=args.two_step)
    elif args.action == "prcensus":
        st0 = model.s0 if state is None else state
        inp = step_tke_inputs(model, st0, step_forcing(model, model.start, 0))
        stars = [float(x) for x in args.n2_stars.split(",") if x]
        with C._Timer() as t:
            cen = pr_census(inp, stars)
        rows = [C.provenance(model, "tke_prcensus", timer=t, n_steps=1, start=model.start,
                             **cen)]
    elif args.action == "locate":
        spec = json.loads(Path(args.spec).read_text())
        extra["spec"] = str(args.spec)
        run_locate(model, spec, out, jpath, extra)
        rows = []
    else:
        win = Window(model)
        extra["forward_s"] = win.forward_timer.wall
        kw = dict(weights=args.weights, box=args.box, iters=args.iters, tol=args.tol,
                  seed=args.seed, top=args.top)
        if args.action == "svd":
            rows = [run_svd(model, win, out, **kw)]
        elif args.action == "cuts":
            singles = [s for s in args.singles.split(",") if s]
            rows = run_cuts(model, win, out, singles, pairs=not args.no_pairs, jpath=jpath,
                            row_extra=extra, **kw)
        elif args.action == "column":
            rows = []
            if args.nodes == "svd":             # the top tke cells of the leading vector
                r = run_svd(model, win, out, **kw)
                nodes = top_nodes(r, int(args.n_nodes))
                rows.append(r)
            else:
                nodes = [int(n) for n in args.nodes.split(",") if n]
            rows.append(run_column(model, win, nodes, power_iters=args.power_iters))
        else:
            rows = []
            if args.vector == "svd":            # the leading vector of this very window
                r = run_svd(model, win, out, **kw)
                v = r["_v"]
                rows.append(r)
            else:
                v = load_vector(args.vector, model.state_of(win.pins[0]))
            rows += run_fdlin(model, win, v, weights=args.weights,
                              hs=args.hs.split(",") if args.hs else None)
    for r in rows:
        r = {k: v for k, v in r.items() if not str(k).startswith("_")}
        r = dict(extra, **r)
        if args.action not in ("cuts", "sites", "locate"):   # appended their rows already
            C.append_row(jpath, r)
        print(json.dumps(C._jsonable({k: v for k, v in r.items()
                                      if k not in ("history", "top_v", "top_u", "columns",
                                                   "field_trace", "predicted")}))[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
