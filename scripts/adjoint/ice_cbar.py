#!/usr/bin/env python
"""Sea-ice drift gradients w.r.t. the ice strength P* and the ice-ocean drag C_d on the CORE2
reference model, against replay and full-model finite differences, in resumable pieces (the
stage-1 "ice bar" of the sea-ice adjoint track).

    python scripts/adjoint/ice_cbar.py fwd --run-dir BASE --save-boundaries --save-ice
    python scripts/adjoint/ice_cbar.py fwd --run-dir R0 --replay BASE        # h = 0 self-check
    python scripts/adjoint/ice_cbar.py fwd --run-dir D --control pstar --h 1e-2 --sign -1 \\
                                       [--replay BASE]
    python scripts/adjoint/ice_cbar.py fdbatch --fd-root FD --base-dir BASE \\
                                       --modes replay,full --controls pstar,cd --hs 3e-3,1e-2,3e-2
    python scripts/adjoint/ice_cbar.py grad --ice-mode M --ice-settings S --setting TKE \\
                                       --window-days 5 --base-dir BASE --grad-root G --tag T
    python scripts/adjoint/ice_cbar.py tkegrad --ice-mode M --setting TKE --cost T \\
                                       --window-days 5 --base-dir TKEBASE --run-dir D
    python scripts/adjoint/ice_cbar.py summary --fd-root FD --grad-root G --out table.json

Run from a checkout with ``PYTHONPATH=<checkout>`` (``scripts/adjoint/job_cpu.sbatch`` on CPU,
the dolpung GH200 wrapper on GPU). Every output path is an argument and must lie outside the
checkout.

**Cost.** Box ``arctic`` (lat >= 70N) or, with ``--box antarctic``, lat <= 60S (both registered
in ``core2_adj.BOXES`` here). Every run records its box (``drift_box``), and a resumed forward,
an FD batch, a gradient, the noise floor and the summary refuse runs made with another box.
After every step ``k`` the accumulator ``drift`` adds
``D_k = sum_i w_i a_i |u_i| / sum_i w_i a_i`` over the wet box nodes (``w`` = node area, ``|u|``
the AD-safe speed). ``J_drift(n) = S(n)/n``: one sweep serves every window, as for
``tke_cbar``'s ``J_Tfull``.

**Controls.** The default log controls of ``core2_adj`` plus, opt-in (``--controls``),
``ln_ice_pstar`` and ``ln_ice_cd``: ``p = p_cfg * exp(ln p)`` with ``p_cfg`` the model's
``IceConfig.pstar`` / ``cd_oce_ice``, handed to the ice step as ``Params.ice_pstar`` /
``Params.ice_cd``. ``ln = 0`` gives the configured values exactly; the forward then equals the
constant-config forward to rounding only (a traced value is a different XLA program from a
constant), so every run that is compared with another one (base, repeat, replay, full FD)
carries the same controls.

**Replay FD.** The base run stores ``(a_ice, m_ice, m_snow, t_skin)`` after every step
(``icestate_<c>.npz``, one per chunk, ``[K, nod2D]`` per field). A ``--replay`` forward
overwrites those four fields with the base values after every step, BEFORE the accumulator,
so the perturbed run differs from the base only through the paths the stage-1 adjoint
differentiates (ice velocity and stress across steps; within a step the ice velocity ->
advection -> fluxes; the ocean). The full-model FD is the plain perturbed forward. Every
forward of this script runs ONE compiled chunk program (replay on/off is a traced flag, the
parameters a traced argument), so base, repeat, replay and full runs share the program.

**Gradients.** ``grad`` runs the reverse sweep of ``J_drift`` over the base boundaries (linked,
never copied) for any ``IceConfig.adjoint_mode`` (``--ice-mode``) with any extra ``IceConfig``
fields (``--ice-settings``, ``name=value+name=value`` or a JSON dict, merged with ``_replace``)
and the TKE derivative setting (``--setting``, ``tke_diag.parse_setting``); resumable legs as
``tke_cbar`` (``LegCheckpoint``, ``--stride``). ``tkegrad`` runs ``tke_cbar``'s J_T / J_Tfull
/ J_MLD on such a model, built with ``tke_cbar.accumulators`` so the TKE CPU base boundaries
load, and writes ``tke_cbar``'s layout (``<cost>_w<days>/done.json``) for ``tke_cbar.py
summary``. ``summary`` tabulates replay-FD, full-FD and adjoint per (control, window), the
verdict with ``tke_cbar.bar_verdict``'s rule at EVERY window, and the amplification of each
reverse trace. ``grad`` or ``tkegrad --dump-cot 29-34 [--dump-primal] [--dump-fields S,T|all]`` writes the
carry cotangent (and the forward carry, re-run from the nearest base boundary) at those chunk
boundaries to ``<run dir>/dump/cot_<c>.npz`` / ``primal_<c>.npz``, keyed like ``field_trace``
(``0.S``), to locate where a reverse trace grows. ``tkegrad --grad-trace`` writes the control
gradient accumulated after every reverse chunk to ``<run dir>/grad_trace.jsonl``: the sensitivity
to the controls from that boundary to the window end, which an FD that starts its perturbation
at the boundary checks. ``tkegrad --field-controls cd,alpha`` makes those TKE constants per-node
fields (all at the default, so the forward model is unchanged): their gradient is a map whose
sum is the scalar gradient, saved to ``<run dir>/grad_maps.npz``, and the trace also gets the
sum over each of :func:`trace_regions` (``ln_tke_cd@box``, ...).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core2_adj as C  # noqa: E402  (runs the import guard)
import tke_cbar as T  # noqa: E402

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fesom_jax import adjoint  # noqa: E402
from fesom_jax import adjoint_costs as AC  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax import tracer_adv  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig, amplification  # noqa: E402
from fesom_jax.forcing import _safe_speed  # noqa: E402
from fesom_jax.ice import IceConfig  # noqa: E402

BOX = "arctic"
#: drift-cost boxes, picked with ``--box`` (default :data:`BOX`)
DRIFT_BOXES = {"arctic": C.Box(70.0, 90.0, -180.0, 180.0, 0.0, 50.0),
               "antarctic": C.Box(-90.0, -60.0, -180.0, 180.0, 0.0, 50.0)}
for _name, _b in DRIFT_BOXES.items():
    C.BOXES.setdefault(_name, _b)
#: control name -> (control-pytree key, IceConfig field, Params field)
ICE_CONTROLS = {"pstar": ("ln_ice_pstar", "pstar", "ice_pstar"),
                "cd": ("ln_ice_cd", "cd_oce_ice", "ice_cd")}
REPLAY_FIELDS = ("a_ice", "m_ice", "m_snow", "t_skin")
FP_FIELDS = ("T", "u_ice", "v_ice", "a_ice", "eta_n")   #: per-boundary state fingerprints
WINDOW_DAYS = (1.0, 2.0, 5.0, 10.0)
HS = (3e-3, 1e-2, 3e-2)
FD_MODES = ("replay", "full")
RUNS_ICE = Path("/work/ab0995/a270088/fesom_jax_dev/runs/ice")


# ---------------------------------------------------------------------------
# paths


def out_path(p) -> Path:
    """``p`` as a Path, refusing any location inside the checkout (outputs live on /work)."""
    q = Path(p).resolve()
    root = C.ROOT.resolve()
    if q == root or root in q.parents:
        raise ValueError(f"output path {q} is inside the checkout {root}; write outputs "
                         f"under {RUNS_ICE}")
    return Path(p)


# ---------------------------------------------------------------------------
# cost


def arctic_weights(mesh, box=BOX) -> np.ndarray:
    """Node-area weights of the wet surface nodes of ``box`` (zero elsewhere)."""
    node, _ = C.box_masks(mesh, box)
    wet = np.asarray(mesh.node_layer_mask[:, 0]) & np.asarray(node, bool)
    return np.where(wet, np.asarray(mesh.areasvol[:, 0]), 0.0)


def drift_value(state, w):
    """Ice-area-weighted mean ice speed over the box of one state (0 without ice; a NaN
    state gives NaN — the no-ice guard tests ``== 0``, so it cannot hide a NaN)."""
    wa = w * state.a_ice
    den = jnp.sum(wa)
    num = jnp.sum(wa * _safe_speed(state.u_ice, state.v_ice))
    empty = den == 0.0
    return jnp.where(empty, 0.0, num / jnp.where(empty, 1.0, den))


def accumulators(mesh, box=BOX):
    """``{"drift": D_k}`` per step (the running sum of the carry)."""
    w = jnp.asarray(arctic_weights(mesh, box))
    return {"drift": lambda s: drift_value(s, w)}


def run_box(run_dir: Path):
    """The drift box a run directory was made with, from its ``results.jsonl`` rows: None
    without rows, :data:`BOX` for rows written before the box became a choice."""
    p = Path(run_dir) / "results.jsonl"
    rows = ([json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            if p.exists() else [])
    boxes = {r.get("drift_box", BOX) for r in rows}
    if len(boxes) > 1:
        raise ValueError(f"{run_dir} mixes drift boxes {sorted(boxes)}")
    return boxes.pop() if boxes else None


def check_box(run_dir: Path, box: str, what: str) -> None:
    """Refuse ``run_dir`` if it holds rows of another drift box than ``box``."""
    b = run_box(run_dir)
    if b is not None and b != box:
        raise ValueError(f"{what} {run_dir} was made with drift box {b!r}; this run uses {box!r}")


def j_drift(S: dict, n: int) -> float:
    """``J_drift`` of an ``n``-step window from ``{steps: S(steps)}``."""
    return float(S[n] / n)


def drift_loss(n: int):
    return lambda carry: carry[1]["drift"] / n


# ---------------------------------------------------------------------------
# controls


def parse_controls(spec) -> tuple:
    names = tuple(x for x in (spec or "").split(",") if x and x != "none")
    bad = [x for x in names if x not in ICE_CONTROLS]
    if bad:
        raise ValueError(f"unknown ice control(s) {bad}; known: {sorted(ICE_CONTROLS)}")
    return names


def ice_params0(controls) -> dict:
    """The default control pytree plus ``ln_ice_<c>`` = 0 for each requested ice control."""
    u = dict(C.default_params0(0))
    for c in controls:
        u[ICE_CONTROLS[c][0]] = np.zeros((), np.float64)
    return u


def make_to_params(ice_cfg, controls):
    """``u -> Params``: :func:`core2_adj.default_to_params` plus the ice controls, ``p =
    p_cfg * exp(ln p)`` with ``p_cfg`` read from ``ice_cfg`` (an ``IceConfig``, or a callable
    returning it at trace time)."""
    get = ice_cfg if callable(ice_cfg) else (lambda: ice_cfg)
    controls = tuple(controls)

    def to_params(u):
        p = C.default_to_params(u)
        if not controls:
            return p
        cfg = get()
        repl = {}
        for c in controls:
            key, field, pfield = ICE_CONTROLS[c]
            repl[pfield] = jnp.asarray(getattr(cfg, field), jnp.float64) * jnp.exp(u[key])
        return dataclasses.replace(p, **repl)
    return to_params


def perturbed(params0: dict, control: str, h: float, sign: int) -> dict:
    u = {k: np.array(v) for k, v in params0.items()}
    if control != "none":
        key = ICE_CONTROLS[control][0]
        if key not in u:
            raise ValueError(f"control {control!r} is not in this model's controls")
        u[key] = np.asarray(sign * h, np.float64)
    return C._strong_f64(u)


# ---------------------------------------------------------------------------
# IceConfig settings pass-through


def _typed(field: str, raw):
    default = getattr(IceConfig(), field)
    if isinstance(raw, str):
        if isinstance(default, bool):
            return raw.lower() in ("1", "true", "yes")
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        return raw
    return raw


def parse_ice_settings(spec) -> dict:
    """``"name=value+name=value"``, a JSON object, a dict, or ``"none"``/``""`` -> a dict of
    ``IceConfig`` fields (values typed like the field's default). Unknown fields raise."""
    if spec is None:
        return {}
    if isinstance(spec, dict):
        d = dict(spec)
    else:
        s = str(spec).strip()
        if s in ("", "none"):
            return {}
        if s.startswith("{"):
            d = json.loads(s)
        else:
            d = {}
            for tok in s.split("+"):
                name, eq, val = tok.partition("=")
                if not eq:
                    raise ValueError(f"ice setting {tok!r} is not name=value")
                d[name.strip()] = val.strip()
    bad = [k for k in d if k not in IceConfig._fields]
    if bad:
        raise ValueError(f"unknown IceConfig field(s) {bad}")
    if "adjoint_mode" in d:
        raise ValueError("set the ice adjoint mode with --ice-mode, not in the settings")
    return {k: _typed(k, v) for k, v in d.items()}


def apply_ice_settings(model, settings: dict) -> None:
    """Merge ``settings`` into the model's ``IceConfig`` (``_replace``) and rebuild the chunk
    programs (they close over the physics). Empty ⇒ nothing changes."""
    if not settings:
        return
    model.phys["ice_cfg"] = model.phys["ice_cfg"]._replace(**settings)
    model.chunk_fn = C._make_chunk_fn(model, rematerialize=model.rematerialize)
    model._chunk_fn_tl = None


# ---------------------------------------------------------------------------
# model


def build(start, n_steps, chunk_steps, *, adjoint_cfg=None, ice_mode="prescribed",
          ice_settings=None, controls=("pstar", "cd"), accs=None, ice_adjoint=None, box=BOX,
          tke_settings=None, fast=None, restart=None, devices=1, **schedule):
    """The CORE2 reference model with the drift accumulator over ``box`` (or ``accs``), the
    ice adjoint mode, its derivative-only mEVP settings (``ice_adjoint``,
    ``core2_adj.build_model``), the extra ``IceConfig`` settings and the opt-in ice controls;
    ``schedule``: ``build_model``'s forcing controls and cost interval (``tke_cbar.schedule_kwargs``)."""
    if box not in DRIFT_BOXES:
        raise ValueError(f"unknown drift box {box!r}; known: {sorted(DRIFT_BOXES)}")
    if accs is None:
        accs = functools.partial(accumulators, box=box)
    controls = tuple(controls)
    holder = {}
    kw = {}
    if controls:
        kw = dict(params0=ice_params0(controls),
                  to_params=make_to_params(lambda: holder["ice"], controls))
    model = C.build_model(start, n_steps, chunk_steps, restart,
                          adjoint_cfg or AdjointConfig.exact(), ice_mode,
                          accumulators=accs, rematerialize=True, ice_adjoint=ice_adjoint,
                          tke_settings=tke_settings, fast=fast, devices=devices, **kw, **schedule)
    apply_ice_settings(model, parse_ice_settings(ice_settings))
    holder["ice"] = model.phys["ice_cfg"]
    model.ice_settings = parse_ice_settings(ice_settings)
    model.ice_controls = controls
    model.drift_box = box
    return model


def _overwrite(state, rep, flag):
    return dataclasses.replace(state, **{f: jnp.where(flag, rep[f], getattr(state, f))
                                         for f in REPLAY_FIELDS})


def fwd_program(step_fn, accs: dict, to_params):
    """``jit(chunk(carry, (xs, rep), u, flag) -> (carry, ys))``: ``step_fn(state, x, params)``
    per step; ``flag`` (traced bool) overwrites :data:`REPLAY_FIELDS` with ``rep`` after each
    step, before the accumulators; ``ys`` = those fields after every step ``[K, ...]``."""
    accs = dict(accs)

    def chunk(carry, xs, u, flag):
        p = to_params(u)

        def body(c, x):
            st, acc = c
            sf, rep = x
            new = _overwrite(step_fn(st, sf, p), rep, flag)
            acc = {k: acc[k] + fn(new) for k, fn in accs.items()}
            return (new, acc), {f: getattr(new, f) for f in REPLAY_FIELDS}
        return jax.lax.scan(body, carry, xs)
    return jax.jit(chunk)


def model_program(model):
    """:func:`fwd_program` for a CORE2 :class:`core2_adj.Model` (built once per model)."""
    prog = getattr(model, "_ice_fwd_prog", None)
    if prog is None:
        mesh, op, fs, phys = model.mesh, model.op, model.forcing.static, model.phys
        adj, dt = model.adjoint_cfg, model.dt

        def step_fn(s, x, p):
            return stepmod.step(s, mesh, op, None, p, dt=dt, is_first_step=False,
                                step_forcing=x, forcing_static=fs, adjoint_cfg=adj, **phys)
        prog = fwd_program(step_fn, model.accumulators, model.to_params)
        model._ice_fwd_prog = prog
    return prog


# ---------------------------------------------------------------------------
# fwd


def _fingerprint(state) -> dict:
    return {f: float(jnp.sum(getattr(state, f))) for f in FP_FIELDS if hasattr(state, f)}


def icestate_path(base_dir: Path, c: int) -> Path:
    return Path(base_dir) / f"icestate_{c}.npz"


def load_icestate(base_dir: Path, c: int) -> dict:
    p = icestate_path(base_dir, c)
    with np.load(p) as z:
        return {f: np.asarray(z[f]) for f in REPLAY_FIELDS}


def _save_icestate(path: Path, ys) -> None:
    if path.exists():
        return
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **{k: np.asarray(v) for k, v in ys.items()})
    tmp.rename(path)


def missing_replay_chunks(base_dir: Path, chunks) -> list:
    return [c for c in chunks if not icestate_path(base_dir, c).exists()]


def run_fwd(model, run_dir: Path, *, params, n_steps: int, save_boundaries: bool = False,
            save_ice: bool = False, replay_dir: Path | None = None,
            budget_s: float = math.inf, prog=None, save_every: int = 1) -> dict:
    """Advance chunk by chunk, appending ``S(steps)`` (``drift_sum``) and state fingerprints to
    ``fwd.jsonl`` at every boundary; ``save_boundaries`` writes ``boundary_<c>.npz`` (the
    :class:`core2_adj.DiskCheckpoint` layout) at every ``save_every``-th boundary and the last
    (a grad over it needs a ``stride`` that is a multiple of ``save_every``), ``save_ice`` the
    per-step replay fields ``icestate_<c>.npz`` of chunk ``c``; ``replay_dir`` overwrites them
    from that base. Out of budget: the carry goes to ``carry_<c>.npz`` (unless a boundary was
    just saved) and the result has ``status="incomplete"``; a rerun resumes from the furthest
    saved carry or boundary.
    Nothing is overwritten. Returns ``{status, chunks_from, chunks_to, first_chunk_s,
    s_per_step}``."""
    run_dir = out_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    K = model.chunk_steps
    nc = model.chunks_for(n_steps)
    prog = prog if prog is not None else model_program(model)
    like = model.init_carry(host=True)
    saved = [(int(p.stem.split("_")[1]), p) for pat in ("carry_*.npz", "boundary_*.npz")
             for p in run_dir.glob(pat)]
    saved = [(c, p) for c, p in saved if c <= nc]
    if replay_dir is not None:
        replay_dir = Path(replay_dir)
        c_from = max(saved)[0] if saved else 0
        miss = missing_replay_chunks(replay_dir, range(c_from, nc))
        if miss:
            raise FileNotFoundError(f"replay base {replay_dir} lacks icestate chunk(s) {miss}")
    if saved:
        c0, p = max(saved)
        carry = jax.device_put(T._host_load(p, like))
    else:
        c0, carry = 0, model.init_carry()
        if save_boundaries:
            T._host_save(run_dir / "boundary_0.npz", jax.device_get(carry))
    flag = jnp.asarray(replay_dir is not None)
    zeros = None
    xs_fn = model.xs_fn()
    clock = T.LegClock(budget_s)
    t0, tc = time.time(), None
    status = "complete"
    c = c0 - 1
    for c in range(c0, nc):
        xs = xs_fn(c)
        if replay_dir is not None:
            rep = jax.device_put(load_icestate(replay_dir, c))
        else:
            if zeros is None:
                st = model.state_of(like)
                zeros = jax.device_put({f: np.zeros((K,) + np.shape(getattr(st, f)),
                                                    np.asarray(getattr(st, f)).dtype)
                                        for f in REPLAY_FIELDS})
            rep = zeros
        nxt, ys = prog(carry, (xs, rep), params, flag)
        nxt = jax.block_until_ready(nxt)
        adjoint._free(xs)
        adjoint._free(carry)
        if replay_dir is not None:
            adjoint._free(rep)
        carry = nxt
        if tc is None:
            tc = time.time() - t0                  # the first chunk includes the compile
        steps = (c + 1) * K
        st = model.state_of(carry)
        row = {"chunk": c + 1, "steps": steps, "drift_sum": float(carry[1]["drift"]),
               "fp": _fingerprint(st), "wall_s": time.time() - t0}
        with open(run_dir / "fwd.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        if save_ice:
            _save_icestate(icestate_path(run_dir, c), jax.device_get(ys))
        adjoint._free(ys)
        kept = save_boundaries and ((c + 1) % save_every == 0 or c + 1 == nc)
        if kept:
            T._host_save(run_dir / f"boundary_{c + 1}.npz", jax.device_get(carry))
        if clock.out_of_time() and c + 1 < nc:
            if not kept:
                T._host_save(run_dir / f"carry_{c + 1}.npz", jax.device_get(carry))
            status = "incomplete"
            break
    adjoint._free(carry)
    n_done = (c + 1 - c0) if nc > c0 else 0
    per_step = ((time.time() - t0 - (tc or 0.0)) / max(1, (n_done - 1) * K)
                if n_done > 1 else float("nan"))
    return {"status": status, "chunks_from": c0, "chunks_to": c + 1, "first_chunk_s": tc,
            "s_per_step": per_step, "run_dir": str(run_dir)}


def fwd_records(run_dir: Path) -> dict:
    """``{steps: row}`` of a fwd run (the last row per step wins; S(0) = 0 is implied)."""
    out = {}
    p = Path(run_dir) / "fwd.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[int(r["steps"])] = r
    return out


def drift_sums(run_dir: Path) -> dict:
    S = {0: 0.0}
    S.update({n: float(r["drift_sum"]) for n, r in fwd_records(run_dir).items()})
    return S


def compare_runs(run_dir: Path, base_dir: Path) -> dict:
    """Boundary-by-boundary comparison of two fwd runs: drift sums and state fingerprints.
    ``bitwise`` = every common record identical."""
    a, b = fwd_records(run_dir), fwd_records(base_dir)
    common = sorted(set(a) & set(b))
    d_s, d_fp, equal = [], {}, True
    for n in common:
        ds = abs(a[n]["drift_sum"] - b[n]["drift_sum"])
        d_s.append(ds)
        equal &= a[n]["drift_sum"] == b[n]["drift_sum"]
        for k, v in a[n].get("fp", {}).items():
            w = b[n].get("fp", {}).get(k)
            if w is None:
                continue
            d_fp[k] = max(d_fp.get(k, 0.0), abs(v - w))
            equal &= v == w
    return {"n_common": len(common), "bitwise": bool(equal and common),
            "max_abs_drift_sum": max(d_s) if d_s else None,
            "max_abs_fp": d_fp, "steps": common}


# ---------------------------------------------------------------------------
# grad


class _NoSave(Exception):
    pass


def run_grad(model, run_dir: Path, base_dir: Path, *, n_steps: int, budget_s: float,
             stride: int = 1, on_cot=None, lazy: bool = True) -> dict:
    """Reverse sweep of ``J_drift(n_steps)`` over the base boundaries (linked into
    ``run_dir``); resumable legs (``tke_cbar.LegCheckpoint``). ``on_cot(c, cot)``: called at
    every chunk boundary with the carry cotangent (:func:`dump_cot_hook`)."""
    run_dir = out_path(run_dir)
    box = getattr(model, "drift_box", BOX)
    check_box(base_dir, box, "base")
    check_box(run_dir, box, "gradient run")
    nc = model.chunks_for(n_steps)
    T.link_base(Path(base_dir), run_dir, nc, stride)
    loss = drift_loss(n_steps)
    fj = run_dir / "forward.json"
    if not fj.exists():
        last = jax.device_put(T._host_load(run_dir / f"boundary_{nc}.npz",
                                           model.init_carry(host=True)))
        fj.write_text(json.dumps({"loss": float(loss(last))}))
        adjoint._free(last)
    rows = C.run_grad(model, (loss,), n_steps=n_steps, stride=stride, check_linear=False,
                      resume_dir=run_dir, budget_s=budget_s, action="ice_grad", on_cot=on_cot,
                      checkpoint_factory=lambda d, p, b: T.LegCheckpoint(d, p, budget_s=b,
                                                                         stride=stride,
                                                                         lazy=lazy))
    r = rows[0]
    r["cost"] = "drift"
    if r.get("status") == "ok":
        g = r["_result"]["grad"]
        r["grads"] = {c: float(np.asarray(g[ICE_CONTROLS[c][0]]))
                      for c in getattr(model, "ice_controls", ()) if ICE_CONTROLS[c][0] in g}
        r["grads_log_controls"] = {k: float(np.asarray(v)) for k, v in g.items()
                                   if np.ndim(v) == 0}
    return r


def grad_dir(grad_root: Path, tag: str, days: float) -> Path:
    return Path(grad_root) / tag / f"drift_w{days:g}"


# ---------------------------------------------------------------------------
# dumps: where a reverse trace grows

#: carry fields ``grad --dump-cot`` keeps by default: the tracers, the layer thickness, the
#: vertical diffusivity and velocity, the SSH and the ice state (~0.35 GB per boundary on CORE2)
DUMP_FIELDS = ("T", "S", "T_old", "S_old", "hnode", "Kv", "w", "eta_n",
               "a_ice", "m_ice", "m_snow", "t_skin", "u_ice", "v_ice")


def parse_boundaries(spec) -> tuple:
    """``"29-34"``, ``"29,31,33"`` or a mix -> the sorted chunk boundaries; ``None`` or ``""``
    -> ``()``."""
    out = set()
    for tok in [t.strip() for t in (spec or "").split(",") if t.strip()]:
        lo, dash, hi = tok.partition("-")
        a, b = int(lo), (int(hi) if dash else int(lo))
        if a < 0 or b < a:
            raise ValueError(f"bad boundary range {tok!r}")
        out.update(range(a, b + 1))
    return tuple(sorted(out))


def parse_fields(spec) -> tuple | None:
    """``"S,T"`` -> ``("S", "T")``; ``"all"`` -> ``None`` (every floating leaf); ``None`` or
    ``""`` -> :data:`DUMP_FIELDS`."""
    if not spec:
        return DUMP_FIELDS
    if spec == "all":
        return None
    return tuple(f.strip() for f in spec.split(",") if f.strip())


def pick_fields(tree, fields) -> dict:
    """``{leaf path: host array}`` for the floating leaves of ``tree`` whose last path component
    is in ``fields`` (every floating leaf for ``None``); paths as in ``field_trace``
    (``0.S``, ``1.drift``). A named field the tree does not have raises."""
    out, seen = {}, set()
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        name = jax.tree_util.keystr(path, simple=True, separator=".")
        short = name.rsplit(".", 1)[-1]
        if fields is not None and short not in fields:
            continue
        a = np.asarray(jax.device_get(leaf))
        if a.dtype == jax.dtypes.float0 or not np.issubdtype(a.dtype, np.floating):
            continue
        out[name] = a
        seen.add(short)
    missing = sorted(set(fields or ()) - seen)
    if missing:
        raise ValueError(f"no floating carry field(s) {missing}")
    return out


def _save_dump(path: Path, tree, fields) -> bool:
    """The chosen fields of ``tree`` to a new npz (written to a temporary name and renamed); an
    existing file is kept. True when written."""
    path = Path(path)
    if path.exists():
        return False
    arrs = pick_fields(tree, fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrs)
    tmp.rename(path)
    return True


def global_carry(model):
    """``carry -> carry`` in the dense global layout: the identity for a dense model; for a sharded one
    (``--devices``) the folded State and the folded node sums unfolded from the owned lanes (the dumps keep the
    dense layout the location tools read)."""
    if model.shard is None:
        return lambda carry: carry
    sh = model.shard

    def unfold(carry):
        if isinstance(carry, tuple):
            return sh.unfold_state(carry[0]), {k: sh.unfold_nodes(v) for k, v in carry[1].items()}
        return sh.unfold_state(carry)
    return unfold


def dump_cot_hook(dump_dir: Path, boundaries, fields=DUMP_FIELDS, unfold=None):
    """``on_cot(c, cot)`` for :func:`run_grad`: at each listed chunk boundary ``c`` (the
    ``boundary`` of ``field_trace``) the chosen fields of the carry cotangent go to
    ``<dump_dir>/cot_<c>.npz`` (``unfold``: a sharded model's :func:`global_carry`)."""
    want = frozenset(boundaries)

    def on_cot(c, cot):
        if c in want:
            _save_dump(Path(dump_dir) / f"cot_{c}.npz", cot if unfold is None else unfold(cot), fields)
    return on_cot


#: ``tkegrad --field-controls`` names -> the control leaf made a per-node field ``[nod2D, 1]``
FIELD_CONTROLS = {"c_k": "ln_tke_c_k", "c_eps": "ln_tke_c_eps", "cd": "ln_tke_cd",
                  "alpha": "ln_tke_alpha"}


def trace_regions(mesh) -> dict:
    """Node masks for the regional sums of a field control's gradient: ``box`` (the MLD cost box
    ``natl_subpolar``), ``spna`` (40-66N, 80W-15E without the box), ``n66_70``, ``n70``
    (poleward of 70N, the latitude Prandtl cut), ``trop`` (15S-15N) and ``rest``."""
    lon, lat = C._lonlat_deg(mesh)
    box = C.box_masks(mesh, C.DEFAULT_BOX)[0]
    spna = (lat >= 40) & (lat < 66) & (lon >= -80) & (lon <= 15) & ~box
    reg = {"box": box, "spna": spna, "n66_70": (lat >= 66) & (lat < 70), "n70": lat >= 70,
           "trop": np.abs(lat) < 15}
    reg["rest"] = ~np.logical_or.reduce(list(reg.values()))
    return reg


def grad_trace_hook(path: Path, regions: dict | None = None):
    """``on_chunk(c, norm, grad)`` for a reverse sweep: after chunk ``c`` the scalar leaves of
    the gradient accumulated so far, i.e. the sensitivity to the controls over chunks ``c`` to
    the window end, go to ``path`` as one json row (``boundary``, ``norm``, ``<leaf>``). An FD
    whose perturbation starts at boundary ``c`` measures the same number. A per-node leaf
    (``--field-controls``) gives its sum (``<leaf>``) and its sum over each of ``regions``
    (``<leaf>@<region>``)."""
    path = Path(path)

    def on_chunk(c, norm, grad):
        row = {"boundary": int(c), "norm": float(norm)}
        for kp, leaf in jax.tree_util.tree_flatten_with_path(grad)[0]:
            if leaf.dtype == jax.dtypes.float0:
                continue
            name = jax.tree_util.keystr(kp, simple=True, separator=".")
            if leaf.size == 1:
                row[name] = float(np.asarray(leaf).reshape(()))
            elif regions is not None:
                a = np.asarray(leaf).reshape(-1)
                row[name] = float(a.sum())
                for r, m in regions.items():
                    row[f"{name}@{r}"] = float(a[m].sum())
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
    return on_chunk


def dump_primal(model, base_dir: Path, dump_dir: Path, boundaries,
                fields=DUMP_FIELDS) -> list:
    """The forward carry at the listed boundaries to ``<dump_dir>/primal_<c>.npz``: each re-run
    from the nearest base boundary at or below it, with the gradient's chunk program
    (``adjoint._jitted_forward`` is cached on ``model.chunk_fn``, so a later sweep does not
    compile it again). Existing files are kept. Returns the boundaries written."""
    saved = sorted(int(p.stem.split("_")[1]) for p in Path(base_dir).glob("boundary_*.npz"))
    fwd = adjoint._jitted_forward(model.chunk_fn)
    like = model.init_carry(host=True)
    xs_fn = model.xs_fn()
    written, s, at = [], None, None
    for c in sorted(boundaries):
        path = Path(dump_dir) / f"primal_{c}.npz"
        if path.exists():
            continue
        below = [k for k in saved if k <= c]
        if not below:
            raise FileNotFoundError(f"no base boundary at or below {c} in {base_dir}")
        if at is None or at > c or at < below[-1]:
            if s is not None:
                adjoint._free(s)
            s, at = jax.device_put(T._host_load(Path(base_dir) / f"boundary_{below[-1]}.npz",
                                                like)), below[-1]
        while at < c:
            xs = xs_fn(at)
            nxt = jax.block_until_ready(fwd(s, xs, model.params0))
            adjoint._free(xs)
            adjoint._free(s)
            s, at = nxt, at + 1
        _save_dump(path, global_carry(model)(s), fields)
        written.append(c)
    if s is not None:
        adjoint._free(s)
    return written


# ---------------------------------------------------------------------------
# summary


def verdict(ad: float, fd: dict) -> dict:
    """``tke_cbar.bar_verdict``'s rule at any window: reference = plateau median
    (``tke_cbar.plateau``) or the FD at the middle h; pass when ``|ad - ref|/|ref| <=
    BAR_RTOL`` or ``ad`` lies inside the FD range over h. Non-finite FDs are dropped; with no
    finite FD (or no adjoint) the bar is ``None``; a non-finite adjoint (a blown-up reverse
    sweep) against finite FDs FAILS the bar (``False``, ``rel_err`` inf)."""
    fin = {h: v for h, v in fd.items() if v is not None and math.isfinite(v)}
    out = {"n_fd": len(fin), "n_fd_nonfinite": len(fd) - len(fin)}
    if not fin or ad is None:
        return {**out, "plateau": None, "fd_ref": None, "rel_err": None, "fd_range": None,
                "fd_spread": None, "bar": None}
    if not math.isfinite(ad):
        hh = sorted(fin)
        vals = [fin[h] for h in hh]
        pl = T.plateau(hh, vals)
        ref = pl["fd"] if pl else vals[len(vals) // 2]
        return {**out, "plateau": pl, "fd_ref": ref, "rel_err": math.inf,
                "fd_range": [min(vals), max(vals)], "fd_spread": None, "bar": False}
    hh = sorted(fin)
    vals = [fin[h] for h in hh]
    pl = T.plateau(hh, vals)
    ref = pl["fd"] if pl else vals[len(vals) // 2]
    rel = abs(ad - ref) / max(abs(ref), 1e-300)
    lo, hi = min(vals), max(vals)
    spread = max(abs(v - ref) for v in vals) / max(abs(ref), 1e-300)
    return {**out, "plateau": pl, "fd_ref": ref, "rel_err": rel, "fd_range": [lo, hi],
            "fd_spread": spread, "bar": bool(rel <= T.BAR_RTOL or lo <= ad <= hi)}


def fd_run_dir(fd_root: Path, mode: str, control: str, h: float, sign: int) -> Path:
    return Path(fd_root) / mode / f"{control}_h{h:g}_{'p' if sign > 0 else 'm'}"


def central_fd(Sp: dict, Sm: dict, n: int, h: float):
    if n not in Sp or n not in Sm:
        return None
    return (j_drift(Sp, n) - j_drift(Sm, n)) / (2.0 * h)


def fd_table(fd_root: Path, windows: list, hs, controls=tuple(ICE_CONTROLS)) -> dict:
    """``{mode: {control: {n: {h: fd}}}}``; a missing run or window leaves its entry out, a
    NaN run gives NaN."""
    out = {}
    for mode in FD_MODES:
        for c in controls:
            for h in hs:
                Sp = drift_sums(fd_run_dir(fd_root, mode, c, h, 1))
                Sm = drift_sums(fd_run_dir(fd_root, mode, c, h, -1))
                for n in windows:
                    v = central_fd(Sp, Sm, n, h)
                    if v is not None:
                        out.setdefault(mode, {}).setdefault(c, {}).setdefault(n, {})[h] = v
    return out


def _as_float(x):
    """A done.json number back as a float: ``core2_adj._jsonable`` writes non-finite values as
    the strings ``"nan"`` / ``"inf"`` / ``"-inf"``. ``None`` stays ``None``."""
    return None if x is None else float(x)


def _amp(trace, chunk_steps):
    if not trace or len(trace) < 2:
        return None
    trace = [_as_float(t) for t in trace]
    if not all(math.isfinite(t) for t in trace):
        return {"clean": False, "nonfinite_trace": True}
    a = amplification(trace, chunk_steps)
    d = a._asdict()
    d["per_step"] = list(d["per_step"])
    return d


def summarize(fd_root: Path, grad_root: Path | None, *, windows: list, hs, chunk_steps: int,
              dt: float = C.DT, controls=tuple(ICE_CONTROLS)) -> dict:
    """Rows per (tag, control, window): adjoint, replay and full FDs over h, the verdict
    against the replay FD (and, for reference, against the full FD), the replay-vs-full gap
    and the amplification of the reverse trace. Tags without a gradient give FD-only rows
    (tag ``None``)."""
    fd = fd_table(Path(fd_root), windows, hs, controls)
    tags = []
    if grad_root is not None and Path(grad_root).exists():
        tags = sorted(p.name for p in Path(grad_root).iterdir() if p.is_dir())
    boxes = {run_box(fd_run_dir(Path(fd_root), m, c, h, s)) for m in FD_MODES for c in controls
             for h in hs for s in (1, -1)}
    for p in (Path(grad_root).glob("*/drift_w*/done.json") if tags else ()):
        boxes.add(json.loads(p.read_text()).get("drift_box", BOX))
    boxes.discard(None)
    if len(boxes) > 1:
        raise ValueError(f"the FD runs and gradients mix drift boxes {sorted(boxes)}")
    rows = []
    for tag in tags or [None]:
        for c in controls:
            for n in windows:
                days = n * dt / 86400.0
                rep = fd.get("replay", {}).get(c, {}).get(n, {})
                full = fd.get("full", {}).get(c, {}).get(n, {})
                ad, amp, status = None, None, "missing"
                if tag is not None:
                    p = grad_dir(Path(grad_root), tag, days) / "done.json"
                    if p.exists():
                        r = json.loads(p.read_text())
                        status = r.get("status", "ok")
                        ad = _as_float(r.get("grads", {}).get(c))
                        amp = _amp(r.get("trace"), chunk_steps)
                v_rep = verdict(ad, rep)
                v_full = verdict(ad, full)
                gap = None
                if v_rep["fd_ref"] is not None and v_full["fd_ref"] is not None:
                    gap = (v_full["fd_ref"] - v_rep["fd_ref"]) / max(abs(v_rep["fd_ref"]),
                                                                    1e-300)
                rows.append({"tag": tag, "control": c, "window_days": days, "steps": n,
                             "grad_status": status, "adjoint": ad,
                             "fd_replay": {f"{h:g}": rep[h] for h in sorted(rep)},
                             "fd_full": {f"{h:g}": full[h] for h in sorted(full)},
                             "verdict": v_rep, "verdict_full": v_full,
                             "gap_full_vs_replay": gap, "amplification": amp,
                             "amp_clean": None if amp is None else amp["clean"]})
    return {"windows_steps": list(windows), "hs": list(hs), "bar_rtol": T.BAR_RTOL,
            "plateau_rtol": T.PLATEAU_RTOL, "drift_box": boxes.pop() if boxes else None,
            "rows": rows}


def noise_floor(base_dir: Path, rep_dir: Path, windows: list, dt: float = C.DT) -> dict:
    """``|J_drift(base) - J_drift(repeat)|`` per window (two runs of the same forward), and
    that difference over ``2·h·J`` for the smallest step (the scale of the smallest FD
    signal is only known once a replay pair exists, so the relative number uses ``J``)."""
    bb, br = run_box(base_dir), run_box(rep_dir)
    if None not in (bb, br) and bb != br:
        raise ValueError(f"base {base_dir} ({bb!r}) and repeat {rep_dir} ({br!r}) differ in box")
    Sb, Sr = drift_sums(base_dir), drift_sums(rep_dir)
    out = {"windows": []}
    for n in windows:
        if n not in Sb or n not in Sr:
            out["windows"].append({"steps": n, "days": n * dt / 86400.0, "missing": True})
            continue
        jb, jr = j_drift(Sb, n), j_drift(Sr, n)
        out["windows"].append({"steps": n, "days": n * dt / 86400.0, "J_base": jb,
                               "J_repeat": jr, "abs_diff": abs(jb - jr),
                               "rel_diff": abs(jb - jr) / max(abs(jb), 1e-300),
                               "bitwise": jb == jr})
    return out


def _f(x, fmt=".4e"):
    return "-" if x is None else format(x, fmt)


def markdown(table: dict) -> str:
    lines = ["| tag | control | window (d) | adjoint | replay FD ref | full FD ref | rel err "
             "(replay) | bar | gap full/replay | amp median | amp sustained | clean |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table["rows"]:
        v, a = r["verdict"], r["amplification"] or {}
        bar = {True: "PASS", False: "fail", None: "-"}[v["bar"]]
        lines.append(
            f"| {r['tag'] or '-'} | {r['control']} | {r['window_days']:g} | {_f(r['adjoint'])} "
            f"| {_f(v['fd_ref'])} | {_f(r['verdict_full']['fd_ref'])} | {_f(v['rel_err'], '.3g')} "
            f"| {bar} | {_f(r['gap_full_vs_replay'], '.3g')} | {_f(a.get('median'), '.5f')} "
            f"| {_f(a.get('sustained'), '.5f')} | {a.get('clean', '-')} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI


def git_head() -> str:
    """HEAD of the checkout: ``git rev-parse``, or (no git on the GPU nodes) read through the
    worktree's ``.git`` file, its ``HEAD`` and the common dir's refs."""
    h = C._git("rev-parse", "HEAD")
    if h:
        return h
    try:
        g = C.ROOT / ".git"
        gd = Path(g.read_text().split("gitdir:", 1)[1].strip()) if g.is_file() else g
        head = (gd / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(":", 1)[1].strip()
        common = gd / (gd / "commondir").read_text().strip() if (gd / "commondir").exists() else gd
        for base in (gd, common):
            if (base / ref).exists():
                return (base / ref).read_text().strip()
        packed = common / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
    except Exception:          # noqa: BLE001 — provenance must never kill a run
        pass
    return ""


def _ice_prov(model) -> dict:
    """The ice-specific provenance fields (added to ``core2_adj.provenance``'s)."""
    sets = getattr(model, "ice_settings", {})
    return {"ice_settings": sets, "ice_controls": list(getattr(model, "ice_controls", ())),
            "ice_cfg": {k: v for k, v in model.phys["ice_cfg"]._asdict().items()
                        if k in ("pstar", "cd_oce_ice", "evp_rheol_steps", "whichEVP",
                                 "adjoint_mode") + C.ICE_ADJOINT_FIELDS or k in sets},
            "drift_box": getattr(model, "drift_box", BOX), "git_head_resolved": git_head(),
            # the backward-only FCT limiter switch (env FESOM_FCT_STOPGRAD_AE, read at import),
            # recorded as the library saw it: an inert switch once voided an A/B
            # (docs/ADJOINT_HORIZON.md)
            "fct_stopgrad_ae": bool(tracer_adv.FCT_STOP_GRADIENT_AE)}


def _prov(model, action, **extra) -> dict:
    row = {**C.provenance(model, action, **extra), **_ice_prov(model)}
    row["git_head"] = row.get("git_head") or git_head()
    return row


def _emit(path: Path, row: dict) -> None:
    clean = {k: v for k, v in row.items() if not str(k).startswith("_")}
    C.append_row(path, clean)
    print(json.dumps(C._jsonable(clean))[:3000], flush=True)


def _fwd_one(model, d: Path, *, control, h, sign, n, replay, save_boundaries, save_ice,
             budget, job, base_for_check=None, save_every: int = 1) -> dict:
    check_box(d, getattr(model, "drift_box", BOX), "run")
    if replay is not None:
        check_box(replay, getattr(model, "drift_box", BOX), "replay base")
    params = perturbed(model.params0, control, h, sign)
    r = run_fwd(model, d, params=params, n_steps=n, save_boundaries=save_boundaries,
                save_ice=save_ice, replay_dir=replay, budget_s=budget, save_every=save_every)
    row = _prov(model, "ice_fwd", n_steps=n, h=h, control=control, sign=sign,
                replay=str(replay) if replay else None, save_boundaries=save_boundaries,
                save_every=save_every, save_ice=save_ice, slurm_job=job, **r)
    if base_for_check is not None and r["status"] == "complete":
        chk = compare_runs(d, Path(base_for_check))
        chk.pop("steps", None)
        row["check_vs_base"] = chk
        (d / "check_vs_base.json").write_text(json.dumps(C._jsonable(
            {**chk, "base": str(base_for_check), "device": row.get("device"),
             "git_head": row.get("git_head"), "slurm_job": job})))
    _emit(d / "results.jsonl", row)
    return row


def _complete(d: Path) -> bool:
    rj = d / "results.jsonl"
    return rj.exists() and any(json.loads(line).get("status") == "complete"
                               for line in rj.read_text().splitlines() if line.strip())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["fwd", "fdbatch", "grad", "tkegrad", "summary", "noise"])
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--box", default=BOX, choices=sorted(DRIFT_BOXES),
                    help="drift cost box (fwd, fdbatch, grad)")
    ap.add_argument("--chunk-steps", type=int, default=24)
    ap.add_argument("--days", type=float, default=10.0, help="fwd/fdbatch: forward length")
    ap.add_argument("--controls", default=None,
                    help="ice controls in the model (default pstar,cd; tkegrad: none)")
    ap.add_argument("--control", default="none", help="fwd: perturbed control (or none)")
    ap.add_argument("--h", type=float, default=0.0)
    ap.add_argument("--sign", type=int, default=1, choices=(1, -1))
    ap.add_argument("--save-boundaries", action="store_true")
    ap.add_argument("--save-every", type=int, default=1, help="fwd: save every Nth boundary "
                                                              "(and the last)")
    ap.add_argument("--save-ice", action="store_true",
                    help="fwd: store the per-step replay fields (the base run)")
    ap.add_argument("--replay", default=None, help="fwd: base run dir to replay the ice from")
    ap.add_argument("--check-base", default=None,
                    help="fwd: compare the finished run with this run dir (bitwise check)")
    ap.add_argument("--modes", default="replay,full", help="fdbatch: replay and/or full")
    ap.add_argument("--hs", default=",".join(f"{h:g}" for h in HS))
    ap.add_argument("--ice-mode", default="prescribed", help="IceConfig.adjoint_mode")
    ap.add_argument("--ice-damping", choices=("none", "local", "frozen_sweep"), default=None,
                    help="IceConfig.adjoint_damping (with --ice-mode exact/dynamics)")
    ap.add_argument("--ice-kappa", type=float, default=None, help="IceConfig.adjoint_kappa")
    ap.add_argument("--ice-alpha-kappa", type=float, default=None,
                    help="IceConfig.adjoint_alpha_kappa (frozen_sweep κ0)")
    ap.add_argument("--ice-theta", type=float, default=None,
                    help="IceConfig.adjoint_theta (frozen_sweep softening θ)")
    ap.add_argument("--ice-thin-mass", type=float, default=None,
                    help="IceConfig.adjoint_thin_mass [kg/m²] (mEVP load derivative cut on thin "
                         "ice; 0 = off)")
    ap.add_argument("--ice-sweep-guard", type=float, default=None,
                    help="IceConfig.adjoint_sweep_guard (frozen_sweep: drop an ice step's mEVP "
                         "derivative whose transposed sweeps grow past this factor; 0 = off)")
    ap.add_argument("--ice-sweeps", type=int, default=None,
                    help="IceConfig.adjoint_sweeps (-1 = all)")
    ap.add_argument("--ice-cuts", default=None,
                    help="IceConfig.adjoint_cuts, e.g. advect+ssh or fct_flux (dynamics/exact)")
    ap.add_argument("--ice-settings", default="none",
                    help="extra IceConfig fields, name=value+name=value or JSON")
    ap.add_argument("--setting", default=None,
                    help="grad/tkegrad (required): tke_diag.parse_setting spec")
    ap.add_argument("--tke-settings", default=None,
                    help="TkeConfig forward options, name=value+name=value (changes the model)")
    ap.add_argument("--restart", default=None,
                    help="tkegrad: the start restart (directory or tag; default: the tag of --start)")
    ap.add_argument("--devices", type=int, default=1,
                    help="tkegrad: > 1 shards every chunk over that many devices (MLD costs, scalar controls)")
    ap.add_argument("--fast", default=None,
                    help="tkegrad: pieces of the fast configuration's physics, e.g. ice2+se "
                         "(core2_adj.FAST_PHYSICS; changes the model; the FD runs: tke_cbar.py fwd --fast)")
    ap.add_argument("--cost", default="T", choices=T.GRAD_COSTS, help="tkegrad cost")
    ap.add_argument("--mask", default=None, help="tkegrad MLDm: npz with the union mask")
    ap.add_argument("--window-days", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--host-boundaries", action="store_true",
                    help="grad/tkegrad: load the used base boundaries to the host first instead "
                         "of reading each from disk when its span is rebuilt")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--fd-root", default=None)
    ap.add_argument("--grad-root", default=None)
    ap.add_argument("--tag", default=None, help="grad: run dir = <grad-root>/<tag>/drift_w<d>")
    ap.add_argument("--dump-cot", default=None,
                    help="grad/tkegrad: chunk boundaries (e.g. 29-34) whose carry cotangent "
                         "goes to <run dir>/dump/cot_<c>.npz")
    ap.add_argument("--dump-primal", action="store_true",
                    help="grad/tkegrad: also the forward carry at those boundaries "
                         "(primal_<c>.npz)")
    ap.add_argument("--dump-fields", default=None,
                    help="grad/tkegrad: carry fields to dump, comma list or 'all' (default: "
                         + ",".join(DUMP_FIELDS) + ")")
    ap.add_argument("--grad-trace", action="store_true",
                    help="tkegrad: the accumulated control gradient after every reverse chunk "
                         "to <run dir>/grad_trace.jsonl")
    ap.add_argument("--field-controls", default=None,
                    help="tkegrad: TKE constants made per-node fields, comma list of "
                         + ",".join(FIELD_CONTROLS) + " (gradient maps to grad_maps.npz, "
                         "regional sums in the grad trace)")
    ap.add_argument("--windows", default=",".join(f"{d:g}" for d in WINDOW_DAYS),
                    help="summary: windows in days")
    ap.add_argument("--budget-hours", type=float, default=0.42,
                    help="wall-clock budget from the process start (30-min jobs)")
    ap.add_argument("--out", default=None, help="summary: JSON path (markdown next to it)")
    T.add_schedule_args(ap)
    args = ap.parse_args(argv)
    if ((C.parse_fast(args.fast) or args.restart or args.devices != 1 or args.forcing_controls or args.cost_from
         or args.cost_to) and args.action != "tkegrad"):
        ap.error("--fast / --restart / --devices / --forcing-controls / --cost-from / --cost-to are for tkegrad "
                 "(the FD forwards: tke_cbar.py fwd ...)")
    if args.devices != 1 and (args.field_controls or args.cost not in ("MLD", "MLDm", "FRAM")):
        ap.error("--devices > 1: MLD / FRAM costs and scalar or forcing controls only (no --field-controls)")
    ice_adj = C._ice_adjoint_args(args)
    job = os.environ.get("SLURM_JOB_ID") or datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    budget = args.budget_hours * 3600.0
    hs = [float(x) for x in args.hs.split(",") if x]

    if args.action == "noise":
        windows = T.window_steps([float(x) for x in args.windows.split(",") if x])
        out = out_path(args.out)
        res = noise_floor(Path(args.base_dir), Path(args.run_dir), windows)
        res["provenance"] = {"git_head": git_head(), "slurm_job": job,
                             "base": args.base_dir, "repeat": args.run_dir,
                             "time": datetime.datetime.now().isoformat(timespec="seconds")}
        for d in (args.base_dir, args.run_dir):
            rows = [json.loads(x) for x in (Path(d) / "results.jsonl").read_text().splitlines()
                    if x.strip()] if (Path(d) / "results.jsonl").exists() else []
            res["provenance"][f"runs_{Path(d).name}"] = [
                {k: r.get(k) for k in ("slurm_job", "device", "git_head", "status",
                                       "chunks_from", "chunks_to")} for r in rows]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(C._jsonable(res), indent=1))
        print(json.dumps(C._jsonable(res), indent=1))
        return 0

    if args.action == "summary":
        windows = T.window_steps([float(x) for x in args.windows.split(",") if x])
        out = out_path(args.out)
        table = summarize(Path(args.fd_root), Path(args.grad_root) if args.grad_root else None,
                          windows=windows, hs=hs, chunk_steps=args.chunk_steps)
        table["provenance"] = {"git_head": git_head(), "slurm_job": job,
                               "fd_root": args.fd_root, "grad_root": args.grad_root,
                               "time": datetime.datetime.now().isoformat(timespec="seconds")}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(C._jsonable(table), indent=1))
        out.with_suffix(".md").write_text(markdown(table))
        print(markdown(table))
        return 0

    if args.action in ("grad", "tkegrad") and args.setting is None:
        ap.error(f"{args.action} needs --setting (the TKE derivative setting; 'none' allowed)")
    controls = parse_controls(args.controls if args.controls is not None
                              else ("none" if args.action == "tkegrad" else "pstar,cd"))

    if args.action in ("fwd", "fdbatch"):
        n = int(round(args.days * 86400.0 / C.DT))
        model = build(args.start, n, args.chunk_steps, ice_mode=args.ice_mode,
                      ice_settings=args.ice_settings, controls=controls,
                      ice_adjoint=ice_adj, box=args.box)
        if args.action == "fwd":
            d = out_path(args.run_dir)
            check = args.check_base or (args.replay if (args.replay and (
                args.control == "none" or args.h == 0.0)) else None)
            row = _fwd_one(model, d, control=args.control, h=args.h, sign=args.sign, n=n,
                           replay=Path(args.replay) if args.replay else None,
                           save_boundaries=args.save_boundaries, save_ice=args.save_ice,
                           budget=budget, job=job, base_for_check=check,
                           save_every=args.save_every)
            return 0 if row["status"] == "complete" else 3
        fd_root = out_path(args.fd_root)
        rc = 0
        for mode in [m for m in args.modes.split(",") if m]:
            if mode not in FD_MODES:
                raise ValueError(f"unknown FD mode {mode!r}: {FD_MODES}")
            for c in controls:
                for h in hs:
                    for sign in (1, -1):
                        d = fd_run_dir(fd_root, mode, c, h, sign)
                        check_box(d, model.drift_box, "FD run")
                        if _complete(d):
                            print(f"skip (complete): {d}", flush=True)
                            continue
                        if T.LegClock(budget).out_of_time():
                            print("out of budget; rerun to continue", flush=True)
                            return 3
                        row = _fwd_one(model, d, control=c, h=h, sign=sign, n=n,
                                       replay=Path(args.base_dir) if mode == "replay" else None,
                                       save_boundaries=False, save_ice=False, budget=budget,
                                       job=job)
                        if row["status"] != "complete":
                            rc = 3
        return rc

    import tke_diag as D
    cfg = D.parse_setting(args.setting)
    n = int(round(args.window_days * 86400.0 / C.fast_dt(args.fast)))
    if args.action == "grad":
        if args.run_dir:
            run_dir = out_path(args.run_dir)
        elif args.grad_root and args.tag:
            run_dir = out_path(grad_dir(Path(args.grad_root), args.tag, args.window_days))
        else:
            ap.error("grad needs --run-dir or --grad-root and --tag")
        done = run_dir / "done.json"
        if done.exists():
            print(f"already done: {done}")
            return 0
        dumps, fields = parse_boundaries(args.dump_cot), parse_fields(args.dump_fields)
        if args.dump_primal and not dumps:
            ap.error("--dump-primal needs --dump-cot boundaries")
        model = build(args.start, n, args.chunk_steps, adjoint_cfg=cfg, ice_mode=args.ice_mode,
                      ice_settings=args.ice_settings, controls=controls,
                      ice_adjoint=ice_adj, box=args.box)
        if dumps and dumps[-1] > model.chunks_for(n):
            ap.error(f"--dump-cot boundary {dumps[-1]} beyond the window's "
                     f"{model.chunks_for(n)} chunks")
        if args.dump_primal:
            check_box(Path(args.base_dir), model.drift_box, "base")
            dump_primal(model, Path(args.base_dir), run_dir / "dump", dumps, fields)
        r = run_grad(model, run_dir, Path(args.base_dir), n_steps=n, budget_s=budget,
                     stride=args.stride, lazy=not args.host_boundaries,
                     on_cot=dump_cot_hook(run_dir / "dump", dumps, fields) if dumps else None)
        r.update(_ice_prov(model))
        r.update(setting=args.setting, adjoint=cfg.describe(), window_days=args.window_days,
                 slurm_job=job, base_dir=args.base_dir, cost="drift")
        if dumps:
            r.update(dump_cot=list(dumps), dump_primal=bool(args.dump_primal),
                     dump_fields=None if fields is None else list(fields))
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        C.append_row(run_dir / "results.jsonl", row)
        if r.get("status") == "ok":
            done.write_text(json.dumps(C._jsonable(row)))
        print(json.dumps(C._jsonable(row))[:3000])
        return 0 if r.get("status") == "ok" else 3

    # tkegrad: tke_cbar's costs on a model with this ice mode/settings, tke_cbar's layout
    run_dir = out_path(args.run_dir)
    done = run_dir / "done.json"
    if done.exists():
        print(f"already done: {done}")
        return 0
    dumps, fields = parse_boundaries(args.dump_cot), parse_fields(args.dump_fields)
    if args.dump_primal and not dumps:
        ap.error("--dump-primal needs --dump-cot boundaries")
    model = build(args.start, n, args.chunk_steps, adjoint_cfg=cfg, ice_mode=args.ice_mode,
                  ice_settings=args.ice_settings, controls=controls,
                  accs=T.accumulators_for(args.cost, args.devices),
                  ice_adjoint=ice_adj, tke_settings=args.tke_settings, fast=args.fast,
                  restart=args.restart, devices=args.devices, **T.schedule_kwargs(args))
    fields_ctl = [x for x in (args.field_controls or "").split(",") if x]
    bad = [x for x in fields_ctl if x not in FIELD_CONTROLS]
    if bad:
        ap.error(f"unknown --field-controls {bad}; known: {list(FIELD_CONTROLS)}")
    if fields_ctl:                  # ln p = 0 on every node: the same Params values, per node
        u = dict(model.params0)
        for x in fields_ctl:
            u[FIELD_CONTROLS[x]] = np.zeros((int(model.mesh.nod2D), 1), np.float64)
        model.params0 = C._strong_f64(u)
    if dumps and dumps[-1] > model.chunks_for(n):
        ap.error(f"--dump-cot boundary {dumps[-1]} beyond the window's "
                 f"{model.chunks_for(n)} chunks")
    if args.dump_primal:
        dump_primal(model, Path(args.base_dir), run_dir / "dump", dumps, fields)
    mask = None
    if args.mask:
        with np.load(args.mask) as z:
            mask = np.asarray(z[f"mask_{n}"], bool)
    regions = trace_regions(model.mesh) if fields_ctl else None
    r = T.run_cbar_grad(model, run_dir, Path(args.base_dir), cost=args.cost, n_steps=n,
                        mask=mask, budget_s=budget, stride=args.stride,
                        lazy=not args.host_boundaries,
                        on_cot=(dump_cot_hook(run_dir / "dump", dumps, fields, global_carry(model))
                                if dumps else None),
                        on_chunk=(grad_trace_hook(run_dir / "grad_trace.jsonl", regions)
                                  if args.grad_trace else None))
    if fields_ctl and r.get("status") == "ok":
        g = r["_result"]["grad"]
        np.savez(run_dir / "grad_maps.npz",
                 **{FIELD_CONTROLS[x]: np.asarray(g[FIELD_CONTROLS[x]]) for x in fields_ctl})
        r["grad_regions"] = {
            FIELD_CONTROLS[x]: {k: float(np.asarray(g[FIELD_CONTROLS[x]]).reshape(-1)[m].sum())
                                for k, m in regions.items()} for x in fields_ctl}
        r["field_controls"] = fields_ctl
    if "forcing_adj" in model.params0 and r.get("status") == "ok":   # the monthly maps [M, nod2D] per field
        gf = r["_result"]["grad"]["forcing_adj"]
        np.savez(run_dir / "grad_forcing.npz", **{k: np.asarray(v) for k, v in gf.items()})
        r["grad_forcing_sum"] = {k: float(np.asarray(v).sum()) for k, v in gf.items()}
    if controls and r.get("status") == "ok":                        # the ice log controls (fd_dirs reads them)
        g = r["_result"]["grad"]
        r["grad_ice"] = {ICE_CONTROLS[c][0]: float(np.asarray(g[ICE_CONTROLS[c][0]])) for c in controls}
    if args.cost == "FRAM":
        r["n_cost"] = T.cost_steps(model, n)
    r.update(_ice_prov(model))
    r.update(action="ice_tkegrad", setting=args.setting, adjoint=cfg.describe(), window_days=args.window_days,
             slurm_job=job, mask=args.mask, cost=args.cost,
             n_masked=None if mask is None else int(mask.sum()))
    if dumps:
        r.update(dump_cot=list(dumps), dump_primal=bool(args.dump_primal),
                 dump_fields=None if fields is None else list(fields))
    row = {k: v for k, v in r.items() if not k.startswith("_")}
    C.append_row(run_dir / "results.jsonl", row)
    if r.get("status") == "ok":
        done.write_text(json.dumps(C._jsonable(row)))
    print(json.dumps(C._jsonable(row))[:3000])
    return 0 if r.get("status") == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
