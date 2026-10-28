#!/usr/bin/env python
"""TKE-constant gradients against finite differences on the CORE2 reference model, in resumable
pieces (CPU): the "TKE-constant bar" of the classical-TKE adjoint work.

    python scripts/adjoint/tke_cbar.py fwd  --const c_k --h 1e-2 --sign +1 --run-dir D
    python scripts/adjoint/tke_cbar.py fwd  --const none --save-boundaries --run-dir BASE
    python scripts/adjoint/tke_cbar.py grad --setting tke_pr_cut=inf --cost T --window-days 5 \\
                                            --base-dir BASE --run-dir G
    python scripts/adjoint/tke_cbar.py summary --fd-root F --grad-root G --out table.json

Run from a checkout with ``PYTHONPATH=<checkout>`` (``scripts/adjoint/job_cpu.sbatch``).

The functionals (:data:`J_NAMES`), on the driver's default box (``natl_subpolar``: 50-60N,
40-20W, 0-500 m), for a window of ``n`` steps that starts at the window start:

* ``J_T`` — the live-volume-weighted box-mean temperature, averaged over the last
  ``min(T_MEAN_STEPS, n)`` steps of the window (5 days: :data:`T_MEAN_STEPS` = 240 steps of
  1800 s), from the running sum ``S(n)`` of the per-step values:
  ``J_T = (S(n) - S(n - m)) / m``, ``m = min(T_MEAN_STEPS, n)``;
* ``J_Tfull`` — the same box-mean temperature averaged over the whole window, ``S(n) / n``
  (the long-window functional: one sweep for any ``n``);
* ``J_MLD`` — the area-weighted box mean of the window-mean mixed-layer depth
  (``mld_field(invalid="column")``, a per-step running sum ``M(n)``): ``J_MLD = <M(n)/n>``.

Every window starts at the same instant, so ONE forward (the base run, ``--const none
--save-boundaries``) with its chunk boundaries on disk serves every window and every adjoint
setting: an adjoint run links the base boundaries ``0..n/K`` into its own directory and runs the
reverse sweep only (:class:`core2_adj.DiskCheckpoint`: resumable under a wall-clock budget;
``--stride N`` uses every Nth boundary and recomputes between, reading each from disk when its
span is rebuilt; ``--host-boundaries`` loads them to the host first).
The forward is bitwise the same for every derivative-only setting on CPU, so the linearisation
point is shared exactly. For a window longer than :data:`T_MEAN_STEPS` the adjoint of ``J_T`` is
assembled from two sweeps: ``J_T(n) = 2·Tfull(n) - J_T(n/2)`` when ``n = 2·T_MEAN_STEPS``
(``Tfull`` = the mean over the whole window; ``grad`` cost ``Tfull``).

Finite differences: one forward per (constant, h, sign) with ``ln p = ±h`` (the other
constants at their defaults), recording ``S`` at every chunk boundary and the per-node ``M`` at
the window ends, so a single 10-day forward gives every window. ``summary`` forms the central
differences, the MLD jump masks (a node jumps in the h-sweep when its second difference exceeds
half its central difference, as :func:`jumping_nodes`), the masked FD (jumping nodes held at the
base value) and the comparison with the adjoint (``MLDm`` cost: the same nodes held fixed).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core2_adj as C  # noqa: E402  (runs the import guard)

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fesom_jax import adjoint  # noqa: E402
from fesom_jax import adjoint_costs as AC  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402

T_MEAN_STEPS = 240                      #: the J_T averaging period: 5 days of 1800-s steps
CONSTS = ("c_k", "c_eps", "cd", "alpha")
PARAM = {"c_k": "ln_tke_c_k", "c_eps": "ln_tke_c_eps", "cd": "ln_tke_cd",
         "alpha": "ln_tke_alpha"}
J_NAMES = ("J_T", "J_Tfull", "J_MLD")
GRAD_COSTS = ("T", "Tfull", "MLD", "MLDm", "FRAM")
JUMP_RATIO = 0.5                        #: second difference > ratio · central difference
JUMP_ATOL = 1e-6                        #: m; smaller second differences are never a jump
BAR_RTOL = 0.10                         #: the bar: adjoint within 10 % of FD
PLATEAU_RTOL = 0.02                     #: consecutive FD values agreeing to this form a plateau
RUNS_DIR = Path("/work/ab0995/a270088/fesom_jax_dev/runs/tke/cbar")
_T_START = time.time()                  #: process start: a leg's budget counts from here
CHUNK_MARGIN = 1.15                     #: the next chunk is assumed this much longer


class LegClock:
    """A leg's wall-clock budget counted from the process start: after each chunk it answers
    "would the NEXT chunk still end in time?", with the last chunk's duration (the first
    includes loading and compiling, so it errs on the safe side) times :data:`CHUNK_MARGIN`.
    A leg therefore never starts a chunk that the job's time limit would kill half-way (a
    killed chunk is lost and a rerun would repeat it)."""

    def __init__(self, budget_s: float):
        self.deadline = _T_START + float(budget_s)
        self.last = time.time()

    def out_of_time(self) -> bool:
        now = time.time()
        dt, self.last = now - self.last, now
        return now + CHUNK_MARGIN * dt > self.deadline


class LegCheckpoint(C.DiskCheckpoint):
    """:class:`core2_adj.DiskCheckpoint` with the :class:`LegClock` budget. With ``stride`` > 1
    only every ``stride``-th linked boundary (and the last) is used; the reverse sweep recomputes
    the spans between (``chunked_grad(boundary_stride=stride)``). ``lazy`` (default) reads each
    of those from disk when its span is rebuilt (:class:`core2_adj.DiskBoundaries`), so the host
    holds one span of ``stride`` states; ``lazy=False`` (``--host-boundaries``) loads them all to
    the host first. A year at stride 24 then holds 31 + 24 states, ~115 GB: a GH200's own CPU
    memory is ~120 GB, and three such sweeps on one node stalled on page reclaim (2026-10-11)."""

    def __init__(self, directory, params_like, *, budget_s: float = math.inf, stride: int = 1,
                 lazy: bool = True):
        super().__init__(directory, params_like, budget_s=budget_s, lazy=lazy)
        self.clock = LegClock(budget_s)
        self.stride = max(1, int(stride))

    def _out_of_time(self) -> bool:
        return self.clock.out_of_time()

    def load_boundaries(self, like) -> dict:
        saved = self._saved()
        keep = [c for c in saved if c % self.stride == 0 or c == saved[-1]]
        return self._boundaries(keep, like)


def mld_accumulators(mesh):
    """``{"mld": per-node MLD}`` per step: the running sums of a sharded model (``--devices``), which sums per-node
    fields only (no ``boxT``, so no J_T / J_Tfull costs)."""
    return {"mld": C.ACCUMULATORS["mld"](mesh)}


def fram_accumulators(mesh):
    """``{"fram": per-node Fram Strait ice export}`` per step (``adjoint_costs.ice_export_field``, m3/s per node): the
    running sum of the cost ``FRAM``, on a dense or a sharded model."""
    return {"fram": C.ACCUMULATORS["fram"](mesh)}


def accumulators_for(cost: str, devices: int = 1):
    """The accumulator factory a model needs for ``cost``: :func:`fram_accumulators` for ``FRAM``, else
    :func:`accumulators` (dense) or :func:`mld_accumulators` (sharded)."""
    if cost == "FRAM":
        return fram_accumulators
    return accumulators if int(devices) == 1 else mld_accumulators


def cost_steps(model, n_steps: int) -> int:
    """The number of steps the accumulators sum in an ``n_steps`` window: the schedule's ``[cost_from, cost_to)``
    cut to the window, else ``n_steps``."""
    sch = getattr(model, "schedule", None) or {}
    lo = int(sch.get("cost_from") or 0)
    hi = n_steps if sch.get("cost_to") is None else min(int(sch["cost_to"]), n_steps)
    if hi <= lo:
        raise ValueError(f"the cost interval [{lo}, {hi}) of an {n_steps}-step window is empty")
    return hi - lo


def accumulators(mesh, box=C.DEFAULT_BOX):
    """``{"mld": per-node MLD, "boxT": box-mean T}`` per step (the running sums of the carry)."""
    node, level = C.box_masks(mesh, box)
    node_j, level_j = jnp.asarray(node), jnp.asarray(level)
    return {"mld": C.ACCUMULATORS["mld"](mesh),
            "boxT": lambda s: AC.box_mean_T_vw(s, mesh, node_j, level_j)}


def surface_weights(mesh, box=C.DEFAULT_BOX) -> np.ndarray:
    """Area weights of the wet surface box nodes (the MLD box mean)."""
    node, _ = C.box_masks(mesh, box)
    wet = np.asarray(mesh.node_layer_mask[:, 0]) & np.asarray(node, bool)
    return np.where(wet, np.asarray(mesh.areasvol[:, 0]), 0.0)


def j_mld(mld_sum, n: int, w: np.ndarray, mask=None, ref_sum=None) -> float:
    """``J_MLD`` from the per-node running sum; ``mask`` nodes take ``ref_sum`` instead."""
    s = np.asarray(mld_sum, np.float64)
    if mask is not None:
        s = np.where(np.asarray(mask, bool), np.asarray(ref_sum, np.float64), s)
    return float(np.sum(w * s / n) / np.sum(w))


def j_t(boxT_sums: dict, n: int) -> float:
    """``J_T`` of an ``n``-step window from ``{steps: S(steps)}`` (``S(0) = 0``)."""
    m = min(T_MEAN_STEPS, n)
    return float((boxT_sums[n] - boxT_sums.get(n - m, 0.0)) / m)


def j_tfull(boxT_sums: dict, n: int) -> float:
    """``J_Tfull`` of an ``n``-step window: the mean over all its steps."""
    return float(boxT_sums[n] / n)


def cost_fn(name: str, n: int, mesh, box=C.DEFAULT_BOX, mask=None, fold=None):
    """The adjoint cost ``carry -> scalar`` for an ``n``-step window: ``T`` (= ``J_T`` for
    ``n <= T_MEAN_STEPS``), ``Tfull`` (the mean of every step), ``MLD`` (= ``J_MLD``), ``MLDm``
    (``J_MLD`` with the ``mask`` nodes' sums held fixed: value unchanged, no derivative), ``FRAM`` (the mean Fram
    Strait ice volume export over the ``n`` summed steps, km3/yr; ``n`` = :func:`cost_steps`). ``fold`` (a sharded
    model's ``shard.fold_nodes``) puts the node weights and the mask in the carry's folded layout, zero off the
    owned lanes."""
    fold = (lambda v: v) if fold is None else fold
    if name == "T":
        if n > T_MEAN_STEPS:
            raise ValueError(f"cost T is J_T only up to {T_MEAN_STEPS} steps; use Tfull "
                             f"and combine (J_T(2m) = 2·Tfull(2m) - J_T(m))")
        return lambda carry: carry[1]["boxT"] / n
    if name == "Tfull":
        return lambda carry: carry[1]["boxT"] / n
    if name == "FRAM":                          # the owned lanes once: a halo lane is another device's node
        own = jnp.asarray(fold(np.ones(int(mesh.nod2D))))
        return lambda carry: jnp.sum(own * carry[1]["fram"]) / n * AC.M3S_TO_KM3YR
    w = jnp.asarray(fold(np.asarray(surface_weights(mesh, box))))
    if name == "MLD":
        return lambda carry: jnp.sum(w * carry[1]["mld"] / n) / jnp.sum(w)
    if name == "MLDm":
        if mask is None:
            raise ValueError("cost MLDm needs a mask")
        m = jnp.asarray(fold(np.asarray(mask, bool)))

        def loss(carry):
            s = carry[1]["mld"]
            s = jnp.where(m, jax.lax.stop_gradient(s), s)
            return jnp.sum(w * s / n) / jnp.sum(w)
        return loss
    raise ValueError(f"unknown cost {name!r}: {GRAD_COSTS}")


def build(args, cfg: AdjointConfig, n_steps: int):
    devices = int(getattr(args, "devices", 1) or 1)
    return C.build_model(args.start, n_steps, args.chunk_steps, getattr(args, "restart", None), cfg, args.ice_mode,
                         accumulators=accumulators_for(getattr(args, "cost", "T"), devices), rematerialize=True,
                         tke_settings=getattr(args, "tke_settings", None), fast=getattr(args, "fast", None),
                         devices=devices, **schedule_kwargs(args))


def schedule_kwargs(args) -> dict:
    """``build_model``'s forcing controls and cost interval from the tools' flags (``--forcing-controls``,
    ``--forcing-months``, ``--cost-from`` / ``--cost-to`` dates, in steps of the window's timestep); absent ⇒ {}."""
    kw = {}
    ctrl = [x for x in (getattr(args, "forcing_controls", None) or "").split(",") if x]
    if ctrl:
        kw.update(forcing_ctrl=tuple(ctrl), n_months=int(args.forcing_months))
    dt = C.fast_dt(getattr(args, "fast", None))
    start = datetime.date.fromisoformat(args.start)
    for flag in ("cost_from", "cost_to"):
        d = getattr(args, flag, None)
        if d:
            days = (datetime.date.fromisoformat(d) - start).days
            kw[flag] = int(round(days * 86400.0 / dt))
    return kw


def add_schedule_args(ap) -> None:
    """The flags of :func:`schedule_kwargs`."""
    ap.add_argument("--forcing-controls", default=None,
                    help="monthly atmospheric anomaly maps as controls, comma list of step.FORCING_ADJ_KEYS")
    ap.add_argument("--forcing-months", type=int, default=0, help="the number of window months of those maps")
    ap.add_argument("--cost-from", default=None, help="the first date the accumulators sum (YYYY-MM-DD)")
    ap.add_argument("--cost-to", default=None, help="the first date they no longer sum (YYYY-MM-DD)")


def perturbed_params(model, const: str, h: float, sign: int):
    u = {k: np.array(v) for k, v in model.params0.items()}
    if const != "none":
        u[PARAM[const]] = np.asarray(sign * h, np.float64)
    return C._strong_f64(u)


def window_steps(days, dt: float = C.DT) -> list:
    return [int(round(d * 86400.0 / dt)) for d in days]


# ---------------------------------------------------------------------------
# fwd: one forward per (constant, h, sign); the base run also keeps its boundaries


def _host_save(path: Path, tree) -> None:
    if path.exists():
        return
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, *[np.asarray(x) for x in jax.tree.leaves(tree)])
    tmp.rename(path)


def _host_load(path: Path, like):
    with np.load(path) as z:
        leaves = [z[f"arr_{i}"] for i in range(len(z.files))]
    return jax.tree.unflatten(jax.tree.structure(like), leaves)


def run_fwd(model, run_dir: Path, *, const: str, h: float, sign: int, n_steps: int,
            ends: list, save_boundaries: bool, budget_s: float, save_every: int = 1, params=None,
            state=None) -> dict:
    """Advance chunk by chunk, appending ``S(steps)`` to ``fwd.jsonl`` at every boundary and
    writing ``mld_<steps>.npz`` at the window ends; with ``save_boundaries`` every
    ``save_every``-th boundary and the last are ``boundary_<c>.npz`` (the
    :class:`core2_adj.DiskCheckpoint` layout; a grad over it needs a ``stride`` that is a
    multiple of ``save_every``). Out of budget: the carry goes to ``carry_<c>.npz`` unless a
    boundary was just saved, and the call returns ``status="incomplete"``; a rerun resumes from
    the furthest saved carry or boundary. Nothing is overwritten. ``params`` / ``state`` (a global State) replace the
    perturbed constant's parameters / the model's start State (``fd_dirs``: any direction)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    K = model.chunk_steps
    nc = model.chunks_for(n_steps)
    params = perturbed_params(model, const, h, sign) if params is None else params
    like = model.init_carry(host=True)
    saved = [(int(p.stem.split("_")[1]), p) for pat in ("carry_*.npz", "boundary_*.npz")
             for p in run_dir.glob(pat)]
    c0, carry = 0, None
    if saved:
        c0, p = max(saved)
        carry = jax.device_put(_host_load(p, like))
    else:
        carry = model.init_carry(state)
        if save_boundaries:
            _host_save(run_dir / "boundary_0.npz", jax.device_get(carry))
    fwd = adjoint._jitted_forward(model.chunk_fn)
    xs_fn = model.xs_fn()
    clock = LegClock(budget_s)
    t0, tc = time.time(), None
    status = "complete"
    with C._Timer() as timer:
        for c in range(c0, nc):
            xs = xs_fn(c)
            nxt = jax.block_until_ready(fwd(carry, xs, params))
            adjoint._free(xs)
            adjoint._free(carry)
            carry = nxt
            if tc is None:
                tc = time.time() - t0            # the first chunk includes the compile
            steps = (c + 1) * K
            acc = jax.device_get(carry[1])       # the accumulators; the State comes to the host only to be saved
            row = {"chunk": c + 1, "steps": steps,
                   "boxT_sum": float(acc["boxT"]) if "boxT" in acc else None, "wall_s": time.time() - t0}
            if "fram" in acc:                    # the running export sum [m3/s x steps] over the owned nodes
                f = acc["fram"] if model.shard is None else model.shard.unfold_nodes(acc["fram"])
                row["fram_sum"] = float(np.sum(f))
            with open(run_dir / "fwd.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if steps in ends:
                for key in ("mld", "fram"):
                    if key in acc:
                        v = acc[key] if model.shard is None else model.shard.unfold_nodes(acc[key])
                        _host_save(run_dir / f"{key}_{steps}.npz", {f"{key}_sum": v})
            kept = save_boundaries and ((c + 1) % save_every == 0 or c + 1 == nc)
            if kept:
                _host_save(run_dir / f"boundary_{c + 1}.npz", jax.device_get(carry))
            if clock.out_of_time() and c + 1 < nc:
                if not kept:
                    _host_save(run_dir / f"carry_{c + 1}.npz", jax.device_get(carry))
                status = "incomplete"
                break
    n_done = (c + 1 - c0) if nc > c0 else 0
    per_step = ((time.time() - t0 - (tc or 0.0)) / max(1, (n_done - 1) * K)
                if n_done > 1 else float("nan"))
    return C.provenance(model, "cbar_fwd", n_steps=n_steps, timer=timer, const=const, h=h,
                        sign=sign, status=status, chunks_from=c0, chunks_to=c + 1,
                        first_chunk_s=tc, s_per_step=per_step, run_dir=str(run_dir))


def fwd_records(run_dir: Path) -> tuple[dict, dict]:
    """``({steps: S(steps)}, {steps: mld_sum})`` of a finished fwd run (S(0) = 0)."""
    S = {0: 0.0}
    with open(run_dir / "fwd.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("boxT_sum") is not None:            # None: a sharded run (no boxT sum)
                S[int(r["steps"])] = float(r["boxT_sum"])
    M = {}
    for p in run_dir.glob("mld_*.npz"):
        with np.load(p) as z:
            M[int(p.stem.split("_")[1])] = np.asarray(z["arr_0"])
    return S, M


# ---------------------------------------------------------------------------
# grad: the reverse sweep over the shared base boundaries


def link_base(base_dir: Path, run_dir: Path, n_chunks: int, stride: int = 1) -> None:
    """Symlink the base run's boundaries ``0..n_chunks`` (every ``stride``-th and the last) into
    ``run_dir`` (never replacing)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for c in range(n_chunks + 1):
        if c % stride and c != n_chunks:
            continue
        src = base_dir / f"boundary_{c}.npz"
        if not src.exists():
            raise FileNotFoundError(f"base boundary missing: {src}")
        dst = run_dir / f"boundary_{c}.npz"
        if not dst.exists():
            dst.symlink_to(src)


def check_base_physics(model, base_dir: Path) -> None:
    """Refuse a base run made with other ``TkeConfig`` forward options, other fast-physics pieces or from another
    restart: its boundaries are another model's trajectory (rows without ``tke_settings`` / ``fast`` are the
    reference physics)."""
    rj = Path(base_dir) / "results.jsonl"
    rows = [json.loads(x) for x in rj.read_text().splitlines() if x.strip()] if rj.exists() else []
    for key, mine in (("tke_settings", dict(getattr(model, "tke_settings", {}) or {})),
                      ("fast", list(getattr(model, "fast", ()) or ()))):
        empty = {} if key == "tke_settings" else []
        want = json.dumps(mine, sort_keys=True)
        have = {json.dumps(r.get(key) or empty, sort_keys=True) for r in rows}
        if have and have != {want}:
            raise ValueError(f"base {base_dir} was run with {key} {sorted(have)}, this model has {want}")
    devs = {int(r.get("devices", 1)) for r in rows}
    if devs and devs != {int(getattr(model, "devices", 1))}:
        raise ValueError(f"base {base_dir} ran on {sorted(devs)} devices (its boundaries are in that layout), this "
                         f"model on {getattr(model, 'devices', 1)}")
    starts = {r["restart"] for r in rows if r.get("restart")}
    if starts and starts != {str(model.restart)}:
        raise ValueError(f"base {base_dir} started from {sorted(starts)}, this model from {model.restart}")


def run_cbar_grad(model, run_dir: Path, base_dir: Path, *, cost: str, n_steps: int,
                  mask=None, budget_s: float, stride: int = 1, lazy: bool = True,
                  on_cot=None, on_chunk=None) -> dict:
    nc = model.chunks_for(n_steps)
    check_base_physics(model, base_dir)
    link_base(base_dir, run_dir, nc, stride)
    loss = cost_fn(cost, cost_steps(model, n_steps) if cost == "FRAM" else n_steps, model.mesh, mask=mask,
                   fold=None if model.shard is None else model.shard.fold_nodes)
    fj = run_dir / "forward.json"
    if not fj.exists():                       # the loss at the window end, from the base run
        last = jax.device_put(_host_load(run_dir / f"boundary_{nc}.npz",
                                         model.init_carry(host=True)))
        fj.write_text(json.dumps({"loss": float(loss(last))}))
        adjoint._free(last)
    rows = C.run_grad(model, (loss,), n_steps=n_steps, stride=stride, check_linear=False,
                      resume_dir=run_dir, budget_s=budget_s, action="cbar_grad", on_cot=on_cot,
                      on_chunk=on_chunk,
                      checkpoint_factory=lambda d, p, b: LegCheckpoint(d, p, budget_s=b,
                                                                       stride=stride, lazy=lazy))
    r = rows[0]
    r["cost"] = cost
    if r.get("status") == "ok":
        g = r["_result"]["grad"]
        # a per-node control (ice_cbar tkegrad --field-controls) sums to the scalar gradient
        r["grads"] = {c: float(np.asarray(g[PARAM[c]]).sum()) for c in CONSTS}
        r["grad_k_gm"] = float(np.asarray(g["ln_k_gm"]))
    return r


# ---------------------------------------------------------------------------
# summary


def jumping_nodes(m_minus, m0, m_plus, *, ratio=JUMP_RATIO, atol=JUMP_ATOL) -> np.ndarray:
    d1 = np.abs(m_plus - m_minus)
    d2 = np.abs(m_plus - 2.0 * m0 + m_minus)
    return d2 > np.maximum(ratio * d1, atol)


def plateau(hs, fds, rtol=PLATEAU_RTOL, nmin=3):
    """The longest run of >= ``nmin`` consecutive h (sorted) whose FD values agree with their
    median to ``rtol``: ``{"h": [...], "fd": median}`` or ``None``."""
    order = np.argsort(hs)
    hs, fds = np.asarray(hs)[order], np.asarray(fds)[order]
    best = None
    for i in range(len(hs)):
        for j in range(i + nmin, len(hs) + 1):
            med = float(np.median(fds[i:j]))
            if np.all(np.abs(fds[i:j] - med) <= rtol * max(abs(med), 1e-300)):
                if best is None or j - i > len(best["h"]):
                    best = {"h": hs[i:j].tolist(), "fd": med}
    return best


def bar_verdict(ad: float, fd: dict, days: float) -> dict:
    """The TKE-constant bar for one adjoint value against the central FDs ``{h: fd}``: the
    reference FD is the plateau median (:func:`plateau`) or, without a plateau, the FD at the
    middle step size; ``rel_err`` = |ad − ref|/|ref|; the FD spread = max |fd − ref|/|ref| over
    h. The bar (windows of 5 days and more; ``None`` below) passes when ``rel_err <=``
    :data:`BAR_RTOL`, or when ``ad`` lies inside the range of the FDs over h (the FD itself
    limited by switches)."""
    hh = sorted(fd)
    vals = [fd[h] for h in hh]
    pl = plateau(hh, vals)
    ref = pl["fd"] if pl else vals[len(vals) // 2]
    rel = abs(ad - ref) / max(abs(ref), 1e-300)
    lo, hi = min(vals), max(vals)
    spread = max(abs(v - ref) for v in vals) / max(abs(ref), 1e-300)
    bar = None
    if days >= 5.0:
        bar = bool(rel <= BAR_RTOL or lo <= ad <= hi)
    return {"plateau": pl, "fd_ref": ref, "rel_err": rel, "fd_range": [lo, hi],
            "fd_spread": spread, "bar": bar}


def fd_table(fd_root: Path, mesh, windows: list, hs: list) -> dict:
    """``{window_steps: {const: {"J_T": {h: fd}, "J_Tfull": {h: fd}, "J_MLD": {h: fd},
    "J_MLDm": {h: fd}, "n_jump": {h: n}}}}`` plus the union masks ``{window_steps: mask}``."""
    w = surface_weights(mesh)
    S0, M0 = fwd_records(fd_root / "base")
    out, masks = {}, {}
    rec = {}
    for c in CONSTS:
        for h in hs:
            for s in (+1, -1):
                d = fd_root / f"{c}_h{h:g}_{'p' if s > 0 else 'm'}"
                if (d / "fwd.jsonl").exists():
                    rec[(c, h, s)] = fwd_records(d)
    for n in windows:
        union = np.zeros_like(M0[n], dtype=bool)
        per = {}
        for c in CONSTS:
            for h in hs:
                if (c, h, 1) not in rec or (c, h, -1) not in rec:
                    continue
                (Sp, Mp), (Sm, Mm) = rec[(c, h, 1)], rec[(c, h, -1)]
                if n not in Mp or n not in Mm or n not in Sp or n not in Sm:
                    continue
                jm = jumping_nodes(Mm[n] / n, M0[n] / n, Mp[n] / n)
                union |= jm
                per[(c, h)] = (Sp, Sm, Mp[n], Mm[n], int(jm.sum()))
        masks[n] = union
        tab = {}
        for (c, h), (Sp, Sm, Mp, Mm, nj) in per.items():
            t = tab.setdefault(c, {"J_T": {}, "J_Tfull": {}, "J_MLD": {}, "J_MLDm": {},
                                   "n_jump": {}})
            t["J_T"][h] = (j_t(Sp, n) - j_t(Sm, n)) / (2 * h)
            t["J_Tfull"][h] = (j_tfull(Sp, n) - j_tfull(Sm, n)) / (2 * h)
            t["J_MLD"][h] = (j_mld(Mp, n, w) - j_mld(Mm, n, w)) / (2 * h)
            t["J_MLDm"][h] = (j_mld(Mp, n, w, union, M0[n]) - j_mld(Mm, n, w, union, M0[n])) / (2 * h)
            t["n_jump"][h] = nj
        out[n] = tab
    return out, masks, (S0, M0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["fwd", "grad", "summary"])
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--chunk-steps", type=int, default=24)
    ap.add_argument("--days", type=float, default=10.0, help="fwd: forward length")
    ap.add_argument("--window-ends", default="0.5,1,2,5,10", help="window ends (days)")
    ap.add_argument("--const", default="none", choices=("none",) + CONSTS)
    ap.add_argument("--h", type=float, default=0.0)
    ap.add_argument("--sign", type=int, default=1, choices=(1, -1))
    ap.add_argument("--save-boundaries", action="store_true")
    ap.add_argument("--save-every", type=int, default=1, help="fwd: save every Nth boundary "
                                                              "(and the last)")
    ap.add_argument("--setting", default="none", help="grad: tke_diag.parse_setting spec, "
                                                      "long_window or long_window_tke")
    ap.add_argument("--cost", default="T", choices=GRAD_COSTS)
    ap.add_argument("--window-days", type=float, default=1.0)
    ap.add_argument("--mask", default=None, help="grad MLDm: npz with the union mask")
    ap.add_argument("--stride", type=int, default=1, help="grad: use every Nth base boundary, "
                                                          "recompute between")
    ap.add_argument("--host-boundaries", action="store_true",
                    help="grad: load the used base boundaries to the host first instead of "
                         "reading each from disk when its span is rebuilt")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--tke-settings", default=None,
                    help="TkeConfig forward options, name=value+name=value (changes the model)")
    ap.add_argument("--fast", default=None,
                    help="pieces of the fast configuration's physics, e.g. ice2+se (core2_adj.FAST_PHYSICS; "
                         "changes the model)")
    ap.add_argument("--restart", default=None,
                    help="the start restart (directory or tag; default: the tag of --start in core2_adj's dirs)")
    ap.add_argument("--devices", type=int, default=1,
                    help="> 1: every chunk sharded over that many devices (core2_adj.build_model devices; the MLD "
                         "costs only)")
    ap.add_argument("--ice-mode", default="prescribed", choices=C.ICE_MODES,
                    help="IceConfig.adjoint_mode (the forward is the same for every mode)")
    add_schedule_args(ap)
    ap.add_argument("--budget-hours", type=float, default=0.42,
                    help="a leg's wall-clock budget from the process start (30-min jobs)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fd-root", default=None, help="summary: FD runs (base, <c>_h<h>_p|m)")
    ap.add_argument("--grad-root", default=None, help="summary: <setting>/<cost>_w<days>")
    ap.add_argument("--hs", default="1e-1,3e-2,1e-2,3e-3", help="summary: FD step sizes")
    ap.add_argument("--masks-out", default=None, help="summary: write the union MLD masks")
    args = ap.parse_args(argv)
    job = os.environ.get("SLURM_JOB_ID") or datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    budget = args.budget_hours * 3600.0
    dt = C.fast_dt(args.fast)
    if args.action == "summary" and dt != C.DT:
        ap.error("summary: the bar table's windows are in dt-1800 steps (T_MEAN_STEPS); no --fast dt2700")
    ends = window_steps([float(x) for x in args.window_ends.split(",") if x], dt)
    if args.action == "fwd":
        n = int(round(args.days * 86400.0 / dt))
        model = build(args, AdjointConfig.exact(), n)
        run_dir = Path(args.run_dir)
        row = run_fwd(model, run_dir, const=args.const, h=args.h, sign=args.sign, n_steps=n,
                      ends=ends, save_boundaries=args.save_boundaries, budget_s=budget,
                      save_every=args.save_every)
        row["slurm_job"] = job
        C.append_row(run_dir / "results.jsonl", {k: v for k, v in row.items()
                                                  if not k.startswith("_")})
        print(json.dumps(C._jsonable({k: v for k, v in row.items() if not k.startswith("_")}))[:3000])
        return 0 if row["status"] == "complete" else 3
    if args.action == "grad":
        import tke_diag as D
        cfg = (AdjointConfig.long_window() if args.setting == "long_window"
               else AdjointConfig.long_window_tke() if args.setting == "long_window_tke"
               else D.parse_setting(args.setting))
        n = int(round(args.window_days * 86400.0 / dt))
        model = build(args, cfg, n)
        mask = None
        if args.mask:
            with np.load(args.mask) as z:
                mask = np.asarray(z[f"mask_{n}"], bool)
        run_dir = Path(args.run_dir)
        done = run_dir / "done.json"
        if done.exists():
            print(f"already done: {done}")
            return 0
        r = run_cbar_grad(model, run_dir, Path(args.base_dir), cost=args.cost, n_steps=n,
                          mask=mask, budget_s=budget, stride=args.stride,
                          lazy=not args.host_boundaries)
        r.update(setting=args.setting, adjoint=cfg.describe(), window_days=args.window_days,
                 slurm_job=job, mask=args.mask,
                 n_masked=None if mask is None else int(mask.sum()))
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        C.append_row(run_dir / "results.jsonl", row)
        if r.get("status") == "ok":
            done.write_text(json.dumps(C._jsonable(row)))
        print(json.dumps(C._jsonable(row))[:3000])
        return 0 if r.get("status") == "ok" else 3
    # summary
    model = build(args, AdjointConfig.exact(), max(ends))
    hs = [float(x) for x in args.hs.split(",") if x]
    table = summarize(Path(args.fd_root), Path(args.grad_root), model.mesh, ends, hs,
                      masks_out=Path(args.masks_out) if args.masks_out else None)
    out = Path(args.out) if args.out else Path(args.grad_root) / "bar_table.json"
    out.write_text(json.dumps(C._jsonable(table), indent=1))
    for r in table["rows"]:
        print(f"{r['setting']:>40s} {r['window_days']:5g} d {r['J']:7s} {r['const']:6s} "
              f"ad {r['adjoint']: .4e} fd {r['fd_ref']: .4e} rel {r['rel_err']:.3g} "
              f"{'PASS' if r['bar'] else ('fail' if r['bar'] is False else '-')}")
    return 0


def _grad_of(grad_root: Path, tag: str, cost: str, days: float):
    p = grad_root / tag / f"{cost}_w{days:g}" / "done.json"
    return json.loads(p.read_text())["grads"] if p.exists() else None


def summarize(fd_root: Path, grad_root: Path, mesh, ends: list, hs: list, *,
              masks_out: Path | None = None) -> dict:
    """The bar table: for every adjoint setting directory under ``grad_root`` (one per
    setting; ``<cost>_w<days>/done.json`` inside), window, functional and constant: the adjoint
    gradient, the central FDs over ``hs`` (full and, for J_MLD, masked), their plateau, the
    reference FD (plateau median, else the FD at the middle h), the relative error, and the bar
    (:func:`bar_verdict`)."""
    fd, masks, _ = fd_table(fd_root, mesh, ends, hs)
    if masks_out is not None:
        np.savez(masks_out, **{f"mask_{n}": m for n, m in masks.items()})
    rows = []
    tags = sorted(p.name for p in grad_root.iterdir() if p.is_dir()) if grad_root.exists() else []
    for tag in tags:
        for n in ends:
            days = n * C.DT / 86400.0
            if n <= T_MEAN_STEPS:
                gT = _grad_of(grad_root, tag, "T", days)
            else:
                gf = _grad_of(grad_root, tag, "Tfull", days)
                gh = _grad_of(grad_root, tag, "T", days / 2.0)
                gT = (None if gf is None or gh is None or n != 2 * T_MEAN_STEPS
                      else {c: 2.0 * gf[c] - gh[c] for c in CONSTS})
            ad = {"J_T": gT, "J_Tfull": _grad_of(grad_root, tag, "Tfull", days),
                  "J_MLD": _grad_of(grad_root, tag, "MLD", days),
                  "J_MLDm": _grad_of(grad_root, tag, "MLDm", days)}
            for J, g in ad.items():
                if g is None:
                    continue
                for c in CONSTS:
                    f = fd.get(n, {}).get(c, {}).get(J, {})
                    if not f:
                        continue
                    v = bar_verdict(g[c], f, days)
                    rows.append({"setting": tag, "window_days": days, "J": J, "const": c,
                                 "adjoint": g[c], "fd": {f"{h:g}": f[h] for h in sorted(f)},
                                 **v,
                                 "n_jump": fd[n][c].get("n_jump"),
                                 "n_masked": int(masks[n].sum()) if n in masks else None})
    return {"hs": hs, "ends": ends, "bar_rtol": BAR_RTOL, "plateau_rtol": PLATEAU_RTOL,
            "t_mean_steps": T_MEAN_STEPS, "rows": rows,
            "fd": {str(n): v for n, v in fd.items()}}


if __name__ == "__main__":
    sys.exit(main())
