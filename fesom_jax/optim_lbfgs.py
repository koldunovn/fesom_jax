"""Host-side limited-memory quasi-Newton minimiser for expensive cost functions (lsopt/m1qn3 style).

Built for calibrations in which ONE evaluation of ``(J, ∇J)`` is a forward plus adjoint run of the
ocean model (minutes to hours). It spends one evaluation per iteration whenever the first trial
step is accepted, and at most three per iteration otherwise. Everything is plain numpy float64,
and every inner product is a correctly rounded sum (:func:`math.fsum`), so the sequence of
proposals depends only on the evaluated ``(u, J, ∇J)`` values: replaying a log on any host
reproduces the next proposal bitwise, which is what makes a resumed calibration identical to an
uninterrupted one.

The algorithm follows the variable-storage quasi-Newton method of Gilbert & Lemaréchal (1989),
as implemented in MITgcm ``lsopt`` (``lsopt_top.F``, ``hessupd.F``, ``dgscale.F``, ``lsupdxx.F``)
and in m1qn3, with a simpler line search.

Direction
    Two-loop recursion over the last ``m`` pairs ``(s_i, y_i)`` (oldest dropped first) with the
    diagonal matrix ``D`` as the initial inverse Hessian: ``d = −H·g``.

Diagonal update (``dgscale.F``; Gilbert & Lemaréchal 1989, the diagonal BFGS update)
    ``D`` starts as the identity after a cold start. Each time a pair is STORED, with
    ``ȳ = y/√(sᵀy)`` and ``s̄ = s/√(sᵀy)``:

    1. rescale: ``D ← D / (ȳᵀ D ȳ)``  (= ``D · sᵀy / yᵀDy``; for ``D = I`` this is the classical
       ``γ·I`` with ``γ = sᵀy/yᵀy``);
    2. safeguard: ``D_i ≤ 0 → rmin``;
    3. update the diagonal of the direct Hessian ``B = D⁻¹`` by BFGS and keep only the diagonal:
       ``D_i ← 1 / (1/D_i + ȳ_i² − (s̄_i/D_i)² / Σ_j s̄_j²/D_j)``;
    4. safeguard: ``D_i ≤ 0 → rmin``; if any entry is non-finite, fall back to ``γ·I``.

    ``rmin = 1e-20`` as in lsopt. On a quadratic with a diagonal Hessian ``diag(h)``,
    ``D = diag(1/h)`` is a fixed point of steps 1-3.

Cold start (no pairs stored)
    First iteration (``lsupdxx.F``, Fletcher's scaling): ``d = −g · 2|f − f_min| / ‖g‖²`` with the
    expected optimum ``f_min = (1 − fmin_frac)·f`` and ``fmin_frac = 0.002`` (the 0.2 % ECCO uses). If
    that step is zero or not finite (``f = 0``), ``d = −g/‖g‖`` (unit length in u).
    After a RESET (below) the 0.2 % rule is not used, because near a twin optimum ``f → 0`` and the
    step would vanish: ``d = −g · L/‖g‖`` with ``L`` the length ``‖s‖`` of the last accepted step,
    or, if no step was ever accepted, the step the failed line search would have tried next.
    After an ACCEPTED step whose pair was not stored while the memory is empty, the length is
    ``expand · ‖s‖`` (option ``expand``, default 1 = the same length): a failed curvature test
    says the step was too short to see positive curvature, where a Wolfe line search (m1qn3,
    lsopt) would extrapolate. With ``expand = 1`` a cost whose slope along short steps does not
    flatten (``sᵀy ≤ 0``: negative curvature, or a gradient that switches between smooth pieces of
    a non-smooth cost) stores no pair and crawls at the first step's length until a pair happens
    to pass; the TKE-constant calibration measured exactly that (six pairs rejected in a row,
    0.5 % decrease per evaluation) and uses ``expand = 4``.

Line search / acceptance (trial ``u + t·d``, ``t = 1`` first; ``c1``, ``c2`` configurable)
    * **Armijo** (sufficient decrease): ``f₊ ≤ f + c1·t·gᵀd`` with ``c1 = 1e-6``. A non-finite
      ``f₊`` or ``g₊`` counts as a failed Armijo test.
    * Armijo met ⇒ ACCEPT the trial. The pair ``(s, y) = (u₊ − u, g₊ − g)`` is stored only if the
      **Wolfe curvature** test ``g₊ᵀd ≥ c2·gᵀd`` holds (``c2 = 0.99999``) AND ``sᵀy > 1e-10·|gᵀs|``;
      otherwise the step is accepted and the pair is dropped (the memory and ``D`` are kept).
    * Armijo failed ⇒ shorten ``t`` and try again, at most ``max_halvings = 2`` extra evaluations.
      The shorter step is the minimiser of the quadratic through ``f(0)``, ``gᵀd`` and ``f(t)``,
      kept in ``[0.01·t, 0.5·t]`` — at least a halving, and exact for a quadratic cost, so a
      first step that overshoots by orders of magnitude is repaired in one trial (lsopt's
      ``lsline.F`` interpolates cubically for the same reason); ``t/10`` if ``f(t)`` is not finite.
    * After the last halving, if some trial had ``f₊ < f`` the lowest one is ACCEPTED (same pair
      rules). If none did, the line search FAILED: stay at ``u``, clear the memory and ``D``, and
      cold-start with the reset rule above.
    * A line search that fails right after a failure-triggered reset (two consecutive failed line
      searches) stops with status ``line_search_stalled``. With the defaults a wrong-signed
      gradient therefore costs at most ``1 + 2·(1 + max_halvings) = 7`` evaluations.
    * A two-loop direction that is not a descent direction (``gᵀd ≥ 0`` or non-finite) triggers
      the same reset (memory cleared, reset cold start) without an evaluation.

    The loose constants accept nearly any decrease on purpose: each extra evaluation is a
    multi-day adjoint run, and a step that decreases J is kept rather than refined. (The online
    MITgcm ``lsline.F`` uses 1e-4 / 0.9 with cubic interpolation.)

Stop reasons (status; each is written to the log)
    ``gtol`` (‖g‖ ≤ gtol at an accepted point, or ‖g‖ = 0), ``ftol`` (relative decrease of f over the
    last ``ftol_iters`` accepted steps below ``ftol``; ``ftol = 0`` disables it), ``maxiter`` (line
    searches completed, accepted or failed), ``maxeval`` (evaluations, the initial one included),
    ``line_search_stalled``.

Log and resume
    :func:`minimize` appends one JSON line per evaluation to ``log_path``: ``u``, ``f``, ``g``
    (Python floats round-trip exactly through JSON), whether the trial was accepted, whether its
    pair was stored, the status, plus the caller's optional ``aux`` dict and the evaluation's
    wall seconds. With ``resume=True`` an existing log is replayed through the state machine
    (every logged ``u`` must equal the machine's proposal bitwise, or it raises) and the run
    continues from the next proposal; an incomplete last line is ignored. The stopping limits may
    differ on resume (a run stopped at ``maxeval`` can be extended); the algorithm options and
    ``u0`` may not. An existing log is never overwritten.

Why not ``optax.lbfgs``: its zoom line search enforces the strong Wolfe conditions and spends
several function+gradient evaluations per iteration; here each one is a multi-day adjoint.

Reference: J.C. Gilbert and C. Lemaréchal (1989), Some numerical experiments with
variable-storage quasi-Newton algorithms, Mathematical Programming 45, 407-435.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

STATUS_RUNNING = "running"
STOP_REASONS = ("gtol", "ftol", "maxiter", "maxeval", "line_search_stalled")
RMIN = 1e-20


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """Correctly rounded inner product: identical on every host, whatever BLAS it links."""
    return math.fsum((a * b).tolist())


def _norm(a: np.ndarray) -> float:
    return math.sqrt(_dot(a, a))


@dataclasses.dataclass(frozen=True)
class LBFGSOptions:
    """Algorithm options. They define the iterates, so a resume must use the same ones."""

    m: int = 5                      # stored (s, y) pairs
    c1: float = 1e-6                # Armijo (sufficient decrease)
    c2: float = 0.99999             # Wolfe curvature, only decides whether a pair is stored
    curv_eps: float = 1e-10         # store a pair only if sᵀy > curv_eps·|gᵀs|
    max_halvings: int = 2           # extra evaluations per line search after the first trial
    fmin_frac: float = 0.002        # first cold start: expected decrease f − f_min = fmin_frac·|f|
    expand: float = 1.0             # cold-start length × this after an accepted step without a pair

    def __post_init__(self):
        if self.m < 1:
            raise ValueError("m must be >= 1")
        if not (0.0 < self.c1 < self.c2 < 1.0):
            raise ValueError("need 0 < c1 < c2 < 1")
        if self.max_halvings < 0:
            raise ValueError("max_halvings must be >= 0")
        if not (0.0 < self.fmin_frac < 1.0):
            raise ValueError("fmin_frac must be in (0, 1)")
        if not (math.isfinite(self.expand) and self.expand >= 1.0):
            raise ValueError("expand must be a finite number >= 1")


@dataclasses.dataclass(frozen=True)
class Limits:
    """Stopping limits. They may change between a run and its resume."""

    maxiter: int = 100
    maxeval: int = 200
    gtol: float = 0.0
    ftol: float = 0.0
    ftol_iters: int = 3


class LBFGS:
    """The state machine. ``ask()`` returns the next point to evaluate (``None`` once stopped),
    ``tell(f, g)`` consumes that evaluation and returns a dict describing what it did.

    Attributes after ``tell``: ``u, f, g`` (current accepted iterate), ``S, Y`` (stored pairs),
    ``D`` (diagonal inverse-Hessian scaling or ``None``), ``last_step`` (length of the last
    accepted step), ``n_iter`` (completed line searches), ``n_eval``, ``status``."""

    def __init__(self, u0, options: LBFGSOptions | None = None, limits: Limits | None = None):
        self.opts = options or LBFGSOptions()
        self.limits = limits or Limits()
        self.u0 = np.array(u0, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(self.u0)):
            raise ValueError("u0 must be finite")
        self.n = self.u0.size
        self.u = self.f = self.g = None
        self.S: deque = deque(maxlen=self.opts.m)
        self.Y: deque = deque(maxlen=self.opts.m)
        self.D = None
        self.last_step = None
        self.n_iter = 0
        self.n_eval = 0
        self.n_fail = 0                 # consecutive failed line searches
        self.status = STATUS_RUNNING
        self.f_hist: list[float] = []   # f at each accepted point (initial included)
        # line-search state
        self.d = None
        self.gd = None
        self.t = None
        self.n_trial = 0
        self.best = None                # (f, u, g, t) of the lowest trial with f < self.f
        self.t_next = None              # the step the line search would try after the last trial
        self._pending = self.u0.copy()  # the point ask() hands out next

    # ------------------------------------------------------------------ public API
    def ask(self):
        if self.status != STATUS_RUNNING:
            return None
        if self.n_eval >= self.limits.maxeval:
            self.status = "maxeval"
            return None
        return self._pending.copy()

    def tell(self, f, g) -> dict:
        if self.status != STATUS_RUNNING or self._pending is None:
            raise RuntimeError(f"tell() without a pending proposal (status {self.status})")
        u_t = self._pending
        f = float(f)
        g = np.array(g, dtype=np.float64).reshape(-1)
        if g.shape != (self.n,):
            raise ValueError(f"gradient has shape {g.shape}, expected ({self.n},)")
        self.n_eval += 1
        info = {"eval": self.n_eval, "accepted": False, "pair_stored": False, "reset": None,
                "t": self.t}
        if self.u is None:                                   # the initial evaluation
            if not (math.isfinite(f) and np.all(np.isfinite(g))):
                raise FloatingPointError("non-finite cost or gradient at u0")
            self.u, self.f, self.g = u_t.copy(), f, g.copy()
            self.f_hist.append(f)
            info["accepted"] = True
            if not self._check_gtol():
                self._start_line_search(self._cold_direction(first=True))
            return self._finish(info)

        ok = math.isfinite(f) and bool(np.all(np.isfinite(g)))
        if ok and f < self.f and (self.best is None or f < self.best[0]):
            self.best = (f, u_t.copy(), g.copy(), self.t)
        self.t_next = self._backtrack(f)
        armijo = ok and f <= self.f + self.opts.c1 * self.t * self.gd
        if armijo:
            self._accept(u_t, f, g, info)
        elif self.n_trial < self.opts.max_halvings:
            self.n_trial += 1
            self.t = self._backtrack(f)
            self._pending = self.u + self.t * self.d
        elif self.best is not None:
            bf, bu, bg, _ = self.best
            self._accept(bu, bf, bg, info)
            info["accepted_best_of_trials"] = True
        else:
            self._fail(info)
        return self._finish(info)

    # ------------------------------------------------------------------ internals
    def _finish(self, info):
        info["status"] = self.status
        info["iter"] = self.n_iter
        return info

    def _check_gtol(self) -> bool:
        gn = _norm(self.g)
        if gn == 0.0 or gn <= self.limits.gtol:
            self.status = "gtol"
            self._pending = None
            return True
        return False

    def _accept(self, u_new, f_new, g_new, info):
        s = u_new - self.u
        y = g_new - self.g
        sy, gs = _dot(s, y), _dot(self.g, s)
        wolfe = _dot(g_new, self.d) >= self.opts.c2 * self.gd
        store = wolfe and sy > self.opts.curv_eps * abs(gs)
        if store:
            self.S.append(s)
            self.Y.append(y)
            self._update_diag(s, y, sy)
        info.update(accepted=True, pair_stored=store, wolfe=bool(wolfe), sy=sy)
        self.u, self.f, self.g = u_new.copy(), float(f_new), g_new.copy()
        self.last_step = _norm(s)
        self.f_hist.append(self.f)
        self.n_iter += 1
        self.n_fail = 0
        if self._check_gtol() or self._check_ftol() or self._check_maxiter():
            return
        if not self.S:                          # no pair yet (all skipped): cold start again,
            # longer by `expand` when this step's own pair was dropped (too short to see curvature)
            self._start_line_search(self._cold_direction(first=False, grow=not store))
            return
        d = self._two_loop()
        gd = _dot(self.g, d)
        if not (math.isfinite(gd) and gd < 0.0 and np.all(np.isfinite(d))):
            self._reset()
            info["reset"] = "not_descent"
            d = self._cold_direction(first=False)
        self._start_line_search(d)

    def _fail(self, info):
        self.n_iter += 1
        self.n_fail += 1
        info["reset"] = "line_search_failed"
        if self.n_fail >= 2:
            self.status = "line_search_stalled"
            self._pending = None
            return
        if self._check_maxiter():
            return
        self._reset()
        self._start_line_search(self._cold_direction(first=False))

    def _check_ftol(self) -> bool:
        k = self.limits.ftol_iters
        if self.limits.ftol > 0.0 and len(self.f_hist) > k:
            f_old = self.f_hist[-1 - k]
            if (f_old - self.f) <= self.limits.ftol * max(abs(f_old), 1e-300):
                self.status = "ftol"
                self._pending = None
                return True
        return False

    def _check_maxiter(self) -> bool:
        if self.n_iter >= self.limits.maxiter:
            self.status = "maxiter"
            self._pending = None
            return True
        return False

    def _reset(self):
        self.S.clear()
        self.Y.clear()
        self.D = None

    def _start_line_search(self, d):
        self.d = d
        self.gd = _dot(self.g, d)
        self.t = 1.0
        self.n_trial = 0
        self.best = None
        self.t_next = None
        self._pending = self.u + self.t * self.d

    def _cold_direction(self, first: bool, grow: bool = False):
        g = self.g
        gn = _norm(g)
        if first:
            fdiff = self.opts.fmin_frac * abs(self.f)       # |f − f_min|, f_min = (1−frac)·f
            scale = 2.0 * fdiff / (gn * gn)
            if not (math.isfinite(scale) and scale > 0.0):
                scale = 1.0 / gn
            return -g * scale
        if self.last_step is not None and self.last_step > 0.0:
            length = self.last_step * (self.opts.expand if grow else 1.0)
        else:                                               # never accepted a step yet
            length = self.t_next * _norm(self.d)
        return -g * (length / gn)

    def _backtrack(self, f_t):
        """The next, shorter trial step after a failed Armijo test at ``self.t``: the minimiser
        of the quadratic through ``f(0)``, ``f'(0) = gᵀd`` and ``f(t)``, kept in
        ``[0.01·t, 0.5·t]`` (at least a halving); ``t/10`` if ``f(t)`` is not finite."""
        t = self.t
        if not math.isfinite(f_t):
            return 0.1 * t
        den = 2.0 * (f_t - self.f - self.gd * t)
        tq = -self.gd * t * t / den if den > 0.0 else 0.5 * t
        return min(0.5 * t, max(0.01 * t, tq))

    def _two_loop(self):
        q = self.g.copy()
        rho = [1.0 / _dot(s, y) for s, y in zip(self.S, self.Y)]
        alpha = [0.0] * len(self.S)
        for i in range(len(self.S) - 1, -1, -1):
            alpha[i] = rho[i] * _dot(self.S[i], q)
            q = q - alpha[i] * self.Y[i]
        r = q * self.D if self.D is not None else q
        for i in range(len(self.S)):
            beta = rho[i] * _dot(self.Y[i], r)
            r = r + (alpha[i] - beta) * self.S[i]
        return -r

    def _update_diag(self, s, y, sy):
        """lsopt ``dgscale``: rescale D, then the diagonal of the BFGS update of B = D⁻¹."""
        D = np.ones(self.n) if self.D is None else self.D.copy()
        root = math.sqrt(sy)
        yb, sb = y / root, s / root
        D = D / _dot(yb * D, yb)
        D = np.where(D > 0.0, D, RMIN)
        den = math.fsum((sb * sb / D).tolist())
        D = 1.0 / (1.0 / D + yb * yb - (sb / D) ** 2 / den)
        D = np.where(D > 0.0, D, RMIN)
        if not np.all(np.isfinite(D)):
            D = np.full(self.n, sy / _dot(y, y))
        self.D = D


# ---------------------------------------------------------------------------- driver
@dataclasses.dataclass
class Result:
    u: np.ndarray
    f: float
    g: np.ndarray
    status: str
    n_iter: int
    n_eval: int
    machine: LBFGS


def _read_log(path: Path):
    header, evals = None, []
    with open(path, "rb") as fh:
        raw = fh.read()
    lines = raw.split(b"\n")
    if lines and lines[-1] != b"":           # incomplete last line (writer interrupted)
        lines = lines[:-1]
    for ln in lines:
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            break                              # a torn line: everything after it is ignored
        if rec.get("type") == "header":
            header = rec
        elif rec.get("type") == "eval":
            evals.append(rec)
    return header, evals


def _append(fh, rec):
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def minimize(fun_and_grad, u0, *, maxiter=100, maxeval=200, gtol=0.0, ftol=0.0, ftol_iters=3,
             log_path=None, resume=True, options: LBFGSOptions | None = None,
             run_info: dict | None = None) -> Result:
    """Minimise ``fun_and_grad(u) -> (f, g)`` or ``(f, g, aux)`` from ``u0`` (numpy float64).

    ``aux`` (optional, JSON-serialisable dict) is logged with the evaluation. With ``log_path``
    every evaluation is appended to a JSONL log; ``resume=True`` replays an existing log first
    (see the module docstring), ``resume=False`` refuses to touch an existing log.
    ``run_info`` (dict) is written into the header (provenance)."""
    opts = options or LBFGSOptions()
    limits = Limits(maxiter=maxiter, maxeval=maxeval, gtol=gtol, ftol=ftol, ftol_iters=ftol_iters)
    mach = LBFGS(u0, opts, limits)
    fh = None
    if log_path is not None:
        log_path = Path(log_path)
        if log_path.exists():
            if not resume:
                raise FileExistsError(f"{log_path} exists; pass resume=True or a new path")
            header, evals = _read_log(log_path)
            if header is None:
                raise ValueError(f"{log_path}: no header record")
            if header["options"] != dataclasses.asdict(opts):
                raise ValueError(f"{log_path}: options differ from the log's "
                                 f"{header['options']} vs {dataclasses.asdict(opts)}")
            if np.array(header["u0"], np.float64).tobytes() != mach.u0.tobytes():
                raise ValueError(f"{log_path}: u0 differs from the log's")
            for rec in evals:
                u = mach.ask()
                if u is None:
                    raise ValueError(f"{log_path}: the machine stopped ({mach.status}) before "
                                     f"logged evaluation {rec['eval']}; limits smaller than the "
                                     f"logged run's?")
                if np.array(rec["u"], np.float64).tobytes() != u.tobytes():
                    raise ValueError(f"{log_path}: evaluation {rec['eval']} does not match the "
                                     f"replayed proposal (log from another run or options?)")
                info = mach.tell(rec["f"], rec["g"])
                if (info["accepted"], info["pair_stored"]) != (rec["accepted"],
                                                              rec["pair_stored"]):
                    raise ValueError(f"{log_path}: replay diverged at evaluation {rec['eval']}")
            fh = open(log_path, "a")
            # a torn last line is kept on disk (nothing is deleted); a newline makes the next
            # record start on its own line, and the reader skips the torn one
            if log_path.stat().st_size and not log_path.read_bytes().endswith(b"\n"):
                fh.write("\n")
            if evals:
                _append(fh, {"type": "resume", "after_eval": len(evals), "time": time.time(),
                             "limits": dataclasses.asdict(limits)})
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log_path, "x")
            _append(fh, {"type": "header", "options": dataclasses.asdict(opts),
                         "limits": dataclasses.asdict(limits), "u0": mach.u0.tolist(),
                         "n": mach.n, "time": time.time(), "run_info": run_info or {}})
    try:
        while True:
            u = mach.ask()
            if u is None:
                break
            t0 = time.time()
            out = fun_and_grad(u)
            dt = time.time() - t0
            f, g = out[0], out[1]
            aux = out[2] if len(out) > 2 else None
            g = np.array(g, dtype=np.float64).reshape(-1)
            info = mach.tell(f, g)
            if fh is not None:
                rec = {"type": "eval", "u": u.tolist(), "f": float(f), "g": g.tolist(),
                       "eval_s": dt}
                rec.update({k: v for k, v in info.items()})
                if aux is not None:
                    rec["aux"] = aux
                _append(fh, rec)
        if fh is not None:
            _append(fh, {"type": "stop", "status": mach.status, "n_iter": mach.n_iter,
                         "n_eval": mach.n_eval, "f": mach.f, "u": mach.u.tolist(),
                         "gnorm": _norm(mach.g), "time": time.time()})
    finally:
        if fh is not None:
            fh.close()
    return Result(u=mach.u.copy(), f=mach.f, g=mach.g.copy(), status=mach.status,
                  n_iter=mach.n_iter, n_eval=mach.n_eval, machine=mach)
