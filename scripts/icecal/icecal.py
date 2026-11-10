#!/usr/bin/env python
"""Evaluation driver and L-BFGS chain of the sea-ice thermodynamic calibration on CORE2.

    python scripts/icecal/icecal.py eval    --run-dir D --state S --start 2013-01-01 --days 365 \\
                                            --obs OBS [--u U] [--chunk-steps 24] [--stride K] \\
                                            [--check-linear zero|full|off] \\
                                            [--terms J_C_NH,J_C_SH,J_H_NH]
    python scripts/icecal/icecal.py fwd    --run-dir D --state S --start ... --days ... --obs OBS \\
                                            [--u U] [--save-state-at 2013-07-01]
    python scripts/icecal/icecal.py fdbatch --root FD --state S --start ... --days ... --obs OBS \\
                                            --controls albsn,con|all --hs 1e-2,3e-2 --signs +,- \\
                                            [--u U] [--base-repeats N]
    python scripts/icecal/icecal.py chain-init --run-dir R --windows 2013 --mode twin|real \\
                                            --prior on|off --maxeval 15 [--truth u.json] \\
                                            [--free con,consn,...] [--fmin-frac 0.05]
    python scripts/icecal/icecal.py chain-step --run-dir R [--window 2013]

Run from a checkout with ``PYTHONPATH=<checkout>`` (``scripts/adjoint/job_cpu.sbatch`` on CPU, the
dolpung GH200 wrapper on GPU; ``scripts/icecal/job_chain_gh200.sh`` submits chain steps). Every
output path is an argument and must lie outside the checkout; nothing is overwritten or deleted.

**Model.** The CORE2 reference ocean–sea-ice model (``configs/core2_full.yaml``: zstar, TKE, GM,
mEVP, JRA55-do, dt 1800 s) built by :func:`core2_adj.build_model` from the State ``--state``
(any restart in the ``load_state`` format whose date is ``--start``), with the derivative-only
settings of the calibration: ``freeze_mixing+freeze_gm_slope+freeze_fct_limiter``
(:data:`SETTING`) and the ice adjoint ``adjoint_mode="exact"`` with the frozen-sweep mEVP
backward (θ 0.97, κ0 1; :data:`ICE_ADJOINT`). The ten thermodynamic controls
(:func:`fesom_jax.controls.ice_thermo_control_set`, ``p = p0·exp(σ·u)``) enter as ONE strong
float64 vector ``{"u": [10]}`` and reach the ice step through ``Params.ice_ctrl``; ``u = 0`` is
the configured ``IceConfig`` (checked at build time).

**Monthly means.** The chunk function is local (it mirrors ``core2_adj._make_chunk_fn``): its
``xs`` is the pair (StepForcing of the chunk's steps, the chunk's month index as a traced int32
scalar from :func:`fesom_jax.seaice_cost.chunk_months`), and it builds the monthly accumulators
of :func:`seaice_cost.monthly_accumulators` as closures over that scalar. One chunk function per
process, so every evaluation, forward and FD forward of a process share one compiled forward
(and one backward).

**Cost.** ``J_data = J_C,NH + J_C,SH + J_H,NH`` of :func:`seaice_cost.cost_terms` against the
super-observations ``--obs`` (an ``obs_<YYYY>.npz`` of ``scripts/icecal/prep_obs.py``, its year
must be the window's; its sha256 is recorded). The prior ``½uᵀu`` is NOT part of ``J_data``;
``eval.json`` records it separately and the chain adds it when ``--prior on``.

**Outputs.** ``eval`` writes into a NEW directory ``D``: ``eval.json`` (the J terms, the
per-month terms, the weight sums ``N_*`` and ``Nact_*``, ``g_u`` of the data part, the prior,
wall / compile seconds and the number of backend compiles, host VmHWM, device peak, Slurm job,
provenance incl. the checkout HEAD read through the worktree's ``.git`` file, the controls, the
settings and the observation file hash), ``fields.npz`` (``C_mean``, ``H_mean`` [12, nod2D]
monthly means of the post-step ``a_ice``/``m_ice``, ``n_mon`` [12] steps per month, ``year``: the
format ``scripts/icecal/score.py --kind jax`` reads) and ``trace.json`` (the per-boundary
cotangent norms, total and per State field); with ``--terms`` the gradient of each listed cost
term from one forward and one reverse walk (``per_term`` in ``eval.json``, one trace per term;
:func:`run_eval`). ``fwd`` runs the same forward without the reverse
(``fwd.json`` + ``fields.npz``; ``--save-state-at`` writes the State at that chunk boundary as
``fesom.<YYYY>.<DOY>.<SSSSS>`` in the ``load_state`` format). ``fdbatch`` runs several perturbed
forwards in one process (one compile): perturbations ``±h`` in ln p of each named control, each
into its own subdirectory ``<control>_h<h>_<p|m>`` (and ``base_r<i>`` for ``--base-repeats``)
with its own ``fwd.json`` and ``fields.npz``; a subdirectory that already holds ``fwd.json`` is
skipped (so a batch split over jobs resumes), an incomplete one is refused.

**Chain** (one evaluation per job; Task 8 of the calibration plan). ``chain-init`` writes
``R/run_config.json`` (once, mode ``"x"``): windows (years; the State
``<states-dir>/fesom.<YYYY>.001.00000`` and observations ``<obs-dir>/obs_<YYYY>.npz`` unless
given), mode twin|real (twin: the truth u from ``--truth``), prior on|off, ``maxeval``, the
``LBFGSOptions(expand=4)`` of the TKE calibration. ``chain-step`` reuses the file-backed
coordinator of ``scripts/calib/tke_lbfgs.py`` by import (``run_mjstep`` replays
``R/iterations.jsonl`` through ``optim_lbfgs.minimize``; ``_mj_try_coordinate`` runs it once per
completed evaluation under an ``O_EXCL`` lock): coordinator -> evaluation k needs window W at u ->
this process evaluates it (``R/eval_<k>/win_<W>.<job>/``) and writes ``R/eval_<k>/win_<W>.json``
NO-CLOBBER (a temporary under ``eval_<k>/.tmp/`` hard-linked to the target with ``os.link``,
which fails if the target exists; the temporary name stays as a second link of the same file) ->
coordinator -> if evaluation k+1 is needed, ``R/STOP`` is absent and k+1 <= maxeval: submit one
chain step per window of k+1 (``job_chain_gh200.sh``, which appends to ``R/jobs.txt`` and the
ledger and refuses at >= 4 dolpung jobs). The coordinator's window ids are the window years: its
window files are named ``win_<YYYY>.json`` and carry the year under the key ``month`` (the key
``tke_lbfgs`` reads). A point outside the physical domain (:meth:`ControlSet.out_of_domain`) is a
failed evaluation recorded by the coordinator without a model run (:class:`DomainControlSet`).
Exit status: the evaluation's (an exception = non-zero, nothing submitted); 3 when the evaluation
succeeded but the next step could not be submitted (the chain is stalled: rerun
``job_chain_gh200.sh R W``, which resumes); 4 when no point was posted in time.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
sys.path.insert(0, str(ROOT / "scripts" / "calib"))
import core2_adj as C  # noqa: E402  (runs the import guard)

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fesom_jax import adjoint, optim_lbfgs, seaice_cost  # noqa: E402
from fesom_jax import adjoint_costs as AC  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax.controls import (ICE_PREFIX, ControlSet, ice_joint_control_set,  # noqa: E402
                                ice_thermo_control_set)
from fesom_jax.params import Params  # noqa: E402

RUNS = Path("/work/ab0995/a270088/fesom_jax_dev/runs/icecal")
OBS_DIR = RUNS / "obs_v2"
STATES_DIR = RUNS / "states"
#: derivative-only TKE/GM/advection settings of the calibration (``tke_diag.parse_setting``)
SETTING = "freeze_mixing+freeze_gm_slope+freeze_fct_limiter"
ICE_MODE = "exact"
#: derivative-only mEVP settings: the frozen-sweep backward, θ 0.97, κ0 1
ICE_ADJOINT = {"adjoint_damping": "frozen_sweep", "adjoint_theta": 0.97,
               "adjoint_alpha_kappa": 1.0}
#: the further derivative-only IceConfig fields ``--ice-cuts/--ice-thin-mass/--ice-sweep-guard``
#: may add to :data:`ICE_ADJOINT` (the farc session's cuts: "fct_flux", "strength_conc", the
#: thin-ice load cut and the frozen-sweep guard). The forward is bitwise the same, so FD runs of one window
#: serve evaluations with and without them.
ICE_ADJOINT_EXTRA = ("adjoint_cuts", "adjoint_thin_mass", "adjoint_sweep_guard")
#: ``--controls-set`` of eval / fwd / fdbatch (the chain keeps the thermodynamic set)
CONTROL_SETS = {"thermo": ice_thermo_control_set, "joint": ice_joint_control_set}


def ice_adjoint(extra: dict | None = None) -> dict:
    """:data:`ICE_ADJOINT` with the ``extra`` fields of :data:`ICE_ADJOINT_EXTRA` set."""
    extra = dict(extra or {})
    bad = sorted(set(extra) - set(ICE_ADJOINT_EXTRA))
    if bad:
        raise ValueError(f"ice adjoint fields {bad} are not among {ICE_ADJOINT_EXTRA}")
    if "adjoint_cuts" in extra:
        extra["adjoint_cuts"] = tuple(extra["adjoint_cuts"])
    return {**ICE_ADJOINT, **extra}
CHUNK_STEPS = 24
JOB_SCRIPT = ROOT / "scripts" / "icecal" / "job_chain_gh200.sh"
#: L-BFGS options of the chain (track F's: grow the cold-start step 4x after a dropped pair)
LBFGS_EXPAND = 4.0
POST_WAIT_S = 120.0       #: chain-step: how long to wait for another process's posted point


# ---------------------------------------------------------------------------
# small helpers


def out_path(p) -> Path:
    """``p`` as a Path, refusing any location inside the checkout (outputs live on /work)."""
    q = Path(p).resolve()
    root = ROOT.resolve()
    if q == root or root in q.parents:
        raise ValueError(f"output path {q} is inside the checkout {root}; write outputs under "
                         f"{RUNS}")
    return Path(p)


def new_dir(p) -> Path:
    """Create the NEW directory ``p`` (parents as needed); an existing one is refused."""
    p = out_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.mkdir(exist_ok=False)
    return p


def write_new_json(path, obj) -> None:
    """``obj`` as JSON at ``path`` (mode ``"x"``: an existing file is refused, never replaced)."""
    with open(path, "x") as f:
        json.dump(C._jsonable(obj), f, indent=1)


def write_new_npz(path, **arrays) -> None:
    with open(path, "xb") as f:
        np.savez(f, **arrays)


def publish_json(path, obj) -> None:
    """Write ``obj`` to ``path`` atomically and NO-CLOBBER: the whole file goes to a temporary
    under ``<dir>/.tmp/`` and is then hard-linked to ``path`` (``os.link`` fails if ``path``
    exists, so a reader sees the complete file or none, and an existing file is never
    replaced). The temporary name is left as a second link to the same file (nothing is
    unlinked)."""
    path = Path(path)
    tdir = path.parent / ".tmp"
    tdir.mkdir(parents=True, exist_ok=True)
    tmp = tdir / f"{path.name}.{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}.{time.time_ns()}"
    with open(tmp, "x") as f:
        json.dump(C._jsonable(obj), f)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise FileExistsError(f"{path} exists; window files are never overwritten (the new "
                              f"content is in {tmp})") from None


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def git_head_file(root=ROOT) -> dict:
    """The checkout's HEAD read from files, without the git binary (the dolpung nodes have none):
    ``.git`` (a worktree's ``gitdir:`` file or the directory) -> ``HEAD`` -> the ref, looked up
    in the gitdir, its ``commondir`` and ``packed-refs``. ``{"head", "ref"}``; ``None`` values
    when something is missing."""
    root = Path(root)
    try:
        g = root / ".git"
        gitdir = g
        if g.is_file():
            line = g.read_text().strip()
            if not line.startswith("gitdir:"):
                return {"head": None, "ref": None}
            gitdir = Path(line.split(":", 1)[1].strip())
            if not gitdir.is_absolute():
                gitdir = (root / gitdir).resolve()
        head = (gitdir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return {"head": head, "ref": None}
        ref = head.split(":", 1)[1].strip()
        dirs = [gitdir]
        cd = gitdir / "commondir"
        if cd.is_file():
            c = Path(cd.read_text().strip())
            dirs.append(c if c.is_absolute() else (gitdir / c).resolve())
        for d in dirs:
            if (d / ref).is_file():
                return {"head": (d / ref).read_text().strip(), "ref": ref}
        for d in dirs:
            pr = d / "packed-refs"
            if pr.is_file():
                for ln in pr.read_text().splitlines():
                    parts = ln.split()
                    if len(parts) == 2 and parts[1] == ref:
                        return {"head": parts[0], "ref": ref}
        return {"head": None, "ref": ref}
    except OSError:
        return {"head": None, "ref": None}


def host_vmhwm_gb() -> float | None:
    """Peak resident set of this process (``VmHWM`` of /proc/self/status) in GB."""
    try:
        for ln in Path("/proc/self/status").read_text().splitlines():
            if ln.startswith("VmHWM:"):
                return float(ln.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return None


def device_peak_gb() -> float | None:
    """Peak device memory in use (``memory_stats()["peak_bytes_in_use"]``; None on CPU)."""
    try:
        st = jax.devices()[0].memory_stats()
    except Exception:          # noqa: BLE001 — a report field must never kill a run
        return None
    if not st or "peak_bytes_in_use" not in st:
        return None
    return float(st["peak_bytes_in_use"]) / 1e9


def _concrete(x) -> bool:
    return not isinstance(x, jax.core.Tracer)


def parse_u(spec, n: int, names=None) -> np.ndarray:
    """``None``/``"zeros"`` -> zeros; a JSON list; a JSON object ``{name: u}`` (missing names 0;
    ``ice.`` prefix optional) or ``{"u": [...]}``; or a path to a JSON file holding either."""
    if spec is None or str(spec).strip() in ("", "zeros", "0"):
        return np.zeros(n, np.float64)
    s = str(spec).strip()
    if not s.startswith(("[", "{")):
        s = Path(s).read_text()
    obj = json.loads(s)
    if isinstance(obj, dict) and "u" in obj:
        obj = obj["u"]
    if isinstance(obj, dict):
        if names is None:
            raise ValueError("a {name: u} control vector needs the control names")
        full = list(names)
        short = [x[len(ICE_PREFIX):] if x.startswith(ICE_PREFIX) else x for x in full]
        u = np.zeros(n, np.float64)
        for k, v in obj.items():
            if k in full:
                u[full.index(k)] = float(v)
            elif k in short:
                u[short.index(k)] = float(v)
            else:
                raise ValueError(f"unknown control {k!r}; known {full}")
        return u
    u = np.asarray(obj, np.float64).reshape(-1)
    if u.shape != (n,):
        raise ValueError(f"expected {n} controls, got {u.shape}")
    return u


def steps_for(*, days=None, steps=None, dt=C.DT) -> int:
    if (days is None) == (steps is None):
        raise ValueError("give exactly one of days / steps")
    if steps is not None:
        return int(steps)
    n = float(days) * 86400.0 / dt
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"{days} days is not a whole number of {dt} s steps")
    return int(round(n))


# ---------------------------------------------------------------------------
# the model


class IceCal:
    """One window of the ice calibration: the CORE2 model from ``state`` (a restart directory in
    the ``load_state`` format whose date is ``start``, or a host ``State``) over ``n_steps``
    steps, its ONE chunk function, the chunk month table, the cost arrays and the loss. Build
    once per process; every ``eval``/``fwd`` reuses the compiled programs."""

    def __init__(self, state, start, n_steps: int, *, obs, chunk_steps: int = CHUNK_STEPS,
                 setting: str = SETTING, cs: ControlSet | None = None,
                 ice_adjoint_extra: dict | None = None):
        import tke_diag
        self.start = C._parse_date(start)
        self.cs = cs if cs is not None else ice_thermo_control_set()
        self.base = Params.defaults()
        self.setting = setting
        self.ice_adjoint = ice_adjoint(ice_adjoint_extra)
        cs_, base = self.cs, self.base
        self.obs_path = Path(obs)
        with np.load(self.obs_path) as z:
            obs_d = {k: np.asarray(z[k]) for k in z.files}
        if "year" in obs_d and int(obs_d["year"]) != self.start.year:
            raise ValueError(f"observations {obs} are for {int(obs_d['year'])}, the window starts "
                             f"in {self.start.year}")
        self.ob = seaice_cost.cost_obs(obs_d)
        self.obs_sha256 = sha256_file(self.obs_path)
        if isinstance(state, (str, os.PathLike)):
            where = dict(restart=Path(state))
        else:                                   # a host State (tests): its date is ``start``
            where = dict(state=state)
        self.model = C.build_model(
            self.start, int(n_steps), int(chunk_steps), **where,
            adjoint_cfg=tke_diag.parse_setting(setting), ice_mode=ICE_MODE,
            accumulators=seaice_cost.monthly_accumulators(0), rematerialize=True,
            ice_adjoint=self.ice_adjoint, params0={"u": np.zeros(cs_.n, np.float64)},
            to_params=lambda c: cs_.to_params(c["u"], base))
        if self.ob.C.shape[1] != self.model.mesh.nod2D:
            raise ValueError(f"observations have {self.ob.C.shape[1]} nodes, the mesh "
                             f"{self.model.mesh.nod2D}")
        icfg = self.model.phys["ice_cfg"]
        for c in self.cs.controls:
            if c.name.startswith(ICE_PREFIX):
                f = c.name[len(ICE_PREFIX):]
                if float(getattr(icfg, f)) != c.p0:
                    raise ValueError(f"u = 0 must be the configured ice: {f} is "
                                     f"{getattr(icfg, f)} in the model, p0 = {c.p0}")
        self.n_steps = int(n_steps)
        self.chunk_steps = int(chunk_steps)
        self.n_chunks = self.model.chunks_for(self.n_steps)
        self.months = seaice_cost.chunk_months(self.start, self.model.dt, self.chunk_steps,
                                               self.n_chunks)
        self.chunk_fn = make_chunk_fn(self.model)
        # anything that would run the model's own chunk program (month-0 accumulators, plain
        # StepForcing xs) now fails loudly on the (forcing, month) xs instead
        self.model.chunk_fn = self.chunk_fn
        self._captured = {}

    # -- inputs ------------------------------------------------------------
    def xs_fn(self):
        """``c -> (StepForcing chunk, int32 month)``, built fresh on every call (the library
        frees what it is given)."""
        forcing = self.model.xs_fn(self.start)
        months = self.months

        def fn(c):
            return (forcing(c), jax.device_put(np.asarray(months[c], np.int32)))
        return fn

    def params(self, u) -> dict:
        u = np.asarray(u, np.float64).reshape(-1)
        if u.shape != (self.cs.n,):
            raise ValueError(f"expected {self.cs.n} controls, got {u.shape}")
        return {"u": jnp.asarray(u, jnp.float64)}

    # -- the loss ----------------------------------------------------------
    def loss_fn(self, term: str = "J_data"):
        """``carry -> J_data`` (or the cost term ``term``, one of ``seaice_cost.TERMS``). On a
        concrete carry (the forward's final state, not the traced call that seeds the reverse
        sweep) it also keeps the accumulators and the cost terms on the host, for
        ``fields.npz`` and ``eval.json``."""
        if term != "J_data" and term not in seaice_cost.TERMS:
            raise ValueError(f"unknown cost term {term!r}; known J_data, {seaice_cost.TERMS}")
        ob, cap = self.ob, self._captured

        def loss(carry):
            terms = seaice_cost.cost_terms(carry[1], ob)
            if _concrete(carry[1]["a_mon"]):
                cap["acc"] = jax.device_get(carry[1])
                cap["terms"] = jax.device_get(terms)
            return terms[term]
        return loss

    def take_captured(self) -> tuple[dict, dict]:
        acc, terms = self._captured.pop("acc"), self._captured.pop("terms")
        self._captured.clear()
        return acc, terms

    # -- runs --------------------------------------------------------------
    def forward(self, u, *, save_state_at=None, save_dir=None):
        """The window forward (no boundaries parked). Returns ``(acc, terms, info)``;
        ``save_state_at`` (a chunk-boundary time) writes that State to
        ``<save_dir>/fesom.<YYYY>.<DOY>.<SSSSS>``."""
        save_c, when = None, None
        if save_state_at is not None:
            when = C._parse_date(save_state_at)
            secs = (when - self.start).total_seconds()
            per = self.chunk_steps * self.model.dt
            if secs <= 0 or abs(secs / per - round(secs / per)) > 1e-9 \
                    or round(secs / per) > self.n_chunks:
                raise ValueError(f"--save-state-at {when} is not a chunk boundary inside the "
                                 f"window ({self.start} + k*{per:g} s, k = 1..{self.n_chunks})")
            save_c = int(round(secs / per))
        params = self.params(u)
        fwd = adjoint._jitted_forward(self.chunk_fn)
        xs_fn = self.xs_fn()
        carry = self.model.init_carry()
        saved = None
        with C._Timer() as t:
            for c in range(self.n_chunks):
                xs = xs_fn(c)
                nxt = jax.block_until_ready(fwd(carry, xs, params))
                adjoint._free(xs)
                adjoint._free(carry)
                carry = nxt
                if save_c is not None and c + 1 == save_c:
                    st = jax.device_get(self.model.state_of(carry))
                    step0 = int(self.model.restart_meta.get("step", 0))
                    saved = C._write_state(
                        Path(save_dir) / C.restart_tag(when), st, self.model.mesh,
                        step=step0 + save_c * self.chunk_steps, when=when,
                        attrs={"source": "icecal fwd --save-state-at",
                               "git_head": git_head_file()["head"],
                               "slurm_job": os.environ.get("SLURM_JOB_ID"),
                               "u": np.asarray(u, np.float64).tolist(),
                               "window_start": self.start.isoformat()})
            float(self.loss_fn()(carry))
        adjoint._free(carry)
        acc, terms = self.take_captured()
        return acc, terms, {**t.fields(), "saved_state": None if saved is None else str(saved)}

    def fields(self, acc) -> dict:
        Cm, Hm, _ = seaice_cost.monthly_means(acc, np)
        return {"C_mean": np.asarray(Cm, np.float64), "H_mean": np.asarray(Hm, np.float64),
                "n_mon": np.asarray(acc["n_mon"], np.float64),
                "year": np.asarray(self.start.year, np.int64)}

    def provenance(self, action: str, u, *, timer=None, stride=None, **extra) -> dict:
        u = np.asarray(u, np.float64)
        return C._jsonable({
            **C.provenance(self.model, action, timer=timer, stride=stride),
            "git_head_file": git_head_file(), "setting": self.setting,
            "ice_adjoint": self.ice_adjoint,
            "ice_cfg": dict(self.model.phys["ice_cfg"]._asdict()),
            "obs": str(self.obs_path), "obs_sha256": self.obs_sha256,
            "state": str(self.model.restart), "months": self.months.tolist(),
            "controls": list(self.cs.names), "p0": self.cs.p0.tolist(),
            "sigma": self.cs.sigma.tolist(), "u": u.tolist(),
            "p": (self.cs.p0 * np.exp(self.cs.sigma * u)).tolist(),
            "in_domain": self.cs.in_domain(u), "out_of_domain": self.cs.out_of_domain(u),
            "host_vmhwm_gb": host_vmhwm_gb(), "device_peak_gb": device_peak_gb(), **extra})


def make_chunk_fn(model):
    """``chunk_fn(carry, (StepForcing, month), {"u": [n]}) -> carry`` with the monthly
    accumulators closed over the chunk's traced month index (mirrors
    ``core2_adj._make_chunk_fn``)."""
    mesh, op, fs, phys = model.mesh, model.op, model.forcing.static, model.phys
    adj, dt, to_params = model.adjoint_cfg, model.dt, model.to_params
    save_names = tuple(model.save_names)

    def chunk_fn(carry, xs, params):
        sf, month = xs
        p = to_params(params)

        def one_step(s, x):
            return stepmod.step(s, mesh, op, None, p, dt=dt, is_first_step=False,
                                step_forcing=x, forcing_static=fs, adjoint_cfg=adj, **phys)

        body = AC.with_running_costs(one_step, seaice_cost.monthly_accumulators(month))
        return adjoint.scan_steps(body, carry, sf, rematerialize=True, save_names=save_names)
    return chunk_fn


def default_stride(n_chunks: int) -> int:
    """≈ √n_chunks (host memory ≈ n/k + k States, minimal there)."""
    return max(1, int(round(math.sqrt(n_chunks))))


def _terms_json(terms: dict) -> dict:
    return {k: (float(v) if np.ndim(v) == 0 else np.asarray(v, np.float64).tolist())
            for k, v in terms.items()}


COT_MAP_FIELDS = ("a_ice", "m_ice", "m_snow", "t_skin", "u_ice", "v_ice")
COT_COLUMN_FIELDS = ("T", "S", "uv")


def cot_maps_hook(out_dir: Path, terms: list):
    """``on_cot(k, c, cot)`` for :func:`adjoint.chunked_grad_multi`: at EVERY boundary ``c`` the
    State cotangent of term ``k`` goes to ``<out_dir>/<term>/b<c:04d>.npz`` (compressed,
    float64): the 2-D ice fields ``COT_MAP_FIELDS`` as they are and the column L2 norm
    ``<f>_col`` of ``COT_COLUMN_FIELDS`` (per node, per element for ``uv``). One 12-h chunk
    whose trace jumps is then located by node from the two files around it."""
    dirs = [Path(out_dir) / t for t in terms]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=False)

    def on_cot(k, c, cot):
        st = cot[0]
        arrs = {f: np.asarray(jax.device_get(getattr(st, f)), np.float64)
                for f in COT_MAP_FIELDS}
        for f in COT_COLUMN_FIELDS:
            a = np.asarray(jax.device_get(getattr(st, f)), np.float64)
            arrs[f + "_col"] = np.sqrt(np.sum(a.reshape(a.shape[0], -1) ** 2, axis=1))
        with open(dirs[k] / f"b{c:04d}.npz", "wb") as fh:
            np.savez_compressed(fh, **arrs)
    return on_cot


def run_eval(cal: IceCal, run_dir, u, *, stride: int | None = None, check_linear=True,
             terms: list | None = None, extra: dict | None = None,
             cot_maps: bool = False) -> dict:
    """One evaluation: ``J_data`` and its gradient w.r.t. u (one ``chunked_grad``, boundaries in
    host RAM, every ``stride``-th kept). Writes ``eval.json``, ``fields.npz`` and ``trace.json``
    into the NEW directory ``run_dir`` and returns the ``eval.json`` content.

    ``terms`` (a list of ``seaice_cost.TERMS``): the gradient of EACH listed cost term instead,
    in one forward and one reverse walk (``adjoint.chunked_grad_multi``; reverse ≈ K × one
    sweep). ``eval.json`` then has ``per_term`` (``{term: {"J", "g_data", "g_lnp"}}``),
    ``g_data`` = the sum of the listed terms' gradients and ``g_data_is_full`` = whether every
    unlisted term is exactly 0 in this window (then that sum IS the gradient of ``J_data``, e.g.
    ``J_H_NH`` in a July window without thickness observations); ``trace.json`` has one trace
    per term (``traces``, ``field_traces``).

    ``cot_maps`` (needs ``terms``): the State cotangent at every boundary to
    ``<run_dir>/cot_maps`` (:func:`cot_maps_hook`; ~1-3 MB per boundary and term)."""
    if cot_maps and not terms:
        raise ValueError("--cot-maps needs --terms (the per-term reverse walk)")
    run_dir = new_dir(run_dir)
    u = np.asarray(u, np.float64).reshape(-1)
    cal.cs.check(u)
    stride = default_stride(cal.n_chunks) if stride is None else int(stride)
    params = cal.params(u)
    common = dict(init_state=cal.model.init_carry(), n_chunks=cal.n_chunks, xs_fn=cal.xs_fn(),
                  chunk_steps=cal.chunk_steps, boundary_stride=stride,
                  check_linear=check_linear, field_trace=True)
    if terms:
        terms = list(terms)
        bad = [x for x in terms if x not in seaice_cost.TERMS]
        if bad or len(set(terms)) != len(terms):
            raise ValueError(f"--terms {terms}: each of {seaice_cost.TERMS} at most once")
        with C._Timer() as t:
            res = adjoint.chunked_grad_multi(
                cal.chunk_fn, params, loss_fns=[cal.loss_fn(x) for x in terms],
                on_cot=cot_maps_hook(run_dir / "cot_maps", terms) if cot_maps else None,
                **common)
        acc, tvals = cal.take_captured()
        gs = [np.asarray(jax.device_get(gk["u"]), np.float64) for gk in res.grads]
        g = np.sum(gs, axis=0)
        loss = float(tvals["J_data"])
        per_term = {x: {"J": float(res.losses[k]), "g_data": gs[k].tolist(),
                        "g_lnp": (gs[k] / cal.cs.sigma).tolist()} for k, x in enumerate(terms)}
        full = all(float(tvals[x]) == 0.0 for x in seaice_cost.TERMS if x not in terms)
        trace = {"boundaries": list(range(res.n_chunks, -1, -1)), "terms": terms,
                 "traces": dict(zip(terms, res.traces)),
                 "field_traces": None if res.field_traces is None
                 else dict(zip(terms, res.field_traces))}
        more = {"per_term": per_term, "g_data_terms": terms, "g_data_is_full": bool(full)}
        finite = math.isfinite(loss) and all(np.all(np.isfinite(x)) for x in gs)
    else:
        with C._Timer() as t:
            res = adjoint.chunked_grad(cal.chunk_fn, params, loss_fn=cal.loss_fn(), **common)
        acc, tvals = cal.take_captured()
        g = np.asarray(jax.device_get(res.grad["u"]), np.float64)
        loss = float(res.loss)
        trace = {"boundaries": list(range(res.n_chunks, -1, -1)), "trace": res.trace,
                 "field_trace": res.field_trace}
        more = {}
        finite = math.isfinite(loss) and bool(np.all(np.isfinite(g)))
    out = {
        "action": "eval", "J_data": loss, "terms": _terms_json(tvals),
        "g_data": g.tolist(), "J_prior": 0.5 * math.fsum((u * u).tolist()),
        "g_prior": u.tolist(), "g_lnp_data": (g / cal.cs.sigma).tolist(), **more,
        "forward_seconds": res.forward_seconds, "reverse_seconds": res.reverse_seconds,
        "host_gb_boundaries": res.host_gb, "n_chunks": res.n_chunks,
        "chunk_steps": res.chunk_steps, "boundary_stride": res.boundary_stride,
        "check_linear": str(check_linear), "linear_check": res.linear_check,
        "finite": bool(finite),
        "provenance": cal.provenance("eval", u, timer=t, stride=stride, **(extra or {})),
    }
    write_new_npz(run_dir / "fields.npz", **cal.fields(acc))
    write_new_json(run_dir / "trace.json", trace)
    write_new_json(run_dir / "eval.json", out)
    return C._jsonable(out)


def run_fwd(cal: IceCal, run_dir, u, *, save_state_at=None, extra: dict | None = None) -> dict:
    """The forward only: ``fwd.json`` (J terms, timing, provenance) and ``fields.npz`` into the
    NEW directory ``run_dir`` (and the State at ``save_state_at``)."""
    run_dir = new_dir(run_dir)
    u = np.asarray(u, np.float64).reshape(-1)
    cal.cs.check(u)
    acc, terms, info = cal.forward(u, save_state_at=save_state_at, save_dir=run_dir)
    out = {"action": "fwd", "J_data": float(terms["J_data"]), "terms": _terms_json(terms),
           "J_prior": 0.5 * math.fsum((u * u).tolist()), "saved_state": info["saved_state"],
           "provenance": cal.provenance("fwd", u, **{k: v for k, v in info.items()
                                                    if k != "saved_state"}, **(extra or {}))}
    write_new_npz(run_dir / "fields.npz", **cal.fields(acc))
    write_new_json(run_dir / "fwd.json", out)
    return C._jsonable(out)


def fd_runs(cs: ControlSet, u0, controls, hs, signs, base_repeats: int = 0) -> list:
    """``[(tag, u, info)]`` of an FD batch: perturbation ``sign·h`` in ln p of each control
    (``u_i += sign·h/σ_i``), and ``base_repeats`` unperturbed runs."""
    u0 = np.asarray(u0, np.float64).reshape(-1)
    names = list(cs.names)
    short = [x[len(ICE_PREFIX):] if x.startswith(ICE_PREFIX) else x for x in names]
    if list(controls) == ["all"]:
        controls = short
    runs = [(f"base_r{i}", u0.copy(), {"control": None, "h_lnp": 0.0, "sign": 0})
            for i in range(int(base_repeats))]
    for name in controls:
        if name in short:
            i = short.index(name)
        elif name in names:
            i = names.index(name)
        else:
            raise ValueError(f"unknown control {name!r}; known {short}")
        for h in hs:
            for s in signs:
                u = u0.copy()
                u[i] = u0[i] + s * float(h) / cs.sigma[i]
                runs.append((f"{short[i]}_h{float(h):g}_{'p' if s > 0 else 'm'}", u,
                             {"control": names[i], "h_lnp": float(h), "sign": int(s),
                              "lnp_delta": float(cs.sigma[i] * (u[i] - u0[i]))}))
    return runs


def run_fdbatch(cal: IceCal, root, u0, controls, hs, signs, *, base_repeats: int = 0) -> list:
    """Several perturbed forwards in one process (one compiled program), each into its own
    subdirectory of ``root``; a subdirectory with ``fwd.json`` is skipped, an incomplete one
    (no ``fwd.json``) is refused."""
    root = out_path(root)
    root.mkdir(parents=True, exist_ok=True)
    runs = fd_runs(cal.cs, u0, controls, hs, signs, base_repeats)
    bad = [tag for tag, _, _ in runs if (root / tag).exists()
           and not (root / tag / "fwd.json").exists()]
    if bad:
        raise FileExistsError(f"{root}: incomplete FD run directories {bad} (no fwd.json); "
                              f"move them aside first")
    done = []
    for tag, u, info in runs:
        d = root / tag
        if (d / "fwd.json").exists():
            print(f"[icecal] {d} done already; skipped", flush=True)
            continue
        r = run_fwd(cal, d, u, extra={"fd": {**info, "u0": np.asarray(u0).tolist()}})
        done.append(tag)
        print(json.dumps({"fd": tag, "J_data": r["J_data"],
                          "wall_s": r["provenance"].get("wall_s")}), flush=True)
    return done


# ---------------------------------------------------------------------------
# the chain (Task 8): the file-backed coordinator of tke_lbfgs, by import


class DomainControlSet(ControlSet):
    """A :class:`ControlSet` whose host-side ``check`` also refuses a point outside the physical
    domain (:meth:`ControlSet.out_of_domain`). ``tke_lbfgs.mj_fun_and_grad`` turns a ``check``
    failure into a failed evaluation (f = g = NaN) WITHOUT posting the point, so the line search
    shortens the step and no model runs for it."""

    def check(self, u):
        u = super().check(u)
        bad = self.out_of_domain(u)
        if bad:
            raise ValueError(f"outside the physical domain: {bad}")
        return u


def _lbfgs():
    import tke_lbfgs
    return tke_lbfgs


def chain_options(fmin_frac: float | None = None) -> optim_lbfgs.LBFGSOptions:
    """The chain's L-BFGS options; ``fmin_frac`` (the expected first decrease as a fraction of J,
    which sets the cold-start step length) replaces the 0.2 % default if given."""
    kw = {} if fmin_frac is None else {"fmin_frac": float(fmin_frac)}
    return optim_lbfgs.LBFGSOptions(expand=LBFGS_EXPAND, **kw)


def chain_init(run_dir, windows, *, mode: str, prior: bool, maxeval: int = 15,
               maxiter: int | None = None, truth=None, states=None, obs=None,
               states_dir=STATES_DIR, obs_dir=OBS_DIR, days: float = 365.0,
               chunk_steps: int = CHUNK_STEPS, stride: int | None = None, u0=None,
               first_check: str = "full", setting: str = SETTING,
               ice_adjoint_extra: dict | None = None, free=None,
               fmin_frac: float | None = None) -> dict:
    """Write ``run_config.json`` (once; mode ``"x"``) for an L-BFGS chain over ``windows``
    (years). ``states`` / ``obs``: ``{year: path}`` overrides of the default locations. The file
    holds the keys ``tke_lbfgs.run_mjstep`` compares on every step (``calib``, ``optimizer``,
    ``options``, ``free``). ``free``: optimise only these controls (names; ``None`` = all), the
    others stay at ``u0``. ``fmin_frac``: see :func:`chain_options`."""
    run_dir = out_path(run_dir)
    if mode not in ("twin", "real"):
        raise ValueError(f"mode must be twin or real, got {mode!r}")
    cs = ice_thermo_control_set()
    windows = [int(w) for w in windows]
    if not windows or len(set(windows)) != len(windows):
        raise ValueError(f"windows must be distinct years, got {windows}")
    if mode == "twin" and truth is None:
        raise ValueError("--mode twin needs --truth")
    if mode == "real" and truth is not None:
        raise ValueError("--truth is for --mode twin only")
    u_truth = None if truth is None else parse_u(truth, cs.n, cs.names)
    u0 = np.zeros(cs.n) if u0 is None else np.asarray(u0, np.float64).reshape(-1)
    if free is not None:                # full ("ice.con") or short ("con") names
        free = [c if c in cs.names else ICE_PREFIX + c for c in free]
        unknown = sorted(set(free) - set(cs.names))
        if unknown or not free:
            raise ValueError(f"free controls {list(free)}: unknown {unknown} (have {list(cs.names)})")
    # the order run_mjstep writes (tke_lbfgs.free_indices: by index), so its comparison agrees
    free_names = [cs.names[i] for i in _lbfgs().free_indices(cs, free)]
    win = {}
    for y in windows:
        st = Path((states or {}).get(y) or Path(states_dir) / C.restart_tag(
            datetime.datetime(y, 1, 1)))
        ob = Path((obs or {}).get(y) or Path(obs_dir) / f"obs_{y}.npz")
        for what, p in (("state", st), ("observations", ob)):
            if not p.exists():
                raise FileNotFoundError(f"window {y}: {what} {p} missing")
        win[str(y)] = {"state": str(st), "obs": str(ob), "obs_sha256": sha256_file(ob),
                       "start": datetime.datetime(y, 1, 1).isoformat()}
    opts = chain_options(fmin_frac)
    cfg = {
        "calib": {"cost": "sum_w J_data(window w) + prior*0.5*u.u",
                  "mode": mode, "prior": bool(prior), "windows": windows, "window_files": win,
                  "days": float(days), "chunk_steps": int(chunk_steps),
                  "stride": None if stride is None else int(stride),
                  "first_check": first_check, "setting": setting, "ice_mode": ICE_MODE,
                  "ice_adjoint": ice_adjoint(ice_adjoint_extra), "controls": list(cs.names),
                  "p0": cs.p0.tolist(), "sigma": cs.sigma.tolist(), "u0": u0.tolist(),
                  "u_truth": None if u_truth is None else u_truth.tolist(),
                  "p_truth": None if u_truth is None else
                  (cs.p0 * np.exp(cs.sigma * u_truth)).tolist(),
                  "maxeval": int(maxeval), "maxiter": int(maxiter or maxeval),
                  "multijob": True},
        "optimizer": "lbfgs", "options": C._jsonable(vars(opts)), "free": free_names,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_head_file": git_head_file(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_config.json", "x") as f:
        json.dump(C._jsonable(cfg), f, indent=1)
    return cfg


def load_chain(run_dir) -> tuple[dict, dict]:
    """``(run_config, coord)``: the coordinator settings for ``tke_lbfgs.run_mjstep``."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    cal = cfg["calib"]
    opts = optim_lbfgs.LBFGSOptions(**cfg["options"])
    cs = DomainControlSet(ice_thermo_control_set().controls)
    if list(cs.names) != cal["controls"]:
        raise ValueError(f"{run_dir}: controls {cal['controls']} differ from this checkout's "
                         f"{list(cs.names)}")
    coord = dict(months=[int(w) for w in cal["windows"]], cs=cs, prior=bool(cal["prior"]),
                 scale=1.0, u0=np.asarray(cal["u0"], np.float64), maxiter=int(cal["maxiter"]),
                 maxeval=int(cal["maxeval"]), gtol=0.0, ftol=0.0, options=opts,
                 free=cfg.get("free") or None,
                 config={"calib": cal})
    return cfg, coord


def _eval_dir(run_dir, k: int) -> Path:
    return _lbfgs().mj_dir(run_dir, k)


def window_file(run_dir, k: int, window: int) -> Path:
    return _eval_dir(run_dir, k) / f"win_{int(window):02d}.json"


def latest_posted(run_dir) -> int:
    """The highest k with ``eval_<k>/u.json`` (0 if none)."""
    ks = [int(p.parent.name.split("_")[1]) for p in Path(run_dir).glob("eval_*/u.json")]
    return max(ks) if ks else 0


def _chain_log(run_dir, row: dict) -> None:
    C.append_row(Path(run_dir) / "chain.jsonl",
                 {"time": datetime.datetime.now().isoformat(timespec="seconds"),
                  "slurm_job": os.environ.get("SLURM_JOB_ID"), **row})


def evaluate_model(cfg: dict, window: int, k: int, u, out_dir, _cache={}) -> dict:  # noqa: B006
    """The real window evaluation: ``run_eval`` of the window's model (built once per
    process) into ``out_dir``. Returns ``{"J", "g", ...}`` for the window file."""
    cal_cfg = cfg["calib"]
    w = cal_cfg["window_files"][str(int(window))]
    if window not in _cache:
        n = steps_for(days=cal_cfg["days"])
        _cache[window] = IceCal(w["state"], w["start"], n, obs=w["obs"],
                                chunk_steps=int(cal_cfg["chunk_steps"]),
                                setting=cal_cfg["setting"],
                                ice_adjoint_extra=_extra_of(cal_cfg.get("ice_adjoint")))
    cal = _cache[window]
    if cal.obs_sha256 != w["obs_sha256"]:
        raise ValueError(f"window {window}: {w['obs']} changed since chain-init")
    lin = cal_cfg.get("first_check", "full") if int(k) == 1 else True
    lin = False if lin == "off" else lin
    r = run_eval(cal, out_dir, u, stride=cal_cfg.get("stride"), check_linear=lin,
                 extra={"chain_eval": int(k), "window": int(window)})
    prov = r["provenance"]
    return {"J": r["J_data"], "g": r["g_data"], "terms": r["terms"],
            "platform": prov.get("platform"), "device": prov.get("device"),
            "compile_s": prov.get("compile_s"), "run_s": prov.get("run_s"),
            "check_linear": r["check_linear"]}


def submit_step(run_dir, window: int) -> str:
    """Submit one chain step for ``window`` through :data:`JOB_SCRIPT`; returns the job id or
    raises ``RuntimeError`` with the script's output."""
    p = subprocess.run([str(JOB_SCRIPT), str(run_dir), str(int(window))],
                       capture_output=True, text=True, timeout=300,
                       env={**os.environ, "WT": str(ROOT)})
    if p.returncode != 0:
        raise RuntimeError(f"{JOB_SCRIPT.name} exit {p.returncode}: {p.stdout.strip()} "
                           f"{p.stderr.strip()}")
    return p.stdout.strip().splitlines()[-1]


def _wait_posted(run_dir, k: int, wait_s: float) -> bool:
    up = _eval_dir(run_dir, k) / "u.json"
    t0 = time.time()
    while not up.exists():
        if time.time() - t0 > wait_s or _lbfgs().mj_finished(run_dir) is not None:
            return up.exists()
        time.sleep(2.0)
    return True


def chain_step(run_dir, window: int | None = None, *, evaluate=None, submit=None,
               wait_s: float = POST_WAIT_S) -> int:
    """One chain step (see the module docstring). ``evaluate(cfg, window, k, u, out_dir) ->
    {"J", "g", ...}`` and ``submit(run_dir, window) -> job id`` default to the model evaluation
    and :data:`JOB_SCRIPT`. Returns the exit status."""
    T = _lbfgs()
    run_dir = Path(run_dir)
    evaluate = evaluate_model if evaluate is None else evaluate
    submit = submit_step if submit is None else submit
    cfg, coord = load_chain(run_dir)
    wins = coord["months"]
    if window is None:
        if len(wins) != 1:
            raise ValueError(f"--window is required with several windows {wins}")
        window = wins[0]
    window = int(window)
    if window not in wins:
        raise ValueError(f"window {window} is not one of {wins}")
    if (run_dir / "STOP").exists():
        _chain_log(run_dir, {"window": window, "event": "stop_file", "evaluated": False})
        print(f"[icecal] {run_dir}/STOP exists; nothing evaluated, nothing submitted", flush=True)
        return 0
    if T.mj_finished(run_dir) is not None:
        print(f"[icecal] {run_dir}: the run has stopped ({T.mj_finished(run_dir)})", flush=True)
        return 0

    # the evaluation this window owes: the latest posted one, or the next one if it is complete
    k = latest_posted(run_dir)
    status = 0
    if k == 0 or all(window_file(run_dir, k, w).exists() for w in wins):
        # k = 0: the chain's start (the orchestrator starts one step per window by hand);
        # k > 0: evaluation k is complete but was never coordinated (its last job ended before
        # the coordinator ran): coordinate now, and give the OTHER windows of k+1 their jobs
        res = T._mj_try_coordinate(run_dir, k, coord)
        if res is not None and res.get("status") != "need":
            _chain_log(run_dir, {"window": window, "event": "finished", "result": res})
            return 0
        if res is not None and k > 0:
            status = _submit_round(run_dir, coord, res, submit,
                                   [w for w in res["missing"] if int(w) != window])
        if not _wait_posted(run_dir, k + 1, wait_s):
            if T.mj_finished(run_dir) is not None:
                return 0
            print(f"[icecal] no point posted for evaluation {k + 1} within {wait_s:.0f} s",
                  flush=True)
            return 4
        k += 1
    wp = window_file(run_dir, k, window)
    if wp.exists():
        print(f"[icecal] {wp} exists; this window of evaluation {k} is done", flush=True)
    else:
        up = json.loads((_eval_dir(run_dir, k) / "u.json").read_text())
        u = np.asarray(up["u"], np.float64)
        cs = coord["cs"]
        row = {"eval": k, "month": window, "window": window, "u": u.tolist(),
               "p": (cs.p0 * np.exp(cs.sigma * u)).tolist(),
               "slurm_job": os.environ.get("SLURM_JOB_ID")}
        bad = cs.out_of_domain(u)
        if bad:                       # the coordinator never posts such a point; be safe anyway
            row.update({"J_w": float("nan"), "g_w": [float("nan")] * cs.n,
                        "out_of_domain": bad, "platform": None})
        else:
            od = _eval_dir(run_dir, k) / f"win_{window:02d}.{os.environ.get('SLURM_JOB_ID', 'local')}"
            r = evaluate(cfg, window, k, u, od)
            row.update({"J_w": float(r["J"]), "g_w": [float(x) for x in r["g"]],
                        "eval_dir": str(od),
                        **{kk: v for kk, v in r.items() if kk not in ("J", "g")}})
        publish_json(wp, row)
        _chain_log(run_dir, {"window": window, "eval": k, "event": "window_written",
                             "J_w": row["J_w"]})
    res = T._mj_try_coordinate(run_dir, k, coord)
    if res is None:
        return status
    if res.get("status") != "need":
        _chain_log(run_dir, {"window": window, "eval": k, "event": "finished", "result": res})
        return status
    return max(status, _submit_round(run_dir, coord, res, submit, res["missing"]))


def _submit_round(run_dir, coord, res, submit, windows) -> int:
    """Submit one chain step per window of the posted evaluation ``res["eval"]`` unless
    ``STOP`` exists or it is beyond ``maxeval``. 0, or 3 if a submission failed."""
    k1 = int(res["eval"])
    if (Path(run_dir) / "STOP").exists():
        _chain_log(run_dir, {"eval": k1, "event": "not_submitted", "reason": "STOP"})
        return 0
    if k1 > coord["maxeval"]:
        _chain_log(run_dir, {"eval": k1, "event": "not_submitted", "reason": "maxeval"})
        return 0
    status = 0
    for w in windows:
        try:
            jid = submit(run_dir, int(w))
            _chain_log(run_dir, {"eval": k1, "window": int(w), "event": "submitted",
                                 "job": jid})
        except Exception as exc:  # noqa: BLE001 — record, do not lose the evaluation
            _chain_log(run_dir, {"eval": k1, "window": int(w), "event": "submit_failed",
                                 "error": str(exc)})
            print(f"[icecal] submitting evaluation {k1} window {w} failed: {exc}", flush=True)
            status = 3
    return status


# ---------------------------------------------------------------------------
# command line


def _floats(s) -> list:
    return [float(x) for x in str(s).split(",") if x.strip()]


def _signs(s) -> list:
    out = []
    for tok in str(s).replace("±", "+,-").split(","):
        tok = tok.strip()
        if tok in ("+", "+1", "1", "p"):
            out.append(1)
        elif tok in ("-", "-1", "m"):
            out.append(-1)
        elif tok:
            raise ValueError(f"bad sign {tok!r}")
    return out


def _check_linear(s):
    return {"zero": True, "full": "full", "off": False}[s]


def _window_args(ap):
    ap.add_argument("--state", required=True, help="start State (load_state format)")
    ap.add_argument("--start", required=True, help="window start (the State's date)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--days", type=float)
    g.add_argument("--steps", type=int)
    ap.add_argument("--obs", required=True, help="obs_<YYYY>.npz of prep_obs.py")
    ap.add_argument("--u", default=None, help="control vector (JSON list / object / file)")
    ap.add_argument("--chunk-steps", type=int, default=CHUNK_STEPS)
    ap.add_argument("--setting", default=SETTING)
    ap.add_argument("--controls-set", choices=tuple(CONTROL_SETS), default="thermo",
                    help="thermo: the ten thermodynamic controls; joint: + pstar, cd_oce_ice, "
                         "cd_atm_ice")
    _ice_args(ap)


def _ice_args(ap):
    ap.add_argument("--ice-cuts", default=None,
                    help="IceConfig.adjoint_cuts, e.g. fct_flux+strength_conc (derivative-only)")
    ap.add_argument("--ice-thin-mass", type=float, default=None,
                    help="IceConfig.adjoint_thin_mass [kg/m2] (mEVP load derivative cut on "
                         "thin ice)")
    ap.add_argument("--ice-sweep-guard", type=float, default=None,
                    help="IceConfig.adjoint_sweep_guard (drop an ice step's mEVP derivative whose "
                         "transposed sweeps grow past this factor)")


def _ice_extra(args) -> dict:
    """The ``--ice-*`` options given, as ``ice_adjoint`` extra fields."""
    out = {}
    if args.ice_cuts is not None:
        out["adjoint_cuts"] = tuple(c for c in args.ice_cuts.split("+") if c and c != "none")
    if args.ice_thin_mass is not None:
        out["adjoint_thin_mass"] = float(args.ice_thin_mass)
    if args.ice_sweep_guard is not None:
        out["adjoint_sweep_guard"] = float(args.ice_sweep_guard)
    return out


def _extra_of(stored) -> dict:
    """The extra fields of a chain's stored ``ice_adjoint`` (older chains stored ICE_ADJOINT)."""
    stored = dict(stored or ICE_ADJOINT)
    base = {k: stored.pop(k) for k in list(stored) if k in ICE_ADJOINT}
    if base != ICE_ADJOINT:
        raise ValueError(f"the chain's ice adjoint {base} is not this driver's {ICE_ADJOINT}")
    return stored


def _parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--run-dir", required=True)
    _window_args(e)
    e.add_argument("--stride", type=int, default=None)
    e.add_argument("--check-linear", choices=("zero", "full", "off"), default="zero")
    e.add_argument("--terms", default=None,
                   help="comma list of cost terms (J_C_NH,J_C_SH,J_H_NH): one gradient each")
    e.add_argument("--cot-maps", action="store_true",
                   help="write the ice cotangent maps at every chunk boundary (needs --terms)")
    f = sub.add_parser("fwd")
    f.add_argument("--run-dir", required=True)
    _window_args(f)
    f.add_argument("--save-state-at", default=None)
    b = sub.add_parser("fdbatch")
    b.add_argument("--root", required=True)
    _window_args(b)
    b.add_argument("--controls", required=True, help="comma list of control names, or all")
    b.add_argument("--hs", default="1e-2,3e-2")
    b.add_argument("--signs", default="+,-")
    b.add_argument("--base-repeats", type=int, default=0)
    ci = sub.add_parser("chain-init")
    ci.add_argument("--run-dir", required=True)
    ci.add_argument("--windows", required=True, help="comma list of years")
    ci.add_argument("--mode", choices=("twin", "real"), required=True)
    ci.add_argument("--prior", choices=("on", "off"), required=True)
    ci.add_argument("--maxeval", type=int, default=15)
    ci.add_argument("--maxiter", type=int, default=None)
    ci.add_argument("--truth", default=None, help="twin truth u (JSON list / object / file)")
    ci.add_argument("--u0", default=None)
    ci.add_argument("--states-dir", default=str(STATES_DIR))
    ci.add_argument("--obs-dir", default=str(OBS_DIR))
    ci.add_argument("--state", action="append", default=[], help="YEAR=PATH override")
    ci.add_argument("--obs", action="append", default=[], help="YEAR=PATH override")
    ci.add_argument("--days", type=float, default=365.0)
    ci.add_argument("--chunk-steps", type=int, default=CHUNK_STEPS)
    ci.add_argument("--stride", type=int, default=None)
    ci.add_argument("--first-check", choices=("zero", "full", "off"), default="full")
    ci.add_argument("--free", default=None,
                    help="comma list of the controls to optimise (default all; the rest stay at u0)")
    ci.add_argument("--fmin-frac", type=float, default=None,
                    help="expected first decrease / J (cold-start step length; default 0.002)")
    _ice_args(ci)
    cs_ = sub.add_parser("chain-step")
    cs_.add_argument("--run-dir", required=True)
    cs_.add_argument("--window", type=int, default=None)
    return ap.parse_args(argv)


def _overrides(items) -> dict:
    out = {}
    for it in items:
        y, eq, p = it.partition("=")
        if not eq:
            raise ValueError(f"expected YEAR=PATH, got {it!r}")
        out[int(y)] = p
    return out


def main(argv=None) -> int:
    args = _parse(argv)
    if args.action == "chain-init":
        cfg = chain_init(args.run_dir, [int(y) for y in args.windows.split(",") if y.strip()],
                         mode=args.mode, prior=args.prior == "on", maxeval=args.maxeval,
                         maxiter=args.maxiter, truth=args.truth,
                         states=_overrides(args.state), obs=_overrides(args.obs),
                         states_dir=args.states_dir, obs_dir=args.obs_dir, days=args.days,
                         chunk_steps=args.chunk_steps, stride=args.stride,
                         u0=None if args.u0 is None else parse_u(args.u0, 10),
                         first_check=args.first_check, ice_adjoint_extra=_ice_extra(args),
                         free=None if args.free is None
                         else [c.strip() for c in args.free.split(",") if c.strip()],
                         fmin_frac=args.fmin_frac)
        print(json.dumps(cfg["calib"]["window_files"]), flush=True)
        return 0
    if args.action == "chain-step":
        return chain_step(args.run_dir, args.window)
    # refuse an existing output BEFORE the (minutes-long) model build
    if args.action in ("eval", "fwd"):
        if Path(out_path(args.run_dir)).exists():
            raise FileExistsError(f"{args.run_dir} exists; every run writes a new directory")
    n = steps_for(days=args.days, steps=args.steps)
    cal = IceCal(args.state, args.start, n, obs=args.obs, chunk_steps=args.chunk_steps,
                 setting=args.setting, ice_adjoint_extra=_ice_extra(args),
                 cs=CONTROL_SETS[args.controls_set]())
    u = parse_u(args.u, cal.cs.n, cal.cs.names)
    if args.action == "eval":
        r = run_eval(cal, args.run_dir, u, stride=args.stride,
                     check_linear=_check_linear(args.check_linear),
                     terms=None if args.terms is None
                     else [x.strip() for x in args.terms.split(",") if x.strip()],
                     cot_maps=args.cot_maps)
        print(json.dumps({"J_data": r["J_data"], "g_data": r["g_data"],
                          "wall_s": r["provenance"].get("wall_s"),
                          "n_compiles": r["provenance"].get("n_compiles"),
                          "host_vmhwm_gb": r["provenance"].get("host_vmhwm_gb")}), flush=True)
    elif args.action == "fwd":
        r = run_fwd(cal, args.run_dir, u, save_state_at=args.save_state_at)
        print(json.dumps({"J_data": r["J_data"], "saved_state": r["saved_state"]}), flush=True)
    else:
        run_fdbatch(cal, args.root, u, [c for c in args.controls.split(",") if c.strip()],
                    _floats(args.hs), _signs(args.signs), base_repeats=args.base_repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
