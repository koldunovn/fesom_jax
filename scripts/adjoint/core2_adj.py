#!/usr/bin/env python
"""The shared CORE2 adjoint driver: one model, one compiled chunk program per process, and the
actions every adjoint study repeats.

    python scripts/adjoint/core2_adj.py ACTION [options]      # ACTION: fwd grad screen fd tl dot
                                                             #         states restartcheck

Run it from a checkout with ``PYTHONPATH=<checkout>`` (``scripts/adjoint/job_cpu.sbatch`` /
``job_gpu.sbatch`` do that). The import guard below refuses to run against any other copy of
``fesom_jax``: the package is an editable install that may point at a different working tree.

The model
    :func:`build_model` assembles the production CORE2 configuration from
    ``configs/core2_full.yaml`` of THIS checkout (zstar + classical TKE + mEVP sea ice + GM,
    dt = 1800 s, JRA55-do forcing; asserted by :func:`assert_physics`), dense on one device, from
    a restart (the hindcast archive, or a State saved by the ``states`` action). It builds ONE
    ``chunk_fn(carry, xs, params) -> carry`` of ``chunk_steps`` steps. Everything that changes
    between windows or iterations — the start state, the dates and their forcing, the cost —
    goes in through ``init_state`` (:meth:`Model.init_carry`), ``xs_fn`` (:meth:`Model.xs_fn`)
    and ``loss_fn``; parameters are strong float64 arrays of a fixed pytree. So a second window,
    or an optimiser iteration, reuses the compiled programs: the library caches them on the
    ``chunk_fn`` object (:func:`fesom_jax.adjoint.chunked_grad`), and a new closure per window
    would recompile the CORE2 VJP (minutes) every time.

    The carry is the ``State``, or ``(State, acc)`` when running accumulators are requested
    (:func:`fesom_jax.adjoint_costs.with_running_costs`, e.g. ``{"mld": [nod2D]}`` for a
    window-mean MLD). The differentiated ``params`` pytree is, by default, a dict of log
    multipliers ``{"ln_k_gm", "ln_tke_c_k", "ln_tke_c_eps", "ln_tke_cd", "ln_tke_alpha"}`` (0 =
    the configured value), plus ``"surface_adj": {"heat", "stress_x", "stress_y"}`` ``[nod2D]``
    fields when the model is built with ``surface_adj=True`` (``Params.surface_adj``: heat in
    W/m² POSITIVE INTO the ocean, stress in N/m²). A caller with its own control vector passes
    ``params0`` and ``to_params(params) -> Params``.

Costs, boxes, directions
    :data:`BOXES` names lon/lat/depth boxes (:data:`DEFAULT_BOX` = ``natl_subpolar``);
    :func:`cost_fn` resolves a registered cost (:data:`COSTS`) on a box into a loss on the carry,
    or passes a caller's callable through. :func:`direction` returns the named tangent
    directions (:data:`DIRECTIONS`) as full carry / params tangents.

Actions (each writes one JSONL row per result with :data:`PROVENANCE_KEYS`)
    ``fwd``      the window forward; cost values; compile and run seconds.
    ``grad``     :func:`~fesom_jax.adjoint.chunked_grad` (one cost; ``boundary_stride``, the
                 linearity guard, a resumable sweep via :class:`DiskCheckpoint`) or
                 :func:`~fesom_jax.adjoint.chunked_grad_multi` (several costs, ``on_cot(k, c,
                 cot)``).
    ``screen``   a terminal-seed sweep with per-field traces and an overflow abort (trace >
                 ``abort_at`` = 1e250, or non-finite): the row records "overflow at day d".
    ``fd``       central differences along a named direction, all evaluations in one process
                 (same compiled forward), optional autodiff value for the comparison.
    ``tl``       the window tangent-linear map along a direction, on PINNED chunk-entry states
                 (:func:`fesom_jax.adjoint_check.window_jvp`, chunk program without remat,
                 compiled without XLA constant folding: :data:`TL_COMPILER_OPTIONS`; 27 s
                 instead of 761 s for a CORE2 step on CPU).
    ``dot``      the pinned-state dot test ``<y, M x>`` vs ``<Mᵀ y, x>`` with ``y`` = the cost's
                 seed at the window end, at several amplitudes.
    ``states``   a forward that writes full States (``zarr_output.write_restart``, tag
                 ``fesom.<YYYY>.<DOY>.<SSSSS>``, loadable by ``read_restart``) on the 1st of each
                 month, daily 2-D diagnostics (:data:`DAILY_FIELDS`), and a ``README.json``.
    ``restartcheck``  restart from a saved State, rerun the next day twice, compare with the
                 continuous run's daily diagnostics (and the next saved State if present);
                 a difference passes up to ``--tie-margin`` (1.05) × the repeat spread.

Nothing here deletes or overwrites: an existing restart directory or daily file is an error
(``--skip-existing`` skips it instead), and JSONL rows are appended.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import importlib.util
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "core2_full.yaml"
#: the CORE2 production configuration whose physics ``build_model(fast=...)`` takes piece by piece
FAST_CONFIG = ROOT / "configs" / "core2_gpu_fast.yaml"


def _check_import_root() -> None:
    """Refuse to run against a ``fesom_jax`` that is not this checkout's (the editable install
    may point at another working tree; ``PYTHONPATH=<checkout>`` is required)."""
    spec = importlib.util.find_spec("fesom_jax")
    origin = os.path.realpath(spec.origin) if (spec is not None and spec.origin) else None
    root = os.path.realpath(ROOT) + os.sep
    if origin is None or not origin.startswith(root):
        raise ImportError(
            f"core2_adj: fesom_jax resolves to {origin}, not to this checkout {root}. "
            f"Run with PYTHONPATH={os.path.realpath(ROOT)} (the job scripts in scripts/adjoint/ "
            f"set it).")


_check_import_root()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fesom_jax import adjoint, adjoint_check, adjoint_stab, ale, obs_compare, partit  # noqa: E402
from fesom_jax import shard_mesh, ssh, ssh_se  # noqa: E402
from fesom_jax import adjoint_costs as AC  # noqa: E402
from fesom_jax import ice_step as _ice_step  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax import surface_forcing, zarr_output  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.params import Params  # noqa: E402
from fesom_jax.run_config import load_yaml  # noqa: E402
from fesom_jax.state import State  # noqa: E402

RESTART_ARCHIVE = Path("/work/ab0995/a270088/port_jax/runs/core2_hindcast_v3/restart_archive")
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")   #: the dist_<N> partitions (build_model devices)
STATES_DIR = Path("/work/ab0995/a270088/fesom_jax_dev/states_2013")
RUNS_DIR = Path("/work/ab0995/a270088/fesom_jax_dev/runs/p0")
DT = 1800.0
ICE_MODES = ("prescribed", "rheology", "frozen", "exact", "dynamics")
#: the IceConfig fields ``build_model(ice_adjoint=...)`` may set (derivative-only, mEVP sweeps)
ICE_ADJOINT_FIELDS = ("adjoint_damping", "adjoint_kappa", "adjoint_sweeps", "adjoint_cuts",
                      "adjoint_alpha_kappa", "adjoint_theta", "adjoint_thin_mass",
                      "adjoint_sweep_guard")
#: ``build_model(fast=...)`` names, each one piece of :data:`FAST_CONFIG`'s physics on top of the reference:
#: ``ice2`` its ``ice.ice_ave_steps`` (ice every second step), ``se`` its ``ssh.split_explicit``, ``dt2700`` its
#: ``dt``, ``sp`` its ``precision``. The timestep piece may name another dt: ``dt<seconds>`` (``dt2400`` = 40 min;
#: Nikolay 2026-10-22: 45 min falls apart, 40 or 30 min may be the stabler choice for the adjoint)
FAST_PHYSICS = ("ice2", "se", "dt2700", "sp")
#: the pieces build_model runs so far (the fast-adjoint plan, docs/plans/20261022-core2-fast-adjoint.md)
FAST_IMPLEMENTED = ("ice2", "se", "dt2700", "sp")
ABORT_AT = 1e250
#: XLA options of the tangent-linear chunk program (``tl``/``dot``): without constant folding it
#: compiles in ~30 s instead of ~13 min on CORE2 (see ``adjoint_check.NO_CONSTANT_FOLDING``).
TL_COMPILER_OPTIONS = adjoint_check.NO_CONSTANT_FOLDING

__all__ = ["Box", "BOXES", "DEFAULT_BOX", "box_masks", "Model", "build_model", "assert_physics",
           "COSTS", "cost_fn", "ACCUMULATORS", "DIRECTIONS", "DIRECTION_UNITS", "Direction",
           "direction", "LOG_CONTROLS", "default_params0", "default_to_params",
           "run_fwd", "run_grad", "run_screen", "run_fd", "run_tl", "run_dot", "run_states",
           "run_restartcheck", "RESTART_TIE_MARGIN", "DiskCheckpoint", "PROVENANCE_KEYS",
           "provenance", "append_row", "load_state", "restart_tag", "compile_clock", "DAILY_FIELDS"]


# ---------------------------------------------------------------------------
# compile time, measured from JAX's own compile events

_TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"
_LOWER_EVENT = "/jax/core/compile/jaxpr_to_mlir_module_duration"
_BACKEND_EVENT = "/jax/core/compile/backend_compile_duration"
_CLOCK = {"trace_s": 0.0, "lower_s": 0.0, "backend_s": 0.0, "n_backend": 0,
          "max_backend_s": 0.0}


def _on_duration(event, duration, **_kw):
    if event == _TRACE_EVENT:
        _CLOCK["trace_s"] += float(duration)
    elif event == _LOWER_EVENT:
        _CLOCK["lower_s"] += float(duration)
    elif event == _BACKEND_EVENT:
        _CLOCK["backend_s"] += float(duration)
        _CLOCK["n_backend"] += 1
        _CLOCK["max_backend_s"] = max(_CLOCK["max_backend_s"], float(duration))


jax.monitoring.register_event_duration_secs_listener(_on_duration)


def compile_clock() -> dict:
    """A copy of the process-wide compile counters (seconds spent tracing, lowering and in the
    backend compiler; number of backend compiles; the longest one). Differences of two copies
    give the compile cost of whatever ran in between."""
    return dict(_CLOCK)


class _Timer:
    """Wall seconds of a block, split into compile (JAX compile events) and the rest."""

    def __enter__(self):
        self.c0, self.t0 = compile_clock(), time.time()
        return self

    def __exit__(self, *exc):
        self.wall = time.time() - self.t0
        c1 = compile_clock()
        self.compile_s = sum(c1[k] - self.c0[k] for k in ("trace_s", "lower_s", "backend_s"))
        self.n_compiles = c1["n_backend"] - self.c0["n_backend"]
        self.run_s = self.wall - self.compile_s
        return False

    def fields(self) -> dict:
        return {"wall_s": self.wall, "compile_s": self.compile_s, "run_s": self.run_s,
                "n_compiles": self.n_compiles}


# ---------------------------------------------------------------------------
# boxes


class Box(NamedTuple):
    """A lon/lat/depth box: degrees north, degrees east in [-180, 180] (``lon_min > lon_max``
    wraps across the date line), depths in metres, positive down, applied to the static
    mid-layer depths."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    z_min: float
    z_max: float


#: Named boxes; tracks may add their own (``BOXES["name"] = Box(...)``) before building costs.
BOXES: dict = {
    "natl_subpolar": Box(50.0, 60.0, -40.0, -20.0, 0.0, 500.0),
    "eq_pacific": Box(-5.0, 5.0, -170.0, -120.0, 0.0, 300.0),
}
DEFAULT_BOX = "natl_subpolar"


def _box(box) -> Box:
    return BOXES[box] if isinstance(box, str) else box


def _lonlat_deg(mesh):
    lon = np.degrees(np.asarray(mesh.geo_coord_nod2D[:, 0]))
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    lat = np.degrees(np.asarray(mesh.geo_coord_nod2D[:, 1]))
    return lon, lat


def box_masks(mesh, box=DEFAULT_BOX):
    """``(node_mask [nod2D], level_mask [nl])`` host bool arrays of ``box``: wet surface nodes
    inside the lon/lat bounds, and the levels whose static mid-depth lies in [z_min, z_max]."""
    b = _box(box)
    lon, lat = _lonlat_deg(mesh)
    in_lon = ((lon >= b.lon_min) & (lon <= b.lon_max) if b.lon_min <= b.lon_max
              else (lon >= b.lon_min) | (lon <= b.lon_max))
    node = in_lon & (lat >= b.lat_min) & (lat <= b.lat_max) & np.asarray(mesh.node_layer_mask[:, 0])
    zmid = -np.asarray(mesh.Z)                                   # [nl-1], positive down
    level = np.zeros(int(mesh.nl), bool)
    level[:zmid.size] = (zmid >= b.z_min) & (zmid <= b.z_max)
    if not node.any() or not level.any():
        raise ValueError(f"box {b} selects no wet node or no level on this mesh")
    return node, level


# ---------------------------------------------------------------------------
# parameters


#: Params constants exposed as log multipliers in the default control pytree.
LOG_CONTROLS = ("k_gm", "tke_c_k", "tke_c_eps", "tke_cd", "tke_alpha")
SURFACE_ADJ_KEYS = ("heat", "stress_x", "stress_y")


def default_params0(nod2D: int, surface_adj: bool = False) -> dict:
    """The default control pytree at the configured values: ``ln_<name>`` = 0 (strong float64
    scalars) for :data:`LOG_CONTROLS`, and zero ``[nod2D]`` surface adjustments if requested."""
    u = {f"ln_{n}": np.zeros((), np.float64) for n in LOG_CONTROLS}
    if surface_adj:
        u["surface_adj"] = {k: np.zeros(int(nod2D), np.float64) for k in SURFACE_ADJ_KEYS}
    return u


def default_to_params(u) -> Params:
    """``Params`` from the default control pytree: ``p = p_default · exp(ln_p)``; the
    ``surface_adj`` dict (if present) is passed through."""
    base = Params.defaults()
    repl = {n: getattr(base, n) * jnp.exp(u[f"ln_{n}"]) for n in LOG_CONTROLS}
    if "surface_adj" in u:
        repl["surface_adj"] = dict(u["surface_adj"])
    if "forcing_adj" in u:                      # the monthly atmospheric maps [M, nod2D] (the step takes one month)
        repl["forcing_adj"] = dict(u["forcing_adj"])
    return dataclasses.replace(base, **repl)


def _strong_f64(tree):
    """Host copy of ``tree`` with every floating leaf a float64 numpy array (a weakly typed
    Python scalar would recompile the chunk program when it changes type)."""
    def one(x):
        a = np.asarray(x)
        return a.astype(np.float64) if np.issubdtype(a.dtype, np.floating) else a
    return jax.tree.map(one, tree)


# ---------------------------------------------------------------------------
# restarts


def restart_tag(when: datetime.datetime) -> str:
    """``fesom.<YYYY>.<DOY>.<SSSSS>`` of a model time (the hindcast archive's naming)."""
    doy = (when - datetime.datetime(when.year, 1, 1)).days + 1
    sec = when.hour * 3600 + when.minute * 60 + when.second
    return f"fesom.{when.year:04d}.{doy:03d}.{sec:05d}"


def _parse_tag(tag: str) -> datetime.datetime:
    _, y, d, s = Path(tag).name.split(".")
    return (datetime.datetime(int(y), 1, 1) + datetime.timedelta(days=int(d) - 1,
                                                                  seconds=int(s)))


def _parse_date(s) -> datetime.datetime:
    if isinstance(s, datetime.datetime):
        return s
    return datetime.datetime.fromisoformat(str(s))


def _resolve_restart(start: datetime.datetime, restart=None) -> Path:
    """The restart directory for ``start``: ``restart`` if it is an existing path, else the tag
    (given, or derived from ``start``) looked up in :data:`STATES_DIR` then
    :data:`RESTART_ARCHIVE`."""
    if restart is not None and Path(restart).is_dir():
        return Path(restart)
    tag = restart if restart is not None else restart_tag(start)
    for base in (STATES_DIR, RESTART_ARCHIVE):
        if (base / tag).is_dir():
            return base / tag
    raise FileNotFoundError(f"no restart {tag} in {STATES_DIR} or {RESTART_ARCHIVE}")


def load_state(path):
    """``(State, meta)``: a full dense State (host numpy) and the restart attributes of a
    :func:`~fesom_jax.zarr_output.write_restart` store (any layout, any device count)."""
    import zarr
    from fesom_jax.precision import CARRY_FIELDS
    from fesom_jax.state import BGC_FIELDS, ICE_AVG_FIELDS, SE_FIELDS
    root = zarr.open_group(str(path), mode="r")
    # as zarr_output.read_restart: the single-precision carry, the ice-cycle fields and the SE rings are in the
    # store only when the writing run used them (None otherwise; build_model sets them for its own physics)
    optional = CARRY_FIELDS + ICE_AVG_FIELDS + SE_FIELDS + BGC_FIELDS
    st = State(**{f: None if f in optional and f not in root
                  else np.asarray(zarr_output.reconstruct_global(path, f))
                  for f in zarr_output._all_state_fields() + ICE_AVG_FIELDS + SE_FIELDS + BGC_FIELDS})
    meta = dict(root.attrs)
    return st, meta


def _check_restart_dt(rpath: Path, meta: dict, dt: float) -> None:
    """A restart's AB2 history (``uv_rhsAB``, ``T_old``) was formed at its ``dt_stage``; every step of a window
    continues AB2, so the window's dt must be that one (a run at another dt starts from a restart its production
    driver wrote after re-bootstrapping AB2, ``run.py``)."""
    if "dt_stage" in meta and float(meta["dt_stage"]) != float(dt):
        raise ValueError(f"restart {rpath} was written at dt {float(meta['dt_stage']):g} s, the window runs at "
                         f"dt {float(dt):g} s: start from a restart written by a dt-{float(dt):g} run")


def _check_restart_date(rpath: Path, meta: dict, start: datetime.datetime) -> None:
    """The restart's calendar date (``YYYY-doyDDD`` attribute, and the seconds of the day from
    a ``fesom.*`` tag name) must be the window start."""
    doy = (start - datetime.datetime(start.year, 1, 1)).days + 1
    if "calendar_date" in meta:
        y, d = str(meta["calendar_date"]).split("-doy")
        if (int(y), int(d)) != (start.year, doy):
            raise ValueError(f"restart {rpath} is at {meta['calendar_date']}, not {start}")
    if rpath.name.startswith("fesom.") and _parse_tag(rpath.name) != start:
        raise ValueError(f"restart {rpath.name} is not at {start}")


def _write_state(out_dir: Path, state, mesh, *, step: int, when: datetime.datetime,
                 attrs: dict | None = None) -> Path:
    """Write a dense host State as a portable restart (``layout='global'``) via a serial
    partition. Refuses an existing directory (nothing is overwritten)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; restarts are never overwritten")
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    sp = shard_mesh.partition_state(jax.device_get(state), part)          # [1, Lmax, ...]
    folded = jax.tree.map(lambda a: np.asarray(a).reshape((-1,) + np.shape(a)[2:]), sp)
    doy = (when - datetime.datetime(when.year, 1, 1)).days + 1
    zarr_output.write_restart(out_dir, folded, sm, part, step=int(step),
                              calendar_date=f"{when.year:04d}-doy{doy:03d}", dt_stage=DT,
                              attrs=attrs)
    return out_dir


# ---------------------------------------------------------------------------
# the model


def parse_tke_settings(spec) -> dict:
    """``"name=value+name=value"``, a dict, or ``None``/``"none"``/``""`` -> a dict of
    ``TkeConfig`` fields (values typed like the field's default). Unknown fields raise."""
    from fesom_jax.tke import TkeConfig
    if spec is None:
        return {}
    if isinstance(spec, Mapping):
        d = dict(spec)
    else:
        text = str(spec).strip()
        if text in ("", "none"):
            return {}
        d = {}
        for tok in text.split("+"):
            name, eq, val = tok.partition("=")
            if not eq:
                raise ValueError(f"tke setting {tok!r} is not name=value")
            d[name.strip()] = val.strip()
    bad = [k for k in d if k not in TkeConfig._fields]
    if bad:
        raise ValueError(f"unknown TkeConfig field(s) {bad}")
    out = {}
    for k, v in d.items():
        default = getattr(TkeConfig(), k)
        if isinstance(v, str) and not isinstance(default, str):
            v = (v.lower() in ("1", "true", "yes") if isinstance(default, bool)
                 else int(v) if isinstance(default, int) else float(v))
        out[k] = v
    return out


def _piece_dt(name: str):
    """The timestep of a ``dt<seconds>`` piece name, else None."""
    m = re.fullmatch(r"dt([1-9][0-9]*)", name)
    return float(m.group(1)) if m else None


def parse_fast(spec) -> tuple:
    """``"ice2+se"`` (or a sequence of names, or None) -> the :data:`FAST_PHYSICS` names in their canonical
    order, a ``dt<seconds>`` piece in the timestep's place; unknown names and two timesteps are refused."""
    if spec is None or spec == "" or spec == "none":
        return ()
    names = [x for x in spec.split("+") if x] if isinstance(spec, str) else list(spec)
    dts = [n for n in names if _piece_dt(n) is not None]
    bad = sorted(set(names) - set(FAST_PHYSICS) - set(dts))
    if bad:
        raise ValueError(f"fast physics: unknown {bad}; known {FAST_PHYSICS} (a timestep as dt<seconds>)")
    if len(set(dts)) > 1:
        raise ValueError(f"fast physics: two timesteps {sorted(set(dts))}")
    return tuple((dts[0] if n == "dt2700" else n) for n in FAST_PHYSICS
                 if (n in names if n != "dt2700" else bool(dts)))


def fast_dt(fast) -> float:
    """The window's timestep [s]: the ``dt<seconds>`` piece's (``dt2700`` = :data:`FAST_CONFIG`'s), else :data:`DT`.
    The tools turn window lengths in days into steps with it before the model is built."""
    d = [_piece_dt(n) for n in parse_fast(fast) if _piece_dt(n) is not None]
    return d[0] if d else DT


def with_fast_physics(cfg, fast):
    """``cfg`` with the :data:`FAST_CONFIG` values of the pieces named in ``fast`` (:func:`parse_fast`)."""
    fast = parse_fast(fast)
    if not fast:
        return cfg
    f = load_yaml(FAST_CONFIG)
    kw = {}
    if "ice2" in fast:
        kw["ice"] = cfg.ice._replace(ice_ave_steps=f.ice.ice_ave_steps)
    if "se" in fast:
        # the barotropic substep of the fast configuration (dt 2700 / M 60 = 45 s, at its CFL guard's bound): at
        # another window dt the substep count scales with it, M = ceil(M_fast dt / dt_fast) (dt 1800: M 40, the
        # guard's M_min there), not the fast configuration's M (Nikolay 2026-10-22)
        se = dict(f.ssh["split_explicit"])
        se["M"] = int(math.ceil(int(se["M"]) * fast_dt(fast) / float(f.dt) - 1e-9))
        kw["ssh"] = {**(cfg.ssh or {}), "split_explicit": se}
    if fast_dt(fast) != DT:
        kw["dt"] = fast_dt(fast)
    if "sp" in fast:
        kw["precision"] = f.precision
    return dataclasses.replace(cfg, **kw)


def assert_physics(cfg, fast=()) -> None:
    """The reference configuration: zstar, classical TKE (no KPP), GM, mEVP sea ice every step, dt 1800 s,
    JRA55-do forcing, the CG SSH solve with the default preconditioner, no dt ramp, double precision; each piece
    named in ``fast`` (:data:`FAST_PHYSICS`) may instead be :data:`FAST_CONFIG`'s."""
    fast = parse_fast(fast)
    bad = []
    if cfg.ale is None:
        bad.append("ale (zstar) is off")
    if cfg.tke is None or cfg.kpp is not None:
        bad.append("mixing is not classical TKE")
    if cfg.gm is None:
        bad.append("GM is off")
    if cfg.ice is None or cfg.ice.whichEVP != 1:
        bad.append("sea ice is not mEVP")
    elif int(cfg.ice.ice_ave_steps) != 1 and "ice2" not in fast:
        bad.append(f"sea ice every {cfg.ice.ice_ave_steps} steps (fast piece ice2)")
    fcfg = with_fast_physics(cfg, fast) if fast else None
    if fcfg is not None and "ice2" in fast and cfg.ice is not None \
            and int(cfg.ice.ice_ave_steps) != int(fcfg.ice.ice_ave_steps):
        bad.append(f"ice_ave_steps {cfg.ice.ice_ave_steps}, not the fast configuration's")
    want_dt = fast_dt(fast)
    if float(cfg.dt) != want_dt or cfg.dt_ramp is not None:
        bad.append(f"dt {cfg.dt} / ramp {cfg.dt_ramp}")
    if cfg.split_explicit() is not None and "se" not in fast:
        bad.append("split-explicit SSH (fast piece se)")
    if "se" in fast and cfg.split_explicit() != fcfg.split_explicit():
        bad.append(f"split_explicit {cfg.split_explicit()}, not the fast configuration's")
    if (cfg.forcing or {}).get("kind") != "core2":
        bad.append(f"forcing {cfg.forcing}")
    if cfg.ssh_cheb()[0] != 0:
        bad.append("Chebyshev SSH preconditioner on")
    if getattr(cfg, "precision", "double") != ("single" if "sp" in fast else "double"):
        bad.append(f"precision {cfg.precision} (fast piece sp)")
    if bad:
        raise AssertionError("core2_adj: not the reference physics: " + "; ".join(bad))


@dataclasses.dataclass(eq=False)
class Model:
    """One CORE2 model and its compiled-program closures. Build with :func:`build_model`.

    ``chunk_fn`` and ``chunk_fn_tl`` are created ONCE per model; pass them to the library
    functions unchanged (their jitted programs are cached on these objects)."""

    cfg: Any
    config_path: Path
    mesh: Any
    op: Any
    forcing: Any
    phys: dict
    adjoint_cfg: AdjointConfig
    ice_mode: str
    start: datetime.datetime
    restart: Path
    restart_meta: dict
    s0: Any                          #: the start State (host numpy)
    n_steps: int
    chunk_steps: int
    save_names: tuple
    rematerialize: bool
    accumulators: dict               #: name -> fn(State) (empty: the carry is the State)
    params0: Any                     #: control pytree (host, strong float64)
    to_params: Callable
    surface_adj: bool
    tke_settings: dict = dataclasses.field(default_factory=dict)   #: TkeConfig forward options
    fast: tuple = ()                 #: the FAST_PHYSICS pieces in the model (empty: the reference physics)
    shard: Any = None                #: fesom_jax.adjoint_sharded.ShardedChunk (devices > 1), else None (dense)
    #: the per-step schedule in xs (None: plain forcing): {"n_months": M (the forcing-control maps' months),
    #: "cost_from": the first window step of the cost (None: every step), "cost_to": the step after its last}
    schedule: Any = None
    acc_factories: dict = dataclasses.field(default_factory=dict)   #: name -> factory(mesh) -> fn (sharded)
    chunk_fn: Callable = None
    forcing_seconds: float = 0.0
    _chunk_fn_tl: Callable = None
    _forcing_lock: Any = dataclasses.field(default_factory=threading.Lock, repr=False)   #: see stack_forcing

    @property
    def dt(self) -> float:
        return float(self.cfg.dt)

    @property
    def n_chunks(self) -> int:
        return self.n_steps // self.chunk_steps

    def chunks_for(self, n_steps: int) -> int:
        if n_steps % self.chunk_steps:
            raise ValueError(f"n_steps {n_steps} is not a multiple of chunk_steps "
                             f"{self.chunk_steps} (a shorter last chunk would compile a second "
                             f"program)")
        return n_steps // self.chunk_steps

    @property
    def chunk_fn_tl(self) -> Callable:
        """The same chunk without rematerialisation, for forward mode (built once)."""
        if self._chunk_fn_tl is None:
            self._chunk_fn_tl = _make_chunk_fn(self, rematerialize=False)
        return self._chunk_fn_tl

    @property
    def devices(self) -> int:
        return 1 if self.shard is None else int(self.shard.npes)

    def init_carry(self, state=None, *, host: bool = False):
        """The chunk carry at a window start: ``state`` (default: the model's start State, a global State) with
        zero accumulators if any. On the device unless ``host``. Sharded: the folded State and folded sums."""
        st = self.s0 if state is None else state
        if self.shard is not None:
            from fesom_jax import adjoint_sharded
            fst = self.shard.fold_state(st)
            if self.accumulators:
                shapes = {k: jax.eval_shape(fn, st).shape for k, fn in self.accumulators.items()}
                carry = (fst, adjoint_sharded.init_accumulators(self.shard, shapes))
            else:
                carry = fst
            return carry if host else self.shard.place(carry)
        if self.accumulators:
            acc = {k: np.zeros(s.shape, s.dtype)
                   for k, s in AC.init_accumulators(self.accumulators, st).items()}
            carry = (st, acc)
        else:
            carry = st
        return jax.device_get(carry) if host else jax.device_put(carry)

    def state_of(self, carry):
        return carry[0] if self.accumulators else carry

    def date(self, start: datetime.datetime, n: int) -> tuple:
        """Forcing date ``(year, doy, sec, month)`` of step ``n`` (0-based) after ``start``."""
        d = start + datetime.timedelta(seconds=n * self.dt)
        doy = (d - datetime.datetime(d.year, 1, 1)).days + 1
        return (d.year, doy, d.hour * 3600.0 + d.minute * 60.0 + d.second, d.month)

    def stack_forcing(self, dates):
        """Host ``StepForcing`` stack for a list of dates, rolling the JRA reader across years. One build at a time
        (the reader is stateful, and :class:`_XsAhead` builds on a worker thread); with ``$FESOM_FORCING_PROCS``
        read-ahead workers each year's JRA records of the list are decoded in parallel first."""
        with self._forcing_lock:
            t0 = time.time()
            steps = []
            for i, d in enumerate(dates):
                if int(d[0]) != int(self.forcing.jra.year):
                    self.forcing.reopen_year(int(d[0]))
                if i == 0 or int(d[0]) != int(dates[i - 1][0]):
                    self.forcing.jra.read_ahead([e for e in dates[i:] if int(e[0]) == int(d[0])])
                steps.append(self.forcing.step_forcing(*d, xp=np))
            out = surface_forcing.StepForcing(*[None if leaves[0] is None else np.stack(leaves, axis=0)
                                                for leaves in zip(*steps)])
            self.forcing_seconds += time.time() - t0
        return out

    def xs_fn(self, start=None) -> Callable[[int], Any]:
        """``c -> xs`` for a window starting at ``start`` (default: the model start): chunk
        ``c``'s per-step forcing, built fresh on every call (the library frees it); the next
        chunk's is built ahead on a worker thread (:class:`_XsAhead`, off with
        ``FESOM_FORCING_PREFETCH=0``)."""
        start = self.start if start is None else _parse_date(start)
        K = self.chunk_steps

        def fn(c):
            dates = [self.date(start, c * K + n) for n in range(K)]
            sf = self.stack_forcing(dates)
            sched = None if self.schedule is None else self.sched(start, c * K, dates)
            if self.shard is not None:
                return self.shard.forcing_xs(sf, sched)
            return jax.device_put(sf if sched is None else (sf, sched))
        if os.environ.get("FESOM_FORCING_PREFETCH", "1") == "0":
            return fn
        return _XsAhead(fn)

    def sched(self, start, n0: int, dates) -> dict:
        """The per-step schedule of the steps ``n0 ..`` of a window starting at ``start``: ``month`` = the window
        month of the step (0 = the start month), ``cost_w`` = 1 inside ``[cost_from, cost_to)``, else 0."""
        start = _parse_date(start)
        sch = self.schedule
        month = np.asarray([(int(d[0]) - start.year) * 12 + int(d[3]) - start.month for d in dates], np.int32)
        M = int(sch.get("n_months") or 0)
        if M and (month.min() < 0 or month.max() >= M):
            raise ValueError(f"window month {int(month.max())} outside the {M} control months")
        n = n0 + np.arange(len(dates))
        lo = sch.get("cost_from")
        hi = sch.get("cost_to")
        w = np.ones(len(dates)) if lo is None else ((n >= int(lo)) & (n < int(hi if hi is not None else 1 << 62)))
        return {"month": month, "cost_w": np.asarray(w, np.float64)}


class _XsAhead:
    """An ``xs_fn`` that builds the next chunk's ``xs`` on ONE worker thread while the device runs this chunk (the
    production loop's forcing prefetch, ``fesom_jax/run.py``; without it the host forcing left the GPUs of a CORE2
    forward chunk ~60 % idle). The next chunk is guessed from the last two calls: one on in the same direction (a
    forward pass, a reverse walk), else ``c + 1`` (the start of a span rebuild). A wrong guess is dropped (built to
    the end if it had started, then freed); a guess that raises (past the window) is ignored. One worker ⇒ the builds
    run one at a time, as without it, and give the same ``xs`` bit for bit."""

    def __init__(self, build: Callable[[int], Any]):
        self.build = build
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xs_ahead")
        self.ahead = None                    #: (chunk, future) of the guess
        self.last = None

    def __call__(self, c):
        c = int(c)
        fut = None
        if self.ahead is not None:
            ca, fa = self.ahead
            self.ahead = None
            if ca == c:
                fut = fa
            elif not fa.cancel():
                try:
                    adjoint._free(fa.result())
                except Exception:            # a guess past the window: nothing was built
                    pass
        if fut is None:
            fut = self.pool.submit(self.build, c)
        d = c - self.last if self.last is not None and abs(c - self.last) == 1 else 1
        self.last = c
        if c + d >= 0:
            self.ahead = (c + d, self.pool.submit(self.build, c + d))
        return fut.result()


def _make_chunk_fn(model: Model, *, rematerialize: bool) -> Callable:
    if model.shard is not None:
        return model.shard.make_chunk_fn(dt=model.dt, phys=model.phys, adjoint_cfg=model.adjoint_cfg,
                                         to_params=model.to_params, accumulators=model.acc_factories,
                                         rematerialize=rematerialize, save_names=tuple(model.save_names))
    mesh, op, fs, phys = model.mesh, model.op, model.forcing.static, model.phys
    adj, dt, to_params = model.adjoint_cfg, model.dt, model.to_params
    accs, save_names = dict(model.accumulators), tuple(model.save_names)

    scheduled = model.schedule is not None

    def chunk_fn(carry, xs, u):
        p = to_params(u)

        def one_step(s, x):
            return stepmod.step(s, mesh, op, None, p, dt=dt, is_first_step=False,
                                step_forcing=x, forcing_static=fs, adjoint_cfg=adj, **phys)

        def step3(s, x, month):                # the scheduled chunk: the month's forcing-control maps
            return stepmod.step(s, mesh, op, None, AC.month_params(p, month), dt=dt, is_first_step=False,
                                step_forcing=x, forcing_static=fs, adjoint_cfg=adj, **phys)

        if scheduled:
            body = AC.with_schedule(step3, accs)
        else:
            body = AC.with_running_costs(one_step, accs) if accs else one_step
        return adjoint.scan_steps(body, carry, xs, rematerialize=rematerialize,
                                  save_names=save_names if rematerialize else ())
    return chunk_fn


#: Registered running accumulators: name -> factory(mesh) -> fn(State).
ACCUMULATORS: dict = {
    "mld": lambda mesh: (lambda s: AC.mld_field(s, mesh, invalid="column")),
    "mld_zero": lambda mesh: (lambda s: AC.mld_field(s, mesh, invalid="zero")),
    "fram": lambda mesh: (lambda s: AC.ice_export_field(s, mesh)),       # Fram Strait ice export [m3/s per node]
}


def build_model(start="2013-01-01", n_steps: int = 48, chunk_steps: int = 48, restart=None,
                adjoint_cfg: AdjointConfig | None = None, ice_mode: str = "prescribed",
                save_names: tuple = (), accumulators=None, rematerialize: bool = True, *,
                params0=None, to_params: Callable | None = None, surface_adj: bool = False,
                config=None, state=None, ice_adjoint: Mapping | None = None,
                tke_settings=None, fast=None, devices: int = 1, transport: str = "padded",
                forcing_ctrl=(), n_months: int = 0, cost_from: int | None = None,
                cost_to: int | None = None) -> Model:
    """Assemble the reference CORE2 model (see the module docstring).

    :param start: window start (ISO date/time or ``datetime``); the restart must be at it.
    :param n_steps: default window length in steps (a multiple of ``chunk_steps``).
    :param restart: restart directory or tag; default ``fesom.<YYYY>.<DOY>.<SSSSS>`` of
        ``start`` from :data:`STATES_DIR`, else the hindcast :data:`RESTART_ARCHIVE`.
    :param adjoint_cfg: derivative-only settings (default :meth:`AdjointConfig.exact`).
    :param ice_mode: ``IceConfig.adjoint_mode`` (default ``"prescribed"``).
    :param ice_adjoint: derivative-only mEVP settings merged into the IceConfig with
        ``_replace``: any of :data:`ICE_ADJOINT_FIELDS` (``adjoint_damping``, ``adjoint_kappa``,
        ``adjoint_sweeps``); validated here as the ice step validates them at trace time.
    :param save_names: remat policy names (e.g. ``("ssh_delta",)``).
    :param accumulators: running per-step sums on the carry: a sequence of names from
        :data:`ACCUMULATORS`, a mapping ``name -> fn(State)``, or a factory ``mesh -> mapping``.
    :param rematerialize: per-step √N remat inside the chunk (reverse mode; the TL program is
        always built without it).
    :param params0, to_params: a caller's control pytree and its map to ``Params`` (default:
        :func:`default_params0` / :func:`default_to_params`).
    :param surface_adj: add the ``[nod2D]`` surface adjustments to the default controls.
    :param state: a host State to start from instead of loading ``restart`` (its date is
        ``start``).
    :param tke_settings: ``TkeConfig`` forward options merged with ``_replace``
        (:func:`parse_tke_settings`; e.g. ``"convection=ramp+conv_ramp_dn2=1e-6"``). They
        change the forward model; empty ⇒ the reference physics.
    :param fast: pieces of the fast configuration's physics on top of ``config`` (:func:`parse_fast`, e.g.
        ``"ice2+se"``; :data:`FAST_PHYSICS`). They change the forward model; empty ⇒ the reference physics.
        ``ice2`` gives the start State the ice-cycle fields at the upstream init values (a restart without them
        starts a new cycle, ``ice_step.init_ice_avg``), ``se`` flat SE rings (a CG restart entering SE,
        ``ssh_se.init_state``).
    :param forcing_ctrl: atmospheric controls: names from ``step.FORCING_ADJ_KEYS``; the control pytree gets
        ``forcing_adj = {name: zeros [n_months, nod2D]}`` (a monthly anomaly map per window month, added to the step
        forcing; the gradient at zero is the sensitivity map), and the chunk's xs a per-step schedule.
    :param cost_from, cost_to: the accumulators sum the window steps ``[cost_from, cost_to)`` only (the schedule's
        cost weight; None: every step).
    :param devices: > 1 runs every chunk sharded over that many devices (the ``dist_<devices>`` partition of
        :data:`CORE2_DIST`, :mod:`fesom_jax.adjoint_sharded`, halo ``transport`` padded or coloured): the carry is
        the folded State, the accumulators per-node fields only (names or a factory), the controls scalars.
    """
    fast = parse_fast(fast)
    todo = [n for n in fast if n not in FAST_IMPLEMENTED and _piece_dt(n) is None]
    if todo:
        raise NotImplementedError(f"fast physics {todo}: not in the adjoint driver yet (have {FAST_IMPLEMENTED})")
    if "sp" in fast and int(devices) == 1:
        raise NotImplementedError("fast piece sp needs devices > 1: single precision exists on the sharded path only "
                                  "(the production runner's casts; the dense step never ran in single precision)")
    config_path = Path(config) if config is not None else CONFIG
    cfg = with_fast_physics(load_yaml(config_path), fast)
    cfg.validate()
    assert_physics(cfg, fast)
    if ice_mode not in ICE_MODES:
        raise ValueError(f"ice_mode must be one of {ICE_MODES}, got {ice_mode!r}")
    ice_kw = dict(ice_adjoint or {})
    bad = sorted(set(ice_kw) - set(ICE_ADJOINT_FIELDS))
    if bad:
        raise ValueError(f"ice_adjoint: unknown field(s) {bad}; allowed {ICE_ADJOINT_FIELDS}")
    ice_cfg = cfg.ice._replace(adjoint_mode=ice_mode, **ice_kw)
    _ice_step.validate_adjoint_settings(ice_cfg)
    if n_steps % chunk_steps:
        raise ValueError(f"n_steps {n_steps} is not a multiple of chunk_steps {chunk_steps}")
    start = _parse_date(start)
    mesh_dir = Path(cfg.mesh) if Path(cfg.mesh).is_absolute() else ROOT / cfg.mesh
    mesh = load_mesh(mesh_dir)
    if state is None:
        rpath = _resolve_restart(start, restart)
        s0, meta = load_state(rpath)
        _check_restart_date(rpath, meta, start)
        _check_restart_dt(rpath, meta, cfg.dt)
    else:
        rpath, s0, meta = Path("<in-memory>"), jax.device_get(state), {}
    # the State of this physics: the ice-cycle fields kept (a run resumes mid-cycle), started at the upstream init
    # values, or dropped with ice every step (ice_step.init_ice_avg); the SE rings kept or flat (a CG restart entering
    # SE); a CG window from an SE restart refused (ssh_se.init_state); the SP carry dropped (double precision)
    s0 = _ice_step.init_ice_avg(s0, ice_cfg, xp=np)
    se_cfg = cfg.split_explicit()
    if se_cfg is not None:
        mesh = ssh_se.with_tables(mesh)
    s0 = ssh_se.init_state(s0, np.asarray(mesh.elem_layer_mask), se_cfg, restart=True)
    if "sp" not in fast:
        from fesom_jax.precision import drop_carry
        s0 = drop_carry(s0)
    sp = "sp" in fast
    # single precision as run.py: the SP preconditioner variant (FESOM2 #995; the CG only), the forcing's per-step
    # values in the point-slope form (#940)
    op = ssh.build_ssh_operator(mesh, dt=cfg.dt, **({"precond_variant": 1} if sp else {}))
    forcing = surface_forcing.build_surface_forcing(mesh, start.year, dt=cfg.dt, mslp=cfg.forcing_mslp(),
                                                   **cfg.forcing_paths())
    if sp:
        forcing.point_slope = True
    phys = cfg.physics_kwargs()
    phys["ice_cfg"] = ice_cfg
    tke_kw = parse_tke_settings(tke_settings)
    if tke_kw:
        phys["tke_cfg"] = phys["tke_cfg"]._replace(**tke_kw).validate()
    adj = (adjoint_cfg if adjoint_cfg is not None else AdjointConfig.exact()).validate()
    if accumulators is None:
        accs, acc_fac = {}, {}
    elif callable(accumulators):
        accs = dict(accumulators(mesh))
        acc_fac = {k: (lambda m, k=k: accumulators(m)[k]) for k in accs}
    elif isinstance(accumulators, Mapping):
        accs, acc_fac = dict(accumulators), None          # functions of the dense mesh's State
    else:
        accs = {n: ACCUMULATORS[n](mesh) for n in accumulators}
        acc_fac = {n: ACCUMULATORS[n] for n in accumulators}
    if params0 is None:
        if to_params is not None:
            raise ValueError("to_params needs params0")
        params0, to_params = default_params0(mesh.nod2D, surface_adj), default_to_params
    elif to_params is None:
        raise ValueError("params0 needs to_params")
    model = Model(cfg=cfg, config_path=config_path, mesh=mesh, op=op, forcing=forcing, phys=phys,
                  adjoint_cfg=adj, ice_mode=ice_mode, start=start, restart=rpath,
                  restart_meta=meta, s0=s0, n_steps=int(n_steps), chunk_steps=int(chunk_steps),
                  save_names=tuple(save_names), rematerialize=bool(rematerialize),
                  accumulators=accs, params0=_strong_f64(params0), to_params=to_params,
                  surface_adj=bool(surface_adj), tke_settings=tke_kw, fast=fast)
    forcing_ctrl = tuple(forcing_ctrl or ())
    if forcing_ctrl:
        bad = sorted(set(forcing_ctrl) - set(stepmod.FORCING_ADJ_KEYS))
        if bad or int(n_months) < 1:
            raise ValueError(f"forcing_ctrl: unknown {bad} (known {stepmod.FORCING_ADJ_KEYS}) or n_months < 1")
        model.params0 = dict(model.params0)
        model.params0["forcing_adj"] = {k: np.zeros((int(n_months), int(mesh.nod2D)), np.float64)
                                        for k in forcing_ctrl}
    if forcing_ctrl or cost_from is not None:
        model.schedule = {"n_months": int(n_months), "cost_from": cost_from, "cost_to": cost_to}
    if int(devices) > 1:
        from fesom_jax import adjoint_sharded
        if acc_fac is None:
            raise ValueError("devices > 1: give the accumulators as names or a factory mesh -> mapping (each is "
                             "rebuilt on the device-local mesh)")
        others = {k: v for k, v in model.params0.items() if k != "forcing_adj"}
        if surface_adj or any(np.ndim(v) for v in jax.tree.leaves(others)):
            raise ValueError("devices > 1: scalar controls (and the forcing-control maps) only; a per-node field "
                             "control other than forcing_adj is not folded")
        part = partit.read_partition(CORE2_DIST, int(devices))
        model.shard = adjoint_sharded.build(mesh, part, op, forcing.static, phys=phys, transport=transport,
                                            precision="single" if sp else "double", node_ids=bool(forcing_ctrl))
        model.acc_factories = acc_fac
    model.chunk_fn = _make_chunk_fn(model, rematerialize=bool(rematerialize))
    return model


# ---------------------------------------------------------------------------
# costs


COSTS = ("box_T_vw", "sst_aw", "ssh_aw", "mld_box", "mld_mean")


def cost_fn(model: Model, cost, box=DEFAULT_BOX, n_steps: int | None = None) -> Callable:
    """A loss ``carry -> scalar``. ``cost`` is a callable (used as is: it receives the carry)
    or a name from :data:`COSTS` evaluated on ``box``:

    - ``box_T_vw``: live-volume-weighted box-mean T of the final state;
    - ``sst_aw`` / ``ssh_aw``: area-weighted box means of the final top-layer T / ``eta_n``;
    - ``mld_box``: area-weighted box mean of the final MLD (``invalid="column"``);
    - ``mld_mean``: area-weighted box mean of the WINDOW-MEAN MLD — needs the ``"mld"``
      accumulator, and ``n_steps`` (default: the model's window).
    State costs are terminal seeds (no running injection)."""
    if callable(cost):
        return cost
    mesh = model.mesh
    node, level = box_masks(mesh, box)
    node_j, level_j = jnp.asarray(node), jnp.asarray(level)
    w_surf = jnp.where(jnp.asarray(mesh.node_layer_mask[:, 0]) & node_j,
                       mesh.areasvol[:, 0], 0.0)
    if cost == "box_T_vw":
        return AC.terminal(lambda s: AC.box_mean_T_vw(s, mesh, node_j, level_j))
    if cost == "sst_aw":
        return AC.terminal(lambda s: AC.sst_area_mean(s, mesh, node_j))
    if cost == "ssh_aw":
        return AC.terminal(lambda s: AC.ssh_area_mean(s, mesh, node_j))
    if cost == "mld_box":
        return AC.terminal(lambda s: jnp.sum(w_surf * AC.mld_field(s, mesh)) / jnp.sum(w_surf))
    if cost == "mld_mean":
        if "mld" not in model.accumulators:
            raise ValueError("cost mld_mean needs the 'mld' accumulator (build_model("
                             "accumulators=('mld',)))")
        n = model.n_steps if n_steps is None else int(n_steps)
        mean = AC.running_mean("mld", n)
        return lambda carry: jnp.sum(w_surf * mean(carry)) / jnp.sum(w_surf)
    raise ValueError(f"unknown cost {cost!r}; registered: {COSTS}")


# ---------------------------------------------------------------------------
# directions


DIRECTIONS = ("T_inside", "T_above", "T_outside", "ln_k_gm", "heat", "stress_x",
              "ln_tke_c_k", "ln_tke_c_eps", "ln_tke_cd", "ln_tke_alpha")
#: size of one unit of each direction
DIRECTION_UNITS = {"T_inside": "1 K at one cell", "T_above": "1 K at one cell",
                   "T_outside": "1 K at one cell", "heat": "1 W/m2 into the ocean on the box "
                   "surface", "stress_x": "0.01 N/m2 on the box surface",
                   **{f"ln_{n}": "1 in ln(p/p_default)" for n in LOG_CONTROLS}}
STRESS_UNIT = 0.01


class Direction(NamedTuple):
    name: str
    carry: Any        #: tangent of the start carry (host; zeros for a parameter direction)
    params: Any       #: tangent of ``params`` (host; zeros for a state direction)
    info: dict        #: where it acts (node, level, depth, ...)


def _point(model: Model, box, where: str):
    """(node, level) of a T point: ``inside`` = the wet box node nearest the box centre at the
    level nearest the box mid-depth; ``above`` = the same column at the surface layer (above
    the inside point; inside the box when the box reaches the surface); ``outside`` = the wet
    node nearest the point 10° of latitude south of the box centre, at the inside level."""
    mesh, b = model.mesh, _box(box)
    lon, lat = _lonlat_deg(mesh)
    node, level = box_masks(mesh, b)
    zmid = -np.asarray(mesh.Z)
    k_in = int(np.argmin(np.abs(zmid - 0.5 * (b.z_min + b.z_max))))
    deep = np.asarray(mesh.node_layer_mask)[:, k_in]
    clat = 0.5 * (b.lat_min + b.lat_max)
    clon = (0.5 * (b.lon_min + b.lon_max) if b.lon_min <= b.lon_max
            else ((0.5 * (b.lon_min + b.lon_max + 360.0) + 180.0) % 360.0) - 180.0)

    def nearest(mask, la, lo):
        d = (lat - la) ** 2 + ((((lon - lo) + 180.0) % 360.0 - 180.0) * np.cos(np.radians(la))) ** 2
        d = np.where(mask, d, np.inf)
        return int(np.argmin(d))

    n_in = nearest(node & deep, clat, clon)
    if where == "inside":
        return n_in, k_in
    if where == "above":
        return n_in, 0
    if where == "outside":
        return nearest(deep & ~node, clat - 10.0, clon), k_in
    raise ValueError(where)


def direction(model: Model, name: str, box=DEFAULT_BOX) -> Direction:
    """The named direction ``name`` (:data:`DIRECTIONS`) as full host tangents of the start carry
    and of ``params`` (unit sizes: :data:`DIRECTION_UNITS`). Parameter directions need the
    default control pytree; ``heat``/``stress_x`` need ``surface_adj=True``."""
    if model.shard is not None and name in ("T_inside", "T_above", "T_outside"):
        raise NotImplementedError("a state direction on the sharded (folded) carry")
    carry0 = model.init_carry(host=True)
    t_carry = jax.tree.map(lambda a: np.zeros(np.shape(a), np.asarray(a).dtype), carry0)
    t_par = jax.tree.map(lambda a: np.zeros(np.shape(a), np.asarray(a).dtype), model.params0)
    info = {}
    if name in ("T_inside", "T_above", "T_outside"):
        n, k = _point(model, box, name[2:])
        st = model.state_of(t_carry)
        T = np.array(st.T)
        T[n, k] = 1.0
        st = dataclasses.replace(st, T=T)
        t_carry = (st, t_carry[1]) if model.accumulators else st
        lon, lat = _lonlat_deg(model.mesh)
        info = {"node": n, "level": k, "lon": float(lon[n]), "lat": float(lat[n]),
                "depth": float(-np.asarray(model.mesh.Z)[k])}
    elif name.startswith("ln_"):
        if model.to_params is not default_to_params or name not in t_par:
            raise ValueError(f"direction {name} needs the default control pytree")
        t_par[name] = np.ones((), np.float64)
    elif name in ("heat", "stress_x"):
        if "surface_adj" not in t_par:
            raise ValueError(f"direction {name} needs build_model(surface_adj=True)")
        node, _ = box_masks(model.mesh, box)
        t_par["surface_adj"][name] = node.astype(np.float64) * (STRESS_UNIT if name == "stress_x"
                                                                else 1.0)
        info = {"n_nodes": int(node.sum())}
    else:
        raise ValueError(f"unknown direction {name!r}; known: {DIRECTIONS}")
    info["unit"] = DIRECTION_UNITS[name]
    return Direction(name, t_carry, t_par, info)


# ---------------------------------------------------------------------------
# provenance and output


PROVENANCE_KEYS = ("action", "git_head", "git_dirty", "slurm_job", "host", "device", "platform",
                   "xla_flags", "tmpdir", "cuda_cache_path", "jax_version", "config",
                   "adjoint_mode", "diag_cuts", "ice_mode", "start", "restart", "n_steps",
                   "window_days", "chunk_steps", "n_chunks", "stride", "direction", "h",
                   "save_names", "rematerialize", "accumulators", "compile_s", "run_s", "time")


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:          # noqa: BLE001 — provenance must never kill a run
        return ""


def provenance(model: Model, action: str, *, n_steps: int | None = None, start=None,
               stride=None, direction=None, h=None, timer: _Timer | None = None,
               **extra) -> dict:
    """The provenance fields every result row carries (:data:`PROVENANCE_KEYS`) plus ``extra``."""
    n = model.n_steps if n_steps is None else int(n_steps)
    dev = jax.devices()[0]
    row = {
        "action": action, "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
        "device": getattr(dev, "device_kind", str(dev)), "platform": dev.platform,
        "xla_flags": os.environ.get("XLA_FLAGS", ""), "tmpdir": os.environ.get("TMPDIR"),
        "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH"), "jax_version": jax.__version__,
        "config": str(model.config_path), "adjoint_mode": ("exact" if model.adjoint_cfg.describe() == "none"
                         else model.adjoint_cfg.describe()),
        "diag_cuts": sorted(model.adjoint_cfg.diag_cuts), "ice_mode": model.ice_mode,
        "start": (_parse_date(start) if start is not None else model.start).isoformat(),
        "restart": str(model.restart), "n_steps": n, "window_days": n * model.dt / 86400.0,
        "chunk_steps": model.chunk_steps, "n_chunks": n // model.chunk_steps, "stride": stride,
        "direction": direction, "h": h, "save_names": list(model.save_names),
        "rematerialize": model.rematerialize, "accumulators": sorted(model.accumulators),
        "compile_s": None, "run_s": None,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if model.tke_settings:
        row["tke_settings"] = dict(model.tke_settings)
    if model.fast:
        row["fast"] = list(model.fast)
    if model.schedule is not None:
        row["schedule"] = dict(model.schedule)
        if "forcing_adj" in model.params0:
            row["forcing_ctrl"] = sorted(model.params0["forcing_adj"])
    if model.shard is not None:
        row["devices"], row["transport"] = model.devices, model.shard.transport
        if model.shard.se_wide:
            row["se_wide"] = model.shard.se_wide
    fenv = {k: v for k, v in os.environ.items() if k.startswith("FESOM_")}
    if fenv:                                     # the GPU options (round-off) and the derivative switches in force
        row["fesom_env"] = fenv
    if timer is not None:
        row.update(timer.fields())
    row.update(extra)
    return row


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, (np.ndarray, jax.Array)):
        a = np.asarray(x)
        return a.item() if a.ndim == 0 else a.tolist()
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, (Path, datetime.datetime)):
        return str(x)
    return x


def append_row(path, row: dict) -> None:
    """Append one JSON row (never rewrites the file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: v for k, v in row.items() if not str(k).startswith("_")}
    with open(path, "a") as f:
        f.write(json.dumps(_jsonable(row)) + "\n")


def _small(tree, limit: int = 16) -> dict:
    """``{leaf path: value}`` of the leaves with at most ``limit`` elements (the rest go to npz)."""
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        a = np.asarray(leaf)
        if a.dtype != jax.dtypes.float0 and a.size <= limit:
            out[jax.tree_util.keystr(path, simple=True, separator=".")] = a.tolist()
    return out


def _save_npz(path: Path, tree) -> str:
    """Save every floating leaf of ``tree`` to a new npz (refuses an existing file)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists")
    arrs = {jax.tree_util.keystr(p, simple=True, separator="."): np.asarray(x)
            for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]
            if np.asarray(x).dtype != jax.dtypes.float0}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrs)
    return str(path)


# ---------------------------------------------------------------------------
# resumable reverse sweep


class DiskBoundaries(MutableMapping):
    """The saved boundaries ``chunks`` of a run directory, read from disk on ``[c]`` (the lazy
    :meth:`~fesom_jax.adjoint.ReverseCheckpoint.load_boundaries`). Only states assigned to it
    (a resumed forward adds its new boundaries) stay on the host. A year at stride 24 is 31
    stored states, 60 GB that the eager load holds on the host for the whole sweep; the reverse
    walk reads each one once, ~2 s per 1.93 GB state."""

    def __init__(self, directory, chunks, like, load):
        self.dir, self.like, self._read = Path(directory), like, load
        self._disk = set(int(c) for c in chunks)
        self._host: dict = {}

    def __getitem__(self, c):
        if c in self._host:
            return self._host[c]
        if c not in self._disk:
            raise KeyError(c)
        return self._read(self.dir / f"boundary_{c}.npz", self.like)

    def __setitem__(self, c, state):
        self._host[c] = state

    def __delitem__(self, c):
        self._host.pop(c, None)
        self._disk.discard(c)

    def __iter__(self):
        return iter(sorted(self._disk | set(self._host)))

    def __len__(self):
        return len(self._disk | set(self._host))

    def resident_bytes(self) -> int:
        return sum(a.nbytes for a in jax.tree.leaves(self._host))


class DiskCheckpoint:
    """A :class:`~fesom_jax.adjoint.ReverseCheckpoint` on disk, for sweeps longer than a job.

    Forward boundaries are saved as ``boundary_<c>.npz`` (written to a temporary name and
    renamed, so a killed job leaves no half file), the forward loss in ``forward.json``, and
    when the wall-clock ``budget_s`` runs out in the reverse walk the carried cotangent, the
    partial gradient and the trace in ``reverse_<next>.npz`` / ``reverse_<next>.json``; the
    sweep then raises :class:`~fesom_jax.adjoint.LegIncomplete` and a rerun with the same
    directory continues. Nothing is overwritten or deleted: the directory is a job's scratch
    and may be removed by hand when the result is in. ``lazy``: serve the boundaries from disk
    (:class:`DiskBoundaries`) instead of loading them all to the host."""

    def __init__(self, directory, params_like, *, budget_s: float = math.inf,
                 lazy: bool = False):
        self.lazy = bool(lazy)
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.params_like = params_like
        self.budget_s, self.t0 = float(budget_s), time.time()
        fj = self.dir / "forward.json"
        self.have_boundaries = fj.exists()
        self.loss = json.loads(fj.read_text())["loss"] if fj.exists() else float("nan")

    def _out_of_time(self) -> bool:
        return time.time() - self.t0 > self.budget_s

    @staticmethod
    def _save(path: Path, tree) -> None:
        if path.exists():
            return
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, *[np.asarray(x) for x in jax.tree.leaves(tree)])
        tmp.rename(path)

    @staticmethod
    def _load(path: Path, like):
        with np.load(path) as z:
            leaves = [z[f"arr_{i}"] for i in range(len(z.files))]
        return jax.tree.unflatten(jax.tree.structure(like), leaves)

    def _saved(self) -> list:
        return sorted(int(p.stem.split("_")[1]) for p in self.dir.glob("boundary_*.npz"))

    def forward_resume(self, like):
        saved = self._saved()
        if not saved:
            return None, 0
        c = max(saved)
        return self._load(self.dir / f"boundary_{c}.npz", like), c

    def put_boundary(self, c, state) -> bool:
        self._save(self.dir / f"boundary_{c}.npz", state)
        return self._out_of_time()

    def load_boundaries(self, like):
        return self._boundaries(self._saved(), like)

    def _boundaries(self, chunks, like):
        if self.lazy:
            return DiskBoundaries(self.dir, chunks, like, self._load)
        return {c: self._load(self.dir / f"boundary_{c}.npz", like) for c in chunks}

    def finish_forward(self, loss: float) -> None:
        fj = self.dir / "forward.json"
        if not fj.exists():
            fj.write_text(json.dumps({"loss": float(loss)}))
        self.have_boundaries, self.loss = True, float(loss)

    def reverse_start(self, seed_cot, n_chunks, like):
        legs = sorted(int(p.stem.split("_")[1]) for p in self.dir.glob("reverse_*.json"))
        if not legs:
            return seed_cot, None, n_chunks - 1, []
        nxt = min(legs)                                   # the furthest-advanced leg
        meta = json.loads((self.dir / f"reverse_{nxt}.json").read_text())
        cot = self._load(self.dir / f"reverse_{nxt}_cot.npz", like)
        grad = self._load(self.dir / f"reverse_{nxt}_grad.npz", self.params_like)
        return cot, grad, nxt, meta["trace"]

    def tick(self, next_chunk, cot, grad, trace) -> bool:
        if next_chunk < 0 or not self._out_of_time():
            return False
        self._save(self.dir / f"reverse_{next_chunk}_cot.npz", jax.device_get(cot))
        self._save(self.dir / f"reverse_{next_chunk}_grad.npz", jax.device_get(grad))
        (self.dir / f"reverse_{next_chunk}.json").write_text(json.dumps({"trace": list(trace)}))
        return True


# ---------------------------------------------------------------------------
# actions


def _window(model: Model, start, n_steps, init):
    start = model.start if start is None else _parse_date(start)
    n = model.n_steps if n_steps is None else int(n_steps)
    nc = model.chunks_for(n)
    carry = model.init_carry(init)
    return start, n, nc, carry


def _free(tree):
    adjoint._free(tree)


def run_fwd(model: Model, costs=("box_T_vw",), *, box=DEFAULT_BOX, params=None, start=None,
            n_steps=None, init=None) -> dict:
    """The window forward (chunked, nothing parked) and the cost values at its end.
    Returns the row; ``row["_final"]`` is the final carry (host)."""
    start, n, nc, carry = _window(model, start, n_steps, init)
    params = model.params0 if params is None else params
    losses = [cost_fn(model, c, box, n) for c in costs]
    with _Timer() as t:
        fin = adjoint.chunked_forward(model.chunk_fn, params, init_state=carry, n_chunks=nc,
                                      xs_fn=model.xs_fn(start))
        vals = [float(L(fin)) for L in losses]
        host = jax.device_get(fin)
    _free(fin)
    return provenance(model, "fwd", n_steps=n, start=start, timer=t,
                      costs=[c if isinstance(c, str) else getattr(c, "__name__", "callable")
                             for c in costs], box=box if isinstance(box, str) else list(box),
                      values=vals, forcing_s=model.forcing_seconds, _final=host)


def run_grad(model: Model, costs=("box_T_vw",), *, box=DEFAULT_BOX, params=None, start=None,
             n_steps=None, init=None, stride: int = 1, check_linear=True, on_cot=None,
             on_chunk=None, resume_dir=None, budget_s: float = math.inf,
             abort_at: float | None = None, action: str = "grad",
             checkpoint_factory: Callable | None = None) -> list:
    """Gradients of one or several costs over a window. One cost: :func:`chunked_grad`
    (resumable with ``resume_dir``); several: :func:`chunked_grad_multi` (``on_cot(k, c,
    cot)``). ``abort_at``: stop the sweep when a boundary cotangent norm exceeds it or is not
    finite (the row gets ``overflow``). ``checkpoint_factory(resume_dir, params, budget_s)``
    replaces :class:`DiskCheckpoint` for a resumable sweep. Returns one row per cost;
    ``row["_result"]`` holds the library result (grads, ``state_cot`` on the host)."""
    start, n, nc, carry = _window(model, start, n_steps, init)
    params = model.params0 if params is None else params
    names = [c if isinstance(c, str) else getattr(c, "__name__", "callable") for c in costs]
    losses = [cost_fn(model, c, box, n) for c in costs]
    multi = len(losses) > 1
    ftr = [[] for _ in losses]
    over = {}

    class _Overflow(Exception):
        pass

    def hook(k, c, cot):
        fn = adjoint_stab.field_norms(cot)
        ftr[k].append({"boundary": c, **fn})
        if abort_at is not None:
            tot = math.sqrt(sum(v * v for v in fn.values()))
            if not math.isfinite(tot) or tot > abort_at:
                over[k] = {"boundary": c, "norm": tot,
                           "day": c * model.chunk_steps * model.dt / 86400.0,
                           "lag_days": (nc - c) * model.chunk_steps * model.dt / 86400.0}
                raise _Overflow()
        if on_cot is not None:
            on_cot(k, c, cot) if multi else on_cot(c, cot)

    rows = []
    leg = None
    with _Timer() as t:
        try:
            if multi:
                if resume_dir is not None:
                    raise ValueError("resume_dir needs a single cost")
                res = adjoint.chunked_grad_multi(
                    model.chunk_fn, params, init_state=carry, n_chunks=nc,
                    xs_fn=model.xs_fn(start), loss_fns=losses, chunk_steps=model.chunk_steps,
                    boundary_stride=stride, check_linear=check_linear, on_cot=hook,
                    on_chunk=on_chunk, field_trace=False)
            else:
                make_ck = checkpoint_factory or (
                    lambda d, p, b: DiskCheckpoint(d, p, budget_s=b))
                ck = None if resume_dir is None else make_ck(resume_dir, params, budget_s)
                res = adjoint.chunked_grad(
                    model.chunk_fn, params, init_state=carry, n_chunks=nc,
                    xs_fn=model.xs_fn(start), loss_fn=losses[0], chunk_steps=model.chunk_steps,
                    boundary_stride=stride, check_linear=check_linear, checkpoint=ck,
                    on_cot=lambda c, cot: hook(0, c, cot), on_chunk=on_chunk, field_trace=False)
        except _Overflow:
            res = None
        except adjoint.LegIncomplete as exc:
            res, leg = None, exc
    if leg is not None:
        return [provenance(model, action, n_steps=n, start=start, stride=stride, timer=t,
                           cost=names[0], status="incomplete", leg_phase=leg.phase,
                           leg_chunk=leg.chunk, resume_dir=str(resume_dir))]
    if res is None:
        k = next(iter(over))
        tr = [f for f in ftr[k]]
        return [provenance(model, action, n_steps=n, start=start, stride=stride, timer=t,
                           cost=names[k], status="overflow", overflow=over[k],
                           trace=[math.sqrt(sum(v * v for kk, v in f.items() if kk != "boundary"))
                                  for f in tr], field_trace=tr,
                           check_linear=str(check_linear))]
    per = ([(res.losses[k], res.grads[k], res.traces[k], res.state_cots[k]) for k in range(len(losses))]
           if multi else [(res.loss, res.grad, res.trace, res.state_cot)])
    for k, (loss, grad, trace, scot) in enumerate(per):
        rows.append(provenance(
            model, action, n_steps=n, start=start, stride=stride, timer=t, cost=names[k],
            box=box if isinstance(box, str) else list(box), status="ok", loss=loss,
            grad=_small(jax.device_get(grad)), trace=trace, field_trace=ftr[k],
            check_linear=str(check_linear), linear_check=res.linear_check,
            host_gb=res.host_gb, forward_seconds=res.forward_seconds,
            reverse_seconds=res.reverse_seconds, n_costs=len(losses),
            forcing_s=model.forcing_seconds,
            _result={"grad": jax.device_get(grad), "state_cot": scot}))
    return rows


def run_screen(model: Model, cost="box_T_vw", *, abort_at: float = ABORT_AT, **kw) -> dict:
    """A terminal-seed sweep with per-field traces and the overflow abort (``status`` =
    ``"overflow"`` with ``overflow = {boundary, day, lag_days, norm}``, or ``"ok"``)."""
    if not callable(cost) and cost in ("mld_mean",):
        raise ValueError("a screen seeds the final state: use a terminal cost (e.g. mld_box)")
    return run_grad(model, (cost,), abort_at=abort_at, action="screen", **kw)[0]


def _tangent_pair(model: Model, dirn: Direction):
    return dirn.carry, dirn.params


def run_fd(model: Model, cost="box_T_vw", dirn="T_inside", hs=(1e-2, 1e-3, 1e-4), *,
           box=DEFAULT_BOX, repeats: int = 1, ad: bool = False, start=None, n_steps=None,
           init=None) -> list:
    """Central differences of ``cost`` along ``dirn`` (a name or a :class:`Direction`) for every
    ``h``, all evaluations with the same compiled forward; ``ad=True`` also runs the gradient
    and reports ``<dJ/dx, d>``. One row per ``h`` (``fd``, ``ad``, ``rel_err``,
    ``noise_over_h``)."""
    start, n, nc, _ = _window(model, start, n_steps, init)
    dirn = direction(model, dirn, box) if isinstance(dirn, str) else dirn
    loss = cost_fn(model, cost, box, n)
    x0 = (model.init_carry(init, host=True), model.params0)
    d = _tangent_pair(model, dirn)
    evals = []

    def J(x):
        fin = adjoint.chunked_forward(model.chunk_fn, x[1], init_state=jax.device_put(x[0]),
                                      n_chunks=nc, xs_fn=model.xs_fn(start))
        v = float(loss(fin))
        _free(fin)
        evals.append(v)
        return v

    with _Timer() as t:
        ad_val = float("nan")
        if ad:
            r = run_grad(model, (cost,), box=box, start=start, n_steps=n, init=init)[0]
            res = r["_result"]
            ad_val = (adjoint_check.tree_vdot(res["state_cot"], d[0])
                      + adjoint_check.tree_vdot(res["grad"], d[1]))
        rows_fd = adjoint_check.fd_sweep(J, x0, d, list(hs), ad_val, repeats=repeats)
    name = cost if isinstance(cost, str) else getattr(cost, "__name__", "callable")
    return [provenance(model, "fd", n_steps=n, start=start, direction=dirn.name, h=r.h, timer=t,
                       cost=name, box=box if isinstance(box, str) else list(box),
                       direction_info=dirn.info, fd=r.fd, ad=r.ad, rel_err=r.rel_err,
                       noise_over_h=r.noise_over_h, repeats=repeats, j0=evals[0],
                       forcing_s=model.forcing_seconds)
            for r in rows_fd]


def _pinned(model: Model, params, carry, nc, start):
    return adjoint_check.forward_states(model.chunk_fn, params, carry, nc, model.xs_fn(start))


def run_tl(model: Model, cost="box_T_vw", dirn="T_inside", *, box=DEFAULT_BOX, start=None,
           n_steps=None, init=None, pinned=None, tl_options=TL_COMPILER_OPTIONS) -> dict:
    """The window tangent along ``dirn`` on pinned chunk-entry states; the cost's directional
    derivative ``<dJ/ds_end, M d>``; tangent norms per boundary and per field at the end.
    ``row["_pinned"]`` returns the pinned states for reuse (host)."""
    start, n, nc, carry = _window(model, start, n_steps, init)
    dirn = direction(model, dirn, box) if isinstance(dirn, str) else dirn
    loss = cost_fn(model, cost, box, n)
    params = model.params0
    with _Timer() as t:
        pins = _pinned(model, params, carry, nc, start) if pinned is None else pinned
        tl = adjoint_check.window_jvp(model.chunk_fn_tl, params, carry, dirn.carry, dirn.params,
                                      nc, model.xs_fn(start), pinned_states=pins,
                                      compiler_options=tl_options)
        seed = jax.vjp(loss, jax.device_put(pins[nc]))[1](1.0)[0]
        dj = adjoint_check.tree_vdot(seed, tl.tangent)
    name = cost if isinstance(cost, str) else getattr(cost, "__name__", "callable")
    return provenance(model, "tl", n_steps=n, start=start, direction=dirn.name, timer=t,
                      cost=name, box=box if isinstance(box, str) else list(box),
                      direction_info=dirn.info, dJ=dj, trace=tl.trace,
                      field_norms_end=adjoint_stab.field_norms(tl.tangent),
                      tl_seconds=tl.seconds, tl_compiler_options=tl_options,
                      forcing_s=model.forcing_seconds, _pinned=pins)


def run_dot(model: Model, cost="box_T_vw", dirn="T_inside", *, box=DEFAULT_BOX,
            amps=(1.0, 1e-8), start=None, n_steps=None, init=None, pinned=None,
            check_linear=True, tl_options=TL_COMPILER_OPTIONS) -> list:
    """Pinned-state dot test of the window map ``M: (start carry, params) -> end carry``:
    ``<y, M(a·x)>`` against ``a·<Mᵀy, x>`` with ``x`` = the direction and ``y`` = the cost's seed
    ``dJ/ds_end``. One row per amplitude with ``rel`` and ``rel_norm`` (divided by
    ``‖M(a·x)‖·‖y‖``)."""
    start, n, nc, carry = _window(model, start, n_steps, init)
    dirn = direction(model, dirn, box) if isinstance(dirn, str) else dirn
    loss = cost_fn(model, cost, box, n)
    params = model.params0
    with _Timer() as t:
        pins = _pinned(model, params, carry, nc, start) if pinned is None else pinned
        y = jax.device_get(jax.vjp(loss, jax.device_put(pins[nc]))[1](1.0)[0])
        xs = model.xs_fn(start)

        def jvp(v):
            return adjoint_check.window_jvp(model.chunk_fn_tl, params, carry, v[0], v[1], nc, xs,
                                            pinned_states=pins,
                                            compiler_options=tl_options).tangent

        def vjp(w):
            r = adjoint_check.window_vjp(model.chunk_fn, params, carry, w, nc, xs,
                                         pinned_states=pins, chunk_steps=model.chunk_steps,
                                         check_linear=check_linear, field_trace=False)
            return (r.state_cot, jax.device_get(r.params_cot))

        rows = adjoint_check.dot_test(None, None, (dirn.carry, dirn.params), y, amps=tuple(amps),
                                      jvp=jvp, vjp=vjp)
    name = cost if isinstance(cost, str) else getattr(cost, "__name__", "callable")
    return [provenance(model, "dot", n_steps=n, start=start, direction=dirn.name, timer=t,
                       cost=name, box=box if isinstance(box, str) else list(box),
                       direction_info=dirn.info, amp=r.amp, lhs=r.lhs, rhs=r.rhs, rel=r.rel,
                       rel_norm=r.rel_norm, tl_compiler_options=tl_options,
                       forcing_s=model.forcing_seconds)
            for r in rows]


# ---------------------------------------------------------------------------
# shared States and daily diagnostics


#: The daily 2-D diagnostics of the ``states`` action: name -> (unit, meaning).
DAILY_FIELDS = {
    "mld": ("m", "density-threshold MLD at the end of the day (deepest wet mid-layer depth "
                 "where mixed to the bottom)"),
    "mld_valid": ("1", "1 where the MLD threshold is crossed in the column, else 0"),
    "sst": ("degC", "top-layer T at the end of the day"),
    "surface_speed": ("m/s", "|uvnode| of the top layer at the end of the day"),
    "a_ice": ("1", "sea-ice concentration at the end of the day"),
    "n2_colmin": ("s-2", "minimum N^2 over the interior wet interfaces at the end of the day "
                         "(0 where the column has none)"),
    "tke_k1": ("m2/s2", "tke at interface k=1 at the end of the day"),
    "tke_floor_frac": ("1", "fraction of the day's steps with tke(k=1) at the floor tke_min"),
    "convective_frac": ("1", "fraction of the day's steps with N^2 < 0 at an interior wet "
                             "interface (convective adjustment active)"),
}


def _interior_iface(mesh):
    from fesom_jax.pp import _interior_iface_mask
    return _interior_iface_mask(mesh.ulevels_nod2D, mesh.nlevels_nod2D, mesh.nl)


def daily_accumulators(mesh, tke_min: float) -> dict:
    """Per-step indicator sums for the fractions in :data:`DAILY_FIELDS`."""
    iface = jnp.asarray(_interior_iface(mesh))
    return {
        "tke_floor": lambda s: jnp.where(s.tke[:, 1] <= tke_min, 1.0, 0.0),
        "convective": lambda s: jnp.where(jnp.any(iface & (s.bvfreq < 0.0), axis=1), 1.0, 0.0),
    }


def _daily_snapshot_fn(mesh, steps: int):
    iface = jnp.asarray(_interior_iface(mesh))

    @jax.jit
    def snap(carry):
        s, acc = carry
        _, Z = ale.live_geometry(mesh, s.hnode)
        _, valid = obs_compare.mld_density_threshold(s.T, s.S, Z, jnp.asarray(mesh.node_layer_mask))
        n2 = jnp.where(iface, s.bvfreq, jnp.inf)
        n2min = jnp.min(n2, axis=1)
        return {
            "mld": AC.mld_field(s, mesh), "mld_valid": valid.astype(jnp.float64),
            "sst": s.T[:, 0],
            "surface_speed": jnp.sqrt(s.uvnode[:, 0, 0] ** 2 + s.uvnode[:, 0, 1] ** 2),
            "a_ice": s.a_ice, "n2_colmin": jnp.where(jnp.isfinite(n2min), n2min, 0.0),
            "tke_k1": s.tke[:, 1],
            "tke_floor_frac": acc["tke_floor"] / steps,
            "convective_frac": acc["convective"] / steps,
        }
    return snap


def run_states(model: Model, out_dir, *, n_days: int, start=None, init=None,
               save_monthly: bool = True, save_at=(), skip_existing: bool = False,
               json_path=None) -> dict:
    """Forward from the model start (or ``start``/``init``) over ``n_days`` days of
    ``model.chunk_steps`` steps each: full States on the 1st of each month at 00:00 (and at the
    boundaries in ``save_at``, chunk indices) as ``out_dir/fesom.<YYYY>.<DOY>.<SSSSS>``, the
    daily diagnostics as ``out_dir/daily/diag_<YYYY>_<DOY>.npz`` (the day starting on that
    date), and ``out_dir/README.json``. ``model`` must be built with ``daily_accumulators``
    and ``rematerialize=False`` (see :func:`build_states_model`)."""
    if set(model.accumulators) != {"tke_floor", "convective"}:
        raise ValueError("run_states needs a model from build_states_model")
    out_dir = Path(out_dir)
    (out_dir / "daily").mkdir(parents=True, exist_ok=True)
    start = model.start if start is None else _parse_date(start)
    K = model.chunk_steps
    carry = model.init_carry(init)
    fwd = adjoint._jitted_forward(model.chunk_fn)
    snap = _daily_snapshot_fn(model.mesh, K)
    zero_acc = {k: jnp.zeros_like(v) for k, v in carry[1].items()}
    step0 = int(model.restart_meta.get("step", 0))
    xs_fn = model.xs_fn(start)
    written, skipped, diag_files = [], [], []
    per_day = []

    def save(c, st):
        when = start + datetime.timedelta(seconds=c * K * model.dt)
        d = out_dir / restart_tag(when)
        if d.exists():
            if skip_existing:
                skipped.append(d.name)
                return
            raise FileExistsError(f"{d} exists (pass skip_existing to continue past it)")
        _write_state(d, st, model.mesh, step=step0 + c * K, when=when,
                     attrs={"source": "core2_adj states", "git_head": _git("rev-parse", "HEAD"),
                            "slurm_job": os.environ.get("SLURM_JOB_ID") or ""})
        written.append(d.name)

    def wanted(c):
        when = start + datetime.timedelta(seconds=c * K * model.dt)
        return (c in set(save_at)) or (save_monthly and when.day == 1 and when.hour == 0
                                       and when.minute == 0 and when.second == 0)

    with _Timer() as t:
        if wanted(0):
            save(0, jax.device_get(model.state_of(carry)))
        for c in range(int(n_days)):
            t_c = time.time()
            xs = xs_fn(c)
            nxt = jax.block_until_ready(fwd(carry, xs, model.params0))
            _free(xs)
            _free(carry)                     # our own device copy (init_carry / zeros below)
            day0 = start + datetime.timedelta(seconds=c * K * model.dt)
            diag = jax.device_get(snap(nxt))
            f = out_dir / "daily" / (f"diag_{day0.year:04d}_"
                                     f"{(day0 - datetime.datetime(day0.year, 1, 1)).days + 1:03d}"
                                     f"_{day0.hour * 3600 + day0.minute * 60:05d}.npz")
            if f.exists():
                if not skip_existing:
                    raise FileExistsError(f"{f} exists")
            else:
                np.savez(f, **diag)
                diag_files.append(f.name)
            _free(nxt[1])
            carry = (nxt[0], jax.tree.map(jnp.zeros_like, zero_acc))
            if wanted(c + 1):
                save(c + 1, jax.device_get(nxt[0]))
            per_day.append(time.time() - t_c)
    readme = out_dir / "README.json"
    if readme.exists():
        readme = out_dir / f"README.{os.environ.get('SLURM_JOB_ID', int(time.time()))}.json"
    row = provenance(model, "states", n_steps=int(n_days) * K, start=start, timer=t,
                     out_dir=str(out_dir), written=written, skipped=skipped,
                     n_daily=len(diag_files), seconds_per_day=per_day,
                     forcing_s=model.forcing_seconds)
    info = {"provenance": {k: v for k, v in row.items() if not k.startswith("_")},
            "state_fields": list(zarr_output._all_state_fields()),
            "state_format": "zarr_output.write_restart, layout canonical_global; load with "
                            "zarr_output.read_restart or core2_adj.load_state",
            "daily_fields": {k: {"unit": u, "meaning": m} for k, (u, m) in DAILY_FIELDS.items()},
            "daily_format": "daily/diag_<YYYY>_<DOY>_<SSSSS>.npz: the day that STARTS at that "
                            "time, float64 [nod2D] arrays",
            "steps_per_day": K, "dt": model.dt}
    readme.write_text(json.dumps(_jsonable(info), indent=1))
    row["readme"] = str(readme)
    if json_path is not None:
        append_row(json_path, row)
    return row


def build_states_model(start="2013-01-01", restart=None, *, chunk_steps: int | None = None,
                       ice_mode: str = "prescribed", state=None) -> Model:
    """The forward-only model of the ``states`` / ``restartcheck`` actions: one chunk = one day
    (``86400/dt`` steps, or ``chunk_steps``), the daily indicator accumulators, no remat."""
    from fesom_jax.tke import TkeConfig
    K = int(chunk_steps) if chunk_steps else int(round(86400.0 / DT))
    probe = load_yaml(CONFIG)
    tke_min = float((probe.tke or TkeConfig()).tke_min)
    return build_model(start=start, n_steps=K, chunk_steps=K, restart=restart,
                       ice_mode=ice_mode, rematerialize=False, state=state,
                       accumulators=lambda mesh: daily_accumulators(mesh, tke_min))


#: restartcheck: a restart-vs-continuous difference passes when it is at most this factor times
#: the repeat spread of the restarted day. On a GPU the spread is itself a sample from a few
#: repeats (a max over 1-4 differences), so an equally good restart lands just above it about
#: as often as just below: 3- and 5-repeat A100 checks of 2013-07-01 put surface speed 1.4e-5
#: relative and column-minimum N² 3 % over their spreads while every other field was well inside.
#: 5 % separates such ties from a restart that is really off (the State then differs by far more
#: than the day's non-determinism). With a spread of 0 (CPU) the check stays bitwise.
RESTART_TIE_MARGIN = 1.05


def _within_spread(diff: dict, spread: dict, margin: float) -> list[str]:
    """Keys whose ``diff`` exceeds ``margin × spread`` (none ⇒ pass); a zero spread needs a zero
    difference."""
    return [k for k in diff if not diff[k] <= margin * spread[k]]


def run_restartcheck(model: Model, states_dir, *, date, repeats: int = 2,
                     tie_margin: float = RESTART_TIE_MARGIN) -> dict:
    """Restart from the saved State at ``date`` in ``states_dir``, run the next chunk (day)
    ``repeats`` times and compare the daily diagnostics with the continuous run's file for that
    day, and the end State with the next saved State if one exists. ``pass`` = every
    restart-vs-continuous difference at most ``tie_margin`` × the repeat spread
    (:data:`RESTART_TIE_MARGIN`; bitwise when the spread is 0); the row records ``tie_margin``
    and the fields over it (``over_margin``)."""
    states_dir = Path(states_dir)
    when = _parse_date(date)
    st, meta = load_state(states_dir / restart_tag(when))
    K = model.chunk_steps
    fwd = adjoint._jitted_forward(model.chunk_fn)
    snap = _daily_snapshot_fn(model.mesh, K)
    xs_fn = model.xs_fn(when)
    doy = (when - datetime.datetime(when.year, 1, 1)).days + 1
    cont = np.load(states_dir / "daily" / f"diag_{when.year:04d}_{doy:03d}_"
                   f"{when.hour * 3600 + when.minute * 60:05d}.npz")
    nxt_when = when + datetime.timedelta(seconds=K * model.dt)
    nxt_dir = states_dir / restart_tag(nxt_when)
    nxt_state = load_state(nxt_dir)[0] if nxt_dir.is_dir() else None
    outs = []
    with _Timer() as t:
        for _ in range(int(repeats)):
            xs = xs_fn(0)
            out = jax.block_until_ready(fwd(model.init_carry(st), xs, model.params0))
            _free(xs)
            outs.append((jax.device_get(snap(out)), jax.device_get(out[0])))
            _free(out)

    def dmax(a, b):
        return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))

    vs_cont = {k: dmax(outs[0][0][k], cont[k]) for k in DAILY_FIELDS}
    spread = {k: max(dmax(o[0][k], outs[0][0][k]) for o in outs[1:]) if len(outs) > 1 else 0.0
              for k in DAILY_FIELDS}
    # the single-precision carry is None in a double-precision run (None vs None compared as NaN)
    fields = [f for f in zarr_output._all_state_fields()
              if getattr(outs[0][1], f) is not None or getattr(nxt_state, f, None) is not None]
    state_spread = {f: max((dmax(getattr(o[1], f), getattr(outs[0][1], f)) for o in outs[1:]),
                           default=0.0) for f in fields}
    state_vs = None
    if nxt_state is not None:
        state_vs = {f: dmax(getattr(outs[0][1], f), getattr(nxt_state, f)) for f in fields}
    over = _within_spread(vs_cont, spread, tie_margin)
    if state_vs is not None:
        over += [f"state.{f}" for f in _within_spread(state_vs, state_spread, tie_margin)]
    return provenance(model, "restartcheck", n_steps=K, start=when, timer=t,
                      states_dir=str(states_dir), restart_meta=meta, vs_continuous=vs_cont,
                      repeat_spread=spread, state_vs_next_saved=state_vs,
                      state_repeat_spread=state_spread, repeats=repeats,
                      tie_margin=float(tie_margin), over_margin=over,
                      **{"pass": not over})


# ---------------------------------------------------------------------------
# CLI


def _adjoint_cfg(args) -> AdjointConfig:
    base = AdjointConfig.long_window() if args.adjoint == "long_window" else AdjointConfig.exact()
    flags = {f if f.startswith("freeze_") else f"freeze_{f}": True for f in args.freeze}
    cfg = base._replace(**flags) if flags else base
    if args.cut:
        cfg = cfg._replace(diag_cuts=frozenset(args.cut))
    return cfg.validate()


def _ice_adjoint_args(args) -> dict:
    """The ``--ice-damping/--ice-kappa/--ice-sweeps/--ice-cuts/--ice-alpha-kappa/--ice-theta/
    --ice-thin-mass/--ice-sweep-guard`` options given, as
    ``ice_adjoint`` (``--ice-cuts advect+ssh`` -> ``("advect", "ssh")``; ``none`` -> ``()``)."""
    cuts = getattr(args, "ice_cuts", None)
    if cuts is not None:
        cuts = tuple(c for c in cuts.split("+") if c and c != "none")
    pairs = (("adjoint_damping", args.ice_damping), ("adjoint_kappa", args.ice_kappa),
             ("adjoint_sweeps", args.ice_sweeps), ("adjoint_cuts", cuts),
             ("adjoint_alpha_kappa", getattr(args, "ice_alpha_kappa", None)),
             ("adjoint_theta", getattr(args, "ice_theta", None)),
             ("adjoint_thin_mass", getattr(args, "ice_thin_mass", None)),
             ("adjoint_sweep_guard", getattr(args, "ice_sweep_guard", None)))
    return {k: v for k, v in pairs if v is not None}


def _parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["fwd", "grad", "screen", "fd", "tl", "dot", "states",
                                       "restartcheck"])
    ap.add_argument("--start", default="2013-01-01", help="window start (ISO), default 2013-01-01")
    ap.add_argument("--restart", default=None, help="restart dir or tag (default: from --start)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--steps", type=int, default=None, help="window length in steps")
    g.add_argument("--days", type=float, default=None, help="window length in days")
    ap.add_argument("--chunk-steps", type=int, default=48)
    ap.add_argument("--adjoint", choices=["exact", "long_window"], default="exact")
    ap.add_argument("--freeze", action="append", default=[], help="extra freeze flag (repeat)")
    ap.add_argument("--cut", action="append", default=[], help="named derivative cut (repeat)")
    ap.add_argument("--ice", choices=ICE_MODES, default="prescribed")
    ap.add_argument("--ice-damping", choices=("none", "local", "frozen_sweep"), default=None,
                    help="IceConfig.adjoint_damping (mEVP, with --ice exact/dynamics)")
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
    ap.add_argument("--save-names", default="", help="comma list, e.g. ssh_delta")
    ap.add_argument("--no-remat", action="store_true")
    ap.add_argument("--accumulate", action="append", default=[],
                    help=f"running accumulator (repeat): {sorted(ACCUMULATORS)}")
    ap.add_argument("--surface-adj", action="store_true",
                    help="add the [nod2D] heat/stress controls (needed for heat, stress_x)")
    ap.add_argument("--cost", action="append", default=[], help=f"cost (repeat): {COSTS}")
    ap.add_argument("--box", default=DEFAULT_BOX, choices=sorted(BOXES))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--check-linear", choices=["zero", "full", "off"], default="zero")
    ap.add_argument("--abort-at", type=float, default=ABORT_AT)
    ap.add_argument("--direction", default="T_inside", choices=DIRECTIONS)
    ap.add_argument("--hs", default="1e-2,1e-3,1e-4")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--ad", action="store_true", help="fd: also compute the autodiff value")
    ap.add_argument("--tl-constant-folding", action="store_true",
                    help="tl/dot: compile the tangent program WITH XLA constant folding (minutes "
                         "on CORE2; default: without)")
    ap.add_argument("--amps", default="1,1e-8")
    ap.add_argument("--resume-dir", default=None, help="grad: resumable sweep directory")
    ap.add_argument("--budget-hours", type=float, default=math.inf)
    ap.add_argument("--out", default=None, help="output dir (results.jsonl); default under "
                                                f"{RUNS_DIR}")
    ap.add_argument("--states-dir", default=str(STATES_DIR))
    ap.add_argument("--n-days", type=int, default=None, help="states: days to run")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--date", default=None, help="restartcheck: the saved State's date")
    ap.add_argument("--tie-margin", type=float, default=RESTART_TIE_MARGIN,
                    help="restartcheck: a difference passes when <= this × the repeat spread "
                         f"(default {RESTART_TIE_MARGIN})")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    tag = os.environ.get("SLURM_JOB_ID") or datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out = Path(args.out) if args.out else RUNS_DIR / f"{args.action}_{tag}"
    jpath = out / "results.jsonl"
    if args.action in ("states", "restartcheck"):
        if args.action == "states":
            m = build_states_model(args.start, args.restart, ice_mode=args.ice)
            n_days = args.n_days if args.n_days is not None else int(args.days or 1)
            row = run_states(m, args.states_dir, n_days=n_days, skip_existing=args.skip_existing)
        else:
            date = args.date or args.start
            m = build_states_model(date, Path(args.states_dir) / restart_tag(_parse_date(date)),
                                   ice_mode=args.ice)
            row = run_restartcheck(m, args.states_dir, date=date, repeats=max(2, args.repeats),
                                   tie_margin=args.tie_margin)
        append_row(jpath, row)
        print(json.dumps(_jsonable({k: v for k, v in row.items() if not k.startswith("_")}))[:4000])
        return 0 if row.get("pass", True) else 1
    n_steps = (args.steps if args.steps is not None
               else int(round((args.days if args.days is not None else 1.0) * 86400.0 / DT)))
    model = build_model(args.start, n_steps, args.chunk_steps, args.restart, _adjoint_cfg(args),
                        args.ice, tuple(s for s in args.save_names.split(",") if s),
                        tuple(args.accumulate) or None, not args.no_remat,
                        surface_adj=args.surface_adj, ice_adjoint=_ice_adjoint_args(args))
    costs = tuple(args.cost) or ("box_T_vw",)
    lin = {"zero": True, "full": "full", "off": False}[args.check_linear]
    tlo = None if args.tl_constant_folding else TL_COMPILER_OPTIONS
    if args.action == "fwd":
        rows = [run_fwd(model, costs, box=args.box)]
    elif args.action == "grad":
        rows = run_grad(model, costs, box=args.box, stride=args.stride, check_linear=lin,
                        resume_dir=args.resume_dir, budget_s=args.budget_hours * 3600.0)
    elif args.action == "screen":
        rows = [run_screen(model, costs[0], box=args.box, stride=args.stride, check_linear=lin,
                           abort_at=args.abort_at)]
    elif args.action == "fd":
        rows = run_fd(model, costs[0], args.direction, [float(h) for h in args.hs.split(",")],
                      box=args.box, repeats=args.repeats, ad=args.ad)
    elif args.action == "tl":
        rows = [run_tl(model, costs[0], args.direction, box=args.box, tl_options=tlo)]
    else:
        rows = run_dot(model, costs[0], args.direction, box=args.box,
                       amps=[float(a) for a in args.amps.split(",")], check_linear=lin,
                       tl_options=tlo)
    for r in rows:
        if "_result" in r:
            _save_npz(out / f"grad_{r['cost']}_{tag}.npz", r["_result"]["grad"])
        append_row(jpath, {k: v for k, v in r.items() if not k.startswith("_")})
        print(json.dumps(_jsonable({k: v for k, v in r.items() if not k.startswith("_")}))[:4000])
    return 3 if any(r.get("status") == "incomplete" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
