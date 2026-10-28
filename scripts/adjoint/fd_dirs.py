#!/usr/bin/env python
"""Directional central FD of a cost on the CORE2 adjoint model, against the adjoint (the Fram plan Task 4).

A direction is ``<kind>[:<box>[:<range>]]``:

* ``pstar``, ``cd``: an ice log control (``ice_cbar.ICE_CONTROLS``), d ln p = 1;
* a ``step.FORCING_ADJ_KEYS`` name (``Tair``, ``shum``, ``longwave``, ...): a box-uniform anomaly of 1 (the field's
  units: K, kg/kg, W/m2, m/s) on the box's nodes in the window months of ``range`` (``2013-01..2013-03``), through
  the model's monthly maps (``--forcing-controls`` must include it);
* ``wsouth``: a southward geographic wind anomaly of 1 m/s on the box's nodes in those months (rotated to the model
  frame per node, ``adjoint_costs.south_vector``: the ``u_air`` and ``v_air`` maps);
* ``T0``: an initial temperature anomaly of 1 K on the box's wet cells at the depths of ``range`` (``0-200m``).

Boxes: :data:`BOXES` (degrees; every node in the lat/lon bounds, the forcing anomalies over land too: the land nodes
have no ocean below and their gradient is 0). An FD run perturbs the model by ``± h`` times the direction.

``fd``: the base (``--save-every`` boundaries for the adjoint) and the ``±h`` forwards of every direction, in one
process (the model is compiled once), the ``tke_cbar.py fwd`` layout under ``--fd-root``: ``base``,
``<slug>_h<h>_<p|m>``. ``score``: ``J`` from each run's last ``fram_sum``, the central FD per direction and the
adjoint's directional derivative from ``--grad-dir``: the scalars (``done.json`` ``grad_ice``), the monthly maps
(``grad_forcing.npz``) and the initial T cotangent (``dump/cot_0.npz`` ``0.T``, from ``tkegrad --dump-cot 0
--dump-fields T,S``).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core2_adj as C  # noqa: E402  (runs the import guard)
import ice_cbar as IC  # noqa: E402
import tke_cbar as T  # noqa: E402

import numpy as np  # noqa: E402

from fesom_jax import adjoint_costs as AC  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402

#: lat/lon boxes [deg] (``core2_adj.Box``; the depths come from a T0 range)
BOXES = {
    "arctic70": C.Box(70.0, 90.0, -180.0, 180.0, 0.0, 1e4),     # the Arctic poleward of 70N
    "fram": C.Box(70.0, 82.0, -20.0, 15.0, 0.0, 1e4),           # Greenland Sea and Fram Strait
    "barents": C.Box(70.0, 80.0, 15.0, 60.0, 0.0, 1e4),         # the Barents Sea
    "eqwpac": C.Box(-5.0, 5.0, 120.0, 150.0, 0.0, 1e4),         # the equatorial western Pacific
}
ICE_KINDS = tuple(IC.ICE_CONTROLS)
WIND = "wsouth"


@dataclasses.dataclass(frozen=True)
class Direction:
    spec: str
    kind: str                       # an ICE_KINDS name, a FORCING_ADJ_KEYS name, WIND or "T0"
    box: str | None = None
    months: tuple = ()              # window month indices (forcing kinds)
    depth: tuple | None = None      # (z_min, z_max) [m, positive down] (T0)

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", self.spec).strip("_")


def month_index(start: datetime.date, ym: str) -> int:
    """The window month (0 = ``start``'s month) of ``YYYY-MM``."""
    y, m = (int(x) for x in ym.split("-"))
    return (y - start.year) * 12 + (m - start.month)


def parse(spec: str, start) -> Direction:
    """``spec`` (module docstring) for a window starting at ``start`` (a date or ISO string)."""
    start = datetime.date.fromisoformat(str(start)[:10])
    kind, _, rest = spec.partition(":")
    box, _, rng = rest.partition(":")
    if kind in ICE_KINDS:
        if rest:
            raise ValueError(f"direction {spec!r}: an ice control takes no box")
        return Direction(spec, kind)
    if box not in BOXES:
        raise ValueError(f"direction {spec!r}: unknown box {box!r}; known {sorted(BOXES)}")
    if kind == "T0":
        m = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)m", rng)
        if not m:
            raise ValueError(f"direction {spec!r}: T0 needs a depth range like 0-200m")
        return Direction(spec, kind, box, depth=(float(m.group(1)), float(m.group(2))))
    if kind not in stepmod.FORCING_ADJ_KEYS and kind != WIND:
        raise ValueError(f"direction {spec!r}: unknown kind {kind!r}")
    lo, sep, hi = rng.partition("..")
    if not sep:
        raise ValueError(f"direction {spec!r}: a forcing direction needs months like 2013-01..2013-03")
    a, b = month_index(start, lo), month_index(start, hi)
    if a < 0 or b < a:
        raise ValueError(f"direction {spec!r}: months {lo}..{hi} before the window start or reversed")
    return Direction(spec, kind, box, months=tuple(range(a, b + 1)))


def _nodes(mesh, box: str) -> np.ndarray:
    lon, lat = C._lonlat_deg(mesh)
    b = BOXES[box]
    return (lon >= b.lon_min) & (lon <= b.lon_max) & (lat >= b.lat_min) & (lat <= b.lat_max)


def pattern(model, d: Direction):
    """The direction as a perturbation of 1: ``{name: [M, nod2D]}`` monthly maps (forcing kinds), ``[nod2D, nl]`` (T0,
    the wet cells of the box at its depths), ``None`` (an ice control)."""
    mesh = model.mesh
    if d.kind in ICE_KINDS:
        return None
    if d.kind == "T0":
        b = BOXES[d.box]
        node, level = C.box_masks(mesh, C.Box(b.lat_min, b.lat_max, b.lon_min, b.lon_max, *d.depth))
        return (node[:, None] & level[None, :] & np.asarray(mesh.node_layer_mask, bool)).astype(np.float64)
    maps = (model.params0.get("forcing_adj") or {})
    M = int((model.schedule or {}).get("n_months") or 0)
    if max(d.months) >= M:
        raise ValueError(f"direction {d.spec!r}: month {max(d.months)} beyond the model's {M} map months")
    node = _nodes(mesh, d.box).astype(np.float64)
    if d.kind == WIND:
        sx, sy = (np.asarray(v, np.float64) for v in AC.south_vector(mesh))
        fields = {"u_air": node * sx, "v_air": node * sy}
    else:
        fields = {d.kind: node}
    out = {}
    for k, f in fields.items():
        if k not in maps:
            raise ValueError(f"direction {d.spec!r} needs the forcing control {k!r} (--forcing-controls)")
        p = np.zeros((M, int(mesh.nod2D)))
        p[list(d.months)] = f
        out[k] = p
    return out


def perturbed(model, d: Direction | None, h: float):
    """``(params, state)``: the model's controls and start State moved by ``h`` along ``d`` (``None``: the base)."""
    u = {k: (dict(v) if isinstance(v, dict) else np.array(v)) for k, v in model.params0.items()}
    state = None
    if d is not None:
        p = pattern(model, d)
        if d.kind in ICE_KINDS:
            key = IC.ICE_CONTROLS[d.kind][0]
            if key not in u:
                raise ValueError(f"direction {d.spec!r}: the model has no ice control {key} (--controls)")
            u[key] = np.asarray(u[key], np.float64) + h
        elif d.kind == "T0":
            s0 = model.s0
            T0 = np.asarray(s0.T)
            state = dataclasses.replace(s0, T=(T0.astype(np.float64) + h * p).astype(T0.dtype))
        else:
            u["forcing_adj"] = {k: (np.asarray(v, np.float64) + h * p[k] if k in p else v)
                                for k, v in u["forcing_adj"].items()}
    return C._strong_f64(u), state


def ad_value(model, d: Direction, grad_dir: Path) -> float:
    """The adjoint's ``dJ/d(direction)`` from a finished ``tkegrad`` run directory."""
    grad_dir = Path(grad_dir)
    if d.kind in ICE_KINDS:
        row = json.loads((grad_dir / "done.json").read_text())
        return float(row["grad_ice"][IC.ICE_CONTROLS[d.kind][0]])
    p = pattern(model, d)
    if d.kind == "T0":
        with np.load(grad_dir / "dump" / "cot_0.npz") as z:
            return float(np.sum(np.asarray(z["0.T"], np.float64) * p))
    with np.load(grad_dir / "grad_forcing.npz") as z:
        return float(sum(np.sum(np.asarray(z[k], np.float64) * v) for k, v in p.items()))


def j_of(run_dir: Path, n_cost: int) -> float:
    """``J_FRAM`` [km3/yr] of a finished forward: its last ``fram_sum`` over the summed steps."""
    rows = [json.loads(x) for x in (Path(run_dir) / "fwd.jsonl").read_text().splitlines() if x.strip()]
    return rows[-1]["fram_sum"] / n_cost * AC.M3S_TO_KM3YR


def build(args, n_steps: int):
    """The ``tkegrad`` model of the FD forwards: ice controls, the cost's accumulators, the schedule. The
    derivative-only ice settings are left out (they change no forward)."""
    return IC.build(args.start, n_steps, args.chunk_steps, adjoint_cfg=AdjointConfig.exact(), ice_mode=args.ice_mode,
                    controls=tuple(x for x in args.controls.split(",") if x),
                    accs=T.accumulators_for(args.cost, args.devices), tke_settings=args.tke_settings, fast=args.fast,
                    restart=args.restart, devices=args.devices, **T.schedule_kwargs(args))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["fd", "score"])
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--days", type=float, required=True)
    ap.add_argument("--chunk-steps", type=int, default=48)
    ap.add_argument("--fd-root", required=True)
    ap.add_argument("--dirs", default="", help="comma list of directions (module docstring); empty: the base only")
    ap.add_argument("--h", default="", help="the FD step per direction, comma list (one value: for all)")
    ap.add_argument("--no-base", action="store_true", help="fd: skip the base run")
    ap.add_argument("--save-every", type=int, default=24)
    ap.add_argument("--budget-hours", type=float, default=24.0)
    ap.add_argument("--cost", default="FRAM", choices=("FRAM",))
    ap.add_argument("--controls", default="pstar,cd", help="ice controls in the model")
    ap.add_argument("--ice-mode", default="exact", choices=C.ICE_MODES)
    ap.add_argument("--tke-settings", default=None)
    ap.add_argument("--fast", default=None)
    ap.add_argument("--restart", default=None)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--grad-dir", default=None, help="score: the tkegrad run directory")
    ap.add_argument("--out", default=None, help="score: JSON (default <fd-root>/fd_table.json)")
    T.add_schedule_args(ap)
    args = ap.parse_args(argv)
    n = int(round(args.days * 86400.0 / C.fast_dt(args.fast)))
    dirs = [parse(s, args.start) for s in args.dirs.split(",") if s]
    hs = [float(x) for x in args.h.split(",") if x]
    if dirs and len(hs) not in (1, len(dirs)):
        ap.error(f"--h: one value or one per direction ({len(dirs)})")
    hs = hs * len(dirs) if len(hs) == 1 else hs
    model = build(args, n)
    n_cost = T.cost_steps(model, n)
    root = Path(args.fd_root)
    job = os.environ.get("SLURM_JOB_ID", "")
    runs = ([] if args.no_base else [(None, 0.0, 1)]) + [(d, h, s) for d, h in zip(dirs, hs) for s in (1, -1)]
    if args.action == "fd":
        for d, h, sign in runs:
            rd = root / ("base" if d is None else f"{d.slug}_h{h:g}_{'p' if sign > 0 else 'm'}")
            rj = rd / "results.jsonl"
            if rj.exists() and any(json.loads(x).get("status") == "complete"
                                   for x in rj.read_text().splitlines() if x.strip()):
                print(f"skip (complete): {rd}", flush=True)
                continue
            params, state = perturbed(model, d, sign * h)
            row = T.run_fwd(model, rd, const="none" if d is None else d.spec, h=h, sign=sign, n_steps=n, ends=[n],
                            save_boundaries=d is None, budget_s=args.budget_hours * 3600.0,
                            save_every=args.save_every, params=params, state=state)
            row.update(slurm_job=job, n_cost=n_cost)
            if row["status"] == "complete":
                row["J"] = j_of(rd, n_cost)
            C.append_row(rj, {k: v for k, v in row.items() if not k.startswith("_")})
            print(f"{rd.name}: {row['status']} J {row.get('J')} first_chunk {row.get('first_chunk_s'):.1f} s, "
                  f"{row.get('s_per_step'):.4f} s/step", flush=True)
        return 0
    # score
    table = {"J_base": j_of(root / "base", n_cost), "n_cost": n_cost, "grad_dir": args.grad_dir, "rows": []}
    for d, h in zip(dirs, hs):
        jp = j_of(root / f"{d.slug}_h{h:g}_p", n_cost)
        jm = j_of(root / f"{d.slug}_h{h:g}_m", n_cost)
        fd = (jp - jm) / (2.0 * h)
        ad = ad_value(model, d, Path(args.grad_dir)) if args.grad_dir else float("nan")
        table["rows"].append({"dir": d.spec, "h": h, "J_p": jp, "J_m": jm, "fd": fd, "ad": ad,
                              "ad_over_fd": ad / fd if fd else float("nan")})
        print(f"{d.spec:>32s} h {h:g}: FD {fd:+.5e} AD {ad:+.5e} AD/FD {ad / fd if fd else float('nan'):.3f}")
    out = Path(args.out) if args.out else root / "fd_table.json"
    out.write_text(json.dumps(C._jsonable(table), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
