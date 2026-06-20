"""Gather-free State health diagnostics (Task B2 — the NG5 stability gate).

The NG5 cold spin-up headline risk is a **blow-up**: a CFL violation / negative layer
thickness drives the fields non-finite within a few hundred steps (a prior JAX NG5 ran only
25 steps). :func:`state_diagnostics` turns a (possibly multi-node sharded) :class:`State`
into a handful of **scalar** health numbers using only global reductions —
``jnp.sum(~isfinite)`` and ``jnp.max(jnp.abs(...))`` — so the only thing crossing the device
mesh is a scalar all-reduce, never a gathered global field. That makes it cheap enough to
call **in-job** on the final State of a 7.4 M-node / 64-GPU run (no restart re-read, no
host pull of a global array), and it works identically on a dense single-device State (the
unit-test path) and a folded ``[P*Lmax]`` multi-process sharded State (the production path).

The verdict is deliberately conservative: the HARD gate is **finiteness** (a real blow-up
makes wet lanes NaN/Inf — padding/dry lanes stay 0 and never raise a false alarm); field
magnitudes are advisory CFL/physical-range sanity bounds (a blow-up also inflates ``max|u|``
/ ``max|T|`` long before everything is NaN, so they catch a divergence that is *on its way*
but not yet non-finite). Min-layer-thickness and volume-weighted KE/ice diagnostics need the
mesh masks (padding lanes read 0) and live in :mod:`scripts.ng5_ladder_check`, which has the
partition; here we stay mask-free so the function is a pure State→scalars map.
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from .state import State

# Advisory sanity bounds (NOT the hard gate — finiteness is). A cold ocean spin-up should
# never approach these; exceeding one while still finite means a divergence is building.
UV_SANITY = 20.0      # |horizontal velocity| [m/s] — real surface currents < ~3
W_SANITY = 1.0        # |vertical velocity| [m/s]
ETA_SANITY = 30.0     # |SSH| [m]
T_RANGE = (-5.0, 45.0)   # potential temperature [°C]
S_RANGE = (0.0, 60.0)    # salinity [psu]


def state_diagnostics(state: State, *, fields=None) -> dict:
    """Reduce a ``State`` to scalar health numbers via gather-free global reductions.

    Returns a dict with: ``n_nonfinite`` (total NaN/Inf count across the requested leaves)
    and ``nonfinite_by_field`` (the per-leaf breakdown, only the non-zero entries);
    ``field_maxabs`` (``max|·|`` per leaf — a blow-up anywhere is visible); and the named
    physical indicators ``max_abs_uv`` / ``max_abs_uvnode`` / ``max_abs_w`` / ``max_abs_eta``
    and the tracer ranges ``T_min`` / ``T_max`` / ``S_min`` / ``S_max`` plus ``a_ice_max`` /
    ``m_ice_max``. All values are HOST python scalars (each reduction blocks on a single
    replicated all-reduce). ``fields`` restricts the leaf scan (default: every State leaf)."""
    names = fields if fields is not None else [f.name for f in dataclasses.fields(State)]

    nonfinite_by_field, field_maxabs = {}, {}
    n_nonfinite = 0
    for name in names:
        arr = getattr(state, name)
        nf = int(jnp.sum(~jnp.isfinite(arr)))         # padding/dry lanes are 0 ⇒ finite ⇒ no false +
        n_nonfinite += nf
        if nf:
            nonfinite_by_field[name] = nf
        # raw max|·| — if this leaf has a NaN it reads NaN (informative: pinpoints the field)
        field_maxabs[name] = float(jnp.max(jnp.abs(arr)))

    def _maxabs(a):
        return float(jnp.max(jnp.abs(a)))

    return {
        "n_nonfinite": n_nonfinite,
        "nonfinite_by_field": nonfinite_by_field,
        "field_maxabs": field_maxabs,
        "max_abs_uv": _maxabs(state.uv),
        "max_abs_uvnode": _maxabs(state.uvnode),
        "max_abs_w": _maxabs(state.w),
        "max_abs_eta": _maxabs(state.eta_n),
        "T_min": float(jnp.min(state.T)), "T_max": float(jnp.max(state.T)),
        "S_min": float(jnp.min(state.S)), "S_max": float(jnp.max(state.S)),
        "a_ice_max": float(jnp.max(state.a_ice)),
        "m_ice_max": float(jnp.max(state.m_ice)),
    }


def verdict(diags: dict) -> tuple:
    """``(ok, warnings)`` from a :func:`state_diagnostics` dict.

    ``ok`` is the HARD gate: **finite** (``n_nonfinite == 0`` and no leaf ``max|·|`` is NaN/Inf).
    ``warnings`` is a list of advisory strings for finite-but-suspicious magnitudes (velocity /
    SSH / tracer-range sanity) — a divergence that is building but has not yet gone non-finite.
    A blow-up trips ``ok=False``; an over-bound-but-finite state passes with a warning so the
    caller can decide (R0 gates on ``ok``; the magnitudes inform the diagnosis)."""
    finite = diags["n_nonfinite"] == 0 and all(
        v == v and abs(v) != float("inf") for v in diags["field_maxabs"].values())
    w = []
    if diags["max_abs_uv"] > UV_SANITY:
        w.append(f"max|uv|={diags['max_abs_uv']:.3g} m/s > {UV_SANITY}")
    if diags["max_abs_w"] > W_SANITY:
        w.append(f"max|w|={diags['max_abs_w']:.3g} m/s > {W_SANITY}")
    if diags["max_abs_eta"] > ETA_SANITY:
        w.append(f"max|eta|={diags['max_abs_eta']:.3g} m > {ETA_SANITY}")
    if not (T_RANGE[0] <= diags["T_min"] and diags["T_max"] <= T_RANGE[1]):
        w.append(f"T range [{diags['T_min']:.3g}, {diags['T_max']:.3g}] outside {T_RANGE}")
    if not (diags["S_max"] <= S_RANGE[1]):   # S_min reads 0 from padding ⇒ only flag the high end
        w.append(f"S_max={diags['S_max']:.3g} > {S_RANGE[1]}")
    return finite, w


def format_diagnostics(diags: dict, *, label: str = "") -> str:
    """A compact multi-line report of a :func:`state_diagnostics` dict + the :func:`verdict`."""
    ok, warns = verdict(diags)
    head = f"[diagnostics{(' ' + label) if label else ''}] " \
           f"{'FINITE' if ok else 'NON-FINITE'}  n_nonfinite={diags['n_nonfinite']}"
    lines = [head,
             f"  max|uv|={diags['max_abs_uv']:.4g}  max|uvnode|={diags['max_abs_uvnode']:.4g}  "
             f"max|w|={diags['max_abs_w']:.4g}  max|eta|={diags['max_abs_eta']:.4g}",
             f"  T[{diags['T_min']:.4g},{diags['T_max']:.4g}]  "
             f"S[{diags['S_min']:.4g},{diags['S_max']:.4g}]  "
             f"a_ice_max={diags['a_ice_max']:.4g}  m_ice_max={diags['m_ice_max']:.4g}"]
    if diags["nonfinite_by_field"]:
        lines.append("  non-finite leaves: " + ", ".join(
            f"{k}={v}" for k, v in diags["nonfinite_by_field"].items()))
    for warn in warns:
        lines.append("  WARN: " + warn)
    return "\n".join(lines)
