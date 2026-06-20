"""Unit tests for the gather-free State health diagnostics (Task B2, the NG5 stability gate).

A tiny synthetic State (``State.zeros`` over a 4-node/3-elem/5-level fake mesh) exercises the
finite / non-finite / over-magnitude branches without loading a real mesh or any device mesh —
the same ``state_diagnostics`` runs unchanged on the production folded multi-process State.
"""
import dataclasses
import types

import jax.numpy as jnp

from fesom_jax.diagnostics import (format_diagnostics, state_diagnostics, verdict)
from fesom_jax.state import State

_FAKE_MESH = types.SimpleNamespace(nod2D=4, elem2D=3, nl=5)


def _zero_state():
    return State.zeros(_FAKE_MESH)


def test_zeros_state_is_finite_and_clean():
    d = state_diagnostics(_zero_state())
    assert d["n_nonfinite"] == 0
    assert d["nonfinite_by_field"] == {}
    ok, warns = verdict(d)
    assert ok is True
    assert warns == []                     # all-zero is within every sanity bound
    # every leaf reduced to a finite scalar
    assert all(v == v for v in d["field_maxabs"].values())
    assert "FINITE" in format_diagnostics(d, label="zeros")


def test_nan_in_tracer_is_caught_and_localized():
    st = _zero_state()
    st = dataclasses.replace(st, T=st.T.at[0, 0].set(jnp.nan))
    d = state_diagnostics(st)
    assert d["n_nonfinite"] == 1
    assert d["nonfinite_by_field"] == {"T": 1}
    ok, _ = verdict(d)
    assert ok is False
    assert "NON-FINITE" in format_diagnostics(d)
    assert "T=1" in format_diagnostics(d)   # the per-leaf breakdown names the field


def test_inf_in_velocity_fails_the_gate():
    st = _zero_state()
    st = dataclasses.replace(st, uv=st.uv.at[0, 0, 0].set(jnp.inf))
    d = state_diagnostics(st)
    assert d["n_nonfinite"] == 1
    ok, _ = verdict(d)
    assert ok is False


def test_finite_but_overspeed_passes_with_warning():
    st = _zero_state()
    st = dataclasses.replace(st, uv=st.uv.at[0, 0, 0].set(50.0))   # 50 m/s — finite but absurd
    d = state_diagnostics(st)
    assert d["n_nonfinite"] == 0
    assert d["max_abs_uv"] == 50.0
    ok, warns = verdict(d)
    assert ok is True                      # finite ⇒ hard gate passes
    assert any("max|uv|" in w for w in warns)


def test_tracer_range_warning():
    st = _zero_state()
    st = dataclasses.replace(st, S=st.S.at[1, 0].set(99.0))        # absurd salinity, finite
    d = state_diagnostics(st)
    ok, warns = verdict(d)
    assert ok is True
    assert any("S_max" in w for w in warns)
