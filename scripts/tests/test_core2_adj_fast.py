"""The adjoint driver's fast-physics pieces (``core2_adj.build_model(fast=...)``, the fast-adjoint plan
docs/plans/20261022-core2-fast-adjoint.md): what each piece takes from ``configs/core2_gpu_fast.yaml``, that the
reference physics refuses every piece not named, and, on CORE2, a two-step window with ice every second step and
the split-explicit SSH whose gradient agrees with FD.

The config tests are light. ``test_ice2_se_window_gradient_matches_fd`` is HEAVY (CPU compute node, >= 200 GB,
~30 min: one CORE2 chunk program, its VJP, the FD forwards); it lives in the ``scripts`` group like
``test_core2_adj.py``.
"""

from __future__ import annotations

import dataclasses
import datetime
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import core2_adj as C  # noqa: E402

from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402
from fesom_jax.run_config import load_yaml  # noqa: E402
from fesom_jax.state import ICE_AVG_FIELDS, SE_FIELDS  # noqa: E402

RESTART = C.RESTART_ARCHIVE / "fesom.2013.001.00000"
START = datetime.datetime(2013, 1, 1)


def test_parse_fast_orders_and_refuses():
    assert C.parse_fast(None) == C.parse_fast("") == C.parse_fast("none") == ()
    assert C.parse_fast("se+ice2") == ("ice2", "se")
    assert C.parse_fast(["sp", "ice2"]) == ("ice2", "sp")
    with pytest.raises(ValueError, match="unknown"):
        C.parse_fast("ice3")
    # the timestep piece: dt<seconds> in the dt2700 slot; one only
    assert C.parse_fast("sp+dt2400+ice2") == ("ice2", "dt2400", "sp")
    assert C.fast_dt("ice2+dt2400") == 2400.0 and C.fast_dt("dt2700") == 2700.0 and C.fast_dt("se") == C.DT
    with pytest.raises(ValueError, match="two timesteps"):
        C.parse_fast("dt2400+dt2700")
    with pytest.raises(ValueError, match="unknown"):
        C.parse_fast("dt0")


def test_with_fast_physics_takes_the_fast_configuration_values():
    ref, fast = load_yaml(C.CONFIG), load_yaml(C.FAST_CONFIG)
    assert C.with_fast_physics(ref, ()) is ref
    got = C.with_fast_physics(ref, "ice2+se+dt2700+sp")
    assert got.ice.ice_ave_steps == fast.ice.ice_ave_steps == 2
    assert got.split_explicit() == fast.split_explicit() is not None
    assert got.dt == fast.dt == 2700.0 and got.precision == fast.precision == "single"
    # everything else stays the reference's
    assert got.ice._replace(ice_ave_steps=1) == ref.ice
    assert (got.tke, got.gm, got.ale, got.forcing, got.mesh) == (ref.tke, ref.gm, ref.ale, ref.forcing, ref.mesh)
    one = C.with_fast_physics(ref, "ice2")
    assert one.split_explicit() is None and one.dt == C.DT and one.precision == "double"
    # the SE substep count follows the window dt (the fast configuration's barotropic substep, dt 2700 / M 60 = 45 s)
    assert fast.split_explicit().M == 60
    for spec, M in (("se", 40), ("se+dt2400", 54), ("se+dt2700", 60)):
        se = C.with_fast_physics(ref, spec).split_explicit()
        assert se.M == M and se._replace(M=60) == fast.split_explicit(), spec


def test_assert_physics_refuses_every_piece_not_named():
    ref = load_yaml(C.CONFIG)
    C.assert_physics(ref)
    pieces = {
        "ice2": ("sea ice every 2 steps", C.with_fast_physics(ref, "ice2")),
        "se": ("split-explicit", C.with_fast_physics(ref, "se")),
        "dt2700": ("dt 2700", C.with_fast_physics(ref, "dt2700")),
        "sp": ("precision single", C.with_fast_physics(ref, "sp")),
    }
    for name, (msg, cfg) in pieces.items():
        with pytest.raises(AssertionError, match=msg):
            C.assert_physics(cfg)
        C.assert_physics(cfg, (name,))
    C.assert_physics(C.with_fast_physics(ref, "ice2+se+dt2700+sp"), "ice2+se+dt2700+sp")
    # a piece named but with other values than the fast configuration's
    with pytest.raises(AssertionError, match="ice_ave_steps 3"):
        C.assert_physics(dataclasses.replace(ref, ice=ref.ice._replace(ice_ave_steps=3)), "ice2")
    with pytest.raises(AssertionError, match="split_explicit"):
        C.assert_physics(dataclasses.replace(ref, ssh={"split_explicit": {"M": 50}}), "se")


def test_schedule_months_and_cost_weight():
    """The per-step schedule (core2_adj.Model.sched): the window month counts from the start month across a year and a
    leap February; the cost weight is 1 on [cost_from, cost_to) only."""
    m = C.Model.__new__(C.Model)
    m.schedule = {"n_months": 25, "cost_from": 3, "cost_to": 5}
    dates = [(2012, 31, 0.0, 1), (2012, 60, 0.0, 2), (2012, 60, 1800.0, 2), (2012, 366, 0.0, 12), (2013, 1, 0.0, 1),
             (2014, 1, 0.0, 1)]
    out = m.sched("2012-01-01", 0, dates)
    np.testing.assert_array_equal(out["month"], [0, 1, 1, 11, 12, 24])
    np.testing.assert_array_equal(out["cost_w"], [0, 0, 0, 1, 1, 0])
    m.schedule = {"n_months": 24, "cost_from": None, "cost_to": None}
    assert np.all(m.sched("2012-01-01", 0, dates[:5])["cost_w"] == 1.0)
    with pytest.raises(ValueError, match="outside"):
        m.sched("2012-01-01", 0, dates)


def test_fast_dt_and_the_restart_dt_check():
    assert C.fast_dt(None) == C.fast_dt("ice2+se") == C.DT == 1800.0
    assert C.fast_dt("dt2700") == C.fast_dt("ice2+dt2700") == 2700.0
    C._check_restart_dt(Path("r"), {"dt_stage": 1800.0}, 1800.0)
    C._check_restart_dt(Path("r"), {}, 2700.0)                 # a restart without the attribute: not checked
    with pytest.raises(ValueError, match="written at dt 1800 s, the window runs at dt 2700 s"):
        C._check_restart_dt(Path("r"), {"dt_stage": 1800.0}, 2700.0)


@pytest.mark.skipif(not ((ROOT / "data" / "mesh_core2").is_dir() and RESTART.is_dir()),
                    reason="CORE2 mesh / 2013 restart missing")
def test_dt2700_refuses_a_dt1800_restart():
    with pytest.raises(ValueError, match="written at dt 1800"):
        C.build_model(START, 1, 1, RESTART, fast="dt2700")


def test_build_model_refuses_the_pieces_not_implemented_yet():
    assert set(C.FAST_IMPLEMENTED) == set(C.FAST_PHYSICS)
    # single precision exists on the sharded path only
    with pytest.raises(NotImplementedError, match="devices > 1"):
        C.build_model(START, 1, 1, RESTART, fast="sp")


@pytest.mark.skipif(not ((ROOT / "data" / "mesh_core2").is_dir() and RESTART.is_dir()),
                    reason="CORE2 mesh / 2013 restart missing")
def test_ice2_se_window_gradient_matches_fd():
    """Two steps from the 2013-01-01 hindcast restart (a CG, ice-every-step restart) with ``fast="ice2+se"``:
    the start State gets the ice-cycle fields and flat SE rings, the rows record the pieces, and the box-T adjoint
    along T_inside agrees with central FD (the state cotangent also carries the ice-cycle fields and the SE
    rings)."""
    m = C.build_model(START, n_steps=2, chunk_steps=1, restart=RESTART, adjoint_cfg=AdjointConfig.exact(),
                      ice_mode="prescribed", accumulators=("mld",), fast="ice2+se")
    assert m.fast == ("ice2", "se") and m.phys["ice_cfg"].ice_ave_steps == 2
    assert m.phys["ale_cfg"].split_explicit == load_yaml(C.FAST_CONFIG).split_explicit()._replace(M=40)   # dt 1800
    assert all(getattr(m.s0, f) is not None for f in ICE_AVG_FIELDS + SE_FIELDS)
    # the upstream init (fesom_module.F90:600-601): counter ice_ave_steps - 1, the next step updates the ice
    assert np.all(np.asarray(m.s0.ice_steps_since_upd) == 1) and np.all(np.asarray(m.s0.ice_update) == 1)
    np.testing.assert_array_equal(m.s0.se_eta[:, 0], m.s0.eta_n)
    row = C.run_grad(m, ("box_T_vw",), check_linear=False)[0]
    assert row["status"] == "ok" and row["fast"] == ["ice2", "se"]
    g = row["_result"]["grad"]
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(g))
    d = C.direction(m, "T_inside")
    ad = C.adjoint_check.tree_vdot(row["_result"]["state_cot"], d.carry)
    fds = C.run_fd(m, "box_T_vw", d, hs=(1e-1, 1e-2))
    best = min(abs(r["fd"] - ad) / abs(ad) for r in fds)
    print(f"\n[core2_adj fast ice2+se] fd {[r['fd'] for r in fds]} ad {ad:.6e} best rel {best:.1e}")
    assert best <= 1e-4
