"""Candidate derivative-only TKE settings on the assembled CORE2 step (zstar + TKE + GM,
JRA55 1958, one step) — ``core2`` group; the column-level limits are in
``test_tke_candidates.py``.

Fixture: the PHC state with a seeded tke = 1e-3, a random velocity field (so the shear, and
with it the Prandtl number's Richardson dependence, is live: from rest S² = 0, Ri is clipped
and the Pr derivative vanishes) and, on every second node, a near-floor first interior
interface (tke = 1.5e-6), so the near-floor km_e cut has columns to act on.

* off ⇒ the identical step program (``AdjointConfig()`` with every candidate ``None``);
* on ⇒ the one-step State bitwise (two configurations cover every code path);
* each candidate changes the one-step VJP (state cotangent); ``tke_pr_cut = inf`` equals the
  ``tke.pr`` diagnostic cut bitwise; ``tke_pr_ratio`` (the step computes the raw N² next to
  the smoothed one) leaves the State bitwise and adds to the N²* cut; so does
  ``tke_pr_raw_cut``; the mixing-length masks ``tke_ml_ratio`` / ``tke_ml_raw_cut`` likewise
  leave the State bitwise and add to ``tke_ml_cut``; so does ``conv_surrogate`` (the
  convective-adjustment ramp, which changes the VJP);
* ``freeze_mixing_state``: the one-step PARAMS cotangent equals the exact one (within a step
  the closure's state inputs do not depend on the constants), the state cotangent changes.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ssh, surface_forcing
from fesom_jax.adjoint_stab import AdjointConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.tests.test_tke_cuts_core2 import (DT, IC_DIR, IN_FIELDS, MESH_DIR, YEAR,
                                                 _bitwise, _leaves, _rel_change, _step_fn)
from fesom_jax.tke import TkeConfig

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file()),
    reason="CORE2 mesh / PHC IC missing")

TKE_SEED = 1.0e-3
TKE_IFACE1 = 1.5e-6
N2_STAR = 1.0e-6
TKE_STAR = 3.0e-6
R_STAR = 2.0       # low enough to be live on the smooth PHC field
RAW_STAR = 1.0e-5  # above N2_STAR, so live where the smoothed N² keeps the derivative
VARIANTS = {
    "pr_all": dict(tke_pr_cut=math.inf),
    "pr_below": dict(tke_pr_cut=N2_STAR),
    "pr_at": dict(tke_pr_cut=N2_STAR, tke_pr_mode="at"),
    "pr_below_ratio": dict(tke_pr_cut=N2_STAR, tke_pr_ratio=R_STAR),
    "pr_below_raw": dict(tke_pr_cut=N2_STAR, tke_pr_raw_cut=RAW_STAR),
    "ml_below": dict(tke_ml_cut=N2_STAR),
    "ml_below_ratio": dict(tke_ml_cut=N2_STAR, tke_ml_ratio=R_STAR),
    "ml_below_raw": dict(tke_ml_cut=N2_STAR, tke_ml_raw_cut=RAW_STAR),
    "conv_surrogate": dict(conv_surrogate=N2_STAR),
    "kme_column": dict(tke_kme_tke=TKE_STAR),
    "kme_ke_diff": dict(tke_kme_tke=TKE_STAR, tke_kme_term="ke_diff"),
    "mixing_state": dict(freeze_mixing_state=True),
    "mixing_state_kme": dict(freeze_mixing_state=True, tke_kme_tke=TKE_STAR),
}


@pytest.fixture(scope="module")
def core2():
    mesh = load_mesh(MESH_DIR)
    state = phc_initial_state(mesh, IC_DIR)
    imask = np.asarray(mesh.node_iface_mask)
    tke = np.where(imask, TKE_SEED, 0.0)
    tke[::2, 1] = np.where(imask[::2, 1], TKE_IFACE1, 0.0)
    rng = np.random.default_rng(21)
    emask = np.asarray(mesh.elem_layer_mask)[..., None]
    uv = np.where(emask, 0.05 * rng.standard_normal(np.shape(state.uv)), 0.0)
    state = dataclasses.replace(state, tke=jnp.asarray(tke, jnp.float64),
                                uv=jnp.asarray(uv, jnp.float64))
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    sf = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
    return dict(mesh=mesh, state=state, op=op, sf=sf, fs=cf.static)


def _args(c):
    return (c["state"], Params.defaults(), c["mesh"], c["op"], c["sf"], c["fs"])


def test_off_traces_the_identical_step(core2):
    ref = str(jax.make_jaxpr(_step_fn(None))(*_args(core2)))
    off = AdjointConfig(tke_pr_cut=None, tke_pr_mode=None, tke_pr_ratio=None,
                        tke_pr_raw_cut=None, tke_kme_tke=None, tke_kme_scope=None,
                        tke_kme_term=None, freeze_mixing_state=False)
    assert off == AdjointConfig() and off.tke_options() is None
    assert str(jax.make_jaxpr(_step_fn(off))(*_args(core2))) == ref
    on = str(jax.make_jaxpr(_step_fn(AdjointConfig(**VARIANTS["pr_below"])))(*_args(core2)))
    assert on != ref


@pytest.mark.parametrize("kw", [
    dict(tke_pr_cut=N2_STAR, tke_kme_tke=TKE_STAR, freeze_mixing_state=True),
    dict(tke_pr_cut=N2_STAR, tke_pr_mode="at", tke_kme_tke=TKE_STAR, tke_kme_scope="iface1",
         tke_kme_term="ke_diff"),
    dict(tke_pr_cut=N2_STAR, tke_pr_ratio=R_STAR, tke_kme_tke=TKE_STAR),
    dict(tke_pr_cut=N2_STAR, tke_pr_raw_cut=RAW_STAR),
    dict(tke_ml_cut=N2_STAR, tke_ml_ratio=R_STAR, tke_ml_raw_cut=RAW_STAR),
    dict(tke_pr_cut=N2_STAR, conv_surrogate=N2_STAR),
], ids=["cut_column_kme_state", "at_iface1_kediff", "cut_ratio_kme", "cut_raw", "ml_masks",
        "conv_surrogate"])
def test_candidates_leave_the_core2_step_bitwise(core2, kw):
    ref = jax.jit(_step_fn(None))(*_args(core2))
    got = jax.jit(_step_fn(AdjointConfig(**kw)))(*_args(core2))
    for f in dataclasses.fields(ref):
        a, b = getattr(ref, f.name), getattr(got, f.name)
        if a is None:
            assert b is None
            continue
        for u, v in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            assert np.array_equal(np.asarray(u), np.asarray(v)), f"State.{f.name} moved"


_VJP: dict = {}


def _cotangent(c):
    rng = np.random.default_rng(5)
    st = c["state"]
    scale = {"T": 1.0, "S": 10.0, "uv": 50.0, "eta_n": 20.0, "tke": 1e3}
    return {k: jnp.asarray(rng.standard_normal(np.shape(getattr(st, k))) * scale[k])
            for k in IN_FIELDS}


def _one_step_vjp(c, cfg):
    key = cfg.validate()
    if key in _VJP:
        return _VJP[key]
    fn = _step_fn(cfg)

    def run(xin, p, st, y, mesh, op, sf, fs):
        def f(xin, p):
            out = fn(dataclasses.replace(st, **xin), p, mesh, op, sf, fs)
            return {k: getattr(out, k) for k in y}
        _, vjp = jax.vjp(f, xin, p)
        return vjp(y)

    st = c["state"]
    xin = {k: getattr(st, k) for k in IN_FIELDS}
    res = jax.device_get(jax.jit(run)(xin, Params.defaults(), st, _cotangent(c), c["mesh"],
                                      c["op"], c["sf"], c["fs"]))
    _VJP[key] = res
    return res


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_candidate_changes_the_core2_vjp(core2, name, capsys):
    base = _one_step_vjp(core2, AdjointConfig())
    got = _one_step_vjp(core2, AdjointConfig(**VARIANTS[name]))
    assert all(np.all(np.isfinite(x)) for x in _leaves(got))
    rs, rp = _rel_change(base[0], got[0]), _rel_change(base[1], got[1])
    with capsys.disabled():
        print(f"\n  CORE2 {name}: one-step VJP relative change state {rs:.3e}, params {rp:.3e}")
    assert rs > 1e-12, f"{name} did not change the CORE2 state cotangent"


def test_pr_cut_everywhere_equals_the_diagnostic_cut_on_core2(core2, capsys):
    """``inf`` is the ``tke.pr`` stop_gradient itself (bitwise). The MASKED path with every
    cell below N²* (1e30) is a different program (a custom_jvp that zeroes the tangent): equal
    to round-off, not bitwise — XLA fuses the explicit zeros differently."""
    a = _one_step_vjp(core2, AdjointConfig(**VARIANTS["pr_all"]))
    b = _one_step_vjp(core2, AdjointConfig(diag_cuts=frozenset({"tke.pr"})))
    assert _bitwise(a[0], b[0]) and _bitwise(a[1], b[1])
    m = _one_step_vjp(core2, AdjointConfig(tke_pr_cut=1e30))
    r = _rel_change(b[0], m[0])
    with capsys.disabled():
        print(f"\n  masked Pr cut (N2* = 1e30) vs tke.pr: state cotangent rel {r:.3e}")
    assert r < 1e-12


def test_pr_ratio_adds_to_the_n2_cut_on_core2(core2, capsys):
    a = _one_step_vjp(core2, AdjointConfig(**VARIANTS["pr_below"]))
    b = _one_step_vjp(core2, AdjointConfig(**VARIANTS["pr_below_ratio"]))
    r = _rel_change(a[0], b[0])
    with capsys.disabled():
        print(f"\n  CORE2 ratio cut (R* = {R_STAR}) on top of N2* = {N2_STAR}: state cotangent "
              f"rel {r:.3e}")
    assert r > 1e-12, "the ratio cut is dead on the CORE2 fixture"


def test_pr_raw_cut_adds_to_the_n2_cut_on_core2(core2, capsys):
    a = _one_step_vjp(core2, AdjointConfig(**VARIANTS["pr_below"]))
    b = _one_step_vjp(core2, AdjointConfig(**VARIANTS["pr_below_raw"]))
    r = _rel_change(a[0], b[0])
    with capsys.disabled():
        print(f"\n  CORE2 raw-N2 cut (N2_raw* = {RAW_STAR}) on top of N2* = {N2_STAR}: state "
              f"cotangent rel {r:.3e}")
    assert r > 1e-12, "the raw-N2 cut is dead on the CORE2 fixture"


@pytest.mark.parametrize("name", ["ml_below_ratio", "ml_below_raw"])
def test_ml_masks_add_to_the_n2_cut_on_core2(core2, name, capsys):
    a = _one_step_vjp(core2, AdjointConfig(**VARIANTS["ml_below"]))
    b = _one_step_vjp(core2, AdjointConfig(**VARIANTS[name]))
    r = _rel_change(a[0], b[0])
    with capsys.disabled():
        print(f"\n  CORE2 {name} on top of the mixing-length N2* = {N2_STAR}: state cotangent "
              f"rel {r:.3e}")
    assert r > 1e-12, f"{name} is dead on the CORE2 fixture"


def test_mixing_state_keeps_the_one_step_params_gradient(core2):
    base = _one_step_vjp(core2, AdjointConfig())
    got = _one_step_vjp(core2, AdjointConfig(freeze_mixing_state=True))
    for n in ("tke_c_k", "tke_c_eps", "tke_cd", "tke_alpha"):
        a, b = float(getattr(base[1], n)), float(getattr(got[1], n))
        assert a != 0.0 and abs(a - b) <= 1e-12 * abs(a), (n, a, b)
    assert _rel_change(base[0], got[0]) > 1e-6
