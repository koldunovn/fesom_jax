"""Tangent-linear exactness of the SSH solve on CORE2, zstar, production physics.

One assembled CORE2 step (zstar + TKE + GM + mEVP sea ice with the prescribed-ice adjoint,
JRA55-do forcing) taken from a non-rest state after two steps, so the SSH solve is
warm-started and its forward stop threshold comes from a non-zero ``ssh_rhs``. The zstar
operator depends on ``hbar``, so the tangent includes the ``dA·δ`` term.

Gates: the step's value is bit-identical with the reference solve (CPU); the dot test of the
step passes to ≤ 1e-10 at amplitudes 1, 1e-4 and 1e-8 while the reference solve fails it
(negative control); the reverse gradient is unchanged (≤ 1e-12 relative); and every CG in
the tangent and adjoint programs stops on its tolerance, not on ``MAXITER``.

Heavy (CPU compute node, ≥ 200 GB): four compiled one-step programs (tangent and adjoint,
with the new and with the reference solve) plus two forward steps.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ssh, surface_forcing
from fesom_jax import ice as icemod
from fesom_jax import step as stepmod
from fesom_jax.ale import AleConfig
from fesom_jax.config import MAXITER
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.tests import _ssh_implicit_common as C
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0
YEAR = 1958
TKE_SEED = 1.0e-3
AMPS = (1.0, 1e-4, 1e-8)
DOT_TOL = 1e-10

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file()),
    reason="CORE2 mesh / PHC IC missing")

ICE_CFG = IceConfig(whichEVP=1, adjoint_mode="prescribed")


def _step(first=False):
    def f(st, mesh, op, sf, fs):
        return stepmod.step(st, mesh, op, None, Params.defaults(), dt=DT,
                            is_first_step=first, step_forcing=sf, forcing_static=fs,
                            ale_cfg=AleConfig(), gm_cfg=GMConfig(), tke_cfg=TkeConfig(),
                            ice_cfg=ICE_CFG)
    return f


@pytest.fixture(scope="module")
def setup():
    """CORE2 state after two steps (TKE and sea ice seeded, both asserted) + the step-3
    forcing, and the structured directions."""
    mesh = load_mesh(MESH_DIR)
    bare = phc_initial_state(mesh, IC_DIR)
    imask = np.asarray(mesh.node_iface_mask)
    st = dataclasses.replace(
        bare, tke=jnp.asarray(np.where(imask, TKE_SEED, 0.0), jnp.float64))
    st = icemod.seed_ice(st, mesh, st.T[:, 0])
    assert float(jnp.max(st.tke)) > 0.0, "TKE seeding did nothing"
    assert int(np.count_nonzero(np.asarray(st.a_ice) > 0.0)) > 0, "no ice seeded"

    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(st.T[:, 0]))
    fs = cf.static
    dates = surface_forcing.dates_for_steps(YEAR, DT, 3)
    sfs = [cf.step_forcing(*d) for d in dates]
    st = jax.jit(_step(first=True))(st, mesh, op, sfs[0], fs)
    st = jax.jit(_step())(st, mesh, op, sfs[1], fs)
    assert float(jnp.max(jnp.abs(st.d_eta))) > 0.0
    assert float(jnp.max(jnp.abs(st.hbar))) > 0.0, "hbar = 0: the zstar dA term is dead"
    x = {"T": C.node3d(mesh, 1.0, 1), "S": C.node3d(mesh, 0.1, 0),
         "uv": C.elem_uv(mesh, 0.02, 1), "eta_n": C.node2d(mesh, 0.05, 0),
         "hbar": C.node2d(mesh, 0.05, 1)}
    y = {"T": C.node3d(mesh, 1.0, 0), "S": C.node3d(mesh, 10.0, 1),
         "uv": C.elem_uv(mesh, 50.0, 0), "eta_n": C.node2d(mesh, 20.0, 1),
         "hbar": C.node2d(mesh, 20.0, 0), "d_eta": C.node2d(mesh, 100.0, 1)}
    return mesh, op, sfs[2], fs, st, x, y


@contextlib.contextmanager
def _cg_iters_logged():
    """Turn on the ``FESOM_CG_ITERS`` trip-count print (read at trace time) and collect its
    lines (``jax.debug.print`` writes through Python's ``sys.stdout``)."""
    buf = io.StringIO()
    old = os.environ.get("FESOM_CG_ITERS")
    os.environ["FESOM_CG_ITERS"] = "1"
    try:
        with contextlib.redirect_stdout(buf):
            yield buf
            jax.effects_barrier()
    finally:
        if old is None:
            os.environ.pop("FESOM_CG_ITERS", None)
        else:
            os.environ["FESOM_CG_ITERS"] = old


@pytest.fixture(scope="module")
def dot_results(setup):
    """Dot test + one-step reverse gradient with the new solve and with the reference
    solve (the latter through a temporary patch of ``ssh.solve_ssh``)."""
    mesh, op, sf, fs, st, x, y = setup
    consts = (mesh, op, sf, fs)
    out = {}
    with _cg_iters_logged() as buf:
        out["new"] = C.step_dot_test(_step(), st, x, y, AMPS, consts=consts,
                                     return_ct=True)
    out["iters"] = C.parse_cg_iters(
        [ln for ln in buf.getvalue().splitlines() if "[cg-iters]" in ln])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ssh, "solve_ssh", C.solve_ssh_reference)
        out["ref"] = C.step_dot_test(_step(), st, x, y, AMPS, consts=consts,
                                     return_ct=True)
    return out


def test_dot_test_step_zstar(dot_results, capsys):
    rows, _ = dot_results["new"]
    with capsys.disabled():
        print("\n" + C.format_rows("CORE2 zstar, solve_ssh", rows))
    for a, _lhs, _rhs, rel, _reln in rows:
        assert rel <= DOT_TOL, f"dot test amp={a:.0e}: rel {rel:.2e} > {DOT_TOL:.0e}"


def test_dot_test_negative_control_reference_solve(dot_results, capsys):
    rows, _ = dot_results["ref"]
    with capsys.disabled():
        print("\n" + C.format_rows("CORE2 zstar, reference solve", rows))
    assert rows[-1][3] > 1e-7, (f"the reference solve passed the dot test at 1e-8 "
                                f"(rel {rows[-1][3]:.2e}): the test cannot see the defect")


def test_reverse_gradient_equals_reference(dot_results):
    _, ct_new = dot_results["new"]
    _, ct_ref = dot_results["ref"]
    for k in sorted(ct_ref):
        a, b = np.asarray(ct_new[k]), np.asarray(ct_ref[k])
        den = np.linalg.norm(b)
        assert den > 0.0, f"zero reference gradient for {k}: the test is not live"
        rel = np.linalg.norm(a - b) / den
        assert rel <= 1e-12, f"d<y,step>/d {k}: rel {rel:.2e} vs the reference path"


def test_tight_solves_stop_before_maxiter(dot_results, capsys):
    its = dot_results["iters"]
    with capsys.disabled():
        print(f"\n  CORE2 CG solves in the one-step TL + ADJ programs (it, rtol): {its}")
    assert len(its) >= 3, f"expected ≥ 3 CG solves (fwd, tangent, transpose), got {its}"
    assert all(it < MAXITER for it, _ in its), f"a CG hit MAXITER={MAXITER}: {its}"


def test_step_forward_bitwise_vs_reference(setup, monkeypatch):
    mesh, op, sf, fs, st, _x, _y = setup
    new = jax.jit(_step())(st, mesh, op, sf, fs)
    monkeypatch.setattr(ssh, "solve_ssh", C.solve_ssh_reference)
    ref = jax.jit(_step())(st, mesh, op, sf, fs)
    for f in dataclasses.fields(new):
        a, b = np.asarray(getattr(new, f.name)), np.asarray(getattr(ref, f.name))
        assert np.array_equal(a, b, equal_nan=True), \
            f"State.{f.name} differs from the reference solve"
