"""A6 gate: the single-invocation config-driven run driver (:mod:`fesom_jax.run`) — ``RUN_DRIVER_OK``.

The headline invariant is the **restart seam**: a run done as ONE invocation equals the SAME run done
as two chained invocations (run N/2 → write portable restart → read it back → run N/2), to the climate-
close floor — the proof that the A1 portable restart + the A4 ``bootstrap_ab2`` AB2-continuation carry
the State across a job boundary correctly (a continuation chunk must NOT cold-start AB2). Plus the pure
chunk-planning / duration-parsing logic and a single-step run.

  PY -m pytest fesom_jax/tests/test_run_entry.py -x
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import core2_forcing, partit, shard_mesh, ssh
from fesom_jax.mesh import load_mesh
from fesom_jax.run import Chunk, parse_duration, plan_chunks, run_from_config
from fesom_jax.run_config import DtRamp, RunConfig
from fesom_jax.state import State
from fesom_jax.kpp import KppConfig

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 1800.0
YEAR = 1958


def _have_jra():
    from fesom_jax import jra55
    return Path(jra55.DEFAULT_JRA_DIR).is_dir()


have_forcing = pytest.mark.skipif(
    not (IC_DIR / "T_ic.npy").exists() or not CORE2_MESH.is_dir() or not _have_jra(),
    reason="CORE2 PHC IC / mesh / JRA55 forcing missing (compute node only)")


# ==========================================================================
# 1. Pure logic — duration parsing + chunk planning (always run, fast)
# ==========================================================================
def test_parse_duration():
    assert parse_duration(7, DT) == 7
    assert parse_duration("5step", DT) == 5
    assert parse_duration("1d", DT) == 48                 # 86400/1800
    assert parse_duration("3mo", DT) == round(3 * 30 * 86400 / DT)
    assert parse_duration("2yr", DT) == round(2 * 365 * 86400 / DT)
    assert parse_duration("180", 180.0) == 180            # bare number ⇒ steps


def test_plan_chunks_cold_and_continuation():
    # cold run: bootstrap AB2 only at the very first step
    ch = plan_chunks(10, 4, start_step=0, dt=180.0)
    assert [(c.start, c.count, c.bootstrap_ab2) for c in ch] == \
        [(0, 4, True), (4, 4, False), (8, 2, False)]
    # restart-continuation (start>0): the first chunk CONTINUES AB2 (no cold bootstrap)
    assert plan_chunks(4, 4, start_step=8, dt=180.0)[0].bootstrap_ab2 is False


def test_plan_chunks_dt_ramp():
    # the dt-ramp splits the chunking at the boundary; the post-ramp chunk uses the NEW dt and
    # re-bootstraps AB2 (the dt change invalidates the AB2 history formed at the old dt)
    ch = plan_chunks(8, 10, start_step=0, dt_ramp=DtRamp(after_step=4, dt=240.0), dt=180.0)
    assert ch == [Chunk(0, 4, 180.0, True), Chunk(4, 4, 240.0, True)]


# ==========================================================================
# 2. The restart seam — continuous == chained (the headline gate)
# ==========================================================================
@pytest.fixture(scope="module")
def core2_setup():
    from fesom_jax.phc_ic import core2_initial_state
    mesh = load_mesh(CORE2_MESH)
    state = core2_initial_state(mesh, IC_DIR)
    cf = core2_forcing.build_core_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    sop = ssh.partition_ssh_operator(ssh.build_ssh_operator(mesh, dt=DT), part)
    # well-separated forcing dates ⇒ a genuine per-step seasonal swing (cf. test_forcing_sharded)
    stack = cf.stack([(YEAR, 15, 0.0, 1), (YEAR, 105, 0.0, 4),
                      (YEAR, 196, 0.0, 7), (YEAR, 288, 0.0, 10)])
    return dict(mesh=mesh, state=state, cf=cf, part=part, sm=sm, sop=sop, stack=stack, n=4)


def _owned_worst(a, b, part, sm, npes):
    worst = 0.0
    for fld in dataclasses.fields(State):
        A, B = np.asarray(getattr(a, fld.name)), np.asarray(getattr(b, fld.name))
        if A.shape[1] == sm.Lmax["nod"]:
            myd = part.myDim_nod2D
        elif A.shape[1] == sm.Lmax["elem"]:
            myd = part.myDim_elem2D
        else:
            continue
        for d in range(npes):
            md = int(myd[d])
            if md:
                worst = max(worst, float(np.max(np.abs(A[d, :md] - B[d, :md]))))
    return worst


@have_forcing
def test_restart_seam_continuous_equals_chained(core2_setup, tmp_path):
    fx = core2_setup
    common = dict(mesh=fx["mesh"], part=fx["part"], sm=fx["sm"], sop=fx["sop"], year=YEAR)
    kw = dict(kpp=KppConfig(), dt=DT)
    n = fx["n"]                                            # 4 steps, split 2 + 2

    # continuous: one invocation, n steps, one chunk
    cont = run_from_config(RunConfig(n_steps=n, **kw), state0=fx["state"],
                           forcing=fx["cf"], forcing_stack=fx["stack"], **common)

    # chained: invocation 1 runs n//2 and writes a portable restart …
    r1 = tmp_path / "restart1"
    half = jax.tree.map(lambda x: x[: n // 2], fx["stack"])
    run_from_config(RunConfig(n_steps=n // 2, restart_out=str(r1), **kw), state0=fx["state"],
                    forcing=fx["cf"], forcing_stack=half, **common)
    # … invocation 2 READS it back and continues the remaining n//2 (start_step from the restart)
    rest = jax.tree.map(lambda x: x[n // 2:], fx["stack"])
    chained = run_from_config(RunConfig(n_steps=n // 2, restart_in=str(r1), **kw),
                              forcing=fx["cf"], forcing_stack=rest, **common)

    assert chained.step == n, f"resumed step counter {chained.step} != {n}"
    worst = _owned_worst(cont.state_p, chained.state_p, fx["part"], fx["sm"], 1)
    # restart round-trip is bit-faithful (A1) + AB2 continues ⇒ chained == continuous to the FCT
    # climate-close floor (the one-scan vs two-scan reassociation; a cold-AB2 bug would be O(1)).
    assert worst < 5e-3, f"restart-seam continuous-vs-chained owned max|Δ|={worst:.3e}"
    print(f"RUN_DRIVER_OK (restart-seam max|Δ|={worst:.2e})")


@have_forcing
def test_single_step_run(core2_setup):
    fx = core2_setup
    res = run_from_config(RunConfig(n_steps=1, kpp=KppConfig(), dt=DT), mesh=fx["mesh"],
                          part=fx["part"], sm=fx["sm"], sop=fx["sop"], state0=fx["state"],
                          forcing=fx["cf"], forcing_stack=jax.tree.map(lambda x: x[:1], fx["stack"]),
                          year=YEAR)
    assert res.step == 1
    assert np.all(np.isfinite(np.asarray(res.state_p.T)))
