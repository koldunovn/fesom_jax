"""A4 gate: per-step time-varying forcing on the SHARDED multi-step path (``SHARDED_FORCING_OK``).

:func:`fesom_jax.integrate_sharded.run_steps_sharded` holds the surface forcing CONSTANT across
its ``lax.scan`` (correct only for TIMING — the per-step cost is forcing-value-independent). The
real seasonal cycle NG5/dars need a forcing that CHANGES every step:
:func:`~fesom_jax.integrate_sharded.run_steps_sharded_forced` folds the partitioned forcing as a
``PartitionSpec(None,'p')`` scan ``xs`` so step ``i`` consumes ``step_forcing_seq[i]``.

Two gates:
  * **fold wiring** (always runs, pure host): ``_fold_forcing_seq`` delivers exactly the per-step
    forcing — its ``[n_steps, P*Lmax]`` step ``i`` == the independent fold of the single step ``i``;
  * **dynamics** (gated on the CORE2 PHC IC + JRA55, like the other sharded forced gates): the
    forced N-step sharded run == the dense forced :func:`fesom_jax.integrate.integrate`, byte-id at
    ``npes=1`` and on OWNED nodes at ``npes=2`` (the constant path stays a SEPARATE function ⇒
    byte-unchanged).

  XLA_FLAGS=--xla_force_host_platform_device_count=4 PY -m pytest fesom_jax/tests/test_forcing_sharded.py -x
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import core2_forcing, partit, shard_mesh, ssh
from fesom_jax import integrate as integ
from fesom_jax import integrate_sharded as ish
from fesom_jax.core2_forcing import StepForcing
from fesom_jax.kpp import KppConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.state import State

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NDEV = len(jax.devices())
DT = 1800.0
YEAR = 1958
_BYTE_ID_ATOL = 1e-9 if jax.devices()[0].platform == "cpu" else 1e-7


def _have_jra():
    from fesom_jax import jra55
    return Path(jra55.DEFAULT_JRA_DIR).is_dir()


have_forcing = pytest.mark.skipif(
    not (IC_DIR / "T_ic.npy").exists() or not CORE2_MESH.is_dir() or not _have_jra(),
    reason="CORE2 PHC IC / mesh / JRA55 forcing missing (compute node only)")
have_dist2 = pytest.mark.skipif(
    not (CORE2_DIST / "dist_2").is_dir(), reason="CORE2 dist_2 partition missing")


# ==========================================================================
# 1. Fold wiring — _fold_forcing_seq delivers exactly the per-step forcing
# ==========================================================================
def test_fold_forcing_seq_delivers_per_step():
    from jax.sharding import PartitionSpec
    nod2D, elem2D, edge2D, npes, T = 20, 30, 40, 3, 4
    part = partit.synth_block_partition(nod2D, elem2D, edge2D, npes)
    fields = StepForcing._fields

    # synthetic global stack [T, nod2D] with a distinct (field, step, node) signature
    base = {name: ((i + 1) * 1e6 + np.arange(T)[:, None] * 1e3
                   + np.arange(nod2D)[None, :]).astype(float)
            for i, name in enumerate(fields)}
    stack = StepForcing(**{name: base[name] for name in fields})

    seq_p = shard_mesh.partition_step_forcing(stack, part)        # [P, T, Lmax]
    seq_folded, seq_spec = ish._fold_forcing_seq(seq_p)           # [T, P*Lmax]

    # the scan xs at step t must equal the INDEPENDENT fold of single step t
    for t in range(T):
        single = StepForcing(**{name: base[name][t] for name in fields})
        single_p = shard_mesh.partition_step_forcing(single, part)   # [P, Lmax]
        ref_folded, _ = ish._fold_forcing(single_p)                 # [P*Lmax]
        for name in fields:
            got = np.asarray(getattr(seq_folded, name))[t]
            exp = np.asarray(getattr(ref_folded, name))
            assert np.array_equal(got, exp), f"{name} step {t}: seq fold != single-step fold"

    # the spec shards the folded device axis, replicates the time axis
    assert getattr(seq_spec, fields[0]) == PartitionSpec(None, "p")


# ==========================================================================
# 2. Dynamics — forced N-step sharded == dense forced integrate
# ==========================================================================
@pytest.fixture(scope="module")
def core2_forced_seq():
    from fesom_jax.phc_ic import core2_initial_state
    mesh = load_mesh(CORE2_MESH)
    state = core2_initial_state(mesh, IC_DIR)
    sst0 = np.asarray(state.T[:, 0])
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, YEAR, sst_ic=sst0)
    # Hand-pick WELL-SEPARATED dates (Jan / Jul / Oct) so the forcing genuinely VARIES per step —
    # JRA55 is piecewise-constant over its 3-hourly interval, so consecutive 30-min steps would
    # share one record (Tair constant) and the per-step gate couldn't tell a per-step wiring from a
    # constant-forcing bug. The dense + sharded paths consume the SAME stack ⇒ the comparison is
    # valid regardless of physical date continuity (the timestep is dt; the forcing is what we inject).
    dates = [(YEAR, 15, 0.0, 1), (YEAR, 196, 0.0, 7), (YEAR, 288, 0.0, 10)]
    stack = cf.stack(dates)                                       # [n, nod2D] leaves
    f = np.asarray(stack.Tair)
    assert not np.allclose(f[0], f[1]) and not np.allclose(f[1], f[2]), \
        "forcing must vary per step for a meaningful per-step gate"
    return dict(mesh=mesh, state=state, op=op, stack=stack, fs=cf.static, n=len(dates))


def _dense_forced(fx, stack=None):
    mesh = fx["mesh"]
    return integ.integrate(fx["state"], mesh, fx["op"], jnp.zeros((mesh.elem2D, 2)),
                           fx["n"], dt=DT, step_forcings=fx["stack"] if stack is None else stack,
                           forcing_static=fx["fs"], kpp_cfg=KppConfig())


def _worst_diff(a_state, b_state, *, fold_b=False):
    worst = 0.0
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(a_state, fld.name))
        b = np.asarray(getattr(b_state, fld.name))
        if fold_b:
            b = b[0][: a.shape[0]]
        if a.size:
            worst = max(worst, float(np.max(np.abs(a - b))))
    return worst


def _sharded_forced(fx, part, npes):
    mesh = fx["mesh"]
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(fx["state"], part)
    sop = ssh.partition_ssh_operator(fx["op"], part)
    seq_p = shard_mesh.partition_step_forcing(fx["stack"], part)
    fs_p = shard_mesh.partition_forcing_static(fx["fs"], part)
    stress_p = jnp.zeros((npes, sm.Lmax["elem"], 2))
    return ish.run_steps_sharded_forced(sm, state_p, sop, stress_p, seq_p, fs_p, fx["n"],
                                        dt=DT, npes=npes, kpp_cfg=KppConfig())


@have_forcing
def test_forced_multistep_serial_matches_dense(core2_forced_seq):
    """npes=1: the forced multi-step sharded scan reproduces the dense forced integrate, AND the
    per-step forcing genuinely matters.

    Two-sided gate: (a) sharded-forced ≈ dense-forced to the climate-close FCT floor (over a few
    steps the FCT tracer limiter's upwind flips put the floor at ~1e-4, not byte-id, once the
    forcing drives real tracer evolution); (b) the per-step result differs from a CONSTANT-forcing
    run by **≫ that match** — so a mis-wired fold (feeding one step's forcing to all) could NOT
    pass (a) by accident. Together with the byte-exact fold-unit test, this pins the wiring."""
    fx = core2_forced_seq
    ser = partit.synth_serial(fx["mesh"].nod2D, fx["mesh"].elem2D, fx["mesh"].edge2D)
    dense = _dense_forced(fx)
    sharded = _sharded_forced(fx, ser, 1)

    match = _worst_diff(dense, sharded, fold_b=True)            # sharded-forced vs dense-forced
    # constant-forcing reference (step-0 forcing held for all n steps) — what a fold bug would give
    const_stack = jax.tree.map(lambda x: jnp.broadcast_to(x[:1], x.shape), fx["stack"])
    dense_const = _dense_forced(fx, stack=const_stack)
    sensitivity = _worst_diff(dense, dense_const)              # how much per-step forcing moves it

    assert match < 5e-3, f"sharded-forced vs dense-forced max|Δ|={match:.3e} (FCT floor 5e-3)"
    assert sensitivity > 20 * match, (
        f"per-step forcing barely changes the state (Δ={sensitivity:.3e}) vs the sharded-vs-dense "
        f"match ({match:.3e}) — the gate would be trivial; pick more-separated forcing dates")
    print(f"SHARDED_FORCING_OK (match={match:.2e}, per-step sensitivity={sensitivity:.2e})")


@have_forcing
@have_dist2
@pytest.mark.parametrize("npes", [2])
def test_forced_multistep_owned_matches(npes, core2_forced_seq):
    """npes=2: the forced multi-step sharded run matches single-device on OWNED nodes — the
    per-step forcing scan + the multi-step halo exchanges are correct across real shards."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} fake-devices, have {NDEV}")
    from fesom_jax.tests.test_step_sharded import _owned_match
    fx = core2_forced_seq
    part = partit.read_partition(CORE2_DIST, npes)
    dense = _dense_forced(fx)
    sharded = _sharded_forced(fx, part, npes)
    _owned_match(dense, sharded, fx["mesh"], part, npes, tag="forced-kpp", fct_atol=1e-1)
