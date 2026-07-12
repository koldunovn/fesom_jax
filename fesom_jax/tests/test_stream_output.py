"""A2 gate: streaming time-mean / variance output (:mod:`fesom_jax.zarr_output`).

The streaming accumulator (:class:`~fesom_jax.zarr_output.OnlineStats`, Welford single-pass)
must reproduce an OFFLINE numpy mean/variance over the same sequence to ≤1e-12 — so a run can
emit a mean state AND an EKE map without storing every step. Plus the cadence predicate
(:func:`~fesom_jax.zarr_output.snapshot_due`), the EKE helper, field subsetting, and the
masked/dry-lane finiteness rule. Emits ``STREAM_OUTPUT_OK``.

Pure-CPU + fast; the one sharded-write check gates on the ``pi`` mesh.

  PY -m pytest fesom_jax/tests/test_stream_output.py -x
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import zarr_output as zo

ROOT = Path(__file__).resolve().parents[2]
# the pi mesh SHIPS inside the package, so this gate runs anywhere (incl. CI) —
# it used to point at the repo-root data/ symlink, which only exists on Levante.
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR as PI_MESH
avail = pytest.mark.skipif(not PI_MESH.is_dir(), reason="pi mesh missing")


def _accumulate(seq):
    """Fold a list of ``{name: array}`` samples through OnlineStats."""
    stats = zo.OnlineStats.init(seq[0])
    for s in seq:
        stats = stats.update(s)
    return stats


# --------------------------------------------------------------------------
# 1. Online mean/variance == offline numpy reference (the ≤1e-12 core gate)
# --------------------------------------------------------------------------
def test_online_matches_offline_numpy():
    rng = np.random.default_rng(0)
    n_t = 37
    # two fields of different shape/scale: a 2-D scalar field and a [*, nl, 2] velocity field
    T_seq = [10.0 + rng.standard_normal((25, 6)) for _ in range(n_t)]
    uv_seq = [rng.standard_normal((17, 6, 2)) for _ in range(n_t)]
    seq = [{"T": jnp.asarray(t), "uv": jnp.asarray(u)} for t, u in zip(T_seq, uv_seq)]

    stats = _accumulate(seq)
    assert stats.nobs() == n_t

    T_stack = np.stack(T_seq)            # [n_t, 25, 6]
    uv_stack = np.stack(uv_seq)
    np.testing.assert_allclose(np.asarray(stats.mean_dict()["T"]),
                               T_stack.mean(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(stats.mean_dict()["uv"]),
                               uv_stack.mean(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(stats.variance()["T"]),
                               T_stack.var(0), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(np.asarray(stats.variance()["uv"]),
                               uv_stack.var(0), rtol=1e-11, atol=1e-12)
    # ddof=1 (unbiased) matches numpy's ddof=1
    np.testing.assert_allclose(np.asarray(stats.variance(ddof=1)["T"]),
                               T_stack.var(0, ddof=1), rtol=1e-11, atol=1e-12)
    # std == sqrt(var)
    np.testing.assert_allclose(np.asarray(stats.std()["T"]),
                               T_stack.std(0), rtol=1e-11, atol=1e-12)


# --------------------------------------------------------------------------
# 2. EKE from the velocity variance == ½⟨u'² + v'²⟩
# --------------------------------------------------------------------------
def test_eke_from_stats():
    rng = np.random.default_rng(1)
    n_t = 23
    uv_seq = [rng.standard_normal((9, 4, 2)) for _ in range(n_t)]
    seq = [{"uv": jnp.asarray(u)} for u in uv_seq]
    stats = _accumulate(seq)

    uv_stack = np.stack(uv_seq)
    var = uv_stack.var(0)                                  # [9, 4, 2]
    eke_ref = 0.5 * (var[..., 0] + var[..., 1])
    eke = np.asarray(zo.eke_from_stats(stats, "uv"))
    assert eke.shape == (9, 4)
    np.testing.assert_allclose(eke, eke_ref, rtol=1e-11, atol=1e-12)


# --------------------------------------------------------------------------
# 3. Masked / dry lanes stay finite (constant-0 lane ⇒ mean 0, var 0; no 0/0)
# --------------------------------------------------------------------------
def test_masked_lanes_finite():
    rng = np.random.default_rng(2)
    n_t = 11
    seq = []
    for _ in range(n_t):
        a = rng.standard_normal((8, 5))
        a[:, 3:] = 0.0                                     # dry lanes: constant 0 every step
        seq.append({"T": jnp.asarray(a)})
    stats = _accumulate(seq)
    m = np.asarray(stats.mean_dict()["T"])
    v = np.asarray(stats.variance()["T"])
    assert np.all(np.isfinite(m)) and np.all(np.isfinite(v))
    # the dry lanes are exactly 0 in both mean and variance
    assert np.array_equal(m[:, 3:], np.zeros((8, 2)))
    assert np.array_equal(v[:, 3:], np.zeros((8, 2)))


# --------------------------------------------------------------------------
# 4. Field subsetting — only tracked keys accumulate; extra update keys ignored
# --------------------------------------------------------------------------
def test_field_subsetting():
    rng = np.random.default_rng(3)
    n_t = 9
    T_seq = [rng.standard_normal((6, 3)) for _ in range(n_t)]
    # init tracks ONLY "T"; updates also carry an untracked "S" that must be ignored
    stats = zo.OnlineStats.init({"T": jnp.asarray(T_seq[0])})
    for t in T_seq:
        stats = stats.update({"T": jnp.asarray(t), "S": jnp.asarray(t * 1e6 + 5.0)})
    assert set(stats.mean_dict()) == {"T"}                 # S never tracked
    np.testing.assert_allclose(np.asarray(stats.mean_dict()["T"]),
                               np.stack(T_seq).mean(0), rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# 5. Snapshot cadence predicate
# --------------------------------------------------------------------------
def test_snapshot_due_cadence():
    due = [s for s in range(0, 21) if zo.snapshot_due(s, 5)]
    assert due == [0, 5, 10, 15, 20]
    # start offset: first snapshot at `start`, then every `every`
    due2 = [s for s in range(0, 21) if zo.snapshot_due(s, 4, start=3)]
    assert due2 == [3, 7, 11, 15, 19]
    # every<=0 disables
    assert not any(zo.snapshot_due(s, 0) for s in range(10))
    assert not any(zo.snapshot_due(s, -1) for s in range(10))
    # before start: nothing
    assert not zo.snapshot_due(2, 4, start=3)


# --------------------------------------------------------------------------
# 6. write_snapshot — None when not due, writes a step-keyed store when due
# --------------------------------------------------------------------------
def test_write_snapshot_gating_returns_none():
    # not due ⇒ returns None WITHOUT touching state/sm/part (passed None on purpose)
    assert zo.write_snapshot("/tmp/never", None, None, None,
                             step=3, every=5) is None


@avail
def test_write_snapshot_writes_when_due(tmp_path):
    import dataclasses
    from fesom_jax import halo, partit, shard_mesh
    from fesom_jax import integrate_sharded as ish
    from fesom_jax.mesh import load_mesh
    from fesom_jax.state import State

    mesh = load_mesh(PI_MESH)
    g0 = State.zeros(mesh)
    g0 = dataclasses.replace(g0, T=g0.T + 3.0)             # a recognizable constant
    part = partit.synth_block_partition(mesh.nod2D, mesh.elem2D, mesh.edge2D, 1)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    sp = shard_mesh.partition_state(g0, part)
    fs, spec = ish.folded_state(sp)
    jmesh = halo.device_mesh("p", devices=jax.devices()[:1])
    placed = ish._to_global_sharded(fs, spec, jmesh)

    # not due at step 3 (every=10) ⇒ None
    assert zo.write_snapshot(tmp_path, placed, sm, part, step=3, every=10,
                             fields=("T", "S")) is None
    # due at step 20 ⇒ writes snap_00000020 with exactly the requested fields
    d = zo.write_snapshot(tmp_path, placed, sm, part, step=20, every=10,
                          fields=("T", "S"))
    assert d is not None and d.name == "snap_00000020" and d.is_dir()
    import zarr
    keys = set(zarr.open_group(str(d), mode="r").array_keys())
    assert "T" in keys and "S" in keys and "uv" not in keys
    np.testing.assert_allclose(zo.reconstruct_global(d, "T"),
                               np.asarray(g0.T), rtol=0, atol=0)


# --------------------------------------------------------------------------
# 7. Aggregate gate
# --------------------------------------------------------------------------
def test_stream_output_ok():
    rng = np.random.default_rng(7)
    seq = [{"uv": jnp.asarray(rng.standard_normal((5, 3, 2)))} for _ in range(13)]
    stats = _accumulate(seq)
    assert np.all(np.isfinite(np.asarray(zo.eke_from_stats(stats))))
    print("STREAM_OUTPUT_OK")
