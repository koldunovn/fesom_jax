"""S.3 gate: the broadcast halo-exchange primitive (:mod:`fesom_jax.halo`).

Ports the C ``fesom_halo_identity_test`` (``fesom_halo.c:212-284``): set each
owned lane to its global id, exchange, and assert every **halo** lane carries its
owner's gid (+ corruption recovery). Covers all three kinds (nod2D / elem2D /
elem2D_full) and a multi-level field, and grad-checks that the exchange is linear
with the correct (reverse-exchange) transpose.

These need ≥2 CPU fake-devices, so they **SKIP under the 1-device suite** and run
via the dedicated invocation:

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        <env-py> -m pytest fesom_jax/tests/test_halo.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import halo, partit, shard_mesh
from fesom_jax.mesh import load_mesh

CORE2_MESH = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NDEV = len(jax.devices())

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_2").is_dir(),
    reason="CORE2 dense mesh or dist partitions missing",
)


def _need(npes: int):
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV} "
                    f"(run with XLA_FLAGS=--xla_force_host_platform_device_count={npes})")


def _gid_field(part: partit.Partition, kind: str, Lmax: int) -> np.ndarray:
    """[P, Lmax] global id per local lane (-1 on pad)."""
    mylist = {"nod": part.myList_nod2D, "elem": part.myList_elem2D,
              "edge": part.myList_edge2D}[kind]
    g = np.full((part.npes, Lmax), -1, dtype=np.int64)
    for d in range(part.npes):
        g[d, : mylist[d].size] = mylist[d]
    return g


# --------------------------------------------------------------------------
# Identity gate (port of fesom_halo_identity_test) — all three kinds
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2, 4])
@pytest.mark.parametrize("kind", ["nod", "elem", "edge"])
def test_identity(npes, kind):
    _need(npes)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(load_mesh(CORE2_MESH), part)
    Lmax = sm.Lmax[kind]
    src_dev, src_lane = sm.exchange[kind]
    gid = _gid_field(part, kind, Lmax)
    owned = sm.owned_mask[kind]
    valid = sm.valid_mask[kind]

    # owned lanes carry their gid; halo + pad start at the sentinel -1
    f = np.where(owned, gid, -1.0).astype(np.float64)
    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    f2 = np.asarray(halo.run_halo_exchange(f, src_dev, src_lane, jmesh))

    # every VALID lane now carries its gid (interior identity + halo from owner)
    assert np.array_equal(f2[valid], gid[valid].astype(np.float64))
    # specifically the halo lanes were refreshed off the sentinel
    halo_lanes = valid & ~owned
    assert np.array_equal(f2[halo_lanes], gid[halo_lanes].astype(np.float64))
    assert np.isfinite(f2).all()


@avail
def test_corruption_recovery():
    """Clobber a halo lane, re-exchange, and confirm it is overwritten back."""
    _need(2)
    part = partit.read_partition(CORE2_DIST, 2)
    sm = shard_mesh.build_sharded_mesh(load_mesh(CORE2_MESH), part)
    Lmax = sm.Lmax["nod"]
    src_dev, src_lane = sm.exchange["nod"]
    gid = _gid_field(part, "nod", Lmax)
    owned, valid = sm.owned_mask["nod"], sm.valid_mask["nod"]
    jmesh = halo.device_mesh(devices=jax.devices()[:2])

    f = np.where(owned, gid, -1.0).astype(np.float64)
    f2 = np.array(halo.run_halo_exchange(f, src_dev, src_lane, jmesh))   # writable copy
    # corrupt the first halo lane on device 0
    d, h = 0, int(sm.counts["myDim_nod"][0])      # first halo local index
    assert valid[d, h] and not owned[d, h]
    f2[d, h] = -99.0
    f3 = np.asarray(halo.run_halo_exchange(f2, src_dev, src_lane, jmesh))
    assert f3[d, h] == gid[d, h]                   # restored
    assert np.array_equal(f3[valid], gid[valid].astype(np.float64))


# --------------------------------------------------------------------------
# Multi-level field
# --------------------------------------------------------------------------
@avail
def test_multilevel_field():
    _need(2)
    part = partit.read_partition(CORE2_DIST, 2)
    sm = shard_mesh.build_sharded_mesh(load_mesh(CORE2_MESH), part)
    Lmax, nl = sm.Lmax["nod"], 5
    src_dev, src_lane = sm.exchange["nod"]
    gid = _gid_field(part, "nod", Lmax)
    owned, valid = sm.owned_mask["nod"], sm.valid_mask["nod"]
    # per-level pattern: gid*10 + k on owned, 0 on halo
    base = np.where(owned, gid, 0).astype(np.float64)
    f = base[:, :, None] * 10.0 + np.arange(nl)[None, None, :]
    f = np.where(valid[:, :, None], f, 0.0)
    jmesh = halo.device_mesh(devices=jax.devices()[:2])
    f2 = np.asarray(halo.run_halo_exchange(f, src_dev, src_lane, jmesh))
    expect = gid[:, :, None].astype(np.float64) * 10.0 + np.arange(nl)[None, None, :]
    assert np.allclose(f2[valid], expect[valid])   # each level carried correctly


# --------------------------------------------------------------------------
# AD: halo_exchange is linear; vjp = reverse exchange
# --------------------------------------------------------------------------
@avail
def test_linear_and_grad():
    _need(2)
    part = partit.read_partition(CORE2_DIST, 2)
    sm = shard_mesh.build_sharded_mesh(load_mesh(CORE2_MESH), part)
    Lmax = sm.Lmax["nod"]
    src_dev, src_lane = sm.exchange["nod"]
    jmesh = halo.device_mesh(devices=jax.devices()[:2])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((part.npes, Lmax))

    def ex(f):
        return halo.run_halo_exchange(f, src_dev, src_lane, jmesh)

    # linearity: exchange(a·x) == a·exchange(x)
    a = 3.7
    assert np.allclose(np.asarray(ex(a * x)), a * np.asarray(ex(x)))

    # grad of a scalar loss vs FD on a few interior + halo entries
    w = rng.standard_normal((part.npes, Lmax))

    def loss(f):
        return jnp.sum(jnp.asarray(w) * ex(f))

    g = np.asarray(jax.grad(loss)(x))
    eps = 1e-5
    for (d, i) in [(0, 0), (1, 5),
                   (0, int(sm.counts["myDim_nod"][0])),       # a halo lane
                   (1, int(sm.counts["myDim_nod"][1]) + 1)]:
        xp = x.copy(); xp[d, i] += eps
        xm = x.copy(); xm[d, i] -= eps
        fd = (float(loss(xp)) - float(loss(xm))) / (2 * eps)
        assert abs(g[d, i] - fd) < 1e-4, f"grad[{d},{i}] {g[d,i]} vs FD {fd}"


# --------------------------------------------------------------------------
# Phase 8b B.0b/c — the ragged_all_to_all primitive == all_gather (fwd + transpose)
# --------------------------------------------------------------------------
def _ragged_setup(npes, kind):
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(load_mesh(CORE2_MESH), part)
    Lmax = sm.Lmax[kind]
    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    valid = np.asarray(sm.valid_mask[kind])
    rng = np.random.default_rng(2)
    field = jnp.asarray(rng.standard_normal((npes, Lmax, sm.nl)))
    return sm, Lmax, jmesh, valid, field, rng


@avail
@pytest.mark.parametrize("npes", [2, 4])
@pytest.mark.parametrize("kind", ["nod", "elem", "edge"])
def test_ragged_primitive_forward_matches_allgather(npes, kind):
    """B.0c: the halo-only ``ragged_all_to_all`` exchange == the ``all_gather`` exchange
    on every VALID lane, BYTE-IDENTICALLY (same owner values, moved point-to-point
    instead of broadcast). The FORWARD is what a scaling run needs. ⚠️ GPU-only —
    ``ragged_all_to_all`` is UNIMPLEMENTED on XLA:CPU, so this SKIPs on CPU."""
    if jax.devices()[0].platform == "cpu":
        pytest.skip("lax.ragged_all_to_all is unimplemented on XLA:CPU; needs GPU (NCCL)")
    _need(npes)
    sm, Lmax, jmesh, valid, field, _ = _ragged_setup(npes, kind)
    src_dev, src_lane = sm.exchange[kind]
    rmap = sm.exchange_ragged[kind]
    ref = np.asarray(halo.run_halo_exchange(field, src_dev, src_lane, jmesh))
    got = np.asarray(halo.run_halo_exchange_ragged(field, rmap, jmesh))
    vm = valid[:, :, None]
    assert np.array_equal(np.where(vm, got, 0.0), np.where(vm, ref, 0.0)), \
        f"ragged != all_gather forward on valid {kind} lanes (npes={npes})"


@avail
@pytest.mark.xfail(reason="JAX 0.10.1 lax.ragged_all_to_all autodiff transpose is broken "
                          "(grad scales with device count); fix = custom_vjp, Phase 8b B.0d",
                   strict=False)
@pytest.mark.parametrize("npes", [2])
@pytest.mark.parametrize("kind", ["nod"])
def test_ragged_primitive_grad_known_broken(npes, kind):
    """KNOWN-BROKEN (B.0c): the ragged exchange's gradient should match the all_gather
    exchange's (the transpose scatter-adds halo cotangents to owners), but JAX's
    ``ragged_all_to_all`` reverse-mode autodiff mis-routes (grad max|Δ| ~ O(npes)). This
    `xfail` documents it + flips to PASS once B.0d wraps it in a `custom_vjp`. GPU-only."""
    if jax.devices()[0].platform == "cpu":
        pytest.skip("lax.ragged_all_to_all is unimplemented on XLA:CPU; needs GPU (NCCL)")
    _need(npes)
    sm, Lmax, jmesh, valid, field, rng = _ragged_setup(npes, kind)
    src_dev, src_lane = sm.exchange[kind]
    rmap = sm.exchange_ragged[kind]
    w = jnp.asarray(rng.standard_normal((npes, Lmax, sm.nl)) * valid[:, :, None])
    g_ref = np.asarray(jax.grad(lambda f: jnp.sum(
        w * halo.run_halo_exchange(f, src_dev, src_lane, jmesh)))(field))
    g_rag = np.asarray(jax.grad(lambda f: jnp.sum(
        w * halo.run_halo_exchange_ragged(f, rmap, jmesh)))(field))
    assert np.allclose(g_rag, g_ref, atol=1e-12, rtol=0), \
        f"ragged grad != all_gather grad ({kind}, npes={npes}); " \
        f"max|Δ|={np.max(np.abs(g_rag - g_ref)):.3e}"
