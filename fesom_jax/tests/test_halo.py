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
# exchange_pair — the fused two-field exchange (the EVP/mEVP u,v refresh)
# --------------------------------------------------------------------------
@avail
def test_exchange_pair_matches_two_singles():
    """``halo.exchange_pair`` (two fields stacked on a trailing axis, ONE exchange)
    must reproduce two single-field exchanges bit-for-bit — it exists purely to
    halve the EVP/mEVP subcycle's collective count. ``exch=None`` (the dense path)
    must return the pair untouched (same objects, graph byte-identity)."""
    _need(2)
    part = partit.read_partition(CORE2_DIST, 2)
    sm = shard_mesh.build_sharded_mesh(load_mesh(CORE2_MESH), part)
    Lmax = sm.Lmax["nod"]
    src_dev, src_lane = sm.exchange["nod"]
    owned = np.asarray(sm.owned_mask["nod"])
    jmesh = halo.device_mesh(devices=jax.devices()[:2])
    rng = np.random.default_rng(3)
    u = np.where(owned, rng.standard_normal((part.npes, Lmax)), -1.0)
    v = np.where(owned, rng.standard_normal((part.npes, Lmax)), -2.0)

    # oracle: two single-field exchanges
    u1 = np.asarray(halo.run_halo_exchange(u, src_dev, src_lane, jmesh))
    v1 = np.asarray(halo.run_halo_exchange(v, src_dev, src_lane, jmesh))

    # fused: exchange_pair around the same primitive, inside shard_map
    spec = halo.PartitionSpec("p")
    fold = lambda a: halo._fold(np.asarray(a))[0]     # noqa: E731
    fn = jax.shard_map(
        lambda uu, vv, sd, sl: halo.exchange_pair(
            lambda f, kind: halo.halo_exchange(f, sd, sl), uu, vv, "nod"),
        mesh=jmesh, in_specs=(spec,) * 4, out_specs=(spec, spec))
    u2, v2 = fn(fold(u), fold(v),
                fold(src_dev).astype(jnp.int32), fold(src_lane).astype(jnp.int32))
    np.testing.assert_array_equal(np.asarray(halo._unfold(u2, 2, Lmax)), u1)
    np.testing.assert_array_equal(np.asarray(halo._unfold(v2, 2, Lmax)), v1)

    # dense path: exch=None returns the pair untouched
    a, b = halo.exchange_pair(None, u, v, "nod")
    assert a is u and b is v


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


def test_ragged_grad_path_refused():
    """GUARD (2026-07-03 review): while the ``ragged_all_to_all`` transpose is broken (the
    xfail above), the gradient entry point must REFUSE ``use_ragged=True`` loudly rather
    than return silently wrong gradients. The raise is at function entry, before any
    argument is touched — hermetic (no devices, no data). The sanctioned point-to-point
    gradient path is ``use_padded=True`` (Phase 8c; grad-gated below)."""
    from fesom_jax import integrate_sharded as ish
    with pytest.raises(ValueError, match="ragged_all_to_all"):
        ish.run_steps_sharded(None, None, None, None, 1, dt=1.0, npes=2,
                              use_ragged=True, return_grad_fn=True)


# --------------------------------------------------------------------------
# Phase 8c — the slot-padded dense all_to_all == all_gather (fwd + TRANSPOSE),
# on EVERY backend (the CPU-capable, AD-correct ragged substitute; merged from
# experiments/padded_halo_a2a after the Levante dist_2..32 gates, 2026-07-13)
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2, 4])
@pytest.mark.parametrize("kind", ["nod", "elem", "edge"])
def test_padded_forward_matches_allgather(npes, kind):
    """The padded dense-a2a exchange == the all_gather exchange on every VALID lane,
    BYTE-IDENTICALLY — same owner values, one tiled ``lax.all_to_all`` of slot-padded
    chunks instead of the broadcast. Unlike the ragged primitive this runs on CPU."""
    _need(npes)
    sm, Lmax, jmesh, valid, field, _ = _ragged_setup(npes, kind)
    src_dev, src_lane = sm.exchange[kind]
    rmap = sm.exchange_ragged[kind]
    ref = np.asarray(halo.run_halo_exchange(field, src_dev, src_lane, jmesh))
    got = np.asarray(halo.run_halo_exchange_padded(field, rmap, jmesh))
    vm = valid[:, :, None]
    assert np.array_equal(np.where(vm, got, 0.0), np.where(vm, ref, 0.0)), \
        f"padded != all_gather forward on valid {kind} lanes (npes={npes})"


@avail
@pytest.mark.parametrize("npes", [2, 4])
@pytest.mark.parametrize("kind", ["nod", "elem", "edge"])
def test_padded_grad_matches_allgather(npes, kind):
    """THE Phase 8c point: the padded exchange's gradient matches the all_gather
    oracle's (the transpose is another ``all_to_all`` — a trusted JAX rule — plus the
    ``pad_valid`` where that kills the duplicated pad-gather cotangents). This is the
    real test the ragged primitive xfails; the weights are zeroed off the valid lanes
    (pad-lane cotangents legitimately differ between the transports)."""
    _need(npes)
    sm, Lmax, jmesh, valid, field, rng = _ragged_setup(npes, kind)
    src_dev, src_lane = sm.exchange[kind]
    rmap = sm.exchange_ragged[kind]
    w = jnp.asarray(rng.standard_normal((npes, Lmax, sm.nl)) * valid[:, :, None])
    g_ref = np.asarray(jax.grad(lambda f: jnp.sum(
        w * halo.run_halo_exchange(f, src_dev, src_lane, jmesh)))(field))
    g_pad = np.asarray(jax.grad(lambda f: jnp.sum(
        w * halo.run_halo_exchange_padded(f, rmap, jmesh)))(field))
    vm = valid[:, :, None]
    dif = np.abs(np.where(vm, g_pad - g_ref, 0.0)).max()
    scale = max(np.abs(g_ref).max(), 1e-300)
    assert dif / scale < 1e-12, \
        f"padded grad != all_gather grad ({kind}, npes={npes}); rel max|Δ|={dif/scale:.3e}"


def test_halo_modes_mutually_exclusive():
    """``use_ragged`` and ``use_padded`` name two transports for the SAME exchange —
    both at once is a caller bug; refused at entry (hermetic, like the guard above)."""
    from fesom_jax import integrate_sharded as ish
    with pytest.raises(ValueError, match="mutually exclusive"):
        ish.run_steps_sharded(None, None, None, None, 1, dt=1.0, npes=2,
                              use_ragged=True, use_padded=True)


@pytest.mark.parametrize("pair", [("use_ragged", "use_coloured"),
                                  ("use_padded", "use_coloured")])
def test_coloured_excludes_the_other_transports(pair):
    """Phase 8d: coloured is a THIRD substitute for the same exchange — refuse any pairing."""
    from fesom_jax import integrate_sharded as ish
    with pytest.raises(ValueError, match="mutually exclusive"):
        ish.run_steps_sharded(None, None, None, None, 1, dt=1.0, npes=2,
                              **{pair[0]: True, pair[1]: True})


@avail
@pytest.mark.parametrize("npes", [2, 4, 8])
@pytest.mark.parametrize("kind", ["nod", "elem", "edge"])
def test_coloured_classes_are_partial_permutations(npes, kind):
    """The property ``lax.ppermute`` REQUIRES, checked on the real partitions: within one
    colour class every device sends to at most one peer and receives from at most one peer
    (a partial permutation). If the colouring ever violated this the perm would be illegal
    — and the exchange would silently drop or overwrite a chunk."""
    _need(npes)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(load_mesh(CORE2_MESH), part)
    c = sm.exchange_coloured[kind]
    seen = set()
    for perm in c.perms:
        srcs = [d for d, _ in perm]
        dsts = [e for _, e in perm]
        assert len(srcs) == len(set(srcs)), f"{kind}/{npes}: a device sends twice in one round"
        assert len(dsts) == len(set(dsts)), f"{kind}/{npes}: a device receives twice in one round"
        assert all(d != e for d, e in perm), "self-edge in the neighbour graph"
        seen |= set(perm)
    # every real neighbour edge is carried by exactly one round (none dropped, none doubled)
    edges = {(d, e) for d in range(npes) for e in range(npes)
             if sm.exchange_ragged[kind].send_sizes[d, e] > 0}
    assert seen == edges, f"{kind}/{npes}: colouring lost or duplicated edges"
    assert len(c.slots) == len(c.perms) and c.total == sum(c.slots)


@avail
@pytest.mark.parametrize("npes", [2, 4, 8])
@pytest.mark.parametrize("kind", ["nod", "elem", "edge"])
def test_coloured_forward_matches_allgather(npes, kind):
    """The coloured-ppermute exchange == the all_gather exchange on every VALID lane,
    BYTE-IDENTICALLY — same owner values, K point-to-point ppermute rounds instead of the
    broadcast. Any backend (ppermute, like all_to_all, exists on XLA:CPU)."""
    _need(npes)
    sm, Lmax, jmesh, valid, field, _ = _ragged_setup(npes, kind)
    src_dev, src_lane = sm.exchange[kind]
    cmap, hmask = sm.exchange_coloured[kind], sm.exchange_ragged[kind].halo_mask
    ref = np.asarray(halo.run_halo_exchange(field, src_dev, src_lane, jmesh))
    got = np.asarray(halo.run_halo_exchange_coloured(field, cmap, hmask, jmesh))
    vm = valid[:, :, None]
    assert np.array_equal(np.where(vm, got, 0.0), np.where(vm, ref, 0.0)), \
        f"coloured != all_gather forward on valid {kind} lanes (npes={npes})"


@avail
@pytest.mark.parametrize("npes", [2, 4])
@pytest.mark.parametrize("kind", ["nod", "elem", "edge"])
def test_coloured_grad_matches_allgather(npes, kind):
    """THE Phase 8d point: the coloured exchange's gradient matches the all_gather oracle's.
    ``ppermute`` transposes to the inverse ``ppermute`` (a trusted JAX rule), and the
    ``send_valid`` where kills the pad lanes' duplicated-gather cotangents — so the AD
    correctness of the padded transport survives dropping its padding. This is the test the
    ragged primitive xfails."""
    _need(npes)
    sm, Lmax, jmesh, valid, field, rng = _ragged_setup(npes, kind)
    src_dev, src_lane = sm.exchange[kind]
    cmap, hmask = sm.exchange_coloured[kind], sm.exchange_ragged[kind].halo_mask
    w = jnp.asarray(rng.standard_normal((npes, Lmax, sm.nl)) * valid[:, :, None])
    g_ref = np.asarray(jax.grad(lambda f: jnp.sum(
        w * halo.run_halo_exchange(f, src_dev, src_lane, jmesh)))(field))
    g_col = np.asarray(jax.grad(lambda f: jnp.sum(
        w * halo.run_halo_exchange_coloured(f, cmap, hmask, jmesh)))(field))
    vm = valid[:, :, None]
    dif = np.abs(np.where(vm, g_col - g_ref, 0.0)).max()
    scale = max(np.abs(g_ref).max(), 1e-300)
    assert dif / scale < 1e-12, \
        f"coloured grad != all_gather grad ({kind}, npes={npes}); rel max|Δ|={dif/scale:.3e}"


@avail
@pytest.mark.parametrize("npes", [4])
def test_coloured_ships_less_than_padded(npes):
    """The whole point of Phase 8d, asserted as an invariant rather than a benchmark: the
    coloured transport's wire volume (Σ per-round slots) is strictly below the padded one's
    (P × global-max-slot), and the gap widens with P (see scripts/bench/probe_pad_factor.py)."""
    _need(npes)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(load_mesh(CORE2_MESH), part)
    for kind in ("nod", "elem", "edge"):
        col = sm.exchange_coloured[kind].total
        pad = npes * sm.exchange_ragged[kind].pad_slot
        assert col < pad, f"{kind}: coloured {col} lanes !< padded {pad}"
