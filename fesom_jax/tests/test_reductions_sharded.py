"""S.5 gate: distributed reductions (:mod:`fesom_jax.reductions`).

A 2/4-device owned-sum + ``psum`` equals the single-device global sum to ~1e-13,
the owned mask correctly excludes halo (a corrupted halo value does not change the
result), and the ``axis_name=None`` helper path is the exact single-device
``jnp.sum`` (so ``_area_mean`` stays byte-identical at ``npes==1``).

The multi-device parts need CPU fake-devices and SKIP otherwise.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec

from fesom_jax import halo, partit, reductions, shard_mesh
from fesom_jax.mesh import load_mesh
from fesom_jax.sss_runoff import _area_mean, sss_runoff_fluxes

CORE2_MESH = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NDEV = len(jax.devices())

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_2").is_dir(),
    reason="CORE2 dense mesh or dist partitions missing",
)


def _device_total(vals_PL, owned_PL, npes, fn):
    """Run ``fn(v_local, m_local, 'p')`` under shard_map and return the replicated
    scalar. ``fn`` is global_sum-like."""
    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    P, Lmax = vals_PL.shape[0], vals_PL.shape[1]
    v = jnp.asarray(vals_PL).reshape(P * Lmax)
    m = jnp.asarray(owned_PL).reshape(P * Lmax)
    spec = PartitionSpec("p")
    smfn = jax.shard_map(lambda vv, mm: fn(vv, mm, "p"),
                         mesh=jmesh, in_specs=(spec, spec), out_specs=PartitionSpec())
    return float(smfn(v, m))


# --------------------------------------------------------------------------
# Helper byte-identity (no devices)
# --------------------------------------------------------------------------
def test_global_sum_single_device_matches_plain_sum():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2000)
    m = np.ones(2000, dtype=bool)
    # axis_name=None + all-True mask == plain jnp.sum (the npes==1 path)
    assert float(reductions.global_sum(x, m, None)) == pytest.approx(float(jnp.sum(jnp.asarray(x))),
                                                                     rel=0, abs=1e-9)
    # masked entries are excluded
    m2 = m.copy(); m2[1000:] = False
    assert float(reductions.global_sum(x, m2, None)) == pytest.approx(float(np.sum(x[:1000])), abs=1e-9)


@avail
def test_area_mean_default_byte_identical():
    """``_area_mean`` with no owned_mask is the exact single-device reduction."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(5000)
    a = rng.uniform(1.0, 2.0, 5000)
    oa = 3.3e14
    got = _area_mean(jnp.asarray(x), jnp.asarray(a), oa)
    expect = jnp.sum(jnp.asarray(x) * jnp.asarray(a)) / oa
    assert float(got) == float(expect)        # identical graph ⇒ identical value


# --------------------------------------------------------------------------
# Multi-device: owned-sum + psum == single-device global sum
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_global_sum_matches_single_device(npes):
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    Ln = sm.Lmax["nod"]
    rng = np.random.default_rng(3)
    x = rng.standard_normal(mesh.nod2D)

    # single-device reference
    ref = float(np.sum(x))
    # partition x → [P, Lmax_nod] (gather owned+halo by myList, pad)
    xPL = np.zeros((npes, Ln))
    for d in range(npes):
        ml = part.myList_nod2D[d]
        xPL[d, : ml.size] = x[ml]
    total = _device_total(xPL, sm.owned_mask["nod"], npes, reductions.global_sum)
    assert total == pytest.approx(ref, rel=1e-12, abs=1e-9)


@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_owned_mask_excludes_halo(npes):
    """A corrupted HALO value does not change the reduction (masked out)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    Ln = sm.Lmax["nod"]
    rng = np.random.default_rng(4)
    x = rng.standard_normal(mesh.nod2D)
    xPL = np.zeros((npes, Ln))
    for d in range(npes):
        ml = part.myList_nod2D[d]
        xPL[d, : ml.size] = x[ml]
    clean = _device_total(xPL, sm.owned_mask["nod"], npes, reductions.global_sum)
    # corrupt a halo lane on device 0 (first lane past myDim)
    h = int(part.myDim_nod2D[0])
    xPL[0, h] += 1e6
    assert (sm.valid_mask["nod"][0, h] and not sm.owned_mask["nod"][0, h])
    corrupted = _device_total(xPL, sm.owned_mask["nod"], npes, reductions.global_sum)
    assert corrupted == pytest.approx(clean, rel=0, abs=1e-9)


@avail
@pytest.mark.parametrize("npes", [2])
def test_sss_runoff_fluxes_sharded_matches_single(npes):
    """S.7 part 3 reduction routing: ``sss_runoff_fluxes`` threaded with
    ``owned_mask``/``axis_name`` (owned-node sum + ``psum`` inside ``shard_map``) matches the
    single-device fluxes on OWNED nodes. The ``_area_mean`` global-mean subtraction
    (virtual-salt / relax-salt / water-flux balance) is the same total whether summed on one
    device or owned-summed + ``psum``'d across devices ⇒ each owned node's balanced flux
    matches. (``owned_mask=None`` is already proven byte-identical to ``v1.0``.)"""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    Ln = sm.Lmax["nod"]
    N = mesh.nod2D
    rng = np.random.default_rng(7)
    S_top = rng.uniform(30.0, 36.0, N)
    water_flux = rng.standard_normal(N) * 1e-6
    Ssurf = rng.uniform(30.0, 36.0, N)
    runoff = np.abs(rng.standard_normal(N)) * 1e-7
    areasvol = np.asarray(mesh.areasvol)[:, 0]
    ocean_area = float(mesh.ocean_area)

    # single-device reference (owned_mask=None ⇒ the v1.0 graph)
    ref = sss_runoff_fluxes(S_top, water_flux, Ssurf, runoff, areasvol, ocean_area)

    # partition the node fields to [P, Ln] (gather owned+halo by myList, pad with 0)
    def _part(x):
        out = np.zeros((npes, Ln))
        for d in range(npes):
            ml = part.myList_nod2D[d]
            out[d, : ml.size] = x[ml]
        return jnp.asarray(out).reshape(npes * Ln)

    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    spec = PartitionSpec("p")

    def body(S_t, wf, Ss, ru, av, m):
        out = sss_runoff_fluxes(S_t, wf, Ss, ru, av, ocean_area,
                                owned_mask=m, axis_name="p")
        return jnp.stack([out.virtual_salt, out.relax_salt, out.water_flux], axis=-1)

    fn = jax.shard_map(body, mesh=jmesh, in_specs=(spec,) * 6, out_specs=spec)
    res = np.asarray(fn(_part(S_top), _part(water_flux), _part(Ssurf), _part(runoff),
                        _part(areasvol), jnp.asarray(sm.owned_mask["nod"]).reshape(npes * Ln)))
    res = res.reshape(npes, Ln, 3)

    for j, name in enumerate(("virtual_salt", "relax_salt", "water_flux")):
        a = np.asarray(getattr(ref, name))
        worst = 0.0
        for d in range(npes):
            md = int(part.myDim_nod2D[d])
            worst = max(worst, float(np.max(np.abs(res[d, :md, j] - a[part.myList_nod2D[d][:md]]))))
        assert worst < 1e-12, f"{name}: owned max|Δ|={worst:.3e}"


@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_global_dot_matches_single_device(npes):
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    Ln = sm.Lmax["nod"]
    rng = np.random.default_rng(5)
    a = rng.standard_normal(mesh.nod2D)
    b = rng.standard_normal(mesh.nod2D)
    ref = float(np.dot(a, b))
    aPL = np.zeros((npes, Ln)); bPL = np.zeros((npes, Ln))
    for d in range(npes):
        ml = part.myList_nod2D[d]
        aPL[d, : ml.size] = a[ml]; bPL[d, : ml.size] = b[ml]

    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    spec = PartitionSpec("p")
    av = jnp.asarray(aPL).reshape(npes * Ln)
    bv = jnp.asarray(bPL).reshape(npes * Ln)
    mv = jnp.asarray(sm.owned_mask["nod"]).reshape(npes * Ln)
    fn = jax.shard_map(lambda x, y, m: reductions.global_dot(x, y, m, "p"),
                       mesh=jmesh, in_specs=(spec, spec, spec), out_specs=PartitionSpec())
    assert float(fn(av, bv, mv)) == pytest.approx(ref, rel=1e-12, abs=1e-9)


@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_global_dot_pair_matches_two_dots(npes):
    """``global_dot_pair`` (the CG's fused ``(r·z, r·r)`` — one length-2 ``psum``
    instead of two scalar ones) must reproduce two ``global_dot`` calls BIT-for-bit:
    the owned-lane locals are the same expressions, and the fused ``psum`` reduces
    each element over the same device order."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    Ln = sm.Lmax["nod"]
    rng = np.random.default_rng(6)
    rPL = rng.standard_normal((npes, Ln))
    zPL = rng.standard_normal((npes, Ln))

    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    spec = PartitionSpec("p")
    rv = jnp.asarray(rPL).reshape(npes * Ln)
    zv = jnp.asarray(zPL).reshape(npes * Ln)
    mv = jnp.asarray(sm.owned_mask["nod"]).reshape(npes * Ln)
    two = jax.shard_map(
        lambda r, z, m: (reductions.global_dot(r, z, m, "p"),
                         reductions.global_dot(r, r, m, "p")),
        mesh=jmesh, in_specs=(spec,) * 3, out_specs=(PartitionSpec(), PartitionSpec()))
    one = jax.shard_map(
        lambda r, z, m: reductions.global_dot_pair(r, z, r, r, m, "p"),
        mesh=jmesh, in_specs=(spec,) * 3, out_specs=(PartitionSpec(), PartitionSpec()))
    rz2, rr2 = two(rv, zv, mv)
    rz1, rr1 = one(rv, zv, mv)
    assert float(rz1) == float(rz2) and float(rr1) == float(rr2)

    # axis_name=None (dense / single-device): plain pair of masked sums
    m1 = np.ones(Ln, dtype=bool)
    la, lb = reductions.global_dot_pair(rPL[0], zPL[0], rPL[0], rPL[0], m1, None)
    assert float(la) == float(reductions.global_dot(rPL[0], zPL[0], m1, None))
    assert float(lb) == float(reductions.global_dot(rPL[0], rPL[0], m1, None))
