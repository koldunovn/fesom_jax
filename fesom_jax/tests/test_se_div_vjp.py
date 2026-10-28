"""The split-explicit divergence's explicit transpose (``ssh_se._divergence_t``, ``FESOM_SE_DIV_VJP=1``): equal to the
default transpose of the gathers (JAX's, a scatter-add) to round-off, on the dense CORE2 tables and on every device's
local tables of the dist_4 partition; the inverse table leaves out the pad slots and has at most three entries per
element (its vertices); ``divergence`` takes the explicit rule only when enabled and the table is there.

CORE2 mesh + dist_4 (compute node; light: no step is traced).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import partit, shard_mesh, ssh_se
from fesom_jax.mesh import load_mesh

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
TOL = 1e-12                      # max |delta| / max |reference|

pytestmark = pytest.mark.skipif(not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_4").is_dir(),
                                reason="CORE2 mesh / dist_4 missing (compute node only)")


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(CORE2_MESH)


def _check(el, c, inv, n_elem, seed):
    rng = np.random.default_rng(seed)
    U = jnp.asarray(rng.standard_normal((n_elem, 2)))
    tbar = jnp.asarray(rng.standard_normal(el.shape[0]))
    _, vjp = jax.vjp(lambda u: ssh_se._divergence(u, jnp.asarray(el), jnp.asarray(c)), U)
    ref = np.asarray(vjp(tbar)[0])
    got = np.asarray(ssh_se._divergence_t(tbar, jnp.asarray(c), jnp.asarray(inv)))
    return float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))


def test_dense_tables_inverse_and_transpose(mesh):
    t = ssh_se.build_tables(mesh)
    inv = t["se_div_inv"]
    n = t["se_div_elem"].size
    assert inv.shape[0] == mesh.elem2D and inv.shape[1] <= 3
    live = inv < n
    assert live.any(axis=1).all()                 # every element is in some row (no cavities on CORE2)
    el_flat = t["se_div_elem"].reshape(-1)
    rows = np.broadcast_to(np.arange(mesh.elem2D)[:, None], inv.shape)
    np.testing.assert_array_equal(el_flat[inv[live]], rows[live])
    r = _check(t["se_div_elem"], t["se_div_c"], inv, mesh.elem2D, 0)
    print(f"\n[se_div_vjp] dense: explicit vs autodiff transpose max rel {r:.1e}")
    assert r <= TOL


def test_sharded_local_tables(mesh):
    part = partit.read_partition(CORE2_DIST, 4)
    sm = shard_mesh.with_se_tables(shard_mesh.build_sharded_mesh(mesh, part), mesh, part)
    f = sm.fields
    assert f["se_div_inv"].shape[:2] == (4, sm.Lmax["elem"])
    for d in range(4):
        r = _check(f["se_div_elem"][d], f["se_div_c"][d], f["se_div_inv"][d], sm.Lmax["elem"], d + 1)
        print(f"[se_div_vjp] device {d}: max rel {r:.1e}")
        assert r <= TOL


def test_divergence_dispatch(mesh, monkeypatch):
    m = ssh_se.with_tables(mesh)
    rng = np.random.default_rng(7)
    U = jnp.asarray(rng.standard_normal((mesh.elem2D, 2)))
    tbar = jnp.asarray(rng.standard_normal(mesh.nod2D))
    monkeypatch.setenv("FESOM_SE_DIV_VJP", "0")
    y0, vjp0 = jax.vjp(lambda u: ssh_se.divergence(u, m), U)
    monkeypatch.setenv("FESOM_SE_DIV_VJP", "1")
    y1, vjp1 = jax.vjp(lambda u: ssh_se.divergence(u, m), U)
    np.testing.assert_array_equal(np.asarray(y1), np.asarray(y0))     # the forward is the same code
    g0, g1 = np.asarray(vjp0(tbar)[0]), np.asarray(vjp1(tbar)[0])
    assert float(np.max(np.abs(g1 - g0)) / np.max(np.abs(g0))) <= TOL
    # without the table (the wide-halo lanes) the default rule is taken
    bare = dataclasses.replace(m, se_div_inv=None)
    _, vjp2 = jax.vjp(lambda u: ssh_se.divergence(u, bare), U)
    np.testing.assert_array_equal(np.asarray(vjp2(tbar)[0]), g0)
