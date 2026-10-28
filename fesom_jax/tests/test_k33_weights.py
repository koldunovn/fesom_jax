"""The Redi K33 interface weights of FESOM2 #944 (0452259a, merged 6526e4be 2026-08-21).

``gm_redi.k33_augmentation`` is the layer-centre ``s²·Ki`` interpolated LINEARLY to the interface ``zbar[k]``
between the centres ``Z[k-1]`` and ``Z[k]`` (``oce_ale_tracer.F90:1296,1329,1331,1372`` in e96a2b13). The pre-#944
weights (``legacy_weights=True``, kept for the gates whose dumps predate it) swap the two half-distances; on a uniform
grid both are 1/2, on a non-uniform one they differ. Hand-built columns on the packaged pi mesh, static and zstar.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ale, gm_redi
from fesom_jax.ale import AleConfig
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.state import State

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(DEFAULT_PI_MESH_DIR)


def _inputs(mesh, seed):
    rng = np.random.default_rng(seed)
    N, nl = int(mesh.nod2D), int(mesh.nl)
    slope = np.zeros((N, nl - 1, 3))
    slope[:, :, 2] = 1e-3 * rng.random((N, nl - 1))
    Ki = 100.0 + 900.0 * rng.random((N, nl))
    return slope, Ki


def _expected(mesh, slope, Ki, zbar, Z):
    """Linear interpolation of s²·Ki from the centres Z[k-1], Z[k] to zbar[k], on the interior interfaces."""
    N, nl = int(mesh.nod2D), int(mesh.nl)
    f = slope[:, :, 2] ** 2 * Ki[:, : nl - 1]                     # (N, nl-1) at the layer centres
    out = np.zeros((N, nl))
    ulev, nlev = np.asarray(mesh.ulevels_nod2D), np.asarray(mesh.nlevels_nod2D)
    for n in range(N):
        for k in range(max(ulev[n], 1), nlev[n] - 1):
            t = (zbar[n, k] - Z[n, k - 1]) / (Z[n, k] - Z[n, k - 1])
            out[n, k] = f[n, k - 1] + t * (f[n, k] - f[n, k - 1])
    return out


def _check(mesh, slope, Ki, zbar, Z, **geo):
    new = np.asarray(gm_redi.k33_augmentation(mesh, jnp.asarray(slope), jnp.asarray(Ki), **geo))
    old = np.asarray(gm_redi.k33_augmentation(mesh, jnp.asarray(slope), jnp.asarray(Ki), legacy_weights=True, **geo))
    ref = _expected(mesh, slope, Ki, zbar, Z)
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(new - ref)) <= 1e-13 * scale
    # the swapped weights are a different interpolation where the two half-distances differ
    assert np.max(np.abs(old - ref)) > 1e-3 * scale
    return new, old


def test_k33_linear_interpolation_static(mesh):
    slope, Ki = _inputs(mesh, 1)
    N = int(mesh.nod2D)
    zbar = np.broadcast_to(np.asarray(mesh.zbar), (N, int(mesh.nl)))
    Z = np.broadcast_to(np.asarray(mesh.Z), (N, int(mesh.nl) - 1))
    _check(mesh, slope, Ki, zbar, Z)


def test_k33_linear_interpolation_zstar(mesh):
    slope, Ki = _inputs(mesh, 2)
    rng = np.random.default_rng(3)
    hnode = np.asarray(State.rest(mesh, ale_cfg=AleConfig()).hnode)
    hnode = np.where(np.asarray(mesh.node_layer_mask), hnode * (1.0 + 0.2 * rng.standard_normal(hnode.shape)), 0.0)
    zbar3, Z3d = ale.live_geometry(mesh, jnp.asarray(np.abs(hnode)))
    _check(mesh, slope, Ki, np.asarray(zbar3), np.asarray(Z3d)[:, : int(mesh.nl) - 1], zbar3=zbar3, Z3d=Z3d)


def test_k33_weights_agree_on_uniform_column(mesh):
    """Uniform layers: the interface is the midpoint of the centres, so the old and new weights coincide."""
    slope, Ki = _inputs(mesh, 4)
    N, nl = int(mesh.nod2D), int(mesh.nl)
    k = np.arange(nl, dtype=float)[None, :]
    zbar3 = jnp.asarray(np.broadcast_to(-10.0 * k, (N, nl)))               # 10 m layers
    Z3d = jnp.asarray(np.broadcast_to(-10.0 * k - 5.0, (N, nl)))           # their centres
    new = gm_redi.k33_augmentation(mesh, jnp.asarray(slope), jnp.asarray(Ki), zbar3=zbar3, Z3d=Z3d)
    old = gm_redi.k33_augmentation(mesh, jnp.asarray(slope), jnp.asarray(Ki), zbar3=zbar3, Z3d=Z3d,
                                   legacy_weights=True)
    assert np.max(np.abs(np.asarray(new) - np.asarray(old))) <= 1e-13 * np.max(np.abs(np.asarray(new)))
