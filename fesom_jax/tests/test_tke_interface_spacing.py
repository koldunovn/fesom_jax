"""``tke.interface_spacing`` (the IDEMIX column's ``dzt``, the Fortran ``dz_trr``): its surface end cap
is half the column's OWN top wet layer, ``hnode[ulevels_nod2D-1]/2`` (``gen_modules_cvmix_idemix.F90
:518-521``, ``gen_modules_cvmix_tke.F90:352-355``). It read ``hnode[:, 0]`` for every column, which
is the same without cavities and 0 under one (``hnode`` above the top wet layer is 0), a zero-volume
surface cell for the IDEMIX surface deposit ``fs / dzt[uln]`` and the lateral volume
``area * dzt`` (budget review 2026-10-26, finding interface-spacing-surface-cap-ignores-ulevels)."""

from types import SimpleNamespace

import numpy as np
import pytest

import jax.numpy as jnp

from fesom_jax import tke


def _fortran_dz_trr(hnode, uln0, nln0, Z):
    """Literal 0-based transcription of F90:518-521 on a static grid (layer centres ``Z``)."""
    out = np.zeros_like(hnode)
    for n in range(hnode.shape[0]):
        u, b = uln0[n], nln0[n]                       # surface and bottom interface
        for k in range(u + 1, b):
            out[n, k] = Z[k - 1] - Z[k]
        out[n, u] = hnode[n, u] / 2.0
        out[n, b] = hnode[n, b - 1] / 2.0
    return out


def test_surface_cap_is_the_own_top_layer_under_a_cavity():
    nl = 8
    zbar = -np.array([0.0, 10.0, 30.0, 60.0, 100.0, 150.0, 220.0, 300.0])
    Z = 0.5 * (zbar[:-1] + zbar[1:])                  # nl-1 layer centres (m, negative down)
    h = -np.diff(zbar)
    uln = np.array([1, 3])                            # 1-based: node 1 under a cavity (top layer 3)
    nlev = np.array([8, 8])
    hnode = np.tile(np.r_[h, 0.0], (2, 1))
    hnode[1, :uln[1] - 1] = 0.0                       # FESOM: no thickness above the top wet layer
    mesh = SimpleNamespace(nl=nl, ulevels_nod2D=jnp.asarray(uln), nlevels_nod2D=jnp.asarray(nlev),
                           Z=jnp.asarray(Z))
    got = np.asarray(tke.interface_spacing(mesh, jnp.asarray(hnode), None))
    want = _fortran_dz_trr(hnode, uln - 1, nlev - 1, Z)
    np.testing.assert_allclose(got, want, rtol=1e-15, atol=0.0)
    assert got[1, 2] == 15.0                          # hnode[2] / 2, not hnode[0] / 2 = 0

    # Port to d7e4e5d: the TKE path (tke_column_inputs -> _dz_trr without nzmin) keeps upstream's
    # hnode[:, :1]/2 cap, 0 at the cavity node's surface interface (test_tke_dzt_... below runs
    # upstream's mixing_tke itself)
    nzmin = jnp.asarray(uln - 1)[:, None]
    k = jnp.arange(nl)[None, :]
    nlev0 = jnp.asarray(nlev - 1)
    is_surf, is_bot = k == nzmin, k == nlev0[:, None]
    is_int = (k >= nzmin + 1) & (k <= nlev0[:, None] - 1)
    Zr = tke._layer_center_Z(mesh, None)
    dZ = tke._shift_down(Zr) - Zr
    tke_dz = tke._dz_trr(jnp.asarray(hnode), dZ, nlev0, is_surf, is_bot, is_int)
    assert float(tke_dz[1, 2]) == 0.0                 # upstream's cap without the cavity block


class _Captured(Exception):
    pass


def _cavity_pi():
    """The shipped pi mesh with ice-shelf cavities at three nodes (top wet level 2 or 3, each
    column keeping >= 2 wet layers below it), ``hnode`` 0 above the top wet layer, and smooth
    synthetic inputs for :func:`tke.mixing_tke`."""
    import dataclasses
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh()
    nl, n, e = int(mesh.nl), int(mesh.nod2D), int(mesh.elem2D)
    u = np.asarray(mesh.ulevels_nod2D).copy()
    nlv = np.asarray(mesh.nlevels_nod2D)
    deep = [i for i in range(n) if nlv[i] - 1 >= 5][:3]
    assert len(deep) == 3
    u[deep] = [2, 3, 2]
    cav = dataclasses.replace(mesh, ulevels_nod2D=jnp.asarray(u, dtype=jnp.asarray(mesh.ulevels_nod2D).dtype))
    h = -np.diff(np.asarray(mesh.zbar, dtype=np.float64))
    hnode = np.zeros((n, nl))
    for i in range(n):
        hnode[i, u[i] - 1:nlv[i] - 1] = h[u[i] - 1:nlv[i] - 1]
    rng = np.random.default_rng(0)
    inp = dict(uv=jnp.asarray(0.1 * rng.standard_normal((e, nl, 2))),
               bvfreq=jnp.asarray(1e-5 * (1.0 + rng.random((n, nl)))),
               tke=jnp.asarray(1e-4 * (1.0 + rng.random((n, nl)))),
               stress_node_surf=jnp.asarray(0.05 * rng.standard_normal((n, 2))),
               hnode=jnp.asarray(hnode))
    return cav, inp, u - 1


def _dzt_fed_to_the_core(monkeypatch, mesh, inp, cavity):
    """The ``dzt`` that upstream's :func:`tke.mixing_tke` hands the column core
    (``integrate_tke_column``'s 5th positional input), captured by a stand-in core."""
    from fesom_jax.params import Params
    seen = {}

    def core(*args, **kw):
        seen["dzt"] = np.asarray(args[4])
        raise _Captured

    monkeypatch.setattr(tke, "integrate_tke_column", core)
    with pytest.raises(_Captured):
        tke.mixing_tke(mesh, inp["uv"], inp["bvfreq"], inp["tke"], inp["stress_node_surf"],
                       inp["hnode"], tke.TkeConfig(), Params.defaults(), cavity=cavity)
    return seen["dzt"]


def test_tke_dzt_after_upstreams_cavity_recap_is_interface_spacing(monkeypatch):
    """Upstream's own path: ``mixing_tke(cavity=True)`` re-caps the TKE ``dz_trr`` from
    ``hnode[uln0]`` (4fe45cc) and shifts each column up by ``ulevels_nod2D - 1``; the ``dzt`` it
    feeds the column core is :func:`tke.interface_spacing` under the same shift, bitwise. With
    ``cavity=False`` the core gets upstream's ``hnode[:, 0]/2`` cap, which is 0 at the cavity
    nodes' surface interface and differs from ``interface_spacing`` there only."""
    mesh, inp, u0 = _cavity_pi()
    nl = int(mesh.nl)
    isp = np.asarray(tke.interface_spacing(mesh, inp["hnode"], None))
    k = np.arange(nl)[None, :]
    src = k + u0[:, None]                                   # column-local k <- global k + u
    shifted = np.where(src <= nl - 1, np.take_along_axis(isp, np.clip(src, 0, nl - 1), axis=1), 0.0)
    got = _dzt_fed_to_the_core(monkeypatch, mesh, inp, cavity=True)
    np.testing.assert_array_equal(got, shifted)

    flat = _dzt_fed_to_the_core(monkeypatch, mesh, inp, cavity=False)
    surf = k == u0[:, None]
    cav = surf & (u0[:, None] > 0)
    assert cav.sum() == 3 and np.all(flat[cav] == 0.0) and np.all(isp[cav] > 0.0)
    np.testing.assert_array_equal(flat[~cav], isp[~cav])


def test_no_cavity_mesh_bitwise_unchanged():
    """On the shipped pi mesh (ulevels_nod2D == 1 everywhere) the cap is the old ``hnode[:, :1]/2``."""
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh()
    assert np.all(np.asarray(mesh.ulevels_nod2D) == 1)
    hnode = jnp.asarray(np.asarray(mesh.hnode) if hasattr(mesh, "hnode") else
                        np.tile(-np.diff(np.asarray(mesh.zbar)), (int(mesh.nod2D), 1)))
    if hnode.shape[1] == mesh.nl - 1:
        hnode = jnp.concatenate([hnode, jnp.zeros_like(hnode[:, :1])], axis=1)
    got = np.asarray(tke.interface_spacing(mesh, hnode, None))
    k = np.arange(mesh.nl)[None, :]
    surf = k == (np.asarray(mesh.ulevels_nod2D) - 1)[:, None]
    assert np.array_equal(got[surf], (np.asarray(hnode)[:, :1] / 2.0)[:, 0])
