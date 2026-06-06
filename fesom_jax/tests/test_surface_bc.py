"""Surface BC wiring + shortwave penetration — Phase 5, Task 5.6.

Verifies the NEW pieces wired into the step against **independent numpy loop
references** (a different code path: literal ports of the C loops), on the pi mesh
(always available, mesh-agnostic wiring):

* :func:`tracer_diff.impl_vert_diff_one` with ``bc_surf`` + ``sw_3d`` ==
  a per-node Thomas-solve port of ``diff_ver_part_impl_ale`` (``fesom_tracer_diff.c``)
  including the ``bc_surface`` increment and the ``sw_3d`` flux divergence.
* :func:`forcing.cal_shortwave_rad` == a per-node loop port of
  ``fesom_cal_shortwave_rad`` (``fesom_bulk.c:362-415``), incl. the break.
* **pi-path bit-identity:** ``bc=None``/``sw=None`` ⇒ exactly the Phase-2 result.
* **AD finiteness:** the bc_T (SST→heat_flux) seam has finite gradients (no masked NaN).

The end-to-end CORE2 dump gate (vs the C ``surfbc`` dump) is separate (Task 5.6 final).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import forcing, tracer_diff
from fesom_jax.config import VCPW
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh

jax.config.update("jax_enable_x64", True)

ALBW = 0.1


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(DEFAULT_PI_MESH_DIR)


# --------------------------------------------------------------------------
# numpy reference: diff_ver_part_impl_ale with bc + sw (per node, Thomas solve)
# --------------------------------------------------------------------------
def _ref_diff(mesh, T, Kv, hnode_new, dt, bc, sw):
    """Literal per-node port of ``diff_ver_part_impl_ale`` (gm=NULL, full-cell linfs)
    with the surface ``bc`` increment and the ``sw`` flux divergence (T only; pass
    ``sw=None`` for S). Returns ``T_new[nod2D, nl]``."""
    Z = np.asarray(mesh.Z)
    area = np.asarray(mesh.area)
    areasvol = np.asarray(mesh.areasvol)
    ulev = np.asarray(mesh.ulevels_nod2D)
    nlev = np.asarray(mesh.nlevels_nod2D)
    T = np.asarray(T).copy()
    Kv = np.asarray(Kv)
    hnn = np.asarray(hnode_new)
    N, nl = T.shape
    out = T.copy()
    for n in range(N):
        nzmin = ulev[n] - 1
        nzmax = nlev[n] - 1
        if nzmax - nzmin < 1:
            continue
        a = np.zeros(nl); b = np.zeros(nl); c = np.zeros(nl); tr = np.zeros(nl)
        # coefficients
        zinv1 = 0.0
        nz = nzmin
        zinv2 = 1.0 / (Z[nz] - Z[nz + 1])
        a[nz] = 0.0
        c[nz] = -Kv[n, nz + 1] * zinv2 * dt * area[n, nz + 1] / areasvol[n, nz]
        b[nz] = -c[nz] + hnn[n, nz]
        zinv1 = zinv2
        for nz in range(nzmin + 1, nzmax - 1):
            zinv2 = 1.0 / (Z[nz] - Z[nz + 1])
            a[nz] = -Kv[n, nz] * zinv1 * dt * (area[n, nz] / areasvol[n, nz])
            c[nz] = -Kv[n, nz + 1] * zinv2 * dt * area[n, nz + 1] / areasvol[n, nz]
            b[nz] = -a[nz] - c[nz] + hnn[n, nz]
            zinv1 = zinv2
        nz = nzmax - 1
        a[nz] = -Kv[n, nz] * zinv1 * dt * (area[n, nz] / areasvol[n, nz])
        c[nz] = 0.0
        b[nz] = -a[nz] + hnn[n, nz]
        # RHS
        nz = nzmin
        dz = hnn[n, nz]
        tr[nz] = -(b[nz] - dz) * T[n, nz] - c[nz] * T[n, nz + 1]
        for nz in range(nzmin + 1, nzmax - 1):
            dz = hnn[n, nz]
            tr[nz] = (-a[nz] * T[n, nz - 1] - (b[nz] - dz) * T[n, nz]
                      - c[nz] * T[n, nz + 1])
        nz = nzmax - 1
        dz = hnn[n, nz]
        tr[nz] = -a[nz] * T[n, nz - 1] - (b[nz] - dz) * T[n, nz]
        # surface BC
        if bc is not None:
            tr[nzmin] += bc[n]
        # shortwave divergence (T only)
        if sw is not None:
            for nz in range(nzmin, nzmax):
                ar = area[n, nz + 1] / areasvol[n, nz]
                tr[nz] += (sw[n, nz] - sw[n, nz + 1] * ar) * dt
        # Thomas solve over [nzmin, nzmax-1]
        cp = np.zeros(nl); tp = np.zeros(nl)
        cp[nzmin] = c[nzmin] / b[nzmin]
        tp[nzmin] = tr[nzmin] / b[nzmin]
        for nz in range(nzmin + 1, nzmax):
            m = b[nz] - cp[nz - 1] * a[nz]
            cp[nz] = c[nz] / m
            tp[nz] = (tr[nz] - tp[nz - 1] * a[nz]) / m
        sol = np.zeros(nl)
        sol[nzmax - 1] = tp[nzmax - 1]
        for nz in range(nzmax - 2, nzmin - 1, -1):
            sol[nz] = tp[nz] - cp[nz] * sol[nz + 1]
        for nz in range(nzmin, nzmax):
            out[n, nz] += sol[nz]
    return out


def _synthetic_inputs(mesh, seed=0):
    rng = np.random.default_rng(seed)
    N, nl = int(mesh.nod2D), int(mesh.nl)
    mask = np.asarray(mesh.node_layer_mask)
    T = np.where(mask, 10.0 + rng.standard_normal((N, nl)), 0.0)
    iface = np.asarray(mesh.node_iface_mask)
    Kv = np.where(iface, 1e-5 + 1e-4 * rng.random((N, nl)), 0.0)
    hnode_new = np.asarray(mesh.zbar_3d_n)
    hnode_new = np.where(mask, np.zeros((N, nl)), 0.0)
    # static reference layer thickness (matches State.rest)
    dz = np.zeros((N, nl))
    dz[:, :-1] = np.asarray(mesh.zbar_3d_n)[:, :-1] - np.asarray(mesh.zbar_3d_n)[:, 1:]
    hnode_new = np.where(mask, dz, 0.0)
    return T, Kv, hnode_new, mask


def test_bc_surf_matches_reference(mesh):
    """bc_T at the surface layer flows through the TDMA exactly as the C tr[nzmin] += bc."""
    T, Kv, hnn, mask = _synthetic_inputs(mesh, seed=1)
    N = int(mesh.nod2D)
    rng = np.random.default_rng(7)
    bc = -500.0 * (rng.standard_normal(N) * 50.0) / VCPW          # ~bc_T scale
    dt = 500.0
    ref = _ref_diff(mesh, T, Kv, hnn, dt, bc, None)
    got = tracer_diff.impl_vert_diff_one(mesh, jnp.asarray(T), jnp.asarray(Kv),
                                         jnp.asarray(hnn), dt=dt, bc_surf=jnp.asarray(bc))
    got = np.asarray(got)
    assert np.max(np.abs(got - ref)) < 1e-12, np.max(np.abs(got - ref))


def test_sw_divergence_matches_reference(mesh):
    """sw_3d flux divergence (T only) added to every layer matches the C loop."""
    T, Kv, hnn, mask = _synthetic_inputs(mesh, seed=2)
    N, nl = int(mesh.nod2D), int(mesh.nl)
    rng = np.random.default_rng(11)
    sw = np.where(mask, 1e-7 * rng.random((N, nl)), 0.0)
    dt = 500.0
    ref = _ref_diff(mesh, T, Kv, hnn, dt, None, sw)
    got = tracer_diff.impl_vert_diff_one(mesh, jnp.asarray(T), jnp.asarray(Kv),
                                         jnp.asarray(hnn), dt=dt, sw_3d=jnp.asarray(sw))
    assert np.max(np.abs(np.asarray(got) - ref)) < 1e-12


def test_bc_and_sw_together(mesh):
    """Both bc + sw together (the T tracer case)."""
    T, Kv, hnn, mask = _synthetic_inputs(mesh, seed=3)
    N, nl = int(mesh.nod2D), int(mesh.nl)
    rng = np.random.default_rng(13)
    bc = -500.0 * (rng.standard_normal(N) * 30.0) / VCPW
    sw = np.where(mask, 1e-7 * rng.random((N, nl)), 0.0)
    dt = 500.0
    ref = _ref_diff(mesh, T, Kv, hnn, dt, bc, sw)
    got = tracer_diff.impl_vert_diff_one(mesh, jnp.asarray(T), jnp.asarray(Kv),
                                         jnp.asarray(hnn), dt=dt,
                                         bc_surf=jnp.asarray(bc), sw_3d=jnp.asarray(sw))
    assert np.max(np.abs(np.asarray(got) - ref)) < 1e-12


def test_pi_path_bit_identical(mesh):
    """bc=None, sw=None ⇒ EXACTLY the Phase-2 diffusion (the 313-test invariant)."""
    T, Kv, hnn, mask = _synthetic_inputs(mesh, seed=4)
    base = tracer_diff.impl_vert_diff_one(mesh, jnp.asarray(T), jnp.asarray(Kv),
                                          jnp.asarray(hnn), dt=500.0)
    none = tracer_diff.impl_vert_diff_one(mesh, jnp.asarray(T), jnp.asarray(Kv),
                                          jnp.asarray(hnn), dt=500.0,
                                          bc_surf=None, sw_3d=None)
    assert np.array_equal(np.asarray(base), np.asarray(none))
    # and a zero bc / zero sw equals the no-forcing result (additive identity)
    zero_bc = tracer_diff.impl_vert_diff_one(
        mesh, jnp.asarray(T), jnp.asarray(Kv), jnp.asarray(hnn), dt=500.0,
        bc_surf=jnp.zeros(int(mesh.nod2D)), sw_3d=jnp.zeros((int(mesh.nod2D), int(mesh.nl))))
    assert np.max(np.abs(np.asarray(zero_bc) - np.asarray(base))) < 1e-13


# --------------------------------------------------------------------------
# cal_shortwave_rad vs a per-node loop reference
# --------------------------------------------------------------------------
def _ref_cal_shortwave(mesh, shortwave, chl, open_water):
    z = np.asarray(mesh.zbar_3d_n)
    ulev = np.asarray(mesh.ulevels_nod2D)
    nlev = np.asarray(mesh.nlevels_nod2D)
    N, nl = z.shape
    sw_3d = np.zeros((N, nl))
    heat_inc = np.zeros(N)
    for n in range(N):
        if not open_water[n]:
            continue
        swsurf = (1.0 - ALBW) * shortwave[n] * 0.54
        heat_inc[n] = swsurf
        cc = max(chl[n], 0.02)
        c = np.log10(cc); c2 = c * c; c3 = c2 * c; c4 = c3 * c; c5 = c4 * c
        v1 = 0.008 * c + 0.132 * c2 + 0.038 * c3 - 0.017 * c4 - 0.007 * c5
        v2 = 0.679 - v1
        v1 = 0.321 + v1
        sc1 = 1.54 - 0.197 * c + 0.166 * c2 - 0.252 * c3 - 0.055 * c4 + 0.042 * c5
        sc2 = 7.925 - 6.644 * c + 3.662 * c2 - 1.815 * c3 - 0.218 * c4 + 0.502 * c5
        swsurf /= VCPW
        nzmin = ulev[n] - 1
        nzmax = nlev[n] - 1
        sw_3d[n, nzmin] = swsurf
        for k in range(nzmin + 1, nzmax + 1):
            aux = v1 * np.exp(z[n, k] / sc1) + v2 * np.exp(z[n, k] / sc2)
            sw_3d[n, k] = swsurf * aux
            if aux < 1e-5 or k == nzmax:
                sw_3d[n, k] = 0.0
                break
    return heat_inc, sw_3d


def test_cal_shortwave_matches_reference(mesh):
    """JAX cal_shortwave_rad == the per-node C loop (incl. the aux<1e-5 break)."""
    N = int(mesh.nod2D)
    rng = np.random.default_rng(21)
    shortwave = 50.0 + 250.0 * rng.random(N)          # W/m²
    chl = 0.02 + 1.5 * rng.random(N)                  # mg/m³ (spans the Sweeney range)
    open_water = np.asarray(mesh.ulevels_nod2D) <= 1
    heat_ref, sw_ref = _ref_cal_shortwave(mesh, shortwave, chl, open_water)
    hf_pene, sw_3d = forcing.cal_shortwave_rad(
        mesh, jnp.zeros(N), jnp.asarray(shortwave), jnp.asarray(chl))
    # heat_flux_pene with input heat_flux=0 == the visible-band increment
    assert np.max(np.abs(np.asarray(hf_pene) - heat_ref)) < 1e-12
    assert np.max(np.abs(np.asarray(sw_3d) - sw_ref)) < 1e-12


def test_cal_shortwave_heat_flux_additive(mesh):
    """heat_flux_pene preserves d/d(SST): it is the bulk heat_flux + a constant offset."""
    N = int(mesh.nod2D)
    rng = np.random.default_rng(22)
    shortwave = 200.0 * rng.random(N)
    chl = jnp.full(N, 0.1)                             # constant-chl seam
    hf_in = jnp.asarray(10.0 * rng.standard_normal(N))
    hf0, _ = forcing.cal_shortwave_rad(mesh, jnp.zeros(N), jnp.asarray(shortwave), chl)
    hf1, _ = forcing.cal_shortwave_rad(mesh, hf_in, jnp.asarray(shortwave), chl)
    # hf1 - hf0 == hf_in (the heat_flux enters purely additively)
    assert np.max(np.abs(np.asarray(hf1 - hf0) - np.asarray(hf_in))) < 1e-12


# --------------------------------------------------------------------------
# AD finiteness of the bc_T (SST → heat_flux) seam through the TDMA
# --------------------------------------------------------------------------
def test_bc_gradient_finite(mesh):
    """d(loss)/d(bc) flows through the TDMA and is finite everywhere (incl. masked lanes)."""
    T, Kv, hnn, mask = _synthetic_inputs(mesh, seed=5)
    N = int(mesh.nod2D)
    Tj, Kvj, hj = jnp.asarray(T), jnp.asarray(Kv), jnp.asarray(hnn)
    maskj = jnp.asarray(mask)

    def loss(bc):
        Tn = tracer_diff.impl_vert_diff_one(mesh, Tj, Kvj, hj, dt=500.0, bc_surf=bc)
        return jnp.sum(jnp.where(maskj, Tn, 0.0) ** 2)

    g = jax.grad(loss)(jnp.zeros(N))
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.max(np.abs(np.asarray(g))) > 0.0           # the surface BC actually moves T
