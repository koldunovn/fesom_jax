"""Task 2.3 gate — PP vertical mixing + convective adjustment (substep 4).

Three layers of verification:

1. **Dump gate (step 1):** at rest (uv=0) ``Kv``/``Av`` reduce to the background
   ``K_ver``/``A_ver`` on interior interfaces (0 at surface/bottom). Verified vs the
   C dump at all node (Kv) and element (Av) probes.
2. **Synthetic-shear unit test:** the step-1 dump leaves shear=0, so the
   shear/N²/factor/Av/Kv³ path is checked against an **independent loop-based numpy
   reference** of ``fesom_pp.c`` for a nonzero ``uvnode`` + positive ``N²``, and a
   convective case (``N²<0`` → bump to ``instabmix_kv``).
3. **AD gate:** ``d(Σ Kv)/d(uvnode)`` (reverse-mode) vs central FD in the smooth
   regime (positive N², away from the ``max(N²,0)`` / convective kinks).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import eos, ic, pp
from fesom_jax.config import A_VER, INSTABMIX_KV, K_VER, MIX_COEFF_PP
from fesom_jax.io_dump import find_record
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh

NODE_PROBES = [1001, 1500, 2000, 2500, 3000]
ELEM_PROBES = [1757, 2656, 3688, 4604, 5575]


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (Task 0.3)")
    return load_mesh()


# --------------------------------------------------------------------------
# Independent loop-based numpy reference of fesom_pp.c (different code path)
# --------------------------------------------------------------------------
def _pp_reference(mesh, uvnode, bvfreq, convect=True):
    nl = mesh.nl
    Z = np.asarray(mesh.Z)
    uln = np.asarray(mesh.ulevels_nod2D)
    nln = np.asarray(mesh.nlevels_nod2D)
    ule = np.asarray(mesh.ulevels)
    nle = np.asarray(mesh.nlevels)
    en = np.asarray(mesh.elem_nodes)
    uvn = np.asarray(uvnode)
    bv = np.asarray(bvfreq)

    factor = np.zeros((mesh.nod2D, nl))
    for n in range(mesh.nod2D):
        for nz in range(uln[n], nln[n] - 1):           # [nzmin+1, nzmax)
            dz = Z[nz - 1] - Z[nz]
            dzi = 1.0 / dz
            du = uvn[n, nz - 1, 0] - uvn[n, nz, 0]
            dv = uvn[n, nz - 1, 1] - uvn[n, nz, 1]
            shear = (du * du + dv * dv) * dzi * dzi
            nsqp = max(bv[n, nz], 0.0)
            factor[n, nz] = shear / (shear + 5.0 * nsqp + 1e-14)

    Av = np.zeros((mesh.elem2D, nl))
    for e in range(mesh.elem2D):
        n0, n1, n2 = en[e]
        for nz in range(ule[e], nle[e] - 1):
            k0, k1, k2 = factor[n0, nz], factor[n1, nz], factor[n2, nz]
            Av[e, nz] = MIX_COEFF_PP * (k0 * k0 + k1 * k1 + k2 * k2) / 3.0 + A_VER

    Kv = np.zeros((mesh.nod2D, nl))
    for n in range(mesh.nod2D):
        for nz in range(uln[n], nln[n] - 1):
            f = factor[n, nz]
            Kv[n, nz] = MIX_COEFF_PP * f * f * f + K_VER

    if convect:
        for n in range(mesh.nod2D):
            for nz in range(uln[n], nln[n] - 1):
                if bv[n, nz] < 0.0 and Kv[n, nz] < INSTABMIX_KV:
                    Kv[n, nz] = INSTABMIX_KV
        for e in range(mesh.elem2D):
            n0, n1, n2 = en[e]
            for nz in range(ule[e], nle[e] - 1):
                if (bv[n0, nz] < 0 or bv[n1, nz] < 0 or bv[n2, nz] < 0) \
                        and Av[e, nz] < INSTABMIX_KV:
                    Av[e, nz] = INSTABMIX_KV
    return Kv, Av


def _synthetic_inputs(mesh, *, with_convection=False):
    """Smooth nonzero uvnode (depth shear) + N² (positive, or some negative)."""
    k = np.arange(mesh.nl)
    u = 0.1 * np.cos(0.3 * k)
    v = 0.05 * np.sin(0.2 * k)
    uvnode = np.zeros((mesh.nod2D, mesh.nl, 2))
    uvnode[:, :, 0] = u[None, :]
    uvnode[:, :, 1] = v[None, :]
    uvnode = jnp.where(mesh.node_layer_mask[..., None], jnp.asarray(uvnode), 0.0)

    nsq = 1e-5 * (k + 1.0)
    bv = np.broadcast_to(nsq[None, :], (mesh.nod2D, mesh.nl)).copy()
    if with_convection:
        bv[:, 3] = -1e-6  # an unstable interface everywhere
    bvfreq = jnp.where(mesh.node_iface_mask, jnp.asarray(bv), 0.0)
    return uvnode, bvfreq


# --------------------------------------------------------------------------
# 1. step-1 dump gate
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gid", NODE_PROBES)
def test_Kv_matches_dump_step1(load_dump, mesh, gid):
    recs = load_dump("pi_cdump.00000")
    st = ic.initial_state(mesh)
    _, _, bvfreq = eos.compute_pressure_bv(mesh, st.T, st.S, st.hnode)
    Kv, _, _ = pp.mixing_pp(mesh, st.uv, bvfreq)
    rec = find_record(recs, step=1, substep=4, field="Kv", probe_gid=gid)
    n = rec.nlevels
    assert np.allclose(np.asarray(Kv)[gid - 1][:n], rec.values, atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("gid", ELEM_PROBES)
def test_Av_matches_dump_step1(load_dump, mesh, gid):
    recs = load_dump("pi_cdump.00000")
    st = ic.initial_state(mesh)
    _, _, bvfreq = eos.compute_pressure_bv(mesh, st.T, st.S, st.hnode)
    _, Av, _ = pp.mixing_pp(mesh, st.uv, bvfreq)
    rec = find_record(recs, step=1, substep=4, field="Av", probe_gid=gid)
    n = rec.nlevels
    assert np.allclose(np.asarray(Av)[gid - 1][:n], rec.values, atol=1e-14, rtol=1e-12)


# --------------------------------------------------------------------------
# 2. compute_vel_nodes + synthetic PP vs numpy reference
# --------------------------------------------------------------------------
def test_compute_vel_nodes_constant(mesh):
    """Area-weighted average of a constant element velocity is that constant."""
    uv = jnp.where(mesh.elem_layer_mask[..., None],
                   jnp.broadcast_to(jnp.array([1.0, 2.0]), (mesh.elem2D, mesh.nl, 2)), 0.0)
    uvnode = np.asarray(pp.compute_vel_nodes(mesh, uv))
    m = np.asarray(mesh.node_layer_mask)
    assert np.allclose(uvnode[..., 0][m], 1.0, atol=1e-12)
    assert np.allclose(uvnode[..., 1][m], 2.0, atol=1e-12)


def test_pp_synthetic_matches_reference(mesh):
    """Nonzero shear + positive N²: JAX PP == independent loop reference."""
    uvnode, bvfreq = _synthetic_inputs(mesh)
    Kv, Av = pp.pp_mixing(mesh, uvnode, bvfreq)
    Kv, Av = pp.mo_convect(mesh, Kv, Av, bvfreq)
    Kv_ref, Av_ref = _pp_reference(mesh, uvnode, bvfreq)
    assert np.allclose(np.asarray(Kv), Kv_ref, atol=1e-14, rtol=1e-12)
    assert np.allclose(np.asarray(Av), Av_ref, atol=1e-14, rtol=1e-12)
    # the synthetic shear must actually move Kv off background somewhere
    assert np.max(np.asarray(Kv)) > K_VER * 1.0001


def test_pp_convective_adjustment(mesh):
    """An unstable interface (N²<0) bumps Kv (node) and Av (elem) to instabmix_kv."""
    uvnode, bvfreq = _synthetic_inputs(mesh, with_convection=True)
    Kv, Av = pp.pp_mixing(mesh, uvnode, bvfreq)
    Kv, Av = pp.mo_convect(mesh, Kv, Av, bvfreq)
    Kv_ref, Av_ref = _pp_reference(mesh, uvnode, bvfreq, convect=True)
    assert np.allclose(np.asarray(Kv), Kv_ref, atol=1e-14, rtol=1e-12)
    assert np.allclose(np.asarray(Av), Av_ref, atol=1e-14, rtol=1e-12)
    # the unstable interface (nz=3) must be at instabmix_kv on wet columns
    nmask = np.asarray(mesh.node_layer_mask)[:, 3]
    assert np.allclose(np.asarray(Kv)[nmask, 3], INSTABMIX_KV)


# --------------------------------------------------------------------------
# 3. AD gate
# --------------------------------------------------------------------------
def test_pp_gradient_ad_vs_fd(mesh):
    """d(Σ Kv)/d(uvnode) AD vs central FD in the smooth regime (N²>0)."""
    uvnode, bvfreq = _synthetic_inputs(mesh)

    def loss(uvn):
        Kv, _ = pp.pp_mixing(mesh, uvn, bvfreq)
        return jnp.sum(Kv)

    grad_ad = np.asarray(jax.grad(loss)(uvnode))
    n, nz, comp = 1000, 5, 0
    g_ad = float(grad_ad[n, nz, comp])
    assert np.isfinite(g_ad) and g_ad != 0.0
    best = np.inf
    for h in (1e-5, 1e-6, 1e-7):
        up = uvnode.at[n, nz, comp].add(h)
        um = uvnode.at[n, nz, comp].add(-h)
        g_fd = float((loss(up) - loss(um)) / (2 * h))
        best = min(best, abs(g_ad - g_fd) / max(abs(g_fd), 1e-300))
    assert best < 1e-6, f"AD vs FD rel err {best:.2e}"
