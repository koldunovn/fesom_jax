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


def test_pp_convective_surrogate_keeps_the_forward_and_ramps_the_switch(mesh):
    """``surrogate_dn2`` (AdjointConfig.conv_surrogate): Kv and Av bitwise; d Kv/d N² =
    −(max(Kv_in, instabmix_kv) − Kv_in)/d on interior interfaces with −d < N² < 0 and 0
    elsewhere (the stable side included); d Kv/d Kv_in and every Av derivative as the exact
    switch outside the band."""
    uvnode, bvfreq = _synthetic_inputs(mesh, with_convection=True)
    Kv_in, Av_in = pp.pp_mixing(mesh, uvnode, bvfreq)
    d = 2.5e-5        # in band: N² = -1e-6 (k = 3); out: every stable interface, 1e-5 and up
    exact = lambda kv, av, b: pp.mo_convect(mesh, kv, av, b)
    surr = lambda kv, av, b: pp.mo_convect(mesh, kv, av, b, surrogate_dn2=d)
    (Kv0, Av0), vjp0 = jax.vjp(exact, Kv_in, Av_in, bvfreq)
    (Kv1, Av1), vjp1 = jax.vjp(surr, Kv_in, Av_in, bvfreq)
    assert np.array_equal(np.asarray(Kv0), np.asarray(Kv1))
    assert np.array_equal(np.asarray(Av0), np.asarray(Av1))

    ones, zeros = jnp.ones_like(Kv0), jnp.zeros_like(Av0)
    gkv0, gav0, gb0 = (np.asarray(x) for x in vjp0((ones, zeros)))
    gkv1, gav1, gb1 = (np.asarray(x) for x in vjp1((ones, zeros)))
    nmask = np.asarray(pp._interior_iface_mask(mesh.ulevels_nod2D, mesh.nlevels_nod2D, mesh.nl))
    b, kv = np.asarray(bvfreq), np.asarray(Kv_in)
    band = nmask & (b > -d) & (b < 0.0)
    want = np.where(band, -(np.maximum(kv, INSTABMIX_KV) - kv) / d, 0.0)
    assert np.all(gb0 == 0.0)                         # the exact switch has no N² derivative
    np.testing.assert_allclose(gb1, want, rtol=1e-12, atol=0.0)
    assert np.any(gb1[band] != 0.0) and np.all(gb1[~band] == 0.0)
    np.testing.assert_array_equal(gkv1[~band], gkv0[~band])
    assert np.all(gav1 == 0.0) and np.all(gav0 == 0.0)

    _, gav0, _ = vjp0((jnp.zeros_like(Kv0), jnp.ones_like(Av0)))
    _, gav1, gb1a = vjp1((jnp.zeros_like(Kv0), jnp.ones_like(Av0)))
    np.testing.assert_array_equal(np.asarray(gav1), np.asarray(gav0))
    assert np.all(np.asarray(gb1a) == 0.0)            # Av keeps the exact switch


def test_pp_convection_off_and_ramp_forward_modes(mesh):
    """The forward convection modes (TkeConfig.convection): "off" returns Kv/Av as given;
    "ramp" is the switch's value where N² <= -d (and exactly where w = 1), the closure's where
    N² >= 0, the linear blend in between (Av: the largest vertex weight), with the N²
    derivative -(max(Kv_in, instabmix_kv) - Kv_in)/d inside the band."""
    uvnode, bvfreq = _synthetic_inputs(mesh, with_convection=True)      # N² = -1e-6 at k = 3
    Kv_in, Av_in = pp.pp_mixing(mesh, uvnode, bvfreq)
    Kv_s, Av_s = pp.mo_convect(mesh, Kv_in, Av_in, bvfreq)
    Kv_o, Av_o = pp.mo_convect(mesh, Kv_in, Av_in, bvfreq, mode="off")
    assert np.array_equal(np.asarray(Kv_o), np.asarray(Kv_in))
    assert np.array_equal(np.asarray(Av_o), np.asarray(Av_in))

    # d well inside |N²|: every unstable interface gets w = 1 -> the switch, bitwise
    Kv_r, Av_r = pp.mo_convect(mesh, Kv_in, Av_in, bvfreq, mode="ramp", ramp_dn2=1e-7)
    assert np.array_equal(np.asarray(Kv_r), np.asarray(Kv_s))
    assert np.array_equal(np.asarray(Av_r), np.asarray(Av_s))

    d = 4e-6                                                            # w = 0.25 at k = 3
    ramp = lambda b: pp.mo_convect(mesh, Kv_in, Av_in, b, mode="ramp", ramp_dn2=d)
    Kv_r, Av_r = ramp(bvfreq)
    nmask = np.asarray(pp._interior_iface_mask(mesh.ulevels_nod2D, mesh.nlevels_nod2D, mesh.nl))
    b, kv = np.asarray(bvfreq), np.asarray(Kv_in)
    w = np.where(nmask, np.clip(-b / d, 0.0, 1.0), 0.0)
    np.testing.assert_allclose(np.asarray(Kv_r), (1 - w) * kv + w * np.maximum(kv, INSTABMIX_KV),
                               rtol=1e-14, atol=0.0)
    assert np.array_equal(np.asarray(Kv_r)[~(nmask & (b < 0))], kv[~(nmask & (b < 0))])
    gb = np.asarray(jax.grad(lambda x: jnp.sum(ramp(x)[0]))(bvfreq))
    band = nmask & (b > -d) & (b < 0)
    np.testing.assert_allclose(gb[band], -(np.maximum(kv, INSTABMIX_KV) - kv)[band] / d,
                               rtol=1e-12, atol=0.0)
    assert np.all(gb[~band] == 0.0)

    # "ramp_kv": the same node Kv, the switch's Av
    Kv_k, Av_k = pp.mo_convect(mesh, Kv_in, Av_in, bvfreq, mode="ramp_kv", ramp_dn2=d)
    assert np.array_equal(np.asarray(Kv_k), np.asarray(Kv_r))
    assert np.array_equal(np.asarray(Av_k), np.asarray(Av_s))

    with pytest.raises(ValueError, match="mode"):
        pp.mo_convect(mesh, Kv_in, Av_in, bvfreq, mode="soft")
    with pytest.raises(ValueError, match="ramp_dn2"):
        pp.mo_convect(mesh, Kv_in, Av_in, bvfreq, mode="ramp")
    with pytest.raises(ValueError, match="convective switch"):
        pp.mo_convect(mesh, Kv_in, Av_in, bvfreq, mode="off", surrogate_dn2=1e-6)


def test_tke_config_convection_options_validate():
    """TkeConfig.n2_source / convection / conv_ramp_dn2: defaults are FESOM2's (smoothed N²,
    the switch); bad values raise."""
    from fesom_jax.tke import TkeConfig
    assert TkeConfig().convection == "switch" and TkeConfig().n2_source == "smoothed"
    TkeConfig(n2_source="local").validate()
    with pytest.raises(ValueError, match="n2_source"):
        TkeConfig(n2_source="raw").validate()
    TkeConfig(convection="ramp", conv_ramp_dn2=1e-6).validate()
    TkeConfig(convection="ramp_kv", conv_ramp_dn2=1e-6).validate()
    TkeConfig(convection="off").validate()
    with pytest.raises(ValueError, match="convection"):
        TkeConfig(convection="soft").validate()
    for bad in (0.0, -1e-6, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="conv_ramp_dn2"):
            TkeConfig(convection="ramp", conv_ramp_dn2=bad).validate()
