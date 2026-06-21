"""Task 2.9 gate — ALE step (linfs): w + hnode_new (substep 13).

* **Dump gate (step 1):** ``w`` (vertical velocity) and ``hnode_new`` (layer
  thickness) at all 5 node probes vs the C dump. Step 1 is a **real** gate here:
  after :func:`~fesom_jax.momentum.update_vel` the velocity is the first nonzero
  (wind-driven) ``uv``, so ``w`` is non-trivial — unlike the at-rest kernels.
  ``hnode_new = hnode`` in linfs (a copy of the static reference thickness).
* **Structure:** ``w`` is nonzero/wind-driven; the bottom interface ``w[nzmax]=0``
  (no-flux BC); ``hnode_new`` equals the input ``hnode`` bit-for-bit.
* **Synthetic test:** the at-rest/wind-driven ``uv`` is small, so the full
  per-level edge→node scatter + reverse cumsum + ``÷area`` is also checked against
  an independent loop-based numpy reference of ``fesom_ale_vert_vel_linfs`` with
  synthetic ``O(0.1)`` velocity.
* **AD gate:** ``w`` is **linear** in ``uv`` → ``jax.grad`` equals central FD
  exactly (to roundoff), is finite even at ``uv=0`` (the at-rest kink-free input),
  and flows end-to-end from the upstream ``du`` through ``custom_linear_solve`` and
  :func:`~fesom_jax.momentum.update_vel` into ``w``.

Fidelity note (see PORTING_LESSONS): like ``hbar``, the stage-3 ``÷area``
(``~1e9–1e12 m²``) divides the near-cancelling per-level divergence's amplified
absolute error back down, so ``w`` matches the dump to ~1e-20 on CPU despite the
loose scatter floor of the raw divergence.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ale, eos, forcing, ic, momentum, pgf, pp, ssh, tracer_adv, verify
from fesom_jax.io_dump import find_record
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.state import State

NODE_PROBES = [1001, 1500, 2000, 2500, 3000]
DT = 100.0

# w (substep 13) dump-gate floor. Observed max|Δ| ~4e-20 on CPU: w is the same
# near-cancelling transport divergence as ssh_rhs (loose ~1e-7 floor) but the
# stage-3 ÷area (1e9–1e12 m²) suppresses it — exactly as for hbar (HBAR_ATOL=1e-12).
# Gate at the hbar-class floor (safe across CPU/GPU scatter reassociation, still
# many orders below the ~1e-6 signal so any real bug fails loudly).
W_ATOL = 1e-12


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (Task 0.3)")
    return load_mesh()


@pytest.fixture(scope="module")
def chain(mesh):
    """Step-1 chain through substep 13. Returns ``(st, du, uv, w, hnode_new)``
    where ``uv`` is the post-:func:`update_vel` velocity (substep 10) and ``w`` /
    ``hnode_new`` are the substep-13 fields."""
    st = ic.initial_state(mesh)
    _, hp, bv = eos.compute_pressure_bv(mesh, st.T, st.S, st.hnode)
    px, py = pgf.pressure_force_linfs(mesh, hp)
    _, Av, _ = pp.mixing_pp(mesh, st.uv, bv)
    uvr, _ = momentum.compute_vel_rhs(mesh, st.uv, st.uv_rhsAB, st.eta_n, px, py,
                                      st.w_e, st.hnode, is_first_step=True, dt=DT)
    uvr = momentum.visc_filt_bidiff(mesh, st.uv, uvr, dt=DT)
    du = momentum.impl_vert_visc(mesh, st.uv, uvr, Av, forcing.surface_stress(mesh),
                                 dt=DT)
    ssh_rhs = ssh.compute_ssh_rhs(mesh, st.uv, du, st.helem)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    d_eta = ssh.solve_ssh(op, ssh_rhs)
    uv = momentum.update_vel(mesh, st.uv, du, d_eta, dt=DT)
    w = ale.compute_w(mesh, uv, st.helem)
    hnode_new = ale.thickness_linfs(st.hnode)
    return st, du, uv, w, hnode_new


# --------------------------------------------------------------------------
# Independent loop-based numpy reference of fesom_ale_vert_vel_linfs.
# --------------------------------------------------------------------------
def _w_ref(mesh, uv, helem):
    """Transcription of ``fesom_ale_vert_vel_linfs`` (``fesom_ale.c:77-187``):
    per-level edge→node divergence scatter, reverse (bottom→top) cumsum, ÷area."""
    edges = np.asarray(mesh.edges); etri = np.asarray(mesh.edge_tri)
    cross = np.asarray(mesh.edge_cross_dxdy)
    ule, nle = np.asarray(mesh.ulevels), np.asarray(mesh.nlevels)
    uln, nln = np.asarray(mesh.ulevels_nod2D), np.asarray(mesh.nlevels_nod2D)
    area = np.asarray(mesh.area)
    uv, h = np.asarray(uv), np.asarray(helem)
    w = np.zeros((mesh.nod2D, mesh.nl))
    for ed in range(mesh.myDim_edge2D):                      # step 2: edge fluxes
        n1, n2 = edges[ed]; el1, el2 = etri[ed]
        if el1 >= 0:
            dxL, dyL = cross[ed, 0], cross[ed, 1]
            for nz in range(ule[el1] - 1, nle[el1] - 1):
                u, v = uv[el1, nz]
                c1 = (v * dxL - u * dyL) * h[el1, nz]
                w[n1, nz] += c1; w[n2, nz] -= c1
        if el2 >= 0:
            dxR, dyR = cross[ed, 2], cross[ed, 3]
            for nz in range(ule[el2] - 1, nle[el2] - 1):
                u, v = uv[el2, nz]
                c1 = -(v * dxR - u * dyR) * h[el2, nz]
                w[n1, nz] += c1; w[n2, nz] -= c1
    for n in range(mesh.nod2D):                              # step 3: cumsum up
        nzmin, nzmax = uln[n] - 1, nln[n] - 1
        for nz in range(nzmax - 1, nzmin - 1, -1):
            w[n, nz] += w[n, nz + 1]
    for n in range(mesh.nod2D):                              # step 4: ÷ area
        nzmin, nzmax = uln[n] - 1, nln[n] - 1
        for nz in range(nzmin, nzmax):
            a = area[n, nz]
            if a > 0.0:
                w[n, nz] /= a
    return w


def _synthetic_uv(mesh):
    e = np.arange(mesh.elem2D)[:, None]; k = np.arange(mesh.nl)[None, :]
    uv = np.stack([0.1 * np.cos(0.3 * k + 0.01 * e),
                   0.05 * np.sin(0.2 * k - 0.013 * e)], axis=-1)
    m = mesh.elem_layer_mask[..., None]
    return jnp.where(m, jnp.asarray(uv), 0.0)


# --------------------------------------------------------------------------
# 1. substep 13 — w dump gate
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gid", NODE_PROBES)
def test_w_matches_dump_step1(load_dump, mesh, chain, gid):
    """w (substep 13) at the node probes — the wind-driven vertical velocity."""
    _, _, _, w, _ = chain
    rec = find_record(load_dump("pi_cdump.00000"), step=1, substep=13,
                      field="w", probe_gid=gid)
    verify.assert_close(np.asarray(w)[gid - 1], rec, kind="scatter", atol=W_ATOL)


def test_w_nonzero_and_wind_driven(mesh, chain):
    """After update_vel uv is nonzero (wind-driven) → w is genuinely nonzero, so
    the substep-13 dump gate is meaningful (not a trivial zero match)."""
    _, _, _, w, _ = chain
    assert float(jnp.max(jnp.abs(w))) > 1e-9


def test_w_bottom_interface_zero(mesh, chain):
    """The no-flux bottom BC: w at each node's bottom interface (nlevels_nod2D−1)
    is exactly 0 (the reverse cumsum starts from a zero bottom and the scatter
    never touches the bottom interface)."""
    _, _, _, w, _ = chain
    w = np.asarray(w)
    nbot = np.asarray(mesh.nlevels_nod2D) - 1
    assert np.max(np.abs(w[np.arange(mesh.nod2D), nbot])) == 0.0


# --------------------------------------------------------------------------
# 2. substep 13 — hnode_new dump gate (linfs: a static copy of hnode)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gid", NODE_PROBES)
def test_hnode_new_matches_dump_step1(load_dump, mesh, chain, gid):
    """hnode_new (substep 13) == the static reference thickness; bit-for-bit map."""
    *_, hnode_new = chain
    rec = find_record(load_dump("pi_cdump.00000"), step=1, substep=13,
                      field="hnode_new", probe_gid=gid)
    verify.assert_close(np.asarray(hnode_new)[gid - 1], rec, kind="map")


def test_hnode_new_is_hnode_copy(mesh, chain):
    """In linfs (dh/dt=0) hnode_new is a verbatim copy of the static hnode."""
    st, _, _, _, hnode_new = chain
    assert np.array_equal(np.asarray(hnode_new), np.asarray(st.hnode))


# --------------------------------------------------------------------------
# 3. synthetic vs numpy reference (exercise the full scatter+cumsum+÷area)
# --------------------------------------------------------------------------
def test_w_synthetic_matches_reference(mesh):
    """Nonzero O(0.1) uv drives the full per-level transport divergence + reverse
    cumsum + ÷area, vs an independent loop reference. Pure scatter reassociation
    (~1e-18 abs floor on a ~3e-3 field) — far below any real-bug signal."""
    st = State.rest(mesh)
    uv = _synthetic_uv(mesh)
    out = np.asarray(ale.compute_w(mesh, uv, st.helem))
    ref = _w_ref(mesh, uv, st.helem)
    assert np.allclose(out, ref, rtol=1e-10, atol=1e-13)
    assert np.max(np.abs(out)) > 0  # genuinely exercised


# --------------------------------------------------------------------------
# 4. AD gates — w is linear in uv
# --------------------------------------------------------------------------
def test_w_gradient_vs_fd(mesh):
    """``compute_w`` is linear in uv (scatter + cumsum + ÷static-area) → AD is
    exact; central FD (exact for a linear map) agrees at a few uv components."""
    st = State.rest(mesh)
    uv = _synthetic_uv(mesh)
    wv = jnp.asarray(np.random.RandomState(7).randn(mesh.nod2D, mesh.nl))

    def loss(u):
        return jnp.sum(wv * ale.compute_w(mesh, u, st.helem))

    g_ad = np.asarray(jax.grad(loss)(uv))
    assert np.all(np.isfinite(g_ad))
    h = 1e-3
    for idx in [(100, 4, 0), (2500, 10, 1), (500, 2, 0)]:
        gf = float((loss(uv.at[idx].add(h)) - loss(uv.at[idx].add(-h))) / (2 * h))
        assert abs(g_ad[idx] - gf) <= 1e-8 * max(abs(gf), 1.0) + 1e-12, \
            f"{idx}: AD {g_ad[idx]:.6e} vs FD {gf:.6e}"


def test_w_gradient_finite_at_rest(mesh):
    """No NaN gradient at the at-rest uv=0 input (a linear map has no kink, but
    the safe ÷area guard must also keep the backward pass finite)."""
    st = State.rest(mesh)
    wv = jnp.asarray(np.random.RandomState(9).randn(mesh.nod2D, mesh.nl))

    def loss(u):
        return jnp.sum(wv * ale.compute_w(mesh, u, st.helem))

    g0 = np.asarray(jax.grad(loss)(jnp.zeros((mesh.elem2D, mesh.nl, 2))))
    assert np.all(np.isfinite(g0))


def test_grad_flows_du_to_w(mesh, chain):
    """End-to-end implicit-diff gate: d(Σ w)/d(du) flows through compute_ssh_rhs →
    custom_linear_solve → update_vel → compute_w. Finite and nonzero."""
    st, du, *_ = chain
    op = ssh.build_ssh_operator(mesh, dt=DT)

    def loss(du_in):
        rhs = ssh.compute_ssh_rhs(mesh, st.uv, du_in, st.helem)
        d_eta = ssh.solve_ssh(op, rhs)
        uv = momentum.update_vel(mesh, st.uv, du_in, d_eta, dt=DT)
        return jnp.sum(ale.compute_w(mesh, uv, st.helem))

    g = np.asarray(jax.grad(loss)(du))
    assert np.all(np.isfinite(g))
    assert np.max(np.abs(g)) > 0


# --------------------------------------------------------------------------
# 6. Task 4.2 — vertical CFL + explicit/implicit w split (wsplit)
#
# use_wsplit=0 in the pi/CORE2-d1800 reference config (fesom_constants.h:56), so the
# split is the identity (w_e=w, w_i=0) on the dump-matching path — and indeed the pi
# CFL never approaches maxcfl=1.0 (max cfl_z ~1e-4). cfl_z is verified against the
# literal C loop; the ACTIVE split branch (the part the dump can't reach) is verified
# against the numpy reference with a synthetic super-critical CFL.
# --------------------------------------------------------------------------
def _cfl_z_ref(mesh, w, hnode_new, dt):
    """Transcription of ``fesom_ale_compute_cflz`` (``fesom_ale.c:204``): per layer add
    ``|w_top|·dt/h`` to its top interface and ``|w_bot|·dt/h`` to its bottom interface."""
    uln, nln = np.asarray(mesh.ulevels_nod2D), np.asarray(mesh.nlevels_nod2D)
    w, h = np.asarray(w), np.asarray(hnode_new)
    cfl = np.zeros((mesh.nod2D, mesh.nl))
    for n in range(mesh.nod2D):
        for nz in range(uln[n] - 1, nln[n] - 1):           # layer loop
            hh = h[n, nz]
            if hh <= 0.0:
                continue
            cfl[n, nz] += abs(w[n, nz]) * dt / hh
            cfl[n, nz + 1] += abs(w[n, nz + 1]) * dt / hh
    return cfl


def _wvel_split_ref(mesh, w, cfl_z, use_wsplit, maxcfl):
    """Transcription of ``fesom_ale_compute_wvel_split`` (``fesom_ale.c:241``)."""
    uln, nln = np.asarray(mesh.ulevels_nod2D), np.asarray(mesh.nlevels_nod2D)
    w, cfl = np.asarray(w), np.asarray(cfl_z)
    inv_maxcfl = 1.0 / max(maxcfl, 1e-12)
    w_e, w_i = np.zeros_like(w), np.zeros_like(w)
    for n in range(mesh.nod2D):
        for nz in range(uln[n] - 1, nln[n] - 1 + 1):       # interface loop, INCLUSIVE
            ww, c = w[n, nz], cfl[n, nz]
            if use_wsplit and c > maxcfl:
                dd = max(c - maxcfl, 0.0) * inv_maxcfl
                inv = 1.0 / (1.0 + dd)
                w_e[n, nz], w_i[n, nz] = ww * inv, ww * dd * inv
            else:
                w_e[n, nz], w_i[n, nz] = ww, 0.0
    return w_e, w_i


def test_cfl_z_matches_reference(mesh, chain):
    """cfl_z (vectorized) == the literal C loop, on the real step-1 w."""
    _, _, _, w, hnode_new = chain
    cfl = np.asarray(ale.compute_cfl_z(mesh, w, hnode_new, dt=DT))
    ref = _cfl_z_ref(mesh, w, hnode_new, DT)
    m = np.asarray(mesh.node_iface_mask)
    assert np.allclose(cfl[m], ref[m], atol=1e-15, rtol=1e-12)
    assert np.max(np.abs(cfl)) > 0                          # nonzero (w is wind-driven)


def test_wvel_split_identity_when_off(mesh, chain):
    """use_wsplit=0 (the reference config) ⇒ w_e=w, w_i=0 — the dump-matching identity."""
    _, _, _, w, hnode_new = chain
    cfl = ale.compute_cfl_z(mesh, w, hnode_new, dt=DT)
    w_e, w_i = ale.compute_wvel_split(mesh, w, cfl, use_wsplit=False)
    assert np.array_equal(np.asarray(w_e), np.asarray(w))
    assert np.all(np.asarray(w_i) == 0.0)


def test_wvel_split_active_matches_reference(mesh):
    """The ACTIVE split (use_wsplit=1) vs the numpy reference, with a synthetic
    super-critical CFL that straddles maxcfl (some interfaces split, some don't)."""
    n = np.arange(mesh.nod2D)[:, None]
    k = np.arange(mesh.nl)[None, :]
    imask = np.asarray(mesh.node_iface_mask)
    w = jnp.where(jnp.asarray(imask),
                  jnp.asarray(0.1 * np.cos(0.3 * k + 0.01 * n)), 0.0)
    cfl = jnp.where(jnp.asarray(imask),                     # ranges ~0.5..2.5, crosses 1.0
                    jnp.asarray(1.5 + 1.0 * np.sin(0.2 * k + 0.013 * n)), 0.0)
    w_e, w_i = ale.compute_wvel_split(mesh, w, cfl, use_wsplit=True, maxcfl=1.0)
    re, ri = _wvel_split_ref(mesh, w, cfl, use_wsplit=True, maxcfl=1.0)
    we, wi = np.asarray(w_e), np.asarray(w_i)
    assert np.allclose(we[imask], re[imask], atol=1e-15, rtol=1e-13)
    assert np.allclose(wi[imask], ri[imask], atol=1e-15, rtol=1e-13)
    # load-bearing: the split must be genuinely active AND inactive across interfaces
    assert np.any((wi != 0.0)[imask]), "split never activated — raise the synthetic CFL"
    assert np.any((we == np.asarray(w))[imask] & (np.asarray(cfl) <= 1.0)[imask])
    # explicit + implicit recovers the total w where split is active (w_e+w_i == w)
    assert np.allclose((we + wi)[imask], np.asarray(w)[imask], atol=1e-14)


def test_impl_vert_visc_wi_zero_is_identity(mesh):
    """w_split: passing w_i=0 adds EXACTLY zero to the momentum tridiagonal, so enabling the
    split never perturbs a low-CFL region (w_i=0). impl_vert_visc(w_i=None) == (w_i=0)."""
    st = ic.initial_state(mesh)
    Av = jnp.zeros((mesh.elem2D, mesh.nl))
    uvr = jnp.zeros((mesh.elem2D, mesh.nl, 2))
    ss = jnp.zeros((mesh.elem2D, 2))
    du_none = momentum.impl_vert_visc(mesh, st.uv, uvr, Av, ss, dt=DT)
    du_zero = momentum.impl_vert_visc(mesh, st.uv, uvr, Av, ss, dt=DT,
                                      w_i=jnp.zeros((mesh.nod2D, mesh.nl)))
    assert np.array_equal(np.asarray(du_none), np.asarray(du_zero))


def test_adv_tra_vert_impl_wi_zero_is_identity(mesh):
    """w_split tracer implicit solve: w_i=0 ⇒ M = hnode_new·I ⇒ the identity (T_new == T)."""
    st = ic.initial_state(mesh)
    T_new = tracer_adv.adv_tra_vert_impl(mesh, jnp.zeros((mesh.nod2D, mesh.nl)),
                                         st.T, st.hnode, dt=DT)
    assert np.array_equal(np.asarray(T_new), np.asarray(st.T))


def test_adv_tra_vert_impl_active_finite_and_bounded(mesh):
    """A synthetic w_i: the implicit upwind solve stays FINITE, genuinely moves T, and stays
    globally bounded (implicit upwind ⇒ no blow-up). The on-path mechanics smoke-test."""
    st = ic.initial_state(mesh)
    n = np.arange(mesh.nod2D)[:, None]
    k = np.arange(mesh.nl)[None, :]
    imask = np.asarray(mesh.node_iface_mask)
    w_i = jnp.where(jnp.asarray(imask), jnp.asarray(0.001 * np.cos(0.3 * k + 0.01 * n)), 0.0)
    T_new = np.asarray(tracer_adv.adv_tra_vert_impl(mesh, w_i, st.T, st.hnode, dt=DT))
    vmask = np.asarray(mesh.node_layer_mask)
    Tv = np.asarray(st.T)[vmask]
    assert np.all(np.isfinite(T_new))
    assert not np.array_equal(T_new, np.asarray(st.T))                  # genuinely advected
    assert T_new[vmask].min() >= Tv.min() - 1.0                         # globally bounded
    assert T_new[vmask].max() <= Tv.max() + 1.0


def test_wvel_split_gradient_finite(mesh):
    """AD through the active split is finite (the 1/(1+dd) is smooth, dd≥0; the
    cfl>maxcfl select is a measure-zero kink) — w_e+w_i must also stay = w."""
    n = np.arange(mesh.nod2D)[:, None]
    k = np.arange(mesh.nl)[None, :]
    imask = jnp.asarray(np.asarray(mesh.node_iface_mask))
    w0 = jnp.where(imask, jnp.asarray(0.1 * np.cos(0.3 * k + 0.01 * n)), 0.0)
    cfl0 = jnp.where(imask, jnp.asarray(1.5 + 1.0 * np.sin(0.2 * k + 0.013 * n)), 0.0)

    def loss(w, cfl):
        w_e, w_i = ale.compute_wvel_split(mesh, w, cfl, use_wsplit=True, maxcfl=1.0)
        return jnp.sum(w_e) + jnp.sum(w_i * w_i)

    gw, gc = jax.grad(loss, argnums=(0, 1))(w0, cfl0)
    assert np.all(np.isfinite(np.asarray(gw)))
    assert np.all(np.isfinite(np.asarray(gc)))
    assert np.max(np.abs(np.asarray(gc))) > 0              # cfl-dependence is real
