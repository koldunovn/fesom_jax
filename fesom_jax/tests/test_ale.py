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

from fesom_jax import ale, eos, forcing, ic, momentum, pgf, pp, ssh, verify
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
