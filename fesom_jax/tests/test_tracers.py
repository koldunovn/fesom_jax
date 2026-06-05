"""Task 2.10 gate — upwind tracer advection + vertical diffusion (substep 15) +
thickness commit (substep 16).

Verification strategy (the C dump runs **FCT**, this runs **upwind**):

* **S vs dump (substep 15):** ``S=35`` is horizontally *and* vertically constant, so
  upwind == FCT == 35 (a constant tracer is preserved exactly by both advection — the
  transport divergence cancels by discrete continuity — and by diffusion — zero
  gradient). A clean tight gate that exercises the whole advection+diffusion machinery.
* **T/S advection vs an independent numpy upwind reference:** the strong gate for the
  upwind-specific path (the dump can't tightly verify upwind T because of FCT).
* **T vs dump (substep 15):** documents the upwind−FCT gap is small/bounded (~3e-7 at
  step 1) — NOT a tight gate; the full T match is a **Phase-4 (FCT)** gate.
* **hnode vs dump (substep 16):** ``hnode = hnode_new`` static, bit-for-bit.
* **Property tests:** constant-tracer-stays-constant (advection); diffusion conserves
  the volume-weighted integral ``Σ areasvol·hnode·T`` and smooths a vertical gradient.
* **AD:** advection is linear in T (AD == FD), kink-safe in ``uv`` (the ``|vflux|``
  upwind kink); diffusion ``d/d(Kv)`` matches FD; end-to-end ``d(ΣT)/d(du)`` flows.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import (ale, eos, forcing, ic, momentum, pgf, pp, ssh, tracer_adv,
                       tracer_diff, verify)
from fesom_jax.io_dump import find_record
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.state import State

NODE_PROBES = [1001, 1500, 2000, 2500, 3000]
DT = 100.0

# T at step 1 advects with the wind-driven uv; upwind (this port) differs from the
# dump's FCT by the limited antidiffusive flux in the curved T-blob region. Observed
# ~3e-7 on CPU — climate-close but NOT a tight gate (the tight T match is Phase 4).
T_FCT_GAP = 1e-5


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (Task 0.3)")
    return load_mesh()


@pytest.fixture(scope="module")
def chain(mesh):
    """Step-1 chain through substep 16. Returns a dict of the relevant fields."""
    st = ic.initial_state(mesh)
    _, hp, bv = eos.compute_pressure_bv(mesh, st.T, st.S, st.hnode)
    px, py = pgf.pressure_force_linfs(mesh, hp)
    Kv, Av, _ = pp.mixing_pp(mesh, st.uv, bv)
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
    T_adv, _ = tracer_adv.advect_one(mesh, uv, w, st.helem, st.hnode, hnode_new,
                                     st.T, st.T_old, dt=DT)
    S_adv, _ = tracer_adv.advect_one(mesh, uv, w, st.helem, st.hnode, hnode_new,
                                     st.S, st.S_old, dt=DT)
    T_dif, S_dif = tracer_diff.impl_vert_diff(mesh, T_adv, S_adv, Kv, hnode_new, dt=DT)
    hnode, helem = ale.commit_thickness(mesh, hnode_new)
    return dict(st=st, uv=uv, w=w, Kv=Kv, du=du, op=op, hnode_new=hnode_new,
                T_adv=T_adv, S_adv=S_adv, T_dif=T_dif, S_dif=S_dif,
                hnode=hnode, helem=helem)


# --------------------------------------------------------------------------
# Independent loop-based numpy reference of the upwind advection driver.
# --------------------------------------------------------------------------
def _advect_ref(mesh, uv, w_e, helem, hnode, hnode_new, T, Told, dt):
    edges = np.asarray(mesh.edges); etri = np.asarray(mesh.edge_tri)
    cross = np.asarray(mesh.edge_cross_dxdy)
    ule, nle = np.asarray(mesh.ulevels), np.asarray(mesh.nlevels)
    uln, nln = np.asarray(mesh.ulevels_nod2D), np.asarray(mesh.nlevels_nod2D)
    area = np.asarray(mesh.area); areasvol = np.asarray(mesh.areasvol)
    uv = np.asarray(uv); W = np.asarray(w_e); h = np.asarray(helem)
    hn = np.asarray(hnode); hnn = np.asarray(hnode_new)
    T = np.asarray(T).copy(); Told = np.asarray(Told)
    nl = mesh.nl
    ttf = -0.6 * Told + 1.6 * T                              # AB2 (step 1: == T)
    fh = np.zeros((mesh.edge2D, nl))
    for e in range(mesh.myDim_edge2D):
        n1, n2 = edges[e]; el1, el2 = etri[e]
        for nz in range(nl):
            vflux = 0.0
            if el1 >= 0 and ule[el1] - 1 <= nz < nle[el1] - 1:
                u, v = uv[el1, nz]
                vflux += (u * cross[e, 1] - v * cross[e, 0]) * h[el1, nz]
            if el2 >= 0 and ule[el2] - 1 <= nz < nle[el2] - 1:
                u, v = uv[el2, nz]
                vflux += (v * cross[e, 2] - u * cross[e, 3]) * h[el2, nz]
            if vflux != 0.0:
                fh[e, nz] = -0.5 * (ttf[n1, nz] * (vflux + abs(vflux))
                                    + ttf[n2, nz] * (vflux - abs(vflux)))
    fv = np.zeros((mesh.nod2D, nl))
    for n in range(mesh.nod2D):
        nzmin, nzmax = uln[n] - 1, nln[n] - 1
        if nzmax <= nzmin:
            continue
        fv[n, nzmin] = -W[n, nzmin] * ttf[n, nzmin] * area[n, nzmin]
        for nz in range(nzmin + 1, nzmax):
            wi = W[n, nz]
            fv[n, nz] = -0.5 * (ttf[n, nz] * (wi + abs(wi))
                                + ttf[n, nz - 1] * (wi - abs(wi))) * area[n, nz]
    dttf = np.zeros((mesh.nod2D, nl))
    for n in range(mesh.nod2D):
        for nz in range(uln[n] - 1, nln[n] - 1):
            if areasvol[n, nz] > 0:
                dttf[n, nz] += (fv[n, nz] - fv[n, nz + 1]) * dt / areasvol[n, nz]
    for e in range(mesh.myDim_edge2D):
        n1, n2 = edges[e]
        for nz in range(nl):
            f = fh[e, nz]
            if f == 0.0:
                continue
            if areasvol[n1, nz] > 0: dttf[n1, nz] += f * dt / areasvol[n1, nz]
            if areasvol[n2, nz] > 0: dttf[n2, nz] -= f * dt / areasvol[n2, nz]
    for n in range(mesh.nod2D):
        for nz in range(uln[n] - 1, nln[n] - 1):
            dttf[n, nz] += T[n, nz] * (hn[n, nz] - hnn[n, nz])
            if hnn[n, nz] > 0:
                T[n, nz] += dttf[n, nz] / hnn[n, nz]
    return T


# --------------------------------------------------------------------------
# 1. substep 15 — S dump gate (constant tracer ⇒ upwind == FCT == 35, tight)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gid", NODE_PROBES)
def test_S_matches_dump_step1(load_dump, mesh, chain, gid):
    rec = find_record(load_dump("pi_cdump.00000"), step=1, substep=15,
                      field="S", probe_gid=gid)
    verify.assert_close(np.asarray(chain["S_dif"])[gid - 1], rec, kind="scatter")


# --------------------------------------------------------------------------
# 2. T/S advection vs the independent numpy upwind reference (strong gate)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tracer", ["T", "S"])
def test_advection_matches_numpy_reference(mesh, chain, tracer):
    st = chain["st"]
    val, old = (st.T, st.T_old) if tracer == "T" else (st.S, st.S_old)
    ref = _advect_ref(mesh, chain["uv"], chain["w"], st.helem, st.hnode,
                      chain["hnode_new"], val, old, DT)
    jx = np.asarray(chain[f"{tracer}_adv"])
    for gid in NODE_PROBES:
        n = int(np.asarray(mesh.nlevels_nod2D)[gid - 1])
        assert np.allclose(jx[gid - 1, :n], ref[gid - 1, :n], rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# 3. substep 15 — T vs dump: the upwind−FCT gap is small & bounded (Phase-4 gate)
# --------------------------------------------------------------------------
def test_T_vs_dump_is_upwind_fct_gap(load_dump, mesh, chain):
    """T differs from the FCT dump only by the limited antidiffusive flux — small
    and bounded (not a tight gate; the tight T match is the Phase-4 FCT gate)."""
    recs = load_dump("pi_cdump.00000")
    T = np.asarray(chain["T_dif"])
    worst = 0.0
    for gid in NODE_PROBES:
        rec = find_record(recs, step=1, substep=15, field="T", probe_gid=gid)
        n = rec.nlevels
        worst = max(worst, np.abs(T[gid - 1, :n] - np.asarray(rec.values)[:n]).max())
    assert worst < T_FCT_GAP, f"upwind−FCT gap {worst:.2e} exceeds {T_FCT_GAP:.0e}"
    assert worst > 1e-12, "T is suspiciously identical to FCT — advection may be inert"


# --------------------------------------------------------------------------
# 4. substep 16 — hnode dump gate + helem vertex-mean
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gid", NODE_PROBES)
def test_hnode_matches_dump_step1(load_dump, mesh, chain, gid):
    rec = find_record(load_dump("pi_cdump.00000"), step=1, substep=16,
                      field="hnode", probe_gid=gid)
    verify.assert_close(np.asarray(chain["hnode"])[gid - 1], rec, kind="map")


def test_helem_is_vertex_mean(mesh, chain):
    """commit_thickness recomputes helem as the static vertex-mean thickness."""
    assert np.allclose(np.asarray(chain["helem"]),
                       np.asarray(State.rest(mesh).helem), rtol=1e-14, atol=1e-14)


# --------------------------------------------------------------------------
# 5. property tests
# --------------------------------------------------------------------------
def test_constant_tracer_stays_constant(mesh, chain):
    """A horizontally+vertically constant tracer is preserved by advection (the
    transport divergence cancels by discrete continuity between w and the edge
    scatter) — exactly on CPU."""
    m = mesh.node_layer_mask
    Tc = jnp.where(m, 20.0, 0.0)
    out, _ = tracer_adv.advect_one(mesh, chain["uv"], chain["w"], chain["st"].helem,
                                   chain["st"].hnode, chain["hnode_new"], Tc, Tc, dt=DT)
    out = np.asarray(out)[np.asarray(m)]
    assert np.max(np.abs(out - 20.0)) < 1e-11


def test_diffusion_conserves_and_smooths(mesh, chain):
    """Implicit diffusion conserves the volume-weighted integral ``Σ areasvol·hnode·T``
    (no surface/bottom flux in this config), leaves a constant-in-z tracer unchanged,
    and shrinks a vertical gradient."""
    hnn = chain["hnode_new"]
    m = np.asarray(mesh.node_layer_mask)
    Kv_big = jnp.where(mesh.node_iface_mask, 1.0, 0.0)
    # (a) constant-in-z tracer unchanged
    Tc = jnp.where(mesh.node_layer_mask, 20.0, 0.0)
    Tc2 = np.asarray(tracer_diff.impl_vert_diff_one(mesh, Tc, Kv_big, hnn, dt=DT))
    assert np.max(np.abs(Tc2[m] - 20.0)) < 1e-12
    # (b) conservation + smoothing of a vertical gradient
    k = np.arange(mesh.nl)[None, :]
    Tg = jnp.asarray(np.where(m, 10.0 + 0.5 * k, 0.0))
    Tg2 = tracer_diff.impl_vert_diff_one(mesh, Tg, Kv_big, hnn, dt=DT)
    av, hn = np.asarray(mesh.areasvol), np.asarray(hnn)
    i0 = np.sum(av * hn * np.asarray(Tg), axis=1)
    i1 = np.sum(av * hn * np.asarray(Tg2), axis=1)
    assert np.max(np.abs(i1 - i0) / np.maximum(np.abs(i0), 1e-30)) < 1e-12
    rng0 = np.where(m, np.asarray(Tg), -1e30).max(1) - np.where(m, np.asarray(Tg), 1e30).min(1)
    rng1 = np.where(m, np.asarray(Tg2), -1e30).max(1) - np.where(m, np.asarray(Tg2), 1e30).min(1)
    assert rng1.mean() < rng0.mean()        # diffusion smooths


# --------------------------------------------------------------------------
# 6. AD gates
# --------------------------------------------------------------------------
def test_advection_gradient_vs_fd(mesh, chain):
    """Advection is linear in the tracer (fixed uv) → AD == central FD."""
    st = chain["st"]
    wv = jnp.asarray(np.random.RandomState(1).randn(mesh.nod2D, mesh.nl))

    def loss(Tin):
        Tn, _ = tracer_adv.advect_one(mesh, chain["uv"], chain["w"], st.helem,
                                      st.hnode, chain["hnode_new"], Tin, st.T_old, dt=DT)
        return jnp.sum(wv * Tn)

    g = np.asarray(jax.grad(loss)(st.T))
    assert np.all(np.isfinite(g))
    h = 1e-3
    for idx in [(1000, 5), (2000, 10), (500, 3)]:
        gf = float((loss(st.T.at[idx].add(h)) - loss(st.T.at[idx].add(-h))) / (2 * h))
        assert abs(g[idx] - gf) <= 1e-7 * max(abs(gf), 1.0) + 1e-9


def test_advection_gradient_finite_in_uv(mesh, chain):
    """Gradient w.r.t. uv is finite despite the upwind ``|vflux|`` kink (the
    wind-driven uv is away from vflux=0)."""
    st = chain["st"]
    wv = jnp.asarray(np.random.RandomState(3).randn(mesh.nod2D, mesh.nl))

    def loss(uvin):
        Tn, _ = tracer_adv.advect_one(mesh, uvin, chain["w"], st.helem, st.hnode,
                                      chain["hnode_new"], st.T, st.T_old, dt=DT)
        return jnp.sum(wv * Tn)

    g = np.asarray(jax.grad(loss)(chain["uv"]))
    assert np.all(np.isfinite(g)) and np.max(np.abs(g)) > 0


def test_diffusion_gradient_vs_fd(mesh, chain):
    """Diffusion ``d/d(Kv)`` (nonlinear via the matrix) matches central FD where the
    gradient is resolvable; the whole gradient is finite (the dZ padding is safe)."""
    st = chain["st"]
    wv = jnp.asarray(np.random.RandomState(2).randn(mesh.nod2D, mesh.nl))

    def loss(Kin):
        return jnp.sum(wv * tracer_diff.impl_vert_diff_one(mesh, st.T, Kin,
                                                           chain["hnode_new"], dt=DT))

    g = np.asarray(jax.grad(loss)(chain["Kv"]))
    assert np.all(np.isfinite(g))
    h = 1e-6
    for idx in [(1000, 3), (1001, 8)]:    # substantial-gradient interior interfaces
        gf = float((loss(chain["Kv"].at[idx].add(h))
                    - loss(chain["Kv"].at[idx].add(-h))) / (2 * h))
        assert abs(g[idx] - gf) <= 1e-3 * max(abs(gf), 1e-6)


def test_grad_flows_du_to_tracers(mesh, chain):
    """End-to-end: d(Σ T)/d(du) flows through compute_ssh_rhs → custom_linear_solve →
    update_vel → compute_w → advection → diffusion. Finite and nonzero."""
    st, op = chain["st"], chain["op"]

    def loss(du_in):
        rhs = ssh.compute_ssh_rhs(mesh, st.uv, du_in, st.helem)
        d_eta = ssh.solve_ssh(op, rhs)
        uv = momentum.update_vel(mesh, st.uv, du_in, d_eta, dt=DT)
        w = ale.compute_w(mesh, uv, st.helem)
        T_adv, _ = tracer_adv.advect_one(mesh, uv, w, st.helem, st.hnode,
                                         chain["hnode_new"], st.T, st.T_old, dt=DT)
        T_dif = tracer_diff.impl_vert_diff_one(mesh, T_adv, chain["Kv"],
                                               chain["hnode_new"], dt=DT)
        return jnp.sum(T_dif)

    g = np.asarray(jax.grad(loss)(chain["du"]))
    assert np.all(np.isfinite(g)) and np.max(np.abs(g)) > 0
