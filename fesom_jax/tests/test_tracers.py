"""Tracer advection + vertical diffusion (substep 15) + thickness commit (substep 16).

Two schemes are exercised: the **upwind** low-order kernel (Task 2.10, still the LO part
of FCT) and the **FCT (Zalesak)** scheme (Task 4.1) that is now wired into ``step``.

Upwind (the LO kernel):

* **S vs dump:** ``S=35`` constant ⇒ upwind == FCT == 35 (preserved exactly). Tight gate.
* **T/S vs an independent numpy upwind reference:** the strong upwind-specific gate.
* **T vs dump:** the *upwind*−FCT gap (~3e-7) — bounded, NOT tight (upwind isn't the dump).
* **hnode (substep 16):** ``hnode = hnode_new`` static, bit-for-bit.
* **Property/AD:** constant-tracer preserved; diffusion conserves ``Σ areasvol·hnode·T``;
  advection linear in T (AD == FD), kink-safe in ``uv``; ``d/d(Kv)`` matches FD.

FCT (Task 4.1 — the dump's live scheme, so it matches the dump **tightly**):

* **T/S vs dump (substep 15):** FCT closes the upwind−FCT gap ⇒ **T matches tightly**
  (needs ``T_old`` = the pre-blob base; see ``ic.py``); S bit-for-bit.
* **vs an independent numpy FCT reference:** smooth (limiter inactive) **and** a synthetic
  sharp-tracer + scaled-velocity case that forces the **limiter ACTIVE** — the strong
  check for the Zalesak min/max/sign-select logic.
* **AD (the limiter subgradient, ``docs/LIMITER_GRADIENTS.md``):** finite + FD-consistent
  where smooth, finite through the ``|vflux|`` + min/max kinks.
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


# ==========================================================================
# 7. FCT (Zalesak) advection — Task 4.1 (the dump's live scheme)
# ==========================================================================
# The dump runs FCT, so FCT `T` matches it **tightly** (the Phase-2 upwind−FCT gap is
# closed) — IF `T_old` is the pre-blob base (the C `valuesold`); see ic.py. The dump's
# step-1 blob is smooth ⇒ the limiter is inactive there, so a **synthetic sharp tracer
# + scaled velocity** test (limiter ACTIVE) vs an independent numpy FCT reference is the
# strong check for the limiter logic + the min/max kinks.
_AB_OLD, _AB_NEW = -0.6, 1.6
_FLUX_EPS, _BIG = 1e-16, 1e3


def _fct_ref(mesh, uv, w_e, helem, hnode, hnode_new, T, Told, dt):
    """Independent loop-based numpy port of `fesom_tracer_advect_one_fct`
    (fesom_tracer_adv.c) — the gold-standard cross-check for the JAX FCT."""
    m = mesh
    edges = np.asarray(m.edges); etri = np.asarray(m.edge_tri); eudt = np.asarray(m.edge_up_dn_tri)
    cross = np.asarray(m.edge_cross_dxdy); edxdy = np.asarray(m.edge_dxdy)
    en = np.asarray(m.elem_nodes); gs = np.asarray(m.gradient_sca)
    ule, nle = np.asarray(m.ulevels), np.asarray(m.nlevels)
    uln, nln = np.asarray(m.ulevels_nod2D), np.asarray(m.nlevels_nod2D)
    ulnmax, nlnmin = np.asarray(m.ulevels_nod2D_max), np.asarray(m.nlevels_nod2D_min)
    area, av = np.asarray(m.area), np.asarray(m.areasvol)
    earea, ecos = np.asarray(m.elem_area), np.asarray(m.elem_cos)
    Z, zbar = np.asarray(m.Z), np.asarray(m.zbar)
    off, flat = np.asarray(m.nod_in_elem2D_offsets), np.asarray(m.nod_in_elem2D)
    UV, W, h = np.asarray(uv), np.asarray(w_e), np.asarray(helem)
    hn, hnn = np.asarray(hnode), np.asarray(hnode_new)
    T = np.asarray(T); ttfAB = _AB_OLD*np.asarray(Told) + _AB_NEW*T
    nl, N, E, EL, RE = m.nl, m.nod2D, m.edge2D, m.elem2D, 6367500.0

    def hor_upw(ttf):
        f = np.zeros((E, nl))
        for e in range(E):
            n1, n2 = edges[e]; e1, e2 = etri[e]
            for nz in range(nl):
                vf = 0.0
                if e1 >= 0 and ule[e1]-1 <= nz < nle[e1]-1:
                    u, v = UV[e1, nz]; vf += (u*cross[e,1]-v*cross[e,0])*h[e1,nz]
                if e2 >= 0 and ule[e2]-1 <= nz < nle[e2]-1:
                    u, v = UV[e2, nz]; vf += (v*cross[e,2]-u*cross[e,3])*h[e2,nz]
                if vf != 0.0:
                    f[e,nz] = -0.5*(ttf[n1,nz]*(vf+abs(vf))+ttf[n2,nz]*(vf-abs(vf)))
        return f

    def ver_upw(ttf):
        f = np.zeros((N, nl))
        for n in range(N):
            a, b = uln[n]-1, nln[n]-1
            if b <= a: continue
            f[n,a] = -W[n,a]*ttf[n,a]*area[n,a]
            for nz in range(a+1, b):
                wi = W[n,nz]
                f[n,nz] = -0.5*(ttf[n,nz]*(wi+abs(wi))+ttf[n,nz-1]*(wi-abs(wi)))*area[n,nz]
        return f

    fh, fv = hor_upw(T), ver_upw(T)
    LO = np.zeros((N, nl))
    for e in range(E):
        n1, n2 = edges[e]; e1, e2 = etri[e]
        a = max(nle[e1]-1 if e1>=0 else 0, nle[e2]-1 if e2>=0 else 0)
        for nz in range(a):
            LO[n1,nz] += fh[e,nz]; LO[n2,nz] -= fh[e,nz]
    for n in range(N):
        for nz in range(uln[n]-1, nln[n]-1):
            if hnn[n,nz] <= 0 or av[n,nz] <= 0: LO[n,nz] = 0.0; continue
            LO[n,nz] = (T[n,nz]*hn[n,nz] + (LO[n,nz]+(fv[n,nz]-fv[n,nz+1]))*dt/av[n,nz])/hnn[n,nz]
    # element gradient + up/dn fill (from values T)
    tr = np.zeros((EL, nl, 2))
    for el in range(EL):
        n0, n1, n2 = en[el]; g = gs[el]
        for nz in range(ule[el]-1, nle[el]-1):
            t0, t1, t2 = T[n0,nz], T[n1,nz], T[n2,nz]
            tr[el,nz,0] = g[0]*t0+g[1]*t1+g[2]*t2; tr[el,nz,1] = g[3]*t0+g[4]*t1+g[5]*t2

    def navg(node, z0, z1, g, cx, cy):
        for nz in range(z0, z1):
            tv = tx = ty = 0.0
            for k in range(off[node], off[node+1]):
                el = flat[k]
                if nz >= nle[el]-1 or nz < ule[el]-1: continue
                tv += earea[el]; tx += tr[el,nz,0]*earea[el]; ty += tr[el,nz,1]*earea[el]
            if tv > 0: g[nz,cx] = tx/tv; g[nz,cy] = ty/tv
    eud = np.zeros((E, nl, 4))
    for ed in range(E):
        n1, n2 = edges[ed]; up, dn = eudt[ed]; g = eud[ed]
        if up >= 0 and dn >= 0:
            zmn = max(ulnmax[n1], ulnmax[n2])-1; zmx = min(nlnmin[n1], nlnmin[n2])-1
            navg(n1, uln[n1]-1, zmn, g, 0, 2); navg(n2, uln[n2]-1, zmn, g, 1, 3)
            for nz in range(zmn, zmx):
                g[nz,0] = tr[up,nz,0]; g[nz,1] = tr[dn,nz,0]; g[nz,2] = tr[up,nz,1]; g[nz,3] = tr[dn,nz,1]
            navg(n1, zmx, nln[n1]-1, g, 0, 2); navg(n2, zmx, nln[n2]-1, g, 1, 3)
        else:
            navg(n1, uln[n1]-1, nln[n1]-1, g, 0, 2); navg(n2, uln[n2]-1, nln[n2]-1, g, 1, 3)
    # HO horizontal (MFCT, num_ord=0) → adf_h = HO - LO_upwind
    adfh = fh.copy()
    for e in range(E):
        n1, n2 = edges[e]; e1, e2 = etri[e]
        a = (RE*ecos[e1]) if e1 >= 0 else 0.0
        if e2 >= 0: a = 0.5*(a + RE*ecos[e2])
        edx, edy = edxdy[e]
        for nz in range(nl):
            vf = 0.0
            if e1 >= 0 and ule[e1]-1 <= nz < nle[e1]-1:
                u, v = UV[e1,nz]; vf += (u*cross[e,1]-v*cross[e,0])*h[e1,nz]
            if e2 >= 0 and ule[e2]-1 <= nz < nle[e2]-1:
                u, v = UV[e2,nz]; vf += (v*cross[e,2]-u*cross[e,3])*h[e2,nz]
            if vf == 0.0: continue
            T1, T2 = ttfAB[n1,nz], ttfAB[n2,nz]; diff = T2-T1; g = eud[e,nz]
            Tm1 = T1 + (2*diff+edx*a*g[0]+edy*RE*g[2])/6
            Tm2 = T2 - (2*diff+edx*a*g[1]+edy*RE*g[3])/6
            hi = -0.5*((vf+abs(vf))*Tm1+(vf-abs(vf))*Tm2)
            adfh[e,nz] = hi - adfh[e,nz]
    # HO vertical (QR4C, num_ord=1) → adf_v = HO - LO_upwind
    adfv = fv.copy()
    for n in range(N):
        a, b = uln[n]-1, nln[n]-1
        if b <= a: continue
        adfv[n,a] = -ttfAB[n,a]*W[n,a]*area[n,a] - adfv[n,a]
        if b-a >= 2:
            for nz in (a+1, b-1):
                adfv[n,nz] = -0.5*(ttfAB[n,nz-1]+ttfAB[n,nz])*W[n,nz]*area[n,nz] - adfv[n,nz]
        adfv[n,b] = 0.0 - adfv[n,b]
        for nz in range(a+2, b-1):
            qc = (ttfAB[n,nz-1]-ttfAB[n,nz])/(Z[nz-1]-Z[nz])
            qu = (ttfAB[n,nz]-ttfAB[n,nz+1])/(Z[nz]-Z[nz+1])
            qd = (ttfAB[n,nz-2]-ttfAB[n,nz-1])/(Z[nz-2]-Z[nz-1])
            Tm1 = ttfAB[n,nz]+(2*qc+qu)*(zbar[nz]-Z[nz])/3
            Tm2 = ttfAB[n,nz-1]+(2*qc+qd)*(zbar[nz]-Z[nz-1])/3
            wi = W[n,nz]
            adfv[n,nz] = -0.5*(Tm1+Tm2)*wi*area[n,nz] - adfv[n,nz]
    # Zalesak limiter
    tmax = np.maximum(LO, T); tmin = np.minimum(LO, T)
    auxm = np.full((EL, nl), -_BIG); auxn = np.full((EL, nl), _BIG)
    for el in range(EL):
        for nz in range(ule[el]-1, nle[el]-1):
            vs = en[el]
            auxm[el,nz] = max(tmax[vs[0],nz], tmax[vs[1],nz], tmax[vs[2],nz])
            auxn[el,nz] = min(tmin[vs[0],nz], tmin[vs[1],nz], tmin[vs[2],nz])
    tvm = np.full((N, nl), -_BIG); tvn = np.full((N, nl), _BIG)
    for n in range(N):
        for nz in range(uln[n]-1, nln[n]-1):
            for k in range(off[n], off[n+1]):
                tvm[n,nz] = max(tvm[n,nz], auxm[flat[k],nz]); tvn[n,nz] = min(tvn[n,nz], auxn[flat[k],nz])
    fmax = np.zeros((N, nl)); fmin = np.zeros((N, nl))
    for n in range(N):
        a, b = uln[n]-1, nln[n]-1
        for nz in range(a, b):
            if nz == a or nz == b-1:
                fmax[n,nz] = tvm[n,nz]-LO[n,nz]; fmin[n,nz] = tvn[n,nz]-LO[n,nz]
            else:
                fmax[n,nz] = max(tvm[n,nz-1], tvm[n,nz], tvm[n,nz+1])-LO[n,nz]
                fmin[n,nz] = min(tvn[n,nz-1], tvn[n,nz], tvn[n,nz+1])-LO[n,nz]
    fp = np.zeros((N, nl)); fm = np.zeros((N, nl))
    for n in range(N):
        for nz in range(uln[n]-1, nln[n]-1):
            ft, fb = adfv[n,nz], adfv[n,nz+1]
            fp[n,nz] += max(ft,0)+max(-fb,0); fm[n,nz] += min(ft,0)+min(-fb,0)
    for e in range(E):
        n1, n2 = edges[e]
        for nz in range(nl):
            f = adfh[e,nz]
            fp[n1,nz] += max(f,0); fm[n1,nz] += min(f,0)
            fp[n2,nz] += max(-f,0); fm[n2,nz] += min(-f,0)
    pf = np.ones((N, nl)); mf = np.ones((N, nl))
    for n in range(N):
        for nz in range(uln[n]-1, nln[n]-1):
            a2 = av[n,nz]; hnw = hnn[n,nz]
            if a2 <= 0 or hnw <= 0: continue
            pf[n,nz] = min(1.0, fmax[n,nz]/(fp[n,nz]*dt/a2/hnw + _FLUX_EPS))
            mf[n,nz] = min(1.0, fmin[n,nz]/(fm[n,nz]*dt/a2/hnw - _FLUX_EPS))
    for n in range(N):
        a, b = uln[n]-1, nln[n]-1
        if b <= a: continue
        f = adfv[n,a]; adfv[n,a] = (pf[n,a] if f >= 0 else mf[n,a])*f
        for nz in range(a+1, b):
            f = adfv[n,nz]
            ae = min(mf[n,nz-1], pf[n,nz]) if f >= 0 else min(pf[n,nz-1], mf[n,nz])
            adfv[n,nz] = ae*f
    for e in range(E):
        n1, n2 = edges[e]
        for nz in range(nl):
            f = adfh[e,nz]
            ae = min(pf[n1,nz], mf[n2,nz]) if f >= 0 else min(mf[n1,nz], pf[n2,nz])
            adfh[e,nz] = ae*f
    # flux2dtracer_fct + reconstruct
    dttf = np.zeros((N, nl))
    for n in range(N):
        for nz in range(uln[n]-1, nln[n]-1):
            dttf[n,nz] += -T[n,nz]*hn[n,nz] + LO[n,nz]*hnn[n,nz]
            if av[n,nz] > 0:
                dttf[n,nz] += (adfv[n,nz]-adfv[n,nz+1])*dt/av[n,nz]
    for e in range(E):
        n1, n2 = edges[e]
        for nz in range(nl):
            f = adfh[e,nz]
            if av[n1,nz] > 0: dttf[n1,nz] += f*dt/av[n1,nz]
            if av[n2,nz] > 0: dttf[n2,nz] -= f*dt/av[n2,nz]
    Tn = T.copy()
    for n in range(N):
        for nz in range(uln[n]-1, nln[n]-1):
            dttf[n,nz] += T[n,nz]*(hn[n,nz]-hnn[n,nz])
            if hnn[n,nz] > 0: Tn[n,nz] += dttf[n,nz]/hnn[n,nz]
    return Tn


@pytest.fixture(scope="module")
def fct_chain(mesh):
    """Step-1 chain through substep 16 using **FCT** advection + diffusion."""
    st = ic.initial_state(mesh)
    _, hp, bv = eos.compute_pressure_bv(mesh, st.T, st.S, st.hnode)
    Kv, Av, _ = pp.mixing_pp(mesh, st.uv, bv)
    px, py = pgf.pressure_force_linfs(mesh, hp)
    uvr, _ = momentum.compute_vel_rhs(mesh, st.uv, st.uv_rhsAB, st.eta_n, px, py,
                                      st.w_e, st.hnode, is_first_step=True, dt=DT)
    uvr = momentum.visc_filt_bidiff(mesh, st.uv, uvr, dt=DT)
    du = momentum.impl_vert_visc(mesh, st.uv, uvr, Av, forcing.surface_stress(mesh), dt=DT)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    d_eta = ssh.solve_ssh(op, ssh.compute_ssh_rhs(mesh, st.uv, du, st.helem))
    uv = momentum.update_vel(mesh, st.uv, du, d_eta, dt=DT)
    w = ale.compute_w(mesh, uv, st.helem)
    hnn = ale.thickness_linfs(st.hnode)
    T_adv, _ = tracer_adv.advect_one_fct(mesh, uv, w, st.helem, st.hnode, hnn,
                                         st.T, st.T_old, dt=DT)
    S_adv, _ = tracer_adv.advect_one_fct(mesh, uv, w, st.helem, st.hnode, hnn,
                                         st.S, st.S_old, dt=DT)
    T_dif, S_dif = tracer_diff.impl_vert_diff(mesh, T_adv, S_adv, Kv, hnn, dt=DT)
    return dict(st=st, uv=uv, w=w, Kv=Kv, hnode_new=hnn, T_dif=T_dif, S_dif=S_dif)


@pytest.mark.parametrize("gid", NODE_PROBES)
def test_FCT_T_matches_dump_step1(load_dump, mesh, fct_chain, gid):
    """The headline Task-4.1 gate: FCT `T` (substep 15) matches the FCT dump **tightly**
    (the antidiffusive flux is correct). The Phase-2 upwind−FCT gap is closed."""
    rec = find_record(load_dump("pi_cdump.00000"), step=1, substep=15, field="T",
                      probe_gid=gid)
    verify.assert_close(np.asarray(fct_chain["T_dif"])[gid - 1], rec, kind="scatter")


@pytest.mark.parametrize("gid", NODE_PROBES)
def test_FCT_S_matches_dump_step1(load_dump, mesh, fct_chain, gid):
    """FCT `S=35` (constant) matches the dump bit-for-bit (limiter clips nothing)."""
    rec = find_record(load_dump("pi_cdump.00000"), step=1, substep=15, field="S",
                      probe_gid=gid)
    verify.assert_close(np.asarray(fct_chain["S_dif"])[gid - 1], rec, kind="scatter")


def test_FCT_matches_numpy_reference_step1(mesh, fct_chain):
    """FCT advection (smooth blob, limiter inactive) vs the independent numpy FCT
    reference — bit-for-bit (same algorithm, different code path)."""
    st = fct_chain["st"]
    jx = np.asarray(tracer_adv.advect_one_fct(mesh, fct_chain["uv"], fct_chain["w"],
                    st.helem, st.hnode, fct_chain["hnode_new"], st.T, st.T_old, dt=DT)[0])
    ref = _fct_ref(mesh, fct_chain["uv"], fct_chain["w"], st.helem, st.hnode,
                   fct_chain["hnode_new"], st.T, st.T_old, DT)
    for gid in NODE_PROBES:
        n = int(np.asarray(mesh.nlevels_nod2D)[gid - 1])
        assert np.allclose(jx[gid - 1, :n], ref[gid - 1, :n], rtol=1e-11, atol=1e-11)


def test_FCT_limiter_active_vs_numpy_reference(mesh, fct_chain):
    """⚠️ The strong limiter test: a **synthetic sharp tracer** + **scaled velocity**
    forces the Zalesak limiter ACTIVE (clipping overshoots) — where the dump's smooth
    step-1 leaves it inactive. JAX must match the numpy FCT reference (the min/max/
    sign-select limiter logic) bit-for-bit."""
    st = fct_chain["st"]
    rng = np.random.RandomState(7)
    # a sharp checkerboard-ish tracer (drives large local gradients → overshoots)
    k = np.arange(mesh.nl)[None, :]
    sharp = np.where(np.asarray(mesh.node_layer_mask),
                     10.0 + 5.0 * np.sin(np.arange(mesh.nod2D)[:, None] * 0.7)
                     * np.cos(k * 1.3), 0.0)
    Tin = jnp.asarray(sharp)
    Told = jnp.asarray(np.where(np.asarray(mesh.node_layer_mask), 10.0, 0.0))
    uv_big = fct_chain["uv"] * 5e3        # scale up so |vflux| is large (limiter binds)
    w_big = fct_chain["w"] * 5e3
    jx = np.asarray(tracer_adv.advect_one_fct(mesh, uv_big, w_big, st.helem, st.hnode,
                    fct_chain["hnode_new"], Tin, Told, dt=DT)[0])
    ref = _fct_ref(mesh, uv_big, w_big, st.helem, st.hnode, fct_chain["hnode_new"],
                   Tin, Told, DT)
    m = np.asarray(mesh.node_layer_mask)
    # confirm the limiter actually engaged: FCT ≠ pure LO somewhere, and bounded
    assert np.max(np.abs(jx[m] - ref[m])) < 1e-9, "JAX FCT ≠ numpy FCT (limiter logic)"
    assert np.max(np.abs(jx[m])) > 1.0          # non-trivial result


def test_FCT_constant_tracer_stays_constant(mesh, fct_chain):
    """A constant tracer is preserved by FCT (HO==LO==const, antidiff 0)."""
    m = mesh.node_layer_mask
    Tc = jnp.where(m, 20.0, 0.0)
    out, _ = tracer_adv.advect_one_fct(mesh, fct_chain["uv"], fct_chain["w"],
                                       fct_chain["st"].helem, fct_chain["st"].hnode,
                                       fct_chain["hnode_new"], Tc, Tc, dt=DT)
    assert np.max(np.abs(np.asarray(out)[np.asarray(m)] - 20.0)) < 1e-11


def test_FCT_gradient_finite_and_fd(mesh, fct_chain):
    """The limiter AD (subgradient strategy, docs/LIMITER_GRADIENTS.md): d(Σ wT)/d(T)
    is finite, and FD-consistent where smooth (the smooth blob, limiter inactive)."""
    st = fct_chain["st"]
    wv = jnp.asarray(np.random.RandomState(11).randn(mesh.nod2D, mesh.nl))

    def loss(Tin):
        Tn, _ = tracer_adv.advect_one_fct(mesh, fct_chain["uv"], fct_chain["w"],
                                          st.helem, st.hnode, fct_chain["hnode_new"],
                                          Tin, st.T_old, dt=DT)
        return jnp.sum(wv * Tn)

    g = np.asarray(jax.grad(loss)(st.T))
    assert np.all(np.isfinite(g)) and np.max(np.abs(g)) > 0
    h = 1e-3
    for idx in [(1000, 5), (2000, 10), (1001, 8)]:
        gf = float((loss(st.T.at[idx].add(h)) - loss(st.T.at[idx].add(-h))) / (2 * h))
        assert abs(g[idx] - gf) <= 1e-6 * max(abs(gf), 1.0) + 1e-9


def test_FCT_gradient_finite_in_uv(mesh, fct_chain):
    """Gradient w.r.t. uv is finite despite the FCT |vflux| + limiter min/max kinks
    (the wind-driven uv is at a generic point, off the kinks)."""
    st = fct_chain["st"]
    wv = jnp.asarray(np.random.RandomState(13).randn(mesh.nod2D, mesh.nl))

    def loss(uvin):
        Tn, _ = tracer_adv.advect_one_fct(mesh, uvin, fct_chain["w"], st.helem,
                                          st.hnode, fct_chain["hnode_new"], st.T,
                                          st.T_old, dt=DT)
        return jnp.sum(wv * Tn)

    g = np.asarray(jax.grad(loss)(fct_chain["uv"]))
    assert np.all(np.isfinite(g)) and np.max(np.abs(g)) > 0
