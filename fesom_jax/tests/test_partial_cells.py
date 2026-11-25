"""Partial bottom cells (FESOM2 ``use_partial_cell``): the geometry builder and its consumers.

The reference is FESOM2 ``oce_ale.F90`` (upstream dbf28153). The literal loops below follow the Fortran
statement by statement, with 1-based level indices, so the vectorized port is checked against the Fortran
control flow rather than against itself.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import partial_cells as pc
from fesom_jax.mesh import load_mesh

jax.config.update("jax_enable_x64", True)

CORE2_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"


@pytest.fixture(scope="module")
def pi_mesh():
    return load_mesh()


@pytest.fixture(scope="module")
def core2_mesh():
    if not CORE2_MESH_DIR.is_dir():
        pytest.skip(f"CORE2 mesh export missing: {CORE2_MESH_DIR}")
    return load_mesh(CORE2_MESH_DIR)


def _meshes(request):
    return request.getfixturevalue(request.param)


# --------------------------------------------------------------------------
# Literal transcriptions of the Fortran loops (1-based levels)
# --------------------------------------------------------------------------
def _fortran_bottom(mesh, thresh=0.0, thers_zbar_lev=5):
    """init_bottom_elem_thickness + init_bottom_node_thickness, loop by loop (oce_ale.F90:258-345, 439-445),
    with the depth read of oce_mesh.F90:617-619."""
    zbar = [float(v) for v in np.asarray(mesh.zbar)]               # zbar(1..nl) -> zbar[k-1]
    Z = [float(v) for v in np.asarray(mesh.Z)]
    nl = len(zbar)
    depth = []
    for x in np.asarray(mesh.depth, np.float64):
        x = float(x)
        if x > 0:
            x = -x
        if x > zbar[thers_zbar_lev - 1]:
            x = zbar[thers_zbar_lev - 1]
        depth.append(x)
    en = np.asarray(mesh.elem_nodes)
    nlev = np.asarray(mesh.nlevels)
    zbar_e_bot = np.zeros(en.shape[0])
    for elem in range(en.shape[0]):
        n1, n2, n3 = (int(v) for v in en[elem])
        dd = (depth[n1] + depth[n2] + depth[n3]) / 3.0
        nle = int(nlev[elem])
        if zbar[nle - 2] - zbar[nle - 1] <= thresh:                 # zbar(nle-1)-zbar(nle) <= thresh
            zbar_e_bot[elem] = zbar[nle - 1]
            continue
        if dd < zbar[nle - 1]:
            if nle == nl:
                zbar_e_bot[elem] = max(dd, zbar[nle - 1] + (zbar[nle - 1] - Z[nle - 2]))
            else:
                zbar_e_bot[elem] = max(Z[nle - 1], dd)              # max(Z(nle), dd)
        else:
            zbar_e_bot[elem] = min(Z[nle - 2], dd)                  # min(Z(nle-1), dd)
    flat = np.asarray(mesh.nod_in_elem2D)
    off = np.asarray(mesh.nod_in_elem2D_offsets)
    zbar_n_bot = np.zeros(off.size - 1)
    for node in range(off.size - 1):
        zbar_n_bot[node] = min(zbar_e_bot[int(e)] for e in flat[off[node]:off[node + 1]])
    return zbar_e_bot, zbar_n_bot


# --------------------------------------------------------------------------
# Task 1: the builder
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["pi_mesh", "core2_mesh"])
def test_apply_matches_fortran_loops(request, name):
    mesh = request.getfixturevalue(name)
    out = pc.apply(mesh)
    ref_e, ref_n = _fortran_bottom(mesh)
    assert np.array_equal(np.asarray(out.zbar_e_bot), ref_e)
    assert np.array_equal(np.asarray(out.zbar_n_bot), ref_n)
    # the partial cells really change the geometry on these meshes
    full = np.asarray(mesh.zbar)[np.asarray(mesh.nlevels) - 1]
    assert np.mean(ref_e != full) > 0.5


@pytest.mark.parametrize("name", ["pi_mesh", "core2_mesh"])
def test_threshold_above_every_layer_gives_full_cells(request, name):
    mesh = request.getfixturevalue(name)
    out = pc.apply(mesh, thresh=1.0e9)
    zbar = np.asarray(mesh.zbar)
    assert np.array_equal(np.asarray(out.zbar_e_bot), zbar[np.asarray(mesh.nlevels) - 1])
    assert np.array_equal(np.asarray(out.zbar_n_bot), zbar[np.asarray(mesh.nlevels_nod2D) - 1])
    assert np.array_equal(np.asarray(out.zbar_3d_n), np.asarray(mesh.zbar_3d_n))


def test_threshold_matches_fortran_loops(core2_mesh):
    out = pc.apply(core2_mesh, thresh=50.0)
    ref_e, ref_n = _fortran_bottom(core2_mesh, thresh=50.0)
    assert np.array_equal(np.asarray(out.zbar_e_bot), ref_e)
    assert np.array_equal(np.asarray(out.zbar_n_bot), ref_n)


def test_branches_by_hand():
    zbar = np.array([0.0, -10.0, -30.0, -60.0])
    Z = 0.5 * (zbar[:-1] + zbar[1:])                                # -5, -20, -45
    nle = np.array([4, 4, 4, 4, 3, 3, 3, 3])
    dd = np.array([-100.0, -65.0, -50.0, -40.0, -100.0, -35.0, -25.0, -5.0])
    got = pc.elem_bottom_depth(dd, nle, zbar, Z)
    # nle == nl: deeper -> max(dd, zbar(nl)+(zbar(nl)-Z(nl-1))) = max(dd, -75); shallower -> min(Z(nl-1), dd)
    # nle <  nl: deeper -> max(Z(nle), dd);                      shallower -> min(Z(nle-1), dd)
    assert got.tolist() == [-75.0, -65.0, -50.0, -45.0, -45.0, -35.0, -25.0, -20.0]
    # threshold: the layer above interface 3 is 20 m thick -> full cell at thresh >= 20
    assert pc.elem_bottom_depth(dd[4:], nle[4:], zbar, Z, thresh=20.0).tolist() == [-30.0] * 4
    assert pc.elem_bottom_depth(dd[4:], nle[4:], zbar, Z, thresh=19.999).tolist() != [-30.0] * 4


def test_depth_clamp():
    zbar = np.array([0.0, -5.0, -10.0, -20.0, -30.0, -40.0])
    got = pc.clamp_depth(np.array([-10.0, 500.0, -3000.0, -30.0, 25.0]), zbar)
    assert got.tolist() == [-30.0, -500.0, -3000.0, -30.0, -30.0]


@pytest.mark.parametrize("name", ["pi_mesh", "core2_mesh"])
def test_bounds(request, name):
    mesh = request.getfixturevalue(name)
    out = pc.apply(mesh)
    zbar, Z = np.asarray(mesh.zbar), np.asarray(mesh.Z)
    nl = zbar.size
    e, n = np.asarray(out.zbar_e_bot), np.asarray(out.zbar_n_bot)
    for bot, nlev in ((e, np.asarray(mesh.nlevels)), (n, np.asarray(mesh.nlevels_nod2D))):
        mid = nlev < nl
        assert np.all(bot[mid] >= Z[nlev[mid] - 1])                 # not deeper than Z(nle)
        assert np.all(bot <= Z[nlev - 2])                           # not shallower than Z(nle-1)
        last = ~mid
        assert np.all(bot[last] >= 2 * zbar[nl - 1] - Z[nl - 2])
    assert np.all(np.asarray(pc.bottom_elem_thickness(out)) > 0)
    assert np.all(np.asarray(pc.bottom_node_thickness(out)) > 0)


@pytest.mark.parametrize("name", ["pi_mesh", "core2_mesh"])
def test_static_node_geometry(request, name):
    mesh = request.getfixturevalue(name)
    out = pc.apply(mesh)
    z3, Z3 = (np.asarray(a) for a in pc.static_node_geometry(out))
    nln = np.asarray(mesh.nlevels_nod2D)
    nl = mesh.nl
    zbar_3d_n = np.asarray(out.zbar_3d_n)
    zbn = np.asarray(out.zbar_n_bot)
    zbar = np.asarray(mesh.zbar)
    for i in range(nln.size):
        k = int(nln[i])
        assert np.array_equal(z3[i, :k], zbar_3d_n[i, :k])         # the init_ale zbar_3d_n on the wet levels
        assert np.all(np.diff(z3[i]) < 0)                           # strictly decreasing, below the bottom too
        assert np.all(Z3[i, :k - 1] < z3[i, :k - 1]) and np.all(Z3[i, :k - 1] > z3[i, 1:k])
        assert Z3[i, k - 2] == zbar[k - 2] + (zbn[i] - zbar[k - 2]) / 2
        assert np.all(np.diff(Z3[i, :k - 1]) < 0)
    assert np.array_equal(Z3[:, -1], Z3[:, -2])                     # pad lane repeats lane nl-2


def test_input_mesh_untouched_and_full_cells_default(pi_mesh):
    assert pi_mesh.zbar_e_bot is None and pi_mesh.zbar_n_bot is None
    z3 = np.asarray(pi_mesh.zbar_3d_n).copy()
    pc.apply(pi_mesh)
    assert np.array_equal(np.asarray(pi_mesh.zbar_3d_n), z3)


def test_refusals(pi_mesh):
    out = pc.apply(pi_mesh)
    with pytest.raises(ValueError, match="already"):
        pc.apply(out)
    cav = dataclasses.replace(pi_mesh, ulevels=jnp.asarray(pi_mesh.ulevels).at[0].set(2))
    with pytest.raises(NotImplementedError, match="cavities"):
        pc.apply(cav)


def test_helpers_trace_and_pad_lanes(pi_mesh):
    out = pc.apply(pi_mesh)
    f = jax.jit(lambda m: (pc.bottom_elem_thickness(m), pc.bottom_node_thickness(m),
                           *pc.static_node_geometry(m)))
    be, bn, z3, Z3 = f(out)
    assert np.array_equal(np.asarray(be), np.asarray(f(out)[0]))
    # a pad lane (level count 0, depth 1.0 as shard_mesh pads floats) stays finite
    pad = dataclasses.replace(out, nlevels_nod2D=jnp.asarray(out.nlevels_nod2D).at[0].set(0),
                              zbar_n_bot=jnp.asarray(out.zbar_n_bot).at[0].set(1.0))
    _, bn_p, z3_p, Z3_p = f(pad)
    assert np.all(np.isfinite(np.asarray(bn_p))) and np.all(np.isfinite(np.asarray(z3_p)))
    assert np.all(np.isfinite(np.asarray(Z3_p)))


# --------------------------------------------------------------------------
# Task 2: thickness init, commit and restart
# --------------------------------------------------------------------------
def _fortran_init_zstar(mesh, hbar):
    """init_thickness_ale, zstar branch (oce_ale.F90:745-813), loop by loop (1-based levels, no cavity)."""
    zbar = [float(v) for v in np.asarray(mesh.zbar)]
    nln = np.asarray(mesh.nlevels_nod2D)
    nmin = np.asarray(mesh.nlevels_nod2D_min)
    zbn = np.asarray(mesh.zbar_n_bot)
    hbar = np.asarray(hbar)
    nl = len(zbar)
    hnode = np.zeros((nln.size, nl))
    for n in range(nln.size):
        nzmax = int(nln[n]) - 1
        mn = int(nmin[n])
        dd = zbar[0] - zbar[mn - 2]                               # zbar(nzmin) - zbar(nlevels_nod2D_min-1)
        for nz in range(1, mn - 1):                               # nz = 1 .. nlevels_nod2D_min-2
            hnode[n, nz - 1] = (zbar[nz - 1] - zbar[nz]) * (1.0 + hbar[n] / dd)
        for nz in range(mn - 1, nzmax):                           # nz = nlevels_nod2D_min-1 .. nzmax-1
            hnode[n, nz - 1] = zbar[nz - 1] - zbar[nz]
        hnode[n, nzmax - 1] = zbar[nzmax - 1] - zbn[n]            # bottom_node_thickness
    en = np.asarray(mesh.elem_nodes)
    nlev = np.asarray(mesh.nlevels)
    zbe = np.asarray(mesh.zbar_e_bot)
    helem = np.zeros((en.shape[0], nl))
    for e in range(en.shape[0]):
        nzmax = int(nlev[e]) - 1
        a, b, c = (int(v) for v in en[e])
        for nz in range(1, nzmax):                                # nz = 1 .. nzmax-1 (nzmin = 1)
            helem[e, nz - 1] = (hnode[a, nz - 1] + hnode[b, nz - 1] + hnode[c, nz - 1]) / 3.0
        helem[e, nzmax - 1] = zbar[nzmax - 1] - zbe[e]            # bottom_elem_thickness
    return hnode, helem


def _fortran_init_linfs(mesh):
    """init_thickness_ale, linfs branch (oce_ale.F90:618-660): hnode from zbar_3d_n, helem nominal, both bottoms
    from the partial cells."""
    zbar = np.asarray(mesh.zbar)
    z3 = np.asarray(mesh.zbar_3d_n)
    nln = np.asarray(mesh.nlevels_nod2D)
    nl = zbar.size
    hnode = np.zeros((nln.size, nl))
    for n in range(nln.size):
        nzmax = int(nln[n]) - 1
        for nz in range(1, nzmax):
            hnode[n, nz - 1] = z3[n, nz - 1] - z3[n, nz]
        hnode[n, nzmax - 1] = zbar[nzmax - 1] - float(np.asarray(mesh.zbar_n_bot)[n])
    nlev = np.asarray(mesh.nlevels)
    zbe = np.asarray(mesh.zbar_e_bot)
    helem = np.zeros((nlev.size, nl))
    for e in range(nlev.size):
        nzmax = int(nlev[e]) - 1
        for nz in range(1, nzmax):
            helem[e, nz - 1] = zbar[nz - 1] - zbar[nz]
        helem[e, nzmax - 1] = zbar[nzmax - 1] - zbe[e]
    return hnode, helem


@pytest.fixture(scope="module")
def pi_pc(pi_mesh):
    return pc.apply(pi_mesh)


def test_init_thickness_zstar_matches_fortran(pi_pc):
    from fesom_jax import ale
    rng = np.random.default_rng(0)
    hbar = jnp.asarray(rng.uniform(-1.5, 1.5, pi_pc.nod2D))
    hnode, helem, _, _ = ale.init_thickness_zstar(pi_pc, hbar, hbar)
    ref_n, ref_e = _fortran_init_zstar(pi_pc, hbar)
    np.testing.assert_allclose(np.asarray(hnode), ref_n, rtol=1e-15, atol=0)
    np.testing.assert_allclose(np.asarray(helem), ref_e, rtol=1e-15, atol=0)
    # the bottom layers are exact, not just close
    rows = np.arange(pi_pc.nod2D)
    k = np.asarray(pi_pc.nlevels_nod2D) - 2
    assert np.array_equal(np.asarray(hnode)[rows, k], ref_n[rows, k])
    rows = np.arange(pi_pc.elem2D)
    k = np.asarray(pi_pc.nlevels) - 2
    assert np.array_equal(np.asarray(helem)[rows, k], ref_e[rows, k])


def test_rest_state_linfs_matches_fortran(pi_pc):
    from fesom_jax.state import State
    ref_n, ref_e = _fortran_init_linfs(pi_pc)
    for xp in (jnp, np):
        st = State.rest(pi_pc, xp=xp)
        assert np.array_equal(np.asarray(st.hnode), ref_n)
        assert np.array_equal(np.asarray(st.helem), ref_e)


def test_cold_start_zstar_equals_linfs_rest(pi_pc):
    """At hbar = 0 the zstar init equals the linfs rest state (the cold start uses the latter for both)."""
    from fesom_jax import ale
    from fesom_jax.state import State
    z = jnp.zeros(pi_pc.nod2D)
    hnode, helem, _, _ = ale.init_thickness_zstar(pi_pc, z, z)
    st = State.rest(pi_pc)
    assert np.array_equal(np.asarray(hnode), np.asarray(st.hnode))
    np.testing.assert_allclose(np.asarray(helem), np.asarray(st.helem), rtol=2e-16, atol=0)


def test_commit_keeps_bottom_helem(pi_pc):
    from fesom_jax import ale
    rng = np.random.default_rng(1)
    hbar = jnp.asarray(rng.uniform(-1.0, 1.0, pi_pc.nod2D))
    hnode, _, _, _ = ale.init_thickness_zstar(pi_pc, hbar, hbar)
    _, helem = ale.commit_thickness(pi_pc, hnode)
    en = np.asarray(pi_pc.elem_nodes)
    hn = np.asarray(hnode)
    mean = (hn[en[:, 0]] + hn[en[:, 1]] + hn[en[:, 2]]) / 3.0
    k = np.asarray(pi_pc.nlevels) - 2
    rows = np.arange(pi_pc.elem2D)
    he = np.asarray(helem)
    assert np.array_equal(he[rows, k], np.asarray(pc.bottom_elem_thickness(pi_pc)))
    above = np.arange(pi_pc.nl)[None, :] < k[:, None]
    # above the bottom: the unchanged vertex mean (XLA's compiled /3 may differ from numpy's by an ulp)
    np.testing.assert_allclose(np.where(above, he, 0.0), np.where(above, mean, 0.0), rtol=1e-15, atol=0)
    # the column of each node adds up to the SSH-displaced partial-cell depth
    col = hn.sum(axis=1)
    np.testing.assert_allclose(col, np.asarray(hbar) - np.asarray(pi_pc.zbar_n_bot), rtol=1e-13)


def test_restart_thickness_reset(pi_mesh, pi_pc):
    """restart_thickness_ale: a full-cell state read into a partial-cell run gets the partial bottoms."""
    from fesom_jax.state import State
    from fesom_jax.zarr_output import restart_thickness_partial_cells as rt
    fc = State.rest(pi_mesh, xp=np)
    pcs = State.rest(pi_pc, xp=np)
    hn = rt("hnode", np.asarray(fc.hnode), pi_pc)
    assert np.array_equal(hn, np.asarray(pcs.hnode))
    he = rt("helem", np.asarray(fc.helem), pi_pc, hn)
    np.testing.assert_allclose(he, np.asarray(pcs.helem), rtol=2e-16, atol=0)
    assert np.array_equal(rt("hnode", hn, pi_pc), hn)                   # idempotent


def test_restart_roundtrip_and_refusal(tmp_path, pi_mesh, pi_pc):
    pytest.importorskip("zarr")
    from fesom_jax import halo, partit, shard_mesh
    from fesom_jax import integrate_sharded as ish
    from fesom_jax import zarr_output as zo
    from fesom_jax.state import State

    def write(mesh, d, attrs):
        g = State.rest(mesh, xp=np)
        part = partit.synth_block_partition(mesh.nod2D, mesh.elem2D, mesh.edge2D, 1)
        sm = shard_mesh.build_sharded_mesh(mesh, part)
        fs, spec = ish.folded_state(shard_mesh.partition_state(g, part))
        placed = ish._to_global_sharded(fs, spec, halo.device_mesh("p", devices=jax.devices()[:1]))
        zo.write_restart(d, placed, sm, part, step=1, calendar_date="1958-01-01", dt_stage=1800.0, attrs=attrs)
        return part

    # full-cell restart -> partial-cell run: the bottoms are reset from the mesh
    part = write(pi_mesh, tmp_path / "fc", {"use_partial_cell": 0})
    st, _ = zo.read_restart(tmp_path / "fc", pi_pc, part)
    want = State.rest(pi_pc, xp=np)
    uf = ish.unfold_state(st, 1)
    assert np.array_equal(np.asarray(uf.hnode)[0, :pi_pc.nod2D], np.asarray(want.hnode))
    np.testing.assert_allclose(np.asarray(uf.helem)[0, :pi_pc.elem2D], np.asarray(want.helem), rtol=2e-16, atol=0)
    # partial-cell restart -> full-cell run: refused
    write(pi_pc, tmp_path / "pc", {"use_partial_cell": 1})
    with pytest.raises(ValueError, match="partial cells"):
        zo.read_restart(tmp_path / "pc", pi_mesh, part)


# --------------------------------------------------------------------------
# Task 3 (+ Task 5 PGF): live geometry and the Shchepetkin PGF
# --------------------------------------------------------------------------
def test_stretch_range_never_reaches_partial_bottom(core2_mesh):
    nmin = np.asarray(core2_mesh.nlevels_nod2D_min)
    nln = np.asarray(core2_mesh.nlevels_nod2D)
    assert np.all(nmin <= nln)                       # => anchor interface min_f-2 < bottom interface nln-1


def test_live_geometry_cold_start_is_static(pi_pc):
    from fesom_jax import ale
    from fesom_jax.state import State
    st = State.rest(pi_pc)
    z3, Z3 = (np.asarray(a) for a in ale.live_geometry(pi_pc, st.hnode))
    s3, S3 = (np.asarray(a) for a in pc.static_node_geometry(pi_pc))
    nln = np.asarray(pi_pc.nlevels_nod2D)
    for i in range(nln.size):
        k = int(nln[i])
        np.testing.assert_allclose(z3[i, :k], s3[i, :k], rtol=1e-15, atol=1e-12)
        np.testing.assert_allclose(Z3[i, :k - 1], S3[i, :k - 1], rtol=1e-15, atol=1e-12)
        assert z3[i, k - 1] == s3[i, k - 1]          # the partial bottom itself is exact


def test_live_geometry_after_ssh_change(pi_pc):
    from fesom_jax import ale
    rng = np.random.default_rng(2)
    hbar = rng.uniform(-1.0, 1.0, pi_pc.nod2D)
    hnode, helem, _, _ = ale.init_thickness_zstar(pi_pc, jnp.asarray(hbar), jnp.asarray(hbar))
    z3, Z3 = (np.asarray(a) for a in ale.live_geometry(pi_pc, hnode))
    nln = np.asarray(pi_pc.nlevels_nod2D)
    rows = np.arange(pi_pc.nod2D)
    assert np.array_equal(z3[rows, nln - 1], np.asarray(pi_pc.zbar_n_bot))      # bottom does not move
    np.testing.assert_allclose(z3[:, 0], hbar, atol=1e-11)                       # surface moves with hbar
    zb = np.asarray(pi_pc.zbar)
    k = nln - 2
    assert np.array_equal(Z3[rows, k], zb[k] + (np.asarray(pi_pc.zbar_n_bot) - zb[k]) / 2)
    zn, Zn = (np.asarray(a) for a in ale.live_geometry_elem(pi_pc, helem))
    nlev = np.asarray(pi_pc.nlevels)
    rows = np.arange(pi_pc.elem2D)
    assert np.array_equal(zn[rows, nlev - 1], np.asarray(pi_pc.zbar_e_bot))      # element anchor
    en = np.asarray(pi_pc.elem_nodes)
    np.testing.assert_allclose(zn[:, 0], hbar[en].mean(axis=1), atol=1e-11)


def _fortran_shchepetkin(mesh, rho, Z3, helem, linfs=False):
    """pressure_force_4_zxxxx_shchepetkin (oce_ale_pressure_bv.F90:2046-2269) or, with linfs,
    pressure_force_4_linfs_shchepetkin (:796-1040), element by element with 1-based levels (no cavity)."""
    from fesom_jax.config import DENSITY_0, G
    g, rho0 = float(G), float(DENSITY_0)
    en = np.asarray(mesh.elem_nodes)
    gs = np.asarray(mesh.gradient_sca)
    nlev, ulev = np.asarray(mesh.nlevels), np.asarray(mesh.ulevels)
    nlev_n, ulev_n = np.asarray(mesh.nlevels_nod2D), np.asarray(mesh.ulevels_nod2D)
    zbe = (np.asarray(mesh.zbar_e_bot) if mesh.zbar_e_bot is not None
           else np.asarray(mesh.zbar)[nlev - 1])
    R, Zd, H = np.asarray(rho), np.asarray(Z3), np.asarray(helem)
    nl = mesh.nl
    px = np.zeros((en.shape[0], nl))
    py = np.zeros((en.shape[0], nl))

    def stencil(n, nlz, lo, Zn):            # drho_dz on Z_3d_n(lo..lo+2) at node n, evaluated at Zn
        z0, z1, z2 = Zd[n, lo - 1], Zd[n, lo], Zd[n, lo + 1]
        f0, f1, f2 = R[n, lo - 1], R[n, lo], R[n, lo + 1]
        dx10, dx21, dx20 = z1 - z0, z2 - z1, z2 - z0
        df10, df21 = f1 - f0, f2 - f1
        return df10 / dx10 + (dx10 * df21 - dx21 * df10) / (dx20 * dx21 * dx10) * ((Zn - z1) + (Zn - z0))

    for e in range(en.shape[0]):
        el = [int(v) for v in en[e]]
        nle, ule = int(nlev[e]) - 1, int(ulev[e])
        he = lambda lv: H[e, lv - 1]
        zbar_n = np.zeros(nl + 2)
        Z_n = np.zeros(nl + 2)
        zbar_n[nle + 1] = zbe[e]
        Z_n[nle] = zbar_n[nle + 1] + he(nle) * 0.5
        for nlz in range(nle, ule, -1):
            zbar_n[nlz] = zbar_n[nlz + 1] + he(nlz)
            Z_n[nlz - 1] = zbar_n[nlz] + he(nlz - 1) * 0.5

        def aux(nlz, dz, comp):
            lo = 0 if comp == 0 else 3
            drho = gs[e, lo] * R[el[0], nlz - 1] + gs[e, lo + 1] * R[el[1], nlz - 1] + gs[e, lo + 2] * R[el[2], nlz - 1]
            if dz is None:
                return drho * he(nlz) * g / rho0
            dzx = gs[e, lo] * Zd[el[0], nlz - 1] + gs[e, lo + 1] * Zd[el[1], nlz - 1] + gs[e, lo + 2] * Zd[el[2], nlz - 1]
            return (drho - (dz[0] + dz[1] + dz[2]) / 3.0 * dzx) * he(nlz) * g / rho0

        # top layer
        nlz = ule
        if linfs and ule == 1:
            dz = None
        else:
            dz = [stencil(n, nlz, nlz, Z_n[nlz]) if nlz - ulev_n[n] == 0 else stencil(n, nlz, nlz - 1, Z_n[nlz])
                  for n in el]
        ix = [aux(nlz, dz, 0), aux(nlz, dz, 1)]
        px[e, nlz - 1], py[e, nlz - 1] = ix[0] * 0.5, ix[1] * 0.5
        # interior layers
        for nlz in range(ule + 1, nle):
            dz = None if linfs else [stencil(n, nlz, nlz - 1, Z_n[nlz]) for n in el]
            a = [aux(nlz, dz, 0), aux(nlz, dz, 1)]
            px[e, nlz - 1], py[e, nlz - 1] = ix[0] + a[0] * 0.5, ix[1] + a[1] * 0.5
            ix = [ix[0] + a[0], ix[1] + a[1]]
        # bottom layer
        nlz = nle
        dz = [stencil(n, nlz, nlz - 2, Z_n[nlz]) if nlev_n[n] - 1 - nlz == 0 else stencil(n, nlz, nlz - 1, Z_n[nlz])
              for n in el]
        a = [aux(nlz, dz, 0), aux(nlz, dz, 1)]
        px[e, nlz - 1], py[e, nlz - 1] = ix[0] + a[0] * 0.5, ix[1] + a[1] * 0.5
    return px, py


def _test_density(mesh, Z3):
    """A smooth stratified density anomaly with horizontal structure."""
    lon, lat = (np.asarray(mesh.coord_nod2D)[:, i] for i in (0, 1))
    z = np.asarray(Z3)
    return (-2.0e-3 * z + 0.3 * np.sin(3 * lon)[:, None] * np.exp(z / 800.0)
            + 0.2 * np.cos(5 * lat)[:, None] * np.exp(z / 2500.0))


@pytest.mark.parametrize("linfs", [False, True])
def test_shchepetkin_matches_fortran(pi_pc, linfs):
    from fesom_jax import ale, pgf
    from fesom_jax.state import State
    if linfs:
        st = State.rest(pi_pc)
        Z3 = pc.static_node_geometry(pi_pc)[1]
        helem = st.helem
    else:
        rng = np.random.default_rng(3)
        hbar = jnp.asarray(rng.uniform(-1.0, 1.0, pi_pc.nod2D))
        hnode, helem, _, _ = ale.init_thickness_zstar(pi_pc, hbar, hbar)
        Z3 = ale.live_geometry(pi_pc, hnode)[1]
    rho = _test_density(pi_pc, Z3)
    px, py = pgf.pressure_force_shchepetkin(pi_pc, jnp.asarray(rho), Z3, helem, linfs=linfs)
    rx, ry = _fortran_shchepetkin(pi_pc, rho, Z3, helem, linfs=linfs)
    lm = np.asarray(pi_pc.elem_layer_mask)
    for got, ref in ((px, rx), (py, ry)):
        got = np.where(lm, np.asarray(got), 0.0)
        scale = np.max(np.abs(ref))
        assert np.max(np.abs(got - ref)) <= 1e-12 * scale, np.max(np.abs(got - ref)) / scale


@pytest.mark.parametrize("linfs", [False, True])
def test_shchepetkin_linear_density_gives_no_force(pi_pc, linfs):
    """A density linear in z is horizontally uniform at every depth: no pressure gradient. The quadratic vertical
    interpolation is exact for it, so the slope term cancels the sloping bottom cells to rounding."""
    from fesom_jax import pgf
    from fesom_jax.config import DENSITY_0, G
    from fesom_jax.state import State
    st = State.rest(pi_pc)
    Z3 = pc.static_node_geometry(pi_pc)[1]
    rho = -2.0e-3 * np.asarray(Z3)
    px, py = pgf.pressure_force_shchepetkin(pi_pc, jnp.asarray(rho), Z3, st.helem, linfs=linfs)
    # scale: the size of one horizontal density-gradient term over the column, before cancellation
    gs = np.abs(np.asarray(pi_pc.gradient_sca))[:, :3].sum(axis=1)
    term = gs[:, None] * np.abs(rho).max() * np.asarray(st.helem) * float(G) / float(DENSITY_0)
    scale = term.sum(axis=1).max()
    for p in (px, py):
        assert np.max(np.abs(np.asarray(p))) <= 1e-11 * scale
    # and the bottom cells do slope: without the slope term there would be a force
    rx, _ = _fortran_shchepetkin(pi_pc, rho, Z3, st.helem, linfs=True)
    assert np.max(np.abs(rx)) <= 1e-11 * scale


def test_shchepetkin_full_cells_unchanged(pi_mesh):
    """linfs=False on a full-cell mesh is the pre-existing kernel (the anchor helper returns zbar[nlevels-1])."""
    from fesom_jax import ale
    m = pi_mesh
    assert np.array_equal(np.asarray(ale.elem_bottom_interface(m)), np.asarray(m.zbar)[np.asarray(m.nlevels) - 1])


# --------------------------------------------------------------------------
# Task 4: SSH operators
# --------------------------------------------------------------------------
def _fortran_stiffness(mesh, dt, alpha=1.0, theta=1.0):
    """init_stiff_mat_ale's edge loop (oce_ale.F90:1341-1390) into a dense matrix, plus the areasvol/dt diagonal."""
    from fesom_jax.config import G
    N = mesh.nod2D
    S = np.zeros((N, N))
    factor = float(G) * dt * alpha * theta
    edges, et = np.asarray(mesh.edges), np.asarray(mesh.edge_tri)
    en, gs, cr = np.asarray(mesh.elem_nodes), np.asarray(mesh.gradient_sca), np.asarray(mesh.edge_cross_dxdy)
    zbar, nlev, ulev = np.asarray(mesh.zbar), np.asarray(mesh.nlevels), np.asarray(mesh.ulevels)
    zbe = np.asarray(mesh.zbar_e_bot) if mesh.zbar_e_bot is not None else zbar[nlev - 1]
    for ed in range(mesh.myDim_edge2D):
        for i in (1, 2):
            el = int(et[ed, i - 1])
            if el < 0:
                continue
            fy = (zbe[el] - zbar[ulev[el] - 1]) * (gs[el, 0:3] * cr[ed, 2 * i - 1] - gs[el, 3:6] * cr[ed, 2 * i - 2])
            if i == 2:
                fy = -fy
            S[edges[ed, 0], en[el]] += fy * factor
            S[edges[ed, 1], en[el]] -= fy * factor
    S[np.arange(N), np.arange(N)] += np.asarray(mesh.areasvol)[:, 0] / dt
    return S


def test_stiffness_matches_fortran(pi_mesh, pi_pc):
    from fesom_jax import ssh
    dt = 1800.0
    for mesh in (pi_mesh, pi_pc):
        op = ssh.build_ssh_operator(mesh, dt=dt)
        got = np.zeros((mesh.nod2D, mesh.nod2D))
        np.add.at(got, (np.asarray(op.rows), np.asarray(op.cols)), np.asarray(op.stiff_vals))
        ref = _fortran_stiffness(mesh, dt)
        assert np.max(np.abs(got - ref)) <= 1e-13 * np.max(np.abs(ref))
        assert np.max(np.abs(got - got.T)) <= 1e-12 * np.max(np.abs(got))
    # the partial cells change the operator
    a = ssh.build_ssh_operator(pi_mesh, dt=dt).stiff_vals
    b = ssh.build_ssh_operator(pi_pc, dt=dt).stiff_vals
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_split_explicit_depth(pi_pc):
    """The split-explicit dynamics take the depth from sum(helem): at rest it is the partial-cell column."""
    from fesom_jax.state import State
    st = State.rest(pi_pc)
    H = np.where(np.asarray(pi_pc.elem_layer_mask), np.asarray(st.helem), 0.0).sum(axis=1)
    np.testing.assert_allclose(H, -np.asarray(pi_pc.zbar_e_bot), rtol=1e-14)


def test_split_explicit_cfl_fallback(pi_pc):
    from fesom_jax import ssh_se
    from fesom_jax.state import State
    st = State.rest(pi_pc)
    w_none, _ = ssh_se.cfl_omega_max(pi_pc)
    w_hel, _ = ssh_se.cfl_omega_max(pi_pc, np.asarray(st.helem))
    np.testing.assert_allclose(w_none, w_hel, rtol=1e-12)


# --------------------------------------------------------------------------
# Task 5: the step with partial cells (pi, unforced)
# --------------------------------------------------------------------------
DT_PI = 1800.0


def _pi_run(mesh, ale_cfg, n=4):
    from fesom_jax import forcing, ic, ssh
    from fesom_jax import step as stepmod
    op = ssh.build_ssh_operator(mesh, dt=DT_PI)
    stress = forcing.surface_stress(mesh)
    st = ic.initial_state(mesh)
    f = jax.jit(lambda s, first: stepmod.step(s, mesh, op, stress, dt=DT_PI, is_first_step=first, ale_cfg=ale_cfg),
                static_argnums=1)
    for i in range(n):
        st = f(st, i == 0)
    return st


@pytest.mark.parametrize("zstar", [False, True])
def test_partial_cell_steps(pi_pc, zstar):
    from fesom_jax.ale import AleConfig
    st = _pi_run(pi_pc, AleConfig() if zstar else None)
    for fld in ("T", "S", "uv", "eta_n", "hnode", "helem", "w", "pgf_x", "pgf_y"):
        assert np.all(np.isfinite(np.asarray(getattr(st, fld)))), fld
    nln = np.asarray(pi_pc.nlevels_nod2D)
    rows = np.arange(pi_pc.nod2D)
    assert np.array_equal(np.asarray(st.hnode)[rows, nln - 2], np.asarray(pc.bottom_node_thickness(pi_pc)))
    nlev = np.asarray(pi_pc.nlevels)
    rows = np.arange(pi_pc.elem2D)
    assert np.array_equal(np.asarray(st.helem)[rows, nlev - 2], np.asarray(pc.bottom_elem_thickness(pi_pc)))
    assert np.max(np.abs(np.asarray(st.uv))) > 0.0                 # it does move


def test_zstar_degenerate_partial_cells_equal_full_cells(pi_mesh):
    """partial_cell_thresh above every layer: the partial-cell code paths on full-cell geometry reproduce the
    full-cell zstar step (same PGF scheme) to rounding."""
    from fesom_jax.ale import AleConfig
    a = _pi_run(pi_mesh, AleConfig(), n=3)
    b = _pi_run(pc.apply(pi_mesh, thresh=1.0e9), AleConfig(), n=3)
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if x is None:
            continue
        x, y = np.asarray(x), np.asarray(y)
        scale = max(np.max(np.abs(x)), 1e-300)
        assert np.max(np.abs(x - y)) <= 1e-12 * scale, (f.name, np.max(np.abs(x - y)) / scale)


def test_linfs_full_cells_keep_hpressure_pgf(pi_mesh):
    """Without partial cells linfs still takes the full-cell hpressure PGF (byte-identical path)."""
    from fesom_jax import pgf
    st = _pi_run(pi_mesh, None, n=1)
    px, _ = pgf.pressure_force_linfs(pi_mesh, st.hpressure)
    # eager vs the jitted step: equal to compilation rounding (the Shchepetkin kernel would differ at O(1))
    ref = np.asarray(st.pgf_x)
    assert np.max(np.abs(np.asarray(px) - ref)) <= 1e-12 * np.max(np.abs(ref))


# --------------------------------------------------------------------------
# Task 6: config, guards, IC
# --------------------------------------------------------------------------
def test_run_config_keys_and_guards():
    from fesom_jax.cavity import CavityConfig
    from fesom_jax.ice import IceConfig
    from fesom_jax.run_config import RunConfig
    c = RunConfig.defaults()
    assert c.use_partial_cell is False and c.partial_cell_thresh == 0.0
    d = RunConfig.from_dict({"use_partial_cell": True, "partial_cell_thresh": 5.0})
    assert d.use_partial_cell and d.partial_cell_thresh == 5.0
    assert RunConfig.from_dict(d.to_dict()) == d
    d.validate()
    with pytest.raises(ValueError, match="partial_cell_thresh"):
        RunConfig(use_partial_cell=True, partial_cell_thresh=-1.0).validate()
    with pytest.raises(TypeError, match="use_partial_cell"):
        RunConfig(use_partial_cell=1).validate()
    with pytest.raises(NotImplementedError, match="cavity"):
        RunConfig(use_partial_cell=True, cavity=CavityConfig(), ice=IceConfig()).validate()
    with pytest.raises(NotImplementedError, match="single"):
        RunConfig(use_partial_cell=True, precision="single").validate()


def test_run_from_config_refuses_mesh_config_mismatch(pi_mesh, pi_pc):
    from fesom_jax.run import run_from_config
    from fesom_jax.run_config import RunConfig
    with pytest.raises(ValueError, match="partial cells"):
        run_from_config(RunConfig(use_partial_cell=True, n_steps=1), mesh=pi_mesh, part=None)
    with pytest.raises(ValueError, match="partial cells"):
        run_from_config(RunConfig(n_steps=1), mesh=pi_pc, part=None)


def test_ic_bottom_layer_per_node():
    """_load_one_variable's bottom option: the bottom layer at each node's own depth, the rest unchanged."""
    from fesom_jax import phc_ic
    rng = np.random.default_rng(4)
    nc_depth = np.array([0.0, 10.0, 30.0, 60.0, 100.0, 150.0])
    nc_lon = np.array([-10.0, 0.0, 10.0, 20.0])          # padded (cyclic halo columns included)
    nc_lat = np.array([-10.0, 0.0, 10.0])
    var = rng.uniform(0.0, 5.0, (nc_depth.size, nc_lat.size, nc_lon.size - 2))
    N = 5
    geo = np.column_stack([rng.uniform(0.5, 9.5, N), rng.uniform(-9.5, 9.5, N)])
    zbar = np.array([0.0, -10.0, -30.0, -60.0, -100.0])
    Z = 0.5 * (zbar[:-1] + zbar[1:])
    nln = np.array([5, 4, 5, 3, 5])
    mask = np.arange(5)[None, :] < (nln[:, None] - 1)
    bi = np.array([phc_ic._binarysearch_d(nc_lon, x) for x in geo[:, 0]])
    bj = np.array([phc_ic._binarysearch_d(nc_lat, y) for y in geo[:, 1]])
    d_indx = np.array([phc_ic._binarysearch_d(nc_depth, -z) for z in Z])
    base = phc_ic._load_one_variable(var, geo, Z, nln, mask, nc_lon, nc_lat, nc_depth, bi, bj, d_indx)
    k_bot = nln - 2
    Z_bot = Z[k_bot] - rng.uniform(-8.0, 8.0, N)
    got = phc_ic._load_one_variable(var, geo, Z, nln, mask, nc_lon, nc_lat, nc_depth, bi, bj, d_indx,
                                    bottom=(k_bot, Z_bot))
    for n in range(N):
        other = np.arange(5) != k_bot[n]
        assert np.array_equal(got[n, other], base[n, other])
        # the same interpolation done with a mesh whose nominal mid-depth IS this node's bottom depth
        Zn = Z.copy()
        Zn[k_bot[n]] = Z_bot[n]
        dn = np.array([phc_ic._binarysearch_d(nc_depth, -z) for z in Zn])
        one = phc_ic._load_one_variable(var, geo[n:n + 1], Zn, nln[n:n + 1], mask[n:n + 1], nc_lon, nc_lat,
                                        nc_depth, bi[n:n + 1], bj[n:n + 1], dn)
        assert got[n, k_bot[n]] == one[0, k_bot[n]]


def test_ic_cache_marker(tmp_path, pi_mesh, pi_pc):
    from fesom_jax import phc_ic
    import json
    phc_ic.check_ic_cache(pi_mesh, tmp_path)                       # full cells, no marker: fine
    with pytest.raises(ValueError, match="full cells"):
        phc_ic.check_ic_cache(pi_pc, tmp_path)
    (tmp_path / phc_ic.PARTIAL_CELL_MARKER).write_text(
        json.dumps({"use_partial_cell": True, "zbar_n_bot_sha256": phc_ic.partial_cell_signature(pi_pc)}))
    phc_ic.check_ic_cache(pi_pc, tmp_path)
    with pytest.raises(ValueError, match="partial cells"):
        phc_ic.check_ic_cache(pi_mesh, tmp_path)
    with pytest.raises(ValueError, match="other partial cells"):
        phc_ic.check_ic_cache(pc.apply(pi_mesh, thresh=1.0e9), tmp_path)   # pi has no layer <= 7 m at a bottom


# --------------------------------------------------------------------------
# Task 7: gradients through partial cells (the masked-NaN probe)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("zstar", [False, True])
def test_gradient_through_partial_cells(pi_pc, zstar):
    """d(mean bottom-layer T after 2 steps)/d(T0): finite on every lane, nonzero on wet layers, exactly 0 below the
    bottom. The partial bottom feeds the PGF stencils, the vertical geometry and the thicknesses of this loss."""
    from fesom_jax import forcing, ic, ssh
    from fesom_jax import step as stepmod
    from fesom_jax.ale import AleConfig
    mesh = pi_pc
    ale_cfg = AleConfig() if zstar else None
    op = ssh.build_ssh_operator(mesh, dt=DT_PI)
    stress = forcing.surface_stress(mesh)
    st0 = ic.initial_state(mesh)
    kb = jnp.asarray(mesh.nlevels_nod2D) - 2
    mlay = np.asarray(mesh.node_layer_mask)

    def loss(T0):
        s = dataclasses.replace(st0, T=T0, T_old=T0)
        s = stepmod.step(s, mesh, op, stress, dt=DT_PI, is_first_step=True, ale_cfg=ale_cfg)
        s = stepmod.step(s, mesh, op, stress, dt=DT_PI, is_first_step=False, ale_cfg=ale_cfg)
        return jnp.mean(jnp.take_along_axis(s.T, kb[:, None], axis=1))

    g = np.asarray(jax.jit(jax.grad(loss))(st0.T))
    assert np.all(np.isfinite(g)), f"{int((~np.isfinite(g)).sum())} non-finite gradient entries"
    assert np.max(np.abs(g[mlay])) > 0.0
    assert np.max(np.abs(g[~mlay])) == 0.0
