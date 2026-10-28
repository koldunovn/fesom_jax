"""Sharded split-explicit SSH (plan docs/plans/20261021-core2-split-explicit.md, Task 5): the SE block under
``shard_map`` on CORE2 ``dist_4`` with 4 fake CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``),
BITWISE the dense block on every owned lane:

* the per-device tables (owned node rows and owned element neighbours = the dense ones, in the same global order);
* the per-substep exchanges (eta^{m+1} and U^{m+1} after every substep, the Kokkos port's two 2-D exchanges);
* the wide halo ``gpu.se_wide: K`` (fesom_jax/se_wide.py): K+1 rings, one refresh every K substeps.
"""

from __future__ import annotations

import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec

from fesom_jax import ssh_se
from fesom_jax.ssh_se import SeConfig

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NDEV = len(jax.devices())
_PLATFORM = jax.devices()[0].platform
DT = 1800.0

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_4").is_dir(),
    reason="CORE2 dense mesh or dist_4 partition missing")

_IN = ("uv", "helem", "eta_n", "hbar", "se_eta", "se_ubt", "se_sbar_old", "du", "wf")
_ELEM_IN = ("uv", "helem", "se_ubt", "se_sbar_old", "du")
_OUT = ("uv", "eta_n", "hbar", "hbar_old", "se_eta", "se_ubt", "se_sbar_old", "umean")
_OUT_ELEM = ("uv", "se_ubt", "se_sbar_old", "umean")


@pytest.fixture(scope="module")
def core2_dist4():
    from fesom_jax import partit, shard_mesh
    from fesom_jax.mesh import load_mesh
    mesh = ssh_se.with_tables(load_mesh(CORE2_MESH))
    part = partit.read_partition(CORE2_DIST, 4)
    sm = shard_mesh.with_se_tables(shard_mesh.build_sharded_mesh(mesh, part), mesh, part)
    return mesh, part, sm


@avail
def test_shard_tables_match_dense(core2_dist4):
    """Owned node rows list the same global elements with the same coefficients in the same order as the dense
    rows; owned elements have the dense neighbours (in global ids) and kappa."""
    mesh, part, sm = core2_dist4
    g = {k: np.asarray(getattr(mesh, k)) for k in ssh_se.TABLE_FIELDS}
    for d in range(part.npes):
        ml_n, ml_e = np.asarray(part.myList_nod2D[d]), np.asarray(part.myList_elem2D[d])
        md_n, md_e = int(part.myDim_nod2D[d]), int(part.myDim_elem2D[d])
        el = sm.fields["se_div_elem"][d, :md_n]
        c = sm.fields["se_div_c"][d, :md_n]
        np.testing.assert_array_equal(c, g["se_div_c"][ml_n[:md_n]])
        live = (c != 0.0).any(axis=-1)
        np.testing.assert_array_equal(ml_e[el][live], g["se_div_elem"][ml_n[:md_n]][live])
        nb = sm.fields["se_elem_nb"][d, :md_e]
        gnb = g["se_elem_nb"][ml_e[:md_e]]
        np.testing.assert_array_equal(np.where(nb >= 0, ml_e[np.maximum(nb, 0)], -1), gnb)
        np.testing.assert_array_equal(sm.fields["se_kappa"][d, :md_e], g["se_kappa"][ml_e[:md_e]])


def _inputs(mesh, seed=0):
    """Smooth-plus-noise global inputs of the SE block (no forcing files): a rest-thickness column, a velocity with
    a baroclinic shear, an increment du, eta and its ring, the U ring near the transport, a water flux."""
    rng = np.random.default_rng(seed)
    lm = np.asarray(mesh.elem_layer_mask)
    zbar = np.asarray(mesh.zbar)
    helem = np.where(lm, np.concatenate([zbar[:-1] - zbar[1:], [0.0]])[None, :], 0.0)
    lon, lat = (np.asarray(mesh.elem_center_x), np.asarray(mesh.elem_center_y))
    base = np.stack([0.2 * np.cos(lat) * np.sin(2 * lon), 0.1 * np.sin(3 * lat)], axis=-1)
    k = np.arange(lm.shape[1])[None, :, None]
    uv = np.where(lm[:, :, None], base[:, None, :] * np.exp(-k / 15.0)
                  + 0.01 * rng.standard_normal(lm.shape + (2,)), 0.0)
    du = np.where(lm[:, :, None], 1e-3 * rng.standard_normal(lm.shape + (2,)), 0.0)
    nlon, nlat = (np.asarray(mesh.coord_nod2D)[:, i] for i in (0, 1))
    eta = 0.5 * np.sin(2 * nlat) * np.cos(nlon) + 0.01 * rng.standard_normal(mesh.nod2D)
    U = np.stack([(np.where(lm, helem * uv[:, :, c], 0.0)).sum(axis=1) for c in (0, 1)], axis=-1)
    se_ubt = np.stack([U, U * (1 + 1e-3 * rng.standard_normal(U.shape)), U * 0.99], axis=1)
    se_eta = np.stack([eta + 1e-3 * rng.standard_normal(eta.shape), eta * 0.999], axis=-1)
    return dict(uv=uv, helem=helem, eta_n=eta, hbar=eta.copy(), se_eta=se_eta, se_ubt=se_ubt,
                se_sbar_old=U * 0.98, du=du, wf=1e-6 * rng.standard_normal(mesh.nod2D))


def _dense(mesh, inp, cfg):
    # the mesh is an ARGUMENT, as in the step's jit: closed over, its arrays become constants and XLA:CPU compiles
    # the column sums differently (measured: 2.8e-14 off the sequential numpy sum, the argument form exact)
    st = {k: jnp.asarray(inp[k]) for k in _IN if k not in ("du", "wf")}
    out = jax.jit(lambda m, s, d, w: ssh_se.se_block(m, types.SimpleNamespace(**s), d, w, cfg, dt=DT,
                                                     is_first_step=False))(
        mesh, st, jnp.asarray(inp["du"]), jnp.asarray(inp["wf"]))
    return {k: np.asarray(getattr(out, k)) for k in _OUT}


def _partition(arr, ml, L):
    a = np.asarray(arr)
    out = np.zeros((len(ml), L) + a.shape[1:], a.dtype)
    for d in range(len(ml)):
        out[d, : np.asarray(ml[d]).size] = a[np.asarray(ml[d])]
    return out


def _sharded(mesh, part, sm, inp, cfg, K=None, skip=-1):
    """The SE block under shard_map: per-substep all_gather exchanges (``K=None``) or the wide halo of width K.
    Returns ``{name: [P, L, ...]}``."""
    from fesom_jax import halo
    from fesom_jax import integrate_sharded as ish
    from fesom_jax import se_wide
    P = part.npes
    fm, fm_spec = ish.folded_mesh(sm)
    ex = {k: (ish._fold(sm.exchange[k][0]).astype(np.int32), ish._fold(sm.exchange[k][1]).astype(np.int32))
          for k in ("nod", "elem")}
    host = se_wide.build_se_wide(mesh, part, sm, K) if K else None
    wa = se_wide.fold_arrays(host) if host is not None else {}
    jm = halo.device_mesh(devices=jax.devices()[:P])
    sp = PartitionSpec("p")

    def body(m, ex_, wa_, *xs):
        kw = dict(zip(_IN, xs))

        def exch(f, kind):
            return halo.halo_exchange(f, ex_[kind][0], ex_[kind][1], "p")
        wide = None
        if wa_:
            wide = se_wide.local_se_wide(wa_, K, "p", skip_refresh=skip)
        st = types.SimpleNamespace(**{k: kw[k] for k in _IN if k not in ("du", "wf")})
        o = ssh_se.se_block(m, st, kw["du"], kw["wf"], cfg, dt=DT, is_first_step=False, exch=exch, wide=wide)
        return tuple(getattr(o, k) for k in _OUT)

    fn = jax.jit(jax.shard_map(
        body, mesh=jm, in_specs=(fm_spec, jax.tree.map(lambda _: sp, ex), jax.tree.map(lambda _: sp, wa))
        + (sp,) * len(_IN), out_specs=(sp,) * len(_OUT), check_vma=False))
    xs = []
    for k in _IN:
        ml, L = ((part.myList_elem2D, sm.Lmax["elem"]) if k in _ELEM_IN else (part.myList_nod2D, sm.Lmax["nod"]))
        xs.append(ish._fold(_partition(inp[k], ml, L)))
    out = fn(fm, ex, wa, *xs)
    return {k: np.asarray(o).reshape((P, -1) + np.asarray(o).shape[1:]) for k, o in zip(_OUT, out)}


@pytest.fixture(scope="module")
def setup4(core2_dist4):
    if NDEV < 4:
        pytest.skip(f"needs 4 devices, have {NDEV}")
    mesh, part, sm = core2_dist4
    return dict(mesh=mesh, part=part, sm=sm, inp=_inputs(mesh))


def _assert_owned_bitwise(got, ref, part, tag):
    for d in range(part.npes):
        ml_n, ml_e = np.asarray(part.myList_nod2D[d]), np.asarray(part.myList_elem2D[d])
        md_n, md_e = int(part.myDim_nod2D[d]), int(part.myDim_elem2D[d])
        for k in _OUT:
            ml, md = (ml_e, md_e) if k in _OUT_ELEM else (ml_n, md_n)
            np.testing.assert_array_equal(got[k][d, :md], ref[k][ml[:md]], err_msg=f"{tag} {k} owned d{d}")


_CFGS = {"two_term": SeConfig(M=50), "drag": SeConfig(M=50, bt_drag=True), "linear": SeConfig(M=50, visc="linear")}


@avail
@pytest.mark.skipif(_PLATFORM != "cpu", reason="bitwise is a CPU property")
@pytest.mark.parametrize("name", list(_CFGS))
def test_sharded_per_substep_bitwise(setup4, name):
    """Per-substep exchanges on 4 devices == the dense SE block BITWISE on every owned lane (eta, U ring, <<U>>,
    the trimmed 3-D velocity)."""
    s = setup4
    cfg = _CFGS[name]
    ref = _dense(s["mesh"], s["inp"], cfg)
    got = _sharded(s["mesh"], s["part"], s["sm"], s["inp"], cfg)
    _assert_owned_bitwise(got, ref, s["part"], name)
    de = np.abs(ref["eta_n"] - s["inp"]["eta_n"]).max()
    print(f"\n[se-sharded {name}] owned lanes bitwise; max|eta^n+1 - eta^n| = {de:.3e} m")
    assert de > 1e-4


# --------------------------------------------------------------------------
# the wide halo (fesom_jax/se_wide.py)
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("K", [2, 4])
def test_se_lanes_core2(core2_dist4, K):
    """Standard lanes are a prefix of the SE lanes; every SE element's vertices are SE nodes; the SE tables on the
    lanes point at the global elements of the dense rows; the maps give every lane its global entity (numpy
    emulation of the compact exchange)."""
    from fesom_jax import se_wide
    mesh, part, sm = core2_dist4
    host = se_wide.build_se_wide(mesh, part, sm, K)
    A = host.arrays
    en_g = np.asarray(mesh.elem_nodes)
    g_el, g_c = np.asarray(mesh.se_div_elem), np.asarray(mesh.se_div_c)
    for r in range(part.npes):
        rg = host.rings[r]
        ml_n, ml_e = np.asarray(part.myList_nod2D[r]), np.asarray(part.myList_elem2D[r])
        np.testing.assert_array_equal(rg["nod"][: ml_n.size], ml_n)
        np.testing.assert_array_equal(rg["elem"][: ml_e.size], ml_e)
        ne, nn = rg["elem"].size, rg["nod"].size
        np.testing.assert_array_equal(rg["nod"][A["en"][r, :ne]], en_g[rg["elem"]])
        live = (A["div_c"][r, :nn] != 0.0).any(axis=-1)
        full = (live.sum(axis=1) == (g_c[rg["nod"]] != 0.0).any(axis=-1).sum(axis=1))
        assert full[: int(part.myDim_nod2D[r])].all()
        np.testing.assert_array_equal(np.where(live, rg["elem"][A["div_elem"][r, :nn]], -1)[full],
                                      np.where((g_c[rg["nod"]] != 0.0).any(axis=-1), g_el[rg["nod"]], -1)[full])

    def apply(field, send, read):
        gathered = np.concatenate([field[d][send[d]] for d in range(part.npes)])
        return np.stack([np.concatenate([field[e], gathered])[read[e]] for e in range(part.npes)])
    for kind, ml, myd, key, (sn, re_, rr) in (
            ("nod", part.myList_nod2D, part.myDim_nod2D, "nod", ("sn", "ren", "rrn")),
            ("elem", part.myList_elem2D, part.myDim_elem2D, "elem", ("se", "ree", "rre"))):
        std = np.full((part.npes, sm.Lmax[kind]), -7.0)
        wide = np.full((part.npes, A[re_].shape[1]), -7.0)
        for d in range(part.npes):
            md = int(myd[d])
            std[d, :md] = np.asarray(ml[d])[:md]
            wide[d, :md] = host.rings[d][key][:md]
        for got, tag in ((apply(std, A[sn], A[re_]), "entry"), (apply(wide, A[sn], A[rr]), "refresh")):
            for d in range(part.npes):
                ids = host.rings[d][key]
                np.testing.assert_array_equal(got[d, : ids.size], ids, err_msg=f"{tag} {kind} {d}")


def _assert_valid_equal(got, ref, part, tag):
    """Equal on every VALID standard lane (owned + halo): after the final exchanges both carry the owners' values."""
    for d in range(part.npes):
        n_n, n_e = np.asarray(part.myList_nod2D[d]).size, np.asarray(part.myList_elem2D[d]).size
        for k in _OUT:
            n = n_e if k in _OUT_ELEM else n_n
            np.testing.assert_array_equal(got[k][d, :n], ref[k][d, :n], err_msg=f"{tag} {k} d{d}")


@pytest.fixture(scope="module")
def ref_per_substep(setup4):
    s = setup4
    return {name: _sharded(s["mesh"], s["part"], s["sm"], s["inp"], cfg) for name, cfg in _CFGS.items()}


@avail
@pytest.mark.skipif(_PLATFORM != "cpu", reason="bitwise is a CPU property")
@pytest.mark.parametrize("K,name", [(2, "two_term"), (3, "two_term"), (5, "two_term"), (7, "two_term"),
                                    (10, "two_term"), (12, "two_term"), (50, "two_term"), (10, "drag"),
                                    (10, "linear")])
def test_se_wide_bitwise(setup4, ref_per_substep, K, name):
    """Wide K == the per-substep sharded SE BITWISE on every valid standard lane (owned and halo), M = 50: K = 3, 7,
    12 leave remainders, K = 50 is one window (no refresh)."""
    s = setup4
    got = _sharded(s["mesh"], s["part"], s["sm"], s["inp"], _CFGS[name], K=K)
    _assert_valid_equal(got, ref_per_substep[name], s["part"], f"K={K} {name}")


@avail
def test_se_wide_negative_control(setup4, ref_per_substep):
    """A skipped refresh at K = 2 MUST change the owned lanes (else the test is blind)."""
    s = setup4
    got = _sharded(s["mesh"], s["part"], s["sm"], s["inp"], _CFGS["two_term"], K=2, skip=0)
    ref = ref_per_substep["two_term"]
    diff = 0.0
    for d in range(s["part"].npes):
        md = int(s["part"].myDim_nod2D[d])
        diff = max(diff, float(np.abs(got["eta_n"][d, :md] - ref["eta_n"][d, :md]).max()))
    assert diff > 0.0, "skipping a refresh did not change the owned lanes: the test is blind"


@avail
@pytest.mark.parametrize("K", [None, 2, 10])
def test_se_wide_gradient(setup4, K):
    """jax.grad of a weighted sum of the owned eta^{n+1} with respect to du, per-substep (K None) and wide, against the
    dense gradient: summed over all lanes holding a copy of an element (an element on several devices is several
    inputs of the sharded program, one of the dense) it matches to rel 1e-12. (Per lane the split between an
    element's copies differs between the two exchange schemes — measured on pi: owned lanes alone 35-70 % off.)"""
    from fesom_jax import halo
    from fesom_jax import integrate_sharded as ish
    from fesom_jax import se_wide
    s = setup4
    mesh, part, sm, inp = s["mesh"], s["part"], s["sm"], s["inp"]
    P = part.npes
    cfg = SeConfig(M=50)              # M >= the CFL guard's M_min (40 at dt 1800): M 20 blows up, the gradient is NaN
    fm, fm_spec = ish.folded_mesh(sm)
    ex = {k: (ish._fold(sm.exchange[k][0]).astype(np.int32), ish._fold(sm.exchange[k][1]).astype(np.int32))
          for k in ("nod", "elem")}
    jm = halo.device_mesh(devices=jax.devices()[:P])
    sp = PartitionSpec("p")
    wglob = 1.0 + 0.1 * (np.arange(mesh.nod2D) % 7)
    own = np.zeros((P, sm.Lmax["nod"]))
    for d in range(P):
        md = int(part.myDim_nod2D[d])
        own[d, :md] = wglob[np.asarray(part.myList_nod2D[d])[:md]]
    wgt = jnp.asarray(ish._fold(own))
    xs = []
    for k in _IN:
        ml, L = ((part.myList_elem2D, sm.Lmax["elem"]) if k in _ELEM_IN else (part.myList_nod2D, sm.Lmax["nod"]))
        xs.append(jnp.asarray(ish._fold(_partition(inp[k], ml, L))))
    i_du = _IN.index("du")
    st = {k: jnp.asarray(inp[k]) for k in _IN if k not in ("du", "wf")}

    def Jd(m, du):
        o = ssh_se.se_block(m, types.SimpleNamespace(**st), du, jnp.asarray(inp["wf"]), cfg, dt=DT,
                            is_first_step=False)
        return jnp.sum(jnp.asarray(wglob) * o.eta_n)
    gd = np.asarray(jax.jit(jax.grad(Jd, argnums=1))(mesh, jnp.asarray(inp["du"])))
    host = se_wide.build_se_wide(mesh, part, sm, K) if K else None
    wa = se_wide.fold_arrays(host) if host is not None else {}

    def body(m, ex_, wa_, *xx):
        kw = dict(zip(_IN, xx))

        def exch(f, kind):
            return halo.halo_exchange(f, ex_[kind][0], ex_[kind][1], "p")
        wide = se_wide.local_se_wide(wa_, K, "p") if wa_ else None
        stl = types.SimpleNamespace(**{k: kw[k] for k in _IN if k not in ("du", "wf")})
        return ssh_se.se_block(m, stl, kw["du"], kw["wf"], cfg, dt=DT, is_first_step=False, exch=exch,
                               wide=wide).eta_n
    fn = jax.shard_map(body, mesh=jm, in_specs=(fm_spec, jax.tree.map(lambda _: sp, ex),
                                                 jax.tree.map(lambda _: sp, wa)) + (sp,) * len(_IN),
                       out_specs=sp, check_vma=False)

    def J(du):
        args = list(xs)
        args[i_du] = du
        return jnp.sum(wgt * fn(fm, ex, wa, *args))
    g = np.asarray(jax.jit(jax.grad(J))(xs[i_du])).reshape((P, sm.Lmax["elem"]) + xs[i_du].shape[1:])
    G = np.zeros_like(gd)
    for d in range(P):
        ml = np.asarray(part.myList_elem2D[d])
        np.add.at(G, ml, g[d, : ml.size])
    scale = np.abs(gd).max()
    rel = np.abs(G - gd).max() / scale
    print(f"\n[se-wide grad] K={K} max|g| {scale:.3e} rel diff to the dense gradient {rel:.3e}")
    assert np.isfinite(rel) and scale > 0 and rel < 1e-12


# --------------------------------------------------------------------------
# through the sharded forced runner (a few CORE2 steps, real forcing)
# --------------------------------------------------------------------------
IC_DIR = ROOT / "data" / "ic_core2"


def _have_jra():
    from fesom_jax import jra55
    return Path(jra55.DEFAULT_JRA_DIR).is_dir()


have_forcing = pytest.mark.skipif(
    not (IC_DIR / "T_ic.npy").exists() or not _have_jra(),
    reason="CORE2 PHC IC or JRA55 forcing missing (compute node only)")


@avail
@have_forcing
@pytest.mark.skipif(_PLATFORM != "cpu", reason="bitwise is a CPU property")
def test_forced_run_se_wide_bitwise():
    """``run_steps_sharded_forced`` with SE (KPP + mEVP + zstar, M 50, dt 1800) on dist_4: the wide halo K = 10 ==
    the per-substep exchange BITWISE on every owned lane of every State field after 3 forced steps (the HaloCtx /
    shard_map plumbing of ``se_wide``); the SE state is non-trivial."""
    import dataclasses
    from fesom_jax import ice, partit, shard_mesh, ssh, surface_forcing
    from fesom_jax import ice_evp
    from fesom_jax import integrate_sharded as ish
    from fesom_jax import se_wide
    from fesom_jax.ale import AleConfig
    from fesom_jax.ice import IceConfig
    from fesom_jax.kpp import KppConfig
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import phc_initial_state
    from fesom_jax.state import State
    if NDEV < 4:
        pytest.skip(f"needs 4 devices, have {NDEV}")
    dt, year, n = 1800.0, 1958, 3
    mesh = load_mesh(CORE2_MESH)
    state = phc_initial_state(mesh, IC_DIR)
    sst0 = np.asarray(state.T[:, 0])
    state = ssh_se.cold_start(ice.seed_ice(state, mesh, sst0), mesh)
    cf = surface_forcing.build_surface_forcing(mesh, year, sst_ic=sst0)
    stack = cf.stack(surface_forcing.dates_for_steps(year, dt, n))
    part = partit.read_partition(CORE2_DIST, 4)
    sm = shard_mesh.with_se_tables(shard_mesh.build_sharded_mesh(mesh, part), mesh, part)
    sop = ssh.partition_ssh_operator(ssh.build_ssh_operator(mesh, dt=dt), part)
    bn_p = _partition(np.asarray(ice_evp.boundary_node_mask(mesh)), part.myList_nod2D, sm.Lmax["nod"])
    ale = AleConfig(split_explicit=SeConfig(M=50))
    kw = dict(dt=dt, npes=4, ice_cfg=IceConfig(whichEVP=1), kpp_cfg=KppConfig(), ale_cfg=ale, boundary_node_p=bn_p)

    def run(wide):
        return ish.run_steps_sharded_forced(
            sm, shard_mesh.partition_state(state, part), sop,
            np.zeros((4, sm.Lmax["elem"], 2)), shard_mesh.partition_step_forcing(stack, part),
            shard_mesh.partition_forcing_static(cf.static, part), n, se_wide=wide, **kw)

    ref = run(None)
    got = run(se_wide.build_se_wide(mesh, part, sm, 10))
    checked = 0
    for fld in dataclasses.fields(State):
        if getattr(got, fld.name) is None:
            continue
        a = np.asarray(getattr(got, fld.name))
        b = np.asarray(getattr(ref, fld.name))
        n0 = np.asarray(getattr(state, fld.name)).shape[0]
        myd = part.myDim_nod2D if n0 == mesh.nod2D else part.myDim_elem2D
        for d in range(4):
            md = int(myd[d])
            np.testing.assert_array_equal(a[d, :md], b[d, :md], err_msg=f"{fld.name} d{d}")
        checked += 1
    assert checked > 10
    eta = np.asarray(ref.eta_n)[0, : int(part.myDim_nod2D[0])]
    ub = np.asarray(ref.se_ubt)[0, : int(part.myDim_elem2D[0])]
    print(f"\n[se-forced] wide K=10 == per-substep on {checked} fields; max|eta| {np.abs(eta).max():.3e} m, "
          f"max|U| {np.abs(ub).max():.3e} m2/s")
    assert np.abs(eta).max() > 1e-2 and np.abs(ub).max() > 1e-1
