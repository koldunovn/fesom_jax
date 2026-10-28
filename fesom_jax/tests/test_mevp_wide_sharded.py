"""The mEVP wide halo (docs/plans/20261020-core2-mevp-wide-halo.md): K-ring ice lanes, one
exchange every K subcycles, owned lanes BITWISE the per-subcycle sharded mEVP (CPU).

Task 1: the host-built rings and maps (numpy). Task 2: the wide subcycle loop against the
per-subcycle sharded mEVP on CORE2 ``dist_4`` with 4 fake CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``), plus the negative control (a skipped
refresh must change the owned lanes at K = 2) and the gradient comparison.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec

from fesom_jax import ice_wide

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NDEV = len(jax.devices())
_PLATFORM = jax.devices()[0].platform

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_4").is_dir(),
    reason="CORE2 dense mesh or dist_4 partition missing")


# --------------------------------------------------------------------------
# Task 1: rings on a small synthetic mesh
# --------------------------------------------------------------------------
def _grid_mesh(n: int = 12):
    """``n × n`` nodes, two triangles per cell: ``(elem_nodes [E,3], node xy [N,2])``."""
    idx = np.arange(n * n).reshape(n, n)
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            tris += [(a, b, d), (a, d, c)]
    xy = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    return np.asarray(tris), xy


@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_rings_synthetic(K):
    en, xy = _grid_mesh()
    owned = np.nonzero(xy[:, 0] < 4)[0]                     # a strip of rows
    rn, re_, e1, km1 = ice_wide.rings(en, owned, K, nod2D=xy.shape[0])
    in_n = np.zeros(xy.shape[0], bool)
    in_n[rn] = True
    # E_1 = elements with an owned vertex
    own = np.zeros(xy.shape[0], bool)
    own[owned] = True
    np.testing.assert_array_equal(e1, own[en].any(axis=1))
    # every E_K vertex is in R_K; every R_{K-1} node has all its elements in E_K
    assert in_n[en[re_]].all()
    in_e = np.zeros(en.shape[0], bool)
    in_e[re_] = True
    touches = km1[en].any(axis=1)
    assert in_e[touches].all()
    # on a structured grid each ring adds one row: R_K = rows < 4 + K
    np.testing.assert_array_equal(np.sort(rn), np.nonzero(xy[:, 0] < 4 + K)[0])
    assert own[km1].sum() == own.sum() and km1[own].all()          # R_0 ⊂ R_{K-1}


# --------------------------------------------------------------------------
# Task 1: CORE2 dist_4 — the ice lanes against the standard layout
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def core2_dist4():
    from fesom_jax import partit, shard_mesh
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh(CORE2_MESH)
    part = partit.read_partition(CORE2_DIST, 4)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    return mesh, part, sm


@avail
def test_ring1_is_standard_layout(core2_dist4):
    """R_1 = the standard local nodes (owned + halo), E_1 = the standard interior elements."""
    mesh, part, _sm = core2_dist4
    en = np.asarray(mesh.elem_nodes)
    for r in range(part.npes):
        ml_n = np.asarray(part.myList_nod2D[r])
        ml_e = np.asarray(part.myList_elem2D[r])
        rn, re_, e1, _ = ice_wide.rings(en, ml_n[: part.myDim_nod2D[r]], 1, mesh.nod2D)
        np.testing.assert_array_equal(np.sort(rn), np.sort(ml_n))
        np.testing.assert_array_equal(np.nonzero(e1)[0], np.sort(ml_e[: part.myDim_elem2D[r]]))
        np.testing.assert_array_equal(re_, np.nonzero(e1)[0])


@avail
@pytest.mark.parametrize("K", [2, 4, 8])
def test_ice_lanes_core2(core2_dist4, K):
    """Standard lanes are a prefix of the ice lanes; every ice element's vertices are ice nodes;
    every R_{K-1} node has its complete element list in the owner's order; an owned node's list
    on its own device is its standard elements in ascending lane order (the scatter order)."""
    mesh, part, sm = core2_dist4
    host = ice_wide.build_mevp_wide(mesh, part, sm, K)
    en_g = np.asarray(mesh.elem_nodes)
    deg = np.bincount(en_g.ravel(), minlength=mesh.nod2D)
    A = host.arrays
    for r in range(part.npes):
        rg = host.rings[r]
        ml_n = np.asarray(part.myList_nod2D[r])
        ml_e = np.asarray(part.myList_elem2D[r])
        np.testing.assert_array_equal(rg["nod"][: ml_n.size], ml_n)
        np.testing.assert_array_equal(rg["elem"][: ml_e.size], ml_e)
        assert np.all(np.diff(rg["nod"][ml_n.size:]) > 0)
        assert np.all(np.diff(rg["elem"][ml_e.size:]) > 0)
        ne, nn = rg["elem"].size, rg["nod"].size
        # ice elem_nodes = the global vertices, in ice-node lanes
        np.testing.assert_array_equal(rg["nod"][A["en"][r, :ne]], en_g[rg["elem"]])
        csr = A["csr"][r, :nn]
        cnt = (csr >= 0).sum(axis=1)
        km1 = rg["r_km1"]
        np.testing.assert_array_equal(cnt[km1], deg[rg["nod"][km1]])
        # each csr slot points at an element of that node, at the node's vertex position
        valid = csr >= 0
        e_lane, k = csr // 3, csr % 3
        row = np.broadcast_to(np.arange(nn)[:, None], csr.shape)
        np.testing.assert_array_equal(A["en"][r][e_lane[valid], k[valid]], row[valid])
        # owned nodes: standard elements only, ascending lanes (= the scatter_add order)
        md = int(part.myDim_nod2D[r])
        co = csr[:md]
        assert (co[co >= 0] < 3 * ml_e.size).all()
        for j in range(md):
            s = co[j][co[j] >= 0]
            assert np.all(np.diff(s) > 0)


def _apply(field_PL, send, read):
    """numpy emulation of the compact exchange: device e reads ``[field_e ; gathered]``."""
    P = field_PL.shape[0]
    gathered = np.concatenate([field_PL[d][send[d]] for d in range(P)])
    return np.stack([np.concatenate([field_PL[e], gathered])[read[e]] for e in range(P)])


@avail
@pytest.mark.parametrize("K", [2, 8])
def test_maps_core2(core2_dist4, K):
    """Entry (standard → ice lanes), refresh (ice → ice) and exit (the standard exchange) give
    every valid lane the value of its global entity, emulated in numpy with gid-valued fields
    that are WRONG off the source lanes (so a lane that reads the wrong place shows)."""
    mesh, part, sm = core2_dist4
    host = ice_wide.build_mevp_wide(mesh, part, sm, K)
    A = host.arrays
    P = part.npes
    for kind, ml, myd, ice_key, (s, re_, rr), (xs, xr) in (
            ("nod", part.myList_nod2D, part.myDim_nod2D, "nod", ("sn", "ren", "rrn"),
             ("xsn", "xrn")),
            ("elem", part.myList_elem2D, part.myDim_elem2D, "elem", ("se", "ree", "rre"),
             ("xse", "xre"))):
        Lstd = sm.Lmax[kind]
        Lice = A[re_].shape[1]
        std = np.full((P, Lstd), -7.0)
        for d in range(P):
            md = int(myd[d])
            std[d, :md] = np.asarray(ml[d])[:md]                 # only interior lanes are right
        got = _apply(std, A[s], A[re_])
        for d in range(P):
            ids = host.rings[d][ice_key]
            np.testing.assert_array_equal(got[d, : ids.size], ids, err_msg=f"entry {kind} {d}")
        ice = np.full((P, Lice), -7.0)
        for d in range(P):
            md = int(myd[d])
            ice[d, :md] = host.rings[d][ice_key][:md]
        got = _apply(ice, A[s], A[rr])
        for d in range(P):
            ids = host.rings[d][ice_key]
            np.testing.assert_array_equal(got[d, : ids.size], ids, err_msg=f"refresh {kind} {d}")
        got = _apply(std, A[xs], A[xr])
        for d in range(P):
            n = np.asarray(ml[d]).size
            np.testing.assert_array_equal(got[d, :n], np.asarray(ml[d]), err_msg=f"exit {kind}")


def test_build_returns_none_for_k_below_2():
    class _P:
        npes = 4
    assert ice_wide.build_mevp_wide(None, _P(), None, 1) is None
    assert ice_wide.build_mevp_wide(None, _P(), None, 0) is None


# --------------------------------------------------------------------------
# Task 2: the wide mEVP loop vs the per-subcycle sharded mEVP (4 fake CPU devices)
# --------------------------------------------------------------------------
def _inputs(mesh):
    """Smooth, deterministic, active-ice global inputs for mevp_dynamics (no forcing files)."""
    lon, lat = (np.asarray(mesh.geo_coord_nod2D)[:, i] for i in (0, 1))
    if np.abs(lat).max() > 4.0:                               # degrees → radians
        lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    polar = np.clip((np.abs(lat) - np.deg2rad(55.0)) / np.deg2rad(15.0), 0.0, 1.0)
    a_ice = 0.95 * polar * (0.8 + 0.2 * np.cos(3 * lon))
    m_ice = a_ice * (1.0 + 1.5 * polar * (1.0 + 0.5 * np.sin(2 * lon)))
    m_snow = 0.2 * m_ice
    E = mesh.elem2D
    ec = np.asarray(mesh.elem_nodes)
    s0 = 1e3 * np.cos(3 * lon[ec].mean(1)) * (a_ice[ec].mean(1) > 0.01)
    return dict(
        a_ice=a_ice, m_ice=m_ice, m_snow=m_snow,
        u_ice=0.05 * np.sin(2 * lon) * polar, v_ice=0.03 * np.cos(lon) * polar,
        sigma11=s0, sigma12=0.3 * s0, sigma22=-0.5 * s0,
        srfoce_u=0.1 * np.sin(lon) * np.cos(lat), srfoce_v=0.08 * np.cos(2 * lon) * np.cos(lat),
        elevation=0.5 * np.sin(2 * lat) * np.cos(lon),
        stress_ax=0.12 * np.cos(3 * lat) * np.sin(lon), stress_ay=0.07 * np.sin(2 * lon),
    ), E


def _partition(arr, ml, L):
    P = len(ml)
    a = np.asarray(arr)
    out = np.zeros((P, L) + a.shape[1:], a.dtype)
    for d in range(P):
        out[d, : np.asarray(ml[d]).size] = a[np.asarray(ml[d])]
    return out


_ELEM_IN = ("sigma11", "sigma12", "sigma22")
_ORDER = ("a_ice", "m_ice", "m_snow", "u_ice", "v_ice", "sigma11", "sigma12", "sigma22",
          "srfoce_u", "srfoce_v", "elevation", "stress_ax", "stress_ay")


def _runner(mesh, part, sm, cfg, K, skip=-1):
    """``f(inputs_p dict of [P, L] arrays) -> (u, v, s11, s12, s22) [P, L]`` running
    :func:`mevp_dynamics` under ``shard_map``: the per-subcycle compact exchange (``K=None``)
    or the wide halo of width ``K``. Differentiable in the inputs."""
    from fesom_jax import halo, ice_evp
    from fesom_jax import integrate_sharded as ish
    from fesom_jax.ice_mevp import mevp_dynamics
    P = part.npes
    fm, fm_spec = ish.folded_mesh(sm)
    bn = _partition(np.asarray(ice_evp.boundary_node_mask(mesh)), part.myList_nod2D,
                    sm.Lmax["nod"])
    cm = {k: halo.compact_exchange_maps(*sm.exchange[k]) for k in ("nod", "elem")}
    cm = {k: (ish._fold(s), ish._fold(r)) for k, (s, r) in cm.items()}
    host = ice_wide.build_mevp_wide(mesh, part, sm, K) if K else None
    wa = ice_wide.fold_arrays(host) if host is not None else {}
    jm = halo.device_mesh(devices=jax.devices()[:P])
    sp = PartitionSpec("p")

    def body(m, bn_, cm_, wa_, *xs):
        kw = dict(zip(_ORDER, xs))

        def exch(f, kind):
            return halo.halo_exchange_compact(f, cm_[kind][0], cm_[kind][1], "p")
        wide = None
        if wa_:
            wide = ice_wide.local_mevp_wide(wa_, K, "p")
            if skip >= 0:
                import dataclasses
                wide = dataclasses.replace(wide, skip_refresh=skip)
        return mevp_dynamics(cfg, m, boundary_node=bn_, exch=exch, wide=wide, **kw)

    fn = jax.jit(jax.shard_map(
        body, mesh=jm,
        in_specs=(fm_spec, sp, jax.tree.map(lambda _: sp, cm), jax.tree.map(lambda _: sp, wa))
        + (sp,) * len(_ORDER),
        out_specs=(sp,) * 5, check_vma=False))

    def run(inp):
        xs = [ish._fold(inp[k]) for k in _ORDER]
        out = fn(fm, ish._fold(bn), cm, wa, *xs)
        return tuple(o.reshape((P, -1)) for o in out)
    return run


@pytest.fixture(scope="module")
def mevp_setup4():
    if NDEV < 4:
        pytest.skip(f"needs 4 devices, have {NDEV}")
    from fesom_jax import partit, shard_mesh
    from fesom_jax.ice import IceConfig
    from fesom_jax.ice_step import _with_ice_dt
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh(CORE2_MESH)
    part = partit.read_partition(CORE2_DIST, 4)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    g, _ = _inputs(mesh)
    inp = {k: _partition(v, part.myList_elem2D if k in _ELEM_IN else part.myList_nod2D,
                         sm.Lmax["elem" if k in _ELEM_IN else "nod"]) for k, v in g.items()}
    cfg = _with_ice_dt(IceConfig(whichEVP=1), 1800.0)
    ref = _runner(mesh, part, sm, cfg, None)
    out1 = ref(inp)
    inp2 = dict(inp, u_ice=np.asarray(out1[0]), v_ice=np.asarray(out1[1]),
                sigma11=np.asarray(out1[2]), sigma12=np.asarray(out1[3]),
                sigma22=np.asarray(out1[4]))
    out2 = ref(inp2)
    return dict(mesh=mesh, part=part, sm=sm, cfg=cfg, inp=inp, inp2=inp2, ref=(out1, out2))


def _lanes(part, sm):
    """Per device: owned node lanes, valid node lanes, owned elem lanes, and the standard
    element lanes whose three vertices are all local (where the per-subcycle path's σ is the
    owner's value; on the other halo elements it reads a sentinel vertex)."""
    en = np.asarray(sm.fields["elem_nodes"])
    out = []
    for d in range(part.npes):
        n_loc = np.asarray(part.myList_nod2D[d]).size
        e_loc = np.asarray(part.myList_elem2D[d]).size
        full = np.zeros(en.shape[1], bool)
        full[:e_loc] = (en[d, :e_loc] >= 0).all(axis=1)
        out.append((slice(0, int(part.myDim_nod2D[d])), slice(0, n_loc),
                    slice(0, int(part.myDim_elem2D[d])), full))
    return out


def _assert_bitwise(got, ref, part, sm, tag):
    names = ("u_ice", "v_ice", "sigma11", "sigma12", "sigma22")
    for d, (own_n, val_n, own_e, full_e) in enumerate(_lanes(part, sm)):
        for i, nm in enumerate(names):
            g, r = np.asarray(got[i])[d], np.asarray(ref[i])[d]
            if i < 2:
                np.testing.assert_array_equal(g[own_n], r[own_n], err_msg=f"{tag} {nm} owned d{d}")
                np.testing.assert_array_equal(g[val_n], r[val_n], err_msg=f"{tag} {nm} halo d{d}")
            else:
                np.testing.assert_array_equal(g[own_e], r[own_e], err_msg=f"{tag} {nm} owned d{d}")
                np.testing.assert_array_equal(g[full_e], r[full_e],
                                              err_msg=f"{tag} {nm} halo d{d}")


@avail
@pytest.mark.skipif(_PLATFORM != "cpu", reason="bitwise is a CPU property")
@pytest.mark.parametrize("K", [2, 4, 8, 7, 9, 60])
def test_wide_mevp_bitwise(mevp_setup4, K):
    """Wide K == the per-subcycle sharded mEVP BITWISE on owned lanes and on the standard halo
    lanes, over two consecutive ice steps (the second carries the first's σ and velocities).
    K = 7 and 9 do not divide 120 (remainders 1 and 3; a remainder of 1 is moved so that no block
    has a single subcycle); K = 60 has a single refresh window."""
    s = mevp_setup4
    run = _runner(s["mesh"], s["part"], s["sm"], s["cfg"], K)
    out1 = run(s["inp"])
    _assert_bitwise(out1, s["ref"][0], s["part"], s["sm"], f"K={K} call1")
    out2 = run(s["inp2"])
    _assert_bitwise(out2, s["ref"][1], s["part"], s["sm"], f"K={K} call2")
    # the owned velocities are not trivial
    own = _lanes(s["part"], s["sm"])[0][0]
    assert np.abs(np.asarray(out1[0])[0][own]).max() > 1e-3


@avail
def test_wide_mevp_negative_control(mevp_setup4):
    """A deliberately skipped refresh at K = 2 MUST change the owned lanes (a stale outer ring
    reaches the owned lanes after two subcycles; at large K its influence would have decayed)."""
    s = mevp_setup4
    run = _runner(s["mesh"], s["part"], s["sm"], s["cfg"], 2, skip=0)
    out = run(s["inp"])
    ref = s["ref"][0]
    diff = 0.0
    for d, (own_n, _v, _oe, _f) in enumerate(_lanes(s["part"], s["sm"])):
        for i in range(2):
            diff = max(diff, float(np.abs(np.asarray(out[i])[d][own_n]
                                          - np.asarray(ref[i])[d][own_n]).max()))
    assert diff > 0.0, "skipping a refresh did not change the owned lanes: the test is blind"


def _grad_of(run, inp, part, sm, wrt="stress_ax"):
    """d/d(inp[wrt]) of a weighted sum of the OWNED u_ice/v_ice."""
    own = np.zeros((part.npes, sm.Lmax["nod"]))
    for d in range(part.npes):
        own[d, : int(part.myDim_nod2D[d])] = 1.0
    w = own * (1.0 + 0.1 * np.arange(sm.Lmax["nod"])[None, :] % 7)

    def J(x):
        out = run(dict(inp, **{wrt: x}))
        return jnp.sum(w * out[0]) + 0.5 * jnp.sum(w * out[1])
    return np.asarray(jax.grad(J)(jnp.asarray(inp[wrt])))


@avail
@pytest.mark.parametrize("K", [2, 4, 8])
def test_wide_mevp_gradient(mevp_setup4, K):
    """``jax.grad`` of a scalar of the owned u_ice/v_ice through the wide path matches the
    per-subcycle sharded gradient (rel 1e-12 of the gradient's max on owned lanes). The
    per-subcycle path's own reverse-pass floor (its gradient against itself through the
    all_gather transport) is printed for scale."""
    s = mevp_setup4
    part, sm = s["part"], s["sm"]
    ref = _runner(s["mesh"], part, sm, s["cfg"], None)
    wide = _runner(s["mesh"], part, sm, s["cfg"], K)
    g_ref = _grad_of(ref, s["inp"], part, sm)
    g_w = _grad_of(wide, s["inp"], part, sm)
    own = np.zeros_like(g_ref, bool)
    for d in range(part.npes):
        own[d, : int(part.myDim_nod2D[d])] = True
    scale = np.abs(g_ref[own]).max()
    rel = np.abs(g_w[own] - g_ref[own]).max() / scale
    print(f"[grad] K={K} max|g|={scale:.3e} rel diff wide vs per-subcycle = {rel:.3e}")
    assert np.isfinite(rel) and scale > 0
    assert rel < 1e-12, f"wide gradient differs from the per-subcycle one: rel {rel:.3e}"


# --------------------------------------------------------------------------
# Task 3: the wide halo through the sharded forced runner (a few CORE2 steps, real forcing)
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
def test_forced_run_wide_bitwise(monkeypatch):
    """``run_steps_sharded_forced(mevp_wide=K=8)`` on dist_4 with the compact halo == the same run
    without it, BITWISE on every owned lane of every State field after 3 forced mEVP steps."""
    import dataclasses
    from fesom_jax import ice, partit, shard_mesh, ssh, surface_forcing
    from fesom_jax import ice_evp
    from fesom_jax import integrate_sharded as ish
    from fesom_jax.ice import IceConfig
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import phc_initial_state
    from fesom_jax.state import State
    if NDEV < 4:
        pytest.skip(f"needs 4 devices, have {NDEV}")
    monkeypatch.setenv("FESOM_HALO_COMPACT", "1")
    dt, year, n = 1800.0, 1958, 3
    mesh = load_mesh(CORE2_MESH)
    state = phc_initial_state(mesh, IC_DIR)
    sst0 = np.asarray(state.T[:, 0])
    state = ice.seed_ice(state, mesh, sst0)
    cf = surface_forcing.build_surface_forcing(mesh, year, sst_ic=sst0)
    stack = cf.stack(surface_forcing.dates_for_steps(year, dt, n))
    part = partit.read_partition(CORE2_DIST, 4)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    sop = ssh.partition_ssh_operator(ssh.build_ssh_operator(mesh, dt=dt), part)
    bn_p = _partition(np.asarray(ice_evp.boundary_node_mask(mesh)), part.myList_nod2D,
                      sm.Lmax["nod"])
    kw = dict(dt=dt, npes=4, ice_cfg=IceConfig(whichEVP=1), boundary_node_p=bn_p)

    def run(wide):
        return ish.run_steps_sharded_forced(
            sm, shard_mesh.partition_state(state, part), sop,
            np.zeros((4, sm.Lmax["elem"], 2)), shard_mesh.partition_step_forcing(stack, part),
            shard_mesh.partition_forcing_static(cf.static, part), n, mevp_wide=wide, **kw)

    ref = run(None)
    got = run(ice_wide.build_mevp_wide(mesh, part, sm, 8))
    checked = 0
    for fld in dataclasses.fields(State):
        if getattr(got, fld.name) is None:   # unused optional field
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
    u = np.asarray(ref.u_ice)
    assert np.abs(u[0, : int(part.myDim_nod2D[0])]).max() > 1e-4, "ice velocities trivial"
