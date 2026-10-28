"""REcoM plan Task 1: the BGC tracer stage (:mod:`fesom_jax.recom.transport`) and the optional ``State.bgc`` fields.

The BGC tracers run through FESOM2's tracer loop with the transport fields T and S use. With the salinity surface
BC, a stack of salinity copies must therefore reproduce ``S`` after every step: the identity test of the whole
stage (FCT advection with w_split and the GM bolus velocity, Redi + K33, implicit vertical diffusion under zstar,
the halo refreshes on the sharded path). The physics must not see the BGC tracers at all. Pi mesh, JRA55-do
forcing, a synthetic T/S cold start (pi has no cached PHC IC).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ice, partit, shard_mesh, ssh, surface_forcing
from fesom_jax import integrate_sharded as ish
from fesom_jax import step as stepmod
from fesom_jax.ale import AleConfig
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.recom import transport
from fesom_jax.recom.config import RecomConfig
from fesom_jax.run_config import RunConfig
from fesom_jax.state import BGC_FIELDS, BGC_MR_FIELDS, State
from fesom_jax.tke import TkeConfig

DT = 2700.0          # FESOM2's test_pi_recom: step_per_day 32
YEAR = 1958
NSTEPS = 3
NSTACK = 4           # forcing steps in the fixture (the every-k tests run two intervals of 2)
NTR = 3
NDEV = len(jax.devices())
PI_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/pi")     # FESOM2's own pi partitions
PHYS = dict(ice_cfg=IceConfig(whichEVP=1), tke_cfg=TkeConfig(), gm_cfg=GMConfig(),
            ale_cfg=AleConfig(use_wsplit=True))
SALT = RecomConfig(passive_tracers=NTR, passive_bc="salt")
# The batched pass sums the FCT/Redi edge contributions in another order than the one-tracer program on a few
# points: 10 of 158400 values on pi dist_4 after 3 steps, at most 1.2e-15 relative (~5 ulp). Passive tracers do not
# feed back, so it stays round-off.
VMAP_RTOL = 1e-14


def _have_jra():
    from fesom_jax import jra55
    return Path(jra55.DEFAULT_JRA_DIR).is_dir()


have_forcing = pytest.mark.skipif(not _have_jra(), reason="JRA55-do forcing not reachable")


def _synthetic_ic(mesh):
    """A rest zstar state with T/S fronts (latitude, longitude and depth structure) and the seeded sea ice."""
    lon, lat = (np.asarray(mesh.geo_coord_nod2D[:, i]) for i in (0, 1))
    Z = np.asarray(mesh.Z)                                          # (nl-1,) layer mid-depths
    z = np.concatenate([Z, Z[-1:]])[None, :]                        # (1, nl), the tracer level count
    decay = np.exp(z / 800.0)
    T = (np.clip(28.0 * np.cos(lat)[:, None] - 4.0, -1.8, 29.0) * decay + 2.0 * (1.0 - decay))
    S = 34.6 + 0.8 * (np.cos(lat) * np.sin(3.0 * lon))[:, None] * decay + 0.2 * (1.0 - decay)
    st = State.rest(mesh, ale_cfg=PHYS["ale_cfg"])
    T, S = jnp.asarray(T), jnp.asarray(S)
    st = dataclasses.replace(st, T=T, S=S, T_old=T, S_old=S)
    return ice.seed_ice(st, mesh, st.T[:, 0])


@pytest.fixture(scope="module")
def pi_forced():
    mesh = load_mesh()
    st = _synthetic_ic(mesh)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(st.T[:, 0]))
    stack = cf.stack(surface_forcing.dates_for_steps(YEAR, DT, NSTACK))
    op = ssh.build_ssh_operator(mesh, dt=DT)
    return dict(mesh=mesh, state=st, stack=stack, fs=cf.static, op=op)


def _dense_run(fx, recom_cfg, state=None, n=NSTEPS):
    st = fx["state"] if state is None else state
    zero = jnp.zeros((fx["mesh"].elem2D, 2))
    for i in range(n):
        sf = jax.tree.map(lambda x: x[i], fx["stack"])
        st = stepmod.step_jit(st, fx["mesh"], fx["op"], zero, dt=DT, is_first_step=(i == 0),
                              step_forcing=sf, forcing_static=fx["fs"], recom_cfg=recom_cfg, **PHYS)
    return st


# ---------------------------------------------------------------------------------------------------------------
def test_config_validation():
    with pytest.raises(NotImplementedError):
        RecomConfig().validate()                      # sources/sinks not ported: passive mode only
    with pytest.raises(ValueError):
        RecomConfig(passive_tracers=2, passive_bc="nope").validate()
    cfg = RunConfig.from_dict({"recom": {"passive_tracers": 31}})
    assert cfg.recom == RecomConfig(passive_tracers=31)
    assert cfg.physics_kwargs()["recom_cfg"] is cfg.recom
    assert "recom_cfg" not in RunConfig().physics_kwargs()          # off: the runner kwargs are unchanged
    assert RunConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(KeyError):
        RunConfig.from_dict({"recom": {"passive_tracer": 3}})


def test_init_passive_layout():
    mesh = load_mesh()
    st = State.rest(mesh)
    assert all(getattr(st, f) is None for f in BGC_FIELDS)
    for xp_state in (st, jax.tree.map(np.asarray, st)):
        b = transport.init_passive(xp_state, NTR)
        assert b.bgc.shape == (mesh.nod2D, mesh.nl, NTR) and b.bgc_old.shape == b.bgc.shape
        assert isinstance(b.bgc, type(xp_state.S))
        for i in range(NTR):
            np.testing.assert_array_equal(np.asarray(b.bgc[..., i]), np.asarray(xp_state.S))


def test_layout_switch(monkeypatch):
    monkeypatch.delenv("FESOM_BGC_LAYOUT", raising=False)
    assert transport.layout() == "major"
    monkeypatch.setenv("FESOM_BGC_LAYOUT", "loop")
    assert transport.layout() == "loop"
    monkeypatch.setenv("FESOM_BGC_LAYOUT", "minor")
    with pytest.raises(ValueError):
        transport.layout()


def test_step_without_bgc_refuses():
    mesh = load_mesh()
    st = State.rest(mesh)
    with pytest.raises(ValueError, match="no BGC tracers"):
        stepmod.step_jit(st, mesh, ssh.build_ssh_operator(mesh, dt=DT), jnp.zeros((mesh.elem2D, 2)), dt=DT,
                         recom_cfg=RecomConfig(passive_tracers=1))


# ---------------------------------------------------------------------------------------------------------------
@have_forcing
@pytest.mark.parametrize("layout", ["loop", "vmap"])
def test_salt_copies_reproduce_salinity(pi_forced, layout, monkeypatch):
    """Copies of S with S's surface BC equal S after every step: bitwise one tracer at a time; the batched pass
    within ``VMAP_RTOL``."""
    monkeypatch.setenv("FESOM_BGC_LAYOUT", layout)
    st = transport.init_passive(pi_forced["state"], NTR)
    out = _dense_run(pi_forced, SALT, state=st)
    S = np.asarray(out.S)
    for i in range(NTR):
        got = np.asarray(out.bgc[..., i])
        if layout == "loop":
            np.testing.assert_array_equal(got, S, err_msg=f"tracer {i}")
        else:
            np.testing.assert_allclose(got, S, rtol=VMAP_RTOL, atol=0, err_msg=f"tracer {i}")
        np.testing.assert_array_equal(np.asarray(out.bgc_old[..., i]), np.asarray(out.S_old))


@have_forcing
def test_physics_does_not_see_bgc(pi_forced):
    """The physics with the BGC stage on is bitwise the physics without it."""
    base = _dense_run(pi_forced, None)
    with_bgc = _dense_run(pi_forced, RecomConfig(passive_tracers=NTR),
                          state=transport.init_passive(pi_forced["state"], NTR))
    for f in dataclasses.fields(State):
        a = getattr(base, f.name)
        if a is None:
            continue
        np.testing.assert_array_equal(np.asarray(getattr(with_bgc, f.name)), np.asarray(a), err_msg=f.name)
    assert np.all(np.isfinite(np.asarray(with_bgc.bgc)))


def _sharded_run(fx, st0, cfg, n, padded=False):
    """``n`` forced steps of ``st0`` on FESOM2's pi ``dist_4`` (4 host devices); returns ``([P, Lmax, ...] State,
    partition)``."""
    mesh = fx["mesh"]
    npes = 4
    part = partit.read_partition(PI_DIST, npes)
    assert part.nod2D == mesh.nod2D
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(jax.tree.map(np.asarray, st0), part)
    sop = ssh.partition_ssh_operator(fx["op"], part)
    seq_p = shard_mesh.partition_step_forcing(jax.tree.map(lambda x: np.asarray(x)[:n], fx["stack"]), part)
    fs_p = shard_mesh.partition_forcing_static(fx["fs"], part)
    stress_p = jnp.zeros((npes, sm.Lmax["elem"], 2))
    out = ish.run_steps_sharded_forced(sm, state_p, sop, stress_p, seq_p, fs_p, n, dt=DT, npes=npes,
                                       use_padded=padded, recom_cfg=cfg, **PHYS)
    return out, part


@have_forcing
@pytest.mark.skipif(NDEV < 4 or not (PI_DIST / "dist_4").is_dir(),
                    reason="needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4) and pi dist_4")
@pytest.mark.parametrize("layout", ["loop", "vmap"])
@pytest.mark.parametrize("padded", [False, True], ids=["all_gather", "padded"])
def test_sharded_salt_copies(pi_forced, layout, padded, monkeypatch):
    """On the sharded forced path (halo refreshes inside the FCT, batched collectives under vmap; the default
    all_gather halo and the padded all_to_all of the GPU stack) the salinity copies equal the run's own S on every
    lane."""
    monkeypatch.setenv("FESOM_BGC_LAYOUT", layout)
    fx = pi_forced
    mesh = fx["mesh"]
    st0 = transport.init_passive(jax.tree.map(np.asarray, fx["state"]), NTR)
    out, _ = _sharded_run(fx, st0, SALT, NSTEPS, padded=padded)
    S = np.asarray(out.S)
    assert np.asarray(out.bgc).shape == S.shape + (NTR,)
    for i in range(NTR):
        got = np.asarray(out.bgc[..., i])
        if layout == "loop":
            np.testing.assert_array_equal(got, S, err_msg=f"tracer {i}")
        else:
            np.testing.assert_allclose(got, S, rtol=VMAP_RTOL, atol=0, err_msg=f"tracer {i}")


def test_restart_roundtrip_bgc(tmp_path):
    """BGC fields go through partition -> fold -> write_restart -> read_restart unchanged; a store without them reads
    as None (a physics-only restart into a REcoM run, which then initialises them)."""
    from fesom_jax import halo, zarr_output as zo
    import zarr
    mesh = load_mesh()
    g0 = transport.init_every_k(transport.init_passive(jax.tree.map(np.asarray, State.rest(mesh)), NTR), 2)
    rng = np.random.default_rng(0)
    g0 = dataclasses.replace(g0, bgc=g0.bgc + rng.standard_normal(g0.bgc.shape),
                             bgc_steps=g0.bgc_steps + 1, bgc_kv=g0.bgc_kv + rng.random(g0.bgc_kv.shape))
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, ser)
    for name, g in (("with", g0), ("without", dataclasses.replace(g0, **{f: None for f in BGC_FIELDS}))):
        fs, spec = ish.folded_state(shard_mesh.partition_state(g, ser))
        placed = ish._to_global_sharded(fs, spec, halo.device_mesh("p", devices=jax.devices()[:1]))
        d = tmp_path / name
        zo.write_restart(d, placed, sm, ser, step=3, calendar_date="1958-01-01", dt_stage=DT)
        keys = set(zarr.open_group(str(d), mode="r").array_keys())
        loaded, _ = zo.read_restart(d, mesh, ser)
        if name == "with":
            present = [f for f in BGC_FIELDS if getattr(g0, f) is not None]     # the passive mode's BGC fields
            assert set(present) <= keys
            for f in present:
                want = getattr(g0, f)
                np.testing.assert_array_equal(np.asarray(getattr(loaded, f))[: want.shape[0]], want, err_msg=f)
        else:
            assert not set(BGC_FIELDS) & keys
            assert all(getattr(loaded, f) is None for f in BGC_FIELDS)


# ---------------------------------------------------------------------------------------------------------------
# Task 2b: the BGC tracers every k-th physics step (RecomConfig.bgc_every) on the k-step mean transport
def _inventory(mesh, st, X):
    """sum over wet layers of X * hnode * areasvol (the FESOM2 tracer content)."""
    w = np.where(np.asarray(mesh.node_layer_mask), np.asarray(st.hnode) * np.asarray(mesh.areasvol), 0.0)
    return np.einsum("nk,nk...->...", w, np.asarray(X))


def test_every_k_init_and_config():
    mesh = load_mesh()
    st = transport.init_passive(State.rest(mesh), NTR)
    assert transport.init_every_k(st, 1) is not st and transport.init_every_k(st, 1).bgc_steps is None
    s2 = transport.init_every_k(st, 3)
    assert all(getattr(s2, f) is not None for f in BGC_MR_FIELDS)
    assert int(np.asarray(s2.bgc_steps).max()) == 0 and s2.bgc_uvh.shape == st.uv.shape
    np.testing.assert_array_equal(np.asarray(s2.bgc_h0), np.asarray(st.hnode))
    assert transport.init_every_k(s2, 3) is s2                       # mid-interval restart: kept
    with pytest.raises(ValueError):
        RecomConfig(passive_tracers=1, bgc_every=0).validate()


@have_forcing
def test_every_k_holds_then_steps(pi_forced):
    """k = 2: after the first step of an interval the BGC tracers are unchanged and the sums hold one step; after the
    second the tracers moved and the sums are reset (bgc_h0 = the new thickness)."""
    cfg = RecomConfig(passive_tracers=NTR, passive_bc="salt", bgc_every=2)
    st0 = transport.init_every_k(transport.init_passive(pi_forced["state"], NTR), 2)
    s1 = _dense_run(pi_forced, cfg, state=st0, n=1)
    np.testing.assert_array_equal(np.asarray(s1.bgc), np.asarray(st0.bgc))
    assert int(np.asarray(s1.bgc_steps)[0]) == 1 and float(np.abs(np.asarray(s1.bgc_kv)).max()) > 0
    s2 = _dense_run(pi_forced, cfg, state=st0, n=2)
    assert int(np.asarray(s2.bgc_steps)[0]) == 0 and float(np.abs(np.asarray(s2.bgc_kv)).max()) == 0
    assert float(np.abs(np.asarray(s2.bgc) - np.asarray(st0.bgc)).max()) > 0
    np.testing.assert_array_equal(np.asarray(s2.bgc_h0), np.asarray(s2.hnode))


@have_forcing
def test_every_k_conserves_and_tracks_salinity(pi_forced):
    """k = 2 over two intervals: at the interval ends the salinity copies hold exactly S's inventory (the means are
    the summed fluxes, the thicknesses the interval ends), and stay close to S pointwise (the k-step scheme's own
    error, reported)."""
    cfg = RecomConfig(passive_tracers=NTR, passive_bc="salt", bgc_every=2)
    st0 = transport.init_every_k(transport.init_passive(pi_forced["state"], NTR), 2)
    out = _dense_run(pi_forced, cfg, state=st0, n=NSTACK)
    mesh = pi_forced["mesh"]
    inv_S = _inventory(mesh, out, out.S)
    inv_X = _inventory(mesh, out, out.bgc)
    np.testing.assert_allclose(inv_X, inv_S, rtol=1e-12, atol=0)
    wet = np.asarray(mesh.node_layer_mask)
    d = np.abs(np.asarray(out.bgc)[..., 0] - np.asarray(out.S))[wet]
    print(f"EVERY_K2 max |X - S| = {d.max():.3e} psu, rms {np.sqrt((d ** 2).mean()):.3e}")
    assert d.max() < 0.05


@have_forcing
@pytest.mark.parametrize("k", [1, 2])
def test_uniform_tracer(pi_forced, k):
    """A tracer that is 1 everywhere with no surface flux: under zstar its content is constant (precipitation and
    runoff add water without tracer, so the concentration itself dilutes: ~5e-3 at river mouths after 4 steps);
    under linfs (constant thickness) it stays 1 — the (mean) mass fluxes agree with the thicknesses, for the
    every-step scheme and the k-step one alike."""
    fx = pi_forced
    mesh = fx["mesh"]
    wet = np.asarray(mesh.node_layer_mask)
    cfg = RecomConfig(passive_tracers=1, bgc_every=k)
    st0 = transport.init_passive(fx["state"], 1)
    st0 = transport.init_every_k(dataclasses.replace(st0, bgc=jnp.ones_like(st0.bgc),
                                                     bgc_old=jnp.ones_like(st0.bgc_old)), k)
    out = _dense_run(fx, cfg, state=st0, n=NSTACK)                       # zstar
    np.testing.assert_allclose(_inventory(mesh, out, out.bgc), _inventory(mesh, st0, st0.bgc), rtol=1e-12)
    zero = jnp.zeros((mesh.elem2D, 2))
    st = st0
    linfs = {key: v for key, v in PHYS.items() if key != "ale_cfg"}
    for i in range(NSTACK):                                              # linfs
        sf = jax.tree.map(lambda x: x[i], fx["stack"])
        st = stepmod.step_jit(st, mesh, fx["op"], zero, dt=DT, is_first_step=(i == 0), step_forcing=sf,
                              forcing_static=fx["fs"], recom_cfg=cfg, **linfs)
    err = np.abs(np.asarray(st.bgc)[..., 0] - 1.0)[wet].max()
    print(f"UNIFORM linfs k={k}: max |X - 1| = {err:.3e}")
    assert err < 1e-12


@have_forcing
@pytest.mark.skipif(NDEV < 4 or not (PI_DIST / "dist_4").is_dir(),
                    reason="needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4) and pi dist_4")
def test_sharded_physics_equals_dense(pi_forced):
    """The sharded forced physics (zstar + w_split + mEVP + TKE + GM/Redi, no BGC) equals the dense run to round-off
    on every owned node and element of every State field after 4 steps. Before the ShardedMesh carried the global
    coastal mask, the runner's default (no ``boundary_node_p``) let the ice step rebuild it from each device's local
    mesh, which finds no coast, so the ice velocity was not zeroed at the 455 coastal nodes: |dS| 0.03 psu, |du_ice|
    0.15 m/s here."""
    fx = pi_forced
    mesh = fx["mesh"]
    dense = _dense_run(fx, None, n=NSTACK)
    out, part = _sharded_run(fx, fx["state"], None, NSTACK)
    lists = {mesh.nod2D: (part.myList_nod2D, part.myDim_nod2D), mesh.elem2D: (part.myList_elem2D, part.myDim_elem2D)}
    for f in dataclasses.fields(State):
        a, b = getattr(dense, f.name), getattr(out, f.name)
        if a is None or np.ndim(a) == 0 or np.shape(a)[0] not in lists:
            continue
        a, b = np.asarray(a), np.asarray(b)
        lst, dim = lists[a.shape[0]]
        scale = max(float(np.abs(a).max()), 1e-300)
        for p in range(4):
            gids = np.asarray(lst[p][: dim[p]])
            # measured with the fix: at most 2e-11 relative (Kv), S and the ice fields below 1e-13
            np.testing.assert_allclose(b[p, : len(gids)], a[gids], rtol=0, atol=1e-9 * scale,
                                       err_msg=f"{f.name} device {p}")


@have_forcing
@pytest.mark.skipif(NDEV < 4 or not (PI_DIST / "dist_4").is_dir(),
                    reason="needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4) and pi dist_4")
def test_every_k_sharded_equals_dense(pi_forced, monkeypatch):
    """k = 2 on the sharded forced path (the interval branch under lax.cond holds the FCT halo collectives) equals the
    dense k = 2 run on every owned node, one tracer at a time."""
    monkeypatch.setenv("FESOM_BGC_LAYOUT", "loop")
    cfg = RecomConfig(passive_tracers=NTR, passive_bc="salt", bgc_every=2)
    st0 = transport.init_every_k(transport.init_passive(pi_forced["state"], NTR), 2)
    dense = _dense_run(pi_forced, cfg, state=st0, n=NSTACK)
    out, part = _sharded_run(pi_forced, st0, cfg, NSTACK)
    got, gS = np.asarray(out.bgc), np.asarray(out.S)
    dX = dS = 0.0
    for p in range(4):
        gids = np.asarray(part.myList_nod2D[p][: part.myDim_nod2D[p]])
        dX = max(dX, float(np.abs(got[p, : len(gids)] - np.asarray(dense.bgc)[gids]).max()))
        dS = max(dS, float(np.abs(gS[p, : len(gids)] - np.asarray(dense.S)[gids]).max()))
    print(f"EVERY_K2 sharded vs dense: max|dX| = {dX:.3e}, physics max|dS| = {dS:.3e}")
    assert dX <= max(10 * dS, 1e-12), (dX, dS)
