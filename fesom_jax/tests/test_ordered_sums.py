"""Ordered sums (docs/plans/20261020-core2-ordered-sums.md, opt-in ``gpu.ordered_sums`` /
``FESOM_ORDERED_SUMS=1``): ``ops.scatter_add`` through the mesh's inverse maps (ordered gathers, each
target adds its entries in ascending entry order) against the default ``segment_sum``, on CPU.

Task 1: the three mesh assemblies (``elem_nodes`` / ``edges`` -> nodes, ``edge_tri`` -> elements) on
CORE2 with 2-D and 3-D values, forward and gradient. Task 2 (below): the sharded mesh's per-device maps
and the sharded step with the option on vs off.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import mesh as mesh_mod
from fesom_jax import ops

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")

avail = pytest.mark.skipif(not CORE2_MESH.is_dir(), reason="CORE2 mesh missing")


@pytest.fixture(scope="module")
def mesh():
    return mesh_mod.with_inverse_maps(mesh_mod.load_mesh(CORE2_MESH))


def _ulps(a, b):
    """Per-entry distance in units in the last place (0 where equal, signed zeros equal)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    eq = a == b
    sp = np.spacing(np.maximum(np.abs(a), np.abs(b)))
    return np.where(eq, 0.0, np.abs(a - b) / np.where(sp > 0, sp, 1.0))


ASSEMBLIES = [("elem_nodes", "nod2D"), ("edges", "nod2D"), ("edge_tri", "elem2D")]


@avail
@pytest.mark.parametrize("seg_name,count", ASSEMBLIES)
@pytest.mark.parametrize("rest", [(), (48,), (48, 2)])
def test_core2_ordered_sum_vs_segment_sum(mesh, seg_name, count, rest, monkeypatch):
    seg = getattr(mesh, seg_name)
    inv = getattr(mesh, "inv_" + seg_name)
    N = getattr(mesh, count)
    rng = np.random.default_rng(7)
    vals = jnp.asarray(rng.standard_normal(seg.shape + rest)
                       * 10.0 ** rng.integers(-6, 6, size=seg.shape + (1,) * len(rest)))
    f = jax.jit(lambda v, iv: ops.scatter_add(v, seg, N, inv=iv))
    monkeypatch.delenv("FESOM_ORDERED_SUMS", raising=False)
    ref = np.asarray(f(vals, inv))
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    got = np.asarray(jax.jit(lambda v, iv: ops.scatter_add(v, seg, N, inv=iv))(vals, inv))
    u = _ulps(got, ref)
    print(f"{seg_name} rest={rest}: D={inv.shape[1]} bitwise={np.array_equal(got, ref)} "
          f"frac_neq={np.mean(u > 0):.3e} max_ulp={u.max():.1f}")
    assert u.max() <= 1.0


@avail
@pytest.mark.parametrize("seg_name,count", ASSEMBLIES)
def test_core2_ordered_sum_gradient(mesh, seg_name, count, monkeypatch):
    seg = getattr(mesh, seg_name)
    inv = getattr(mesh, "inv_" + seg_name)
    N = getattr(mesh, count)
    rng = np.random.default_rng(8)
    vals = jnp.asarray(rng.standard_normal(seg.shape + (48,)))
    w = jnp.asarray(rng.standard_normal((N, 48)))
    J = lambda v: jnp.sum(w * jnp.sin(ops.scatter_add(v, seg, N, inv=inv)))
    monkeypatch.delenv("FESOM_ORDERED_SUMS", raising=False)
    g_ref = np.asarray(jax.jit(jax.grad(J))(vals))
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    g_on = np.asarray(jax.jit(jax.grad(lambda v: J(v)))(vals))
    scale = np.abs(g_ref).max()
    np.testing.assert_allclose(g_on, g_ref, rtol=0, atol=1e-13 * scale)


# --------------------------------------------------------------------------
# Task 2: the sharded mesh's per-device maps, the sharded step on vs off
# --------------------------------------------------------------------------
NDEV = len(jax.devices())
IC_DIR = ROOT / "data" / "ic_core2"
DT = 1800.0
avail_dist = pytest.mark.skipif(not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_4").is_dir(),
                                reason="CORE2 mesh or dist partitions missing")


def _have_forcing():
    from fesom_jax import jra55
    return (IC_DIR / "T_ic.npy").exists() and Path(jra55.DEFAULT_JRA_DIR).is_dir()


@pytest.fixture(scope="module")
def dense_mesh():
    return mesh_mod.load_mesh(CORE2_MESH)


def _owned_report(got, ref, nod_dim, elem_dim, part, tag, bar=1e-11):
    """Owned lanes of every [P, Lmax, ...] State field: print bitwise / max|d| / scale, assert
    max|d| <= bar * the field's owned max|ref|. Returns {field: bitwise}."""
    from fesom_jax.state import State
    import dataclasses
    out, bad = {}, []
    print(f"\n[{tag}] owned lanes, on vs off:")
    for fld in dataclasses.fields(State):
        a, b = np.asarray(getattr(got, fld.name)), np.asarray(getattr(ref, fld.name))
        if a.ndim < 2:
            continue
        myd = part.myDim_nod2D if a.shape[1] == nod_dim else (
            part.myDim_elem2D if a.shape[1] == elem_dim else None)
        if myd is None:
            continue
        aa = np.concatenate([a[d, :int(myd[d])].reshape(-1) for d in range(part.npes)])
        bb = np.concatenate([b[d, :int(myd[d])].reshape(-1) for d in range(part.npes)])
        diff = float(np.max(np.abs(aa - bb))) if aa.size else 0.0
        scale = max(float(np.max(np.abs(bb))) if bb.size else 0.0, 1e-300)
        out[fld.name] = bool(np.array_equal(aa, bb))
        u = _ulps(aa, bb)
        print(f"   {fld.name:16s} bitwise={out[fld.name]!s:5s} max|d|={diff:.3e} rel={diff / scale:.3e} "
              f"neq={np.mean(u > 0):.2e} max_ulp={u.max() if u.size else 0:.3g}")
        if diff > bar * scale:
            bad.append(f"{fld.name}: max|d| {diff:.3e} > {bar:.0e} * {scale:.3e}")
    assert not bad, f"[{tag}] " + "; ".join(bad)
    return out


@avail_dist
@pytest.mark.parametrize("npes", [2, 4])
def test_sharded_inverse_maps(dense_mesh, npes, monkeypatch):
    """Per device: the maps of the LOCAL connectivity (``-1`` lanes excluded), one width D, the
    sentinel = the flattened local index array's length; local ordered sums == the local
    segment_sum bitwise (CPU); with the option off the local Mesh carries no maps."""
    from fesom_jax import integrate_sharded as ish
    from fesom_jax import partit, shard_mesh
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(dense_mesh, part)
    smi = shard_mesh.with_inverse_maps(sm)
    assert shard_mesh.with_inverse_maps(smi) is smi
    rng = np.random.default_rng(9)
    for name, src, tgt in mesh_mod.INVERSE_MAP_FIELDS:
        conn, inv = sm.fields[src], smi.fields[name]
        assert inv.shape[:2] == (npes, sm.Lmax[tgt]) and inv.dtype == np.int32
        for d in range(npes):
            ref = ops.inverse_map(conn[d], sm.Lmax[tgt], width=inv.shape[2])
            np.testing.assert_array_equal(inv[d], ref)
            assert (inv[d] <= conn[d].size).all()          # sentinel = the local flattened length
    monkeypatch.delenv("FESOM_ORDERED_SUMS", raising=False)
    assert ish.local_mesh(smi, 0).inv_edges is None
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    for d in range(npes):
        lm = ish.local_mesh(sm, d)                       # built from sm on the fly
        for name, src, tgt in mesh_mod.INVERSE_MAP_FIELDS:
            seg, inv = getattr(lm, src), getattr(lm, name)
            N = lm.nod2D if tgt == "nod" else lm.elem2D
            v = jnp.asarray(rng.standard_normal(seg.shape + (5,)))
            got = np.asarray(ops.scatter_add(v, seg, N, inv=inv))
            ref = np.asarray(ops.scatter_add(v, seg, N))
            np.testing.assert_array_equal(got, ref, err_msg=f"{name} d{d}")


@avail_dist
@pytest.mark.parametrize("npes", [2])
def test_sharded_ocean_step_ordered_matches(dense_mesh, npes, monkeypatch):
    """The full sharded ocean step (run_step_sharded, the column_unroll/compact test's setup) with
    the ordered sums on vs off: round-off (1e-11 of each field's scale) on owned lanes."""
    import dataclasses
    from fesom_jax import integrate_sharded as ish
    from fesom_jax import partit, shard_mesh, ssh
    from fesom_jax.state import State
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = dense_mesh
    op = ssh.build_ssh_operator(mesh, dt=DT)
    st = State.rest(mesh)
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]
    bump = 0.5 * np.cos(2 * lat)[:, None]
    state = dataclasses.replace(st, T=jnp.asarray(np.asarray(st.T) + np.where(
        np.asarray(mesh.node_layer_mask), bump, 0.0)))
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = jnp.zeros((npes, sm.Lmax["elem"], 2))
    kw = dict(dt=DT, is_first_step=True, npes=npes)
    monkeypatch.delenv("FESOM_ORDERED_SUMS", raising=False)
    st_off = ish.run_step_sharded(sm, state_p, sop, stress_p, **kw)
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    st_on = ish.run_step_sharded(sm, state_p, sop, stress_p, **kw)
    _owned_report(st_on, st_off, sm.Lmax["nod"], sm.Lmax["elem"], part, "ocean step")


@avail_dist
@pytest.mark.skipif(not _have_forcing(), reason="CORE2 IC or JRA55 forcing missing")
@pytest.mark.parametrize("n,mode", [(1, "1"), (3, "1"), (1, "2"), (1, "3"), (3, "2"), (3, "unroll")])
def test_forced_run_ordered_matches(dense_mesh, n, mode, monkeypatch):
    """3 forced CORE2 steps on dist_4 with core2_gpu.yaml's physics (zstar, GM/Redi, TKE, mEVP) and its
    speed options (Chebyshev 3, the gathered SSH solve, compact halo, unrolled columns, mEVP wide K=12):
    ordered sums on vs off at
    round-off on owned lanes. Off: the lowered program does not change when the sharded mesh carries
    the maps (they are dropped), so the default program is the one without the option; on: it does."""
    import hashlib
    from fesom_jax import ice, ice_evp, ice_wide, partit, shard_mesh, ssh, surface_forcing
    from fesom_jax import integrate_sharded as ish
    from fesom_jax.phc_ic import phc_initial_state
    from fesom_jax.run_config import load_yaml
    if NDEV < 4:
        pytest.skip(f"needs 4 devices, have {NDEV}")
    monkeypatch.setenv("FESOM_HALO_COMPACT", "1")
    monkeypatch.setenv("FESOM_COLUMN_UNROLL", "1")
    cfg = load_yaml(ROOT / "configs" / "core2_gpu.yaml")
    mesh = dense_mesh
    state = phc_initial_state(mesh, IC_DIR)
    sst0 = np.asarray(state.T[:, 0])
    state = ice.seed_ice(state, mesh, sst0)
    cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=sst0)
    stack = cf.stack(surface_forcing.dates_for_steps(1958, DT, n))
    part = partit.read_partition(CORE2_DIST, 4)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    op = ssh.enable_cheb_precond(ssh.build_ssh_operator(mesh, dt=DT), 3, kappa_guess=30.0)
    sop = ssh.partition_ssh_operator(op, part)
    bn = np.asarray(ice_evp.boundary_node_mask(mesh))
    bn_p = np.zeros((4, sm.Lmax["nod"]), bn.dtype)
    for d in range(4):
        ml = part.myList_nod2D[d]
        bn_p[d, :ml.size] = bn[ml]
    wide = ice_wide.build_mevp_wide(mesh, part, sm, 12)
    kw = dict(dt=DT, npes=4, boundary_node_p=bn_p, mevp_wide=wide, **cfg.physics_kwargs())

    def run(s, **extra):                     # the gathered solve's tables read the switch when built
        return ish.run_steps_sharded_forced(
            s, shard_mesh.partition_state(state, part), sop,
            np.zeros((4, sm.Lmax["elem"], 2)), shard_mesh.partition_step_forcing(stack, part),
            shard_mesh.partition_forcing_static(cf.static, part), n,
            ssh_gather=ssh.build_ssh_gather(op, mesh, part), **kw, **extra)

    def hlo(s):
        jfn, args, _ = run(s, return_executable=True)
        return hashlib.sha256(jfn.lower(*args).as_text().encode()).hexdigest()

    monkeypatch.delenv("FESOM_ORDERED_SUMS", raising=False)
    if mode == "unroll":                    # calibration: the accepted round-off change of gpu.column_unroll
        monkeypatch.delenv("FESOM_COLUMN_UNROLL", raising=False)
        ref = run(sm)
        monkeypatch.setenv("FESOM_COLUMN_UNROLL", "1")
        _owned_report(run(sm), ref, sm.Lmax["nod"], sm.Lmax["elem"], part,
                      f"{n} forced steps, column_unroll on vs off (calibration)", bar=1e30)
        return
    h_off = hlo(sm)
    assert hlo(shard_mesh.with_inverse_maps(sm)) == h_off
    ref = run(sm)
    monkeypatch.setenv("FESOM_ORDERED_SUMS", mode)
    h_on = hlo(sm)
    print(f"\nHLO sha256 off {h_off[:16]}  on {h_on[:16]}")
    assert h_on != h_off
    got = run(sm)
    # One step: a few ulp (rel <= 5e-16 in the prognostic fields, <= 3e-13 in the SSH solve's outputs, the
    # CG's response to a 1-ulp rhs change), the same in every mode (jobs 28292121/28292146/28294897); the
    # CPU program is deterministic either way (job 28295532). Over 3 steps the mEVP amplifies it (sigma
    # ~1e-6 relative; the column_unroll A/A reaches 1e-9 from fewer 1-ulp seeds), so that is only reported.
    bit = _owned_report(got, ref, sm.Lmax["nod"], sm.Lmax["elem"], part, f"{n} forced steps, mode {mode}",
                        bar=1e-12 if n == 1 else 1e30)
    assert np.abs(np.asarray(ref.u_ice)[0, :int(part.myDim_nod2D[0])]).max() > 1e-4
    print("bitwise fields:", sum(bit.values()), "of", len(bit))


@avail
def test_xla_folds_an_add_into_the_scatter(mesh, monkeypatch):
    """Diagnostic: ``x + scatter_add(v)`` under jit. The default program's result is the sequential sum
    started from ``x`` (XLA folds the add into the scatter: ``add(A, scatter(0, ...))`` ->
    ``scatter(A, ...)``), not "sum from 0, then add" (up to 6824 ulp apart on random data, job 28293423),
    so a written ``x + sum`` is not what the default program computes; the ordered program gave the
    from-``x`` result too in that job. Prints which order each program used; asserts the default's is one
    of the two."""
    seg, inv, N = mesh.edges, mesh.inv_edges, mesh.nod2D
    rng = np.random.default_rng(10)
    v = rng.standard_normal(seg.shape) * 10.0 ** rng.integers(-6, 6, size=seg.shape)
    x = rng.standard_normal(N) * 1e3
    s_flat, v_flat = np.asarray(seg).reshape(-1), v.reshape(-1)
    from_x, from_0 = x.copy(), np.zeros(N)
    for i, t in enumerate(s_flat):
        from_x[t] = from_x[t] + v_flat[i]
        from_0[t] = from_0[t] + v_flat[i]
    then_add = x + from_0
    f = lambda xx, vv: xx + ops.scatter_add(vv, seg, N, inv=inv)
    monkeypatch.delenv("FESOM_ORDERED_SUMS", raising=False)
    off = np.asarray(jax.jit(f)(jnp.asarray(x), jnp.asarray(v)))
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    on = np.asarray(jax.jit(f)(jnp.asarray(x), jnp.asarray(v)))
    print(f"\ndefault: == sum-then-add {np.array_equal(off, then_add)}, == sum-from-x "
          f"{np.array_equal(off, from_x)}; ordered: == sum-then-add {np.array_equal(on, then_add)}, "
          f"== sum-from-x {np.array_equal(on, from_x)}, max ulp vs sum-then-add {_ulps(on, then_add).max()}, "
          f"vs sum-from-x {_ulps(on, from_x).max()}")
    s_on = np.asarray(jax.jit(lambda vv: ops.scatter_add(vv, seg, N, inv=inv))(jnp.asarray(v)))
    print(f"ordered sum alone == from_0: {np.array_equal(s_on, from_0)}")
    assert np.array_equal(off, from_x) or np.array_equal(off, then_add)
