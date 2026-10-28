"""The placed-constants cache of ``run_steps_sharded_forced(reuse_executable=True)``
(``integrate_sharded._PLACED_CACHE``, ``FESOM_PLACE_CACHE``): the mesh, operator, stress, halo arrays
and forcing constants are folded, cast and placed once per run instead of every chunk. The values are
the same arrays, so two chunks with the cache are BITWISE the same two chunks without it, in double
and in single precision, and the second chunk reuses the first chunk's entry."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
IC_DIR = ROOT / "data" / "ic_core2"
NDEV = len(jax.devices())
_PLATFORM = jax.devices()[0].platform


def _have_jra():
    from fesom_jax import jra55
    return Path(jra55.DEFAULT_JRA_DIR).is_dir()


needs_core2 = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_4").is_dir()
    or not (IC_DIR / "T_ic.npy").exists() or not _have_jra(),
    reason="CORE2 mesh, dist_4, PHC IC or JRA55 forcing missing (compute node only)")


@pytest.fixture(scope="module")
def forced_setup():
    from fesom_jax import ice, ice_evp, partit, shard_mesh, ssh, surface_forcing
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import phc_initial_state
    if NDEV < 4:
        pytest.skip(f"needs 4 devices, have {NDEV}")
    dt, year, n = 1800.0, 1958, 2
    mesh = load_mesh(CORE2_MESH)
    state = phc_initial_state(mesh, IC_DIR)
    sst0 = np.asarray(state.T[:, 0])
    state = ice.seed_ice(state, mesh, sst0)
    cf = surface_forcing.build_surface_forcing(mesh, year, sst_ic=sst0)
    stacks = [cf.stack(surface_forcing.dates_for_steps(year, dt, 2 * n)[i * n:(i + 1) * n])
              for i in range(2)]
    part = partit.read_partition(CORE2_DIST, 4)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    sop = ssh.partition_ssh_operator(ssh.build_ssh_operator(mesh, dt=dt), part)
    L = sm.Lmax["nod"]
    bn = np.asarray(ice_evp.boundary_node_mask(mesh))
    bn_p = np.zeros((4, L), bn.dtype)
    for d in range(4):
        ml = np.asarray(part.myList_nod2D[d])
        bn_p[d, :ml.size] = bn[ml]
    return dict(dt=dt, n=n, state=state, part=part, sm=sm, sop=sop, bn_p=bn_p,
                stress=np.zeros((4, sm.Lmax["elem"], 2)),
                seqs=[shard_mesh.partition_step_forcing(s, part) for s in stacks],
                static=shard_mesh.partition_forcing_static(cf.static, part))


def _two_chunks(S, precision):
    from fesom_jax import integrate_sharded as ish
    from fesom_jax import shard_mesh
    from fesom_jax.ice import IceConfig
    fs = ish.folded_state(shard_mesh.partition_state(S["state"], S["part"]))[0]
    ent = []
    for seq in S["seqs"]:
        fs = ish.run_steps_sharded_forced(
            S["sm"], fs, S["sop"], S["stress"], seq, S["static"], S["n"], dt=S["dt"], npes=4,
            ice_cfg=IceConfig(whichEVP=1), boundary_node_p=S["bn_p"], state_is_folded=True,
            return_folded=True, reuse_executable=True, precision=precision)
        ent.append([id(v[0]) for v in ish._PLACED_CACHE.values()])
    return jax.tree.map(np.asarray, fs), ent


@needs_core2
@pytest.mark.skipif(_PLATFORM != "cpu", reason="bitwise is a CPU property")
def test_place_cache_bitwise_and_reused(forced_setup, monkeypatch, precision="double"):
    from fesom_jax import integrate_sharded as ish
    ish.clear_forced_jit_cache()
    # the production transport (core2_gpu.yaml); with the default full all_gather the forced 4-device CPU run
    # crashes the CPU runtime after the scan (segfault in unfold_state, 2026-10-21, cache or not)
    monkeypatch.setenv("FESOM_HALO_COMPACT", "1")
    monkeypatch.setenv("FESOM_PLACE_CACHE", "0")
    ref, ent_off = _two_chunks(forced_setup, precision)
    assert ent_off == [[], []], "FESOM_PLACE_CACHE=0 must not fill the cache"
    monkeypatch.setenv("FESOM_PLACE_CACHE", "1")
    got, ent_on = _two_chunks(forced_setup, precision)
    assert len(ent_on[0]) == 1 and ent_on[1] == ent_on[0], "the second chunk must reuse the entry"
    for name in ref.__dataclass_fields__:
        a, b = getattr(got, name), getattr(ref, name)
        if a is None:
            assert b is None, name
            continue
        np.testing.assert_array_equal(a, b, err_msg=name)
    ish.clear_forced_jit_cache()


@pytest.mark.parametrize("precision", ["double", "single"])
def test_place_cache_driver_bitwise(tmp_path, monkeypatch, precision):
    """The production driver (``run_from_config``, pi mesh, 4 fake devices, the ``core2_gpu.yaml`` options, executable
    reuse on): 6 steps in 2-step chunks write the same restart with the cache off and on, bit for bit, in double and
    in single precision (the CORE2 single-precision physics needs the full production setup, which the driver
    builds)."""
    from fesom_jax.tests.test_precision_shard import _cfg, _load, _run
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("FESOM_REUSE_EXE", "1")
    out = {}
    for pc in ("0", "1"):
        monkeypatch.setenv("FESOM_PLACE_CACHE", pc)
        _run(cfg, "--precision", precision, "--steps", "6", "--chunk-steps", "2",
             "--restart-out", str(tmp_path / f"r{pc}"))
        out[pc] = _load(tmp_path / f"r{pc}")
    assert out["0"].keys() == out["1"].keys()
    for k in out["0"]:
        np.testing.assert_array_equal(out["0"][k], out["1"][k], err_msg=k)
