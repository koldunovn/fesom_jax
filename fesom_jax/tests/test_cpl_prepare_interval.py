"""Lane ocean_speed3 (CPU): ``coupled.prepare_interval`` (the forcing stack built ahead, where the forcing lives,
then placed) gives the interval bitwise what ``ocean_interval`` gives with the forcing dict itself, on one device
and on CORE2 ``dist_2`` (two fake CPU devices); and ``gpu.o2a_lanes_surface`` gives ``ocean_to_atm`` the same
fields on ``dist_2``. One coupled interval of the CORE2 coupled configuration in mode F from its restart."""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "core2_cpl.yaml"
DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/dist_2")
RST = Path("/work/ab0995/a270088/port_jax/runs/core2_hindcast_v3/restart_archive/fesom.1989.001.00000")

avail = pytest.mark.skipif(not (ROOT / "data" / "mesh_core2").is_dir() or not RST.exists() or not DIST.is_dir()
                           or len(jax.devices()) < 2,
                           reason="needs the CORE2 mesh, the core2_cpl restart, dist_2 and 2 (fake) devices")


def _forcing(n):
    rng = np.random.default_rng(3)
    f = {k: jnp.asarray(s * rng.standard_normal(n).astype(np.float32)) for k, s in
         (("A_TauX_oce", 0.1), ("A_TauY_oce", 0.1), ("A_TauX_ice", 0.1), ("A_TauY_ice", 0.1),
          ("A_Qns_oce", 50.0), ("A_Qs_all", 50.0), ("A_Q_ice", 20.0))}
    f.update({k: jnp.asarray(1e-8 * np.abs(rng.standard_normal(n)).astype(np.float32)) for k in
              ("A_Precip_liquid", "A_Precip_solid", "A_Evap", "A_Subl")})
    f["A_Ice_tskin"] = jnp.asarray((260.0 + rng.standard_normal(n)).astype(np.float32))
    f["hydr_oce"] = jnp.zeros((n,), jnp.float32)
    f["calv_oce"] = jnp.zeros((n,), jnp.float32)
    return f


def _leaves(state):
    return {k: np.asarray(v) for k, v in vars(state).items() if isinstance(v, jax.Array)}


@avail
@pytest.mark.parametrize("partition", ["serial", "dist_2"])
def test_prepared_interval_bitwise(monkeypatch, partition):
    from fesom_jax import coupled as C
    monkeypatch.setenv("FESOM_CPL_FORCING_JIT", "1")
    monkeypatch.setenv("FESOM_DIST_DIR", str(DIST.parent))
    out = []
    for prep in (False, True):
        state, st = C.init_ocean(str(CFG), restart_in=str(RST), mode="flux", nsteps=2,
                                 partition=None if partition == "serial" else partition)
        fo = C.oasis_to_fesom(_forcing(st.nod2D))
        f = C.prepare_interval(fo, st, st.step, 2) if prep else fo
        state = C.ocean_interval(state, st, f, nsteps=2)
        out.append(_leaves(state))
    a, b = out
    assert sorted(a) == sorted(b)
    bad = [k for k in a if not np.array_equal(a[k], b[k], equal_nan=True)]
    assert not bad, bad


@avail
def test_o2a_lanes_surface_dist2(monkeypatch):
    from fesom_jax import coupled as C
    monkeypatch.setenv("FESOM_DIST_DIR", str(DIST.parent))
    monkeypatch.setenv("FESOM_O2A_SURFACE", "1")
    state, st = C.init_ocean(str(CFG), restart_in=str(RST), mode="flux", nsteps=2, partition="dist_2")
    ref = {k: np.asarray(v) for k, v in C.ocean_to_atm(state, st).items()}
    monkeypatch.setenv("FESOM_O2A_LANES_SURFACE", "1")
    st._o2a = None
    new = {k: np.asarray(v) for k, v in C.ocean_to_atm(state, st).items()}
    assert sorted(ref) == sorted(new)
    for k in ref:
        np.testing.assert_array_equal(new[k], ref[k], err_msg=k)
