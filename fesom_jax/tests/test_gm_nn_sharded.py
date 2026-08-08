"""NN-GM Task 7 gate, sharded half — the ``gm_nn`` seam is halo-correct.

Why this cannot live in ``test_gm_nn_seam.py``: it needs CPU fake-devices, so it belongs to the
suite's SHARDING group (``test_*_sharded.py``, run with
``--xla_force_host_platform_device_count=4``) and skips cleanly at one device.

**What it protects.** The seam's features derive from ``sigma_xy``, an element→node scatter whose
result is INCOMPLETE on halo nodes. Multiply ``Ki`` before ``gm_diagnostics``' halo exchange and
the owner's correct value is what gets broadcast; multiply after it and every halo node carries a
multiplier computed from a partial scatter. Both versions pass every single-device test in the
seam file, and both pass a standing suite run at one device — the difference only appears on ≥2
devices, on owned boundary nodes.

This mirrors ``test_step_sharded.test_gm_diagnostics_sharded_owned_matches`` exactly, with one
change: a NON-TRIVIAL network in ``Params``. Run with a ``zero_last`` network it would pass no
matter where the multiplication sits, since the multiplier is 1 everywhere.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import integrate_sharded as ish
from fesom_jax import gm_nn, partit, shard_mesh
from fesom_jax.gm import GMConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NDEV = len(jax.devices())

_PLATFORM = jax.devices()[0].platform
_BYTE_ID_ATOL = 1e-9 if _PLATFORM == "cpu" else 1e-7

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_2").is_dir(),
    reason="CORE2 dense mesh or dist partitions missing")
have_ic = pytest.mark.skipif(
    not (IC_DIR / "T_ic.npy").exists(),
    reason="CORE2 PHC IC cache missing (data/ic_core2/)")


@avail
@have_ic
@pytest.mark.parametrize("npes", [2])
def test_gm_nn_diagnostics_sharded_owned_matches(npes):
    """With a trained-shape ``gm_nn`` in ``Params``, the sharded GM chain matches single-device
    on OWNED entities to the scatter floor, for all three fields the tracer step consumes.

    A post-exchange multiplication would show up here as an O(1) error on owned boundary nodes
    (their ``sigma_xy`` is a partial scatter), which is qualitatively different from the ~1e-16
    reassociation noise the floor allows."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    from fesom_jax import ale, eos, gm
    from fesom_jax.phc_ic import phc_initial_state
    from fesom_jax.shard_mesh import _shard_along_axis, local_sizes

    mesh = load_mesh(CORE2_MESH)
    state = phc_initial_state(mesh, IC_DIR)
    gm_cfg = GMConfig()
    nn = gm_nn.init_gm_nn(jax.random.PRNGKey(11), zero_last=False)
    params = dataclasses.replace(Params.defaults(), gm_nn=nn)

    _, _, bvfreq = eos.compute_pressure_bv(mesh, state.T, state.S, state.hnode)
    hnode_new = ale.thickness_linfs(state.hnode)
    diag = gm.gm_diagnostics(mesh, state.T, state.S, bvfreq, hnode_new, state.helem,
                             params, gm_cfg)

    # The NN must be doing something, or this test proves nothing about placement.
    base = gm.gm_diagnostics(mesh, state.T, state.S, bvfreq, hnode_new, state.helem,
                             Params.defaults(), gm_cfg)
    wet = np.asarray(base.Ki) > 0.0
    ratio = np.asarray(diag.Ki)[wet] / np.asarray(base.Ki)[wet]
    assert float(np.std(np.log(ratio))) > 1e-3, \
        "the NN multiplier is ~constant on this state; the placement test would be vacuous"

    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    _, Lmax = local_sizes(part)
    pn = lambda a: _shard_along_axis(np.asarray(a), part.myList_nod2D, Lmax["nod"], 0, 1.0)
    pe = lambda a: _shard_along_axis(np.asarray(a), part.myList_elem2D, Lmax["elem"], 0, 1.0)
    fer_uv_N, slope_N, Ki_N = ish.run_gm_diag_sharded(
        sm, pn(state.T), pn(state.S), pn(bvfreq), pn(hnode_new), pe(state.helem),
        npes=npes, gm_cfg=gm_cfg, params=params)

    for name, ref, got, myl, mydim in (
            ("fer_uv", diag.fer_uv, fer_uv_N, part.myList_elem2D, part.myDim_elem2D),
            ("slope_tapered", diag.slope_tapered, slope_N, part.myList_nod2D, part.myDim_nod2D),
            ("Ki", diag.Ki, Ki_N, part.myList_nod2D, part.myDim_nod2D)):
        a, B = np.asarray(ref), np.asarray(got)
        worst = 0.0
        for d in range(npes):
            md = int(mydim[d])
            worst = max(worst, float(np.max(np.abs(B[d, :md] - a[myl[d][:md]]))))
        print(f"[gm-nn-diag] {name:14s} owned max|Δ|={worst:.3e}")
        assert worst < _BYTE_ID_ATOL, (
            f"{name}: owned max|Δ|={worst:.3e} (floor {_BYTE_ID_ATOL:.0e}) — the NN multiplier "
            f"is being applied on the wrong side of the halo exchange")
