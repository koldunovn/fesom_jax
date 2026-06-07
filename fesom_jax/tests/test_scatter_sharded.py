"""S.4 gate: the local-scatter + broadcast correctness gate + the exchange schedule.

The crux of the whole sharding scheme: a kernel's **local** ``segment_sum`` (over a
device's local owned+halo entities) plus the post-kernel broadcast must reproduce
the single-device global scatter **on owned entities** to ~1e-12. We verify this on
a representative **edge→node** scatter (the ``compute_ssh_rhs`` shape) and an
**edge→element** scatter (the biharmonic-viscosity shape), and check the broadcast
then makes the halo copies correct too.

Also validates the :mod:`fesom_jax.halo_points` schedule (a pure-data check that runs
anywhere). The broadcast parts need CPU fake-devices and SKIP otherwise.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import halo, halo_points, ops, partit, shard_mesh
from fesom_jax.mesh import load_mesh

CORE2_MESH = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NDEV = len(jax.devices())

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_2").is_dir(),
    reason="CORE2 dense mesh or dist partitions missing",
)


# --------------------------------------------------------------------------
# Exchange schedule (pure data — runs anywhere)
# --------------------------------------------------------------------------
def test_schedule_valid():
    halo_points.validate(halo_points.OCEAN_SCHEDULE)
    halo_points.validate(halo_points.ICE_SCHEDULE)
    # the ocean schedule has both post and intra exchanges, and the known fused
    # kernels are flagged for splitting
    assert len(halo_points.post_exchanges(halo_points.OCEAN_SCHEDULE)) > 10
    assert len(halo_points.intra_exchanges(halo_points.OCEAN_SCHEDULE)) >= 5
    kernels = {f.kernel for f in halo_points.FUSED_KERNELS_NEEDING_SPLIT}
    assert "momentum.visc_filt_bidiff" in kernels
    assert "ssh._pcg (CG)" in kernels
    assert "ice_evp EVP subcycle" in kernels
    # node-vs-elem kind sanity on a few well-known fields
    by_field = {(e.after, e.field): e for e in halo_points.OCEAN_SCHEDULE}
    assert by_field[("3 pgf.pressure_force_linfs", "pgf_x")].kind == "elem"
    assert by_field[("8 ssh.compute_ssh_rhs", "ssh_rhs")].kind == "nod"
    # the EVP subcycle exchange is intra (inside the lax.scan)
    evp = [e for e in halo_points.ICE_SCHEDULE if "EVP" in e.after]
    assert evp and all(e.placement == "intra" for e in evp)


# --------------------------------------------------------------------------
# Scatter correctness gate
# --------------------------------------------------------------------------
def _edge_vals(n_edge: int) -> np.ndarray:
    """A smooth O(1) per-edge field for the scatter (deterministic)."""
    return np.sin(np.arange(n_edge) * 0.001 + 0.3)


@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_edge_to_node_scatter_owned_complete(npes):
    """Local edge→node ``segment_sum`` gives each OWNED node its complete sum
    (no broadcast needed) — the loop-bound rule. Antisymmetric contribution
    (+/-) mirrors a transport-divergence scatter (compute_ssh_rhs)."""
    if NDEV < 1:
        pytest.skip("need a device")
    mesh = load_mesh(CORE2_MESH)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    edges_g = np.asarray(mesh.edges)
    ev = _edge_vals(mesh.edge2D)

    # global edge→node scatter: +ev to node0, -ev to node1
    contrib_g = np.stack([ev, -ev], axis=1)
    out_g = np.asarray(ops.scatter_add(jnp.asarray(contrib_g), jnp.asarray(edges_g), mesh.nod2D))

    Le = sm.Lmax["edge"]; Ln = sm.Lmax["nod"]
    for d in range(npes):
        myl_e = part.myList_edge2D[d]
        ev_loc = np.zeros(Le); ev_loc[: myl_e.size] = ev[myl_e]
        contrib_loc = np.stack([ev_loc, -ev_loc], axis=1)
        edges_loc = sm.fields["edges"][d]                       # [Le, 2] local node idx (-1 pad/unmappable)
        out_loc = np.asarray(ops.scatter_add(jnp.asarray(contrib_loc),
                                             jnp.asarray(edges_loc), Ln))
        md = int(part.myDim_nod2D[d])
        owned_gids = part.myList_nod2D[d][:md]
        assert np.allclose(out_loc[:md], out_g[owned_gids], atol=1e-11, rtol=1e-10), \
            f"dev {d}: owned-node local scatter != global"


@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_edge_to_elem_scatter_owned_complete(npes):
    """Local edge→element scatter (biharmonic shape; ``edge_tri`` carries -1 at
    boundaries) gives each OWNED element its complete sum, incl. redundantly-owned
    boundary elements on every owning device."""
    mesh = load_mesh(CORE2_MESH)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    edge_tri_g = np.asarray(mesh.edge_tri)
    ev = _edge_vals(mesh.edge2D)

    contrib_g = np.stack([ev, -ev], axis=1)
    out_g = np.asarray(ops.scatter_add(jnp.asarray(contrib_g),
                                       jnp.asarray(edge_tri_g), mesh.elem2D))

    Le = sm.Lmax["edge"]; Lel = sm.Lmax["elem"]
    for d in range(npes):
        myl_e = part.myList_edge2D[d]
        ev_loc = np.zeros(Le); ev_loc[: myl_e.size] = ev[myl_e]
        contrib_loc = np.stack([ev_loc, -ev_loc], axis=1)
        et_loc = sm.fields["edge_tri"][d]                       # [Le, 2] local elem idx (-1)
        out_loc = np.asarray(ops.scatter_add(jnp.asarray(contrib_loc),
                                             jnp.asarray(et_loc), Lel))
        me = int(part.myDim_elem2D[d])
        owned_gids = part.myList_elem2D[d][:me]
        assert np.allclose(out_loc[:me], out_g[owned_gids], atol=1e-11, rtol=1e-10), \
            f"dev {d}: owned-elem local scatter != global"


@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_scatter_broadcast_fills_halo(npes):
    """After the local scatter + broadcast, the HALO node copies equal the global
    scatter too (so the next kernel reads correct halo values). Needs fake-devices."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    edges_g = np.asarray(mesh.edges)
    ev = _edge_vals(mesh.edge2D)
    contrib_g = np.stack([ev, -ev], axis=1)
    out_g = np.asarray(ops.scatter_add(jnp.asarray(contrib_g), jnp.asarray(edges_g), mesh.nod2D))

    Le, Ln = sm.Lmax["edge"], sm.Lmax["nod"]
    out_loc = np.zeros((npes, Ln))
    for d in range(npes):
        myl_e = part.myList_edge2D[d]
        ev_loc = np.zeros(Le); ev_loc[: myl_e.size] = ev[myl_e]
        contrib_loc = np.stack([ev_loc, -ev_loc], axis=1)
        out_loc[d] = np.asarray(ops.scatter_add(jnp.asarray(contrib_loc),
                                                jnp.asarray(sm.fields["edges"][d]), Ln))
    # broadcast: halo lanes pick up their owner's (complete) value
    src_dev, src_lane = sm.exchange["nod"]
    jmesh = halo.device_mesh(devices=jax.devices()[:npes])
    out_ex = np.asarray(halo.run_halo_exchange(out_loc, src_dev, src_lane, jmesh))
    for d in range(npes):
        n_loc = part.myList_nod2D[d].size
        gids = part.myList_nod2D[d][:n_loc]
        assert np.allclose(out_ex[d, :n_loc], out_g[gids], atol=1e-11, rtol=1e-10), \
            f"dev {d}: owned+halo after broadcast != global"
