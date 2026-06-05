"""Task 2.1 gate — EOS / hydrostatic pressure / N² (substep 1).

Verifies the JAX ``eos`` port against the C-port per-substep dump (``pi_cdump``):
``density``, ``pressure``, ``bvfreq`` (post-smooth) probe columns vs substep 1, at
every pinned node probe, within the per-kind tolerance. Plus the smoother is shown
to be load-bearing (raw bvfreq fails, smoothed passes) and an AD gate
``d(mean density)/d(T)`` (reverse-mode vs central finite differences).

The IC is constant T=10/S=35 **+ the Gaussian T-blob** (``ic.initial_state``); the
dump was produced with exactly that (see ``docs/PORTING_LESSONS.md``). T/S are
frozen over the dump window, so substep-1 fields are step-independent here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import eos, ic
from fesom_jax.io_dump import find_record
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.verify import assert_close, compare_column

NODE_PROBES = [1001, 1500, 2000, 2500, 3000]


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (Task 0.3)")
    return load_mesh()


@pytest.fixture(scope="module")
def eos_fields(mesh):
    """Compute (density, hpressure, bvfreq_smoothed) once for the module."""
    st = ic.initial_state(mesh)
    density, hpressure, bvfreq = eos.compute_pressure_bv(mesh, st.T, st.S, st.hnode)
    return (np.asarray(density), np.asarray(hpressure), np.asarray(bvfreq))


# --------------------------------------------------------------------------
# Rung-1 probe-column gates vs the C dump (all five node probes)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gid", NODE_PROBES)
def test_density_matches_dump(load_dump, mesh, eos_fields, gid):
    recs = load_dump("pi_cdump.00000")
    density = eos_fields[0]
    rec = find_record(recs, step=1, substep=1, field="density", probe_gid=gid)
    assert_close(density[gid - 1], rec, kind="map")


@pytest.mark.parametrize("gid", NODE_PROBES)
def test_pressure_matches_dump(load_dump, mesh, eos_fields, gid):
    recs = load_dump("pi_cdump.00000")
    hpressure = eos_fields[1]
    rec = find_record(recs, step=1, substep=1, field="pressure", probe_gid=gid)
    assert_close(hpressure[gid - 1], rec, kind="map")


@pytest.mark.parametrize("gid", NODE_PROBES)
def test_bvfreq_matches_dump(load_dump, mesh, eos_fields, gid):
    recs = load_dump("pi_cdump.00000")
    bvfreq = eos_fields[2]
    rec = find_record(recs, step=1, substep=1, field="bvfreq", probe_gid=gid)
    # N² is a node-patch scatter/reduction → 1e-12 class.
    assert_close(bvfreq[gid - 1], rec, kind="scatter")


def test_smoother_is_load_bearing(load_dump, mesh):
    """The dump's bvfreq is POST-smooth: raw N² must FAIL the gate and the
    smoothed N² must PASS — proving the smoother is necessary, not decorative."""
    recs = load_dump("pi_cdump.00000")
    st = ic.initial_state(mesh)
    _, _, bv_raw = eos.pressure_bv(mesh, st.T, st.S, st.hnode)
    bv_sm = eos.smooth_nod3D(mesh, bv_raw, 1)
    rec = find_record(recs, step=1, substep=1, field="bvfreq", probe_gid=1001)
    assert not compare_column(np.asarray(bv_raw)[1000], rec, kind="scatter").passed
    assert compare_column(np.asarray(bv_sm)[1000], rec, kind="scatter").passed


# --------------------------------------------------------------------------
# Smoother unit properties (independent of the dump)
# --------------------------------------------------------------------------
def test_smoother_preserves_constant(mesh):
    """An area-weighted patch average of a spatially-constant field is that
    constant, on every valid interface level (and 0 below bottom)."""
    c = jnp.where(mesh.node_iface_mask, 2.5, 0.0)
    out = np.asarray(eos.smooth_nod3D(mesh, c, 1))
    expect = np.asarray(c)
    assert np.allclose(out, expect, atol=1e-12, rtol=0)


# --------------------------------------------------------------------------
# Rung-4 gradient gate — d(mean density)/d(T), AD vs finite differences
# --------------------------------------------------------------------------
def test_density_gradient_ad_vs_fd(mesh):
    """Reverse-mode d(mean density)/d(T) at an in-blob, smooth-regime node
    matches a central finite-difference sweep. EOS is a smooth polynomial and
    pointwise in (node,level), so AD must agree to FD's convergence plateau."""
    st = ic.initial_state(mesh)
    S, hnode = st.S, st.hnode
    valid = mesh.node_layer_mask
    nvalid = jnp.sum(valid)

    def loss(T):
        density, _, _ = eos.pressure_bv(mesh, T, S, hnode)
        return jnp.sum(jnp.where(valid, density, 0.0)) / nvalid

    grad_ad = np.asarray(jax.grad(loss)(st.T))

    n, nz = 1000, 5  # node 1001 (in blob), a wet layer; T≈13, S=35 → no EOS kinks
    assert bool(valid[n, nz])
    g_ad = float(grad_ad[n, nz])

    base = st.T
    best = np.inf
    for h in (1e-4, 1e-5, 1e-6, 1e-7):
        Tp = base.at[n, nz].add(h)
        Tm = base.at[n, nz].add(-h)
        g_fd = float((loss(Tp) - loss(Tm)) / (2.0 * h))
        best = min(best, abs(g_ad - g_fd) / max(abs(g_fd), 1e-300))
    assert best < 1e-6, f"AD vs FD rel err {best:.2e} (g_ad={g_ad:.6e})"
