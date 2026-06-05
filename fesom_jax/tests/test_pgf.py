"""Task 2.2 gate — pressure-gradient force at elements (substep 3).

Verifies ``pgf.pressure_force_linfs`` against the C-port element dump: ``pgf_x`` /
``pgf_y`` probe columns vs substep 3, at every pinned **element** probe (the first
cell incident to each node probe), within the gather tolerance. Plus a light AD
sanity check that gradients flow through EOS→PGF.

The PGF consumes the substep-1 ``hpressure``, so the IC is the same constant + blob
(``ic.initial_state``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import eos, ic, pgf
from fesom_jax.io_dump import find_record
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.verify import assert_close

# Element probes = first cell incident to each node probe (fixture, REFERENCE_RUNS.md).
ELEM_PROBES = [1757, 2656, 3688, 4604, 5575]


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (Task 0.3)")
    return load_mesh()


@pytest.fixture(scope="module")
def pgf_fields(mesh):
    st = ic.initial_state(mesh)
    _, hpressure, _ = eos.pressure_bv(mesh, st.T, st.S, st.hnode)
    px, py = pgf.pressure_force_linfs(mesh, hpressure)
    return np.asarray(px), np.asarray(py)


@pytest.mark.parametrize("gid", ELEM_PROBES)
def test_pgf_x_matches_dump(load_dump, mesh, pgf_fields, gid):
    recs = load_dump("pi_cdump.00000")
    rec = find_record(recs, step=1, substep=3, field="pgf_x", probe_gid=gid)
    assert_close(pgf_fields[0][gid - 1], rec, kind="gather")


@pytest.mark.parametrize("gid", ELEM_PROBES)
def test_pgf_y_matches_dump(load_dump, mesh, pgf_fields, gid):
    recs = load_dump("pi_cdump.00000")
    rec = find_record(recs, step=1, substep=3, field="pgf_y", probe_gid=gid)
    assert_close(pgf_fields[1][gid - 1], rec, kind="gather")


def test_pgf_zero_below_bottom(mesh, pgf_fields):
    """PGF is a layer field: zero wherever ``elem_layer_mask`` is False."""
    invalid = ~np.asarray(mesh.elem_layer_mask)
    assert np.all(pgf_fields[0][invalid] == 0.0)
    assert np.all(pgf_fields[1][invalid] == 0.0)


def test_pgf_gradient_flows(mesh):
    """AD discipline: a scalar through EOS→PGF differentiates w.r.t. T and matches
    a central finite difference (PGF is linear in hpressure → smooth)."""
    st = ic.initial_state(mesh)
    S, hnode = st.S, st.hnode
    emask = mesh.elem_layer_mask
    nvalid = jnp.sum(emask)

    def loss(T):
        _, hpressure, _ = eos.pressure_bv(mesh, T, S, hnode)
        px, py = pgf.pressure_force_linfs(mesh, hpressure)
        return jnp.sum(jnp.where(emask, px + py, 0.0)) / nvalid

    g_ad = float(np.asarray(jax.grad(loss)(st.T))[1000, 5])
    h = 1e-6
    g_fd = float((loss(st.T.at[1000, 5].add(h)) - loss(st.T.at[1000, 5].add(-h))) / (2 * h))
    assert np.isfinite(g_ad)
    assert abs(g_ad - g_fd) <= 1e-9 + 1e-6 * abs(g_fd), f"AD {g_ad:.3e} vs FD {g_fd:.3e}"
