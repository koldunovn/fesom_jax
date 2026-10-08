"""Surface-flux adjustment seam (``Params.surface_adj``) on CORE2.

One assembled CORE2 step (zstar + TKE + GM, JRA55-do forcing) from the PHC initial state with
TKE and sea ice seeded, on both surface branches: the prognostic sea-ice step with the
prescribed-ice adjoint (the production branch) and the static-ice (no-ice-model) branch.

Gates: ``surface_adj=None`` and ``{}`` trace the identical program; all-zero adjustments give
the same values up to round-off at field scale (a different compiled program; see the test); a positive heat adjustment in a box raises the box's surface
temperature by about ``dt·heat/(VCPW·h)``; a stress adjustment changes the surface TKE and
the velocity; the gradient of the box-mean SST w.r.t. the heat adjustment is finite and
positive in the box.

Heavy (CPU compute node, ≥ 200 GB): per branch two forward programs and one reverse program.
The data-free checks of the seam helper live in ``test_surface_adj_unit.py`` (ci).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ice as icemod
from fesom_jax import ssh, surface_forcing
from fesom_jax import step as stepmod
from fesom_jax.ale import AleConfig
from fesom_jax.config import VCPW
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0
YEAR = 1958
TKE_SEED = 1.0e-3
HEAT = 200.0          # W/m², into the ocean
STRESS = 0.1          # N/m²
BRANCHES = {"ice_prescribed": IceConfig(whichEVP=1, adjoint_mode="prescribed"),
            "no_ice": None}

core2_only = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file()),
    reason="CORE2 mesh / PHC IC missing")


def _step_fn(ice_cfg):
    def f(st, p, mesh, op, sf, fs):
        return stepmod.step(st, mesh, op, None, p, dt=DT, is_first_step=True,
                            step_forcing=sf, forcing_static=fs, ale_cfg=AleConfig(),
                            gm_cfg=GMConfig(), tke_cfg=TkeConfig(), ice_cfg=ice_cfg)
    return f


@pytest.fixture(scope="module")
def setup():
    if not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file()):
        pytest.skip("CORE2 mesh / PHC IC missing")
    mesh = load_mesh(MESH_DIR)
    bare = phc_initial_state(mesh, IC_DIR)
    imask = np.asarray(mesh.node_iface_mask)
    st = dataclasses.replace(
        bare, tke=jnp.asarray(np.where(imask, TKE_SEED, 0.0), jnp.float64))
    st = icemod.seed_ice(st, mesh, st.T[:, 0])
    assert float(jnp.max(st.tke)) > 0.0 and int(np.count_nonzero(np.asarray(st.a_ice))) > 0
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(st.T[:, 0]))
    sf = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
    # box: tropical central Pacific, deep, ice-free
    lon, lat = (np.degrees(np.asarray(mesh.geo_coord_nod2D[:, i])) for i in (0, 1))
    lon = (lon + 180.0) % 360.0 - 180.0
    box = ((lat > -10.0) & (lat < 10.0) & (lon > -150.0) & (lon < -120.0)
           & (np.asarray(mesh.nlevels_nod2D) >= 20))
    assert box.sum() >= 10, f"box has {box.sum()} nodes"
    assert not np.any(np.asarray(st.a_ice)[box] > 0.0)
    return SimpleNamespace(mesh=mesh, op=op, sf=sf, fs=cf.static, st=st,
                           box=box, box_j=jnp.asarray(box))


def _params(adj):
    return dataclasses.replace(Params.defaults(), surface_adj=adj)


def _full_adj(setup, heat=0.0, sx=0.0, sy=0.0):
    b = setup.box.astype(np.float64)
    return {"heat": jnp.asarray(heat * b), "stress_x": jnp.asarray(sx * b),
            "stress_y": jnp.asarray(sy * b)}


_FWD: dict = {}


def _fwd(setup, branch):
    if branch not in _FWD:
        _FWD[branch] = jax.jit(_step_fn(BRANCHES[branch]))
    return _FWD[branch]


def _run(setup, branch, params):
    s = setup
    return jax.device_get(_fwd(setup, branch)(s.st, params, s.mesh, s.op, s.sf, s.fs))


_RESULTS: dict = {}


def _result(setup, branch, key):
    """Cached forward results: 'none', 'zero', 'heat', 'stress'."""
    k = (branch, key)
    if k not in _RESULTS:
        adj = {"none": None, "zero": _full_adj(setup), "heat": _full_adj(setup, heat=HEAT),
               "stress": _full_adj(setup, sx=STRESS, sy=STRESS)}[key]
        _RESULTS[k] = _run(setup, branch, _params(adj))
    return _RESULTS[k]


# --------------------------------------------------------------------------
# CORE2, both surface branches
# --------------------------------------------------------------------------
@core2_only
def test_the_two_surface_branches_differ(setup):
    """Control for the parametrisation: the ice and no-ice branches are different programs
    (they differ at the ice nodes), so a test passing on both covers both."""
    a, b = _result(setup, "ice_prescribed", "none"), _result(setup, "no_ice", "none")
    assert not np.array_equal(np.asarray(a.T[:, 0]), np.asarray(b.T[:, 0]))


@core2_only
@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_none_and_empty_trace_the_identical_program(setup, branch):
    s = setup
    fn = _step_fn(BRANCHES[branch])

    def text(p):
        return str(jax.make_jaxpr(fn)(s.st, p, s.mesh, s.op, s.sf, s.fs))
    ref = text(Params.defaults())
    assert text(_params({})) == ref
    assert text(_params({"heat": jnp.zeros(s.mesh.nod2D)})) != ref        # control


ROUNDOFF_ULPS = 16


@core2_only
@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_zero_adjustment_gives_the_same_values(setup, branch, capsys):
    """All-zero adjustments give the None values up to round-off at each field's scale.

    ⚠️ Not bitwise, and not because of the seam's arithmetic (adding 0.0 is exact): the
    program with the seam is a DIFFERENT compiled program, and XLA:CPU's fusion (with LLVM
    FMA contraction) rounds the unchanged producers differently. Measured on this fixture
    (jobs 27736152/3, 27736180): T differs at ~500-650 surface-affected nodes by at most
    3.6e-15 K (one ulp at 30 °C) and tke at ~9500 values by at most 1.7e-18; every other
    field is bitwise; three algebraically different seam forms (add, recompute bc_T from the
    adjusted heat flux, optimization barrier before the add) give identical differences,
    while two equivalent programs WITHOUT the seam (params as arguments vs closed over) are
    bitwise. The None path itself is bitwise the pre-seam program (jaxpr and State sha256
    compared with the parent commit, jobs 27736109/27736110). The bound is
    ``ROUNDOFF_ULPS · eps · max|field|`` per field."""
    a, b = _result(setup, branch, "none"), _result(setup, branch, "zero")
    eps = np.finfo(np.float64).eps
    moved = {}
    for f in dataclasses.fields(a):
        x, y = np.asarray(getattr(a, f.name)), np.asarray(getattr(b, f.name))
        if np.array_equal(x, y):
            continue
        assert x.dtype.kind == "f", f"State.{f.name} (non-float) moved under a zero surface_adj"
        d, scale = float(np.abs(x - y).max()), float(np.abs(x).max())
        moved[f.name] = (int(np.count_nonzero(x != y)), d, d / (eps * scale))
        assert d <= ROUNDOFF_ULPS * eps * scale, (
            f"State.{f.name} moved by {d:.3e} under a zero surface_adj "
            f"(> {ROUNDOFF_ULPS} ulp of max|field| = {scale:.3e})")
    with capsys.disabled():
        print(f"\n  [{branch}] zero surface_adj vs None: " + (", ".join(
            f"{k}: {n} values, max {d:.2e} ({u:.2f} ulp of max|field|)"
            for k, (n, d, u) in moved.items()) or "bitwise"))


@core2_only
@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_positive_heat_warms_the_box(setup, branch, capsys):
    s = setup
    ref, got = _result(setup, branch, "none"), _result(setup, branch, "heat")
    dT = np.asarray(got.T[:, 0]) - np.asarray(ref.T[:, 0])
    expect = DT * HEAT / (VCPW * np.asarray(s.st.hnode[:, 0]))
    ratio = dT[s.box].mean() / expect[s.box].mean()
    with capsys.disabled():
        print(f"\n  [{branch}] box SST change {dT[s.box].mean():.4e} K "
              f"(min {dT[s.box].min():.3e}), dt·Q/(VCPW·h) {expect[s.box].mean():.4e} K, "
              f"ratio {ratio:.3f}; max |dT| outside {np.abs(dT[~s.box]).max():.3e}")
    assert np.all(dT[s.box] > 0.0), "a positive (into the ocean) heat adjustment cooled a box node"
    assert 0.1 < ratio < 1.5
    assert np.abs(dT[~s.box]).max() < dT[s.box].mean()


@core2_only
@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_stress_changes_the_surface_tke_and_velocity(setup, branch, capsys):
    s = setup
    ref, got = _result(setup, branch, "none"), _result(setup, branch, "stress")
    dtke = np.asarray(got.tke[:, 0]) - np.asarray(ref.tke[:, 0])
    duv = np.abs(np.asarray(got.uv[:, 0, :]) - np.asarray(ref.uv[:, 0, :])).max()
    frac = np.mean(dtke[s.box] != 0.0)
    with capsys.disabled():
        print(f"\n  [{branch}] surface tke change in box: fraction changed {frac:.2f}, "
              f"mean {dtke[s.box].mean():.3e}; max |d uv(k=0)| {duv:.3e}")
    assert frac > 0.9, "the stress adjustment did not reach the TKE surface forcing"
    assert duv > 0.0


@core2_only
@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_heat_gradient_is_finite_and_positive_in_the_box(setup, branch, capsys):
    s = setup
    fn = _step_fn(BRANCHES[branch])
    w = jnp.where(s.box_j, 1.0, 0.0) / float(s.box.sum())

    def loss(heat, st, mesh, op, sf, fs):
        out = fn(st, _params({"heat": heat}), mesh, op, sf, fs)
        return jnp.sum(w * out.T[:, 0])

    g = np.asarray(jax.jit(jax.grad(loss))(jnp.zeros(s.mesh.nod2D), s.st, s.mesh, s.op,
                                           s.sf, s.fs))
    expect = DT / (VCPW * np.asarray(s.st.hnode[:, 0])) / s.box.sum()
    with capsys.disabled():
        print(f"\n  [{branch}] dJ/dheat in box: mean {g[s.box].mean():.4e} "
              f"(dt/(VCPW·h·n) {expect[s.box].mean():.4e}); max |g| outside "
              f"{np.abs(g[~s.box]).max():.3e}")
    assert np.all(np.isfinite(g))
    assert np.all(g[s.box] > 0.0)
    assert 0.1 < g[s.box].mean() / expect[s.box].mean() < 1.5
