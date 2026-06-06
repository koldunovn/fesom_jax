"""Task 2.11 gate — the assembled forward step on pi (GATE 2).

* **Step-1 integration:** one `step()` reproduces every per-kernel substep dump gate
  at the probes (density/bvfreq/ssh_rhs/d_eta/uv/hbar/eta_n/w/hnode/T/S all tight —
  **T is now tight too** since Phase 4 wired FCT in). Confirms the substep order +
  step-1 state are wired correctly.
* **Rest state:** constant T/S (no blob) + zero wind ⇒ the model stays at rest to
  machine precision (no spurious flow), T/S exactly constant.
* **Multi-step history threading:** S stays exactly 35 over many steps (constant-tracer
  preservation ⇒ AB2/threading is correct); the physical SSH/velocity fields stay
  climate-close to the dump; the **CG warm-start is load-bearing** (step-2 d_eta matches
  the dump far better warm-started than from zero — the C never zeros d_eta between steps).
* **100-step stability:** no NaN, bounded `|uv|`/`|eta|`, S exactly constant.

The Phase-4 FCT port (+ the IC `T_old` fix — `valuesold` is the pre-blob base, not the
blob) closed the Phase-2 upwind−FCT `T` gap, so step-1 `T` (and the step-2 fields it
cascades into) now match the dump tightly. The jitted-step `density` FMA shift (~1e-13)
means the per-step tight bit-gates run on **eager** `step()`.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import eos, forcing, ic, momentum, pgf, pp, ssh, verify
from fesom_jax import step as stepmod
from fesom_jax.io_dump import find_record
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params
from fesom_jax.state import State

NODE_PROBES = [1001, 1500, 2000, 2500, 3000]
ELEM_PROBES = [1757, 2656, 3688, 4604, 5575]
DT = 100.0

# (state field, dump substep, dump field, kind, atol) — the per-kernel step-1 gates,
# re-checked through the integrated step(). atols mirror the per-kernel tests.
NODE_GATES = [
    ("density", 1, "density", "map", None),
    ("bvfreq", 1, "bvfreq", "scatter", None),
    ("ssh_rhs", 8, "ssh_rhs", "scatter", 1e-7),
    ("d_eta", 9, "d_eta", "scatter", None),
    ("hbar", 11, "hbar", "scatter", 1e-12),
    ("eta_n", 12, "eta_n", "scatter", 1e-12),
    ("w", 13, "w", "scatter", 1e-12),
    ("hnode", 16, "hnode", "map", None),
    ("T", 15, "T", "scatter", None),
    ("S", 15, "S", "scatter", None),
]


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (Task 0.3)")
    return load_mesh()


@pytest.fixture(scope="module")
def op(mesh):
    return ssh.build_ssh_operator(mesh, dt=DT)


@pytest.fixture(scope="module")
def stress(mesh):
    return forcing.surface_stress(mesh)


@pytest.fixture(scope="module")
def traj(mesh, op, stress):
    """States after steps 1..3 of the dump config (blob + analytical wind)."""
    states = []
    st = ic.initial_state(mesh)
    for i in range(3):
        st = stepmod.step(st, mesh, op, stress, dt=DT, is_first_step=(i == 0))
        states.append(st)
    return states


# --------------------------------------------------------------------------
# 1. step-1 integration: end-of-step state == the per-kernel dump gates
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field,substep,dfield,kind,atol", NODE_GATES)
@pytest.mark.parametrize("gid", NODE_PROBES)
def test_step1_node_fields_match_dump(load_dump, mesh, traj, gid, field, substep,
                                      dfield, kind, atol):
    arr = np.asarray(getattr(traj[0], field))
    col = arr[gid - 1] if arr.ndim == 2 else arr[gid - 1:gid]
    rec = find_record(load_dump("pi_cdump.00000"), step=1, substep=substep,
                      field=dfield, probe_gid=gid)
    verify.assert_close(col, rec, kind=kind, atol=atol)


@pytest.mark.parametrize("gid", ELEM_PROBES)
@pytest.mark.parametrize("dfield,ci", [("uv_u", 0), ("uv_v", 1)])
def test_step1_uv_matches_dump(load_dump, mesh, traj, gid, dfield, ci):
    rec = find_record(load_dump("pi_cdump.00000"), step=1, substep=10,
                      field=dfield, probe_gid=gid)
    verify.assert_close(np.asarray(traj[0].uv)[gid - 1, :, ci], rec, kind="gather",
                        atol=1e-13)


def test_step1_T_matches_dump_tight(load_dump, mesh, traj):
    """T (substep 15) matches the FCT dump **tightly** (Phase-4: FCT is now the port's
    advection, so the old upwind−FCT gap is closed). Eager `step()` lands ~1.8e-15."""
    recs = load_dump("pi_cdump.00000")
    T = np.asarray(traj[0].T)
    worst = 0.0
    for gid in NODE_PROBES:
        rec = find_record(recs, step=1, substep=15, field="T", probe_gid=gid)
        n = rec.nlevels
        worst = max(worst, np.abs(T[gid - 1, :n] - np.asarray(rec.values)[:n]).max())
    assert worst < 1e-11, f"FCT T step-1 |Δ|={worst:.2e} (expected tight)"


# --------------------------------------------------------------------------
# 2. rest state — constant T/S + zero wind stays at rest to machine precision
# --------------------------------------------------------------------------
def test_rest_state_stays_at_rest(mesh, op):
    rest = State.rest(mesh, T0=10.0, S0=35.0)
    zero_stress = jnp.zeros((mesh.elem2D, 2))
    st = rest
    for i in range(5):
        st = stepmod.step(st, mesh, op, zero_stress, dt=DT, is_first_step=(i == 0))
    m = np.asarray(mesh.node_layer_mask)
    assert np.max(np.abs(np.asarray(st.uv))) < 1e-12
    assert np.max(np.abs(np.asarray(st.eta_n))) < 1e-12
    assert np.max(np.abs(np.asarray(st.d_eta))) < 1e-12
    assert np.max(np.abs(np.asarray(st.T)[m] - 10.0)) == 0.0
    assert np.max(np.abs(np.asarray(st.S)[m] - 35.0)) == 0.0


# --------------------------------------------------------------------------
# 3. multi-step: S exact, climate-close to dump, warm-start load-bearing
# --------------------------------------------------------------------------
def test_salinity_exactly_preserved_multistep(mesh, traj):
    """S stays exactly 35 on wet layers across steps — constant-tracer preservation
    AND a sensitive check that the AB2 / history threading is correct (a bug would
    corrupt it)."""
    m = np.asarray(mesh.node_layer_mask)
    for st in traj:
        assert np.max(np.abs(np.asarray(st.S)[m] - 35.0)) == 0.0


def test_step2_physical_fields_match_dump(load_dump, mesh, traj):
    """At step 2 the SSH fields match the dump **tightly** — with FCT the step-1 `T`
    matches (1.8e-15), so density at step 2 matches and the SSH solve no longer carries
    the old upwind−FCT cascade. (d_eta/hbar/eta_n are the ÷area-suppressed tight class.)"""
    recs = load_dump("pi_cdump.00000")
    st2 = traj[1]
    for field, dfield in [("d_eta", "d_eta"), ("hbar", "hbar"), ("eta_n", "eta_n")]:
        arr = np.asarray(getattr(st2, field))
        worst = 0.0
        for gid in NODE_PROBES:
            rec = find_record(recs, step=2, substep={"d_eta": 9, "hbar": 11,
                              "eta_n": 12}[dfield], field=dfield, probe_gid=gid)
            worst = max(worst, abs(arr[gid - 1] - rec.values[0]))
        assert worst < 1e-11, f"{field} step-2 |Δ|={worst:.2e}"


def test_step2_uv_rhs_visc_matches_dump(load_dump, mesh, traj):
    """Substep-6 (biharmonic viscosity) `uv_rhs` at **step 2** — an element field the
    step-1 gate can only check at rest (uv=0 ⇒ viscosity adds nothing). With FCT making
    the multi-step trajectory tight, step 2 has a real wind-driven velocity field, so this
    is the first end-to-end dump gate on opt_visc=7 acting on nonzero flow.

    pi velocities are small (|du_edge| ≤ ~1e-4 at step 2), so this exercises the
    *constant-coefficient* biharmonic regime (`max(g0, inner)=g0`); the flow-aware g1/g2
    branches are verified separately against the numpy reference at strong synthetic flow
    (`test_momentum.test_visc_filter_flow_aware_branches_vs_reference`)."""
    recs = load_dump("pi_cdump.00000")
    st1 = traj[0]                                   # state after step 1 = input to step 2
    pr = Params.defaults()
    _, hpressure, _ = eos.compute_pressure_bv(mesh, st1.T, st1.S, st1.hnode)
    pgf_x, pgf_y = pgf.pressure_force_linfs(mesh, hpressure)
    uv_rhs, _ = momentum.compute_vel_rhs(
        mesh, st1.uv, st1.uv_rhsAB, st1.eta_n, pgf_x, pgf_y, st1.w_e, st1.hnode,
        is_first_step=False, dt=DT)
    uv_rhs6 = np.asarray(momentum.visc_filt_bidiff(mesh, st1.uv, uv_rhs, dt=DT))
    pre = np.asarray(uv_rhs)
    changed, worst = 0.0, 0.0
    for gid in ELEM_PROBES:
        for dfield, ci in [("uv_rhs_u", 0), ("uv_rhs_v", 1)]:
            rec = find_record(recs, step=2, substep=6, field=dfield, probe_gid=gid)
            n = rec.nlevels
            worst = max(worst, np.abs(uv_rhs6[gid - 1, :n, ci] - rec.values[:n]).max())
            changed = max(changed, np.abs(uv_rhs6[gid - 1, :n, ci]
                                          - pre[gid - 1, :n, ci]).max())
    assert worst < 1e-12, f"step-2 substep-6 uv_rhs |Δ|={worst:.2e}"
    assert changed > 0.0, "viscosity made no change to uv_rhs at step 2"


def test_warm_start_is_load_bearing(load_dump, mesh, op, traj):
    """The CG warm-start (x0 = previous d_eta) is load-bearing: step-2 d_eta matches
    the dump markedly better warm-started than solved from zero — the C never zeros
    d_eta between steps."""
    recs = load_dump("pi_cdump.00000")
    st2 = traj[1]
    d_ws = np.asarray(st2.d_eta)
    d_no = np.asarray(ssh.solve_ssh(op, st2.ssh_rhs, x0=None))   # no warm start
    # at the cancellation nodes the warm start is orders better
    for gid in [1500, 2000]:
        cv = find_record(recs, step=2, substep=9, field="d_eta", probe_gid=gid).values[0]
        assert abs(d_ws[gid - 1] - cv) < 0.05 * abs(d_no[gid - 1] - cv)


# --------------------------------------------------------------------------
# 4. 100-step stability
# --------------------------------------------------------------------------
def test_step_jit_matches_eager(mesh, op, stress, traj):
    """The jitted step equals the eager step to FMA-fusion level (~1e-12) — XLA may
    contract the EOS polynomial to FMAs, so density shifts ~1e-13 from the eager
    bit-exact value (still climate-close; the tight per-kernel gates use eager)."""
    st0 = ic.initial_state(mesh)
    j = stepmod.step_jit(st0, mesh, op, stress, dt=DT, is_first_step=True)
    e = traj[0]
    for f in ["uv", "eta_n", "d_eta", "hbar", "w", "density", "bvfreq", "T", "S"]:
        d = np.max(np.abs(np.asarray(getattr(j, f)) - np.asarray(getattr(e, f))))
        assert d < 1e-11, f"{f}: jit−eager {d:.2e}"


def test_100_step_stability(mesh, op, stress):
    st = stepmod.run(ic.initial_state(mesh), mesh, op, stress, 100, dt=DT)
    uv, eta = np.asarray(st.uv), np.asarray(st.eta_n)
    T, S = np.asarray(st.T), np.asarray(st.S)
    m = np.asarray(mesh.node_layer_mask)
    assert not (np.isnan(uv).any() or np.isnan(eta).any() or np.isnan(T).any()
                or np.isnan(S).any())
    assert np.max(np.abs(uv)) < 0.5          # gentle wind ⇒ well below the ~0.3 CFL cap
    assert np.max(np.abs(eta)) < 5.0         # |eta| < 5 m
    assert np.max(np.abs(S[m] - 35.0)) == 0.0   # S exactly constant
    assert 9.0 < T[m].min() and T[m].max() < 20.0   # T bounded (blob ~+5°C)


def test_1000_step_stability(mesh, op, stress):
    """Task 4.3 / GATE 4: pi 1000 steps at dt=100 with the full pi physics (FCT +
    opt_visc7 + the wsplit machinery) stays stable — no NaN, bounded |uv|/|eta|, S
    **exactly** 35 over the whole window, T bounded. ~48 s (the jitted `run` amortizes
    compile across the 1000 steps). The vertical CFL stays ≪ maxcfl=1.0 throughout, so
    the (disabled) wsplit would be inactive even if turned on — consistent with the
    use_wsplit=0 reference config. The long-window AD-stability risk is tracked
    separately by `test_gradient.py` (run at a modest N to stay off the chaos floor)."""
    st = stepmod.run(ic.initial_state(mesh), mesh, op, stress, 1000, dt=DT)
    uv, eta = np.asarray(st.uv), np.asarray(st.eta_n)
    T, S, cfl = np.asarray(st.T), np.asarray(st.S), np.asarray(st.cfl_z)
    m = np.asarray(mesh.node_layer_mask)
    assert not (np.isnan(uv).any() or np.isnan(eta).any() or np.isnan(T).any()
                or np.isnan(S).any())
    assert np.max(np.abs(uv)) < 0.5             # well below the ~0.3 CFL cap
    assert np.max(np.abs(eta)) < 5.0            # |eta| < 5 m
    assert np.max(np.abs(S[m] - 35.0)) == 0.0   # S exactly constant over 1000 steps
    assert 9.0 < T[m].min() and T[m].max() < 20.0   # T bounded (blob ~+5°C)
    assert np.max(np.abs(cfl)) < 1.0            # CFL ≪ maxcfl ⇒ wsplit inactive (self-consistent)
