"""``fesom_jax.adjoint_costs`` on a CORE2 state (the 2013-01-01 restart of the hindcast).

Weighted means (uniform field ⇒ the field value; the weights are the live wet box volume; the
gradient of the box mean w.r.t. T is the normalised weights), the MLD field and its two
conventions where the column is mixed to the bottom (``"column"`` = the deepest wet mid-layer,
continuous at the valid→invalid switch, with the old bottom-interface value as the negative
control; ``"zero"`` bitwise the threshold routine), and the running accumulators over two
production-physics steps against the manual sums (bitwise, CPU).

Heavy (CPU compute node, ≥ 200 GB): one compiled CORE2 step and one two-step scan.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import adjoint_costs as AC
from fesom_jax import ale, obs_compare, ssh, surface_forcing
from fesom_jax import step as stepmod
from fesom_jax.adjoint import scan_steps
from fesom_jax.ale import AleConfig
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.state import State
from fesom_jax.tke import TkeConfig
from fesom_jax.zarr_output import _all_state_fields, reconstruct_global

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
RESTART = Path("/work/ab0995/a270088/port_jax/runs/core2_hindcast_v3/restart_archive/"
               "fesom.2013.001.00000")
DT = 1800.0
YEAR = 2013
ICE_CFG = IceConfig(whichEVP=1, adjoint_mode="prescribed")

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and RESTART.is_dir()), reason="CORE2 mesh / 2013 restart missing")


@pytest.fixture(scope="module")
def setup():
    mesh = load_mesh(MESH_DIR)
    st = State(**{f: jnp.asarray(reconstruct_global(RESTART, f)) for f in _all_state_fields()})
    assert st.T.shape == (mesh.nod2D, mesh.nl)
    lon = np.degrees(np.asarray(mesh.geo_coord_nod2D[:, 0]))
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    lat = np.degrees(np.asarray(mesh.geo_coord_nod2D[:, 1]))
    box = (lon >= -50.0) & (lon <= -10.0) & (lat >= 50.0) & (lat <= 65.0)
    levels = np.arange(mesh.nl) < 10
    assert box.sum() > 50
    return mesh, st, jnp.asarray(box), jnp.asarray(levels)


def test_box_mean_of_a_uniform_field(setup):
    mesh, st, box, lev = setup
    u = dataclasses.replace(st, T=jnp.full_like(st.T, 7.25))
    assert float(AC.box_mean_T_vw(u, mesh, box, lev)) == pytest.approx(7.25, rel=1e-14)
    assert float(AC.sst_area_mean(u, mesh, box)) == pytest.approx(7.25, rel=1e-14)
    e = dataclasses.replace(st, eta_n=jnp.full_like(st.eta_n, -0.3))
    assert float(AC.ssh_area_mean(e, mesh)) == pytest.approx(-0.3, rel=1e-14)


def test_box_weights_are_the_live_wet_box_volume(setup):
    mesh, st, box, lev = setup
    w = np.asarray(AC.box_volume_weights(st, mesh, box, lev))
    m = (np.asarray(mesh.node_layer_mask) & np.asarray(box)[:, None]
         & np.asarray(lev)[None, :])
    vol = float(np.sum(np.where(m, np.asarray(st.hnode) * np.asarray(mesh.areasvol), 0.0)))
    assert vol > 0.0
    assert float(w.sum()) == pytest.approx(vol, rel=1e-13)
    assert np.all(w[~m] == 0.0)
    # live thickness: scaling hnode scales the weights
    st2 = dataclasses.replace(st, hnode=st.hnode * 1.01)
    assert float(jnp.sum(AC.box_volume_weights(st2, mesh, box, lev))) == \
        pytest.approx(1.01 * vol, rel=1e-13)


def test_box_mean_gradient_is_the_normalised_weights(setup):
    mesh, st, box, lev = setup
    g = jax.grad(lambda T: AC.box_mean_T_vw(dataclasses.replace(st, T=T), mesh, box, lev))(st.T)
    w = AC.box_volume_weights(st, mesh, box, lev)
    ref = w / jnp.sum(w)
    assert float(jnp.max(jnp.abs(g - ref))) <= 1e-14 * float(jnp.max(ref))
    assert np.all(np.asarray(g)[np.asarray(w) == 0.0] == 0.0)


def test_mld_field_conventions(setup):
    mesh, st, box, _ = setup
    # mix three deep wet box columns to the bottom so the invalid branch is exercised
    nlev = np.asarray(mesh.nlevels_nod2D)
    cand = np.flatnonzero(np.asarray(box) & (nlev > 20))[:3]
    assert cand.size == 3
    lm = np.asarray(mesh.node_layer_mask)
    T, S = np.asarray(st.T).copy(), np.asarray(st.S).copy()
    for n in cand:
        T[n, lm[n]] = 4.0
        S[n, lm[n]] = 35.0
    mixed = dataclasses.replace(st, T=jnp.asarray(T), S=jnp.asarray(S))

    _, Z = ale.live_geometry(mesh, mixed.hnode)
    ref, valid = obs_compare.mld_density_threshold(mixed.T, mixed.S, Z,
                                                   jnp.asarray(mesh.node_layer_mask))
    valid = np.asarray(valid)
    assert not valid[cand].any()
    zero = np.asarray(AC.mld_field(mixed, mesh, invalid="zero"))
    col = np.asarray(AC.mld_field(mixed, mesh))
    assert np.array_equal(zero, np.asarray(ref))                  # D2a-compatible: bitwise
    assert np.array_equal(col[valid], zero[valid])
    deep = np.asarray(AC.deepest_wet_mid_depth(mixed, mesh))
    wet = lm[:, 0]
    inv = wet & ~valid
    assert np.array_equal(col[inv], deep[inv]) and np.all(zero[inv] == 0.0)
    assert np.all(col[~wet] == 0.0)
    # the datum: the deepest wet mid-layer is the last True of node_layer_mask, on the same live
    # Z_3d_n the threshold routine interpolates on
    kd = lm.shape[1] - 1 - np.argmax(lm[:, ::-1], axis=1)
    assert np.array_equal(kd[wet], (nlev - 2)[wet])
    assert np.array_equal(deep[wet], -np.asarray(Z)[wet, kd[wet]])
    # no valid crossing is deeper than it (on the whole restart, not only the mixed columns;
    # the interpolation z_hi + 1·(z_lo - z_hi) may round by an ulp)
    assert valid.sum() > 1000
    assert np.all(zero[valid] <= deep[valid] * (1.0 + 4e-16))
    # the bottom interface lies below the deepest wet mid-layer (the old invalid value)
    bottom = np.asarray(AC.column_bottom_depth(mixed, mesh))
    kdeep = nlev[cand] - 2
    assert np.all(bottom[cand] > -np.asarray(Z)[cand, kdeep])
    assert np.all(np.abs(bottom[cand] + np.asarray(mesh.zbar_3d_n)[cand, nlev[cand] - 1])
                  <= 1e-6 * bottom[cand])
    with pytest.raises(ValueError, match="invalid"):
        AC.mld_field(mixed, mesh, invalid="nan")


# The continuity sweep: one deep box column, T = 4 °C and S = 35 psu on every wet layer except
# the deepest, whose salinity is 35 + s. σ is referenced to the surface, so the σ-excess at the
# deepest layer is a function of s alone and crosses 0 at s = s0 (found by bisection below);
# every shallower layer below ref_depth has excess ≈ -dsigma, so the only possible crossing is
# between the last two wet layers and its weight is e / (e + dsigma) short of 1 — the MLD's gap
# to the deepest wet mid-layer is ≈ e / (e + dsigma) · Δz, linear in s - s0.
SWEEP = (0.5, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8, -1e-8, -1e-4, -0.5)     # (s - s0) / s0


def _excess(s):
    T = jnp.full((1, 2), 4.0)
    S = jnp.asarray([[35.0, 35.0 + s]])
    sig = obs_compare.potential_density(T, S, jnp.ones((1, 2), bool))
    return float(sig[0, 1] - sig[0, 0]) - 0.03


def _bisect_s0():
    lo, hi = 0.0, 1.0                              # excess(0) = -0.03 < 0, excess(1) > 0
    assert _excess(lo) < 0.0 < _excess(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        lo, hi = (mid, hi) if _excess(mid) < 0.0 else (lo, mid)
    return hi


@pytest.fixture(scope="module")
def sweep(setup):
    """MLD of the sweep column for every offset in SWEEP, under the new and the old rule."""
    mesh, st, box, _ = setup
    nlev = np.asarray(mesh.nlevels_nod2D)
    n = int(np.flatnonzero(np.asarray(box) & (nlev > 20))[0])
    kd = int(nlev[n]) - 2
    lm = np.asarray(mesh.node_layer_mask)[n]
    s0 = _bisect_s0()

    @jax.jit
    def mlds(T, S):
        s = dataclasses.replace(st, T=T, S=S)
        _, Z = ale.live_geometry(mesh, s.hnode)
        m0, valid = obs_compare.mld_density_threshold(T, S, Z, jnp.asarray(mesh.node_layer_mask))
        old = jnp.where(valid, m0, AC.column_bottom_depth(s, mesh))   # the rule before the fix
        return AC.mld_field(s, mesh), old, valid

    rows = []
    for d in SWEEP:
        T, S = np.asarray(st.T).copy(), np.asarray(st.S).copy()
        T[n, lm] = 4.0
        S[n, lm] = 35.0
        S[n, kd] = 35.0 + s0 * (1.0 + d)
        new, old, valid = mlds(jnp.asarray(T), jnp.asarray(S))
        rows.append((d, float(new[n]), float(old[n]), bool(valid[n])))
    _, Z = ale.live_geometry(mesh, st.hnode)
    Zn = np.asarray(Z)[n]
    return dict(rows=rows, deep=-Zn[kd], dz=Zn[kd - 1] - Zn[kd], h_bottom=float(st.hnode[n, kd]),
                s0=s0, n=n, kd=kd)


def _assert_continuous_at_switch(rows, deep, dz):
    """The valid MLD approaches the invalid value as the crossing deepens to the last wet mid-layer
    (gap ≈ d/(1+d)·Δz), and the invalid value equals that limit."""
    val = [(d, m) for d, m, _, v in rows if v]
    inv = [m for d, m, _, v in rows if not v]
    assert [d > 0 for d, _, _, _ in rows] == [v for _, _, _, v in rows], "sweep sign != validity"
    inv_value = inv[0]
    assert all(m == inv_value for m in inv), "invalid value depends on how far from the switch"
    gaps = [(d, inv_value - m) for d, m in val]
    for (d1, g1), (d2, g2) in zip(gaps, gaps[1:]):
        assert g2 < g1, f"gap to the invalid value does not shrink as the crossing deepens: {gaps}"
    for d, g in gaps:
        lin = d / (1.0 + d) * dz
        assert 0.5 * lin <= g <= 2.0 * lin, f"gap {g} m at d={d}, expected ≈ {lin} m (Δz {dz} m)"
    return gaps


def test_mld_column_is_continuous_at_the_switch(sweep, capsys):
    rows, deep, dz = sweep["rows"], sweep["deep"], sweep["dz"]
    new_rows = [(d, m, None, v) for d, m, _, v in rows]
    gaps = _assert_continuous_at_switch(new_rows, deep, dz)
    assert [m for _, m, _, v in rows if not v][0] == deep       # the invalid value is the datum
    with capsys.disabled():
        print(f"\n[mld continuity] node {sweep['n']} kd {sweep['kd']} s0 {sweep['s0']:.6e} psu, "
              f"deepest mid {deep:.3f} m, Δz {dz:.3f} m; gap by (s-s0)/s0: "
              + ", ".join(f"{d:g}: {g:.3e} m" for d, g in gaps))
    assert gaps[-1][1] <= 2e-8 * dz                             # d = 1e-8: gap ≈ Δz·1e-8


def test_mld_column_bottom_interface_fails_continuity(sweep):
    """Negative control: the old invalid value (the bottom interface) fails the continuity test,
    jumping by half the bottom layer at the switch."""
    rows = sweep["rows"]
    old_rows = [(d, o, None, v) for d, _, o, v in rows]
    with pytest.raises(AssertionError):
        _assert_continuous_at_switch(old_rows, sweep["deep"], sweep["dz"])
    last_valid = [o for d, _, o, v in rows if v][-1]
    first_inv = [o for d, _, o, v in rows if not v][0]
    assert first_inv - last_valid == pytest.approx(0.5 * sweep["h_bottom"], rel=1e-6)


def _step_fn(mesh, op, fs):
    def f(state, x):
        return stepmod.step(state, mesh, op, None, Params.defaults(), dt=DT,
                            is_first_step=False, step_forcing=x, forcing_static=fs,
                            ale_cfg=AleConfig(), gm_cfg=GMConfig(), tke_cfg=TkeConfig(),
                            ice_cfg=ICE_CFG)
    return f


@pytest.fixture(scope="module")
def two_steps(setup):
    mesh, st, box, lev = setup
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(st.T[:, 0]))
    xs = cf.stack(surface_forcing.dates_for_steps(YEAR, DT, 2))
    step_fn = _step_fn(mesh, op, cf.static)
    accs = {"box_T": lambda s: AC.box_mean_T_vw(s, mesh, box, lev),
            "mld": lambda s: AC.mld_field(s, mesh)}
    return step_fn, accs, xs


def _x(xs, i):
    return jax.tree.map(lambda a: a[i], xs)


def test_running_accumulators_over_two_steps(setup, two_steps, capsys):
    """Scalar (box mean T) and field (MLD) accumulators over two production-physics steps.

    Bitwise: the wrapped step applied twice equals the manual sum of the two post-step values
    computed in the same program (the accumulator adds exactly ``fn(new_state)`` after every
    step, nothing for the initial state). Through ``scan_steps`` (the chunk-function path) the
    sums come from a different compiled program, and two compiled programs of this model differ
    by round-off amplified by the early-stopped SSH solve; they must agree to within 10× the
    measured spread of two equivalent programs (the final-state costs of a scan of the plain
    step against the unrolled program)."""
    mesh, st, _, _ = setup
    step_fn, accs, xs = two_steps
    acc0 = AC.init_accumulators(accs, st)
    assert acc0["box_T"].shape == () and acc0["mld"].shape == (mesh.nod2D,)
    assert float(acc0["box_T"]) == 0.0 and float(jnp.max(jnp.abs(acc0["mld"]))) == 0.0
    wrapped = AC.with_running_costs(step_fn, accs)

    @jax.jit
    def unrolled(s0, xs_):
        c1 = wrapped((s0, AC.init_accumulators(accs, s0)), _x(xs_, 0))
        c2 = wrapped(c1, _x(xs_, 1))
        man = {k: fn(c1[0]) + fn(c2[0]) for k, fn in accs.items()}
        return c2, man

    (s2, acc), man = unrolled(st, xs)
    assert float(acc["box_T"]) == float(man["box_T"])
    assert np.array_equal(np.asarray(acc["mld"]), np.asarray(man["mld"]))
    assert float(jnp.max(acc["mld"])) > 0.0

    # the chunk-function path, and the spread of two equivalent programs of the same two
    # steps: a scan of the plain step against the unrolled program above, measured on the
    # final-state costs
    s_scan, acc_scan = jax.jit(lambda c, x: scan_steps(wrapped, c, x, rematerialize=False))(
        (st, acc0), xs)
    s_plain = jax.jit(lambda s, x: scan_steps(step_fn, s, x, rematerialize=False))(st, xs)
    fin_unr = {k: jax.jit(fn)(s2) for k, fn in accs.items()}
    fin_plain = {k: jax.jit(fn)(s_plain) for k, fn in accs.items()}

    def box_rel(a, b):
        return abs(float(a["box_T"]) - float(b["box_T"])) / abs(float(b["box_T"]))

    def mld_abs(a, b):
        return float(jnp.max(jnp.abs(a["mld"] - b["mld"])))

    spread_box, spread_mld = box_rel(fin_plain, fin_unr), mld_abs(fin_plain, fin_unr)
    d_box, d_mld = box_rel(acc_scan, acc), mld_abs(acc_scan, acc)
    with capsys.disabled():
        print(f"\n[accumulators] scan vs unrolled sums: box_T rel {d_box:.2e}, mld max|diff| "
              f"{d_mld:.2e} m, T max|diff| {float(jnp.max(jnp.abs(s_scan.T - s2.T))):.2e}; "
              f"spread (final-state cost, plain-step scan vs unrolled): box_T rel "
              f"{spread_box:.2e}, mld {spread_mld:.2e} m, T "
              f"{float(jnp.max(jnp.abs(s_plain.T - s2.T))):.2e}; scan wrapped vs plain T "
              f"bitwise {bool(np.array_equal(np.asarray(s_scan.T), np.asarray(s_plain.T)))}")
    # the accumulators ride along: the model trajectory of the scan is the plain one
    assert np.array_equal(np.asarray(s_scan.T), np.asarray(s_plain.T))
    # a plumbing error (a missed step, the initial state added) would be O(the value itself)
    assert d_box <= 10 * max(spread_box, 1e-16)
    assert d_mld <= 10 * max(spread_mld, 1e-12)

    # losses on the carry
    carry = (s2, acc)
    box_j = jax.jit(accs["box_T"])
    assert float(AC.terminal(accs["box_T"])(carry)) == float(box_j(s2))
    assert float(AC.terminal(accs["box_T"])(s2)) == float(box_j(s2))
    assert float(AC.running_mean("box_T", 2)(carry)) == float(acc["box_T"]) / 2
