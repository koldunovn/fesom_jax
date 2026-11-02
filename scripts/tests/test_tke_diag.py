"""The TKE-adjoint diagnosis driver ``scripts/adjoint/tke_diag.py`` on the CORE2 reference model.

The model is the shared driver's (``scripts/adjoint/core2_adj.py``: the 2013-01-01 hindcast
restart, ``configs/core2_full.yaml`` physics — zstar, classical TKE, GM, mEVP — prescribed sea
ice), with 1-step chunks.

⚠️ HEAVY (CPU compute node, >= 200 GB): several CORE2 programs are compiled (the chunk forward,
the capture of the step's column inputs, the chunk VJP, the tangent chunk). The file lives in
the ``scripts`` group; run that group with ``--mem=200G``.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import core2_adj as C  # noqa: E402
import tke_diag as D  # noqa: E402

from fesom_jax import adjoint_check  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402

RESTART = C.RESTART_ARCHIVE / "fesom.2013.001.00000"
pytestmark = pytest.mark.skipif(
    not ((ROOT / "data" / "mesh_core2").is_dir() and RESTART.is_dir()),
    reason="CORE2 mesh / 2013 restart missing")

START = datetime.datetime(2013, 1, 1)


@pytest.fixture(scope="module")
def model():
    return C.build_model(START, n_steps=2, chunk_steps=1, restart=RESTART,
                         adjoint_cfg=AdjointConfig.exact(), ice_mode="prescribed")


@pytest.fixture(scope="module")
def pins(model):
    """The chunk-entry States of the 2-step window (the chunk forward the adjoint uses)."""
    return adjoint_check.forward_states(model.chunk_fn, model.params0, model.init_carry(), 2,
                                        model.xs_fn())


def _inputs_at(model, pins, n):
    st = model.s0 if n == 0 else pins[n]
    return D.step_tke_inputs(model, st, D.step_forcing(model, START, n))


@pytest.mark.parametrize("n", [0, 1])
def test_column_core_on_the_step_inputs_reproduces_the_step(model, pins, n):
    """The column core on the recorded inputs gives that step's ``tke_new`` bit for bit, at the
    first step (from the restart) and the second (from the model's own step-1 State), and the
    recorded ``uvnode`` is the one the step writes. The driver's chunk program (the same step
    inside a rematerialised scan) is another compilation: it agrees to round-off."""
    inp = _inputs_at(model, pins, n)
    tke_new = np.asarray(D.column_core(inp)[0])
    own = np.asarray(inp["new_state"].tke)
    nbad = int(np.sum(tke_new != own))
    assert nbad == 0, f"step {n}: {nbad} interfaces differ, max {np.max(np.abs(tke_new - own)):.3e}"
    assert np.array_equal(np.asarray(inp["uvnode"]), np.asarray(inp["new_state"].uvnode))
    ref = np.asarray(pins[n + 1].tke)
    spread = np.max(np.abs(own - ref))
    print(f"step {n}: step program vs chunk program: {int(np.sum(own != ref))} of {ref.size} "
          f"interfaces differ, max {spread:.3e} (max tke {np.max(ref):.3e})")
    assert spread <= 1e-12 * np.max(np.abs(ref))          # round-off, not a different input
    # the fixture is live: tke above the floor on part of the domain, not a cold column
    tmin = float(inp["tke_min"])
    assert np.sum(np.asarray(inp["tke_old"]) > 1.01 * tmin) > 1000


def test_recording_does_not_change_the_step(model):
    """The recording ``mixing_tke`` adds outputs, not arithmetic: the recorded program's new
    State equals a plain jitted step's bit for bit."""
    x = D.step_forcing(model, START, 0)
    inp = D.step_tke_inputs(model, model.s0, x)
    plain = jax.jit(lambda s, x, u: D.stepmod.step(
        s, model.mesh, model.op, None, model.to_params(u), dt=model.dt, is_first_step=False,
        step_forcing=x, forcing_static=model.forcing.static, adjoint_cfg=model.adjoint_cfg,
        **model.phys))(jax.device_put(model.s0), jax.device_put(x),
                       jax.device_put(model.params0))
    for f in ("tke", "T", "S", "uv", "eta_n"):
        assert np.array_equal(np.asarray(getattr(plain, f)),
                              np.asarray(getattr(inp["new_state"], f))), f


def test_a_planted_input_error_is_caught(model, pins):
    """Negative control: the forcing of the wrong step (6 h later) gives different column
    inputs and a different ``tke_new`` — the bitwise check above has resolving power."""
    good = _inputs_at(model, pins, 0)
    bad = D.step_tke_inputs(model, model.s0, D.step_forcing(model, START, 12))
    assert not np.array_equal(np.asarray(good["forc_tke_surf"]), np.asarray(bad["forc_tke_surf"]))
    assert not np.array_equal(np.asarray(D.column_core(bad)[0]),
                              np.asarray(good["new_state"].tke))


def test_regime_logger_runs_on_the_step_inputs(model, pins):
    """``tke_regimes`` on the model's own inputs: every wet interior interface gets a regime,
    the stiffness is finite everywhere, and the free and both wall-bound regimes occur (an
    interior ``floor`` length is rare: counted, not required)."""
    inp = _inputs_at(model, pins, 0)
    r = D.regimes(inp)
    interior = np.asarray(inp["is_interior"])
    assert np.all(r["regime"][interior] >= 0)
    assert np.all(np.isfinite(r["F"]))
    present, counts = np.unique(r["regime"][interior], return_counts=True)
    print("interior regimes:", dict(zip(present.tolist(), counts.tolist())))
    assert set(present.tolist()) >= {0, 1, 2}, present


# ---------------------------------------------------------------------------
# the actions on a 2-step window


@pytest.fixture(scope="module")
def win(model):
    return D.Window(model)


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("tke_diag")


def _complete(row):
    missing = [k for k in C.PROVENANCE_KEYS if k not in row]
    assert not missing, missing
    import json
    json.dumps(C._jsonable({k: v for k, v in row.items() if not str(k).startswith("_")}))


def test_incoming_uvnode_cotangent_below_the_surface_is_exactly_zero(model):
    """Why the norms use ``uv`` (elements), never ``uvnode``: the step recomputes ``uvnode``
    from ``uv`` before its mixing reads it, so the cotangent of the INCOMING ``uvnode`` of a
    one-step VJP is identically zero below the surface layer. The surface layer is not: the
    forced step's bulk formulae and ice-ocean stress read the previous step's surface current
    ``state.uvnode[:, 0]`` (step.py docstring), so a small surface cotangent is real."""
    row = C.run_grad(model, ("box_T_vw",), n_steps=1)[0]
    cot = row["_result"]["state_cot"]
    un, uv = np.asarray(cot.uvnode), np.asarray(cot.uv)
    assert np.count_nonzero(un[:, 1:, :]) == 0
    assert np.count_nonzero(uv) > 0
    print(f"one-step VJP: |cot uvnode[:,0]| = {np.linalg.norm(un[:, 0]):.3e} "
          f"({np.count_nonzero(un[:, 0])} entries), |cot uv| = {np.linalg.norm(uv):.3e}")


def test_screen_rows_maps_and_growth(model, out_dir):
    rows = D.run_screen(model, out_dir, repeats=2, map_every=1, label="screen_t")
    assert len(rows) == 2
    for r in rows:
        _complete(r)
        assert r["status"] == "ok" and len(r["trace"]) == 3
        g = r["growth"]
        assert np.isfinite(g["median"]) and np.isfinite(g["overall"]) and g["median"] > 0
        assert set(r["field_growth"]) >= {"tke", "T", "uv"}
        z = np.load(r["maps"])
        assert list(z["boundary"]) == [2, 1, 0]
        assert z["tke"].shape == (3, model.mesh.nod2D) and z["uv"].shape == (3, model.mesh.elem2D)
        assert np.all(np.isfinite(z["T"])) and z["T"][0].max() > 0
    # CPU: the two repeats are the same compiled programs on the same inputs
    assert rows[0]["trace"] == rows[1]["trace"]


def test_screen_overflow_abort(model, out_dir):
    row = D.run_screen(model, out_dir, abort_at=1e-300, label="screen_abort")[0]
    assert row["status"] == "overflow" and row["overflow"]["boundary"] == 2


@pytest.fixture(scope="module")
def svd_row(model, win, out_dir):
    return D.run_svd(model, win, out_dir, weights="tke", iters=3, tol=0.0, top=5, label="svd_t")


def test_svd_power_iteration(model, svd_row):
    r = svd_row
    _complete(r)
    assert len(r["history"]) == 3 and r["sigma"] > 0
    # sigma never exceeds the leading singular value: the iterates grow monotonically
    s = [h["sigma"] for h in r["history"]] + [r["sigma"]]
    assert all(b >= a * (1 - 1e-9) for a, b in zip(s, s[1:])), s
    # the seed lives in the box: every top input cell is a box node
    node, level = C.box_masks(model.mesh, C.DEFAULT_BOX)
    for c in r["top_v"]:
        assert node[c["node"]] and level[c["level"]] and c["regime"] in D.cvmix_tke.REGIMES
    z = np.load(r["vectors"])
    assert set(z.files) == {"v.tke", "u.tke"}


def test_svd_prog_norm_uses_uv_not_uvnode(model, win, out_dir):
    r = D.run_svd(model, win, out_dir, weights="prog", iters=1, tol=0.0, top=3, label="svd_p",
                  save=False)
    assert {c["field"] for c in r["top_v"]} == {"T", "S", "uv", "eta_n"}
    assert np.count_nonzero(np.asarray(r["_v"].uvnode)) == 0


def test_cuts_singles_and_pairs(model, win, out_dir):
    jpath = out_dir / "cuts.jsonl"
    rows = D.run_cuts(model, win, out_dir, ["tke.km_e", "freeze_tke_state"], jpath=jpath,
                      min_effect=0.0, weights="tke", iters=2, tol=0.0, top=2)
    assert [r["setting"] for r in rows] == ["none", "tke.km_e", "freeze_tke_state",
                                            rows[3]["setting"]]
    assert set(rows[3]["setting"].split("+")) == {"tke.km_e", "freeze_tke_state"}
    for r in rows:
        _complete(r)
    assert rows[0]["ratio"] == 1.0 and rows[1]["diag_cuts"] == ["tke.km_e"]
    # freeze_tke_state cuts tke_new: the tke -> tke map over the window is gone
    assert rows[2]["sigma"] < 1e-12 * rows[0]["sigma"] + 1e-300
    assert len(jpath.read_text().splitlines()) == 4


@pytest.fixture(scope="module")
def column_model():
    return C.build_model(START, n_steps=1, chunk_steps=1, restart=RESTART,
                         adjoint_cfg=AdjointConfig.exact(), ice_mode="prescribed")


def _shallow_live_columns(model, n=2, max_nlev=8):
    """Shallow columns whose tke is above the floor at every interior interface before AND
    after the step (the floor ``max(tke_solve, tke_min)`` has zero derivative, so a floored
    row would make every cut invisible)."""
    inp = D.step_tke_inputs(model, model.s0, D.step_forcing(model, START, 0))
    nlev = np.asarray(inp["nlev"])
    tmin = float(inp["tke_min"])
    interior = np.asarray(inp["is_interior"])
    ok_old = np.all(np.where(interior, np.asarray(inp["tke_old"]) > 10 * tmin, True), axis=1)
    ok_new = np.all(np.where(interior, np.asarray(inp["new_state"].tke) > 10 * tmin, True),
                    axis=1)
    live = ok_old & ok_new & (nlev >= 3) & (nlev <= max_nlev)
    return [int(i) for i in np.flatnonzero(live)[:n]]


def test_column_method_control(column_model):
    """The full model's one-step tke block at a column (tangent-linear, pinned) equals
    ``jax.jacfwd`` of the column core on the step's own inputs to 1e-12, and so do their σ₁;
    the driver's power iteration restricted to the column never exceeds that σ₁."""
    m = column_model
    nodes = _shallow_live_columns(m)
    assert len(nodes) == 2
    w = D.Window(m)
    row = D.run_column(m, w, nodes, power_iters=4)
    _complete(row)
    assert row["max_rel_diff"] <= 1e-12, row["columns"]
    assert row["max_sigma_rel"] <= 1e-12
    for c in row["columns"]:
        assert c["outside_block_max"] == 0.0 and len(c["diag"]) == c["nlev"] + 1
    s1 = row["columns"][0]["sigma_core"]
    assert 0.5 * s1 <= row["power_sigma"] <= s1 * (1 + 1e-12)


def test_column_method_control_catches_a_planted_error(column_model):
    """Negative control: the core Jacobian with the dissipation derivative cut disagrees with
    the (uncut) full model's block."""
    m = column_model
    nodes = _shallow_live_columns(m, n=1)
    inp = D.step_tke_inputs(m, m.s0, D.step_forcing(m, START, 0))
    good = D.core_column_jacobians(inp, nodes)
    bad = D.core_column_jacobians(inp, nodes, cuts=frozenset({"tke.diss"}))
    assert np.max(np.abs(good - bad)) > 1e-6 * np.max(np.abs(good))
    k = int(np.asarray(inp["nlev"])[nodes[0]]) + 1
    g = good[0, :k, :k]
    assert np.max(np.abs(g - g.T)) > 1e-6 * np.max(np.abs(g))   # a transposed block is caught


def test_fdlin_rows(model, win, svd_row):
    """fdlin along the leading tke vector: rows complete; the regime of each row consistent with
    the predicted range and the CPU roughness floor. The leading vector concentrates on the
    closure's switches, so its FD error is a measurement: every row is either close to the TL
    or classified (beyond the predicted range, or explained by the floor)."""
    v = D.load_vector(svd_row["vectors"], model.s0)
    # no h = 1e-8 row: there the CPU roughness of this window is as large as the FD error (2026-10-28, recom/port:
    # rel_err 0.86 against a floor of 0.39 from the larger of two samples at 1e-9 and 1e-10, and the floor grows with
    # the sample step), so the row's regime is a coin toss rather than a measurement of the tangent
    rows = D.run_fdlin(model, win, v, hs=[1e-4, 1e-6])
    assert [r["h"] for r in rows] == [1e-4, 1e-6]
    pr = rows[0]["predicted"]
    assert len(pr) == 2 and rows[0]["predicted_h"] == min(p["min_h"] for p in pr)
    assert rows[0]["predicted_h_tke"] == min(p["min_h_tke"] for p in pr) >= rows[0]["predicted_h"]
    # the convective entry uses the State's bvfreq, which a step computes from its ENTRY T and S:
    # a tke-only tangent reaches it only from the second step on, so on this 2-step window
    # (entries 0 and 1) it is still zero and the distance infinite
    assert pr[0]["convective"]["h"] == float("inf") == pr[1]["convective"]["h"]
    assert rows[0]["noise"] == "roughness" and np.isfinite(rows[0]["noise_floor"])
    for r in rows:
        _complete(r)
        assert np.isfinite(r["rel_err"])
        print(f"fdlin tke vector: h={r['h']:.0e} rel_err={r['rel_err']:.3e} {r['regime']} "
              f"noise_rel={r['noise_rel']:.3e} (predicted h {r['predicted_h']:.3e}) "
              f"per_field={ {k: float(f'{x:.2e}') for k, x in r['per_field'].items()} }")
        if r["regime"] == "linear":
            assert r["h"] <= r["predicted_h"] and r["rel_err"] < 2e-2
        elif r["regime"] == "beyond-predicted-range":
            assert r["h"] > r["predicted_h"]


def test_fdlin_control_direction(model, win):
    """Control for the FD machinery: along a smooth direction (1 K at the box's inside T point)
    the FD of the 2-step map matches the tangent-linear response in a norm on that point's own
    column of T. (In the global prognostic norm this direction's response is tiny against the
    forward's roughness, whose floor ``run_fdlin`` reports.)"""
    d = C.direction(model, "T_inside")
    node = d.info["node"]
    col = np.zeros(np.shape(model.s0.T))
    col[node, :] = np.asarray(model.mesh.node_layer_mask)[node, :]
    rows = D.run_fdlin(model, win, d.carry, weights={"T": col}, hs=[1e-2, 1e-3, 1e-4])
    for r in rows:
        print(f"fdlin T_inside (own column): h={r['h']:.0e} rel_err={r['rel_err']:.3e} "
              f"{r['regime']} noise_rel={r['noise_rel']:.3e}")
    assert min(r["rel_err"] for r in rows) < 1e-4
    g = D.run_fdlin(model, win, d.carry, weights="prog", hs=[1e-4], noise="roughness")[0]
    print(f"fdlin T_inside (prog norm): h=1e-4 rel_err={g['rel_err']:.3e} {g['regime']} "
          f"noise_rel={g['noise_rel']:.3e}")


def test_parse_setting():
    cfg = D.parse_setting("tke.km_e+convection")
    assert cfg.diag_cuts == frozenset({"tke.km_e"}) and cfg.freeze_convection
    assert D.parse_setting("none") == AdjointConfig.exact().validate()
    with pytest.raises(ValueError):
        D.parse_setting("tke.nonsense")
    c = D.parse_setting("mixing_state+tke_kme_tke=3e-6+tke_kme_term=ke_diff")
    assert c.freeze_mixing_state and c.tke_kme_tke == 3e-6 and c.tke_kme_term == "ke_diff"
    c = D.parse_setting("tke_pr_cut=inf")
    assert c.tke_pr_cut == float("inf") and c.tke_options().pr_mode == "cut"
    with pytest.raises(ValueError):
        D.parse_setting("tke_pr_mode=at")                  # a mode without N²*
    c = D.parse_setting("tke_pr_cut=1e-8+tke_pr_ratio=10")
    assert c.tke_pr_cut == 1e-8 and c.tke_pr_ratio == 10.0 and c.tke_options().pr_ratio == 10.0
    c = D.parse_setting("tke_pr_cut=1e-8+tke_pr_raw_cut=0+tke.kh_floor")
    assert c.tke_pr_raw_cut == 0.0 and c.tke_options().pr_raw_cut == 0.0
    assert c.diag_cuts == frozenset({"tke.kh_floor"})
    with pytest.raises(ValueError):
        D.parse_setting("tke_nonsense=1")
    assert D.parse_also("long_window,freeze_convection*2", 3) == [("long_window", 3),
                                                                ("freeze_convection", 2)]
    assert D.parse_also("") == []


# ---------------------------------------------------------------------------
# windows at later times, the sites action, the SSH forward tolerance option

STEP1 = START + datetime.timedelta(seconds=C.DT)


def test_window_at_a_later_start_is_the_tail_of_the_window(model, pins, win):
    """A one-chunk Window from the pinned step-1 State at the step-1 time is the second chunk
    of the 2-step window: same pins, same tangent (CPU, same programs: bitwise), and its
    regimes are those of step 1's own inputs. A later start without its State is refused."""
    w1 = D.Window(model, start_state=model.state_of(pins[1]), start=STEP1, n_chunks=1)
    assert w1.start == STEP1 and w1.n_steps == 1
    for a, b in zip(jax.tree.leaves(w1.pins[1]), jax.tree.leaves(pins[2])):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    v = D.random_state_vector(model.s0, {"tke": 1.0, "T": 1.0}, seed=3)
    t1 = w1.M(v)
    t2 = win.M(v, n_chunks=1, start_chunk=1)
    for f in ("tke", "T", "uv", "eta_n"):
        assert np.array_equal(np.asarray(getattr(t1, f)), np.asarray(getattr(t2, f))), f
    ref = D.regimes(_inputs_at(model, pins, 1))
    assert np.array_equal(w1.regimes(0)["regime"], ref["regime"])
    with pytest.raises(ValueError):
        D.Window(model, start=STEP1, n_chunks=1)
    # the window reads the forcing of ITS time: at +3 h (the forcing is 3-hourly, so steps 0 and
    # 1 share their forcing and the check above cannot see it) the first chunk's forcing is that
    # of +3 h, and it differs from the model start's
    t3 = START + datetime.timedelta(hours=3)
    w3 = D.Window(model, start_state=model.s0, start=t3, n_chunks=1)
    x_ref, x_0 = D.step_forcing(model, t3, 0), D.step_forcing(model, START, 0)
    got = jax.tree.map(lambda a: np.asarray(a)[0], w3.xs(0))
    assert any(not np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(jax.tree.leaves(x_ref), jax.tree.leaves(x_0)))
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(x_ref)):
        assert np.array_equal(a, np.asarray(b))


def test_sites_action(model, out_dir):
    """``run_sites``: advances to the window time, repeats the baseline (bitwise on CPU), runs
    the singles on the same pins, appends complete rows, and records the window start."""
    spec = {"windows": [{"name": "a", "at": STEP1.isoformat(), "steps": 1, "box": C.DEFAULT_BOX,
                         "weights": ["tke"], "cuts": True}],
            "singles": ["tke.km_e"], "pairs": False, "iters": 2, "tol": 0.0, "top": 2,
            "baseline_repeats": 2}
    jpath = out_dir / "sites.jsonl"
    rows = D.run_sites(model, spec, out_dir, jpath, {"tag": "t"})
    assert [(r["setting"], r["repeat"]) for r in rows] == [("none", 0), ("none", 1),
                                                           ("tke.km_e", 0)]
    for r in rows:
        _complete(r)
        assert r["start"] == STEP1.isoformat() and r["n_steps"] == 1 and r["site"] == "a"
    assert rows[0]["sigma"] == rows[1]["sigma"] > 0
    c = rows[0]["census_v"]
    assert abs(sum(c["regime"].values()) - 1.0) < 1e-12 and abs(sum(c["level"].values()) - 1.0) < 1e-12
    assert rows[0]["mld_box"]["n_nodes"] > 0 and rows[0]["mld_box"]["max"] > 0
    assert rows[2]["sigma"] != rows[0]["sigma"]          # km_e is live in the tke -> tke map
    assert len(jpath.read_text().splitlines()) == 3


def test_ssh_forward_tol_option(model):
    """``set_ssh_forward_tol`` changes the solve every later trace uses (the step reads the
    function through the module), converges it (1e-12 lands far closer to a 1e-13 reference
    than the production 1e-5 stop does), and ``None`` restores the library function."""
    from fesom_jax import ssh
    assert D.stepmod.ssh is ssh is D.sshmod
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal(int(model.mesh.nod2D)) * 1e-3
    base = np.asarray(ssh.solve_ssh(model.op, rhs))
    try:
        D.set_ssh_forward_tol(1e-12)
        assert D.ssh_forward_tol() == (1e-12, 2000)
        tight = np.asarray(ssh.solve_ssh(model.op, rhs))
        D.set_ssh_forward_tol(1e-13)
        ref = np.asarray(ssh.solve_ssh(model.op, rhs))
    finally:
        D.set_ssh_forward_tol(None)
    assert ssh.solve_ssh is D._SOLVE_SSH and D.ssh_forward_tol()[0] == 1e-5
    d_base = np.linalg.norm(base - ref)
    d_tight = np.linalg.norm(tight - ref)
    print(f"ssh solve: |prod - ref| {d_base:.3e}, |1e-12 - ref| {d_tight:.3e}, |ref| "
          f"{np.linalg.norm(ref):.3e}")
    assert d_base > 0 and d_tight < 1e-4 * d_base


def test_window_forcing_follows_its_start(model):
    """A Window's forcing is that of ITS start, not the model's (a planted
    ``model.xs_fn()`` in ``Window`` fails here: six hours later the forcing differs)."""
    t = START + datetime.timedelta(hours=6)
    w = D.Window(model, start_state=model.s0, start=t, n_chunks=1)
    got = jax.tree.leaves(jax.device_get(w.xs(0)))
    want = jax.tree.leaves(jax.device_get(model.xs_fn(t)(0)))
    base = jax.tree.leaves(jax.device_get(model.xs_fn()(0)))
    assert all(np.array_equal(a, b) for a, b in zip(got, want))
    assert any(not np.array_equal(a, b) for a, b in zip(got, base))


def test_locate_action(model, pins, out_dir):
    """``run_locate`` end to end on 1-step windows: scan rows at both times, the site and the
    quiet control with their windows, the saved site State (loads back bitwise), sites.json,
    and the cuts at the top site — baseline repeats and a second forward identical on the CPU
    (zero realisation differences), singles warm-started, the pair among them."""
    import json
    spec = {"scan_times": [START.isoformat(), STEP1.isoformat()], "scan_steps": 1,
            "scan_iters": 2, "n_sites": 1, "quiet": {"name": "quiet",
                                                     "box": [20, 26, -150, -140, 0, 1000]},
            "site_windows": [1, 2], "site_weights": ["tke"], "site_iters": 2,
            "cuts": {"windows": [1, 2], "singles": ["tke.km_e", "freeze_tke_state"],
                     "pairs": True, "min_effect": 0.0, "baseline_repeats": 2,
                     "new_pins": True, "iters": 2, "tol": 0.0},
            "tol": 0.0, "top": 3}
    d = out_dir / "locate"
    d.mkdir()
    jpath = d / "results.jsonl"
    rec = D.run_locate(model, spec, d, jpath, {"tag": "t"})
    rows = [json.loads(x) for x in jpath.read_text().splitlines()]
    acts = [r["action"] for r in rows]
    assert acts.count("tke_scan") == 2 and acts.count("tke_site") == 4
    cuts = [r for r in rows if r["action"] == "tke_cuts"]
    assert [(r["setting"], r["n_steps"], r["repeat"]) for r in cuts[:3]] == [
        ("none", 1, 1), ("none", 2, 1), ("none", 2, "pins2")]
    assert {r["setting"] for r in cuts[3:]} == {"tke.km_e", "freeze_tke_state",
                                                 "tke.km_e+freeze_tke_state"} or {
        r["setting"] for r in cuts[3:]} == {"tke.km_e", "freeze_tke_state",
                                            "freeze_tke_state+tke.km_e"}
    assert all(r["warm_start"] for r in cuts[3:]) and len(cuts) == 3 + 3 * 2
    for r in rows:
        _complete(r)
    site = rec["sites"][0]
    base = {r["n_steps"]: r["sigma"] for r in rows if r["action"] == "tke_site"
            and r["site"] == site["name"]}
    for r in cuts[:3]:                     # CPU: repeats and a second forward are bitwise
        assert r["sigma"] == base[r["n_steps"]] and r["ratio"] == 1.0
    for dd in cuts[2]["realisation_diff"]:
        assert all(v == 0 for reg in dd.values() for v in reg.values())
    js = json.loads((d / "sites.json").read_text())
    assert [s["name"] for s in js["sites"]] == ["site1", "quiet"]
    st, meta = C.load_state(site["state"])
    t = C._parse_date(site["at"])
    ref = model.s0 if t == START else model.state_of(pins[1])
    assert t in (START, STEP1)
    for f in ("T", "tke", "uv", "eta_n"):
        assert np.array_equal(np.asarray(getattr(st, f)), np.asarray(getattr(ref, f))), f
    assert site["state_bytes"] > 0 and "mld_box" in site


def test_switch_distances_convective(model):
    """The convective entry is min |bvfreq| / |dbvfreq| over the wet interior interfaces."""
    x = D.step_forcing(model, START, 0)
    inp = D.step_tke_inputs(model, model.s0, x)
    interior = np.asarray(inp["is_interior"])
    dtke = np.where(interior, 1.0, 0.0)
    d = D.switch_distances(model, model.s0, x, dtke, dbvfreq=np.full(np.shape(model.s0.bvfreq), 2.0))
    bv = np.abs(np.asarray(model.s0.bvfreq))
    want = np.nanmin(np.where(interior, bv, np.nan)) / 2.0
    assert d["convective"]["h"] == want and d["min_h"] <= want
    assert d["min_h_tke"] == min(d[k]["h"] for k in ("floor", "cap", "mixing_length"))


def test_realisation_diff_counts_planted_flips(model):
    """``realisation_diff`` of a State with itself is all zero; one planted convective flip,
    one tke floor contact and one 1 K skin-temperature change are each counted once (globally,
    and in a box that holds them)."""
    import dataclasses
    a = model.s0
    z = D.realisation_diff(model, a, a, C.DEFAULT_BOX)
    assert all(v == 0 for reg in z.values() for v in reg.values())
    node, level = C.box_masks(model.mesh, C.DEFAULT_BOX)
    wet = np.asarray(model.mesh.node_iface_mask)
    n = int(np.nonzero(node & wet[:, 2])[0][0])
    bv = np.array(a.bvfreq, copy=True)
    bv[n, 2] = -1e-6 if bv[n, 2] >= 0 else 1e-6
    tk = np.array(a.tke, copy=True)
    tk[n, 2] = 0.0 if tk[n, 2] > 1e-6 else 1.0
    ts = np.array(a.t_skin, copy=True)
    ts[n] += 1.0
    b = dataclasses.replace(a, bvfreq=bv, tke=tk, t_skin=ts)
    d = D.realisation_diff(model, a, b, C.DEFAULT_BOX)
    for reg in ("global", "box"):
        assert d[reg]["conv_flips"] == 1 and d[reg]["tke_floor_flips"] == 1
        assert d[reg]["t_skin_gt_0.1K"] == 1 and d[reg]["t_skin_max"] == 1.0


def test_locate_given_sites(model, pins, out_dir):
    """``run_locate`` with given sites (no scan) and a per-length window plan: the site State is
    the model's own State at that time (advanced, bitwise on the CPU), only the planned
    windows/norms run, and the cuts use the site's kept window."""
    import json
    node, _ = C.box_masks(model.mesh, C.DEFAULT_BOX)
    lon, lat = C._lonlat_deg(model.mesh)
    n0 = int(np.nonzero(node)[0][0])
    spec = {"sites": [{"at": STEP1.isoformat(), "lon": float(lon[n0]), "lat": float(lat[n0])}],
            "site_windows": {"1": ["tke"]}, "site_iters": 2,
            "cuts": {"windows": [1], "singles": ["tke.km_e"], "pairs": False,
                     "baseline_repeats": 1, "new_pins": False, "iters": 2, "tol": 0.0},
            "tol": 0.0, "top": 3}
    d = out_dir / "locate_given"
    d.mkdir()
    jpath = d / "results.jsonl"
    rec = D.run_locate(model, spec, d, jpath, {"tag": "g"})
    rows = [json.loads(x) for x in jpath.read_text().splitlines()]
    assert [(r["action"], r["n_steps"], r["weights"], r["setting"]) for r in rows] == [
        ("tke_site", 1, "tke", "none"), ("tke_cuts", 1, "tke", "tke.km_e")]
    site = rec["sites"][0]
    assert site["name"] == "site1" and site["at"] == STEP1.isoformat()
    assert site["box"][0] < float(lat[n0]) < site["box"][1]
    st, _ = C.load_state(site["state"])
    ref = model.state_of(pins[1])
    for f in ("T", "tke", "uv", "eta_n"):
        assert np.array_equal(np.asarray(getattr(st, f)), np.asarray(getattr(ref, f))), f
    assert rows[1]["warm_start"] and rows[1]["ratio"] != 1.0


def test_top_nodes():
    row = {"top_v": [{"field": "T", "node": 9}, {"field": "tke", "node": 5},
                     {"field": "tke", "node": 5}, {"field": "tke", "node": 7},
                     {"field": "tke", "node": 3}]}
    assert D.top_nodes(row, 2) == [5, 7] and D.top_nodes(row, 5) == [5, 7, 3]


def test_screen_cli_writes_each_setting(out_dir):
    """``tke_diag.py screen`` writes each setting's rows as it finishes (a long job keeps what
    it has done), with the setting name on the rows of the ``--also`` settings."""
    import json
    d = out_dir / "cli_screen"
    rc = D.main(["screen", "--start", START.isoformat(), "--restart", str(RESTART),
                 "--steps", "2", "--chunk-steps", "1", "--setting", "tke.pr",
                 "--also", "freeze_mixing*2,=tke.n2", "--tag", "cli", "--out", str(d)])
    rows = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()]
    assert rc == 0 and [r["setting"] for r in rows] == ["tke.pr", "freeze_mixing",
                                                        "freeze_mixing", "tke.n2"]
    assert [r["repeat"] for r in rows] == [0, 0, 1, 0] and all(r["tag"] == "cli" for r in rows)
    # '+' settings add to --setting; a leading '=' replaces it
    assert rows[1]["diag_cuts"] == ["tke.pr"] and rows[3]["diag_cuts"] == ["tke.n2"]


def test_locate_in_the_prognostic_norm(model, out_dir):
    """``run_locate`` with ``scan_weights``/``cuts.weights`` = ``prog``: the scan, the site
    windows and the cuts all use the volume-weighted T, S, uv, eta_n norm; the site is the
    peak of the scan's leading input vector among its located cells."""
    import json
    spec = {"scan_times": [START.isoformat()], "scan_steps": 1, "scan_iters": 2,
            "scan_weights": "prog", "n_sites": 1, "site_windows": {"1": ["prog"]},
            "site_iters": 2, "save_states": False,
            "cuts": {"weights": "prog", "windows": [1], "singles": ["freeze_mixing"],
                     "pairs": False, "baseline_repeats": 1, "new_pins": False, "iters": 2,
                     "tol": 0.0},
            "tol": 0.0, "top": 3}
    d = out_dir / "locate_prog"
    d.mkdir()
    jpath = d / "results.jsonl"
    rec = D.run_locate(model, spec, d, jpath, {"tag": "p"})
    rows = [json.loads(x) for x in jpath.read_text().splitlines()]
    assert [(r["action"], r["weights"], r["setting"]) for r in rows] == [
        ("tke_scan", "prog", "none"), ("tke_site", "prog", "none"),
        ("tke_cuts", "prog", "freeze_mixing")]
    site = rec["sites"][0]
    assert site["field"] in ("T", "S") and site["scan_weights"] == "prog"
    top = max((c for c in rows[0]["top_v"] if "lon" in c), key=lambda c: abs(c["value"]))
    assert (site["lon"], site["lat"]) == (top["lon"], top["lat"])
    assert rows[2]["sigma"] != rows[1]["sigma"]          # freeze_mixing is live here


def test_column_jacobians_all_and_the_recursion_census(model, pins, out_dir):
    """The all-column TKE recursion (one tangent per level, all columns at once) equals the
    per-column jacfwd of the core (block diagonal by column); the census reads ρ on the wet
    block; a cut changes the recursion; ``run_colcensus`` rows are complete."""
    inp = _inputs_at(model, pins, 0)
    J = D.column_jacobians_all(inp)
    nodes = _shallow_live_columns(model, n=3, max_nlev=30)
    Jc = D.core_column_jacobians(inp, nodes)
    for i, n in enumerate(nodes):
        assert np.allclose(J[n], Jc[i], rtol=0, atol=1e-13 * max(1.0, np.abs(Jc[i]).max()))
    lon, lat = C._lonlat_deg(model.mesh)
    reg = D.regimes(inp)
    cen, rho = D.recursion_census(J, inp["nlev"], lon, lat, reg, top=3)
    m = int(np.asarray(inp["nlev"])[nodes[0]])
    want = np.max(np.abs(np.linalg.eigvals(J[nodes[0], :m + 1, :m + 1])))
    assert np.isclose(rho[nodes[0]], want, rtol=1e-12) and cen["n_wet"] > 0
    Jd = D.column_jacobians_all(inp, frozenset({"tke.diss"}))
    assert np.abs(Jd - J).max() > 0
    rows = D.run_colcensus(model, model.s0, START, ["tke.diss"], out_dir, label="cc", top=2)
    assert [r["setting"] for r in rows] == ["none", "tke.diss"]
    for r in rows:
        _complete(r)
    assert "stabilised_share" in rows[1] and len(rows[0]["top"]) == 2


def test_colcensus_candidate_settings_two_step_and_prcensus(model, pins, out_dir):
    """Candidate settings in the census: ``tke_kme_tke = inf`` gives the ``tke.km_e`` recursion
    exactly and flags every wet column, ``0`` the plain one; the two-step ρ of ``J1·J0`` equals a
    direct eigenvalue computation; the Pr census shares are monotone in N²* and 1 at +inf."""
    inp = _inputs_at(model, pins, 0)
    J = D.column_jacobians_all(inp)
    cuts, adj = D._setting_cuts("tke_kme_tke=inf")
    assert np.array_equal(D.column_jacobians_all(inp, cuts, adj),
                          D.column_jacobians_all(inp, frozenset({"tke.km_e"})))
    cuts0, adj0 = D._setting_cuts("tke_kme_tke=0")
    # an empty mask is the exact recursion; the masked program fuses differently (round-off)
    J0 = D.column_jacobians_all(inp, cuts0, adj0)
    assert np.abs(J0 - J).max() <= 1e-12 * np.abs(J).max()
    rows = D.run_colcensus(model, model.s0, START, ["tke_kme_tke=inf", "tke_kme_tke=3e-6"],
                           out_dir, label="cc2", top=2, two_step=True)
    assert [r["setting"] for r in rows] == ["none", "tke_kme_tke=inf", "tke_kme_tke=3e-6"]
    wet = int((np.asarray(inp["nlev"]) > 0).sum())
    assert rows[1]["n_flagged"] == wet and 0 <= rows[2]["n_flagged"] <= wet
    assert "two_step" in rows[0] and "stabilised_share" in rows[1]["two_step"]
    inp1 = _inputs_at(model, pins, 1)
    J1 = D.column_jacobians_all(inp1)
    r2 = D.two_step_rho(J, J1, inp["nlev"])
    n = _shallow_live_columns(model, n=1, max_nlev=30)[0]
    m = int(np.asarray(inp["nlev"])[n])
    want = np.sqrt(np.max(np.abs(np.linalg.eigvals(J1[n, :m + 1, :m + 1] @ J[n, :m + 1, :m + 1]))))
    assert np.isclose(r2[n], want, rtol=1e-12)
    pc = D.pr_census(inp, [1e-10, 1e-8, 1e-6, float("inf")])
    sh = [x["share_G"] for x in pc["per_n2_star"]]
    assert pc["n_live"] > 0 and sh == sorted(sh) and sh[-1] == 1.0


def test_ts_norm_is_the_tracer_part_of_prog(model):
    p = D.norm_weights(model, model.s0, "prog")
    t = D.norm_weights(model, model.s0, "ts")
    assert set(t) == {"T", "S"} and set(p) >= {"T", "S", "uv", "eta_n"}
    for k in t:
        assert np.array_equal(t[k], p[k])


def test_vector_census_counts_tracer_leaves(model, pins):
    """The census of a T/S vector uses the given weights; shares sum to one."""
    import dataclasses
    reg = D.regimes(_inputs_at(model, pins, 0))
    w = D.norm_weights(model, model.s0, "ts")
    v = D.random_state_vector(model.s0, w, seed=1)
    c = D.vector_census(v, reg, w)
    assert abs(sum(c["regime"].values()) - 1.0) < 1e-12 and c["norm2"] > 0
    zero_t = dataclasses.replace(v, T=np.zeros_like(np.asarray(v.T)))
    c2 = D.vector_census(zero_t, reg, w)
    assert 0 < c2["norm2"] < c["norm2"]
