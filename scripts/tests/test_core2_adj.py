"""The shared CORE2 adjoint driver ``scripts/adjoint/core2_adj.py`` on two production steps.

The model is built exactly as the tracks build it (the 2013-01-01 hindcast restart, the
reference physics from ``configs/core2_full.yaml``, prescribed sea ice, the window-mean MLD
accumulator on the carry) with 1-step chunks, and every action runs once: ``fwd``, ``grad``
(one cost, several costs, resumed across legs), ``screen`` (and its overflow abort), ``fd``,
``tl``, ``dot``, ``states`` and ``restartcheck``. Checked: the tangent and adjoint agree on pinned
states (dot test and ``dJ`` vs ``<state_cot, d>``), FD agrees with the adjoint, a resumed sweep
equals the uninterrupted one bit for bit (CPU), saved States load back unchanged and a restarted
step reproduces the continuous run bit for bit, JSONL rows carry every provenance key, a second
window in the same process compiles nothing new, and the import guard fires when ``fesom_jax``
is not this checkout's.

⚠️ HEAVY (CPU compute node, >= 200 GB, ~1 h): five CORE2 programs are compiled (chunk forward
with remat, its VJP, the tangent-linear chunk, the no-remat forward of the ``states`` action,
the daily snapshot). The file lives in the ``scripts`` group; run that group with ``--mem=200G``.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts" / "adjoint" / "core2_adj.py"
sys.path.insert(0, str(DRIVER.parent))
import core2_adj as C  # noqa: E402

from fesom_jax import adjoint, partit, shard_mesh, zarr_output  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402

RESTART = C.RESTART_ARCHIVE / "fesom.2013.001.00000"
pytestmark = pytest.mark.skipif(
    not ((ROOT / "data" / "mesh_core2").is_dir() and RESTART.is_dir()),
    reason="CORE2 mesh / 2013 restart missing")

N_STEPS = 2
START = datetime.datetime(2013, 1, 1)


def _eq_tree(a, b) -> bool:
    return all(np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


@pytest.fixture(scope="module")
def model():
    return C.build_model(START, n_steps=N_STEPS, chunk_steps=1, restart=RESTART,
                         adjoint_cfg=AdjointConfig.exact(),
                         ice_mode="prescribed", accumulators=("mld",))


@pytest.fixture(scope="module")
def grad_row(model):
    return C.run_grad(model, ("box_T_vw",), check_linear="full")[0]


def _check_row(row):
    missing = [k for k in C.PROVENANCE_KEYS if k not in row]
    assert not missing, missing
    json.dumps(C._jsonable({k: v for k, v in row.items() if not k.startswith("_")}))
    assert row["n_chunks"] == N_STEPS and row["chunk_steps"] == 1
    assert row["config"] == str(ROOT / "configs" / "core2_full.yaml")
    assert row["ice_mode"] == "prescribed" and row["diag_cuts"] == []


def test_model_is_the_reference_configuration(model):
    assert model.config_path == ROOT / "configs" / "core2_full.yaml"
    assert model.restart == RESTART and model.restart_meta["calendar_date"] == "2013-doy001"
    assert model.phys["ice_cfg"].adjoint_mode == "prescribed"
    assert model.phys["ice_cfg"].whichEVP == 1
    assert model.phys["tke_cfg"] is not None and model.phys["kpp_cfg"] is None
    assert model.phys["ale_cfg"] is not None and model.phys["gm_cfg"] is not None
    for leaf in jax.tree.leaves(model.params0):
        assert isinstance(leaf, np.ndarray) and leaf.dtype == np.float64
    bad = dataclasses.replace(model.cfg, tke=None, kpp=model.cfg.kpp)
    with pytest.raises(AssertionError, match="not classical TKE"):
        C.assert_physics(bad)
    with pytest.raises(ValueError, match="multiple of chunk_steps"):
        C.build_model(START, 3, 2)


def test_the_driver_builds_and_validates_the_dynamics_mode():
    """``ice_mode="dynamics"`` with the stage-1 damping merges into the IceConfig via
    ``_replace`` and is validated at build time; unknown fields and damping under a mode that
    does not unroll the sweeps are refused; the CLI options map onto ``ice_adjoint``."""
    assert "dynamics" in C.ICE_MODES
    m = C.build_model(START, n_steps=1, chunk_steps=1, restart=RESTART, ice_mode="dynamics",
                      ice_adjoint={"adjoint_damping": "local", "adjoint_kappa": 0.25})
    ic = m.phys["ice_cfg"]
    assert (ic.adjoint_mode, ic.adjoint_damping, ic.adjoint_kappa, ic.adjoint_sweeps) == \
        ("dynamics", "local", 0.25, -1)
    assert ic.whichEVP == 1 and m.ice_mode == "dynamics"
    with pytest.raises(ValueError, match="unknown field"):
        C.build_model(START, 1, 1, RESTART, ice_mode="dynamics", ice_adjoint={"adjoint_tn": 7})
    with pytest.raises(ValueError, match="does not"):
        C.build_model(START, 1, 1, RESTART, ice_mode="prescribed",
                      ice_adjoint={"adjoint_damping": "local"})
    with pytest.raises(ValueError, match="adjoint_kappa"):
        C.build_model(START, 1, 1, RESTART, ice_mode="dynamics",
                      ice_adjoint={"adjoint_damping": "local", "adjoint_kappa": -1.0})
    args = C._parse(["grad", "--ice", "dynamics", "--ice-damping", "local",
                     "--ice-kappa", "0.3", "--ice-sweeps", "16"])
    assert C._ice_adjoint_args(args) == {"adjoint_damping": "local", "adjoint_kappa": 0.3,
                                         "adjoint_sweeps": 16}
    assert C._ice_adjoint_args(C._parse(["grad"])) == {}
    assert C._ice_adjoint_args(C._parse(["grad", "--ice-cuts", "advect+ssh"])) == \
        {"adjoint_cuts": ("advect", "ssh")}
    assert C._ice_adjoint_args(C._parse(["grad", "--ice-cuts", "none"])) == {"adjoint_cuts": ()}
    with pytest.raises(ValueError, match="dynamics"):
        C.build_model(START, 1, 1, RESTART, ice_mode="prescribed",
                      ice_adjoint={"adjoint_cuts": ("ssh",)})


def test_boxes_and_directions(model):
    for name in C.BOXES:
        node, level = C.box_masks(model.mesh, name)
        assert node.sum() > 20 and level.sum() >= 5
    node, level = C.box_masks(model.mesh)
    zmid = -np.asarray(model.mesh.Z)
    assert zmid[level[:zmid.size]].max() <= 500.0
    for name in ("T_inside", "T_above", "T_outside"):
        d = C.direction(model, name)
        st = model.state_of(d.carry)
        assert float(np.sum(st.T)) == 1.0 and np.count_nonzero(st.T) == 1
        n, k = d.info["node"], d.info["level"]
        assert bool(np.asarray(model.mesh.node_layer_mask)[n, k])
        assert bool(node[n]) == (name != "T_outside")
        assert all(np.count_nonzero(np.asarray(x)) == 0 for x in jax.tree.leaves(d.params))
    d = C.direction(model, "ln_tke_c_k")
    assert float(d.params["ln_tke_c_k"]) == 1.0
    with pytest.raises(ValueError, match="surface_adj"):
        C.direction(model, "heat")
    sa = dataclasses.replace(model, params0=C.default_params0(model.mesh.nod2D, True),
                             surface_adj=True)
    h = C.direction(sa, "heat")
    assert np.array_equal(h.params["surface_adj"]["heat"] > 0, node)
    s = C.direction(sa, "stress_x")
    assert float(s.params["surface_adj"]["stress_x"].max()) == C.STRESS_UNIT


def test_fwd_row_and_costs(model):
    row = C.run_fwd(model, ("box_T_vw", "sst_aw", "ssh_aw", "mld_box", "mld_mean"))
    _check_row(row)
    assert all(np.isfinite(row["values"]))
    assert 0.0 < row["values"][3] and 0.0 < row["values"][4]
    assert row["compile_s"] > 0.0 and row["run_s"] > 0.0
    fin = row["_final"]
    assert fin[1]["mld"].shape == (model.mesh.nod2D,)
    print(f"\n[core2_adj fwd] compile {row['compile_s']:.1f} s, run {row['run_s']:.1f} s "
          f"({N_STEPS} steps), forcing {row['forcing_s']:.1f} s")


def test_grad_single_multi_and_the_guard(model, grad_row):
    _check_row(grad_row)
    assert grad_row["status"] == "ok"
    lc = grad_row["linear_check"]
    assert lc["mode"] == "full" and lc["zero_max"] == 0.0
    assert len(grad_row["trace"]) == N_STEPS + 1 and len(grad_row["field_trace"]) == N_STEPS + 1
    g = grad_row["_result"]["grad"]
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(g))
    assert float(np.abs(g["ln_k_gm"])) > 0.0
    multi = C.run_grad(model, ("box_T_vw", "mld_mean"), check_linear=False)
    assert [r["cost"] for r in multi] == ["box_T_vw", "mld_mean"]
    assert _eq_tree(multi[0]["_result"]["grad"], g)
    assert _eq_tree(multi[0]["_result"]["state_cot"], grad_row["_result"]["state_cot"])
    assert float(np.abs(multi[1]["_result"]["grad"]["ln_tke_c_k"])) > 0.0
    print(f"\n[core2_adj grad] compile {grad_row['compile_s']:.1f} s, fwd "
          f"{grad_row['forward_seconds']:.1f} s, rev {grad_row['reverse_seconds']:.1f} s "
          f"(incl. full guard), multi(K=2) wall {multi[0]['wall_s']:.1f} s")


def test_tl_and_dot_on_pinned_states(model, grad_row):
    ad = C.adjoint_check.tree_vdot(grad_row["_result"]["state_cot"],
                                   C.direction(model, "T_inside").carry)
    tl = C.run_tl(model, "box_T_vw", "T_inside")
    _check_row(tl)
    assert tl["direction"] == "T_inside" and len(tl["trace"]) == N_STEPS + 1
    assert abs(tl["dJ"] - ad) <= 1e-9 * abs(ad), (tl["dJ"], ad)
    pins = tl["_pinned"]
    rows = C.run_dot(model, "box_T_vw", "T_inside", amps=(1.0, 1e-8), pinned=pins)
    rows += C.run_dot(model, "box_T_vw", "ln_tke_c_k", amps=(1.0, 1e-8), pinned=pins)
    # rel_norm (divided by ||M x||·||y||) is the criterion; the plain relative mismatch is
    # bounded loosely: the tangent program (no constant folding, no remat) and the reverse
    # program are different compiled programs whose in-chunk primals differ at round-off, which
    # this model amplifies (measured 1.3e-8 along ln_tke_c_k over two steps); a transpose error
    # is O(1)
    for r in rows:
        _check_row(r)
        assert r["rel_norm"] <= 1e-9 and r["rel"] <= 1e-6, r
    print("\n[core2_adj dot] " + ", ".join(f"{r['direction']}@{r['amp']:g}: rel {r['rel']:.1e} "
                                         f"rel_norm {r['rel_norm']:.1e}" for r in rows)
          + f"; TL dJ {tl['dJ']:.6e} vs ADJ {ad:.6e}; tl compile {tl['compile_s']:.1f} s")


def test_fd_agrees_with_the_adjoint(model, grad_row):
    d = C.direction(model, "T_inside")
    ad = C.adjoint_check.tree_vdot(grad_row["_result"]["state_cot"], d.carry)
    rows = C.run_fd(model, "box_T_vw", d, hs=(1e-1, 1e-2))
    for r in rows:
        _check_row(r)
        assert r["h"] in (1e-1, 1e-2) and r["noise_over_h"] == 0.0
    best = min(abs(r["fd"] - ad) / abs(ad) for r in rows)
    print(f"\n[core2_adj fd] T_inside: fd {[r['fd'] for r in rows]} ad {ad:.6e} best rel {best:.1e}")
    assert best <= 1e-4


def test_screen_and_its_overflow_abort(model):
    ok = C.run_screen(model, "box_T_vw", check_linear=False)
    _check_row(ok)
    assert ok["status"] == "ok" and len(ok["field_trace"]) == N_STEPS + 1
    # the carry is (State, accumulators): leaf names carry the tuple index
    assert "0.T" in ok["field_trace"][0] and "1.mld" in ok["field_trace"][0]
    assert ok["field_trace"][0]["boundary"] == N_STEPS
    over = C.run_screen(model, "box_T_vw", check_linear=False, abort_at=1e-30)
    _check_row(over)
    assert over["status"] == "overflow"
    assert over["overflow"]["boundary"] == N_STEPS
    assert over["overflow"]["day"] == pytest.approx(N_STEPS * 1800.0 / 86400.0)
    assert over["overflow"]["lag_days"] == 0.0
    with pytest.raises(ValueError, match="terminal"):
        C.run_screen(model, "mld_mean")


def test_resumed_sweep_equals_the_uninterrupted_one(model, grad_row, tmp_path):
    rows = []
    for _ in range(12):
        r = C.run_grad(model, ("box_T_vw",), resume_dir=tmp_path / "ck", budget_s=0.0,
                       check_linear=False)[0]
        rows.append(r)
        if r["status"] == "ok":
            break
    assert rows[-1]["status"] == "ok" and len(rows) >= 3
    assert all(r["status"] == "incomplete" for r in rows[:-1])
    assert _eq_tree(rows[-1]["_result"]["grad"], grad_row["_result"]["grad"])
    assert _eq_tree(rows[-1]["_result"]["state_cot"], grad_row["_result"]["state_cot"])
    print(f"\n[core2_adj resume] {len(rows)} legs: "
          + ", ".join(f"{r.get('leg_phase', 'done')}@{r.get('leg_chunk', '')}" for r in rows))


def test_xs_built_ahead_equal_the_plain_build(model, monkeypatch):
    """The chunk forcing built ahead on the worker thread (``core2_adj._XsAhead``) is bit for bit the plain build,
    in the forward order and in the reverse walk's (chunks repeated and taken backwards)."""
    monkeypatch.setenv("FESOM_FORCING_PREFETCH", "0")
    plain = model.xs_fn()
    monkeypatch.delenv("FESOM_FORCING_PREFETCH")
    ahead = model.xs_fn()
    assert isinstance(ahead, C._XsAhead) and not isinstance(plain, C._XsAhead)
    for c in (0, 1, 1, 0, 1):
        assert _eq_tree(ahead(c), plain(c)), c


def test_a_second_window_compiles_nothing(model):
    first = C.run_fwd(model, ("box_T_vw",))
    fwd = adjoint._jitted_forward(model.chunk_fn)
    bwd = adjoint._jitted_backward(model.chunk_fn)
    sizes = (fwd._cache_size(), bwd._cache_size())
    assert sizes[0] >= 1 and sizes[1] >= 1
    c0 = C.compile_clock()
    start2 = START + datetime.timedelta(seconds=N_STEPS * 1800.0)
    row = C.run_grad(model, ("box_T_vw", "mld_mean"), start=start2,
                     init=model.state_of(first["_final"]), check_linear=True)
    c1 = C.compile_clock()
    assert row[0]["status"] == "ok" and row[0]["start"] == start2.isoformat()
    assert (fwd._cache_size(), bwd._cache_size()) == sizes
    new_backend = c1["backend_s"] - c0["backend_s"]
    print(f"\n[core2_adj window 2] backend compiles {c1['n_backend'] - c0['n_backend']} "
          f"({new_backend:.2f} s), chunk cache sizes {sizes}")
    assert new_backend < 10.0


def test_states_and_restartcheck(tmp_path):
    m = C.build_states_model(START, RESTART, chunk_steps=1)
    out = tmp_path / "states"
    row = C.run_states(m, out, n_days=2, save_monthly=False, save_at=(0, 1, 2),
                       json_path=tmp_path / "r.jsonl")
    tags = ["fesom.2013.001.00000", "fesom.2013.001.01800", "fesom.2013.001.03600"]
    assert row["written"] == tags and row["n_daily"] == 2
    info = json.loads((out / "README.json").read_text())
    assert set(info["daily_fields"]) == set(C.DAILY_FIELDS)
    assert info["state_fields"] == list(zarr_output._all_state_fields())
    day = np.load(out / "daily" / "diag_2013_001_00000.npz")
    assert set(day.files) == set(C.DAILY_FIELDS)
    for k in ("tke_floor_frac", "convective_frac"):
        assert np.all((day[k] >= 0.0) & (day[k] <= 1.0))
    assert float(day["mld"].max()) > 0.0
    # the saved start State is the loaded restart, bit for bit, both readers
    st0, meta = C.load_state(out / tags[0])
    assert meta["calendar_date"] == "2013-doy001"
    assert _eq_tree(st0, m.s0)
    part = partit.synth_serial(m.mesh.nod2D, m.mesh.elem2D, m.mesh.edge2D)
    rr, rmeta = zarr_output.read_restart(out / tags[1], m.mesh, part)
    st1, _ = C.load_state(out / tags[1])
    ref = shard_mesh.partition_state(st1, part)
    for f in zarr_output._all_state_fields():
        a, b = np.asarray(getattr(rr, f)), np.asarray(getattr(ref, f))
        assert np.array_equal(a.reshape(b.shape), b), f
    with pytest.raises(FileExistsError):
        C.run_states(m, out, n_days=1, save_monthly=False, save_at=(0,))
    # restart from the 00:30 State and rerun the step twice against the continuous run
    rc = C.run_restartcheck(m, out, date="2013-01-01T00:30:00")
    assert rc["pass"], rc
    assert rc["tie_margin"] == C.RESTART_TIE_MARGIN == 1.05 and rc["over_margin"] == []
    assert all(v == 0.0 for v in rc["vs_continuous"].values())
    assert all(v == 0.0 for v in rc["state_vs_next_saved"].values())
    print(f"\n[core2_adj states] {row['compile_s']:.1f} s compile, per chunk "
          f"{[round(s, 1) for s in row['seconds_per_day']]} s; restartcheck bitwise")


def test_restartcheck_tie_margin():
    """The restartcheck rule on its own: a difference passes up to 1.05 × the repeat spread, a
    zero spread needs a zero difference, and NaN never passes. The two near-ties of the A100
    checks of 2013-07-01 (surface speed 1.4e-5 relative over, column-minimum N² 3 % over) pass;
    a field 6 % over does not (planted)."""
    m = C.RESTART_TIE_MARGIN
    spread = {"surface_speed": 2.75869e-2, "n2_colmin": 3.0722743e-06, "sst": 0.0, "mld": 2.83}
    ties = {"surface_speed": 2.75873e-2, "n2_colmin": 3.1720345e-06, "sst": 0.0, "mld": 2.28}
    assert C._within_spread(ties, spread, m) == []
    assert C._within_spread(ties, spread, 1.0) == ["surface_speed", "n2_colmin"], \
        "the strict rule flags exactly the two ties"
    planted = dict(ties, mld=1.06 * spread["mld"], sst=1e-300, surface_speed=float("nan"))
    assert C._within_spread(planted, spread, m) == ["surface_speed", "sst", "mld"]


def test_the_import_guard_fires_for_another_fesom_jax(tmp_path):
    fake = tmp_path / "fake" / "fesom_jax"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("")
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "fake"), JAX_PLATFORMS="cpu")
    p = subprocess.run([sys.executable, str(DRIVER), "fwd", "--help"], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode != 0
    assert "not to this checkout" in p.stderr
    ok = subprocess.run([sys.executable, str(DRIVER), "fwd", "--help"], cwd=str(tmp_path),
                        env=dict(env, PYTHONPATH=str(ROOT)), capture_output=True, text=True,
                        timeout=300)
    assert ok.returncode == 0 and "restartcheck" in ok.stdout
