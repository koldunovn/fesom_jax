"""The TKE-constant calibration driver ``scripts/calib/tke_lbfgs.py`` on two-step CORE2 windows.

Checked: the cost's misfit is the earlier WOA calibration's weighted misfit, rebuilt from that
script's own WOA loader and its inline weight formula, on a fixed synthetic MLD field; the MLD
accumulator's two conventions for fully mixed columns; two windows × two optimiser iterations
compile the chunk VJP exactly once (and nothing at all after the first window); at the twin truth
the data term is exactly zero, so ``J = ½·u_truthᵀu_truth`` bit for bit; ``dJ/du`` from the
adjoint matches central finite differences; the host-side Adam loop resumes from its log bit for
bit.

⚠️ HEAVY (CPU compute node, >= 200 GB, ~1 h): the CORE2 chunk forward and its VJP are compiled
once for the module (1-step chunks, 2-step windows). The file lives in the ``scripts`` group.
The WOA targets are read from their only copy (read-only), by absolute path.
"""

from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "calib"))
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
import tke_lbfgs as T  # noqa: E402

C = T.C
from fesom_jax import adjoint, obs_compare  # noqa: E402
from fesom_jax import adjoint_costs as AC  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402

RESTART = C.RESTART_ARCHIVE / "fesom.2013.001.00000"
HAVE_CORE2 = (ROOT / "data" / "mesh_core2").is_dir() and RESTART.is_dir() and T.WOA_NPZ.is_file()
core2 = pytest.mark.skipif(not HAVE_CORE2, reason="CORE2 mesh / 2013 restart / WOA targets missing")

N_STEPS = 2
JAN = datetime.datetime(2013, 1, 1)
FEB = datetime.datetime(2013, 2, 1)


# ---------------------------------------------------------------------------
# model-free: the host-side Adam loop


def _quad(u):
    h = np.array([1.0, 4.0, 0.5, 2.0])
    return 0.5 * float(np.sum(h * (u - 1.0) ** 2)), h * (u - 1.0), {"k": 1}


def test_adam_logs_and_resumes_bitwise(tmp_path):
    full = tmp_path / "full.jsonl"
    r = T.adam_minimize(_quad, np.zeros(4), n_evals=8, lr=0.3, log_path=full)
    recs = [json.loads(line) for line in full.read_text().splitlines()]
    assert [x["type"] for x in recs] == ["header"] + ["eval"] * 8 + ["stop"]
    assert recs[0]["optimizer"] == "adam" and recs[0]["lr"] == 0.3
    lrs = [x["lr"] for x in recs[1:9]]
    assert lrs[0] == pytest.approx(0.3) and all(a > b for a, b in zip(lrs, lrs[1:]))
    assert r["f_best"] < recs[1]["f"] and r["n_eval"] == 8
    # interrupted after 3 evaluations, then resumed: the same iterates, bit for bit
    part = tmp_path / "part.jsonl"
    part.write_text("\n".join(full.read_text().splitlines()[:4]) + "\n")
    r2 = T.adam_minimize(_quad, np.zeros(4), n_evals=8, lr=0.3, log_path=part)
    evs = [json.loads(x) for x in part.read_text().splitlines()]
    evs = [x for x in evs if x["type"] == "eval"]
    assert [e["u"] for e in evs] == [x["u"] for x in recs[1:9]]
    assert r2["u_next"] == r["u_next"]
    # a log made with other settings is refused
    with pytest.raises(ValueError, match="not an Adam log"):
        T.adam_minimize(_quad, np.zeros(4), n_evals=8, lr=0.2, log_path=full)


# ---------------------------------------------------------------------------
# CORE2


@pytest.fixture(scope="module")
def model():
    return T.build_calib_model(JAN, N_STEPS, 1, restart=RESTART,
                               adjoint_cfg=AdjointConfig.exact())


@pytest.fixture(scope="module")
def two_windows(model):
    """Jan and Feb windows, both started from the Jan-1 State (a compile/plumbing check, not
    physics: the Feb window's target and weights are February's)."""
    return T.Calibration(model, [JAN, FEB], states={JAN: model.s0, FEB: model.s0})


@core2
def test_two_windows_two_iterations_compile_once(model, two_windows, tmp_path):
    """Runs first in the module, so it sees the process's first chunk compilation."""
    bwd = adjoint._jitted_backward(model.chunk_fn)
    miss0 = adjoint._jitted_backward.cache_info().misses
    size0 = bwd._cache_size()
    out = tmp_path / "run"
    calib = T.Calibration(model, [JAN, FEB], states={JAN: model.s0, FEB: model.s0},
                          out_dir=out, hmap=two_windows.hmap)
    r = T.run_lbfgs(calib, out, maxiter=2, maxeval=2)
    rows = [json.loads(x) for x in (out / "windows.jsonl").read_text().splitlines()]
    print(f"\n[tke_lbfgs compile] VJP cache {size0} -> {bwd._cache_size()}, lru misses "
          f"+{adjoint._jitted_backward.cache_info().misses - miss0}; per window: "
          + ", ".join(f"e{x['eval']}/{x['month']}: {x['n_compiles']} compiles "
                      f"{x['compile_s']:.1f} s, run {x['run_s']:.1f} s" for x in rows))
    assert r.n_eval == 2 and len(rows) == 4
    assert [(x["eval"], x["month"]) for x in rows] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert bwd._cache_size() == 1 and size0 == 0
    assert adjoint._jitted_backward.cache_info().misses - miss0 == 0   # fixture's lookup
    assert rows[0]["n_compiles"] > 0
    assert [x["n_compiles"] for x in rows[1:]] == [0, 0, 0]
    # the guard: "full" on each window's first evaluation, then off
    assert [x["check_linear"] for x in rows] == ["full", "full", "False", "False"]
    assert all(x["linear_check"]["mode"] == "full" for x in rows[:2])
    assert all(x["linear_check"] is None for x in rows[2:])
    # provenance and cost settings in every row; the optimiser log
    for x in rows:
        assert not [k for k in C.PROVENANCE_KEYS if k not in x]
        assert x["calib"]["sigma_obs_m"] == 20.0 and x["calib"]["n_eff"] == 500.0
        assert x["calib"]["mld_invalid"] == "depth" and x["ice_mode"] == "prescribed"
        assert x["n_steps"] == N_STEPS and x["chunk_steps"] == 1
    log = [json.loads(x) for x in (out / "iterations.jsonl").read_text().splitlines()]
    assert [x["type"] for x in log] == ["header", "eval", "eval", "stop"]
    assert log[0]["run_info"]["calib"]["windows"] == [JAN.isoformat(), FEB.isoformat()]
    ev = log[1]
    assert ev["f"] == ev["aux"]["J_data"] + ev["aux"]["J_prior"]
    assert ev["aux"]["J_data"] == math.fsum(w["J"] for w in ev["aux"]["windows"])
    # the host sums: data gradient = Σ_w g_w (correctly rounded), g = g_data + u (prior on)
    for e in log[1:3]:
        per = [w["g"] for w in e["aux"]["windows"]]
        assert len(per) == 2
        assert e["aux"]["g_data"] == [math.fsum(c) for c in zip(*per)]
        assert e["g"] == (np.array(e["aux"]["g_data"]) + np.array(e["u"])).tolist()
    assert [w["g"] for w in ev["aux"]["windows"]] == [x["g_w"] for x in rows[:2]]
    assert ev["aux"]["compile_s"] > 0 and log[2]["aux"]["compile_s"] < 5.0
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["optimizer"] == "lbfgs" and cfg["calib"]["prior"] is True
    # negative control of the counters: a NEW chunk function is a new cache entry
    other = C._make_chunk_fn(model, rematerialize=True)
    adjoint._jitted_backward(other)
    assert adjoint._jitted_backward.cache_info().misses - miss0 == 1


@core2
def test_misfit_equals_the_d2a_formula(model, two_windows):
    import core2_paper_calib_tke_obs as d2a
    d2a.WOA_NPZ = T.WOA_NPZ                       # its only copy, read-only
    mesh = model.mesh
    lon, lat = C._lonlat_deg(mesh)
    field = 40.0 + 30.0 * np.cos(np.radians(lat)) ** 2 + 5.0 * np.sin(np.radians(3.0 * lon))
    field = np.where(np.asarray(mesh.node_layer_mask[:, 0]), field, 0.0)
    carry = (None, {"mld": jnp.asarray(N_STEPS * field)})
    for w in two_windows.windows:
        mine = float(two_windows.loss_fn(w)(carry)) / two_windows.scale
        Hmap, (woa_mld, woa_mld_ok, woa_sst, _), _ = d2a.load_woa(mesh, w.month)
        _, cell_valid = obs_compare.to_obs_surface(jnp.ones((mesh.nod2D,)), Hmap)
        wt = Hmap.cell_area * woa_mld_ok * (woa_sst > -1.0) * cell_valid
        cell, _ = obs_compare.to_obs_surface(jnp.asarray(field), Hmap)
        ref = float(obs_compare.misfit(cell, woa_mld, wt))
        print(f"\n[tke_lbfgs misfit] month {w.month}: F {mine!r} D2a {ref!r} "
              f"rel {abs(mine - ref) / ref:.2e}")
        assert abs(mine - ref) <= 1e-14 * ref
        assert np.array_equal(np.asarray(w.weight), np.asarray(wt))
        # the comparison discriminates: dropping the ice mask changes the misfit
        wt2 = Hmap.cell_area * woa_mld_ok * cell_valid
        assert abs(float(obs_compare.misfit(cell, woa_mld, wt2)) - ref) > 1e-6 * ref
    assert (two_windows.windows[0].target != two_windows.windows[1].target).any()


@core2
def test_mld_invalid_conventions(model):
    s0 = jax.device_put(model.s0)
    depth = np.asarray(T.mld_accumulators("depth")(model.mesh)["mld"](s0))
    zero = np.asarray(T.mld_accumulators("zero")(model.mesh)["mld"](s0))
    assert np.array_equal(depth, np.asarray(AC.mld_field(s0, model.mesh, invalid="column")))
    assert np.array_equal(zero, np.asarray(AC.mld_field(s0, model.mesh, invalid="zero")))
    mixed = depth != zero
    print(f"\n[tke_lbfgs mld] columns mixed to the bottom on 2013-01-01: {int(mixed.sum())}")
    assert mixed.any() and (zero[mixed] == 0.0).all() and (depth[mixed] > 0.0).all()
    assert model.accumulators.keys() == {"mld"}


@core2
def test_twin_truth_is_exact(model, two_windows, tmp_path):
    cs = two_windows.cs
    u_t = cs.to_u(T.TRUTH_DEFAULT)
    kw = dict(mode="twin", u_truth=u_t, states={JAN: model.s0, FEB: model.s0},
              out_dir=tmp_path, hmap=two_windows.hmap)
    twin = T.Calibration(model, [JAN, FEB], **kw)
    J, g, aux = twin.fun_and_grad(u_t)
    print(f"\n[tke_lbfgs twin] J {J!r} prior {0.5 * math.fsum((u_t * u_t).tolist())!r} "
          f"J_data {aux['J_data']!r} g_data {aux['g_data']}")
    assert aux["J_data"] == 0.0 and all(w["misfit_m2"] == 0.0 for w in aux["windows"])
    assert J == 0.5 * math.fsum((u_t * u_t).tolist())
    assert np.all(np.asarray(aux["g_data"]) == 0.0) and np.array_equal(g, u_t)
    # away from the truth the data term is positive (the twin is not trivially zero)
    Jp, auxp = twin.fun(u_t + np.array([0.05, 0.0, 0.0, 0.0]))
    print(f"[tke_lbfgs twin] J_data at u_truth + 0.05 e_ck: {auxp['J_data']!r}")
    assert auxp["J_data"] > 0.0
    # the targets are kept with the run and reused; another truth is refused
    again = T.Calibration(model, [JAN, FEB], **kw)
    for a, b in zip(twin.windows, again.windows):
        assert np.array_equal(np.asarray(a.target), np.asarray(b.target))
    with pytest.raises(ValueError, match="another truth"):
        T.Calibration(model, [JAN, FEB], **{**kw, "u_truth": np.zeros(4)})
    # another run shares these targets (targets_from): same arrays, nothing written there
    shared = T.Calibration(model, [JAN, FEB], **{**kw, "out_dir": tmp_path / "other",
                                                 "targets_from": tmp_path / "twin_targets.npz"})
    for a, b in zip(twin.windows, shared.windows):
        assert np.array_equal(np.asarray(a.target), np.asarray(b.target))
    assert not (tmp_path / "other" / "twin_targets.npz").exists()
    assert shared.config()["twin_targets_from"] == str(tmp_path / "twin_targets.npz")
    with pytest.raises(FileNotFoundError):
        T.Calibration(model, [JAN, FEB], **{**kw, "targets_from": tmp_path / "missing.npz"})
    # a subset of the windows takes its targets by start date
    sub = T.Calibration(model, [FEB], **{**kw, "out_dir": tmp_path / "sub",
                                         "targets_from": tmp_path / "twin_targets.npz"})
    assert np.array_equal(np.asarray(sub.windows[0].target), np.asarray(twin.windows[1].target))
    MAR = datetime.datetime(2013, 3, 1)
    with pytest.raises(ValueError, match="another truth"):
        T.Calibration(model, [MAR], **{**kw, "out_dir": tmp_path / "mar", "states": {MAR: model.s0},
                                       "targets_from": tmp_path / "twin_targets.npz"})


@core2
def test_gradient_matches_central_fd(model, two_windows, tmp_path):
    """dJ/du of the calibration cost (prior off: the data term) against central differences.

    The window-mean MLD is not a smooth function of the constants: the density-threshold pick
    of the FIRST crossing jumps between levels. A jump inside ``[u - h, u + h]`` adds
    ``O(jump / h)`` to one FD value, and its probability is ``O(h)``, so the FD keeps an ``O(1)``
    bias that does not shrink with ``h``. (Columns that become fully mixed no longer jump: they
    enter at the deepest wet mid-layer depth, the limit of a valid crossing.) Measured on this
    window with that fill (job 27740260, ``tke_lbfgs.py fdrule``): c_eps, cd and alpha agree to
    <= 1e-3 for ``h`` in [1e-5, 3e-2]; c_k agrees to 2e-3 at h = 1e-3 and 3e-4, to 1e-2..3e-2
    at h >= 3e-3 (1-3 columns jump by > 1 m) and to 2.5e-2 / 0.18 at h = 3e-5 / 1e-5, where
    no column jumps by more than 1.2e-3 m and the SST control degrades too (1.2e-4 at 1e-5):
    a forward that is not smooth at ~1e-9 relative, not the MLD pick. With the jumping
    columns held fixed c_k agrees to <= 3.5e-3 for h in [1e-4, 3e-2] (2e-2 at 3e-5). Hence the bound 5e-3 for
    the MLD cost, and a second, strict check on a smooth functional of the same model and
    controls (the terminal area-mean SST, <= 1e-4 at h = 3e-4, measured <= 1e-5)."""
    u = np.array([0.3, -0.2, 0.1, 0.2])
    calib = T.Calibration(model, [JAN], prior=False, states={JAN: model.s0}, out_dir=tmp_path,
                          hmap=two_windows.hmap)
    rows = T.run_fd(calib, u, hs=(1e-3, 3e-4))
    hrows = [r for r in rows if r["row"] == "h"]
    J0 = hrows[0]["J"]
    for name in calib.cs.names:
        rs = [r for r in hrows if r["direction"] == f"u_{name}"]
        best = min(r["rel_err"] for r in rs)
        print(f"\n[tke_lbfgs fd] {name}: ad {rs[0]['ad']!r} "
              + ", ".join(f"h={r['h']:g} fd {r['fd']!r} rel {r['rel_err']:.2e}" for r in rs))
        assert abs(rs[0]["ad"]) > 1e-6 * abs(J0), f"{name}: zero gradient"
        assert best <= 5e-3, f"{name}: adjoint vs FD {best:.2e}"
        # a planted chain-rule error (the gradient in ln p instead of u) would be caught
        assert min(abs(r["fd"] - r["ad"] / calib.cs.sigma[0]) / abs(r["fd"]) for r in rs) > 0.1
    assert len([r for r in rows if r["row"] == "plateau"]) == calib.cs.n
    assert all(r["h"] in (1e-3, 3e-4) and r["direction"].startswith("u_") for r in hrows)
    # the smooth functional: terminal area-mean SST, same compiled programs, same controls
    sst = AC.terminal(lambda s: AC.sst_area_mean(s, model.mesh))
    g = np.asarray(C.run_grad(model, (sst,), params=u.copy(), init=model.s0,
                              check_linear=False)[0]["_result"]["grad"]).reshape(-1)
    w = calib.windows[0]

    def J(x):
        fin = calib._forward(x, w)
        v = float(sst(fin))
        C._free(fin)
        return v
    h = 3e-4
    for i, name in enumerate(calib.cs.names):
        e = np.zeros(4)
        e[i] = h
        fd = (J(u + e) - J(u - e)) / (2 * h)
        print(f"[tke_lbfgs fd sst] {name}: ad {g[i]!r} fd {fd!r} rel {abs(fd - g[i]) / abs(g[i]):.2e}")
        assert abs(fd - g[i]) <= 1e-4 * abs(g[i])
    # the prior adds exactly u to the gradient, and ½uᵀu to J
    on = T.Calibration(model, [JAN], prior=True, states={JAN: model.s0}, hmap=two_windows.hmap)
    Jon, gon, _ = on.fun_and_grad(u)
    Joff, goff, _ = calib.fun_and_grad(u)
    assert np.array_equal(gon, goff + u) and Jon == Joff + 0.5 * math.fsum((u * u).tolist())


@core2
def test_cli_calib_free_and_holdout(tmp_path):
    """The command line for a one-constant real fit with a held-out month: only c_k moves,
    the hold-out rows compare the result with the start (its own model: one more compile)."""
    out = tmp_path / "ck"
    rc = T.main(["calib", "--mode", "real", "--months", "1", "--window-steps", "2",
                 "--chunk-steps", "1", "--free", "tke_c_k", "--maxiter", "2", "--maxeval", "2",
                 "--holdout-months", "2", "--out", str(out)])
    assert rc == 0
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["free"] == ["tke_c_k"]
    ev = [json.loads(x) for x in (out / "iterations.jsonl").read_text().splitlines()]
    ev = [x for x in ev if x["type"] == "eval"]
    assert len(ev) == 2 and all(len(e["u"]) == 1 and len(e["g"]) == 1 for e in ev)
    for e in ev:
        assert e["aux"]["u_full"][1:] == [0.0, 0.0, 0.0]
        assert e["g"][0] == e["aux"]["g_full"][0]
    assert ev[1]["u"] != ev[0]["u"]                          # c_k moved
    hold = [json.loads(x) for x in (out / "holdout.jsonl").read_text().splitlines()]
    assert [h["point"] for h in hold] == ["start", "result"]
    assert [w["month"] for w in hold[0]["windows"]] == [2]
    assert hold[0]["u"] == [0.0, 0.0, 0.0, 0.0] and hold[1]["u"][1:] == [0.0, 0.0, 0.0]
    assert "rel_change_vs_start" in hold[1] and hold[0]["misfit_sum_m2"] > 0.0


@core2
def test_cli_fwd(tmp_path):
    """The command line end to end (its own model: one more forward compilation)."""
    out = tmp_path / "cli"
    rc = T.main(["fwd", "--months", "1", "--window-steps", "2", "--chunk-steps", "1",
                 "--p", "0.1,0.7,3.75,30", "--mld-invalid", "zero", "--out", str(out)])
    assert rc == 0
    row = json.loads((out / "fwd.jsonl").read_text().splitlines()[0])
    assert not [k for k in C.PROVENANCE_KEYS if k not in row]
    assert row["u"] == [0.0, 0.0, 0.0, 0.0] and row["calib"]["mld_invalid"] == "zero"
    assert row["J"] == row["J_data"] + row["J_prior"] and row["J_prior"] == 0.0
    assert row["windows"][0]["month"] == 1 and row["J_data"] > 0.0


# ---------------------------------------------------------------------------
# the window-length FD rule and identifiability (F5)


def test_jumping_nodes_flags_jumps_not_smooth_nodes():
    h = 1e-3
    u = np.array([-h, 0.0, h])
    smooth = 50.0 + 20.0 * u + 300.0 * u ** 2          # m''·h² ≪ m'·h
    jump = 50.0 + 20.0 * u + np.where(u > 0.5 * h, 10.0, 0.0)
    flat = np.full(3, 80.0) + np.array([0.0, 1e-9, -1e-9])   # round-off only
    kink = 50.0 + np.where(u > 0.0, 5.0, -1.0) * u
    rows = np.stack([smooth, jump, flat, kink], 1)          # [3 points, 4 nodes]
    got = T.jumping_nodes(rows[0], rows[1], rows[2])
    assert got.tolist() == [False, True, False, True]
    # a larger atol drops the kink (second difference 6e-3 m)
    assert T.jumping_nodes(rows[0], rows[1], rows[2], atol=1e-2).tolist() == \
        [False, True, False, False]


def test_plateau_needs_three_agreeing_step_sizes():
    hs = [1e-2, 1e-3, 1e-4, 1e-5]
    pl = T._plateau(hs, [1.0, 1.01, 0.99, 3.0], 1.0, 0.02)
    assert pl is not None and pl["n"] == 3 and pl["h_min"] == 1e-4 and pl["h_max"] == 1e-2
    assert T._plateau(hs, [1.0, 1.5, 0.5, 3.0], 1.0, 0.02) is None
    assert T._plateau(hs[:2], [1.0, 1.0], 1.0, 0.02) is None        # two rows: no plateau


def test_restrict_optimises_a_subset():
    """The subset wrapper: the fixed components stay at u0, the gradient is the free part, and
    L-BFGS on it finds the conditional minimum of a coupled quadratic."""
    A = np.array([[4.0, 1.0, 0.0, 0.0], [1.0, 3.0, 0.5, 0.0], [0.0, 0.5, 2.0, 0.3],
                  [0.0, 0.0, 0.3, 1.0]])
    b = np.array([1.0, -2.0, 0.5, 0.25])

    def fg(u):
        return 0.5 * float(u @ A @ u) - float(b @ u), A @ u - b, {"k": 1}
    u0 = np.array([0.0, 0.3, -0.2, 0.1])
    cs = T.tke_control_set()
    idx = T.free_indices(cs, ["tke_c_k"])
    assert idx == [0] and T.free_indices(cs, None) == [0, 1, 2, 3]
    assert T.free_indices(cs, ["tke_alpha", "tke_c_k"]) == [0, 3]
    red = T.restrict(fg, u0, idx)
    f, g, aux = red(np.array([0.5]))
    assert aux["u_full"] == [0.5, 0.3, -0.2, 0.1] and g.shape == (1,)
    assert g[0] == (A @ np.array(aux["u_full"]) - b)[0] and aux["k"] == 1
    from fesom_jax import optim_lbfgs
    r = optim_lbfgs.minimize(red, u0[idx], maxiter=20, maxeval=40, gtol=1e-12)
    want = (b[0] - A[0, 1:] @ u0[1:]) / A[0, 0]            # ∂f/∂u0 = 0 with the rest fixed
    assert abs(r.u[0] - want) < 1e-10
    # a subset that is not a leading block: (c_eps, alpha)
    idx2 = T.free_indices(cs, ["tke_c_eps", "tke_alpha"])
    red2 = T.restrict(fg, u0, idx2)
    f2, g2, aux2 = red2(np.array([0.7, -0.4]))
    full = np.array([0.0, 0.7, -0.2, -0.4])
    assert aux2["u_full"] == full.tolist() and np.array_equal(g2, (A @ full - b)[[1, 3]])
    r2 = optim_lbfgs.minimize(red2, u0[idx2], maxiter=40, maxeval=80, gtol=1e-12)
    M, rhs = A[np.ix_(idx2, idx2)], b[idx2] - A[np.ix_(idx2, [0, 2])] @ u0[[0, 2]]
    assert np.allclose(r2.u, np.linalg.solve(M, rhs), rtol=0, atol=1e-10)


def _mj_windows():
    rng = np.random.default_rng(0)
    months = [1, 2, 3]
    A = {m: np.diag(rng.uniform(1.0, 5.0, 4)) for m in months}
    a = {m: 0.3 * rng.normal(size=4) for m in months}

    def win(m, u):
        e = np.asarray(u) - a[m]
        return 0.5 * float(e @ A[m] @ e), A[m] @ e
    return months, win


def test_multijob_protocol_equals_the_single_process_run(tmp_path):
    """The coordinator (run_mjstep) and per-window result files reproduce the single-process
    L-BFGS bit for bit: each round names the missing windows, the 'jobs' write them, and the
    replay continues."""
    from fesom_jax import optim_lbfgs
    cs = T.tke_control_set()
    months, win = _mj_windows()
    opts = optim_lbfgs.LBFGSOptions(expand=4.0)
    kw = dict(months=months, cs=cs, prior=True, scale=1.0, u0=None, maxiter=40, maxeval=40,
              gtol=1e-10, ftol=0.0, options=opts, config={"calib": {"test": 1}})
    out = tmp_path / "mj"
    rounds = 0
    while True:
        r = T.run_mjstep(out, **kw)
        if r["status"] != "need":
            break
        rounds += 1
        u = np.array(r["u"])
        assert r["missing"] == months                  # nothing computed yet for this eval
        for m in r["missing"]:
            J, g = win(m, u)
            T._write_json_atomic(T.mj_dir(out, r["eval"]) / f"win_{m:02d}.json",
                                 {"month": m, "u": u.tolist(), "J_w": J, "g_w": g.tolist(),
                                  "platform": "cpu"})

    def fg(u):
        per = [win(m, u) for m in months]
        jd = math.fsum(p[0] for p in per)
        gd = np.array([math.fsum(c) for c in zip(*[p[1].tolist() for p in per])])
        return jd + T._prior(u), gd + u
    ref = optim_lbfgs.minimize(fg, np.zeros(4), maxiter=40, maxeval=40, gtol=1e-10,
                               options=opts)
    assert r["status"] == ref.status == "gtol" and r["n_eval"] == ref.n_eval == rounds
    assert np.array(r["u"]).tobytes() == ref.u.tobytes()
    ev = [json.loads(x) for x in (out / "iterations.jsonl").read_text().splitlines()]
    assert sum(x["type"] == "eval" for x in ev) == rounds
    assert all(x["aux"]["platform"] == "cpu" for x in ev if x["type"] == "eval")
    assert json.loads((out / "run_config.json").read_text())["options"]["expand"] == 4.0


class _FakeWindowCalib:
    """Duck-typed stand-in for a one-window Calibration (what run_winjob touches)."""

    def __init__(self, month, win):
        self.windows = [T.Window(datetime.datetime(2013, month, 1), month, None, None, None)]
        self._win, self.scale, self.first_check = win, 1.0, "full"

    def grad_window(self, u, w, check_linear=False):
        J, g = self._win(w.month, u)
        return J, np.asarray(g), {"platform": "cpu", "compile_s": 0.0, "run_s": 0.0,
                                  "slurm_job": "test"}

    def physical(self, u):
        return list(u)

    def config(self):
        return {"fake": True}


@pytest.mark.parametrize("optimizer", ["lbfgs", "adam"])
def test_persistent_workers_reproduce_the_single_process_run(tmp_path, optimizer):
    """Three concurrent window workers (threads) coordinate through eval_<k>/coord.lock: every
    evaluation is posted once, and the optimiser's iterates equal the single-process run's."""
    import threading
    from fesom_jax import optim_lbfgs
    cs = T.tke_control_set()
    months, win = _mj_windows()
    opts = optim_lbfgs.LBFGSOptions(expand=4.0)
    coord = dict(months=months, cs=cs, prior=True, scale=1.0, u0=None, maxiter=12, maxeval=12,
                 gtol=0.0, ftol=0.0, options=opts, config={"calib": {"test": 1}},
                 optimizer=optimizer, lr=0.3, n_evals=12)
    out = tmp_path / optimizer
    errs = []

    def work(m):
        try:
            T.run_winloop(_FakeWindowCalib(m, win), out, coord, poll_s=0.01, max_wait_s=60.0)
        except Exception as e:                            # surfaced below
            errs.append(e)
    th = [threading.Thread(target=work, args=(m,)) for m in months]
    for t in th:
        t.start()
    for t in th:
        t.join(120)
    assert not errs and not any(t.is_alive() for t in th)
    stop = T.mj_finished(out)
    assert stop is not None

    def fg(u):
        per = [win(m, u) for m in months]
        jd = math.fsum(p[0] for p in per)
        gd = np.array([math.fsum(c) for c in zip(*[p[1].tolist() for p in per])])
        return jd + T._prior(u), gd + u
    ev = [json.loads(x) for x in (out / "iterations.jsonl").read_text().splitlines()]
    ev = [x for x in ev if x["type"] == "eval"]
    if optimizer == "lbfgs":
        ref = optim_lbfgs.minimize(fg, np.zeros(4), maxiter=12, maxeval=12, options=opts)
        assert np.array(stop["u"]).tobytes() == ref.u.tobytes() and len(ev) == ref.n_eval
    else:
        ref = T.adam_minimize(fg, np.zeros(4), n_evals=12, lr=0.3)
        assert stop["u_last"] == ref["u_last"] and len(ev) == 12
    # one coordinator run per posted evaluation, plus the one that stopped the run
    locks = sorted(out.glob("eval_*/coord.lock"))
    assert len(locks) == len(ev) + 1


def test_multijob_refuses_foreign_or_mixed_window_files(tmp_path):
    cs = T.tke_control_set()
    months, win = _mj_windows()
    fg = T.mj_fun_and_grad(tmp_path, months, prior=False, cs=cs, scale=1.0)
    u = np.zeros(4)
    with pytest.raises(T.NeedEvaluation) as e:
        fg(u)
    assert e.value.k == 1 and e.value.missing == months
    d = T.mj_dir(tmp_path, 1)
    for m, plat in zip(months, ["cpu", "cpu", "gpu"]):
        J, g = win(m, u)
        T._write_json_atomic(d / f"win_{m:02d}.json", {"month": m, "u": u.tolist(), "J_w": J,
                                                       "g_w": g.tolist(), "platform": plat})
    with pytest.raises(ValueError, match="several platforms"):
        fg(u)
    with pytest.raises(ValueError, match="another point"):
        fg(u + 1e-3)                                  # eval_001/u.json holds u


def test_wall_budget_pauses_and_the_resume_is_bitwise(tmp_path, monkeypatch):
    """--wall-budget-s: the budgeted callback refuses an evaluation that would overrun, the log
    stays consistent, and a resumed run ends where an uninterrupted one does (both optimisers)."""
    from fesom_jax import optim_lbfgs
    clock = [0.0]
    monkeypatch.setattr(T, "_T_START", 0.0)
    monkeypatch.setattr(T.time, "time", lambda: clock[0])

    def fg(u):                                     # each evaluation "takes" 10 s
        clock[0] += 10.0
        return _quad(u)
    opts = optim_lbfgs.LBFGSOptions(expand=4.0)
    full = optim_lbfgs.minimize(fg, np.zeros(4), maxiter=9, maxeval=9, options=opts)
    log = tmp_path / "l.jsonl"
    clock[0] = 0.0
    with pytest.raises(T.WallBudget):
        optim_lbfgs.minimize(T.wall_budgeted(fg, 35.0, 10.0), np.zeros(4), maxiter=9,
                             maxeval=9, options=opts, log_path=log)
    ev = [json.loads(x) for x in log.read_text().splitlines()]
    assert sum(x["type"] == "eval" for x in ev) == 3      # 10, 20, 30 s; the 4th would end at 40
    clock[0] = 0.0
    r = optim_lbfgs.minimize(T.wall_budgeted(fg, 1e9, 10.0), np.zeros(4), maxiter=9, maxeval=9,
                             options=opts, log_path=log)
    assert r.u.tobytes() == full.u.tobytes() and r.n_eval == full.n_eval
    alog = tmp_path / "a.jsonl"
    ref = T.adam_minimize(fg, np.zeros(4), n_evals=7, lr=0.3)
    clock[0] = 0.0
    with pytest.raises(T.WallBudget):
        T.adam_minimize(T.wall_budgeted(fg, 25.0, 10.0), np.zeros(4), n_evals=7, lr=0.3,
                        log_path=alog)
    clock[0] = 0.0
    r2 = T.adam_minimize(T.wall_budgeted(fg, 1e9, 10.0), np.zeros(4), n_evals=7, lr=0.3,
                         log_path=alog)
    assert r2["u_last"] == ref["u_last"] and r2["n_eval"] == 7
    # a stopped run is recognised (the driver then exits before building a model)
    assert T.mj_finished(alog.parent / "none") is None
    (tmp_path / "iterations.jsonl").write_text(alog.read_text())
    assert T.mj_finished(tmp_path)["type"] == "stop"


def test_multijob_nonfinite_windows_and_out_of_domain_points(tmp_path):
    """A window file with non-finite values (stored as strings) gives a non-finite evaluation
    (the line search then shortens the step) instead of an error; a point outside the controls'
    domain is a failed evaluation that posts nothing and runs no window."""
    cs = T.tke_control_set()
    months, win = _mj_windows()
    fg = T.mj_fun_and_grad(tmp_path, months, prior=True, cs=cs, scale=1.0)
    with pytest.raises(T.NeedEvaluation):
        fg(np.zeros(4))
    d = T.mj_dir(tmp_path, 1)
    for m in months:
        J, g = win(m, np.zeros(4))
        gw = ["nan"] * 4 if m == 2 else g.tolist()
        T._write_json_atomic(d / f"win_{m:02d}.json", {"month": m, "u": [0.0] * 4, "J_w": J,
                                                       "g_w": gw, "platform": "cpu"})
    f, g, aux = fg(np.zeros(4))
    assert np.isfinite(f) and np.all(np.isnan(g))
    far = np.array([0.0, 60.0, 0.0, 0.0])                 # |u| > 50
    f2, g2, aux2 = fg(far)
    assert np.isnan(f2) and np.all(np.isnan(g2)) and "out_of_domain" in aux2
    assert not T.mj_dir(tmp_path, 2).exists()            # nothing posted
    f3, _, aux3 = fg(np.array([0.0, np.inf, 0.0, 0.0]))
    assert np.isnan(f3) and "out_of_domain" in aux3


def test_mj_drive_batches_whole_runs_in_priority_order(tmp_path):
    """scripts/calib/mj_drive.py: jobs of whole runs up to the per-job cap, in the given order,
    nothing beyond the room left; a run larger than the cap is split; '#reset' restarts the
    failure count; held jobs are neither active nor failures."""
    import mj_drive as MD
    ready = [("a", 5, list(range(1, 13))), ("b", 3, list(range(1, 13))),
             ("c", 2, [1, 3, 5, 7, 9, 11]), ("d", 2, [2, 4, 6, 8, 10, 12]), ("e", 1, [1])]
    jobs = MD.plan_batches(ready, 24, 72)
    assert [len(j) for j in jobs] == [24, 13]
    assert {n for n, _, _ in jobs[0]} == {"a", "b"} and {n for n, _, _ in jobs[1]} == {"c", "d", "e"}
    assert [len(j) for j in MD.plan_batches(ready, 24, 30)] == [24, 6]   # room caps the total
    assert [len(j) for j in MD.plan_batches([("x", 1, list(range(30)))], 24, 72)] == [24, 6]
    assert MD.plan_batches(ready, 24, 0) == []
    jf = tmp_path / "job_01.txt"
    jf.write_text("11\n12\n#reset\n13\n")
    assert MD.jobs_after_reset(jf) == ["13"] and MD.jobs_after_reset(tmp_path / "no") == []
    d = tmp_path / "run" / "eval_002"
    d.mkdir(parents=True)
    (d / "win_01.json").write_text("{}")
    (d / "job_02.txt").write_text("21\n")                  # running
    (d / "job_03.txt").write_text("31\n32\n")             # ended twice
    (d / "job_04.txt").write_text("41\n42\n")             # ended once, one held
    (d / "job_05.txt").write_text("51\n52\n#reset\n")    # reset: idle
    states = {"21": "RUNNING", "31": "FAILED", "32": "COMPLETED", "41": "FAILED", "42": "HELD",
              "51": "FAILED", "52": "FAILED"}
    ws = MD.window_status(tmp_path / "run", 2, [1, 2, 3, 4, 5, 6], state=states.get)
    assert ws == {"have": [1], "active": [2], "idle": [4, 5, 6], "failed": [3]}


def test_choose_lr_takes_the_lowest_final_cost():
    traj = {"0.03": {"lr": 0.03, "f": [10.0, 8.0, 6.0]},
            "0.1": {"lr": 0.1, "f": [10.0, 3.0, 2.0]},
            "0.3": {"lr": 0.3, "f": [10.0, 1.0, 4.0]},          # passed lower, ends higher
            "1": {"lr": 1.0, "f": [10.0, float("nan")]}}
    assert T.choose_lr(traj) == 0.1
    traj["0.05"] = {"lr": 0.05, "f": [10.0, 2.0]}                # a tie goes to the smaller lr
    assert T.choose_lr(traj) == 0.05
    with pytest.raises(ValueError):
        T.choose_lr({"1": {"lr": 1.0, "f": [float("inf")]}})


def test_lr_from_probe(tmp_path):
    (tmp_path / "p.json").write_text(json.dumps({"chosen_lr": 0.1, "trajectories": {}}))
    assert T.lr_from_probe(tmp_path / "p.json") == 0.1
    (tmp_path / "q.json").write_text(json.dumps({"chosen_lr": -1.0}))
    with pytest.raises(ValueError):
        T.lr_from_probe(tmp_path / "q.json")
    args = T._parse(["calib", "--optimizer", "adam", "--lr-from", str(tmp_path / "p.json")])
    assert args.lr_from == str(tmp_path / "p.json")


def test_adam_evaluates_on_the_default_device(tmp_path):
    """Only the optimiser's arithmetic is pinned to the host CPU: the model callback must see no
    default-device override, or on a GPU node the whole CORE2 forward and adjoint run on the
    CPU (they did: ~50x slower, GPU idle)."""
    seen = []

    def fg(u):
        seen.append(jax.config.jax_default_device)
        return _quad(u)
    T.adam_minimize(fg, np.zeros(4), n_evals=3, lr=0.3, log_path=tmp_path / "a.jsonl")
    assert seen == [None, None, None]
    assert jax.config.jax_default_device is None


def test_adam_pause_resumes_on_the_same_schedule(tmp_path):
    full = tmp_path / "full.jsonl"
    T.adam_minimize(_quad, np.zeros(4), n_evals=6, lr=0.3, log_path=full)
    part = tmp_path / "part.jsonl"
    r1 = T.adam_minimize(_quad, np.zeros(4), n_evals=6, lr=0.3, log_path=part, pause_after=2)
    assert r1["n_eval"] == 2
    types = [json.loads(x)["type"] for x in part.read_text().splitlines()]
    assert types == ["header", "eval", "eval", "pause"]
    T.adam_minimize(_quad, np.zeros(4), n_evals=6, lr=0.3, log_path=part)
    a = [json.loads(x) for x in full.read_text().splitlines()]
    b = [json.loads(x) for x in part.read_text().splitlines()]
    ea = [(x["u"], x["lr"]) for x in a if x["type"] == "eval"]
    eb = [(x["u"], x["lr"]) for x in b if x["type"] == "eval"]
    assert ea == eb and len(eb) == 6 and b[-1]["type"] == "stop"


@core2
def test_masked_cost_holds_the_masked_nodes(model, two_windows):
    """The masked data term equals the full one at u and differentiates without the masked
    nodes: an empty mask gives the full gradient bit for bit, a full mask a zero gradient,
    and a partial mask a different one (so the stop_gradient is live)."""
    u = np.array([0.3, -0.2, 0.1, 0.2])
    calib = T.Calibration(model, [JAN], prior=False, states={JAN: model.s0},
                          hmap=two_windows.hmap)
    w = calib.windows[0]
    nod = model.mesh.nod2D
    J, g, _ = calib.grad_window(u, w)
    J0, g0, _ = calib.grad_window(u, w, calib.masked_loss_fn(w, np.zeros(nod, bool)))
    J1, g1, _ = calib.grad_window(u, w, calib.masked_loss_fn(w, np.ones(nod, bool)))
    ff = calib.forward_fields(u, w)
    deep = ff["mld_sum"] > np.quantile(ff["mld_sum"], 0.9)          # the deepest 10 %
    Jd, gd, _ = calib.grad_window(u, w, calib.masked_loss_fn(w, deep))
    print(f"\n[tke_lbfgs masked] g {g.tolist()} | deepest-10% masked {gd.tolist()}")
    assert J0 == J and J1 == J and Jd == J
    assert np.array_equal(g0, g) and np.all(g1 == 0.0)
    assert np.linalg.norm(gd - g) > 1e-6 * np.linalg.norm(g)
    # the host-side masked cost: the masked nodes replaced by the reference
    assert calib.masked_J(w, ff["mld_sum"], deep, ff["mld_sum"]) == ff["J"]
    assert calib.masked_J(w, ff["mld_sum"] * 1.1, np.ones(nod, bool), ff["mld_sum"]) == ff["J"]


@core2
def test_fd_rule_ck_two_steps(model, two_windows, tmp_path):
    """The F5 rule on the 2-step window, c_k (the constant whose MLD-cost FD was biased by
    jumps): the rows, the per-h jump counts and the verdict. The step sizes span both regimes
    measured on this window (job 27740260): the MLD cost (a) has its plateau at h in
    [1e-4, 1e-3], the SST control (c) at h <= 3e-4 (at h >= 1e-3 a switch flip moves the
    terminal SST: 4-5e-3 off, which the masked control (c') removes)."""
    u = np.array([0.3, -0.2, 0.1, 0.2])
    calib = T.Calibration(model, [JAN], prior=False, states={JAN: model.s0}, out_dir=tmp_path,
                          hmap=two_windows.hmap)
    rows = T.run_fd_rule(calib, u, components=["tke_c_k"], hs=(1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
                         repeats=2, grad_repeats=2)
    hr = [r for r in rows if r["row"] == "h"]
    sm = [r for r in rows if r["row"] == "summary"]
    assert len(hr) == 5 and len(sm) == 1
    s = sm[0]
    for r in hr:
        print(f"\n[tke_lbfgs fdrule] h={r['h']:g} a {r['rel_a']:.2e} b {r['rel_b']:.2e} "
              f"c {r['rel_c']:.2e} c' {r['rel_c2']:.2e} jumps {r['n_jump']} (union "
              f"{r['n_jump_union']}, >1 m {r['n_jump_gt_1m']}) sst jumps {r['n_jump_sst']}")
    print(f"[tke_lbfgs fdrule] verdict {s['verdict']} / rule {s['verdict_rule']} "
          f"pa {s['plateau_a']} pb {s['plateau_b']} pc {s['plateau_c']} pc2 {s['plateau_c2']}")
    assert s["noise_J"] == 0.0 and s["noise_node_mld_m"] == 0.0         # CPU forward
    assert s["rel_spread_a"] == 0.0 and s["rel_spread_b"] == 0.0 and s["rel_spread_c"] == 0.0
    assert s["first_check"] == "full" and s["linear_check"]["mode"] == "full"
    sp = [r for r in rows if r["row"] == "spread"]
    assert len(sp) == 1 and len(sp[0]["g_a_repeats"]) == 2
    assert s["J_masked_adjoint_program"] == s["J_adjoint_program"]
    assert s["ok_c"] and s["ok_c2"] and (s["ok_a"] or s["ok_b"])
    assert s["verdict"] == "pass" and s["verdict_rule"] == "pass"
    assert all(r["n_jump"] <= r["n_jump_union"] for r in hr)
    assert not [k for k in C.PROVENANCE_KEYS if k not in s]
    assert (tmp_path / "fd_rule.jsonl").is_file()


@core2
def test_ident_tangent_gradient_matches_adjoint(model, two_windows, tmp_path):
    """The forward-mode cell Jacobian behind the Gauss-Newton Hessian reproduces the adjoint
    data gradient (prior off) — this is what makes the Hessian trustworthy — and H is
    symmetric positive semi-definite."""
    u = np.array([0.3, -0.2, 0.1, 0.2])
    calib = T.Calibration(model, [JAN], prior=False, states={JAN: model.s0}, out_dir=tmp_path,
                          hmap=two_windows.hmap)
    _, g, _ = calib.grad_window(u, calib.windows[0])
    row = T.run_ident(calib, {"x": u})["x"]
    gtl = np.array(row["g_data_tl"])
    H = np.array(row["H_data"])
    rel = np.abs(gtl - g) / np.abs(g)
    print(f"\n[tke_lbfgs ident] adjoint {g.tolist()} tangent {gtl.tolist()} rel {rel.tolist()}"
          f"\n[tke_lbfgs ident] eig {row['eigvals']}")
    assert np.all(rel <= 1e-5)
    # the row's own adjoint gradient (same program, CPU): bit for bit, and its TL comparison
    assert row["g_data_adj"] == g.tolist() and np.all(np.array(row["rel_tl_adj"]) <= 1e-5)
    assert row["J_data_fwd"] > 0.0 and len(row["windows"][0]["rel_tl_adj_w"]) == 4
    assert np.array_equal(H, H.T) and min(row["eigvals"]) >= -1e-9 * max(row["eigvals"])
    assert len(row["identifiable"]) == 4 and (tmp_path / "ident.jsonl").is_file()
