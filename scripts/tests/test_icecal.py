"""``scripts/icecal/icecal.py`` (the evaluation driver of the sea-ice calibration): eval, fwd and the
FD batch on CORE2 from the JAX-hindcast State of 2013-02-01 (``core2_adj.STATES_DIR``), against the
2013 super-observations (``runs/icecal/obs_v2``), with 4-step chunks and a 2-chunk window.

⚠️ HEAVY (CPU compute node, >= 200 GB): one CORE2 chunk forward + VJP program (the all-on model,
ice adjoint ``exact`` + frozen sweep) plus one single-step program for the independent check.
``scripts`` group.

What is pinned:

* the eval gradient w.r.t. u of two controls (``c_melt``, ``h0``) agrees with a central FD of
  ``J_data`` (plain forwards at u ± h e_i) at a step size where the FD is resolved; ``st_oce_ice``
  and ``con`` are reported, not asserted (below);
* boundary stride 2 (one span rebuilt from the stored boundary) gives the stride-1 J and
  gradient to 1e-12 relative;
* two further evaluations in the same process (another stride, another u) and all the forwards
  compile nothing new (one chunk function per process);
* ``eval --terms`` (one forward, one reverse walk, one seed per cost term): the term values are
  the total evaluation's and the term gradients sum to its gradient to 1e-12 relative; a subset
  that leaves out a non-zero term is flagged (``g_data_is_full`` false);
* ``fields.npz`` (the monthly means of the accumulators, in the score.py schema) equals an
  INDEPENDENT numpy mean of ``a_ice``/``m_ice`` recorded step by step by a plain single-step
  forward; the window starts on Feb 1, so the month index must reach the program (a month stuck
  at 0 would fill the January row);
* ``fwd --save-state-at`` writes the State at a chunk boundary in the ``load_state`` format with
  the right date and step, bitwise the chunk program's State there;
* an existing run directory is refused before anything is built; the data-free helpers.

⚠️ The CORE2 ocean–sea-ice forward is sensitive to round-off at the 1e-4 level within a few steps:
a 1-ulp change of the start ``T`` (or of ``a_ice``) moves ``a_ice`` by up to 1.3e-4, ``m_ice`` by
4.5e-4 and ``t_skin`` by up to 4.3 K (about 30 nodes) after 4 steps, in one and the same compiled
program (diagnostic job 28032950, 2013-04-01 State; the 2013-02-01 State behaves the same). Two
compiled programs of the same step (the chunk scan and a single-step loop) therefore differ at
that level too (1.0e-4 in a node's monthly mean concentration, job 28032252), and ``J_data``
carries a round-off "noise" of a few 1e-4 (of ~3e3). Hence: the independent mean is compared with
a bound set by that measured spread (and the test shows it can still tell a one-step shift of the
average apart), the FD uses the two controls with the largest gradients and u-steps 0.03-0.3, its
noise is measured in the test (J at u + 1e-12 e_j), and a step size counts only if that noise is
below a quarter of the tolerance; the saved State is compared with the same chunk program.

The FD is ASSERTED for ``c_melt`` and ``h0``, which act on the concentration directly (the melt
and lead-closing terms of the A update), and REPORTED for ``st_oce_ice`` and ``con``, which act
through the heat budget: on this 8-step window from the hindcast State their adjoint and FD part
systematically, beyond the measured noise and independently of the derivative setting —
``st_oce_ice`` adjoint -0.536 (calibration setting, job 28033085) / -0.532 (exact derivative,
job 28033589) against an FD of -0.493 … -0.510 over u-steps 0.03-0.3 (noise 0.1-0.8 %), and
``con`` 0.051 against 0.039 at u-step 0.1 (job 28032252), while ``c_melt`` agrees to 1e-3. That
gap is a property of the model's derivative (forward switches and kinks the pathwise adjoint does
not see, the plan's S1 gap (a)), the same driver code carries all ten controls, and S1 measures it
on the production windows.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "icecal"))
import icecal as I  # noqa: E402

C = I.C
STATE = C.STATES_DIR / "fesom.2013.032.00000"
START = "2013-02-01"
OBS = I.OBS_DIR / "obs_2013.npz"
K, NC = 4, 2
HAVE_DATA = (ROOT / "data" / "mesh_core2").is_dir() and STATE.is_dir() and OBS.is_file()
heavy = pytest.mark.skipif(not HAVE_DATA, reason="CORE2 mesh / 2013 State / obs_v2 missing")

#: controls of the FD check and their u-steps (u = 1 is one prior sigma: ln 2 here)
EXACT = "none"            #: tke_diag.parse_setting: AdjointConfig.exact(), nothing frozen
FD_CONTROLS = ("ice.c_melt", "ice.h0")
#: FD reported, not asserted (see the module docstring), at the two larger u-steps
FD_REPORT = ("ice.st_oce_ice", "ice.con")
FD_REPORT_HS = (3e-1, 1e-1)
FD_HS = (3e-1, 1e-1, 3e-2)
FD_RTOL = 2e-2            #: |adjoint/FD - 1| at a resolved step size
#: FD noise probes: J at u + 1e-12 e_j (a true change of ~1e-12 g_j) minus J(u)
NOISE_PROBES = ("ice.albsn", "ice.con", "ice.h0")
NOISE_SAFETY = 3.0
#: independent monthly mean vs the accumulators: 20x the measured program-to-program spread
#: (max 1.0e-4 in C, 3.2e-4 in H; job 28032252)
FIELD_TOL_C, FIELD_TOL_H = 2e-3, 6e-3
U_A = np.array([0.05, -0.05, 0.05, -0.05, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1], np.float64)


# ---------------------------------------------------------------------------
# data-free


def test_existing_run_dir_is_refused_before_the_build(tmp_path):
    d = tmp_path / "taken"
    d.mkdir()
    for action in ("eval", "fwd"):
        with pytest.raises(FileExistsError, match="exists"):
            I.main([action, "--run-dir", str(d), "--state", "/nonexistent", "--start", START,
                    "--steps", "8", "--obs", "/nonexistent.npz"])
    with pytest.raises(FileExistsError):
        I.new_dir(d)
    with pytest.raises(ValueError, match="inside the checkout"):
        I.new_dir(ROOT / "scripts" / "icecal" / "nope")


def test_parse_u_and_fd_runs():
    cs = I.ice_thermo_control_set()
    assert np.array_equal(I.parse_u(None, cs.n), np.zeros(cs.n))
    u = I.parse_u('{"con": 0.5, "ice.h0": -1}', cs.n, cs.names)
    assert u[cs.names.index("ice.con")] == 0.5 and u[cs.names.index("ice.h0")] == -1.0
    assert u.sum() == -0.5
    with pytest.raises(ValueError, match="unknown control"):
        I.parse_u('{"albw": 1}', cs.n, cs.names)
    runs = I.fd_runs(cs, U_A, ["con", "ice.albsn"], [1e-2], [1, -1], base_repeats=2)
    assert [r[0] for r in runs] == ["base_r0", "base_r1", "con_h0.01_p", "con_h0.01_m",
                                    "albsn_h0.01_p", "albsn_h0.01_m"]
    i = cs.names.index("ice.albsn")
    for tag, uu, info in runs[4:]:
        d = uu - U_A
        assert np.count_nonzero(d) == 1
        assert abs(cs.sigma[i] * d[i] - info["sign"] * 1e-2) < 1e-15   # h is in ln p
    assert I.steps_for(days=365) == 17520 and I.steps_for(steps=8) == 8
    with pytest.raises(ValueError):
        I.steps_for(days=0.01)


def test_the_calibration_setting_freezes_three_terms():
    import tke_diag
    cfg = tke_diag.parse_setting(I.SETTING)
    assert cfg.freeze_mixing and cfg.freeze_gm_slope and cfg.freeze_fct_limiter
    assert tke_diag.parse_setting(EXACT).describe() == "none"
    assert I.ICE_MODE == "exact" and I.ICE_ADJOINT == {
        "adjoint_damping": "frozen_sweep", "adjoint_theta": 0.97, "adjoint_alpha_kappa": 1.0}


def test_ice_adjoint_extras_from_the_command_line():
    assert I.ice_adjoint() == I.ICE_ADJOINT
    args = I._parse(["eval", "--run-dir", "x", "--state", "s", "--start", START, "--days", "30",
                     "--obs", "o", "--ice-cuts", "fct_flux+strength_conc", "--ice-thin-mass",
                     "100", "--ice-sweep-guard", "10"])
    extra = I._ice_extra(args)
    assert extra == {"adjoint_cuts": ("fct_flux", "strength_conc"), "adjoint_thin_mass": 100.0,
                     "adjoint_sweep_guard": 10.0}
    full = I.ice_adjoint(extra)
    assert {k: full[k] for k in I.ICE_ADJOINT} == I.ICE_ADJOINT
    assert I._ice_extra(I._parse(["eval", "--run-dir", "x", "--state", "s", "--start", START,
                                  "--days", "30", "--obs", "o"])) == {}
    # a chain stores the merged dict as JSON (tuples become lists); the extras come back out,
    # and an older chain's plain ICE_ADJOINT gives none
    stored = json.loads(json.dumps(full))
    assert I.ice_adjoint(I._extra_of(stored)) == full
    assert I._extra_of(dict(I.ICE_ADJOINT)) == {} and I._extra_of(None) == {}
    with pytest.raises(ValueError, match="not this driver"):
        I._extra_of({**I.ICE_ADJOINT, "adjoint_theta": 0.9})
    with pytest.raises(ValueError, match="not among"):
        I.ice_adjoint({"adjoint_theta": 0.9})


def test_controls_set_from_the_command_line():
    base = ["eval", "--run-dir", "x", "--state", "s", "--start", START, "--days", "30",
            "--obs", "o"]
    assert I._parse(base).controls_set == "thermo"
    a = I._parse(base + ["--controls-set", "joint"])
    cs = I.CONTROL_SETS[a.controls_set]()
    assert cs.names == I.ice_joint_control_set().names
    assert cs.names[:10] == I.ice_thermo_control_set().names
    assert cs.names[10:] == ("ice.pstar", "ice.cd_oce_ice", "ice.cd_atm_ice")
    u = I.parse_u('{"pstar": 0.3, "con": -1}', cs.n, cs.names)
    assert u[10] == 0.3 and u[4] == -1.0 and np.count_nonzero(u) == 2
    with pytest.raises(SystemExit):
        I._parse(base + ["--controls-set", "dyn"])


def test_cot_maps_hook_writes_one_file_per_term_and_boundary(tmp_path):
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    nod, elem, nl = 7, 5, 3
    st = SimpleNamespace(**{f: rng.normal(size=nod) for f in I.COT_MAP_FIELDS},
                         T=rng.normal(size=(nod, nl)), S=rng.normal(size=(nod, nl)),
                         uv=rng.normal(size=(elem, nl, 2)))
    hook = I.cot_maps_hook(tmp_path / "cm", ["J_C_NH", "J_H_NH"])
    hook(1, 12, (st, None))
    with np.load(tmp_path / "cm" / "J_H_NH" / "b0012.npz") as z:
        assert set(z.files) == set(I.COT_MAP_FIELDS) | {"T_col", "S_col", "uv_col"}
        assert np.array_equal(z["m_snow"], st.m_snow)
        assert np.allclose(z["T_col"], np.linalg.norm(st.T, axis=1))
        assert z["uv_col"].shape == (elem,)
    assert not any((tmp_path / "cm" / "J_C_NH").iterdir())
    with pytest.raises(ValueError, match="needs --terms"):
        I.run_eval(None, tmp_path / "e", np.zeros(10), cot_maps=True)
    assert I._parse(["eval", "--run-dir", "x", "--state", "s", "--start", START, "--days", "1",
                     "--obs", "o", "--terms", "J_C_NH", "--cot-maps"]).cot_maps


def test_git_head_file_matches_the_checkout():
    import subprocess
    got = I.git_head_file(ROOT)
    want = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    assert got["head"] == want and len(want) == 40


# ---------------------------------------------------------------------------
# CORE2


@pytest.fixture(scope="module")
def cal():
    return I.IceCal(STATE, START, K * NC, obs=OBS, chunk_steps=K)


@pytest.fixture(scope="module")
def evals(cal, tmp_path_factory):
    root = tmp_path_factory.mktemp("icecal_eval")
    e1 = I.run_eval(cal, root / "e1", U_A, stride=1, check_linear="full")
    e2 = I.run_eval(cal, root / "e2", U_A, stride=2)
    e3 = I.run_eval(cal, root / "e3", np.zeros(10), stride=1)
    return root, e1, e2, e3


@heavy
def test_joint_controls_at_u0_are_the_configured_ice(cal):
    """IceCal refuses a control whose p0 is not the model's configured value; the three dynamic
    controls of the joint set pass that check on the calibration's CORE2 model."""
    icfg = cal.model.phys["ice_cfg"]
    for c in I.ice_joint_control_set().controls[10:]:
        assert float(getattr(icfg, c.name[len(I.ICE_PREFIX):])) == c.p0, c.name


@heavy
def test_eval_outputs_and_month_table(cal, evals):
    root, e1, _, _ = evals
    assert cal.months.tolist() == [1, 1]
    for f in ("eval.json", "fields.npz", "trace.json"):
        assert (root / "e1" / f).is_file()
    with pytest.raises(FileExistsError):
        I.run_eval(cal, root / "e1", U_A)
    p = e1["provenance"]
    assert p["obs_sha256"] == I.sha256_file(OBS) and p["controls"] == list(cal.cs.names)
    assert p["git_head_file"]["head"] and p["setting"] == I.SETTING
    assert cal.model.adjoint_cfg.freeze_mixing and cal.model.adjoint_cfg.freeze_fct_limiter
    t = e1["terms"]
    assert abs(t["J_C_NH"] + t["J_C_SH"] + t["J_H_NH"] - e1["J_data"]) <= 1e-12 * e1["J_data"]
    assert t["J_C_NH_month"][0] == 0.0 and t["J_C_NH_month"][1] > 0.0
    assert t["Nact_C_NH"] <= t["N_C_NH"] and t["N_H_NH"] > 0
    tr = json.loads((root / "e1" / "trace.json").read_text())
    assert len(tr["trace"]) == NC + 1 and all(np.isfinite(tr["trace"]))
    assert e1["linear_check"]["mode"] == "full"
    print(f"[icecal] J_data {e1['J_data']!r} terms C_NH {t['J_C_NH']!r} C_SH {t['J_C_SH']!r} "
          f"H_NH {t['J_H_NH']!r}; g_u {e1['g_data']}; first eval wall {p['wall_s']:.0f} s "
          f"compile {p['compile_s']:.0f} s n_compiles {p['n_compiles']} VmHWM "
          f"{p['host_vmhwm_gb']:.1f} GB")


@heavy
def test_stride_two_equals_stride_one(evals):
    _, e1, e2, _ = evals
    assert e2["boundary_stride"] == 2 and e1["boundary_stride"] == 1
    g1, g2 = np.array(e1["g_data"]), np.array(e2["g_data"])
    dj = abs(e2["J_data"] - e1["J_data"]) / abs(e1["J_data"])
    dg = np.max(np.abs(g2 - g1)) / np.max(np.abs(g1))
    print(f"[icecal] stride 2 vs 1: rel dJ {dj:.3e}, rel max dg {dg:.3e}")
    assert dj <= 1e-12 and dg <= 1e-12


@heavy
def test_later_evaluations_compile_nothing(evals):
    _, e1, e2, e3 = evals
    n = [e["provenance"]["n_compiles"] for e in (e1, e2, e3)]
    print(f"[icecal] backend compiles per eval {n}")
    assert n[0] >= 2 and n[1] == 0 and n[2] == 0
    assert e3["J_data"] != e1["J_data"]


@heavy
def test_per_term_eval_sums_to_the_total(cal, evals):
    """``eval --terms``: one forward, one reverse walk with three seeds; each term's J is the
    total evaluation's term, the three gradients sum to the J_data gradient, one trace each;
    a subset that leaves out a non-zero term is flagged as not the full gradient."""
    from fesom_jax import seaice_cost
    root, e1, _, _ = evals
    e4 = I.run_eval(cal, root / "e4", U_A, stride=1, terms=list(seaice_cost.TERMS))
    g1 = np.array(e1["g_data"])
    gsum = np.sum([e4["per_term"][t]["g_data"] for t in seaice_cost.TERMS], axis=0)
    for t in seaice_cost.TERMS:
        assert e4["per_term"][t]["J"] == e1["terms"][t]
        np.testing.assert_allclose(np.array(e4["per_term"][t]["g_lnp"]) * cal.cs.sigma,
                                   e4["per_term"][t]["g_data"], rtol=1e-15, atol=0)
    dg = np.max(np.abs(gsum - g1)) / np.max(np.abs(g1))
    print(f"[icecal] per-term eval: rel max |sum of term gradients - total| {dg:.3e}; "
          f"per-term g {[e4['per_term'][t]['g_data'] for t in seaice_cost.TERMS]}")
    assert dg <= 1e-12
    assert e4["g_data_is_full"] and e4["J_data"] == e1["J_data"]
    tr = json.loads((root / "e4" / "trace.json").read_text())
    assert tr["terms"] == list(seaice_cost.TERMS)
    assert all(len(tr["traces"][t]) == NC + 1 for t in seaice_cost.TERMS)
    assert e1["terms"]["J_H_NH"] > 0.0
    e5 = I.run_eval(cal, root / "e5", U_A, stride=2, terms=["J_C_NH", "J_C_SH"])
    assert not e5["g_data_is_full"]
    np.testing.assert_allclose(e5["per_term"]["J_C_SH"]["g_data"],
                               e4["per_term"]["J_C_SH"]["g_data"], rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="at most once"):
        I.run_eval(cal, root / "e6", U_A, terms=["J_C_NH", "J_C_NH"])


@heavy
def test_eval_gradient_matches_central_fd(cal, evals):
    _, e1, _, _ = evals
    g = np.array(e1["g_data"])
    j0 = e1["J_data"]
    c0 = C.compile_clock()["n_backend"]
    # the round-off noise of J (same program, a negligible change of u)
    probes = []
    for name in NOISE_PROBES:
        u = U_A.copy()
        u[cal.cs.names.index(name)] += 1e-12
        probes.append(abs(float(cal.forward(u)[1]["J_data"]) - j0))
    _, t_rep, _ = cal.forward(U_A)
    assert float(t_rep["J_data"]) == j0, "the forward is not reproducible"
    noise = NOISE_SAFETY * max(probes)
    print(f"[icecal] J noise probes {probes} (J {j0!r}); noise used {noise:.3e}")
    res = {}
    for name in FD_CONTROLS + FD_REPORT:
        i = cal.cs.names.index(name)
        rows = []
        for h in (FD_REPORT_HS if name in FD_REPORT else FD_HS):
            Js = []
            for s in (1, -1):
                u = U_A.copy()
                u[i] += s * h
                Js.append(float(cal.forward(u)[1]["J_data"]))
            fd = (Js[0] - Js[1]) / (2 * h)
            rel_noise = 2 * noise / abs(Js[0] - Js[1]) if Js[0] != Js[1] else np.inf
            rows.append({"h": h, "fd": fd, "rel_err": abs(g[i] / fd - 1), "rel_noise": rel_noise})
        print(f"[icecal] {name}: adjoint {g[i]!r}; FD {rows}", flush=True)
        res[name] = (g[i], rows)
    assert C.compile_clock()["n_backend"] == c0, "the forwards recompiled"
    for name in FD_CONTROLS:
        gi, rows = res[name]
        ok = [r for r in rows if r["rel_noise"] <= FD_RTOL / 4]
        assert ok, f"{name}: the FD is not resolved at any step size {rows}"
        assert min(r["rel_err"] for r in ok) <= FD_RTOL, f"{name}: adjoint {gi} vs FD {rows}"


@heavy
def test_fields_equal_an_independent_step_by_step_mean_and_save_state(cal, evals, tmp_path):
    root, e1, _, _ = evals
    with np.load(root / "e1" / "fields.npz") as z:
        got = {k: np.asarray(z[k]) for k in z.files}
    assert sorted(got) == ["C_mean", "H_mean", "n_mon", "year"]
    assert int(got["year"]) == 2013
    assert got["n_mon"].tolist() == [0.0, K * NC] + [0.0] * 10
    # the independent forward: one jitted model step, the monthly mean taken in numpy
    m = cal.model
    p_u = {"u": jnp.asarray(U_A, jnp.float64)}

    def one(s, x, pu):
        return I.stepmod.step(s, m.mesh, m.op, None, m.to_params(pu), dt=m.dt,
                              is_first_step=False, step_forcing=x,
                              forcing_static=m.forcing.static, adjoint_cfg=m.adjoint_cfg,
                              **m.phys)
    step = jax.jit(one)
    s = jax.device_put(m.s0)
    start = datetime.datetime.fromisoformat(START)
    a, mi = [], []
    for n in range(K * NC):
        sf = m.stack_forcing([m.date(start, n)])
        x = jax.device_put(jax.tree.map(lambda v: v[0], sf))
        s = step(s, x, p_u)
        a.append(np.asarray(jax.device_get(s.a_ice)))
        mi.append(np.asarray(jax.device_get(s.m_ice)))
    a_ref, m_ref = np.mean(a, axis=0), np.mean(mi, axis=0)
    da = np.max(np.abs(got["C_mean"][1] - a_ref))
    dm = np.max(np.abs(got["H_mean"][1] - m_ref))
    big = {t: int(np.sum(np.abs(got["C_mean"][1] - a_ref) > t)) for t in (1e-9, 1e-6, 1e-4)}
    # what a one-step shift of the average (the pre-step states) would give
    a_pre = np.mean([np.asarray(m.s0.a_ice)] + a[:-1], axis=0)
    shift = np.max(np.abs(a_pre - a_ref))
    print(f"[icecal] fields vs independent mean: max|dC| {da:.3e} max|dH| {dm:.3e}; nodes with "
          f"|dC| above 1e-9/1e-6/1e-4: {big}; a one-step shift would differ by {shift:.3e}")
    assert np.all(got["C_mean"][[0] + list(range(2, 12))] == 0.0)
    assert np.all(got["H_mean"][[0] + list(range(2, 12))] == 0.0)
    assert da <= FIELD_TOL_C and dm <= FIELD_TOL_H
    assert shift > 5 * FIELD_TOL_C                # the bound can tell a one-step shift apart
    # fwd --save-state-at: the State after the first chunk, in the load_state format
    when = start + datetime.timedelta(seconds=K * m.dt)
    r = I.run_fwd(cal, tmp_path / "f", U_A, save_state_at=when.isoformat())
    assert abs(r["J_data"] - e1["J_data"]) <= 1e-12 * abs(e1["J_data"])
    path = Path(r["saved_state"])
    assert path.name == C.restart_tag(when) == "fesom.2013.032.07200"
    st, meta = C.load_state(path)
    C._check_restart_date(path, meta, when)
    assert int(meta["step"]) == int(m.restart_meta["step"]) + K
    fwd = I.adjoint._jitted_forward(cal.chunk_fn)
    first = jax.device_get(fwd(m.init_carry(), cal.xs_fn()(0), cal.params(U_A))[0])
    for f in ("a_ice", "m_ice", "t_skin", "T", "S", "eta_n", "u_ice"):
        np.testing.assert_array_equal(np.asarray(getattr(st, f)),
                                      np.asarray(getattr(first, f)), err_msg=f)
    with pytest.raises(ValueError, match="chunk boundary"):
        cal.forward(U_A, save_state_at=(start + datetime.timedelta(hours=1)).isoformat(),
                    save_dir=tmp_path)
