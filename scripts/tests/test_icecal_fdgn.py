"""``scripts/icecal/fdgn.py`` (FD Gauss–Newton steps of the sea-ice calibration) on a synthetic
window: six nodes, an observation file, and monthly means that are a known LINEAR function of u
(so the Gauss–Newton model is exact) plus optional GPU-like noise. ``scripts`` group (numpy only,
seconds).

What is pinned:

* noiseless, prior on: the full step lands on the minimum of the true J (its central differences
  in every active direction vanish), the model J equals the true J there, the inactive controls do
  not move, and ``accept`` picks that trial with ρ = 1;
* the noise term is s²/(2 d_i²) with s² from an independent loop over the repeats; a column of
  pure noise (a control with no response) carries exactly that term, and the corrected data
  block has the trace of the noiseless one;
* a trust radius below |δ_GN| gives |δ| = Δ with (H + μI)δ = −g, μ > 0; a trial outside the
  physical domain gets no file;
* the adjoint report: the exact ln p gradient gives angle 0 and scale 1;
* refusals: a pair that moves two controls, base runs at two points, a wrong J in ``fwd.json``,
  an existing output directory.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "icecal"))
sys.path.insert(0, str(ROOT))
import fdgn as G  # noqa: E402
from fesom_jax import seaice_cost  # noqa: E402
from fesom_jax.controls import ice_thermo_control_set  # noqa: E402

CS = ice_thermo_control_set()
NAMES = [n.split(".", 1)[1] for n in CS.names]
NOD = 6
LAT = np.array([70.0, 75.0, 80.0, 85.0, -65.0, -70.0])
PROV = {"start": "2013-01-01T00:00:00", "n_steps": 17520, "chunk_steps": 24,
        "obs_sha256": "abc", "controls": list(CS.names), "setting": "s"}
ACTIVE = ["con", "h0", "albsnm", "consn"]           # consn: no response (a pure-noise column)

rng0 = np.random.default_rng(5)
C0 = 0.2 + 0.6 * rng0.random((12, NOD))
H0 = 0.5 + 2.0 * rng0.random((12, NOD))
A_C = {c: 0.04 * rng0.standard_normal((12, NOD)) for c in ("con", "h0", "albsnm")}
A_H = {c: 0.3 * rng0.standard_normal((12, NOD)) for c in ("con", "h0", "albsnm")}
A_C["consn"] = np.zeros((12, NOD))
A_H["consn"] = np.zeros((12, NOD))


def obs_dict():
    rng = np.random.default_rng(3)
    nC = rng.integers(1, 4, (12, NOD)).astype(np.float64)
    nH = np.where((LAT > 0)[None, :], 1.0, 0.0) * np.ones((12, 1))
    C_obs = np.clip(C0 + 0.1 + 0.05 * rng.standard_normal((12, NOD)), 0, 1)
    H_obs = np.where(nH > 0, H0 - 0.4 + 0.2 * rng.standard_normal((12, NOD)), np.nan)
    return {"C_obs": C_obs, "sC": np.full((12, NOD), 0.15), "nC": nC,
            "H_obs": H_obs, "sH": np.where(nH > 0, 0.4, np.nan), "nH": nH,
            "nh": LAT > 0, "year": np.int64(2013)}


def fields(u, noise_rng=None, noise=0.0):
    C, H = C0.copy(), H0.copy()
    for c in A_C:
        C = C + A_C[c] * u[NAMES.index(c)]
        H = H + A_H[c] * u[NAMES.index(c)]
    if noise_rng is not None:
        C = C + noise * noise_rng.standard_normal(C.shape)
        H = H + noise * noise_rng.standard_normal(H.shape)
    return C, H


def true_J(u, ob, prior=True):
    C, H = fields(u)
    return seaice_cost.cost_terms_np(C, H, ob)["J_data"] + (0.5 * float(u @ u) if prior else 0.0)


def write_run(d, u, ob, obs_path, *, fd=None, noise_rng=None, noise=0.0, J_shift=0.0):
    d = Path(d)
    d.mkdir(parents=True)
    C, H = fields(u, noise_rng, noise)
    with open(d / "fields.npz", "xb") as f:
        np.savez(f, C_mean=C, H_mean=H, n_mon=np.full(12, 48.0), year=np.int64(2013))
    J = seaice_cost.cost_terms_np(C, H, ob)["J_data"] + J_shift
    with open(d / "fwd.json", "x") as f:
        json.dump({"action": "fwd", "J_data": J, "terms": {"J_data": J},
                   "provenance": {**PROV, "u": list(map(float, u)), "obs": str(obs_path),
                                  "fd": fd}}, f)
    return d


def write_pairs(root, u_c, ob, obs_path, d_u, rng=None, noise=0.0):
    for c in ACTIVE:
        i = NAMES.index(c)
        h = d_u * CS.sigma[i]
        for s in (1, -1):
            u = np.array(u_c, np.float64)
            u[i] += s * h / CS.sigma[i]
            write_run(root / f"{c}_h{h:g}_{'p' if s > 0 else 'm'}", u, ob, obs_path,
                      fd={"control": "ice." + c, "h_lnp": h, "sign": s, "lnp_delta": s * h,
                          "u0": list(map(float, u_c))}, noise_rng=rng, noise=noise)


@pytest.fixture()
def world(tmp_path):
    od = obs_dict()
    obs_path = tmp_path / "obs.npz"
    with open(obs_path, "xb") as f:
        np.savez(f, **od)
    ob = seaice_cost.cost_obs(od)
    u0 = np.zeros(10)
    u0[NAMES.index("albi")] = 0.3                    # an inactive control away from 0
    bases = [write_run(tmp_path / "base" / f"r{k}", u0, ob, obs_path) for k in range(2)]
    write_pairs(tmp_path / "jac", u0, ob, obs_path, d_u=0.3)
    return {"tmp": tmp_path, "ob": ob, "obs": obs_path, "u0": u0, "bases": bases}


def test_noiseless_step_lands_on_the_minimum_and_accept_picks_it(world):
    w = world
    res = G.step(w["tmp"] / "step", base=w["bases"], jac=w["tmp"] / "jac", noise=w["bases"],
                 controls=ACTIVE, prior=True, radii=(math.inf, 0.05))
    assert res["jacobian_is_current"] and res["noise_s2"] == 0.0
    f1 = next(t for t in res["trials"] if t["tag"] == "rinf")
    u1 = np.asarray(f1["u"])
    # inactive controls unchanged
    act = [NAMES.index(c) for c in ACTIVE]
    assert np.array_equal(np.delete(u1, act), np.delete(w["u0"], act))
    # the true J is stationary at u1 in every active direction (central differences, exact for
    # a quadratic J)
    for i in act:
        e = np.zeros(10)
        e[i] = 1e-3
        gi = (true_J(u1 + e, w["ob"]) - true_J(u1 - e, w["ob"])) / 2e-3
        assert abs(gi) < 1e-6 * max(1.0, abs(res["g"][act.index(i)])), (NAMES[i], gi)
    assert f1["J_model"] == pytest.approx(true_J(u1, w["ob"]), rel=1e-10)
    assert res["J_k"] == pytest.approx(true_J(w["u0"], w["ob"]), rel=1e-12)
    # the FD of J along the pairs equals the GN gradient (linear F)
    assert res["g_fd_of_J_vs_gn"]["angle_deg"] < 1e-4
    assert res["g_fd_of_J_vs_gn"]["scale"] == pytest.approx(1.0, rel=1e-9)
    with open(f1["file"]) as f:
        assert json.load(f)["u"] == f1["u"]
    # run the two trials and accept
    runs = {t["tag"]: write_run(w["tmp"] / "trials" / t["tag"], np.asarray(t["u"]), w["ob"],
                                w["obs"]) for t in res["trials"]}
    acc = G.accept(w["tmp"] / "step", runs)
    assert acc["best"] == "rinf" and acc["accepted"]
    row = next(r for r in acc["trials"] if r["tag"] == "rinf")
    assert row["rho"] == pytest.approx(1.0, rel=1e-8)
    with pytest.raises(FileExistsError):
        G.accept(w["tmp"] / "step", runs)


def test_noise_term_and_pure_noise_column(world, tmp_path):
    w = world
    rng = np.random.default_rng(17)
    noise = 0.05                  # noise term N ~ 5 against data curvatures ~ 2
    reps = [write_run(tmp_path / "rep" / f"r{k}", w["u0"], w["ob"], w["obs"], noise_rng=rng,
                      noise=noise) for k in range(4)]
    write_pairs(tmp_path / "jacn", w["u0"], w["ob"], w["obs"], d_u=0.3, rng=rng, noise=noise)
    res = G.step(tmp_path / "stepn", base=reps, jac=tmp_path / "jacn", noise=reps,
                 controls=ACTIVE, prior=True, radii=(math.inf,))
    # independent loop for s2 over the cost's observation mask
    space = G.Fields(w["obs"], [G.read_run(d) for d in reps]).space
    Fs = []
    for d in reps:
        with np.load(Path(d) / "fields.npz") as z:
            Fs.append(np.concatenate([z["C_mean"][space.mC], z["H_mean"][space.mH]]))
    Fbar = sum(Fs) / len(Fs)
    s2 = sum(float(np.sum(space.W * (F - Fbar) ** 2)) for F in Fs) / (len(Fs) - 1)
    assert res["noise_s2"] == pytest.approx(s2, rel=1e-10)
    assert res["noise_diag"] == pytest.approx([s2 / (2 * 0.3 ** 2)] * len(ACTIVE), rel=1e-10)
    # the pure-noise column: its raw diagonal IS the noise term (120 observations: a chi-square
    # spread of ~13 %), so the correction leaves nothing of it
    j = ACTIVE.index("consn")
    assert abs(res["H_raw"][j][j] / res["noise_diag"][j] - 1.0) < 0.5
    # the corrected data block matches the noiseless one: its trace (= the sum of the eigenvalues
    # after the correction, before the clip) against the noiseless AᵀWA, whereas the raw trace is
    # off by the whole noise term
    clean = G.step(tmp_path / "stepc", base=w["bases"], jac=w["tmp"] / "jac", noise=w["bases"],
                   controls=ACTIVE, prior=True, radii=(math.inf,))
    tr_true = np.trace(clean["H_raw"])
    tr_cor = sum(res["data_eigenvalues_after_noise"])
    assert tr_cor == pytest.approx(np.trace(res["H_raw"]) - sum(res["noise_diag"]), rel=1e-10)
    assert abs(tr_cor - tr_true) < 0.25 * sum(res["noise_diag"])
    assert abs(np.trace(res["H_raw"]) - tr_true) > 0.75 * sum(res["noise_diag"])


def test_radius_cap_and_out_of_domain(world, tmp_path):
    w = world
    res = G.step(tmp_path / "s_cap", base=w["bases"], jac=w["tmp"] / "jac", noise=w["bases"],
                 controls=ACTIVE, prior=True, radii=(0.02, 0.01))
    assert res["delta_gn_norm"] > 0.02
    g, H = np.asarray(res["g"]), np.asarray(res["H"])
    for t in res["trials"]:
        dl = np.asarray(t["delta"])
        assert np.linalg.norm(dl) == pytest.approx(t["radius"], rel=1e-10)
        assert t["mu"] > 0
        np.testing.assert_allclose((H + t["mu"] * np.eye(len(g))) @ dl, -g, rtol=1e-8,
                                   atol=1e-10 * np.linalg.norm(g))
        assert t["J_model"] < res["J_k"]
    # prior off, one weak direction: an enormous step leaves the domain (con beyond a factor 10)
    tmp = tmp_path / "weak"
    write_run(tmp / "base", w["u0"], w["ob"], w["obs"])
    i = NAMES.index("con")
    for s in (1, -1):
        u = w["u0"].copy()
        u[i] += s * 1e-3
        # a response 1e-3 of the real one: the step for the misfit is huge
        d = tmp / "jac" / f"con_{'p' if s > 0 else 'm'}"
        d.mkdir(parents=True)
        C = C0 + A_C["con"] * 1e-3 * u[i]
        H = H0 + A_H["con"] * 1e-3 * u[i]
        with open(d / "fields.npz", "xb") as f:
            np.savez(f, C_mean=C, H_mean=H, n_mon=np.full(12, 48.0), year=np.int64(2013))
        J = seaice_cost.cost_terms_np(C, H, w["ob"])["J_data"]
        with open(d / "fwd.json", "x") as f:
            json.dump({"J_data": J, "terms": {"J_data": J},
                       "provenance": {**PROV, "u": list(map(float, u)), "obs": str(w["obs"]),
                                      "fd": {"control": "ice.con", "h_lnp": 1e-3 * CS.sigma[i],
                                             "sign": s, "lnp_delta": s * 1e-3 * CS.sigma[i],
                                             "u0": list(map(float, w["u0"]))}}}, f)
    res = G.step(tmp / "step", base=[tmp / "base"], jac=tmp / "jac", noise=[tmp / "base"],
                 controls=["con"], prior=False, radii=(math.inf,))
    t = res["trials"][0]
    assert abs(t["u"][i]) > 10.0
    assert list(t["out_of_domain"]) == ["ice.con"] and t["file"] is None
    assert not (tmp / "step" / "trial_rinf.u.json").exists()


def test_adjoint_report(world, tmp_path):
    w = world
    # exact ln p gradient of J_data at u0 (linear F): central differences of the true J in u / σ
    g_lnp = np.zeros(10)
    for i in range(10):
        e = np.zeros(10)
        e[i] = 1e-3
        g_lnp[i] = (true_J(w["u0"] + e, w["ob"], prior=False)
                    - true_J(w["u0"] - e, w["ob"], prior=False)) / 2e-3 / CS.sigma[i]
    ev = tmp_path / "eval.json"
    ev.write_text(json.dumps({"g_lnp_data": g_lnp.tolist(),
                              "provenance": {**PROV, "u": w["u0"].tolist()}}))
    res = G.step(tmp_path / "s_adj", base=w["bases"], jac=w["tmp"] / "jac", noise=w["bases"],
                 controls=ACTIVE, prior=True, radii=(math.inf,), eval_path=ev)
    assert res["g_adjoint_vs_gn"]["angle_deg"] < 1e-3
    assert res["g_adjoint_vs_gn"]["scale"] == pytest.approx(1.0, rel=1e-6)


def test_refusals(world, tmp_path):
    w = world
    kw = dict(controls=ACTIVE, prior=True, radii=(math.inf,))
    with pytest.raises(FileExistsError):
        G.step(w["tmp"] / "jac", base=w["bases"], jac=w["tmp"] / "jac", noise=w["bases"], **kw)
    other = write_run(tmp_path / "other", w["u0"] + 0.1, w["ob"], w["obs"])
    with pytest.raises(ValueError, match="--base"):
        G.step(tmp_path / "s1", base=w["bases"] + [other], jac=w["tmp"] / "jac",
               noise=w["bases"], **kw)
    bad = write_run(tmp_path / "badJ", w["u0"], w["ob"], w["obs"], J_shift=1.0)
    with pytest.raises(ValueError, match="fields.npz"):
        G.step(tmp_path / "s2", base=[bad], jac=w["tmp"] / "jac", noise=w["bases"], **kw)
    # a pair run that moves a second control
    jac2 = tmp_path / "jac2"
    write_pairs(jac2, w["u0"], w["ob"], w["obs"], d_u=0.3)
    p = next(jac2.glob("con_*_p"))
    fj = json.loads((p / "fwd.json").read_text())
    fj["provenance"]["u"][NAMES.index("h0_s")] = 0.5
    (p / "fwd.json").write_text(json.dumps(fj))
    with pytest.raises(ValueError, match="moves controls"):
        G.step(tmp_path / "s3", base=w["bases"], jac=jac2, noise=w["bases"], **kw)
