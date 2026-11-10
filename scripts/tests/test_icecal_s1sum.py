"""``scripts/icecal/s1sum.py`` (the S1 summary of the sea-ice calibration) on a synthetic window:
six nodes, an observation file, base repeats and FD runs of three controls whose monthly means
are a known function of ln p plus GPU-like noise, and an ``eval.json`` with chosen adjoints.
``scripts`` group (numpy only, seconds).

What is pinned:

* the verdict rules (pass at the reference h, pass inside the FD range, fail, unresolved, no
  adjoint, an unresolved FD never passes) on hand-made numbers;
* end to end: the adjoint equal to the true derivative passes, 1.5 × it fails, a control with no
  response is unresolved; the noise floor is the std of the base repeats;
* every run's J_data is recomputed from its fields (a wrong J in ``fwd.json`` is refused) and a
  run of another window is refused;
* the Gauss–Newton Hessian equals an independent loop over the observations (columns at the
  largest h, the noise term s²/(2h²) from the base repeats taken off the diagonal), and H_u =
  Σ H_lnp Σ;
* the amplification screen finds a one-boundary spike at its date and nothing in a smooth trace;
* the parity verdict's 1σ response equals a per-node loop.
"""

from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "icecal"))
sys.path.insert(0, str(ROOT))
import s1sum as S  # noqa: E402
from fesom_jax import seaice_cost  # noqa: E402

LAT = np.array([70.0, 75.0, 80.0, 85.0, -65.0, -70.0])
LON = np.array([0.0, 10.0, 20.0, 30.0, 0.0, 10.0])
AREA = np.array([1.0e9, 2.0e9, 1.5e9, 0.5e9, 3.0e9, 2.5e9])
NOD = LAT.size
CONTROLS = ["ice.albsn", "ice.con", "ice.h0"]
SIGMA = [0.05 / 0.81, math.log(2.0), math.log(2.0)]
NOISE = 1e-5                      # std of the per-node "GPU" noise of a monthly mean
N_BASE = 6                        # base repeats (the noise floor has N_BASE - 1 dof)
HS = {"albsn": [1e-2], "con": [1e-2], "h0": [1e-2, 3e-2]}
PROV = {"start": "2013-01-01T00:00:00", "n_steps": 1440, "chunk_steps": 24, "window_days": 30.0,
        "obs_sha256": "abc", "controls": CONTROLS, "setting": "s", "sigma": SIGMA}

rng0 = np.random.default_rng(7)
C0 = 0.2 + 0.6 * rng0.random((12, NOD))
H0 = 0.5 + 2.0 * rng0.random((12, NOD))
# sensitivities per ln p: albsn and h0 respond, con does not (unresolved)
A_C = {"albsn": 0.05 * rng0.standard_normal((12, NOD)), "con": np.zeros((12, NOD)),
       "h0": 0.08 * rng0.standard_normal((12, NOD))}
A_H = {"albsn": 0.3 * rng0.standard_normal((12, NOD)), "con": np.zeros((12, NOD)),
       "h0": 0.2 * rng0.standard_normal((12, NOD))}
Q = 0.5                           # curvature: one-sided FDs differ, the central is exact for F


def obs_dict():
    rng = np.random.default_rng(3)
    nC = rng.integers(0, 4, (12, NOD)).astype(np.float64)
    nC[:, 0] = 2.0
    nH = np.where((LAT > 0)[None, :] & (np.arange(12) % 3 != 1)[:, None], 1.0, 0.0)
    nH[:, 4] = 3.0                # southern thickness: must be ignored by the cost (CCI = north)
    C_obs = np.where(nC > 0, np.clip(C0 + 0.1 * rng.standard_normal((12, NOD)), 0, 1), np.nan)
    H_obs = np.where(nH > 0, H0 + 0.3 * rng.standard_normal((12, NOD)), np.nan)
    return {"C_obs": C_obs, "sC": np.where(nC > 0, 0.15, np.nan), "nC": nC,
            "H_obs": H_obs, "sH": np.where(nH > 0, 0.4, np.nan), "nH": nH,
            "nh": LAT > 0, "year": np.int64(2013)}


def model(delta: dict, noise_rng):
    C = C0.copy()
    H = H0.copy()
    for c, d in delta.items():
        C += A_C[c] * d + Q * np.abs(A_C[c]) * d * d
        H += A_H[c] * d + Q * np.abs(A_H[c]) * d * d
    if noise_rng is not None:
        C += NOISE * noise_rng.standard_normal(C.shape)
        H += NOISE * noise_rng.standard_normal(H.shape)
    return C, H


def write_run(d, C, H, ob, info, prov=None):
    d.mkdir(parents=True)
    n_mon = np.full(12, 48.0)
    with open(d / "fields.npz", "xb") as f:
        np.savez(f, C_mean=C, H_mean=H, n_mon=n_mon, year=np.int64(2013))
    t = seaice_cost.cost_terms_np(C, H, ob)
    p = dict(PROV if prov is None else prov)
    p["fd"] = info
    with open(d / "fwd.json", "x") as f:
        json.dump({"action": "fwd", "J_data": t["J_data"],
                   "terms": {k: float(v) for k, v in t.items() if np.ndim(v) == 0},
                   "provenance": {**p, "obs": str(d.parents[3] / "obs.npz")}}, f)   # tmp/
    return t


def true_grad(ob):
    """dJ/dln p at δ = 0 of the NOISELESS model (central differences of F are exact there)."""
    Cb, Hb = model({}, None)
    out = {}
    for c in ("albsn", "con", "h0"):
        g = {}
        for t in S.TERMS:
            # J is quadratic in F: dJ = sum W (F - o) dF on the cost mask
            terms_p = seaice_cost.cost_terms_np(Cb + 1e-7 * A_C[c], Hb + 1e-7 * A_H[c], ob)
            terms_m = seaice_cost.cost_terms_np(Cb - 1e-7 * A_C[c], Hb - 1e-7 * A_H[c], ob)
            g[t] = (terms_p[t] - terms_m[t]) / 2e-7
        out[c] = g
    return out


@pytest.fixture()
def world(tmp_path):
    win = tmp_path / "s1" / "jan30"
    win.mkdir(parents=True)
    od = obs_dict()
    with open(win.parents[1] / "obs.npz", "xb") as f:
        np.savez(f, **od)
    ob = seaice_cost.cost_obs(od)
    rng = np.random.default_rng(11)
    u0 = [0.0] * 3
    for i in range(N_BASE):
        C, H = model({}, rng)
        write_run(win / "fd" / f"base_r{i}", C, H, ob,
                  {"control": None, "h_lnp": 0.0, "sign": 0, "u0": u0})
    for c, hs in HS.items():
        for h in hs:
            for s in (1, -1):
                C, H = model({c: s * h}, rng)
                write_run(win / "fd" / f"{c}_h{h:g}_{'p' if s > 0 else 'm'}", C, H, ob,
                          {"control": "ice." + c, "h_lnp": h, "sign": s, "lnp_delta": s * h,
                           "u0": u0})
    g = true_grad(ob)
    adj = [g["albsn"]["J_data"], g["con"]["J_data"], 1.5 * g["h0"]["J_data"]]
    ev = win / "eval_full"
    ev.mkdir()
    tr = list(np.linspace(1.0, 50.0, 61))
    tr[33] = 3.0 * max(tr[32], tr[34])         # index 33 = boundary 60 - 33 = 27: a spike
    with open(ev / "eval.json", "x") as f:
        json.dump({"J_data": 0.0, "terms": {t: 0.0 for t in S.TERMS}, "g_lnp_data": adj,
                   "linear_check": None, "provenance": {**PROV, "u": u0}}, f)
    with open(ev / "trace.json", "x") as f:
        json.dump({"trace": tr, "field_trace": None}, f)
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    np.save(mesh / "geo_coord_nod2D.npy", np.radians(np.stack([LON, LAT], axis=1)))
    np.save(mesh / "ulevels_nod2D.npy", np.ones(NOD, dtype=np.int32))
    np.save(mesh / "nlevels_nod2D.npy", np.full(NOD, 10, dtype=np.int32))
    np.save(mesh / "area.npy", np.stack([AREA, 0.9 * AREA], axis=1))
    return {"win": win, "ob": ob, "od": od, "g": g, "adj": adj, "mesh": mesh, "tmp": tmp_path}


# ---------------------------------------------------------------------------
# verdict rules


def test_verdict_rules():
    v = S.verdict(1.05, {0.01: 1.0}, 1e-4)
    assert v["verdict"] == "pass" and v["pass_by"] == "ratio"
    assert S.verdict(1.2, {0.01: 1.0}, 1e-4)["verdict"] == "fail"
    v = S.verdict(1.2, {0.01: 1.0, 0.03: 1.3}, 1e-4)
    assert v["verdict"] == "pass" and v["pass_by"] == "range"
    # |FD| = 1 < 3 * noise / h = 3 at h 0.01: unresolved, even though adj == FD
    assert S.verdict(1.0, {0.01: 1.0}, 0.01)["verdict"] == "unresolved"
    # resolved at 0.03 only (3 * 0.01 / 0.03 = 1 <= 1.1): the ratio is judged there (|1/1.1 - 1|
    # = 9 %), the noise-dominated 0.01 match does not count
    v = S.verdict(1.0, {0.01: 1.0, 0.03: 1.1}, 0.01)
    assert v["resolved"] == {0.01: False, 0.03: True} and v["verdict"] == "pass"
    assert v["ratio_h"] == 0.03
    assert S.verdict(1.0, {0.01: 1.0, 0.03: 1.2}, 0.01)["verdict"] == "fail"
    v = S.verdict(1.0, {0.01: 1.0, 0.03: 1.5}, 0.01)
    assert v["verdict"] == "fail"          # the range needs two RESOLVED FDs
    assert S.verdict(None, {0.01: 1.0}, 1e-4)["verdict"] == "no adjoint"
    v = S.verdict(1.0, {0.01: 1.02}, None)
    assert v["verdict"] == "pass" and not v["noise_known"]


# ---------------------------------------------------------------------------
# end to end


def test_bar_noise_and_window_end_to_end(world):
    res = S.summarize(world["win"])
    base_J = [res["base"][i]["J"]["J_data"] for i in range(N_BASE)]
    assert res["noise_J"]["J_data"] == pytest.approx(np.std(base_J, ddof=1), rel=1e-12)
    assert res["noise_J"]["J_data"] > 0
    bar = res["bar"]
    assert bar["albsn"]["J_data"]["verdict"] == "pass", bar["albsn"]["J_data"]
    assert bar["con"]["J_data"]["verdict"] == "unresolved", bar["con"]["J_data"]
    assert bar["h0"]["J_data"]["verdict"] == "fail", bar["h0"]["J_data"]
    # the per-term FDs are reported without an adjoint
    assert bar["albsn"]["J_C_NH"]["verdict"] == "no adjoint"
    fd = bar["albsn"]["J_data"]["fd"][0.01]
    assert fd == pytest.approx(world["g"]["albsn"]["J_data"], rel=0.05)
    one = bar["h0"]["J_data"]["one_sided"][0.03]
    assert one["plus"] != pytest.approx(one["minus"], rel=1e-3)    # the curvature shows
    assert res["J_recompute_max_rel"] < 1e-12
    # main() writes a new directory and refuses an existing one
    out = world["tmp"] / "sum"
    assert S.main(["--window", str(world["win"]), "--out", str(out)]) == 0
    assert json.loads((out / "s1sum.json").read_text())["bar"]["albsn"]["J_data"]["verdict"] \
        == "pass"
    assert S.main(["--window", str(world["win"]), "--out", str(out)]) == 3


def test_wrong_J_and_foreign_window_are_refused(world):
    fj = world["win"] / "fd" / "base_r1" / "fwd.json"
    d = json.loads(fj.read_text())
    d2 = json.loads(fj.read_text())
    d["terms"]["J_data"] *= 1.0 + 1e-6
    bad = world["tmp"] / "bad"
    (bad / "fd" / "base_r1").mkdir(parents=True)
    (bad / "fd" / "base_r1" / "fwd.json").write_text(json.dumps(d))
    (bad / "fd" / "base_r1" / "fields.npz").write_bytes(
        (world["win"] / "fd" / "base_r1" / "fields.npz").read_bytes())
    with pytest.raises(ValueError, match="not the cost's"):
        S.summarize(bad, obs=str(world["win"].parents[1] / "obs.npz"))
    d2["provenance"]["n_steps"] = 1488
    (world["win"] / "fd" / "zz_other").mkdir()
    (world["win"] / "fd" / "zz_other" / "fwd.json").write_text(json.dumps(d2))
    with pytest.raises(ValueError, match="not the same window"):
        S.summarize(world["win"])


def test_gauss_newton_equals_an_independent_loop(world):
    res = S.summarize(world["win"])
    hs = res["hessian"]
    assert hs["controls"] == ["albsn", "con", "h0"]
    assert hs["h"] == [0.01, 0.01, 0.03]              # the largest h with both signs
    od = world["od"]

    def fields(tag):
        with np.load(world["win"] / "fd" / tag / "fields.npz") as z:
            return z["C_mean"], z["H_mean"]

    def wsum(fa, fb):
        """sum over the cost's observations of W * fa * fb (explicit loops)."""
        (Ca, Ha), (Cb, Hb) = fa, fb
        s = 0.0
        for m in range(12):
            for n in range(NOD):
                if od["nC"][m, n] > 0:
                    s += od["nC"][m, n] / 16.0 / od["sC"][m, n] ** 2 * Ca[m, n] * Cb[m, n]
                if od["nH"][m, n] > 0 and LAT[n] > 0:
                    s += od["nH"][m, n] / 16.0 / od["sH"][m, n] ** 2 * Ha[m, n] * Hb[m, n]
        return s

    base = [fields(f"base_r{i}") for i in range(N_BASE)]
    Cbar = np.mean([b[0] for b in base], axis=0)
    Hbar = np.mean([b[1] for b in base], axis=0)
    s2 = sum(wsum((b[0] - Cbar, b[1] - Hbar), (b[0] - Cbar, b[1] - Hbar)) for b in base) \
        / (N_BASE - 1)
    cols = {}
    for c, h in (("albsn", 0.01), ("con", 0.01), ("h0", 0.03)):
        Cp, Hp = fields(f"{c}_h{h:g}_p")
        Cm, Hm = fields(f"{c}_h{h:g}_m")
        cols[c] = ((Cp - Cm) / (2 * h), (Hp - Hm) / (2 * h), h)
    names = ["albsn", "con", "h0"]
    H = np.zeros((3, 3))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            H[i, j] = wsum(cols[a][:2], cols[b][:2])
        H[i, i] -= s2 / (2 * cols[a][2] ** 2)
    np.testing.assert_allclose(np.array(hs["H_lnp"]), H, rtol=1e-11, atol=1e-11 * abs(H).max())
    assert hs["noise_s2"] == pytest.approx(s2, rel=1e-11)
    Sg = np.diag(SIGMA)
    np.testing.assert_allclose(np.array(hs["H_u_corrected"]), Sg @ H @ Sg, rtol=1e-11,
                               atol=1e-11 * abs(H).max())
    lam = [s["eigenvalue"] for s in hs["spectrum_corrected"]]
    np.testing.assert_allclose(sorted(lam, reverse=True),
                               np.sort(np.linalg.eigvalsh(Sg @ H @ Sg))[::-1], rtol=1e-10)
    # con has no response: its raw diagonal is the noise term (~90 observations, so ±~20 %),
    # the corrected one near zero
    assert hs["H_lnp_raw"][1][1] == pytest.approx(hs["diag_noise"][1], rel=0.6)
    assert abs(hs["H_lnp"][1][1]) < 0.6 * hs["diag_noise"][1]


def test_screen_finds_the_spike(world):
    res = S.summarize(world["win"])
    sc = res["screen"][str(world["win"] / "eval_full")]["J_data"]
    assert [s["boundary"] for s in sc["spikes"]] == [27]
    start = datetime.datetime(2013, 1, 1)
    assert sc["spikes"][0]["date"] == (start + datetime.timedelta(seconds=27 * 24 * 1800)
                                       ).isoformat()
    assert sc["finite"] and sc["growth"] == pytest.approx(3.0 * np.linspace(1.0, 50.0, 61)[34])
    smooth = S.screen_trace(np.linspace(1.0, 50.0, 61), start, 24 * 1800.0)
    assert smooth["spikes"] == []


def test_parity_response_equals_a_node_loop(world):
    par = {"months": [1, 2, 3], "monthly": {
        hem: [{"month": m, "area_diff": 0.001 * m, "volume_diff": -0.002, "rms_C": 0.01,
               "rms_H": 0.02} for m in (1, 2, 3)] for hem in ("NH", "SH")}}
    pfile = world["tmp"] / "parity.json"
    pfile.write_text(json.dumps(par))
    res = S.summarize(world["win"], parity=str(pfile), mesh_dir=world["mesh"])
    r = res["parity"]["controls"]["h0"]["NH"]
    with np.load(world["win"] / "fd" / "h0_h0.03_p" / "fields.npz") as z:
        Cp = z["C_mean"]
    with np.load(world["win"] / "fd" / "h0_h0.03_m" / "fields.npz") as z:
        Cm = z["C_mean"]
    best = 0.0
    for m in range(3):
        ap = sum(min(max(Cp[m, n], 0), 1) * AREA[n] for n in range(NOD) if LAT[n] > 0)
        am = sum(min(max(Cm[m, n], 0), 1) * AREA[n] for n in range(NOD) if LAT[n] > 0)
        best = max(best, abs(ap - am) / 1e12 * SIGMA[2] / (2 * 0.03))
    assert r["response_1sigma"]["area_diff"] == pytest.approx(best, rel=1e-12)
    assert r["ratio"]["area_diff"] == pytest.approx(0.003 / best, rel=1e-12)
    assert "con" in res["parity"]["controls"]
    assert res["parity"]["negligible_for_every_control"] is False   # con: zero response
