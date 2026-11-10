"""``scripts/adjoint/ice_cbar.py`` without CORE2: window and FD algebra, the verdict at every
window, the control mapping, the ice-settings parser, the summary on synthetic run
directories (missing runs, a NaN run), and the forward bookkeeping — base, per-step ice
saves, replay, resume of an interrupted leg, a replay base with a missing chunk — on a toy
model with the State's ice fields. ``scripts`` group (needs the repo ROOT on sys.path).
The CORE2 smoke is ``test_cbar_ice_core2.py`` (named so: a test_ice_* file belongs to the ice group, everything under scripts/tests/ to the scripts group).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import core2_adj as C  # noqa: E402
import ice_cbar as I  # noqa: E402
import tke_cbar as T  # noqa: E402

from fesom_jax.ice import IceConfig  # noqa: E402
from fesom_jax.params import Params  # noqa: E402


# ---------------------------------------------------------------------------
# algebra


def test_windows_and_drift_algebra():
    assert T.window_steps(I.WINDOW_DAYS) == [48, 96, 240, 480]
    S = {0: 0.0, 48: 4.8, 96: 19.2}
    assert I.j_drift(S, 48) == 4.8 / 48 and I.j_drift(S, 96) == 19.2 / 96


def test_another_config_brings_its_own_time_step(monkeypatch):
    """Task 26 (farc, dt 900): ``run_dt`` reads the config's dt, the windows are counted in its
    steps, and ``build`` hands ``config`` / ``restart`` to ``core2_adj.build_model``."""
    assert I.run_dt() == C.DT
    assert I.run_dt(ROOT / "configs" / "farc_full.yaml") == 900.0
    assert T.window_steps([2, 90], 900.0) == [192, 8640]
    assert T.window_steps([2]) == [96]
    seen = {}

    class Stop(Exception):
        pass

    def fake(*a, **kw):
        seen.update(args=a, kw=kw)
        raise Stop

    monkeypatch.setattr(C, "build_model", fake)
    with pytest.raises(Stop):
        I.build("2013-01-02", 192, 24, config="configs/farc_full.yaml", restart="/r")
    assert seen["args"][3] == "/r" and seen["kw"]["config"] == "configs/farc_full.yaml"


def test_central_fd_cancels_the_even_part():
    a, b, n = 0.7, 3.0, 48
    for h in (3e-3, 1e-2, 3e-2):
        Sp = {n: n * (1.0 + a * h + b * h * h)}
        Sm = {n: n * (1.0 - a * h + b * h * h)}
        assert I.central_fd(Sp, Sm, n, h) == pytest.approx(a, rel=1e-12)
    assert I.central_fd({48: 1.0}, {}, 48, 1e-2) is None


@pytest.mark.parametrize("days", [1.0, 2.0, 5.0, 10.0])
def test_verdict_rule_at_every_window(days):
    fd = {3e-3: 1.00, 1e-2: 1.01, 3e-2: 1.30}
    for ad, want in ((1.05, True), (1.25, True), (0.80, False), (1.40, False)):
        v = I.verdict(ad, fd)
        assert v["bar"] is want, (ad, v)
        ref = T.bar_verdict(ad, fd, days)
        for k in ("fd_ref", "rel_err", "fd_range", "fd_spread", "plateau"):
            assert v[k] == ref[k]
        # tke_cbar's own verdict has no bar below 5 days; ice_cbar's rule is the same at any
        assert ref["bar"] is (want if days >= 5.0 else None)


def test_verdict_drops_nonfinite_fds():
    v = I.verdict(1.0, {3e-3: float("nan"), 1e-2: 1.02, 3e-2: 1.04})
    assert v["n_fd"] == 2 and v["n_fd_nonfinite"] == 1 and v["bar"] is True
    v = I.verdict(1.0, {3e-3: float("nan")})
    assert v["bar"] is None and v["fd_ref"] is None
    assert I.verdict(None, {1e-2: 1.0})["bar"] is None
    # a blown-up reverse sweep fails the bar; it is not "no verdict"
    v = I.verdict(float("inf"), {1e-2: 1.0})
    assert v["bar"] is False and v["rel_err"] == math.inf and v["fd_ref"] == 1.0
    assert I.verdict(float("nan"), {1e-2: 1.0})["bar"] is False
    assert I.verdict(float("nan"), {1e-2: float("nan")})["bar"] is None


# ---------------------------------------------------------------------------
# controls and settings


def test_control_mapping_is_exact_at_zero():
    cfg = IceConfig(whichEVP=1)
    u0 = I.ice_params0(("pstar", "cd"))
    assert set(u0) == {f"ln_{n}" for n in C.LOG_CONTROLS} | {"ln_ice_pstar", "ln_ice_cd"}
    p = I.make_to_params(cfg, ("pstar", "cd"))(C._strong_f64(u0))
    assert float(p.ice_pstar) == cfg.pstar and float(p.ice_cd) == cfg.cd_oce_ice
    ref = C.default_to_params(C.default_params0(0))
    for f in ("k_ver", "a_ver", "k_gm", "redi_kmax", "tke_c_k", "tke_c_eps", "tke_cd",
              "tke_alpha"):
        assert float(getattr(p, f)) == float(getattr(ref, f))
    u = dict(u0, ln_ice_cd=np.asarray(0.1))
    p = I.make_to_params(lambda: cfg, ("cd",))(u)
    assert p.ice_pstar is None
    assert float(p.ice_cd) == pytest.approx(cfg.cd_oce_ice * math.exp(0.1), rel=1e-15)
    p = I.make_to_params(cfg, ())(C.default_params0(0))
    assert p.ice_pstar is None and p.ice_cd is None
    assert len(jax.tree.leaves(p)) == len(jax.tree.leaves(Params.defaults()))


def test_perturbed_and_parse_controls():
    u0 = I.ice_params0(("pstar",))
    u = I.perturbed(u0, "pstar", 1e-2, -1)
    assert float(u["ln_ice_pstar"]) == -1e-2 and float(u["ln_k_gm"]) == 0.0
    with pytest.raises(ValueError):
        I.perturbed(u0, "cd", 1e-2, 1)
    assert I.parse_controls("none") == () and I.parse_controls("pstar,cd") == ("pstar", "cd")
    with pytest.raises(ValueError):
        I.parse_controls("pstar,cdd")


def test_parse_ice_settings():
    assert I.parse_ice_settings("none") == {} and I.parse_ice_settings(None) == {}
    d = I.parse_ice_settings("evp_rheol_steps=4+pstar=2e4+whichEVP=1")
    assert d == {"evp_rheol_steps": 4, "pstar": 20000.0, "whichEVP": 1}
    assert isinstance(d["evp_rheol_steps"], int) and isinstance(d["pstar"], float)
    assert I.parse_ice_settings('{"evp_rheol_steps": 8}') == {"evp_rheol_steps": 8}
    assert I.parse_ice_settings({"pstar": 1.0}) == {"pstar": 1.0}
    with pytest.raises(ValueError, match="unknown IceConfig"):
        I.parse_ice_settings("adjoint_tn=7")
    with pytest.raises(ValueError, match="ice-mode"):
        I.parse_ice_settings("adjoint_mode=exact")
    with pytest.raises(ValueError, match="name=value"):
        I.parse_ice_settings("pstar")
    # merges into an IceConfig the way the driver does it
    cfg = IceConfig(whichEVP=1)._replace(**d)
    assert cfg.evp_rheol_steps == 4 and cfg.pstar == 20000.0


def test_output_paths_refuse_the_checkout(tmp_path):
    with pytest.raises(ValueError, match="inside the checkout"):
        I.out_path(ROOT / "runs_here")
    with pytest.raises(ValueError, match="inside the checkout"):
        I.out_path(ROOT)
    assert I.out_path(tmp_path / "x") == tmp_path / "x"


# ---------------------------------------------------------------------------
# toy model: the forward bookkeeping


@dataclasses.dataclass(frozen=True)
class _St:
    a_ice: jax.Array
    m_ice: jax.Array
    m_snow: jax.Array
    t_skin: jax.Array
    u_ice: jax.Array
    v_ice: jax.Array
    T: jax.Array


jax.tree_util.register_dataclass(_St, data_fields=[f.name for f in dataclasses.fields(_St)],
                                 meta_fields=[])
NOD = 7
W = np.linspace(1.0, 2.0, NOD)


class _P:
    def __init__(self, k):
        self.k = k


def _toy_step(s, x, p):
    """u is driven by the forcing scaled by the control; a, m follow u (so a replay of a/m
    matters), t_skin follows T."""
    u = 0.8 * s.u_ice + 0.1 * p.k * x
    v = 0.9 * s.v_ice - 0.05 * x
    a = jnp.clip(s.a_ice + 0.02 * u, 0.0, 1.0)
    m = s.m_ice * (1.0 + 0.01 * u)
    return dataclasses.replace(s, u_ice=u, v_ice=v, a_ice=a, m_ice=m, m_snow=s.m_snow + 0.001,
                               t_skin=0.5 * s.t_skin + 0.1 * s.T, T=s.T + 0.01 * u)


class _Toy:
    """Implements what ``ice_cbar.run_fwd`` uses of a ``core2_adj.Model``."""

    def __init__(self, K=3):
        self.chunk_steps = K
        self.accumulators = {"drift": lambda s: I.drift_value(s, jnp.asarray(W))}
        z = np.zeros(NOD)
        self.s0 = _St(a_ice=np.full(NOD, 0.5), m_ice=np.full(NOD, 1.0), m_snow=z + 0.1,
                      t_skin=z - 2.0, u_ice=z + 0.01, v_ice=z, T=np.linspace(-1.8, 2.0, NOD))
        self.prog = I.fwd_program(_toy_step, self.accumulators,
                                  lambda u: _P(jnp.exp(u["ln_k"])))

    def chunks_for(self, n):
        assert n % self.chunk_steps == 0
        return n // self.chunk_steps

    def init_carry(self, host=False):
        carry = (self.s0, {"drift": np.zeros((), np.float64)})
        return jax.device_get(carry) if host else jax.device_put(carry)

    def state_of(self, carry):
        return carry[0]

    def xs_fn(self):
        K = self.chunk_steps
        return lambda c: jnp.asarray([[math.sin(0.3 * (c * K + n) + 0.1 * i) for i in range(NOD)]
                                      for n in range(K)])


def _u(lnk=0.0):
    return {"ln_k": np.asarray(lnk, np.float64)}


def _run(m, d, lnk=0.0, n=12, **kw):
    return I.run_fwd(m, d, params=_u(lnk), n_steps=n, prog=m.prog, **kw)


@pytest.fixture()
def toy():
    return _Toy()


@pytest.fixture()
def base(toy, tmp_path):
    d = tmp_path / "base"
    r = _run(toy, d, save_boundaries=True, save_ice=True)
    assert r["status"] == "complete"
    return d


def test_base_saves_every_step_of_the_ice_state(toy, base):
    for c in range(4):
        z = I.load_icestate(base, c)
        assert set(z) == set(I.REPLAY_FIELDS)
        assert all(v.shape == (3, NOD) for v in z.values())
    assert sorted(int(p.stem.split("_")[1]) for p in base.glob("boundary_*.npz")) == list(range(5))
    # the last saved step equals the state in the last boundary
    st = T._host_load(base / "boundary_4.npz", toy.init_carry(host=True))[0]
    z = I.load_icestate(base, 3)
    for f in I.REPLAY_FIELDS:
        np.testing.assert_array_equal(z[f][-1], np.asarray(getattr(st, f)))


def test_saving_does_not_change_the_forward(toy, base, tmp_path):
    d = tmp_path / "plain"
    assert _run(toy, d)["status"] == "complete"
    chk = I.compare_runs(d, base)
    assert chk["bitwise"] and chk["n_common"] == 4


def test_replay_at_zero_reproduces_the_base(toy, base, tmp_path):
    d = tmp_path / "r0"
    assert _run(toy, d, replay_dir=base)["status"] == "complete"
    chk = I.compare_runs(d, base)
    assert chk["bitwise"] and chk["n_common"] == 4 and chk["max_abs_drift_sum"] == 0.0


def test_replay_holds_the_ice_state_and_full_does_not(toy, base, tmp_path):
    rep, full = tmp_path / "rep", tmp_path / "full"
    _run(toy, rep, lnk=0.1, replay_dir=base, save_ice=True)
    _run(toy, full, lnk=0.1, save_ice=True)
    for c in range(4):
        zb, zr, zf = (I.load_icestate(d, c) for d in (base, rep, full))
        for f in I.REPLAY_FIELDS:
            np.testing.assert_array_equal(zr[f], zb[f])
        assert not np.array_equal(zf["a_ice"], zb["a_ice"])
    rb, rr, rf = (I.fwd_records(d) for d in (base, rep, full))
    for n in (3, 6, 9, 12):
        assert rr[n]["fp"]["u_ice"] == rf[n]["fp"]["u_ice"] != rb[n]["fp"]["u_ice"]
        assert rr[n]["drift_sum"] != rf[n]["drift_sum"]     # a differs only in the full run


def test_replay_accumulates_after_the_overwrite(toy, base, tmp_path):
    """The replay's ``D_k`` is the drift of the state AFTER the base ice fields were written
    into it (a plain Python loop of the same toy step is the reference)."""
    d = tmp_path / "rep"
    _run(toy, d, lnk=0.2, replay_dir=base)
    s = jax.device_put(toy.s0)
    p = _P(jnp.exp(jnp.asarray(0.2)))
    xs = toy.xs_fn()
    acc, want = 0.0, {}
    for c in range(4):
        z = I.load_icestate(base, c)
        x = xs(c)
        for k in range(3):
            s = _toy_step(s, x[k], p)
            s = dataclasses.replace(s, **{f: jnp.asarray(z[f][k]) for f in I.REPLAY_FIELDS})
            acc = acc + float(I.drift_value(s, jnp.asarray(W)))
        want[3 * (c + 1)] = acc
    got = I.drift_sums(d)
    for n, v in want.items():
        assert got[n] == pytest.approx(v, rel=1e-13, abs=0.0), n


def test_replay_base_with_a_missing_chunk(toy, tmp_path):
    short = tmp_path / "short"
    _run(toy, short, n=6, save_ice=True)                    # chunks 0, 1 only
    with pytest.raises(FileNotFoundError, match=r"\[2, 3\]"):
        _run(toy, tmp_path / "r", replay_dir=short)
    assert not (tmp_path / "r" / "fwd.jsonl").exists()      # failed before computing


@pytest.mark.parametrize("replay", [False, True])
def test_resume_of_an_interrupted_leg(toy, base, tmp_path, replay):
    ref = tmp_path / "ref"
    _run(toy, ref, lnk=-0.05, replay_dir=base if replay else None)
    d = tmp_path / "legs"
    statuses = []
    for _ in range(6):
        r = _run(toy, d, lnk=-0.05, replay_dir=base if replay else None, budget_s=0.0)
        statuses.append(r["status"])
        if r["status"] == "complete":
            break
    assert statuses == ["incomplete"] * 3 + ["complete"]
    assert sorted(int(p.stem.split("_")[1]) for p in d.glob("carry_*.npz")) == [1, 2, 3]
    chk = I.compare_runs(d, ref)
    assert chk["bitwise"] and chk["n_common"] == 4


def _kept(d, pat):
    return sorted(int(p.stem.split("_")[1]) for p in d.glob(pat))


def test_save_every_keeps_every_nth_boundary_and_the_last(toy, base, tmp_path):
    d = tmp_path / "sparse"
    assert _run(toy, d, save_boundaries=True, save_every=3)["status"] == "complete"
    assert _kept(d, "boundary_*.npz") == [0, 3, 4]
    chk = I.compare_runs(d, base)
    assert chk["bitwise"] and chk["n_common"] == 4
    for c in (3, 4):
        a = T._host_load(d / f"boundary_{c}.npz", toy.init_carry(host=True))
        b = T._host_load(base / f"boundary_{c}.npz", toy.init_carry(host=True))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(x, y)


def test_save_every_resume_keeps_a_carry_only_between_boundaries(toy, base, tmp_path):
    d = tmp_path / "legs"
    statuses = []
    for _ in range(6):
        r = _run(toy, d, save_boundaries=True, save_every=3, budget_s=0.0)
        statuses.append(r["status"])
        if r["status"] == "complete":
            break
    assert statuses == ["incomplete"] * 3 + ["complete"]
    assert _kept(d, "carry_*.npz") == [1, 2]
    assert _kept(d, "boundary_*.npz") == [0, 3, 4]
    chk = I.compare_runs(d, base)
    assert chk["bitwise"] and chk["n_common"] == 4


def test_grad_links_every_stride_th_base_boundary(monkeypatch, tmp_path):
    seen = {}

    def link(base_dir, run_dir, n_chunks, stride=1, factor=1):
        seen.update(n_chunks=n_chunks, stride=stride, factor=factor)
        raise RuntimeError("linked")
    monkeypatch.setattr(I.T, "link_base", link)
    with pytest.raises(RuntimeError, match="linked"):
        I.run_grad(_Toy(), tmp_path / "g", tmp_path / "base", n_steps=12, budget_s=1.0, stride=2)
    assert seen == {"n_chunks": 4, "stride": 2, "factor": 1}


def test_grad_chunk_spans_several_base_chunks(tmp_path):
    """A base at chunk 1 saved every 24 steps (farc ``base*_k1``) serves a gradient at chunk 24:
    the gradient's boundary c is the base's 24·c."""
    base, run = tmp_path / "base", tmp_path / "run"
    base.mkdir()
    assert I.base_factor(base, 24) == 1                      # no rows: same chunk
    (base / "results.jsonl").write_text(json.dumps({"chunk_steps": 1}) + "\n")
    assert I.base_factor(base, 24) == 24 and I.base_factor(base, 1) == 1
    (base / "results.jsonl").write_text(json.dumps({"chunk_steps": 24}) + "\n")
    with pytest.raises(ValueError, match="multiple"):
        I.base_factor(base, 36)
    for c in (0, 24, 48, 72, 96):
        (base / f"boundary_{c}.npz").write_bytes(b"")
    T.link_base(base, run, 4, stride=2, factor=24)
    got = {p.name: Path(p.readlink()).name for p in run.glob("boundary_*.npz")}
    assert got == {"boundary_0.npz": "boundary_0.npz", "boundary_2.npz": "boundary_48.npz",
                   "boundary_4.npz": "boundary_96.npz"}


def test_cli_passes_the_partition_to_build(monkeypatch, tmp_path):
    seen = {}

    def build(*a, **k):
        seen.update(k)
        raise RuntimeError("built")
    monkeypatch.setattr(I, "build", build)
    with pytest.raises(RuntimeError, match="built"):
        I.main(["grad", "--run-dir", str(tmp_path / "g"), "--setting", "none", "--dist", "4",
                "--dist-dir", "/pool/x", "--transport", "padded", "--window-days", "1"])
    assert (seen["dist"], seen["dist_dir"], seen["transport"]) == (4, "/pool/x", "padded")
    with pytest.raises(SystemExit):
        I.main(["grad", "--run-dir", str(tmp_path / "g"), "--setting", "none", "--dist", "4",
                "--dump-cot", "1", "--dump-primal", "--window-days", "1"])


def test_nan_run_propagates_to_the_fd(toy, base, tmp_path):
    d = tmp_path / "nan"
    _run(toy, d, lnk=float("nan"))
    S = I.drift_sums(d)
    assert math.isnan(S[12])
    # and an ice-free state gives 0, not NaN
    s0 = toy.s0
    z = dataclasses.replace(s0, a_ice=np.zeros(NOD))
    assert float(I.drift_value(z, jnp.asarray(W))) == 0.0


# ---------------------------------------------------------------------------
# summary on synthetic run directories


def _write_run(d: Path, sums: dict):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "fwd.jsonl", "w") as f:
        for n, s in sorted(sums.items()):
            f.write(json.dumps({"steps": n, "drift_sum": s, "fp": {}}) + "\n")


def test_summary_with_missing_and_nan_runs(tmp_path):
    fd, gr = tmp_path / "fd", tmp_path / "grad"
    windows = [48, 96]
    a = {"replay": {"pstar": 0.02, "cd": -0.5}, "full": {"pstar": 0.03, "cd": -0.6}}
    _write_run(fd / "base", {48: 48 * 0.1, 96: 96 * 0.1})
    for mode in I.FD_MODES:
        for c in ("pstar", "cd"):
            for h in I.HS:
                for s in (1, -1):
                    if mode == "full" and c == "cd" and h == 3e-2:
                        continue                                # missing pair
                    val = 0.1 + a[mode][c] * s * h
                    if mode == "replay" and c == "cd" and h == 1e-2 and s == 1:
                        val = float("nan")                      # a NaN run
                    _write_run(I.fd_run_dir(fd, mode, c, h, s), {48: 48 * val, 96: 96 * val})
    g = I.grad_dir(gr, "tagA", 1.0)
    g.mkdir(parents=True)
    g.joinpath("done.json").write_text(json.dumps(
        {"status": "ok", "grads": {"pstar": 0.0201, "cd": -0.9},
         "trace": [1.0, 1.01, 1.02, 1.03]}))
    # a blown-up gradient as core2_adj._jsonable writes it: non-finite numbers as strings
    gb = I.grad_dir(gr, "tagB", 1.0)
    gb.mkdir(parents=True)
    gb.joinpath("done.json").write_text(json.dumps(
        {"status": "ok", "grads": {"pstar": "nan", "cd": "-inf"},
         "trace": [1.0, 3.0e200, "inf", "nan"]}))
    tab = I.summarize(fd, gr, windows=windows, hs=I.HS, chunk_steps=24)
    rows_b = [r for r in tab["rows"] if r["tag"] == "tagB" and r["window_days"] == 1.0]
    assert len(rows_b) == 2
    for r in rows_b:
        assert math.isnan(r["adjoint"]) or r["adjoint"] == -math.inf
        assert r["verdict"]["bar"] is False and r["amp_clean"] is False
    I.markdown(tab)
    rows = {(r["control"], r["window_days"]): r for r in tab["rows"] if r["tag"] == "tagA"}
    assert set(rows) == {(c, d) for c in ("pstar", "cd") for d in (1.0, 2.0)}
    r = rows[("pstar", 1.0)]
    assert r["verdict"]["bar"] is True and r["adjoint"] == 0.0201
    assert r["verdict"]["fd_ref"] == pytest.approx(0.02, rel=1e-9)
    assert r["gap_full_vs_replay"] == pytest.approx(0.5, rel=1e-6)
    assert r["amplification"]["clean"] is not None
    r = rows[("cd", 1.0)]
    assert r["verdict"]["n_fd"] == 2 and r["verdict"]["n_fd_nonfinite"] == 1
    assert r["verdict"]["bar"] is False                         # -0.9 vs -0.5
    assert len(r["fd_full"]) == 2                               # the missing pair is absent
    r = rows[("pstar", 2.0)]
    assert r["grad_status"] == "missing" and r["adjoint"] is None and r["verdict"]["bar"] is None
    md = I.markdown(tab)
    assert md.count("\n") == 2 + len(tab["rows"])
    # no grad root at all: FD-only rows
    tab = I.summarize(fd, None, windows=windows, hs=I.HS, chunk_steps=24)
    assert {r["tag"] for r in tab["rows"]} == {None}


def test_noise_floor(tmp_path):
    _write_run(tmp_path / "b", {48: 4.8, 96: 9.6})
    _write_run(tmp_path / "r", {48: 4.8 + 4.8e-9, 96: 9.6})
    out = I.noise_floor(tmp_path / "b", tmp_path / "r", [48, 96, 240])
    w = {x["steps"]: x for x in out["windows"]}
    assert w[48]["abs_diff"] == pytest.approx(1e-10, rel=1e-6) and not w[48]["bitwise"]
    assert w[96]["bitwise"] and w[96]["abs_diff"] == 0.0
    assert w[240]["missing"]


# ---------------------------------------------------------------------------
# drift box (--box)


class _Mesh:
    """Five wet surface nodes at 80N, 65N, 50S, 65S, 80S, one level, unit areas."""
    geo_coord_nod2D = np.radians(np.array([[0.0, 80.0], [10.0, 65.0], [20.0, -50.0],
                                           [30.0, -65.0], [40.0, -80.0]]))
    node_layer_mask = np.ones((5, 2), bool)
    areasvol = np.ones((5, 2))
    Z = np.array([-5.0])
    nl = 2


def _rows(d: Path, *rows):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "results.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.mark.parametrize("box,nodes", [("arctic", [0]), ("antarctic", [3, 4])])
def test_drift_boxes_select_their_hemisphere(box, nodes):
    assert np.flatnonzero(I.arctic_weights(_Mesh(), box)).tolist() == nodes


def test_build_refuses_an_unknown_box():
    with pytest.raises(ValueError, match="unknown drift box"):
        I.build("2013-01-01", 48, 24, box="tropics")


def test_runs_refuse_another_drift_box(tmp_path):
    old = tmp_path / "old"                       # rows from before --box existed: arctic
    _rows(old, {"status": "complete"})
    ant = tmp_path / "ant"
    _rows(ant, {"status": "complete", "drift_box": "antarctic"})
    assert I.run_box(old) == "arctic" and I.run_box(ant) == "antarctic"
    assert I.run_box(tmp_path / "none") is None
    I.check_box(old, "arctic", "base")
    I.check_box(tmp_path / "none", "antarctic", "base")       # no rows yet: any box
    with pytest.raises(ValueError, match="drift box 'arctic'"):
        I.check_box(old, "antarctic", "base")
    mixed = tmp_path / "mixed"
    _rows(mixed, {"drift_box": "arctic"}, {"drift_box": "antarctic"})
    with pytest.raises(ValueError, match="mixes"):
        I.run_box(mixed)


def test_grad_and_fwd_refuse_a_base_of_another_box(monkeypatch, tmp_path):
    base = tmp_path / "base"
    _rows(base, {"status": "complete", "drift_box": "arctic"})
    monkeypatch.setattr(I.T, "link_base", lambda *a, **k: pytest.fail("linked a foreign base"))
    toy = _Toy()
    toy.drift_box = "antarctic"
    with pytest.raises(ValueError, match="base .* drift box 'arctic'"):
        I.run_grad(toy, tmp_path / "g", base, n_steps=12, budget_s=1.0)
    with pytest.raises(ValueError, match="run .* drift box 'arctic'"):     # resume into it
        I._fwd_one(toy, base, control="none", h=0.0, sign=1, n=6, replay=None,
                   save_boundaries=False, save_ice=False, budget=60.0, job="t")


def test_summary_and_noise_refuse_mixed_boxes(tmp_path):
    fd = tmp_path / "fd"
    for c in ("pstar", "cd"):
        for s in (1, -1):
            d = I.fd_run_dir(fd, "full", c, 1e-2, s)
            _write_run(d, {48: 4.8 + s * 1e-3})
            _rows(d, {"status": "complete", "drift_box": "antarctic"})
    tab = I.summarize(fd, None, windows=[48], hs=[1e-2], chunk_steps=24)
    assert tab["drift_box"] == "antarctic"
    g = I.grad_dir(tmp_path / "grad", "t", 1.0)
    g.mkdir(parents=True)
    g.joinpath("done.json").write_text(json.dumps({"status": "ok", "grads": {},
                                                   "drift_box": "arctic"}))
    with pytest.raises(ValueError, match="mix drift boxes"):
        I.summarize(fd, tmp_path / "grad", windows=[48], hs=[1e-2], chunk_steps=24)
    b, r = tmp_path / "b", tmp_path / "r"
    _write_run(b, {48: 4.8})
    _rows(b, {"drift_box": "arctic"})
    _write_run(r, {48: 4.8})
    _rows(r, {"drift_box": "antarctic"})
    with pytest.raises(ValueError, match="differ in box"):
        I.noise_floor(b, r, [48])


def test_cli_passes_the_box_to_build(monkeypatch, tmp_path):
    seen = {}

    def build(*a, **k):
        seen.update(k)
        raise RuntimeError("built")
    monkeypatch.setattr(I, "build", build)
    with pytest.raises(RuntimeError, match="built"):
        I.main(["fwd", "--run-dir", str(tmp_path / "r"), "--box", "antarctic", "--days", "1"])
    assert seen["box"] == "antarctic"
    with pytest.raises(SystemExit):
        I.main(["fwd", "--run-dir", str(tmp_path / "r"), "--box", "tropics"])


# ---------------------------------------------------------------------------
# dumps (grad --dump-cot / --dump-primal)


def test_parse_boundaries_and_fields():
    assert I.parse_boundaries("29-31,33,30") == (29, 30, 31, 33)
    assert I.parse_boundaries(None) == () and I.parse_boundaries("") == ()
    for bad in ("5-3", "-1"):
        with pytest.raises(ValueError):
            I.parse_boundaries(bad)
    assert I.parse_fields(None) == I.DUMP_FIELDS and I.parse_fields("all") is None
    assert I.parse_fields("S, T") == ("S", "T")


def test_dump_cot_hook_keeps_the_listed_boundaries_once(toy, tmp_path):
    cot = toy.init_carry(host=True)
    hook = I.dump_cot_hook(tmp_path / "dump", (1, 3), ("a_ice", "T"))
    for c in range(5):
        hook(c, cot)
    assert _kept(tmp_path / "dump", "cot_*.npz") == [1, 3]
    with np.load(tmp_path / "dump" / "cot_3.npz") as z:
        assert sorted(z.files) == ["0.T", "0.a_ice"]
        np.testing.assert_array_equal(z["0.T"], cot[0].T)
    other = (dataclasses.replace(cot[0], T=cot[0].T + 1.0), cot[1])
    hook(3, other)                                           # an existing dump is kept
    with np.load(tmp_path / "dump" / "cot_3.npz") as z:
        np.testing.assert_array_equal(z["0.T"], cot[0].T)
    I.dump_cot_hook(tmp_path / "all", (0,), None)(0, cot)   # None = every floating leaf
    with np.load(tmp_path / "all" / "cot_0.npz") as z:
        assert len(z.files) == 8 and "1.drift" in z.files
    with pytest.raises(ValueError, match="no floating carry field"):
        I.dump_cot_hook(tmp_path / "bad", (0,), ("S",))(0, cot)


def _chunk(carry, xs, params):
    s, acc = carry
    p = _P(jnp.exp(params["ln_k"]))
    for k in range(xs.shape[0]):
        s = _toy_step(s, xs[k], p)
        acc = {"drift": acc["drift"] + I.drift_value(s, jnp.asarray(W))}
    return s, acc


class _ChunkToy(_Toy):
    """``_Toy`` plus what :func:`ice_cbar.dump_primal` uses of a ``core2_adj.Model``."""

    def __init__(self):
        super().__init__()
        self.chunk_fn = _chunk
        self.params0 = _u(0.3)
        self.shard = None          # a dense model


def test_dump_primal_reruns_from_the_nearest_base_boundary(tmp_path):
    m = _ChunkToy()
    xs = m.xs_fn()
    ref = [m.init_carry(host=True)]
    for c in range(4):
        ref.append(jax.device_get(_chunk(ref[-1], xs(c), m.params0)))
    base = tmp_path / "base"
    base.mkdir()
    C.DiskCheckpoint._save(base / "boundary_0.npz", ref[0])
    # boundary 2 is NOT the state the chunks reach: a dump after it must restart from it
    moved = (dataclasses.replace(ref[2][0], u_ice=ref[2][0].u_ice + 0.5), ref[2][1])
    C.DiskCheckpoint._save(base / "boundary_2.npz", moved)
    after = jax.device_get(_chunk(moved, xs(2), m.params0))
    assert I.dump_primal(m, base, tmp_path / "dump", (1, 2, 3), ("u_ice", "drift")) == [1, 2, 3]
    want = {1: ref[1], 2: moved, 3: after}
    for c, carry in want.items():
        with np.load(tmp_path / "dump" / f"primal_{c}.npz") as z:
            assert sorted(z.files) == ["0.u_ice", "1.drift"]
            np.testing.assert_allclose(z["0.u_ice"], carry[0].u_ice, rtol=1e-13, atol=1e-15)
            np.testing.assert_allclose(z["1.drift"], carry[1]["drift"], rtol=1e-13)
    assert I.dump_primal(m, base, tmp_path / "dump", (1, 2, 3), ("u_ice",)) == []
    with pytest.raises(FileNotFoundError):
        I.dump_primal(m, tmp_path / "empty", tmp_path / "dump", (4,), ("u_ice",))


def test_cli_passes_the_dumps(monkeypatch, tmp_path):
    seen = {}

    class _M:
        drift_box = "arctic"

        def chunks_for(self, n):
            return n // 24

    monkeypatch.setattr(I, "build", lambda *a, **k: _M())
    monkeypatch.setattr(I, "dump_primal", lambda m, b, d, cs, f: seen.update(primal=(cs, f)))

    def run_grad(model, run_dir, base_dir, **k):
        seen["on_cot"] = k["on_cot"]
        raise RuntimeError("ran")
    monkeypatch.setattr(I, "run_grad", run_grad)
    common = ["grad", "--setting", "none", "--base-dir", str(tmp_path / "base"),
              "--run-dir", str(tmp_path / "g"), "--window-days", "2"]
    with pytest.raises(RuntimeError, match="ran"):
        I.main(common + ["--dump-cot", "2-3", "--dump-primal", "--dump-fields", "S,T"])
    assert seen["primal"] == ((2, 3), ("S", "T")) and callable(seen["on_cot"])
    seen.clear()
    with pytest.raises(RuntimeError, match="ran"):
        I.main(common)
    assert seen == {"on_cot": None}
    for bad in (["--dump-cot", "5"], ["--dump-primal"]):     # beyond 4 chunks; no boundaries
        with pytest.raises(SystemExit):
            I.main(common + bad)


def test_provenance_records_the_fct_switch(monkeypatch):
    class _M:
        phys = {"ice_cfg": IceConfig()}

    assert I._ice_prov(_M())["fct_stopgrad_ae"] is False
    monkeypatch.setattr(I.tracer_adv, "FCT_STOP_GRADIENT_AE", True)
    assert I._ice_prov(_M())["fct_stopgrad_ae"] is True


# ---------------------------------------------------------------------------
# terminal ice costs: volume and area


class _Mesh7:
    """The toy's seven nodes: three north of 70N, two in 0-70N, two south; unit areas."""
    geo_coord_nod2D = np.radians(np.array([[0.0, 85.0], [10.0, 75.0], [20.0, 71.0],
                                           [30.0, 60.0], [40.0, 10.0], [50.0, -10.0],
                                           [60.0, -70.0]]))
    node_layer_mask = np.ones((NOD, 2), bool)
    areasvol = np.ones((NOD, 2))
    Z = np.array([-5.0])
    nl = 2


def test_cost_boxes_select_their_nodes():
    assert np.flatnonzero(I.arctic_weights(_Mesh7(), "nh")).tolist() == [0, 1, 2, 3, 4]
    assert np.flatnonzero(I.arctic_weights(_Mesh7(), "arctic")).tolist() == [0, 1, 2]


def test_ice_cost_value_is_the_weighted_sum():
    st = _Toy().s0
    w = np.linspace(1e9, 3e9, NOD)
    assert float(I.ice_cost_value(st, w, "vol")) == pytest.approx(np.sum(w * st.m_ice) / 1e12,
                                                                  rel=1e-15)
    assert float(I.ice_cost_value(st, w, "area")) == pytest.approx(np.sum(w * st.a_ice) / 1e12,
                                                                   rel=1e-15)
    with pytest.raises(ValueError, match="unknown ice cost"):
        I.ice_cost_loss(_Toy(), "thickness", "nh")


def test_ice_cost_seed_is_the_weights_on_its_field():
    """d J / d final carry: ``w / divisor`` on the cost's field, zero on every other leaf."""
    toy = _Toy()
    toy.mesh = _Mesh7()
    w = I.arctic_weights(toy.mesh, "nh")
    carry = jax.tree.map(jnp.asarray, toy.init_carry(host=True))
    for cost, (field, div) in I.ICE_COSTS.items():
        st, acc = jax.grad(I.ice_cost_loss(toy, cost, "nh"))(carry)
        for f in dataclasses.fields(st):
            want = w / div if f.name == field else np.zeros(NOD)
            np.testing.assert_allclose(np.asarray(getattr(st, f.name)), want, rtol=1e-15, atol=0)
        assert float(acc["drift"]) == 0.0


def test_fwd_rows_carry_the_ice_values_of_the_boundary_state(tmp_path):
    toy = _Toy()
    toy.mesh = _Mesh7()
    d = tmp_path / "b"
    assert _run(toy, d, save_boundaries=True)["status"] == "complete"
    like = toy.init_carry(host=True)
    for c in (1, 4):
        st = toy.state_of(T._host_load(d / f"boundary_{c}.npz", like))
        for b in I.COST_BOXES:
            w = I.arctic_weights(toy.mesh, b)
            for cost, (f, div) in I.ICE_COSTS.items():
                got = I.ice_values(d, f"{cost}_{b}")[3 * c]
                assert got == pytest.approx(np.sum(w * np.asarray(getattr(st, f))) / div,
                                            rel=1e-14), (cost, b, c)


def test_a_model_without_a_mesh_writes_no_ice_values(toy, base):
    assert I.ice_values(base, "vol_nh") == {}


def test_grad_dir_names_the_cost():
    assert I.grad_dir(Path("/g"), "t", 90) == Path("/g/t/drift_w90")
    assert I.grad_dir(Path("/g"), "t", 256, "vol", "nh") == Path("/g/t/vol_nh_w256")
    assert I.grad_dir(Path("/g"), "t", 90.5, "area", "arctic") == Path("/g/t/area_arctic_w90.5")


def test_folded_node_weights_on_one_device_are_the_weights():
    class _M:
        sharded = None
    w = np.arange(4.0)
    assert np.array_equal(I.folded_node_weights(_M(), w), w)


def test_grad_refuses_an_unknown_cost_box(tmp_path):
    with pytest.raises(ValueError, match="unknown cost box"):
        I.run_grad(_Toy(), tmp_path / "g", tmp_path / "base", n_steps=12, budget_s=1.0,
                   cost="vol", cost_box="tropics")


def test_cli_passes_the_ice_cost_to_the_gradient(monkeypatch, tmp_path):
    seen = {}

    class _M:
        def chunks_for(self, n):
            return 1

    def run_grad(model, run_dir, base_dir, **kw):
        seen.update(run_dir=run_dir, **kw)
        raise RuntimeError("grad")
    monkeypatch.setattr(I, "build", lambda *a, **k: _M())
    monkeypatch.setattr(I, "run_grad", run_grad)
    common = ["grad", "--grad-root", str(tmp_path), "--tag", "t", "--setting", "none",
              "--base-dir", str(tmp_path / "b"), "--window-days", "256"]
    with pytest.raises(RuntimeError, match="grad"):
        I.main(common + ["--ice-cost", "vol"])
    assert (seen["cost"], seen["cost_box"]) == ("vol", "nh")
    assert seen["run_dir"] == tmp_path / "t" / "vol_nh_w256"
    with pytest.raises(RuntimeError, match="grad"):
        I.main(common)
    assert seen["cost"] == "drift" and seen["run_dir"] == tmp_path / "t" / "drift_w256"
    with pytest.raises(SystemExit):
        I.main(common + ["--ice-cost", "vol", "--cost-box", "tropics"])
