"""CORE2 smoke of ``scripts/adjoint/ice_cbar.py`` (2013-01-01 restart, ``configs/core2_full.yaml``,
4-step window, 2-step chunks).

⚠️ HEAVY (CPU compute node, >= 200 GB): CORE2 chunk forward and VJP programs. ``scripts`` group.
Two independent halves, each one model (run them as two jobs with ``-k``):

* ``prescribed`` — base forward (boundaries + per-step ice state), the same forward without
  saves (bitwise: saving changes nothing), a replay at h = 0 (bitwise the base), a replay and a
  full forward along ln P* (the replay holds the four ice fields at the base values, the full
  run does not), and the drift gradient under "prescribed": EXACTLY zero w.r.t. ln P* and ln C_d
  (the ice velocity carries no tangent); plus the ``tkegrad`` model's carry matches
  ``tke_cbar``'s (so the TKE CPU base boundaries load);
* ``exact4`` — ``--ice-mode exact`` with the settings pass-through ``evp_rheol_steps=4``: the
  drift gradient w.r.t. ln P* and ln C_d agrees with a central FD of the same model, also when
  the reverse sweep is cut into legs by a zero budget and resumed; and the terminal ice volume
  (>= 70N) gradient w.r.t. ln C_d against a central FD of the forward rows' ``ice`` values
  (``-k ice_cost``; measured 2.8e-4 at h 1e-4, job 28195464). The other three are PRINTED, not
  gated: over 4 steps their FD does not converge in h, the response is a few 1e-7 of J and
  crosses the model's kinks (a_ice clipped to [0, 1]): volume ln P* FD 1.78e-6 (h 1e-3) /
  1.59e-6 (h 1e-4) vs AD 1.77e-6; area ln P* 1.8e-5 / -2.8e-6 vs 1.5e-5; area ln C_d 5.40e-5 /
  5.18e-5 vs 5.55e-5. The seed itself (d J / d final carry = weights on m_ice or a_ice) is gated
  exactly in test_cbar_ice.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import core2_adj as C  # noqa: E402
import ice_cbar as I  # noqa: E402
import tke_cbar as T  # noqa: E402

from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402

RESTART = C.STATES_DIR / "fesom.2013.001.00000"
TKE_BASE = Path("/work/ab0995/a270088/fesom_jax_dev/runs/tke/a8/cbar/fd/base")
pytestmark = pytest.mark.skipif(
    not ((ROOT / "data" / "mesh_core2").is_dir() and RESTART.is_dir()),
    reason="CORE2 mesh / 2013 States missing")
N, K = 4, 2


@pytest.fixture(scope="module")
def presc():
    return I.build("2013-01-01", N, K, ice_mode="prescribed", controls=("pstar", "cd"))


@pytest.fixture(scope="module")
def pbase(presc, tmp_path_factory):
    d = tmp_path_factory.mktemp("ice") / "base"
    r = I.run_fwd(presc, d, params=presc.params0, n_steps=N, save_boundaries=True,
                  save_ice=True)
    assert r["status"] == "complete"
    return d


def test_prescribed_fwd_save_replay(presc, pbase, tmp_path):
    for c in range(N // K):
        z = I.load_icestate(pbase, c)
        assert all(v.shape == (K, presc.mesh.nod2D) for v in z.values())
    # the per-step saving does not change the forward
    plain = tmp_path / "plain"
    I.run_fwd(presc, plain, params=presc.params0, n_steps=N)
    chk = I.compare_runs(plain, pbase)
    assert chk["bitwise"] and chk["n_common"] == N // K, chk
    # a replay at h = 0 reproduces the base
    r0 = tmp_path / "r0"
    I.run_fwd(presc, r0, params=presc.params0, n_steps=N, replay_dir=pbase)
    chk = I.compare_runs(r0, pbase)
    assert chk["bitwise"], chk
    # along ln P*: the replay holds the ice state, the full run does not
    u = I.perturbed(presc.params0, "pstar", 1e-1, 1)
    rep, full = tmp_path / "rep", tmp_path / "full"
    I.run_fwd(presc, rep, params=u, n_steps=N, replay_dir=pbase, save_ice=True)
    I.run_fwd(presc, full, params=u, n_steps=N, save_ice=True)
    zb, zr, zf = (I.load_icestate(d, N // K - 1) for d in (pbase, rep, full))
    for f in I.REPLAY_FIELDS:
        np.testing.assert_array_equal(zr[f], zb[f])
    assert not np.array_equal(zf["m_ice"], zb["m_ice"]), "P* did not move the ice state"
    Sb, Sr, Sf = (I.drift_sums(d) for d in (pbase, rep, full))
    print(f"[smoke] drift sums base {Sb[N]!r} replay {Sr[N]!r} full {Sf[N]!r}")
    assert Sr[N] != Sb[N] and Sf[N] != Sb[N]


def test_prescribed_drift_gradient_is_exactly_zero(presc, pbase, tmp_path):
    r = I.run_grad(presc, tmp_path / "g", pbase, n_steps=N, budget_s=1e9)
    assert r["status"] == "ok", r.get("status")
    print(f"[smoke] prescribed grads {r['grads']} log {r['grads_log_controls']}")
    assert r["grads"] == {"pstar": 0.0, "cd": 0.0}
    assert all(np.isfinite(v) for v in r["grads_log_controls"].values())


@pytest.mark.skipif(not (TKE_BASE / "boundary_0.npz").exists(), reason="TKE CPU base missing")
def test_tkegrad_model_matches_tke_cbar_carry():
    class _A:
        start, chunk_steps, ice_mode = "2013-01-01", 24, "prescribed"
    ref = T.build(_A, AdjointConfig.exact(), 48)
    mine = I.build("2013-01-01", 48, 24, ice_mode="prescribed", controls=(),
                   accs=T.accumulators)
    like = mine.init_carry(host=True)
    assert jax.tree.structure(like) == jax.tree.structure(ref.init_carry(host=True))
    assert sorted(mine.params0) == sorted(ref.params0)
    assert mine.phys["ice_cfg"] == ref.phys["ice_cfg"]
    with np.load(TKE_BASE / "boundary_0.npz") as z:
        assert len(z.files) == len(jax.tree.leaves(like))


@pytest.fixture(scope="module")
def exact4():
    return I.build("2013-01-01", N, K, ice_mode="exact", ice_settings="evp_rheol_steps=4",
                   controls=("pstar", "cd"))


def test_exact4_drift_gradient_matches_fd(exact4, tmp_path):
    m = exact4
    assert m.phys["ice_cfg"].evp_rheol_steps == 4 and m.phys["ice_cfg"].adjoint_mode == "exact"
    base = tmp_path / "base"
    assert I.run_fwd(m, base, params=m.params0, n_steps=N, save_boundaries=True,
                     save_ice=True)["status"] == "complete"
    r = I.run_grad(m, tmp_path / "g", base, n_steps=N, budget_s=1e9)
    assert r["status"] == "ok"
    # the same sweep cut into legs by a zero budget, then resumed
    legs = tmp_path / "legs"
    r1 = I.run_grad(m, legs, base, n_steps=N, budget_s=0.0)
    assert r1["status"] == "incomplete"
    r2 = I.run_grad(m, legs, base, n_steps=N, budget_s=1e9)
    assert r2["status"] == "ok" and r2["grads"] == r["grads"], (r2["grads"], r["grads"])
    S0 = I.drift_sums(base)
    for c in ("pstar", "cd"):
        ad = r["grads"][c]
        fds = {}
        for h in (1e-3, 1e-4):
            J = {}
            for s in (1, -1):
                d = tmp_path / f"{c}_{h:g}_{s}"
                I.run_fwd(m, d, params=I.perturbed(m.params0, c, h, s), n_steps=N)
                J[s] = I.j_drift(I.drift_sums(d), N)
            fds[h] = (J[1] - J[-1]) / (2 * h)
        rel = min(abs(v - ad) / abs(ad) for v in fds.values())
        print(f"[smoke] exact4 {c}: J {I.j_drift(S0, N):.15e} ad {ad:.15e} fd {fds} rel {rel:.3e}")
        assert np.isfinite(ad) and ad != 0.0
        assert rel < 1e-3, (c, ad, fds)


def test_exact4_ice_cost_gradients_match_fd(exact4, tmp_path):
    m = exact4
    base = tmp_path / "base"
    assert I.run_fwd(m, base, params=m.params0, n_steps=N,
                     save_boundaries=True)["status"] == "complete"
    runs = {}
    for c in ("pstar", "cd"):
        for h in (1e-3, 1e-4):
            for s in (1, -1):
                d = tmp_path / f"{c}_{h:g}_{s}"
                assert I.run_fwd(m, d, params=I.perturbed(m.params0, c, h, s),
                                 n_steps=N)["status"] == "complete"
                runs[c, h, s] = d
    bad = []
    for cost, box in (("vol", "arctic"), ("area", "nh")):
        g = tmp_path / f"g_{cost}"
        r = I.run_grad(m, g, base, n_steps=N, budget_s=1e9, cost=cost, cost_box=box)
        assert r["status"] == "ok" and (r["cost"], r["cost_box"]) == (cost, box)
        key = f"{cost}_{box}"
        J0 = I.ice_values(base, key)[N]
        assert json.loads((g / "forward.json").read_text())["loss"] == pytest.approx(J0,
                                                                                    rel=1e-12)
        for c in ("pstar", "cd"):
            ad = r["grads"][c]
            fds = {h: (I.ice_values(runs[c, h, 1], key)[N] - I.ice_values(runs[c, h, -1], key)[N])
                   / (2 * h) for h in (1e-3, 1e-4)}
            rel = min(abs(v - ad) / abs(ad) for v in fds.values()) if ad else float("inf")
            print(f"[smoke] exact4 {key} {c}: J {J0:.15e} ad {ad:.15e} fd {fds} rel {rel:.3e}")
            gated = (key, c) == ("vol_arctic", "cd")
            if not (np.isfinite(ad) and ad != 0.0 and (rel < 1e-3 or not gated)):
                bad.append((key, c, ad, fds))
    assert not bad, bad


def test_remat_blocks_leaves_the_drift_gradient_unchanged():
    """``--remat-blocks`` (farc's memory fix): the one-chunk drift gradient w.r.t. ln P* and ln C_d
    under the frozen-sweep ice adjoint equals the one without nested remat (same forward) to
    rounding: the recomputed blocks are a different XLA program, and the ice map amplifies a
    one-ulp change (CPU 2026-10-11: ln C_d -0.0175920147 vs -0.0175920143, 2e-8 relative)."""
    grads = []
    for rb in (False, True):
        m = I.build("2013-01-01", 2, 2, adjoint_cfg=AdjointConfig.exact(), ice_mode="exact",
                    controls=("pstar", "cd"), ice_adjoint={"adjoint_damping": "frozen_sweep"},
                    remat_blocks=rb)
        assert m.phys.get("remat_blocks", False) is rb
        carry, xs, loss = m.init_carry(), m.xs_fn()(0), I.drift_loss(2)
        g = jax.grad(lambda u: loss(m.chunk_fn(carry, xs, u)))(m.params0)
        grads.append({k: float(np.asarray(v)) for k, v in g.items() if np.ndim(v) == 0})
    assert grads[0].keys() == grads[1].keys() and grads[0]
    for k in grads[0]:
        assert grads[1][k] == pytest.approx(grads[0][k], rel=1e-6, abs=1e-12), k


FORCING_FD = (("tair", 1e-2, 1e-3), ("uair", 1e-2, 1e-3))   # control, two h (K, m/s)


def test_exact4_forcing_gradients_match_fd(exact4, tmp_path):
    """Forcing controls (``--controls ...,tair,uair``: window-constant ``[nod2D]`` anomalies of
    Tair and the model-frame u wind in ``Params.forcing_adj``): with zero anomalies the forward
    rows equal those of the model without them (the seam adds nothing); the terminal ice volume
    (>= 70N) gradient summed over the box nodes (``grads["tair:arctic"]``) agrees with a central
    FD of a uniform anomaly over the box (``perturbed(.., "tair:arctic", h, ±1, mesh)``), and the
    saved map is that gradient."""
    m = I.build("2013-01-01", N, K, ice_mode="exact", ice_settings="evp_rheol_steps=4",
                controls=("pstar", "cd", "tair", "uair"))
    assert set(m.params0["forcing_adj"]) == {"Tair", "u_air"}
    base = tmp_path / "base"
    assert I.run_fwd(m, base, params=m.params0, n_steps=N,
                     save_boundaries=True)["status"] == "complete"
    ref = tmp_path / "ref"
    assert I.run_fwd(exact4, ref, params=exact4.params0, n_steps=N)["status"] == "complete"
    key = "vol_arctic"
    assert I.ice_values(base, key) == I.ice_values(ref, key)
    assert I.drift_sums(base) == I.drift_sums(ref)
    r = I.run_grad(m, tmp_path / "g", base, n_steps=N, budget_s=1e9, cost="vol",
                   cost_box="arctic")
    assert r["status"] == "ok"
    fg = r["_forcing_grad"]
    nodes = I.forcing_box_nodes(m.mesh, "arctic")
    bad = []
    for c, *hs in FORCING_FD:
        assert r["grads"][f"{c}:arctic"] == pytest.approx(float(fg[c][nodes].sum()), rel=1e-15)
        ad = r["grads"][f"{c}:arctic"]
        fds = {}
        for h in hs:
            J = {}
            for s in (1, -1):
                d = tmp_path / f"{c}_{h:g}_{s}"
                u = I.perturbed(m.params0, f"{c}:arctic", h, s, mesh=m.mesh)
                assert np.count_nonzero(u["forcing_adj"][I.FORCING_CONTROLS[c]]) == nodes.sum()
                assert I.run_fwd(m, d, params=u, n_steps=N)["status"] == "complete"
                J[s] = I.ice_values(d, key)[N]
            fds[h] = (J[1] - J[-1]) / (2 * h)
        rel = min(abs(v - ad) / abs(ad) for v in fds.values()) if ad else float("inf")
        print(f"[smoke] exact4 {key} {c}:arctic: ad {ad:.15e} fd {fds} rel {rel:.3e} "
              f"map max|g| {np.abs(fg[c]).max():.3e} outside-box sum {fg[c][~nodes].sum():.3e}")
        if not (np.isfinite(ad) and ad != 0.0 and rel < 1e-3):
            bad.append((c, ad, fds))
    assert not bad, bad
    with pytest.raises(ValueError, match="box"):
        I.perturbed(m.params0, "tair", 1.0, 1, mesh=m.mesh)
    with pytest.raises(ValueError, match="no box"):
        I.perturbed(m.params0, "pstar:arctic", 1.0, 1, mesh=m.mesh)
