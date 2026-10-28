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
  the reverse sweep is cut into legs by a zero budget and resumed.
"""

from __future__ import annotations

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
