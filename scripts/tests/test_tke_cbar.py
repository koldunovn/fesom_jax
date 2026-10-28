"""The TKE-constant bar driver ``scripts/adjoint/tke_cbar.py`` on the CORE2 reference model
(2013-01-01 restart, ``configs/core2_full.yaml``, prescribed sea ice), 2-step windows, 1-step
chunks.

⚠️ HEAVY (CPU compute node, >= 200 GB): CORE2 chunk forward and VJP programs. ``scripts`` group.

* the base forward's records (``S`` at every boundary, the per-node MLD sums at the window
  ends) equal the carry of a plain chunked forward, and ``J_T``/``J_MLD`` from them equal the
  driver's cost functions on that carry;
* the reverse sweep over the base run's linked boundaries gives the gradient of a direct
  ``run_grad`` bitwise (CPU), also when it is cut into legs by a zero budget;
* the same at boundary stride 2 (every 2nd boundary used, the span recomputed), with the
  boundaries read from disk per span (default) and loaded to the host first;
  and over a base that saved only every 2nd boundary (``save_every``);
* a perturbed forward resumed from its saved carry equals the uninterrupted one bitwise;
* ``MLDm`` with an empty mask equals ``MLD``; with every node masked its gradient is zero;
* the FD of ``J_T`` along ln c_k agrees with the adjoint (2 steps: linear);
* ``jumping_nodes``, ``plateau`` and the ``J_T`` / ``J_Tfull`` window algebra on synthetic data.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import core2_adj as C  # noqa: E402
import tke_cbar as B  # noqa: E402

from fesom_jax import adjoint  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402

RESTART = C.STATES_DIR / "fesom.2013.001.00000"
pytestmark = pytest.mark.skipif(
    not ((ROOT / "data" / "mesh_core2").is_dir() and RESTART.is_dir()),
    reason="CORE2 mesh / 2013 States missing")
START = datetime.datetime(2013, 1, 1)
N = 2


class _Args:
    start = "2013-01-01"
    chunk_steps = 1
    ice_mode = "prescribed"           # the --ice-mode default


@pytest.fixture(scope="module")
def model():
    return B.build(_Args, AdjointConfig.exact(), N)


@pytest.fixture(scope="module")
def base(model, tmp_path_factory):
    d = tmp_path_factory.mktemp("cbar") / "base"
    row = B.run_fwd(model, d, const="none", h=0.0, sign=1, n_steps=N, ends=[1, 2],
                    save_boundaries=True, budget_s=1e9)
    assert row["status"] == "complete"
    return d


def _grads(row):
    return {c: row["grads"][c] for c in B.CONSTS}


# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------
def test_j_t_window_algebra(monkeypatch):
    monkeypatch.setattr(B, "T_MEAN_STEPS", 3)
    S = {0: 0.0, 1: 1.0, 2: 3.5, 3: 6.0, 4: 10.0, 5: 11.0, 6: 17.0}
    assert B.j_t(S, 2) == 3.5 / 2
    assert B.j_t(S, 6) == (17.0 - 6.0) / 3
    # J_T(2m) = 2·Tfull(2m) - J_T(m)
    assert abs(B.j_t(S, 6) - (2 * S[6] / 6 - B.j_t(S, 3))) < 1e-15
    assert B.j_tfull(S, 6) == 17.0 / 6 and B.j_tfull(S, 2) == B.j_t(S, 2)


def test_jumping_nodes_and_plateau():
    m0 = np.array([10.0, 10.0, 10.0, 10.0])
    mp = np.array([10.1, 15.0, 10.0, 10.0 + 1e-9])
    mm = np.array([9.9, 10.0, 10.0, 10.0 - 1e-9])
    assert B.jumping_nodes(mm, m0, mp).tolist() == [False, True, False, False]
    pl = B.plateau([1e-1, 3e-2, 1e-2, 3e-3], [1.30, 1.01, 1.00, 0.99])
    assert pl["h"] == [3e-3, 1e-2, 3e-2] and abs(pl["fd"] - 1.00) < 1e-12
    assert B.plateau([1e-1, 1e-2, 1e-3], [1.0, 2.0, 3.0]) is None


def test_bar_verdict():
    fd = {3e-3: 1.00, 1e-2: 1.01, 3e-2: 1.02}              # a plateau (2 %) at 1.01
    v = B.bar_verdict(1.08, fd, 5.0)
    assert v["plateau"] and abs(v["fd_ref"] - 1.01) < 1e-12 and v["bar"]
    assert B.bar_verdict(1.2, fd, 5.0)["bar"] is False
    assert B.bar_verdict(1.2, fd, 2.0)["bar"] is None       # below 5 days: not judged
    wide = {3e-3: 0.8, 1e-2: 1.0, 3e-2: 1.3}                # no plateau: middle h, or the range
    v = B.bar_verdict(1.25, wide, 10.0)
    assert v["plateau"] is None and v["fd_ref"] == 1.0 and v["bar"] and v["rel_err"] > 0.1
    assert abs(v["fd_spread"] - 0.3) < 1e-12
    assert B.bar_verdict(1.4, wide, 10.0)["bar"] is False


# ---------------------------------------------------------------------------
# CORE2
# ---------------------------------------------------------------------------
def test_base_records_equal_the_forward_and_the_costs(model, base):
    fin = adjoint.chunked_forward(model.chunk_fn, model.params0, init_state=model.init_carry(),
                                  n_chunks=N, xs_fn=model.xs_fn())
    S, M = B.fwd_records(base)
    host = __import__("jax").device_get(fin)
    assert S[N] == float(host[1]["boxT"])
    assert np.array_equal(M[N], np.asarray(host[1]["mld"]))
    assert abs(B.j_t(S, N) - float(B.cost_fn("T", N, model.mesh)(fin))) <= 1e-15 * abs(B.j_t(S, N))
    w = B.surface_weights(model.mesh)
    jm = B.j_mld(M[N], N, w)
    assert abs(jm - float(B.cost_fn("MLD", N, model.mesh)(fin))) <= 1e-13 * abs(jm)
    assert sorted(p.name for p in base.glob("boundary_*.npz")) == [
        "boundary_0.npz", "boundary_1.npz", "boundary_2.npz"]


def test_grad_over_linked_boundaries_equals_direct_and_resumes(model, base, tmp_path):
    direct = C.run_grad(model, (B.cost_fn("T", N, model.mesh),), check_linear=False)[0]
    g_direct = {c: float(np.asarray(direct["_result"]["grad"][B.PARAM[c]])) for c in B.CONSTS}
    one = B.run_cbar_grad(model, tmp_path / "one", base, cost="T", n_steps=N, budget_s=1e9)
    assert one["status"] == "ok"
    assert _grads(one) == g_direct
    # legs: a zero budget stops after every reverse chunk; rerun until done
    legs = 0
    while True:
        r = B.run_cbar_grad(model, tmp_path / "legs", base, cost="T", n_steps=N, budget_s=0.0)
        legs += 1
        if r["status"] == "ok":
            break
        assert r["status"] == "incomplete" and legs < 5
    assert legs == N and _grads(r) == g_direct
    assert all(abs(v) > 0 for v in g_direct.values())


@pytest.mark.parametrize("lazy", [True, False])
def test_leg_checkpoint_stride_loads_every_nth_and_the_last(tmp_path, lazy):
    for c in range(8):
        np.savez(tmp_path / f"boundary_{c}.npz", np.array([float(c)]))
    got = B.LegCheckpoint(tmp_path, None, stride=3, lazy=lazy).load_boundaries(like=np.zeros(1))
    assert sorted(got) == [0, 3, 6, 7] and float(got[7][0]) == 7.0
    with pytest.raises(KeyError):
        got[4]
    assert adjoint._host_gb(got) == (0.0 if lazy else 4 * 8 / 1e9)
    every = B.LegCheckpoint(tmp_path, None, lazy=lazy).load_boundaries(like=np.zeros(1))
    assert sorted(every) == list(range(8))
    if lazy:                     # a resumed forward's new boundaries stay on the host
        every[8] = np.array([8.0])
        assert sorted(every) == list(range(9)) and adjoint._host_gb(every) == 8 / 1e9


def test_grad_with_boundary_stride_equals_stride_one(model, base, tmp_path):
    one = B.run_cbar_grad(model, tmp_path / "s1", base, cost="T", n_steps=N, budget_s=1e9)
    two = B.run_cbar_grad(model, tmp_path / "s2", base, cost="T", n_steps=N, budget_s=1e9,
                          stride=2)
    host = B.run_cbar_grad(model, tmp_path / "s2h", base, cost="T", n_steps=N, budget_s=1e9,
                           stride=2, lazy=False)
    assert one["status"] == two["status"] == host["status"] == "ok"
    assert _grads(two) == _grads(one) == _grads(host)
    assert two["host_gb"] == 0.0 < host["host_gb"]

def test_sparse_base_with_strided_grad(model, base, tmp_path):
    sparse = tmp_path / "sparse"
    row = B.run_fwd(model, sparse, const="none", h=0.0, sign=1, n_steps=N, ends=[1, 2],
                    save_boundaries=True, budget_s=1e9, save_every=2)
    assert row["status"] == "complete"
    assert sorted(p.name for p in sparse.glob("boundary_*.npz")) == [
        "boundary_0.npz", "boundary_2.npz"]
    dense = B.run_cbar_grad(model, tmp_path / "g1", base, cost="T", n_steps=N, budget_s=1e9)
    strided = B.run_cbar_grad(model, tmp_path / "g2", sparse, cost="T", n_steps=N,
                              budget_s=1e9, stride=2)
    assert dense["status"] == strided["status"] == "ok"
    assert _grads(strided) == _grads(dense)

def test_masked_mld_cost(model, base, tmp_path):
    full = B.run_cbar_grad(model, tmp_path / "mld", base, cost="MLD", n_steps=N, budget_s=1e9)
    none = B.run_cbar_grad(model, tmp_path / "m0", base, cost="MLDm", n_steps=N, budget_s=1e9,
                           mask=np.zeros(model.mesh.nod2D, bool))
    allm = B.run_cbar_grad(model, tmp_path / "m1", base, cost="MLDm", n_steps=N, budget_s=1e9,
                           mask=np.ones(model.mesh.nod2D, bool))
    assert _grads(none) == _grads(full)
    assert all(v == 0.0 for v in _grads(allm).values())
    assert any(v != 0.0 for v in _grads(full).values())
    assert json.loads((tmp_path / "m1" / "forward.json").read_text())["loss"] == \
        json.loads((tmp_path / "mld" / "forward.json").read_text())["loss"]


def test_resumed_forward_equals_uninterrupted_and_fd_matches(model, base, tmp_path):
    h = 1e-2
    a = B.run_fwd(model, tmp_path / "a", const="c_k", h=h, sign=1, n_steps=N, ends=[N],
                  save_boundaries=False, budget_s=1e9)
    b1 = B.run_fwd(model, tmp_path / "b", const="c_k", h=h, sign=1, n_steps=N, ends=[N],
                   save_boundaries=False, budget_s=0.0)
    assert a["status"] == "complete" and b1["status"] == "incomplete"
    assert (tmp_path / "b" / "carry_1.npz").exists()
    b2 = B.run_fwd(model, tmp_path / "b", const="c_k", h=h, sign=1, n_steps=N, ends=[N],
                   save_boundaries=False, budget_s=1e9)
    assert b2["status"] == "complete" and b2["chunks_from"] == 1
    Sa, Ma = B.fwd_records(tmp_path / "a")
    Sb, Mb = B.fwd_records(tmp_path / "b")
    assert Sa[N] == Sb[N] and np.array_equal(Ma[N], Mb[N])
    B.run_fwd(model, tmp_path / "m", const="c_k", h=h, sign=-1, n_steps=N, ends=[N],
              save_boundaries=False, budget_s=1e9)
    Sm, _ = B.fwd_records(tmp_path / "m")
    fd = (B.j_t(Sa, N) - B.j_t(Sm, N)) / (2 * h)
    ad = B.run_cbar_grad(model, tmp_path / "g", base, cost="T", n_steps=N,
                         budget_s=1e9)["grads"]["c_k"]
    print(f"\n  2-step J_T d/d ln c_k: FD {fd:.6e}  adjoint {ad:.6e}")
    assert abs(fd - ad) <= 2e-2 * abs(ad)
