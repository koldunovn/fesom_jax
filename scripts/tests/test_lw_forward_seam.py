"""Long-window Task A3 — the restartable + parameter-injectable forward seam.

Two invariants that B1 (restart from the spin-up), D2 (restart + c_k±δ FD) and F (restart +
the trained NN) all depend on, tested fast on CPU (the pi mesh — no CORE2 forcing build):

1. **params=None / injection** (``build_run_params``): no flags ⇒ ``None`` (the byte-identical
   default); ``--ck`` ⇒ a Params with ``tke_c_k`` set and every other leaf at its default;
   ``--nn-pkl`` ⇒ a Params carrying the loaded ``tke_nn``. And on a short forward,
   ``params=None`` is byte-identical to ``params=Params.defaults()`` (the port-wide invariant).
2. **restart bit-identity**: saving the FULL State at step k (``device_get`` → pickle →
   ``device_put``) and continuing with ``is_first_step=False`` reproduces the straight-through
   run BYTE-for-byte — because the saved State already carries the AB2 history (T_old/S_old/…),
   so a restart is indistinguishable from never having stopped.

Token: LW_FORWARD_SEAM_OK. Runs standalone (`pytest scripts/tests/`).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/ on the path
import core2_kpp_climate_run as climr   # noqa: E402

from fesom_jax import calibrate, forcing, ic, ssh, tke_nn   # noqa: E402
from fesom_jax import step as stepmod   # noqa: E402
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh   # noqa: E402
from fesom_jax.params import Params   # noqa: E402

DT = 100.0


# ============================================================================
# 1. build_run_params — the injection seam + the params=None default
# ============================================================================
def test_no_flags_is_none():
    # the byte-identical default: no override ⇒ None (step → Params.defaults() internally)
    assert climr.build_run_params(None, "") is None
    assert climr.build_run_params() is None


def test_ck_sets_only_tke_c_k():
    p = climr.build_run_params(ck=0.7)
    assert isinstance(p, Params)
    assert float(np.asarray(p.tke_c_k)) == pytest.approx(0.7)
    # every other leaf stays at its config default
    d = Params.defaults()
    assert float(np.asarray(p.k_ver)) == float(np.asarray(d.k_ver))
    assert float(np.asarray(p.tke_c_eps)) == float(np.asarray(d.tke_c_eps))


def test_nn_pkl_injects_tke_nn(tmp_path):
    nn = tke_nn.init_tke_nn(jax.random.PRNGKey(0), hidden=(4,), zero_last=True)
    blob = dict(nn=jax.device_get(nn), hidden=[4], seed=0)
    pkl = tmp_path / "nn.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(blob, f)
    p = climr.build_run_params(nn_pkl=str(pkl))
    assert isinstance(p, Params)
    assert isinstance(p.tke_nn, tke_nn.TkeNN)
    # weights round-tripped exactly
    assert np.array_equal(np.asarray(p.tke_nn.Ws[0]), np.asarray(nn.Ws[0]))


# ============================================================================
# 2. restart bit-identity on the pi mesh
# ============================================================================
@pytest.fixture(scope="module")
def pi():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    mesh = load_mesh()
    return dict(mesh=mesh, op=ssh.build_ssh_operator(mesh, dt=DT),
                stress=forcing.surface_stress(mesh), st0=ic.initial_state(mesh))


def _cont(pi, state, n, params=None):
    """Continue n steps with NO cold-start (is_first_step=False throughout) — the restart loop."""
    for _ in range(n):
        state = stepmod.step_jit(state, pi["mesh"], pi["op"], pi["stress"], params,
                                 dt=DT, is_first_step=False)
    return state


def _roundtrip(state):
    """device_get → pickle → device_put, exactly as --save-state / --load-state do."""
    return jax.device_put(pickle.loads(pickle.dumps(jax.device_get(state))))


def _assert_byte_identical(a, b, msg):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for i, (x, y) in enumerate(zip(la, lb)):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape, f"{msg}: leaf {i} shape {x.shape} vs {y.shape}"
        assert np.array_equal(x, y, equal_nan=True), \
            f"{msg}: leaf {i} differs (max|Δ|={np.nanmax(np.abs(x - y)):.2e})"


@pytest.mark.parametrize("k", [1, 3])
def test_restart_is_byte_identical(pi, k):
    M = 6
    straight = stepmod.run(pi["st0"], pi["mesh"], pi["op"], pi["stress"], M, dt=DT)
    sk = stepmod.run(pi["st0"], pi["mesh"], pi["op"], pi["stress"], k, dt=DT)
    sk_rt = _roundtrip(sk)                       # save + reload (full State)
    restart = _cont(pi, sk_rt, M - k)            # continue, is_first_step=False
    _assert_byte_identical(straight, restart, f"restart@k={k}")


def test_params_none_equals_defaults(pi):
    # passing explicit Params.defaults() (via build_params({})) is byte-identical to params=None
    fin_none = stepmod.run(pi["st0"], pi["mesh"], pi["op"], pi["stress"], 3, params=None, dt=DT)
    fin_def = stepmod.run(pi["st0"], pi["mesh"], pi["op"], pi["stress"], 3,
                          params=calibrate.build_params({}), dt=DT)
    _assert_byte_identical(fin_none, fin_def, "params=None vs defaults")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
