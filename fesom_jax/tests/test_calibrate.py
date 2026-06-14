"""A3 gate: the calibrate.py optimizer seam — `optimize` / `grid_scan` / `build_params`.

Fast CPU unit tests (no CORE2 forward — keeps the suite green) that exercise the *machinery*
the calibration (§2) and hybrid-ML (§3) pillars drive: recover an analytic minimum over a
multi-leaf pytree, confirm the grid_scan misfit bowl, and check the dict→Params seam.
The CORE2 perfect-model twin (`TWIN_RECOVER_OK`) is a separate GPU acceptance gate; here we
prove the loop converges and the seam is wired. Token: CALIBRATE_SEAM_OK.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fesom_jax import calibrate
from fesom_jax.config import (A_VER, K_GM_MAX, K_VER, REDI_KMAX,
                              TKE_C_EPS, TKE_C_K, TKE_CD, TKE_ALPHA)
from fesom_jax.params import Params


# --------------------------------------------------------------------------
# optimize — analytic minimum over a multi-leaf pytree
# --------------------------------------------------------------------------
def _bowl(p):
    """loss({'a','b'}) = (a-3)² + (b+1)²; min at (3, -1), value 0."""
    return (p["a"] - 3.0) ** 2 + (p["b"] + 1.0) ** 2


def test_optimize_recovers_analytic_min():
    init = {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)}
    opt = optax.adam(0.2)
    final, history = calibrate.optimize(_bowl, init, opt, n_iters=400)
    assert abs(float(final["a"]) - 3.0) < 1e-3
    assert abs(float(final["b"]) + 1.0) < 1e-3
    # monotone-ish descent: the last loss is far below the first.
    assert history[-1]["loss"] < 1e-6 * history[0]["loss"] + 1e-9
    # history records carry it / loss / gnorm / params.
    assert {"it", "loss", "gnorm", "params"} <= set(history[0])
    assert history[0]["it"] == 1 and history[-1]["it"] == len(history)


def test_optimize_cosine_decay_like_the_twin_recipe():
    """The real `k_gm` twin uses Adam over a cosine-decay schedule (big early, fine late);
    confirm that schedule converges on the bowl too."""
    init = {"a": jnp.asarray(-5.0), "b": jnp.asarray(5.0)}
    iters = 300
    sched = optax.cosine_decay_schedule(init_value=0.3, decay_steps=iters, alpha=0.02)
    final, _ = calibrate.optimize(_bowl, init, optax.adam(sched), n_iters=iters)
    assert abs(float(final["a"]) - 3.0) < 1e-2
    assert abs(float(final["b"]) + 1.0) < 1e-2


def test_optimize_stop_fn_early_stops_and_keeps_triggering_params():
    """stop_fn is checked BEFORE the update, so the returned params are the ones that
    crossed the threshold (not one extra step past it)."""
    init = {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)}
    final, history = calibrate.optimize(
        _bowl, init, optax.adam(0.2), n_iters=10_000,
        stop_fn=lambda r: r["loss"] < 1e-4)
    assert history[-1]["loss"] < 1e-4
    assert len(history) < 10_000                      # stopped early
    # returned params == the last history snapshot's params (the triggering iterate)
    np.testing.assert_array_equal(np.asarray(final["a"]),
                                  np.asarray(history[-1]["params"]["a"]))


def test_optimize_keep_params_false_omits_snapshots():
    init = {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)}
    _, history = calibrate.optimize(_bowl, init, optax.adam(0.2),
                                    n_iters=5, keep_params=False)
    assert all("params" not in r for r in history)
    assert all({"it", "loss", "gnorm"} <= set(r) for r in history)


# --------------------------------------------------------------------------
# grid_scan — the misfit bowl
# --------------------------------------------------------------------------
def test_grid_scan_bowl_argmin_at_injected_value():
    """A 1-D bowl in 'k' centred at 1500 — argmin of the sweep must land on 1500."""
    target = 1500.0

    def loss(p):
        return (p["k"] - target) ** 2 + 7.0    # +const ⇒ exercises the offset too

    base = {"k": jnp.asarray(target)}
    values = list(np.linspace(625.0, 1675.0, 11)) + [target]
    scan = calibrate.grid_scan(loss, base, "k", values)
    vs = np.array([v for v, _ in scan])
    losses = np.array([l for _, l in scan])
    assert vs.shape == (12,) and losses.shape == (12,)
    # the injected value is the argmin and hits the bowl floor (7.0)
    vmin = vs[int(np.argmin(losses))]
    assert vmin == pytest.approx(target)
    assert losses.min() == pytest.approx(7.0, abs=1e-9)
    # the bowl is convex away from the min (monotone on each side)
    order = np.argsort(vs)
    vs_s, ls_s = vs[order], losses[order]
    left = vs_s <= target
    assert np.all(np.diff(ls_s[left]) <= 1e-9)        # decreasing up to the min
    assert np.all(np.diff(ls_s[~left]) >= -1e-9)      # increasing after it


# --------------------------------------------------------------------------
# build_params — the dict → Params seam
# --------------------------------------------------------------------------
def test_build_params_defaults_when_empty_matches_Params_defaults():
    p = calibrate.build_params({})
    d = Params.defaults()
    for name in ("k_ver", "a_ver", "k_gm", "redi_kmax",
                 "tke_c_k", "tke_c_eps", "tke_cd", "tke_alpha"):
        np.testing.assert_array_equal(np.asarray(getattr(p, name)),
                                      np.asarray(getattr(d, name)))


def test_build_params_sets_named_leaves_defaults_the_rest():
    p = calibrate.build_params({"k_gm": 1500.0, "redi_kmax": 1500.0})
    assert float(p.k_gm) == 1500.0
    assert float(p.redi_kmax) == 1500.0
    assert p.k_gm.dtype == jnp.float64
    # untouched leaves keep the config defaults
    assert float(p.k_ver) == K_VER
    assert float(p.a_ver) == A_VER
    assert float(p.tke_c_k) == TKE_C_K


def test_build_params_is_generic_no_hidden_kgm_redi_sync():
    """build_params is generic — tuning only k_gm leaves redi_kmax at its default (the
    coupling is the caller's job; the namelist writer enforces it on the Fortran side)."""
    p = calibrate.build_params({"k_gm": 1500.0})
    assert float(p.k_gm) == 1500.0
    assert float(p.redi_kmax) == REDI_KMAX        # NOT synced to 1500


def test_build_params_rejects_unknown_leaf():
    with pytest.raises(ValueError, match="unknown tunable leaf"):
        calibrate.build_params({"not_a_param": 1.0})


def test_calibrate_seam_ok_token():
    """Aggregate gate: the seam is wired (optimize converges, grid_scan bowls, build_params
    maps) — prints CALIBRATE_SEAM_OK for the acceptance log."""
    init = {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)}
    final, _ = calibrate.optimize(_bowl, init, optax.adam(0.2), n_iters=400)
    converged = abs(float(final["a"]) - 3.0) < 1e-3 and abs(float(final["b"]) + 1.0) < 1e-3
    seam = float(calibrate.build_params({"k_gm": 1234.0}).k_gm) == 1234.0
    assert converged and seam
    print("CALIBRATE_SEAM_OK")
