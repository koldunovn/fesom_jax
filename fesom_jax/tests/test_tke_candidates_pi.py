"""``AdjointConfig.freeze_mixing_state`` (the parametric-only TKE adjoint) and the candidate
settings on the assembled ``mixing_tke``, pi mesh (``unit`` group; the column core is in
``test_tke_candidates.py``, the CORE2 step in ``test_tke_candidates_core2.py``).

* off (``freeze_state=False``, ``tke_adj=None``) ⇒ the ``mixing_tke`` jaxpr pinned before the
  cut points existed;
* on ⇒ ``(Kv, Av, uvnode, tke_new)`` bitwise;
* the cotangents of uv, bvfreq, hnode and the stress are exactly zero; those of tke and of the
  four TKE constants equal the exact ones (within one call no state input depends on them).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax.adjoint_stab import AdjointConfig
from fesom_jax.tests.test_tke_cuts import _sha, check_pin


def _opts(**kw):
    return AdjointConfig(**kw).validate().tke_options()


# ---------------------------------------------------------------------------
# freeze_mixing_state on the assembled mixing_tke (pi mesh)
# ---------------------------------------------------------------------------
def _pi():
    from fesom_jax.tests.test_tke_cuts_pi import _pi_driver
    return _pi_driver()


def _mixing(mesh, a, p, **kw):
    from fesom_jax import tke as tkemod
    return tkemod.mixing_tke(mesh, a["uv"], a["bvfreq"], a["tke"], a["stress_node_surf"],
                             a["hnode"], tkemod.TkeConfig(), p, dt=1800.0, Z3d=a["Z3d"], **kw)


_PI: dict = {}


def _pi_vjp(freeze_state, tke_adj=None):
    key = (freeze_state, tke_adj)
    if key not in _PI:
        from fesom_jax.params import Params
        mesh, a = _pi()
        ins = {k: a[k] for k in ("uv", "bvfreq", "tke", "stress_node_surf", "hnode")}
        p0 = Params.defaults()
        pc = {n: getattr(p0, n) for n in ("tke_c_k", "tke_c_eps", "tke_cd", "tke_alpha")}

        def f(ins, pc):
            p = dataclasses.replace(p0, **pc)
            Kv, Av, _, tke_new = _mixing(mesh, {**a, **ins}, p, freeze_state=freeze_state,
                                         tke_adj=tke_adj)
            return Kv, Av, tke_new

        out, vjp = jax.vjp(f, ins, pc)
        rng = np.random.default_rng(4)
        ys = tuple(jnp.asarray(rng.standard_normal(np.shape(o))) for o in out)
        _PI[key] = (jax.device_get(out), jax.device_get(vjp(ys)))
    return _PI[key]


def test_mixing_state_off_is_the_pinned_mixing_tke_program():
    from fesom_jax.tests.test_tke_cuts_pi import PINNED_PI
    mesh, a = _pi()
    from fesom_jax.params import Params
    got = _sha(str(jax.make_jaxpr(lambda a: _mixing(mesh, a, Params.defaults(),
                                                    freeze_state=False, tke_adj=None))(a)))
    check_pin(PINNED_PI, "mixing_tke_pi", got)


def test_mixing_state_forward_bitwise():
    from fesom_jax.params import Params
    mesh, a = _pi()
    ref = jax.jit(lambda a: _mixing(mesh, a, Params.defaults()))(a)
    got = jax.jit(lambda a: _mixing(mesh, a, Params.defaults(), freeze_state=True,
                                    tke_adj=_opts(tke_pr_cut=1e-5, tke_kme_tke=1.2e-4)))(a)
    for u, v in zip(ref, got):
        assert np.array_equal(np.asarray(u), np.asarray(v))


def test_pr_ratio_on_mixing_tke_is_forward_bitwise_and_needs_the_raw_n2():
    import pytest
    from fesom_jax.params import Params
    mesh, a = _pi()
    raw = -50.0 * a["bvfreq"]                      # every interface a cancellation residual
    ref = jax.jit(lambda a: _mixing(mesh, a, Params.defaults()))(a)
    got = jax.jit(lambda a: _mixing(mesh, a, Params.defaults(),
                                    tke_adj=_opts(tke_pr_cut=1e-5, tke_pr_ratio=10.0),
                                    bvfreq_raw=raw))(a)
    for u, v in zip(ref, got):
        assert np.array_equal(np.asarray(u), np.asarray(v))
    with pytest.raises(ValueError, match="unsmoothed"):
        _mixing(mesh, a, Params.defaults(), tke_adj=_opts(tke_pr_ratio=10.0))


def test_pr_raw_cut_on_mixing_tke_is_forward_bitwise_and_needs_the_raw_n2():
    import pytest
    from fesom_jax.params import Params
    mesh, a = _pi()
    raw = -a["bvfreq"]                             # every stable interface locally inverted
    ref = jax.jit(lambda a: _mixing(mesh, a, Params.defaults()))(a)
    got = jax.jit(lambda a: _mixing(mesh, a, Params.defaults(),
                                    tke_adj=_opts(tke_pr_cut=1e-5, tke_pr_raw_cut=0.0),
                                    bvfreq_raw=raw))(a)
    for u, v in zip(ref, got):
        assert np.array_equal(np.asarray(u), np.asarray(v))
    with pytest.raises(ValueError, match="unsmoothed"):
        _mixing(mesh, a, Params.defaults(), tke_adj=_opts(tke_pr_raw_cut=0.0))


def test_mixing_state_cuts_the_state_and_keeps_tke_and_constants():
    _, (ci0, cp0) = _pi_vjp(False)
    _, (ci1, cp1) = _pi_vjp(True)
    for k in ("uv", "bvfreq", "hnode", "stress_node_surf"):
        assert np.any(np.asarray(ci0[k]) != 0.0), f"exact {k} cotangent is zero: fixture dead"
        assert np.all(np.asarray(ci1[k]) == 0.0), f"{k} cotangent survived freeze_mixing_state"
    np.testing.assert_allclose(np.asarray(ci1["tke"]), np.asarray(ci0["tke"]), rtol=1e-13,
                               atol=1e-300)
    for k in cp0:
        a, b = float(np.asarray(cp0[k])), float(np.asarray(cp1[k]))
        assert a != 0.0 and abs(a - b) <= 1e-13 * abs(a), (k, a, b)
