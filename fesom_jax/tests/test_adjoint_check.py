"""``fesom_jax.adjoint_check`` — the chunked window maps on the pi mesh.

The window maps are tested on the pi mesh (8 steps, 2 chunks of 4) against a monolithic
:func:`jax.jvp` / :func:`jax.vjp` of the same 8 steps, with a state tangent and with a parameter
tangent, and pinned against unpinned; plus the pinned window dot test and ``volume_weights``.
The model-free checks (dot test, FD sweep, power iteration, fdlin) are in
``test_adjoint_check_unit.py``.

Tolerances between two DIFFERENT compiled programs of this model are the measured spread of
two equivalent programs (the monolithic VJP with remat on and off), at most ten times it: the
early-stopped SSH solve turns round-off differences into iterate differences, so a fixed 1e-14
would test the solver's stop, not the chunking.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import adjoint_check as AC
from fesom_jax import forcing, ic, ssh
from fesom_jax.adjoint import scan_steps
from fesom_jax.config import A_VER, K_VER
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params
from fesom_jax.step import step
from fesom_jax.tests import _ssh_implicit_common as C


# ---------------------------------------------------------------------------
# window maps on the pi mesh

DT = 100.0
N_STEPS = 8
K = 4                     # 2 chunks
XS_HOST = np.asarray(1.0 + 0.05 * np.cos(np.arange(N_STEPS)), dtype=np.float64)


@pytest.fixture(scope="module")
def pi():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    mesh = load_mesh()
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress = forcing.surface_stress(mesh)
    st = ic.initial_state(mesh)
    st = dataclasses.replace(st, T=st.T + C.node3d(mesh, 0.5, 0),
                             uv=st.uv + C.elem_uv(mesh, 0.02, 0))

    def adv(s, x, p):
        return step(s, mesh, op, stress * x, p, dt=DT, is_first_step=False)

    def chunk_fn(s, xs, p):
        return scan_steps(lambda a, x: adv(a, x, p), s, xs, rematerialize=True)

    def chunk_fn_tl(s, xs, p):
        return scan_steps(lambda a, x: adv(a, x, p), s, xs, rematerialize=False)

    def window(s, p, remat):
        return scan_steps(lambda a, x: adv(a, x, p), s, jnp.asarray(XS_HOST),
                          rematerialize=remat)

    params = Params(k_ver=jnp.asarray(K_VER, jnp.float64), a_ver=jnp.asarray(A_VER, jnp.float64))
    zero_st = jax.tree.map(jnp.zeros_like, st)
    zero_p = jax.tree.map(jnp.zeros_like, params)
    t_state = (dataclasses.replace(zero_st, T=C.node3d(mesh, 1.0, 1),
                                   uv=C.elem_uv(mesh, 0.02, 1)), zero_p)
    t_params = (zero_st, dataclasses.replace(zero_p, k_ver=jnp.asarray(K_VER, jnp.float64)))
    w = dataclasses.replace(zero_st, T=C.node3d(mesh, 1.0, 0), S=C.node3d(mesh, 10.0, 1),
                            uv=C.elem_uv(mesh, 50.0, 0), eta_n=C.node2d(mesh, 20.0, 1))
    pins = AC.forward_states(chunk_fn, params, st, 2, _xs_fn)
    return dict(mesh=mesh, st=st, params=params, chunk_fn=chunk_fn, chunk_fn_tl=chunk_fn_tl,
                window=window, t_state=t_state, t_params=t_params, w=w, pins=pins)


def _xs_fn(c):
    return jnp.asarray(XS_HOST[c * K:(c + 1) * K])      # fresh every call (the xs_fn contract)


def _rel(a, b) -> float:
    d = jax.tree.map(lambda x, y: np.asarray(x, np.float64) - np.asarray(y, np.float64), a, b)
    return AC._tree_norm(d) / AC._tree_norm(b)


def _bitwise(a, b) -> bool:
    return all(np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


@pytest.fixture(scope="module")
def mono_jvp(pi):
    return jax.jit(lambda s, p, ts, tp: jax.jvp(lambda s_, p_: pi["window"](s_, p_, False),
                                                (s, p), (ts, tp)))


@pytest.fixture(scope="module")
def mono_vjp(pi):
    """Monolithic VJPs with remat on and off, and the spread between them."""
    def make(remat):
        @jax.jit
        def f(s, p, w):
            _, vj = jax.vjp(lambda s_, p_: pi["window"](s_, p_, remat), s, p)
            return vj(w)
        return f
    on = make(True)(pi["st"], pi["params"], pi["w"])
    off = make(False)(pi["st"], pi["params"], pi["w"])
    spread_s = _rel(off[0], on[0])
    spread_k = abs(float(off[1].k_ver - on[1].k_ver)) / abs(float(on[1].k_ver))
    return on, spread_s, spread_k


def test_forward_states_are_the_chunked_grad_boundaries(pi):
    pins = pi["pins"]
    assert len(pins) == 3
    assert _bitwise(pins[0], pi["st"])
    end = jax.jit(lambda s, p: pi["window"](s, p, True))(pi["st"], pi["params"])
    assert _rel(pins[2], end) <= 1e-13


@pytest.mark.parametrize("which", ["t_state", "t_params"])
def test_window_jvp_equals_the_monolithic_jvp(pi, mono_jvp, which, capsys):
    ts, tp = pi[which]
    prim, tan = mono_jvp(pi["st"], pi["params"], ts, tp)
    wj = AC.window_jvp(pi["chunk_fn_tl"], pi["params"], pi["st"], ts, tp, 2, _xs_fn)
    r_t, r_s = _rel(wj.tangent, tan), _rel(wj.state, prim)
    with capsys.disabled():
        print(f"\n[window_jvp {which}] vs monolithic jvp: tangent {r_t:.2e}, primal {r_s:.2e}")
    # the same arithmetic in both (no remat on either side): measured 0.0 on CPU
    assert r_t <= 1e-14 and r_s <= 1e-14
    assert AC._tree_norm(wj.tangent) > 0.0 and len(wj.trace) == 3


def test_window_jvp_without_constant_folding(pi, mono_jvp, capsys):
    """``compiler_options=NO_CONSTANT_FOLDING`` compiles a separate tangent program (its own cache
    entry) that computes the same map: on pi within round-off of the monolithic JVP."""
    ts, tp = pi["t_state"]
    prim, tan = mono_jvp(pi["st"], pi["params"], ts, tp)
    wj = AC.window_jvp(pi["chunk_fn_tl"], pi["params"], pi["st"], ts, tp, 2, _xs_fn,
                       compiler_options=AC.NO_CONSTANT_FOLDING)
    r_t, r_s = _rel(wj.tangent, tan), _rel(wj.state, prim)
    with capsys.disabled():
        print(f"\n[window_jvp no constant folding] vs monolithic jvp: tangent {r_t:.2e}, "
              f"primal {r_s:.2e}")
    assert r_t <= 1e-13 and r_s <= 1e-13 and AC._tree_norm(wj.tangent) > 0.0
    opts = tuple(sorted(AC.NO_CONSTANT_FOLDING.items()))
    assert AC._jitted_tangent(pi["chunk_fn_tl"], opts) is not AC._jitted_tangent(pi["chunk_fn_tl"])
    assert AC._jitted_tangent(pi["chunk_fn_tl"], opts) is AC._jitted_tangent(pi["chunk_fn_tl"], opts)


@pytest.mark.parametrize("which", ["t_state", "t_params"])
def test_window_jvp_pinned_and_unpinned(pi, which, capsys):
    ts, tp = pi[which]
    unp = AC.window_jvp(pi["chunk_fn_tl"], pi["params"], pi["st"], ts, tp, 2, _xs_fn)
    # pinned on the tangent sweep's OWN primal: the identical computation, bit for bit
    first = AC.window_jvp(pi["chunk_fn_tl"], pi["params"], pi["st"], ts, tp, 1, _xs_fn)
    own = AC.window_jvp(pi["chunk_fn_tl"], pi["params"], pi["st"], ts, tp, 2, _xs_fn,
                        pinned_states=[pi["st"], first.state])
    assert _bitwise(own.tangent, unp.tangent)
    # pinned on forward_states (the forward program, not the JVP program's primal): the
    # two programs' primals differ by round-off, and so do the tangents
    pin = AC.window_jvp(pi["chunk_fn_tl"], pi["params"], pi["st"], ts, tp, 2, _xs_fn,
                        pinned_states=pi["pins"])
    r_prim = _rel(first.state, pi["pins"][1])
    r_t = _rel(pin.tangent, unp.tangent)
    with capsys.disabled():
        print(f"\n[window_jvp {which}] pinned(forward_states) vs unpinned: tangent {r_t:.2e} "
              f"(chunk-1 primal jvp-program vs forward-program {r_prim:.2e})")
    assert r_t <= 1e-13
    # its primal end comes from the JVP program too: equal to forward_states' end only to
    # round-off, not bitwise
    assert _rel(pin.state, pi["pins"][2]) <= 1e-13


def test_window_vjp_equals_the_monolithic_vjp(pi, mono_vjp, capsys):
    on, spread_s, spread_k = mono_vjp
    for pinned in (None, pi["pins"]):
        wv = AC.window_vjp(pi["chunk_fn"], pi["params"], pi["st"], pi["w"], 2, _xs_fn,
                           chunk_steps=K, pinned_states=pinned)
        r_s = _rel(wv.state_cot, on[0])
        r_k = abs(float(wv.params_cot.k_ver - on[1].k_ver)) / abs(float(on[1].k_ver))
        with capsys.disabled():
            print(f"\n[window_vjp pinned={pinned is not None}] vs monolithic vjp: state "
                  f"{r_s:.2e} (remat on/off spread {spread_s:.2e}), k_ver {r_k:.2e} "
                  f"(spread {spread_k:.2e})")
        assert r_s <= 10 * max(spread_s, 1e-16)
        assert r_k <= 10 * max(spread_k, 1e-16)
        assert wv.result.n_chunks == 2 and len(wv.result.trace) == 3


def test_window_vjp_pinned_and_unpinned_are_bitwise(pi):
    a = AC.window_vjp(pi["chunk_fn"], pi["params"], pi["st"], pi["w"], 2, _xs_fn, chunk_steps=K)
    b = AC.window_vjp(pi["chunk_fn"], pi["params"], pi["st"], pi["w"], 2, _xs_fn, chunk_steps=K,
                      pinned_states=pi["pins"])
    assert _bitwise((a.state_cot, a.params_cot), (b.state_cot, b.params_cot))
    assert b.result.forward_seconds == 0.0           # no forward of its own
    with pytest.raises(ValueError, match="entries"):
        AC.window_vjp(pi["chunk_fn"], pi["params"], pi["st"], pi["w"], 2, _xs_fn,
                      pinned_states=pi["pins"][:2])


def test_window_dot_test_on_pinned_states(pi, capsys):
    """The tangent-linear and adjoint window maps on the same pinned trajectory are exact
    transposes (state and params tangent together)."""
    P, st, pins = pi["params"], pi["st"], pi["pins"]
    ts, _ = pi["t_state"]
    _, tp = pi["t_params"]

    def jvp(t):
        return AC.window_jvp(pi["chunk_fn_tl"], P, st, t[0], t[1], 2, _xs_fn,
                             pinned_states=pins).tangent

    def vjp(c):
        r = AC.window_vjp(pi["chunk_fn"], P, st, c, 2, _xs_fn, chunk_steps=K,
                          pinned_states=pins)
        return r.state_cot, r.params_cot

    rows = AC.dot_test(None, None, (ts, tp), pi["w"], jvp=jvp, vjp=vjp)
    with capsys.disabled():
        print("\n[window dot test, pinned] " + ", ".join(
            f"amp {r.amp:.0e}: rel {r.rel:.1e}" for r in rows))
    for r in rows:
        assert r.rel <= 1e-12, r


def test_volume_weights(pi):
    mesh, st = pi["mesh"], pi["st"]
    wts = AC.volume_weights(st, mesh)
    assert set(wts) == {"T", "S", "uv", "eta_n"}
    assert float(jnp.sum(wts["T"])) == pytest.approx(1.0, rel=1e-12)
    assert float(jnp.sum(wts["S"])) * 0.1 ** 2 == pytest.approx(1.0, rel=1e-12)
    assert float(jnp.sum(wts["uv"])) * 0.1 ** 2 == pytest.approx(1.0, rel=1e-12)
    assert wts["uv"].shape[-1] == 1                    # broadcasts over (u, v)
    assert np.all(np.asarray(wts["T"])[~np.asarray(mesh.node_layer_mask)] == 0.0)
    # uniform T perturbation of 1 K has norm 1 in these weights
    one = dataclasses.replace(jax.tree.map(jnp.zeros_like, st), T=jnp.ones_like(st.T))
    assert AC._wnorm2(one, wts) == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(ValueError, match="uvnode"):
        AC.volume_weights(st, mesh, fields=("uvnode",))
