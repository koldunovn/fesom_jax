"""Named derivative cuts (``AdjointConfig.diag_cuts``), presets and ``FREEZE_FLAGS``.

pi mesh, linfs, PP mixing. The cuts ``mix.kv_out`` / ``mix.av_out`` sit on the mixing
coefficients after their halo exchange; PP's Kv and Av depend on the state (N² and shear),
so the cuts are live from a non-rest state: two steps from a perturbed T/uv initial state,
as in ``test_ssh_implicit.py``.

Properties:

* an empty cut set traces the identical program (jaxpr text) as ``adjoint_cfg=None``;
* every cut leaves the forward bitwise;
* each cut changes the one-step VJP, and the pair equals ``freeze_mixing`` bitwise (state
  and params cotangents) — the pair and the flag put ``stop_gradient`` on the same values;
* ``validate()`` coerces and rejects; ``dcut`` rejects an unregistered call-site name.

``differentiate_at`` (no mesh): value of ``fn(args)``, derivative at ``alt``; plain autodiff
when ``alt == args``; transpose-consistent; inside ``jax.checkpoint`` and ``lax.scan``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import adjoint_stab, forcing, ic, ssh
from fesom_jax import step as stepmod
from fesom_jax.adjoint_stab import (FREEZE_FLAGS, KNOWN_CUTS, AdjointConfig, dcut,
                                    differentiate_at)
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params
from fesom_jax.tests import _ssh_implicit_common as C

DT = 100.0
PHASE0_CUTS = {"mix.kv_out", "mix.av_out"}


@pytest.fixture(scope="module")
def setup():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    mesh = load_mesh()
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress = forcing.surface_stress(mesh)
    st = ic.initial_state(mesh)
    st = dataclasses.replace(st, T=st.T + C.node3d(mesh, 0.5, 0),
                             uv=st.uv + C.elem_uv(mesh, 0.02, 0))
    st = jax.jit(_step_fn(None, first=True))(st, Params.defaults(), mesh, op, stress)
    st = jax.jit(_step_fn(None))(st, Params.defaults(), mesh, op, stress)
    assert float(jnp.max(jnp.abs(st.uv))) > 0.0
    return mesh, op, stress, st


def _step_fn(cfg, first=False):
    def f(st, p, mesh, op, stress):
        return stepmod.step(st, mesh, op, stress, p, dt=DT, is_first_step=first,
                            adjoint_cfg=cfg)
    return f


def _jaxpr_text(setup, cfg):
    mesh, op, stress, st = setup
    return str(jax.make_jaxpr(_step_fn(cfg))(st, Params.defaults(), mesh, op, stress))


IN_FIELDS = ("T", "S", "uv", "eta_n")


def _cotangent(mesh):
    return {"T": C.node3d(mesh, 1.0, 0), "S": C.node3d(mesh, 10.0, 1),
            "uv": C.elem_uv(mesh, 50.0, 0), "eta_n": C.node2d(mesh, 20.0, 1)}


_VJP_CACHE: dict = {}


def _one_step_vjp(setup, cfg):
    """(state cotangent dict, params cotangent) of <y, step(state, params)>; one compile per
    distinct cfg (cached)."""
    key = cfg if cfg is None else cfg.validate()
    if key in _VJP_CACHE:
        return _VJP_CACHE[key]
    mesh, op, stress, st = setup
    fn = _step_fn(cfg)

    def run(xin, p, st, y, mesh, op, stress):
        def f(xin, p):
            out = fn(dataclasses.replace(st, **xin), p, mesh, op, stress)
            return {k: getattr(out, k) for k in y}
        _, vjp = jax.vjp(f, xin, p)
        return vjp(y)

    xin = {k: getattr(st, k) for k in IN_FIELDS}
    ct_x, ct_p = jax.jit(run)(xin, Params.defaults(), st, _cotangent(mesh), mesh, op, stress)
    res = (jax.device_get(ct_x), jax.device_get(ct_p))
    _VJP_CACHE[key] = res
    return res


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def _bitwise(a, b):
    la, lb = _leaves(a), _leaves(b)
    return len(la) == len(lb) and all(np.array_equal(x, y) for x, y in zip(la, lb))


def _rel_change(a, b):
    num = sum(float(np.sum((x - y) ** 2)) for x, y in zip(_leaves(a), _leaves(b)))
    den = sum(float(np.sum(x ** 2)) for x in _leaves(a))
    return np.sqrt(num / max(den, 1e-300))


# --------------------------------------------------------------------------
# config object: fields, validate, presets, describe
# --------------------------------------------------------------------------
def test_freeze_flags_and_known_cuts():
    from fesom_jax.adjoint_stab import TKE_CANDIDATE_FIELDS
    assert FREEZE_FLAGS == tuple(f for f in AdjointConfig._fields
                                 if f != "diag_cuts" and f not in TKE_CANDIDATE_FIELDS)
    assert all(f.startswith("freeze_") for f in FREEZE_FLAGS)
    assert all(getattr(AdjointConfig(), f) is False for f in FREEZE_FLAGS)
    assert all(getattr(AdjointConfig(), f) is None for f in TKE_CANDIDATE_FIELDS)
    assert AdjointConfig().diag_cuts == frozenset()
    assert set(KNOWN_CUTS) >= PHASE0_CUTS
    # cut names never duplicate a flag
    assert not {c.split(".")[-1] for c in KNOWN_CUTS} & {f[7:] for f in FREEZE_FLAGS}


def test_validate_coerces_and_rejects():
    for given in ({"mix.kv_out"}, ["mix.kv_out"], ("mix.kv_out",), frozenset({"mix.kv_out"})):
        v = AdjointConfig(diag_cuts=given).validate()
        assert type(v.diag_cuts) is frozenset and v.diag_cuts == {"mix.kv_out"}
        hash(v)                                   # static-hashable after validate
    d = AdjointConfig()
    assert d.validate() is d
    with pytest.raises(ValueError, match="mix.kv_typo"):
        AdjointConfig(diag_cuts={"mix.kv_typo"}).validate()
    with pytest.raises(TypeError):
        AdjointConfig(diag_cuts="mix.kv_out").validate()       # a string is not a set
    with pytest.raises(TypeError):
        AdjointConfig(diag_cuts={1}).validate()
    # _replace bypasses any constructor: validate still catches it
    with pytest.raises(ValueError):
        AdjointConfig()._replace(diag_cuts={"nope"}).validate()


def test_presets_and_describe():
    assert AdjointConfig.exact() == AdjointConfig()
    lw = AdjointConfig.long_window()
    on = {f for f in FREEZE_FLAGS if getattr(lw, f)}
    assert on == {"freeze_mixing", "freeze_gm_slope"} and lw.diag_cuts == frozenset()
    assert AdjointConfig().describe() == "none" and not AdjointConfig().active
    c = AdjointConfig(diag_cuts=frozenset({"mix.kv_out", "mix.av_out"}))
    assert c.active and c.describe() == "cuts=mix.av_out,mix.kv_out"
    c2 = AdjointConfig(freeze_mixing=True, diag_cuts=frozenset({"mix.kv_out"}))
    assert c2.describe() == "mixing+cuts=mix.kv_out"
    assert lw.describe() == "mixing+gm_slope"
    lt = AdjointConfig.long_window_tke()
    assert not any(getattr(lt, f) for f in FREEZE_FLAGS) and lt.diag_cuts == frozenset()
    assert (lt.tke_pr_cut, lt.tke_pr_ratio, lt.tke_pr_raw_cut, lt.tke_ml_cut) == \
        (1e-8, 100.0, -1e-6, 1e-8)
    assert lt.validate() == lt and lt.active
    lc = AdjointConfig.long_window_combined()
    assert {f for f in FREEZE_FLAGS if getattr(lc, f)} == {"freeze_fct_limiter"}
    assert (lc.tke_ml_ratio, lc.tke_ml_raw_cut) == (100.0, -1e-6)
    assert lc._replace(freeze_fct_limiter=False, tke_ml_ratio=None, tke_ml_raw_cut=None) == lt
    assert lc.validate() == lc and lc.active
    yc = AdjointConfig.year_window_combined()
    assert yc._replace(tke_pr_cut=1e-8, tke_pr_lat_cut=None) == lc
    assert (yc.tke_pr_cut, yc.tke_pr_lat_cut, yc.tke_ml_cut) == (1e-6, 70.0, 1e-8)
    assert yc.validate() == yc
    opt = lc.tke_options()
    assert (opt.ml_n2_star, opt.ml_ratio, opt.ml_raw_cut) == (1e-8, 100.0, -1e-6)


def test_dcut_primitive():
    x = {"a": jnp.arange(3.0), "i": jnp.arange(3)}
    cfg = AdjointConfig(diag_cuts=frozenset({"mix.kv_out"}))
    assert dcut(None, "mix.kv_out", x) is x
    assert dcut(AdjointConfig(), "mix.kv_out", x) is x
    assert dcut(cfg, "mix.av_out", x) is x
    with pytest.raises(ValueError, match="not registered"):
        dcut(cfg, "mix.kv_outt", x)
    with pytest.raises(ValueError, match="not registered"):
        dcut(None, "mix.kv_outt", x)          # a misspelled call site fails even when off
    g = jax.grad(lambda a: jnp.sum(dcut(cfg, "mix.kv_out", {"a": a, "i": x["i"]})["a"] * a))(
        x["a"])
    np.testing.assert_array_equal(np.asarray(g), np.asarray(x["a"]))   # only the uncut factor
    y = dcut(cfg, "mix.kv_out", x)
    assert y["i"] is x["i"]                   # integer leaves untouched


def test_gm_n2_cut_config_and_mask():
    g = AdjointConfig(gm_n2_ratio=100.0, gm_n2_raw_cut=-1e-6).validate()
    assert g.gm_n2_cut and g.active and not AdjointConfig().gm_n2_cut
    assert g.describe() == "gm_n2_ratio=100.0+gm_n2_raw_cut=-1e-06"
    assert g.tke_options() is None            # a GM-input cut, nothing for the TKE core
    assert AdjointConfig(gm_n2_raw_cut=0.0).validate().gm_n2_cut
    with pytest.raises(ValueError, match="freeze_gm_slope"):
        AdjointConfig(gm_n2_ratio=100.0, freeze_gm_slope=True).validate()
    with pytest.raises(ValueError, match="> 0"):
        AdjointConfig(gm_n2_ratio=0.0).validate()
    with pytest.raises(ValueError, match="finite"):
        AdjointConfig(gm_n2_raw_cut=float("inf")).validate()
    st = AdjointConfig(gm_slope_cut=2e-3).validate()
    assert st.active and not st.gm_n2_cut and st.describe() == "gm_slope_cut=0.002"
    with pytest.raises(ValueError, match="freeze_gm_slope"):
        AdjointConfig(gm_slope_cut=2e-3, freeze_gm_slope=True).validate()
    with pytest.raises(ValueError, match="> 0"):
        AdjointConfig(gm_slope_cut=-1.0).validate()
    assert {"gm.slope_n2", "gm.cm_n2", "gm.bvp_n2"} <= set(KNOWN_CUTS)
    # the mask: the cancellation residual (|raw| > R |smoothed|) or the low raw N², joined
    n2 = jnp.array([1.5e-7, 1e-5, 1e-5, 2e-4, 0.0])
    raw = jnp.array([-6.9e-5, 5e-5, -2e-6, 1.6e-4, 0.0])
    m = adjoint_stab.n2_cancel_mask
    assert m(n2, raw, 100.0, None).tolist() == [True, False, False, False, False]
    assert m(n2, raw, None, -1e-6).tolist() == [True, False, True, False, False]
    assert m(n2, raw, 100.0, -1e-6).tolist() == [True, False, True, False, False]
    assert m(n2, raw).tolist() == [False] * 5
    # cut_where with that mask: forward bitwise, derivative zero only in the masked cells
    mask = m(n2, raw, 100.0, -1e-6)
    f = lambda x: jnp.sum(adjoint_stab.cut_where(x, mask) ** 2)   # noqa: E731
    assert float(f(n2)) == float(jnp.sum(n2 ** 2))
    np.testing.assert_array_equal(np.asarray(jax.grad(f)(n2)),
                                  np.where(np.asarray(mask), 0.0, 2.0 * np.asarray(n2)))


# --------------------------------------------------------------------------
# traced program / forward identity
# --------------------------------------------------------------------------
def test_empty_cut_set_traces_the_identical_program(setup):
    ref = _jaxpr_text(setup, None)
    assert _jaxpr_text(setup, AdjointConfig()) == ref
    assert _jaxpr_text(setup, AdjointConfig(diag_cuts=set())) == ref
    assert _jaxpr_text(setup, AdjointConfig.exact()) == ref
    # control: a live cut does change the traced program (one stop_gradient more)
    cut = _jaxpr_text(setup, AdjointConfig(diag_cuts=frozenset({"mix.kv_out"})))
    assert cut != ref
    assert cut.count("stop_gradient") == ref.count("stop_gradient") + 1
    # a plain set is coerced inside step: same program as the frozenset
    assert _jaxpr_text(setup, AdjointConfig(diag_cuts={"mix.kv_out"})) == cut


def test_step_rejects_unknown_cut(setup):
    with pytest.raises(ValueError, match="unknown AdjointConfig.diag_cuts"):
        _jaxpr_text(setup, AdjointConfig(diag_cuts={"mix.bogus"}))


def test_all_cuts_leave_the_forward_bitwise(setup):
    mesh, op, stress, st = setup
    p = Params.defaults()
    ref = jax.jit(_step_fn(None))(st, p, mesh, op, stress)
    got = jax.jit(_step_fn(AdjointConfig(diag_cuts=frozenset(KNOWN_CUTS))))(
        st, p, mesh, op, stress)
    for f in dataclasses.fields(ref):
        a, b = np.asarray(getattr(ref, f.name)), np.asarray(getattr(got, f.name))
        assert np.array_equal(a, b), f"State.{f.name} moved under all diag_cuts"


# --------------------------------------------------------------------------
# effect on the one-step VJP
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cut", sorted(PHASE0_CUTS))
def test_each_cut_changes_the_one_step_vjp(setup, cut, capsys):
    base = _one_step_vjp(setup, None)
    got = _one_step_vjp(setup, AdjointConfig(diag_cuts=frozenset({cut})))
    assert all(np.all(np.isfinite(x)) for x in _leaves(got))
    rs, rp = _rel_change(base[0], got[0]), _rel_change(base[1], got[1])
    with capsys.disabled():
        print(f"\n  {cut}: one-step VJP relative change state {rs:.3e}, params {rp:.3e}")
    assert rs > 1e-12, f"{cut} did not change the state cotangent — is the cut point live?"
    assert rp > 1e-12, f"{cut} did not change the params cotangent"


def test_pair_of_cuts_equals_freeze_mixing_bitwise(setup):
    pair = _one_step_vjp(setup, AdjointConfig(diag_cuts=frozenset(PHASE0_CUTS)))
    frz = _one_step_vjp(setup, AdjointConfig(freeze_mixing=True))
    assert _bitwise(pair[0], frz[0]), "state cotangent: {kv_out, av_out} != freeze_mixing"
    assert _bitwise(pair[1], frz[1]), "params cotangent: {kv_out, av_out} != freeze_mixing"
    # control: freeze_mixing is not the identity on this fixture
    base = _one_step_vjp(setup, None)
    assert not _bitwise(base[0], frz[0])


# --------------------------------------------------------------------------
# differentiate_at
# --------------------------------------------------------------------------
def _f(x, y):
    """A module-level nonlinear map R^5 x R^5 -> R^12."""
    return jnp.concatenate([jnp.sin(x) * y ** 2, jnp.exp(0.3 * x * y),
                            jnp.tanh(x)[:2] * jnp.sum(y)])


def _sq(x):
    return x ** 2


def _g(c):
    """One 'step' of a small nonlinear recurrence (for the scan test)."""
    return jnp.tanh(c) + 0.5 * jnp.roll(c, 1) ** 2


def _alt_of(c):
    return 0.7 * c + 0.1


@jax.custom_jvp
def _planted_at_args(x, alt):
    """Negative control: a rule that (wrongly) linearises at ``args``, not at ``alt``."""
    return _sq(x)


@_planted_at_args.defjvp
def _planted_jvp(primals, tangents):
    x, _alt = primals
    dx, _ = tangents
    return jax.jvp(_sq, (x,), (dx,))


XY = (jnp.linspace(-1.0, 1.3, 5), jnp.linspace(0.4, 2.0, 5))
ALT = (jnp.linspace(0.2, -0.9, 5), jnp.linspace(1.5, 0.1, 5))
W = jnp.linspace(-2.0, 3.0, 12)


def _vdot_tree(a, b):
    return sum(float(jnp.vdot(u, v)) for u, v in
               zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def test_differentiate_at_self_is_plain_autodiff(capsys):
    def at_self(x, y):
        return differentiate_at(_f, (x, y), (x, y))
    v0, vjp0 = jax.vjp(_f, *XY)
    v1, vjp1 = jax.vjp(at_self, *XY)
    assert np.array_equal(np.asarray(v0), np.asarray(v1))
    g0, g1 = vjp0(W), vjp1(W)
    rel = max(float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(a))) for a, b in zip(g0, g1))
    with capsys.disabled():
        print(f"\n  differentiate_at(alt=args) vs autodiff VJP: max rel {rel:.2e}"
              f" ({'bitwise' if rel == 0 else 'not bitwise'})")
    assert rel <= 1e-15
    t = (jnp.ones(5), jnp.linspace(0.0, 1.0, 5))
    np.testing.assert_allclose(np.asarray(jax.jvp(at_self, XY, t)[1]),
                               np.asarray(jax.jvp(_f, XY, t)[1]), rtol=1e-15, atol=0)


def _analytic_error(fn):
    x = jnp.array([1.0, -2.0, 3.0])
    dx = jnp.array([0.5, 1.0, -0.25])
    val, tan = jax.jvp(lambda x: fn(x, 2.0 * x), (x,), (dx,))
    grad = jax.grad(lambda x: jnp.sum(fn(x, 2.0 * x) * dx))(x)
    np.testing.assert_array_equal(np.asarray(val), np.asarray(x ** 2))   # value at args
    want = 4.0 * x * dx                                                  # 2·alt·dx, alt = 2x
    return max(float(jnp.max(jnp.abs(tan - want))), float(jnp.max(jnp.abs(grad - want))))


def test_differentiate_at_analytic_square():
    """f(x) = x², alt = 2x ⇒ tangent 2·alt·dx = 4x·dx (and the same cotangent), value x²."""
    assert _analytic_error(lambda x, a: differentiate_at(_sq, (x,), (a,))) <= 1e-15


def test_differentiate_at_analytic_negative_control():
    """The analytic check must see a rule that linearises at args instead of alt."""
    assert _analytic_error(_planted_at_args) > 1.0


def test_differentiate_at_dot_test():
    """<w, J(alt)·v> == <J(alt)ᵀ·w, v> to 1e-14 (alt ≠ args); a VJP taken at args instead
    (a wrong transpose) fails the same test."""
    def h(x, y):
        return differentiate_at(_f, (x, y), ALT)
    v = (jnp.linspace(0.3, -0.6, 5), jnp.linspace(-1.0, 0.5, 5))
    jv = jax.jvp(h, XY, v)[1]
    jtw = jax.vjp(h, *XY)[1](W)
    lhs, rhs = float(jnp.vdot(W, jv)), _vdot_tree(jtw, v)
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    assert rel <= 1e-14, f"dot test rel {rel:.2e}"
    wrong = jax.vjp(_f, *XY)[1](W)
    rel_wrong = abs(lhs - _vdot_tree(wrong, v)) / max(abs(lhs), abs(rhs))
    assert rel_wrong > 1e-3, f"the dot test cannot see a wrong transpose ({rel_wrong:.2e})"
    # alt gets no derivative of its own
    ga = jax.grad(lambda a0, a1: jnp.vdot(W, differentiate_at(_f, XY, (a0, a1))),
                  argnums=(0, 1))(*ALT)
    assert all(np.array_equal(np.asarray(g), np.zeros(5)) for g in ga)


def test_differentiate_at_inside_checkpoint_and_scan():
    """Gradient through a checkpointed lax.scan whose body linearises at ``_alt_of(c)``
    equals the product of the Jacobians of ``_g`` at those points (the definition)."""
    n = 4
    c0 = jnp.linspace(-0.8, 0.9, 6)
    w = jnp.linspace(1.0, -1.0, 6)

    def body(c, _):
        return differentiate_at(_g, (c,), (_alt_of(c),)), None

    def run(c0, remat):
        b = jax.checkpoint(body) if remat else body
        return jax.lax.scan(b, c0, None, length=n)[0]

    # reference: forward states, then Jᵀ at the alt points in reverse
    cs = [c0]
    for _ in range(n):
        cs.append(_g(cs[-1]))
    ct = w
    for k in reversed(range(n)):
        ct = jax.jacobian(_g)(_alt_of(cs[k])).T @ ct
    for remat in (False, True):
        fin = jax.jit(lambda c: run(c, remat))(c0)
        np.testing.assert_array_equal(np.asarray(fin), np.asarray(cs[-1]))
        g = jax.jit(jax.grad(lambda c: jnp.vdot(w, run(c, remat))))(c0)
        np.testing.assert_allclose(np.asarray(g), np.asarray(ct), rtol=1e-14, atol=1e-15)
        # forward mode through the same program
        dc = jnp.linspace(0.1, 0.6, 6)
        tan = jax.jvp(lambda c: run(c, remat), (c0,), (dc,))[1]
        ref = dc
        for k in range(n):
            ref = jax.jacobian(_g)(_alt_of(cs[k])) @ ref
        np.testing.assert_allclose(np.asarray(tan), np.asarray(ref), rtol=1e-14, atol=1e-15)


def test_differentiate_at_rejects_mismatched_alt():
    x = jnp.ones(3)
    with pytest.raises(ValueError, match="structure"):
        differentiate_at(_f, (x, x), (x,))
    with pytest.raises(ValueError, match="does not match"):
        differentiate_at(_sq, (x,), (jnp.ones(4),))
    with pytest.raises(ValueError, match="does not match"):
        differentiate_at(_sq, (x,), (jnp.ones(3, jnp.int32),))
    # integer leaves are fine (float0 tangents)
    g = jax.grad(lambda x: jnp.sum(differentiate_at(lambda a, n: a * n, (x, jnp.arange(3)),
                                                    (x + 1.0, jnp.arange(3)))))(x)
    np.testing.assert_array_equal(np.asarray(g), np.arange(3.0))
