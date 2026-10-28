"""TKE derivative cut points on the assembled CORE2 step (zstar + TKE + GM, JRA55 1958,
seeded tke = 1e-3 > tke_min, one step).

* an empty cut set traces the identical step program as ``adjoint_cfg=None``; the full
  ``tke.*`` set adds exactly the expected cut operations (so every cut site is on this path);
* every known cut leaves the one-step State bitwise (CPU);
* ``{mix.kv_out, mix.av_out} + freeze_tke_state`` gives the one-step VJP (state and params)
  of ``freeze_mixing + freeze_tke_state`` bitwise on CPU;
* ``tke.km_e`` and ``tke.deep`` change the one-step VJP (live on CORE2).

The column-level effect of every single cut is in ``test_tke_cuts.py`` (unit group).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ssh, surface_forcing
from fesom_jax import step as stepmod
from fesom_jax.adjoint_stab import KNOWN_CUTS, AdjointConfig
from fesom_jax.ale import AleConfig
from fesom_jax.gm import GMConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0
YEAR = 1958
TKE_SEED = 1.0e-3
TKE_CUTS = frozenset(c for c in KNOWN_CUTS if c.startswith("tke."))
MIX_PAIR = frozenset({"mix.kv_out", "mix.av_out"})
IN_FIELDS = ("T", "S", "uv", "eta_n", "tke")

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file()),
    reason="CORE2 mesh / PHC IC missing")


@pytest.fixture(scope="module")
def core2():
    mesh = load_mesh(MESH_DIR)
    state = phc_initial_state(mesh, IC_DIR)
    imask = np.asarray(mesh.node_iface_mask)
    state = dataclasses.replace(
        state, tke=jnp.asarray(np.where(imask, TKE_SEED, 0.0), jnp.float64))
    assert float(jnp.max(state.tke)) > TkeConfig().tke_min
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    sf = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
    return dict(mesh=mesh, state=state, op=op, sf=sf, fs=cf.static)


def _step_fn(cfg):
    def f(st, p, mesh, op, sf, fs):
        return stepmod.step(st, mesh, op, None, p, dt=DT, is_first_step=True,
                            step_forcing=sf, forcing_static=fs, ale_cfg=AleConfig(),
                            gm_cfg=GMConfig(), tke_cfg=TkeConfig(), adjoint_cfg=cfg)
    return f


def _args(c):
    return (c["state"], Params.defaults(), c["mesh"], c["op"], c["sf"], c["fs"])


def _jaxpr_text(c, cfg):
    return str(jax.make_jaxpr(_step_fn(cfg))(*_args(c)))


def test_empty_cut_set_traces_the_identical_step(core2):
    ref = _jaxpr_text(core2, None)
    assert _jaxpr_text(core2, AdjointConfig()) == ref
    assert _jaxpr_text(core2, AdjointConfig(diag_cuts=frozenset())) == ref
    # control + reach: the eleven stop_gradient cut sites of the column core, the masked
    # tke.deep cut (a custom_jvp on each of the three outputs) and the stop_gradient on the
    # thicknesses its depth mask is built from, and the masked tke.ml_wall and tke.kh_floor
    # cuts (one custom_jvp each, their masks comparisons) all sit on the traced CORE2 step
    cut = _jaxpr_text(core2, AdjointConfig(diag_cuts=TKE_CUTS))
    n_sg = cut.count("stop_gradient") - ref.count("stop_gradient")
    n_cj = cut.count("custom_jvp_call") - ref.count("custom_jvp_call")
    print(f"\nall tke cuts: +{n_sg} stop_gradient, +{n_cj} custom_jvp_call")
    masked = {"tke.deep", "tke.ml_wall", "tke.kh_floor"}
    assert n_sg == (len(TKE_CUTS) - len(masked)) + 1, n_sg
    assert n_cj == 3 + 1 + 1, n_cj      # tke.deep on tke_new, KappaM, KappaH; ml_wall; kh_floor


def test_all_cuts_leave_the_core2_step_bitwise(core2):
    ref = jax.jit(_step_fn(None))(*_args(core2))
    got = jax.jit(_step_fn(AdjointConfig(diag_cuts=frozenset(KNOWN_CUTS))))(*_args(core2))
    for f in dataclasses.fields(ref):
        a, b = getattr(ref, f.name), getattr(got, f.name)
        if a is None:
            assert b is None
            continue
        for u, v in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            assert np.array_equal(np.asarray(u), np.asarray(v)), \
                f"State.{f.name} moved under all diag_cuts"
    assert float(jnp.max(ref.tke)) > 0.0


_VJP: dict = {}


def _cotangent(c):
    rng = np.random.default_rng(5)
    st = c["state"]
    scale = {"T": 1.0, "S": 10.0, "uv": 50.0, "eta_n": 20.0, "tke": 1e3}
    return {k: jnp.asarray(rng.standard_normal(np.shape(getattr(st, k))) * scale[k])
            for k in IN_FIELDS}


def _one_step_vjp(c, cfg):
    key = cfg.validate()
    if key in _VJP:
        return _VJP[key]
    fn = _step_fn(cfg)

    def run(xin, p, st, y, mesh, op, sf, fs):
        def f(xin, p):
            out = fn(dataclasses.replace(st, **xin), p, mesh, op, sf, fs)
            return {k: getattr(out, k) for k in y}
        _, vjp = jax.vjp(f, xin, p)
        return vjp(y)

    st = c["state"]
    xin = {k: getattr(st, k) for k in IN_FIELDS}
    res = jax.device_get(jax.jit(run)(xin, Params.defaults(), st, _cotangent(c), c["mesh"],
                                      c["op"], c["sf"], c["fs"]))
    _VJP[key] = res
    return res


def _leaves(t):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(t)]


def _bitwise(a, b):
    la, lb = _leaves(a), _leaves(b)
    return len(la) == len(lb) and all(np.array_equal(x, y) for x, y in zip(la, lb))


def _rel_change(a, b):
    num = sum(float(np.sum((x - y) ** 2)) for x, y in zip(_leaves(a), _leaves(b)))
    den = sum(float(np.sum(x ** 2)) for x in _leaves(a))
    return np.sqrt(num / max(den, 1e-300))


def test_mix_pair_plus_tke_state_equals_freeze_mixing_plus_tke_state(core2):
    pair = _one_step_vjp(core2, AdjointConfig(freeze_tke_state=True, diag_cuts=MIX_PAIR))
    frz = _one_step_vjp(core2, AdjointConfig(freeze_mixing=True, freeze_tke_state=True))
    assert all(np.all(np.isfinite(x)) for x in _leaves(pair))
    assert _bitwise(pair[0], frz[0]), "state cotangent differs"
    assert _bitwise(pair[1], frz[1]), "params cotangent differs"
    # control: the configuration is not the plain tke-state cut
    base = _one_step_vjp(core2, AdjointConfig(freeze_tke_state=True))
    assert not _bitwise(base[0], frz[0])


# cut -> does it change the one-step PARAMS cotangent? tke.km_e cuts only the path to the
# incoming tke (c_k multiplies sqrt(tke), whose derivative is cut, not c_k's own partial), and
# over one step no parameter reaches the incoming tke: the params cotangent is unchanged
# (measured exactly 0). tke.deep cuts the outputs' dependence on c_k etc. below 500 m.
CORE2_EFFECT = {"tke.km_e": False, "tke.deep": True}


@pytest.mark.parametrize("cut", sorted(CORE2_EFFECT))
def test_tke_cut_changes_the_core2_vjp(core2, cut, capsys):
    base = _one_step_vjp(core2, AdjointConfig())
    got = _one_step_vjp(core2, AdjointConfig(diag_cuts=frozenset({cut})))
    assert all(np.all(np.isfinite(x)) for x in _leaves(got))
    rs, rp = _rel_change(base[0], got[0]), _rel_change(base[1], got[1])
    with capsys.disabled():
        print(f"\n  CORE2 {cut}: one-step VJP relative change state {rs:.3e}, params {rp:.3e}")
    assert rs > 1e-12, f"{cut} did not change the CORE2 state cotangent"
    if CORE2_EFFECT[cut]:
        assert rp > 1e-12, f"{cut} did not change the CORE2 params cotangent"
    else:
        assert rp < 1e-12, f"{cut} changed the one-step params cotangent ({rp:.3e})"
