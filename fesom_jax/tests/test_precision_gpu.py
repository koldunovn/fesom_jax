"""Single precision on a GPU (``pytest -m gpu``; skipped without one).

XLA:GPU with its default ``xla_allow_excess_precision=true`` folds f64→f32→f64 convert pairs, which removes the
float32 roundings the rounding carry measures (job 28288484 on A100, fesom3-jax job 28274339 on GH200); XLA:CPU keeps
them, so only a GPU test can see it. Checked here: the decisive kernels of ``scripts/perf/sp_convert_pairs.py`` with
the production option (exact) and without it (the negative control: folded), and the content conservation of the
ALE reconstruction with the carry, jitted as the model jits it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import precision as PR

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.gpu


def _kernels():
    spec = importlib.util.spec_from_file_location("sp_convert_pairs", ROOT / "scripts" / "perf" / "sp_convert_pairs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_convert_pairs_kept_with_the_option_and_folded_without():
    res = _kernels().run_variants(["a", "b"])["variants"]
    for kernel in res["b"].values():                       # the production option: exact
        for key, r in kernel.items():
            if key != "hlo":
                assert r["bitwise"], (key, r)
    a = res["a"]                                            # negative control: the fold is still there
    assert a["feed"]["feed_y"]["zeros"] == a["feed"]["feed_y"]["ref_zeros"] + 4096, a["feed"]


def test_carry_conserves_on_the_gpu():
    from fesom_jax import tracer_adv
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh()
    n, nl = mesh.nod2D, mesh.nl
    rng = np.random.default_rng(3)
    m = np.asarray(mesh.node_layer_mask)
    T = rng.uniform(-2, 30, (n, nl)).astype(np.float32)
    h = np.where(m, rng.uniform(5.0, 250.0, (n, nl)), 0.0)
    hn = np.where(m, h * rng.uniform(0.999, 1.001, (n, nl)), 0.0)
    d = rng.uniform(-1.0, 1.0, (n, nl)) * 1e-2 * h
    c0 = np.zeros((n, nl), np.float32)
    mesh32 = PR.cast_tree(mesh)
    args = (jnp.asarray(T), jnp.asarray(d, jnp.float64), jnp.asarray(h, jnp.float64),
            jnp.asarray(hn, jnp.float64), jnp.asarray(c0))

    def f(T, d, h, hn, c):
        return tracer_adv.ale_reconstruct_carry(mesh32, T, d, h, hn, c)

    before = np.where(m, T.astype(np.float64) * h + d, 0.0).sum()

    def err(fn):
        with PR.scope("single"):
            Tn, cn = fn(*args)
        after = np.where(m, np.asarray(Tn, np.float64) * hn + np.asarray(cn, np.float64), 0.0).sum()
        return abs(after - before) / abs(before), np.asarray(cn)

    e_opt, c_opt = err(PR.jit(f, "single"))
    assert e_opt <= 1e-13
    assert np.count_nonzero(c_opt) > 0.5 * m.sum()       # the residual is really carried
    e_def, c_def = err(jax.jit(f))                         # default flags: the carry may fold to zero
    assert np.count_nonzero(c_def) < np.count_nonzero(c_opt) or e_def > 10 * e_opt
