"""Sea-ice FCT advection gate — Phase 6, Task 6.5.

Verifies :mod:`fesom_jax.ice_adv` against the isolated C FCT dump
(``data/ice_full_dump_core2/ice_fct_dump_s1_rank0.txt``, written by the
``FESOM_ICE_FCT_DUMP_DIR`` hook at config-C full-ice). The dump records, per all-node at
step 1, the advecting velocity ``u_ice``/``v_ice`` + the 3 ice tracers BEFORE and AFTER
``fct_solve`` (before cut_off/thermo). JAX is fed the C inputs and matched output-for-output
— the element-based FCT (no CSR) reproduces the C to machine precision.

SKIPS cleanly if the mesh / FCT dump are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
FCTD = ROOT / "data" / "ice_full_dump_core2" / "ice_fct_dump_s1_rank0.txt"

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.exists() and FCTD.exists()),
    reason="CORE2 mesh / ice FCT dump missing (run port2/jobs/jax_ice_full_dump_core2.sh)",
)


@pytest.fixture(scope="module")
def fct():
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_adv
    from fesom_jax.mesh import load_mesh
    mesh = load_mesh(MESH_DIR)
    cfg = IceConfig()
    with open(FCTD) as f:
        names = f.readline().split("var=")[1].split()[0].split(",")
    d = np.loadtxt(FCTD)
    col = {n: d[:, i] for i, n in enumerate(names)}
    a_out, m_out, ms_out = ice_adv.fct_solve(
        cfg, mesh, jnp.asarray(col["a_in"]), jnp.asarray(col["m_in"]),
        jnp.asarray(col["ms_in"]), jnp.asarray(col["u_ice"]), jnp.asarray(col["v_ice"]))
    return mesh, col, (np.asarray(a_out), np.asarray(m_out), np.asarray(ms_out))


@pytest.mark.parametrize("field,out_i", [("a", 0), ("m", 1), ("ms", 2)])
def test_fct_matches_dump(fct, field, out_i):
    """Each advected tracer matches the C FCT output to machine precision (element-based)."""
    _mesh, col, outs = fct
    j = outs[out_i]
    c = col[f"{field}_out"]
    d = np.abs(j - c)
    assert d.max() < 1e-12, f"{field}: max|Δ|={d.max():.3e} rel={d.max()/(np.abs(c).max()+1e-30):.3e}"


def test_fct_is_active(fct):
    """The dump exercises real advection (the FCT moved the tracers — the gate isn't trivial)."""
    _mesh, col, _outs = fct
    assert np.abs(col["a_out"] - col["a_in"]).max() > 1e-4
    assert np.abs(col["m_out"] - col["m_in"]).max() > 1e-4


def test_fct_bounded(fct):
    """The Zalesak limiter keeps the solution BOUNDED — only a small antidiffusive overshoot
    past the input range (the C does the same; cut_off clamps a≤1 afterward). Without the
    limiter the high-order MFCT would overshoot far more. The exact overshoot matches the C
    (the dump gate), so this just confirms boundedness, not a tight bound."""
    _mesh, col, outs = fct
    a_out = outs[0]
    over = a_out.max() - col["a_in"].max()
    under = col["a_in"].min() - a_out.min()
    assert over < 0.05 and under < 0.05, f"overshoot {over:.4f} / undershoot {under:.4f}"
    # and it equals the C overshoot (JAX == C, already gated tight)
    assert abs((a_out.max() - col["a_out"].max())) < 1e-12


def test_fct_ad_finite(fct):
    """``d(Σm_out)/d(m_in)`` and ``d(·)/d(u_ice)`` finite everywhere — the limiter ratio floor
    (1e-12) keeps the min/max/where VJP NaN-safe; no division by a tracer value."""
    import jax
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_adv
    mesh, col, _outs = fct
    cfg = IceConfig()
    a_in = jnp.asarray(col["a_in"]); m_in = jnp.asarray(col["m_in"])
    ms_in = jnp.asarray(col["ms_in"]); u = jnp.asarray(col["u_ice"]); v = jnp.asarray(col["v_ice"])

    def loss_m(mm):
        _a, mo, _ms = ice_adv.fct_solve(cfg, mesh, a_in, mm, ms_in, u, v)
        return jnp.sum(mo)
    g = np.asarray(jax.grad(loss_m)(m_in))
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)

    def loss_u(uu):
        ao, mo, mso = ice_adv.fct_solve(cfg, mesh, a_in, m_in, ms_in, uu, v)
        return jnp.sum(ao + mo + mso)
    g2 = np.asarray(jax.grad(loss_u)(u))
    assert np.all(np.isfinite(g2)) and np.any(g2 != 0.0)
