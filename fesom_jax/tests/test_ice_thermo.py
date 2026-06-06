"""Sea-ice thermodynamics gate — Phase 6, Task 6.2.

Verifies :mod:`fesom_jax.ice_thermo` (`therm_ice_cell`) against the C per-node thermo dump
(``data/ice_thermo_dump_core2/ice_thermo_dump_s1_rank0.txt``, written by
``port2/jobs/jax_ice_thermo_dump_core2.sh`` at config-A: thermo ON, EVP+FCT off, so the
thermo input a/m/snow == the cold-start IC and the kernel is ISOLATED). The C dump captures
the per-node INPUTS + OUTPUTS for all 126858 CORE2 nodes; JAX is fed the exact inputs and
matched output-for-output — a pure FP-reassociation MAP gate (thermo is per-node, no
scatter). The real CORE2 forcing exercises every regime (ice/open-water/melt/freeze), so
no synthetic reference is needed.

Also gates the two Phase-6 deliverables of this kernel: **runoff activation**
(``d(fw)/d(runoff) == 1`` — runoff enters ``prec``) and **AD-safety** (the masked-NaN probe
``d(Σehf)/d(SST)`` finite on every node incl. ice-free lanes).

SKIPS cleanly if the C dump is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "data" / "ice_thermo_dump_core2" / "ice_thermo_dump_s1_rank0.txt"

pytestmark = pytest.mark.skipif(
    not DUMP.exists(),
    reason="ice thermo dump missing (run port2/jobs/jax_ice_thermo_dump_core2.sh)",
)

# therm_ice_cell positional arg order (see ice_thermo.therm_ice_cell)
_ARG_ORDER = ["h", "hsn", "A", "fsh", "flo", "Ta", "qa", "rain", "snow", "runo",
              "rsss", "ug", "ustar", "T_oc", "S_oc", "ch", "ce", "t_in", "lid_clo"]
# (jax field name, dump column name, atol, rtol)  — calibrated MAP-class tolerances
_OUT_SPEC = [
    ("h",     "h_out",   1e-13, 1e-12),
    ("hsn",   "hsn_out", 1e-13, 1e-12),
    ("A",     "A_out",   1e-13, 1e-12),
    ("t",     "t_out",   1e-11, 1e-10),
    ("fw",    "fw",      1e-15, 1e-9),
    ("ehf",   "ehf",     1e-7,  1e-8),
    ("thdgr", "thdgr",   1e-15, 1e-9),
]


@pytest.fixture(scope="module")
def gate():
    """Load the C dump, run `therm_ice_cell` on its inputs, return (col dict, ThermoOut)."""
    import jax
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_thermo as it

    with open(DUMP) as f:
        names = f.readline().split("var=")[1].split()[0].split(",")
    data = np.loadtxt(DUMP)
    col = {nm: data[:, i] for i, nm in enumerate(names)}

    import jax.numpy as jnp
    cfg = IceConfig()
    args = [jnp.asarray(col[k], jnp.float64) for k in _ARG_ORDER]
    out = jax.vmap(lambda *a: it.therm_ice_cell(cfg, *a))(*args)
    return col, out, cfg


# --------------------------------------------------------------------------
# The forward MAP gate
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field,cname,atol,rtol", _OUT_SPEC)
def test_thermo_matches_c_dump(gate, field, cname, atol, rtol):
    """Each thermo output matches the C dump within MAP-class tolerance (all 126858 nodes)."""
    col, out, _cfg = gate
    j = np.asarray(getattr(out, field))
    c = col[cname]
    d = np.abs(j - c)
    ok = d <= atol + rtol * np.abs(c)
    nbad = int((~ok).sum())
    assert nbad == 0, (f"{field}: {nbad} nodes exceed tol (max|Δ|={d.max():.3e}, "
                       f"max rel={np.max(d/(np.abs(c)+1e-300)):.3e})")


def test_thermo_dump_is_meaningful(gate):
    """The dump actually exercises ice physics (so the gate isn't trivial)."""
    col, _out, _cfg = gate
    iced = col["A"] > 0.0
    assert iced.sum() > 5000, "too few ice nodes in the dump"
    assert (col["A"] == 0.0).sum() > 5000, "too few open-water nodes"
    # high-lat ice is supercooled-capped: ice nodes have T_oc near/below freezing
    assert np.median(col["T_oc"][iced]) < 0.5


# --------------------------------------------------------------------------
# Runoff activation (the Phase-6 reason this kernel matters)
# --------------------------------------------------------------------------
def test_runoff_activates(gate):
    """``d(fw)/d(runoff) == 1`` on every node — runoff enters ``prec = rain+runo+snow(1-A)``
    → ``fw``, the mechanism that activates river freshwater in Phase 6."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import ice_thermo as it
    col, _out, cfg = gate
    args = [jnp.asarray(col[k], jnp.float64) for k in _ARG_ORDER]
    ri = _ARG_ORDER.index("runo")

    def fw_sum(runo):
        a = list(args); a[ri] = runo
        return jnp.sum(jax.vmap(lambda *x: it.therm_ice_cell(cfg, *x))(*a).fw)

    g = np.asarray(jax.grad(fw_sum)(args[ri]))
    assert np.allclose(g, 1.0, atol=1e-12), f"d(fw)/d(runoff) deviates from 1: max|g-1|={np.max(np.abs(g-1)):.2e}"


# --------------------------------------------------------------------------
# AD-safety (the masked-NaN rule — finite everywhere incl. ice-free lanes)
# --------------------------------------------------------------------------
def test_ad_finite_all_nodes(gate):
    """``d(Σ(ehf+fw))/d(SST)`` is finite on every node — no ``0·inf`` masked-NaN from the
    thermo divides/sqrts (tfrez √S³, con/hice, /rsss, the Newton /A3), incl. open-water."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import ice_thermo as it
    col, _out, cfg = gate
    args = [jnp.asarray(col[k], jnp.float64) for k in _ARG_ORDER]
    ti = _ARG_ORDER.index("T_oc")

    def loss(toc):
        a = list(args); a[ti] = toc
        o = jax.vmap(lambda *x: it.therm_ice_cell(cfg, *x))(*a)
        return jnp.sum(o.ehf) + jnp.sum(o.fw)

    g = np.asarray(jax.grad(loss)(args[ti]))
    assert np.all(np.isfinite(g)), f"{int((~np.isfinite(g)).sum())} non-finite d/d(SST)"
    assert np.any(g != 0.0)        # the seam is live


def test_ad_finite_wrt_skin_temp(gate):
    """``d(Σehf)/d(t_skin)`` finite everywhere — the 5-iter Newton + albedo kink are AD-safe."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import ice_thermo as it
    col, _out, cfg = gate
    args = [jnp.asarray(col[k], jnp.float64) for k in _ARG_ORDER]
    si = _ARG_ORDER.index("t_in")

    def loss(ts):
        a = list(args); a[si] = ts
        return jnp.sum(jax.vmap(lambda *x: it.therm_ice_cell(cfg, *x))(*a).ehf)

    g = np.asarray(jax.grad(loss)(args[si]))
    assert np.all(np.isfinite(g))


def test_ad_vs_fd_ehf_sst(gate):
    """Quantitative FD↔AD for ``d(Σehf)/d(SST)`` on a SMOOTH interior-ice subset (deep ice,
    snow present, skin temp off the t=0 clamp, SST off freezing → every max/min kink
    inactive). The thermo is near-linear in SST through obudget + o2ihf there, so the
    directional FD↔AD plateaus at large h (rel ~1e-16) — the per-task gradient-magnitude
    check (the assembled-model FD↔AD is the GATE-6 task)."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import ice_thermo as it
    col, _out, cfg = gate
    args = [jnp.asarray(col[k], jnp.float64) for k in _ARG_ORDER]
    ti = _ARG_ORDER.index("T_oc")

    tfrez = (-0.0575 * col["S_oc"] + 1.7105e-3 * np.sqrt(col["S_oc"] ** 3)
             - 2.155e-4 * col["S_oc"] ** 2)
    sub = ((col["A_out"] > 0.7) & (col["h_out"] > 1.0) & (col["hsn_out"] > 0.05)
           & (col["t_out"] < -1.0) & (np.abs(col["T_oc"] - tfrez) > 0.2))
    assert sub.sum() > 500, "smooth ice subset too small"
    d = jnp.asarray(sub.astype(np.float64))

    def loss(toc):
        a = list(args); a[ti] = toc
        o = jax.vmap(lambda *x: it.therm_ice_cell(cfg, *x))(*a)
        return jnp.sum(jnp.where(d > 0, o.ehf, 0.0))

    ad = float(jax.grad(loss)(args[ti]) @ d)
    base = args[ti]
    rels = []
    for h in (1e-1, 1e-2, 1e-3):
        fd = float((loss(base + h * d) - loss(base - h * d)) / (2 * h))
        rels.append(abs(ad - fd) / (abs(fd) + 1e-30))
    assert min(rels) < 1e-6, f"FD↔AD plateau {min(rels):.3e} (AD={ad:.6e})"
