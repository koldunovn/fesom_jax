"""Gate G1 -- REcoM's carbonate chemistry (mocsy) vs a standalone Fortran driver (Task 8).

``fesom_jax.recom.carbonate`` against ``scripts/recom/fortran/mocsy_driver.F90``, which calls
REcoM 692f804's own mocsy sources (built with REcoM's Intel flags) on the G1 input grid of
``scripts/recom/make_mocsy_g1.py`` (4324 points; output ``/work/ab0995/a270088/recom_ref/
mocsy_g1/g1.npz``). Gates, on every point the REcoM floors can produce:

* every output of ``vars_sprac`` (interior), ``pistonvel``, ``flxco2`` and ``o2flux`` (surface)
  and the equilibrium constants: relative error <= 1e-12; pH absolute error <= 1e-12. The
  differences ``co2flux``, ``dpco2`` and ``o2ex`` are measured relative to the size of the
  terms that cancel in them (``kw660*CO2``, ``pCO2``, ``kw660*O2``), not to their own value,
  which crosses zero.
* Measured (2026-10-25, jit path): outputs <= 1.9e-13 (CO3, Omega), constants <= 4.7e-13
  (Kb); p, rhoSW, kw660 bitwise; pH <= 1e-13 absolute (9e-13 at the DIC floor). The floor is
  the transcendental/division ulps of ifort (``-fimf-use-svml -no-prec-div -no-prec-sqrt``)
  vs XLA, amplified by the cancellation in ln Kb (terms ~150 summing to ~-20).
* BetaD (``betad=True``; REcoM computes and discards it): <= 1e-9. Measured 7.7e-11: it is a
  centred difference with dDIC = 1e-7 mol/kg, i.e. a relative step of 5e-5, which amplifies the
  1e-15 solver round-off by ~2e4.
* DIC at REcoM's floor (``tiny*1e-3 = 2.2e-19 mol m-3``, groups ``tiny_dic``/``tiny_both``):
  pH and the constants pass the regular gate; the carbonate species do NOT reach 1e-12 and are
  held to 1e-3 (measured 7.7e-5). There mocsy forms CO2* from ``2*DIC - Ac`` with
  ``Ac = Alk + HSO4 - ... - Kw/h ...`` of order 1e-3, so both codes return round-off of that
  cancellation (CO2* comes out ~1e3 times DIC); the ulp differences decide its value.
* pH-solver iteration counts equal Fortran's or differ by one (~4 % of points): near the
  root the residual is round-off, and whether it is EXACTLY zero (Fortran's EXIT without a
  count, ``phsolvers.F90:584``) is decided by the last ulp. The extra step changes [H+] at
  round-off only, which the value gates above show.

Unit tests (no driver needed): shapes and broadcasting, jit, ``jax.grad`` finite and equal
to central differences w.r.t. DIC, Alk and T, frozen-convergence invariance (more
iterations change nothing), the mocsy DIC guard, and the option check.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fesom_jax  # noqa: F401  (enables x64)
from fesom_jax.recom import carbonate as cb

G1_FILE = Path("/work/ab0995/a270088/recom_ref/mocsy_g1/g1.npz")
TINY_GROUPS = ("tiny_dic", "tiny_both")
SPECIES = ("pco2", "fco2", "co2", "hco3", "co3", "omega_a", "omega_c")


# --------------------------------------------------------------------------
# G1 vs the Fortran driver
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def g1():
    if not G1_FILE.is_file():
        pytest.skip(f"G1 driver output missing: {G1_FILE} (build with "
                    "scripts/recom/fortran/build_mocsy_driver.sh, then run "
                    "scripts/recom/make_mocsy_g1.py)")
    d = np.load(G1_FILE)
    inp = {k[3:]: d[k] for k in d.files if k.startswith("in_")}
    ref = {k[4:]: d[k] for k in d.files if k.startswith("out_")}
    return inp, ref, d["group"]


@pytest.fixture(scope="module")
def g1_jax(g1):
    inp, _, _ = g1
    i = inp
    interior = jax.jit(lambda *a: cb.vars_sprac(*a, betad=True))(
        i["temp"], i["sal"], i["alk"], i["dic"], i["sil"], i["phos"], i["patm"], i["depth"],
        i["lat"])
    kw = cb.pistonvel(i["wind"], i["fice"])
    surf = jax.jit(lambda *a: cb.flxco2(*a, betad=True))(
        i["temp"], i["sal"], i["alk"], i["dic"], i["sil"], i["phos"], kw, i["xco2"],
        i["patm"], i["lat"])
    o2ex = cb.o2flux(i["temp"], i["sal"], kw, i["patm"], i["o2"])  # ppo = patm (recom_forcing:275)
    k = cb.constants(i["temp"], i["sal"], i["patm"], i["depth"], i["lat"])
    out = {n: getattr(interior, n) for n in
           ("ph", "kspc", "betad", "rho_sw", "p", "tempis", "niter") + SPECIES}
    out.update(kw660=kw, co2flux=surf.co2flux, co2ex=surf.co2ex, dpco2=surf.dpco2,
               s_k0=surf.k0, o2ex=o2ex)
    for n in ("ph", "betad", "rho_sw", "p", "tempis", "niter") + SPECIES:
        out["s_" + n] = getattr(surf.carb, n)
    out.update({n: getattr(k, n) for n in cb.Constants._fields})
    return {n: np.asarray(v) for n, v in out.items()}


def _rel_err(name, got, ref, inp):
    """|got - ref| / scale; for the differences, scale = the cancelling terms."""
    scale = np.abs(ref[name])
    if name in ("co2flux", "co2ex"):
        scale = np.maximum(scale, ref["kw660"] * ref["s_co2"])
    elif name == "dpco2":
        scale = np.maximum(scale, ref["s_pco2"])
    elif name == "o2ex":
        scale = np.maximum(scale, ref["kw660"] * inp["o2"])
    return np.abs(got[name] - ref[name]) / np.where(scale > 0.0, scale, 1.0)


GATED = (["ph", "kspc", "rho_sw", "p", "tempis"] + list(SPECIES)
         + ["kw660", "co2flux", "co2ex", "dpco2", "s_k0", "o2ex"]
         + ["s_" + n for n in ("ph", "rho_sw", "p", "tempis") + SPECIES]
         + list(cb.Constants._fields))


@pytest.mark.parametrize("name", GATED)
def test_g1_output(g1, g1_jax, name):
    inp, ref, group = g1
    valid = inp["dic"] > 0.0
    regular = valid & ~np.isin(group, TINY_GROUPS)
    mask = regular if name.removeprefix("s_") in SPECIES else valid
    rel = _rel_err(name, g1_jax, ref, inp)[mask]
    assert rel.max() <= 1e-12, f"{name}: max rel {rel.max():.2e} ({group[mask][rel.argmax()]})"


@pytest.mark.parametrize("name", ["ph", "s_ph"])
def test_g1_ph_absolute(g1, g1_jax, name):
    inp, ref, _ = g1
    valid = inp["dic"] > 0.0
    err = np.abs(g1_jax[name] - ref[name])[valid]
    assert err.max() <= 1e-12, f"{name}: max abs {err.max():.2e}"


@pytest.mark.parametrize("name", ["betad", "s_betad"])
def test_g1_betad(g1, g1_jax, name):
    inp, ref, group = g1
    regular = (inp["dic"] > 0.0) & ~np.isin(group, TINY_GROUPS)
    rel = _rel_err(name, g1_jax, ref, inp)[regular]
    assert rel.max() <= 1e-9, f"{name}: max rel {rel.max():.2e}"


@pytest.mark.parametrize("name", list(SPECIES) + ["s_" + n for n in SPECIES])
def test_g1_tiny_dic_species(g1, g1_jax, name):
    """Ill-conditioned at the DIC floor (module docstring): a loose bound, measured 7.7e-5."""
    inp, ref, group = g1
    tiny = np.isin(group, TINY_GROUPS)
    rel = _rel_err(name, g1_jax, ref, inp)[tiny]
    assert rel.max() <= 1e-3, f"{name}: max rel {rel.max():.2e}"


def test_g1_dic_guard_fill(g1, g1_jax):
    inp, ref, _ = g1
    guard = ~(inp["dic"] > 0.0)
    assert guard.sum() >= 1
    for name in ("ph", "co2", "rho_sw", "p", "tempis", "co2flux", "dpco2", "s_ph"):
        assert np.all(ref[name][guard] == cb.FILL)
        assert np.all(g1_jax[name][guard] == cb.FILL), name


@pytest.mark.parametrize("name", ["niter", "s_niter"])
def test_g1_iteration_counts(g1, g1_jax, name):
    inp, ref, _ = g1
    valid = inp["dic"] > 0.0
    got, want = g1_jax[name][valid], ref[name][valid].astype(np.int64)
    diff = np.abs(got - want)
    assert diff.max() <= 1
    assert np.mean(diff == 0) >= 0.9
    assert got.max() <= cb.G1_WORST_NITER


def test_g1_worst_niter_constant(g1):
    """``G1_WORST_NITER`` in the module is the Fortran worst case on this grid."""
    _, ref, _ = g1
    worst = max(ref[k].max() for k in ("niter", "s_niter", "niter_betad2", "s_niter_betad2"))
    assert worst == cb.G1_WORST_NITER
    assert np.all(ref["replica_ok"] == 1.0) and np.all(ref["s_replica_ok"] == 1.0)


# --------------------------------------------------------------------------
# Unit tests (no driver)
# --------------------------------------------------------------------------
def _column(n=5, nl=4):
    """A small (nod, nl) block of ocean values."""
    rng = np.random.default_rng(0)
    shape = (n, nl)
    return dict(
        temp=rng.uniform(2.0, 25.0, shape), sal=rng.uniform(33.0, 36.0, shape),
        alk=rng.uniform(2.2, 2.4, shape), dic=rng.uniform(1.9, 2.2, shape),
        sil=rng.uniform(0.0, 0.1, shape), phos=rng.uniform(0.0, 2e-3, shape),
        patm=np.full(shape, 1.0), depth=np.tile(np.array([0.0, 50.0, 500.0, 4000.0]), (n, 1)),
        lat=np.tile(rng.uniform(-60, 60, (n, 1)), (1, nl)),
    )


ARGS = ("temp", "sal", "alk", "dic", "sil", "phos", "patm", "depth", "lat")


def test_shapes_and_broadcast():
    c = _column()
    v = cb.vars_sprac(*(c[k] for k in ARGS))
    for name in ("ph", "pco2", "co2", "hco3", "co3", "omega_c", "kspc", "rho_sw", "niter"):
        assert getattr(v, name).shape == (5, 4)
    assert v.betad is None
    # a scalar patm and a per-node latitude broadcast against (nod, nl)
    v2 = cb.vars_sprac(*(c[k] for k in ARGS[:6]), 1.0, c["depth"], c["lat"][:, :1])
    np.testing.assert_array_equal(np.asarray(v2.ph), np.asarray(v.ph))
    s = cb.flxco2(*(c[k][:, 0] for k in ARGS[:6]), 1e-5, 278.0, 1.0, c["lat"][:, 0])
    assert s.co2flux.shape == (5,)
    assert np.all(np.isfinite(np.asarray(s.co2flux)))


def test_jit_matches_eager():
    """Not bitwise: XLA fuses (and FMA-contracts) differently under jit; measured <= 1.3e-13,
    and the iteration count moves by one where the exactly-zero residual flips (G1 docstring)."""
    c = _column()
    eager = cb.vars_sprac(*(c[k] for k in ARGS))
    jitted = jax.jit(cb.vars_sprac)(*(c[k] for k in ARGS))
    for name, a, b in zip(cb.MocsyVars._fields, eager, jitted):
        if name == "niter":
            assert np.abs(np.asarray(b) - np.asarray(a)).max() <= 1
        elif a is not None:
            np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12, atol=0)


def test_frozen_convergence_invariance():
    """More iterations change nothing once every point has stopped (the frozen loop)."""
    c = _column(n=50)
    base = cb.vars_sprac(*(c[k] for k in ARGS), n_iter=cb.G1_WORST_NITER + 1)
    for n_iter in (cb.G1_WORST_NITER + 5, cb.PH_MAXITER):
        more = cb.vars_sprac(*(c[k] for k in ARGS), n_iter=n_iter)
        for a, b in zip(base, more):
            if a is not None:
                np.testing.assert_array_equal(np.asarray(b), np.asarray(a))
    # too few iterations: the unconverged points are Fortran's failure value
    few = cb.vars_sprac(*(c[k] for k in ARGS), n_iter=1)
    assert np.all(np.asarray(few.ph) == cb.FILL)


def test_dic_guard_and_options():
    c = _column(n=1, nl=4)
    dic = c["dic"].copy()
    dic[0, 1] = 0.0
    v = cb.vars_sprac(c["temp"], c["sal"], c["alk"], dic, *(c[k] for k in ARGS[4:]))
    assert float(v.ph[0, 1]) == cb.FILL and float(v.niter[0, 1]) == -1
    assert np.all(np.asarray(v.ph)[0, [0, 2, 3]] < 9.0)
    with pytest.raises(NotImplementedError):
        cb.vars_sprac(*(c[k] for k in ARGS), optK1K2="m10")
    cb.vars_sprac(*(c[k] for k in ARGS), optT="Tpot   ", optK1K2="l  ")   # REcoM's padding


def _surface_cost(temp, alk, dic):
    kw = cb.pistonvel(8.0, 0.1)
    s = cb.flxco2(temp, 35.0, alk, dic, 0.01, 1e-3, kw, 278.0, 1.0, 45.0)
    v = cb.vars_sprac(temp, 35.0, alk, dic, 0.01, 1e-3, 1.0, 1000.0, 45.0)
    return jnp.sum(s.co2flux) * 1e6 + jnp.sum(v.ph) + jnp.sum(v.co3) * 1e2


@pytest.mark.parametrize("argnum", [0, 1, 2])
def test_grad_matches_central_differences(argnum):
    """Reverse-mode gradient (implicit root derivative) vs central differences."""
    x = [jnp.array([5.0, 15.0, 25.0]), jnp.array([2.25, 2.30, 2.35]),
         jnp.array([2.00, 2.05, 2.10])]
    grad = jax.grad(_surface_cost, argnums=argnum)(*x)
    assert np.all(np.isfinite(np.asarray(grad)))
    step = 1e-4 if argnum == 0 else 1e-7
    fd = []
    for j in range(3):
        xp = [a for a in x]
        xm = [a for a in x]
        xp[argnum] = x[argnum].at[j].add(step)
        xm[argnum] = x[argnum].at[j].add(-step)
        fd.append((_surface_cost(*xp) - _surface_cost(*xm)) / (2 * step))
    np.testing.assert_allclose(np.asarray(grad), np.asarray(fd), rtol=1e-5)


def test_grad_under_jit_on_a_column():
    c = _column()
    f = jax.jit(jax.grad(lambda d: jnp.sum(cb.vars_sprac(
        c["temp"], c["sal"], c["alk"], d, *(c[k] for k in ARGS[4:])).pco2)))
    g = np.asarray(f(jnp.asarray(c["dic"])))
    assert g.shape == (5, 4) and np.all(np.isfinite(g)) and np.all(g > 0.0)
