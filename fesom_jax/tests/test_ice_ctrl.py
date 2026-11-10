"""Generic sea-ice controls ``Params.ice_ctrl = {IceConfig field: traced scalar}``.

``ice_surface_step`` replaces each named float field of the ``IceConfig`` by the traced value
(``cfg._replace(**ice_ctrl)``). What is pinned here:

* the name check: unknown names, integer and string fields, the ``adjoint_*`` settings and the
  derived ``ice_dt`` raise a ``ValueError`` naming the field; so does a non-scalar value and a
  field also set through ``ice_pstar`` / ``ice_cd`` (data-free);
* a traced thermodynamic field passes through the frozen-sweep mEVP backward with a zero
  cotangent, next to a non-zero one for P* (data-free, synthetic mEVP mesh);
* unset (``None`` or an empty dict) ⇒ the ice step traces the program pinned in
  test_ice_stanton (recorded before the controls existed); set ⇒ a different program;
* the ten thermodynamic controls of the ice calibration, traced together at their configured
  values, give the constant-config step to rounding, in one compile for two control vectors;
* CORE2, ``adjoint_mode="exact"`` + frozen sweep: the reverse-mode gradient of a 2-step
  hemispheric ice-area + volume functional w.r.t. ln p of each of the ten controls agrees with
  a central finite difference whose perturbation is applied to the ``IceConfig`` directly,
  NOT through ``ice_ctrl`` (so a swap that dropped a field would show adjoint 0 against a
  non-zero FD).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ice as icemod
from fesom_jax import ice_mevp, ice_mevp_frozen, ice_step, ssh
from fesom_jax import step as stepmod
from fesom_jax.ice import IceConfig
from fesom_jax.params import Params
from fesom_jax.tests import test_ice_stanton as stan
from fesom_jax.tests.test_ice_damping_config import _mevp_cfg, _syn

DT = stan.DT

#: the ten thermodynamic controls of the ice calibration
THERMO = ("albsn", "albsnm", "albi", "albim", "con", "consn", "h0", "h0_s", "c_melt",
          "st_oce_ice")


def _with(base: Params, **kw) -> Params:
    return dataclasses.replace(base, **kw)


def _ctrl(cfg: IceConfig, lnp=None) -> dict:
    """{field: p0·exp(lnp_i)} for the ten controls, strong float64 scalars."""
    lnp = jnp.zeros(len(THERMO), jnp.float64) if lnp is None else lnp
    return {f: jnp.asarray(getattr(cfg, f), jnp.float64) * jnp.exp(lnp[i])
            for i, f in enumerate(THERMO)}


# ---------------------------------------------------------------------------
# the name check (data-free)


def test_the_ten_controls_are_allowed():
    assert set(THERMO) <= set(icemod.ICE_CTRL_FIELDS)
    for f in ("pstar", "cd_oce_ice", "cd_atm_ice", "ellipse"):
        assert f in icemod.ICE_CTRL_FIELDS
    for f in ("ice_dt", "whichEVP", "evp_rheol_steps", "iclasses", "adjoint_mode",
              "adjoint_theta", "adjoint_kappa", "adjoint_alpha_kappa", "adjoint_cuts",
              "use_virt_salt", "snowdist"):
        assert f not in icemod.ICE_CTRL_FIELDS


@pytest.mark.parametrize("name,match", [
    ("albedo_snow", "unknown"), ("whichEVP", "not a float"), ("evp_rheol_steps", "not a float"),
    ("iclasses", "not a float"), ("adjoint_mode", "adjoint"), ("adjoint_theta", "adjoint"),
    ("adjoint_kappa", "adjoint"), ("ice_dt", "ice_dt")])
def test_refused_names_raise(name, match):
    with pytest.raises(ValueError, match=match) as e:
        icemod.check_ice_ctrl({name: jnp.asarray(1.0)})
    assert name in str(e.value)


def test_non_scalar_value_raises():
    with pytest.raises(ValueError, match="albsn"):
        icemod.check_ice_ctrl({"albsn": jnp.ones(3)})
    with pytest.raises(TypeError):
        icemod.check_ice_ctrl([("albsn", 0.8)])


def test_check_passes_the_ten():
    d = _ctrl(IceConfig())
    assert icemod.check_ice_ctrl(d) == d


# ---------------------------------------------------------------------------
# frozen sweep: a traced thermodynamic field passes through with a zero cotangent (data-free)


def test_thermo_field_passes_the_frozen_sweep():
    mesh, fields = _syn()
    cfg = _mevp_cfg(adjoint_mode="exact", adjoint_damping="frozen_sweep")

    def out(p, a=None):
        c = cfg._replace(pstar=p) if a is None else cfg._replace(pstar=p, albsn=a)
        assert ice_mevp_frozen._traced_fields(c) == (("pstar",) if a is None
                                                     else ("pstar", "albsn"))
        return ice_mevp.mevp_dynamics(c, mesh, exch=None, **fields)

    p0, a0 = jnp.asarray(cfg.pstar), jnp.asarray(cfg.albsn)

    def J(p, a):
        u, v, s11, s12, s22 = out(p, a)
        return jnp.sum(u * u + v * v) + 1e-8 * jnp.sum(s11 * s11)

    gp, ga = jax.grad(J, argnums=(0, 1))(p0, a0)
    assert float(ga) == 0.0
    assert np.isfinite(float(gp)) and float(gp) != 0.0
    # tracing the unused field leaves the jitted forward bitwise unchanged (an eager or
    # constant-P* reference would differ by rounding: another XLA program)
    ref = jax.jit(lambda p: out(p))(p0)
    got = jax.jit(out)(p0, a0)
    for r, g in zip(ref, got):
        np.testing.assert_array_equal(np.asarray(r), np.asarray(g))


# ---------------------------------------------------------------------------
# CORE2


@pytest.fixture(scope="module")
def setup():
    if not stan._HAVE_CORE2:
        pytest.skip("CORE2 mesh / PHC IC missing (sea ice needs the forced path)")
    from fesom_jax import surface_forcing
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import phc_initial_state
    mesh = load_mesh(stan.MESH_DIR)
    bare = phc_initial_state(mesh, stan.IC_DIR)
    state = icemod.seed_ice(bare, mesh, bare.T[:, 0])
    cf = surface_forcing.build_surface_forcing(mesh, stan.YEAR,
                                               sst_ic=np.asarray(state.T[:, 0]))
    sfs = [cf.step_forcing(*d) for d in surface_forcing.dates_for_steps(stan.YEAR, DT, 2)]
    op = ssh.build_ssh_operator(mesh, dt=DT)
    return mesh, state, op, cf.static, sfs


def _setup4(setup):
    """The (mesh, state, forcing static, first step forcing) tuple of test_ice_stanton."""
    mesh, state, op, fs, sfs = setup
    return mesh, state, fs, sfs[0]


@stan.core2
def test_unset_traces_the_pinned_ice_step(setup):
    """No controls (``params`` None, ``Params.defaults()``, ``ice_ctrl={}``) ⇒ the ice step of
    the commit before ``ice_ctrl`` existed (the hashes pinned in test_ice_stanton)."""
    s4 = _setup4(setup)
    prod = IceConfig(**stan.PROD)
    for kw in ({}, {"params": None}, {"params": Params.defaults()},
               {"params": _with(Params.defaults(), ice_ctrl={})}):
        assert stan._sha(stan.ice_step_jaxpr(s4, IceConfig(), **kw)) == \
            stan.PINNED["ice_step_evp"], kw
        assert stan._sha(stan.ice_step_jaxpr(s4, prod, use_virt_salt=False, **kw)) == \
            stan.PINNED["ice_step_mevp_fs_zstar"], kw
    # setting one is visible (the comparison can fail)
    p = _with(Params.defaults(), ice_ctrl={"albsn": jnp.asarray(0.81, jnp.float64)})
    assert stan._sha(stan.ice_step_jaxpr(s4, prod, use_virt_salt=False, params=p)) != \
        stan.PINNED["ice_step_mevp_fs_zstar"]


@stan.core2
def test_ice_step_refuses_bad_names(setup):
    s4 = _setup4(setup)
    for d, match in (({"bogus": jnp.asarray(1.0)}, "bogus"),
                     ({"ice_dt": jnp.asarray(900.0)}, "ice_dt")):
        with pytest.raises(ValueError, match=match):
            stan.ice_step_jaxpr(s4, IceConfig(), params=_with(Params.defaults(), ice_ctrl=d))
    p = _with(Params.defaults(), ice_pstar=jnp.asarray(3e4), ice_ctrl={"pstar": jnp.asarray(3e4)})
    with pytest.raises(ValueError, match="pstar"):
        stan.ice_step_jaxpr(s4, IceConfig(), params=p)


#: max|Δ|/max|x| of every float State field after one step, ten controls traced at their
#: configured values vs the constant config (measured: worst 3.9e-13 in eta_n/hbar, every other
#: field ≤ 3.1e-13; job 28030434, CPU). Thermodynamic controls do not touch the mEVP subcycles,
#: so σ needs no looser bound here (unlike P*/C_d in test_ice_controls).
STEP_RTOL = 1e-11


@stan.core2
def test_ten_controls_at_configured_values_give_the_step_to_rounding(setup):
    if jax.devices()[0].platform != "cpu":
        pytest.skip("the rounding bound is calibrated on CPU")
    mesh, state, op, fs, sfs = setup
    cfg = IceConfig(**stan.PROD)

    def one(st, p):
        return stepmod.step(st, mesh, op, None, p, dt=DT, is_first_step=True,
                            step_forcing=sfs[0], forcing_static=fs, ice_cfg=cfg)

    run = jax.jit(one)
    run_ctl = jax.jit(lambda st, lnp: one(st, _with(Params.defaults(),
                                                    ice_ctrl=_ctrl(cfg, lnp))))

    ref = run(state, Params.defaults())
    alt = run_ctl(state, jnp.zeros(len(THERMO), jnp.float64))
    rel = {}
    for f in dataclasses.fields(ref):
        a, b = np.asarray(getattr(ref, f.name)), np.asarray(getattr(alt, f.name))
        if a.dtype.kind == "f":
            rel[f.name] = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))
    print(f"[ice_ctrl at constants] worst rel {sorted(rel.items(), key=lambda kv: -kv[1])[:6]}")
    bad = {k: v for k, v in rel.items() if not np.isfinite(v) or v > STEP_RTOL}
    assert not bad, f"beyond rounding: {bad}"
    # a second control vector reuses the executable (one compile for the ten controls)
    alt2 = run_ctl(state, jnp.full(len(THERMO), 0.01, jnp.float64))
    assert run_ctl._cache_size() == 1
    assert float(jnp.max(jnp.abs(alt2.a_ice - alt.a_ice))) > 0.0, "the controls did nothing"


@stan.core2
def test_ice_ctrl_equals_the_directly_replaced_config(setup):
    """One ice step with ``ice_ctrl`` = the step with ``cfg._replace`` of the same traced values
    (to rounding), for each of the ten controls moved by 1 % in turn, and each move changes the
    ice state. Bare ice on the western half of the Antarctic pack (January sun) so that the
    bare-ice albedos albi/albim act (seed_ice puts snow on every ice node)."""
    mesh, state, op, fs, sfs = setup
    lon = np.asarray(mesh.geo_coord_nod2D)[:, 0]
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]
    bare_ice = jnp.asarray((lat < 0.0) & (lon < 0.0))
    st0 = dataclasses.replace(state, m_snow=jnp.where(bare_ice, 0.0, state.m_snow))
    assert int(jnp.count_nonzero(bare_ice & (st0.a_ice > 0))) > 1000
    cfg = IceConfig(**stan.PROD)

    def ice(c, p, s):
        o = ice_step.ice_surface_step(c, mesh, s, sfs[0], fs, dt=DT, params=p)
        return o.a_ice, o.m_ice, o.m_snow, o.t_skin

    via_ctrl = jax.jit(lambda lnp, s: ice(cfg, _with(Params.defaults(),
                                                     ice_ctrl=_ctrl(cfg, lnp)), s))
    via_cfg = jax.jit(lambda lnp, s: ice(cfg._replace(**_ctrl(cfg, lnp)), None, s))
    z = jnp.zeros(len(THERMO), jnp.float64)
    ref = [np.asarray(x) for x in via_ctrl(z, st0)]
    quiet, mismatch = [], []
    for i, f in enumerate(THERMO):
        lnp = z.at[i].set(0.01)
        a = [np.asarray(x) for x in via_ctrl(lnp, st0)]
        b = [np.asarray(x) for x in via_cfg(lnp, st0)]
        d_ab = max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))
        moved = max(float(np.max(np.abs(x - r))) for x, r in zip(a[:3], ref[:3]))
        print(f"[ice_ctrl plumbing] {f:11s} max|ctrl - cfg| {d_ab:.3e}  max|moved| {moved:.3e}")
        if d_ab > 1e-13:
            mismatch.append((f, d_ab))
        if not moved > 1e-12:
            quiet.append((f, moved))
    assert via_ctrl._cache_size() == 1
    assert not mismatch, f"ice_ctrl and the directly replaced config differ: {mismatch}"
    assert not quiet, f"a 1 % move left the ice state unchanged: {quiet}"


# ---------------------------------------------------------------------------
# CORE2: 2-step gradient vs central FD of the directly perturbed IceConfig


HS = (1e-4, 1e-5, 1e-6)
#: |FD| at or below NOISE_ULPS·eps·|J0|/h is treated as noise (reported, not asserted). The FD
#: spread of the smooth controls (con, consn) across h implies a rounding error of J of about
#: 10 ulps of J0 (job 28030434), so 100 is a tenfold margin.
NOISE_ULPS = 100.0
#: adjoint vs FD at the best resolved h. A plumbing error is O(1) (adjoint 0, or another field's
#: sensitivity). Measured (job 28030434): ≤ 3.3e-4 for con, consn, h0, h0_s, c_melt,
#: st_oce_ice, albsn; 3.1e-3 for albsnm, whose FD itself varies by 0.6 % across h: the
#: dry/melting albedo switch (t < 0, ice_thermo._budget), a forward jump the pathwise adjoint
#: does not see. albi/albim: FD and adjoint exactly 0 (seed_ice puts snow on every ice node, so
#: the bare-ice albedos are never read); their plumbing is checked by
#: test_ice_ctrl_equals_the_directly_replaced_config on a bare-ice start instead. (A bare-ice
#: start made this 2-step FD jump by O(1) between h for every control, jobs 28030810/28031160.)
GRAD_RTOL = 1e-2


def _functional(mesh):
    """Hemispheric ice area + volume: Σ area·(a_ice + m_ice) over both hemispheres, in units of
    1e12 m² (smooth in the state)."""
    w = jnp.asarray(mesh.areasvol[:, 0]) * jnp.asarray(mesh.node_layer_mask[:, 0]) / 1e12

    def J(s):
        return jnp.sum(w * (s.a_ice + s.m_ice))
    return J


def _two_steps(setup, cfg, params, s=None):
    mesh, state, op, fs, sfs = setup
    s = state if s is None else s
    for k in range(2):
        s = stepmod.step(s, mesh, op, None, params, dt=DT, is_first_step=(k == 0),
                         step_forcing=sfs[k], forcing_static=fs, ice_cfg=cfg)
    return s


@pytest.fixture(scope="module")
def grad_and_fd(setup):
    mesh = setup[0]
    cfg = IceConfig(**stan.PROD)
    J = _functional(mesh)

    # adjoint: through Params.ice_ctrl
    def J_ctl(lnp):
        return J(_two_steps(setup, cfg, _with(Params.defaults(), ice_ctrl=_ctrl(cfg, lnp))))
    z = jnp.zeros(len(THERMO), jnp.float64)
    j_ctl, g = jax.jit(jax.value_and_grad(J_ctl))(z)

    # FD: the IceConfig fields themselves (no ice_ctrl), one compile for all perturbations
    def J_cfg(vals):
        c = cfg._replace(**{f: vals[i] for i, f in enumerate(THERMO)})
        return J(_two_steps(setup, c, Params.defaults()))
    Jc = jax.jit(J_cfg)
    p0 = np.array([getattr(cfg, f) for f in THERMO], np.float64)
    j0 = float(Jc(jnp.asarray(p0)))
    fd = np.zeros((len(HS), len(THERMO)))
    for a, h in enumerate(HS):
        for i in range(len(THERMO)):
            e = np.zeros(len(THERMO))
            e[i] = h
            jp = float(Jc(jnp.asarray(p0 * np.exp(e))))
            jm = float(Jc(jnp.asarray(p0 * np.exp(-e))))
            fd[a, i] = (jp - jm) / (2.0 * h)
    return dict(j_ctl=float(j_ctl), j0=j0, g=np.asarray(g), fd=fd, p0=p0, cfg=cfg, J=J)


@stan.core2
def test_controls_at_p0_give_the_same_cost(grad_and_fd):
    r = grad_and_fd
    assert abs(r["j_ctl"] - r["j0"]) <= 1e-12 * abs(r["j0"]), (r["j_ctl"], r["j0"])


@stan.core2
def test_static_config_fd_equals_traced_config_fd(setup, grad_and_fd):
    """The traced-config FD stands for the plan's untraced one: at one control and h the FD of
    two separately compiled constant configs agrees with it (to rounding of the cost)."""
    r = grad_and_fd
    cfg, J = r["cfg"], r["J"]
    i, h = THERMO.index("st_oce_ice"), HS[0]
    p = r["p0"][i]
    st0 = setup[1]

    def run(v):
        return float(jax.jit(lambda s0: J(_two_steps(setup, cfg._replace(st_oce_ice=v),
                                                     Params.defaults(), s0)))(st0))
    jp, jm = run(float(p * np.exp(h))), run(float(p * np.exp(-h)))
    fd_static = (jp - jm) / (2.0 * h)
    noise = NOISE_ULPS * np.finfo(float).eps * abs(r["j0"]) / h
    print(f"[static FD] st_oce_ice h={h:g}: static {fd_static:.12e} traced {r['fd'][0, i]:.12e} "
          f"noise {noise:.2e}")
    assert abs(fd_static - r["fd"][0, i]) <= noise


@stan.core2
def test_gradient_matches_direct_config_fd(grad_and_fd):
    r = grad_and_fd
    g, fd, j0 = r["g"], r["fd"], r["j0"]
    eps = np.finfo(float).eps
    rows, bad, quiet = [], [], []
    for i, f in enumerate(THERMO):
        noise = np.array([NOISE_ULPS * eps * abs(j0) / h for h in HS])
        rel = np.abs(fd[:, i] - g[i]) / np.maximum(np.abs(fd[:, i]), 1e-300)
        resolved = np.abs(fd[:, i]) > noise
        rows.append(f"  {f:11s} adj {g[i]: .10e} "
                    + " ".join(f"fd(h={h:g}) {fd[a, i]: .10e} rel {rel[a]:.2e}{'' if resolved[a] else '*'}"
                               for a, h in enumerate(HS)))
        if not np.isfinite(g[i]):
            bad.append((f, "non-finite adjoint"))
        elif not resolved.any():
            quiet.append(f)
        elif g[i] == 0.0:
            bad.append((f, "adjoint 0 against a resolved FD"))
        elif rel[resolved].min() > GRAD_RTOL:
            bad.append((f, f"best rel {rel[resolved].min():.2e}"))
    print(f"[ice_ctrl grad] J0 {j0:.15e} (* = FD at the noise)\n" + "\n".join(rows))
    print(f"[ice_ctrl grad] at the noise (reported, not asserted): {quiet}")
    assert not bad, bad
    assert len(quiet) < len(THERMO), "every FD at the noise: the test resolves nothing"
