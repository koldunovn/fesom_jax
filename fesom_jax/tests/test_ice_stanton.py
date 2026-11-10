"""The ocean-to-ice Stanton number ``IceConfig.st_oce_ice`` (was the literal 0.006 in
``ice_thermo.therm_ice_cell``; ``ice_thermo_oce.F90:582`` in FESOM2).

What is pinned here:

* the default traces exactly the program of the literal: the jaxpr hashes of one CORE2
  ``thermodynamics`` call and of one assembled ice step (standard EVP, and the production
  mEVP + frozen-sweep adjoint on the zstar path) equal the values recorded at the commit
  before the field existed;
* ``IceConfig().st_oce_ice == 0.006``;
* on a synthetic cell the ocean-to-ice heat flux is linear in ``st_oce_ice`` with the analytic
  slope (the ice-covered part ``(T_oc - T_f) * St * u* * cc * A``; the open-water part does not
  depend on it).

The hash tests need the CORE2 mesh, PHC IC and the 1958 forcing; they skip without them.
Re-pin only if JAX is upgraded or the ice physics changes on purpose, from the commit before
the change: ``python fesom_jax/tests/test_ice_stanton.py`` prints the current hashes.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ice as icemod
from fesom_jax import ice_step, ice_thermo, surface_forcing
from fesom_jax.ice import IceConfig

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0
YEAR = 1958

_HAVE_CORE2 = MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file()
core2 = pytest.mark.skipif(not _HAVE_CORE2,
                           reason="CORE2 mesh / PHC IC missing (sea ice needs the forced path)")

#: the production ice configuration of the ice calibration (mEVP, exact ice adjoint with the
#: frozen-sweep damping); the zstar path (use_virt_salt=False)
PROD = dict(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep")

# sha256 of the canonicalised jaxpr text, jax 0.10.1, CPU, x64, computed by ``jaxpr_hashes`` below.
# First pinned at af5413c (ice-calib, before st_oce_ice existed; job 28030375). Re-pinned on
# merge/adjoint_all (job 28882403/28882468): the calibration and farc commits on top of origin/main
# ded5355 trace exactly ded5355's programs (thermo 50ada3f4, ice_step_evp 08959180,
# ice_step_mevp_fs_zstar 098346a9); the coupled line (coupled/coupled_grad e39f2d4) changes both
# ice-step programs (its switches and the shelf-front FCT bounds, CORE2 forward still bitwise over
# 8 steps), and these are the hashes with it.
PINNED_JAX = "0.10.1"
PINNED = {
    "thermo": "50ada3f4d65b8e5fb913d11e206b5adc824c2413448e83010d3c253e6e6de838",
    "ice_step_evp": "d5525c43a7d8d95a4c99438ae222e6e28cdf42d8bb0fe737f4df429a70f17483",
    "ice_step_mevp_fs_zstar": "8f4f2703ed8220e22d36a3d12d5554728fdba697643f7d0730ab7070f26609f5",
}


def _canon(text: str) -> str:
    return re.sub(r"0x[0-9a-fA-F]+", "0x?", text)


def _sha(text: str) -> str:
    return hashlib.sha256(_canon(text).encode()).hexdigest()


def build_setup():
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import phc_initial_state
    mesh = load_mesh(MESH_DIR)
    bare = phc_initial_state(mesh, IC_DIR)
    state = icemod.seed_ice(bare, mesh, bare.T[:, 0])
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    sf = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
    return mesh, state, cf.static, sf


def thermo_jaxpr(setup, cfg: IceConfig) -> str:
    """The jaxpr of one ``thermodynamics`` call on CORE2 shapes (zstar freshwater path)."""
    mesh, state, fs, sf = setup
    geo_lat = jnp.asarray(mesh.geo_coord_nod2D)[:, 1]
    coef = jnp.full_like(state.a_ice, 1.3e-3)
    cfg = cfg._replace(ice_dt=DT)

    def f(st):
        return ice_thermo.thermodynamics(
            cfg, m_ice=st.m_ice, m_snow=st.m_snow, a_ice=st.a_ice, u_ice=st.u_ice,
            v_ice=st.v_ice, t_skin=st.t_skin, srfoce_temp=st.T[:, 0], srfoce_salt=st.S[:, 0],
            srfoce_u=st.uvnode[:, 0, 0], srfoce_v=st.uvnode[:, 0, 1],
            fsh=sf.shortwave, flo=sf.longwave, Tair=sf.Tair, qa=sf.shum,
            u_wind=sf.u_air, v_wind=sf.v_air, rain=sf.prec_rain, snow=sf.prec_snow,
            runo=fs.runoff_node, ch=coef, ce=coef, geo_lat=geo_lat,
            non_cavity=fs.open_water, use_virt_salt=False)
    return str(jax.make_jaxpr(f)(state))


def ice_step_jaxpr(setup, cfg: IceConfig, *, use_virt_salt=True, **kw) -> str:
    """The jaxpr of one assembled ice step (``ice_surface_step``) on CORE2."""
    mesh, state, fs, sf = setup
    return str(jax.make_jaxpr(
        lambda st: ice_step.ice_surface_step(cfg, mesh, st, sf, fs, dt=DT,
                                             use_virt_salt=use_virt_salt, **kw))(state))


def jaxpr_hashes(setup, cfg_kw=None) -> dict:
    kw = cfg_kw or {}
    return {
        "thermo": _sha(thermo_jaxpr(setup, IceConfig(**kw))),
        "ice_step_evp": _sha(ice_step_jaxpr(setup, IceConfig(**kw))),
        "ice_step_mevp_fs_zstar": _sha(ice_step_jaxpr(setup, IceConfig(**PROD, **kw),
                                                      use_virt_salt=False)),
    }


@pytest.fixture(scope="module")
def setup():
    return build_setup()


# ---------------------------------------------------------------------------


def test_default_value():
    assert IceConfig().st_oce_ice == 0.006
    assert isinstance(IceConfig().st_oce_ice, float)
    # not appended at the end of the adjoint block: test_ice_frozen_sweep pins its order
    # the adjoint block keeps its order; the farc ice work (adjoint_warm_ts) and the coupled
    # ocean (mevp_mass_floor, the melt-pond fields, ...) appended fields after it
    f = IceConfig._fields
    i = f.index("adjoint_alpha_kappa")
    assert f[i:i + 5] == ("adjoint_alpha_kappa", "adjoint_theta", "adjoint_thin_mass",
                          "adjoint_sweep_guard", "adjoint_warm_ts")


@core2
def test_default_traces_the_pinned_programs(setup):
    assert jax.__version__ == PINNED_JAX, (
        f"the pinned jaxpr hashes were taken with jax {PINNED_JAX}, this is {jax.__version__}: "
        "re-pin from the commit before the change (see the comment above PINNED)")
    got = jaxpr_hashes(setup)
    bad = {k: (got[k][:12], PINNED[k][:12]) for k in PINNED if got[k] != PINNED[k]}
    assert not bad, f"the traced program changed (got, pinned): {bad}"


@core2
def test_hash_sees_the_stanton_number(setup):
    """The comparison can fail: another value of the field changes every pinned program."""
    got = jaxpr_hashes(setup, {"st_oce_ice": 0.007})
    assert all(got[k] != PINNED[k] for k in PINNED), got


# ---------------------------------------------------------------------------
# synthetic cell: the ocean-to-ice heat flux is linear in st_oce_ice


def _cell(cfg, A):
    """One thick-ice, snow-free cell under a warm mixed layer, returning the new thickness.
    With hsn = 0 nothing melts snow and flooding is a no-op, so h_new = h + rh_atm - o2ihf *
    ice_dt / cl, where only o2ihf depends on st_oce_ice."""
    f64 = lambda x: jnp.asarray(x, jnp.float64)            # noqa: E731
    args = dict(h=f64(2.0), hsn=f64(0.0), A=f64(A), fsh=f64(50.0), flo=f64(250.0),
                Ta=f64(-5.0), qa=f64(2e-3), rain=f64(0.0), snow=f64(0.0), runo=f64(0.0),
                rsss=f64(34.0), ug=f64(6.0), ustar=f64(0.012), T_oc=f64(0.5), S_oc=f64(34.0),
                ch=f64(1.3e-3), ce=f64(1.3e-3), t=f64(-3.0), lid_clo=f64(0.5))
    out = ice_thermo.therm_ice_cell(cfg, **args, use_virt_salt=False)
    return out, args


@pytest.mark.parametrize("A", [1.0, 0.7])
def test_ocean_to_ice_flux_is_linear_in_stanton(A):
    base = IceConfig(ice_dt=DT)
    h = {}
    for st in (0.0, 0.006, 0.012):
        out, a = _cell(base._replace(st_oce_ice=st), A)
        h[st] = float(out.h)
    assert h[0.012] > 0.0, "the cell melted through: not the linear regime"
    tf = float(ice_thermo.tfrez(base, a["S_oc"]))
    # d h_new / d St = -(T_oc - T_f) * u* * cc * A * ice_dt / cl
    slope = -(float(a["T_oc"]) - tf) * float(a["ustar"]) * base.cc * A * DT / base.cl
    assert slope < 0.0
    np.testing.assert_allclose(h[0.006] - h[0.0], slope * 0.006, rtol=1e-9)
    np.testing.assert_allclose(h[0.012] - h[0.0], slope * 0.012, rtol=1e-9)
    # second difference: rounding only
    assert abs(h[0.012] - 2.0 * h[0.006] + h[0.0]) <= 1e-12 * abs(h[0.0])


def test_default_equals_literal_on_a_cell():
    """At the default the cell output is that of an explicit 0.006, bitwise."""
    base = IceConfig(ice_dt=DT)
    o1, _ = _cell(base, 0.8)
    o2, _ = _cell(base._replace(st_oce_ice=0.006), 0.8)
    for f in o1._fields:
        np.testing.assert_array_equal(np.asarray(getattr(o1, f)), np.asarray(getattr(o2, f)))


if __name__ == "__main__":          # print the current hashes (to re-pin)
    print(jax.__version__)
    for k, v in jaxpr_hashes(build_setup()).items():
        print(f'    "{k}": "{v}",')
