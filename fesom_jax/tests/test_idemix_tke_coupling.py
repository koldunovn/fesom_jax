"""Ruling 2's energy for TKE (patch 0013). The Fortran's TKE reads ``tke_in3d_iwe = iwe_n``
(``gen_modules_cvmix_tke.F90:305``), the node energy AFTER the in-place lateral propagation
(``gen_modules_cvmix_idemix.F90:630-631`` and loops B-E); only ``iwe_Tdis_n``/``iwe_alpha_c_n`` are
column-step fields. The host handed TKE the column-step energy: a first-order Kv difference where the
Rinum cap ``KappaM N^2 / (alpha_c E^2)`` binds (budget review 2026-10-26, finding
tke-rinum-uses-prelateral-E). ``idemix.tke_coupling`` builds the triple ``step`` passes to
``tke.mixing_tke``; ``IdemixConfig.tke_e_iw``: ``"post_lateral"`` (default) = ``iwe_new``,
``"column"`` = the old field, bitwise. ``IdemixOut.E_iw`` stays the column-step energy. On the shipped
pi mesh after 2 host steps."""

import numpy as np
import pytest

import jax.numpy as jnp

pytest.importorskip("idemix_jax")  # the IDEMIX library is a separate package

from idemix_jax import IdemixForcing, IdemixParams
from idemix_jax.mesh import masks_from_levels

from fesom_jax import idemix as I

DT = 100.0
PARAMS = IdemixParams.fesom_operational()


@pytest.fixture(scope="module")
def host():
    from fesom_jax import forcing as host_forcing
    from fesom_jax import ssh
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import PACKAGED_PI_IC_DIR, cold_start_state
    from fesom_jax.step import step

    mesh = load_mesh()
    st = cold_start_state(mesh, PACKAGED_PI_IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress = host_forcing.surface_stress(mesh)
    st = step(st, mesh, op, stress, dt=DT, is_first_step=True)
    st = step(st, mesh, op, stress, dt=DT)
    is_wet = np.asarray(masks_from_levels(mesh.ulevels_nod2D - 1, mesh.nlevels_nod2D - 1,
                                          mesh.nl - 1))
    rng = np.random.default_rng(20261026)
    n = int(mesh.nod2D)
    iwe0 = jnp.asarray(rng.uniform(1e-4, 5e-3, (n, mesh.nl)) * is_wet)
    forcing = IdemixForcing(jnp.asarray(rng.uniform(1e-7, 1e-6, n)),
                            jnp.asarray(rng.uniform(1e-7, 5e-6, n)))
    return dict(mesh=mesh, st=st, op=op, is_wet=is_wet, iwe0=iwe0, forcing=forcing,
                lateral=I.build_lateral_operator(mesh))


def _run(h, cfg):
    return I.mixing_idemix(h["mesh"], h["iwe0"], h["st"].bvfreq, h["st"].hnode, h["forcing"],
                           PARAMS, cfg, h["lateral"] if cfg.lateral else None, dt=DT)


def test_tke_takes_the_post_lateral_energy_by_default(host):
    cfg = I.IdemixConfig()
    assert cfg.tke_e_iw == "post_lateral"
    out = _run(host, cfg)
    assert np.any(np.asarray(out.E_iw) != np.asarray(out.iwe_new))     # the lateral step moved E
    d, a, e = I.tke_coupling(out, cfg)
    assert d is out.iw_diss and a is out.alpha_c and e is out.iwe_new
    d, a, e = I.tke_coupling(out, cfg._replace(tke_e_iw="column"))
    assert d is out.iw_diss and a is out.alpha_c and e is out.E_iw   # patches 0001-0012


def test_the_option_changes_no_idemix_output(host):
    a = _run(host, I.IdemixConfig())
    b = _run(host, I.IdemixConfig(tke_e_iw="column"))
    for name in I.IdemixOut._fields:
        np.testing.assert_array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)),
                                      err_msg=name)


def test_lateral_off_both_options_hand_tke_the_same_energy(host):
    out = _run(host, I.IdemixConfig(lateral=False))
    e1 = I.tke_coupling(out, I.IdemixConfig(lateral=False))[2]
    e2 = I.tke_coupling(out, I.IdemixConfig(lateral=False, tke_e_iw="column"))[2]
    np.testing.assert_array_equal(np.asarray(e1), np.asarray(e2))


def test_the_rinum_cap_sees_the_difference(host):
    """mixing_tke's KappaH (through Kv) changes with the energy the cap reads; KappaM does not
    depend on it (the cap enters the Prandtl number only)."""
    from fesom_jax import tke
    from fesom_jax.params import Params
    mesh, st = host["mesh"], host["st"]
    out = _run(host, I.IdemixConfig())
    stress = jnp.full((mesh.nod2D, 2), 0.05)
    tke0 = jnp.where(jnp.asarray(host["is_wet"]), 1e-4, 0.0)
    args = (mesh, st.uv, st.bvfreq, tke0, stress, st.hnode, tke.TkeConfig(only_tke=False),
            Params.defaults())
    new = tke.mixing_tke(*args, dt=DT, idemix=I.tke_coupling(out, I.IdemixConfig()))
    old = tke.mixing_tke(*args, dt=DT,
                         idemix=I.tke_coupling(out, I.IdemixConfig(tke_e_iw="column")))
    assert not np.array_equal(np.asarray(new[0]), np.asarray(old[0]))       # Kv


def test_step_gives_tke_the_configured_energy(host):
    """``step`` takes TKE's energy from ``IdemixConfig.tke_e_iw``: with lateral propagation the two
    values change Kv, without it they are the same array, so the step is bitwise the same."""
    import dataclasses

    from fesom_jax import tke
    from fesom_jax.params import Params
    from fesom_jax.step import step
    from fesom_jax.surface_forcing import ForcingStatic, StepForcing
    mesh = host["mesh"]
    params = dataclasses.replace(Params.defaults(), idemix=PARAMS)
    st = dataclasses.replace(host["st"], iwe=host["iwe0"],
                             tke=jnp.where(jnp.asarray(host["is_wet"]), 1e-4, 0.0))
    # TKE needs the forced step's nodal stress: a uniform atmosphere, no forcing file
    n = int(mesh.nod2D)
    full = lambda v: jnp.full(n, v)                                # noqa: E731
    sf = StepForcing(u_air=full(8.0), v_air=full(2.0), shum=full(0.008), shortwave=full(150.0),
                     longwave=full(320.0), Tair=st.T[:, 0], prec_rain=full(0.0),
                     prec_snow=full(0.0), Ssurf_month=st.S[:, 0], chl=full(0.1))
    surf = mesh.areasvol[:, 0]
    fs = ForcingStatic(runoff_node=full(0.0), areasvol_surf=surf, ocean_area=jnp.sum(surf),
                       open_water=jnp.ones(n, dtype=bool), a_ice=full(0.0))

    def kv(cfg):
        return np.asarray(step(st, mesh, host["op"], None, params, dt=DT, step_forcing=sf,
                               forcing_static=fs, tke_cfg=tke.TkeConfig(only_tke=False),
                               idemix_cfg=cfg, idemix_forcing=host["forcing"],
                               idemix_lateral=host["lateral"] if cfg.lateral else None).Kv)

    new, old = kv(I.IdemixConfig()), kv(I.IdemixConfig(tke_e_iw="column"))
    assert np.all(np.isfinite(new)) and not np.array_equal(new, old)
    np.testing.assert_array_equal(kv(I.IdemixConfig(lateral=False)),
                                  kv(I.IdemixConfig(lateral=False, tke_e_iw="column")))


def test_validate_and_yaml_key():
    with pytest.raises(ValueError, match="tke_e_iw"):
        I.IdemixConfig(tke_e_iw="after").validate()
    from fesom_jax.run_config import RunConfig
    old = RunConfig.from_dict({"idemix": {"tke_e_iw": "column"}})
    old.validate()
    assert old.idemix_config().tke_e_iw == "column"
    assert RunConfig.from_dict({"idemix": {}}).idemix_config().tke_e_iw == "post_lateral"
