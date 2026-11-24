"""The spectral fork's 2026-11-19/20 options (``sveta-loza/fesom2`` ``fesom2.6_recom_tra_diags_spectral`` 84ad0b5c,
cb446565) in ``recom_sources_sinks_fork``, on a synthetic column set (no data, no twin): ``&pacalc calc_scale_cocco``
and ``calc_on_net_cocco`` (the coccolithophore PIC production), ``&spectral cdom_remin_dic`` (the CDOM sinks credited
to DIC and O2). Each is checked against the unswitched step through the budget the Fortran change states; the
defaults leave the result bitwise unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

from fesom_jax.recom import layout as LY                                           # noqa: E402
from fesom_jax.recom import params as P                                            # noqa: E402
from fesom_jax.recom.sms import CARB_FIELDS, CarbCache                             # noqa: E402
from fesom_jax.recom.sms_fork import REDO2C                                        # noqa: E402
from fesom_jax.recom.stage1 import BgcStepInputs, RecomGeometry, recom_sources_sinks_fork  # noqa: E402

NML = Path(__file__).resolve().parents[2] / "scripts/recom/fortran/namelists_albedo_spectral/namelist.recom"
DT = 2700.0
N, NL1 = 6, 5


def _case():
    sw, par = P.load_namelist(NML, check=False)
    lay = LY.layout_of(sw)
    assert lay.cdom and sw.enable_coccos
    rng = np.random.default_rng(7)
    typ = {"DIN": 10.0, "DIC": 2100.0, "Alk": 2300.0, "DSi": 20.0, "O2": 250.0, "DFe": 0.5}
    bgc = np.empty((N, NL1, lay.ntr))
    for i, n in enumerate(lay.names):
        bgc[..., i] = typ.get(n, 0.2) * rng.uniform(0.5, 1.5, (N, NL1))
    for g in ("Phy", "Dia", "Cocco", "Phaeo"):        # N:C quotas inside (NCmin, NCmax), so uptake and respiration are live
        bgc[..., lay.names.index(g + "C")] = 8.0 * bgc[..., lay.names.index(g + "N")] * rng.uniform(0.9, 1.1, (N, NL1))
    thick = np.full((N, NL1), 10.0)
    zbar = -np.concatenate([np.zeros((N, 1)), np.cumsum(thick, axis=1)], axis=1)
    geom = RecomGeometry(hnode=thick, zbar=zbar, nlev=np.full(N, NL1, np.int32),
                         lat_rad=np.deg2rad(np.linspace(-40.0, 40.0, N)), area_top=np.full(N, 1e10),
                         ocean_area=np.float64(N * 1e10))
    one = np.ones(N)
    inp = BgcStepInputs(daynew=180, ndpyr=365, mstep=5, fe_dust=0.0 * one, n_dust=0.0 * one, co2_atm=280.0,
                        press_air=101325.0 * one, u_wind=5.0 * one, v_wind=0.0 * one, shortwave=200.0 * one,
                        a_ice=0.0 * one)
    cv = dict(CO2=1.2e-2, pH=8.1, pCO2=300.0, HCO3=1.8, CO3=0.2,          # CO2, HCO3, CO3 [mol m-3]
              OmegaC=4.0, kspc=4.3e-7, rhoSW=1025.0)
    carb = CarbCache(*(np.full((N, NL1), cv[n]) for n in CARB_FIELDS))
    temp = np.full((N, NL1), 15.0)
    salt = np.full((N, NL1), 35.0)
    return sw, par, lay, (bgc, carb, np.full((N, 4), 1.0), bgc[:, 0, lay.names.index("Alk")], temp, salt, geom, inp)


def _run(sw, par, lay, args):
    return jax.device_get(recom_sources_sinks_fork(*args, DT, par, sw, lay))


def test_new_keys_read(tmp_path):
    """The three keys are read from their Fortran groups and default to the old behaviour."""
    assert P.RecomParams().calc_scale_cocco == 1.0
    assert P.RecomSwitches().calc_on_net_cocco is False and P.RecomSwitches().cdom_remin_dic is False
    f = tmp_path / "namelist.recom"
    f.write_text("&pacalc\n calc_scale_cocco = 0.25\n calc_on_net_cocco = .true.\n/\n&spectral\n cdom_remin_dic = .true.\n/\n")
    sw, par = P.load_namelist(f, check=False)
    assert par.calc_scale_cocco == 0.25 and sw.calc_on_net_cocco is True and sw.cdom_remin_dic is True


def test_calc_scale_and_net():
    sw, par, lay, args = _case()
    r0 = _run(sw, par, lay, args)
    c0 = r0.sms.diag3d["calcif"]
    assert c0.min() > 0.0
    # the explicit defaults are the old program's values
    r1 = _run(replace(sw, calc_on_net_cocco=False), par._replace(calc_scale_cocco=1.0), lay, args)
    assert np.array_equal(r1.bgc, r0.bgc)
    # calc_scale_cocco scales the PIC production; DIC and Alk follow at 1 : 2 (S:4739, 4821)
    rs = _run(sw, par._replace(calc_scale_cocco=0.25), lay, args)
    np.testing.assert_allclose(rs.sms.diag3d["calcif"], 0.25 * c0, rtol=1e-14)
    i = lay.names.index
    ddic = rs.bgc[..., i("DIC")] - r0.bgc[..., i("DIC")]
    dalk = rs.bgc[..., i("Alk")] - r0.bgc[..., i("Alk")]
    assert ddic.min() > 0.0
    np.testing.assert_allclose(dalk, 2.0 * ddic, rtol=1e-6)
    np.testing.assert_allclose(rs.bgc[..., i("PhyCalc")] - r0.bgc[..., i("PhyCalc")], -ddic, rtol=1e-6)
    # calc_on_net_cocco: gross Cphot_cocco -> max(0, Cphot_cocco - phyRespRate_cocco); with the diagnostics
    # NPPc3D = (Cphot_cocco - phyRespRate_cocco) CoccoC and respc = phyRespRate_cocco CoccoC (res_ovf off)
    pn = par._replace(res_ovf=0.0, res_ovf_d=-1.0, res_ovf_c=-1.0, res_ovf_p=-1.0)
    rg = _run(sw, pn, lay, args)
    rn = _run(replace(sw, calc_on_net_cocco=True), pn, lay, args)
    cg, npp, resp = rg.sms.diag3d["calcif"], rg.sms.diag3d["NPPc3D"], rg.sms.diag3d["respc"]
    assert resp.min() > 0.0
    np.testing.assert_allclose(rn.sms.diag3d["calcif"], cg * np.maximum(0.0, npp) / (npp + resp), rtol=1e-12)
    assert (rn.sms.diag3d["calcif"] < cg).all()


def test_cdom_remin_dic():
    sw, par, lay, args = _case()
    r0 = _run(sw, par, lay, args)
    r1 = _run(replace(sw, cdom_remin_dic=True), par, lay, args)
    i = lay.names.index
    ddic = r1.bgc[..., i("DIC")] - r0.bgc[..., i("DIC")]
    doxy = r1.bgc[..., i("O2")] - r0.bgc[..., i("O2")]
    assert ddic.min() > 0.0
    np.testing.assert_allclose(doxy, -REDO2C * ddic, rtol=1e-6)
    # nothing else moves, the CDOM sink itself included
    others = [k for k, n in enumerate(lay.names) if n not in ("DIC", "O2")]
    assert np.array_equal(r1.bgc[..., others], r0.bgc[..., others])
    # the credited carbon is the CDOM loss of the step minus its sources: with the sources cut, DIC gains what CDOM loses
    assert ddic.max() < args[0][..., i("CDOM")].max()
