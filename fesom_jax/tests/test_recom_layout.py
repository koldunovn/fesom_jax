"""Lane P of the spectral port (docs/plans/20261028-recom-spectral-port.md): the tracer layout as a function of the
REcoM switches (fesom_jax.recom.layout) and the fork's namelist keys (fesom_jax.recom.params)."""

from pathlib import Path

import pytest

from fesom_jax.recom import layout as L
from fesom_jax.recom import params as P
from fesom_jax.recom import tracers as T

REPO = Path(__file__).resolve().parents[2]
FORK_NML = REPO / "scripts/recom/fortran/namelists_albedo_spectral/namelist.recom"
REF_NML = REPO / "scripts/recom/fortran/namelists_jaxphys/namelist.recom"


def test_default_layout_is_the_tracers_module():
    d = L.DEFAULT
    assert d.ids == T.BGC_IDS and d.names == T.NAMES and d.ntr == T.NTR == 31
    assert d.slot(1037) == T.idicremin == 30 and d.imiczoon == T.imiczoon
    assert d.sinking_ids == T.SINKING_IDS and d.remin_ids == T.REMIN_IDS
    assert d.det_ids == T.DET_IDS and d.det2_ids == T.DET2_IDS
    assert dict(L.BC_SURFACE_SOURCE) == T.BC_SURFACE_SOURCE


@pytest.mark.parametrize("z, c, cdom, ntr, last, cdom_id", [
    # the fork's initialize_tracer_indices (recom_modules.F90:759-884) and validate_recom_tracers
    (True, True, True, 37, 1037, 1037), (True, True, False, 36, 1036, None),
    (True, False, True, 31, 1031, 1031), (True, False, False, 30, 1030, None),
    (False, True, True, 29, 1029, 1029), (False, True, False, 28, 1028, None),
    (False, False, True, 23, 1023, 1023), (False, False, False, 22, 1022, None),
])
def test_fork_layouts(z, c, cdom, ntr, last, cdom_id):
    lay = L.layout_for(enable_3zoo2det=z, enable_coccos=c, cdom=cdom)
    assert lay.ntr == ntr and lay.ids[-1] == last and lay.cdom_id == cdom_id
    assert lay.ids == tuple(range(1001, 1001 + ntr))           # the fork numbers contiguously
    if c:
        assert lay.cocco_ids == (1029, 1030, 1031) if z else (1023, 1024, 1025)
        assert lay.phaeo_ids == (1032, 1033, 1034) if z else (1026, 1027, 1028)
        assert lay.icocn == lay.slot(lay.cocco_ids[0]) and lay.names[lay.iphachl] == "PhaeoChl"
    if z:
        assert lay.det2_ids == (1025, 1026, 1027, 1028)
        assert (lay.imiczoon, lay.imiczooc) == ((34, 35) if c else (28, 29))   # fork: 35/36 resp. 29/30, 1-based
    assert all(i in lay.ids for i in lay.sinking_ids)
    assert hash(lay) == hash(L.layout_for(enable_3zoo2det=z, enable_coccos=c, cdom=cdom))


def test_cdom_and_dicremin_are_exclusive():
    with pytest.raises(ValueError):
        L.layout_for(enable_3zoo2det=True, enable_coccos=False, cdom=True, dicremin=True)
    with pytest.raises(KeyError):
        L.DEFAULT.slot(1031)


def test_fork_namelist_loads_with_fork_defaults():
    sw, par = P.load_namelist(FORK_NML, check=False)
    assert sw.spectral_fork and sw.enable_coccos and sw.enable_3zoo2det and sw.bgc_num == 37
    assert sw.RECOM_CDOM and sw.OASIM and sw.RECOM_RADTRANS and sw.RECOM_CALC_ACDOM and sw.RECOM_BMASS
    assert sw.par_layer_mean is True                    # FORK_DEFAULTS: the fork declares .true.
    assert par.cdomcoeff == 0.13 and par.bb_to_b == 0.01 and par.NCmax_p == 0.2
    assert (par.darwin_bbphy_1, par.darwin_bbphy_2, par.darwin_bbphy_3, par.darwin_bbphy_4) == (0.002, 0.0105, 0.02, 0.002)
    lay = L.layout_of(sw)
    assert lay == L.SPECTRAL_4P3Z2D and lay.ntr == sw.bgc_num
    d = P.derived(par)
    assert d.tiny_N_c == P.TINY_CHL / par.Chl2N_max_c and d.tiny_C_p == d.tiny_N_p / par.NCmax_p


def test_fork_namelist_is_accepted_and_its_unported_switches_refused():
    """Lanes E, L and W landed (2026-10-28): the fork's reference namelist (4 PFT, 37 tracers) passes the switch
    check; the switches the port leaves out (RECOM_MARSHALL, RECOM_CALC_APHYT, ...) are still refused, and a bgc_num
    that fits none of the fork's layouts is a ValueError."""
    sw, _ = P.load_namelist(FORK_NML, check=False)
    assert sw.spectral_fork and sw.enable_coccos
    P.check_supported(sw)
    import dataclasses
    with pytest.raises(NotImplementedError, match="RECOM_MARSHALL"):
        P.check_supported(dataclasses.replace(sw, RECOM_MARSHALL=True))
    with pytest.raises(ValueError, match="bgc_num"):
        P.check_supported(dataclasses.replace(sw, bgc_num=35))


def test_reference_namelist_unchanged():
    sw, par = P.load_namelist(REF_NML)
    assert not sw.spectral_fork and sw.par_layer_mean is False and par.NCmax_p == 0.1
    assert L.layout_of(sw) == L.DEFAULT
    assert P.derived(par)[:5] == P.derived(par)[:5]


def test_recom_config_layout():
    from fesom_jax.recom.config import RecomConfig
    cfg = RecomConfig(namelist=str(REF_NML)).validate()
    assert cfg.ntr == 31 and cfg.layout == L.DEFAULT
