"""REcoM parameters, tracer map and RecomConfig (parent plan Task 6, lane S step S0).

The bin_2p3z2d namelist the Fortran reference ran (``scripts/recom/fortran/namelists/namelist.recom``) must load
with its values; every out-of-scope switch raises ``NotImplementedError``; unknown groups or keys raise.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from fesom_jax.recom import params as P
from fesom_jax.recom import tracers as T
from fesom_jax.recom.config import RecomConfig

REPO = Path(__file__).resolve().parents[2]
NML = REPO / "scripts" / "recom" / "fortran" / "namelists" / "namelist.recom"


def _write(tmp_path, text, name="namelist.recom"):
    p = tmp_path / name
    p.write_text(text)
    return p


def _patched(tmp_path, key, value):
    """The bin_2p3z2d namelist with one ``key = value`` line replaced."""
    lines = []
    hit = False
    for line in NML.read_text().splitlines():
        if line.split("=")[0].strip().lower() == key.lower():
            line, hit = f"{key} = {value}", True
        lines.append(line)
    assert hit, key
    return _write(tmp_path, "\n".join(lines) + "\n")


def test_bin_2p3z2d_values_load():
    sw, p = P.load_namelist(NML)
    # switches of bin_2p3z2d
    assert sw.enable_3zoo2det and not sw.enable_coccos and sw.bgc_num == 31 and sw.biostep == 1
    assert sw.Grazing_detritus and sw.use_photodamage and not sw.OmegaC_diss and sw.constant_CO2
    assert sw.fe_data_source == "Albani" and sw.nm_fe_data_file == "DustClimMonthlyAlbani_pimesh.nc"
    assert sw.REcoM_restart == "@RECOM_RESTART@"           # unfilled template placeholder: kept, no effect
    # namelist overrides of the recom_config defaults
    assert p.graz_max == 0.31 and P.RecomParams().graz_max == 2.4
    assert p.w_ref1 == 5.0 and p.visc_ref_water == 0.00158 and p.depth_scaling1 == 0.01
    assert p.Vcalc == 0.0144 and p.Vdet_a == 0.0288 and p.Redfield == 6.625 and p.RiverFeConc == 100.0
    assert p.permil == 0.000000976 and p.surf_relax_Alk == 3.2e-07 and p.Cunits == 976.5625
    assert p.k_Fe_c == 0.09 and p.deg_Chl_d == 0.15 and p.fecal_rate_n == 0.104
    # a default the namelist does not set
    assert p.ord_phy == -1.2154 and p.kScavFe2 == 0.0
    # the Fortran-zero defaults (0.d00158, 0.d015)
    assert P.RecomParams().visc_ref_water == 0.0 and P.RecomParams().depth_scaling1 == 0.0


def test_filled_run_namelist_equals_template():
    run = Path("/work/ab0995/a270088/recom_ref/pi_30d_jan/namelist.recom")
    if not run.is_file():
        pytest.skip("Fortran reference run directory absent")
    sw, p = P.load_namelist(run)
    sw0, p0 = P.load_namelist(NML)
    assert p == p0
    assert sw.REcoM_restart is False and dataclasses.replace(sw, REcoM_restart=sw0.REcoM_restart,
                                                            REcoMDataPath=sw0.REcoMDataPath) == sw0


def test_params_are_a_pytree_and_switches_hashable():
    import jax
    sw, p = P.load_namelist(NML)
    hash(sw)
    leaves = jax.tree_util.tree_leaves(p)
    assert len(leaves) == len(P.RecomParams._fields) and all(isinstance(x, float) for x in leaves)


def test_derived_floors():
    _, p = P.load_namelist(NML)
    d = P.derived(p)
    assert d.tiny_N == 1e-5 / 3.15 and d.tiny_C == d.tiny_N / 0.2
    assert d.tiny_N_d == 1e-5 / 4.2 and d.tiny_C_d == d.tiny_N_d / 0.2 and d.tiny_Si == d.tiny_C_d * 0.8


@pytest.mark.parametrize("key,value", [
    ("ciso", ".true."), ("use_atbox", ".true."), ("enable_coccos", ".true."), ("useRivers", ".true."),
    ("useErosion", ".true."), ("use_MEDUSA", ".true."), ("biostep", "2"), ("NitrogenSS", ".true."),
    ("useAeolianN", ".true."), ("OmegaC_diss", ".true."),
    ("Grazing_detritus", ".false."), ("fe_data_source", "'Mahowald'"), ("enable_3zoo2det", ".false."),
])
def test_out_of_scope_switch_raises(tmp_path, key, value):
    path = _patched(tmp_path, key, value)
    P.load_namelist(path, check=False)                      # reads fine
    with pytest.raises(NotImplementedError, match=key):
        P.load_namelist(path)
    with pytest.raises(NotImplementedError):
        RecomConfig(namelist=str(path)).validate()


def test_unknown_group_and_key_raise(tmp_path):
    with pytest.raises(KeyError, match="unknown key"):
        P.load_namelist(_write(tmp_path, "&pavariables\nbiostepp = 1\n/\n"))
    with pytest.raises(KeyError, match="unknown REcoM namelist group"):
        P.load_namelist(_write(tmp_path, "&nosuchgroup\nx = 1\n/\n"))
    with pytest.raises(TypeError):
        P.load_namelist(_write(tmp_path, "&pazooplankton\ngraz_max = .true.\n/\n"))
    with pytest.raises(ValueError, match="not closed"):
        P.read_namelist(_write(tmp_path, "&pazooplankton\ngraz_max = 1.\n"))


def test_reader_syntax(tmp_path):
    g = P.read_namelist(_write(tmp_path, "! head\n&pazooplankton graz_max = 1.5d-1, epsilonr=2 ! c\n"
                                         "Redfield = 7.e0 /\n&nam_rsbc\nnm_fe_data_file = 'a!b/c.nc'\n/\n"))
    assert g["pazooplankton"] == {"graz_max": 0.15, "epsilonr": 2, "redfield": 7.0}
    assert g["nam_rsbc"] == {"nm_fe_data_file": "a!b/c.nc"}
    sw, p = P.from_namelist_dict(g)
    assert p.graz_max == 0.15 and p.epsilonr == 2.0 and sw.nm_fe_data_file == "a!b/c.nc"


def test_tracer_map():
    assert T.NTR == 31 and T.BGC_IDS[-1] == 1037 and T.slot(1037) == T.idicremin == 30
    assert [T.slot(i) for i in (1001, 1002, 1003, 1022, 1023, 1028, 1029, 1030)] == \
        [T.idin, T.idic, T.ialk, T.ioxy, T.izoo2n, T.idetz2calc, T.imiczoon, T.imiczooc]
    assert len(T.SINKING_IDS) == 16 and set(T.SINKING_IDS) == set(range(1004, 1009)) | set(range(1013, 1018)) \
        | {1020, 1021} | set(range(1025, 1029))
    assert T.REMIN_IDS == (1001, 1002, 1003, 1018, 1019, 1022)
    assert set(T.IC_FILES) == {1001, 1002, 1003, 1018, 1019, 1022}
    T.check_tracer_ids(T.BGC_IDS)
    with pytest.raises(ValueError):
        T.check_tracer_ids(T.BGC_IDS[:-1])
    with pytest.raises(KeyError):
        T.slot(1)
    _, p = P.load_namelist(NML)
    assert T.init_value(1004, p) == 1e-5 / 3.15 and T.init_value(1010, p) == 2.23e-16 * 6.625
    assert T.init_value(1016, p) == 1e-5 / 4.2 / 0.2 / 0.8
    with pytest.raises(KeyError):
        T.init_value(1037, p)


def test_recom_config_modes():
    with pytest.raises(NotImplementedError):
        RecomConfig().validate()
    full = RecomConfig(namelist=str(NML)).validate()
    assert full.mode == "full" and full.ntr == 31 and full.load()[1].graz_max == 0.31
    assert RecomConfig(passive_tracers=3).mode == "passive" and RecomConfig(passive_tracers=3).ntr == 3
    with pytest.raises(ValueError, match="exclusive"):
        RecomConfig(passive_tracers=3, namelist=str(NML)).validate()
    from fesom_jax.run_config import RunConfig
    cfg = RunConfig.from_dict({"recom": {"namelist": str(NML)}})
    assert cfg.recom == full and RunConfig.from_dict(cfg.to_dict()) == cfg
