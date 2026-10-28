"""REcoM's BGC tracer map for the ``bin_2p3z2d`` configuration (parent plan Task 6).

The configuration is 2 phytoplankton + 3 zooplankton + 2 detritus (``enable_3zoo2det = .true.``,
``enable_coccos = .false.``): the 30 tracers 1001-1030 plus 1037 (DICremin), 31 in all, in FESOM's
tracer order (``namelist.tra`` ``nml_tracer_list``, after T and S). ``State.bgc`` carries them on its
trailing axis in exactly this order, so **slot = position in** :data:`BGC_IDS` **= Fortran's REcoM
index minus one** (``recom_modules.F90:432-440,702-705``: ``idin = 1`` ... ``ioxy = 22``,
``izoo2n = 23`` ... ``idetz2calc = 28``, ``imiczoon = 29``, ``imiczooc = 30``, ``idicremin = 31``;
the IDs are assigned in ``recom_init.F90:589-643``). REcoM checks the namelist order
(``validate_tracer_id_sequence``, ``recom_modules.F90:1649``); :func:`check_tracer_ids` is that check.

Sets (all in Fortran's own order):

* :data:`SINKING_IDS` -- ``recom_sinking_tracer_id(1:16)`` for this case (``recom_modules.F90:716-718``):
  the tracers that get ``ver_sinking_recom`` in the tracer loop.
* :data:`REMIN_IDS` -- ``recom_remin_tracer_id`` without the ciso IDs (``recom_modules.F90:467``): the
  tracers that receive the benthic return flux ``dtr_bf`` (``diff_ver_recom_expl``).
* :data:`BC_SURFACE_IDS` -- the BGC cases of FESOM's ``bc_surface`` (``oce_ale_tracer.F90:2232-2290``) with
  a nonzero surface flux in this configuration and what feeds them (rivers, erosion, MEDUSA and the
  virtual fluxes are off or zero under zstar, so 1011/1012/1018 carry nothing here; 1018 is listed by
  FESOM but its terms are all zero, see :data:`BC_SURFACE_SOURCE`).
* :data:`IC_FILES` -- the 3-D initial-condition files of ``namelist.tra`` ``tracer_init3d``
  (``bin_2p3z2d``, the 5-degree pi versions); every other BGC tracer starts from
  :func:`init_value` (``recom_init.F90:120-134,647-708``), DICremin keeps FESOM's default (0).
"""

from __future__ import annotations

from typing import Dict, Tuple

# FESOM tracer IDs of the BGC block, in State.bgc slot order (namelist.tra order).
BGC_IDS: Tuple[int, ...] = tuple(range(1001, 1031)) + (1037,)
NTR = len(BGC_IDS)                      # 31 = bgc_num of bin_2p3z2d
SLOT: Dict[int, int] = {tid: s for s, tid in enumerate(BGC_IDS)}

# Short names (REcoM's state-variable names, recom_sms.F90 and recom_modules.F90:313-385).
NAMES: Tuple[str, ...] = (
    "DIN", "DIC", "Alk", "PhyN", "PhyC", "PhyChl", "DetN", "DetC", "HetN", "HetC",
    "DON", "DOC", "DiaN", "DiaC", "DiaChl", "DiaSi", "DetSi", "DSi", "DFe", "PhyCalc",
    "DetCalc", "O2", "Zoo2N", "Zoo2C", "DetZ2N", "DetZ2C", "DetZ2Si", "DetZ2Calc", "MicZooN",
    "MicZooC", "DICremin",
)
assert len(NAMES) == NTR

# REcoM's slot indices (0-based; Fortran's idin... minus one). The names follow recom_config.
idin, idic, ialk, iphyn, iphyc, ipchl, idetn, idetc, ihetn, ihetc = range(10)
idon, idoc, idian, idiac, idchl, idiasi, idetsi, isi, ife, iphycal = range(10, 20)
idetcal, ioxy, izoo2n, izoo2c, idetz2n, idetz2c, idetz2si, idetz2calc, imiczoon, imiczooc = range(20, 30)
idicremin = 30

# recom_sinking_tracer_id(1:16) of the 3zoo2det case (recom_modules.F90:716-718), in that order.
SINKING_IDS: Tuple[int, ...] = (1007, 1008, 1017, 1021, 1004, 1005, 1020, 1006,
                                1013, 1014, 1016, 1015, 1025, 1026, 1027, 1028)
# recom_remin_tracer_id without the ciso tracers 1302/1402 (recom_modules.F90:467).
REMIN_IDS: Tuple[int, ...] = (1001, 1002, 1003, 1018, 1019, 1022)
# Detritus classes REcoM's ballasting reads (recom_det_tracer_id / recom_det2_tracer_id, recom_modules.F90:477,707).
DET_IDS: Tuple[int, ...] = (1007, 1008, 1017, 1021)
DET2_IDS: Tuple[int, ...] = (1025, 1026, 1027, 1028)

# bc_surface (oce_ale_tracer.F90:2232-2290): the BGC IDs with a surface flux and the stage-1 field
# that feeds it [mmol m-2 s-1, Fe umol m-2 s-1] (bc_surface multiplies by dt). Every other BGC ID has
# bc_surface = 0 (CASE 1004:1010, 1013:1017, 1020:...; 1011/1012 are rivers only).
BC_SURFACE_SOURCE: Dict[int, str] = {
    1001: "AtmNInput",               # + virtual_din + rivers + erosion: all zero here
    1002: "GloCO2flux_seaicemask",   # + virtual_dic + rivers + erosion: all zero here
    1003: "relax_alk",               # + virtual_alk + rivers: zero here
    1018: "zero",                    # virtual_dsi + rivers + erosion: all zero here
    1019: "AtmFeInput",              # + virtual_dfe (+ RiverFe only with useRivFe)
    1022: "GloO2flux_seaicemask",    # + virtual_oxy
}
BC_SURFACE_IDS: Tuple[int, ...] = tuple(BC_SURFACE_SOURCE)

# namelist.tra &tracer_init3d of bin_2p3z2d with the 5-degree pi files (test/input/recom):
# ID -> (file, variable). Fe from file is multiplied by 1e9 (recom_init.F90:124-127) and capped at the
# hydrothermal vents (mask_hydrothermal_vents, recom_init.F90:706-750), both Task 13.
IC_FILES: Dict[int, Tuple[str, str]] = {
    1001: ("din5deg.nc", "n_an"),
    1002: ("GLODAPv2.2016b.PI_TCO2_fesom2_mmol_fix_z_Fillvalue.nc", "PI_TCO2_mmol"),
    1003: ("talk5deg.nc", "TAlk_mmol"),
    1018: ("si5deg.nc", "i_an"),
    1019: ("fe5deg.nc", "Fe"),
    1022: ("oxy5deg.nc", "oxygen_mmol"),
}


def slot(tracer_id: int) -> int:
    """The ``State.bgc`` slot of FESOM tracer ``tracer_id`` (KeyError for a non-BGC ID)."""
    try:
        return SLOT[int(tracer_id)]
    except KeyError:
        raise KeyError(f"tracer ID {tracer_id} is not a BGC tracer of bin_2p3z2d {BGC_IDS}") from None


def check_tracer_ids(ids) -> None:
    """Raise unless the BGC IDs of a tracer list (FESOM order, T and S excluded) are :data:`BGC_IDS`,
    REcoM's ``validate_tracer_id_sequence`` (``recom_modules.F90:1649``) for this configuration."""
    ids = tuple(int(i) for i in ids)
    if ids != BGC_IDS:
        raise ValueError(f"BGC tracer IDs {ids} differ from the bin_2p3z2d sequence {BGC_IDS} "
                         "(REcoM's validate_tracer_id_sequence)")


def init_value(tracer_id: int, params) -> float:
    """The constant cold-start value of a BGC tracer without an IC file,
    ``get_tracer_init_value`` (``recom_init.F90:647-708``), from a :class:`~fesom_jax.recom.params.RecomParams`.
    Quirks kept: DiaN uses ``Chl2N_max`` (not ``_d``) and DiaSi divides by ``SiCmax``."""
    p = params
    tid = int(tracer_id)
    tiny = TINY
    if tid in (1004, 1013):                                  # PhyN, DiaN
        return TINY_CHL / p.Chl2N_max
    if tid in (1005, 1014):                                  # PhyC, DiaC
        return TINY_CHL / p.Chl2N_max / p.NCmax
    if tid in (1006, 1015):                                  # PhyChl, DiaChl
        return TINY_CHL
    if tid in (1007, 1008, 1009, 1011, 1012, 1017, 1021, 1023, 1025, 1026, 1027, 1028, 1029):
        return tiny
    if tid in (1010, 1020, 1024, 1030):                      # HetC, PhyCalc, Zoo2C, MicZooC
        return tiny * p.Redfield
    if tid == 1016:                                          # DiaSi
        return TINY_CHL / p.Chl2N_max_d / p.NCmax_d / p.SiCmax
    raise KeyError(f"tracer {tid} is initialised from a file or by FESOM (IC_FILES / DICremin), "
                   "not by get_tracer_init_value")


TINY = 2.23e-16          # recom_config tiny (recom_modules.F90:490)
TINY_CHL = 1.0e-5        # recom_config tiny_chl (recom_modules.F90:491; 0.00001 under -r8)
