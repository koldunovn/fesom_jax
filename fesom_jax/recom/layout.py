"""The BGC tracer layout as a function of the REcoM switches (spectral port, lane P).

REcoM numbers its tracers by configuration (``initialize_tracer_indices``, fork ``recom_modules.F90:759-884``;
REcoM 692f804 ``recom_modules.F90:432-440, 702-705``): the 22 base tracers 1001-1022 always, then the optional
groups in a fixed order, so the slot of e.g. the microzooplankton moves when coccolithophores are on. Two REcoM
flavours share this module:

* **REcoM 692f804** (the submodule the port started from, ``bin_2p3z2d``): the 3zoo2det group 1023-1030 and
  ``1037 = DICremin``; no coccolithophores ported.
* **the spectral fork** (``sveta-loza/fesom2`` ``fesom2.6_recom_tra_diags_spectral``): 1023-1028 (Zoo2, DetZ2),
  coccolithophores 1029-1031 and Phaeocystis 1032-1034 (``enable_coccos``), microzooplankton 1035-1036, and with
  ``RECOM_CDOM`` one CDOM tracer whose ID is the next free one (1037 / 1031 / 1029 / 1023 by configuration).
  The fork has **no** DICremin. ``RECOM_MARSHALL`` (D1 tracers 1038-1041) is not ported.

:func:`layout_for` builds the :class:`Layout` of a switch set; :data:`DEFAULT` is the 692f804 ``bin_2p3z2d``
layout, identical to the module constants of :mod:`fesom_jax.recom.tracers` (asserted at import), so the
existing programs are unchanged. A layout is a hashable NamedTuple of tuples: it is a static argument.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

BASE_IDS: Tuple[int, ...] = tuple(range(1001, 1023))
BASE_NAMES: Tuple[str, ...] = (
    "DIN", "DIC", "Alk", "PhyN", "PhyC", "PhyChl", "DetN", "DetC", "HetN", "HetC",
    "DON", "DOC", "DiaN", "DiaC", "DiaChl", "DiaSi", "DetSi", "DSi", "DFe", "PhyCalc",
    "DetCalc", "O2")
ZOO2_NAMES = ("Zoo2N", "Zoo2C", "DetZ2N", "DetZ2C", "DetZ2Si", "DetZ2Calc")
COCCO_NAMES = ("CoccoN", "CoccoC", "CoccoChl", "PhaeoN", "PhaeoC", "PhaeoChl")
MICZOO_NAMES = ("MicZooN", "MicZooC")

# recom_remin_tracer_id without the ciso IDs (both flavours): the benthic return flux
REMIN_IDS: Tuple[int, ...] = (1001, 1002, 1003, 1018, 1019, 1022)
# bc_surface (oce_ale_tracer.F90): the BGC IDs with a surface flux and the stage-1 field that feeds it
BC_SURFACE_SOURCE: Tuple[Tuple[int, str], ...] = (
    (1001, "AtmNInput"), (1002, "GloCO2flux_seaicemask"), (1003, "relax_alk"), (1018, "zero"),
    (1019, "AtmFeInput"), (1022, "GloO2flux_seaicemask"))


class Layout(NamedTuple):
    """The tracer layout of one REcoM configuration (all tuples, so the layout is hashable and static)."""
    ids: Tuple[int, ...]                 # FESOM tracer IDs in State.bgc slot order (namelist.tra order)
    names: Tuple[str, ...]               # REcoM's state-variable names, same order
    enable_3zoo2det: bool
    enable_coccos: bool
    cdom: bool
    dicremin: bool                       # REcoM 692f804's 1037 (the fork has none)
    sinking_ids: Tuple[int, ...]         # recom_sinking_tracer_id of this configuration, Fortran order
    det_ids: Tuple[int, ...]             # recom_det_tracer_id (ballasting class 1)
    det2_ids: Tuple[int, ...]            # recom_det2_tracer_id (class 2; empty without 3zoo2det)
    cocco_ids: Tuple[int, ...]
    phaeo_ids: Tuple[int, ...]

    # -- slots --------------------------------------------------------------------------------------------
    @property
    def ntr(self) -> int:
        return len(self.ids)

    def slot(self, tracer_id: int) -> int:
        """The ``State.bgc`` slot of FESOM tracer ``tracer_id`` (KeyError for a tracer not in this layout)."""
        try:
            return self.ids.index(int(tracer_id))
        except ValueError:
            raise KeyError(f"tracer ID {tracer_id} is not a BGC tracer of this layout {self.ids}") from None

    def has(self, tracer_id: int) -> bool:
        return int(tracer_id) in self.ids

    def slots(self, tracer_ids) -> Tuple[int, ...]:
        return tuple(self.slot(t) for t in tracer_ids)

    @property
    def slot_map(self) -> Dict[int, int]:
        return {tid: s for s, tid in enumerate(self.ids)}

    def _named(self, name: str) -> int:
        return self.names.index(name)

    # REcoM's index names (0-based slots). The base 22 never move.
    idin, idic, ialk, iphyn, iphyc, ipchl, idetn, idetc, ihetn, ihetc = range(10)
    idon, idoc, idian, idiac, idchl, idiasi, idetsi, isi, ife, iphycal = range(10, 20)
    idetcal, ioxy = 20, 21

    @property
    def izoo2n(self): return self._named("Zoo2N")
    @property
    def izoo2c(self): return self._named("Zoo2C")
    @property
    def idetz2n(self): return self._named("DetZ2N")
    @property
    def idetz2c(self): return self._named("DetZ2C")
    @property
    def idetz2si(self): return self._named("DetZ2Si")
    @property
    def idetz2calc(self): return self._named("DetZ2Calc")
    @property
    def imiczoon(self): return self._named("MicZooN")
    @property
    def imiczooc(self): return self._named("MicZooC")
    @property
    def icocn(self): return self._named("CoccoN")
    @property
    def icocc(self): return self._named("CoccoC")
    @property
    def icchl(self): return self._named("CoccoChl")
    @property
    def iphan(self): return self._named("PhaeoN")
    @property
    def iphac(self): return self._named("PhaeoC")
    @property
    def iphachl(self): return self._named("PhaeoChl")
    @property
    def icdom(self): return self._named("CDOM")
    @property
    def idicremin(self): return self._named("DICremin")

    @property
    def cdom_id(self) -> Optional[int]:
        return self.ids[self.icdom] if self.cdom else None

    @property
    def bc_surface_source(self) -> Dict[int, str]:
        return dict(BC_SURFACE_SOURCE)

    @property
    def remin_ids(self) -> Tuple[int, ...]:
        return REMIN_IDS

    def describe(self) -> str:
        return (f"{self.ntr} BGC tracers: 3zoo2det={self.enable_3zoo2det} coccos={self.enable_coccos} "
                f"CDOM={self.cdom} DICremin={self.dicremin}; IDs {self.ids[0]}..{self.ids[-1]}")


def layout_for(*, enable_3zoo2det: bool, enable_coccos: bool, cdom: bool = False,
               dicremin: bool = False) -> Layout:
    """The layout of a switch set. ``dicremin`` is REcoM 692f804's DICremin tracer (never together with the
    fork's ``cdom``: the two flavours use ID 1037 differently)."""
    if cdom and dicremin:
        raise ValueError("cdom (the spectral fork) and dicremin (REcoM 692f804) are exclusive: both own ID 1037")
    ids = list(BASE_IDS)
    names = list(BASE_NAMES)
    det2: Tuple[int, ...] = ()
    cocco: Tuple[int, ...] = ()
    phaeo: Tuple[int, ...] = ()
    nxt = 1023
    if enable_3zoo2det:                                   # 1023-1028 Zoo2 N/C, DetZ2 N/C/Si/Calc
        ids += list(range(nxt, nxt + 6)); names += list(ZOO2_NAMES)
        det2 = tuple(range(nxt + 2, nxt + 6))
        nxt += 6
    if enable_coccos:                                     # cocco N/C/Chl, Phaeo N/C/Chl
        cocco = tuple(range(nxt, nxt + 3)); phaeo = tuple(range(nxt + 3, nxt + 6))
        ids += list(range(nxt, nxt + 6)); names += list(COCCO_NAMES)
        nxt += 6
    if enable_3zoo2det:                                   # MicZoo N/C (after the coccos, fork and 692f804 alike)
        ids += [nxt, nxt + 1]; names += list(MICZOO_NAMES)
        nxt += 2
    if cdom:
        ids.append(nxt); names.append("CDOM"); nxt += 1
    if dicremin:
        ids.append(1037); names.append("DICremin")
    # recom_sinking_tracer_id (fork recom_modules.F90:63-68; 692f804 :716-718): detritus, phyto, diatoms,
    # (det2), (coccos, phaeo); the list is global in Fortran and filtered by `any(id == ...)` per present tracer
    sinking = (1007, 1008, 1017, 1021, 1004, 1005, 1020, 1006, 1013, 1014, 1016, 1015) + det2 + cocco + phaeo
    return Layout(ids=tuple(ids), names=tuple(names), enable_3zoo2det=bool(enable_3zoo2det),
                  enable_coccos=bool(enable_coccos), cdom=bool(cdom), dicremin=bool(dicremin),
                  sinking_ids=tuple(i for i in sinking if i in ids), det_ids=(1007, 1008, 1017, 1021),
                  det2_ids=det2, cocco_ids=cocco, phaeo_ids=phaeo)


def layout_of(switches) -> Layout:
    """The layout of a :class:`fesom_jax.recom.params.RecomSwitches` (``spectral_fork`` selects the flavour)."""
    fork = bool(getattr(switches, "spectral_fork", False))
    return layout_for(enable_3zoo2det=switches.enable_3zoo2det, enable_coccos=switches.enable_coccos,
                      cdom=fork and bool(getattr(switches, "RECOM_CDOM", False)), dicremin=not fork)


# REcoM 692f804 bin_2p3z2d: the layout every existing program and test uses.
DEFAULT = layout_for(enable_3zoo2det=True, enable_coccos=False, dicremin=True)
# The fork's reference configuration (tmp_run_kdind15: 4 phytoplankton, 3 zooplankton, 2 detritus, CDOM).
SPECTRAL_4P3Z2D = layout_for(enable_3zoo2det=True, enable_coccos=True, cdom=True)


def _check_against_tracers_module() -> None:
    from . import tracers as T
    assert DEFAULT.ids == T.BGC_IDS and DEFAULT.names == T.NAMES and DEFAULT.ntr == T.NTR
    assert DEFAULT.sinking_ids == T.SINKING_IDS and DEFAULT.det2_ids == T.DET2_IDS
    assert (DEFAULT.izoo2n, DEFAULT.imiczooc, DEFAULT.idicremin) == (T.izoo2n, T.imiczooc, T.idicremin)
    assert SPECTRAL_4P3Z2D.ntr == 37 and SPECTRAL_4P3Z2D.ids[-1] == 1037 and SPECTRAL_4P3Z2D.icocn == 28
    assert SPECTRAL_4P3Z2D.imiczoon == 34 and SPECTRAL_4P3Z2D.icdom == 36


_check_against_tracers_module()
