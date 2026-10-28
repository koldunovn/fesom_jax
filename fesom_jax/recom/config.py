"""The REcoM run configuration (``RunConfig.recom``; ``None`` ⇒ REcoM off, the physics program unchanged).

Two modes:

* **passive** (``passive_tracers > 0``, Task 1): the BGC transport without sources or sinks, for the transport
  cost ceiling (Task 2) and the transport identity tests.
* **full** (``namelist`` set, Task 6): REcoM's ``bin_2p3z2d`` ecosystem, its switches and parameters read from a
  ``namelist.recom`` (:mod:`fesom_jax.recom.params`), the 31 tracers of :mod:`fesom_jax.recom.tracers`. Every
  switch outside the ported configuration raises ``NotImplementedError`` in :meth:`RecomConfig.validate`.

The two are exclusive; the step dispatches on :attr:`RecomConfig.mode`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple, Optional

PASSIVE_BC = ("zero", "salt")
IC_SETS = ("pi", "core2", "core2_present")
# The BGC fields a run can write into its mean-output streams (fesom_jax.recom.wiring.output_fields): 3-D tracers
# (Chl = PhyChl + DiaChl) and the 2-D diagnostics of State.bgc_diag
OUTPUT_FIELDS = ("DIC", "Alk", "DIN", "DSi", "DFe", "O2", "Chl", "co2flux", "pco2surf", "npp", "export100", "ph",
                 "o2flux")


class RecomConfig(NamedTuple):
    """``passive_tracers``: carry this many passive tracers through the BGC tracer stage (FCT advection, Redi + K33,
    implicit vertical diffusion) with no sources or sinks. ``passive_bc``: their surface flux, ``"zero"`` or
    ``"salt"`` (the salinity BC ``bc_S``, so a stack of salinity copies must reproduce ``S`` — the transport
    identity test). ``namelist``: path of a REcoM ``namelist.recom`` ⇒ the full REcoM mode (then no passive
    tracers). Neither set raises ``NotImplementedError`` (nothing to run).

    ``bgc_every`` (k, default 1 = FESOM2): step the BGC tracers once every k physics steps, with ``k·dt`` and the
    k-step mean of the transport the physics used (the ``ice_ave_steps`` idea applied to the BGC tracers; a stated
    deviation, user 2026-10-25). The mean mass fluxes and the thicknesses at the interval ends keep the BGC
    inventories exact under zstar. k = 1 is the every-step program unchanged. Both modes. ⚠️ With full REcoM, k > 1
    keeps the inventories (CORE2 month, k = 4: <= 2e-5) but NOT the biology (NPP -8 %, CO2 flux -6 %, export -14 %
    after one January, ``docs/recom/VALIDATION.md``): a transport shortcut, not a faithful mode; production is k = 1
    (user 2026-10-26)."""
    passive_tracers: int = 0
    passive_bc: str = "zero"
    namelist: Optional[str] = None
    bgc_every: int = 1
    # full mode only (fesom_jax.recom.wiring): the cold-start IC set ("pi": ic.PI_IC_FILES, the 5-degree files of
    # FESOM2's test/input/recom; "core2": ic.CORE2_IC_FILES, preindustrial DIC; "core2_present": the same with the
    # present-day GLODAP DIC, ic.CORE2_PRESENT_IC_FILES) under ic_dir (FESOM's ClimateDataPath); the BGC fields of
    # the run's mean-output streams (OUTPUT_FIELDS); budget: carry the per-column budget sums (State.bgc_budget)
    ic_set: str = "pi"
    ic_dir: Optional[str] = None
    output: tuple = ("DIC", "Alk", "DIN", "DSi", "DFe", "O2", "Chl", "co2flux", "pco2surf", "npp", "export100")
    budget: bool = False

    @property
    def mode(self) -> str:
        """``"full"`` (REcoM from a namelist) or ``"passive"``."""
        return "full" if self.namelist is not None else "passive"

    @property
    def ntr(self) -> int:
        """The number of BGC tracers the State carries (the trailing axis of ``State.bgc``)."""
        if self.namelist is not None:
            from .tracers import NTR
            return NTR
        return int(self.passive_tracers)

    def namelist_path(self) -> str:
        """``namelist`` as given, or relative to the repository root when it does not exist relative to the cwd."""
        from pathlib import Path
        p = Path(self.namelist)
        if not p.is_absolute() and not p.exists():
            q = Path(__file__).resolve().parents[2] / p
            if q.exists():
                return str(q)
        return str(p)

    def load(self):
        """``(switches, params)`` of the namelist (:func:`fesom_jax.recom.params.load_namelist`, checked; cached)."""
        if self.namelist is None:
            raise ValueError("recom.namelist is not set (the passive mode has no REcoM parameters)")
        return _load_cached(self.namelist_path())

    def validate(self) -> "RecomConfig":
        if int(self.bgc_every) < 1:
            raise ValueError(f"recom.bgc_every must be >= 1, got {self.bgc_every}")
        if self.namelist is not None:
            if self.passive_tracers:
                raise ValueError("recom.namelist (full REcoM) and recom.passive_tracers are exclusive")
            sw, _ = self.load()                     # NotImplementedError for an out-of-scope switch
            if sw.bgc_num != self.ntr:
                raise ValueError(f"namelist bgc_num {sw.bgc_num} != the {self.ntr} tracers of fesom_jax.recom.tracers")
            if self.ic_set not in IC_SETS:
                raise ValueError(f"recom.ic_set must be one of {IC_SETS}, got {self.ic_set!r}")
            if not isinstance(self.output, tuple):
                raise TypeError("recom.output must be a tuple (RunConfig.from_dict converts a YAML list)")
            bad = sorted(set(self.output) - set(OUTPUT_FIELDS))
            if bad:
                raise ValueError(f"recom.output: unknown fields {bad}; allowed {list(OUTPUT_FIELDS)}")
            return self
        if self.passive_tracers <= 0:
            raise NotImplementedError(
                "recom needs a namelist (full REcoM, plan 20261025-recom-port) or passive_tracers > 0 "
                "(the passive transport mode)")
        if self.passive_bc not in PASSIVE_BC:
            raise ValueError(f"recom.passive_bc must be one of {PASSIVE_BC}, got {self.passive_bc!r}")
        return self


@lru_cache(maxsize=8)
def _load_cached(path: str):
    from .params import load_namelist
    return load_namelist(path, check=True)
