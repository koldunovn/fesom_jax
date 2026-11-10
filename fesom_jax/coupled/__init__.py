"""fesom_jax as the ocean / sea-ice component of the coupled AWI-CM JAX system (awicm_jax).

Opt-in: nothing in the standalone (file-forced) model imports this package. The API (docs/coupling/awicm.md):

* :func:`init_ocean` -> ``(state, static)``: the CORE2 ocean from a restart (or the PHC cold start), on one device
  or (``partition="dist_N"``) on the first N devices of the process;
* :func:`ocean_interval` -> ``state``: run one coupling interval (``nsteps`` ocean steps) with the atmosphere's
  fields on the FESOM nodes, ``mode="bulk"`` (B, near-surface state -> fesom_jax's bulk formulae) or ``"flux"``
  (F, the fluxes as AWI-ESM3's FESOM2 receives them);
* :func:`ocean_to_atm` -> ``{name: (nod2D,)}``: the eight fields FESOM2's ``cpl_driver`` sends (``sst_feom`` ...);
* :func:`global_view`: State fields in global node order (the owner lanes gathered with ``dist_N``);
* :func:`write_ocean_restart` (:func:`ocean_restart_writer`: the State and calendar now, the store later); the monthly
  means are written by :func:`ocean_interval` when ``monthly_out`` is set (``OceanStatic.submit``: on a writer thread).
"""

from .flux import OASIS_TO_FESOM, oasis_to_fesom
from .ocean import (MODE_B_FIELDS, MODE_F_FIELDS, OCEAN_TO_ATM, OceanStatic, flush_monthly, global_view,
                    init_ocean, ocean_interval, ocean_restart_writer, ocean_to_atm,
                    write_ocean_restart)

__all__ = ["OASIS_TO_FESOM", "oasis_to_fesom", "MODE_B_FIELDS", "MODE_F_FIELDS", "OCEAN_TO_ATM", "OceanStatic", "flush_monthly", "global_view",
           "init_ocean",
           "ocean_interval", "ocean_restart_writer", "ocean_to_atm", "write_ocean_restart"]
