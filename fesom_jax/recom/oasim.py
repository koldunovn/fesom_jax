"""OASIM's spectral surface irradiance for the spectral fork (lane F of ``docs/plans/completed/20261028-recom-spectral-port.md``).

The fork reads, at every monthly update (``gen_surface_forcing.F90:1800-1822`` of ``sveta-loza/fesom2``
``fesom2.6_recom_tra_diags_spectral``), the direct ``Ed`` and diffuse ``Es`` irradiance just below the surface
[W m-2] per band from ``<OASIM_path>/Ed_<lambda>_<OASIM_file_pattern>`` and ``Es_...`` (13 files each, one monthly
climatology of 12 records on a 1-degree lon/lat grid), through ``read_other_NetCDF(..., check_dummy=.true.,
do_onvert=.true.)``: the missing values filled by the expanding neighbourhood mean, then bilinearly interpolated to
the nodes (``interp_2d_field``). The held month follows the same rule as the Albani dust (``update_monthly_flag``,
``i = month`` at ``mstep = 1``, else ``month + 1``): :class:`fesom_jax.surface_forcing.BgcForcing` holds both.

:func:`read_oasim_clim` is that read for all 12 months and 13 bands at once (host numpy), through the port of
``read_other_NetCDF`` in :mod:`fesom_jax.sss_runoff` (the SSS climatology takes the same route).
"""

from __future__ import annotations

import os

import numpy as np

from .. import sss_runoff
from .spectral import PWAVES, TLAM

RAD = np.pi / 180.0


def oasim_files(path, pattern: str) -> dict:
    """``{"Ed": [13 paths], "Es": [13 paths]}`` of the OASIM climatology under ``path`` (``OASIM_path``) for the
    ``OASIM_file_pattern`` ``pattern``, in band order."""
    return {kind: [os.path.join(str(path), f"{kind}_{pw}_{pattern}") for pw in PWAVES] for kind in ("Ed", "Es")}


def read_oasim_clim(mesh, path, pattern: str, ed_var: str = "Ed", es_var: str = "Es"):
    """``(ed, es)``, each ``(12, nod2D, 13)`` float64: the monthly OASIM ``Ed``/``Es`` [W m-2] per band on the nodes
    of ``mesh`` (its geographic coordinates, longitudes wrapped to ``[0, 360)`` as the Fortran does)."""
    geo = np.asarray(mesh.geo_coord_nod2D, dtype=np.float64) / RAD
    lon_mod = np.where(geo[:, 0] < 0.0, geo[:, 0] + 360.0, geo[:, 0])
    lat_mod = geo[:, 1].copy()
    files = oasim_files(path, pattern)
    n = lon_mod.shape[0]
    out = {}
    for kind, var in (("Ed", ed_var), ("Es", es_var)):
        a = np.zeros((12, n, TLAM), np.float64)
        for b, f in enumerate(files[kind]):
            if not os.path.isfile(f):
                raise FileNotFoundError(f"OASIM {kind} band {PWAVES[b]} nm: {f}")
            for m in range(12):
                a[m, :, b] = sss_runoff._read_other_netcdf(f, var, m + 1, lon_mod, lat_mod, check_dummy=True)
        out[kind] = a
    return out["Ed"], out["Es"]
