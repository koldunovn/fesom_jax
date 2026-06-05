"""Surface forcing for the FESOM2 → JAX port — Phase-2 analytical wind.

Port of ``fesom_forcing_set_analytical`` (``fesom_forcing_analytical.c``) **plus**
the per-step element re-averaging in ``fesom_ice_oce_fluxes_mom``
(``fesom_ice_coupling.c:256-264``, no-ice blend is a no-op but the re-average runs
every step before the ocean step — ``fesom_main.c:983``). Phase-2 pi config:
``tau0=0.05`` N/m², ``Ly_factor=2.0`` (``fesom_main.c:816``).

The raw element stress is a steady zonal cosine pattern:

    raw[e] = (−tau0·cos((2/Ly_factor)·lat_e), 0)

with ``lat_e`` the mean **geographic** latitude (radians) of the element's 3
vertices. But the stress that ``impl_vert_visc`` actually reads is **double-averaged**
(``oce_fluxes_mom``): raw element → **area-weighted node** average (``set_analytical``)
→ **simple mean of the 3 vertices** (``oce_fluxes_mom``). Skipping the re-average is
a ~5e-4 surface-velocity error. Use :func:`surface_stress`.
"""

from __future__ import annotations

import jax.numpy as jnp

from . import ops
from .mesh import Mesh


def raw_element_stress(mesh: Mesh, tau0: float = 0.05, Ly_factor: float = 2.0):
    """Raw analytical element wind stress ``[elem2D, 2]`` (the ``set_analytical``
    element write, before re-averaging). v-component is 0."""
    inv_period = 2.0 / Ly_factor
    lat = mesh.geo_coord_nod2D[:, 1]                       # (nod2D,) geographic radians
    lat_e = lat[mesh.elem_nodes].mean(axis=1)             # (elem2D,)
    sx = -tau0 * jnp.cos(inv_period * lat_e)
    return jnp.stack([sx, jnp.zeros_like(sx)], axis=-1)


def node_stress(mesh: Mesh, raw):
    """Area-weighted element→node stress average ``[nod2D, 2]`` (``set_analytical``
    node interpolation): ``sns[n] = Σ_{el∋n} area_el·raw[el] / Σ area_el``."""
    e = mesh.elem2D
    area = mesh.elem_area[:, None]                         # (elem2D,1)
    num = ops.scatter_add(jnp.broadcast_to((area * raw)[:, None], (e, 3, 2)),
                          mesh.elem_nodes, mesh.nod2D)      # (nod2D,2)
    den = ops.scatter_add(jnp.broadcast_to(mesh.elem_area[:, None], (e, 3)),
                          mesh.elem_nodes, mesh.nod2D)      # (nod2D,)
    safe = jnp.where(den > 0.0, den, 1.0)[:, None]
    return num / safe


def surface_stress(mesh: Mesh, tau0: float = 0.05, Ly_factor: float = 2.0):
    """Element surface stress ``[elem2D, 2]`` as ``impl_vert_visc`` reads it:
    raw → area-weighted node average → simple mean of the element's 3 vertices."""
    raw = raw_element_stress(mesh, tau0, Ly_factor)
    sns = node_stress(mesh, raw)
    return sns[mesh.elem_nodes].mean(axis=1)               # (elem2D,2)
