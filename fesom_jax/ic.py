"""Initial conditions for the FESOM2 → JAX port.

Mirrors the C port's IC path (``fesom_ic.c`` + the call site in
``fesom_main.c:377-753``) for the **Phase-2 pi configuration**:

* constant ``T=10`` / ``S=35`` on every wet layer (``fesom_ic_tracers_constant``),
* **plus a Gaussian temperature blob** (``fesom_ic_tracer_T_blob``) that the C
  ``main`` adds whenever no PHC path is given — which the reference-dump run does
  not give. So the dump's IC is constant **+ blob**, not bare constant. This is
  load-bearing: every T/S-dependent gate (EOS at substep 1, hence pressure → PGF →
  momentum → …) must reproduce the blob. See ``docs/PORTING_LESSONS.md`` and
  ``docs/REFERENCE_RUNS.md``.
* rest thickness (``zbar`` differences, full-cell linfs) and zero dynamics, via
  :meth:`fesom_jax.state.State.rest`.

The blob (``fesom_main.c:744-753``): centre ``(lon0,lat0)=(−45°,40°)`` geographic,
horizontal ``σ=10°`` (with a ``cos(lat0)`` small-circle correction and a **4σ
cutoff**), vertical ``σ=300 m`` about ``z=Z[nz]`` (negative downward), amplitude
``+5 °C``, added **additively** to T on every wet layer; S is unchanged.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from .config import RAD
from .mesh import Mesh
from .state import State

# Default T-blob parameters — verbatim from fesom_main.c:747-752.
BLOB_DEFAULTS = dict(
    lon0_deg=-45.0,
    lat0_deg=40.0,
    sigma_deg=10.0,
    sigma_z=300.0,
    amp_C=5.0,
)


def _wrap180(x):
    """Wrap degrees into ``[-180, 180)`` (the C while-loop reduction)."""
    return jnp.mod(x + 180.0, 360.0) - 180.0


def tracer_T_blob(
    mesh: Mesh,
    T,
    *,
    lon0_deg: float,
    lat0_deg: float,
    sigma_deg: float,
    sigma_z: float,
    amp_C: float,
):
    """Add the Gaussian +``amp_C`` °C temperature anomaly to ``T`` (additive).

    Literal port of ``fesom_ic_tracer_T_blob`` (``fesom_ic.c:82-129``), vectorized.
    ``T`` is ``[nod2D, nl]``; returns the updated array (only wet layers change).
    """
    nl = mesh.nl
    inv_sig2_h = 1.0 / (sigma_deg * sigma_deg)
    inv_sig2_z = 1.0 / (sigma_z * sigma_z)
    lon0 = _wrap180(jnp.asarray(lon0_deg, jnp.float64))

    # geo_coord is (lon,lat) in radians → degrees; wrap lon like the C.
    lon = _wrap180(mesh.geo_coord_nod2D[:, 0] / RAD)          # (nod2D,)
    lat = mesh.geo_coord_nod2D[:, 1] / RAD                    # (nod2D,)

    dlon = lon - lon0
    dlon = jnp.where(dlon > 180.0, dlon - 360.0, dlon)
    dlon = jnp.where(dlon < -180.0, dlon + 360.0, dlon)
    dlat = lat - lat0_deg

    cos_lat0 = jnp.cos(lat0_deg * RAD)
    r2_h = (dlon * cos_lat0) ** 2 * inv_sig2_h + dlat * dlat * inv_sig2_h   # (nod2D,)

    # 4σ horizontal cutoff: nodes with r²_h > 16 get no contribution (C `continue`).
    prof_h = jnp.where(r2_h > 16.0, 0.0, jnp.exp(-0.5 * r2_h))             # (nod2D,)

    # vertical profile at mid-layer depths Z (pad to nl; below-bottom is masked off).
    Zp = jnp.concatenate([mesh.Z, mesh.Z[-1:]])                            # (nl,)
    prof_z = jnp.exp(-0.5 * Zp * Zp * inv_sig2_z)                          # (nl,)

    blob = amp_C * prof_h[:, None] * prof_z[None, :]                       # (nod2D, nl)
    blob = jnp.where(mesh.node_layer_mask, blob, 0.0)
    return jnp.asarray(T) + blob


def initial_state(mesh: Mesh, T0: float = 10.0, S0: float = 35.0, blob: bool = True) -> State:
    """Phase-2 pi initial state: rest thickness + constant ``T0``/``S0`` (masked to
    wet layers) + the default T-blob. ``T_old``/``S_old`` initialise to the same."""
    st = State.rest(mesh, T0, S0)
    T = jnp.where(mesh.node_layer_mask, st.T, 0.0)
    S = jnp.where(mesh.node_layer_mask, st.S, 0.0)
    if blob:
        T = tracer_T_blob(mesh, T, **BLOB_DEFAULTS)
    return dataclasses.replace(st, T=T, S=S, T_old=T, S_old=S)
