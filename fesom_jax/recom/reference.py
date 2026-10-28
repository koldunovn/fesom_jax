"""The physics-matched Fortran twin as a JAX starting point (gates G4b and G5, lane W).

``recom_ref/pi_30d_jan_jaxphys`` (``docs/recom/FORTRAN_REFERENCE.md``) is FESOM2 + REcoM with the physics of
``configs/pi_recom.yaml``. :func:`twin_step0_state` reads its cold start from the step-1 dumps: the physics as
``scripts/recom/physics_floor.py`` (T and S from ``recom_in``, their AB2 history from ``tra_in``, the rest
thicknesses, the seeded sea ice), the BGC tracers from ``recom_in`` (the IC after ``recom_init``; the ice step does
not touch them), their AB2 history ``valuesold`` from ``tra_in`` (FESOM2's IC before ``recom_init``), ``Benthos`` and
``Alk_surf`` from ``recom_in``, the carbonate cache zero (as Fortran's) and ``mstep`` 0.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import jax
import numpy as np

from . import tracers as T

TWIN = Path(os.environ.get("FESOM_RECOM_TWIN", "/work/ab0995/a270088/recom_ref/pi_30d_jan_jaxphys"))


def pad_levels(a, nl):
    """A Fortran layer field ``(n, nl-1[, ...])`` into ``nl`` levels (zero below)."""
    a = np.asarray(a, np.float64)
    if a.shape[1] == nl:
        return a
    pad = [(0, 0)] * a.ndim
    pad[1] = (0, nl - a.shape[1])
    return np.pad(a, pad)


def twin_step0_state(mesh, d1, ale_cfg, rcfg):
    """The twin's step-0 State (host numpy) from its step-1 dump ``d1`` (``io_dump.load_recom_dump`` dict)."""
    from .. import ice
    from ..state import State
    from .wiring import segment_fields
    nl = int(mesh.nl)
    mask = np.asarray(mesh.node_layer_mask)
    st = jax.tree.map(np.asarray, State.rest(mesh, ale_cfg=ale_cfg))
    T_ = np.where(mask, pad_levels(d1["recom_in.tr1"], nl), 0.0)
    S_ = np.where(mask, pad_levels(d1["recom_in.tr2"], nl), 0.0)
    T_old = np.where(mask, pad_levels(d1["tra_in.tr1.valuesold"], nl), 0.0)
    S_old = np.where(mask, pad_levels(d1["tra_in.tr2.valuesold"], nl), 0.0)
    st = dataclasses.replace(st, T=T_, S=S_, T_old=T_old, S_old=S_old)
    st = ice.seed_ice(st, mesh, st.T[:, 0], xp=np)
    bgc = np.stack([pad_levels(d1[f"recom_in.tr{i}"], nl) for i in T.BGC_IDS], axis=-1)
    bgc_old = np.stack([pad_levels(d1[f"tra_in.tr{i}.valuesold"], nl) for i in T.BGC_IDS], axis=-1)
    st = dataclasses.replace(st, bgc=bgc, bgc_old=bgc_old, bgc_benthos=np.asarray(d1["recom_in.Benthos"]).T.copy(),
                             bgc_alk_surf=np.asarray(d1["recom_in.Alk_surf"], np.float64),
                             **segment_fields(mesh.nod2D, nl - 1, rcfg, np))
    return st


def dump_names(step_end: bool = True):
    """The dump records :func:`twin_step0_state` and the step comparisons read."""
    r = ["mesh.gid_nod", "mesh.nlevels_nod2D", "mesh.geo_coord_nod2D", "recom_in.tr1", "recom_in.tr2",
         "tra_in.tr1.valuesold", "tra_in.tr2.valuesold", "recom_in.Benthos", "recom_in.Alk_surf"]
    for i in T.BGC_IDS:
        r += [f"recom_in.tr{i}", f"tra_in.tr{i}.valuesold"]
    if step_end:
        r += ["step_end.Benthos"] + [f"step_end.tr{i}" for i in (1, 2) + T.BGC_IDS]
        r += ["recom_out.GloCO2flux", "recom_out.GloPCO2surf", "recom_out.GloHplus", "recom_out.NPPn",
              "recom_out.NPPd", "recom_out.relax_alk", "recom_out.Benthos"]
        r += [f"recom_out.tr{i}" for i in T.BGC_IDS]
    return r
