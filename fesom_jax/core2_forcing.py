"""DEPRECATED — moved to :mod:`fesom_jax.surface_forcing`. Import that instead.

The old name misinforms twice over: it is named after a **mesh** (CORE2) although the
module is entirely mesh-agnostic (it runs on pi, farc, dars, CORE2, NG5), and
"CORE2 forcing" is a real but *different* dataset (CORE-II, Large & Yeager) — whereas this
driver reads **JRA55-do**. In this repository "CORE2" names a mesh, never a forcing dataset.

Renames (old → new)::

    fesom_jax.core2_forcing              →  fesom_jax.surface_forcing
    core2_forcing.build_core_forcing     →  surface_forcing.build_surface_forcing
    core2_forcing.CoreForcing            →  surface_forcing.SurfaceForcing

Every other public name is unchanged: ``StepForcing``, ``ForcingStatic``,
``SurfaceFluxes``, ``compute_surface_fluxes``, ``dates_for_steps``, ``ice_ic_aice``.

This shim re-exports :mod:`fesom_jax.surface_forcing` verbatim — the old names are aliases
bound to the *same* objects, so existing code (e.g. long-running SLURM chains) keeps working
unchanged, with identical numerics. It emits a :class:`DeprecationWarning` on import and will
be removed in a future release.
"""

from __future__ import annotations

import warnings

from . import surface_forcing as _surface_forcing
from .surface_forcing import (  # noqa: F401  — unchanged public API, re-exported
    CD_OCE_ICE,
    RHO_CD,
    ForcingStatic,
    StepForcing,
    SurfaceFluxes,
    SurfaceForcing,
    build_surface_forcing,
    compute_surface_fluxes,
    dates_for_steps,
    ice_ic_aice,
)

warnings.warn(
    "fesom_jax.core2_forcing is deprecated and will be removed in a future release. "
    "It was renamed to fesom_jax.surface_forcing: the module is mesh-agnostic (not "
    "CORE2-specific) and drives JRA55-do, not CORE-II forcing. Also rename "
    "build_core_forcing -> build_surface_forcing and CoreForcing -> SurfaceForcing.",
    DeprecationWarning,
    stacklevel=2,
)

# Deprecated aliases — the SAME objects under their old names (no behaviour change).
CoreForcing = SurfaceForcing
build_core_forcing = build_surface_forcing

__all__ = [
    "CD_OCE_ICE",
    "RHO_CD",
    "CoreForcing",             # deprecated alias of SurfaceForcing
    "ForcingStatic",
    "StepForcing",
    "SurfaceFluxes",
    "SurfaceForcing",
    "build_core_forcing",      # deprecated alias of build_surface_forcing
    "build_surface_forcing",
    "compute_surface_fluxes",
    "dates_for_steps",
    "ice_ic_aice",
]


def __getattr__(name: str):
    """Forward anything else (private helpers, later additions) to the new module, so no
    pre-rename caller can break on an attribute this shim forgot to list."""
    try:
        return getattr(_surface_forcing, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} (deprecated alias of 'fesom_jax.surface_forcing') "
            f"has no attribute {name!r}") from None
