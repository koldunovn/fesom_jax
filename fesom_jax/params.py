"""Differentiable physics parameters — the ML-hook seam (Phase 3, Task 3.2).

The Phase-2 kernels read their tunables from module-level constants in
:mod:`fesom_jax.config` (``K_VER``, ``A_VER``, …). To take a gradient
``d(loss)/d(param)`` those tunables must instead be **traced leaves** flowing
through ``step``. :class:`Params` is that container: a small registered pytree of
the differentiable scalars, threaded ``step → pp.mixing_pp`` (and onward as the
physics grows).

This is also the first concrete **ML-hook seam**. Phase 7 swaps the PP mixing for
a trainable NN; that NN's weights live here too (alongside / replacing the scalar
backgrounds), so the same ``grad(loss)(params)`` call trains them. Keep it a
pytree (every field a leaf) so :func:`jax.grad` returns the same structure.

Phase 3 carried the PP vertical-mixing backgrounds (the **1st ML-hook**, mixing):

* ``k_ver`` — background tracer vertical diffusivity [m²/s]  (the plan's gradient
  target; enters ``Kv = mix·factor³ + k_ver`` additively → ``dKv/dk_ver = 1``,
  clean past the PP ``max(N²,0)`` kink);
* ``a_ver`` — background momentum vertical viscosity [m²/s]  (the sibling; enters
  ``Av`` and so the *within-step* CG path via ``impl_vert_visc``).

Phase 6B adds the GM/Redi eddy-diffusivity ceilings (the **2nd ML-hook**, eddy
fluxes — the parent plan's designated 2nd swap point):

* ``k_gm`` — GM thickness-diffusivity ceiling [m²/s]  (enters ``init_redi_gm``'s
  ``fer_K`` → the bolus streamfunction RHS → the bolus velocity);
* ``redi_kmax`` — Redi isoneutral-diffusivity ceiling [m²/s]  (enters ``Ki`` → the
  Redi neutral-diffusion flux + the K33 augmentation).

Both GM leaves carry **defaults** (the config constants) so the Phase-2/3 two-field
construction ``Params(k_ver=…, a_ver=…)`` stays valid and numerically identical
(when ``gm_cfg is None`` the GM leaves are simply unused → ``d/d(k_gm)=0``).

Phase 9b adds the CVMix classical-TKE constants (the **PRIMARY ML-hook seam** — TKE is
a prognostic mixing closure whose constants are exactly what Phase 7a tunes / Phase 7
NN-replaces, so ``Params``-exposure is first-class, not an add-on):

* ``tke_c_k`` — mixing-length→KappaM coefficient (``KappaM = c_k·mxl·√tke``);
* ``tke_c_eps`` — dissipation coefficient (the Patankar quasi-implicit ``c_eps·√tke/mxl``);
* ``tke_cd`` — surface-flux coefficient (Neumann BC ``cd·forc^{3/2}/dzt[0]``);
* ``tke_alpha`` — TKE self-diffusivity multiplier (``ke = alpha·½(KappaM[k+1]+KappaM[k])``).

Like the GM leaves they carry **defaults** so older constructions stay valid; when
``tke_cfg is None`` they are unused → ``d/d(tke_c_k)=0``.

:meth:`Params.defaults` reproduces the config constants exactly (so passing
``Params.defaults()`` is numerically identical to the un-parameterised path).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import tree_util

from .config import (A_VER, K_GM_MAX, K_VER, REDI_KMAX,
                     TKE_ALPHA, TKE_C_EPS, TKE_C_K, TKE_CD)


@dataclasses.dataclass(frozen=True)
class Params:
    """Differentiable physics parameters (a JAX pytree; every field a leaf)."""

    k_ver: jax.Array  # background tracer vertical diffusivity [m²/s]
    a_ver: jax.Array  # background momentum vertical viscosity [m²/s]
    # GM/Redi eddy diffusivities (2nd ML-hook, Phase 6B). Defaults so the older
    # two-arg `Params(k_ver=, a_ver=)` construction is unchanged.
    k_gm: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.asarray(K_GM_MAX, jnp.float64))
    redi_kmax: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.asarray(REDI_KMAX, jnp.float64))
    # CVMix classical-TKE constants (the PRIMARY ML-hook, Phase 9b). Defaults so the
    # older constructions are unchanged; unused (→ d/d=0) when tke_cfg is None.
    tke_c_k: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.asarray(TKE_C_K, jnp.float64))
    tke_c_eps: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.asarray(TKE_C_EPS, jnp.float64))
    tke_cd: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.asarray(TKE_CD, jnp.float64))
    tke_alpha: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.asarray(TKE_ALPHA, jnp.float64))

    @staticmethod
    def defaults() -> "Params":
        """The config-constant baseline as float64 scalar arrays (the value used
        when ``step``/``integrate`` are called with ``params=None``)."""
        return Params(
            k_ver=jnp.asarray(K_VER, jnp.float64),
            a_ver=jnp.asarray(A_VER, jnp.float64),
            k_gm=jnp.asarray(K_GM_MAX, jnp.float64),
            redi_kmax=jnp.asarray(REDI_KMAX, jnp.float64),
            tke_c_k=jnp.asarray(TKE_C_K, jnp.float64),
            tke_c_eps=jnp.asarray(TKE_C_EPS, jnp.float64),
            tke_cd=jnp.asarray(TKE_CD, jnp.float64),
            tke_alpha=jnp.asarray(TKE_ALPHA, jnp.float64),
        )


tree_util.register_dataclass(
    Params,
    data_fields=["k_ver", "a_ver", "k_gm", "redi_kmax",
                 "tke_c_k", "tke_c_eps", "tke_cd", "tke_alpha"],
    meta_fields=[],
)
