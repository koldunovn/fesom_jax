"""Structure-preserving NN-on-TKE closure (Paper-experiments Task A5 / §3 hybrid-ML pillar).

A small **pure-JAX** MLP maps local column features → a **bounded multiplier** on the
classical-TKE constants ``c_k`` / ``c_eps`` / ``c_d``. Three structural guarantees make it
deployable, not a toy:

1. **Bounded ⇒ positive-definite.** ``m = exp(s·tanh(raw))`` with ``s = log(m_max)`` lands in
   ``(1/m_max, m_max)`` and is **always positive**, so the scaled constants — and hence the TKE
   diffusivities — stay positive-definite for *any* weights/inputs (the structural-stability
   argument; no clamping, no NaN). Symmetric in log-space (a natural diffusivity multiplier).
2. **Zero last layer ⇒ identity.** With the final ``(W, b) = 0`` the pre-activation ``raw ≡ 0``
   ⇒ ``tanh(0)=0`` ⇒ ``m ≡ 1`` **exactly**, so ``c_k·m = c_k`` bit-for-bit → default TKE is
   recovered to the last ULP. This is BOTH the ``params=None`` regression invariant AND the
   deployment fallback net (an untrained / disabled NN never harms the model).
3. **Just a Params leaf.** The weights are a registered pytree (:class:`TkeNN`), so the *same*
   ``grad(loss)(params)`` that calibrates scalar constants trains the NN — "the NN is scalar
   calibration with ``c_k/c_eps`` promoted from constants to ``NN(state)``" (the design thesis).

~150 lines, no new dependency ⇒ Fortran-transferable (tier-3 transfer: NN→inference). Features
are normalized to O(1) by a fixed ``arcsinh`` map (finite for *every* finite input — required
so a degenerate/dry column can never inject a NaN that ``W=0`` would propagate as ``0·NaN``).
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import tree_util

# Characteristic feature scales (SI) for the arcsinh normalization — typical ocean magnitudes
# so the MLP sees O(1) inputs. Static (NOT trained); chosen, not tuned.
_TAU0 = 1.0e-4   # surface TKE forcing |τ|/ρ₀     [m²/s²]
_F0 = 1.0e-4     # Coriolis f                      [1/s]
_D0 = 1.0e3      # bottom depth                    [m]
_N20 = 1.0e-5    # buoyancy frequency N²           [1/s²]
_S20 = 1.0e-6    # vertical shear²                 [1/s²]
_E0 = 1.0e-4     # turbulent kinetic energy        [m²/s²]

N_FEATURES = 6
N_OUT = 3        # multipliers for (c_k, c_eps, c_d)


@dataclasses.dataclass(frozen=True)
class TkeNN:
    """MLP weights as a JAX pytree leaf (the ML-hook). ``Ws``/``bs`` are the per-layer weight
    matrices / bias vectors (data leaves); ``log_m_max`` = ``log(m_max)`` is the static bound
    (meta). Built by :func:`init_tke_nn`."""
    Ws: tuple        # tuple of [d_in, d_out] weight matrices
    bs: tuple        # tuple of [d_out] bias vectors
    log_m_max: float = 1.0986122886681098     # = log(3.0)


tree_util.register_dataclass(TkeNN, data_fields=["Ws", "bs"], meta_fields=["log_m_max"])


def init_tke_nn(key, *, hidden: Sequence[int] = (16, 16), n_features: int = N_FEATURES,
                n_out: int = N_OUT, m_max: float = 3.0, w_scale: float = 0.5,
                zero_last: bool = True) -> TkeNN:
    """Initialize a :class:`TkeNN`. ``zero_last=True`` (the default) sets the final layer to
    **zero** ⇒ multiplier ≡ 1 ⇒ default TKE recovered exactly (the training/deployment start
    point). Hidden layers use scaled-normal init (``w_scale/√fan_in``)."""
    sizes = [n_features, *hidden, n_out]
    keys = jax.random.split(key, len(sizes) - 1)
    Ws, bs = [], []
    for i, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        if last and zero_last:
            W = jnp.zeros((din, dout), jnp.float64)
        else:
            W = w_scale * jax.random.normal(keys[i], (din, dout), jnp.float64) / jnp.sqrt(din)
        Ws.append(W)
        bs.append(jnp.zeros((dout,), jnp.float64))
    return TkeNN(Ws=tuple(Ws), bs=tuple(bs), log_m_max=float(jnp.log(m_max)))


def mlp_raw(nn: TkeNN, feats):
    """Forward MLP ``feats[N, n_features] → raw[N, n_out]`` (tanh hidden layers, **linear**
    output). With a zero final layer ``raw ≡ 0``."""
    h = feats
    n = len(nn.Ws)
    for i in range(n):
        h = h @ nn.Ws[i] + nn.bs[i]
        if i < n - 1:
            h = jnp.tanh(h)
    return h


def multiplier(nn: TkeNN, feats):
    """Bounded, positive multiplier ``m[N, n_out] = exp(log_m_max · tanh(raw))`` ∈
    ``(1/m_max, m_max)``. ``raw=0`` (zero last layer) ⇒ ``m=1`` exactly. Always finite and
    positive for any finite ``feats`` ⇒ positive-definite scaled diffusivities."""
    raw = mlp_raw(nn, feats)
    return jnp.exp(nn.log_m_max * jnp.tanh(raw))


def column_features(forc_tke_surf, coriolis, depth, n2_col, shear2_col, tke_col, interior):
    """Assemble + normalize the per-column feature matrix ``[N, N_FEATURES]`` for the NN.

    Per-node scalars: ``forc_tke_surf`` ``|τ|/ρ₀`` ``[N]``, ``coriolis`` f ``[N]``, ``depth``
    bottom depth ``[N]``. Columns: ``n2_col``/``shear2_col`` (``bvfreq2``/``vshear2``,
    ``[N, nl]``) reduced to their **interior mean** (``interior`` bool mask, count-guarded so a
    dry column is finite, not 0/0), ``tke_col`` ``[N, nl]`` sampled at the surface. Every
    feature is ``arcsinh(x/scale)`` — finite and smooth for **all** finite inputs (the masked-NaN
    guarantee: no dry/degenerate column can inject a NaN that a zero last layer would turn into
    ``0·NaN``)."""
    cnt = jnp.maximum(jnp.sum(interior, axis=1), 1.0)              # [N], ≥1
    n2 = jnp.sum(jnp.where(interior, n2_col, 0.0), axis=1) / cnt   # interior-mean N²
    sh = jnp.sum(jnp.where(interior, shear2_col, 0.0), axis=1) / cnt
    e0 = tke_col[:, 0]                                             # surface TKE
    feats = jnp.stack([
        jnp.arcsinh(forc_tke_surf / _TAU0),
        jnp.arcsinh(coriolis / _F0),
        jnp.arcsinh(depth / _D0),
        jnp.arcsinh(n2 / _N20),
        jnp.arcsinh(sh / _S20),
        jnp.arcsinh(e0 / _E0),
    ], axis=1)                                                    # [N, N_FEATURES]
    return feats
