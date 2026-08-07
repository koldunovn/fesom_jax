"""``gm_nn`` — a small MLP producing a **bounded multiplier** on the GM/Redi diffusivities.

The Arm-A network of the NN-GM project (plan Task 7). Deliberately the same shape as
:mod:`fesom_jax.tke_nn`: a pure-JAX per-column MLP, arcsinh-normalized features, and a bounded
positive output, with ``zero_last`` making the untrained network bit-for-bit identical to
default GM. Read that module first; the differences from it are the interesting part.

Why NOT a CNN, given that is what the mesoscale-ML literature mostly uses
--------------------------------------------------------------------------
GZ21 (Guillaumin & Zanna 2021) and the Perezhogin/Zanna line of work predict subgrid forcing
with fully-convolutional networks. Two things stop that transferring here.

* **The mesh.** Those models run on structured grids where a 3x3 stencil is meaningful. FESOM
  is unstructured triangles: there is no stencil, and a CNN has nothing to slide over. The
  mesh-agnostic equivalent of a small-receptive-field CNN is a point-wise network on local
  invariants, which is what this is.
* **What is being predicted.** They predict a subgrid *momentum forcing*, which is a signed
  vector field that can and should inject energy in places (hence the backscatter and
  stochastic literature). Arm A predicts a bounded multiplier on a positive GM coefficient, so
  the parameterization stays an APE sink **by construction** whatever the network does. That is
  a much stronger safety property than a free CNN offers, and it is why there is no stochastic
  term here: additive noise on a multiplier would break exactly the guarantee that makes this
  arm safe to run unattended. A stochastic closure is a separate arm, not a variant of this one.

What DOES carry over from that literature
-----------------------------------------
* Non-dimensionalize inputs by local dynamical scales, or the network cannot generalize across
  regions — the ``arcsinh(x/scale)`` map, as in ``tke_nn``.
* Bound the output. An unbounded ML closure is a stability incident waiting for the right
  initial condition.
* Keep it TINY. ~1-2k weights is a hand-written matmul in the Fortran round-trip (Task 13);
  anything larger drags an ML runtime into FESOM, which the MOM6 implementations show is the
  expensive part.
* Offline skill does not predict online skill. This project's answer is to train online through
  the adjoint from the start (Task 9), so offline pretraining (Task 8) is only ever an
  initialization.

**Two-scale features — the one thing a point-wise net genuinely loses.** A CNN has a receptive
field; a strictly point-wise MLP has none, and eddy energy is not a local quantity even though
slope and stratification largely are. Rather than reach for a graph network, every feature can
be supplied at two scales: its local value and a mesh-Laplacian-smoothed value (the same filter
Task 5 uses for coarse-graining). That buys horizontal context, stays differentiable, stays
mesh-agnostic, and costs one extra smoothing per feature. Pass ``smoothed`` to
:func:`column_features` to use it; ``None`` gives the purely local net.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import tree_util

# Characteristic feature scales (SI) for the arcsinh normalization — typical ocean magnitudes
# so the MLP sees O(1) inputs. Static (NOT trained); chosen, not tuned.
_S0 = 1.0e-3     # isopycnal slope |S|                     [1]
_N20 = 1.0e-5    # buoyancy frequency N^2                  [1/s^2]
_GB0 = 1.0e-8    # lateral buoyancy gradient |grad_h b|    [1/s^2]
_F0 = 1.0e-4     # Coriolis f                              [1/s]
_D0 = 1.0e3      # bottom depth                            [m]
_M20 = 1.0e-10   # strain/shear invariant                  [1/s^2]

N_FEATURES_LOCAL = 7   # slope, N2, |grad_h b|, f, depth, depth fraction, strain
N_OUT = 2              # multipliers for (K_GM, K_Redi)


@dataclasses.dataclass(frozen=True)
class GmNN:
    """MLP weights as a JAX pytree leaf (the 2nd ML-hook, alongside ``TkeNN``).

    ``Ws``/``bs`` are per-layer weight matrices / bias vectors (data leaves, so they train);
    ``log_m_max`` is the static bound (meta, so it does not). Built by :func:`init_gm_nn`."""
    Ws: tuple        # tuple of [d_in, d_out] weight matrices
    bs: tuple        # tuple of [d_out] bias vectors
    log_m_max: float = 1.0986122886681098     # = log(3.0)


tree_util.register_dataclass(GmNN, data_fields=["Ws", "bs"], meta_fields=["log_m_max"])


def n_features(two_scale: bool = False) -> int:
    """Input width. Two-scale doubles it: each feature local AND mesh-Laplacian-smoothed."""
    return N_FEATURES_LOCAL * (2 if two_scale else 1)


def init_gm_nn(key, *, hidden: Sequence[int] = (16, 16), two_scale: bool = False,
               n_out: int = N_OUT, m_max: float = 3.0, w_scale: float = 0.5,
               zero_last: bool = True) -> GmNN:
    """Initialize a :class:`GmNN`.

    ``zero_last=True`` (the default) zeroes the final layer, so ``tanh(0)=0`` ⇒ multiplier ≡ 1
    ⇒ ``kappa = base * 1`` **bit-for-bit** ⇒ the untrained network reproduces default GM
    exactly. That is both the training start point and the deployment fallback, and it is what
    makes "the NN is wired in" a change no regression test can detect until it is trained."""
    sizes = [n_features(two_scale), *hidden, n_out]
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
    return GmNN(Ws=tuple(Ws), bs=tuple(bs), log_m_max=float(jnp.log(m_max)))


def mlp_raw(nn: GmNN, feats):
    """Forward MLP ``feats[..., n_features] → raw[..., n_out]`` — tanh hidden, LINEAR output.

    Linear output on purpose: the squashing lives in :func:`multiplier`, so the bound is applied
    in exactly one place and cannot be half-applied by a caller."""
    h = feats
    n = len(nn.Ws)
    for i, (W, b) in enumerate(zip(nn.Ws, nn.bs)):
        h = h @ W + b
        if i < n - 1:
            h = jnp.tanh(h)
    return h


def multiplier(nn: GmNN, feats):
    """Bounded positive multiplier ``m = exp(log_m_max * tanh(raw))`` ∈ [1/m_max, m_max].

    Positive for every finite input, so ``kappa = base * m`` keeps the sign of ``base`` and the
    GM term remains an APE sink no matter what the weights do. ``tanh`` saturates rather than
    clipping, so the gradient is small but never exactly zero at the bounds — a clip would make
    d(loss)/d(weights) structurally zero for any column pinned at a bound, which is the failure
    mode that cost this project a day on 2026-08-07 (see docs/PORTING_LESSONS.md)."""
    return jnp.exp(nn.log_m_max * jnp.tanh(mlp_raw(nn, feats)))


def column_features(slope, n2, grad_b, coriolis, depth, depth_frac, strain, *, smoothed=None):
    """Assemble the arcsinh-normalized feature matrix ``[..., n_features]``.

    Every feature is ``arcsinh(x/scale)``: finite and smooth for ALL finite inputs, including
    negative and zero, which a ``log`` is not. That matters because these fields carry masked
    lanes on an unstructured mesh and a single NaN poisons the whole reverse pass.

    ``smoothed`` — optional tuple of the SAME seven fields after mesh-Laplacian smoothing. When
    given, the output is the local features concatenated with the smoothed ones, giving the
    point-wise network the horizontal context a CNN would get from its receptive field. Keep the
    ordering identical between the two halves or the network sees a permuted input."""
    def _f(sl, n_2, gb, f, d, df, st):
        return jnp.stack([
            jnp.arcsinh(sl / _S0),
            jnp.arcsinh(n_2 / _N20),
            jnp.arcsinh(gb / _GB0),
            jnp.arcsinh(f / _F0),
            jnp.arcsinh(d / _D0),
            df,                              # already O(1) by construction: z/H ∈ [0,1]
            jnp.arcsinh(st / _M20),
        ], axis=-1)

    local = _f(slope, n2, grad_b, coriolis, depth, depth_frac, strain)
    if smoothed is None:
        return local
    if len(smoothed) != N_FEATURES_LOCAL:
        raise ValueError(
            f"smoothed must hold the same {N_FEATURES_LOCAL} fields as the local ones, in the "
            f"same order; got {len(smoothed)}")
    return jnp.concatenate([local, _f(*smoothed)], axis=-1)
