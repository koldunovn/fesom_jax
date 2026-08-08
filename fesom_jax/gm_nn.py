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

    This is the ONE place the feature list is defined and ordered, and every producer must come
    through it: the model seam (:func:`fesom_jax.gm.gm_nn_multipliers`), the offline fit's
    product-derived features, and the twin. Two orderings, or two sets of arcsinh scales, is a
    silent train/deploy skew that no test on either side alone would catch.

    ``strain`` is currently passed as **0.0** by the seam — the strain/shear invariant needs the
    element velocities threaded into the GM coefficient block, which it does not receive, so the
    feature is deferred to the plan's D2 screen. It is kept in the layout rather than dropped so
    that a weight file trained now still loads if strain lands later; the cost is that its
    first-layer row sees no gradient until then.

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


# ============================================================================
# Weight (de)serialization — the deployment path
# ============================================================================
def save_gm_nn(path, nn: GmNN) -> None:
    """Write ``nn`` to an ``.npz``: ``W0..W{n-1}``, ``b0..b{n-1}``, ``log_m_max``, ``hidden``,
    ``two_scale``.

    The file format IS the deployment interface: training writes it, a forward run reads it
    through ``RunConfig.params_file``, and a restart-leg chain re-reads it on every leg.

    **``log_m_max`` must be in the file.** It is a ``meta`` field of the pytree precisely so the
    optimiser cannot train the safety bound away — which means it is NOT among the weights, and
    an npz of weights alone would silently rebuild with the class default. A network trained
    under ``m_max=3`` reloaded as ``m_max=3`` is the whole safety argument; reloaded as anything
    else it is a different closure wearing the same weights.

    ``hidden``/``two_scale`` are redundant with the weight shapes on purpose: they let
    :func:`load_gm_nn` catch a file whose shapes and intent disagree, rather than discovering it
    at the first matmul inside a jitted step."""
    import numpy as np
    arrays = {f"W{i}": np.asarray(W, np.float64) for i, W in enumerate(nn.Ws)}
    arrays.update({f"b{i}": np.asarray(b, np.float64) for i, b in enumerate(nn.bs)})
    arrays["log_m_max"] = np.asarray(nn.log_m_max, np.float64)
    arrays["hidden"] = np.asarray([W.shape[1] for W in nn.Ws[:-1]], np.int64)
    arrays["two_scale"] = np.asarray(nn.Ws[0].shape[0] == n_features(True), bool)
    np.savez(path, **arrays)


def load_gm_nn(path) -> GmNN:
    """Read a :class:`GmNN` back from :func:`save_gm_nn`'s ``.npz`` — float64, exact, and with
    the same treedef (``log_m_max`` included) as the object that was saved.

    Validates the recorded ``hidden``/``two_scale`` against the actual weight shapes and raises
    on disagreement: a weight file is the one artefact that outlives the session that made it,
    so it either reconstructs the network it describes or it fails here."""
    import numpy as np
    with np.load(path) as z:
        n = sum(1 for k in z.files if k.startswith("W"))
        if n == 0 or any(f"b{i}" not in z.files for i in range(n)):
            raise ValueError(f"{path}: not a gm_nn weight file (found keys {sorted(z.files)})")
        Ws = tuple(jnp.asarray(z[f"W{i}"], jnp.float64) for i in range(n))
        bs = tuple(jnp.asarray(z[f"b{i}"], jnp.float64) for i in range(n))
        if "hidden" in z.files:
            hidden = tuple(int(h) for h in z["hidden"])
            actual = tuple(int(W.shape[1]) for W in Ws[:-1])
            if hidden != actual:
                raise ValueError(f"{path}: recorded hidden sizes {hidden} disagree with the "
                                 f"weight shapes {actual}")
        if "two_scale" in z.files:
            want = n_features(bool(z["two_scale"]))
            if int(Ws[0].shape[0]) != want:
                raise ValueError(f"{path}: two_scale={bool(z['two_scale'])} implies "
                                 f"{want} input features, but W0 takes {Ws[0].shape[0]}")
        return GmNN(Ws=Ws, bs=bs, log_m_max=float(z["log_m_max"]))
