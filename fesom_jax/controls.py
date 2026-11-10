"""Conditioned (non-dimensional) control variables for parameter calibration.

A calibration optimises a vector ``u`` of non-dimensional controls, not the physical constants
themselves. Each control has a first guess ``p0`` and a multiplicative uncertainty ``sigma``:

    p = p0 · exp(sigma · u),        u = ln(p / p0) / sigma,

so ``u = 0`` is the first guess, ``u = ±1`` is one standard deviation (a factor ``exp(±sigma)``),
and ``p`` stays positive for every finite ``u``. The prior is standard normal on ``u``,

    J_prior(u) = ½ · uᵀu,           ∇J_prior = u,        ∇²J_prior = I,

which is the ECCO control formulation (non-dimensional controls scaled by a prior uncertainty,
with an identity prior Hessian): it puts constants of very different size and unit (``c_k`` ≈ 0.1,
``alpha`` ≈ 30) on one scale, so a quasi-Newton optimiser starts from a well-conditioned problem,
and "is this constant constrained by the data?" becomes "is the data curvature large compared with
the prior's curvature of 1?". The gradient in ``u`` follows from the chain rule,

    ∂J/∂u_i = sigma_i · p_i · ∂J/∂p_i        (+ u_i from the prior),

and JAX produces it directly when :meth:`ControlSet.to_params` is called INSIDE the differentiated
function (``jax.grad(lambda u: J(cs.to_params(u)))``).

Usage::

    cs = tke_control_set()                        # c_k, c_eps, cd, alpha; sigma = ln 2 each
    u = jnp.zeros(cs.n, jnp.float64)              # the first guess
    params = cs.to_params(u)                      # a full Params, the four TKE leaves replaced
    cs.check(np.asarray(u))                       # host-side domain check before each evaluation

``ControlSet`` is a frozen, hashable dataclass, so it can be a static argument of ``jax.jit`` and
the key of an executable cache. u must enter jit as a strong float64 array of fixed shape
(``jnp.asarray(u, jnp.float64)``): a weakly typed Python scalar changes type after the first
update and recompiles the whole model.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
import numpy as np

from .ice import ICE_CTRL_FIELDS, IceConfig
from .params import Params

# Params leaves that are pytrees (neural networks, field dicts, the ice-control dict), not scalar
# constants: never a control name here.
_NON_SCALAR_FIELDS = frozenset({"tke_nn", "gm_nn", "surface_adj", "ice_ctrl"})
_SCALAR_FIELDS = tuple(f.name for f in dataclasses.fields(Params)
                       if f.name not in _NON_SCALAR_FIELDS)

#: Prefix of the sea-ice controls: ``ice.<IceConfig field>`` (a float field of
#: :data:`fesom_jax.ice.ICE_CTRL_FIELDS`); ``to_params`` collects them into ``Params.ice_ctrl``.
ICE_PREFIX = "ice."
_ICE_NAMES = tuple(ICE_PREFIX + f for f in ICE_CTRL_FIELDS)

#: Largest |u| accepted by :meth:`ControlSet.check`. At sigma = ln 2 this is a factor 2^50 ≈ 1e15
#: from the first guess; beyond it exp overflows for larger sigma, or the constant is meaningless.
U_MAX = 50.0

#: Default multiplicative uncertainty: one standard deviation = a factor of 2 either way.
SIGMA_DEFAULT = math.log(2.0)

#: Physical domain used by :meth:`ControlSet.in_domain`: the largest albedo, and the factor
#: within which every other control must stay of its first guess.
ALBEDO_MAX = 0.95
DOMAIN_FACTOR = 10.0


@dataclasses.dataclass(frozen=True)
class LogNormalControl:
    """One positive scalar constant ``name`` as ``p0·exp(sigma·u)``: a scalar :class:`Params`
    field, or ``ice.<field>`` for a float :class:`~fesom_jax.ice.IceConfig` field (carried in
    ``Params.ice_ctrl``)."""

    name: str
    p0: float
    sigma: float

    def __post_init__(self):
        if self.name.startswith(ICE_PREFIX):
            if self.name not in _ICE_NAMES:
                raise ValueError(f"{self.name!r}: {self.name[len(ICE_PREFIX):]!r} is not a float "
                                 f"IceConfig field that can be a control; expected one of "
                                 f"{ICE_CTRL_FIELDS}")
        elif self.name not in _SCALAR_FIELDS:
            raise ValueError(f"{self.name!r} is not a scalar Params field; "
                             f"expected one of {_SCALAR_FIELDS} or 'ice.<IceConfig field>'")
        p0, sigma = float(self.p0), float(self.sigma)
        if not (math.isfinite(p0) and p0 > 0.0):
            raise ValueError(f"{self.name}: first guess p0 must be finite and > 0, got {self.p0!r}")
        if not (math.isfinite(sigma) and sigma > 0.0):
            raise ValueError(f"{self.name}: sigma must be finite and > 0, got {self.sigma!r}")
        # Plain Python floats, so equal specs hash equal whatever numeric type the caller passed.
        object.__setattr__(self, "p0", p0)
        object.__setattr__(self, "sigma", sigma)

    def to_physical(self, u):
        """``p0·exp(sigma·u)``; traceable (jnp) — call it inside the differentiated function."""
        return self.p0 * jnp.exp(self.sigma * u)

    def to_u(self, p):
        """``ln(p/p0)/sigma`` on the host (numpy float64). Raises for p ≤ 0 or non-finite p."""
        p = np.asarray(p, np.float64)
        if not np.all(np.isfinite(p)) or np.any(p <= 0.0):
            raise ValueError(f"{self.name}: physical value must be finite and > 0, got {p!r}")
        return np.log(p / self.p0) / self.sigma


@dataclasses.dataclass(frozen=True)
class ControlSet:
    """An ordered set of :class:`LogNormalControl`; component ``i`` of ``u`` drives ``controls[i]``."""

    controls: tuple[LogNormalControl, ...]

    def __post_init__(self):
        ctrls = tuple(self.controls)
        if not ctrls:
            raise ValueError("ControlSet needs at least one control")
        for c in ctrls:
            if not isinstance(c, LogNormalControl):
                raise TypeError(f"expected LogNormalControl, got {type(c).__name__}")
        names = [c.name for c in ctrls]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate control names: {names}")
        object.__setattr__(self, "controls", ctrls)

    @property
    def n(self) -> int:
        return len(self.controls)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.controls)

    @property
    def p0(self) -> np.ndarray:
        return np.array([c.p0 for c in self.controls], np.float64)

    @property
    def sigma(self) -> np.ndarray:
        return np.array([c.sigma for c in self.controls], np.float64)

    def to_physical(self, u):
        """The physical values ``[n]`` for ``u[n]`` (traceable)."""
        u = jnp.asarray(u, jnp.float64)
        return jnp.asarray(self.p0) * jnp.exp(jnp.asarray(self.sigma) * u)

    def to_u(self, p) -> np.ndarray:
        """The controls ``[n]`` for physical values ``p[n]`` (host, numpy float64)."""
        p = np.asarray(p, np.float64).reshape(-1)
        if p.shape != (self.n,):
            raise ValueError(f"expected {self.n} physical values, got shape {p.shape}")
        return np.array([c.to_u(p[i]) for i, c in enumerate(self.controls)], np.float64)

    def to_params(self, u, base: Params | None = None) -> Params:
        """``base`` (default :meth:`Params.defaults`) with each control's field replaced by
        ``p0·exp(sigma·u_i)``. Call it inside the function being differentiated; every other
        leaf of ``base`` is passed through unchanged. The ``ice.<field>`` controls go into ONE
        ``Params.ice_ctrl`` dict ``{field: value}`` (merged over ``base.ice_ctrl`` if that is
        set); the other names replace their scalar field."""
        u = jnp.asarray(u, jnp.float64)
        if u.shape != (self.n,):
            raise ValueError(f"expected u of shape ({self.n},), got {u.shape}")
        base = Params.defaults() if base is None else base
        scalar, ice = {}, {}
        for i, c in enumerate(self.controls):
            if c.name.startswith(ICE_PREFIX):
                ice[c.name[len(ICE_PREFIX):]] = c.to_physical(u[i])
            else:
                scalar[c.name] = c.to_physical(u[i])
        if ice:
            scalar["ice_ctrl"] = {**(base.ice_ctrl or {}), **ice}
        return dataclasses.replace(base, **scalar)

    @staticmethod
    def prior(u):
        """Standard-normal prior on u: ``½·uᵀu`` (gradient u, Hessian I). Traceable."""
        u = jnp.asarray(u, jnp.float64)
        return 0.5 * jnp.sum(u * u)

    def grad_u(self, u, dJ_dp) -> np.ndarray:
        """Host-side chain rule without the prior: ``sigma_i·p_i·dJ/dp_i`` (numpy float64)."""
        u = np.asarray(u, np.float64)
        p = self.p0 * np.exp(self.sigma * u)
        return self.sigma * p * np.asarray(dJ_dp, np.float64)

    def check(self, u) -> np.ndarray:
        """Host-side domain check before an evaluation: raises ``ValueError`` if ``u`` has the
        wrong shape, a non-finite entry, or ``|u_i| > U_MAX``. Returns ``u`` as float64 numpy."""
        u = np.asarray(u, np.float64)
        if u.shape != (self.n,):
            raise ValueError(f"expected u of shape ({self.n},), got {u.shape}")
        if not np.all(np.isfinite(u)):
            raise ValueError(f"non-finite control vector: {dict(zip(self.names, u.tolist()))}")
        if np.any(np.abs(u) > U_MAX):
            bad = {c.name: float(u[i]) for i, c in enumerate(self.controls) if abs(u[i]) > U_MAX}
            raise ValueError(f"controls outside |u| <= {U_MAX}: {bad}")
        return u

    def out_of_domain(self, u) -> dict:
        """The controls whose physical value lies outside the physical domain, as
        ``{name: p}`` (empty ⇒ inside). Albedos (``ice.alb*``) must lie in (0, ALBEDO_MAX];
        every other control within a factor DOMAIN_FACTOR of its first guess. A log-normal
        albedo passes 1 at finite u (0.81·exp(0.0617·u) > 1 for u > 3.4), which the line
        search of a prior-off run can reach, so the calibration coordinator turns such a point
        into a failed evaluation instead of running the model. A wrong shape or a non-finite u
        is reported under the key ``"u"``."""
        u = np.asarray(u, np.float64)
        if u.shape != (self.n,) or not np.all(np.isfinite(u)):
            return {"u": u.tolist()}
        p = self.p0 * np.exp(self.sigma * u)
        bad = {}
        for i, c in enumerate(self.controls):
            if c.name.startswith(ICE_PREFIX + "alb"):
                ok = 0.0 < p[i] <= ALBEDO_MAX
            else:
                ok = c.p0 / DOMAIN_FACTOR <= p[i] <= c.p0 * DOMAIN_FACTOR
            if not ok:
                bad[c.name] = float(p[i])
        return bad

    def in_domain(self, u) -> bool:
        """True if every control's physical value is inside its domain (:meth:`out_of_domain`)."""
        return not self.out_of_domain(u)


#: Default TKE constants (the config values) and their order in the control vector.
TKE_FIELDS = ("tke_c_k", "tke_c_eps", "tke_cd", "tke_alpha")


def tke_control_set(sigma=SIGMA_DEFAULT, p0=None) -> ControlSet:
    """The four CVMix-TKE constants ``(c_k, c_eps, cd, alpha)`` as log-normal controls.

    ``p0`` defaults to the :meth:`Params.defaults` values (0.1, 0.7, 3.75, 30). ``sigma``
    (scalar or one per constant) defaults to ln 2 each: one prior standard deviation is a factor
    of two, a CHOICE (the constants are known to within a factor of a few, not to 10 %),
    overridable per call."""
    d = Params.defaults()
    if p0 is None:
        p0 = [float(getattr(d, f)) for f in TKE_FIELDS]
    sig = np.broadcast_to(np.asarray(sigma, np.float64), (len(TKE_FIELDS),))
    return ControlSet(tuple(LogNormalControl(f, float(p), float(s))
                            for f, p, s in zip(TKE_FIELDS, p0, sig)))


#: The sea-ice thermodynamic controls of the ice calibration, in control-vector order: snow and
#: ice albedos (dry, melting), ice and snow conductivities, the lead-closing thickness (north,
#: south), the compactness melt coefficient and the ocean-to-ice Stanton number.
ICE_THERMO_FIELDS = ("albsn", "albsnm", "albi", "albim", "con", "consn", "h0", "h0_s",
                     "c_melt", "st_oce_ice")
#: Absolute one-sigma uncertainty of the albedos (σ_ln p = ALBEDO_SIGMA / p0).
ALBEDO_SIGMA = 0.05


def ice_thermo_control_set() -> ControlSet:
    """The ten sea-ice thermodynamic controls ``ice.<field>`` with p0 the ``IceConfig()``
    defaults (``Params.defaults()`` carries no ice values, so p0 is always explicit) and the
    prior σ in ln p: ``ALBEDO_SIGMA / p0`` for the four albedos (±0.05 absolute at u = ±1),
    ln 2 (a factor 2) for the other six."""
    d = IceConfig()
    ctrls = []
    for f in ICE_THERMO_FIELDS:
        p0 = float(getattr(d, f))
        sigma = ALBEDO_SIGMA / p0 if f.startswith("alb") else SIGMA_DEFAULT
        ctrls.append(LogNormalControl(ICE_PREFIX + f, p0, sigma))
    return ControlSet(tuple(ctrls))


#: The sea-ice dynamic controls appended by :func:`ice_joint_control_set`: the ice strength P*,
#: the ice-ocean and the atmosphere-ice drag coefficients.
ICE_DYN_FIELDS = ("pstar", "cd_oce_ice", "cd_atm_ice")


def ice_joint_control_set() -> ControlSet:
    """The ten thermodynamic controls of :func:`ice_thermo_control_set` followed by the three
    dynamic ones of :data:`ICE_DYN_FIELDS`, p0 the ``IceConfig()`` defaults, σ = ln 2 each."""
    d = IceConfig()
    dyn = tuple(LogNormalControl(ICE_PREFIX + f, float(getattr(d, f)), SIGMA_DEFAULT)
                for f in ICE_DYN_FIELDS)
    return ControlSet(ice_thermo_control_set().controls + dyn)
