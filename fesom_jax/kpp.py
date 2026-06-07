"""KPP (K-Profile Parameterization) vertical mixing — Phase 6C.

Ports the FESOM2 C KPP subsystem (``fesom_kpp.c``, 1046 lines), itself a
validated line-by-line port of the Fortran ``oce_ale_mixing_kpp.F90``. KPP is the
**real FESOM2 CORE2 default** vertical-mixing scheme (``mix_scheme='KPP'``,
``mix_scheme_nmb==1``); the JAX port has been running the opt-in PP (``pp.py``,
``nmb==2``) so far. Like GM/Redi, KPP is **stateless** (every field recomputed each
step from T/S/N²/forcing — no new :class:`~fesom_jax.state.State` fields), and it is
a **CORE2 forced-path feature** (it needs surface forcing → ``u*``, surface buoyancy
``Bo``; the pi analytical path has none and keeps PP).

KPP replaces PP's local-Richardson ``Kv``/``Av`` with an **ocean-boundary-layer (OBL)
profile**: find the OBL depth ``hbl`` from a bulk-Richardson criterion, build a cubic
shape function for the diffusivity inside the OBL (matched to the interior at its
base), and use shear-Ri + background mixing in the interior — producing the **same
two outputs PP does**: ``Kv`` (node, tracer diffusivity) and ``Av`` (element,
momentum viscosity). It slots in at the exact mixing seam (``step.py``, substep 4)
behind a static ``kpp_cfg``: ``kpp_cfg=None ⇒ the PP path, bit-identical`` (the
``gm_cfg``/``ice_cfg`` precedent).

Driver ``fesom_kpp_mixing`` (``fesom_kpp.c:770-924``), the eight-stage data flow
(each stage is one ported kernel — see ``docs/plans/20260607-fesom-jax-kpp.md`` §5):

1. ``dVsq``      — shear of node velocity re the surface       (``:792-809``)
2. pre-step      — ``ustar = sqrt(sqrt(|τ|/ρ₀))``, surface ``Bo`` (``:811-821``)
3. ``ri_iwmix``  — interior shear-Ri + background mixing         (``:219-274``)  K.3
4. ``ddmix``     — double diffusion — **GATE ONLY** (CORE2 off)  (``:828-831``)  K.4
5. ``bldepth``   — OBL depth ``hbl``/``kbl``, ``bfsfc``, ``caseA`` (``:317-435``)  K.5
6. ``blmix``     — BL coeffs ``blmc[3]`` + ``dkm1`` + ``ghats``  (``:449-579``)  K.6
7. ``enhance``   — blend at ``kbl-1``                            (``:588-621``)  K.7
8. assembly      — ``smooth_blmc`` + combine + node→elem ``Av``  (``:863-918``)  K.7

Helper ``wscale`` (turbulent velocity scales ``wm``/``ws`` from a 2-D lookup table,
``:173-210``, K.2) is used by stages 5 & 6; the table + derived scalars are built once
(K.1). ``mo_convect`` runs **after** KPP (shared with PP, already ported in
:func:`fesom_jax.pp.mo_convect`).

**AD-safety (KPP is the kink-heaviest scheme).** The bar is *no NaN/Inf in the
backward, finite incl. masked lanes* + a well-conditioned gradient where one
physically exists. Treatments (full inventory in §4 of the sub-plan): ``ustar`` via
:func:`_safe_sqrt` (∞ backward slope at zero wind; ``ustar`` is in many denominators);
the discrete OBL level ``kbl`` / ``wscale`` bin index / ``caseA`` are ``stop_gradient``
ed *integers* with the **continuous** ``hbl`` interpolation weight kept differentiable;
and ``EPSLN=1e-40`` denominators sitting on physically-small quantities get **physical
floors** (1e-40 stops Inf but not gradient blow-up).
"""

from __future__ import annotations

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import eos, ops
from .config import DENSITY_0, G, VCPW
from .config import A_VER as _A_VER
from .config import K_VER as _K_VER
from .mesh import Mesh

# ---------------------------------------------------------------------------
# KPP constants — verbatim from fesom_kpp.c:30-52, 125-138 (= the CORE2 KPP
# reference namelists, docs/kpp_reference_namelists/). These are the single
# source of truth; KppConfig below is the typed bundle defaulting to them.
# ---------------------------------------------------------------------------
# module parameters (fesom_kpp.c:30-52)
_EPSLN = 1.0e-40        # denominator floor — NOT a physical ε (see §4 / _safe_sqrt)
_EPSILON = 0.1          # surface-layer fraction
_VONK = 0.4             # von Kármán
_CONC1 = 5.0
_ZMIN = -4.0e-7         # lookup-table zehat min
_ZMAX = 0.0
_UMIN = 0.0             # lookup-table ustar min
_UMAX = 0.04
_RICR = 0.3             # critical bulk Richardson number
_CONCV = 1.6
_VISC_SH_LIMIT = 5.0e-3  # interior shear-Ri viscosity limit
_DIFF_SH_LIMIT = 5.0e-3  # interior shear-Ri diffusivity limit
_RIINFTY = 0.8           # shear-Ri shape-function limit
_MINMIX = 3.0e-3         # surface element-viscosity floor
_CEKMAN = 0.7            # bldepth Ekman limit
_CMONOB = 1.0            # bldepth Monin-Obukhov limit
# table-build constants (fesom_kpp.c:125-127)
_CSTAR = 10.0
_CONAM = 1.257
_CONCM = 8.380
_CONC2 = 16.0
_ZETAM = -0.2
_CONAS = -28.86
_CONCS = 98.96
_CONC3 = 16.0
_ZETAS = -1.0
# lookup-table dimensions (fesom_kpp.h:54-55)
_NNI = 890              # zehat axis
_NNJ = 480              # ustar axis

# derived scalars — ported VERBATIM from fesom_kpp.c:130-138 (the same operator
# association as the C; research surfaced a minor disagreement on the Vtc/cg form,
# so trust the C). math.sqrt/math.pow are the libm routines the C calls ⇒ bit-equal.
# NOTE: these are frozen at the CORE2 values; if Ricr/concv/… ever change (Phase 7a
# tuning), re-derive — they are NOT auto-recomputed from the KppConfig fields.
_VTC = _CONCV * math.sqrt(0.2 / _CONCS / _EPSILON) / (_VONK * _VONK) / _RICR
_CG = _CSTAR * _VONK * math.pow(_CONCS * _VONK * _EPSILON, 1.0 / 3.0)
_DELTAZ = (_ZMAX - _ZMIN) / (_NNI + 1)
_DELTAU = (_UMAX - _UMIN) / (_NNJ + 1)


class KppConfig(NamedTuple):
    """Static KPP constants (closed over the step / passed as a ``static_argname``,
    like :class:`fesom_jax.gm.GMConfig` and ``IceConfig``). All fields are plain
    Python scalars/bools so the tuple is hashable — KPP carries **no** differentiable
    leaves of its own (the mixing-seam tunables ``k_ver``/``a_ver`` already live in
    :class:`fesom_jax.params.Params`; KPP's own ``Ricr``/``visc_sh_limit``/backgrounds
    become Phase-7a tuning targets there).

    Defaults = the verified CORE2 KPP reference config (``namelist.oce``/``namelist.tra``
    + the hardcoded ``#define``s in ``fesom_kpp.c``). All SI.
    """

    # --- master gates (which branches CORE2 takes) ---
    double_diffusion: bool = False    # ddmix — GATE ONLY (port the gate, defer body)
    use_kpp_nonlclflx: bool = False   # ghats consumed? — computed but NOT wired (CORE2)
    smooth_blmc: bool = True          # 3-sweep smoothing of the BL coeffs (applied)
    smooth_hbl: bool = False          # skipped
    use_sw_pene: bool = True          # shortwave penetration in bfsfc (always on, CORE2)

    # --- module parameters (fesom_kpp.c:30-52) ---
    epsln: float = _EPSLN             # forward-Inf denominator guard (NOT an AD floor)
    epsilon_kpp: float = _EPSILON     # surface-layer fraction
    vonk: float = _VONK               # von Kármán
    conc1: float = _CONC1
    zmin: float = _ZMIN
    zmax: float = _ZMAX
    umin: float = _UMIN
    umax: float = _UMAX
    ricr: float = _RICR               # critical bulk Richardson number
    concv: float = _CONCV
    visc_sh_limit: float = _VISC_SH_LIMIT
    diff_sh_limit: float = _DIFF_SH_LIMIT
    riinfty: float = _RIINFTY
    minmix: float = _MINMIX           # surface element-viscosity floor
    cekman: float = _CEKMAN
    cmonob: float = _CMONOB

    # --- table-build constants (fesom_kpp.c:125-127) ---
    cstar: float = _CSTAR
    conam: float = _CONAM
    concm: float = _CONCM
    conc2: float = _CONC2
    zetam: float = _ZETAM
    conas: float = _CONAS
    concs: float = _CONCS
    conc3: float = _CONC3
    zetas: float = _ZETAS

    # --- lookup-table dimensions (fesom_kpp.h:54-55) ---
    nni: int = _NNI                   # zehat axis
    nnj: int = _NNJ                   # ustar axis

    # --- derived scalars (fesom_kpp.c:130-138, ported verbatim) ---
    vtc: float = _VTC                 # eqn 23 velocity-scale coefficient
    cg: float = _CG                   # eqn 20 nonlocal-transport coefficient
    deltaz: float = _DELTAZ           # zehat table step
    deltau: float = _DELTAU           # ustar table step

    # --- external physical constants (mirror config; self-contained bundle) ---
    g: float = G
    rho0: float = DENSITY_0
    vcpw: float = VCPW                # volumetric heat capacity of seawater
    a_bg: float = _A_VER              # background momentum viscosity (= A_ver)
    k_bg: float = _K_VER              # background tracer diffusivity (= K_ver)


def _safe_sqrt(x):
    """Double-``where`` safe sqrt: forward-identical to ``sqrt(max(x,0))`` but with a
    finite gradient at ``x=0`` (a bare ``sqrt(0)`` has grad ``1/(2·0)=∞`` that a
    downstream mask does not stop in the backward pass). The project idiom shared with
    :func:`fesom_jax.gm._safe_sqrt`; the #1 KPP AD priority is ``ustar`` =
    ``_safe_sqrt(_safe_sqrt(|τ|/ρ₀))`` (∞ backward slope at zero wind, and ``ustar``
    sits in many denominators)."""
    safe = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.sqrt(safe), 0.0)


# ============================================================================
# K.1 — wm/ws turbulent-velocity-scale lookup tables (fesom_kpp.c:140-165)
# ============================================================================
@functools.lru_cache(maxsize=None)
def build_wscale_tables(cfg: KppConfig):
    """Build the ``wmt``/``wst`` ``(nni+2, nnj+2)`` velocity-scale lookup tables —
    the literal port of the ``fesom_kpp_init`` table build (``fesom_kpp.c:140-165``).

    **Constant data** (a function only of the static :class:`KppConfig`), so it is
    built once on the host (numpy, ``lru_cache``d on ``cfg``) and **no gradient flows
    through the table values** — the differentiable part of the lookup is the bilinear
    interpolation weight in :func:`wscale`. Returns ``(wmt, wst)`` as ``jnp.float64``
    arrays (concrete constants ⇒ baked into the jit).

    Index convention matches the C ``KPP_TBL(i,j) = i*(nnj+2)+j`` (``i`` = zehat row,
    ``j`` = ustar column). The fractional ``pow(·, 1/3)`` / ``pow(·, 1/4|1/2)`` bases
    are clamped ``≥0`` for the **discarded** ``jnp.where`` lanes (the *kept* lane of
    each branch has the base ``>0`` — verified analytically: in ``zeta≤zetas`` the
    ``conas·u³−concs·zehat`` base ``≥70·|zehat|>0``; ``conam·u³−concm·zehat>0`` for
    ``zehat<0``; ``1−conc2·zeta``/``1−conc3·zeta>0`` since ``zeta<0``) ⇒ the clamp is
    exact, only suppressing NaN in lanes the ``where`` throws away."""
    nni, nnj, eps = cfg.nni, cfg.nnj, cfg.epsln
    i = np.arange(nni + 2)[:, None]
    j = np.arange(nnj + 2)[None, :]
    zehat = cfg.deltaz * i + cfg.zmin                       # (nni+2,1)
    usta = cfg.deltau * j + cfg.umin                        # (1,nnj+2)
    zehat, usta = (np.broadcast_to(zehat, (nni + 2, nnj + 2)).astype(np.float64),
                   np.broadcast_to(usta, (nni + 2, nnj + 2)).astype(np.float64))
    u3 = usta * usta * usta
    zeta = zehat / (u3 + eps)

    # stable forcing (zehat >= 0): wm == ws
    wm_stable = cfg.vonk * usta / (1.0 + cfg.conc1 * zeta)

    # unstable wm: zeta>zetam (table) vs the cube-root tail
    wm_hi = cfg.vonk * usta * np.power(np.maximum(1.0 - cfg.conc2 * zeta, 0.0), 0.25)
    wm_lo = cfg.vonk * np.power(np.maximum(cfg.conam * u3 - cfg.concm * zehat, 0.0),
                                1.0 / 3.0)
    wm_unstable = np.where(zeta > cfg.zetam, wm_hi, wm_lo)

    # unstable ws: zeta>zetas (table) vs the cube-root tail
    ws_hi = cfg.vonk * usta * np.power(np.maximum(1.0 - cfg.conc3 * zeta, 0.0), 0.5)
    ws_lo = cfg.vonk * np.power(np.maximum(cfg.conas * u3 - cfg.concs * zehat, 0.0),
                                1.0 / 3.0)
    ws_unstable = np.where(zeta > cfg.zetas, ws_hi, ws_lo)

    wmt = np.where(zehat >= 0.0, wm_stable, wm_unstable)
    wst = np.where(zehat >= 0.0, wm_stable, ws_unstable)
    return jnp.asarray(wmt), jnp.asarray(wst)


# ============================================================================
# K.2 — wscale: turbulent velocity scales from the 2-D lookup table
# ============================================================================
def wscale(cfg: KppConfig, wmt, wst, zehat, us):
    """Turbulent velocity scales ``wm``/``ws`` — literal port of ``kpp_wscale``
    (``fesom_kpp.c:173-210``). ``zehat``/``us`` are arbitrary-shaped arrays (per node,
    per interface, …); ``wmt``/``wst`` are the :func:`build_wscale_tables` constants.

    Two regimes, selected by ``zehat <= zmax`` (a regime switch ⇒ ``jnp.where``):

    * **unstable** (``zehat<=0``): a bilinear lookup. The C ``(int)`` bin indices
      ``iz``/``ju`` are **discrete** — ``trunc`` + clamp (``jnp.trunc`` has zero
      gradient a.e., so the integer selection carries no cotangent), but the fractional
      weights ``zfrac``/``ufrac`` stay **differentiable** in ``zehat``/``us`` (the cubic
      shape uses these — "which cell" discrete, "where within the cell" smooth). The
      gathered table values are constants ⇒ the gradient flows only through the
      bilinear weights, giving the table's piecewise-linear slope.
    * **stable** (``zehat>0``): the analytic ``vonk·us·u³/(u³+conc1·zehat+epsln)``.

    AD-safety: the stable denominator is replaced by a **safe** value in the masked
    (unstable) lanes so the unused branch can never produce a non-finite that the
    ``where``-backward would turn into a masked NaN (``0·inf``); in the stable lanes
    the denom is ``≥conc1·zehat>0`` so ``epsln`` there is a pure forward guard."""
    nni, nnj = float(cfg.nni), float(cfg.nnj)

    # --- unstable: bilinear table lookup -------------------------------------
    zq = (zehat - cfg.zmin) / cfg.deltaz
    iz = jnp.clip(jnp.trunc(zq), 0.0, nni)                  # clamp BEFORE astype (no overflow)
    izi = iz.astype(jnp.int32)
    izp1 = izi + 1
    zfrac = zq - iz                                         # differentiable weight
    fz = 1.0 - zfrac

    # ⚠️ the C clamps udiff/deltau to nnj for the INDEX ju but keeps the UNCLAMPED
    # numerator for the fractional weight (fesom_kpp.c:184-191) — so ustar beyond the
    # table's UMAX extrapolates linearly (ufrac>1), recovering e.g. vonk·us at us>0.04.
    uq_raw = (us - cfg.umin) / cfg.deltau                  # unclamped numerator
    uq = jnp.minimum(uq_raw, nnj)                          # float-clamp ≤ nnj for the index
    ju = jnp.clip(jnp.trunc(uq), 0.0, nnj)
    jui = ju.astype(jnp.int32)
    jup1 = jui + 1
    ufrac = uq_raw - ju                                    # unclamped numerator − clamped index

    wam = fz * wmt[izi, jup1] + zfrac * wmt[izp1, jup1]
    wbm = fz * wmt[izi, jui] + zfrac * wmt[izp1, jui]
    wm_tbl = (1.0 - ufrac) * wbm + ufrac * wam
    was = fz * wst[izi, jup1] + zfrac * wst[izp1, jup1]
    wbs = fz * wst[izi, jui] + zfrac * wst[izp1, jui]
    ws_tbl = (1.0 - ufrac) * wbs + ufrac * was

    # --- stable: analytic branch (safe denom in the masked unstable lanes) ---
    use_tbl = zehat <= cfg.zmax
    u3 = us * us * us
    denom = u3 + cfg.conc1 * zehat + cfg.epsln
    safe_denom = jnp.where(use_tbl, 1.0, denom)            # unstable lanes: dummy 1 (masked away)
    wm_st = cfg.vonk * us * u3 / safe_denom

    wm = jnp.where(use_tbl, wm_tbl, wm_st)
    ws = jnp.where(use_tbl, ws_tbl, wm_st)                 # ws == wm in the stable branch
    return wm, ws


def _shift_down(x):
    """``out[..., k] = x[..., k-1]`` with edge replication at ``k=0`` (Δ=0 there —
    finite, AD-safe; the surface interface is masked/edge-copied anyway). The
    :mod:`fesom_jax.pp` / :mod:`fesom_jax.eos` idiom."""
    return jnp.concatenate([x[..., :1], x[..., :-1]], axis=-1)


# ============================================================================
# K.3 — ri_iwmix: interior shear-Ri + background mixing (fesom_kpp.c:219-274)
# ============================================================================
def ri_iwmix(mesh: Mesh, uvnode, bvfreq, cfg: KppConfig):
    """Interior viscosity/diffusivity from local shear-Richardson instability +
    the constant backgrounds — literal port of ``kpp_ri_iwmix`` (``fesom_kpp.c:219``).

    Per interior interface ``nz ∈ [nzmin+1, nzmax-1]``:
    ``Ri = max(N²,0)/(shear²+epsln)``, ``frit = (1 − min(max(Ri,0)/Riinfty, 1)²)³``,
    ``viscA = visc_sh_limit·frit + A_bg``, ``diffKt = diffKs = diff_sh_limit·frit +
    K_bg`` (the ``Kv0_const`` branch ⇒ the T and S channels are identical). The
    surface (``nzmin``) and bottom (``nzmax``) interfaces are **edge copies** of the
    adjacent interior interface (the C's two-pass scratch — the Ri edge copies in pass
    1 are fully overwritten by the viscA/diffK edge copies in pass 2, so this single
    edge-copy-of-the-result is exactly equivalent).

    ``uvnode`` is ``[N,nl,2]`` (``pp.compute_vel_nodes``); ``bvfreq`` ``[N,nl]`` (post
    smooth). Returns ``(viscA, diffKt, diffKs)``, each ``[N,nl]`` on the iface range
    (``node_iface_mask`` = ``[nzmin,nzmax]``); ``diffKt is diffKs`` (same array).

    AD: the shear uses the dz-clamped ``pp``/``eos`` pattern (no Inf at the masked
    surface/bottom-pad lanes); ``epsln`` floors the shear denominator (a forward-Inf
    guard — at any realistic shear ``shear≫epsln`` so it is inert, and where
    ``shear→0`` the outcome is ``frit∈{0,1}`` with a clamped/zero ``ratio`` ⇒ no
    backward blow-up; the masked-NaN gate K.10 confirms end-to-end finiteness)."""
    nl = mesh.nl
    Zp = jnp.concatenate([mesh.Z, mesh.Z[-1:]])            # (nl,) pad invalid tail
    dz = _shift_down(Zp) - Zp                              # Z[nz-1]-Z[nz] (>0 interior)
    dz = jnp.where(dz == 0.0, 1.0, dz)                     # k=0 + bottom-pad: avoid 0 (masked)
    dz_inv = (1.0 / dz)[None, :]                           # (1,nl)

    u, v = uvnode[..., 0], uvnode[..., 1]                  # (N,nl)
    du = _shift_down(u) - u                                # u[nz-1]-u[nz]
    dv = _shift_down(v) - v
    shear = (du * du + dv * dv) * dz_inv * dz_inv          # (N,nl)

    nsq_pos = jnp.maximum(bvfreq, 0.0)
    Ri = nsq_pos / (shear + cfg.epsln)
    ratio = jnp.minimum(jnp.maximum(Ri, 0.0) / cfg.riinfty, 1.0)
    frit = (1.0 - ratio * ratio)
    frit = frit * frit * frit
    viscA = cfg.visc_sh_limit * frit + cfg.a_bg
    diffK = cfg.diff_sh_limit * frit + cfg.k_bg

    # edge copies: fill the output range [nzmin,nzmax] from the interior
    # [nzmin+1,nzmax-1] by gathering with a clipped level index (nzmin←nzmin+1,
    # nzmax←nzmax-1, interior unchanged) — the eos.bvfreq edge-pad idiom.
    k = jnp.arange(nl).reshape(1, -1)
    lo = mesh.ulevels_nod2D.reshape(-1, 1)                 # = nzmin+1
    hi = (mesh.nlevels_nod2D - 2).reshape(-1, 1)           # = nzmax-1
    idx = jnp.clip(k, lo, hi)
    viscA = jnp.take_along_axis(viscA, idx, axis=1)
    diffK = jnp.take_along_axis(diffK, idx, axis=1)

    viscA = jnp.where(mesh.node_iface_mask, viscA, 0.0)
    diffK = jnp.where(mesh.node_iface_mask, diffK, 0.0)
    return viscA, diffK, diffK


# ============================================================================
# K.4 — ddmix (double diffusion): GATE ONLY (CORE2 double_diffusion=.false.)
# ============================================================================
def assert_no_double_diffusion(cfg: KppConfig):
    """Port the ``ddmix`` gate, defer the body (the port-what-CORE2-uses rule). CORE2
    has ``double_diffusion=.false.`` so the driver never calls ``ddmix`` (the C
    ``#error``s if it is enabled, ``fesom_kpp.c:828-831``). This mirrors that: a no-op
    when the gate is off, a loud failure if a config ever turns it on (port the
    salt-fingering / diffusive-convection body of ``oce_ale_mixing_kpp.F90:1012-1085``
    first). The nonlocal transport flux (``use_kpp_nonlclflx``) is likewise gated off —
    ``ghats`` is *computed* in :func:`blmix` but never wired into the tracer flux."""
    if cfg.double_diffusion:
        raise NotImplementedError(
            "KPP double_diffusion=True unsupported: port ddmix "
            "(oce_ale_mixing_kpp.F90:1012-1085) first — CORE2 uses double_diffusion=False.")


def _heaviside(x):
    """``0.5 + copysign(0.5, x)`` = ``(x>=0 ? 1 : 0)`` (with the C's +0.0→1 / −0.0→0
    convention via ``jnp.copysign``). A regime switch ⇒ the value is 0/1 and the
    gradient is 0 a.e. (``copysign`` is piecewise-constant), so it is intrinsically
    ``stop_gradient``ed — exactly the KPP ``stable``/``caseA`` treatment."""
    return 0.5 + jnp.copysign(0.5, x)


def _first_crossing(mask, axis, fallback):
    """First index along ``axis`` where ``mask`` is True; ``fallback`` (an int array)
    where there is no True. ``jnp.argmax`` of a bool returns the first True. The result
    is a **discrete** index ⇒ the caller ``stop_gradient``s it (the continuous
    interpolation weight is kept differentiable separately)."""
    any_true = jnp.any(mask, axis=axis)
    idx = jnp.argmax(mask, axis=axis)
    return jnp.where(any_true, idx, fallback)


# ============================================================================
# K.5 — pre-step: dVsq (shear re surface), ustar (safe-sqrt), Bo (fesom_kpp.c:792-821)
# ============================================================================
def prestep(mesh: Mesh, uvnode, stress_node_surf, heat_flux, water_flux,
            sw_alpha, sw_beta, S, cfg: KppConfig):
    """KPP driver pre-step — literal port of ``fesom_kpp.c:792-821``.

    * ``dVsq[nz] = |u_surf − ½(u[nz-1]+u[nz])|²`` (shear of the interface velocity re
      the surface layer; ``[nzmin+1,nzmax-1]``, surface 0, bottom edge-copied).
    * ``ustar = sqrt(sqrt(|τ|/ρ₀))`` — the double :func:`_safe_sqrt` (∞ backward slope
      at zero wind; ``ustar`` sits in many KPP denominators — the #1 AD priority).
    * ``Bo = −g·(α_surf·heat_flux/VCPW + β_surf·water_flux·S_surf)`` (surface turbulent
      buoyancy forcing).

    ``uvnode`` ``[N,nl,2]``; ``stress_node_surf`` ``[N,2]``; ``heat_flux``/``water_flux``
    ``[N]``; ``sw_alpha``/``sw_beta``/``S`` ``[N,nl]``. Returns ``(dVsq[N,nl], ustar[N],
    Bo[N])``."""
    nl = mesh.nl
    nzmin = (mesh.ulevels_nod2D - 1).reshape(-1, 1)        # (N,1); 0 for CORE2

    # --- dVsq: |u_surf − mid-interface velocity|² --------------------------------
    u, v = uvnode[..., 0], uvnode[..., 1]                  # (N,nl)
    u_surf = jnp.take_along_axis(u, nzmin, axis=1)         # (N,1) surface-layer u
    v_surf = jnp.take_along_axis(v, nzmin, axis=1)
    u_mid = 0.5 * (_shift_down(u) + u)                     # ½(u[nz-1]+u[nz])
    v_mid = 0.5 * (_shift_down(v) + v)
    du = u_surf - u_mid
    dv = v_surf - v_mid
    dVsq = du * du + dv * dv                               # (N,nl)
    # edge handling: surface → 0, bottom nzmax → nzmax-1; interior [nzmin+1,nzmax-1].
    k = jnp.arange(nl).reshape(1, -1)
    lo = mesh.ulevels_nod2D.reshape(-1, 1)                 # nzmin+1
    hi = (mesh.nlevels_nod2D - 2).reshape(-1, 1)           # nzmax-1
    dVsq = jnp.take_along_axis(dVsq, jnp.clip(k, lo, hi), axis=1)   # fill nzmax←nzmax-1
    dVsq = jnp.where(k == nzmin, 0.0, dVsq)                # surface = 0
    dVsq = jnp.where(mesh.node_iface_mask, dVsq, 0.0)

    # --- ustar = sqrt(sqrt(|τ|/ρ₀)) ---------------------------------------------
    tau = _safe_sqrt(stress_node_surf[:, 0] ** 2 + stress_node_surf[:, 1] ** 2)  # |τ|
    ustar = _safe_sqrt(tau / cfg.rho0)

    # --- Bo (surface buoyancy forcing) ------------------------------------------
    a_surf = jnp.take_along_axis(sw_alpha, nzmin, axis=1)[:, 0]
    b_surf = jnp.take_along_axis(sw_beta, nzmin, axis=1)[:, 0]
    S_surf = jnp.take_along_axis(S, nzmin, axis=1)[:, 0]
    Bo = -cfg.g * (a_surf * heat_flux / cfg.vcpw + b_surf * water_flux * S_surf)
    return dVsq, ustar, Bo


# ============================================================================
# K.5 — bldepth: OBL depth hbl/kbl + bfsfc/stable/caseA (fesom_kpp.c:317-435)
# ============================================================================
def bldepth(mesh: Mesh, dVsq, ustar, Bo, bvfreq, dbsfc, sw_3d, sw_alpha, wmt, wst,
            cfg: KppConfig):
    """Oceanic-boundary-layer depth ``hbl`` + level ``kbl`` + ``bfsfc``/``stable``/
    ``caseA`` — literal port of ``kpp_bldepth`` (``fesom_kpp.c:317-435``). The
    highest-risk KPP kernel: a per-node bulk-Richardson first-crossing search.

    The C's two sequential loops are vectorized as **two masked first-crossings**
    (:func:`_first_crossing`): loop 1 finds the first interface where the bulk Ri
    ``Rib_k > Ricr`` → ``kbl1`` + the interpolated ``hbl`` (the interp weight stays
    differentiable; the integer ``kbl1`` is ``stop_gradient``ed); the Ekman/Monin-
    Obukhov limit clamps ``hbl`` where the surface forcing is stabilizing; loop 2 finds
    ``kbl`` = the first interface deeper than the final ``hbl`` and sets the final
    ``bfsfc``/``caseA``. ``Rib_k`` has **no** inter-level dependence (each level is a
    pure function of its own forcing) — the only sequential quantity, ``Rib_km1``, is
    the gather ``Rib_k[kbl-1]`` (with a ``Rib_k[nzmin]=0`` sentinel so a first-level
    crossing recovers the C's ``Rib_km1=0`` init).

    All inputs are the C-dumped controlled-replay fields (``dVsq``/``Bo``/``bvfreq``/
    ``dbsfc``/``sw_3d`` ``[N,nl]``, ``ustar`` ``[N]``, ``sw_alpha`` ``[N,nl]``) + the
    ``wmt``/``wst`` tables. Returns ``(hbl[N], kbl[N] int, bfsfc[N], stable[N],
    caseA[N])``."""
    nl = mesh.nl
    zbar = mesh.zbar_3d_n                                   # (N,nl) signed interface depth (≤0)
    absz = jnp.abs(zbar)                                   # zk = |zbar|
    nzmin = mesh.ulevels_nod2D - 1                         # (N,)
    nzmax = mesh.nlevels_nod2D - 1
    k = jnp.arange(nl).reshape(1, -1)
    nzmin_c, nzmax_c = nzmin.reshape(-1, 1), nzmax.reshape(-1, 1)

    coeff_sw = cfg.g * jnp.take_along_axis(sw_alpha, nzmin_c, axis=1)   # (N,1) g·α[nzmin]
    sw_surf = jnp.take_along_axis(sw_3d, nzmin_c, axis=1)              # (N,1)
    Bo_c = Bo.reshape(-1, 1)
    us_full = jnp.broadcast_to(ustar.reshape(-1, 1), (ustar.shape[0], nl))

    # --- per-level bulk Richardson Rib_k (no inter-level dependence) -------------
    bfsfc = Bo_c + coeff_sw * (sw_surf - sw_3d)            # (N,nl) top-of-loop bfsfc
    stable = _heaviside(bfsfc)
    sigma = stable + (1.0 - stable) * cfg.epsilon_kpp
    zehat = cfg.vonk * sigma * absz * bfsfc
    _, ws = wscale(cfg, wmt, wst, zehat, us_full)
    Vtsq = absz * ws * _safe_sqrt(jnp.abs(bvfreq)) * cfg.vtc
    Ritop = absz * dbsfc
    Rib_k = Ritop / (dVsq + Vtsq + cfg.epsln)
    Rib_k = jnp.where(k == nzmin_c, 0.0, Rib_k)           # sentinel: Rib_k[nzmin]=0

    # --- loop 1: first crossing Rib_k > Ricr → kbl1 + interpolated hbl -----------
    search = (k >= nzmin_c + 1) & (k <= nzmax_c)
    above = (Rib_k > cfg.ricr) & search
    crossed = jnp.any(above, axis=1)                       # (N,)
    kbl1 = lax.stop_gradient(_first_crossing(above, 1, nzmax))   # discrete
    kbl1c = kbl1.reshape(-1, 1)
    zk = jnp.take_along_axis(absz, kbl1c, 1)[:, 0]
    zkm1 = jnp.take_along_axis(absz, kbl1c - 1, 1)[:, 0]
    Rib_at = jnp.take_along_axis(Rib_k, kbl1c, 1)[:, 0]
    Rib_prev = jnp.take_along_axis(Rib_k, kbl1c - 1, 1)[:, 0]    # = Rib_km1 (0 if kbl1=nzmin+1)
    hbl_cross = zkm1 + (zk - zkm1) * (cfg.ricr - Rib_prev) / (Rib_at - Rib_prev + cfg.epsln)
    hbl_bottom = jnp.take_along_axis(absz, nzmax_c, 1)[:, 0]     # |zbar[nzmax]| (bottomed-out)
    hbl = jnp.where(crossed, hbl_cross, hbl_bottom)

    # --- Ekman / Monin-Obukhov limits (gated bfsfc>0 && nzmin==0) ----------------
    # loop-1-end bfsfc = bfsfc at kbl1 (the broken iteration's top-of-loop value; for
    # the never-crossed case the C's final refine collapses to the same expression).
    bfsfc1 = Bo + coeff_sw[:, 0] * (sw_surf[:, 0] - jnp.take_along_axis(sw_3d, kbl1c, 1)[:, 0])
    stable1 = _heaviside(bfsfc1)
    hekman = cfg.cekman * ustar / jnp.maximum(jnp.abs(mesh.coriolis_node), cfg.epsln)
    hmonob = cfg.cmonob * ustar ** 3 / cfg.vonk / (bfsfc1 + cfg.epsln)
    hlimit = stable1 * jnp.minimum(hekman, hmonob)
    hbl_lim = jnp.maximum(jnp.minimum(hbl, hlimit), absz[:, 1])   # max(min(hbl,hlimit), |zbar[1]|)
    gate = (bfsfc1 > 0.0) & (nzmin == 0)
    hbl = jnp.where(gate, hbl_lim, hbl)

    # --- loop 2: new kbl = first interface deeper than hbl -----------------------
    deeper = (absz > hbl.reshape(-1, 1)) & search
    crossed2 = jnp.any(deeper, axis=1)
    kbl = lax.stop_gradient(_first_crossing(deeper, 1, nzmax))
    kblc = kbl.reshape(-1, 1)

    # --- final bfsfc (sw interp to hbl using SIGNED zbar) + caseA ----------------
    sw_km1 = jnp.take_along_axis(sw_3d, kblc - 1, 1)[:, 0]
    sw_k = jnp.take_along_axis(sw_3d, kblc, 1)[:, 0]
    zbar_km1 = jnp.take_along_axis(zbar, kblc - 1, 1)[:, 0]
    zbar_k = jnp.take_along_axis(zbar, kblc, 1)[:, 0]
    frac = (hbl + zbar_km1) / (zbar_km1 - zbar_k)
    bfsfc_f = Bo + coeff_sw[:, 0] * (sw_surf[:, 0] - (sw_km1 + (sw_k - sw_km1) * frac))
    stable_f = _heaviside(bfsfc_f)
    bfsfc_f = bfsfc_f + stable_f * cfg.epsln
    dzup = zbar_km1 - zbar_k
    caseA = _heaviside(jnp.abs(zbar_k) - 0.5 * dzup - hbl)
    return hbl, kbl, bfsfc_f, stable_f, caseA


# ============================================================================
# K.6 — blmix: BL coeffs blmc[3] + dkm1[3] + ghats (fesom_kpp.c:449-579)
# ============================================================================
def blmix(mesh: Mesh, hnode, viscA, diffKt, diffKs, hbl, bfsfc, stable, caseA, kbl,
          ustar, wmt, wst, cfg: KppConfig):
    """Boundary-layer mixing coefficients — literal port of ``kpp_blmix``
    (``fesom_kpp.c:449-579``). Matches the interior diffusivities (the ``ri_iwmix``
    ``viscA``/``diffKt``/``diffKs`` = ``dcol``) to the surface-layer scaling at ``hbl``
    via the ``caseA``-selected level ``kn``, the cubic shape ``G(σ)``, and
    :func:`wscale`, producing ``blmc[3]`` (momentum/T/S) over the BL interfaces
    ``[nzmin+1, kbl-1]`` + ``dkm1[3]`` at ``kbl-1`` + ``ghats`` (computed; CORE2 gates
    it off in the combine).

    Channels: ``blmc``/``dkm1`` comp 0=momentum (from ``viscA``/``wm``), 1=T (from
    ``diffKt``/``ws``), 2=S (from ``diffKs``/``ws``) — note the C cross-wiring
    (T←dcol ch1, S←dcol ch2). The matching level ``kn`` and the cubic loop bound are
    discrete ⇒ ``stop_gradient``ed; the one-sided slope ``½(dvdz+|dvdz|)=max(dvdz,0)``
    is an AD-safe kink. Skipped nodes (``nlevels<3`` or ``nlevels-ulevels<2``) get
    ``blmc=dkm1=ghats=0``.

    All inputs are the C controlled-replay fields: ``viscA``/``diffKt``/``diffKs``
    ``[N,nl]`` (ri_iwmix outputs), ``hbl``/``bfsfc``/``stable``/``caseA``/``ustar``
    ``[N]``, ``kbl`` ``[N]`` int. Returns ``(blmc_m, blmc_t, blmc_s [N,nl], ghats
    [N,nl], dkm1 [N,3])``."""
    nl = mesh.nl
    N = viscA.shape[0]
    Zabs = jnp.abs(jnp.concatenate([mesh.Z, mesh.Z[-1:]]))[None, :]   # |Z| (1,nl)
    zbar = mesh.zbar_3d_n                                  # (N,nl) signed
    k = jnp.arange(nl).reshape(1, -1)
    nzmin = mesh.ulevels_nod2D - 1
    nzmax = mesh.nlevels_nod2D - 1
    nzmin_c, nzmax_c = nzmin.reshape(-1, 1), nzmax.reshape(-1, 1)
    hbl_c = hbl.reshape(-1, 1)
    us_full = jnp.broadcast_to(ustar.reshape(-1, 1), (N, nl))

    def gcol(arr, idx):                                    # gather (N,) at per-node idx
        return jnp.take_along_axis(arr, idx.reshape(-1, 1), axis=1)[:, 0]

    # --- dthick (interface thicknesses): interior ½(h[nz-1]+h[nz]); nzmin ½h[nzmin];
    #     nzmax ½h[nzmax-1] ------------------------------------------------------
    dthick = 0.5 * (_shift_down(hnode) + hnode)
    dthick = jnp.where(k == nzmin_c, 0.5 * gcol(hnode, nzmin).reshape(-1, 1), dthick)
    dthick = jnp.where(k == nzmax_c, 0.5 * gcol(hnode, nzmax - 1).reshape(-1, 1), dthick)

    # dcol = the ri_iwmix outputs (already carry the nzmax←nzmax-1 edge copy).
    dcol = (viscA, diffKt, diffKs)

    # --- velocity scales at hbl (gat1/dat1 use these) ---------------------------
    sigma_h = stable + (1.0 - stable) * cfg.epsilon_kpp
    wm_h, ws_h = wscale(cfg, wmt, wst, cfg.vonk * sigma_h * hbl * bfsfc, ustar)

    # --- caseA-selected matching level kn (discrete ⇒ stop_gradient) ------------
    ca = (caseA + cfg.epsln).astype(jnp.int32)            # 0/1
    kn = jnp.minimum(kbl - ca, nzmax - 1)
    knm1 = jnp.maximum(kn - 1, nzmin)
    knp1 = jnp.minimum(kn + 1, nzmax)
    kn, knm1, knp1 = (lax.stop_gradient(x) for x in (kn, knm1, knp1))

    delhat = gcol(jnp.broadcast_to(Zabs, (N, nl)), kn) - hbl   # |Z[kn]| − hbl
    dth_kn = gcol(dthick, kn)
    dth_knp1 = gcol(dthick, knp1)
    R = 1.0 - delhat / dth_kn

    def slope_valh(ch):
        d_kn = gcol(ch, kn)
        dvdzup = (gcol(ch, knm1) - d_kn) / dth_kn
        dvdzdn = (d_kn - gcol(ch, knp1)) / dth_knp1
        slope = 0.5 * ((1.0 - R) * (dvdzup + jnp.abs(dvdzup))
                       + R * (dvdzdn + jnp.abs(dvdzdn)))
        return slope, d_kn + slope * delhat                # (viscp,visch) etc.

    viscp, visch = slope_valh(dcol[0])
    difsp, difsh = slope_valh(dcol[2])                     # S ← dcol ch2
    diftp, difth = slope_valh(dcol[1])                     # T ← dcol ch1

    f1 = stable * cfg.conc1 * bfsfc / (ustar ** 4 + cfg.epsln)

    def gat_dat(valh, slope, w):
        gat1 = valh / (hbl + cfg.epsln) / (w + cfg.epsln)
        dat1 = jnp.minimum(-slope / (w + cfg.epsln) + f1 * valh, 0.0)
        return gat1, dat1

    gat1m, dat1m = gat_dat(visch, viscp, wm_h)
    gat1s, dat1s = gat_dat(difsh, difsp, ws_h)
    gat1t, dat1t = gat_dat(difth, diftp, ws_h)

    def cubic(sig, w, gat1, dat1):
        a1, a2, a3 = sig - 2.0, 3.0 - 2.0 * sig, sig - 1.0
        G = a1 + a2 * gat1[:, None] + a3 * dat1[:, None]
        return hbl_c * w * sig * (1.0 + sig * G)

    # --- BL coeffs at interfaces nz ∈ [nzmin+1, min(kbl-1, nzmax-1)] -------------
    valid_node = ((mesh.nlevels_nod2D >= 3)
                  & (mesh.nlevels_nod2D - mesh.ulevels_nod2D >= 2)).reshape(-1, 1)
    cubic_mask = ((k >= nzmin_c + 1) & (k <= nzmax_c - 1) & (k < kbl.reshape(-1, 1))
                  & valid_node)
    sig = Zabs / (hbl_c + cfg.epsln)                       # (N,nl)
    sigma = stable.reshape(-1, 1) * sig + (1.0 - stable.reshape(-1, 1)) * jnp.minimum(
        sig, cfg.epsilon_kpp)
    wm, ws = wscale(cfg, wmt, wst, cfg.vonk * sigma * hbl_c * bfsfc.reshape(-1, 1), us_full)
    blmcM = jnp.where(cubic_mask, cubic(sig, wm, gat1m, dat1m), 0.0)
    blmcT = jnp.where(cubic_mask, cubic(sig, ws, gat1t, dat1t), 0.0)
    blmcS = jnp.where(cubic_mask, cubic(sig, ws, gat1s, dat1s), 0.0)
    ghats = jnp.where(cubic_mask,
                      (1.0 - stable.reshape(-1, 1)) * cfg.cg / (ws * hbl_c + cfg.epsln), 0.0)

    # --- dkm1 at kbl-1 (σ from zbar, not Z) -------------------------------------
    sig1 = jnp.abs(gcol(zbar, kbl - 1)) / (hbl + cfg.epsln)
    sigma1 = stable * sig1 + (1.0 - stable) * jnp.minimum(sig1, cfg.epsilon_kpp)
    wm1, ws1 = wscale(cfg, wmt, wst, cfg.vonk * sigma1 * hbl * bfsfc, ustar)

    def cubic1(w, gat1, dat1):
        a1, a2, a3 = sig1 - 2.0, 3.0 - 2.0 * sig1, sig1 - 1.0
        return hbl * w * sig1 * (1.0 + sig1 * (a1 + a2 * gat1 + a3 * dat1))

    vn = valid_node[:, 0]
    dkm1 = jnp.stack([jnp.where(vn, cubic1(wm1, gat1m, dat1m), 0.0),
                      jnp.where(vn, cubic1(ws1, gat1t, dat1t), 0.0),
                      jnp.where(vn, cubic1(ws1, gat1s, dat1s), 0.0)], axis=1)   # (N,3) M/T/S
    return blmcM, blmcT, blmcS, ghats, dkm1


# ============================================================================
# K.7 — enhance (blend at kbl-1) + smooth_blmc + combine + node→elem (Av)
# ============================================================================
def enhance(mesh: Mesh, blmcM, blmcT, blmcS, ghats, dkm1, viscA, diffKt, diffKs,
            hbl, caseA, kbl, cfg: KppConfig):
    """Enhance the BL coeffs at the ``kbl-1`` interface — literal port of
    ``kpp_enhance`` (``fesom_kpp.c:588-621``). Blends the interior (``caseA``)
    coefficient, the BL coefficient, and ``dkm1`` at ``k=kbl-1`` with the fractional
    position ``delta = (hbl + zbar[kbl-1])/(zbar[kbl-1]−zbar[kbl])``; also scales
    ``ghats[kbl-1]`` by ``(1−caseA)``. Modifies only the single ``kbl-1`` interface per
    node. Returns the updated ``(blmcM, blmcT, blmcS, ghats)``."""
    nl = mesh.nl
    zbar = mesh.zbar_3d_n
    kk = kbl - 1                                           # k = kbl-1 (≥0)

    def gcol(arr, idx):
        return jnp.take_along_axis(arr, idx.reshape(-1, 1), axis=1)[:, 0]

    zk = gcol(zbar, kk)
    zk1 = gcol(zbar, kk + 1)
    delta = (hbl + zk) / (zk - zk1)
    om = 1.0 - delta
    om2, d2 = om * om, delta * delta

    def blend(interior, blmc, dk):
        intr = gcol(interior, kk)
        blc = gcol(blmc, kk)
        dkmp5 = caseA * intr + (1.0 - caseA) * blc
        dstar = om2 * dk + d2 * dkmp5
        return om * intr + delta * dstar                  # (N,)

    newM = blend(viscA, blmcM, dkm1[:, 0])
    newT = blend(diffKt, blmcT, dkm1[:, 1])
    newS = blend(diffKs, blmcS, dkm1[:, 2])

    k = jnp.arange(nl).reshape(1, -1)
    at_kk = (k == kk.reshape(-1, 1))
    blmcM = jnp.where(at_kk, newM.reshape(-1, 1), blmcM)
    blmcT = jnp.where(at_kk, newT.reshape(-1, 1), blmcT)
    blmcS = jnp.where(at_kk, newS.reshape(-1, 1), blmcS)
    ghats = jnp.where(at_kk, ghats * (1.0 - caseA).reshape(-1, 1), ghats)
    return blmcM, blmcT, blmcS, ghats


def _node_to_elem_visc(mesh: Mesh, viscA, cfg: KppConfig):
    """Element viscosity ``Av`` = 3-vertex mean of the (combined) node ``viscA`` on
    the element layer range, bottom-filled at ``nzmax``, surface-floored at ``minmix``
    (``fesom_kpp.c:911-924``). Same node→elem scatter as :func:`pp.pp_mixing`'s Av."""
    nl = mesh.nl
    corners = ops.gather_nodes_to_elem(viscA, mesh.elem_nodes)   # (E,3,nl)
    Av = corners.sum(axis=1) / 3.0
    k = jnp.arange(nl).reshape(1, -1)
    nzmin = (mesh.ulevels - 1).reshape(-1, 1)
    nzmax = (mesh.nlevels - 1).reshape(-1, 1)
    Av = jnp.take_along_axis(Av, jnp.clip(k, nzmin, nzmax - 1), axis=1)   # bottom-fill nzmax←nzmax-1
    Av = jnp.where(k == nzmin, jnp.maximum(Av, cfg.minmix), Av)           # surface floor
    return jnp.where(mesh.elem_iface_mask, Av, 0.0)


def assemble_mixing(mesh: Mesh, blmcM, blmcT, blmcS, ghats, viscA, diffKt, diffKs, kbl,
                    cfg: KppConfig):
    """smooth_blmc (3-sweep) + combine + node→elem — the KPP driver tail
    (``fesom_kpp.c:875-918``). Each ``blmc`` channel is smoothed 3× with the
    area-weighted node-patch smoother (:func:`eos.smooth_nod3D`); within the BL
    (``nz<kbl``) the interior ``viscA``/``diffKt``/``diffKs`` are raised to the smoothed
    ``blmc`` (``max``), and ``ghats`` is zeroed below the BL; the node ``viscA`` is then
    averaged to the element ``Av`` (+ bottom fill + ``minmix`` floor). ``Kv`` = the
    combined ``diffKt`` (the T-channel, used for BOTH T and S in CORE2).

    Returns ``(Kv [N,nl], Av [E,nl], viscA, diffKt, diffKs, ghats)`` — the final
    module-gate fields."""
    blmcM = eos.smooth_nod3D(mesh, blmcM, 3)
    blmcT = eos.smooth_nod3D(mesh, blmcT, 3)
    blmcS = eos.smooth_nod3D(mesh, blmcS, 3)

    k = jnp.arange(mesh.nl).reshape(1, -1)
    nzmin = (mesh.ulevels_nod2D - 1).reshape(-1, 1)
    nzmax = (mesh.nlevels_nod2D - 1).reshape(-1, 1)
    kblc = kbl.reshape(-1, 1)
    interior = (k >= nzmin + 1) & (k <= nzmax - 1)
    in_bl = interior & (k < kblc)
    below_bl = interior & (k >= kblc)
    viscA = jnp.where(in_bl, jnp.maximum(viscA, blmcM), viscA)
    diffKt = jnp.where(in_bl, jnp.maximum(diffKt, blmcT), diffKt)
    diffKs = jnp.where(in_bl, jnp.maximum(diffKs, blmcS), diffKs)
    ghats = jnp.where(below_bl, 0.0, ghats)

    Av = _node_to_elem_visc(mesh, viscA, cfg)
    return diffKt, Av, viscA, diffKt, diffKs, ghats
