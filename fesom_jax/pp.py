"""Pacanowski-Philander vertical mixing + convective adjustment (substep 4 / Task 2.3).

Literal vectorized port of ``fesom_compute_vel_nodes`` + ``fesom_pp_mixing`` +
``fesom_mo_convect`` (``fesom_pp.c``, driven from ``fesom_step.c:122-135``) for the
Phase-2 pi config (PP scheme, ``use_instabmix`` convective adjustment on, no
momix/windmix). Outputs the substep-4 dump fields:

* ``Kv`` — tracer vertical diffusivity, **node**, written on **interior** interfaces
  ``nz ∈ [nzmin+1, nzmax)`` only (surface ``nzmin`` and bottom ``nzmax`` stay 0,
  exactly as the dump shows);
* ``Av`` — momentum vertical viscosity, **element**, same interior-interface range.

PP structure (the order is load-bearing, ``fesom_pp.c:62-145``):

1. ``factor = shear² / (shear² + 5·max(N²,0) + 1e-14)``   (overwrites Kv; dimensionless)
2. ``Av = mix_coeff · mean_i(factor_i²) + A_ver``          (reads factor BEFORE step 3)
3. ``Kv = mix_coeff · factor³ + K_ver``

then ``mo_convect``: where ``N²<0`` bump ``Kv``/``Av`` up to ``instabmix_kv``.

``shear`` uses node-interpolated velocity ``uvnode`` (area-weighted element→node
average). At rest (uv=0) shear=0 → factor=0 → ``Kv=K_ver``, ``Av=A_ver``; the
shear/N² path is exercised by the synthetic unit + gradient tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import adjoint_stab, ops
from .config import A_VER, INSTABMIX_KV, K_VER, MIX_COEFF_PP, USE_INSTABMIX

CONVECTION_MODES = ("switch", "ramp", "ramp_kv", "off")   #: mo_convect forward modes (FESOM2: "switch")
from .mesh import Mesh


def _interior_iface_mask(ulevels, nlevels, nl):
    """``[n_entity, nl]`` bool for the PP/convection interior interfaces
    ``nz ∈ [ulevels-1+1, nlevels-1)`` = ``[ulevels, nlevels-1)``."""
    k = jnp.arange(nl).reshape(1, -1)
    lo = jnp.asarray(ulevels).reshape(-1, 1)              # = nzmin+1
    hi = (jnp.asarray(nlevels) - 1).reshape(-1, 1)        # = nzmax
    return (k >= lo) & (k < hi)


def _shift_down(x):
    """``out[..., k] = x[..., k-1]`` with edge replication at ``k=0`` (so Δ=0
    there — finite, AD-safe; the k=0 interface is masked off anyway)."""
    return jnp.concatenate([x[..., :1], x[..., :-1]], axis=-1)


def compute_vel_nodes(mesh: Mesh, uv):
    """Area-weighted element→node velocity interpolation (``compute_vel_nodes``).

    ``uv`` is ``[elem2D, nl, 2]`` → ``uvnode`` ``[nod2D, nl, 2]`` (layer field). Per
    node/layer: ``Σ_{el∋n} area_el·uv[el] / Σ area_el`` over the element's valid
    layers (``elem_layer_mask``)."""
    e, nl = mesh.elem2D, mesh.nl
    area = mesh.elem_area[:, None, None]                          # (elem2D,1,1)
    lmask = mesh.elem_layer_mask[..., None]                       # (elem2D,nl,1)
    contrib = jnp.where(lmask, area * uv, 0.0)                    # (elem2D,nl,2)
    area_lev = jnp.where(mesh.elem_layer_mask,
                         jnp.broadcast_to(mesh.elem_area[:, None], (e, nl)), 0.0)

    num = ops.scatter_add(jnp.broadcast_to(contrib[:, None], (e, 3, nl, 2)),
                          mesh.elem_nodes, mesh.nod2D, inv=mesh.inv_elem_nodes)            # (nod2D,nl,2)
    den = ops.scatter_add(jnp.broadcast_to(area_lev[:, None], (e, 3, nl)),
                          mesh.elem_nodes, mesh.nod2D, inv=mesh.inv_elem_nodes)            # (nod2D,nl)
    safe = jnp.where(den > 0.0, den, 1.0)[..., None]
    uvnode = jnp.where(mesh.node_layer_mask[..., None], num / safe, 0.0)
    return uvnode


def pp_mixing(mesh: Mesh, uvnode, bvfreq, *, k_ver=K_VER, a_ver=A_VER, Z3d=None):
    """``(Kv, Av)`` from node velocity ``uvnode`` ``[nod2D,nl,2]`` and the
    (post-smooth) ``bvfreq`` ``[nod2D,nl]``. Mirrors ``fesom_pp_mixing`` exactly.

    ``k_ver``/``a_ver`` are the background tracer diffusivity / momentum viscosity.
    They default to the config constants (``K_VER``/``A_VER`` — the Phase-2 path),
    but accept **traced** values so ``d(loss)/d(k_ver)`` flows (the ML-hook seam,
    :class:`fesom_jax.params.Params`).

    ``Z3d`` (zstar live mid-depths ``[nod2D,nl]``; ``None`` ⇒ static, byte-identical)
    re-points the shear layer spacing ``dz`` to the moving coordinate (JZ.6)."""
    nl = mesh.nl
    # shear layer spacing dz = Z[nz-1]-Z[nz]: static column-uniform or per-node live.
    if Z3d is None:
        Zp = jnp.concatenate([mesh.Z, mesh.Z[-1:]])
        dz = _shift_down(Zp) - Zp                          # Z[nz-1]-Z[nz]
        dz = dz.at[0].set(1.0)                             # k=0 unused; avoid 0
        # The bottom/pad interfaces have dz==0 (the Zp tail duplicates Z[-1]); they are masked
        # out downstream (``nmask``), but ``1/dz`` is +inf there and the AD backward of the
        # masked ``shear`` below is then 0·inf=NaN — a masked-NaN trap the sharded reverse pass
        # exposes (the dense single-device XLA folds the structural zero; shard_map/check_vma
        # keeps it). Guard the divisor so ``dz_inv`` is FINITE on those lanes. Forward
        # byte-identical: the inf lanes were always masked (the 2-yr v1.0 run proves they carry
        # no live output), so the value on every unmasked lane is unchanged.
        safe_dz = jnp.where(dz != 0.0, dz, 1.0)
        dz_inv = (1.0 / safe_dz)[None, :]                  # (1,nl), finite everywhere
    else:
        dz = _shift_down(Z3d) - Z3d                        # (nod2D,nl) per-node Z[nz-1]-Z[nz]
        safe_dz = jnp.where(dz != 0.0, dz, 1.0)            # surface/pad lanes (dz=0) → finite
        dz_inv = 1.0 / safe_dz                             # (nod2D,nl)

    u, v = uvnode[..., 0], uvnode[..., 1]                  # (nod2D,nl)
    du = _shift_down(u) - u                                # u[nz-1]-u[nz]
    dv = _shift_down(v) - v
    shear = (du * du + dv * dv) * dz_inv * dz_inv          # (nod2D,nl)
    nsq_pos = jnp.maximum(bvfreq, 0.0)
    factor = shear / (shear + 5.0 * nsq_pos + 1.0e-14)     # dimensionless ratio

    nmask = _interior_iface_mask(mesh.ulevels_nod2D, mesh.nlevels_nod2D, nl)
    factor = jnp.where(nmask, factor, 0.0)

    # Av at elements from mean of factor² over the 3 vertices (+ background)
    f_corners = ops.gather_nodes_to_elem(factor, mesh.elem_nodes)   # (elem2D,3,nl)
    f2mean = (f_corners * f_corners).sum(axis=1) / 3.0              # (elem2D,nl)
    emask = _interior_iface_mask(mesh.ulevels, mesh.nlevels, nl)
    Av = jnp.where(emask, MIX_COEFF_PP * f2mean + a_ver, 0.0)

    # Kv = mix_coeff·factor³ + background (the Kv0_const branch)
    Kv = jnp.where(nmask, MIX_COEFF_PP * factor ** 3 + k_ver, 0.0)
    return Kv, Av


def mo_convect(mesh: Mesh, Kv, Av, bvfreq, *, freeze_increment: bool = False,
               surrogate_dn2: float | None = None, mode: str = "switch",
               ramp_dn2: float | None = None):
    """Convective adjustment (``fesom_mo_convect``, use_instabmix branch): where
    ``N²<0`` raise ``Kv`` (node) / ``Av`` (element, if ANY vertex N²<0) to
    ``instabmix_kv``. No-op where the column is stably stratified.

    ``mode`` (a forward option, :attr:`fesom_jax.tke.TkeConfig.convection`; FESOM2 has only
    the first): ``"switch"`` the adjustment above; ``"off"`` none (``Kv``/``Av`` returned
    as given: the closure alone mixes unstable columns); ``"ramp"`` the switch smoothed on the
    convecting side, ``Kv_in + w·(max(Kv_in, instabmix_kv) − Kv_in)`` with
    ``w = clip(−N²/ramp_dn2, 0, 1)`` (node), and for ``Av`` the largest ``w`` of the element's
    vertices: the full bump for ``N² ≤ −ramp_dn2``, none for ``N² ≥ 0``, so the flux
    ``Kv·N²`` stays monotone in N² (a ramp on the stable side would be anti-diffusive);
    ``"ramp_kv"`` the node ``Kv`` ramp with the ``Av`` switch.

    ``surrogate_dn2`` (``AdjointConfig.conv_surrogate``, backward only): the node ``Kv`` keeps
    its value, and its derivative is that of ``Kv_in + w·(max(Kv_in, instabmix_kv) − Kv_in)``
    with the one-sided ramp ``w = clip(−N²/surrogate_dn2, 0, 1)``: the switch gets the N²
    derivative ``−(max(Kv_in, instabmix_kv) − Kv_in)/surrogate_dn2`` where
    ``−surrogate_dn2 < N² < 0`` and none elsewhere; outside the band the derivative is the
    exact one. The ramp sits on the convecting side only: on the stable side a falling
    ``Kv(N²)`` makes the linearised flux ``Kv·N²`` decrease with N² (anti-diffusive), and a
    centred ramp blew the 30-day reverse sweep up in its first chunk. ``Av`` keeps the exact
    switch."""
    if mode not in CONVECTION_MODES:
        raise ValueError(f"mo_convect mode must be one of {CONVECTION_MODES}, got {mode!r}")
    if mode != "switch" and (freeze_increment or surrogate_dn2 is not None):
        raise ValueError("freeze_convection / conv_surrogate act on the convective switch; "
                         f"the forward convection mode is {mode!r}")
    if not USE_INSTABMIX or mode == "off":
        return Kv, Av
    Kv_in, Av_in = Kv, Av
    nl = mesh.nl
    imix = INSTABMIX_KV
    nmask = _interior_iface_mask(mesh.ulevels_nod2D, mesh.nlevels_nod2D, nl)
    if mode in ("ramp", "ramp_kv"):
        if ramp_dn2 is None or not ramp_dn2 > 0.0:
            raise ValueError(f"mo_convect ramp needs ramp_dn2 > 0, got {ramp_dn2!r}")
        # (1 − w)·a + w·b: exactly the switch's value at w = 1 and the closure's at w = 0
        w = jnp.clip(-bvfreq / ramp_dn2, 0.0, 1.0)                    # (nod2D,nl)
        wn = jnp.where(nmask, w, 0.0)
        Kv = (1.0 - wn) * Kv_in + wn * jnp.maximum(Kv_in, imix)
        if mode == "ramp":
            w_e = jnp.max(ops.gather_nodes_to_elem(w, mesh.elem_nodes), axis=1)   # (elem2D,nl)
            we = jnp.where(_interior_iface_mask(mesh.ulevels, mesh.nlevels, nl), w_e, 0.0)
            Av = (1.0 - we) * Av_in + we * jnp.maximum(Av_in, imix)
            return Kv, Av
    else:
        Kv = jnp.where(nmask & (bvfreq < 0.0), jnp.maximum(Kv, imix), Kv)

    b_corners = ops.gather_nodes_to_elem(bvfreq, mesh.elem_nodes)   # (elem2D,3,nl)
    any_neg = jnp.any(b_corners < 0.0, axis=1)                      # (elem2D,nl)
    emask = _interior_iface_mask(mesh.ulevels, mesh.nlevels, nl)
    Av = jnp.where(emask & any_neg, jnp.maximum(Av, imix), Av)
    if freeze_increment:
        # Backward-only (AdjointConfig.freeze_convection): keep the bump's VALUE but differentiate
        # only the pre-convection coefficients, so the N^2<0 switch contributes no gradient
        # while the closure's own state-dependence stays live.
        # Straight-through form (adjoint_stab.freeze_increment): the forward is exactly the
        # bumped value, which the increment form Kv_in + sg(Kv - Kv_in) does not guarantee.
        Kv, Av = adjoint_stab.freeze_increment((Kv_in, Av_in), (Kv, Av))
    if surrogate_dn2 is not None:
        w = jnp.where(nmask, jnp.clip(-bvfreq / surrogate_dn2, 0.0, 1.0), 0.0)
        Kv = adjoint_stab.freeze_increment(Kv_in + w * (jnp.maximum(Kv_in, imix) - Kv_in), Kv)
    return Kv, Av


def mixing_pp(mesh: Mesh, uv, bvfreq, *, k_ver=K_VER, a_ver=A_VER, Z3d=None,
              freeze_convection: bool = False, conv_surrogate: float | None = None):
    """Driver mirror of ``fesom_step.c:122-135``: node velocity → PP → convection.
    Returns ``(Kv, Av, uvnode)`` — the substep-4 dump fields (+ uvnode for reuse).
    ``k_ver``/``a_ver`` thread the differentiable backgrounds (default = config).
    ``Z3d`` (zstar live mid-depths; ``None`` ⇒ static, byte-identical) re-points the PP
    shear spacing (JZ.6)."""
    uvnode = compute_vel_nodes(mesh, uv)
    Kv, Av = pp_mixing(mesh, uvnode, bvfreq, k_ver=k_ver, a_ver=a_ver, Z3d=Z3d)
    Kv, Av = mo_convect(mesh, Kv, Av, bvfreq, freeze_increment=freeze_convection,
                        surrogate_dn2=conv_surrogate)
    return Kv, Av, uvnode
