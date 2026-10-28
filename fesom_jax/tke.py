"""CVMix classical-TKE driver + config — Phase 9b (the analogue of ``fesom_tke.c``,
the 534-LOC driver around the pure column core ``cvmix_tke.py``).

TKE is the project's **primary hybrid-ML seam**: a prognostic 1-equation turbulent-
kinetic-energy closure whose constants (``c_k``, ``c_eps``, ``cd``, ``alpha_tke``) are
exactly what Phase 7a tunes and Phase 7 replaces/augments with NNs. Differentiability and
``Params`` exposure are first-class here, not add-ons. The structural switches stay static
in :class:`TkeConfig`; the trainable constants live in :class:`~fesom_jax.params.Params`
(the GM ``k_gm`` precedent — a deliberate divergence from KPP's static-only constants,
justified by TKE being the designated ML seam).

Module split mirrors the C (locked decision 2): :mod:`fesom_jax.cvmix_tke` = the pure
column core (no mesh/state — the analogue of ``fesom_cvmix_tke.c``, 404 LOC, lands in JT.1)
+ this module = the driver/state/wiring (``fesom_tke.c``). Same split the C chose for dump
traceability (precedent: ``gm.py``/``gm_redi.py``).

**Config gate** (locked decision 3): the presence of a :class:`TkeConfig` ⇒ TKE mixing;
``tke_cfg=None`` ⇒ today's KPP/PP dispatch, byte-identical. The mixing dispatch in
:mod:`fesom_jax.step` becomes 3-way; it is an **error if both ``kpp_cfg`` and ``tke_cfg``
are set** (the C runs exactly one mixing scheme per process — fail loudly), and TKE raises
on the pi path (it needs ``stress_node_surf`` — the KPP precedent).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import precision as PR
from . import ops, pp
from . import tke_nn as _tke_nn
from .config import DENSITY_0, DT_DEFAULT
from .cvmix_tke import _safe_sqrt, _shift_down, integrate_tke_column


N2_SOURCES = ("smoothed", "local")   #: TkeConfig.n2_source (FESOM2: "smoothed")


class TkeConfig(NamedTuple):
    """Static classical-TKE config — the **classical Gaspar (1990) variant only** (the
    ICON/Brüggemann CVMix fork FESOM2 vendors), ``tke_mxl_choice=2``, ``only_tke``,
    **Neumann** surface+bottom BCs. Closed over the step / passed as a ``static_argname``,
    exactly like :class:`~fesom_jax.gm.GMConfig` / :class:`~fesom_jax.kpp.KppConfig` /
    :class:`~fesom_jax.ale.AleConfig` / ``IceConfig``. Hashable (all fields are Python
    scalars/bools); carries **no** differentiable leaves — the tunables ``c_k``/``c_eps``/
    ``cd``/``alpha_tke`` live in :class:`~fesom_jax.params.Params` (the ML-hook seam).

    **The presence of a ``TkeConfig`` ⇒ TKE; ``tke_cfg=None`` ⇒ the KPP/PP path**
    (byte-identical to the pre-Phase-9b model).

    Defaults = the verified CORE2 namelist.cvmix reference config (echo-verified in
    ``docs/tke_reference_namelists/PROVENANCE.md:43-58``, ported by ``fesom_tke.c:227-241``).
    All SI. The numeric structural statics live here; the Pr-law literal ``6.6`` stays a
    static double in :mod:`fesom_jax.cvmix_tke` (``TKE_C66``).

    **Un-ported combinations raise** (:meth:`validate`, the C ``fesom_tke_alloc:247-253``
    init-abort parity): IDEMIX (``not only_tke``), Langmuir (``l_lc``), Dirichlet BCs
    (``use_dirichlet``), ``mxl_choice != 2``. The port covers exactly the reference config;
    a future edit flipping one of these fails loudly rather than running un-ported physics.
    """

    # --- structural statics (the column-core bounds, fesom_tke.c:231-240) ---
    mxl_min: float = 1.0e-8       # mixing-length floor (Blanke–Delecluse)
    tke_min: float = 1.0e-6       # tke floor (Part 5 reset, only_tke)
    kappaM_max: float = 100.0     # KappaM ceiling (min(kappaM_max, c_k·mxl·√tke))

    # --- master gates (the C runs exactly the reference branch; else validate raises) ---
    mxl_choice: int = 2           # Blanke–Delecluse wall constraints (choice 2 only)
    only_tke: bool = True         # IDEMIX coupling OFF (.not.only_tke is gate-only)
    use_dirichlet: bool = False   # Dirichlet surface+bottom BCs (Neumann is executed)
    l_lc: bool = False            # Langmuir (tke_dolangmuir=F; gate-only)

    # --- test-only diagnostics (the 13 budget/aux terms; never in model State) ---
    with_diags: bool = False      # compute + return the budget diagnostics (JT.1 gate)

    # --- forward options that are NOT in FESOM2 (defaults = FESOM2, bitwise) ---
    n2_source: str = "smoothed"   # the N² TKE and its convective adjustment read: "smoothed" (FESOM2,
    #                               one sweep of eos.smooth_nod3D) or "local" (the unsmoothed node N²)
    convection: str = "switch"    # the shared convective adjustment: "switch" (FESOM2), "ramp",
    #                               "ramp_kv" (Kv ramp, Av switch), "off"
    conv_ramp_dn2: float = 1.0e-6  # "ramp"/"ramp_kv": N² width (s⁻²) of the one-sided ramp (pp.mo_convect)

    # NOTE: how the BACKWARD pass treats this closure is NOT set here — it is one flag of
    # :class:`fesom_jax.adjoint_stab.AdjointConfig` (``freeze_mixing=True``), passed to ``step`` as
    # ``adjoint_cfg``, so that every backward-only simplification lives in one object rather
    # than being scattered across the physics configs. See docs/ADJOINT_HORIZON.md.

    def validate(self) -> "TkeConfig":
        """Raise on an un-ported combination (the C ``fesom_tke_alloc:247-253`` abort
        parity). Called at the step seam whenever ``tke_cfg is not None``; returns ``self``
        so it can wrap an expression. ``tke_cfg=None`` is the KPP/PP path and never reaches
        here."""
        if (not self.only_tke or self.l_lc or self.use_dirichlet
                or self.mxl_choice != 2):
            raise ValueError(
                "TkeConfig only supports the classical-TKE reference config "
                "(only_tke=True, l_lc=False, use_dirichlet=False, mxl_choice=2). "
                "IDEMIX / Langmuir / Dirichlet BCs / mxl_choice!=2 are gate-only, not "
                "ported (no C reference run ⇒ no oracle). Use tke_cfg=None for the "
                "KPP/PP path. (fesom_tke.c:247-253 abort parity)")
        if self.n2_source not in N2_SOURCES:
            raise ValueError(f"TkeConfig.n2_source must be one of {N2_SOURCES}, "
                             f"got {self.n2_source!r}")
        if self.convection not in pp.CONVECTION_MODES:
            raise ValueError(f"TkeConfig.convection must be one of {pp.CONVECTION_MODES}, "
                             f"got {self.convection!r}")
        if not (math.isfinite(self.conv_ramp_dn2) and self.conv_ramp_dn2 > 0.0):
            raise ValueError(f"TkeConfig.conv_ramp_dn2 must be finite and > 0, "
                             f"got {self.conv_ramp_dn2!r}")
        return self


def _layer_center_Z(mesh, Z3d):
    """The per-node layer-center depths ``Z`` ``[N, nl]`` for the interface ΔZ (shear,
    dz_trr). ``Z3d`` is the live zstar layer centers (``ale.live_geometry``); ``None`` ⇒
    the static column-uniform ``mesh.Z`` (``nl-1`` mid-depths) padded to ``nl`` (the
    kpp.ri_iwmix convention). The padded tail row is unused (masked below-bottom)."""
    if Z3d is None:
        Zp = jnp.concatenate([mesh.Z, mesh.Z[-1:]])             # (nl,)
        return Zp[None, :]                                      # (1, nl) broadcast
    return Z3d                                                  # (N, nl) live


def _wire_kv_av(mesh, KappaM, KappaH, is_surf, is_bot, exch):
    """Wire the column-core ``KappaM``/``KappaH`` into ``(Kv, Av)`` (``fesom_tke.c:484-505``,
    PRE-``mo_convect``). Zero both at the surface (``uln0``) + below-bottom (``nln0+1``)
    interfaces (faithful — consumers read the interior only); ``Kv`` = the full ``KappaH``
    slab (the C ``aux->Kv = tke_Kv`` after its node exchange — refreshed by ``step.py``'s
    post-mixing ``Kv`` exchange); ``Av`` = the node→elem 3-vertex mean of the **exchanged**
    node ``KappaM`` over the interior element interfaces.

    ⚠️ The node ``KappaM`` MUST be halo-exchanged BEFORE the node→elem mean (``exch``,
    ``fesom_tke.c:491``): a boundary OWNED element has HALO vertices, so the gather reads
    the halo (the ``kpp.py:787-789`` internal-viscA precedent). Omit it and the port passes
    eager/1-device but FAILS the sharded N-vs-1 on boundary-element ``Av``. ``exch=None`` ⇒
    the identity (the dense/1-device path)."""
    zero_ends = is_surf | is_bot
    KappaM = jnp.where(zero_ends, 0.0, KappaM)                  # tke_Av[uln0]=tke_Av[nln0+1]=0
    KappaH = jnp.where(zero_ends, 0.0, KappaH)                  # tke_Kv[uln0]=tke_Kv[nln0+1]=0
    Kv = KappaH                                                 # aux->Kv = tke_Kv (full slab)

    # the REQUIRED internal node exchange before the gather (boundary owned elem ← halo vertices)
    KappaM_x = exch(KappaM, "nod") if exch is not None else KappaM
    corners = ops.gather_nodes_to_elem(KappaM_x, mesh.elem_nodes)   # (E, 3, nl)
    Av = corners.sum(axis=1) / 3.0                                  # node→elem 3-vertex mean
    # interior element interfaces only: nz ∈ [ule0+1, nle0-1] (fesom_tke.c:502)
    ke = jnp.arange(mesh.nl)[None, :]
    ule0 = (mesh.ulevels - 1)[:, None]
    nle0 = (mesh.nlevels - 1)[:, None]
    elem_interior = (ke >= ule0 + 1) & (ke <= nle0 - 1)
    Av = jnp.where(elem_interior, Av, 0.0)
    return Kv, Av


COLUMN_INPUTS = ("tke_old", "Ssqr", "Nsqr", "dzw", "dzt", "forc_tke_surf", "nlev")
"""The keys of :func:`tke_column_inputs` that are the positional inputs of
:func:`fesom_jax.cvmix_tke.integrate_tke_column`, in its order."""


def tke_column_inputs(mesh, uv, bvfreq, tke, stress_node_surf, hnode, *, Z3d=None,
                      bvfreq_raw=None):
    """Assemble the per-column inputs of the TKE column core from mesh/state
    (``fesom_tke.c:330-373``) — the part of :func:`mixing_tke` before the column solve.

    Returns a dict: the column-core inputs ``tke_old`` (= ``tke``), ``Ssqr`` (vertical shear²
    of the node velocity over the live layer-centre spacing, interior interfaces only),
    ``Nsqr`` (``bvfreq`` zeroed at the surface and bottom interfaces), ``dzw`` (= ``hnode``),
    ``dzt`` (interface spacing with ``hnode/2`` end caps), ``forc_tke_surf``
    (``|stress_node_surf|/ρ₀``), ``nlev`` (per-node layer count); plus ``uvnode`` (the
    element→node velocity the step reuses) and the interface masks ``is_surf``, ``is_bot``,
    ``is_interior``. ``integrate_tke_column(*(d[k] for k in COLUMN_INPUTS), ...)`` is
    exactly the solve :func:`mixing_tke` performs, so diagnostics that re-run the column core
    on these inputs see the model's own column state, not a re-implementation of it.

    ``bvfreq_raw`` (the unsmoothed N², only when a ``tke_pr_ratio`` / ``tke_pr_raw_cut``
    Prandtl or ``tke_ml_ratio`` / ``tke_ml_raw_cut`` mixing-length cut is on) adds
    ``Nsqr_raw``, zeroed where ``Nsqr`` is."""
    nl = mesh.nl
    k = jnp.arange(nl)[None, :]
    nzmin = (mesh.ulevels_nod2D - 1)[:, None]                   # surface interface (0, no cavity)
    nlev = (mesh.nlevels_nod2D - 1)                             # per-node layer count (bottom = nlev)
    nzmax = nlev[:, None]                                       # bottom interface
    is_surf = (k == nzmin)
    is_bot = (k == nzmax)
    is_interior = (k >= nzmin + 1) & (k <= nzmax - 1)           # interfaces [uln0+1, nln0]

    # --- per-column assembly (fesom_tke.c:330-373) ---
    uvnode = pp.compute_vel_nodes(mesh, uv)                     # element→node velocity
    # forc_tke_surf = |stress_node_surf|/ρ₀ (safe-norm: finite grad at zero wind)
    sx, sy = stress_node_surf[:, 0], stress_node_surf[:, 1]
    forc_tke_surf = _safe_sqrt(sx * sx + sy * sy) / DENSITY_0

    Z = _layer_center_Z(mesh, Z3d)                             # (N|1, nl) layer centers
    dZ = _shift_down(Z) - Z                                    # Z[nz-1]-Z[nz] (>0 interior; 0 at k=0)
    dZ_safe = jnp.where(dZ == 0.0, 1.0, dZ)                    # guard the masked surface/pad lanes
    du = _shift_down(uvnode[..., 0]) - uvnode[..., 0]         # u[nz-1]-u[nz]
    dv = _shift_down(uvnode[..., 1]) - uvnode[..., 1]
    vshear2 = jnp.where(is_interior, (du * du + dv * dv) / (dZ_safe * dZ_safe), 0.0)
    # bvfreq2 = the shared smoothed N², ZEROED at surface+bottom (nonzero only interior) —
    # ⚠️ a naive slice leaks the nonzero surface value (fesom_tke.c:362-364).
    bvfreq2 = jnp.where(is_interior, bvfreq, 0.0)
    # dz_trr: interior |ΔZ| (tracer-point spacing) + hnode/2 surface & bottom end caps
    dz_trr = jnp.where(is_interior, jnp.abs(dZ), 0.0)
    dz_trr = jnp.where(is_surf, hnode[:, :1] / 2.0, dz_trr)    # dz_trr[uln0]=hnode[uln0]/2
    hnode_bot = jnp.take_along_axis(hnode, (nlev - 1)[:, None], axis=-1)   # hnode[nln0]
    dz_trr = jnp.where(is_bot, hnode_bot / 2.0, dz_trr)        # dz_trr[nln0+1]=hnode[nln0]/2
    out = dict(tke_old=tke, Ssqr=vshear2, Nsqr=bvfreq2, dzw=hnode, dzt=dz_trr,
               forc_tke_surf=forc_tke_surf, nlev=nlev, uvnode=uvnode,
               is_surf=is_surf, is_bot=is_bot, is_interior=is_interior)
    if bvfreq_raw is not None:
        out["Nsqr_raw"] = jnp.where(is_interior, bvfreq_raw, 0.0)
    return out


def mixing_tke(mesh, uv, bvfreq, tke, stress_node_surf, hnode, cfg: TkeConfig,
               params, *, dt: float = DT_DEFAULT, exch=None, Z3d=None,
               freeze_convection: bool = False, cuts=frozenset(), tke_adj=None,
               freeze_state: bool = False, bvfreq_raw=None, conv_surrogate=None,
               ice_conc=None, turner=None):
    """Assembled classical-TKE vertical-mixing driver — the mirror of ``fesom_tke_mixing``
    (``fesom_tke.c:270-505``) followed by the shared ``mo_convect`` (``fesom_step.c:264``).
    Returns ``(Kv, Av, uvnode, tke_new)`` — the ``(Kv, Av, uvnode)`` triple PP/KPP return
    PLUS the advanced prognostic ``tke_new`` (the structural difference: TKE is stateful).

    Assembles the per-column inputs from mesh/state (:func:`tke_column_inputs`), pushes them
    through the pure column core (:func:`fesom_jax.cvmix_tke.integrate_tke_column`), wires
    ``Kv``/``Av``, and applies ``mo_convect`` — a drop-in at the ``step.py`` mixing seam.
    ``exch`` is REQUIRED for the internal node-``Av`` exchange (see :func:`_wire_kv_av`);
    ``Z3d`` is the live zstar layer-center geometry (``None`` ⇒ the static mesh,
    byte-identical under linfs). ``cuts``: the ``tke.*`` diagnostic derivative cuts of
    ``AdjointConfig.diag_cuts``, passed to the column core (empty ⇒ the identical program).
    ``tke_adj``: ``AdjointConfig.tke_options()`` (the column core's candidate settings).
    ``freeze_state`` (``AdjointConfig.freeze_mixing_state``): the column inputs that carry the
    ocean state — shear², N², the layer/interface thicknesses and the surface flux — enter
    the closure as given constants in the backward; ``tke`` and the constants stay live.
    ``bvfreq_raw``: the unsmoothed N² the ``tke_pr_ratio`` / ``tke_pr_raw_cut`` Prandtl and
    ``tke_ml_ratio`` / ``tke_ml_raw_cut`` mixing-length cuts read (as a forward value only;
    ``None`` unless one of them is on). ``conv_surrogate``: ``AdjointConfig.conv_surrogate``,
    the backward-only ramp of the convective-adjustment switch (:func:`fesom_jax.pp.mo_convect`).
    ``ice_conc``: the sea-ice concentration ``[N]`` the ``tke_pr_ice_cut`` Prandtl cut reads (as a
    forward value only; ``None`` unless it is on). The ``tke_pr_lat_cut`` cut reads the
    geographic latitude from ``mesh``. ``turner``: the interface Turner angle ``[N, nl]`` the
    ``tke_pr_tu_cut`` cut reads (:func:`fesom_jax.eos.turner_angle`; ``None`` unless it is on).
    ``cfg.convection`` / ``cfg.conv_ramp_dn2``: the forward convective-adjustment mode (default
    ``"switch"`` = FESOM2).
    """
    inp = tke_column_inputs(mesh, uv, bvfreq, tke, stress_node_surf, hnode, Z3d=Z3d,
                            bvfreq_raw=bvfreq_raw)
    if freeze_state:
        sg = jax.lax.stop_gradient
        inp = {**inp, **{n: sg(inp[n]) for n in ("Ssqr", "Nsqr", "dzw", "dzt", "forc_tke_surf")}}
        hnode = inp["dzw"]
    uvnode, forc_tke_surf = inp["uvnode"], inp["forc_tke_surf"]
    vshear2, bvfreq2, dz_trr, nlev = inp["Ssqr"], inp["Nsqr"], inp["dzt"], inp["nlev"]
    is_surf, is_bot, is_interior = inp["is_surf"], inp["is_bot"], inp["is_interior"]

    # --- optional structure-preserving NN multiplier on c_k/c_eps/c_d (the §3 hybrid-ML hook,
    # Phase A5). ``params.tke_nn is None`` ⇒ the scalar constants pass through UNCHANGED
    # (byte-identical to the pre-A5 model); a zero-last-layer NN ⇒ multiplier 1 ⇒ STILL
    # bit-identical (``c_k·1 == c_k``). The NN reads the LIVE per-column inputs assembled above
    # (forcing, f, depth, interior-mean N²/shear², surface TKE) → a bounded, positive-definite
    # multiplier ``[N, 3]``, broadcast over levels via the ``[N,1]`` shape. ---
    c_k, c_eps, cd = params.tke_c_k, params.tke_c_eps, params.tke_cd
    nn = getattr(params, "tke_nn", None)
    if nn is not None:
        feats = _tke_nn.column_features(
            forc_tke_surf, mesh.coriolis_node, mesh.depth,
            bvfreq2, vshear2, tke, is_interior)
        m = _tke_nn.multiplier(nn, feats)                      # [N, 3] ∈ (1/m_max, m_max)
        c_k = params.tke_c_k * m[:, 0:1]                       # [N,1] → broadcast over levels
        c_eps = params.tke_c_eps * m[:, 1:2]
        cd = params.tke_cd * m[:, 2:3]

    # --- column core (fesom_tke.c:421-442) ---
    # single precision: CVMix runs in cvmix_r8 upstream — WP inputs copied into real64 buffers, WP namelist
    # constants widened, the outputs copied back to WP (cvmix_driver/gen_modules_cvmix_tke.F90:290-298).
    _f, _l = PR.full, PR.lit
    _f_opt = lambda x: None if x is None else _f(x)       # noqa: E731  (the adjoint's optional inputs)
    tke_new, KappaM, KappaH = integrate_tke_column(
        _f(tke), _f(vshear2), _f(bvfreq2), _f(hnode), _f(dz_trr), _f(forc_tke_surf), nlev,
        c_k=_f(c_k), c_eps=_f(c_eps), cd=_f(cd),
        alpha_tke=_f(params.tke_alpha), mxl_min=_l(cfg.mxl_min), tke_min=_l(cfg.tke_min),
        kappaM_max=_l(cfg.kappaM_max), dt=dt, with_diags=False, cuts=cuts, adj=tke_adj,
        n2_raw=_f_opt(inp.get("Nsqr_raw")), ice_conc=_f_opt(ice_conc),
        lat=(None if tke_adj is None or tke_adj.pr_lat is None
             else _f(mesh.geo_coord_nod2D[:, 1])), turner=turner)
    tke_new, KappaM, KappaH = PR.wp(tke_new), PR.wp(KappaM), PR.wp(KappaH)

    # --- Kv/Av wiring (fesom_tke.c:484-505, pre-mo_convect) + the shared mo_convect ---
    Kv, Av = _wire_kv_av(mesh, KappaM, KappaH, is_surf, is_bot, exch)
    Kv, Av = pp.mo_convect(mesh, Kv, Av, bvfreq, freeze_increment=freeze_convection,
                           surrogate_dn2=conv_surrogate, mode=cfg.convection,
                           ramp_dn2=cfg.conv_ramp_dn2)
    return Kv, Av, uvnode, tke_new
