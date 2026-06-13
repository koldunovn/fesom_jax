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

from typing import NamedTuple


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
        return self


def mixing_tke(mesh, uv, bvfreq, tke, stress_node_surf, hnode, cfg: TkeConfig,
               params, *, exch=None, Z3d=None, zbar3=None):
    """Assembled classical-TKE vertical-mixing driver — the mirror of ``fesom_tke_mixing``
    (``fesom_tke.c:270-534``) followed by the shared ``mo_convect``. Returns
    ``(Kv, Av, uvnode, tke_new)`` — the ``(Kv, Av, uvnode)`` triple PP/KPP return PLUS the
    advanced prognostic ``tke_new`` (the structural difference: TKE is stateful).

    The driver assembles per-column inputs (``forc_tke_surf`` from the ice-blended nodal
    stress, ``vshear2`` from ``uvnode`` diffs, ``bvfreq2`` = the shared smoothed N²,
    ``dz_trr`` interface spacings with hnode/2 end caps — ALL geometry derived from
    ``hnode`` + the Z-source, so linfs=static / zstar=live for free), pushes each through
    the pure column core (:mod:`fesom_jax.cvmix_tke`), zeros Kv/Av at the surface +
    below-bottom interfaces, copies Kv full-slab, **exchanges node ``tke_Av`` (via
    ``exch``) BEFORE** the node→elem 3-vertex Av mean (boundary owned elements have halo
    vertices — the ``kpp.py:787-789`` internal-viscA precedent, REQUIRED for the sharded
    N-vs-1 gate), and applies ``mo_convect``.

    ``exch`` is REQUIRED (the internal node-Av exchange); ``Z3d``/``zbar3`` are the live
    zstar geometry (``None`` ⇒ the static mesh geometry, byte-identical under linfs).
    """
    raise NotImplementedError(
        "mixing_tke lands in Phase 9b JT.2 (the column core is JT.1). JT.0 wires the "
        "config seam + State.tke + dispatch skeleton only — tke_cfg=None is byte-identical.")
