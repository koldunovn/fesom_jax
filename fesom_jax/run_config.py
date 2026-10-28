"""Unified, single-YAML run configuration for the FESOM2 → JAX model (Task A3).

:class:`RunConfig` **composes** the existing per-physics sub-configs (``AleConfig`` /
``GMConfig`` / ``KppConfig`` / ``TkeConfig`` / ``IceConfig``) with the **promoted**
run-dependent scalars the model paper needs to vary per mesh — the horizontal-viscosity γ's
(:class:`~fesom_jax.config.ViscConfig`; NG5 wants ``gamma1=0.2``), the timestep ``dt`` and its
cold→prod **ramp**, and the tracer scheme selector (:class:`~fesom_jax.config.TracerConfig`) —
plus the run-orchestration spec (mesh / forcing / output / restart / length / cadences) the A6
driver consumes. One YAML file drives an entire invocation.

**Bit-identity is the regression guard.** :meth:`RunConfig.defaults` reproduces today's
``config.py`` values exactly — all physics sub-configs ``None`` (linfs / PP / no-GM / no-ice),
``ViscConfig()`` = the module γ's, ``dt = DT_DEFAULT`` — so a default-config step is **byte-for-
byte** identical to a bare :func:`fesom_jax.step.step` (asserted in ``test_run_config.py``).

**Scope (YAGNI).** Only mesh/run-dependent settings are promoted; physical constants
(``G``/``OMEGA``/``R_EARTH``) stay module-level. This is **not** a ``namelist.oce`` mimic — the
docstrings map the concepts, but the format is a clean YAML (see :func:`to_yaml`).

YAML schema (every key optional; absent ⇒ the bit-identical default)::

    ale:   {}            # {} = zstar on (defaults); null/absent = linfs
    gm:    null          # null = GM off; {} or {k_gm_min: …} = on (+ overrides)
    kpp:   {}            # the vertical-mixing scheme sub-configs (mutually exclusive: kpp xor tke)
    tke:   null
    ice:   {}            # mEVP sea ice (needed for NG5 LKFs)
    visc:  {gamma1: 0.2} # only the NON-default γ's (NG5)
    tracer: {}           # the implemented MFCT/QR4C/FCT (0,1); other schemes raise
    dt: 180.0
    dt_ramp: {after_step: 480, dt: 240.0}   # cold 180 → prod 240 (a dt-CHANGE across restart)
    mesh: /work/.../ng5
    partition: dist_64
    forcing: {kind: core2, start_year: 1958}
    # …plus OPTIONAL input-path overrides inside `forcing:` (absent ⇒ $FESOM_JRA_DIR /
    # $FESOM_SSS_PATH / $FESOM_RUNOFF_PATH / $FESOM_CHL_PATH, else the Levante default —
    # see fesom_jax/paths.py + docs/DATA.md):
    #   forcing: {kind: core2, start_year: 1958, jra_dir: /data/JRA55-do-v1.4.0,
    #             sss_path: …, runoff_path: …, chl_path: …}
    output_dir: /work/.../ng5_out
    snapshot_every: 240     # steps; 0 = off
    checkpoint_every: 480   # steps; 0 = off
    restart_in: null
    restart_out: /work/.../ng5_out/restart
    restart_archive_out: /work/.../ng5_out/restart_archive   # null = archive off
    restart_archive_period: year    # 'year' | 'month' | 'day'
    restart_archive_length: 1       # fire every Nth `period` boundary (1 = every one)
    n_steps: 1440           # OR a duration string the A6 driver parses
    duration: 2yr
    params_file: /work/.../gm_nn_stage0.npz   # trained gm_nn weights; absent/null = default GM

``params_file`` is the **deployment path for a trained closure** (NN-GM plan Task 7): the ``.npz``
:func:`fesom_jax.gm_nn.save_gm_nn` writes, re-read on every leg of a restart chain, so which
weights a run used is a property of its config rather than of the submitting shell. Absent or
``null`` ⇒ ``params.gm_nn is None`` ⇒ default GM/Redi, byte-identical to a pre-seam run.

``restart_out``/``checkpoint_every`` are the ROLLING restart (crash recovery + chain hand-off —
frequent, single directory, always overwritten, mirrors nothing before it existed). By contrast
``restart_archive_*`` is a SEPARATE, calendar-cadenced stream: each firing writes an immutable,
uniquely-named directory (mirrors FESOM3's ``fesom.<YYYY>.<DDD>.<SSSSS>`` convention exactly —
see ``fesom_jax/run.py``'s ``_archive_tag``) that is never overwritten or auto-deleted, so a past
restart is always resumable (point ``--restart-in`` at it directly) for branching experiments.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple, Optional

from .ale import AleConfig
from .config import DT_DEFAULT, TracerConfig, ViscConfig
from .gm import GMConfig
from .ice import IceConfig
from .kpp import KppConfig
from .recom.config import RecomConfig
from .tke import TkeConfig


_ALE_SE: dict = {}       # RunConfig.ale_cfg's objects, one per (AleConfig, SeConfig) value


class DtRamp(NamedTuple):
    """A single cold→prod timestep change: switch to ``dt`` once ``step >= after_step``.

    A dt change INVALIDATES the AB2 history formed at the old dt ⇒ the A6 driver re-bootstraps
    AB2 (``is_first_step=True``) on the first post-ramp step. ``after_step`` is an absolute step
    index so a resumed job (which persists its step counter) ramps at the right moment."""
    after_step: int
    dt: float


# Map each physics slot name → its sub-config type (the (de)serialization registry).
_SUBCONFIGS = {
    "ale": AleConfig, "gm": GMConfig, "kpp": KppConfig,
    "tke": TkeConfig, "ice": IceConfig, "recom": RecomConfig,
}
# Non-physics scalar/spec fields that pass through YAML verbatim.
_SPEC_FIELDS = ("dt", "mesh", "partition", "precision", "forcing", "ssh", "gpu", "output_dir",
                "snapshot_every",
                "checkpoint_every", "restart_in", "restart_out",
                "restart_archive_out", "restart_archive_period", "restart_archive_length",
                "n_steps", "duration", "params_file")
_ARCHIVE_PERIODS = ("year", "month", "day")
# The `forcing:` mapping — WHAT the forcing is (`kind`/`start_year`) plus WHERE its input
# files live. The four path keys are optional overrides of the env var / Levante default
# (:mod:`fesom_jax.paths`); absent ⇒ the reader resolves them itself. Strict-keyed like every
# other block: an unknown key raises rather than being silently ignored.
# `mslp` (bool, default false) is FESOM2's l_mslp: read the JRA55-do sea-level pressure (psl) and add its
# gradient to the momentum equation (oce_ale_vel_rhs.F90:208); REcoM reads the same pressure for its gas
# solubility, so a REcoM run requires it (validate).
# `dust_file` (full REcoM): the Albani DustClim(months, nod2d) file of the mesh; absent ⇒ the REcoM namelist's
# nm_fe_data_file when that is an absolute path (FESOM2 reads it from ClimateDataPath, which a run YAML does not have).
# `co2_file` (full REcoM with constant_CO2 off): the AtmCO2_<year> monthly time series; absent ⇒ the namelist's
# nm_co2_data_file when that is an absolute path (as dust_file).
_FORCING_KEYS = ("kind", "start_year", "jra_dir", "sss_path", "runoff_path", "chl_path",
                 "on_device", "mslp", "dust_file", "co2_file")
_FORCING_PATH_KEYS = ("jra_dir", "sss_path", "runoff_path", "chl_path")
# The `ssh:` mapping — SSH CG solver options. `cheb_degree` (int ≥1) enables the degree-k
# Chebyshev polynomial preconditioner (CGPOLY, ssh.enable_cheb_precond) in place of the
# MITgcm M⁻¹ — a many-node comm lever (fewer CG iterations ⇒ fewer psums); absent/0 ⇒ the
# byte-identical default. `cheb_kappa` tunes the assumed condition number (default 30).
# `gathered` (bool, default false) solves the 2-D system on every device after one all-gather
# instead of the sharded CG (ssh.solve_ssh_gathered) — an option for small meshes only.
# `split_explicit` (a mapping of fesom_jax.ssh_se.SeConfig fields, {} = the defaults M 50, chi 0.1): replace the CG
# by the split-explicit barotropic subcycling (a PHYSICS change, Sergey Danilov's AB3-AM4 scheme as the Kokkos port
# built it — a stated deviation from upstream FESOM2; zstar only). It rides on the step as
# AleConfig.split_explicit (:meth:`RunConfig.ale_cfg`). Absent/null ⇒ the CG.
_SSH_KEYS = ("cheb_degree", "cheb_kappa", "gathered", "split_explicit")
# The `gpu:` mapping — layout/transport options for GPU runs (CORE2 GPU speed plan, Task 7). Each
# key maps to the trace-time environment switch that implements it; the values move by round-off
# at most. `halo_compact`: the all_gather halo ships only the lanes other devices read
# (halo.halo_exchange_compact, FESOM_HALO_COMPACT); `column_unroll`: the level sweeps run unrolled
# on the [..., K] arrays (ops.unrolled_scan, FESOM_COLUMN_UNROLL). Off when absent.
# `mevp_wide` (an integer K >= 0, not a bool): the sharded mEVP exchanges once every K subcycles on
# K-ring ice lanes (fesom_jax.ice_wide, FESOM_MEVP_WIDE; owned lanes bitwise the per-subcycle
# exchange on CPU); 0/1 = the per-subcycle exchange. `ssh_ell`: the SSH matvecs on the operator's ELL
# layout (ssh.to_ell, FESOM_SSH_ELL; the segment_sum to round-off, no atomics on a GPU).
# `cg_batch` (an integer k >= 1): the CG runs k masked iterations per while-loop trip, the same iterate
# with k times fewer host syncs (ssh.cg_batch, FESOM_CG_BATCH); 0/1 = one iteration per trip.
# `ordered_sums`: the element/edge -> node/element assemblies (ops.scatter_add on mesh.elem_nodes /
# edges / edge_tri) as ordered gathers over per-device inverse maps (ops.ordered_sum,
# FESOM_ORDERED_SUMS; the segment_sum to round-off, no atomics on a GPU, deterministic).
_GPU_ENV = {"halo_compact": "FESOM_HALO_COMPACT", "column_unroll": "FESOM_COLUMN_UNROLL",
            "mevp_wide": "FESOM_MEVP_WIDE", "ssh_ell": "FESOM_SSH_ELL", "cg_batch": "FESOM_CG_BATCH",
            "mevp_unroll": "FESOM_MEVP_UNROLL", "column_kernel": "FESOM_COLUMN_KERNEL",
            "ssh_inc_ell": "FESOM_SSH_INC_ELL", "ordered_sums": "FESOM_ORDERED_SUMS",
            "forcing_procs": "FESOM_FORCING_PROCS", "output_async": "FESOM_OUTPUT_ASYNC",
            "forcing_ahead": "FESOM_FORCING_AHEAD", "restart_async": "FESOM_RESTART_ASYNC",
            "se_wide": "FESOM_SE_WIDE", "se_unroll": "FESOM_SE_UNROLL", "step_unroll": "FESOM_STEP_UNROLL",
            "ice_static": "FESOM_ICE_STATIC", "chunk_tail": "FESOM_CHUNK_TAIL", "bgc_layout": "FESOM_BGC_LAYOUT",
            "bgc_halo": "FESOM_BGC_HALO"}
# `bgc_layout`: how the REcoM tracer stage runs over the tracer axis (a string): "major" (default: one batched pass
# over the stack held tracer-major in memory, one halo exchange per FCT stage), "vmap" (the batched pass over the
# trailing tracer axis, XLA's layout) or "loop" (one tracer at a time, the reference; fesom_jax.recom.transport).
# `se_wide` (an integer K >= 0): the sharded split-explicit substeps exchange once every K substeps on K+1-ring lanes
# (fesom_jax.se_wide, FESOM_SE_WIDE; owned lanes bitwise the per-substep exchange on CPU); 0/1 = two exchanges per
# substep. `se_unroll` (an integer u >= 1): u substeps per loop trip (ssh_se.se_unroll, FESOM_SE_UNROLL; round-off).
# Both need ssh.split_explicit.
# `step_unroll` (an integer u >= 1): the forced runner's step scan runs u steps per loop trip, so the host syncs on the
# loop predicate u times less often (integrate_sharded.step_unroll, FESOM_STEP_UNROLL; round-off). `ice_static`
# (ice_ave_steps > 1): each step's ice update branch is fixed at trace time from the cycle's phase, no lax.cond and
# no host sync a step (run_steps_sharded_forced ice_phase, FESOM_ICE_STATIC).
_GPU_INT = ("mevp_wide", "cg_batch", "mevp_unroll", "forcing_procs", "forcing_ahead", "se_wide", "se_unroll",
            "step_unroll", "chunk_tail")
# `chunk_tail` (an integer q >= 0): a chunk shorter than the chunk length whose length is a multiple of q runs as
# q-chunks (run.split_tails, FESOM_CHUNK_TAIL): one compiled length less where month ends leave 2q.
_GPU_STR = ("bgc_layout", "bgc_halo")        # string-valued switches, passed through as written
_GPU_STR_VALUES = {"bgc_layout": ("major", "vmap", "loop"), "bgc_halo": ("grouped",)}
# `bgc_halo`: "grouped" sends the BGC stack's halo exchanges by the grouped transport (fesom_jax.halo_grouped, GPU
# only), the rest of the step's by `halo_transport`; unset: all by `halo_transport`.
_GPU_HOST = ("lane_order", "halo_transport", "chunk_steps")   # applied by the driver itself, not environment switches
# `halo_transport`: "all_gather" (default; `halo_compact` compacts it), "padded", "coloured", "grouped" (the coloured
# rounds in one NCCL group, GPU only: fesom_jax.halo_grouped) or "ragged" (the --padded / --coloured / --ragged flags,
# which win). `chunk_steps` (an integer > 0): the chunk length (--chunk-steps wins; 240 when neither is given).
_HALO_TRANSPORTS = ("all_gather", "padded", "coloured", "grouped", "ragged")


@dataclasses.dataclass(frozen=True)
class RunConfig:
    # --- physics sub-configs (None ⇒ that physics OFF / the bit-identical baseline) ---
    ale: Optional[AleConfig] = None        # None = linfs; AleConfig() = zstar
    gm: Optional[GMConfig] = None          # None = GM off
    kpp: Optional[KppConfig] = None        # None = PP mixing; KppConfig() = KPP
    tke: Optional[TkeConfig] = None        # None = no TKE (mutually exclusive with kpp)
    ice: Optional[IceConfig] = None        # None = no sea ice; IceConfig() = mEVP
    recom: Optional[RecomConfig] = None    # None = no REcoM biogeochemistry (fesom_jax.recom)
    # --- promoted run-dependent scalars ---
    visc: ViscConfig = ViscConfig()        # the horizontal-viscosity γ's (NG5: gamma1=0.2)
    tracer: TracerConfig = TracerConfig()  # the tracer scheme selector (implemented = NG5)
    dt: float = DT_DEFAULT
    dt_ramp: Optional[DtRamp] = None
    # --- run orchestration (consumed by the A6 driver; not part of the step kernel) ---
    mesh: Optional[str] = None
    partition: Optional[str] = None
    forcing: Optional[dict] = None
    ssh: Optional[dict] = None             # SSH solver options (_SSH_KEYS); None = defaults
    gpu: Optional[dict] = None             # GPU layout/transport options (_GPU_ENV); None = off
    output_dir: Optional[str] = None
    snapshot_every: int = 0
    checkpoint_every: int = 0
    restart_in: Optional[str] = None
    restart_out: Optional[str] = None
    restart_archive_out: Optional[str] = None       # None = archival restarts off
    restart_archive_period: Optional[str] = None    # 'year' | 'month' | 'day'
    restart_archive_length: int = 1                 # fire every Nth `period` boundary
    n_steps: Optional[int] = None
    duration: Optional[str] = None
    params_file: Optional[str] = None   # .npz of trained gm_nn weights; None ⇒ default GM
    # working precision: "double" (default, every program unchanged) or "single" — FESOM2's single-precision build
    # (#1207 policy, fesom_jax.precision; CORE2 production physics only)
    precision: str = "double"

    # ----------------------------------------------------------------------
    @classmethod
    def defaults(cls) -> "RunConfig":
        """Today's ``config.py`` baseline — all physics off, default γ's/dt ⇒ a step built from
        this is **bit-identical** to a bare :func:`fesom_jax.step.step` (the regression guard)."""
        return cls()

    def forcing_paths(self) -> dict:
        """The input-path overrides from the ``forcing:`` block, as kwargs for
        :func:`fesom_jax.surface_forcing.build_surface_forcing` (absent key ⇒ ``None`` ⇒ the
        reader resolves it via ``$FESOM_*`` / the Levante default, :mod:`fesom_jax.paths`)."""
        f = self.forcing or {}
        return {k: f.get(k) for k in _FORCING_PATH_KEYS}

    def bgc_dust_path(self):
        """The Albani dust file for REcoM's per-step inputs (:class:`fesom_jax.surface_forcing.BgcForcing`), or None
        unless full REcoM is on: ``forcing.dust_file``, else the namelist's ``nm_fe_data_file`` if it is an absolute
        path. Asserts the ported dust switches (Albani dust, no aeolian N)."""
        if self.recom is None or self.recom.mode != "full":
            return None
        sw, _ = self.recom.load()
        if sw.fe_data_source != "Albani" or sw.useAeolianN:
            raise NotImplementedError("REcoM's per-step inputs are ported for fe_data_source 'Albani' and "
                                      "useAeolianN off only")
        from pathlib import Path
        path = (self.forcing or {}).get("dust_file") or sw.nm_fe_data_file
        if not path or not Path(path).is_absolute():
            raise ValueError(f"full REcoM needs the Albani dust file: set forcing.dust_file (the namelist's "
                             f"nm_fe_data_file {sw.nm_fe_data_file!r} is relative to FESOM2's ClimateDataPath)")
        return str(path)

    def bgc_co2(self):
        """REcoM's atmospheric CO2 time series (:class:`fesom_jax.surface_forcing.AtmCO2Series`) when full REcoM runs
        with ``constant_CO2`` off, else None (``CO2_for_spinup``): the file ``forcing.co2_file``, else the namelist's
        ``nm_co2_data_file`` if it is an absolute path; the cycle years from the namelist."""
        if self.recom is None or self.recom.mode != "full":
            return None
        sw, _ = self.recom.load()
        if sw.constant_CO2:
            return None
        from pathlib import Path
        from .surface_forcing import AtmCO2Series
        path = (self.forcing or {}).get("co2_file") or sw.nm_co2_data_file
        if not path or not Path(path).is_absolute():
            raise ValueError(f"constant_CO2 off needs the CO2 time series: set forcing.co2_file (the namelist's "
                             f"nm_co2_data_file {sw.nm_co2_data_file!r} is relative to FESOM2's ClimateDataPath)")
        return AtmCO2Series(path, first_year=sw.firstyearoffesomcycle, last_year=sw.lastyearoffesomcycle,
                            n_cycles=sw.numofCO2cycles, cycle=sw.currentCO2cycle)

    def forcing_mslp(self) -> bool:
        """``forcing.mslp`` (FESOM2's ``l_mslp``): the sea-level pressure read and its momentum term on.
        False when absent or the run is unforced."""
        return bool((self.forcing or {}).get("mslp", False))

    def load_params(self):
        """The differentiable :class:`~fesom_jax.params.Params` this run's ``params_file``
        describes, or ``None`` when there is no file.

        ``None`` rather than ``Params.defaults()`` on purpose: the runners already treat
        ``params=None`` as the defaults, so a config without ``params_file`` takes the exact
        pre-existing code path and stays byte-identical. Called once per leg by
        :func:`fesom_jax.run.run_from_config`, so a restart chain picks the weights up again on
        every resubmit."""
        if self.params_file is None:
            return None
        from .gm_nn import load_gm_nn
        from .params import Params
        return dataclasses.replace(Params.defaults(), gm_nn=load_gm_nn(self.params_file))

    def ssh_cheb(self) -> tuple:
        """The CGPOLY knobs from the ``ssh:`` block as ``(degree, kappa_guess)``;
        ``degree == 0`` ⇒ the lever is OFF (the byte-identical MITgcm default)."""
        s = self.ssh or {}
        return int(s.get("cheb_degree", 0) or 0), float(s.get("cheb_kappa", 30.0))

    def ssh_gathered(self) -> bool:
        """``ssh.gathered``: the SSH solve gathered onto every device (off by default)."""
        return bool((self.ssh or {}).get("gathered", False))

    def split_explicit(self):
        """``ssh.split_explicit`` as a :class:`~fesom_jax.ssh_se.SeConfig` (``{}`` = its defaults), ``None`` when
        absent or null (the CG)."""
        se = (self.ssh or {}).get("split_explicit")
        if se is None:
            return None
        from .ssh_se import SeConfig
        if not isinstance(se, dict):
            raise TypeError(f"ssh.split_explicit must be a mapping or null, got {type(se).__name__}")
        _check_keys(se, SeConfig._fields, "ssh.split_explicit")
        return SeConfig(**se)

    def ale_cfg(self):
        """The step's :class:`~fesom_jax.ale.AleConfig`: :attr:`ale` with ``split_explicit`` from the ``ssh:`` block
        (``ale`` itself when the CG runs, so a CG config passes the identical object)."""
        se = self.split_explicit()
        if se is None or self.ale is None:
            return self.ale
        # one object per (ale, se) value: the sharded runner keys its executable cache on id(ale_cfg)
        return _ALE_SE.setdefault((self.ale, se), self.ale._replace(split_explicit=se))

    def gpu_env(self) -> dict:
        """The environment switches the ``gpu:`` block turns on, ``{name: "1"}`` (empty when off).
        :func:`fesom_jax.run.run_from_config` sets them for the run; an explicit environment
        setting wins, so ``FESOM_HALO_COMPACT=0`` still A/B-tests a config that turns it on."""
        return {_GPU_ENV[k]: (str(int(v)) if k in _GPU_INT else str(v) if k in _GPU_STR else "1")
                for k, v in (self.gpu or {}).items() if v and k not in _GPU_HOST}

    def lane_order(self):
        """``gpu.lane_order``: the order of the lanes inside the shards (:mod:`fesom_jax.lane_order`), None = the
        partition's own (global-id) order."""
        return (self.gpu or {}).get("lane_order") or None

    def halo_transport(self) -> str:
        """``gpu.halo_transport``: the ocean halo exchange (``"all_gather"`` when absent)."""
        return (self.gpu or {}).get("halo_transport") or "all_gather"

    def chunk_steps(self):
        """``gpu.chunk_steps``: the chunk length in steps, None when absent."""
        return (self.gpu or {}).get("chunk_steps") or None

    def validate(self) -> None:
        """Cross-field validity (mirrors the in-kernel guards, fail-fast at config load)."""
        from .precision import PRECISIONS
        if isinstance(self.forcing, dict) and not isinstance(self.forcing.get("mslp", False), bool):
            raise TypeError(f"forcing.mslp must be true or false, got {self.forcing['mslp']!r}")
        # REcoM's CO2 and O2 solubility read press_air, which is zero without l_mslp, and a zero air pressure
        # outgasses CO2 and O2 without limit (recom_main.F90:326-336). The passive-tracer transport mode
        # (recom.passive_tracers > 0) runs no chemistry and keeps the physics of the run as it is.
        if (self.recom is not None and getattr(self.recom, "passive_tracers", 0) <= 0
                and not self.forcing_mslp()):
            raise ValueError("recom needs forcing.mslp: true (FESOM2's l_mslp): REcoM's CO2 and O2 solubility read "
                             "the sea-level pressure, which is zero without it, and a zero air pressure outgasses "
                             "CO2 and O2 without limit (recom_main.F90:326-336). l_mslp also adds the SLP gradient "
                             "to the momentum equation, as in FESOM2-REcoM (bin_2p3z2d)")
        if self.recom is not None:
            self.recom.validate()
            if self.recom.mode == "full":
                if self.ale is None:
                    raise NotImplementedError("full REcoM is ported for zstar (ale: {}) only")
                # single precision + full REcoM: an opt-in deviation (FESOM2 refuses USE_SINGLE_PRECISION with
                # RECOM_COUPLED), the #1207 tracer scheme for the BGC stack, REcoM in float64; bgc_every 1 only
                # (plan 20261026-recom-lane-sp). The production-physics guard below applies as for every SP run.
                if self.precision != "double" and int(self.recom.bgc_every) != 1:
                    raise NotImplementedError("precision: single with full REcoM needs recom.bgc_every: 1 (the "
                                              "BGC-every-k sums and k-step stage have no single-precision scheme)")
                if self.forcing is None:
                    raise ValueError("full REcoM needs the forced path (forcing: with mslp: true)")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        if self.precision == "single":
            bad = []
            if self.ale is None or self.ale.use_wsplit:
                bad.append("zstar without w_split")
            if self.tke is None or self.kpp is not None:
                bad.append("classical TKE")
            if self.ice is None or self.ice.whichEVP != 1:
                bad.append("mEVP sea ice")
            if self.forcing is None:
                bad.append("forcing")
            if bad:
                raise NotImplementedError("precision: single supports the CORE2 production physics only; "
                                          "missing: " + ", ".join(bad))
        if self.forcing is not None:
            if not isinstance(self.forcing, dict):
                raise TypeError(f"forcing must be a mapping or null, "
                                f"got {type(self.forcing).__name__}")
            _check_keys(self.forcing, _FORCING_KEYS, "forcing")
        if self.ssh is not None:
            if not isinstance(self.ssh, dict):
                raise TypeError(f"ssh must be a mapping or null, got {type(self.ssh).__name__}")
            _check_keys(self.ssh, _SSH_KEYS, "ssh")
            if int(self.ssh.get("cheb_degree", 0) or 0) < 0:
                raise ValueError(f"ssh.cheb_degree must be >= 0, got {self.ssh['cheb_degree']}")
            se = self.split_explicit()
            if se is not None:
                if self.ale is None:
                    raise ValueError("ssh.split_explicit needs zstar (ale: {}): the subcycling's thickness law is z*, "
                                     "as the Kokkos port's FESOM_SSH_MODE=se requires FESOM_ALE=zstar")
                if self.ssh_gathered() or int(self.ssh.get("cheb_degree", 0) or 0):
                    raise ValueError("ssh.split_explicit replaces the CG: ssh.gathered / ssh.cheb_degree do not apply")
                se.validate()
        for _k in ("se_wide", "se_unroll"):
            if (self.gpu or {}).get(_k) and self.split_explicit() is None:
                raise ValueError(f"gpu.{_k} needs the split-explicit SSH (ssh.split_explicit)")
        if self.gpu is not None:
            if not isinstance(self.gpu, dict):
                raise TypeError(f"gpu must be a mapping or null, got {type(self.gpu).__name__}")
            _check_keys(self.gpu, tuple(_GPU_ENV) + _GPU_HOST, "gpu")
            if self.gpu.get("lane_order"):
                from .lane_order import METHODS
                if self.gpu["lane_order"] not in METHODS:
                    raise ValueError(f"gpu.lane_order must be one of {METHODS} or null, got {self.gpu['lane_order']!r}")
            if self.halo_transport() not in _HALO_TRANSPORTS:
                raise ValueError(f"gpu.halo_transport must be one of {_HALO_TRANSPORTS}, got {self.halo_transport()!r}")
            if self.halo_transport() != "all_gather" and self.gpu.get("halo_compact"):
                raise ValueError("gpu.halo_compact compacts the all_gather transport; it cannot be combined with "
                                 f"gpu.halo_transport {self.halo_transport()!r}")
            cs = self.gpu.get("chunk_steps")
            if cs is not None and (isinstance(cs, bool) or not isinstance(cs, int) or cs <= 0):
                raise TypeError(f"gpu.chunk_steps must be an integer > 0, got {cs!r}")
            for k, v in self.gpu.items():
                if k in _GPU_HOST:
                    continue
                if k in _GPU_INT:
                    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                        raise TypeError(f"gpu.{k} must be an integer >= 0, got {v!r}")
                elif k in _GPU_STR:
                    if v not in _GPU_STR_VALUES[k]:
                        raise ValueError(f"gpu.{k} must be one of {_GPU_STR_VALUES[k]}, got {v!r}")
                elif not isinstance(v, bool):
                    raise TypeError(f"gpu.{k} must be true or false, got {v!r}")
        if self.ice is not None:
            n = self.ice.ice_ave_steps
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError(f"ice.ice_ave_steps must be an integer >= 1, got {n!r}")
        if self.kpp is not None and self.tke is not None:
            raise ValueError(
                "kpp and tke are both set — the model runs exactly one vertical-mixing scheme "
                "(KPP xor classical-TKE); set one of them to null.")
        if self.ale is not None:
            self.ale.validate()
        if self.tke is not None:
            self.tke.validate()
        self.tracer.validate()
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.dt_ramp is not None and self.dt_ramp.dt <= 0:
            raise ValueError(f"dt_ramp.dt must be > 0, got {self.dt_ramp.dt}")
        if (self.restart_archive_out is not None
                and self.restart_archive_period not in _ARCHIVE_PERIODS):
            raise ValueError(f"restart_archive_period must be one of {_ARCHIVE_PERIODS}, "
                             f"got {self.restart_archive_period!r}")
        if self.restart_archive_length < 1:
            raise ValueError(f"restart_archive_length must be >= 1, got {self.restart_archive_length}")

    # -- the seam to the step kernel ---------------------------------------
    def physics_kwargs(self) -> dict:
        """The per-step physics config kwargs for :func:`fesom_jax.step.step` /
        :func:`fesom_jax.integrate.integrate` / ``run_steps_sharded`` (everything but ``dt``,
        which the driver ramps via :meth:`dt_at`)."""
        kw = dict(ice_cfg=self.ice, gm_cfg=self.gm, kpp_cfg=self.kpp, tke_cfg=self.tke,
                  ale_cfg=self.ale_cfg(), visc_cfg=self.visc, tracer_cfg=self.tracer)
        if self.recom is not None:      # only when on: a runner without REcoM support then refuses it loudly
            kw["recom_cfg"] = self.recom
        return kw

    def dt_at(self, step: int) -> float:
        """The timestep at absolute ``step`` index, honoring :attr:`dt_ramp`."""
        if self.dt_ramp is not None and step >= self.dt_ramp.after_step:
            return self.dt_ramp.dt
        return float(self.dt)

    def is_ramp_step(self, step: int) -> bool:
        """Whether ``step`` is exactly the dt-ramp boundary (⇒ the driver re-bootstraps AB2)."""
        return self.dt_ramp is not None and step == self.dt_ramp.after_step

    # -- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict:
        """A minimal plain-dict view: physics slots as ``None`` (off) or their NON-default
        overrides, the γ/tracer overrides, and the spec scalars. Round-trips through
        :meth:`from_dict` to an identical :class:`RunConfig`."""
        d: dict = {}
        for name, cls in _SUBCONFIGS.items():
            sub = getattr(self, name)
            d[name] = None if sub is None else _overrides(sub, cls())
        d["visc"] = _overrides(self.visc, ViscConfig())
        d["tracer"] = _overrides(self.tracer, TracerConfig())
        d["dt_ramp"] = None if self.dt_ramp is None else dict(self.dt_ramp._asdict())
        for f in _SPEC_FIELDS:
            d[f] = getattr(self, f)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        """Reconstruct from a (YAML-parsed) dict; raises ``KeyError`` on any unknown key."""
        known = set(_SUBCONFIGS) | {"visc", "tracer", "dt_ramp"} | set(_SPEC_FIELDS)
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown RunConfig key(s): {sorted(unknown)} "
                           f"(known: {sorted(known)})")
        kw: dict = {}
        for name, sub_cls in _SUBCONFIGS.items():
            kw[name] = _build_sub(d.get(name), sub_cls, name)
        if kw.get("recom") is not None and isinstance(kw["recom"].output, list):   # YAML list -> hashable tuple
            kw["recom"] = kw["recom"]._replace(output=tuple(kw["recom"].output))
        if d.get("visc") is not None:
            kw["visc"] = _build_sub(d["visc"], ViscConfig, "visc")
        if d.get("tracer") is not None:
            kw["tracer"] = _build_sub(d["tracer"], TracerConfig, "tracer")
        dr = d.get("dt_ramp")
        if dr is not None:
            _check_keys(dr, DtRamp._fields, "dt_ramp")
            kw["dt_ramp"] = DtRamp(**dr)
        for f in _SPEC_FIELDS:
            if f in d:
                kw[f] = d[f]
        fc = kw.get("forcing")
        if fc is not None:                      # strict keys inside `forcing:` too (no silent typos)
            if not isinstance(fc, dict):
                raise TypeError(f"forcing must be a mapping or null, got {type(fc).__name__}")
            _check_keys(fc, _FORCING_KEYS, "forcing")
        return cls(**kw)

    def to_yaml(self, path=None) -> str:
        """Serialize to YAML (writing to ``path`` if given). Reads back via :func:`load_yaml`."""
        import yaml
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        if path is not None:
            from pathlib import Path
            Path(path).write_text(text)
        return text


def _overrides(sub, default) -> dict:
    """The NamedTuple fields of ``sub`` that DIFFER from ``default`` (so a defaults sub-config
    serializes to ``{}`` = 'on with defaults')."""
    return {k: v for k, v in sub._asdict().items() if v != getattr(default, k)}


def _check_keys(d: dict, fields, where: str) -> None:
    unknown = set(d) - set(fields)
    if unknown:
        raise KeyError(f"unknown {where} key(s): {sorted(unknown)} (known: {sorted(fields)})")


def _build_sub(val, cls, where: str):
    """``None`` ⇒ None (physics off); a dict ⇒ ``cls(**overrides)`` with key validation."""
    if val is None:
        return None
    if not isinstance(val, dict):
        raise TypeError(f"{where} must be a mapping or null, got {type(val).__name__}")
    _check_keys(val, cls._fields, where)
    return cls(**val)


def load_yaml(path) -> RunConfig:
    """Load a :class:`RunConfig` from a single YAML file (the model paper's run config).

    Unknown top-level or sub-config keys raise (no silent typos). The YAML maps to FESOM
    ``namelist.oce`` concepts (``gamma1`` ↔ ``visc_gamma1``, ``dt`` ↔ ``step_per_day``, the
    ``nml_tracer_list`` scheme) but is a clean, composable format — not a namelist mimic."""
    import yaml
    from pathlib import Path
    d = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(d, dict):
        raise TypeError(f"{path}: top-level YAML must be a mapping, got {type(d).__name__}")
    cfg = RunConfig.from_dict(d)
    cfg.validate()
    return cfg


def to_yaml(cfg: RunConfig, path=None) -> str:
    """Module-level alias for :meth:`RunConfig.to_yaml`."""
    return cfg.to_yaml(path)
