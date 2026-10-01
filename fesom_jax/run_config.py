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
    idemix: null            # IDEMIX internal-wave mixing; null = off, {} = on (defaults)

``idemix`` turns on the IDEMIX internal-wave closure (the idemix-jax integration; physics in
:mod:`fesom_jax.idemix`). ``null``/absent ⇒ off, byte-identical. A mapping ⇒ on; every key
optional::

    idemix:
      n_hor_iter: 5            # lateral sub-steps         } fesom_jax.idemix.IdemixConfig
      lateral: true            # horizontal propagation    }   (defaults when absent)
      lateral_cfl_cap: 0.1     # v0 CFL cap (0 = off)      }
      params: fesom_operational   # | oe13 | {tau_v, tau_h, gamma, jstar, mu0} (all five)
      surf_file: /pool/.../fourier_smooth_2005_cfsr_inert_rgrid.nc   # } idemix_jax.
      bot_file:  /pool/.../tidal_energy_gx1v6_20090205_rgrid.nc      # } load_idemix_forcing
      sforcusage: 0.2                                                # } (its defaults
      botforc_Etot: 0.0                                              # }  when absent)

``params`` defaults to ``fesom_operational`` (FESOM2's namelist.cvmix, Pollmann et al. 2017);
``oe13`` is the Olbers & Eden (2013) set (``IdemixParams()``). The run driver loads the forcing
onto the mesh nodes, builds the lateral operator and sets ``Params.idemix``. IDEMIX runs alone
(Osborn ``Kv``/``Av``) or coupled to TKE, which then needs ``tke: {only_tke: false}``; never with
``kpp``. Inconsistent combinations raise at config load (:meth:`RunConfig.validate`).

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
from .tke import TkeConfig


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
    "tke": TkeConfig, "ice": IceConfig,
}
# Non-physics scalar/spec fields that pass through YAML verbatim.
_SPEC_FIELDS = ("dt", "mesh", "partition", "forcing", "ssh", "output_dir", "snapshot_every",
                "checkpoint_every", "restart_in", "restart_out",
                "restart_archive_out", "restart_archive_period", "restart_archive_length",
                "n_steps", "duration", "params_file")
_ARCHIVE_PERIODS = ("year", "month", "day")
# The `forcing:` mapping — WHAT the forcing is (`kind`/`start_year`) plus WHERE its input
# files live. The four path keys are optional overrides of the env var / Levante default
# (:mod:`fesom_jax.paths`); absent ⇒ the reader resolves them itself. Strict-keyed like every
# other block: an unknown key raises rather than being silently ignored.
_FORCING_KEYS = ("kind", "start_year", "jra_dir", "sss_path", "runoff_path", "chl_path",
                 "on_device")
_FORCING_PATH_KEYS = ("jra_dir", "sss_path", "runoff_path", "chl_path")
# The `ssh:` mapping — SSH CG solver options. `cheb_degree` (int ≥1) enables the degree-k
# Chebyshev polynomial preconditioner (CGPOLY, ssh.enable_cheb_precond) in place of the
# MITgcm M⁻¹ — a many-node comm lever (fewer CG iterations ⇒ fewer psums); absent/0 ⇒ the
# byte-identical default. `cheb_kappa` tunes the assumed condition number (default 30).
_SSH_KEYS = ("cheb_degree", "cheb_kappa")
# The `idemix:` mapping (see the module docstring). Kept as plain names here so importing
# run_config never imports idemix_jax; the config/params objects are built lazily.
_IDEMIX_CFG_KEYS = ("n_hor_iter", "lateral", "lateral_cfl_cap")      # IdemixConfig._fields
_IDEMIX_FORCING_KEYS = ("surf_file", "bot_file", "sforcusage", "botforc_Etot")
_IDEMIX_KEYS = _IDEMIX_CFG_KEYS + ("params",) + _IDEMIX_FORCING_KEYS
_IDEMIX_PARAM_PRESETS = ("fesom_operational", "oe13")
_IDEMIX_PARAM_FIELDS = ("tau_v", "tau_h", "gamma", "jstar", "mu0")  # IdemixParams._fields


@dataclasses.dataclass(frozen=True)
class RunConfig:
    # --- physics sub-configs (None ⇒ that physics OFF / the bit-identical baseline) ---
    ale: Optional[AleConfig] = None        # None = linfs; AleConfig() = zstar
    gm: Optional[GMConfig] = None          # None = GM off
    kpp: Optional[KppConfig] = None        # None = PP mixing; KppConfig() = KPP
    tke: Optional[TkeConfig] = None        # None = no TKE (mutually exclusive with kpp)
    ice: Optional[IceConfig] = None        # None = no sea ice; IceConfig() = mEVP
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
    idemix: Optional[dict] = None       # the `idemix:` mapping; None ⇒ IDEMIX off

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

    def idemix_config(self):
        """The static :class:`~fesom_jax.idemix.IdemixConfig` of the ``idemix:`` block, or
        ``None`` when IDEMIX is off.

        Built ONCE per ``RunConfig`` and returned as the SAME object on every call: the forced
        runner's executable cache (``integrate_sharded._FORCED_JIT_CACHE``) keys on the
        IDENTITY of each physics config, relying on every one being a single persistent object
        per run (the sub-config fields are; this derived one is memoized to match). A fresh
        object per :meth:`physics_kwargs` call would recompile every chunk and grow the cache."""
        if self.idemix is None:
            return None
        cached = self.__dict__.get("_idemix_cfg")
        if cached is None:
            from .idemix import IdemixConfig
            cached = IdemixConfig(**{k: self.idemix[k] for k in _IDEMIX_CFG_KEYS
                                     if k in self.idemix})
            object.__setattr__(self, "_idemix_cfg", cached)   # frozen dataclass: memo only
        return cached

    def idemix_params(self):
        """The :class:`idemix_jax.IdemixParams` named by ``idemix.params`` (default
        ``fesom_operational``), or ``None`` when IDEMIX is off."""
        if self.idemix is None:
            return None
        from idemix_jax import IdemixParams
        spec = self.idemix.get("params", "fesom_operational")
        if spec == "fesom_operational":
            return IdemixParams.fesom_operational()
        if spec == "oe13":
            return IdemixParams()
        if isinstance(spec, dict):
            if set(spec) != set(_IDEMIX_PARAM_FIELDS):
                raise KeyError(f"idemix.params mapping needs exactly {list(_IDEMIX_PARAM_FIELDS)}, "
                               f"got {sorted(spec)}")
            return IdemixParams(**{k: float(spec[k]) for k in _IDEMIX_PARAM_FIELDS})
        raise ValueError(f"idemix.params must be one of {list(_IDEMIX_PARAM_PRESETS)} or a "
                         f"mapping of {list(_IDEMIX_PARAM_FIELDS)}, got {spec!r}")

    def idemix_inputs(self, mesh):
        """``(IdemixForcing, LateralOperator | None)`` for ``mesh`` — the GLOBAL per-node
        energy fluxes from ``idemix_jax.load_idemix_forcing`` on ``geo_coord_nod2D`` (degrees,
        with the block's ``surf_file``/``bot_file``/``sforcusage``/``botforc_Etot``; the
        ``botforc_Etot`` normalization area is each node's top-wet-layer ``areasvol``) and the
        lateral operator (``None`` when ``lateral: false``). Host NumPy, built once per run."""
        if self.idemix is None:
            return None, None
        import numpy as np
        from idemix_jax import load_idemix_forcing
        from .idemix import build_lateral_operator
        kw = {k: self.idemix[k] for k in _IDEMIX_FORCING_KEYS if k in self.idemix}
        if kw.get("botforc_Etot", 0.0) != 0.0:
            top = np.asarray(mesh.ulevels_nod2D) - 1
            kw["node_area"] = np.take_along_axis(np.asarray(mesh.areasvol), top[:, None], 1)[:, 0]
        geo = np.degrees(np.asarray(mesh.geo_coord_nod2D, dtype=np.float64))
        forcing = load_idemix_forcing(geo[:, 0], geo[:, 1], **kw)
        lateral = build_lateral_operator(mesh) if self.idemix_config().lateral else None
        return forcing, lateral

    def ssh_cheb(self) -> tuple:
        """The CGPOLY knobs from the ``ssh:`` block as ``(degree, kappa_guess)``;
        ``degree == 0`` ⇒ the lever is OFF (the byte-identical MITgcm default)."""
        s = self.ssh or {}
        return int(s.get("cheb_degree", 0) or 0), float(s.get("cheb_kappa", 30.0))

    def validate(self) -> None:
        """Cross-field validity (mirrors the in-kernel guards, fail-fast at config load)."""
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
        if self.kpp is not None and self.tke is not None:
            raise ValueError(
                "kpp and tke are both set — the model runs exactly one vertical-mixing scheme "
                "(KPP xor classical-TKE); set one of them to null.")
        if self.idemix is not None:
            if not isinstance(self.idemix, dict):
                raise TypeError(f"idemix must be a mapping or null, "
                                f"got {type(self.idemix).__name__}")
            _check_keys(self.idemix, _IDEMIX_KEYS, "idemix")
            if self.kpp is not None:
                raise ValueError(
                    "idemix and kpp are both set: IDEMIX couples to TKE or runs alone (Osborn "
                    "shortcut), never with KPP; set kpp to null.")
            self.idemix_config().validate()
            self.idemix_params()
        if self.tke is not None and (not self.tke.only_tke) != (self.idemix is not None):
            if self.idemix is None:
                raise ValueError(
                    "tke.only_tke: false requires an idemix: block (the TKE then consumes the "
                    "IDEMIX wave dissipation); set only_tke: true or add idemix.")
            raise ValueError(
                "idemix with tke requires tke.only_tke: false (FESOM's coupled "
                "cvmix_TKE+cvmix_IDEMIX); or drop tke for IDEMIX-only (Osborn) mixing.")
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
        which the driver ramps via :meth:`dt_at`). With an ``idemix:`` block it also carries
        ``idemix_cfg`` (the data — forcing, lateral operator, ``Params.idemix`` — comes from
        :meth:`idemix_inputs` / :meth:`idemix_params`); without one the dict is unchanged."""
        kw = dict(ice_cfg=self.ice, gm_cfg=self.gm, kpp_cfg=self.kpp, tke_cfg=self.tke,
                  ale_cfg=self.ale, visc_cfg=self.visc, tracer_cfg=self.tracer)
        if self.idemix is not None:
            kw["idemix_cfg"] = self.idemix_config()
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
        if self.idemix is not None:             # absent when off ⇒ pre-IDEMIX yaml unchanged
            d["idemix"] = dict(self.idemix)
        for f in _SPEC_FIELDS:
            d[f] = getattr(self, f)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        """Reconstruct from a (YAML-parsed) dict; raises ``KeyError`` on any unknown key."""
        known = set(_SUBCONFIGS) | {"visc", "tracer", "dt_ramp", "idemix"} | set(_SPEC_FIELDS)
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown RunConfig key(s): {sorted(unknown)} "
                           f"(known: {sorted(known)})")
        kw: dict = {}
        for name, sub_cls in _SUBCONFIGS.items():
            kw[name] = _build_sub(d.get(name), sub_cls, name)
        if d.get("visc") is not None:
            kw["visc"] = _build_sub(d["visc"], ViscConfig, "visc")
        if d.get("tracer") is not None:
            kw["tracer"] = _build_sub(d["tracer"], TracerConfig, "tracer")
        dr = d.get("dt_ramp")
        if dr is not None:
            _check_keys(dr, DtRamp._fields, "dt_ramp")
            kw["dt_ramp"] = DtRamp(**dr)
        iw = d.get("idemix")
        if iw is not None:                      # strict keys inside `idemix:` (no silent typos)
            if not isinstance(iw, dict):
                raise TypeError(f"idemix must be a mapping or null, got {type(iw).__name__}")
            _check_keys(iw, _IDEMIX_KEYS, "idemix")
            kw["idemix"] = dict(iw)
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
