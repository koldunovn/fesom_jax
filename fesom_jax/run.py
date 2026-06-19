"""Single-invocation, config-driven run entry point (Task A6).

One YAML → one invocation: **load restart (or cold PHC IC) → run N steps (or a duration) → write
restart + output → exit.** No in-process orchestration — multi-job campaigns are a *series* of
these invocations chained by SLURM ``--dependency=afterok`` (``scripts/chain_submit.sh``), each
picking up the previous restart. This is the driver the model-paper forward runs (CORE2 / farc /
dars / NG5) launch.

Three pieces, each independently testable:
  * :func:`parse_duration` — ``"2yr"`` / ``"3mo"`` / ``"5d"`` / ``"10step"`` / an int → a step count;
  * :func:`plan_chunks` — split a run into **fine (≈ few-day) time-chunks** (the A4 forcing-memory
    contract: a whole NG5 year can't be pre-stacked) AND across a **dt-ramp boundary**, tagging each
    chunk with the timestep and whether it bootstraps AB2 (cold / post-ramp) or continues it;
  * :func:`run_from_config` — the orchestration: per chunk, build that chunk's per-step forcing, run
    it through the sharded forced scan (:func:`~fesom_jax.integrate_sharded.run_steps_sharded_forced`),
    continue the State across chunks, and write the portable restart at the end.

**dt-ramp = a dt CHANGE across a restart boundary.** The AB2 history (``uv_rhsAB``/``T_old``) was
formed at the OLD dt, so the first step after the ramp re-bootstraps AB2 (``is_first_step=True`` ⇒
``bootstrap_ab2``); ``dt_stage`` is persisted so a resumed job knows the current dt.

**Forward-only at scale** (single-GPU for adjoints — the sharded ragged-halo AD bug). The chunked
host round-trip of the State between chunks is fine at CORE2/farc/dars scale; for NG5 the on-device
state chaining (``return_executable`` + keep the folded shards resident) is the B2 optimization —
noted, not yet wired (de-risk on the smaller meshes first per the run plan).
"""
from __future__ import annotations

import datetime
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import halo, shard_mesh, ssh, zarr_output
from .integrate_sharded import (_fold, _to_global_sharded, folded_state,
                                run_steps_sharded_forced, unfold_state)
from .run_config import RunConfig

_SECONDS = {"s": 1.0, "h": 3600.0, "d": 86400.0, "mo": 30 * 86400.0, "yr": 365 * 86400.0}


# --------------------------------------------------------------------------
# Duration parsing
# --------------------------------------------------------------------------
def parse_duration(spec, dt: float) -> int:
    """A run length → an integer step count. ``spec`` is an int (steps), or a string
    ``"<n><unit>"`` with unit ``step`` / ``s`` / ``h`` / ``d`` / ``mo`` / ``yr`` (a 365-day year,
    30-day month — the cadence approximation, not the model calendar). ``"5step"`` ⇒ 5; ``"1d"`` at
    ``dt=1800`` ⇒ 48; ``"2yr"`` ⇒ ``round(2·365·86400/dt)``."""
    if isinstance(spec, (int, np.integer)):
        return int(spec)
    s = str(spec).strip().lower()
    if s.endswith("step"):
        return int(s[:-4])
    for unit in ("yr", "mo", "h", "d", "s"):          # 2-char units before 1-char (mo before s)
        if s.endswith(unit):
            return int(round(float(s[: -len(unit)]) * _SECONDS[unit] / float(dt)))
    return int(s)                                     # bare number ⇒ steps


# --------------------------------------------------------------------------
# Chunk planning (time-chunking + dt-ramp split + AB2 bootstrap flags)
# --------------------------------------------------------------------------
class Chunk(NamedTuple):
    start: int           # absolute step index of this chunk's first step
    count: int           # number of steps in this chunk
    dt: float            # timestep for this chunk (honors the dt-ramp)
    bootstrap_ab2: bool  # True ⇒ this chunk's first step re-bootstraps AB2 (cold / post-ramp)


def plan_chunks(n_steps: int, chunk_steps: int, *, start_step: int = 0, dt_ramp=None,
                dt: float) -> list:
    """Split ``[start_step, start_step+n_steps)`` into ``Chunk``s of ≤ ``chunk_steps``, also
    splitting at the **dt-ramp boundary** so no chunk straddles a dt change.

    ``bootstrap_ab2`` is True only for (a) the very first step of a COLD run (``start_step==0``,
    first chunk) and (b) the chunk that begins exactly at ``dt_ramp.after_step`` (the dt changed ⇒
    the AB2 history is stale). A restart-continuation chunk (``start_step>0``, same dt) carries AB2
    forward ⇒ a chained run is bit-identical to a continuous one (the restart-seam invariant)."""
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be > 0")
    end = start_step + n_steps
    ramp_at = dt_ramp.after_step if dt_ramp is not None else None
    chunks, s = [], start_step
    while s < end:
        cdt = dt_ramp.dt if (ramp_at is not None and s >= ramp_at) else dt
        stop = min(end, s + chunk_steps)
        if ramp_at is not None and s < ramp_at < stop:    # don't straddle the dt change
            stop = ramp_at
        boot = (s == 0) or (s == ramp_at)
        chunks.append(Chunk(start=s, count=stop - s, dt=float(cdt), bootstrap_ab2=boot))
        s = stop
    return chunks


def _chunk_dates(year: int, dt: float, start_step: int, count: int):
    """Model ``(year, doy, sec, month)`` tuples for the ``count`` steps starting at absolute
    ``start_step`` — the chunk-local slice of
    :func:`fesom_jax.core2_forcing.dates_for_steps`, computed without materializing the whole
    multi-year date list (NG5 has ~175 k steps/yr)."""
    base = datetime.datetime(int(year), 1, 1)
    out = []
    for n in range(start_step, start_step + count):
        d = base + datetime.timedelta(seconds=n * float(dt))
        doy = (d - datetime.datetime(d.year, 1, 1)).days + 1
        out.append((d.year, doy, d.hour * 3600.0 + d.minute * 60.0 + d.second, d.month))
    return out


# --------------------------------------------------------------------------
# Restart-state plumbing ([P,Lmax] → folded sharded, for write_restart)
# --------------------------------------------------------------------------
def _place_folded(state_p, npes, devices=None):
    """``[P,Lmax]`` State → folded ``[P*Lmax]`` device-sharded State (the form
    :func:`~fesom_jax.zarr_output.write_restart` writes from)."""
    fs, spec = folded_state(state_p)
    devs = jax.devices()[:npes] if devices is None else list(devices)
    return _to_global_sharded(fs, spec, halo.device_mesh("p", devices=devs))


class RunResult(NamedTuple):
    state_p: object     # [P, Lmax] final State
    stats: object       # OnlineStats over chunk-final folded states, or None
    step: int           # absolute step index after the run
    dt_stage: float     # the dt the run ended at (persisted for resume)


def run_from_config(cfg: RunConfig, *, mesh, part, sm=None, sop=None, forcing=None,
                    state0=None, forcing_stack=None, start_step=0, year=1958,
                    chunk_steps=None, devices=None, out_dir=None,
                    accumulate_stats=False, stats_fields=("T", "S", "uv")):
    """Run a configured forward integration: load → chunked forced steps → write restart.

    Components may be **injected** (the test path / a pre-building driver) or built from ``cfg``.
    ``state0`` (a GLOBAL host :class:`~fesom_jax.state.State`) seeds a cold start; else
    ``cfg.restart_in`` is read (device-count-portable) and ``start_step`` taken from its metadata.
    ``forcing_stack`` (a ``StepForcing`` stacked over the FULL run) overrides the calendar forcing
    (the test path); production builds each chunk's forcing from ``forcing`` (a ``CoreForcing``) via
    the calendar. ``accumulate_stats`` folds each chunk-final State into an
    :class:`~fesom_jax.zarr_output.OnlineStats` (a coarse, gather-free time mean). Returns a
    :class:`RunResult`; writes the portable restart to ``out_dir`` / ``cfg.restart_out`` if set."""
    cfg.validate()
    npes = part.npes
    if sm is None:
        sm = shard_mesh.build_sharded_mesh(mesh, part)

    # The SSH operator is dt-DEPENDENT (the implicit-solve matrix), so a dt-ramp run needs a fresh
    # operator for the post-ramp dt — cache one per dt (the injected `sop` covers cfg.dt).
    _sop_cache = {}

    def _sop_for(d):
        if sop is not None and d == cfg.dt:
            return sop
        if d not in _sop_cache:
            _sop_cache[d] = ssh.partition_ssh_operator(ssh.build_ssh_operator(mesh, dt=d), part)
        return _sop_cache[d]

    # --- initial [P, Lmax] State ------------------------------------------
    if state0 is not None:
        state_p = shard_mesh.partition_state(state0, part)
    elif cfg.restart_in is not None:
        folded, meta = zarr_output.read_restart(cfg.restart_in, mesh, part, devices=devices)
        state_p = unfold_state(folded, npes)
        start_step = int(meta.get("step", start_step))
    else:
        raise ValueError("run_from_config: provide state0 (cold IC) or cfg.restart_in")

    # --- run length + chunk plan ------------------------------------------
    if cfg.n_steps is not None:
        n_steps = int(cfg.n_steps)
    elif cfg.duration is not None:
        n_steps = parse_duration(cfg.duration, cfg.dt)
    else:
        raise ValueError("run_from_config: set cfg.n_steps or cfg.duration")
    cs = int(chunk_steps) if chunk_steps else max(1, n_steps)
    chunks = plan_chunks(n_steps, cs, start_step=start_step, dt_ramp=cfg.dt_ramp, dt=cfg.dt)

    if forcing is None and forcing_stack is None:
        raise ValueError("run_from_config: provide forcing (CoreForcing) or forcing_stack")
    fstatic = forcing.static
    fs_p = shard_mesh.partition_forcing_static(fstatic, part)
    stress_p = jnp.zeros((npes, sm.Lmax["elem"], 2))

    # --- chunk loop -------------------------------------------------------
    stats = None
    for ch in chunks:
        if forcing_stack is not None:
            lo = ch.start - start_step
            seq = jax.tree.map(lambda x, lo=lo: x[lo: lo + ch.count], forcing_stack)
        else:
            seq = forcing.stack(_chunk_dates(year, ch.dt, ch.start, ch.count))
        seq_p = shard_mesh.partition_step_forcing(seq, part)
        state_p = run_steps_sharded_forced(
            sm, state_p, _sop_for(ch.dt), stress_p, seq_p, fs_p, ch.count, dt=ch.dt, npes=npes,
            bootstrap_ab2=ch.bootstrap_ab2, **cfg.physics_kwargs())
        if accumulate_stats:
            folded = {k: _fold(getattr(state_p, k)) for k in stats_fields}   # [P*Lmax] leaves
            stats = (zarr_output.OnlineStats.init(folded) if stats is None
                     else stats).update(folded)

    end_step = start_step + n_steps
    end_dt = chunks[-1].dt if chunks else cfg.dt

    # --- write the portable restart ---------------------------------------
    target = out_dir if out_dir is not None else cfg.restart_out
    if target is not None:
        yr, doy, _, _ = _chunk_dates(year, end_dt, end_step, 1)[0]
        zarr_output.write_restart(target, _place_folded(state_p, npes, devices=devices), sm, part,
                                  step=end_step, calendar_date=f"{yr}-doy{doy:03d}", dt_stage=end_dt)
    return RunResult(state_p=state_p, stats=stats, step=end_step, dt_stage=float(end_dt))
