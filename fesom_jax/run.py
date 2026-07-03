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
import math
import os
import time
from typing import NamedTuple

import jax
import numpy as np

from . import shard_mesh, ssh, ushow_output, zarr_output
from .integrate_sharded import run_steps_sharded_forced
from .run_config import RunConfig

_SECONDS = {"s": 1.0, "h": 3600.0, "d": 86400.0, "mo": 30 * 86400.0, "yr": 365 * 86400.0}

# Reuse the compiled per-chunk executable across chunks (avoid the ~25 s/chunk XLA recompile the perf
# decomposition exposed: a 96 ms/step all-on CORE2 step ran the hindcast at 520 ms/step). OPT-IN via
# FESOM_REUSE_EXE so the default working-tree behavior is byte-identical to before (the bit-identity
# guard); enabled once the GPU bit-identity test (reuse == fresh-compile) is green.
_REUSE_EXE = bool(os.environ.get("FESOM_REUSE_EXE"))


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
                dt: float, split_at=None) -> list:
    """Split ``[start_step, start_step+n_steps)`` into ``Chunk``s of ≤ ``chunk_steps``, also
    splitting at the **dt-ramp boundary** so no chunk straddles a dt change, and at each step in
    ``split_at`` (the **forcing-year boundaries**, :func:`_year_boundaries`) so no chunk straddles
    a forcing-year switch.

    ``bootstrap_ab2`` is True only for (a) the very first step of a COLD run (``start_step==0``,
    first chunk) and (b) the chunk that begins exactly at ``dt_ramp.after_step`` (the dt changed ⇒
    the AB2 history is stale). A restart-continuation chunk (``start_step>0``, same dt) carries AB2
    forward ⇒ a chained run is bit-identical to a continuous one (the restart-seam invariant). A
    ``split_at`` (year) boundary is forcing-only — same dt, AB2 continuous ⇒ it does NOT bootstrap."""
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be > 0")
    end = start_step + n_steps
    ramp_at = dt_ramp.after_step if dt_ramp is not None else None
    splits = sorted(x for x in (split_at or ()) if start_step < x < end)
    chunks, s = [], start_step
    while s < end:
        cdt = dt_ramp.dt if (ramp_at is not None and s >= ramp_at) else dt
        stop = min(end, s + chunk_steps)
        if ramp_at is not None and s < ramp_at < stop:    # don't straddle the dt change
            stop = ramp_at
        for sp in splits:                                 # don't straddle a forcing-year boundary
            if s < sp < stop:
                stop = sp
                break
        boot = (s == 0) or (s == ramp_at)
        chunks.append(Chunk(start=s, count=stop - s, dt=float(cdt), bootstrap_ab2=boot))
        s = stop
    return chunks


def _elapsed_seconds(n: int, dt0: float, dt_ramp=None) -> float:
    """Model elapsed time AT absolute step ``n`` = ``n·dt0``, but PIECEWISE if a dt-ramp fires at
    ``dt_ramp.after_step`` (steps ≥ after_step are taken with ``dt_ramp.dt``). Mirrors the per-step
    model clock so the forcing CALENDAR stays exact across a mid-run dt change (e.g. 180→240 after
    year 1) — a single ``n·dt`` map would mis-date every step past the ramp."""
    if dt_ramp is None or n <= dt_ramp.after_step:
        return n * float(dt0)
    R = dt_ramp.after_step
    return R * float(dt0) + (n - R) * float(dt_ramp.dt)


def _step_at_elapsed(T: float, dt0: float, dt_ramp=None) -> int:
    """Inverse of :func:`_elapsed_seconds`: the smallest absolute step ``n`` with elapsed ≥ ``T``."""
    if dt_ramp is None or T <= dt_ramp.after_step * float(dt0):
        return int(math.ceil(T / float(dt0)))
    R = dt_ramp.after_step
    return R + int(math.ceil((T - R * float(dt0)) / float(dt_ramp.dt)))


def _chunk_dates(year: int, dt: float, start_step: int, count: int, dt_ramp=None):
    """Model ``(year, doy, sec, month)`` tuples for the ``count`` steps starting at absolute
    ``start_step`` — the chunk-local slice of
    :func:`fesom_jax.core2_forcing.dates_for_steps`, computed without materializing the whole
    multi-year date list (NG5 has ~175 k steps/yr). ``dt`` is the BASE timestep; ``dt_ramp`` (if
    set) makes the elapsed-time clock PIECEWISE (:func:`_elapsed_seconds`) so the calendar is exact
    across a mid-run dt change. ``dt_ramp=None`` ⇒ the plain ``n·dt`` map (bit-identical to before)."""
    base = datetime.datetime(int(year), 1, 1)
    out = []
    for n in range(start_step, start_step + count):
        d = base + datetime.timedelta(seconds=_elapsed_seconds(n, dt, dt_ramp))
        doy = (d - datetime.datetime(d.year, 1, 1)).days + 1
        out.append((d.year, doy, d.hour * 3600.0 + d.minute * 60.0 + d.second, d.month))
    return out


def _year_boundaries(year: int, dt: float, start_step: int, n_steps: int, dt_ramp=None):
    """Absolute step indices in ``(start_step, start_step+n_steps)`` where the model CALENDAR YEAR
    changes — passed to :func:`plan_chunks` as ``split_at`` so no chunk straddles a forcing-year
    switch (the JRA reader holds one year; :meth:`fesom_jax.jra55.JRA55Reader.reopen_year` rolls it
    AT a boundary).

    Ramp-aware: inverts :func:`_elapsed_seconds` via :func:`_step_at_elapsed`, so with a dt-ramp the
    1959 boundary is still at the dt0 step but 1960+ land at their dt1 steps (exact incl. leap years).
    ``dt_ramp=None`` ⇒ the plain ``ceil(secs/dt)`` map (unchanged)."""
    base = datetime.datetime(int(year), 1, 1)
    end = int(start_step) + int(n_steps)
    bounds = []
    y = (base + datetime.timedelta(seconds=_elapsed_seconds(int(start_step), dt, dt_ramp))).year
    while True:
        secs = (datetime.datetime(y + 1, 1, 1) - base).total_seconds()
        nb = _step_at_elapsed(secs, dt, dt_ramp)
        if nb >= end:
            break
        if nb > int(start_step):
            bounds.append(nb)
        y += 1
    return bounds


def _period_boundaries(year: int, dt: float, start_step: int, n_steps: int,
                       *, period: str, dt_ramp=None):
    """Absolute step indices in ``(start_step, start_step+n_steps)`` where the calendar DAY
    (``period='day'``) or MONTH (``period='month'``) rolls over. **dt-INDEPENDENT** (date-based via
    :func:`_step_at_elapsed`) — splitting chunks here keeps each chunk within ONE output period, so the
    per-step output accumulation yields a TRUE per-period time-mean regardless of steps-per-day (the
    diurnal-cycle-aliasing fix works at any mesh/dt). Mirrors :func:`_year_boundaries`."""
    base = datetime.datetime(int(year), 1, 1)
    end = int(start_step) + int(n_steps)
    d0 = base + datetime.timedelta(seconds=_elapsed_seconds(int(start_step), dt, dt_ramp))
    if period == "day":
        cur = datetime.datetime(d0.year, d0.month, d0.day)
        nxt = lambda c: c + datetime.timedelta(days=1)                       # noqa: E731
    elif period == "month":
        cur = datetime.datetime(d0.year, d0.month, 1)
        nxt = lambda c: datetime.datetime(c.year + (c.month == 12),          # noqa: E731
                                          1 if c.month == 12 else c.month + 1, 1)
    else:
        raise ValueError(f"period must be 'day' or 'month', got {period!r}")
    bounds = []
    while True:
        cur = nxt(cur)
        nb = _step_at_elapsed((cur - base).total_seconds(), dt, dt_ramp)
        if nb >= end:
            break
        if nb > int(start_step):
            bounds.append(nb)
    return bounds


def _archive_tag(yr: int, doy: int, sec: float) -> str:
    """``fesom.<YYYY>.<DDD>.<SSSSS>`` — mirrors FESOM3's archival-checkpoint folder name
    (``mod_io_restart.F90``) exactly: year, day-of-year, second-of-day, all zero-padded, so
    lexical sort == chronological sort. A calendar boundary (year/month start) always has
    ``sec=0``; only a mid-period "always checkpoint the final step" write can have a nonzero one."""
    return f"fesom.{int(yr):04d}.{int(doy):03d}.{int(round(sec)):05d}"


def _archive_boundaries(year: int, dt: float, start_step: int, n_steps: int,
                        *, period: str, length: int = 1, dt_ramp=None):
    """Absolute step indices in ``(start_step, start_step+n_steps)`` where an ARCHIVAL restart
    should fire: every ``length``-th ``period`` ('year'/'month'/'day') boundary. ``length=1``
    (the common case) fires at every one. Reuses :func:`_year_boundaries`/:func:`_period_boundaries`
    for the raw boundary set (so it's exactly consistent with the existing forcing-reopen /
    output-period splits) and filters by an ABSOLUTE elapsed-period count (year number, or
    year*12+month) so the cadence is stable across chain-job segments regardless of where a given
    segment happens to start — not merely "every Nth boundary within this call"."""
    if period == "year":
        raw = _year_boundaries(year, dt, start_step, n_steps, dt_ramp)
    elif period in ("month", "day"):
        raw = _period_boundaries(year, dt, start_step, n_steps, period=period, dt_ramp=dt_ramp)
    else:
        raise ValueError(f"restart_archive_period must be 'year'/'month'/'day', got {period!r}")
    length = int(length)
    if length <= 1:
        return list(raw)
    out = []
    for s in raw:
        yr, _doy, _sec, mo = _chunk_dates(year, dt, s, 1, dt_ramp)[0]
        idx = yr if period == "year" else (yr * 12 + mo)
        if idx % length == 0:
            out.append(s)
    return out


class _MeanStream:
    """A per-calendar-period TRUE time-mean output stream — the diurnal-safe replacement for the old
    once-per-chunk snapshot. The chunk scan sums this stream's fields over EVERY step and returns the
    sum; :meth:`add` accumulates the sum per period key with the step count, and :meth:`flush` writes
    ``sum/count`` — a real time-mean at any dt (no fixed-time-of-day aliasing of the diurnal cycle).
    ``keys`` are the accumulated-dict keys this stream owns; ``key_fn(step)`` gives the period key;
    ``write(key, mean_dict, count)`` emits it. Chunks must not straddle this period's boundary (the
    driver adds :func:`_period_boundaries` to ``plan_chunks`` ``split_at``), so a whole chunk's sum
    belongs to one period."""

    def __init__(self, keys, key_fn, write, start_step):
        self.keys = tuple(keys)
        self.key_fn = key_fn
        self.write = write
        self.start_step = int(start_step)
        self.sum = None
        self.count = 0
        self.pkey = None

    def add(self, gate_step, key_step, acc_sum, n):
        # gate_step (the chunk-final step) decides activation; key_step (the chunk's FIRST step) picks
        # the period — a chunk split at a period boundary ends AT the next period's first step, so
        # keying by the final step would misattribute the whole chunk one period late. The chunk never
        # straddles a boundary, so key_step's period IS the chunk's period.
        if gate_step <= self.start_step:
            return
        key = self.key_fn(key_step)
        if self.sum is not None and key != self.pkey:
            self.flush()
        sub = {k: acc_sum[k] for k in self.keys}
        self.sum = sub if self.sum is None else jax.tree.map(lambda a, b: a + b, self.sum, sub)
        self.count += int(n)
        self.pkey = key

    def flush(self):
        if self.sum is not None and self.count > 0:
            c = float(self.count)
            self.write(self.pkey, {k: v / c for k, v in self.sum.items()}, self.count)
        self.sum = None
        self.count = 0


class RunResult(NamedTuple):
    state_p: object     # FOLDED [P*Lmax] final State (device-sharded; restart-ready)
    stats: object       # OnlineStats over chunk-final folded states, or None
    step: int           # absolute step index after the run
    dt_stage: float     # the dt the run ended at (persisted for resume)


def run_from_config(cfg: RunConfig, *, mesh, part, sm=None, sop=None, forcing=None,
                    state0=None, forcing_stack=None, start_step=0, year=1958,
                    chunk_steps=None, devices=None, out_dir=None, use_ragged=False,
                    accumulate_stats=False, stats_fields=("T", "S", "uv"), progress=False,
                    local_forcing=None, checkpoint_every=None,
                    restart_archive_out=None, restart_archive_period=None,
                    restart_archive_length=1,
                    daily_out=None, daily_start_step=0,
                    monthly_out=None, monthly_start_step=0, chunk_diagnostics=False,
                    output_layout="global"):
    """Run a configured forward integration: load → chunked forced steps → write restart.

    Components may be **injected** (the test path / a pre-building driver) or built from ``cfg``.
    ``state0`` (a GLOBAL host :class:`~fesom_jax.state.State`) seeds a cold start; else
    ``cfg.restart_in`` is read (device-count-portable) and ``start_step`` taken from its metadata.
    ``forcing_stack`` (a ``StepForcing`` stacked over the FULL run) overrides the calendar forcing
    (the test path); production builds each chunk's forcing from ``forcing`` (a ``CoreForcing``) via
    the calendar. ``accumulate_stats`` folds each chunk-final State into an
    :class:`~fesom_jax.zarr_output.OnlineStats` (a coarse, gather-free time mean). Returns a
    :class:`RunResult`; writes the portable restart to ``out_dir`` / ``cfg.restart_out`` if set.

    ``restart_archive_out`` (if set) additionally writes ARCHIVAL restarts: an immutable,
    uniquely-named directory (``_archive_tag`` — mirrors FESOM3's ``fesom.<YYYY>.<DDD>.<SSSSS>``)
    at every ``restart_archive_length``-th ``restart_archive_period`` ('year'/'month'/'day')
    calendar boundary, PLUS unconditionally at the end of this run (mirrors FESOM3's "the last
    step is always checkpointed regardless of cadence") — so a clean chain resubmit never loses
    anything; only a mid-segment crash falls back to the last calendar boundary. Nothing is ever
    overwritten or auto-deleted (resume any of them directly via ``restart_in``); a
    ``restart.latest`` pointer (:func:`zarr_output.write_restart_latest`) always names the
    newest one. This is a SEPARATE, additive stream — the plain rolling ``checkpoint_every``
    restart (if also set) is unaffected."""
    cfg.validate()
    npes = part.npes
    # Output / restart on-disk layout. 'global' (default) = partition-INDEPENDENT canonical node
    # order (xarray/ushow read it directly; byte-identical at any device count) — it host-gathers
    # owned lanes on ONE process. 'folded' = the gather-free sharded [P*Lmax] write. A multi-node
    # (multi-process) run CANNOT host-gather ⇒ MUST use 'folded'. HINT: if output/restart OOMs or
    # you scale to NG5 multi-node, set output_layout='folded' (CLI: --output-layout folded).
    if output_layout not in ("global", "folded"):
        raise ValueError(f"output_layout must be 'global' or 'folded', got {output_layout!r}")
    # 'global' = partition-independent canonical OUTPUT *and* RESTART (host_gather single-process,
    # all_gather multi-process — both byte-identical at any device count); 'folded' = the gather-free
    # sharded write (fastest; for max throughput / huge-mesh OOM). The restart follows the same layout
    # (its canonical writer is multi-process now); a 'folded' restart still reloads onto ANY device
    # count. HINT: set output_layout='folded' if the all_gather replicated field OOMs on a huge mesh.
    multiproc = jax.process_count() > 1
    restart_layout = output_layout
    if jax.process_index() == 0:
        _om = (("host_gather" if not multiproc else "all_gather")
               if output_layout == "global" else "folded")
        print(f"[run] output_layout={output_layout} (output+restart → {_om})", flush=True)
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

    # --- initial State: cold = HOST [P,Lmax]; resume = FOLDED [P*Lmax] device (read_restart) ---
    # The State is kept FOLDED through the chunk loop + restart (no [P,Lmax]↔[P*Lmax] reshape, which
    # MATERIALIZES the global array on one device under multi-node ⇒ OOM). Cold start is the one
    # [P,Lmax] (host) entry; run_steps_sharded_forced folds it once and returns folded thereafter.
    if state0 is not None:
        state_p = shard_mesh.partition_state(state0, part)   # [P, Lmax] HOST
        folded_in = False
    elif cfg.restart_in is not None:
        state_p, meta = zarr_output.read_restart(cfg.restart_in, mesh, part, devices=devices)
        folded_in = True                                     # read_restart returns folded [P*Lmax]
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
    # split chunks at forcing-year boundaries AND (when a mean-output stream is active) at the output
    # period boundary — DAY for daily output (finest; months align to days), else MONTH — so no chunk
    # straddles a period and its every-step sum belongs to exactly one period. dt-independent, so the
    # true-time-mean output (the diurnal-aliasing fix) works at any mesh/dt. See :class:`_MeanStream`.
    _out_period = "day" if daily_out is not None else ("month" if monthly_out is not None else None)
    _split = list(_year_boundaries(year, cfg.dt, start_step, n_steps, cfg.dt_ramp))
    if _out_period is not None:
        _split += _period_boundaries(year, cfg.dt, start_step, n_steps,
                                     period=_out_period, dt_ramp=cfg.dt_ramp)
    _archive_targets = set()
    if restart_archive_out is not None and restart_archive_period is not None:
        _archive_targets = set(_archive_boundaries(
            year, cfg.dt, start_step, n_steps, period=restart_archive_period,
            length=restart_archive_length, dt_ramp=cfg.dt_ramp))
        _split += list(_archive_targets)
    chunks = plan_chunks(n_steps, cs, start_step=start_step, dt_ramp=cfg.dt_ramp, dt=cfg.dt,
                         split_at=sorted(set(_split)))

    if forcing is None and forcing_stack is None:
        raise ValueError("run_from_config: provide forcing (CoreForcing) or forcing_stack")
    fstatic = forcing.static
    fs_p = shard_mesh.partition_forcing_static(fstatic, part)
    stress_p = np.zeros((npes, sm.Lmax["elem"], 2))        # host (Phase-8b B.3; not on GPU 0)

    # mEVP sea-ice boundary condition (the EVP zeros u_ice/v_ice at PHYSICAL-coast nodes,
    # fesom_ice_evp.c:430-437). MUST be the GLOBAL coastal mask, partitioned in — else the sharded
    # fallback computes it from each device's LOCAL mesh, wrongly treating PARTITION CUTS as coasts
    # (artificial walls between partitions). Only ice runs need it. Mirrors bench_forward_scaling.
    boundary_node_p = None
    if cfg.ice is not None:
        from . import ice_evp
        bn = np.asarray(ice_evp.boundary_node_mask(mesh))   # GLOBAL [nod2D] bool (device→host, ~MB)
        boundary_node_p = shard_mesh._shard_along_axis(
            bn, part.myList_nod2D, sm.Lmax["nod"], 0, False)

    # --- chunk loop -------------------------------------------------------
    # `progress` (diagnostic, default off ⇒ no behavior change): flushed per-chunk timing split
    # into HOST forcing build (stack+partition) vs DEVICE steps (block_until_ready forces a sync so
    # the device time is real — it removes the natural host/device overlap, so this is for measuring
    # WHERE the wall-clock goes, not a production fast-path). Lets a long NG5 run be diagnosed live.
    stats = None
    if progress:
        print(f"[run.progress] setup done @ {time.perf_counter():.0f}s; {len(chunks)} chunks "
              f"x ~{chunks[0].count if chunks else 0} steps, n_steps={n_steps} npes={npes}",
              flush=True)
    # restart/checkpoint target + a small writer (used for the rolling checkpoint AND the final
    # restart). Intermediate checkpoints make a multi-hour run crash-safe (resume from the last
    # one), give the segment-chaining hand-off, and are a gather-free sharded write (~one restart
    # cost) so a coarse cadence (checkpoint_every steps) is <<1% overhead.
    target = out_dir if out_dir is not None else cfg.restart_out

    def _write_restart_at(sp, step, dtv):
        yr, doy, _, _ = _chunk_dates(year, cfg.dt, step, 1, cfg.dt_ramp)[0]
        zarr_output.write_restart(target, sp, sm, part, step=step,
                                  calendar_date=f"{yr}-doy{doy:03d}", dt_stage=dtv,
                                  layout=restart_layout)

    # ARCHIVAL restart: a SEPARATE, immutable, never-overwritten directory per firing (calendar
    # boundary or end-of-run) — see the docstring. `restart_archive_out=None` ⇒ fully inert (no
    # extra writes, no extra directory created).
    def _write_archive_restart(sp, step, dtv):
        yr, doy, sec, _ = _chunk_dates(year, cfg.dt, step, 1, cfg.dt_ramp)[0]
        tag = _archive_tag(yr, doy, sec)
        zarr_output.write_restart(f"{restart_archive_out}/{tag}", sp, sm, part, step=step,
                                  calendar_date=f"{yr}-doy{doy:03d}", dt_stage=dtv,
                                  layout=restart_layout)
        zarr_output.write_restart_latest(restart_archive_out, tag)
        return tag

    # daily-/monthly-mean output (gather-free). Each active stream is a TRUE per-calendar-period time-
    # mean: the chunk scan sums the stream's fields over EVERY step and the driver divides by the step
    # count (:class:`_MeanStream` below) — dt-INDEPENDENT, and it AVERAGES the diurnal cycle (fixing the
    # once-per-chunk-snapshot artifact that, at day-aligned chunks, froze a fixed time-of-day and aliased
    # the diurnal SST cycle into a wavenumber-1 pattern). Matches Fortran's every-timestep mean.
    # output writer dispatch (daily/monthly): 'global' (default) = partition-INDEPENDENT canonical
    # node order (write_global_zarr; xarray/ushow read it directly, byte-identical at any device
    # count); 'folded' = the gather-free sharded [P*Lmax] write (write_ushow_sharded) — the multi-
    # node / output-OOM path. Both take the same (path, {name: folded field}, sm, part, mesh).
    def _emit_ushow(path, fields_dict, attrs, time_days=None):
        if output_layout == "folded":
            ushow_output.write_ushow_sharded(path, fields_dict, sm, part, mesh, attrs=attrs,
                                             time_days=time_days)
        else:
            ushow_output.write_global_zarr(path, fields_dict, sm, part, mesh, attrs=attrs,
                                           time_days=time_days)

    # TRUE time-mean output streams (daily / monthly). Each names the fields to average; the chunk scan
    # sums them over EVERY step (out_sample_fn below) and the mean divides by the step count — a real
    # time-mean that AVERAGES the diurnal cycle at ANY dt. This replaces the old once-per-chunk snapshot,
    # which at day-aligned chunks froze a fixed time-of-day and aliased the diurnal cycle into a
    # wavenumber-1 SST pattern (JAX−Fortran meanstate artifact). Matches Fortran's every-timestep mean.
    # Chunks are split at the finest active period boundary (below) so a whole chunk's sum is one period.
    # (Accumulators do NOT persist across chain-job processes ⇒ a period straddling a job boundary keeps
    # only its post-boundary steps — ~1 period per multi-year segment, negligible for a climatology.)
    streams = []
    out_sample_specs = []
    k100 = None
    if daily_out is not None:
        k100 = int(np.argmin(np.abs(np.asarray(mesh.Z) - (-100.0))))
        k500 = int(np.argmin(np.abs(np.asarray(mesh.Z) - (-500.0))))

        def _daily_fields(s, _k100=k100, _k500=k500):
            return {"sst": s.T[:, 0], "sss": s.S[:, 0], "ssh": s.eta_n,
                    "temp100": s.T[:, _k100],
                    "usurf": s.uvnode[:, 0, 0], "vsurf": s.uvnode[:, 0, 1],
                    "u100": s.uvnode[:, _k100, 0], "v100": s.uvnode[:, _k100, 1],
                    "u500": s.uvnode[:, _k500, 0], "v500": s.uvnode[:, _k500, 1],
                    "a_ice": s.a_ice, "m_ice": s.m_ice}

        def _write_daily(day, mean, count, _k100=k100, _k500=k500):
            yr, doy = day
            date = datetime.date(yr, 1, 1) + datetime.timedelta(days=doy - 1)
            _emit_ushow(f"{daily_out}/day_{yr:04d}_{doy:03d}", mean,
                        {"calendar_date": f"{yr:04d}-doy{doy:03d}", "n_samples": int(count),
                         "depth_100m_level": int(_k100), "depth_500m_level": int(_k500)},
                        time_days=ushow_output.cf_days(date))

        def _daily_key(step):
            yr, doy, _, _ = _chunk_dates(year, cfg.dt, step, 1, cfg.dt_ramp)[0]
            return (int(yr), int(doy))

        streams.append(_MeanStream(("sst", "sss", "ssh", "temp100", "usurf", "vsurf",
                                    "u100", "v100", "u500", "v500", "a_ice", "m_ice"),
                                   _daily_key, _write_daily, daily_start_step))
        out_sample_specs.append(_daily_fields)

    if monthly_out is not None:
        def _monthly_fields(s):
            return {"temp": s.T, "salt": s.S, "u": s.uvnode[:, :, 0], "v": s.uvnode[:, :, 1],
                    "ssh": s.eta_n, "a_ice": s.a_ice, "m_ice": s.m_ice}

        def _write_monthly(key, mean, count):
            yr, mo = key
            date = datetime.date(yr, mo, 15)   # mid-month stamp (matches paper_jax's mid-month convention)
            _emit_ushow(f"{monthly_out}/{yr:04d}_{mo:02d}", mean,
                        {"calendar_month": f"{yr:04d}-{mo:02d}", "n_samples": int(count)},
                        time_days=ushow_output.cf_days(date))

        def _monthly_key(step):
            yr, _, _, mo = _chunk_dates(year, cfg.dt, step, 1, cfg.dt_ramp)[0]
            return (int(yr), int(mo))

        streams.append(_MeanStream(("temp", "salt", "u", "v", "ssh", "a_ice", "m_ice"),
                                   _monthly_key, _write_monthly, monthly_start_step))
        out_sample_specs.append(_monthly_fields)

    # The ONE sample_fn threaded into the chunk scan = union of active streams' fields, summed each step.
    # Built ONCE (stable id ⇒ the reuse-executable cache stays warm). None ⇒ no output ⇒ the scan returns
    # just the State (byte-identical). Split period = DAY if daily output is active (finest; months align
    # to day boundaries) else MONTH — dt-independent, so each chunk lands in exactly one output period.
    if out_sample_specs:
        def out_sample_fn(s, _specs=tuple(out_sample_specs)):
            d = {}
            for f in _specs:
                d.update(f(s))
            return d
    else:
        out_sample_fn = None

    t_loop0 = time.perf_counter()
    cur_year = int(year)                     # the year the forcing reader was built for
    for ci, ch in enumerate(chunks):
        tc0 = time.perf_counter()
        # Roll the forcing reader to this chunk's CALENDAR YEAR if it changed (calendar paths only;
        # chunks never straddle a year boundary — plan_chunks split_at=_year_boundaries). reopen_year
        # swaps only the per-year file handles and KEEPS the interpolation stencil (year-independent).
        if forcing_stack is None:
            cy = _chunk_dates(year, cfg.dt, ch.start, 1, cfg.dt_ramp)[0][0]
            if cy != cur_year:
                (local_forcing if local_forcing is not None else forcing).reopen_year(cy)
                cur_year = cy
                if progress:
                    print(f"[run.progress] forcing → year {cy} @ step {ch.start}", flush=True)
        if forcing_stack is not None:
            lo = ch.start - start_step
            seq = jax.tree.map(lambda x, lo=lo: x[lo: lo + ch.count], forcing_stack)
            seq_p = shard_mesh.partition_step_forcing(seq, part)
        elif local_forcing is not None:
            # LOCAL forcing build (the NG5 host-forcing fix): interpolate ONLY this process's
            # local-partition nodes (~npes× less) and scatter into [P, n_steps, Lmax] — bit-
            # identical to the global build's local shards (the non-local rows are never read).
            seq_p = local_forcing.stack_partitioned(
                _chunk_dates(year, cfg.dt, ch.start, ch.count, cfg.dt_ramp), xp=np)
        else:
            # HOST-build the per-chunk forcing (xp=np): a global [n_steps, nod2D] stack is
            # ~2.65 GB/field at NG5 (7.4 M × 48 steps) — building it on GPU 0 OOMs before
            # partition_step_forcing can shard it (the forcing analog of the host-IC fix).
            seq = forcing.stack(_chunk_dates(year, cfg.dt, ch.start, ch.count, cfg.dt_ramp), xp=np)
            seq_p = shard_mesh.partition_step_forcing(seq, part)
        tc_host = time.perf_counter()
        _chunk_out = run_steps_sharded_forced(
            sm, state_p, _sop_for(ch.dt), stress_p, seq_p, fs_p, ch.count, dt=ch.dt, npes=npes,
            bootstrap_ab2=ch.bootstrap_ab2, state_is_folded=folded_in, return_folded=True,
            use_ragged=use_ragged, boundary_node_p=boundary_node_p, reuse_executable=_REUSE_EXE,
            sample_fn=out_sample_fn, **cfg.physics_kwargs())
        # out_sample_fn ⇒ (final_state, per-step field SUMS over this chunk); else just the final state.
        state_p, acc_sum = _chunk_out if out_sample_fn is not None else (_chunk_out, None)
        folded_in = True                                     # the scan output is folded [P*Lmax]
        step_now = ch.start + ch.count
        if progress:
            jax.block_until_ready(state_p)
            now = time.perf_counter()
            print(f"[run.progress] chunk {ci + 1}/{len(chunks)} step{ch.start}->{ch.start+ch.count} "
                  f"host={tc_host - tc0:.1f}s device={now - tc_host:.1f}s "
                  f"(loop elapsed {now - t_loop0:.0f}s)", flush=True)
        # per-chunk health TRAJECTORY (diagnostic, default off ⇒ no behavior change). Gather-free
        # scalar reductions on the FOLDED sharded State after each chunk ⇒ with small chunks this is
        # a max|uv|/max|eta| curve over the run, so a slow blow-up (e.g. the dars dt=180 2Δx velocity
        # growth) is visible STEP-BY-STEP — what the end-of-run --diagnostics can't show (it only
        # reports the final state, and a NaN blow-up never reaches it). Each reduction is a blocking
        # all-reduce; only worth it at small chunk sizes for a stability probe.
        if chunk_diagnostics:
            import jax.numpy as jnp
            from .diagnostics import state_diagnostics
            d = state_diagnostics(state_p)               # COLLECTIVE (all ranks must call) ...
            # per-level max|T| (gather-free: reduce the sharded node axis, keep the vertical axis ⇒
            # a replicated (nl,) vector) ⇒ WHICH level the tracer overshoot lives on (surface k≈0 ⇒
            # forcing/KPP-BL/advection; deep ⇒ convection or a zstar layer-thickness collapse). Plus
            # max|cfl_z| (vertical CFL) — a collapsing layer drives it up before T goes non-finite.
            Tlev = jnp.max(jnp.abs(jnp.nan_to_num(state_p.T, nan=0.0)), axis=0)   # (nl,)
            klev = int(jnp.argmax(Tlev)); Tk = float(Tlev[klev]); Tsurf = float(Tlev[0])
            cflz = float(jnp.max(jnp.abs(jnp.nan_to_num(state_p.cfl_z, nan=0.0))))
            if jax.process_index() == 0:                 # ... but only the lead prints
                print(f"[chunk-diag] step{step_now} max|uv|={d['max_abs_uv']:.4g} "
                      f"max|eta|={d['max_abs_eta']:.4g} T[{d['T_min']:.3g},{d['T_max']:.3g}] "
                      f"|T|max@k={klev}({Tk:.3g}) |T|surf={Tsurf:.3g} max_cfl_z={cflz:.3g} "
                      f"n_nonfinite={d['n_nonfinite']}", flush=True)
        if accumulate_stats:
            leaves = {k: getattr(state_p, k) for k in stats_fields}   # already folded [P*Lmax]
            stats = (zarr_output.OnlineStats.init(leaves) if stats is None
                     else stats).update(leaves)

        # TRUE time-mean accumulation: add this chunk's per-step field SUMS (acc_sum) and step count to
        # each active output stream, keyed by the chunk's calendar period. Chunks never straddle a period
        # boundary (plan_chunks split_at includes _period_boundaries), so the whole chunk belongs to one
        # period; the stream flushes sum/count at the rollover. Replaces the old once-per-chunk snapshot
        # (the diurnal-aliasing fix — averages the full diurnal cycle at any dt).
        if acc_sum is not None:
            for _stream in streams:
                _stream.add(step_now, ch.start, acc_sum, ch.count)

        # rolling intermediate checkpoint: when this chunk crossed a `checkpoint_every` step
        # boundary (and it's not the last chunk — the final restart covers that), overwrite the
        # restart so a crash resumes from here. start_step from the restart metadata.
        if (checkpoint_every and target is not None and ci < len(chunks) - 1
                and step_now // checkpoint_every > ch.start // checkpoint_every):
            if progress:
                jax.block_until_ready(state_p)
            tw = time.perf_counter()
            _write_restart_at(state_p, step_now, ch.dt)
            if progress:
                print(f"[run.progress] checkpoint @ step {step_now} -> {target} "
                      f"({time.perf_counter() - tw:.1f}s)", flush=True)

        # archival restart: fires on every configured calendar boundary crossed by THIS chunk
        # (never the last chunk — the unconditional end-of-run write below covers that exactly
        # once, avoiding a redundant double-write at the same step).
        if (restart_archive_out is not None and ci < len(chunks) - 1
                and step_now in _archive_targets):
            if progress:
                jax.block_until_ready(state_p)
            tw = time.perf_counter()
            tag = _write_archive_restart(state_p, step_now, ch.dt)
            if progress:
                print(f"[run.progress] archival restart @ step {step_now} -> "
                      f"{restart_archive_out}/{tag} ({time.perf_counter() - tw:.1f}s)", flush=True)

    end_step = start_step + n_steps
    end_dt = chunks[-1].dt if chunks else cfg.dt

    # --- write the final portable restart (state_p is already folded [P*Lmax]) ---
    if target is not None:
        if progress:
            jax.block_until_ready(state_p)
            print(f"[run.progress] chunk loop done @ {time.perf_counter() - t_loop0:.0f}s; "
                  f"writing restart -> {target}", flush=True)
        tw = time.perf_counter()
        _write_restart_at(state_p, end_step, end_dt)
        if progress:
            print(f"[run.progress] restart written in {time.perf_counter() - tw:.1f}s", flush=True)

    # unconditional end-of-run archival restart (mirrors FESOM3: "the last step is always
    # checkpointed regardless of cadence") — so a CLEAN chain resubmit never loses anything; only
    # an actual mid-segment crash falls back to the last calendar-boundary archival restart.
    if restart_archive_out is not None:
        if progress:
            jax.block_until_ready(state_p)
        tw = time.perf_counter()
        tag = _write_archive_restart(state_p, end_step, end_dt)
        if progress:
            print(f"[run.progress] final archival restart -> {restart_archive_out}/{tag} "
                  f"({time.perf_counter() - tw:.1f}s)", flush=True)

    # flush the final (partial) calendar period so the last day/month of the run/job is written too
    for _stream in streams:
        _stream.flush()
    return RunResult(state_p=state_p, stats=stats, step=end_step, dt_stage=float(end_dt))
