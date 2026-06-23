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
import time
from typing import NamedTuple

import jax
import numpy as np

from . import shard_mesh, ssh, ushow_output, zarr_output
from .integrate_sharded import run_steps_sharded_forced
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
                    daily_out=None, daily_start_step=0,
                    monthly_out=None, monthly_start_step=0, chunk_diagnostics=False):
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
    chunks = plan_chunks(n_steps, cs, start_step=start_step, dt_ramp=cfg.dt_ramp, dt=cfg.dt,
                         split_at=_year_boundaries(year, cfg.dt, start_step, n_steps, cfg.dt_ramp))

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
                                  calendar_date=f"{yr}-doy{doy:03d}", dt_stage=dtv)

    # daily-mean output (requested SST/SSS/T@100m/u@100m/v@100m), gather-free, active for absolute
    # steps > daily_start_step (set to the year-1 end so daily output begins in year 2). The chunk-
    # final surface/100m slices are folded into a per-calendar-day OnlineStats (~10 samples/day at
    # dt=180, ~7.5 at dt=240); each day's mean is written as one ushow-readable zarr at the rollover.
    # A day straddling a chain-job boundary is the post-boundary samples only (~1% of days) — the
    # daily_stats does not persist across the separate per-job processes.
    daily_stats = daily_day = k100 = None
    if daily_out is not None:
        k100 = int(np.argmin(np.abs(np.asarray(mesh.Z) - (-100.0))))

        def _write_daily(day, st):
            yr, doy = day
            ushow_output.write_ushow_sharded(
                f"{daily_out}/day_{yr:04d}_{doy:03d}", st.mean_dict(), sm, part, mesh,
                attrs={"calendar_date": f"{yr:04d}-doy{doy:03d}", "n_samples": st.nobs(),
                       "depth_100m_level": int(k100)})

    # monthly-mean output (the CORE2 1958-2020 hindcast climatology): FULL 3-D temp/salt/u/v + 2-D
    # ssh/a_ice/m_ice, accumulated per CALENDAR MONTH into a gather-free OnlineStats (chunk-final
    # samples; ~1/day at chunk_steps=48,dt=1800 ⇒ ~30/month ⇒ a real monthly mean), written as one
    # ushow-readable sharded zarr per month at the rollover. Active for steps > monthly_start_step.
    # Like daily_out, the accumulator does NOT persist across chain-job processes ⇒ a month straddling
    # a job boundary keeps only its post-boundary samples (~1 month per multi-year segment ⇒ negligible
    # for a multi-decadal climatology). Parallels the daily_out path (kept separate, not merged: only
    # one cadence runs per mesh — CORE2 monthly, FORCA20 daily — so duplication beats coupling here).
    monthly_stats = monthly_key = None
    if monthly_out is not None:
        def _write_monthly(key, st):
            yr, mo = key
            ushow_output.write_ushow_sharded(
                f"{monthly_out}/{yr:04d}_{mo:02d}", st.mean_dict(), sm, part, mesh,
                attrs={"calendar_month": f"{yr:04d}-{mo:02d}", "n_samples": st.nobs()})

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
        state_p = run_steps_sharded_forced(
            sm, state_p, _sop_for(ch.dt), stress_p, seq_p, fs_p, ch.count, dt=ch.dt, npes=npes,
            bootstrap_ab2=ch.bootstrap_ab2, state_is_folded=folded_in, return_folded=True,
            use_ragged=use_ragged, boundary_node_p=boundary_node_p, **cfg.physics_kwargs())
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

        if daily_out is not None and step_now > daily_start_step:
            yr, doy, _, _ = _chunk_dates(year, cfg.dt, step_now, 1, cfg.dt_ramp)[0]
            day = (int(yr), int(doy))
            samp = {"sst": state_p.T[:, 0], "sss": state_p.S[:, 0], "temp100": state_p.T[:, k100],
                    "u100": state_p.uvnode[:, k100, 0], "v100": state_p.uvnode[:, k100, 1]}
            if daily_stats is not None and day != daily_day:
                _write_daily(daily_day, daily_stats)         # the previous calendar day completed
                daily_stats = None
            daily_stats = (zarr_output.OnlineStats.init(samp) if daily_stats is None
                           else daily_stats.update(samp))
            daily_day = day

        if monthly_out is not None and step_now > monthly_start_step:
            yr, _, _, mo = _chunk_dates(year, cfg.dt, step_now, 1, cfg.dt_ramp)[0]
            key = (int(yr), int(mo))
            samp = {"temp": state_p.T, "salt": state_p.S,
                    "u": state_p.uvnode[:, :, 0], "v": state_p.uvnode[:, :, 1],
                    "ssh": state_p.eta_n, "a_ice": state_p.a_ice, "m_ice": state_p.m_ice}
            if monthly_stats is not None and key != monthly_key:
                _write_monthly(monthly_key, monthly_stats)   # the previous calendar month completed
                monthly_stats = None
            # init().update() folds the FIRST sample too (init alone is count=0/zeros) — so a
            # single-chunk partial month at a job boundary writes its real value, not zeros.
            monthly_stats = (zarr_output.OnlineStats.init(samp).update(samp) if monthly_stats is None
                             else monthly_stats.update(samp))
            monthly_key = key

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

    # flush the final (partial) calendar day so the last day of the run/job is written too
    if daily_out is not None and daily_stats is not None:
        _write_daily(daily_day, daily_stats)
    # flush the final (partial) calendar month so the last month of the run/job is written too
    if monthly_out is not None and monthly_stats is not None:
        _write_monthly(monthly_key, monthly_stats)
    return RunResult(state_p=state_p, stats=stats, step=end_step, dt_stage=float(end_dt))
