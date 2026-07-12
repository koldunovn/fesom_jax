"""A6 gate: the single-invocation config-driven run driver (:mod:`fesom_jax.run`) — ``RUN_DRIVER_OK``.

The headline invariant is the **restart seam**: a run done as ONE invocation equals the SAME run done
as two chained invocations (run N/2 → write portable restart → read it back → run N/2), to the climate-
close floor — the proof that the A1 portable restart + the A4 ``bootstrap_ab2`` AB2-continuation carry
the State across a job boundary correctly (a continuation chunk must NOT cold-start AB2). Plus the pure
chunk-planning / duration-parsing logic and a single-step run.

  PY -m pytest fesom_jax/tests/test_run_entry.py -x
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import surface_forcing, partit, shard_mesh, ssh
from fesom_jax.mesh import load_mesh
from fesom_jax.run import (Chunk, _chunk_dates, _elapsed_seconds, _step_at_elapsed,
                           _year_boundaries, parse_duration, plan_chunks, run_from_config)
from fesom_jax.run_config import DtRamp, RunConfig
from fesom_jax.state import State
from fesom_jax.kpp import KppConfig

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 1800.0
YEAR = 1958


def _have_jra():
    from fesom_jax import jra55
    return Path(jra55.DEFAULT_JRA_DIR).is_dir()


have_forcing = pytest.mark.skipif(
    not (IC_DIR / "T_ic.npy").exists() or not CORE2_MESH.is_dir() or not _have_jra(),
    reason="CORE2 PHC IC / mesh / JRA55 forcing missing (compute node only)")


# ==========================================================================
# 1. Pure logic — duration parsing + chunk planning (always run, fast)
# ==========================================================================
def test_parse_duration():
    assert parse_duration(7, DT) == 7
    assert parse_duration("5step", DT) == 5
    assert parse_duration("1d", DT) == 48                 # 86400/1800
    assert parse_duration("3mo", DT) == round(3 * 30 * 86400 / DT)
    assert parse_duration("2yr", DT) == round(2 * 365 * 86400 / DT)
    assert parse_duration("180", 180.0) == 180            # bare number ⇒ steps


def test_plan_chunks_cold_and_continuation():
    # cold run: bootstrap AB2 only at the very first step
    ch = plan_chunks(10, 4, start_step=0, dt=180.0)
    assert [(c.start, c.count, c.bootstrap_ab2) for c in ch] == \
        [(0, 4, True), (4, 4, False), (8, 2, False)]
    # restart-continuation (start>0): the first chunk CONTINUES AB2 (no cold bootstrap)
    assert plan_chunks(4, 4, start_step=8, dt=180.0)[0].bootstrap_ab2 is False


def test_plan_chunks_dt_ramp():
    # the dt-ramp splits the chunking at the boundary; the post-ramp chunk uses the NEW dt and
    # re-bootstraps AB2 (the dt change invalidates the AB2 history formed at the old dt)
    ch = plan_chunks(8, 10, start_step=0, dt_ramp=DtRamp(after_step=4, dt=240.0), dt=180.0)
    assert ch == [Chunk(0, 4, 180.0, True), Chunk(4, 4, 240.0, True)]


def test_year_boundaries():
    # 3 model years from 1958 at dt=180: the reader must roll at the first step of 1959 and 1960
    n = parse_duration("3yr", 180.0)                      # 3*365 days = 525600 steps
    assert _year_boundaries(1958, 180.0, 0, n) == [175200, 350400]
    # a sub-year run has no boundary; a mid-year RESUME only reports boundaries ahead of start
    assert _year_boundaries(1958, 180.0, 0, 1000) == []
    assert _year_boundaries(1958, 180.0, 200000, 200000) == [350400]


def test_archive_tag():
    from fesom_jax.run import _archive_tag
    assert _archive_tag(1958, 1, 0) == "fesom.1958.001.00000"       # a clean calendar boundary
    assert _archive_tag(1958, 213, 43200) == "fesom.1958.213.43200"  # a mid-period (crash-fallback-like) stamp
    # lexical sort == chronological sort (the whole point of the zero-padded FESOM3 convention)
    tags = sorted([_archive_tag(1958, 1, 0), _archive_tag(1958, 213, 43200), _archive_tag(1959, 1, 0)])
    assert tags == [_archive_tag(1958, 1, 0), _archive_tag(1958, 213, 43200), _archive_tag(1959, 1, 0)]


def test_archive_boundaries_year_and_month_and_length():
    from fesom_jax.run import _archive_boundaries
    n = parse_duration("3yr", 180.0)
    # length=1 (the common case) == exactly the raw year boundaries
    assert _archive_boundaries(1958, 180.0, 0, n, period="year", length=1) == \
        _year_boundaries(1958, 180.0, 0, n)
    # length=2 keeps only every OTHER year boundary (1958 % 2 == 0 is the reference; 1959 is odd,
    # dropped; 1960 is even, kept) — filtered by the boundary's OWN absolute year, not call-local index
    assert _archive_boundaries(1958, 180.0, 0, n, period="year", length=2) == \
        [b for b in _year_boundaries(1958, 180.0, 0, n) if b == 350400]
    # month period fires 12x more often than year within the same window
    assert len(_archive_boundaries(1958, 180.0, 0, n, period="month", length=1)) == 35  # 36 months - 1 (start)
    # an invalid period raises rather than silently doing nothing
    with pytest.raises(ValueError):
        _archive_boundaries(1958, 180.0, 0, n, period="fortnight", length=1)


def test_plan_chunks_splits_at_year_boundary():
    # a forcing-year boundary forces a chunk boundary (so the reader rolls cleanly) WITHOUT
    # re-bootstrapping AB2 — a year switch is forcing-only; the dynamics are continuous.
    ch = plan_chunks(200, 48, start_step=0, dt=180.0, split_at=[100])
    assert 100 in [c.start for c in ch]
    assert all(not (c.start < 100 < c.start + c.count) for c in ch)    # none straddles 100
    assert next(c for c in ch if c.start == 100).bootstrap_ab2 is False
    assert ch[0].start == 0 and sum(c.count for c in ch) == 200        # exact, contiguous cover


def test_calendar_ramp_aware_dt_change():
    # dt 180→240 after year 1 (the production dars run): the calendar must stay EXACT across the
    # mid-run dt change, else years 2-3 forcing is mis-dated (a single n·dt map drifts past the ramp).
    from fesom_jax.run_config import DtRamp
    R = 175200                                   # 1 year at dt=180 (365*86400/180)
    ramp = DtRamp(after_step=R, dt=240.0)
    # the elapsed clock is piecewise (dt0 up to R, dt1 after); _step_at_elapsed inverts it
    assert _elapsed_seconds(R, 180.0, ramp) == R * 180.0                 # exactly 1 model year
    assert _elapsed_seconds(R + 10, 180.0, ramp) == R * 180.0 + 10 * 240.0
    assert _step_at_elapsed(R * 180.0, 180.0, ramp) == R
    assert _step_at_elapsed(2 * 365 * 86400, 180.0, ramp) == 306600      # 1960 boundary (post-ramp)
    assert _step_at_elapsed(3 * 365 * 86400, 180.0, ramp) == 438000      # 3 model years total
    # the first post-ramp step is 1959-01-01 (NOT mis-dated by an absolute n·240)
    assert _chunk_dates(1958, 180.0, R, 1, ramp)[0] == (1959, 1, 0.0, 1)
    # year boundaries WITH the ramp: 1959 at the dt0 step, 1960 at the dt1 step
    assert _year_boundaries(1958, 180.0, 0, 438000, ramp) == [175200, 306600]
    # dt_ramp=None ⇒ bit-identical to the plain calendar (the off-path invariant)
    assert _chunk_dates(1958, 180.0, 200000, 3) == _chunk_dates(1958, 180.0, 200000, 3, None)
    assert _year_boundaries(1958, 180.0, 0, 525600) == [175200, 350400]


# ==========================================================================
# 2. The restart seam — continuous == chained (the headline gate)
# ==========================================================================
@pytest.fixture(scope="module")
def core2_setup():
    from fesom_jax.phc_ic import phc_initial_state
    mesh = load_mesh(CORE2_MESH)
    state = phc_initial_state(mesh, IC_DIR)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    sop = ssh.partition_ssh_operator(ssh.build_ssh_operator(mesh, dt=DT), part)
    # well-separated forcing dates ⇒ a genuine per-step seasonal swing (cf. test_forcing_sharded)
    stack = cf.stack([(YEAR, 15, 0.0, 1), (YEAR, 105, 0.0, 4),
                      (YEAR, 196, 0.0, 7), (YEAR, 288, 0.0, 10)])
    return dict(mesh=mesh, state=state, cf=cf, part=part, sm=sm, sop=sop, stack=stack, n=4)


def _owned_worst(a, b, part, sm, npes):
    worst = 0.0
    for fld in dataclasses.fields(State):
        A, B = np.asarray(getattr(a, fld.name)), np.asarray(getattr(b, fld.name))
        if A.shape[1] == sm.Lmax["nod"]:
            myd = part.myDim_nod2D
        elif A.shape[1] == sm.Lmax["elem"]:
            myd = part.myDim_elem2D
        else:
            continue
        for d in range(npes):
            md = int(myd[d])
            if md:
                worst = max(worst, float(np.max(np.abs(A[d, :md] - B[d, :md]))))
    return worst


@have_forcing
def test_restart_seam_continuous_equals_chained(core2_setup, tmp_path):
    fx = core2_setup
    common = dict(mesh=fx["mesh"], part=fx["part"], sm=fx["sm"], sop=fx["sop"], year=YEAR)
    kw = dict(kpp=KppConfig(), dt=DT)
    n = fx["n"]                                            # 4 steps, split 2 + 2

    # continuous: one invocation, n steps, one chunk
    cont = run_from_config(RunConfig(n_steps=n, **kw), state0=fx["state"],
                           forcing=fx["cf"], forcing_stack=fx["stack"], **common)

    # chained: invocation 1 runs n//2 and writes a portable restart …
    r1 = tmp_path / "restart1"
    half = jax.tree.map(lambda x: x[: n // 2], fx["stack"])
    run_from_config(RunConfig(n_steps=n // 2, restart_out=str(r1), **kw), state0=fx["state"],
                    forcing=fx["cf"], forcing_stack=half, **common)
    # … invocation 2 READS it back and continues the remaining n//2 (start_step from the restart)
    rest = jax.tree.map(lambda x: x[n // 2:], fx["stack"])
    chained = run_from_config(RunConfig(n_steps=n // 2, restart_in=str(r1), **kw),
                              forcing=fx["cf"], forcing_stack=rest, **common)

    assert chained.step == n, f"resumed step counter {chained.step} != {n}"
    # RunResult.state_p is FOLDED [P*Lmax] (the restart-ready form); unfold to [P,Lmax] to compare
    from fesom_jax.integrate_sharded import unfold_state
    worst = _owned_worst(unfold_state(cont.state_p, 1), unfold_state(chained.state_p, 1),
                         fx["part"], fx["sm"], 1)
    # restart round-trip is bit-faithful (A1) + AB2 continues ⇒ chained == continuous to the FCT
    # climate-close floor (the one-scan vs two-scan reassociation; a cold-AB2 bug would be O(1)).
    assert worst < 5e-3, f"restart-seam continuous-vs-chained owned max|Δ|={worst:.3e}"
    print(f"RUN_DRIVER_OK (restart-seam max|Δ|={worst:.2e})")


def test_run_boundary_node_is_global_coastal_mask():
    """run_from_config seeds the mEVP boundary condition from the GLOBAL coastal mask, partitioned
    in (`boundary_node_p`), NOT the per-device local-mesh fallback — which would treat PARTITION
    CUTS as coasts and zero ice velocity on artificial interior walls. Verify the sharded mask
    run.py builds reconstructs to `ice_evp.boundary_node_mask(mesh)` exactly (host-only)."""
    from fesom_jax import ice_evp, partit, shard_mesh
    from fesom_jax.zarr_output import _folded_gid_owned
    # the PACKAGED pi mesh: the property under test (global coastal mask vs the per-device
    # fallback) is mesh-agnostic, and pi ships with the package, so this gate runs everywhere
    # rather than silently needing the Levante-only CORE2 mesh. pi has 455 coastal nodes.
    mesh = load_mesh()
    part = partit.synth_block_partition(mesh.nod2D, mesh.elem2D, mesh.edge2D, 4)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    bn = np.asarray(ice_evp.boundary_node_mask(mesh)).astype(bool)         # GLOBAL [nod2D]
    bn_p = shard_mesh._shard_along_axis(bn, part.myList_nod2D, sm.Lmax["nod"], 0, False)
    gid, owned = _folded_gid_owned(part, sm, "nod")                        # [P*Lmax]
    flat = np.asarray(bn_p).reshape(-1).astype(bool)
    recon = np.zeros(int(mesh.nod2D), bool)
    recon[gid[owned]] = flat[owned]
    np.testing.assert_array_equal(recon, bn)                               # exact global mask
    assert bn.sum() > 100                                                  # the BC is meaningful


@have_forcing
def test_checkpoint_cadence(core2_setup, tmp_path, monkeypatch):
    """`checkpoint_every` writes a ROLLING intermediate restart each time a chunk crosses a
    step-boundary (crash-safety / segment hand-off), then the final restart — and NOT a double
    write on the last chunk. Spy on write_restart to lock the cadence (the resume == continuous
    correctness is the restart-seam test; this is the new when-to-write logic)."""
    from fesom_jax import zarr_output
    fx = core2_setup
    common = dict(mesh=fx["mesh"], part=fx["part"], sm=fx["sm"], sop=fx["sop"], year=YEAR)
    calls = []
    orig = zarr_output.write_restart
    monkeypatch.setattr(zarr_output, "write_restart",
                        lambda *a, **kw: (calls.append(int(kw["step"])), orig(*a, **kw))[1])
    # 4 steps, 1/chunk, checkpoint_every=2 ⇒ checkpoint at step 2 (chunk 4 is last ⇒ no ckpt) + final at 4
    run_from_config(RunConfig(n_steps=4, kpp=KppConfig(), dt=DT), state0=fx["state"],
                    forcing=fx["cf"], forcing_stack=fx["stack"], chunk_steps=1,
                    out_dir=str(tmp_path / "r"), checkpoint_every=2, **common)
    assert calls == [2, 4], f"checkpoint steps {calls} != [2, 4]"
    # no checkpoint_every ⇒ only the final restart (unchanged default behavior)
    calls.clear()
    run_from_config(RunConfig(n_steps=4, kpp=KppConfig(), dt=DT), state0=fx["state"],
                    forcing=fx["cf"], forcing_stack=fx["stack"], chunk_steps=1,
                    out_dir=str(tmp_path / "r2"), **common)
    assert calls == [4], f"without checkpoint_every, only the final restart; got {calls}"


@have_forcing
def test_restart_archive_cadence(core2_setup, tmp_path, monkeypatch):
    """`restart_archive_out` writes an IMMUTABLE, uniquely-named restart at every calendar
    boundary crossed, PLUS unconditionally at the end of the run (mirrors FESOM3's "the last
    step is always checkpointed") — never overwriting a prior one, and updates `restart.latest`
    to the newest. A SEPARATE stream from the plain rolling `checkpoint_every`/`out_dir` restart,
    which this test also exercises unchanged (both fire independently, no interference)."""
    from fesom_jax import zarr_output
    fx = core2_setup
    common = dict(mesh=fx["mesh"], part=fx["part"], sm=fx["sm"], sop=fx["sop"], year=YEAR)
    calls = []
    orig = zarr_output.write_restart
    monkeypatch.setattr(zarr_output, "write_restart",
                        lambda out_dir, *a, **kw: (calls.append((out_dir, int(kw["step"]))),
                                                    orig(out_dir, *a, **kw))[1])
    archive = tmp_path / "archive"
    # 60 steps @ DT=1800 (48 steps/day) crosses ONE day boundary (step 48) — mirrors test_daily_output.
    # fx["stack"] is only pre-built for fx["n"]=4 steps, so build a correctly-sized one here (same
    # pattern as test_daily_output).
    n = 60
    stack = fx["cf"].stack(_chunk_dates(YEAR, DT, 0, n, None))
    run_from_config(RunConfig(n_steps=n, kpp=KppConfig(), dt=DT), state0=fx["state"],
                    forcing=fx["cf"], forcing_stack=stack, chunk_steps=12,
                    out_dir=str(tmp_path / "r"), checkpoint_every=100,   # never fires in 60 steps
                    restart_archive_out=str(archive), restart_archive_period="day", **common)
    archive_calls = sorted(s for d, s in calls if str(d).startswith(str(archive)))
    assert archive_calls == [48, 60], f"archival steps {archive_calls} != [48, 60] (boundary + final)"
    rolling_calls = sorted(s for d, s in calls if d == str(tmp_path / "r"))
    assert rolling_calls == [60], f"unaffected rolling stream: {rolling_calls} != [60]"
    # both archival directories exist, independently, forever (no overwrite) — and restart.latest
    # names the chronologically LAST one (step 60), not merely the last call in the calls list.
    import zarr as _zarr
    last = zarr_output.resolve_latest_restart(str(archive))
    assert last is not None and Path(last).is_dir()
    g_last = _zarr.open_group(last, mode="r")
    assert int(g_last.attrs["step"]) == 60
    dirs = sorted(p.name for p in archive.iterdir() if p.is_dir())
    assert len(dirs) == 2, f"expected 2 immutable archival dirs, got {dirs}"


@have_forcing
def test_inrun_multichunk_equals_singlechunk(core2_setup):
    """In ONE ``run_from_config`` call, splitting into multiple fine forcing-chunks (the A4/A6
    memory contract — NG5 can't pre-stack a whole year) carries the FOLDED state chunk→chunk
    **on-device** (no disk round-trip, unlike the restart seam) ⇒ near-bit-identical to running
    it as a single chunk. Guards the in-job multi-chunk path the NG5 ladder runs at scale; the
    dars B0 only ever drove 1 chunk per job, so this is otherwise unexercised."""
    fx = core2_setup
    common = dict(mesh=fx["mesh"], part=fx["part"], sm=fx["sm"], sop=fx["sop"], year=YEAR)
    kw = dict(kpp=KppConfig(), dt=DT)
    n = fx["n"]                                            # 4 steps, run as 1×4 vs 2×2
    one_chunk = run_from_config(RunConfig(n_steps=n, **kw), state0=fx["state"], forcing=fx["cf"],
                                forcing_stack=fx["stack"], chunk_steps=n, **common)
    two_chunk = run_from_config(RunConfig(n_steps=n, **kw), state0=fx["state"], forcing=fx["cf"],
                                forcing_stack=fx["stack"], chunk_steps=n // 2, **common)
    from fesom_jax.integrate_sharded import unfold_state
    worst = _owned_worst(unfold_state(one_chunk.state_p, 1), unfold_state(two_chunk.state_p, 1),
                         fx["part"], fx["sm"], 1)
    # The chunk BOUNDARY step is a direct bootstrap-`one()` call instead of a scan iteration ⇒ XLA
    # reassociates it at the SAME ~5.7e-8 float64 floor as the restart seam (the seam's error is this
    # boundary step, NOT the bit-faithful disk round-trip). Meaningless physically (≈3e-10 rel on T),
    # while an O(1) chaining bug — wrong forcing slice / dropped state / bad bootstrap flag — is caught.
    assert worst < 1e-6, f"in-run multichunk vs singlechunk owned max|Δ|={worst:.3e}"
    print(f"MULTICHUNK_OK (in-run 2-chunk vs 1-chunk owned max|Δ|={worst:.2e})")


@have_forcing
def test_single_step_run(core2_setup):
    fx = core2_setup
    res = run_from_config(RunConfig(n_steps=1, kpp=KppConfig(), dt=DT), mesh=fx["mesh"],
                          part=fx["part"], sm=fx["sm"], sop=fx["sop"], state0=fx["state"],
                          forcing=fx["cf"], forcing_stack=jax.tree.map(lambda x: x[:1], fx["stack"]),
                          year=YEAR)
    assert res.step == 1
    assert np.all(np.isfinite(np.asarray(res.state_p.T)))


@have_forcing
def test_daily_output(core2_setup, tmp_path):
    """Daily-mean output: one ushow-readable zarr per calendar day with SST/SSS/T@100m/u@100m/v@100m,
    gather-free, gated on ``daily_start_step``. A 60-step run at DT=1800 (48 steps/day) crosses ONE
    day boundary ⇒ a completed day (mid-loop rollover write) + a partial day (final flush)."""
    import zarr
    fx = core2_setup
    n = 60                                                # 1.25 days at DT=1800 (48 steps/day)
    stack = fx["cf"].stack(_chunk_dates(YEAR, DT, 0, n, None))
    common = dict(forcing=fx["cf"], forcing_stack=stack, mesh=fx["mesh"], part=fx["part"],
                  sm=fx["sm"], sop=fx["sop"], year=YEAR, chunk_steps=12)
    daily = tmp_path / "daily"
    run_from_config(RunConfig(n_steps=n, kpp=KppConfig(), dt=DT), state0=fx["state"],
                    daily_out=str(daily), daily_start_step=0, **common)
    days = sorted(p.name for p in daily.glob("day_*"))
    assert len(days) >= 2, f"expected >=2 daily zarrs (crossed a day boundary), got {days}"
    g = zarr.open_group(str(daily / days[0]), "r")
    keys = set(g.array_keys())
    assert {"sst", "sss", "temp100", "u100", "v100", "lon", "lat", "time"} <= keys, f"fields {keys}"
    # sst carries a LEADING time axis (size 1) so ushow's per-point time-series extraction can find
    # it (it keys off the DATA variable's own _ARRAY_DIMENSIONS, not just a bare time coordinate) --
    # lon/lat stay node-only (mesh-invariant, never time-varying).
    assert tuple(g["sst"].shape) == (1,) + tuple(g["lon"].shape), "sst must be [1, nod2] (leading time)"
    assert g["sst"].attrs["_ARRAY_DIMENSIONS"][0] == "time", "sst's first dim must be named 'time'"
    assert tuple(g["time"].shape) == (1,), "the time coordinate itself is a length-1 array"
    lat, sst = np.asarray(g["lat"]), np.asarray(g["sst"])[0]
    owned = lat > -89.0                                   # non-sentinel (real-owner) lanes
    assert owned.any() and np.all(np.isfinite(sst[owned])), "owned SST daily means must be finite"

    # the start-step gate: with daily_start_step past the run, NOTHING is written (year-2 behaviour)
    daily2 = tmp_path / "daily2"
    run_from_config(RunConfig(n_steps=n, kpp=KppConfig(), dt=DT), state0=fx["state"],
                    daily_out=str(daily2), daily_start_step=10_000, **common)
    assert not list(daily2.glob("day_*")), "daily_start_step gate must suppress output before it"
    print(f"DAILY_OUTPUT_OK ({len(days)} day zarrs {days}; fields {sorted(keys - {'lon', 'lat'})})")


@have_forcing
def test_monthly_output(core2_setup, tmp_path):
    """Monthly-mean output (the CORE2 1958-2020 hindcast climatology): one ushow-readable zarr per
    calendar month with FULL temp/salt/u/v (3-D) + ssh/a_ice/m_ice (2-D), gather-free, gated on
    ``monthly_start_step``. A 60-step run at DT=1800 stays within January 1958 ⇒ the final-flush
    writes one month (``1958_01``) with the right fields, finite owned means, and ``n_samples>0``
    (``init().update()`` folds the first sample — a count=0 month would write all zeros). The
    month-ROLLOVER write is the daily rollover's twin (exercised by the day-boundary cross in
    test_daily_output)."""
    import zarr
    fx = core2_setup
    n = 60                                                # 1.25 days at DT=1800 — all within Jan 1958
    stack = fx["cf"].stack(_chunk_dates(YEAR, DT, 0, n, None))
    common = dict(forcing=fx["cf"], forcing_stack=stack, mesh=fx["mesh"], part=fx["part"],
                  sm=fx["sm"], sop=fx["sop"], year=YEAR, chunk_steps=12)
    monthly = tmp_path / "monthly"
    run_from_config(RunConfig(n_steps=n, kpp=KppConfig(), dt=DT), state0=fx["state"],
                    monthly_out=str(monthly), monthly_start_step=0, **common)
    months = sorted(p.name for p in monthly.glob("[0-9]*_[0-9]*"))
    assert months == ["1958_01"], f"expected the partial-Jan flush, got {months}"
    g = zarr.open_group(str(monthly / months[0]), "r")
    keys = set(g.array_keys())
    assert {"temp", "salt", "u", "v", "ssh", "a_ice", "m_ice", "lon", "lat", "time"} <= keys, f"fields {keys}"
    # every data field gains a LEADING time axis (size 1) -- see test_daily_output's comment.
    assert g["temp"].ndim == 3 and g["ssh"].ndim == 2, "temp is [1,nz,nod2]; ssh is [1,nod2]"
    assert g["temp"].attrs["_ARRAY_DIMENSIONS"][0] == "time", "temp's first dim must be named 'time'"
    assert int(g.attrs["n_samples"]) > 0, "init().update() must fold samples (count>0, not zeros)"
    lat, temp = np.asarray(g["lat"]), np.asarray(g["temp"])[0]
    owned = lat > -89.0                                   # non-sentinel (real-owner) lanes
    assert owned.any() and np.all(np.isfinite(temp[:, owned])), "owned monthly temp means finite"

    # the start-step gate: with monthly_start_step past the run, NOTHING is written
    monthly2 = tmp_path / "monthly2"
    run_from_config(RunConfig(n_steps=n, kpp=KppConfig(), dt=DT), state0=fx["state"],
                    monthly_out=str(monthly2), monthly_start_step=10_000, **common)
    assert not list(monthly2.glob("[0-9]*_[0-9]*")), "monthly_start_step gate must suppress output"
    print(f"MONTHLY_OUTPUT_OK ({months}; fields {sorted(keys - {'lon', 'lat'})}, "
          f"n_samples={int(g.attrs['n_samples'])})")
