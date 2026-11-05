#!/usr/bin/env python
"""Time and profile the coupled CORE3 ocean alone, the way awicm's concurrent coupler drives it (coupler.py
ConcurrentOcean._job): per coupling interval the OASIS-level forcing of a coupled restart (coupler_pending.npz,
held fixed), fesom_jax.coupled.oasis_to_fesom, ocean_interval (2 steps of 1800 s), ocean_to_atm, and a block on
the fields for the atmosphere. Monthly sums and the w/fw extras are on as in `--setup jaxcpl`.

    ocean_speed.py run OUT --config C [--restart DIR] [--pending NPZ] [--partition dist_2] [--warm 3]
                       [--intervals 24] [--prof]
    ocean_speed.py compare A.npz B.npz [--json OUT]

run writes OUT.json (ms per interval: every timed interval, median, mean; peak device memory) and OUT.npz (every
State leaf at the end in global entity order, its own dtype). --prof calls cudaProfilerStart/Stop around the timed
intervals (run under nsys --capture-range=cudaProfilerApi); --compare-to REF.npz compares the final State with a
reference in the run (with --no-save, no State copy is written). compare prints per field max |a-b|, the max relative
difference (|a-b| / max|a|) and whether the two are bitwise equal.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RST = "/work/ba1583/a270088/awicm/core3/cpl/int/y1855_chl01/restart/18550701"


def _cudart():
    import nvidia.cuda_runtime
    lib = glob.glob(os.path.join(list(nvidia.cuda_runtime.__path__)[0], "lib", "libcudart.so*"))
    return ctypes.CDLL(sorted(lib)[0])


def _state_arrays(state, st):
    import jax
    from fesom_jax.coupled import global_view
    out = {}
    names = [k for k, v in vars(state).items() if isinstance(v, jax.Array)]
    if st.npes == 1:
        for k in names:
            out[k] = np.asarray(getattr(state, k))
    else:
        for k in names:
            try:
                out[k] = np.asarray(global_view(state, st, (k,))[k])
            except ValueError:
                pass                                   # not a folded node / element field (scalars)
    return out


def _save_state(state, st, path, extra=None):
    out = {**_state_arrays(state, st), **(extra or {})}
    np.savez(path, **out)                                  # native dtypes (float32 State in single precision)
    return sorted(out)


def _compare(A, B):
    """Per field of two State dicts (or npz): bitwise, max |a-b|, max relative difference, count."""
    rows = {}
    for k in sorted(set(A.keys() if isinstance(A, dict) else A.files) & set(B.keys() if isinstance(B, dict)
                                                                           else B.files)):
        x, y = np.asarray(A[k]), np.asarray(B[k])
        if x.shape != y.shape:
            rows[k] = {"shape": [list(x.shape), list(y.shape)]}
            continue
        if x.dtype.kind != "f":
            rows[k] = {"bitwise": bool(np.array_equal(x, y)), "ndiff": int((x != y).sum())}
            continue
        x, y = x.astype(np.float64), y.astype(np.float64)
        d = np.abs(x - y)
        s = float(np.nanmax(np.abs(x))) if x.size else 0.0
        rows[k] = {"bitwise": bool(np.array_equal(x, y, equal_nan=True)), "max_abs": float(np.nanmax(d)) if d.size
                   else 0.0, "max_rel": (float(np.nanmax(d)) / s if s > 0 else 0.0), "ndiff": int((d > 0).sum()),
                   "scale": s}
    return rows


class _GilProbe:
    """--gil-probe / --gil-load F (diagnostic): a second Python thread, as the coupled run's atmosphere main thread
    is. It sleeps 0.2 ms at a time and records how late it wakes (a wake-up needs the GIL: a late wake-up means the
    ocean's thread held the GIL meanwhile); with --gil-load F > 0 it also spins in pure Python for F of the time
    (holding the GIL like a busy dispatching thread)."""

    def __init__(self, load=0.0):
        import threading
        self.load, self.lat, self.run = float(load or 0.0), [], True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.run:
            t0 = time.perf_counter()
            time.sleep(0.0002)
            self.lat.append(time.perf_counter() - t0 - 0.0002)
            if self.load > 0:                      # spin F/(1-F) x the sleep
                t1 = time.perf_counter()
                while time.perf_counter() - t1 < 0.0002 * self.load / max(1e-3, 1 - self.load):
                    pass

    def stop(self):
        self.run = False
        self.t.join()
        x = 1e3 * np.asarray(self.lat)
        return {"load": self.load, "n": int(x.size), "late_ms_max": round(float(x.max()), 2),
                "late_ms_p99": round(float(np.percentile(x, 99)), 3), "late_ms_mean": round(float(x.mean()), 3),
                "late_over_2ms_total_ms": round(float(x[x > 2].sum()), 1)}


def cmd_run(a):
    import jax
    import jax.numpy as jnp
    from fesom_jax import coupled as C
    t0 = time.perf_counter()
    mon = a.out + "_monthly"
    state, st = C.init_ocean(a.config, restart_in=a.restart, mode="flux", year0=1855, nsteps=2,
                             partition=a.partition, monthly_out=mon, cpl_runoff="coupled",
                             monthly_extra=("w", "fw"))
    t_init = time.perf_counter() - t0
    print(f"[speed] init {t_init:.1f}s, {st.calendar()}, npes {st.npes}, gpu {st.cfg.gpu}", flush=True)
    z = np.load(a.pending)
    fo = {k: jnp.asarray(z[k].astype(np.float32)) for k in z.files}
    fo = C.oasis_to_fesom(fo)
    if a.check_forcing:            # the jitted forcing build against the eager one, on this device, bitwise
        from fesom_jax.coupled import flux
        dates = st.dates(2)
        _, e = flux.flux_step_forcing(state, fo, st, dates)
        _, j = flux.flux_step_forcing_jit(state, fo, st, dates)
        bad = {f: int((np.asarray(getattr(e, f)) != np.asarray(getattr(j, f))).sum()) for f in e._fields
               if getattr(e, f) is not None and not np.array_equal(np.asarray(getattr(e, f)), np.asarray(getattr(j, f)))}
        print(f"[speed] check-forcing: jitted vs eager forcing build {'BITWISE' if not bad else 'DIFFER ' + str(bad)}",
              flush=True)
    times, t_first = [], None
    rt = _cudart() if a.prof else None
    probe = _GilProbe(a.gil_load) if (a.gil_probe or a.gil_load) else None
    pyprof, prep = None, None
    for i in range(a.warm + a.intervals):
        if i == a.warm and rt is not None:
            rt.cudaProfilerStart()
        if i == a.warm and a.pyprof:          # diagnostic: the host's Python time over the timed intervals
            import cProfile
            pyprof = cProfile.Profile()
            pyprof.enable()
        t1 = time.perf_counter()
        if a.prep_ahead:           # the coupler's AWICM_OCEAN_PREP: the forcing built before the interval
            if prep is None:
                prep = C.prepare_interval(fo, st, st.step, 2)
            state = C.ocean_interval(state, st, C.place_prepared(prep, st), nsteps=2)
            o2a = C.ocean_to_atm(state, st)
            prep = C.prepare_interval(fo, st, st.step, 2)        # the next interval's, while this one ends
        else:
            state = C.ocean_interval(state, st, fo, nsteps=2)
            o2a = C.ocean_to_atm(state, st)
        jax.block_until_ready(o2a)
        dt = time.perf_counter() - t1
        if i == 0:
            t_first = dt
        if i >= a.warm:
            times.append(dt)
    if rt is not None:
        rt.cudaProfilerStop()
    if pyprof is not None:
        import io
        import pstats
        pyprof.disable()
        out = io.StringIO()
        pstats.Stats(pyprof, stream=out).sort_stats("tottime").print_stats(25)
        pstats.Stats(pyprof, stream=out).sort_stats("cumulative").print_stats(40)
        with open(a.pyprof, "w") as fh:
            fh.write(out.getvalue())
    gil = probe.stop() if probe is not None else None
    ms = 1e3 * np.asarray(times)
    fin = all(bool(np.isfinite(np.asarray(getattr(state, f), np.float64)).all())
              for f in ("T", "S", "uv", "eta_n", "a_ice", "m_ice"))
    mem = {}
    for d in jax.local_devices()[:st.npes]:
        try:
            mem[str(d)] = round(d.memory_stats()["peak_bytes_in_use"] / 2**30, 2)
        except Exception:                                # noqa: BLE001
            pass
    rec = {"config": a.config, "partition": a.partition, "gpu": st.cfg.gpu, "ssh": st.cfg.ssh,
           "init_s": round(t_init, 1), "first_interval_s": round(t_first, 1), "warm": a.warm,
           "intervals": a.intervals, "ms_per_interval_median": round(float(np.median(ms)), 2),
           "ms_per_interval_mean": round(float(ms.mean()), 2), "ms_per_interval_min": round(float(ms.min()), 2),
           "ms_per_step_median": round(float(np.median(ms)) / 2, 2), "s_per_model_day": round(float(np.median(ms))
                                                                                                * 24 / 1e3, 3),
           "finite": fin, "peak_gb": mem, "calendar_end": st.calendar(),
           "xla_flags": os.environ.get("XLA_FLAGS", ""), "ms_all": [round(float(x), 2) for x in ms]}
    if gil is not None:
        rec["gil"] = gil
    o2a_last = {"o2a_" + k: np.asarray(v) for k, v in o2a.items()}    # the last interval's fields for the atmosphere
    if a.compare_to:               # compare the final State with a reference npz here instead of saving it
        rows = _compare({**_state_arrays(state, st), **o2a_last}, np.load(a.compare_to))
        rec["compare_to"] = a.compare_to
        rec["bitwise"] = all(r.get("bitwise", False) for r in rows.values())
        rec["differ"] = {k: r for k, r in rows.items() if not r.get("bitwise", False)}
    if not a.no_save:
        rec["saved"] = _save_state(state, st, a.out + ".npz", extra=o2a_last)
    with open(a.out + ".json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print("[speed] " + json.dumps({k: v for k, v in rec.items() if k not in ("ms_all", "saved", "differ")}),
          flush=True)
    if a.compare_to:
        print(f"[speed] compare: {'BITWISE' if rec['bitwise'] else 'differ'} against {a.compare_to}; " + "; ".join(
            f"{k} max_rel {r.get('max_rel')} n {r.get('ndiff')}" for k, r in list(rec["differ"].items())[:12]),
            flush=True)


def cmd_compare(a):
    rows = _compare(np.load(a.a), np.load(a.b))
    allbit = all(r.get("bitwise", False) for r in rows.values())
    print(f"compare {a.a} vs {a.b}: {'BITWISE' if allbit else 'differ'}")
    for k in ("T", "S", "uv", "eta_n", "a_ice", "m_ice", "m_snow", "u_ice", "tke", "w"):
        if k in rows:
            print(f"  {k:8s} {rows[k]}")
    other = [k for k in rows if k not in ("T", "S", "uv", "eta_n", "a_ice", "m_ice", "m_snow", "u_ice", "tke", "w")
             and not rows[k].get("bitwise", True)]
    print(f"  other fields that differ: {other}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"bitwise": allbit, "fields": rows}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--config", default=str(ROOT / "configs" / "core3_cpl.yaml"))
    r.add_argument("--restart", default=RST + "/ocean")
    r.add_argument("--pending", default=RST + "/coupler_pending.npz")
    r.add_argument("--partition", default=None)
    r.add_argument("--warm", type=int, default=3)
    r.add_argument("--intervals", type=int, default=24)
    r.add_argument("--prof", action="store_true")
    r.add_argument("--prep-ahead", action="store_true", help="each interval's forcing built ahead with "
                   "coupled.prepare_interval (dispatched while the previous interval ends), as awicm's AWICM_OCEAN_PREP")
    r.add_argument("--pyprof", default=None, help="diagnostic: cProfile of the timed intervals to this file")
    r.add_argument("--gil-probe", action="store_true", help="diagnostic: a Python thread measuring GIL hold times")
    r.add_argument("--gil-load", type=float, default=0.0, help="diagnostic: that thread also busy in Python this "
                   "fraction of the time")
    r.add_argument("--no-save", action="store_true")
    r.add_argument("--compare-to", default=None, help="reference State npz: compare the final State with it "
                   "(bitwise / max relative difference per field, in OUT.json) instead of keeping a copy")
    r.add_argument("--check-forcing", action="store_true", help="compare the jitted and the eager forcing builds")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json", default=None)
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
