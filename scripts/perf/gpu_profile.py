#!/usr/bin/env python
"""CORE2 GPU speed plan (docs/plans/20261020-core2-gpu-speed.md), Task 1: time, profile and compare
the PRODUCTION run path — ``scripts/run_from_config.py``'s own ``main()``, every CLI flag of it
(halo transport, ``--cheb-degree``, ``--forcing-on-device``, ...) passed through unchanged.

    gpu_profile.py time  OUT [--chunk-steps K] [--warm-chunks W] [--steps N] -- CONFIG.yaml [run flags]
    gpu_profile.py prof  OUT [same]                -- ...   (run it under nsys --capture-range=cudaProfilerApi)
    gpu_profile.py noise OUT [--reps R] [--steps N] -- ...   (R final States + their pooled noise floor)
    gpu_profile.py compare A.npz B.npz --floor noise.json   (a change against the GPU's own noise)
    gpu_profile.py stats OUT PREFIX [--steps N] [--devices P]   (nsys stats CSVs -> per step and GPU)

time/prof: one run of W*K + N steps in K-step chunks. The first W chunks are the warm-up: the first
call compiles the cold-input executable, the second the folded-input one (run.py compiles TWO), so
W >= 2. At the first timed chunk the timer drains the device once and starts the clock (and the
nsys capture for ``prof``); the clock stops when the last chunk's State is ready. In between, the
host builds each chunk's forcing while the device runs the previous one, exactly as in production,
so ``steady_ms_per_step`` is the production loop's steady cost. Also recorded: the host time
between chunk calls (forcing build; if it exceeds the device time the run is host-bound), the time
inside the calls (dispatch, or a host sync), peak device memory, and whether the final State is
finite (a run gone NaN looks fast). ``time`` also runs on several nodes: JDIST=1 under ``srun`` with one
process per node (run_from_config's import initialises jax.distributed); each process prints its own line.

noise: R runs from the cold start, each final State saved as OUT/state_<i>.npz (the FOLDED sharded
State: same partition ⇒ same layout, padding lanes stay 0), and per field the largest relative
difference over all pairs (OUT/noise.json). GPU runs are not bit-reproducible (atomic scatter-adds);
a change passes when its difference to the base is within this floor (``compare``).

Never writes model output: a configuration with a restart or output stream is refused (several
production configs point restart_out at a production run directory; use the ``*_full_bench.yaml``
twins). FESOM_REUSE_EXE is forced on (without it every chunk recompiles and the timing is
meaningless). prof and noise: single process only (one node).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import dataclasses
import glob
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_KW = ("out_dir", "checkpoint_every", "restart_archive_out", "daily_out", "monthly_out")


def say(msg):
    proc = f" p{os.environ.get('SLURM_PROCID', '?')}" if os.environ.get("JDIST") else ""
    print(f"[{time.strftime('%H:%M:%S')}{proc}] {msg}", flush=True)


def sypd(ms_per_step: float, dt: float) -> float:
    """Simulated years per wall day at time step ``dt`` [s] (365-day years)."""
    return dt / (ms_per_step / 1e3 * 365.0)


class ChunkTimer:
    """Wraps the per-chunk call. Calls 0..warm_chunks-1 are untimed; at call ``warm_chunks`` the
    state argument is drained (``sync``), the clock starts and ``on_start`` runs (cudaProfilerStart).
    Inside the window it records the host time between calls and the time inside them."""

    def __init__(self, warm_chunks, sync, on_start=None, clock=time.perf_counter, state_index=0):
        self.warm_chunks, self.sync, self.on_start = warm_chunks, sync, on_start
        self.clock, self.state_index = clock, state_index
        self.calls = self.timed_calls = 0
        self.t0 = self._last_return = None
        self.gaps, self.call_s = [], 0.0

    def wrap(self, fn):
        def wrapped(*args, **kw):
            if self.calls == self.warm_chunks:
                self.sync(args[self.state_index])
                self.t0 = self.clock()
                if self.on_start is not None:
                    self.on_start()
            elif self.t0 is not None:
                self.gaps.append(self.clock() - self._last_return)
            t = self.clock()
            out = fn(*args, **kw)
            r = self.clock()
            if self.t0 is not None:
                self.call_s += r - t
                self.timed_calls += 1
            self._last_return = r
            self.calls += 1
            return out
        return wrapped

    def summary(self, t_end, timed_steps):
        if self.t0 is None:
            raise RuntimeError(f"the timing window never opened: {self.calls} chunk calls, "
                               f"{self.warm_chunks} warm")
        window = t_end - self.t0
        per_chunk = timed_steps / self.timed_calls
        gap = sum(self.gaps) / len(self.gaps) if self.gaps else 0.0
        return {"window_s": window, "steady_ms_per_step": 1e3 * window / timed_steps,
                "calls": self.calls, "timed_calls": self.timed_calls,
                "host_gap_ms_per_step": 1e3 * gap / per_chunk,
                "call_ms_per_step": 1e3 * self.call_s / timed_steps}


def check_no_output(kw):
    on = [k for k in _OUTPUT_KW if kw.get(k)]
    if on:
        raise SystemExit(f"gpu_profile: refusing a run that writes output ({', '.join(on)}); "
                         "use an output-free config (e.g. configs/core2_full_bench.yaml)")


def field_diffs(a: dict, b: dict) -> dict:
    """Per field: entries that differ, max |a - b|, and that relative to max |a|."""
    import numpy as np
    out = {}
    for f, x in a.items():
        y = b[f]
        d = np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))
        scale = float(np.max(np.abs(x))) if np.size(x) else 0.0
        mx = float(d.max()) if d.size else 0.0
        out[f] = {"n_diff": int(np.count_nonzero(x != y)), "size": int(np.size(x)), "max_abs": mx,
                  "max_rel": mx / (scale or 1.0)}
    return out


def pooled_floor(states: list) -> dict:
    """Per field, the largest relative difference over all pairs of repeat runs."""
    if len(states) < 2:
        raise ValueError("a noise floor needs at least two runs")
    floor = {}
    for a, b in itertools.combinations(states, 2):
        for f, r in field_diffs(a, b).items():
            floor[f] = max(floor.get(f, 0.0), r["max_rel"])
    return floor


def judge(diffs: dict, floor: dict, factor: float = 1.0):
    """``(ok, rows)``: every field's relative difference within ``factor`` x its noise floor (a field
    whose floor is exactly 0 must not move at all)."""
    rows = []
    for f, r in diffs.items():
        fl = floor.get(f, 0.0)
        ok = r["max_rel"] <= factor * fl
        rows.append({"field": f, "max_rel": r["max_rel"], "floor": fl,
                     "ratio": r["max_rel"] / fl if fl else (0.0 if r["max_rel"] == 0 else float("inf")),
                     "ok": ok})
    return all(r["ok"] for r in rows), rows


def save_state(path, fields: dict):
    import numpy as np
    np.savez(path, **fields)


def load_state(path) -> dict:
    import numpy as np
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _read_csv(prefix, report):
    p = f"{prefix}_{report}.csv"
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def nsys_summary(prefix, steps, devices) -> dict:
    """Per step and GPU from ``nsys stats`` CSVs (cuda_api_sum, cuda_gpu_kern_sum,
    cuda_gpu_mem_time_sum): launches, graph launches, syncs, D2D copies, kernel and NCCL time."""
    api = _read_csv(prefix, "cuda_api_sum")
    kern = _read_csv(prefix, "cuda_gpu_kern_sum")
    mem = _read_csv(prefix, "cuda_gpu_mem_time_sum")
    per = steps * devices
    syncs = ("cuStreamSynchronize", "cudaStreamSynchronize", "cuEventSynchronize",
             "cudaEventSynchronize", "cuCtxSynchronize")

    def api_count(*names):
        return sum(int(r["Num Calls"]) for r in api if r["Name"].startswith(names))

    def api_ns(*names):
        return sum(float(r["Total Time (ns)"]) for r in api if r["Name"].startswith(names))

    ktot = sum(float(r["Total Time (ns)"]) for r in kern)
    knccl = sum(float(r["Total Time (ns)"]) for r in kern if "nccl" in r["Name"].lower())
    return {
        "launches_per_step_gpu": api_count("cuLaunchKernel", "cudaLaunchKernel") / per,
        "graph_launches_per_step_gpu": api_count("cuGraphLaunch", "cudaGraphLaunch") / per,
        "syncs_per_step_gpu": api_count(*syncs) / per,
        "d2d_copies_per_step_gpu": sum(int(r["Count"]) for r in mem
                                       if "Device-to-Device" in r["Operation"]) / per,
        "kernel_ms_per_step_gpu": ktot / 1e6 / per,
        "nccl_ms_per_step_gpu": knccl / 1e6 / per,
        "launch_api_ms_per_step_gpu": api_ns("cuLaunchKernel", "cudaLaunchKernel") / 1e6 / per,
        "sync_api_ms_per_step_gpu": api_ns(*syncs) / 1e6 / per,
        "top_kernels": [{"name": r["Name"][:120], "pct": float(r["Time (%)"]),
                         "instances": int(r["Instances"])}
                        for r in sorted(kern, key=lambda r: -float(r["Total Time (ns)"]))[:12]],
    }


def busy_from_sqlite(path, steps, devices) -> dict:
    """GPU busy time per step and GPU from the nsys export: each GPU's union of kernel, memcpy,
    memset and CUDA graph intervals (with ``--cuda-graph-trace=graph`` a graph's kernels are one
    record only), averaged over the GPUs."""
    import collections
    import sqlite3
    if not os.path.exists(path):
        return {}
    db = sqlite3.connect(path)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    per = steps * devices
    out, iv = {}, collections.defaultdict(list)
    for key, t in (("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"), ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
                   ("memset", "CUPTI_ACTIVITY_KIND_MEMSET"), ("graph", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE")):
        rows = db.execute(f"select start, end, deviceId from {t}").fetchall() if t in tables else []
        out[f"{key}_sql_ms_per_step_gpu"] = sum(e - s for s, e, _ in rows) / 1e6 / per
        for s, e, d in rows:
            iv[d].append((s, e))
    busy = 0
    for rows in iv.values():
        rows.sort()
        cs = ce = None
        for s, e in rows:
            if ce is None or s > ce:
                busy += 0 if ce is None else ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        busy += 0 if ce is None else ce - cs
    out["busy_ms_per_step_gpu"] = busy / 1e6 / per
    return out


# ------------------------------------------------------------------------------------------------
# the production run, instrumented
# ------------------------------------------------------------------------------------------------
_RFC_EARLY = None      # under JDIST: run_from_config loaded before any jax call (its import initialises jax.distributed)


def _rfc_module():
    spec = importlib.util.spec_from_file_location("run_from_config_cli",
                                                  ROOT / "scripts" / "run_from_config.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # @dataclass resolves string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def run_production(rfc_args, total_steps, chunk_steps, timer=None):
    """Run ``scripts/run_from_config.py`` main() with ``rfc_args`` for ``total_steps`` in
    ``chunk_steps`` chunks; returns its RunResult. ``timer`` (a ChunkTimer) wraps the chunk call."""
    for flag in ("--steps", "--chunk-steps", "--restart-out", "--restart-archive-out",
                 "--checkpoint-every", "--daily-out", "--monthly-out"):
        if any(a == flag or a.startswith(flag + "=") for a in rfc_args):
            raise SystemExit(f"gpu_profile: {flag} is set by the tool or refused, not passed through")
    import fesom_jax.run as R
    mod = _RFC_EARLY or _rfc_module()
    inner, got = mod.run_from_config, {}

    def instrumented(cfg, **kw):
        check_no_output(kw)
        if cfg.restart_out:
            check_no_output({"out_dir": cfg.restart_out})
        orig = R.run_steps_sharded_forced
        if timer is not None:
            R.run_steps_sharded_forced = timer.wrap(orig)
        try:
            got["res"] = inner(cfg, **kw)
        finally:
            R.run_steps_sharded_forced = orig
        return got["res"]

    mod.run_from_config = instrumented
    argv = sys.argv
    sys.argv = ["run_from_config.py", *rfc_args, "--steps", str(total_steps),
                "--chunk-steps", str(chunk_steps)]
    try:
        mod.main()
    finally:
        sys.argv = argv
    return got["res"]


def _state_fields(state_p) -> dict:
    import jax
    import numpy as np
    return {f.name: np.asarray(jax.device_get(getattr(state_p, f.name)))
            for f in dataclasses.fields(state_p) if getattr(state_p, f.name) is not None}


def _finite(state_p):
    from fesom_jax.diagnostics import state_diagnostics, verdict
    d = state_diagnostics(state_p)
    return verdict(d)[0], d


def _peaks():
    import jax
    return [round((d.memory_stats() or {}).get("peak_bytes_in_use", 0) / 1024 ** 3, 2)
            for d in jax.local_devices()]


def _cudart():
    import nvidia.cuda_runtime
    lib = glob.glob(os.path.join(list(nvidia.cuda_runtime.__path__)[0], "lib", "libcudart.so*"))
    return ctypes.CDLL(sorted(lib)[0])


def _git_head(root) -> str:
    """The checkout's HEAD commit; without git on PATH (dolpung's aarch64 env) read it from the
    worktree's ``.git`` file / directory."""
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except FileNotFoundError:
        gd = Path(root) / ".git"
        if gd.is_file():
            gd = Path(gd.read_text().split("gitdir:", 1)[1].strip())
        return (gd / "HEAD").read_text().strip()     # a detached snapshot: the sha itself


def _provenance(a):
    import jax
    head = _git_head(ROOT)
    return {"commit": head, "rfc_args": a.rfc, "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "devices": [str(d) for d in jax.devices()], "job": os.environ.get("SLURM_JOB_ID")}


def cmd_time(a, profile):
    import jax
    rt = _cudart() if profile else None
    on_start = (lambda: rt.cudaProfilerStart()) if rt is not None else None
    timer = ChunkTimer(a.warm_chunks, sync=jax.block_until_ready, on_start=on_start, state_index=1)
    res = run_production(a.rfc, a.warm_chunks * a.chunk_steps + a.steps, a.chunk_steps, timer)
    jax.block_until_ready(res.state_p)
    t_end = timer.clock()
    if rt is not None:
        rt.cudaProfilerStop()
    rec = timer.summary(t_end, a.steps)
    ok, diags = _finite(res.state_p)
    rec.update({"what": "prof" if profile else "time", "dt": res.dt_stage,
                "steady_sypd": sypd(rec["steady_ms_per_step"], res.dt_stage),
                "chunk_steps": a.chunk_steps, "warm_chunks": a.warm_chunks, "steps": a.steps,
                "peak_GB": _peaks(), "finite": ok,
                "sst_range": [diags["T_min"], diags["T_max"]], **_provenance(a)})
    say(f"steady {rec['steady_ms_per_step']:.2f} ms/step = {rec['steady_sypd']:.1f} SYPD at dt "
        f"{res.dt_stage:g}; host between chunks {rec['host_gap_ms_per_step']:.2f} ms/step, inside "
        f"the calls {rec['call_ms_per_step']:.2f}; peak {rec['peak_GB']} GB; finite {ok}")
    return rec


def cmd_noise(a):
    out = Path(a.out)
    states = []
    for i in range(a.reps):
        res = run_production(a.rfc, a.steps, a.chunk_steps)
        ok, _ = _finite(res.state_p)
        if not ok:
            raise SystemExit(f"gpu_profile noise: run {i} is not finite")
        st = _state_fields(res.state_p)
        save_state(out / f"state_{i}.npz", st)
        states.append(st)
        say(f"run {i + 1}/{a.reps} saved")
    rec = {"steps": a.steps, "reps": a.reps, **_provenance(a)}
    if len(states) >= 2:
        rec["floor"] = pooled_floor(states)
        for f, v in sorted(rec["floor"].items(), key=lambda kv: -kv[1]):
            say(f"  floor {f:14s} {v:.3e}")
    return rec


def cmd_compare(a):
    floor = json.loads(Path(a.floor).read_text())["floor"]
    ok, rows = judge(field_diffs(load_state(a.a), load_state(a.b)), floor, a.factor)
    for r in sorted(rows, key=lambda r: -r["ratio"]):
        print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['field']:14s} rel {r['max_rel']:.3e}  "
              f"floor {r['floor']:.3e}  ratio {r['ratio']:.2f}")
    print("WITHIN_NOISE_FLOOR" if ok else "OUTSIDE_NOISE_FLOOR")
    return ok


def cmd_stats(a):
    out = Path(a.out)
    rec = nsys_summary(a.prefix, a.steps, a.devices)
    rec.update(busy_from_sqlite(f"{a.prefix}.sqlite", a.steps, a.devices))
    for name in ("time", "prof"):
        p = out / f"{name}.json"
        if p.exists():
            rec[name] = json.loads(p.read_text())
    (out / "profile.json").write_text(json.dumps(rec, indent=1))
    tm = rec.get("time", {})
    print(f"P={a.devices}: steady {tm.get('steady_ms_per_step', float('nan')):.2f} ms/step, kernels "
          f"{rec['kernel_ms_per_step_gpu']:.2f} ms (NCCL {rec['nccl_ms_per_step_gpu']:.2f}), launches "
          f"{rec['launches_per_step_gpu']:.0f} + graphs {rec['graph_launches_per_step_gpu']:.0f}, D2D "
          f"{rec['d2d_copies_per_step_gpu']:.0f}, syncs {rec['syncs_per_step_gpu']:.0f} per step per "
          f"GPU; busy {rec.get('busy_ms_per_step_gpu', float('nan')):.2f} ms/step", flush=True)


def main():
    argv = sys.argv[1:]
    rfc = []
    if "--" in argv:
        i = argv.index("--")
        argv, rfc = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("what", choices=("time", "prof", "noise", "compare", "stats"))
    p.add_argument("out", help="output directory (compare: the first .npz)")
    p.add_argument("prefix", nargs="?", help="stats: the nsys stats prefix; compare: the second .npz")
    p.add_argument("--chunk-steps", type=int, default=96)
    p.add_argument("--warm-chunks", type=int, default=2)
    p.add_argument("--steps", type=int, default=None, help="timed steps (time/prof) or run length (noise)")
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--floor", help="compare: noise.json")
    p.add_argument("--factor", type=float, default=1.0, help="compare: allowed multiple of the floor")
    a = p.parse_args(argv)
    a.rfc = rfc
    if a.what == "compare":
        a.a, a.b = a.out, a.prefix
        sys.exit(0 if cmd_compare(a) else 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.what == "stats":
        a.steps = a.steps or 96
        return cmd_stats(a)
    if not rfc:
        p.error("pass the run_from_config arguments after --")
    os.environ["FESOM_REUSE_EXE"] = "1"            # read at fesom_jax.run import time
    if os.environ.get("JDIST"):
        if a.what != "time":
            raise SystemExit("gpu_profile: only `time` runs on several processes (JDIST)")
        global _RFC_EARLY
        _RFC_EARLY = _rfc_module()
    import jax
    if jax.process_count() != 1 and not os.environ.get("JDIST"):
        raise SystemExit("gpu_profile: single process only")
    import fesom_jax.run as R
    assert R._REUSE_EXE, "FESOM_REUSE_EXE must be on before fesom_jax.run is imported"
    if a.warm_chunks < 2:
        p.error("--warm-chunks >= 2 (run.py compiles a cold-input and a folded-input executable)")
    if a.what == "noise":
        a.steps = a.steps or 96
        rec = cmd_noise(a)
        (out / "noise.json").write_text(json.dumps(rec, indent=1))
        return
    a.steps = a.steps or (4 * a.chunk_steps if a.what == "time" else a.chunk_steps)
    if a.steps % a.chunk_steps:
        p.error("--steps must be a multiple of --chunk-steps")
    rec = cmd_time(a, profile=a.what == "prof")
    suffix = f".p{jax.process_index()}" if jax.process_count() > 1 else ""
    (out / f"{a.what}{suffix}.json").write_text(json.dumps(rec, indent=1))
    sys.exit(0 if rec["finite"] else 1)


if __name__ == "__main__":
    main()
