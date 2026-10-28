"""The CORE2 GPU speed plan's measuring tools (``scripts/perf/``), on synthetic inputs.

The tools only ever run on a GPU node against a real CORE2 run; what can go wrong without a GPU is
their arithmetic and their parsing, and a wrong number there would steer the whole plan. Checked
here, with no JAX device work:

* ``gpu_profile``: SYPD at a given dt; the chunk timer's window (it drains the device once, at the
  first timed chunk, and times only what follows); the per-field noise floor pooled over repeats and
  the pass/fail judgement against it; the refusal to run a configuration that writes output; the
  nsys summary per step and GPU.
* ``hlo_kernel_map``: kernels -> fusions -> Python frames -> the model area, with the ``ops.py``
  helpers attributed to their caller, and helper kernels without a model frame placed by their
  launch count (CG iterations, mEVP subcycles).
* ``nccl_waits``: a collective's cost is its shortest duration across the GPUs; the rest is waiting.
"""

from __future__ import annotations

import csv
import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

PERF = Path(__file__).resolve().parents[1] / "perf"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PERF / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod             # @dataclass resolves string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


gp = _load("gpu_profile")
hk = _load("hlo_kernel_map")
nw = _load("nccl_waits")


# --------------------------------------------------------------------------------------------
# gpu_profile
# --------------------------------------------------------------------------------------------
def test_sypd():
    assert gp.sypd(43.7, 1800.0) == pytest.approx(112.85, abs=0.01)    # JUPITER 4 GH200 row
    assert gp.sypd(7.40, 2700.0) == pytest.approx(1000.0, rel=1e-3)    # fesom3-jax's 1000 SYPD mark
    assert gp.sypd(4.93, 1800.0) == pytest.approx(1000.0, rel=2e-3)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_chunk_timer_window():
    """4 chunks of 10 steps, the first 2 warm: the window opens at chunk 3 after one drain."""
    clock, synced, started = _Clock(), [], []
    timer = gp.ChunkTimer(warm_chunks=2, sync=synced.append, on_start=lambda: started.append(clock.t),
                          clock=clock)

    def chunk(state, n):                      # the wrapped chunk call: 1 s of host time inside it
        clock.t += 1.0
        return state + n

    wrapped = timer.wrap(chunk)
    state = 0
    for i in range(4):
        clock.t += 5.0                        # host work between chunk calls (forcing build)
        state = wrapped(state, 10)
    clock.t += 2.0                            # the final wait for the device
    rec = timer.summary(t_end=clock.t, timed_steps=20)

    assert synced == [20], "drains once, on the state entering the first timed chunk"
    assert started == [12.0 + 5.0], "the window opens after the host work of chunk 3"
    assert rec["window_s"] == pytest.approx(1.0 + 5.0 + 1.0 + 2.0)
    assert rec["steady_ms_per_step"] == pytest.approx(1e3 * 9.0 / 20)
    assert rec["calls"] == 4 and rec["timed_calls"] == 2
    assert rec["host_gap_ms_per_step"] == pytest.approx(1e3 * 5.0 / 10)   # one gap inside the window
    assert rec["call_ms_per_step"] == pytest.approx(1e3 * 2.0 / 20)


def test_chunk_timer_needs_enough_chunks():
    timer = gp.ChunkTimer(warm_chunks=3, sync=lambda s: None)
    wrapped = timer.wrap(lambda s, n: s)
    wrapped(0, 1)
    with pytest.raises(RuntimeError, match="never opened"):
        timer.summary(t_end=1.0, timed_steps=1)


def _states():
    rng = np.random.default_rng(0)
    base = {"T": rng.normal(10.0, 3.0, (50, 4)), "eta_n": rng.normal(0.0, 0.5, 50),
            "a_ice": np.zeros(50)}
    out = []
    for amp in (0.0, 1e-9, 3e-9):
        s = {k: v.copy() for k, v in base.items()}
        s["T"][7, 2] += amp * 10.0
        s["eta_n"][3] += amp
        out.append(s)
    return out


def test_field_diffs_and_pooled_floor():
    a, b, c = _states()
    d = gp.field_diffs(a, b)
    assert d["T"]["n_diff"] == 1 and d["a_ice"]["n_diff"] == 0
    assert d["T"]["max_abs"] == pytest.approx(1e-8, rel=1e-6)
    assert d["T"]["max_rel"] == pytest.approx(1e-8 / np.max(np.abs(a["T"])), rel=1e-6)
    floor = gp.pooled_floor([a, b, c])
    # the largest pair (a, c) sets each field's floor
    assert floor["T"] == pytest.approx(gp.field_diffs(a, c)["T"]["max_rel"])
    assert floor["eta_n"] == pytest.approx(gp.field_diffs(a, c)["eta_n"]["max_rel"])
    assert floor["a_ice"] == 0.0
    with pytest.raises(ValueError):
        gp.pooled_floor([a])


def test_judge_against_floor():
    a, b, c = _states()
    floor = gp.pooled_floor([a, b, c])
    ok, rows = gp.judge(gp.field_diffs(a, b), floor)
    assert ok and all(r["ok"] for r in rows)
    worse = {k: v.copy() for k, v in a.items()}
    worse["eta_n"][3] += 1e-6                       # far outside the floor
    ok, rows = gp.judge(gp.field_diffs(a, worse), floor)
    assert not ok and [r["field"] for r in rows if not r["ok"]] == ["eta_n"]
    moved = {k: v.copy() for k, v in a.items()}
    moved["a_ice"][0] = 1e-12                       # a field whose floor is exactly 0
    ok, rows = gp.judge(gp.field_diffs(a, moved), floor)
    assert not ok and [r["field"] for r in rows if not r["ok"]] == ["a_ice"]


def test_state_npz_round_trip(tmp_path):
    a = _states()[0]
    gp.save_state(tmp_path / "s.npz", a)
    back = gp.load_state(tmp_path / "s.npz")
    assert set(back) == set(a) and all(np.array_equal(back[k], a[k]) for k in a)


@pytest.mark.parametrize("kw", [{"out_dir": "/x"}, {"checkpoint_every": 48},
                                {"restart_archive_out": "/x"}, {"daily_out": "/x"},
                                {"monthly_out": "/x"}])
def test_refuses_runs_that_write(kw):
    with pytest.raises(SystemExit, match="writes output"):
        gp.check_no_output(dict(out_dir=None, checkpoint_every=None, restart_archive_out=None,
                                daily_out=None, monthly_out=None) | kw)
    gp.check_no_output(dict(out_dir=None, checkpoint_every=None, restart_archive_out=None,
                            daily_out=None, monthly_out=None))


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_nsys_stats_per_step_and_gpu(tmp_path):
    pre = tmp_path / "prof"
    _write_csv(f"{pre}_cuda_api_sum.csv", ["Name", "Num Calls", "Total Time (ns)"],
               [["cuLaunchKernel", 8000, 4e6], ["cuGraphLaunch", 400, 1e6],
                ["cuStreamSynchronize", 40, 2e6]])
    _write_csv(f"{pre}_cuda_gpu_kern_sum.csv", ["Name", "Total Time (ns)", "Time (%)", "Instances"],
               [["loop_fusion_1", 6e8, 75.0, 4000], ["ncclDevKernel_SendRecv", 2e8, 25.0, 800]])
    _write_csv(f"{pre}_cuda_gpu_mem_time_sum.csv", ["Operation", "Count"],
               [["[CUDA memcpy Device-to-Device]", 160], ["[CUDA memcpy Host-to-Device]", 9]])
    rec = gp.nsys_summary(str(pre), steps=10, devices=4)
    assert rec["launches_per_step_gpu"] == pytest.approx(200.0)
    assert rec["graph_launches_per_step_gpu"] == pytest.approx(10.0)
    assert rec["syncs_per_step_gpu"] == pytest.approx(1.0)
    assert rec["kernel_ms_per_step_gpu"] == pytest.approx(20.0)
    assert rec["nccl_ms_per_step_gpu"] == pytest.approx(5.0)
    assert rec["d2d_copies_per_step_gpu"] == pytest.approx(4.0)


def test_busy_is_per_gpu(tmp_path):
    """Busy time is each GPU's own union of intervals, averaged over GPUs — not one union over
    all GPUs (which shrinks with the device count when the GPUs overlap in time)."""
    db = tmp_path / "p.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, deviceId int)")
    # 2 GPUs, 1 step: each busy 0-100 and 150-200 ns (150 ns), at the same times
    con.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?)",
                    [(0, 60, 0), (40, 100, 0), (150, 200, 0), (0, 100, 1), (150, 200, 1)])
    con.commit()
    con.close()
    rec = gp.busy_from_sqlite(str(db), steps=1, devices=2)
    assert rec["busy_ms_per_step_gpu"] == pytest.approx(150e-6)
    assert rec["kernel_sql_ms_per_step_gpu"] == pytest.approx((60 + 60 + 50 + 100 + 50) * 1e-6 / 2)


# --------------------------------------------------------------------------------------------
# hlo_kernel_map
# --------------------------------------------------------------------------------------------
_ROOT = "/work/x/fesom_jax_dev/wt_perf/fesom_jax"
HLO = f"""\
HloModule jit_chunk, entry_computation_layout={{()->()}}

ENTRY %main () -> f64[8] {{
  %loop_fusion.3 = f64[8]{{0}} fusion(f64[8]{{0}} %p0), kind=kLoop, calls=%fc.3, metadata={{op_name="a" stack_frame_id=2}}
  %input_scatter_fusion.7 = f64[8]{{0}} fusion(f64[8]{{0}} %p1), kind=kInput, calls=%fc.7, metadata={{op_name="b" stack_frame_id=4}}
  ROOT %loop_fusion.9 = f64[8]{{0}} fusion(f64[8]{{0}} %p2), kind=kLoop, calls=%fc.9, metadata={{op_name="c" stack_frame_id=5}}
}}

FileNames
1 "{_ROOT}/ops.py"
2 "{_ROOT}/tracer_diff.py"
3 "{_ROOT}/ice_mevp.py"
4 "/sw/site-packages/jax/_src/lax/lax.py"

FunctionNames
1 "tdma"
2 "impl_vert_diff"
3 "_mevp_subcycle"
4 "scatter"

FileLocations
1 {{file_name_id=1 function_name_id=1 line=150 end_line=150 column=1 end_column=2}}
2 {{file_name_id=2 function_name_id=2 line=127 end_line=127 column=1 end_column=2}}
3 {{file_name_id=3 function_name_id=3 line=200 end_line=200 column=1 end_column=2}}
4 {{file_name_id=4 function_name_id=4 line=9 end_line=9 column=1 end_column=2}}
5 {{file_name_id=1 function_name_id=4 line=40 end_line=40 column=1 end_column=2}}

StackFrames
1 {{file_location_id=2 parent_frame_id=0}}
2 {{file_location_id=1 parent_frame_id=1}}
3 {{file_location_id=3 parent_frame_id=0}}
4 {{file_location_id=4 parent_frame_id=3}}
5 {{file_location_id=5 parent_frame_id=0}}
"""


@pytest.fixture
def tables(tmp_path):
    p = tmp_path / "after_optimizations.txt"
    p.write_text(HLO)
    return hk.parse_hlo(p)


def test_parse_hlo(tables):
    assert set(tables.inst) == {"loop_fusion.3", "input_scatter_fusion.7", "loop_fusion.9"}
    assert hk.chain(tables, 2) == ["ops.py:150 tdma", "tracer_diff.py:127 impl_vert_diff"]
    assert hk.chain(tables, 4) == ["ice_mevp.py:200 _mevp_subcycle"], "non-fesom_jax frames dropped"


def test_area_of_files():
    assert hk.area_of("ssh.py") == "SSH (CG solve, rhs)"
    assert hk.area_of("ice_mevp.py") == "sea ice dynamics (mEVP)"
    assert hk.area_of("ice_thermo.py") == "sea ice thermo, advection, coupling"
    assert hk.area_of("cvmix_tke.py") == "TKE"
    assert hk.area_of("gm_redi.py") == "GM/Redi"
    assert hk.area_of("not_a_model_file.py") == "not_a_model_file.py"


def test_aggregate_by_area(tables):
    rows = [  # nsys sums over all GPUs and steps: 2 steps here
        {"Name": "loop_fusion_3", "Total Time (ns)": 4e6, "Instances": 20},
        {"Name": "input_scatter_fusion_7", "Total Time (ns)": 2e6, "Instances": 240},
        {"Name": "loop_fusion_9", "Total Time (ns)": 1e6, "Instances": 480},   # ops-only frame
        {"Name": "ncclDevKernel_SendRecv", "Total Time (ns)": 3e6, "Instances": 100},
    ]
    agg = hk.aggregate(rows, tables, steps=2, by="area", cg_iters=None, mevp_sub=120)
    assert agg["tracer vertical diffusion"][0] == pytest.approx(2.0)   # tdma credited to its caller
    assert agg["sea ice dynamics (mEVP)"][0] == pytest.approx(1.0 + 0.5)  # + the ops-only kernel at 240/step
    assert agg["sea ice dynamics (mEVP)"][1] == pytest.approx(120 + 240)
    assert agg["(NCCL)"][0] == pytest.approx(1.5)


# --------------------------------------------------------------------------------------------
# nccl_waits
# --------------------------------------------------------------------------------------------
def test_nccl_cost_vs_wait(tmp_path):
    db = tmp_path / "prof.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table StringIds (id integer, value text)")
    con.executemany("insert into StringIds values (?, ?)",
                    [(1, "loop_fusion_1"), (2, "ncclDevKernel_SendRecv")])
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, deviceId int, shortName int)")
    rows = []
    for step in range(2):                   # per step: a compute kernel, then one collective
        t = step * 1000
        rows += [(t, t + 100, 0, 1), (t + 100, t + 110, 0, 2),        # GPU 0 arrives last: 10 ns
                 (t, t + 80, 1, 1), (t + 80, t + 110, 1, 2)]          # GPU 1 waits 20 ns
    con.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    rec = nw.analyse(str(db), steps=2)
    assert rec["gpus"] == 2 and rec["collectives_per_step"] == 1
    assert rec["cost_ms_per_step"] == pytest.approx(10e-6)
    assert rec["wait_ms_per_step"] == pytest.approx(10e-6)            # (0 + 20) / 2 GPUs


def test_git_head_without_git(tmp_path, monkeypatch):
    """dolpung's aarch64 env has no git on PATH: the HEAD comes from the worktree's .git file."""
    gitdir = tmp_path / "gitdir"
    gitdir.mkdir()
    (gitdir / "HEAD").write_text("0123abcd\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {gitdir}\n")
    monkeypatch.setenv("PATH", str(tmp_path / "nobin"))
    assert gp._git_head(wt) == "0123abcd"
