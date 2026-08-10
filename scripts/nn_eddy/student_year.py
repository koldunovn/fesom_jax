"""NN-GM revised Stage-1 feasibility — the student's OWN year vs the teacher (no teacher-init).

Design under test (plan Task 9 revision, 2026-08-08): the student is NEVER initialised from
coarse-grained teacher fields. Teacher and student share only the true common start — the PHC
cold start 1958-01-01 under JRA55-do, the same transient the NG5 teacher runs — and afterwards
the teacher enters through the LOSS alone (coarse-grained snapshots as observations). The two
feasibility numbers this produces:

  * D(t) — how far the free-running default-GM student drifts from the coarse-grained teacher
    over the shared year, in the exact Stage-1 loss metric (``signal_eval.py`` computes it);
  * S(t) — how much a 10-day window endpoint moves across the GM κ multiplier's own training
    bounds (×1/3 … ×3): the ENTIRE signal available to the optimiser at this window length,
    measured from the student's own developed states so no initialisation shock contaminates
    it (the measured 2026-08-08 blocker on teacher-init: 51k holes + unbalanced u,v/SSH).

``--year``: 365 d forward from the cold start; (T, S, eta) pulled at every 5-day mark (the
teacher snapshot grid; teacher records sit one fine-dt short of the round mark — a 2–4 min
offset, negligible for 5-day large-scale fields) and the full State pickled at the four branch
days. ``--branches``: from each pickled State, 10 d at m ∈ {1/3, 1, 3}, ``k_gm`` AND
``redi_kmax`` scaled together exactly as the gm_nn multiplier bounds do. The m=1 branch
re-runs the year's own segment: its mismatch vs the year marks is the GPU-nondeterminism floor
of 10-day trajectory spread — the natural zero of the signal scale.

Forward-only, one A100. The model pieces (config, step body, forcing handling) mirror
``horizon_scan.py`` — the driver HORIZON_OK validated — with the cold start from
``cold_start_state`` (PHC + seeded ice, the canonical entry). This file adds bookkeeping only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# ⚠️ MUST precede `import fesom_jax`. `python scripts/foo.py` puts the SCRIPT's directory on
# sys.path and NEVER the cwd, so a bare import resolves through the editable install — the MAIN
# checkout, not this worktree. Harmless while the two agree; silently wrong the moment they do
# not, which is now (`gm_nn.py` and the `gm.py` seam are worktree-only). Job 26798423 died on it.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fesom_jax  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np

_got = Path(fesom_jax.__file__).resolve()
if ROOT not in _got.parents:
    raise SystemExit(f"FATAL: fesom_jax imported from {_got}, not from {ROOT}")

from fesom_jax import ssh, surface_forcing
from fesom_jax.adjoint import scan_steps
from fesom_jax.ale import AleConfig
from fesom_jax.calibrate import build_params
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import cold_start_state
from fesom_jax.step import step
from fesom_jax.tke import TkeConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_data as _nd  # noqa: E402

MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist864"
DT = 1800.0
STEPS_PER_DAY = 48
MARK_DAYS = 5
MARK_STEPS = MARK_DAYS * STEPS_PER_DAY          # 240
NMARKS = 73                                     # days 5..365
BRANCH_DAYS = (60, 150, 240, 330)               # Mar/May/Aug/Nov — the seasonal spread
BRANCH_LEN_DAYS = 10
MULTS = (1.0 / 3.0, 1.0, 3.0)                   # the gm_nn bounds and their centre
# --branches-split: vary ONE coefficient at a time (k_gm alone, redi_kmax alone) — the
# gm_nn head is two independent bounded multipliers (N_OUT=2), so each axis's leverage
# on the loss is a separate trainable signal worth knowing before training (user ask
# 2026-08-08: indications exist that separating GM and Redi helps).
SPLITS = ((3.0, 1.0), (1.0 / 3.0, 1.0), (1.0, 3.0), (1.0, 1.0 / 3.0))
OUT_DEFAULT = _nd.STUDENT_DIR

ALE_CFG = AleConfig()
TKE_CFG = TkeConfig()
GM_CFG = GMConfig()
ICE_CFG = IceConfig(whichEVP=1)                 # forward run: adjoint_mode irrelevant


def mult_tag(m: float) -> str:
    """0.333.. -> 'm0333', 1.0 -> 'm1000', 3.0 -> 'm3000' (stable file tags)."""
    return f"m{round(m * 1000):04d}"


def branch_start_date(year: int, day: int) -> tuple[int, int]:
    """(month, day-of-month) of the wall clock after ``day`` full days from Jan 1."""
    d = datetime.date(year, 1, 1) + datetime.timedelta(days=day)
    return d.month, d.day


def snap_path(out: Path, day: int) -> Path:
    return out / f"snap_day{day:03d}.pkl"


def split_path(out: Path, day: int, m_gm: float, m_redi: float) -> Path:
    return out / f"branch2_d{day:03d}_g{round(m_gm*1000):04d}_r{round(m_redi*1000):04d}.npz"


def _atomic_savez(path: Path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)


def build_model():
    """Mesh + operator + forcing + the two jitted 5-day chunk functions (m is a traced arg)."""
    t0 = time.time()
    mesh = load_mesh(MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    st0 = jax.device_put(cold_start_state(mesh, IC_DIR))
    cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=np.asarray(st0.T[:, 0]))
    fs = cf.static
    kgm0 = float(Params.defaults().k_gm)
    kredi0 = float(Params.defaults().redi_kmax)

    def params_for(m_gm, m_redi):
        # two independent Params leaves (equal defaults, synced only by convention);
        # the coupled experiment passes (m, m), the split one moves a single axis
        return build_params({"k_gm": kgm0 * m_gm, "redi_kmax": kredi0 * m_redi})

    def body_for(p):
        def one_step(s, sf):
            return step(s, mesh, op, None, p, dt=DT, is_first_step=False, step_forcing=sf,
                        forcing_static=fs, ice_cfg=ICE_CFG, gm_cfg=GM_CFG, tke_cfg=TKE_CFG,
                        ale_cfg=ALE_CFG, remat_blocks=False)
        return one_step

    @jax.jit
    def chunk(state0, sfs, m_gm, m_redi):        # MARK_STEPS steps, developed state
        # rematerialize=False: this is a FORWARD-only run. jax.checkpoint buys reverse-pass
        # memory, and there is no reverse pass here, so it would be pure compile-time overhead.
        return scan_steps(body_for(params_for(m_gm, m_redi)), state0, sfs, rematerialize=False)

    @jax.jit
    def first_chunk(state0, sfs, m_gm, m_redi):  # cold start: step 1 eager AB2 bootstrap
        p = params_for(m_gm, m_redi)
        sf0 = jax.tree.map(lambda x: x[0], sfs)
        st1 = step(state0, mesh, op, None, p, dt=DT, is_first_step=True, step_forcing=sf0,
                   forcing_static=fs, ice_cfg=ICE_CFG, gm_cfg=GM_CFG, tke_cfg=TKE_CFG,
                   ale_cfg=ALE_CFG, remat_blocks=False)
        rest = jax.tree.map(lambda x: x[1:], sfs)
        return scan_steps(body_for(p), st1, rest, rematerialize=False)

    print(f"[setup] backend={jax.default_backend()} built in {time.time()-t0:.1f}s "
          f"kgm0={kgm0:g} config=zstar+TKE+mEVP(whichEVP=1)+GM coldstart=PHC+ice", flush=True)
    return mesh, st0, cf, chunk, first_chunk, kgm0


def _pull_and_check(st, day):
    T, S, eta = jax.device_get((st.T, st.S, st.eta_n))
    if not (np.isfinite(T).all() and np.isfinite(S).all() and np.isfinite(eta).all()):
        raise FloatingPointError(f"NONFINITE state at day {day}")
    if not (-4.0 < T.min() and T.max() < 45.0):
        raise FloatingPointError(f"unphysical T at day {day}: [{T.min():.2f}, {T.max():.2f}]")
    return T.astype(np.float32), S.astype(np.float32), eta.astype(np.float32)


def _free(tree):
    jax.tree.map(lambda x: x.delete() if hasattr(x, "delete") else None, tree)


def run_year(out: Path):
    marks_file = out / "marks_year.npz"
    if marks_file.exists():
        raise SystemExit(f"REFUSING to overwrite {marks_file} — move it away first")
    out.mkdir(parents=True, exist_ok=True)
    mesh, st, cf, chunk, first_chunk, kgm0 = build_model()
    n_all = NMARKS * MARK_STEPS
    dates_all = surface_forcing.dates_for_steps(1958, DT, n_all, start_month=1, start_day=1)
    nl = int(st.T.shape[1])
    T_m = np.empty((NMARKS, st.T.shape[0], nl), np.float32)
    S_m = np.empty_like(T_m)
    eta_m = np.empty((NMARKS, st.T.shape[0]), np.float32)
    m1 = jnp.asarray(1.0, jnp.float64)
    t0 = time.time()
    for k in range(1, NMARKS + 1):
        sfs = cf.stack(dates_all[(k - 1) * MARK_STEPS : k * MARK_STEPS])
        st = (first_chunk if k == 1 else chunk)(st, sfs, m1, m1)
        _free(sfs)
        day = MARK_DAYS * k
        T_m[k - 1], S_m[k - 1], eta_m[k - 1] = _pull_and_check(st, day)
        print(f"[year] day {day:3d}  wall {time.time()-t0:7.1f}s  "
              f"SST [{T_m[k-1][:,0].min():6.2f},{T_m[k-1][:,0].max():6.2f}] °C  "
              f"eta [{eta_m[k-1].min():6.2f},{eta_m[k-1].max():6.2f}] m", flush=True)
        if day in BRANCH_DAYS:
            p = snap_path(out, day)
            with open(p.with_suffix(".tmp"), "wb") as f:
                pickle.dump(jax.device_get(st), f)
            p.with_suffix(".tmp").rename(p)
            print(f"[year] pickled branch state -> {p.name}", flush=True)
    _atomic_savez(marks_file, T=T_m, S=S_m, eta=eta_m,
                  day=np.arange(MARK_DAYS, MARK_DAYS * (NMARKS + 1), MARK_DAYS))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    (out / "meta.json").write_text(json.dumps(dict(
        kgm0=kgm0, dt=DT, start_year=1958, mark_days=MARK_DAYS, nmarks=NMARKS,
        branch_days=BRANCH_DAYS, mults=MULTS, ic="cold_start_state(PHC+ice)",
        config="zstar+TKE+mEVP(whichEVP=1)+GM", commit=commit,
        created=datetime.datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"[year] DONE 365 d in {(time.time()-t0)/60:.1f} min -> {marks_file}", flush=True)


def run_branches(out: Path):
    mesh, _st0, cf, chunk, _first, _kgm0 = build_model()
    t0 = time.time()
    for day0 in BRANCH_DAYS:
        with open(snap_path(out, day0), "rb") as f:
            base = jax.device_put(pickle.load(f))
        mo, dy = branch_start_date(1958, day0)
        dates = surface_forcing.dates_for_steps(1958, DT, BRANCH_LEN_DAYS * STEPS_PER_DAY,
                                                start_month=mo, start_day=dy)
        for m in MULTS:
            dest = out / f"branch_d{day0:03d}_{mult_tag(m)}.npz"
            if dest.exists():
                print(f"[branch] {dest.name} exists — skipped", flush=True)
                continue
            mj = jnp.asarray(m, jnp.float64)
            sfs = cf.stack(dates[:MARK_STEPS])
            st5 = chunk(base, sfs, mj, mj)
            _free(sfs)
            T5, S5, e5 = _pull_and_check(st5, day0 + 5)
            sfs = cf.stack(dates[MARK_STEPS:])
            st10 = chunk(st5, sfs, mj, mj)
            _free(sfs)
            T10, S10, e10 = _pull_and_check(st10, day0 + 10)
            _atomic_savez(dest, T5=T5, S5=S5, eta5=e5, T10=T10, S10=S10, eta10=e10,
                          mult=np.float64(m), day0=np.int32(day0))
            print(f"[branch] d{day0:03d} m={m:.4f}  wall {time.time()-t0:7.1f}s -> {dest.name}",
                  flush=True)
        _free(base)
    print(f"[branch] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


def run_branches_split(out: Path):
    """One coefficient at a time: (m_gm, m_redi) over SPLITS — the per-axis leverage."""
    mesh, _st0, cf, chunk, _first, _kgm0 = build_model()
    t0 = time.time()
    for day0 in BRANCH_DAYS:
        with open(snap_path(out, day0), "rb") as f:
            base = jax.device_put(pickle.load(f))
        mo, dy = branch_start_date(1958, day0)
        dates = surface_forcing.dates_for_steps(1958, DT, BRANCH_LEN_DAYS * STEPS_PER_DAY,
                                                start_month=mo, start_day=dy)
        for m_gm, m_redi in SPLITS:
            dest = split_path(out, day0, m_gm, m_redi)
            if dest.exists():
                print(f"[split] {dest.name} exists — skipped", flush=True)
                continue
            g = jnp.asarray(m_gm, jnp.float64)
            r = jnp.asarray(m_redi, jnp.float64)
            sfs = cf.stack(dates[:MARK_STEPS])
            st5 = chunk(base, sfs, g, r)
            _free(sfs)
            T5, S5, e5 = _pull_and_check(st5, day0 + 5)
            sfs = cf.stack(dates[MARK_STEPS:])
            st10 = chunk(st5, sfs, g, r)
            _free(sfs)
            T10, S10, e10 = _pull_and_check(st10, day0 + 10)
            _atomic_savez(dest, T5=T5, S5=S5, eta5=e5, T10=T10, S10=S10, eta10=e10,
                          mult_gm=np.float64(m_gm), mult_redi=np.float64(m_redi),
                          day0=np.int32(day0))
            print(f"[split] d{day0:03d} g={m_gm:.4f} r={m_redi:.4f}  "
                  f"wall {time.time()-t0:7.1f}s -> {dest.name}", flush=True)
        _free(base)
    print(f"[split] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--year", action="store_true")
    g.add_argument("--branches", action="store_true")
    g.add_argument("--branches-split", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    if args.year:
        run_year(args.out)
    elif args.branches:
        run_branches(args.out)
    else:
        run_branches_split(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
