"""CORE2 adjoint-window de-risking sweep — Paper-experiments Task A7 (review MAJOR-2).

THE load-bearing autonomous result: measure, on a single A100, how the backward pass of the
**fast** calibration target ``d(mean MLD)/d(tke_c_k)`` behaves as the adjoint window ``N`` grows —
both (a) **peak backward memory** and (b) **FD↔AD agreement** (the "is the gradient real / not yet
chaotic" probe). Together these set **N_max** (the largest window with a clean gradient fitting an
80 GB A100) and the **adjoint↔EKI boundary**: fast targets (MLD/SST) use the adjoint up to N_max;
slow targets (GM→T/S stratification) must use EKI (:mod:`fesom_jax.eki`). It also **reconciles the
inherited 37.8 GB-at-N=20 figure** (that was GM; this measures the TKE→MLD config) with a measured
curve.

Config: **zstar + TKE** (the paper's live-geometry mixing path), **GM / KPP / ice OFF** — to
isolate the ``c_k → Kv → T/S → ρ → MLD`` signal and to **lower-bound** memory. The all-on config
(adding GM's streamfunction TDMA + mEVP) only ADDS memory, so the all-on adjoint window is
**≤ N_max measured here** (stated honestly, not assumed away).

ONE ``N`` per process (peak memory is a process-global high-water mark — a fresh process per N is
the only clean per-N measurement). The ``.sbatch`` loops N = 4, 20, 50, 100, 200; each run appends
a JSON line to the results file; :mod:`scripts.fig_window_snr` reads them, finds N_max, and emits
``WINDOW_DERISK_OK``. Single-GPU only (the sharded ragged-halo AD bug — see docs/JAX_RAGGED_A2A_BUG).

Usage (GPU):  python scripts/archive/core2_adjoint_window_sweep.py --n 20 --results <file.jsonl>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import fesom_jax  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import ale, surface_forcing, obs_compare, ssh
from fesom_jax.ale import AleConfig
from fesom_jax.integrate import integrate
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
DT = 1800.0
H_SWEEP = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
TKE_CFG = TkeConfig()
ALE_CFG = AleConfig()
DEFAULT_RESULTS = ROOT / "scripts" / "adjoint_window_sweep_results.jsonl"


def gpu_limit_gb():
    """Actual GPU memory limit (GB) from the device, so the fits decision adapts to whatever
    card the job landed on (Levante has both a100_40 and a100_80 — the .sbatch requests _80)."""
    for d in jax.devices():
        try:
            lim = (d.memory_stats() or {}).get("bytes_limit")
            if lim:
                return lim / 1e9
        except Exception:
            pass
    return 80.0


def build(year, n_max):
    mesh = load_mesh(MESH_DIR)
    state = phc_initial_state(mesh, IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, year, sst_ic=np.asarray(state.T[:, 0]))
    sfs = cf.stack(surface_forcing.dates_for_steps(year, DT, n_max))
    return mesh, state, op, cf.static, sfs


def peak_gb():
    pk = 0.0
    for d in jax.devices():
        try:
            st = d.memory_stats() or {}
            pk = max(pk, (st.get("peak_bytes_in_use") or 0) / 1e9)
        except Exception:
            pass
    return pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="adjoint window length (steps)")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = ap.parse_args()
    n = args.n

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh, st0, op, fs, sfs_max = build(args.year, n)
    sfs = jax.tree.map(lambda x: x[:n], sfs_max)
    node_mask = mesh.node_layer_mask
    nwet = float(jnp.sum(node_mask[:, 0]))
    print(f"[setup] built in {time.time()-t0:.1f}s; N={n} ({n*DT/86400:.3f} days), "
          f"zstar+TKE (GM/KPP/ice off)", flush=True)

    def mean_mld(c_k):
        """The fast calibration target: mean mixed-layer depth [m] after an N-step window, as a
        function of the TKE constant c_k. c_k → Kv → T/S → ρ → MLD (all differentiable)."""
        p = dataclasses.replace(Params.defaults(), tke_c_k=c_k)
        fin = integrate(st0, mesh, op, None, n_steps=n, dt=DT, step_forcings=sfs,
                        forcing_static=fs, tke_cfg=TKE_CFG, ale_cfg=ALE_CFG, params=p)
        _, Z3d = ale.live_geometry(mesh, fin.hnode)
        mld, valid = obs_compare.mld_density_threshold(fin.T, fin.S, Z3d, node_mask)
        return jnp.sum(jnp.where(valid, mld, 0.0)) / jnp.sum(valid)

    ck0 = jnp.asarray(float(Params.defaults().tke_c_k), jnp.float64)   # 0.1
    gpu_gb = gpu_limit_gb()
    print(f"[setup] GPU memory limit = {gpu_gb:.0f} GB", flush=True)

    rec = {"N": n, "days": n * DT / 86400.0, "dt": DT, "config": "zstar+TKE", "gpu_gb": gpu_gb}
    try:
        loss_jit = jax.jit(mean_mld)
        grad_jit = jax.jit(jax.grad(mean_mld))
        l0 = float(loss_jit(ck0))
        t1 = time.time()
        ad = float(grad_jit(ck0))
        bwd_s = time.time() - t1
        pk = peak_gb()
        # FD agreement (the "is the gradient real" probe): smallest relative AD-FD gap over h
        rows = []
        for h in H_SWEEP:
            ckp, ckm = ck0 * (1.0 + h), ck0 * (1.0 - h)
            fd = float((loss_jit(ckp) - loss_jit(ckm)) / (ckp - ckm))
            rows.append((h, fd))
        plateau = min(abs(ad - fd) / max(abs(fd), 1e-300) for _, fd in rows)
        fd_vals = np.array([fd for _, fd in rows])
        fd_spread = float(np.std(fd_vals))
        snr = abs(ad) / max(fd_spread, 1e-300)             # gradient signal vs FD noise floor
        clean = bool(np.isfinite(ad) and ad != 0.0 and plateau < 1e-2)
        fits = bool(pk < gpu_gb)
        rec.update(dict(loss=l0, grad=ad, peak_gb=pk, bwd_s=bwd_s, plateau=plateau,
                        fd_spread=fd_spread, snr=snr, clean=clean, fits_gpu=fits,
                        oom=False, ok=clean and fits))
        print(f"\n  N={n} ({rec['days']:.2f} d): mean MLD={l0:.3f} m  "
              f"d(MLD)/d(c_k) AD={ad:+.6e}  peak={pk:.2f}/{gpu_gb:.0f} GB  bwd={bwd_s:.1f}s", flush=True)
        for h, fd in rows:
            print(f"      h={h:.0e}  FD={fd:+.6e}  rel={abs(ad-fd)/max(abs(fd),1e-300):.2e}")
        print(f"      plateau(min rel)={plateau:.3e}  SNR={snr:.2e}  clean={clean}  "
              f"fits_GPU={fits}", flush=True)
    except Exception as e:                                  # OOM or other runtime failure
        msg = type(e).__name__ + ": " + str(e)[:200]
        rec.update(dict(loss=None, grad=None, peak_gb=peak_gb(), oom=("RESOURCE" in str(e).upper()
                        or "memory" in str(e).lower() or "OOM" in str(e).upper()),
                        clean=False, fits_gpu=False, ok=False, error=msg))
        print(f"\n  N={n}: FAILED ({msg})", flush=True)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  appended result for N={n} → {args.results}", flush=True)
    print(f"WINDOW_SWEEP_N{n}_{'OK' if rec.get('ok') else 'FAIL'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
