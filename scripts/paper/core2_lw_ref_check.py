#!/usr/bin/env python
"""Long-window Task B1 — the 10-yr REFERENCE trajectory VERDICT (REF_10YR_OK).

Gates the ensemble-averaged adjoint (Part D) on the reference produced by
``core2_lw_reference.sbatch`` (restart from ``spinup5_state.pkl``, 10 yr from 1963;
outputs on ``$WORK/longwindow/ref10_{climate,snaps}`` + ``ref10_final.pkl``). D1
probes THIS trajectory with many short frozen-ice adjoint bursts, so the reference
must be (i) finite & physical the whole way and (ii) carry a well-defined climate
observable — the area-weighted-mean MLD D1 differentiates.

Reuses the A1 drift reducers verbatim (``core2_lw_foundation_check``: ``load_weights``,
``read_climate``, ``decel``, ``lin_slope_per_year``) so the physicality / drift bar is
the SAME falsifiable instrument, just over 10 yr instead of 3. The velocity cap is the
run's own 5.0 (the Somali-Current fix), not A1's old 3.0.

Bars (REF_10YR_OK):
  (a) every sampled snapshot finite; min wet hnode > 0; max|vel| < 5.0; |SSH| < 5.0;
      local SST/SSS physical (no supercooling / no salinity blow-up)
  (b) the decade is NOT running away: |lin_slope(T0700)| over 10 yr < 0.1 °C/yr AND the
      second-half 5-yr drift ≤ the first-half (continued deceleration past the 3-yr base)
  (c) the climate MLD is well-defined: the 10-yr-mean area-weighted MLD (de Boyer Montégut
      density-threshold, computed on the actual snapshot States via live geometry — the
      EXACT signflip / D1 code path) is finite, positive, and in a physical band (5-400 m)

Heavy reads from ``$WORK`` only; CPU is enough (MLD is a forward eval, no adjoint). Writes a
single small ``scripts/lw_ref_check.jsonl``.

Usage (CPU):
  JAX_PLATFORMS=cpu python scripts/paper/core2_lw_ref_check.py                    # full
  JAX_PLATFORMS=cpu python scripts/paper/core2_lw_ref_check.py --snap-stride 146  # ~5 snapshots (smoke)
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import time
from pathlib import Path

import fesom_jax  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import ale, obs_compare
from fesom_jax.mesh import load_mesh

from core2_lw_foundation_check import (  # reuse the A1 instrument verbatim
    MESH_DIR, SST_LO, SST_HI, SSS_LO, SSS_HI,
    decel, lin_slope_per_year, load_weights, read_climate,
)

WORK_DEFAULT = "/work/ab0995/a270088/port_jax/longwindow"
RESULTS = Path(__file__).resolve().parents[1] / "lw_ref_check.jsonl"

VEL_ABSMAX = 5.0          # the run's own cap (Somali-Current fix), not A1's 3.0
SSH_ABSMAX = 5.0
DTDT_BAR = 0.1            # °C/yr decade-slope runaway bar
MLD_LO, MLD_HI = 5.0, 400.0   # physical band for the global area-weighted-mean MLD


def first_second_half(annual: np.ndarray):
    """Deceleration over a decade: mean of years 1-5 vs 6-10, and the per-half drift slope."""
    a = np.asarray(annual, dtype=np.float64)
    if a.size < 4:
        return None
    h = a.size // 2
    early = a[:h]
    late = a[h:]
    d_early = float(early[-1] - early[0]) / max(1, early.size - 1)
    d_late = float(late[-1] - late[0]) / max(1, late.size - 1)
    return dict(first_half_mean=round(float(early.mean()), 5),
                second_half_mean=round(float(late.mean()), 5),
                drift_first_half=round(d_early, 5), drift_second_half=round(d_late, 5),
                decelerating=bool(abs(d_late) <= abs(d_early) + 1e-12))


def read_snapshots_mld(snap_dir: Path, final_pkl: Path, W, stride: int):
    """Per-snapshot physicality + the area-weighted-mean MLD (de Boyer Montégut density
    threshold on the actual State via live geometry — the EXACT signflip/D1 observable)."""
    mesh = W["mesh"]
    layer_mask = W["layer_mask"]
    surf = W["surf_mask"]
    node_mask = jnp.asarray(mesh.node_layer_mask)
    area0 = jnp.asarray(np.asarray(mesh.area[:, 0]))

    @jax.jit
    def mld_mean(T, S, hnode):
        _, Z3d = ale.live_geometry(mesh, hnode)
        mld, valid = obs_compare.mld_density_threshold(T, S, Z3d, node_mask)
        w = area0 * valid
        den = jnp.sum(w)
        return jnp.sum(w * mld) / jnp.where(den > 0, den, 1.0)

    files = []
    if stride and stride > 0:
        files = sorted(glob.glob(str(snap_dir / "snap_step*.pkl")))[:: stride]
    if final_pkl.exists():
        files = files + [str(final_pkl)]

    phys = []
    for f in files:
        with open(f, "rb") as fh:
            st = pickle.load(fh)
        hn = np.asarray(st.hnode)
        uv = np.asarray(st.uv)
        T0 = np.asarray(st.T)[:, 0]
        S0 = np.asarray(st.S)[:, 0]
        finite = bool(np.isfinite(hn).all() and np.isfinite(uv).all()
                      and np.isfinite(np.asarray(st.T)).all()
                      and np.isfinite(np.asarray(st.S)).all()
                      and np.isfinite(np.asarray(st.eta_n)).all())
        mld = float(mld_mean(jnp.asarray(st.T), jnp.asarray(st.S), jnp.asarray(st.hnode)))
        phys.append(dict(file=Path(f).name, finite=finite,
                         min_wet_hnode=float(np.min(hn[layer_mask])),
                         max_abs_uv=float(np.max(np.abs(uv))),
                         sst_min=float(np.min(np.where(surf, T0, np.inf))),
                         sst_max=float(np.max(np.where(surf, T0, -np.inf))),
                         sss_min=float(np.min(np.where(surf, S0, np.inf))),
                         sss_max=float(np.max(np.where(surf, S0, -np.inf))),
                         mean_mld=mld))
        del st, hn, uv, T0, S0
    return phys


def make_verdict(series, phys):
    t_decel = decel(series["t0700"])
    s_decel = decel(series["s0700"])
    t_ann = np.asarray(t_decel["annual"], dtype=np.float64)
    t_decade = first_second_half(t_ann)
    t_slope10 = lin_slope_per_year(series["t0700"])

    all_finite = all(p["finite"] for p in phys) if phys else False
    min_hnode = min((p["min_wet_hnode"] for p in phys), default=float("nan"))
    max_vel = max((p["max_abs_uv"] for p in phys), default=float("nan"))
    ssh_absmax = float(series["ssh_absmax"].max())
    sst_lo = min((p["sst_min"] for p in phys), default=float("nan"))
    sst_hi = max((p["sst_max"] for p in phys), default=float("nan"))
    sss_lo = min((p["sss_min"] for p in phys), default=float("nan"))
    sss_hi = max((p["sss_max"] for p in phys), default=float("nan"))

    physical = bool(all_finite and (min_hnode > 0.0) and (max_vel < VEL_ABSMAX)
                    and (ssh_absmax < SSH_ABSMAX)
                    and (SST_LO <= sst_lo) and (sst_hi <= SST_HI)
                    and (SSS_LO <= sss_lo) and (sss_hi <= SSS_HI))

    mlds = np.asarray([p["mean_mld"] for p in phys], dtype=np.float64)
    mld_finite = bool(np.isfinite(mlds).all()) if mlds.size else False
    mld_mean10 = float(np.mean(mlds)) if mlds.size else float("nan")
    mld_well_defined = bool(mld_finite and (MLD_LO <= mld_mean10 <= MLD_HI)
                            and bool((mlds > 0).all()))

    not_running_away = bool(abs(t_slope10) < DTDT_BAR
                            and (t_decade is not None) and t_decade["decelerating"])

    ok = bool(physical and not_running_away and mld_well_defined)

    return dict(
        REF_10YR_OK=ok,
        physical=physical, not_running_away=not_running_away, mld_well_defined=mld_well_defined,
        t0700_lin_slope_per_year=round(t_slope10, 5), dTdt_bar=DTDT_BAR,
        t0700_decade=t_decade,
        mld_10yr_mean=round(mld_mean10, 4), mld_per_snapshot=[round(float(x), 3) for x in mlds],
        mld_band=[MLD_LO, MLD_HI],
        checks=dict(all_finite=all_finite, min_wet_hnode=min_hnode, max_abs_uv=max_vel,
                    ssh_absmax=ssh_absmax, sst_range=[sst_lo, sst_hi], sss_range=[sss_lo, sss_hi]),
        t0700=t_decel, s0700=s_decel,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=str, default=WORK_DEFAULT)
    ap.add_argument("--start-year", type=int, default=1963)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--snap-stride", type=int, default=73,
                    help="read every Nth snapshot for physicality+MLD (730 snaps; 73 ⇒ ~10)")
    ap.add_argument("--results", type=str, default=str(RESULTS))
    args = ap.parse_args()

    work = Path(args.work)
    climate_dir = work / "ref10_climate"
    snap_dir = work / "ref10_snaps"
    final_pkl = work / "ref10_final.pkl"

    t0 = time.time()
    print(f"[B1] backend={jax.default_backend()}  mesh + static weights from {MESH_DIR}", flush=True)
    W = load_weights(MESH_DIR)

    print(f"[B1] reading monthly climate {climate_dir} ({args.years} yr from {args.start_year})", flush=True)
    series, months = read_climate(climate_dir, W, args.start_year, args.years)

    print(f"[B1] reading snapshots {snap_dir} (stride={args.snap_stride}) + {final_pkl.name} "
          f"→ physicality + MLD", flush=True)
    phys = read_snapshots_mld(snap_dir, final_pkl, W, args.snap_stride)

    verdict = make_verdict(series, phys)

    print("\n=== B1 10-YR REFERENCE VERDICT ===", flush=True)
    print(f"{'month':>8} {'SST':>7} {'SSS':>7} {'T0700':>8} {'S0700':>8} "
          f"{'iceA(e12)':>10} {'arcMice':>8}", flush=True)
    for i in range(0, len(months), 6):                       # every 6th month (semiannual) to keep it short
        print(f"{months[i]:>8} {series['sst'][i]:7.3f} {series['sss'][i]:7.3f} "
              f"{series['t0700'][i]:8.4f} {series['s0700'][i]:8.4f} "
              f"{series['ice_area'][i]/1e12:10.3f} {series['arctic_mice'][i]:8.4f}", flush=True)
    td = verdict["t0700_decade"]
    print(f"\n-- T0700 drift -- annual={verdict['t0700']['annual']}", flush=True)
    print(f"   lin_slope(10yr)={verdict['t0700_lin_slope_per_year']}/yr (<{DTDT_BAR}?)  "
          f"half-drift {td['drift_first_half']}→{td['drift_second_half']} decel={td['decelerating']}",
          flush=True)
    print(f"\n-- MLD (de Boyer Montégut, area-weighted) per snapshot --", flush=True)
    for p in phys:
        print(f"   {p['file']:>22}  MLD={p['mean_mld']:7.2f} m  |vel|={p['max_abs_uv']:.3f}  "
              f"finite={p['finite']}", flush=True)
    print(f"   10-yr-mean MLD = {verdict['mld_10yr_mean']} m (band {MLD_LO}-{MLD_HI})", flush=True)
    c = verdict["checks"]
    print(f"\n-- physicality -- finite={c['all_finite']} min_wet_hnode={c['min_wet_hnode']:.4g} "
          f"max|vel|={c['max_abs_uv']:.3f} |SSH|max={c['ssh_absmax']:.3f} "
          f"SST{c['sst_range']} SSS{c['sss_range']}", flush=True)
    print(f"\n  physical={verdict['physical']}  not_running_away={verdict['not_running_away']}  "
          f"mld_well_defined={verdict['mld_well_defined']}", flush=True)

    rec = dict(task="B1", t_sec=round(time.time() - t0, 1), years=args.years,
               start_year=args.start_year, snap_stride=args.snap_stride,
               n_snapshots=len(phys), months=months,
               series={k: [round(float(x), 6) for x in v] for k, v in series.items()},
               phys=phys, verdict=verdict)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n[B1] wrote {args.results}", flush=True)

    if verdict["REF_10YR_OK"]:
        print("\nREF_10YR_OK", flush=True)
        return 0
    print("\nREF_10YR_NOT_OK — unphysical, running away, or MLD ill-defined; document BEFORE D1", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
