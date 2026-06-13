#!/usr/bin/env python
"""CORE2 KPP multi-year climate run with MONTHLY-MEAN NetCDF output — Phase 6C follow-up.

Runs the full production assembled CORE2 model (KPP vertical mixing + GM/Redi +
prognostic sea ice) forward ``--years`` model years at ``--dt`` (default 1800 s, the
Fortran KPP reference timestep) and writes **true monthly-mean** fields per variable in
the **C-port output format** (``<var>.fesom.<yr>.monthly.nc``, 12 records/year), so:

  * the AWI ``ushow`` viewer reads them directly (lon/lat/z/time embedded, CF-1.8), and
  * ``port_kokkos/scripts/m32_climate_compare.py`` compares them VERBATIM against the
    C-port-KPP (`/work/.../port/kpp_5yr_fix`) + Fortran-KPP (`/scratch/.../fortran_kpp_5yr_fix`)
    references — the same annual-mean surface corr/bias/RMS the Kokkos port used
    (CUDA-vs-C-port-KPP sst RMS ~1.4e-2 °C, corr ~1.0 for 1958).

Monthly means (NOT instantaneous snapshots — the snapshot sampling noise ~0.1 °C would
swamp the ~1.4e-2 °C climate signal) are accumulated on-device and flushed at each
calendar-month boundary. Variables: ``sst``/``sss``/``ssh``/``a_ice``/``m_ice`` 2-D
``(time, nod2)`` + ``temp``/``salt`` 3-D ``(time, nz_1, nod2)`` (below-bottom masked NaN).

Multi-year forcing: the JRA55 reader is single-year, so the CoreForcing is rebuilt at
each year boundary (the static a_ice IC mask is held fixed). Stability is monitored; the
run aborts on blow-up. Forcing is streamed per step (no OOM).

Usage:  python scripts/core2_kpp_climate_run.py --years 2 --dt 1800
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from fesom_jax import core2_forcing, ice, ssh
from fesom_jax import step as stepmod
from fesom_jax.ale import AleConfig
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.kpp import KppConfig
from fesom_jax.tke import TkeConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import core2_initial_state

ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
IC_DIR_ZSTAR = ROOT / "data" / "ic_core2_dist16"   # --ale on: the zstar-canonical IC

SSH_ABSMAX, VEL_ABSMAX, MICE_MAX = 5.0, 3.0, 20.0
# C-port variable names / metadata (the m32_climate_compare surface fields + 3-D T/S)
VARS_2D = [("sst", "degC", "sea surface temperature"),
           ("sss", "psu", "sea surface salinity"),
           ("ssh", "m", "sea surface height"),
           ("a_ice", "1", "sea ice concentration"),
           ("m_ice", "m", "sea ice volume per unit area")]
VARS_3D = [("temp", "degC", "sea water potential temperature"),
           ("salt", "psu", "sea water salinity")]


def make_diag(mesh):
    surf = jnp.asarray(np.asarray(mesh.node_layer_mask)[:, 0])
    em = jnp.asarray(mesh.elem_layer_mask)[:, :, None]
    areasvol0 = jnp.asarray(np.asarray(mesh.areasvol)[:, 0])

    @jax.jit
    def diag(state):
        T0, S0, eta, uv = state.T[:, 0], state.S[:, 0], state.eta_n, state.uv
        finite = (jnp.isfinite(state.T).all() & jnp.isfinite(state.S).all()
                  & jnp.isfinite(uv).all() & jnp.isfinite(eta).all()
                  & jnp.isfinite(state.a_ice).all() & jnp.isfinite(state.m_ice).all())
        vel_elem = jnp.max(jnp.where(em, jnp.abs(uv), 0.0), axis=(1, 2))
        return dict(finite=finite,
                    sst_min=jnp.min(jnp.where(surf, T0, jnp.inf)),
                    sst_max=jnp.max(jnp.where(surf, T0, -jnp.inf)),
                    sss_min=jnp.min(jnp.where(surf, S0, jnp.inf)),
                    sss_max=jnp.max(jnp.where(surf, S0, -jnp.inf)),
                    ssh_absmax=jnp.max(jnp.where(surf, jnp.abs(eta), -jnp.inf)),
                    vel_absmax=jnp.max(vel_elem),
                    ice_area=jnp.sum(state.a_ice * areasvol0),
                    mice_max=jnp.max(state.m_ice), aice_max=jnp.max(state.a_ice))
    return diag


def stable(d):
    if not bool(d["finite"]):
        return False, "NaN/Inf"
    if float(d["ssh_absmax"]) >= SSH_ABSMAX:
        return False, f"|SSH|>={SSH_ABSMAX}"
    if float(d["vel_absmax"]) >= VEL_ABSMAX:
        return False, f"max|vel|>={VEL_ABSMAX}"
    if float(d["mice_max"]) >= MICE_MAX:
        return False, f"m_ice>={MICE_MAX}"
    return True, ""


def report(tag, d, dt_step=None):
    t = "" if dt_step is None else f"  [{dt_step:5.2f}s]"
    print(f"{tag:>12}  fin={int(d['finite'])}  SST[{d['sst_min']:+6.2f},{d['sst_max']:6.2f}]  "
          f"SSS[{d['sss_min']:5.2f},{d['sss_max']:5.2f}]  |SSH|={d['ssh_absmax']:.2e} "
          f"|vel|={d['vel_absmax']:.2e}  ice[a={d['aice_max']:.3f} m={d['mice_max']:.3f}]{t}",
          flush=True)


class MonthlyWriter:
    """Per-(var, year) CF-1.8 NetCDF in the C-port ``<var>.fesom.<yr>.monthly.nc`` format
    (12 records/year), readable by ushow + m32_climate_compare."""

    def __init__(self, outdir, mesh, nz1):
        self.dir = Path(outdir); self.dir.mkdir(parents=True, exist_ok=True)
        self.mesh, self.nz1 = mesh, nz1
        g = np.asarray(mesh.geo_coord_nod2D) * (180.0 / np.pi)
        self.lon, self.lat = g[:, 0], g[:, 1]
        self.z = -np.asarray(mesh.Z)
        self.surf = np.asarray(mesh.node_layer_mask)[:, 0]
        self.m3T = np.asarray(mesh.node_layer_mask)[:, :nz1].T   # (nz1, nod2)
        self.files = {}                                          # (var, year) -> [ds, vh, idx]

    def _get(self, var, year, is3d, units, ln):
        key = (var, year)
        if key not in self.files:
            ds = Dataset(self.dir / f"{var}.fesom.{year}.monthly.nc", "w",
                         format="NETCDF4_CLASSIC")
            ds.createDimension("time", None); ds.createDimension("nod2", len(self.lon))
            tv = ds.createVariable("time", "f8", ("time",))
            tv.units = f"days since {year:04d}-01-01 00:00:00"; tv.calendar = "standard"
            tv.long_name = "model time"
            lo = ds.createVariable("lon", "f8", ("nod2",)); lo.units = "degrees_east"; lo[:] = self.lon
            la = ds.createVariable("lat", "f8", ("nod2",)); la.units = "degrees_north"; la[:] = self.lat
            if is3d:
                ds.createDimension("nz_1", self.nz1)
                z = ds.createVariable("z", "f8", ("nz_1",)); z.units = "m"; z.positive = "down"; z[:] = self.z
                vh = ds.createVariable(var, "f8", ("time", "nz_1", "nod2"), fill_value=np.nan)
            else:
                vh = ds.createVariable(var, "f8", ("time", "nod2"), fill_value=np.nan)
            vh.long_name = ln; vh.units = units; vh.coordinates = "lon lat"
            ds.Conventions = "CF-1.8"
            self.files[key] = [ds, vh, 0]
        return self.files[key]

    def write_month(self, year, day_center, means):
        """means: dict var-> numpy (2-D var → (nod2,); 3-D var → (nod2, nz1))."""
        for var, units, ln in VARS_2D:
            ds, vh, idx = self._get(var, year, False, units, ln)
            a = means[var]
            vh[idx, :] = np.where(self.surf, a, np.nan) if var in ("sst", "sss", "ssh") else a
            ds.variables["time"][idx] = day_center; self.files[(var, year)][2] = idx + 1; ds.sync()
        for var, units, ln in VARS_3D:
            ds, vh, idx = self._get(var, year, True, units, ln)
            a = means[var].T                                    # (nz1, nod2)
            vh[idx, :, :] = np.where(self.m3T, a, np.nan)
            ds.variables["time"][idx] = day_center; self.files[(var, year)][2] = idx + 1; ds.sync()

    def close(self):
        for ds, _, _ in self.files.values():
            ds.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--start-year", type=int, default=1958)
    ap.add_argument("--dt", type=float, default=1800.0)
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--ale", choices=["on", "off"], default="off",
                    help="on ⇒ zstar (ale_cfg=AleConfig() + the dist16 IC); off ⇒ linfs (default)")
    ap.add_argument("--tke", choices=["on", "off"], default="off",
                    help="on ⇒ classical-TKE mixing (tke_cfg=TkeConfig(), KPP off — the c_tke_2yr "
                         "config; use --ic-dir data/ic_core2_dist864 to match that 864r oracle)")
    ap.add_argument("--ic-dir", type=str, default="",
                    help="override the IC cache dir (e.g. data/ic_core2_dist864 to match an 864r oracle)")
    ap.add_argument("--steps", type=int, default=0, help="override n_steps (smoke test)")
    args = ap.parse_args()
    dt = args.dt
    n_steps = args.steps if args.steps > 0 else int(round(args.years * 365 * 86400 / dt))
    ale_cfg = AleConfig() if args.ale == "on" else None
    ic_dir = Path(args.ic_dir) if args.ic_dir else (IC_DIR_ZSTAR if args.ale == "on" else IC_DIR)
    out = args.out or str(ROOT / "data" / ("zstar_climate" if args.ale == "on" else "kpp_climate_2yr"))

    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0 = time.time()
    mesh = load_mesh(MESH_DIR)
    sst0 = np.asarray(core2_initial_state(mesh, ic_dir).T[:, 0])
    state = ice.seed_ice(core2_initial_state(mesh, ic_dir), mesh, sst0)
    op = ssh.build_ssh_operator(mesh, dt=dt)
    dates = core2_forcing.dates_for_steps(args.start_year, dt, n_steps)
    cf_year = args.start_year
    cf = core2_forcing.build_core_forcing(mesh, cf_year, sst_ic=sst0)
    ice_cfg, gm_cfg = IceConfig(), GMConfig()
    # 3-way mixing: --tke on ⇒ classical-TKE (KPP off, the c_tke_2yr config); else KPP.
    tke_cfg = TkeConfig() if args.tke == "on" else None
    kpp_cfg = None if args.tke == "on" else KppConfig()
    diag = make_diag(mesh)
    nz1 = int(mesh.nl) - 1
    writer = MonthlyWriter(out, mesh, nz1)
    print(f"[setup] built in {time.time()-t0:.1f}s; {n_steps} steps × dt={dt:.0f} = "
          f"{n_steps*dt/86400:.1f} days; ale={args.ale}; monthly means → {out}", flush=True)

    @jax.jit
    def accumulate(acc, st):
        return dict(sst=acc["sst"] + st.T[:, 0], sss=acc["sss"] + st.S[:, 0],
                    ssh=acc["ssh"] + st.eta_n, a_ice=acc["a_ice"] + st.a_ice,
                    m_ice=acc["m_ice"] + st.m_ice,
                    temp=acc["temp"] + st.T[:, :nz1], salt=acc["salt"] + st.S[:, :nz1])

    def zero_acc():
        N = int(mesh.nod2D)
        return dict(sst=jnp.zeros(N), sss=jnp.zeros(N), ssh=jnp.zeros(N),
                    a_ice=jnp.zeros(N), m_ice=jnp.zeros(N),
                    temp=jnp.zeros((N, nz1)), salt=jnp.zeros((N, nz1)))

    def flush(year, month, acc, count, last_doy):
        means = {k: np.asarray(v) / count for k, v in acc.items()}
        writer.write_month(year, float(last_doy), means)
        print(f"  [month {year}-{month:02d}] mean of {count} steps → "
              f"{writer.dir}/<var>.fesom.{year}.monthly.nc", flush=True)

    d = jax.device_get(diag(state)); report("init", d)
    am_year, am_month = dates[0][0], dates[0][3]
    acc, count, last_doy = zero_acc(), 0, dates[0][1]
    worst_vel = 0.0
    for i in range(n_steps):
        y, doy, sec, month = dates[i]
        if y != cf_year:
            cf = core2_forcing.build_core_forcing(mesh, y, sst_ic=sst0); cf_year = y
            print(f"  [forcing] rebuilt for year {y}", flush=True)
        sf = cf.step_forcing(y, doy, sec, month)
        ts = time.time()
        state = stepmod.step_jit(state, mesh, op, None, dt=dt, is_first_step=(i == 0),
                                 step_forcing=sf, forcing_static=cf.static,
                                 ice_cfg=ice_cfg, gm_cfg=gm_cfg, kpp_cfg=kpp_cfg,
                                 tke_cfg=tke_cfg, ale_cfg=ale_cfg)
        if (y, month) != (am_year, am_month):              # entered a new month → flush prev
            flush(am_year, am_month, acc, count, last_doy)
            acc, count = zero_acc(), 0
            am_year, am_month = y, month
        acc = accumulate(acc, state); count += 1; last_doy = doy
        step = i + 1
        if step % args.every == 0 or step <= 2 or step == n_steps:
            d = jax.device_get(diag(state)); ok, why = stable(d)
            worst_vel = max(worst_vel, float(d["vel_absmax"]))
            report(f"s{step}·d{int(step*dt//86400)}", d, time.time() - ts)
            if not ok:
                print(f"\nFAIL at step {step} (day {step*dt/86400:.2f}): {why}", flush=True)
                flush(am_year, am_month, acc, count, last_doy); writer.close()
                return 1
    flush(am_year, am_month, acc, count, last_doy)         # final month
    writer.close()
    print(f"\nPASS: {n_steps} steps ({n_steps*dt/86400:.1f} days) stable; worst |vel|={worst_vel:.3e}; "
          f"monthly means → {args.out}", flush=True)
    print("KPP_CLIMATE_RUN_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
