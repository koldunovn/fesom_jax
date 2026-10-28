"""Figures and animations of a CORE2 + REcoM run for the page to REcoM's authors (nereus env; a compute node).

Reads the run's monthly means (``<run>/monthly/<YYYY_MM>``, zarr) and ``obs_on_mesh.npz`` (prep_obs.py); every map is
drawn with nereus (regridded to 1 degree, nearest node), never as a scatter plot. Writes into ``--out``:

* ``summary.json``: monthly and annual global numbers (NPP, export at 100 m, air-sea CO2 flux [Pg C yr-1], area-mean
  surface Chl and pCO2, inventories [Pmol] and their change), the map period, the figure list;
* ``chl_maps.png`` (annual-mean surface Chl: model, GlobColour, log ratio, over the months both have),
  ``chl_zonal.png`` (zonal-mean Chl by month), ``chl_regions.png`` (seasonal cycles in four regions),
  ``npp_export.png``, ``co2.png`` (CO2 flux, surface pCO2), ``nutrients.png`` (surface DIN and DSi vs WOA13),
  ``carbon.png`` (surface DIC and Alk vs GLODAPv2), ``o2.png`` (O2 at ~400 m vs WOA18), ``sections.png`` (zonal-mean
  DIN and O2 minus the climatologies), ``timeseries.png``;
* ``chl_season.mp4`` (+ ``frames_season/``): the monthly surface Chl climatology, model next to GlobColour;
  ``run.mp4`` (+ ``frames_run/``): every month of the run, surface Chl and the CO2 flux.

    python scripts/recom/page/make_figures.py <run>/monthly --obs <dir>/obs_on_mesh.npz --out <dir>
        [--label "2000-2019"] [--dic-ref DIC|DIC_PI] [--map-years Y0 Y1] [--clim-years Y0 Y1]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MMOL_D_TO_PGC_YR = 365.0 * 12.011e-3 / 1e15          # mmol C d-1 -> Pg C yr-1
INV = ("DIC", "Alk", "DIN", "DSi", "DFe", "O2")
SURF = ("Chl", "DIN", "DSi", "DFe", "DIC", "Alk")
FLUX2D = ("npp", "export100", "co2flux", "pco2surf")
DEEP = ("DIN", "DSi", "O2", "DIC", "Alk")             # annual means over the map period, full depth
REGIONS = {                                           # (lat0, lat1, [(lon0, lon1), ...]) in degrees, lon in -180..180
    "Subpolar North Atlantic": (45.0, 65.0, [(-60.0, -10.0)]),
    "Equatorial Pacific": (-5.0, 5.0, [(160.0, 180.0), (-180.0, -90.0)]),
    "North Pacific gyre": (15.0, 30.0, [(160.0, 180.0), (-180.0, -140.0)]),
    "Southern Ocean": (-60.0, -45.0, [(-180.0, 180.0)]),
}
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
FFMPEG = shutil.which("ffmpeg") or "/work/kv0653/spack-flo/ffmpeg-4.3.2-6bmsu6/bin/ffmpeg"


def mesh_arrays(m: Path) -> dict:
    lon, lat = np.rad2deg(np.load(m / "geo_coord_nod2D.npy")).T
    lon = ((lon + 180.0) % 360.0) - 180.0
    zbar = np.load(m / "zbar.npy")
    nlev = np.load(m / "nlevels_nod2D.npy").astype(int)
    dz = zbar[:-1] - zbar[1:]
    nl1 = dz.size
    wet = np.arange(nl1)[None, :] < (nlev - 1)[:, None]
    areasvol = np.load(m / "areasvol.npy")
    return dict(lon=lon, lat=lat, area=np.load(m / "area.npy")[:, 0], zbar=zbar, dz=dz, nl1=nl1, wet=wet,
                Z=np.abs(np.load(m / "Z.npy"))[:nl1],
                vol=np.where(wet, dz[None, :] * areasvol[:, :nl1], 0.0))


def region_mask(lon, lat, spec):
    lat0, lat1, boxes = spec
    m = np.zeros(lon.shape, bool)
    for a, b in boxes:
        m |= (lon >= a) & (lon <= b)
    return m & (lat >= lat0) & (lat <= lat1)


def read_run(monthly: Path, M: dict, deep_years):
    """Monthly surface fields, 2-D fluxes and inventories; annual-mean deep fields over ``deep_years``."""
    import zarr
    months = sorted(p for p in monthly.iterdir() if (p / "Chl").exists())
    n, nm = M["lon"].size, len(months)
    surf = {k: np.zeros((nm, n), np.float32) for k in SURF}
    flux = {k: np.zeros((nm, n), np.float32) for k in FLUX2D}
    deep = {k: np.zeros((n, M["nl1"])) for k in DEEP}
    n_deep, rows = 0, []
    for i, p in enumerate(months):
        g = zarr.open_group(str(p), mode="r")
        row = {"month": p.name}
        in_deep = deep_years[0] <= int(p.name[:4]) <= deep_years[1]
        for k in set(SURF) | set(INV):
            x = np.asarray(g[k][0], dtype=np.float64).T[:, :M["nl1"]]           # (nod2D, nl-1)
            if k in SURF:
                surf[k][i] = x[:, 0]
            if k in INV:
                row[k] = float(np.sum(np.where(M["wet"], x, 0.0) * M["vol"])) / 1e18      # mmol -> Pmol
            if k in DEEP and in_deep:
                deep[k] += x
        n_deep += int(in_deep)
        for k in FLUX2D:
            flux[k][i] = np.asarray(g[k][0], dtype=np.float32)
            if k == "pco2surf":
                row[k] = float(np.sum(flux[k][i] * M["area"]) / np.sum(M["area"]))
            else:
                row[k] = float(np.sum(flux[k][i].astype(np.float64) * M["area"])) * MMOL_D_TO_PGC_YR
        row["Chl"] = float(np.sum(surf["Chl"][i] * M["area"]) / np.sum(M["area"]))
        rows.append(row)
        print(f"read {p.name}: NPP {row['npp']:.2f}, export {row['export100']:.2f}, CO2 {row['co2flux']:.2f} Pg C/yr",
              flush=True)
    for r in rows:
        for k in INV:
            r[k + "_rel"] = r[k] / rows[0][k] - 1.0
    deep = {k: np.where(M["wet"], v / max(n_deep, 1), np.nan) for k, v in deep.items()}
    return months, surf, flux, deep, rows


class Maps:
    """nereus maps on Robinson with one shared interpolator."""

    def __init__(self, M):
        import cartopy.crs as ccrs
        self.M, self.proj, self.interp = M, ccrs.Robinson(central_longitude=-150.0), None

    def draw(self, fig, ax, v, title, cmap, vmin, vmax, label, ticks=None, ticklabels=None, cbar=True):
        import nereus
        from matplotlib.collections import QuadMesh
        _, _, self.interp = nereus.plot(np.asarray(v, float), self.M["lon"], self.M["lat"], projection=self.proj,
                                        resolution=1.0, interpolator=self.interp, method="nearest",
                                        influence_radius=150e3, cmap=cmap, vmin=vmin, vmax=vmax, land=True,
                                        colorbar=False, ax=ax, rasterized=True)
        ax.set_global()
        ax.set_title(title, loc="left", fontsize=9)
        if cbar:
            qm = [c for c in ax.collections if isinstance(c, QuadMesh)][-1]
            cb = fig.colorbar(qm, ax=ax, orientation="horizontal", fraction=0.06, pad=0.03, aspect=35)
            cb.set_label(label, fontsize=8)
            if ticks is not None:
                cb.set_ticks(ticks)
                cb.set_ticklabels(ticklabels)
            cb.ax.tick_params(labelsize=7)


def cmap(name, fallback):
    try:
        import cmocean
        return getattr(cmocean.cm, name)
    except Exception:
        return fallback


CHL_TICKS = np.log10([0.03, 0.1, 0.3, 1.0, 3.0])
CHL_LABELS = ["0.03", "0.1", "0.3", "1", "3"]
CHL_LIM = (np.log10(0.02), np.log10(5.0))


def log_chl(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(np.where(x > 0, x, np.nan))


def zonal(v, lat, w, edges):
    """Area-weighted zonal means of ``v`` (..., n) in latitude bins; NaN-aware."""
    idx = np.digitize(lat, edges) - 1
    nb = edges.size - 1
    out = np.full(v.shape[:-1] + (nb,), np.nan)
    for b in range(nb):
        s = idx == b
        if not s.any():
            continue
        vv, ww = v[..., s], np.broadcast_to(w[s], v[..., s].shape)
        ok = np.isfinite(vv)
        den = np.sum(np.where(ok, ww, 0.0), axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[..., b] = np.where(den > 0, np.sum(np.where(ok, vv * ww, 0.0), axis=-1) / den, np.nan)
    return out


def encode(frames_dir: Path, out: Path, fps: int):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "f%03d.png"),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
           "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg libx264 failed, trying mpeg4:", r.stderr[-300:])
        cmd[cmd.index("libx264")] = "mpeg4"
        subprocess.run(cmd, check=True)
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("monthly", type=Path)
    ap.add_argument("--obs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, default=REPO / "data" / "mesh_core2")
    ap.add_argument("--label", default="")
    ap.add_argument("--dic-ref", default="DIC", choices=("DIC", "DIC_PI"))
    ap.add_argument("--map-years", type=int, nargs=2, default=None, help="annual-mean maps (default: last 10 years)")
    ap.add_argument("--clim-years", type=int, nargs=2, default=None,
                    help="years of the model's seasonal Chl climatology (default: the GlobColour years, else all)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 150, "savefig.dpi": 150,
                         "savefig.bbox": "tight", "savefig.facecolor": "white"})

    M = mesh_arrays(args.mesh)
    obs = np.load(args.obs)
    lon, lat, area = M["lon"], M["lat"], M["area"]
    years = sorted({int(p.name[:4]) for p in args.monthly.iterdir() if (p / "Chl").exists()})
    map_years = tuple(args.map_years) if args.map_years else (max(years[0], years[-1] - 9), years[-1])
    months, surf, flux, deep, rows = read_run(args.monthly, M, map_years)
    yr = np.array([int(p.name[:4]) for p in months])
    mo = np.array([int(p.name[5:7]) for p in months])
    in_map = (yr >= map_years[0]) & (yr <= map_years[1])
    cy = tuple(int(x) for x in obs["chl_years"])
    if args.clim_years:
        clim_years = tuple(args.clim_years)
    elif years[0] <= cy[1] and years[-1] >= cy[0]:
        clim_years = (max(years[0], cy[0]), min(years[-1], cy[1]))
    else:
        clim_years = (years[0], years[-1])
    in_clim = (yr >= clim_years[0]) & (yr <= clim_years[1])
    period = f"{map_years[0]}" if map_years[0] == map_years[1] else f"{map_years[0]}-{map_years[1]}"
    tag = f" ({args.label})" if args.label else ""

    # ---- numbers ----
    annual = {}
    for y in years:
        s = yr == y
        annual[str(y)] = {k: float(np.mean([r[k] for r, t in zip(rows, s) if t]))
                          for k in ("npp", "export100", "co2flux", "pco2surf", "Chl")}
    per = {k: float(np.mean([r[k] for r, t in zip(rows, in_map) if t])) for k in ("npp", "export100", "co2flux",
                                                                                   "pco2surf", "Chl")}
    per["e_ratio"] = per["export100"] / per["npp"]

    # model seasonal Chl climatology and the matched GlobColour months
    mchl = np.stack([surf["Chl"][in_clim & (mo == m)].mean(axis=0) if np.any(in_clim & (mo == m))
                     else np.full(lon.size, np.nan) for m in range(1, 13)])
    ochl = obs["chl_clim"].astype(float)
    both = np.isfinite(ochl) & np.isfinite(mchl)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_ann = np.where(both.any(0), np.nansum(np.where(both, mchl, 0.0), 0) / both.sum(0), np.nan)
        o_ann = np.where(both.any(0), np.nansum(np.where(both, ochl, 0.0), 0) / both.sum(0), np.nan)
    ok = np.isfinite(m_ann) & np.isfinite(o_ann) & (m_ann > 0) & (o_ann > 0)
    lr_m, lr_o = np.log10(m_ann[ok]), np.log10(o_ann[ok])
    w = area[ok]
    r_log = float(np.corrcoef(lr_m, lr_o)[0, 1])
    bias_log = float(np.sum((lr_m - lr_o) * w) / np.sum(w))
    per.update(chl_model_matched=float(np.sum(m_ann[ok] * w) / np.sum(w)),
               chl_obs_matched=float(np.sum(o_ann[ok] * w) / np.sum(w)), chl_log_corr=r_log, chl_log_bias=bias_log)
    print("period means:", {k: round(v, 3) for k, v in per.items()}, flush=True)

    figs = {}
    maps = Maps(M)
    algae = cmap("algae", "viridis")

    # ---- Chl maps ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), layout="constrained", subplot_kw={"projection": maps.proj})
    maps.draw(fig, axes[0], np.where(ok, np.log10(np.maximum(m_ann, 1e-9)), np.nan),
              f"REcoM{tag}, {clim_years[0]}-{clim_years[1]}", algae, *CHL_LIM, "surface Chl a [mg m$^{-3}$]",
              CHL_TICKS, CHL_LABELS)
    maps.draw(fig, axes[1], np.where(ok, np.log10(np.maximum(o_ann, 1e-9)), np.nan),
              f"GlobColour (merged satellites), {cy[0]}-{cy[1]}", algae, *CHL_LIM, "surface Chl a [mg m$^{-3}$]",
              CHL_TICKS, CHL_LABELS)
    ratio = np.full(lon.size, np.nan)
    ratio[ok] = lr_m - lr_o
    maps.draw(fig, axes[2], ratio, f"model / satellite (log scale): r = {r_log:.2f}", "RdBu_r", -1.0, 1.0,
              "ratio", np.log10([0.1, 0.3, 1, 3, 10]), ["1/10", "1/3", "1", "3", "10"])
    fig.savefig(args.out / "chl_maps.png")
    plt.close(fig)
    figs["chl_maps"] = "chl_maps.png"

    # ---- zonal-mean Chl by month ----
    edges = np.arange(-80.0, 82.0, 2.0)
    zm = zonal(np.where(both, mchl, np.nan), lat, area, edges)
    zo = zonal(np.where(both, ochl, np.nan), lat, area, edges)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained", sharey=True)
    for ax, z, t in ((axes[0], zm, f"REcoM{tag}"), (axes[1], zo, "GlobColour")):
        pc = ax.pcolormesh(np.arange(13) + 0.5, edges, log_chl(z).T, cmap=algae, vmin=CHL_LIM[0], vmax=CHL_LIM[1],
                           shading="flat")
        ax.set_xticks(range(1, 13), [m[0] for m in MONTHS])
        ax.set_title(f"{t}: zonal-mean surface Chl a", loc="left")
        ax.set_ylim(-80, 80)
    axes[0].set_ylabel("latitude")
    cb = fig.colorbar(pc, ax=axes, fraction=0.03, pad=0.01, ticks=CHL_TICKS)
    cb.set_ticklabels(CHL_LABELS)
    cb.set_label("mg m$^{-3}$")
    fig.savefig(args.out / "chl_zonal.png")
    plt.close(fig)
    figs["chl_zonal"] = "chl_zonal.png"

    # ---- regional seasonal cycles ----
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), layout="constrained")
    reg = {}
    for ax, (name, spec) in zip(axes, REGIONS.items()):
        rm = region_mask(lon, lat, spec)
        cm, co = [], []
        for m in range(12):
            s = rm & both[m]
            cm.append(float(np.sum(mchl[m, s] * area[s]) / np.sum(area[s])) if s.any() else np.nan)
            co.append(float(np.sum(ochl[m, s] * area[s]) / np.sum(area[s])) if s.any() else np.nan)
        reg[name] = {"model": cm, "obs": co}
        ax.plot(range(1, 13), co, "o-", color="0.35", ms=3.5, lw=1.2, label="GlobColour")
        ax.plot(range(1, 13), cm, "-", color="#1b7f5f", lw=2.0, label="REcoM")
        ax.set_yscale("log")
        ax.set_xticks(range(1, 13), [m[0] for m in MONTHS])
        ax.set_title(name, loc="left")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("surface Chl a [mg m$^{-3}$]")
    axes[0].legend(frameon=False)
    fig.savefig(args.out / "chl_regions.png")
    plt.close(fig)
    figs["chl_regions"] = "chl_regions.png"

    # ---- NPP and export ----
    npp = flux["npp"][in_map].mean(0)
    exp = flux["export100"][in_map].mean(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained", subplot_kw={"projection": maps.proj})
    hi = float(np.nanpercentile(npp, 99.0))
    maps.draw(fig, axes[0], npp, f"net primary production, {period}: {per['npp']:.1f} Pg C yr$^{{-1}}$",
              cmap("tempo", "viridis"), 0.0, hi, "mmol C m$^{-2}$ d$^{-1}$")
    hi = float(np.nanpercentile(exp, 99.0))
    maps.draw(fig, axes[1], exp, f"POC export at 100 m: {per['export100']:.2f} Pg C yr$^{{-1}}$ "
              f"(e-ratio {per['e_ratio']:.2f})", cmap("matter", "magma_r"), 0.0, hi, "mmol C m$^{-2}$ d$^{-1}$")
    fig.savefig(args.out / "npp_export.png")
    plt.close(fig)
    figs["npp_export"] = "npp_export.png"

    # ---- CO2 flux and pCO2 ----
    co2 = flux["co2flux"][in_map].mean(0)
    pco2 = flux["pco2surf"][in_map].mean(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained", subplot_kw={"projection": maps.proj})
    hi = float(np.nanpercentile(np.abs(co2), 98.0))
    maps.draw(fig, axes[0], co2, f"air-sea CO$_2$ flux (positive into the ocean), {period}: "
              f"{per['co2flux']:.2f} Pg C yr$^{{-1}}$", "RdBu_r", -hi, hi, "mmol C m$^{-2}$ d$^{-1}$")
    lo, hi = np.nanpercentile(pco2, [2, 98])
    maps.draw(fig, axes[1], pco2, f"surface ocean pCO$_2$, {period}", cmap("thermal", "plasma"), lo, hi, "µatm")
    fig.savefig(args.out / "co2.png")
    plt.close(fig)
    figs["co2"] = "co2.png"

    # ---- surface nutrients and carbon vs climatologies ----
    def triptych(name, rows_spec, fname):
        fig, axes = plt.subplots(len(rows_spec), 3, figsize=(15, 3.5 * len(rows_spec)), layout="constrained",
                                 subplot_kw={"projection": maps.proj}, squeeze=False)
        stats = {}
        for (k, ref, refname, units, cm_), ax in zip(rows_spec, axes):
            mod = surf[k][in_map].mean(0).astype(float)
            o = obs[ref][:, 0].astype(float)
            lo, hi = np.nanpercentile(np.concatenate([mod, o]), [1, 99])
            maps.draw(fig, ax[0], mod, f"REcoM {k}{tag}, {period}", cm_, lo, hi, units)
            maps.draw(fig, ax[1], o, refname, cm_, lo, hi, units)
            d = mod - o
            dd = float(np.nanpercentile(np.abs(d), 98))
            bias = float(np.sum(d * area) / np.sum(area))
            rmse = float(np.sqrt(np.sum(d ** 2 * area) / np.sum(area)))
            stats[k] = {"bias": bias, "rmse": rmse}
            maps.draw(fig, ax[2], d, f"difference: mean {bias:+.2f}, rms {rmse:.2f}", "RdBu_r", -dd, dd, units)
        fig.savefig(args.out / fname)
        plt.close(fig)
        figs[name] = fname
        return stats

    st_nut = triptych("nutrients", [("DIN", "DIN", "WOA13 nitrate (annual)", "mmol m$^{-3}$", cmap("matter", "magma_r")),
                                     ("DSi", "DSi", "WOA13 silicate (annual)", "mmol m$^{-3}$", cmap("haline", "cividis"))],
                      "nutrients.png")
    dic_name = "GLODAPv2 DIC (present-day, 2002)" if args.dic_ref == "DIC" else "GLODAPv2 DIC (preindustrial)"
    st_car = triptych("carbon", [("DIC", args.dic_ref, dic_name, "mmol m$^{-3}$", cmap("dense", "viridis")),
                                 ("Alk", "Alk", "GLODAPv2 alkalinity", "mmol m$^{-3}$", cmap("deep", "viridis"))],
                      "carbon.png")

    # ---- O2 at ~400 m ----
    k400 = int(np.argmin(np.abs(M["Z"] - 400.0)))
    o2m, o2o = deep["O2"][:, k400], np.where(M["wet"][:, k400], obs["O2"][:, k400], np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), layout="constrained", subplot_kw={"projection": maps.proj})
    lo, hi = 0.0, float(np.nanpercentile(o2o, 99))
    oxy = cmap("oxy", "viridis")
    maps.draw(fig, axes[0], o2m, f"REcoM O$_2$ at {M['Z'][k400]:.0f} m{tag}, {period}", oxy, lo, hi, "mmol m$^{-3}$")
    maps.draw(fig, axes[1], o2o, "WOA18 oxygen (annual)", oxy, lo, hi, "mmol m$^{-3}$")
    d = o2m - o2o
    dd = float(np.nanpercentile(np.abs(d), 98))
    maps.draw(fig, axes[2], d, "difference", "RdBu_r", -dd, dd, "mmol m$^{-3}$")
    fig.savefig(args.out / "o2.png")
    plt.close(fig)
    figs["o2"] = "o2.png"

    # ---- zonal-mean sections ----
    edges = np.arange(-78.0, 90.1, 2.0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), layout="constrained", sharex=True, sharey=True)
    for j, (k, ref, units, cm_) in enumerate([("DIN", "DIN", "mmol m$^{-3}$", cmap("matter", "magma_r")),
                                               ("O2", "O2", "mmol m$^{-3}$", oxy)]):
        mod = zonal(deep[k].T, lat, area, edges)                                     # (nl-1, nb)
        ob = zonal(np.where(M["wet"], obs[ref], np.nan).T.astype(float), lat, area, edges)
        lo, hi = np.nanpercentile(ob, [1, 99])
        pc = axes[0, j].pcolormesh(edges, M["zbar"][:M["nl1"] + 1] * -1.0 if M["zbar"][1] < 0 else M["zbar"][:M["nl1"] + 1],
                                   mod, cmap=cm_, vmin=lo, vmax=hi, shading="flat")
        fig.colorbar(pc, ax=axes[0, j], fraction=0.04, pad=0.01).set_label(units)
        axes[0, j].set_title(f"REcoM {k}, zonal mean{tag}, {period}", loc="left")
        dif = mod - ob
        dd = float(np.nanpercentile(np.abs(dif), 98))
        pc = axes[1, j].pcolormesh(edges, M["zbar"][:M["nl1"] + 1] * -1.0 if M["zbar"][1] < 0 else M["zbar"][:M["nl1"] + 1],
                                   dif, cmap="RdBu_r", vmin=-dd, vmax=dd, shading="flat")
        fig.colorbar(pc, ax=axes[1, j], fraction=0.04, pad=0.01).set_label(units)
        axes[1, j].set_title(f"minus {'WOA13 nitrate' if k == 'DIN' else 'WOA18 oxygen'}", loc="left")
    for ax in axes.ravel():
        ax.set_ylim(5500, 0)
        ax.set_facecolor("0.8")
    for ax in axes[:, 0]:
        ax.set_ylabel("depth [m]")
    for ax in axes[1]:
        ax.set_xlabel("latitude")
    fig.savefig(args.out / "sections.png")
    plt.close(fig)
    figs["sections"] = "sections.png"

    # ---- time series ----
    t = yr + (mo - 0.5) / 12.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2), layout="constrained")
    for k, lab, c in (("npp", "NPP", "#1b7f5f"), ("export100", "export at 100 m", "#9a4a12")):
        axes[0].plot(t, [r[k] for r in rows], color=c, lw=1.4, label=lab)
    axes[0].set_ylabel("Pg C yr$^{-1}$")
    axes[0].set_title("global NPP and export (monthly)", loc="left")
    axes[0].legend(frameon=False)
    axes[1].plot(t, [r["co2flux"] for r in rows], color="#2b5c9e", lw=1.4)
    axes[1].axhline(0.0, color="0.5", lw=0.6)
    axes[1].set_ylabel("Pg C yr$^{-1}$ into the ocean")
    axes[1].set_title("global air-sea CO$_2$ flux (monthly)", loc="left")
    for k, c in zip(INV, ("#2b5c9e", "#6c4a9e", "#9a4a12", "#4a7f2b", "#b8862b", "#1b7f9f")):
        axes[2].plot(t, [100.0 * r[k + "_rel"] for r in rows], color=c, lw=1.2, label=k)
    axes[2].set_ylabel("% change from the first month")
    axes[2].set_title("global inventories", loc="left")
    axes[2].legend(frameon=False, ncol=3, fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.savefig(args.out / "timeseries.png")
    plt.close(fig)
    figs["timeseries"] = "timeseries.png"

    # ---- animation 1: the seasonal cycle, model next to the satellites ----
    fr = args.out / "frames_season"
    fr.mkdir(exist_ok=True)
    for m in range(12):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.7), layout="constrained", subplot_kw={"projection": maps.proj})
        maps.draw(fig, axes[0], log_chl(np.where(np.isfinite(ochl[m]), mchl[m], np.nan)),
                  f"REcoM, {MONTHS[m]} ({clim_years[0]}-{clim_years[1]})", algae, *CHL_LIM,
                  "surface Chl a [mg m$^{-3}$]", CHL_TICKS, CHL_LABELS)
        maps.draw(fig, axes[1], log_chl(ochl[m]), f"GlobColour, {MONTHS[m]} ({cy[0]}-{cy[1]})", algae, *CHL_LIM,
                  "surface Chl a [mg m$^{-3}$]", CHL_TICKS, CHL_LABELS)
        fig.savefig(fr / f"f{m:03d}.png", bbox_inches=None)
        plt.close(fig)
    encode(fr, args.out / "chl_season.mp4", 2)
    figs["chl_season"] = "chl_season.mp4"

    # ---- animation 2: every month of the run ----
    fr = args.out / "frames_run"
    fr.mkdir(exist_ok=True)
    hi = float(np.nanpercentile(np.abs(flux["co2flux"]), 98.0))
    for i, p in enumerate(months):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.7), layout="constrained", subplot_kw={"projection": maps.proj})
        maps.draw(fig, axes[0], log_chl(surf["Chl"][i]), f"surface Chl a, {MONTHS[mo[i] - 1]} {yr[i]}", algae,
                  *CHL_LIM, "mg m$^{-3}$", CHL_TICKS, CHL_LABELS)
        maps.draw(fig, axes[1], flux["co2flux"][i], f"air-sea CO$_2$ flux, {MONTHS[mo[i] - 1]} {yr[i]} "
                  f"({rows[i]['co2flux']:+.2f} Pg C yr$^{{-1}}$)", "RdBu_r", -hi, hi, "mmol C m$^{-2}$ d$^{-1}$ into the ocean")
        fig.savefig(fr / f"f{i:03d}.png", bbox_inches=None)
        plt.close(fig)
    encode(fr, args.out / "run.mp4", 2 if len(months) <= 24 else 8)
    figs["run"] = "run.mp4"

    (args.out / "summary.json").write_text(json.dumps(
        {"label": args.label, "years": [years[0], years[-1]], "map_years": list(map_years),
         "clim_years": list(clim_years), "chl_obs_years": list(cy), "period": per, "annual": annual,
         "regions": reg, "surface_stats": {**st_nut, **st_car}, "months": rows, "figures": figs}, indent=1))
    print("wrote", args.out / "summary.json")


if __name__ == "__main__":
    main()
