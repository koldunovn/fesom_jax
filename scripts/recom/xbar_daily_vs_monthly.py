"""Is a daily reference solution worth having? Daily x-bar against the monthly x-bar interpolated to daily.

The scaling of docs/plans/20261105-recom-parameter-estimation.md is x = xbar * xt with a time-varying reference. From the
daily BGC output of the FESOM 2.8 run kdind15w28sl1d (2001): the floored daily field xd; its calendar-month means,
interpolated linearly in time between mid-months (cyclic over the year, the forcing being repeat-year) to xm; the relative error
|xm - xd| / xd per tracer, volume-weighted, by latitude band and depth class, and for the surface by month.

Negative or too small values are replaced by REcoM's own floors, max(tiny_i, x), the same clip recom_sms.F90 applies to the
state before using it (lines 520-710; the output is the transported state, which can be below the floor): ``tiny`` for the
nutrients, carbonate tracers, dissolved organics, zooplankton and detritus, ``tiny_chl`` for chlorophyll, and the per-group
``tiny_N``, ``tiny_C``, ``tiny_Si`` derived from it. The floor is applied when the data are read; the output files are untouched.
The monthly means are taken from the floored daily fields.

``--epsilon`` adds a local floor for the scaling, per grid point and level: xs = max(x, eps * max_t x), each product from its
own information, so the daily field with the maximum over its 365 days and the interpolated monthly field with the maximum of
its 12 means. Results go to OUT/eps_<eps>.

    sbatch scripts/albedo/run_xbar_compare.sbatch            (one JSON per tracer and summary.txt in OUT)
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fesom_jax.recom import params as P  # noqa: E402

RUN = Path("/albedo/scratch/user/sloza/output_runs/fesom2.8_spectral_kdind15w28sl1d")
MESH = Path("/albedo/work/projects/p_specbic/sloza/fesom_jax_data/core2_mesh_ic/mesh_core2")
OUT = Path("/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/xbar_daily_vs_monthly")
YEAR_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
BANDS = (-90, -60, -45, -15, 15, 45, 65, 90)                    # latitude bands, standing in for biomes
BAND_NAMES = ("S of 60S", "60-45S", "45-15S", "15S-15N", "15-45N", "45-65N", "N of 65N")
DEPTHS = (0, 50, 200, 1000, 1e5)
DEPTH_NAMES = ("0-50 m", "50-200 m", "200-1000 m", "below 1000 m")
NBIN = 82                                                       # log10 relative error -4..4 in 0.1, under and over


def recom_floors(namelist: Path) -> dict:
    """``{output file name: floor}``: the clip of recom_sms.F90 (fork: ``tiny_Si = tiny_C_d / SiCmax``, S:362)."""
    _, p = P.load_namelist(namelist, check=False)
    d = P.derived(p)
    t, c = P.TINY, P.TINY_CHL
    floors = {n: t for n in ("DIN", "DSi", "DFe", "DIC", "Alk", "O2", "DON", "DOC", "CDOM", "PhyCalc", "HetN", "HetC", "Zoo2N",
                             "Zoo2C", "Zoo3N", "Zoo3C", "DetN", "DetC", "DetSi", "DetCalc", "idetz2n", "idetz2c", "idetz2si",
                             "idetz2calc")}
    floors.update(PhyN=d.tiny_N, PhyC=d.tiny_C, PhyChl=c, DiaN=d.tiny_N_d, DiaC=d.tiny_C_d, DiaChl=c,
                  DiaSi=d.tiny_C_d / p.SiCmax, CoccoN=d.tiny_N_c, CoccoC=d.tiny_C_c, CoccoChl=c, PhaeoN=d.tiny_N_p,
                  PhaeoC=d.tiny_C_p, PhaeoChl=c)
    return floors


def monthly_to_daily():
    """For each day of the year ``(i, j, a)``: interpolate ``(1 - a) M[i] + a M[j]`` between cyclic mid-month points."""
    edges = np.cumsum((0,) + YEAR_DAYS)
    mid = (edges[:-1] + edges[1:] - 1) / 2.0
    out = []
    for t in range(365):
        j = int(np.searchsorted(mid, t, side="right")) % 12          # next mid-month
        i = (j - 1) % 12
        span = (mid[j] - mid[i]) % 365
        out.append((i, j, ((t - mid[i]) % 365) / span))
    return out


def compare(args):
    name, floor, cls, wet, weight, ncls, eps, out = args
    with netCDF4.Dataset(RUN / f"{name}.fesom.2001.nc") as ds:
        x = np.ma.filled(ds[name][:], np.nan).astype(np.float32)
    if x.shape[0] != 365:
        raise ValueError(f"{name}: {x.shape[0]} records")
    w = x[:, wet]                                                    # (365, wet cells)
    below = float(np.mean(w < floor))
    negative = float(np.mean(w < 0))
    raw_min = float(np.nanmin(w))
    w = np.maximum(w, np.float32(floor))
    edges = np.cumsum((0,) + YEAR_DAYS)
    month_of = np.searchsorted(edges[1:], np.arange(365), side="right")
    means = np.stack([w[edges[m]:edges[m + 1]].mean(axis=0) for m in range(12)])
    scale_d = scale_m = None
    local = 0.0
    if eps:
        scale_d, scale_m = eps * w.max(axis=0), eps * means.max(axis=0)
        local = float(np.mean(w < scale_d))
    hist = np.zeros(ncls * NBIN)
    surf = np.zeros((12, NBIN))
    surface = np.flatnonzero(cls % len(DEPTH_NAMES) == 0)            # 0-50 m
    for t, (i, j, a) in enumerate(monthly_to_daily()):
        xm = (1 - a) * means[i] + a * means[j]
        xd = w[t]
        if eps:
            xd, xm = np.maximum(xd, scale_d), np.maximum(xm, scale_m)
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.abs(xm - xd) / xd
            b = np.floor((np.log10(e) + 4.0) * 10.0)
        b = np.clip(np.nan_to_num(b, nan=-1.0, neginf=-1.0, posinf=NBIN), -1, NBIN - 2).astype(np.int64) + 1
        hist += np.bincount(cls * NBIN + b, weights=weight, minlength=ncls * NBIN)
        surf[month_of[t]] += np.bincount(b[surface], weights=weight[surface], minlength=NBIN)
    rec = dict(name=name, floor=float(floor), epsilon=eps, frac_local_floor=local, raw_min=raw_min, frac_below_floor=below,
               frac_negative=negative,
               hist=hist.reshape(ncls, NBIN).tolist(), surface_by_month=surf.tolist())
    (out / f"{name}.json").write_text(json.dumps(rec))
    return name


def quantile(h, q):
    """The relative error at quantile ``q`` of a volume-weighted histogram (lower edge of the 0.1-decade bin)."""
    c = np.cumsum(h)
    if c[-1] <= 0:
        return float("nan")
    k = int(np.searchsorted(c, q * c[-1]))
    return 10.0 ** (-4.0 + (k - 1) * 0.1) if k > 0 else 0.0


def share_above(h, r):
    """The volume fraction with a relative error above ``r``."""
    k = int(np.round((np.log10(r) + 4.0) * 10.0)) + 1
    return float(h[k:].sum() / h.sum()) if h.sum() > 0 else float("nan")


def summarise(names, out):
    lines = [f"{'tracer':>11} {'floor':>9} {'raw min':>10} {'%neg':>6} {'%<floor':>7} {'%<local':>7}  " +
             "  ".join(f"{d} med/p90/>50%" for d in DEPTH_NAMES)]
    for n in names:
        r = json.loads((out / f"{n}.json").read_text())
        h = np.array(r["hist"])
        by_d = [h[d::len(DEPTH_NAMES)].sum(axis=0) for d in range(len(DEPTH_NAMES))]
        cells = "  ".join(f"{quantile(b, .5):6.3f}/{quantile(b, .9):6.3f}/{100 * share_above(b, .5):5.1f}%" for b in by_d)
        lines.append(f"{n:>11} {r['floor']:9.2e} {r['raw_min']:10.2e} {100 * r['frac_negative']:6.2f} "
                     f"{100 * r['frac_below_floor']:7.2f} {100 * r['frac_local_floor']:7.2f}  {cells}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tracers", nargs="+", default=None, help="default: every tracer with a floor")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epsilon", type=float, nargs="+", default=[0.0], help="local scaling floors; 0 = REcoM's floor only")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    floors = recom_floors(RUN / "namelist.recom")
    names = a.tracers or sorted(floors)
    lat = np.degrees(np.load(MESH / "geo_coord_nod2D.npy")[:, 1])
    nlev = np.load(MESH / "nlevels_nod2D.npy")
    ulev = np.load(MESH / "ulevels_nod2D.npy")
    zbar = np.abs(np.load(MESH / "zbar.npy"))
    Z = np.abs(np.load(MESH / "Z.npy"))
    areas = np.load(MESH / "areasvol.npy")
    nl1 = Z.size
    k = np.arange(nl1)[:, None]
    wet = (k < nlev[None, :] - 1) & (k >= ulev[None, :] - 1)         # (nl-1, nod2D)
    band = np.digitize(lat, BANDS[1:-1])
    deep = np.digitize(Z, DEPTHS[1:-1])
    layer_band = np.broadcast_to(band[None, :], wet.shape)
    layer_deep = np.broadcast_to(deep[:, None], wet.shape)
    cls = (layer_band * len(DEPTH_NAMES) + layer_deep)[wet].astype(np.int64)
    vol = (areas.T[:nl1] if areas.shape[0] != nl1 + 1 else areas[:nl1]) * np.diff(zbar)[:, None]
    weight = vol[wet]
    ncls = len(BAND_NAMES) * len(DEPTH_NAMES)
    print(f"{len(names)} tracers, {wet.sum()} wet cells, {a.workers} workers", flush=True)
    for eps in a.epsilon:
        out = OUT / f"eps_{eps:g}" if eps else OUT
        out.mkdir(parents=True, exist_ok=True)
        jobs = [(n, floors[n], cls, wet, weight, ncls, eps, out) for n in names]
        with ProcessPoolExecutor(a.workers) as ex:
            for done in ex.map(compare, jobs):
                print("done", eps, done, flush=True)
        print(f"epsilon {eps:g}", flush=True)
        summarise(names, out)


if __name__ == "__main__":
    main()
