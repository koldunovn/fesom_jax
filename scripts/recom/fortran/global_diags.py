"""Area-weighted global means of the daily REcoM/physics output of a pi reference run
(CO2 flux in PgC/yr, positive into the ocean) and a non-finite check over every output file.
Usage: global_diags.py RUN_DIR"""
import glob
import sys
from pathlib import Path

import netCDF4
import numpy as np

AREA = Path(__file__).resolve().parents[3] / "fesom_jax" / "data" / "mesh_pi" / "area.npy"


def main(rd):
    rd = Path(rd)
    a = np.load(AREA)[:, 0]  # surface node area [m2], pi
    bad = []
    for f in sorted(glob.glob(str(rd / "*.fesom.*.nc"))):
        ds = netCDF4.Dataset(f)
        for v in ds.variables:
            if v in ("time", "time_bounds"):
                continue
            x = np.asarray(ds[v][:], dtype=float)
            if not np.isfinite(x).all():
                bad.append((Path(f).name, int((~np.isfinite(x)).sum())))
    print("non-finite output:", bad or "none")

    def series(name):
        files = sorted(glob.glob(str(rd / f"{name}.fesom.*.nc")))
        return np.concatenate([np.asarray(netCDF4.Dataset(f)[name][:], dtype=float) for f in files])

    co2 = series("CO2f")  # mmolC/m2/d, positive downward
    gm = lambda x: (x * a).sum(-1) / a.sum()
    pg = (co2 * a).sum(-1) * 1e-3 * 12.011 * 365.0 / 1e15
    print("days:", co2.shape[0])
    print("CO2 flux [PgC/yr]   day1 %.3f  last %.3f  mean %.3f" % (pg[0], pg[-1], pg.mean()))
    for n in ("dpCO2s", "pCO2s", "NPPn", "NPPd", "GPPn", "GPPd", "aFe", "Hp", "xCO2atm", "sst", "a_ice"):
        x = series(n)
        print("%-8s day1 %.6g  last %.6g" % (n, gm(x[0]), gm(x[-1])))
    npp = (series("NPPn") + series("NPPd")) * a  # mmolC/m2/d
    print("NPP [PgC/yr]        day1 %.3f  last %.3f" % (npp[0].sum() * 1e-3 * 12.011 * 365 / 1e15,
                                                     npp[-1].sum() * 1e-3 * 12.011 * 365 / 1e15))


if __name__ == "__main__":
    main(sys.argv[1])
