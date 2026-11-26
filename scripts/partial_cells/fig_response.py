"""Figure of the partial-cells validation (nereus env): the response to partial cells (partial minus full cells)
of Fortran FESOM2 and of fesom-jax, side by side, from the g2_fields.npz that g2_compare.py writes.

    /work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/partial_cells/fig_response.py <runs root>
"""
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh

import nereus

root = Path(sys.argv[1])
z = np.load(root / "g2_fields.npz")
lon, lat = z["lon"], z["lat"]
year = "zstar_year_1958" if "zstar_year_1958__bottom_T__F" in z else "zstar_month_1"
rows = [("zstar_day_1__bottom_T", "bottom-layer temperature, day 1", "°C"),
        (f"{year}__bottom_T", f"bottom-layer temperature, {year.split('_', 1)[1].replace('_', ' ')} mean", "°C"),
        (f"{year}__ssh", f"SSH, {year.split('_', 1)[1].replace('_', ' ')} mean", "m")]
proj = ccrs.Robinson()
fig, axes = plt.subplots(len(rows) + 1, 2, figsize=(9.0, 2.35 * (len(rows) + 1)), layout="constrained",
                         subplot_kw={"projection": proj})
interp = None
shift = z["bottom_shift"]
q = np.nanpercentile(np.abs(shift), 99)
_, _, interp = nereus.plot(shift, lon, lat, projection=proj, resolution=0.5, interpolator=interp, method="nearest",
                           influence_radius=120e3, cmap="RdBu_r", vmin=-q, vmax=q, land=True, colorbar=False,
                           ax=axes[0, 0], rasterized=True)
axes[0, 0].set_title("(a) node bottom depth, partial minus full cells (m)", loc="left", fontsize=8)
cb = fig.colorbar(next(c for c in axes[0, 0].collections if isinstance(c, QuadMesh)), ax=axes[0, 0],
                  orientation="horizontal", fraction=0.05, pad=0.03)
cb.ax.tick_params(labelsize=6)
axes[0, 1].set_visible(False)
letters = iter("bcdefg")
for r, (key, title, unit) in enumerate(rows, start=1):
    F, J = z[f"{key}__F"], z[f"{key}__J"]
    q = np.nanpercentile(np.abs(F), 98)
    for c, (v, who) in enumerate(((F, "Fortran FESOM2"), (J, "fesom-jax"))):
        ax = axes[r, c]
        _, _, interp = nereus.plot(v, lon, lat, projection=proj, resolution=0.5, interpolator=interp,
                                   method="nearest", influence_radius=120e3, cmap="RdBu_r", vmin=-q, vmax=q,
                                   land=True, colorbar=False, ax=ax, rasterized=True)
        ax.set_title(f"({next(letters)}) {who}: {title}", loc="left", fontsize=8)
        cb = fig.colorbar(next(cc for cc in ax.collections if isinstance(cc, QuadMesh)), ax=ax,
                          orientation="horizontal", fraction=0.05, pad=0.03)
        cb.set_label(f"partial minus full cells ({unit})", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
fig.savefig(root / "fig_partial_cells_response.png", dpi=160)
print("wrote", root / "fig_partial_cells_response.png")
