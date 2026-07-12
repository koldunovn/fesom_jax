import numpy as np, xarray as xr, os
ds = xr.open_dataset("/home/a/a270088/paper_jax/data/drift.nc")
print("VARS:", list(ds.data_vars))
print("attrs:", dict(ds.attrs))
t = ds["time"].values; z = ds["z"].values
print(f"n_time={t.size}  time {t[0]:.3f}..{t[-1]:.3f}")
print(f"z ({z.size}): {np.round(z,1)}")
RHO0_CP = 1025.0*3990.0
V = float(ds.attrs["total_ocean_volume_m3"])
print(f"\ntot_vol={V:.4e}  RHO0_CP={RHO0_CP:.4e}")

def gap(base):
    j = ds[f"{base}_jax"].values; f = ds[f"{base}_fortran"].values
    both = np.where(np.isfinite(j)&np.isfinite(f))[0]
    if not both.size: return np.nan
    return j[both[-1]]-f[both[-1]]

print("\n--- final (2020) JAX - Fortran gaps ---")
for b in ["tbar","tbar700","sbar","ohc","sst","sss"]:
    print(f"  {b:8s}: {gap(b): .5f}")

# verify OHC == RHO0_CP*V*tbar
ohc = ds["ohc_jax"].values                    # ZJ
tbar = ds["tbar_jax"].values
pred = RHO0_CP*V*tbar/1e21
print("\n--- OHC vs RHO0_CP*V*Tbar (ZJ) ---")
print(f"  OHC_jax range      {np.nanmin(ohc):.2f}..{np.nanmax(ohc):.2f}")
print(f"  pred range         {np.nanmin(pred):.2f}..{np.nanmax(pred):.2f}")
print(f"  max|OHC-pred|      {np.nanmax(np.abs(ohc-pred)):.4f} ZJ   (should be ~0)")
print(f"  corr(OHC,Tbar)     {np.corrcoef(ohc[np.isfinite(ohc)],tbar[np.isfinite(ohc)])[0,1]:.8f}")

# OHC swing vs Tbar swing
print(f"\n  Tbar swing (full) {np.nanmax(tbar)-np.nanmin(tbar):.5f} degC")
print(f"  OHC swing         {np.nanmax(ohc)-np.nanmin(ohc):.2f} ZJ")
print(f"  1 ZJ = {1e21/(RHO0_CP*V):.6e} degC of Tbar")

# ---- salinity drift by level ----
sz = ds["sz_jax"].values          # [time, z]
fsz = ds["sz_fortran"].values
dS = sz - sz[0:1,:]               # anomaly vs start
dSf = fsz - fsz[0:1,:]
print("\n--- salinity drift by level (JAX): S(z,end)-S(z,start) ---")
end = dS[-1,:]
for k in range(z.size):
    fm = dSf[-1,k] if np.isfinite(dSf[-1,k]) else np.nan
    print(f"  z={z[k]:7.1f} m : dS_jax={end[k]:+.5f}   dS_for={fm:+.5f}")

# volume-weighted contribution to sbar drift, per level
sbar = ds["sbar_jax"].values
print(f"\n  sbar drift (vol-mean S): {sbar[-1]-sbar[0]:+.5f} psu")
print(f"  layer with max |dS|: z={z[np.nanargmax(np.abs(end))]:.1f} m  dS={end[np.nanargmax(np.abs(end))]:+.5f}")
ds.close()
