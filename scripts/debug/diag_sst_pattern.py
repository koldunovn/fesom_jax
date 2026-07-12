import numpy as np, xarray as xr
ds = xr.open_dataset("/home/a/a270088/paper_jax/data/meanstate.nc")
lon = ds["lon"].values; lat = ds["lat"].values
d = ds["sst_jaxfor"].values     # JAX - Fortran SST [nod2]
ok = np.isfinite(d)
print(f"nodes={ok.sum()}  mean={np.nanmean(d):+.4f}  std={np.nanstd(d):.4f}  min/max={np.nanmin(d):+.3f}/{np.nanmax(d):+.3f}")

# ---- zonal-mean by latitude band ----
print("\n--- zonal-mean SST diff by latitude ---")
for lo,hi in [(-90,-60),(-60,-40),(-40,-20),(-20,-5),(-5,5),(5,20),(20,40),(40,60),(60,90)]:
    m = ok & (lat>=lo) & (lat<hi)
    if m.any(): print(f"  {lo:+4d}..{hi:+4d}: {np.average(d[m]):+.4f}  (n={m.sum()})")

# ---- basin split (crude lon boxes) along the tropics (|lat|<20) ----
def box(latlo,lathi,lonlo,lonhi,label):
    lon2 = np.where(lon>180,lon-360,lon)
    m = ok & (lat>=latlo)&(lat<lathi)&(lon2>=lonlo)&(lon2<lonhi)
    if m.any(): print(f"  {label:28s}: {np.average(d[m]):+.4f}  (n={m.sum()})")
print("\n--- tropical basin boxes (|lat|<20) ---")
box(-20,20,-70,20,"Atlantic  (70W-20E)")
box(-20,20,50,100,"Indian    (50-100E)")
box(-20,20,120,180,"W Pacific (120-180E)")
box(-20,20,-180,-80,"E Pacific (180-80W)")

# ---- equatorial east-west profile (|lat|<5), binned in lon ----
print("\n--- equatorial (|lat|<5) east-west profile, 30-deg lon bins ---")
lon2 = np.where(lon>180,lon-360,lon)
eq = ok & (np.abs(lat)<5)
for c in range(-180,180,30):
    m = eq & (lon2>=c)&(lon2<c+30)
    if m.any(): print(f"  lon {c:+4d}..{c+30:+4d}: {np.average(d[m]):+.4f}  (n={m.sum()})")

# ---- eastern-boundary upwelling boxes ----
print("\n--- eastern-boundary upwelling regions ---")
box(-30,-10,-90,-70,"Peru-Humboldt (SE Pac)")
box(-30,-5,0,15,"Benguela (SE Atl)")
box(10,30,-30,-10,"Canary (NE Atl)")
box(20,40,-140,-115,"California (NE Pac)")

# ---- where are the extrema ----
print("\n--- strongest warm (JAX>F) nodes ---")
idx = np.argsort(np.where(ok,d,-1e9))[::-1][:8]
for i in idx: print(f"  lon={lon2[i]:+7.1f} lat={lat[i]:+6.1f}  d={d[i]:+.3f}")
print("--- strongest cool (JAX<F) nodes ---")
idx = np.argsort(np.where(ok,d,1e9))[:8]
for i in idx: print(f"  lon={lon2[i]:+7.1f} lat={lat[i]:+6.1f}  d={d[i]:+.3f}")
ds.close()
