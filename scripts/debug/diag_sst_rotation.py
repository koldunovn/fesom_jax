import numpy as np, xarray as xr
ds = xr.open_dataset("/home/a/a270088/paper_jax/data/meanstate.nc")
lon = ds["lon"].values; lat = ds["lat"].values
d = ds["sst_jaxfor"].values
ok = np.isfinite(d)
lon2 = np.where(lon>180, lon-360, lon)          # [-180,180]
# crude area weight ~ cos(lat)
w = np.cos(np.deg2rad(lat))

# ---- hemisphere split by geographic longitude ----
def hm(mask,label):
    m = ok & mask
    print(f"  {label:34s}: {np.average(d[m],weights=w[m]):+.4f}  (n={m.sum()})")
print("=== hemisphere means (cos-lat weighted) ===")
hm(lon2<0, "Western hemi  (lon -180..0)")
hm(lon2>=0, "Eastern hemi  (lon 0..180)")
hm((lon2>=-90)&(lon2<90),  "centered on 0   (-90..90)")
hm((lon2<-90)|(lon2>=90),  "centered on 180 (90..270)")

# ---- wavenumber-1 zonal harmonic fit: d ~ a + A cos(lon - phi) ----
def k1_fit(mask, label):
    m = ok & mask
    lam = np.deg2rad(lon2[m]); y = d[m]; ww = w[m]
    # weighted least squares on [1, cos, sin]
    X = np.column_stack([np.ones_like(lam), np.cos(lam), np.sin(lam)])
    W = np.diag(ww)  # small enough? no -> use normal equations with weights
    XtW = X.T*ww
    beta = np.linalg.solve(XtW@X, XtW@y)
    a,b,c = beta
    A = np.hypot(b,c); phi = np.rad2deg(np.arctan2(c,b))
    yhat = X@beta
    ss_tot = np.sum(ww*(y-np.average(y,weights=ww))**2)
    ss_res = np.sum(ww*(y-yhat)**2)
    r2 = 1-ss_res/ss_tot
    print(f"  {label:22s}: A={A:.4f} degC, peak@lon={phi:+6.1f} (trough@{phi+180 if phi<0 else phi-180:+6.1f}), "
          f"mean={a:+.4f}, R^2(k1)={r2:.2f}")
print("\n=== wavenumber-1 zonal harmonic  d = mean + A cos(lon - phi) ===")
k1_fit(np.ones_like(ok,bool), "global")
k1_fit(np.abs(lat)<30, "|lat|<30")
k1_fit(np.abs(lat)<10, "|lat|<10 (equator)")
k1_fit(lat>30, "NH>30")
k1_fit(lat<-30, "SH<-30")

# ---- fine longitude profile (all lats, cos-weighted) ----
print("\n=== SST diff vs geographic longitude (20-deg bins, all lat, cos-weighted) ===")
for c in range(-180,180,20):
    m = ok & (lon2>=c)&(lon2<c+20)
    if m.any(): print(f"  lon {c:+4d}..{c+20:+4d}: {np.average(d[m],weights=w[m]):+.4f}")
ds.close()
