#!/usr/bin/env python3
"""Climate check of the AB2 epsilon fix: C dt=1800 (both eps=0.1 fixes) vs
Fortran+PP dt=1800 (also eps=0.1). Monthly-mean surface fields, correlation +
bias by month. Confirms the fix doesn't degrade (should match-or-improve) the
climate now that the C matches Fortran's AB2 stabilization.

Run: /work/ab0995/a270088/mambaforge/envs/nereus/bin/python eps_climate_compare.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, netCDF4 as nc

C = "/work/ab0995/a270088/port/eps_val_1yr"
F = "/scratch/a/a270088/fortran_pp_dt1800"

def surf(path, var):
    d = nc.Dataset(path); a = d.variables[var]
    x = np.asarray(a[:, 0, :]) if a.ndim == 3 else np.asarray(a[:])  # (time,nz,n)->surf or (time,n)
    d.close(); return x

print("field  month |  corr   meanΔ(C−F)   |Δ|max   (C vs Fortran+PP, dt=1800, surface)")
for var, cf, ff in [("sst","sst","sst"), ("sss","sss","sss"),
                    ("ssh","ssh","ssh"), ("a_ice","a_ice","a_ice"), ("m_ice","m_ice","m_ice")]:
    try:
        c = surf(f"{C}/{cf}.fesom.1958.monthly.nc", var); f = surf(f"{F}/{ff}.fesom.1958.nc", var)
    except Exception as e:
        print(f"  {var}: skip ({e})"); continue
    nm = min(c.shape[0], f.shape[0])
    for m in range(nm):
        cc, ff_ = c[m], f[m]
        g = np.isfinite(cc) & np.isfinite(ff_) & (np.abs(ff_) < 1e30) & (np.abs(cc) < 1e30)
        if g.sum() < 100: continue
        corr = np.corrcoef(cc[g], ff_[g])[0,1]
        d = cc[g] - ff_[g]
        print(f"  {var:5s} {m+1:2d}   | {corr:6.4f}  {d.mean():+.4f}      {np.abs(d).max():.3f}")
    print()
