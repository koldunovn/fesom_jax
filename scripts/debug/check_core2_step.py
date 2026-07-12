"""Diagnostic: JAX CORE2 step-1 surface forcing + integrated T/S vs the C dump.

Run:  JAX_PLATFORMS=cpu <env python> scripts/debug/check_core2_step.py
"""
import numpy as np
from pathlib import Path

from fesom_jax import io_dump, surface_forcing, step as stepmod, ssh
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import core2_initial_state
from fesom_jax.verify import compare_column

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "data" / "step_dump_core2" / "core2_cdump.00000"
MESH = ROOT / "data" / "mesh_core2"
PROBES = [1001, 33778, 43828, 61202, 66921, 79663, 94122]
DT = 500.0
YEAR = 1958

print("load mesh + IC + op ...")
mesh = load_mesh(MESH)
state = core2_initial_state(mesh)
op = ssh.build_ssh_operator(mesh, dt=DT)

print("build forcing (year %d) ..." % YEAR)
cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
sf0 = cf.step_forcing(YEAR, 1, 0.0, 1)            # (year, day, sec, month)
fs = cf.static
print("a_ice>0 nodes:", int((np.asarray(fs.a_ice) > 0).sum()), "/", mesh.nod2D)

recs = io_dump.load_records(DUMP)

print("\n=== uvnode at IC (should be 0):", float(np.max(np.abs(np.asarray(state.uvnode)))))

# --- forcing gate (start-of-step state) ---
sfx = surface_forcing.compute_surface_fluxes(mesh, state, sf0, fs, dt=DT)
scal = {"heat_flux": sfx.heat_flux, "water_flux": sfx.water_flux,
        "virtual_salt": sfx.virtual_salt, "relax_salt": sfx.relax_salt}
print("\n=== STEP-1 SURFACE FORCING (sub 0) ===")
for fld, arr in scal.items():
    a = np.asarray(arr)
    worst = 0.0
    for g in PROBES:
        r = io_dump.find_record(recs, step=1, substep=0, probe_gid=g, field=fld)
        d = abs(float(a[g - 1]) - float(r.values[0]))
        worst = max(worst, d)
    print(f"  {fld:13s} max|Δ| over probes = {worst:.3e}")

sw = np.asarray(sfx.sw_3d)
worst_sw = 0.0
for g in PROBES:
    r = io_dump.find_record(recs, step=1, substep=0, probe_gid=g, field="sw_3d")
    res = compare_column(sw[g - 1], r, kind="map")
    worst_sw = max(worst_sw, res.max_abs)
print(f"  {'sw_3d':13s} max|Δ| over probes = {worst_sw:.3e}")

# --- integration gate: one eager step ---
print("\n=== eager step 1 (~32 s) ... ===")
st1 = stepmod.step(state, mesh, op, None, dt=DT, is_first_step=True,
                   step_forcing=sf0, forcing_static=fs)
for fld, arr in [("T", st1.T), ("S", st1.S)]:
    a = np.asarray(arr)
    worst = 0.0
    worst_g = None
    for g in PROBES:
        r = io_dump.find_record(recs, step=1, substep=15, probe_gid=g, field=fld)
        res = compare_column(a[g - 1], r, kind="scatter")
        if res.max_abs > worst:
            worst, worst_g = res.max_abs, g
    print(f"  post-step {fld}: max|Δ| over probes = {worst:.3e} (worst gid {worst_g})")

# also report the rest-state pi-style sanity: are there NaNs?
print("\nNaN check: T", bool(np.isnan(np.asarray(st1.T)).any()),
      " S", bool(np.isnan(np.asarray(st1.S)).any()),
      " uv", bool(np.isnan(np.asarray(st1.uv)).any()))
print("done.")
