"""JZ.7 pgf-tail debug: dump the Shchepetkin intermediates for the worst element (240789,
nlev=34, tilted vnlev=[39,34,34]) where JAX pgf ≈ 29× the C. The pgf (substep 3) depends only
on the IC density + geometry, so reproduce it standalone and compare per-level terms to the C
z2_cdump."""
import numpy as np
from pathlib import Path
from fesom_jax.mesh import load_mesh
from fesom_jax import ale, eos, ice, io_dump, pgf
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.config import DENSITY_0, G

CORE2_MESH = Path("data/mesh_core2"); CORE2_IC = Path("data/ic_core2_dist16")
ORACLE = Path("/work/ab0995/a270088/port/zstar/z2_cdump/dump")
NOD, ELEM, NL = 126858, 244659, 48
E = 240789

mesh = load_mesh(CORE2_MESH)
state = phc_initial_state(mesh, CORE2_IC)
sst = np.asarray(state.T[:, 0])
st = ice.seed_ice(state, mesh, sst)
zbar3, Z3d = ale.live_geometry(mesh, st.hnode)
density, _, _ = eos.compute_pressure_bv(mesh, st.T, st.S, st.hnode, Z3d=Z3d)
# eos `density` IS ALREADY the anomaly (insitu − rho0). Test BOTH: the buggy double-subtraction
# (density − DENSITY_0 = insitu − 2·rho0) vs the correct anomaly (density).
px_bug, _ = pgf.pressure_force_shchepetkin(mesh, density - DENSITY_0, Z3d, st.helem)
px_fix, _ = pgf.pressure_force_shchepetkin(mesh, density, Z3d, st.helem)
px_bug = np.asarray(px_bug); px_fix = np.asarray(px_fix)
dmr = np.asarray(density)                            # the (correct) anomaly
px = px_bug; Z3d = np.asarray(Z3d)
# Σ gs (the shape-function gradient sum, analytically 0) — its FP residual drives the bug.
gsa = np.asarray(mesh.gradient_sca)
sumgsx = gsa[:, 0] + gsa[:, 1] + gsa[:, 2]
print(f"Σgs_x: |.|max={np.abs(sumgsx).max():.2e} median={np.median(np.abs(sumgsx)):.2e}")
f0, _ = io_dump.load_ale_dump(ORACLE, ["pgf_x"], step=1, n_nod=NOD, n_elem=ELEM)
cpx0 = f0["pgf_x"]
lm47 = np.asarray(mesh.elem_layer_mask)[:, :47]
db = np.abs(px_bug[:, :47] - cpx0)[lm47]; df = np.abs(px_fix[:, :47] - cpx0)[lm47]
print(f"pgf_x vs C  BUG(density-rho0): p99.9={np.percentile(db,99.9):.2e} max={db.max():.2e}")
print(f"pgf_x vs C  FIX(density):      p99.9={np.percentile(df,99.9):.2e} max={df.max():.2e}")
print(f"  worst elem 240789 Σgs_x={sumgsx[E]:.3e}  -rho0·Σgs={-DENSITY_0*sumgsx[E]:.3e}")

f, _ = io_dump.load_ale_dump(ORACLE, ["pgf_x"], step=1, n_nod=NOD, n_elem=ELEM)
cpx = f["pgf_x"][E]                                  # (47,) C pgf_x for the element

en = np.asarray(mesh.elem_nodes)[E]
gs = np.asarray(mesh.gradient_sca)[E]
hel = np.asarray(st.helem)[E]
nlev = int(np.asarray(mesh.nlevels)[E]); ule = int(np.asarray(mesh.ulevels)[E])
vnlev = np.asarray(mesh.nlevels_nod2D)[en]
R = dmr[en]                                          # (3, nl)
Zd = Z3d[en]                                         # (3, nl)
print(f"elem {E}: nlev={nlev} ule={ule} vnlev={vnlev}")
print(f"{'k':>3} {'drho_dx':>11} {'dz_dx':>11} {'helem':>9} {'pgf_jax':>12} {'pgf_C':>12} {'|Δ|':>10}")
for k in range(nlev - 1):
    drho_dx = gs[0] * R[0, k] + gs[1] * R[1, k] + gs[2] * R[2, k]
    dz_dx = gs[0] * Zd[0, k] + gs[1] * Zd[1, k] + gs[2] * Zd[2, k]
    pj = px[E, k]; pc = cpx[k]
    print(f"{k:3d} {drho_dx:11.3e} {dz_dx:11.3e} {hel[k]:9.3e} {pj:12.4e} {pc:12.4e} {abs(pj-pc):10.2e}")
# the vertex depths at the bottom (where vertices have different bottoms)
print("\nbottom-layer node mid-depths Z_3d_n[vertex, k] (k=29..33):")
for ni in range(3):
    print(f"  v{ni}(nlev={vnlev[ni]}): {Zd[ni, 29:34]}")
print("static mesh.Z[29:33]:", np.asarray(mesh.Z)[29:33])

# the IC T/S/density at the 3 vertices around the divergence (k=13..17): is there a spike?
Tv = np.asarray(state.T)[en]; Sv = np.asarray(state.S)[en]; Dv = dmr[en]
print("\nIC T[vertex, k=13..17]:")
for ni in range(3):
    print(f"  v{ni}: {Tv[ni, 13:18]}")
print("IC S[vertex, k=13..17]:")
for ni in range(3):
    print(f"  v{ni}: {Sv[ni, 13:18]}")
print("density-rho0[vertex, k=13..17]:")
for ni in range(3):
    print(f"  v{ni}: {Dv[ni, 13:18]}")
# horizontal density gradient component per vertex (gs·density) — which vertex spikes at k=15?
print("gs·(density) per vertex at k=14,15,16 (drho_dx = sum):")
for k in (14, 15, 16):
    terms = [gs[ni] * Dv[ni, k] for ni in range(3)]
    print(f"  k={k}: v0={terms[0]:.3e} v1={terms[1]:.3e} v2={terms[2]:.3e}  sum={sum(terms):.3e}")
