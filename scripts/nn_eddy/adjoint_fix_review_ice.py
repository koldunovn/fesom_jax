"""The ice review, redone with ICE ACTUALLY PRESENT.

The previous pass measured `a_ice nonzero at 0 nodes`. `test_ice_adjoint_mode.py` builds its
state with a bare `phc_initial_state(mesh, IC_DIR)`, which does NOT seed sea ice —
`seed_sea_ice` is a parameter of `cold_start_state`, a different function. `phc_ic.py:503` states
the rule: "Every validated entry point does `ice.seed_ice(phc_initial_state(...), mesh, sst)`",
and every other ice test file does exactly that (`test_ice_step.py`, `test_ice_ic.py`,
`test_ale_zstar.py`, `test_gradient_sharded.py`). This one file omits it, so it has been testing
the sea-ice adjoint on an ice-free ocean.

That makes the earlier ice numbers unusable as evidence about the modes, and it means the
proposed "change the control variable" fix was aimed at the wrong thing. Everything is remeasured
here with the ice seeded, at the two controls that matter, plus the solver's own boundary.

What to look for:
  * with ice present, do "exact", "rheology" and "frozen" finally separate?
  * is the SST control enough, or is an ocean-velocity control still needed to reach the EVP?
  * at the mEVP boundary, does `adjoint_subcycles=0` give exactly zero against a live default?
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

import fesom_jax

_got = Path(fesom_jax.__file__).resolve()
if ROOT.resolve() not in _got.parents:
    sys.exit(f"FATAL: fesom_jax imported from {_got}, not from {ROOT}")
print(f"[import] fesom_jax <- {_got}", flush=True)

from fesom_jax import ice as icemod                                # noqa: E402
from fesom_jax import ice_mevp, ssh, surface_forcing               # noqa: E402
from fesom_jax import step as stepmod                              # noqa: E402
from fesom_jax.ice import IceConfig                                # noqa: E402
from fesom_jax.mesh import load_mesh                               # noqa: E402
from fesom_jax.params import Params                                # noqa: E402
from fesom_jax.phc_ic import phc_initial_state                     # noqa: E402

MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0
YEAR = 1958
MODES = ("exact", "rheology", "frozen")

print("[setup] loading CORE2 ...", flush=True)
mesh = load_mesh(MESH_DIR)
bare = phc_initial_state(mesh, IC_DIR)
state = icemod.seed_ice(bare, mesh, bare.T[:, 0])          # <-- the missing line
op = ssh.build_ssh_operator(mesh, dt=DT)
cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(bare.T[:, 0]))
fs = cf.static
sf0 = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
print(f"[setup] BARE  a_ice nonzero={int(jnp.count_nonzero(bare.a_ice))}  "
      f"max={float(jnp.max(bare.a_ice)):.3f}", flush=True)
print(f"[setup] SEEDED a_ice nonzero={int(jnp.count_nonzero(state.a_ice))}  "
      f"max={float(jnp.max(state.a_ice)):.3f}  "
      f"m_ice max={float(jnp.max(state.m_ice)):.3f}", flush=True)
print("[setup] done", flush=True)


def row(label, v, note=""):
    z = "  <-- EXACTLY ZERO" if v == 0.0 else ""
    print(f"  {label:<46s} {v:+.14e}{z}   {note}", flush=True)


def _loss(fin):
    return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)


def _step(st, p, ice):
    return stepmod.step(st, mesh, op, None, p, dt=DT, is_first_step=True,
                        step_forcing=sf0, forcing_static=fs, ice_cfg=ice)


def g_kver(ice, st):
    base = Params.defaults()

    def loss(x):
        return _loss(_step(st, dataclasses.replace(base, k_ver=x), ice))
    return float(jax.jit(jax.grad(loss))(base.k_ver))


def g_sst(ice, st):
    def loss(eps):
        s = dataclasses.replace(st, T=st.T.at[:, 0].add(jnp.where(wet0, eps, 0.0)))
        return _loss(_step(s, Params.defaults(), ice))
    return float(jax.jit(jax.grad(loss))(jnp.asarray(0.0, jnp.float64)))


def g_uv(ice, st):
    def loss(eps):
        return _loss(_step(dataclasses.replace(st, uv=st.uv + eps), Params.defaults(), ice))
    return float(jax.jit(jax.grad(loss))(jnp.asarray(0.0, jnp.float64)))


for cname, fn in (("k_ver", g_kver), ("SST eps", g_sst), ("ocean uv eps", g_uv)):
    print("\n" + "=" * 78)
    print(f"Ice modes WITH ICE SEEDED — control = {cname}")
    print("=" * 78, flush=True)
    v = {}
    for m in MODES:
        v[m] = fn(IceConfig(whichEVP=1, adjoint_mode=m), state)
        row(f"d/d({cname})  adjoint_mode={m!r}", v[m])
    print(f"    rheology == frozen ? {v['rheology'] == v['frozen']}", flush=True)
    print(f"    rheology == exact  ? {v['rheology'] == v['exact']}", flush=True)
    print(f"    all three distinct ? {len(set(v.values())) == 3}", flush=True)
    if v['frozen'] != 0.0:
        print(f"    rheology vs frozen: {abs(v['rheology']-v['frozen'])/abs(v['frozen']):.3e} "
              f"relative", flush=True)

# ------------------------------------------------- the knob at the solver's own boundary
print("\n" + "=" * 78)
print("mEVP boundary — adjoint_subcycles=0 vs default, ice seeded")
print("=" * 78, flush=True)
cfg_ice = IceConfig(whichEVP=1)
zero = jnp.zeros(mesh.nod2D, jnp.float64)


def g_mevp(adjoint_subcycles):
    kw = {} if adjoint_subcycles < 0 else {"adjoint_subcycles": adjoint_subcycles}

    def loss(eps):
        u, v_, *_ = ice_mevp.mevp_dynamics(
            cfg_ice, mesh, a_ice=state.a_ice, m_ice=state.m_ice, m_snow=state.m_snow,
            u_ice=state.u_ice, v_ice=state.v_ice,
            sigma11=state.sigma11, sigma12=state.sigma12, sigma22=state.sigma22,
            srfoce_u=zero + eps, srfoce_v=zero, elevation=zero,
            stress_ax=zero + 0.05, stress_ay=zero, boundary_node=None, exch=None, **kw)
        return jnp.sum(u) + jnp.sum(v_)
    return float(jax.jit(jax.grad(loss))(jnp.asarray(0.0, jnp.float64)))


try:
    gf = g_mevp(-1)
    row("d(sum u_ice)/d(srfoce_u)  default (all sweeps)", gf, "expect nonzero")
    gz = g_mevp(0)
    row("d(sum u_ice)/d(srfoce_u)  adjoint_subcycles=0", gz, "expect EXACTLY 0")
    print(f"    knob observable at the solver boundary ? {gf != gz}", flush=True)
except Exception as e:                                             # noqa: BLE001
    print(f"    mEVP direct call failed: {type(e).__name__}: {e}", flush=True)

print("\n[done]", flush=True)
