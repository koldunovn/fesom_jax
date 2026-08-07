"""Review gaps on CORE2: the three parts of the proposed fix that were argued, not measured.

  G4  ``freeze_tke_state`` at TWO steps. At one step it moved the gradient by 1.11e-16 — one
      ulp, i.e. nothing. The claim is that this is structural: the flag stop_gradients
      ``tke_new``, which is the carry to the NEXT step, so a one-step fixture produces it and
      never consumes it. If that reading is right, two steps must give a real difference. If it
      is wrong, the flag is untestable and should be xfailed with a reason instead.

  G5  The mEVP knob at ITS OWN level. ``rheology`` could not be separated from ``exact`` through
      a full step by any control tried (SST: identical; ocean velocity: identical to *frozen*).
      But ``ice_mevp.py:279-283`` stop_gradients the entire carried state when
      ``adjoint_subcycles=0``, so at the solver's own boundary the gradient should be EXACTLY
      zero against a nonzero default. That is the honest place to test this knob.

  G6  Does seeding the TKE break forward bit-identity? The forward-identity tests currently run
      against a DORMANT closure, so they have never actually exercised the TKE code path. Waking
      it could surface a real forward-moving flag that was masked. This is the one review check
      that could turn up a genuine library bug rather than a test defect, so it runs over all
      nine flags.

Prints tables; asserts nothing.
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

from fesom_jax import ice_mevp, ssh, surface_forcing               # noqa: E402
from fesom_jax import step as stepmod                              # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig                   # noqa: E402
from fesom_jax.ice import IceConfig                                # noqa: E402
from fesom_jax.mesh import load_mesh                               # noqa: E402
from fesom_jax.params import Params                                # noqa: E402
from fesom_jax.phc_ic import phc_initial_state                     # noqa: E402
from fesom_jax.tke import TkeConfig                                # noqa: E402

MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0
YEAR = 1958
TKE_SEED = 1.0e-3

print("[setup] loading CORE2 ...", flush=True)
mesh = load_mesh(MESH_DIR)
state = phc_initial_state(mesh, IC_DIR)
op = ssh.build_ssh_operator(mesh, dt=DT)
cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
fs = cf.static
dates = surface_forcing.dates_for_steps(YEAR, DT, 2)
sf0 = cf.step_forcing(*dates[0])
sf1 = cf.step_forcing(*dates[1])
wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
imask = np.asarray(mesh.node_iface_mask)
state_warm = dataclasses.replace(
    state, tke=jnp.asarray(np.where(imask, TKE_SEED, 0.0), jnp.float64))
print(f"[setup] ice present: a_ice nonzero at {int(jnp.count_nonzero(state.a_ice))} nodes, "
      f"max={float(jnp.max(state.a_ice)):.3f}", flush=True)
print("[setup] done", flush=True)


def row(label, v, note=""):
    z = "  <-- EXACTLY ZERO" if v == 0.0 else ""
    print(f"  {label:<46s} {v:+.12e}{z}   {note}", flush=True)


def _loss(fin):
    return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)


# ------------------------------------------------------------------- G4: two steps
print("\n" + "=" * 78)
print("G4 — freeze_tke_state over TWO steps (one step gave 1.11e-16 = one ulp)")
print("=" * 78, flush=True)


def g_sst_nsteps(cfg, nsteps):
    def loss(eps):
        st = dataclasses.replace(
            state_warm, T=state_warm.T.at[:, 0].add(jnp.where(wet0, eps, 0.0)))
        for sf in ([sf0] if nsteps == 1 else [sf0, sf1]):
            st = stepmod.step(st, mesh, op, None, Params.defaults(), dt=DT,
                              is_first_step=(sf is sf0), step_forcing=sf, forcing_static=fs,
                              tke_cfg=TkeConfig(), adjoint_cfg=cfg)
        return _loss(st)
    return float(jax.jit(jax.grad(loss))(jnp.asarray(0.0, jnp.float64)))


for n in (1, 2):
    b = g_sst_nsteps(AdjointConfig(), n)
    f = g_sst_nsteps(AdjointConfig(freeze_tke_state=True), n)
    row(f"{n} step(s)  baseline", b)
    row(f"{n} step(s)  freeze_tke_state", f,
        "SAME" if f == b else f"differs by {abs(f - b):.3e}")

# --------------------------------------------------------- G5: the mEVP's own boundary
print("\n" + "=" * 78)
print("G5 — the mEVP knob tested where it acts (adjoint_subcycles=0 vs default)")
print("=" * 78, flush=True)
cfg_ice = IceConfig(whichEVP=1)


def g_mevp(adjoint_subcycles):
    """d(sum u_ice)/d(eps) for a bump on the ocean surface velocity fed to the solver."""
    kw = {} if adjoint_subcycles < 0 else {"adjoint_subcycles": adjoint_subcycles}
    su = jnp.zeros(mesh.nod2D, jnp.float64)

    def loss(eps):
        u, v, *_ = ice_mevp.mevp_dynamics(
            cfg_ice, mesh, a_ice=state.a_ice, m_ice=state.m_ice, m_snow=state.m_snow,
            u_ice=state.u_ice, v_ice=state.v_ice,
            sigma11=state.sigma11, sigma12=state.sigma12, sigma22=state.sigma22,
            srfoce_u=su + eps, srfoce_v=su, elevation=jnp.zeros(mesh.nod2D, jnp.float64),
            stress_ax=jnp.zeros(mesh.nod2D, jnp.float64),
            stress_ay=jnp.zeros(mesh.nod2D, jnp.float64),
            boundary_node=None, exch=None, **kw)
        return jnp.sum(u) + jnp.sum(v)
    return float(jax.jit(jax.grad(loss))(jnp.asarray(0.0, jnp.float64)))


try:
    g_full = g_mevp(-1)
    row("d(sum u_ice)/d(srfoce_u)  default (all sweeps)", g_full, "expect nonzero")
    g_none = g_mevp(0)
    row("d(sum u_ice)/d(srfoce_u)  adjoint_subcycles=0", g_none, "expect EXACTLY 0")
    print(f"    knob is observable at the solver boundary ? {g_full != g_none}", flush=True)
except Exception as e:                                            # noqa: BLE001
    print(f"    mEVP direct call failed: {type(e).__name__}: {e}", flush=True)

# ------------------------------------------- G6: does waking the closure move the forward?
print("\n" + "=" * 78)
print("G6 — with the TKE AWAKE, is the forward still bit-identical under every flag?")
print("     (the current tests ran a dormant closure, so this path is UNTESTED)")
print("=" * 78, flush=True)


def fwd(cfg, st):
    def loss(_):
        fin = stepmod.step(st, mesh, op, None, Params.defaults(), dt=DT, is_first_step=True,
                           step_forcing=sf0, forcing_static=fs, tke_cfg=TkeConfig(),
                           adjoint_cfg=cfg)
        return _loss(fin)
    return float(jax.jit(loss)(jnp.asarray(0.0, jnp.float64)))


ref = fwd(AdjointConfig(), state_warm)
print(f"  reference forward (seeded tke): {ref!r}", flush=True)
bad = []
for flag in AdjointConfig._fields:
    v = fwd(AdjointConfig(**{flag: True}), state_warm)
    ok = (v == ref)
    if not ok:
        bad.append((flag, v))
    print(f"  {flag:<26s} {v!r}   {'identical' if ok else '*** MOVED ***'}", flush=True)
print(f"\n  flags that move the forward with the closure awake: "
      f"{[f for f, _ in bad] if bad else 'none'}", flush=True)

print("\n[done]", flush=True)
