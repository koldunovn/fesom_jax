"""Does the PROPOSED fix actually make the adjoint-knob tests measure something?

Probe 1 (``adjoint_test_probe.py``) established why they fail. This one checks that the
replacement fixtures work, because "the diagnosis is right" and "the fix is right" are two
different claims and only the second one ships.

Two changes are on trial.

  A. **Seed the TKE.** The CORE2 fixture starts from ``phc_initial_state``, whose ``tke`` is
     zero (``state.py:81`` — the C cold-start ``calloc`` mirror). ``KappaH`` is proportional to
     the turbulent kinetic energy, so a cold column emits ``Kv/Av = 0`` and EVERY derivative
     through the closure is exactly zero — measured, both platforms. The same test file already
     seeds ``tke = 1e-3`` in its closure-level fixture for exactly this reason. Here: seed the
     step-level state the same way and check that ``d/d(tke_c_k)`` comes alive and that the
     TKE-path flags separate.

  B. **Perturb the ocean velocity for the ice.** ``ice_surface_step`` is handed the input state
     and no ``Params``, so no parameter reaches the ice at step one. An SST bump reaches the ice
     THERMODYNAMICS but not the momentum solve — ``_evp(...)`` takes ``a_ice/m_ice/u_ice/sigma``
     from the previous step plus ``srfoce_u/srfoce_v/elevation``, and no temperature — so SST
     can separate "frozen" from the other two but must leave "exact" and "rheology" identical,
     since the only thing between those two IS the momentum solve. An ocean-VELOCITY bump goes
     straight into ``srfoce_u/v`` and therefore into the EVP. Both are measured below.

Prints a table; asserts nothing.
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

from fesom_jax import ssh, surface_forcing                        # noqa: E402
from fesom_jax import step as stepmod                             # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig                  # noqa: E402
from fesom_jax.ice import IceConfig                               # noqa: E402
from fesom_jax.mesh import load_mesh                              # noqa: E402
from fesom_jax.params import Params                               # noqa: E402
from fesom_jax.phc_ic import phc_initial_state                    # noqa: E402
from fesom_jax.tke import TkeConfig                               # noqa: E402

MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0
YEAR = 1958
TKE_PATH_FLAGS = ["freeze_mixing", "freeze_stratification", "freeze_convection",
                  "freeze_tke_state", "freeze_vert_visc"]
TKE_SEED = 1.0e-3          # the value the closure-level fixture in the same test file uses

print("[setup] loading CORE2 mesh + PHC IC + JRA55 forcing ...", flush=True)
mesh = load_mesh(MESH_DIR)
state = phc_initial_state(mesh, IC_DIR)
op = ssh.build_ssh_operator(mesh, dt=DT)
cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
fs = cf.static
sf0 = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])

# A wet-interface TKE seed, the closure fixture's pattern lifted to the full column.
imask = np.asarray(mesh.node_iface_mask)
state_warm = dataclasses.replace(
    state, tke=jnp.asarray(np.where(imask, TKE_SEED, 0.0), jnp.float64))
print(f"[setup] cold tke: max={float(jnp.max(state.T*0 + jnp.max(state.tke))):.3e}   "
      f"seeded tke: max={float(jnp.max(state_warm.tke)):.3e}  "
      f"nonzero={int(jnp.count_nonzero(state_warm.tke))}", flush=True)
print("[setup] done", flush=True)


def _loss_from(fin):
    return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)


def _step(st, p, *, adj=None, ice=None, tke=None):
    return stepmod.step(st, mesh, op, None, p, dt=DT, is_first_step=True,
                        step_forcing=sf0, forcing_static=fs,
                        tke_cfg=tke, ice_cfg=ice, adjoint_cfg=adj)


def grad_param(field, st, *, adj=None, ice=None, tke=None):
    base = Params.defaults()

    def loss(x):
        return _loss_from(_step(st, dataclasses.replace(base, **{field: x}),
                                adj=adj, ice=ice, tke=tke))

    return float(jax.jit(jax.grad(loss))(getattr(base, field)))


def grad_sst(st, *, adj=None, ice=None, tke=None):
    def loss(eps):
        s = dataclasses.replace(st, T=st.T.at[:, 0].add(jnp.where(wet0, eps, 0.0)))
        return _loss_from(_step(s, Params.defaults(), adj=adj, ice=ice, tke=tke))

    return float(jax.jit(jax.grad(loss))(jnp.asarray(0.0, jnp.float64)))


def grad_uv(st, *, adj=None, ice=None, tke=None):
    """d/d(eps) for a uniform bump on the ocean velocity — reaches srfoce_u/v ⇒ the EVP."""
    def loss(eps):
        s = dataclasses.replace(st, uv=st.uv + eps)
        return _loss_from(_step(s, Params.defaults(), adj=adj, ice=ice, tke=tke))

    return float(jax.jit(jax.grad(loss))(jnp.asarray(0.0, jnp.float64)))


def row(label, v, note=""):
    z = "  <-- EXACTLY ZERO" if v == 0.0 else ""
    print(f"  {label:<44s} {v:+.12e}{z}   {note}", flush=True)


# --------------------------------------------------------------- A. seeded TKE
print("\n" + "=" * 78)
print("A. TKE path with a SEEDED tke — does the closure wake up?")
print("=" * 78, flush=True)
tke_cfg = TkeConfig()
row("d/d(tke_c_k)  cold tke   AdjointConfig()", grad_param("tke_c_k", state, tke=tke_cfg),
    "the failing status quo")
row("d/d(tke_c_k)  SEEDED     AdjointConfig()", grad_param("tke_c_k", state_warm, tke=tke_cfg),
    "must be nonzero for the fix to work")
row("d/d(tke_c_k)  SEEDED     freeze_mixing",
    grad_param("tke_c_k", state_warm, adj=AdjointConfig(freeze_mixing=True), tke=tke_cfg),
    "expect 0: freeze_mixing cuts Kv/Av, the only route for a mixing param")

print("\n  control = SST eps, SEEDED tke, per flag:", flush=True)
g0 = grad_sst(state_warm, adj=AdjointConfig(), tke=tke_cfg)
row("d/d(SST eps)  SEEDED     AdjointConfig()", g0, "baseline")
for flag in TKE_PATH_FLAGS:
    g = grad_sst(state_warm, adj=AdjointConfig(**{flag: True}), tke=tke_cfg)
    row(f"d/d(SST eps)  SEEDED     {flag}", g,
        "SAME as baseline" if g == g0 else f"differs by {abs(g - g0):.3e}")

# ------------------------------------------------------------- B. ice controls
print("\n" + "=" * 78)
print("B. Ice modes — which control separates all three?")
print("=" * 78, flush=True)
for name, fn in (("SST eps", grad_sst), ("ocean uv eps", grad_uv)):
    print(f"\n  control = {name}:", flush=True)
    v = {}
    for mode in ("exact", "rheology", "frozen"):
        v[mode] = fn(state, ice=IceConfig(whichEVP=1, adjoint_mode=mode))
        row(f"d/d({name})  adjoint_mode={mode!r}", v[mode])
    print(f"    rheology == frozen ? {v['rheology'] == v['frozen']}", flush=True)
    print(f"    rheology == exact  ? {v['rheology'] == v['exact']}", flush=True)
    print(f"    all three distinct ? "
          f"{len({v['exact'], v['rheology'], v['frozen']}) == 3}", flush=True)

print("\n[done]", flush=True)
