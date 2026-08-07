"""Why the adjoint-knob CORE2 tests fail — measure it, do not infer it.

Reading the code says three things. This script checks all three, because a session that has
already been wrong once about a gradient it only looked at does not get to reason its way to a
conclusion about another one.

  (1) On the TKE path ``Params.k_ver`` is structurally unused — ``step.py`` reads it only in
      the ``else:`` PP branch — so d(loss)/d(k_ver) is EXACTLY zero there, with or without any
      AdjointConfig flag. If so, every ``test_adjoint_config_tke.py`` test that differentiates
      w.r.t. ``kver`` is comparing 0 with 0.

  (2) ``freeze_mixing`` puts ``stop_gradient`` on Kv and Av. Over ONE step the loss (mean
      surface T) sees a mixing parameter ONLY through Kv/Av, so the gradient w.r.t. ANY mixing
      parameter — ``tke_c_k`` included — must also be exactly zero under that flag. If so,
      swapping the control variable for another mixing constant does not fix the tests; only a
      control that reaches T by some other route does.

  (3) ``ice_surface_step`` is called with the INPUT state and is never passed ``params``, so
      no ``Params`` field has a gradient path into the ice on step one. If so, "exact",
      "rheology" and "frozen" are bit-identical as gradients w.r.t. ``k_ver``, and the ice
      tests are unobservable by construction rather than broken.

The proposed control variable is a scalar SST perturbation: ``eps`` added to the surface layer
of T at wet nodes. It reaches the ocean through stratification AND the ice through ocean2ice,
which is exactly the pathway the flags are meant to cut, so it should separate everything.

CPU, CORE2, one step. Prints a table; asserts nothing.
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

print("[setup] loading CORE2 mesh + PHC IC + JRA55 forcing ...", flush=True)
mesh = load_mesh(MESH_DIR)
state = phc_initial_state(mesh, IC_DIR)
op = ssh.build_ssh_operator(mesh, dt=DT)
cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
fs = cf.static
sf0 = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
print("[setup] done", flush=True)


def _loss_from(fin):
    return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)


def _step(st, p, *, adj=None, ice=None, tke=None):
    return stepmod.step(st, mesh, op, None, p, dt=DT, is_first_step=True,
                        step_forcing=sf0, forcing_static=fs,
                        tke_cfg=tke, ice_cfg=ice, adjoint_cfg=adj)


def grad_wrt_param(field, *, adj=None, ice=None, tke=None):
    """d(mean surface T after one step)/d(that Params field), at its default value."""
    base = Params.defaults()
    x0 = getattr(base, field)

    def loss(x):
        p = dataclasses.replace(base, **{field: x})
        return _loss_from(_step(state, p, adj=adj, ice=ice, tke=tke))

    return float(jax.jit(jax.grad(loss))(x0))


def grad_wrt_sst(*, adj=None, ice=None, tke=None):
    """d(mean surface T after one step)/d(eps), eps a uniform SST bump at wet surface nodes."""
    p = Params.defaults()

    def loss(eps):
        st = dataclasses.replace(state, T=state.T.at[:, 0].add(jnp.where(wet0, eps, 0.0)))
        return _loss_from(_step(st, p, adj=adj, ice=ice, tke=tke))

    return float(jax.jit(jax.grad(loss))(jnp.asarray(0.0, jnp.float64)))


def row(label, value, note=""):
    z = "  <-- EXACTLY ZERO" if value == 0.0 else ""
    print(f"  {label:<46s} {value:+.12e}{z}   {note}", flush=True)


# ---------------------------------------------------------------- claim 1 + 2: the TKE path
print("\n" + "=" * 78)
print("TKE path (tke_cfg=TkeConfig(), no ice) — which control variables are alive?")
print("=" * 78, flush=True)
tke_cfg = TkeConfig()
row("d/d(k_ver)      AdjointConfig()", grad_wrt_param("k_ver", adj=AdjointConfig(), tke=tke_cfg),
    "claim 1: k_ver unused on the TKE path")
row("d/d(tke_c_k)    AdjointConfig()", grad_wrt_param("tke_c_k", adj=AdjointConfig(), tke=tke_cfg),
    "the real TKE knob")
row("d/d(tke_c_k)    freeze_mixing=True",
    grad_wrt_param("tke_c_k", adj=AdjointConfig(freeze_mixing=True), tke=tke_cfg),
    "claim 2: mixing params die under freeze_mixing too")
row("d/d(SST eps)    AdjointConfig()", grad_wrt_sst(adj=AdjointConfig(), tke=tke_cfg),
    "the proposed control")

print("\n  per-flag, control = SST eps (does each flag reach the TKE path?):", flush=True)
g_base = grad_wrt_sst(adj=AdjointConfig(), tke=tke_cfg)
for flag in TKE_PATH_FLAGS:
    g = grad_wrt_sst(adj=AdjointConfig(**{flag: True}), tke=tke_cfg)
    same = "SAME as baseline" if g == g_base else f"differs by {abs(g - g_base):.3e}"
    row(f"d/d(SST eps)    {flag}=True", g, same)

# ------------------------------------------------------------------- claim 3: the ice modes
print("\n" + "=" * 78)
print("Ice path (ice_cfg, PP mixing) — is the ice adjoint mode observable at all?")
print("=" * 78, flush=True)
for control, fn in (("k_ver", lambda ice: grad_wrt_param("k_ver", ice=ice)),
                    ("SST eps", lambda ice: grad_wrt_sst(ice=ice))):
    print(f"\n  control = {control}:", flush=True)
    vals = {}
    for mode in ("exact", "rheology", "frozen"):
        vals[mode] = fn(IceConfig(whichEVP=1, adjoint_mode=mode))
        row(f"d/d({control})  adjoint_mode={mode!r}", vals[mode])
    print(f"    rheology == frozen ? {vals['rheology'] == vals['frozen']}", flush=True)
    print(f"    rheology == exact  ? {vals['rheology'] == vals['exact']}", flush=True)

print("\n[done]", flush=True)
