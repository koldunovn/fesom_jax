"""Review gaps on the pi mesh: does an SST control actually work for test_adjoint_config.py?

The proposed fix swaps that file's ``k_ver`` control for a surface-temperature perturbation,
because ``freeze_mixing`` stop_gradients Kv/Av and therefore drives ANY mixing parameter to
exactly zero. Three of its assertions depend on the replacement behaving, and none of them has
been measured — only argued:

  G1  test_active_flag_changes_the_gradient — do mixing / convection / vert_visc / advection
      each still move the gradient under the new control? (freeze_pgf is excluded: its
      stop_gradient lives inside the `ale_cfg is not None` branch and the pi fixture is linfs.)
  G2  test_flags_compose — are g(mix), g(adv) and g(mix+adv) three distinct numbers? Under
      k_ver this failed because g(mix) and g(mix+adv) were both exactly 0.
  G3  test_frozen_mixing_gradient_stays_sane — is g(freeze_mixing) nonzero AND same-signed as
      the baseline?

Mirrors test_adjoint_config.py's fixture exactly: pi mesh, 8 steps, dt=100 s, PP mixing.
Cheap — CPU, seconds per gradient. Prints a table; asserts nothing.
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

from fesom_jax import forcing, ic, ssh                            # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig                  # noqa: E402
from fesom_jax.config import A_VER                                # noqa: E402
from fesom_jax.integrate import integrate                         # noqa: E402
from fesom_jax.mesh import load_mesh                              # noqa: E402
from fesom_jax.params import Params                               # noqa: E402

DT = 100.0
N = 8
K0 = jnp.asarray(1.0e-4, jnp.float64)

mesh = load_mesh()
op = ssh.build_ssh_operator(mesh, dt=DT)
stress = forcing.surface_stress(mesh)
st0 = ic.initial_state(mesh)
print(f"[setup] pi mesh, {N} steps, dt={DT}", flush=True)


def _run(state, p, cfg):
    fin = integrate(state, mesh, op, stress, n_steps=N, params=p, dt=DT, adjoint_cfg=cfg)
    return jnp.mean(fin.T[:, 0])


def g_kver(cfg):
    """The CURRENT control — for reference, to reproduce the failures here."""
    def loss(kver):
        return _run(st0, Params(k_ver=kver, a_ver=jnp.asarray(A_VER, jnp.float64)), cfg)
    return float(jax.jit(jax.grad(loss))(K0))


def g_sst(cfg):
    """The PROPOSED control — a uniform surface-temperature perturbation."""
    p = Params(k_ver=K0, a_ver=jnp.asarray(A_VER, jnp.float64))

    def loss(eps):
        return _run(dataclasses.replace(st0, T=st0.T.at[:, 0].add(eps)), p, cfg)
    return float(jax.jit(jax.grad(loss))(jnp.asarray(0.0, jnp.float64)))


def row(label, v, note=""):
    z = "  <-- EXACTLY ZERO" if v == 0.0 else ""
    print(f"  {label:<44s} {v:+.12e}{z}   {note}", flush=True)


print("\n" + "=" * 78)
print("Reference: reproduce the CURRENT failures with the k_ver control")
print("=" * 78, flush=True)
base_k = g_kver(AdjointConfig())
row("d/d(k_ver)  baseline", base_k)
mix_k = g_kver(AdjointConfig(freeze_mixing=True))
row("d/d(k_ver)  freeze_mixing", mix_k, "G3 needs this nonzero — expect 0")

print("\n" + "=" * 78)
print("G1 — does each pi-active flag move the gradient under the SST control?")
print("=" * 78, flush=True)
base = g_sst(AdjointConfig())
row("d/d(SST)  baseline", base)
vals = {}
for flag in ["freeze_mixing", "freeze_convection", "freeze_vert_visc", "freeze_advection",
             "freeze_pgf"]:
    v = g_sst(AdjointConfig(**{flag: True}))
    vals[flag] = v
    same = "SAME as baseline" if v == base else f"differs by {abs(v - base):.3e}"
    row(f"d/d(SST)  {flag}", v, same)

print("\n" + "=" * 78)
print("G2 — test_flags_compose: are mix, adv and mix+adv three distinct numbers?")
print("=" * 78, flush=True)
a = vals["freeze_mixing"]
b = vals["freeze_advection"]
ab = g_sst(AdjointConfig(freeze_mixing=True, freeze_advection=True))
row("d/d(SST)  freeze_mixing", a)
row("d/d(SST)  freeze_advection", b)
row("d/d(SST)  both", ab)
print(f"    ab != a ? {ab != a}      ab != b ? {ab != b}      G2 passes ? {ab != a and ab != b}",
      flush=True)

print("\n" + "=" * 78)
print("G3 — test_frozen_mixing_gradient_stays_sane: nonzero and same sign?")
print("=" * 78, flush=True)
print(f"    baseline {base:+.6e}   frozen-mixing {a:+.6e}", flush=True)
print(f"    nonzero ? {a != 0.0}      same sign ? {np.sign(a) == np.sign(base)}"
      f"      G3 passes ? {a != 0.0 and np.sign(a) == np.sign(base)}", flush=True)

print("\n[done]", flush=True)
