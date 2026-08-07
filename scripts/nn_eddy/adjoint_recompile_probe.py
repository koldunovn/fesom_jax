"""Is `test_core2_tke_forward_bit_identical` comparing flags, or comparing compilations?

The GPU failures in that test move a different subset of flags on every run, and setting
`freeze_gm_slope` — which `step.py:259` reads only inside `if gm_cfg is not None:`, a Python
branch the fixture never enters — moves the forward by one ulp. An inert flag cannot change a
computation, so the suspicion is that the test is measuring compilation nondeterminism: each
`make()` builds a fresh `jax.jit` closure, so it compares separately-compiled instances of a
semantically identical program.

This settles it without involving any flag at all. Compile THE SAME configuration several times
and compare the results. If they differ, the assertion is invalid on this platform regardless of
what the flags do, and no amount of fixing the fixture will help — only the assertion can change.

Run on BOTH platforms: the expectation is CPU stable, GPU not.
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
print(f"[platform] {jax.devices()}", flush=True)

from fesom_jax import ssh, surface_forcing                         # noqa: E402
from fesom_jax import step as stepmod                              # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig                   # noqa: E402
from fesom_jax.mesh import load_mesh                               # noqa: E402
from fesom_jax.params import Params                                # noqa: E402
from fesom_jax.phc_ic import phc_initial_state                     # noqa: E402
from fesom_jax.tke import TkeConfig                                # noqa: E402

MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT, YEAR, TKE_SEED, N_REPEAT = 1800.0, 1958, 1.0e-3, 5

mesh = load_mesh(MESH_DIR)
state = phc_initial_state(mesh, IC_DIR)
op = ssh.build_ssh_operator(mesh, dt=DT)
cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
sf0 = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
imask = np.asarray(mesh.node_iface_mask)
warm = dataclasses.replace(state, tke=jnp.asarray(np.where(imask, TKE_SEED, 0.0), jnp.float64))
print("[setup] done", flush=True)


def fwd_freshly_compiled(cfg, st):
    """Exactly what the test's `make()` does: a brand-new jit closure every call."""
    def loss(_):
        fin = stepmod.step(st, mesh, op, None, Params.defaults(), dt=DT, is_first_step=True,
                           step_forcing=sf0, forcing_static=cf.static, tke_cfg=TkeConfig(),
                           adjoint_cfg=cfg)
        return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)
    return float(jax.jit(loss)(jnp.asarray(0.0, jnp.float64)))


for label, st in (("TKE dormant (as the test runs today)", state),
                  ("TKE seeded (as the fix would run it)", warm)):
    print("\n" + "=" * 78)
    print(f"Same AdjointConfig(), compiled {N_REPEAT}x — {label}")
    print("=" * 78, flush=True)
    vals = []
    for i in range(N_REPEAT):
        v = fwd_freshly_compiled(AdjointConfig(), st)
        vals.append(v)
        print(f"  compile {i}: {v!r}", flush=True)
    uniq = sorted(set(vals))
    print(f"  distinct values: {len(uniq)}", flush=True)
    if len(uniq) > 1:
        spread = uniq[-1] - uniq[0]
        ulp = np.spacing(abs(uniq[0]))
        print(f"  spread {spread:.3e} = {spread/ulp:.1f} ulp", flush=True)
        print("  >>> the forward is NOT reproducible across compilations on this platform;", flush=True)
        print("  >>> `float(l_off) == float(l_on)` cannot hold here whatever the flags do.", flush=True)
    else:
        print("  >>> stable across recompiles: exact equality is a valid assertion here.", flush=True)

print("\n[done]", flush=True)
