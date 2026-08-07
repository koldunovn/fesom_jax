"""Where exactly does the sea-ice adjoint go non-finite?

The 2-day scan with `--ice exact` returns NaN at the first chunk boundary, but that only says
"somewhere in the first 48 steps". The mEVP unit tests differentiate the rheology successfully,
so the failure is not simply "the ice adjoint does not work" — something happens between a
working unit-level gradient and a dead assembled one, and this narrows down what.

Three questions, cheapest first:

1. **How many steps survive?** Take d(mean SST)/d(k_gm multiplier) through N assembled steps
   with the ice differentiated, for N = 1, 2, 4, 8, ... If N=1 is already NaN the problem is
   inside a single step. If it survives to N=8 and dies at N=16, it is accumulation.

2. **Which field goes first?** At the smallest failing N, walk the cotangent pytree and report
   every State field that is non-finite, in State field order. The ice velocity going NaN while
   temperature is still clean points at the rheology; everything going at once points at a
   shared upstream quantity.

3. **Is it the rheology or the rest of the ice?** Rerun the smallest failing N with the
   rheology's strain-rate dependence frozen (the Toyoda/Lyu approximation) and with the
   whole ice frozen, to bracket which part is responsible.

No truth trajectory and no cached weights: the loss is just mean SST, which keeps the compile
small and each N cheap. This is a bisection, not a measurement.
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fesom_jax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

if Path(fesom_jax.__file__).resolve().parents[1] != _ROOT:
    raise SystemExit(f"fesom_jax resolved to {fesom_jax.__file__}, not {_ROOT}")

from fesom_jax import ice_mevp, ssh, surface_forcing  # noqa: E402
from fesom_jax.calibrate import build_params  # noqa: E402
from fesom_jax.ice import IceConfig  # noqa: E402
from fesom_jax.integrate import _run_steps  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.params import Params  # noqa: E402
from fesom_jax.phc_ic import phc_initial_state  # noqa: E402
from fesom_jax.step import step  # noqa: E402
from horizon_scan import ALE_CFG, DT, GM_CFG, IC_DIR, MESH_DIR, TKE_CFG, seed_date  # noqa: E402


def nonfinite_fields(tree):
    """Every named State field with a non-finite entry, in field order, with the count."""
    out = []
    names = getattr(tree, "_fields", None)
    leaves = jax.tree.leaves(tree)
    labels = names if names and len(names) == len(leaves) else [f"leaf{i}" for i in range(len(leaves))]
    for lab, leaf in zip(labels, leaves):
        a = np.asarray(leaf)
        if a.dtype.kind != "f":
            continue
        bad = int((~np.isfinite(a)).sum())
        if bad:
            out.append((lab, bad, a.size))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--steps", default="1,2,4,8,16,32",
                    help="comma-separated step counts to try, ascending")
    ap.add_argument("--run-start-year", type=int, default=1958)
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",") if x.strip()]

    print(f"[setup] backend={jax.default_backend()}", flush=True)
    mesh = load_mesh(MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    with open(args.snap, "rb") as f:
        st0 = jax.device_put(pickle.load(f))
    yr, mo, dy = seed_date(args.snap, args.run_start_year)
    sst0 = np.asarray(phc_initial_state(mesh, IC_DIR).T[:, 0])
    cf = surface_forcing.build_surface_forcing(mesh, yr, sst_ic=sst0)
    dates = surface_forcing.dates_for_steps(yr, DT, max(steps), start_month=mo, start_day=dy)
    fs = cf.static
    kgm0 = float(Params.defaults().k_gm)
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])

    ICE_EXACT = IceConfig(whichEVP=1, adjoint_mode="exact")
    ICE_FROZEN = IceConfig(whichEVP=1, adjoint_mode="frozen")
    print(f"[setup] seed {yr}-{mo:02d}-{dy:02d}, dt={DT:.0f}s, mEVP subcycles="
          f"{ICE_EXACT.evp_rheol_steps}, delta_min={ICE_EXACT.delta_min:.1e}", flush=True)

    def build(n, ice_cfg):
        """(loss(m) -> scalar, run(state, m) -> final state) over n assembled steps."""
        sfs = cf.stack(dates[:n])

        def run(state, m):
            p = build_params({"k_gm": kgm0 * m, "redi_kmax": kgm0 * m})

            def body(carry, sf):
                return step(carry, mesh, op, None, p, dt=DT, is_first_step=False,
                            step_forcing=sf, forcing_static=fs, ice_cfg=ice_cfg,
                            gm_cfg=GM_CFG, tke_cfg=TKE_CFG, ale_cfg=ALE_CFG,
                            remat_blocks=True), None

            return _run_steps(body, state, xs=sfs, checkpoint=True, segments=0)

        def loss(m):
            fin = run(st0, m)
            return jnp.sum(jnp.where(wet0, fin.T[:, 0], 0.0)) / jnp.sum(wet0)

        return loss, run

    m0 = jnp.asarray(1.0, jnp.float64)
    inv_wet = 1.0 / float(jnp.sum(wet0))

    # ---- 1. how many assembled steps survive with the ice differentiated -------------
    print("\n--- ice DIFFERENTIATED: how many assembled steps survive? ---", flush=True)
    first_bad = None
    for n in steps:
        loss, run = build(n, ICE_EXACT)
        fwd = float(jax.jit(loss)(m0))
        gv = float(jax.jit(jax.grad(loss))(m0))
        ok = np.isfinite(gv)
        print(f"  n={n:3d}  loss={fwd:+.9e} {'finite' if np.isfinite(fwd) else 'NON-FINITE'}"
              f"   dL/dm={gv:+.6e} {'ok' if ok else '<-- NaN/inf'}", flush=True)
        if not ok:
            first_bad = n
            break

    if first_bad is None:
        print(f"  every n in {steps} gave a finite gradient with the ice differentiated. The "
              f"assembled ice adjoint is not the problem at these window lengths.", flush=True)
        return 0

    # ---- 2. which State field goes non-finite first ----------------------------------
    print(f"\n--- at the smallest failing n={first_bad}: what is non-finite? ---", flush=True)
    loss, run = build(first_bad, ICE_EXACT)
    out, vjp_fn = jax.vjp(lambda s: run(s, m0), st0)
    print("  FORWARD final state :", nonfinite_fields(out) or "all fields finite", flush=True)
    seed = jax.tree.map(jnp.zeros_like, out)
    seed = dataclasses.replace(seed, T=seed.T.at[:, 0].set(jnp.where(wet0, inv_wet, 0.0)))
    (cot,) = vjp_fn(seed)
    print("  BACKWARD cotangent  :", nonfinite_fields(cot) or "all fields finite", flush=True)
    print("  (field order is State's own; the first ice field to appear here is the lead)",
          flush=True)

    # ---- 3. bracket: rheology viscosity frozen, then the whole ice frozen -------------
    print(f"\n--- bracketing at n={first_bad} ---", flush=True)
    orig = ice_mevp.mevp_iterate
    try:
        def wrapped(*a, **k):
            k["freeze_viscosity"] = True
            return orig(*a, **k)
        ice_mevp.mevp_iterate = wrapped
        l2, _ = build(first_bad, ICE_EXACT)
        g = float(jax.jit(jax.grad(l2))(m0))
        print(f"  rheology viscosity frozen (Toyoda/Lyu): dL/dm={g:+.6e} "
              f"{'-> FIXED' if np.isfinite(g) else '-> still NaN'}", flush=True)
    finally:
        ice_mevp.mevp_iterate = orig
    l3, _ = build(first_bad, ICE_FROZEN)
    g = float(jax.jit(jax.grad(l3))(m0))
    print(f"  whole ice frozen (current default)     : dL/dm={g:+.6e} "
          f"{'-> FIXED, so the ice IS the source' if np.isfinite(g) else '-> still NaN, so the ice is NOT the source'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
