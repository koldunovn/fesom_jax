#!/usr/bin/env python
"""All-ON triple smoke (zstar + TKE + mEVP) — Phase 9c, JM.5 (OPTIONAL, JAX-first).

Runs the FULL CORE2 model with ALL THREE Phase-9 options live simultaneously:
``ale_cfg=AleConfig()`` (zstar) + ``tke_cfg=TkeConfig()`` (TKE mixing) +
``ice_cfg=IceConfig(whichEVP=1)`` (mEVP) + GM/Redi. ⚠️ This combination is **untested in the C**
(the C deliberately validated single knobs; zstar+TKE is the only C-validated pair) — so this is
a SMOKE gate only: the step lowers, runs, and stays stable/finite/physical for a few steps. It is
explicitly NOT a fidelity gate (no oracle exists). A clean run shows the three seams compose
without NaN / blow-up / column collapse.

Usage:  python scripts/core2_all_on_smoke.py --steps 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import core2_forcing, ice, ssh
from fesom_jax import step as stepmod
from fesom_jax.ale import AleConfig
from fesom_jax.gm import GMConfig
from fesom_jax.ice import IceConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import cold_start_state
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist16"
DT = 1800.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--year", type=int, default=1958)
    args = ap.parse_args()
    print(f"[setup] backend={jax.default_backend()} devices={jax.devices()}", flush=True)

    mesh = load_mesh(MESH_DIR)
    state = cold_start_state(mesh, IC_DIR)                 # PHC IC + seeded sea-ice IC (canonical)
    sst = np.asarray(state.T[:, 0])
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, args.year, sst_ic=sst)
    dates = core2_forcing.dates_for_steps(args.year, DT, args.steps)
    cfgs = dict(gm_cfg=GMConfig(), tke_cfg=TkeConfig(), ice_cfg=IceConfig(whichEVP=1),
                ale_cfg=AleConfig())     # zstar + TKE + mEVP + GM, all live
    surf = jnp.asarray(np.asarray(mesh.node_layer_mask)[:, 0])
    nlm = jnp.asarray(mesh.node_layer_mask)

    @jax.jit
    def diag(s):
        fin = (jnp.isfinite(s.T).all() & jnp.isfinite(s.S).all() & jnp.isfinite(s.uv).all()
               & jnp.isfinite(s.a_ice).all() & jnp.isfinite(s.u_ice).all()
               & jnp.isfinite(s.hnode).all() & jnp.isfinite(s.tke).all())
        return dict(fin=fin, ssh=jnp.max(jnp.where(surf, jnp.abs(s.eta_n), -jnp.inf)),
                    vel=jnp.max(jnp.abs(s.uv)), uice=jnp.max(jnp.sqrt(s.u_ice**2 + s.v_ice**2)),
                    mice=jnp.max(s.m_ice), hmin=jnp.min(jnp.where(nlm, s.hnode, jnp.inf)),
                    sst_min=jnp.min(jnp.where(surf, s.T[:, 0], jnp.inf)))

    print(f"[run] zstar + TKE + mEVP + GM, {args.steps} steps (dt={DT}) — SMOKE (no oracle)",
          flush=True)
    ok = True
    for i in range(args.steps):
        sf = cf.step_forcing(*dates[i])
        ts = time.time()
        state = stepmod.step_jit(state, mesh, op, None, dt=DT, is_first_step=(i == 0),
                                 step_forcing=sf, forcing_static=cf.static, **cfgs)
        d = jax.device_get(diag(state))
        bad = (not bool(d["fin"]) or float(d["ssh"]) >= 5.0 or float(d["vel"]) >= 3.0
               or float(d["mice"]) >= 20.0 or float(d["hmin"]) <= 0.0)
        print(f"  s{i+1}  fin={int(d['fin'])} |SSH|={float(d['ssh']):.2e} |vel|={float(d['vel']):.2e} "
              f"|uv_ice|={float(d['uice']):.2e} m_ice={float(d['mice']):.3f} hmin={float(d['hmin']):.3e} "
              f"sst_min={float(d['sst_min']):+.2f}  [{time.time()-ts:.1f}s]", flush=True)
        if bad:
            ok = False
            print(f"\nFAIL at step {i+1}: not finite/bounded", flush=True)
            break
    print(f"\nALL_ON_SMOKE_{'OK' if ok else 'FAIL'} (zstar+TKE+mEVP; JAX-first, no fidelity oracle)",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
