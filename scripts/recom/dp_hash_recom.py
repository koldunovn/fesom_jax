"""Lane SP (plan docs/plans/20261026-recom-lane-sp.md): the DP lock for REcoM's programs.

``fesom_jax/tests/test_dp_lock.py`` locks the physics-only DP programs; this lowers the DOUBLE-precision pi step with
full REcoM (the twin's step 0, ``configs/pi_recom_bgc.yaml``; first and later step, with and without the budget sums)
and with 4 passive tracers, strips the source locations and prints one sha256 per program as JSON. Run it with the
base commit's code and with the branch's code (``PYTHONPATH``) and compare: equal hashes = the DP programs did not
change. Lowering only (no compile, no run); needs the twin dumps.

    PYTHONPATH=<tree> python scripts/recom/dp_hash_recom.py [--out f.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _digest(text: str) -> str:
    clean = re.sub(r"\s*loc\([^()]*(\([^()]*\)[^()]*)*\)", "", text)
    clean = "\n".join(ln for ln in clean.splitlines() if not ln.startswith("#loc"))
    return hashlib.sha256(clean.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from fesom_jax import step as stepmod
    from fesom_jax import surface_forcing
    from fesom_jax.tests import test_recom_step as TS
    out = {}
    for budget in (False, True):
        fx = TS._setup(budget=budget)
        mesh, cfg = fx["mesh"], fx["cfg"]
        st = jax.tree.map(jnp.asarray, fx["st"])
        sf = fx["cf"].step_forcing(*surface_forcing.dates_for_steps(1958, cfg.dt, 1)[0])
        zero = jnp.zeros((mesh.elem2D, 2))
        for first in (True, False):
            txt = stepmod.step_jit.lower(st, mesh, fx["op"], zero, dt=cfg.dt, is_first_step=first, step_forcing=sf,
                                         forcing_static=fx["fs"], **fx["phys"]).as_text()
            out[f"full_budget{int(budget)}_first{int(first)}"] = _digest(txt)
    # the passive mode (4 copies of S)
    from fesom_jax.recom.config import RecomConfig
    from fesom_jax.recom.transport import init_passive
    fx = TS._setup()
    mesh, cfg = fx["mesh"], fx["cfg"]
    phys = dict(fx["phys"], recom_cfg=RecomConfig(passive_tracers=4))
    st = jax.tree.map(jnp.asarray, init_passive(dataclasses.replace(
        fx["st"], **{f: None for f in ("bgc_benthos", "bgc_alk_surf", "bgc_carb", "bgc_mstep", "bgc_diag",
                                       "bgc_budget")}), 4))
    sf = fx["cf"].step_forcing(*surface_forcing.dates_for_steps(1958, cfg.dt, 1)[0])
    txt = stepmod.step_jit.lower(st, mesh, fx["op"], jnp.zeros((mesh.elem2D, 2)), dt=cfg.dt, is_first_step=False,
                                 step_forcing=sf, forcing_static=fx["fs"], **phys).as_text()
    out["passive4"] = _digest(txt)
    s = json.dumps(out, indent=1, sort_keys=True)
    print(s)
    if a.out:
        open(a.out, "w").write(s)


if __name__ == "__main__":
    main()
