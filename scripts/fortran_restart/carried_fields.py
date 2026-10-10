"""Which State fields does one fesom-jax step READ (carried state), and which does it only write?

A Fortran FESOM2 restart holds a subset of the JAX State. Only the fields a step reads before it
overwrites them need a value from the restart (or a derived one); the rest the step recomputes. This
script measures that on CORE2 (configs/core2_full.yaml physics, the 2013-01-01 State): each field in
turn is perturbed (x -> x + 1 + 0.1 |x|, every lane), one step is taken, and the field counts as read
when any output leaf differs from the unperturbed step. A masked lane (x * 0) does not count.

  python scripts/fortran_restart/carried_fields.py --out <json>     (CPU node; CORE2 is not for login)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", default="2013-01-01")
    args = ap.parse_args(argv)
    import core2_adj as C  # noqa: E402  (import guard: fesom_jax from this checkout)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from fesom_jax import step as stepmod

    t0 = time.time()
    model = C.build_model(start=args.date, n_steps=1, chunk_steps=1)
    mesh = model.mesh
    x0 = jax.tree.map(lambda a: a[0], model.stack_forcing([model.date(model.start, 0)]))
    s0, u0, phys = model.s0, model.params0, model.phys

    def one(s):
        return stepmod.step(s, mesh, model.op, None, model.to_params(u0), dt=model.dt,
                            is_first_step=False, step_forcing=x0,
                            forcing_static=model.forcing.static, adjoint_cfg=model.adjoint_cfg,
                            **phys)
    f = jax.jit(one)
    base = jax.device_get(f(s0))
    names = [fl.name for fl in dataclasses.fields(s0)]
    res = {}
    for name in names:
        x = getattr(s0, name)
        s = dataclasses.replace(s0, **{name: x + 1.0 + 0.1 * jnp.abs(x)})
        out = jax.device_get(f(s))
        diffs = {k: float(np.max(np.abs(np.asarray(getattr(out, k)) - np.asarray(getattr(base, k)))))
                 for k in names}
        changed = sorted(k for k, d in diffs.items() if d > 0.0)
        res[name] = {"read": bool(changed), "outputs_changed": changed}
        print(f"[{name}] read={bool(changed)} changes {len(changed)} outputs", flush=True)
    doc = {"date": args.date, "config": str(C.CONFIG), "fields": res,
           "read": [k for k in names if res[k]["read"]],
           "write_only": [k for k in names if not res[k]["read"]], "wall_s": time.time() - t0}
    Path(args.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({"read": doc["read"], "write_only": doc["write_only"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
