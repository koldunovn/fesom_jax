"""Single-precision sub-plan, Task 2: the DP lock.

Lowers (does not compile or run) the double-precision programs the SP work must leave untouched and hashes their
StableHLO text with source locations removed:

    fwd_core2_full   the production forced chunk of ``scripts/run_from_config.py`` (``configs/core2_full_bench.yaml``,
                     every speed option off), dist_4 on 4 fake CPU devices
    fwd_core2_gpu    the same with ``configs/core2_gpu.yaml`` (Chebyshev CG, on-device forcing, gathered SSH, compact
                     halo, column unroll, mEVP wide halo)
    adj_core2        the reverse program of ``scripts/adjoint/core2_adj.py`` (``core2_full.yaml``,
                     ``AdjointConfig.long_window()``, ice ``rheology``, remat on): grad of a scalar of one 2-step chunk
                     with respect to the default controls, from the cold start of 1958-01-01

Each entry records the hash, the number of f32/f64 tensor types in the text, and its length. Same code ⇒ the same
lowered program ⇒ the same compiled executable on every backend.

    python scripts/perf/dp_hlo_hash.py [--write fesom_jax/tests/data/dp_hlo_hashes.json] [--only NAME ...]

Must run on a compute node (CORE2 mesh; a few minutes on CPU). Sets 4 fake CPU devices itself.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

# DP_HASH_ROOT: hash another checkout (e.g. the merge base) with this script
# One host thread for numpy/BLAS: host-built constants of the programs (e.g. the Chebyshev bounds from a power
# iteration, ssh.enable_cheb_precond) depend on the BLAS summation order, i.e. on the thread count, at the last ulp.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
ROOT = Path(os.environ.get("DP_HASH_ROOT") or Path(__file__).resolve().parents[2])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
_DEV = "--xla_force_host_platform_device_count=4"
if _DEV not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _DEV).strip()

PROGRAMS = ("fwd_core2_full", "fwd_core2_gpu", "adj_core2")
DIST_DIR = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2"     # the dist_4 partition production uses


class _Captured(Exception):
    pass


def _digest(text: str) -> dict:
    clean = re.sub(r"\s*loc\([^()]*(\([^()]*\)[^()]*)*\)", "", text)       # inline locations
    clean = "\n".join(ln for ln in clean.splitlines() if not ln.startswith("#loc"))
    return {"sha256": hashlib.sha256(clean.encode()).hexdigest(),
            "f32_types": len(re.findall(r"xf32>|tensor<f32>", clean)),
            "f64_types": len(re.findall(r"xf64>|tensor<f64>", clean)),
            "chars": len(clean)}


def _rfc_module():
    spec = importlib.util.spec_from_file_location("run_from_config_cli", ROOT / "scripts" / "run_from_config.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def lower_forward(config: str, extra=()) -> str:
    """The lowered StableHLO of the first forced chunk run_from_config would run (2 steps)."""
    import fesom_jax.run as R
    orig = R.run_steps_sharded_forced
    got = {}

    def capture(*a, **kw):
        kw = dict(kw, return_executable=True)
        jfn, args, _npes = orig(*a, **kw)
        got["text"] = jfn.lower(*args).as_text()
        raise _Captured()

    mod = _rfc_module()
    R.run_steps_sharded_forced = capture
    argv = sys.argv
    sys.argv = ["run_from_config.py", str(ROOT / config), "--dist-dir", DIST_DIR, "--steps", "2",
                "--chunk-steps", "2", *extra]
    try:
        mod.main()
    except _Captured:
        pass
    finally:
        sys.argv = argv
        R.run_steps_sharded_forced = orig
    return got["text"]


def lower_adjoint() -> str:
    import numpy as np
    import jax
    import jax.numpy as jnp
    spec = importlib.util.spec_from_file_location("core2_adj", ROOT / "scripts" / "adjoint" / "core2_adj.py")
    A = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = A
    spec.loader.exec_module(A)
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import cold_start_state
    cfg = A.load_yaml(A.CONFIG)
    mesh = load_mesh(ROOT / cfg.mesh)
    s0 = cold_start_state(mesh, None, xp=np)
    model = A.build_model(start="1958-01-01", n_steps=2, chunk_steps=2, state=s0,
                          adjoint_cfg=A.AdjointConfig.long_window(), ice_mode="rheology")
    carry = model.init_carry()
    xs = model.xs_fn()(0)

    def loss(u):
        out = model.chunk_fn(carry, xs, u)
        st = model.state_of(out[0] if isinstance(out, tuple) else out)
        return jnp.sum(st.T[:, 0])

    return jax.jit(jax.grad(loss)).lower(model.params0).as_text()


def compute(only=None) -> dict:
    """Each program is lowered in its OWN process: the forced runner's executable cache is keyed by ``id()`` of
    per-run objects, and a second run in the same process can reuse a freed object's id and so be handed the first
    run's jitted function (seen 2026-10-21: the core2_gpu hash changed only when lowered after core2_full)."""
    names = [n for n in PROGRAMS if not only or n in only]
    if len(names) > 1:
        import subprocess
        import tempfile
        out = {}
        for name in names:
            with tempfile.TemporaryDirectory() as td:
                js = Path(td) / "one.json"
                r = subprocess.run([sys.executable, __file__, "--only", name, "--write", str(js)],
                                   env=dict(os.environ), capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"{name}: {r.stdout[-2000:]}{r.stderr[-2000:]}")
                out[name] = json.loads(js.read_text())["programs"][name]
            print(f"[dp_hlo_hash] {name}: {out[name]}", flush=True)
        return out
    out = {}
    for name in names:
        if name == "fwd_core2_full":
            text = lower_forward("configs/core2_full_bench.yaml")
        elif name == "fwd_core2_gpu":
            text = lower_forward("configs/core2_gpu.yaml")
        else:
            text = lower_adjoint()
        out[name] = _digest(text)
        if os.environ.get("DP_HASH_DUMP"):          # the cleaned text, to diff two checkouts
            clean = re.sub(r"\s*loc\([^()]*(\([^()]*\)[^()]*)*\)", "", text)
            clean = "\n".join(ln for ln in clean.splitlines() if not ln.startswith("#loc"))
            Path(os.environ["DP_HASH_DUMP"]).mkdir(parents=True, exist_ok=True)
            (Path(os.environ["DP_HASH_DUMP"]) / f"{name}.mlir").write_text(clean)
        print(f"[dp_hlo_hash] {name}: {out[name]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", help="write the hashes to this JSON file")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    res = compute(a.only)
    if a.write:
        import subprocess
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
        import jax
        Path(a.write).write_text(json.dumps({"commit": head, "jax": jax.__version__, "programs": res},
                                            indent=1) + "\n")
        print(f"wrote {a.write}")


if __name__ == "__main__":
    main()
