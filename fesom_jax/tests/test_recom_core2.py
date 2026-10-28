"""REcoM plan Task 15: a CORE2 smoke test of full REcoM through the production driver.

``scripts/run_from_config.py configs/core2_recom.yaml`` on one CPU device (``--partition serial``) for a few steps
from the v3 1990 physics restart, so the BGC starts from the 1-degree IC set
(``recom.ic_set: core2``) and the Albani dust. Checks: the run finishes, the restart carries the BGC fields FESOM2
restarts and not the ones it rebuilds per run segment, every wet BGC value and Benthos are finite, and the global
means of DIC, Alk, DIN, DSi and O2 are in the IC set's ranges (lane I's CORE2 smoke, lesson 2026-10-25).

One device, not ``dist_4``: with 4 in-process CPU devices the CORE2 restart write aborts in XLA:CPU's all_gather
rendezvous (``id < num_threads``, the bug of ``docs/PARALLELISM.md``, seen before only at >= 32 devices), with or
without REcoM (job 28425055); pi on ``dist_4`` runs. The sharded REcoM path is tested on pi (``test_recom_step.py``)
and on the GPUs.

SKIPS when the CORE2 mesh, the v3 restart or the colleague's REcoM inputs are absent. Heavy (CORE2 DP on CPU, several
minutes of compile): run it on a compute node.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MESH = ROOT / "data" / "mesh_core2"
DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
RESTART = Path("/work/ab0995/a270088/port_jax/runs/core2_hindcast_v3/restart_archive/fesom.1990.001.00000")
INPUTS = Path("/work/ab0246/a270092/input/fesom2/recom")
NSTEPS = 3

pytestmark = pytest.mark.skipif(
    not (MESH.exists() and RESTART.exists()
         and (INPUTS / "CORE2" / "DustClimMonthlyAlbani.nc").exists()),
    reason="CORE2 mesh / v3 1990 restart / REcoM CORE2 inputs missing")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("core2_recom")
    s = (ROOT / "configs" / "core2_recom.yaml").read_text()
    for key, val in (("mesh", str(MESH)), ("output_dir", str(tmp)), ("duration", "null")):
        s = re.sub(rf"^{key}:.*$", f"{key}: {val}", s, flags=re.M)
    cfg = tmp / "core2_recom.yaml"
    cfg.write_text(s)
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT))
    env.pop("XLA_FLAGS", None)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_from_config.py"), str(cfg), "--dist-dir",
                        str(DIST), "--partition", "serial", "--restart-in", str(RESTART),
                        "--restart-out", str(tmp / "restart"), "--steps", str(NSTEPS)],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"exit {r.returncode}\n" + r.stdout[-4000:] + r.stderr[-4000:]
    return tmp / "restart"


def test_core2_recom_smoke(run):
    import zarr
    from fesom_jax.recom.tracers import SLOT
    from fesom_jax.zarr_output import reconstruct_global

    keys = set(zarr.open_group(str(run), mode="r").array_keys())
    assert {"bgc", "bgc_old", "bgc_benthos", "bgc_alk_surf"} <= keys
    assert not keys & {"bgc_carb", "bgc_mstep", "bgc_diag", "bgc_budget"}
    X = reconstruct_global(run, "bgc")                               # (nod2D, nl-1, 31)
    nlev = np.load(MESH / "nlevels_nod2D.npy").astype(int)
    wet = np.arange(X.shape[1])[None, :] < (nlev - 1)[:, None]
    assert np.isfinite(X[wet]).all()
    ben = reconstruct_global(run, "bgc_benthos")
    assert np.isfinite(ben).all() and (ben >= 0).all()
    means = {name: float(X[..., SLOT[tid]][wet].mean())
             for name, tid in (("DIN", 1001), ("DIC", 1002), ("Alk", 1003), ("DSi", 1018), ("O2", 1022))}
    print("CORE2_RECOM_MEANS", {k: round(v, 3) for k, v in means.items()})
    # lane I's CORE2 IC ranges (lesson 2026-10-25): DIN 0.008-53, DIC 1083-2468, Alk 1103-2715, DSi 0.34-245,
    # O2 1.8-461; a few steps keep the volume means well inside them
    assert 10 < means["DIN"] < 50 and 2000 < means["DIC"] < 2400 and 2200 < means["Alk"] < 2500
    assert 20 < means["DSi"] < 200 and 50 < means["O2"] < 350
