"""Lane W of the spectral port (docs/plans/completed/20261028-recom-spectral-port.md): the spectral fork sharded equals dense.

Two full steps of the production spectral configuration (``configs/albedo/pi_recom_spectral.yaml``: 37 tracers, the
13-band light from the OASIM tables, GM/Redi, TKE, mEVP) on FESOM2's pi ``dist_4`` (4 host devices, the forced
sharded runner) against the dense steps, on every owned node. The partition is the packaged pi mesh's (the fork
bundle has the same nodes, elements and edges in the same order; only three polar elements' levels differ, which a
partition does not encode). A copy of ``test_recom_step.py::test_sharded_equals_dense`` for the fork, from the
grown state of the 30-day spectral run (:func:`test_recom_step_fork.warm_start`): detritus everywhere, so the
ballasted sinking is exercised.

Finding (2026-10-29): the first run differed in the sinking tracers only (DetC 2.6e-2, DetSi 2.3e-2 relative after
two steps, the physics at 1e-14): the fork's whole-array ballast reset (``any(scaling_density <= tiny)``,
:func:`fesom_jax.recom.sinking.ballast`) was a per-device ``any``, Fortran's per-rank domain, so the 4-device run
reproduced a 4-rank Fortran and the dense run a 1-rank one. The reset now reduces over the whole mesh through the
step's global sum (the stated deviation: fesom-jax reproduces the 1-rank twins and is device-count independent).
Every tracer is judged over all its cells; the per-tracer table prints the magnitude for the record.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fesom_jax import partit, shard_mesh, ssh, surface_forcing  # noqa: E402
from fesom_jax import integrate_sharded as ish  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax.tests import test_recom_step_fork as TF  # noqa: E402

PI_DIST = Path(__file__).resolve().parents[1] / "data" / "mesh_pi"
NDEV = len(jax.devices())
RUN = "pi_1d_spectral_jaxphys"


@pytest.mark.skipif(not TF.HAVE or RUN not in TF.RUNS or NDEV < 4 or not (PI_DIST / "dist_4").is_dir()
                    or not TF.RUN_CFG.is_file(),
                    reason="needs the spectral twin's dumps, the production spectral config, 4 devices and pi dist_4")
def test_spectral_sharded_equals_dense():
    fx = TF._setup(RUN, TF.RUN_CFG)
    mesh, lay, dt = fx["mesh"], fx["lay"], fx["cfg"].dt
    st0, cf, day = TF.warm_start(fx)
    print(f"SHARDED-fork start: day {day} of 1958 ({'restart ' + TF.RESTART if day > 1 else 'twin step 0'})")
    n = 2
    dates = surface_forcing.dates_for_steps(1958, dt, n, start_day=day)
    st = st0
    zero = jnp.zeros((mesh.elem2D, 2))
    for i in range(n):
        st = stepmod.step_jit(st, mesh, fx["op"], zero, dt=dt, is_first_step=(day == 1 and i == 0),
                              step_forcing=cf.step_forcing(*dates[i]), forcing_static=fx["fs"], **fx["phys"])
    dense = jax.device_get(st)
    part = partit.read_partition(PI_DIST, 4)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(jax.device_get(st0), part)
    sop = ssh.partition_ssh_operator(fx["op"], part)
    stack = cf.stack(dates, xp=np)
    seq_p = shard_mesh.partition_step_forcing(stack, part)
    fs_p = shard_mesh.partition_forcing_static(jax.tree.map(lambda a: None if a is None else np.asarray(a), fx["fs"]),
                                               part)
    stress_p = np.zeros((4, sm.Lmax["elem"], 2))
    from fesom_jax import ice_evp           # the GLOBAL mEVP coast mask, partitioned in, as run_from_config does
    bn_p = shard_mesh._shard_along_axis(np.asarray(ice_evp.boundary_node_mask(mesh)), part.myList_nod2D,
                                        sm.Lmax["nod"], 0, False)
    out = jax.device_get(ish.run_steps_sharded_forced(sm, state_p, sop, stress_p, seq_p, fs_p, n, dt=dt, npes=4,
                                                      boundary_node_p=bn_p, bootstrap_ab2=(day == 1), **fx["phys"]))
    rel = lambda a, b: float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-300)   # noqa: E731
    dX, dA = np.zeros(lay.ntr), np.zeros(lay.ntr)                         # relative / absolute
    mx = np.max(np.abs(np.asarray(dense.bgc)), axis=(0, 1))
    dT = dS = dB = 0.0
    for p in range(4):
        g = np.asarray(part.myList_nod2D[p][: part.myDim_nod2D[p]])
        m = len(g)
        for s in range(lay.ntr):
            a, b = np.asarray(out.bgc)[p, :m, :, s], np.asarray(dense.bgc)[g, :, s]
            dX[s] = max(dX[s], rel(a, b))
            dA[s] = max(dA[s], float(np.max(np.abs(a - b))))
        dT = max(dT, rel(np.asarray(out.T)[p, :m], np.asarray(dense.T)[g]))
        dS = max(dS, rel(np.asarray(out.S)[p, :m], np.asarray(dense.S)[g]))
        dB = max(dB, rel(np.asarray(out.bgc_benthos)[p, :m], np.asarray(dense.bgc_benthos)[g]))
    for s in np.argsort(-dX):
        print(f"SHARDED-fork {lay.ids[s]} {lay.names[s]:>10}: rel {dX[s]:.2e} (abs {dA[s]:.2e}, max|dense| {mx[s]:.2e})")
    w = int(np.argmax(dX))
    print(f"SHARDED-fork vs dense after {n} steps: BGC worst {lay.names[w]} rel {dX[w]:.2e} "
          f"(median {np.median(dX):.2e}), Benthos {dB:.2e}; physics T {dT:.2e}, S {dS:.2e}")
    floor = max(dT, dS)
    assert dX.max() <= max(TF.G4B_FACTOR * floor, 1e-12) and dB <= max(TF.G4B_FACTOR * floor, 1e-12)
