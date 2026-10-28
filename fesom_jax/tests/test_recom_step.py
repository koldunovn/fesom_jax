"""REcoM plan Task 12 (lane W): full REcoM in the model step.

* G4b -- one full JAX step on pi from the physics-matched twin's step-0 state (``recom_ref/pi_30d_jan_jaxphys``,
  :mod:`fesom_jax.recom.reference`) against the twin's step 1, for the 31 BGC tracers and ``Benthos``, judged
  against the physics floor of the same pair and step (T and S, which reach the BGC tracers through the transport).
* the closed JAX budget over a few steps (``RecomConfig.budget``): every tracer's inventory change = surface BC +
  stage 1 + REcoM's tracer-loop terms; the element budgets with the benthos pools.
* the BGC-every-k mode for full REcoM (k = 2): runs, keeps the budget closed.
* sharded (FESOM2's pi ``dist_4``, 4 host devices) against dense for REcoM steps.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fesom_jax import partit, shard_mesh, ssh, surface_forcing  # noqa: E402
from fesom_jax import integrate_sharded as ish  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax.io_dump import load_recom_dump  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.recom import reference as R  # noqa: E402
from fesom_jax.recom import tracers as T  # noqa: E402
from fesom_jax.recom import wiring  # noqa: E402
from fesom_jax.run_config import load_yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "configs" / "pi_recom_bgc.yaml"
DUMPS = R.TWIN / "dumps"
HAVE_TWIN = (DUMPS / "step_000001").is_dir()
need_twin = pytest.mark.skipif(not HAVE_TWIN, reason=f"REcoM twin dumps absent ({DUMPS})")
PI_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/pi")
NDEV = len(jax.devices())
# G4b: a BGC tracer's step-1 error, relative to its max, may be at most G4B_FACTOR times the larger relative error of
# T and S after the same step (the physics floor of the pair; with FESOM2 #944/#1116 in, T <= 2e-6 degC). The BGC
# tracers see the physics difference through the transport and, in stage 1, through a_ice (2.7e-7 after the ice
# step); the factor allows for tracers with sharper vertical gradients than T and S.
G4B_FACTOR = 100.0


@lru_cache(maxsize=1)
def _setup(budget: bool = False, bgc_every: int = 1):
    cfg = load_yaml(CFG)
    cfg = dataclasses.replace(cfg, recom=cfg.recom._replace(budget=budget, bgc_every=bgc_every))
    phys = cfg.physics_kwargs()
    mesh = load_mesh()
    d1 = load_recom_dump(DUMPS, 1, names=R.dump_names())[0]
    st = R.twin_step0_state(mesh, d1, phys["ale_cfg"], cfg.recom)
    if bgc_every > 1:
        from fesom_jax.recom.transport import init_every_k
        st = init_every_k(st, bgc_every)
    cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=np.asarray(st.T[:, 0]), dt=cfg.dt,
                                               mslp=True, bgc_dust_path=cfg.bgc_dust_path(), **cfg.forcing_paths())
    fs = wiring.with_recom_static(cf.static, mesh)
    op = ssh.build_ssh_operator(mesh, dt=cfg.dt)
    return dict(cfg=cfg, phys=phys, mesh=mesh, d1=d1, st=st, cf=cf, fs=fs, op=op)


def _run(fx, n):
    mesh = fx["mesh"]
    dates = surface_forcing.dates_for_steps(1958, fx["cfg"].dt, n)
    st = jax.tree.map(jnp.asarray, fx["st"])
    zero = jnp.zeros((mesh.elem2D, 2))
    states = [st]
    for i in range(n):
        sf = fx["cf"].step_forcing(*dates[i])
        st = stepmod.step_jit(st, mesh, fx["op"], zero, dt=fx["cfg"].dt, is_first_step=(i == 0), step_forcing=sf,
                              forcing_static=fx["fs"], **fx["phys"])
        states.append(st)
    return states


def _rel(a, b, mask):
    a, b = np.asarray(a)[mask], np.asarray(b)[mask]
    den = float(np.max(np.abs(b)))
    return (float(np.max(np.abs(a - b))) / den if den > 0 else float(np.max(np.abs(a)))), den


def g4b_errors():
    """``{name: (rel. error, max |fortran|)}`` after one JAX step from the twin's step 0: the 31 tracers, Benthos, T,
    S and stage 1's surface diagnostics."""
    fx = _setup()
    mesh = fx["mesh"]
    nl1 = mesh.nl - 1
    st1 = jax.device_get(_run(fx, 1)[1])
    f = fx["d1"]
    nlev = np.asarray(mesh.nlevels_nod2D) - 1
    wet = np.arange(nl1)[None, :] < nlev[:, None]
    allnod = np.ones(mesh.nod2D, bool)
    e = {"T": _rel(st1.T[:, :nl1], f["step_end.tr1"], wet), "S": _rel(st1.S[:, :nl1], f["step_end.tr2"], wet)}
    for s, tid in enumerate(T.BGC_IDS):
        e[f"{tid}_{T.NAMES[s]}"] = _rel(st1.bgc[:, :nl1, s], f[f"step_end.tr{tid}"], wet)
    for i in range(4):
        e[f"Benthos{i + 1}"] = _rel(st1.bgc_benthos[:, i], f["step_end.Benthos"][i], allnod)
    dg = dict(zip(wiring.DIAG2D, np.moveaxis(np.asarray(st1.bgc_diag), -1, 0)))
    e["co2flux"] = _rel(dg["co2flux"], f["recom_out.GloCO2flux"], allnod)
    e["pco2surf"] = _rel(dg["pco2surf"], f["recom_out.GloPCO2surf"], allnod)
    e["ph"] = _rel(dg["ph"], f["recom_out.GloHplus"], allnod)
    e["npp"] = _rel(dg["npp"], f["recom_out.NPPn"] + f["recom_out.NPPd"], allnod)
    return e


@need_twin
def test_g4b_one_step_vs_twin():
    e = g4b_errors()
    floor = max(e["T"][0], e["S"][0])
    for k, v in sorted(e.items(), key=lambda kv: -kv[1][0]):
        print(f"G4B {k:>16}: rel {v[0]:.2e} (max |F| {v[1]:.3e})")
    print(f"G4B physics floor (max rel of T, S) {floor:.2e}")
    bad = {k: v for k, v in e.items() if k not in ("T", "S") and v[0] > max(G4B_FACTOR * floor, 1e-12)}
    assert not bad, (floor, bad)


@need_twin
@pytest.mark.parametrize("bgc_every", [1, 2])
def test_budget_closed(bgc_every):
    """Over 4 steps every tracer's inventory change equals the accumulated surface + stage 1 + tracer-loop content
    (the transport conserves sum X hnode areasvol), and N, Si and C (water column + benthos pools) change by the clip
    alone (no N, Si or C enters through the surface except the air-sea CO2 flux)."""
    fx = _setup(budget=True, bgc_every=bgc_every)
    mesh = fx["mesh"]
    n = 4
    states = _run(fx, n)
    s0, sN = jax.device_get(states[0]), jax.device_get(states[-1])
    if bgc_every > 1:      # the BGC content of a held interval refers to the interval-start thickness
        assert int(np.asarray(sN.bgc_steps)[0]) == 0
    I0 = wiring.inventory(s0.bgc, s0.hnode, mesh)
    IN = wiring.inventory(sN.bgc, sN.hnode, mesh)
    b = np.asarray(sN.bgc_budget).sum(axis=0)                     # (4 terms, ntr)
    pred = b[0] + b[1] + b[2]
    # DICremin starts at zero (and is reset at the surface every step): scale by the larger inventory or the terms
    scale = np.maximum.reduce([np.abs(I0), np.abs(IN), np.abs(b[:3]).sum(axis=0), np.full_like(I0, 1e-300)])
    rel = np.abs(IN - I0 - pred) / scale
    worst = int(np.argmax(rel))
    print(f"BUDGET k={bgc_every}: worst tracer {T.BGC_IDS[worst]} {T.NAMES[worst]} |dI - terms|/I = {rel[worst]:.2e}; "
          f"median {np.median(rel):.2e}")
    assert np.all(rel <= 1e-12), dict(zip(T.NAMES, rel))
    # element budgets: water column + benthos pools [mmol], changed by the surface BC and the clip only
    area = np.asarray(mesh.area)[:, 0]
    ben = lambda s, i: float(np.sum(np.asarray(s.bgc_benthos)[:, i] * area))     # noqa: E731
    groups = {"N": ([1001, 1004, 1007, 1009, 1011, 1013, 1023, 1025, 1029], (0,)),
              "Si": ([1016, 1017, 1018, 1027], (2,)),
              "C": ([1002, 1005, 1008, 1010, 1012, 1014, 1020, 1021, 1024, 1026, 1028, 1030], (1, 3))}
    for el, (ids, pools) in groups.items():
        sl = [T.SLOT[i] for i in ids]
        d = float((IN - I0)[sl].sum()) + sum(ben(sN, q) - ben(s0, q) for q in pools)
        src = float((b[0] + b[3])[sl].sum())
        tot = float(I0[sl].sum())
        print(f"BUDGET k={bgc_every} {el}: d(water + benthos) {d:.6e}, surface + clip {src:.6e}, inventory {tot:.6e}, "
              f"rel residual {abs(d - src) / tot:.2e}")
        assert abs(d - src) <= 1e-12 * tot, el


@pytest.mark.skipif(not HAVE_TWIN or NDEV < 4 or not (PI_DIST / "dist_4").is_dir(),
                    reason="needs the twin dumps, 4 devices and pi dist_4")
def test_sharded_equals_dense():
    """Two full REcoM steps on FESOM2's pi dist_4 (4 host devices, the forced sharded runner) against the dense
    steps, on every owned node: the BGC tracers within the physics' own sharded-vs-dense difference."""
    fx = _setup()
    mesh = fx["mesh"]
    n = 2
    dense = jax.device_get(_run(fx, n)[-1])
    part = partit.read_partition(PI_DIST, 4)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(fx["st"], part)
    sop = ssh.partition_ssh_operator(fx["op"], part)
    stack = fx["cf"].stack(surface_forcing.dates_for_steps(1958, fx["cfg"].dt, n), xp=np)
    seq_p = shard_mesh.partition_step_forcing(stack, part)
    fs_p = shard_mesh.partition_forcing_static(jax.tree.map(lambda a: None if a is None else np.asarray(a), fx["fs"]),
                                               part)
    stress_p = np.zeros((4, sm.Lmax["elem"], 2))
    # the GLOBAL mEVP coast mask, partitioned in, as run_from_config does (the sharded fallback would build it from
    # each device's local mesh and treat the partition cuts as coasts)
    from fesom_jax import ice_evp
    bn_p = shard_mesh._shard_along_axis(np.asarray(ice_evp.boundary_node_mask(mesh)), part.myList_nod2D,
                                        sm.Lmax["nod"], 0, False)
    out = jax.device_get(ish.run_steps_sharded_forced(sm, state_p, sop, stress_p, seq_p, fs_p, n, dt=fx["cfg"].dt,
                                                      npes=4, boundary_node_p=bn_p, **fx["phys"]))
    rel = lambda a, b: float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-300)   # noqa: E731
    dX = np.zeros(T.NTR)
    dT = dS = dB = 0.0
    for p in range(4):
        g = np.asarray(part.myList_nod2D[p][: part.myDim_nod2D[p]])
        m = len(g)
        for s in range(T.NTR):
            dX[s] = max(dX[s], rel(np.asarray(out.bgc)[p, :m, :, s], np.asarray(dense.bgc)[g, :, s]))
        dT = max(dT, rel(np.asarray(out.T)[p, :m], np.asarray(dense.T)[g]))
        dS = max(dS, rel(np.asarray(out.S)[p, :m], np.asarray(dense.S)[g]))
        dB = max(dB, rel(np.asarray(out.bgc_benthos)[p, :m], np.asarray(dense.bgc_benthos)[g]))
    w = int(np.argmax(dX))
    print(f"SHARDED vs dense after {n} steps: BGC worst {T.NAMES[w]} rel {dX[w]:.2e} (median {np.median(dX):.2e}), "
          f"Benthos {dB:.2e}; physics T {dT:.2e}, S {dS:.2e}")
    floor = max(dT, dS)
    assert dX.max() <= max(G4B_FACTOR * floor, 1e-12) and dB <= max(G4B_FACTOR * floor, 1e-12)


def test_recom_off_static_has_no_leaves():
    """The REcoM constants are optional ForcingStatic fields: off, they add no leaf (the DP lock's program)."""
    from fesom_jax.surface_forcing import ForcingStatic
    fs = ForcingStatic(runoff_node=np.zeros(3), areasvol_surf=np.ones(3), ocean_area=np.asarray(3.0),
                       open_water=np.ones(3, bool), a_ice=np.zeros(3))
    assert len(jax.tree.leaves(fs)) == 5
    from fesom_jax import partit as _pt
    ser = _pt.synth_serial(3, 1, 3)
    fp = shard_mesh.partition_forcing_static(fs, ser)
    assert fp.bgc_seafloor is None and len(jax.tree.leaves(fp)) == 5
