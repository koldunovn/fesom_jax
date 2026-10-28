"""Lane SP (plan ``docs/plans/20261026-recom-lane-sp.md``): single precision with full REcoM.

An opt-in deviation: FESOM2 refuses ``USE_SINGLE_PRECISION`` with ``RECOM_COUPLED``. The BGC tracers get FESOM2
#1207's tracer scheme of T and S (WP values, float64 ``del_ttf``, the carried reconstruction, the carried implicit
diffusion; ``State.bgc_carry``), REcoM's own code runs in float64 (REcoM's ``WP = 8``: widened inputs, explicit constants), and every REcoM
increment enters the one carried rounding as float64 content (S1 design, deviations R0-R6).

* (b) one SP step from the physics-matched twin's step 0 against the DP step: every BGC tracer's ``|SP - DP|`` relative
  to its max, next to T's and S's of the same pair (the float32 floor of the physics); the State dtypes; no implicit
  float32/float64 mixing anywhere in the traced step (REcoM's island is float64 throughout, the transport WP/WP_full);
* (c) the SP budget over 4 and 30 steps, the carry counted as content, and the negative control (the carry zeroed
  before every step);
* (d) pi ``dist_4`` sharded SP against dense SP;
* (e) SP restarts through ``run_from_config``: the BGC carry on disk (float64), back to WP, a DP restart starts an SP
  run, an SP restart starts a DP run;
* (f) the refusals and the cast policy.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from functools import lru_cache

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fesom_jax import partit, shard_mesh, ssh, surface_forcing  # noqa: E402
from fesom_jax import integrate_sharded as ish  # noqa: E402
from fesom_jax import precision as PR  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.params import Params  # noqa: E402
from fesom_jax.recom import tracers as T  # noqa: E402
from fesom_jax.recom import wiring  # noqa: E402
from fesom_jax.recom.config import RecomConfig  # noqa: E402
from fesom_jax.state import State  # noqa: E402
from fesom_jax.tests import test_recom_restart as TR  # noqa: E402
from fesom_jax.tests import test_recom_step as TS  # noqa: E402

need_twin = TS.need_twin
# (b), (d): a BGC tracer's SP - DP (or sharded - dense) relative error may be at most this factor times the larger of
# T's and S's of the same pair (the G4b rule of test_recom_step): the BGC tracers see the physics' float32
# differences through the transport and through stage 1's inputs, and carry their own float32 rounding
FLOOR_FACTOR = 100.0


@lru_cache(maxsize=2)
def _sp_setup(budget: bool = False):
    """The SP inputs of the twin's pi case (TS._setup's DP fixture cast as run_from_config casts a single-precision
    run): the WP mesh, ForcingStatic (with REcoM's node constants), SSH operator (preconditioner variant 1, #995),
    Params, forcing in point-slope form (#940) and the State with its carries."""
    fx = TS._setup(budget=budget)
    cfg, mesh = fx["cfg"], fx["mesh"]
    cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=np.asarray(fx["st"].T[:, 0]), dt=cfg.dt, mslp=True,
                                               bgc_dust_path=cfg.bgc_dust_path(), **cfg.forcing_paths())
    cf.point_slope = True
    fs = wiring.with_recom_static(cf.static, mesh)
    return dict(fx, cf_sp=cf, m32=PR.cast_tree(mesh), fs32=PR.cast_tree(fs),
                op32=PR.cast_tree(ssh.build_ssh_operator(mesh, dt=cfg.dt, precond_variant=1)),
                p32=PR.cast_tree(Params.defaults()), st32=PR.cast_state(fx["st"]))


def _sp_forcing(sx, n):
    dates = surface_forcing.dates_for_steps(1958, sx["cfg"].dt, n)
    return [PR.cast_tree(sx["cf_sp"].step_forcing(*d, xp=np)) for d in dates]


def _sp_step_fn(sx):
    def one(st, sf, *, first):
        return stepmod.step(st, sx["m32"], sx["op32"], None, sx["p32"], dt=sx["cfg"].dt, is_first_step=first,
                            step_forcing=sf, forcing_static=sx["fs32"], **sx["phys"])
    return one


def _sp_run(sx, n, zero_carry=False):
    """``n`` dense SP steps from the twin's step 0 (one jitted step per first/later); ``zero_carry`` drops the BGC
    rounding residual before every step (the negative control)."""
    one = _sp_step_fn(sx)
    jit = {f: PR.jit(functools.partial(one, first=f), "single") for f in (True, False)}
    sfs = _sp_forcing(sx, n)
    st = jax.device_put(sx["st32"])
    states = [st]
    with PR.scope("single"):
        for i in range(n):
            if zero_carry:
                st = dataclasses.replace(st, bgc_carry=jnp.zeros_like(st.bgc_carry))
            st = jit[i == 0](st, sfs[i])
            states.append(st)
    return states


def _wet(mesh):
    nl1 = mesh.nl - 1
    return np.arange(nl1)[None, :] < (np.asarray(mesh.nlevels_nod2D) - 1)[:, None]


def _rel(a, b, mask):
    a, b = np.asarray(a, np.float64)[mask], np.asarray(b, np.float64)[mask]
    den = float(np.max(np.abs(b)))
    return float(np.max(np.abs(a - b))) / den if den > 0 else float(np.max(np.abs(a)))


# ---- (b) one step: SP vs DP ----------------------------------------------------------------------------------------
@need_twin
def test_sp_step_vs_dp_and_dtypes():
    sx = _sp_setup()
    mesh = sx["mesh"]
    nl1 = mesh.nl - 1
    wet = _wet(mesh)
    dp = jax.device_get(TS._run(sx, 1)[1])
    sp = jax.device_get(_sp_run(sx, 1)[1])
    # dtypes per the policy (State WP, hnode WP_full, REcoM's own arrays float64, the carries WP)
    for f in dataclasses.fields(sp):
        a = getattr(sp, f.name)
        if a is None:
            continue
        if f.name == "bgc_mstep":
            assert np.asarray(a).dtype == np.int32
            continue
        want = np.float64 if f.name in PR.STATE_F64 + PR.STATE_F64_RECOM else np.float32
        assert np.asarray(a).dtype == want, f.name
        assert np.all(np.isfinite(np.asarray(a))), f.name
    assert sp.bgc_carry.shape == sp.bgc.shape and float(np.abs(sp.bgc_carry).max()) > 0.0
    e = {"T": _rel(sp.T[:, :nl1], dp.T[:, :nl1], wet), "S": _rel(sp.S[:, :nl1], dp.S[:, :nl1], wet)}
    for s, tid in enumerate(T.BGC_IDS):
        e[f"{tid}_{T.NAMES[s]}"] = _rel(sp.bgc[:, :nl1, s], dp.bgc[:, :nl1, s], wet)
    allnod = np.ones(mesh.nod2D, bool)
    for i in range(4):
        e[f"Benthos{i + 1}"] = _rel(sp.bgc_benthos[:, i], dp.bgc_benthos[:, i], allnod)
    for j, name in enumerate(wiring.DIAG2D):
        e[name] = _rel(sp.bgc_diag[:, j], dp.bgc_diag[:, j], allnod)
    floor = max(e["T"], e["S"])
    for k, v in sorted(e.items(), key=lambda kv: -kv[1]):
        print(f"SP-DP {k:>16}: rel {v:.2e}")
    print(f"SP-DP physics floor (max rel of T, S) {floor:.2e}")
    bad = {k: v for k, v in e.items() if k not in ("T", "S") and v > FLOOR_FACTOR * floor}
    assert not bad, (floor, bad)


# (b2) the BGC scheme alone: stage 3 on the SAME transport fields in SP and in DP. The step-level pair above is
# dominated by the physics' own SP - DP on this pi case (the TKE Kv of the first step from zero TKE differs by ~36 %
# rms between SP and DP, T by 1.2e-3 rms; the float32-rounded initial state alone moves T by 2e-7 rms), so it cannot
# see a BGC scheme error of float32 size. Here the inputs are the G4a dumps (Fortran's transport fields of the upstream
# pi test, w_split dropped in both), rounded to WP where the SP step holds them in WP.
STAGE3_SP_TOL = 2e-5          # max |SP - DP| / max |DP| per tracer over the wet cells (float32 rounding of the inputs)


@pytest.mark.skipif(not (Path("/work/ab0995/a270088/recom_ref") / "pi_30d_jan" / "dumps").is_dir(),
                    reason="REcoM G4a dumps absent")
@pytest.mark.parametrize("step", [1, 224])
def test_sp_stage3_vs_dp_same_transport(step):
    from fesom_jax.adjoint_stab import AdjointConfig
    from fesom_jax.recom import transport
    from fesom_jax.tests import test_recom_stage3 as G
    mesh, X1, X0, kw, _ = G._stage3_inputs("pi_30d_jan", step)
    kw = dict(kw, w_i=None)
    closed = {k: kw.pop(k) for k in ("params", "switches", "dt", "geom_sink")}
    ident = lambda f, kind: f                                         # noqa: E731
    common = dict(exch=ident, ckpt=lambda g: g, adj=AdjointConfig())
    dp = jax.jit(lambda X1, X0, dyn: transport.recom_stage3(mesh, X1, X0, **dyn, **closed, **common))(X1, X0, kw)
    # SP: WP transport and geometry, float64 hnode/hnode_new and the diffusion column's Z3d (WP_full), REcoM's
    # surface fluxes, benthos and sinking geometry float64 (REcoM's WP = 8), stage 1's X1 float64
    from fesom_jax import ale
    wp_keys = ("uv_adv", "w_e_adv", "helem", "zbar3_live", "Z3d_live", "Kv_eff", "gm_diag", "temp", "salt")
    kw_sp = {k: (PR.cast_tree(v) if k in wp_keys else v) for k, v in kw.items()}
    kw_sp["Z3d_new"] = ale.live_geometry(mesh, kw["hnode_new"], keep_f64=True)[1]
    m32 = PR.cast_tree(mesh)
    X0_32 = PR.cast_tree(X0)
    carry0 = jnp.zeros(X1.shape, jnp.float32)
    with PR.scope("single"):
        sp = PR.jit(lambda X1, X0, c, dyn: transport.recom_stage3(m32, X1, X0, carry=c, **dyn, **closed, **common),
                    "single")(X1, X0_32, carry0, kw_sp)
    X_dp, X_sp, carry = np.asarray(dp[0]), np.asarray(sp[0]), np.asarray(sp[-1])
    assert X_sp.dtype == np.float32 and carry.dtype == np.float32 and np.asarray(sp[2]).dtype == np.float64
    nl1 = mesh.nl - 1
    wet = _wet(mesh)
    err = {T.NAMES[s]: _rel(X_sp[:, :nl1, s], X_dp[:, :nl1, s], wet) for s in range(T.NTR)}
    eb = _rel(np.asarray(sp[2]), np.asarray(dp[2]), np.ones(mesh.nod2D, bool))
    w = max(err, key=err.get)
    print(f"SP STAGE3 step {step}: worst {w} {err[w]:.2e}, median {np.median(list(err.values())):.2e}, "
          f"Benthos {eb:.2e}; |carry| max {np.abs(carry).max():.2e}")
    for k, v in sorted(err.items(), key=lambda kv: -kv[1])[:6]:
        print(f"SP STAGE3 step {step}   {k:>9} {v:.2e}")
    assert max(err.values()) <= STAGE3_SP_TOL and eb <= STAGE3_SP_TOL, err


@need_twin
def test_sp_step_has_no_implicit_promotion():
    """Every float64 use in the SP + REcoM step is explicit (``PR.full``, the WP_full fields, REcoM's float64
    island): no jnp operation mixes a float32 and a float64 operand implicitly."""
    from jax._src import dtypes as D
    sx = _sp_setup()
    one = _sp_step_fn(sx)
    sf = _sp_forcing(sx, 1)[0]
    st = jax.device_put(sx["st32"])
    orig = D.lattice_result_type
    F = {np.dtype("float32"), np.dtype("float64")}
    mixed = []

    def chk(*args):
        fl = set()
        for a in args:
            dt = getattr(a, "dtype", None)
            weak = getattr(a, "weak_type", False) or getattr(getattr(a, "aval", None), "weak_type", False)
            if dt is not None and not weak and np.dtype(dt) in F:
                fl.add(np.dtype(dt))
        if len(fl) == 2:
            import traceback
            fr = [f for f in traceback.extract_stack() if "/fesom_jax/" in f.filename
                  and "/tests/" not in f.filename and "precision.py" not in f.filename]
            mixed.append(" <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}" for f in fr[::-1][:3]) if fr else "?")
        return orig(*args)

    D.lattice_result_type = chk
    try:
        with PR.scope("single"):
            jax.make_jaxpr(functools.partial(one, first=True))(st, sf)
    finally:
        D.lattice_result_type = orig
    assert not mixed, sorted(set(mixed))


# ---- (c) the budget ------------------------------------------------------------------------------------------------
def _closure(s0, sN, mesh, with_carry=True):
    """Per tracer ``|ΔI − (surface + stage 1 + loop)| / scale`` with the inventory ``Σ (X·hnode [+ carry])·areasvol``
    (``mesh`` the WP mesh the run used: its float32 ``areasvol`` widened)."""
    c0 = s0.bgc_carry if with_carry else None
    cN = sN.bgc_carry if with_carry else None
    I0 = wiring.inventory(s0.bgc, s0.hnode, mesh, carry=c0)
    IN = wiring.inventory(sN.bgc, sN.hnode, mesh, carry=cN)
    b = np.asarray(sN.bgc_budget, np.float64).sum(axis=0)
    pred = b[0] + b[1] + b[2]
    scale = np.maximum.reduce([np.abs(I0), np.abs(IN), np.abs(b[:3]).sum(axis=0), np.full_like(I0, 1e-300)])
    return np.abs(IN - I0 - pred) / scale


SP_BUDGET_TOL = 1e-10


@need_twin
@pytest.mark.parametrize("n", [4, 30])
def test_sp_budget_closed(n):
    """The SP budget closes once the carry is counted as content; with the carry dropped before every step it does not
    (what the carry buys). Measured (jobs 28438266 and the plan's S2 table): with the carry the median tracer closes
    to 8e-14 (4 steps) / 2e-14 (30 steps), the worst to 1.7e-11 / 8.6e-12; without it 1e-7 / 7e-7. The DP budget
    closes to <= 7e-15 (test_recom_step). The SP residual above 1e-12 sits in the five tracers that grow by orders of
    magnitude from REcoM's tiny initial value in the first steps (DetCalc, DICremin, PhyCalc, DOC, DON; always a small
    loss, not growing with the step count); stage 3 alone conserves to <= 3e-13 on the G4a inputs (the WP Redi
    increments, as upstream's), so it enters where stage 1 meets the tracer step: not isolated (an open item of the
    plan). SP_BUDGET_TOL leaves an order of magnitude above the measured worst."""
    sx = _sp_setup(budget=True)
    m32 = sx["m32"]
    st = jax.device_get(_sp_run(sx, n))
    rel = _closure(st[0], st[-1], m32)
    neg = jax.device_get(_sp_run(sx, n, zero_carry=True))
    rel_neg = _closure(neg[0], neg[-1], m32, with_carry=False)
    rel_nocount = _closure(st[0], st[-1], m32, with_carry=False)
    w, wn = int(np.argmax(rel)), int(np.argmax(rel_neg))
    print(f"SP BUDGET n={n}: worst {T.NAMES[w]} {rel[w]:.2e}, median {np.median(rel):.2e}; carry dropped every step: "
          f"worst {T.NAMES[wn]} {rel_neg[wn]:.2e}, median {np.median(rel_neg):.2e}; carry kept but not counted: "
          f"worst {rel_nocount.max():.2e}")
    for s in np.argsort(-rel_neg)[:8]:
        print(f"SP BUDGET n={n}   {T.NAMES[s]:>8}: carry {rel[s]:.2e}  dropped {rel_neg[s]:.2e}")
    assert np.all(rel <= SP_BUDGET_TOL), dict(zip(T.NAMES, rel))
    assert np.median(rel_neg) >= 1000 * np.median(rel) and rel_neg.max() >= 1000 * rel.max()


# ---- (d) sharded ---------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not TS.HAVE_TWIN or TS.NDEV < 4 or not (TS.PI_DIST / "dist_4").is_dir(),
                    reason="needs the twin dumps, 4 devices and pi dist_4")
def test_sp_sharded_equals_dense():
    """Two SP steps on FESOM2's pi dist_4 (4 host devices, the forced sharded runner, precision='single') against the
    dense SP steps on every owned node, next to the same DP pair. A sharded run sums in another order, so its
    difference to the dense run is a rounding difference amplified by the model: each BGC tracer's SP difference may be
    at most 10 x (eps32 / eps64) times its DP difference (or FLOOR_FACTOR x the SP physics' own T/S difference). A halo
    defect would show as an O(1) relative error instead. Measured (job 28439736): DOC 1.2e-4 = ~2000 eps32, DON 7e-5,
    the rest <= 6e-6; the DP worst 4e-13 (DetN, test_recom_step) is ~3600 eps64."""
    sx = _sp_setup()
    mesh, cfg = sx["mesh"], sx["cfg"]
    n = 2
    part = partit.read_partition(TS.PI_DIST, 4)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(sx["st"], part)
    stress_p = np.zeros((4, sm.Lmax["elem"], 2))
    from fesom_jax import ice_evp
    bn_p = shard_mesh._shard_along_axis(np.asarray(ice_evp.boundary_node_mask(mesh)), part.myList_nod2D,
                                        sm.Lmax["nod"], 0, False)

    def sharded(precision, cf):
        sop = ssh.partition_ssh_operator(
            ssh.build_ssh_operator(mesh, dt=cfg.dt, precond_variant=1 if precision == "single" else 0), part)
        seq_p = shard_mesh.partition_step_forcing(cf.stack(surface_forcing.dates_for_steps(1958, cfg.dt, n), xp=np),
                                                  part)
        fs_p = shard_mesh.partition_forcing_static(
            jax.tree.map(lambda a: None if a is None else np.asarray(a), wiring.with_recom_static(cf.static, mesh)),
            part)
        return jax.device_get(ish.run_steps_sharded_forced(sm, state_p, sop, stress_p, seq_p, fs_p, n, dt=cfg.dt,
                                                           npes=4, boundary_node_p=bn_p, precision=precision,
                                                           **sx["phys"]))

    out = sharded("single", sx["cf_sp"])
    assert np.asarray(out.bgc).dtype == np.float32 and np.asarray(out.bgc_carry).dtype == np.float32
    assert np.asarray(out.bgc_benthos).dtype == np.float64
    dense = jax.device_get(_sp_run(sx, n)[-1])
    out_dp = sharded("double", sx["cf"])
    dense_dp = jax.device_get(TS._run(sx, n)[-1])
    rel = lambda a, b: float(np.max(np.abs(np.asarray(a, np.float64) - b))) / max(float(np.max(np.abs(b))), 1e-300)  # noqa: E731

    def diffs(o, d):
        dX = np.zeros(T.NTR)
        dT = dS = dB = 0.0
        for p in range(4):
            g = np.asarray(part.myList_nod2D[p][: part.myDim_nod2D[p]])
            m = len(g)
            for s in range(T.NTR):
                dX[s] = max(dX[s], rel(np.asarray(o.bgc)[p, :m, :, s], np.asarray(d.bgc, np.float64)[g, :, s]))
            dT = max(dT, rel(np.asarray(o.T)[p, :m], np.asarray(d.T, np.float64)[g]))
            dS = max(dS, rel(np.asarray(o.S)[p, :m], np.asarray(d.S, np.float64)[g]))
            dB = max(dB, rel(np.asarray(o.bgc_benthos)[p, :m], np.asarray(d.bgc_benthos)[g]))
        return dX, dT, dS, dB

    dX, dT, dS, dB = diffs(out, dense)
    dXd, dTd, dSd, dBd = diffs(out_dp, dense_dp)
    w = int(np.argmax(dX))
    print(f"SP SHARDED vs dense after {n} steps: BGC worst {T.NAMES[w]} rel {dX[w]:.2e} (median {np.median(dX):.2e}), "
          f"Benthos {dB:.2e}; physics T {dT:.2e}, S {dS:.2e} | DP: BGC worst {dXd.max():.2e}, T {dTd:.2e}")
    ratio = float(np.finfo(np.float32).eps / np.finfo(np.float64).eps)
    floor = max(dT, dS, 1e-7)
    bound = np.maximum(FLOOR_FACTOR * floor, 10 * ratio * np.maximum(dXd, np.finfo(np.float64).eps))
    for s in np.argsort(-dX)[:6]:
        print(f"SP SHARDED   {T.NAMES[s]:>9}: SP {dX[s]:.2e} = {dX[s] / np.finfo(np.float32).eps:.0f} eps32 | "
              f"DP {dXd[s]:.2e} = {dXd[s] / np.finfo(np.float64).eps:.0f} eps64 | bound {bound[s]:.2e}")
    assert np.all(dX <= bound), dict(zip(T.NAMES, dX / bound))
    assert dB <= max(FLOOR_FACTOR * floor, 10 * ratio * max(dBd, np.finfo(np.float64).eps))


# ---- (e) restarts through the run driver ---------------------------------------------------------------------------
@need_twin
def test_sp_restart_through_driver(tmp_path):
    """run_from_config with precision: single and full REcoM: the restart holds bgc_carry (float64 on disk) next to
    carry_T/carry_S; read back and cast it is the run's float32 carry exactly; an SP run continues from it; a DP
    restart starts an SP run (the carry from zero) and an SP restart starts a DP run (the carries dropped)."""
    import zarr
    from fesom_jax import zarr_output as zo
    from fesom_jax.run import run_from_config
    fx = TS._setup()
    cfg, mesh = fx["cfg"], fx["mesh"]
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, ser)

    def run(c, state0, out):
        sop = ssh.partition_ssh_operator(
            ssh.build_ssh_operator(mesh, dt=cfg.dt, precond_variant=1 if c.precision == "single" else 0), ser)
        forcing = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=np.asarray(fx["st"].T[:, 0]), dt=cfg.dt,
                                                        mslp=True, bgc_dust_path=cfg.bgc_dust_path(),
                                                        **cfg.forcing_paths())
        return run_from_config(c, mesh=mesh, part=ser, sm=sm, sop=sop, forcing=forcing, state0=state0, year=1958,
                               chunk_steps=2, out_dir=str(out))

    sp_cfg = dataclasses.replace(cfg, precision="single", n_steps=2, duration=None)
    r = run(sp_cfg, fx["st"], tmp_path / "sp1")
    st = jax.device_get(r.state_p)
    assert np.asarray(st.bgc_carry).dtype == np.float32 and float(np.abs(np.asarray(st.bgc_carry)).max()) > 0
    g = zarr.open_group(str(tmp_path / "sp1"), mode="r")
    keys = set(g.array_keys())
    assert {"bgc_carry", "carry_T", "carry_S", "bgc", "bgc_benthos"} <= keys
    assert g["bgc_carry"].dtype == np.float64 and g["bgc"].dtype == np.float64
    loaded, _ = zo.read_restart(tmp_path / "sp1", mesh, ser)
    back = PR.cast_state(loaded)
    n = mesh.nod2D
    np.testing.assert_array_equal(np.asarray(back.bgc_carry)[:n], np.asarray(st.bgc_carry)[:n])
    np.testing.assert_array_equal(np.asarray(back.bgc)[:n], np.asarray(st.bgc)[:n])
    assert np.asarray(back.bgc_benthos).dtype == np.float64
    # SP continues from the SP restart
    r2 = run(dataclasses.replace(sp_cfg, n_steps=1, restart_in=str(tmp_path / "sp1")), None, tmp_path / "sp2")
    assert np.all(np.isfinite(np.asarray(jax.device_get(r2.state_p).bgc)))
    # DP from the SP restart: the carries are dropped
    r3 = run(dataclasses.replace(cfg, n_steps=1, duration=None, restart_in=str(tmp_path / "sp1")), None,
             tmp_path / "dp2")
    s3 = jax.device_get(r3.state_p)
    assert s3.bgc_carry is None and s3.carry_T is None and np.asarray(s3.bgc).dtype == np.float64
    assert "bgc_carry" not in set(zarr.open_group(str(tmp_path / "dp2"), mode="r").array_keys())
    # SP from the DP restart: the BGC carry starts at zero
    r4 = run(dataclasses.replace(sp_cfg, n_steps=1, restart_in=str(tmp_path / "dp2")), None, tmp_path / "sp3")
    s4 = jax.device_get(r4.state_p)
    assert np.asarray(s4.bgc_carry).dtype == np.float32 and np.all(np.isfinite(np.asarray(s4.bgc)))


# ---- (f) refusals and the cast policy ------------------------------------------------------------------------------
def test_sp_refusals_and_config():
    from pathlib import Path
    from fesom_jax.run_config import RunConfig, load_yaml
    repo = Path(__file__).resolve().parents[2]
    d = load_yaml(repo / "configs" / "pi_recom_bgc.yaml").to_dict()
    RunConfig.from_dict(dict(d, precision="single")).validate()               # the in-scope combination
    with pytest.raises(NotImplementedError, match="bgc_every"):
        RunConfig.from_dict(dict(d, precision="single", recom=dict(d["recom"], bgc_every=4))).validate()
    with pytest.raises(NotImplementedError):                                  # linfs
        RunConfig.from_dict(dict(d, precision="single", ale=None)).validate()
    with pytest.raises(NotImplementedError):                                  # not the production physics (EVP)
        RunConfig.from_dict(dict(d, precision="single", ice={"whichEVP": 0})).validate()
    with pytest.raises(NotImplementedError):                                  # KPP instead of TKE
        RunConfig.from_dict(dict(d, precision="single", tke=None, kpp={})).validate()
    with pytest.raises(NotImplementedError, match="bgc_every"):              # the step's trace-time guard
        wiring.check_step_inputs(None, None, object(), True, 2)
    # DP keeps its every-k mode
    RunConfig.from_dict(dict(d, recom=dict(d["recom"], bgc_every=4))).validate()


def test_cast_state_full_and_passive():
    mesh = load_mesh()
    rng = np.random.default_rng(3)
    rcfg = RecomConfig(namelist="x", budget=True)
    g = wiring.with_bgc_state(jax.tree.map(np.asarray, State.rest(mesh)), TR._fake_ic(mesh, rng), rcfg)
    s = PR.cast_state(g)
    assert s.bgc.dtype == np.float32 and s.bgc_old.dtype == np.float32
    assert s.bgc_carry.dtype == np.float32 and s.bgc_carry.shape == s.bgc.shape and not np.any(s.bgc_carry)
    for f in PR.STATE_F64_RECOM:
        assert np.asarray(getattr(s, f)).dtype == np.float64, f
    assert s.bgc_mstep.dtype == np.int32
    d = PR.drop_carry(s)
    assert d.bgc_carry is None and d.carry_T is None
    # the passive mode has no BGC carry (it keeps the plan's Task 2 cost estimate)
    from fesom_jax.recom.transport import init_passive
    p = PR.cast_state(init_passive(jax.tree.map(np.asarray, State.rest(mesh)), 3))
    assert p.bgc.dtype == np.float32 and p.bgc_carry is None
