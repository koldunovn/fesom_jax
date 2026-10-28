"""The sharded adjoint chunk (``core2_adj.build_model(devices=4)``, :mod:`fesom_jax.adjoint_sharded`; the fast-adjoint
plan Task 5) against the dense one on CORE2, in the year-window setting: the Y4raw7 derivative-only cuts, exact sea
ice with the frozen-sweep backward and the year-window ice settings, J_MLD (natl_subpolar, the window mean), the
five scalar controls.

Two steps in one chunk from the 2013-01-01 hindcast restart, the padded halo transport, 4 CPU devices:
* the host layout maps: fold/unfold of a State and of a node field round-trip on the owned lanes, and the folded
  cost weights count each node once;
* with the fast pieces ice2 + se (no CG: every solver explicit or fixed-count) the forward — the unfolded State and
  the MLD running sum — and d J_MLD / d (c_k, c_eps, cd, alpha, k_gm) equal the dense ones to round-off;
* the SE wide halo (FESOM_SE_WIDE=10: one exchange every 10 substeps on 11-ring lanes) keeps forward and gradient
  equal to the dense ones to round-off, and so does the SE divergence's explicit transpose (FESOM_SE_DIV_VJP=1);
* with ice2 + se the chunk's forward also equals the production sharded forward
  (``integrate_sharded.run_steps_sharded_forced``, same partition and transport) to round-off;
* single precision (fast piece sp, ice2 + se + sp): the carry's dtypes are the production runner's (float32, hnode
  float64, the sums float64), the forward equals the production SP sharded forward to float32 round-off (1e-7), and
  the gradient is finite and within 0.1 of the double-precision one (a different model; see TOL_SP_GRAD);
* with the reference physics (the CG SSH solve) no two compiled programs agree to round-off: the CG stops at
  FESOM2's relative residual 1e-5, and two programs that round differently reach different iterates at the SAME trip
  count (117 and 111, FESOM_CG_ITERS, job 28343278), differing at the level of the unconverged error — the chunk
  against the production sharded forward by ~2e-5 (a_ice) after two steps, against the dense one ~2e-4 (u_ice).
  Printed here and bounded loosely.

⚠️ HEAVY (CPU compute node, the whole node, ~1 h): run with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``;
skips with fewer than 4 devices. Two CORE2 programs and their VJPs are compiled (dense and sharded).
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import core2_adj as C  # noqa: E402
import tke_cbar as T  # noqa: E402
import tke_diag as D  # noqa: E402

from fesom_jax import partit  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402

RESTART = C.RESTART_ARCHIVE / "fesom.2013.001.00000"
START = datetime.datetime(2013, 1, 1)
NDEV = 4
N_STEPS = 2
Y4RAW7 = ("tke_pr_cut=1e-6+tke_pr_ratio=100+tke_pr_raw_cut=-1e-6+tke_ml_cut=1e-8+freeze_fct_limiter+tke_ml_ratio=100"
          "+tke_ml_raw_cut=-1e-7+tke_pr_lat_cut=70+tke_pr_ice_cut=0.15")
ICE_ARGS = ["grad", "--ice-damping", "frozen_sweep", "--ice-alpha-kappa", "1", "--ice-theta", "0.97",
            "--ice-sweep-guard", "10", "--ice-thin-mass", "100", "--ice-cuts", "fct_flux+strength_conc"]
CONTROLS = ("ln_tke_c_k", "ln_tke_c_eps", "ln_tke_cd", "ln_tke_alpha", "ln_k_gm")
TOL_STATE = 1e-9          # max |Δ| / max |x|, the unfolded State after two steps
# relative, the five control gradients: measured 2.8e-9 (k_gm) to 4.9e-7 (cd, the smallest, 6.4e-4), job 28343746,
# with the forward equal to 1e-12: round-off through the reverse pass (the GPU repeat spread of a 12-day window is
# ~1e-5)
TOL_GRAD = 1e-6

pytestmark = [
    pytest.mark.skipif(not ((ROOT / "data" / "mesh_core2").is_dir() and RESTART.is_dir()
                            and (C.CORE2_DIST / f"dist_{NDEV}").is_dir()),
                       reason="CORE2 mesh / 2013 restart / dist_4 missing"),
    pytest.mark.skipif(len(jax.devices()) < NDEV, reason=f"needs {NDEV} devices"),
]


FIELDS = ("T", "S", "uv", "eta_n", "tke", "a_ice", "m_ice", "u_ice")
TOL_CG = 1e-2             # the reference physics, any two programs: the CG's 1e-5 stop, amplified (see above)


def _model(devices, fast=None, **kw):
    return C.build_model(START, n_steps=N_STEPS, chunk_steps=N_STEPS, restart=RESTART,
                         adjoint_cfg=D.parse_setting(Y4RAW7), ice_mode="exact",
                         ice_adjoint=C._ice_adjoint_args(C._parse(ICE_ARGS)),
                         accumulators=T.mld_accumulators, devices=devices, fast=fast, **kw)


@pytest.fixture(scope="module")
def models():
    return _model(1, "ice2+se"), _model(NDEV, "ice2+se")


def _forward(m):
    return jax.block_until_ready(C.adjoint._jitted_forward(m.chunk_fn)(m.init_carry(), m.xs_fn()(0),
                                                                       jax.device_put(m.params0)))


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)))
                 / max(float(np.max(np.abs(np.asarray(b, np.float64)))), 1e-300))


def test_layout_maps_round_trip(models):
    _, ms = models
    sh = ms.shard
    st = ms.s0
    back = sh.unfold_state(sh.fold_state(st))
    for f in ("T", "S", "uv", "eta_n", "a_ice", "helem", "tke"):
        np.testing.assert_array_equal(np.asarray(getattr(back, f)), np.asarray(getattr(st, f)), err_msg=f)
    v = np.arange(ms.mesh.nod2D, dtype=np.float64)
    vf = sh.fold_nodes(v)
    np.testing.assert_array_equal(sh.unfold_nodes(vf), v)
    assert vf.sum() == v.sum()                              # each node once: zero on halo and pad lanes
    assert ms.devices == NDEV and ms.shard.transport == "padded"


def test_forward_and_gradient_match_the_dense_chunk(models):
    """ice2 + se: no iterative solver, so sharded == dense to round-off, forward and gradient."""
    md, ms = models
    n = N_STEPS
    fwd_d, fwd_s = _forward(md), _forward(ms)
    st_d, st_s = jax.device_get(fwd_d[0]), ms.shard.unfold_state(fwd_s[0])
    worst = {f: _rel(getattr(st_s, f), getattr(st_d, f)) for f in FIELDS}
    mld_rel = _rel(ms.shard.unfold_nodes(fwd_s[1]["mld"]), np.asarray(fwd_d[1]["mld"]))
    print(f"\n[core2_adj sharded fwd ice2+se] state rel {worst}  mld rel {mld_rel:.2e}")
    # the pad lanes of the U ring keep their start values (their pad Coriolis would make them overflow)
    pad = ~np.asarray(ms.shard.sm.valid_mask["elem"]).reshape(-1)
    u0, u2 = np.asarray(ms.init_carry(host=True)[0].se_ubt), np.asarray(jax.device_get(fwd_s[0].se_ubt))
    assert pad.any() and np.array_equal(u2[pad], u0[pad])
    assert max(worst.values()) <= TOL_STATE and mld_rel <= TOL_STATE

    lossd = T.cost_fn("MLD", n, md.mesh)
    losss = T.cost_fn("MLD", n, ms.mesh, fold=ms.shard.fold_nodes)
    assert abs(float(losss(fwd_s)) - float(lossd(fwd_d))) <= 1e-10 * abs(float(lossd(fwd_d)))
    rd = C.run_grad(md, (lossd,), n_steps=n, check_linear=False)[0]
    rs = C.run_grad(ms, (losss,), n_steps=n, check_linear=False)[0]
    assert rd["status"] == rs["status"] == "ok" and rs["devices"] == NDEV
    gd, gs = rd["_result"]["grad"], rs["_result"]["grad"]
    rows = {k: (float(gs[k]), float(gd[k]), abs(float(gs[k]) - float(gd[k])) / abs(float(gd[k]))) for k in CONTROLS}
    print("[core2_adj sharded grad] " + "  ".join(f"{k} {a:+.10e} vs {b:+.10e} rel {r:.1e}"
                                                 for k, (a, b, r) in rows.items()))
    print(f"[core2_adj sharded grad] wall dense {rd.get('wall_s', float('nan')):.0f} s, sharded "
          f"{rs.get('wall_s', float('nan')):.0f} s")
    for k, (a, b, r) in rows.items():
        assert np.isfinite(a) and b != 0.0 and r <= TOL_GRAD, (k, a, b, r)


def _production(m):
    """``run_steps_sharded_forced`` over the chunk's steps from the model's start State (folded, padded)."""
    from fesom_jax import integrate_sharded as ish
    from fesom_jax import shard_mesh
    sh = m.shard
    sf = m.stack_forcing([m.date(m.start, k) for k in range(N_STEPS)])
    out = ish.run_steps_sharded_forced(
        sh.sm, sh.fold_state(m.s0), shard_mesh_sop(m), np.zeros((NDEV, sh.sm.Lmax["elem"], 2)),
        shard_mesh.partition_step_forcing(sf, sh.part), shard_mesh.partition_forcing_static(m.forcing.static,
                                                                                            sh.part),
        N_STEPS, dt=m.dt, npes=NDEV, params=m.to_params(jax.device_put(m.params0)), boundary_node_p=boundary_node_p(m),
        use_padded=True, bootstrap_ab2=False, state_is_folded=True, return_folded=True, **dict(m.phys))
    return sh.unfold_state(out)


def test_chunk_is_the_production_sharded_forward(models):
    """ice2 + se: the chunk's forward == ``run_steps_sharded_forced`` to round-off."""
    _, ms = models
    st_c, st_p = ms.shard.unfold_state(_forward(ms)[0]), _production(ms)
    vs_prod = {f: _rel(getattr(st_c, f), getattr(st_p, f)) for f in FIELDS}
    print(f"\n[core2_adj sharded fwd ice2+se] chunk vs production sharded {vs_prod}")
    assert max(vs_prod.values()) <= TOL_STATE


def test_reference_physics_cg_bounds():
    """The reference physics (CG): the chunk against the production sharded and the dense forward, loosely."""
    md, ms = _model(1), _model(NDEV)
    st_c, st_p, st_d = ms.shard.unfold_state(_forward(ms)[0]), _production(ms), jax.device_get(_forward(md)[0])
    vs_prod = {f: _rel(getattr(st_c, f), getattr(st_p, f)) for f in FIELDS}
    vs_dense = {f: _rel(getattr(st_c, f), getattr(st_d, f)) for f in FIELDS}
    print(f"\n[core2_adj sharded fwd CG] chunk vs production sharded {vs_prod}\n"
          f"[core2_adj sharded fwd CG] chunk vs dense {vs_dense}")
    assert max(vs_prod.values()) <= TOL_CG and max(vs_dense.values()) <= TOL_CG


def shard_mesh_sop(m):
    from fesom_jax import ssh
    return ssh.partition_ssh_operator(m.op, m.shard.part)


def boundary_node_p(m):
    from fesom_jax import ice_evp, shard_mesh
    return shard_mesh._shard_along_axis(np.asarray(ice_evp.boundary_node_mask(m.mesh)), m.shard.part.myList_nod2D,
                                        m.shard.sm.Lmax["nod"], 0, False)


TOL_SP_STATE = 1e-4       # SP chunk vs SP production forward: float32 round-off of two programs over two steps
# SP vs DP gradient, two steps: two different models, not a measure of the SP adjoint's accuracy (that is SP adjoint
# vs SP FD, on the GPU): measured 9e-5 (k_gm), 2e-3...7.5e-3 (alpha, c_k, c_eps), 6.2e-2 (cd, the smallest, 6e-4),
# job 28346856 — the window-mean MLD's threshold reacts to the SP rounding of the state at single nodes
TOL_SP_GRAD = 0.1


def test_single_precision_chunk(models):
    _, ms = models
    m = _model(NDEV, "ice2+se+sp")
    assert m.shard.precision == "single" and m.chunk_fn.precision == "single" and m.forcing.point_slope
    c0 = m.init_carry(host=True)
    assert c0[0].T.dtype == np.float32 and c0[0].hnode.dtype == np.float64 and c0[0].carry_T is not None
    assert c0[1]["mld"].dtype == np.float64
    fwd = _forward(m)
    assert fwd[0].T.dtype == np.float32 and fwd[1]["mld"].dtype == np.float64
    from fesom_jax import integrate_sharded as ish
    from fesom_jax import shard_mesh
    sh = m.shard
    sf = m.stack_forcing([m.date(m.start, k) for k in range(N_STEPS)])
    prod = ish.run_steps_sharded_forced(
        sh.sm, jax.tree.map(ish._fold, shard_mesh.partition_state(m.s0, sh.part)), shard_mesh_sop(m),
        np.zeros((NDEV, sh.sm.Lmax["elem"], 2)), shard_mesh.partition_step_forcing(sf, sh.part),
        shard_mesh.partition_forcing_static(m.forcing.static, sh.part), N_STEPS, dt=m.dt, npes=NDEV,
        params=m.to_params(jax.device_put(m.params0)), boundary_node_p=boundary_node_p(m), use_padded=True,
        bootstrap_ab2=False, state_is_folded=True, return_folded=True, precision="single", **dict(m.phys))
    st_c, st_p = sh.unfold_state(fwd[0]), sh.unfold_state(prod)
    vs_prod = {f: _rel(getattr(st_c, f), getattr(st_p, f)) for f in FIELDS}
    print(f"\n[core2_adj sharded fwd SP] chunk vs production SP sharded {vs_prod}")
    assert max(vs_prod.values()) <= TOL_SP_STATE
    loss_sp = T.cost_fn("MLD", N_STEPS, m.mesh, fold=sh.fold_nodes)
    loss_dp = T.cost_fn("MLD", N_STEPS, ms.mesh, fold=ms.shard.fold_nodes)
    g_sp = C.run_grad(m, (loss_sp,), n_steps=N_STEPS, check_linear=False)[0]["_result"]["grad"]
    g_dp = C.run_grad(ms, (loss_dp,), n_steps=N_STEPS, check_linear=False)[0]["_result"]["grad"]
    rows = {k: (float(g_sp[k]), float(g_dp[k]), abs(float(g_sp[k]) - float(g_dp[k])) / abs(float(g_dp[k])))
            for k in CONTROLS}
    print("[core2_adj sharded grad SP] " + "  ".join(f"{k} {a:+.8e} vs DP {b:+.8e} rel {r:.1e}"
                                                    for k, (a, b, r) in rows.items()))
    for k, (a, b, r) in rows.items():
        assert np.isfinite(a) and r <= TOL_SP_GRAD, (k, a, b, r)


def test_se_wide_halo_chunk(models, monkeypatch):
    """FESOM_SE_WIDE=10 (M 40 at dt 1800: four windows of 10): forward and gradient == dense to round-off."""
    md, _ = models
    monkeypatch.setenv("FESOM_SE_WIDE", "10")
    m = _model(NDEV, "ice2+se")
    assert m.shard.se_wide == 10
    fwd_d, fwd_s = _forward(md), _forward(m)
    st_d, st_s = jax.device_get(fwd_d[0]), m.shard.unfold_state(fwd_s[0])
    worst = {f: _rel(getattr(st_s, f), getattr(st_d, f)) for f in FIELDS}
    print(f"\n[core2_adj sharded fwd se_wide 10] state rel {worst}")
    assert max(worst.values()) <= TOL_STATE
    lossd = T.cost_fn("MLD", N_STEPS, md.mesh)
    losss = T.cost_fn("MLD", N_STEPS, m.mesh, fold=m.shard.fold_nodes)
    gd = C.run_grad(md, (lossd,), n_steps=N_STEPS, check_linear=False)[0]["_result"]["grad"]
    rs = C.run_grad(m, (losss,), n_steps=N_STEPS, check_linear=False)[0]
    assert rs["se_wide"] == 10 and rs["fesom_env"]["FESOM_SE_WIDE"] == "10"
    gs = rs["_result"]["grad"]
    rows = {k: abs(float(gs[k]) - float(gd[k])) / abs(float(gd[k])) for k in CONTROLS}
    print("[core2_adj sharded grad se_wide 10] " + "  ".join(f"{k} rel {r:.1e}" for k, r in rows.items()))
    assert max(rows.values()) <= TOL_GRAD


def test_se_div_vjp_chunk(models, monkeypatch):
    """FESOM_SE_DIV_VJP=1 (the divergence's explicit transpose, ordered gathers): the gradient == dense to round-off."""
    md, _ = models
    monkeypatch.setenv("FESOM_SE_DIV_VJP", "1")
    m = _model(NDEV, "ice2+se")
    lossd = T.cost_fn("MLD", N_STEPS, md.mesh)
    losss = T.cost_fn("MLD", N_STEPS, m.mesh, fold=m.shard.fold_nodes)
    gd = C.run_grad(md, (lossd,), n_steps=N_STEPS, check_linear=False)[0]["_result"]["grad"]
    rs = C.run_grad(m, (losss,), n_steps=N_STEPS, check_linear=False)[0]
    assert rs["fesom_env"]["FESOM_SE_DIV_VJP"] == "1"
    gs = rs["_result"]["grad"]
    rows = {k: abs(float(gs[k]) - float(gd[k])) / abs(float(gd[k])) for k in CONTROLS}
    print("\n[core2_adj sharded grad se_div_vjp] " + "  ".join(f"{k} rel {r:.1e}" for k, r in rows.items()))
    assert max(rows.values()) <= TOL_GRAD


def test_ordered_vjp_chunk(models, monkeypatch):
    """FESOM_ORDERED_SUMS=1 with and without FESOM_ORDERED_VJP=1 (the ordered sums' explicit gather transpose): the two
    sharded gradients agree to round-off (each entry of a gather transpose gets ONE cotangent value, no sum), and both
    are the dense one to TOL_ORDERED (the ordered sums change the forward's summation order: 1.9e-6 on cd, the
    smallest, job 28360094)."""
    md, _ = models
    lossd = T.cost_fn("MLD", N_STEPS, md.mesh)
    gd = C.run_grad(md, (lossd,), n_steps=N_STEPS, check_linear=False)[0]["_result"]["grad"]
    monkeypatch.setenv("FESOM_ORDERED_SUMS", "1")
    g = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("FESOM_ORDERED_VJP", flag)
        m = _model(NDEV, "ice2+se")
        losss = T.cost_fn("MLD", N_STEPS, m.mesh, fold=m.shard.fold_nodes)
        g[flag] = C.run_grad(m, (losss,), n_steps=N_STEPS, check_linear=False)[0]["_result"]["grad"]
    vs_off = {k: abs(float(g["1"][k]) - float(g["0"][k])) / abs(float(g["0"][k])) for k in CONTROLS}
    vs_dense = {k: abs(float(g["1"][k]) - float(gd[k])) / abs(float(gd[k])) for k in CONTROLS}
    print("\n[core2_adj sharded grad ordered_vjp] vs off " + "  ".join(f"{k} {r:.1e}" for k, r in vs_off.items())
          + "\n   vs dense " + "  ".join(f"{k} {r:.1e}" for k, r in vs_dense.items()))
    assert max(vs_off.values()) <= TOL_STATE and max(vs_dense.values()) <= TOL_ORDERED


TOL_ORDERED = 1e-5        # the ordered sums' forward against the dense segment sums, the gradient after two steps


FCTRL = dict(forcing_ctrl=("Tair", "u_air"), n_months=2)
# The map gradients, per node (job 28374155): on the 1216 halo nodes of dist_4 (the lanes the localize transpose sums)
# they agree to round-off; the map sums (Tair 5.8e-7, u_air 2.4e-6) and L2 (1.2e-5, 3.0e-5) do too, but 3 (Tair) / 7
# (u_air) interior nodes of device 3 differ by up to 1.2e-4 of the map maximum (51 % / 190 % of their own, ~1e-8,
# values): compact snow-free Antarctic summer ice at the freezing point, two with the skin temperature at the 0 C
# clamp — a forward-continuous clamp (min(t, 0), max(h, 0)) whose derivative round-off picks; the forward agrees to
# 1e-12. A localize/transpose error would show on the halo nodes.
TOL_MAP_SUM = 1e-5        # the map's sum, relative
MAX_SWITCH_NODES = 10     # interior nodes allowed beyond TOL_GRAD x the map maximum


def test_forcing_ctrl_chunk(models):
    """Monthly atmospheric maps (Params.forcing_adj [M, nod2D], the schedule's month): a zero anomaly is the plain
    forward bit for bit; the map gradients (Tair, u_air) sharded == dense to round-off; the second month (no step of
    the window in it) gets none; FD of a box-uniform Tair anomaly against the AD box sum (dense)."""
    md0, _ = models
    md, ms = _model(1, "ice2+se", **FCTRL), _model(NDEV, "ice2+se", **FCTRL)
    assert md.schedule["n_months"] == 2 and md.params0["forcing_adj"]["Tair"].shape == (2, md.mesh.nod2D)
    f0, fd_ = jax.device_get(_forward(md0)), jax.device_get(_forward(md))
    for a, b in zip(jax.tree.leaves(f0), jax.tree.leaves(fd_)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    lossd = T.cost_fn("MLD", N_STEPS, md.mesh)
    losss = T.cost_fn("MLD", N_STEPS, ms.mesh, fold=ms.shard.fold_nodes)
    rd = C.run_grad(md, (lossd,), n_steps=N_STEPS, check_linear=False)[0]
    rs = C.run_grad(ms, (losss,), n_steps=N_STEPS, check_linear=False)[0]
    assert rs["forcing_ctrl"] == ["Tair", "u_air"] and rs["schedule"]["n_months"] == 2
    gd, gs = rd["_result"]["grad"], rs["_result"]["grad"]
    lon, lat = C._lonlat_deg(md.mesh)
    part = partit.read_partition(C.CORE2_DIST, NDEV)
    halo = np.zeros(md.mesh.nod2D, bool)
    for dv in range(part.npes):
        lst, my, e = np.asarray(part.myList_nod2D[dv]), int(part.myDim_nod2D[dv]), int(part.eDim_nod2D[dv])
        halo[lst[my:my + e]] = True
    res = {}
    for k in ("Tair", "u_air"):
        a, b = np.asarray(gs["forcing_adj"][k]), np.asarray(gd["forcing_adj"][k])
        d, m = np.abs(a[0] - b[0]), np.max(np.abs(b[0]))
        r = float(np.max(d) / m)
        l2 = float(np.linalg.norm(a[0] - b[0]) / np.linalg.norm(b[0]))
        print(f"\n[core2_adj forcing_ctrl] {k}: sharded vs dense max rel {r:.1e}, L2 rel {l2:.1e}, max |g| {m:.3e}, "
              f"sum rel {abs(a[0].sum() - b[0].sum()) / abs(b[0].sum()):.1e}, nodes |d| > 1e-6 max: "
              f"{int((d > 1e-6 * m).sum())}, > 1e-4 max: {int((d > 1e-4 * m).sum())} of {int((b[0] != 0).sum())} "
              f"nonzero")
        for i in np.argsort(d)[::-1][:6]:
            print(f"[core2_adj forcing_ctrl]   node {i} ({lon[i]:7.2f}E {lat[i]:6.2f}N, halo {bool(halo[i])}): dense "
                  f"{b[0, i]:+.6e} sharded {a[0, i]:+.6e} rel {d[i] / max(abs(b[0, i]), 1e-300):.1e}")
        r_halo = float(np.max(d[halo]) / m)
        print(f"[core2_adj forcing_ctrl]   {int(halo.sum())} halo nodes: max rel {r_halo:.1e}")
        res[k] = (a, b, r_halo, int((d > TOL_GRAD * m).sum()), abs(a[0].sum() - b[0].sum()) / abs(b[0].sum()))
    for k, (a, b, r_halo, n_bad, r_sum) in res.items():
        assert np.max(np.abs(b[0])) > 0 and np.all(b[1] == 0) and np.all(a[1] == 0), k
        assert r_halo <= TOL_GRAD and r_sum <= TOL_MAP_SUM and n_bad <= MAX_SWITCH_NODES, k
    # FD (dense): Tair + h on the cost box's surface nodes, month 0, against the box-mean SST at the window end, matched
    # by the EXACT adjoint (no derivative-only cuts; the year setting's AD is printed beside it, 0.6 % apart). h is
    # 1e-4: the winter subpolar box's forward jumps at switches between, so the FD at h 1e-2..1 scatters 9-37 % around
    # the AD while at h 1e-4 it is the AD to 1e-6, central, forward and backward alike (h sweep, job 28375980)
    node, _ = C.box_masks(md.mesh, C.DEFAULT_BOX)
    w = jnp.asarray(np.where(node, np.asarray(md.mesh.area)[:, 0], 0.0))

    def loss_sst(carry):
        return jnp.sum(w * carry[0].T[:, 0]) / jnp.sum(w)
    d = np.zeros((2, md.mesh.nod2D))
    d[0, node] = 1.0
    me = C.build_model(START, n_steps=N_STEPS, chunk_steps=N_STEPS, restart=RESTART, adjoint_cfg=AdjointConfig.exact(),
                       ice_mode="exact", accumulators=T.mld_accumulators, fast="ice2+se", **FCTRL)
    ads = {}
    for name, m in (("year setting", md), ("exact", me)):
        g_sst = C.run_grad(m, (loss_sst,), n_steps=N_STEPS, check_linear=False)[0]["_result"]["grad"]
        ads[name] = float(np.sum(np.asarray(g_sst["forcing_adj"]["Tair"]) * d))
    fwd = C.adjoint._jitted_forward(md.chunk_fn)

    def J(h):
        u = dict(md.params0)
        u["forcing_adj"] = dict(u["forcing_adj"], Tair=u["forcing_adj"]["Tair"] + h * d)
        return float(loss_sst(fwd(md.init_carry(), md.xs_fn()(0), jax.device_put(C._strong_f64(u)))))
    fds = {h: (J(h) - J(-h)) / (2 * h) for h in (1e-4,)}
    for h, fd in fds.items():
        print(f"[core2_adj forcing_ctrl] Tair box, FD h {h:g}: {fd:.6e}; AD " + ", ".join(
            f"{k} {v:.6e} (rel {abs(v - fd) / abs(fd):.1e})" for k, v in ads.items()))
    fd = fds[1e-4]
    assert abs(ads["exact"] - fd) <= 1e-3 * abs(fd)
