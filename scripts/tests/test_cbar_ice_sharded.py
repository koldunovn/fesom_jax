"""The sharded drift gradient of ``scripts/adjoint/ice_cbar.py grad --dist`` equals the dense one.

The farc ice adjoint does not fit one GPU, so its reverse sweep runs the chunk program on the
devices of one node (``core2_adj.build_model(dist=)``, ``integrate_sharded.sharded_chunk_program``)
over the boundaries of a ONE-device base run, folded on load (``ice_cbar.FoldingLegCheckpoint``).
This file checks that path end to end on CORE2 (2013-01-01 restart, ``configs/core2_full.yaml``,
4-step window, 2-step chunks) with the farc settings: exact ice mode, frozen_sweep (κ0 = 1,
θ = 0.97), freeze_mixing + freeze_gm_slope + freeze_fct_limiter, nested remat.

Gates, against the dense sweep over the same base:

* d J_drift / d ln P*, d ln C_d: sharded == dense to 1e-3 relative, for the allgather, padded and
  coloured halo transports. Not to rounding: the sharded FORWARD is not bitwise the dense one
  (FCT upwind flips, SSH cancellation; test_step_sharded.py gates the ice scalars at 2e-2), and
  this adjoint amplifies a rounding change. Measured: CPU 4.7e-5 (P*) and 2.5e-4 (C_d) for every
  transport (job 28047661); on GH200 the dense gradient itself spreads 7e-4 between runs
  (28047507-9). At ONE step the drift VJP agrees to 1e-15 (28047151);
* the three transports against each other (the same program with another exchange, whose
  transposes sum the halo copies in another order): CPU 1e-6 for the gradients (measured 2e-7)
  and 1e-2 relative L2 for each initial-state cotangent field (measured 2e-6 to 7e-6, a_ice
  4e-3; job 28048168; a mis-paired exchange gave 1e5), GPU 1e-3 for the gradients;
* the initial-state cotangent against the dense one (every copy of an entity summed,
  ``Model.dense_cotangent``) is PRINTED, not gated: it follows the trajectory more closely than
  the integrated gradient (CPU relative L2: T 3e-3, eta_n 5e-2, a_ice 1e-2, u_ice/v_ice 0.23 —
  the frozen-sweep backward is linearized at a near-kink final iterate);
* coloured: the sweep cut into legs by a zero budget and resumed gives the same gradient (the
  reverse leg files hold folded cotangents); with ``stride`` 2 the reverse walk rebuilds the
  span with the SHARDED forward from the folded boundary 0, so chunk 1 is linearized at the
  sharded state instead of the dense base one: 3e-3 (measured P* 1.2e-3, C_d 7e-5, job
  28048846; the farc 30/90 d runs use strides 10/20).

* the terminal ice volume (``ice_cbar.ice_cost_loss`` over the folded carry, weights on the
  owner lanes only): its value on the folded start carry equals the dense one, its seed summed
  over the copies (``Model.dense_cotangent``) equals the dense seed exactly, and its coloured
  gradient the dense one, ln C_d to :data:`TOL_GRAD`, ln P* to :data:`TOL_ICE_PSTAR` (measured
  9e-5 and 1.2e-3, job 28195465: the 4-step volume response to P* is ~1e-7 of J and its FD does
  not converge in h, test_cbar_ice_core2.py; a rounding change of the forward moves it more)
  (``-k ice_cost``).

That the sharded adjoint is the adjoint of the sharded forward, controls and initial ice
velocity included, is test_cbar_ice_sharded_fd.py (one process each: CPU JIT memory).

⚠️ HEAVY: two GPUs (dolpung, ``runs/dolpung/job_gh200x4.sbatch --gpus=2``), or a CPU compute node
(>= 200 GB, ``XLA_FLAGS=--xla_force_host_platform_device_count=2``). Skips with fewer devices or
without the CORE2 dist_2 partition.

⚠️ On CPU fake devices the XLA:CPU runtime starts independent collectives in a different order
on different devices: the 4-step chunk VJP deadlocked in a rendezvous (all-reduce against
all-gather, jobs 28047151 and 28047362), and with the padded and coloured transports, where many
same-shaped collectives can pair with the wrong partner, it returned wrong cotangents without
an error (u_ice 0.9 and 1e5 relative, jobs 28047363/4). With XLA_FLAGS
``--xla_cpu_enable_concurrency_optimized_scheduler=false`` the three transports agree to ~1e-9
(job 28047661); without it this file skips on CPU.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import core2_adj as C  # noqa: E402
import ice_cbar as I  # noqa: E402
import tke_diag as D  # noqa: E402

RESTART = C.STATES_DIR / "fesom.2013.001.00000"
DIST = C.POOL_MESHES / "core2"
NPES = 2
N, K = 4, 2
SETTING = "freeze_mixing+freeze_gm_slope+freeze_fct_limiter"
ICE_ADJ = {"adjoint_damping": "frozen_sweep", "adjoint_alpha_kappa": 1.0, "adjoint_theta": 0.97}
TOL_GRAD = 1e-3          # relative (see the module docstring for the measured noise)
TOL_CROSS_CPU = 1e-6     # gradients, transport against transport on CPU (measured 2e-7)
TOL_CROSS_FIELD = 1e-2   # initial-state cotangents, relative L2, the same (measured <= 4e-3)
TOL_ICE_PSTAR = 1e-2     # the terminal volume's ln P* gradient, sharded vs dense (measured 1.2e-3)
TOL_STRIDE = 3e-3        # stride 2 against stride 1 (another linearization point; measured 1.2e-3)
TRANSPORTS = ("allgather", "padded", "coloured")
FIELDS = ("T", "S", "uv", "eta_n", "hnode", "a_ice", "m_ice", "u_ice", "v_ice", "sigma11")

_CPU_FLAG = "--xla_cpu_enable_concurrency_optimized_scheduler=false"

pytestmark = [
    pytest.mark.skipif(jax.devices()[0].platform == "cpu"
                       and _CPU_FLAG not in os.environ.get("XLA_FLAGS", ""),
                       reason=f"CPU fake devices mis-order collectives without XLA_FLAGS {_CPU_FLAG}"),
    pytest.mark.skipif(not ((ROOT / "data" / "mesh_core2").is_dir() and RESTART.is_dir()
                            and (DIST / f"dist_{NPES}").is_dir()),
                       reason="CORE2 mesh / 2013 States / dist_2 partition missing"),
    pytest.mark.skipif(len(jax.devices()) < NPES, reason=f"needs {NPES} devices"),
]


def build(**kw):
    return I.build("2013-01-01", N, K, adjoint_cfg=D.parse_setting(SETTING), ice_mode="exact",
                   ice_adjoint=ICE_ADJ, controls=("pstar", "cd"), remat_blocks=True, **kw)


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    m = build()
    root = tmp_path_factory.mktemp("ice_sh")
    base = root / "base"
    assert I.run_fwd(m, base, params=m.params0, n_steps=N, save_boundaries=True)["status"] \
        == "complete"
    r = I.run_grad(m, root / "g_dense", base, n_steps=N, budget_s=1e9)
    assert r["status"] == "ok"
    return dict(model=m, base=base, root=root, grads=r["grads"],
                cot=jax.device_get(r["_result"]["state_cot"]))


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def check_grads(tag, got, ref, tol=TOL_GRAD) -> list:
    """Print both control gradients; return the failures."""
    bad = []
    for c in ("pstar", "cd"):
        print(f"[ice-sharded {tag}] d/d ln {c}: {got[c]:+.15e} vs {ref[c]:+.15e} "
              f"rel {rel(got[c], ref[c]):.2e}")
        if not (np.isfinite(got[c]) and got[c] != 0.0 and rel(got[c], ref[c]) < tol):
            bad.append((tag, c, got[c], ref[c]))
    return bad


def field_rel(a_st, b_st) -> dict:
    """{field: (relative L2, relative max)} of a against b over :data:`FIELDS`."""
    out = {}
    for f in FIELDS:
        a, b = np.asarray(getattr(a_st, f)), np.asarray(getattr(b_st, f))
        n2, mx = float(np.linalg.norm(b)), float(np.max(np.abs(b)))
        out[f] = (float(np.linalg.norm(a - b)) / n2 if n2 > 0 else float(np.linalg.norm(a)),
                  float(np.max(np.abs(a - b))) / mx if mx > 0 else float(np.max(np.abs(a))))
    return out


def test_sharded_drift_gradient_equals_dense(dense):
    cpu = jax.devices()[0].platform == "cpu"
    tol_x = TOL_CROSS_CPU if cpu else TOL_GRAD
    res, bad = {}, []
    for transport in TRANSPORTS:
        m = build(dist=NPES, transport=transport)
        assert m.sharded is not None and m.sharded.npes == NPES
        r = I.run_grad(m, dense["root"] / transport / "g", dense["base"], n_steps=N,
                       budget_s=1e9)
        assert r["status"] == "ok", r.get("status")
        assert r["dist"] == NPES and r["transport"] == transport
        st = m.dense_cotangent(jax.device_get(r["_result"]["state_cot"]))[0]
        assert all(np.isfinite(np.asarray(getattr(st, f))).all() for f in FIELDS), transport
        bad += check_grads(f"{transport} vs dense", r["grads"], dense["grads"])
        for f, (d2, dmax) in field_rel(st, dense["cot"][0]).items():
            print(f"[ice-sharded {transport} vs dense] state cot {f:8s} rel L2 {d2:.2e} "
                  f"rel max {dmax:.2e}")
        res[transport] = (m, r, st)
    # the transports are three exchanges of the same program
    ref_t = TRANSPORTS[0]
    for transport in TRANSPORTS[1:]:
        bad += check_grads(f"{transport} vs {ref_t}", res[transport][1]["grads"],
                           res[ref_t][1]["grads"], tol=tol_x)
        if cpu:
            for f, (d2, _) in field_rel(res[transport][2], res[ref_t][2]).items():
                if d2 > TOL_CROSS_FIELD:
                    bad.append((transport, ref_t, f, d2))
    assert not bad, bad

    # coloured (the production transport): the span rebuilt by the sharded forward from the
    # folded boundary 0, and the sweep cut into legs by a zero budget and resumed
    m, r, _ = res["coloured"]
    root = dense["root"] / "coloured"
    r2 = I.run_grad(m, root / "g_stride2", dense["base"], n_steps=N, budget_s=1e9, stride=2)
    assert r2["status"] == "ok"
    assert not check_grads("coloured stride 2", r2["grads"], r["grads"], tol=TOL_STRIDE)
    legs = root / "legs"
    r3 = I.run_grad(m, legs, dense["base"], n_steps=N, budget_s=0.0)
    assert r3["status"] == "incomplete"
    r4 = I.run_grad(m, legs, dense["base"], n_steps=N, budget_s=1e9)
    assert r4["status"] == "ok"
    assert not check_grads("coloured legs", r4["grads"], r["grads"], tol=tol_x)


def test_sharded_ice_cost_gradient_equals_dense(dense):
    """The terminal ice volume over the folded carry (weights on the owner lanes only): its
    value on the folded start carry and its gradient equal the dense ones."""
    m_d = dense["model"]
    m = build(dist=NPES, transport="coloured")
    w = I.arctic_weights(m.mesh, "arctic")
    assert float(np.sum(I.folded_node_weights(m, w))) == pytest.approx(float(np.sum(w)), rel=1e-14)
    j_s = float(I.ice_cost_loss(m, "vol", "arctic")(m.init_carry(host=True)))
    j_d = float(I.ice_cost_loss(m_d, "vol", "arctic")(m_d.init_carry(host=True)))
    assert j_s == pytest.approx(j_d, rel=1e-13)
    st, acc = m.init_carry(host=True)
    loss = I.ice_cost_loss(m, "vol", "arctic")
    g = jax.grad(lambda mi: loss((dataclasses.replace(st, m_ice=mi), acc)))(jnp.asarray(st.m_ice))
    cot = (dataclasses.replace(jax.tree.map(np.zeros_like, st), m_ice=np.asarray(g)),
           jax.tree.map(np.zeros_like, acc))
    np.testing.assert_allclose(np.asarray(m.dense_cotangent(cot)[0].m_ice), w / 1e12,
                               rtol=1e-15, atol=0)
    rd = I.run_grad(m_d, dense["root"] / "vol_dense", dense["base"], n_steps=N, budget_s=1e9,
                    cost="vol", cost_box="arctic")
    rs = I.run_grad(m, dense["root"] / "vol_coloured", dense["base"], n_steps=N, budget_s=1e9,
                    cost="vol", cost_box="arctic")
    assert rd["status"] == "ok" and rs["status"] == "ok"
    bad = check_grads("vol coloured vs dense", rs["grads"], rd["grads"])
    assert not [b for b in bad if b[1] == "cd"], bad
    assert not check_grads("vol coloured vs dense (P* bound)", rs["grads"], rd["grads"],
                           tol=TOL_ICE_PSTAR), bad


def test_sharded_coast_cut_gradient_equals_dense(dense):
    """IceConfig.adjoint_cuts "fct_pin" + "fct_bound_coast": each lane builds the coast ring
    (``ice_adv.coast_ring``) from the partitioned global boundary mask and its local elements, so
    the sharded terminal-volume gradient equals the dense one with the same cuts."""
    adj = dict(ICE_ADJ, adjoint_cuts=("fct_pin", "fct_bound_coast"))

    def mk(**kw):
        return I.build("2013-01-01", N, K, adjoint_cfg=D.parse_setting(SETTING), ice_mode="exact",
                       ice_adjoint=adj, controls=("pstar", "cd"), remat_blocks=True, **kw)
    rd = I.run_grad(mk(), dense["root"] / "coast_dense", dense["base"], n_steps=N, budget_s=1e9,
                    cost="vol", cost_box="arctic")
    rs = I.run_grad(mk(dist=NPES, transport="coloured"), dense["root"] / "coast_coloured",
                    dense["base"], n_steps=N, budget_s=1e9, cost="vol", cost_box="arctic")
    assert rd["status"] == "ok" and rs["status"] == "ok"
    bad = check_grads("vol fct_bound_coast coloured vs dense", rs["grads"], rd["grads"])
    assert not [b for b in bad if b[1] == "cd"], bad
    assert not check_grads("vol fct_bound_coast coloured vs dense (P* bound)", rs["grads"],
                           rd["grads"], tol=TOL_ICE_PSTAR), bad


def test_sharded_forcing_gradient_equals_dense(dense):
    """Forcing controls (``Params.forcing_adj``, global ``[nod2D]`` fields in the replicated
    control pytree, cut to the lanes inside the chunk program): the coloured sharded terminal
    volume gradient maps equal the dense ones, the box sums of Tair (the thermodynamic path) to
    :data:`TOL_GRAD` and of the u wind (through the ice-atmosphere stress and the mEVP, the
    path of the volume's ln P*) to :data:`TOL_ICE_PSTAR`, the maps to 1e-2 relative L2.
    Measured (CPU, job 28208207): Tair 1.3e-8, map 2.3e-6; u wind 2.3e-3 / 2.8e-3, map 3.3e-3
    (rounding amplified like the volume's ln P*, 1.2e-3; a lane or halo error would show in
    Tair too). (The forward with zero anomalies is gated in test_cbar_ice_core2.py; the base
    forward is one-device.)"""
    ctl = ("pstar", "cd", "tair", "uair")
    root = dense["root"] / "forcing"
    m_d = I.build("2013-01-01", N, K, adjoint_cfg=D.parse_setting(SETTING), ice_mode="exact",
                  ice_adjoint=ICE_ADJ, controls=ctl, remat_blocks=True)
    m_s = I.build("2013-01-01", N, K, adjoint_cfg=D.parse_setting(SETTING), ice_mode="exact",
                  ice_adjoint=ICE_ADJ, controls=ctl, remat_blocks=True, dist=NPES,
                  transport="coloured")
    rd = I.run_grad(m_d, root / "dense", dense["base"], n_steps=N, budget_s=1e9, cost="vol",
                    cost_box="arctic")
    rs = I.run_grad(m_s, root / "coloured", dense["base"], n_steps=N, budget_s=1e9, cost="vol",
                    cost_box="arctic")
    assert rd["status"] == "ok" and rs["status"] == "ok"
    bad = []
    for c in ("tair", "uair"):
        for b in ("arctic", "nh"):
            k = f"{c}:{b}"
            print(f"[ice-sharded forcing] d vol / d {k}: {rs['grads'][k]:+.15e} vs "
                  f"{rd['grads'][k]:+.15e} rel {rel(rs['grads'][k], rd['grads'][k]):.2e}")
            tol = TOL_GRAD if c == "tair" else TOL_ICE_PSTAR
            if not (np.isfinite(rs["grads"][k]) and rs["grads"][k] != 0.0
                    and rel(rs["grads"][k], rd["grads"][k]) < tol):
                bad.append((k, rs["grads"][k], rd["grads"][k]))
        a, b_ = rs["_forcing_grad"][c], rd["_forcing_grad"][c]
        d2 = float(np.linalg.norm(a - b_) / np.linalg.norm(b_))
        print(f"[ice-sharded forcing] {c} map rel L2 {d2:.2e}, finite {np.isfinite(a).all()}")
        if not (np.isfinite(a).all() and d2 < 1e-2):
            bad.append((c, "map", d2))
    assert not bad, bad
