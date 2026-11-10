"""The sharded adjoint of ``ice_cbar grad --dist`` is the adjoint of the sharded forward.

CORE2 (2013-01-01 restart, ``configs/core2_full.yaml``), 4-step window, 2-step chunks, on 2
devices with the coloured halo transport, and an EXACT adjoint everywhere: an exact 4-subcycle
mEVP adjoint (``evp_rheol_steps=4``, no damping; the frozen-sweep θ < 1 one is approximate by
design) and no frozen ocean derivatives. The farc settings freeze_mixing + freeze_gm_slope +
freeze_fct_limiter drop derivative terms on purpose, and C_d reaches the ocean through the
ice-ocean stress: with them the 4-step AD is 1.7 % off the FD along ln C_d and 9 % along the
initial ice velocity (job 28048169) while the sharded and dense AD agree to 3e-8 (28048737/8).
The sharded chunked AD of J_drift, its own forward and reverse sweep, against a central FD of
the sharded forward along

ln P* and ln C_d: the AD within 1e-3 of the FD at one of h = 1e-3, 1e-4, as
test_cbar_ice_core2.py's exact4 does for the dense model (measured 3.8e-7 for C_d, job 28048844).

The initial ice velocity, a fixed direction δ on the ice-covered nodes, through the
initial-state cotangent with every copy of a node summed (``Model.dense_cotangent``): against
the DENSE model's AD, within 1e-3 (measured 1.7e-5 relative L2 for the whole u_ice cotangent,
job 28048737). Its FD is printed, not gated: the plastic ice responds to a velocity change
non-smoothly and the central difference does not converge in h (-2.82e-4, -2.31e-4, -4.29e-4 at
h·max|δ| = 1e-4, 1e-5, 1e-6 m/s against the AD -2.10e-4). Its own process (CPU JIT memory). ⚠️ HEAVY: a CPU compute node with
``XLA_FLAGS="--xla_force_host_platform_device_count=2
--xla_cpu_enable_concurrency_optimized_scheduler=false"`` (see test_cbar_ice_sharded.py), or two
GPUs.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import core2_adj as C  # noqa: E402
import ice_cbar as I  # noqa: E402

from fesom_jax import adjoint  # noqa: E402
from fesom_jax.adjoint_stab import AdjointConfig  # noqa: E402

RESTART = C.STATES_DIR / "fesom.2013.001.00000"
DIST = C.POOL_MESHES / "core2"
NPES = 2
N, K = 4, 2
_CPU_FLAG = "--xla_cpu_enable_concurrency_optimized_scheduler=false"

pytestmark = [
    pytest.mark.skipif(not ((ROOT / "data" / "mesh_core2").is_dir() and RESTART.is_dir()
                            and (DIST / f"dist_{NPES}").is_dir()),
                       reason="CORE2 mesh / 2013 States / dist_2 partition missing"),
    pytest.mark.skipif(len(jax.devices()) < NPES, reason=f"needs {NPES} devices"),
    pytest.mark.skipif(jax.devices()[0].platform == "cpu"
                       and _CPU_FLAG not in os.environ.get("XLA_FLAGS", ""),
                       reason=f"CPU fake devices mis-order collectives without XLA_FLAGS {_CPU_FLAG}"),
]


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def test_sharded_ad_matches_sharded_fd():
    m = I.build("2013-01-01", N, K, adjoint_cfg=AdjointConfig.exact(), ice_mode="exact",
                ice_settings="evp_rheol_steps=4", controls=("pstar", "cd"), remat_blocks=True,
                dist=NPES, transport="coloured")
    loss = I.drift_loss(N)
    row = C.run_grad(m, (loss,), n_steps=N, check_linear=False)[0]
    assert row["status"] == "ok"
    g = row["_result"]["grad"]
    cot = m.dense_cotangent(jax.device_get(row["_result"]["state_cot"]))[0]

    # a direction of the initial ice velocity on the ice-covered nodes, |δ| <= 0.1 m/s
    s0 = m.s0
    rng = np.random.default_rng(1)
    ice = np.asarray(s0.a_ice) > 0.15
    du = np.where(ice, rng.uniform(-0.1, 0.1, ice.shape), 0.0)
    dv = np.where(ice, rng.uniform(-0.1, 0.1, ice.shape), 0.0)
    ad_dir = float(np.sum(np.asarray(cot.u_ice) * du) + np.sum(np.asarray(cot.v_ice) * dv))

    def J(u=None, h=0.0):
        st = dataclasses.replace(s0, u_ice=np.asarray(s0.u_ice) + h * du,
                                 v_ice=np.asarray(s0.v_ice) + h * dv)
        fin = adjoint.chunked_forward(m.chunk_fn, m.params0 if u is None else u,
                                      init_state=m.init_carry(st), n_chunks=N // K,
                                      xs_fn=m.xs_fn())
        return float(loss(fin))

    bad = []
    for c in ("pstar", "cd"):
        ad = float(np.asarray(g[I.ICE_CONTROLS[c][0]]))
        fds = {h: (J(I.perturbed(m.params0, c, h, 1)) - J(I.perturbed(m.params0, c, h, -1)))
               / (2 * h) for h in (1e-3, 1e-4)}
        r = min(rel(ad, v) for v in fds.values())
        print(f"[ice-sharded fd] {c}: ad {ad:+.15e} fd {fds} rel {r:.2e}")
        if not (np.isfinite(ad) and ad != 0.0 and r < 1e-3):
            bad.append((c, ad, fds))
    fds = {h: (J(h=h) - J(h=-h)) / (2 * h) for h in (1e-3, 1e-4)}
    print(f"[ice-sharded fd] u_ice direction: sharded ad {ad_dir:+.15e} fd (not gated) {fds}")
    del m

    md = I.build("2013-01-01", N, K, adjoint_cfg=AdjointConfig.exact(), ice_mode="exact",
                 ice_settings="evp_rheol_steps=4", controls=("pstar", "cd"), remat_blocks=True)
    rd = C.run_grad(md, (loss,), n_steps=N, check_linear=False)[0]
    assert rd["status"] == "ok"
    cd_ = jax.device_get(rd["_result"]["state_cot"])[0]
    ad_d = float(np.sum(np.asarray(cd_.u_ice) * du) + np.sum(np.asarray(cd_.v_ice) * dv))
    print(f"[ice-sharded fd] u_ice direction: sharded ad {ad_dir:+.15e} dense ad {ad_d:+.15e} "
          f"rel {rel(ad_dir, ad_d):.2e}")
    if not (np.isfinite(ad_dir) and ad_dir != 0.0 and rel(ad_dir, ad_d) < 1e-3):
        bad.append(("u_ice direction vs dense", ad_dir, ad_d))
    assert not bad, bad
