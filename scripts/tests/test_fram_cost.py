"""The Fram Strait sea-ice export cost (``tke_cbar`` cost ``FRAM``, ``adjoint_costs.ice_export_field``; the Fram plan
Task 3) on CORE2.

Light (the mesh only):
* the band 78.5-79.5N, 20W-12E: its nodes, its length (the node areas / 1 degree of latitude) and the southward unit
  vector in the model frame, checked against the geometry (the vector, rotated back to geographic Cartesian with the
  mesh's Euler rotation, points along the local geographic south);
* the sign (ice moving south exports, north imports) and the cost interval's step count (``cost_steps``).

Heavy (4 CPU devices, ``XLA_FLAGS=--xla_force_host_platform_device_count=4``; the year-window setting of
``test_core2_adj_sharded``, ice2 + se, two steps from the 2013-01-01 restart, the ice controls ln P*, ln C_d):
* J and dJ / d(ln P*, ln C_d, the TKE constants, ln k_gm) sharded == dense to round-off;
* dense: the central FD of J in ln C_d against the adjoint (the frozen-sweep ice adjoint at theta 0.97 is not exact,
  hence the 10 % bar of the year windows).
"""

from __future__ import annotations

import datetime
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import core2_adj as C  # noqa: E402
import ice_cbar as IC  # noqa: E402
import tke_cbar as T  # noqa: E402
import tke_diag as D  # noqa: E402

from fesom_jax import adjoint_costs as AC  # noqa: E402
from fesom_jax.config import RAD  # noqa: E402
from fesom_jax.jra55 import _rotation_matrix  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402

MESH = ROOT / "data" / "mesh_core2"
RESTART = C.RESTART_ARCHIVE / "fesom.2013.001.00000"
START = datetime.datetime(2013, 1, 1)
NDEV = 4
N_STEPS = 2
# the year-window setting, as test_core2_adj_sharded
Y4RAW7 = ("tke_pr_cut=1e-6+tke_pr_ratio=100+tke_pr_raw_cut=-1e-6+tke_ml_cut=1e-8+freeze_fct_limiter+tke_ml_ratio=100"
          "+tke_ml_raw_cut=-1e-7+tke_pr_lat_cut=70+tke_pr_ice_cut=0.15")
ICE_ARGS = ["grad", "--ice-damping", "frozen_sweep", "--ice-alpha-kappa", "1", "--ice-theta", "0.97",
            "--ice-sweep-guard", "10", "--ice-thin-mass", "100", "--ice-cuts", "fct_flux+strength_conc"]
CONTROLS = ("ln_ice_pstar", "ln_ice_cd", "ln_tke_c_k", "ln_tke_c_eps", "ln_tke_cd", "ln_tke_alpha", "ln_k_gm")
TOL_GRAD = 1e-6           # relative, as test_core2_adj_sharded (round-off through the reverse pass)
TOL_FD = 0.10             # the year windows' bar (tke_cbar.BAR_RTOL): the ice adjoint's theta 0.97 is approximate

pytestmark = pytest.mark.skipif(not MESH.is_dir(), reason="CORE2 mesh missing (compute node only)")
heavy = pytest.mark.skipif(len(jax.devices()) < NDEV or not RESTART.is_dir()
                           or not (C.CORE2_DIST / f"dist_{NDEV}").is_dir(),
                           reason=f"needs {NDEV} devices, the 2013 restart and dist_{NDEV}")


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(MESH)


def test_band_nodes_length_and_south_vector(mesh):
    wx, wy = (np.asarray(x) for x in AC.band_flux_weights(mesh, **AC.FRAM_BAND))
    w = np.hypot(wx, wy)
    band = w > 0
    area = np.asarray(mesh.area)[:, 0]
    np.testing.assert_allclose(w[band], area[band] / AC.FRAM_BAND["width"], rtol=1e-12)   # |s| = 1
    glon, glat = (np.asarray(mesh.geo_coord_nod2D)[:, i] for i in (0, 1))
    rlon, rlat = (np.asarray(mesh.coord_nod2D)[:, i] for i in (0, 1))
    lat_d, lon_d = glat / RAD, (glon / RAD + 180.0) % 360.0 - 180.0
    inside = (lat_d >= 78.5) & (lat_d <= 79.5) & (lon_d >= -20.0) & (lon_d <= 12.0)
    np.testing.assert_array_equal(band, inside)
    length = w.sum()                             # the band's area / its width = its ocean length [m]
    print(f"\n[fram] {band.sum()} band nodes, ocean length {length / 1e3:.0f} km "
          f"(79N, 20W-12E: {32 * 111.2 * np.cos(79 * np.pi / 180):.0f} km of latitude circle, land included)")
    assert band.sum() > 20 and 3e5 < length < 7.5e5
    # geometry: the model-frame vector (sx, sy) on the rotated (east, north) basis, mapped to geographic Cartesian
    # by the inverse Euler rotation, is the local geographic south
    sx, sy = wx[band] / w[band], wy[band] / w[band]
    rl, ro, gl, go = rlat[band], rlon[band], glat[band], glon[band]
    e_r = np.stack([-np.sin(ro), np.cos(ro), np.zeros_like(ro)])
    n_r = np.stack([-np.sin(rl) * np.cos(ro), -np.sin(rl) * np.sin(ro), np.cos(rl)])
    d_g = _rotation_matrix().reshape(3, 3).T @ (sx * e_r + sy * n_r)
    e_g = np.stack([-np.sin(go), np.cos(go), np.zeros_like(go)])
    n_g = np.stack([-np.sin(gl) * np.cos(go), -np.sin(gl) * np.sin(go), np.cos(gl)])
    south, east = -np.sum(d_g * n_g, axis=0), np.sum(d_g * e_g, axis=0)
    print(f"[fram] south component min {south.min():.8f}, |east| max {np.abs(east).max():.1e}")
    assert np.all(np.abs(south - 1.0) < 1e-6) and np.all(np.abs(east) < 1e-5)


def test_export_sign(mesh):
    wx, wy = (np.asarray(x) for x in AC.band_flux_weights(mesh, **AC.FRAM_BAND))
    w = np.hypot(wx, wy)
    sx, sy = np.where(w > 0, wx / np.where(w > 0, w, 1.0), 0.0), np.where(w > 0, wy / np.where(w > 0, w, 1.0), 0.0)
    m = np.full(mesh.nod2D, 2.0)
    for sign in (1.0, -1.0):                     # ice moving south (+) exports, north (-) imports
        st = types.SimpleNamespace(m_ice=jnp.asarray(m), u_ice=jnp.asarray(sign * 0.1 * sx),
                                   v_ice=jnp.asarray(sign * 0.1 * sy))
        f = np.asarray(AC.ice_export_field(st, mesh))
        np.testing.assert_allclose(f, sign * 0.2 * w, rtol=1e-12, atol=1e-300)


def test_cost_steps():
    def model(**sch):
        return types.SimpleNamespace(schedule=sch or None)
    assert T.cost_steps(model(), 100) == 100
    assert T.cost_steps(model(n_months=24, cost_from=17568, cost_to=None), 35088) == 17520   # 2013 of 2012-13
    assert T.cost_steps(model(cost_from=10, cost_to=30), 20) == 10
    with pytest.raises(ValueError, match="empty"):
        T.cost_steps(model(cost_from=30), 20)
    args = types.SimpleNamespace(start="2012-01-01", fast="ice2+se+dt1800+sp", forcing_controls="Tair,u_air",
                                 forcing_months=24, cost_from="2013-01-01", cost_to=None)
    assert T.schedule_kwargs(args) == dict(forcing_ctrl=("Tair", "u_air"), n_months=24, cost_from=17568)


@pytest.fixture(scope="module")
def models():
    kw = dict(adjoint_cfg=D.parse_setting(Y4RAW7), ice_mode="exact",
              ice_adjoint=C._ice_adjoint_args(C._parse(ICE_ARGS)), controls=("pstar", "cd"),
              accs=T.fram_accumulators, fast="ice2+se", restart=RESTART)
    return (IC.build(START, N_STEPS, N_STEPS, devices=1, **kw),
            IC.build(START, N_STEPS, N_STEPS, devices=NDEV, **kw))


def _forward(m, params=None):
    p = m.params0 if params is None else params
    return jax.block_until_ready(C.adjoint._jitted_forward(m.chunk_fn)(m.init_carry(), m.xs_fn()(0),
                                                                       jax.device_put(p)))


@heavy
def test_fram_sharded_equals_dense_and_fd(models):
    md, ms = models
    lossd = T.cost_fn("FRAM", N_STEPS, md.mesh)
    losss = T.cost_fn("FRAM", N_STEPS, ms.mesh, fold=ms.shard.fold_nodes)
    jd, js = float(lossd(_forward(md))), float(losss(_forward(ms)))
    print(f"\n[fram] J dense {jd:.10e} sharded {js:.10e} km3/yr")
    assert jd > 0 and abs(js - jd) <= 1e-10 * abs(jd)
    rd = C.run_grad(md, (lossd,), n_steps=N_STEPS, check_linear=False)[0]
    rs = C.run_grad(ms, (losss,), n_steps=N_STEPS, check_linear=False)[0]
    assert rd["status"] == rs["status"] == "ok"
    gd, gs = rd["_result"]["grad"], rs["_result"]["grad"]
    rows = {k: (float(gs[k]), float(gd[k]), abs(float(gs[k]) - float(gd[k])) / max(abs(float(gd[k])), 1e-300))
            for k in CONTROLS}
    print("[fram grad] " + "  ".join(f"{k} {a:+.8e} vs {b:+.8e} rel {r:.1e}" for k, (a, b, r) in rows.items()))
    assert abs(float(gd["ln_ice_cd"])) > 0 and abs(float(gd["ln_ice_pstar"])) > 0
    for k, (_, b, r) in rows.items():
        assert b == 0.0 or r <= TOL_GRAD, k
    # FD (dense) in ln C_d
    h = 1e-3

    def J(s):
        u = {k: np.array(v) for k, v in md.params0.items()}
        u["ln_ice_cd"] = np.asarray(s * h, np.float64)
        return float(lossd(_forward(md, C._strong_f64(u))))
    fd = (J(1) - J(-1)) / (2 * h)
    ad = float(gd["ln_ice_cd"])
    print(f"[fram] dJ/dln C_d: AD {ad:.6e} FD {fd:.6e} rel {abs(ad - fd) / abs(fd):.2e}")
    assert abs(ad - fd) <= TOL_FD * abs(fd)
