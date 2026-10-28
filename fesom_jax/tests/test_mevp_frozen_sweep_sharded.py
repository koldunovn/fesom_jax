"""The frozen-sweep ice backward (``IceConfig.adjoint_damping="frozen_sweep"``,
fesom_jax/ice_mevp_frozen.py) under ``shard_map``: sharded gradient == dense gradient.

The custom_vjp calls the halo exchanger inside its forward and its backward, and takes per-node
β_t as the maximum of α_t over the node's elements. Both are correct on owned nodes only if the
partition gives every owned node all of its elements (the match below shows the CORE2 one does),
and the backward's exchange transposes to the reverse (halo → owner) accumulation. A pad lane is a
node with no element at all; before the fix in ``raised_coefficients`` its β_t was −inf and the
backward put NaN on pad lanes and into d/d(C_d). This file checks all of it on the assembled CORE2
step: mEVP, exact ice adjoint, frozen_sweep at the defaults (κ0 = 1, θ = 0.97), 2 CPU devices,
the default all_gather halo transport.

Setup: PHC IC with the cold-start ice seed, one dense forced step (1958-01-01) so the ice has a
velocity and a stress, then the SECOND step differentiated, dense and sharded, from that same
state. Loss: area-weighted mean ice speed² over the ocean surface, reduced outside the shard_map
over OWNED nodes (nodes are uniquely owned, so the owned sum over devices is the dense sum).

Gates:
* the forward loss sharded vs dense (printed; its relative difference is the floor below);
* d/d(Params.ice_pstar), d/d(Params.ice_cd): finite, nonzero, sharded == dense;
* d/d(u_ice), d/d(m_ice) at entry: finite on every lane, 0 on pad lanes, and the owner + halo
  copies scatter-added (the all_gather transpose) equal the dense field.

CPU fake devices, compute node only (the grad compile is heavy). Skips at < 2 devices.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import integrate_sharded as ish
from fesom_jax import partit, shard_mesh, ssh
from fesom_jax import step as stepmod
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NDEV = len(jax.devices())
NPES = 2
DT = 1800.0
YEAR = 1958

# Tolerances. Measured on CPU at θ = 0.95 (job 27995348): forward loss rel 4.5e-16; d/dP* rel 2.6e-13, d/dC_d
# 5.1e-14; u_ice / m_ice cotangents max|Δ|/max|g| 1.3e-10 / 8.7e-12. With the halo exchange
# dropped from the backward's sweep (a planted defect, job 27995038) they are 3.0e-2 (P*) and
# 5.3e-2 / 1.1e-1 (fields).
TOL_PARAM = 1e-9         # relative, the two scalar control gradients
TOL_FIELD = 1e-8         # max |Δ| / max |g_dense|, the reconstructed entry-state cotangents


def _have_jra():
    from fesom_jax import jra55
    return Path(jra55.DEFAULT_JRA_DIR).is_dir()


pytestmark = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / f"dist_{NPES}").is_dir()
    or not (IC_DIR / "T_ic.npy").exists() or not _have_jra(),
    reason="CORE2 mesh, dist partitions, PHC IC or JRA55 forcing missing (compute node only)")


def ice_cfg():
    from fesom_jax.ice import IceConfig
    return IceConfig(whichEVP=1, adjoint_mode="exact", adjoint_damping="frozen_sweep")


def controls():
    """Params with the two ice controls traced at the configured values."""
    c = ice_cfg()
    return dataclasses.replace(Params.defaults(), ice_pstar=jnp.asarray(c.pstar, jnp.float64),
                               ice_cd=jnp.asarray(c.cd_oce_ice, jnp.float64))


def boundary_node_p(mesh, part):
    """The GLOBAL coastal mask, partitioned (a local recompute flags partition edges)."""
    from fesom_jax import ice_evp
    from fesom_jax.shard_mesh import _shard_along_axis, local_sizes
    bn = np.asarray(ice_evp.boundary_node_mask(mesh))
    _, Lmax = local_sizes(part)
    return _shard_along_axis(bn, part.myList_nod2D, Lmax["nod"], 0, False)


def speed2_p(st, w_p):
    """Area-weighted mean ice speed² over owned nodes of a [P, Lmax] State."""
    return jnp.sum(w_p * (st.u_ice ** 2 + st.v_ice ** 2)) / jnp.sum(w_p)


def speed2_dense(st, w):
    return jnp.sum(w * (st.u_ice ** 2 + st.v_ice ** 2)) / jnp.sum(w)


def reconstruct_2d(g_p, part, n_global):
    """Scatter-add a [P, Lmax] node cotangent over each node's owner + halo copies (the
    all_gather transpose) to the dense [n_global]."""
    G = np.asarray(g_p)
    out = np.zeros(n_global)
    for d in range(part.npes):
        n = int(part.myDim_nod2D[d] + part.eDim_nod2D[d])
        np.add.at(out, part.myList_nod2D[d][:n], G[d, :n])
    return out


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def build_setup():
    """Mesh, partition, the state after one dense forced step, step-2 forcing, dense grads."""
    from fesom_jax import ice, surface_forcing
    from fesom_jax.phc_ic import phc_initial_state
    mesh = load_mesh(CORE2_MESH)
    state = phc_initial_state(mesh, IC_DIR)
    sst0 = np.asarray(state.T[:, 0])
    state = ice.seed_ice(state, mesh, sst0)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=sst0)
    d1, d2 = surface_forcing.dates_for_steps(YEAR, DT, 2)
    sf1, sf2 = cf.step_forcing(*d1), cf.step_forcing(*d2)
    fs = cf.static
    zstress = jnp.zeros((mesh.elem2D, 2))
    cfg = ice_cfg()

    s1 = jax.jit(lambda s, p: stepmod.step(s, mesh, op, zstress, p, dt=DT, is_first_step=True,
                                           step_forcing=sf1, forcing_static=fs, ice_cfg=cfg)
                 )(state, controls())
    w = jnp.where(jnp.asarray(mesh.node_layer_mask[:, 0]), jnp.asarray(mesh.area[:, 0]), 0.0)

    def loss_de(params, u_ice, m_ice):
        sd = dataclasses.replace(s1, u_ice=u_ice, m_ice=m_ice)
        st = stepmod.step(sd, mesh, op, zstress, params, dt=DT, is_first_step=False,
                          step_forcing=sf2, forcing_static=fs, ice_cfg=cfg)
        return speed2_dense(st, w)

    L_de, (gp_de, gu_de, gm_de) = jax.jit(jax.value_and_grad(loss_de, argnums=(0, 1, 2)))(
        controls(), s1.u_ice, s1.m_ice)

    part = partit.read_partition(CORE2_DIST, NPES)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    return dict(mesh=mesh, op=op, s1=s1, sf2=sf2, fs=fs, cfg=cfg, part=part, sm=sm,
                L_de=float(L_de), gp_de=gp_de, gu_de=np.asarray(gu_de), gm_de=np.asarray(gm_de),
                ice_frac=float(np.mean(np.asarray(s1.a_ice) > 0)),
                umax=float(np.max(np.hypot(np.asarray(s1.u_ice), np.asarray(s1.v_ice)))))


@pytest.fixture(scope="module")
def setup():
    return build_setup()


def sharded_grads(fx):
    """Loss and d/d(controls, u_ice, m_ice) of the second step under shard_map."""
    mesh, part, sm = fx["mesh"], fx["part"], fx["sm"]
    s1_p = shard_mesh.partition_state(fx["s1"], part)
    sf_p = shard_mesh.partition_step_forcing(fx["sf2"], part)
    fs_p = shard_mesh.partition_forcing_static(fx["fs"], part)
    sop = ssh.partition_ssh_operator(fx["op"], part)
    stress_p = jnp.asarray(np.zeros((part.npes, sm.Lmax["elem"], 2)))
    bn_p = boundary_node_p(mesh, part)
    owned = jnp.asarray(sm.owned_mask["nod"])
    wet0 = jnp.asarray(sm.fields["node_layer_mask"])[:, :, 0]
    w_p = jnp.where(owned & wet0, jnp.asarray(sm.fields["area"])[:, :, 0], 0.0)

    def loss_sh(params, u_ice, m_ice):
        sp = dataclasses.replace(s1_p, u_ice=u_ice, m_ice=m_ice)
        st = ish.run_step_sharded(sm, sp, sop, stress_p, dt=DT, is_first_step=False,
                                  npes=NPES, params=params, step_forcing=sf_p,
                                  forcing_static=fs_p, boundary_node_p=bn_p,
                                  ice_cfg=fx["cfg"])
        return speed2_p(st, w_p)

    L, (gp, gu, gm) = jax.value_and_grad(loss_sh, argnums=(0, 1, 2))(
        controls(), s1_p.u_ice, s1_p.m_ice)
    return float(L), gp, np.asarray(gu), np.asarray(gm)


def compare(fx, L_sh, gp_sh, gu_sh, gm_sh):
    """The numbers the gate reads: loss, control and field-cotangent differences."""
    part, sm, n = fx["part"], fx["sm"], fx["mesh"].nod2D
    valid = np.asarray(sm.valid_mask["nod"])
    out = dict(L_sh=L_sh, L_de=fx["L_de"], L_rel=rel(L_sh, fx["L_de"]))
    for k in ("ice_pstar", "ice_cd"):
        a, b = float(getattr(gp_sh, k)), float(getattr(fx["gp_de"], k))
        out[k] = (a, b, rel(a, b))
    for k, g_sh, g_de in (("u_ice", gu_sh, fx["gu_de"]), ("m_ice", gm_sh, fx["gm_de"])):
        recon = reconstruct_2d(g_sh, part, n)
        out[k] = dict(finite=bool(np.isfinite(g_sh).all()),
                      pad_max=float(np.max(np.abs(g_sh[~valid]), initial=0.0)),
                      live_max=float(np.max(np.abs(g_sh[valid]))),
                      dmax_rel=float(np.max(np.abs(recon - g_de)) / np.max(np.abs(g_de))))
    return out


def report(tag, fx, c):
    print(f"\n[ice-frozen-sweep-sharded{tag}] npes={NPES}  ice nodes {fx['ice_frac']:.3f}  "
          f"max|u_ice| entry {fx['umax']:.3e} m/s")
    print(f"   loss  sharded={c['L_sh']:.15e}  dense={c['L_de']:.15e}  rel={c['L_rel']:.2e}")
    for k in ("ice_pstar", "ice_cd"):
        a, b, r = c[k]
        print(f"   d/d({k:9s}) sharded={a:+.12e}  dense={b:+.12e}  rel={r:.2e}")
    for k in ("u_ice", "m_ice"):
        d = c[k]
        print(f"   d/d({k} entry): finite={d['finite']}  pad max|g|={d['pad_max']:.2e}  "
              f"live max|g|={d['live_max']:.3e}  recon vs dense max|Δ|/max|g|={d['dmax_rel']:.2e}")


@pytest.mark.skipif(NDEV < NPES, reason=f"needs {NPES} devices")
def test_frozen_sweep_sharded_gradient_equals_dense(setup, capsys):
    c = compare(setup, *sharded_grads(setup))
    with capsys.disabled():
        report("", setup, c)

    assert c["L_rel"] < 1e-12, "the sharded forward differs from the dense one"
    for k in ("ice_pstar", "ice_cd"):
        a, b, r = c[k]
        assert np.isfinite(a) and a != 0.0, f"d/d({k}) sharded is {a}"
        assert r < TOL_PARAM, f"d/d({k}): sharded {a:.6e} vs dense {b:.6e} (rel {r:.2e})"
    for k in ("u_ice", "m_ice"):
        d = c[k]
        assert d["finite"], f"non-finite {k} cotangent lanes"
        assert d["pad_max"] == 0.0, f"pad lanes carry a {k} cotangent"
        assert d["live_max"] > 0.0, f"{k} cotangent identically zero"
        assert d["dmax_rel"] < TOL_FIELD, \
            f"{k} cotangent: reconstructed vs dense {d['dmax_rel']:.2e}"


@pytest.mark.skipif(NDEV < NPES, reason=f"needs {NPES} devices")
def test_frozen_sweep_sharded_sweep_guard(setup, capsys):
    """``IceConfig.adjoint_sweep_guard`` under shard_map (the sweep norms summed / maxed over the
    devices): G = 1e6 gives the dense unguarded gradient within the tolerances above; G = 1e-12
    trips on every device, so nothing reaches the controls or the entering ice velocity."""
    big = compare(setup, *sharded_grads({**setup, "cfg": setup["cfg"]._replace(
        adjoint_sweep_guard=1e6)}))
    with capsys.disabled():
        report(" guard 1e6", setup, big)
    for k in ("ice_pstar", "ice_cd"):
        a, b, r = big[k]
        assert r < TOL_PARAM, f"d/d({k}) with the guard: {a:.6e} vs dense {b:.6e} (rel {r:.2e})"
    for k in ("u_ice", "m_ice"):
        assert big[k]["dmax_rel"] < TOL_FIELD, f"{k} with the guard: {big[k]['dmax_rel']:.2e}"
    _, gp, gu, _ = sharded_grads({**setup, "cfg": setup["cfg"]._replace(adjoint_sweep_guard=1e-12)})
    assert float(gp.ice_pstar) == 0.0 and float(gp.ice_cd) == 0.0, (gp.ice_pstar, gp.ice_cd)
    assert np.all(gu == 0.0), np.max(np.abs(gu))
