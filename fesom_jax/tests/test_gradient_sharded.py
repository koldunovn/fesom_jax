"""S.8 gate: AD through the sharded step's collectives (the gradient gate).

The forward of the whole sharded model lowers and is N-vs-1 correct on owned (S.7). S.8
gates the **BACKWARD**: ``jax.grad`` of a scalar loss of a sharded run must equal the
single-device gradient (the ``v1.0`` baseline), and the masked/halo/pad lanes must carry a
FINITE value + 0 cotangent across the device axis (the Phase 3/5/6 masked-NaN discipline,
now on the device-pad axis). The collectives' transposes are what makes this work:

* an ``all_gather`` (the halo broadcast) transposes to a **reduce-scatter / scatter-add** —
  a halo lane's cotangent flows additively back to its owner's interior lane (the reverse
  exchange). This is the T0-field gate: the dense ``d(loss)/d(node n)`` is reconstructed by
  **scatter-adding** the sharded cotangent over each node's owner-interior + halo copies
  (``Bᵀ``, the transpose of the partition's owner→halo broadcast).
* a ``psum`` (the CG dots, the ``_area_mean`` reductions, and a CLOSED-OVER replicated
  param's cotangent) transposes to a ``psum`` — so a sharded ``d(loss)/d(k_ver)`` (a scalar,
  replicated across devices) equals the single-device scalar to the reassociation floor.
* the CG ``custom_linear_solve`` ``transpose_solve`` already runs sharded (S.6); ``a_ver``'s
  gradient routes through it (a_ver → Av → impl_vert_visc → du → ssh_rhs → CG → d_eta → uv).
* the ice EVP 120-subcycle ``jax.checkpoint``'d ``lax.scan`` with an in-scan ``all_gather``:
  its forward lowers (S.7p3); the backward RECOMPUTE re-runs the ``all_gather`` whose
  transpose is the reverse exchange — gated by the forced (ice-ON) grad below.

Loss = mean SST, reduced **outside** the ``shard_map`` over the unfolded ``[P, Lmax]``
state, masked to OWNED ∧ wet-surface nodes (nodes are uniquely owned ⇒ the owned-sum over
devices == the dense global sum; masking halo+pad keeps the loss independent of them ⇒ 0
cotangent there — the masked-NaN rule). Gated field-appropriately (Decision 4): the CLEAN
param gradients (k_ver/a_ver — averaged scalars) tight; the FCT-coupled T0-field
reconstruction to the climate-close upwind-flip floor (the same non-determinism the forward
T/S carry — the gradient inherits it through the tracer advection).

Runs on CPU fake-devices (the grad compile is heavier than the forward — a focused sbatch
on the compute node, NEVER login). SKIPS at <2 devices.
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
from fesom_jax.config import A_VER, K_VER
from fesom_jax.gm import GMConfig
from fesom_jax.kpp import KppConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.params import Params
from fesom_jax.state import State

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NDEV = len(jax.devices())
DT = 1800.0
YEAR = 1958

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_2").is_dir(),
    reason="CORE2 dense mesh or dist partitions missing")


def _have_jra():
    from fesom_jax import jra55
    return Path(jra55.DEFAULT_JRA_DIR).is_dir()


have_forcing = pytest.mark.skipif(
    not (IC_DIR / "T_ic.npy").exists() or not _have_jra(),
    reason="CORE2 PHC IC or JRA55 forcing missing (compute node only)")


# --------------------------------------------------------------------------
# Setup helpers (host-side; mirror test_step_sharded.py)
# --------------------------------------------------------------------------
def _perturbed_state(mesh):
    """A non-trivial, deterministic State (smooth lat perturbation of rest) so the step
    does real work ⇒ the gradient is nonzero and well-conditioned."""
    st = State.rest(mesh)
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]
    bump = 0.5 * np.cos(2 * lat)[:, None]
    T = np.asarray(st.T) + np.where(np.asarray(mesh.node_layer_mask), bump, 0.0)
    return dataclasses.replace(st, T=jnp.asarray(T))


def _stress_p(mesh, part, Le):
    """Zero element wind stress, partitioned to [P, Lmax_elem, 2]."""
    return jnp.asarray(np.zeros((part.npes, Le, 2)))


def _ocean_setup(npes):
    """Partition the OCEAN (no-forcing) model to ``npes`` devices (host build, seconds)."""
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = _perturbed_state(mesh)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = _stress_p(mesh, part, sm.Lmax["elem"])
    return dict(mesh=mesh, op=op, state=state, part=part, sm=sm,
                state_p=state_p, sop=sop, stress_p=stress_p)


# --------------------------------------------------------------------------
# Loss (owned-masked global mean SST; the sharded analog of test_gradient._mean_sst)
# --------------------------------------------------------------------------
def _surf_owned_mask(sm):
    """``[P, Lmax]`` bool: OWNED ∧ wet-surface nodes — the mean-SST reduction support.
    Nodes are uniquely owned, so summing over it across devices == the dense global sum;
    halo+pad are excluded ⇒ the loss cannot depend on them ⇒ 0 cotangent there."""
    owned = jnp.asarray(sm.owned_mask["nod"])
    wet0 = jnp.asarray(sm.fields["node_layer_mask"])[:, :, 0]
    return owned & wet0


def _mean_sst_p(state_p, surf):
    """Mean SST over owned wet-surface lanes of a ``[P, Lmax, nl]`` partitioned State
    (reduced OUTSIDE the shard_map — a cross-device all-reduce; its transpose is a
    broadcast, so the backward into the shard_map output is exact)."""
    T0 = state_p.T[:, :, 0]
    return jnp.sum(jnp.where(surf, T0, 0.0)) / jnp.sum(surf)


def _mean_sst_dense(state, mesh):
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
    return jnp.sum(jnp.where(wet0, state.T[:, 0], 0.0)) / jnp.sum(wet0)


def _reconstruct_global(g_p, part, n_global, nl):
    """``Bᵀ(g_p)``: scatter-add the sharded ``[P, Lmax, nl]`` cotangent over each global
    node's owner-interior + halo copies (the ``all_gather`` transpose) → dense
    ``[n_global, nl]``. Valid lanes are ``myList[d][:myDim+eDim]`` (pad/dry lanes carry 0,
    asserted separately). The result equals the single-device ``d(loss)/d(T0)``."""
    G = np.asarray(g_p)
    recon = np.zeros((n_global, nl))
    for d in range(part.npes):
        n = int(part.myDim_nod2D[d] + part.eDim_nod2D[d])
        np.add.at(recon, part.myList_nod2D[d][:n], G[d, :n])
    return recon


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# --------------------------------------------------------------------------
# 1. OCEAN 1-step: param grad == dense (tight) + T0 field grad reconstruction + masked-NaN
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2])
def test_grad_ocean_param_matches_dense(npes, capsys):
    """The headline S.8 gate: ``d(mean SST)/d(k_ver)`` and ``d(.)/d(a_ver)`` of the sharded
    1-step OCEAN run == the single-device gradient to a TIGHT tol. Both params are scalars
    closed over the ``shard_map`` (replicated) ⇒ their cotangents are ``psum``'d across
    devices (the ``psum``-transpose-is-``psum`` proof, Decision 6). ``a_ver`` routes the
    gradient through the CG ``custom_linear_solve`` ``transpose_solve`` SHARDED (the S.6
    AD completion). GM is off ⇒ ``d/d(k_gm) == 0`` exactly (the unused-leaf check)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    fx = _ocean_setup(npes)
    mesh, op, state = fx["mesh"], fx["op"], fx["state"]
    sm, state_p, sop, stress_p = fx["sm"], fx["state_p"], fx["sop"], fx["stress_p"]
    surf = _surf_owned_mask(sm)
    p0 = Params.defaults()

    def loss_sh(params):
        st = ish.run_step_sharded(sm, state_p, sop, stress_p, dt=DT, is_first_step=True,
                                  npes=npes, params=params)
        return _mean_sst_p(st, surf)

    def loss_de(params):
        st = stepmod.step(state, mesh, op, jnp.zeros((mesh.elem2D, 2)), params,
                          dt=DT, is_first_step=True)
        return _mean_sst_dense(st, mesh)

    g_sh = jax.grad(loss_sh)(p0)
    g_de = jax.grad(loss_de)(p0)

    with capsys.disabled():
        print(f"\n[grad-ocean-param] npes={npes}")
        for name in ("k_ver", "a_ver"):
            a, b = float(getattr(g_sh, name)), float(getattr(g_de, name))
            print(f"   d/d({name:5s})  sharded={a:+.8e}  dense={b:+.8e}  rel={_rel(a, b):.2e}")
        print(f"   d/d(k_gm)    sharded={float(g_sh.k_gm):+.3e}  (GM off ⇒ expect 0)")

    # Field-appropriate tol (the gradient analog of the forward Decision-4 gate): k_ver
    # routes through the CLEAN tracer vertical DIFFUSION (additive into Kv) ⇒ machine-floor
    # match; a_ver routes through the FCT tracer ADVECTION (uv → upwind-flip floor) AND its
    # gradient is tiny (~3e-8, near the loss round-off) ⇒ the climate-close reassociation
    # floor (~4e-4, within the dense path's own FD accuracy — test_grad_flows_through_cg).
    _PTOL = {"k_ver": 1e-6, "a_ver": 5e-3}
    for name in ("k_ver", "a_ver"):
        a, b = float(getattr(g_sh, name)), float(getattr(g_de, name))
        assert np.isfinite(a) and a != 0.0, f"d/d({name}) sharded = {a}"
        assert _rel(a, b) < _PTOL[name], (
            f"d/d({name}): sharded {a:.6e} vs dense {b:.6e} rel {_rel(a,b):.2e} > {_PTOL[name]:.0e}")
    assert float(g_sh.k_gm) == 0.0, "GM off ⇒ d/d(k_gm) must be exactly 0"


@avail
@pytest.mark.parametrize("npes", [2])
def test_grad_ocean_ic_field_and_masked_nan(npes, capsys):
    """``d(mean SST)/d(T₀)`` (the IC temperature field, vector-valued) of the sharded 1-step
    OCEAN run: (a) **masked-NaN across devices** — finite EVERYWHERE (incl. halo/pad/
    below-bottom lanes), exactly 0 cotangent on the dry/pad lanes (the loss cannot depend on
    them), nonzero on owned-wet; (b) the **scatter-add reconstruction** ``Bᵀ(g_p)`` ==
    single-device ``d(loss)/d(T₀)`` (the ``all_gather`` transpose carries halo cotangents
    back to owners additively). Field-appropriate: the T0→FCT-advection→SST path inherits
    the upwind-flip floor, so the per-node reconstruction matches to the climate-close
    budget (the bulk tight, a few near-flip nodes elevated — printed)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    fx = _ocean_setup(npes)
    mesh, op, state = fx["mesh"], fx["op"], fx["state"]
    sm, state_p, sop, stress_p, part = (fx["sm"], fx["state_p"], fx["sop"],
                                        fx["stress_p"], fx["part"])
    surf = _surf_owned_mask(sm)

    def loss_sh(T0):
        sp = dataclasses.replace(state_p, T=T0)            # keep T_old (the IC AB2 history)
        st = ish.run_step_sharded(sm, sp, sop, stress_p, dt=DT, is_first_step=True,
                                  npes=npes)
        return _mean_sst_p(st, surf)

    def loss_de(T0):
        sd = dataclasses.replace(state, T=T0)
        st = stepmod.step(sd, mesh, op, jnp.zeros((mesh.elem2D, 2)), dt=DT,
                          is_first_step=True)
        return _mean_sst_dense(st, mesh)

    g_p = np.asarray(jax.grad(loss_sh)(state_p.T))         # [P, Lmax, nl]
    g_de = np.asarray(jax.grad(loss_de)(jnp.asarray(state.T)))   # [nod2D, nl]

    # (a) masked-NaN across the device axis
    valid = np.asarray(sm.valid_mask["nod"])               # [P, Lmax] interior+halo
    nlm = np.asarray(sm.fields["node_layer_mask"])         # [P, Lmax, nl] wet
    live = valid[:, :, None] & nlm                         # owned/halo wet lanes
    assert np.isfinite(g_p).all(), f"{int((~np.isfinite(g_p)).sum())} non-finite grad lanes"
    assert np.max(np.abs(g_p[~live])) == 0.0, "dry/pad lanes carry spurious cotangent"
    assert np.max(np.abs(g_p[live])) > 0.0, "IC cotangent identically zero on live lanes"

    # (b) reconstruction Bᵀ(g_p) == dense d(loss)/d(T0)
    recon = _reconstruct_global(g_p, part, mesh.nod2D, mesh.nl)
    mlay = np.asarray(mesh.node_layer_mask)
    diff = np.abs(recon - g_de)[mlay]                      # compare on wet lanes
    denom = np.maximum(np.abs(g_de)[mlay], 1e-30)
    rel = diff / denom
    with capsys.disabled():
        print(f"\n[grad-ocean-ic] npes={npes}  reconstruct Bᵀ(g_p) vs dense d(loss)/d(T0):")
        print(f"   max|Δ|={diff.max():.3e}  median|Δ|={np.median(diff):.3e}  "
              f"p99|Δ|={np.percentile(diff, 99):.3e}")
        print(f"   rel: median={np.median(rel):.3e}  p99={np.percentile(rel, 99):.3e}  "
              f"max={rel.max():.3e}")
    # the bulk is clean (median tight); a few near-flip nodes ride the FCT floor (max bounded)
    assert np.median(diff) < 1e-9, f"IC grad reconstruction median|Δ|={np.median(diff):.3e}"
    assert diff.max() < 1e-3, f"IC grad reconstruction max|Δ|={diff.max():.3e} (FCT floor)"


@avail
@pytest.mark.parametrize("npes", [2])
def test_grad_ocean_fd_self_consistent(npes, capsys):
    """FD spot-check on the well-conditioned ``k_ver`` seam (it enters ``Kv = mix + k_ver``
    additively ⇒ smooth, the existing single-device gate's choice): central FD of the
    SHARDED loss == its AD gradient. This validates the sharded forward+backward are mutually
    consistent INDEPENDENT of the dense baseline (a wrong backward would miss FD even if it
    somehow matched a wrong dense). Evaluated at a background ``k_ver`` where the FD signal
    clears the loss's ~eps·SST round-off floor."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    fx = _ocean_setup(npes)
    sm, state_p, sop, stress_p = fx["sm"], fx["state_p"], fx["sop"], fx["stress_p"]
    surf = _surf_owned_mask(sm)

    def loss_kver(kv):
        p = dataclasses.replace(Params.defaults(), k_ver=kv)
        st = ish.run_step_sharded(sm, state_p, sop, stress_p, dt=DT, is_first_step=True,
                                  npes=npes, params=p)
        return _mean_sst_p(st, surf)

    k0 = jnp.asarray(1e-4, jnp.float64)
    g_ad = float(jax.grad(loss_kver)(k0))
    rows = []
    for h in (1e-3, 1e-4, 1e-5):
        kp, km = k0 * (1.0 + h), k0 * (1.0 - h)
        g_fd = float((loss_kver(kp) - loss_kver(km)) / (kp - km))
        rows.append((h, g_fd, _rel(g_ad, g_fd)))
    with capsys.disabled():
        print(f"\n[grad-ocean-fd] npes={npes}  d(mean SST)/d(k_ver) AD={g_ad:+.8e}")
        for h, g_fd, rel in rows:
            print(f"   h={h:.0e}  FD={g_fd:+.8e}  rel|AD-FD|={rel:.2e}")
    assert np.isfinite(g_ad) and g_ad != 0.0
    plateau = min(r for _, _, r in rows)
    assert plateau < 1e-4, f"sharded FD plateau rel err {plateau:.2e} ≥ 1e-4"


# --------------------------------------------------------------------------
# 2. Multi-step scan backward (the checkpointed-scan + in-scan-collective transpose)
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2])
def test_grad_multistep_scan_backward(npes, capsys):
    """``jax.grad`` of a 2-step :func:`run_steps_sharded` (step-1 eager + a ``jax.checkpoint``'d
    ``lax.scan`` of the rest) — the BACKWARD through the checkpointed scan + the in-scan
    halo/CG collectives LOWERS and runs FINITE and nonzero. A free-running 2-step trajectory
    decorrelates chaotically from single-device (Decision 4 — the step-1 FCT flip floor feeds
    step 2), so this gates the scan-backward MECHANISM (the collective-in-scan transpose
    works), not a tight dense match (that is the teacher-forced / 1-step regime above)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    fx = _ocean_setup(npes)
    sm, state_p, sop, stress_p = fx["sm"], fx["state_p"], fx["sop"], fx["stress_p"]
    surf = _surf_owned_mask(sm)

    def loss(kv):
        p = dataclasses.replace(Params.defaults(), k_ver=kv)
        st = ish.run_steps_sharded(sm, state_p, sop, stress_p, 2, dt=DT, npes=npes, params=p)
        return _mean_sst_p(st, surf)

    g = float(jax.grad(loss)(jnp.asarray(K_VER, jnp.float64)))
    with capsys.disabled():
        print(f"\n[grad-multistep] npes={npes}  d(2-step mean SST)/d(k_ver) = {g:+.6e}")
    assert np.isfinite(g) and g != 0.0, f"2-step scan-backward grad = {g}"


# --------------------------------------------------------------------------
# 3. Forced (KPP + GM/Redi + prognostic ice): the full assembled backward
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def core2_forced():
    from fesom_jax import surface_forcing
    from fesom_jax.phc_ic import core2_initial_state
    mesh = load_mesh(CORE2_MESH)
    state = core2_initial_state(mesh, IC_DIR)
    sst0 = np.asarray(state.T[:, 0])
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=sst0)
    sf = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
    return dict(mesh=mesh, state=state, op=op, sf=sf, fs=cf.static)


def _global_boundary_node_p(mesh, part):
    from fesom_jax import ice_evp
    from fesom_jax.shard_mesh import _shard_along_axis, local_sizes
    bn = np.asarray(ice_evp.boundary_node_mask(mesh))
    _, Lmax = local_sizes(part)
    return _shard_along_axis(bn, part.myList_nod2D, Lmax["nod"], 0, False)


@avail
@have_forcing
@pytest.mark.parametrize("npes", [2])
def test_grad_assembled_forced_backward(npes, core2_forced, capsys):
    """The full assembled CORE2 step (KPP + GM/Redi + prognostic ice + bulk/SSS forcing)
    differentiated under ``shard_map`` — the heaviest backward in the port. ONE backward
    sweep (``argnums=(params, T0)``) gates ALL the AD-critical forced-path collectives'
    transposes firing together:

    * the ice EVP 120-subcycle ``jax.checkpoint``'d ``lax.scan`` whose backward RECOMPUTE
      re-runs the in-scan ``u_ice/v_ice`` ``all_gather`` (transpose = the reverse exchange)
      — the hardest AD placement in the port. **It is exercised by ``d/d(T₀)``, NOT
      ``d/d(params)``**: the ocean mixing/eddy params don't enter the ice dynamics, but T₀ →
      SST → ice thermo (a_ice) + ocean-ice stress → the EVP scan → ``u_ice`` →
      ``stress_surf`` → ocean ``uv`` → loss, so the cotangent flows back THROUGH the EVP
      subcycle scan;
    * the CG ``custom_linear_solve`` ``transpose_solve`` sharded (the SSH d_eta cotangent);
    * the KPP smoother / GM / ``_area_mean`` reductions ``psum`` transposes.

    Gate: (1) ``d/d(T₀)`` FINITE everywhere across the device axis (the EVP-scan-backward +
    every masked divide lower and run finite — the masked-NaN rule on the device-pad axis),
    0 cotangent on dry/pad lanes; (2) every param gradient (``k_gm`` now ACTIVE — the 2nd
    ML-hook — and ``k_ver`` via KPP→Kv) finite + nonzero + == the single-device gradient
    field-appropriately (the FCT/ice coupling loosens the floor vs pure OCEAN)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    from fesom_jax import ice
    from fesom_jax.ice import IceConfig
    fx = core2_forced
    mesh, op, sf, fs = fx["mesh"], fx["op"], fx["sf"], fx["fs"]
    sst0 = np.asarray(fx["state"].T[:, 0])
    state = ice.seed_ice(fx["state"], mesh, sst0)
    cfg = dict(kpp_cfg=KppConfig(), gm_cfg=GMConfig(), ice_cfg=IceConfig())

    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sf_p = shard_mesh.partition_step_forcing(sf, part)
    fs_p = shard_mesh.partition_forcing_static(fs, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = _stress_p(mesh, part, sm.Lmax["elem"])
    bn_p = _global_boundary_node_p(mesh, part)
    surf = _surf_owned_mask(sm)

    def loss_sh(params, T0):
        sp = dataclasses.replace(state_p, T=T0)
        st = ish.run_step_sharded(sm, sp, sop, stress_p, dt=DT, is_first_step=True,
                                  npes=npes, params=params, step_forcing=sf_p,
                                  forcing_static=fs_p, boundary_node_p=bn_p, **cfg)
        return _mean_sst_p(st, surf)

    def loss_de(params, T0):
        sd = dataclasses.replace(state, T=T0)
        st = stepmod.step(sd, mesh, op, jnp.zeros((mesh.elem2D, 2)), params,
                          dt=DT, is_first_step=True, step_forcing=sf, forcing_static=fs,
                          **cfg)
        return _mean_sst_dense(st, mesh)

    g_sh, gT_sh = jax.grad(loss_sh, argnums=(0, 1))(Params.defaults(), state_p.T)
    g_de, gT_de = jax.grad(loss_de, argnums=(0, 1))(Params.defaults(),
                                                    jnp.asarray(state.T))

    # (1) d/d(T0) — the ice-EVP-scan-backward + masked-NaN-across-devices gate
    gT = np.asarray(gT_sh)
    valid = np.asarray(sm.valid_mask["nod"])
    nlm = np.asarray(sm.fields["node_layer_mask"])
    live = valid[:, :, None] & nlm
    recon = _reconstruct_global(gT, part, mesh.nod2D, mesh.nl)
    mlay = np.asarray(mesh.node_layer_mask)
    dT = np.abs(recon - np.asarray(gT_de))[mlay]

    with capsys.disabled():
        print(f"\n[grad-assembled] npes={npes}  (KPP+GM+ice forced backward)")
        print(f"   d/d(T0): finite={bool(np.isfinite(gT).all())}  "
              f"dry/pad max|g|={np.max(np.abs(gT[~live])):.3e}  "
              f"reconstruct vs dense: median|Δ|={np.median(dT):.3e} max|Δ|={dT.max():.3e}")
        for name in ("k_ver", "a_ver", "k_gm", "redi_kmax"):
            a, b = float(getattr(g_sh, name)), float(getattr(g_de, name))
            print(f"   d/d({name:9s})  sharded={a:+.8e}  dense={b:+.8e}  rel={_rel(a, b):.2e}")

    assert np.isfinite(gT).all(), (f"{int((~np.isfinite(gT)).sum())} non-finite T0-cotangent "
                                   "lanes (ice-EVP-scan backward NaN across devices?)")
    assert np.max(np.abs(gT[~live])) == 0.0, "dry/pad lanes carry spurious T0 cotangent"
    assert np.max(np.abs(gT[live])) > 0.0, "T0 cotangent identically zero on live lanes"

    # (2) param gradients. All FINITE (the backward lowered through KPP/GM/ice). With KPP
    # active the PP backgrounds k_ver/a_ver are UNUSED (KPP replaces PP) ⇒ their grad is
    # exactly 0 == dense (NOT a dead backward — the eddy seam is GM's k_gm/redi_kmax). The GM
    # params carry the eddy-seam gradient: redi_kmax → CLEAN Redi diffusion matches dense
    # tightly (the GM-backward correctness proof, alongside the T0 reconstruction above);
    # k_gm → GM bolus → FCT tracer ADVECTION inherits the GM+PHC-IC upwind-flip floor on a
    # TINY (~3e-9) gradient (large rel / small abs — climate-close, like the OCEAN a_ver but
    # larger: the bolus perturbs the advecting velocity directly and PHC fronts are sharp).
    for name in ("k_ver", "a_ver", "k_gm", "redi_kmax"):
        assert np.isfinite(float(getattr(g_sh, name))), f"d/d({name}) sharded non-finite"
    for name in ("k_ver", "a_ver"):              # PP backgrounds unused under KPP ⇒ 0 == dense
        a, b = float(getattr(g_sh, name)), float(getattr(g_de, name))
        assert abs(a - b) < 1e-14, f"d/d({name}) (PP-unused under KPP): {a:.3e} vs dense {b:.3e}"
    assert float(g_sh.k_gm) != 0.0 and float(g_sh.redi_kmax) != 0.0, "GM eddy-seam backward did not carry"
    rk_s, rk_d = float(g_sh.redi_kmax), float(g_de.redi_kmax)
    assert _rel(rk_s, rk_d) < 1e-2, f"d/d(redi_kmax): {rk_s:.6e} vs dense {rk_d:.6e} rel {_rel(rk_s,rk_d):.2e}"
    kg_s, kg_d = float(g_sh.k_gm), float(g_de.k_gm)   # tiny FCT-floor gradient: same sign + abs floor
    assert np.sign(kg_s) == np.sign(kg_d) and abs(kg_s - kg_d) < 2e-9, \
        f"d/d(k_gm): sharded {kg_s:.3e} vs dense {kg_d:.3e} (climate flip floor)"
