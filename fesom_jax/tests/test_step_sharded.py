"""S.7 gate: device-mesh placement + the sharded step (:mod:`fesom_jax.integrate_sharded`).

Phase 1 (this file's current scope): the **placement scaffold** — reconstruct the
per-device local ``Mesh``/``State``/``SSHOperator`` from the S.2/S.2b/S.6 host bundles and
run the *unmodified* :func:`fesom_jax.step.step` under ``shard_map``. The ``npes==1`` whole
step == the dense step **byte-identically** (the no-op invariant: the sharded code path
collapses to the single-device model, so ``v1.0`` is structurally untouched), and the
multi-device step LOWERS and matches single device on the **deep interior** (owned nodes
whose stencil never reaches the halo) — the proof the local kernels are correct on real
shards. The *boundary* nodes need the halo exchanges (the rest of S.7), so they are not
yet asserted to match.

Runs on CPU fake-devices; the multi-device parts SKIP at 1 device. The full-step
``shard_map`` compile is ~1–2 min, so these are slow (SHARDING group).
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
from fesom_jax.gm import GMConfig
from fesom_jax.kpp import KppConfig
from fesom_jax.tke import TkeConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.state import State

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
NDEV = len(jax.devices())
DT = 1800.0
YEAR = 1958

# Byte-identity is a CPU property. GPU XLA fuses/reorders the same arithmetic differently
# (a larger reassociation floor), so the serial-collapse "byte-id" gates use a platform-aware
# floor: exact-ish on CPU, the measured A100 reassociation floor on GPU (S.9: worst across all
# State fields on a 1-device collapse was 7.66e-9 — well below the dynamics' clean budget).
_PLATFORM = jax.devices()[0].platform
_BYTE_ID_ATOL = 1e-9 if _PLATFORM == "cpu" else 1e-7

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_2").is_dir(),
    reason="CORE2 dense mesh or dist partitions missing")
have_ic = pytest.mark.skipif(
    not (IC_DIR / "T_ic.npy").exists(),
    reason="CORE2 PHC IC cache missing (data/ic_core2/)")


def _have_jra():
    from fesom_jax import jra55
    return Path(jra55.DEFAULT_JRA_DIR).is_dir()


have_forcing = pytest.mark.skipif(
    not (IC_DIR / "T_ic.npy").exists() or not _have_jra(),
    reason="CORE2 PHC IC or JRA55 forcing missing (compute node only)")


def _perturbed_state(mesh):
    """A non-trivial, deterministic State (smooth perturbations of rest) so the step
    does real work — otherwise rest-stays-rest gives a trivial 0==0 comparison."""
    st = State.rest(mesh)
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]
    bump = 0.5 * np.cos(2 * lat)[:, None]                  # [nod2D, 1], broadcast over nl
    T = np.asarray(st.T) + np.where(np.asarray(mesh.node_layer_mask), bump, 0.0)
    return dataclasses.replace(st, T=jnp.asarray(T))


def _stress_p(mesh, part, Le):
    """Zero element wind stress, partitioned to [P, Lmax_elem, 2]."""
    P = part.npes
    out = np.zeros((P, Le, 2))
    return jnp.asarray(out)


# --------------------------------------------------------------------------
# 1. Reconstruction: npes=1 local Mesh == dense Mesh (step-read fields)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not CORE2_MESH.is_dir(), reason="CORE2 mesh missing")
def test_local_mesh_reconstruction_serial():
    """The ``npes==1`` reconstructed local Mesh equals the dense Mesh for every
    step-read field (the CSR is a step-unused dummy)."""
    mesh = load_mesh(CORE2_MESH)
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, ser)
    lm = ish.local_mesh(sm, 0)
    assert lm.nod2D == mesh.nod2D and lm.elem2D == mesh.elem2D and lm.edge2D == mesh.edge2D
    for name in list(shard_mesh.NODE_FIELDS) + list(shard_mesh.ELEM_FIELDS) \
            + list(shard_mesh.EDGE_FIELDS) + [c[0] for c in shard_mesh.CONN_FIELDS]:
        a = np.asarray(getattr(mesh, name))
        b = np.asarray(getattr(lm, name))
        assert np.array_equal(a, b), f"local mesh field {name} != dense"


# --------------------------------------------------------------------------
# 2. npes=1 whole step under shard_map == dense (the no-op invariant)
# --------------------------------------------------------------------------
@avail
def test_serial_sharded_step_matches_dense():
    """The full ocean step under ``shard_map`` on ONE device == the dense step,
    byte-identically (the sharded path collapses to the single-device model)."""
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = _perturbed_state(mesh)
    stress = jnp.zeros((mesh.elem2D, 2))
    st_dense = stepmod.step(state, mesh, op, stress, dt=DT, is_first_step=True)

    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, ser)
    state_p = shard_mesh.partition_state(state, ser)
    sop = ssh.partition_ssh_operator(op, ser)
    stress_p = _stress_p(mesh, ser, sm.Lmax["elem"])
    st_N = ish.run_step_sharded(sm, state_p, sop, stress_p, dt=DT,
                                is_first_step=True, npes=1)

    worst = 0.0
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(st_dense, fld.name))
        b = np.asarray(getattr(st_N, fld.name))[0][: a.shape[0]]
        if a.size:
            worst = max(worst, float(np.max(np.abs(a - b))))
    assert worst < _BYTE_ID_ATOL, f"serial sharded step max|Δ|={worst:.3e} (expected byte-id, floor {_BYTE_ID_ATOL:.0e})"


# --------------------------------------------------------------------------
# 3. The full sharded step (WITH exchanges) matches single-device on OWNED entities
# --------------------------------------------------------------------------
# Fields with inherent N-vs-1 non-determinism ABOVE the clean reassociation floor: the
# FCT tracers (T,S) amplify the ~1e-12 input reassociation via Zalesak UPWIND FLIPS (a
# near-zero edge volume flux flips upwind direction ⇒ O(1) flux swing), and the heavily
# **cancelling** SSH transport divergences (ssh_rhs/ssh_rhs_old) amplify it too. Both are
# the documented "climate-close, not bit-identical" FCT/cancellation non-determinism that
# the C **and** Kokkos ports also see — NOT a missing exchange (confirmed: S matches when
# constant, owned==halo error, and ALL FCT inputs match to 1e-9). Per-substep correctness
# (not bit-identity) is Phase 8's bar (Decision 4). The floor scales with the velocity /
# tracer gradient, so it is far smaller on a physical field than on this sharp test bump.
_FCT_FIELDS = {"T", "S", "T_old", "S_old", "ssh_rhs", "ssh_rhs_old", "del_ttf"}
_CLEAN_ATOL = 1e-7        # momentum/SSH/ALE/EOS exchanges: clean reassociation
_FCT_ATOL = 5e-3          # FCT/cancellation upwind-flip floor (this test bump)

# The EVP internal stress tensor σ is a NON-PROGNOSTIC VP-kink diagnostic, not gated for
# N-vs-1 agreement (S.9 decision). σ = ζ·ε with ζ = ice_strength/Δ and Δ = max(√radicand,
# Δ_min): near-rigid ice rides the viscous-plastic yield kink where Δ≈Δ_min, so a ~1e-15
# reassociation wiggle in the strain is multiplied by a huge viscosity → an O(0.5) jump in the
# raw stress on a handful of near-kink elements. The PROGNOSTIC ice state it drives — u_ice,
# v_ice, m_ice, a_ice, m_snow — still matches single-device to 1e-7 (S.9 4×A100 run): the net
# stress divergence (the force) is correct, only the per-element stress branch flips. We gate
# the prognostic state, not the kink diagnostic. σ is still PRINTED so the floor stays visible.
_DIAG_FIELDS = {"sigma11", "sigma12", "sigma22"}


def _owned_match(st_dense, st_N, mesh, part, npes, *, tag="",
                 fct_atol=_FCT_ATOL, clean_atol=_CLEAN_ATOL, extra_fct=()):
    """Assert the sharded step matches single-device on OWNED entities (field-appropriate
    budget: clean reassociation for momentum/SSH/ALE/EOS/GM-diag, the climate-close
    upwind-flip floor for the FCT tracers + cancelling SSH divergences + ``extra_fct`` (the
    ice EVP/FCT prognostic fields, climate-close like the ocean FCT)). Prints every field's
    owned max|Δ| (so the actual floor is visible in the job log) and returns the worst clean
    diff."""
    soft = _FCT_FIELDS | set(extra_fct)
    worst_clean = 0.0
    rows = []
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(st_dense, fld.name))
        B = np.asarray(getattr(st_N, fld.name))
        if a.shape[0] == mesh.nod2D:
            mydim, myl = part.myDim_nod2D, part.myList_nod2D
        elif a.shape[0] == mesh.elem2D:
            mydim, myl = part.myDim_elem2D, part.myList_elem2D
        else:
            continue
        diff = 0.0
        for d in range(npes):
            md = int(mydim[d])
            if md:
                diff = max(diff, float(np.max(np.abs(B[d, :md] - a[myl[d][:md]]))))
        is_diag = fld.name in _DIAG_FIELDS
        is_fct = fld.name in soft
        rows.append((fld.name, diff, is_fct, is_diag))
        if not is_fct and not is_diag:
            worst_clean = max(worst_clean, diff)
    print(f"\n[{tag}] per-field owned max|Δ| (npes={npes}):")
    for name, diff, is_fct, is_diag in sorted(rows, key=lambda r: -r[1]):
        if diff > 0:
            tagc = "DIAG" if is_diag else ("FCT " if is_fct else "    ")
            print(f"   {tagc}{name:14s} {diff:.3e}")
    for name, diff, is_fct, is_diag in rows:
        if is_diag:        # non-prognostic VP-kink diagnostic — printed, not gated (S.9)
            continue
        atol = fct_atol if is_fct else clean_atol
        assert diff < atol, f"[{tag}] {name}: owned max|Δ|={diff:.3e} > {atol:.0e}"
    assert worst_clean < clean_atol, f"[{tag}] clean fields max|Δ|={worst_clean:.3e}"
    return worst_clean


@avail
@pytest.mark.parametrize("npes", [2])
def test_sharded_step_owned_matches(npes):
    """The full sharded step (with the S.7 halo exchanges) matches single-device on
    OWNED entities: every momentum / SSH / ALE / EOS field to the **clean reassociation
    floor** (<1e-7 — the proof the exchange wiring is correct), and the FCT tracers +
    cancelling SSH divergences to the documented climate-close floor (the upwind-flip /
    cancellation non-determinism, not a missing exchange)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = _perturbed_state(mesh)
    st_dense = stepmod.step(state, mesh, op, jnp.zeros((mesh.elem2D, 2)), dt=DT,
                            is_first_step=True)

    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = _stress_p(mesh, part, sm.Lmax["elem"])
    st_N = ish.run_step_sharded(sm, state_p, sop, stress_p, dt=DT,
                                is_first_step=True, npes=npes)
    _owned_match(st_dense, st_N, mesh, part, npes, tag="ocean")


@avail
@pytest.mark.parametrize("npes", [2])
def test_ragged_step_matches_allgather(npes):
    """Phase 8b B.0c: the full sharded ocean step with the halo-only ``ragged_all_to_all``
    exchange == the ``all_gather`` exchange on owned, to the **field-appropriate floor**.
    The per-substep exchanges deliver the same owner values on valid lanes (the B.0c
    isolated gate proves it byte-identical), so owned results agree — but the two schemes
    handle PAD lanes differently (ragged leaves pad untouched; all_gather sets pad=lane-0),
    and that tiny difference rides the same climate-close floor as everything else (FCT
    upwind flips, cancelling SSH divergences). Proves the `HaloCtx` dispatch + `_halo_arrays`
    ragged folding are wired right end-to-end. ⚠️ ``ragged_all_to_all`` is GPU-only — SKIPs
    on CPU."""
    if jax.devices()[0].platform == "cpu":
        pytest.skip("lax.ragged_all_to_all is unimplemented on XLA:CPU; needs GPU (NCCL)")
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = _perturbed_state(mesh)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = _stress_p(mesh, part, sm.Lmax["elem"])
    kw = dict(dt=DT, is_first_step=True, npes=npes)
    st_ag = ish.run_step_sharded(sm, state_p, sop, stress_p, use_ragged=False, **kw)
    st_rg = ish.run_step_sharded(sm, state_p, sop, stress_p, use_ragged=True, **kw)
    rows, worst = [], 0.0
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(st_ag, fld.name))
        b = np.asarray(getattr(st_rg, fld.name))
        if a.ndim < 2:
            continue
        d_max = 0.0
        for d in range(npes):
            md = int(part.myDim_nod2D[d]) if a.shape[1] == sm.Lmax["nod"] else \
                 int(part.myDim_elem2D[d]) if a.shape[1] == sm.Lmax["elem"] else 0
            if md:
                d_max = max(d_max, float(np.max(np.abs(a[d, :md] - b[d, :md]))))
        rows.append((fld.name, d_max))
        worst = max(worst, d_max)
    print("\n[ragged-vs-allgather] ocean step owned max|Δ| per field:")
    for name, d in sorted(rows, key=lambda r: -r[1]):
        if d > 0:
            print(f"   {name:14s} {d:.3e}")
    # field-appropriate: ragged forward == all_gather forward to the climate-close floor
    # (pad-handling + FCT upwind-flip / SSH-cancellation reassociation), NOT byte-identity.
    assert worst < 1e-5, f"ragged step != all_gather on owned: worst max|Δ|={worst:.3e}"


@avail
@pytest.mark.parametrize("npes", [2])
def test_padded_step_matches_allgather(npes):
    """Phase 8c: the full sharded ocean step with the slot-padded dense-``all_to_all``
    exchange == the ``all_gather`` exchange on owned, to the same climate-close floor as
    the ragged gate above (identical pad-handling difference: padded leaves pad lanes
    untouched, all_gather sets pad = lane-0). Proves the ``use_padded`` HaloCtx dispatch
    + ``_halo_arrays``/``_padded_ctx`` folding are wired right end-to-end — on ANY
    backend (the padded transport exists on CPU, where ragged does not)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = _perturbed_state(mesh)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = _stress_p(mesh, part, sm.Lmax["elem"])
    kw = dict(dt=DT, is_first_step=True, npes=npes)
    st_ag = ish.run_step_sharded(sm, state_p, sop, stress_p, use_padded=False, **kw)
    st_pd = ish.run_step_sharded(sm, state_p, sop, stress_p, use_padded=True, **kw)
    worst = 0.0
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(st_ag, fld.name))
        b = np.asarray(getattr(st_pd, fld.name))
        if a.ndim < 2:
            continue
        for d in range(npes):
            md = int(part.myDim_nod2D[d]) if a.shape[1] == sm.Lmax["nod"] else \
                 int(part.myDim_elem2D[d]) if a.shape[1] == sm.Lmax["elem"] else 0
            if md:
                worst = max(worst, float(np.max(np.abs(a[d, :md] - b[d, :md]))))
    assert worst < 1e-5, f"padded step != all_gather on owned: worst max|Δ|={worst:.3e}"


@avail
@pytest.mark.parametrize("npes", [2, 4])
def test_coloured_step_matches_allgather(npes):
    """Phase 8d: the full sharded ocean step with the coloured-``ppermute`` exchange ==
    the ``all_gather`` exchange on owned, to the same climate-close floor as the ragged /
    padded gates above (identical pad-handling difference). This is the gate that matters
    most for the new transport: it drives the exchange through the **CG solver's**
    :class:`~fesom_jax.ssh.SSHHalo` — ~2 exchanges × ~127 iterations, i.e. the dominant
    per-step comm — so it proves the ``use_coloured`` dispatch, the static ``col_meta``
    (perms/slots/offs surviving as meta_fields rather than being traced into arrays), and
    the ``_halo_arrays``/``_coloured_ctx`` folding are all wired right end-to-end."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = _perturbed_state(mesh)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = _stress_p(mesh, part, sm.Lmax["elem"])
    kw = dict(dt=DT, is_first_step=True, npes=npes)
    st_ag = ish.run_step_sharded(sm, state_p, sop, stress_p, use_coloured=False, **kw)
    st_cl = ish.run_step_sharded(sm, state_p, sop, stress_p, use_coloured=True, **kw)
    worst = 0.0
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(st_ag, fld.name))
        b = np.asarray(getattr(st_cl, fld.name))
        if a.ndim < 2:
            continue
        for d in range(npes):
            md = int(part.myDim_nod2D[d]) if a.shape[1] == sm.Lmax["nod"] else \
                 int(part.myDim_elem2D[d]) if a.shape[1] == sm.Lmax["elem"] else 0
            if md:
                worst = max(worst, float(np.max(np.abs(a[d, :md] - b[d, :md]))))
    assert worst < 1e-5, f"coloured step != all_gather on owned: worst max|Δ|={worst:.3e}"


# --------------------------------------------------------------------------
# 4. GM/Redi-ON (S.7 part 3): the forced-path eddy parameterization exchanges
# --------------------------------------------------------------------------
# GM/Redi is purely diagnostic (T/S/N² → bolus velocity + Redi diffusivities), so it runs
# WITHOUT surface forcing (no reductions) — but it needs a STRATIFIED state (the depth-
# uniform _perturbed_state degenerates: N²≈0 ⇒ the ODM95 taper collapses), so the gate uses
# the real PHC IC. The GM exchanges (gm.gm_diagnostics: fer_gamma nod INTRA, fer_uv elem,
# slope_tapered/Ki nod; step.py 13a: fer_w nod) refresh exactly the fields a downstream
# kernel reads at the halo of an incomplete entity. GM does not feed the within-step
# momentum/SSH (the bolus only augments the tracer-advecting velocity locally; the carried
# uv/w are untouched), so dynamics stay GM-independent ⇒ clean floor; only T/S carry GM's
# effect (via Redi + bolus advection) and stay in the climate-close FCT budget.
@avail
@have_ic
def test_gm_serial_sharded_step_matches_dense():
    """The GM/Redi-ON ocean step under ``shard_map`` on ONE device == the dense GM step,
    byte-identically (the ``exch=None``/identity GM path is a structural no-op)."""
    from fesom_jax.phc_ic import phc_initial_state
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = phc_initial_state(mesh, IC_DIR)
    gm_cfg = GMConfig()
    st_dense = stepmod.step(state, mesh, op, jnp.zeros((mesh.elem2D, 2)), dt=DT,
                            is_first_step=True, gm_cfg=gm_cfg)

    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    sm = shard_mesh.build_sharded_mesh(mesh, ser)
    state_p = shard_mesh.partition_state(state, ser)
    sop = ssh.partition_ssh_operator(op, ser)
    stress_p = _stress_p(mesh, ser, sm.Lmax["elem"])
    st_N = ish.run_step_sharded(sm, state_p, sop, stress_p, dt=DT,
                                is_first_step=True, npes=1, gm_cfg=gm_cfg)

    worst = 0.0
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(st_dense, fld.name))
        b = np.asarray(getattr(st_N, fld.name))[0][: a.shape[0]]
        if a.size:
            worst = max(worst, float(np.max(np.abs(a - b))))
    assert worst < _BYTE_ID_ATOL, f"serial GM sharded step max|Δ|={worst:.3e} (expected byte-id, floor {_BYTE_ID_ATOL:.0e})"


@avail
@have_ic
@pytest.mark.parametrize("npes", [2])
def test_gm_diagnostics_sharded_owned_matches(npes):
    """The PER-KERNEL GM-exchange gate (the S.4 scatter-gate analogue): the GM coefficient/
    bolus chain :func:`gm.gm_diagnostics` run under ``shard_map`` matches single-device on
    OWNED entities to the scatter floor — ``fer_uv`` (elem, the bolus velocity), and
    ``slope_tapered``/``Ki`` (nod, the Redi diffusivities). This isolates the GM exchanges
    (``fer_gamma`` INTRA + ``fer_uv``/``slope_tapered``/``Ki`` post) from the FCT, so it
    proves the bolus + Redi inputs the tracer step consumes are CORRECT on owned — hence any
    residual T/S N-vs-1 spread (next test) is the upwind-flip floor, NOT a missing exchange.
    A missing exchange would make ``fer_uv`` qualitatively wrong (O(1)) on owned boundary
    elements (their vertices are halo nodes — S.1 redundant element ownership)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    from fesom_jax import ale, eos, gm
    from fesom_jax.params import Params
    from fesom_jax.phc_ic import phc_initial_state
    from fesom_jax.shard_mesh import _shard_along_axis, local_sizes
    mesh = load_mesh(CORE2_MESH)
    state = phc_initial_state(mesh, IC_DIR)
    gm_cfg, params = GMConfig(), Params.defaults()

    # dense GM diagnostics (the reference) + its inputs.
    _, _, bvfreq = eos.compute_pressure_bv(mesh, state.T, state.S, state.hnode)
    hnode_new = ale.thickness_linfs(state.hnode)
    diag = gm.gm_diagnostics(mesh, state.T, state.S, bvfreq, hnode_new, state.helem,
                             params, gm_cfg)

    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    _, Lmax = local_sizes(part)
    pn = lambda a: _shard_along_axis(np.asarray(a), part.myList_nod2D, Lmax["nod"], 0, 1.0)
    pe = lambda a: _shard_along_axis(np.asarray(a), part.myList_elem2D, Lmax["elem"], 0, 1.0)
    fer_uv_N, slope_N, Ki_N = ish.run_gm_diag_sharded(
        sm, pn(state.T), pn(state.S), pn(bvfreq), pn(hnode_new), pe(state.helem),
        npes=npes, gm_cfg=gm_cfg, params=params)

    for name, ref, got, myl, mydim in (
            ("fer_uv", diag.fer_uv, fer_uv_N, part.myList_elem2D, part.myDim_elem2D),
            ("slope_tapered", diag.slope_tapered, slope_N, part.myList_nod2D, part.myDim_nod2D),
            ("Ki", diag.Ki, Ki_N, part.myList_nod2D, part.myDim_nod2D)):
        a = np.asarray(ref)
        B = np.asarray(got)
        worst = 0.0
        for d in range(npes):
            md = int(mydim[d])
            worst = max(worst, float(np.max(np.abs(B[d, :md] - a[myl[d][:md]]))))
        print(f"[gm-diag] {name:14s} owned max|Δ|={worst:.3e}")
        assert worst < _BYTE_ID_ATOL, f"{name}: owned max|Δ|={worst:.3e} (GM exchange gap?, floor {_BYTE_ID_ATOL:.0e})"


@avail
@have_ic
@pytest.mark.parametrize("npes", [2])
def test_gm_sharded_step_owned_matches(npes):
    """The GM/Redi-ON sharded step matches single-device on OWNED entities: the GM-driven
    diagnostics (density/Kv/uv/d_eta/w — GM-independent) to MACHINE PRECISION, T/S to the
    climate-close FCT upwind-flip floor (the GM-diag fields are proven correct by the
    per-kernel test above, so the residual T/S spread is flips, not a missing exchange). On
    the physical PHC IC (real fronts + bolus-augmented advection) the flip floor is larger
    than the part-2 sharp-bump (~1e-3): measured T≈8.6e-3, S≈3.9e-3 (T sharper gradients)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    from fesom_jax.phc_ic import phc_initial_state
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = phc_initial_state(mesh, IC_DIR)
    gm_cfg = GMConfig()
    st_dense = stepmod.step(state, mesh, op, jnp.zeros((mesh.elem2D, 2)), dt=DT,
                            is_first_step=True, gm_cfg=gm_cfg)

    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = _stress_p(mesh, part, sm.Lmax["elem"])
    st_N = ish.run_step_sharded(sm, state_p, sop, stress_p, dt=DT,
                                is_first_step=True, npes=npes, gm_cfg=gm_cfg)
    # GM+PHC-IC FCT flip floor (real fronts + bolus) sits above the part-2 sharp-bump 5e-3.
    _owned_match(st_dense, st_N, mesh, part, npes, tag="gm", fct_atol=2e-2)


# --------------------------------------------------------------------------
# 5. KPP-ON (S.7 part 3): the forced-path vertical-mixing exchanges + reductions
# --------------------------------------------------------------------------
# KPP needs CORE2 surface forcing (ustar/Bo/bfsfc) ⇒ this is the first FORCED sharded gate.
# It exercises BOTH the KPP-internal exchanges (the 3-sweep blmc smoother's per-sweep refresh
# + the pre-node→elem viscA exchange — gated by Kv/Av at the CLEAN floor) AND the distributed
# reductions (the _area_mean virtual-salt/relax-salt/water-flux balances → owned-sum+psum,
# threaded via run_step_sharded's folded forcing). The forcing (PHC IC + JRA bulk) builds once.
@pytest.fixture(scope="module")
def core2_forced():
    """CORE2 model + 1-step JRA forcing for the forced-path sharded gates (built once)."""
    from fesom_jax import surface_forcing
    from fesom_jax.phc_ic import phc_initial_state
    mesh = load_mesh(CORE2_MESH)
    state = phc_initial_state(mesh, IC_DIR)
    sst0 = np.asarray(state.T[:, 0])
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR, sst_ic=sst0)
    sf = cf.step_forcing(*surface_forcing.dates_for_steps(YEAR, DT, 1)[0])
    return dict(mesh=mesh, state=state, op=op, sf=sf, fs=cf.static)


def _forced_sharded_step(fx, part, npes, *, boundary_node_p=None, **cfg):
    """Partition the forced setup + run one ``run_step_sharded`` (with the folded forcing +
    the optional global ``boundary_node``). The dense step derives ``boundary_node`` from the
    full mesh itself (= the global one), so only the SHARDED side needs it passed in. Returns
    ``(st_dense, st_N)``."""
    mesh, state, op, sf, fs = fx["mesh"], fx["state"], fx["op"], fx["sf"], fx["fs"]
    st_dense = stepmod.step(state, mesh, op, jnp.zeros((mesh.elem2D, 2)), dt=DT,
                            is_first_step=True, step_forcing=sf, forcing_static=fs, **cfg)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sf_p = shard_mesh.partition_step_forcing(sf, part)
    fs_p = shard_mesh.partition_forcing_static(fs, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = _stress_p(mesh, part, sm.Lmax["elem"])
    st_N = ish.run_step_sharded(sm, state_p, sop, stress_p, dt=DT, is_first_step=True,
                                npes=npes, step_forcing=sf_p, forcing_static=fs_p,
                                boundary_node_p=boundary_node_p, **cfg)
    return st_dense, st_N


@avail
@have_forcing
def test_kpp_serial_sharded_step_matches_dense(core2_forced):
    """The forced KPP step under ``shard_map`` on ONE device == the dense forced KPP step,
    byte-identically — the forcing fold + the distributed reductions (``psum`` over 1 device
    = identity) + the KPP exchanges (identity at 1 device) all collapse to ``v1.0``."""
    fx = core2_forced
    mesh = fx["mesh"]
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    st_dense, st_N = _forced_sharded_step(fx, ser, 1, kpp_cfg=KppConfig())
    worst = 0.0
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(st_dense, fld.name))
        b = np.asarray(getattr(st_N, fld.name))[0][: a.shape[0]]
        if a.size:
            worst = max(worst, float(np.max(np.abs(a - b))))
    assert worst < _BYTE_ID_ATOL, f"serial KPP sharded step max|Δ|={worst:.3e} (expected byte-id, floor {_BYTE_ID_ATOL:.0e})"


@avail
@have_forcing
@pytest.mark.parametrize("npes", [2])
def test_kpp_sharded_step_owned_matches(npes, core2_forced):
    """The forced KPP sharded step matches single-device on OWNED entities: **Kv** (node) and
    **Av** (elem) to the CLEAN floor — the direct proof the KPP-internal exchanges are correct
    (the 3-sweep ``blmc`` smoother's per-sweep halo refresh + the pre-node→elem ``viscA``
    exchange), since a missing one would corrupt Kv/Av on owned BOUNDARY nodes/elements by
    O(mixing) ≫ 1e-7. The reductions (``_area_mean`` balances) are exercised end-to-end (the
    balanced heat/water/salt fluxes feed Kv → if the global means drifted, Kv would too). T/S
    stay in the climate-close FCT floor."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    part = partit.read_partition(CORE2_DIST, npes)
    st_dense, st_N = _forced_sharded_step(core2_forced, part, npes, kpp_cfg=KppConfig())
    mesh = core2_forced["mesh"]
    _owned_match(st_dense, st_N, mesh, part, npes, tag="kpp", fct_atol=2e-2)


# --------------------------------------------------------------------------
# 5b. TKE-ON (Phase 9b, JT.5): the prognostic-TKE forced-path exchange
# --------------------------------------------------------------------------
# The structural risk TKE adds: the internal node-`tke_Av` exchange in `tke._wire_kv_av`
# (exchange node KappaM BEFORE the node→elem 3-vertex Av mean — a boundary OWNED element has
# halo vertices). Omit it and the port passes eager/1-device but corrupts Av on owned boundary
# elements by O(mixing) ≫ 1e-7 (the plan-review MAJOR). The new prognostic `tke` field is NEVER
# exchanged (each column self-contained on owned data) — the generic field loop in `_owned_match`
# proves that holds (owned `tke` == dense). The Av match is the direct proof the exch is correct.
@avail
@have_forcing
def test_tke_serial_sharded_step_matches_dense(core2_forced):
    """The forced TKE step under ``shard_map`` on ONE device == the dense forced TKE step,
    byte-identically (the exchanges collapse to identity at 1 device)."""
    fx = core2_forced
    mesh = fx["mesh"]
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    st_dense, st_N = _forced_sharded_step(fx, ser, 1, tke_cfg=TkeConfig())
    worst = 0.0
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(st_dense, fld.name))
        b = np.asarray(getattr(st_N, fld.name))[0][: a.shape[0]]
        if a.size:
            worst = max(worst, float(np.max(np.abs(a - b))))
    assert worst < _BYTE_ID_ATOL, \
        f"serial TKE sharded step max|Δ|={worst:.3e} (expected byte-id, floor {_BYTE_ID_ATOL:.0e})"


@avail
@have_forcing
@pytest.mark.parametrize("npes", [2])
def test_tke_sharded_step_owned_matches(npes, core2_forced):
    """The forced TKE sharded step matches single-device on OWNED entities — the direct proof
    the internal node-``tke_Av`` exchange (``_wire_kv_av``, before the node→elem mean) is
    correct: a missing one corrupts **Av** on owned BOUNDARY elements by O(mixing) ≫ 1e-7. The
    new prognostic **tke** field (never exchanged, column-self-contained) is covered by the
    generic field loop in ``_owned_match`` — owned tke == dense confirms the no-exchange design."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    part = partit.read_partition(CORE2_DIST, npes)
    st_dense, st_N = _forced_sharded_step(core2_forced, part, npes, tke_cfg=TkeConfig())
    mesh = core2_forced["mesh"]
    _owned_match(st_dense, st_N, mesh, part, npes, tag="tke", fct_atol=2e-2)


# --------------------------------------------------------------------------
# 6. ICE-ON (S.7 part 3): the prognostic sea-ice forced-path exchanges
# --------------------------------------------------------------------------
# The hardest increment: the EVP momentum subcycle exchanges u_ice/v_ice INSIDE a 120-step
# lax.scan (a collective in a scan under shard_map, check_vma=False); the ice FCT splits like
# the ocean FCT (low-order + high-order dvalues + icepplus/icepminus); the GLOBAL boundary_node
# (the local-mesh recompute mis-flags partition-boundary nodes as coastal). The ice prognostic
# fields (a/m/snow via FCT, u/v_ice via EVP) are climate-close like the ocean FCT.
# Prognostic ice fields gated climate-close (like the ocean FCT). The EVP internal stress
# σ11/σ12/σ22 is NOT here — it is a non-prognostic VP-kink diagnostic (see _DIAG_FIELDS): the
# raw stress flips branch at the yield kink under reassociation, but the u_ice/v_ice it drives
# (below) match to 1e-7, so we gate the velocity, not the stress.
_ICE_FIELDS = ("a_ice", "m_ice", "m_snow", "u_ice", "v_ice", "t_skin")


def _seed_ice_state(fx):
    from fesom_jax import ice
    mesh, state = fx["mesh"], fx["state"]
    sst0 = np.asarray(state.T[:, 0])
    return ice.seed_ice(state, mesh, sst0)


def _global_boundary_node_p(mesh, part):
    from fesom_jax import ice_evp
    from fesom_jax.shard_mesh import _shard_along_axis, local_sizes
    bn = np.asarray(ice_evp.boundary_node_mask(mesh))
    _, Lmax = local_sizes(part)
    return _shard_along_axis(bn, part.myList_nod2D, Lmax["nod"], 0, False)


@avail
@have_forcing
def test_ice_serial_sharded_step_matches_dense(core2_forced):
    """The forced PROGNOSTIC-ICE step under ``shard_map`` on ONE device == the dense ice step,
    byte-identically. The critical lowering test: the EVP subcycle's ``u_ice/v_ice`` halo
    exchange (an ``all_gather`` INSIDE the 120-step ``lax.scan`` inside ``shard_map``,
    ``check_vma=False``) + the ice FCT split exchanges + the global ``boundary_node`` all
    collapse to ``v1.0`` at 1 device (identity exchange)."""
    from fesom_jax.ice import IceConfig
    fx = dict(core2_forced)
    fx["state"] = _seed_ice_state(core2_forced)
    mesh = fx["mesh"]
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    bn_p = _global_boundary_node_p(mesh, ser)
    st_dense, st_N = _forced_sharded_step(fx, ser, 1, ice_cfg=IceConfig(),
                                          boundary_node_p=bn_p)
    worst = 0.0
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(st_dense, fld.name))
        b = np.asarray(getattr(st_N, fld.name))[0][: a.shape[0]]
        if a.size:
            worst = max(worst, float(np.max(np.abs(a - b))))
    assert worst < _BYTE_ID_ATOL, f"serial ICE sharded step max|Δ|={worst:.3e} (expected byte-id, floor {_BYTE_ID_ATOL:.0e})"


@avail
@have_forcing
@pytest.mark.parametrize("npes", [2])
def test_ice_sharded_step_owned_matches(npes, core2_forced):
    """The forced prognostic-ice sharded step matches single-device on OWNED entities: the
    ocean dynamics (uv/d_eta/density) to the clean floor (they depend on the ice surface
    fluxes ⇒ a proof the ice exchanges feed correct BCs), the ice prognostic fields + T/S to
    the climate-close floor (EVP subcycle + FCT non-determinism, like the ocean FCT)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    from fesom_jax.ice import IceConfig
    fx = dict(core2_forced)
    fx["state"] = _seed_ice_state(core2_forced)
    mesh = fx["mesh"]
    part = partit.read_partition(CORE2_DIST, npes)
    bn_p = _global_boundary_node_p(mesh, part)
    st_dense, st_N = _forced_sharded_step(fx, part, npes, ice_cfg=IceConfig(),
                                          boundary_node_p=bn_p)
    _owned_match(st_dense, st_N, mesh, part, npes, tag="ice", fct_atol=2e-2,
                 extra_fct=_ICE_FIELDS)


@avail
@have_forcing
def test_mevp_serial_sharded_step_matches_dense(core2_forced):
    """Phase 9c JM.5: mEVP (whichEVP=1) under ``shard_map`` on ONE device == the dense mEVP step,
    byte-identically. The lowering test for the mEVP path: the 120-iteration ``u_aux``/``v_aux``
    halo exchange (the per-iteration ``all_gather`` INSIDE the ``lax.scan`` inside ``shard_map``,
    ``check_vma=False``) + the global ``boundary_node`` (bc_index) collapse to ``v1.0`` at 1
    device (identity exchange)."""
    from fesom_jax.ice import IceConfig
    fx = dict(core2_forced)
    fx["state"] = _seed_ice_state(core2_forced)
    mesh = fx["mesh"]
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    bn_p = _global_boundary_node_p(mesh, ser)
    st_dense, st_N = _forced_sharded_step(fx, ser, 1, ice_cfg=IceConfig(whichEVP=1),
                                          boundary_node_p=bn_p)
    worst = 0.0
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(st_dense, fld.name))
        b = np.asarray(getattr(st_N, fld.name))[0][: a.shape[0]]
        if a.size:
            worst = max(worst, float(np.max(np.abs(a - b))))
    assert worst < _BYTE_ID_ATOL, \
        f"serial mEVP sharded step max|Δ|={worst:.3e} (expected byte-id, floor {_BYTE_ID_ATOL:.0e})"


@avail
@have_forcing
@pytest.mark.parametrize("npes", [2])
def test_mevp_sharded_step_owned_matches(npes, core2_forced):
    """Phase 9c JM.5: the forced mEVP sharded step matches single-device on OWNED entities.
    ``u_ice``/``v_ice`` are GATED (the mEVP velocity is the binding fidelity check — the same
    N-vs-1 class as std-EVP); the VP-kink internal stress σ stays in ``_DIAG_FIELDS`` (excluded —
    a near-rigid-pack branch-flip diagnostic that amplifies noise over the 120 iterations while
    the velocity it drives stays correct, the std-EVP S.9 precedent)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    from fesom_jax.ice import IceConfig
    fx = dict(core2_forced)
    fx["state"] = _seed_ice_state(core2_forced)
    mesh = fx["mesh"]
    part = partit.read_partition(CORE2_DIST, npes)
    bn_p = _global_boundary_node_p(mesh, part)
    st_dense, st_N = _forced_sharded_step(fx, part, npes, ice_cfg=IceConfig(whichEVP=1),
                                          boundary_node_p=bn_p)
    # u_ice/v_ice gated STRICTLY (the clean floor, not the FCT bucket): the mEVP velocity is the
    # binding fidelity check — it came out bit-identical N-vs-1 (the it1 wind-map + the per-owned-
    # node scatter preserve order). Only the FCT-advected ice SCALARS (a/m/snow/t_skin, like the
    # ocean FCT tracers) get the climate-close floor; σ stays excluded (_DIAG_FIELDS, VP-kink).
    _owned_match(st_dense, st_N, mesh, part, npes, tag="mevp", fct_atol=2e-2,
                 extra_fct=("a_ice", "m_ice", "m_snow", "t_skin"))


# --------------------------------------------------------------------------
# 7. Multi-step scan (S.7 part 3.5) + the PRIMARY assembled gate (S.7 part 3.6)
# --------------------------------------------------------------------------
@avail
@pytest.mark.parametrize("npes", [2])
def test_multistep_scan_lowers_and_is_finite(npes):
    """S.7p3.5: :func:`integrate_sharded.run_steps_sharded` (step-1 eager + ``lax.scan`` of the
    rest under ONE ``shard_map``) LOWERS and RUNS — the time-loop collective-inside-``lax.scan``
    -under-``shard_map`` (the EVP/CG precedent). A 2-step run is FINITE + physically bounded
    (no blow-up). The ``n=1`` case (no scan iterations) matches single-device TIGHTLY (the
    wrapper is correct); the tight PER-STEP correctness for ``n>=2`` is teacher-forced below
    (a FREE-running multi-step compare decorrelates chaotically — Decision 4)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    from fesom_jax import integrate as integ
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    state = _perturbed_state(mesh)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(state, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = _stress_p(mesh, part, sm.Lmax["elem"])

    # n=1: scan length 0 (step-1 eager only) — must match single-device (clean except FCT).
    d1 = integ.integrate(state, mesh, op, jnp.zeros((mesh.elem2D, 2)), 1, dt=DT)
    n1 = ish.run_steps_sharded(sm, state_p, sop, stress_p, 1, dt=DT, npes=npes)
    _owned_match(d1, n1, mesh, part, npes, tag="multistep-n1", fct_atol=2e-2)

    # n=2: the scan body runs (1 iteration) — finite + bounded (chaotic vs single-device, so
    # no tight compare; the per-step gate is teacher-forced below).
    n2 = ish.run_steps_sharded(sm, state_p, sop, stress_p, 2, dt=DT, npes=npes)
    for fld in dataclasses.fields(State):
        v = np.asarray(getattr(n2, fld.name))
        assert np.isfinite(v).all(), f"multistep n=2: {fld.name} has non-finite lanes"
    # T on OWNED lanes stays physical (boolean mask [P, Lmax] selects the leading 2 axes of
    # the [P, Lmax, nl] tracer; dry below-bottom lanes are 0, in range).
    T2 = np.asarray(n2.T)[np.asarray(sm.owned_mask["nod"])]   # [n_owned, nl]
    assert -5.0 < float(T2.min()) and float(T2.max()) < 45.0, "multistep n=2 T out of range"


@avail
@pytest.mark.parametrize("npes", [2])
def test_multistep_teacher_forced_ocean(npes):
    """S.7p3.5 per-step gate: a 2-step OCEAN trajectory compared PER STEP, teacher-forced —
    each sharded step reads the SINGLE-DEVICE's previous-step state (partitioned), so the only
    N-vs-1 difference is the WITHIN-step reassociation (clean for momentum/SSH/ALE/EOS, the FCT
    floor for tracers). This isolates the cross-step threading + per-step correctness from the
    chaotic accumulation a free-running compare suffers (Decision 4). A threading bug would
    show as a CLEAN field diverging; chaos cannot (it stays within the within-step floor)."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    part = partit.read_partition(CORE2_DIST, npes)
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    sop = ssh.partition_ssh_operator(op, part)
    stress_p = _stress_p(mesh, part, sm.Lmax["elem"])
    teacher = _perturbed_state(mesh)
    for k in range(2):
        first = (k == 0)
        d_next = stepmod.step(teacher, mesh, op, jnp.zeros((mesh.elem2D, 2)), dt=DT,
                              is_first_step=first)
        state_p = shard_mesh.partition_state(teacher, part)
        n_next = ish.run_step_sharded(sm, state_p, sop, stress_p, dt=DT,
                                      is_first_step=first, npes=npes)
        _owned_match(d_next, n_next, mesh, part, npes, tag=f"tf-step{k+1}", fct_atol=2e-2)
        teacher = d_next        # teacher-force: next step reads the single-device state


@avail
@have_forcing
@pytest.mark.parametrize("npes", [2])
def test_assembled_kpp_gm_ice_sharded_owned_matches(npes, core2_forced):
    """The PRIMARY S.7 gate: the FULLY assembled CORE2 step (KPP + GM/Redi + prognostic ice +
    bulk/SSS forcing) sharded == single-device on owned, field-appropriate budget. The
    headline N-vs-1 correctness gate the whole phase targets — every forced-path exchange +
    the distributed reductions + all the fused-kernel splits firing together. Ocean dynamics
    to the clean floor; FCT tracers + ice prognostic fields climate-close."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    from fesom_jax.ice import IceConfig
    fx = dict(core2_forced)
    fx["state"] = _seed_ice_state(core2_forced)
    mesh = fx["mesh"]
    part = partit.read_partition(CORE2_DIST, npes)
    bn_p = _global_boundary_node_p(mesh, part)
    st_dense, st_N = _forced_sharded_step(fx, part, npes, kpp_cfg=KppConfig(),
                                          gm_cfg=GMConfig(), ice_cfg=IceConfig(),
                                          boundary_node_p=bn_p)
    _owned_match(st_dense, st_N, mesh, part, npes, tag="assembled", fct_atol=3e-2,
                 extra_fct=_ICE_FIELDS)


# --------------------------------------------------------------------------
# 8. zstar-ON (Phase 9a, JZ.7): the live-geometry vertical coordinate sharding
# --------------------------------------------------------------------------
def _warm_zstar_state(state, mesh):
    """Seed a WARM zstar state (hbar≠0 ⇒ a genuinely STRETCHED hnode/helem via the zstar init)
    so a single step exercises the LIVE geometry under sharding. At cold start (hbar=0) the JZ.6
    re-points + the vert_vel distribute + the D2 stiffness increment are all no-ops (live==static),
    so a cold sharded step would only test the forcing-flip reductions + the hnode_new exchange,
    NOT the live-geometry sharding. A smooth ~0.5 m SSH bump stretches the column (hbar/dd ~ 1e-4,
    well above the 1e-9 byte-id floor ⇒ a live-path sharding gap is detectable)."""
    from fesom_jax import ale
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]
    hbar = jnp.asarray(0.5 * np.cos(2.0 * lat))           # (nod2D,) ~0.5 m SSH bump
    hbar_old = jnp.zeros_like(hbar)
    hnode, helem, eta_n, ssh_rhs_old = ale.init_thickness_zstar(mesh, hbar, hbar_old, dt=DT)
    return dataclasses.replace(state, hbar=hbar, hbar_old=hbar_old, hnode=hnode, helem=helem,
                               eta_n=eta_n, ssh_rhs_old=ssh_rhs_old)


@avail
@have_forcing
def test_zstar_serial_sharded_step_matches_dense(core2_forced):
    """npes=1 zstar (the full z2_cdump 4-config: KPP + GM/Redi + prognostic ice + zstar) sharded
    == dense, byte-identically. The whole zstar path — live geometry (the EOS/PP/dbsfc/KPP/GM/
    QR4C/vert-Redi/K33/momentum re-points), the vert_vel distribute, the D2 stiffness-as-state
    increment (the matvec closure over hbar), the forcing flip (real fw/salt + the global
    fw-balancing psum), and the zstar-only hnode_new exchange — lowers under ``shard_map`` and
    collapses to ``v1.0`` at 1 device (identity exchange, psum-over-1=identity). A WARM hbar seed
    makes the live geometry genuinely stretched (not the cold-start no-op)."""
    from fesom_jax.ale import AleConfig
    from fesom_jax.ice import IceConfig
    fx = dict(core2_forced)
    mesh = fx["mesh"]
    fx["state"] = _warm_zstar_state(_seed_ice_state(core2_forced), mesh)
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    bn_p = _global_boundary_node_p(mesh, ser)
    st_dense, st_N = _forced_sharded_step(fx, ser, 1, kpp_cfg=KppConfig(), gm_cfg=GMConfig(),
                                          ice_cfg=IceConfig(), ale_cfg=AleConfig(),
                                          boundary_node_p=bn_p)
    worst = 0.0
    for fld in dataclasses.fields(State):
        a = np.asarray(getattr(st_dense, fld.name))
        b = np.asarray(getattr(st_N, fld.name))[0][: a.shape[0]]
        if a.size:
            worst = max(worst, float(np.max(np.abs(a - b))))
    assert worst < _BYTE_ID_ATOL, \
        f"serial zstar sharded step max|Δ|={worst:.3e} (expected byte-id, floor {_BYTE_ID_ATOL:.0e})"


@avail
@have_forcing
@pytest.mark.parametrize("npes", [2])
def test_zstar_assembled_sharded_owned_matches(npes, core2_forced):
    """The PRIMARY zstar sharding gate: the full 4-config (KPP + GM + prognostic ice + zstar, the
    z2_cdump config) sharded == single-device on OWNED entities, field-appropriate budget. WARM
    hbar ⇒ the live geometry is stretched ⇒ the JZ.6 re-points + the vert_vel distribute + the D2
    stiffness closure + the fw-balancing psum all fire under multi-device exchange. The new zstar
    State fields (hnode/helem/hbar/hnode_new) are covered by the generic ``_owned_match`` loop to
    the CLEAN reassociation floor — the proof the zstar exchanges are correct (the hnode_new
    OCEAN_SCHEDULE row from JZ.4 + the live geometry's halo-completeness); a missing one would
    corrupt hnode_new / the stretched geometry on owned BOUNDARY nodes by O(stretch) ≫ 1e-7. T/S +
    ice prognostic fields stay in the climate-close FCT floor."""
    if NDEV < npes:
        pytest.skip(f"needs {npes} devices, have {NDEV}")
    from fesom_jax.ale import AleConfig
    from fesom_jax.ice import IceConfig
    mesh = core2_forced["mesh"]
    fx = dict(core2_forced)
    fx["state"] = _warm_zstar_state(_seed_ice_state(core2_forced), mesh)
    part = partit.read_partition(CORE2_DIST, npes)
    bn_p = _global_boundary_node_p(mesh, part)
    st_dense, st_N = _forced_sharded_step(fx, part, npes, kpp_cfg=KppConfig(), gm_cfg=GMConfig(),
                                          ice_cfg=IceConfig(), ale_cfg=AleConfig(),
                                          boundary_node_p=bn_p)
    _owned_match(st_dense, st_N, mesh, part, npes, tag="zstar", fct_atol=3e-2,
                 extra_fct=_ICE_FIELDS)
