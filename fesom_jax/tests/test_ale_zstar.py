"""Phase 9a (zstar vertical coordinate) tests — grows one task at a time (JZ.0..JZ.8).

**JZ.0 (scaffolding, NO behavior change):**

* :class:`~fesom_jax.ale.AleConfig` — the static zstar seam: derived ``use_virt_salt`` /
  ``is_nonlinfs`` (the C ``fesom_ale.c:31-32`` globals), the unsupported-mode guard
  (``fesom_ale_mode_init`` abort parity), hashability (it rides as a ``static_argname``).
* The ``ale_cfg`` threading is a **no-op** while no kernel branches on it: ``step(…,
  ale_cfg=None)`` is bit-identical to ``step(…)``, and the seam validates a bad cfg.
* The ALE dump reader (``io_dump.load_ale_dump``) round-trips the multi-rank
  ``z2_cdump`` (12 tags, merge-by-gid), matching the C ``fesom_ale_dump.c`` layout.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ale, forcing, ic, io_dump, ssh
from fesom_jax import step as stepmod
from fesom_jax.ale import AleConfig
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.state import State

# The 3-step 16-rank C dump set (plan §1 oracle; existence verified 2026-06-11). The
# tags live in the `dump/` subdir written by FESOM_ALE_DUMP_DIR.
ZSTAR_ORACLE = Path("/work/ab0995/a270088/port/zstar/z2_cdump/dump")
NG5_NOD2D = 126858    # global node count of the NG5 dist_16 oracle mesh
NG5_ELEM2D = 244659   # global element count
NG5_NL = 48
DT = 100.0


# ==========================================================================
# 1. AleConfig — the static zstar seam (no mesh needed; fast)
# ==========================================================================
def test_aleconfig_defaults_are_zstar():
    cfg = AleConfig()
    assert cfg.zstar is True
    # the C's two mode globals (fesom_ale.c:31-32): zstar ⇒ no virtual salt, nonlinfs=1
    assert cfg.use_virt_salt is False
    assert cfg.is_nonlinfs == 1.0


def test_aleconfig_validate_rejects_unsupported_mode():
    # presence of the cfg ⇒ zstar; zstar=False is the unsupported zlevel/linfs request
    # (linfs is ale_cfg=None, not an AleConfig). The C aborts (fesom_ale.c:25-30).
    with pytest.raises(ValueError, match="zstar"):
        AleConfig(zstar=False).validate()
    # the valid one validates to itself (wrappable)
    cfg = AleConfig()
    assert cfg.validate() is cfg


def test_aleconfig_is_hashable_static_arg():
    # rides as a static_argname (like GMConfig/KppConfig/IceConfig) ⇒ must be hashable
    assert hash(AleConfig()) == hash(AleConfig())
    assert AleConfig() == AleConfig()
    # derived properties are not tuple fields (don't break equality/hash)
    d = {AleConfig(): 1}
    assert d[AleConfig()] == 1


# ==========================================================================
# 2. ale_cfg threading is a no-op at JZ.0 (None ⇒ bit-identical; bad cfg raises)
# ==========================================================================
@pytest.fixture(scope="module")
def pi_model():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (Task 0.3)")
    mesh = load_mesh()
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress = forcing.surface_stress(mesh)
    st0 = ic.initial_state(mesh)
    return mesh, op, stress, st0


def test_ale_cfg_none_is_bit_identical(pi_model):
    """Threading ``ale_cfg=None`` through ``step`` reproduces the pre-seam step exactly
    (the standing ``ale_cfg=None`` byte-identity invariant — asserted here directly,
    and by the whole suite at every task boundary)."""
    mesh, op, stress, st0 = pi_model
    base = stepmod.step(st0, mesh, op, stress, dt=DT, is_first_step=True)
    threaded = stepmod.step(st0, mesh, op, stress, dt=DT, is_first_step=True, ale_cfg=None)
    for f in dataclasses.fields(base):
        a = np.asarray(getattr(base, f.name))
        b = np.asarray(getattr(threaded, f.name))
        assert np.array_equal(a, b), f"ale_cfg=None changed field {f.name}"


def test_ale_cfg_seam_validates_in_step(pi_model):
    """The step seam runs the abort-parity guard: an unsupported cfg raises before any
    work (the C ``fesom_ale_mode_init`` exit(1))."""
    mesh, op, stress, st0 = pi_model
    with pytest.raises(ValueError, match="zstar"):
        stepmod.step(st0, mesh, op, stress, dt=DT, is_first_step=True,
                     ale_cfg=AleConfig(zstar=False))


# ==========================================================================
# 3. ALE dump reader — multi-rank merge-by-gid round-trip on z2_cdump
# ==========================================================================
oracle_missing = pytest.mark.skipif(
    not ZSTAR_ORACLE.is_dir(),
    reason=f"zstar C dump oracle missing: {ZSTAR_ORACLE} (plan §1)")


@oracle_missing
def test_ale_dump_all_tags_present_headers():
    """All 12 tags exist for steps 1..3 × 16 ranks, and each header's ncomp matches the
    C layout (scalar ⇒ len(comps); layer ⇒ nl-1; iface ⇒ nl)."""
    expected_ncomp = {
        "scalar": None,  # len(comps), filled per tag below
        "layer": NG5_NL - 1,
        "iface": NG5_NL,
    }
    for tag, spec in io_dump.ALE_TAGS.items():
        want = len(spec.comps) if spec.kind == "scalar" else expected_ncomp[spec.kind]
        for step in (1, 2, 3):
            files = sorted(ZSTAR_ORACLE.glob(f"ale_dump_s{step}_{tag}_rank*.txt"))
            assert len(files) == 16, f"{tag} step {step}: {len(files)} ranks (want 16)"
            # cheap: parse only the header line of rank 0
            with open(files[0]) as fh:
                hdr = io_dump._parse_gid_header(fh.readline())
            assert hdr["ncomp"] == want, f"{tag}: ncomp {hdr['ncomp']} != {want}"
            assert hdr["tag"] == tag


@oracle_missing
@pytest.mark.parametrize("tag,n,ncomp", [
    ("forcing", NG5_NOD2D, 4),      # scalar node pack
    ("sshsolve", NG5_NOD2D, 2),     # scalar node pack
    ("hbar", NG5_NOD2D, 4),         # scalar node pack
    ("dhe", NG5_ELEM2D, 1),         # scalar element
    ("Wvel", NG5_NOD2D, NG5_NL),    # iface node column
    ("hnode", NG5_NOD2D, NG5_NL - 1),    # layer node column
    ("helem", NG5_ELEM2D, NG5_NL - 1),   # layer element column (boundary-ring dupes)
])
def test_ale_dump_merge_shapes_and_coverage(tag, n, ncomp):
    """Merge-by-gid yields a fully-covered global array of the right shape (no NaN
    holes ⇒ the disjoint node / bit-identical-overlap element invariant held)."""
    fields, meta = io_dump.load_ale_dump(
        ZSTAR_ORACLE, [tag], step=1, n_nod=NG5_NOD2D, n_elem=NG5_ELEM2D)
    arr = fields[tag]
    assert arr.shape == (n, ncomp)
    assert meta[tag]["nranks"] == 16
    assert not np.isnan(arr).any(), f"{tag}: NaN holes ⇒ incomplete gid coverage"


@oracle_missing
def test_ale_dump_merge_matches_single_rank_read():
    """Spot-check the merged global array against a direct single-rank read: every gid
    in rank-0's owned set must carry rank-0's exact row."""
    fields, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["forcing"], step=1, n_nod=NG5_NOD2D)
    g0, v0, _ = io_dump.read_gid_table(ZSTAR_ORACLE / "ale_dump_s1_forcing_rank0.txt")
    merged = fields["forcing"]
    assert np.array_equal(merged[g0 - 1], v0)


@oracle_missing
def test_ale_component_accessor():
    """The named-component accessor splits a scalar pack by the C getter order."""
    fields, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["forcing"], step=1, n_nod=NG5_NOD2D)
    wf = io_dump.ale_component(fields, "forcing", "water_flux")
    rsf = io_dump.ale_component(fields, "forcing", "real_salt_flux")
    assert wf.shape == (NG5_NOD2D,)
    assert np.array_equal(wf, fields["forcing"][:, 0])     # water_flux is comp 0
    assert np.array_equal(rsf, fields["forcing"][:, 3])    # real_salt_flux is comp 3
    # a column tag has no named components
    with pytest.raises(ValueError, match="column"):
        io_dump.ale_component(fields, "hnode", "anything")


@oracle_missing
def test_ale_dump_strict_detects_truncated_rank_set(tmp_path):
    """A missing rank ⇒ uncovered gids ⇒ strict load raises (guards against a silently
    partial dump being read as complete)."""
    # copy only rank 0 of a node tag into an isolated dir
    src = ZSTAR_ORACLE / "ale_dump_s1_sshsolve_rank0.txt"
    (tmp_path / src.name).write_bytes(src.read_bytes())
    with pytest.raises(ValueError, match="never written"):
        io_dump.load_ale_dump(tmp_path, ["sshsolve"], step=1, n_nod=NG5_NOD2D)


# ==========================================================================
# 4. JZ.1 — thickness machinery: init_thickness_zstar + live_geometry
# ==========================================================================
@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (Task 0.3)")
    return load_mesh()


def _ref_init_hnode(mesh, hbar):
    """Independent numpy loop reference for the zstar init thickness (golden-rule:
    a different code path than the vectorized `init_thickness_zstar`)."""
    zbar = np.asarray(mesh.zbar); nl = zbar.shape[0]
    minf = np.asarray(mesh.nlevels_nod2D_min)
    nmask = np.asarray(mesh.node_layer_mask)
    hbar = np.asarray(hbar)
    hnode = np.zeros((mesh.nod2D, nl))
    for n in range(mesh.nod2D):
        dd = zbar[0] - zbar[minf[n] - 2]
        for nz in range(nl):
            if not nmask[n, nz]:
                continue
            dz = zbar[nz] - zbar[nz + 1]
            f = (1.0 + hbar[n] / dd) if nz < minf[n] - 2 else 1.0
            hnode[n, nz] = dz * f
    return hnode


def _ref_live_geometry(mesh, hnode):
    """Independent numpy reference for `live_geometry`: the C bottom→top recurrence
    (`fesom_ale.c:233-239`) anchored at the nominal interface min_f-2."""
    zbar = np.asarray(mesh.zbar); nl = zbar.shape[0]
    minf = np.asarray(mesh.nlevels_nod2D_min)
    Znom = np.asarray(mesh.Z)
    hnode = np.asarray(hnode)
    zb = np.broadcast_to(zbar[None, :], (mesh.nod2D, nl)).copy()      # nominal anchor + below
    Z = np.concatenate([Znom, Znom[-1:]])[None, :].repeat(mesh.nod2D, 0).copy()
    for n in range(mesh.nod2D):
        for nz in range(minf[n] - 3, -1, -1):                        # bottom→top stretch
            zb[n, nz] = zb[n, nz + 1] + hnode[n, nz]
            Z[n, nz] = zb[n, nz + 1] + hnode[n, nz] / 2.0
    return zb, Z


def test_coldstart_degeneracy_init_equals_linfs(mesh):
    """At cold start (hbar=0) the zstar init thickness is BIT-FOR-BIT the linfs rest init
    (the free Z1 gate): stretch factor (1+hbar/dd)=1 ⇒ nominal everywhere; eta_n/ssh_rhs_old
    are 0."""
    rest = State.rest(mesh)
    z = jnp.zeros((mesh.nod2D,), jnp.float64)
    hnode, helem, eta_n, ssh_rhs_old = ale.init_thickness_zstar(mesh, z, z)
    assert np.array_equal(np.asarray(hnode), np.asarray(rest.hnode))
    assert np.array_equal(np.asarray(helem), np.asarray(rest.helem))
    assert float(jnp.max(jnp.abs(eta_n))) == 0.0
    assert float(jnp.max(jnp.abs(ssh_rhs_old))) == 0.0


def test_state_rest_ale_cfg_is_coldstart_identical(mesh):
    """State.rest(ale_cfg=AleConfig()) is byte-identical to the linfs rest (the explicit
    zstar IC entry point is a no-op on the values at cold start)."""
    rest = State.rest(mesh)
    rest_z = State.rest(mesh, ale_cfg=AleConfig())
    for f in ("hnode", "hnode_new", "helem", "eta_n", "ssh_rhs_old", "T", "S"):
        assert np.array_equal(np.asarray(getattr(rest, f)),
                              np.asarray(getattr(rest_z, f))), f


def test_live_geometry_reproduces_static_under_nominal(mesh):
    """`live_geometry(nominal hnode)` reproduces the static `zbar_3d_n`/`Z` on the WET
    lanes (linfs neutrality). On pi the round depths telescope EXACTLY (bitwise); the
    ≤1-ulp allowance is for general bathymetry (cumsum reassociation, lesson #13 class)."""
    rest = State.rest(mesh)
    zbar3, Z3 = ale.live_geometry(mesh, rest.hnode)
    zbar = np.asarray(mesh.zbar); Znom = np.asarray(mesh.Z)
    imask = np.asarray(mesh.node_iface_mask); lmask = np.asarray(mesh.node_layer_mask)
    dz = np.abs(np.asarray(zbar3) - zbar[None, :])
    dZ = np.abs(np.asarray(Z3)[:, :-1] - Znom[None, :])
    assert dz[imask].max() <= 1e-9, f"zbar_3d_n wet |Δ|={dz[imask].max():.2e}"
    assert dZ[lmask[:, :-1]].max() <= 1e-9, f"Z_3d_n wet |Δ|={dZ[lmask[:, :-1]].max():.2e}"
    # pi is exactly bitwise (round depths) — assert the stronger property here
    assert dz[imask].max() == 0.0 and dZ[lmask[:, :-1]].max() == 0.0


def test_live_geometry_ad_safe_positive_spacing(mesh):
    """No zero/negative interface spacing anywhere (the D5 AD rule: nominal `zbar` fills
    the below-bottom lanes, NOT the C's 0-padding ⇒ no dense-JAX inf factory)."""
    rest = State.rest(mesh)
    zbar3, _ = ale.live_geometry(mesh, rest.hnode)
    dz = np.asarray(zbar3)[:, :-1] - np.asarray(zbar3)[:, 1:]
    assert dz.min() > 0.0, f"min interface spacing {dz.min()} ≤ 0 ⇒ divide trap"


@pytest.mark.parametrize("amp", [0.0, 0.5, 2.0])
def test_init_thickness_matches_numpy_reference(mesh, amp):
    """`init_thickness_zstar` matches an independent numpy loop for a non-trivial hbar
    (stretch layers scale by (1+hbar/dd), bottom-intersecting + bottom stay nominal)."""
    rng = np.random.default_rng(0)
    hbar = jnp.asarray(rng.uniform(-amp, amp, size=mesh.nod2D))
    hnode, _, _, _ = ale.init_thickness_zstar(mesh, hbar, jnp.zeros_like(hbar))
    ref = _ref_init_hnode(mesh, hbar)
    assert np.max(np.abs(np.asarray(hnode) - ref)) < 1e-12


def test_live_geometry_matches_numpy_reference(mesh):
    """`live_geometry` matches the C bottom→top recurrence for a stretched hnode (the
    reverse-cumsum == the recurrence association)."""
    rng = np.random.default_rng(1)
    hbar = jnp.asarray(rng.uniform(-1.0, 1.0, size=mesh.nod2D))
    hnode, _, _, _ = ale.init_thickness_zstar(mesh, hbar, jnp.zeros_like(hbar))
    zbar3, Z3 = ale.live_geometry(mesh, hnode)
    ref_zb, ref_Z = _ref_live_geometry(mesh, hnode)
    imask = np.asarray(mesh.node_iface_mask); lmask = np.asarray(mesh.node_layer_mask)
    assert np.abs(np.asarray(zbar3) - ref_zb)[imask].max() < 1e-9
    assert np.abs(np.asarray(Z3)[:, :-1] - ref_Z[:, :-1])[lmask[:, :-1]].max() < 1e-9


def test_eta_n_init_reversed_weights(mesh):
    """init eta_n uses the REVERSED AB weights `α·hbar_old + (1−α)·hbar` (lesson #7), the
    mirror of the per-step blend `α·hbar + (1−α)·hbar_old`."""
    rng = np.random.default_rng(2)
    hbar = jnp.asarray(rng.uniform(-1, 1, mesh.nod2D))
    hbar_old = jnp.asarray(rng.uniform(-1, 1, mesh.nod2D))
    _, _, eta_n, _ = ale.init_thickness_zstar(mesh, hbar, hbar_old, alpha=0.3)
    want = 0.3 * np.asarray(hbar_old) + 0.7 * np.asarray(hbar)
    assert np.max(np.abs(np.asarray(eta_n) - want)) < 1e-15


def test_live_geometry_gradient_finite(mesh):
    """d(zbar_3d_n)/d(hbar) through the init→live_geometry chain is finite on every lane
    (the new state path the GATE-9a `d/d(hbar-IC)` gate exercises)."""
    imask = jnp.asarray(mesh.node_iface_mask)

    def loss(hbar):
        hn, _, _, _ = ale.init_thickness_zstar(mesh, hbar, jnp.zeros_like(hbar))
        zb, _ = ale.live_geometry(mesh, hn)
        return jnp.sum(jnp.where(imask, zb, 0.0) ** 2)

    g = jax.grad(loss)(jnp.full((mesh.nod2D,), 0.1))
    assert bool(np.all(np.isfinite(np.asarray(g))))


# ==========================================================================
# 5. JZ.2 — forcing flip: rsf producer + evap split + water-flux balancing
# ==========================================================================
def _ice_inputs(n=6):
    """Synthetic Arctic ice column inputs for therm_ice_cell (ice + snow, cold air ⇒ growth)."""
    from fesom_jax.ice import IceConfig
    cfg = IceConfig(ice_dt=1800.0)
    f = lambda v: jnp.full((n,), float(v))
    # therm_ice_cell positional order
    args = [f(1.5), f(0.3), f(0.9), f(0.0), f(150.0), f(-20.0), f(1e-3), f(0.0), f(1e-8),
            f(0.0), f(33.0), f(8.0), f(0.01), f(-1.7), f(33.0), f(1.3e-3), f(1.3e-3),
            f(-15.0), f(cfg.h0)]
    return cfg, args


def test_thermo_split_and_rsf_producer():
    """zstar real-salt path (``use_virt_salt=False``): ``evaporation`` is the BUNDLED
    ``evap_ow·(1−A) + subli·A`` (== ``evap``, the C's ``*evap = _evap + _subli`` — the
    5a61bb0 budget fix; the pre-fix open-water-only ``evaporation`` was the salinity-drift
    leak) and ``evaporation − ice_sublimation`` recovers the open-water part the balance
    needs; ``rsf = fwice·Sice − iflice·ρice/ρwat·Sice``; the virtual-salt path keeps
    ``rsf=0`` and ``thdgr``/``evap`` are path-independent."""
    from fesom_jax import ice_thermo as it
    cfg, args = _ice_inputs()
    ov = jax.vmap(lambda *x: it.therm_ice_cell(cfg, *x, use_virt_salt=True))(*args)
    orl = jax.vmap(lambda *x: it.therm_ice_cell(cfg, *x, use_virt_salt=False))(*args)
    # post-5a61bb0 semantics on both paths: evaporation ≡ evap (bundled, bit-identical),
    # ice_sublimation = subli·A nonzero on these icy sublimating columns (so the
    # balance's evaporation − ice_sublimation genuinely subtracts something)
    for o in (ov, orl):
        assert float(jnp.max(jnp.abs(o.evaporation - o.evap))) == 0.0
        assert float(jnp.min(jnp.abs(o.ice_sublimation))) > 0.0
    # rsf: 0 under virtual; the exact producer formula under real
    assert float(jnp.max(jnp.abs(ov.rsf))) == 0.0
    fwice = -orl.thdgr * cfg.rhoice / cfg.rhowat
    rsf_expect = fwice * cfg.Sice - orl.iflice * cfg.rhoice / cfg.rhowat * cfg.Sice
    assert float(jnp.max(jnp.abs(orl.rsf - rsf_expect))) == 0.0
    assert bool(jnp.all(jnp.isfinite(orl.rsf))) and float(jnp.max(jnp.abs(orl.rsf))) > 0.0
    # growth + bundled evaporation are independent of the salt path
    assert float(jnp.max(jnp.abs(ov.thdgr - orl.thdgr))) == 0.0
    assert float(jnp.max(jnp.abs(ov.evap - orl.evap))) == 0.0


def test_thermo_default_path_byte_identical():
    """``use_virt_salt`` defaults to True ⇒ the new code path is byte-identical to the
    explicit virtual-salt call (the linfs invariant at the kernel level)."""
    from fesom_jax import ice_thermo as it
    cfg, args = _ice_inputs()
    od = jax.vmap(lambda *x: it.therm_ice_cell(cfg, *x))(*args)               # default
    ov = jax.vmap(lambda *x: it.therm_ice_cell(cfg, *x, use_virt_salt=True))(*args)
    for fld in ("h", "hsn", "A", "t", "fw", "ehf", "thdgr", "evap", "iflice"):
        assert float(jnp.max(jnp.abs(getattr(od, fld) - getattr(ov, fld)))) == 0.0, fld


def test_fresh_water_balance_zstar_removes_global_mean():
    """``fresh_water_balance_zstar`` adds the area-weighted global mean of the freshwater
    ``flux`` to ``water_flux`` (``fesom_ice_coupling.c:193-216``), so the post-balance global
    integral of the per-node increment equals the original mean (volume conservation)."""
    from fesom_jax import ice_coupling
    from fesom_jax.ice import IceConfig
    cfg = IceConfig()
    rng = np.random.default_rng(3)
    n = 200
    g = lambda: jnp.asarray(rng.uniform(-1e-6, 1e-6, n))
    wf0, evap, subl, pr, ps, a0, ro, thdgr, thdgrsn = (g() for _ in range(9))
    areas = jnp.asarray(rng.uniform(1e8, 1e10, n))
    oa = float(jnp.sum(areas))
    wf1 = ice_coupling.fresh_water_balance_zstar(
        wf0, evap, subl, pr, ps, a0, ro, thdgr, thdgrsn, areas, oa, cfg)
    inv = 1.0 / cfg.rhowat
    flux = (evap - subl + pr + ps * (1.0 - a0) + ro
            - thdgr * cfg.rhoice * inv - thdgrsn * cfg.rhosno * inv)
    net = float(jnp.sum(flux * areas) / oa)
    # every node got the SAME increment = net
    assert float(jnp.max(jnp.abs((wf1 - wf0) - net))) < 1e-18
    assert abs(net) > 0.0


def test_bc_S_zstar_drops_virtual_salt_keeps_rsf():
    """The bc_S flip (``fesom_tracer_diff.c:65``): linfs ⇒ ``dt·(virtual_salt+relax_salt)``;
    zstar ⇒ ``dt·(relax_salt + real_salt_flux)`` with virtual_salt≡0 — +dt, NO sval·wf
    (sign-trap lesson #3). Checked at the IceOceFluxes level."""
    from fesom_jax import ice_coupling
    rng = np.random.default_rng(4)
    n = 50
    g = lambda: jnp.asarray(rng.uniform(-1e-4, 1e-4, n))
    S_top, flx_fw, flx_h, Ssurf, runoff = g(), g(), g(), jnp.asarray(35.0 + rng.normal(size=n)), g()
    areas = jnp.ones(n); oa = float(n); rsf = g()
    virt = ice_coupling.ice_oce_fluxes(S_top, flx_fw, flx_h, Ssurf, runoff, areas, oa,
                                       use_virt_salt=True)
    zst = ice_coupling.ice_oce_fluxes(S_top, flx_fw, flx_h, Ssurf, runoff, areas, oa,
                                      use_virt_salt=False, real_salt_flux=rsf)
    assert float(jnp.max(jnp.abs(virt.real_salt_flux))) == 0.0   # linfs ⇒ rsf carried as 0
    assert float(jnp.max(jnp.abs(zst.virtual_salt))) == 0.0      # zstar ⇒ virtual_salt ≡ 0
    assert float(jnp.max(jnp.abs(zst.real_salt_flux - rsf))) == 0.0
    # relax_salt is identical (the SSS restoring is path-independent)
    assert float(jnp.max(jnp.abs(virt.relax_salt - zst.relax_salt))) == 0.0


# ==========================================================================
# 6. JZ.2 — forcing dump gate vs the C z2_cdump `forcing` tag (CORE2, compute-node)
# ==========================================================================
CORE2_MESH = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
# dist_16 IC: the z2_cdump oracle ran on 16 ranks and the C IC is PARTITION-DEPENDENT
# (order-dependent GS land fill) — the serial cache (data/ic_core2, for the 1-rank legacy
# oracles) differs at GS-filled nodes. Built by scripts/tools/rebuild_ic_dist16.py.
CORE2_IC = Path(__file__).resolve().parents[2] / "data" / "ic_core2_dist16"
DT_ZSTAR = 1800.0
ZSTAR_YEAR = 1958

core2_forcing_missing = pytest.mark.skipif(
    not (CORE2_MESH.exists() and (CORE2_IC / "T_ic.npy").exists() and ZSTAR_ORACLE.is_dir()),
    reason="CORE2 mesh / PHC IC / z2_cdump forcing oracle missing (compute-node gate)")


@core2_forcing_missing
def test_forcing_flip_linfs_vs_zstar(capsys):
    """JZ.2 gate (config-INDEPENDENT): the linfs→zstar forcing FLIP on CORE2 (126858 nodes,
    dt=1800). Running the SAME ice step with ``use_virt_salt`` True vs False isolates exactly
    what zstar changes — independent of whether the JAX harness reproduces the z2_cdump's exact
    inputs:

    * ``relax_salt`` is **bit-identical** (the SSS restoring is path-independent — proves the
      zstar code does not corrupt it);
    * ``virtual_salt`` is nonzero under linfs and **≡ 0** under zstar;
    * ``real_salt_flux`` is **≡ 0** under linfs and **nonzero** under zstar (the live producer);
    * ``water_flux`` differs only by the global balancing increment (a near-uniform offset).

    Combined with the existing linfs forcing dump gates (which validate the linfs forcing vs the
    C at dt=500), this transitively validates the zstar forcing. The DIRECT z2_cdump comparison
    is :func:`test_forcing_dump_gate_zstar_diagnostic` (config-matching is a follow-on)."""
    from fesom_jax import core2_forcing, ice, ice_step
    from fesom_jax.ice import IceConfig
    from fesom_jax.phc_ic import core2_initial_state
    mesh = load_mesh(CORE2_MESH)
    state = core2_initial_state(mesh, CORE2_IC)
    sst = np.asarray(state.T[:, 0])
    state0 = ice.seed_ice(state, mesh, sst)
    cf = core2_forcing.build_core_forcing(mesh, ZSTAR_YEAR, sst_ic=sst)
    fs = cf.static
    sf0 = cf.step_forcing(*core2_forcing.dates_for_steps(ZSTAR_YEAR, DT_ZSTAR, 1)[0])
    lin = ice_step.ice_surface_step(IceConfig(), mesh, state0, sf0, fs, dt=DT_ZSTAR,
                                    use_virt_salt=True)
    zst = ice_step.ice_surface_step(IceConfig(), mesh, state0, sf0, fs, dt=DT_ZSTAR,
                                    use_virt_salt=False)
    relax_dmax = float(np.max(np.abs(np.asarray(lin.relax_salt) - np.asarray(zst.relax_salt))))
    with capsys.disabled():
        print(f"\n[forcing-flip] relax_salt path |Δ|={relax_dmax:.2e} (want 0)  "
              f"virt linfs max|.|={float(np.abs(lin.virtual_salt).max()):.2e}→zstar "
              f"{float(np.abs(zst.virtual_salt).max()):.2e}  "
              f"rsf linfs={float(np.abs(lin.real_salt_flux).max()):.2e}→zstar "
              f"{float(np.abs(zst.real_salt_flux).max()):.2e}")
    assert relax_dmax == 0.0                                      # path-independent
    assert float(np.abs(lin.virtual_salt).max()) > 0.0           # linfs ⇒ virtual salt active
    assert float(np.abs(zst.virtual_salt).max()) == 0.0          # zstar ⇒ virtual_salt ≡ 0
    assert float(np.abs(lin.real_salt_flux).max()) == 0.0        # linfs ⇒ no rsf
    assert float(np.abs(zst.real_salt_flux).max()) > 0.0         # zstar ⇒ live rsf producer


@core2_forcing_missing
def test_forcing_dump_gate_zstar_diagnostic(capsys):
    """HARD GATE (since the dist_16 IC rebuild, 2026-06-12): the JAX zstar forcing vs the C
    ``z2_cdump`` ``forcing`` tag at step 1. The former ~1e-5 "config/input mismatch" was the
    partition-dependent IC all along (SST/S_top feed evap, ice growth, and relax):
    ``virtual_salt≡0`` exact, ``relax_salt`` max=3.4e-18 (bit class), ``real_salt_flux``
    max=1.6e-10, ``water_flux`` max=1.4e-9 (bulk-formula ulp accumulation)."""
    from fesom_jax import core2_forcing, ice, ice_step
    from fesom_jax.ice import IceConfig
    from fesom_jax.phc_ic import core2_initial_state
    mesh = load_mesh(CORE2_MESH)
    state = core2_initial_state(mesh, CORE2_IC)
    sst = np.asarray(state.T[:, 0])
    state0 = ice.seed_ice(state, mesh, sst)
    cf = core2_forcing.build_core_forcing(mesh, ZSTAR_YEAR, sst_ic=sst)
    sf0 = cf.step_forcing(*core2_forcing.dates_for_steps(ZSTAR_YEAR, DT_ZSTAR, 1)[0])
    out = ice_step.ice_surface_step(IceConfig(), mesh, state0, sf0, cf.static,
                                    dt=DT_ZSTAR, use_virt_salt=False)
    f, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["forcing"], step=1, n_nod=NG5_NOD2D)
    jx = {"water_flux": np.asarray(out.water_flux), "virtual_salt": np.asarray(out.virtual_salt),
          "relax_salt": np.asarray(out.relax_salt), "real_salt_flux": np.asarray(out.real_salt_flux)}
    with capsys.disabled():
        print("\n[forcing-diag] JAX zstar forcing vs z2_cdump (step 1):")
        for name, j in jx.items():
            c = io_dump.ale_component(f, "forcing", name)
            d = np.abs(j - c)
            print(f"   {name:16s} max|Δ|={d.max():.3e}  rel={d.max()/max(np.abs(c).max(),1e-30):.3e}")
    assert np.abs(jx["virtual_salt"]).max() == 0.0               # zstar virtual_salt≡0 (exact)
    for name, tol in (("water_flux", 1e-8), ("relax_salt", 1e-16), ("real_salt_flux", 1e-9)):
        c = io_dump.ale_component(f, "forcing", name)
        d = np.abs(jx[name] - c).max()
        assert d < tol, f"{name} max|Δ|={d:.3e} (≥{tol:.0e} — stale IC cache or forcing regression)"


# ==========================================================================
# 7. JZ.3 — SSH plumbing: wf tail, hbar wf term, D2 stiffness increment
#    (config-INDEPENDENT kernel algebra on the pi mesh; validate via flip/degeneracy)
# ==========================================================================
def test_ssh_rhs_zstar_wf_tail_is_exact(pi_model):
    """The zstar ssh_rhs tail (``fesom_ssh.c:413-421``): passing ``water_flux`` adds
    EXACTLY ``−α·wf·areasvol[:,0]`` on non-cavity nodes — a config-independent algebraic
    check. Probed at ``uv=0`` so the tail IS the whole field (the C ssh_rhs base is a
    near-cancelling ~1e6 scatter — differencing two large bases would lose ~8 digits to
    catastrophic cancellation; see the ssh/rhs lesson)."""
    from fesom_jax.config import SSH_ALPHA
    mesh, op, stress, st0 = pi_model
    rng = np.random.default_rng(30)
    wf = jnp.asarray(rng.normal(size=mesh.nod2D) * 1e-6)
    nocav = np.asarray(mesh.ulevels_nod2D) == 1
    want = np.where(nocav, -SSH_ALPHA * np.asarray(wf) * np.asarray(mesh.areasvol[:, 0]), 0.0)
    uv0 = jnp.zeros(st0.uv.shape)
    zst0 = ssh.compute_ssh_rhs(mesh, uv0, uv0, st0.helem, water_flux=wf)
    assert np.max(np.abs(np.asarray(zst0) - want)) < 1e-18
    # additive + gated: with a real (uv≠0) base, water_flux=None is byte-identical to linfs.
    uv = jnp.asarray(rng.normal(size=st0.uv.shape) * 0.1)
    base = ssh.compute_ssh_rhs(mesh, uv, jnp.zeros_like(uv), st0.helem)
    assert np.array_equal(np.asarray(base), np.asarray(
        ssh.compute_ssh_rhs(mesh, uv, jnp.zeros_like(uv), st0.helem, water_flux=None)))


def test_compute_hbar_zstar_wf_term_and_ordering(pi_model):
    """``compute_hbar`` zstar term (``fesom_momentum.c:839-846``): ``ssh_rhs_old −=
    wf·areasvol[:,0]`` on non-cavity, and the hbar update consumes the wf-MODIFIED
    ssh_rhs_old (the ordering trap). Probed at ``uv=0`` (base ssh_rhs_old ≡ 0) so the wf
    term is isolated from the near-cancelling transport-divergence base."""
    mesh, op, stress, st0 = pi_model
    rng = np.random.default_rng(31)
    wf = jnp.asarray(rng.normal(size=mesh.nod2D) * 1e-6)
    hbar0 = jnp.asarray(rng.normal(size=mesh.nod2D) * 0.05)
    nocav = np.asarray(mesh.ulevels_nod2D) == 1
    area0 = np.asarray(mesh.areasvol[:, 0])
    want = np.where(nocav, -np.asarray(wf) * area0, 0.0)
    uv0 = jnp.zeros(st0.uv.shape)
    sro, hbar_z = ssh.compute_hbar(mesh, uv0, st0.helem, hbar0, dt=DT, water_flux=wf)
    assert np.max(np.abs(np.asarray(sro) - want)) < 1e-18
    # hbar consumes the wf-modified ssh_rhs_old (the ordering): hbar == hbar0 + want·dt/area
    want_hbar = np.asarray(hbar0) + np.where(nocav, want * DT / area0, 0.0)
    assert np.max(np.abs(np.asarray(hbar_z) - want_hbar)) < 1e-15
    # water_flux=None ⇒ byte-identical to linfs compute_hbar (with a real uv≠0 base)
    uv = jnp.asarray(rng.normal(size=st0.uv.shape) * 0.1)
    sro_lin, hbar_lin = ssh.compute_hbar(mesh, uv, st0.helem, hbar0, dt=DT)
    sro_none, hbar_none = ssh.compute_hbar(mesh, uv, st0.helem, hbar0, dt=DT, water_flux=None)
    assert np.array_equal(np.asarray(sro_lin), np.asarray(sro_none))
    assert np.array_equal(np.asarray(hbar_lin), np.asarray(hbar_none))


def test_stiff_increment_zero_at_coldstart(pi_model):
    """Cold-start degeneracy (the free Z-gate): at ``hbar=0`` the D2 increment is exactly
    0 for any iterate, so the step-1 zstar solve is the linfs solve BIT-FOR-BIT."""
    mesh, op, stress, st0 = pi_model
    rng = np.random.default_rng(32)
    x = jnp.asarray(rng.normal(size=mesh.nod2D))
    z = jnp.zeros(mesh.nod2D)
    dA = ssh.stiff_increment_matvec(mesh, z, x, dt=DT)
    assert float(jnp.max(jnp.abs(dA))) == 0.0
    rhs = jnp.asarray(np.random.default_rng(33).normal(size=mesh.nod2D) * 1e3)
    d_lin = ssh.solve_ssh(op, rhs)                              # hbar=None (static op)
    d_z = ssh.solve_ssh(op, rhs, mesh=mesh, hbar=z, dt=DT)      # hbar=0 (increment ≡ 0)
    assert float(jnp.max(jnp.abs(d_lin - d_z))) == 0.0


def test_stiff_increment_is_symmetric(pi_model):
    """``ΔA`` is symmetric (a weighted Laplacian ``div(−mean₃(hbar)·grad)``), so
    ``custom_linear_solve(symmetric=True)`` is valid: ``⟨y, ΔA·x⟩ == ⟨x, ΔA·y⟩``."""
    mesh, op, stress, st0 = pi_model
    rng = np.random.default_rng(34)
    hbar = jnp.asarray(rng.normal(size=mesh.nod2D) * 0.05)
    x = jnp.asarray(rng.normal(size=mesh.nod2D))
    y = jnp.asarray(rng.normal(size=mesh.nod2D))
    lhs = float(jnp.dot(y, ssh.stiff_increment_matvec(mesh, hbar, x, dt=DT)))
    rhs = float(jnp.dot(x, ssh.stiff_increment_matvec(mesh, hbar, y, dt=DT)))
    assert abs(lhs - rhs) <= 1e-9 * max(abs(lhs), abs(rhs), 1e-30)


def test_solve_ssh_state_dependent_transpose_residual(pi_model):
    """With a state-dependent ``A(hbar≠0)`` the ``custom_linear_solve`` gradient is the
    TIGHT implicit-diff ``A⁻¹``: ``d⟨w,d_eta⟩/d(ssh_rhs) = A⁻¹·w`` (A symmetric), verified
    by the transpose residual ``‖A·g − w‖ ≈ 0`` — the increment is correctly in the
    transpose path — plus finiteness (the Task-5.8 pattern, zstar-ON)."""
    mesh, op, stress, st0 = pi_model
    rng = np.random.default_rng(35)
    hbar = jnp.asarray(rng.normal(size=mesh.nod2D) * 0.05)
    rhs = jnp.asarray(rng.normal(size=mesh.nod2D) * 1e3)
    w = jnp.asarray(rng.normal(size=mesh.nod2D))
    g = jax.grad(lambda b: jnp.dot(w, ssh.solve_ssh(op, b, mesh=mesh, hbar=hbar, dt=DT)))(rhs)
    assert bool(np.all(np.isfinite(np.asarray(g))))
    Ag = ssh.ssh_matvec(op, g) + ssh.stiff_increment_matvec(mesh, hbar, g, dt=DT)
    res = float(jnp.max(jnp.abs(Ag - w)))
    assert res < 1e-6, f"transpose residual {res:.2e}"


def test_solve_ssh_grad_through_hbar_finite(pi_model):
    """``d(d_eta)/d(hbar)`` through the D2 increment is finite & nonzero (the new GATE-9a
    state path ``d/d(hbar-IC)``): the operator ``A(hbar)`` closure propagates a clean
    implicit-diff cotangent into ``hbar``, not just the rhs."""
    mesh, op, stress, st0 = pi_model
    rhs = jnp.asarray(np.random.default_rng(36).normal(size=mesh.nod2D) * 1e3)
    g = jax.grad(lambda hbar: jnp.sum(ssh.solve_ssh(op, rhs, mesh=mesh, hbar=hbar, dt=DT) ** 2))(
        jnp.full((mesh.nod2D,), 0.05))
    assert bool(np.all(np.isfinite(np.asarray(g))))
    assert float(jnp.max(jnp.abs(g))) > 0.0


# ==========================================================================
# 8. JZ.3 — D2 foundation vs z2_cdump (config-INDEPENDENT C-internal: dhe == mean₃(Δhbar),
#    telescoping Σdhe == mean₃(hbar) — validates stiff_increment_matvec's increment depth)
# ==========================================================================
core2_dhe_missing = pytest.mark.skipif(
    not (CORE2_MESH.exists() and ZSTAR_ORACLE.is_dir()),
    reason="CORE2 mesh / z2_cdump oracle missing (login-node C-internal gate)")


@core2_dhe_missing
def test_dhe_recompute_matches_dump():
    """Config-INDEPENDENT: the C ``dhe`` tag == ``mean₃(hbar − hbar_old)`` recomputed from
    the C's OWN ``hbar`` tag (``fesom_step.c:216-227``) — validates the element-vertex mean
    the D2 increment depth is built on, with NO JAX forcing (sidesteps the config gap)."""
    mesh = load_mesh(CORE2_MESH)
    en = np.asarray(mesh.elem_nodes)
    ucav = np.asarray(mesh.ulevels) == 1
    for step in (1, 2, 3):                                       # step 1 trivially 0 (cold)
        fh, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["hbar"], step=step,
                                      n_nod=NG5_NOD2D, n_elem=NG5_ELEM2D)
        dch = io_dump.ale_component(fh, "hbar", "hbar") - io_dump.ale_component(fh, "hbar", "hbar_old")
        fd, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["dhe"], step=step,
                                      n_nod=NG5_NOD2D, n_elem=NG5_ELEM2D)
        dhe_re = np.where(ucav, (dch[en[:, 0]] + dch[en[:, 1]] + dch[en[:, 2]]) / 3.0, 0.0)
        d = np.abs(dhe_re - fd["dhe"][:, 0])
        assert d.max() < 1e-12, f"step {step}: dhe recompute |Δ|={d.max():.2e}"


@core2_dhe_missing
def test_dhe_telescoping_equals_mean3_hbar():
    """The D2 telescoping identity (the foundation of ``stiff_increment_matvec``): at cold
    start ``Σ_{s=1..3} dhe_s == mean₃(hbar_3)`` — so the live matrix increment ``−mean₃(hbar)``
    is recomputable from the carried ``hbar`` alone, no cumulative CSR. Config-independent."""
    mesh = load_mesh(CORE2_MESH)
    en = np.asarray(mesh.elem_nodes); ucav = np.asarray(mesh.ulevels) == 1
    dhe_sum = np.zeros(NG5_ELEM2D)
    for step in (1, 2, 3):
        fd, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["dhe"], step=step,
                                      n_nod=NG5_NOD2D, n_elem=NG5_ELEM2D)
        dhe_sum = dhe_sum + fd["dhe"][:, 0]
    fh, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["hbar"], step=3, n_nod=NG5_NOD2D, n_elem=NG5_ELEM2D)
    h3 = io_dump.ale_component(fh, "hbar", "hbar")
    m3 = np.where(ucav, (h3[en[:, 0]] + h3[en[:, 1]] + h3[en[:, 2]]) / 3.0, 0.0)
    assert np.abs(dhe_sum - m3).max() < 1e-9


# ==========================================================================
# 9. JZ.4 — vert_vel_zstar_distribute (config-INDEPENDENT kernel algebra on pi)
# ==========================================================================
def _ref_vert_vel_zstar(mesh, w, hnode, zbar3, hbar, hbar_old, wf, dt):
    """Independent numpy loop reference for vert_vel_zstar_distribute (the C loop,
    ``fesom_ale.c:169-201``)."""
    w = np.asarray(w).copy(); hnode_new = np.asarray(hnode).copy()
    zbar3 = np.asarray(zbar3); hbar = np.asarray(hbar); hbar_old = np.asarray(hbar_old); wf = np.asarray(wf)
    min_f = np.asarray(mesh.nlevels_nod2D_min); nocav = np.asarray(mesh.ulevels_nod2D) == 1
    for n in range(mesh.nod2D):
        if not nocav[n]:
            continue
        nzmax_f = min_f[n] - 1                                # 1-based
        dd1 = zbar3[n, nzmax_f - 1]
        col = zbar3[n, 0] - dd1
        dd = (hbar[n] - hbar_old[n]) / col if col != 0.0 else 0.0
        dddt = dd / dt
        for nz in range(0, nzmax_f - 1):                     # 0-based, nz < min_f-2
            w[n, nz] -= (zbar3[n, nz] - dd1) * dddt
            hnode_new[n, nz] = hnode[n, nz] + (zbar3[n, nz] - zbar3[n, nz + 1]) * dd
        w[n, 0] -= wf[n]                                     # surface BC
    return w, hnode_new


def test_vert_vel_zstar_distribute_matches_numpy_ref(mesh):
    """vert_vel_zstar_distribute matches the independent numpy loop (golden rule): the
    vertically-integrated Wvel correction + stretched hnode_new + surface wf BC."""
    rng = np.random.default_rng(40)
    nl = mesh.zbar.shape[0]
    rest = State.rest(mesh)
    zbar3, _ = ale.live_geometry(mesh, rest.hnode)           # carried (pre-commit) geometry
    hbar = jnp.asarray(rng.uniform(-1.0, 1.0, mesh.nod2D))
    hbar_old = jnp.asarray(rng.uniform(-1.0, 1.0, mesh.nod2D))
    w0 = jnp.asarray(rng.normal(size=(mesh.nod2D, nl)) * 1e-6)
    wf = jnp.asarray(rng.normal(size=mesh.nod2D) * 1e-6)
    w, hnode_new = ale.vert_vel_zstar_distribute(mesh, w0, rest.hnode, zbar3, hbar, hbar_old, wf, dt=DT_ZSTAR)
    ref_w, ref_h = _ref_vert_vel_zstar(mesh, w0, rest.hnode, zbar3, hbar, hbar_old, wf, DT_ZSTAR)
    assert np.max(np.abs(np.asarray(w) - ref_w)) < 1e-15
    assert np.max(np.abs(np.asarray(hnode_new) - ref_h)) < 1e-15


def test_vert_vel_zstar_coldstart_noop(mesh):
    """At ``hbar=hbar_old`` (no SSH change) ``dd=0`` ⇒ w unchanged (bar the wf BC) and
    ``hnode_new == hnode`` — the degeneracy gate."""
    rest = State.rest(mesh)
    zbar3, _ = ale.live_geometry(mesh, rest.hnode)
    rng = np.random.default_rng(41)
    hbar = jnp.asarray(rng.uniform(-1, 1, mesh.nod2D))
    w0 = jnp.asarray(rng.normal(size=(mesh.nod2D, mesh.zbar.shape[0])) * 1e-6)
    w, hnode_new = ale.vert_vel_zstar_distribute(
        mesh, w0, rest.hnode, zbar3, hbar, hbar, jnp.zeros(mesh.nod2D), dt=DT_ZSTAR)
    assert float(jnp.max(jnp.abs(w - w0))) == 0.0
    assert np.array_equal(np.asarray(hnode_new), np.asarray(rest.hnode))


def test_vert_vel_zstar_surface_wf_bc(mesh):
    """The surface continuity BC subtracts ``water_flux`` at ``w[:,0]`` (non-cavity); at
    ``hbar=0`` it is the ONLY change (the stretch correction is 0)."""
    rest = State.rest(mesh)
    zbar3, _ = ale.live_geometry(mesh, rest.hnode)
    rng = np.random.default_rng(42)
    wf = jnp.asarray(rng.normal(size=mesh.nod2D) * 1e-6)
    z = jnp.zeros(mesh.nod2D)
    w0 = jnp.zeros((mesh.nod2D, mesh.zbar.shape[0]))
    w, _ = ale.vert_vel_zstar_distribute(mesh, w0, rest.hnode, zbar3, z, z, wf, dt=DT_ZSTAR)
    nocav = np.asarray(mesh.ulevels_nod2D) == 1
    assert np.max(np.abs(np.asarray(w[:, 0]) - np.where(nocav, -np.asarray(wf), 0.0))) < 1e-18
    assert float(jnp.max(jnp.abs(w[:, 1:]))) == 0.0         # only the surface interface moves


def test_vert_vel_zstar_grad_finite(mesh):
    """``d(w, hnode_new)/d(hbar)`` is finite (the new state path; AD-safe stretchable-depth
    divide)."""
    rest = State.rest(mesh)
    zbar3, _ = ale.live_geometry(mesh, rest.hnode)
    w0 = jnp.zeros((mesh.nod2D, mesh.zbar.shape[0])); z = jnp.zeros(mesh.nod2D)

    def loss(hbar):
        w, hn = ale.vert_vel_zstar_distribute(mesh, w0, rest.hnode, zbar3, hbar, z, z, dt=DT_ZSTAR)
        return jnp.sum(w ** 2) + jnp.sum(hn ** 2)

    g = jax.grad(loss)(jnp.full((mesh.nod2D,), 0.1))
    assert bool(np.all(np.isfinite(np.asarray(g))))


# ==========================================================================
# 10. JZ.3+JZ.4 — assembled pi zstar step SMOKE (no oracle; finiteness + traces)
# ==========================================================================
def test_pi_zstar_step_runs_finite(pi_model):
    """A pi zstar step (``ale_cfg=AleConfig()``, no CORE2 forcing ⇒ ``water_flux=None``,
    virtual-salt off) assembles & traces the JZ.3 (ssh increment) + JZ.4 (vert_vel
    distribute, hnode_new override) branches and runs FINITE, no NaN. There is no pi-zstar
    oracle (the C z2_cdump is CORE2 — the assembled dump gate is JZ.7); this is the
    no-crash/finiteness gate that the zstar branches wire together correctly."""
    mesh, op, stress, st0 = pi_model
    out = stepmod.step(st0, mesh, op, stress, dt=DT, is_first_step=True, ale_cfg=AleConfig())
    for fld in ("T", "S", "uv", "hbar", "hbar_old", "hnode", "hnode_new", "eta_n", "d_eta",
                "w", "ssh_rhs", "ssh_rhs_old"):
        assert np.all(np.isfinite(np.asarray(getattr(out, fld)))), f"{fld} non-finite"
    # the wind drives hbar≠0 ⇒ the JZ.4 distribute genuinely stretched hnode_new off st.hnode
    assert float(np.max(np.abs(np.asarray(out.hbar)))) > 0.0
    assert float(np.max(np.abs(np.asarray(out.hnode_new) - np.asarray(st0.hnode)))) > 0.0
    # a 2nd zstar step (warm hbar ⇒ the D2 stiffness increment is now live) also stays finite
    out2 = stepmod.step(out, mesh, op, stress, dt=DT, is_first_step=False, ale_cfg=AleConfig())
    assert np.all(np.isfinite(np.asarray(out2.hbar))) and np.all(np.isfinite(np.asarray(out2.d_eta)))


# ==========================================================================
# 11. JZ.5 — Shchepetkin density-Jacobian PGF (config-INDEPENDENT vs numpy loop ref)
# ==========================================================================
def _ref_shchepetkin(mesh, rho, Z3, hel, g, rho0):
    """Independent numpy loop reference for the Shchepetkin PGF — a direct transcription of
    the C element loop (``fesom_eos.c:364-497``, 1-based stacks)."""
    nl = mesh.zbar.shape[0]
    zbar = np.asarray(mesh.zbar)
    nlev = np.asarray(mesh.nlevels); ulev = np.asarray(mesh.ulevels)
    nlev_n = np.asarray(mesh.nlevels_nod2D); ulev_n = np.asarray(mesh.ulevels_nod2D)
    EN = np.asarray(mesh.elem_nodes); GSA = np.asarray(mesh.gradient_sca)
    rho = np.asarray(rho); Z3 = np.asarray(Z3); hel = np.asarray(hel)
    ne = EN.shape[0]
    px = np.zeros((ne, nl)); py = np.zeros((ne, nl))

    def qz(Rm, R0, Rp, Zm, Z0, Zp, Zn):                  # quadratic-Newton drho_dz
        dx10 = Z0 - Zm; dx21 = Zp - Z0; dx20 = Zp - Zm
        df10 = R0 - Rm; df21 = Rp - R0
        return df10 / dx10 + (dx10 * df21 - dx21 * df10) / (dx20 * dx21 * dx10) * ((Zn - Z0) + (Zn - Zm))

    for e in range(ne):
        nle = nlev[e] - 1; ule = ulev[e]                 # 1-based
        if nle < ule + 1:                                # skip degenerate nlevels<3 (C UB)
            continue
        en = EN[e]; gs = GSA[e]
        RHO = lambda ni, lvl: rho[en[ni], lvl - 1]
        Z3D = lambda ni, lvl: Z3[en[ni], lvl - 1]
        HEL = lambda lvl: hel[e, lvl - 1]
        # 1-based element stack zbar_n[1..nle+1], Z_n[1..nle]
        zbar_n = np.zeros(nle + 2); Z_n = np.zeros(nle + 2)
        zbar_n[nle + 1] = zbar[nlev[e] - 1]
        Z_n[nle] = zbar_n[nle + 1] + HEL(nle) * 0.5
        for nlz in range(nle, ule, -1):
            zbar_n[nlz] = zbar_n[nlz + 1] + HEL(nlz)
            Z_n[nlz - 1] = zbar_n[nlz] + HEL(nlz - 1) * 0.5
        zbar_n[ule] = zbar_n[ule + 1] + HEL(ule)
        idp = [0.0, 0.0]
        for nlz in range(ule, nle + 1):
            dz = [0.0, 0.0, 0.0]
            for ni in range(3):
                if nlz == ule and (nlz - ulev_n[en[ni]]) == 0:               # surface forward
                    dz[ni] = qz(RHO(ni, nlz), RHO(ni, nlz + 1), RHO(ni, nlz + 2),
                                Z3D(ni, nlz), Z3D(ni, nlz + 1), Z3D(ni, nlz + 2), Z_n[nlz])
                elif nlz == nle and (nlev_n[en[ni]] - 1 - nlz) == 0:         # bottom backward
                    dz[ni] = qz(RHO(ni, nlz - 2), RHO(ni, nlz - 1), RHO(ni, nlz),
                                Z3D(ni, nlz - 2), Z3D(ni, nlz - 1), Z3D(ni, nlz), Z_n[nlz])
                else:                                                         # centered
                    dz[ni] = qz(RHO(ni, nlz - 1), RHO(ni, nlz), RHO(ni, nlz + 1),
                                Z3D(ni, nlz - 1), Z3D(ni, nlz), Z3D(ni, nlz + 1), Z_n[nlz])
            mdz = (dz[0] + dz[1] + dz[2]) / 3.0
            for comp, (g0, g1, g2, out) in enumerate(
                    [(gs[0], gs[1], gs[2], px), (gs[3], gs[4], gs[5], py)]):
                drho = g0 * RHO(0, nlz) + g1 * RHO(1, nlz) + g2 * RHO(2, nlz)
                dzc = g0 * Z3D(0, nlz) + g1 * Z3D(1, nlz) + g2 * Z3D(2, nlz)
                aux = (drho - mdz * dzc) * HEL(nlz) * g / rho0
                out[e, nlz - 1] = idp[comp] + aux * 0.5
                idp[comp] += aux
    return px, py


def test_shchepetkin_matches_numpy_ref(mesh):
    """The vectorized Shchepetkin PGF matches the independent numpy loop (golden rule) on a
    stratified density — the surface/interior/bottom stencils + the vertical integral. Gated
    on nlevels≥3 elements (the degenerate single-mid-layer case is C UB; masked here)."""
    from fesom_jax import pgf
    from fesom_jax.config import DENSITY_0, G
    rest = State.rest(mesh)
    Z3, _ = ale.live_geometry(mesh, rest.hnode)          # == static Z on pi (nominal)
    Z3 = np.asarray(Z3)
    rng = np.random.default_rng(50)
    # stratified ρ−ρ0: denser deeper (smooth in the live mid-depth Z3) + small horizontal var
    horiz = rng.normal(size=(mesh.nod2D, 1)) * 0.05
    rho = (-0.02 * Z3 + 28.0 + horiz) * np.asarray(mesh.node_layer_mask)
    rho_j = jnp.asarray(rho); Z3_j = jnp.asarray(Z3)
    px, py = pgf.pressure_force_shchepetkin(mesh, rho_j, Z3_j, rest.helem)
    rx, ry = _ref_shchepetkin(mesh, rho, Z3, np.asarray(rest.helem), float(G), DENSITY_0)
    # compare only nlevels≥3 element layers (the ref skips degenerate elems → 0 there)
    big = np.asarray(mesh.nlevels) >= 3
    lm = np.asarray(mesh.elem_layer_mask) & big[:, None]
    dx = np.abs(np.asarray(px) - rx)[lm]; dy = np.abs(np.asarray(py) - ry)[lm]
    assert dx.max() < 1e-12, f"pgf_x |Δ|={dx.max():.2e}"
    assert dy.max() < 1e-12, f"pgf_y |Δ|={dy.max():.2e}"


def test_shchepetkin_grad_finite(mesh):
    """``d(pgf)/d(density)`` is finite (the safe denominators keep the overridden/below-bottom
    stencil lanes AD-clean) — the GATE-9a masked-NaN discipline for the new dense stencil."""
    from fesom_jax import pgf
    rest = State.rest(mesh)
    Z3, _ = ale.live_geometry(mesh, rest.hnode)
    lm = jnp.asarray(mesh.node_layer_mask)

    def loss(rho):
        px, py = pgf.pressure_force_shchepetkin(mesh, jnp.where(lm, rho, 0.0), Z3, rest.helem)
        return jnp.sum(px ** 2 + py ** 2)

    g = jax.grad(loss)(jnp.asarray(np.random.default_rng(51).normal(size=(mesh.nod2D, mesh.zbar.shape[0])) * 0.1))
    assert bool(np.all(np.isfinite(np.asarray(g))))


# ==========================================================================
# 12. JZ.7 — assembled CORE2 zstar step vs z2_cdump (config-clean subset, compute-node)
# ==========================================================================
def _relax_mismatch(mesh):
    """Per-node ``|Δrelax_salt|`` vs the C z2_cdump — the IC-mismatch proxy, sitting at a
    ~7.2e-9 reduction floor (the global-mean subtraction reassociation). Since the dist_16
    partition-faithful IC rebuild (2026-06-12) NO node sits above the floor (the 488
    brackish outliers were the serial-vs-16-rank GS fill-order difference); kept as a
    stale-IC-cache tripwire."""
    from fesom_jax import sss_runoff
    S_top = np.load(CORE2_IC / "S_ic.npy")[:, 0].astype(np.float64)
    sss = sss_runoff.build_reader(mesh)
    Ssurf = np.asarray(sss.month(1)).astype(np.float64)
    av = np.asarray(mesh.areasvol[:, 0]).astype(np.float64)
    oa = float(mesh.ocean_area)
    raw = sss_runoff.SURF_RELAX_S * (Ssurf - S_top)
    relax_jax = raw - np.sum(raw * av) / oa
    f, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["forcing"], step=1, n_nod=NG5_NOD2D)
    relax_c = io_dump.ale_component(f, "forcing", "relax_salt").astype(np.float64)
    return np.abs(relax_jax - relax_c)


def _config_clean_node_mask(mesh, thresh=1e-8):
    """Config-clean node mask: ``|Δrelax| ≤ thresh`` ⇒ the IC matches the C (at/near the
    reduction floor). With the dist_16 IC this is ALL-TRUE (asserted in the JZ.7 gate);
    a node dropping out means the IC cache went stale (rerun scripts/tools/rebuild_ic_dist16.py)."""
    return _relax_mismatch(mesh) <= thresh


@core2_forcing_missing
def test_jz7_assembled_zstar_step1(capsys):
    """JZ.7 (compute-node): the FULL assembled CORE2 zstar step — KPP + GM/Redi + EVP ice +
    zstar, the z2_cdump config, the first time all four config knobs run together. Gate:
    (a) the model assembles + runs FINITE (the integration smoke); (b) the step-1 dump tags
    match the C z2_cdump on the FULL mesh — pgf at the bit-identity class (max<1e-14; the
    dist_16 partition-faithful IC closed the former deep-IC tail), d_eta/hbar at the CG
    early-stop tolerance. At step 1 the geometry re-points are no-ops (live==static cold
    start), so this validates JZ.2-5 + the assembled order/threading end-to-end. Multi-step
    (steps 2-3) gates need JZ.6 complete (the live geometry re-points), deferred."""
    from fesom_jax import core2_forcing, ice
    from fesom_jax.gm import GMConfig
    from fesom_jax.kpp import KppConfig
    from fesom_jax.ice import IceConfig
    from fesom_jax.phc_ic import core2_initial_state
    mesh = load_mesh(CORE2_MESH)
    state = core2_initial_state(mesh, CORE2_IC)
    sst = np.asarray(state.T[:, 0])
    state0 = ice.seed_ice(state, mesh, sst)
    op = ssh.build_ssh_operator(mesh, dt=DT_ZSTAR)
    cf = core2_forcing.build_core_forcing(mesh, ZSTAR_YEAR, sst_ic=sst)
    sf0 = cf.step_forcing(*core2_forcing.dates_for_steps(ZSTAR_YEAR, DT_ZSTAR, 1)[0])
    st1 = stepmod.step(state0, mesh, op, None, dt=DT_ZSTAR, is_first_step=True,
                       step_forcing=sf0, forcing_static=cf.static,
                       kpp_cfg=KppConfig(), gm_cfg=GMConfig(), ice_cfg=IceConfig(),
                       ale_cfg=AleConfig())
    # (a) integration smoke — the full model assembles + every output field is finite.
    for fld in ("T", "S", "uv", "hbar", "hbar_old", "hnode", "hnode_new", "pgf_x", "pgf_y",
                "d_eta", "w", "eta_n", "ssh_rhs", "a_ice"):
        a = np.asarray(getattr(st1, fld))
        assert np.all(np.isfinite(a)), f"{fld} has non-finite (assembled zstar step)"

    # (b) step-1 dump gates. Since the dist_16 partition-faithful IC rebuild (2026-06-12) the
    # "config-clean subset" is the FULL mesh — assert it (an IC-cache-staleness tripwire), then
    # gate every tag on its tolerance class over the whole domain.
    clean_node = _config_clean_node_mask(mesh)
    assert clean_node.all(), (f"{int((~clean_node).sum())} nodes above the relax reduction floor "
                              "— stale IC cache? rerun scripts/tools/rebuild_ic_dist16.py")
    en = np.asarray(mesh.elem_nodes)
    clean_elem = clean_node[en[:, 0]] & clean_node[en[:, 1]] & clean_node[en[:, 2]]
    nim = np.asarray(mesh.node_iface_mask)
    with capsys.disabled():
        print(f"\n[jz7] assembled zstar step FINITE; config-clean nodes="
              f"{int(clean_node.sum())}/{mesh.nod2D} elems={int(clean_elem.sum())}/{mesh.elem2D}")

    def _rep(tag, jx, c, mask):
        d = np.abs(np.asarray(jx) - np.asarray(c))[mask]
        with capsys.disabled():
            print(f"[jz7] {tag:10s} |Δ| p50={np.percentile(d, 50):.2e} p99={np.percentile(d, 99):.2e}"
                  f" p99.9={np.percentile(d, 99.9):.2e} max={d.max():.2e}")
        return d

    # pgf (element layer) — bit-identical class, FULL mesh. The former ~3e-5 deep-IC tail was the
    # serial-vs-dist_16 GS fill-order difference in phc_ic (root-caused via scripts/debug/jz7_pgf_debug.py,
    # fixed 2026-06-12 by the partition-faithful extrap + bilinear association fix): observed
    # max=5.6e-16 (ulp class; the standalone density→pgf path is 2.7e-20 — the assembled step adds
    # the AB2 tracer-blend reassociation).
    fp, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["pgf_x", "pgf_y"], step=1,
                                  n_nod=NG5_NOD2D, n_elem=NG5_ELEM2D)
    lm47 = np.asarray(mesh.elem_layer_mask)[:, :NG5_NL - 1]
    for tag, jx in (("pgf_x", st1.pgf_x), ("pgf_y", st1.pgf_y)):
        d = np.abs(np.asarray(jx)[:, :NG5_NL - 1] - fp[tag])[lm47]
        with capsys.disabled():
            print(f"[jz7] {tag}  p50={np.percentile(d, 50):.2e} p99={np.percentile(d, 99):.2e} "
                  f"p99.9={np.percentile(d, 99.9):.2e} max={d.max():.2e}")
        assert np.percentile(d, 99) < 1e-15, f"{tag} p99={np.percentile(d, 99):.2e}"
        assert d.max() < 1e-14, f"{tag} max={d.max():.2e} (full-mesh bit-identity regressed)"

    # sshsolve + hbar (node scalar) + Wvel (node iface) — diagnostics (loose sanity gate; the
    # tight per-tag tolerances are calibrated from these prints, then asserted).
    fsh, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["sshsolve", "hbar"], step=1, n_nod=NG5_NOD2D)
    fwv, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["Wvel"], step=1, n_nod=NG5_NOD2D)
    # ssh_rhs is the near-cancelling transport divergence (Phase-2 ssh/rhs lesson: abs floor is the
    # upstream du amplified by dx·helem~1e7) — diagnostic only, no tight absolute gate.
    _rep("ssh_rhs", st1.ssh_rhs, io_dump.ale_component(fsh, "sshsolve", "ssh_rhs"), clean_node)
    # d_eta/hbar/eta_n: the C early-stops the CG at soltol=1e-5, so the dump is the early-stopped
    # iterate; JAX matches to that solve tolerance on the bulk, with the config-gap HALO (the 488
    # brackish nodes' RHS perturbation spread GLOBALLY by the elliptic solve) in the worst ~1 %.
    d_deta = _rep("d_eta", st1.d_eta, io_dump.ale_component(fsh, "sshsolve", "d_eta"), clean_node)
    d_hbar = _rep("hbar", st1.hbar, io_dump.ale_component(fsh, "hbar", "hbar"), clean_node)
    _rep("eta_n", st1.eta_n, io_dump.ale_component(fsh, "hbar", "eta_n"), clean_node)
    # Wvel: bit-faithful bulk (the ÷area crushes the scatter), robust tail.
    d_wvel = _rep("Wvel", st1.w, fwv["Wvel"], clean_node[:, None] & nim)
    # With the dist_16 IC the elliptic config-gap halo is gone: d_eta/hbar sit AT the CG
    # early-stop tolerance through max (observed max=5.5e-7), Wvel max=2.8e-8.
    assert np.percentile(d_hbar, 99) < 1e-6 and d_hbar.max() < 5e-6, \
        f"hbar p99={np.percentile(d_hbar, 99):.2e} max={d_hbar.max():.2e}"
    assert np.percentile(d_deta, 99) < 1e-6 and d_deta.max() < 5e-6, \
        f"d_eta p99={np.percentile(d_deta, 99):.2e} max={d_deta.max():.2e}"
    assert np.percentile(d_wvel, 50) < 1e-9 and d_wvel.max() < 1e-6, \
        f"Wvel p50={np.percentile(d_wvel, 50):.2e} max={d_wvel.max():.2e}"


@core2_dhe_missing
def test_jz7_ssh_solve_controlled_replay(capsys):
    """⚠️ The DECISIVE fidelity check: the SSH solve (incl. the zstar D2 stiffness increment) is
    **BYTE-IDENTICAL on CPU** with IDENTICAL inputs — proving the ~mm chained-multistep d_eta/hbar
    divergence (next test) is the upstream velocity/ssh_rhs FP-reassociation amplified by the
    near-cancelling SSH machinery (the trajectory butterfly EVERY ocean model has, incl. the C vs
    Fortran), NOT a solve/D2 bug.

    Controlled replay = feed the C's OWN dumped solve inputs (not the JAX chained state, which the
    dump can't fully reconstruct — no T/S/uv tags):
      * step 1: solve(C_ssh_rhs[1], x0=0,          hbar=0)         vs C_d_eta[1]   (D2 a cold no-op)
      * step 2: solve(C_ssh_rhs[2], x0=C_d_eta[1], hbar=C_hbar[1]) vs C_d_eta[2]   (D2 LIVE)
    Both match the C dump to **~1e-15** (the map/gather floor — NOT the 5.5e-7 the *assembled*
    step-1 gate shows, which is the JAX's OWN ssh_rhs differing from the C's). So with identical
    inputs the iterated near-null-space CG + the D2 closure reproduce the C bit-for-bit; the chained
    divergence is purely the input difference (the Phase-2 ssh/rhs `dx·helem~1e7` cancellation floor
    propagated). (`scripts/debug/jz7_ssh_replay_check.py`.)"""
    mesh = load_mesh(CORE2_MESH)
    op = ssh.build_ssh_operator(mesh, dt=DT_ZSTAR)
    nlm0 = np.asarray(mesh.node_layer_mask)[:, 0]
    z = jnp.zeros(mesh.nod2D)

    def load(step):
        fsh, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["sshsolve", "hbar"], step=step, n_nod=NG5_NOD2D)
        return (io_dump.ale_component(fsh, "sshsolve", "ssh_rhs").astype(np.float64),
                io_dump.ale_component(fsh, "sshsolve", "d_eta").astype(np.float64),
                io_dump.ale_component(fsh, "hbar", "hbar").astype(np.float64))

    rhs1, deta1, hbar1 = load(1)        # hbar1 = end-of-step-1 hbar = start-of-step-2 (D2 input)
    rhs2, deta2, _ = load(2)
    jx1 = np.asarray(ssh.solve_ssh(op, jnp.asarray(rhs1), x0=z, mesh=mesh, hbar=z, dt=DT_ZSTAR))
    jx2 = np.asarray(ssh.solve_ssh(op, jnp.asarray(rhs2), x0=jnp.asarray(deta1),
                                   mesh=mesh, hbar=jnp.asarray(hbar1), dt=DT_ZSTAR))
    d1 = np.abs(jx1 - deta1)[nlm0]
    d2 = np.abs(jx2 - deta2)[nlm0]
    with capsys.disabled():
        print(f"\n[ssh-replay] step1 (x0=0,hbar=0)        max|Δ|={d1.max():.2e}")
        print(f"[ssh-replay] step2 (x0=C_deta1,hbar=C_hbar1) max|Δ|={d2.max():.2e}  (D2 live)")
    # byte-faithful with identical inputs: the map/gather floor, BOTH steps. The step-2 D2
    # increment must not move it off that floor (a wrong increment ⇒ ~mm even with identical inputs).
    assert d1.max() < 1e-12, f"step1 controlled-replay d_eta max={d1.max():.2e} (solve not byte-faithful)"
    assert d2.max() < 1e-12, f"step2 controlled-replay d_eta max={d2.max():.2e} (D2 increment broke byte-fidelity)"


@core2_forcing_missing
def test_jz7_assembled_zstar_steps123(capsys):
    """JZ.7 multi-step (compute-node): the assembled CORE2 zstar model run for 3 steps vs the
    z2_cdump 3-step dump set. This is the REAL validation of the JZ.6 live-geometry re-points:
    at step 1 live==static (a cold-start no-op), so the EOS/PP/dbsfc/KPP/GM/QR4C/vert-Redi/K33/
    momentum/forcing geometry only diverges from static at steps ≥2 — here the SSH change has
    stretched the column and every re-pointed consumer reads genuinely-moving depths.

    The JAX state is CHAINED (the C dumps only the 12 ALE tags, not full T/S/uv — so this CANNOT
    be a controlled replay that resets to the C state each step). ⚠️ KEY FINDING (2026-06-12,
    corrected): the SSH-solve-derived fields (d_eta/hbar/eta_n + the hbar-built zbar_3d_n/Z_3d_n)
    diverge to ~mm at step ≥2, but this is NOT a property of the solve — the
    ``test_jz7_ssh_solve_controlled_replay`` above proves that with IDENTICAL inputs the solve (+
    the zstar D2 increment) reproduces the C d_eta to **~1e-15** (byte-faithful, BOTH steps). The
    chained ~mm divergence is the **upstream velocity/ssh_rhs FP-reassociation amplified**: the
    JAX computes its OWN ssh_rhs from the JAX velocity, ~1e-12-different from the C's, which the
    near-cancelling ssh_rhs (the Phase-2 ssh/rhs ``dx·helem~1e7`` floor) + the near-null-space
    solve blow up to ~mm in d_eta — the FP-trajectory butterfly EVERY ocean model has (the C vs
    Fortran too, hence the year-scale climate gate). BOUNDED (s2≈s3≈7e-3, not growing).

    So gate by what each tag's noise floor allows (per step):

      * **pgf** (shchepetkin on live geometry) — the STRONG JZ.6 gate: bit-faithful at step 1
        (live==static), then p50~4e-9 / p99<1e-6 at steps ≥2. pgf reads BOTH density (T/S, the
        tracer chain) AND the live ``Z_3d_n`` — its bit-faithfulness proves the re-pointed geometry
        is correct (a wrong reconstruction corrupts pgf at the geometry scale, not at 1e-9; and a
        diverged T/S would corrupt the density, so 4e-9 also certifies the re-pointed tracer chain).
      * **d_eta/hbar/eta_n + zbar_3d_n/Z_3d_n** — the upstream-reassociation-amplified class (~mm,
        bounded). A loose "no blow-up" gate, NOT a precision gate (the chained ssh_rhs differs; the
        solve+D2 byte-fidelity is gated by ``test_jz7_ssh_solve_controlled_replay`` + JZ.3).
      * **hnode/helem** — per-layer thickness (smaller, ~1e-3). **Wvel** — bit-faithful bulk
        (÷area) + CG tail. **ssh_rhs** — near-cancelling diagnostic (no gate).

    Bounds calibrated from the per-step prints (house style); finiteness is the hard gate at
    every step (no NaN through 3 assembled zstar steps = the integration milestone)."""
    from fesom_jax import core2_forcing, ice
    from fesom_jax.gm import GMConfig
    from fesom_jax.kpp import KppConfig
    from fesom_jax.ice import IceConfig
    from fesom_jax.phc_ic import core2_initial_state
    mesh = load_mesh(CORE2_MESH)
    state = core2_initial_state(mesh, CORE2_IC)
    sst = np.asarray(state.T[:, 0])
    state0 = ice.seed_ice(state, mesh, sst)
    op = ssh.build_ssh_operator(mesh, dt=DT_ZSTAR)
    cf = core2_forcing.build_core_forcing(mesh, ZSTAR_YEAR, sst_ic=sst)
    dates = core2_forcing.dates_for_steps(ZSTAR_YEAR, DT_ZSTAR, 3)
    sfs = [cf.step_forcing(*d) for d in dates]

    clean_node = _config_clean_node_mask(mesh)
    assert clean_node.all(), (f"{int((~clean_node).sum())} nodes above the relax reduction floor "
                              "— stale IC cache? rerun scripts/tools/rebuild_ic_dist16.py")
    en = np.asarray(mesh.elem_nodes)
    clean_elem = clean_node[en[:, 0]] & clean_node[en[:, 1]] & clean_node[en[:, 2]]
    nim = np.asarray(mesh.node_iface_mask)
    elm = np.asarray(mesh.elem_layer_mask)
    nlm = np.asarray(mesh.node_layer_mask)

    def _stat(tag, jx, c, mask):
        d = np.abs(np.asarray(jx, dtype=np.float64) - np.asarray(c, dtype=np.float64))[mask]
        with capsys.disabled():
            print(f"[jz7-ms] s{_sN} {tag:11s} |Δ| p50={np.percentile(d, 50):.2e} "
                  f"p99={np.percentile(d, 99):.2e} p99.9={np.percentile(d, 99.9):.2e} max={d.max():.2e}")
        return d

    cfgs = dict(kpp_cfg=KppConfig(), gm_cfg=GMConfig(), ice_cfg=IceConfig(), ale_cfg=AleConfig())
    st = state0
    R = {}                                   # (step, tag) → |Δ| array, for the deferred gates
    for i in range(3):
        _sN = i + 1
        st = stepmod.step(st, mesh, op, None, dt=DT_ZSTAR, is_first_step=(i == 0),
                          step_forcing=sfs[i], forcing_static=cf.static, **cfgs)
        # finiteness — the hard gate at every step (a NaN here is a genuine integration failure).
        for fld in ("T", "S", "uv", "hbar", "hnode", "hnode_new", "helem", "pgf_x", "pgf_y",
                    "d_eta", "w", "eta_n", "ssh_rhs", "a_ice"):
            a = np.asarray(getattr(st, fld))
            assert np.all(np.isfinite(a)), f"{fld} non-finite at zstar step {_sN}"

        # --- load this step's dump tags ---
        fp, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["pgf_x", "pgf_y"], step=_sN,
                                      n_nod=NG5_NOD2D, n_elem=NG5_ELEM2D)
        fsh, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["sshsolve", "hbar"], step=_sN, n_nod=NG5_NOD2D)
        fwv, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["Wvel"], step=_sN, n_nod=NG5_NOD2D)
        fth, _ = io_dump.load_ale_dump(ZSTAR_ORACLE, ["hnode", "zbar_3d_n", "Z_3d_n", "helem"],
                                       step=_sN, n_nod=NG5_NOD2D, n_elem=NG5_ELEM2D)

        # post-commit live geometry (the DIRECT JZ.6 gate: live_geometry vs the C reconstruction).
        zbar3_j, Z3d_j = ale.live_geometry(mesh, st.hnode)
        R[_sN, "hnode"] = _stat("hnode", st.hnode[:, :NG5_NL - 1], fth["hnode"], nlm[:, :NG5_NL - 1])
        R[_sN, "zbar_3d_n"] = _stat("zbar_3d_n", zbar3_j, fth["zbar_3d_n"], nim)
        R[_sN, "Z_3d_n"] = _stat("Z_3d_n", Z3d_j[:, :NG5_NL - 1], fth["Z_3d_n"], nlm[:, :NG5_NL - 1])
        R[_sN, "helem"] = _stat("helem", st.helem[:, :NG5_NL - 1], fth["helem"], elm[:, :NG5_NL - 1])
        # pgf (element layer)
        R[_sN, "pgf_x"] = _stat("pgf_x", st.pgf_x[:, :NG5_NL - 1], fp["pgf_x"], elm[:, :NG5_NL - 1])
        R[_sN, "pgf_y"] = _stat("pgf_y", st.pgf_y[:, :NG5_NL - 1], fp["pgf_y"], elm[:, :NG5_NL - 1])
        # ssh / hbar / Wvel
        _stat("ssh_rhs", st.ssh_rhs, io_dump.ale_component(fsh, "sshsolve", "ssh_rhs"), clean_node)
        R[_sN, "d_eta"] = _stat("d_eta", st.d_eta, io_dump.ale_component(fsh, "sshsolve", "d_eta"), clean_node)
        R[_sN, "hbar"] = _stat("hbar", st.hbar, io_dump.ale_component(fsh, "hbar", "hbar"), clean_node)
        R[_sN, "eta_n"] = _stat("eta_n", st.eta_n, io_dump.ale_component(fsh, "hbar", "eta_n"), clean_node)
        R[_sN, "Wvel"] = _stat("Wvel", st.w, fwv["Wvel"], clean_node[:, None] & nim)

    # --- deferred per-class gates (all 3 steps' prints already emitted; calibrated 2026-06-12
    #     from the run above, ~2× headroom over the observed bounded divergence) ---
    def mx(s, t):
        return R[s, t].max()

    def pc(s, t, q):
        return np.percentile(R[s, t], q)

    for s in (1, 2, 3):
        # pgf — the STRONG JZ.6 gate. Step 1 live==static ⇒ bit (max<1e-14); steps ≥2 the
        # geometry is genuinely live ⇒ p99<1e-6 (observed s2/s3 p99~3.7e-7), max<1e-5 (~4.5e-6).
        pgf_p99 = 1e-15 if s == 1 else 1e-6
        pgf_max = 1e-14 if s == 1 else 1e-5
        assert pc(s, "pgf_x", 99) < pgf_p99 and pc(s, "pgf_y", 99) < pgf_p99, \
            f"pgf p99 x={pc(s, 'pgf_x', 99):.2e} y={pc(s, 'pgf_y', 99):.2e} step {s} (JZ.6 geom regressed)"
        assert mx(s, "pgf_x") < pgf_max and mx(s, "pgf_y") < pgf_max, \
            f"pgf max x={mx(s, 'pgf_x'):.2e} y={mx(s, 'pgf_y'):.2e} step {s} (JZ.6 geom regressed)"

        # SSH-solve-derived fields — the upstream-ssh_rhs-reassociation-amplified class (the solve
        # itself is byte-faithful, test_jz7_ssh_solve_controlled_replay). Step 1 the JAX ssh_rhs is
        # ~5.5e-7-different from C's; steps ≥2 it blows up to ~mm (bounded, s2≈s3). A loose "no
        # blow-up" gate, NOT a precision gate (the chained ssh_rhs differs from the C's).
        cg_p99 = 1e-6 if s == 1 else 5e-3      # observed: s1~2.5e-7, s2/s3~1.2e-3/1.8e-3
        cg_max = 1e-6 if s == 1 else 1.5e-2    # observed: s1~5.5e-7, s2/s3~7.3e-3/7.7e-3
        for t in ("d_eta", "hbar", "eta_n", "zbar_3d_n", "Z_3d_n"):
            assert pc(s, t, 99) < cg_p99 and mx(s, t) < cg_max, \
                f"{t} p99={pc(s, t, 99):.2e} max={mx(s, t):.2e} step {s} (SSH-class blew up)"
        # per-layer thickness — a fraction of the cumulative-geometry class.
        th_p99 = 1e-6 if s == 1 else 5e-4      # observed s2/s3 p99~6e-5/8e-5
        th_max = 1e-6 if s == 1 else 5e-3      # observed s2/s3 max~1.5e-3/1.2e-3
        for t in ("hnode", "helem"):
            assert pc(s, t, 99) < th_p99 and mx(s, t) < th_max, \
                f"{t} p99={pc(s, t, 99):.2e} max={mx(s, t):.2e} step {s}"
        # Wvel — bit-faithful bulk (÷area crushes the scatter) + CG-class tail.
        wv_p50 = 1e-9 if s == 1 else 1e-6      # observed s1~1e-10, s2/s3~1.7e-7/3.3e-7
        assert pc(s, "Wvel", 50) < wv_p50 and mx(s, "Wvel") < cg_max, \
            f"Wvel p50={pc(s, 'Wvel', 50):.2e} max={mx(s, 'Wvel'):.2e} step {s}"


# ==========================================================================
# 13. JZ.8 — gradient gates (GATE 9a §4): the differentiability contract, zstar-ON
# ==========================================================================
# The assembled-model masked-NaN probe + the param/IC gradients with ale_cfg ON. N=1 (one
# step, no scan) — the cheapest backward that still traverses every zstar masked-divide lane
# (live geometry's ÷thickness guards, the shchepetkin safe denominators, the vert_vel /
# D2-closure / forcing-flip paths). KPP+GM+zstar+forcing, NO ice (the EVP 120-subcycle scan
# backward is the memory hog — ice AD is covered by K.10/JZ.2; every zstar-specific AD path is
# in the OCEAN step). A WARM hbar seed ⇒ the live geometry is genuinely active under AD. The
# CG transpose with the state-dependent D2 matvec is already gated by JZ.3
# (test_solve_ssh_state_dependent_transpose_residual); the quantitative FD↔AD plateau is the
# GPU gate (scripts/, JZ.8 deliverable). These run on a compute node (-p compute, CPU).
def _mean_sst_z(state, mesh):
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
    return jnp.sum(jnp.where(wet0, state.T[:, 0], 0.0)) / jnp.sum(wet0)


def _jz8_grad_setup(mesh):
    """CORE2 warm-zstar state (stretched hnode via the zstar init) + 1-step forcing + op."""
    from fesom_jax import core2_forcing
    from fesom_jax.phc_ic import core2_initial_state
    state = core2_initial_state(mesh, CORE2_IC)
    sst = np.asarray(state.T[:, 0])
    lat = np.asarray(mesh.geo_coord_nod2D)[:, 1]
    hbar = jnp.asarray(0.5 * np.cos(2.0 * lat))           # ~0.5 m bump ⇒ stretched column
    hnode, helem, eta_n, sro = ale.init_thickness_zstar(mesh, hbar, jnp.zeros_like(hbar), dt=DT_ZSTAR)
    state = dataclasses.replace(state, hbar=hbar, hbar_old=jnp.zeros_like(hbar), hnode=hnode,
                                helem=helem, eta_n=eta_n, ssh_rhs_old=sro)
    op = ssh.build_ssh_operator(mesh, dt=DT_ZSTAR)
    cf = core2_forcing.build_core_forcing(mesh, ZSTAR_YEAR, sst_ic=sst)
    sf = cf.step_forcing(*core2_forcing.dates_for_steps(ZSTAR_YEAR, DT_ZSTAR, 1)[0])
    return state, op, cf.static, sf


@core2_forcing_missing
def test_jz8_grad_ic_field_finite_zstar():
    """GATE 9a masked-NaN probe: ``d(mean SST)/d(T₀)`` over one assembled KPP+GM+**zstar** step
    is finite EVERYWHERE (incl. below-bottom/masked lanes — the strong probe a scalar gradient
    misses), nonzero on wet layers, and EXACTLY 0 on masked lanes. Traverses every zstar
    masked-divide backward: the live geometry's ÷thickness double-``where`` guards (EOS/PP/
    dbsfc/KPP/QR4C/momentum/GM re-points), the shchepetkin safe denominators, the vert_vel
    distribute, and the D2-stiffness closure in the CG — the proof the JZ.6 re-points + the
    JZ.1-5 kernels are AD-safe under the moving coordinate (the masked-NaN rule, lesson)."""
    import jax
    from fesom_jax.gm import GMConfig
    from fesom_jax.kpp import KppConfig
    mesh = load_mesh(CORE2_MESH)
    st0, op, fs, sf = _jz8_grad_setup(mesh)
    mlay = np.asarray(mesh.node_layer_mask)
    cfgs = dict(kpp_cfg=KppConfig(), gm_cfg=GMConfig(), ale_cfg=AleConfig())

    def loss(T0):
        s = dataclasses.replace(st0, T=T0)        # keep T_old = the const base (AB2)
        fin = stepmod.step(s, mesh, op, None, dt=DT_ZSTAR, is_first_step=True,
                           step_forcing=sf, forcing_static=fs, **cfgs)
        return _mean_sst_z(fin, mesh)

    g = np.asarray(jax.grad(loss)(st0.T))
    assert np.all(np.isfinite(g)), f"{int(np.isnan(g).sum())} non-finite grad entries (zstar masked-NaN)"
    assert np.max(np.abs(g[mlay])) > 0.0, "IC gradient identically zero on wet layers (zstar)"
    assert np.max(np.abs(g[~mlay])) == 0.0, "below-bottom lanes carry spurious gradient (zstar)"


@core2_forcing_missing
def test_jz8_grad_kver_finite_zstar():
    """``d(mean SST)/d(k_ver)`` over one **zstar** step is finite + nonzero — the PP background
    diffusivity (the 1st ML-hook) routes through ``Kv`` → the vertical tracer diffusion (whose
    layer-center spacings are now the live ``Z_3d_n``-from-``hnode_new``), with the bulk
    ``heat_flux→bc_T`` + the zstar ``−dt·sval·wf`` surface term in the RHS. PP path (KPP off)
    so ``k_ver`` is live; zstar ON so the diffusion geometry is the moving coordinate."""
    import jax
    import jax.numpy as _jnp
    from fesom_jax.config import A_VER
    from fesom_jax.params import Params
    mesh = load_mesh(CORE2_MESH)
    st0, op, fs, sf = _jz8_grad_setup(mesh)

    def loss(kver):
        p = Params(k_ver=kver, a_ver=_jnp.asarray(A_VER, _jnp.float64))
        fin = stepmod.step(st0, mesh, op, None, params=p, dt=DT_ZSTAR, is_first_step=True,
                           step_forcing=sf, forcing_static=fs, ale_cfg=AleConfig())
        return _mean_sst_z(fin, mesh)

    g_ad = float(jax.grad(loss)(_jnp.asarray(1e-4, _jnp.float64)))
    assert np.isfinite(g_ad) and g_ad != 0.0, f"d(mean SST)/d(k_ver) zstar = {g_ad}"


@core2_forcing_missing
def test_jz8_grad_hbar_ic_finite_zstar():
    """GATE 9a NEW state path: ``d(mean SST)/d(hbar-IC)`` over one assembled GM+**zstar** step is
    finite + nonzero. The initial SSH ``hbar`` rebuilds the stretched ``hnode``/``helem`` (via
    ``init_thickness_zstar``) ⇒ the live geometry ⇒ density/PGF/diffusion ⇒ SST, AND feeds the
    D2 stiffness increment (the ``custom_linear_solve`` matvec closing over ``hbar``). This is
    the gradient through the zstar-only prognostic-thickness path that linfs has no analog for —
    the proof the derived-geometry (D1) + stiffness-as-state (D2) closures are differentiable."""
    import jax
    import jax.numpy as _jnp
    from fesom_jax.gm import GMConfig
    mesh = load_mesh(CORE2_MESH)
    st0, op, fs, sf = _jz8_grad_setup(mesh)
    z = _jnp.zeros(mesh.nod2D)
    cfgs = dict(gm_cfg=GMConfig(), ale_cfg=AleConfig())

    def loss(hbar_ic):
        hnode, helem, eta_n, sro = ale.init_thickness_zstar(mesh, hbar_ic, z, dt=DT_ZSTAR)
        s = dataclasses.replace(st0, hbar=hbar_ic, hbar_old=z, hnode=hnode, helem=helem,
                                eta_n=eta_n, ssh_rhs_old=sro)
        fin = stepmod.step(s, mesh, op, None, dt=DT_ZSTAR, is_first_step=True,
                           step_forcing=sf, forcing_static=fs, **cfgs)
        return _mean_sst_z(fin, mesh)

    g = np.asarray(jax.grad(loss)(_jnp.asarray(st0.hbar)))
    assert np.all(np.isfinite(g)), f"{int(np.isnan(g).sum())} non-finite d/d(hbar-IC) entries"
    assert np.max(np.abs(g)) > 0.0, "d(mean SST)/d(hbar-IC) identically zero (new state path dead)"
