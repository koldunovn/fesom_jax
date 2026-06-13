"""mEVP sea-ice rheology gate — Phase 9c (plan ``docs/plans/20260611-fesom-jax-mevp.md``).

Verifies :mod:`fesom_jax.ice_mevp` (whichEVP=1, the Bouillon-2013 modified-EVP solver)
against the 16-rank C cdump on ``/work/ab0995/a270088/port/mevp/cdump_16r/dump`` (per-substep
gid-keyed text dumps: entry inputs Q/U0/F, precompute P, iterates it1/it2/it60/it120, final
UF). The C ran dist_16 / 2 steps / dt=1800; node partitions are disjoint and merge by gid into
the global field, so a single-device JAX replay compares global-vs-global (tighter than the
C's own C-vs-Fortran floor — same algebra).

Grows across the JM ladder:
  JM.0 — IceConfig mEVP fields + validation; dispatch stub; EVP-dump readers round-trip.
  JM.1 — shared-helper extraction (EVP graph-identity) + bc_index.
  JM.2 — the kernel: precompute (P) + per-iterate (it*) dump gates + the 14-trap checks.

SKIPS cleanly if the mesh / cdump are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
MEVP_DUMP = Path("/work/ab0995/a270088/port/mevp/cdump_16r/dump")
EVPD = ROOT / "data" / "ice_evp_dump_core2"          # std-EVP dump (JM.1 graph-identity)
EVP_BASELINE = ROOT / "fesom_jax" / "tests" / "data" / "evp_baseline_jm1.npz"
DIST16_PARENT = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")  # holds dist_16/
IC_CORE2 = ROOT / "data" / "ic_core2"                # PHC IC (live-dispatch harness)

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and MEVP_DUMP.is_dir()
         and (MEVP_DUMP / "evp_dump_s1_Q_node_rank0.txt").is_file()),
    reason="CORE2 mesh / mEVP cdump missing (see /work/ab0995/a270088/port/mevp/cdump_16r)")


@pytest.fixture(scope="module")
def mesh():
    from fesom_jax.mesh import load_mesh
    return load_mesh(MESH_DIR)


# ==========================================================================
# JM.0 — config fields + validation
# ==========================================================================
def test_iceconfig_mevp_fields():
    from fesom_jax.ice import IceConfig
    cfg = IceConfig()
    assert cfg.whichEVP == 0                              # default ⇒ standard EVP (byte-identical)
    assert cfg.alpha_evp == 250.0 and cfg.beta_evp == 250.0
    # derived relaxation weights (fesom_ice_maevp.c:110-111)
    assert cfg.mevp_det2 == pytest.approx(1.0 / 251.0, rel=0, abs=0)
    assert cfg.mevp_det1 == pytest.approx(250.0 / 251.0, rel=0, abs=0)
    # whichEVP=1 accepted; _replace preserves derived weights
    m = IceConfig(whichEVP=1, alpha_evp=300.0)
    assert m.whichEVP == 1 and m.mevp_det2 == pytest.approx(1.0 / 301.0)


def test_iceconfig_raises_aevp():
    """Direct construction with whichEVP=2 (aEVP) raises — C abort parity (no reference)."""
    from fesom_jax.ice import IceConfig
    with pytest.raises(ValueError, match="aEVP|whichEVP"):
        IceConfig(whichEVP=2)
    with pytest.raises(ValueError):
        IceConfig(whichEVP=-1)


def test_mevp_dispatch_signature_parity():
    """``mevp_dynamics`` has the SAME call signature as ``evp_dynamics`` ⇒ the ``ice_step``
    whichEVP dispatch (a function swap) is valid. The kernel itself is exercised by the JM.2
    gates; this guards the dispatch contract against signature drift."""
    import inspect
    from fesom_jax import ice_evp, ice_mevp
    pe = list(inspect.signature(ice_evp.evp_dynamics).parameters)
    pm = list(inspect.signature(ice_mevp.mevp_dynamics).parameters)
    assert pe == pm, f"signature mismatch: evp={pe} vs mevp={pm}"


# ==========================================================================
# JM.0 — EVP-dump readers round-trip
# ==========================================================================
def test_evp_reader_node_roundtrip(mesh):
    """Node points (Q/U0/F/P/it1/UF) merge across the 16 ranks into the full global field
    (disjoint partitions ⇒ every gid 1..nod2D written exactly once), columns in C order."""
    from fesom_jax import io_dump
    n_nod = int(mesh.nod2D)
    pts = ["Q", "U0", "F", "P", "it1", "it120", "UF"]
    fields, meta = io_dump.load_mevp_dump(MEVP_DUMP, pts, step=1, array="node",
                                          n_nod=n_nod, strict=True)
    for p in pts:
        assert fields[p].shape[0] == n_nod, f"{p}: N {fields[p].shape[0]} != nod2D {n_nod}"
        assert np.isfinite(fields[p]).all(), f"{p}: NaN ⇒ a gid was never written (strict bug)"
        assert meta[p]["nranks"] == 16
    # column layout + real data: Q carries the cold-start ice IC (a_ice>0 somewhere)
    a_ice = io_dump.evp_component(fields, meta, "Q", "a_ice")
    assert (a_ice > 0.01).sum() > 1000, "Q a_ice has no ice — wrong column or empty dump"
    assert fields["Q"].shape[1] == 4 and fields["U0"].shape[1] == 2 and fields["P"].shape[1] == 4


def test_evp_reader_elem_roundtrip(mesh):
    """Element points (P/it*) merge by gid; the boundary-ring overlap rows are bit-identical
    (strict asserts it) — pressure_fac (1 comp) + the σ iterates (3 comp)."""
    from fesom_jax import io_dump
    n_elem = int(mesh.elem2D)
    fields, meta = io_dump.load_mevp_dump(MEVP_DUMP, ["P", "it1", "it120"], step=1,
                                          array="elem", n_elem=n_elem, strict=True)
    assert fields["P"].shape == (n_elem, 1)
    assert fields["it1"].shape == (n_elem, 3) and fields["it120"].shape == (n_elem, 3)
    assert np.isfinite(fields["P"]).all()
    pf = io_dump.evp_component(fields, meta, "P", "pressure_fac")
    assert (pf > 0.0).sum() > 1000, "pressure_fac all zero — no iced elements?"


def test_evp_reader_infers_ncomp():
    """:func:`read_evp_table` infers ncomp from the row width (the header has no ncomp)."""
    from fesom_jax import io_dump
    g, v, meta = io_dump.read_evp_table(MEVP_DUMP / "evp_dump_s1_Q_node_rank0.txt")
    assert meta["ncomp"] == 4 and v.shape == (meta["N"], 4)
    assert meta["point"] == "Q" and meta["array"] == "node" and meta["step"] == 1
    assert g.min() >= 1                                   # 1-based gids
    g2, v2, m2 = io_dump.read_evp_table(MEVP_DUMP / "evp_dump_s1_it1_elem_rank0.txt")
    assert m2["ncomp"] == 3                               # σ11/σ12/σ22


# ==========================================================================
# JM.1 — shared-helper extraction (EVP graph-identity, BITWISE) + bc_index
# ==========================================================================
@pytest.mark.skipif(not (EVPD.is_dir() and EVP_BASELINE.is_file()),
                    reason="std-EVP dump / pre-refactor baseline missing")
def test_evp_graph_identity(mesh):
    """The JM.1 refactor (extracting ``strain_rates`` + ``stress_div_scatter`` shared with mEVP)
    must leave the EVP path **bitwise-identical** — max|Δ|==0 vs the pre-refactor baseline (the
    binding gate; HLO comparison advisory). Covers σ, the rhs scatter, the velocity update, the
    full 120-subcycle ``evp_dynamics``, and the bare ``strain_rates`` block on a random field."""
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_evp as ie
    b = np.load(EVP_BASELINE)
    cfg = IceConfig()

    def L(pt, cls):
        return np.loadtxt(EVPD / f"evp_dump_s1_{pt}_{cls}_rank0.txt")
    Q = L("Q", "node"); F = L("F", "node")
    a_ice, m_ice, m_snow, elev = (jnp.asarray(Q[:, i]) for i in (1, 2, 3, 4))
    sax, say, u_w, v_w = (jnp.asarray(F[:, i]) for i in (1, 2, 3, 4))
    z = jnp.zeros(int(mesh.nod2D)); ze = jnp.zeros(int(mesh.elem2D))
    bn = ie.boundary_node_mask(mesh)

    st = ie.evp_setup(cfg, mesh, a_ice, m_ice, m_snow, elev)
    s11, s12, s22 = ie.stress_tensor(cfg, mesh, z, z, ze, ze, ze, st.ice_strength)
    ur, vr = ie.stress2rhs(cfg, mesh, s11, s12, s22, st.ice_strength,
                           st.inv_areamass, st.tilt_u, st.tilt_v)
    u4, v4 = ie.velocity_update(cfg, mesh, z, z, ur, vr, u_w, v_w, sax, say,
                                st.inv_mass, a_ice, bn)
    ue, ve, *_ = ie.evp_dynamics(
        cfg, mesh, a_ice=a_ice, m_ice=m_ice, m_snow=m_snow, u_ice=z, v_ice=z,
        sigma11=ze, sigma12=ze, sigma22=ze, srfoce_u=u_w, srfoce_v=v_w,
        elevation=elev, stress_ax=sax, stress_ay=say, boundary_node=bn)
    e11, e22, e12 = ie.strain_rates(mesh, jnp.asarray(b["utest"]), jnp.asarray(b["vtest"]))

    pairs = [(s11, "s11"), (s12, "s12"), (s22, "s22"), (ur, "ur"), (vr, "vr"),
             (u4, "u4"), (v4, "v4"), (ue, "ue"), (ve, "ve"),
             (e11, "strain_e11"), (e22, "strain_e22"), (e12, "strain_e12")]
    for j, k in pairs:
        d = float(np.abs(np.asarray(j) - b[k]).max())
        assert d == 0.0, f"EVP refactor changed {k}: max|Δ|={d:.3e} (must be bitwise 0)"


def test_bc_index_complement(mesh):
    """``bc_index_nod2D = 1 − boundary_node_mask``: binary, complement of the coastal mask,
    interior-majority (== the C ``fesom_ice.c:249-258`` build)."""
    import jax.numpy as jnp
    from fesom_jax import ice_evp as ie
    bn = ie.boundary_node_mask(mesh)
    bc = np.asarray(ie.bc_index_nod2D(bn))
    bnf = np.asarray(bn).astype(np.float64)
    assert np.all((bc == 0.0) | (bc == 1.0))             # binary
    assert np.allclose(bc + bnf, 1.0, atol=0, rtol=0)    # exact complement
    assert (bc == 0.0).sum() == int(bnf.sum()) > 1000    # coastal nodes zeroed
    assert (bc == 1.0).sum() > 10 * (bc == 0.0).sum()    # interior-majority


@pytest.mark.skipif(not (DIST16_PARENT / "dist_16").is_dir(), reason="dist_16 partition missing")
def test_bc_index_no_seam_flagged(mesh):
    """dist_16 spot-check (C trap #1): bc_index from the GLOBAL mask flags NO partition-seam
    node as coastal. Seam nodes = halo nodes (owned elsewhere, in some rank's stencil) that are
    interior in the global mesh — a LOCAL submesh recompute would wrongly zero them."""
    import jax.numpy as jnp
    from fesom_jax import ice_evp as ie
    from fesom_jax.partit import read_partition
    part = read_partition(DIST16_PARENT, 16)
    mask = np.asarray(ie.boundary_node_mask(mesh))       # global coastal mask
    bc = np.asarray(ie.bc_index_nod2D(jnp.asarray(mask)))
    # union of every rank's HALO nodes (the eDim tail of myList_nod2D — owned by other ranks)
    halo = set()
    for r in range(16):
        ml = np.asarray(part.myList_nod2D[r])
        my = int(part.myDim_nod2D[r])
        halo.update(int(g) for g in ml[my:])             # 0-based gids
    halo = np.array(sorted(halo), dtype=np.int64)
    assert (halo >= 0).all() and (halo < int(mesh.nod2D)).all()
    seam_interior = halo[mask[halo] == False]            # noqa: E712  partition-boundary interior
    assert seam_interior.size > 1000, "no interior seam nodes — vacuous test"
    assert np.all(bc[seam_interior] == 1.0), \
        f"{int((bc[seam_interior] != 1.0).sum())} seam nodes wrongly flagged coastal"


# ==========================================================================
# JM.2 — the mEVP kernel: precompute (P) + per-iterate (it*) dump gates vs cdump_16r
# The cdump is dist_16 / 2 steps / dt=1800; step-1 entry inputs Q/U0/F merged global. The
# cold start has u_w=elev=0 (ocean at rest) ⇒ it1 is a pure per-node wind-stress map (no
# scatter ⇒ bit-identical); the scatter-reassociation floor accumulates over the 120 iterations
# (velocity is the BINDING metric; σ is VP-kink noise-amplified — tracked with context).
# ==========================================================================
@pytest.fixture(scope="module")
def mevp_s1(mesh):
    """s1 entry inputs (Q/U0/F merged global) → mEVP setup + 120 eager iterations (the production
    ``mevp_iterate`` body, exch=None) → snapshots at it1/2/60/120 (the C dump points)."""
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_evp as ie, ice_mevp as im, io_dump
    n_nod, n_elem = int(mesh.nod2D), int(mesh.elem2D)
    cfg = IceConfig(whichEVP=1, ice_dt=1800.0)               # the cdump dt (rdt = ice_dt)
    fn, mn = io_dump.load_mevp_dump(MEVP_DUMP, ["Q", "U0", "F"], step=1, array="node", n_nod=n_nod)

    def g(p, c):
        return jnp.asarray(io_dump.evp_component(fn, mn, p, c))
    inp = dict(a_ice=g("Q", "a_ice"), m_ice=g("Q", "m_ice"), m_snow=g("Q", "m_snow"),
               elev=g("Q", "elevation"), u0=g("U0", "u_ice"), v0=g("U0", "v_ice"),
               sax=g("F", "stress_atmice_x"), say=g("F", "stress_atmice_y"),
               u_w=g("F", "u_w"), v_w=g("F", "v_w"))
    bn = ie.boundary_node_mask(mesh)
    st = im.mevp_setup(cfg, mesh, inp["a_ice"], inp["m_ice"], inp["m_snow"], inp["elev"], bn)
    ze = jnp.zeros(n_elem)
    carry = (inp["u0"], inp["v0"], ze, ze, ze)               # σ starts 0 (cold start)
    snaps = {}
    for it in range(1, 121):
        carry = im.mevp_iterate(cfg, mesh, *carry, st, inp["u0"], inp["v0"],
                                inp["u_w"], inp["v_w"], inp["sax"], inp["say"], cfg.ice_dt)
        if it in (1, 2, 60, 120):
            snaps[it] = tuple(np.asarray(x) for x in carry)
    return dict(cfg=cfg, st=st, snaps=snaps, bn=bn, n_nod=n_nod, n_elem=n_elem, **inp)


def _pt(point, array, n):
    from fesom_jax import io_dump
    f, m = io_dump.load_mevp_dump(MEVP_DUMP, [point], step=1, array=array,
                                  n_nod=(n if array == "node" else None),
                                  n_elem=(n if array == "elem" else None))
    return f, m


def test_mevp_precompute_P(mevp_s1):
    """Setup precompute (inv_thickness, mass, ssh-tilt rhs_a/rhs_m, pressure_fac) bit-faithful
    vs the C P dump (~1e-13 bar; actually bit-identical — these are per-node/elem maps)."""
    from fesom_jax import io_dump
    st = mevp_s1["st"]
    fn, mn = _pt("P", "node", mevp_s1["n_nod"])
    fe, me = _pt("P", "elem", mevp_s1["n_elem"])

    def mx(j, c):
        return float(np.abs(np.asarray(j) - c).max())
    assert mx(st.inv_thickness, io_dump.evp_component(fn, mn, "P", "inv_thickness")) < 1e-13
    assert mx(st.mass, io_dump.evp_component(fn, mn, "P", "mass")) < 1e-13
    assert mx(st.tilt_u, io_dump.evp_component(fn, mn, "P", "rhs_a")) < 1e-13
    assert mx(st.tilt_v, io_dump.evp_component(fn, mn, "P", "rhs_m")) < 1e-13
    pf = io_dump.evp_component(fe, me, "P", "pressure_fac")
    assert mx(st.pressure_fac, pf) < 1e-13
    assert np.abs(pf).max() > 10.0                           # pressure_fac is non-trivial


@pytest.mark.parametrize("it,tol", [(1, 1e-13), (2, 1e-13), (60, 1e-11), (120, 5e-12)])
def test_mevp_iterate_velocity(mevp_s1, it, tol):
    """Per-iterate u_aux/v_aux (the BINDING metric) vs the C it{1,2,60,120} dump. it1 is
    bit-identical (pure wind-stress map, no scatter); the late-iterate floor is the accumulated
    element→node scatter reassociation (16-rank C vs single-device JAX)."""
    from fesom_jax import io_dump
    f, m = _pt(f"it{it}", "node", mevp_s1["n_nod"])
    ua = io_dump.evp_component(f, m, f"it{it}", "u_aux")
    va = io_dump.evp_component(f, m, f"it{it}", "v_aux")
    u, v = mevp_s1["snaps"][it][0], mevp_s1["snaps"][it][1]
    du = float(np.abs(u - ua).max()); dv = float(np.abs(v - va).max())
    assert du < tol and dv < tol, f"it{it} u|Δ|={du:.2e} v|Δ|={dv:.2e} (tol {tol:.0e})"
    if it == 120:
        assert np.abs(ua).max() > 0.05                       # ice actually moves (gate meaningful)


def test_mevp_coldstart_sigma_it1(mevp_s1):
    """Cold-start σ ≡ 0 at it1 (Δ=0 at u_aux=0 ⇒ pressure·0 ⇒ no stress) — reproduced bit-exact
    vs the C it1 σ dump (trap 11: σ persists, but the first iterate from a 0 carry is 0)."""
    from fesom_jax import io_dump
    f, m = _pt("it1", "elem", mevp_s1["n_elem"])
    s11, s12, s22 = (mevp_s1["snaps"][1][k] for k in (2, 3, 4))
    assert float(np.abs(s11).max()) == 0.0                   # JAX σ(it1) is exactly 0
    for j, name in [(s11, "sigma11"), (s12, "sigma12"), (s22, "sigma22")]:
        c = io_dump.evp_component(f, m, "it1", name)
        assert float(np.abs(np.asarray(j) - c).max()) == 0.0  # C σ(it1) is 0 too


def test_mevp_sigma_tracked(mevp_s1):
    """σ at it120 is tracked with context (NOT a binding gate): VP-kink noise-amplified near
    rigid pack — absolute O(1e-6) but RELATIVE ~1e-9 (the C's M3 saw e-13→e-8 growth; σ is in
    the sharded _DIAG_FIELDS exclusion). The velocity at the same iterate binds at ~1e-12."""
    from fesom_jax import io_dump
    f, m = _pt("it120", "elem", mevp_s1["n_elem"])
    s11 = mevp_s1["snaps"][120][2]
    c = io_dump.evp_component(f, m, "it120", "sigma11")
    rel = float(np.abs(s11 - c).max()) / max(float(np.abs(c).max()), 1e-30)
    assert np.isfinite(rel) and rel < 1e-6, f"σ11 relative |Δ|={rel:.2e} (VP-kink noise)"


def test_mevp_dynamics_scan_vs_UF(mevp_s1, mesh):
    """The production driver ``mevp_dynamics`` (checkpointed lax.scan, jitted) reproduces the C
    final velocity UF within the XLA-fusion + scatter floor (~1e-10) — validates the scan/
    checkpoint/exch wiring (the per-iterate gates above validate the kernel math eagerly)."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import io_dump, ice_mevp as im
    cfg = mevp_s1["cfg"]; ze = jnp.zeros(mevp_s1["n_elem"])
    fn = jax.jit(lambda **k: im.mevp_dynamics(cfg, mesh, **k))
    ue, ve, *_ = fn(a_ice=mevp_s1["a_ice"], m_ice=mevp_s1["m_ice"], m_snow=mevp_s1["m_snow"],
                    u_ice=mevp_s1["u0"], v_ice=mevp_s1["v0"], sigma11=ze, sigma12=ze, sigma22=ze,
                    srfoce_u=mevp_s1["u_w"], srfoce_v=mevp_s1["v_w"], elevation=mevp_s1["elev"],
                    stress_ax=mevp_s1["sax"], stress_ay=mevp_s1["say"], boundary_node=mevp_s1["bn"])
    f, m = _pt("UF", "node", mevp_s1["n_nod"])
    uf_u = io_dump.evp_component(f, m, "UF", "u_ice"); uf_v = io_dump.evp_component(f, m, "UF", "v_ice")
    du = float(np.abs(np.asarray(ue) - uf_u).max()); dv = float(np.abs(np.asarray(ve) - uf_v).max())
    assert du < 1e-10 and dv < 1e-10, f"UF u|Δ|={du:.2e} v|Δ|={dv:.2e}"


def test_mevp_entry_anchor(mevp_s1, mesh):
    """Decisions #4 / trap 1: the backward-Euler rhs anchors on the FROZEN ENTRY ``(u_ice,
    v_ice)``, NOT the current iterate (the std-EVP template). The cold-start it2 catches it —
    entry-anchored matches the C bit-exact; iterate-anchored (the bug) diverges grossly."""
    import jax.numpy as jnp
    from fesom_jax import io_dump, ice_mevp as im
    cfg = mevp_s1["cfg"]; st = mevp_s1["st"]; ze = jnp.zeros(mevp_s1["n_elem"])
    u0, v0, u_w, v_w = (mevp_s1[k] for k in ("u0", "v0", "u_w", "v_w"))
    sax, say = mevp_s1["sax"], mevp_s1["say"]
    f, m = _pt("it2", "node", mevp_s1["n_nod"])
    ua = io_dump.evp_component(f, m, "it2", "u_aux")
    # correct: anchor = frozen entry (u0) every iteration
    c = (u0, v0, ze, ze, ze)
    for _ in range(2):
        c = im.mevp_iterate(cfg, mesh, *c, st, u0, v0, u_w, v_w, sax, say, cfg.ice_dt)
    # buggy: anchor = the current iterate (std-EVP template)
    b = (u0, v0, ze, ze, ze)
    for _ in range(2):
        b = im.mevp_iterate(cfg, mesh, *b, st, b[0], b[1], u_w, v_w, sax, say, cfg.ice_dt)
    d_correct = float(np.abs(np.asarray(c[0]) - ua).max())
    d_buggy = float(np.abs(np.asarray(b[0]) - ua).max())
    assert d_correct < 1e-13, f"entry-anchored it2 should match C ({d_correct:.2e})"
    assert d_buggy > 1e-7, f"iterate-anchored it2 should diverge ({d_buggy:.2e})"


def test_mevp_trap13_nonice_retained(mesh):
    """Trap 13 (plan MAJOR): a non-ice INTERIOR node keeps its velocity across an iteration
    (identity carry — std-EVP's else-zero would wrongly zero ice-edge velocities); a boundary
    node is zeroed by the edge-BC. The cold-start dumps can't distinguish identity-vs-zero (all
    velocities are 0 at s1), so this is a dedicated synthetic test."""
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_evp as ie, ice_mevp as im
    cfg = IceConfig(whichEVP=1, ice_dt=1800.0)
    n_nod, n_elem = int(mesh.nod2D), int(mesh.elem2D)
    # ice everywhere a_ice=0 EXCEPT we pick one interior non-ice node to seed
    a_ice = jnp.zeros(n_nod); m_ice = jnp.zeros(n_nod); m_snow = jnp.zeros(n_nod)
    elev = jnp.zeros(n_nod); z = jnp.zeros(n_nod); ze = jnp.zeros(n_elem)
    bn = np.asarray(ie.boundary_node_mask(mesh))
    interior = np.where(~bn)[0]
    n_int = int(interior[len(interior) // 2])                # an interior (bc_index=1) non-ice node
    n_bnd = int(np.where(bn)[0][0])                          # a coastal node
    u_seed = np.zeros(n_nod); u_seed[n_int] = 5.0; u_seed[n_bnd] = 5.0
    u_seed = jnp.asarray(u_seed)
    st = im.mevp_setup(cfg, mesh, a_ice, m_ice, m_snow, elev, jnp.asarray(bn))
    assert not bool(st.ice_nod[n_int])                       # the seeded node is non-ice
    u1, v1, *_ = im.mevp_iterate(cfg, mesh, u_seed, z, ze, ze, ze, st,
                                 z, z, z, z, z, z, cfg.ice_dt)
    assert float(np.asarray(u1)[n_int]) == 5.0, "non-ice interior node was NOT retained (trap 13)"
    assert float(np.asarray(u1)[n_bnd]) == 0.0, "boundary node was not zeroed by the edge-BC"


# ==========================================================================
# JM.3 — step wiring: s2 controlled replay (σ persistence) + live end-to-end dispatch
# ==========================================================================
def test_mevp_s2_replay(mesh):
    """s2 controlled replay: feed the C's STEP-2 entry (Q/U0/F) + the s1-final σ (it120) and
    reproduce step-2 it*/UF. Two things at once: (1) the FULL rheology under realistic ocean
    state (|u_w|=0.39, |elev|=0.35, entry |u_ice|=0.23, nonzero σ), tighter than s1's wind-only
    cold start; (2) σ PERSISTENCE (T11) — feeding s1's final σ as the s2 entry reproduces s2, so
    σ is NOT zeroed on entry (a wrong zero would diverge here)."""
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import ice_evp as ie, ice_mevp as im, io_dump
    n_nod, n_elem = int(mesh.nod2D), int(mesh.elem2D)
    cfg = IceConfig(whichEVP=1, ice_dt=1800.0)
    fn, mn = io_dump.load_mevp_dump(MEVP_DUMP, ["Q", "U0", "F"], step=2, array="node", n_nod=n_nod)

    def g(p, c):
        return jnp.asarray(io_dump.evp_component(fn, mn, p, c))
    a_ice, m_ice, m_snow, elev = g("Q", "a_ice"), g("Q", "m_ice"), g("Q", "m_snow"), g("Q", "elevation")
    u0, v0 = g("U0", "u_ice"), g("U0", "v_ice")
    sax, say, u_w, v_w = (g("F", n) for n in ("stress_atmice_x", "stress_atmice_y", "u_w", "v_w"))
    fe, me = io_dump.load_mevp_dump(MEVP_DUMP, ["it120"], step=1, array="elem", n_elem=n_elem)
    s11_0, s12_0, s22_0 = (jnp.asarray(io_dump.evp_component(fe, me, "it120", s))
                           for s in ("sigma11", "sigma12", "sigma22"))   # σ carried from s1 (T11)
    bn = ie.boundary_node_mask(mesh)
    st = im.mevp_setup(cfg, mesh, a_ice, m_ice, m_snow, elev, bn)
    carry = (u0, v0, s11_0, s12_0, s22_0)
    snaps = {}
    for it in range(1, 121):
        carry = im.mevp_iterate(cfg, mesh, *carry, st, u0, v0, u_w, v_w, sax, say, cfg.ice_dt)
        if it in (1, 2, 60, 120):
            snaps[it] = tuple(np.asarray(x) for x in carry)
    tol = {1: 1e-13, 2: 1e-13, 60: 1e-11, 120: 5e-12}
    for it in (1, 2, 60, 120):
        f, m = io_dump.load_mevp_dump(MEVP_DUMP, [f"it{it}"], step=2, array="node", n_nod=n_nod)
        ua = io_dump.evp_component(f, m, f"it{it}", "u_aux")
        va = io_dump.evp_component(f, m, f"it{it}", "v_aux")
        du = float(np.abs(snaps[it][0] - ua).max()); dv = float(np.abs(snaps[it][1] - va).max())
        assert du < tol[it] and dv < tol[it], f"s2 it{it} u|Δ|={du:.2e} v|Δ|={dv:.2e}"
    f, m = io_dump.load_mevp_dump(MEVP_DUMP, ["UF"], step=2, array="node", n_nod=n_nod)
    duf = float(np.abs(snaps[120][0] - io_dump.evp_component(f, m, "UF", "u_ice")).max())
    assert duf < 5e-12, f"s2 UF u|Δ|={duf:.2e}"


@pytest.fixture(scope="module")
def ice_harness(mesh):
    """The assembled-ice-step harness (PHC IC + cold-start ice + Jan-1958 forcing) for the live
    mEVP dispatch test."""
    import jax.numpy as jnp                                  # noqa: F401
    from fesom_jax import core2_forcing, ice
    from fesom_jax.phc_ic import core2_initial_state
    sst = np.asarray(core2_initial_state(mesh, IC_CORE2).T[:, 0])
    state0 = ice.seed_ice(core2_initial_state(mesh, IC_CORE2), mesh, sst)
    cf = core2_forcing.build_core_forcing(mesh, 1958, sst_ic=sst)
    sf = cf.step_forcing(1958, 1, 0.0, 1)                    # 1958-01-01 00:00, January
    return state0, cf, sf


@pytest.mark.skipif(not (IC_CORE2 / "T_ic.npy").is_file(), reason="PHC IC (data/ic_core2) missing")
def test_mevp_live_dispatch(mesh, ice_harness):
    """The whichEVP=1 dispatch routes the ASSEMBLED ice step (setup→mEVP→FCT→thermo→coupling)
    to the mEVP kernel — eager AND jit. Confirms: (1) finite, physical ice velocity; (2) the
    dispatch is LIVE (mEVP u_ice ≠ EVP u_ice); (3) jit traces the static-branch dispatch."""
    import jax
    from fesom_jax.ice import IceConfig
    from fesom_jax.ice_step import ice_surface_step
    state0, cf, sf = ice_harness
    DT = 1800.0
    out0 = ice_surface_step(IceConfig(whichEVP=0), mesh, state0, sf, cf.static, dt=DT)  # EVP
    out1 = ice_surface_step(IceConfig(whichEVP=1), mesh, state0, sf, cf.static, dt=DT)  # mEVP
    for o, nm in [(out0, "EVP"), (out1, "mEVP")]:
        ui = np.asarray(o.u_ice); vi = np.asarray(o.v_ice)
        assert np.isfinite(ui).all() and np.isfinite(vi).all(), f"{nm} non-finite u_ice"
        assert max(np.abs(ui).max(), np.abs(vi).max()) < 2.0, f"{nm} unphysical ice speed"
    d = float(np.abs(np.asarray(out1.u_ice) - np.asarray(out0.u_ice)).max())
    assert d > 1e-3, f"mEVP u_ice == EVP u_ice ({d:.2e}) — dispatch not live"
    # jit the mEVP step (cfg/mesh/sf/static closed over ⇒ the whichEVP branch is trace-time)
    j = jax.jit(lambda st: ice_surface_step(IceConfig(whichEVP=1), mesh, st, sf, cf.static, dt=DT))
    out1j = j(state0)
    dj = float(np.abs(np.asarray(out1j.u_ice) - np.asarray(out1.u_ice)).max())
    assert dj < 1e-10, f"jit vs eager mEVP u_ice |Δ|={dj:.2e}"


# ==========================================================================
# JM.4 — gradient gates (the differentiability contract; the mEVP kernel)
# ==========================================================================
@pytest.fixture(scope="module")
def mevp_grad_inputs(mesh):
    """s1 entry inputs (a subset) + a SHORT-scan mEVP cfg for the CPU AD probes."""
    import jax.numpy as jnp
    from fesom_jax.ice import IceConfig
    from fesom_jax import io_dump
    n_nod, n_elem = int(mesh.nod2D), int(mesh.elem2D)
    fn, mn = io_dump.load_mevp_dump(MEVP_DUMP, ["Q", "U0", "F"], step=1, array="node", n_nod=n_nod)

    def g(p, c):
        return jnp.asarray(io_dump.evp_component(fn, mn, p, c))
    return dict(
        cfg=IceConfig(whichEVP=1, ice_dt=1800.0, evp_rheol_steps=6),   # short scan for CPU AD
        a_ice=g("Q", "a_ice"), m_ice=g("Q", "m_ice"), m_snow=g("Q", "m_snow"),
        elev=g("Q", "elevation"), u0=g("U0", "u_ice"), v0=g("U0", "v_ice"),
        sax=g("F", "stress_atmice_x"), say=g("F", "stress_atmice_y"),
        u_w=g("F", "u_w"), v_w=g("F", "v_w"), n_nod=n_nod, n_elem=n_elem)


def test_mevp_grad_masked_nan(mevp_grad_inputs, mesh):
    """Masked-NaN probe: ``d(Σu_ice²)/d(stress_ax)`` through the mEVP scan is FINITE everywhere
    — including the ice-free (``a_ice<0.01``) lanes where the mass/thickness divides and the
    identity carry could leak a ``0·inf`` backward (the AD-safe masked divides + safe-sqrt +
    safe-speed + identity carry). Nonzero on the iced lanes (the gradient flows)."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import ice_evp as ie, ice_mevp as im
    gi = mevp_grad_inputs
    ze = jnp.zeros(gi["n_elem"]); bn = ie.boundary_node_mask(mesh)

    def loss(stress_ax):
        u, v, *_ = im.mevp_dynamics(
            gi["cfg"], mesh, a_ice=gi["a_ice"], m_ice=gi["m_ice"], m_snow=gi["m_snow"],
            u_ice=gi["u0"], v_ice=gi["v0"], sigma11=ze, sigma12=ze, sigma22=ze,
            srfoce_u=gi["u_w"], srfoce_v=gi["v_w"], elevation=gi["elev"],
            stress_ax=stress_ax, stress_ay=gi["say"], boundary_node=bn)
        return jnp.sum(u ** 2 + v ** 2)

    grad = np.asarray(jax.grad(loss)(gi["sax"]))
    nbad = int((~np.isfinite(grad)).sum())
    assert nbad == 0, f"{nbad} non-finite gradient entries (masked-NaN leak)"
    assert np.any(grad != 0.0), "gradient is identically zero (no flow)"
    # the ice-free lanes must be finite too (the binding masked-NaN check)
    icefree = np.asarray(gi["a_ice"]) < 0.01
    assert np.all(np.isfinite(grad[icefree])), "non-finite gradient on ice-free lanes"


def test_mevp_grad_ice_ic_finite(mevp_grad_inputs, mesh):
    """``d(loss)/d(m_ice)`` (an ice IC field) through the assembled mEVP scan is FINITE (the
    rheology gradient is stiff — `∂p/∂Δ` near rigid pack is huge — but finite by construction;
    finiteness is the contract, magnitude is documented as the std-EVP ~1e16 stiff class)."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import ice_evp as ie, ice_mevp as im
    gi = mevp_grad_inputs
    ze = jnp.zeros(gi["n_elem"]); bn = ie.boundary_node_mask(mesh)

    def loss(m_ice):
        u, v, *_ = im.mevp_dynamics(
            gi["cfg"], mesh, a_ice=gi["a_ice"], m_ice=m_ice, m_snow=gi["m_snow"],
            u_ice=gi["u0"], v_ice=gi["v0"], sigma11=ze, sigma12=ze, sigma22=ze,
            srfoce_u=gi["u_w"], srfoce_v=gi["v_w"], elevation=gi["elev"],
            stress_ax=gi["sax"], stress_ay=gi["say"], boundary_node=bn)
        return jnp.sum(u ** 2 + v ** 2)

    grad = np.asarray(jax.grad(loss)(gi["m_ice"]))
    assert np.all(np.isfinite(grad)), f"{int((~np.isfinite(grad)).sum())} non-finite d/d(m_ice)"
    assert np.any(grad != 0.0)


def test_mevp_delta_singularity(mevp_grad_inputs, mesh):
    """``d(Σσ²)/d(u_aux)`` is FINITE at u_aux=0 (Δ→0): the additive δmin (``pressure =
    pf/(Δ+δmin)``) is C¹-continuous — stiff (pressure ~ pf/δmin ~ 1e13) but never 1/√0. Contrast
    std-EVP's ``max(Δ,δmin)`` kink; both are finite, mEVP's is large-but-smooth."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import ice_evp as ie, ice_mevp as im
    gi = mevp_grad_inputs
    st = im.mevp_setup(gi["cfg"], mesh, gi["a_ice"], gi["m_ice"], gi["m_snow"], gi["elev"],
                       ie.boundary_node_mask(mesh))
    ze = jnp.zeros(gi["n_elem"])

    def loss(u):
        _, _, s11, s12, s22 = im.mevp_iterate(
            gi["cfg"], mesh, u, u, ze, ze, ze, st, u, u,
            gi["u_w"], gi["v_w"], gi["sax"], gi["say"], gi["cfg"].ice_dt)
        return jnp.sum(s11 ** 2 + s12 ** 2 + s22 ** 2)

    grad = np.asarray(jax.grad(loss)(jnp.zeros(gi["n_nod"])))
    assert np.all(np.isfinite(grad)), f"{int((~np.isfinite(grad)).sum())} non-finite at Δ=0"
