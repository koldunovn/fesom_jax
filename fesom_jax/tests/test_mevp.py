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


def test_mevp_stub_routes():
    """The whichEVP=1 dispatch reaches the mEVP kernel (a stub until JM.2)."""
    from fesom_jax import ice_mevp
    from fesom_jax.ice import IceConfig
    with pytest.raises(NotImplementedError):
        ice_mevp.mevp_dynamics(
            IceConfig(whichEVP=1), None, a_ice=None, m_ice=None, m_snow=None,
            u_ice=None, v_ice=None, sigma11=None, sigma12=None, sigma22=None,
            srfoce_u=None, srfoce_v=None, elevation=None, stress_ax=None, stress_ay=None)


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
