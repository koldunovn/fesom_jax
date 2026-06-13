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
