"""Phase 6C Task K.7 gate — KPP enhance + smooth_blmc + combine + node→elem (Av).

The KPP driver tail: feed the C-dumped blmix outputs (blmc_m/t/s, bl_ghats, dkm1) +
bldepth (hbl/caseA/kbl) + ri_iwmix (viscA/diffKt/diffKs) through
:func:`kpp.enhance` → :func:`kpp.assemble_mixing` and match the C final-dump fields
(post-combine node viscA/diffKt/diffKs/ghats + element viscAE). ``Kv = diffKt`` (the
T-channel, used for both T and S in CORE2). Plus AD finiteness.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import kpp
from fesom_jax.io_dump import load_kpp_dump
from fesom_jax.mesh import load_mesh

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH_DIR = ROOT / "data" / "mesh_core2"
KPP_DUMP_DIR = ROOT / "data" / "kpp_dump_core2"

pytestmark = pytest.mark.skipif(
    not (CORE2_MESH_DIR.is_dir() and (KPP_DUMP_DIR / "kpp_dump_s1_viscAE_rank0.txt").is_file()),
    reason=f"CORE2 mesh / KPP final dump missing ({KPP_DUMP_DIR}); run jax_kpp_dump_core2.sh",
)


@pytest.fixture(scope="module")
def setup():
    mesh = load_mesh(CORE2_MESH_DIR)
    cfg = kpp.KppConfig()
    nl = mesh.nl
    f, _ = load_kpp_dump(KPP_DUMP_DIR, ["ri_viscA", "ri_diffKt", "ri_diffKs", "bldepth",
                                        "blmc_m", "blmc_t", "blmc_s", "bl_ghats", "dkm1",
                                        "viscA", "diffKt", "diffKs", "ghats", "viscAE"])
    bd = f["bldepth"]
    hbl = jnp.asarray(bd[:, 0]); caseA = jnp.asarray(bd[:, 4])
    kbl = jnp.asarray((bd[:, 1] - 1).astype(np.int32))
    riA, riT, riS = (jnp.asarray(f["ri_viscA"]), jnp.asarray(f["ri_diffKt"]),
                     jnp.asarray(f["ri_diffKs"]))
    ghats_in = jnp.asarray(np.pad(f["bl_ghats"], ((0, 0), (0, 1))))   # (N,nl-1)→(N,nl)
    bM, bT, bS, gh = kpp.enhance(mesh, jnp.asarray(f["blmc_m"]), jnp.asarray(f["blmc_t"]),
                                 jnp.asarray(f["blmc_s"]), ghats_in, jnp.asarray(f["dkm1"]),
                                 riA, riT, riS, hbl, caseA, kbl, cfg)
    Kv, Av, vA, dT, dS, ghf = kpp.assemble_mixing(mesh, bM, bT, bS, gh, riA, riT, riS, kbl, cfg)
    out = {n: np.asarray(x) for n, x in
           [("viscA", vA), ("diffKt", dT), ("diffKs", dS), ("viscAE", Av),
            ("ghats", ghf), ("Kv", np.asarray(Kv))]}
    return mesh, f, out


def test_final_node_fields_match_c_dump(setup):
    """Post-combine viscA / diffKt / diffKs reproduce the C final dump."""
    mesh, f, out = setup
    m = np.asarray(mesh.node_iface_mask)
    for name in ("viscA", "diffKt", "diffKs"):
        d = float(np.max(np.abs(out[name][m] - f[name][m])))
        print(f"\n{name} max|Δ|={d:.3e}  (scale {np.abs(f[name][m]).max():.3e})")
        assert d < 1e-12
    assert np.array_equal(out["Kv"], out["diffKt"])     # Kv = T-channel diffK


def test_viscAE_matches_c_dump(setup):
    """Element viscosity Av (3-vertex mean + bottom fill + minmix floor)."""
    mesh, f, out = setup
    m = np.asarray(mesh.elem_iface_mask)
    d = float(np.max(np.abs(out["viscAE"][m] - f["viscAE"][m])))
    print(f"\nviscAE max|Δ|={d:.3e}  (scale {np.abs(f['viscAE'][m]).max():.3e})")
    assert d < 1e-12


def test_ghats_zeroed_below_bl_match(setup):
    """ghats (combine zeroes it below the BL) — relative (huge dynamic range)."""
    mesh, f, out = setup
    nl = mesh.nl
    m = np.asarray(mesh.node_iface_mask)[:, :nl - 1]
    got, ref = out["ghats"][:, :nl - 1], f["ghats"]
    viol = float(np.max(np.abs(got[m] - ref[m]) - (1e-12 + 1e-12 * np.abs(ref[m]))))
    print(f"\nghats max|Δ|={np.abs(got[m]-ref[m]).max():.3e}  viol={viol:.3e}")
    assert viol <= 0.0


def test_enhance_assemble_ad_finite(setup):
    """d(ΣKv+ΣAv)/d(ri viscA/diffKt/diffKs, blmc, hbl) finite everywhere."""
    mesh, f, _ = setup
    cfg = kpp.KppConfig()
    bd = f["bldepth"]
    hbl = jnp.asarray(bd[:, 0]); caseA = jnp.asarray(bd[:, 4])
    kbl = jnp.asarray((bd[:, 1] - 1).astype(np.int32))
    dkm1 = jnp.asarray(f["dkm1"])
    ghats_in = jnp.asarray(np.pad(f["bl_ghats"], ((0, 0), (0, 1))))

    def loss(riA, riT, riS, bM0, hbl_):
        bM, bT, bS, gh = kpp.enhance(mesh, bM0, jnp.asarray(f["blmc_t"]),
                                     jnp.asarray(f["blmc_s"]), ghats_in, dkm1,
                                     riA, riT, riS, hbl_, caseA, kbl, cfg)
        Kv, Av, *_ = kpp.assemble_mixing(mesh, bM, bT, bS, gh, riA, riT, riS, kbl, cfg)
        return jnp.sum(Kv) + jnp.sum(Av)

    g = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(
        jnp.asarray(f["ri_viscA"]), jnp.asarray(f["ri_diffKt"]), jnp.asarray(f["ri_diffKs"]),
        jnp.asarray(f["blmc_m"]), hbl)
    assert all(bool(jnp.all(jnp.isfinite(gi))) for gi in g)
