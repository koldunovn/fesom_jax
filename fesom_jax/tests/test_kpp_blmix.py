"""Phase 6C Task K.6 gate — KPP boundary-layer mixing coefficients (blmix).

The C's hardest controlled replay (it hit max|Δ|=3.18e-13). Feed :func:`kpp.blmix`
the C-dumped bldepth outputs (hbl/kbl/bfsfc/stable/caseA) + ri_iwmix outputs
(viscA/diffKt/diffKs = dcol) + ustar + the step-1 hnode (from the GM dump — static
full-cell linfs, same PHC IC), and match ``blmc_m``/``blmc_t``/``blmc_s``/``ghats``/
``dkm1`` to the C dump. ``ghats`` is gated RELATIVE (its ``cg/(ws·hbl)`` form reaches
~2e3 where the velocity scale →0 — the GM huge-dynamic-range pattern). Plus AD.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import kpp
from fesom_jax.io_dump import load_gm_dump, load_kpp_dump
from fesom_jax.mesh import load_mesh

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH_DIR = ROOT / "data" / "mesh_core2"
KPP_DUMP_DIR = ROOT / "data" / "kpp_dump_core2"
GM_DUMP_DIR = ROOT / "data" / "gm_dump_core2"

pytestmark = pytest.mark.skipif(
    not (CORE2_MESH_DIR.is_dir()
         and (KPP_DUMP_DIR / "kpp_dump_s1_blmc_m_rank0.txt").is_file()
         and (GM_DUMP_DIR / "gm_meta.txt").is_file()),
    reason=f"CORE2 mesh / KPP dump / GM dump (for hnode) missing; "
           "run jax_kpp_dump_core2.sh + jax_gm_dump_core2.sh",
)


@pytest.fixture(scope="module")
def setup():
    mesh = load_mesh(CORE2_MESH_DIR)
    cfg = kpp.KppConfig()
    f, _ = load_kpp_dump(KPP_DUMP_DIR, ["ri_viscA", "ri_diffKt", "ri_diffKs", "bldepth",
                                        "prestep", "blmc_m", "blmc_t", "blmc_s",
                                        "bl_ghats", "dkm1"])
    gmf, _ = load_gm_dump(GM_DUMP_DIR)
    bd = f["bldepth"]                                   # 0=hbl,1=kbl+1,2=bfsfc,3=stable,4=caseA
    wmt, wst = kpp.build_wscale_tables(cfg)
    out = kpp.blmix(
        mesh, jnp.asarray(gmf["hnode"]),
        jnp.asarray(f["ri_viscA"]), jnp.asarray(f["ri_diffKt"]), jnp.asarray(f["ri_diffKs"]),
        jnp.asarray(bd[:, 0]), jnp.asarray(bd[:, 2]), jnp.asarray(bd[:, 3]),
        jnp.asarray(bd[:, 4]), jnp.asarray((bd[:, 1] - 1).astype(np.int32)),
        jnp.asarray(f["prestep"][:, 0]), wmt, wst, cfg)
    return mesh, f, tuple(np.asarray(x) for x in out)


def test_blmc_matches_c_dump(setup):
    """blmc[3] (momentum/T/S) over the BL interfaces — the C's 3.18e-13 class."""
    mesh, f, (bM, bT, bS, gh, dkm1) = setup
    m = np.asarray(mesh.node_iface_mask)
    for name, got in [("blmc_m", bM), ("blmc_t", bT), ("blmc_s", bS)]:
        d = float(np.max(np.abs(got[m] - f[name][m])))
        print(f"\n{name} max|Δ|={d:.3e}  (scale {np.abs(f[name][m]).max():.3e})")
        assert d < 1e-12


def test_ghats_matches_c_dump_relative(setup):
    """ghats = (1−stable)·cg/(ws·hbl+ε): huge dynamic range (→2e3 where ws→0) ⇒ gate
    RELATIVE (atol + rtol·|ref|), like the GM neutral-slope."""
    mesh, f, (bM, bT, bS, gh, dkm1) = setup
    nl = mesh.nl
    m = np.asarray(mesh.node_iface_mask)[:, :nl - 1]
    got, ref = gh[:, :nl - 1], f["bl_ghats"]
    viol = float(np.max(np.abs(got[m] - ref[m]) - (1e-12 + 1e-12 * np.abs(ref[m]))))
    print(f"\nghats max|Δ|={np.abs(got[m]-ref[m]).max():.3e}  viol={viol:.3e}  "
          f"(scale {np.abs(ref[m]).max():.3e})")
    assert viol <= 0.0


def test_dkm1_matches_c_dump(setup):
    """dkm1[3] at kbl-1 (the enhance input)."""
    mesh, f, (bM, bT, bS, gh, dkm1) = setup
    d = float(np.max(np.abs(dkm1 - f["dkm1"])))
    print(f"\ndkm1 max|Δ|={d:.3e}  (scale {np.abs(f['dkm1']).max():.3e})")
    assert d < 1e-11


def test_blmix_ad_finite(setup):
    """d(Σblmc+Σghats+Σdkm1)/d(viscA,diffKt,diffKs,hbl,bfsfc) finite everywhere."""
    mesh, f, _ = setup
    cfg = kpp.KppConfig()
    gmf, _ = load_gm_dump(GM_DUMP_DIR)
    hnode = jnp.asarray(gmf["hnode"])
    bd = f["bldepth"]
    stable = jnp.asarray(bd[:, 3]); caseA = jnp.asarray(bd[:, 4])
    kbl = jnp.asarray((bd[:, 1] - 1).astype(np.int32)); ustar = jnp.asarray(f["prestep"][:, 0])
    wmt, wst = kpp.build_wscale_tables(cfg)

    def loss(viscA, diffKt, diffKs, hbl, bfsfc):
        a, b, c, g, d = kpp.blmix(mesh, hnode, viscA, diffKt, diffKs, hbl, bfsfc, stable,
                                  caseA, kbl, ustar, wmt, wst, cfg)
        return jnp.sum(a) + jnp.sum(b) + jnp.sum(c) + jnp.sum(g) + jnp.sum(d)

    g = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(
        jnp.asarray(f["ri_viscA"]), jnp.asarray(f["ri_diffKt"]), jnp.asarray(f["ri_diffKs"]),
        jnp.asarray(bd[:, 0]), jnp.asarray(bd[:, 2]))
    assert all(bool(jnp.all(jnp.isfinite(gi))) for gi in g)
