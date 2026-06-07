"""Phase 6C Task K.5 gates — pre-step (dVsq/ustar/Bo) + dbsfc + bldepth.

The highest-risk KPP kernel (the OBL-depth bulk-Richardson search), gated by
**controlled replay**: the C-dumped bldepth inputs (dVsq/ustar/Bo/bvfreq/dbsfc/sw_3d/
sw_alpha — the last two added to the dump in K.5) are fed to :func:`kpp.bldepth` and
its outputs (hbl/kbl/bfsfc/stable/caseA) compared to the C ``bldepth`` dump. ``dbsfc``
(:func:`eos.compute_dbsfc`, the one missing EOS input) is gated against its own dump
via the GM dump's step-1 T/S (the same PHC IC). The pre-step formulas
(:func:`kpp.prestep`) are checked synthetically (dVsq=0 at the cold-start step 1;
ustar/Bo against hand values) + the ``ustar`` τ=0 safe-sqrt AD (the #1 priority).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import eos, kpp
from fesom_jax.io_dump import load_gm_dump, load_kpp_dump
from fesom_jax.mesh import load_mesh

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH_DIR = ROOT / "data" / "mesh_core2"
KPP_DUMP_DIR = ROOT / "data" / "kpp_dump_core2"
GM_DUMP_DIR = ROOT / "data" / "gm_dump_core2"

pytestmark = pytest.mark.skipif(
    not (CORE2_MESH_DIR.is_dir()
         and (KPP_DUMP_DIR / "kpp_dump_s1_bldepth_rank0.txt").is_file()
         and (KPP_DUMP_DIR / "kpp_dump_s1_sw_3d_rank0.txt").is_file()),
    reason=f"CORE2 mesh / KPP dump (incl. sw_3d) missing ({KPP_DUMP_DIR}); "
           "run jax_kpp_dump_core2.sh",
)


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(CORE2_MESH_DIR)


@pytest.fixture(scope="module")
def cfg():
    return kpp.KppConfig()


@pytest.fixture(scope="module")
def bld(mesh, cfg):
    """Run bldepth on the C-dumped inputs (controlled replay)."""
    f, _ = load_kpp_dump(KPP_DUMP_DIR, ["dVsq", "prestep", "ri_bvfreq", "dbsfc",
                                        "sw_3d", "sw_alpha", "bldepth"])
    wmt, wst = kpp.build_wscale_tables(cfg)
    out = kpp.bldepth(mesh, jnp.asarray(f["dVsq"]), jnp.asarray(f["prestep"][:, 0]),
                      jnp.asarray(f["prestep"][:, 1]), jnp.asarray(f["ri_bvfreq"]),
                      jnp.asarray(f["dbsfc"]), jnp.asarray(f["sw_3d"]),
                      jnp.asarray(f["sw_alpha"]), wmt, wst, cfg)
    return f, tuple(np.asarray(x) for x in out)


# ---------------------------------------------------------------------------
# dbsfc (eos.compute_dbsfc) — the missing EOS input
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (GM_DUMP_DIR / "gm_meta.txt").is_file(),
                    reason="GM dump (for step-1 T/S) missing")
def test_dbsfc_matches_c_dump(mesh):
    """compute_dbsfc reproduces the C dbsfc dump (driven by the GM dump's step-1 T/S —
    the same PHC IC; the EOS runs on the pre-mixing state so the scheme doesn't matter)."""
    gmf, _ = load_gm_dump(GM_DUMP_DIR)
    dbsfc = np.asarray(eos.compute_dbsfc(mesh, jnp.asarray(gmf["T"]), jnp.asarray(gmf["S"])))
    kf, _ = load_kpp_dump(KPP_DUMP_DIR, ["dbsfc"])
    m = np.asarray(mesh.node_iface_mask)
    d = float(np.max(np.abs(dbsfc[m] - kf["dbsfc"][m])))
    print(f"\ndbsfc max|Δ|={d:.3e}  (scale {np.abs(kf['dbsfc'][m]).max():.3e})")
    assert d < 1e-12
    # surface dbsfc is identically 0 (surface parcel at its own depth)
    nzmin = np.asarray(mesh.ulevels_nod2D) - 1
    assert np.max(np.abs(dbsfc[np.arange(len(nzmin)), nzmin])) == 0.0


# ---------------------------------------------------------------------------
# pre-step (dVsq / ustar / Bo)
# ---------------------------------------------------------------------------
def test_prestep_formulas(mesh, cfg):
    N, nl = int(mesh.nod2D), mesh.nl
    z = jnp.zeros
    sw_a = z((N, nl))
    # dVsq=0 when uv=0 (the cold-start step-1 state — matches the ~0 dump); ustar=0 at τ=0
    dV, us0, _ = kpp.prestep(mesh, z((N, nl, 2)), z((N, 2)), z(N), z(N), sw_a, z((N, nl)),
                             jnp.full((N, nl), 35.0), cfg)
    assert float(jnp.max(jnp.abs(dV))) == 0.0
    assert float(jnp.max(jnp.abs(us0))) == 0.0

    # ustar = sqrt( sqrt(τx²+τy²) / ρ0 ) — hand value
    st = z((N, 2)).at[:, 0].set(0.1)
    _, us, _ = kpp.prestep(mesh, z((N, nl, 2)), st, z(N), z(N), sw_a, z((N, nl)),
                           jnp.full((N, nl), 35.0), cfg)
    assert abs(float(us[0]) - ((0.1 ** 2) ** 0.5 / cfg.rho0) ** 0.5) < 1e-15

    # Bo = -g·(α·hf/VCPW + β·wf·S)
    aS, bS = jnp.full((N, nl), 2e-4), jnp.full((N, nl), 7.8e-4)
    _, _, bo = kpp.prestep(mesh, z((N, nl, 2)), z((N, 2)), jnp.full(N, 50.0),
                           jnp.full(N, 1e-6), aS, bS, jnp.full((N, nl), 35.0), cfg)
    exp = -cfg.g * (2e-4 * 50.0 / cfg.vcpw + 7.8e-4 * 1e-6 * 35.0)
    assert abs(float(bo[0]) - exp) < 1e-18


def test_prestep_ustar_safe_sqrt_ad(mesh, cfg):
    """d(Σustar)/d(stress) at τ=0 finite — the nested safe-sqrt (#1 KPP AD priority:
    ustar sits in u*³/u*⁴ denominators downstream)."""
    N, nl = int(mesh.nod2D), mesh.nl
    z = jnp.zeros

    def loss(stress):
        _, u, _ = kpp.prestep(mesh, z((N, nl, 2)), stress, z(N), z(N), z((N, nl)),
                              z((N, nl)), jnp.full((N, nl), 35.0), cfg)
        return jnp.sum(u)

    g = jax.grad(loss)(z((N, 2)))
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# bldepth — controlled-replay vs the C dump (hbl/kbl/bfsfc/stable/caseA)
# ---------------------------------------------------------------------------
def test_bldepth_matches_c_dump(bld):
    f, (hbl, kbl, bfsfc, stable, caseA) = bld
    ref = f["bldepth"]                                  # 0=hbl,1=kbl+1,2=bfsfc,3=stable,4=caseA
    assert float(np.max(np.abs(hbl - ref[:, 0]))) < 1e-9          # interp, scale ~790 m
    assert int(np.sum((kbl + 1) != ref[:, 1])) == 0              # discrete search EXACT
    assert float(np.max(np.abs(bfsfc - ref[:, 2]))) < 1e-12
    assert np.array_equal(stable, ref[:, 3])                     # Heaviside exact
    assert np.array_equal(caseA, ref[:, 4])


def test_bldepth_ad_finite(mesh, cfg):
    """d(Σhbl+Σbfsfc)/d(continuous inputs) finite everywhere — stop-grad kbl, safe-sqrt
    Vtsq, the differentiable hbl interp weight."""
    f, _ = load_kpp_dump(KPP_DUMP_DIR, ["dVsq", "prestep", "ri_bvfreq", "dbsfc",
                                        "sw_3d", "sw_alpha"])
    ustar = jnp.asarray(f["prestep"][:, 0])
    sw_alpha = jnp.asarray(f["sw_alpha"])
    wmt, wst = kpp.build_wscale_tables(cfg)

    def loss(dVsq, Bo, bv, dbsfc, sw):
        hbl, kbl, bfsfc, st, cA = kpp.bldepth(mesh, dVsq, ustar, Bo, bv, dbsfc, sw,
                                              sw_alpha, wmt, wst, cfg)
        return jnp.sum(hbl) + jnp.sum(bfsfc)

    g = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(
        jnp.asarray(f["dVsq"]), jnp.asarray(f["prestep"][:, 1]),
        jnp.asarray(f["ri_bvfreq"]), jnp.asarray(f["dbsfc"]), jnp.asarray(f["sw_3d"]))
    assert all(bool(jnp.all(jnp.isfinite(gi))) for gi in g)
