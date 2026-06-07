"""Phase 6C Tasks K.3 + K.4 gates — KPP interior mixing + the ddmix gate.

* **K.3** — :func:`kpp.ri_iwmix` (shear-Ri + background) vs the C ``ri_viscA``/
  ``ri_diffKt``/``ri_diffKs`` dump. The step-1 dump has ``uvnode=0`` (cold start) ⇒
  ``shear=0`` ⇒ ``frit∈{0,1}`` (a pure ``sign(N²)`` map), which the replay reproduces
  **bit-exactly** (it exercises the edge copies + masking + the static-instability
  branch); a **synthetic** test exercises the intermediate cubic ``frit`` the
  ``shear=0`` dump cannot. Plus AD finiteness.
* **K.4** — the ``ddmix`` double-diffusion gate is OFF in CORE2 (a no-op); enabling it
  raises (the C ``#error`` analog).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import kpp
from fesom_jax.io_dump import load_kpp_dump
from fesom_jax.mesh import load_mesh

CORE2_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
KPP_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "kpp_dump_core2"

pytestmark = pytest.mark.skipif(
    not (CORE2_MESH_DIR.is_dir()
         and (KPP_DUMP_DIR / "kpp_dump_s1_ri_viscA_rank0.txt").is_file()),
    reason=f"CORE2 mesh / KPP dump missing ({KPP_DUMP_DIR}); run jax_kpp_dump_core2.sh",
)


@pytest.fixture(scope="module")
def setup():
    mesh = load_mesh(CORE2_MESH_DIR)
    cfg = kpp.KppConfig()
    fields, _ = load_kpp_dump(KPP_DUMP_DIR,
                              ["ri_bvfreq", "ri_viscA", "ri_diffKt", "ri_diffKs"])
    bv = jnp.asarray(fields["ri_bvfreq"])
    N, nl = bv.shape
    uvnode0 = jnp.zeros((N, nl, 2))                    # step-1 uv=0 ⇒ shear=0
    vA, dT, dS = kpp.ri_iwmix(mesh, uvnode0, bv, cfg)
    return mesh, cfg, fields, bv, (np.asarray(vA), np.asarray(dT), np.asarray(dS))


def test_ri_iwmix_matches_c_dump(setup):
    """viscA/diffKt/diffKs reproduce the C ri_iwmix dump on the iface range."""
    mesh, cfg, f, bv, (vA, dT, dS) = setup
    m = np.asarray(mesh.node_iface_mask)
    for name, got in [("viscA", vA), ("diffKt", dT), ("diffKs", dS)]:
        d = float(np.max(np.abs(got[m] - f["ri_" + name][m])))
        print(f"\nri_{name} max|Δ|={d:.3e}  (scale {np.abs(f['ri_'+name][m]).max():.3e})")
        assert d < 1e-12
    assert np.array_equal(dT, dS)                      # Kv0_const ⇒ T and S identical


def test_ri_iwmix_cubic_frit(setup):
    """The intermediate cubic frit=(1−min(Ri/Riinfty,1)²)³ (the shear=0 dump only
    samples frit∈{0,1}): a synthetic linear u-profile sets a controlled shear, N²
    targets Ri=0.4·… across the curve."""
    mesh, cfg, f, bv, _ = setup
    N, nl = bv.shape
    Z = np.asarray(mesh.Z)
    n = int(np.argmax(np.asarray(mesh.nlevels_nod2D)))  # deepest column, nzmin=0
    for Ri_target in (0.2 * cfg.riinfty, 0.5 * cfg.riinfty, 0.9 * cfg.riinfty):
        nz = 5
        dzc = Z[nz - 1] - Z[nz]
        c = 0.01
        uv = np.zeros((N, nl, 2))
        uv[n, :, 0] = c * np.arange(nl)                 # du = -c ⇒ shear = (c/dz)²
        shear = (c / dzc) ** 2
        bv2 = np.zeros((N, nl))
        bv2[n, nz] = Ri_target * shear                  # Ri = N²/shear (epsln negligible)
        vA2, _, _ = kpp.ri_iwmix(mesh, jnp.asarray(uv), jnp.asarray(bv2), cfg)
        ratio = min(max(Ri_target, 0.0) / cfg.riinfty, 1.0)
        frit = (1.0 - ratio * ratio) ** 3
        expect = cfg.visc_sh_limit * frit + cfg.a_bg
        assert abs(float(vA2[n, nz]) - expect) < 1e-14


def test_ri_iwmix_ad_finite(setup):
    """d(Σ viscA + Σ diffKt)/d(uvnode), /d(bvfreq) finite everywhere (incl. masked)."""
    mesh, cfg, f, bv, _ = setup
    N, nl = bv.shape
    uvnode0 = jnp.zeros((N, nl, 2))

    def loss(uvn, bvf):
        a, b, _ = kpp.ri_iwmix(mesh, uvn, bvf, cfg)
        return jnp.sum(a) + jnp.sum(b)

    gu, gb = jax.grad(loss, argnums=(0, 1))(uvnode0, bv)
    assert bool(jnp.all(jnp.isfinite(gu)))
    assert bool(jnp.all(jnp.isfinite(gb)))


# ---------------------------------------------------------------------------
# K.4 — ddmix gate (CORE2 double_diffusion=False ⇒ no-op; enabling it raises)
# ---------------------------------------------------------------------------
def test_ddmix_gate_off_by_default():
    assert kpp.KppConfig().double_diffusion is False
    assert kpp.KppConfig().use_kpp_nonlclflx is False
    kpp.assert_no_double_diffusion(kpp.KppConfig())     # no-op


def test_ddmix_gate_raises_if_enabled():
    with pytest.raises(NotImplementedError, match="double_diffusion"):
        kpp.assert_no_double_diffusion(kpp.KppConfig(double_diffusion=True))
