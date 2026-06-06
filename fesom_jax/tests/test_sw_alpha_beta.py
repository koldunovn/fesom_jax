"""Task G.1 gate — sw_alpha_beta (McDougall 1987), substep 2 / Phase 6B.

``eos.compute_sw_alpha_beta`` is a pure per-node polynomial map (thermal expansion
``sw_alpha``, saline contraction ``sw_beta``) consumed by GM/Redi (and KPP). The
bit-exact-vs-C gate comes with the GM dump (the C ``fesom_gm_dump`` hook); this file
establishes correctness against an **independent numpy transcription** of
``fesom_eos.c:336-372`` (a different code path — numpy, not jnp), exercises the
``s35`` salinity terms with synthetic varied S (the pi blob IC has S≡35 → those
terms vanish), checks physical ranges + below-bottom masking, and gates AD
finiteness (the masked-NaN probe).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import eos, ic
from fesom_jax.io_dump import load_gm_dump
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh

CORE2_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
GM_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "gm_dump_core2"


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (Task 0.3)")
    return load_mesh()


def _np_sw_alpha_beta(mesh, T, S):
    """Independent numpy transcription of fesom_eos.c:336-372 (McDougall 1987).

    Vectorized over [nod2D, nl]; masked to the layer range. Deliberately a separate
    implementation from eos.compute_sw_alpha_beta to catch transcription slips.
    """
    T = np.asarray(T, np.float64)
    S = np.asarray(S, np.float64)
    t1 = T * 1.00024
    s1 = S
    Z = np.asarray(mesh.Z, np.float64)
    Zp = np.concatenate([Z, Z[-1:]])                       # (nl-1,) -> (nl,)
    p1 = np.abs(Zp)[None, :]                               # (1, nl)

    t1_2, t1_3, t1_4 = t1 * t1, t1 ** 3, t1 ** 4
    p1_2, p1_3 = p1 * p1, p1 ** 3
    s35 = s1 - 35.0
    s35_2 = s35 * s35

    beta = (
        0.785567e-3
        - 0.301985e-5 * t1
        + 0.555579e-7 * t1_2
        - 0.415613e-9 * t1_3
        + s35 * (-0.356603e-6 + 0.788212e-8 * t1
                 + 0.408195e-10 * p1 - 0.602281e-15 * p1_2)
        + s35_2 * (0.515032e-8)
        + p1 * (-0.121555e-7 + 0.192867e-9 * t1 - 0.213127e-11 * t1_2)
        + p1_2 * (0.176621e-12 - 0.175379e-14 * t1)
        + p1_3 * (0.121551e-17)
    )
    a_over_b = (
        0.665157e-1
        + 0.170907e-1 * t1
        - 0.203814e-3 * t1_2
        + 0.298357e-5 * t1_3
        - 0.255019e-7 * t1_4
        + s35 * (0.378110e-2 - 0.846960e-4 * t1
                 - 0.164759e-6 * p1 - 0.251520e-11 * p1_2)
        + s35_2 * (-0.678662e-5)
        + p1 * (0.380374e-4 - 0.933746e-6 * t1 + 0.791325e-8 * t1_2)
        + p1_2 * t1_2 * (0.512857e-12)
        - p1_3 * (0.302285e-13)
    )
    m = np.asarray(mesh.node_layer_mask)
    return np.where(m, a_over_b * beta, 0.0), np.where(m, beta, 0.0)


def _synthetic_TS(mesh, seed=0):
    """Physical varied T/S over the layer range (so the s35 terms are exercised)."""
    rng = np.random.default_rng(seed)
    shp = (mesh.nod2D, mesh.nl)
    T = rng.uniform(-2.0, 30.0, shp)
    S = rng.uniform(30.0, 38.0, shp)
    m = np.asarray(mesh.node_layer_mask)
    return jnp.asarray(np.where(m, T, 0.0)), jnp.asarray(np.where(m, S, 0.0))


# --------------------------------------------------------------------------
# Bit-exact vs the independent numpy reference
# --------------------------------------------------------------------------
def test_matches_numpy_ref_blob_ic(mesh):
    """Realistic pi IC (constant + T-blob, S≡35). S35 terms vanish; the T/p path
    must still match the numpy reference to map-class (≈ bit-exact, a pure map)."""
    st = ic.initial_state(mesh)
    a, b = eos.compute_sw_alpha_beta(mesh, st.T, st.S)
    a_ref, b_ref = _np_sw_alpha_beta(mesh, st.T, st.S)
    assert np.max(np.abs(np.asarray(a) - a_ref)) < 1e-18
    assert np.max(np.abs(np.asarray(b) - b_ref)) < 1e-18


def test_matches_numpy_ref_synthetic(mesh):
    """Varied T/S exercises every polynomial term (s35, s35², all p1 powers)."""
    T, S = _synthetic_TS(mesh)
    a, b = eos.compute_sw_alpha_beta(mesh, T, S)
    a_ref, b_ref = _np_sw_alpha_beta(mesh, T, S)
    # Map class; eager JAX vs numpy on float64 should be ~1e-15 relative.
    assert np.max(np.abs(np.asarray(a) - a_ref)) < 1e-15
    assert np.max(np.abs(np.asarray(b) - b_ref)) < 1e-15


# --------------------------------------------------------------------------
# Bit-exact vs the C GM-ON dump (all-node, CORE2) — the Path-A gate
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not (CORE2_MESH_DIR.is_dir() and (GM_DUMP_DIR / "gm_meta.txt").is_file()),
    reason=f"CORE2 mesh / GM dump missing ({GM_DUMP_DIR}); run jax_gm_dump_core2.sh",
)
def test_sw_alpha_beta_matches_c_dump():
    """Feed the C's dumped step-1 T/S to the JAX kernel; match ``sw_alpha``/
    ``sw_beta`` over ALL 126858 CORE2 nodes (pure per-node MAP → bit-exact class)."""
    mesh_c2 = load_mesh(CORE2_MESH_DIR)
    fields, meta = load_gm_dump(GM_DUMP_DIR)
    T_c = jnp.asarray(fields["T"])          # (N, nl), the C's step-1 T input
    S_c = jnp.asarray(fields["S"])
    a, b = eos.compute_sw_alpha_beta(mesh_c2, T_c, S_c)
    a, b = np.asarray(a), np.asarray(b)
    a_c, b_c = fields["sw_alpha"], fields["sw_beta"]
    m = np.asarray(mesh_c2.node_layer_mask)

    da = np.max(np.abs(a[m] - a_c[m]))
    db = np.max(np.abs(b[m] - b_c[m]))
    print(f"\nsw_alpha max|Δ|={da:.3e}  sw_beta max|Δ|={db:.3e} over {m.sum()} wet lanes")
    assert da < 1e-15 and db < 1e-15       # MAP class (per-node polynomial)
    # below-bottom: the C calloc's the array (0 at step 1); JAX masks to 0.
    assert np.all(a_c[~m] == 0.0) and np.all(a[~m] == 0.0)
    assert np.all(b_c[~m] == 0.0) and np.all(b[~m] == 0.0)


# --------------------------------------------------------------------------
# Physical sanity + masking
# --------------------------------------------------------------------------
def test_physical_ranges(mesh):
    """On wet lanes, beta ≈ 7-8e-4 /(g/kg) and alpha ≈ 0..4e-4 /K (warm→cold)."""
    T, S = _synthetic_TS(mesh, seed=1)
    a, b = eos.compute_sw_alpha_beta(mesh, T, S)
    m = np.asarray(mesh.node_layer_mask)
    bw = np.asarray(b)[m]
    aw = np.asarray(a)[m]
    assert np.all(bw > 5e-4) and np.all(bw < 1e-3)
    assert np.all(aw > -1e-4) and np.all(aw < 5e-4)


def test_masked_below_bottom(mesh):
    """Both coefficients are exactly 0 below the bottom (the layer mask)."""
    T, S = _synthetic_TS(mesh, seed=2)
    a, b = eos.compute_sw_alpha_beta(mesh, T, S)
    invalid = ~np.asarray(mesh.node_layer_mask)
    assert np.all(np.asarray(a)[invalid] == 0.0)
    assert np.all(np.asarray(b)[invalid] == 0.0)


# --------------------------------------------------------------------------
# AD — the masked-NaN probe (must be finite everywhere, incl. dry lanes)
# --------------------------------------------------------------------------
def test_ad_finite_wrt_T_and_S(mesh):
    """d(Σsw_alpha)/d(T₀ field) and d(Σsw_beta)/d(S₀ field) finite EVERYWHERE
    (incl. below-bottom padding) — the polynomial is smooth, so this is the
    masked-NaN baseline GM builds on. Nonzero on wet, 0 on masked."""
    T0, S0 = _synthetic_TS(mesh, seed=3)
    m = np.asarray(mesh.node_layer_mask)

    def loss_a(T):
        a, _ = eos.compute_sw_alpha_beta(mesh, T, S0)
        return jnp.sum(a)

    def loss_b(S):
        _, b = eos.compute_sw_alpha_beta(mesh, T0, S)
        return jnp.sum(b)

    gT = np.asarray(jax.grad(loss_a)(T0))
    gS = np.asarray(jax.grad(loss_b)(S0))
    assert np.all(np.isfinite(gT)) and np.all(np.isfinite(gS))
    # masked lanes carry no gradient; wet lanes do.
    assert np.all(gT[~m] == 0.0) and np.all(gS[~m] == 0.0)
    assert np.any(gT[m] != 0.0) and np.any(gS[m] != 0.0)
