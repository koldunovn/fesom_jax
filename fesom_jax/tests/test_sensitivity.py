"""C1 gate: the field-leaf sensitivity seam — promote a scalar :class:`~fesom_jax.params.Params`
leaf to a ``[nod2D]`` field and take ONE backward pass (the §1 sensitivity-map machinery).

These CPU unit tests (pi mesh) guard exactly what ``scripts/core2_paper_sensitivity.py`` does on
CORE2: ``calibrate.build_params`` widens a tunable to an array leaf, the model differentiates it,
and the gradient is a finite ``[nod2D]`` sensitivity map. Per the plan's Testing Strategy — *field-
leaf misfit + backward on a tiny state, masked-NaN clean*:

  * **k_gm** ``[nod2D]`` field through the assembled ``integrate`` (GM/Redi live): the map is finite
    EVERYWHERE (the AD masked-NaN rule), nonzero on wet nodes, the scalar adjoint == ``Σ map`` == FD
    (the uniform-broadcast identity + the existing grad-gate proof), and a per-node FD spot-check
    agrees;
  * **tke_c_k** ``[nod2D, 1]`` field through the ``tke.mixing_tke`` driver (TKE raises on the pi
    integrate path — ``mixing_tke`` is the faithful consumption site): finite, masked-clean, nonzero;
  * **build_params** preserves field-leaf shapes and leaves the untouched leaves at scalar defaults.

Token: SENSITIVITY_SEAM_OK. The CORE2 maps + the adjoint↔EKI cross-check are the GPU gate
(SENSITIVITY_MAP_OK, ``scripts/core2_paper_sensitivity.py`` / ``scripts/fig_sensitivity.py``).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import ale, forcing, ic, ssh
from fesom_jax.calibrate import build_params
from fesom_jax.gm import GMConfig
from fesom_jax.integrate import integrate
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params
from fesom_jax.state import State
from fesom_jax.tke import TkeConfig, mixing_tke

DT = 100.0
N = 4                                   # short window (the pi GM scatter compile is the cost)
BAND = 10                               # upper-ocean layer band for the k_gm T metric
K_GM0 = 1000.0
H_SWEEP = (1e-2, 1e-3, 1e-4, 1e-5)
RNG = np.random.default_rng(20260614)


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    return load_mesh()


# ==========================================================================
# k_gm [nod2D] field through the assembled integrate (GM/Redi live)
# ==========================================================================
def _kgm_band_loss(mesh):
    """``loss(k_gm_field[nod2D]) -> mean upper-ocean T`` after an N-step GM window. k_gm and the
    C-synced redi_kmax both derive from the field (the grad-gate convention)."""
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress = forcing.surface_stress(mesh)
    st0 = ic.initial_state(mesh)
    node_mask = jnp.asarray(mesh.node_layer_mask)
    band = node_mask & (jnp.arange(mesh.nl)[None, :] < BAND)
    nband = jnp.sum(band)

    def loss(kgm_field):
        p = build_params({"k_gm": kgm_field, "redi_kmax": kgm_field})
        fin = integrate(st0, mesh, op, stress, n_steps=N, params=p, dt=DT, gm_cfg=GMConfig())
        return jnp.sum(jnp.where(band, fin.T, 0.0)) / nband

    return loss


@pytest.fixture(scope="module")
def kgm(mesh):
    """Compile the GM forward + backward ONCE; share the field map across the k_gm tests."""
    loss = _kgm_band_loss(mesh)
    loss_j = jax.jit(loss)
    base = jnp.full((mesh.nod2D,), K_GM0, jnp.float64)
    gfield = np.asarray(jax.jit(jax.grad(loss))(base))
    return dict(loss_j=loss_j, gfield=gfield, base=base, M=mesh.nod2D)


def test_kgm_field_map_finite_nonzero(kgm, mesh):
    """The [nod2D] sensitivity map is finite EVERYWHERE (masked-NaN rule) + nonzero (live path)."""
    g = kgm["gfield"]
    assert g.shape == (mesh.nod2D,)                       # field-leaf shape preserved
    assert np.all(np.isfinite(g)), f"{int((~np.isfinite(g)).sum())} non-finite map entries"
    assert np.sum(g != 0.0) > 0, "k_gm sensitivity map identically zero"


def test_kgm_scalar_adjoint_equals_sum_and_matches_fd(kgm):
    """The scalar adjoint == Σ(field map) (uniform broadcast) == FD (the grad-gate proof). A
    uniform field θ·1 ⇒ dJ/dθ = Σ_x ∂J/∂θ(x); FD-verify the sum with a step sweep (plateau)."""
    g_sum = float(np.sum(kgm["gfield"]))
    lj = kgm["loss_j"]
    M = kgm["M"]

    def jf(s):
        return float(lj(jnp.full((M,), s, jnp.float64)))

    rels = []
    for h in H_SWEEP:
        fd = (jf(K_GM0 * (1.0 + h)) - jf(K_GM0 * (1.0 - h))) / (K_GM0 * 2.0 * h)
        rels.append(abs(g_sum - fd) / max(abs(fd), 1e-300))
    assert np.isfinite(g_sum) and g_sum != 0.0
    assert min(rels) < 1e-5, f"scalar adjoint==FD plateau {min(rels):.2e} ≥ 1e-5"


def test_kgm_field_fd_spotcheck(kgm):
    """Per-node FD spot-check at the most-sensitive node — the map value matches a central FD of
    the basin-mean functional w.r.t. that single node's k_gm."""
    g = kgm["gfield"]
    lj = kgm["loss_j"]
    M = kgm["M"]
    idx = int(np.argmax(np.abs(g)))
    ei = jnp.asarray(np.eye(1, M, idx).reshape(M))
    hn = 10.0
    fd = float((lj(kgm["base"] + hn * ei) - lj(kgm["base"] - hn * ei)) / (2.0 * hn))
    assert np.sign(fd) == np.sign(g[idx]), "spot-check sign mismatch"
    assert abs(g[idx] - fd) / max(abs(fd), 1e-300) < 1e-3


# ==========================================================================
# tke_c_k [nod2D, 1] field through the mixing_tke driver (the faithful TKE site)
# ==========================================================================
@pytest.fixture(scope="module")
def ck(mesh):
    """``d(Σ Kv)/d(tke_c_k field)`` through ``tke.mixing_tke`` — c_k as a ``[nod2D, 1]`` field
    (broadcast over levels). A seeded tke>0 so KappaM = c_k·mxl·√tke engages (else d/dc_k=0)."""
    st = State.rest(mesh)
    M = mesh.nod2D
    imask = np.asarray(mesh.node_iface_mask)
    tke_seed = jnp.asarray(np.where(imask, 1e-3, 0.0))
    stress_node = jnp.asarray(RNG.standard_normal((M, 2)) * 0.05)
    _, Z3d = ale.live_geometry(mesh, st.hnode)

    def loss(ck_field):                                  # ck_field [M, 1]
        p = dataclasses.replace(Params.defaults(), tke_c_k=ck_field)
        Kv, _, _, _ = mixing_tke(mesh, st.uv, st.bvfreq, tke_seed, stress_node, st.hnode,
                                 TkeConfig(), p, dt=1800.0, Z3d=Z3d)
        return jnp.sum(Kv)

    ck0 = jnp.full((M, 1), float(Params.defaults().tke_c_k), jnp.float64)
    g = np.asarray(jax.grad(loss)(ck0))
    return dict(g=g, M=M)


def test_ck_field_finite_nonzero_masked_clean(ck, mesh):
    """The c_k field gradient is finite EVERYWHERE (the TKE safe-sqrt/pow/clamp guards survive a
    field leaf — masked-NaN rule), keeps the ``[nod2D, 1]`` shape, and is nonzero."""
    g = ck["g"]
    assert g.shape == (mesh.nod2D, 1)                    # field-leaf shape (broadcast over levels)
    assert np.all(np.isfinite(g)), f"{int((~np.isfinite(g)).sum())} non-finite c_k grad entries"
    assert np.sum(g != 0.0) > 0, "c_k sensitivity map identically zero"


# ==========================================================================
# build_params field-leaf seam (no model — fast)
# ==========================================================================
def test_build_params_field_leaf_seam(mesh):
    """``build_params`` accepts field leaves (shapes preserved), leaves the untouched leaves at
    scalar defaults, and the gradient through it returns the field shape."""
    M = mesh.nod2D
    kf = jnp.asarray(RNG.standard_normal(M))
    ckf = jnp.full((M, 1), 0.1, jnp.float64)
    p = build_params({"k_gm": kf, "redi_kmax": kf, "tke_c_k": ckf})
    assert p.k_gm.shape == (M,) and p.redi_kmax.shape == (M,)
    assert p.tke_c_k.shape == (M, 1)
    assert jnp.ndim(p.tke_c_eps) == 0 and jnp.ndim(p.k_ver) == 0     # untouched ⇒ scalar defaults

    g = jax.grad(lambda f: jnp.sum(build_params({"k_gm": f}).k_gm ** 2))(kf)
    assert np.asarray(g).shape == (M,)
    np.testing.assert_allclose(np.asarray(g), 2.0 * np.asarray(kf), rtol=0, atol=1e-12)


def test_sensitivity_seam_ok_token(kgm, ck):
    """Aggregate gate — both field leaves flow a finite, nonzero backward; prints the token."""
    assert np.all(np.isfinite(kgm["gfield"])) and np.sum(kgm["gfield"] != 0.0) > 0
    assert np.all(np.isfinite(ck["g"])) and np.sum(ck["g"] != 0.0) > 0
    print("SENSITIVITY_SEAM_OK")
