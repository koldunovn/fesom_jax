"""E1 gate (CPU): the NN-of-TKE perfect-model twin recipe — recover a KNOWN ``tke_nn`` instance
through the global adjoint (:func:`fesom_jax.calibrate.optimize` over the NN-weight pytree).

These pi-mesh unit tests guard the recovery *mechanics* that ``scripts/core2_paper_nn_twin.py`` runs
on CORE2 (the §3 hybrid-ML proof) — cheaply, through the **faithful TKE consumption site**
``tke.mixing_tke`` (TKE raises on the pi ``integrate`` path, so — exactly like the C1 sensitivity
seam test — the recipe is exercised at ``mixing_tke``; the full end-to-end ``integrate`` adjoint is
the GPU gate NN_TWIN_OK):

  * **NN→0 ≡ default** — a zero-last-layer trainee is bit-identical to default TKE (the safe start);
  * **adjoint recovers the NN** — Adam on the ``{'tke_nn': nn}`` pytree, minimizing the induced-mixing
    (Kv) misfit vs a seeded truth NN, collapses the misfit ⇒ the optimizer + the NN-weight gradient
    through the assembled mixing driver recover the truth's induced mixing;
  * **induced-mixing field recovered** — the recovered NN's c_k multiplier moves to the truth's where
    Kv constrains it (closer than the NN→0 start).

Token: NN_TWIN_RECIPE_OK. The CORE2 GPU gate is NN_TWIN_OK (``scripts/core2_paper_nn_twin.py``).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fesom_jax import ale, tke_nn
from fesom_jax.calibrate import optimize
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params
from fesom_jax.state import State
from fesom_jax.tke import TkeConfig, mixing_tke

RNG = np.random.default_rng(20260615)


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    return load_mesh()


@pytest.fixture(scope="module")
def twin(mesh):
    """Seed a tke>0 state so KappaM = c_k·mxl·√tke engages, inject a known truth NN → truth Kv,
    then recover a trainee NN from NN→0 by minimizing the Kv misfit through ``mixing_tke``."""
    st = State.rest(mesh)
    M = mesh.nod2D
    imask = np.asarray(mesh.node_iface_mask)
    tke_seed = jnp.asarray(np.where(imask, 1e-3, 0.0))
    stress_node = jnp.asarray(RNG.standard_normal((M, 2)) * 0.05)
    _, Z3d = ale.live_geometry(mesh, st.hnode)
    wet = jnp.asarray(imask)

    def kv_of(nn):
        p = dataclasses.replace(Params.defaults(), tke_nn=nn)
        Kv, _, _, _ = mixing_tke(mesh, st.uv, st.bvfreq, tke_seed, stress_node, st.hnode,
                                 TkeConfig(), p, dt=1800.0, Z3d=Z3d)
        return jnp.where(wet, Kv, 0.0)

    truth_nn = tke_nn.init_tke_nn(jax.random.PRNGKey(11), zero_last=False, w_scale=1.0)
    # amplify the output layer so the induced multiplier (hence Kv) is non-trivially perturbed
    truth_nn = dataclasses.replace(truth_nn, Ws=truth_nn.Ws[:-1] + (truth_nn.Ws[-1] * 3.0,))
    truth_Kv = kv_of(truth_nn)

    # default (NN-off) Kv and the NN→0 trainee Kv must coincide (the bit-identity start)
    base_Kv = jnp.where(wet, mixing_tke(mesh, st.uv, st.bvfreq, tke_seed, stress_node, st.hnode,
                                        TkeConfig(), Params.defaults(), dt=1800.0, Z3d=Z3d)[0], 0.0)
    trainee0 = tke_nn.init_tke_nn(jax.random.PRNGKey(404), zero_last=True)

    def loss(d):
        diff = kv_of(d["tke_nn"]) - truth_Kv
        return jnp.sum(diff * diff)

    J0 = float(loss({"tke_nn": trainee0}))
    opt = optax.adam(optax.cosine_decay_schedule(0.05, 80))
    rec_params, _ = optimize(loss, {"tke_nn": trainee0}, opt, n_iters=80, keep_params=False)
    rec_nn = rec_params["tke_nn"]
    Jf = float(loss(rec_params))

    # induced c_k multiplier on the seeded features (the same the driver compares)
    nl = mesh.nl
    k = jnp.arange(nl)[None, :]
    nzmin = (mesh.ulevels_nod2D - 1)[:, None]
    nzmax = (mesh.nlevels_nod2D - 1)[:, None]
    is_int = (k >= nzmin + 1) & (k <= nzmax - 1)
    feats = tke_nn.column_features(jnp.zeros(M), mesh.coriolis_node, mesh.depth,
                                   jnp.where(is_int, st.bvfreq, 0.0), jnp.zeros((M, nl)),
                                   tke_seed, is_int)
    m_truth = np.asarray(tke_nn.multiplier(truth_nn, feats)[:, 0])
    m_rec = np.asarray(tke_nn.multiplier(rec_nn, feats)[:, 0])
    return dict(J0=J0, Jf=Jf, base_Kv=np.asarray(base_Kv), zero_Kv=np.asarray(kv_of(trainee0)),
                truth_Kv=np.asarray(truth_Kv), m_truth=m_truth, m_rec=m_rec)


def test_nn_zero_start_bit_identical_to_default(twin):
    """NN→0 (zero last layer) Kv is bit-for-bit the default-TKE Kv (the safe-start invariant)."""
    np.testing.assert_array_equal(twin["zero_Kv"], twin["base_Kv"])


def test_adjoint_recovers_nn_mixing(twin):
    """Adam through the mixing-driver adjoint over the NN-weight pytree collapses the Kv misfit:
    the optimizer recovers the truth NN's induced mixing (the §3 end-to-end-training mechanic)."""
    assert twin["truth_Kv"].max() > 0, "degenerate truth (no mixing to recover)"
    assert twin["Jf"] < 0.1 * twin["J0"], f"Kv misfit not collapsed: {twin['Jf']:.3e} vs {twin['J0']:.3e}"


def test_induced_multiplier_recovered(twin):
    """The recovered NN's c_k multiplier is much closer to the truth's than the NN→0 start (1.0)."""
    m_truth, m_rec = twin["m_truth"], twin["m_rec"]
    err_rec = np.sqrt(np.mean((m_rec - m_truth) ** 2))
    err_start = np.sqrt(np.mean((1.0 - m_truth) ** 2))
    assert err_rec < 0.5 * err_start, f"multiplier not recovered: rec err {err_rec:.3f} vs start {err_start:.3f}"


def test_nn_twin_recipe_ok_token(twin):
    """Aggregate gate — bit-identical start + adjoint recovery; prints the token."""
    assert np.array_equal(twin["zero_Kv"], twin["base_Kv"])
    assert twin["Jf"] < 0.1 * twin["J0"]
    print("NN_TWIN_RECIPE_OK")
