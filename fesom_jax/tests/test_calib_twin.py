"""D1 gate (CPU): the perfect-model ``k_gm`` twin recipe — grid-scan the misfit bowl, then
recover an injected truth with the global adjoint (:func:`fesom_jax.calibrate.optimize`).

These pi-mesh unit tests guard exactly what ``scripts/paper/core2_paper_calib_twin.py`` does on CORE2
(the §2 adjoint-as-optimizer proof), cheaply and config-agnostically on the differentiable
zstar+GM path (the GPU gate runs the FULL zstar+TKE+mEVP+GM model; the recovery *mechanics* are
the same):

  * **bowl well-posedness** — ``J(k)=‖T(k)−T(truth)‖²_w`` is a convex bowl whose argmin sits at the
    injected truth and which decreases monotonically toward it from both sides (the forward-only
    check the script runs FIRST, before any backward is trusted);
  * **adjoint recovery** — cosine-decay Adam on the normalized leaf ``u=k_gm/1000`` recovers the
    truth from a wrong start: the misfit collapses and the recovered value lands near the truth.

A perfect-model twin is well-posed at any window (J=0 at the truth), so a tiny N=4 pi window
suffices. Token: TWIN_RECIPE_OK. The CORE2 GPU gate is TWIN_RECOVER_OK
(``scripts/paper/core2_paper_calib_twin.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fesom_jax import forcing, ic, ssh
from fesom_jax.calibrate import build_params, optimize
from fesom_jax.gm import GMConfig
from fesom_jax.integrate import integrate
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh

DT = 100.0
N = 4                                    # short window (the pi GM scatter compile is the cost)
BAND = 10                                # upper-ocean layer band for the T metric
TRUTH = 2000.0                           # injected k_gm
INIT = 1000.0                            # recovery start
GRID = np.arange(1000.0, 3001.0, 250.0)  # brackets truth (2000) on both sides


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR}")
    return load_mesh()


@pytest.fixture(scope="module")
def twin(mesh):
    """Compile ONE GM forward (k_gm -> upper-ocean T), inject truth, run the bowl scan + recovery
    once; share across the recipe tests. ``k_gm`` and the C-synced ``redi_kmax`` ride the scalar."""
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress = forcing.surface_stress(mesh)
    st0 = ic.initial_state(mesh)
    node_mask = jnp.asarray(mesh.node_layer_mask)
    w = jnp.where(node_mask & (jnp.arange(mesh.nl)[None, :] < BAND), jnp.asarray(mesh.area), 0.0)

    @jax.jit
    def model_T(k_gm):
        p = build_params({"k_gm": k_gm, "redi_kmax": k_gm})
        fin = integrate(st0, mesh, op, stress, n_steps=N, params=p, dt=DT, gm_cfg=GMConfig())
        return fin.T

    def wmse(T, truth):
        d = T - truth
        return jnp.sum(w * d * d) / jnp.sum(w)

    truth = model_T(jnp.asarray(TRUTH, jnp.float64))
    Js = np.array([float(wmse(model_T(jnp.asarray(k, jnp.float64)), truth)) for k in GRID])

    # recover via cosine-decay Adam on u = k_gm/1000 (scale-free lr), loss normalized by the start
    # misfit J0 so Adam's eps (1e-8) doesn't swamp the TINY GM short-window gradient (J0 ~ 1e-12).
    def raw_loss(d):
        return wmse(model_T(1000.0 * d["u"]), truth)

    J0 = float(Js[int(np.argmin(np.abs(GRID - INIT)))])         # raw misfit at the start
    opt = optax.adam(optax.cosine_decay_schedule(0.05, 60))
    rec_params, _ = optimize(lambda d: raw_loss(d) / J0, {"u": jnp.asarray(INIT / 1000.0, jnp.float64)},
                             opt, n_iters=60)
    k_rec = 1000.0 * float(rec_params["u"])
    return dict(grid=GRID, Js=Js, k_rec=k_rec, J_init=J0, J_final=float(raw_loss(rec_params)))


def test_bowl_argmin_at_truth(twin):
    """Forward-only bowl: argmin sits exactly at the injected truth (perfect model ⇒ J=0 there)."""
    grid, Js = twin["grid"], twin["Js"]
    assert np.all(np.isfinite(Js))
    assert abs(float(grid[int(np.argmin(Js))]) - TRUTH) < 1e-6, "bowl argmin not at the truth"


def test_bowl_monotone_toward_truth(twin):
    """The bowl decreases monotonically approaching the truth from BOTH sides (convex, well-posed —
    not just a lucky J=0 hit at the truth grid point)."""
    grid, Js = twin["grid"], twin["Js"]
    it = int(np.argmin(np.abs(grid - TRUTH)))
    left, right = Js[: it + 1], Js[it:]
    assert np.all(np.diff(left) < 0), f"left arm not decreasing toward truth: {left}"
    assert np.all(np.diff(right) > 0), f"right arm not increasing past truth: {right}"


def test_adjoint_recovers_truth(twin):
    """Cosine-decay Adam through the global adjoint recovers the injected k_gm: the misfit collapses
    and the recovered value lands near the truth (the CORE2 gate's tight 2% bar; loose here for the
    tiny pi window)."""
    k_rec, J0, Jf = twin["k_rec"], twin["J_init"], twin["J_final"]
    assert abs(k_rec - TRUTH) < abs(INIT - TRUTH), "recovery did not move toward the truth"
    assert Jf < 0.05 * J0, f"misfit not collapsed: {Jf:.3e} vs init {J0:.3e}"
    assert abs(k_rec - TRUTH) / TRUTH < 0.1, f"recovered k_gm={k_rec:.1f} not within 10% of {TRUTH}"


def test_twin_recipe_ok_token(twin):
    """Aggregate gate — bowl well-posed + adjoint recovers; prints the token."""
    assert abs(float(twin["grid"][int(np.argmin(twin["Js"]))]) - TRUTH) < 1e-6
    assert twin["J_final"] < 0.05 * twin["J_init"]
    print("TWIN_RECIPE_OK")
