"""Task 3.2 — end-to-end gradient gate (**GATE 3**, the project's de-risking gate).

This is THE test that retires the biggest risk: that the hard AD patterns hold on the
*real* model over a multi-step window. It differentiates a scalar loss (mean SST) of
the ``integrate`` output (checkpointed ``lax.scan``) w.r.t.:

* **a scalar physics parameter** — the PP background diffusivities ``k_ver``/``a_ver``,
  threaded as :class:`~fesom_jax.params.Params` leaves (the ML-hook seam). The gradient
  routes through the CG ``custom_linear_solve`` (``k_ver`` across steps, ``a_ver`` even
  within a step) — so a passing FD↔AD check proves the implicit-diff transpose solve is
  correct on the assembled model.
* **an initial-condition field** — ``d(loss)/d(T₀)`` (vector-valued); the stronger
  masked-NaN probe (it exposed and now guards the ``eos`` ``bvfreq`` bottom-padding
  ``1/zdiff`` backward-NaN trap — see ``docs/PORTING_LESSONS.md``).

**Smooth regime (required for the FD check):** the gradient target ``k_ver`` enters
``Kv = mix·factor³ + k_ver`` *additively* (``dKv/dk_ver = 1``), clean past the PP
``max(N²,0)`` kink. We further verify the probe (blob) column stays **stratified**
(``bvfreq>0``, away from the convective ``max`` kink), ``S`` stays **35** (≫ the 0.5
salinity-floor kink), and the FD↔AD agreement itself certifies we're off the upwind
``|vflux|`` kink.

**FD methodology:** central, relative, float64, with a step-size **sweep**. The
end-to-end FD floor is set by the loss's intermediate-sum magnitude (mean SST ~10 ⇒
~``eps·10`` round-off), not the AD accuracy — so we evaluate the headline check at a
``k_ver`` where the signal clears that floor and assert the **plateau** (best over the
sweep), not any single ``h``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fesom_jax import forcing, ic, ssh
from fesom_jax.config import A_VER, K_VER
from fesom_jax.integrate import integrate_jit
from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
from fesom_jax.params import Params

DT = 100.0          # the validated pi config (100-step stable)
N = 20              # modest window (the model is mildly chaotic; long windows amplify)
PROBE = 1001        # a node inside the T-blob → stratified (bvfreq>0, off the N² kink)
H_SWEEP = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)   # relative central-FD step sizes


@pytest.fixture(scope="module")
def mesh():
    if not DEFAULT_PI_MESH_DIR.is_dir():
        pytest.skip(f"pi mesh export missing: {DEFAULT_PI_MESH_DIR} (Task 0.3)")
    return load_mesh()


@pytest.fixture(scope="module")
def op(mesh):
    return ssh.build_ssh_operator(mesh, dt=DT)


@pytest.fixture(scope="module")
def stress(mesh):
    return forcing.surface_stress(mesh)


@pytest.fixture(scope="module")
def st0(mesh):
    return ic.initial_state(mesh)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _mean_sst(state, mesh):
    """Scalar loss: mean sea-surface temperature over wet surface nodes."""
    wet0 = jnp.asarray(mesh.node_layer_mask[:, 0])
    return jnp.sum(jnp.where(wet0, state.T[:, 0], 0.0)) / jnp.sum(wet0)


def _kver_loss(mesh, op, stress, st0):
    """``loss(k_ver)`` = mean SST after N steps (a_ver at its default)."""
    def loss(kver):
        p = Params(k_ver=kver, a_ver=jnp.asarray(A_VER, jnp.float64))
        fin = integrate_jit(st0, mesh, op, stress, n_steps=N, params=p, dt=DT)
        return _mean_sst(fin, mesh)
    return loss


def _fd_sweep(loss, k0):
    """Central relative FD over :data:`H_SWEEP`; returns ``(g_ad, [(h, g_fd, rel)…])``."""
    g_ad = float(jax.grad(loss)(k0))
    rows = []
    for h in H_SWEEP:
        kp, km = k0 * (1.0 + h), k0 * (1.0 - h)
        g_fd = float((loss(kp) - loss(km)) / (kp - km))
        rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-300)
        rows.append((h, g_fd, rel))
    return g_ad, rows


# --------------------------------------------------------------------------
# 1. d(loss)/d(k_ver): AD vs FD step-size sweep — the headline gate
# --------------------------------------------------------------------------
def test_grad_kver_vs_fd_sweep(mesh, op, stress, st0, capsys):
    """``d(mean SST)/d(k_ver)`` AD vs central FD, swept over h. Assert the FD-converged
    **plateau** < 1e-4 (not any single h: round-off floor below, truncation above).
    Evaluated at a representative background ``k_ver=1e-4`` where the FD signal clears
    the loss's ~eps·10 round-off floor (the AD value is k_ver-robust — see the physical
    -point test)."""
    loss = _kver_loss(mesh, op, stress, st0)
    k0 = jnp.asarray(1e-4, jnp.float64)
    g_ad, rows = _fd_sweep(loss, k0)

    with capsys.disabled():
        print(f"\n  d(mean SST)/d(k_ver) AD = {g_ad:+.6e}  (k_ver={float(k0):.0e})")
        for h, g_fd, rel in rows:
            print(f"    h={h:.0e}  FD={g_fd:+.6e}  rel|AD-FD|={rel:.2e}")

    assert np.isfinite(g_ad) and g_ad != 0.0
    plateau = min(rel for _, _, rel in rows)
    assert plateau < 1e-4, f"FD plateau rel err {plateau:.2e} ≥ 1e-4"


def test_grad_kver_physical_point(mesh, op, stress, st0):
    """At the default background ``k_ver=K_VER=1e-5`` the gradient is finite, has the
    physical sign (more vertical mixing ⇒ heat leaves the warm surface ⇒ SST falls ⇒
    ``d(SST)/d(k_ver) < 0``), and agrees with FD to the round-off-floor-limited ~1e-4
    (the tight plateau needs a larger k_ver to lift the FD signal — see above)."""
    loss = _kver_loss(mesh, op, stress, st0)
    k0 = jnp.asarray(K_VER, jnp.float64)
    g_ad, rows = _fd_sweep(loss, k0)
    assert np.isfinite(g_ad)
    assert g_ad < 0.0, f"expected d(SST)/d(k_ver) < 0, got {g_ad:+.3e}"
    plateau = min(rel for _, _, rel in rows)
    assert plateau < 1e-3, f"FD plateau rel err {plateau:.2e} at k_ver=1e-5"


# --------------------------------------------------------------------------
# 2. gradient flows through the CG custom_linear_solve
# --------------------------------------------------------------------------
def test_grad_flows_through_cg(mesh, op, stress, st0, capsys):
    """``d(loss)/d(a_ver)`` exercises the CG ``custom_linear_solve`` *within* a step
    (a_ver → Av → impl_vert_visc → du → compute_ssh_rhs → solve_ssh → d_eta → uv → …),
    so a correct FD↔AD match certifies the implicit-diff transpose solve on the
    assembled model. Finite, nonzero, FD-consistent."""
    def loss(aver):
        p = Params(k_ver=jnp.asarray(K_VER, jnp.float64), a_ver=aver)
        fin = integrate_jit(st0, mesh, op, stress, n_steps=N, params=p, dt=DT)
        return _mean_sst(fin, mesh)

    a0 = jnp.asarray(1e-4, jnp.float64)
    g_ad, rows = _fd_sweep(loss, a0)
    with capsys.disabled():
        print(f"\n  d(mean SST)/d(a_ver) AD = {g_ad:+.6e}  (flows through CG)")
    assert np.isfinite(g_ad) and g_ad != 0.0
    plateau = min(rel for _, _, rel in rows)
    assert plateau < 1e-4, f"a_ver FD plateau rel err {plateau:.2e}"


# --------------------------------------------------------------------------
# 3. gradient w.r.t. an initial-condition field (vector-valued)
# --------------------------------------------------------------------------
def test_grad_ic_field_finite(mesh, op, stress, st0):
    """``d(loss)/d(T₀)`` (the IC temperature field) — finite **everywhere** (the
    masked/below-bottom lanes too: the ``eos`` zdiff fix), nonzero on wet layers, and
    exactly 0 on the masked lanes (the loss truly cannot depend on below-bottom T).
    This vector-valued gradient is the masked-NaN probe a scalar-param gradient misses."""
    mlay = np.asarray(mesh.node_layer_mask)

    def loss(T0):
        s = dataclasses.replace(st0, T=T0, T_old=T0)
        fin = integrate_jit(s, mesh, op, stress, n_steps=N, dt=DT)
        return _mean_sst(fin, mesh)

    g = np.asarray(jax.grad(loss)(st0.T))
    assert np.all(np.isfinite(g)), f"{int(np.isnan(g).sum())} non-finite grad entries"
    assert np.max(np.abs(g[mlay])) > 0.0, "IC gradient identically zero on wet layers"
    assert np.max(np.abs(g[~mlay])) == 0.0, "below-bottom lanes carry spurious gradient"


# --------------------------------------------------------------------------
# 4. smooth-regime certification (the FD check is only meaningful here)
# --------------------------------------------------------------------------
def test_smooth_regime(mesh, op, stress, st0):
    """Certify the window stays in a differentiable-smooth regime: the probe (blob)
    column is **stratified** throughout (``bvfreq>0`` — off the PP ``max(N²,0)`` /
    convective kinks), ``S`` stays **35** (≫ the 0.5 salinity-floor kink), and no
    column goes genuinely convective (``bvfreq`` only dips to FP-noise ~ -1e-13 in the
    dead constant-T region, which carries no gradient)."""
    mlay = np.asarray(mesh.node_layer_mask)
    mif = np.asarray(mesh.node_iface_mask)
    fin = integrate_jit(st0, mesh, op, stress, n_steps=N, dt=DT)

    # S never touches the floor
    assert np.min(np.asarray(fin.S)[mlay]) > 1.0

    # probe blob column is strictly stratified on its wet interfaces (off the N² kink)
    bv = np.asarray(fin.bvfreq)[PROBE - 1]
    col = mif[PROBE - 1]
    assert np.min(bv[col]) > 0.0, "probe column not strictly stratified"

    # no genuine convection anywhere (only machine-zero noise in the dead region)
    assert np.min(np.asarray(fin.bvfreq)[mif]) > -1e-12
