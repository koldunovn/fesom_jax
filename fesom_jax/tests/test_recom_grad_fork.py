"""Lane W of the spectral port (docs/plans/completed/20261028-recom-spectral-port.md), gate G6 for the fork: ``jax.grad`` of a
1-day NPP cost through the full model step with the spectral light (the 13-band direct 3-stream solver, the 4-PFT
growth) against central finite differences, as ``test_recom_grad.py`` for REcoM 692f804.

The cost is the area-integrated column NPP of the four groups (``State.bgc_diag``'s ``npp``) summed over the steps
of one day on pi with the production spectral configuration (``configs/albedo/pi_recom_spectral.yaml``), from the
end of the 30-day spectral run (``FESOM_G6_FORK_RESTART``, default the lane-W run of job 48603201: a grown bloom),
else from the twin's step-0 state. Parameters, traced leaves of ``Params.recom``: the temperature exponents of the
nanophytoplankton and coccolithophore photosynthesis maxima ``expon_phy`` / ``expon_cocco`` (the fork's 4-PFT build
takes ``pMax`` from its ``Temp_*`` functions, so ``P_cm``, ``P_cm_d``, ``P_cm_c`` are dead there, as are the light
slopes ``alfa*`` under the spectral light, which uses the band absorption ``alpha_I``: their finite differences are
exactly 0), the mesozooplankton grazing rate ``graz_max``, the DON remineralisation rate ``rho_N``, the chlorophyll
degradation rate ``deg_Chl`` and the specific CDOM absorption ``cdomcoeff`` (the spectral path: CDOM -> the IOPs ->
the light -> every group's growth; the only one that moves a one-step NPP, the others act through the state).

Set ``FESOM_G6_STEPS`` to shorten the window (default 32).
"""

from __future__ import annotations

import dataclasses
import os
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax import surface_forcing  # noqa: E402
from fesom_jax.params import Params  # noqa: E402
from fesom_jax.recom import wiring  # noqa: E402
from fesom_jax.tests import test_recom_step_fork as TF  # noqa: E402

NAMES = ("expon_phy", "expon_cocco", "graz_max", "rho_N", "deg_Chl", "cdomcoeff")
NEGLIGIBLE = 1e-12     # a derivative below this fraction of J per unit log-parameter is numerically zero
NSTEPS = int(os.environ.get("FESOM_G6_STEPS", "32"))
FD_REL = 1e-4          # gate tolerance (the 692f804 G6)
H_REL = 1e-3           # central-difference step, relative to the parameter
RUN = "pi_1d_spectral_jaxphys"
def _cost_fn(fx, n):
    mesh = fx["mesh"]
    dt = fx["cfg"].dt
    st0, cf, day = TF.warm_start(fx)
    print(f"G6-fork start: day {day} of 1958 ({'restart ' + TF.RESTART if day > 1 else 'twin step 0'})")
    stack = cf.stack(surface_forcing.dates_for_steps(1958, dt, n, start_day=day))
    zero = jnp.zeros((mesh.elem2D, 2))
    _, p0 = fx["cfg"].recom.load()
    a2 = mesh.areasvol[:, 0]
    k_npp = wiring.DIAG2D.index("npp")

    def cost(theta):
        params = dataclasses.replace(Params.defaults(), recom=p0._replace(**dict(zip(NAMES, theta))))

        def one(st, sf, first):
            st = stepmod.step(st, mesh, fx["op"], zero, params, dt=dt, is_first_step=first, step_forcing=sf,
                              forcing_static=fx["fs"], **fx["phys"])
            return st, jnp.sum(st.bgc_diag[:, k_npp] * a2)

        st, j = one(st0, jax.tree.map(lambda x: x[0], stack), day == 1)     # a restart carries the AB2 history
        if n > 1:
            def body(c, sf):
                s, acc = c
                s, jj = one(s, sf, False)
                return (s, acc + jj), None
            (st, j), _ = jax.lax.scan(jax.checkpoint(body), (st, j), jax.tree.map(lambda x: x[1:], stack))
        return j

    return cost, jnp.array([getattr(p0, k) for k in NAMES], jnp.float64)


@pytest.mark.skipif(not TF.HAVE or RUN not in TF.RUNS or not TF.RUN_CFG.is_file(),
                    reason="needs the spectral twin's dumps and the production spectral config")
def test_g6_fork_grad_vs_fd():
    fx = TF._setup(RUN, TF.RUN_CFG)
    cost, theta = _cost_fn(fx, NSTEPS)
    jc = jax.jit(cost)
    j0 = float(jc(theta))
    g = np.asarray(jax.jit(jax.grad(cost))(theta))
    print(f"G6-fork {NSTEPS} steps: J = {j0:.6e} [mmol C d-1 summed over steps], grad = {g}")
    assert np.all(np.isfinite(g))
    for i, name in enumerate(NAMES):
        h = H_REL * float(theta[i])
        e = jnp.zeros(len(NAMES)).at[i].set(h)
        fd = (float(jc(theta + e)) - float(jc(theta - e))) / (2 * h)
        rel = abs(g[i] - fd) / abs(fd) if fd != 0 else float("inf")
        print(f"G6-fork {name:>9}: adjoint {g[i]: .10e}  FD {fd: .10e}  rel {rel:.2e}  "
              f"(dJ/dlog p = {g[i] * theta[i]:.3e}, J {j0:.3e})")
        if abs(g[i] * float(theta[i])) <= NEGLIGIBLE * abs(j0) and abs(fd * float(theta[i])) <= NEGLIGIBLE * abs(j0):
            continue                                       # both numerically zero
        assert rel <= FD_REL, (name, g[i], fd)
