"""REcoM plan Task 14, gate G6 (lane W): ``jax.grad`` of a 1-day NPP cost with respect to REcoM parameters through
the full model step (stage 1 + stage 3 + the physics) against central finite differences.

The cost is the area-integrated column NPP (``State.bgc_diag``'s ``npp``) summed over the 32 steps of one day on pi,
from the end of the G5 run (day 30 of the cold start, ``FESOM_G6_RESTART``, default the lane-W G5 restart: a grown
bloom with grazers; the cold start has none, so the grazing derivative is zero there), else from the twin's step-0
state (:mod:`fesom_jax.recom.reference`). The parameters enter as
traced leaves of ``Params.recom``; nothing in the physics reads a BGC tracer, so the gradient runs through the BGC
part of the steps only. Parameters: the maximum photosynthesis rate of the nanophytoplankton ``P_cm``, the
mesozooplankton maximum grazing rate ``graz_max``, the DON remineralisation rate ``rho_N`` and the diatoms' light
slope ``alfa_d``. After the 30-day cold start the mesozooplankton are still at their tiny initial value, so the
grazing derivative is numerically zero (checked as such: both the adjoint and FD below 1e-12 J per unit log p).

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
from fesom_jax.tests import test_recom_step as TS  # noqa: E402

NAMES = ("P_cm", "graz_max", "rho_N", "alfa_d")
# a derivative below this fraction of J per unit log-parameter is numerically zero: central differences of J cannot
# resolve it (graz_max after the 30-day cold start: the mesozooplankton are still at their tiny initial value)
NEGLIGIBLE = 1e-12
NSTEPS = int(os.environ.get("FESOM_G6_STEPS", "32"))
FD_REL = 1e-4          # gate tolerance (plan Testing Strategy, G6)
H_REL = 1e-3           # central-difference step, relative to the parameter


RESTART = os.environ.get("FESOM_G6_RESTART", "/work/ab0995/a270088/recom_jax_runs/g5_pi_30d_v1/restart")


def _start(fx):
    """``(state, forcing, start day)``: the G5 day-30 restart (a new run segment: mstep 1, a fresh BgcForcing) when
    it exists, else the twin's step 0."""
    from pathlib import Path
    mesh, cfg = fx["mesh"], fx["cfg"]
    if not Path(RESTART, "zarr.json").exists() and not Path(RESTART, ".zgroup").exists():
        return jax.tree.map(jnp.asarray, fx["st"]), fx["cf"], 1
    from fesom_jax import partit
    from fesom_jax import zarr_output as zo
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    st, meta = zo.read_restart(RESTART, mesh, ser)
    st = wiring.start_segment(jax.tree.map(lambda a: jnp.asarray(np.asarray(a)), st), cfg.recom)
    assert st.T.shape == fx["st"].T.shape and st.uv.shape == fx["st"].uv.shape   # serial: no lane padding
    day = 1 + int(meta["step"]) * int(cfg.dt) // 86400
    cf = surface_forcing.build_surface_forcing(mesh, 1958, sst_ic=np.asarray(fx["st"].T[:, 0]), dt=cfg.dt,
                                               mslp=True, bgc_dust_path=cfg.bgc_dust_path(), **cfg.forcing_paths())
    return st, cf, day


def _cost_fn(fx, n):
    mesh = fx["mesh"]
    dt = fx["cfg"].dt
    st0, cf, day = _start(fx)
    print(f"G6 start: day {day} of 1958 ({'restart ' + RESTART if day > 1 else 'twin step 0'})")
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


@TS.need_twin
def test_g6_grad_vs_fd():
    fx = TS._setup()
    cost, theta = _cost_fn(fx, NSTEPS)
    jc = jax.jit(cost)
    j0 = float(jc(theta))
    g = np.asarray(jax.jit(jax.grad(cost))(theta))
    print(f"G6 {NSTEPS} steps: J = {j0:.6e} [mmol C d-1 summed over steps], grad = {g}")
    assert np.all(np.isfinite(g))
    for i, name in enumerate(NAMES):
        h = H_REL * float(theta[i])
        e = jnp.zeros(len(NAMES)).at[i].set(h)
        fd = (float(jc(theta + e)) - float(jc(theta - e))) / (2 * h)
        rel = abs(g[i] - fd) / abs(fd) if fd != 0 else float("inf")
        print(f"G6 {name:>9}: adjoint {g[i]: .10e}  FD {fd: .10e}  rel {rel:.2e}  (dJ/dlog p = {g[i] * theta[i]:.3e}, "
              f"J {j0:.3e})")
        if abs(g[i] * float(theta[i])) <= NEGLIGIBLE * abs(j0) and abs(fd * float(theta[i])) <= NEGLIGIBLE * abs(j0):
            continue                                       # both numerically zero
        assert rel <= FD_REL, (name, g[i], fd)
