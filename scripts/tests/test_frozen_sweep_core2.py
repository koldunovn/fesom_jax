"""The library frozen-sweep backward (``IceConfig.adjoint_damping="frozen_sweep"``,
fesom_jax/ice_mevp_frozen.py) against the mode harness that measured it (TLfT in
scripts/adjoint/ice_modes.py, docs/PORTING_LESSONS.md I5g), on the captured mEVP call of one real
CORE2 step (2013-01-01).

The two are written independently. The harness linearizes its own sweep with the stress of
``stress_terms`` (mode "Tθ") through a custom_jvp, and gets the vjp by ``jax.linear_transpose``.
The library uses a custom_vjp with ``theta_sweep`` and an explicit transposed loop. Both see the
wind stress as a function of the entry velocity: the harness inside its sweep, the library
through the outer autodiff of ``atm_ice_stress``. They agree up to the rounding of two compiled
forwards, which the frozen tangent then carries; the tolerance below is the measured agreement
with a margin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
STATES = Path("/work/ab0995/a270088/fesom_jax_dev/states_2013")
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import ice_modes as IM  # noqa: E402


@pytest.fixture(scope="module")
def icemap():
    if not ((ROOT / "data" / "mesh_core2").is_dir() and (STATES / "fesom.2013.001.00000").is_dir()):
        pytest.skip("CORE2 mesh / 2013 States missing")
    return IM.IceMap("2013-01-01")


@pytest.mark.parametrize("theta,kappa0", [(0.75, 1.0), (0.95, 1.0), (0.97, 1.0)])
def test_core2_library_backward_equals_the_harness_tlft(icemap, theta, kappa0):
    import jax
    import jax.numpy as jnp
    from fesom_jax import forcing as _forcing
    from fesom_jax import ice_mevp
    M = icemap
    cfg = M.cfg._replace(adjoint_damping="frozen_sweep", adjoint_alpha_kappa=kappa0,
                         adjoint_theta=theta)
    # the captured fields are jit ARGUMENTS (closed over, XLA constant-folds the setup with
    # other rounding: the harness's own lesson)
    fl = {k: M.kw[k] for k in ("a_ice", "m_ice", "m_snow", "srfoce_u", "srfoce_v", "elevation")}
    fl.update(u_air=M.inp["u_air"], v_air=M.inp["v_air"], bn=M.inp["bn"])

    def lib(x, f):
        sax, say = _forcing.atm_ice_stress(f["u_air"], f["v_air"], x[0], x[1])
        return ice_mevp.mevp_dynamics(
            cfg, M.mesh, a_ice=f["a_ice"], m_ice=f["m_ice"], m_snow=f["m_snow"],
            u_ice=x[0], v_ice=x[1], sigma11=x[2], sigma12=x[3], sigma22=x[4],
            srfoce_u=f["srfoce_u"], srfoce_v=f["srfoce_v"], elevation=f["elevation"],
            stress_ax=sax, stress_ay=say, boundary_node=f["bn"], exch=None)

    rng = np.random.default_rng(12)
    nflat = int(M.offs[-1])
    w = rng.standard_normal(nflat) * np.asarray(M.dscale) ** -1      # cotangent of raw outputs
    def vjp_lib(x, f, wt):
        y, back = jax.vjp(lambda x_: lib(x_, f), x)
        return y, back(wt)[0]
    y_lib, ct = jax.jit(vjp_lib)(M.x0, fl, M.unflat(jnp.asarray(w)))
    ct_lib = np.asarray(M.flat(ct))
    # the forward of the frozen form is the plain one (the frozen form's primal under vjp)
    y_plain = jax.jit(lambda x, f: ice_mevp.mevp_dynamics(
        M.cfg, M.mesh, a_ice=f["a_ice"], m_ice=f["m_ice"], m_snow=f["m_snow"], u_ice=x[0],
        v_ice=x[1], sigma11=x[2], sigma12=x[3], sigma22=x[4], srfoce_u=f["srfoce_u"],
        srfoce_v=f["srfoce_v"], elevation=f["elevation"],
        stress_ax=_forcing.atm_ice_stress(f["u_air"], f["v_air"], x[0], x[1])[0],
        stress_ay=_forcing.atm_ice_stress(f["u_air"], f["v_air"], x[0], x[1])[1],
        boundary_node=f["bn"], exch=None))(M.x0, fl)
    for a, b in zip(y_lib, y_plain):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()

    _, p_vjp, _ = M.programs("tlf", f"T{theta:g}")
    ones = M.ones()
    gk = jnp.asarray([0.0, 0.0, 0.0, kappa0])
    ct_h = np.asarray(p_vjp(M.flat(M.x0), M.c0, *ones[:3], gk,
                            jnp.asarray(w) * M.dscale)) / np.asarray(M.dscale)
    rel = np.linalg.norm(ct_lib - ct_h) / np.linalg.norm(ct_h)
    print(f"library vs harness TLfT: relative difference {rel:.3e}")
    assert np.all(np.isfinite(ct_lib))
    assert rel < 1e-10                                   # measured 4e-15 (θ 0.75)
