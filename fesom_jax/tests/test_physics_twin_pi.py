"""The pi physics gate against the physics-matched FESOM2 e96a2b13 twin (``recom_ref/pi_30d_jan_jaxphys``,
``docs/recom/FORTRAN_REFERENCE.md``), steps 1-3 from the Fortran step-0 state.

JAX ``configs/pi_recom.yaml`` (zstar, classical TKE, mEVP, GM/Redi, opt_visc 7, CG SSH, dt 2700, l_mslp) starts from
the twin's cold start, read from its step-1 dumps: T, S from ``recom_in`` (the ice step does not touch them), their
AB2 history from ``tra_in.valuesold``, rest thicknesses, the seeded sea ice. Fortran T and S after step N are
``step_end(N)``. Investigation G (``docs/plans/20261025-physics-ice-T-gap.md``) traced the former 3.4e-3 degC step-1
gap to FESOM2 #944 (Redi K33 weights) and the step-2 one to #1116 (shortwave depth from the surface); with both
mirrored the floor is T 1.2e-6 / 1.2e-6 / 1.8e-6 degC and S 5.5e-7 / 6.0e-7 / 7.1e-7 psu (steps 1-3).
"""

import dataclasses
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from fesom_jax import ice, jra55, ssh, surface_forcing  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax.io_dump import load_recom_dump  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.run_config import load_yaml  # noqa: E402
from fesom_jax.state import State  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DUMPS = Path("/work/ab0995/a270088/recom_ref/pi_30d_jan_jaxphys/dumps")
STEPS = (1, 2, 3)
T_TOL = 2e-6       # degC
S_TOL = 1e-6       # psu

pytestmark = pytest.mark.skipif(
    not all((DUMPS / f"step_{n:06d}" / "index.txt").is_file() for n in STEPS)
    or not Path(jra55.DEFAULT_JRA_DIR).is_dir(),
    reason=f"physics-matched twin dumps ({DUMPS}) or JRA55-do forcing not reachable")


def _pad(a, nl):
    return a if a.shape[1] == nl else np.pad(a, [(0, 0), (0, nl - a.shape[1])] + [(0, 0)] * (a.ndim - 2))


def test_pi_twin_steps123():
    cfg = load_yaml(REPO / "configs" / "pi_recom.yaml")
    phys, dt = cfg.physics_kwargs(), float(cfg.dt)
    mesh = load_mesh()
    nl = int(mesh.nl)
    d = {n: load_recom_dump(DUMPS, n)[0] for n in STEPS}
    assert np.array_equal(d[1]["mesh.gid_nod"], np.arange(1, mesh.nod2D + 1))
    nlev = np.asarray(mesh.nlevels_nod2D)
    assert np.array_equal(d[1]["mesh.nlevels_nod2D"], nlev)

    mask = np.asarray(mesh.node_layer_mask)
    st = jax.tree.map(np.asarray, State.rest(mesh, ale_cfg=phys["ale_cfg"]))
    fld = {k: np.where(mask, _pad(d[1][src], nl), 0.0) for k, src in
           (("T", "recom_in.tr1"), ("S", "recom_in.tr2"), ("T_old", "tra_in.tr1.valuesold"),
            ("S_old", "tra_in.tr2.valuesold"))}
    st = ice.seed_ice(dataclasses.replace(st, **fld), mesh, fld["T"][:, 0], xp=np)
    st = jax.tree.map(jnp.asarray, st)

    cf = surface_forcing.build_surface_forcing(mesh, cfg.forcing["start_year"], sst_ic=np.asarray(st.T[:, 0]),
                                               dt=dt, mslp=cfg.forcing_mslp(), **cfg.forcing_paths())
    dates = surface_forcing.dates_for_steps(cfg.forcing["start_year"], dt, max(STEPS))
    op = ssh.build_ssh_operator(mesh, dt=dt)
    zero = jnp.zeros((mesh.elem2D, 2))
    lay = np.arange(nl - 1)[None, :] < (nlev - 1)[:, None]
    rows = []
    for n in range(1, max(STEPS) + 1):
        st = stepmod.step_jit(st, mesh, op, zero, dt=dt, is_first_step=(n == 1),
                              step_forcing=cf.step_forcing(*dates[n - 1]), forcing_static=cf.static, **phys)
        dT = np.max(np.abs(np.where(lay, np.asarray(st.T)[:, :nl - 1] - d[n]["step_end.tr1"], 0.0)))
        dS = np.max(np.abs(np.where(lay, np.asarray(st.S)[:, :nl - 1] - d[n]["step_end.tr2"], 0.0)))
        rows.append((n, dT, dS))
        print(f"\nstep {n}: max|dT| {dT:.3e} degC, max|dS| {dS:.3e} psu")
    for n, dT, dS in rows:
        assert dT <= T_TOL, (n, dT)
        assert dS <= S_TOL, (n, dS)
