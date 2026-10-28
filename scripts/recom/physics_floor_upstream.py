"""The pi physics floor (``physics_floor.py``) over 17 steps, compared at the twin's dumped steps 1, 2, 3, 17, with or
without the two FESOM2 changes that investigation G found (plan ``docs/plans/20261025-physics-ice-T-gap.md``) and
that fesom-jax now mirrors (plan ``docs/plans/20261025-physics-upstream-944-1116.md``):

  k33  PR #944 (0452259a, merged 6526e4be 2026-08-21): the Redi K33 interface interpolation weights
       (``gm_redi.k33_augmentation``; ``legacy_k33_weights=True`` = the pre-#944 swapped weights).
       Pre-#944, step 1: T 3.4e-3 degC in the top layers of weakly stratified (ice-covered) columns.
  sw   PR #1116 (4aab0efd, merged ba245401 2026-10-14): the shortwave attenuation depth below the surface interface
       (``forcing.cal_shortwave_rad``; ``legacy_sw_depth=True`` = the pre-#1116 absolute depth). Identical at rest, so
       the pre-#1116 gap starts at step 2: T 2e-4 degC in sunlit open water.

Run on a compute node: ``python scripts/recom/physics_floor_upstream.py {current,legacy,legacy_k33,legacy_sw}``
(``current`` = both mirrored, the default).
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics_floor import DUMPS, REPO, _stats, step0_state  # noqa: E402

from fesom_jax import ssh, surface_forcing  # noqa: E402
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax.io_dump import load_recom_dump  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.run_config import load_yaml  # noqa: E402


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "current"
    legacy = dict(current={}, legacy=dict(legacy_k33_weights=True, legacy_sw_depth=True),
                  legacy_k33=dict(legacy_k33_weights=True), legacy_sw=dict(legacy_sw_depth=True))[mode]
    print(f"[upstream] mode {mode} {legacy}")
    cfg = load_yaml(REPO / "configs" / "pi_recom.yaml")
    phys, dt = cfg.physics_kwargs(), float(cfg.dt)
    mesh = load_mesh()
    nl = int(mesh.nl)
    d = {n: load_recom_dump(DUMPS, n)[0] for n in (1, 2, 3, 17)}
    st = jax.tree.map(jnp.asarray, step0_state(mesh, d[1], phys["ale_cfg"]))
    cf = surface_forcing.build_surface_forcing(mesh, cfg.forcing["start_year"], sst_ic=np.asarray(st.T[:, 0]),
                                               dt=dt, mslp=cfg.forcing_mslp(), **cfg.forcing_paths())
    dates = surface_forcing.dates_for_steps(cfg.forcing["start_year"], dt, max(d))
    op = ssh.build_ssh_operator(mesh, dt=dt)
    zero = jnp.zeros((mesh.elem2D, 2))
    nlev = np.asarray(mesh.nlevels_nod2D)
    lay = np.arange(nl - 1)[None, :] < (nlev - 1)[:, None]
    lev = np.arange(nl)[None, :] < nlev[:, None]
    lat = np.degrees(np.asarray(mesh.geo_coord_nod2D[:, 1]))
    for n in range(1, max(d) + 1):
        st = stepmod.step_jit(st, mesh, op, zero, dt=dt, is_first_step=(n == 1),
                              step_forcing=cf.step_forcing(*dates[n - 1]), forcing_static=cf.static, **phys, **legacy)
        if n not in d:
            continue
        f = d[n]
        for r in (_stats("a_ice", np.asarray(st.a_ice), f["recom_in.a_ice"], np.ones(mesh.nod2D, bool)),
                  _stats("Kv", np.asarray(st.Kv), f["transport.Kv"], lev),
                  _stats("T", np.asarray(st.T)[:, :nl - 1], f["step_end.tr1"], lay),
                  _stats("S", np.asarray(st.S)[:, :nl - 1], f["step_end.tr2"], lay)):
            print(f"{n:>3} {r['field']:>6} {r['maxdiff']:10.3e} rel {r['rel']:9.2e} at {r['at']} lat {lat[r['at'][0]]:6.1f}")
        dT = np.abs(np.where(lay, np.asarray(st.T)[:, :nl - 1] - f["step_end.tr1"], 0.0))
        print(f"    cells |dT|>1e-6: {(dT > 1e-6).sum()}, >1e-4: {(dT > 1e-4).sum()}, >1e-3: {(dT > 1e-3).sum()}")
    print("PHYSICS_FLOOR_UPSTREAM_DONE")


if __name__ == "__main__":
    main()
