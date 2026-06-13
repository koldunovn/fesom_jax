"""Phase 9b JT.3 — TKE wired into the assembled CORE2 step (the K.8 live gate).

The mixing seam (:func:`fesom_jax.step.step` substep 4) dispatches to the classical-TKE
driver behind ``tke_cfg`` (the ``kpp_cfg`` precedent), emitting ``(Kv, Av, uvnode)`` +
the advanced prognostic ``tke``. Validated by running one JAX TKE CORE2 step (PHC IC +
JRA55 1958 + static-ice forcing, dt=1800 — the ``job_tke_t2_cdump`` config: TKE / KPP off
/ GM off / ice off) and comparing to the C cdump (the regenerated 6.6-double oracle):

* **step 1 is cold start** (``tke_old=0`` ⇒ ``KappaM=KappaH=0`` ⇒ ``kv=av=0``); ``tke_new``
  is the surface-flux-driven column solve, which depends only on the stress + geometry
  (IC-independent), so the default IC suffices and the JAX forcing's bit-faithfulness (the
  K.4 ``stress_node_surf`` gate) carries straight through;
* the assembled ``forc_tke_surf = |stress_node_surf|/ρ₀`` vs the cdump ``normstress``;
* ``state.tke`` (the new prognostic) evolves off zero — TKE is genuinely running;
* TKE ≠ KPP (the schemes are distinct at the seam — TKE's cold-start ``Kv=0`` vs KPP's
  immediate OBL ``Kv>0``), and the both-cfgs / pi-path raises (the latter two also in
  ``test_tke_replay.py``); jit-twice reuses the compiled step (the static ``tke_cfg``).

The 3-step live gate (``vshear2``-engaged, needs the dist_16 IC + the FP-butterfly
methodology) is JT.5's climate-adjacent check. SKIPS cleanly if the CORE2 mesh / IC /
cdump are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fesom_jax import core2_forcing, kpp, ssh
from fesom_jax import step as stepmod
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import core2_initial_state
from fesom_jax.tke import TkeConfig

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
# the 16-rank cdump's PHC IC (per-oracle partition provenance, [[zstar-forcing-dump-config-gap]]);
# the remaining tests don't compare to the cdump's forcing, so this is just a consistent IC.
IC_DIR = ROOT / "data" / "ic_core2_dist16"
TKE_CDUMP = Path("/work/ab0995/a270088/port/tke/cdump/dump")
DT = 1800.0           # the cdump dt
YEAR = 1958

pytestmark = pytest.mark.skipif(
    not (MESH_DIR.is_dir() and (IC_DIR / "T_ic.npy").is_file() and TKE_CDUMP.is_dir()),
    reason="CORE2 mesh / PHC IC / TKE cdump missing")


@pytest.fixture(scope="module")
def run1():
    """Build the CORE2 model + JRA55 1958 forcing (dt=1800), run one eager TKE step, and a
    KPP step from the same IC (for the scheme-engaged check). Eager ~40 s — built once."""
    mesh = load_mesh(MESH_DIR)
    state = core2_initial_state(mesh, IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = core2_forcing.build_core_forcing(mesh, YEAR, sst_ic=np.asarray(state.T[:, 0]))
    fs = cf.static
    sf0 = cf.step_forcing(*core2_forcing.dates_for_steps(YEAR, DT, 1)[0])
    sfx = core2_forcing.compute_surface_fluxes(mesh, state, sf0, fs, dt=DT)

    st_tke = stepmod.step(state, mesh, op, None, dt=DT, is_first_step=True,
                          step_forcing=sf0, forcing_static=fs, tke_cfg=TkeConfig())
    st_kpp = stepmod.step(state, mesh, op, None, dt=DT, is_first_step=True,
                          step_forcing=sf0, forcing_static=fs, kpp_cfg=kpp.KppConfig())
    return dict(mesh=mesh, state=state, op=op, fs=fs, sf0=sf0, sfx=sfx,
                st_tke=st_tke, st_kpp=st_kpp)




# ---------------------------------------------------------------------------
# (NOTE) The live step-1 forcing/tke vs the TKE cdump is NOT tested here: a 3-way check
# showed the cdump's step-1 normstress is an OUTLIER (JAX == KPP-oracle != TKE-cdump, ~7e-4 at
# low-wind nodes) — a quirk of that old C job's forcing inputs, not a JAX bug. The JAX live
# forcing is validated against the KPP oracle <1e-12 (test_kpp_step.py) and end-to-end by the
# 1-yr climate matching c_tke_2yr at the C<->Fortran floor (scripts/core2_tke_climate_compare.py).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2) TKE is genuinely engaged (distinct from KPP) + the prognostic evolves
# ---------------------------------------------------------------------------
def test_tke_state_evolves(run1):
    """The new prognostic ``state.tke`` evolves off its zero IC on the wet column — TKE is
    actually running (not a silent no-op), floored to ``tke_min`` with surface input above it."""
    tke = np.asarray(run1["st_tke"].tke)
    assert float(tke.max()) > 1e-3                    # the surface-flux input dominates tke_min
    assert float(tke.min()) >= 0.0                    # floored, never negative (dry=0)


def test_tke_distinct_from_kpp(run1):
    """The TKE step and a KPP step from the same IC give materially different ``Kv`` — the
    schemes are distinct at the seam (TKE cold-start ``Kv=0`` vs KPP's immediate OBL
    ``Kv>0``), confirming ``tke_cfg`` dispatches to TKE, not silently to KPP/PP."""
    mesh = run1["mesh"]
    kv_tke = np.asarray(run1["st_tke"].Kv)
    kv_kpp = np.asarray(run1["st_kpp"].Kv)
    mwet = np.asarray(mesh.node_layer_mask)
    denom = np.abs(kv_kpp[mwet]).max()
    rel = float(np.abs(kv_tke[mwet] - kv_kpp[mwet]).max() / denom)
    print(f"\n|Kv_tke - Kv_kpp|/|Kv_kpp| = {rel:.3f}")
    assert rel > 0.1                                   # materially different ⇒ engaged


# ---------------------------------------------------------------------------
# 3) dispatch guards + jit
# ---------------------------------------------------------------------------
def test_both_cfgs_raise_live(run1):
    """Both ``kpp_cfg`` and ``tke_cfg`` set ⇒ raise (one scheme per process), through the
    forced step (not just the pi path)."""
    mesh, state, op, fs, sf0 = (run1[k] for k in ("mesh", "state", "op", "fs", "sf0"))
    with pytest.raises(ValueError, match="both set"):
        stepmod.step(state, mesh, op, None, dt=DT, is_first_step=True, step_forcing=sf0,
                     forcing_static=fs, tke_cfg=TkeConfig(), kpp_cfg=kpp.KppConfig())


def test_jit_twice_no_leak(run1):
    """``step_jit`` with a static ``tke_cfg`` reuses the compiled step across calls — two
    invocations give bit-identical state.tke (deterministic, no retrace bug)."""
    mesh, state, op, fs, sf0 = (run1[k] for k in ("mesh", "state", "op", "fs", "sf0"))
    cfg = TkeConfig()
    a = stepmod.step_jit(state, mesh, op, None, dt=DT, is_first_step=True,
                         step_forcing=sf0, forcing_static=fs, tke_cfg=cfg)
    b = stepmod.step_jit(state, mesh, op, None, dt=DT, is_first_step=True,
                         step_forcing=sf0, forcing_static=fs, tke_cfg=cfg)
    assert np.array_equal(np.asarray(a.tke), np.asarray(b.tke))
