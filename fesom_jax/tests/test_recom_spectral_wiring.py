"""Lane W of the spectral port (docs/plans/20261028-recom-spectral-port.md): the run-level pieces of the spectral
fork against its twin's dumps and inputs on Albedo.

* the cold start: :func:`fesom_jax.recom.ic.bgc_initial_state` with the fork's layout (37 tracers, the fork's
  ``recom_init`` constants, the 5-degree pi IC files; ``DIC_PI`` makes FESOM2 read the GLODAP PI DIC whatever
  namelist.tra names) must equal the tracers the Fortran
  twin had at its first step (``recom_in.tr*`` of step 1 of ``pi_30d_spectral_adv01``: FESOM's IC after
  ``recom_init``, before any biology);
* the OASIM forcing: :func:`fesom_jax.recom.oasim.read_oasim_clim` for January must equal the twin's
  ``recom_out.oasim_ed2D/es2D`` of step 1 (the fields ``sbc_do_recom`` interpolated to the nodes);
* the run configuration ``configs/albedo/pi_recom_spectral.yaml`` validates (its namelist, layout, IC set, optics
  and OASIM paths), and the fork's defaults leave the 692f804 config untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from fesom_jax.io_dump import load_recom_dump
from fesom_jax.mesh import load_mesh
from fesom_jax.recom import ic as IC
from fesom_jax.recom import layout as LY
from fesom_jax.recom import params as P
from fesom_jax.recom.config import RecomConfig

REF = Path(os.environ.get("FESOM_RECOM_FORK_REF", "/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/recom_ref"))
DATA = Path(os.environ.get("FESOM_JAX_DATA", "/albedo/work/projects/p_specbic/sloza/fesom_jax_data"))
REPO = Path(__file__).resolve().parents[2]
NML = REPO / "scripts" / "recom" / "fortran" / "namelists_albedo_spectral" / "namelist.recom"
YAML = REPO / "configs" / "albedo" / "pi_recom_spectral.yaml"
RUN = os.environ.get("FESOM_RECOM_SPECTRAL_RUN", "pi_30d_spectral_adv01")
DUMPS = REF / RUN / "dumps"
need_ref = pytest.mark.skipif(not (DUMPS / "step_000001").is_dir() or not (DATA / "mesh_pi_fork").is_dir(),
                              reason=f"the spectral twin's dumps or the pi bundle are absent ({DUMPS}, {DATA})")


@need_ref
def test_cold_ic_matches_the_twin():
    sw, par = P.load_namelist(NML)
    lay = LY.layout_of(sw)
    assert lay.ntr == 37 and lay.cdom
    mesh = load_mesh(DATA / "mesh_pi_fork")
    assert sw.DIC_PI                                                       # the twin read the PI DIC file
    ic = IC.bgc_initial_state(mesh, IC.PI_PRESENT_IC_FILES, sw, par, ic_dir=DATA / "recom_pi")   # DIC_PI overrides 1002
    f, _ = load_recom_dump(DUMPS, 1, names=[f"recom_in.tr{i}" for i in lay.ids] + ["mesh.nlevels_nod2D"])
    nlev = f["mesh.nlevels_nod2D"].astype(int) - 1
    wet = np.arange(mesh.nl - 1)[None, :] < nlev[:, None]
    bad = {}
    for s, tid in enumerate(lay.ids):
        a, b = np.asarray(ic.bgc[..., s])[wet], np.asarray(f[f"recom_in.tr{tid}"])[wet]
        den = np.max(np.abs(b))
        err = np.max(np.abs(a - b)) / den if den > 0 else np.max(np.abs(a))
        if err > 1e-12:
            bad[f"{tid} {lay.names[s]}"] = f"{err:.2e}"
    assert not bad, bad


@need_ref
@pytest.mark.skipif(not Path("/albedo/work/projects/p_specbic/sloza/forcing_oasim_v2").is_dir(),
                    reason="the OASIM climatology is absent")
def test_oasim_january_matches_the_twin():
    from fesom_jax.recom.oasim import read_oasim_clim
    mesh = load_mesh(DATA / "mesh_pi_fork")
    ed, es = read_oasim_clim(mesh, "/albedo/work/projects/p_specbic/sloza/forcing_oasim_v2",
                             "oasim_clim1998-2012_darwin400.nc")
    f, meta = load_recom_dump(DUMPS, 1, names=["recom_out.oasim_ed2D", "recom_out.oasim_es2D"])
    assert int(meta["month"]) == 1
    for got, name in ((ed[0], "oasim_ed2D"), (es[0], "oasim_es2D")):
        want = np.asarray(f[f"recom_out.{name}"]).T                    # (nod2D, 13)
        err = np.max(np.abs(got - want)) / np.max(np.abs(want))
        assert err < 1e-12, (name, err)


def test_spectral_config_validates():
    if not YAML.is_file():
        pytest.skip("no spectral run config")
    from fesom_jax.run_config import load_yaml
    cfg = load_yaml(str(YAML))
    rc = cfg.recom
    assert rc.spectral and rc.ntr == 37 and rc.layout.cdom and rc.ic_set == "pi"
    oasim = cfg.bgc_oasim()
    assert oasim is not None and oasim["pattern"] == "oasim_clim1998-2012_darwin400.nc"
    assert cfg.bgc_dust_path().endswith("DustClimMonthlyAlbani_pimesh.nc")


def test_fork_defaults_off():
    """The 692f804 configuration: no spectral fields set, the same IC constants."""
    rc = RecomConfig()
    assert rc.spectral is False and rc.optics_dir is None
    from fesom_jax.recom import tracers as T
    p = P.RecomParams()
    assert T.init_value(1013, p) == T.TINY_CHL / p.Chl2N_max                       # the 692f804 quirk kept
    assert T.init_value(1013, p, LY.SPECTRAL_4P3Z2D) == T.TINY_CHL / p.Chl2N_max_d  # the fork's own rule
    assert T.init_value(1023, p, LY.SPECTRAL_4P3Z2D) == 0.0
    assert T.init_value(1037, p, LY.SPECTRAL_4P3Z2D) == T.TINY
