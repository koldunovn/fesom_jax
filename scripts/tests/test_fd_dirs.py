"""The directional FD tool (``scripts/adjoint/fd_dirs.py``, the Fram plan Task 4).

Light: the direction specs (window months, depth ranges, errors), the perturbation patterns and the perturbed
controls / start State on a stand-in model with the CORE2 mesh.

Heavy (CPU compute node, dense, ~30 min): the whole pipeline on a two-step window from the 2013-01-01 restart, the
year-window setting, ice2 + se, cost FRAM — ``fd_dirs fd`` (the base and the ±h forwards of ln C_d, a Tair box
anomaly, a southward wind anomaly over the Fram box, an initial T anomaly in the top 200 m of the Fram box), then
``ice_cbar tkegrad`` (the ice controls, the monthly maps, the initial-state cotangent dump), then ``fd_dirs score``:
the adjoint's directional derivatives against the central FDs. The frozen-sweep ice adjoint (theta 0.97) is not
exact, hence the 10 % bar of the year windows.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import core2_adj as C  # noqa: E402
import fd_dirs as F  # noqa: E402
import ice_cbar as IC  # noqa: E402

from fesom_jax import adjoint_costs as AC  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402

MESH = ROOT / "data" / "mesh_core2"
RESTART = C.RESTART_ARCHIVE / "fesom.2013.001.00000"
Y4RAW7 = ("tke_pr_cut=1e-6+tke_pr_ratio=100+tke_pr_raw_cut=-1e-6+tke_ml_cut=1e-8+freeze_fct_limiter+tke_ml_ratio=100"
          "+tke_ml_raw_cut=-1e-7+tke_pr_lat_cut=70+tke_pr_ice_cut=0.15")
ICE = ["--ice-mode", "exact", "--ice-damping", "frozen_sweep", "--ice-alpha-kappa", "1", "--ice-theta", "0.97",
       "--ice-sweep-guard", "10", "--ice-thin-mass", "100", "--ice-cuts", "fct_flux+strength_conc"]
TOL_FD = 0.10

pytestmark = pytest.mark.skipif(not MESH.is_dir(), reason="CORE2 mesh missing (compute node only)")


def test_parse():
    s = datetime.date(2012, 1, 1)
    d = F.parse("Tair:arctic70:2013-01..2013-03", s)
    assert (d.kind, d.box, d.months) == ("Tair", "arctic70", (12, 13, 14)) and d.slug == "Tair_arctic70_2013_01_2013_03"
    assert F.parse("wsouth:fram:2012-02..2012-02", "2012-01-01").months == (1,)
    assert F.parse("T0:barents:0-200m", s).depth == (0.0, 200.0)
    assert F.parse("cd", s).kind == "cd"
    for bad in ("cd:fram", "Tair:nowhere:2013-01..2013-02", "Tair:fram", "Tair:fram:2011-12..2012-01",
                "Tair:fram:2013-03..2013-01", "T0:fram:200m", "tair:fram:2013-01..2013-01"):
        with pytest.raises(ValueError):
            F.parse(bad, s)


@dataclasses.dataclass(frozen=True)
class _State:                       # the start State's T only (dataclasses.replace works on it)
    T: np.ndarray


@pytest.fixture(scope="module")
def stand_in():
    mesh = load_mesh(MESH)
    n, nl = int(mesh.nod2D), int(mesh.nl)
    maps = {k: np.zeros((3, n)) for k in ("Tair", "u_air", "v_air")}
    return types.SimpleNamespace(mesh=mesh, params0={"forcing_adj": maps, "ln_ice_cd": np.float64(0.0)},
                                 schedule={"n_months": 3}, s0=_State(np.full((n, nl), 2.0, np.float32)))


def test_patterns_and_perturbation(stand_in):
    m = stand_in
    d = F.parse("Tair:arctic70:2012-02..2012-03", "2012-01-01")
    p = F.pattern(m, d)["Tair"]
    lon, lat = C._lonlat_deg(m.mesh)
    assert np.all(p[0] == 0) and np.array_equal(p[1] > 0, lat >= 70.0) and np.array_equal(p[1], p[2])
    w = F.pattern(m, F.parse("wsouth:fram:2012-01..2012-01", "2012-01-01"))
    sx, sy = (np.asarray(v) for v in AC.south_vector(m.mesh))
    box = F._nodes(m.mesh, "fram")
    np.testing.assert_allclose(np.hypot(w["u_air"][0], w["v_air"][0])[box], 1.0, rtol=1e-12)
    assert np.all(w["u_air"][0][~box] == 0) and np.allclose(w["u_air"][0][box], sx[box])
    t0 = F.pattern(m, F.parse("T0:barents:0-200m", "2012-01-01"))
    zmid = -np.asarray(m.mesh.Z)
    assert t0.shape == m.s0.T.shape and t0.sum() > 0 and np.all(t0[:, :zmid.size][:, zmid > 200.0] == 0)
    u, st = F.perturbed(m, d, 0.5)
    assert st is None and float(np.asarray(u["forcing_adj"]["Tair"]).sum()) == pytest.approx(0.5 * p.sum())
    u, st = F.perturbed(m, F.parse("T0:barents:0-200m", "2012-01-01"), 0.25)
    assert st.T.dtype == np.float32 and np.allclose(st.T - m.s0.T, 0.25 * t0)
    u, _ = F.perturbed(m, F.parse("cd", "2012-01-01"), -1e-3)
    assert float(np.asarray(u["ln_ice_cd"])) == -1e-3
    with pytest.raises(ValueError, match="beyond"):
        F.pattern(m, F.parse("Tair:fram:2012-04..2012-04", "2012-01-01"))
    with pytest.raises(ValueError, match="pstar"):
        F.perturbed(m, F.parse("pstar", "2012-01-01"), 1.0)


DIRS = "cd,Tair:arctic70:2013-01..2013-01,wsouth:fram:2013-01..2013-01,T0:fram:0-200m"
HS = "1e-3,0.1,0.1,1e-2"


@pytest.mark.skipif(not RESTART.is_dir(), reason="the 2013 restart missing")
def test_pipeline_two_steps(tmp_path):
    days = str(2 * 1800.0 / 86400.0)
    common = ["--start", "2013-01-01", "--chunk-steps", "2", "--fast", "ice2+se", "--restart", str(RESTART),
              "--forcing-controls", "Tair,u_air,v_air", "--forcing-months", "1", "--cost-from", "2013-01-01"]
    fd_root, grad = tmp_path / "fd", tmp_path / "grad"
    assert F.main(["fd", "--days", days, "--fd-root", str(fd_root), "--dirs", DIRS, "--h", HS,
                   "--save-every", "1"] + common) == 0
    assert IC.main(["tkegrad", "--setting", Y4RAW7, *ICE, "--controls", "pstar,cd", "--cost", "FRAM",
                    "--window-days", days, "--base-dir", str(fd_root / "base"), "--run-dir", str(grad),
                    "--dump-cot", "0", "--dump-fields", "T,S"] + common) == 0
    row = json.loads((grad / "done.json").read_text())
    assert set(row["grad_ice"]) == {"ln_ice_pstar", "ln_ice_cd"} and row["n_cost"] == 2
    assert (grad / "grad_forcing.npz").exists() and (grad / "dump" / "cot_0.npz").exists()
    assert F.main(["score", "--days", days, "--fd-root", str(fd_root), "--dirs", DIRS, "--h", HS, "--no-base",
                   "--grad-dir", str(grad)] + common) == 0
    table = json.loads((fd_root / "fd_table.json").read_text())
    print(f"\n[fd_dirs] J base {table['J_base']:.6e} km3/yr, loss {json.loads((grad / 'forward.json').read_text())}")
    for r in table["rows"]:
        print(f"[fd_dirs] {r['dir']:>32s}: FD {r['fd']:+.6e} AD {r['ad']:+.6e} AD/FD {r['ad_over_fd']:.4f}")
    assert table["J_base"] == pytest.approx(json.loads((grad / "forward.json").read_text())["loss"], rel=1e-12)
    for r in table["rows"]:
        assert r["fd"] != 0.0 and abs(r["ad_over_fd"] - 1.0) <= TOL_FD, r["dir"]
