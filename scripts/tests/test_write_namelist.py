"""A6 gate: scalar → Fortran-namelist transfer (`scripts/write_namelist.py`).

Pure-Python (no JAX) round-trip tests: patch tuned constants into a FESOM namelist, re-parse, and
confirm (a) the patched keys read back EXACTLY the input, (b) the K_GM_max↔Redi_Kmax auto-sync,
(c) exact-key matching (a K_GM_min decoy is untouched), and (d) every other line is byte-preserved.
Token: FORTRAN_TRANSFER_OK. Runs standalone (`pytest scripts/tests/`), separate from the JAX suite.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/ on the path
import write_namelist as wn   # noqa: E402


# a representative &oce_dyn slice (real key spellings/formatting from fesom2/work_all3)
OCE_TEMPLATE = """\
&oce_dyn
scale_area         = 5.8e9   ! reference element area
Fer_GM             = .true.  ! GM on/off
K_GM_max           = 1000.0  ! max. GM thickness diffusivity (m2/s)
K_GM_min           = 2.0     ! GM floor (DECOY — must not change)
Redi               = .true.  ! enable Redi
Redi_Kmax          = 0.0     ! <=0 ⇒ sync to K_GM_max
Redi_Kmin          = 100.0   ! min Redi diffusivity
/
"""

CVMIX_TEMPLATE = """\
&param_tke
tke_c_k          = 0.1          ! TKE parameter c_k
tke_c_eps        = 0.7          ! dissipation c_eps
tke_alpha        = 30.0         ! stability parameter
tke_kappaM_max   = 100.0        ! ceiling (DECOY)
tke_cd           = 3.75         ! surface BC parameter
/
"""


def _parse(text: str) -> dict:
    """Parse `key = value` (real-valued) lines of a namelist, ignoring comments/headers."""
    out = {}
    for line in text.splitlines():
        line = line.split("!", 1)[0].strip()
        if "=" in line and not line.startswith("&") and not line.startswith("/"):
            k, v = line.split("=", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                out[k.strip()] = v.strip()        # non-numeric (.true./.false.)
    return out


# --------------------------------------------------------------------------
# params_to_namelist mapping + auto-sync
# --------------------------------------------------------------------------
def test_params_to_namelist_maps_and_syncs_redi():
    nl = wn.params_to_namelist({"k_gm": 1500.0})
    assert nl == {"K_GM_max": 1500.0, "Redi_Kmax": 1500.0}      # auto-sync
    # explicit redi_kmax is respected (no override)
    nl2 = wn.params_to_namelist({"k_gm": 1500.0, "redi_kmax": 900.0})
    assert nl2 == {"K_GM_max": 1500.0, "Redi_Kmax": 900.0}
    # tke leaves map to themselves
    nl3 = wn.params_to_namelist({"tke_c_k": 0.15, "tke_c_eps": 0.8})
    assert nl3 == {"tke_c_k": 0.15, "tke_c_eps": 0.8}
    # sync off
    nl4 = wn.params_to_namelist({"k_gm": 1500.0}, sync_redi=False)
    assert nl4 == {"K_GM_max": 1500.0}


def test_params_to_namelist_rejects_unknown_leaf():
    with pytest.raises(ValueError, match="unknown tunable leaf"):
        wn.params_to_namelist({"bogus": 1.0})


# --------------------------------------------------------------------------
# round-trip + auto-sync + decoy + preservation
# --------------------------------------------------------------------------
def test_oce_roundtrip_autosync_and_decoy(tmp_path):
    tmpl = tmp_path / "namelist.oce"
    out = tmp_path / "namelist.oce.tuned"
    tmpl.write_text(OCE_TEMPLATE)
    res = wn.write_namelist({"k_gm": 1500.0}, tmpl, out)        # only k_gm → auto-sync Redi

    parsed = _parse(out.read_text())
    assert parsed["K_GM_max"] == 1500.0
    assert parsed["Redi_Kmax"] == 1500.0                       # auto-synced
    assert parsed["K_GM_min"] == 2.0                           # DECOY untouched
    assert parsed["Redi_Kmin"] == 100.0                        # untouched
    assert res["patched"] == {"K_GM_max": 1500.0, "Redi_Kmax": 1500.0}

    # every non-patched line is byte-identical; comments preserved on patched lines
    orig = OCE_TEMPLATE.splitlines()
    new = out.read_text().splitlines()
    assert len(orig) == len(new)
    for o, n in zip(orig, new):
        if o.strip().startswith(("K_GM_max", "Redi_Kmax")):
            assert "!" in n and n.split("!", 1)[1] == o.split("!", 1)[1]   # comment kept
        else:
            assert o == n                                       # byte-identical


def test_cvmix_roundtrip_and_decoy(tmp_path):
    tmpl = tmp_path / "namelist.cvmix"
    out = tmp_path / "namelist.cvmix.tuned"
    tmpl.write_text(CVMIX_TEMPLATE)
    wn.write_namelist({"tke_c_k": 0.15, "tke_c_eps": 0.85, "tke_cd": 4.0}, tmpl, out)
    parsed = _parse(out.read_text())
    assert parsed["tke_c_k"] == 0.15
    assert parsed["tke_c_eps"] == 0.85
    assert parsed["tke_cd"] == 4.0
    assert parsed["tke_alpha"] == 30.0          # untouched
    assert parsed["tke_kappaM_max"] == 100.0    # DECOY untouched


def test_not_found_keys_reported(tmp_path):
    """TKE keys are absent from namelist.oce → reported as not_found (so the caller patches the
    other file)."""
    tmpl = tmp_path / "namelist.oce"; out = tmp_path / "out"
    tmpl.write_text(OCE_TEMPLATE)
    res = wn.write_namelist({"K_GM_max": 1500.0, "tke_c_k": 0.2}, tmpl, out)
    assert "K_GM_max" in res["patched"]
    assert "tke_c_k" in res["not_found"]


def test_raw_namelist_key_path_also_syncs_redi(tmp_path):
    """The CLI / raw-namelist-key path (`{'K_GM_max': ...}`) also auto-syncs Redi_Kmax."""
    tmpl = tmp_path / "namelist.oce"; out = tmp_path / "out"
    tmpl.write_text(OCE_TEMPLATE)
    res = wn.write_namelist({"K_GM_max": 1500.0}, tmpl, out)
    parsed = _parse(out.read_text())
    assert parsed["K_GM_max"] == 1500.0 and parsed["Redi_Kmax"] == 1500.0
    assert "Redi_Kmax" in res["patched"]


def test_idempotent_repatch(tmp_path):
    """Patching the output again with the same values is a no-op (round-trip stable)."""
    tmpl = tmp_path / "n"; out1 = tmp_path / "o1"; out2 = tmp_path / "o2"
    tmpl.write_text(OCE_TEMPLATE)
    wn.write_namelist({"k_gm": 1234.0}, tmpl, out1)
    wn.write_namelist({"k_gm": 1234.0}, out1, out2)
    assert out1.read_text() == out2.read_text()


def test_against_real_namelist_if_present(tmp_path):
    """Smoke test on the actual fesom2 namelist.oce if reachable on this host."""
    real = Path("/home/a/a270088/port2/fesom2/work_all3/namelist.oce")
    if not real.exists():
        pytest.skip("real fesom2 namelist.oce not present")
    out = tmp_path / "namelist.oce"
    res = wn.write_namelist({"k_gm": 1500.0}, real, out)
    parsed = _parse(out.read_text())
    assert parsed["K_GM_max"] == 1500.0 and parsed["Redi_Kmax"] == 1500.0
    assert res["patched"] == {"K_GM_max": 1500.0, "Redi_Kmax": 1500.0}
    # nothing else changed: line count identical, only the two keys differ
    diff = [(o, n) for o, n in zip(real.read_text().splitlines(), out.read_text().splitlines())
            if o != n]
    assert all(d[1].strip().startswith(("K_GM_max", "Redi_Kmax")) for d in diff)


def test_fortran_transfer_ok_token(tmp_path):
    tmpl = tmp_path / "n"; out = tmp_path / "o"
    tmpl.write_text(OCE_TEMPLATE)
    res = wn.write_namelist({"k_gm": 1500.0}, tmpl, out)
    parsed = _parse(out.read_text())
    ok = parsed["K_GM_max"] == 1500.0 and parsed["Redi_Kmax"] == 1500.0 and parsed["K_GM_min"] == 2.0
    assert ok and res["patched"]
    print("FORTRAN_TRANSFER_OK")
