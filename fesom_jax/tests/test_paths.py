"""Input-path resolution (:mod:`fesom_jax.paths`) — DATA-FREE: these must pass off Levante.

The contract three ways: the precedence chain (explicit > env > Levante default), the
actionable ``require()`` failure (it must name the env var so a new user can act on it), and the
run-YAML ``forcing:`` path keys (round-trip + strict keys). Nothing here opens a file, and
nothing here touches the numerics — the whole point of the refactor is that on Levante, with no
env vars and no YAML keys, every resolve() returns exactly the string that used to be hardcoded.
"""
from __future__ import annotations

import pytest

from fesom_jax import paths
from fesom_jax.run_config import RunConfig


# --------------------------------------------------------------------------
# precedence: explicit > env > default
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", paths.NAMES)
def test_default_when_no_env(name, monkeypatch):
    monkeypatch.delenv(paths.spec(name).env, raising=False)
    assert paths.resolve(name) == paths.spec(name).default


@pytest.mark.parametrize("name", paths.NAMES)
def test_env_beats_default(name, monkeypatch):
    monkeypatch.setenv(paths.spec(name).env, "/env/path")
    assert paths.resolve(name) == "/env/path"


@pytest.mark.parametrize("name", paths.NAMES)
def test_explicit_beats_env_and_default(name, monkeypatch):
    monkeypatch.setenv(paths.spec(name).env, "/env/path")
    assert paths.resolve(name, "/explicit/path") == "/explicit/path"


def test_levante_defaults_are_the_historical_strings(monkeypatch):
    # the exact strings that used to be hardcoded in jra55/sss_runoff/phc_ic/partit — with no
    # env override, resolve() must still return them (byte-identical default behaviour)
    for n in ("jra_dir", "phc_path", "sss_path", "runoff_path", "chl_path", "mesh_root"):
        monkeypatch.delenv(paths.spec(n).env, raising=False)
    assert paths.resolve("jra_dir") == "/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0"
    assert paths.LEVANTE_JRA_DIR == "/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0"
    assert paths.LEVANTE_PHC_PATH == "/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc"
    assert paths.LEVANTE_SSS_PATH == "/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/PHC2_salx.nc"
    assert paths.LEVANTE_RUNOFF_PATH == "/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/CORE2_runoff.nc"
    assert paths.LEVANTE_CHL_PATH == "/pool/data/AWICM/FESOM2/FORCING/Sweeney/Sweeney_2005.nc"
    assert paths.LEVANTE_MESH_ROOT == "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1"


def test_env_read_at_call_time(monkeypatch):
    # late binding: the module snapshots nothing at import — setting the var after import works
    monkeypatch.delenv("FESOM_JRA_DIR", raising=False)
    before = paths.resolve("jra_dir")
    monkeypatch.setenv("FESOM_JRA_DIR", "/late/bound")
    assert paths.resolve("jra_dir") == "/late/bound" != before


def test_empty_env_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("FESOM_JRA_DIR", "")
    assert paths.resolve("jra_dir") == paths.LEVANTE_JRA_DIR


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        paths.resolve("no_such_input")


# --------------------------------------------------------------------------
# require(): existence check with an ACTIONABLE message
# --------------------------------------------------------------------------
def test_require_missing_names_env_var_and_yaml_key(tmp_path):
    missing = str(tmp_path / "definitely_absent")
    with pytest.raises(FileNotFoundError) as ei:
        paths.require("jra_dir", missing)
    msg = str(ei.value)
    assert missing in msg                    # (a) which path
    assert "JRA55-do forcing" in msg         # (b) what the data is
    assert "FESOM_JRA_DIR" in msg            # (c) the env var to set
    assert "forcing.jra_dir" in msg          # (d) the run-YAML key
    assert "docs/DATA.md" in msg             # (e) where to read more


@pytest.mark.parametrize("name", paths.NAMES)
def test_require_message_always_names_its_env_var(name, tmp_path):
    with pytest.raises(FileNotFoundError) as ei:
        paths.require(name, str(tmp_path / "absent"))
    assert paths.spec(name).env in str(ei.value)


def test_require_returns_existing_path(tmp_path):
    p = tmp_path / "phc.nc"
    p.write_text("")
    assert paths.require("phc_path", str(p)) == str(p)


def test_describe_covers_every_input():
    rows = paths.describe()
    assert {r["name"] for r in rows} == set(paths.NAMES)
    assert all(r["env"].startswith("FESOM_") for r in rows)


# --------------------------------------------------------------------------
# run-YAML `forcing:` path keys
# --------------------------------------------------------------------------
def test_forcing_path_keys_roundtrip():
    forcing = {"kind": "core2", "start_year": 1958,
               "jra_dir": "/data/jra", "sss_path": "/data/sss.nc",
               "runoff_path": "/data/runoff.nc", "chl_path": "/data/chl.nc"}
    cfg = RunConfig(forcing=forcing)
    cfg.validate()
    back = RunConfig.from_dict(cfg.to_dict())
    assert back == cfg and back.forcing == forcing
    assert back.forcing_paths() == {"jra_dir": "/data/jra", "sss_path": "/data/sss.nc",
                                    "runoff_path": "/data/runoff.nc", "chl_path": "/data/chl.nc"}


def test_forcing_without_path_keys_resolves_to_none():
    # the historical config (kind + start_year only) ⇒ every path kwarg None ⇒ env/default
    cfg = RunConfig(forcing={"kind": "core2", "start_year": 1958})
    cfg.validate()
    assert cfg.forcing_paths() == {"jra_dir": None, "sss_path": None,
                                   "runoff_path": None, "chl_path": None}
    assert RunConfig().forcing_paths() == cfg.forcing_paths()     # forcing absent entirely


def test_unknown_forcing_key_raises():
    d = {"forcing": {"kind": "core2", "start_year": 1958, "jra_dirs": "/typo"}}
    with pytest.raises(KeyError, match="forcing"):
        RunConfig.from_dict(d)
    with pytest.raises(KeyError, match="forcing"):
        RunConfig(forcing=d["forcing"]).validate()


def test_forcing_must_be_a_mapping():
    with pytest.raises(TypeError):
        RunConfig.from_dict({"forcing": "core2"})


def test_forcing_yaml_roundtrip(tmp_path):
    from fesom_jax.run_config import load_yaml
    cfg = RunConfig(forcing={"kind": "core2", "start_year": 1958, "jra_dir": "/data/jra"},
                    dt=180.0)
    p = tmp_path / "run.yaml"
    cfg.to_yaml(p)
    assert load_yaml(p) == cfg


# --------------------------------------------------------------------------
# the reader modules still expose their legacy names, sourced from paths
# --------------------------------------------------------------------------
def test_reader_modules_expose_legacy_defaults():
    from fesom_jax import jra55, phc_ic, sss_runoff
    assert jra55.DEFAULT_JRA_DIR == paths.resolve("jra_dir")
    assert sss_runoff.DEFAULT_SSS_PATH == paths.resolve("sss_path")
    assert sss_runoff.DEFAULT_RUNOFF_PATH == paths.resolve("runoff_path")
    assert sss_runoff.DEFAULT_CHL_PATH == paths.resolve("chl_path")
    assert phc_ic.DEFAULT_PHC_PATH == paths.resolve("phc_path")
    assert str(phc_ic.DEFAULT_IC_DIR).endswith("ic_core2")       # repo-relative, not /pool


def test_default_mesh_dir_composes_from_root(monkeypatch):
    from fesom_jax import partit
    monkeypatch.setenv("FESOM_MESH_ROOT", "/data/meshes")
    assert str(partit.default_mesh_dir("core2")) == "/data/meshes/core2"
    assert str(partit.default_mesh_dir("core2", "/other")) == "/other/core2"
