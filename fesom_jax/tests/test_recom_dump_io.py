"""The REcoM Fortran reference dumps (lane F, docs/recom/FORTRAN_REFERENCE.md) load through
``fesom_jax.io_dump.load_recom_dump``: the format round trip (always), and the shapes and tracer
IDs of the reference run (skipped when the reference directory is absent).

The reference lives outside the repository; ``FESOM_RECOM_REF_DIR`` overrides the default
``/work/ab0995/a270088/recom_ref``.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from fesom_jax.io_dump import (load_recom_dump, recom_dump_steps, recom_dump_tracer_ids,
                               write_recom_dump)

REF = Path(os.environ.get("FESOM_RECOM_REF_DIR", "/work/ab0995/a270088/recom_ref"))
RUN_30D = REF / "pi_30d_jan" / "dumps"

# bin_2p3z2d: 2 phyto, 3 zoo, 2 detritus, no coccos -> 31 BGC tracers, 1001-1030 + 1037
BGC_IDS = list(range(1001, 1031)) + [1037]
REMIN_IDS = [1001, 1002, 1003, 1018, 1019, 1022]
SINKING_IDS = [1004, 1005, 1006, 1007, 1008, 1013, 1014, 1015, 1016, 1017, 1020, 1021,
               1025, 1026, 1027, 1028]
NOD2D, ELEM2D, NL = 3140, 5839, 48  # pi


def test_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    fields = {
        "recom_in.tr1001": rng.standard_normal((7, 5)),
        "transport.uv": rng.standard_normal((4, 5, 2)),
        "recom_in.dt": np.array([2700.0]),
        "mesh.gid_nod": np.arange(1, 8, dtype=np.int32),
    }
    meta = {"mstep": 17, "yearnew": 1958, "daynew": 1, "timenew": 45900.0, "month": 1,
            "ndpyr": 365, "dt": 2700.0, "nod2D": 7, "elem2D": 4, "nl": 6}
    write_recom_dump(tmp_path, 17, fields, meta)
    assert recom_dump_steps(tmp_path) == [17]
    got, gmeta = load_recom_dump(tmp_path, 17)
    assert gmeta["order"] == list(fields)
    for k, v in fields.items():
        assert got[k].shape == v.shape and got[k].dtype.kind == v.dtype.kind
        np.testing.assert_array_equal(got[k], v)
    for k in ("mstep", "nod2D", "nl"):
        assert gmeta[k] == meta[k]
    assert gmeta["timenew"] == 45900.0
    only, _ = load_recom_dump(tmp_path, 17, names=["transport.uv"])
    assert list(only) == ["transport.uv"]


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_recom_dump(tmp_path, 1)
    write_recom_dump(tmp_path, 1, {"a": np.zeros(3)}, {})
    with pytest.raises(KeyError):
        load_recom_dump(tmp_path, 1, names=["b"])
    with pytest.raises(FileExistsError):  # never overwrite a dump
        write_recom_dump(tmp_path, 1, {"a": np.zeros(3)}, {})
    d = tmp_path / "step_000001"
    (d / "index.txt").write_text((d / "index.txt").read_text() + "a f8 1 3 24\n")
    with pytest.raises(ValueError):  # duplicate record name
        load_recom_dump(tmp_path, 1)


ref = pytest.mark.skipif(not RUN_30D.is_dir(), reason=f"REcoM reference dumps absent ({RUN_30D})")


@ref
def test_reference_steps_and_header():
    assert recom_dump_steps(RUN_30D) == [1, 2, 17, 224, 225, 960]
    _, meta = load_recom_dump(RUN_30D, 1, names=["recom_in.dt"])
    assert (meta["nod2D"], meta["elem2D"], meta["nl"]) == (NOD2D, ELEM2D, NL)
    assert meta["dt"] == 2700.0 and meta["mstep"] == 1 and meta["yearnew"] == 1958


@ref
def test_reference_shapes_and_tracer_ids():
    f, meta = load_recom_dump(RUN_30D, 1)
    for stage in ("recom_in", "recom_out", "step_end"):
        assert recom_dump_tracer_ids(f, stage) == [1, 2] + BGC_IDS
    np.testing.assert_array_equal(f["mesh.gid_nod"], np.arange(1, NOD2D + 1))
    assert sorted(f["mesh.gid_elem"]) == list(range(1, ELEM2D + 1))
    assert f["recom_in.tr1001"].shape == (NOD2D, NL - 1)
    assert f["recom_in.a_ice"].shape == (NOD2D,)
    assert f["recom_in.Benthos"].shape == (4, NOD2D)
    assert f["recom_out.CO23D"].shape == (NOD2D, NL - 1)
    assert f["transport.uv"].shape == (ELEM2D, NL - 1, 2)
    assert f["transport.w"].shape == (NOD2D, NL)
    assert f["transport.Kv"].shape == (NOD2D, NL)
    assert f["transport.slope_tapered"].shape == (NOD2D, NL - 1, 3)
    assert f["vel_rhs.slp_force"].shape == (ELEM2D, 2)
    for i in REMIN_IDS:
        assert f[f"tra_bf.tr{i}.dtr_bf"].shape == (NOD2D, NL - 1)
    for i in SINKING_IDS:
        assert f[f"tra_sink.tr{i}.vert_sink"].shape == (NOD2D, NL - 1)
        assert f[f"tra_ball.tr{i}.rho_particle1"].shape == (NOD2D, NL - 1)
        assert f[f"tra_sink.tr{i}.Benthos_tr"].shape == (4, NOD2D)
    for i in [1, 2] + BGC_IDS:
        for stage in ("tra_in", "tra_adv", "tra_redi", "tra_tstar", "tra_impl", "tra_out"):
            assert f"{stage}.tr{i}.values" in f
    # dumped steps carry finite values; l_mslp is on (press_air ~ 1e5 Pa)
    for k in ("recom_out.tr1002", "recom_out.GloCO2flux_seaicemask", "step_end.tr1022"):
        assert np.isfinite(f[k]).all(), k
    assert 9.0e4 < f["recom_in.press_air"].mean() < 1.1e5
