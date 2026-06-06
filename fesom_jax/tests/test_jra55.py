"""Task 5.3 gate: the numpy JRA55 reader (``jra55.JRA55Reader``) reproduces the C
``fesom_jra55_step`` 8 physics fields on CORE2, verified against the C ``jra_dump_*``
all-node dumps at two dates:

  * ``d1_s0``    — (day 1, sec 0): the start-of-year boundary (constant-extrapolate the
    first time slice, ``t_indx`` boundary branch).
  * ``interior`` — (day 100, 12:00:00): genuine linear-in-time interpolation between two
    3-hourly slices + a ``getcoeffld`` cache refresh + the wind g2r rotation.

The reader reproduces the C bilinear gather **bit-for-bit** (same ``(s·dx)·dy`` order +
divide-at-end), so the linear-in-time fields come out **exactly** equal to the C: the 6
scalar fields are bit-identical (max|diff|=0 over all 126858 nodes, both dates) and only
the wind carries ~3.5e-15 (1-2 ULP from the g2r rotation's libm ``sin``/``cos``). This
bit-exactness matters: the C's time-interp ``field = rdate·coef_a + coef_b`` cancels two
~2.4e6 (Julian-day) numbers, so a mere ~1e-13 reassociation in the bilinear gather would
blow up to ~1e-8 in the interpolated field — folding ``1/denom`` into the weights does
exactly that, hence the bit-exact gather.

SKIPS unless the CORE2 mesh export, the C JRA dump, and the JRA55 NetCDF all exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CORE2_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
JRA_DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "jra_dump_core2"
JRA_DIR = Path("/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0")
YEAR = 1958
# The dump job's interior point (jobs/jax_jra_dump_core2.sh: DAY=100, SEC=43200).
INTERIOR_DAY = 100
INTERIOR_SEC = 43200.0

pytestmark = pytest.mark.skipif(
    not (CORE2_MESH_DIR.is_dir() and JRA_DUMP_DIR.is_dir()
         and (JRA_DIR / f"uas.{YEAR}.nc").is_file()),
    reason="needs CORE2 mesh export + C JRA dump + JRA55 NetCDF (Task 5.3 artifacts)",
)

# Map/gather class. Achieved: scalars bit-exact (0), wind ~3.5e-15 (rotation libm).
# 1e-12 keeps a ~300× margin over the wind floor while catching any real regression
# (a wrong field index / rotation / time bracket shows O(1) diffs).
ATOL = 1e-12
# Field-name → JRAFields attribute, in the dump's column order.
FIELDS = ["u_wind", "v_wind", "shum", "shortwave", "longwave", "Tair",
          "prec_rain", "prec_snow"]


@pytest.fixture(scope="module")
def jra():
    from fesom_jax import mesh as meshmod, jra55
    m = meshmod.load_mesh(CORE2_MESH_DIR)
    reader = jra55.JRA55Reader(m, YEAR, JRA_DIR)
    f_d1 = reader.step(YEAR, 1, 0.0)
    f_in = reader.step(YEAR, INTERIOR_DAY, INTERIOR_SEC)
    reader.close()

    def _load(tag):
        d = np.loadtxt(JRA_DUMP_DIR / f"jra_dump_{tag}_rank0.txt")
        return d[np.argsort(d[:, 0])]        # order by 1-based gid → index = gid-1

    return dict(mesh=m, d1=f_d1, inr=f_in,
                c_d1=_load("d1_s0"), c_in=_load("interior"))


def test_dump_is_full_mesh_in_order(jra):
    for key in ("c_d1", "c_in"):
        gid = jra[key][:, 0].astype(np.int64)
        assert np.array_equal(gid, np.arange(1, jra["mesh"].nod2D + 1))


def test_node_coords_match(jra):
    """Sanity: the reader's node order == the C dump's (same geo coords, deg). The C
    dump writes lon/lat as ``geo_coord_nod2D/RAD`` WITHOUT the <0 wrap, so compare the
    unwrapped degrees directly."""
    from fesom_jax.config import RAD
    geo = np.asarray(jra["mesh"].geo_coord_nod2D)
    assert np.max(np.abs(geo[:, 0] / RAD - jra["c_d1"][:, 9])) < 1e-9
    assert np.max(np.abs(geo[:, 1] / RAD - jra["c_d1"][:, 10])) < 1e-9


@pytest.mark.parametrize("tag", ["d1", "in"])
def test_fields_match_c(jra, tag):
    fields = jra["d1"] if tag == "d1" else jra["inr"]
    cdump = jra["c_d1"] if tag == "d1" else jra["c_in"]
    worst = {}
    for col, name in enumerate(FIELDS, start=1):
        got = np.asarray(getattr(fields, name), dtype=np.float64)
        ref = cdump[:, col]
        d = np.max(np.abs(got - ref))
        worst[name] = d
    bad = {k: v for k, v in worst.items() if not v < ATOL}
    assert not bad, f"[{tag}] fields exceeding atol={ATOL}: {bad}\n all: {worst}"


def test_wind_rotation_is_active(jra):
    """The integrated wind matching the (rotated) C dump in ``test_fields_match_c``
    already proves the rotation fired correctly. Here pin the two properties it must
    have: magnitude preservation and being a genuine (non-identity) rotation."""
    from fesom_jax import jra55
    M = jra55._rotation_matrix()
    # Use the mesh's own (geographic, rotated) coord pairs — a consistent g2r mapping,
    # which is the only case where the vector rotation is magnitude-preserving.
    geo = np.asarray(jra["mesh"].geo_coord_nod2D)
    rot = np.asarray(jra["mesh"].coord_nod2D)
    sub = slice(0, 5000)
    glon, glat = geo[sub, 0], geo[sub, 1]
    rlon, rlat = rot[sub, 0], rot[sub, 1]
    rng = np.random.default_rng(0)
    u0 = rng.normal(size=glon.shape[0])
    v0 = rng.normal(size=glon.shape[0])
    u1, v1 = jra55._vector_g2r(u0, v0, glon, glat, rlon, rlat, M)
    # magnitude-preserving (a rotation): |rot| == |geo|.
    assert np.allclose(np.hypot(u1, v1), np.hypot(u0, v0), atol=1e-10)
    # non-trivial: the rotation genuinely changes direction (not the identity).
    assert np.max(np.abs(u1 - u0)) > 0.1
    # and the integrated field is physical.
    spd = np.hypot(np.asarray(jra["inr"].u_wind), np.asarray(jra["inr"].v_wind))
    assert np.isfinite(spd).all() and 1.0 < spd.max() < 60.0
