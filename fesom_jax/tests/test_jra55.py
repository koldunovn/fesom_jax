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


def test_reopen_year_keeps_stencil_and_switches_data(jra):
    """``reopen_year`` swaps the forcing YEAR in place: the bilinear stencil (idx4/weights) and the
    wind rotation are year-independent and KEPT (same array objects) — only the data source moves
    to the new year's files. Backs the multi-year run (run.py rolls the reader at a year boundary)."""
    from fesom_jax import jra55
    if not (JRA_DIR / f"uas.{YEAR + 1}.nc").is_file():
        pytest.skip(f"JRA {YEAR + 1} data not available")
    r = jra55.JRA55Reader(jra["mesh"], YEAR, JRA_DIR)
    idx4, dx4, dy4, denom, M = r.idx4, r.dx4, r.dy4, r.denom, r.M
    f0 = r.step(YEAR, 1, 0.0)                              # Jan 1 of YEAR
    assert r.reopen_year(YEAR) is r and r.year == YEAR     # same-year reopen is a no-op
    r.reopen_year(YEAR + 1)
    assert r.year == YEAR + 1
    # the expensive interpolation knowledge is REUSED, not rebuilt (identity)
    assert r.idx4 is idx4 and r.dx4 is dx4 and r.dy4 is dy4 and r.denom is denom and r.M is M
    f1 = r.step(YEAR + 1, 1, 0.0)                          # Jan 1 of YEAR+1
    assert not np.array_equal(np.asarray(f0.u_wind), np.asarray(f1.u_wind))  # weather differs
    r.close()


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


def test_clamp_holds_and_releases(jra):
    """The clamped bracket (before a field's first record) must HOLD — one
    ``_getcoeffld`` per field on entry, NOT one per step (the C's clamp never
    released: ``lo == hi`` re-read + re-gathered a slice per field per step for the
    first 1.5 h of every run, 12 h for prra/prsn, and again at every year end —
    port_kokkos fix ``7f64be1``). And it must RELEASE: crossing a field's first
    record refreshes exactly that field, exactly once. Values in the window are the
    constant first-slice extrapolation (also pinned vs C by
    ``test_fields_match_c[d1]``, which runs at day 1 sec 0 — inside the window)."""
    import collections
    from fesom_jax import jra55
    r = jra55.JRA55Reader(jra["mesh"], YEAR, JRA_DIR)
    calls = collections.Counter()
    orig = r._getcoeffld
    r._getcoeffld = lambda f, rdate: (calls.update([f.var]), orig(f, rdate))[1]

    # every field's time axis is mid-interval-shifted ⇒ its first record sits
    # strictly after Jan 1 00:00 — the whole point of the clamp
    first = {f.var: float(f.nc_time[0]) for f in r.fields}
    rdate0 = float(jra55._julday(YEAR, 1, 1, r.fields[0].calendar))
    assert all(t0 > rdate0 for t0 in first.values())

    # three dt=1800 steps inside EVERY field's pre-first-record window
    secs = (0.0, 1800.0, 3600.0)
    assert all(rdate0 + s / 86400.0 < min(first.values()) for s in secs)
    outs = [r.step(YEAR, 1, s) for s in secs]
    assert calls == {v: 1 for v in jra55.JRA_VARS}, \
        f"clamp re-built per step: {calls}"
    for o in outs[1:]:                       # constant extrapolation, bit-identical
        for name in FIELDS:
            np.testing.assert_array_equal(np.asarray(getattr(o, name)),
                                          np.asarray(getattr(outs[0], name)))

    # release: step past the earliest first record → exactly the crossed fields
    # refresh once; the still-clamped ones (e.g. daily prra/prsn) stay quiet
    sec_in = (min(first.values()) - rdate0) * 86400.0 + 1800.0
    rdate_in = rdate0 + sec_in / 86400.0     # the reader's own rdate expression
    calls.clear()
    r.step(YEAR, 1, sec_in)
    expected = {v: 1 for v, t0 in first.items() if rdate_in > t0}
    assert expected and calls == expected, (dict(calls), expected)
    r.close()


def test_g2r_trig_cache_is_bit_identical(jra):
    """``_vector_g2r`` fed the precomputed ``_g2r_trig`` table (the reader's per-step
    path) must reproduce the on-the-fly trig path bit-for-bit."""
    from fesom_jax import jra55
    geo = np.asarray(jra["mesh"].geo_coord_nod2D)
    rot = np.asarray(jra["mesh"].coord_nod2D)
    glon, glat, rlon, rlat = geo[:, 0], geo[:, 1], rot[:, 0], rot[:, 1]
    M = jra55._rotation_matrix()
    rng = np.random.default_rng(1)
    u0 = rng.normal(size=glon.shape[0])
    v0 = rng.normal(size=glon.shape[0])
    a = jra55._vector_g2r(u0, v0, glon, glat, rlon, rlat, M)
    b = jra55._vector_g2r(u0, v0, glon, glat, rlon, rlat, M,
                          trig=jra55._g2r_trig(glon, glat, rlon, rlat))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


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


def test_bracket_schedule_matches_step(jra):
    """``bracket_schedule`` (the on-device-forcing host seam): the per-chunk tables +
    schedule reproduce ``step()`` BIT-for-bit over a window with the year-start clamp,
    several 3-hourly rolls and the daily prra/prsn bracket — and ``_getcoeffld`` fires
    per BRACKET (~once per roll), not per step."""
    import collections
    from fesom_jax import jra55

    T = 60                                    # 30 h of dt=1800 from Jan 1 00:00
    dates = [(YEAR, 1 + int(i * 1800.0 // 86400), (i * 1800.0) % 86400.0, 1)
             for i in range(T)]

    r_ref = jra55.JRA55Reader(jra["mesh"], YEAR, JRA_DIR)
    calls_ref = collections.Counter()
    orig_ref = r_ref._getcoeffld
    r_ref._getcoeffld = lambda f, rdate: (calls_ref.update([f.var]), orig_ref(f, rdate))[1]
    ref = [r_ref.step(y, d, s) for (y, d, s, _m) in dates]
    r_ref.close()

    r = jra55.JRA55Reader(jra["mesh"], YEAR, JRA_DIR)
    calls = collections.Counter()
    orig = r._getcoeffld
    r._getcoeffld = lambda f, rdate: (calls.update([f.var]), orig(f, rdate))[1]
    nb = int(T * 1800) // 10800 + 2
    ca, cb, rdates, fidx = r.bracket_schedule(dates, nb)
    assert ca.shape == (jra55.N_FLD, nb, r.N) and fidx.shape == (T, jra55.N_FLD)
    # bracket economy: EXACTLY the step()-walk's _getcoeffld count (one per bracket),
    # and every field fits the driver's nb formula
    assert calls == calls_ref, f"schedule {dict(calls)} != step walk {dict(calls_ref)}"
    assert all(c <= nb for c in calls.values()), f"nb formula too small: {calls}"

    trig = r._trig
    for i in range(T):
        vals = [rdates[i] * ca[f, fidx[i, f]] + cb[f, fidx[i, f]]
                for f in range(jra55.N_FLD)]
        u, v = vals[jra55.I_XWIND], vals[jra55.I_YWIND]
        if jra55.FORCE_ROTATION:
            u, v = jra55._vector_g2r(u, v, None, None, None, None, r.M, trig=trig)
        got = dict(u_wind=u, v_wind=v, shum=vals[jra55.I_HUMI],
                   shortwave=vals[jra55.I_QSR], longwave=vals[jra55.I_QLW],
                   Tair=vals[jra55.I_TAIR] - 273.15,
                   prec_rain=vals[jra55.I_PREC] / 1000.0,
                   prec_snow=vals[jra55.I_SNOW] / 1000.0)
        for name in FIELDS:
            np.testing.assert_array_equal(
                got[name], np.asarray(getattr(ref[i], name)),
                err_msg=f"step {i} field {name}")
    r.close()
    print("BRACKET_SCHEDULE_OK (bit-identical to step(), getcoeffld per bracket)")


def test_combine_device_matches_host_bit(jra):
    """``surface_forcing.combine_step_forcing`` (the in-scan device combine, run here
    on the test backend) reproduces the host reader's fields BIT-for-bit from the same
    tables — the op-order contract of the on-device forcing plan. Also pins the
    month-table selection (Ssurf/chl) against ``step_forcing``'s."""
    import jax
    import jax.numpy as jnp
    from fesom_jax import jra55
    from fesom_jax.surface_forcing import (ForcingTables, ForcingDeviceConst,
                                           combine_step_forcing)

    assert jax.config.jax_enable_x64, "combine gate needs x64"
    T = 12                                    # 6 h of dt=1800 from day 100 noon (interior)
    dates = [(YEAR, INTERIOR_DAY, INTERIOR_SEC + i * 1800.0, 4) for i in range(T)]

    r = jra55.JRA55Reader(jra["mesh"], YEAR, JRA_DIR)
    ref = [r.step(y, d, s) for (y, d, s, _m) in dates]

    r2 = jra55.JRA55Reader(jra["mesh"], YEAR, JRA_DIR)
    nb = int(T * 1800) // 10800 + 2
    ca, cb, rdates, fidx = r2.bracket_schedule(dates, nb)
    N = r2.N
    rng = np.random.default_rng(3)
    sss_tab = rng.normal(size=(2, N))
    chl_tab = rng.normal(size=(2, N))
    tables = ForcingTables(coef_a=jnp.asarray(ca), coef_b=jnp.asarray(cb),
                           sss=jnp.asarray(sss_tab), chl=jnp.asarray(chl_tab))
    const = ForcingDeviceConst(*[jnp.asarray(t) for t in r2._trig],
                               M=jnp.asarray(r2.M),
                               prec_div=jnp.asarray(np.float64(1000.0)))
    # Platform-divide probe: XLA CPU's f64 divide is NOT correctly rounded (1 ULP off
    # vs numpy on ~13 % of operands, measured; flags don't help), while CUDA's
    # div.rn.f64 is IEEE. Demand bit-exactness for prec_rain/prec_snow exactly where
    # the backend can deliver it, ≤1 ULP otherwise.
    probe = np.linspace(1e-9, 5e-6, 4096)
    div_ieee = bool(np.array_equal(
        np.asarray(jnp.asarray(probe) / jnp.asarray(np.float64(1000.0))),
        probe / 1000.0))

    sf_names = {"u_wind": "u_air", "v_wind": "v_air"}
    for i in range(T):
        sf = combine_step_forcing(tables, jnp.asarray(rdates[i]),
                                  jnp.asarray(fidx[i]), jnp.asarray(np.int32(i % 2)),
                                  const)
        for name in FIELDS:
            got = np.asarray(getattr(sf, sf_names.get(name, name)))
            want = np.asarray(getattr(ref[i], name))
            if name in ("prec_rain", "prec_snow") and not div_ieee:
                np.testing.assert_allclose(got, want, rtol=2.3e-16, atol=0.0,
                                           err_msg=f"step {i} field {name} (>1 ULP)")
            else:
                np.testing.assert_array_equal(got, want,
                                              err_msg=f"step {i} field {name}")
        np.testing.assert_array_equal(np.asarray(sf.Ssurf_month), sss_tab[i % 2])
        np.testing.assert_array_equal(np.asarray(sf.chl), chl_tab[i % 2])
    r.close(); r2.close()
    print(f"COMBINE_DEVICE_OK (bit-identical; prec {'bit' if div_ieee else '<=1 ULP, non-IEEE platform divide'})")
