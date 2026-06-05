"""Task 0.2 gate: the comparison harness behaves correctly at both tolerances.

Synthetic records (no real dumps needed) feed through compare_column to confirm
pass/fail at the per-kind tolerances, the nlevels truncation, and the round-trip
of the binary reader.
"""

import struct

import numpy as np
import pytest

from fesom_jax.io_dump import DumpRecord, SUBSTEP_NAMES, read_records, find_record
from fesom_jax.verify import compare_column, assert_close


def _rec(values, field="density", nlevels=None):
    values = np.asarray(values, dtype=np.float64)
    return DumpRecord(
        step=1,
        substep=1,
        probe_gid=1001,
        nlevels=nlevels if nlevels is not None else len(values),
        field=field,
        values=values,
    )


# --- compare_column -------------------------------------------------------

def test_identical_passes_and_zero_diff():
    c = _rec([1.0, 2.0, 3.0])
    r = compare_column(c.values.copy(), c, "map")
    assert r.passed
    assert r.max_abs == 0.0
    assert bool(r) is True


def test_within_map_tol_scales_with_magnitude():
    # rtol(map)=1e-15; at |c|=1000 the tolerance is ~1e-12, so 5e-13 passes.
    c = _rec([1.0, 1000.0])
    j = c.values.copy()
    j[1] += 5e-13
    assert compare_column(j, c, "map").passed


def test_above_map_tol_fails():
    c = _rec([1.0, 1.0])
    j = c.values.copy()
    j[0] += 1e-9
    r = compare_column(j, c, "map")
    assert not r.passed
    assert r.worst_level == 0


def test_kind_tolerances_differ():
    # A 1e-13 diff on an O(1) value: over the map floor (~1e-14) but under scatter (~1e-11).
    c = _rec([1.0])
    j = np.array([1.0 + 1e-13])
    assert not compare_column(j, c, "map").passed
    assert compare_column(j, c, "scatter").passed


def test_truncation_to_nlevels():
    # JAX column carries below-bottom padding; only the top nlevels are compared.
    c = _rec([1.0, 2.0, 3.0], nlevels=3)
    j = np.array([1.0, 2.0, 3.0, 9e9, -9e9])  # garbage below bottom
    assert compare_column(j, c, "map").passed


def test_too_short_raises():
    c = _rec([1.0, 2.0, 3.0], nlevels=3)
    with pytest.raises(ValueError):
        compare_column(np.array([1.0, 2.0]), c, "map")


def test_unknown_kind_raises():
    c = _rec([1.0])
    with pytest.raises(ValueError):
        compare_column(c.values, c, "nonsense")


def test_assert_close_raises_with_report():
    c = _rec([1.0])
    with pytest.raises(AssertionError):
        assert_close(np.array([2.0]), c, "map")
    # and returns a result when it passes
    assert assert_close(c.values.copy(), c, "map").passed


def test_custom_tolerance_override():
    c = _rec([1.0])
    j = np.array([1.0 + 1e-6])
    assert not compare_column(j, c, "map").passed
    assert compare_column(j, c, "map", atol=1e-5).passed


# --- io_dump binary round-trip -------------------------------------------

def test_read_records_roundtrip(tmp_path):
    # Write two records in the shim's exact layout, read them back.
    recs = [
        (1, 1, 1001, 3, b"density", np.array([1.0, 2.0, 3.0])),
        (1, 8, 1001, 2, b"ssh_rhs", np.array([-4.5, 6.25e-3])),
    ]
    path = tmp_path / "probe.dump"
    with open(path, "wb") as fh:
        for step, sub, gid, nlev, name, vals in recs:
            fh.write(struct.pack("<iiii24s", step, sub, gid, nlev, name.ljust(24)))
            fh.write(struct.pack(f"<{nlev}d", *vals))

    got = list(read_records(path))
    assert len(got) == 2
    assert got[0].field == "density"
    assert got[0].substep_name == SUBSTEP_NAMES[1]
    np.testing.assert_array_equal(got[0].values, [1.0, 2.0, 3.0])
    assert got[1].field == "ssh_rhs"
    assert got[1].substep == 8

    # find_record by name + numeric substep
    r = find_record(path, step=1, substep="ssh_rhs", field="ssh_rhs")
    assert r.nlevels == 2
    with pytest.raises(LookupError):
        find_record(path, step=1, substep=0, field="nope")
