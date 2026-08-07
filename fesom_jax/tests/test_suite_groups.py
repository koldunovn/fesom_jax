"""The suite's group manifest must cover every test file — enforced as a test, not as a habit.

``scripts/runs/suite_groups.txt`` says which cost group each ``test_*.py`` runs in. Before this
existed, the runner had three groups — ice, sharding, and "everything except those" — and the
third was a default rather than a category. It reached 920 tests and 3:28 of wall time without
anyone choosing that, because a new file joined it by doing nothing at all and nothing reported
per-test cost.

Checking the manifest from inside the suite (rather than only in the sbatch that reads it) means
the failure lands on whoever adds the file, in the run they are already doing, instead of on
whoever next looks at why a group takes three and a half hours.
"""

from __future__ import annotations

import collections
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
MANIFEST = TESTS_DIR.parents[1] / "scripts" / "runs" / "suite_groups.txt"
VALID_GROUPS = {"unit", "core2", "ice", "sharding"}

# A file in `unit` that touches any of these is really a CORE2-cost file and belongs in `core2`;
# putting it in `unit` is how a group silently acquires minutes-long tests.
CORE2_MARKERS = ("mesh_core2", "phc_initial_state", "ic_core2", "build_surface_forcing",
                 "CORE2_DIST")

# This file necessarily contains every marker above — they are the list right there — so a
# textual scan flags it as CORE2 cost. It loads no mesh; it reads one text file. Exempt by name
# rather than by making the scan cleverer, because a cleverer scan is a thing that can be wrong
# quietly, and this exemption cannot.
MARKER_SCAN_EXEMPT = {Path(__file__).name}


def _manifest():
    if not MANIFEST.is_file():
        pytest.skip(f"group manifest missing: {MANIFEST}")
    entries = {}
    dupes = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        group, _, name = line.partition(" ")
        name = name.strip()
        if name in entries:
            dupes.append(name)
        entries[name] = group
    return entries, dupes


def test_every_test_file_is_classified():
    entries, _ = _manifest()
    on_disk = {p.name for p in TESTS_DIR.glob("test_*.py")}
    missing = sorted(on_disk - set(entries))
    assert not missing, (
        f"these test files are in no suite group: {missing}\n"
        f"Add each to {MANIFEST.name} — 'core2' if it loads the CORE2 mesh/IC/forcing, "
        f"'unit' otherwise. An unclassified file used to be silently added to the biggest "
        f"group, which is how that group grew to 3.5 hours unnoticed.")


def test_the_manifest_has_no_stale_entries():
    entries, _ = _manifest()
    on_disk = {p.name for p in TESTS_DIR.glob("test_*.py")}
    stale = sorted(set(entries) - on_disk)
    assert not stale, (f"{MANIFEST.name} lists files that do not exist: {stale} — a renamed "
                       f"file would otherwise stop running with nothing to show for it")


def test_no_file_is_listed_twice():
    _, dupes = _manifest()
    assert not dupes, f"listed in more than one group (it would run twice): {sorted(set(dupes))}"


def test_all_groups_are_known():
    entries, _ = _manifest()
    bad = {n: g for n, g in entries.items() if g not in VALID_GROUPS}
    assert not bad, (f"unknown group names {bad}; the runner only has an array task for "
                     f"{sorted(VALID_GROUPS)}, so anything else would never run")


def test_every_group_is_non_empty():
    entries, _ = _manifest()
    have = collections.Counter(entries.values())
    empty = sorted(VALID_GROUPS - set(have))
    assert not empty, (f"groups with no files: {empty} — the runner would launch an array task "
                       f"that collects nothing and reports success")


def test_the_conventions_that_define_the_groups_hold():
    """The two mechanical rules, so the manifest cannot drift away from what its header says."""
    entries, _ = _manifest()
    wrong = []
    for name, group in sorted(entries.items()):
        p = TESTS_DIR / name
        if not p.is_file():
            continue
        if name.startswith("test_ice_") and group != "ice":
            wrong.append(f"{name}: named test_ice_* but in {group!r}")
        if (name.endswith("_sharded.py") or name == "test_halo.py") and group != "sharding":
            wrong.append(f"{name}: needs 4 fake devices but is in {group!r}")
        if group == "unit" and name not in MARKER_SCAN_EXEMPT:
            src = p.read_text()
            hit = [m for m in CORE2_MARKERS if m in src]
            if hit:
                wrong.append(f"{name}: in 'unit' but references {hit} — that is CORE2 cost")
    assert not wrong, "manifest violates its own conventions:\n  " + "\n  ".join(wrong)


def test_the_cost_split_is_still_worth_having():
    """If `unit` ever stops being the large-and-cheap group, the split has lost its point and
    should be revisited rather than left in place looking like it works."""
    entries, _ = _manifest()
    have = collections.Counter(entries.values())
    assert have["unit"] > have["core2"], (
        f"unit={have['unit']} files, core2={have['core2']} — the cheap group is supposed to be "
        f"the big one; if that has inverted, re-derive the grouping")
