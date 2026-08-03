"""Shared pytest fixtures: locate the cached Fortran reference dumps.

Per-substep dump fixtures are produced by the Fortran shim (Task 0.4,
``docs/REFERENCE_RUNS.md``) and cached under ``fesom_jax/tests/fixtures/``.
Tests that need them depend on ``load_dump`` and **SKIP cleanly** until the
fixtures exist, so the suite stays green before Phase 1/2.
"""

from pathlib import Path

import pytest

from fesom_jax import io_dump

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Pinned probe global id — one of the shim's hardcoded DUMP_PROBE_GIDS
# ([1001, 1500, 2000, 2500, 3000], fesom_dump_shim.F90:60). See plan Task 0.4.
PROBE_GID = 1001


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def probe_gid() -> int:
    return PROBE_GID


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Close with an honest coverage summary (usability review P5): most verification gates
    read unpublished Fortran/C reference dumps and SKIP without them, so a green run off
    Levante must say what it did NOT verify — otherwise a newcomer reads 'green' as
    'numerics verified locally' when the gates never ran."""
    skipped = terminalreporter.stats.get("skipped", [])
    if not skipped:
        return
    needles = ("dump", "fixture missing", "reference")
    gated = sum(1 for r in skipped
                if any(n in str(getattr(r, "longrepr", "")).lower() for n in needles))
    terminalreporter.write_sep("-", "fesom-jax gate coverage")
    terminalreporter.write_line(
        f"{len(skipped)} test(s) skipped; {gated} of them are verification gates needing "
        f"Fortran/C reference dumps that are not published (docs/REFERENCE_RUNS.md).")
    if gated:
        terminalreporter.write_line(
            "A green result does NOT mean those gates ran — the vs-Fortran numerics "
            "verification only executes where the dumps exist. The data-free gates did run.")


@pytest.fixture
def load_dump(fixtures_dir):
    """Return ``load(name) -> list[DumpRecord]``; skips the test if the dump is
    not yet cached under fixtures/."""

    def _load(name: str):
        path = fixtures_dir / name
        if not path.exists():
            pytest.skip(
                f"reference dump fixture missing: {path} "
                f"(produce via Task 0.4 / docs/REFERENCE_RUNS.md)"
            )
        return io_dump.load_records(path)

    return _load
