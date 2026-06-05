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
