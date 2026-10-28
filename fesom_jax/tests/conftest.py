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


def _map_count() -> int:
    """Mappings this process holds, against ``/proc/sys/vm/max_map_count``. -1 off Linux."""
    try:
        with open("/proc/self/maps") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return -1


def _map_limit() -> int:
    try:
        with open("/proc/sys/vm/max_map_count") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return -1


_PEAK_MAPS = 0


@pytest.fixture(scope="module", autouse=True)
def _bound_the_jit_footprint():
    """Drop JAX's compilation caches when a test MODULE finishes.

    ⚠️ This is not tidiness; without it the `unit` group cannot pass. Every compiled executable
    XLA's CPU JIT holds costs several process memory mappings, JAX keeps them alive in caches
    keyed on the traced callable, and nothing releases them for the life of the process. Measured
    across the group (job 26854740, one line per test): mappings climb to **56,521 against
    ``vm.max_map_count`` = 65,530** by the last file — 86 % of the ceiling — and the next test
    compiles the largest graph in the suite, a non-rematerialised reverse pass. Its ``mmap`` gets
    ENOMEM, which XLA reports as ``LLVM compilation error: Cannot allocate memory`` (194 of them,
    one per parallel codegen unit) and then ``INTERNAL: Failed to materialize symbols``. It reads
    as a numerical or an adjoint failure and is neither.

    Clearing at module boundaries holds the same run to **32,564, i.e. 50 % of the ceiling**
    (measured, same job, arm 1) for a few per cent more wall time from recompilation. It does not
    fire inside a module, so nothing that depends on a warm cache within one file is affected.

    ⚠️ **50 % is headroom, not safety.** Clearing does not free what one FILE accumulates — the
    last file alone goes from 5,308 to 32,564 — so a group that grows, or one long test file that
    compiles more variants, walks back toward the same wall. The count printed at the end of every
    run is the instrument for noticing that; the bar to watch is the percentage, not this number.

    ⚠️ **The resource is the map COUNT, not memory.** Peak resident was 17 GB against a 300 GB
    request, so every reading that starts from "out of memory" — a bigger ``--mem``, a smaller
    working set — is looking at the wrong number. Two earlier explanations died here: an
    ``lru_cache(maxsize=32)`` on compiled executables (real, worth fixing, not this), and a test
    helper defeating that cache (also real, also not this). The count is now printed at the end
    of every run so the next approach to the ceiling is visible before it is a failure."""
    yield
    # Sample BEFORE clearing: within a module the count only grows (compiles accumulate and
    # nothing releases them), so a module's maximum is exactly here. Sampling only at exit — which
    # is what this file did first — reads the count AFTER the last teardown has already cleared,
    # and reported 5,518 for a run whose real peak was 32,564. An instrument that under-reads by
    # 6x is worse than none: it is the reassuring number that would have to be wrong for anyone
    # to look.
    global _PEAK_MAPS
    _PEAK_MAPS = max(_PEAK_MAPS, _map_count())
    import jax
    jax.clear_caches()

# Pinned probe global id — one of the shim's hardcoded DUMP_PROBE_GIDS
# ([1001, 1500, 2000, 2500, 3000], fesom_dump_shim.F90:60). See plan Task 0.4.
PROBE_GID = 1001


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def probe_gid() -> int:
    return PROBE_GID


def pytest_collection_modifyitems(config, items):
    """``@pytest.mark.gpu`` tests (GPU kernels against their XLA reference, CORE2 GPU speed plan)
    run only where JAX's default backend is a GPU; everywhere else they SKIP, never fail."""
    if not any(item.get_closest_marker("gpu") for item in items):
        return
    import jax
    if jax.default_backend() == "gpu":
        return
    skip = pytest.mark.skip(reason=f"needs a GPU (default backend: {jax.default_backend()})")
    for item in items:
        if item.get_closest_marker("gpu"):
            item.add_marker(skip)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Close with an honest coverage summary (usability review P5): most verification gates
    read unpublished Fortran/C reference dumps and SKIP without them, so a green run off
    Levante must say what it did NOT verify — otherwise a newcomer reads 'green' as
    'numerics verified locally' when the gates never ran."""
    exit_used, limit = _map_count(), _map_limit()
    peak = max(_PEAK_MAPS, exit_used)
    if peak > 0 and limit > 0:
        # The resource that actually ran out once (see _bound_the_jit_footprint). Printed on
        # every run, green or not, because the failure it produces names memory and LLVM and
        # gives no hint that a per-process mapping ceiling is what was hit.
        #
        # ⚠️ PEAK is the number to read, and it is not the number at exit. The peak is sampled
        # per module just before the cache clear; `at exit` is measured here, after the last
        # clear, and on a real run it read 5,518 against a true peak of 32,564.
        terminalreporter.write_sep("-", "process memory mappings")
        terminalreporter.write_line(
            f"peak {peak} of {limit} (vm.max_map_count), {100 * peak / limit:.0f}%"
            f"   [{exit_used} at exit, after the final cache clear — not the peak]")
        if peak > limit // 2:
            terminalreporter.write_line(
                "WARNING: past half the ceiling. XLA's CPU JIT costs several mappings per "
                "compiled executable and they are never released; at ~85% the next large "
                "compile fails as 'Cannot allocate memory' / 'Failed to materialize symbols', "
                "which looks like a numerical failure and is not.")

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
