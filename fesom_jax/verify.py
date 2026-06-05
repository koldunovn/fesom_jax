"""Per-substep probe-column comparison — rung 1 of the verification ladder.

Compares a JAX-computed probe column against a Fortran reference
:class:`~fesom_jax.io_dump.DumpRecord` with a per-kind tolerance, mirroring the
climate-close fidelity target (see the plan, "Why bit-identity is not the
target"):

  - **map / gather** kernels  → ~1e-15 (FMA / transcendental differences only)
  - **scatter / reduction**   → ~1e-12 (``segment_sum`` / ``.at[].add`` reassociate
    the FP sum; JAX has no bit-identical edge order)

Field magnitudes span many orders (``eta``~O(1), ``density``~1e3,
``pressure``~1e7), so an *absolute* 1e-15 is impossible for the large fields.
The pass test is therefore the ``numpy.isclose`` form::

    |Δ| <= atol + rtol * |c|

with ``rtol`` the per-kind value above (→ ~1e-15 for O(1) fields, and correctly
scaled for large ones) and ``atol`` a small **calibratable** near-zero absolute
floor — the same way the Kokkos snapshot gate calibrated its per-field ceilings
once real runs existed. Both absolute and relative diagnostics are reported.

**Always** truncate the JAX column to ``c_record.nlevels`` first: the shim drops
the below-bottom padding, so a full-length compare spuriously fails on the tail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_dump import DumpRecord

# Per-kind relative tolerance (the plan's stated targets).
KIND_RTOL: dict[str, float] = {
    "map": 1e-15,
    "gather": 1e-15,
    "scatter": 1e-12,
    "reduction": 1e-12,
}
# Per-kind near-zero absolute floor (calibratable against real dumps).
KIND_ATOL: dict[str, float] = {
    "map": 1e-14,
    "gather": 1e-14,
    "scatter": 1e-11,
    "reduction": 1e-11,
}


@dataclass
class CompareResult:
    """Outcome of comparing one column; truthy iff it passed."""

    field: str
    kind: str
    nlevels: int
    rtol: float
    atol: float
    max_abs: float
    max_rel: float
    worst_level: int  # 0-based level of the largest absolute diff (-1 if empty)
    worst_c: float
    worst_jax: float
    passed: bool

    def __bool__(self) -> bool:
        return self.passed

    def report(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{self.kind:<9s}] {self.field:<12s} {status}  "
            f"max|Δ|={self.max_abs:.3e}  rel={self.max_rel:.3e}  "
            f"@nz={self.worst_level} c={self.worst_c:.6e} jax={self.worst_jax:.6e}  "
            f"tol={self.atol:.0e}+{self.rtol:.0e}·|c|  nlev={self.nlevels}"
        )


def compare_column(
    jax_vals,
    c_record: DumpRecord,
    kind: str = "map",
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> CompareResult:
    """Compare a JAX column against a reference record.

    ``jax_vals`` is truncated to ``c_record.nlevels`` (drop below-bottom padding);
    it is an error for it to be *shorter* than ``nlevels``.
    """
    if kind not in KIND_RTOL:
        raise ValueError(f"unknown kind {kind!r}; one of {list(KIND_RTOL)}")
    rtol = KIND_RTOL[kind] if rtol is None else rtol
    atol = KIND_ATOL[kind] if atol is None else atol

    c = np.asarray(c_record.values, dtype=np.float64).ravel()
    n = int(c_record.nlevels)
    j = np.asarray(jax_vals, dtype=np.float64).ravel()
    if j.shape[0] < n:
        raise ValueError(
            f"jax column has {j.shape[0]} levels < nlevels={n} for field "
            f"{c_record.field!r}"
        )
    j = j[:n]  # truncate below-bottom padding
    c = c[:n]

    diff = np.abs(j - c)
    tol = atol + rtol * np.abs(c)
    passed = bool(np.all(diff <= tol)) if diff.size else True

    if diff.size:
        worst_level = int(np.argmax(diff))
        max_abs = float(diff[worst_level])
        rel = diff / np.maximum(np.abs(c), np.finfo(np.float64).tiny)
        max_rel = float(rel.max())
        worst_c = float(c[worst_level])
        worst_jax = float(j[worst_level])
    else:
        worst_level, max_abs, max_rel, worst_c, worst_jax = -1, 0.0, 0.0, 0.0, 0.0

    return CompareResult(
        field=c_record.field,
        kind=kind,
        nlevels=n,
        rtol=rtol,
        atol=atol,
        max_abs=max_abs,
        max_rel=max_rel,
        worst_level=worst_level,
        worst_c=worst_c,
        worst_jax=worst_jax,
        passed=passed,
    )


def assert_close(
    jax_vals,
    c_record: DumpRecord,
    kind: str = "map",
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> CompareResult:
    """Like :func:`compare_column` but raise ``AssertionError`` (with the pretty
    report) when the column is out of tolerance. Returns the result on success."""
    result = compare_column(jax_vals, c_record, kind, rtol=rtol, atol=atol)
    if not result.passed:
        raise AssertionError(result.report())
    return result
