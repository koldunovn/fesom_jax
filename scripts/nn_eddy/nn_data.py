"""NN-GM shared teacher-data layer (child plan M2) — ONE implementation for training and eval.

Everything that reads a coarse teacher product and lines it up against a student state lives
here: mark indexing, the level trim, the hole mask, weight loading, the eta join and its
de-meaning. ``signal_eval.py`` imports these rather than defining them, and the training driver
will import the same functions — because the alternative is two implementations of "which cells
count", and the one in the training loop would be the one nobody checked.

The four things this module exists to get right
----------------------------------------------
**Marks.** Teacher records sit one fine-dt SHORT of the round 5-day mark, and the offset CHANGES
with the teacher's dt ramp (120 → 240 s at month 03 of 1958). :func:`mark_index` rounds, which
absorbs both. Never compute a mark time from a record number.

**Levels.** The JAX mesh arrays carry 48 columns, the deepest two globally dry; the products
carry 47 true layers. :func:`model_to_teacher_levels` asserts every extra column is globally dry
before it trims. Slicing blind here cost this project a failed job on 2026-08-08.

**Holes.** ``cell_area == 0`` means NO DATA, not zero — 1.4% of student-wet cells (51,178 of
them, measured) have no teacher behind them, because the two meshes' coastlines disagree. A zero
that reaches a loss is a fabricated observation of freezing fresh water, and it would be fitted.
:func:`combined_mask` intersects the student's wet mask with the teacher's and reports what it
dropped.

**The eta mean.** SSH is only defined up to a constant, so the loss compares ANOMALIES: both
sides de-meaned with the same area weights (:func:`demean`). Measured 2026-08-08: the offset is
in fact ~1e-12 m on the teacher side and below 1e-5 m on the student side — FESOM conserves
volume and the area-weighted aggregation preserves that — so this is insurance, not a large
correction. It stays because a configuration where it is NOT zero (a different mesh pair, a
freshwater-flux experiment) must not silently become a constant that drowns its own gradient.
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path

import numpy as np

MARK_SECONDS = 5 * 86400.0

# ---------------------------------------------------------------------------
# Where this project's data lives — ONE definition, imported by every script
# ---------------------------------------------------------------------------
# **ba1583, not ab0995** (user decision 2026-08-08, and the parent plan's Revision Log (j)
# reached the same place for the M4 trajectories). Measured that day: the ab0995 /work mount was
# at **98% — 2.5 P used of 2.6 P, 53 T free** — while ba1583 sat at 10% with 117 T free and
# already held the 2.9 TB teacher run. The growth is not speculative either: coarse products run
# ~13 GB per teacher year (ten years ≈ 130 GB) and M4's stored trajectories ~130 GB per year per
# refresh (≈ 1.3 TB across the planned ten). Putting the products next to the teacher output that
# generates them also means one directory to archive when the paper is written.
#
# Override with $NN_EDDY_DATA for a scratch experiment; do NOT hardcode a path in a script.
DATA_ROOT = Path(os.environ.get("NN_EDDY_DATA", "/work/ba1583/a270088/nn_eddy"))
COARSE_DIR = DATA_ROOT / "coarse"                 # coarse_ng5_*.nc + coarse_eta_ng5_*.nc
TEACHER_DIR = DATA_ROOT / "ng5_teacher"           # the raw NG5 snapshot stream
MESH_NG5 = DATA_ROOT / "mesh_ng5"                 # the NG5 mesh .npy export
STUDENT_DIR = DATA_ROOT / "student_free_1958"     # student_year.py output + signal_eval.json


def data_path(*parts) -> Path:
    """A path under :data:`DATA_ROOT`. Use this for any NEW output directory so a future move is
    one constant again, rather than another round of grep-and-edit."""
    return DATA_ROOT.joinpath(*parts)


def mark_index(t_seconds: float) -> int:
    """Teacher records sit one fine-dt short of the round 5-day mark (431880/431760 s);
    round() absorbs the offset for any dt ≪ half a mark, INCLUDING across the teacher's dt
    ramp, which is why this rounds instead of subtracting a known offset."""
    return int(round(float(t_seconds) / MARK_SECONDS))


def month_files(coarse_dir: Path, year: int):
    """The 3-D products for ``year``, sorted by month."""
    pat = re.compile(rf"^coarse_ng5_{year}_(\d\d)\.nc$")
    return sorted(p for p in Path(coarse_dir).iterdir() if pat.match(p.name))


def eta_month_files(coarse_dir: Path, year: int):
    """The eta SIDE-products for ``year``, sorted by month (M2; may be absent)."""
    pat = re.compile(rf"^coarse_eta_ng5_{year}_(\d\d)\.nc$")
    return sorted(p for p in Path(coarse_dir).iterdir() if pat.match(p.name))


def model_to_teacher_levels(wet_model, nz_teacher: int) -> slice:
    """The JAX mesh arrays carry trailing below-bottom PADDING columns (CORE2: 48 columns,
    the deepest two globally dry — verified 2026-08-08); the teacher product stores true
    layers only (nz1=47). Trimming is exact ONLY if every extra column is globally dry —
    assert it rather than assume it."""
    wm = np.asarray(wet_model)
    assert wm.shape[1] >= nz_teacher, (wm.shape, nz_teacher)
    assert wm[:, nz_teacher:].sum() == 0, \
        "model columns beyond the teacher's carry wet cells — not padding; do NOT trim"
    return slice(0, nz_teacher)


@dataclasses.dataclass
class Teacher:
    """Coarse teacher marks for one year, keyed by 5-day mark index.

    ``marks[k] = (T, S)`` on the product's own 47 levels; ``eta[k]`` is the surface field when
    the side-product is present, else the dict is empty. ``wet`` is the product's hole mask
    (True = the cell has fine data behind it), intersected across records — a cell that is a
    hole in ANY record is a hole in all of them, because the training driver masks once."""
    marks: dict
    wet: np.ndarray
    klist: list
    eta: dict = dataclasses.field(default_factory=dict)
    eta_wet: np.ndarray | None = None
    n_intersected: int = 0

    @property
    def has_eta(self) -> bool:
        return bool(self.eta)


def assemble_teacher(coarse_dir: Path, year: int, *, with_eta: bool = False,
                     verbose: bool = True) -> Teacher:
    """Read a year of coarse products into a :class:`Teacher`.

    ``with_eta`` additionally joins the eta side-products ON THE MARK INDEX (never on record
    order — the two file sets are written by different scripts and a missing month would shift
    one of them). A mark present in the 3-D products but missing from the eta ones is kept
    without eta, so a partially-coarsened eta stream degrades to a T/S-only loss rather than
    silently dropping marks."""
    import netCDF4
    marks, mask, n_intersected = {}, None, 0
    for p in month_files(coarse_dir, year):
        with netCDF4.Dataset(p) as ds:
            tt = np.asarray(ds.variables["time"][:])
            for r, t in enumerate(tt):
                k = mark_index(t)
                T = np.asarray(ds.variables["temp"][r]).astype(np.float32)   # (N, nz1)
                S = np.asarray(ds.variables["salt"][r]).astype(np.float32)
                wet = np.asarray(ds.variables["cell_area"][r]) > 0.0
                if mask is None:
                    mask = wet
                elif not np.array_equal(mask, wet):
                    n_intersected += int((mask != wet).sum())
                    mask &= wet
                marks[k] = (T, S)
    if n_intersected and verbose:
        print(f"[teacher] WARNING: wet mask varied between records; intersected "
              f"({n_intersected} cell-instances)", flush=True)

    eta, eta_wet = {}, None
    if with_eta:
        for p in eta_month_files(coarse_dir, year):
            with netCDF4.Dataset(p) as ds:
                tt = np.asarray(ds.variables["time"][:])
                for r, t in enumerate(tt):
                    k = mark_index(t)
                    eta[k] = np.asarray(ds.variables["eta"][r]).astype(np.float32)
                    w = np.asarray(ds.variables["cell_area"][r]) > 0.0
                    eta_wet = w if eta_wet is None else (eta_wet & w)
        if verbose:
            missing = [k for k in marks if k not in eta]
            print(f"[teacher] eta joined on {len(eta)} marks"
                  + (f"; {len(missing)} T/S marks have no eta yet" if missing else ""),
                  flush=True)
    return Teacher(marks=marks, wet=mask, klist=sorted(marks), eta=eta, eta_wet=eta_wet,
                   n_intersected=n_intersected)


def combined_mask(wet_model, wet_teacher, *, verbose: bool = True):
    """``(mask, n_excluded)`` — cells that are wet in the model AND carry teacher data.

    The excluded count is the plan's 51,178 (1.4% of student-wet cells): the mesh-pair coastline
    mismatch. It is printed because it is the number that says whether a product regenerated
    later is the same product — a change here means the mask changed, and every recorded loss
    number moves with it."""
    wm = np.asarray(wet_model, bool)
    wt = np.asarray(wet_teacher, bool)
    assert wm.shape == wt.shape, (wm.shape, wt.shape)
    mask = wm & wt
    excl = int((wm & ~wt).sum())
    if verbose:
        print(f"[mask] student-wet cells with no teacher data (excluded): {excl} "
              f"({100 * excl / max(wm.sum(), 1):.2f}% of wet)", flush=True)
    return mask, excl


def student_grid(mesh, nz_teacher: int):
    """``(wet_model, areasvol, sl)`` for a loaded mesh, trimmed to the teacher's levels.

    One function so the trim and the area array can never come from different slices — the
    failure mode being an area weight applied to the wrong level."""
    wet_model = np.asarray(mesh.node_layer_mask, bool)
    sl = model_to_teacher_levels(wet_model, nz_teacher)
    return wet_model[:, sl], np.asarray(mesh.areasvol)[:, sl], sl


def demean(field, weight):
    """Subtract the area-weighted mean of ``field`` over cells with ``weight > 0``.

    For the SSH term of the loss, applied to BOTH sides with the SAME weights. Cells with no
    weight are left untouched (they are masked out downstream anyway); subtracting a mean from
    a hole would turn a no-data cell into a plausible-looking anomaly."""
    f = np.asarray(field, np.float64)
    w = np.asarray(weight, np.float64)
    sel = w > 0.0
    if not sel.any():
        return f
    m = float((w[sel] * f[sel]).sum() / w[sel].sum())
    return np.where(sel, f - m, f)


def load_weights(path: Path):
    """``(w_T, w_S)`` from a ``stage1_weights_*.npz`` — the per-level inverse-variance weights.

    The recorded precedent is the teacher's own 1958 anomaly variance, floored
    (``horizon_scan.level_weights``), at
    ``/work/ab0995/a270088/nn_eddy/student_free_1958/stage1_weights_teacher1958.npz``. It carries
    47 levels — the TEACHER's count — so a model-shaped array must be trimmed with
    :func:`model_to_teacher_levels` before these are applied."""
    z = np.load(path)
    return np.asarray(z["w_T"], np.float64), np.asarray(z["w_S"], np.float64)
