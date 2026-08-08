#!/usr/bin/env python
"""Coarse-grain one month of NG5 teacher SSH snapshots onto CORE2 — the eta side-product (M2).

A **separate file** ``coarse_eta_ng5_YYYY_MM.nc``, never an edit of the verified 3-D products.
Two reasons, both hard: ``coarsen_stream.write_output`` opens with mode ``"w"``, which truncates,
so "add eta to the product" would mean regenerating a 3-D month (~35 min each) and would put
twelve files whose numbers are already recorded in a passed gate at risk of a partial write.
eta is 2-D and takes seconds; there is no reason for the two to share a lifetime.

eta needs no new Fortran output: the teacher already writes ``ssh_snap.fesom.YYYY_MM.nc`` on the
same 5-day snapshot stream as ``temp_snap``, with the same one-fine-dt offset from the round mark
(``nn_data``/``signal_eval.mark_index`` absorbs it by rounding — do not re-derive it).

**The filter is the same Voronoi aggregation** as the 3-D path, with the SURFACE layer mask, so
the hole rule is identical by construction: ``cell_area == 0`` marks a coarse cell with no fine
data under it, and must never be read as ``eta = 0``. That identity is checked against the 3-D
product's own surface ``cell_area`` when one is present (``--check-against``), which is what
makes "identical hole rule" a measured claim rather than an intention.

**The mean is not a target.** NG5 and CORE2 have different bathymetry and volume, so their global
mean η differs by a constant that no bounded multiplier on an eddy diffusivity can affect; an
un-de-meaned η term in the loss is a constant that drowns its own gradient. The de-meaning is the
loss's job (both sides, area-weighted), not the product's — so this file stores the RAW field and
records ``eta_mean`` per record, which is the offset in question, as a number rather than a
worry.

Usage:
    python scripts/nn_eddy/coarsen_eta.py --month 1959_01
    python scripts/nn_eddy/coarsen_eta.py --month 1958_01 --check-against \\
        /work/ab0995/a270088/nn_eddy/coarse/coarse_ng5_1958_01.nc
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import netCDF4
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp                                              # noqa: E402

from fesom_jax import coarsen                                        # noqa: E402

# Same guard as coarsen_stream: `python scripts/foo.py` never puts the cwd on sys.path, and the
# editable install resolves fesom_jax to the MAIN checkout — which once made a worktree-only
# change inert and invalidated an A/B (2026-08-07).
_got = pathlib.Path(coarsen.__file__).resolve()
if _ROOT not in _got.parents:
    raise SystemExit(f"FATAL: fesom_jax imported from {_got}, not from {_ROOT}")

from coarsen_stream import (COARSE_MESH, FILL_ABOVE, FINE_MESH,     # noqa: E402
                           OUT, TEACHER, FineMesh, build_or_load_map)

ETA_ABSMAX = 10.0          # a coarse-grained SSH outside +-10 m is a bug, not an ocean


def read_eta_slice(ds, rec):
    """One SSH record as float64 with FESOM's fill turned into an explicit zero.

    ``read_slice`` indexes ``[rec, k, :]``; SSH is ``[rec, :]``, so this is its 2-D twin. The
    fill handling is not optional: ``np.asarray`` on a netCDF4 MaskedArray hands 9.96921e+36 back
    as an ordinary number, and a single such value inside an area-weighted mean is visible across
    a whole coarse cell."""
    raw = np.ma.asarray(ds.variables["ssh_snap"][rec, :])
    a = np.ma.filled(np.ma.masked_greater(np.ma.masked_invalid(raw), FILL_ABOVE), 0.0)
    return np.asarray(a, dtype=np.float64)


def process_month(month, rmap, fine, teacher: pathlib.Path, records=None):
    """Aggregate every SSH record of ``month`` onto the coarse mesh.

    Returns ``(times, eta, weight)`` with ``eta``/``weight`` shaped ``(nrec, n_coarse)``. No
    vertical remap exists for a 2-D field, so unlike the 3-D path the weight here IS the raw
    summed wet fine-node area of each coarse cell's surface."""
    path = teacher / f"ssh_snap.fesom.{month}.nc"
    if not path.exists():
        raise SystemExit(f"missing snapshot file {path}")
    surf = jnp.asarray(fine.layer_mask(0))          # the SURFACE mask — the 3-D path's level 0
    times, etas, weights = [], [], []
    with netCDF4.Dataset(path) as ds:
        n_rec = ds.variables["ssh_snap"].shape[0]
        recs = records if records is not None else list(range(n_rec))
        print(f"[run] month {month}: {n_rec} records in file, processing {recs}", flush=True)
        for r in recs:
            t0 = time.time()
            t_val = float(ds.variables["time"][r])      # read it, never compute it
            eta_f = jnp.asarray(read_eta_slice(ds, r))
            ec, w = coarsen.aggregate_horizontal_with_weight(eta_f, rmap, fine_mask=surf)
            ec, w = np.asarray(ec), np.asarray(w)
            times.append(t_val)
            etas.append(ec)
            weights.append(w)
            wet = w > 0.0
            am = float((w[wet] * ec[wet]).sum() / w[wet].sum())
            print(f"[rec {r}] {time.time() - t0:5.1f} s  t={t_val:.0f} s "
                  f"({t_val / 86400.0:.3f} d)  wet {wet.sum()}/{wet.size} "
                  f"({100 * wet.mean():.1f}%)  eta [{ec[wet].min():+.3f}, {ec[wet].max():+.3f}] m"
                  f"  area-mean {am:+.4f} m", flush=True)
    return times, np.stack(etas), np.stack(weights)


def check(eta, weight, against: pathlib.Path | None):
    """Physical bounds, the hole rule, and — when a 3-D product is given — that the hole PATTERN
    is identical to its surface ``cell_area``.

    The pattern check is the one that matters: ``nn_data`` will join this file to the 3-D product
    on the mark index and mask both with one hole mask, so a cell that is a hole in one and not
    the other silently contributes a difference against zero. Returns a nonzero count of
    problems, so the caller can exit on it."""
    bad = 0
    wet = weight > 0.0
    if not np.isfinite(eta).all():
        print(f"CHECK FAIL: {int((~np.isfinite(eta)).sum())} non-finite eta values")
        bad += 1
    absmax = float(np.abs(eta[wet]).max()) if wet.any() else 0.0
    if absmax > ETA_ABSMAX:
        print(f"CHECK FAIL: |eta| max {absmax:.3f} m exceeds {ETA_ABSMAX} m on wet cells")
        bad += 1
    else:
        print(f"[check] |eta| max on wet cells {absmax:.3f} m (bar {ETA_ABSMAX})")
    n_hole = int((~wet[0]).sum())
    print(f"[check] holes (cell_area == 0): {n_hole}/{wet.shape[1]} coarse cells "
          f"({100 * n_hole / wet.shape[1]:.2f}%)")
    if not (wet == wet[0]).all():
        print("CHECK FAIL: the hole pattern changes between records — the surface mask is static, "
              "so it must not")
        bad += 1

    if against is not None:
        if not against.exists():
            print(f"CHECK FAIL: --check-against {against} does not exist")
            return bad + 1
        with netCDF4.Dataset(against) as ds:
            ca = np.asarray(ds.variables["cell_area"][0, :, 0], dtype=np.float64)
            t3 = np.asarray(ds.variables["time"][:], dtype=np.float64)
        wet3 = ca > 0.0
        n_diff = int((wet3 != wet[0]).sum())
        if n_diff:
            print(f"CHECK FAIL: hole pattern differs from {against.name}'s surface cell_area on "
                  f"{n_diff} cells (eta-only {int((wet[0] & ~wet3).sum())}, "
                  f"3D-only {int((wet3 & ~wet[0]).sum())})")
            bad += 1
        else:
            print(f"[check] hole pattern identical to {against.name} surface cell_area "
                  f"({int(wet3.sum())} wet)")
        # the weights themselves differ by the 3-D path's vertical remap of the top layer; that
        # is expected and small, so it is REPORTED, not asserted
        both = wet[0] & wet3
        if both.any():
            rel = np.abs(weight[0][both] - ca[both]) / np.maximum(ca[both], 1e-300)
            print(f"[check] surface weight vs remapped 3-D cell_area: median rel "
                  f"{float(np.median(rel)):.2e}, max {float(rel.max()):.2e} "
                  f"(the 3-D one is vertically remapped; a small difference is expected)")
        print(f"[check] 3-D product times {t3[:3]}... vs eta times — compare after the join")
    return bad


def write_output(path: pathlib.Path, month, times, eta, weight):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"REFUSING to overwrite {path} — move it away first")
    tmp = path.with_suffix(".tmp.nc")
    wet = weight > 0.0
    mean = np.array([float((weight[i][wet[i]] * eta[i][wet[i]]).sum() / weight[i][wet[i]].sum())
                     for i in range(eta.shape[0])])
    with netCDF4.Dataset(tmp, "w") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("nod2", eta.shape[1])
        ds.description = (
            "NG5 teacher SSH snapshots coarse-grained onto the CORE2 mesh (Voronoi cell average "
            "over the SURFACE wet mask — the same filter and hole rule as the 3-D products). "
            "cell_area == 0 marks NO DATA, not zero. The field is RAW: a loss using it must "
            "de-mean both sides area-weighted, because NG5 and CORE2 have different volume and "
            "bathymetry and so different global mean eta (see eta_mean).")
        ds.source_month = month
        tv = ds.createVariable("time", "f8", ("time",))
        tv[:] = np.asarray(times)
        tv.long_name = "seconds since the run start, read from the snapshot file"
        v = ds.createVariable("eta", "f4", ("time", "nod2"), zlib=False)
        v[:] = eta.astype(np.float32)
        v.units = "m"
        v.long_name = "coarse-grained sea surface elevation (raw, not de-meaned)"
        v = ds.createVariable("cell_area", "f4", ("time", "nod2"), zlib=False)
        v[:] = weight.astype(np.float32)
        v.long_name = "summed wet fine-node surface area behind each coarse value; 0 = no data"
        v = ds.createVariable("eta_mean", "f8", ("time",))
        v[:] = mean
        v.long_name = ("area-weighted global mean of this field; the offset a loss must remove "
                       "from both sides")
    tmp.rename(path)                          # write-then-rename: never a torn product
    print(f"[out] wrote {path}  (eta_mean {mean.min():+.4f} .. {mean.max():+.4f} m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="e.g. 1959_01")
    ap.add_argument("--records", default=None, help="comma-separated record indices; default all")
    ap.add_argument("--teacher", type=pathlib.Path, default=TEACHER)
    ap.add_argument("--fine-mesh", type=pathlib.Path, default=FINE_MESH)
    ap.add_argument("--coarse-mesh", type=pathlib.Path, default=COARSE_MESH)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--map-cache", type=pathlib.Path, default=None)
    ap.add_argument("--check-against", type=pathlib.Path, default=None,
                    help="a 3-D coarse product for this month; asserts an identical hole pattern")
    ap.add_argument("--check-only", action="store_true", help="run the checks, write nothing")
    args = ap.parse_args()

    cache = args.map_cache or (args.out / "remap_ng5_to_core2.npz")
    rmap = build_or_load_map(cache, args.fine_mesh, args.coarse_mesh)
    t0 = time.time()
    fine = FineMesh(args.fine_mesh)
    print(f"[mesh] fine nod2D={fine.nod2D} nl={fine.nl} ({time.time() - t0:.1f} s)", flush=True)

    recs = [int(x) for x in args.records.split(",")] if args.records else None
    times, eta, weight = process_month(args.month, rmap, fine, args.teacher, records=recs)

    against = args.check_against
    if against is None:
        cand = args.out / f"coarse_ng5_{args.month}.nc"
        against = cand if cand.exists() else None
        if against is not None:
            print(f"[check] found the 3-D product for this month, checking against {against.name}")
    bad = check(eta, weight, against)
    if bad:
        print("PRODUCT CHECK FAILED — do not train on this file")
        return 1
    if args.check_only:
        print("[check-only] checks passed; nothing written")
        return 0
    write_output(args.out / f"coarse_eta_ng5_{args.month}.nc", args.month, times, eta, weight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
