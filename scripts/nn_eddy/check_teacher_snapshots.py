#!/usr/bin/env python
"""Audit the NG5 teacher snapshot stream: is it the training data we asked for, and how big?

Read-only, header-and-time-only (no field reads), so it is cheap enough for the login node and
safe to run while the chain is going. It deliberately touches only months whose files the model
has already closed — a month still being written is held open by 8192 ranks, and HDF5 has no
business being opened twice.

What it checks, and why each one has burned someone:

* **The variable set.** ``&nml_snap`` is a separate code path from the mean streams; a field
  silently missing from it would only surface when the training pipeline asked for it.
* **Shapes against the mesh.** T, S nodal on ``nz1`` layers; w nodal on ``nl``; u, v ELEMENTAL.
  The elemental ones are the trap: collocation to nodes happens in our Python pipeline, so a
  file that arrived already node-shaped would mean the model rotated/interpolated behind us.
* **The time axis, read rather than assumed.** Records sit one time step SHORT of the round
  number (day 5 is 431880 s, not 432000), and the offset is one ``dt``, so it CHANGES from
  120 s to 240 s at the ramp switch. Assuming ``n x 5 days`` mislabels every window's start
  and target time by a step, silently.
* **Volume per set**, against the 14.7 GB/set and 1.07 TB/yr the cadence decision was made on.

Usage:  python scripts/nn_eddy/check_teacher_snapshots.py [--run DIR] [--month 1958_01]
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

import netCDF4
import numpy as np

RUN = pathlib.Path("/work/ba1583/a270088/nn_eddy/ng5_teacher")
MESH = pathlib.Path("/work/ab0995/a270088/nn_eddy/mesh_ng5")

# field -> (expected horizontal dimension, expected vertical dimension or None for 2D)
EXPECT = {
    "temp": ("nod2", "nz1"), "salt": ("nod2", "nz1"), "w": ("nod2", "nz"),
    "u": ("elem", "nz1"), "v": ("elem", "nz1"),
    "ssh": ("nod2", None), "sst": ("nod2", None), "sss": ("nod2", None),
    "a_ice": ("nod2", None), "m_ice": ("nod2", None), "m_snow": ("nod2", None),
    "uice": ("nod2", None), "vice": ("nod2", None),
}
GB = 1024.0 ** 3


def mesh_sizes():
    """nod2D / elem2D / nl straight from the JAX export, so the check is against the mesh the
    Python side will actually load, not against numbers copied from a namelist."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from fesom_jax.mesh import load_mesh
    m = load_mesh(str(MESH))
    return int(m.nod2D), int(m.elem2D), int(m.nl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=RUN)
    ap.add_argument("--month", default=None, help="e.g. 1958_01; default = every CLOSED month")
    ap.add_argument("--skip-mesh", action="store_true",
                    help="do not load the 16 GB NG5 export (skips the shape cross-check)")
    args = ap.parse_args()

    files = sorted(args.run.glob("*_snap.fesom.*.nc"))
    if not files:
        print(f"no snapshot files under {args.run}")
        return 1

    months = collections.defaultdict(dict)
    for f in files:
        m = re.match(r"(.+)_snap\.fesom\.(\d{4}_\d{2})\.nc$", f.name)
        if m:
            months[m.group(2)][m.group(1)] = f

    # A month is CLOSED iff a later month's files exist; the newest one is still being written.
    closed = sorted(months)[:-1] if len(months) > 1 else []
    todo = [args.month] if args.month else closed
    if not todo:
        print(f"months present: {sorted(months)}; none closed yet (the newest is still open)")
        return 1
    print(f"months present {sorted(months)}; closed {closed}; auditing {todo}")

    nod2D = elem2D = nl = None
    if not args.skip_mesh:
        nod2D, elem2D, nl = mesh_sizes()
        print(f"mesh_ng5: nod2D={nod2D} elem2D={elem2D} nl={nl}")

    bad = []
    for mon in todo:
        got = months[mon]
        missing = sorted(set(EXPECT) - set(got))
        extra = sorted(set(got) - set(EXPECT))
        if missing:
            bad.append(f"{mon}: snap stream is MISSING {missing}")
        if extra:
            print(f"  note {mon}: unexpected snap fields {extra}")

        print(f"\n=== {mon} ===")
        print(f"{'field':8s} {'shape':>26s} {'GB':>8s} {'GB/rec':>8s}  records")
        total = nrec_ref = None
        total = 0.0
        times_ref = None
        for name in sorted(got):
            path = got[name]
            size = path.stat().st_size / GB
            total += size
            with netCDF4.Dataset(path) as ds:
                # ⚠️ The variable inside a snapshot file is named "<field>_snap", not "<field>":
                # &nml_snap defines a stream whose id carries the suffix, and io_meandata names
                # the variable after the stream. A pipeline that opens temp_snap.fesom.*.nc and
                # asks for "temp" gets a KeyError, and one that globs the single non-coordinate
                # variable gets it right by accident. Assert the name instead of guessing.
                vname = f"{name}_snap"
                if vname not in ds.variables:
                    bad.append(f"{mon}/{name}: no variable {vname!r} in {path.name}; "
                               f"file has {sorted(ds.variables)}")
                    continue
                var = ds.variables[vname]
                dims = var.dimensions
                shape = var.shape
                if var.dtype != np.float32:
                    bad.append(f"{mon}/{name}: dtype {var.dtype}, expected float32 "
                               f"(precision 4 in snap_list)")
                t = np.asarray(ds.variables["time"][:], dtype=float)
            nrec = shape[0]
            print(f"{name:8s} {str(dims) + str(shape):>26s} {size:8.3f} {size / max(nrec,1):8.3f}"
                  f"  {nrec}")

            exp_h, exp_v = EXPECT.get(name, (None, None))
            if exp_h and exp_h not in dims:
                bad.append(f"{mon}/{name}: horizontal dim {dims} is not {exp_h!r} — "
                           f"{'u,v must stay ELEMENTAL' if exp_h == 'elem' else 'expected nodal'}")
            if exp_v and exp_v not in dims:
                bad.append(f"{mon}/{name}: vertical dim {dims} is not {exp_v!r}")
            if exp_v is None and len(shape) != 2:
                bad.append(f"{mon}/{name}: expected a 2D field (time, space), got {shape}")
            if nod2D is not None:
                want = nod2D if exp_h == "nod2" else elem2D
                if want not in shape:
                    bad.append(f"{mon}/{name}: shape {shape} contains neither the mesh's "
                               f"{exp_h} size ({want}) — file and mesh disagree")
            if nrec_ref is None:
                nrec_ref, times_ref = nrec, t
            elif nrec != nrec_ref:
                bad.append(f"{mon}/{name}: {nrec} records but the first field has {nrec_ref} — "
                           f"a window needs the FULL state at one time")
            elif not np.array_equal(t, times_ref):
                bad.append(f"{mon}/{name}: time axis differs from the other fields")

        print(f"{'TOTAL':8s} {'':>26s} {total:8.3f} {total / max(nrec_ref or 1, 1):8.3f}"
              f"  {nrec_ref} sets")

        # --- the time axis, read not assumed --------------------------------------------
        print(f"time (s): {times_ref.tolist()}")
        d = np.diff(times_ref)
        print(f"spacing : {sorted(set(d.tolist()))} s"
              f"  ({', '.join(f'{x/86400:g} d' for x in sorted(set(d.tolist())))})")
        off = times_ref % 86400.0
        nominal = np.round(times_ref / 432000.0) * 432000.0
        short = nominal - times_ref
        print(f"short of the round 5-day mark by: {sorted(set(short.tolist()))} s "
              f"(one dt: 120 s in the ramp, 240 s from month 3 — do NOT assume n x 5 days)")
        if len(set(d.tolist())) > 1:
            bad.append(f"{mon}: snapshot spacing is not uniform: {sorted(set(d.tolist()))}")
        if not np.all(short >= 0):
            bad.append(f"{mon}: some records are LATER than the round mark ({short})")

        # --- volume, against the numbers the cadence decision used ---------------------
        per_set = total / max(nrec_ref or 1, 1)
        print(f"\nper snapshot set : {per_set:.2f} GB  (namelist.io header estimate 14.7 GB)")
        print(f"73 sets/yr       : {per_set * 73 / 1024:.2f} TB/yr "
              f"(header estimate 1.07 TB/yr)")

    print()
    if bad:
        print("PROBLEMS:")
        for b in bad:
            print("  -", b)
        return 1
    print("snapshot stream OK: variable set, shapes, record counts and time axis all consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
