"""NN-GM revised Stage-1 feasibility — evaluate D(t) and the κ-branch signal S(t).

Consumes ``student_year.py`` output plus the Task-6 coarse teacher products and reports, in
the exact Stage-1 loss form (``horizon_scan.level_weights`` / ``stage1_misfit``, T+S terms;
the teacher products carry no SSH, so w_eta=0 here):

  * D(k)  — free-running default-GM student vs coarse-grained teacher at every shared 5-day
    mark of 1958: the standing distance the loss would sit on under the no-teacher-init
    design (its size and growth decide how much of the loss is un-fixable drift);
  * branch table — loss at +5 d and +10 d from each branch state at κ multiplier
    m ∈ {1/3, 1, 3}, and the spread across m: the ENTIRE signal a κ-bounded network can
    express at this window length;
  * floors — the m=1 branch re-runs the year's own segment, so its distance TO the year's
    own marks is the GPU-nondeterminism trajectory floor; teacher/student 5-day self-motion
    gives the scale fields move per mark.

Weights are the per-level inverse variance of the TEACHER over its available 1958 marks
(the plan's "teacher anomaly variance"), floored as in horizon_scan; they are saved for
reuse. Cells where the teacher has no data (``cell_area == 0`` — the measured 1.4% mesh-pair
mismatch) are EXCLUDED from every sum, per the plan rule that an aggregated 0 must never
reach a loss. CPU job; run under ``JAX_PLATFORMS=cpu``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

MARK_SECONDS = 5 * 86400.0


def mark_index(t_seconds: float) -> int:
    """Teacher records sit one fine-dt short of the round 5-day mark (431880/431760 s);
    round() absorbs the offset for any dt ≪ half a mark."""
    return int(round(float(t_seconds) / MARK_SECONDS))


def month_files(coarse_dir: Path, year: int):
    pat = re.compile(rf"^coarse_ng5_{year}_(\d\d)\.nc$")
    return sorted(p for p in coarse_dir.iterdir() if pat.match(p.name))


def model_to_teacher_levels(wet_model, nz_teacher: int) -> slice:
    """The JAX mesh arrays carry trailing below-bottom PADDING columns (CORE2: 48 columns,
    the deepest two globally dry — verified 2026-08-08); the teacher product stores true
    layers only (nz1=47). Trimming is exact ONLY if every extra column is globally dry —
    assert it rather than assume it."""
    import numpy as np
    wm = np.asarray(wet_model)
    assert wm.shape[1] >= nz_teacher, (wm.shape, nz_teacher)
    assert wm[:, nz_teacher:].sum() == 0, \
        "model columns beyond the teacher's carry wet cells — not padding; do NOT trim"
    return slice(0, nz_teacher)


def assemble_teacher(coarse_dir: Path, year: int):
    """-> (marks dict k -> (T[N,nz], S[N,nz]), wet mask [N,nz] bool, sorted mark list)."""
    import netCDF4
    marks, mask = {}, None
    n_intersected = 0
    for p in month_files(coarse_dir, year):
        ds = netCDF4.Dataset(p)
        tt = np.asarray(ds.variables["time"][:])
        for r, t in enumerate(tt):
            k = mark_index(t)
            T = np.asarray(ds.variables["temp"][r]).astype(np.float32)   # (N, nz1)
            S = np.asarray(ds.variables["salt"][r]).astype(np.float32)
            a = np.asarray(ds.variables["cell_area"][r])
            wet = a > 0.0
            if mask is None:
                mask = wet
            elif not np.array_equal(mask, wet):
                n_intersected += int((mask != wet).sum())
                mask &= wet
            marks[k] = (T, S)
        ds.close()
    if n_intersected:
        print(f"[teacher] WARNING: wet mask varied between records; intersected "
              f"({n_intersected} cell-instances)", flush=True)
    return marks, mask, sorted(marks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse-dir", type=Path, default=Path("/work/ab0995/a270088/nn_eddy/coarse"))
    ap.add_argument("--student-dir", type=Path,
                    default=Path("/work/ab0995/a270088/nn_eddy/student_free_1958"))
    ap.add_argument("--mesh-dir", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "mesh_core2")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    json_out = args.json_out or (args.student_dir / "signal_eval.json")

    import fesom_jax  # noqa: F401  (x64)
    import jax.numpy as jnp
    from fesom_jax.mesh import load_mesh
    import horizon_scan as hs

    print(f"[setup] teacher <- {args.coarse_dir}, student <- {args.student_dir}", flush=True)
    marks_t, wet_t, klist = assemble_teacher(args.coarse_dir, args.year)
    print(f"[teacher] {len(klist)} marks: days {[5*k for k in klist[:3]]}…{5*klist[-1]}", flush=True)

    z = np.load(args.student_dir / "marks_year.npz")
    T_s, S_s, day_s = z["T"], z["S"], z["day"]
    kmap_s = {int(d) // 5: i for i, d in enumerate(day_s)}

    mesh = load_mesh(args.mesh_dir)
    wet_model = np.asarray(mesh.node_layer_mask, bool)
    sl = model_to_teacher_levels(wet_model, wet_t.shape[1])
    wet_model = wet_model[:, sl]
    T_s, S_s = T_s[:, :, sl], S_s[:, :, sl]
    areasvol = np.asarray(mesh.areasvol)[:, sl]
    assert wet_model.shape == wet_t.shape == T_s.shape[1:], \
        (wet_model.shape, wet_t.shape, T_s.shape)
    wet = wet_model & wet_t
    excl = int((wet_model & ~wet_t).sum())
    print(f"[mask] student-wet cells with no teacher data (excluded): {excl} "
          f"({100*excl/max(wet_model.sum(),1):.2f}% of wet)", flush=True)
    wet3d = jnp.asarray(wet, jnp.float64)
    area3d = jnp.asarray(areasvol, jnp.float64) * wet3d

    # weights: teacher anomaly variance over its own year of marks (floored), w_eta unused
    Tt_stack = jnp.asarray(np.stack([marks_t[k][0] for k in klist]), jnp.float64)
    St_stack = jnp.asarray(np.stack([marks_t[k][1] for k in klist]), jnp.float64)
    zeros_eta = jnp.zeros((len(klist), wet.shape[0]), jnp.float64)
    w_T, w_S, _ = hs.level_weights(Tt_stack, St_stack, zeros_eta, wet3d, area3d)
    del Tt_stack, St_stack, zeros_eta
    np.savez(args.student_dir / "stage1_weights_teacher1958.npz",
             w_T=np.asarray(w_T), w_S=np.asarray(w_S))
    print(f"[weights] w_T surf={float(w_T[0]):.3e} bottom={float(w_T[-1]):.3e} (saved)", flush=True)

    e0 = jnp.zeros(wet.shape[0], jnp.float64)
    w_eta = jnp.asarray(0.0, jnp.float64)

    def misfit(Ta, Sa, Tb, Sb):
        return float(hs.stage1_misfit(jnp.asarray(Ta, jnp.float64), jnp.asarray(Sa, jnp.float64),
                                      e0, jnp.asarray(Tb, jnp.float64), jnp.asarray(Sb, jnp.float64),
                                      e0, w_T, w_S, w_eta, wet3d, area3d))

    # --- D(k): student vs teacher at every shared mark
    D = []
    for k in klist:
        if k in kmap_s:
            i = kmap_s[k]
            D.append(dict(day=5 * k, loss=misfit(T_s[i], S_s[i], *marks_t[k])))
    print("\n== D(t): free default-GM student vs coarse teacher ==", flush=True)
    for r in D:
        print(f"  day {r['day']:3d}  D = {r['loss']:.6e}")

    # --- self-motion scales (per 5-day mark)
    def motion(seq):
        return [dict(day=5 * kb, loss=misfit(*seq(ka), *seq(kb)))
                for ka, kb in zip(klist, klist[1:]) if kb - ka == 1]
    mot_t = motion(lambda k: marks_t[k])
    mot_s = motion(lambda k: (T_s[kmap_s[k]], S_s[kmap_s[k]])) \
        if all(k in kmap_s for k in klist) else []

    # --- branch table
    branches, floors, spreads = [], [], []
    for day0 in (60, 150, 240, 330):
        rows = []
        for f in sorted(args.student_dir.glob(f"branch_d{day0:03d}_m*.npz")):
            b = np.load(f)
            m = float(b["mult"])
            T5, S5 = b["T5"][:, sl], b["S5"][:, sl]
            T10, S10 = b["T10"][:, sl], b["S10"][:, sl]
            k5, k10 = day0 // 5 + 1, day0 // 5 + 2
            row = dict(day0=day0, mult=m)
            if k5 in marks_t:
                row["L5"] = misfit(T5, S5, *marks_t[k5])
            if k10 in marks_t:
                row["L10"] = misfit(T10, S10, *marks_t[k10])
            if abs(m - 1.0) < 1e-9 and k5 in kmap_s and k10 in kmap_s:
                floors.append(dict(day0=day0,
                                   L5_floor=misfit(T5, S5,
                                                   T_s[kmap_s[k5]], S_s[kmap_s[k5]]),
                                   L10_floor=misfit(T10, S10,
                                                    T_s[kmap_s[k10]], S_s[kmap_s[k10]])))
            rows.append(row)
        if rows:
            branches.extend(rows)
            for key, tag in (("L5", "S5"), ("L10", "S10")):
                vals = [r[key] for r in rows if key in r]
                if len(vals) >= 2:
                    spreads.append(dict(day0=day0, kind=tag, spread=max(vals) - min(vals),
                                        rel=(max(vals) - min(vals)) / max(vals)))

    print("\n== branch table (loss vs teacher; spread across m is the trainable signal) ==")
    for r in branches:
        print(f"  d{r['day0']:03d} m={r['mult']:.4f}  "
              f"L5={r.get('L5', float('nan')):.6e}  L10={r.get('L10', float('nan')):.6e}")
    for r in spreads:
        print(f"  d{r['day0']:03d} {r['kind']}: spread={r['spread']:.3e}  rel={100*r['rel']:.2f}%")
    for r in floors:
        print(f"  d{r['day0']:03d} m=1 vs year marks (nondet floor): "
              f"L5={r['L5_floor']:.3e}  L10={r['L10_floor']:.3e}")

    # --- split branches: one coefficient at a time (k_gm axis vs redi_kmax axis)
    split_rows, split_spreads = [], []
    for day0 in (60, 150, 240, 330):
        by_mult = {}
        for f in sorted(args.student_dir.glob(f"branch2_d{day0:03d}_g*_r*.npz")):
            b = np.load(f)
            mg, mr = float(b["mult_gm"]), float(b["mult_redi"])
            k5, k10 = day0 // 5 + 1, day0 // 5 + 2
            row = dict(day0=day0, mult_gm=mg, mult_redi=mr)
            if k5 in marks_t:
                row["L5"] = misfit(b["T5"][:, sl], b["S5"][:, sl], *marks_t[k5])
            if k10 in marks_t:
                row["L10"] = misfit(b["T10"][:, sl], b["S10"][:, sl], *marks_t[k10])
            by_mult[(round(mg, 4), round(mr, 4))] = row
            split_rows.append(row)
        for axis, up, dn in (("GM", (3.0, 1.0), (0.3333, 1.0)),
                             ("Redi", (1.0, 3.0), (1.0, 0.3333))):
            if up in by_mult and dn in by_mult:
                for key, tag in (("L5", "S5"), ("L10", "S10")):
                    if key in by_mult[up] and key in by_mult[dn]:
                        split_spreads.append(dict(
                            day0=day0, axis=axis, kind=tag,
                            spread=abs(by_mult[up][key] - by_mult[dn][key]),
                            up_minus_dn=by_mult[up][key] - by_mult[dn][key]))
    if split_rows:
        print("\n== split branches: one coefficient at a time ==")
        for r in split_rows:
            print(f"  d{r['day0']:03d} gm={r['mult_gm']:.4f} redi={r['mult_redi']:.4f}  "
                  f"L5={r.get('L5', float('nan')):.6e}  L10={r.get('L10', float('nan')):.6e}")
        for r in split_spreads:
            sign = "up better" if r["up_minus_dn"] < 0 else "down better"
            print(f"  d{r['day0']:03d} {r['axis']:4s} {r['kind']}: "
                  f"spread={r['spread']:.3e}  ({sign})")

    json_out.write_text(json.dumps(dict(
        D=D, teacher_motion=mot_t, student_motion=mot_s, branches=branches,
        spreads=spreads, floors=floors, split_branches=split_rows,
        split_spreads=split_spreads, excluded_cells=excl,
        n_teacher_marks=len(klist)), indent=1))
    print(f"\n[out] {json_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
