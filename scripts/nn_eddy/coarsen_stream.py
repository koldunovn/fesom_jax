#!/usr/bin/env python
"""Coarse-grain one month of NG5 teacher snapshots onto the CORE2 mesh (NN-GM Task 6).

Emits, per snapshot record:

* the coarse-grained STATE — T, S, u, v on CORE2 nodes and levels, which is what a training
  window is initialised from and scored against;
* the diagnosed EDDY FLUXES — ``agg(u·τ) − agg(u)·agg(τ)`` for τ ∈ {T, S}, which is the
  offline target of Task 8 and the thing the parameterization is supposed to reproduce.

Why the aggregation IS the filter
---------------------------------
The plan writes the eddy flux as ``filter(u·τ) − filter(u)·filter(τ)``. Here "filter" is the
Voronoi aggregation itself: a top-hat average over each coarse node's cell, i.e. exactly the
scale separation the student cannot resolve. That makes the decomposition a Reynolds average
over a well-defined control volume, so the two terms are consistent by construction and the
residual is the genuine subgrid covariance. Applying a Laplacian smoother *as well* would be a
second, different filter and the cross terms would no longer cancel — do the extra smoothing
afterwards on the coarse mesh if it is wanted, not in the middle of this identity.

Order of operations, and why
----------------------------
1. **Element→node** for u, v on the FINE mesh. The products ``u·T`` must be formed where both
   live, and T is nodal; collocating u first is the only place the two can meet. ⚠️ This step
   is an interpolation, not a conservative remap, so the velocity path does not conserve its
   area integral (``fesom_jax/tests/test_coarsen_remap.py`` measures it).
2. **Products, then horizontal aggregation, level by level.** Both orders of aggregation and
   vertical remap agree (tested), so the cheap one is free: aggregating first reduces 7.4 M
   columns to 126 k before anything touches the vertical.
3. **Vertical remap** 70 → 48 layers, thickness-weighted, on the coarse columns.

Memory
------
Strictly level-at-a-time. One NG5 level is 7.4 M nodes (59 MB f8) or 14.7 M elements (118 MB);
a whole 3-D field would be 4.1 GB and there are eight of them once the products exist. The
mesh arrays are read from the ``.npy`` export with ``mmap_mode='r'`` and only the columns
actually needed are materialised — ``area.npy`` alone is 4.1 GB on NG5.

Usage
-----
    python scripts/nn_eddy/coarsen_stream.py --month 1958_01 [--records 0,1] [--out DIR]

The remap map is cached (``--map-cache``): the KD-tree query over 7.4 M points is the one
genuinely slow setup step and it is identical for every month.
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

import jax                                                          # noqa: E402
import jax.numpy as jnp                                            # noqa: E402

from fesom_jax import coarsen                                       # noqa: E402
from fesom_jax.coarsen import RemapMap                              # noqa: E402
import nn_data as _nd                                             # noqa: E402

# ⚠️ `python scripts/foo.py` puts the SCRIPT's directory on sys.path, never the cwd, and the
# editable install resolves fesom_jax to the MAIN checkout — which silently invalidated an
# experiment on 2026-08-07 (an env switch that existed only in the worktree was inert, so an
# A/B compared a configuration with itself). Assert what we actually imported.
_got = pathlib.Path(coarsen.__file__).resolve()
if _ROOT not in _got.parents:
    raise SystemExit(f"FATAL: fesom_jax imported from {_got}, not from {_ROOT}")

TEACHER = pathlib.Path("/work/ba1583/a270088/nn_eddy/ng5_teacher")
FINE_MESH = _nd.MESH_NG5
COARSE_MESH = pathlib.Path("/work/ab0995/a270088/port_jax/data/mesh_core2")
OUT = _nd.COARSE_DIR


# ---------------------------------------------------------------------------
# fine-mesh pieces, read lazily from the export
# ---------------------------------------------------------------------------

class FineMesh:
    """The four arrays this pipeline needs from a mesh export, and nothing else.

    Deliberately not ``load_mesh``: that materialises the whole 16 GB NG5 bundle, of which
    this uses about 0.5 GB. ``elem_to_node`` only reads ``elem_nodes``, ``elem_area`` and
    ``nod2D``, so a duck-typed holder is enough and keeps the tested function in play rather
    than re-implementing the collocation here.
    """

    def __init__(self, mesh_dir):
        d = pathlib.Path(mesh_dir)
        self.elem_nodes = jnp.asarray(np.load(d / "elem_nodes.npy"))
        self.elem_area = jnp.asarray(np.load(d / "elem_area.npy"))
        self.nod2D = int(np.load(d / "nlevels_nod2D.npy", mmap_mode="r").shape[0])
        self.elem2D = int(self.elem_area.shape[0])
        # layer validity per node: valid k satisfies ulevels-1 <= k < nlevels-1 (mesh.py's
        # convention — nlevels is an INTERFACE count, so the layer bound is exclusive)
        self._ulev = np.load(d / "ulevels_nod2D.npy")
        self._nlev = np.load(d / "nlevels_nod2D.npy")
        # ELEMENTAL level counts, for u/v. Same interface-count convention as the nodal ones.
        self._eulev = np.load(d / "ulevels.npy")
        self._enlev = np.load(d / "nlevels.npy")
        self.nl = int(np.load(d / "zbar.npy").shape[0])
        self.zbar = np.load(d / "zbar.npy")

    def layer_mask(self, k: int) -> np.ndarray:
        """1.0 where layer ``k`` is a real wet layer of this node, else 0.0."""
        return ((self._ulev - 1 <= k) & (k < self._nlev - 1)).astype(np.float64)

    def elem_layer_mask(self, k: int) -> np.ndarray:
        """1.0 where layer ``k`` is a real wet layer of this ELEMENT, else 0.0.

        ⚠️ Needed because ``elem_to_node`` averages every adjacent element, so a wet node next
        to a dry element inherits a third of what that element holds — and FESOM writes
        ``_FillValue = 9.96921e+36`` below the bottom. See the note on ``elem_to_node``.
        """
        return ((self._eulev - 1 <= k) & (k < self._enlev - 1)).astype(np.float64)


def build_or_load_map(cache: pathlib.Path, fine_dir, coarse_dir) -> RemapMap:
    """The KD-tree query over 7.4 M points is the only slow setup step, and it is the same for
    every month — so it is cached. Cached by CONTENT of the two mesh dirs' node counts, and
    the loader re-derives everything else, so a stale cache from a different mesh cannot be
    silently reused."""
    if cache.exists():
        z = np.load(cache)
        rmap = RemapMap(fine_cell=jnp.asarray(z["fine_cell"]),
                        fine_area=jnp.asarray(z["fine_area"]),
                        n_coarse=int(z["n_coarse"]),
                        fine_depth=jnp.asarray(z["fine_depth"]),
                        coarse_depth=jnp.asarray(z["coarse_depth"]))
        n_fine = int(np.load(pathlib.Path(fine_dir) / "nlevels_nod2D.npy",
                             mmap_mode="r").shape[0])
        n_coarse = int(np.load(pathlib.Path(coarse_dir) / "nlevels_nod2D.npy",
                               mmap_mode="r").shape[0])
        if rmap.fine_cell.shape[0] != n_fine or rmap.n_coarse != n_coarse:
            raise SystemExit(
                f"map cache {cache} is for a different mesh pair "
                f"({rmap.fine_cell.shape[0]} -> {rmap.n_coarse}, meshes say "
                f"{n_fine} -> {n_coarse}); delete it by hand, this script never will")
        print(f"[map] loaded cache {cache}")
        return rmap

    t0 = time.time()
    rmap = coarsen.build_remap(fine_dir, coarse_dir)
    print(f"[map] built in {time.time() - t0:.1f} s: "
          f"{rmap.fine_cell.shape[0]} fine -> {rmap.n_coarse} coarse nodes")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, fine_cell=np.asarray(rmap.fine_cell), fine_area=np.asarray(rmap.fine_area),
             n_coarse=rmap.n_coarse, fine_depth=np.asarray(rmap.fine_depth),
             coarse_depth=np.asarray(rmap.coarse_depth))
    return rmap


# ---------------------------------------------------------------------------
# the per-level kernel
# ---------------------------------------------------------------------------

FILL_ABOVE = 1e30      # FESOM's netCDF float fill is 9.96921e+36


def read_slice(ds, var, rec, k):
    """One (record, level) slice as float64 with the fill turned into an explicit NaN-free zero.

    ⚠️ ``np.asarray(ds.variables[v][rec, k, :])`` STRIPS netCDF4's mask and hands the fill value
    back as an ordinary number. That is the trap ``mesh_diag.load_w`` documents, and this script
    walked into it: below-bottom entries entered the products at 9.96921e+36. Zeroing here is
    necessary but NOT sufficient — a zeroed dry element still dilutes its neighbouring nodes, so
    the caller must ALSO pass the element mask. Both, or neither works.
    """
    raw = np.ma.asarray(ds.variables[var][rec, k, :])
    a = np.ma.filled(np.ma.masked_greater(np.ma.masked_invalid(raw), FILL_ABOVE), 0.0)
    return np.asarray(a, dtype=np.float64)


def level_products(T, S, u_el, v_el, mask, rmap, fine, elem_mask=None):
    """One fine level in, the coarse state and eddy fluxes for that level out.

    Returns ``(state, flux, weight)`` where ``state`` is (T,S,u,v), ``flux`` is
    (u'T', v'T', u'S', v'S') and ``weight`` is the wet cell area — 0 marks a coarse cell with
    nothing under it, which the caller must keep as "no data" rather than as a zero.
    """
    u = coarsen.elem_to_node(u_el, fine, elem_mask=elem_mask)
    v = coarsen.elem_to_node(v_el, fine, elem_mask=elem_mask)

    def agg(x):
        val, wgt = coarsen.aggregate_horizontal_with_weight(x, rmap, fine_mask=mask)
        return val, wgt

    (Tc, w), (Sc, _), (uc, _), (vc, _) = agg(T), agg(S), agg(u), agg(v)
    (uT, _), (vT, _), (uS, _), (vS, _) = agg(u * T), agg(v * T), agg(u * S), agg(v * S)
    # the Reynolds residual: mean of the product minus product of the means, on the same cell
    return (Tc, Sc, uc, vc), (uT - uc * Tc, vT - vc * Tc, uS - uc * Sc, vS - vc * Sc), w


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

STATE_NAMES = ("temp", "salt", "u", "v")
FLUX_NAMES = ("uT_eddy", "vT_eddy", "uS_eddy", "vS_eddy")


def verify_conservation(month, rec, rmap, fine, teacher: pathlib.Path, max_levels=None):
    """Check the horizontal aggregation ON THE REAL FILES, three ways.

    ⚠️ **The area-integral check alone proves less than it looks.** The coarse value is
    ``num/den`` and the coarse weight is ``den``, so ``Σ(value × weight) ≈ Σ num`` is close to an
    algebraic identity: it holds even if the mask is wrong, even if the slice is transposed, as
    long as BOTH sides use the same wrong thing. It is worth printing — it would catch a
    normalisation or scatter bug — but it is not the plumbing check. Two others are:

    * **Range.** An area-weighted average cannot leave the range of its inputs. If any coarse
      value falls outside the fine level's own min/max, the script is aggregating something other
      than what it measured the range of — a transpose, a wrong record, or netCDF fill leaking in
      (FESOM writes 9.96921e+36 for dry cells and ``np.asarray`` on a MaskedArray hands those
      back as numbers).
    * **Wet area.** The summed surface weight must equal the fine mesh's own total wet area. A
      mask built from the wrong level, or off by one on ``nlevels``, changes this and nothing
      else notices.

    The vertical remap is deliberately NOT in the loop — separate operator, its own conservation
    test, and mixing them would leave a failure ambiguous.
    """
    with netCDF4.Dataset(teacher / f"temp_snap.fesom.{month}.nc") as ds:
        nz1 = ds.variables["temp_snap"].shape[1]
        if max_levels:
            nz1 = min(nz1, int(max_levels))
    worst = 0.0
    worst_k = -1
    area = np.asarray(rmap.fine_area)
    _mask_warned = set()
    out_of_range = []
    area_err = None
    for k in range(nz1):
        m = np.asarray(fine.layer_mask(k))
        with netCDF4.Dataset(teacher / f"temp_snap.fesom.{month}.nc") as ds:
            T = read_slice(ds, "temp_snap", rec, k)
            # cross-check the MESH mask against the DATA's own fill pattern. They are independent
            # sources of the same truth, so a disagreement means one of them is wrong — which is
            # the only way to catch an off-by-one in either.
            filled = np.asarray(np.ma.getmaskarray(
                np.ma.masked_greater(np.ma.asarray(ds.variables["temp_snap"][rec, k, :]),
                                     FILL_ABOVE)))
        n_disagree = int(((m > 0) & filled).sum() + ((m == 0) & ~filled).sum())
        if n_disagree and k not in _mask_warned:
            _mask_warned.add(k)
            print(f"  ⚠️ level {k + 1}: mesh mask and data fill disagree on {n_disagree} nodes "
                  f"(mesh wet & filled: {int(((m > 0) & filled).sum())}, "
                  f"mesh dry & unfilled: {int(((m == 0) & ~filled).sum())})")
        Tc, w = coarsen.aggregate_horizontal_with_weight(
            jnp.asarray(T), rmap, fine_mask=jnp.asarray(m))
        Tc, w = np.asarray(Tc), np.asarray(w)

        fine_int = float((area * m * T).sum())
        coarse_int = float((Tc * w).sum())
        rel = abs(coarse_int - fine_int) / max(abs(fine_int), 1e-30)
        if rel > worst:
            worst, worst_k = rel, k

        wetf = m > 0
        if wetf.any():
            lo, hi = float(T[wetf].min()), float(T[wetf].max())
            got = Tc[w > 0]
            if got.size and (got.min() < lo - 1e-9 or got.max() > hi + 1e-9):
                out_of_range.append((k + 1, lo, hi, float(got.min()), float(got.max())))
        if k == 0:
            fine_wet_area = float((area * m).sum())
            coarse_wet_area = float(w.sum())
            area_err = abs(coarse_wet_area - fine_wet_area) / fine_wet_area
            print(f"  surface wet area: fine {fine_wet_area / 1e12:.4f} x 1e12 m^2, "
                  f"coarse {coarse_wet_area / 1e12:.4f}, rel {area_err:.2e}")
        if k % 10 == 0 or k == nz1 - 1:
            print(f"  level {k + 1:3d}/{nz1}: fine {fine_int:+.10e}  coarse {coarse_int:+.10e}"
                  f"  rel {rel:.2e}  range [{T[wetf].min() if wetf.any() else 0:+.3f}, "
                  f"{T[wetf].max() if wetf.any() else 0:+.3f}]", flush=True)

    print(f"\narea integral: worst relative discrepancy {worst:.3e} at level {worst_k + 1}")
    print(f"wet area     : relative discrepancy {area_err:.3e} at the surface")
    print(f"range        : {len(out_of_range)} level(s) with a coarse value outside the fine "
          f"level's own min/max")
    for k, lo, hi, glo, ghi in out_of_range[:5]:
        print(f"    level {k}: fine [{lo:+.4f}, {hi:+.4f}], coarse [{glo:+.4f}, {ghi:+.4f}]")
    ok = worst < 1e-12 and (area_err or 0.0) < 1e-12 and not out_of_range
    print("VERIFY OK" if ok else "VERIFY FAILED")
    return 0 if ok else 1


def process_record(month, rec, rmap, fine, teacher: pathlib.Path, verbose=True,
                   max_levels=None):
    """Coarse-grain one snapshot record, level by level. Returns (state, flux, weight, time).

    ``max_levels`` truncates the column — for smoke tests only. The vertical remap is then
    reading a fine column that stops early, so the deep coarse layers come back as no-data;
    the output is not physically usable, it just proves the machinery runs.

    ⚠️ The temp/salt files are chunked [1, 2, ~322k], i.e. one chunk spans TWO levels, so a
    strictly level-by-level read decompresses each of those chunks twice. u/v are chunked one
    level per chunk and dominate the volume anyway (59 MB vs 30 MB per level), so this is a
    ~10% effect on the read, not worth pairing the loop for.
    """
    def open_snap(field):
        p = teacher / f"{field}_snap.fesom.{month}.nc"
        if not p.exists():
            raise SystemExit(f"missing snapshot file {p}")
        return netCDF4.Dataset(p)

    ds = {f: open_snap(f) for f in ("temp", "salt", "u", "v")}
    try:
        # ⚠️ read the time, do not compute it: records sit one dt SHORT of the round 5-day
        # mark, and that offset changes when the time step does (120 s -> 240 s at month 03).
        t_val = float(ds["temp"].variables["time"][rec])
        nz1 = ds["temp"].variables["temp_snap"].shape[1]
        if max_levels:
            nz1 = min(nz1, int(max_levels))
        n_lev_out = int(rmap.coarse_depth.shape[0]) - 1

        state = [np.zeros((rmap.n_coarse, nz1)) for _ in STATE_NAMES]
        flux = [np.zeros((rmap.n_coarse, nz1)) for _ in FLUX_NAMES]
        weight = np.zeros((rmap.n_coarse, nz1))

        t0 = time.time()
        for k in range(nz1):
            m = jnp.asarray(fine.layer_mask(k))
            em = jnp.asarray(fine.elem_layer_mask(k))
            T = jnp.asarray(read_slice(ds["temp"], "temp_snap", rec, k))
            S = jnp.asarray(read_slice(ds["salt"], "salt_snap", rec, k))
            ue = jnp.asarray(read_slice(ds["u"], "u_snap", rec, k))
            ve = jnp.asarray(read_slice(ds["v"], "v_snap", rec, k))
            st, fl, w = level_products(T, S, ue, ve, m, rmap, fine, elem_mask=em)
            for arr, x in zip(state, st):
                arr[:, k] = np.asarray(x)
            for arr, x in zip(flux, fl):
                arr[:, k] = np.asarray(x)
            weight[:, k] = np.asarray(w)
            if verbose and (k % 10 == 0 or k == nz1 - 1):
                print(f"  level {k + 1:3d}/{nz1}  {time.time() - t0:6.1f} s", flush=True)
    finally:
        for d in ds.values():
            d.close()

    # vertical remap on the coarse columns, masked by whether the cell had anything in it
    fz, cz = np.asarray(rmap.fine_depth)[:nz1 + 1], np.asarray(rmap.coarse_depth)
    wet = jnp.asarray((weight > 0.0).astype(np.float64))
    state_v = [np.asarray(coarsen.remap_vertical(jnp.asarray(a), fz, cz, fine_mask=wet))
               for a in state]
    flux_v = [np.asarray(coarsen.remap_vertical(jnp.asarray(a), fz, cz, fine_mask=wet))
              for a in flux]
    weight_v = np.asarray(coarsen.remap_vertical(jnp.asarray(weight), fz, cz, fine_mask=wet))
    assert state_v[0].shape == (rmap.n_coarse, n_lev_out), state_v[0].shape
    return state_v, flux_v, weight_v, t_val


def write_output(path, month, times, states, fluxes, weights, n_coarse, n_lev):
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("nod2", n_coarse)
        ds.createDimension("nz1", n_lev)
        ds.description = ("NG5 teacher snapshots coarse-grained onto the CORE2 mesh. "
                          "Eddy fluxes are agg(u*tau) - agg(u)*agg(tau) with the Voronoi "
                          "cell average as the filter. cell_area == 0 marks NO DATA, not zero.")
        ds.source_month = month
        tv = ds.createVariable("time", "f8", ("time",))
        tv[:] = np.asarray(times)
        tv.long_name = "seconds since the run start, read from the snapshot file"
        for name, arrs in list(zip(STATE_NAMES, states)) + list(zip(FLUX_NAMES, fluxes)):
            v = ds.createVariable(name, "f4", ("time", "nod2", "nz1"), zlib=False)
            v[:] = np.stack(arrs).astype(np.float32)
        v = ds.createVariable("cell_area", "f4", ("time", "nod2", "nz1"), zlib=False)
        v[:] = np.stack(weights).astype(np.float32)
        v.long_name = "summed wet fine-node area behind each coarse value; 0 = no data"
    print(f"[out] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="e.g. 1958_01")
    ap.add_argument("--records", default=None,
                    help="comma-separated record indices; default all in the file")
    ap.add_argument("--teacher", type=pathlib.Path, default=TEACHER)
    ap.add_argument("--fine-mesh", type=pathlib.Path, default=FINE_MESH)
    ap.add_argument("--coarse-mesh", type=pathlib.Path, default=COARSE_MESH)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--map-cache", type=pathlib.Path, default=None)
    ap.add_argument("--max-levels", type=int, default=None,
                    help="truncate the column — SMOKE TEST ONLY, output is not usable")
    ap.add_argument("--verify-conservation", action="store_true",
                    help="check the area integral is preserved on the REAL files, per level, "
                         "and exit; writes nothing")
    args = ap.parse_args()

    cache = args.map_cache or (args.out / "remap_ng5_to_core2.npz")
    rmap = build_or_load_map(cache, args.fine_mesh, args.coarse_mesh)
    t0 = time.time()
    fine = FineMesh(args.fine_mesh)
    print(f"[mesh] fine nod2D={fine.nod2D} elem2D={fine.elem2D} nl={fine.nl} "
          f"({time.time() - t0:.1f} s)")

    with netCDF4.Dataset(args.teacher / f"temp_snap.fesom.{args.month}.nc") as ds:
        n_rec = ds.variables["temp_snap"].shape[0]
    recs = ([int(x) for x in args.records.split(",")] if args.records
            else list(range(n_rec)))
    print(f"[run] month {args.month}: {n_rec} records in file, processing {recs}")

    if args.verify_conservation:
        rc = 0
        for r in recs:
            print(f"[verify rec {r}] area-integral conservation, level by level")
            rc |= verify_conservation(args.month, r, rmap, fine, args.teacher,
                                      max_levels=args.max_levels)
        return rc

    times, states, fluxes, weights = [], [], [], []
    for r in recs:
        print(f"[rec {r}] start", flush=True)
        t0 = time.time()
        st, fl, w, tv = process_record(args.month, r, rmap, fine, args.teacher,
                                       max_levels=args.max_levels)
        times.append(tv)
        states.append(st)
        fluxes.append(fl)
        weights.append(w)
        print(f"[rec {r}] done in {time.time() - t0:.1f} s, t={tv:.0f} s "
              f"({tv / 86400.0:.3f} d)", flush=True)
        _report(st, fl, w, rmap, args.coarse_mesh)

    # transpose: per-variable list over records
    states_t = [[s[i] for s in states] for i in range(len(STATE_NAMES))]
    fluxes_t = [[f[i] for f in fluxes] for i in range(len(FLUX_NAMES))]
    tag = f"_SMOKE{args.max_levels}lev" if args.max_levels else ""
    write_output(args.out / f"coarse_ng5_{args.month}{tag}.nc", args.month, times,
                 states_t, fluxes_t, weights, rmap.n_coarse,
                 int(rmap.coarse_depth.shape[0]) - 1)
    return 0


def _report(state, flux, weight, rmap, coarse_mesh):
    """Sanity numbers, printed rather than asserted — the physical gate is a human looking at
    where the eddy-flux maxima are. A coarse-graining that had gone wrong shows up here first:
    temperatures outside [-3, 40], a wet fraction far from the ocean's ~71%, or an eddy heat
    flux whose hot spots are not the ACC, the Agulhas and the western boundary currents.
    """
    T, S = state[0], state[1]
    wet = weight[:, 0] > 0
    print(f"  wet coarse nodes: {wet.sum()}/{wet.size} ({100 * wet.mean():.1f}%)")
    for name, a in (("temp", T), ("salt", S)):
        v = a[weight > 0]
        print(f"  {name:5s} min/mean/max = {v.min():8.3f} {v.mean():8.3f} {v.max():8.3f}")
    mag = np.hypot(flux[0][:, 0], flux[1][:, 0])          # surface eddy heat flux magnitude
    lat = np.degrees(np.load(pathlib.Path(coarse_mesh) / "geo_coord_nod2D.npy")[:, 1])
    top = np.argsort(mag)[-200:]
    print(f"  surface |u'T'| max {mag.max():.4g} K m/s; top-200 nodes: "
          f"lat {lat[top].min():.0f}..{lat[top].max():.0f}, "
          f"{100 * np.mean((lat[top] < -35) & (lat[top] > -65)):.0f}% in the ACC band "
          f"(-65..-35), {100 * np.mean(lat[top] > 20):.0f}% north of 20N")


if __name__ == "__main__":
    raise SystemExit(main())
