"""NN-GM child-plan M3 — the Arm-A network's features and the formula base κ, computed from a
coarse TEACHER product instead of from a live student state.

The offline fit (Stage 0) needs three things at every teacher mark: the feature vector the seam
would show the network, the GM/Redi coefficients the default formula would produce there, and
the geometry (neutral slope, ∇_h σ, N²) that the κ̂ projection needs to interpret the product's
eddy fluxes. All three come from the SAME derived fields, and this module produces them.

**Nothing here reimplements the model.** Every derived quantity is computed by calling the
model's own function on the product's T and S:

    eos.compute_pressure_bv      → N² (with the model's own one-pass smoother)
    eos.compute_sw_alpha_beta    → α, β
    gm.compute_sigma_xy          → ∇_h σ, the density gradient the GM block actually uses
    gm.compute_neutral_slope     → neutral slope, ODM95-tapered slope, the taper factor
    gm.init_redi_gm              → fer_K (GM) and Ki (Redi) — the FORMULA BASE the fit's
                                   target ratios are taken against
    gm.gm_nn_features            → the feature tensor, the same call the seam makes

That list is the point. A second implementation of the feature list, of the arcsinh scales, or
of the slope taper would be a train/deploy skew that no test on either side alone could see:
the offline fit would learn a mapping from features the deployed seam never produces, and the
symptom would appear months later as "Stage 0 initialisation does not help".

**The documented approximation: static zbar geometry.** The seam runs inside a zstar model and
sees live layer thicknesses (``Z3d``/``zbar3`` from the state's ``hnode``). A coarse product
carries T and S but no thicknesses, so everything here is computed at rest geometry
(``State.rest``, thicknesses from ``mesh.zbar_3d_n``, ``Z3d=None``). This keeps M3 independent
of M4 — the student trajectories that would supply live thicknesses do not exist for 1959+ until
M4 runs — at the price of a small, one-signed error wherever the free surface is displaced.
:func:`parity_against_state` measures that error rather than assuming it is small, and is what
the plan's parity checkbox reports.

**Holes.** ``cell_area == 0`` in the product means NO DATA. Most such cells are simply below
bathymetry and agree with the student's own dry mask, but 3,293 whole columns (2.60 %) are
coastline mismatch: wet for the student, absent for the teacher, and stored as a literal 0.0 in
T and S. A 0 °C / 0 psu cell next to a 20 °C ocean is a density cliff, and ∇_h σ is an
element-wise gradient, so an unfilled hole corrupts a whole ring of NEIGHBOURING columns that
are themselves perfectly good data. Two separate steps handle that, and they are different
claims: :func:`fill_holes` makes the operator see no cliff, and ``usable`` excludes the hole AND
its one-ring from anything that gets fitted.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# ⚠️ MUST come before `import fesom_jax` — `python scripts/foo.py` puts the SCRIPT's directory
# on sys.path and never the cwd, so a bare import resolves through the editable install to the
# MAIN checkout. `gm.gm_nn_features` and the whole seam exist only in this worktree.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fesom_jax  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

_got = Path(fesom_jax.__file__).resolve()
if ROOT not in _got.parents:
    raise SystemExit(f"FATAL: fesom_jax imported from {_got}, not from {ROOT}")

from fesom_jax import eos, gm, ops  # noqa: E402
from fesom_jax.calibrate import build_params  # noqa: E402
from fesom_jax.gm import GMConfig  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.params import Params  # noqa: E402
from fesom_jax.state import State  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_data as _nd  # noqa: E402

MESH_DIR = ROOT / "data" / "mesh_core2"
GM_CFG = GMConfig()

# How many neighbour-averaging passes the hole fill gets. Three reaches three rings inward,
# which covers every coastline hole measured on this mesh pair (the widest is a 2-cell inlet);
# whatever is still unfilled falls back to the level mean, and `usable` excludes it regardless.
FILL_PASSES = 3


# ---------------------------------------------------------------------------
# geometry and hole handling
# ---------------------------------------------------------------------------
def expand_levels(arr47, nl: int):
    """``(N, nz_teacher) → (N, nl)``, zero-padding the model's trailing below-bottom columns.

    The inverse of :func:`nn_data.model_to_teacher_levels`, and it asserts the same thing that
    function does: the extra model columns must be padding. Slicing or padding blind here cost
    this project a failed job on 2026-08-08."""
    a = np.asarray(arr47)
    assert a.ndim == 2, a.shape
    n, nz = a.shape
    assert nz <= nl, (nz, nl)
    out = np.zeros((n, nl), a.dtype)
    out[:, :nz] = a
    return out


def node_neighbour_sum(mesh, field, weight):
    """``(Σ_neighbours w·f, Σ_neighbours w)`` over the element graph, per node and level.

    Element-based adjacency: every node accumulates from each element it belongs to, so the
    stencil is exactly the one :func:`fesom_jax.gm.compute_sigma_xy` gathers over. Uses the
    library scatter so the mesh's own index conventions apply."""
    e3 = mesh.elem_nodes                                   # (E,3)
    f_e = ops.gather_nodes_to_elem(field, e3)              # (E,3,nl)
    w_e = ops.gather_nodes_to_elem(weight, e3)
    # each element's contribution to each of its nodes is the SUM over the element's 3 nodes
    fw = jnp.sum(f_e * w_e, axis=1)                        # (E,nl)
    ww = jnp.sum(w_e, axis=1)
    vals = jnp.stack([jnp.broadcast_to(fw[:, None, :], f_e.shape),
                      jnp.broadcast_to(ww[:, None, :], w_e.shape)], axis=-1)   # (E,3,nl,2)
    acc = ops.scatter_add(vals, e3, mesh.nod2D)            # (N,nl,2)
    return acc[..., 0], acc[..., 1]


def fill_holes(mesh, field, have, *, passes: int = FILL_PASSES, fallback_axis: int = 0):
    """Replace no-data cells with a neighbour average, so the gradient operators see no cliff.

    ``have`` is True where the value is real. Each pass replaces a missing cell by the mean of
    its element-graph neighbours that are already known, then marks it known. Cells still
    missing after ``passes`` take the level mean over known cells (an isolated inland lake on
    one mesh and not the other has no neighbour to borrow from).

    This is cosmetic for the FIT — every filled cell and its one-ring is excluded from fitting
    by ``usable`` — but it is load-bearing for everything OUTSIDE that ring: a literal 0.0 °C
    beside a 20 °C column puts a fabricated 20 K gradient into the element stencil, and
    ``compute_sigma_xy`` would hand that to every good column sharing an element with it."""
    f = jnp.where(have, jnp.asarray(field, jnp.float64), 0.0)
    known = jnp.asarray(have, jnp.float64)
    for _ in range(int(passes)):
        s, c = node_neighbour_sum(mesh, f, known)
        avail = c > 0.0
        newv = jnp.where(avail, s / jnp.where(avail, c, 1.0), 0.0)
        take = (known <= 0.0) & avail
        f = jnp.where(take, newv, f)
        known = jnp.where(take, 1.0, known)
    # level-mean fallback for anything still unknown
    kn = known > 0.0
    cnt = jnp.sum(kn, axis=fallback_axis, keepdims=True)
    tot = jnp.sum(jnp.where(kn, f, 0.0), axis=fallback_axis, keepdims=True)
    lvl = jnp.where(cnt > 0, tot / jnp.where(cnt > 0, cnt, 1.0), 0.0)
    f = jnp.where(kn, f, jnp.broadcast_to(lvl, f.shape))
    return f, kn


def one_ring(mesh, flag):
    """True at every node/level that IS ``flag`` or shares an element with a ``flag`` cell."""
    s, _ = node_neighbour_sum(mesh, jnp.asarray(flag, jnp.float64),
                              jnp.ones_like(jnp.asarray(flag, jnp.float64)))
    return (s > 0.0) | jnp.asarray(flag, bool)


# ---------------------------------------------------------------------------
# the derived bundle
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Derived:
    """Everything M3 needs at one teacher mark. Layer-centred fields are ``(N, nl-1)``,
    interface fields ``(N, nl)``; ``feats`` is ``(N, nl-1, n_features)``."""
    T: jnp.ndarray             # (N,nl) model-shaped, hole-filled
    S: jnp.ndarray
    bvfreq: jnp.ndarray        # (N,nl) interfaces, model-smoothed N²
    sw_alpha: jnp.ndarray      # (N,nl)
    sw_beta: jnp.ndarray
    sigma_xy: jnp.ndarray      # (N,nl,2) ∇_h σ  [kg/m^4]
    neutral_slope: jnp.ndarray  # (N,nl-1,3) untapered (x, y, |s|)
    slope_tapered: jnp.ndarray  # (N,nl-1,3) ODM95-tapered — the operator's own slope
    fer_K: jnp.ndarray         # (N,nl) GM thickness diffusivity on INTERFACES  [m^2/s]
    Ki: jnp.ndarray            # (N,nl) Redi diffusivity on LAYERS              [m^2/s]
    fer_K_lay: jnp.ndarray     # (N,nl-1) fer_K averaged to layer centres
    feats: jnp.ndarray         # (N,nl-1,F)
    lay: jnp.ndarray           # (N,nl-1) bool: model's true layer range
    have: jnp.ndarray          # (N,nl) bool: the teacher had data here
    usable: jnp.ndarray        # (N,nl-1) bool: layer cells safe to FIT on
    n_filled: int
    n_ring_excluded: int
    # Baroclinic wave speed squared, (N,), from init_redi_gm. Kept because the a-priori
    # evaluation stratifies by Δ·|f|/√fer_C — grid scale over Rossby radius — which is the
    # resolution signal the network has no input for. Optional so that older constructions of
    # this dataclass (and the state-derived path below) stay valid.
    fer_C: jnp.ndarray | None = None


def derive(mesh, T47, S47, have47, *, params: Params | None = None,
           gm_cfg: GMConfig = GM_CFG, hnode=None, verbose: bool = True) -> Derived:
    """Derived fields at one mark, from the product's own T and S on static zbar geometry.

    ``have47`` is ``cell_area > 0`` from the product. ``hnode``/``Z3d``: rest geometry, see the
    module docstring. ``params`` defaults to :meth:`Params.defaults`, i.e. the same ``k_gm`` and
    ``redi_kmax`` the student runs with, so ``fer_K``/``Ki`` here ARE the base the seam's
    multiplier would scale."""
    nl = mesh.nl
    if params is None:
        params = build_params({})
    if hnode is None:
        hnode = State.rest(mesh).hnode          # thicknesses from mesh.zbar_3d_n

    have = jnp.asarray(expand_levels(np.asarray(have47, bool), nl))
    wet = jnp.asarray(mesh.node_layer_mask, bool)
    # A cell the student calls wet but the teacher has no data for is the coastline mismatch —
    # it must be filled before the gradient operators run and excluded from the fit afterwards.
    missing = wet & ~have
    n_missing = int(jnp.sum(missing))

    T, _ = fill_holes(mesh, expand_levels(np.asarray(T47, np.float64), nl), have)
    S, _ = fill_holes(mesh, expand_levels(np.asarray(S47, np.float64), nl), have)
    T = jnp.where(wet, T, 0.0)
    S = jnp.where(wet, S, 0.0)

    _, _, bvfreq = eos.compute_pressure_bv(mesh, T, S, hnode, Z3d=None)
    sw_alpha, sw_beta = eos.compute_sw_alpha_beta(mesh, T, S, Z3d=None)
    sigma_xy = gm.compute_sigma_xy(mesh, T, S, sw_alpha, sw_beta, gm_cfg)
    neutral_slope, slope_tapered, fer_tapfac = gm.compute_neutral_slope(mesh, sigma_xy,
                                                                       bvfreq, gm_cfg)
    fer_K, Ki, fer_C, _scal = gm.init_redi_gm(mesh, bvfreq, hnode, fer_tapfac, params,
                                              gm_cfg, zbar3=None)
    feats, lay = gm.gm_nn_features(mesh, slope_tapered, sigma_xy, bvfreq, gm_cfg, Z3d=None)

    # fer_K lives on interfaces; the fit's target is a per-LAYER ratio, so average it the same
    # way gm_nn_features averages bvfreq. (The seam applies m_GM[:,k] to the interface ABOVE
    # layer k — a deliberate half-cell offset documented on gm.gm_nn_multipliers. Averaging
    # here is the layer-centred base that ratio is taken against; the offset is smaller than
    # the 500 m exponential it rides on.)
    fer_K_lay = 0.5 * (fer_K[:, :nl - 1] + fer_K[:, 1:nl])

    ring = one_ring(mesh, missing)[:, :nl - 1]
    usable = lay & ~ring
    n_ring = int(jnp.sum(ring & lay))
    if verbose:
        print(f"[derive] teacher holes inside the student's wet mask: {n_missing} cells; "
              f"filled and excluded with their one-ring ({n_ring} layer cells, "
              f"{100 * n_ring / max(int(jnp.sum(lay)), 1):.2f}% of wet layer cells)", flush=True)
    return Derived(T=T, S=S, bvfreq=bvfreq, sw_alpha=sw_alpha, sw_beta=sw_beta,
                   sigma_xy=sigma_xy, neutral_slope=neutral_slope, slope_tapered=slope_tapered,
                   fer_K=fer_K, Ki=Ki, fer_K_lay=fer_K_lay, feats=feats, lay=lay, have=have,
                   usable=usable, n_filled=n_missing, n_ring_excluded=n_ring, fer_C=fer_C)


def derive_from_state(mesh, st, *, params: Params | None = None, gm_cfg: GMConfig = GM_CFG,
                      static_geometry: bool = False) -> Derived:
    """The same bundle from a student ``State`` — the reference side of the parity test.

    ``static_geometry=True`` reruns it at rest geometry with the SAME T and S, which isolates
    the static-zbar approximation from every other difference between the two paths."""
    nl = mesh.nl
    if params is None:
        params = build_params({})
    from fesom_jax import ale
    if static_geometry:
        hnode, Z3d, zbar3 = State.rest(mesh).hnode, None, None
    else:
        hnode = st.hnode
        zbar3, Z3d = ale.live_geometry(mesh, st.hnode)   # NB order: (zbar_3d_n, Z_3d_n)

    T, S = st.T, st.S
    _, _, bvfreq = eos.compute_pressure_bv(mesh, T, S, hnode, Z3d=Z3d)
    sw_alpha, sw_beta = eos.compute_sw_alpha_beta(mesh, T, S, Z3d=Z3d)
    sigma_xy = gm.compute_sigma_xy(mesh, T, S, sw_alpha, sw_beta, gm_cfg)
    neutral_slope, slope_tapered, fer_tapfac = gm.compute_neutral_slope(mesh, sigma_xy,
                                                                       bvfreq, gm_cfg)
    fer_K, Ki, fer_C, _s = gm.init_redi_gm(mesh, bvfreq, hnode, fer_tapfac, params, gm_cfg,
                                           zbar3=zbar3)
    feats, lay = gm.gm_nn_features(mesh, slope_tapered, sigma_xy, bvfreq, gm_cfg, Z3d=Z3d)
    fer_K_lay = 0.5 * (fer_K[:, :nl - 1] + fer_K[:, 1:nl])
    wet = jnp.asarray(mesh.node_layer_mask, bool)
    return Derived(T=T, S=S, bvfreq=bvfreq, sw_alpha=sw_alpha, sw_beta=sw_beta,
                   sigma_xy=sigma_xy, neutral_slope=neutral_slope, slope_tapered=slope_tapered,
                   fer_K=fer_K, Ki=Ki, fer_K_lay=fer_K_lay, feats=feats, lay=lay, have=wet,
                   usable=lay, n_filled=0, n_ring_excluded=0, fer_C=fer_C)


# ---------------------------------------------------------------------------
# the parity test (plan M3 checkbox 1)
# ---------------------------------------------------------------------------
def parity_against_state(mesh, st, *, gm_cfg: GMConfig = GM_CFG, verbose: bool = True):
    """How far the product path's features sit from the seam's, on the SAME ocean.

    Build a teacher-shaped product from a student state (T, S trimmed to the teacher's levels,
    every student-wet cell marked as having data), push it through :func:`derive`, and compare
    against :func:`derive_from_state` on the state itself. There is no remapping and no
    coarse-graining in between, so **every difference is the static-zbar approximation** plus
    the hole handling, which is exactly what the plan asks this test to quantify.

    Reported per feature as the median and 99th percentile of |Δ| over wet layer cells, in the
    network's own arcsinh-normalised units — the units that matter, because that is what the
    first layer multiplies. A tolerance is deliberately NOT asserted here: the number IS the
    result, it goes in the plan, and the fit's sensitivity to it is measured downstream by
    refitting on state-derived features. Asserting a threshold picked before seeing the number
    would be setting a bar the reference has not proved it can meet."""
    nz_t = int(np.asarray(mesh.node_layer_mask).shape[1]) - 1
    wet_model = np.asarray(mesh.node_layer_mask, bool)
    sl = _nd.model_to_teacher_levels(wet_model, nz_t)
    T47 = np.asarray(st.T)[:, sl]
    S47 = np.asarray(st.S)[:, sl]
    have47 = wet_model[:, sl]

    d_prod = derive(mesh, T47, S47, have47, gm_cfg=gm_cfg, verbose=verbose)
    d_live = derive_from_state(mesh, st, gm_cfg=gm_cfg)
    d_stat = derive_from_state(mesh, st, gm_cfg=gm_cfg, static_geometry=True)

    lay = np.asarray(d_live.lay, bool)
    names = ["arcsinh(slope)", "arcsinh(N2)", "arcsinh(|grad_b|)", "arcsinh(f)",
             "arcsinh(depth)", "depth_frac", "arcsinh(strain)"]
    rows = []
    fp, fl, fs = (np.asarray(x.feats) for x in (d_prod, d_live, d_stat))
    for i, nm in enumerate(names[:fp.shape[-1]]):
        dv_live = np.abs(fp[..., i] - fl[..., i])[lay]
        dv_stat = np.abs(fp[..., i] - fs[..., i])[lay]
        rows.append(dict(feature=nm,
                         med_vs_live=float(np.median(dv_live)),
                         p99_vs_live=float(np.percentile(dv_live, 99)),
                         max_vs_live=float(dv_live.max()),
                         med_vs_static=float(np.median(dv_stat)),
                         p99_vs_static=float(np.percentile(dv_stat, 99))))
    kap = []
    for nm, a, b, c in (("fer_K_lay", d_prod.fer_K_lay, d_live.fer_K_lay, d_stat.fer_K_lay),
                        ("Ki", d_prod.Ki[:, :-1], d_live.Ki[:, :-1], d_stat.Ki[:, :-1])):
        a, b, c = (np.asarray(x)[lay] for x in (a, b, c))
        den = np.maximum(np.abs(b), 1e-30)
        kap.append(dict(field=nm, med_rel_vs_live=float(np.median(np.abs(a - b) / den)),
                        p99_rel_vs_live=float(np.percentile(np.abs(a - b) / den, 99)),
                        med_rel_vs_static=float(np.median(np.abs(a - c)
                                                          / np.maximum(np.abs(c), 1e-30)))))
    if verbose:
        print("\n[parity] product-derived features vs the seam's, over wet layer cells")
        print("         (arcsinh units — what the first layer actually multiplies)")
        print(f"  {'feature':20s} {'med|Δ| live':>12s} {'p99 live':>10s} {'max live':>10s} "
              f"{'med|Δ| static':>14s} {'p99 static':>11s}")
        for r in rows:
            print(f"  {r['feature']:20s} {r['med_vs_live']:12.3e} {r['p99_vs_live']:10.3e} "
                  f"{r['max_vs_live']:10.3e} {r['med_vs_static']:14.3e} "
                  f"{r['p99_vs_static']:11.3e}")
        print("\n[parity] formula base κ, relative")
        for r in kap:
            print(f"  {r['field']:12s} median {r['med_rel_vs_live']:.3e}  "
                  f"p99 {r['p99_rel_vs_live']:.3e}  (vs static geometry: "
                  f"{r['med_rel_vs_static']:.3e})")
        print("\n  'static' columns isolate the zbar approximation: they compare the product "
              "path\n  against the SAME state also evaluated at rest geometry, so anything "
              "left there is\n  hole handling and level round-tripping, not the free surface.",
              flush=True)
    return dict(features=rows, kappa=kap,
                n_filled=d_prod.n_filled, n_ring_excluded=d_prod.n_ring_excluded)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_product_mark(path: Path, record: int):
    """``(T, S, have, t_seconds)`` for one record of a coarse product."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        t = float(np.asarray(ds.variables["time"][record]))
        T = np.asarray(ds.variables["temp"][record], np.float64)
        S = np.asarray(ds.variables["salt"][record], np.float64)
        have = np.asarray(ds.variables["cell_area"][record]) > 0.0
    return T, S, have, t


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--month", default="1958_06", help="e.g. 1958_06")
    ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--coarse-dir", type=Path, default=_nd.COARSE_DIR)
    ap.add_argument("--parity-snap", type=Path, default=None,
                    help="a student State pickle; runs the plan's parity check on it")
    ap.add_argument("--dump", type=Path, default=None,
                    help="write the derived bundle to this .npz (large; diagnostics only)")
    args = ap.parse_args()

    t0 = time.time()
    mesh = load_mesh(MESH_DIR)
    print(f"[setup] mesh {mesh.nod2D} nodes x {mesh.nl} levels in {time.time()-t0:.1f}s",
          flush=True)

    if args.parity_snap is not None:
        import pickle
        with open(args.parity_snap, "rb") as fh:
            st = pickle.load(fh)
        st = jax.tree.map(lambda x: jnp.asarray(x) if hasattr(x, "shape") else x, st)
        parity_against_state(mesh, st)
        return

    p = Path(args.coarse_dir) / f"coarse_ng5_{args.month}.nc"
    T47, S47, have47, tsec = load_product_mark(p, args.record)
    print(f"[product] {p.name} record {args.record}: t={tsec:.0f}s "
          f"(mark {_nd.mark_index(tsec)})", flush=True)
    d = derive(mesh, T47, S47, have47)

    lay = np.asarray(d.lay, bool)
    use = np.asarray(d.usable, bool)
    f = np.asarray(d.feats)
    print(f"[derive] {int(lay.sum())} wet layer cells, {int(use.sum())} usable "
          f"({100*use.sum()/max(lay.sum(),1):.2f}%)")
    for i, nm in enumerate(["slope", "N2", "|grad_b|", "f", "depth", "depth_frac", "strain"]):
        v = f[..., i][use]
        print(f"  feat {i} {nm:11s} min {v.min():+.4f}  med {np.median(v):+.4f}  "
              f"max {v.max():+.4f}  finite {np.isfinite(v).all()}")
    for nm, arr in (("fer_K_lay", d.fer_K_lay), ("Ki", d.Ki[:, :-1])):
        v = np.asarray(arr)[use]
        print(f"  base {nm:10s} min {v.min():.4g}  med {np.median(v):.4g}  max {v.max():.4g}"
              f"  [m^2/s]")
    if args.dump is not None:
        np.savez_compressed(args.dump, feats=f.astype(np.float32),
                            fer_K_lay=np.asarray(d.fer_K_lay, np.float32),
                            Ki=np.asarray(d.Ki, np.float32), usable=use, lay=lay)
        print(f"[out] {args.dump}")
    print(f"[done] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
