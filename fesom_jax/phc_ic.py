"""PHC3.0 initial-condition reader (Task 5.2) — a faithful **numpy** port of
``fesom_phc.c`` (``load_one_variable`` + ``extrap_nod3D`` + ``insitu2pot``) for the
no-cavity case (``ulevels_nod2D==1``), bit-exact in both the serial (npes=1) and the
partition-faithful (``rank_nodes=…``) extrapolation orders.

The IC is a **one-time, host-side, non-differentiable setup** computation (it is *not*
in the autodiff path); the produced ``T``/``S`` fields are the model's initial state and
are a valid gradient target. We mirror the C bit-for-bit so the JAX CORE2 run starts from
the exact C initial condition (verified against the C ``phc_dump_*`` surface dumps —
byte-equal, ``test_phc_ic.py``).

Pipeline (``fesom_phc.c:387`` ``fesom_phc_load_ic``):
  1. read PHC ``temp``/``salt`` ``(depth,lat,lon)`` + coords; cyclic-pad lon by ±1 column.
  2. per-node bilinear bracket (``binarysearch_d`` on geographic node coords).
  3. ``load_one_variable``: bilinear-horizontal + linear-vertical interp onto ``mesh.Z``,
     leaving ``PHC_DUMMY`` where land/below-deepest-PHC. FP association mirrors the C
     exactly (``((v·wx)·wy)``, ``fesom_phc.c:215-218``) — bit-equality, not ~1e-14.
  4. ``extrap_nod3D``: per-layer **sequential Gauss-Seidel** land fill (order-dependent —
     replicated exactly) + a top→down vertical fill.
  5. cleanup (dummy→0, below ``nlevels``→0, K→°C) and ``insitu2pot`` (Bryden-1973 ptheta).

⚠️ The GS extrapolation is order-dependent: each dummy node is filled once, with the mean
of its neighbours valid *at fill time*. A vectorized Jacobi would give different values, so
we walk dummy nodes in fill order, updating in place (``fesom_phc.c:318-342``).

⚠️⚠️ Order-dependence ⇒ **the C IC is PARTITION-DEPENDENT**: under MPI each rank sweeps
only its OWN nodes (local ``myList_nod2D`` order) with halo values frozen between
``exchange_nod`` calls, so a 1-rank and a 16-rank C run differ at GS-filled nodes by up to
~25 PSU (Baltic/Kara basins seeded from different fill fronts). To match a C dump oracle
bit-for-bit, build the IC with that run's partition (``rank_nodes`` →
:func:`_extrap_nod3D_mpi`; ``scripts/rebuild_ic_dist16.py`` for the dist_16 z2_cdump).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import netCDF4

from .config import RAD

DEFAULT_IC_DIR = Path(__file__).resolve().parents[1] / "data" / "ic_core2"

PHC_DUMMY = 1.0e10
_DUMMY_HI = 0.99 * PHC_DUMMY
DEFAULT_PHC_PATH = "/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc"


# --------------------------------------------------------------------------
# Scalar helpers (literal ports)
# --------------------------------------------------------------------------
def _binarysearch_d(arr: np.ndarray, value: float) -> int:
    """``fesom_phc.c:38`` — 0-based index of element ≤ value, with a 1e-9 exact-match
    short-circuit and ceil-midpoint; returns ``right`` (= -1 if value < arr[0])."""
    d = 1e-9
    left, right = 0, len(arr) - 1
    while left <= right:
        middle = (left + right + 1) // 2
        if abs(arr[middle] - value) <= d:
            return middle
        if arr[middle] > value:
            right = middle - 1
        else:
            left = middle + 1
    return right


def _atg(s: float, t: float, p: float) -> float:
    """Adiabatic temperature gradient (Bryden 1973), ``fesom_phc.c:56``."""
    ds = s - 35.0
    return (((-2.1687e-16 * t + 1.8676e-14) * t - 4.6206e-13) * p
            + ((2.7759e-12 * t - 1.1351e-10) * ds + ((-5.4481e-14 * t
               + 8.733e-12) * t - 6.7795e-10) * t + 1.8741e-8)) * p \
        + (-4.2393e-8 * t + 1.8932e-6) * ds \
        + ((6.6228e-10 * t - 6.836e-8) * t + 8.5258e-6) * t + 3.5803e-5


def _ptheta(s, t, p, pr):
    """Potential temperature via RK4 (Bryden 1973), ``fesom_phc.c:66``. Vectorized over
    the node/level axis (s,t arrays; p array; pr scalar 0)."""
    h = pr - p
    xk = h * _atg(s, t, p)
    t = t + 0.5 * xk
    q = xk
    p = p + 0.5 * h
    xk = h * _atg(s, t, p)
    t = t + 0.29289322 * (xk - q)
    q = 0.58578644 * xk + 0.121320344 * q
    xk = h * _atg(s, t, p)
    t = t + 1.707106781 * (xk - q)
    q = 3.414213562 * xk - 4.121320344 * q
    p = p + 0.5 * h
    xk = h * _atg(s, t, p)
    return t + (xk - 2.0 * q) / 6.0


# --------------------------------------------------------------------------
# Stage 3: bilinear-horizontal + linear-vertical interpolation
# --------------------------------------------------------------------------
def _load_one_variable(ncvar, geo_deg, Z, nlevels_nod2D, node_layer_mask,
                       nc_lon, nc_lat, nc_depth, bilin_i, bilin_j, d_indx):
    """Port of ``load_one_variable`` (``fesom_phc.c:97``) for npes=1, no cavity (ul1=0).
    Returns ``out[N, nl]`` with ``PHC_DUMMY`` where unfilled. Vectorized over nodes."""
    Ndepth, Nlat, Nlon_real = ncvar.shape
    Nlon = Nlon_real + 2  # padded

    # ncdata[ilon, ilat, idep], cyclic-padded; NaN / |huge| → DUMMY (fesom_phc.c:136-162).
    real = np.transpose(np.asarray(ncvar, dtype=np.float64), (2, 1, 0))  # (Nlon_real,Nlat,Nd)
    ncdata = np.empty((Nlon, Nlat, Ndepth), dtype=np.float64)
    ncdata[1:Nlon - 1] = real
    ncdata[0] = real[Nlon_real - 1]   # west halo = last real (C: ncdata[0]=ncdata[Nlon-2])
    ncdata[Nlon - 1] = real[0]        # east halo = first real (C: ncdata[Nlon-1]=ncdata[1])
    bad = np.isnan(ncdata) | (ncdata < -_DUMMY_HI) | (ncdata > PHC_DUMMY)
    ncdata[bad] = PHC_DUMMY

    N = geo_deg.shape[0]
    nl = Z.shape[0] + 1
    out = np.full((N, nl), PHC_DUMMY, dtype=np.float64)

    i, j = bilin_i, bilin_j
    valid = (i >= 0) & (j >= 0)
    ic = np.where(valid, i, 0)
    jc = np.where(valid, j, 0)
    ip1 = ic + 1
    jp1 = jc + 1

    # 4 corner columns (N, Ndepth) — clamped indices; invalid nodes masked out below.
    v00 = ncdata[ic, jc, :]
    v10 = ncdata[ip1, jc, :]
    v01 = ncdata[ic, jp1, :]
    v11 = ncdata[ip1, jp1, :]

    # surface-corner gate (depth 0): skip the node if any corner is dummy (fesom_phc.c:194).
    corner_ok = ((v00[:, 0] <= _DUMMY_HI) & (v10[:, 0] <= _DUMMY_HI)
                 & (v01[:, 0] <= _DUMMY_HI) & (v11[:, 0] <= _DUMMY_HI))
    node_ok = valid & corner_ok

    x = geo_deg[:, 0]
    y = geo_deg[:, 1]
    x1 = nc_lon[ic]; x2 = nc_lon[ip1]
    y1 = nc_lat[jc]; y2 = nc_lat[jp1]
    denom = (x2 - x1) * (y2 - y1)
    wx2 = (x2 - x)[:, None]; wx1 = (x - x1)[:, None]
    wy2 = (y2 - y)[:, None]; wy1 = (y - y1)[:, None]
    # C association order (fesom_phc.c:215-218): ((v·wx)·wy) summed left-to-right —
    # grouping wx·wy first differs by ~1 ulp at ~27k nodes (vs the C 16r dump).
    data1d = ((v00 * wx2) * wy2 + (v10 * wx1) * wy2
              + (v01 * wx2) * wy1 + (v11 * wx1) * wy1) / denom[:, None]
    # per-depth: if any corner dummy at depth k → data1d[:,k]=DUMMY (fesom_phc.c:219).
    corner_dummy = ((v00 > _DUMMY_HI) | (v10 > _DUMMY_HI)
                    | (v01 > _DUMMY_HI) | (v11 > _DUMMY_HI))
    data1d = np.where(corner_dummy, PHC_DUMMY, data1d)

    # Vertical interp onto Z[k], k ∈ [0, nlevels_nod2D-1) (no cavity ul1=0). d_indx is
    # node-independent (depth_pos = -Z[k]). (fesom_phc.c:226-250)
    Ndepth_ = Ndepth
    for k in range(nl - 1):
        di = int(d_indx[k])
        lay = node_ok & node_layer_mask[:, k]   # node has this layer
        if di >= 0 and di < Ndepth_ - 1:
            d1 = data1d[:, di]
            d2 = data1d[:, di + 1]
            both = (d1 <= _DUMMY_HI) & (d2 <= _DUMMY_HI)
            delta_d = nc_depth[di + 1] - nc_depth[di]
            cf_a = (d2 - d1) / delta_d
            cf_b = d1 - cf_a * nc_depth[di]
            val = -cf_a * Z[k] + cf_b
            out[:, k] = np.where(lay & both, val, out[:, k])
        elif di == -1:
            d0 = data1d[:, 0]
            out[:, k] = np.where(lay & (d0 <= _DUMMY_HI), d0, out[:, k])
        # di == Ndepth-1 (below deepest PHC): stays DUMMY → vertical fill handles it.
    return out


# --------------------------------------------------------------------------
# Stage 4: extrap_nod3D — sequential Gauss-Seidel land fill + vertical fill
# --------------------------------------------------------------------------
def _extrap_nod3D(arr, nlevels_nod2D, nlevels_elem, off, flat, elem_nodes):
    """Port of ``extrap_nod3D`` (``fesom_phc.c:264``) for npes=1 (no MPI exchange).
    In-place. Per layer: sequential GS over dummy nodes in ascending index order
    (a node filled earlier in a sweep is visible to later nodes); then a top→down
    vertical fill. ``arr`` is ``[N, nl]``."""
    N, nl = arr.shape
    for nz in range(nl - 1):
        col = arr[:, nz]
        # active = dummy nodes that own this layer (fesom_phc.c:319-320), index order.
        has_layer = (nlevels_nod2D - 1) > nz
        active = np.nonzero((col > _DUMMY_HI) & has_layer)[0].tolist()
        if not active:
            continue
        progress = True
        sweep = 0
        while progress and sweep < 200:
            progress = False
            sweep += 1
            still = []
            for n in active:
                val = 0.0
                cnt = 0
                for kk in range(off[n], off[n + 1]):
                    el = flat[kk]
                    if nz > nlevels_elem[el] - 1:      # element doesn't reach this layer
                        continue
                    for jv in range(3):
                        v = elem_nodes[el, jv]
                        if v < 0 or v >= N:
                            continue
                        if col[v] <= _DUMMY_HI and (nlevels_nod2D[v] - 1) > nz:
                            val += col[v]
                            cnt += 1
                if cnt > 0:
                    col[n] = val / cnt           # in-place → visible to later n this sweep
                    progress = True
                else:
                    still.append(n)
            active = still

    # Vertical fill — top→down within each node's valid range (fesom_phc.c:361-369).
    for n in range(N):
        nl1 = nlevels_nod2D[n] - 1
        col = arr[n]
        for nz in range(1, nl1):
            if col[nz] > _DUMMY_HI:
                col[nz] = col[nz - 1]


def _extrap_nod3D_mpi(arr, nlevels_nod2D, nlevels_elem, off, flat, elem_nodes,
                      rank_nodes):
    """Port of ``extrap_nod3D`` (``fesom_phc.c:264``) under an ``npes>1`` partition.

    The GS land fill is order-dependent, so the C IC is **partition-dependent**: each
    rank sweeps only its OWN nodes in LOCAL index order, halo values are frozen at the
    last ``exchange_nod`` (once per outer iteration, ``fesom_phc.c:348``), and the outer
    loop continues only while the **surface** layer still has dummies anywhere
    (``fesom_phc.c:299-309``) — deep layers whose cross-rank propagation is unfinished
    when the surface converges are left to the vertical fill. ``rank_nodes`` is a list
    of per-rank 0-based global node indices in the C local order (``myList_nod2D``).

    Simulation of the concurrent ranks: all ranks of one outer iteration read the same
    snapshot (the post-exchange state); owned results merge into ``arr`` afterwards.
    ``arr`` is ``[N, nl]``, modified in place."""
    N, nl = arr.shape
    rank_nodes = [np.asarray(rn, dtype=np.int64) for rn in rank_nodes]
    it_outer = 0
    while it_outer < 200:
        it_outer += 1
        if arr[:, 0].max() <= _DUMMY_HI:
            break
        snapshot = arr.copy()
        any_progress = False
        for nz in range(nl - 1):
            col_snap = snapshot[:, nz]
            has_layer = (nlevels_nod2D - 1) > nz
            for own in rank_nodes:
                cand = own[(col_snap[own] > _DUMMY_HI) & has_layer[own]]
                if cand.size == 0:
                    continue
                work = col_snap.copy()      # owned live, halo frozen at snapshot
                active = cand.tolist()      # C local order
                progress = True
                sweep = 0
                while progress and sweep < 200:
                    progress = False
                    sweep += 1
                    still = []
                    for n in active:
                        val = 0.0
                        cnt = 0
                        for kk in range(off[n], off[n + 1]):
                            el = flat[kk]
                            if nz > nlevels_elem[el] - 1:
                                continue
                            for jv in range(3):
                                v = elem_nodes[el, jv]
                                if work[v] <= _DUMMY_HI and (nlevels_nod2D[v] - 1) > nz:
                                    val += work[v]
                                    cnt += 1
                        if cnt > 0:
                            work[n] = val / cnt
                            progress = True
                            any_progress = True
                        else:
                            still.append(n)
                    active = still
                arr[own, nz] = work[own]
        if not any_progress:
            break   # sim-only shortcut: a stalled iteration can never change the output

    # Vertical fill — top→down within each node's valid range (fesom_phc.c:361-369).
    for n in range(N):
        nl1 = nlevels_nod2D[n] - 1
        col = arr[n]
        for nz in range(1, nl1):
            if col[nz] > _DUMMY_HI:
                col[nz] = col[nz - 1]


# --------------------------------------------------------------------------
# Top-level
# --------------------------------------------------------------------------
@dataclasses.dataclass
class PHCResult:
    T: np.ndarray            # (N, nl) potential temperature, °C
    S: np.ndarray            # (N, nl) salinity, PSU
    bilin_i: np.ndarray      # (N,) lon bracket index (-1 = out of domain)
    bilin_j: np.ndarray      # (N,) lat bracket index
    T_pre_surf: np.ndarray   # (N,) surface T BEFORE extrap (for the pre-extrap gate)
    S_pre_surf: np.ndarray


def load_phc_ic(mesh, path: str = DEFAULT_PHC_PATH, t_insitu: bool = True,
                rank_nodes=None) -> PHCResult:
    """Build the PHC initial condition on ``mesh``. Returns :class:`PHCResult`
    with the final potential-T / S ``[N, nl]`` and the pre-extrap surface + bracket
    indices (for verifying against the C ``phc_dump_*`` surface dumps).

    ``rank_nodes`` (list of per-rank 0-based node-index arrays in C local order)
    selects the partition-faithful MPI extrapolation — the GS land fill is
    order-dependent, so matching a C oracle run bit-for-bit requires its partition
    (e.g. dist_16 for the 16-rank ``z2_cdump``). ``None`` = serial (npes=1) order."""
    nl = int(mesh.nl)
    Z = np.asarray(mesh.Z, dtype=np.float64)
    geo = np.asarray(mesh.geo_coord_nod2D, dtype=np.float64)
    geo_deg = geo / RAD
    geo_deg = geo_deg.copy()
    geo_deg[:, 0] = np.where(geo_deg[:, 0] < 0.0, geo_deg[:, 0] + 360.0, geo_deg[:, 0])
    geo_deg[:, 0] = np.where(geo_deg[:, 0] > 360.0, geo_deg[:, 0] - 360.0, geo_deg[:, 0])

    nlevels_nod2D = np.asarray(mesh.nlevels_nod2D, dtype=np.int64)
    nlevels_elem = np.asarray(mesh.nlevels, dtype=np.int64)
    off = np.asarray(mesh.nod_in_elem2D_offsets, dtype=np.int64)
    flat = np.asarray(mesh.nod_in_elem2D, dtype=np.int64)
    elem_nodes = np.asarray(mesh.elem_nodes, dtype=np.int64)
    node_layer_mask = np.asarray(mesh.node_layer_mask)

    nc = netCDF4.Dataset(path)
    try:
        nc_lon_real = np.asarray(nc.variables["lon"][:], dtype=np.float64)
        nc_lat = np.asarray(nc.variables["lat"][:], dtype=np.float64)
        nc_depth = np.asarray(nc.variables["depth"][:], dtype=np.float64)
        temp = np.ma.filled(nc.variables["temp"][:], np.nan).astype(np.float64)
        salt = np.ma.filled(nc.variables["salt"][:], np.nan).astype(np.float64)
    finally:
        nc.close()

    # Cyclic lon padding (fesom_phc.c:421-443): nc_lon[0]=last-360, [Nlon+1]=first+360.
    Nlon_real = nc_lon_real.shape[0]
    nc_lon = np.empty(Nlon_real + 2, dtype=np.float64)
    nc_lon[1:Nlon_real + 1] = nc_lon_real
    nc_lon[0] = nc_lon_real[-1] - 360.0
    nc_lon[Nlon_real + 1] = nc_lon_real[0] + 360.0
    Nlon = Nlon_real + 2

    # Per-node bilinear bracket (fesom_phc.c:450-462).
    N = geo_deg.shape[0]
    bilin_i = np.full(N, -1, dtype=np.int64)
    bilin_j = np.full(N, -1, dtype=np.int64)
    lon_lo, lon_hi = nc_lon[0], nc_lon[Nlon - 1]
    lat_lo, lat_hi = nc_lat[0], nc_lat[-1]
    for n in range(N):
        x = geo_deg[n, 0]
        y = geo_deg[n, 1]
        bi = _binarysearch_d(nc_lon, x) if (lon_lo <= x <= lon_hi) else -1
        bj = _binarysearch_d(nc_lat, y) if (lat_lo <= y <= lat_hi) else -1
        if bi >= Nlon - 1:
            bi = -1
        if bj >= len(nc_lat) - 1:
            bj = -1
        bilin_i[n] = bi
        bilin_j[n] = bj

    # depth bracket per layer (node-independent: depth_pos = -Z[k]).
    Ndepth = nc_depth.shape[0]
    d_indx = np.array([_binarysearch_d(nc_depth, -Z[k]) for k in range(nl - 1)], dtype=np.int64)

    T = _load_one_variable(temp, geo_deg, Z, nlevels_nod2D, node_layer_mask,
                           nc_lon, nc_lat, nc_depth, bilin_i, bilin_j, d_indx)
    S = _load_one_variable(salt, geo_deg, Z, nlevels_nod2D, node_layer_mask,
                           nc_lon, nc_lat, nc_depth, bilin_i, bilin_j, d_indx)
    T_pre_surf = T[:, 0].copy()
    S_pre_surf = S[:, 0].copy()

    if rank_nodes is None:
        _extrap_nod3D(T, nlevels_nod2D, nlevels_elem, off, flat, elem_nodes)
        _extrap_nod3D(S, nlevels_nod2D, nlevels_elem, off, flat, elem_nodes)
    else:
        _extrap_nod3D_mpi(T, nlevels_nod2D, nlevels_elem, off, flat, elem_nodes, rank_nodes)
        _extrap_nod3D_mpi(S, nlevels_nod2D, nlevels_elem, off, flat, elem_nodes, rank_nodes)

    # Cleanup (fesom_phc.c:528-540): dummy→0, below nlevels→0, K→°C.
    for arr in (T, S):
        np.putmask(arr, arr > 0.9 * PHC_DUMMY, 0.0)
    k = np.arange(nl)[None, :]
    below = k >= (nlevels_nod2D[:, None] - 1)
    T[below] = 0.0
    S[below] = 0.0
    np.putmask(T, T > 100.0, T - 273.15)

    # insitu → potential temperature (fesom_phc.c:544-557): ptheta over valid layers.
    if t_insitu:
        valid = node_layer_mask                       # k ∈ [ulevels-1, nlevels-1), ul=1
        pp = np.abs(Z)[None, :]                        # (1, nl-1) pressure ≈ |Z|
        Tlay = T[:, :nl - 1]
        Slay = S[:, :nl - 1]
        Tpot = _ptheta(Slay, Tlay, pp, 0.0)
        T[:, :nl - 1] = np.where(valid[:, :nl - 1], Tpot, Tlay)

    return PHCResult(T=T, S=S, bilin_i=bilin_i, bilin_j=bilin_j,
                     T_pre_surf=T_pre_surf, S_pre_surf=S_pre_surf)


def build_and_cache_ic(mesh, path: str = DEFAULT_PHC_PATH,
                       out_dir: str | Path = DEFAULT_IC_DIR,
                       rank_nodes=None) -> PHCResult:
    """Build the PHC IC on ``mesh`` and cache ``T_ic.npy``/``S_ic.npy`` to ``out_dir``
    (one-time; ``out_dir`` is gitignored under ``data/``)."""
    res = load_phc_ic(mesh, path, rank_nodes=rank_nodes)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "T_ic.npy", res.T)
    np.save(out_dir / "S_ic.npy", res.S)
    return res


def core2_initial_state(mesh, ic_dir: str | Path = DEFAULT_IC_DIR,
                        base_T: float = 10.0, base_S: float = 35.0):
    """Build a CORE2 :class:`~fesom_jax.state.State` from the cached PHC IC: a rest
    state (uv=0, eta=0, ``hnode`` from ``zbar_3d_n``) with ``T``/``S`` set to the PHC
    fields.

    ⚠️ **``T_old``/``S_old`` (the AB2 step-1 history) are the CONSTANT BASE
    (``base_T``/``base_S`` = 10/35), NOT the PHC field.** The C sets
    ``values = constant 10/35`` (``fesom_main.c:413``), runs the rest-sanity
    ``advect_one`` which saves ``valuesold = values = 10/35`` (``:724-756``), and only
    THEN loads PHC into ``values`` (``:778``) — leaving ``valuesold`` at the base. So at
    step 1 ``ttfAB = −(0.5+ε)·base + (1.5+ε)·PHC``, not ``PHC``. This is the CORE2 analog
    of the pi ``T_old``-is-the-pre-blob-base lesson; using ``T_old=PHC`` corrupts the
    step-1 FCT advection (~2e-3 in surface T)."""
    import jax.numpy as jnp
    from .state import State
    ic_dir = Path(ic_dir)
    T = jnp.asarray(np.load(ic_dir / "T_ic.npy"))
    S = jnp.asarray(np.load(ic_dir / "S_ic.npy"))
    mask = mesh.node_layer_mask
    T_old = jnp.where(mask, base_T, 0.0)
    S_old = jnp.where(mask, base_S, 0.0)
    st = State.rest(mesh, T0=base_T, S0=base_S)            # hnode/helem from zbar_3d_n
    return dataclasses.replace(st, T=T, S=S, T_old=T_old, S_old=S_old)
