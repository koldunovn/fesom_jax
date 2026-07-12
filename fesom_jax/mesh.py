"""Static mesh & geometry for the FESOM2 → JAX port (Task 1.1).

Loads the C-exported pi mesh (``data/mesh_pi/``, produced by
``fesom2_port/src/fesom_mesh_export.c`` on branch ``jax-mesh-export``) into a
frozen :class:`Mesh` dataclass that is registered as a JAX **pytree**: the
geometry/connectivity arrays are pytree *leaves* (so a ``Mesh`` can be passed
through ``jit``/``scan`` or closed over), while the scalar counts
(``nod2D``, ``elem2D`` …) are static *metadata* (they fix array shapes and
``segment_sum`` segment counts, so they must be Python ints, not traced).

Provenance & conventions (see ``docs/MESH_EXPORT_LAYOUT.md``):

* **Indices are already 0-based** in the export (the C exporter converted them
  from the 1-based mesh files on write). We do **not** re-convert. ``edge_tri``
  and ``edge_up_dn_tri`` carry ``-1`` for a missing/boundary neighbour.
* Arrays are C order; a C ``[n*k]`` array is shape ``(n, k)``. ``real_t`` →
  ``float64`` (x64), C ``int`` → ``int32``.
* The 3D field layout is row-major ``[n_entity, nl]`` (C macro
  ``FESOM_NODE3D(node,lev,nl) = node*nl + lev``), so a JAX ``[n_entity, nl]``
  array indexes identically.

Vertical ragged extent (the masks). For node ``n`` the C kernels use
``nzmin = ulevels_nod2D[n]-1`` (0-based top interface) and
``nzmax = nlevels_nod2D[n]-1`` (0-based; *exclusive* layer bound). Hence:

* **layer** fields (T, S, density, pressure, u, v, …) are valid for levels
  ``k ∈ [nzmin, nzmax)`` — i.e. ``ulevels-1 ≤ k < nlevels-1``.
* **interface** fields (bvfreq, w, Kv, Av) are valid for ``k ∈ [nzmin, nzmax]``
  inclusive — i.e. ``ulevels-1 ≤ k < nlevels``  (one deeper than the layers).

(verified against ``fesom_eos.c:93-208`` — the density loop runs ``nz<nzmax``;
bvfreq is filled on the interior and padded at ``nzmin``/``nzmax``.)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from .config import CYCLIC_LENGTH_RAD

# The pi mesh (3140 nodes, 5 MB) SHIPS INSIDE the package, so a pip-installed fesom-jax
# can run the model with no external data. The repo-root <repo>/data/mesh_pi (the /work
# export) is honoured as a fallback for older checkouts.
_PACKAGED_PI_MESH_DIR = Path(__file__).resolve().parent / "data" / "mesh_pi"
_REPO_PI_MESH_DIR = Path(__file__).resolve().parents[1] / "data" / "mesh_pi"
DEFAULT_PI_MESH_DIR = (_PACKAGED_PI_MESH_DIR if _PACKAGED_PI_MESH_DIR.is_dir()
                       else _REPO_PI_MESH_DIR)


def _meta(static: bool = False) -> dataclasses.Field:
    """A dataclass field flagged as static pytree metadata when ``static``."""
    return dataclasses.field(metadata={"static": static})


@dataclasses.dataclass(frozen=True)
class Mesh:
    """Frozen, static FESOM2 mesh + geometry (a registered JAX pytree).

    Field groups mirror ``docs/MESH_EXPORT_LAYOUT.md``. All array fields are
    pytree leaves; the trailing scalar counts are static metadata.
    """

    # --- nodes (leading dim nod2D) ---
    coord_nod2D: jax.Array          # (nod2D,2) f8  (lon,lat) ROTATED radians
    geo_coord_nod2D: jax.Array      # (nod2D,2) f8  (lon,lat) GEOGRAPHIC radians
    coast_flag: jax.Array           # (nod2D,)  i4  0=interior 1=coast
    nlevels_nod2D: jax.Array        # (nod2D,)  i4  K_v⁺ (MAX over cells)
    nlevels_nod2D_min: jax.Array    # (nod2D,)  i4  K_v⁻ (MIN over cells)
    ulevels_nod2D: jax.Array        # (nod2D,)  i4  upper level (1 = no cavity)
    ulevels_nod2D_max: jax.Array    # (nod2D,)  i4  MAX of ulevels over cells
    depth: jax.Array                # (nod2D,)  f8  bathymetry, m
    mesh_resolution: jax.Array      # (nod2D,)  f8  Voronoi diameter, m
    coriolis_node: jax.Array        # (nod2D,)  f8  2Ω sin(geo lat), s⁻¹
    area: jax.Array                 # (nod2D,nl) f8 upper-edge scalar CV area, m²
    areasvol: jax.Array             # (nod2D,nl) f8 "mid" CV area, m²
    zbar_3d_n: jax.Array            # (nod2D,nl) f8 per-node interface depths, m
    nod_in_elem2D_offsets: jax.Array  # (nod2D+1,) i4 CSR offsets

    # --- elements (leading dim elem2D) ---
    elem_nodes: jax.Array           # (elem2D,3) i4 node ids 0-based, CW
    nlevels: jax.Array              # (elem2D,)  i4 K_c per-cell level count
    ulevels: jax.Array              # (elem2D,)  i4 upper level per cell
    elem_area: jax.Array            # (elem2D,)  f8 cell area, m²
    elem_cos: jax.Array             # (elem2D,)  f8 cos(rot lat at centroid)
    metric_factor: jax.Array        # (elem2D,)  f8 tan(rot lat)/R_earth, m⁻¹
    coriolis: jax.Array             # (elem2D,)  f8 2Ω sin(geo lat centroid), s⁻¹
    elem_center_x: jax.Array        # (elem2D,)  f8 centroid x, rot radians
    elem_center_y: jax.Array        # (elem2D,)  f8 centroid y, rot radians
    gradient_sca: jax.Array         # (elem2D,6) f8 ∂N/∂x(0..2),∂N/∂y(3..5), 1/m

    # --- edges (leading dim edge2D, except edge_up_dn_tri = myDim_edge2D) ---
    edges: jax.Array                # (edge2D,2) i4 node ids 0-based
    edge_tri: jax.Array             # (edge2D,2) i4 adj elem ids; -1 = boundary
    edge_dxdy: jax.Array            # (edge2D,2) f8 node2-node1, rot radians
    edge_cross_dxdy: jax.Array      # (edge2D,4) f8 (c1-emid,c2-emid), METERS
    edge_up_dn_tri: jax.Array       # (myDim_edge2D,2) i4 MFCT up/down tri; -1 absent

    # --- flat / vertical ---
    nod_in_elem2D: jax.Array        # (Σcounts,) i4 CSR flat, 0-based
    zbar: jax.Array                 # (nl,)   f8 interface depths, m (≤0)
    Z: jax.Array                    # (nl-1,) f8 mid-layer depths

    # --- derived ragged-level masks (built on load; see module docstring) ---
    node_layer_mask: jax.Array      # (nod2D,nl) bool  valid tracer/u/v layers
    node_iface_mask: jax.Array      # (nod2D,nl) bool  valid bvfreq/w/Kv interfaces
    elem_layer_mask: jax.Array      # (elem2D,nl) bool valid element layers (u,v,pgf)
    elem_iface_mask: jax.Array      # (elem2D,nl) bool valid element interfaces (Av)

    # --- static scalar metadata ---
    nod2D: int = _meta(static=True)
    elem2D: int = _meta(static=True)
    edge2D: int = _meta(static=True)
    nl: int = _meta(static=True)
    edge2D_in: int = _meta(static=True)
    myDim_edge2D: int = _meta(static=True)
    ocean_area: float = _meta(static=True)


# Register Mesh as a pytree: array fields are leaves, scalar fields are static.
_MESH_META = [f.name for f in dataclasses.fields(Mesh) if f.metadata.get("static")]
_MESH_DATA = [f.name for f in dataclasses.fields(Mesh) if not f.metadata.get("static")]
tree_util.register_dataclass(Mesh, data_fields=_MESH_DATA, meta_fields=_MESH_META)


# --------------------------------------------------------------------------
# Level masks
# --------------------------------------------------------------------------
def level_masks(ulevels, nlevels, nl: int):
    """Boolean ``[n_entity, nl]`` layer- and interface-validity masks.

    ``ulevels``/``nlevels`` are the 1-based per-entity upper/lower level counts
    (``ulevels_nod2D``/``nlevels_nod2D`` for nodes, ``ulevels``/``nlevels`` for
    elements). Returns ``(layer_mask, iface_mask)`` with, per entity (0-based k):

      * ``layer_mask[k] = (ulevels-1) ≤ k < (nlevels-1)``   (T,S,ρ,p,u,v)
      * ``iface_mask[k] = (ulevels-1) ≤ k < nlevels``       (bvfreq,w,Kv,Av)
    """
    ulevels = jnp.asarray(ulevels).reshape(-1, 1)
    nlevels = jnp.asarray(nlevels).reshape(-1, 1)
    k = jnp.arange(nl, dtype=ulevels.dtype).reshape(1, -1)
    top = k >= (ulevels - 1)
    layer_mask = top & (k < (nlevels - 1))
    iface_mask = top & (k < nlevels)
    return layer_mask, iface_mask


# --------------------------------------------------------------------------
# Orientation invariant (the pi↔CORE2 trap)
# --------------------------------------------------------------------------
def check_cw_orientation(coord_nod2D, elem_nodes, *,
                         cyclic_length_rad: float = CYCLIC_LENGTH_RAD):
    """Verify every triangle is **clockwise (CW)** in the rotated (lon,lat) plane,
    mirroring the C port's ``orient_cw`` (``fesom_mesh.c:430-459``): per element
    ``r = bx*cy - by*cx`` after a cyclic-length wrap of the lon components, where
    ``b = v2-v1``, ``c = v3-v1``. The C swaps v2↔v3 whenever ``r > 0`` so that
    **all** elements end up CW (``r < 0``) *before* any geometry is derived.

    Why this is a hard load-time gate, not a comment: FESOM normalizes pi
    **and** CORE2 to CW, but the derived ``gradient_sca`` / SSH-stiffness sign
    and the antisymmetric edge fluxes assume that convention. CORE2's raw mesh is
    almost entirely CCW (~244654/244659 elements swapped, FRESH_START §4); if a
    mesh export ever reaches the JAX side **without** ``orient_cw`` (a different
    reader path, a partition artifact, a regression), the kernels silently flip
    sign → the historical CORE2 Aleutian-Trench blow-up (FRESH_START §4/§11/§14.8).
    The JAX kernels were validated against the uniformly-CW pi export, so we
    enforce CW on every loaded mesh and fail loudly otherwise.

    Returns the per-element signed cross product ``r`` (numpy; all ``< 0`` when
    CW). Raises ``ValueError`` if any element is CCW or degenerate (``r >= 0``).
    """
    coord = np.asarray(coord_nod2D)
    en = np.asarray(elem_nodes)
    half = 0.5 * cyclic_length_rad
    n0, n1, n2 = en[:, 0], en[:, 1], en[:, 2]
    ax, ay = coord[n0, 0], coord[n0, 1]
    bx = coord[n1, 0] - ax
    by = coord[n1, 1] - ay
    cx = coord[n2, 0] - ax
    cy = coord[n2, 1] - ay
    bx = np.where(bx > half, bx - cyclic_length_rad, bx)
    bx = np.where(bx < -half, bx + cyclic_length_rad, bx)
    cx = np.where(cx > half, cx - cyclic_length_rad, cx)
    cx = np.where(cx < -half, cx + cyclic_length_rad, cx)
    r = bx * cy - by * cx
    bad = int((r >= 0).sum())
    if bad:
        idx = np.nonzero(r >= 0)[0][:10].tolist()
        raise ValueError(
            f"mesh orientation: {bad}/{r.size} triangle(s) are CCW or degenerate "
            f"(r >= 0); FESOM requires all clockwise. The mesh export must run "
            f"orient_cw (fesom_mesh.c:430) BEFORE deriving gradient_sca. "
            f"first bad elem ids: {idx}, max r = {float(r.max()):.3e}"
        )
    return r


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------
def _read_meta_txt(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        key, val = line.split()
        out[key] = float(val)
    return out


def _as_device(a: np.ndarray) -> jax.Array:
    """int-kind → int32, else float64; values byte-identical to the export."""
    if a.dtype.kind in "iu":
        return jnp.asarray(a.astype(np.int32))
    return jnp.asarray(a.astype(np.float64))


def load_mesh(mesh_dir: str | Path = DEFAULT_PI_MESH_DIR,
              *, check_orientation: bool = True) -> Mesh:
    """Load a C-exported mesh directory into a :class:`Mesh` pytree.

    ``mesh_dir`` holds one ``<name>.npy`` per array plus ``meta.txt`` of scalar
    counts (see ``docs/MESH_EXPORT_LAYOUT.md``). Indices are taken as-is
    (already 0-based); the four ragged-level masks are derived on load.

    ``check_orientation`` (default True) asserts every triangle is clockwise via
    :func:`check_cw_orientation` — the pi↔CORE2 trap; set False only to *inspect*
    a deliberately non-CW mesh.
    """
    mesh_dir = Path(mesh_dir)
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"mesh dir not found: {mesh_dir}")

    meta = _read_meta_txt(mesh_dir / "meta.txt")
    nl = int(meta["nl"])

    def load(name: str) -> jax.Array:
        return _as_device(np.load(mesh_dir / f"{name}.npy"))

    arrays: dict[str, jax.Array] = {
        f.name: load(f.name)
        for f in dataclasses.fields(Mesh)
        if not f.metadata.get("static") and not f.name.endswith("_mask")
    }

    if check_orientation:
        check_cw_orientation(arrays["coord_nod2D"], arrays["elem_nodes"])

    node_layer, node_iface = level_masks(
        arrays["ulevels_nod2D"], arrays["nlevels_nod2D"], nl
    )
    elem_layer, elem_iface = level_masks(arrays["ulevels"], arrays["nlevels"], nl)

    return Mesh(
        **arrays,
        node_layer_mask=node_layer,
        node_iface_mask=node_iface,
        elem_layer_mask=elem_layer,
        elem_iface_mask=elem_iface,
        nod2D=int(meta["nod2D"]),
        elem2D=int(meta["elem2D"]),
        edge2D=int(meta["edge2D"]),
        nl=nl,
        edge2D_in=int(meta["edge2D_in"]),
        myDim_edge2D=int(meta["myDim_edge2D"]),
        ocean_area=float(meta["ocean_area"]),
    )


def mesh_field_files() -> Mapping[str, str]:
    """Map each non-derived Mesh array field to its ``<name>.npy`` filename
    (the field name *is* the file stem). Used by the load-verification test to
    diff each loaded array against the raw export."""
    return {
        f.name: f"{f.name}.npy"
        for f in dataclasses.fields(Mesh)
        if not f.metadata.get("static") and not f.name.endswith("_mask")
    }
