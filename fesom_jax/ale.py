"""ALE step (linfs) — substep 13 (Task 2.9).

Literal vectorized port of ``fesom_ale.c`` for the Phase-2 pi config (linfs ALE,
no cavity / no partial cells, single MPI rank):

* :func:`thickness_linfs` — ``hnode_new = hnode`` (``fesom_ale_thickness_linfs``,
  ``fesom_ale.c:10-16``). In **linfs** ``dh/dt = 0``, so the layer thickness is
  *static* for the whole run; the C routine is a ``memcpy``.

* :func:`compute_w` — vertical velocity ``w`` at interfaces
  (``fesom_ale_vert_vel_linfs``, ``fesom_ale.c:77``). The SAME antisymmetric
  edge→node ``(v·dx − u·dy)·helem`` transport-divergence scatter as
  :func:`~fesom_jax.ssh.compute_ssh_rhs` / :func:`~fesom_jax.ssh.compute_hbar`,
  but kept **per level** (not summed over the column) and driven by the **new**
  (post-:func:`~fesom_jax.momentum.update_vel`) velocity ``uv``. Then a reverse
  (bottom→top) cumulative sum over levels — the vertical integral of the
  horizontal divergence up from the no-flux bottom interface ``w[nzmax]=0`` — and
  finally ``÷ area(n, nz)`` → m/s.

  ⚠️ **Trap:** stage 3 divides by ``mesh.area`` (the *upper-edge scalar* CV area),
  **NOT** ``areasvol`` (which :func:`~fesom_jax.ssh.compute_hbar` used) — a
  different array. The C guards ``if (area > 0)``; we mirror it with a safe divide
  so masked / bottom lanes (where ``w = 0``) and the backward pass stay finite.
  Like ``hbar``, the ``÷area`` (``~1e9–1e12 m²``) divides the near-cancelling
  divergence's amplified absolute error back down, so ``w`` matches the dump
  tightly despite the loose per-level scatter floor.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from . import ops
from .config import DT_DEFAULT, USE_WSPLIT, WSPLIT_MAXCFL
from .mesh import Mesh


class AleConfig(NamedTuple):
    """Static ALE (vertical-coordinate) config — the Phase-9a ``which_ALE='zstar'``
    seam (locked decision 2). Closed over the step / passed as a ``static_argname``,
    exactly like :class:`~fesom_jax.gm.GMConfig` / :class:`~fesom_jax.kpp.KppConfig`
    / ``IceConfig``. Hashable (all fields are Python scalars/bools); carries **no**
    differentiable leaves.

    **The presence of an ``AleConfig`` ⇒ zstar; ``ale_cfg=None`` ⇒ the linfs path**
    (byte-identical to the pre-Phase-9 model). This mirrors the C runtime switch
    ``fesom_ale_mode_init`` (``fesom_ale.c:13-33``): ``FESOM_ALE`` is ``linfs`` (default)
    or ``zstar``; **any other mode aborts** — zlevel / local-zstar / partial cells /
    floating ice / cavities are out of scope (no reference run ⇒ no oracle). We port that
    abort as :meth:`validate` (a runtime guard at the step seam, the C's
    ``exit(1)`` parity), since ``typing.NamedTuple`` forbids a ``__new__`` override.

    Derived (the C's two mode globals, ``fesom_ale.c:31-32``):

    * :attr:`use_virt_salt` = ``not zstar`` (zstar ⇒ ``False``: real freshwater/salt
      fluxes replace the virtual-salt flux);
    * :attr:`is_nonlinfs` = ``1.0 if zstar else 0.0`` (the float multiplier the surface
      BCs / forcing terms gate on).
    """

    zstar: bool = True            # which_ALE = 'zstar' (the only supported non-linfs mode)

    @property
    def use_virt_salt(self) -> bool:
        """``fesom_use_virt_salt = !fesom_ale_zstar`` (``fesom_ale.c:31``)."""
        return not self.zstar

    @property
    def is_nonlinfs(self) -> float:
        """``fesom_is_nonlinfs = fesom_ale_zstar ? 1.0 : 0.0`` (``fesom_ale.c:32``)."""
        return 1.0 if self.zstar else 0.0

    def validate(self) -> "AleConfig":
        """Raise on an unsupported mode (the C ``fesom_ale_mode_init`` abort parity,
        ``fesom_ale.c:25-30``). Called at the step seam whenever ``ale_cfg is not None``;
        returns ``self`` so it can wrap an expression. ``ale_cfg=None`` is the linfs path
        and never reaches here."""
        if not self.zstar:
            raise ValueError(
                "AleConfig only supports which_ALE='zstar' (zstar=True); zlevel / "
                "local-zstar / partial cells are out of scope (no C reference run). "
                "Use ale_cfg=None for the linfs path. (fesom_ale.c:25-30 abort parity)")
        return self


def _shift_down_zero(x):
    """``out[..., k] = x[..., k-1]``, **zero**-padded at ``k=0`` (no edge-replicate)."""
    return jnp.concatenate([jnp.zeros_like(x[..., :1]), x[..., :-1]], axis=-1)


def thickness_linfs(hnode):
    """``hnode_new = hnode`` (substep 13, linfs; ``fesom_ale.c:10-16``).

    Linfs has ``dh/dt = 0`` so the per-vertex layer thickness never changes; the C
    ``fesom_ale_thickness_linfs`` is a ``memcpy(hnode_new, hnode)``. Returned as a
    distinct ``[nod2D, nl]`` array for the ``State.hnode_new`` slot (consumed by
    the tracer ALE reconstruction in Task 2.10). The substep-13 ``hnode_new`` node
    dump should therefore equal the static ``hnode``.
    """
    return jnp.asarray(hnode)


def commit_thickness(mesh: Mesh, hnode_new):
    """Commit the ALE thickness (substep 16, ``fesom_ale_commit_thickness``,
    ``fesom_ale.c:18``). Returns ``(hnode, helem)``:

    * ``hnode = hnode_new`` (a copy; linfs ⇒ unchanged).
    * ``helem`` = the per-element vertex mean ``(hnode[n0]+hnode[n1]+hnode[n2])/3``,
      masked to ``elem_layer_mask``. In linfs this reproduces the static reference
      thickness (all 3 vertices share the same full-cell ``hnode``).

    The substep-16 node dump records ``hnode`` (bit-for-bit the static thickness, as
    ``hnode_new`` was at substep 13)."""
    hnode = jnp.asarray(hnode_new)
    h3 = ops.gather_nodes_to_elem(hnode, mesh.elem_nodes)   # (elem2D, 3, nl)
    helem = (h3[:, 0] + h3[:, 1] + h3[:, 2]) / 3.0
    helem = ops.mask_below_bottom(helem, mesh.elem_layer_mask)
    return hnode, helem


# ===========================================================================
# zstar (Phase 9a) — thickness machinery: init + derived live geometry (JZ.1)
# ===========================================================================
# Decision D1: zbar_3d_n / Z_3d_n are NOT carried State — they are recomputed from
# the prognostic `hnode` by `live_geometry`, replicating the C commit's bottom→top
# reconstruction (`fesom_ale_update_thickness_zstar`, fesom_ale.c:228-240). Decision
# D5 (AD): every below-stretch / below-bottom lane is filled with NOMINAL spacing
# (the strictly-decreasing `mesh.zbar`), never the C's 0-padding (`mesh.zbar_3d_n`
# below bottom is 0 ⇒ dz=0 ⇒ a dense-JAX inf factory; verified all 3140 pi nodes).


def _stretch_mask(mesh: Mesh):
    """Static boolean ``[nod2D, nl]``: the zstar **stretch range** — layers/interfaces
    ``nz ≤ nlevels_nod2D_min(n) − 3`` (0-based), i.e. ``nz < min_f − 2``. These are the
    layers above the shallowest neighbouring bottom; only they stretch with the SSH
    change (the bottom-intersecting + bottom layers keep nominal spacing). Derives
    purely from the static integer level array ⇒ precomputable, traceable as a constant
    (`fesom_ale.c:70,78` ``nz = 1..min-2`` 1-based)."""
    nl = mesh.zbar.shape[0]
    k = jnp.arange(nl)[None, :]                              # (1, nl) 0-based level
    min_f = jnp.asarray(mesh.nlevels_nod2D_min)[:, None]     # (nod2D,1) 1-based K_v⁻
    return k < (min_f - 2)


def init_thickness_zstar(mesh: Mesh, hbar, hbar_old, *, alpha: float = 1.0,
                         dt: float = DT_DEFAULT):
    """zstar thickness initialisation (``fesom_ale_init_thickness_zstar``,
    ``fesom_ale.c:45-146``). Returns ``(hnode, helem, eta_n, ssh_rhs_old)``.

    * **Stretch layers** (``nz < min_f−2``): ``hnode = (zbar[nz]−zbar[nz+1])·(1 +
      hbar/dd)`` with ``dd = zbar[0] − zbar[min_f−2]`` (the stretchable column depth,
      ``fesom_ale.c:75-82``). **Non-stretch layers** (bottom-intersecting + bottom):
      nominal ``zbar[nz]−zbar[nz+1]`` — in full-cell (``use_partial_cell=.false.``)
      this equals the C's ``bottom_node_thickness``, so the whole column is
      ``(zbar diff)·(1 + (hbar/dd)·stretch)`` with no separate bottom field.
    * **helem** = vertex mean of ``hnode`` (full mean; full-cell ⇒ matches the C's
      ``update_thickness_zstar`` stretch-mean + ``bottom_elem_thickness`` to ≤1 ulp).
    * ``eta_n = α·hbar_old + (1−α)·hbar`` — ⚠️ **REVERSED weights vs the per-step
      blend** ``α·hbar + (1−α)·hbar_old`` (C lesson #7, ``fesom_ale.c:62-63``).
    * ``ssh_rhs_old = (hbar − hbar_old)·areasvol[:,top]/dt`` (``fesom_ale.c:58-59``).

    **Cold start (``hbar=hbar_old=0``)** ⇒ ``hnode``/``helem`` = nominal (linfs), ``eta_n
    = ssh_rhs_old = 0`` — the zstar init is bit-for-bit the linfs rest init (the free Z1
    degeneracy gate). AD-safe: ``dd`` gets the double-``where`` guard (nominal where 0)."""
    zbar = mesh.zbar                                          # (nl,)
    nl = zbar.shape[0]
    # nominal layer thickness dz[nz] = zbar[nz]-zbar[nz+1] > 0, padded to nl (last 0).
    dz = jnp.zeros((nl,)).at[:-1].set(zbar[:-1] - zbar[1:])   # (nl,)
    min_f = jnp.asarray(mesh.nlevels_nod2D_min)               # (nod2D,) 1-based
    # dd = zbar[0] - zbar[min_f-2] (per node); guard against dd==0 (double-where, AD).
    anchor_dep = zbar[min_f - 2]                              # (nod2D,) static gather
    dd = zbar[0] - anchor_dep                                 # (nod2D,)
    dd_safe = jnp.where(dd != 0.0, dd, 1.0)
    stretch = _stretch_mask(mesh)                             # (nod2D, nl)
    ratio = jnp.where(dd != 0.0, hbar / dd_safe, 0.0)[:, None]   # (nod2D,1)
    factor = 1.0 + jnp.where(stretch, ratio, 0.0)            # 1 off the stretch range
    hnode = jnp.where(mesh.node_layer_mask, dz[None, :] * factor, 0.0)

    hnode, helem = commit_thickness(mesh, hnode)            # vertex-mean helem (full cell)
    eta_n = alpha * hbar_old + (1.0 - alpha) * hbar         # ⚠️ reversed weights (lesson #7)
    ssh_rhs_old = (hbar - hbar_old) * mesh.areasvol[:, 0] / dt
    return hnode, helem, eta_n, ssh_rhs_old


def live_geometry(mesh: Mesh, hnode):
    """Derived per-node interface/mid-layer depths ``(zbar_3d_n, Z_3d_n)`` ``[nod2D, nl]``
    from the carried ``hnode`` (decision D1) — the zstar commit's bottom→top
    reconstruction (``fesom_ale_update_thickness_zstar``, ``fesom_ale.c:228-240``) as a
    PURE function of state (no carried geometry):

        zbar_3d_n[nz] = zbar_3d_n[nz+1] + hnode[nz]   (nz = min_f−3 .. 0, bottom→top)
        Z_3d_n[nz]    = zbar_3d_n[nz+1] + hnode[nz]/2

    anchored at the nominal interface ``min_f−2`` (which keeps static spacing). Vectorized
    as ``zbar_3d_n[nz] = anchor + Σ_{j≥nz} hnode_stretch[j]`` (a reverse cumsum over the
    stretch range). At step 1 (nominal hnode) this is bit-identical to the C recurrence
    (it telescopes); at step ≥2 the cumsum reassociates ~1e-10 vs the C's sequential
    ``zbar[nz]=zbar[nz+1]+hnode[nz]`` — harmless for the gates (verified: switching to the
    exact recurrence left the JZ.7 step-1 pgf byte-for-byte identical). If the JZ.7 multi-step
    gate ever surfaces a geometry tail at step ≥2, swap this for a ``lax.scan`` recurrence.

    **AD-safety (D5):** the non-stretch / below-bottom lanes are filled with the
    strictly-decreasing **nominal ``zbar``** (positive spacing everywhere), NOT the
    stored ``mesh.zbar_3d_n`` (0-padded below bottom ⇒ ``dz=0`` ⇒ inf on divide).
    In the wet range ``mesh.zbar_3d_n == zbar`` (non-cavity full-cell), so the output
    matches the static geometry there; consumers mask the below-bottom lanes anyway."""
    zbar = mesh.zbar                                          # (nl,)
    nl = zbar.shape[0]
    min_f = jnp.asarray(mesh.nlevels_nod2D_min)              # (nod2D,) 1-based
    stretch = _stretch_mask(mesh)                            # (nod2D, nl)
    anchor = zbar[min_f - 2][:, None]                        # (nod2D,1) nominal anchor depth

    # zbar: bottom→top reverse cumsum of the stretch-layer thicknesses, on the anchor.
    h_str = jnp.where(stretch, hnode, 0.0)
    revcum = lax.cumsum(h_str, axis=1, reverse=True)        # Σ_{j≥nz} h_str[j]
    zbar_nom = jnp.broadcast_to(zbar[None, :], (mesh.nod2D, nl))   # AD-safe nominal (dz>0)
    zbar_3d_n = jnp.where(stretch, anchor + revcum, zbar_nom)

    # Z mid-layer: zbar_3d_n[nz+1] + hnode[nz]/2 in the stretch range; nominal mesh.Z below.
    zbar_below = jnp.concatenate([zbar_3d_n[:, 1:], zbar_3d_n[:, -1:]], axis=1)  # [nz+1]
    Zp = jnp.concatenate([mesh.Z, mesh.Z[-1:]])             # (nl,) nominal mid-depths, padded
    Z_3d_n = jnp.where(stretch, zbar_below + 0.5 * hnode, Zp[None, :])
    return zbar_3d_n, Z_3d_n


def vert_vel_zstar_distribute(mesh: Mesh, w, hnode, zbar_3d_n, hbar, hbar_old,
                              water_flux, *, dt: float = DT_DEFAULT):
    """zstar branch of vert_vel (substep 13 add-on, ``fesom_ale_vert_vel_zstar_distribute``,
    ``fesom_ale.c:162-201``). Distributes the per-step SSH change ``(hbar−hbar_old)``
    proportionally over the **stretched** part of the column, ON TOP of the shared
    divergence-built ``w`` (:func:`compute_w`), and produces the new layer thickness
    ``hnode_new``. Returns ``(w, hnode_new)``.

    Per non-cavity node (``zbar_3d_n``/``hnode`` are the **pre-commit/carried** live
    geometry from ``st.hnode``; ``hbar``/``hbar_old`` the post-:func:`~fesom_jax.ssh.compute_hbar`
    values):

        dd1 = zbar_3d_n[min_f−2]                 (anchor interface, first non-stretch)
        dd  = (hbar − hbar_old) / (zbar_3d_n[0] − dd1)
        nz < min_f−2:  w[nz]        −= (zbar_3d_n[nz] − dd1)·dd/dt     (vertically-INTEGRATED,
                       hnode_new[nz] = hnode[nz] + (zbar_3d_n[nz]−zbar_3d_n[nz+1])·dd   NOT per-layer)
        w[0] −= water_flux                        (real-volume surface continuity BC)

    The ``w`` correction is the bottom→top integral ``Σ dh/dt`` (depth-weighted), **not**
    the per-layer ``h·dd/dt``; below the stretch range (``nz ≥ min_f−2``) ``hnode_new`` keeps
    the nominal ``hnode`` (bottom layers don't stretch). At **cold start** (``hbar=hbar_old``)
    ``dd=0`` ⇒ ``w`` unchanged and ``hnode_new=hnode`` (the degeneracy gate). AD-safe: the
    stretchable depth gets the double-``where`` guard; live ``zbar_3d_n`` is strictly
    decreasing (positive spacing) so every divide/subtract is finite.

    The caller (``step.py``) exchanges BOTH ``w`` and ``hnode_new`` (the C's zstar-only
    ``exchange_nod(hnode_new)``, ``fesom_ale.c:157`` — its halo feeds the Z1 commit)."""
    min_f = jnp.asarray(mesh.nlevels_nod2D_min)             # (nod2D,) 1-based K_v⁻
    nocav = (mesh.ulevels_nod2D == 1)                       # non-cavity node mask
    # anchor interface dd1 = zbar_3d_n[n, min_f−2] (the first non-stretch interface, nominal)
    dd1 = jnp.take_along_axis(zbar_3d_n, (min_f - 2)[:, None], axis=1)[:, 0]   # (nod2D,)
    col = zbar_3d_n[:, 0] - dd1                             # stretchable column depth (>0)
    col_safe = jnp.where(col != 0.0, col, 1.0)
    dd = jnp.where(col != 0.0, (hbar - hbar_old) / col_safe, 0.0)              # (nod2D,)
    dddt = dd / dt
    stretch = _stretch_mask(mesh) & nocav[:, None]         # (nod2D,nl) — the C loop range nz<min_f−2

    # Wvel correction: subtract the vertically-integrated (zbar_3d_n − dd1)·dd/dt on the stretch range.
    w = w - jnp.where(stretch, (zbar_3d_n - dd1[:, None]) * dddt[:, None], 0.0)
    # hnode_new: stretch layers grow by (Δzbar_3d_n)·dd; non-stretch (+ below bottom) stay nominal hnode.
    zbar_below = jnp.concatenate([zbar_3d_n[:, 1:], zbar_3d_n[:, -1:]], axis=1)   # [nz+1]
    hnode_new = jnp.where(stretch, hnode + (zbar_3d_n - zbar_below) * dd[:, None], hnode)
    # surface freshwater BC w[0] −= wf (non-cavity; independent of the stretch range, fesom_ale.c:189).
    # water_flux=None ⇒ zero flux (the C's `water_flux ? water_flux[n] : 0.0` startup arm).
    if water_flux is not None:
        w = w.at[:, 0].add(-jnp.where(nocav, water_flux, 0.0))
    return w, hnode_new


def compute_w(mesh: Mesh, uv, helem):
    """Vertical velocity ``w`` ``[nod2D, nl]`` at interfaces (substep 13).

    Mirror of ``fesom_ale_vert_vel_linfs`` (``fesom_ale.c:77-187``), three stages:

    1. **Per-level transport-divergence scatter.** For edge ``ed`` with adjacent
       cells ``(el1, el2)`` and cross vectors ``(dx1,dy1)`` / ``(dx2,dy2)``,
       ``c[nz] = (v[el1]·dx1 − u[el1]·dy1)·h[el1] − (v[el2]·dx2 − u[el2]·dy2)·h[el2]``
       per level ``nz``, scattered ``w[n1,nz] += c``, ``w[n2,nz] −= c``. This is
       :func:`~fesom_jax.ssh.compute_ssh_rhs`'s flux with ``alpha=1`` and no
       AB-velocity, **kept per level**. The per-element ``elem_layer_mask`` keeps
       each cell's contribution in ``[ulevels-1, nlevels-1)`` — ⊆ the node's range,
       so the node's bottom interface ``w[nzmax]`` receives nothing and stays 0
       (the no-flux BC).

    2. **Reverse (bottom→top) cumulative sum** over the level axis:
       ``w[nz] = Σ_{j≥nz} div[j]`` (the vertical integral of the divergence). Since
       the scatter is already 0 at and below ``nzmax``, a full suffix-sum equals
       the C loop ``for nz=nzmax-1..nzmin: w[nz] += w[nz+1]`` while preserving
       ``w[nzmax] = 0``.

    3. **``÷ area(n, nz)`` → m/s** (``mesh.area``, *not* ``areasvol``). The C's
       ``if (area > 0)`` guard becomes a safe divide ``where(area>0, area, 1)``:
       the only nonzero ``w`` lanes are ``[nzmin, nzmax)`` where ``area > 0``, so
       this is exact and AD-finite. A final ``node_iface_mask`` keeps ``w`` on the
       valid interface range ``[nzmin, nzmax]`` (zeroing the suffix-sum's spill
       above a cavity node's ``nzmin``; a no-op for non-cavity pi).
    """
    et = mesh.edge_tri
    el1, el2 = et[:, 0], et[:, 1]
    has1, has2 = el1 >= 0, el2 >= 0
    el1s = jnp.where(has1, el1, 0)
    el2s = jnp.where(has2, el2, 0)
    cross = mesh.edge_cross_dxdy
    U, V = uv[:, :, 0], uv[:, :, 1]
    lm = mesh.elem_layer_mask

    def cterm(els, has, dxcol, dycol, sign):
        dx = cross[:, dxcol : dxcol + 1]                  # (edge,1)
        dy = cross[:, dycol : dycol + 1]
        u = U[els]; v = V[els]; h = helem[els]            # (edge,nl)
        m = lm[els] & has[:, None]
        term = jnp.where(m, (v * dx - u * dy) * h, 0.0)
        return sign * term                                # (edge,nl) — per level

    # Stage 1: per-level antisymmetric edge→node divergence scatter.
    c = cterm(el1s, has1, 0, 1, 1.0) + cterm(el2s, has2, 2, 3, -1.0)   # (edge,nl)
    vals = jnp.stack([c, -c], axis=1)                     # (edge,2,nl)
    div = ops.scatter_add(vals, mesh.edges, mesh.nod2D)   # (nod2D,nl)

    # Stage 2: reverse (bottom→top) cumulative sum over levels.
    w = lax.cumsum(div, axis=1, reverse=True)

    # Stage 3: ÷ area (NOT areasvol); safe divide mirrors the C `if (area > 0)`.
    safe_area = jnp.where(mesh.area > 0.0, mesh.area, 1.0)
    w = w / safe_area
    return ops.mask_below_bottom(w, mesh.node_iface_mask)


def compute_cfl_z(mesh: Mesh, w, hnode_new, *, dt=DT_DEFAULT):
    """Vertical CFL number at interfaces ``cfl_z`` ``[nod2D, nl]`` (``fesom_ale_compute_cflz``,
    ``fesom_ale.c:204``). Each interface accumulates ``|w|·dt/h`` from the layers above and
    below it::

        cfl_z[i] = |w[i]|·dt·( [layer i  valid]/h[i]  +  [layer i-1 valid]/h[i-1] )

    so the surface/bottom interfaces get a single term and interior interfaces get both
    (matching the C ``+=`` accumulation over each layer's top/bottom faces). ``h`` is the
    layer thickness ``hnode_new``; the C skips ``h<=0`` layers, mirrored by the layer mask.

    Only consumed by :func:`compute_wvel_split` when ``use_wsplit`` is on (off for the pi
    reference config), but computed every step as in the C so ``State.cfl_z`` is populated."""
    h = hnode_new
    safe_h = jnp.where(h > 0.0, h, 1.0)
    inv_h = jnp.where(mesh.node_layer_mask & (h > 0.0), dt / safe_h, 0.0)  # layer field
    below = inv_h                       # layer i (below interface i)
    above = _shift_down_zero(inv_h)     # layer i-1 (above interface i); 0 at i=0
    cfl = jnp.abs(w) * (below + above)
    return ops.mask_below_bottom(cfl, mesh.node_iface_mask)


def compute_wvel_split(mesh: Mesh, w, cfl_z, *, use_wsplit=USE_WSPLIT,
                       maxcfl=WSPLIT_MAXCFL):
    """Split the vertical velocity ``w`` into explicit ``w_e`` and implicit ``w_i`` parts
    (``fesom_ale_compute_wvel_split``, ``fesom_ale.c:241``). Returns ``(w_e, w_i)``.

    Where the vertical CFL exceeds ``maxcfl`` the excess is moved to the implicit part so
    the explicit advection stays CFL-stable::

        dd  = max(cfl_z − maxcfl, 0) / maxcfl
        w_e = w / (1 + dd),   w_i = w · dd / (1 + dd)      (cfl_z > maxcfl)
        w_e = w,              w_i = 0                        (otherwise)

    **``use_wsplit`` is 0 in the pi (and CORE2 dt=1800) reference config**
    (``fesom_constants.h:56`` — the split seeded a Fortran day-92 blow-up), so this is the
    identity ``(w, 0)`` on the dump-matching path; the active branch is exercised by the
    synthetic high-CFL test. AD-safe: ``dd ≥ 0`` ⇒ ``1+dd ≥ 1`` (no zero divide), and the
    static ``maxcfl = 1.0`` floors the ``1/maxcfl``.

    ⚠️ ``w_i`` (the implicit part) feeds the ``impl_vert_visc`` advective tridiagonal terms,
    which the Phase-2 kernel drops under the ``w_i=0`` (``use_wsplit=0``) simplification —
    re-enabling those terms is a Phase-5/CORE2 item, needed only when ``use_wsplit`` is on."""
    if not use_wsplit:
        return jnp.asarray(w), jnp.zeros_like(w)
    inv_maxcfl = 1.0 / max(maxcfl, 1e-12)            # static (maxcfl is a Python float)
    dd = jnp.maximum(cfl_z - maxcfl, 0.0) * inv_maxcfl
    inv_1_dd = 1.0 / (1.0 + dd)                      # dd ≥ 0 ⇒ always finite
    split = cfl_z > maxcfl
    w_e = jnp.where(split, w * inv_1_dd, w)
    w_i = jnp.where(split, w * dd * inv_1_dd, 0.0)
    w_e = ops.mask_below_bottom(w_e, mesh.node_iface_mask)
    w_i = ops.mask_below_bottom(w_i, mesh.node_iface_mask)
    return w_e, w_i
