"""Fortran FESOM2 restart -> fesom-jax :class:`~fesom_jax.state.State`.

FESOM2 (``io_restart.F90``, ``ini_ocean_io`` / ``ini_ice_io``) restarts a run from two groups of
NetCDF variables per year, ``fesom.<YYYY>.oce.restart`` and ``fesom.<YYYY>.ice.restart`` (a file
each, or a directory with one file per variable):

  ocean  ssh, hbar, ssh_rhs_old, hnode, u, v, urhs_AB, vrhs_AB, temp, salt, temp_AB, salt_AB,
         temp_M1, salt_M1 (optional), w, w_expl, w_impl, tke (cvmix TKE, optional)
  ice    area, hice, hsnow, uice, vice, ice_temp, ice_albedo (optional)

Not stored: the mEVP stresses (FESOM restarts them at zero), Kv/Av, density, pressure and the other
per-step diagnostics. :func:`state_from_fortran` maps the stored fields onto a reference State on
the same mesh (a fesom-jax cold start), wet lanes from the restart and dry lanes from the
reference, and rebuilds what FESOM rebuilds at a restart (``restart_thickness_ale``: ``helem`` as
the element mean of ``hnode``; ``hbar_old`` = ``hbar``; ``hnode_new`` = ``hnode``). The fields a
fesom-jax step reads before it writes them (``scripts/fortran_restart/carried_fields.py``) are
exactly the ones that need a value; the rest the first step recomputes.

FESOM2 main (2.8, e.g. 0aaa2d73, the jaxcpl ocean) restarts more: ``d_eta`` (the CG warm start), the mEVP stresses
``sigma11/12/22``, the Newton seed ``t_skin`` [degC] next to ``ice_temp`` [K], and ``hnode_new``. When a restart has
them they are taken from it (``d_eta``, the stresses, ``t_skin``); a FESOM 2.7 restart without them converts exactly
as before (``d_eta`` from the reference, stresses zero, ``t_skin`` from ``ice_temp``). With cavities the masks are the
``ulevels``-aware wet lanes, so the levels inside the ice shelves keep the reference values (FESOM stores 0 there).

Velocities stay in the model's (rotated) frame: FESOM writes ``u``/``v``/``uice``/``vice`` from
its internal arrays, which are the frame fesom-jax computes in.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np

#: State field <- restart variable (node or element, layers or interfaces), the 2-D/3-D
#: prognostics FESOM2 restarts from
OCE_MAP = {
    "eta_n": "ssh", "hbar": "hbar", "ssh_rhs_old": "ssh_rhs_old", "hnode": "hnode",
    "T": "temp", "S": "salt", "w": "w", "w_e": "w_expl", "w_i": "w_impl", "tke": "tke",
}
ICE_MAP = {"a_ice": "area", "m_ice": "hice", "m_snow": "hsnow", "u_ice": "uice",
           "v_ice": "vice", "t_skin": "ice_temp"}
#: fields at layer midpoints (the last of the nl slots is unused) / at the nl interfaces
LAYER_FIELDS = ("T", "S", "T_old", "S_old", "hnode", "uv", "uv_rhsAB")
INTERFACE_FIELDS = ("w", "w_e", "w_i", "tke")

#: Where every State field of a converted restart comes from, and whether one fesom-jax step
#: reads it (measured on CORE2 by ``scripts/fortran_restart/carried_fields.py``, 2026-10-10:
#: each field perturbed, one step, outputs compared). Every field the step reads is either
#: restarted, rebuilt like FESOM rebuilds it, or set to the value FESOM itself restarts with.
SOURCES = {
    # read by the step, from the Fortran restart
    "T": "restart", "S": "restart", "T_old": "restart", "S_old": "restart", "uv": "restart",
    "uv_rhsAB": "restart", "w_e": "restart", "eta_n": "restart", "hnode": "restart",
    "hbar": "restart", "tke": "restart", "a_ice": "restart", "m_ice": "restart",
    "m_snow": "restart", "u_ice": "restart", "v_ice": "restart",
    # read, rebuilt as FESOM's restart_thickness_ale / ale.commit_thickness do
    "helem": "derived",
    # read, at the value FESOM restarts with: the CG warm start (FESOM's d_eta starts at 0), the
    # ice skin temperature (not in the restart; FESOM keeps its initial value), the mEVP stresses
    "d_eta": "reference", "t_skin": "reference", "sigma11": "zero", "sigma12": "zero",
    "sigma22": "zero",
    # read only into themselves (accumulators with no effect on the other outputs)
    "del_ttf": "reference", "uvnode_rhs": "reference",
    # not read: the step recomputes them (restarted or rebuilt anyway where FESOM has them)
    "w": "restart", "w_i": "restart", "ssh_rhs_old": "restart", "hnode_new": "derived",
    "hbar_old": "derived", "uv_rhs": "reference", "uvnode": "reference", "cfl_z": "reference",
    "ssh_rhs": "reference", "density": "reference", "hpressure": "reference",
    "bvfreq": "reference", "Kv": "reference", "Av": "reference", "pgf_x": "reference",
    "pgf_y": "reference",
}
# The optional State fields of other run modes, None in a converted restart (each mode's driver builds them: the
# single-precision rounding carry starts at 0 in precision.cast_state, ice_ave_steps > 1 in ice_step.init_ice_avg,
# the split-explicit SSH, REcoM, the coupled ocean's mode F).
SOURCES.update({k: "optional" for k in (
    "carry_T", "carry_S", "ice_steps_since_upd", "ice_update", "srfoce_temp", "srfoce_salt", "srfoce_ssh",
    "srfoce_u", "srfoce_v", "ice_flx_h", "ice_flx_fw", "ice_evaporation", "ice_sublimation", "ice_thdgr",
    "ice_thdgrsn", "ice_real_salt_flux", "ice_a_old", "se_eta", "se_ubt", "se_sbar_old", "bgc", "bgc_old",
    "bgc_carry", "bgc_steps", "bgc_uvh", "bgc_we", "bgc_wi", "bgc_kv", "bgc_bc", "bgc_h0", "bgc_benthos",
    "bgc_alk_surf", "bgc_carb", "bgc_mstep", "bgc_diag", "bgc_budget", "cpl_ice_temp", "cpl_ice_alb", "cpl_apnd",
    "cpl_hpnd", "cpl_ipnd")})
#: the fields one step reads (the same measurement)
STEP_READS = ("T", "S", "T_old", "S_old", "del_ttf", "uv", "uv_rhsAB", "uvnode_rhs", "w_e",
              "eta_n", "d_eta", "hnode", "helem", "hbar", "tke", "a_ice", "m_ice", "m_snow",
              "u_ice", "v_ice", "t_skin", "sigma11", "sigma12", "sigma22")


class FortranRestart:
    """One restart group (``oce`` or ``ice``): a NetCDF file, or a directory of per-variable
    files. :meth:`read` returns the LAST time record as ``(n_entities,)`` or
    ``(n_entities, n_levels)`` float64, whatever order the file stores the axes in."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._files = {}
        if self.path.is_dir():
            for f in sorted(self.path.glob("*.nc")):
                self._files.update({v: f for v in self._variables(f)})
        else:
            self._files = {v: self.path for v in self._variables(self.path)}

    @staticmethod
    def _variables(f):
        import netCDF4
        with netCDF4.Dataset(f) as ds:
            return [v for v, x in ds.variables.items() if x.ndim >= 1 and v not in ds.dimensions]

    @property
    def variables(self):
        return sorted(self._files)

    def has(self, name) -> bool:
        return name in self._files

    def read(self, name, n_entities: int) -> np.ndarray:
        """The last record of ``name`` with the entity axis (size ``n_entities``) first."""
        import netCDF4
        with netCDF4.Dataset(self._files[name]) as ds:
            v = ds.variables[name]
            v.set_auto_mask(False)
            dims = list(v.dimensions)
            a = np.asarray(v[:], dtype=np.float64)
        tdim = [i for i, d in enumerate(dims) if d.lower() in ("time", "t", "nrec")]
        if tdim:
            a = np.take(a, -1, axis=tdim[0])
            dims.pop(tdim[0])
        # the entity axis by its name (FESOM2: node / elem), by its size only as a fallback
        ent = [i for i, d in enumerate(dims) if d.lower().startswith(("node", "nod", "elem"))
               and a.shape[i] == n_entities]
        if not ent:
            ent = [i for i, s in enumerate(a.shape) if s == n_entities]
        if len(ent) != 1:
            raise ValueError(f"{name}: no unique axis of size {n_entities} in shape {a.shape} "
                             f"(dims {dims})")
        return np.moveaxis(a, ent[0], 0)


def _masks(mesh):
    """Wet masks (host numpy): node layers / interfaces and element layers, from the 1-based
    FESOM level counts (``ulevels`` = first wet level, ``nlevels`` = bottom interface)."""
    nl = int(mesh.nl)
    k = np.arange(nl)[None, :]
    un, nn = (np.asarray(mesh.ulevels_nod2D).ravel()[:, None],
              np.asarray(mesh.nlevels_nod2D).ravel()[:, None])
    ue, ne = (np.asarray(mesh.ulevels).ravel()[:, None], np.asarray(mesh.nlevels).ravel()[:, None])
    return {"nod_layer": (k >= un - 1) & (k < nn - 1),
            "nod_iface": (k >= un - 1) & (k <= nn - 1),
            "elem_layer": (k >= ue - 1) & (k < ne - 1)}


def _place(ref, val, mask):
    """``ref`` with ``val`` written into the ``mask`` lanes; ``val`` may carry fewer levels than
    ``ref`` (a layer variable with nl-1 levels in the file)."""
    out = np.array(ref, dtype=np.float64, copy=True)
    if out.ndim == 1:
        return np.where(mask, val, out) if mask is not None else np.asarray(val, np.float64)
    nk = min(out.shape[1], val.shape[1])
    m = mask[:, :nk]
    out[:, :nk] = np.where(m, val[:, :nk], out[:, :nk])
    return out


def state_from_fortran(mesh, oce: FortranRestart, ice: FortranRestart | None, ref):
    """A dense host :class:`State` from a FESOM2 restart; ``ref`` = a State on the same mesh
    (a cold start) that supplies the dry lanes and every field FESOM does not restart. Returns
    ``(state, report)``; ``report`` lists what came from where."""
    import dataclasses
    n, e, nl = int(mesh.nod2D), int(mesh.elem2D), int(mesh.nl)
    M = _masks(mesh)
    ref = {f.name: (None if getattr(ref, f.name) is None else np.asarray(getattr(ref, f.name)))
           for f in dataclasses.fields(ref)}           # optional State fields (None) stay None
    out = dict(ref)
    rep = {"restart": [], "derived": [], "reference": [], "zeroed": [], "missing": []}

    def node(name, var, iface=False):
        a = oce.read(var, n)
        out[name] = _place(ref[name], a, M["nod_iface" if iface else "nod_layer"]
                           if a.ndim == 2 else None)
        rep["restart"].append(f"{name}<-{var}")

    for name, var in OCE_MAP.items():
        if not oce.has(var):
            rep["missing"].append(var)
            continue
        node(name, var, iface=name in INTERFACE_FIELDS)
    # the AB2 tracer slot (valuesold): FESOM's temp_M1 / salt_M1; without them, the current values
    for name, var, base in (("T_old", "temp_M1", "T"), ("S_old", "salt_M1", "S")):
        if oce.has(var):
            node(name, var)
        else:
            out[name] = out[base].copy()
            rep["derived"].append(f"{name}={base} ({var} absent)")
    for name, (vu, vv) in (("uv", ("u", "v")), ("uv_rhsAB", ("urhs_AB", "vrhs_AB"))):
        u, v = oce.read(vu, e), oce.read(vv, e)
        val = np.stack([u, v], axis=-1)                                   # (e, nk, 2)
        nk = min(nl, val.shape[1])
        o = np.array(ref[name], dtype=np.float64, copy=True)
        m = M["elem_layer"][:, :nk, None]
        o[:, :nk] = np.where(m, val[:, :nk], o[:, :nk])
        out[name] = o
        rep["restart"].append(f"{name}<-{vu},{vv}")
    # what FESOM rebuilds after reading a restart (restart_thickness_ale) or at each step start
    hn = out["hnode"]
    en = np.asarray(mesh.elem_nodes)
    if en.shape[0] == 3 and en.shape[1] != 3:
        en = en.T
    helem = hn[en].mean(axis=1)                                            # (e, nl)
    out["helem"] = np.where(M["elem_layer"], helem, ref["helem"])
    out["hnode_new"] = hn.copy()
    out["hbar_old"] = out["hbar"].copy()
    rep["derived"] += ["helem = element mean of hnode", "hnode_new = hnode", "hbar_old = hbar"]
    if ice is not None:
        imap = dict(ICE_MAP)
        if ice.has("t_skin"):                                              # FESOM main: the Newton seed itself
            imap["t_skin"] = "t_skin"
        for name, var in imap.items():
            if not ice.has(var):
                rep["missing"].append(var)
                continue
            a = ice.read(var, n)
            if name == "t_skin" and var == "ice_temp" and np.nanmax(np.abs(a)) > 150.0:   # Kelvin on file
                a = a - 273.15
            out[name] = a
            rep["restart"].append(f"{name}<-{var}")
    if oce.has("d_eta"):                                                   # FESOM main: CG warm start
        out["d_eta"] = oce.read("d_eta", n)
        rep["restart"].append("d_eta<-d_eta")
    for name in ("sigma11", "sigma12", "sigma22"):
        if ice is not None and ice.has(name):                              # FESOM main restarts them
            out[name] = ice.read(name, e)
            rep["restart"].append(f"{name}<-{name}")
        else:
            out[name] = np.zeros_like(ref[name])
            rep["zeroed"].append(name)
    done = {r.split("<-")[0].split("=")[0].split(" ")[0] for r in rep["restart"] + rep["derived"]}
    rep["reference"] = sorted(k for k in out if k not in done and k not in rep["zeroed"])
    from .state import State
    return State(**out), rep


def restart_step(when: datetime.datetime, start_year: int, dt: float) -> int:
    """The fesom-jax step count of ``when`` from ``start_year``-01-01 00:00 at ``dt`` seconds."""
    s = (when - datetime.datetime(start_year, 1, 1)).total_seconds() / float(dt)
    if abs(s - round(s)) > 1e-9:
        raise ValueError(f"{when} is not a whole number of {dt} s steps from {start_year}-01-01")
    return int(round(s))
