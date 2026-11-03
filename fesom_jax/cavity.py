"""Ice-shelf cavities (FESOM2 ``use_cavity``): the ocean-ice-shelf boundary kernels of ``cavity_param.F90``.

Port of FESOM2 main 0aaa2d73 ``src/cavity_param.F90`` (lane L-OCN, plan
``~/awicm_jax/docs/plans/20261103-jaxcpl-core3.md``; inventory in ``docs/lanes/core3.md``):

* :func:`three_eq_melt` / :func:`cavity_heat_water_fluxes_3eq` — the three-equation melt at the ice-shelf base
  (Hellmer & Olbers 1989, Holland & Jenkins 1999; BRIOS ``iceshelf``), with ``cavity_gamma_scale``;
* :func:`cavity_momentum_fluxes` — quadratic drag of the ice-shelf base on the top wet layer;
* :func:`cavity_ice_clean_vel` / :func:`cavity_ice_clean_ma` — no sea ice (velocity, area, mass) under the shelves;
* :func:`potit` — in-situ temperature from potential temperature (fixed-point iteration, RK4 ``pttmpr``).

Everything here is a no-op off the cavities (``ulevels_nod2D == 1`` / ``ulevels == 1``) and nothing calls it unless
``use_cavity`` is on, so the default model is unchanged.

Precision. FESOM compiles with ``-r8`` (double) / ``-r4`` (single, ``USE_SINGLE_PRECISION``), so a bare literal
(``-0.0575``) is ``WP`` like a ``_WP`` one: every constant here is a plain Python float, and the arithmetic runs in the
dtype of the inputs (float64, or float32 under :func:`fesom_jax.precision.scope` single precision). ``hnode`` is
``WP_full`` (float64) in both builds: ``re = hnode/un`` is formed in float64 and rounded to ``WP``. The Intel build
also uses ``-no-prec-div -fimf-use-svml``, so divisions/``log``/``sqrt`` may differ from IEEE by an ulp.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class CavityConfig(NamedTuple):
    """The ``use_cavity`` switch block (``namelist.config &run_config`` + ``namelist.oce``). Planned as the
    ``cavity:`` slot of :class:`fesom_jax.run_config.RunConfig` (``None`` = no cavities, today's model); not yet
    registered there because nothing in the step consumes it yet (lane L-OCN, ``docs/lanes/core3.md``).

    ``gamma_scale``: ``cavity_gamma_scale`` (jaxcpl 0.6). ``density_ref_T``/``density_ref_S``: the reference water
    mass of ``use_density_ref``, which FESOM forces on with cavities (``eos.reference_density``).
    """
    gamma_scale: float = 1.0
    density_ref_T: float = 2.0
    density_ref_S: float = 34.0


# --- cavity_heat_water_fluxes_3eq parameters (cavity_param.F90:157-180) ----------------------------------------------
_A, _B, _C = -0.0575, 0.0901, 7.61e-4   # Foldvik & Kvinge (1974) freezing point
_PR, _SC = 13.8, 2432.0                       # Prandtl, Schmidt numbers
_AK = 2.50e-3                                       # drag coefficient
_UN = 1.95e-6                                       # kinematic viscosity [m2/s]
_TOB = -20.0                                        # ice-shelf surface temperature [degC]
_RHOI, _RHOFW = 920.0, 1000.0
_CPW, _LHF = 4180.0, 334000.0
_TDIF = 1.54e-6                                     # thermal diffusivity of the shelf [m2/s]
_ATK = 273.15
_TWO_THIRDS = 2.0 / 3.0

# --- adlprt (adiabatic lapse rate) coefficients -----------------------------------------------
_S0 = 35.0
_A0, _A1, _A2, _A3 = 3.5803e-5, 8.5258e-6, -6.8360e-8, 6.6228e-10
_B0, _B1 = 1.8932e-6, -4.2393e-8
_C0, _C1, _C2, _C3 = 1.8741e-8, -6.7795e-10, 8.7330e-12, -5.4481e-14
_D0, _D1 = -1.1351e-10, 2.7759e-12
_E0, _E1, _E2 = -4.6206e-13, 1.8676e-14, -2.1687e-16


def _consts(dtype):
    """The parameters the compiler folds (``pr1 = pr**(2./3.)``, ``sak1 = sqrt(ak)``, ``cpi``), in ``dtype``."""
    f = np.dtype(dtype).type
    pr1 = np.power(f(_PR), f(_TWO_THIRDS))
    sc1 = np.power(f(_SC), f(_TWO_THIRDS))
    sak1 = np.sqrt(f(_AK))
    cpi = f(152.5) + f(7.122) * (f(_ATK) + f(_TOB))   # Paterson, "The Physics of Glaciers"
    return float(pr1), float(sc1), float(sak1), float(cpi)


def adlprt(salz, temp, pres):
    """Adiabatic temperature gradient [K/dbar] (``cavity_param.F90:adlprt``)."""
    ds = salz - _S0
    return ((((_E2 * temp + _E1) * temp + _E0) * pres
             + ((_D1 * temp + _D0) * ds
                + ((_C3 * temp + _C2) * temp + _C1) * temp + _C0)) * pres
            + (_B1 * temp + _B0) * ds + ((_A3 * temp + _A2) * temp + _A1) * temp + _A0)


def pttmpr(salz, temp, pres, rfpres):
    """Potential temperature referenced to ``rfpres`` (4th-order Runge-Kutta, ``cavity_param.F90:pttmpr``)."""
    ct2, ct3 = 0.29289322, 1.707106781                    # _WP literals
    cq2a, cq2b, cq3a, cq3b = 0.58578644, 0.121320344, 3.414213562, -4.121320344
    p, t = pres, temp
    dp = rfpres - pres
    dt = dp * adlprt(salz, t, p)
    t = t + 0.5 * dt
    q = dt
    p = p + 0.5 * dp
    dt = dp * adlprt(salz, t, p)
    t = t + ct2 * (dt - q)
    q = cq2a * dt + cq2b * q
    dt = dp * adlprt(salz, t, p)
    t = t + ct3 * (dt - q)
    q = cq3a * dt + cq3b * q
    p = rfpres
    dt = dp * adlprt(salz, t, p)
    return t + (dt - q - q) / 6.0


def potit(salz, pt, pres, rfpres, n_iter: int = 100):
    """In-situ temperature [degC] from potential temperature ``pt`` (``cavity_param.F90:potit``).

    Fortran iterates ``tin = pt + epsi; epsi -= pttmpr(tin) - pt`` and RETURNS at the first ``|ptd| < 0.001``. Here
    every lane runs the same fixed trip count and freezes once converged (a ``fori_loop``: reverse-differentiable),
    so each lane returns the ``tin`` of its own first converged iteration, as Fortran does. (Fortran stops the model
    if 100 iterations do not converge; a lane that does not converge keeps its last ``tin`` here.)
    """
    tpmd = 0.001

    def body(_, carry):
        tin, epsi, done = carry
        tin_i = pt + epsi
        ptd = pttmpr(salz, tin_i, pres, rfpres) - pt
        conv = jnp.abs(ptd) < tpmd
        tin = jnp.where(done, tin, tin_i)
        epsi = jnp.where(done | conv, epsi, epsi - ptd)
        return tin, epsi, done | conv

    z = jnp.zeros_like(pt)
    tin, _, _ = jax.lax.fori_loop(0, n_iter, body, (z, z, jnp.zeros(pt.shape, bool)))
    return tin


def three_eq_melt(temp, sal, zice, u_top, v_top, h_top, rhow, gamma_scale):
    """The three-equation ice-shelf melt at the top wet cell of each cavity node.

    Inputs per node (any shape, broadcast): potential temperature ``temp`` and salinity ``sal`` of the top wet layer,
    its mid depth ``zice`` (<0, ``Z_3d_n(nzmin)``), the node velocity ``u_top``/``v_top`` there, the layer thickness
    ``h_top`` (``hnode(nzmin)``, float64 = ``WP_full``), the in-situ density ``rhow`` (= ``density_m_rho0 +
    density_ref``), and ``gamma_scale`` (``cavity_gamma_scale``). Returns ``(heat_flux, water_flux)`` in FESOM's
    convention: ``heat_flux`` [W/m2] positive upward (ocean -> shelf), ``water_flux`` [m/s] positive = freshwater out
    of the ocean (melt is negative). ``cavity_param.F90:143-322``, line by line.
    """
    dtype = temp.dtype
    pr1, sc1, sak1, cpi = _consts(dtype)
    tin = potit(sal, temp, jnp.abs(zice) * 1.025, 0.0)          # depth [m] -> pressure [dbar]
    vt1 = jnp.sqrt(u_top * u_top + v_top * v_top)
    vt1 = jnp.maximum(vt1, 0.001)
    re = (h_top / _UN).astype(dtype)                             # hnode is WP_full: divide in float64, store WP
    gats1 = sak1 * vt1
    gats2 = 2.12 * jnp.log(gats1 * re) - 9.0
    gat = gamma_scale * gats1 / (gats2 + 12.5 * pr1)
    gas = gamma_scale * gats1 / (gats2 + 12.5 * sc1)
    rhor = _RHOI / rhow
    ep1 = _CPW * gat
    ep2 = cpi * gas
    ep3 = _LHF * gas
    ep31 = -rhor * rhor * cpi * _TDIF / zice                     # rhor^2: D_ice = D_draft / rhor
    ep4 = _B + _C * zice
    tf0 = _A * sal + ep4
    cold = tin < tf0                                             # heat conduction branch
    ex1 = jnp.where(cold, _A * (ep1 + ep31), _A * (ep1 - ep2))
    ex2 = jnp.where(cold, ep1 * (tin - ep4) + ep3 + ep31 * (_TOB - ep4),
                    ep1 * (ep4 - tin) + ep2 * (_TOB + _A * sal - ep4) - ep3)
    ex3 = jnp.where(cold, ep3 * sal, sal * (ep2 * (ep4 - _TOB) + ep3))
    ex6 = jnp.where(cold, 0.5, -0.5)
    ex4 = ex2 / ex1
    ex5 = ex3 / ex1
    sr1 = jnp.maximum(0.25 * ex4 * ex4 - ex5, 0.0)              # guard against a negative discriminant
    sr2 = ex6 * ex4
    sf1 = sr2 + jnp.sqrt(sr1)
    tf1 = _A * sf1 + ep4
    sf2 = sr2 - jnp.sqrt(sr1)
    tf2 = _A * sf2 + ep4
    pos = sf1 > 0.0
    tf = jnp.where(pos, tf1, tf2)
    sf = jnp.where(pos, sf1, sf2)
    heat_flux = rhow * _CPW * gat * (tin - tf)
    water_flux = rhow / _RHOFW * gas * (sf - sal) / sf
    return heat_flux, water_flux


def _top(field, top):
    """``field[n, top[n]]`` for a (nod2D, nl) or (nod2D, nl, 2) field."""
    idx = top.reshape((-1,) + (1,) * (field.ndim - 1))
    return jnp.take_along_axis(field, idx, axis=1)[:, 0]


def cavity_heat_water_fluxes_3eq(mesh, T, S, uvnode, hnode, Z_3d_n, rhow, heat_flux, water_flux, gamma_scale):
    """Overwrite ``heat_flux``/``water_flux`` at the cavity nodes with the three-equation melt.

    ``T``, ``S``, ``hnode``, ``Z_3d_n``, ``rhow`` are (nod2D, nl) node columns (``rhow`` the in-situ density), ``uvnode``
    (nod2D, nl, 2). Off-cavity nodes keep their values. Returns ``(heat_flux, water_flux, melt_mask)``. FESOM also
    sets ``ice%flx_h = -heat_flux`` / ``ice%flx_fw = -water_flux`` there (diagnostics only).
    """
    cav = mesh.ulevels_nod2D > 1
    top = (mesh.ulevels_nod2D - 1).astype(jnp.int32)
    uv = _top(uvnode, top)
    hf, wf = three_eq_melt(_top(T, top), _top(S, top), _top(Z_3d_n, top), uv[:, 0], uv[:, 1],
                           _top(hnode, top), _top(rhow, top), gamma_scale)
    return jnp.where(cav, hf, heat_flux), jnp.where(cav, wf, water_flux), cav


def cavity_momentum_fluxes(mesh, uv, uvnode, stress_surf, stress_node_surf, density_0, C_d):
    """Ice-shelf base drag on the top wet layer (``cavity_param.F90:cavity_momentum_fluxes``).

    ``stress = -|u| * density_0 * C_d * u`` at the cavity elements (``uv`` (elem2D, nl, 2), ``stress_surf``
    (elem2D, 2)) and cavity nodes (``uvnode`` (nod2D, nl, 2), ``stress_node_surf`` (nod2D, 2)); unchanged elsewhere.
    """
    def drag(u, ulev, old):
        top = (ulev - 1).astype(jnp.int32)
        ut = _top(u, top)                                          # (n, 2)
        aux = jnp.sqrt(ut[:, 0] ** 2 + ut[:, 1] ** 2) * density_0 * C_d
        return jnp.where((ulev > 1)[:, None], -aux[:, None] * ut, old)

    return drag(uv, mesh.ulevels, stress_surf), drag(uvnode, mesh.ulevels_nod2D, stress_node_surf)


def cavity_ice_clean_vel(mesh, u_ice, v_ice):
    """No sea-ice velocity under the shelves (``cavity_ice_clean_vel``)."""
    cav = mesh.ulevels_nod2D > 1
    return jnp.where(cav, 0.0, u_ice), jnp.where(cav, 0.0, v_ice)


def cavity_ice_clean_ma(mesh, a_ice, m_ice, m_snow):
    """No sea ice (area, ice and snow volume) under the shelves (``cavity_ice_clean_ma``)."""
    cav = mesh.ulevels_nod2D > 1
    return jnp.where(cav, 0.0, a_ice), jnp.where(cav, 0.0, m_ice), jnp.where(cav, 0.0, m_snow)


def cavity_boundary_node_mask(mesh, boundary_node):
    """The sea-ice velocity boundary of the mEVP/EVP with cavities (``ice_maEVP.F90:365-380``).

    FESOM zeroes the ice velocity (``bc_zero``) at the endpoints of coastal edges AND, with ``use_cavity``, at the
    endpoints of every edge with a cavity cell (``ulevels > 1``) on either side: the ice-shelf front acts as a coast.
    ``boundary_node`` is the coastal mask (:func:`fesom_jax.ice_evp.boundary_node_mask`, GLOBAL when sharded);
    returns it OR the cavity-edge endpoints. Callers pass this instead of the coastal mask when ``use_cavity``.
    """
    from . import ops
    et = jnp.asarray(mesh.edge_tri)
    ul = jnp.asarray(mesh.ulevels)
    cav1 = ul[jnp.maximum(et[:, 0], 0)] > 1
    cav2 = (et[:, 1] >= 0) & (ul[jnp.maximum(et[:, 1], 0)] > 1)
    ce = (cav1 | cav2).astype(jnp.int32)
    cnt = ops.scatter_add(ce, mesh.edges[:, 0], mesh.nod2D) + ops.scatter_add(ce, mesh.edges[:, 1], mesh.nod2D)
    return jnp.asarray(boundary_node) | (cnt > 0)
