"""REcoM's carbonate chemistry: the mocsy paths REcoM 692f804 calls (parent plan Task 8, gate G1).

A literal f64 port of exactly the mocsy routines REcoM calls, with REcoM's option values.
Every function is element-wise over arbitrary (broadcast-compatible) array shapes, pure, and
jit/vmap/grad safe (``jax.lax`` control flow only).

Entry points and their REcoM call sites (``REcoM_src`` at 692f804):

* :func:`vars_sprac` -- the interior water-column carbonate system
  (``recom_carbonate_chemistry.F90:240-247``, called per layer from ``recom_sms.F90:400``),
  ``mocsy/vars.F90:254-705``.
* :func:`pistonvel` -- ``kw660`` from the 10-m wind and the ice fraction
  (``recom_forcing.F90:195``), ``mocsy/gasx.F90:428-470``.
* :func:`flxco2` -- the air-sea CO2 flux and the surface carbonate system
  (``recom_forcing.F90:222-226``), ``mocsy/gasx.F90:8-379``; it calls ``vars_sprac`` at
  zero depth, ``x2pCO2atm``/``vapress`` and ``p2fCO2``.
* :func:`o2flux` -- the air-sea O2 flux (``recom_forcing.F90:278``), ``mocsy/gasx.F90:1002-1082``
  with ``o2sato`` and ``sco2``.
* :func:`scco2` -- the CO2 Schmidt number; REcoM calls it directly only on the 13C/14C
  path (``recom_forcing.F90:344``, out of scope), flxco2 uses it internally.

REcoM's options, the only ones ported (any other value raises ``NotImplementedError``):
``optCON='mol/m3'`` (concentrations in mol m-3; mocsy divides by the in-situ density),
``optT='Tpot'`` (potential temperature in, in-situ temperature computed), ``optP='m'``
(depth in m; pressure from ``p80``), ``optB='u74'`` (Uppstrom 1974 boron), ``optK1K2='l'``
(Lueker et al. 2000), ``optKf='dg'`` (Dickson and Riley 1979), ``optGAS='Pinsitu'``
(in-situ pressure in K0 and the fugacity coefficient), ``optS='Sprc'`` (practical salinity).

Units, as REcoM passes them: ``temp`` potential temperature [degC]; ``sal`` [psu]; ``alk``
[eq m-3]; ``dic``, ``sil``, ``phos`` [mol m-3]; ``patm`` [atm]; ``depth`` [m, positive down];
``lat`` [degrees N]; ``kw660`` [m s-1]; ``xco2`` [ppm]; ``o2`` [mol m-3]. Outputs: pH on the
total scale; pCO2, fCO2, dpCO2 [uatm]; CO2*, HCO3-, CO3-- [mol m-3]; Omega [-]; Kspc
[mol2 kg-2]; rhoSW [kg m-3]; p [dbar]; tempis [degC]; co2flux [mol m-2 s-1, air to sea];
K0 [(mol/kg)/atm]; o2ex [mol m-2 s-1].

What belongs to the CALLER (parent Tasks 9-10), not ported here:

* The input clamps and floors REcoM applies before mocsy. Interior
  (``recom_sms_init.F90:285-339``): ``T = min(max(2, T), 40)``, ``S = min(max(21, S), 43)``,
  ``DIC/Alk/Si = max(tiny*1e-3, (state + sms)*1e-3)``, ``PO4 = max(tiny*1e-3, DIN*1e-3)/16``
  with ``tiny = 2.23e-16``. Surface (``recom_forcing.F90:154-162``): the same floors from
  ``state(nzmin)`` (no ``sms``), ``T = min(max(2, T), 40)`` but ``S = max(21, S)`` only (no
  upper clamp). ``patm = Loc_slp / 101325`` from ``press_air`` [Pa]; ``ppo`` for O2 is the same
  value. The interior ``depth`` is ``-zbar_3d_n(k)`` (the TOP interface of layer k, not the
  mid-layer depth; ``recom_carbonate_chemistry.F90:235``, ``recom_extra.F90:110-112``).
* The DIC range guard (``recom_forcing.F90:197``: ``DIC > 10 mol m-3`` stops the run), the
  ``|co2flux| > 1e10`` stop, and the cavity branch (``nzmin > 1``: no flxco2/o2flux, NaN fills).
* The per-layer refresh schedule of the interior cache (``recom_carbonate_chemistry.F90:236-249``).
* ``GloHplus`` stores the surface **pH**, not [H+], and nothing reads it as a first guess:
  mocsy's solver always starts from its own ``ahini_for_at`` estimate.

The pH solver (``phsolvers.F90:429-630``, SolveSAPHE ``solve_at_general``) is a safeguarded
Newton iteration on ln[H+] with bisection fall-back, stopped per point at
``|delta ln h| < 1e-8`` (``phsolvers.F90:23``, ``:612``) or after ``jp_maxniter_atgen = 50``
iterations (then ``h = -1``, pH = 1e20). Here it runs a FIXED ``n_iter`` iterations in which each
point FREEZES once it has stopped, so every point takes exactly Fortran's iterates and the
result does not depend on ``n_iter`` as long as ``n_iter`` is at least that point's count. The
default ``n_iter = 50`` reproduces Fortran's cap and failure value too. On the G1 grid
(4324 points, ``scripts/recom/make_mocsy_g1.py``) the worst case is 11 iterations (12 for the
BetaD solves); ocean values take 2-4. The cost is linear in ``n_iter`` (one Levante CPU node,
2e5 points: 0.10 s at 13 iterations, 0.30 s at 50, forward or gradient); a caller that needs the
speed passes a smaller ``n_iter`` (>= ``G1_WORST_NITER + 1``: an exactly-zero residual exits one
evaluation later) and can watch ``niter``. The root's derivative is implicit (custom JVP:
``dh = -(dF/dtheta . dtheta) / (dF/dh)`` at the converged h), so reverse mode stores no
iterates and the gradient is that of the exact root.

Fortran literals are double here (REcoM compiles mocsy with ``-r8``, ``REcoM_src/CMakeLists.txt:31``,
and leaves ``USE_PRECISION`` undefined, so ``SGLE(x)`` is a no-op on doubles). The arithmetic
order follows the Fortran statement by statement; the G1 floor is the transcendental and
division ulps of ifort ``-fimf-use-svml -no-prec-div -no-prec-sqrt`` vs XLA (see
``fesom_jax/tests/test_recom_carbonate.py``).
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from .. import precision as PR

# REcoM's option strings (recom_carbonate_chemistry.F90:243-245, recom_forcing.F90:225-226)
RECOM_OPTIONS = dict(optCON="mol/m3", optT="Tpot", optP="m", optB="u74", optK1K2="l",
                     optKf="dg", optGAS="Pinsitu", optS="Sprc")

PH_MAXITER = 50          # jp_maxniter_atgen, phsolvers.F90:29
PH_RDEL_TARGET = 1.0e-8  # pp_rdel_ah_target, phsolvers.F90:23
G1_WORST_NITER = 12      # worst solver count on the G1 grid (BetaD solve; 11 for pH itself)
FILL = 1.0e20            # mocsy's masked value

_R_BAR = 83.14472            # constants.F90:349, cm3 bar / (mol K)
_RGAS_ATM = 82.05736         # cm3 atm / (mol K), CODATA 2006
_VBAR_CO2 = 32.3             # cm3 / mol, Weiss 1974
_SQRT2 = math.sqrt(2.0)

# Millero (1995) pressure-correction coefficients, constants.F90:271-293 (index 0..11 =
# K1, K2, Kb, Kw, Ks, Kf, Kspc, Kspa, K1p, K2p, K3p, Ksi)
_A0 = (-25.5, -15.82, -29.48, -20.02, -18.03, -9.78, -48.76, -45.96, -14.51, -23.12, -26.57,
       -29.48)
_A1 = (0.1271, -0.0219, 0.1622, 0.1119, 0.0466, -0.0090, 0.5304, 0.5304, 0.1211, 0.1758,
       0.2020, 0.1622)
_A2 = (0.0, 0.0, -2.608e-3, -1.409e-3, 0.316e-3, -0.942e-3, 0.0, 0.0, -0.321e-3, -2.647e-3,
       -3.042e-3, -2.6080e-3)
_B0 = (-3.08e-3, 1.13e-3, -2.84e-3, -5.13e-3, -4.53e-3, -3.91e-3, -11.76e-3, -11.76e-3,
       -2.67e-3, -5.15e-3, -4.08e-3, -2.84e-3)
_B1 = (0.0877e-3, -0.1475e-3, 0.0, 0.0794e-3, 0.09e-3, 0.054e-3, 0.3692e-3, 0.3692e-3,
       0.0427e-3, 0.09e-3, 0.0714e-3, 0.0)
_B2 = (0.0,) * 12


def _check_options(opts: dict) -> None:
    for key, val in opts.items():
        if val is None:
            continue
        want = RECOM_OPTIONS[key]
        if val.strip() != want:
            raise NotImplementedError(
                f"mocsy {key}={val!r}: only REcoM's {key}={want!r} is ported "
                "(fesom_jax/recom/carbonate.py)")


# --------------------------------------------------------------------------------------
# Seawater helpers
# --------------------------------------------------------------------------------------
def p80(depth, lat):
    """Pressure [dbar] from depth [m] and latitude [deg] (Saunders 1981), ``p80.F90:7-49``."""
    pi = 3.141592654
    plat = jnp.abs(lat * pi / 180.0)
    d = jnp.sin(plat)
    c1 = 5.92e-3 + (d * d) * 5.25e-3
    return ((1.0 - c1) - jnp.sqrt(((1.0 - c1) * (1.0 - c1)) - (8.84e-6 * depth))) / 4.42e-6


def sw_adtg(s, t, p):
    """Adiabatic temperature gradient [degC/dbar] (UNESCO 1983), ``sw_adtg.F90:7-53``."""
    a0, a1, a2, a3 = 3.5803e-5, 8.5258e-6, -6.836e-8, 6.6228e-10
    b0, b1 = 1.8932e-6, -4.2393e-8
    c0, c1, c2, c3 = 1.8741e-8, -6.7795e-10, 8.733e-12, -5.4481e-14
    d0, d1 = -1.1351e-10, 2.7759e-12
    e0, e1, e2 = -4.6206e-13, 1.8676e-14, -2.1687e-16
    sref = 35.0
    return (a0 + (a1 + (a2 + a3 * t) * t) * t
            + (b0 + b1 * t) * (s - sref)
            + ((c0 + (c1 + (c2 + c3 * t) * t) * t) + (d0 + d1 * t) * (s - sref)) * p
            + (e0 + (e1 + e2 * t) * t) * p * p)


def sw_ptmp(s, t, p, pr):
    """Potential temperature at reference pressure ``pr`` (4th-order Runge-Kutta),
    ``sw_ptmp.F90:7-68``."""
    del_p = pr - p
    del_th = del_p * sw_adtg(s, t, p)
    th = t + 0.5 * del_th
    q = del_th
    del_th = del_p * sw_adtg(s, th, p + 0.5 * del_p)
    th = th + (1.0 - 1.0 / _SQRT2) * (del_th - q)
    q = (2.0 - _SQRT2) * del_th + (-2.0 + 3.0 / _SQRT2) * q
    del_th = del_p * sw_adtg(s, th, p + 0.5 * del_p)
    th = th + (1.0 + 1.0 / _SQRT2) * (del_th - q)
    q = (2.0 + _SQRT2) * del_th + (-2.0 - 3.0 / _SQRT2) * q
    del_th = del_p * sw_adtg(s, th, p + del_p)
    return th + (del_th - 2.0 * q) / (2.0 * 3.0)


def sw_temp(s, t, p, pr):
    """In-situ temperature from potential temperature, ``sw_temp.F90:7-81``:
    ``sw_ptmp(s, t, pr, p)``."""
    return sw_ptmp(s, t, pr, p)


def tpot_to_tinsitu(temp, sal, p):
    """``optT='Tpot'`` -> ``(tempis68, tempis90)``, ``vars.F90:466-473``."""
    tempot68 = (temp - 0.0002) / 0.99975
    tempis68 = sw_temp(sal, tempot68, p, 0.0)
    tempis90 = 0.99975 * tempis68 + 0.0002
    return tempis68, tempis90


def rho(salt, temp, pbar):
    """In-situ density [kg m-3] (UNESCO EOS-80, IPTS-68 temperature), ``rho.F90:7-93``."""
    s, x, p = salt, temp, pbar
    rhow = (999.842594 + 6.793952e-2 * x
            - 9.095290e-3 * x * x + 1.001685e-4 * x ** 3
            - 1.120083e-6 * x ** 4 + 6.536332e-9 * x ** 5)
    a = (8.24493e-1 - 4.0899e-3 * x
         + 7.6438e-5 * x * x - 8.2467e-7 * x ** 3 + 5.3875e-9 * x ** 4)
    b = -5.72466e-3 + 1.0227e-4 * x - 1.6546e-6 * x * x
    c = 4.8314e-4
    sqrts = jnp.sqrt(s)
    rho0 = rhow + a * s + b * s * sqrts + c * (s * s)
    ksbmw = (19652.21 + 148.4206 * x - 2.327105 * x * x
             + 1.360477e-2 * x ** 3 - 5.155288e-5 * x ** 4)
    ksbm0 = (ksbmw + s * (54.6746 - 0.603459 * x + 1.09987e-2 * x ** 2 - 6.1670e-5 * x ** 3)
             + s * sqrts * (7.944e-2 + 1.6483e-2 * x - 5.3009e-4 * x ** 2))
    ksbm = (ksbm0
            + p * (3.239908 + 1.43713e-3 * x + 1.16092e-4 * x ** 2 - 5.77905e-7 * x ** 3)
            + p * s * (2.2838e-3 - 1.0981e-5 * x - 1.6078e-6 * x ** 2)
            + p * s * sqrts * 1.91075e-4
            + p * p * (8.50935e-5 - 6.12293e-6 * x + 5.2787e-8 * x ** 2)
            + p * p * s * (-9.9348e-7 + 2.0816e-8 * x + 9.1697e-10 * x ** 2))
    return rho0 / (1.0 - p / ksbm)


# --------------------------------------------------------------------------------------
# Equilibrium constants
# --------------------------------------------------------------------------------------
class Constants(NamedTuple):
    """Equilibrium constants on the total scale at in-situ T and p (``constants.F90``)."""
    K0: jax.Array
    K1: jax.Array
    K2: jax.Array
    Kb: jax.Array
    Kw: jax.Array
    Ks: jax.Array
    Kf: jax.Array
    Kspc: jax.Array
    Kspa: jax.Array
    K1p: jax.Array
    K2p: jax.Array
    K3p: jax.Array
    Ksi: jax.Array
    St: jax.Array
    Ft: jax.Array
    Bt: jax.Array


def constants(temp, sal, patm, depth, lat) -> Constants:
    """``constants(...)`` with REcoM's options (Tpot, m, u74, l, dg, Pinsitu, Sprc),
    ``constants.F90:9-709``.

    ``temp`` is potential temperature; the in-situ temperature, pressure and ionic strength
    are recomputed here as in Fortran. Outside ``-5 <= temp < 100`` every constant is 1e20
    (``constants.F90:356``, ``:689-706``).
    """
    temp, sal, patm, depth, lat = jnp.broadcast_arrays(
        *(jnp.asarray(x, dtype=jnp.float64) for x in (temp, sal, patm, depth, lat)))
    ok = (temp >= -5.0) & (temp < 1.0e2)
    temp_s = jnp.where(ok, temp, 10.0)        # safe lanes: no NaN/inf in the masked branch

    p = p80(depth, lat)                                            # :348-352
    _, tempis = tpot_to_tinsitu(temp_s, sal, p)                    # :354-358
    ssal = jnp.where(sal < 0.0, 0.0, sal)                          # :361-365
    t = tempis
    tk = 273.15 + t
    invtk = 1.0 / tk
    dlogtk = jnp.log(tk)
    patmd = patm
    prb = p / 10.0
    s = ssal
    s2 = s * s
    sqrts = jnp.sqrt(s)
    s15 = s ** 1.5
    scl = s / 1.80655
    is_ = 19.924 * s / (1000.0 - 1.005 * s)                        # :411
    is2 = is_ * is_
    sqrtis = jnp.sqrt(is_)
    St = 0.14 * scl / 96.062                                       # :418
    Ft = 0.000067 * scl / 18.9984
    Bt = 0.000232 * scl / 10.811                                   # :429 (u74)
    # Pinsitu (:443-447)
    tk0 = tk
    phydro_atm = prb / 1.01325
    ptot = patmd + phydro_atm
    tmp = 9345.17 / tk0 - 60.2409 + 23.3585 * jnp.log(tk0 / 100.0)
    nk0we74 = tmp + s * (0.023517 - 0.00023656 * tk0 + 0.0047036e-4 * tk0 * tk0)
    K0 = jnp.exp(nk0we74)                                          # :455
    # Lueker et al. (2000), :461-465
    K1 = 10.0 ** (-1.0 * (3633.86 * invtk - 61.2172 + 9.6777 * dlogtk
                          - 0.011555 * s + 0.0001152 * s2))
    K2 = 10.0 ** (-1.0 * (471.78 * invtk + 25.9290 - 3.16967 * dlogtk
                          - 0.01781 * s + 0.0001122 * s2))
    Kb = jnp.exp((-8966.90 - 2890.53 * sqrts - 77.942 * s
                  + 1.728 * s15 - 0.0996 * s2) * invtk
                 + (148.0248 + 137.1942 * sqrts + 1.62142 * s)
                 + (-24.4344 - 25.085 * sqrts - 0.2474 * s)
                 * dlogtk + 0.053105 * sqrts * tk)                 # :503
    K1p = jnp.exp(-4576.752 * invtk + 115.540 - 18.453 * dlogtk
                  + (-106.736 * invtk + 0.69171) * sqrts
                  + (-0.65643 * invtk - 0.01844) * s)              # :516
    K2p = jnp.exp(-8814.715 * invtk + 172.1033 - 27.927 * dlogtk
                  + (-160.340 * invtk + 1.3566) * sqrts
                  + (0.37335 * invtk - 0.05778) * s)
    K3p = jnp.exp(-3070.75 * invtk - 18.126
                  + (17.27039 * invtk + 2.81197)
                  * sqrts + (-44.99486 * invtk - 0.09984) * s)
    log_ion = jnp.log(1.0 - 0.001005 * s)
    Ksi = jnp.exp(-8904.2 * invtk + 117.400 - 19.334 * dlogtk
                  + (-458.79 * invtk + 3.5913) * sqrtis
                  + (188.74 * invtk - 1.5998) * is_
                  + (-12.1652 * invtk + 0.07871) * is2
                  + log_ion)                                       # :548
    Kw = jnp.exp(-13847.26 * invtk + 148.9802 - 23.6521 * dlogtk
                 + (118.67 * invtk - 5.977 + 1.0495 * dlogtk)
                 * sqrts - 0.01615 * s)                            # :560
    Ks_0p = jnp.exp(-4276.1 * invtk + 141.328 - 23.093 * dlogtk
                    + (-13856.0 * invtk + 324.57 - 47.986 * dlogtk) * sqrtis
                    + (35474.0 * invtk - 771.54 + 114.723 * dlogtk) * is_
                    - 2698.0 * invtk * is_ ** 1.5 + 1776.0 * invtk * is2
                    + log_ion)                                     # :567
    Kf_0p = jnp.exp(1590.2 * invtk - 12.641 + 1.525 * sqrtis
                    + log_ion
                    + jnp.log(1.0 + St / Ks_0p))                   # :577 (dg)
    Kspc = 10.0 ** (-171.9065 - 0.077993 * tk + 2839.319 / tk
                    + 71.595 * PR.log10(tk)
                    + (-0.77712 + 0.0028426 * tk + 178.34 / tk) * sqrts
                    - 0.07711 * s + 0.0041249 * s15)               # :593
    Kspa = 10.0 ** (-171.945 - 0.077993 * tk + 2903.293 / tk
                    + 71.595 * PR.log10(tk)
                    + (-0.068393 + 0.0017276 * tk + 88.135 / tk) * sqrts
                    - 0.10018 * s + 0.0059415 * s15)
    K0 = K0 * jnp.exp(((1.0 - ptot) * _VBAR_CO2) / (_RGAS_ATM * tk0))   # :611
    # Millero (1995) pressure corrections, :616-623
    lnkpok0 = []
    for ipc in range(12):
        deltav = _A0[ipc] + _A1[ipc] * t + _A2[ipc] * t * t
        deltak = _B0[ipc] + _B1[ipc] * t + _B2[ipc] * t * t
        lnkpok0.append((-deltav + (0.5 * deltak * prb)) * prb / (_R_BAR * tk))
    Ks = Ks_0p * jnp.exp(lnkpok0[4])                               # :625
    total2free = 1.0 / (1.0 + St / Ks)
    total2free_0p = 1.0 / (1.0 + St / Ks_0p)
    Kf_0p = Kf_0p * total2free_0p
    Kf = Kf_0p * jnp.exp(lnkpok0[5])
    Kf = Kf / total2free
    free2sws = 1.0 + St / Ks + Ft / (Kf * total2free)
    total2sws = total2free * free2sws
    sws2total = 1.0 / total2sws                                    # :640
    free2sws_0p = 1.0 + St / Ks_0p + Ft / Kf_0p
    total2sws_0p = total2free_0p * free2sws_0p
    K1 = K1 * total2sws_0p                                         # :648-652 (Lueker: total -> SWS)
    K2 = K2 * total2sws_0p
    Kb = Kb * total2sws_0p
    K1 = K1 * jnp.exp(lnkpok0[0])
    K2 = K2 * jnp.exp(lnkpok0[1])
    Kb = Kb * jnp.exp(lnkpok0[2])
    Kw = Kw * jnp.exp(lnkpok0[3])
    Kspc = Kspc * jnp.exp(lnkpok0[6])
    Kspa = Kspa * jnp.exp(lnkpok0[7])
    K1p = K1p * jnp.exp(lnkpok0[8])
    K2p = K2p * jnp.exp(lnkpok0[9])
    K3p = K3p * jnp.exp(lnkpok0[10])
    Ksi = Ksi * jnp.exp(lnkpok0[11])
    K1 = K1 * sws2total                                            # :676-683
    K2 = K2 * sws2total
    K1p = K1p * sws2total
    K2p = K2p * sws2total
    K3p = K3p * sws2total
    Kb = Kb * sws2total
    Ksi = Ksi * sws2total
    Kw = Kw * sws2total
    out = Constants(K0, K1, K2, Kb, Kw, Ks, Kf, Kspc, Kspa, K1p, K2p, K3p, Ksi, St, Ft, Bt)
    return Constants(*(jnp.where(ok, v, FILL) for v in out))


# --------------------------------------------------------------------------------------
# pH solver (SolveSAPHE general, phsolvers.F90)
# --------------------------------------------------------------------------------------
def _equation_at(alk, h, dic, bor, po4, sil, so4, flu, K1, K2, Kb, Kw, Ks, Kf,
                 K1p, K2p, K3p, Ksi):
    """Alkalinity residual and its h-derivative, ``phsolvers.F90:81-218`` (K0 unused)."""
    aphscale = 1.0 + so4 / Ks
    znumer_dic = 2.0 * K1 * K2 + h * K1
    zdenom_dic = K1 * K2 + h * (K1 + h)
    zalk_dic = dic * (znumer_dic / zdenom_dic)
    zdenom_bor = Kb + h
    zalk_bor = bor * (Kb / zdenom_bor)
    znumer_po4 = 3.0 * K1p * K2p * K3p + h * (2.0 * K1p * K2p + h * K1p)
    zdenom_po4 = K1p * K2p * K3p + h * (K1p * K2p + h * (K1p + h))
    zalk_po4 = po4 * (znumer_po4 / zdenom_po4 - 1.0)
    zdenom_sil = Ksi + h
    zalk_sil = sil * (Ksi / zdenom_sil)
    zdenom_so4 = Ks + h
    zalk_so4 = so4 * (Ks / zdenom_so4 - 1.0)
    zdenom_flu = Kf + h
    zalk_flu = flu * (Kf / zdenom_flu - 1.0)
    zalk_wat = Kw / h - h / aphscale
    eqn = (zalk_dic + zalk_bor + zalk_po4 + zalk_sil
           + zalk_so4 + zalk_flu
           + zalk_wat - alk)
    zdnumer_dic = K1 * K1 * K2 + h * (4.0 * K1 * K2 + h * K1)
    zdalk_dic = -dic * (zdnumer_dic / (zdenom_dic * zdenom_dic))
    zdalk_bor = -bor * (Kb / (zdenom_bor * zdenom_bor))
    zdnumer_po4 = (K1p * K2p * K1p * K2p * K3p
                   + h * (4.0 * K1p * K1p * K2p * K3p
                          + h * (9.0 * K1p * K2p * K3p + K1p * K1p * K2p
                                 + h * (4.0 * K1p * K2p
                                        + h * K1p))))
    zdalk_po4 = -po4 * (zdnumer_po4 / (zdenom_po4 * zdenom_po4))
    zdalk_sil = -sil * (Ksi / (zdenom_sil * zdenom_sil))
    zdalk_so4 = -so4 * (Ks / (zdenom_so4 * zdenom_so4))
    zdalk_flu = -flu * (Kf / (zdenom_flu * zdenom_flu))
    deqn = (zdalk_dic + zdalk_bor + zdalk_po4 + zdalk_sil
            + zdalk_so4 + zdalk_flu
            - Kw / (h * h) - 1.0 / aphscale)
    return eqn, deqn


def _ahini_for_at(alkcb, dic, bor, K1, K2, Kb):
    """Initial [H+] guess, ``phsolvers.F90:367-425``."""
    safe_alk = jnp.where(alkcb > 0.0, alkcb, 1.0)
    zca = dic / safe_alk
    zba = bor / safe_alk
    za2 = Kb * (1.0 - zba) + K1 * (1.0 - zca)
    za1 = K1 * Kb * (1.0 - zba - zca) + K1 * K2 * (1.0 - (zca + zca))
    za0 = K1 * K2 * Kb * (1.0 - zba - (zca + zca))
    zd = za2 * za2 - 3.0 * za1
    dpos = zd > 0.0
    zsqrtd = jnp.sqrt(jnp.where(dpos, zd, 1.0))
    zhmin = jnp.where(za2 < 0.0, (-za2 + zsqrtd) / 3.0, -za1 / (za2 + zsqrtd))
    rad = -(za0 + zhmin * (za1 + zhmin * (za2 + zhmin))) / zsqrtd
    hini_cubic = zhmin + jnp.sqrt(jnp.where(dpos, rad, 1.0))
    hini = jnp.where(dpos, hini_cubic, 1.0e-7)
    hini = jnp.where(alkcb >= (2.0 * dic + bor), 1.0e-10, hini)
    return jnp.where(alkcb <= 0.0, 1.0e-3, hini)


def _solve_at_general_iter(n_iter, alk, dic, bor, po4, sil, so4, flu, K1, K2, Kb, Kw, Ks, Kf,
                           K1p, K2p, K3p, Ksi):
    """The iteration of ``solve_at_general`` (``phsolvers.F90:429-630``), each point frozen
    once Fortran would EXIT. Returns ``(h, niter)``; ``h = -1`` where the cap was reached."""
    consts = (K1, K2, Kb, Kw, Ks, Kf, K1p, K2p, K3p, Ksi)
    aphscale = 1.0 + so4 / Ks
    zh_ini = _ahini_for_at(alk, dic, bor, K1, K2, Kb)
    alknw_inf = -po4 - so4 - flu                                   # anw_infsup, :47-77
    alknw_sup = dic + dic + bor + po4 + po4 + sil
    zdelta = (alk - alknw_inf) ** 2 + 4.0 * Kw / aphscale          # :545-556
    zh_min = jnp.where(alk >= alknw_inf,
                       2.0 * Kw / (alk - alknw_inf + jnp.sqrt(zdelta)),
                       aphscale * (-(alk - alknw_inf) + jnp.sqrt(zdelta)) / 2.0)
    zdelta = (alk - alknw_sup) ** 2 + 4.0 * Kw / aphscale
    zh_max = jnp.where(alk <= alknw_sup,
                       aphscale * (-(alk - alknw_sup) + jnp.sqrt(zdelta)) / 2.0,
                       2.0 * Kw / (alk - alknw_sup + jnp.sqrt(zdelta)))
    zh = jnp.maximum(jnp.minimum(zh_max, zh_ini), zh_min)
    niter = jnp.zeros(zh.shape, jnp.int32)
    done = jnp.zeros(zh.shape, bool)
    absmin = jnp.full(zh.shape, jnp.finfo(jnp.float64).max, zh.dtype)   # HUGE(1._wp)

    def body(_, carry):
        zh, zh_min, zh_max, absmin, niter, done = carry
        zh_prev = zh
        zeqn, zdeqndh = _equation_at(alk, zh_prev, dic, bor, po4, sil, so4, flu, *consts)
        exact = zeqn == 0.0                                        # :580-586 EXIT, no count
        new_min = jnp.where(zeqn > 0.0, zh_prev, zh_min)
        new_max = jnp.where(zeqn < 0.0, zh_prev, zh_max)
        # bisection (:590-592)
        zh_b = jnp.sqrt(new_max * new_min)
        lnf_b = (zh_b - zh_prev) / zh_prev
        # Newton on ln h (:594-608)
        lnf_n = -zeqn / (zdeqndh * zh_prev)
        zh_n = jnp.where(jnp.abs(lnf_n) > 1.0,
                         zh_prev * jnp.exp(lnf_n),
                         zh_prev + lnf_n * zh_prev)
        below = zh_n < new_min
        zh_n = jnp.where(below, jnp.sqrt(zh_prev * new_min), zh_n)
        lnf_n = jnp.where(below, (zh_n - zh_prev) / zh_prev, lnf_n)
        above = zh_n > new_max
        zh_n = jnp.where(above, jnp.sqrt(zh_prev * new_max), zh_n)
        lnf_n = jnp.where(above, (zh_n - zh_prev) / zh_prev, lnf_n)
        bisect = jnp.abs(zeqn) >= 0.5 * absmin
        zh_new = jnp.where(bisect, zh_b, zh_n)
        lnf = jnp.where(bisect, lnf_b, lnf_n)
        new_absmin = jnp.minimum(jnp.abs(zeqn), absmin)
        step = ~done & ~exact                                      # this point iterates now
        zh = jnp.where(step, zh_new, zh)
        zh_min = jnp.where(step, new_min, zh_min)
        zh_max = jnp.where(step, new_max, zh_max)
        absmin = jnp.where(step, new_absmin, absmin)
        niter = niter + step.astype(jnp.int32)
        done = done | exact | (step & (jnp.abs(lnf) < PH_RDEL_TARGET))
        return zh, zh_min, zh_max, absmin, niter, done

    zh, _, _, _, niter, done = jax.lax.fori_loop(
        0, n_iter, body, (zh, zh_min, zh_max, absmin, niter, done))
    return jnp.where(done, zh, -1.0), niter


def solve_at_general(alk, dic, bor, po4, sil, so4, flu, K1, K2, Kb, Kw, Ks, Kf,
                     K1p, K2p, K3p, Ksi, *, n_iter: int = PH_MAXITER):
    """[H+] (total scale, mol/kg) from total alkalinity and DIC (``phsolvers.F90:429-630``).

    Returns ``(h, niter)``. Values are Fortran's per-point iterates (see the module
    docstring); the derivative of ``h`` is the implicit one of the root.
    """
    args = jnp.broadcast_arrays(*(jnp.asarray(x, dtype=jnp.float64) for x in (
        alk, dic, bor, po4, sil, so4, flu, K1, K2, Kb, Kw, Ks, Kf, K1p, K2p, K3p, Ksi)))
    h, niter = _solve_root(n_iter, *args)
    return h, niter.astype(jnp.int32)


def _solve_root_impl(n_iter, *args):
    h, niter = _solve_at_general_iter(n_iter, *args)
    return h, niter.astype(jnp.float64)     # float, so the custom JVP can give it a zero tangent


_solve_root = jax.custom_jvp(_solve_root_impl, nondiff_argnums=(0,))


@_solve_root.defjvp
def _solve_root_jvp(n_iter, primals, tangents):
    """Implicit derivative of the root: ``dh = -(dF/dtheta . dtheta) / (dF/dh)``; 0 where
    the solver failed (``h = -1``)."""
    h, niter = _solve_root_impl(n_iter, *primals)
    alk, dic, bor, po4, sil, so4, flu, *consts = primals
    ok = h > 0.0
    h_safe = jnp.where(ok, h, 1.0e-8)

    def resid(*theta):
        return _equation_at(theta[0], h_safe, *theta[1:])[0]

    _, dF = jax.jvp(resid, tuple(primals), tuple(tangents))
    dFdh = _equation_at(alk, h_safe, dic, bor, po4, sil, so4, flu, *consts)[1]
    return (h, niter), (jnp.where(ok, -dF / dFdh, 0.0), jnp.zeros_like(niter))


# --------------------------------------------------------------------------------------
# varsolver / vars_sprac
# --------------------------------------------------------------------------------------
class MocsyVars(NamedTuple):
    """Outputs of ``vars_sprac`` (``vars.F90:254-705``); 1e20 where mocsy masks the point."""
    ph: jax.Array
    pco2: jax.Array
    fco2: jax.Array
    co2: jax.Array
    hco3: jax.Array
    co3: jax.Array
    omega_a: jax.Array
    omega_c: jax.Array
    kspc: jax.Array
    betad: Optional[jax.Array]   # None unless ``betad=True`` (REcoM never reads it)
    rho_sw: jax.Array
    p: jax.Array
    tempis: jax.Array
    niter: jax.Array             # pH-solver iterations (diagnostic, int32; -1 where masked)


def _varsolver(tempis90, salt, ta, tc, pt, sit, k: Constants, patm, prb, drho, n_iter):
    """``varsolver.F90:7-247`` with ``optGAS='Pinsitu'``; returns the fields + niter."""
    h, niter = solve_at_general(ta, tc, k.Bt, pt, sit, k.St, k.Ft, k.K1, k.K2, k.Kb, k.Kw,
                                k.Ks, k.Kf, k.K1p, k.K2p, k.K3p, k.Ksi, n_iter=n_iter)
    hpos = h > 0.0
    ph = jnp.where(hpos, -1.0 * PR.log10(jnp.where(hpos, h, 1.0)), FILL)   # :156-160
    hso4 = k.St / (1.0 + k.Ks / (h / (1.0 + k.St / k.Ks)))
    hf = 1.0 / (1.0 + k.Kf / h)
    hsi = 1.0 / (1.0 + h / k.Ksi)
    h3 = h ** 3
    hpo4 = ((k.K1p * k.K2p * (h + 2.0 * k.K3p) - h3)
            / (h3 + k.K1p * h ** 2 + k.K1p * k.K2p * h + k.K1p * k.K2p * k.K3p))
    ab = k.Bt / (1.0 + h / k.Kb)
    aw = k.Kw / h - h / (1.0 + k.St / k.Ks)
    ac = ta + hso4 - sit * hsi - ab - aw + k.Ft * hf - pt * hpo4
    cu = (2.0 * tc - ac) / (2.0 + k.K1 / h)
    cb = k.K1 * cu / h
    cc = k.K2 * cb / h
    co2 = cu * drho
    hco3 = cb * drho
    co3 = cc * drho
    fco2 = cu * 1.0e6 / k.K0
    tk = 273.15 + tempis90
    tk0 = tk                                                       # Pinsitu (:203-207)
    phydro_atm = prb / 1.01325
    ptot = patm + phydro_atm
    b = -1636.75 + 12.0408 * tk0 - 0.0327957 * (tk0 * tk0) + 0.0000316528 * (tk0 * tk0 * tk0)
    delta = 57.7 - 0.118 * tk0
    xco2approx = fco2 * 1.0e-6
    xco2approx = xco2approx * jnp.exp(((1.0 - ptot) * 32.3) / (82.05736 * tk0))
    xc2 = (1.0 - xco2approx) ** 2
    fugcoeff = jnp.exp(ptot * (b + 2.0 * xc2 * delta) / (_RGAS_ATM * tk0))
    pco2 = fco2 / fugcoeff
    ca = (0.02128 / 40.078) * salt / 1.80655
    omega_a = (ca * cc) / k.Kspa
    omega_c = (ca * cc) / k.Kspc
    return ph, pco2, fco2, co2, hco3, co3, omega_a, omega_c, niter


def vars_sprac(temp, sal, alk, dic, sil, phos, patm, depth, lat, *,
               n_iter: int = PH_MAXITER, betad: bool = False,
               optCON=None, optT=None, optP=None, optB=None, optK1K2=None, optKf=None,
               optGAS=None, optS=None) -> MocsyVars:
    """The carbonate system from T, S, Alk, DIC, Si, PO4 (``vars.F90:254-705``).

    REcoM's interior call (``recom_carbonate_chemistry.F90:240-247``); all inputs element-wise
    and broadcast together. The ``opt*`` keywords exist only to reject non-REcoM options.
    ``betad=True`` also returns the Revelle factor, which costs two more pH solves (REcoM
    computes and discards it). Points with ``dic <= 0`` or ``dic >= 1e4`` (the mocsy guard,
    ``vars.F90:508``) come back as 1e20 in every field.
    """
    _check_options(dict(optCON=optCON, optT=optT, optP=optP, optB=optB, optK1K2=optK1K2,
                        optKf=optKf, optGAS=optGAS, optS=optS))
    temp, sal, alk, dic, sil, phos, patm, depth, lat = jnp.broadcast_arrays(
        *(jnp.asarray(x, dtype=jnp.float64)
          for x in (temp, sal, alk, dic, sil, phos, patm, depth, lat)))
    ok = (dic > 0.0) & (dic < 1.0e4)
    dic_s = jnp.where(ok, dic, 2.0)                                # safe lanes

    p = p80(depth, lat)                                            # :454
    tempis68, tempis90 = tpot_to_tinsitu(temp, sal, p)             # :466-473
    ssal = jnp.where(sal < 0.0, 0.0, sal)                          # :521-545
    sphos = jnp.where(phos < 0.0, 0.0, phos)
    ssil = jnp.where(sil < 0.0, 0.0, sil)
    sdic = jnp.where(dic_s < 0.0, 0.0, dic_s)
    salk = jnp.where(alk < 0.0, 0.0, alk)
    prb = p / 10.0                                                 # :555
    rho_sw = rho(ssal, tempis68, prb)                              # :581
    drho = rho_sw                                                  # :588 (mol/m3)
    ta = salk / drho                                               # :600-603
    tc = sdic / drho
    pt = sphos / drho
    sit = ssil / drho
    s = ssal
    k = constants(temp, s, patm, depth, lat)                       # :637-644
    ph, pco2, fco2, co2, hco3, co3, omega_a, omega_c, niter = _varsolver(
        tempis90, s, ta, tc, pt, sit, k, patm, prb, drho, n_iter)  # :646-651
    rf = None
    if betad:                                                      # :661-681
        dx = 0.1 * 1e-6
        pco2del = []
        for minusplus in (-1.0, 1.0):
            dicdel = tc + minusplus * dx / 2.0
            out = _varsolver(tempis90, s, ta, dicdel, pt, sit, k, patm, prb, drho, n_iter)
            pco2del.append(out[1])
        dicdel1, dicdel2 = tc - dx / 2.0, tc + dx / 2.0
        rf = (pco2del[1] - pco2del[0]) / (dicdel2 - dicdel1)
        rf = rf * tc / pco2del[1]      # mocsy divides by the LAST solve's pCO2 (dpco2), :680
        rf = jnp.where(ok, rf, FILL)
    fill = lambda v: jnp.where(ok, v, FILL)                        # :685-700
    return MocsyVars(
        ph=fill(ph), pco2=fill(pco2), fco2=fill(fco2), co2=fill(co2), hco3=fill(hco3),
        co3=fill(co3), omega_a=fill(omega_a), omega_c=fill(omega_c), kspc=fill(k.Kspc),
        betad=rf, rho_sw=fill(rho_sw), p=fill(p), tempis=fill(tempis90),
        niter=jnp.where(ok, niter, -1))


# --------------------------------------------------------------------------------------
# Gas exchange (gasx.F90)
# --------------------------------------------------------------------------------------
def pistonvel(windspeed, fice):
    """kw660 [m/s] = 0.251 U^2 (1 - Fice) [cm/h -> m/s] (Wanninkhof 2014),
    ``gasx.F90:428-470``."""
    a = 0.251
    xfac = 0.01 / 3600.0
    return a * windspeed ** 2 * (1.0 - fice) * xfac


def scco2(tc):
    """CO2 Schmidt number (Wanninkhof 2014), ``gasx.F90:527-542``."""
    return 2116.8 - 136.25 * tc + 4.7353 * tc ** 2 - 0.092307 * tc ** 3 + 0.0007555 * tc ** 4


def sco2(tc):
    """O2 Schmidt number (Wanninkhof 2014), ``gasx.F90:545-560``."""
    return 1920.4 - 135.6 * tc + 5.2122 * tc ** 2 - 0.10939 * tc ** 3 + 0.00093777 * tc ** 4


def vapress(temp, salt):
    """Vapor pressure of seawater [atm] (Weiss and Price 1980), ``gasx.F90:888-921``."""
    tk = 273.15 + temp
    return jnp.exp(24.4543 - 67.4509 * (100.0 / tk) - 4.8489 * jnp.log(tk / 100.0)
                   - 0.000544 * salt)


def x2pco2atm(xco2, temp, salt, patm):
    """Atmospheric pCO2 [uatm] = (Patm - pH2O) xCO2, ``gasx.F90:563-607``."""
    return (patm - vapress(temp, salt)) * xco2


def p2fco2(pco2, temp, patm, p):
    """fCO2 [uatm] from pCO2 with the fugacity coefficient, ``p2fCO2.F90:7-70``."""
    tk = 273.15 + temp
    prb = p / 10.0
    ptot = patm + prb / 1.01325
    b = -1636.75 + 12.0408 * tk - 0.0327957 * (tk * tk) + 0.0000316528 * (tk * tk * tk)
    delta = 57.7 - 0.118 * tk
    xco2approx = pco2 * 1.0e-6
    xc2 = (1.0 - xco2approx) ** 2
    fugcoeff = jnp.exp(ptot * (b + 2.0 * xc2 * delta) / (_RGAS_ATM * tk))
    return pco2 * fugcoeff


class FlxCO2(NamedTuple):
    """Outputs of ``flxco2`` (``gasx.F90:8-379``)."""
    co2flux: jax.Array   # air-to-sea CO2 flux [mol m-2 s-1]; 1e20 where masked
    co2ex: jax.Array     # = co2flux (REcoM's mocsy does not divide by dz1, gasx.F90:372)
    dpco2: jax.Array     # ocean minus atmosphere pCO2 [uatm]
    k0: jax.Array        # surface K0 at potential T, no pressure term [(mol/kg)/atm]
    carb: MocsyVars      # the surface carbonate system (vars_sprac at depth 0)


def flxco2(temp, sal, alk, dic, sil, phos, kw660, xco2, patm, lat, *,
           n_iter: int = PH_MAXITER, betad: bool = False, **opts) -> FlxCO2:
    """Air-sea CO2 flux + the surface carbonate system, ``gasx.F90:8-379``.

    REcoM's surface call (``recom_forcing.F90:222-226``). Its ``dz1`` and ``lon`` arguments do
    not reach any result with REcoM's options and are omitted. ``lat`` enters only through
    ``p80(0, lat)``, which is 0.
    """
    temp, sal, alk, dic, sil, phos, kw660, xco2, patm, lat = jnp.broadcast_arrays(
        *(jnp.asarray(x, dtype=jnp.float64)
          for x in (temp, sal, alk, dic, sil, phos, kw660, xco2, patm, lat)))
    depth0 = jnp.zeros_like(temp)                                  # gasx.F90:323
    carb = vars_sprac(temp, sal, alk, dic, sil, phos, patm, depth0, lat,
                      n_iter=n_iter, betad=betad, **opts)          # :327-330
    salprac = sal                                                  # :336-338 (Sprc)
    pco2atm = x2pco2atm(xco2, temp, salprac, patm)                 # :342
    fco2atm = p2fco2(pco2atm, temp, patm, depth0)                  # :346
    masked = carb.co3 == FILL                                      # :350
    tk = temp + 273.15
    invtk = 1.0 / tk
    kwco2 = kw660 * (660.0 / scco2(temp)) ** 0.5                   # :360
    tmp = 9345.17 * invtk - 60.2409 + 23.3585 * jnp.log(tk / 100.0)
    k0 = jnp.exp(tmp + salprac * (0.023517 - 0.00023656 * tk + 0.0047036e-4 * tk * tk))
    co2starair = k0 * fco2atm * 1.0e-6 * carb.rho_sw               # :367
    co2flux = kwco2 * (co2starair - carb.co2)
    dpco2 = carb.pco2 - pco2atm
    return FlxCO2(co2flux=jnp.where(masked, FILL, co2flux),
                  co2ex=jnp.where(masked, FILL, co2flux),
                  dpco2=jnp.where(masked, FILL, dpco2),
                  k0=jnp.where(masked, FILL, k0),   # Fortran leaves K0 unset there
                  carb=carb)


def o2sato(temp, sal):
    """O2 saturation at 1 atm [mol m-3] (Garcia and Gordon 1992), ``gasx.F90:924-999``."""
    a0, a1, a2, a3, a4, a5 = 2.00907, 3.22014, 4.05010, 4.94457, -2.56847e-1, 3.88767
    b0, b1, b2, b3 = -6.24523e-3, -7.37614e-3, -1.03410e-2, -8.17083e-3
    c0 = -4.88682e-7
    tt = 298.15 - temp
    tk = 273.15 + temp
    ts = jnp.log(tt / tk)
    ts2 = ts ** 2
    ts3 = ts ** 3
    ts4 = ts ** 4
    ts5 = ts ** 5
    tmp = (a0 + a1 * ts + a2 * ts2 + a3 * ts3 + a4 * ts4 + a5 * ts5
           + sal * (b0 + b1 * ts + b2 * ts2 + b3 * ts3)
           + c0 * (sal * sal))
    return jnp.exp(tmp) / 22391.6 * 1000.0


def o2flux(temp, sal, kw660, ppo, o2):
    """Air-sea O2 flux [mol m-2 s-1, air to sea], ``gasx.F90:1002-1082``."""
    kwo2 = kw660 * (660.0 / sco2(temp)) ** 0.5
    o2sat = o2sato(temp, sal) * ppo
    return kwo2 * (o2sat - o2)
