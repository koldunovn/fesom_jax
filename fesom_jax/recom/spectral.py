"""The spectral fork's light model (lane L of ``docs/plans/20261028-recom-spectral-port.md``): 13 wavebands, the
inherent optical properties of water, phytoplankton, CDOM and detritus, and the three-stream (Ed, Es, Eu) radiative
transfer of ``module REcoM_spectral`` (``src/int_recom/recom_modules.F90:2357-4544`` of ``sveta-loza/fesom2``
``fesom2.6_recom_tra_diags_spectral``, worktree ``/albedo/work/user/sloza/SPECTRAL/recom_spectral_r692``; M = that
file, F = ``recom_forcing.F90``, S = ``recom_sms.F90`` of the same tree).

What the fork does each step, per column (F:306-830, inside ``REcoM_Forcing`` before ``REcoM_sms``):

1. the chlorophyll ``max(tiny_chl, Chl_X)`` and carbon ``max(tiny_C_X, C_X)`` of the 2 or 4 phytoplankton groups, the
   particles ``DetC (+ DetZ2C)`` and the CDOM tracer (F:306-358);
2. the absorption ``a_k`` and scattering ``bt_k``/``bb_k`` per layer and band (F:596-667): water (``abw25par.dat``),
   CDOM (``cdomcoeff * CDOM * exp(-0.021 (lambda - 450))`` under ``RECOM_CALC_ACDOM``, else the ``aCDOM13amtVK2006.dat``
   spectrum), phytoplankton (``Chl_X * aphy_chl_X`` and ``C_X * bphy_chl_X`` with ``RECOM_BMASS``), detritus
   (``RECOM_CALC_APART``: ``part * 0.016 exp(-0.013 (lambda - 440))``, ``part * 0.345 (550/lambda)^0.5``, times
   ``bb_to_b``; else the per-particle ``optics_detritus_3bb.dat`` over ``1.06e-13 mmol C``), the optional background
   backscatter ``bb_bg443 (443/lambda)^bb_bg_slope`` and the floor ``darwin_bbmin``;
3. the surface irradiance: OASIM's direct ``Ed`` and diffuse ``Es`` per band (W m-2), already shaded by sea ice
   ``(1 - a_ice)`` by the caller (``recom_main.F90:429-430``); without OASIM ``Ed = wb_width * sf * SurfSW``,
   ``Es = tiny`` (F:469-485);
4. ``MONOD_RADTRANS_DIRECT`` (M:4070-4333): ``Ed`` integrated exactly, ``Es``/``Eu`` from the amplitudes of the exact
   two-mode solution, a ``2 kbot`` tridiagonal system solved by Gaussian elimination with pivoting
   (``SOLVE_TRIDIAGONAL_PIVOT``, M:4338-4419), skipped where both ``Ed`` and ``Es`` are below ``1e-4``; the scalar
   irradiance at the cell centre ``sqrt(Etop Ebot)`` of ``(rmud Ed + rmus Es + rmuu Eu) WtouEins`` per band
   (``tirrwq``, uEin m-2 s-1) and its band sum ``tirrq``;
5. what ``REcoM_sms`` reads (S:1457-1465, 1572-1602): ``PARave = sum_lambda PARw / WtouEins`` (W m-2, floored at
   ``tiny``), ``PARl = tirrq`` and ``alpha_I_X = 86400 sum_lambda alphachl_nl_X PARw`` with
   ``alphachl_nl_X = mQY_X * aphy_chl_ps_X``, ``mQY_X = sum (QYmax_X / 13) / WtouEins`` (``WAVEBANDS_INIT_VARI``,
   M:3607-3638).

Every sum over bands or groups is accumulated in the Fortran order (a Python loop over the 13 static bands), so the
gate against the fork's dumps holds at round-off. The solver is a pair of ``lax.scan`` sweeps over the padded
``2 (nl-1)`` rows (the rows beyond ``2 kbot`` are identity rows, which the pivoting elimination and the back
substitution pass through exactly), with every division guarded so the unselected pivot branch never divides by zero
(the forward values are unchanged; the reverse pass stays finite).

Not ported (``RecomSwitches`` keeps them refused in :data:`fesom_jax.recom.params.SUPPORTED`): ``RECOM_MARSHALL`` and
``RECOM_CALC_APHYT`` (the D1 photoinhibition tracers and the D1-dependent absorption), ``RECOM_CALC_REFLEC``,
``MONOD_RADTRANS_ITER`` (``darwin_radtrans_niter`` is the compile-time ``-2``, the direct solver) and
``DAR_RADTRANS_RMUS_PAR`` (a module logical, ``.false.``). ``RECOM_RADTRANS = .false.`` (the per-band Beer-Lambert of
F:497-590) is ported as :func:`beer_lambert` but has no twin dump yet.
"""

from __future__ import annotations

import os
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .params import SECONDS_PER_DAY, TINY, TINY_CHL, RecomParams, derived

TLAM = 13
PWAVES = tuple(range(400, 701, 25))                  # M:2804-2816 darwin_waves (the band "representative" values)

# ---- module constants of REcoM_spectral (M:2357-2700), not namelist entries in the fork ----
PLANCK = 6.6256e-34                                  # M:2822 [J s]
C_LIGHT = 2.998e8                                    # M:2823 [m s-1]
AVOGADRO = 6.023e23                                  # M:2825
RD = 1.5                                             # M:4164 Ackleson et al. 1994
RU = 3.0                                             # M:4165
DARWIN_RMUS = 1.204819277                            # M:2644 1/0.83
DARWIN_RMUU = 2.5                                    # M:2645 1/0.4
DARWIN_BBW = 0.5                                     # M:2646 backscatter to forward scattering ratio, water
DARWIN_BBMIN = 0.0002                                # M:2652 [m-1]
DARWIN_RADMOD_THRESH = 1e-4                          # M:2642 [W m-2]
DARWIN_SDOM = 0.021                                  # M:2588
DARWIN_LAMBDA_ACDOM = 450.0                          # M:2589
DARWIN_ACDOM_FAC = 0.2                               # M:2594 (RECOM_CALC_ACDOM without the CDOM tracer)
DARWIN_SAPAR = 0.013                                 # M:2600
DARWIN_LAMBDA_APART = 440.0                          # M:2601
APARCOEFF = 0.016                                    # M:2603
DARWIN_SBPAR = 0.5                                   # M:2605
DARWIN_LAMBDA_BPART = 550.0                          # M:2606
BPARCOEFF = 0.345                                    # M:2608
DARWIN_PART_SIZE_P = 1.06e-13                        # M:2617 [mmol C particle-1]
AP_MISSING = (9.9e2, 8.9e2, 8.9e2, 8.9e2)            # M:3527-3570 sentinel spectra of a group the file lacks


def _band_edges():
    """``darwin_wavebands`` (M:2884-2897): the first and last edges at the first and last band values, the others
    half-way; ``wb_width`` and ``wb_totalWidth`` (M:2901-2905)."""
    pw = [float(p) for p in PWAVES]
    edges = [pw[0]] + [0.5 * (pw[i - 1] + pw[i]) for i in range(1, TLAM)] + [pw[-1]]
    width = np.array([edges[i + 1] - edges[i] for i in range(TLAM)], np.float64)
    total = 0.0
    for w in width:                                   # the Fortran accumulation order
        total = total + float(w)
    return np.array(edges, np.float64), width, total


def wtou_eins() -> np.ndarray:
    """``WtouEins`` (M:2822-2836): W to uEin s-1 per band, ``1e6 * lambda[m] * (1 / (h c)) * (1 / N_A)``."""
    hc = 1.0 / (PLANCK * C_LIGHT)
    oavo = 1.0 / AVOGADRO
    hcoavo = hc * oavo
    out = np.zeros(TLAM, np.float64)
    for i, p in enumerate(PWAVES):
        rlamm = float(p) * 1.0e-9
        out[i] = 1.0e6 * rlamm * hcoavo
    return out


WB_EDGES, WB_WIDTH, WB_TOTAL_WIDTH = _band_edges()
WTOUEINS = wtou_eins()


# ---- the optics tables (WAVEBANDS_INIT_FIXED / _VARI, M:2736-3638) --------------------------------------------------

class SpectralTables(NamedTuple):
    """The fixed per-band tables of one run (numpy f64; a pytree of leaves when passed to a jitted program)."""
    aw: np.ndarray            # (13) water absorption [m-1]              abw25par.dat
    bw: np.ndarray            # (13) water scattering [m-1]
    aphy_chl: np.ndarray      # (4, 13) Chl-specific absorption per group [m2 mg-1], groups Others, Diatom, Cocco, Phaeo
    aphy_chl_ps: np.ndarray   # (4, 13) photosynthetic pigments only
    bphy_chl: np.ndarray      # (4, 13) scattering (per mg Chl, or per mmol C with RECOM_BMASS)
    bbphy_chl: np.ndarray     # (4, 13) backscattering
    alphachl_nl: np.ndarray   # (4, 13) mQY_X * aphy_chl_ps_X
    alpha_mean: np.ndarray    # (4,) width-weighted mean of alphachl_nl (the Ek diagnostic only)
    sf: np.ndarray            # (13) PAR-normalised surface spectrum per nm (no OASIM only), else zeros
    acdom: np.ndarray         # (13) constant CDOM absorption [m-1] (no RECOM_CALC_ACDOM), else zeros
    excdom: np.ndarray        # (13) exp(-Sdom (lambda - 450))
    apart_P: np.ndarray       # (13) per mmol C particle absorption (no RECOM_CALC_APART), else zeros
    bpart_P: np.ndarray
    bbpart_P: np.ndarray
    exapar: np.ndarray        # (13) exp(-Sapar (lambda - 440))
    exbpar: np.ndarray        # (13) (550 / lambda)^Sbpar
    nabp_read: int            # phytoplankton spectra blocks the file supplied


def _data_lines(path, header: int, n: int, label: str):
    with open(path) as fh:
        lines = fh.read().splitlines()
    body = lines[header:header + n]
    if len(body) < n:
        raise ValueError(f"{label}: {path} has {len(body)} data lines after {header} header lines, {n} needed")
    return body, lines


def _check_lambda(got, ilam, label, path):
    if int(got) != PWAVES[ilam]:
        raise ValueError(f"{label}: wavelength {int(got)} of {path} at band {ilam + 1} is not {PWAVES[ilam]}")


def read_water(path) -> tuple[np.ndarray, np.ndarray]:
    """``abw25par.dat`` (M:2949-2973): six header lines, then ``lambda aw bw`` per band."""
    body, _ = _data_lines(path, 6, TLAM, "water spectrum")
    aw = np.zeros(TLAM); bw = np.zeros(TLAM)
    for i, ln in enumerate(body):
        lam, saw, sbw = ln.split()[:3]
        _check_lambda(lam, i, "water spectrum", path)
        aw[i] = float(saw); bw[i] = float(sbw)
    return aw, bw


def read_phyto(path, *, nonspectral_bb_ratio: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """The phytoplankton optics (M:2995-3121): six header lines, then up to four blocks of one title line and 13
    ``lambda ap ap_ps bp [bbp]`` lines (Others, Diatom, Coccolithophore, Phaeocystis). Returns ``(ap, ap_ps, bp, bbp,
    nabp_read)`` with the blocks the file lacks left at zero. ``nonspectral_bb_ratio``
    (``DAR_NONSPECTRAL_BACKSCATTERING_RATIO``) reads four columns and leaves ``bbp`` zero."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    ap = np.zeros((4, TLAM)); ap_ps = np.zeros((4, TLAM)); bp = np.zeros((4, TLAM)); bbp = np.zeros((4, TLAM))
    pos = 6
    nabp_read = 0
    for nabp in range(4):
        if pos >= len(lines):
            break                                                     # clean end of file
        pos += 1                                                      # the block title
        if pos + TLAM > len(lines):
            raise ValueError(f"phyto optics: block {nabp + 1} of {path} is truncated (one title + {TLAM} lines)")
        for i in range(TLAM):
            tok = lines[pos + i].split()
            if len(tok) < (4 if nonspectral_bb_ratio else 5):
                raise ValueError(f"phyto optics: bad record {lines[pos + i]!r} in {path}")
            _check_lambda(tok[0], i, "phyto spectrum", path)
            ap[nabp, i] = float(tok[1]); ap_ps[nabp, i] = float(tok[2]); bp[nabp, i] = float(tok[3])
            if not nonspectral_bb_ratio:
                bbp[nabp, i] = float(tok[4])
        pos += TLAM
        nabp_read = nabp + 1
    if nabp_read < 2:
        raise ValueError(f"phyto optics: {path} supplies {nabp_read} spectra block(s); at least Others and Diatom")
    return ap, ap_ps, bp, bbp, nabp_read


def read_surface_spectrum(path) -> np.ndarray:
    """``surfspec_13amt6.dat`` (M:3125-3151): three header lines, then ``lambda sf``."""
    body, _ = _data_lines(path, 3, TLAM, "surface spectrum")
    sf = np.zeros(TLAM)
    for i, ln in enumerate(body):
        lam, ssf = ln.split()[:2]
        _check_lambda(lam, i, "surface spectrum", path)
        sf[i] = float(ssf)
    return sf


def read_acdom(path) -> np.ndarray:
    """``aCDOM13amtVK2006.dat`` (M:3157-3174): six header lines, then one absorption per band."""
    body, _ = _data_lines(path, 6, TLAM, "CDOM spectrum")
    return np.array([float(ln.split()[0]) for ln in body], np.float64)


def read_particles(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``optics_detritus_3bb.dat`` (M:3240-3268): six header lines, then ``lambda apart bpart bbpart`` per particle;
    returns the per-mmol-C ``apart_P, bpart_P, bbpart_P`` (over ``darwin_part_size_P``)."""
    body, _ = _data_lines(path, 6, TLAM, "particle spectrum")
    ap = np.zeros(TLAM); bp = np.zeros(TLAM); bbp = np.zeros(TLAM)
    for i, ln in enumerate(body):
        lam, sap, sbp, sbbp = ln.split()[:4]
        _check_lambda(lam, i, "particle spectrum", path)
        ap[i] = float(sap) / DARWIN_PART_SIZE_P
        bp[i] = float(sbp) / DARWIN_PART_SIZE_P
        bbp[i] = float(sbbp) / DARWIN_PART_SIZE_P
    return ap, bp, bbp


def load_tables(sw, p: RecomParams, optics_dir) -> SpectralTables:
    """``WAVEBANDS_INIT_FIXED`` + ``WAVEBANDS_INIT_VARI`` for the switches ``sw`` (:class:`RecomSwitches`) and the
    parameters ``p`` (``QYmax_X``, ``darwin_bbphy_X``), the ``.dat`` files named by the switches in ``optics_dir``."""
    d = str(optics_dir)
    aw, bw = read_water(os.path.join(d, sw.darwin_waterabsorbFile))
    ratio = bool(getattr(sw, "DAR_NONSPECTRAL_BACKSCATTERING_RATIO", False))
    bbphy_ratio = np.array([p.darwin_bbphy_1, p.darwin_bbphy_2, p.darwin_bbphy_3, p.darwin_bbphy_4], np.float64)
    if ratio and float(np.max(np.abs(bbphy_ratio))) <= 0.0:
        raise ValueError("DAR_NONSPECTRAL_BACKSCATTERING_RATIO with every darwin_bbphy zero (M:2988-2994)")
    ap, ap_ps, bp, bbp, nabp_read = read_phyto(os.path.join(d, sw.darwin_phytoabsorbFile), nonspectral_bb_ratio=ratio)
    if sw.enable_coccos and nabp_read < 4:
        raise ValueError(f"enable_coccos needs four phytoplankton spectra, the file supplies {nabp_read} (M:3097-3109)")
    sf = np.zeros(TLAM)
    if not sw.OASIM:
        sf = read_surface_spectrum(os.path.join(d, sw.darwin_surfacespecFile))
    acdom = np.zeros(TLAM)
    lam = np.array([float(q) for q in PWAVES])
    excdom = np.exp(-DARWIN_SDOM * (lam - DARWIN_LAMBDA_ACDOM))
    if not sw.RECOM_CALC_ACDOM:
        acdom = read_acdom(os.path.join(d, sw.darwin_acdomFile)) if sw.darwin_acdomFile.strip() else np.zeros(TLAM)
    apart_P = bpart_P = bbpart_P = np.zeros(TLAM)
    exapar = np.exp(-DARWIN_SAPAR * (lam - DARWIN_LAMBDA_APART))
    exbpar = (DARWIN_LAMBDA_BPART / lam) ** DARWIN_SBPAR
    if not sw.RECOM_CALC_APART:
        if sw.darwin_particleabsorbFile.strip():
            apart_P, bpart_P, bbpart_P = read_particles(os.path.join(d, sw.darwin_particleabsorbFile))
    # WAVEBANDS_INIT_VARI (M:3462-3575): the per-group spectra, the sentinels for groups the file lacks
    ngroup = 4 if sw.enable_coccos else 2
    aphy = np.zeros((4, TLAM)); aphy_ps = np.zeros((4, TLAM)); bphy = np.zeros((4, TLAM)); bbphy = np.zeros((4, TLAM))
    for g in range(ngroup):
        if g < nabp_read:
            aphy[g] = ap[g]; aphy_ps[g] = ap_ps[g]; bphy[g] = bp[g]
            bbphy[g] = bp[g] * bbphy_ratio[g] if ratio else bbp[g]
        else:
            aphy[g] = aphy_ps[g] = bphy[g] = bbphy[g] = AP_MISSING[g]
    # the spectral PI slope (M:3597-3638)
    qy = [p.QYmax, p.QYmax_d, p.QYmax_cocco, p.QYmax_phaeo]
    alphachl = np.zeros((4, TLAM)); alpha_mean = np.zeros(4)
    for g in range(4):
        mqy = 0.0
        for nl in range(TLAM):
            mqy = mqy + ((float(qy[g]) / float(TLAM)) / WTOUEINS[nl])
        mqy = max(TINY, mqy)
        if g < 2 or sw.enable_coccos:
            alphachl[g] = mqy * aphy_ps[g]
        cu = 0.0
        for nl in range(TLAM):
            cu = cu + WB_WIDTH[nl] * alphachl[g, nl]
        alpha_mean[g] = cu / WB_TOTAL_WIDTH if WB_TOTAL_WIDTH > 0.0 else 0.0
    return SpectralTables(aw=aw, bw=bw, aphy_chl=aphy, aphy_chl_ps=aphy_ps, bphy_chl=bphy, bbphy_chl=bbphy,
                          alphachl_nl=alphachl, alpha_mean=alpha_mean, sf=sf, acdom=acdom, excdom=excdom,
                          apart_P=apart_P, bpart_P=bpart_P, bbpart_P=bbpart_P, exapar=exapar, exbpar=exbpar,
                          nabp_read=int(nabp_read))


# ---- the inherent optical properties (F:306-358, 596-667) ----------------------------------------------------------

class Iops(NamedTuple):
    a_k: jax.Array            # (N, nl1, 13) total absorption [m-1]
    bt_k: jax.Array           # (N, nl1, 13) total scattering
    bb_k: jax.Array           # (N, nl1, 13) backscattering
    actot: jax.Array          # (N, nl1, 13) phytoplankton absorption (diagnostic)
    acdom_k: jax.Array        # (N, nl1, 13)
    apart_k: jax.Array        # (N, nl1, 13)


def column_iops(bgc, tab: SpectralTables, p: RecomParams, sw, lay) -> Iops:
    """``a_k``, ``bt_k``, ``bb_k`` of every layer and band from the BGC stack ``bgc`` ``(N, nl1, ntr)`` (the state at
    the start of the step, as ``REcoM_Forcing`` reads it): the clamps of F:323-335, the CDOM/particle absorption of
    F:360-458 and the totals of F:600-658. The sums run in the Fortran order."""
    dz = derived(p)
    f64 = jnp.float64
    sv = lambda i: jnp.asarray(bgc[..., i], f64)                                        # noqa: E731
    coccos = bool(sw.enable_coccos)
    chl = [jnp.maximum(TINY_CHL, sv(lay.ipchl)), jnp.maximum(TINY_CHL, sv(lay.idchl))]
    carbon = [jnp.maximum(dz.tiny_C, sv(lay.iphyc)), jnp.maximum(dz.tiny_C_d, sv(lay.idiac))]
    if coccos:
        chl += [jnp.maximum(TINY_CHL, sv(lay.icchl)), jnp.maximum(TINY_CHL, sv(lay.iphachl))]
        carbon += [jnp.maximum(dz.tiny_C_c, sv(lay.icocc)), jnp.maximum(dz.tiny_C_p, sv(lay.iphac))]
    part = sv(lay.idetc)
    if lay.enable_3zoo2det:
        part = part + sv(lay.idetz2c)
    ngroup = len(chl)
    T = lambda a: jnp.asarray(a, f64)[None, None, :]                                    # noqa: E731
    # CDOM absorption
    if sw.RECOM_CALC_ACDOM:
        if sw.RECOM_CDOM:
            acdom_k = (p.cdomcoeff * sv(lay.icdom))[..., None] * T(tab.excdom)          # MONOD_ACDOM, M:3680-3686
        else:                                                                           # MONOD_no_ACDOM, M:3747-3759
            nla = PWAVES.index(int(DARWIN_LAMBDA_ACDOM))
            actot450 = 0.0
            for g in range(ngroup):
                actot450 = actot450 + chl[g] * float(tab.aphy_chl[g, nla])
            atot450 = float(tab.aw[nla]) + actot450
            acdom_k = (DARWIN_ACDOM_FAC * atot450)[..., None] * T(tab.excdom)
    else:
        acdom_k = jnp.broadcast_to(T(tab.acdom), part.shape + (TLAM,))
    # particles (F:437-458)
    if sw.RECOM_CALC_APART:
        apart_k = (part * APARCOEFF)[..., None] * T(tab.exapar)
        bpart_k = (part * BPARCOEFF)[..., None] * T(tab.exbpar)
        bbpart_k = (part * BPARCOEFF)[..., None] * T(tab.exbpar) * p.bb_to_b
    else:
        apart_k = part[..., None] * T(tab.apart_P)
        bpart_k = part[..., None] * T(tab.bpart_P)
        bbpart_k = part[..., None] * T(tab.bbpart_P)
    # the totals (F:600-658)
    actot = jnp.zeros_like(apart_k); bctot = jnp.zeros_like(apart_k); bbctot = jnp.zeros_like(apart_k)
    scat = carbon if sw.RECOM_BMASS else chl
    for g in range(ngroup):
        actot = actot + chl[g][..., None] * T(tab.aphy_chl[g])
        bctot = bctot + scat[g][..., None] * T(tab.bphy_chl[g])
        bbctot = bbctot + scat[g][..., None] * T(tab.bbphy_chl[g])
    a_k = T(tab.aw) + acdom_k + actot + apart_k
    bt_k = T(tab.bw) + bctot + bpart_k
    bb_k = DARWIN_BBW * T(tab.bw) + bbctot + bbpart_k
    lam = jnp.asarray([float(q) for q in PWAVES], f64)                                  # F:654-656: bb_bg443 > 0 only
    bb_k = jnp.where(p.bb_bg443 > 0.0, bb_k + p.bb_bg443 * (443.0 / lam)[None, None, :] ** p.bb_bg_slope, bb_k)
    bb_k = jnp.maximum(DARWIN_BBMIN, bb_k)
    return Iops(a_k=a_k, bt_k=bt_k, bb_k=bb_k, actot=actot, acdom_k=acdom_k, apart_k=apart_k)


# ---- MONOD_RADTRANS_DIRECT (M:4070-4333) ----------------------------------------------------------------------------

class RadTrans(NamedTuple):
    """Per node, layer and band (``(N, nl1, 13)``): the irradiances [W m-2] and the scalar irradiance [uEin m-2 s-1]."""
    Edtop: jax.Array
    Edbot: jax.Array          # Edz
    Estop: jax.Array
    Esbot: jax.Array          # Esz
    Eutop: jax.Array
    Eubot: jax.Array          # Euz
    amp1: jax.Array           # c1 (downward decreasing mode)
    amp2: jax.Array           # c2
    tirrwq: jax.Array         # (N, nl1, 13) scalar irradiance per band
    tirrq: jax.Array          # (N, nl1) its band sum (PARl)


def _safe_div(num, den):
    return num / jnp.where(den == 0.0, 1.0, den)


def solve_tridiagonal_pivot(a, b, c, y):
    """``SOLVE_TRIDIAGONAL_PIVOT`` (M:4338-4419) on the last axis: ``a`` the lower, ``b`` the main, ``c`` the upper
    diagonal, ``y`` the right-hand side; returns ``x``. Gaussian elimination with row pivoting as two
    fixed-structure ``lax.scan`` sweeps; rows that are identity rows pass through unchanged, so a system of ``n``
    unknowns padded to ``2 (nl-1)`` rows gives the Fortran result on its first ``n`` entries."""
    n = a.shape[-1]
    a = jnp.moveaxis(a, -1, 0); b = jnp.moveaxis(b, -1, 0); c = jnp.moveaxis(c, -1, 0); y = jnp.moveaxis(y, -1, 0)
    c = c.at[n - 1].set(0.0)                                                            # M:4384 c3d(n) = 0

    def forward(carry, xs):
        bk, ck, yk = carry                                                              # row k as modified so far
        a1, b1, c1, y1 = xs                                                             # row k+1
        noswap = jnp.abs(bk) >= jnp.abs(a1)
        rec = _safe_div(1.0, jnp.where(noswap, bk, 1.0))
        ck_n = ck * rec; yk_n = yk * rec
        b1_n = b1 - a1 * ck_n; y1_n = y1 - a1 * yk_n
        rec_s = _safe_div(1.0, jnp.where(noswap, 1.0, a1))
        ck_s = b1 * rec_s; dk_s = c1 * rec_s; yk_s = y1 * rec_s
        b1_s = ck - bk * ck_s; c1_s = -bk * dk_s; y1_s = yk - bk * yk_s
        ck_o = jnp.where(noswap, ck_n, ck_s); yk_o = jnp.where(noswap, yk_n, yk_s); dk_o = jnp.where(noswap, 0.0, dk_s)
        b1_o = jnp.where(noswap, b1_n, b1_s); c1_o = jnp.where(noswap, c1, c1_s); y1_o = jnp.where(noswap, y1_n, y1_s)
        return (b1_o, c1_o, y1_o), (ck_o, yk_o, dk_o)

    (bn, cn, yn), (c_el, y_el, d_el) = jax.lax.scan(forward, (b[0], c[0], y[0]), (a[1:], b[1:], c[1:], y[1:]))
    yn = yn * _safe_div(1.0, bn)                                                        # M:4411-4412
    c_all = jnp.concatenate([c_el, cn[None]], axis=0)
    d_all = jnp.concatenate([d_el, jnp.zeros_like(cn)[None]], axis=0)
    y_all = jnp.concatenate([y_el, yn[None]], axis=0)

    def backward(carry, xs):
        y1, y2 = carry                                                                  # x[k+1], x[k+2]
        yk, ck, dk = xs
        xk = yk - ck * y1 - dk * y2
        return (xk, y1), xk

    zero = jnp.zeros_like(yn)
    _, xs_rev = jax.lax.scan(backward, (yn, zero), (y_all[:-1][::-1], c_all[:-1][::-1], d_all[:-1][::-1]))
    x = jnp.concatenate([xs_rev[::-1], yn[None]], axis=0)
    return jnp.moveaxis(x, 0, -1)


def radtrans_direct(H, rmud, Edsf, Essf, a_k, bt_k, bb_k) -> RadTrans:
    """``MONOD_RADTRANS_DIRECT`` for every column: ``H`` ``(N, nl1)`` layer thickness (0 below the bottom), ``rmud``
    ``(N,)`` the inverse cosine of the direct beam below the surface, ``Edsf``/``Essf`` ``(N, 13)`` the sub-surface
    direct/diffuse irradiance, the IOPs ``(N, nl1, 13)``. ``kbot`` is the deepest layer with ``H > 0`` (M:4181-4185,
    ``darwin_radtrans_kmax = nl-1``); bands with both ``Edsf, Essf < darwin_radmodThresh`` are skipped (zero)."""
    f64 = jnp.float64
    H = jnp.asarray(H, f64); N, nl1 = H.shape
    rmus, rmuu = DARWIN_RMUS, DARWIN_RMUU
    rmud = jnp.asarray(rmud, f64)[:, None, None]
    k = jnp.arange(nl1)
    kbot = jnp.max(jnp.where(H > 0.0, k[None, :] + 1, 0), axis=1)                      # (N,) layers 0..kbot-1 wet
    wet = (k[None, :] < kbot[:, None])[..., None]                                       # (N, nl1, 1)
    zd = H[..., None]
    cd = (a_k + bt_k) * rmud
    au = a_k * rmuu
    Bu = RU * bb_k * rmuu
    Cu = au + Bu
    as_ = a_k * rmus
    Bs = RD * bb_k * rmus
    Cs = as_ + Bs
    Bd = bb_k * rmud
    Fd = (bt_k - bb_k) * rmud
    bquad = Cs + Cu
    D = 0.5 * (bquad + jnp.sqrt(jnp.maximum(bquad * bquad - 4.0 * Bs * Bu, 0.0)))
    kappa1 = D - Cs
    kappa2 = Cs - Bs * Bu / D
    r1 = Bu / D
    r2 = Bs / D
    denom = (cd - Cs) * (cd + Cu) + Bs * Bu
    x = -((cd + Cu) * Fd + Bu * Bd) / denom
    y = (-Bs * Fd + (cd - Cs) * Bd) / denom
    ed = jnp.exp(-cd * zd)
    e1 = jnp.exp(-kappa1 * zd)
    e2 = jnp.exp(-kappa2 * zd)
    # Ed integrated exactly (M:4219-4224): Edtop(k+1) = Edbot(k) = Edtop(k) ed(k), layer by layer
    Ed0 = jnp.asarray(Edsf, f64)

    def ed_step(top, edk):
        bot = top * edk
        return bot, (top, bot)
    _, (Edtop, Edbot) = jax.lax.scan(ed_step, Ed0, jnp.moveaxis(ed, 1, 0))
    Edtop = jnp.moveaxis(Edtop, 0, 1) * wet
    Edbot = jnp.moveaxis(Edbot, 0, 1) * wet
    # the tridiagonal system (M:4232-4256), rows 0..2 nl1 - 1, variables c2(0), c1(0), c2(1), ...
    pad = lambda v: jnp.concatenate([v, v[:, -1:]], axis=1)                             # noqa: E731  v[k+1] at k = nl1-1
    r1n, r2n, xn, yn, e1n = (pad(v)[:, 1:] for v in (r1, r2, x, y, e1))
    a_even = (1.0 - r2 * r1n) * e2
    b_even = r1 - r1n
    c_even = -1.0 + r2n * r1n
    y_even = (xn - x - r1n * (yn - y)) * Edbot
    a_odd = 1.0 - r1 * r2
    b_odd = r2 - r2n
    c_odd = (-1.0 + r1n * r2) * e1n
    y_odd = (yn - y - r2 * (xn - x)) * Edbot
    # row 2k (0-based, k = 0..nl1-1) = Fortran row 2k+1: the continuity row "2k+1" of layer k, except row 0 (top b.c.)
    # row 2k+1 = Fortran row 2k+2 = the continuity row "2(k+1)" of layer k, except the last (bottom b.c.)
    shift = lambda v: jnp.concatenate([jnp.zeros_like(v[:, :1]), v[:, :-1]], axis=1)   # noqa: E731  v[k-1]
    A = jnp.stack([shift(a_odd), a_even], axis=2).reshape(N, 2 * nl1, TLAM)
    B = jnp.stack([shift(b_odd), b_even], axis=2).reshape(N, 2 * nl1, TLAM)
    C = jnp.stack([shift(c_odd), c_even], axis=2).reshape(N, 2 * nl1, TLAM)
    Y = jnp.stack([shift(y_odd), y_even], axis=2).reshape(N, 2 * nl1, TLAM)
    row = jnp.arange(2 * nl1)[None, :, None]
    last = (2 * kbot - 1)[:, None, None]
    top = row == 0
    bot = row == last
    beyond = row > last
    A = jnp.where(top | bot | beyond, 0.0, A)
    B = jnp.where(top | bot | beyond, 1.0, B)
    C = jnp.where(top, e1[:, :1] * r1[:, :1], jnp.where(bot | beyond, 0.0, C))
    Y = jnp.where(top, (jnp.asarray(Essf, f64) - x[:, 0] * Ed0)[:, None, :], jnp.where(bot | beyond, 0.0, Y))
    sol = solve_tridiagonal_pivot(jnp.moveaxis(A, 1, -1), jnp.moveaxis(B, 1, -1), jnp.moveaxis(C, 1, -1),
                                  jnp.moveaxis(Y, 1, -1))                               # (N, 13, 2 nl1)
    sol = jnp.moveaxis(sol, -1, 1).reshape(N, nl1, 2, TLAM)
    c2 = sol[:, :, 0]; c1 = sol[:, :, 1]
    Estop = (c2 + r1 * e1 * c1 + x * Edtop) * wet
    Esbot = (e2 * c2 + r1 * c1 + x * Edbot) * wet
    Eutop = (r2 * c2 + e1 * c1 + y * Edtop) * wet
    Eubot = (r2 * e2 * c2 + c1 + y * Edbot) * wet
    amp1 = c1 * wet; amp2 = c2 * wet
    # the first dry layer takes the bottom values of the last wet one (M:4268-4271)
    at_kbot = (k[None, :] == kbot[:, None])[..., None]
    Estop = jnp.where(at_kbot, shift(Esbot), Estop)
    Eutop = jnp.where(at_kbot, shift(Eubot), Eutop)
    lit = ((jnp.asarray(Edsf, f64) >= DARWIN_RADMOD_THRESH) | (jnp.asarray(Essf, f64) >= DARWIN_RADMOD_THRESH))[:, None, :]
    has_bot = (kbot > 0)[:, None, None]
    on = lit & has_bot
    Edtop, Edbot, Estop, Esbot, Eutop, Eubot, amp1, amp2 = (jnp.where(on, v, 0.0)
                                                            for v in (Edtop, Edbot, Estop, Esbot, Eutop, Eubot, amp1, amp2))
    # scalar irradiance in quanta (M:4287-4327), every layer
    W = jnp.asarray(WTOUEINS, f64)[None, None, :]
    Etopwq = (rmud * Edtop + rmus * Estop + rmuu * Eutop) * W
    Ebotwq = (rmud * Edbot + rmus * Esbot + rmuu * Eubot) * W
    tirrwq = jnp.sqrt(Etopwq * Ebotwq)
    tirrq = jnp.zeros((N, nl1), f64)
    for nl in range(TLAM):                                                              # darwin_PAR_ilamLo..Hi = 1..13
        tirrq = tirrq + tirrwq[..., nl]
    return RadTrans(Edtop=Edtop, Edbot=Edbot, Estop=Estop, Esbot=Esbot, Eutop=Eutop, Eubot=Eubot, amp1=amp1, amp2=amp2,
                    tirrwq=tirrwq, tirrq=tirrq)


def beer_lambert(H, Edsf, Essf, a_k):
    """``RECOM_RADTRANS = .false.`` (F:497-590): per-band attenuation ``PARwdn = PARwup exp(-a_k H)`` with the layer
    value ``sqrt(PARwup PARwdn)``, in quanta; returns ``(PARw (N, nl1, 13), PARl (N, nl1))``. Layers with ``H <= 0``
    (the fork tests ``zNodes(k) > 0``) are zero and do not attenuate."""
    f64 = jnp.float64
    W = jnp.asarray(WTOUEINS, f64)[None, :]
    up0 = W * (jnp.asarray(Edsf, f64) + jnp.asarray(Essf, f64))
    H = jnp.asarray(H, f64)

    def step(up, xs):
        hk, ak = xs
        wetk = (hk > 0.0)[:, None]
        dn = up * jnp.exp(-ak * hk[:, None])
        parw = jnp.where(wetk, jnp.sqrt(up * dn), 0.0)
        return jnp.where(wetk, dn, up), parw
    _, parw = jax.lax.scan(step, up0, (jnp.moveaxis(H, 1, 0), jnp.moveaxis(a_k, 1, 0)))
    parw = jnp.moveaxis(parw, 0, 1)
    parl = jnp.zeros(H.shape, f64)
    for nl in range(TLAM):
        parl = parl + parw[..., nl]
    return parw, parl


# ---- what the sources and sinks read (S:1457-1465, 1572-1602) -------------------------------------------------------

class SpectralLight(NamedTuple):
    """The per-layer light the fork's ``REcoM_sms`` reads in a spectral build (all ``(N, nl1)`` but ``PARw``)."""
    PARw: jax.Array           # (N, nl1, 13) scalar irradiance per band [uEin m-2 s-1] (PARwlocal)
    PARl: jax.Array           # band sum [uEin m-2 s-1] (PARlocal)
    PARave: jax.Array         # sum PARw / WtouEins [W m-2], >= tiny (S:1458-1463)
    alpha_I: jax.Array        # 86400 sum alphachl_nl PARw, per group (S:1577-1602) [d-1 (mmol C mg Chl-1)...]
    alpha_I_dia: jax.Array
    alpha_I_cocco: jax.Array
    alpha_I_phaeo: jax.Array


def sms_light(parw, parl, tab: SpectralTables) -> SpectralLight:
    f64 = jnp.float64
    parw = jnp.asarray(parw, f64)
    PARave = jnp.zeros(parw.shape[:-1], f64)
    for nl in range(TLAM):
        PARave = PARave + (parw[..., nl] / float(WTOUEINS[nl]))
    PARave = jnp.maximum(TINY, PARave)
    al = []
    for g in range(4):
        s = jnp.zeros(parw.shape[:-1], f64)
        for nl in range(TLAM):
            s = s + float(tab.alphachl_nl[g, nl]) * parw[..., nl]
        al.append(s * SECONDS_PER_DAY)
    return SpectralLight(PARw=parw, PARl=jnp.asarray(parl, f64), PARave=PARave, alpha_I=al[0], alpha_I_dia=al[1],
                         alpha_I_cocco=al[2], alpha_I_phaeo=al[3])


def surface_irradiance(oasim_ed, oasim_es, a_ice, surf_sw, tab: SpectralTables, sw):
    """``Edwsf``/``Eswsf`` ``(N, 13)`` [W m-2] below the surface (F:469-485): OASIM's fields shaded by ``(1 - a_ice)``
    (``recom_main.F90:429-430``), or without OASIM ``wb_width * sf * SurfSW`` and ``Es = tiny``."""
    f64 = jnp.float64
    if sw.OASIM:
        shade = (1.0 - jnp.asarray(a_ice, f64))[:, None]
        return jnp.asarray(oasim_ed, f64) * shade, jnp.asarray(oasim_es, f64) * shade
    ed = (jnp.asarray(WB_WIDTH, f64) * jnp.asarray(tab.sf, f64))[None, :] * jnp.asarray(surf_sw, f64)[:, None]
    return ed, jnp.full_like(ed, TINY)


def rmud_of(cos_ai):
    """``rmud = max(min(1 / cosAI, 1.5), 0)`` (F:489-493)."""
    c = jnp.asarray(cos_ai, jnp.float64)
    rmudl = 1.0 / jnp.where(c == 0.0, TINY, c)
    return jnp.maximum(jnp.minimum(rmudl, 1.5), 0.0)


class SpectralStep(NamedTuple):
    light: SpectralLight
    rt: Optional[RadTrans]    # None under RECOM_RADTRANS = .false.
    iops: Iops


def spectral_step(bgc, hnode, cos_ai, Edsf, Essf, tab: SpectralTables, p: RecomParams, sw, lay) -> SpectralStep:
    """The whole spectral light of one step from the BGC stack ``bgc`` ``(N, nl1, ntr)``, the thicknesses ``hnode``
    ``(N, nl1)``, ``cos_ai`` ``(N,)`` and the sub-surface irradiances ``(N, 13)``."""
    iops = column_iops(bgc, tab, p, sw, lay)
    if sw.RECOM_RADTRANS:
        rt = radtrans_direct(hnode, rmud_of(cos_ai), Edsf, Essf, iops.a_k, iops.bt_k, iops.bb_k)
        light = sms_light(rt.tirrwq, rt.tirrq, tab)
    else:
        parw, parl = beer_lambert(hnode, Edsf, Essf, iops.a_k)
        rt = None
        light = sms_light(parw, parl, tab)
    return SpectralStep(light=light, rt=rt, iops=iops)
