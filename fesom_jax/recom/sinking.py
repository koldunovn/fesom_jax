"""REcoM's per-tracer additions inside FESOM2's tracer loop: the benthic return flux, ballasting, sinking and
the benthos deposit (parent plan Task 11, gate G3).

A literal f64 port of ``recom_sinking.F90`` (REcoM_src 692f804) and of the TEOS-10 routines its ``ballast``
calls (``mocsy/gsw_*.F90``), vectorised over ``(nod2D, nl-1)`` layers / ``(nod2D, nl)`` interfaces: Fortran
loops over the nodes and, per node, over the layers; every column is independent, the vertical couplings
(the TVD stencil of the sinking flux, the column sums of the benthic routines) are explicit shifts, gathers
and sums along the level axis. Pure, ``jax.lax``/``jnp.where`` control flow only, jit- and grad-safe.

The sequence (FESOM2 ``oce_ale_tracer.F90``, the instrumented reference clone ``fesom2_recom`` fc698a30;
the same code at ``fesom2_sp_zstar/src/oce_ale_tracer.F90:725-866``). The tracer loop runs over T, S and
the BGC tracers in ``namelist.tra`` order (:data:`fesom_jax.recom.tracers.BGC_IDS`). For each tracer:

1. ``init_tracers_AB``, the advection (into ``del_ttf``), the horizontal and explicit Redi diffusion
   (``diff_tracers_ale`` up to ``diff_ver_part_redi_expl``): only ``del_ttf`` changes, ``values`` is still
   the tracer at the start of the loop (for the BGC tracers: the output of ``recom()``, stage 1);
2. ``dtr_bf = vert_sink = str_bf = 0`` (``:654-657``); for a remin tracer (1001 1002 1003 1018 1019 1022,
   :data:`~fesom_jax.recom.tracers.REMIN_IDS`): :func:`diff_ver_recom_expl` from ``GlodecayBenthos``
   (stage 1's ``sfc.GlodecayBenthos``, already ``/86400``) and ``values += dtr_bf`` on the wet layers
   (``:753-776``);
3. for a sinking tracer (:data:`~fesom_jax.recom.tracers.SINKING_IDS`, ``:780-866``):
   :func:`get_seawater_viscosity` from ``tracers(1)``, ``tracers(2)`` = T and S, FINAL for this step (their
   own loop iterations, implicit diffusion included, ran first; S before the salinity clip, which runs
   after the whole loop); :func:`get_particle_density` from the CURRENT detritus tracers: those already
   finished in this loop hold their final values, the others (and the tracer itself) still hold the
   stage-1 values; :func:`ballast` (TEOS-10 in-situ density, :func:`seawater_density`) → the scaling
   factors; :func:`ver_sinking_recom` → ``vert_sink`` and :func:`ver_sinking_recom_benthos` → ``str_bf``
   and this tracer's benthos buffer ``Benthos_tr``, both from ``values`` at the start of the loop; then
   ``values = values + vert_sink`` and ``values = values + str_bf`` (two additions, in that order);
4. the T* update with the modified ``values`` (``:868-905``): ``del_ttf += values*(hnode - hnode_new)``,
   ``values += del_ttf/hnode_new``, so every increment enters as ``Δ·hnode/hnode_new``; then the implicit
   vertical diffusion with ``bc_surface`` (``:910-925``), ``relax_to_clim`` (nothing for BGC), the halo
   exchange.

After the loop (``:531-541``): ``Benthos = Benthos + Benthos_tr(:, :, tr)`` for every tracer in loop order
(``__usetp`` is always compiled with REcoM, so ``Benthos`` is constant inside the loop). FESOM's
``SinkingVel1/2`` are zeroed at every tracer start and never written (``ver_sinking_recom`` writes
``Sinkvel1_tr``/``Sinkvel2_tr``, whose merge is commented out, ``:539-540``): the dumped ``SinkingVel1/2``
are zero; the actual velocity is :func:`sinking_velocity` (returned as ``wsink``).

:func:`recom_tracer_loop_terms` runs steps 2-3 for the whole BGC stack in Fortran's order and calls the
caller's ``finish`` (stage 3: step 4) to complete each tracer, so the density read by a later detritus
tracer sees the earlier ones' final values. Its ``batched=True`` schedule (default) is value-identical to
the sequential one: only the detritus tracers read the density (the ballasted speed applies to them
alone), the two detritus classes read disjoint densities, so the 23 other tracers finish in one batch and
the detritus in four rounds of two (DetN+DetZ2N, DetC+DetZ2C, DetSi+DetZ2Si, DetCalc+DetZ2Calc).

Configuration (``bin_2p3z2d``, :data:`fesom_jax.recom.params.SUPPORTED`): 3 zoo + 2 detritus, no
coccolithophores, ballasting with density and viscosity scaling, no MEDUSA, no ciso; ``allow_var_sinking``
is read here (both branches ported). Parameters (``&pavariables``, ``&pasinking``, ``&pairon``,
``&paballasting`` of :class:`~fesom_jax.recom.params.RecomParams`): ``VDet``, ``VDet_zoo2``, ``VPhy``,
``VDia``, ``Vdet_a``, ``Fe2N_benthos``, ``rho_POC``, ``rho_PON``, ``rho_CaCO3``, ``rho_opal``,
``rho_ref_part``, ``rho_ref_water``, ``visc_ref_water``, ``w_ref1``, ``w_ref2``, ``depth_scaling1``,
``depth_scaling2``, ``max_sinking_velocity``; constants ``tiny``, ``SecondsPerDay``, ``redO2C``.

Quirks kept (all verified against the dumps):

* the benthos deposit uses its OWN speed (``Vben``): ``VDet``/``VPhy``/``VDia`` + ``Vdet_a |zbar|`` (every
  sinking tracer, phytoplankton and diatoms included, whose water-column sinking is skipped because
  ``VPhy = VDia = 0`` is not ``> 0.1``), ``VDet_zoo2`` for the second detritus class; never the ballasted
  speed. The chlorophyll tracers (1006, 1015) lose their bottom flux (``str_bf``) but feed no benthos pool;
* the ballasted speed of the first detritus class sits inside the ``allow_var_sinking`` branch; a
  negative density excess gives a negative speed, and ``|w| > max_sinking_velocity`` sets ``w = +max``;
* ``ver_sinking_recom`` has no flux through the surface or the bottom interface (the bottom goes through
  ``str_bf``), and its CFL cap is ``0.9 dz/dt``, with ``dz`` the distance between tracer points (half a
  layer at the surface and the bottom); ``dt`` is an argument (``k·dt`` under the BGC-every-k option);
* ``gsw_saar`` reads ``p_ref(ndepth_max)`` with ``ndepth_max = 999`` (the land flag of ``ndepth_ref``) at
  coastal nodes, past the end of ``p_ref(45)``: in the reference binary that address is zero-filled
  ``.bss`` (``mod_ice ... write_t_ice_thermo`` IOMSG buffer, ``nm libfesom.so``), so the pressure is capped
  at 0 and those columns take the surface Absolute Salinity Anomaly at every depth (72 of the 3140 pi
  nodes; the dumped density scaling of exactly those columns confirms it). Mirrored
  (:data:`SAAR_OOB_P_REF`); another Fortran build may read something else there (undefined behaviour).

The SAAR table (``gsw_mod_saar_data.F90``, GSW 3.05.6, 91 x 45 x 45) is parsed from the REcoM source tree
(:func:`load_saar_table`; ``$FESOM_RECOM_SRC`` or ``/work/ab0995/a270088/REcoM_src``); its horizontal
interpolation depends only on the node position, so it is done once per mesh on the host
(:func:`saar_columns`), the vertical part per step (:func:`gsw_saar`).

Cavities (``ulevels_nod2D > 1``) are out of scope (:func:`fesom_jax.recom.sms.check_no_cavities`): every
column starts at layer 0.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .carbonate import p80
from .params import REDO2C, SECONDS_PER_DAY, TINY, RecomParams, RecomSwitches
from .tracers import BGC_IDS, DET2_IDS, DET_IDS, NTR, REMIN_IDS, SINKING_IDS, SLOT

f64 = jnp.float64

# --------------------------------------------------------------------------------------
# Tracer groups (recom_modules.F90:477-479, recom_init.F90:589-643, recom_sinking.F90:111-137, 647-693)
# --------------------------------------------------------------------------------------
PHY_IDS = (1004, 1005, 1020, 1006)          # recom_phy_tracer_id (ciso entries dropped)
DIA_IDS = (1013, 1014, 1016, 1015)          # recom_dia_tracer_id (ciso entries dropped)
# ver_sinking_recom_benthos's benthos pools (recom_sinking.F90:216-304): 1 N, 2 C, 3 Si, 4 Cal
BENTHOS_POOL: Dict[int, int] = {1004: 0, 1007: 0, 1013: 0, 1025: 0,
                                1005: 1, 1008: 1, 1014: 1, 1026: 1,
                                1016: 2, 1017: 2, 1027: 2,
                                1020: 3, 1021: 3, 1028: 3}
# get_particle_density's components, in its order b1..b4 = C, N, Si, Cal (recom_sinking.F90:1098-1108)
DET1_CNSICAL = (1008, 1007, 1017, 1021)
DET2_CNSICAL = (1026, 1025, 1027, 1028)

CFL_SAFETY = 0.9                            # recom_sinking.F90:636
ONESIXTH = 1.0 / 6.0                        # recom_sinking.F90:634

# --------------------------------------------------------------------------------------
# TEOS-10 (mocsy/gsw_mod_teos10_constants.F90, gsw_mod_specvol_coefficients.F90)
# --------------------------------------------------------------------------------------
GSW_SSO = 35.16504
GSW_UPS = GSW_SSO / 35.0
GSW_SFAC = 0.0248826675584615
GSW_OFFSET = 5.971840214030754e-1
GSW_CP0 = 3991.86795711963
GSW_ERROR_LIMIT = 1e10
# p_ref(999) in the reference binary (see the module docstring)
SAAR_OOB_P_REF = 0.0

_V = dict(
    v000=1.0769995862e-3, v001=-6.0799143809e-5, v002=9.9856169219e-6, v003=-1.1309361437e-6,
    v004=1.0531153080e-7, v005=-1.2647261286e-8, v006=1.9613503930e-9, v010=-3.1038981976e-4,
    v011=2.4262468747e-5, v012=-5.8484432984e-7, v013=3.6310188515e-7, v014=-1.1147125423e-7,
    v020=6.6928067038e-4, v021=-3.4792460974e-5, v022=-4.8122251597e-6, v023=1.6746303780e-8,
    v030=-8.5047933937e-4, v031=3.7470777305e-5, v032=4.9263106998e-6, v040=5.8086069943e-4,
    v041=-1.7322218612e-5, v042=-1.7811974727e-6, v050=-2.1092370507e-4, v051=3.0927427253e-6,
    v060=3.1932457305e-5, v100=-1.5649734675e-5, v101=1.8505765429e-5, v102=-1.1736386731e-6,
    v103=-3.6527006553e-7, v104=3.1454099902e-7, v110=3.5009599764e-5, v111=-9.5677088156e-6,
    v112=-5.5699154557e-6, v113=-2.7295696237e-7, v120=-4.3592678561e-5, v121=1.1100834765e-5,
    v122=5.4620748834e-6, v130=3.4532461828e-5, v131=-9.8447117844e-6, v132=-1.3544185627e-6,
    v140=-1.1959409788e-5, v141=2.5909225260e-6, v150=1.3864594581e-6, v200=2.7762106484e-5,
    v201=-1.1716606853e-5, v202=2.1305028740e-6, v203=2.8695905159e-7, v210=-3.7435842344e-5,
    v211=-2.3678308361e-7, v212=3.9137387080e-7, v220=3.5907822760e-5, v221=2.9283346295e-6,
    v222=-6.5731104067e-7, v230=-1.8698584187e-5, v231=-4.8826139200e-7, v240=3.8595339244e-6,
    v300=-1.6521159259e-5, v301=7.9279656173e-6, v302=-4.6132540037e-7, v310=2.4141479483e-5,
    v311=-3.4558773655e-6, v312=7.7618888092e-9, v320=-1.4353633048e-5, v321=3.1655306078e-7,
    v330=2.2863324556e-6, v400=6.9111322702e-6, v401=-3.4102187482e-6, v402=-6.3352916514e-8,
    v410=-8.7595873154e-6, v411=1.2956717783e-6, v420=4.3703680598e-6, v500=-8.0539615540e-7,
    v501=5.0736766814e-7, v510=-3.3052758900e-7, v600=2.0543094268e-7)


def gsw_specvol(sa, ct, p):
    """Specific volume [m3 kg-1], the 75-term polynomial, ``mocsy/gsw_specvol.F90`` (Horner order kept)."""
    v = _V
    xs = jnp.sqrt(GSW_SFAC * sa + GSW_OFFSET)
    ys = ct * 0.025
    z = p * 1e-4
    return (v["v000"] + xs * (v["v010"] + xs * (v["v020"] + xs * (v["v030"] + xs * (v["v040"] + xs * (v["v050"]
            + v["v060"] * xs))))) + ys * (v["v100"] + xs * (v["v110"] + xs * (v["v120"] + xs * (v["v130"]
            + xs * (v["v140"] + v["v150"] * xs)))) + ys * (v["v200"] + xs * (v["v210"] + xs * (v["v220"]
            + xs * (v["v230"] + v["v240"] * xs))) + ys * (v["v300"] + xs * (v["v310"] + xs * (v["v320"]
            + v["v330"] * xs)) + ys * (v["v400"] + xs * (v["v410"] + v["v420"] * xs) + ys * (v["v500"]
            + v["v510"] * xs + v["v600"] * ys))))) + z * (v["v001"] + xs * (v["v011"] + xs * (v["v021"]
            + xs * (v["v031"] + xs * (v["v041"] + v["v051"] * xs)))) + ys * (v["v101"] + xs * (v["v111"]
            + xs * (v["v121"] + xs * (v["v131"] + v["v141"] * xs))) + ys * (v["v201"] + xs * (v["v211"]
            + xs * (v["v221"] + v["v231"] * xs)) + ys * (v["v301"] + xs * (v["v311"] + v["v321"] * xs)
            + ys * (v["v401"] + v["v411"] * xs + v["v501"] * ys)))) + z * (v["v002"] + xs * (v["v012"]
            + xs * (v["v022"] + xs * (v["v032"] + v["v042"] * xs))) + ys * (v["v102"] + xs * (v["v112"]
            + xs * (v["v122"] + v["v132"] * xs)) + ys * (v["v202"] + xs * (v["v212"] + v["v222"] * xs)
            + ys * (v["v302"] + v["v312"] * xs + v["v402"] * ys))) + z * (v["v003"] + xs * (v["v013"]
            + v["v023"] * xs) + ys * (v["v103"] + v["v113"] * xs + v["v203"] * ys) + z * (v["v004"]
            + v["v014"] * xs + v["v104"] * ys + z * (v["v005"] + v["v006"] * z))))))


def gsw_rho(sa, ct, p):
    """In-situ density [kg m-3], ``mocsy/gsw_rho.F90``: ``1/gsw_specvol``."""
    return 1.0 / gsw_specvol(sa, ct, p)


def gsw_ct_from_pt(sa, pt):
    """Conservative from potential temperature, ``mocsy/gsw_ct_from_pt.F90``."""
    x2 = GSW_SFAC * sa
    x = jnp.sqrt(x2)
    y = pt * 0.025
    pot_enthalpy = (61.01362420681071 + y * (168776.46138048015
                    + y * (-2735.2785605119625 + y * (2574.2164453821433
                    + y * (-1536.6644434977543 + y * (545.7340497931629
                    + (-50.91091728474331 - 18.30489878927802 * y) * y)))))
                    + x2 * (268.5520265845071 + y * (-12019.028203559312
                    + y * (3734.858026725145 + y * (-2046.7671145057618
                    + y * (465.28655623826234 + (-0.6370820302376359
                    - 10.650848542359153 * y) * y))))
                    + x * (937.2099110620707 + y * (588.1802812170108
                    + y * (248.39476522971285 + (-3.871557904936333
                    - 2.6268019854268356 * y) * y))
                    + x * (-1687.914374187449 + x * (246.9598888781377
                    + x * (123.59576582457964 - 48.5891069025409 * x))
                    + y * (936.3206544460336
                    + y * (-942.7827304544439 + y * (369.4389437509002
                    + (-33.83664947895248 - 9.987880382780322 * y) * y)))))))
    return pot_enthalpy / GSW_CP0


# --------------------------------------------------------------------------------------
# The Absolute Salinity Anomaly Ratio (mocsy/gsw_saar.F90 and helpers)
# --------------------------------------------------------------------------------------
class SaarTable(NamedTuple):
    """``gsw_mod_saar_data`` (numpy, 0-based; ``saar_ref`` is ``(nz, ny, nx)`` as Fortran's ``(nz,ny,nx)``)."""
    p_ref: np.ndarray        # (45,)
    lats_ref: np.ndarray     # (45,)
    longs_ref: np.ndarray    # (91,)
    saar_ref: np.ndarray     # (45, 45, 91)
    ndepth_ref: np.ndarray   # (45, 91) int (999 = land)
    longs_pan: np.ndarray    # (6,)
    lats_pan: np.ndarray     # (6,)


class SaarColumns(NamedTuple):
    """The per-node, step-independent part of ``gsw_saar`` / ``gsw_sa_from_sp`` (:func:`saar_columns`)."""
    h: jax.Array             # (nod2D, 45) the horizontally interpolated SAAR at every p_ref level
    p_cap: jax.Array         # (nod2D,) p_ref(ndepth_max) [dbar]
    has_data: jax.Array      # (nod2D,) bool: ndepth_max /= -1 (else saar = 0)
    baltic: jax.Array        # (nod2D,) bool: gsw_sa_from_sp_baltic applies
    p_ref: jax.Array         # (45,)


def _default_recom_src() -> Path:
    return Path(os.environ.get("FESOM_RECOM_SRC", "/work/ab0995/a270088/REcoM_src"))


def _fortran_reals(body: str) -> np.ndarray:
    toks = [t for t in re.split(r"[\s,&]+", body) if t]
    return np.array([float(t.lower().replace("_r8", "").replace("d", "e")) for t in toks], np.float64)


@lru_cache(maxsize=2)
def load_saar_table(path: Optional[str] = None) -> SaarTable:
    """Parse ``mocsy/gsw_mod_saar_data.F90`` (``data`` statements; the decimal literals are read with
    Python's correctly rounded ``float``, as the compiler does). ``path`` defaults to
    ``$FESOM_RECOM_SRC/mocsy/gsw_mod_saar_data.F90``."""
    p = Path(path) if path is not None else _default_recom_src() / "mocsy" / "gsw_mod_saar_data.F90"
    text = p.read_text()
    nx, ny, nz = (int(re.search(rf"integer, parameter :: {n} = (\d+)", text).group(1)) for n in ("nx", "ny", "nz"))
    saar = np.full((nz, ny, nx), np.nan)
    got = np.zeros((ny, nx), bool)
    simple = {}
    for m in re.finditer(r"^\s*data\s+(\w+)(\(([^)]*)\))?\s*/(.*?)/", text, re.M | re.S):
        name, idx, body = m.group(1), m.group(3), m.group(4)
        if name == "saar_ref":
            _, j, i = idx.split(",")
            saar[:, int(j) - 1, int(i) - 1] = _fortran_reals(body)
            got[int(j) - 1, int(i) - 1] = True
        elif name in ("p_ref", "lats_ref", "longs_ref", "longs_pan", "lats_pan", "ndepth_ref"):
            simple[name] = _fortran_reals(body)
    if not got.all():
        raise ValueError(f"{p}: saar_ref incomplete")
    nd = simple["ndepth_ref"].astype(np.int64).reshape(nx, ny).T       # Fortran (ny, nx), column-major
    return SaarTable(p_ref=simple["p_ref"], lats_ref=simple["lats_ref"], longs_ref=simple["longs_ref"],
                     saar_ref=saar, ndepth_ref=nd, longs_pan=simple["longs_pan"], lats_pan=simple["lats_pan"])


def _util_indx_np(x, z):
    """``gsw_util_indx`` (1-based k with ``x(k) <= z < x(k+1)``, clamped to ``[1, n-1]``), numpy."""
    return np.clip(np.searchsorted(x, z, side="right"), 1, len(x) - 1)


def _add_mean_np(d):
    """``gsw_add_mean`` along the last axis (4 corners), the sequential sum of Fortran."""
    nmean = np.zeros(d.shape[:-1])
    s = np.zeros(d.shape[:-1])
    for k in range(4):
        ok = np.abs(d[..., k]) <= 100.0
        nmean = nmean + ok
        s = np.where(ok, s + d[..., k], s)
    mean = np.where(nmean == 0, 0.0, s / np.where(nmean == 0, 1.0, nmean))
    return np.where(np.abs(d) >= 100.0, mean[..., None], d)


def _above_line_np(t, x, lat_test):
    k = _util_indx_np(t.longs_pan, x) - 1
    r = (x - t.longs_pan[k]) / (t.longs_pan[k + 1] - t.longs_pan[k])
    lats_line = t.lats_pan[k] + r * (t.lats_pan[k + 1] - t.lats_pan[k])
    return lats_line <= lat_test, lats_line


def _add_barrier_np(t, d, long, lat, long_grid, lat_grid, dlong, dlat):
    """``gsw_add_barrier`` (the Panama barrier) along the last axis; per-node scalars broadcast."""
    a0, _ = _above_line_np(t, long, lat)
    k = _util_indx_np(t.longs_pan, long_grid) - 1
    r = (long_grid - t.longs_pan[k]) / (t.longs_pan[k + 1] - t.longs_pan[k])
    line14 = t.lats_pan[k] + r * (t.lats_pan[k + 1] - t.lats_pan[k])
    k = _util_indx_np(t.longs_pan, long_grid + dlong) - 1
    r = (long_grid + dlong - t.longs_pan[k]) / (t.longs_pan[k + 1] - t.longs_pan[k])
    line23 = t.lats_pan[k] + r * (t.lats_pan[k + 1] - t.lats_pan[k])
    above = np.stack([line14 <= lat_grid, line23 <= lat_grid, line23 <= lat_grid + dlat,
                      line14 <= lat_grid + dlat], axis=-1)              # corners 1, 2, 3, 4
    same = above == a0[..., None]
    same = same[:, None, :]                                              # broadcast over the p levels
    nmean = np.zeros(d.shape[:-1])
    s = np.zeros(d.shape[:-1])
    for kk in range(4):
        ok = (np.abs(d[..., kk]) <= 100.0) & same[..., kk]
        nmean = nmean + ok
        s = np.where(ok, s + d[..., kk], s)
    mean = np.where(nmean == 0, 0.0, s / np.where(nmean == 0, 1.0, nmean))
    return np.where((np.abs(d) >= 1e10) | ~same, mean[..., None], d)


def _xinterp1_np(x, y, x0):
    k = _util_indx_np(x, x0) - 1
    r = (x0 - x[k]) / (x[k + 1] - x[k])
    return y[k] + r * (y[k + 1] - y[k])


def saar_columns(lon_deg, lat_deg, table: Optional[SaarTable] = None,
                 oob_p_ref: float = SAAR_OOB_P_REF) -> SaarColumns:
    """The node-position part of ``gsw_saar`` (``mocsy/gsw_saar.F90``) and the Baltic test of
    ``gsw_sa_from_sp_baltic``, on the host (numpy), once per mesh: per node the corner indices, ``r1``/``s1``,
    the barrier / mean fill of the four corners and the bilinear value at every ``p_ref`` level (the
    expression of ``sa_upper``/``sa_lower``), and ``p_ref(ndepth_max)``; ``ndepth_max = 999`` (a land
    corner) reads ``oob_p_ref`` (Fortran's out-of-bounds ``p_ref(999)``, module docstring).

    ``lon_deg``, ``lat_deg`` (nod2D,): ``geo_coord_nod2D / rad`` as ``ballast`` computes them."""
    t = load_saar_table() if table is None else table
    lon = np.asarray(lon_deg, np.float64)
    lat = np.asarray(lat_deg, np.float64)
    if (lat < -86.0).any() or (lat > 90.0).any():
        raise NotImplementedError("gsw_saar error return (latitude outside [-86, 90]) is not ported")
    nx, ny = len(t.longs_ref), len(t.lats_ref)
    long_tmp = np.where(lon < 0.0, lon + 360.0, lon)
    dlong = t.longs_ref[1] - t.longs_ref[0]
    dlat = t.lats_ref[1] - t.lats_ref[0]
    indx0 = np.floor(1.0 + (nx - 1.0) * (long_tmp - t.longs_ref[0]) / (t.longs_ref[nx - 1] - t.longs_ref[0]))
    indx0 = indx0.astype(np.int64)
    indx0 = np.where(indx0 == nx, nx - 1, indx0)
    indy0 = np.floor(1.0 + (ny - 1.0) * (lat - t.lats_ref[0]) / (t.lats_ref[ny - 1] - t.lats_ref[0]))
    indy0 = indy0.astype(np.int64)
    indy0 = np.where(indy0 == ny, ny - 1, indy0)
    ix, iy = indx0 - 1, indy0 - 1                                         # 0-based
    deli, delj = (0, 1, 1, 0), (0, 0, 1, 1)
    nd = np.stack([t.ndepth_ref[iy + delj[k], ix + deli[k]] for k in range(4)], axis=-1)
    ndepth_max = np.full(lon.shape, -1, np.int64)
    for k in range(4):
        ndepth_max = np.where(nd[:, k] > 0, np.maximum(ndepth_max, nd[:, k]), ndepth_max)
    has_data = ndepth_max != -1
    nz = len(t.p_ref)
    inb = (ndepth_max >= 1) & (ndepth_max <= nz)
    p_cap = np.where(inb, t.p_ref[np.clip(ndepth_max, 1, nz) - 1], oob_p_ref)
    r1 = (long_tmp - t.longs_ref[ix]) / (t.longs_ref[ix + 1] - t.longs_ref[ix])
    s1 = (lat - t.lats_ref[iy]) / (t.lats_ref[iy + 1] - t.lats_ref[iy])
    corners = np.stack([t.saar_ref[:, iy + delj[k], ix + deli[k]].T for k in range(4)], axis=-1)  # (nod,nz,4)
    barrier = ((t.longs_pan[0] <= long_tmp) & (long_tmp <= t.longs_pan[-1] - 0.001)
               & (t.lats_pan[-1] <= lat) & (lat <= t.lats_pan[0]))
    fill_mean = np.abs(corners.sum(axis=-1)) >= 1e10
    c_bar = _add_barrier_np(t, corners, long_tmp, lat, t.longs_ref[ix], t.lats_ref[iy], dlong, dlat)
    c_mean = _add_mean_np(corners)
    c = np.where(barrier[:, None, None], c_bar, np.where(fill_mean[..., None], c_mean, corners))
    r1c, s1c = r1[:, None], s1[:, None]
    h = ((1.0 - s1c) * (c[..., 0] + r1c * (c[..., 1] - c[..., 0]))
         + s1c * (c[..., 3] + r1c * (c[..., 2] - c[..., 3])))
    # gsw_sa_from_sp_baltic's region test (the raw longitude, mocsy/gsw_sa_from_sp_baltic.F90)
    xb_left, yb_left = np.array([12.6, 7.0, 26.0]), np.array([50.0, 59.0, 69.0])
    xb_right, yb_right = np.array([45.0, 26.0]), np.array([50.0, 69.0])
    box = (xb_left[1] < lon) & (lon < xb_right[0]) & (yb_left[0] < lat) & (lat < yb_left[2])
    xx_left = _xinterp1_np(yb_left, xb_left, lat)
    xx_right = _xinterp1_np(yb_right, xb_right, lat)
    baltic = box & (xx_left <= lon) & (lon <= xx_right)
    h = np.where(has_data[:, None], h, 0.0)
    return SaarColumns(h=jnp.asarray(h), p_cap=jnp.asarray(p_cap), has_data=jnp.asarray(has_data),
                       baltic=jnp.asarray(baltic), p_ref=jnp.asarray(t.p_ref))


def gsw_saar(p, cols: SaarColumns):
    """The pressure part of ``gsw_saar`` (``mocsy/gsw_saar.F90``) for ``p`` (nod2D, ...) [dbar]."""
    extra = p.ndim - 1
    ex = lambda a: a.reshape(a.shape + (1,) * extra)
    p_cap = ex(cols.p_cap)
    p_tmp = jnp.where(p > p_cap, p_cap, p)
    nz = cols.p_ref.shape[0]
    k0 = jnp.clip(jnp.searchsorted(cols.p_ref, p_tmp, side="right"), 1, nz - 1) - 1    # gsw_util_indx - 1
    p0, p1 = cols.p_ref[k0], cols.p_ref[k0 + 1]
    t1 = (p_tmp - p0) / (p1 - p0)
    kf = k0.reshape(k0.shape[0], -1)
    sa_upper = jnp.take_along_axis(cols.h, kf, axis=1).reshape(p.shape)
    sa_lower = jnp.take_along_axis(cols.h, kf + 1, axis=1).reshape(p.shape)
    sa_lower = jnp.where(jnp.abs(sa_lower) >= 1e10, sa_upper, sa_lower)
    saar = sa_upper + t1 * (sa_lower - sa_upper)
    return jnp.where(ex(cols.has_data), saar, 0.0)


def gsw_sa_from_sp(sp, p, cols: SaarColumns):
    """Absolute from practical salinity, ``mocsy/gsw_sa_from_sp.F90`` (the Baltic branch included; the
    ``saar > gsw_error_limit`` error return cannot occur after :func:`saar_columns`)."""
    extra = sp.ndim - 1
    baltic = cols.baltic.reshape(cols.baltic.shape + (1,) * extra)
    sa = GSW_UPS * sp * (1.0 + gsw_saar(p, cols))
    sa_baltic = ((GSW_SSO - 0.087) / 35.0) * sp + 0.087
    return jnp.where(baltic, sa_baltic, sa)


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
class SinkingGeometry(NamedTuple):
    """What the tracer-loop additions read, at the tracer step (the ``transport`` dump's arrays)."""
    hnode: jax.Array          # (nod2D, nl-1) layer thickness at the start of the step [m]
    hnode_new: jax.Array      # (nod2D, nl-1) layer thickness at the end of the step [m]
    zbar: jax.Array           # (nod2D, nl) interface depths zbar_3d_n [m, negative down]
    z: jax.Array              # (nod2D, nl-1) mid-layer depths Z_3d_n [m, negative down]
    area: jax.Array           # (nod2D, nl) mesh area (upper-interface control-volume area) [m2]
    areasvol: jax.Array       # (nod2D, nl) mesh areasvol [m2]
    seafloor: jax.Array       # (nod2D, nl-1) seafloor area under each layer (:func:`seafloor_area`) [m2]
    nlev: jax.Array           # (nod2D,) int: nlevels_nod2D - 1, the wet layers are k < nlev
    lat_deg: jax.Array        # (nod2D,) geo latitude geo_coord_nod2D(2)/rad [deg]
    saar: SaarColumns         # :func:`saar_columns` of the nodes


def seafloor_area(elem_nodes, elem_nlevels, elem_area, nod2D: int, nl: int):
    """``seafloor(nz)`` of ``ver_sinking_recom_benthos`` / ``diff_ver_recom_expl`` (``recom_sinking.F90:147-151,
    563-567``) for every node: 1/3 of the area of each element around the node, in the element's deepest
    wet layer ``nlevels(e) - 1``. ``elem_nodes`` (elem2D, 3) 0-based, ``elem_nlevels`` FESOM's ``nlevels``."""
    k = jnp.asarray(elem_nlevels) - 2                                   # 0-based layer nlevels(e) - 1
    w = jnp.asarray(elem_area, f64) / 3.0
    out = jnp.zeros((nod2D, nl - 1), f64)
    en = jnp.asarray(elem_nodes)
    for j in range(3):
        out = out.at[en[:, j], k].add(w)
    return out


def _wet_areasvol(geom: SinkingGeometry, wet):
    """``areasvol`` on the wet layers, 1 below the bottom (where it is 0). The callers mask those layers out of the
    result anyway; the placeholder keeps the masked division finite, so its derivative is not ``0 * inf = nan``
    (the forward values are unchanged)."""
    nl1 = geom.hnode.shape[1]
    return jnp.where(wet, geom.areasvol[:, :nl1], 1.0)


def _levels(geom: SinkingGeometry, n: int):
    return jnp.arange(n)[None, :], geom.nlev[:, None]


# --------------------------------------------------------------------------------------
# diff_ver_recom_expl (recom_sinking.F90:443-582)
# --------------------------------------------------------------------------------------
def benthic_bottom_flux(tracer_id: int, decay_benthos, params: RecomParams):
    """``bottom_flux`` of a remin tracer from ``GlodecayBenthos`` (nod2D, 4) [mmol m-2 s-1]
    (``recom_sinking.F90:513-546``, the non-MEDUSA case)."""
    g = decay_benthos
    tid = int(tracer_id)
    if tid == 1001:
        return g[:, 0]
    if tid == 1002:
        return g[:, 1] + g[:, 3]
    if tid == 1003:
        return g[:, 3] * 2.0 - 1.0625 * g[:, 0]
    if tid == 1018:
        return g[:, 2]
    if tid == 1019:
        return g[:, 0] * params.Fe2N_benthos
    if tid == 1022:
        return -g[:, 1] * REDO2C
    raise ValueError(f"tracer {tid} is not a remin tracer {REMIN_IDS} (Fortran stops here)")


def diff_ver_recom_expl(tracer_id: int, decay_benthos, geom: SinkingGeometry, dt, params: RecomParams):
    """The benthic return flux ``dtr_bf`` (nod2D, nl-1) of a remin tracer [tracer units per step]:
    the areal flux times ``area(1)``, spread over the layers in proportion to the seafloor area under each,
    entering each layer through its bottom face: ``seafloor(k) tot/sum(seafloor) dt/areasvol(k)/hnode_new(k)``."""
    k, nlev = _levels(geom, geom.hnode.shape[1])
    wet = k < nlev
    tot = benthic_bottom_flux(tracer_id, decay_benthos, params) * geom.area[:, 0]
    sf = jnp.where(wet, geom.seafloor, 0.0)
    ssum = jnp.sum(sf, axis=1, keepdims=True)
    vd = sf * tot[:, None] / ssum
    nl1 = geom.hnode.shape[1]
    out = vd * dt / _wet_areasvol(geom, wet) / jnp.where(wet, geom.hnode_new, 1.0)
    return jnp.where(wet, out, 0.0)


# --------------------------------------------------------------------------------------
# Ballasting (recom_sinking.F90:928-1233)
# --------------------------------------------------------------------------------------
def get_seawater_viscosity(temp, salt, nlev):
    """Dynamic seawater viscosity [kg m-1 s-1], Sharqawy et al. (2010), ``recom_sinking.F90:1186-1233``;
    0 below the bottom. ``temp``, ``salt`` (nod2D, nl-1): FESOM's T and S."""
    t, s = temp, salt
    a = 1.541 + 1.998 * 0.01 * t - 9.52 * 1e-5 * t * t
    b = 7.974 - 7.561 * 0.01 * t + 4.724 * 1e-4 * t * t
    mu_w = 4.2844 * 1.0e-5 + (1.0 / (0.157 * (t + 64.993) * (t + 64.993) - 91.296))
    visc = mu_w * (1.0 + a * s * 0.001 + b * s * 0.001 * s * 0.001)
    wet = jnp.arange(t.shape[1])[None, :] < jnp.asarray(nlev)[:, None]
    return jnp.where(wet, visc, 0.0)


def get_particle_density(det_c, det_n, det_si, det_cal, nlev, params: RecomParams):
    """Bulk density of a detritus class [kg m-3] from its C, N, Si and CaCO3 concentrations (Cram et al.
    2018), ``recom_sinking.F90:1060-1168``: the components floored at ``tiny``, their fractions of the sum
    weight ``rho_POC``, ``rho_PON``, ``rho_opal``, ``rho_CaCO3``; 0 below the bottom."""
    p = params
    b1, b2, b3, b4 = (jnp.maximum(TINY, x) for x in (det_c, det_n, det_si, det_cal))
    aux = b1 + b2 + b3 + b4
    a1, a2, a3, a4 = b1 / aux, b2 / aux, b3 / aux, b4 / aux
    rho = p.rho_CaCO3 * a4 + p.rho_opal * a3 + p.rho_POC * a1 + p.rho_PON * a2
    wet = jnp.arange(det_c.shape[1])[None, :] < jnp.asarray(nlev)[:, None]
    return jnp.where(wet, rho, 0.0)


def particle_densities(stack, nlev, params: RecomParams):
    """``(rho_particle1, rho_particle2)`` from a BGC stack (nod2D, nl-1, ntr) in State.bgc slot order."""
    pick = lambda ids: [stack[..., SLOT[i]] for i in ids]
    return (get_particle_density(*pick(DET1_CNSICAL), nlev, params),
            get_particle_density(*pick(DET2_CNSICAL), nlev, params))


def seawater_density(temp, salt, geom: SinkingGeometry):
    """``ballast``'s in-situ seawater density [kg m-3] (``recom_sinking.F90:984-994``): pressure from
    ``depth2press`` (``p80``) at ``|Z_3d_n|``, ``gsw_sa_from_sp``, ``gsw_ct_from_pt`` (T is potential
    temperature), ``gsw_rho``. Dry cells take safe placeholder values (and are masked by the callers)."""
    k, nlev = _levels(geom, temp.shape[1])
    wet = k < nlev
    t = jnp.where(wet, temp, 10.0)
    s = jnp.where(wet, salt, 35.0)
    depth = jnp.where(wet, jnp.abs(geom.z), 10.0)
    pres = p80(depth, geom.lat_deg[:, None])
    sa = gsw_sa_from_sp(s, pres, geom.saar)
    ct = gsw_ct_from_pt(sa, t)
    return gsw_rho(sa, ct, pres)


def _extend_bottom(x, nlev, fill):
    """(nod2D, nl-1) layer field → (nod2D, nl) with the bottom value copied to interface ``nlev``
    (``recom_sinking.F90:1036-1040``) and ``fill`` below it (Fortran leaves those entries stale)."""
    nod, nl1 = x.shape
    k = jnp.arange(nl1 + 1)[None, :]
    nlev = jnp.asarray(nlev)[:, None]
    xe = jnp.concatenate([x, jnp.zeros((nod, 1), x.dtype)], axis=1)
    bottom = jnp.take_along_axis(x, jnp.maximum(nlev - 1, 0), axis=1)
    return jnp.where(k < nlev, xe, jnp.where(k == nlev, bottom, fill))


class Ballast(NamedTuple):
    scaling_visc: jax.Array       # (nod2D, nl)
    scaling_density1: jax.Array   # (nod2D, nl)
    scaling_density2: jax.Array   # (nod2D, nl)


def ballast(rho_particle1, rho_particle2, rho_seawater, visc, nlev, params: RecomParams) -> Ballast:
    """The scaling factors of the ballasted sinking speed, ``recom_sinking.F90:928-1043``
    (``use_density_scaling``, ``use_viscosity_scaling``, ``enable_3zoo2det`` all on): density
    ``(rho_particle - rho_seawater)/(rho_ref_part - rho_ref_water)`` (no positivity guard), viscosity
    ``visc_ref_water/visc`` (1 where ``visc = 0``); on the wet layers and copied to the bottom interface."""
    p = params
    k = jnp.arange(visc.shape[1])[None, :]
    wet = k < jnp.asarray(nlev)[:, None]
    sd1 = (rho_particle1 - rho_seawater) / (p.rho_ref_part - p.rho_ref_water)
    sd2 = (rho_particle2 - rho_seawater) / (p.rho_ref_part - p.rho_ref_water)
    sv = jnp.where(visc != 0.0, p.visc_ref_water / jnp.where(visc != 0.0, visc, 1.0), 1.0)
    sd1, sd2, sv = (jnp.where(wet, x, 1.0) for x in (sd1, sd2, sv))
    return Ballast(scaling_visc=_extend_bottom(sv, nlev, 1.0),
                   scaling_density1=_extend_bottom(sd1, nlev, 1.0),
                   scaling_density2=_extend_bottom(sd2, nlev, 1.0))


# --------------------------------------------------------------------------------------
# ver_sinking_recom (recom_sinking.F90:601-908)
# --------------------------------------------------------------------------------------
def base_sinking_speed(tracer_id: int, params: RecomParams):
    """``Vsink`` [m d-1] of ``ver_sinking_recom`` (``recom_sinking.F90:645-693``, no coccolithophores)."""
    tid = int(tracer_id)
    p = params
    if tid in DET_IDS:
        return p.VDet
    if tid in (1004, 1005, 1006, 1020):
        return p.VPhy
    if tid in DIA_IDS:
        return p.VDia
    if tid in DET2_IDS:
        return p.VDet_zoo2
    return 0.0


def _tracer_point_distance(geom: SinkingGeometry):
    """``dz_trr`` (nod2D, nl): half a layer at the surface and the bottom, the mid-layer distance between."""
    nod, nl1 = geom.hnode.shape
    k = jnp.arange(nl1 + 1)[None, :]
    nlev = geom.nlev[:, None]
    zz = geom.z
    inner = jnp.abs(zz[:, :-1] - zz[:, 1:])                               # k = 1 .. nl-2
    inner = jnp.concatenate([jnp.zeros((nod, 1), inner.dtype), inner, jnp.zeros((nod, 1), inner.dtype)], axis=1)
    top = geom.hnode[:, :1] / 2.0
    bot = jnp.take_along_axis(geom.hnode, jnp.maximum(nlev - 1, 0), axis=1) / 2.0
    dz = jnp.where((k >= 1) & (k <= nlev - 1), inner, 0.0)
    dz = jnp.where(k == 0, top, dz)
    return jnp.where(k == nlev, bot, dz)


def _static_bool(x):
    """A Python bool when ``x`` is concrete, else None (a traced parameter)."""
    try:
        return bool(x)
    except (jax.errors.ConcretizationTypeError, TypeError):
        return None


def sinking_velocity(tracer_id: int, bal: Optional[Ballast], geom: SinkingGeometry, dt, params: RecomParams,
                     switches: RecomSwitches = RecomSwitches(enable_3zoo2det=True)):
    """``Wvel_flux`` (nod2D, nl) [m s-1, negative down] at the interfaces ``0..nlev`` of a sinking tracer,
    CFL-capped at ``0.9 dz/dt`` (``recom_sinking.F90:711-806``); 0 below. ``bal`` is needed for the
    detritus tracers (the ballasted speed)."""
    p = params
    tid = int(tracer_id)
    vs = base_sinking_speed(tid, p)
    absz = jnp.abs(geom.zbar)
    spd = SECONDS_PER_DAY
    w = jnp.broadcast_to(-vs / spd, absz.shape).astype(f64)
    if switches.allow_var_sinking:
        w = -((p.Vdet_a * absz / spd) + vs / spd)
        if switches.use_ballasting and tid in DET_IDS:
            w = p.w_ref1 * bal.scaling_density1 * bal.scaling_visc
            w = jnp.where(p.depth_scaling1 > 0.0, w + (p.depth_scaling1 * absz), w)
            w = jnp.where(jnp.abs(w) > p.max_sinking_velocity, p.max_sinking_velocity, w)
            w = -1.0 * w / spd
    if switches.enable_3zoo2det and tid in DET2_IDS:
        w = jnp.broadcast_to(-p.VDet_zoo2 / spd, absz.shape).astype(f64)
        if switches.use_ballasting:
            w = p.w_ref2 * bal.scaling_density2 * bal.scaling_visc
            w = jnp.where(p.depth_scaling2 > 0.0, w + (p.depth_scaling2 * absz), w)
            w = jnp.where(jnp.abs(w) > p.max_sinking_velocity, p.max_sinking_velocity, w)
            w = -1.0 * w / spd
    dz = _tracer_point_distance(geom)
    cap = CFL_SAFETY * dz / dt
    w = jnp.where(dz > 0.0, jnp.copysign(jnp.minimum(jnp.abs(w), cap), w), w)
    k = jnp.arange(dz.shape[1])[None, :]
    return jnp.where(k <= geom.nlev[:, None], w, 0.0)


def ver_sinking_recom(tracer_id: int, values, wvel, geom: SinkingGeometry, dt, params: RecomParams):
    """``vert_sink`` (nod2D, nl-1): the DST3 flux-limited sinking flux divergence of ``ver_sinking_recom``
    (``recom_sinking.F90:808-905``) for ``values`` (nod2D, nl-1) with ``wvel`` from :func:`sinking_velocity`;
    the interfaces ``1..nlev-1`` carry flux, the surface and the bottom none. Zero unless ``Vsink > 0.1``."""
    vs = base_sinking_speed(tracer_id, params)
    on = _static_bool(vs > 0.1)
    nod, nl1 = values.shape
    if on is False:
        return jnp.zeros_like(values)
    kk = jnp.arange(nl1 + 1)[None, :]
    nlev = geom.nlev[:, None]
    c = lambda i: jnp.clip(i, 0, nl1 - 1)
    take = lambda i: jnp.take_along_axis(values, c(jnp.broadcast_to(i, (nod, nl1 + 1))), axis=1)
    t_k = take(kk)
    t_p1 = take(jnp.minimum(kk + 1, nlev - 1))
    t_m1 = take(jnp.maximum(kk - 1, 0))
    t_m2 = take(jnp.maximum(kk - 2, 0))
    rjp = t_k - t_p1
    rj = t_m1 - t_k
    rjm = t_m2 - t_m1
    flux_if = (kk >= 1) & (kk <= nlev - 1)
    dz = jnp.where(flux_if, _tracer_point_distance(geom), 1.0)
    w = jnp.where(flux_if, wvel, 0.0)
    cfl = jnp.abs(w * dt / dz)
    wps = w + jnp.abs(w)
    wm = w - jnp.abs(w)
    d0 = (2.0 - cfl) * (1.0 - cfl) * ONESIXTH
    d1 = (1.0 - cfl * cfl) * ONESIXTH
    theta_p = rjm / (1.0e-20 + rj)
    psi_p = d0 + d1 * theta_p
    psi_p = jnp.maximum(0.0, jnp.minimum(jnp.minimum(1.0, psi_p), (1.0 - cfl) / (1.0e-20 + cfl) * theta_p))
    theta_m = rjp / (1.0e-20 + rj)
    psi_m = d0 + d1 * theta_m
    psi_m = jnp.maximum(0.0, jnp.minimum(jnp.minimum(1.0, psi_m), (1.0 - cfl) / (1.0e-20 + cfl) * theta_m))
    tv = (0.5 * wps * (t_k + psi_m * rj) + 0.5 * wm * (t_m1 + psi_p * rj))
    vd = jnp.where(flux_if, -tv * geom.area, 0.0)
    wet = kk[:, :nl1] < nlev
    out = (vd[:, :nl1] - vd[:, 1:]) * dt / _wet_areasvol(geom, wet) / jnp.where(wet, geom.hnode_new, 1.0)
    out = jnp.where(wet, out, 0.0)
    return out if on else jnp.where(vs > 0.1, out, 0.0)


# --------------------------------------------------------------------------------------
# ver_sinking_recom_benthos (recom_sinking.F90:48-418)
# --------------------------------------------------------------------------------------
def benthos_speed(tracer_id: int, geom: SinkingGeometry, params: RecomParams,
                  switches: RecomSwitches = RecomSwitches(enable_3zoo2det=True)):
    """``Vben`` (nod2D, nl) [m s-1, positive down] of ``ver_sinking_recom_benthos`` (``:108-140``)."""
    p = params
    tid = int(tracer_id)
    vb = 0.0
    if tid in DET_IDS:
        vb = p.VDet
    if tid in PHY_IDS:
        vb = p.VPhy
    if tid in DIA_IDS:
        vb = p.VDia
    vben = jnp.broadcast_to(jnp.asarray(vb, f64), geom.zbar.shape)
    if switches.allow_var_sinking:
        vben = p.Vdet_a * jnp.abs(geom.zbar) + vben
    if switches.enable_3zoo2det and tid in DET2_IDS:
        vben = jnp.broadcast_to(jnp.asarray(p.VDet_zoo2, f64), geom.zbar.shape)
    return vben / SECONDS_PER_DAY


def ver_sinking_recom_benthos(tracer_id: int, values, geom: SinkingGeometry, dt, params: RecomParams,
                              switches: RecomSwitches = RecomSwitches(enable_3zoo2det=True)):
    """``(str_bf, benthos_tr)``: the loss of each layer to the seafloor under it
    (``-values Vben seafloor dt/areasvol/(zbar(k) - zbar(k+1))``, nod2D x nl-1) and the deposit per unit
    area into this tracer's benthos pool (nod2D, 4) [mmol m-2] (zero for the chlorophylls)."""
    nod, nl1 = values.shape
    k = jnp.arange(nl1)[None, :]
    wet = k < geom.nlev[:, None]
    vben = benthos_speed(tracer_id, geom, params, switches)[:, :nl1]
    aux = jnp.where(wet, -(values * vben) * geom.seafloor, 0.0)
    dzb = geom.zbar[:, :nl1] - geom.zbar[:, 1:]
    str_bf = jnp.where(wet, aux * dt / _wet_areasvol(geom, wet) / jnp.where(wet, dzb, 1.0), 0.0)
    add = -jnp.sum(aux * dt, axis=1) / geom.area[:, 0]
    btr = jnp.zeros((nod, 4), f64)
    pool = BENTHOS_POOL.get(int(tracer_id))
    if pool is not None:
        btr = btr.at[:, pool].set(add)
    return str_bf, btr


# --------------------------------------------------------------------------------------
# The whole per-tracer sequence
# --------------------------------------------------------------------------------------
class TracerTerms(NamedTuple):
    values_mod: jax.Array               # (nod2D, nl-1) values after the additions (into the T* update)
    dtr_bf: Optional[jax.Array]         # remin tracers
    vert_sink: Optional[jax.Array]      # sinking tracers
    str_bf: Optional[jax.Array]
    wsink: Optional[jax.Array]          # (nod2D, nl) Wvel_flux (Fortran's Sinkvel*_tr), sinking tracers
    benthos_tr: Optional[jax.Array]     # (nod2D, 4)
    ballast: Optional[Ballast]          # the scaling factors at this tracer (diagnostics)
    rho_particle: Optional[Tuple[jax.Array, jax.Array]]


class LoopContext(NamedTuple):
    """The tracer-independent inputs of the loop (T/S final for the step, so computed once)."""
    visc: jax.Array            # (nod2D, nl-1) get_seawater_viscosity
    rho_seawater: jax.Array    # (nod2D, nl-1) ballast's in-situ density


def loop_context(temp, salt, geom: SinkingGeometry) -> LoopContext:
    temp, salt = jnp.asarray(temp, f64), jnp.asarray(salt, f64)
    return LoopContext(visc=get_seawater_viscosity(temp, salt, geom.nlev),
                       rho_seawater=seawater_density(temp, salt, geom))


def tracer_terms(tracer_id: int, current, ctx: LoopContext, geom: SinkingGeometry, decay_benthos, dt,
                 params: RecomParams, switches: RecomSwitches = RecomSwitches(enable_3zoo2det=True), *,
                 need_ballast: Optional[bool] = None) -> TracerTerms:
    """Steps 2-3 of the module docstring for ONE BGC tracer. ``current`` (nod2D, nl-1, ntr) is the BGC stack
    as Fortran holds it when this tracer's turn comes (finished tracers final, the rest at the start of the
    loop); this tracer's own column is its start-of-loop value. ``need_ballast`` (default: detritus only)
    computes the densities and scaling factors (Fortran computes them for every sinking tracer; only the
    detritus speeds read them)."""
    tid = int(tracer_id)
    v = current[..., SLOT[tid]]
    nl1 = v.shape[1]
    wet = jnp.arange(nl1)[None, :] < geom.nlev[:, None]
    dtr_bf = vert_sink = str_bf = wsink = btr = bal = rho = None
    vmod = v
    if tid in REMIN_IDS:
        dtr_bf = diff_ver_recom_expl(tid, decay_benthos, geom, dt, params)
        vmod = jnp.where(wet, v + dtr_bf, v)
    if tid in SINKING_IDS:
        if need_ballast is None:
            need_ballast = tid in DET_IDS or tid in DET2_IDS
        if need_ballast:
            rho = particle_densities(current, geom.nlev, params)
            bal = ballast(rho[0], rho[1], ctx.rho_seawater, ctx.visc, geom.nlev, params)
        wsink = sinking_velocity(tid, bal, geom, dt, params, switches)
        vert_sink = ver_sinking_recom(tid, v, wsink, geom, dt, params)
        str_bf, btr = ver_sinking_recom_benthos(tid, v, geom, dt, params, switches)
        vmod = jnp.where(wet, (v + vert_sink) + str_bf, v)
    return TracerTerms(values_mod=vmod, dtr_bf=dtr_bf, vert_sink=vert_sink, str_bf=str_bf, wsink=wsink,
                       benthos_tr=btr, ballast=bal, rho_particle=rho)


class LoopTermsResult(NamedTuple):
    values_mod: jax.Array      # (nod2D, nl-1, ntr) each tracer's values after the additions (T* input)
    bgc: jax.Array             # (nod2D, nl-1, ntr) the final tracers (``finish`` outputs)
    benthos: jax.Array         # (nod2D, 4) Benthos after the merge
    benthos_tr: jax.Array      # (nod2D, 4, ntr) the per-tracer deposits (Benthos_tr)
    terms: Optional[Dict[int, TracerTerms]]   # per tracer ID, with ``diagnostics=True``


def _schedule(batched: bool) -> Tuple[Tuple[int, ...], ...]:
    """The tracer IDs finished together, in order. Sequential: Fortran's loop, one by one."""
    if not batched:
        return tuple((i,) for i in BGC_IDS)
    dets = set(DET_IDS) | set(DET2_IDS)
    rest = tuple(i for i in BGC_IDS if i not in dets)
    return (rest,) + tuple(zip(DET_IDS, DET2_IDS))


Finish = Callable[[Tuple[int, ...], jax.Array], jax.Array]


class _Columns:
    """A BGC stack held as one (nod2D, nl-1) column per slot: ``cols[..., slot]`` is that column, as
    :func:`tracer_terms` and :func:`particle_densities` read a stack."""

    def __init__(self, cols):
        self.cols = cols

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] is Ellipsis and isinstance(key[1], int)):
            raise TypeError(f"_Columns supports [..., slot] only, got {key!r}")
        return self.cols[key[1]]


def recom_tracer_loop_terms(bgc, temp, salt, geom: SinkingGeometry, decay_benthos, benthos, dt,
                            params: RecomParams, finish: Finish, *,
                            switches: RecomSwitches = RecomSwitches(enable_3zoo2det=True),
                            batched: bool = True, diagnostics: bool = False) -> LoopTermsResult:
    """REcoM's part of FESOM2's tracer loop for the whole BGC stack, in Fortran's order.

    Inputs (stage 3 hands them over):

    * ``bgc`` (nod2D, nl-1, ntr): the BGC tracers at the start of the tracer loop = stage 1's output
      (``Stage1Result.bgc``), State.bgc slot order;
    * ``temp``, ``salt`` (nod2D, nl-1): T and S FINAL for this step (after their own T* and implicit
      diffusion, S BEFORE the ``S_FLOOR`` clip);
    * ``geom``: :class:`SinkingGeometry` at the tracer step (``hnode``/``hnode_new`` = the old/new layer
      thicknesses of the T* update, ``zbar``/``z`` the ones the tracer step sees);
    * ``decay_benthos`` (nod2D, 4) = ``Stage1Result.sfc.GlodecayBenthos``; ``benthos`` (nod2D, 4) =
      ``Stage1Result.benthos``; ``dt`` the tracer step [s] (``k·dt`` under the BGC-every-k option);
    * ``finish(ids, values_mod)``: stage 3's completion of the tracers ``ids`` (a tuple of tracer IDs):
      given their modified values (nod2D, nl-1, len(ids)) it returns their FINAL values this step, i.e.
      the T* update with ``values_mod`` in place of ``values`` (``del_ttf += values_mod*(hnode -
      hnode_new)``, ``values = values_mod + del_ttf/hnode_new``, with the tracer's advected + Redi
      ``del_ttf``), the implicit vertical diffusion with ``bc_surface``. It is called once per group of
      :func:`_schedule` (``batched``: 5 calls; sequential: 31).

    Returns the modified values, the final stack, the merged ``Benthos`` and the per-tracer deposits; with
    ``diagnostics`` the :class:`TracerTerms` of every tracer (ballast included for all sinking tracers).
    """
    bgc = jnp.asarray(bgc, f64)
    ctx = loop_context(temp, salt, geom)
    ntr = bgc.shape[-1]
    # The stack as Fortran holds it at each tracer's turn, kept as one column per slot (finished tracers final, the
    # rest at the start of the loop) and assembled once at the end: writing each group back into the full
    # (nod2D, nl-1, ntr) stack cost ten copies of it a step (5 ms on 4 GH200 in float64, job 28430158).
    cur_cols = [bgc[..., s] for s in range(ntr)]
    vmod_cols = list(cur_cols)
    btr_cols = [None] * ntr
    cur = _Columns(cur_cols)
    terms: Dict[int, TracerTerms] = {}
    for group in _schedule(batched):
        vm = []
        for tid in group:
            tt = tracer_terms(tid, cur, ctx, geom, decay_benthos, dt, params, switches,
                              need_ballast=(True if diagnostics and tid in SINKING_IDS else None))
            vm.append(tt.values_mod)
            if tt.benthos_tr is not None:
                btr_cols[SLOT[tid]] = tt.benthos_tr
            if diagnostics:
                terms[tid] = tt
        fin = finish(tuple(group), jnp.stack(vm, axis=-1))
        for k, tid in enumerate(group):
            vmod_cols[SLOT[tid]] = vm[k]
            cur_cols[SLOT[tid]] = fin[..., k]
    zero_b = jnp.zeros(bgc.shape[:1] + (4,), f64)
    btr_all = jnp.stack([zero_b if c is None else c for c in btr_cols], axis=-1)
    vmod_all = jnp.stack(vmod_cols, axis=-1)
    cur = jnp.stack(cur_cols, axis=-1)
    b = jnp.asarray(benthos, f64)
    for tid in BGC_IDS:                                   # oce_ale_tracer.F90:536-541, tracer order
        if tid in BENTHOS_POOL:
            b = b + btr_all[:, :, SLOT[tid]]
    return LoopTermsResult(values_mod=vmod_all, bgc=cur, benthos=b, benthos_tr=btr_all,
                           terms=terms if diagnostics else None)


def sinking_geometry(hnode, hnode_new, zbar, z, area, areasvol, seafloor, nlev, lat_deg,
                     saar: SaarColumns) -> SinkingGeometry:
    """Build a :class:`SinkingGeometry` (f64 arrays, int levels)."""
    a = lambda x: jnp.asarray(x, f64)
    return SinkingGeometry(hnode=a(hnode), hnode_new=a(hnode_new), zbar=a(zbar), z=a(z), area=a(area),
                           areasvol=a(areasvol), seafloor=a(seafloor), nlev=jnp.asarray(nlev, jnp.int32),
                           lat_deg=a(lat_deg), saar=saar)
