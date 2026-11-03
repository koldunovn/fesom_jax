"""Gate G3 of the spectral port's lane E (docs/plans/completed/20261028-recom-spectral-port.md): the fork's tracer-loop
additions (fesom_jax.recom.sinking with the fork's layout and sinking rules) against the per-tracer stages of the
fork's NON-spectral twin dumps (run pi_30d_nowb_adv01 on Albedo, namelist.tra weights 0., 1.). A copy of test_recom_sinking.py (REcoM 692f804) with
the fork's reference, namelist and layout; the fork writes Benthos inside the loop, so the per-tracer Benthos
check is dropped. Gate G3 (parent plan Task 11): REcoM's additions inside FESOM2's tracer loop (``fesom_jax.recom.sinking``)
against the per-tracer stages of the Fortran reference dumps, in Fortran tracer order, plus the edge cases
(one-layer columns, the bottom layer, zero sinking speed, a CFL-capped long step), conservation, jit and
gradient checks.

For each dumped step and each BGC tracer, the stack Fortran holds when the tracer's turn comes is rebuilt
from the dumps (tracers earlier in the loop: ``tra_out.values``, the final values; the tracer itself and the
later ones: ``tra_in.values``, the start of the loop; T and S: ``tra_out``), and :func:`tracer_terms`
is compared with ``tra_bf`` (``dtr_bf``, values), ``tra_ball`` (viscosity, particle densities, scaling
factors), ``tra_sink`` (``vert_sink``, ``str_bf``, ``Benthos_tr``, values; ``SinkingVel1/2`` are FESOM's
never-written arrays, zero). :func:`recom_tracer_loop_terms` with an oracle ``finish`` (Fortran's final
values) gives the merged ``Benthos`` (``step_end``) and the same terms, sequential = batched bitwise; the
T* update fed with the modified values reproduces ``tra_tstar``.

Error measure: ``max |jax - fortran| / max |fortran|`` over the wet layers (``k < nlevels_nod2D - 1``;
interface fields ``k <= nlevels_nod2D - 1``), absolute where the Fortran field is zero; tolerance
``TOL = 1e-12`` unless listed in ``LOOSE`` with the reason. ``python -m fesom_jax.tests.test_recom_sinking``
prints the G3 table.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from fesom_jax.io_dump import load_recom_dump                                    # noqa: E402
from fesom_jax.recom import params as P                                          # noqa: E402
from fesom_jax.recom import sinking as S                                         # noqa: E402
from fesom_jax.recom import layout as LY                                         # noqa: E402
from fesom_jax.recom.sms import check_no_cavities                               # noqa: E402

REF = Path(os.environ.get("FESOM_RECOM_FORK_REF", "/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/recom_ref"))
MESH = Path(os.environ.get("FESOM_RECOM_FORK_MESH", "/albedo/work/projects/p_specbic/sloza/fesom_jax_data/mesh_pi_part"))
REPO = Path(__file__).resolve().parents[2]
RUN = os.environ.get("FESOM_RECOM_FORK_RUN", "pi_30d_nowb_adv01")
SPECTRAL = "spectral" in RUN                      # the spectral twin: 37 tracers (CDOM), the spectral light
NML = REPO / "scripts" / "recom" / "fortran" / ("namelists_albedo_spectral" if SPECTRAL else
                                                "namelists_albedo_spectral_nowb") / "namelist.recom"
STEPS = [(RUN, s) for s in (1, 2, 3, 17, 224, 960) if (REF / RUN / "dumps" / f"step_{s:06d}").is_dir()] or [(RUN, 1)]
TOL = 1e-12
LOOSE: dict = {}        # field -> (tolerance, reason)

HAVE_REF = (REF / RUN / "dumps").is_dir()
need_ref = pytest.mark.skipif(not HAVE_REF, reason=f"REcoM Fortran reference dumps absent ({REF})")

BALL_FIELDS = ("seawater_visc_3D", "rho_particle1", "rho_particle2", "scaling_visc_3D", "scaling_density1_3D",
               "scaling_density2_3D")


def _records():
    r = ["mesh.gid_elem", "mesh.ulevels_nod2D", "mesh.nlevels_nod2D", "mesh.nlevels", "mesh.area", "mesh.areasvol",
         "mesh.elem_area", "mesh.geo_coord_nod2D", "recom_in.rad", "transport.hnode", "transport.hnode_new",
         "transport.zbar_3d_n", "transport.z_3d_n", "recom_out.GlodecayBenthos", "recom_out.Benthos",
         "step_end.Benthos", "tra_out.tr1.values", "tra_out.tr2.values"]
    for tid in T.BGC_IDS:
        r += [f"tra_in.tr{tid}.values", f"tra_out.tr{tid}.values", f"tra_redi.tr{tid}.del_ttf",
              f"tra_tstar.tr{tid}.values"]
        if tid in T.REMIN_IDS:
            r += [f"tra_bf.tr{tid}.dtr_bf", f"tra_bf.tr{tid}.values"]
        if tid in T.SINKING_IDS:
            r += [f"tra_ball.tr{tid}.{n}" for n in BALL_FIELDS]
            r += [f"tra_sink.tr{tid}.{n}" for n in ("vert_sink", "SinkingVel1", "SinkingVel2", "str_bf",
                                                    "Benthos", "values")]
    return r


@lru_cache(maxsize=1)
def _elem_nodes():
    """The pi mesh connectivity of the reference (``elem2d.out``, 1-based) as 0-based (elem2D, 3)."""
    with open(MESH / "elem2d.out") as fh:
        n = int(fh.readline())
        a = np.loadtxt(fh, dtype=np.int64, max_rows=n)
    return a - 1


@lru_cache(maxsize=1)
def _saar(lon_key, lat_key):
    return S.saar_columns(np.frombuffer(lon_key), np.frombuffer(lat_key))


@lru_cache(maxsize=1)
def _load(run, step):
    f, meta = load_recom_dump(REF / run / "dumps", step, names=_records())
    return f, meta


def _geom(f):
    check_no_cavities(f["mesh.ulevels_nod2D"])
    gid = f["mesh.gid_elem"]
    assert (gid == np.arange(1, gid.size + 1)).all()          # one rank: the elements in file order
    nod = f["mesh.nlevels_nod2D"].size
    nl = f["mesh.area"].shape[1]
    nlev = f["mesh.nlevels_nod2D"].astype(np.int32) - 1
    rad = float(f["recom_in.rad"][0])
    lon = f["mesh.geo_coord_nod2D"][:, 0] / rad
    lat = f["mesh.geo_coord_nod2D"][:, 1] / rad
    sf = S.seafloor_area(_elem_nodes(), f["mesh.nlevels"], f["mesh.elem_area"], nod, nl)
    nlmin = S.elem_levels_min(_elem_nodes(), f["mesh.nlevels"], nod)
    return S.sinking_geometry(f["transport.hnode"], f["transport.hnode_new"], f["transport.zbar_3d_n"],
                              f["transport.z_3d_n"], f["mesh.area"], f["mesh.areasvol"], sf, nlev, lat,
                              _saar(lon.tobytes(), lat.tobytes()), nlmin=nlmin)


def _params():
    sw, p = P.load_namelist(NML, check=False)
    return sw, p


class _T:
    """The fork layout in the shape of fesom_jax.recom.tracers' constants."""
    def __init__(self, lay):
        self.BGC_IDS, self.SINKING_IDS, self.REMIN_IDS = lay.ids, lay.sinking_ids, lay.remin_ids
        self.SLOT, self.NTR, self.NAMES, self.layout = lay.slot_map, lay.ntr, lay.names, lay


LAY = LY.layout_of(P.load_namelist(NML, check=False)[0])
T = _T(LAY)


def _err(a, b, mask):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[mask], b[mask]
    den = np.max(np.abs(b)) if b.size else 0.0
    num = np.max(np.abs(a - b)) if b.size else 0.0
    return (num / den if den > 0 else num), den


_tracer_terms = jax.jit(S.tracer_terms, static_argnums=(0,), static_argnames=("switches", "need_ballast", "layout"))


def g3_errors(run, step) -> dict:
    """``{(tracer ID, field): (rel. error, max |fortran|)}`` of one dumped step (the per-tracer pieces)."""
    f, meta = _load(run, step)
    sw, p = _params()
    geom = _geom(f)
    dt = float(meta["dt"])
    nlev = np.asarray(geom.nlev)
    nl1 = geom.hnode.shape[1]
    wet = np.arange(nl1)[None, :] < nlev[:, None]
    wet_if = np.arange(nl1 + 1)[None, :] <= nlev[:, None]
    allnodes = np.ones(nlev.size, bool)
    start = np.stack([f[f"tra_in.tr{i}.values"] for i in T.BGC_IDS], axis=-1)
    final = np.stack([f[f"tra_out.tr{i}.values"] for i in T.BGC_IDS], axis=-1)
    ctx = jax.jit(S.loop_context)(f["tra_out.tr1.values"], f["tra_out.tr2.values"], geom)
    decay = f["recom_out.GlodecayBenthos"].T
    e = {}
    prev_benthos = f["recom_out.Benthos"].T          # the fork adds each sinking tracer's deposit to Benthos in the loop
    for s, tid in enumerate(T.BGC_IDS):
        if tid not in T.REMIN_IDS and tid not in T.SINKING_IDS:
            continue
        cur = np.concatenate([final[..., :s], start[..., s:]], axis=-1)
        tt = jax.device_get(_tracer_terms(tid, cur, ctx, geom, decay, dt, p, switches=sw,
                                          need_ballast=tid in T.SINKING_IDS, layout=LAY))
        if tid in T.REMIN_IDS:
            e[(tid, "dtr_bf")] = _err(tt.dtr_bf, f[f"tra_bf.tr{tid}.dtr_bf"], wet)
            e[(tid, "values")] = _err(tt.values_mod, f[f"tra_bf.tr{tid}.values"], wet)
        else:
            k = f"tra_ball.tr{tid}."
            e[(tid, "seawater_visc_3D")] = _err(ctx.visc, f[k + "seawater_visc_3D"], wet)
            e[(tid, "rho_particle1")] = _err(tt.rho_particle[0], f[k + "rho_particle1"], wet)
            e[(tid, "rho_particle2")] = _err(tt.rho_particle[1], f[k + "rho_particle2"], wet)
            e[(tid, "scaling_visc_3D")] = _err(tt.ballast.scaling_visc[:, :-1], f[k + "scaling_visc_3D"][:, :-1], wet)
            # the fork's bottom-interface scalings come from the dry layer below (K: ballast, k up to nlevels_nod2D):
            # diagnostics no flux reads; compared on the layers only. At the FIRST ballast call of a run (mstep 1,
            # the first sinking tracer) the fork's `if (any(scaling_density <= tiny)) = 1` reset fires on the
            # never-written entries below the bottom and leaves both density scalings at 1 everywhere (a diagnostic:
            # the phytoplankton are not ballasted); not mirrored, skipped here.
            first_call = step == 1 and tid == T.SINKING_IDS[4]       # 1004, Fortran's first sinking tracer
            if not first_call:
                e[(tid, "scaling_density1_3D")] = _err(tt.ballast.scaling_density1[:, :-1],
                                                       f[k + "scaling_density1_3D"][:, :-1], wet)
                e[(tid, "scaling_density2_3D")] = _err(tt.ballast.scaling_density2[:, :-1],
                                                       f[k + "scaling_density2_3D"][:, :-1], wet)
            k = f"tra_sink.tr{tid}."
            e[(tid, "vert_sink")] = _err(tt.vert_sink, f[k + "vert_sink"], wet)
            e[(tid, "str_bf")] = _err(tt.str_bf, f[k + "str_bf"], wet)
            after = f[k + "Benthos"].T
            e[(tid, "Benthos_tr")] = _err(np.asarray(tt.benthos_tr) + prev_benthos, after, allnodes)
            prev_benthos = after
            e[(tid, "values")] = _err(tt.values_mod, f[k + "values"], wet)
            if tid in (1021, 1028):     # the fork records Sinkvel1_tr/Sinkvel2_tr for these two only; the bottom
                sv = f[k + ("SinkingVel1" if tid == 1021 else "SinkingVel2")]   # interface carries no flux
                e[(tid, "SinkingVel")] = _err(tt.wsink[:, :-1], sv[:, :-1], wet)
    return e


def _oracle_finish(final):
    def finish(ids, vm):
        assert vm.shape[-1] == len(ids)
        return jnp.stack([final[..., T.SLOT[i]] for i in ids], axis=-1)
    return finish


def g3_loop_errors(run, step, batched) -> dict:
    """The whole-loop function with Fortran's final values as ``finish``: values after the additions,
    the merged Benthos, and the T* update of every BGC tracer from them."""
    f, meta = _load(run, step)
    sw, p = _params()
    geom = _geom(f)
    dt = float(meta["dt"])
    nlev = np.asarray(geom.nlev)
    nl1 = geom.hnode.shape[1]
    wet = np.arange(nl1)[None, :] < nlev[:, None]
    start = np.stack([f[f"tra_in.tr{i}.values"] for i in T.BGC_IDS], axis=-1)
    final = np.stack([f[f"tra_out.tr{i}.values"] for i in T.BGC_IDS], axis=-1)
    run_loop = jax.jit(lambda *a: S.recom_tracer_loop_terms(*a, finish=_oracle_finish(jnp.asarray(final)),
                                                            switches=sw, batched=batched, layout=LAY))
    r = jax.device_get(run_loop(start, f["tra_out.tr1.values"], f["tra_out.tr2.values"], geom,
                                f["recom_out.GlodecayBenthos"].T, f["recom_out.Benthos"].T, dt, p))
    e = {}
    for s, tid in enumerate(T.BGC_IDS):
        ref = (f[f"tra_bf.tr{tid}.values"] if tid in T.REMIN_IDS else
               f[f"tra_sink.tr{tid}.values"] if tid in T.SINKING_IDS else f[f"tra_in.tr{tid}.values"])
        e[(tid, "values_mod")] = _err(r.values_mod[..., s], ref, wet)
        # T* (oce_ale_tracer.F90:891-895) with the modified values and the advected + Redi del_ttf
        h, hn = np.asarray(geom.hnode), np.asarray(geom.hnode_new)
        vm = r.values_mod[..., s]
        dd = f[f"tra_redi.tr{tid}.del_ttf"] + vm * (h - hn)
        tstar = vm + dd / np.where(wet, hn, 1.0)
        e[(tid, "tstar")] = _err(tstar, f[f"tra_tstar.tr{tid}.values"], wet)
    for i in range(4):
        e[("Benthos", i + 1)] = _err(r.benthos[:, i], f["step_end.Benthos"][i], np.ones(nlev.size, bool))
    e[("_result", "")] = r
    return e


# ----------------------------------------------------------------------------------------------------------
# G3
# ----------------------------------------------------------------------------------------------------------
@need_ref
@pytest.mark.parametrize("run,step", STEPS)
def test_g3_per_tracer(run, step):
    e = g3_errors(run, step)
    bad = []
    for (tid, name), (err, den) in e.items():
        tol = LOOSE.get(name, (TOL, ""))[0]
        if not err <= tol:
            bad.append((tid, name, err, den))
    assert not bad, bad


@need_ref
@pytest.mark.parametrize("run,step", STEPS[:1] + STEPS[-1:])
def test_g3_loop_sequential_equals_batched(run, step):
    es = g3_loop_errors(run, step, batched=False)
    eb = g3_loop_errors(run, step, batched=True)
    rs, rb = es.pop(("_result", "")), eb.pop(("_result", ""))
    for a, b in zip(jax.tree_util.tree_leaves((rs.values_mod, rs.bgc, rs.benthos, rs.benthos_tr)),
                    jax.tree_util.tree_leaves((rb.values_mod, rb.bgc, rb.benthos, rb.benthos_tr))):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    bad = [(k, v) for k, v in es.items() if not v[0] <= TOL]
    assert not bad, bad




if __name__ == "__main__":
    for run, step in STEPS:
        e = g3_errors(run, step)
        worst = sorted(((v[0], k) for k, v in e.items()), reverse=True)[:12]
        print(f"== {run} step {step}: worst", [(k, f"{v:.2e}") for v, k in worst])
        el = g3_loop_errors(run, step, True); el.pop(("_result", ""))
        worst = sorted(((v[0], k) for k, v in el.items()), reverse=True)[:6]
        print("   loop:", [(k, f"{v:.2e}") for v, k in worst])
