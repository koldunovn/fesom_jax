"""Gate G3 (parent plan Task 11): REcoM's additions inside FESOM2's tracer loop (``fesom_jax.recom.sinking``)
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
from fesom_jax.recom import tracers as T                                         # noqa: E402
from fesom_jax.recom.sms import check_no_cavities                               # noqa: E402

REF = Path(os.environ.get("FESOM_RECOM_REF", "/work/ab0995/a270088/recom_ref"))
MESH = REF / "mesh_pi_part"
REPO = Path(__file__).resolve().parents[2]
NML = REPO / "scripts" / "recom" / "fortran" / "namelists" / "namelist.recom"
STEPS = [("pi_30d_jan", s) for s in (1, 2, 17, 224, 225, 960)] + [("pi_1d_jul", s) for s in (1, 17)]
TOL = 1e-12
LOOSE: dict = {}        # field -> (tolerance, reason)

HAVE_REF = (REF / "pi_30d_jan" / "dumps").is_dir()
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
                                                    "Benthos_tr", "Benthos", "values")]
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
    return S.sinking_geometry(f["transport.hnode"], f["transport.hnode_new"], f["transport.zbar_3d_n"],
                              f["transport.z_3d_n"], f["mesh.area"], f["mesh.areasvol"], sf, nlev, lat,
                              _saar(lon.tobytes(), lat.tobytes()))


def _params():
    sw, p = P.load_namelist(NML)
    P.check_supported(sw)
    return sw, p


def _err(a, b, mask):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[mask], b[mask]
    den = np.max(np.abs(b)) if b.size else 0.0
    num = np.max(np.abs(a - b)) if b.size else 0.0
    return (num / den if den > 0 else num), den


_tracer_terms = jax.jit(S.tracer_terms, static_argnums=(0,), static_argnames=("switches", "need_ballast"))


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
    for s, tid in enumerate(T.BGC_IDS):
        if tid not in T.REMIN_IDS and tid not in T.SINKING_IDS:
            continue
        cur = np.concatenate([final[..., :s], start[..., s:]], axis=-1)
        tt = jax.device_get(_tracer_terms(tid, cur, ctx, geom, decay, dt, p, switches=sw,
                                          need_ballast=tid in T.SINKING_IDS))
        if tid in T.REMIN_IDS:
            e[(tid, "dtr_bf")] = _err(tt.dtr_bf, f[f"tra_bf.tr{tid}.dtr_bf"], wet)
            e[(tid, "values")] = _err(tt.values_mod, f[f"tra_bf.tr{tid}.values"], wet)
        else:
            k = f"tra_ball.tr{tid}."
            e[(tid, "seawater_visc_3D")] = _err(ctx.visc, f[k + "seawater_visc_3D"], wet)
            e[(tid, "rho_particle1")] = _err(tt.rho_particle[0], f[k + "rho_particle1"], wet)
            e[(tid, "rho_particle2")] = _err(tt.rho_particle[1], f[k + "rho_particle2"], wet)
            e[(tid, "scaling_visc_3D")] = _err(tt.ballast.scaling_visc, f[k + "scaling_visc_3D"], wet_if)
            e[(tid, "scaling_density1_3D")] = _err(tt.ballast.scaling_density1, f[k + "scaling_density1_3D"],
                                                   wet_if)
            e[(tid, "scaling_density2_3D")] = _err(tt.ballast.scaling_density2, f[k + "scaling_density2_3D"],
                                                   wet_if)
            k = f"tra_sink.tr{tid}."
            e[(tid, "vert_sink")] = _err(tt.vert_sink, f[k + "vert_sink"], wet)
            e[(tid, "str_bf")] = _err(tt.str_bf, f[k + "str_bf"], wet)
            e[(tid, "Benthos_tr")] = _err(tt.benthos_tr, f[k + "Benthos_tr"].T, allnodes)
            e[(tid, "values")] = _err(tt.values_mod, f[k + "values"], wet)
            e[(tid, "SinkingVel")] = (float(max(np.abs(f[k + "SinkingVel1"]).max(),
                                                np.abs(f[k + "SinkingVel2"]).max())), 0.0)
            e[(tid, "Benthos_const")] = _err(f[k + "Benthos"], f["recom_out.Benthos"], np.ones((4,), bool))
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
                                                            switches=sw, batched=batched))
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
        if name == "SinkingVel":
            if err != 0.0:
                bad.append((tid, name, err))
        elif not err <= tol:
            bad.append((tid, name, err, den))
    assert not bad, bad


@need_ref
@pytest.mark.parametrize("run,step", [("pi_30d_jan", 960), ("pi_1d_jul", 17), ("pi_30d_jan", 1)])
def test_g3_loop_sequential_equals_batched(run, step):
    es = g3_loop_errors(run, step, batched=False)
    eb = g3_loop_errors(run, step, batched=True)
    rs, rb = es.pop(("_result", "")), eb.pop(("_result", ""))
    for a, b in zip(jax.tree_util.tree_leaves((rs.values_mod, rs.bgc, rs.benthos, rs.benthos_tr)),
                    jax.tree_util.tree_leaves((rb.values_mod, rb.bgc, rb.benthos, rb.benthos_tr))):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    bad = [(k, v) for k, v in es.items() if not v[0] <= TOL]
    assert not bad, bad


@need_ref
def test_saar_out_of_bounds_quirk():
    """``gsw_saar``'s ``p_ref(999)`` at coastal nodes reads 0 in the reference binary: with any other cap the
    dumped density scaling of those columns is missed, with 0 it is matched (the G3 test above)."""
    f, meta = _load("pi_30d_jan", 960)
    sw, p = _params()
    geom = _geom(f)
    rad = float(f["recom_in.rad"][0])
    lon, lat = (f["mesh.geo_coord_nod2D"][:, i] / rad for i in (0, 1))
    t = S.load_saar_table()
    ix = np.floor(1.0 + 90.0 * np.where(lon < 0, lon + 360.0, lon) / 360.0).astype(int) - 1
    iy = np.floor(1.0 + 44.0 * (lat + 86.0) / 176.0).astype(int) - 1
    land = np.zeros(lon.size, bool)
    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
        land |= t.ndepth_ref[iy + dj, ix + di] == 999
    assert 0 < land.sum() < lon.size and np.all(np.asarray(geom.saar.p_cap)[land] == S.SAAR_OOB_P_REF)
    nlev = np.asarray(geom.nlev)
    wet = np.arange(geom.hnode.shape[1])[None, :] < nlev[:, None]
    ref = f["tra_ball.tr1008.scaling_density1_3D"][:, :-1]
    cur = np.stack([f[f"tra_out.tr{i}.values"] if i < 1008 else f[f"tra_in.tr{i}.values"] for i in T.BGC_IDS], -1)
    rho1, rho2 = S.particle_densities(jnp.asarray(cur), geom.nlev, p)
    for cap, expect_match in ((S.SAAR_OOB_P_REF, True), (1e10, False)):
        g = geom._replace(saar=S.saar_columns(lon, lat, oob_p_ref=cap))
        ctx = S.loop_context(f["tra_out.tr1.values"], f["tra_out.tr2.values"], g)
        sd1 = np.asarray(S.ballast(rho1, rho2, ctx.rho_seawater, ctx.visc, g.nlev, p).scaling_density1)[:, :-1]
        m = wet & land[:, None]
        err = np.max(np.abs(sd1 - ref)[m]) / np.max(np.abs(ref)[wet])
        assert (err < 1e-13) if expect_match else (err > 1e-9), (cap, err)


# ----------------------------------------------------------------------------------------------------------
# Synthetic columns: edge cases, conservation, gradients
# ----------------------------------------------------------------------------------------------------------
def _toy(nod=6, nl=8, seed=0):
    """A small zstar-like geometry: column depths 1..nl-1 layers (node 0 has ONE layer), layer
    thicknesses 10..80 m, areas shrinking with depth, and a toy BGC stack."""
    rng = np.random.default_rng(seed)
    nlev = np.array([1, 2, nl - 1, nl - 1, 4, 3][:nod], np.int32)
    h = np.tile(np.linspace(10.0, 80.0, nl - 1), (nod, 1)) * (1.0 + 0.05 * rng.random((nod, nl - 1)))
    hn = h * (1.0 + 1e-4 * rng.standard_normal((nod, nl - 1)))
    zbar = np.concatenate([np.zeros((nod, 1)), -np.cumsum(h, axis=1)], axis=1)
    z = 0.5 * (zbar[:, :-1] + zbar[:, 1:])
    a0 = 1e9 * (1.0 + rng.random(nod))
    frac = np.linspace(1.0, 0.6, nl)[None, :]
    area = a0[:, None] * frac
    k = np.arange(nl)[None, :]
    area = np.where(k <= nlev[:, None], area, 0.0)
    areasvol = np.where(k < nlev[:, None], area, 0.0)
    seafloor = np.zeros((nod, nl - 1))
    for n in range(nod):
        m = nlev[n]
        drop = -np.diff(area[n, :m + 1])                     # area lost under each layer
        seafloor[n, :m] = drop
        seafloor[n, m - 1] += area[n, m]                     # the rest at the bottom layer
    bgc = np.zeros((nod, nl - 1, T.NTR))
    for s in range(T.NTR):
        bgc[..., s] = 1e-3 + rng.random((nod, nl - 1)) * (s + 1)
    temp = 2.0 + 18.0 * rng.random((nod, nl - 1))
    salt = 33.0 + 3.0 * rng.random((nod, nl - 1))
    lon = np.array([-30.0, 10.0, 150.0, -150.0, 20.0, 60.0][:nod])     # node 4 sits in the Baltic box
    lat = np.array([-60.0, 0.0, 30.0, 45.0, 58.0, -20.0][:nod])
    geom = S.sinking_geometry(h, hn, zbar, z, area, areasvol, seafloor, nlev, lat, S.saar_columns(lon, lat))
    decay = 1e-6 * (1.0 + rng.random((nod, 4)))
    benthos = 10.0 * (1.0 + rng.random((nod, 4)))
    return geom, bgc, temp, salt, decay, benthos


@pytest.fixture(scope="module")
def toy():
    sw, p = _params()
    return (sw, p) + _toy()


def test_one_layer_and_bottom_conservation(toy):
    sw, p, geom, bgc, temp, salt, decay, benthos = toy
    ctx = S.loop_context(temp, salt, geom)
    dt = 2700.0
    vol = np.asarray(geom.areasvol[:, :-1] * geom.hnode_new)
    for tid in (1007, 1025):                                 # both detritus classes sink in the column
        tt = S.tracer_terms(tid, bgc, ctx, geom, decay, dt, p, sw)
        vs = np.asarray(tt.vert_sink)
        assert np.all(vs[0] == 0.0)                         # one layer: no interior interface, no flux
        # flux form without surface or bottom flux: the column content is unchanged
        col = np.sum(vs * vol, axis=1)
        assert np.all(np.abs(col) <= 1e-12 * np.sum(np.abs(vs) * vol, axis=1) + 1e-300)
        # the benthic deposit is exactly what str_bf removes (dzb = hnode here up to the zstar change)
        dzb = np.asarray(geom.zbar[:, :-1] - geom.zbar[:, 1:])
        loss = -np.sum(np.asarray(tt.str_bf) * np.asarray(geom.areasvol[:, :-1]) * dzb, axis=1)
        dep = np.asarray(tt.benthos_tr).sum(axis=1) * np.asarray(geom.area[:, 0])
        np.testing.assert_allclose(dep, loss, rtol=1e-13)
        assert np.all(np.asarray(tt.str_bf)[0, 0] < 0.0)    # the one-layer column still deposits
    # the return flux of a remin tracer: everything enters the column, all of it at the one-layer node
    for tid in T.REMIN_IDS:
        d = np.asarray(S.diff_ver_recom_expl(tid, decay, geom, dt, p))
        tot = np.asarray(S.benthic_bottom_flux(tid, decay, p) * geom.area[:, 0]) * dt
        np.testing.assert_allclose(np.sum(d * vol, axis=1), tot, rtol=1e-13)
        assert d[0, 1:].tolist() == [0.0] * (d.shape[1] - 1)


def test_zero_sinking_speed(toy):
    sw, p, geom, bgc, temp, salt, decay, benthos = toy
    ctx = S.loop_context(temp, salt, geom)
    p0 = p._replace(VDet=0.0)
    w = S.sinking_velocity(1007, S.ballast(*S.particle_densities(bgc, geom.nlev, p0), ctx.rho_seawater,
                                           ctx.visc, geom.nlev, p0), geom, 2700.0, p0, sw)
    # Vsink = VDet = 0 is not > 0.1: no water-column sinking (static skip) although the ballasted speed is not 0
    assert np.any(np.asarray(w) != 0.0)
    assert np.all(np.asarray(S.ver_sinking_recom(1007, bgc[..., T.SLOT[1007]], w, geom, 2700.0, p0)) == 0.0)
    # the same with a traced VDet (jnp.where path)
    f = jax.jit(lambda v: S.ver_sinking_recom(1007, bgc[..., T.SLOT[1007]], w, geom, 2700.0, p0._replace(VDet=v)))
    assert np.all(np.asarray(f(0.0)) == 0.0) and np.any(np.asarray(f(20.0)) != 0.0)
    # phytoplankton and diatoms (VPhy = VDia = 0): no vert_sink, but the benthos deposit at Vdet_a |zbar|
    for tid in (1004, 1006, 1013, 1016, 1020):
        tt = S.tracer_terms(tid, bgc, ctx, geom, decay, 2700.0, p, sw)
        assert np.all(np.asarray(tt.vert_sink) == 0.0) and np.any(np.asarray(tt.str_bf) != 0.0)
    assert np.all(np.asarray(S.tracer_terms(1006, bgc, ctx, geom, decay, 2700.0, p, sw).benthos_tr) == 0.0)


def test_cfl_cap_long_step(toy):
    sw, p, geom, bgc, temp, salt, decay, benthos = toy
    ctx = S.loop_context(temp, salt, geom)
    bal = S.ballast(*S.particle_densities(bgc, geom.nlev, p), ctx.rho_seawater, ctx.visc, geom.nlev, p)
    dz = np.asarray(S._tracer_point_distance(geom))
    kk = np.arange(dz.shape[1])[None, :]
    on = kk <= np.asarray(geom.nlev)[:, None]
    w_free = np.asarray(S.sinking_velocity(1025, bal, geom, 1.0, p, sw))     # dt = 1 s: never capped
    for k_every in (1, 4, 40):
        dt = 2700.0 * k_every
        w = np.asarray(S.sinking_velocity(1025, bal, geom, dt, p, sw))
        cap = S.CFL_SAFETY * dz / dt
        np.testing.assert_allclose(w[on], np.copysign(np.minimum(np.abs(w_free), cap), w_free)[on], rtol=4e-16)
        vs = np.asarray(S.ver_sinking_recom(1025, bgc[..., T.SLOT[1025]], w, geom, dt, p))
        assert np.all(np.isfinite(vs))
    # at 40 x 2700 s the cap is what sets the speed in most of the column
    assert np.mean((np.abs(w_free) > cap)[on]) > 0.5


def _sequential_reference(bgc, temp, salt, geom, decay, benthos, dt, p, sw, finish):
    """The Fortran loop written out tracer by tracer, independently of the schedule code."""
    ctx = S.loop_context(temp, salt, geom)
    cur = jnp.asarray(bgc)
    b = jnp.asarray(benthos)
    for s, tid in enumerate(T.BGC_IDS):
        tt = S.tracer_terms(tid, cur, ctx, geom, decay, dt, p, sw)
        cur = cur.at[..., s].set(finish((tid,), tt.values_mod[..., None])[..., 0])
        if tt.benthos_tr is not None:
            b = b + tt.benthos_tr
    return cur, b


def test_loop_schedules_jit(toy):
    sw, p, geom, bgc, temp, salt, decay, benthos = toy

    def finish(ids, vm):          # a nonlinear stand-in for T* + implicit diffusion
        return vm * 0.97 + 0.01 * jnp.roll(vm, 1, axis=1) ** 2

    ref_bgc, ref_b = _sequential_reference(bgc, temp, salt, geom, decay, benthos, 2700.0, p, sw, finish)
    out = {}
    for batched in (False, True):
        f = jax.jit(lambda b0, t, s_, pp: S.recom_tracer_loop_terms(b0, t, s_, geom, decay, benthos, 2700.0, pp,
                                                                    finish, switches=sw, batched=batched))
        r = out[batched] = f(bgc, temp, salt, p)
        # eager (op by op) vs jitted (fused): XLA's own association, a few ulp
        assert np.max(np.abs(np.asarray(r.bgc) - np.asarray(ref_bgc))) <= 1e-15 * np.max(np.abs(ref_bgc))
        np.testing.assert_allclose(np.asarray(r.benthos), np.asarray(ref_b), rtol=1e-14)
    # the two schedules compute the same terms from the same inputs: equal to round-off of the fusion
    assert np.max(np.abs(np.asarray(out[True].bgc) - np.asarray(out[False].bgc))) <= 1e-15 * np.max(np.abs(ref_bgc))
    # the detritus order matters: a different density history changes the result
    g = S.tracer_terms(1008, bgc, S.loop_context(temp, salt, geom), geom, decay, 2700.0, p, sw)
    assert not np.array_equal(np.asarray(g.values_mod), np.asarray(r.values_mod[..., T.SLOT[1008]]))


def test_gradient_vs_fd(toy):
    sw, p, geom, bgc, temp, salt, decay, benthos = toy
    names = ("w_ref1", "rho_POC", "Vdet_a", "rho_CaCO3", "VDet")   # (w_ref2: capped at 250 m/d here)

    def finish(ids, vm):
        return vm

    def cost(theta, dt=2700.0):
        pp = p._replace(**dict(zip(names, theta)))
        r = S.recom_tracer_loop_terms(bgc, temp, salt, geom, decay, benthos, dt, pp, finish, switches=sw)
        vol = geom.areasvol[:, :-1] * geom.hnode_new
        # the detritus mass in the upper half of the columns + the benthic carbon pool
        k = jnp.arange(vol.shape[1])[None, :] < 3
        det = sum(jnp.sum(jnp.where(k, r.values_mod[..., T.SLOT[i]] ** 2 * vol, 0.0)) for i in (1008, 1026))
        return det * 1e-12 + jnp.sum(r.benthos[:, 1] * geom.area[:, 0]) * 1e-9

    theta = jnp.array([getattr(p, n) for n in names])
    g = np.asarray(jax.jit(jax.grad(cost))(theta))
    jc = jax.jit(cost)
    assert np.all(np.isfinite(g)) and np.all(g != 0)
    for i, n in enumerate(names):
        h = 1e-5 * float(theta[i])
        e = jnp.zeros(len(names)).at[i].set(h)
        fd = (float(jc(theta + e)) - float(jc(theta - e))) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-5), n


# ----------------------------------------------------------------------------------------------------------
# TEOS-10 pieces
# ----------------------------------------------------------------------------------------------------------
def test_saar_table_and_baltic():
    t = S.load_saar_table()
    assert t.saar_ref.shape == (45, 45, 91) and t.ndepth_ref.shape == (45, 91)
    assert t.p_ref[0] == 0.0 and t.p_ref[-1] == 6131.0 and t.longs_ref[-1] == 360.0
    assert set(np.unique(t.ndepth_ref)) <= set(range(1, 46)) | {999}
    lon = np.array([20.0, 20.0, -30.0])
    lat = np.array([58.0, 40.0, -60.0])
    c = S.saar_columns(lon, lat)
    assert np.asarray(c.baltic).tolist() == [True, False, False]
    sp = jnp.full((3, 2), 7.0)
    pres = jnp.array([[10.0, 50.0]] * 3)
    sa = np.asarray(S.gsw_sa_from_sp(sp, pres, c))
    np.testing.assert_allclose(sa[0], ((S.GSW_SSO - 0.087) / 35.0) * 7.0 + 0.087, rtol=1e-15)


def test_gsw_against_python_gsw():
    gsw = pytest.importorskip("gsw")
    rng = np.random.default_rng(1)
    n = 200
    lon = rng.uniform(-180, 180, n)
    lat = rng.uniform(-80, 85, n)
    c = S.saar_columns(lon, lat)
    # only nodes whose four corners all have data (the p_ref(999) quirk aside) compare with GSW-C
    ok = np.asarray(c.p_cap) > 0
    p = rng.uniform(1, 3000, (n, 3))
    sp = rng.uniform(30, 38, (n, 3))
    pt = rng.uniform(-1, 25, (n, 3))
    sa = np.asarray(S.gsw_sa_from_sp(jnp.asarray(sp), jnp.asarray(p), c))
    ref = gsw.SA_from_SP(sp, p, lon[:, None], lat[:, None])
    good = ok[:, None] & np.isfinite(ref) & ~np.asarray(c.baltic)[:, None]
    np.testing.assert_allclose(sa[good], ref[good], rtol=1e-12)
    ct = np.asarray(S.gsw_ct_from_pt(jnp.asarray(sa), jnp.asarray(pt)))
    np.testing.assert_allclose(ct, gsw.CT_from_pt(sa, pt), rtol=1e-12, atol=1e-13)
    rho = np.asarray(S.gsw_rho(jnp.asarray(sa), jnp.asarray(ct), jnp.asarray(p)))
    np.testing.assert_allclose(rho, gsw.rho(sa, ct, p), rtol=1e-13)


# ----------------------------------------------------------------------------------------------------------
def _table():
    """Print the G3 table: per dumped step, the max error per field over the tracer groups."""
    groups = {"remin": T.REMIN_IDS, "det1": T.DET_IDS, "det2": T.DET2_IDS,
              "phy+dia": tuple(i for i in T.SINKING_IDS if i not in T.DET_IDS + T.DET2_IDS)}
    for run, step in STEPS:
        e = g3_errors(run, step)
        el = g3_loop_errors(run, step, batched=True)
        el.pop(("_result", ""))
        print(f"== {run} mstep {step}")
        for g, ids in groups.items():
            fields = sorted({n for (t, n) in e if t in ids})
            row = []
            for n in fields:
                vals = [e[(t, n)][0] for t in ids if (t, n) in e]
                row.append(f"{n} {max(vals):.1e}")
            print(f"  {g:8s} " + ", ".join(row))
        print("  loop     values_mod {:.1e}, tstar {:.1e}, Benthos {:.1e}".format(
            max(v[0] for (t, n), v in el.items() if n == "values_mod"),
            max(v[0] for (t, n), v in el.items() if n == "tstar"),
            max(v[0] for (t, n), v in el.items() if t == "Benthos")))


if __name__ == "__main__":
    _table()
