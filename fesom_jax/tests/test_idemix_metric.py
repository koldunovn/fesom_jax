"""``IdemixConfig.metric`` (patch 0014): which energy total the IDEMIX step conserves.

Node placement (ruling 1) runs the Fortran's element-prism column on a node's dual cell, whose area
shrinks at every sea-floor step. ``"mixed"`` (patches 0001-0013): the area-blind column conserves
``sum A0 dzt E`` while the lateral step conserves ``sum area_if dzt E`` (budget review 2026-10-26,
B1). ``"volume"`` (default): the finite-volume node column (idemix_jax ``layer_area``) on the lateral
step's cells; ``"prism"``: the lateral step on the column's prism cells ``A0 * dzt``. On the shipped
pi mesh after 2 host steps; the fixture asserts that it has both prism and narrowing nodes.
"""

import dataclasses

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from idemix_jax import IdemixForcing, IdemixParams
from idemix_jax.mesh import horizontal_propagation, idemix_step_columns, masks_from_levels

from fesom_jax import idemix as I
from fesom_jax.tke import interface_spacing

DT = 100.0
PARAMS = IdemixParams.fesom_operational()


@pytest.fixture(scope="module")
def host():
    from fesom_jax import forcing as host_forcing
    from fesom_jax import ssh
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import PACKAGED_PI_IC_DIR, cold_start_state
    from fesom_jax.step import step

    mesh = load_mesh()
    st = cold_start_state(mesh, PACKAGED_PI_IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    stress = host_forcing.surface_stress(mesh)
    st = step(st, mesh, op, stress, dt=DT, is_first_step=True)
    st = step(st, mesh, op, stress, dt=DT)
    is_wet = np.asarray(masks_from_levels(mesh.ulevels_nod2D - 1, mesh.nlevels_nod2D - 1,
                                          mesh.nl - 1))
    rng = np.random.default_rng(20261026)
    n = int(mesh.nod2D)
    iwe0 = jnp.asarray(rng.uniform(1e-4, 5e-3, (n, mesh.nl)) * is_wet)
    forcing = IdemixForcing(jnp.asarray(rng.uniform(1e-7, 1e-6, n)),
                            jnp.asarray(rng.uniform(1e-7, 5e-6, n)))
    dzt = np.asarray(interface_spacing(mesh, st.hnode, None))
    a = np.asarray(mesh.areasvol)
    nl = mesh.nl
    u0, b0 = np.asarray(mesh.ulevels_nod2D) - 1, np.asarray(mesh.nlevels_nod2D) - 1
    prism = np.array([np.all(a[i, u0[i]:b0[i]] == a[i, u0[i]]) for i in range(n)])
    return dict(mesh=mesh, st=st, op=op, stress=stress, bvfreq=st.bvfreq, hnode=st.hnode, iwe0=iwe0,
                forcing=forcing, is_wet=is_wet, dzt=dzt, lateral=I.build_lateral_operator(mesh),
                prism=prism, A0=np.asarray(I.node_area(mesh)), nl=nl)


def _core(h, cfg, forcing=None):
    return I._idemix_core(h["mesh"], h["iwe0"], h["bvfreq"], h["hnode"],
                          h["forcing"] if forcing is None else forcing, PARAMS, cfg,
                          h["lateral"] if cfg.lateral else None, DT, None)


def test_mixed_is_the_old_composition_bitwise(host):
    """metric 'mixed': the area-blind column + horizontal_propagation on interface_area, exactly
    the pre-0014 operations."""
    assert I.IdemixConfig().metric == "volume" and I.IdemixConfig._fields[-1] == "metric"
    cfg = I.IdemixConfig(metric="mixed")
    res, iwe_new, dzt, is_wet = _core(host, cfg)
    mesh = host["mesh"]
    ref = idemix_step_columns(host["iwe0"], host["bvfreq"], host["hnode"][:, :mesh.nl - 1], dzt,
                              mesh.coriolis_node, host["forcing"].fsrf, host["forcing"].fbot, DT,
                              PARAMS, is_wet=is_wet)
    v0 = jnp.minimum(ref.v0, I._v0_cap(PARAMS, cfg, host["lateral"], DT))
    tend = horizontal_propagation(ref.E_iw, v0, dzt, I.interface_area(mesh), host["lateral"].nbr_idx,
                                  I.lateral_face_weight(host["lateral"], mesh.nl, cfg), DT, PARAMS,
                                  n_iter=int(cfg.n_hor_iter), is_wet=is_wet)
    np.testing.assert_array_equal(np.asarray(iwe_new), np.asarray(ref.E_iw + DT * tend))
    np.testing.assert_array_equal(np.asarray(res.iw_diss), np.asarray(ref.iw_diss))


def test_volume_column_conserves_the_lateral_cells_and_takes_the_full_input(host):
    """metric 'volume', lateral off: per node sum(area_if dzt Tdif) = 0 and sum(area_if dzt
    (Tsur + Tbot)) = A0 (fs + fb); prism nodes are the area-blind column bitwise; tapering nodes
    deposit bottom power above their deepest interface."""
    cfg = I.IdemixConfig(lateral=False)             # the default metric
    assert cfg.metric == "volume"
    res, _, dzt, is_wet = _core(host, cfg)
    V = np.asarray(I.interface_area(host["mesh"])) * np.asarray(dzt)
    Tdif = np.where(is_wet, np.asarray(res.Tdif), 0.0)
    scale = np.sum(np.abs(V * Tdif), axis=1)
    assert np.all(np.abs(np.sum(V * Tdif, axis=1)) <= 1e-12 * scale + 1e-30)
    fs, fb = np.asarray(host["forcing"].fsrf), np.asarray(host["forcing"].fbot)
    np.testing.assert_allclose(np.sum(V * np.asarray(res.Tsur + res.Tbot), axis=1),
                               host["A0"] * (fs + fb), rtol=1e-12)
    old, _, _, _ = _core(host, cfg._replace(metric="mixed"))
    p = host["prism"]
    assert p.any() and (~p).any()
    for name in ("E_iw", "iw_diss", "alpha_c", "Tdif", "Tbot", "Tsur"):
        np.testing.assert_array_equal(np.asarray(getattr(res, name))[p],
                                      np.asarray(getattr(old, name))[p], err_msg=name)
    k = np.arange(host["nl"])[None, :]
    above_bottom = k < (np.asarray(host["mesh"].nlevels_nod2D) - 1)[:, None]
    assert np.any((np.asarray(res.Tbot) > 0) & above_bottom & ~p[:, None])


def test_volume_metric_budget_closes_with_zero_dif(host):
    """budget_terms with metric 'volume', lateral on: Dif = 0 per node, P_bot = A0 fb, the lateral
    term sums to 0, and dEdt = P_surf + P_bot - D + Lat."""
    cfg = I.IdemixConfig()
    assert cfg.metric == "volume"
    res, iwe_new, _, _ = _core(host, cfg)

    class S:
        pass
    s0, s1 = S(), S()
    s0.iwe, s0.hnode = host["iwe0"], host["hnode"]
    s1.iwe, s1.bvfreq = iwe_new, host["bvfreq"]
    t = {k: np.asarray(v) for k, v in I.budget_terms(host["mesh"], s0, s1, host["forcing"], PARAMS,
                                                     cfg, host["lateral"], dt=DT).items()}
    V = np.asarray(I.interface_area(host["mesh"])) * host["dzt"]
    gross = np.sum(np.abs(np.where(host["is_wet"], V * np.asarray(res.Tdif), 0.0)), axis=1)
    assert np.all(np.abs(t["Dif"]) <= 1e-12 * gross + 1e-25)
    np.testing.assert_allclose(t["P_bot"], host["A0"] * np.asarray(host["forcing"].fbot), rtol=1e-12)
    assert abs(t["Lat"].sum()) <= 1e-12 * np.abs(t["Lat"]).sum()
    np.testing.assert_allclose(t["dEdt"].sum(), (t["P_surf"] + t["P_bot"] - t["D"] + t["Lat"]).sum(),
                               rtol=1e-9)


def test_mixed_metric_budget_closes_with_its_spurious_terms(host):
    """budget_terms with metric 'mixed' (the old campaigns' budget): it closes, per node, only
    because Dif carries the column/lateral mismatch, and P_bot is the deepest interface's area
    alone, below the nominal A0 fb (read the two together: the docstring)."""
    cfg = I.IdemixConfig(metric="mixed")
    _, iwe_new, _, _ = _core(host, cfg)

    class S:
        pass
    s0, s1 = S(), S()
    s0.iwe, s0.hnode = host["iwe0"], host["hnode"]
    s1.iwe, s1.bvfreq = iwe_new, host["bvfreq"]
    t = {k: np.asarray(v) for k, v in I.budget_terms(host["mesh"], s0, s1, host["forcing"], PARAMS,
                                                     cfg, host["lateral"], dt=DT).items()}
    rhs = t["P_surf"] + t["P_bot"] - t["D"] + t["Dif"] + t["Lat"]
    np.testing.assert_allclose(t["dEdt"], rhs, rtol=1e-9, atol=1e-9 * np.max(np.abs(rhs)))
    nominal = host["A0"] * np.asarray(host["forcing"].fbot)
    p = host["prism"]
    np.testing.assert_allclose(t["P_bot"][p], nominal[p], rtol=1e-12)
    assert np.all(t["P_bot"][~p] < nominal[~p])
    assert np.abs(t["Dif"][~p]).sum() > 1e-6 * np.abs(t["D"][~p]).sum()


def test_prism_metric_conserves_the_prism_total(host):
    """metric 'prism': the column is unchanged; the lateral step conserves sum(A0 dzt E) (zero
    forcing, lateral on vs off) and no longer sum(area_if dzt E)."""
    mesh = host["mesh"]
    zero = IdemixForcing(jnp.zeros(mesh.nod2D), jnp.zeros(mesh.nod2D))
    cfg = I.IdemixConfig(metric="prism", lateral_cfl_cap=0.0)
    res, on, dzt, _ = _core(host, cfg, zero)
    mixed, _, _, _ = _core(host, cfg._replace(metric="mixed"), zero)
    for name in ("E_iw", "iw_diss", "Tdif"):
        np.testing.assert_array_equal(np.asarray(getattr(res, name)),
                                      np.asarray(getattr(mixed, name)), err_msg=name)
    off = np.asarray(res.E_iw)
    on = np.asarray(on)
    assert not np.array_equal(on, off)
    Vp = np.asarray(I.cell_area(mesh, cfg)) * np.asarray(dzt)
    Vi = np.asarray(I.interface_area(mesh)) * np.asarray(dzt)
    np.testing.assert_allclose(np.sum(Vp * on, axis=0), np.sum(Vp * off, axis=0), rtol=1e-12,
                               atol=1e-12 * np.max(np.abs(np.sum(Vp * off, axis=0))))
    assert abs(np.sum(Vi * (on - off))) > 1e-9 * np.sum(np.abs(Vi * (on - off)))


def test_prism_metric_budget_sums_in_the_prism_cells(host):
    """budget_terms with metric 'prism': the volumes are A0 dzt, not interface_area dzt, so Dif = 0
    per node, P_bot = A0 fb also where the node narrows, and the budget closes."""
    cfg = I.IdemixConfig(metric="prism")
    res, iwe_new, _, _ = _core(host, cfg)

    class S:
        pass
    s0, s1 = S(), S()
    s0.iwe, s0.hnode = host["iwe0"], host["hnode"]
    s1.iwe, s1.bvfreq = iwe_new, host["bvfreq"]
    t = {k: np.asarray(v) for k, v in I.budget_terms(host["mesh"], s0, s1, host["forcing"], PARAMS,
                                                     cfg, host["lateral"], dt=DT).items()}
    V = host["A0"][:, None] * host["dzt"]
    gross = np.sum(np.abs(np.where(host["is_wet"], V * np.asarray(res.Tdif), 0.0)), axis=1)
    assert np.all(np.abs(t["Dif"]) <= 1e-12 * gross + 1e-25)
    p = host["prism"]
    assert (~p).any()
    np.testing.assert_allclose(t["P_bot"], host["A0"] * np.asarray(host["forcing"].fbot), rtol=1e-12)
    np.testing.assert_allclose(t["E"], np.sum(np.where(host["is_wet"], V * np.asarray(iwe_new), 0.0),
                                              axis=1), rtol=1e-14)
    np.testing.assert_allclose(t["dEdt"].sum(), (t["P_surf"] + t["P_bot"] - t["D"] + t["Lat"]).sum(),
                               rtol=1e-9)


def test_stability_preflight_uses_the_metric_cells(host):
    from idemix_jax.mesh import lateral_step_stability
    mesh, lat = host["mesh"], host["lateral"]
    cfg = I.IdemixConfig(metric="prism")
    st = I.lateral_stability(mesh, host["iwe0"], host["bvfreq"], host["hnode"], host["forcing"],
                             PARAMS, cfg, lat, dt=DT)
    res, _, dzt, is_wet = _core(host, cfg)
    v0 = jnp.minimum(res.v0, I._v0_cap(PARAMS, cfg, lat, DT))
    ref = lateral_step_stability(np.asarray(v0), np.asarray(dzt), np.asarray(I.cell_area(mesh, cfg)),
                                 np.asarray(lat.nbr_idx), np.asarray(I.lateral_face_weight(lat, mesh.nl,
                                                                                           cfg)),
                                 DT, PARAMS, n_iter=cfg.n_hor_iter, is_wet=np.asarray(is_wet))
    np.testing.assert_allclose(st.by_level, ref.by_level, rtol=1e-9, atol=0.0)


def _cavity(mesh):
    """``mesh`` with one node whose second wet layer is wider than its first: a cavity ceiling."""
    node = 5
    u, b = int(mesh.ulevels_nod2D[node]) - 1, int(mesh.nlevels_nod2D[node]) - 2
    assert u + 1 <= b
    a = np.asarray(mesh.areasvol).copy()
    a[node, u + 1] = 1.5 * a[node, u]
    return dataclasses.replace(mesh, areasvol=jnp.asarray(a))


@pytest.mark.parametrize("lateral", [True, False])
def test_cavity_mesh_is_refused_on_the_direct_path(host, lateral):
    """The default metric needs layer areas that do not grow with depth. A mesh driven from Python
    meets the check in the column step, lateral on or off, also under a jit that closes over it;
    the two metrics that do not need it run on the same mesh."""
    from fesom_jax.params import Params
    from fesom_jax.step import step
    bad = _cavity(host["mesh"])
    cfg = I.IdemixConfig(lateral=lateral)
    lat = host["lateral"] if lateral else None
    args = (host["iwe0"], host["bvfreq"], host["hnode"], host["forcing"], PARAMS)
    grow = "do not grow with depth"
    with pytest.raises(ValueError, match=grow):
        I.mixing_idemix(bad, *args, cfg, lat, dt=DT)
    with pytest.raises(ValueError, match=grow):
        jax.jit(lambda e: I.mixing_idemix(bad, e, *args[1:], cfg, lat, dt=DT).iwe_new)(host["iwe0"])

    class S:
        pass
    s0, s1 = S(), S()
    s0.iwe, s0.hnode = host["iwe0"], host["hnode"]
    s1.iwe, s1.bvfreq = host["iwe0"], host["bvfreq"]
    with pytest.raises(ValueError, match=grow):
        I.budget_terms(bad, s0, s1, host["forcing"], PARAMS, cfg, lat, dt=DT)
    if lateral:
        with pytest.raises(ValueError, match=grow):
            I.lateral_stability(bad, *args, cfg, lat, dt=DT)
    params = dataclasses.replace(Params.defaults(), idemix=PARAMS)
    with pytest.raises(ValueError, match=grow):
        step(host["st"], bad, host["op"], host["stress"], params, dt=DT, idemix_cfg=cfg,
             idemix_forcing=host["forcing"], idemix_lateral=lat)
    for metric in ("mixed", "prism"):
        out = I.mixing_idemix(bad, *args, cfg._replace(metric=metric), lat, dt=DT)
        assert np.all(np.isfinite(np.asarray(out.iwe_new)))


def test_the_guard_leaves_a_traced_mesh_alone(host):
    """A mesh that is itself a jit argument (``step_jit``, ``integrate_jit``) cannot be read: the
    guard stands aside instead of failing on the tracer, and the step is the same."""
    mesh = host["mesh"]
    cfg = I.IdemixConfig()
    args = (host["iwe0"], host["bvfreq"], host["hnode"], host["forcing"], PARAMS)
    direct = I.mixing_idemix(mesh, *args, cfg, host["lateral"], dt=DT)
    traced = jax.jit(lambda m: I.mixing_idemix(m, *args, cfg, host["lateral"], dt=DT))(mesh)
    np.testing.assert_allclose(np.asarray(traced.iwe_new), np.asarray(direct.iwe_new), rtol=1e-12,
                               atol=1e-12 * float(np.max(np.abs(np.asarray(direct.iwe_new)))))


def test_lateral_operator_refuses_a_cavity_under_the_default_metric(host):
    mesh = host["mesh"]
    bad = _cavity(mesh)
    with pytest.raises(ValueError, match="do not grow with depth"):
        I.build_lateral_operator(bad)
    with pytest.raises(ValueError, match="do not grow with depth"):
        I.build_lateral_operator(bad, metric="volume")
    for metric in ("mixed", "prism"):
        op = I.build_lateral_operator(bad, metric=metric)
        np.testing.assert_array_equal(np.asarray(op.face_seg), np.asarray(host["lateral"].face_seg))


def test_run_config_and_run_entry_refuse_a_cavity(host, monkeypatch):
    """Both keep their own check, before anything is read: the config for a run that builds its
    inputs, the entry for one that is handed them (no operator to build, so nothing else looks at
    the mesh). ``mixed`` and ``prism`` pass both."""
    from fesom_jax import partit, run
    from fesom_jax.run_config import RunConfig
    mesh = host["mesh"]
    bad = _cavity(mesh)
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)

    class PastTheGuard(Exception):
        pass

    def stop(*a, **k):
        raise PastTheGuard

    monkeypatch.setattr("idemix_jax.load_idemix_forcing", stop)
    monkeypatch.setattr(I, "partition_idemix_inputs", stop)
    cfg = RunConfig.from_dict({"idemix": {"lateral": False}})
    with pytest.raises(ValueError, match="do not grow with depth"):
        cfg.idemix_inputs(bad)
    with pytest.raises(ValueError, match="do not grow with depth"):
        run.run_from_config(cfg, mesh=bad, part=part, idemix_forcing=host["forcing"])
    for metric in ("mixed", "prism"):
        other = RunConfig.from_dict({"idemix": {"lateral": False, "metric": metric}})
        with pytest.raises(PastTheGuard):
            other.idemix_inputs(bad)
        with pytest.raises(PastTheGuard):
            run.run_from_config(other, mesh=bad, part=part, idemix_forcing=host["forcing"])


def test_layer_area_check_and_yaml_key(host):
    I.check_layer_areas(host["mesh"])
    from types import SimpleNamespace
    bad = SimpleNamespace(nl=3, areasvol=np.array([[1.0, 2.0, 0.0]]),
                          ulevels_nod2D=np.array([1]), nlevels_nod2D=np.array([3]))
    with pytest.raises(ValueError, match="do not grow"):
        I.check_layer_areas(bad)
    zero = SimpleNamespace(nl=3, areasvol=np.array([[1.0, 0.0, 0.0]]),
                           ulevels_nod2D=np.array([1]), nlevels_nod2D=np.array([3]))
    with pytest.raises(ValueError, match="areasvol > 0 on every wet layer"):
        I.check_layer_areas(zero)
    dry = SimpleNamespace(nl=3, areasvol=np.array([[1.0, 0.0, 5.0]]),    # below the column
                          ulevels_nod2D=np.array([1]), nlevels_nod2D=np.array([2]))
    I.check_layer_areas(dry)
    with pytest.raises(ValueError, match="metric"):
        I.IdemixConfig(metric="true").validate()
    from fesom_jax.run_config import RunConfig
    c = RunConfig.from_dict({"idemix": {"metric": "volume"}})
    c.validate()
    assert c.idemix_config().metric == "volume"
    assert RunConfig.from_dict({"idemix": {}}).idemix_config().metric == "volume"
    for m in ("mixed", "prism"):
        assert RunConfig.from_dict({"idemix": {"metric": m}}).idemix_config().metric == m
