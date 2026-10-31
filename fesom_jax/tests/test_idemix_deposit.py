"""The prescribed interior deposit (patch 0016): a static ``[nod2D, nl]`` dissipation that reaches
TKE on the path of ``iw_diss`` and never touches the wave energy.

``idemix.IdemixForcingDeposit`` carries it, ``tke_coupling`` adds it to ``iw_diss`` and to the
Rinum cap's ``alpha_c E^2``, ``budget_terms`` reports it as ``Dep``. A plain ``IdemixForcing`` is the
old path, array for array. On the shipped pi mesh after 2 host steps (no /pool files)."""

import dataclasses

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from idemix_jax import IdemixForcing, IdemixParams
from idemix_jax.mesh import masks_from_levels

from fesom_jax import cvmix_tke, idemix as I
from fesom_jax.params import Params
from fesom_jax.tke import interface_spacing

DT = 100.0
PARAMS = IdemixParams.fesom_operational()
TKE_KW = dict(c_k=0.1, c_eps=0.7, cd=3.75, alpha_tke=30.0, mxl_min=1e-8, tke_min=1e-6,
              kappaM_max=100.0, dt=1800.0)


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
    nl, n = int(mesh.nl), int(mesh.nod2D)
    is_wet = np.asarray(masks_from_levels(mesh.ulevels_nod2D - 1, mesh.nlevels_nod2D - 1, nl - 1))
    k = np.arange(nl)[None, :]
    inter = ((k > (np.asarray(mesh.ulevels_nod2D) - 1)[:, None])
             & (k < (np.asarray(mesh.nlevels_nod2D) - 1)[:, None]))
    rng = np.random.default_rng(20261031)
    iwe0 = jnp.asarray(rng.uniform(1e-4, 5e-3, (n, nl)) * is_wet)
    plain = IdemixForcing(jnp.asarray(rng.uniform(1e-7, 1e-6, n)),
                          jnp.asarray(rng.uniform(1e-7, 5e-6, n)))
    dep = rng.uniform(1e-10, 1e-8, (n, nl)) * inter
    return dict(mesh=mesh, st=st, op=op, stress=stress, is_wet=is_wet, inter=inter, iwe0=iwe0,
                plain=plain, dep=dep, with_dep=I.IdemixForcingDeposit(plain.fsrf, plain.fbot,
                                                                      jnp.asarray(dep)),
                lateral=I.build_lateral_operator(mesh))


def _run(h, forcing, cfg=I.IdemixConfig()):
    return I.mixing_idemix(h["mesh"], h["iwe0"], h["st"].bvfreq, h["st"].hnode, forcing, PARAMS,
                           cfg, h["lateral"] if cfg.lateral else None, dt=DT)


def test_without_a_deposit_tke_coupling_returns_the_arrays_themselves(host):
    cfg = I.IdemixConfig()
    out = _run(host, host["plain"])
    d, a, e = I.tke_coupling(out, cfg, deposit=I.forcing_deposit(host["plain"]))
    assert d is out.iw_diss and a is out.alpha_c and e is out.iwe_new
    assert I.budget_keys(host["plain"]) == I.BUDGET_TERMS


@pytest.mark.parametrize("lateral", [True, False])
def test_the_deposit_leaves_the_wave_energy_alone(host, lateral):
    cfg = I.IdemixConfig(lateral=lateral)
    a, b = _run(host, host["plain"], cfg), _run(host, host["with_dep"], cfg)
    for name in a._fields:
        np.testing.assert_array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)))


def test_the_deposit_joins_the_forcing_and_the_rinum_cap(host):
    cfg = I.IdemixConfig()
    out = _run(host, host["with_dep"])
    d, a, e = I.tke_coupling(out, cfg, deposit=host["dep"])
    np.testing.assert_array_equal(np.asarray(d), np.asarray(out.iw_diss) + host["dep"])
    assert a is out.alpha_c
    ac, e0, e1 = np.asarray(out.alpha_c), np.asarray(out.iwe_new), np.asarray(e)
    on = (host["dep"] > 0) & (ac > 0)
    assert on.sum() > 1000
    np.testing.assert_allclose(ac[on] * e1[on] ** 2, ac[on] * e0[on] ** 2 + host["dep"][on],
                               rtol=1e-12)
    np.testing.assert_array_equal(e1[~on], e0[~on])            # dry, surface and bottom interfaces


def test_a_zero_deposit_changes_no_tke_input(host):
    """No sqrt of a square where the deposit is 0: the column inputs stay bitwise."""
    cfg = I.IdemixConfig()
    out = _run(host, host["plain"])
    for x, y in zip(I.tke_coupling(out, cfg), I.tke_coupling(out, cfg, deposit=0.0 * host["dep"])):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_the_cap_gives_the_deposit_the_mixing_of_wave_dissipation():
    """Quiet wave field, strong deposit: in the cap the column has a low Prandtl number, and the
    same power out of the cap would mix at Prandtl 10. The column is steady under its own forcing,
    so this measures the coupling, not a transient."""
    nl, N = 12, 1
    k = np.arange(nl)[None, :]
    nlev = np.array([nl - 2], dtype=np.int32)
    interior = (k >= 1) & (k <= nlev[:, None] - 1)
    wet = k <= nlev[:, None]
    n2 = np.where(interior, 1e-7, 0.0)
    s2 = np.where(interior, 1e-11, 0.0)
    dz = 100.0 * np.ones((N, nl))
    dep = np.where(interior, 1e-9, 0.0)
    alpha = np.where(wet, 1.0, 0.0)
    e_small = np.where(wet, 1e-7, 0.0)

    def run(e, nstep=4000):
        def one(t, _):
            tke_new = cvmix_tke.integrate_tke_column(
                t, jnp.asarray(s2), jnp.asarray(n2), jnp.asarray(dz), jnp.asarray(dz),
                jnp.zeros(N), jnp.asarray(nlev), **TKE_KW, iw_diss=jnp.asarray(dep),
                alpha_c=jnp.asarray(alpha), E_iw=jnp.asarray(e))[0]
            return tke_new, None
        tke = jax.lax.scan(one, jnp.asarray(1e-6 * wet), None, length=nstep)[0]
        _, km, kh = cvmix_tke.integrate_tke_column(
            tke, jnp.asarray(s2), jnp.asarray(n2), jnp.asarray(dz), jnp.asarray(dz), jnp.zeros(N),
            jnp.asarray(nlev), **TKE_KW, iw_diss=jnp.asarray(dep), alpha_c=jnp.asarray(alpha),
            E_iw=jnp.asarray(e))
        return float(km[0, 5]), float(kh[0, 5])

    km_out, kh_out = run(e_small)                        # deposit left out of the cap
    km_in, kh_in = run(np.sqrt(e_small ** 2 + dep / np.where(wet, alpha, 1.0)))
    assert km_out / kh_out == pytest.approx(10.0, rel=1e-6)
    assert km_in / kh_in < 3.0
    assert kh_in > 2.0 * kh_out


def test_budget_reports_the_deposit_and_keeps_the_closure(host):
    mesh, st = host["mesh"], host["st"]
    cfg = I.IdemixConfig()

    class S:
        pass
    s0, s1 = S(), S()
    s0.iwe, s0.hnode = host["iwe0"], st.hnode
    s1.iwe, s1.bvfreq = _run(host, host["with_dep"]).iwe_new, st.bvfreq
    a = I.budget_terms(mesh, s0, s1, host["plain"], PARAMS, cfg, host["lateral"], dt=DT)
    b = I.budget_terms(mesh, s0, s1, host["with_dep"], PARAMS, cfg, host["lateral"], dt=DT)
    assert tuple(a) == I.BUDGET_TERMS
    assert tuple(b) == I.BUDGET_TERMS + ("Dep",) == I.budget_keys(host["with_dep"])
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))
    V = np.asarray(I.cell_area(mesh, cfg)) * np.asarray(interface_spacing(mesh, st.hnode, None))
    np.testing.assert_allclose(np.asarray(b["Dep"]), np.sum(V * host["dep"], axis=1), rtol=1e-13)


@pytest.mark.parametrize("what,msg", [("shape", "shape"), ("nan", "finite"), ("neg", "finite"),
                                      ("surface", "interior"), ("bottom", "interior")])
def test_check_deposit_refuses(host, what, msg):
    mesh = host["mesh"]
    d = host["dep"].copy()
    node = int(np.flatnonzero(host["inter"].sum(axis=1) > 2)[0])
    u = int(mesh.ulevels_nod2D[node]) - 1
    b = int(mesh.nlevels_nod2D[node]) - 1
    if what == "shape":
        d = d[:, :-1]
    elif what == "nan":
        d[node, u + 1] = np.nan
    elif what == "neg":
        d[node, u + 1] = -1e-9
    elif what == "surface":
        d[node, u] = 1e-9
    else:
        d[node, b] = 1e-9
    with pytest.raises(ValueError, match=msg):
        I.check_deposit(mesh, d)


def test_load_and_config(host, tmp_path):
    import yaml
    from fesom_jax.run_config import load_yaml
    p = tmp_path / "deposit.npz"
    np.savez(p, deposit=host["dep"])
    f = I.load_deposit(host["mesh"], host["plain"], p)
    assert isinstance(f, I.IdemixForcingDeposit)
    np.testing.assert_array_equal(np.asarray(f.deposit), host["dep"])

    base = yaml.safe_load(open(_pi_yaml()).read())
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(dict(base, idemix={"deposit_file": str(p)}, forcing={
        "kind": "core2", "start_year": 1958}, ice={}, tke={"only_tke": False})))
    assert load_yaml(str(good)).idemix["deposit_file"] == str(p)
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(dict(base, idemix={"deposit_file": str(p)})))
    with pytest.raises(ValueError, match="deposit_file"):
        load_yaml(str(bad))


def test_run_entry_refuses_a_yaml_deposit_with_a_plain_injected_forcing(host, monkeypatch):
    """``idemix.deposit_file`` is read only by ``RunConfig.idemix_inputs``, which an injected
    ``idemix_forcing`` skips: a plain ``IdemixForcing`` there would drop the deposit silently, so
    the run entry refuses it; an injected ``IdemixForcingDeposit`` is used as given."""
    from fesom_jax import partit, run
    from fesom_jax.run_config import RunConfig
    mesh = host["mesh"]
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)

    class PastTheGuard(Exception):
        pass

    def stop(*a, **k):
        raise PastTheGuard

    monkeypatch.setattr(I, "partition_idemix_inputs", stop)
    cfg = RunConfig.from_dict({"idemix": {"deposit_file": "unused.npz", "lateral": False},
                               "tke": {"only_tke": False}})
    with pytest.raises(ValueError, match="deposit would be silently ignored"):
        run.run_from_config(cfg, mesh=mesh, part=part, idemix_forcing=host["plain"])
    with pytest.raises(PastTheGuard):
        run.run_from_config(cfg, mesh=mesh, part=part, idemix_forcing=host["with_dep"])
    no_file = RunConfig.from_dict({"idemix": {"lateral": False}, "tke": {"only_tke": False}})
    with pytest.raises(PastTheGuard):
        run.run_from_config(no_file, mesh=mesh, part=part, idemix_forcing=host["plain"])


def _pi_yaml():
    import pathlib
    import fesom_jax
    return pathlib.Path(fesom_jax.__file__).resolve().parents[1] / "configs" / "pi_demo.yaml"


def test_the_osborn_path_refuses_a_deposit(host):
    from fesom_jax.step import step
    params = dataclasses.replace(Params.defaults(), idemix=PARAMS)
    with pytest.raises(ValueError, match="deposit"):
        step(host["st"], host["mesh"], host["op"], host["stress"], params, dt=DT,
             idemix_cfg=I.IdemixConfig(), idemix_forcing=host["with_dep"],
             idemix_lateral=host["lateral"])


def test_sharded_is_refused(host):
    cfg = I.IdemixConfig(lateral=False)
    with pytest.raises(NotImplementedError, match="deposit"):
        I.mixing_idemix(host["mesh"], host["iwe0"], host["st"].bvfreq, host["st"].hnode,
                        host["with_dep"], PARAMS, cfg, None, dt=DT, sharded=True)


def test_partition_keeps_the_type_and_the_field(host):
    from fesom_jax import partit
    from fesom_jax.integrate_sharded import _fold_forcing
    mesh = host["mesh"]
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    f, _ = I.partition_idemix_inputs(host["with_dep"], None, part)
    assert isinstance(f, I.IdemixForcingDeposit)
    assert np.asarray(f.deposit).shape == (1,) + host["dep"].shape
    np.testing.assert_array_equal(np.asarray(f.deposit)[0], host["dep"])
    g, _ = I.partition_idemix_inputs(host["plain"], None, part)
    assert isinstance(g, IdemixForcing)
    folded, _ = _fold_forcing(f)
    np.testing.assert_array_equal(np.asarray(folded.deposit), host["dep"])
