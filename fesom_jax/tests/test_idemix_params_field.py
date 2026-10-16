"""``Params.idemix_field``: per-node log multipliers of IDEMIX's ``mu0`` and of the bottom tidal
forcing (idemix-jax patch 0010, paper-readiness Task 4.5: the adjoint sensitivity map).

``IdemixParamsField(log_mu0, log_fbot)`` (each ``[nod2D]`` or ``None``) makes the node ``i``
run with ``mu0 * exp(log_mu0[i])`` and the bottom energy flux ``fbot[i] * exp(log_fbot[i])``.
The gradient w.r.t. the field at 0 is ``dJ/d log mu0(x)`` / ``dJ/d log fbot(x)`` per node, and
its sum over the nodes is the gradient w.r.t. the scalar ``log mu0`` (the wiring check).
``None`` (default) is the scalar path, bitwise unchanged; an all-``None`` field likewise.

On the shipped pi mesh with the stratification and layer thickness of 2 host steps from the
packaged PHC state and a synthetic positive forcing (no /pool files needed).
"""

import dataclasses

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from idemix_jax import IdemixForcing, IdemixParams
from idemix_jax.mesh import masks_from_levels

from fesom_jax import idemix as I
from fesom_jax.params import Params

DT = 100.0
PARAMS = IdemixParams(*(jnp.asarray(v, jnp.float64)
                        for v in IdemixParams.fesom_operational()))
FIELDS = ("iwe_new", "iw_diss", "alpha_c", "E_iw", "KappaH", "KappaM")


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
    rng = np.random.default_rng(20261016)
    n = int(mesh.nod2D)
    iwe0 = jnp.asarray(rng.uniform(1e-4, 5e-3, (n, mesh.nl)) * is_wet)
    forcing = IdemixForcing(jnp.asarray(rng.uniform(1e-7, 1e-6, n)),
                            jnp.asarray(rng.uniform(1e-7, 5e-6, n)))
    return dict(mesh=mesh, bvfreq=st.bvfreq, hnode=st.hnode, iwe0=iwe0, forcing=forcing,
                lateral=I.build_lateral_operator(mesh), rng=rng, n=n)


def _run(h, field, *, lateral=True, params=PARAMS, forcing=None):
    cfg = I.IdemixConfig(lateral=lateral)
    return I.mixing_idemix(h["mesh"], h["iwe0"], h["bvfreq"], h["hnode"],
                           h["forcing"] if forcing is None else forcing, params, cfg,
                           h["lateral"] if lateral else None, dt=DT, params_field=field)


def _same(a, b):
    for f in FIELDS:
        np.testing.assert_array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f)),
                                      err_msg=f)


def test_default_and_empty_field_are_the_scalar_path_bitwise(host):
    a = I.mixing_idemix(host["mesh"], host["iwe0"], host["bvfreq"], host["hnode"],
                        host["forcing"], PARAMS, I.IdemixConfig(), host["lateral"], dt=DT)
    _same(a, _run(host, None))
    _same(a, _run(host, I.IdemixParamsField()))
    assert I.IdemixParamsField()._fields == ("log_mu0", "log_fbot")
    assert Params.defaults().idemix_field is None


def test_zero_field_is_the_scalar_path(host):
    """exp(0) = 1 exactly, so the per-node path evaluates the same arithmetic per column."""
    z = jnp.zeros(host["n"])
    ref = _run(host, None)
    for field in (I.IdemixParamsField(log_mu0=z), I.IdemixParamsField(log_fbot=z),
                  I.IdemixParamsField(z, z)):
        out = _run(host, field)
        for f in FIELDS:
            np.testing.assert_allclose(np.asarray(getattr(out, f)),
                                       np.asarray(getattr(ref, f)), rtol=1e-14, atol=0.0,
                                       err_msg=f)


def test_uniform_field_is_the_scaled_scalar(host):
    c = 0.3
    u = jnp.full(host["n"], c)
    p = PARAMS._replace(mu0=PARAMS.mu0 * np.exp(c))
    np.testing.assert_allclose(np.asarray(_run(host, I.IdemixParamsField(log_mu0=u)).iw_diss),
                               np.asarray(_run(host, None, params=p).iw_diss),
                               rtol=1e-12, atol=0.0)
    fb = IdemixForcing(host["forcing"].fsrf, host["forcing"].fbot * np.exp(c))
    np.testing.assert_allclose(np.asarray(_run(host, I.IdemixParamsField(log_fbot=u)).iwe_new),
                               np.asarray(_run(host, None, forcing=fb).iwe_new),
                               rtol=1e-12, atol=0.0)


def test_the_field_acts_per_node(host):
    """Lateral off: a log_mu0 / log_fbot change at one node changes that column only."""
    i = int(np.argmax(np.asarray(host["mesh"].nlevels_nod2D)))
    e = jnp.zeros(host["n"]).at[i].set(0.5)
    ref = _run(host, None, lateral=False)
    for field, name in ((I.IdemixParamsField(log_mu0=e), "alpha_c"),
                        (I.IdemixParamsField(log_fbot=e), "E_iw")):
        out = _run(host, field, lateral=False)
        d = np.any(np.asarray(getattr(out, name)) != np.asarray(getattr(ref, name)), axis=1)
        assert d[i] and d.sum() == 1, (name, np.flatnonzero(d)[:10])


def test_field_gradient_sums_to_the_scalar_gradient(host):
    """d/d log mu0 (scalar) = sum_x d/d log_mu0(x): in one reverse pass, and against the
    scalar path's own gradient (another program: round-off)."""
    mesh = host["mesh"]
    wet = np.asarray(masks_from_levels(mesh.ulevels_nod2D - 1, mesh.nlevels_nod2D - 1,
                                       mesh.nl - 1))
    w = jnp.asarray(host["rng"].uniform(0.5, 1.5, wet.shape) * wet)

    def J(theta, lm, lf, field=True):
        p = PARAMS._replace(mu0=jnp.exp(theta))
        f = I.IdemixParamsField(lm, lf) if field else None
        o = _run(host, f, params=p)
        return jnp.sum(w * (o.iw_diss * 1e9 + o.iwe_new * 1e2))

    th = jnp.log(PARAMS.mu0)
    z = jnp.zeros(host["n"])
    g_th, g_m, g_f = jax.grad(J, argnums=(0, 1, 2))(th, z, z)
    g_scalar = jax.grad(lambda t: J(t, None, None, field=False))(th)
    assert np.isfinite(float(g_th)) and float(g_th) != 0.0
    np.testing.assert_allclose(float(jnp.sum(g_m)), float(g_th), rtol=1e-11)
    np.testing.assert_allclose(float(jnp.sum(g_m)), float(g_scalar), rtol=1e-11)
    # the bottom-forcing sum is the derivative along a uniform log scaling of fbot (central FD)
    h = 1e-4
    u = jnp.full(host["n"], h)
    fd = (float(J(th, z, u)) - float(J(th, z, -u))) / (2 * h)
    np.testing.assert_allclose(float(jnp.sum(g_f)), fd, rtol=1e-6)
    assert np.count_nonzero(np.asarray(g_m)) > host["n"] // 2


def test_bad_shapes_and_sharded_are_refused(host):
    with pytest.raises(ValueError, match="log_mu0"):
        _run(host, I.IdemixParamsField(log_mu0=jnp.zeros(host["n"] + 1)))
    with pytest.raises(ValueError, match="log_fbot"):
        _run(host, I.IdemixParamsField(log_fbot=jnp.zeros((host["n"], 1))))
    with pytest.raises(NotImplementedError, match="idemix_field"):
        I.mixing_idemix(host["mesh"], host["iwe0"], host["bvfreq"], host["hnode"],
                        host["forcing"], PARAMS, I.IdemixConfig(lateral=False), None, dt=DT,
                        sharded=True, params_field=I.IdemixParamsField(
                            log_mu0=jnp.zeros(host["n"])))


def test_params_field_is_a_pytree_leaf_set(host):
    base = Params.defaults()
    n0 = len(jax.tree_util.tree_leaves(base))
    z = jnp.zeros(host["n"])
    p = dataclasses.replace(base, idemix_field=I.IdemixParamsField(log_mu0=z))
    assert len(jax.tree_util.tree_leaves(p)) == n0 + 1
    p2 = dataclasses.replace(base, idemix_field=I.IdemixParamsField(z, z))
    assert len(jax.tree_util.tree_leaves(p2)) == n0 + 2
    g = jax.grad(lambda q: jnp.sum(q.idemix_field.log_mu0 * 2.0))(p)
    np.testing.assert_array_equal(np.asarray(g.idemix_field.log_mu0), 2.0)


def test_step_refuses_a_field_without_idemix(host):
    from fesom_jax import forcing as host_forcing
    from fesom_jax import ssh
    from fesom_jax.phc_ic import PACKAGED_PI_IC_DIR, cold_start_state
    from fesom_jax.step import step
    mesh = host["mesh"]
    st = cold_start_state(mesh, PACKAGED_PI_IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    p = dataclasses.replace(Params.defaults(),
                            idemix_field=I.IdemixParamsField(log_mu0=jnp.zeros(host["n"])))
    with pytest.raises(ValueError, match="idemix_field"):
        step(st, mesh, op, host_forcing.surface_stress(mesh), p, dt=DT, is_first_step=True)
