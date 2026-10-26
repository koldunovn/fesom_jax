"""The depth-masked lateral face (idemix-jax patch 0011, fix F2 of the CORE2 EKI blow-up).

``build_lateral_operator`` keeps, per neighbour slot, the two cross segments of the edge's dual
face (one per adjacent element, over the node distance) and the elements' wet interface ranges.
With ``IdemixConfig.lateral_face_mask = True`` (the default) the lateral step uses at interface
``k`` only the segments of the elements wet at ``k`` (FESOM ``ulevels-1 <= k <= nlevels-1``,
0-based; the Fortran's loops C/D/E); ``False`` is the full face at every level, bitwise the
operator of patches 0001-0010. On the shipped pi mesh with the stratification and layer
thickness of 2 host steps from the packaged PHC state and a synthetic positive forcing.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from idemix_jax import IdemixForcing, IdemixParams
from idemix_jax.mesh import (horizontal_propagation, idemix_step_columns, lateral_step_stability,
                             masks_from_levels, validate_lateral_inputs)

from fesom_jax import idemix as I

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
    rng = np.random.default_rng(20261018)
    n = int(mesh.nod2D)
    iwe0 = jnp.asarray(rng.uniform(1e-4, 5e-3, (n, mesh.nl)) * is_wet)
    forcing = IdemixForcing(jnp.asarray(rng.uniform(1e-7, 1e-6, n)),
                            jnp.asarray(rng.uniform(1e-7, 5e-6, n)))
    return dict(mesh=mesh, bvfreq=st.bvfreq, hnode=st.hnode, iwe0=iwe0, forcing=forcing,
                is_wet=is_wet, lateral=I.build_lateral_operator(mesh))


def _expected_face_weight(mesh):
    """Independent rebuild, per edge and interface, from the mesh arrays: the cross segments
    of the adjacent elements wet at k over the great-circle node distance -> {(a, b): [nl]}."""
    from fesom_jax.config import R_EARTH
    nl = mesh.nl
    edges = np.asarray(mesh.edges)
    et = np.asarray(mesh.edge_tri)
    cr = np.asarray(mesh.edge_cross_dxdy, dtype=np.float64)
    geo = np.asarray(mesh.geo_coord_nod2D, dtype=np.float64)
    ul, nlv = np.asarray(mesh.ulevels), np.asarray(mesh.nlevels)
    k = np.arange(nl)
    out = {}
    for e, (a, b) in enumerate(edges):
        la, pa, lb, pb = geo[a, 0], geo[a, 1], geo[b, 0], geo[b, 1]
        h = np.sin((pb - pa) / 2) ** 2 + np.cos(pa) * np.cos(pb) * np.sin((lb - la) / 2) ** 2
        dist = R_EARTH * 2 * np.arcsin(np.sqrt(h))
        segs = (np.hypot(cr[e, 0], cr[e, 1]), np.hypot(cr[e, 2], cr[e, 3]))
        w = np.zeros(nl)
        for side in (0, 1):
            el = et[e, side]
            if el >= 0:
                w += segs[side] * ((k >= ul[el] - 1) & (k <= nlv[el] - 1))
        out[(int(a), int(b))] = w / dist
    return out


def test_config_default_and_validate():
    cfg = I.IdemixConfig()
    assert cfg.lateral_face_mask is True
    assert I.IdemixConfig._fields[:4] == ("n_hor_iter", "lateral", "lateral_cfl_cap",
                                          "lateral_face_mask")      # positional order unchanged
    with pytest.raises(ValueError, match="lateral_face_mask"):
        I.IdemixConfig(lateral_face_mask=1).validate()


def test_face_weight_is_the_wet_elements_segments(host):
    """The per-level weight the step uses equals the independent rebuild on every edge and
    interface (rtol 1e-12: haversine in a different order), is the 2-D weight bitwise where
    both elements are wet, is symmetric/non-negative/0 on pads (validate_lateral_inputs), and
    on pi shrinks some faces."""
    mesh, lat = host["mesh"], host["lateral"]
    nl = mesh.nl
    w3 = np.asarray(I.lateral_face_weight(lat, nl, I.IdemixConfig()))
    w2 = np.asarray(lat.nbr_weight)
    idx = np.asarray(lat.nbr_idx)
    assert w3.shape == idx.shape + (nl,)
    validate_lateral_inputs(np.asarray(I.interface_area(mesh)), host["is_wet"], nbr_idx=idx,
                            nbr_weight=w3)
    want = _expected_face_weight(mesh)
    seen = 0
    for (a, b), w in want.items():
        for i, j in ((a, b), (b, a)):
            m = int(np.flatnonzero(idx[i] == j)[0])
            np.testing.assert_allclose(w3[i, m], w, rtol=1e-12, atol=0.0)
            seen += 1
    assert seen == 2 * len(want)
    full = w3 == w2[:, :, None]
    assert np.all(w3 <= w2[:, :, None])
    # where both node columns are wet the face shrinks somewhere on pi (ragged bathymetry)
    both = host["is_wet"][:, None, :] & host["is_wet"][idx]
    pad = idx == np.arange(idx.shape[0])[:, None]
    shrunk = both & ~full & ~pad[:, :, None]
    assert shrunk.sum() > 0
    # the surface interface never shrinks (no cavities on pi: every element is wet at k = 0)
    assert not shrunk[:, :, 0].any()
    # mask off: the 2-D operator itself
    assert I.lateral_face_weight(lat, nl, I.IdemixConfig(lateral_face_mask=False)) is \
        lat.nbr_weight


def _composed(h, cfg, w):
    """The old composition (patches 0001-0010): column step, v0 capped, then
    horizontal_propagation with the weight ``w``."""
    mesh = h["mesh"]
    wet = jnp.asarray(h["is_wet"])
    from fesom_jax.tke import interface_spacing
    dzt = interface_spacing(mesh, h["hnode"], None)
    res = idemix_step_columns(h["iwe0"], h["bvfreq"], h["hnode"][:, :mesh.nl - 1], dzt,
                              mesh.coriolis_node, h["forcing"].fsrf, h["forcing"].fbot, DT,
                              PARAMS, is_wet=wet)
    v0 = res.v0
    if cfg.lateral_cfl_cap > 0.0:
        v0 = jnp.minimum(v0, I._v0_cap(PARAMS, cfg, h["lateral"], DT))
    tend = horizontal_propagation(res.E_iw, v0, dzt, I.interface_area(mesh),
                                  h["lateral"].nbr_idx, w, DT, PARAMS,
                                  n_iter=int(cfg.n_hor_iter), is_wet=wet)
    return res.E_iw + DT * tend


def _run(h, cfg):
    return I.mixing_idemix(h["mesh"], h["iwe0"], h["bvfreq"], h["hnode"], h["forcing"], PARAMS,
                           cfg, h["lateral"], dt=DT)


@pytest.mark.parametrize("cap", [0.1, 1e-5])
def test_mask_off_is_bitwise_the_old_operator(host, cap):
    """lateral_face_mask=False: the step is bitwise the 0001-0010 composition with the 2-D
    full-face weight (cap 1e-5 makes the v0 cap bind on pi); mask on: bitwise the same
    composition with the per-level weight."""
    cfg = I.IdemixConfig(lateral_cfl_cap=cap, n_hor_iter=3, lateral_face_mask=False, metric="mixed",
                         lateral_face_thickness="node")
    off = _run(host, cfg)
    np.testing.assert_array_equal(np.asarray(off.iwe_new),
                                  np.asarray(_composed(host, cfg, host["lateral"].nbr_weight)))
    on_cfg = cfg._replace(lateral_face_mask=True)
    on = _run(host, on_cfg)
    w3 = I.lateral_face_weight(host["lateral"], host["mesh"].nl, on_cfg)
    np.testing.assert_array_equal(np.asarray(on.iwe_new),
                                  np.asarray(_composed(host, on_cfg, w3)))
    for name in ("iw_diss", "alpha_c", "E_iw", "KappaH", "KappaM"):     # column fields
        np.testing.assert_array_equal(np.asarray(getattr(on, name)),
                                      np.asarray(getattr(off, name)), err_msg=name)


def test_mask_changes_the_tendency_only_where_faces_shrink(host):
    """One sub-iteration (so a change cannot travel beyond the cell): the default (masked)
    lateral step differs from the unmasked one exactly on the (node, interface) cells that
    have a shrunk face to a neighbour with volume there, and on all of them the masked step
    moves less energy across."""
    mesh, lat = host["mesh"], host["lateral"]
    cfg = I.IdemixConfig(n_hor_iter=1, lateral_cfl_cap=0.0, lateral_face_thickness="node")
    on = np.asarray(_run(host, cfg).iwe_new)
    off = np.asarray(_run(host, cfg._replace(lateral_face_mask=False)).iwe_new)
    col = np.asarray(_run(host, cfg._replace(lateral=False)).iwe_new)
    idx = np.asarray(lat.nbr_idx)
    w3 = np.asarray(I.lateral_face_weight(lat, mesh.nl, cfg))
    w2 = np.asarray(lat.nbr_weight)
    hv = host["is_wet"] & (np.asarray(I.interface_area(mesh)) > 0)
    both = hv[:, None, :] & hv[idx]
    shrunk_cell = np.any(both & (w3 != w2[:, :, None]), axis=1)          # [N, nl]
    changed = on != off
    assert shrunk_cell.any()
    assert not np.any(changed & ~shrunk_cell), "a cell without a shrunk face changed"
    # a shrunk face carries a flux unless v0 or the q difference vanishes there: nearly all
    # shrunk cells change
    assert changed.sum() >= 0.9 * shrunk_cell.sum(), (changed.sum(), shrunk_cell.sum())
    assert np.all(on[~shrunk_cell] == off[~shrunk_cell])
    assert np.any(off != col)


def test_mask_conserves_energy(host):
    """The masked operator conserves the volume integral sum(area_if * dzt * E) (zero forcing,
    lateral on vs off, n_hor_iter 5, no cap)."""
    from fesom_jax.tke import interface_spacing
    mesh = host["mesh"]
    zero = IdemixForcing(jnp.zeros(mesh.nod2D), jnp.zeros(mesh.nod2D))
    vol = np.asarray(I.interface_area(mesh)) * np.asarray(
        interface_spacing(mesh, host["hnode"], None))
    cfg = I.IdemixConfig(lateral_cfl_cap=0.0)
    kw = dict(dt=DT)
    on = I.mixing_idemix(mesh, host["iwe0"], host["bvfreq"], host["hnode"], zero, PARAMS, cfg,
                         host["lateral"], **kw)
    off = I.mixing_idemix(mesh, host["iwe0"], host["bvfreq"], host["hnode"], zero, PARAMS,
                          cfg._replace(lateral=False), None, **kw)
    a, b = np.asarray(on.iwe_new), np.asarray(off.iwe_new)
    assert not np.array_equal(a, b)
    np.testing.assert_allclose(np.sum(vol * a, axis=0), np.sum(vol * b, axis=0), rtol=1e-12,
                               atol=1e-12 * np.max(np.abs(np.sum(vol * b, axis=0))))


def test_partition_on_one_shard_keeps_the_face(host):
    """partition_idemix_inputs on the serial partition carries the face fields unchanged, so
    the sharded runners' one-shard lateral step sees the same per-level weight."""
    from fesom_jax import partit
    mesh, lat = host["mesh"], host["lateral"]
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    _, lp = I.partition_idemix_inputs(host["forcing"], lat, part)
    for name in I.LateralOperator._fields:
        np.testing.assert_array_equal(np.asarray(getattr(lp, name))[0],
                                      np.asarray(getattr(lat, name)), err_msg=name)
    w_glob = np.asarray(I.lateral_face_weight(lat, mesh.nl, I.IdemixConfig()))
    lp0 = I.LateralOperator(*(np.asarray(getattr(lp, f))[0] for f in I.LateralOperator._fields))
    np.testing.assert_array_equal(np.asarray(I.lateral_face_weight(lp0, mesh.nl,
                                                                   I.IdemixConfig())), w_glob)


def test_lateral_stability_preflight(host):
    """``lateral_stability`` is ``lateral_step_stability`` on the v0 (capped) and the face the
    step uses; the masked face never raises s*mu_max on any level (its Laplacian is the full
    one minus a PSD part), and the result scales with tau_h/n_hor_iter."""
    mesh, lat = host["mesh"], host["lateral"]
    cfg = I.IdemixConfig(lateral_cfl_cap=1e-5, metric="mixed",      # binds on pi (dt = 100 s)
                         lateral_face_thickness="node")
    st = I.lateral_stability(mesh, host["iwe0"], host["bvfreq"], host["hnode"], host["forcing"],
                             PARAMS, cfg, lat, dt=DT)
    from fesom_jax.tke import interface_spacing
    dzt = interface_spacing(mesh, host["hnode"], None)
    res = idemix_step_columns(host["iwe0"], host["bvfreq"], host["hnode"][:, :mesh.nl - 1], dzt,
                              mesh.coriolis_node, host["forcing"].fsrf, host["forcing"].fbot,
                              DT, PARAMS, is_wet=jnp.asarray(host["is_wet"]))
    v0 = jnp.minimum(res.v0, I._v0_cap(PARAMS, cfg, lat, DT))
    ref = lateral_step_stability(np.asarray(v0), np.asarray(dzt),
                                 np.asarray(I.interface_area(mesh)), np.asarray(lat.nbr_idx),
                                 np.asarray(I.lateral_face_weight(lat, mesh.nl, cfg)), DT,
                                 PARAMS, n_iter=cfg.n_hor_iter, is_wet=host["is_wet"])
    np.testing.assert_allclose(st.by_level, ref.by_level, rtol=1e-9, atol=0.0)
    assert st.s_mu_max > 0.0
    off = I.lateral_stability(mesh, host["iwe0"], host["bvfreq"], host["hnode"],
                              host["forcing"], PARAMS, cfg._replace(lateral_face_mask=False),
                              lat, dt=DT)
    assert np.all(st.by_level <= off.by_level * (1 + 1e-9) + 1e-15)
    with pytest.raises(ValueError, match="lateral"):
        I.lateral_stability(mesh, host["iwe0"], host["bvfreq"], host["hnode"], host["forcing"],
                            PARAMS, cfg._replace(lateral=False), lat, dt=DT)


def test_run_config_idemix_block_takes_the_flag():
    """The yaml ``idemix:`` block reaches the flag (old campaigns: ``lateral_face_mask:
    false``); absent, the default (masked) face."""
    from fesom_jax.run_config import RunConfig
    from fesom_jax.run_config import _IDEMIX_CFG_KEYS
    assert set(_IDEMIX_CFG_KEYS) == set(I.IdemixConfig._fields)
    old = RunConfig.from_dict({"idemix": {"lateral_face_mask": False,
                                          "lateral_face_thickness": "node"}})
    old.validate()
    assert old.idemix_config() == I.IdemixConfig(lateral_face_mask=False,
                                                 lateral_face_thickness="node")
    assert RunConfig.from_dict({"idemix": {}}).idemix_config().lateral_face_mask is True
