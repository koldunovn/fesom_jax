"""``IdemixConfig.lateral_face_thickness`` (patch 0015): the face thickness of the lateral flux.

The host's two-point flux takes ``0.5 (dzt_i + dzt_j)`` from the node columns; the Fortran takes the
elements' ``dz_trr`` (``gen_modules_cvmix_idemix.F90:570-598``): the mean of the two elements' where
both are wet (loop C, ``:686``), the wet element's own where only one is (loops D/E, ``:714``,
``:742``) -- half the element's own bottom layer at its bottom interface, where both node columns
may still be a full spacing (the face-thickness deviation F1, SPEC_S4_COUPLED §14.1; budget review
2026-10-26, B6). ``"element"``, the default, multiplies the depth-masked face by ``dz_el / (0.5
(dzt_i + dzt_j))``; ``"node"`` is the old face. On pi after 2 host steps."""

import numpy as np
import pytest

import jax.numpy as jnp

pytest.importorskip("idemix_jax")  # the IDEMIX library is a separate package

from idemix_jax import IdemixForcing, IdemixParams
from idemix_jax.mesh import masks_from_levels, validate_lateral_inputs

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
    rng = np.random.default_rng(20261027)
    n = int(mesh.nod2D)
    iwe0 = jnp.asarray(rng.uniform(1e-4, 5e-3, (n, mesh.nl)) * is_wet)
    forcing = IdemixForcing(jnp.asarray(rng.uniform(1e-7, 1e-6, n)),
                            jnp.asarray(rng.uniform(1e-7, 5e-6, n)))
    dzt = interface_spacing(mesh, st.hnode, None)
    return dict(mesh=mesh, bvfreq=st.bvfreq, hnode=st.hnode, iwe0=iwe0, forcing=forcing,
                is_wet=is_wet, dzt=dzt, lateral=I.build_lateral_operator(mesh))


def _expected_w_dz(mesh, hnode):
    """Independent per-edge rebuild of (face segment / node distance) x Fortran dz_el per
    interface: {(a, b): [nl]}. helem = vertex mean of hnode on the element's wet layers."""
    from fesom_jax.config import R_EARTH
    nl = mesh.nl
    edges, et = np.asarray(mesh.edges), np.asarray(mesh.edge_tri)
    cr = np.asarray(mesh.edge_cross_dxdy, dtype=np.float64)
    geo = np.asarray(mesh.geo_coord_nod2D, dtype=np.float64)
    ul, nlv = np.asarray(mesh.ulevels), np.asarray(mesh.nlevels)
    en = np.asarray(mesh.elem_nodes)
    h = np.asarray(hnode)

    def dz_trr(e):                          # F90:570-577, 0-based
        u, b = ul[e] - 1, nlv[e] - 1
        he = h[en[e]].mean(axis=0)
        d = np.zeros(nl)
        d[u] = he[u] / 2
        for k in range(u + 1, b):
            d[k] = he[k - 1] / 2 + he[k] / 2
        d[b] = he[b - 1] / 2
        return d, u, b

    out = {}
    for e_, (a, b) in enumerate(edges):
        la, pa, lb, pb = geo[a, 0], geo[a, 1], geo[b, 0], geo[b, 1]
        hv = np.sin((pb - pa) / 2) ** 2 + np.cos(pa) * np.cos(pb) * np.sin((lb - la) / 2) ** 2
        dist = R_EARTH * 2 * np.arcsin(np.sqrt(hv))
        segs = (np.hypot(cr[e_, 0], cr[e_, 1]), np.hypot(cr[e_, 2], cr[e_, 3]))
        parts = []
        for side in (0, 1):
            el = et[e_, side]
            if el >= 0:
                parts.append((segs[side], *dz_trr(el)))
        w = np.zeros(nl)
        for k in range(nl):
            wet = [(s, d[k]) for s, d, u, bb in parts if u <= k <= bb]
            if len(wet) == 2:
                w[k] = (wet[0][0] + wet[1][0]) / dist * 0.5 * (wet[0][1] + wet[1][1])
            elif len(wet) == 1:
                w[k] = wet[0][0] / dist * wet[0][1]
        out[(int(a), int(b))] = w
    return out


def test_node_is_the_face_weight_bitwise(host):
    assert I.IdemixConfig().lateral_face_thickness == "element"
    cfg = I.IdemixConfig(lateral_face_thickness="node")
    w = I.lateral_weight(host["mesh"], host["lateral"], cfg, host["hnode"], host["dzt"])
    np.testing.assert_array_equal(np.asarray(w), np.asarray(
        I.lateral_face_weight(host["lateral"], host["mesh"].nl, cfg)))


def test_element_thickness_is_the_fortran_dz_el(host):
    """weight x 0.5 (dzt_i + dzt_j) == segment/distance x dz_el on every edge and wet interface
    (rtol 1e-12), symmetric, and thinner than the node face somewhere (element bottoms)."""
    mesh, lat = host["mesh"], host["lateral"]
    cfg = I.IdemixConfig()
    w = np.asarray(I.lateral_weight(mesh, lat, cfg, host["hnode"], host["dzt"]))
    idx = np.asarray(lat.nbr_idx)
    dzt = np.asarray(host["dzt"])
    validate_lateral_inputs(np.asarray(I.interface_area(mesh)), host["is_wet"], nbr_idx=idx,
                            nbr_weight=w)
    want = _expected_w_dz(mesh, host["hnode"])
    both = host["is_wet"][:, None, :] & host["is_wet"][idx]
    for (a, b), wd in want.items():
        for i, j in ((a, b), (b, a)):
            m = int(np.flatnonzero(idx[i] == j)[0])
            got = w[i, m] * 0.5 * (dzt[i] + dzt[j])
            sel = both[i, m]
            np.testing.assert_allclose(got[sel], wd[sel], rtol=1e-12, atol=0.0)
    w_node = np.asarray(I.lateral_face_weight(lat, mesh.nl, cfg))
    assert np.any(w < w_node * (1 - 1e-6))


def test_element_thickness_conserves_energy(host):
    mesh = host["mesh"]
    zero = IdemixForcing(jnp.zeros(mesh.nod2D), jnp.zeros(mesh.nod2D))
    vol = np.asarray(I.interface_area(mesh)) * np.asarray(host["dzt"])
    cfg = I.IdemixConfig(lateral_cfl_cap=0.0)
    on = I.mixing_idemix(mesh, host["iwe0"], host["bvfreq"], host["hnode"], zero, PARAMS, cfg,
                         host["lateral"], dt=DT)
    node = I.mixing_idemix(mesh, host["iwe0"], host["bvfreq"], host["hnode"], zero, PARAMS,
                           cfg._replace(lateral_face_thickness="node"), host["lateral"], dt=DT)
    off = I.mixing_idemix(mesh, host["iwe0"], host["bvfreq"], host["hnode"], zero, PARAMS,
                          cfg._replace(lateral=False), None, dt=DT)
    a, b, c = np.asarray(on.iwe_new), np.asarray(off.iwe_new), np.asarray(node.iwe_new)
    assert not np.array_equal(a, c)
    np.testing.assert_allclose(np.sum(vol * a, axis=0), np.sum(vol * b, axis=0), rtol=1e-12,
                               atol=1e-12 * np.max(np.abs(np.sum(vol * b, axis=0))))


def test_partition_carries_face_el_and_validate(host):
    from fesom_jax import partit
    mesh, lat = host["mesh"], host["lateral"]
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    _, lp = I.partition_idemix_inputs(host["forcing"], lat, part)
    np.testing.assert_array_equal(np.asarray(lp.face_el)[0], np.asarray(lat.face_el))
    with pytest.raises(ValueError, match="lateral_face_thickness"):
        I.IdemixConfig(lateral_face_thickness="fem").validate()
    with pytest.raises(ValueError, match="lateral_face_mask"):
        I.IdemixConfig(lateral_face_mask=False).validate()      # the default thickness is element
    with pytest.raises(ValueError, match="lateral_face_mask"):  # not only at validate()
        I.lateral_weight(mesh, lat, I.IdemixConfig(lateral_face_mask=False), host["hnode"],
                         host["dzt"])
    I.IdemixConfig(lateral_face_mask=False, lateral_face_thickness="node").validate()
    from fesom_jax.run_config import RunConfig
    assert RunConfig.from_dict({"idemix": {}}).idemix_config().lateral_face_thickness == "element"
    c = RunConfig.from_dict({"idemix": {"lateral_face_thickness": "node"}})
    c.validate()
    assert c.idemix_config().lateral_face_thickness == "node"


def test_partition_maps_face_el_to_the_element_lanes(host):
    """The step indexes its own mesh's elements: a device whose element list is not the identity
    gets element lanes, not global ids (the ids stay the same elements)."""
    import dataclasses

    from fesom_jax import partit
    mesh, lat = host["mesh"], host["lateral"]
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    perm = np.random.default_rng(20261030).permutation(int(mesh.elem2D)).astype(np.int32)
    _, lp = I.partition_idemix_inputs(None, lat, dataclasses.replace(part, myList_elem2D=(perm,)))
    got, glob = np.asarray(lp.face_el)[0], np.asarray(lat.face_el)
    has = glob >= 0
    assert has.any() and (~has).any()
    np.testing.assert_array_equal(perm[got[has]], glob[has])
    assert np.all(got[~has] == -1)


def test_element_absent_from_the_device_is_a_missing_element(host):
    """An element the device does not hold has no thickness to read: its face slot says so with
    the empty range as well as ``face_el = -1``, or ``lateral_weight`` would read element lane 0
    (``P > 1`` with lateral propagation is refused, so this is a device's halo lanes only)."""
    import dataclasses

    from fesom_jax import partit
    mesh, lat = host["mesh"], host["lateral"]
    part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    keep = int(mesh.elem2D) // 2                      # identity lanes for the elements it holds
    _, lp = I.partition_idemix_inputs(
        None, lat, dataclasses.replace(part, myList_elem2D=(np.arange(keep, dtype=np.int32),)))
    glob = np.asarray(lat.face_el)
    gone, held = glob >= keep, (glob >= 0) & (glob < keep)
    assert gone.any() and held.any()
    get = lambda name: np.asarray(getattr(lp, name))[0]             # noqa: E731
    assert np.all(get("face_el")[gone] == -1) and np.all(get("face_bot")[gone] == -1)
    assert np.all(get("face_top")[gone] >= mesh.nl) and np.all(get("face_seg")[gone] == 0.0)
    for name in ("face_el", "face_top", "face_bot", "face_seg"):
        np.testing.assert_array_equal(get(name)[held], np.asarray(getattr(lat, name))[held], name)
    # the weight of the device's operator has no thickness from a lane that is not the element
    w = np.asarray(I.lateral_weight(mesh, lp._replace(
        nbr_idx=lp.nbr_idx[0], nbr_weight=lp.nbr_weight[0], resolution=lp.resolution[0],
        face_seg=lp.face_seg[0], face_top=lp.face_top[0], face_bot=lp.face_bot[0],
        face_el=lp.face_el[0]), I.IdemixConfig(), host["hnode"], host["dzt"]))
    assert np.all(np.isfinite(w))
    ref = np.asarray(I.lateral_weight(mesh, lat._replace(
        face_el=jnp.asarray(np.where(gone, -1, glob)),
        face_top=jnp.asarray(np.where(gone, mesh.nl, np.asarray(lat.face_top))),
        face_bot=jnp.asarray(np.where(gone, -1, np.asarray(lat.face_bot))),
        face_seg=jnp.asarray(np.where(gone, 0.0, np.asarray(lat.face_seg)))),
        I.IdemixConfig(), host["hnode"], host["dzt"]))
    np.testing.assert_array_equal(w, ref)


def test_a_device_holds_every_element_of_its_owned_nodes(host):
    """Why dropping an absent element can touch halo nodes only: on the shipped 2-device pi
    partition no owned node loses a face."""
    import pathlib

    import fesom_jax
    from fesom_jax import partit
    mesh, lat = host["mesh"], host["lateral"]
    part = partit.read_partition(pathlib.Path(fesom_jax.__file__).parent / "data" / "mesh_pi", 2)
    _, lp = I.partition_idemix_inputs(None, lat, part)
    glob = np.asarray(lat.face_el)
    for d in range(2):
        ml = np.asarray(part.myList_nod2D[d])
        got = np.asarray(lp.face_el)[d, :ml.size]
        idx = np.asarray(lp.nbr_idx)[d, :ml.size]
        reach = idx != np.arange(ml.size)[:, None]        # a neighbour on this device
        lost = reach[:, :, None] & (glob[ml] >= 0) & (got < 0)
        assert (glob[ml[:int(part.myDim_nod2D[d])]] >= 0).any()
        assert not lost[:int(part.myDim_nod2D[d])].any()
