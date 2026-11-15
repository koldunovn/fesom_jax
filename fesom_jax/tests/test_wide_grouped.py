"""The wide halos' point-to-point maps (ice_wide.grouped_wide_maps, fesom3_jax's ``--halo-wide grouped``): every lane
gets exactly the value the compact all_gather exchange (ice_wide._xfer) gives it, for the entry (a shorter sent field),
a refresh and the exit; on the host (numpy, the maps alone) and inside shard_map on fake CPU devices (the ppermute
rounds; the grouped NCCL call moves the same rows on the GPU)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from fesom_jax import ice_wide
from fesom_jax.halo import compact_exchange_maps


def _maps(P=5, Lfn=40, Ln=60, Lfe=70, Le=90, seed=0):
    """Synthetic all_gather maps of node and element lanes: device r owns its first ``my[r]`` lanes (identity), the
    lanes up to ``n[r]`` read a random other device's owned lane, the rest are pads (own lane 0)."""
    rng = np.random.default_rng(seed)

    def one(Lf, L):
        my = rng.integers(5, Lf // 2, P)
        n = np.minimum(L, my + rng.integers(1, L - Lf // 2, P))
        sd = np.repeat(np.arange(P)[:, None], L, axis=1)
        sl = np.zeros((P, L), np.int64)
        for r in range(P):
            sl[r, :my[r]] = np.arange(my[r])
            for lane in range(my[r], n[r]):
                d = rng.choice([x for x in range(P) if x != r])
                sd[r, lane], sl[r, lane] = d, rng.integers(0, my[d])
        return sd.astype(np.int32), sl.astype(np.int32)

    return one(Lfn, Ln), one(Lfe, Le)


def _reference(xn, xe, sd_n, sl_n, sd_e, sl_e):
    """The compact all_gather exchange's values, by its definition: a remote lane its source's row, an own lane
    its own row ``src_lane`` (identity, or lane 0 for a pad)."""
    P = len(xn)
    outn = np.stack([[xn[sd_n[r, i]][sl_n[r, i]] for i in range(sd_n.shape[1])] for r in range(P)])
    oute = np.stack([[xe[sd_e[r, i]][sl_e[r, i]] for i in range(sd_e.shape[1])] for r in range(P)])
    return outn, oute


def _simulate(xn, xe, maps, meta, Ln):
    """The point-to-point exchange on the host: pack, move every round's chunks, read out."""
    perms, slots, offs, sendn = meta
    P = len(xn)
    cn, ce = xn.shape[2], xe.shape[2]
    C = max(cn, ce)
    pad = lambda a: np.concatenate([a, np.zeros(a.shape[:2] + (C - a.shape[2],))], axis=2)   # noqa: E731
    x = np.concatenate([pad(xn), pad(xe)], axis=1)
    buf = np.where(maps["sv"][..., None], np.take_along_axis(x, maps["si"][..., None], axis=1), 0.0)
    recv = np.zeros_like(buf)
    for k, (p, s, o) in enumerate(zip(perms, slots, offs)):
        for d, e in p:
            n = sendn[k][d]
            assert maps["sv"][d, o:o + s].sum() == n and maps["sv"][d, o:o + n].all()
            recv[e, o:o + n] = buf[d, o:o + n]
    out = np.where(maps["hm"][..., None], np.take_along_axis(recv, maps["cp"][..., None], axis=1),
                   np.take_along_axis(x, maps["self"][..., None], axis=1))
    return out[:, :Ln, :cn], out[:, Ln:, :ce]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("which", ["entry", "refresh"])
def test_grouped_maps_give_the_compact_values(seed, which):
    (sd_n, sl_n), (sd_e, sl_e) = _maps(seed=seed)
    P, Ln = sd_n.shape
    Le = sd_e.shape[1]
    Lfn = 40 if which == "entry" else Ln          # the entry sends a standard (shorter) field
    Lfe = 70 if which == "entry" else Le
    rng = np.random.default_rng(10 + seed)
    xn = rng.standard_normal((P, Lfn, 2))
    xe = rng.standard_normal((P, Lfe, 5))
    maps, meta = ice_wide.grouped_wide_maps(sd_n, sl_n, sd_e, sl_e, Lfn)
    on, oe = _simulate(xn, xe, maps, meta, Ln)
    rn, re_ = _reference(xn, xe, sd_n, sl_n, sd_e, sl_e)
    np.testing.assert_array_equal(on, rn)
    np.testing.assert_array_equal(oe, re_)
    # a round is a partial permutation (one send and one receive a device)
    for p in meta[0]:
        assert len({d for d, _ in p}) == len(p) and len({e for _, e in p}) == len(p)


def test_compact_reference_matches_the_all_gather_reader():
    """_reference is the compact exchange's rule (compact_exchange_maps' read into [field; gathered])."""
    (sd_n, sl_n), _ = _maps(seed=3)
    P, Ln = sd_n.shape
    rng = np.random.default_rng(3)
    x = rng.standard_normal((P, 40))
    send, read = compact_exchange_maps(sd_n, sl_n, field_len=40)
    gathered = np.concatenate([x[d][send[d]] for d in range(P)])
    want = np.stack([np.concatenate([x[r], gathered])[read[r]] for r in range(P)])
    got = np.stack([[x[sd_n[r, i]][sl_n[r, i]] for i in range(Ln)] for r in range(P)])
    np.testing.assert_array_equal(got, want)


def test_xfer_grouped_in_shard_map_matches_xfer():
    """ice_wide._xfer_grouped (the ppermute rounds off the grouped transport) against ice_wide._xfer (the compact
    all_gather) inside shard_map on 4 fake CPU devices, bit for bit."""
    import jax
    if len(jax.devices()) < 4 or jax.devices()[0].platform != "cpu":
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=4 on the CPU")
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as Ps
    os.environ.pop("FESOM_HALO_GROUPED", None)
    P = 4
    (sd_n, sl_n), (sd_e, sl_e) = _maps(P=P, seed=5)
    Ln, Le = sd_n.shape[1], sd_e.shape[1]
    Lfn, Lfe = 40, 70
    sn, rn = compact_exchange_maps(sd_n, sl_n, field_len=Lfn)
    se, re_ = compact_exchange_maps(sd_e, sl_e, field_len=Lfe)
    arrays = {}
    gm = ice_wide.add_grouped_maps(arrays, "ent", sd_n, sl_n, sd_e, sl_e, Lfn, Ln)
    rng = np.random.default_rng(7)
    xn = rng.standard_normal((P, Lfn, 3)).astype(np.float32)
    xe = rng.standard_normal((P, Lfe, 4)).astype(np.float32)
    keys = ice_wide.grouped_keys(("ent",))
    fold = lambda a: jnp.asarray(np.ascontiguousarray(a).reshape((-1,) + a.shape[2:]))   # noqa: E731
    mesh = Mesh(np.array(jax.devices()[:P]), ("p",))

    def body(n, e, sn_, rn_, se_, re2, *g):
        a = dict(zip(keys, g))
        o1 = ice_wide._xfer(n, e, sn_, rn_, se_, re2, "p")
        o2 = ice_wide._xfer_grouped(n, e, a, "ent", gm, "p")
        return o1 + o2

    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(Ps("p"),) * (6 + len(keys)), out_specs=(Ps("p"),) * 4))
    out = f(fold(xn), fold(xe), fold(sn), fold(rn), fold(se), fold(re_), *[fold(arrays[k]) for k in keys])
    n1, e1, n2, e2 = (np.asarray(o) for o in out)
    np.testing.assert_array_equal(n1, n2)
    np.testing.assert_array_equal(e1, e2)
    assert n1.shape == (P * Ln, 3) and e1.shape == (P * Le, 4)
