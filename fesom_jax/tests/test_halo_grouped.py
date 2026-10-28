"""The grouped halo transport's tables (fesom_jax/halo_grouped.py) on the real CORE2 partitions, host-side: every
coloured pair's rows sent equal the rows its receiver expects, in the round the colouring put it, and the transpose's
tables are the forward's with the directions swapped. The exchange itself is a GPU custom call; its bitwise check
against the coloured rounds is scripts/perf/grouped_check.py (one node or several)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fesom_jax import halo_grouped, partit, shard_mesh
from fesom_jax.mesh import load_mesh

CORE2_MESH = Path(__file__).resolve().parents[2] / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")

avail = pytest.mark.skipif(
    not CORE2_MESH.is_dir() or not (CORE2_DIST / "dist_4").is_dir(),
    reason="CORE2 dense mesh or dist partitions missing",
)


@avail
@pytest.mark.parametrize("npes", [4, 16])
def test_grouped_tables_match_the_coloured_pairs(npes):
    sm = shard_mesh.build_sharded_mesh(load_mesh(CORE2_MESH), partit.read_partition(CORE2_DIST, npes))
    for kind in ("nod", "elem", "edge"):
        c, r = sm.exchange_coloured[kind], sm.exchange_ragged[kind]
        meta = (c.perms, c.slots, c.offs, halo_grouped.send_rows(c.send_valid, c.slots, c.offs))
        rb = 8 * 47
        item, seg_off, seg_len, seg_round, seg_send, seg_recv, dst, src = halo_grouped._layout(meta, rb)
        R = len(c.perms)
        assert item == c.total * rb and seg_round == tuple(range(R))
        assert seg_off == tuple(o * rb for o in c.offs) and seg_len == tuple(s * rb for s in c.slots)
        send, recv = np.reshape(seg_send, (R, npes)), np.reshape(seg_recv, (R, npes))
        dst, src = np.reshape(dst, (R, npes)), np.reshape(src, (R, npes))
        for k, rd in enumerate(c.perms):
            for s, d in rd:
                n = int(r.send_sizes[s, d])                    # the ragged map's chunk s -> d
                assert dst[k, s] == d and src[k, d] == s
                assert send[k, s] == n * rb and recv[k, d] == n * rb, f"{kind}/{npes}: round {k} pair {s}->{d}"
                assert n <= c.slots[k]
        assert not send[dst < 0].any() and not recv[src < 0].any(), f"{kind}/{npes}: a length without a partner"
        # the transpose: every shard sends back what it received, to where it came from
        back = halo_grouped._reverse(halo_grouped._layout(meta, rb))
        assert back[4] == tuple(recv.ravel()) and back[5] == tuple(send.ravel())
        assert back[6] == tuple(src.ravel()) and back[7] == tuple(dst.ravel())
