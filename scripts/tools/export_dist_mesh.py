#!/usr/bin/env python
"""Export a per-device sharded CORE2 mesh bundle (Phase 8, Task S.2).

Reads the dense CORE2 mesh (``data/mesh_core2/``) + the bit-identical FESOM
``dist_<NP>`` partition, builds the :class:`~fesom_jax.shard_mesh.ShardedMesh`,
and writes it as a flat ``.npy`` bundle (loadable with
:func:`~fesom_jax.shard_mesh.load_sharded_mesh`). Large arrays ⇒ default output
on ``/work``.

Usage:
    python scripts/tools/export_dist_mesh.py <npes> [out_dir]
    python scripts/tools/export_dist_mesh.py 4
    python scripts/tools/export_dist_mesh.py 2 /work/ab0995/a270088/mesh_core2_dist2
"""

from __future__ import annotations

import sys
from pathlib import Path

import fesom_jax  # noqa: F401 — enable x64
from fesom_jax import partit, shard_mesh
from fesom_jax.mesh import load_mesh

REPO = Path(__file__).resolve().parents[2]
CORE2_MESH = REPO / "data" / "mesh_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
DEFAULT_OUT_ROOT = Path("/work/ab0995/a270088")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    npes = int(argv[1])
    out_dir = Path(argv[2]) if len(argv) > 2 else DEFAULT_OUT_ROOT / f"mesh_core2_dist{npes}"

    print(f"[export_dist_mesh] loading dense mesh {CORE2_MESH}")
    mesh = load_mesh(CORE2_MESH)
    if npes == 1:
        part = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    else:
        print(f"[export_dist_mesh] reading partition dist_{npes} under {CORE2_DIST}")
        part = partit.read_partition(CORE2_DIST, npes)

    print(f"[export_dist_mesh] building sharded mesh (P={part.npes})")
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    print(f"[export_dist_mesh]   Lmax = {sm.Lmax}")
    print(f"[export_dist_mesh]   per-device myDim_nod2D = {sm.counts['myDim_nod'].tolist()}")

    shard_mesh.export_sharded_mesh(sm, out_dir)
    nfiles = len(list(out_dir.glob("*.npy")))
    print(f"[export_dist_mesh] wrote {nfiles} arrays + meta.txt → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
