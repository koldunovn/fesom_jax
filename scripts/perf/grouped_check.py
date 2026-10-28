"""The grouped halo transport (fesom_jax/halo_grouped.py) against the coloured ppermute rounds it replaces, on GPUs:
the same CORE2 partition's maps, random fields, every kind (nod / elem / edge), in float64 and float32, a field with
trailing axes, the BGC tracers' case (a vmap over the trailing axis) and the transpose (a vjp of the exchange). Every
comparison is bitwise; the maps' wire volume is printed too (padded all_to_all / coloured rounds / grouped real rows).

One process on one node, or one per node under srun with JDIST=1 (as run_from_config):

    srun -N 4 --ntasks-per-node=1 env JDIST=1 python scripts/perf/grouped_check.py 16

Prints GROUPED_CHECK_OK when every comparison is equal."""

import os
import sys

import jax

if os.environ.get("JDIST"):
    jax.distributed.initialize(local_device_ids=list(range(int(os.environ.get("GPUS_PER_NODE", "4")))))
jax.config.update("jax_enable_x64", True)

from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fesom_jax import halo, halo_grouped, partit, shard_mesh  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402

LEAD = jax.process_index() == 0
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")


def log(*a):
    if LEAD:
        print(*a, flush=True)


def main():
    P = int(sys.argv[1]) if len(sys.argv) > 1 else len(jax.devices())
    part = partit.read_partition(CORE2_DIST, P)
    sm = shard_mesh.build_sharded_mesh(load_mesh(ROOT / "data" / "mesh_core2"), part)
    os.environ["FESOM_HALO_GROUPED"] = "1"                 # device_mesh sets up the communicator
    jmesh = halo.device_mesh(devices=jax.devices()[:P])
    spec = PartitionSpec("p")
    shard = NamedSharding(jmesh, spec)

    def put(a):                     # a global numpy [P*L, ...], every process holds all of it
        a = np.asarray(a)
        return jax.make_array_from_callback(a.shape, shard, lambda idx: a[idx])

    rng = np.random.default_rng(7)
    ok = True
    for kind in ("nod", "elem", "edge"):
        c, r = sm.exchange_coloured[kind], sm.exchange_ragged[kind]
        meta = (c.perms, c.slots, c.offs, halo_grouped.send_rows(c.send_valid, c.slots, c.offs))
        Lmax = sm.Lmax[kind]
        fold = lambda a: np.asarray(a).reshape((P * a.shape[1],) + a.shape[2:])       # noqa: E731
        maps = [put(fold(c.send_idx).astype(np.int32)), put(fold(c.send_valid)),
                put(fold(c.colpos).astype(np.int32)), put(fold(r.halo_mask))]
        sent = np.asarray(meta[3]).sum()
        log(f"{kind}: P {P}, {len(c.perms)} rounds, rows a shard sends (grouped: the mean): padded all_to_all "
            f"{np.asarray(r.pad_src).shape[1]}, coloured {c.total}, grouped {sent / P:.0f}")

        def exch(grouped, f, si, sv, cp, hm):
            os.environ["FESOM_HALO_GROUPED"] = "1" if grouped else "0"
            return halo.halo_exchange_coloured(f, {"send_idx": si, "send_valid": sv, "colpos": cp,
                                                    "halo_mask": hm}, meta, "p")

        def cases(grouped):
            def body(f3, f5, w5, si, sv, cp, hm):
                e = lambda f: exch(grouped, f, si, sv, cp, hm)                     # noqa: E731
                plain = e(f3)
                single = e(f3.astype(jnp.float32))
                vm = jax.vmap(e, in_axes=-1, out_axes=-1)(f5)                       # (L, nl, ntr) over ntr
                _, vjp = jax.vjp(lambda f: jax.vmap(e, in_axes=-1, out_axes=-1)(f), f5)
                return plain, single, vm, vjp(w5)[0]
            return jax.jit(jax.shard_map(body, mesh=jmesh, in_specs=(spec,) * 7, out_specs=(spec,) * 4))

        f3 = put(rng.standard_normal((P * Lmax, sm.nl, 2)))
        f5 = put(rng.standard_normal((P * Lmax, sm.nl, 5)))
        w5 = put(rng.standard_normal((P * Lmax, sm.nl, 5)) * fold(np.asarray(sm.valid_mask[kind]))[:, None, None])
        ref_exe = cases(False).lower(f3, f5, w5, *maps).compile()
        ref, ref2 = ref_exe(f3, f5, w5, *maps), ref_exe(f3, f5, w5, *maps)
        got = cases(True).lower(f3, f5, w5, *maps).compile()(f3, f5, w5, *maps)
        hlo = cases(True).lower(f3, f5, w5, *maps).as_text()

        def diff(a, b):
            return jax.jit(jax.shard_map(lambda x, y: lax.pmax(jnp.max(jnp.abs(x.astype(jnp.float64) - y)), "p"),
                                         mesh=jmesh, in_specs=(spec, spec), out_specs=PartitionSpec()))(a, b)

        # the forward moves values: bitwise. The vjp ends in the send gather's transpose, a scatter-add with
        # duplicate indices (a lane sent to several neighbours), atomic on a GPU: its order (and last bit) can change
        # from run to run, so it is held to the coloured program's own spread over two runs, else 1e-14
        for name, a, b, b2 in zip(("float64 (L, nl, 2)", "float32", "vmap over 5 tracers", "vjp under the vmap"),
                                  got, ref, ref2):
            d, d0 = float(diff(a, b)), float(diff(b2, b))
            ok &= d == 0.0 if not name.startswith("vjp") else d <= max(d0, 1e-14)
            log(f"  {name}: max|grouped - coloured| = {d:.3e} (coloured run twice: {d0:.3e})")
        n_ffi = hlo.count(halo_grouped.TARGET)
        n_cp = hlo.count("collective_permute") + hlo.count("collective-permute")
        log(f"  the grouped program: {n_ffi} custom calls, {n_cp} collective permutes")
    log("GROUPED_CHECK_OK" if ok else "GROUPED_CHECK_FAILED")


if __name__ == "__main__":
    main()
