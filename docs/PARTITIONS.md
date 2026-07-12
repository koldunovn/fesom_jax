# Running on more than one device (partitions)

A single GPU (or CPU) needs none of this — skip it. This is for splitting one ocean across several
devices.

## The idea, in one paragraph

To run on `N` devices, the mesh is cut into `N` sub-domains, one per device. Each device owns its own
nodes and elements, and every timestep it exchanges a thin strip of values — the **halo** — with its
neighbours, because computing a gradient at the edge of your sub-domain needs values from the
neighbouring one. FESOM2 already does exactly this, and writes the ownership and exchange lists into a
directory called `dist_<N>/`. **fesom-jax reads those same files**, so a decomposition made for FESOM
works here unchanged, and vice versa.

The consequence you have to remember: **`dist_8` means eight devices. Not seven, not nine.** The
partition has to match your device count exactly.

## Try it with no GPU and no download

The `pi` mesh's decompositions (`dist_2`, `dist_4`, `dist_8`, `dist_16`) **ship inside the package**,
and JAX will happily pretend one CPU is several devices. So you can run and understand a sharded model
before you ever touch a cluster:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 python -c "
from fesom_jax.mesh import load_mesh, DEFAULT_PI_MESH_DIR
from fesom_jax import partit
part = partit.read_partition(DEFAULT_PI_MESH_DIR, 4)   # the packaged dist_4
print([int(n) for n in part.myDim_nod2D], 'nodes per device')"
```

Worked through, with the sharded-vs-single-device comparison, in
[`examples/03_how_the_model_works.ipynb`](../examples/03_how_the_model_works.ipynb) §8.

## Getting them for CORE2

The decompositions ship with the CORE2 data package — `core2_partitions.zip`, about 180 MB, covering
`dist_2` through `dist_864`:

```bash
python scripts/fetch_data.py --dest ~/fesom-data
eval "$(python scripts/fetch_data.py --dest ~/fesom-data --print-env)"   # sets $FESOM_DIST_DIR
```

For **your own mesh**, they come out of FESOM's own partitioner when you set the mesh up — see
[`NEW_MESH.md`](NEW_MESH.md). fesom-jax does not generate them.

## Two directories, and the mistake everyone makes

```
MESH_JAX     the dense .npy bundle the model loads          -> --mesh-dir
DIST_DIR     the directory CONTAINING dist_2/ dist_4/ ...   -> --dist-dir
```

They are **not** the same directory, and `--dist-dir` points at the *parent* of the `dist_<N>`
folders, not at a `dist_<N>` folder itself. `--partition dist_8` then reads `DIST_DIR/dist_8/`.

## Running

```bash
# ONE device: no partition, no dist dir. This is the default.
python scripts/run_from_config.py configs/core2_full.yaml \
    --mesh-dir $FESOM_MESH_DIR --ic-dir $FESOM_IC_DIR \
    --partition serial --steps 480 --restart-out runs/core2/seg0

# FOUR devices (e.g. one GPU node):
python scripts/run_from_config.py configs/core2_full.yaml \
    --mesh-dir $FESOM_MESH_DIR --ic-dir $FESOM_IC_DIR \
    --dist-dir $FESOM_DIST_DIR --partition dist_4 \
    --steps 480 --restart-out runs/core2/seg0
```

`serial` (or `dist_1`) synthesises a trivial one-device partition and reads no files at all.

**Multi-node** adds `jax.distributed`: one process per node, four GPUs each. Set `JDIST=1` and launch
under `srun`; the runner places each process's shard directly (no global array is ever built on one
GPU). The prebuilt sbatch scripts in `scripts/bench/` do all of this — read one before writing your
own.

Or from Python, if you want to drive it yourself:

```python
from fesom_jax import partit, shard_mesh, ssh
from fesom_jax import integrate_sharded as ish

part     = partit.read_partition(DIST_DIR, 8)       # reads DIST_DIR/dist_8/
sm       = shard_mesh.build_sharded_mesh(mesh, part)
state_p  = shard_mesh.partition_state(state0, part)
sop      = ssh.partition_ssh_operator(op, part)

final = ish.run_steps_sharded(sm, state_p, sop, stress_p, n_steps=25, dt=1800.0,
                              npes=8, use_ragged=True)
```

## Three things that will bite you

**1. Gradients do not shard.** The forward model is correct on any number of devices. The *backward*
pass is not: a bug in JAX's `ragged_all_to_all` transpose makes the sharded adjoint wrong (it
over-counts roughly by the device count). The forward is byte-exact; only the gradient is affected.

> **So: run gradients on ONE device.** For a sharded run that must differentiate, use
> `use_ragged=False` — the `all_gather` halo has a correct adjoint, at the cost of much heavier
> communication. Scale gradient work by launching many single-device jobs, not by sharding one.
> Full record, minimal reproducer and the planned fix: [`JAX_RAGGED_A2A_BUG.md`](JAX_RAGGED_A2A_BUG.md).

**2. The fast halo is GPU-only.** `ragged_all_to_all` is unimplemented on XLA's CPU backend, so
multi-device *CPU* runs must use `use_ragged=False`. (You can still exercise sharding on a CPU with
XLA's fake devices: `XLA_FLAGS=--xla_force_host_platform_device_count=4`. That is how the sharding
tests run.)

**3. Big meshes need a minimum device count — memory, not speed.** The compiled timestep's working set
is heavier per node than FESOM's hand-managed memory, so the largest meshes simply do not fit on too
few devices: **dars needs ≥ 2 nodes, NG5 ≥ 8 nodes** (A100-80GB). If you OOM, spread wider rather than
shrinking the timestep.

## What partitioning does *not* change

**Restarts are portable across device counts.** Write a restart on 64 devices, resume it on 8, or on
one. The on-disk restart is keyed by global index, not by rank, so nothing about the decomposition
leaks into it. That is deliberate: you can change your allocation between segments of a long run.

**Output is written per-shard, in parallel.** Each device writes its own piece of the Zarr store —
nothing is gathered onto rank 0 — and the store reads back identically no matter how many devices
produced it.

**The answer is not bit-identical across device counts**, and cannot be. Summing the same numbers in a
different order gives a slightly different floating-point result (~1e-12), and the ocean is chaotic, so
those differences grow. Runs on 4 and 8 devices are *climate-close*, not identical. The C and Kokkos
ports behave the same way. If you need to compare two runs exactly, run them on the same device count.

## Choosing N

More devices is not automatically faster. Each device gets a smaller sub-domain, so the halo exchange
becomes a larger fraction of the work, and past a point you are paying communication to do less
compute. CORE2 (127 k nodes) stops scaling at about **4 GPUs** — it is simply too small to spread
further. Bigger meshes scale further because each device still has real work to do.

The rule of thumb: pick the smallest `N` that fits in memory. Then check it actually got faster.
