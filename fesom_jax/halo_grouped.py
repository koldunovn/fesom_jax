"""The grouped halo transport (``gpu.halo_transport: grouped``), ported from fesom3_jax ``shard/grouped.py`` (cf2c810):
the coloured transport's rounds (:class:`~fesom_jax.shard_mesh.ColouredExchange`, the same maps and packed buffer as
:func:`fesom_jax.halo.halo_exchange_coloured`), but every round of an exchange in ONE NCCL group of point-to-point
sends and receives instead of one ``lax.ppermute`` (one XLA collective) per round, and each shard sends only its
pairs' real rows, not the round's padding. On JUPITER (fesom3_jax docs/PERFORMANCE.md, "The grouped transport") it
took NG5 on 128 GH200 from 30.6 to 22.9 ms a step and CORE2 on 8 GH200 from 6.2 (all_to_all) to 5.7.

XLA has no grouped point-to-point collective, so this is an XLA FFI custom call (``_nccl_group.cc``, compiled on first
use with the NCCL and CUDA headers of JAX's pip packages) on a communicator of our own: NCCL rank = the shard's index
on the axis ``"p"``, created once per process for the device mesh (:func:`init`, called by
:func:`fesom_jax.halo.device_mesh`) with the unique id through JAX's distributed key-value store. The call runs on
XLA's stream for the device, between the computations around it, like any other kernel.

The values are bytes moved, so the exchange is the coloured one bit for bit; its transpose is the same exchange with
the directions swapped (custom_vjp). GPU only. ``FESOM_HALO_GROUPED=1`` (read when traced and by ``device_mesh``)
switches the coloured transport's rounds to it; the driver sets it for ``halo_transport: grouped``."""

import ctypes
import hashlib
import os
import platform
import subprocess
from functools import partial
from pathlib import Path

import numpy as np

TARGET = "fesom_grouped_exchange"
_SRC = Path(__file__).with_name("_nccl_group.cc")
_BUILD = Path(__file__).with_name("_build")
_CXXFLAGS = ["-O2", "-std=c++17", "-fPIC", "-shared"]
_lib = None
_mesh_key = None


def enabled() -> bool:
    """``FESOM_HALO_GROUPED=1``: the coloured transport's rounds go out as one NCCL group (read when traced)."""
    return os.environ.get("FESOM_HALO_GROUPED", "0").strip() not in ("", "0")


def bulk() -> bool:
    """``FESOM_BGC_HALO=grouped`` (``gpu.bgc_halo``): only the BGC stack's exchanges (31 tracers a call, bound by their
    bytes) go grouped, every other exchange of the step keeps the run's transport (padded: the physics' small exchanges
    are bound by their latency, where one all_to_all is faster; 16 A100, job 28487014). Read on the host, when the
    sharded driver folds its maps and builds the mesh."""
    return os.environ.get("FESOM_BGC_HALO", "").strip().lower() == "grouped"


def wanted() -> bool:
    """The communicator is needed: every exchange grouped, or the BGC stack's."""
    return enabled() or bulk()


def _packages():
    """(include dirs, library dirs) of JAX's XLA FFI headers and the nvidia-nccl / -cuda-runtime / -cuda-nvcc /
    -cuda-cccl pip packages (cuda_runtime.h's crt/ headers come with nvcc's, cuda_fp16.h's nv/target with cccl's)."""
    import jax
    import nvidia.cuda_cccl
    import nvidia.cuda_nvcc
    import nvidia.cuda_runtime
    import nvidia.nccl
    roots = [Path(m.__path__[0]) for m in (nvidia.nccl, nvidia.cuda_runtime, nvidia.cuda_nvcc, nvidia.cuda_cccl)]
    return [jax.ffi.include_dir()] + [str(p / "include") for p in roots], [str(p / "lib") for p in roots[:2]]


def library_name(machine=None):
    tag = hashlib.sha256(_SRC.read_bytes() + " ".join(_CXXFLAGS).encode()).hexdigest()[:12]
    return f"libfesom_nccl_group_{machine or platform.machine()}_{tag}.so"


def library():
    """Compile (once per source version and machine, into fesom_jax/_build) and load the custom call's library."""
    global _lib
    if _lib is not None:
        return _lib
    so = _BUILD / library_name()
    if not so.exists():
        _BUILD.mkdir(exist_ok=True)
        tmp = so.with_suffix(f".so.{os.getpid()}.tmp")          # a name per process; the rename is atomic
        incs, libs = _packages()
        cxx = os.environ.get("CXX", "g++")
        cmd = [cxx, *_CXXFLAGS, "-o", str(tmp), str(_SRC), *(f"-I{d}" for d in incs), *(f"-L{d}" for d in libs),
               "-l:libnccl.so.2", "-l:libcudart.so.12", *(f"-Wl,-rpath,{d}" for d in libs)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(f"building {so.name} needs a C++ compiler: none found as {cxx!r} (set CXX)") from None
        if r.returncode != 0:
            raise RuntimeError(f"building {so.name} with {cxx} failed:\n{r.stderr}")
        tmp.replace(so)
    lib = ctypes.CDLL(str(so))
    lib.fjx_nccl_unique_id.argtypes, lib.fjx_nccl_unique_id.restype = [ctypes.c_char_p], ctypes.c_int
    ip = ctypes.POINTER(ctypes.c_int)
    lib.fjx_nccl_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ip, ip]
    lib.fjx_nccl_init.restype = ctypes.c_int
    lib.fjx_nccl_error.argtypes, lib.fjx_nccl_error.restype = [ctypes.c_int], ctypes.c_char_p
    _lib = lib
    return lib


def init(devices):
    """The communicator over ``devices`` (the shards in their order on the axis ``"p"``), once per process; every
    process calls it at the same point (``halo.device_mesh``). Registers the custom call."""
    global _mesh_key
    import jax
    devices = list(devices)
    key = tuple(d.id for d in devices)
    if _mesh_key == key:
        return
    if _mesh_key is not None:
        raise RuntimeError("the grouped halo transport: one device mesh per process")
    lib = library()
    mine = [(r, d) for r, d in enumerate(devices) if d.process_index == jax.process_index()]
    uid = ctypes.create_string_buffer(128)
    if jax.process_count() == 1 or jax.process_index() == devices[0].process_index:
        res = lib.fjx_nccl_unique_id(uid)
        if res:
            raise RuntimeError(f"ncclGetUniqueId: {lib.fjx_nccl_error(res).decode()}")
    if jax.process_count() > 1:
        from jax._src import distributed
        client = distributed.global_state.client
        name = "fesom_grouped_nccl_id_" + hashlib.sha256(repr(key).encode()).hexdigest()[:12]
        if jax.process_index() == devices[0].process_index:
            client.key_value_set_bytes(name, uid.raw)
        else:
            uid = ctypes.create_string_buffer(client.blocking_key_value_get_bytes(name, 600_000), 128)
    ords = (ctypes.c_int * len(mine))(*[d.local_hardware_id for _, d in mine])
    ranks = (ctypes.c_int * len(mine))(*[r for r, _ in mine])
    res = lib.fjx_nccl_init(uid.raw, len(devices), len(mine), ords, ranks)
    if res:
        raise RuntimeError(f"the grouped halo transport's ncclCommInitRank: "
                           f"{'CUDA failed' if res == -1 else lib.fjx_nccl_error(res).decode()}")
    jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(lib.FjxGroupedExchange), platform="CUDA")
    _mesh_key = key


def send_rows(send_valid: np.ndarray, slots, offs) -> tuple:
    """``sendn[k][r]``: the real rows shard r sends in round k of a coloured map (``send_valid`` ``[P, total]``); the
    rest of the round's slot is padding, which the grouped exchange does not send."""
    return tuple(tuple(int(n) for n in send_valid[:, o:o + s].sum(axis=1)) for o, s in zip(offs, slots))


def _layout(meta, row_bytes: int) -> tuple:
    """The custom call's static attributes for a coloured map ``meta = (perms, slots, offs, sendn)`` whose rows are
    ``row_bytes`` wide: (item, seg_off, seg_len, seg_round, seg_send, seg_recv, dst, src), one segment a round, every
    table flat in ``[k * P + r]`` order."""
    perms, slots, offs, sendn = meta
    R, P = len(perms), len(sendn[0])
    dst = np.full((R, P), -1, dtype=np.int64)
    src = np.full((R, P), -1, dtype=np.int64)
    for k, rd in enumerate(perms):
        for s, d in rd:
            dst[k, s], src[k, d] = d, s
    sent = np.asarray(sendn, dtype=np.int64)                                     # (R, P)
    recv = np.where(src >= 0, np.take_along_axis(sent, np.maximum(src, 0), axis=1), 0)
    flat = lambda a: tuple(int(v) for v in np.asarray(a).ravel())                # noqa: E731
    return (int(sum(slots)) * row_bytes, flat(np.asarray(offs) * row_bytes), flat(np.asarray(slots) * row_bytes),
            tuple(range(R)), flat(sent * row_bytes), flat(recv * row_bytes), flat(dst), flat(src))


def _reverse(lay: tuple) -> tuple:
    """The transpose's attributes: what a shard received goes back to where it came from, as many bytes."""
    item, seg_off, seg_len, seg_round, seg_send, seg_recv, dst, src = lay
    return item, seg_off, seg_len, seg_round, seg_recv, seg_send, src, dst


def _call(x, lay):
    import jax
    item, seg_off, seg_len, seg_round, seg_send, seg_recv, dst, src = lay
    y = jax.ffi.ffi_call(TARGET, jax.ShapeDtypeStruct(x.shape, x.dtype), vmap_method="broadcast_all")(
        x, item=np.int64(item), seg_off=np.asarray(seg_off, np.int64), seg_len=np.asarray(seg_len, np.int64),
        seg_round=np.asarray(seg_round, np.int32), seg_send=np.asarray(seg_send, np.int64),
        seg_recv=np.asarray(seg_recv, np.int64), dst=np.asarray(dst, np.int32), src=np.asarray(src, np.int32))
    return _vary_like(y, x)


def _vary_like(y, x):
    """``y`` varying over ``x``'s manual axes: an FFI call's result varies over none, and under ``shard_map`` with
    ``check_vma`` a transpose must return its primal's type (a type cast only, no data movement)."""
    import jax
    from jax import lax
    want = getattr(getattr(jax.typeof(x), "manual_axis_type", None), "varying", frozenset())
    have = getattr(getattr(jax.typeof(y), "manual_axis_type", None), "varying", frozenset())
    missing = tuple(sorted(want - have))
    return lax.pcast(y, missing, to="varying") if missing else y


def _make():
    import jax

    @partial(jax.custom_vjp, nondiff_argnums=(1,))
    def core(x, lay):
        return _call(x, lay)

    def fwd(x, lay):
        return _call(x, lay), None

    def bwd(lay, _, g):
        return (_call(g, _reverse(lay)),)

    core.defvjp(fwd, bwd)
    return core


_core = None


def exchange(buf, meta):
    """The coloured rounds of the packed send buffer ``buf`` ``[total, *rest]`` (round k at ``[offs[k],
    offs[k]+slots[k])``) in one custom call: returns ``[total, *rest]`` with round k's received rows at the front of
    its slot, as ``concatenate([ppermute(buf[o:o+s], perm) ...])`` has them; the pad rows nobody sends are left unset
    (no halo lane reads them). Under a vmap (the BGC tracers) the items move into the rows' columns, ``(n, total, C)
    -> (total, n C)``, so they share each round's message (fesom3_jax 46a29ea: one message a round and neighbour, not
    one an item). The derivative is the core's own (custom_vjp: custom_vmap has no reverse mode)."""
    global _core
    import jax
    import jax.numpy as jnp
    if _core is None:
        _core = _make()

    def flat(x):            # [total, *rest] -> [total, C] and the call with this row width
        x2 = x.reshape(x.shape[0], -1)
        return _core(x2, _layout(meta, x2.shape[1] * x2.dtype.itemsize)).reshape(x.shape)

    @jax.custom_batching.custom_vmap
    def batched(x):
        return flat(x)

    @batched.def_vmap
    def rule(n, in_batched, x):
        x = x if in_batched[0] else jnp.broadcast_to(x, (n,) + x.shape)
        y = batched(jnp.moveaxis(x, 0, 1))                  # (total, n, *rest): one item, n times the columns
        return jnp.moveaxis(y, 1, 0), True

    @jax.custom_vjp
    def f(x):
        return batched(x)

    def f_fwd(x):
        return batched(x), None

    def f_bwd(_, g):        # linear: the derivative of the core at any point (g's shape and, under shard_map, its
        _, vjp = jax.vjp(flat, jnp.zeros_like(g))          # varying axes are the input's)
        return vjp(g)

    f.defvjp(f_fwd, f_bwd)
    return f(buf)
