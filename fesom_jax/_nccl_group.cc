// The grouped halo transport (fesom_jax/halo_grouped.py; ported unchanged from fesom3_jax shard/_nccl_group.cc at
// cf2c810, where it was written and measured): every coloured round of one exchange in ONE NCCL group of
// point-to-point sends and receives, on a communicator of our own over the shards (NCCL rank = shard index on the
// axis "p"), issued on XLA's stream for the device through an XLA FFI custom call. XLA's ppermute makes each round a
// collective of its own; here a shard's sends and receives to all its neighbours go out together.
//
// Buffers: one batch item of `item` bytes holds the rows of every round (the coloured transport's packed buffer,
// round after round, one segment a round); segment j (seg_off / seg_len bytes, the same on every shard: a round is as
// wide as its widest pair) belongs to round seg_round[j]. In round k shard r sends the first seg_send[j * P + r]
// bytes of its segment j to dst[k * P + r] and receives seg_recv[j * P + r] bytes into the front of it from
// src[k * P + r] (-1: none): its pair's real rows, not the round's padding. The bytes received from no one are left
// as they are (pad rows, which no halo lane reads). Two shards exchange in one round only, so their sends and receives
// match in segment order and length. A batched call (vmap, broadcast_all) holds several items back to back, each
// exchanged alike in the same group.

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

struct Comms {
  int nranks = 0;
  std::vector<int> device;        // CUDA ordinal of each local shard
  std::vector<int> rank;          // its NCCL rank (shard index)
  std::vector<ncclComm_t> comm;
};

Comms g_comms;

// An exchange in flight on a stream of its own (FjxGroupedStart / FjxGroupedDone): per local shard the stream and,
// per call site (`tag`), the event the stream waits for (the send buffer is ready) and the event of its end.
struct Async {
  cudaStream_t comm = nullptr;
  std::vector<cudaEvent_t> ready, done;
};

std::vector<Async> g_async;

int Slot(int32_t ordinal) {
  for (size_t i = 0; i < g_comms.device.size(); ++i)
    if (g_comms.device[i] == ordinal) return static_cast<int>(i);
  return -1;
}

ffi::Error Enqueue(cudaStream_t stream, int slot, ffi::AnyBuffer x, void* recv, int64_t item,
                   ffi::Span<const int64_t> seg_off, ffi::Span<const int64_t> seg_len,
                   ffi::Span<const int32_t> seg_round, ffi::Span<const int64_t> seg_send,
                   ffi::Span<const int64_t> seg_recv, ffi::Span<const int32_t> dst, ffi::Span<const int32_t> src);

ffi::Error Exchange(cudaStream_t stream, int32_t ordinal, ffi::AnyBuffer x, ffi::Result<ffi::AnyBuffer> y,
                    int64_t item, ffi::Span<const int64_t> seg_off, ffi::Span<const int64_t> seg_len,
                    ffi::Span<const int32_t> seg_round, ffi::Span<const int64_t> seg_send,
                    ffi::Span<const int64_t> seg_recv, ffi::Span<const int32_t> dst, ffi::Span<const int32_t> src) {
  const int slot = Slot(ordinal);
  if (slot < 0)
    return ffi::Error::Internal("fjx grouped exchange: no communicator for device " + std::to_string(ordinal));
  return Enqueue(stream, slot, x, y->untyped_data(), item, seg_off, seg_len, seg_round, seg_send, seg_recv, dst, src);
}

ffi::Error Enqueue(cudaStream_t stream, int slot, ffi::AnyBuffer x, void* recv, int64_t item,
                   ffi::Span<const int64_t> seg_off, ffi::Span<const int64_t> seg_len,
                   ffi::Span<const int32_t> seg_round, ffi::Span<const int64_t> seg_send,
                   ffi::Span<const int64_t> seg_recv, ffi::Span<const int32_t> dst, ffi::Span<const int32_t> src) {
  const int P = g_comms.nranks, r = g_comms.rank[slot];
  const size_t R = dst.size() / P, nseg = seg_off.size();
  if (dst.size() != R * P || src.size() != R * P || seg_len.size() != nseg || seg_round.size() != nseg ||
      seg_send.size() != nseg * P || seg_recv.size() != nseg * P)
    return ffi::Error::InvalidArgument("fjx grouped exchange: segments / dst / src do not match the communicator");
  for (size_t j = 0; j < nseg; ++j) {
    if (seg_round[j] < 0 || static_cast<size_t>(seg_round[j]) >= R || seg_off[j] < 0 || seg_off[j] + seg_len[j] > item)
      return ffi::Error::InvalidArgument("fjx grouped exchange: a segment outside the item or the rounds");
    const int64_t ns = seg_send[j * P + r], nr = seg_recv[j * P + r];
    if (ns < 0 || nr < 0 || ns > seg_len[j] || nr > seg_len[j])
      return ffi::Error::InvalidArgument("fjx grouped exchange: a shard's length outside its segment");
  }
  const size_t total = x.size_bytes();
  if (item <= 0 || total % item != 0)
    return ffi::Error::InvalidArgument("fjx grouped exchange: the buffer is no whole number of items");
  const size_t batch = total / item;
  const char* sb = static_cast<const char*>(x.untyped_data());
  char* rb = static_cast<char*>(recv);
  ncclComm_t comm = g_comms.comm[slot];
  ncclGroupStart();
  for (size_t b = 0; b < batch; ++b)
    for (size_t j = 0; j < nseg; ++j) {
      const size_t o = b * item + seg_off[j], ns = seg_send[j * P + r], nr = seg_recv[j * P + r];
      const int d = dst[seg_round[j] * P + r], s = src[seg_round[j] * P + r];
      if (d >= 0 && ns > 0) ncclSend(sb + o, ns, ncclUint8, d, comm, stream);
      if (s >= 0 && nr > 0) ncclRecv(rb + o, nr, ncclUint8, s, comm, stream);
    }
  ncclResult_t res = ncclGroupEnd();
  if (res != ncclSuccess)
    return ffi::Error::Internal(std::string("fjx grouped exchange: ") + ncclGetErrorString(res));
  return ffi::Error::Success();
}

// The exchange started on the shard's own stream after the XLA stream's work so far (the send buffer), returned at
// once: the XLA stream runs on while NCCL moves the bytes into y. FjxGroupedDone makes the XLA stream wait for it; it
// takes y and x (so XLA keeps both buffers until then) and returns y in place.
ffi::Error Start(cudaStream_t stream, int32_t ordinal, ffi::AnyBuffer x, ffi::Result<ffi::AnyBuffer> y,
                 ffi::Result<ffi::AnyBuffer> token, int64_t item, ffi::Span<const int64_t> seg_off,
                 ffi::Span<const int64_t> seg_len, ffi::Span<const int32_t> seg_round,
                 ffi::Span<const int64_t> seg_send, ffi::Span<const int64_t> seg_recv, ffi::Span<const int32_t> dst,
                 ffi::Span<const int32_t> src, int32_t tag) {
  const int slot = Slot(ordinal);
  if (slot < 0 || tag < 0)
    return ffi::Error::Internal("fjx grouped start: no communicator for device " + std::to_string(ordinal));
  Async& a = g_async[slot];
  if (a.comm == nullptr || a.ready.size() <= static_cast<size_t>(tag)) {
    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess || cudaSetDevice(ordinal) != cudaSuccess)
      return ffi::Error::Internal("fjx grouped start: cannot select the device");
    if (a.comm == nullptr && cudaStreamCreateWithFlags(&a.comm, cudaStreamNonBlocking) != cudaSuccess)
      return ffi::Error::Internal("fjx grouped start: cannot create the stream");
    while (a.ready.size() <= static_cast<size_t>(tag)) {
      cudaEvent_t r, d;
      if (cudaEventCreateWithFlags(&r, cudaEventDisableTiming) != cudaSuccess ||
          cudaEventCreateWithFlags(&d, cudaEventDisableTiming) != cudaSuccess)
        return ffi::Error::Internal("fjx grouped start: cannot create the events");
      a.ready.push_back(r);
      a.done.push_back(d);
    }
    cudaSetDevice(prev);
  }
  if (cudaEventRecord(a.ready[tag], stream) != cudaSuccess || cudaStreamWaitEvent(a.comm, a.ready[tag], 0) != cudaSuccess)
    return ffi::Error::Internal("fjx grouped start: cannot order the stream after XLA's");
  ffi::Error e = Enqueue(a.comm, slot, x, y->untyped_data(), item, seg_off, seg_len, seg_round, seg_send, seg_recv,
                         dst, src);
  if (e.failure()) return e;
  if (cudaEventRecord(a.done[tag], a.comm) != cudaSuccess)
    return ffi::Error::Internal("fjx grouped start: cannot record the end");
  return ffi::Error::Success();
}

ffi::Error Done(cudaStream_t stream, int32_t ordinal, ffi::AnyBuffer y, ffi::AnyBuffer x, ffi::AnyBuffer token,
                ffi::Result<ffi::AnyBuffer> out, int32_t tag) {
  const int slot = Slot(ordinal);
  if (slot < 0 || tag < 0 || g_async[slot].done.size() <= static_cast<size_t>(tag))
    return ffi::Error::Internal("fjx grouped done: no exchange started under this tag");
  if (cudaStreamWaitEvent(stream, g_async[slot].done[tag], 0) != cudaSuccess)
    return ffi::Error::Internal("fjx grouped done: cannot wait for the exchange");
  if (out->untyped_data() != y.untyped_data() &&
      cudaMemcpyAsync(out->untyped_data(), y.untyped_data(), y.size_bytes(), cudaMemcpyDeviceToDevice, stream) !=
          cudaSuccess)
    return ffi::Error::Internal("fjx grouped done: cannot copy the received buffer");
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(FjxGroupedStart, Start,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int64_t>("item")
                                  .Attr<ffi::Span<const int64_t>>("seg_off")
                                  .Attr<ffi::Span<const int64_t>>("seg_len")
                                  .Attr<ffi::Span<const int32_t>>("seg_round")
                                  .Attr<ffi::Span<const int64_t>>("seg_send")
                                  .Attr<ffi::Span<const int64_t>>("seg_recv")
                                  .Attr<ffi::Span<const int32_t>>("dst")
                                  .Attr<ffi::Span<const int32_t>>("src")
                                  .Attr<int32_t>("tag"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(FjxGroupedDone, Done,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int32_t>("tag"));

#define FJX_GROUPED_BINDING                              \
  ffi::Ffi::Bind()                                       \
      .Ctx<ffi::PlatformStream<cudaStream_t>>()          \
      .Ctx<ffi::DeviceOrdinal>()                         \
      .Arg<ffi::AnyBuffer>()                             \
      .Ret<ffi::AnyBuffer>()                             \
      .Attr<int64_t>("item")                             \
      .Attr<ffi::Span<const int64_t>>("seg_off")         \
      .Attr<ffi::Span<const int64_t>>("seg_len")         \
      .Attr<ffi::Span<const int32_t>>("seg_round")       \
      .Attr<ffi::Span<const int64_t>>("seg_send")        \
      .Attr<ffi::Span<const int64_t>>("seg_recv")        \
      .Attr<ffi::Span<const int32_t>>("dst")             \
      .Attr<ffi::Span<const int32_t>>("src")

XLA_FFI_DEFINE_HANDLER_SYMBOL(FjxGroupedExchange, Exchange, FJX_GROUPED_BINDING);

// The same handler as a command-buffer (CUDA graph) compatible custom call: XLA records its NCCL calls
// into the graph around it (stream capture) instead of ending the graph there (shard/grouped.py GROUPED_CB).
XLA_FFI_DEFINE_HANDLER_SYMBOL(FjxGroupedExchangeGraph, Exchange, FJX_GROUPED_BINDING,
                              {ffi::Traits::kCmdBufferCompatible});

extern "C" {

// NCCL_UNIQUE_ID_BYTES (128) bytes of a new unique id into out; 0 or the ncclResult_t.
int fjx_nccl_unique_id(char* out) {
  ncclUniqueId id;
  ncclResult_t res = ncclGetUniqueId(&id);
  std::memcpy(out, id.internal, NCCL_UNIQUE_ID_BYTES);
  return res;
}

// One communicator of nranks shards; this process holds n of them: CUDA ordinals devices[i], ranks ranks[i]. Every
// process calls it with the same id. 0 or the ncclResult_t (-1: CUDA failed, -2: already initialised).
int fjx_nccl_init(const char* id_bytes, int nranks, int n, const int* devices, const int* ranks) {
  if (!g_comms.comm.empty()) return -2;
  ncclUniqueId id;
  std::memcpy(id.internal, id_bytes, NCCL_UNIQUE_ID_BYTES);
  int prev = 0;
  if (cudaGetDevice(&prev) != cudaSuccess) return -1;
  std::vector<ncclComm_t> comm(n, nullptr);
  ncclGroupStart();
  for (int i = 0; i < n; ++i) {
    if (cudaSetDevice(devices[i]) != cudaSuccess) return -1;
    ncclCommInitRank(&comm[i], nranks, id, ranks[i]);
  }
  ncclResult_t res = ncclGroupEnd();
  cudaSetDevice(prev);
  if (res != ncclSuccess) return res;
  g_comms.nranks = nranks;
  g_comms.device.assign(devices, devices + n);
  g_comms.rank.assign(ranks, ranks + n);
  g_comms.comm = comm;
  g_async.assign(n, Async());
  return 0;
}

const char* fjx_nccl_error(int res) { return ncclGetErrorString(static_cast<ncclResult_t>(res)); }

}  // extern "C"
