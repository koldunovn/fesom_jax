"""Minimal self-contained reproducer — jax.lax.ragged_all_to_all: correct FORWARD, WRONG
reverse-mode autodiff transpose.

Uses JAX's OWN docstring example for `ragged_all_to_all` (axis size 2). The forward matches
the documented result, but (a) the transpose on a concrete cotangent disagrees with the
hand-derived gradient and (b) the adjoint identity <f(x),y> == <x, f^T(y)> — which MUST hold
for any linear map — is violated by O(1).

Requires >= 2 GPUs (or TPUs): `ragged_all_to_all` is unimplemented on XLA:CPU.

    python ragged_a2a_transpose_bug.py

Dependencies: jax, numpy only (no project code). Tested on JAX 0.10.1 + A100.

Sharding note: the per-device operand/offsets are passed by FOLDING the device axis into the
leading dim — a global `[P, k]` array reshaped to `[P*k]` and sharded `PartitionSpec('x')`, so
each device's shard is `[k]` (rank-1, as ragged_all_to_all requires) with no stray size-1 axis.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as Pspec

jax.config.update("jax_enable_x64", True)

P, K, OUT = 2, 3, 4          # 2 devices; operand size 3/device; output buffer size 4/device


def _fold(a):                # [P, k, ...] -> [P*k, ...]  (device axis into leading dim)
    a = np.asarray(a)
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])


def _unfold(a):              # [P*k, ...] -> [P, k, ...]
    a = np.asarray(a)
    return a.reshape((P, a.shape[0] // P) + a.shape[1:])


def main():
    devs = jax.devices()
    if devs[0].platform == "cpu":
        print("SKIP: lax.ragged_all_to_all is unimplemented on XLA:CPU; needs >=2 GPUs/TPUs.")
        return
    if len(devs) < P:
        print(f"SKIP: need >= {P} devices, have {len(devs)}.")
        return

    mesh = Mesh(np.array(devs[:P]), ("x",))
    spec = Pspec("x")

    # --- JAX `ragged_all_to_all` docstring example (axis size 2) ---
    # device 0: operand [1,2,2];  device 1: operand [3,4,0]   (folded to [P*K] = [6])
    operand = jnp.asarray(_fold([[1., 2., 2.], [3., 4., 0.]]))
    io = jnp.asarray(_fold([[0, 1], [0, 1]]).astype(np.int32))   # input_offsets
    ss = jnp.asarray(_fold([[1, 2], [1, 1]]).astype(np.int32))   # send_sizes
    oo = jnp.asarray(_fold([[0, 0], [1, 2]]).astype(np.int32))   # output_offsets
    rs = jnp.asarray(_fold([[1, 1], [2, 1]]).astype(np.int32))   # recv_sizes
    EXPECTED_FWD = np.array([[1, 3, 0, 0], [2, 2, 4, 0]], float)  # documented result

    def body(op, a, b, c, d):    # per device: op [3], offsets [2]  (all rank-1)
        return lax.ragged_all_to_all(op, jnp.zeros(OUT, op.dtype), a, b, c, d, axis_name="x")
    ragged = jax.shard_map(body, mesh=mesh, in_specs=(spec,) * 5, out_specs=spec)

    def f(op):                   # linear in `op`; the offsets/sizes are constants
        return ragged(op, io, ss, oo, rs)

    # (1) FORWARD == documented result  → the setup is valid (not a misuse).
    fwd = _unfold(f(operand))
    fwd_ok = np.array_equal(fwd, EXPECTED_FWD)
    print(f"(1) forward             : {fwd.tolist()}")
    print(f"    expected (docstring): {EXPECTED_FWD.tolist()}   -> {'OK' if fwd_ok else 'MISMATCH'}")

    # (2) TRANSPOSE on y = ones: f^T(ones)[i] = #times operand[i] is sent (each sent elem -> 1).
    #     Correct: device0 sends all 3 -> [1,1,1]; device1 sends operand[0],[1], not [2] -> [1,1,0].
    EXPECTED_GT = np.array([[1, 1, 1], [1, 1, 0]], float)
    _, vjp_ones = jax.vjp(f, operand)
    gT = _unfold(vjp_ones(jnp.ones(P * OUT))[0])
    gT_ok = np.array_equal(gT, EXPECTED_GT)
    print(f"(2) transpose f^T(ones) : {gT.tolist()}")
    print(f"    expected (by hand)  : {EXPECTED_GT.tolist()}   -> {'OK' if gT_ok else 'WRONG'}")

    # (3) ADJOINT IDENTITY <f(x),y> == <x, f^T(y)> (must hold for any linear f), random x,y.
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(P * K))
    y = jnp.asarray(rng.standard_normal(P * OUT))
    fx, vjp_y = jax.vjp(f, x)
    xbar = vjp_y(y)[0]
    lhs, rhs = float(jnp.sum(fx * y)), float(jnp.sum(x * xbar))
    gap = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    print(f"(3) adjoint identity    : <f(x),y>={lhs:+.6e}  <x,f^T(y)>={rhs:+.6e}  rel|Δ|={gap:.3e}"
          f"  -> {'OK' if gap < 1e-9 else 'VIOLATED'}")

    bug = fwd_ok and (not gT_ok or gap > 1e-9)
    print(f"\n==> {'BUG: forward correct but transpose/gradient WRONG' if bug else 'no bug observed'}")
    print(f"    jax {jax.__version__}; {len(devs)} x {devs[0].device_kind}; x64={jax.config.jax_enable_x64}")


if __name__ == "__main__":
    main()
