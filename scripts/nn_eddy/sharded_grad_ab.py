"""Does sharding the adjoint across GPUs actually pay, and is the gradient still right?

Everything in this project's adjoint work so far has run on ONE GPU. That is a property of
``adjoint_horizon.py`` / ``seam_gate.py``, not of the library: ``integrate_sharded`` has a
differentiable sharded runner (``run_steps_sharded(..., return_grad_fn=True)``) and
``tests/test_gradient_sharded.py`` already asserts the sharded gradient EQUALS the dense
single-device one at 2 devices. Measured CORE2 FORWARD scaling (JUPITER_SCALING_FINDINGS,
A100, padded): 99.06 -> 63.96 -> 43.97 ms/step at 1/2/4 GPUs, i.e. 2.25x on 4. Nobody has
measured the BACKWARD, which is the half that dominates a long-window run: the reverse pass
costs ~4.7x the forward per step (70.7 s per 48-step chunk against 15 s per simulated day),
and it has a different communication pattern — more, smaller exchanges in the transposed halo.
So the forward speedup does NOT transfer by assumption.

CORRECTNESS FIRST, then timing. A speedup on a wrong gradient is worthless, and this session
has already produced one plausible-but-refuted diagnosis; the ordering is deliberate. Every
device count must return the SAME d(loss)/d(k_gm) to round-off before any millisecond is
quoted, because sharding changes the reduction ORDER (the scatter-adds land in a different
sequence per shard), so exact bit-equality is not expected but round-off agreement is.

**Real JRA55-do forcing, per step.** The first version of this measured the CONSTANT-forcing
runner, because that was the only differentiable sharded path that existed. A speedup measured
without the seasonal cycle does not describe any window we actually run, so
``run_steps_sharded_forced`` was given a ``return_grad_fn`` (this session) and that is what is
timed here. The forward path of that runner is byte-unchanged when the flag is False.

**The dense single-device gradient is the reference, not npes=1.** ``npes=1`` uses the identity
partition, which makes the sharded code reduce to the dense model — so it is a good SPEEDUP
baseline but a poor CORRECTNESS one, since a bug in the sharded machinery could sit in the
halo exchange and be invisible at one device with no halo. The dense reference comes through
``scan_steps``, the same path every recorded adjoint number in this project was produced with.

Usage (one node, up to 4 A100s):  python scripts/nn_eddy/sharded_grad_ab.py --npes 1,2,4 --n 96
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fesom_jax  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

if ROOT not in Path(fesom_jax.__file__).resolve().parents:
    raise SystemExit(f"FATAL: fesom_jax from {fesom_jax.__file__}, not {ROOT}")

from fesom_jax import calibrate, integrate_sharded as ish, partit, shard_mesh, ssh  # noqa: E402
from fesom_jax.adjoint import scan_steps  # noqa: E402  (the dense reference path)
from fesom_jax.step import step as dense_step  # noqa: E402
from fesom_jax import surface_forcing  # noqa: E402
from fesom_jax.ale import AleConfig  # noqa: E402
from fesom_jax.gm import GMConfig  # noqa: E402
from fesom_jax.ice import IceConfig  # noqa: E402
from fesom_jax.mesh import load_mesh  # noqa: E402
from fesom_jax.phc_ic import phc_initial_state  # noqa: E402
from fesom_jax.tke import TkeConfig  # noqa: E402

import pickle  # noqa: E402

MESH_DIR = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2_dist864"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
DT = 1800.0
SNAP = Path("/work/ab0995/a270088/port_jax/foundation_snaps/snap_step037440.pkl")
M_TRUE = 1.25          # the twin's truth kappa multiplier (adjoint_horizon's value)

# Every scalar leaf of `Params`. Differentiating w.r.t. all of them costs ONE reverse pass —
# the same one a single dL/dk_gm costs — and buys eight independent agreements instead of one,
# through eight different parts of the step: background vertical diffusivity and momentum
# viscosity, GM, Redi, and the four classical-TKE constants. The NN leaves (`tke_nn`, `gm_nn`)
# are left at None on purpose: they would add thousands of comparisons but also a second thing
# that can fail, and this test exists to convict the halo transport, nothing else.
TUNABLES = ("k_ver", "a_ver", "k_gm", "redi_kmax",
            "tke_c_k", "tke_c_eps", "tke_cd", "tke_alpha")


def tunable_vec():
    """The baseline tunables as a flat dict of float64 scalars — the differentiation point."""
    p = calibrate.build_params({})
    return {k: jnp.asarray(float(getattr(p, k)), jnp.float64) for k in TUNABLES}


def truth_vec():
    """The twin's truth: the same model with GM and Redi scaled by ``M_TRUE``."""
    v = dict(tunable_vec())
    v["k_gm"] = v["k_gm"] * M_TRUE
    v["redi_kmax"] = v["redi_kmax"] * M_TRUE
    return v


def gather_owned_nodes(arr_p, part, nod2D):
    """Inverse of ``shard_mesh.partition_state`` for a node-leading field: ``[P, Lmax, …]``
    back to global ``[nod2D, …]``, taking each device's INTERIOR lanes only.

    ``partition_state`` gathers by ``myList_nod2D`` and pads to ``Lmax``; a device's list is
    interior lanes first (``myDim_nod2D``) then halo (``eDim_nod2D``), so the interior prefix
    is exactly the set this device owns and the prefixes partition the domain. The assert is
    not decoration: if a future partition reader changed that ordering, a silent overlap would
    make the comparison below pass by averaging away the very halo bug it exists to catch.
    """
    a = np.asarray(arr_p)
    out = np.zeros((nod2D,) + a.shape[2:], dtype=a.dtype)
    seen = np.zeros(nod2D, dtype=bool)
    for d, ml in enumerate(part.myList_nod2D):
        n_own = int(part.myDim_nod2D[d])
        gids = np.asarray(ml)[:n_own]
        if seen[gids].any():
            raise RuntimeError(f"device {d} claims {int(seen[gids].sum())} nodes another "
                               f"device already owns — interior lanes are not a partition")
        out[gids] = a[d, :n_own]
        seen[gids] = True
    if not seen.all():
        raise RuntimeError(f"{int((~seen).sum())} of {nod2D} global nodes owned by nobody")
    return out


def field_metrics(g, ref):
    """Relative disagreement between two gradient FIELDS, in the two ways that fail apart.

    ``l2`` is the bulk verdict; ``linf`` catches a defect confined to a halo ring, which is a
    few thousand of 6.09e6 cells and can hide entirely under an L2 norm. Both are relative to
    the reference's own magnitude, so they are scale-free and comparable across passes.
    """
    d = np.asarray(g, np.float64) - np.asarray(ref, np.float64)
    rn = np.linalg.norm(np.asarray(ref, np.float64))
    mx = np.max(np.abs(np.asarray(ref, np.float64)))
    return (float(np.linalg.norm(d) / max(rn, 1e-300)),
            float(np.max(np.abs(d)) / max(mx, 1e-300)))


def dense_reference(n_steps, mesh, dense, op, sf_seq, fs):
    """Same window, same probe, through the DENSE single-device path — the correctness oracle."""
    st0 = jax.device_put(dense)
    cfg = dict(gm_cfg=GMConfig(), tke_cfg=TkeConfig(), ale_cfg=AleConfig(),
               ice_cfg=IceConfig(whichEVP=1, adjoint_mode="rheology"))
    wet = jnp.asarray(mesh.node_layer_mask, jnp.float64)

    def body_for(p):
        def one_step(s, sf):
            return dense_step(s, mesh, op, None, p, dt=DT, is_first_step=False,
                              step_forcing=sf, forcing_static=fs, remat_blocks=True,
                              **cfg)
        return one_step

    # `segs = max(2, round(√n_steps))` used to be computed here and passed in. `scan_steps`
    # derives exactly that from the length it is given, and this file writes no results row that
    # reported it, so there is nothing left for the variable to do.

    def fwd_params(p):
        return scan_steps(body_for(p), st0, sf_seq, rematerialize=True)

    # ⚠️ TWIN misfit, NOT a bulk sum(T^2). The bulk norm was the first thing I wrote and it is
    # NOT A MEASUREMENT: GM's effect on the domain-integrated T^2 over one day is a minute
    # difference between two enormous numbers, so the gradient is pure nondeterminism. Two runs
    # of the identical configuration returned +3.4930e-03 and +5.8646e-02 — a factor of 17
    # (26803606 / 26803693) — which makes the reference useless as a correctness oracle.
    # The twin form is the one with evidence: truth = the SAME model at m_true, so loss(m_true)
    # is 0 by construction and the signal is the departure from it.
    st_true = jax.block_until_ready(fwd_params(calibrate.build_params(truth_vec())))
    T_hat, S_hat = jnp.asarray(st_true.T), jnp.asarray(st_true.S)
    del st_true

    # ⚠️ ONE SCALAR, dL/dk_gm, IS NOT A USABLE ORACLE — and that is a property of the quantity,
    # not of sharding. Over a ONE-DAY window the kappa=1.0 twin has barely separated from the
    # kappa=1.25 one: the loss came out 7.9e-06 and its kappa-derivative 4e-06, assembled by
    # summing millions of far larger contributions that nearly cancel. The unstructured
    # scatter-adds this model is built from use atomics, so the summation ORDER changes between
    # runs; the absolute jitter is roughly fixed, and against a 4e-06 answer it is a FIFTH of it.
    # Measured, job 26819176: two passes of the identical dense configuration gave +4.249558e-06
    # and +3.410337e-06, 20 % apart, against a 2e-4 bar. Finite differences are WORSE here, not
    # better: they need the loss gap between kappa(1±h) to clear the loss's own ~1e-3 relative
    # noise, and over one day it does not. At 30 days both work — the signal has grown orders of
    # magnitude while the noise floor has not, which is why --n now defaults to 5 days.
    #
    # Two changes make the verdict real, and neither touches the library:
    #  1. differentiate w.r.t. EVERY tunable at once. `params` is already the differentiable
    #     input of the sharded runner, so all eight scalars come out of ONE reverse pass at no
    #     extra cost. They run through different parts of the step — vertical diffusivity,
    #     momentum viscosity, GM, Redi, four TKE constants — with different magnitudes and
    #     different cancellation structure, so eight agreements are far stronger evidence than
    #     one, and a transport bug confined to one pathway cannot hide behind a lucky scalar.
    #  2. keep the whole FINAL STATE and compare it field-by-field as well. That separates a
    #     forward halo bug from a backward transpose bug, which a gradient-only comparison
    #     cannot do: if the states already disagree, the gradient disagreement means nothing.
    # (The gradient w.r.t. the INITIAL state would be richer still — 6.09e6 numbers — but the
    # sharded runner bakes the initial state in as a closure and only exposes `params`, so that
    # needs surgery on the hot path of every future sharded gradient. Not worth it for this.)
    def loss_of(vec):
        stf = fwd_params(calibrate.build_params(dict(vec)))
        dT, dS = stf.T - T_hat, stf.S - S_hat
        return jnp.sum(wet * (dT * dT + dS * dS)) / jnp.sum(wet)

    vec0 = tunable_vec()
    t0 = time.time()
    st_end = jax.block_until_ready(fwd_params(calibrate.build_params(dict(vec0))))
    lv = float(loss_of(vec0))
    t_fwd = time.time() - t0
    t1 = time.time()
    g = jax.block_until_ready(jax.grad(loss_of)(vec0))
    t_bwd = time.time() - t1
    return dict(npes=0, loss=lv, grad=float(g["k_gm"]), fwd_s=t_fwd, bwd_s=t_bwd,
                gvec={k: float(v) for k, v in g.items()},
                T_end=np.asarray(st_end.T), S_end=np.asarray(st_end.S))


def run_one(npes, n_steps, mesh, dense, op, sf_seq, fs, verbose=True,
            transport="padded"):
    """d(loss)/d(k_gm) over ``n_steps`` on ``npes`` devices, plus forward/backward seconds."""
    # npes==1 has no dist_ files: read_partition refuses it and points at the identity
    # partition instead, which makes the sharded path reduce to the dense single-device
    # model exactly (partit.py's "S.2 no-op invariant"). That makes npes=1 the RIGHT
    # baseline here — it is the same code path, not a different one, so the speedup column
    # measures sharding rather than sharded-vs-dense plumbing differences.
    part = (partit.synth_serial(int(mesh.nod2D), int(mesh.elem2D), int(mesh.edge2D))
            if npes == 1 else partit.read_partition(CORE2_DIST, npes))
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(dense, part)
    fs_p = shard_mesh.partition_forcing_static(fs, part)
    sop = ssh.partition_ssh_operator(op, part)
    Le = sm.fields["node_layer_mask"].shape[1]
    stress_p = jnp.zeros((npes, sm.fields.get("elem_layer_mask").shape[1], 2)) \
        if "elem_layer_mask" in sm.fields else jnp.zeros((npes, Le, 2))

    cfg = dict(gm_cfg=GMConfig(), tke_cfg=TkeConfig(), ale_cfg=AleConfig(),
               ice_cfg=IceConfig(whichEVP=1, adjoint_mode="rheology"))
    # use_ragged=False and no padded/coloured => the all_gather halo, the autodiff-CORRECT
    # transport (the ragged all_to_all transpose is wrong, docs/JAX_RAGGED_A2A_BUG.md).
    # per-step forcing sequence, partitioned: [P, n_steps, Lmax]
    seq_p = shard_mesh.partition_step_forcing(sf_seq, part)
    # TWO AD-correct transports, and only two. `ragged` is excluded on purpose and the library
    # refuses it at entry with `return_grad_fn=True`: `lax.ragged_all_to_all`'s transpose is
    # wrong by O(1) (docs/JAX_RAGGED_A2A_BUG.md), so it can never carry a backward and has no
    # place in an adjoint comparison. `all_gather` is excluded for memory: it REPLICATES the
    # global field on every device, carries the whole tape per device, and OOM'd at npes=1
    # during XLA autotuning (26803606).
    #   padded   — slot-padded dense all_to_all; transpose is another all_to_all. Fastest GPU
    #              transport at <=16 devices, but its volume grows ~linearly in P
    #              (1.8x@P4, 5.4x@P16, 15.3x@P32, 41x@P128).
    #   coloured — K bipartite-coloured `lax.ppermute` rounds; transpose is the inverse
    #              ppermute. **The only transport whose volume does NOT grow with P**
    #              (1.0-1.4x the true halo at any P) and the fastest measured at NG5-64.
    # That last property is why this comparison matters beyond CORE2: a 10 km mesh needs tens
    # of devices, where padded's tax is the dominant term and coloured's is not.
    _run = ish.run_steps_sharded_forced(sm, state_p, sop, stress_p, seq_p, fs_p,
                                        n_steps=n_steps, dt=DT, npes=npes,
                                        use_ragged=False,
                                        use_padded=(transport == "padded"),
                                        use_coloured=(transport == "coloured"),
                                        return_grad_fn=True,
                                        bootstrap_ab2=False, **cfg)

    owned = jnp.asarray(sm.owned_mask["nod"])[:, :, None]
    nlm = jnp.asarray(sm.fields["node_layer_mask"])
    w3d = (owned & nlm).astype(jnp.float64)

    # the same twin as the dense reference, generated through THIS path so the two losses are
    # the same mathematical quantity, each in its own space (owned masks partition the domain
    # exactly, so the weighted sums agree).
    st_true = jax.block_until_ready(_run(calibrate.build_params(truth_vec())))
    T_hat, S_hat = jnp.asarray(st_true.T), jnp.asarray(st_true.S)
    del st_true

    def loss_of(vec):
        st = _run(calibrate.build_params(dict(vec)))
        dT, dS = st.T - T_hat, st.S - S_hat
        return jnp.sum(w3d * (dT * dT + dS * dS)) / jnp.sum(w3d)

    vec0 = tunable_vec()
    t0 = time.time()
    st_end = jax.block_until_ready(_run(calibrate.build_params(dict(vec0))))
    lv = float(loss_of(vec0))
    t_fwd = time.time() - t0
    t1 = time.time()
    g = jax.block_until_ready(jax.grad(loss_of)(vec0))
    t_bwd = time.time() - t1
    if verbose:
        print(f"[npes={npes}] loss={lv:.12e}  dL/dk_gm={float(g['k_gm']):+.12e}  "
              f"fwd {t_fwd:.1f}s  fwd+bwd {t_bwd:.1f}s", flush=True)
    # Back to GLOBAL node numbering so the fields are comparable with the dense path. npes==1 is
    # the identity partition and `partition_state` squeezes the P axis away there, so the arrays
    # already ARE global — gathering would index a 2-D array as if it were 3-D.
    if npes == 1:
        T_end, S_end = np.asarray(st_end.T), np.asarray(st_end.S)
    else:
        T_end = gather_owned_nodes(st_end.T, part, int(mesh.nod2D))
        S_end = gather_owned_nodes(st_end.S, part, int(mesh.nod2D))
    return dict(npes=npes, loss=lv, grad=float(g["k_gm"]), fwd_s=t_fwd, bwd_s=t_bwd,
                gvec={k: float(v) for k, v in g.items()},
                T_end=T_end, S_end=S_end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-ref-out", type=Path, default=None,
                    help="compute ONLY the dense reference and write it here, then exit.")
    ap.add_argument("--dense-ref-in", type=Path, default=None,
                    help="reuse a dense reference from --dense-ref-out instead of computing it. "
                         "⚠️ ONE PROCESS PER CONFIGURATION is the point: JAX does not hand back "
                         "device memory when a jitted graph goes out of scope, so a dense "
                         "backward run earlier in the SAME process leaves the card nearly full "
                         "and the next configuration dies in XLA autotuning trying to allocate "
                         "62 MiB (26818559, and the same signature in 26803606). It also makes "
                         "a benchmark whose timings depend on what ran before them.")
    ap.add_argument("--transport", default="padded,coloured",
                    help="comma-separated AD-correct transports to compare. `ragged` is not "
                         "accepted: its autodiff transpose is broken and the library refuses "
                         "it with return_grad_fn=True.")
    ap.add_argument("--npes", default="2,4",
                    help="comma-separated device counts. npes=1 is deliberately NOT the "
                         "default: the identity partition gives no distribution benefit while "
                         "still paying the shard_map machinery, and it OOM'd (26803606). The "
                         "dense single-GPU run is the honest 1-device baseline anyway.")
    ap.add_argument("--n", type=int, default=48,
                    help="steps (48 = 1 day). ⚠️ THE WINDOW IS CAPPED BY MEMORY, NOT BY CHOICE. "
                         "The sharded runner remats each step body inside one lax.scan, so the "
                         "reverse pass stores EVERY step's carry: device memory grows LINEARLY "
                         "in --n, unlike the dense path, which uses sqrt-segment checkpointing "
                         "and is flat. At npes=1 that is 48 x 1.93 GB = 93 GB of carries, which "
                         "is exactly the 124 GB XLA reported it could not get under (26819176); "
                         "npes divides it, so ~1.5 d fits at 2 devices and ~3 d at 4. Lengthening "
                         "the window to lift the gradient above the atomics floor is therefore "
                         "NOT available here until the sharded runner gets chunked reverse "
                         "accumulation. Read the per-tunable spread printed below instead: "
                         "background diffusivity and viscosity act immediately and resolve at a "
                         "day, while GM barely moves the loss in that time.")
    ap.add_argument("--rtol", type=float, default=2e-4,
                    help="gradient agreement bar against the dense reference. NOT round-off: "
                         "1e-9 was wrong and would have failed a correct result. The model is "
                         "nondeterministic on GPU (atomic scatter-adds), and the two-level "
                         "chunking validation measured the size of it directly — three "
                         "identical-in-exact-arithmetic runs gave forward losses spread 6e-4 "
                         "relative and gradients spread 3e-5 (job 26803468). Sharding adds a "
                         "different reduction order on top. So the honest bar is the model's "
                         "own reproducibility, ~2e-4; anything at percent level is a real bug, "
                         "anything at 1e-5..1e-4 is agreement at the precision available.")
    ap.add_argument("--results", type=Path,
                    default=Path("/work/ba1583/a270088/nn_eddy/seam_gate/sharded_ab.jsonl"))
    args = ap.parse_args()
    npes_list = [int(x) for x in args.npes.split(",") if x.strip()]

    print(f"[setup] devices visible: {len(jax.devices())} -> {jax.devices()}", flush=True)
    mesh = load_mesh(MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    with open(SNAP, "rb") as f:
        dense = pickle.load(f)
    sst0 = np.asarray(phc_initial_state(mesh, IC_DIR).T[:, 0])
    cf = surface_forcing.build_surface_forcing(mesh, 1960, sst_ic=sst0)
    dates = surface_forcing.dates_for_steps(1960, DT, args.n, start_month=2, start_day=20)
    sf_seq = cf.stack(dates)                 # [n_steps] REAL per-step JRA55-do, not a constant
    print(f"[setup] N={args.n} steps ({args.n * DT / 86400:.2f} d) with per-step JRA55-do "
          f"forcing from {dates[0]} to {dates[-1]}", flush=True)

    # Dense single-device reference — run TWICE. The previous probe's reference varied 17x
    # between identical runs and I only noticed by comparing two separate jobs by hand. An
    # oracle that is not reproducible cannot validate anything, so it now proves itself in
    # process, before it is used to judge the sharded numbers.
    # The fields do not fit in JSON, so the reference is a JSON of scalars beside an .npz of
    # arrays. Both are written and read as one unit; the .npz path is derived, never configured,
    # so the two halves cannot drift apart.
    def _npz_of(p):
        return Path(str(p) + ".fields.npz")

    if args.dense_ref_in is not None:
        d = json.loads(args.dense_ref_in.read_text())
        if "usable" not in d or not _npz_of(args.dense_ref_in).exists():
            raise SystemExit(
                f"{args.dense_ref_in} is a reference in the OLD single-scalar format (no "
                f"'usable' list, no {_npz_of(args.dense_ref_in).name}). Reusing it would judge "
                f"today's sharded runs against a quantity this script no longer computes. "
                f"Regenerate it with --dense-ref-out under the current name.")
        dref, dref2, usable = d["pass1"], d["pass2"], d["usable"]
        z = np.load(_npz_of(args.dense_ref_in))
        for k, r in (("1", dref), ("2", dref2)):
            r["T_end"], r["S_end"] = z[f"T_end{k}"], z[f"S_end{k}"]
        print(f"[dense ref] reused from {args.dense_ref_in}; usable tunables: "
              f"{usable or '(none)'}", flush=True)
    else:
        dref = dense_reference(args.n, mesh, dense, op, sf_seq, cf.static)
        dref2 = dense_reference(args.n, mesh, dense, op, sf_seq, cf.static)
        # ⚠️ MEASURE THE FLOOR, DO NOT ASSUME IT. Each tunable proves itself separately: two
        # identical dense passes, and whichever leaves reproduce better than the bar are the
        # ones allowed to carry a verdict. A leaf that does not is not evidence of a sharding
        # bug and must never be reported as one — it is a quantity this window cannot resolve.
        usable = [k for k in TUNABLES
                  if abs(dref2["gvec"][k] - dref["gvec"][k])
                  / max(abs(dref["gvec"][k]), 1e-300) <= args.rtol]
        if args.dense_ref_out is not None:
            args.dense_ref_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(_npz_of(args.dense_ref_out),
                     T_end1=dref["T_end"], S_end1=dref["S_end"],
                     T_end2=dref2["T_end"], S_end2=dref2["S_end"])
            args.dense_ref_out.write_text(json.dumps(dict(
                pass1={k: v for k, v in dref.items() if k not in ("T_end", "S_end")},
                pass2={k: v for k, v in dref2.items() if k not in ("T_end", "S_end")},
                usable=usable, n=args.n), indent=1))
            print(f"[dense ref] written to {args.dense_ref_out} (+ fields npz); exiting so the "
                  f"sharded configurations each get a CLEAN device", flush=True)
            return 0

    print(f"[dense ref] loss {dref['loss']:.12e} / {dref2['loss']:.12e}  "
          f"(fwd {dref['fwd_s']:.1f}s, fwd+bwd {dref['bwd_s']:.1f}s)")
    print("[dense ref] per-tunable gradient and its OWN two-pass spread — the measured floor:")
    for k in TUNABLES:
        a, b = dref["gvec"][k], dref2["gvec"][k]
        sp = abs(b - a) / max(abs(a), 1e-300)
        print(f"    {k:12s} {a:+.9e}   self-spread {sp:.3e}  "
              f"{'usable' if k in usable else 'TOO NOISY — cannot judge with it'}")
    fl2, flinf = field_metrics(dref2["T_end"], dref["T_end"])
    sl2, slinf = field_metrics(dref2["S_end"], dref["S_end"])
    print(f"[dense ref] final-state self-spread: T rel-L2 {fl2:.3e} max {flinf:.3e}; "
          f"S rel-L2 {sl2:.3e} max {slinf:.3e}", flush=True)
    field_floor = max(fl2, sl2, flinf, slinf)
    if not usable:
        print(f"\nDENSE_REF_NOT_REPRODUCIBLE — not one of the {len(TUNABLES)} tunables "
              f"reproduces to {args.rtol:.1e} between two identical dense passes. The probe is "
              f"noise-dominated at N={args.n} steps; NO gradient verdict can be read from this "
              f"run. Raise --n. The FORWARD field comparison below is still valid, and the "
              f"timings are still valid.", flush=True)

    transports = [t.strip() for t in args.transport.split(",") if t.strip()]
    if "ragged" in transports:
        raise SystemExit("ragged has no working adjoint (its transpose is wrong by O(1)); the "
                         "library refuses it with return_grad_fn=True. Drop it.")
    recs = []
    for tr in transports:
        for p in npes_list:
            if len(jax.devices()) < p:
                print(f"[{tr} npes={p}] SKIPPED — only {len(jax.devices())} devices visible",
                      flush=True)
                continue
            print(f"\n----- transport={tr}  npes={p} -----", flush=True)
            try:
                r = run_one(p, args.n, mesh, dense, op, sf_seq, cf.static, transport=tr)
            except Exception as e:
                print(f"[{tr} npes={p}] FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
                recs.append(dict(npes=p, transport=tr, error=f"{type(e).__name__}: {str(e)[:300]}"))
                continue
            r["transport"] = tr
            recs.append(r)

    if not recs:
        raise SystemExit("no device count ran")

    # --- CORRECTNESS FIRST -------------------------------------------------------------
    # ⚠️ A configuration that RAISED carries only "error" — it has no "grad", no "gvec" and no
    # fields. Indexing it here is what turned one OOM into a KeyError that killed the whole
    # summary of job 26819176, losing the configurations that HAD run. Failures are reported,
    # then skipped.
    ref = dref                       # the DENSE path is the oracle, not npes=1
    good_recs = [r for r in recs if "error" not in r]
    for r in recs:
        if "error" in r:
            print(f"  {r.get('transport','?'):9s} npes={r['npes']:2d}  DID NOT RUN: {r['error'][:120]}")

    # The FORWARD first: if the sharded final state already differs from the dense one, a
    # gradient disagreement says nothing about the transposed halo — it is just the forward
    # discrepancy carried backwards. This ordering is the whole reason the fields are kept.
    print("\n=== forward: sharded final state vs the DENSE one (floor "
          f"{field_floor:.3e} from two identical dense passes) ===")
    fwd_ok = True
    for r in good_recs:
        t2, tinf = field_metrics(r["T_end"], ref["T_end"])
        s2, sinf = field_metrics(r["S_end"], ref["S_end"])
        worst = max(t2, tinf, s2, sinf)
        good = worst <= max(field_floor, args.rtol)
        fwd_ok &= good
        print(f"  {r['transport']:9s} npes={r['npes']:2d}  T rel-L2 {t2:.3e} max {tinf:.3e}  "
              f"S rel-L2 {s2:.3e} max {sinf:.3e}  {'OK' if good else 'MISMATCH'}")

    print("\n=== backward: every tunable's gradient against the DENSE single-device one ===")
    if not usable:
        print("  skipped — the dense reference resolves none of the tunables at this window.")
    ok = bool(usable) and fwd_ok
    for r in good_recs:
        for k in TUNABLES:
            rel = abs(r["gvec"][k] - ref["gvec"][k]) / max(abs(ref["gvec"][k]), 1e-300)
            if k not in usable:
                print(f"  {r['transport']:9s} npes={r['npes']:2d}  {k:12s} "
                      f"{r['gvec'][k]:+.9e}  rel={rel:.3e}  (not judged — below the floor)")
                continue
            good = rel <= args.rtol
            ok &= good
            print(f"  {r['transport']:9s} npes={r['npes']:2d}  {k:12s} "
                  f"{r['gvec'][k]:+.9e}  rel={rel:.3e}  {'OK' if good else 'MISMATCH'}")
    if not ok:
        print("\nSHARDED_GRAD_MISMATCH — do NOT read the timings below as a speedup of "
              "anything useful; the sharded gradient is not the same quantity.", flush=True)

    # --- then, and only then, timing ---------------------------------------------------
    # Baseline is the DENSE single-GPU run, because that is the practical question: does
    # spending 4 GPUs beat the one-GPU path every adjoint number so far was produced with?
    print("\n=== speedup against the DENSE single-GPU path (what we run today) ===")
    print(f"  dense 1GPU  fwd {dref['fwd_s']:7.1f}s            fwd+bwd {dref['bwd_s']:7.1f}s")
    for r in good_recs:
        print(f"  {r['transport']:9s} npes={r['npes']:2d}  "
              f"fwd {r['fwd_s']:7.1f}s ({dref['fwd_s']/r['fwd_s']:5.2f}x)  "
              f"fwd+bwd {r['bwd_s']:7.1f}s ({dref['bwd_s']/r['bwd_s']:5.2f}x)")
    if len(good_recs) < len(recs):
        print(f"  ({len(recs) - len(good_recs)} configuration(s) did not run — listed above; "
              f"no timing exists for them, and none is implied)")
    print("\nNote: the FORWARD scaling on this mesh is already known (2.25x at 4 GPUs, "
          "JUPITER_SCALING_FINDINGS). The backward column is the new number.", flush=True)

    # The end-state fields are megabytes and belong in the log's verdict, not in a JSONL row.
    def _slim(r):
        return {k: v for k, v in r.items() if k not in ("T_end", "S_end")}

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a") as fh:
        fh.write(json.dumps(dict(n_steps=args.n, grad_match=bool(ok), fwd_match=bool(fwd_ok),
                                 usable_tunables=usable, field_floor=field_floor,
                                 forcing="jra55do_perstep", rtol=args.rtol,
                                 dense_ref=_slim(dref), runs=[_slim(r) for r in recs])) + "\n")
    print("\nSHARDED_GRAD_" + ("OK" if ok else "MISMATCH"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
