"""Checkpoint and resume for a chunked reverse accumulation — so window length stops being
capped by the queue's wall clock.

**What this buys.** A year-long gradient is ~10 h of a single A100 against a 12 h GPU limit,
and two or three years do not fit at all. But the reverse sweep is a loop over chunks whose
entire carried state is one cotangent ``State`` (1.93 GB) plus a scalar accumulator and an
index. Persist those and the sweep can stop anywhere and continue in the next job. Window
length then costs jobs, not a bigger slot.

⚠️ **THE MODEL ALREADY HAS FORWARD RESTARTS. This module is not them, and does not replace
them.** ``fesom_jax.zarr_output.write_restart``/``read_restart`` are a canonical,
partition-INDEPENDENT Zarr restart — a rolling stream for crash recovery and chain hand-off
plus a separate immutable archival stream on a calendar cadence, portable across device counts
(write on 64 devices, resume on 8), read field-by-field so the host peak is one global field
rather than the whole ~40-leaf State. They are wired into ``run_from_config``
(``--restart-in``/``--restart-out``).

What this module adds is the piece those do NOT cover: the REVERSE sweep's carried cotangent
and accumulator, and the many boundary States a reverse pass needs along one trajectory rather
than a single point on it. ``adjoint_horizon.py`` is also a hand-rolled DENSE single-device
loop that never goes through ``run_from_config``, and the existing restart works on the FOLDED
SHARDED representation ``run_steps_sharded`` consumes — so reusing it here is a real conversion,
not a drop-in. Measure before choosing. (Recorded because the author of this file first
concluded "the forward is not resumable" from reading only ``adjoint_horizon.py`` — the same
already-documented-but-unreachable failure this project logged the day before.)

**What is and is not persisted, and why.**

* The chunk-boundary states from the forward pass: written ONCE, one file per boundary. At
  ``--boundary-stride 19`` a year is ~74 GB, which is minutes of I/O against the 1.5 h of
  compute it would take to regenerate them.
* The reverse cotangent, the accumulated gradient, and the next chunk index: written every
  ``ckpt_every`` chunks. This is what makes a resume cheap.
* NOT the forcing stacks or the compiled executables — both are rebuilt on demand and cost
  seconds.

**Nothing here ever deletes anything.** The cotangent alternates between two fixed slots,
``cot_a`` and ``cot_b``, and the manifest names the live one. A write interrupted halfway
therefore lands in the slot the manifest is NOT pointing at, and the previous checkpoint is
still intact and still named. Slots are overwritten in turn rather than accumulated because one
file per reverse chunk would be 706 GB over a year; two slots are 4 GB. The manifest itself is
written last, and by atomic rename, so it never names a partial file. (Workspace house rule: no
unattended deletion, ever. Overwriting a slot the manifest does not point at is the strongest
form of that rule this can take while staying bounded in disk.)

**The identity guard is the part that matters most.** Resuming a checkpoint written by a
different experiment would silently produce a gradient that belongs to neither — finite,
plausible, and wrong, which is this project's recorded worst failure mode. :meth:`Checkpoint.open`
therefore refuses to resume unless every field of the run's identity matches: seed snapshot,
window length, chunk size, boundary stride, the freeze knobs, the truth settings, the weights
file, and the pytree's leaf shapes. A mismatch is a hard exit with the differing key named.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax
import numpy as np

MANIFEST = "manifest.json"
# Distinct exit code for "stopped on purpose, resume me" — so a chain script can tell a
# deliberate stop from a crash. 0 would look like completion; a crash gives 1 or a signal.
EXIT_INCOMPLETE = 42


def _save_tree(path: Path, tree) -> int:
    """Write a pytree of arrays to ``path`` atomically. Returns bytes written.

    Atomically means: to ``path.tmp`` first, then ``os.replace``, which is atomic within a
    directory on every filesystem here. A reader therefore sees either the old file or the new
    one, never a half-written one — which is the whole point when the writer can be killed by
    the scheduler at any instant."""
    leaves = jax.tree.leaves(tree)
    arrs = {f"l{i}": np.asarray(x) for i, x in enumerate(leaves)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrs)          # NOT compressed: this is bandwidth, not storage, bound
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path.stat().st_size


def _load_tree(path: Path, like):
    """Read a pytree of arrays written by :func:`_save_tree`, shaped like ``like``."""
    treedef = jax.tree.structure(like)
    ref = jax.tree.leaves(like)
    with np.load(path) as z:
        leaves = [z[f"l{i}"] for i in range(len(ref))]
    for i, (a, b) in enumerate(zip(leaves, ref)):
        if a.shape != b.shape:
            raise SystemExit(f"FATAL: {path} leaf {i} has shape {a.shape}, expected {b.shape} "
                             f"— this checkpoint was written by a different model build")
    return jax.tree.unflatten(treedef, leaves)


class Checkpoint:
    """Persistent state of one chunked reverse accumulation.

    Lifecycle::

        ck = Checkpoint.open(dir, identity, resume=args.resume, ckpt_every=10,
                             stop_after=args.stop_after)
        if ck.have_boundaries:                     # forward already done in an earlier job
            boundaries, loss = ck.load_boundaries(like=st0), ck.loss
        else:
            s_fin, boundaries = forward(...)
            ck.save_boundaries(boundaries, loss)
        cot, g, c_next = ck.reverse_start(seed_cot=..., n_chunks=n)
        for c in reversed(range(c_next + 1)):
            ...
            if ck.tick(c, cot, g, trace):          # True => time to stop
                raise SystemExit(EXIT_INCOMPLETE)
    """

    def __init__(self, root: Path, identity: dict, *, ckpt_every: int = 10,
                 stop_after: float | None = None, verbose: bool = True):
        self.root = Path(root)
        self.identity = identity
        self.ckpt_every = max(1, int(ckpt_every))
        self.stop_after = stop_after
        self.verbose = verbose
        self.t0 = time.time()
        self.state = dict(phase="init", loss=None, next_chunk=None, grad=0.0,
                          slot="a", trace=[], n_boundaries=0, legs=0, bwd_seconds=0.0)
        self._last_ckpt_chunk = None

    # -- construction ------------------------------------------------------
    @classmethod
    def open(cls, root, identity: dict, *, resume: bool, **kw) -> "Checkpoint":
        ck = cls(root, identity, **kw)
        ck.root.mkdir(parents=True, exist_ok=True)
        man = ck.root / MANIFEST
        if not man.exists():
            if resume:
                raise SystemExit(f"FATAL: --resume given but no {man}")
            ck._write_manifest()
            return ck
        prev = json.loads(man.read_text())
        if not resume:
            raise SystemExit(
                f"FATAL: {man} already exists and --resume was not given. Refusing to "
                f"overwrite an existing checkpoint — pass --resume to continue it, or point "
                f"--ckpt-dir somewhere else. (Nothing in this directory has been touched.)")
        bad = [k for k, v in identity.items() if prev.get("identity", {}).get(k) != v]
        if bad:
            lines = "\n".join(f"    {k}: checkpoint has {prev['identity'].get(k)!r}, "
                              f"this run has {identity[k]!r}" for k in bad)
            raise SystemExit(
                f"FATAL: refusing to resume — this run is not the run that wrote "
                f"{man}:\n{lines}\nResuming across a configuration change would produce a "
                f"gradient belonging to neither run, which is finite, plausible and wrong.")
        ck.state = prev["state"]
        ck.state["legs"] = int(ck.state.get("legs", 0)) + 1
        if ck.verbose:
            done = (0 if ck.state["next_chunk"] is None
                    else identity["n_chunks"] - 1 - ck.state["next_chunk"])
            print(f"[ckpt] resuming leg {ck.state['legs']} from {ck.root}: phase="
                  f"{ck.state['phase']}, {done} of {identity['n_chunks']} reverse chunks "
                  f"already done, dL/dm so far = {ck.state['grad']:+.8e}", flush=True)
        return ck

    # -- manifest ----------------------------------------------------------
    def _write_manifest(self):
        payload = dict(identity=self.identity, state=self.state,
                       written=time.strftime("%Y-%m-%dT%H:%M:%S"))
        tmp = self.root / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, self.root / MANIFEST)

    # -- forward phase -----------------------------------------------------
    @property
    def have_boundaries(self) -> bool:
        return self.state["phase"] in ("forward_done", "reverse")

    @property
    def loss(self):
        return self.state["loss"]

    @property
    def _bnd_dir(self) -> Path:
        d = self.root / "boundaries"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def forward_resume(self, like):
        """``(state, chunk_index)`` to restart the FORWARD from, or ``(None, 0)`` fresh.

        Boundary ``b`` is the state ENTERING chunk ``b``, so the returned index is the next
        chunk to run. A leg killed mid-forward therefore redoes at most ``boundary_stride``
        chunks — ~5 minutes at the year-scale stride of 19, instead of the whole 1.5 h forward.

        ⚠️ This is NOT the model's restart. `fesom_jax.zarr_output.write_restart` is a
        canonical partition-independent Zarr restart wired into `run_from_config`; it persists
        ONE state in the folded sharded representation. What is needed here is many boundary
        states along one trajectory, in the dense single-device representation
        `adjoint_horizon.py` uses. Different problem, deliberately kept out of the library."""
        c = int(self.state.get("fwd_done_upto", -1))
        if c <= 0:
            return None, 0
        st = _load_tree(self._bnd_dir / f"bnd_{c:06d}.npz", like)
        if self.verbose:
            print(f"[ckpt] forward resumes at chunk {c} ({self.state['n_boundaries']} "
                  f"boundaries already on disk)", flush=True)
        return st, c

    def put_boundary(self, c: int, state) -> bool:
        """Persist one chunk boundary as the forward produces it. True ⇒ time to stop.

        Written and manifested one at a time rather than all at the end, because that is the
        entire difference between a forward that can be resumed and one that cannot; the write
        was going to happen anyway."""
        _save_tree(self._bnd_dir / f"bnd_{c:06d}.npz", state)
        n = len(list(self._bnd_dir.glob("bnd_*.npz")))
        self.state.update(phase="forward", fwd_done_upto=int(c), n_boundaries=n)
        self._write_manifest()
        if self.stop_after is not None and (time.time() - self.t0) >= self.stop_after:
            print(f"[ckpt] time budget {self.stop_after/60:.0f} min reached during the FORWARD, "
                  f"at boundary {c}. Checkpoint is complete and consistent; resubmit with "
                  f"--resume. Exiting {EXIT_INCOMPLETE}.", flush=True)
            return True
        return False

    def finish_forward(self, loss: float) -> None:
        """Mark the forward complete — the manifest goes last, by atomic rename, so
        ``have_boundaries`` is only ever true once every boundary file is on disk."""
        n = len(list(self._bnd_dir.glob("bnd_*.npz")))
        gb = sum(p.stat().st_size for p in self._bnd_dir.glob("bnd_*.npz")) / 1e9
        self.state.update(phase="forward_done", loss=float(loss), n_boundaries=n)
        self._write_manifest()
        if self.verbose:
            print(f"[ckpt] forward complete: {n} boundaries, {gb:.1f} GB on disk", flush=True)

    def save_boundaries(self, boundaries: dict, loss: float):
        """Persist a whole finished forward at once — the non-streaming path, kept for callers
        that already hold every boundary. Prefer :meth:`put_boundary` as you go: it is what
        makes the forward resumable."""
        t = time.time()
        for c, st in sorted(boundaries.items()):
            _save_tree(self._bnd_dir / f"bnd_{c:06d}.npz", st)
        self.finish_forward(loss)
        if self.verbose:
            print(f"[ckpt] ({len(boundaries)} boundaries written in {time.time()-t:.1f}s)",
                  flush=True)

    def load_boundaries(self, like) -> dict:
        t = time.time()
        out = {}
        for p in sorted(self._bnd_dir.glob("bnd_*.npz")):
            out[int(p.stem.split("_")[1])] = _load_tree(p, like)
        if len(out) != self.state["n_boundaries"]:
            raise SystemExit(f"FATAL: manifest says {self.state['n_boundaries']} boundaries, "
                             f"found {len(out)} in {self._bnd_dir}")
        if self.verbose:
            print(f"[ckpt] {len(out)} boundaries read in {time.time()-t:.1f}s "
                  f"(no forward pass needed)", flush=True)
        return out

    # -- reverse phase -----------------------------------------------------
    def reverse_start(self, seed_cot, n_chunks: int, like):
        """``(cotangent, accumulated grad, next chunk index, trace)`` to continue from.

        On a fresh run this is just the caller's seed and ``n_chunks-1``. On a resume it is the
        cotangent that was carried into chunk ``next_chunk``, read back from its slot."""
        if self.state["phase"] != "reverse":
            self.state.update(phase="reverse", next_chunk=n_chunks - 1, grad=0.0,
                              trace=list(self.state.get("trace") or []))
            return seed_cot, 0.0, n_chunks - 1, self.state["trace"]
        path = self.root / f"cot_{self.state['slot']}.npz"
        cot = _load_tree(path, like)
        return (cot, float(self.state["grad"]), int(self.state["next_chunk"]),
                list(self.state["trace"]))

    def tick(self, next_chunk: int, cot, grad: float, trace: list) -> bool:
        """Record progress after finishing a reverse chunk. Returns True if it is time to stop.

        ``next_chunk`` is the chunk that will be processed NEXT (i.e. ``c-1`` after doing
        ``c``); -1 means the sweep is complete. Writes at most every ``ckpt_every`` chunks, and
        always when the time budget is up so a deliberate stop never loses a chunk."""
        done = next_chunk < 0
        out_of_time = (self.stop_after is not None
                       and (time.time() - self.t0) >= self.stop_after)
        due = (self._last_ckpt_chunk is None
               or (self._last_ckpt_chunk - next_chunk) >= self.ckpt_every)
        if done:
            self.state.update(phase="done", next_chunk=-1, grad=float(grad), trace=list(trace))
            self._write_manifest()
            return False
        if not (due or out_of_time):
            return False
        slot = "b" if self.state["slot"] == "a" else "a"      # the one NOT currently named
        _save_tree(self.root / f"cot_{slot}.npz", cot)
        self.state.update(phase="reverse", next_chunk=int(next_chunk), grad=float(grad),
                          slot=slot, trace=list(trace))
        self._write_manifest()
        self._last_ckpt_chunk = next_chunk
        if self.verbose:
            el = time.time() - self.t0
            print(f"[ckpt] chunk {next_chunk} next; dL/dm so far {grad:+.8e}; slot {slot}; "
                  f"{el/60:.1f} min into this leg", flush=True)
        if out_of_time:
            print(f"[ckpt] time budget {self.stop_after/60:.0f} min reached with "
                  f"{next_chunk + 1} reverse chunks left. Checkpoint is complete and "
                  f"consistent; resubmit with --resume to continue. "
                  f"Exiting {EXIT_INCOMPLETE}.", flush=True)
            return True
        return False

    @property
    def finished(self) -> bool:
        return self.state["phase"] == "done"
