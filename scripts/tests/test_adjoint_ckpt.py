"""Tests for checkpoint/resume of a chunked reverse sweep (`scripts/nn_eddy/adjoint_ckpt.py`).

This exists so that window length stops being capped by the queue's wall clock. What it can
get wrong is not a crash — it is producing a gradient that is finite, plausible and belongs to
neither run:

* resuming a checkpoint written by a DIFFERENT experiment (different seed, window, stride,
  freeze set) and silently continuing;
* losing the last checkpoint to a write that was killed halfway;
* restarting the reverse sweep from the wrong chunk, so a chunk is counted twice or skipped.

Each has a test below. No mesh, no model — the module is generic over pytrees of arrays, and
these run in under a second.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load():
    path = _ROOT / "scripts" / "nn_eddy" / "adjoint_ckpt.py"
    spec = importlib.util.spec_from_file_location("adjoint_ckpt", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["adjoint_ckpt"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ck_mod():
    return _load()


def _tree(scale=1.0, n=4):
    return {"a": np.arange(n, dtype=np.float64) * scale,
            "b": np.ones((2, n), np.float64) * scale}


IDENT = dict(snap="snap_step037440.pkl", n_days=10, n_chunks=10, chunk_steps=48,
             boundary_stride=1, freeze=["mix"], truth_days=30)


# ---------------------------------------------------------------------------
# the identity guard
# ---------------------------------------------------------------------------

def test_resume_refuses_a_different_experiment(ck_mod, tmp_path):
    """The most dangerous failure this module has: continuing someone else's sweep. The
    accumulated gradient would be part of one run and part of another — finite and wrong."""
    ck = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=False, verbose=False)
    ck.save_boundaries({0: _tree(), 1: _tree(2.0)}, loss=1.5)
    for key, val in (("n_days", 20), ("boundary_stride", 2), ("freeze", ["mix", "fct"]),
                     ("snap", "snap_day150.pkl"), ("truth_days", 90)):
        other = dict(IDENT, **{key: val})
        with pytest.raises(SystemExit) as e:
            ck_mod.Checkpoint.open(tmp_path, other, resume=True, verbose=False)
        assert key in str(e.value), f"the refusal must name the field that differs ({key})"


def test_matching_identity_resumes(ck_mod, tmp_path):
    ck = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=False, verbose=False)
    ck.save_boundaries({0: _tree(), 1: _tree(2.0)}, loss=1.5)
    ck2 = ck_mod.Checkpoint.open(tmp_path, dict(IDENT), resume=True, verbose=False)
    assert ck2.have_boundaries and ck2.loss == 1.5
    assert ck2.state["legs"] == 1


def test_refuses_to_clobber_an_existing_checkpoint(ck_mod, tmp_path):
    """Starting fresh on top of an existing checkpoint would throw away hours of a previous
    leg. It must be an error, not a silent overwrite."""
    ck_mod.Checkpoint.open(tmp_path, IDENT, resume=False, verbose=False)
    with pytest.raises(SystemExit) as e:
        ck_mod.Checkpoint.open(tmp_path, IDENT, resume=False, verbose=False)
    assert "--resume" in str(e.value)


def test_resume_without_a_checkpoint_is_an_error(ck_mod, tmp_path):
    with pytest.raises(SystemExit):
        ck_mod.Checkpoint.open(tmp_path / "nope", IDENT, resume=True, verbose=False)


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------

def test_boundaries_round_trip_exactly(ck_mod, tmp_path):
    """Bit-for-bit: these states re-enter a VJP, so a lossy round trip is a wrong gradient."""
    ck = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=False, verbose=False)
    b = {0: _tree(1.0), 5: _tree(np.pi), 10: _tree(-7.25)}
    ck.save_boundaries(b, loss=0.5)
    back = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=True,
                                  verbose=False).load_boundaries(like=_tree())
    assert sorted(back) == sorted(b)
    for c in b:
        for k in b[c]:
            assert np.array_equal(back[c][k], b[c][k]), (c, k)


def test_load_refuses_a_shape_mismatch(ck_mod, tmp_path):
    """A checkpoint from a different mesh must not be silently reshaped into this one."""
    ck = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=False, verbose=False)
    ck.save_boundaries({0: _tree(n=4)}, loss=0.0)
    ck2 = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=True, verbose=False)
    with pytest.raises(SystemExit) as e:
        ck2.load_boundaries(like=_tree(n=8))
    assert "shape" in str(e.value)


# ---------------------------------------------------------------------------
# the reverse sweep's arithmetic
# ---------------------------------------------------------------------------

def _sweep(ck_mod, root, ident, n_chunks, stop_at_leg_seconds=None, resume=False,
           per_chunk=None, ckpt_every=2):
    """Drive a fake reverse sweep through the checkpoint, returning (grad, chunks_done)."""
    ck = ck_mod.Checkpoint.open(root, ident, resume=resume, ckpt_every=ckpt_every,
                                stop_after=stop_at_leg_seconds, verbose=False)
    if not ck.have_boundaries:
        ck.save_boundaries({c: _tree(float(c)) for c in range(n_chunks + 1)}, loss=1.0)
    cot, g, c_start, tr = ck.reverse_start(_tree(99.0), n_chunks, like=_tree())
    done = []
    for c in range(c_start, -1, -1):
        g += per_chunk(c)
        cot = _tree(float(c))
        tr.append(float(c))
        done.append(c)
        if ck.tick(c - 1, cot, g, tr):
            return g, done, False
    return g, done, ck.finished


def test_a_split_sweep_sums_the_same_chunks_exactly_once(ck_mod, tmp_path):
    """Stop mid-sweep, resume, finish — and the accumulated gradient must equal the one-shot
    run to the bit. Double-counting or skipping one chunk is the failure, and with a smooth
    per-chunk contribution neither would look obviously wrong in the total."""
    n = 10
    per = lambda c: 1.0 / (c + 1) ** 1.5  # noqa: E731  distinct, non-uniform contributions
    g_ref, done_ref, fin = _sweep(ck_mod, tmp_path / "ref", IDENT, n, per_chunk=per)
    assert fin and done_ref == list(range(n - 1, -1, -1))

    root = tmp_path / "split"
    # leg 1: stop_after=0 makes every tick "out of time", so it stops at the first checkpoint
    g1, d1, fin1 = _sweep(ck_mod, root, IDENT, n, stop_at_leg_seconds=0.0, per_chunk=per)
    assert not fin1 and len(d1) < n, "leg 1 must stop early"
    total, done = g1, list(d1)
    legs = 1
    while not fin1:
        legs += 1
        assert legs < 20, "the sweep is not making progress across legs"
        g1, d1, fin1 = _sweep(ck_mod, root, IDENT, n, stop_at_leg_seconds=0.0,
                              resume=True, per_chunk=per)
        done += d1
    assert legs > 2, "the point of the test is several legs"
    assert done == done_ref, "every chunk exactly once, in reverse order"
    assert g1 == g_ref, f"split sweep {g1!r} != one shot {g_ref!r}"


def test_manifest_never_names_the_slot_being_written(ck_mod, tmp_path):
    """The two-slot scheme is what survives a kill mid-write: the live slot is never the one
    just overwritten by an interrupted leg. Check the alternation actually happens."""
    n = 8
    root = tmp_path / "slots"
    slots = []
    ck = ck_mod.Checkpoint.open(root, IDENT, resume=False, ckpt_every=1, verbose=False)
    ck.save_boundaries({c: _tree(float(c)) for c in range(n + 1)}, loss=1.0)
    cot, g, c_start, tr = ck.reverse_start(_tree(99.0), n, like=_tree())
    for c in range(c_start, 0, -1):
        ck.tick(c - 1, _tree(float(c)), float(c), tr)
        slots.append(json.loads((root / "manifest.json").read_text())["state"]["slot"])
    expect = [("b", "a")[i % 2] for i in range(len(slots))]
    assert slots == expect, f"slots must alternate, got {slots}"
    assert (root / "cot_a.npz").exists() and (root / "cot_b.npz").exists()


def test_resume_reloads_the_cotangent_that_was_carried(ck_mod, tmp_path):
    n = 6
    root = tmp_path / "cot"
    ck = ck_mod.Checkpoint.open(root, IDENT, resume=False, ckpt_every=1, verbose=False)
    ck.save_boundaries({c: _tree(float(c)) for c in range(n + 1)}, loss=1.0)
    _, _, c_start, tr = ck.reverse_start(_tree(99.0), n, like=_tree())
    marker = _tree(1234.5)
    ck.tick(c_start - 1, marker, 3.25, tr + [1.0])

    ck2 = ck_mod.Checkpoint.open(root, IDENT, resume=True, verbose=False)
    cot, g, c_next, tr2 = ck2.reverse_start(_tree(0.0), n, like=_tree())
    assert c_next == c_start - 1 and g == 3.25
    for k in marker:
        assert np.array_equal(cot[k], marker[k])
    assert len(tr2) == len(tr) + 1, "the cotangent-norm trace must survive the leg boundary"


def test_completing_the_sweep_marks_it_done(ck_mod, tmp_path):
    n = 3
    ck = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=False, ckpt_every=1, verbose=False)
    ck.save_boundaries({c: _tree(float(c)) for c in range(n + 1)}, loss=1.0)
    _, _, c_start, tr = ck.reverse_start(_tree(1.0), n, like=_tree())
    for c in range(c_start, -1, -1):
        assert ck.tick(c - 1, _tree(float(c)), float(c), tr) is False
    assert ck.finished
    assert json.loads((tmp_path / "manifest.json").read_text())["state"]["phase"] == "done"


# ---------------------------------------------------------------------------
# the FORWARD resume (added 2026-08-09) — what makes a multi-year window reachable
#
# Without this the forward is all-or-nothing: a leg killed part way redoes it, 1.5 h at a year
# and 7.4 h at five, which is what put a 5-year rung out of reach — leg 1 would need 7.4 h of
# truth plus 7.4 h of forward before the reverse could even start, against a 12 h slot.

def test_a_fresh_checkpoint_starts_the_forward_from_scratch(ck_mod, tmp_path):
    ck = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=False, verbose=False)
    st, c = ck.forward_resume(like=_tree())
    assert st is None and c == 0
    assert not ck.have_boundaries, "an unfinished forward must never look finished"


def test_forward_resumes_at_the_last_boundary_that_reached_disk(ck_mod, tmp_path):
    """Boundary b is the state ENTERING chunk b, so the resume index is the next chunk to run
    — off by one here and the window silently integrates a chunk twice or skips one."""
    ck = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=False, verbose=False)
    for c in (0, 2, 4):                      # a stride-2 forward, killed after boundary 4
        assert ck.put_boundary(c, _tree(float(c))) is False
    assert not ck.have_boundaries, "phase must still be 'forward'"

    ck2 = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=True, verbose=False)
    st, c = ck2.forward_resume(like=_tree())
    assert c == 4
    for k in st:
        assert np.array_equal(st[k], _tree(4.0)[k])
    back = ck2.load_boundaries(like=_tree())
    assert sorted(back) == [0, 2, 4], "the earlier boundaries must come back too"


def test_finish_forward_is_what_flips_have_boundaries(ck_mod, tmp_path):
    ck = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=False, verbose=False)
    ck.put_boundary(0, _tree(0.0))
    ck.put_boundary(1, _tree(1.0))
    assert not ck.have_boundaries
    ck.finish_forward(loss=0.25)
    assert ck.have_boundaries and ck.loss == 0.25
    assert ck_mod.Checkpoint.open(tmp_path, IDENT, resume=True,
                                  verbose=False).state["n_boundaries"] == 2


def test_put_boundary_stops_the_leg_when_the_budget_is_gone(ck_mod, tmp_path):
    """The forward must be interruptible on the SAME budget as the reverse, or a long window's
    first leg runs past its wall clock and is killed mid-write instead of stopping cleanly."""
    ck = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=False, stop_after=0.0, verbose=False)
    assert ck.put_boundary(0, _tree(0.0)) is True
    # and the boundary it had just written is on disk and reachable
    ck2 = ck_mod.Checkpoint.open(tmp_path, IDENT, resume=True, verbose=False)
    assert ck2.state["fwd_done_upto"] == 0 and ck2.state["n_boundaries"] == 1
