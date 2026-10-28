"""``core2_adj._XsAhead``: the next chunk's ``xs`` built on a worker thread while the device runs this one.

A fake chunk builder (no model, no mesh) is called in the orders the drivers use: a forward pass, ``chunked_grad``'s
reverse walk with span rebuilds (stride 3), a repeated chunk. Checked: every call returns what a direct build of that
chunk returns; the builds never overlap (one worker, the stateful forcing reader); a wrong guess is freed; a guess
that raises (past the window) is ignored while a requested chunk that raises raises.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "adjoint"))
import core2_adj as C  # noqa: E402


class Builder:
    """Chunk ``c`` -> ``{"c": jnp.full(3, c)}``, slowly; records the order, the results and any overlap."""

    def __init__(self, n_chunks: int | None = None):
        self.n = n_chunks
        self.order, self.made = [], []
        self.running = 0
        self.overlap = False
        self.lock = threading.Lock()

    def __call__(self, c):
        with self.lock:
            self.running += 1
            self.overlap |= self.running > 1
        try:
            time.sleep(0.01)
            if self.n is not None and not 0 <= c < self.n:
                raise ValueError(f"chunk {c} outside the window")
            out = {"c": jnp.full(3, float(c))}
            self.order.append(c)
            self.made.append((c, out))
            return out
        finally:
            with self.lock:
                self.running -= 1


def _reverse_walk(n: int, stride: int) -> list:
    """The xs_fn calls of ``adjoint.chunked_grad``: forward 0..n-1, then per span (last first) the rebuild lo..hi-2
    and the reverse hi-1..lo."""
    calls = list(range(n))
    for lo in reversed(range(0, n, stride)):
        hi = min(lo + stride, n)
        calls += list(range(lo, hi - 1)) + list(reversed(range(lo, hi)))
    return calls


@pytest.mark.parametrize("calls", [list(range(6)), _reverse_walk(9, 3), [0, 1, 1, 0, 1, 2]],
                         ids=["forward", "reverse_walk", "repeats"])
def test_every_call_returns_its_chunk_and_builds_never_overlap(calls):
    b = Builder()
    xs = C._XsAhead(b)
    got = []
    for c in calls:
        out = xs(c)
        assert float(out["c"][0]) == c and not out["c"].is_deleted()
        got.append(out)
    xs.pool.shutdown(wait=True)
    assert not b.overlap
    # every build was handed out or freed, apart from the guess after the last call
    returned = {id(o) for o in got}
    left = [c for c, out in b.made if id(out) not in returned and not out["c"].is_deleted()]
    assert len(left) <= 1, left


def test_a_forward_pass_guesses_right():
    b = Builder()
    xs = C._XsAhead(b)
    for c in range(6):
        xs(c)
    xs.pool.shutdown(wait=True)
    assert b.order == list(range(7))                    # each chunk once, plus the guess after the last
    assert not any(out["c"].is_deleted() for c, out in b.made if c < 6)


def test_a_wrong_guess_is_freed():
    b = Builder()
    xs = C._XsAhead(b)
    xs(4)
    xs(5)                                                # guesses 6
    xs(2)                                                # 6 was wrong
    wrong = [out for c, out in b.made if c == 6]
    assert len(wrong) == 1 and wrong[0]["c"].is_deleted()


def test_a_guess_past_the_window_is_ignored_and_a_bad_request_raises():
    b = Builder(n_chunks=3)
    xs = C._XsAhead(b)
    for c in range(3):                                   # the last guess (3) raises in the worker
        assert float(xs(c)["c"][0]) == c
    assert float(xs(1)["c"][0]) == 1                     # dropping the failed guess raises nothing
    with pytest.raises(ValueError, match="outside the window"):
        xs(5)


def test_prefetch_off_is_the_plain_builder(monkeypatch):
    class M:                                             # what xs_fn uses of a Model
        start, chunk_steps, schedule, shard = C._parse_date("2013-01-01"), 1, None, None

        def date(self, start, n):
            return (2013, 1 + n, 0.0, 1)

        def stack_forcing(self, dates):
            return {"day": jnp.asarray([d[1] for d in dates], jnp.float32)}

    monkeypatch.setenv("FESOM_FORCING_PREFETCH", "0")
    assert not isinstance(C.Model.xs_fn(M()), C._XsAhead)
    monkeypatch.delenv("FESOM_FORCING_PREFETCH")
    fn = C.Model.xs_fn(M())
    assert isinstance(fn, C._XsAhead) and float(fn(2)["day"][0]) == 3
