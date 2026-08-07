"""The CPU-exact / GPU-tolerance split used by the ``stop_gradient`` identity gates.

These gates enforce the property that makes every backward-only switch safe to run in
production: ``stop_gradient`` is the identity in the forward, so ``AdjointConfig(...)`` and
``IceConfig(adjoint_mode=...)`` change the gradient and nothing else. The trajectory stays
bit-for-bit the validated model.

**On CPU that is literally checkable, and it is checked exactly.** Nothing below relaxes it.

On GPU it is not checkable, and the reason is not rounding in the physics. The two sides of
the comparison are separate ``jax.jit`` closures, so XLA compiles two *different* — though
mathematically equivalent — programs and is free to fuse and reassociate each one its own
way. The spread that produces is not "an ulp". Measured on an A100, job ``26780293``,
2026-08-07:

===========================================  ==============  ========
comparison                                   relative        in ulp
===========================================  ==============  ========
forward, 1 CORE2 step, mEVP                  1.6e-16         1
forward, 1 CORE2 step, whichEVP=0 (120 sub)  2.7e-14         ~160
gradient, 8 pi steps, reverse pass           4.7e-11         ~2e5
===========================================  ==============  ========

Three tests failed on GPU while passing on CPU because their tolerances had been set from the
first row. The fix is not a bigger magic number. Each helper here compares the difference
under test against a **scale measured in the same test**, so the bound carries a meaning that
survives a hardware or XLA change:

* :func:`assert_forward_identity` — the mode difference must be a small fraction of the
  loss's response to a deliberately negligible perturbation of the control. On the ice gates
  that reads: switching the adjoint mode moves the one-step SST field by less than a tenth of
  what a 1e-10 K initial-temperature perturbation does.
* :func:`assert_dead_branch_gradient` — the difference between two configurations that should
  be identical must be orders of magnitude below the difference a configuration that really
  changes something makes.

Both also assert that the calibrating scale is nonzero, which is a live-control check: a
fixture whose perturbation does not reach the loss (the trap that made a whole file of these
tests measure a switched-off closure) fails here instead of passing vacuously.
"""

from __future__ import annotations

import jax

# Fractions of the measured scale. Both leave 2+ orders of headroom over the numbers in the
# table above; neither is loose enough to hide a physics change, which would be O(1) relative.
FWD_FRACTION_OF_PROBE = 0.1      # measured worst case: 2.9e-13 against a 1e-10 scale = 0.003
GRAD_FRACTION_OF_SIGNAL = 1e-6   # measured: 3.9e-11 against a 7e-3 signal = 5.6e-9

PROBE = 1e-10   # size of the negligible control perturbation used to calibrate the forward


def on_cpu() -> bool:
    return jax.default_backend() == "cpu"


def assert_forward_identity(ref, alt, k0, what, *, probe=PROBE,
                            fraction=FWD_FRACTION_OF_PROBE):
    """Assert ``alt`` has the same forward as ``ref`` at the control point ``k0``.

    ``ref`` and ``alt`` are jitted loss functions of one scalar control. Exact equality on
    CPU. On GPU, ``|ref - alt|`` must be at most ``fraction`` of ``|ref(k0+probe) - ref(k0)|``
    — the loss's response to a perturbation far below anything physically meaningful (1e-10 K
    against an SST of O(10 K)).

    Costs one extra *call* on GPU, not an extra compile: the probe reuses ``ref``.
    """
    a, b = float(ref(k0)), float(alt(k0))
    if on_cpu():
        assert a == b, f"{what} moved the forward: {a!r} vs {b!r}"
        return
    scale = abs(float(ref(k0 + probe)) - a)
    assert scale > 0.0, (
        f"{what}: a {probe:g} perturbation of the control did not move the loss at all, so "
        f"there is no scale to compare against and the fixture is not exercising anything")
    assert abs(a - b) <= fraction * scale, (
        f"{what} moved the forward by {abs(a - b):.3e} — more than {fraction:g} of the "
        f"{scale:.3e} response to a {probe:g} control perturbation, so this is a real change "
        f"and not XLA compiling two equivalent programs differently ({a!r} vs {b!r})")


def assert_dead_branch_gradient(g_ref, g_alt, what, *, signal_fn,
                                fraction=GRAD_FRACTION_OF_SIGNAL):
    """Assert two configurations that should be identical give the same gradient.

    ``signal_fn`` is a zero-argument callable returning the gradient under a configuration
    that genuinely changes something — the scale a real effect has in this fixture. Exact
    equality on CPU, where ``signal_fn`` is never called, so the extra compile it costs is
    paid only on GPU. There the difference must be at most ``fraction`` of that effect, i.e.
    at least a million times smaller than the smallest difference the knobs exist to detect.
    """
    if on_cpu():
        assert g_ref == g_alt, f"{what}: gradients differ, {g_ref!r} vs {g_alt!r}"
        return
    signal = abs(signal_fn() - g_alt)
    assert signal > 0.0, (
        f"{what}: the reference flag did not change the gradient either, so there is no "
        f"signal to calibrate against — the fixture is measuring a dead code path")
    assert abs(g_ref - g_alt) <= fraction * signal, (
        f"{what}: gradients differ by {abs(g_ref - g_alt):.3e}, more than {fraction:g} of the "
        f"{signal:.3e} a real flag moves it. That is too large to be compilation noise "
        f"({g_ref!r} vs {g_alt!r})")
