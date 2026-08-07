"""Tests for the identity-gate helpers themselves.

Three ``stop_gradient`` identity gates now delegate their pass/fail decision to
``_identity_gates``, so a bug in it would weaken all three at once and in the direction that
is hardest to notice — silently accepting a real change. These tests cost no model evaluation:
the "losses" are one-line lambdas, so this file runs in milliseconds on any backend and can be
grouped with the cheap unit tests.

The GPU branch is exercised by replacing ``on_cpu`` rather than by needing a GPU, which is the
only way to test it in the CPU suite. That is legitimate here because the helper's backend
dependence is exactly one function call and nothing else in it touches a device.
"""

from __future__ import annotations

import math

import pytest

from fesom_jax.tests import _identity_gates as G

ONE_ULP_ABOVE_ONE = math.nextafter(1.0, 2.0)   # 1.0 + 1e-16 rounds back to 1.0; this does not


@pytest.fixture
def as_gpu(monkeypatch):
    """Take the non-CPU branch of every helper."""
    monkeypatch.setattr(G, "on_cpu", lambda: False)


# --- CPU: bit-identical, and nothing weaker ------------------------------------------------

def test_cpu_forward_accepts_only_exact_equality():
    G.assert_forward_identity(lambda x: 1.0, lambda x: 1.0, 0.0, "identical")
    with pytest.raises(AssertionError, match="moved the forward"):
        G.assert_forward_identity(lambda x: 1.0, lambda x: ONE_ULP_ABOVE_ONE, 0.0, "one ulp")


def test_cpu_gradient_accepts_only_exact_equality_and_needs_no_signal():
    """The reference-effect gradient costs an extra compile, so the CPU branch must not build
    it. If this ever regresses, the CPU suite silently doubles the cost of two tests."""
    def must_not_be_called():
        raise AssertionError("signal_fn was evaluated on the CPU branch")

    G.assert_dead_branch_gradient(1.0, 1.0, "identical", signal_fn=must_not_be_called)
    with pytest.raises(AssertionError, match="gradients differ"):
        G.assert_dead_branch_gradient(1.0, ONE_ULP_ABOVE_ONE, "one ulp",
                                      signal_fn=must_not_be_called)


# --- GPU: calibrated against a scale measured in the test ----------------------------------

def _ref(x):
    """A loss with a ~1.0 response per unit of control, so a 1e-10 probe gives a 1e-10 scale."""
    return 10.0 + float(x)


def test_gpu_forward_accepts_the_measured_compilation_spread(as_gpu):
    """2.9e-13 is the largest forward spread ever measured between two separately-compiled
    equivalent programs here (whichEVP=0, job 26780293). It must pass, or the gate is unusable
    on GPU; and it must pass with margin, not by a hair."""
    G.assert_forward_identity(_ref, lambda x: 10.0 + 2.9e-13, 0.0, "measured EVP spread")


def test_gpu_forward_rejects_a_difference_above_the_probe_fraction(as_gpu):
    with pytest.raises(AssertionError, match="more than 0.1"):
        G.assert_forward_identity(_ref, lambda x: 10.0 + 2e-11, 0.0, "20% of the probe")


def test_gpu_forward_rejects_a_control_that_does_not_reach_the_loss(as_gpu):
    """The live-control check. A fixture whose perturbation never reaches the loss is the trap
    that made a whole file of adjoint tests measure a switched-off closure (PORTING_LESSONS,
    2026-08-07); here it fails loudly instead of passing vacuously."""
    with pytest.raises(AssertionError, match="did not move the loss"):
        G.assert_forward_identity(lambda x: 10.0, lambda x: 10.0, 0.0, "dead control")


def test_gpu_gradient_accepts_the_measured_reverse_pass_spread(as_gpu):
    """The two values are what adjoint_cfg=None and AdjointConfig() actually returned on an
    A100 (3.9e-11 apart), against the ~7e-3 that freeze_mixing moves the same gradient."""
    G.assert_dead_branch_gradient(0.8206025750113952, 0.8206025749724217, "measured pi case",
                                  signal_fn=lambda: 0.8138278)


def test_gpu_gradient_rejects_a_difference_that_is_a_real_effect(as_gpu):
    with pytest.raises(AssertionError, match="too large to be compilation noise"):
        G.assert_dead_branch_gradient(0.82, 0.8200001, "1e-7 apart", signal_fn=lambda: 0.8138278)


def test_gpu_gradient_rejects_a_reference_flag_with_no_effect(as_gpu):
    """If the calibrating flag does nothing either, there is no scale and the comparison is
    meaningless — that must fail rather than divide by a zero signal."""
    with pytest.raises(AssertionError, match="no .*signal to calibrate"):
        G.assert_dead_branch_gradient(1.0, 1.0, "dead signal", signal_fn=lambda: 1.0)


def test_the_gpu_bounds_stay_far_from_a_physics_change():
    """Guard on the constants themselves: both must be small enough that an O(1) relative
    change — what a genuinely different physics path would produce — cannot slip through."""
    assert G.FWD_FRACTION_OF_PROBE <= 0.1
    assert G.GRAD_FRACTION_OF_SIGNAL <= 1e-6
    assert G.PROBE <= 1e-9, "the calibrating perturbation must stay physically negligible"
