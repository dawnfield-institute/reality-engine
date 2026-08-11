"""
Tests for PACTracker — conservation, enforcement, history.
"""

import pytest
import torch

from src.substrate.state import FieldState
from src.dynamics.pac import PACTracker


DEVICE = "cpu"
N_U, N_V = 16, 8


@pytest.fixture
def initial_state():
    torch.manual_seed(42)
    P = torch.ones(N_U, N_V) * 0.5 + torch.randn(N_U, N_V) * 0.01
    A = torch.ones(N_U, N_V) * 0.5 + torch.randn(N_U, N_V) * 0.01
    M = torch.zeros(N_U, N_V)
    return FieldState(P=P, A=A, M=M, t=0)


@pytest.fixture
def pac(initial_state):
    return PACTracker(initial_state)


class TestPACMeasure:
    def test_initial_residual_zero(self, pac, initial_state):
        diag = pac.measure(initial_state)
        assert diag["residual"] < 1e-12

    def test_history_grows(self, pac, initial_state):
        pac.measure(initial_state)
        pac.measure(initial_state)
        assert len(pac.history) == 2

    def test_detects_perturbation(self, pac, initial_state):
        perturbed = initial_state.clone()
        perturbed.P += 0.1
        diag = pac.measure(perturbed)
        assert diag["residual"] > 1e-6


class TestPACEnforce:
    def test_corrects_perturbation(self, pac, initial_state):
        perturbed = initial_state.clone()
        perturbed.P += 0.1  # deliberate drift
        corrected = pac.enforce(perturbed)

        # After enforcement, total should match target
        C = (corrected.P.sum() + corrected.A.sum() + corrected.M.sum()).item()
        assert abs(C - pac.C_target) < 1e-4  # float32 precision

    def test_noop_when_conserved(self, pac, initial_state):
        before = initial_state.clone()
        after = pac.enforce(initial_state)
        # Should be essentially unchanged
        assert torch.allclose(before.P, after.P, atol=1e-12)

    def test_conservation_over_many_steps(self, pac, initial_state):
        """Simulate drift + enforcement for 1000 steps."""
        state = initial_state.clone()
        for i in range(1000):
            # Simulate small drift
            state.P += torch.randn_like(state.P) * 0.001
            state.A += torch.randn_like(state.A) * 0.001
            state = pac.enforce(state)

        C_final = (state.P.sum() + state.A.sum() + state.M.sum()).item()
        assert abs(C_final - pac.C_target) < 1e-8
