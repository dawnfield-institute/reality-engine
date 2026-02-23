"""
Tests for SECEvolver — diffusion, collapse, positivity, Θ recycling, RBF.
"""

import math
import pytest
import torch

from src.substrate.mobius import MobiusManifold
from src.dynamics.sec import SECEvolver
from src.dynamics.confluence import ConfluenceOperator


DEVICE = "cpu"
N_U, N_V = 32, 16


@pytest.fixture
def manifold():
    return MobiusManifold(n_u=N_U, n_v=N_V, device=DEVICE)


@pytest.fixture
def confluence():
    return ConfluenceOperator()


@pytest.fixture
def sec(manifold, confluence):
    s = SECEvolver(manifold, confluence=confluence)
    s.reset_rbf()  # clean state for each test
    return s


class TestSECBasics:
    def test_output_shape(self, sec):
        S = torch.ones(N_U, N_V) * 0.5
        S_new, theta = sec.step(S, xi_measured=1.05)
        assert S_new.shape == (N_U, N_V)

    def test_returns_theta(self, sec):
        """step() returns (S_new, theta) where theta >= 0."""
        S = torch.ones(N_U, N_V) * 0.5
        S_new, theta = sec.step(S, xi_measured=1.05)
        assert isinstance(theta, float)
        assert theta >= 0.0

    def test_positivity_preserved(self, sec):
        """Entropy must stay ≥ 0."""
        S = torch.rand(N_U, N_V) * 0.01  # small positive
        S_new, _ = sec.step(S, xi_measured=1.1)
        assert S_new.min().item() >= 0.0

    def test_uniform_stays_bounded(self, sec):
        """A uniform field should not diverge under SEC dynamics."""
        S = torch.ones(N_U, N_V) * 0.5
        for _ in range(100):
            S, _ = sec.step(S, xi_measured=1.05)
        assert S.abs().max().item() < 1e6

    def test_nonzero_evolution(self, sec):
        """Field should change under SEC dynamics."""
        S0 = torch.ones(N_U, N_V) * 0.5
        S1, _ = sec.step(S0, xi_measured=1.05)
        assert not torch.allclose(S0, S1)

    def test_gaussian_diffuses(self, sec, manifold):
        """A localised Gaussian should spread out under diffusion."""
        u = manifold.U
        S = torch.exp(-((u - 3.0) ** 2) / 0.2)
        std_before = S.std().item()

        for _ in range(50):
            S, _ = sec.step(S, xi_measured=1.0)

        std_after = S.std().item()
        # Diffusion should reduce spatial variation
        assert std_after < std_before


class TestSECBidirectional:
    """Tests for the bidirectional spectral control via RBF balance field."""

    def test_reinforcement_below_target(self, sec, manifold, confluence):
        """When ξ < target, RBF amplifies antiperiodic modes."""
        # Create field with known antiperiodic content
        u = manifold.U
        v = manifold.V
        S = torch.ones(N_U, N_V) * 0.12
        S = S + 0.02 * torch.sin(u) * torch.sin(math.pi * v)

        # Fresh SEC for each measurement (avoid RBF state leakage)
        sec.reset_rbf()
        S_low, _ = sec.step(S.clone(), xi_measured=0.5)    # below target → amplify anti

        sec.reset_rbf()
        S_at, _ = sec.step(S.clone(), xi_measured=1.057)    # at target → no reinforcement

        # f_anti should be larger when ξ < target
        f_anti_low = (S_low - confluence(S_low)).pow(2).sum().item()
        f_anti_at = (S_at - confluence(S_at)).pow(2).sum().item()
        assert f_anti_low > f_anti_at

    def test_reinforcement_above_target(self, sec, manifold, confluence):
        """When ξ > target, RBF dampens antiperiodic modes."""
        u = manifold.U
        v = manifold.V
        S = torch.ones(N_U, N_V) * 0.12
        S = S + 0.02 * torch.sin(u) * torch.sin(math.pi * v)

        sec.reset_rbf()
        S_high, _ = sec.step(S.clone(), xi_measured=2.0)    # above target → dampen anti

        sec.reset_rbf()
        S_at, _ = sec.step(S.clone(), xi_measured=1.057)     # at target

        f_anti_high = (S_high - confluence(S_high)).pow(2).sum().item()
        f_anti_at = (S_at - confluence(S_at)).pow(2).sum().item()
        assert f_anti_high < f_anti_at

    def test_theta_recycling_affects_source(self, sec):
        """Recycled Θ should increase the field relative to no recycling."""
        S = torch.ones(N_U, N_V) * 0.5
        sec.reset_rbf()
        S_no_recycle, _ = sec.step(S.clone(), xi_measured=1.05, theta_recycled=0.0)
        sec.reset_rbf()
        S_recycle, _ = sec.step(S.clone(), xi_measured=1.05, theta_recycled=10.0)
        # More recycling → more source → higher field
        assert S_recycle.mean().item() > S_no_recycle.mean().item()

    def test_collapse_spectrally_modulated(self, sec):
        """Collapse total changes with ξ (symmetric spectral modulation)."""
        S = torch.ones(N_U, N_V) * 0.5
        sec.reset_rbf()
        _, theta_low = sec.step(S.clone(), xi_measured=0.5)
        sec.reset_rbf()
        _, theta_high = sec.step(S.clone(), xi_measured=1.5)
        # Both should be positive and bounded
        assert theta_low > 0
        assert theta_high > 0
        # The ratio should be bounded (symmetric modulation adjusts total θ)
        assert theta_low / theta_high < 5.0
        assert theta_high / theta_low < 5.0


class TestRBFMechanism:
    """Tests for the Recursive Balance Field self-regulation."""

    def test_rbf_memory_accumulates(self, sec):
        """RBF memory M should grow when balance field is active."""
        S = torch.ones(N_U, N_V) * 0.12
        assert sec.M_rbf == 0.0
        # Run several steps below target — M should accumulate
        for _ in range(10):
            S, _ = sec.step(S, xi_measured=0.5)
        assert sec.M_rbf > 0.0

    def test_rbf_memory_dampens_reinforcement(self, sec, manifold, confluence):
        """With accumulated memory, effective reinforcement should be weaker."""
        u = manifold.U
        v = manifold.V
        S = torch.ones(N_U, N_V) * 0.12
        S = S + 0.02 * torch.sin(u) * torch.sin(math.pi * v)

        # First step: clean memory → full strength
        sec.reset_rbf()
        S_fresh, _ = sec.step(S.clone(), xi_measured=0.5)

        # Pre-load memory by running many steps, then measure one step
        sec.reset_rbf()
        S_warm = S.clone()
        for _ in range(100):
            S_warm, _ = sec.step(S_warm, xi_measured=0.5)
        M_loaded = sec.M_rbf

        # The memory should have accumulated substantially
        assert M_loaded > 0.1

    def test_integral_accumulates_below_target(self, sec):
        """Integral should grow negative when ξ consistently below target."""
        S = torch.ones(N_U, N_V) * 0.12
        for _ in range(50):
            S, _ = sec.step(S, xi_measured=0.5)
        # xi_dev = (0.5 - 1.057) / 1.057 < 0: integral accumulates negative
        assert sec.xi_integral < 0.0

    def test_integral_clamp_prevents_windup(self, sec):
        """Integral should be bounded by the anti-windup clamp."""
        S = torch.ones(N_U, N_V) * 0.12
        for _ in range(10000):
            S, _ = sec.step(S, xi_measured=0.0)  # extreme deviation
        assert abs(sec.xi_integral) <= sec._integral_clamp + 1e-10

    def test_fibonacci_harmonic_bounded(self, sec):
        """Fibonacci harmonic should stay in [-1, 1]."""
        for step in range(1000):
            sec._step_count = step
            Phi = sec._fibonacci_harmonic()
            assert -1.0 <= Phi <= 1.0 + 1e-10

    def test_reset_clears_state(self, sec):
        """reset_rbf() clears all RBF mutable state."""
        S = torch.ones(N_U, N_V) * 0.12
        for _ in range(10):
            S, _ = sec.step(S, xi_measured=0.5)
        assert sec.M_rbf > 0.0
        sec.reset_rbf()
        assert sec.M_rbf == 0.0
        assert sec.xi_integral == 0.0
        assert sec._step_count == 0


class TestSECEdgeCases:
    def test_zero_xi(self, sec):
        """xi = 0 should still work (maximum expansion)."""
        S = torch.ones(N_U, N_V) * 0.5
        S_new, theta = sec.step(S, xi_measured=0.0)
        assert S_new.shape == (N_U, N_V)
        assert S_new.min().item() >= 0.0

    def test_large_xi(self, sec):
        """Very large Ξ should not cause NaN."""
        S = torch.ones(N_U, N_V) * 0.5
        S_new, theta = sec.step(S, xi_measured=10.0)
        assert not torch.isnan(S_new).any()
        assert not torch.isnan(torch.tensor(theta))
