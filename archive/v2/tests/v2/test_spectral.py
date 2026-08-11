"""
Tests for SpectralAnalyzer — decomposition, FFT, Ξ measurement.
"""

import math
import pytest
import torch

from src.dynamics.confluence import ConfluenceOperator
from src.analysis.spectral import SpectralAnalyzer


DEVICE = "cpu"
N_U, N_V = 64, 16


@pytest.fixture
def C():
    return ConfluenceOperator()


@pytest.fixture
def spectral(C):
    return SpectralAnalyzer(C)


class TestSpectralDecomposition:
    def test_decompose_matches_confluence(self, spectral, C):
        torch.manual_seed(0)
        f = torch.randn(N_U, N_V)
        f_sym, f_anti = spectral.decompose(f)
        f_sym2, f_anti2 = C.decompose(f)
        assert torch.allclose(f_sym, f_sym2)
        assert torch.allclose(f_anti, f_anti2)


class TestXiMeasurement:
    def test_returns_float(self, spectral):
        f = torch.randn(N_U, N_V)
        xi = spectral.compute_xi(f)
        assert isinstance(xi, float)

    def test_positive(self, spectral):
        torch.manual_seed(42)
        f = torch.randn(N_U, N_V)
        xi = spectral.compute_xi(f)
        assert xi > 0

    def test_pure_symmetric_mode(self, spectral):
        """A pure symmetric (periodic) mode should give low Ξ."""
        u = torch.linspace(0, 2 * math.pi, N_U + 1)[:-1]
        # cos(n*u) is symmetric under the Möbius map when n is integer
        f = torch.cos(2 * u).unsqueeze(1).expand(N_U, N_V)
        xi = spectral.compute_xi(f)
        # Ξ should be small (most energy in symmetric component)
        assert xi < 2.0  # generous bound

    def test_pure_antisymmetric_mode(self, spectral):
        """A pure antiperiodic mode should give high Ξ."""
        u = torch.linspace(0, 2 * math.pi, N_U + 1)[:-1]
        # sin((n+½)u) is antisymmetric under the Möbius map
        f = torch.sin(0.5 * u).unsqueeze(1).expand(N_U, N_V)
        xi = spectral.compute_xi(f)
        # Ξ should be large (most energy in antisymmetric component)
        assert xi > 0.5

    def test_degenerate_constant_field(self, spectral):
        """Constant field → xi = 1.0 (degenerate fallback)."""
        f = torch.ones(N_U, N_V)
        xi = spectral.compute_xi(f)
        assert xi == pytest.approx(1.0, abs=0.1)


class TestPowerSpectrum:
    def test_returns_dict(self, spectral):
        f = torch.randn(N_U, N_V)
        ps = spectral.power_spectrum(f)
        assert "sym" in ps and "anti" in ps

    def test_correct_length(self, spectral):
        f = torch.randn(N_U, N_V)
        ps = spectral.power_spectrum(f)
        assert ps["sym"].shape[0] == N_U
        assert ps["anti"].shape[0] == N_U
