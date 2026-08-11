"""
Tests for ConfluenceOperator — period, norm, decomposition.
"""

import pytest
import torch

from src.dynamics.confluence import ConfluenceOperator


DEVICE = "cpu"
N_U, N_V = 32, 16


@pytest.fixture
def C():
    return ConfluenceOperator()


@pytest.fixture
def field():
    torch.manual_seed(0)
    return torch.randn(N_U, N_V)


class TestConfluenceBasics:
    def test_not_identity(self, C, field):
        """C(A) ≠ A."""
        result = C(field)
        assert not torch.allclose(result, field)

    def test_shape_preserved(self, C, field):
        assert C(field).shape == field.shape

    def test_norm_preserved(self, C, field):
        """‖C(A)‖² = ‖A‖² — energy conservation."""
        norm_before = field.pow(2).sum().item()
        norm_after = C(field).pow(2).sum().item()
        assert abs(norm_after - norm_before) < 1e-3  # float32 precision

    def test_period_4(self, C, field):
        """C⁴(A) = A — four applications is identity."""
        result = field
        for _ in range(4):
            result = C(result)
        assert torch.allclose(result, field, atol=1e-12)

    def test_period_2(self, C, field):
        """C²(A) = A — the confluence map is an involution.
        
        C(f)(u,v) = f(u+π, 1-v), so C(C(f))(u,v) = f(u+2π, v) = f(u, v).
        The shift-by-π + v-flip composed twice returns to identity.
        """
        result = C(C(field))
        assert torch.allclose(result, field, atol=1e-6)


class TestDecomposition:
    def test_reconstruction(self, C, field):
        """f = f_sym + f_antisym (exact reconstruction)."""
        f_sym, f_anti = C.decompose(field)
        reconstructed = f_sym + f_anti
        assert torch.allclose(reconstructed, field, atol=1e-5)  # float32

    def test_idempotent_projection(self, C, field):
        """Projecting twice gives the same result."""
        f_anti = C.project_antiperiodic(field)
        f_anti2 = C.project_antiperiodic(f_anti)
        assert torch.allclose(f_anti, f_anti2, atol=1e-12)

    def test_orthogonal_decomposition(self, C, field):
        """‖f‖² = ‖f_sym‖² + ‖f_anti‖² (Parseval)."""
        f_sym, f_anti = C.decompose(field)
        norm_total = field.pow(2).sum().item()
        norm_parts = f_sym.pow(2).sum().item() + f_anti.pow(2).sum().item()
        assert abs(norm_total - norm_parts) < 1e-3  # float32

    def test_symmetric_component_invariant_under_C(self, C, field):
        """C(f_sym) = f_sym."""
        f_sym, _ = C.decompose(field)
        assert torch.allclose(C(f_sym), f_sym, atol=1e-12)

    def test_antisymmetric_component_negated_by_C(self, C, field):
        """C(f_anti) = −f_anti."""
        _, f_anti = C.decompose(field)
        assert torch.allclose(C(f_anti), -f_anti, atol=1e-12)


class TestNormCheck:
    def test_norm_check_small(self, C, field):
        assert C.norm_check(field) < 1e-3  # float32 precision
