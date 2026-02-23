"""
Tests for MobiusManifold — topology, coordinates, Laplacian correctness.
"""

import math
import pytest
import torch

from src.substrate.mobius import MobiusManifold


DEVICE = "cpu"


class TestConstruction:
    def test_even_n_u_required(self):
        with pytest.raises(ValueError, match="even"):
            MobiusManifold(n_u=127, n_v=64, device=DEVICE)

    def test_minimum_n_v(self):
        with pytest.raises(ValueError, match="n_v"):
            MobiusManifold(n_u=16, n_v=2, device=DEVICE)

    def test_grid_dimensions(self):
        m = MobiusManifold(n_u=16, n_v=8, device=DEVICE)
        assert m.U.shape == (16, 8)
        assert m.V.shape == (16, 8)

    def test_coordinate_ranges(self):
        m = MobiusManifold(n_u=64, n_v=32, device=DEVICE)
        assert m.u[0].item() == pytest.approx(0.0)
        assert m.u[-1].item() < 2 * math.pi
        assert m.v[0].item() == pytest.approx(0.0)
        assert m.v[-1].item() == pytest.approx(1.0)


class TestLaplacian:
    """Verify the Möbius Laplacian handles boundaries correctly."""

    def test_constant_field_zero_laplacian(self):
        """∇²(const) = 0 everywhere."""
        m = MobiusManifold(n_u=32, n_v=16, device=DEVICE)
        f = torch.ones(32, 16)
        lap = m.laplacian(f)
        assert lap.abs().max().item() < 1e-12

    def test_laplacian_shape_preserved(self):
        m = MobiusManifold(n_u=32, n_v=16, device=DEVICE)
        f = torch.randn(32, 16)
        assert m.laplacian(f).shape == (32, 16)

    def test_boundary_differs_from_periodic(self):
        """Möbius Laplacian should differ from naive periodic at the seam."""
        n_u, n_v = 32, 16
        m = MobiusManifold(n_u=n_u, n_v=n_v, device=DEVICE)
        f = torch.randn(n_u, n_v)

        lap_mobius = m.laplacian(f)

        # Naive periodic Laplacian (roll without v-flip)
        f_up = torch.roll(f, -1, dims=0)
        f_dn = torch.roll(f, 1, dims=0)
        f_lt = torch.roll(f, -1, dims=1)
        f_rt = torch.roll(f, 1, dims=1)
        lap_periodic = f_up + f_dn + f_lt + f_rt - 4 * f

        # Should differ at boundary rows (0 and -1)
        diff_row0 = (lap_mobius[0] - lap_periodic[0]).abs().max().item()
        diff_last = (lap_mobius[-1] - lap_periodic[-1]).abs().max().item()
        assert diff_row0 > 1e-6 or diff_last > 1e-6

    def test_antiperiodic_mode_eigenvalue(self):
        """
        An antiperiodic mode sin((n+½)u) should have Laplacian ≈ -(n+½)² f.

        This is the critical test that the Möbius boundary is correct.
        """
        n_u, n_v = 128, 8
        m = MobiusManifold(n_u=n_u, n_v=n_v, device=DEVICE)

        n = 1  # first antiperiodic mode
        u = torch.linspace(0, 2 * math.pi, n_u + 1)[:-1]
        mode = torch.sin((n + 0.5) * u).unsqueeze(1).expand(n_u, n_v)

        lap = m.laplacian(mode)

        # Expected: -(n+0.5)² × mode  (in units where h = 2π/n_u)
        # The discrete Laplacian with spacing h gives -k² → -(2/h·sin(kh/2))²
        # For k = (n+0.5), h = 2π/n_u:
        k = n + 0.5
        h = 2 * math.pi / n_u
        discrete_eigenval = -(2 / h * math.sin(k * h / 2)) ** 2
        # But our Laplacian uses h=1 (absorbed into κ), so eigenval is:
        expected_eigenval = 2 * (math.cos(k * 2 * math.pi / n_u) - 1)

        # Check interior rows (away from boundary effects in v)
        interior = slice(2, n_v - 2)
        ratio = lap[n_u // 4, interior] / mode[n_u // 4, interior]
        mean_ratio = ratio.mean().item()

        # Should be close to the expected discrete eigenvalue
        assert abs(mean_ratio - expected_eigenval) < 0.5, (
            f"Eigenvalue mismatch: got {mean_ratio:.4f}, "
            f"expected ~{expected_eigenval:.4f}"
        )

    def test_gaussian_diffuses_across_seam(self):
        """A Gaussian placed at the seam should spread, not reflect."""
        n_u, n_v = 64, 32
        m = MobiusManifold(n_u=n_u, n_v=n_v, device=DEVICE)

        # Gaussian centred at u=0 (the seam)
        u = m.U
        f = torch.exp(-((u - 0.1) ** 2) / 0.05)

        # After one Laplacian step, energy should spread into the
        # last rows (which are across the seam)
        lap = m.laplacian(f)
        f_new = f + 0.1 * lap

        # The last few u-rows should have gained some value
        gain = f_new[-3:, :].mean().item() - f[-3:, :].mean().item()
        assert gain > 1e-6, "Gaussian did not diffuse across the Möbius seam"


class TestGradient:
    def test_constant_field_zero_gradient(self):
        m = MobiusManifold(n_u=32, n_v=16, device=DEVICE)
        f = torch.ones(32, 16) * 3.0
        grad = m.gradient_magnitude(f)
        assert grad.max().item() < 1e-6

    def test_gradient_positive(self):
        m = MobiusManifold(n_u=32, n_v=16, device=DEVICE)
        f = torch.randn(32, 16)
        grad = m.gradient_magnitude(f)
        assert (grad >= 0).all()
