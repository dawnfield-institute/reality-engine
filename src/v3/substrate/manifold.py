"""MobiusManifold — vectorized Möbius strip geometry.

Provides coordinate grids, antiperiodic projection, and neighbour utilities.
All operations are pure torch — GPU-accelerable out of the box.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from .constants import XI, TWIST_ANGLE


class MobiusManifold:
    """Discrete Möbius strip with antiperiodic boundary conditions.

    The manifold has shape (nu, nv) where nu is the circumference (must be even)
    and nv is the width. The fundamental constraint is:

        f(u + π, 1 - v) = -f(u, v)

    This antiperiodic boundary is what gives rise to Ξ-balance.
    """

    def __init__(
        self,
        nu: int = 128,
        nv: int = 32,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if nu % 2 != 0:
            raise ValueError(f"nu must be even for Möbius topology, got {nu}")
        self.nu = nu
        self.nv = nv
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self._half = nu // 2  # discrete π shift

        # Pre-compute coordinate grids
        u = torch.linspace(0, 2 * TWIST_ANGLE, nu, dtype=dtype, device=self.device)
        v = torch.linspace(0, 1, nv, dtype=dtype, device=self.device)
        self.u_grid, self.v_grid = torch.meshgrid(u, v, indexing="ij")

        # 5-point 2D Laplacian stencil as conv2d kernel
        kernel = torch.zeros(1, 1, 3, 3, dtype=dtype, device=self.device)
        kernel[0, 0, 0, 1] = 1.0
        kernel[0, 0, 2, 1] = 1.0
        kernel[0, 0, 1, 0] = 1.0
        kernel[0, 0, 1, 2] = 1.0
        kernel[0, 0, 1, 1] = -4.0
        self._laplacian_kernel = kernel

    # -- Antiperiodic projection ------------------------------------------

    def twist(self, field: torch.Tensor) -> torch.Tensor:
        """Compute f(u+π, 1-v) — the Möbius-twisted version of *field*."""
        shifted = torch.roll(field, shifts=self._half, dims=0)
        return torch.flip(shifted, dims=[1])

    def project_antiperiodic(self, field: torch.Tensor) -> torch.Tensor:
        """Project onto the antiperiodic subspace: (f - f_twisted) / 2."""
        return (field - self.twist(field)) / 2.0

    def validate_antiperiodicity(self, field: torch.Tensor) -> float:
        """RMS error from perfect antiperiodicity: ||f_twisted + f||."""
        return (self.twist(field) + field).pow(2).mean().sqrt().item()

    # -- Differential operators -------------------------------------------

    @torch.no_grad()
    def laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """2D Laplacian via F.conv2d with replicate padding."""
        f = field.unsqueeze(0).unsqueeze(0)  # (1, 1, nu, nv)
        f_padded = F.pad(f, (1, 1, 1, 1), mode="replicate")
        lap = F.conv2d(f_padded, self._laplacian_kernel)
        return lap.squeeze(0).squeeze(0)

    @torch.no_grad()
    def gradient_magnitude(self, field: torch.Tensor) -> torch.Tensor:
        """||∇f|| via central differences."""
        du = torch.roll(field, -1, 0) - torch.roll(field, 1, 0)
        dv = torch.roll(field, -1, 1) - torch.roll(field, 1, 1)
        return torch.sqrt(du.pow(2) + dv.pow(2) + 1e-10)
