"""
MobiusManifold — discrete Möbius band with topology-aware operators.

Parameterisation
    u  ∈ [0, 2π)   angular coordinate, *n_u* points (MUST be even)
    v  ∈ [0, 1]    cross-sectional coordinate, *n_v* points

Fundamental identification
    (u, v)  ~  (u + 2π,  1 − v)

The half-twist (u → u + π) corresponds to a shift of n_u // 2 cells in
the u-direction combined with flipping v → 1 − v.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class MobiusManifold:
    """Discrete Möbius band with topology-aware Laplacian and gradient."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        n_u: int = 128,
        n_v: int = 64,
        device: str | torch.device = "cuda",
    ) -> None:
        if n_u % 2 != 0:
            raise ValueError(f"n_u must be even for Möbius half-twist, got {n_u}")
        if n_v < 3:
            raise ValueError(f"n_v must be ≥ 3, got {n_v}")

        self.n_u = n_u
        self.n_v = n_v
        self.device = torch.device(device)

        # Grid spacings (for correct physical scaling)
        self.h_u: float = 2.0 * math.pi / n_u      # angular step
        self.h_v: float = 1.0 / (n_v - 1)           # cross-section step

        # Coordinate grids — persistent, on GPU
        u = torch.linspace(0, 2 * math.pi, n_u + 1, device=self.device)[:-1]
        v = torch.linspace(0, 1, n_v, device=self.device)
        self.u = u                                  # (n_u,)
        self.v = v                                  # (n_v,)
        self.U, self.V = torch.meshgrid(u, v, indexing="ij")  # (n_u, n_v)

    # ------------------------------------------------------------------
    # Möbius Laplacian  (5-point stencil)
    # ------------------------------------------------------------------
    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        r"""
        Compute ∇²f respecting Möbius topology.

        Interior: standard 5-point stencil.
        u-boundary: **antiperiodic** — wrapping u flips v.
        v-boundary: **Neumann** (zero-flux ∂f/∂v = 0 at edges).

        Returns the Laplacian tensor, same shape as *f*.
        """
        n_u, n_v = f.shape

        # --- u-direction (angular) -------------------------------------------
        # Start with periodic roll, then fix the Möbius seam rows.
        f_u_plus = torch.roll(f, -1, dims=0)          # f[i+1, j]
        f_u_minus = torch.roll(f, 1, dims=0)           # f[i−1, j]

        # Möbius correction: when wrapping around u, v flips.
        # Row n_u−1's "up" neighbor is row 0 with v reversed.
        f_u_plus[-1, :] = f[0, :].flip(0)
        # Row 0's "down" neighbor is row n_u−1 with v reversed.
        f_u_minus[0, :] = f[-1, :].flip(0)

        # --- v-direction (cross-section) — Neumann at both edges -------------
        f_v_plus = torch.roll(f, -1, dims=1)           # f[i, j+1]
        f_v_minus = torch.roll(f, 1, dims=1)            # f[i, j−1]

        # Ghost-cell mirror for zero-flux:
        f_v_plus[:, -1] = f[:, -2]
        f_v_minus[:, 0] = f[:, 1]

        # --- Combine with isotropic spacing (h = 1, absorbed into κ) ---------
        lap = f_u_plus + f_u_minus + f_v_plus + f_v_minus - 4.0 * f
        return lap

    # ------------------------------------------------------------------
    # Gradient magnitude  (used by actualization detector)
    # ------------------------------------------------------------------
    def gradient_magnitude(self, f: torch.Tensor) -> torch.Tensor:
        """
        |∇f| via central differences, Möbius-aware in u, Neumann in v.
        """
        n_u, n_v = f.shape

        # u-derivative
        f_u_plus = torch.roll(f, -1, dims=0)
        f_u_minus = torch.roll(f, 1, dims=0)
        f_u_plus[-1, :] = f[0, :].flip(0)
        f_u_minus[0, :] = f[-1, :].flip(0)
        df_du = (f_u_plus - f_u_minus) / 2.0

        # v-derivative (Neumann at edges: one-sided)
        f_v_plus = torch.roll(f, -1, dims=1)
        f_v_minus = torch.roll(f, 1, dims=1)
        f_v_plus[:, -1] = f[:, -2]
        f_v_minus[:, 0] = f[:, 1]
        df_dv = (f_v_plus - f_v_minus) / 2.0

        return torch.sqrt(df_du ** 2 + df_dv ** 2 + 1e-20)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def uniform_noise(
        self,
        mean: float = 0.5,
        std: float = 0.01,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Return a near-uniform field with small Gaussian perturbation."""
        if seed is not None:
            torch.manual_seed(seed)
        return (
            torch.ones(self.n_u, self.n_v, device=self.device) * mean
            + torch.randn(self.n_u, self.n_v, device=self.device) * std
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"MobiusManifold(n_u={self.n_u}, n_v={self.n_v}, "
            f"device={self.device})"
        )
