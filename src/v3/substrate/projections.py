"""2D projection operators for the Möbius manifold.

Adapted from fracton's 3D projections (fracton.field.projections) for the
2D (nu, nv) Möbius strip grid used by Reality Engine v3.

Key insight from DFT: Maxwell (EM) uses antisymmetric projection (curl),
gravity uses symmetric projection (divergence). Both project from the
same pre-field constructed by stacking E, I, M along a field-type axis.

Pre-field: torch.stack([E, I, M], dim=0) → (3, nu, nv)
- Symmetric projection → gravity potential (amplitude envelope)
- Antisymmetric projection → EM field (phase structure)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


_EPS = 1e-12


# ---------------------------------------------------------------------------
# 2D differential operators
# ---------------------------------------------------------------------------

@torch.no_grad()
def _gradient_axis(field: torch.Tensor, dx: float, axis: int) -> torch.Tensor:
    """Central-difference gradient along one axis with periodic boundaries."""
    fwd = torch.roll(field, -1, dims=axis)
    bwd = torch.roll(field, 1, dims=axis)
    return (fwd - bwd) / (2 * dx)


@torch.no_grad()
def gradient_2d(
    field: torch.Tensor, du: float = 1.0, dv: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D gradient of a scalar field.

    Args:
        field: (nu, nv) or (..., nu, nv) tensor.
        du: Grid spacing in u-direction.
        dv: Grid spacing in v-direction.

    Returns:
        (grad_u, grad_v) tuple, each same shape as field.
    """
    grad_u = _gradient_axis(field, du, axis=-2)
    grad_v = _gradient_axis(field, dv, axis=-1)
    return grad_u, grad_v


@torch.no_grad()
def divergence_2d(
    Fu: torch.Tensor, Fv: torch.Tensor, du: float = 1.0, dv: float = 1.0,
) -> torch.Tensor:
    """Compute 2D divergence: div(F) = dFu/du + dFv/dv.

    This is the GRAVITY operator — sources create scalar potential.
    """
    dFu_du = _gradient_axis(Fu, du, axis=-2)
    dFv_dv = _gradient_axis(Fv, dv, axis=-1)
    return dFu_du + dFv_dv


@torch.no_grad()
def curl_2d(
    Fu: torch.Tensor, Fv: torch.Tensor, du: float = 1.0, dv: float = 1.0,
) -> torch.Tensor:
    """Compute 2D curl (scalar): curl(F) = dFv/du - dFu/dv.

    In 2D, curl of a vector field is a scalar (the z-component of the 3D curl).
    This is the MAXWELL operator — circulation creates field.
    """
    dFv_du = _gradient_axis(Fv, du, axis=-2)
    dFu_dv = _gradient_axis(Fu, dv, axis=-1)
    return dFv_du - dFu_dv


# ---------------------------------------------------------------------------
# Pre-field projections (Maxwell/Gravity from same pre-field)
# ---------------------------------------------------------------------------

@torch.no_grad()
def project_symmetric_2d(prefield: torch.Tensor) -> torch.Tensor:
    """Project pre-field symmetrically (Gravity).

    Extracts AMPLITUDE information by averaging absolute values over the
    field-type axis. Returns scalar gravitational potential.

    Args:
        prefield: (N_fields, nu, nv) tensor (e.g. stack of E, I, M).

    Returns:
        Scalar potential field (nu, nv).
    """
    return prefield.abs().mean(dim=0)


@torch.no_grad()
def project_antisymmetric_2d(
    prefield: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project pre-field antisymmetrically (Maxwell/EM).

    Extracts PHASE information by weighting with oscillatory phases over
    the field-type axis. Returns 2D vector field (Fu, Fv).

    In 2D we get two EM field components (u and v directions) from the
    real and imaginary parts of the phase-weighted sum.

    Args:
        prefield: (N_fields, nu, nv) tensor.

    Returns:
        (Fu, Fv) tuple of (nu, nv) tensors.
    """
    N = prefield.shape[0]

    # Oscillatory phase weights: exp(2πi·k/N) for k = 0..N-1
    phase = torch.exp(
        2j * math.pi * torch.arange(N, device=prefield.device, dtype=torch.float64) / N
    ).to(torch.complex128)

    # Reshape for broadcasting: (N, 1, 1)
    phase = phase.reshape(N, 1, 1)

    # Weight and average over field-type axis
    prefield_c = prefield.to(torch.complex128)
    weighted = prefield_c * phase
    projected = weighted.mean(dim=0)

    Fu = projected.real.to(prefield.dtype)
    Fv = projected.imag.to(prefield.dtype)
    return Fu, Fv


@torch.no_grad()
def depth_2_projection_2d(
    prefield: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """MED depth-2 projection for 2D fields.

    The same pre-field produces BOTH EM and gravity.

    Args:
        prefield: (N_fields, nu, nv) tensor.

    Returns:
        ((Fu, Fv), grav_potential) — EM vector field and gravity scalar.
    """
    em = project_antisymmetric_2d(prefield)
    grav = project_symmetric_2d(prefield)
    return em, grav
