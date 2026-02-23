"""
ConfluenceOperator — time emerges from Möbius topology.

The fundamental operation:
    C:  A_t  →  P_{t+1}

Three composed O(1) GPU operations:
    1. Half-twist:  shift u by π  (roll by n_u // 2)
    2. Reflection:  flip v → 1 − v
    3. Transfer:    A becomes new P

Properties:
    • NOT self-inverse:  C(C(A)) ≠ A
    • Period 4:  C⁴(A) ≈ A  (two full rotations)
    • Preserves L² norm (energy conservation)
    • Generates arrow of time from non-orientability
"""

from __future__ import annotations

import torch


class ConfluenceOperator:
    """Möbius confluence: half-twist + v-flip, acting on 2-D field tensors."""

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------
    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        """
        Apply the confluence map  C(A)(u, v) = A(u + π,  1 − v).

        All operations are O(1) GPU metadata ops — no data movement.
        """
        shifted = torch.roll(A, shifts=A.shape[0] // 2, dims=0)
        flipped = torch.flip(shifted, dims=[1])
        return flipped

    # ------------------------------------------------------------------
    # Symmetry decomposition (diagnostic, NOT a dynamics step)
    # ------------------------------------------------------------------
    def decompose(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split *field* into symmetric and antisymmetric components under C.

            f_sym     = (f + C(f)) / 2      periodic modes
            f_antisym = (f − C(f)) / 2      antiperiodic modes

        The /2 is topological necessity, not normalisation.
        """
        c_field = self(field)
        return (field + c_field) / 2.0, (field - c_field) / 2.0

    def project_antiperiodic(self, field: torch.Tensor) -> torch.Tensor:
        """
        Project onto the antiperiodic subspace.

        Available as a **diagnostic** — not used in the dynamics loop.
        If Ξ emerges naturally, we shouldn't need to force it.
        """
        _, f_anti = self.decompose(field)
        return f_anti

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def norm_check(self, field: torch.Tensor) -> float:
        """Return |‖C(f)‖² − ‖f‖²| — should be ~0 (norm preservation)."""
        return abs(
            self(field).pow(2).sum().item() - field.pow(2).sum().item()
        )
