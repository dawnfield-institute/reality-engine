"""
Möbius Confluence Operator - Geometric Time Stepping

Implements time evolution via geometric inversion on the Möbius manifold:
    P_{t+1}(u, v) = A_t(u+π, 1-v)

This geometric transformation:
- Inverts the Potential field through Möbius topology
- Creates temporal flow from spatial geometry
- Preserves anti-periodic boundaries
- Enables time to emerge from structure

Key Concept:
Time is not a background coordinate - it emerges from the confluence
of Potential (what could be) and Actual (what is). The Möbius twist
creates directional flow from timeless geometry.
"""

import torch
from typing import Tuple, Optional
import torch.nn.functional as F


class MobiusConfluence:
    """
    Confluence operator: Time stepping via geometric inversion.
    
    The confluence operation transforms Potential → Actual through
    the Möbius manifold's non-orientable topology. This creates
    temporal flow without assuming time exists a priori.
    
    Mathematical Form:
        P_{t+1}(u, v) = A_t(u+π, 1-v)
    
    This implements:
    1. Spatial shift: u → u+π (half-twist)
    2. Vertical flip: v → 1-v (reflection)
    3. Field transfer: A_t → P_{t+1} (confluence)
    """
    
    def __init__(
        self,
        size: Tuple[int, int],
        device: str = 'cpu'
    ):
        """
        Initialize Möbius confluence operator.
        
        Args:
            size: Field dimensions (nu, nv) - must match substrate
            device: Computation device
        """
        self.size = size
        self.nu, self.nv = size
        self.device = device
        
        # Track confluence statistics
        self.total_steps = 0
        self.total_confluence_magnitude = 0.0
        
        # Verify size is valid for Möbius (nu must be even)
        if self.nu % 2 != 0:
            raise ValueError(f"Möbius manifold requires even nu dimension, got {self.nu}")
    
    def step(
        self,
        A: torch.Tensor,
        enforce_antiperiodicity: bool = True
    ) -> torch.Tensor:
        """
        Perform one confluence time step: P_{t+1} = A_t(u+π, 1-v)
        
        Args:
            A: Current Actual field
            enforce_antiperiodicity: Whether to enforce anti-periodic constraint
        
        Returns:
            P_next: New Potential field for next time step
        """
        # Compute shifts for Möbius transformation
        # u → u+π means shift by half the u-dimension
        u_shift = self.nu // 2
        
        # Apply geometric transformation
        # 1. Shift u by π (half-twist)
        A_shifted = torch.roll(A, shifts=u_shift, dims=0)
        
        # 2. Flip v coordinate (1-v means reverse v-axis)
        A_flipped = torch.flip(A_shifted, dims=[1])
        
        # 3. This is our new Potential
        P_next = A_flipped
        
        # 4. Optionally enforce anti-periodicity
        if enforce_antiperiodicity:
            P_next = self._enforce_antiperiodicity(P_next)
        
        # Track confluence
        self.total_steps += 1
        confluence_magnitude = (P_next - A).abs().mean().item()
        self.total_confluence_magnitude += confluence_magnitude
        
        return P_next
    
    def _enforce_antiperiodicity(self, field: torch.Tensor) -> torch.Tensor:
        """
        Enforce anti-periodic boundary condition: f(u+π, v) = -f(u, 1-v)
        
        Projects field onto the subspace satisfying the constraint.
        
        Args:
            field: Field to enforce constraint on
        
        Returns:
            Field with anti-periodicity enforced
        """
        # Get shifted and flipped version (what f(u+π, 1-v) should equal)
        u_shift = self.nu // 2
        field_shifted = torch.roll(field, shifts=u_shift, dims=0)
        field_twisted = torch.flip(field_shifted, dims=[1])
        
        # Anti-periodic: f(u+π, 1-v) = -f(u,v)
        # Project: f_new = (f - f_twisted) / 2
        field_corrected = (field - field_twisted) / 2.0
        
        return field_corrected
    
    def validate_antiperiodicity(self, field: torch.Tensor) -> float:
        """
        Validate anti-periodic constraint.
        
        Args:
            field: Field to validate
        
        Returns:
            RMS error from perfect anti-periodicity
        """
        u_shift = self.nu // 2
        field_shifted = torch.roll(field, shifts=u_shift, dims=0)
        field_twisted = torch.flip(field_shifted, dims=[1])
        
        # Should have: f(u+π, 1-v) = -f(u,v)
        # Error: ||f(u+π, 1-v) + f(u,v)||
        error = (field_twisted + field).pow(2).mean().sqrt()
        
        return error.item()
    
    def compute_confluence_velocity(
        self,
        P_current: torch.Tensor,
        P_next: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Compute velocity field of Potential evolution.
        
        v_P = (P_next - P_current) / dt
        
        Args:
            P_current: Current Potential
            P_next: Next Potential (from confluence)
            dt: Time interval (default 1 step)
        
        Returns:
            Velocity field
        """
        velocity = (P_next - P_current) / dt
        return velocity
    
    def compute_confluence_divergence(self, velocity: torch.Tensor) -> float:
        """
        Compute divergence of confluence velocity field.
        
        ∇·v = ∂v_u/∂u + ∂v_v/∂v
        
        Args:
            velocity: Velocity field from confluence
        
        Returns:
            Mean absolute divergence
        """
        # Compute partial derivatives via finite differences
        dv_du = velocity[1:, :] - velocity[:-1, :]
        dv_dv = velocity[:, 1:] - velocity[:, :-1]
        
        # Divergence (approximate, interior points)
        # Pad to match original size
        dv_du_padded = F.pad(dv_du, (0, 0, 0, 1))
        dv_dv_padded = F.pad(dv_dv, (0, 1, 0, 0))
        
        divergence = dv_du_padded + dv_dv_padded
        
        return divergence.abs().mean().item()
    
    def get_confluence_state(self) -> dict:
        """
        Get current confluence operator state.
        
        Returns:
            Dictionary with confluence statistics
        """
        avg_magnitude = (
            self.total_confluence_magnitude / self.total_steps 
            if self.total_steps > 0 else 0.0
        )
        
        return {
            'total_steps': self.total_steps,
            'total_confluence_magnitude': self.total_confluence_magnitude,
            'average_confluence_magnitude': avg_magnitude,
            'size': self.size,
            'device': self.device
        }
    
    def __repr__(self) -> str:
        """String representation."""
        avg_magnitude = (
            self.total_confluence_magnitude / self.total_steps 
            if self.total_steps > 0 else 0.0
        )
        
        return (
            f"MobiusConfluence(size={self.size}, device='{self.device}')\n"
            f"  Total steps: {self.total_steps}\n"
            f"  Average confluence magnitude: {avg_magnitude:.6f}"
        )


def create_confluence_operator(
    size: Tuple[int, int],
    device: str = 'cpu'
) -> MobiusConfluence:
    """
    Convenience function to create confluence operator.
    
    Args:
        size: Field dimensions (nu, nv)
        device: Computation device
    
    Returns:
        Configured confluence operator
    """
    return MobiusConfluence(size=size, device=device)
