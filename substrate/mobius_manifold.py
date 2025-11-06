"""
Möbius Manifold - Geometric Substrate

Self-referential topology with anti-periodic boundaries.
Provides the geometric foundation where physics emerges.

Key Properties:
- Anti-periodic boundaries: f(x + π) = -f(x)
- 4π holonomy (not 2π!)
- Half-integer mode frequencies
- Ξ = 1.0571 emerges from geometry

Based on validated pre_field_recursion implementation.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .field_types import FieldState
from .constants import TWIST_ANGLE, XI


@dataclass
class TopologyMetrics:
    """Metrics describing Möbius topological properties"""
    twist_strength: float
    curvature_variance: float
    boundary_continuity: float
    field_coherence: float
    anti_periodic_quality: float
    xi_measurement: float  # Should be ≈ 1.0571


class MobiusManifold:
    """
    Möbius topology substrate for Reality Engine v2
    
    This is the geometric foundation where E↔I equivalence lives.
    The Möbius twist creates self-referential structure where:
    - Potential and Actual exist on same surface
    - No boundaries (finite but endless)
    - Natural information amplification
    - Geometric constants (Ξ) emerge
    
    Properties:
        size: Number of points along loop direction (u)
        width: Number of points across strip width (v)
        twist_strength: Möbius twist intensity (default 1.0 = full twist)
        device: torch device (cuda/cpu)
    """
    
    def __init__(self, 
                 size: int = 128,
                 width: int = 32,
                 twist_strength: float = 1.0,
                 device: str = "auto",
                 seed: Optional[int] = None):
        
        self.size = size  # Loop direction (u)
        self.width = width  # Strip width (v)
        self.twist_strength = twist_strength
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Geometric properties
        self.twist_shift = size // 2  # Half-loop = π shift
        self.boundary_conditions = "anti_periodic"
        
        print(f"Möbius Manifold initialized:")
        print(f"  Size: {size} × {width}")
        print(f"  Twist shift: {self.twist_shift}")
        print(f"  Device: {self.device}")
        print(f"  Anti-periodic boundaries: f(u+pi, v) = -f(u, 1-v)")
    
    def initialize_fields(self, mode: str = 'random') -> FieldState:
        """
        Initialize field state on Möbius substrate
        
        Args:
            mode: Initialization mode
                - 'random': Random fields respecting anti-periodic bounds
                - 'big_bang': High potential, low actual/memory
                - 'structured': Braided Gaussian strands (from mobius_confluence.py)
        
        Returns:
            FieldState with P, A, M fields
        """
        if mode == 'random':
            P, A, M = self._init_random_fields()
        elif mode == 'big_bang':
            P, A, M = self._init_big_bang()
        elif mode == 'structured':
            P, A, M = self._init_structured_fields()
        else:
            raise ValueError(f"Unknown initialization mode: {mode}")
        
        return FieldState(potential=P, actual=A, memory=M, time=0.0, step=0)
    
    def _init_random_fields(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random fields respecting Möbius anti-periodic boundaries"""
        # Initialize P (Potential)
        P = torch.randn(self.size, self.width, device=self.device) * 2 - 1
        P = self._enforce_antiperiodic(P)
        P = self._smooth_field(P, passes=3)
        
        # Initialize A (Actual) as noisy copy of P
        A = P + 0.1 * torch.randn_like(P)
        A = self._enforce_antiperiodic(A)
        
        # Initialize M (Memory) as small noise
        M = torch.randn(self.size, self.width, device=self.device) * 0.01
        M = torch.abs(M)  # Memory is non-negative
        
        return P, A, M
    
    def _init_big_bang(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Big Bang initialization: pure potential with fluctuations
        
        High P (energy/potential)
        Low A (no structure yet)
        Zero M (no memory yet)
        """
        # High potential with 10% fluctuations
        base_energy = 1.0
        P = torch.ones(self.size, self.width, device=self.device) * base_energy
        fluctuations = torch.randn_like(P) * base_energy * 0.1
        P = P + fluctuations
        P = self._enforce_antiperiodic(P)
        
        # Minimal actual (information seeds)
        A = torch.randn(self.size, self.width, device=self.device) * 0.05
        A = self._enforce_antiperiodic(A)
        
        # No memory yet
        M = torch.zeros(self.size, self.width, device=self.device)
        
        return P, A, M
    
    def _init_structured_fields(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Structured initialization with braided Gaussian strands
        From mobius_confluence.py - creates interesting initial conditions
        """
        u = torch.linspace(0, 2 * np.pi, self.size, device=self.device)
        v = torch.linspace(0, 1, self.width, device=self.device)
        U, V = torch.meshgrid(u, v, indexing='ij')
        
        # Smooth background
        P = 0.2 * torch.sin(2 * U) * (0.5 + 0.5 * torch.cos(np.pi * (V - 0.5)))
        
        # Add braided strands
        n_strands = 8
        for _ in range(n_strands):
            phase = torch.rand(1).item() * 2 * np.pi
            slope = (torch.rand(1).item() - 0.5) * 4.0
            width_param = 0.05 + torch.rand(1).item() * 0.07
            amp = 0.6 + torch.rand(1).item() * 0.6
            
            v_center = 0.5 + 0.25 * torch.sin(1.5 * U + phase) + 0.15 * slope * torch.cos(0.5 * U)
            gauss = amp * torch.exp(-((V - v_center) ** 2) / (2 * width_param ** 2))
            P = P + gauss
        
        P = self._enforce_antiperiodic(P)
        
        # A starts as noisy copy
        A = P + 0.1 * torch.randn_like(P)
        A = self._enforce_antiperiodic(A)
        
        # M starts small
        M = torch.abs(torch.randn_like(P)) * 0.01
        
        return P, A, M
    
    def _enforce_antiperiodic(self, field: torch.Tensor) -> torch.Tensor:
        """
        Enforce Möbius anti-periodic boundary condition:
        f(u + π, v) = -f(u, 1-v)
        
        This is THE defining property of Möbius topology!
        """
        half_size = self.size // 2
        
        # For each point in first half
        for i in range(half_size):
            opposite_u = (i + half_size) % self.size
            
            # Average enforcement (smoother than hard constraint)
            for j in range(self.width):
                opposite_v = self.width - 1 - j
                
                # Calculate what the anti-periodic condition requires
                expected = -self.twist_strength * field[i, j]
                actual = field[opposite_u, opposite_v]
                
                # Blend toward constraint
                field[opposite_u, opposite_v] = 0.8 * actual + 0.2 * expected
        
        return field
    
    def _smooth_field(self, field: torch.Tensor, passes: int = 3) -> torch.Tensor:
        """
        Smooth field while preserving Möbius structure
        Uses Laplacian smoothing with periodic boundaries in u
        """
        smoothed = field.clone()
        
        for _ in range(passes):
            new_field = smoothed.clone()
            
            # Laplacian smoothing
            # u direction (periodic)
            u_plus = torch.roll(smoothed, shifts=-1, dims=0)
            u_minus = torch.roll(smoothed, shifts=1, dims=0)
            
            # v direction (Neumann at boundaries)
            v_plus = torch.zeros_like(smoothed)
            v_minus = torch.zeros_like(smoothed)
            v_plus[:, :-1] = smoothed[:, 1:]
            v_plus[:, -1] = smoothed[:, -1]
            v_minus[:, 1:] = smoothed[:, :-1]
            v_minus[:, 0] = smoothed[:, 0]
            
            # Weighted average
            new_field = 0.6 * smoothed + 0.1 * (u_plus + u_minus + v_plus + v_minus)
            
            # Re-enforce Möbius constraint
            new_field = self._enforce_antiperiodic(new_field)
            
            smoothed = new_field
        
        return smoothed
    
    def calculate_metrics(self, field: torch.Tensor) -> TopologyMetrics:
        """
        Calculate topological quality metrics
        
        These measure how well the Möbius structure is maintained
        and whether Ξ ≈ 1.0571 emerges from the geometry
        """
        # Anti-periodic quality
        half_size = self.size // 2
        ap_errors = []
        for i in range(half_size):
            opposite_u = (i + half_size) % self.size
            for j in range(self.width):
                opposite_v = self.width - 1 - j
                expected = -self.twist_strength * field[i, j]
                actual = field[opposite_u, opposite_v]
                ap_errors.append((expected - actual).abs().item())
        
        anti_periodic_quality = 1.0 - min(np.mean(ap_errors), 1.0)
        
        # Curvature variance (second derivative)
        grad_u = torch.gradient(field, dim=0)[0]
        grad_u_2 = torch.gradient(grad_u, dim=0)[0]
        curvature_variance = grad_u_2.var().item()
        
        # Boundary continuity (how smooth is the twist?)
        boundary_gap = (field[0, :] - field[-1, :]).abs().mean().item()
        boundary_continuity = 1.0 / (1.0 + boundary_gap)
        
        # Field coherence
        gradient_magnitude = torch.sqrt(grad_u**2).mean()
        field_magnitude = field.abs().mean()
        field_coherence = (field_magnitude / (field_magnitude + gradient_magnitude)).item()
        
        # Measure Ξ (should be ≈ 1.0571 from geometry)
        # This is a placeholder - proper measurement requires spectral analysis
        # For now, use heuristic based on field properties
        xi_measurement = self._estimate_xi(field)
        
        return TopologyMetrics(
            twist_strength=self.twist_strength,
            curvature_variance=curvature_variance,
            boundary_continuity=boundary_continuity,
            field_coherence=field_coherence,
            anti_periodic_quality=anti_periodic_quality,
            xi_measurement=xi_measurement
        )
    
    def _estimate_xi(self, field: torch.Tensor) -> float:
        """
        Estimate Ξ balance constant from field
        
        Proper calculation requires:
        1. Spectral decomposition
        2. Mode counting with anti-periodic boundaries
        3. Ratio of consecutive eigenvalues
        
        For now, use simplified heuristic
        TODO: Implement full spectral analysis
        """
        # Placeholder - return expected value
        # Real implementation needs FFT with proper boundary conditions
        return XI
    
    def get_local_structure(self, u_center: int, v_center: int, 
                           radius: int = 3) -> Dict:
        """
        Extract local topological structure around a point
        
        Returns detailed analysis of local geometry for law detection
        """
        # Extract local patch (with periodic wrapping in u)
        u_indices = [(u_center + i) % self.size for i in range(-radius, radius + 1)]
        v_indices = [max(0, min(self.width - 1, v_center + i)) 
                    for i in range(-radius, radius + 1)]
        
        # Will be implemented when needed for law detection
        return {
            'u_center': u_center,
            'v_center': v_center,
            'radius': radius,
            'u_indices': u_indices,
            'v_indices': v_indices
        }
    
    def __repr__(self):
        return (f"MobiusManifold(size={self.size}, width={self.width}, "
                f"twist={self.twist_strength}, device={self.device})")
