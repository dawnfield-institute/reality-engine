"""
Symbolic Entropy Collapse (SEC) Operator for Reality Engine

Implements energy functional minimization for field evolution:
    E(A|P,T) = α||A-P||² + β||∇A||² + γ∫T·|A|² 

Where:
- α: Potential-Actual coupling (how strongly A is pulled toward P)
- β: Spatial smoothness (MED - Macro Emergence Dynamics)
- γ: Thermodynamic coupling (information-energy duality via Landauer)

Based on validated PACEngine GeometricSEC but adapted for Möbius topology
with explicit thermodynamic integration.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from substrate.constants import XI, LAMBDA, ALPHA_SEC, BETA_MED


class SymbolicEntropyCollapse:
    """
    SEC operator: Energy functional minimization with thermodynamic coupling.
    
    Evolution via gradient descent on energy functional:
        ∂A/∂t = -∇E(A|P,T) + thermal_noise
    
    Generates heat from:
    - Information erasure (Landauer principle)
    - Collapse events (rapid entropy reduction)
    - Spatial smoothing (dissipation)
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.05,
        gamma: float = 0.01,
        device: str = 'cpu'
    ):
        """
        Initialize SEC operator.
        
        Args:
            alpha: Potential-Actual coupling strength (0.05-0.2 typical)
            beta: Spatial smoothing strength (MED) (0.01-0.1 typical)
            gamma: Thermodynamic coupling strength (0.001-0.01 typical)
            device: Computation device ('cpu' or 'cuda')
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device
        
        # Track thermodynamic quantities
        self.total_heat_generated = 0.0
        self.total_entropy_reduced = 0.0
        self.collapse_event_count = 0
        self.collapse_events = []
    
    def compute_energy(
        self,
        A: torch.Tensor,
        P: torch.Tensor,
        T: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute energy functional components.
        
        E(A|P,T) = α||A-P||² + β||∇A||² + γ∫T·|A|²
        
        Args:
            A: Actual field
            P: Potential field
            T: Temperature field (optional)
        
        Returns:
            Dictionary with energy components
        """
        # Potential-Actual coupling energy
        E_coupling = self.alpha * (A - P).pow(2).sum()
        
        # Spatial smoothness energy (∇A via finite differences)
        grad_x = A[1:, :] - A[:-1, :]
        grad_v = A[:, 1:] - A[:, :-1]
        E_smoothness = self.beta * (grad_x.pow(2).sum() + grad_v.pow(2).sum())
        
        # Thermodynamic energy (field intensity weighted by temperature)
        E_thermal = 0.0
        if T is not None:
            E_thermal = self.gamma * (T * A.pow(2)).sum()
        
        E_total = E_coupling + E_smoothness + E_thermal
        
        return {
            'total': E_total.item(),
            'coupling': E_coupling.item(),
            'smoothness': E_smoothness.item(),
            'thermal': float(E_thermal.item() if isinstance(E_thermal, torch.Tensor) else E_thermal)
        }
    
    def evolve(
        self,
        A: torch.Tensor,
        P: torch.Tensor,
        T: torch.Tensor,
        dt: float = 0.001,
        add_thermal_noise: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        Evolve Actual field via SEC dynamics.
        
        Evolution equation:
            ∂A/∂t = -α(A-P) - β∇²A + thermal_noise
        
        Args:
            A: Current Actual field
            P: Potential field (target)
            T: Temperature field
            dt: Time step
            add_thermal_noise: Whether to add Langevin thermal fluctuations
        
        Returns:
            (A_new, heat_generated)
        """
        # Compute energy gradient components
        
        # 1. Potential-Actual coupling: pulls A toward P
        dA_coupling = -self.alpha * (A - P)
        
        # 2. Spatial smoothing (MED): ∇²A via finite differences
        laplacian = self._compute_laplacian_2d(A)
        dA_smooth = self.beta * laplacian
        
        # 3. Thermal coupling: higher T allows more deviation
        dA_thermal = torch.zeros_like(A)
        if T is not None:
            # Reduce effective coupling in hot regions
            dA_thermal = -self.gamma * T * A
        
        # Total deterministic evolution
        dA_dt = dA_coupling + dA_smooth + dA_thermal
        
        # Forward Euler step
        A_new = A + dt * dA_dt
        
        # Add thermal fluctuations (Langevin dynamics)
        if add_thermal_noise and T is not None:
            noise_amplitude = torch.sqrt(2 * T * dt)
            thermal_noise = noise_amplitude * torch.randn_like(A)
            A_new = A_new + thermal_noise
        
        # Compute heat generated from this step
        heat = self._compute_heat_generation(A, A_new, dt)
        self.total_heat_generated += heat
        
        # Track entropy reduction
        entropy_before = self._compute_field_entropy(A)
        entropy_after = self._compute_field_entropy(A_new)
        entropy_reduced = entropy_before - entropy_after
        self.total_entropy_reduced += entropy_reduced
        
        # Detect collapse events (rapid entropy reduction)
        if entropy_reduced > 0.1:  # Threshold for "collapse"
            self.collapse_event_count += 1
            self.collapse_events.append({
                'time': self.collapse_event_count,
                'entropy_reduced': entropy_reduced,
                'heat_generated': heat
            })
        
        return A_new, heat
    
    def _compute_laplacian_2d(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D Laplacian using finite differences.
        
        ∇²f ≈ (f[i+1,j] + f[i-1,j] + f[i,j+1] + f[i,j-1] - 4f[i,j])
        
        Args:
            field: 2D field tensor
        
        Returns:
            Laplacian of field
        """
        # Pad with zeros for boundary conditions
        padded = F.pad(field, (1, 1, 1, 1), mode='constant', value=0)
        
        # Compute finite differences
        laplacian = (
            padded[2:, 1:-1] +   # i+1, j
            padded[:-2, 1:-1] +  # i-1, j
            padded[1:-1, 2:] +   # i, j+1
            padded[1:-1, :-2] -  # i, j-1
            4 * field            # -4*f[i,j]
        )
        
        return laplacian
    
    def _compute_heat_generation(
        self,
        A_before: torch.Tensor,
        A_after: torch.Tensor,
        dt: float
    ) -> float:
        """
        Compute heat generated from field evolution.
        
        Heat comes from:
        1. Kinetic energy of field motion (normalized per cell)
        2. Entropy reduction (Landauer principle)
        
        Args:
            A_before: Field before evolution
            A_after: Field after evolution
            dt: Time step
        
        Returns:
            Heat generated (total, in units suitable for temperature field)
        """
        n_cells = A_before.numel()
        
        # Kinetic energy: (1/2) * |dA/dt|² per cell (mean, not sum)
        dA = (A_after - A_before) / dt
        kinetic_energy_per_cell = 0.5 * (dA.pow(2).mean().item())
        
        # Landauer heat from entropy reduction
        entropy_before = self._compute_field_entropy(A_before)
        entropy_after = self._compute_field_entropy(A_after)
        entropy_reduced = max(0, entropy_before - entropy_after)
        
        # Landauer: E = kT ln(2) per bit
        # Normalize by number of cells to get heat per cell
        landauer_heat_per_cell = (entropy_reduced * np.log(2)) / n_cells
        
        # Total heat per cell (will be scaled appropriately when added to T)
        total_heat_per_cell = kinetic_energy_per_cell + landauer_heat_per_cell
        
        # Scale to reasonable magnitude: typical fields have |A| ~ 0.1-1
        # Kinetic energy per cell ~ 0.5 * (dA/dt)² ~ 0.5 * (0.1/0.1)² ~ 0.5
        # We want total heat generation to be modest (< 10 for stability)
        # Current: ~1200 total, we want ~ 1-10 total
        scale_factor = 0.01  # Reduce by 100x
        
        return total_heat_per_cell * n_cells * scale_factor
    
    def _compute_field_entropy(self, field: torch.Tensor, bins: int = 100) -> float:
        """
        Compute Shannon entropy of field distribution.
        
        H = -Σ p(x) log p(x)
        
        Args:
            field: Field to analyze
            bins: Number of histogram bins
        
        Returns:
            Shannon entropy in nats
        """
        # Handle edge cases
        if torch.any(~torch.isfinite(field)):
            # Field has inf/nan - return high entropy as fallback
            return float(bins)  # Maximum possible entropy
        
        field_min = field.min().item()
        field_max = field.max().item()
        
        # If field is constant, entropy is zero
        if abs(field_max - field_min) < 1e-10:
            return 0.0
        
        # Create histogram
        hist = torch.histc(field.flatten(), bins=bins, min=field_min, max=field_max)
        
        # Normalize to probabilities
        prob = hist / hist.sum()
        
        # Compute entropy (avoid log(0))
        prob = prob[prob > 0]
        entropy = -(prob * torch.log(prob)).sum()
        
        return entropy.item()
    
    def detect_collapse_regions(
        self,
        A: torch.Tensor,
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Detect regions where field is collapsing (high local gradient).
        
        Args:
            A: Actual field
            threshold: Gradient magnitude threshold
        
        Returns:
            Binary mask of collapse regions
        """
        # Compute gradient magnitude
        grad_x = A[1:, :] - A[:-1, :]
        grad_v = A[:, 1:] - A[:, :-1]
        
        # Pad to match original size
        grad_x = F.pad(grad_x, (0, 0, 0, 1))
        grad_v = F.pad(grad_v, (0, 1, 0, 0))
        
        grad_magnitude = torch.sqrt(grad_x.pow(2) + grad_v.pow(2))
        
        # Regions above threshold are "collapsing"
        collapse_mask = (grad_magnitude > threshold).float()
        
        return collapse_mask
    
    def get_sec_state(self) -> Dict:
        """
        Get current SEC operator state.
        
        Returns:
            Dictionary with SEC statistics
        """
        return {
            'total_heat_generated': self.total_heat_generated,
            'total_entropy_reduced': self.total_entropy_reduced,
            'collapse_event_count': self.collapse_event_count,
            'parameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma
            }
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SymbolicEntropyCollapse(device='{self.device}')\n"
            f"  Parameters: α={self.alpha:.4f}, β={self.beta:.4f}, γ={self.gamma:.4f}\n"
            f"  Heat generated: {self.total_heat_generated:.6f}\n"
            f"  Entropy reduced: {self.total_entropy_reduced:.6f}\n"
            f"  Collapse events: {self.collapse_event_count}"
        )


def create_sec_operator(
    coupling_strength: float = 0.1,
    smoothness_strength: float = 0.05,
    thermal_strength: float = 0.01,
    device: str = 'cpu'
) -> SymbolicEntropyCollapse:
    """
    Convenience function to create SEC operator with standard parameters.
    
    Args:
        coupling_strength: Alpha parameter (Potential-Actual coupling)
        smoothness_strength: Beta parameter (spatial smoothing/MED)
        thermal_strength: Gamma parameter (thermodynamic coupling)
        device: Computation device
    
    Returns:
        Configured SEC operator
    """
    return SymbolicEntropyCollapse(
        alpha=coupling_strength,
        beta=smoothness_strength,
        gamma=thermal_strength,
        device=device
    )
