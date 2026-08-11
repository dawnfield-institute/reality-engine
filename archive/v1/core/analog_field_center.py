"""
Analog Field Center - Continuous field representation at specific scale

This implements the core concept of the analog tensor architecture:
- Continuous field equations (not discrete grids)
- Compute values on-demand (not stored everywhere)
- Scale-specific physics
- Sparse storage for active regions only

Each center operates at a characteristic scale (quantum, atomic, stellar, etc.)
and uses appropriate field equations for that scale.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import sys

ENGINE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ENGINE_ROOT))


@dataclass
class FieldEquationConfig:
    """Configuration for field equations at specific scale"""
    scale: float  # Characteristic length scale in meters
    field_type: str  # 'quantum', 'atomic', 'molecular', 'classical', 'stellar'
    coupling_strength: float = 1.0
    diffusion_coefficient: float = 0.1
    nonlinearity: float = 0.0


class AnalogFieldCenter:
    """
    Continuous field representation at a specific characteristic scale.
    
    Unlike discrete tensors that store values at every grid point, this
    center computes field values on-demand using continuous equations.
    Only "active regions" (where fields are non-negligible) are stored.
    
    Key features:
    - Continuous field equations (not discretized)
    - Sparse storage (only active regions)
    - Scale-appropriate physics
    - GPU-accelerated computation
    """
    
    def __init__(
        self,
        characteristic_scale: float,
        bounds: Tuple[float, float, float, float] = None,
        device: str = 'cpu',
        resolution_factor: int = 4,
        parent_center: Optional['AnalogFieldCenter'] = None
    ):
        """
        Initialize analog field center.
        
        Args:
            characteristic_scale: Length scale in meters (1e-15 for quantum, 1e9 for stellar)
            bounds: (x_min, x_max, y_min, y_max) spatial bounds in scale units
            device: 'cpu' or 'cuda'
            resolution_factor: How much finer than parent (for hierarchical refinement)
            parent_center: Parent scale center (for boundary coupling)
        """
        self.scale = characteristic_scale
        self.device = device
        self.resolution_factor = resolution_factor
        self.parent = parent_center
        
        # Default bounds if not specified
        if bounds is None:
            # Create small region around origin
            extent = 10.0  # ±10 scale units
            self.bounds = (-extent, extent, -extent, extent)
        else:
            self.bounds = bounds
        
        # Determine field equation type from scale
        self.field_type = self._classify_scale(characteristic_scale)
        
        # Sparse storage for computed field values
        # Only store where fields are "active" (non-negligible)
        self.active_regions = {}  # Dict: position_hash -> field_values
        self.active_threshold = 1e-6  # Below this, don't store
        
        # Field state (for regions we've computed)
        self.computed_fields = {
            'potential': {},  # Information field (I)
            'actual': {},     # Energy field (E)
            'memory': {},     # Memory field (M)
            'temperature': {}  # Temperature field (T)
        }
        
        # Statistics
        self.num_active_points = 0
        self.total_computations = 0
        self.spawned_from_herniation = False
        
        # Lifecycle tracking
        self.age = 0.0  # Time since creation (seconds)
        self.total_energy = 1.0  # Track total energy in center
        self.energy_decay_rate = 0.01  # 1% energy decay per second (faster turnover)
        
        # Center position (if spawned from herniation)
        self.center_position = None
        
        print(f"📐 Created AnalogFieldCenter:")
        print(f"   Scale: {self.scale:.2e} m ({self.field_type})")
        print(f"   Bounds: {self.bounds}")
        print(f"   Device: {self.device}")
        
    def _classify_scale(self, scale: float) -> str:
        """Determine physics regime from characteristic scale"""
        if scale < 1e-12:
            return 'quantum'
        elif scale < 1e-9:
            return 'atomic'
        elif scale < 1e-6:
            return 'molecular'
        elif scale < 1e3:
            return 'classical'
        else:
            return 'stellar'
    
    def compute_field_at_position(
        self,
        position: torch.Tensor,
        field_name: str = 'potential'
    ) -> torch.Tensor:
        """
        Compute field value at position using continuous field equation.
        
        This is the KEY difference from discrete tensors:
        Instead of looking up tensor[i,j], we COMPUTE the value
        from the field equation.
        
        Args:
            position: (x, y) coordinates in scale units
            field_name: Which field to compute ('potential', 'actual', etc.)
            
        Returns:
            Field value at that position
        """
        self.total_computations += 1
        
        # Check if already computed and stored
        pos_hash = self._hash_position(position)
        if pos_hash in self.computed_fields[field_name]:
            return self.computed_fields[field_name][pos_hash]
        
        # Otherwise, compute from field equation
        if self.field_type == 'quantum':
            value = self._quantum_field_equation(position, field_name)
        elif self.field_type == 'atomic':
            value = self._atomic_field_equation(position, field_name)
        elif self.field_type == 'molecular':
            value = self._molecular_field_equation(position, field_name)
        else:  # classical/stellar
            value = self._classical_field_equation(position, field_name)
        
        # Store if significant (sparse storage)
        if abs(value) > self.active_threshold:
            self.computed_fields[field_name][pos_hash] = value
            self.num_active_points += 1
        
        return value
    
    def _quantum_field_equation(
        self,
        position: torch.Tensor,
        field_name: str
    ) -> torch.Tensor:
        """
        Quantum field equation (simplified Schrödinger-like).
        
        For quantum scales, fields follow wave mechanics:
        ψ(r) = exp(-r²/2σ²) * cos(kr)
        """
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=torch.float64, device=self.device)
        
        r = torch.norm(position)
        
        # Wave packet: Gaussian envelope × oscillation
        sigma = self.scale * 2.0  # Spread
        k = 1.0 / self.scale  # Wave number
        
        # Quantum wavefunction
        gaussian = torch.exp(-r**2 / (2 * sigma**2))
        oscillation = torch.cos(k * r)
        
        psi = gaussian * oscillation
        
        # Different fields have different quantum behavior
        if field_name == 'potential':
            return psi
        elif field_name == 'actual':
            return psi**2  # Probability density
        elif field_name == 'memory':
            return gaussian * 0.5  # Slower decay
        else:  # temperature
            return gaussian * abs(oscillation)
    
    def _atomic_field_equation(
        self,
        position: torch.Tensor,
        field_name: str
    ) -> torch.Tensor:
        """
        Atomic field equation (orbital-like).
        
        For atomic scales, use hydrogen orbital approximation:
        ψ(r) ∝ r^l * exp(-r/a₀) * Y_lm(θ,φ)
        """
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=torch.float64, device=self.device)
        
        r = torch.norm(position)
        
        # Bohr radius scaling
        a0 = self.scale
        
        # Simplified 1s orbital
        value = (r / a0) * torch.exp(-r / a0)
        
        if field_name == 'potential':
            return value
        elif field_name == 'actual':
            return value**2
        else:
            return value * 0.5
    
    def _molecular_field_equation(
        self,
        position: torch.Tensor,
        field_name: str
    ) -> torch.Tensor:
        """
        Molecular field equation (bond-like).
        
        For molecular scales, use Morse potential approximation.
        """
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=torch.float64, device=self.device)
        
        r = torch.norm(position)
        
        # Morse potential parameters
        r_eq = self.scale  # Equilibrium distance
        alpha = 1.0 / self.scale  # Width parameter
        
        value = torch.exp(-alpha * (r - r_eq)) * (1 - torch.exp(-alpha * (r - r_eq)))
        
        if field_name == 'actual':
            return abs(value)
        else:
            return value
    
    def _classical_field_equation(
        self,
        position: torch.Tensor,
        field_name: str
    ) -> torch.Tensor:
        """
        Classical field equation (harmonic oscillator).
        
        For classical/stellar scales, use potential field.
        """
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=torch.float64, device=self.device)
        
        r = torch.norm(position)
        
        # Harmonic potential
        value = torch.exp(-r**2 / (2 * self.scale**2))
        
        return value
    
    def _hash_position(self, position: torch.Tensor) -> tuple:
        """Hash position for sparse storage lookup"""
        if isinstance(position, torch.Tensor):
            pos = position.cpu().numpy()
        else:
            pos = np.array(position)
        
        # Round to reasonable precision
        precision = int(-np.log10(self.scale)) + 2
        return tuple(np.round(pos, precision))
    
    def evolve(self, dt: float):
        """
        Evolve fields forward in time using scale-appropriate equations.
        
        Args:
            dt: Timestep in seconds
        """
        # Update lifecycle
        self.age += dt
        self.total_energy *= (1 - self.energy_decay_rate * dt)
        
        if self.field_type == 'quantum':
            self._evolve_quantum(dt)
        elif self.field_type in ['atomic', 'molecular']:
            self._evolve_atomic(dt)
        else:
            self._evolve_classical(dt)
    
    def _evolve_quantum(self, dt: float):
        """
        Evolve quantum fields using Schrödinger equation.
        
        ∂ψ/∂t = -iℏ∇²ψ/2m
        """
        # Simplified: Phase evolution
        for pos_hash, value in list(self.computed_fields['potential'].items()):
            # Quantum phase accumulation
            phase = dt / self.scale  # ℏ/E timescale
            
            # Rotate phase
            if isinstance(value, torch.Tensor):
                # Complex phase rotation (approximate)
                self.computed_fields['potential'][pos_hash] = value * np.cos(phase)
    
    def _evolve_atomic(self, dt: float):
        """Evolve atomic/molecular fields with slower dynamics"""
        # Atomic timescale ~ 1e-15 s
        # Fields decay slowly
        decay = np.exp(-dt / (self.scale * 1e5))
        
        for field_name in self.computed_fields:
            for pos_hash in list(self.computed_fields[field_name].keys()):
                self.computed_fields[field_name][pos_hash] *= decay
    
    def _evolve_classical(self, dt: float):
        """Evolve classical fields with diffusion"""
        # Simple diffusion
        diffusion = 0.01
        
        for field_name in self.computed_fields:
            for pos_hash in list(self.computed_fields[field_name].keys()):
                self.computed_fields[field_name][pos_hash] *= (1 - diffusion * dt)
    
    def couple_to_parent(self, xi_operator: float = 1.0571):
        """
        Exchange conserved quantities with parent scale via PAC.
        
        Args:
            xi_operator: PAC balance constant
        """
        if self.parent is None:
            return
        
        # Extract boundary values from this center
        boundary_energy = self._get_boundary_energy()
        
        # Transform via Xi operator (scale-dependent)
        scale_ratio = self.parent.scale / self.scale
        transformed_energy = boundary_energy * xi_operator * np.log(scale_ratio + 1)
        
        # Inject into parent as source term
        if self.center_position is not None:
            self.parent.inject_energy_at_position(
                self.center_position,
                transformed_energy
            )
    
    def _get_boundary_energy(self) -> float:
        """Calculate total energy at boundary of this center"""
        total = 0.0
        for pos_hash, value in self.computed_fields['actual'].items():
            if isinstance(value, torch.Tensor):
                total += value.item()
            else:
                total += float(value)
        return total
    
    def inject_energy_at_position(self, position: Tuple[float, float], energy: float):
        """Inject energy from child scale into this center"""
        pos_hash = self._hash_position(torch.tensor(position))
        
        # Add to actual field
        if pos_hash in self.computed_fields['actual']:
            self.computed_fields['actual'][pos_hash] += energy
        else:
            self.computed_fields['actual'][pos_hash] = energy
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about this field center"""
        return {
            'scale': self.scale,
            'field_type': self.field_type,
            'num_active_points': self.num_active_points,
            'total_computations': self.total_computations,
            'bounds': self.bounds,
            'spawned_from_herniation': self.spawned_from_herniation,
            'memory_mb': self._estimate_memory_usage(),
            'age': self.age,
            'total_energy': self.total_energy,
            'is_alive': self.total_energy > 0.01  # Dead if < 1% energy
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        # Each stored value is ~64 bytes (position hash + 4 fields × 8 bytes)
        bytes_per_point = 64
        total_bytes = self.num_active_points * bytes_per_point
        return total_bytes / (1024 * 1024)
    
    def clear_inactive_regions(self):
        """Remove stored values below activity threshold (garbage collection)"""
        removed = 0
        
        for field_name in self.computed_fields:
            to_remove = []
            for pos_hash, value in self.computed_fields[field_name].items():
                if abs(value) < self.active_threshold:
                    to_remove.append(pos_hash)
            
            for pos_hash in to_remove:
                del self.computed_fields[field_name][pos_hash]
                removed += 1
        
        self.num_active_points -= removed
        
        if removed > 0:
            print(f"  🗑️  Cleared {removed} inactive points from {self.field_type} center")


def create_quantum_center_at_herniation(
    herniation_position: Tuple[float, float],
    herniation_intensity: float,
    parent_center: Optional[AnalogFieldCenter] = None,
    device: str = 'cpu'
) -> AnalogFieldCenter:
    """
    Spawn a quantum-scale field center at herniation site.
    
    This is the dynamic scale spawning mechanism: when a herniation
    is intense enough, create a finer-scale center to resolve it.
    
    Args:
        herniation_position: (x, y) in parent scale units
        herniation_intensity: Strength of herniation
        parent_center: Parent scale center
        device: Compute device
        
    Returns:
        New AnalogFieldCenter at quantum scale
    """
    # Scale proportional to inverse intensity (stronger = smaller)
    quantum_scale = 1e-12 / (1.0 + herniation_intensity)
    
    # Bounds around herniation site (±10 scale units)
    extent = 10.0 * quantum_scale
    x, y = herniation_position
    bounds = (x - extent, x + extent, y - extent, y + extent)
    
    # Create quantum center
    center = AnalogFieldCenter(
        characteristic_scale=quantum_scale,
        bounds=bounds,
        device=device,
        resolution_factor=4,
        parent_center=parent_center
    )
    
    center.spawned_from_herniation = True
    center.center_position = herniation_position
    
    print(f"🌀 Spawned quantum center at herniation site")
    print(f"   Position: {herniation_position}")
    print(f"   Intensity: {herniation_intensity:.3f}")
    print(f"   Quantum scale: {quantum_scale:.2e} m")
    
    return center
