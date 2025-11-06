"""
Field State Types

Defines the fundamental fields with thermodynamic-information duality:
- P (Potential): Energy-like field + thermal energy, "what could be"
- A (Actual): Information-like field + structural entropy, "what is"
- M (Memory): Matter-like field, "what persists"
- T (Temperature): Local thermal energy field (emerges from variance)

Key Insight: Fields carry BOTH information content AND thermal energy.
This prevents the "cold universe" problem of pure information theory.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class FieldState:
    """
    Container for the fundamental fields on Möbius substrate
    
    Maps to Dawn Field Theory with thermodynamic extension:
    - P ≈ E + thermal_energy (potential + heat)
    - A ≈ I + structural_entropy (information + disorder)
    - M = M (memory/matter - accumulated structure)
    - T = f(variance(A)) (temperature emerges from field dynamics)
    
    Thermodynamic Coupling:
    - Information collapse generates heat (SEC → entropy production)
    - Temperature gradients drive information flow
    - Landauer principle: Information erasure costs k_T ln(2) per bit
    - Thermal fluctuations prevent "freezing" into static patterns
    """
    potential: torch.Tensor  # P field (energy + thermal)
    actual: torch.Tensor     # A field (information + entropy)
    memory: torch.Tensor     # M field (matter/structure)
    temperature: Optional[torch.Tensor] = None  # T field (emerges from variance)
    time: float = 0.0
    step: int = 0
    
    def __post_init__(self):
        """Validate field shapes match"""
        assert self.potential.shape == self.actual.shape == self.memory.shape, \
            "P, A, M fields must have same shape"
        
        # Initialize temperature if not provided
        if self.temperature is None:
            self.temperature = self._compute_initial_temperature()
        else:
            assert self.temperature.shape == self.potential.shape, \
                "Temperature field must match P, A, M shape"
    
    def _compute_initial_temperature(self) -> torch.Tensor:
        """
        Initialize temperature from field variance (equipartition theorem)
        Higher variance = higher local temperature
        """
        variance = torch.var(self.actual, dim=0, keepdim=True)
        # Expand to match field shape
        temp = torch.sqrt(variance + 1e-8)  # Prevent zero temperature
        if temp.shape != self.potential.shape:
            temp = temp.expand_as(self.potential)
        return temp
    
    @property
    def P(self):
        """Alias for potential (backward compatibility)"""
        return self.potential
    
    @property
    def A(self):
        """Alias for actual (backward compatibility)"""
        return self.actual
    
    @property
    def M(self):
        """Alias for memory (backward compatibility)"""
        return self.memory
    
    @property
    def T(self):
        """Temperature field"""
        return self.temperature
    
    @property
    def shape(self):
        """Get field shape"""
        return self.potential.shape
    
    @property
    def device(self):
        """Get tensor device"""
        return self.potential.device
    
    def to(self, device):
        """Move all fields to device"""
        return FieldState(
            potential=self.potential.to(device),
            actual=self.actual.to(device),
            memory=self.memory.to(device),
            temperature=self.temperature.to(device) if self.temperature is not None else None,
            time=self.time,
            step=self.step
        )
    
    def clone(self):
        """Deep copy of field state"""
        return FieldState(
            potential=self.potential.clone(),
            actual=self.actual.clone(),
            memory=self.memory.clone(),
            temperature=self.temperature.clone() if self.temperature is not None else None,
            time=self.time,
            step=self.step
        )
    
    def total_pac(self) -> float:
        """Total PAC quantity (should be conserved!)"""
        return (self.potential.sum() + self.actual.sum() + self.memory.sum()).item()
    
    def energy(self) -> float:
        """Total energy (potential field + thermal)"""
        total_energy = self.potential.sum().item()
        if self.temperature is not None:
            thermal_energy = self.temperature.sum().item()
            total_energy += thermal_energy
        return total_energy
    
    def information(self) -> float:
        """Total information (actual field)"""
        return self.actual.sum().item()
    
    def matter(self) -> float:
        """Total matter (memory field)"""
        return self.memory.sum().item()
    
    def thermal_energy(self) -> float:
        """Total thermal energy"""
        if self.temperature is not None:
            return self.temperature.sum().item()
        return 0.0
    
    def entropy(self) -> float:
        """
        Total entropy (Shannon entropy of actual field + thermal entropy)
        S = -Σ p*log(p) + k*T
        """
        # Structural entropy from information
        # Normalize to make it a probability distribution
        A_positive = torch.abs(self.actual) + 1e-10
        A_norm = A_positive / A_positive.sum()
        structural_entropy = -torch.sum(A_norm * torch.log(A_norm + 1e-10)).item()
        
        # Thermal entropy (simplified: just sum of temperature)
        if self.temperature is not None:
            thermal_entropy = torch.abs(self.temperature).sum().item()
            return structural_entropy + thermal_entropy
        
        return structural_entropy
    
    def free_energy(self) -> float:
        """
        Free energy: F = E - TS
        This is what the system minimizes (drives evolution toward equilibrium)
        """
        E = self.energy()
        S = self.entropy()
        return E - S
    
    def disequilibrium(self) -> float:
        """
        Measure of how far from equilibrium
        Higher values = more pressure to equilibrate = faster time
        """
        return torch.abs(self.potential - self.actual).sum().item()
    
    def thermal_variance(self) -> float:
        """
        Variance of temperature field
        Low variance = approaching heat death (uniform temperature)
        """
        if self.temperature is not None:
            return torch.var(self.temperature).item()
        return 0.0
