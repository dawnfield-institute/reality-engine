"""
Internal Rearrangement Tensor - The Core of PAC Conservation

This is the KEY implementation that distinguishes PAC from standard physics:
    Total P + A + M = constant (only internal transfers allowed)

The universe doesn't expand - fields redistribute internally.

This resolves the fundamental issue with discrete field updates:
    - Standard: Each field can grow/shrink independently
    - PAC: Zero-sum constraint - one field's gain is another's loss

The Rearrangement Tensor R^μν tracks field-to-field transfers:
    dP/dt = R^01 - R^02  (potential gains from active, loses to material)
    dA/dt = R^02 - R^01  (active gains from material, loses to potential)
    dM/dt = R^12 - R^21  (material exchanges with both)

Constraint: All R^μν terms sum to zero.

Reference:
- dawn-field-theory/foundational/docs/preprints/drafts/[pac][D][v1.0][C5][I5][E]_pac_necessity_proof.md
- "Theorem 2: Rearrangement vs Growth"
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


# Fundamental constants
PHI = (1 + np.sqrt(5)) / 2
XI = 1.0571
MASS_SQUARED = (XI - 1) / XI


class FieldType(Enum):
    """The three fundamental PAC fields."""
    POTENTIAL = 0  # P - stored/latent energy
    ACTIVE = 1     # A - kinetic/dynamic energy  
    MATERIAL = 2   # M - mass/structure energy


@dataclass
class TransferEvent:
    """Record of a single field transfer."""
    source: FieldType
    target: FieldType
    amount: float
    location: Tuple[int, ...]  # Grid position
    timestamp: int


@dataclass
class ConservationMetrics:
    """Metrics for conservation tracking."""
    total_pac: float  # P + A + M
    initial_total: float
    drift: float  # Fractional change
    p_fraction: float
    a_fraction: float
    m_fraction: float
    transfers_this_step: int
    cumulative_transfers: float


class RearrangementTensor:
    """
    Implements zero-sum field redistribution.
    
    Core principle: The universe is a closed system.
    Total P + A + M is fixed. Fields can only exchange,
    never grow or shrink in aggregate.
    
    This is the "internal rearrangement rather than expansion"
    that makes PAC cosmologically distinct from ΛCDM.
    
    The tensor R^μν has 6 independent components:
        R^01: P → A transfer rate
        R^10: A → P transfer rate  
        R^02: P → M transfer rate
        R^12: A → M transfer rate
        R^20: M → P transfer rate
        R^21: M → A transfer rate
        
    Constraint: Net flow = 0 (∑R = 0)
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        initial_total: float = 1.0,
        device: str = 'cpu'
    ):
        """
        Initialize rearrangement tensor.
        
        Args:
            shape: Spatial grid shape
            initial_total: Total P+A+M (conserved forever)
            device: Compute device
        """
        self.shape = shape
        self.device = device
        self.initial_total = initial_total
        
        # Initialize the three fields - equal distribution
        third = initial_total / 3.0
        self.P = torch.ones(*shape, device=device) * third / np.prod(shape)
        self.A = torch.ones(*shape, device=device) * third / np.prod(shape)
        self.M = torch.ones(*shape, device=device) * third / np.prod(shape)
        
        # Transfer history
        self.transfer_log: List[TransferEvent] = []
        self.cumulative_transferred = 0.0
        self.step = 0
        
        # The 6 independent tensor components (as rate fields)
        # R[i,j] = transfer rate from field i to field j
        # Initialized to zero (no spontaneous transfer)
        self.R = torch.zeros(3, 3, *shape, device=device)
        
        print(f"🔄 Rearrangement Tensor initialized:")
        print(f"   Shape: {shape}")
        print(f"   Initial P+A+M = {self._compute_total():.6f}")
        print(f"   Constraint: This total is FOREVER FIXED")
    
    def _compute_total(self) -> float:
        """Compute total P + A + M."""
        return (self.P.sum() + self.A.sum() + self.M.sum()).item()
    
    def set_transfer_rate(
        self,
        source: FieldType,
        target: FieldType,
        rate_field: torch.Tensor
    ):
        """
        Set transfer rate from source to target field.
        
        The rate_field specifies local transfer rate at each grid point.
        Positive values = transfer from source to target.
        """
        i, j = source.value, target.value
        if i == j:
            raise ValueError("Cannot transfer field to itself")
        self.R[i, j] = rate_field
    
    def compute_driven_transfers(
        self,
        driving_field: torch.Tensor,
        coupling_strength: float = 0.1
    ) -> None:
        """
        Compute transfer rates driven by an external field.
        
        This models how gradients/dynamics drive redistribution.
        High gradient regions → more active transfer.
        
        Args:
            driving_field: Field that drives transfers (e.g., curvature)
            coupling_strength: How strongly driving affects transfers
        """
        # Gradient magnitude drives transfer rates
        if driving_field.dim() == 1:
            grad = torch.zeros_like(driving_field)
            grad[1:-1] = (driving_field[2:] - driving_field[:-2]) / 2
        elif driving_field.dim() == 2:
            grad_x = torch.zeros_like(driving_field)
            grad_y = torch.zeros_like(driving_field)
            grad_x[:, 1:-1] = (driving_field[:, 2:] - driving_field[:, :-2]) / 2
            grad_y[1:-1, :] = (driving_field[2:, :] - driving_field[:-2, :]) / 2
            grad = torch.sqrt(grad_x**2 + grad_y**2)
        else:
            # 3D - use simple absolute value as proxy
            grad = driving_field.abs()
        
        # Transfer rates proportional to gradient
        # P → A where gradient is high (potential activates)
        # A → M where gradient is low (active materializes)
        # M → P where gradient is medium (material stores)
        
        grad_norm = grad / (grad.max() + 1e-10)
        
        # P → A: High gradients activate potential
        self.R[0, 1] = coupling_strength * grad_norm * self.P
        
        # A → M: Low gradients materialize active
        self.R[1, 2] = coupling_strength * (1 - grad_norm) * self.A
        
        # M → P: Medium gradients store material
        medium = 1 - 2*torch.abs(grad_norm - 0.5)
        self.R[2, 0] = coupling_strength * medium * self.M
    
    def apply_transfers(self, dt: float = 0.01) -> ConservationMetrics:
        """
        Apply one timestep of field redistribution.
        
        This is the core operation: move energy between P, A, M
        while keeping total constant.
        
        Uses explicit conservation: after computing deltas,
        normalize to ensure exact zero-sum.
        """
        # Compute net flow for each field
        # dP/dt = (flows into P) - (flows out of P)
        # dP/dt = R[1,0] + R[2,0] - R[0,1] - R[0,2]
        
        delta_P = (self.R[1,0] + self.R[2,0] - self.R[0,1] - self.R[0,2]) * dt
        delta_A = (self.R[0,1] + self.R[2,1] - self.R[1,0] - self.R[1,2]) * dt
        delta_M = (self.R[0,2] + self.R[1,2] - self.R[2,0] - self.R[2,1]) * dt
        
        # CRITICAL: Force exact conservation
        # The deltas should theoretically sum to zero, but enforce numerically
        total_delta = delta_P.sum() + delta_A.sum() + delta_M.sum()
        
        if abs(total_delta.item()) > 1e-15:
            # Redistribute the error equally
            correction = total_delta / 3.0
            delta_P = delta_P - correction / delta_P.numel()
            delta_A = delta_A - correction / delta_A.numel()
            delta_M = delta_M - correction / delta_M.numel()
        
        # Apply deltas
        self.P = self.P + delta_P
        self.A = self.A + delta_A  
        self.M = self.M + delta_M
        
        # Enforce non-negativity (energy can't be negative)
        # If any field goes negative, redistribute from others
        self._enforce_positivity()
        
        # Track
        transferred = (
            self.R[0,1].abs().sum() + self.R[0,2].abs().sum() +
            self.R[1,0].abs().sum() + self.R[1,2].abs().sum() +
            self.R[2,0].abs().sum() + self.R[2,1].abs().sum()
        ).item() * dt
        self.cumulative_transferred += transferred
        self.step += 1
        
        # Compute metrics
        total = self._compute_total()
        return ConservationMetrics(
            total_pac=total,
            initial_total=self.initial_total,
            drift=abs(total - self.initial_total) / self.initial_total,
            p_fraction=self.P.sum().item() / total,
            a_fraction=self.A.sum().item() / total,
            m_fraction=self.M.sum().item() / total,
            transfers_this_step=int(transferred * 1e6),  # Micro-transfers
            cumulative_transfers=self.cumulative_transferred
        )
    
    def _enforce_positivity(self):
        """
        Ensure all fields are non-negative while preserving total.
        
        If any field has negative values, take from others to cover.
        """
        # Find negative regions
        p_neg = torch.minimum(self.P, torch.zeros_like(self.P))
        a_neg = torch.minimum(self.A, torch.zeros_like(self.A))
        m_neg = torch.minimum(self.M, torch.zeros_like(self.M))
        
        # Total negative (need to redistribute)
        total_neg = p_neg.sum() + a_neg.sum() + m_neg.sum()
        
        if total_neg < -1e-15:
            # Clip to zero
            self.P = torch.maximum(self.P, torch.zeros_like(self.P))
            self.A = torch.maximum(self.A, torch.zeros_like(self.A))
            self.M = torch.maximum(self.M, torch.zeros_like(self.M))
            
            # Redistribute the removed negative from positive regions
            total_pos = self.P.sum() + self.A.sum() + self.M.sum()
            scale = (self.initial_total) / (total_pos + 1e-15)
            
            self.P = self.P * scale
            self.A = self.A * scale
            self.M = self.M * scale
    
    def project_to_phi_ratios(self):
        """
        Project field fractions toward golden ratio relationships.
        
        PAC predicts equilibrium at:
            P : A : M = 1 : φ : φ²
            
        This is the attractor state.
        """
        total = self._compute_total()
        
        # Target fractions
        phi_sum = 1 + PHI + PHI**2
        target_p = total * 1 / phi_sum
        target_a = total * PHI / phi_sum
        target_m = total * PHI**2 / phi_sum
        
        # Smooth projection (don't jump, approach)
        alpha = 0.1  # Projection strength
        
        current_p = self.P.sum().item()
        current_a = self.A.sum().item()
        current_m = self.M.sum().item()
        
        # Compute corrections
        delta_p = alpha * (target_p - current_p) / self.P.numel()
        delta_a = alpha * (target_a - current_a) / self.A.numel()
        delta_m = alpha * (target_m - current_m) / self.M.numel()
        
        # Apply uniformly
        self.P = self.P + delta_p
        self.A = self.A + delta_a
        self.M = self.M + delta_m
    
    def evolve_with_klein_gordon(
        self,
        kg_field: torch.Tensor,
        steps: int,
        dt: float = 0.01
    ) -> List[ConservationMetrics]:
        """
        Evolve PAC fields driven by Klein-Gordon field.
        
        The KG field provides the dynamics that drive transfers.
        """
        metrics = []
        
        for i in range(steps):
            # Use KG field to compute transfer rates
            self.compute_driven_transfers(kg_field, coupling_strength=0.1)
            
            # Apply one step
            m = self.apply_transfers(dt)
            metrics.append(m)
            
            # Optional: project toward φ equilibrium
            if i % 100 == 0:
                self.project_to_phi_ratios()
        
        return metrics
    
    def get_state(self) -> Dict:
        """Get current field state."""
        total = self._compute_total()
        return {
            'P': self.P.clone(),
            'A': self.A.clone(),
            'M': self.M.clone(),
            'total': total,
            'drift': abs(total - self.initial_total) / self.initial_total,
            'fractions': {
                'P': self.P.sum().item() / total,
                'A': self.A.sum().item() / total,
                'M': self.M.sum().item() / total
            },
            'phi_ratios': {
                'target': (1/PHI, 1, PHI),
                'actual': (
                    self.P.sum().item() / self.A.sum().item() if self.A.sum().item() > 0 else 0,
                    1,
                    self.M.sum().item() / self.A.sum().item() if self.A.sum().item() > 0 else 0
                )
            }
        }


def test_rearrangement_tensor():
    """Test the rearrangement tensor with conservation enforcement."""
    print("=" * 60)
    print("REARRANGEMENT TENSOR TEST")
    print("=" * 60)
    
    # Initialize
    rt = RearrangementTensor(shape=(32, 32), initial_total=100.0)
    
    initial_total = rt._compute_total()
    print(f"\nInitial total: {initial_total:.6f}")
    print(f"Initial P/A/M: {rt.P.sum():.4f} / {rt.A.sum():.4f} / {rt.M.sum():.4f}")
    
    # Create a driving field (simulate some dynamics)
    x = torch.linspace(-1, 1, 32)
    y = torch.linspace(-1, 1, 32)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    driving = torch.sin(2*np.pi*X) * torch.cos(2*np.pi*Y)
    
    print("\nEvolving for 1000 steps with dynamic driving...")
    
    metrics = []
    for i in range(1000):
        # Rotate driving field
        phase = i * 0.01
        driving_t = torch.sin(2*np.pi*X + phase) * torch.cos(2*np.pi*Y + phase)
        
        rt.compute_driven_transfers(driving_t, coupling_strength=0.1)
        m = rt.apply_transfers(dt=0.01)
        metrics.append(m)
        
        if i % 100 == 0:
            rt.project_to_phi_ratios()
    
    # Final state
    state = rt.get_state()
    
    print("\n" + "-" * 40)
    print("CONSERVATION RESULTS")
    print("-" * 40)
    print(f"Final total:   {state['total']:.6f}")
    print(f"Initial total: {initial_total:.6f}")
    print(f"Drift:         {state['drift']:.2e} ({state['drift']*100:.6f}%)")
    
    print("\n" + "-" * 40)
    print("FIELD DISTRIBUTION")
    print("-" * 40)
    print(f"P fraction: {state['fractions']['P']:.4f}")
    print(f"A fraction: {state['fractions']['A']:.4f}")
    print(f"M fraction: {state['fractions']['M']:.4f}")
    
    print("\n" + "-" * 40)
    print("PHI RATIO ANALYSIS")
    print("-" * 40)
    print(f"Target P:A:M = 1 : φ : φ² = 1 : {PHI:.4f} : {PHI**2:.4f}")
    print(f"Normalized: {1/(1+PHI+PHI**2):.4f} : {PHI/(1+PHI+PHI**2):.4f} : {PHI**2/(1+PHI+PHI**2):.4f}")
    actual = state['phi_ratios']['actual']
    print(f"Actual P:A:M ratios: {actual[0]:.4f} : {actual[1]:.4f} : {actual[2]:.4f}")
    
    # Conservation check
    all_drifts = [m.drift for m in metrics]
    max_drift = max(all_drifts)
    avg_drift = sum(all_drifts) / len(all_drifts)
    
    print("\n" + "-" * 40)
    print("CONSERVATION QUALITY")
    print("-" * 40)
    print(f"Max drift during evolution: {max_drift:.2e}")
    print(f"Avg drift during evolution: {avg_drift:.2e}")
    print(f"Conservation maintained:    {'✓ YES' if max_drift < 1e-10 else '✗ NO'}")
    
    print("\n" + "=" * 60)
    print("REARRANGEMENT TENSOR TEST COMPLETE")
    print("=" * 60)
    
    return state, metrics


if __name__ == '__main__':
    test_rearrangement_tensor()
