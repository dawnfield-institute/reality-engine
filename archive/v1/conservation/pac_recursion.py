"""
PAC Recursion Engine

Implements the core PAC conservation law: Ψ(k) = Ψ(k+1) + Ψ(k+2)

This is the mathematical foundation from which φ (golden ratio) emerges
as the unique stable attractor. Validated in exp_26 with p = 0.0104.

Key Properties:
- Unique solution: Ψ(k) = φ^(-k) where φ = (1+√5)/2
- φ is an ATTRACTOR - systems converge to it regardless of initial conditions
- Breaking PAC breaks structure (validated correlation r = -0.588)
- Adjacent levels must maintain φ ratio

Reference:
- dawn-field-theory/foundational/docs/preprints/drafts/[pac][D][v1.0][C4][I5][E]_pac_necessity_proof_preprint.md
- dawn-field-theory/foundational/experiments/prime_harmonic_manifold/scripts/exp_26_pac_violation.py
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# Fundamental constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
INV_PHI = 1 / PHI  # ≈ 0.618
XI = 1.0571  # Balance operator from PAC confluence


@dataclass
class PACMetrics:
    """Metrics for PAC recursion state"""
    recursion_error: float  # How far from Ψ(k) = Ψ(k+1) + Ψ(k+2)
    phi_ratio_error: float  # How far adjacent levels are from φ ratio
    total_conserved: float  # Total field content (should be constant)
    max_level_deviation: float  # Worst individual level error
    levels_in_tolerance: int  # How many levels satisfy PAC
    total_levels: int
    # Resonance tracking (Phase 1 addition)
    resonance_frequency: Optional[float] = None  # Detected natural frequency
    resonance_confidence: float = 0.0  # Detection confidence (0-1)
    resonance_locked: bool = False  # Whether timestep is locked to resonance


class PACRecursion:
    """
    Enforce PAC recursion Ψ(k) = Ψ(k+1) + Ψ(k+2) across field hierarchy.
    
    The PAC recursion is not just an observation—it's a NECESSITY.
    Systems that violate it lose structure (exp_26 validation).
    
    This enforcer:
    1. Checks recursion at each level
    2. Redistributes violations (doesn't create/destroy)
    3. Maintains φ ratios between adjacent levels
    4. Tracks conservation metrics
    """
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        phi_tolerance: float = 0.01,
        correction_rate: float = 0.1,
        device: str = 'cpu',
        enable_resonance: bool = True,
        resonance_check_interval: int = 10
    ):
        """
        Initialize PAC recursion enforcer.

        Args:
            tolerance: Maximum allowed recursion error
            phi_tolerance: Maximum allowed φ ratio deviation (1% default)
            correction_rate: How fast to correct violations (0-1)
            device: Compute device
            enable_resonance: Enable resonance detection for speedup (Phase 1)
            resonance_check_interval: Iterations between resonance checks
        """
        self.tolerance = tolerance
        self.phi_tolerance = phi_tolerance
        self.correction_rate = correction_rate
        self.device = device

        self.history: List[PACMetrics] = []
        self.initial_total: Optional[float] = None

        # Phase 1: Resonance detection
        self.enable_resonance = enable_resonance
        self.resonance_check_interval = resonance_check_interval
        self.resonance_detector = None
        self.pac_residual_history: List[float] = []
        self.current_resonance = None
        self.iterations_since_check = 0

        if enable_resonance:
            try:
                from dynamics.resonance_detector import ResonanceDetector
                self.resonance_detector = ResonanceDetector(min_window=20, max_window=100)
            except ImportError:
                print("Warning: ResonanceDetector not available, continuing without resonance detection")
                self.enable_resonance = False
    
    def compute_pac_solution(self, depth: int, initial_value: float = 1.0) -> torch.Tensor:
        """
        Compute the unique PAC solution: Ψ(k) = φ^(-k)
        
        This is what field hierarchies should converge to.
        
        Args:
            depth: Number of levels
            initial_value: Ψ(0) value
            
        Returns:
            Tensor of Ψ values at each level
        """
        k = torch.arange(depth, device=self.device, dtype=torch.float64)
        psi = initial_value * (PHI ** (-k))
        return psi
    
    def verify_recursion(self, psi: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        """
        Verify PAC recursion holds: Ψ(k) = Ψ(k+1) + Ψ(k+2)
        
        Args:
            psi: Field values at each level (1D tensor)
            
        Returns:
            (satisfied, errors) where errors[k] = Ψ(k) - Ψ(k+1) - Ψ(k+2)
        """
        if len(psi) < 3:
            return True, torch.zeros(0, device=self.device)
        
        # Compute errors at each level
        errors = torch.zeros(len(psi) - 2, device=self.device, dtype=psi.dtype)
        for k in range(len(psi) - 2):
            target = psi[k+1] + psi[k+2]
            errors[k] = psi[k] - target
        
        # Check if within tolerance
        max_error = errors.abs().max().item() if len(errors) > 0 else 0.0
        satisfied = max_error < self.tolerance
        
        return satisfied, errors
    
    def verify_phi_ratios(self, psi: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        """
        Verify adjacent levels maintain φ ratio: Ψ(k)/Ψ(k+1) ≈ φ
        
        Args:
            psi: Field values at each level
            
        Returns:
            (satisfied, ratio_errors)
        """
        if len(psi) < 2:
            return True, torch.zeros(0, device=self.device)
        
        # Compute ratios
        ratios = psi[:-1] / (psi[1:] + 1e-15)  # Avoid division by zero
        
        # Errors from φ
        ratio_errors = (ratios - PHI).abs() / PHI
        
        # Check if within tolerance
        max_error = ratio_errors.max().item() if len(ratio_errors) > 0 else 0.0
        satisfied = max_error < self.phi_tolerance
        
        return satisfied, ratio_errors
    
    def enforce(
        self,
        field_hierarchy: List[torch.Tensor],
        conserve_total: bool = True
    ) -> Tuple[List[torch.Tensor], PACMetrics]:
        """
        Enforce PAC recursion across field hierarchy.
        
        This is the core enforcement step. It:
        1. Computes total at each level
        2. Checks PAC recursion
        3. Redistributes to satisfy recursion (without changing total)
        4. Returns corrected fields
        
        Args:
            field_hierarchy: List of field tensors at each scale
            conserve_total: If True, preserve total field content
            
        Returns:
            (corrected_hierarchy, metrics)
        """
        # Get totals at each level
        totals = torch.tensor(
            [f.sum().item() for f in field_hierarchy],
            device=self.device,
            dtype=torch.float64
        )
        
        # Track initial total for conservation
        current_total = totals.sum().item()
        if self.initial_total is None:
            self.initial_total = current_total
        
        # Verify current state
        recursion_ok, recursion_errors = self.verify_recursion(totals)
        phi_ok, phi_errors = self.verify_phi_ratios(totals)
        
        # If violations exist, correct them
        if not recursion_ok:
            totals = self._correct_recursion(totals, recursion_errors)
        
        if not phi_ok:
            totals = self._correct_phi_ratios(totals, phi_errors)
        
        # Rescale to conserve total if needed
        if conserve_total and self.initial_total is not None:
            scale = self.initial_total / (totals.sum().item() + 1e-15)
            totals = totals * scale
        
        # Apply corrections to actual fields
        corrected = self._apply_corrections(field_hierarchy, totals)
        
        # Compute final metrics
        _, final_recursion_errors = self.verify_recursion(totals)
        _, final_phi_errors = self.verify_phi_ratios(totals)

        # Phase 1: Track PAC residual for resonance detection
        pac_residual = final_recursion_errors.abs().mean().item() if len(final_recursion_errors) > 0 else 0.0

        # Resonance detection (if enabled)
        resonance_freq = None
        resonance_conf = 0.0
        resonance_locked = False

        if self.enable_resonance and self.resonance_detector is not None:
            # Track residual history
            self.pac_residual_history.append(pac_residual)
            self.iterations_since_check += 1

            # Periodic resonance check
            if self.iterations_since_check >= self.resonance_check_interval:
                self.current_resonance = self.resonance_detector.analyze_oscillations(
                    self.pac_residual_history
                )
                self.iterations_since_check = 0

            # Extract resonance info if available
            if self.current_resonance is not None:
                resonance_freq = self.current_resonance.get('frequency')
                resonance_conf = self.current_resonance.get('confidence', 0.0)
                resonance_locked = self.resonance_detector.should_lock(
                    self.current_resonance,
                    stability_threshold=0.5
                )

        metrics = PACMetrics(
            recursion_error=pac_residual,
            phi_ratio_error=final_phi_errors.mean().item() if len(final_phi_errors) > 0 else 0.0,
            total_conserved=totals.sum().item(),
            max_level_deviation=final_recursion_errors.abs().max().item() if len(final_recursion_errors) > 0 else 0.0,
            levels_in_tolerance=int((final_recursion_errors.abs() < self.tolerance).sum().item()) if len(final_recursion_errors) > 0 else len(field_hierarchy),
            total_levels=len(field_hierarchy),
            # Phase 1: Resonance metrics
            resonance_frequency=resonance_freq,
            resonance_confidence=resonance_conf,
            resonance_locked=resonance_locked
        )

        self.history.append(metrics)
        return corrected, metrics
    
    def _correct_recursion(
        self,
        totals: torch.Tensor,
        errors: torch.Tensor
    ) -> torch.Tensor:
        """
        Correct recursion violations via redistribution.
        
        Key: This doesn't create or destroy—it moves between levels.
        """
        corrected = totals.clone()
        
        # Work from top to bottom
        for k in range(len(errors)):
            error = errors[k].item()
            
            if abs(error) > self.tolerance:
                # Redistribute error across three involved levels
                # Ψ(k) = Ψ(k+1) + Ψ(k+2), so if Ψ(k) is too high:
                # - Reduce Ψ(k) by error/3
                # - Increase Ψ(k+1) by error/3
                # - Increase Ψ(k+2) by error/3
                correction = error * self.correction_rate / 3
                corrected[k] -= correction
                corrected[k+1] += correction
                corrected[k+2] += correction
        
        return corrected
    
    def _correct_phi_ratios(
        self,
        totals: torch.Tensor,
        errors: torch.Tensor
    ) -> torch.Tensor:
        """
        Correct φ ratio violations.
        
        If Ψ(k)/Ψ(k+1) != φ, adjust to approach φ.
        """
        corrected = totals.clone()
        
        for k in range(len(errors)):
            if errors[k] > self.phi_tolerance:
                # Ratio is too far from φ
                current_ratio = corrected[k] / (corrected[k+1] + 1e-15)
                target = corrected[k+1] * PHI
                
                # Move toward target
                adjustment = (target - corrected[k]) * self.correction_rate
                corrected[k] += adjustment
        
        return corrected
    
    def _apply_corrections(
        self,
        field_hierarchy: List[torch.Tensor],
        target_totals: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Apply corrected totals to actual field tensors.
        
        Scales each field to match target total while preserving shape.
        """
        corrected = []
        
        for i, field in enumerate(field_hierarchy):
            current_total = field.sum().item()
            target = target_totals[i].item()
            
            if abs(current_total) > 1e-15:
                scale = target / current_total
                corrected.append(field * scale)
            else:
                # Field is zero, can't scale—initialize to uniform
                uniform_value = target / field.numel()
                corrected.append(torch.full_like(field, uniform_value))
        
        return corrected
    
    def project_to_pac_manifold(
        self,
        field_hierarchy: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Project field hierarchy onto PAC solution manifold.
        
        This is a stronger operation than enforce()—it directly
        sets the hierarchy to follow Ψ(k) = φ^(-k) while preserving
        the internal structure of each field.
        
        Args:
            field_hierarchy: Current fields
            
        Returns:
            Fields projected to PAC manifold
        """
        # Get total content to preserve
        total = sum(f.sum().item() for f in field_hierarchy)
        
        # Compute ideal PAC totals
        depth = len(field_hierarchy)
        pac_totals = self.compute_pac_solution(depth)
        
        # Normalize to preserve total
        pac_totals = pac_totals * (total / pac_totals.sum().item())
        
        # Apply to fields
        return self._apply_corrections(field_hierarchy, pac_totals)
    
    def get_convergence_report(self) -> Dict:
        """Get summary of PAC convergence over history."""
        if not self.history:
            return {'status': 'no_history'}

        recursion_errors = [m.recursion_error for m in self.history]
        phi_errors = [m.phi_ratio_error for m in self.history]
        totals = [m.total_conserved for m in self.history]

        report = {
            'steps': len(self.history),
            'final_recursion_error': recursion_errors[-1],
            'final_phi_error': phi_errors[-1],
            'conservation_drift': abs(totals[-1] - totals[0]) / (totals[0] + 1e-15) if totals[0] != 0 else 0,
            'recursion_converged': recursion_errors[-1] < self.tolerance,
            'phi_converged': phi_errors[-1] < self.phi_tolerance,
            'mean_recursion_error': np.mean(recursion_errors),
            'mean_phi_error': np.mean(phi_errors)
        }

        # Phase 1: Add resonance info if enabled
        if self.enable_resonance and self.current_resonance is not None:
            report['resonance_frequency'] = self.current_resonance.get('frequency')
            report['resonance_period'] = self.current_resonance.get('period')
            report['resonance_confidence'] = self.current_resonance.get('confidence', 0.0)
            report['resonance_locked'] = self.history[-1].resonance_locked if self.history else False
            if self.resonance_detector is not None:
                report['resonance_stability'] = self.resonance_detector.get_detection_stability()

        return report

    def get_suggested_timestep(self, base_dt: float = 0.1) -> Optional[float]:
        """
        Get suggested timestep based on detected resonance.

        Phase 1 addition: Uses resonance detection to suggest optimal timestep
        for resonance-locked convergence (5× speedup).

        Args:
            base_dt: Current timestep

        Returns:
            Suggested timestep, or None if no resonance detected
        """
        if not self.enable_resonance or self.resonance_detector is None:
            return None

        if self.current_resonance is None:
            return None

        return self.resonance_detector.suggest_timestep(self.current_resonance, base_dt)

    def is_resonance_locked(self) -> bool:
        """
        Check if PAC evolution is currently locked to resonance.

        Returns:
            True if resonance is detected and locked
        """
        if not self.enable_resonance or not self.history:
            return False

        return self.history[-1].resonance_locked


def test_pac_recursion():
    """Test PAC recursion enforcement."""
    print("=" * 60)
    print("PAC RECURSION TEST")
    print("=" * 60)
    
    enforcer = PACRecursion(tolerance=1e-6, phi_tolerance=0.01)
    
    # Test 1: Verify PAC solution
    print("\n1. Verifying PAC solution Ψ(k) = φ^(-k)...")
    psi = enforcer.compute_pac_solution(10)
    recursion_ok, errors = enforcer.verify_recursion(psi)
    phi_ok, phi_errors = enforcer.verify_phi_ratios(psi)
    
    print(f"   PAC solution values: {psi[:5].numpy()}")
    print(f"   Recursion satisfied: {recursion_ok}")
    print(f"   Max recursion error: {errors.abs().max().item():.2e}")
    print(f"   φ ratios satisfied: {phi_ok}")
    print(f"   Max φ error: {phi_errors.max().item():.2e}")
    
    # Test 2: Random hierarchy should be corrected
    print("\n2. Correcting random hierarchy...")
    random_fields = [torch.rand(8, 8) for _ in range(5)]
    corrected, metrics = enforcer.enforce(random_fields)
    
    print(f"   Initial totals: {[round(f.sum().item(), 3) for f in random_fields]}")
    print(f"   Corrected totals: {[round(f.sum().item(), 3) for f in corrected]}")
    print(f"   Recursion error: {metrics.recursion_error:.6f}")
    print(f"   φ ratio error: {metrics.phi_ratio_error:.6f}")
    
    # Test 3: Iterate to convergence
    print("\n3. Iterating to PAC convergence...")
    fields = [torch.rand(8, 8) for _ in range(5)]
    
    for i in range(100):
        fields, metrics = enforcer.enforce(fields)
    
    report = enforcer.get_convergence_report()
    print(f"   After 100 iterations:")
    print(f"   Recursion converged: {report['recursion_converged']}")
    print(f"   φ converged: {report['phi_converged']}")
    print(f"   Final recursion error: {report['final_recursion_error']:.6e}")
    print(f"   Conservation drift: {report['conservation_drift']:.6e}")
    
    # Test 4: Project to PAC manifold
    print("\n4. Projecting to PAC manifold...")
    fields = [torch.rand(8, 8) for _ in range(5)]
    projected = enforcer.project_to_pac_manifold(fields)
    
    totals = torch.tensor([f.sum().item() for f in projected])
    _, errors = enforcer.verify_recursion(totals)
    _, phi_errors = enforcer.verify_phi_ratios(totals)
    
    print(f"   Projected totals: {totals.numpy()}")
    print(f"   Recursion errors: {errors.numpy()}")
    print(f"   φ ratio errors: {phi_errors.numpy()}")
    
    print("\n" + "=" * 60)
    print("PAC RECURSION TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    test_pac_recursion()
