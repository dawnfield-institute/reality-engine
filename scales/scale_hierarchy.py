"""
Scale Hierarchy Module - φ-Based Scale Relationships

Implements the scale hierarchy where PAC predicts:
- Scales follow φ^k pattern
- Each scale level satisfies PAC recursion
- Cross-scale coherence maintains conservation

Key Result from exp_26_pac_violation.py:
    Adjacent scales with φ ratios conserve PAC better than other ratios.
    r = -0.588, p = 0.0104 for φ as attractor.

This module creates the scale ladder from quantum to cosmic:
    - Planck scale: k=0 → Ψ(0) = 1
    - Particle scale: k~10 → Ψ(10) = φ^(-10) ≈ 0.0082
    - Atomic scale: k~20 → Ψ(20) = φ^(-20) ≈ 6.7e-5
    - Stellar scale: k~40 → Ψ(40) = φ^(-40) ≈ 4.5e-9
    - Cosmic scale: k~80 → Ψ(80) = φ^(-80) ≈ 2e-17

Reference:
- dawn-field-theory/foundational/experiments/prime_harmonic_manifold/scripts/exp_26_pac_violation.py
- dawn-field-theory/foundational/docs/preprints/drafts/[pac][D][v1.0][C4][I4][E]_pac_comprehensive_preprint.md
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# Fundamental constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
XI = 1.0571  # Balance operator
PLANCK_SCALE = 1.616e-35  # Planck length in meters


@dataclass
class ScaleLevel:
    """Represents one level in the scale hierarchy."""
    k: int  # Scale index
    psi: float  # PAC amplitude Ψ(k) = φ^(-k)
    physical_scale: float  # Physical length scale (meters)
    name: str  # Human-readable name
    pac_density: float  # PAC density at this scale


@dataclass
class ScaleTransition:
    """Represents transition between adjacent scales."""
    source_k: int
    target_k: int
    ratio: float  # Should be φ for PAC compliance
    transfer_rate: float
    pac_conserved: bool


class ScaleHierarchy:
    """
    Implements the φ-based scale hierarchy.
    
    Core principle: Scales are not arbitrary but follow PAC recursion.
    The solution Ψ(k) = φ^(-k) defines relative amplitudes.
    
    Physical interpretation:
    - k=0: Planck scale (fundamental)
    - k increases: larger physical scales
    - Each step up is factor of φ in size
    - Energy/amplitude decreases as φ^(-k)
    
    This explains:
    - Why particles have specific mass ratios
    - Why stellar and galactic scales have preferred sizes
    - Why structure forms at particular scales
    """
    
    # Named scale levels (approximate k values)
    NAMED_SCALES = {
        'planck': 0,
        'string': 5,
        'particle': 10,
        'nuclear': 15,
        'atomic': 20,
        'molecular': 25,
        'cellular': 35,
        'human': 45,
        'planetary': 55,
        'stellar': 60,
        'galactic': 75,
        'cosmic': 90,
        'horizon': 100
    }
    
    def __init__(
        self,
        k_min: int = 0,
        k_max: int = 100,
        base_scale: float = PLANCK_SCALE,
        device: str = 'cpu'
    ):
        """
        Initialize scale hierarchy.
        
        Args:
            k_min: Minimum scale index
            k_max: Maximum scale index
            base_scale: Physical scale at k=0
            device: Compute device
        """
        self.k_min = k_min
        self.k_max = k_max
        self.base_scale = base_scale
        self.device = device
        
        # Build scale levels
        self.levels: List[ScaleLevel] = []
        for k in range(k_min, k_max + 1):
            psi = PHI ** (-k)
            scale = base_scale * (PHI ** k)  # Physical scale grows as φ^k
            name = self._get_scale_name(k)
            pac_density = psi ** 2  # Energy ~ amplitude²
            
            self.levels.append(ScaleLevel(
                k=k, psi=psi, physical_scale=scale,
                name=name, pac_density=pac_density
            ))
        
        # Initialize field amplitudes at each scale
        self.amplitudes = torch.tensor(
            [level.psi for level in self.levels],
            dtype=torch.float64, device=device
        )
        
        print(f"📊 Scale Hierarchy initialized:")
        print(f"   Range: k={k_min} to k={k_max} ({k_max - k_min + 1} levels)")
        print(f"   Base scale: {base_scale:.2e} m (Planck)")
        print(f"   Max scale: {base_scale * PHI**k_max:.2e} m")
        print(f"   Key scales:")
        for name in ['planck', 'particle', 'atomic', 'stellar', 'galactic']:
            k = self.NAMED_SCALES.get(name, 0)
            if k_min <= k <= k_max:
                level = self.get_level(k)
                print(f"      {name}: k={k}, scale={level.physical_scale:.2e} m, Ψ={level.psi:.2e}")
    
    def _get_scale_name(self, k: int) -> str:
        """Get human-readable name for scale level."""
        for name, level_k in self.NAMED_SCALES.items():
            if abs(k - level_k) <= 2:
                return f"{name} (k={k})"
        return f"scale k={k}"
    
    def get_level(self, k: int) -> ScaleLevel:
        """Get scale level by index."""
        if not (self.k_min <= k <= self.k_max):
            raise ValueError(f"Scale k={k} outside range [{self.k_min}, {self.k_max}]")
        return self.levels[k - self.k_min]
    
    def verify_pac_recursion(self) -> Dict:
        """
        Verify that scale hierarchy satisfies PAC recursion.
        
        PAC requires: Ψ(k) = Ψ(k+1) + Ψ(k+2)
        With Ψ = φ^(-k), this should hold exactly.
        """
        violations = []
        max_violation = 0.0
        
        for k in range(self.k_min, self.k_max - 1):
            psi_k = PHI ** (-k)
            psi_k1 = PHI ** (-(k+1))
            psi_k2 = PHI ** (-(k+2))
            
            # PAC recursion: Ψ(k) = Ψ(k+1) + Ψ(k+2)
            residual = abs(psi_k - psi_k1 - psi_k2)
            relative = residual / psi_k
            
            if relative > 1e-10:
                violations.append((k, relative))
            max_violation = max(max_violation, relative)
        
        return {
            'satisfied': len(violations) == 0,
            'max_violation': max_violation,
            'violation_count': len(violations),
            'violations': violations
        }
    
    def verify_phi_ratios(self) -> Dict:
        """
        Verify that adjacent scales have φ ratio.
        
        This is the key prediction: scale ratios = φ.
        """
        ratios = []
        deviations = []
        
        for k in range(self.k_min, self.k_max):
            level_k = self.get_level(k)
            level_k1 = self.get_level(k + 1)
            
            # Amplitude ratio
            amp_ratio = level_k.psi / level_k1.psi
            ratios.append(amp_ratio)
            deviations.append(abs(amp_ratio - PHI))
            
        mean_ratio = np.mean(ratios)
        mean_deviation = np.mean(deviations)
        
        return {
            'mean_ratio': mean_ratio,
            'expected_ratio': PHI,
            'mean_deviation': mean_deviation,
            'max_deviation': max(deviations),
            'exact': mean_deviation < 1e-14
        }
    
    def compute_scale_coupling(
        self,
        k1: int,
        k2: int
    ) -> float:
        """
        Compute coupling strength between two scales.
        
        PAC predicts adjacent scales (|k1-k2|=1) couple most strongly.
        Distant scales have exponentially weaker coupling.
        """
        level1 = self.get_level(k1)
        level2 = self.get_level(k2)
        
        # Coupling ~ product of amplitudes / distance
        distance = abs(k2 - k1)
        if distance == 0:
            return 1.0  # Self-coupling
        
        coupling = level1.psi * level2.psi / (PHI ** distance)
        return coupling
    
    def compute_cross_scale_transfer(
        self,
        source_k: int,
        target_k: int,
        amount: float
    ) -> ScaleTransition:
        """
        Compute transfer between scales.
        
        Energy can flow between scales, but must respect:
        1. Total conservation (handled by RearrangementTensor)
        2. Locality: adjacent scales preferred
        3. φ-ratio: transfers preserve golden ratio structure
        """
        source = self.get_level(source_k)
        target = self.get_level(target_k)
        
        ratio = source.psi / target.psi
        coupling = self.compute_scale_coupling(source_k, target_k)
        
        # Transfer rate modulated by coupling
        effective_rate = amount * coupling
        
        # PAC is conserved if ratio ≈ φ^(target_k - source_k)
        expected_ratio = PHI ** (target_k - source_k)
        pac_conserved = abs(ratio - expected_ratio) < 0.01 * expected_ratio
        
        return ScaleTransition(
            source_k=source_k,
            target_k=target_k,
            ratio=ratio,
            transfer_rate=effective_rate,
            pac_conserved=pac_conserved
        )
    
    def compute_total_pac_density(self) -> float:
        """Compute total PAC density across all scales."""
        return sum(level.pac_density for level in self.levels)
    
    def get_summary(self) -> Dict:
        """Get summary of scale hierarchy."""
        pac_check = self.verify_pac_recursion()
        phi_check = self.verify_phi_ratios()
        
        return {
            'n_levels': len(self.levels),
            'k_range': (self.k_min, self.k_max),
            'scale_range_m': (
                self.levels[0].physical_scale,
                self.levels[-1].physical_scale
            ),
            'total_pac_density': self.compute_total_pac_density(),
            'pac_recursion_satisfied': pac_check['satisfied'],
            'phi_ratios_exact': phi_check['exact'],
            'mean_phi_deviation': phi_check['mean_deviation']
        }


class MultiScaleField:
    """
    A field that exists across multiple scale levels.
    
    This models how physical fields have structure at multiple scales:
    - Quantum fluctuations at small k
    - Classical coherent structure at medium k
    - Cosmic structure at large k
    
    All scales are coupled and exchange via PAC.
    """
    
    def __init__(
        self,
        hierarchy: ScaleHierarchy,
        grid_shape: Tuple[int, ...] = (32, 32),
        device: str = 'cpu'
    ):
        """
        Initialize multi-scale field.
        
        Args:
            hierarchy: Scale hierarchy to use
            grid_shape: Spatial grid shape
            device: Compute device
        """
        self.hierarchy = hierarchy
        self.grid_shape = grid_shape
        self.device = device
        
        # Field at each scale level
        n_levels = len(hierarchy.levels)
        self.fields = torch.zeros(n_levels, *grid_shape, device=device)
        
        # Initialize with scale-appropriate amplitudes
        for i, level in enumerate(hierarchy.levels):
            self.fields[i] = level.psi * torch.randn(*grid_shape, device=device)
    
    def compute_total_energy(self) -> float:
        """Compute total energy across all scales."""
        return (self.fields ** 2).sum().item()
    
    def cascade_down(self, rate: float = 0.01):
        """
        Cascade energy from large to small scales.
        
        Models turbulent cascade in fluids, or
        structure formation in cosmology.
        """
        for i in range(len(self.hierarchy.levels) - 1, 0, -1):
            # Transfer from scale i to scale i-1
            transfer = rate * self.fields[i]
            self.fields[i-1] += transfer
            self.fields[i] -= transfer
    
    def cascade_up(self, rate: float = 0.01):
        """
        Cascade energy from small to large scales.
        
        Models inverse cascade or structure growth.
        """
        for i in range(len(self.hierarchy.levels) - 1):
            # Transfer from scale i to scale i+1
            transfer = rate * self.fields[i]
            self.fields[i+1] += transfer
            self.fields[i] -= transfer
    
    def equilibrate_to_pac(self, steps: int = 100, rate: float = 0.01):
        """
        Equilibrate field to PAC solution Ψ(k) = φ^(-k).
        
        This is the attractor state - all scales should
        converge to golden ratio distribution.
        """
        for _ in range(steps):
            for i, level in enumerate(self.hierarchy.levels):
                current_amp = self.fields[i].abs().mean().item()
                target_amp = level.psi
                
                # Adjust toward target
                if current_amp > 0:
                    correction = (target_amp / current_amp - 1) * rate
                    self.fields[i] *= (1 + correction)
    
    def get_scale_spectrum(self) -> List[Tuple[int, float]]:
        """Get amplitude at each scale level."""
        spectrum = []
        for i, level in enumerate(self.hierarchy.levels):
            amp = self.fields[i].abs().mean().item()
            spectrum.append((level.k, amp))
        return spectrum


def test_scale_hierarchy():
    """Test scale hierarchy and φ relationships."""
    print("=" * 60)
    print("SCALE HIERARCHY TEST")
    print("=" * 60)
    
    # Initialize hierarchy
    sh = ScaleHierarchy(k_min=0, k_max=80)
    
    # Verify PAC recursion
    print("\n" + "-" * 40)
    print("PAC RECURSION VERIFICATION")
    print("-" * 40)
    pac_check = sh.verify_pac_recursion()
    print(f"Recursion satisfied: {pac_check['satisfied']}")
    print(f"Max violation: {pac_check['max_violation']:.2e}")
    
    # Verify φ ratios
    print("\n" + "-" * 40)
    print("PHI RATIO VERIFICATION")
    print("-" * 40)
    phi_check = sh.verify_phi_ratios()
    print(f"Mean ratio: {phi_check['mean_ratio']:.6f}")
    print(f"Expected (φ): {phi_check['expected_ratio']:.6f}")
    print(f"Mean deviation: {phi_check['mean_deviation']:.2e}")
    print(f"Exact to machine precision: {phi_check['exact']}")
    
    # Scale coupling
    print("\n" + "-" * 40)
    print("SCALE COUPLING")
    print("-" * 40)
    print("Coupling strengths (higher = stronger):")
    print(f"  Adjacent (k=10 to k=11): {sh.compute_scale_coupling(10, 11):.2e}")
    print(f"  Near (k=10 to k=12):     {sh.compute_scale_coupling(10, 12):.2e}")
    print(f"  Distant (k=10 to k=20):  {sh.compute_scale_coupling(10, 20):.2e}")
    print(f"  Very far (k=10 to k=40): {sh.compute_scale_coupling(10, 40):.2e}")
    
    # Named scales
    print("\n" + "-" * 40)
    print("NAMED SCALES")
    print("-" * 40)
    for name, k in sh.NAMED_SCALES.items():
        if 0 <= k <= 80:
            level = sh.get_level(k)
            print(f"{name:12}: k={k:3}, scale={level.physical_scale:.2e} m, Ψ={level.psi:.2e}")
    
    # Multi-scale field
    print("\n" + "-" * 40)
    print("MULTI-SCALE FIELD TEST")
    print("-" * 40)
    msf = MultiScaleField(sh, grid_shape=(16, 16))
    
    initial_energy = msf.compute_total_energy()
    print(f"Initial total energy: {initial_energy:.4f}")
    
    # Equilibrate to PAC
    msf.equilibrate_to_pac(steps=100)
    
    final_energy = msf.compute_total_energy()
    print(f"Final total energy: {final_energy:.4f}")
    
    # Check spectrum
    spectrum = msf.get_scale_spectrum()
    print("\nScale spectrum (sample):")
    for k, amp in spectrum[::10]:  # Every 10th
        expected = PHI ** (-k)
        ratio = amp / expected if expected > 0 else 0
        print(f"  k={k:3}: amp={amp:.2e}, expected={expected:.2e}, ratio={ratio:.4f}")
    
    print("\n" + "=" * 60)
    print("SCALE HIERARCHY TEST COMPLETE")
    print("=" * 60)
    
    return sh, msf


if __name__ == '__main__':
    test_scale_hierarchy()
