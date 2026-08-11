"""
Gravity Analyzer - Measure gravitational forces and compare to Newton/Einstein.

Tracks actual forces between structures and measures:
- Force strength in simulation units
- Scaling to physical units (if calibrated)
- Deviation from 1/r² law
- Comparison to known gravitational constants
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from scipy import stats
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_analyzer import BaseAnalyzer, Detection


class GravityAnalyzer(BaseAnalyzer):
    """
    Analyzes gravitational phenomena and measures actual forces.
    
    Can be calibrated to physical units for direct comparison with reality.
    
    Unit Calibration Examples:
    - Atomic scale: length_scale=1e-10 (Ångström per grid unit)
    - Stellar scale: length_scale=1e9 (1000 km per grid unit)
    - Galactic scale: length_scale=9.46e15 (1 light-year per grid unit)
    """
    
    # Known physical constants for comparison
    G_SI = 6.67430e-11  # m³ kg⁻¹ s⁻² (Newton's gravitational constant)
    
    def __init__(self, 
                 length_scale: float = 1.0,  # meters per grid unit
                 mass_scale: float = 1.0,     # kg per mass unit
                 time_scale: float = 1.0,     # seconds per time unit
                 min_confidence: float = 0.5):
        """
        Initialize with optional unit calibration.
        
        Args:
            length_scale: Physical length per grid unit
            mass_scale: Physical mass per simulation mass unit
            time_scale: Physical time per simulation time unit
            min_confidence: Minimum confidence for reporting
        """
        super().__init__("gravity_analyzer", min_confidence)
        
        self.length_scale = length_scale
        self.mass_scale = mass_scale
        self.time_scale = time_scale
        
        # Compute G in simulation units if calibrated
        # G_sim = G_real * (time² / (length³ / mass))
        self.G_calibrated = self.G_SI * (self.time_scale**2 * self.mass_scale) / (self.length_scale**3)
        
        # Track measurements over time
        self.force_measurements = []
        self.G_measurements = []
        
    def analyze(self, state: Dict) -> List[Detection]:
        """Analyze gravitational forces in current state"""
        detections = []
        
        # Need structures to measure forces between
        if 'structures' not in state or len(state['structures']) < 2:
            return detections
        
        structures = state['structures']
        current_time = state.get('time', 0.0)
        
        # Measure pairwise forces
        forces = []
        for i, s1 in enumerate(structures):
            for j, s2 in enumerate(structures[i+1:], i+1):
                force_data = self._measure_force(s1, s2, state)
                if force_data:
                    forces.append(force_data)
                    self.force_measurements.append({
                        'time': current_time,
                        **force_data
                    })
        
        if not forces:
            return detections
        
        # Analyze force scaling with distance
        distances = np.array([f['distance'] for f in forces])
        force_mags = np.array([f['force'] for f in forces])
        mass_products = np.array([f['m1'] * f['m2'] for f in forces])
        
        # Fit to inverse power law: F = G·m1·m2/r^n
        if len(forces) >= 3:
            n, G_measured, r_squared = self._fit_power_law(
                distances, force_mags, mass_products
            )
            
            self.G_measurements.append({
                'time': current_time,
                'G': G_measured,
                'n': n,
                'r_squared': r_squared
            })
            
            # Convert to physical units if calibrated
            G_physical = G_measured * (self.length_scale**3) / (self.mass_scale * self.time_scale**2)
            
            # Compute deviation from Newton's law
            newton_deviation = abs(n - 2.0)
            G_ratio = G_physical / self.G_SI if self.G_SI else None
            
            # Create main gravity detection
            detection = Detection(
                type="gravitational_force",
                confidence=r_squared * (1.0 - newton_deviation / 2.0),  # Penalize deviation from n=2
                time=current_time,
                equation=f"F = {G_measured:.3e}·m₁·m₂/r^{n:.2f}",
                parameters={
                    'G_simulation': G_measured,
                    'G_physical': G_physical,
                    'power_law_exponent': n,
                    'r_squared': r_squared,
                    'num_measurements': len(forces)
                },
                properties={
                    'matches_newton': newton_deviation < 0.1,
                    'G_ratio_to_reality': G_ratio,
                    'mean_force': float(np.mean(force_mags)),
                    'max_force': float(np.max(force_mags)),
                    'min_force': float(np.min(force_mags)),
                    'mean_distance': float(np.mean(distances))
                }
            )
            detections.append(detection)
            
            # Check if it matches Newtonian gravity (n ≈ 2, good fit)
            if newton_deviation < 0.1 and r_squared > 0.8:
                detections.append(Detection(
                    type="newtonian_gravity",
                    confidence=r_squared * (1.0 - newton_deviation * 5.0),
                    time=current_time,
                    equation=f"F = G·m₁·m₂/r²",
                    parameters={
                        'G_simulation': G_measured,
                        'G_physical': G_physical,
                        'G_SI': self.G_SI,
                        'deviation_from_n_equals_2': newton_deviation
                    },
                    properties={
                        'force_strength_comparison': self._compare_force_strength(G_physical),
                        'matches_reality': G_ratio is not None and 0.1 < G_ratio < 10.0 if G_ratio else False
                    }
                ))
        
        # Detect gravitational collapse (structures merging)
        collapse = self._detect_collapse(state, current_time)
        if collapse:
            detections.append(collapse)
        
        # Detect orbital motion if we have velocity data
        orbits = self._detect_orbits(structures, current_time)
        detections.extend(orbits)
        
        return detections
    
    def _measure_force(self, s1, s2, state: Dict) -> Optional[Dict]:
        """Measure actual force between two structures"""
        # Get positions
        r1 = np.array(s1.center)
        r2 = np.array(s2.center)
        
        # Distance
        dr = r2 - r1
        distance = np.linalg.norm(dr)
        
        if distance < 1e-10:
            return None
        
        # Get masses
        m1 = s1.mass
        m2 = s2.mass
        
        # Measure force from acceleration (if available)
        if hasattr(s1, 'acceleration') and hasattr(s2, 'acceleration'):
            # Use actual measured acceleration from persistent tracking
            a1 = np.array(s1.acceleration) if s1.acceleration is not None else np.zeros(len(r1))
            a2 = np.array(s2.acceleration) if s2.acceleration is not None else np.zeros(len(r2))
            
            # Force magnitude from F = m·a
            F1 = m1 * np.linalg.norm(a1)
            F2 = m2 * np.linalg.norm(a2)
            
            # Average (should be equal by Newton's 3rd law)
            force_mag = (F1 + F2) / 2.0
        else:
            # Estimate from field gradient (information density)
            # In Dawn Field Theory, gravity emerges from ∇ρ_I where ρ_I = |A - P|
            if 'actual' in state and 'potential' in state:
                A = state['actual']
                P = state['potential']
                
                # Convert to numpy if tensor
                if torch.is_tensor(A):
                    A = A.cpu().numpy() if A.is_cuda else A.numpy()
                if torch.is_tensor(P):
                    P = P.cpu().numpy() if P.is_cuda else P.numpy()
                
                # Get grid indices (clamp to bounds)
                i1 = int(np.clip(r1[0], 0, A.shape[0] - 1))
                j1 = int(np.clip(r1[1], 0, A.shape[1] - 1))
                i2 = int(np.clip(r2[0], 0, A.shape[0] - 1))
                j2 = int(np.clip(r2[1], 0, A.shape[1] - 1))
                
                # Information density at each structure
                rho1 = abs(A[i1, j1] - P[i1, j1])
                rho2 = abs(A[i2, j2] - P[i2, j2])
                
                # Force proportional to density gradient and masses
                # F ~ m1·m2·Δρ/r²
                drho = abs(rho2 - rho1)
                force_mag = drho * m1 * m2 / (distance**2 + 1e-10)
            else:
                return None
        
        return {
            'distance': distance,
            'force': force_mag,
            'm1': m1,
            'm2': m2,
            'r_vector': dr
        }
    
    def _fit_power_law(self, distances: np.ndarray, forces: np.ndarray, 
                       mass_products: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit F = G·m1·m2/r^n and return (n, G, r_squared).
        
        Args:
            distances: Array of inter-structure distances
            forces: Array of measured forces
            mass_products: Array of m1*m2 for each pair
        
        Returns:
            Tuple of (power_law_exponent, gravitational_constant, r_squared)
        """
        # Normalize forces by mass products: F/(m1·m2) = G/r^n
        normalized_forces = forces / (mass_products + 1e-10)
        
        # Convert to log space: log(F/(m1·m2)) = log(G) - n·log(r)
        log_r = np.log(distances + 1e-10)
        log_F_norm = np.log(normalized_forces + 1e-10)
        
        # Linear fit in log space
        slope, intercept, r_value, _, _ = stats.linregress(log_r, log_F_norm)
        
        n = -slope  # Power law exponent (should be ~2 for gravity)
        G = np.exp(intercept)  # Gravitational constant in simulation units
        r_squared = r_value**2  # Goodness of fit
        
        return n, G, r_squared
    
    def _compare_force_strength(self, G_physical: float) -> str:
        """Compare measured G to reality"""
        if G_physical is None:
            return "unknown (no calibration)"
        
        ratio = G_physical / self.G_SI
        
        if 0.5 < ratio < 2.0:
            return f"matches reality ({ratio:.2f}x)"
        elif ratio < 0.5:
            return f"weaker than reality ({ratio:.2e}x)"
        else:
            return f"stronger than reality ({ratio:.2e}x)"
    
    def _detect_collapse(self, state: Dict, current_time: float) -> Optional[Detection]:
        """Detect gravitational collapse signature"""
        if len(self.history) < 50:
            return None
        
        # Check if structures are merging over time
        structure_counts = []
        for h in self.history[-50:]:
            count = h['summary'].get('structure_count', 0)
            if count > 0:
                structure_counts.append(count)
        
        if len(structure_counts) < 20:
            return None
        
        # Measure coalescence rate
        initial = np.mean(structure_counts[:10])
        final = np.mean(structure_counts[-10:])
        
        if initial > 0:
            coalescence = (initial - final) / initial
            
            if coalescence > 0.3:  # >30% reduction
                return Detection(
                    type="gravitational_collapse",
                    confidence=min(coalescence * 1.5, 1.0),
                    time=current_time,
                    properties={
                        'initial_structures': float(initial),
                        'final_structures': float(final),
                        'coalescence_rate': float(coalescence),
                        'timescale': len(self.history) * 10  # Steps elapsed
                    }
                )
        
        return None
    
    def _detect_orbits(self, structures: List, current_time: float) -> List[Detection]:
        """Detect orbital motion patterns"""
        detections = []
        
        # Need velocity data for orbit detection
        structures_with_velocity = [s for s in structures if hasattr(s, 'velocity') and s.velocity is not None]
        
        if len(structures_with_velocity) < 2:
            return detections
        
        # Look for pairs where velocity is perpendicular to separation (orbital motion)
        for i, s1 in enumerate(structures_with_velocity):
            for s2 in structures_with_velocity[i+1:]:
                r1 = np.array(s1.center)
                r2 = np.array(s2.center)
                v1 = np.array(s1.velocity)
                v2 = np.array(s2.velocity)
                
                # Relative position and velocity
                dr = r2 - r1
                dv = v2 - v1
                
                # Check if velocity is perpendicular to position (orbital condition)
                # dot(dr, dv) ≈ 0 for circular orbits
                dot_product = abs(np.dot(dr, dv))
                magnitude_product = np.linalg.norm(dr) * np.linalg.norm(dv)
                
                if magnitude_product > 1e-10:
                    perpendicularity = 1.0 - (dot_product / magnitude_product)
                    
                    if perpendicularity > 0.8:  # Nearly perpendicular
                        detections.append(Detection(
                            type="orbital_motion",
                            confidence=perpendicularity,
                            time=current_time,
                            location=(tuple(r1), tuple(r2)),
                            properties={
                                'mass1': s1.mass,
                                'mass2': s2.mass,
                                'separation': float(np.linalg.norm(dr)),
                                'relative_velocity': float(np.linalg.norm(dv)),
                                'perpendicularity': float(perpendicularity)
                            }
                        ))
        
        return detections
    
    def get_force_statistics(self) -> Dict:
        """Get statistical summary of all force measurements"""
        if not self.force_measurements:
            return {}
        
        forces = [f['force'] for f in self.force_measurements]
        distances = [f['distance'] for f in self.force_measurements]
        
        return {
            'total_measurements': len(self.force_measurements),
            'force': {
                'mean': float(np.mean(forces)),
                'std': float(np.std(forces)),
                'min': float(np.min(forces)),
                'max': float(np.max(forces))
            },
            'distance': {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances))
            }
        }
    
    def get_G_evolution(self) -> Dict:
        """Get time evolution of measured gravitational constant"""
        if not self.G_measurements:
            return {}
        
        times = [m['time'] for m in self.G_measurements]
        Gs = [m['G'] for m in self.G_measurements]
        ns = [m['n'] for m in self.G_measurements]
        
        return {
            'times': times,
            'G_values': Gs,
            'power_law_exponents': ns,
            'mean_G': float(np.mean(Gs)),
            'std_G': float(np.std(Gs)),
            'mean_n': float(np.mean(ns)),
            'std_n': float(np.std(ns))
        }
