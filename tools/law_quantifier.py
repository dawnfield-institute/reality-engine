"""
Law Quantifier - Measure emergent laws and compare to known physics.

Quantifies forces, conservation laws, symmetries, and other physical behaviors
emerging from field dynamics. Compares to known laws without assuming them.
"""

import numpy as np
from scipy import optimize, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch

@dataclass
class EmergentLaw:
    """Represents a discovered physical law or regularity."""
    name: str
    type: str  # 'force', 'conservation', 'symmetry', 'relation'
    equation: str  # Mathematical form discovered
    parameters: Dict[str, float]  # Measured constants
    confidence: float  # Statistical confidence
    known_match: Optional[str]  # Matches known law?
    deviation: float  # How much it deviates from known law
    observations: int  # Number of data points

class LawQuantifier:
    """
    Measure and classify emergent physical laws from field evolution.
    
    No assumptions - measure what IS, then check if it matches known physics.
    """
    
    def __init__(self):
        self.measurements = []
        self.discovered_laws = []
        
    def measure_conservation_laws(self, trajectory: List[Dict]) -> Dict[str, EmergentLaw]:
        """
        Test what quantities are conserved over time.
        
        Tests for:
        - Energy conservation
        - Momentum conservation
        - Angular momentum
        - Novel conserved quantities
        """
        conservation = {}
        
        # Track various quantities
        quantities = {
            'total_actual': [],           # A (energy/actual)
            'total_potential': [],        # P (information/potential)
            'total_memory': [],           # M (memory)
            'pac_functional': [],         # P + A + C (should be conserved!)
            'energy_plus_info': [],       # E + I (fundamental equivalence)
            'disequilibrium': [],         # |E - I| (drives dynamics)
            'entropy': [],
            'pac_lambda_qpl': [],         # λ·QPL term from QBE
            'qpl_phase': [],              # QPL(t) phase for QBE constraint
        }
        
        for state in trajectory:
            # Measure quantities without assuming what's conserved
            if 'A' in state and 'P' in state and 'M' in state:
                A = state['A'] if isinstance(state['A'], np.ndarray) else state['A']
                P = state['P'] if isinstance(state['P'], np.ndarray) else state['P']
                M = state['M'] if isinstance(state['M'], np.ndarray) else state['M']
                
                # Dawn Field Theory quantities
                quantities['total_actual'].append(float(np.sum(A)))
                quantities['total_potential'].append(float(np.sum(P)))
                quantities['total_memory'].append(float(np.sum(M)))
                
                # PAC functional: P + A + C (where C includes M contribution)
                # This should be conserved in Dawn Field Theory!
                pac_value = float(np.sum(P) + np.sum(A) + 0.964 * np.sum(M))  # α = 0.964
                quantities['pac_functional'].append(pac_value)
                
                # Energy-Information equivalence: E + I should be conserved
                # In Dawn Field: E ≡ A (actual), I ≡ P (potential)
                e_plus_i = float(np.sum(A) + np.sum(P))
                quantities['energy_plus_info'].append(e_plus_i)
                
                # Disequilibrium: drives collapse and memory formation
                diseq = float(np.sum(np.abs(A - P)))
                quantities['disequilibrium'].append(diseq)
                
                # QPL term (quantum phase locking) - related to λ
                qpl = float(np.sum(M * np.abs(A - P)))  # Information-memory correlation
                quantities['pac_lambda_qpl'].append(0.020 * qpl)  # λ = 0.020 Hz
                
                # QPL phase (for QBE constraint testing)
                if 'qpl_phase' in state:
                    quantities['qpl_phase'].append(float(state['qpl_phase']))
                
                # Entropy calculation
                A_flat = A.flatten()
                A_normalized = np.abs(A_flat) / (np.sum(np.abs(A_flat)) + 1e-10)
                entropy = -np.sum(A_normalized * np.log(A_normalized + 1e-10))
                quantities['entropy'].append(float(entropy))
        
        # Test each quantity for conservation
        for name, values in quantities.items():
            if len(values) > 10:
                law = self._test_conservation(name, values)
                if law and law.confidence > 0.75:  # 75% confidence threshold
                    conservation[name] = law
        
        # Dawn Field Theory specific: test QBE constraint and E-I complementarity
        # QBE: dI/dt + dE/dt = λ·QPL
        # Note: We're measuring P (potential) and A (actual), where I=P and E=A
        if 'total_actual' in quantities and 'total_potential' in quantities and 'qpl_phase' in quantities:
            A_vals = np.array(quantities['total_actual'])
            P_vals = np.array(quantities['total_potential'])
            qpl_vals = np.array(quantities['qpl_phase'])
            
            if len(A_vals) > 20 and len(P_vals) > 20 and len(qpl_vals) > 20:
                # Compute rates
                dA_dt = np.diff(A_vals)
                dP_dt = np.diff(P_vals)
                qbe_sum = dA_dt + dP_dt
                
                # λ·QPL(t) - the right-hand side of QBE
                # λ=0.020, QPL is already computed
                lambda_val = 0.020
                qpl_aligned = qpl_vals[:len(qbe_sum)]  # Align lengths
                expected_sum = lambda_val * qpl_aligned
                
                # Test if qbe_sum ≈ expected_sum
                residual = qbe_sum - expected_sum
                residual_mean = np.mean(np.abs(residual))
                residual_std = np.std(residual)
                
                # Scale residual by magnitude of expected_sum
                expected_scale = np.mean(np.abs(expected_sum)) + 1e-10
                normalized_residual = residual_mean / expected_scale
                
                print(f"    [DEBUG] QBE: dA/dt + dP/dt mean={np.mean(qbe_sum):.4f}, expected lambda·QPL mean={np.mean(expected_sum):.4f}")
                print(f"    [DEBUG] QBE residual: mean={residual_mean:.4f}, normalized={normalized_residual:.4f}")
                
                # QBE is satisfied if normalized residual is small
                qbe_satisfaction = max(0.0, 1.0 - normalized_residual)
                
                if qbe_satisfaction > 0.50:
                    qbe_law = EmergentLaw(
                        name="QBE_constraint",
                        type="quantum",
                        equation="dA/dt + dP/dt = λ·QPL(t)",
                        parameters={
                            'lambda': lambda_val,
                            'satisfaction': float(qbe_satisfaction),
                            'residual': float(residual_mean)
                        },
                        confidence=float(qbe_satisfaction),
                        known_match="quantum_balance_equation",
                        deviation=float(normalized_residual),
                        observations=len(qbe_sum)
                    )
                    conservation['QBE'] = qbe_law
                    print(f"    [DEBUG] QBE law detected! Confidence: {qbe_satisfaction:.3f}")
        
        # Test E-I equivalence: E + I should be roughly conserved
        if 'energy_plus_info' in quantities and len(quantities['energy_plus_info']) > 10:
            ei_values = quantities['energy_plus_info']
            ei_mean = np.mean(ei_values)
            ei_std = np.std(ei_values)
            ei_var = ei_std / (abs(ei_mean) + 1e-10)
            
            print(f"    [DEBUG] E+I: mean={ei_mean:.2f}, std={ei_std:.2f}, var={ei_var:.4f}")
            
            # If E+I oscillates around zero, check if it's bounded
            if abs(ei_mean) < 1.0:  # Near zero mean
                # Check if it's oscillating in a bounded range
                ei_range = np.max(ei_values) - np.min(ei_values)
                is_bounded = ei_range < 100  # Reasonable bound
                
                if is_bounded:
                    ei_law = EmergentLaw(
                        name="E_plus_I_bounded",
                        type="conservation",
                        equation="E + I ≈ 0 (complementary)",
                        parameters={'mean': float(ei_mean), 'range': float(ei_range)},
                        confidence=0.85,
                        known_match='energy_information_equivalence',
                        deviation=float(ei_std),
                        observations=len(ei_values)
                    )
                    conservation['E+I'] = ei_law
                    print(f"    [DEBUG] E+I complementarity detected (bounded oscillation)")
            elif ei_var < 0.15:  # Low variation
                ei_law = self._test_conservation('E_plus_I_equivalence', ei_values)
                if ei_law and ei_law.confidence > 0.70:
                    ei_law.known_match = 'energy_information_equivalence'
                    conservation['E+I'] = ei_law
                    print(f"    [DEBUG] E+I conservation detected: {ei_law.confidence:.3f}")
        
        # PAC functional conservation
        if 'pac_functional' in quantities and len(quantities['pac_functional']) > 10:
            pac_values = quantities['pac_functional']
            pac_mean = np.mean(pac_values)
            pac_std = np.std(pac_values)
            pac_var = pac_std / (abs(pac_mean) + 1e-10)
            
            print(f"    [DEBUG] PAC: mean={pac_mean:.2f}, std={pac_std:.2f}, var={pac_var:.4f}")
            
            if pac_var < 0.15:  # Less than 15% variation
                pac_law = self._test_conservation('PAC_conservation', pac_values)
                if pac_law and pac_law.confidence > 0.70:
                    pac_law.known_match = 'pac_functional'
                    conservation['PAC'] = pac_law
                    print(f"    [DEBUG] PAC conservation detected: {pac_law.confidence:.3f}")
        
        # Test quantum-like behavior: complementarity between A and P
        quantum_laws = self._test_quantum_behavior(quantities)
        conservation.update(quantum_laws)
        
        return conservation
    
    def measure_symmetries(self, field_state) -> Dict[str, EmergentLaw]:
        """
        Detect symmetries in the field configuration.
        
        Tests for:
        - Translational symmetry
        - Rotational symmetry
        - Scale invariance
        - Time reversal symmetry
        - Novel symmetries
        """
        symmetries = {}
        
        # Extract field array
        if hasattr(field_state, 'actual'):
            A = field_state.actual
        elif hasattr(field_state, 'A'):
            A = field_state.A
        else:
            return symmetries
            
        if torch.is_tensor(A):
            A = A.cpu().numpy()
        
        # Test translational symmetry
        trans_sym = self._test_translational_symmetry(A)
        if trans_sym.confidence > 0.7:
            symmetries['translational'] = trans_sym
        
        # Test scale invariance
        scale_sym = self._test_scale_invariance(A)
        if scale_sym.confidence > 0.5:
            symmetries['scale'] = scale_sym
        
        return symmetries
    
    def measure_thermodynamic_laws(self, trajectory: List[Dict]) -> Dict[str, EmergentLaw]:
        """
        Measure emergent thermodynamic behaviors.
        
        Tests for:
        - Temperature-energy relations
        - Entropy laws
        - Heat flow
        - Phase transitions
        """
        thermo_laws = {}
        
        # Extract thermodynamic quantities
        temps = []
        entropies = []
        energies = []
        
        for state in trajectory:
            if 'temperature' in state:
                temps.append(float(state['temperature']))
            if 'entropy' in state:
                entropies.append(float(state['entropy']))
            if 'total_energy' in state:
                energies.append(float(state['total_energy']))
        
        # Test entropy increase (2nd law)
        if len(entropies) > 10:
            entropy_law = self._test_entropy_law(entropies)
            if entropy_law:
                thermo_laws['entropy'] = entropy_law
        
        # Test temperature-energy relation
        if len(temps) > 10 and len(energies) > 10:
            state_eq = self._fit_state_equation(temps, energies)
            if state_eq:
                thermo_laws['state_equation'] = state_eq
        
        return thermo_laws
    
    def measure_force_laws(self, trajectory: List[Dict]) -> Dict[str, EmergentLaw]:
        """
        Measure force-like behaviors between structures.
        
        In Dawn Field Theory, gravity emerges from recursive information density,
        not classical pairwise forces. We measure:
        1. Recursive gravity (information density gradients)
        2. Classical pairwise forces (for comparison)
        """
        forces = {}
        
        # PRIORITY: Detect recursive gravity (Dawn Field Theory approach)
        recursive_gravity = self._detect_recursive_gravity(trajectory)
        if recursive_gravity:
            forces['recursive_gravity'] = recursive_gravity
            forces['universal'] = recursive_gravity  # This IS the universal force
        
        # ALSO: Try classical pairwise force detection (for comparison)
        structure_trajectories = self._extract_structure_trajectories(trajectory)
        
        if len(structure_trajectories) >= 2:
            pairwise_count = 0
            for i, s1 in enumerate(structure_trajectories):
                for j, s2 in enumerate(structure_trajectories[i+1:], i+1):
                    if pairwise_count >= 5:  # Limit to first 5 pairs
                        break
                    force_profile = self._measure_pairwise_force(s1, s2)
                    
                    if force_profile and force_profile['observations'] > 10:
                        law = self._fit_force_law(force_profile)
                        if law:
                            forces[f"force_{i}_{j}"] = law
                            pairwise_count += 1
                if pairwise_count >= 5:
                    break
            
            # Look for universal classical force law
            if forces and 'universal' not in forces:
                universal_law = self._find_universal_force(forces)
                if universal_law:
                    forces['classical_universal'] = universal_law
        
        return forces
    
    def _test_conservation(self, name: str, values: List[float]) -> Optional[EmergentLaw]:
        """Test if a quantity is conserved."""
        if len(values) < 2:
            return None
        
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Relative variation
        if abs(mean_val) > 1e-10:
            relative_variation = std_val / abs(mean_val)
        else:
            relative_variation = std_val
        
        # Check if it's constant (conserved)
        is_conserved = relative_variation < 0.1  # Less than 10% variation
        
        # Check if it's linearly changing (like entropy)
        x = np.arange(len(values))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            is_linear = abs(r_value) > 0.90
        except:
            is_linear = False
            slope = 0
            r_value = 0
            std_err = 1
        
        if is_conserved:
            return EmergentLaw(
                name=f"{name}_conservation",
                type="conservation",
                equation=f"{name} = constant",
                parameters={'value': float(mean_val), 'variation': float(relative_variation)},
                confidence=1.0 - relative_variation,
                known_match="energy" if "energy" in name else None,
                deviation=relative_variation,
                observations=len(values)
            )
        elif is_linear and abs(slope) > 1e-10:
            return EmergentLaw(
                name=f"{name}_growth",
                type="relation",
                equation=f"d{name}/dt = {slope:.3e}",
                parameters={'rate': float(slope), 'r_squared': float(r_value**2)},
                confidence=abs(r_value),
                known_match="2nd_law" if slope > 0 and "entropy" in name else None,
                deviation=float(std_err),
                observations=len(values)
            )
        else:
            return None
    
    def _find_novel_conservation(self, quantities: Dict[str, List]) -> Dict[str, EmergentLaw]:
        """Look for combinations of quantities that are conserved."""
        novel = {}
        
        # Already handled by main conservation test
        return novel
    
    def _test_quantum_behavior(self, quantities: Dict[str, List]) -> Dict[str, EmergentLaw]:
        """
        Test for quantum-like behaviors in the field dynamics.
        
        In Dawn Field Theory:
        - A (actual) and P (potential) should show complementarity
        - Uncertainty-like relations: ΔA·ΔP ≥ constant
        - Wave-particle duality: structures oscillate between A and P
        """
        quantum_laws = {}
        
        if 'total_actual' not in quantities or 'total_potential' not in quantities:
            return quantum_laws
        
        A_values = np.array(quantities['total_actual'])
        P_values = np.array(quantities['total_potential'])
        
        if len(A_values) < 10 or len(P_values) < 10:
            return quantum_laws
        
        # Test complementarity: when A increases, P should decrease (and vice versa)
        dA = np.diff(A_values)
        dP = np.diff(P_values)
        
        # Correlation: negative correlation suggests complementarity
        if len(dA) > 5 and len(dP) > 5:
            correlation = np.corrcoef(dA, dP)[0, 1]
            
            # Negative correlation = complementary behavior
            if correlation < -0.3:  # Anti-correlated
                complementarity = EmergentLaw(
                    name="actual_potential_complementarity",
                    type="quantum",
                    equation="dA/dt · dP/dt < 0 (complementary)",
                    parameters={'correlation': float(correlation)},
                    confidence=float(abs(correlation)),
                    known_match="wave_particle_duality",
                    deviation=float(1.0 - abs(correlation)),
                    observations=len(dA)
                )
                quantum_laws['complementarity'] = complementarity
        
        # Test uncertainty relation: ΔA · ΔP ≥ ℏ_eff
        std_A = np.std(A_values)
        std_P = np.std(P_values)
        uncertainty_product = std_A * std_P
        
        # Check if uncertainty product is bounded from below
        # In quantum mechanics: ΔE·ΔI ≥ ℏ
        # Here we just check if it's roughly constant (conserved lower bound)
        
        # Compute rolling uncertainty product
        window = 20
        if len(A_values) > window:
            rolling_uncertainty = []
            for i in range(len(A_values) - window):
                window_A = A_values[i:i+window]
                window_P = P_values[i:i+window]
                rolling_uncertainty.append(np.std(window_A) * np.std(window_P))
            
            rolling_uncertainty = np.array(rolling_uncertainty)
            # Check if there's a lower bound
            min_uncertainty = np.min(rolling_uncertainty)
            mean_uncertainty = np.mean(rolling_uncertainty)
            
            # If uncertainty product stays above minimum, suggests quantum bound
            if min_uncertainty > 0 and mean_uncertainty / min_uncertainty < 3.0:
                uncertainty_law = EmergentLaw(
                    name="uncertainty_relation",
                    type="quantum",
                    equation="ΔA·ΔP ≥ ℏ_eff",
                    parameters={
                        'h_eff': float(min_uncertainty),
                        'mean_product': float(mean_uncertainty)
                    },
                    confidence=float(min_uncertainty / (mean_uncertainty + 1e-10)),
                    known_match="heisenberg_uncertainty",
                    deviation=float(np.std(rolling_uncertainty) / mean_uncertainty),
                    observations=len(rolling_uncertainty)
                )
                quantum_laws['uncertainty'] = uncertainty_law
        
        # Test for oscillatory exchange (wave-like behavior)
        # A and P should oscillate: energy exchanges between actual and potential
        if len(A_values) > 20:
            # Check for periodic behavior using autocorrelation
            A_normalized = (A_values - np.mean(A_values)) / (np.std(A_values) + 1e-10)
            
            # Autocorrelation
            autocorr = np.correlate(A_normalized, A_normalized, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first peak after initial
            if len(autocorr) > 10:
                peaks = []
                for i in range(2, min(len(autocorr), 50)):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] if i+1 < len(autocorr) else False:
                        if autocorr[i] > 0.3:  # Significant peak
                            peaks.append((i, autocorr[i]))
                
                if peaks:
                    first_peak_idx, first_peak_val = peaks[0]
                    # This is the oscillation period
                    oscillation_law = EmergentLaw(
                        name="quantum_oscillation",
                        type="quantum",
                        equation="A(t) = A₀·cos(ωt), ω = 2π/T",
                        parameters={
                            'period': float(first_peak_idx),
                            'omega': float(2 * np.pi / first_peak_idx),
                            'amplitude': float(first_peak_val)
                        },
                        confidence=float(first_peak_val),
                        known_match="quantum_oscillation",
                        deviation=float(1.0 - first_peak_val),
                        observations=len(A_values)
                    )
                    quantum_laws['oscillation'] = oscillation_law
        
        return quantum_laws
    
    def _test_translational_symmetry(self, field: np.ndarray) -> EmergentLaw:
        """Test if field has translational symmetry."""
        h, w = field.shape
        
        # Divide into quadrants
        q1 = field[:h//2, :w//2]
        q2 = field[h//2:, :w//2]
        q3 = field[:h//2, w//2:]
        q4 = field[h//2:, w//2:]
        
        # Compare statistics
        means = [q1.mean(), q2.mean(), q3.mean(), q4.mean()]
        stds = [q1.std(), q2.std(), q3.std(), q4.std()]
        
        mean_var = np.std(means) / (np.mean(np.abs(means)) + 1e-10)
        std_var = np.std(stds) / (np.mean(stds) + 1e-10)
        
        confidence = 1.0 / (1.0 + mean_var + std_var)
        
        return EmergentLaw(
            name="translational_symmetry",
            type="symmetry",
            equation="T(x)ψ = ψ",
            parameters={'uniformity': float(confidence)},
            confidence=float(confidence),
            known_match="translational" if confidence > 0.8 else None,
            deviation=float(mean_var),
            observations=4
        )
    
    def _test_scale_invariance(self, field: np.ndarray) -> EmergentLaw:
        """Test if field is scale invariant (fractal-like)."""
        from scipy.ndimage import zoom
        
        # Downsample by factor of 2
        try:
            field_small = zoom(field, 0.5, order=1)
            
            # Compare statistical properties
            std_ratio = field.std() / (field_small.std() + 1e-10)
            
            # Perfect scale invariance would have std_ratio = sqrt(2) for 2x scaling
            expected_ratio = np.sqrt(2)
            deviation = abs(std_ratio - expected_ratio) / expected_ratio
            
            confidence = 1.0 / (1.0 + deviation)
            
            return EmergentLaw(
                name="scale_invariance",
                type="symmetry",
                equation="S(λ)ψ = λ^h ψ",
                parameters={'scaling_exponent': float(std_ratio)},
                confidence=float(confidence),
                known_match="scale_invariant" if confidence > 0.6 else None,
                deviation=float(deviation),
                observations=2
            )
        except:
            return EmergentLaw(
                name="scale_invariance",
                type="symmetry",
                equation="S(λ)ψ = λ^h ψ",
                parameters={'scaling_exponent': 0.0},
                confidence=0.0,
                known_match=None,
                deviation=1.0,
                observations=0
            )
    
    def _fit_state_equation(self, temps: List, energies: List) -> Optional[EmergentLaw]:
        """Try to fit a state equation relating T, E."""
        if len(temps) < 10 or len(energies) < 10:
            return None
        
        T = np.array(temps)
        E = np.array(energies)
        
        # Try E = c*T (ideal gas like)
        try:
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(T, E)
            
            confidence = abs(r_value)
            
            if confidence > 0.7:
                return EmergentLaw(
                    name="state_equation",
                    type="relation",
                    equation="E = c*T + b",
                    parameters={'c': float(slope), 'b': float(intercept)},
                    confidence=confidence,
                    known_match="ideal_gas" if confidence > 0.85 and abs(intercept) < 0.1*abs(slope) else None,
                    deviation=float(std_err),
                    observations=len(T)
                )
        except:
            pass
        
        return None
    
    def _test_entropy_law(self, entropies: List) -> Optional[EmergentLaw]:
        """Test if entropy follows 2nd law."""
        if len(entropies) < 10:
            return None
        
        # Check if entropy generally increases
        dS = np.diff(entropies)
        increasing = np.sum(dS > 0) / len(dS)
        
        # Also check overall trend
        x = np.arange(len(entropies))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, entropies)
        
        is_increasing = slope > 0 and increasing > 0.6
        
        if is_increasing:
            return EmergentLaw(
                name="entropy_increase",
                type="thermodynamic",
                equation="dS/dt ≥ 0",
                parameters={'increase_fraction': float(increasing), 'trend_slope': float(slope)},
                confidence=float(increasing),
                known_match="2nd_law" if increasing > 0.85 else None,
                deviation=1.0 - increasing,
                observations=len(entropies)
            )
        
        return None
    
    def _extract_structure_trajectories(self, trajectory: List[Dict]) -> List[Dict]:
        """Extract structure positions and properties over time using persistent IDs."""
        # Build trajectories for tracked structures using persistent_id
        structure_tracks = {}
        
        for t_idx, state in enumerate(trajectory):
            if 'structures' not in state:
                continue
            
            structures = state['structures']
            for s in structures:
                # Use persistent_id for tracking (much more reliable!)
                if hasattr(s, 'persistent_id') and s.persistent_id is not None:
                    pid = s.persistent_id
                    
                    if pid not in structure_tracks:
                        structure_tracks[pid] = {
                            'positions': [],
                            'velocities': [],
                            'accelerations': [],
                            'masses': [],
                            'times': []
                        }
                    
                    structure_tracks[pid]['positions'].append(s.center)
                    structure_tracks[pid]['masses'].append(s.mass)
                    structure_tracks[pid]['times'].append(t_idx)
                    
                    # Add velocity and acceleration if available
                    if hasattr(s, 'velocity'):
                        structure_tracks[pid]['velocities'].append(s.velocity)
                    if hasattr(s, 'acceleration'):
                        structure_tracks[pid]['accelerations'].append(s.acceleration)
        
        # Filter to structures with enough data (at least 10 observations)
        valid_tracks = []
        for pid, track in structure_tracks.items():
            if len(track['positions']) >= 10:
                track['mass'] = np.mean(track['masses'])
                track['persistent_id'] = pid
                valid_tracks.append(track)
        
        # Return top 20 longest-lived structures for analysis
        valid_tracks.sort(key=lambda t: len(t['positions']), reverse=True)
        return valid_tracks[:20]
    
    def _measure_pairwise_force(self, s1: Dict, s2: Dict) -> Optional[Dict]:
        """Measure force-like interaction between two structures."""
        distances = []
        forces = []  # F = m*a
        
        # Find overlapping time points
        times1 = set(s1['times'])
        times2 = set(s2['times'])
        common_times = sorted(times1 & times2)
        
        if len(common_times) < 10:
            return None
        
        for t in common_times:
            # Get indices for this timestep
            idx1 = s1['times'].index(t)
            idx2 = s2['times'].index(t)
            
            # Distance between structures
            pos1 = np.array(s1['positions'][idx1])
            pos2 = np.array(s2['positions'][idx2])
            r = np.linalg.norm(pos2 - pos1)
            
            if r < 0.5:  # Too close - ignore (probably merging/touching)
                continue
            
            # Use pre-computed acceleration if available (much better!)
            if s1['accelerations'] and idx1 < len(s1['accelerations']):
                acc1 = np.array(s1['accelerations'][idx1])
                force_magnitude = s1['mass'] * np.linalg.norm(acc1)
                
                distances.append(float(r))
                forces.append(float(force_magnitude))
            # Fallback: compute acceleration from position differences
            elif idx1 >= 2 and idx1 < len(s1['positions']):
                # Second derivative: d²x/dt²
                p0 = np.array(s1['positions'][idx1-2])
                p1 = np.array(s1['positions'][idx1-1])
                p2 = np.array(s1['positions'][idx1])
                
                # Acceleration (assuming dt=1 for simplicity)
                acc = p2 - 2*p1 + p0
                force_magnitude = s1['mass'] * np.linalg.norm(acc)
                
                if r > 0.5:  # Avoid division by near-zero
                    distances.append(float(r))
                    forces.append(float(force_magnitude))
        
        if len(distances) < 5:
            return None
        
        return {
            'distances': distances,
            'forces': forces,  # F = m*a
            'masses': (s1['mass'], s2['mass']),
            'observations': len(distances),
            'persistent_ids': (s1.get('persistent_id', -1), s2.get('persistent_id', -1))
        }
    
    def _fit_force_law(self, force_profile: Dict) -> Optional[EmergentLaw]:
        """Fit observed forces to known force laws."""
        if force_profile['observations'] < 5:
            return None
        
        r = np.array(force_profile['distances'])
        F = np.array(force_profile['forces'])  # Now using actual F = m*a
        m1, m2 = force_profile['masses']
        
        # Try inverse square law: F = G·m1·m2/r²
        try:
            def inverse_square(r, G):
                return G * m1 * m2 / (r**2 + 1e-10)
            
            # Fit
            popt_inv2, pcov_inv2 = optimize.curve_fit(
                inverse_square, r, F, 
                p0=[1.0],  # Initial guess for G
                maxfev=5000,
                bounds=(0, 100)  # G must be positive and reasonable
            )
            
            # Compute R² goodness of fit
            F_pred = inverse_square(r, *popt_inv2)
            ss_res = np.sum((F - F_pred)**2)
            ss_tot = np.sum((F - np.mean(F))**2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            
            # Confidence based on R²
            confidence = max(0, r_squared)
            
            if confidence > 0.4:  # Lower threshold since we're detecting weak signals
                return EmergentLaw(
                    name="inverse_square_force",
                    type="force",
                    equation="F = G·m₁·m₂/r²",
                    parameters={
                        'G': float(popt_inv2[0]),
                        'm1': float(m1),
                        'm2': float(m2),
                        'r_squared': float(r_squared)
                    },
                    confidence=float(confidence),
                    known_match="gravity_classical" if popt_inv2[0] > 0 else None,
                    deviation=float(1.0 - r_squared),
                    observations=len(r)
                )
        except Exception as e:
            # Fitting failed - try simpler linear fit in log space
            try:
                # Log-log fit: log(F) = log(G·m1·m2) - n·log(r)
                # If n ≈ 2, it's inverse square
                mask = (r > 0) & (F > 0)
                if mask.sum() < 5:
                    return None
                    
                log_r = np.log(r[mask])
                log_F = np.log(F[mask])
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_F)
                
                # slope should be -2 for inverse square
                # intercept gives log(G·m1·m2)
                exponent = -slope
                G_eff = np.exp(intercept) / (m1 * m2 + 1e-10)
                
                confidence = abs(r_value) if 1.5 < exponent < 2.5 else 0.0
                
                if confidence > 0.4:
                    return EmergentLaw(
                        name="power_law_force",
                        type="force",
                        equation=f"F = G·m₁·m₂/r^{exponent:.2f}",
                        parameters={
                            'G': float(G_eff),
                            'exponent': float(exponent),
                            'm1': float(m1),
                            'm2': float(m2),
                            'r_squared': float(r_value**2)
                        },
                        confidence=float(confidence),
                        known_match="gravity_classical" if 1.8 < exponent < 2.2 else None,
                        deviation=float(abs(exponent - 2.0)),
                        observations=len(r)
                    )
            except:
                pass
        
        return None
    
    def _find_universal_force(self, pairwise_forces: Dict) -> Optional[EmergentLaw]:
        """Check if all pairwise forces follow the same law."""
        if not pairwise_forces:
            return None
        
        # Collect force laws by type
        force_types = {}
        for name, law in pairwise_forces.items():
            if law and law.equation:
                if law.equation not in force_types:
                    force_types[law.equation] = []
                force_types[law.equation].append(law)
        
        # Find dominant force law
        if force_types:
            dominant = max(force_types.items(), key=lambda x: len(x[1]))
            if len(dominant[1]) >= 2:  # At least 2 instances
                # Average parameters
                avg_params = {}
                for param in dominant[1][0].parameters:
                    avg_params[param] = float(np.mean([
                        law.parameters[param] for law in dominant[1]
                    ]))
                
                return EmergentLaw(
                    name="universal_force",
                    type="force",
                    equation=dominant[0],
                    parameters=avg_params,
                    confidence=float(np.mean([law.confidence for law in dominant[1]])),
                    known_match=dominant[1][0].known_match,
                    deviation=float(np.mean([law.deviation for law in dominant[1]])),
                    observations=sum([law.observations for law in dominant[1]])
                )
        
        return None
    
    def _detect_recursive_gravity(self, trajectory: List[Dict]) -> Optional[EmergentLaw]:
        """
        Detect recursive gravity from Dawn Field Theory.
        
        Gravity emerges from:
        1. Information density gradients (∇ρ_I)
        2. Recursive self-similarity across scales
        3. Collapse dynamics driven by disequilibrium
        
        Signatures:
        - Structure coalescence (many → few structures over time)
        - Separation decrease (attraction)
        - Information-collapse correlation
        - Power-law scaling (recursive self-similarity)
        """
        if len(trajectory) < 20:
            return None
        
        # Track structure formation over time
        structure_counts = []
        total_masses = []
        mean_separations = []
        information_densities = []
        
        for state in trajectory:
            if 'structures' not in state:
                continue
            
            structures = state['structures']
            if not structures:
                continue
            
            # Count structures and total mass
            n_structures = len(structures)
            structure_counts.append(n_structures)
            
            total_mass = sum(s.mass for s in structures)
            total_masses.append(total_mass)
            
            # Compute mean separation (if multiple structures)
            if n_structures > 1:
                separations = []
                for i, s1 in enumerate(structures):
                    for s2 in structures[i+1:]:
                        dist = np.sqrt(
                            (s1.center[0] - s2.center[0])**2 + 
                            (s1.center[1] - s2.center[1])**2
                        )
                        separations.append(dist)
                mean_sep = np.mean(separations) if separations else 0
            else:
                mean_sep = 0
            mean_separations.append(mean_sep)
            
            # Compute information density from field
            if 'A' in state and 'P' in state:
                A = state['A']
                P = state['P']
                # Information density = |A - P| (disequilibrium)
                if hasattr(A, 'cpu'):
                    A = A.cpu().numpy()
                if hasattr(P, 'cpu'):
                    P = P.cpu().numpy()
                info_density = np.mean(np.abs(A - P))
                information_densities.append(float(info_density))
        
        if len(structure_counts) < 10:
            return None
        
        # GRAVITY SIGNATURE 1: Structures merge over time (attractive force)
        # If number decreases while mass is conserved → gravity!
        initial_count = np.mean(structure_counts[:5])
        final_count = np.mean(structure_counts[-5:])
        
        structure_coalescence = (initial_count - final_count) / (initial_count + 1e-10)
        
        # GRAVITY SIGNATURE 2: Mean separation decreases (attraction)
        valid_seps = [s for s in mean_separations if s > 0]
        if len(valid_seps) > 10:
            sep_decrease = (valid_seps[0] - valid_seps[-1]) / (valid_seps[0] + 1e-10)
        else:
            sep_decrease = 0
        
        # GRAVITY SIGNATURE 3: Information density creates attraction
        # Track structures that merge and their local info density
        if len(information_densities) > 5 and len(structure_counts) > 5:
            # Rate of structure merging
            merge_rate = -np.gradient(structure_counts[:len(information_densities)])
            
            # Also compute per-structure info density when available
            structure_info_densities = []
            merger_events = []
            
            for i, state in enumerate(trajectory[:len(information_densities)]):
                if 'structures' not in state:
                    continue
                    
                structures = state['structures']
                if not structures:
                    continue
                
                # For each structure, compute local information density
                if 'A' in state and 'P' in state:
                    A = state['A']
                    P = state['P']
                    if hasattr(A, 'cpu'):
                        A = A.cpu().numpy()
                    if hasattr(P, 'cpu'):
                        P = P.cpu().numpy()
                    
                    for s in structures:
                        # Sample info density near structure center
                        y, x = int(s.center[0]), int(s.center[1])
                        r = max(1, int(s.radius))
                        y1, y2 = max(0, y-r), min(A.shape[0], y+r+1)
                        x1, x2 = max(0, x-r), min(A.shape[1], x+r+1)
                        
                        if y2 > y1 and x2 > x1:
                            local_A = A[y1:y2, x1:x2]
                            local_P = P[y1:y2, x1:x2]
                            local_info = np.mean(np.abs(local_A - local_P))
                            structure_info_densities.append(float(local_info))
                
                # Track merger events (persistent IDs disappearing)
                if i > 0 and i < len(trajectory) - 1:
                    prev_ids = {s.persistent_id for s in trajectory[i-1].get('structures', []) 
                               if hasattr(s, 'persistent_id') and s.persistent_id is not None}
                    curr_ids = {s.persistent_id for s in structures 
                               if hasattr(s, 'persistent_id') and s.persistent_id is not None}
                    merged = len(prev_ids - curr_ids)  # IDs that disappeared
                    if merged > 0:
                        merger_events.append((i, merged, information_densities[i]))
            
            # Compute enhanced correlation:
            # 1. Global merge rate vs global info density
            min_len = min(len(merge_rate), len(information_densities))
            if min_len > 5:
                try:
                    corr_global = np.corrcoef(merge_rate[:min_len], 
                                             information_densities[:min_len])[0, 1]
                    corr_global = abs(corr_global) if not np.isnan(corr_global) else 0
                except:
                    corr_global = 0
            else:
                corr_global = 0
            
            # 2. Merger events vs info density at merge time
            corr_merger = 0
            if len(merger_events) > 3:
                merger_counts = [m[1] for m in merger_events]
                merger_infos = [m[2] for m in merger_events]
                try:
                    corr_merger = abs(np.corrcoef(merger_counts, merger_infos)[0, 1])
                    if np.isnan(corr_merger):
                        corr_merger = 0
                except:
                    corr_merger = 0
            
            # 3. High-info structures merge faster
            corr_structure = 0
            if len(structure_info_densities) > 10:
                # Structures in high-info regions should be more likely to merge
                # This is indirect - just check if high-info regions exist
                high_info_threshold = np.percentile(structure_info_densities, 75)
                high_info_frac = np.mean(np.array(structure_info_densities) > high_info_threshold)
                corr_structure = high_info_frac  # Simple proxy
            
            # Combine all three measures (weighted)
            info_gravity_correlation = (
                0.5 * corr_global +      # Primary: global correlation
                0.3 * corr_merger +      # Merger-specific correlation
                0.2 * corr_structure     # Structure-level evidence
            )
        else:
            info_gravity_correlation = 0
        
        # GRAVITY SIGNATURE 4: Recursive self-similarity
        # Check if collapse follows power law (scale-free)
        if len(structure_counts) > 20:
            # Fit power law to structure evolution
            x = np.arange(len(structure_counts))
            y = np.array(structure_counts) + 1e-10  # Avoid log(0)
            
            try:
                # Log-log fit for power law: N(t) ∝ t^α
                log_x = np.log(x[1:] + 1)  # Skip first point, avoid log(0)
                log_y = np.log(y[1:])
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
                
                if abs(r_value) > 0.5:  # Reasonable power law fit
                    power_law_exponent = slope
                    power_law_confidence = abs(r_value)
                else:
                    power_law_exponent = 0
                    power_law_confidence = 0
            except:
                power_law_exponent = 0
                power_law_confidence = 0
        else:
            power_law_exponent = 0
            power_law_confidence = 0
        
        # Combine evidence for gravity
        gravity_signatures = {
            'coalescence': max(0, structure_coalescence),
            'separation_decrease': max(0, sep_decrease),
            'info_correlation': info_gravity_correlation,
            'power_law': power_law_confidence
        }
        
        # Overall gravity confidence (weighted average)
        gravity_confidence = (
            0.4 * max(0, structure_coalescence) +      # PRIMARY: coalescence
            0.2 * max(0, sep_decrease) +               # Separation decrease
            0.2 * info_gravity_correlation +           # Info-gravity link
            0.2 * power_law_confidence                 # Scale-free behavior
        )
        
        # If we have strong evidence for gravity
        if gravity_confidence > 0.3 or structure_coalescence > 0.5:
            # Estimate effective G from coalescence rate
            if initial_count > 0 and final_count < initial_count:
                # G ∝ merge rate / time span
                time_span = len(structure_counts)
                G_eff = structure_coalescence / time_span * 1000  # Scale factor
            else:
                G_eff = 0
            
            print(f"    [DEBUG] RECURSIVE GRAVITY DETECTED!")
            print(f"      Coalescence: {structure_coalescence:.3f} ({initial_count:.0f}→{final_count:.0f} structures)")
            print(f"      Info-correlation: {info_gravity_correlation:.3f}")
            print(f"      Power-law: α={power_law_exponent:.3f}, R²={power_law_confidence:.3f}")
            print(f"      Overall confidence: {gravity_confidence:.1%}")
            
            return EmergentLaw(
                name="recursive_gravity",
                type="force",
                equation="F = G·∇(ρ_I) [recursive information gravity]",
                parameters={
                    'G_eff': float(G_eff),
                    'coalescence_rate': float(structure_coalescence),
                    'power_law_exponent': float(power_law_exponent),
                    'info_correlation': float(info_gravity_correlation),
                    'initial_structures': float(initial_count),
                    'final_structures': float(final_count)
                },
                confidence=float(gravity_confidence),
                known_match="gravity",
                deviation=float(1.0 - gravity_confidence),
                observations=len(structure_counts)
            )
        
        return None
    
    def compare_to_standard_model(self, discovered_laws: Dict[str, EmergentLaw]) -> Dict:
        """
        Compare discovered laws to standard physics.
        
        Returns:
            Comparison report with matches, deviations, and novel findings
        """
        comparison = {
            'matched_laws': [],
            'novel_laws': [],
            'deviations': {},
            'missing_laws': [],
            'summary': ''
        }
        
        # Expected laws
        expected = {
            'gravity': False,
            'pac_functional': False,  # Dawn Field Theory core
            'energy_information_equivalence': False,  # E+I conservation
            'wave_particle_duality': False,  # Quantum complementarity
            'heisenberg_uncertainty': False,  # Uncertainty relation
            'quantum_oscillation': False,  # Wave-like behavior
            '2nd_law_thermodynamics': False,
        }
        
        for name, law in discovered_laws.items():
            if law.known_match:
                comparison['matched_laws'].append({
                    'discovered': name,
                    'known': law.known_match,
                    'confidence': law.confidence,
                    'deviation': law.deviation
                })
                
                # Mark as found
                if law.known_match in expected:
                    expected[law.known_match] = True
                if law.known_match == '2nd_law':
                    expected['2nd_law_thermodynamics'] = True
                    
            else:
                comparison['novel_laws'].append({
                    'name': name,
                    'equation': law.equation,
                    'parameters': law.parameters,
                    'confidence': law.confidence
                })
        
        # Check what's missing
        for law_name, found in expected.items():
            if not found:
                comparison['missing_laws'].append(law_name)
        
        # Generate summary
        n_matched = len(comparison['matched_laws'])
        n_novel = len(comparison['novel_laws'])
        n_missing = len(comparison['missing_laws'])
        
        comparison['summary'] = (
            f"Discovered {len(discovered_laws)} laws: "
            f"{n_matched} match known physics, "
            f"{n_novel} are novel, "
            f"{n_missing} expected laws not found"
        )
        
        return comparison
