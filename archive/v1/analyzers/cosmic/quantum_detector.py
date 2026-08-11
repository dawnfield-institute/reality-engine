"""
Quantum Phenomena Detector - Identifies quantum-like behaviors.

Looks for:
- Superposition (multiple states simultaneously)
- Entanglement (correlated distant structures)
- Tunneling (barrier penetration)
- Uncertainty relations (position-momentum trade-offs)
- Wave-particle duality signatures
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from ..base_analyzer import BaseAnalyzer, Detection


class QuantumDetector(BaseAnalyzer):
    """
    Detects quantum-like phenomena emerging from field dynamics.
    
    Quantum signatures:
    - Entanglement: Distant structures with correlated properties
    - Superposition: Structures in multiple semi-stable states
    - Tunneling: Passing through high-potential barriers
    - Uncertainty: ΔE·Δt or Δx·Δp ~ constant
    """
    
    def __init__(self, min_confidence: float = 0.6):
        super().__init__("quantum_detector", min_confidence)
        self.entangled_pairs = {}  # (id1, id2) -> correlation history
        self.tunneling_candidates = []
        self.uncertainty_measurements = []
        
    def analyze(self, state) -> List[Detection]:
        """Detect quantum-like phenomena."""
        detections = []
        current_step = state['step']
        structures = state.get('structures', [])
        
        if not structures or len(structures) < 2:
            return detections
        
        # Detect entanglement
        detections.extend(self._detect_entanglement(structures, current_step))
        
        # Detect superposition
        detections.extend(self._detect_superposition(structures, current_step))
        
        # Detect tunneling
        detections.extend(self._detect_tunneling(state, current_step))
        
        # Detect uncertainty relations
        if len(self.uncertainty_measurements) > 50:
            detection = self._detect_uncertainty_principle(current_step)
            if detection:
                detections.append(detection)
        
        # Detect wave-particle duality
        detections.extend(self._detect_wave_particle_duality(structures, current_step))
        
        return detections
    
    def _detect_entanglement(self, structures, step: int) -> List[Detection]:
        """
        Detect entangled pairs: distant structures with correlated properties.
        
        Entanglement signature: Properties change together despite separation.
        """
        detections = []
        
        # Check all pairs
        for i, s1 in enumerate(structures):
            for s2 in structures[i+1:]:
                id1, id2 = s1.id, s2.id
                pair_key = tuple(sorted([id1, id2]))
                
                # Compute separation
                c1 = np.array(s1.center)
                c2 = np.array(s2.center)
                separation = np.linalg.norm(c1 - c2)
                
                # Must be distant (no direct interaction)
                if separation < 5:
                    continue
                
                # Check property correlations
                mass1, mass2 = s1.mass, s2.mass
                # Use binding_energy as proxy for energy
                energy1, energy2 = s1.binding_energy, s2.binding_energy
                # Use entropy as proxy for information
                info1, info2 = s1.entropy, s2.entropy
                
                # Track correlation history
                if pair_key not in self.entangled_pairs:
                    self.entangled_pairs[pair_key] = {
                        'mass_pairs': [],
                        'energy_pairs': [],
                        'info_pairs': [],
                        'separations': []
                    }
                
                history = self.entangled_pairs[pair_key]
                history['mass_pairs'].append((mass1, mass2))
                history['energy_pairs'].append((energy1, energy2))
                history['info_pairs'].append((info1, info2))
                history['separations'].append(separation)
                
                # Need history to detect correlation
                if len(history['mass_pairs']) < 20:
                    continue
                
                # Compute correlations
                mass_corr = self._compute_correlation(history['mass_pairs'])
                energy_corr = self._compute_correlation(history['energy_pairs'])
                info_corr = self._compute_correlation(history['info_pairs'])
                
                # High correlation despite distance = entanglement
                mean_corr = np.mean([abs(mass_corr), abs(energy_corr), abs(info_corr)])
                
                if mean_corr > 0.7 and separation > 10:
                    confidence = min(0.95, 0.5 + 0.4 * mean_corr)
                    
                    detections.append(Detection(
                        type="quantum_entanglement",
                        confidence=confidence,
                        location=tuple((c1 + c2) / 2),
                        time=step,
                        properties={
                            'structure_ids': [id1, id2],
                            'separation': separation,
                            'mass_correlation': mass_corr,
                            'energy_correlation': energy_corr,
                            'info_correlation': info_corr,
                            'mean_correlation': mean_corr
                        }
                    ))
        
        return detections
    
    def _compute_correlation(self, pairs: List[Tuple[float, float]]) -> float:
        """Compute Pearson correlation coefficient for paired measurements."""
        if len(pairs) < 2:
            return 0.0
        
        x = np.array([p[0] for p in pairs[-30:]])  # Recent history
        y = np.array([p[1] for p in pairs[-30:]])
        
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            return 0.0
        
        return np.corrcoef(x, y)[0, 1]
    
    def _detect_superposition(self, structures, step: int) -> List[Detection]:
        """
        Detect superposition: structures oscillating between multiple states.
        
        Signature: Regular switching between distinct property values.
        """
        detections = []
        
        for s in structures:
            sid = s.id
            
            # Need structure history
            if not hasattr(self, 'structure_history'):
                self.structure_history = {}
            
            if sid not in self.structure_history:
                self.structure_history[sid] = {
                    'energy_history': [],
                    'mass_history': [],
                    'coherence_history': []
                }
            
            history = self.structure_history[sid]
            history['energy_history'].append(s.binding_energy)
            history['mass_history'].append(s.mass)
            history['coherence_history'].append(s.coherence)
            
            if len(history['energy_history']) < 50:
                continue
            
            # Look for bimodal distribution (two preferred states)
            energy = np.array(history['energy_history'][-50:])
            
            # Detect peaks in distribution
            hist, edges = np.histogram(energy, bins=10)
            peaks = []
            for i in range(1, len(hist)-1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 3:
                    peaks.append(edges[i])
            
            # Superposition = oscillating between 2+ states
            if len(peaks) >= 2:
                # Check if oscillation is coherent (not random)
                coherence = s.coherence
                oscillation_variance = np.std(np.diff(energy))
                
                if coherence > 0.6 and oscillation_variance > 0.1:
                    confidence = min(0.9, 0.4 + 0.3 * coherence + 0.2 * len(peaks) / 3)
                    
                    detections.append(Detection(
                        type="quantum_superposition",
                        confidence=confidence,
                        location=s.center,
                        time=step,
                        properties={
                            'structure_id': sid,
                            'n_states': len(peaks),
                            'energy_states': [float(p) for p in peaks],
                            'coherence': coherence,
                            'oscillation_rate': oscillation_variance
                        }
                    ))
        
        return detections
    
    def _detect_tunneling(self, state, step: int) -> List[Detection]:
        """
        Detect quantum tunneling: structures passing through barriers.
        
        Signature: Structure crosses high-potential region without enough energy.
        """
        detections = []
        
        structures = state.get('structures', [])
        field_E = state.get('field_E')
        field_I = state.get('field_I')
        
        if field_E is None or field_I is None:
            return detections
        
        for s in structures:
            sid = s.id
            center = s.center
            energy = s.binding_energy  # Use binding_energy as proxy
            
            # Sample potential barrier around structure
            x, y = int(center[0]), int(center[1])
            h, w = field_E.shape
            
            # Look in 5x5 neighborhood
            barrier_detected = False
            max_barrier = 0
            
            for dx in [-3, -2, -1, 0, 1, 2, 3]:
                for dy in [-3, -2, -1, 0, 1, 2, 3]:
                    nx, ny = (x + dx) % h, (y + dy) % w
                    potential = field_E[nx, ny] + field_I[nx, ny]
                    
                    # High potential = barrier
                    if potential > energy * 2:
                        barrier_detected = True
                        max_barrier = max(max_barrier, potential)
            
            # Structure is near barrier but has insufficient energy
            if barrier_detected and energy < max_barrier:
                # Classical: shouldn't pass. Quantum: can tunnel.
                
                # Track if it actually crosses
                if not hasattr(self, 'tunneling_watch'):
                    self.tunneling_watch = {}
                
                if sid not in self.tunneling_watch:
                    self.tunneling_watch[sid] = {
                        'barrier_encounters': 0,
                        'positions': []
                    }
                
                watch = self.tunneling_watch[sid]
                watch['barrier_encounters'] += 1
                watch['positions'].append(center)
                
                # If structure moved significantly despite barrier
                if len(watch['positions']) > 10:
                    recent_positions = watch['positions'][-10:]
                    displacement = np.linalg.norm(
                        np.array(recent_positions[-1]) - np.array(recent_positions[0])
                    )
                    
                    if displacement > 3 and watch['barrier_encounters'] > 5:
                        confidence = min(0.85, 0.4 + 0.3 * (watch['barrier_encounters'] / 10))
                        
                        detections.append(Detection(
                            type="quantum_tunneling",
                            confidence=confidence,
                            location=center,
                            time=step,
                            properties={
                                'structure_id': sid,
                                'structure_energy': energy,
                                'barrier_height': max_barrier,
                                'barrier_ratio': max_barrier / max(energy, 1e-6),
                                'displacement': displacement
                            }
                        ))
                        
                        # Reset after detection
                        watch['barrier_encounters'] = 0
        
        return detections
    
    def _detect_uncertainty_principle(self, step: int) -> Optional[Detection]:
        """
        Detect uncertainty principle: ΔE·Δt ≥ ℏ or Δx·Δp ≥ ℏ.
        
        Check if there's a minimum uncertainty product.
        """
        if len(self.uncertainty_measurements) < 50:
            return None
        
        # Analyze recent measurements
        recent = self.uncertainty_measurements[-100:]
        products = [m['uncertainty_product'] for m in recent]
        
        mean_product = np.mean(products)
        std_product = np.std(products)
        min_product = np.min(products)
        
        # Uncertainty principle: product bounded from below
        if std_product / mean_product < 0.5 and min_product > 0:
            confidence = min(0.9, 0.5 + 0.3 * (1 - std_product / mean_product))
            
            return Detection(
                type="uncertainty_principle",
                confidence=confidence,
                location=None,
                time=step,
                equation="delta_x * delta_p >= h_eff",
                parameters={
                    'mean_product': mean_product,
                    'min_product': min_product,
                    'effective_hbar': min_product
                },
                properties={
                    'n_measurements': len(recent),
                    'relative_std': std_product / mean_product,
                    'interpretation': 'Minimum uncertainty product detected'
                }
            )
        
        return None
    
    def _detect_wave_particle_duality(self, structures, step: int) -> List[Detection]:
        """
        Detect wave-particle duality: structures showing both wave and particle behavior.
        
        Wave behavior: Interference, diffraction patterns
        Particle behavior: Localized, discrete
        """
        detections = []
        
        for s in structures:
            coherence = s.coherence
            size = s.radius  # Use radius as size
            mass = s.mass
            
            # Particle-like: localized, discrete mass
            is_particle_like = size < 3 and mass > 0.01
            
            # Wave-like: extended, coherent
            is_wave_like = coherence > 0.8 and size > 2
            
            # Duality: exhibits both properties
            if is_particle_like and is_wave_like:
                confidence = min(0.85, 0.5 + 0.3 * coherence)
                
                # De Broglie wavelength: λ = h/p
                # Estimate momentum from mass and field gradient
                wavelength_estimate = 1.0 / max(mass, 0.01)
                
                detections.append(Detection(
                    type="wave_particle_duality",
                    confidence=confidence,
                    location=s.center,
                    time=step,
                    equation="lambda = h/p (de Broglie relation)",
                    properties={
                        'structure_id': s.id,
                        'mass': mass,
                        'size': size,
                        'coherence': coherence,
                        'wavelength_estimate': wavelength_estimate,
                        'particle_score': float(is_particle_like),
                        'wave_score': coherence
                    }
                ))
        
        return detections
    
    def record_uncertainty_measurement(self, delta_x: float, delta_p: float):
        """Record position-momentum uncertainty measurement."""
        self.uncertainty_measurements.append({
            'delta_x': delta_x,
            'delta_p': delta_p,
            'uncertainty_product': delta_x * delta_p
        })
        
        # Keep only recent measurements
        if len(self.uncertainty_measurements) > 200:
            self.uncertainty_measurements = self.uncertainty_measurements[-200:]
