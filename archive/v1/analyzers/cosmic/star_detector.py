"""
Star Detector - Identifies stellar-mass objects and fusion-like processes.

Looks for:
- High-mass concentrations (stellar cores)
- Energy generation (fusion signatures)
- Stable luminous objects
- Main sequence relationships (mass-luminosity)
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from ..base_analyzer import BaseAnalyzer, Detection


class StarDetector(BaseAnalyzer):
    """
    Detects stellar-like objects: massive, stable, energy-generating structures.
    
    Stellar criteria:
    - Mass >> atomic (typically > 100 in simulation units)
    - High energy density (fusion-like activity)
    - Long lifetime (> 100 steps)
    - Radiative signature (energy outflow)
    """
    
    def __init__(self, min_confidence: float = 0.7):
        super().__init__("star_detector", min_confidence)
        self.stellar_candidates = {}  # id -> {first_seen, properties}
        self.fusion_events = []  # Track energy generation
        self.mass_luminosity_data = []  # For H-R diagram equivalent
        
    def analyze(self, state) -> List[Detection]:
        """Detect stellar objects and fusion processes."""
        detections = []
        current_step = state['step']
        structures = state.get('structures', [])
        
        if not structures:
            return detections
        
        # Update stellar candidates
        self._update_candidates(structures, current_step)
        
        # Detect stellar objects
        detections.extend(self._detect_stars(structures, current_step))
        
        # Detect fusion-like processes
        detections.extend(self._detect_fusion(state, current_step))
        
        # Detect main sequence relationship
        if len(self.mass_luminosity_data) > 20:
            detection = self._detect_main_sequence(current_step)
            if detection:
                detections.append(detection)
        
        # Detect stellar evolution
        detections.extend(self._detect_evolution(structures, current_step))
        
        return detections
    
    def _update_candidates(self, structures, step: int):
        """Track structures that might be stellar."""
        for s in structures:
            sid = s.id
            mass = s.mass
            
            # Stellar mass threshold
            if mass > 100:
                if sid not in self.stellar_candidates:
                    self.stellar_candidates[sid] = {
                        'first_seen': step,
                        'mass_history': [],
                        'energy_history': [],
                        'luminosity_history': []
                    }
                
                candidate = self.stellar_candidates[sid]
                candidate['mass_history'].append(mass)
                
                # Estimate energy and luminosity from available properties
                # Use binding_energy as proxy for internal energy
                energy = s.binding_energy
                # Use entropy as proxy for information content
                info = s.entropy
                candidate['energy_history'].append(energy)
                candidate['luminosity_history'].append(energy + info)
    
    def _detect_stars(self, structures, step: int) -> List[Detection]:
        """Identify stellar objects based on mass, stability, and energy."""
        detections = []
        
        for s in structures:
            sid = s.id
            if sid not in self.stellar_candidates:
                continue
            
            candidate = self.stellar_candidates[sid]
            lifetime = step - candidate['first_seen']
            
            if lifetime < 100:  # Need long observation
                continue
            
            mass = s.mass
            coherence = s.coherence
            
            # Stellar criteria
            is_massive = mass > 100
            is_stable = lifetime > 100 and coherence > 0.85
            
            mass_history = candidate['mass_history']
            is_mass_stable = len(mass_history) > 10 and \
                           np.std(mass_history[-20:]) / np.mean(mass_history[-20:]) < 0.2
            
            energy_history = candidate['energy_history']
            is_energetic = len(energy_history) > 10 and \
                          np.mean(energy_history[-20:]) > 0.5
            
            if is_massive and is_stable and is_mass_stable and is_energetic:
                confidence = min(0.95, 0.5 + 0.1 * (lifetime / 100) + 0.2 * coherence)
                
                # Classify star type by mass
                if mass > 10000:
                    star_type = "supergiant"
                elif mass > 1000:
                    star_type = "giant"
                elif mass > 100:
                    star_type = "main_sequence"
                else:
                    star_type = "dwarf"
                
                luminosity = np.mean(candidate['luminosity_history'][-10:]) if candidate['luminosity_history'] else 0
                
                detections.append(Detection(
                    type="stellar_object",
                    confidence=confidence,
                    location=s.center,
                    time=step,
                    properties={
                        'mass': mass,
                        'lifetime': lifetime,
                        'coherence': coherence,
                        'star_type': star_type,
                        'luminosity': luminosity,
                        'stable_mass': is_mass_stable
                    }
                ))
        
        return detections
    
    def _detect_fusion(self, state, step: int) -> List[Detection]:
        """Detect fusion-like energy generation processes."""
        detections = []
        
        structures = state.get('structures', [])
        
        for s in structures:
            sid = s.id
            if sid not in self.stellar_candidates:
                continue
            
            candidate = self.stellar_candidates[sid]
            energy_history = candidate['energy_history']
            
            if len(energy_history) < 20:
                continue
            
            # Look for sustained energy generation
            recent_energy = energy_history[-10:]
            baseline_energy = energy_history[-20:-10]
            
            energy_increase = np.mean(recent_energy) - np.mean(baseline_energy)
            is_generating = energy_increase > 0.1 and np.mean(recent_energy) > 0.5
            
            # Look for mass loss (fusion converts mass to energy)
            mass_history = candidate['mass_history']
            if len(mass_history) > 20:
                recent_mass = mass_history[-10:]
                baseline_mass = mass_history[-20:-10]
                mass_decrease = np.mean(baseline_mass) - np.mean(recent_mass)
                
                # Fusion signature: energy increase + mass decrease
                if is_generating and mass_decrease > 0:
                    confidence = min(0.9, 0.5 + 0.3 * (energy_increase / 1.0) + 0.1 * (mass_decrease / 10))
                    
                    # Record mass-luminosity data for H-R diagram
                    luminosity = np.mean(recent_energy)
                    mass = s.mass
                    self.mass_luminosity_data.append((mass, luminosity))
                    
                    detections.append(Detection(
                        type="fusion_process",
                        confidence=confidence,
                        location=s.center,
                        time=step,
                        equation="m → E (mass-energy conversion)",
                        properties={
                            'energy_increase': energy_increase,
                            'mass_decrease': mass_decrease,
                            'luminosity': luminosity,
                            'efficiency': energy_increase / max(mass_decrease, 1e-6)
                        }
                    ))
        
        return detections
    
    def _detect_main_sequence(self, step: int) -> Optional[Detection]:
        """Detect main sequence mass-luminosity relationship (L ∝ M^α)."""
        if len(self.mass_luminosity_data) < 20:
            return None
        
        # Recent data only
        recent_data = self.mass_luminosity_data[-100:]
        masses = np.array([d[0] for d in recent_data])
        luminosities = np.array([d[1] for d in recent_data])
        
        # Filter valid data
        valid = (masses > 100) & (luminosities > 0.1)
        if np.sum(valid) < 20:
            return None
        
        masses = masses[valid]
        luminosities = luminosities[valid]
        
        # Fit power law: L = k * M^α (in log space: log(L) = log(k) + α*log(M))
        log_masses = np.log(masses)
        log_luminosities = np.log(luminosities)
        
        # Linear regression
        A = np.vstack([log_masses, np.ones(len(log_masses))]).T
        alpha, log_k = np.linalg.lstsq(A, log_luminosities, rcond=None)[0]
        
        # Predict and compute R²
        predicted = alpha * log_masses + log_k
        ss_res = np.sum((log_luminosities - predicted) ** 2)
        ss_tot = np.sum((log_luminosities - np.mean(log_luminosities)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        if r_squared > 0.5:  # Reasonable correlation
            confidence = min(0.95, r_squared)
            
            # Real stars: L ∝ M^3.5 for main sequence
            comparison = f"Real main sequence: α ≈ 3.5, Simulation: α = {alpha:.2f}"
            
            return Detection(
                type="main_sequence_relationship",
                confidence=confidence,
                location=None,
                time=step,
                equation=f"L = k * M^{alpha:.2f}",
                parameters={'alpha': alpha, 'k': np.exp(log_k), 'r_squared': r_squared},
                properties={
                    'n_stars': len(masses),
                    'mass_range': (float(np.min(masses)), float(np.max(masses))),
                    'luminosity_range': (float(np.min(luminosities)), float(np.max(luminosities))),
                    'comparison': comparison
                }
            )
        
        return None
    
    def _detect_evolution(self, structures, step: int) -> List[Detection]:
        """Detect stellar evolution events (mass gain/loss, explosions)."""
        detections = []
        
        for s in structures:
            sid = s.id
            if sid not in self.stellar_candidates:
                continue
            
            candidate = self.stellar_candidates[sid]
            mass_history = candidate['mass_history']
            
            if len(mass_history) < 30:
                continue
            
            recent_mass = mass_history[-5:]
            baseline_mass = mass_history[-30:-5]
            
            mean_recent = np.mean(recent_mass)
            mean_baseline = np.mean(baseline_mass)
            
            # Detect rapid mass loss (supernova-like)
            if mean_baseline > 500 and mean_recent < mean_baseline * 0.5:
                mass_loss_rate = (mean_baseline - mean_recent) / 5
                confidence = min(0.9, 0.6 + 0.3 * (mass_loss_rate / mean_baseline))
                
                detections.append(Detection(
                    type="stellar_explosion",
                    confidence=confidence,
                    location=s.center,
                    time=step,
                    properties={
                        'initial_mass': mean_baseline,
                        'final_mass': mean_recent,
                        'mass_lost': mean_baseline - mean_recent,
                        'loss_rate': mass_loss_rate
                    }
                ))
            
            # Detect mass accretion
            elif mean_recent > mean_baseline * 1.5:
                accretion_rate = (mean_recent - mean_baseline) / 5
                confidence = min(0.85, 0.5 + 0.2 * (accretion_rate / mean_baseline))
                
                detections.append(Detection(
                    type="mass_accretion",
                    confidence=confidence,
                    location=s.center,
                    time=step,
                    properties={
                        'initial_mass': mean_baseline,
                        'final_mass': mean_recent,
                        'mass_gained': mean_recent - mean_baseline,
                        'accretion_rate': accretion_rate
                    }
                ))
        
        return detections
    
    def get_hr_diagram_data(self) -> Dict:
        """Get mass-luminosity data for Hertzsprung-Russell diagram equivalent."""
        if not self.mass_luminosity_data:
            return {}
        
        masses = [d[0] for d in self.mass_luminosity_data]
        luminosities = [d[1] for d in self.mass_luminosity_data]
        
        return {
            'masses': masses,
            'luminosities': luminosities,
            'n_points': len(masses)
        }
