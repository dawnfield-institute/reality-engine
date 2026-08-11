"""
Atom Detector - Identify atomic-like stable structures.

Looks for:
- Stable, coherent structures (like atoms)
- Mass quantization patterns
- Electron shell-like configurations
- Periodic table emergence
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from base_analyzer import BaseAnalyzer, Detection


class AtomDetector(BaseAnalyzer):
    """
    Detects atomic-like structures in emergent matter.
    
    Identifies stable structures that might correspond to
    atoms, molecules, or other fundamental matter units.
    """
    
    def __init__(self, min_confidence: float = 0.6):
        super().__init__("atom_detector", min_confidence)
        
        # Track mass distribution to find quantization
        self.mass_histogram = defaultdict(int)
        self.stable_structures = []
        
    def analyze(self, state: Dict) -> List[Detection]:
        """Detect atom-like structures"""
        detections = []
        
        if 'structures' not in state:
            return detections
        
        structures = state['structures']
        current_time = state.get('time', 0.0)
        
        # Analyze each structure
        for s in structures:
            # Record mass for quantization analysis
            mass_bin = self._quantize_mass(s.mass)
            self.mass_histogram[mass_bin] += 1
            
            # Check if structure is stable and coherent (atom-like)
            if self._is_atomic(s):
                detection = Detection(
                    type="atomic_structure",
                    confidence=self._compute_atomic_confidence(s),
                    time=current_time,
                    location=tuple(s.center),
                    properties={
                        'mass': s.mass,
                        'mass_class': self._classify_mass(s.mass),
                        'coherence': getattr(s, 'coherence', 0.0),
                        'lifetime': getattr(s, 'lifetime', 0),
                        'radius': getattr(s, 'size', 0.0),
                        'stability': self._measure_stability(s)
                    }
                )
                detections.append(detection)
                self.stable_structures.append(s)
        
        # Look for mass quantization (periodic table signature)
        if len(self.mass_histogram) > 10 and self.step_count % 500 == 0:
            quantization = self._detect_mass_quantization()
            if quantization:
                detections.append(quantization)
        
        # Look for molecular bonds (clusters of atoms)
        if len(structures) > 1:
            molecules = self._detect_molecules(structures, current_time)
            detections.extend(molecules)
        
        return detections
    
    def _quantize_mass(self, mass: float, bin_size: float = 0.5) -> float:
        """Bin mass into discrete levels"""
        return round(mass / bin_size) * bin_size
    
    def _is_atomic(self, structure) -> bool:
        """Check if structure has atomic properties"""
        # Atoms are:
        # 1. Stable (long lifetime)
        # 2. Coherent (high coherence score)
        # 3. Localized (small size)
        # 4. Not too massive (not stars/black holes)
        
        lifetime = getattr(structure, 'lifetime', 0)
        coherence = getattr(structure, 'coherence', 0.0)
        mass = structure.mass
        
        is_stable = lifetime > 20  # Lived >20 steps
        is_coherent = coherence > 0.9  # Very coherent
        is_localized = mass < 100  # Not too massive
        is_substantial = mass > 0.01  # Not too light
        
        return is_stable and is_coherent and is_localized and is_substantial
    
    def _compute_atomic_confidence(self, structure) -> float:
        """Compute confidence that structure is atomic"""
        confidence = 0.5  # Base confidence
        
        # Increase for coherence
        coherence = getattr(structure, 'coherence', 0.0)
        if coherence > 0.95:
            confidence += 0.3
        elif coherence > 0.90:
            confidence += 0.2
        
        # Increase for stability
        lifetime = getattr(structure, 'lifetime', 0)
        if lifetime > 100:
            confidence += 0.2
        elif lifetime > 50:
            confidence += 0.1
        
        # Increase for appropriate mass
        mass = structure.mass
        if 0.1 < mass < 10:  # "Light" atoms
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _classify_mass(self, mass: float) -> str:
        """Classify structure by mass (like element groups)"""
        if mass < 0.1:
            return "ultra_light"  # Like neutrinos?
        elif mass < 1.0:
            return "light"  # Like hydrogen/helium
        elif mass < 10:
            return "medium"  # Like carbon/oxygen
        elif mass < 100:
            return "heavy"  # Like iron/uranium
        else:
            return "super_heavy"  # Beyond periodic table
    
    def _measure_stability(self, structure) -> float:
        """Measure structure stability (0-1)"""
        lifetime = getattr(structure, 'lifetime', 0)
        coherence = getattr(structure, 'coherence', 0.0)
        
        # Stability combines lifetime and coherence
        lifetime_score = min(lifetime / 100.0, 1.0)
        coherence_score = coherence
        
        return 0.5 * lifetime_score + 0.5 * coherence_score
    
    def _detect_mass_quantization(self) -> Optional[Detection]:
        """
        Detect if masses cluster at discrete values (periodic table signature).
        """
        if len(self.mass_histogram) < 10:
            return None
        
        # Get mass bins and counts
        masses = sorted(self.mass_histogram.keys())
        counts = [self.mass_histogram[m] for m in masses]
        
        # Look for peaks in distribution (quantization)
        # Real atoms have discrete masses
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        # Find significant peaks (>2 std above mean)
        peaks = []
        for m, c in zip(masses, counts):
            if c > mean_count + 2 * std_count:
                peaks.append((m, c))
        
        if len(peaks) >= 3:  # At least 3 discrete mass levels
            # Calculate spacing between peaks
            peak_masses = [p[0] for p in peaks]
            spacings = np.diff(peak_masses)
            
            # Check if spacing is relatively uniform (like atomic numbers)
            mean_spacing = np.mean(spacings)
            std_spacing = np.std(spacings)
            uniformity = 1.0 - (std_spacing / (mean_spacing + 1e-10))
            
            if uniformity > 0.5:  # Reasonably uniform spacing
                return Detection(
                    type="mass_quantization",
                    confidence=min(uniformity, 1.0),
                    time=self.step_count,
                    equation=f"Δm ≈ {mean_spacing:.2f} (uniform spacing)",
                    parameters={
                        'num_levels': len(peaks),
                        'mean_spacing': float(mean_spacing),
                        'std_spacing': float(std_spacing),
                        'uniformity': float(uniformity)
                    },
                    properties={
                        'peak_masses': [float(p[0]) for p in peaks],
                        'peak_counts': [int(p[1]) for p in peaks],
                        'resembles_periodic_table': uniformity > 0.7
                    }
                )
        
        return None
    
    def _detect_molecules(self, structures: List, current_time: float) -> List[Detection]:
        """Detect molecular bonds between atomic structures"""
        detections = []
        
        # Only consider atomic structures
        atoms = [s for s in structures if self._is_atomic(s)]
        
        if len(atoms) < 2:
            return detections
        
        # Look for bound pairs (molecules)
        for i, a1 in enumerate(atoms):
            for a2 in atoms[i+1:]:
                r1 = np.array(a1.center)
                r2 = np.array(a2.center)
                distance = np.linalg.norm(r2 - r1)
                
                # Check if distance suggests bonding
                # Bond length typically 1-3 grid units
                if 0.5 < distance < 5.0:
                    # Check if they're moving together (bonded)
                    if hasattr(a1, 'velocity') and hasattr(a2, 'velocity'):
                        v1 = np.array(a1.velocity)
                        v2 = np.array(a2.velocity)
                        vel_diff = np.linalg.norm(v2 - v1)
                        vel_avg = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0
                        
                        # Moving together = bonded
                        if vel_avg > 1e-6:
                            coherence = 1.0 - vel_diff / (vel_avg + 1e-10)
                            
                            if coherence > 0.7:  # Moving coherently
                                detections.append(Detection(
                                    type="molecular_bond",
                                    confidence=coherence,
                                    time=current_time,
                                    location=(tuple(r1), tuple(r2)),
                                    properties={
                                        'atom1_mass': a1.mass,
                                        'atom2_mass': a2.mass,
                                        'bond_length': float(distance),
                                        'bond_strength': float(coherence),
                                        'total_mass': a1.mass + a2.mass
                                    }
                                ))
        
        return detections
    
    def get_mass_distribution(self) -> Dict:
        """Get distribution of detected masses"""
        if not self.mass_histogram:
            return {}
        
        masses = sorted(self.mass_histogram.keys())
        counts = [self.mass_histogram[m] for m in masses]
        
        return {
            'masses': masses,
            'counts': counts,
            'total_structures': sum(counts),
            'num_mass_levels': len(masses),
            'mean_mass': float(np.average(masses, weights=counts)),
            'std_mass': float(np.sqrt(np.average((np.array(masses) - np.average(masses, weights=counts))**2, weights=counts)))
        }
