"""
Atomic Structure Analyzer - Detect and classify emergent atoms.

Detects stable oscillating patterns that could represent atomic structures.
Classifies by mass, quantum states, and stability.
"""
import numpy as np
from scipy import signal, ndimage
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import torch

@dataclass
class Atom:
    """Represents an emergent atomic structure."""
    element: str
    atomic_number: int
    mass: float
    position: Tuple[float, float]
    stability: float
    quantum_states: List[int]
    ionization_energy: float
    temperature: float

class AtomicAnalyzer:
    """Detect and classify atomic structures in the reality field."""
    
    # Approximate element classification by mass ranges
    ELEMENT_MAP = {
        (0, 1.5): ('H', 1),      # Hydrogen
        (1.5, 4.5): ('He', 2),    # Helium  
        (4.5, 7.5): ('Li', 3),    # Lithium
        (7.5, 10.5): ('Be', 4),   # Beryllium
        (10.5, 12.5): ('B', 5),   # Boron
        (12.5, 14.5): ('C', 6),   # Carbon
        (14.5, 16.5): ('N', 7),   # Nitrogen
        (16.5, 19.5): ('O', 8),   # Oxygen
        (19.5, 21.5): ('F', 9),   # Fluorine
        (21.5, 24.5): ('Ne', 10), # Neon
        (24.5, 28.5): ('Na', 11), # Sodium
        (28.5, 32.5): ('Mg', 12), # Magnesium
        (32.5, 36.5): ('Al', 13), # Aluminum
        (36.5, 40.5): ('Si', 14), # Silicon
    }
    
    def __init__(self, min_stability: float = 0.65):
        self.min_stability = min_stability
        
    def detect_atoms(self, state) -> List[Atom]:
        """Detect atomic structures in the field state."""
        M = state.M.cpu().numpy()
        A = state.A.cpu().numpy()
        T = state.T.cpu().numpy()
        P = state.P.cpu().numpy()
        
        # Find stable, localized structures
        atoms = []
        
        # Smooth the field slightly
        M_smooth = ndimage.gaussian_filter(M, sigma=0.5)
        
        # Find local maxima
        local_max = ndimage.maximum_filter(M_smooth, size=3)
        maxima = (M_smooth == local_max) & (M_smooth > 0.1)
        
        # Label connected regions
        labeled, num_features = ndimage.label(maxima)
        
        for i in range(1, num_features + 1):
            mask = labeled == i
            
            if mask.sum() == 0:
                continue
                
            # Get properties
            pos = ndimage.center_of_mass(M * mask)
            mass = M[mask].sum()
            
            # Check stability (P ≈ A in region)
            y, x = int(pos[0]), int(pos[1])
            y1, y2 = max(0, y-2), min(M.shape[0], y+3)
            x1, x2 = max(0, x-2), min(M.shape[1], x+3)
            
            local_P = P[y1:y2, x1:x2]
            local_A = A[y1:y2, x1:x2]
            
            if local_P.size > 0 and local_A.size > 0:
                diseq = np.abs(local_P - local_A)
                stability = 1.0 / (1.0 + diseq.mean())
            else:
                stability = 0.0
            
            if stability < self.min_stability:
                continue
            
            # Classify element
            element, atomic_number = self._classify_element(mass)
            
            # Detect quantum states from oscillation patterns
            quantum_states = self._detect_quantum_states(A, pos, mass)
            
            # Calculate ionization energy (binding strength)
            ionization = self._calculate_ionization_energy(M, T, pos)
            
            # Get local temperature
            temp = T[y1:y2, x1:x2].mean() if T[y1:y2, x1:x2].size > 0 else 0.0
            
            atom = Atom(
                element=element,
                atomic_number=atomic_number,
                mass=float(mass),
                position=pos,
                stability=float(stability),
                quantum_states=quantum_states,
                ionization_energy=float(ionization),
                temperature=float(temp)
            )
            
            atoms.append(atom)
        
        return atoms
    
    def _classify_element(self, mass: float) -> Tuple[str, int]:
        """Classify element based on mass."""
        for mass_range, (element, z) in self.ELEMENT_MAP.items():
            if mass_range[0] <= mass < mass_range[1]:
                return element, z
        
        # Unknown heavy element
        z_estimate = int(mass / 2) + 1
        return f'X{z_estimate}', z_estimate
    
    def _detect_quantum_states(self, A: np.ndarray, position: Tuple, mass: float) -> List[int]:
        """
        Detect quantum states from oscillation patterns.
        Look for standing wave patterns around the atomic center.
        """
        y, x = int(position[0]), int(position[1])
        
        # Extract radial profile
        radius = int(min(5, max(2, np.sqrt(mass))))
        y1, y2 = max(0, y-radius), min(A.shape[0], y+radius+1)
        x1, x2 = max(0, x-radius), min(A.shape[1], x+radius+1)
        
        local_field = A[y1:y2, x1:x2]
        
        if local_field.size < 4:
            return [1]  # Ground state only
        
        # Compute radial average
        center = (local_field.shape[0]//2, local_field.shape[1]//2)
        y_grid, x_grid = np.ogrid[:local_field.shape[0], :local_field.shape[1]]
        r = np.sqrt((y_grid - center[0])**2 + (x_grid - center[1])**2)
        
        # Find peaks in radial distribution (quantum levels)
        radial_profile = []
        for ri in range(radius):
            mask = (r >= ri) & (r < ri + 1)
            if mask.sum() > 0:
                radial_profile.append(np.abs(local_field[mask]).mean())
        
        if len(radial_profile) < 2:
            return [1]
        
        # Find peaks (quantum states)
        radial_profile = np.array(radial_profile)
        peaks, _ = signal.find_peaks(radial_profile, height=radial_profile.mean()*0.5)
        
        # Convert to quantum numbers (n = peak_index + 1)
        quantum_states = [int(p) + 1 for p in peaks[:3]]  # Max 3 states
        
        return quantum_states if quantum_states else [1]
    
    def _calculate_ionization_energy(self, M: np.ndarray, T: np.ndarray, 
                                     position: Tuple) -> float:
        """
        Estimate ionization energy from binding strength.
        Higher M and lower T means stronger binding.
        """
        y, x = int(position[0]), int(position[1])
        
        # Sample local region
        y1, y2 = max(0, y-1), min(M.shape[0], y+2)
        x1, x2 = max(0, x-1), min(M.shape[1], x+2)
        
        local_M = M[y1:y2, x1:x2].mean() if M[y1:y2, x1:x2].size > 0 else 0.0
        local_T = T[y1:y2, x1:x2].mean() if T[y1:y2, x1:x2].size > 0 else 0.01
        
        # Ionization energy ∝ M/T (binding vs thermal energy)
        ionization = local_M / (local_T + 0.01)
        
        return ionization

def build_periodic_table(atoms_list: List[List[Atom]]) -> Dict:
    """
    Build a periodic table from observed atoms across multiple timesteps.
    """
    periodic_table = {}
    
    for atoms in atoms_list:
        for atom in atoms:
            z = atom.atomic_number
            
            if z not in periodic_table:
                periodic_table[z] = {
                    'element': atom.element,
                    'atomic_number': z,
                    'occurrences': 0,
                    'avg_mass': 0.0,
                    'avg_stability': 0.0,
                    'avg_ionization': 0.0,
                    'avg_temperature': 0.0,
                    'quantum_states_observed': set(),
                }
            
            entry = periodic_table[z]
            n = entry['occurrences']
            
            # Update running averages
            entry['avg_mass'] = (entry['avg_mass'] * n + atom.mass) / (n + 1)
            entry['avg_stability'] = (entry['avg_stability'] * n + atom.stability) / (n + 1)
            entry['avg_ionization'] = (entry['avg_ionization'] * n + atom.ionization_energy) / (n + 1)
            entry['avg_temperature'] = (entry['avg_temperature'] * n + atom.temperature) / (n + 1)
            
            # Update quantum states
            for state in atom.quantum_states:
                entry['quantum_states_observed'].add(state)
            
            entry['occurrences'] += 1
    
    # Convert sets to sorted lists for serialization
    for z in periodic_table:
        periodic_table[z]['quantum_states_observed'] = sorted(
            list(periodic_table[z]['quantum_states_observed'])
        )
    
    return periodic_table
