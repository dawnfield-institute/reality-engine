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
    """
    Detect atomic structures in the reality field.
    
    NOTE: This analyzer is being phased out in favor of emergence_observer.py
    which discovers patterns without predefined assumptions.
    
    The ELEMENT_MAP has been removed - let the periodic table emerge naturally!
    """
    
    def __init__(self, min_stability: float = 0.65):
        self.min_stability = min_stability
        self.discovered_patterns = {}  # Track emergent pattern types
        
    def detect_atoms(self, state) -> List[Atom]:
        """Detect atomic structures in the field state."""
        # Handle both FieldState and custom state objects
        if hasattr(state, 'memory'):
            M = state.memory.cpu().numpy()
            A = state.actual.cpu().numpy()
            T = state.temperature.cpu().numpy()
            P = state.potential.cpu().numpy()
        else:
            M = state.M.cpu().numpy()
            A = state.A.cpu().numpy()
            T = state.T.cpu().numpy()
            P = state.P.cpu().numpy()
        
        # Find stable, localized structures
        atoms = []
        
        # Smooth the field slightly
        M_smooth = ndimage.gaussian_filter(M, sigma=0.5)
        
        # Find local maxima (lowered threshold for emergent structures)
        local_max = ndimage.maximum_filter(M_smooth, size=3)
        maxima = (M_smooth == local_max) & (M_smooth > 0.02)  # Lowered from 0.1
        
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
            
            # Let mass quantization emerge - no predefined elements
            # Classify by observed properties, not assumptions
            element, atomic_number = self._classify_by_observation(mass, stability)
            
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
    
    def _classify_by_observation(self, mass: float, stability: float) -> Tuple[str, int]:
        """
        Classify structures by observed properties, not predefined assumptions.
        Let natural quantization emerge from the data.
        """
        # Round mass to nearest 0.5 to find natural clustering
        mass_bin = round(mass * 2) / 2
        
        # Create pattern signature from mass
        pattern_key = f"M{mass_bin:.1f}"
        
        if pattern_key not in self.discovered_patterns:
            # New pattern discovered!
            pattern_id = len(self.discovered_patterns) + 1
            self.discovered_patterns[pattern_key] = {
                'id': pattern_id,
                'mass_range': (mass_bin - 0.25, mass_bin + 0.25),
                'observations': 0,
                'avg_stability': stability,
            }
        
        pattern = self.discovered_patterns[pattern_key]
        pattern['observations'] += 1
        
        # Update running average
        n = pattern['observations']
        pattern['avg_stability'] = (pattern['avg_stability'] * (n-1) + stability) / n
        
        # Label with discovered pattern
        return pattern_key, pattern['id']
    
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
    
    def track_atom_lifecycle(self, history: List[Dict], atom_id: Optional[int] = None) -> Dict:
        """
        Track lifecycle of atoms across simulation history.
        
        Analyzes why atoms form and disappear to identify stability issues.
        
        Args:
            history: List of simulation states (dicts with 'A', 'P', 'M', 'T' fields)
            atom_id: Specific atom to track (None = track all)
            
        Returns:
            Dictionary with lifecycle statistics and instability causes
        """
        lifecycle_data = {
            'formation_events': [],
            'dissolution_events': [],
            'average_lifetime': 0,
            'max_lifetime': 0,
            'instability_causes': {
                'thermal_fluctuation': 0,  # High T disrupts structure
                'memory_decay': 0,         # M field dissipates
                'field_divergence': 0,     # A and P separate
                'neighbor_collision': 0    # Merger with another atom
            },
            'stability_correlations': {}
        }
        
        # Track atoms across time
        atom_tracks = {}  # atom_id -> list of (step, atom_dict)
        
        for step, state_dict in enumerate(history):
            # Convert state to object-like structure
            class StateWrapper:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, torch.tensor(v) if isinstance(v, np.ndarray) else v)
            
            state = StateWrapper(state_dict)
            current_atoms = self.detect_atoms(state)
            
            # Convert atoms to dicts for tracking
            atom_dicts = []
            for atom in current_atoms:
                atom_dicts.append({
                    'element': atom.element,
                    'mass': atom.mass,
                    'position': atom.position,
                    'stability': atom.stability,
                    'temperature': atom.temperature,
                    'ionization_energy': atom.ionization_energy,
                    'memory_density': state_dict['M'][int(atom.position[0]), int(atom.position[1])],
                    'equilibrium': 1.0 - abs(state_dict['A'][int(atom.position[0]), int(atom.position[1])] - 
                                            state_dict['P'][int(atom.position[0]), int(atom.position[1])])
                })
            
            # Match atoms between steps (simple nearest neighbor)
            for atom in atom_dicts:
                matched_id = self._find_matching_atom(atom, atom_tracks, step)
                if matched_id not in atom_tracks:
                    atom_tracks[matched_id] = []
                    lifecycle_data['formation_events'].append({
                        'step': step,
                        'atom_id': matched_id,
                        'initial_mass': atom['mass'],
                        'initial_temp': atom['temperature']
                    })
                atom_tracks[matched_id].append((step, atom))
        
        # Analyze dissolution events
        for aid, track in atom_tracks.items():
            last_step = track[-1][0]
            lifetime = len(track)
            
            if last_step < len(history) - 1:
                # Atom disappeared - analyze why
                last_atom = track[-1][1]
                dissolution_cause = self._analyze_dissolution(
                    last_atom, 
                    history[last_step],
                    history[min(last_step + 1, len(history) - 1)]
                )
                
                lifecycle_data['dissolution_events'].append({
                    'step': last_step,
                    'atom_id': aid,
                    'lifetime': lifetime,
                    'cause': dissolution_cause,
                    'final_mass': last_atom['mass']
                })
                
                lifecycle_data['instability_causes'][dissolution_cause] += 1
            
            lifecycle_data['max_lifetime'] = max(lifecycle_data['max_lifetime'], lifetime)
        
        # Calculate statistics
        if atom_tracks:
            lifetimes = [len(track) for track in atom_tracks.values()]
            lifecycle_data['average_lifetime'] = float(np.mean(lifetimes))
            lifecycle_data['lifetime_std'] = float(np.std(lifetimes))
        
        # Correlate stability with field properties
        lifecycle_data['stability_correlations'] = self._compute_stability_correlations(
            atom_tracks, history
        )
        
        return lifecycle_data
    
    def _find_matching_atom(self, atom: Dict, tracks: Dict, step: int, 
                           max_distance: float = 2.0) -> int:
        """Find closest atom from previous step or assign new ID."""
        if not tracks:
            return 0
        
        min_dist = float('inf')
        best_match = None
        
        for atom_id, track in tracks.items():
            if track and track[-1][0] == step - 1:
                prev_atom = track[-1][1]
                dist = np.linalg.norm(
                    np.array(atom['position']) - np.array(prev_atom['position'])
                )
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    best_match = atom_id
        
        return best_match if best_match is not None else max(tracks.keys()) + 1
    
    def _analyze_dissolution(self, atom: Dict, state: Dict, next_state: Dict) -> str:
        """Determine primary cause of atom dissolution."""
        # Check thermal disruption
        local_temp = atom['temperature']
        if local_temp > atom['mass'] * 0.5:  # High T/M ratio
            return 'thermal_fluctuation'
        
        # Check memory decay
        if 'memory_density' in atom and atom['memory_density'] < 0.1:
            return 'memory_decay'
        
        # Check field divergence
        if 'equilibrium' in atom and atom['equilibrium'] < 0.5:
            return 'field_divergence'
        
        # Default to collision
        return 'neighbor_collision'
    
    def _compute_stability_correlations(self, tracks: Dict, history: List[Dict]) -> Dict:
        """Compute correlations between atom stability and field properties."""
        correlations = {}
        
        stable_atoms = []
        unstable_atoms = []
        
        for atom_id, track in tracks.items():
            lifetime = len(track)
            avg_props = self._average_atom_properties(track)
            
            if lifetime > 50:  # Stable threshold
                stable_atoms.append(avg_props)
            else:
                unstable_atoms.append(avg_props)
        
        if stable_atoms and unstable_atoms:
            # Compare average properties
            stable_avg = self._average_dict_list(stable_atoms)
            unstable_avg = self._average_dict_list(unstable_atoms)
            
            correlations['mass_ratio'] = stable_avg.get('mass', 0) / max(unstable_avg.get('mass', 1), 0.01)
            correlations['temp_ratio'] = stable_avg.get('temperature', 0) / max(unstable_avg.get('temperature', 1), 0.01)
            correlations['memory_ratio'] = stable_avg.get('memory_density', 0) / max(unstable_avg.get('memory_density', 1), 0.01)
        
        return correlations
    
    def _average_atom_properties(self, track: List[Tuple]) -> Dict:
        """Compute average properties over atom lifetime."""
        if not track:
            return {}
        
        props = ['mass', 'temperature', 'memory_density', 'equilibrium']
        averages = {}
        
        for prop in props:
            values = [atom.get(prop, 0) for _, atom in track if prop in atom]
            if values:
                averages[prop] = float(np.mean(values))
        
        return averages
    
    def _average_dict_list(self, dict_list: List[Dict]) -> Dict:
        """Average dictionary values across list."""
        if not dict_list:
            return {}
        
        result = {}
        all_keys = set()
        for d in dict_list:
            all_keys.update(d.keys())
        
        for key in all_keys:
            values = [d.get(key, 0) for d in dict_list if key in d]
            if values:
                result[key] = float(np.mean(values))
        
        return result

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
