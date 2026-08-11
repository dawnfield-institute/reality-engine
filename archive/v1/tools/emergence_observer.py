"""
Emergence Pattern Observer - Detect and quantify emergent structures.

No assumptions about what "atoms" should look like.
Just observe stable patterns and classify them by their properties.
"""
import numpy as np
from scipy import signal, ndimage
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import torch

@dataclass
class EmergentStructure:
    """A stable pattern that emerges from field dynamics."""
    id: int
    mass: float  # Total information content
    center: Tuple[float, float]
    radius: float  # Spatial extent
    coherence: float  # How well E and I are aligned
    persistence: float  # How stable over time
    frequency: float  # Dominant oscillation frequency
    entropy: float  # Local entropy
    neighbors: List[int]  # Other structures within interaction range
    pattern_class: Tuple[int, int] = (0, 0)  # Emergent classification (mass_class, radius_class)
    binding_energy: float = 0.0  # Energy binding internal components
    angular_momentum: float = 0.0  # Rotational component
    charge_like: float = 0.0  # Field divergence analog
    lifetime: int = 1  # How many timesteps observed
    persistent_id: Optional[int] = None  # Tracks structure across timesteps
    velocity: Tuple[float, float] = (0.0, 0.0)  # Movement velocity
    acceleration: Tuple[float, float] = (0.0, 0.0)  # Acceleration for force measurement
    
class EmergenceObserver:
    """
    Observe and quantify emergent patterns without preconceptions.
    
    We don't know what will emerge - we just measure:
    - Where information accumulates (M field peaks)
    - How stable these accumulations are
    - What patterns they form
    - How they interact
    """
    
    def __init__(self):
        self.structure_history = []  # Track all structures over time
        self.pattern_classes = {}  # Emergent classification from observation
        self.persistent_structures = {}  # Track structures across time: {persistent_id: [structure, structure, ...]}
        self.next_persistent_id = 0  # Counter for assigning new IDs
        self.previous_structures = []  # Last timestep's structures for tracking
        
    def observe(self, state) -> List[EmergentStructure]:
        """
        Observe current field state for emergent structures.
        No assumptions - just find stable patterns.
        """
        # Get fields from state
        if hasattr(state, 'memory'):
            M = state.memory.cpu().numpy() if torch.is_tensor(state.memory) else state.memory
            E = state.actual.cpu().numpy() if torch.is_tensor(state.actual) else state.actual
            I = state.potential.cpu().numpy() if torch.is_tensor(state.potential) else state.potential
        else:
            M = state.M.cpu().numpy() if torch.is_tensor(state.M) else state.M
            E = state.A.cpu().numpy() if torch.is_tensor(state.A) else state.A
            I = state.P.cpu().numpy() if torch.is_tensor(state.P) else state.P
        
        structures = []
        
        # Find information accumulation points (no threshold assumptions!)
        # Use adaptive thresholding based on field statistics
        M_mean = M.mean()
        M_std = M.std()
        peak_threshold = M_mean + M_std  # Detect structure centers
        extent_threshold = M_mean + 0.5 * M_std  # Capture full extent
        
        # Find peaks in M field (structure centers)
        local_max = ndimage.maximum_filter(M, size=3)
        peaks = (M == local_max) & (M > peak_threshold)
        
        # Label based on extended regions (not just peaks)
        # This captures the full spatial extent of structures
        extended_regions = M > extent_threshold
        labeled, num_features = ndimage.label(extended_regions)
        
        # Label based on extended regions (not just peaks)
        # This captures the full spatial extent of structures
        extended_regions = M > extent_threshold
        labeled, num_features = ndimage.label(extended_regions)
        
        # Verify each region has a peak (is a real structure, not noise)
        for i in range(1, num_features + 1):
            mask = labeled == i
            if mask.sum() == 0:
                continue
            
            # Check if this region contains a peak
            region_has_peak = np.any(peaks & mask)
            if not region_has_peak:
                continue  # Skip noise regions without peaks
            
            # Measure properties without assumptions
            structure = self._measure_structure(i, mask, M, E, I)
            if structure:
                structures.append(structure)
        
        # Find interaction networks
        self._identify_interactions(structures)
        
        # Track structures across timesteps for persistent IDs and velocity
        structures = self._track_structures(structures)
        
        # Update emergent classification
        self._update_pattern_classes(structures)
        
        # Store for next timestep
        self.previous_structures = structures
        
        return structures
    
    def _measure_structure(self, sid: int, mask: np.ndarray, 
                          M: np.ndarray, E: np.ndarray, I: np.ndarray) -> Optional[EmergentStructure]:
        """
        Measure properties of an emergent structure.
        No assumptions about what it "should" be.
        """
        # Basic measurements
        center = ndimage.center_of_mass(M * mask)
        mass = float(M[mask].sum())
        
        # Spatial extent (radius of gyration)
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return None
            
        radius = float(np.sqrt(((y_coords - center[0])**2 + (x_coords - center[1])**2).mean()))
        
        # Coherence: How well E and I align in this region
        E_local = E[mask]
        I_local = I[mask]
        if len(E_local) > 0 and len(I_local) > 0:
            coherence = float(1.0 - np.abs(E_local - I_local).mean())
        else:
            coherence = 0.0
        
        # Persistence: How concentrated is M relative to surroundings
        y, x = int(center[0]), int(center[1])
        r = max(1, int(radius))
        y1, y2 = max(0, y-r), min(M.shape[0], y+r+1)
        x1, x2 = max(0, x-r), min(M.shape[1], x+r+1)
        
        local_region = M[y1:y2, x1:x2]
        if local_region.size > 0:
            persistence = float(mass / (local_region.sum() + 1e-10))
        else:
            persistence = 0.0
        
        # Frequency: Dominant oscillation in E field
        E_region = E[y1:y2, x1:x2]
        if E_region.size > 4:
            # Simple FFT to find dominant frequency
            try:
                fft = np.fft.fft2(E_region)
                freqs = np.abs(fft).flatten()
                frequency = float(freqs[1:].max() / (freqs.sum() + 1e-10))  # Normalized peak frequency
            except:
                frequency = 0.0
        else:
            frequency = 0.0
        
        # Local entropy
        if E_region.size > 0:
            E_flat = E_region.flatten()
            E_flat = E_flat[E_flat > 0]  # Positive values only
            if len(E_flat) > 0:
                E_norm = E_flat / E_flat.sum()
                entropy = float(-np.sum(E_norm * np.log(E_norm + 1e-10)))
            else:
                entropy = 0.0
        else:
            entropy = 0.0
        
        return EmergentStructure(
            id=sid,
            mass=mass,
            center=center,
            radius=radius,
            coherence=coherence,
            persistence=persistence,
            frequency=frequency,
            entropy=entropy,
            neighbors=[]
        )
    
    def _identify_interactions(self, structures: List[EmergentStructure]):
        """
        Identify which structures are interacting based on proximity.
        Interaction distance emerges from the structures themselves.
        """
        if len(structures) < 2:
            return
        
        # Adaptive interaction range: mean radius * 2
        mean_radius = np.mean([s.radius for s in structures])
        interaction_range = mean_radius * 2
        
        for i, s1 in enumerate(structures):
            for j, s2 in enumerate(structures[i+1:], i+1):
                dist = np.sqrt((s1.center[0] - s2.center[0])**2 + 
                              (s1.center[1] - s2.center[1])**2)
                
                if dist < interaction_range:
                    s1.neighbors.append(s2.id)
                    s2.neighbors.append(s1.id)
    
    def _update_pattern_classes(self, structures: List[EmergentStructure]):
        """
        Classify structures based on emergent properties.
        Classes emerge from clustering, not predefinition.
        """
        if not structures:
            return
        
        # Build feature vectors
        features = []
        for s in structures:
            features.append([s.mass, s.radius, s.coherence, s.frequency, s.entropy])
        
        features = np.array(features)
        
        # Simple emergent classification: group by similar properties
        # Using quantiles to create natural breaks
        for i, s in enumerate(structures):
            # Create a simple hash of quantized properties
            mass_class = int(s.mass * 10)  # Quantize mass
            radius_class = int(s.radius * 2)  # Quantize radius
            pattern_key = (mass_class, radius_class)
            
            # Store pattern class on structure
            s.pattern_class = pattern_key
            
            if pattern_key not in self.pattern_classes:
                self.pattern_classes[pattern_key] = {
                    'count': 0,
                    'avg_coherence': 0,
                    'avg_persistence': 0,
                    'avg_frequency': 0,
                }
            
            # Update running statistics
            pc = self.pattern_classes[pattern_key]
            n = pc['count']
            pc['avg_coherence'] = (pc['avg_coherence'] * n + s.coherence) / (n + 1)
            pc['avg_persistence'] = (pc['avg_persistence'] * n + s.persistence) / (n + 1)
            pc['avg_frequency'] = (pc['avg_frequency'] * n + s.frequency) / (n + 1)
            pc['count'] += 1
    
    def analyze_emergence_patterns(self, history: List[List[EmergentStructure]]) -> Dict:
        """
        Analyze what patterns have emerged over time.
        Look for:
        - Stable configurations
        - Interaction patterns
        - Phase transitions
        - Self-organization
        """
        analysis = {
            'total_structures_observed': 0,
            'unique_patterns': len(self.pattern_classes),
            'pattern_distribution': {},
            'stability_analysis': {},
            'interaction_networks': {},
            'phase_transitions': [],
        }
        
        # Track structures across time
        all_structures = []
        for structures in history:
            all_structures.extend(structures)
            analysis['total_structures_observed'] += len(structures)
        
        if not all_structures:
            return analysis
        
        # Pattern distribution
        for key, stats in self.pattern_classes.items():
            analysis['pattern_distribution'][str(key)] = {
                'occurrences': stats['count'],
                'percentage': stats['count'] / len(all_structures) * 100,
                'avg_coherence': stats['avg_coherence'],
                'avg_persistence': stats['avg_persistence'],
                'avg_frequency': stats['avg_frequency'],
            }
        
        # Stability: which patterns persist longest
        if len(history) > 1:
            persistence_times = {}
            for i in range(len(history) - 1):
                curr_structures = history[i]
                next_structures = history[i+1]
                
                for structure in curr_structures:
                    pattern = (int(structure.mass * 10), int(structure.radius * 2))
                    if pattern not in persistence_times:
                        persistence_times[pattern] = []
                    
                    # Check if similar structure exists in next timestep
                    exists_next = any(
                        abs(s.mass - structure.mass) < 0.1 and
                        np.sqrt((s.center[0] - structure.center[0])**2 + 
                               (s.center[1] - structure.center[1])**2) < structure.radius * 2
                        for s in next_structures
                    )
                    persistence_times[pattern].append(1 if exists_next else 0)
            
            # Calculate average persistence
            for pattern, times in persistence_times.items():
                if times:
                    analysis['stability_analysis'][str(pattern)] = {
                        'avg_persistence': float(np.mean(times)),
                        'total_observations': len(times),
                    }
        
        # Detect phase transitions (sudden changes in pattern distribution)
        structure_counts = [len(h) for h in history]
        if len(structure_counts) > 10:
            # Look for sudden changes
            diffs = np.diff(structure_counts)
            mean_diff = np.abs(diffs).mean()
            if mean_diff > 0:
                transitions = np.where(np.abs(diffs) > 3 * mean_diff)[0]
                
                for t in transitions:
                    analysis['phase_transitions'].append({
                        'timestep': int(t),
                        'before_count': int(structure_counts[t]),
                        'after_count': int(structure_counts[t+1]),
                        'change': int(diffs[t]),
                    })
        
        return analysis
    
    def find_molecular_patterns(self, structures: List[EmergentStructure]) -> List[Dict]:
        """
        Look for bound systems of multiple structures.
        Molecules emerge as persistent multi-structure configurations.
        """
        molecules = []
        
        # Find groups of interacting structures
        processed = set()
        for s in structures:
            if s.id in processed:
                continue
            
            if len(s.neighbors) > 0:
                # Build molecule from connected structures
                molecule_structures = [s]
                to_process = list(s.neighbors)
                processed.add(s.id)
                
                while to_process:
                    neighbor_id = to_process.pop(0)
                    if neighbor_id not in processed:
                        neighbor = next((x for x in structures if x.id == neighbor_id), None)
                        if neighbor:
                            molecule_structures.append(neighbor)
                            to_process.extend(neighbor.neighbors)
                            processed.add(neighbor_id)
                
                if len(molecule_structures) > 1:
                    # Characterize the molecule
                    total_mass = sum(s.mass for s in molecule_structures)
                    center_of_mass = np.average(
                        [s.center for s in molecule_structures],
                        weights=[s.mass for s in molecule_structures],
                        axis=0
                    )
                    
                    molecules.append({
                        'structure_count': len(molecule_structures),
                        'total_mass': float(total_mass),
                        'center': tuple(center_of_mass),
                        'binding_energy': float(np.mean([s.coherence for s in molecule_structures])),
                        'structure_ids': [s.id for s in molecule_structures],
                    })
        
        return molecules
    
    def discover_periodic_patterns(self, history: List[List[EmergentStructure]]) -> Dict:
        """
        Discover periodic patterns in emergent structures without assumptions.
        Let the periodic table emerge naturally from observed correlations.
        
        Args:
            history: List of structure observations over time
            
        Returns:
            Dictionary with discovered patterns, bonding rules, and periodicities
        """
        discovery = {
            'mass_clusters': {},      # Natural mass quantization
            'bonding_matrix': {},     # Which patterns bond with which
            'pattern_evolution': {},  # How patterns transform
            'stability_groups': {},   # Groups with similar stability
            'periodic_structure': {}  # Emergent periodic organization
        }
        
        # Flatten history for analysis
        all_structures = []
        for timestep_structures in history:
            all_structures.extend(timestep_structures)
        
        if not all_structures:
            return discovery
        
        # 1. Find natural mass quantization (clustering in mass space)
        masses = np.array([s.mass for s in all_structures])
        
        # Use histogram to find natural peaks
        hist, bin_edges = np.histogram(masses, bins=50)
        peaks_idx = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0] + 1
        
        for peak_idx in peaks_idx:
            mass_center = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
            
            # Find structures in this cluster
            in_cluster = [s for s in all_structures 
                         if bin_edges[peak_idx] <= s.mass < bin_edges[peak_idx + 1]]
            
            if len(in_cluster) > 0:
                cluster_id = len(discovery['mass_clusters'])
                discovery['mass_clusters'][cluster_id] = {
                    'mass_center': float(mass_center),
                    'mass_range': (float(bin_edges[peak_idx]), float(bin_edges[peak_idx + 1])),
                    'count': len(in_cluster),
                    'avg_coherence': float(np.mean([s.coherence for s in in_cluster])),
                    'avg_persistence': float(np.mean([s.persistence for s in in_cluster])),
                    'avg_frequency': float(np.mean([s.frequency for s in in_cluster])),
                }
        
        # 2. Build bonding matrix (which mass clusters interact)
        for timestep_structures in history:
            for s in timestep_structures:
                if not s.neighbors:
                    continue
                
                # Find mass clusters for this structure and its neighbors
                s_cluster = self._find_mass_cluster(s.mass, discovery['mass_clusters'])
                
                neighbor_structs = [n for n in timestep_structures if n.id in s.neighbors]
                for neighbor in neighbor_structs:
                    n_cluster = self._find_mass_cluster(neighbor.mass, discovery['mass_clusters'])
                    
                    if s_cluster is not None and n_cluster is not None:
                        bond_key = tuple(sorted([s_cluster, n_cluster]))
                        
                        if bond_key not in discovery['bonding_matrix']:
                            discovery['bonding_matrix'][bond_key] = {
                                'count': 0,
                                'avg_binding': 0.0,
                            }
                        
                        bond = discovery['bonding_matrix'][bond_key]
                        n = bond['count']
                        bond['avg_binding'] = (bond['avg_binding'] * n + s.coherence) / (n + 1)
                        bond['count'] += 1
        
        # 3. Find stability groups (cluster by stability properties)
        if len(all_structures) == 0:
            return discovery
            
        coherences = np.array([s.coherence for s in all_structures])
        persistences = np.array([s.persistence for s in all_structures])
        
        # Simple grouping: high/medium/low stability
        # Handle case where all values are identical
        if coherences.std() < 1e-10:
            # All structures have same coherence - put them all in medium
            medium_stability = all_structures
            high_stability = []
            low_stability = []
        else:
            high_stability = [s for s in all_structures if s.coherence > np.percentile(coherences, 75)]
            medium_stability = [s for s in all_structures 
                               if np.percentile(coherences, 25) <= s.coherence <= np.percentile(coherences, 75)]
            low_stability = [s for s in all_structures if s.coherence < np.percentile(coherences, 25)]
        
        for group_name, group_structures in [
            ('high', high_stability),
            ('medium', medium_stability),
            ('low', low_stability)
        ]:
            if group_structures:
                discovery['stability_groups'][group_name] = {
                    'count': len(group_structures),
                    'mass_range': (float(min(s.mass for s in group_structures)),
                                  float(max(s.mass for s in group_structures))),
                    'avg_coherence': float(np.mean([s.coherence for s in group_structures])),
                    'avg_persistence': float(np.mean([s.persistence for s in group_structures])),
                }
        
        # 4. Look for periodic structure (patterns in bonding behavior)
        # Group clusters by bonding characteristics
        if discovery['mass_clusters']:
            cluster_ids = sorted(discovery['mass_clusters'].keys())
            
            for cluster_id in cluster_ids:
                # Find what this cluster bonds with
                bonding_partners = []
                for bond_key, bond_info in discovery['bonding_matrix'].items():
                    if cluster_id in bond_key:
                        partner = bond_key[1] if bond_key[0] == cluster_id else bond_key[0]
                        bonding_partners.append({
                            'partner_cluster': partner,
                            'bond_strength': bond_info['avg_binding'],
                            'bond_count': bond_info['count']
                        })
                
                discovery['periodic_structure'][cluster_id] = {
                    'mass_cluster': cluster_id,
                    'bonding_partners': bonding_partners,
                    'bonding_versatility': len(bonding_partners),  # Like valence electrons
                    'properties': discovery['mass_clusters'][cluster_id]
                }
        
        return discovery
    
    def _find_mass_cluster(self, mass: float, mass_clusters: Dict) -> Optional[int]:
        """Find which mass cluster a given mass belongs to."""
        for cluster_id, cluster_info in mass_clusters.items():
            if cluster_info['mass_range'][0] <= mass < cluster_info['mass_range'][1]:
                return cluster_id
        return None
    
    def discover_natural_classifications(self, history: List[List[EmergentStructure]], 
                                       min_samples: int = 10) -> Dict:
        """
        Discover natural classifications without assuming categories.
        Uses multiple clustering methods to find consensus structure types.
        
        Returns classifications based on:
        - Physical properties (mass, radius, coherence)
        - Dynamical behavior (growth, decay, oscillation)
        - Conservation laws obeyed
        - Interaction patterns
        """
        if not history or len(history) < 5:
            return {'n_classes': 0, 'discovered_types': {}}
        
        # Collect all structure properties
        all_structures = []
        for structures in history:
            all_structures.extend(structures)
        
        if len(all_structures) < min_samples:
            return {'n_classes': 0, 'discovered_types': {}}
        
        # Build feature matrix
        features = []
        for s in all_structures:
            features.append([
                s.mass,
                s.radius,
                s.coherence,
                s.persistence,
                s.entropy,
                s.frequency,
                len(s.neighbors),
                s.binding_energy,
                s.lifetime
            ])
        
        features = np.array(features)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Use DBSCAN for natural clustering
        from sklearn.cluster import DBSCAN
        
        # Try to find optimal epsilon
        best_clustering = None
        best_n_clusters = 0
        
        for eps in [0.3, 0.5, 0.7, 1.0]:
            clustering = DBSCAN(eps=eps, min_samples=max(3, min_samples//5)).fit(features_scaled)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            
            if 2 <= n_clusters <= 15 and n_clusters > best_n_clusters:
                best_clustering = clustering
                best_n_clusters = n_clusters
        
        if best_clustering is None:
            return {'n_classes': 0, 'discovered_types': {}}
        
        # Generate class descriptions
        discovered_types = {}
        labels = best_clustering.labels_
        
        for class_id in range(best_n_clusters):
            class_mask = labels == class_id
            class_structures = [s for i, s in enumerate(all_structures) if class_mask[i]]
            
            if not class_structures:
                continue
            
            # Compute mean properties
            mean_mass = np.mean([s.mass for s in class_structures])
            mean_radius = np.mean([s.radius for s in class_structures])
            mean_coherence = np.mean([s.coherence for s in class_structures])
            mean_lifetime = np.mean([s.lifetime for s in class_structures])
            mean_neighbors = np.mean([len(s.neighbors) for s in class_structures])
            
            # Generate descriptive name
            mass_desc = 'heavy' if mean_mass > 5 else 'light' if mean_mass < 1 else 'medium'
            stability_desc = 'stable' if mean_lifetime > 50 else 'transient' if mean_lifetime < 10 else 'metastable'
            coherence_desc = 'coherent' if mean_coherence > 0.9 else 'diffuse' if mean_coherence < 0.5 else 'mixed'
            interaction_desc = f"{int(mean_neighbors)}-bonded" if mean_neighbors > 0 else 'isolated'
            
            discovered_types[class_id] = {
                'name': f"{mass_desc}_{stability_desc}_{coherence_desc}",
                'interaction': interaction_desc,
                'mean_mass': float(mean_mass),
                'mean_radius': float(mean_radius),
                'mean_coherence': float(mean_coherence),
                'mean_lifetime': float(mean_lifetime),
                'mean_neighbors': float(mean_neighbors),
                'count': len(class_structures),
                'examples': [s.id for s in class_structures[:3]]  # First 3 examples
            }
        
        return {
            'n_classes': best_n_clusters,
            'discovered_types': discovered_types,
            'method': 'DBSCAN_consensus'
        }
    
    def _track_structures(self, current_structures: List[EmergentStructure]) -> List[EmergentStructure]:
        """
        Track structures across timesteps to assign persistent IDs and compute velocities.
        
        Uses nearest-neighbor matching with mass similarity constraint.
        """
        if not self.previous_structures:
            # First timestep - assign new IDs to all
            for s in current_structures:
                s.persistent_id = self.next_persistent_id
                self.persistent_structures[self.next_persistent_id] = [s]
                self.next_persistent_id += 1
            return current_structures
        
        # Build cost matrix for matching (distance + mass difference)
        n_prev = len(self.previous_structures)
        n_curr = len(current_structures)
        
        if n_prev == 0 or n_curr == 0:
            # Handle edge case - assign all as new
            for s in current_structures:
                s.persistent_id = self.next_persistent_id
                self.persistent_structures[self.next_persistent_id] = [s]
                self.next_persistent_id += 1
            return current_structures
        
        cost_matrix = np.zeros((n_prev, n_curr))
        
        for i, prev_s in enumerate(self.previous_structures):
            for j, curr_s in enumerate(current_structures):
                # Distance cost
                distance = np.sqrt(
                    (prev_s.center[0] - curr_s.center[0])**2 + 
                    (prev_s.center[1] - curr_s.center[1])**2
                )
                
                # Mass similarity cost (relative difference)
                mass_diff = abs(prev_s.mass - curr_s.mass) / (max(prev_s.mass, curr_s.mass) + 1e-10)
                
                # Combined cost (weighted)
                cost_matrix[i, j] = distance + 5.0 * mass_diff
        
        # Greedy matching (simple but effective)
        matched_prev = set()
        matched_curr = set()
        matches = []
        
        # Sort by cost and greedily assign
        costs_flat = [(cost_matrix[i, j], i, j) for i in range(n_prev) for j in range(n_curr)]
        costs_flat.sort()
        
        for cost, i, j in costs_flat:
            if i not in matched_prev and j not in matched_curr:
                # Only match if cost is reasonable (< 10 radius units)
                if cost < 10.0:  # Max matching distance
                    matches.append((i, j))
                    matched_prev.add(i)
                    matched_curr.add(j)
        
        # Assign persistent IDs based on matches
        for i, j in matches:
            prev_s = self.previous_structures[i]
            curr_s = current_structures[j]
            
            # Inherit persistent ID
            curr_s.persistent_id = prev_s.persistent_id
            
            # Compute velocity (pixels per timestep)
            curr_s.velocity = (
                curr_s.center[0] - prev_s.center[0],
                curr_s.center[1] - prev_s.center[1]
            )
            
            # Compute acceleration if we have velocity history
            if hasattr(prev_s, 'velocity') and prev_s.velocity != (0.0, 0.0):
                curr_s.acceleration = (
                    curr_s.velocity[0] - prev_s.velocity[0],
                    curr_s.velocity[1] - prev_s.velocity[1]
                )
            
            # Update history
            if curr_s.persistent_id in self.persistent_structures:
                self.persistent_structures[curr_s.persistent_id].append(curr_s)
            else:
                self.persistent_structures[curr_s.persistent_id] = [curr_s]
        
        # Unmatched current structures are new - assign new IDs
        for j in range(n_curr):
            if j not in matched_curr:
                curr_s = current_structures[j]
                curr_s.persistent_id = self.next_persistent_id
                self.persistent_structures[self.next_persistent_id] = [curr_s]
                self.next_persistent_id += 1
        
        return current_structures
