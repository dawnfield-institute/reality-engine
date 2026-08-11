"""
Galaxy Analyzer - Detects large-scale structure formation.

Looks for:
- Galactic rotation curves
- Dark matter signatures (missing mass)
- Large-scale clustering
- Cosmic web structures
- Hubble-like expansion
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from ..base_analyzer import BaseAnalyzer, Detection
from scipy.spatial import distance_matrix
from collections import defaultdict


class GalaxyAnalyzer(BaseAnalyzer):
    """
    Detects galaxy-like structures and large-scale cosmic phenomena.
    
    Galactic signatures:
    - Rotating systems with flat rotation curves
    - Hierarchical clustering (groups, clusters, superclusters)
    - Filamentary structure (cosmic web)
    - Dark matter halos (mass discrepancy)
    """
    
    def __init__(self, min_confidence: float = 0.65):
        super().__init__("galaxy_analyzer", min_confidence)
        self.galaxy_candidates = {}  # id -> properties
        self.cluster_history = []
        self.expansion_data = []
        
    def analyze(self, state) -> List[Detection]:
        """Detect galaxy-scale structures and cosmic phenomena."""
        detections = []
        current_step = state['step']
        structures = state.get('structures', [])
        
        if len(structures) < 10:  # Need enough structures
            return detections
        
        # Detect galaxies (rotating systems)
        detections.extend(self._detect_galaxies(structures, current_step))
        
        # Detect rotation curves
        detections.extend(self._detect_rotation_curves(structures, current_step))
        
        # Detect dark matter
        detections.extend(self._detect_dark_matter(structures, current_step))
        
        # Detect clustering
        detections.extend(self._detect_clustering(structures, current_step))
        
        # Detect cosmic web
        detection = self._detect_cosmic_web(structures, current_step)
        if detection:
            detections.append(detection)
        
        # Detect expansion
        if len(self.expansion_data) > 30:
            detection = self._detect_expansion(current_step)
            if detection:
                detections.append(detection)
        
        return detections
    
    def _detect_galaxies(self, structures, step: int) -> List[Detection]:
        """Identify galaxy-like rotating systems."""
        detections = []
        
        # Group nearby structures (potential galaxy members)
        positions = np.array([s.center for s in structures])
        masses = np.array([s.mass for s in structures])
        
        if len(positions) < 3:
            return detections
        
        # Find dense clusters
        clusters = self._find_clusters(positions, masses, radius=10)
        
        for cluster_idx, members in clusters.items():
            if len(members) < 5:  # Need multiple members
                continue
            
            member_structures = [structures[i] for i in members]
            cluster_positions = positions[members]
            cluster_masses = masses[members]
            
            # Compute cluster center of mass
            total_mass = np.sum(cluster_masses)
            if total_mass < 100:  # Too small for galaxy
                continue
            
            center_of_mass = np.average(cluster_positions, weights=cluster_masses, axis=0)
            
            # Check for rotation
            velocities = np.array([s.velocity for s in member_structures])
            rotation_score = self._compute_rotation_score(
                cluster_positions, velocities, center_of_mass
            )
            
            # Galaxy criteria: massive, rotating, multi-member
            if rotation_score > 0.5 and total_mass > 100:
                confidence = min(0.9, 0.5 + 0.2 * rotation_score + 0.1 * len(members) / 10)
                
                # Estimate galaxy radius
                distances = np.linalg.norm(cluster_positions - center_of_mass, axis=1)
                radius = np.percentile(distances, 90)
                
                detections.append(Detection(
                    type="galaxy",
                    confidence=confidence,
                    location=tuple(center_of_mass),
                    time=step,
                    properties={
                        'n_members': len(members),
                        'total_mass': total_mass,
                        'radius': radius,
                        'rotation_score': rotation_score,
                        'member_ids': [structures[i].id for i in members]
                    }
                ))
                
                # Track for rotation curve analysis
                self.galaxy_candidates[step] = {
                    'center': center_of_mass,
                    'members': members,
                    'positions': cluster_positions,
                    'velocities': velocities,
                    'masses': cluster_masses
                }
        
        return detections
    
    def _find_clusters(self, positions: np.ndarray, masses: np.ndarray, 
                      radius: float) -> Dict[int, List[int]]:
        """Find clusters of structures using density-based clustering."""
        n = len(positions)
        if n == 0:
            return {}
        
        # Compute distance matrix
        dist_mat = distance_matrix(positions, positions)
        
        # Simple density-based clustering
        clusters = {}
        assigned = set()
        cluster_id = 0
        
        for i in range(n):
            if i in assigned or masses[i] < 10:  # Skip assigned or too light
                continue
            
            # Find neighbors within radius
            neighbors = np.where(dist_mat[i] < radius)[0]
            
            if len(neighbors) >= 3:  # Minimum cluster size
                clusters[cluster_id] = list(neighbors)
                assigned.update(neighbors)
                cluster_id += 1
        
        return clusters
    
    def _compute_rotation_score(self, positions: np.ndarray, velocities: np.ndarray,
                               center: np.ndarray) -> float:
        """
        Compute how much a system is rotating vs random motion.
        
        Returns score 0-1, where 1 = perfect rotation.
        """
        # Vectors from center to each structure
        r_vectors = positions - center
        
        # Tangential direction (perpendicular to radial)
        tangential = np.column_stack([-r_vectors[:, 1], r_vectors[:, 0]])
        tangential_norm = np.linalg.norm(tangential, axis=1, keepdims=True)
        tangential_norm[tangential_norm < 1e-6] = 1  # Avoid division by zero
        tangential_unit = tangential / tangential_norm
        
        # Project velocities onto tangential direction
        v_tangential = np.sum(velocities * tangential_unit, axis=1)
        v_magnitude = np.linalg.norm(velocities, axis=1)
        
        # Rotation score: fraction of velocity in tangential direction
        tangential_fraction = np.abs(v_tangential) / (v_magnitude + 1e-6)
        
        # Also check if all rotating same direction
        rotation_direction = np.sign(v_tangential)
        direction_coherence = np.abs(np.mean(rotation_direction))
        
        return float(np.mean(tangential_fraction) * direction_coherence)
    
    def _detect_rotation_curves(self, structures, step: int) -> List[Detection]:
        """
        Detect flat rotation curves (signature of dark matter).
        
        Real galaxies: v(r) ≈ constant at large r (not Keplerian v ∝ 1/√r)
        """
        detections = []
        
        # Need galaxy candidates
        if step not in self.galaxy_candidates:
            return detections
        
        galaxy = self.galaxy_candidates[step]
        positions = galaxy['positions']
        velocities = galaxy['velocities']
        masses = galaxy['masses']
        center = galaxy['center']
        
        # Compute radial distances
        r_vectors = positions - center
        radii = np.linalg.norm(r_vectors, axis=1)
        
        # Compute tangential velocities
        tangential = np.column_stack([-r_vectors[:, 1], r_vectors[:, 0]])
        tangential_unit = tangential / (np.linalg.norm(tangential, axis=1, keepdims=True) + 1e-6)
        v_tangential = np.abs(np.sum(velocities * tangential_unit, axis=1))
        
        # Sort by radius
        sorted_indices = np.argsort(radii)
        radii_sorted = radii[sorted_indices]
        v_sorted = v_tangential[sorted_indices]
        
        if len(radii_sorted) < 5:
            return detections
        
        # Divide into inner and outer regions
        median_r = np.median(radii_sorted)
        inner = radii_sorted < median_r
        outer = radii_sorted >= median_r
        
        if np.sum(inner) < 2 or np.sum(outer) < 2:
            return detections
        
        v_inner = np.mean(v_sorted[inner])
        v_outer = np.mean(v_sorted[outer])
        
        # Flat curve: outer velocity ≈ inner velocity
        # Keplerian: outer < inner
        flatness = v_outer / (v_inner + 1e-6)
        
        if flatness > 0.7:  # Reasonably flat
            confidence = min(0.9, 0.5 + 0.3 * flatness)
            
            # Fit power law: v(r) = A * r^β (β=0 is flat, β=-0.5 is Keplerian)
            log_r = np.log(radii_sorted + 1)
            log_v = np.log(v_sorted + 0.01)
            
            A = np.vstack([log_r, np.ones(len(log_r))]).T
            beta, log_A = np.linalg.lstsq(A, log_v, rcond=None)[0]
            
            detections.append(Detection(
                type="flat_rotation_curve",
                confidence=confidence,
                location=tuple(center),
                time=step,
                equation=f"v(r) ~ r^{beta:.2f}",
                parameters={'beta': beta, 'flatness': flatness},
                properties={
                    'v_inner': v_inner,
                    'v_outer': v_outer,
                    'n_points': len(radii_sorted),
                    'interpretation': 'Flat curve suggests dark matter' if flatness > 0.8 else 'Mildly flat'
                }
            ))
        
        return detections
    
    def _detect_dark_matter(self, structures, step: int) -> List[Detection]:
        """
        Detect dark matter signatures: gravitational effects without visible mass.
        
        Look for: rotation curves inconsistent with visible mass.
        """
        detections = []
        
        if step not in self.galaxy_candidates:
            return detections
        
        galaxy = self.galaxy_candidates[step]
        positions = galaxy['positions']
        velocities = galaxy['velocities']
        masses = galaxy['masses']
        center = galaxy['center']
        
        # Compute total visible mass
        visible_mass = np.sum(masses)
        
        # Compute expected velocity from visible mass (Keplerian)
        # v² = GM/r
        r_vectors = positions - center
        radii = np.linalg.norm(r_vectors, axis=1)
        
        # Tangential velocities
        tangential = np.column_stack([-r_vectors[:, 1], r_vectors[:, 0]])
        tangential_unit = tangential / (np.linalg.norm(tangential, axis=1, keepdims=True) + 1e-6)
        v_tangential = np.abs(np.sum(velocities * tangential_unit, axis=1))
        
        # Expected Keplerian velocity at each radius
        G_sim = 1.0  # Simulation gravitational constant
        v_expected = np.sqrt(G_sim * visible_mass / (radii + 1))
        
        # Observed velocity
        v_observed = v_tangential
        
        # Mass discrepancy: v_observed² / v_expected² ≈ M_total / M_visible
        mass_ratio = np.mean((v_observed / (v_expected + 0.1)) ** 2)
        
        # Dark matter detected if observed velocity >> expected
        if mass_ratio > 2:
            dark_mass = visible_mass * (mass_ratio - 1)
            confidence = min(0.9, 0.5 + 0.2 * min(mass_ratio / 5, 1))
            
            detections.append(Detection(
                type="dark_matter_signature",
                confidence=confidence,
                location=tuple(center),
                time=step,
                properties={
                    'visible_mass': visible_mass,
                    'inferred_total_mass': visible_mass * mass_ratio,
                    'dark_mass': dark_mass,
                    'dark_fraction': (mass_ratio - 1) / mass_ratio,
                    'mass_ratio': mass_ratio
                }
            ))
        
        return detections
    
    def _detect_clustering(self, structures, step: int) -> List[Detection]:
        """Detect hierarchical clustering of structures."""
        detections = []
        
        positions = np.array([s.center for s in structures])
        masses = np.array([s.mass for s in structures])
        
        # Multi-scale clustering
        for scale in [5, 10, 20]:
            clusters = self._find_clusters(positions, masses, radius=scale)
            
            if len(clusters) >= 2:
                # Compute cluster statistics
                cluster_masses = []
                cluster_sizes = []
                
                for members in clusters.values():
                    cluster_mass = np.sum(masses[members])
                    cluster_masses.append(cluster_mass)
                    cluster_sizes.append(len(members))
                
                if len(cluster_masses) >= 2:
                    mean_mass = np.mean(cluster_masses)
                    std_mass = np.std(cluster_masses)
                    
                    # Significant clustering
                    if mean_mass > 50:
                        confidence = min(0.85, 0.5 + 0.1 * len(clusters))
                        
                        detections.append(Detection(
                            type="structure_clustering",
                            confidence=confidence,
                            location=None,
                            time=step,
                            properties={
                                'scale': scale,
                                'n_clusters': len(clusters),
                                'mean_cluster_mass': mean_mass,
                                'mean_cluster_size': np.mean(cluster_sizes),
                                'mass_variation': std_mass / mean_mass if mean_mass > 0 else 0
                            }
                        ))
        
        return detections
    
    def _detect_cosmic_web(self, structures, step: int) -> Optional[Detection]:
        """Detect cosmic web: filamentary large-scale structure."""
        positions = np.array([s.center for s in structures])
        
        if len(positions) < 20:
            return None
        
        # Compute density field
        # Use 2D histogram to find filaments
        H, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1], bins=20
        )
        
        # Filaments = elongated high-density regions
        # Compute principal axes of density
        y_coords, x_coords = np.mgrid[0:H.shape[0], 0:H.shape[1]]
        
        # Weighted by density
        total_density = np.sum(H)
        if total_density < 1:
            return None
        
        x_mean = np.sum(x_coords * H) / total_density
        y_mean = np.sum(y_coords * H) / total_density
        
        # Covariance matrix
        x_var = np.sum((x_coords - x_mean)**2 * H) / total_density
        y_var = np.sum((y_coords - y_mean)**2 * H) / total_density
        xy_cov = np.sum((x_coords - x_mean) * (y_coords - y_mean) * H) / total_density
        
        cov_matrix = np.array([[x_var, xy_cov], [xy_cov, y_var]])
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # Filamentary if one axis >> other axis
        anisotropy = eigenvalues[1] / (eigenvalues[0] + 1e-6)
        
        if anisotropy > 3:  # Elongated structure
            confidence = min(0.85, 0.5 + 0.1 * min(anisotropy / 5, 1))
            
            return Detection(
                type="cosmic_web_filament",
                confidence=confidence,
                location=None,
                time=step,
                properties={
                    'anisotropy': anisotropy,
                    'n_structures': len(positions),
                    'density_std': float(np.std(H)),
                    'interpretation': 'Large-scale filamentary structure detected'
                }
            )
        
        return None
    
    def _detect_expansion(self, step: int) -> Optional[Detection]:
        """Detect Hubble-like expansion: v ∝ d (recession velocity ∝ distance)."""
        if len(self.expansion_data) < 30:
            return None
        
        # Analyze recent data
        recent = self.expansion_data[-50:]
        
        distances = np.array([d['distance'] for d in recent])
        velocities = np.array([d['velocity'] for d in recent])
        
        # Fit linear relationship: v = H₀ * d
        A = np.vstack([distances, np.ones(len(distances))]).T
        H0, intercept = np.linalg.lstsq(A, velocities, rcond=None)[0]
        
        # Compute R²
        predicted = H0 * distances + intercept
        ss_res = np.sum((velocities - predicted) ** 2)
        ss_tot = np.sum((velocities - np.mean(velocities)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        if r_squared > 0.5 and H0 > 0:
            confidence = min(0.9, r_squared)
            
            return Detection(
                type="hubble_expansion",
                confidence=confidence,
                location=None,
                time=step,
                equation=f"v = H0 * d",
                parameters={'H0': H0, 'r_squared': r_squared},
                properties={
                    'hubble_constant': H0,
                    'n_measurements': len(recent),
                    'interpretation': 'Universe expanding (Hubble law detected)'
                }
            )
        
        return None
    
    def record_expansion_measurement(self, distance: float, recession_velocity: float):
        """Record distance-velocity pair for expansion analysis."""
        self.expansion_data.append({
            'distance': distance,
            'velocity': recession_velocity
        })
        
        # Keep recent only
        if len(self.expansion_data) > 100:
            self.expansion_data = self.expansion_data[-100:]
