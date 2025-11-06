"""
Structure Analyzer - Multi-scale emergent structure detection.

Detects structures at various scales:
- Atomic structures (stable oscillating patterns)
- Molecular bonds (coupled atoms)
- Gravity wells (density concentrations)
- Dark matter regions (high M, low A)
- Stellar regions (hot + dense)

This is production-quality analysis code extracted from successful experiments.
Used for Phase 2+ structure stabilization and discovery.
"""
import torch
import numpy as np
from scipy import ndimage
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from tools.atomic_analyzer import AtomicAnalyzer, Atom


class StructureAnalyzer:
    """
    Analyze emergent structures in the evolving universe.
    
    Multi-scale structure detection from atoms to stellar regions.
    Integrates with AtomicAnalyzer for molecular chemistry.
    
    Usage:
        analyzer = StructureAnalyzer(engine)
        structures = analyzer.analyze_step(step=100)
        
        print(f"Atoms: {len(structures['atoms'])}")
        print(f"Molecules: {len(structures['molecules'])}")
        print(f"Gravity wells: {len(structures['gravity_wells'])}")
    """
    
    def __init__(self, engine, min_atom_stability: float = 0.65):
        """
        Initialize structure analyzer.
        
        Args:
            engine: RealityEngine instance to analyze
            min_atom_stability: Minimum stability threshold for atomic detection (0.65 = 65% equilibrium)
        """
        self.engine = engine
        self.atomic_analyzer = AtomicAnalyzer(min_stability=min_atom_stability)
        self.history = {
            'gravity_wells': [],
            'dark_matter_regions': [],
            'stellar_regions': [],
            'atoms': [],
            'molecules': [],
            'time': []
        }
        
    def detect_gravity_wells(self, M: torch.Tensor, threshold: float = 1.5) -> List[Dict]:
        """
        Detect gravity wells as high-density regions in memory field.
        
        Gravity wells are concentrations of mass (M field) that form
        naturally from self-attraction dynamics. These represent the
        emergent gravitational potential without hardcoding 1/r².
        
        Args:
            M: Memory field (mass/information accumulation)
            threshold: Standard deviations above mean for detection (default 1.5σ)
            
        Returns:
            List of dicts with well properties:
            - center: (y, x) position
            - mass: Total accumulated mass
            - radius: Effective radius
            - field_strength: Maximum gradient magnitude
            - density: Average density in well
        """
        M_np = M.cpu().numpy()
        mean_density = M_np.mean()
        std_density = M_np.std()
        
        if std_density < 1e-6:
            return []
        
        # Find regions > threshold standard deviations above mean
        high_density = M_np > (mean_density + threshold * std_density)
        
        # Find connected components (gravity wells)
        labeled, num_wells = ndimage.label(high_density)
        
        wells = []
        for i in range(1, num_wells + 1):
            mask = labeled == i
            if mask.sum() > 4:  # Minimum size
                center = ndimage.center_of_mass(M_np * mask)
                mass = M_np[mask].sum()
                radius = np.sqrt(mask.sum() / np.pi)
                
                # Calculate "gravitational" field strength (density gradient)
                grad_y, grad_x = np.gradient(M_np * mask)
                field_strength = np.sqrt(grad_y**2 + grad_x**2).max()
                
                wells.append({
                    'center': center,
                    'mass': float(mass),
                    'radius': float(radius),
                    'field_strength': float(field_strength),
                    'density': float(M_np[mask].mean())
                })
        
        return wells
    
    def detect_dark_matter(self, M: torch.Tensor, A: torch.Tensor) -> List[Dict]:
        """
        Detect dark matter as regions with high memory but low actualization.
        
        Dark matter emerges as "invisible mass" - information that gravitates
        (affects M field) but doesn't interact strongly (low A field).
        This is NOT programmed - it emerges from the M/A dynamics!
        
        Args:
            M: Memory field (accumulated mass/information)
            A: Activity field (actualization/interaction)
            
        Returns:
            List of dicts with dark matter halo properties:
            - center: (y, x) position
            - dark_mass: Total M in region
            - visible_mass: Total |A| in region
            - dark_ratio: M/A ratio (how "dark" it is)
            - size: Number of cells in halo
        """
        M_np = M.cpu().numpy()
        A_np = A.cpu().numpy()
        
        # Dark matter: high M, low A (gravitates but doesn't interact much)
        dark_ratio = M_np / (np.abs(A_np) + 0.01)
        
        # Find regions with high dark ratio
        threshold = np.percentile(dark_ratio, 95)
        dark_regions = dark_ratio > threshold
        
        # Analyze dark matter halos
        labeled, num_regions = ndimage.label(dark_regions)
        
        dark_matter = []
        for i in range(1, num_regions + 1):
            mask = labeled == i
            if mask.sum() > 8:  # Minimum halo size
                center = ndimage.center_of_mass(M_np * mask)
                mass = M_np[mask].sum()
                
                dark_matter.append({
                    'center': center,
                    'dark_mass': float(mass),
                    'visible_mass': float(np.abs(A_np[mask]).sum()),
                    'dark_ratio': float(dark_ratio[mask].mean()),
                    'size': int(mask.sum())
                })
        
        return dark_matter
    
    def detect_stellar_regions(self, T: torch.Tensor, M: torch.Tensor, 
                              t_percentile: float = 85, m_percentile: float = 85) -> List[Dict]:
        """
        Detect proto-stellar regions (hot, dense areas).
        
        Stars form where heat and mass concentrate together.
        These are regions of active information collapse generating heat.
        Luminosity scales as T⁴ (Stefan-Boltzmann-like emergence).
        
        Args:
            T: Temperature field
            M: Memory/mass field
            t_percentile: Temperature threshold percentile (default 85th)
            m_percentile: Mass threshold percentile (default 85th)
            
        Returns:
            List of dicts with stellar properties:
            - center: (y, x) position
            - temperature: Average T
            - mass: Total accumulated mass
            - luminosity: Integrated T⁴ (Stefan-Boltzmann analog)
            - radius: Effective radius
        """
        T_np = T.cpu().numpy()
        M_np = M.cpu().numpy()
        
        # Stars form in hot, dense regions
        T_threshold = np.percentile(T_np, t_percentile)
        M_threshold = np.percentile(M_np, m_percentile)
        
        stellar_candidates = (T_np > T_threshold) & (M_np > M_threshold)
        
        labeled, num_stars = ndimage.label(stellar_candidates)
        
        stars = []
        for i in range(1, num_stars + 1):
            mask = labeled == i
            if mask.sum() > 2:  # Minimum stellar core size
                center = ndimage.center_of_mass(T_np * mask)
                temperature = T_np[mask].mean()
                mass = M_np[mask].sum()
                luminosity = (T_np[mask] ** 4).sum()  # Stefan-Boltzmann-like
                
                stars.append({
                    'center': center,
                    'temperature': float(temperature),
                    'mass': float(mass),
                    'luminosity': float(luminosity),
                    'radius': float(np.sqrt(mask.sum() / np.pi))
                })
        
        return stars
    
    def detect_molecules(self, atoms: List[Atom], bond_distance: float = 3.0) -> List[Dict]:
        """
        Detect molecular bonds between atoms.
        
        Molecules form when atoms are close enough that their oscillation
        patterns couple. This is proximity-based bonding - no chemistry
        rules programmed! H₂, H₂O, etc. form naturally.
        
        Args:
            atoms: List of detected atoms from AtomicAnalyzer
            bond_distance: Maximum distance for bond formation (default 3.0 grid units)
            
        Returns:
            List of dicts with molecular properties:
            - formula: Molecular formula (e.g., "HH" for H₂)
            - atoms: List of element symbols
            - bond_length: Distance between atoms
            - total_mass: Sum of atomic masses
            - binding_energy: 1/distance approximation
            - center: Center of mass position
        """
        if len(atoms) < 2:
            return []
        
        molecules = []
        used_atoms = set()
        
        for i, atom1 in enumerate(atoms):
            if i in used_atoms:
                continue
                
            for j, atom2 in enumerate(atoms[i+1:], i+1):
                if j in used_atoms:
                    continue
                    
                # Calculate distance
                pos1 = np.array(atom1.position)
                pos2 = np.array(atom2.position)
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < bond_distance:
                    # Found a bond!
                    used_atoms.add(i)
                    used_atoms.add(j)
                    
                    # Classify molecule
                    formula = ''.join(sorted([atom1.element, atom2.element]))
                    
                    molecules.append({
                        'formula': formula,
                        'atoms': [atom1.element, atom2.element],
                        'bond_length': float(distance),
                        'total_mass': atom1.mass + atom2.mass,
                        'binding_energy': float(1.0 / distance),
                        'center': ((pos1 + pos2) / 2).tolist()
                    })
        
        return molecules
    
    def analyze_step(self, step: int) -> Dict:
        """
        Analyze current engine state for all emergent structures.
        
        Runs all detection algorithms and stores results in history.
        This is the main analysis entry point.
        
        Args:
            step: Current simulation step number
            
        Returns:
            Dict containing all detected structures:
            - gravity_wells: List of gravitational concentrations
            - dark_matter: List of dark matter halos
            - stellar_regions: List of proto-stars
            - atoms: List of Atom objects
            - molecules: List of molecular bonds
        """
        state = self.engine.current_state
        M = state.M
        A = state.A
        T = state.T
        
        # Detect various structures
        wells = self.detect_gravity_wells(M)
        dark = self.detect_dark_matter(M, A)
        stars = self.detect_stellar_regions(T, M)
        atoms = self.atomic_analyzer.detect_atoms(state)
        molecules = self.detect_molecules(atoms)
        
        # Store in history
        self.history['gravity_wells'].append(len(wells))
        self.history['dark_matter_regions'].append(len(dark))
        self.history['stellar_regions'].append(len(stars))
        self.history['atoms'].append(len(atoms))
        self.history['molecules'].append(len(molecules))
        self.history['time'].append(step)
        
        return {
            'gravity_wells': wells,
            'dark_matter': dark,
            'stellar_regions': stars,
            'atoms': atoms,
            'molecules': molecules
        }
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics of detected structures over time.
        
        Returns:
            Dict with max/mean/current counts for each structure type.
        """
        return {
            'gravity_wells': {
                'max': max(self.history['gravity_wells']) if self.history['gravity_wells'] else 0,
                'mean': np.mean(self.history['gravity_wells']) if self.history['gravity_wells'] else 0,
                'current': self.history['gravity_wells'][-1] if self.history['gravity_wells'] else 0
            },
            'dark_matter_regions': {
                'max': max(self.history['dark_matter_regions']) if self.history['dark_matter_regions'] else 0,
                'mean': np.mean(self.history['dark_matter_regions']) if self.history['dark_matter_regions'] else 0,
                'current': self.history['dark_matter_regions'][-1] if self.history['dark_matter_regions'] else 0
            },
            'stellar_regions': {
                'max': max(self.history['stellar_regions']) if self.history['stellar_regions'] else 0,
                'mean': np.mean(self.history['stellar_regions']) if self.history['stellar_regions'] else 0,
                'current': self.history['stellar_regions'][-1] if self.history['stellar_regions'] else 0
            },
            'atoms': {
                'max': max(self.history['atoms']) if self.history['atoms'] else 0,
                'mean': np.mean(self.history['atoms']) if self.history['atoms'] else 0,
                'current': self.history['atoms'][-1] if self.history['atoms'] else 0
            },
            'molecules': {
                'max': max(self.history['molecules']) if self.history['molecules'] else 0,
                'mean': np.mean(self.history['molecules']) if self.history['molecules'] else 0,
                'current': self.history['molecules'][-1] if self.history['molecules'] else 0
            }
        }
