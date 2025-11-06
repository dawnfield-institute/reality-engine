"""
Stellar Structure Analyzer - Detect Emergent Gravitational Systems

Identifies large-scale structures that emerge from field dynamics:
- Mass concentrations (proto-stars, stars)
- Gravitational wells (potential minima)
- Fusion regions (high energy + mass density)
- Black holes (extreme mass-energy concentration)

These should emerge naturally and drive stellar nucleosynthesis.
Pure PyTorch implementation for GPU acceleration.
"""

import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StellarStructure:
    """Represents an emergent gravitational structure"""
    position: Tuple[int, int, int]  # Center of mass
    total_mass: float  # Total memory field
    radius: float  # Effective radius
    core_density: float  # Central mass density
    potential_depth: float  # Gravitational well depth
    energy_density: float  # Core energy concentration
    temperature: float  # Effective temperature (from energy)
    particle_count: int  # Number of constituent particles
    structure_type: str  # star, proto-star, black_hole, etc.
    
    def is_fusion_active(self) -> bool:
        """Check if conditions favor fusion (high density + energy)"""
        # Fusion requires high mass density AND high energy
        return (self.core_density > 100.0 and 
                self.energy_density > 50.0 and
                self.temperature > 10.0)
    
    def schwarzschild_radius(self) -> float:
        """Estimate Schwarzschild radius (in grid units)"""
        # In natural units: r_s ~ 2M
        return 2.0 * self.total_mass


@dataclass
class FusionEvent:
    """Represents a fusion event creating new particles"""
    location: Tuple[int, int, int]
    parent_masses: List[float]
    product_mass: float
    energy_released: float
    timestamp: int  # Simulation step


class StellarAnalyzer:
    """Detect and analyze emergent stellar structures"""
    
    def __init__(self, mass_threshold: float = 500.0):
        """
        Args:
            mass_threshold: Minimum mass for stellar structure
        """
        self.mass_threshold = mass_threshold
        self.structures: List[StellarStructure] = []
        self.fusion_events: List[FusionEvent] = []
    
    def detect_structures(self, E: torch.Tensor, I: torch.Tensor, M: torch.Tensor, 
                         particles: List = None) -> List[StellarStructure]:
        """
        Detect large-scale gravitational structures using pure PyTorch.
        
        Args:
            E, I, M: Field tensors (energy, information, memory) - can be on GPU
            particles: Optional list of detected particles
        
        Returns:
            List of detected stellar structures
        """
        self.structures = []
        
        # Ensure tensors are on same device
        device = M.device
        
        # 1. Smooth memory field using 3D convolution (torch equivalent of gaussian_filter)
        M_smooth = self._gaussian_smooth_3d(M, sigma=3.0)
        
        # 2. Find high-mass regions
        mass_mean = M_smooth.mean()
        mass_std = M_smooth.std()
        threshold = mass_mean + 2.0 * mass_std
        
        high_mass = M_smooth > threshold
        
        if not high_mass.any():
            return self.structures
        
        # 3. Label connected regions (torch equivalent of scipy.ndimage.label)
        labeled, num_features = self._label_connected_components(high_mass)
        
        # 4. Analyze each region
        for region_id in range(1, num_features + 1):
            mask = (labeled == region_id)
            
            # Calculate total mass
            total_mass = M[mask].sum().item()
            
            # Skip small concentrations
            if total_mass < self.mass_threshold:
                continue
            
            # Find center of mass
            indices = torch.where(mask)
            if len(indices[0]) == 0:
                continue
                
            masses = M[mask]
            center = (
                int((indices[0].float() * masses).sum() / masses.sum()),
                int((indices[1].float() * masses).sum() / masses.sum()),
                int((indices[2].float() * masses).sum() / masses.sum())
            )
            
            # Calculate properties
            radius = self._calculate_radius(mask, center)
            core_density = self._calculate_core_density(M, center, radius)
            potential_depth = self._calculate_potential(M, center, radius)
            energy_density = self._calculate_energy_density(E, I, center, radius)
            temperature = self._estimate_temperature(E, I, center, radius)
            
            # Count particles in region (if provided)
            particle_count = 0
            if particles:
                particle_count = sum(1 for p in particles 
                                   if self._is_in_region(p.position, mask))
            
            # Classify structure
            structure_type = self._classify_structure(
                total_mass, core_density, energy_density, 
                temperature, potential_depth, radius
            )
            
            structure = StellarStructure(
                position=center,
                total_mass=total_mass,
                radius=radius,
                core_density=core_density,
                potential_depth=potential_depth,
                energy_density=energy_density,
                temperature=temperature,
                particle_count=particle_count,
                structure_type=structure_type
            )
            
            self.structures.append(structure)
        
        return self.structures
    
    def _gaussian_smooth_3d(self, tensor: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
        """
        Apply 3D Gaussian smoothing using separable convolutions.
        Pure PyTorch implementation.
        """
        # Create 1D Gaussian kernel
        kernel_size = int(2 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Generate 1D Gaussian
        x = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device) - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # Add batch and channel dimensions for conv3d
        # Input: [D, H, W] -> [1, 1, D, H, W]
        smoothed = tensor.unsqueeze(0).unsqueeze(0)
        
        # Apply separable 3D convolution (much faster than full 3D kernel)
        # Smooth along each axis
        kernel_3d = gauss_1d.view(1, 1, -1, 1, 1)
        smoothed = torch.nn.functional.conv3d(smoothed, kernel_3d, padding=(kernel_size//2, 0, 0))
        
        kernel_3d = gauss_1d.view(1, 1, 1, -1, 1)
        smoothed = torch.nn.functional.conv3d(smoothed, kernel_3d, padding=(0, kernel_size//2, 0))
        
        kernel_3d = gauss_1d.view(1, 1, 1, 1, -1)
        smoothed = torch.nn.functional.conv3d(smoothed, kernel_3d, padding=(0, 0, kernel_size//2))
        
        return smoothed.squeeze(0).squeeze(0)
    
    def _label_connected_components(self, mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Label connected components in 3D binary mask.
        Simple flood-fill algorithm using PyTorch.
        """
        device = mask.device
        labeled = torch.zeros_like(mask, dtype=torch.int32)
        current_label = 0
        
        # Convert to CPU for easier indexing (this part is tricky to vectorize)
        mask_cpu = mask.cpu()
        labeled_cpu = labeled.cpu()
        
        # Find all True positions
        positions = torch.where(mask_cpu)
        if len(positions[0]) == 0:
            return labeled, 0
        
        # Simple connected components using 6-connectivity
        visited = torch.zeros_like(mask_cpu, dtype=torch.bool)
        
        for i in range(len(positions[0])):
            if visited[positions[0][i], positions[1][i], positions[2][i]]:
                continue
            
            current_label += 1
            # Flood fill from this position
            stack = [(positions[0][i].item(), positions[1][i].item(), positions[2][i].item())]
            
            while stack:
                x, y, z = stack.pop()
                
                if x < 0 or x >= mask_cpu.shape[0] or \
                   y < 0 or y >= mask_cpu.shape[1] or \
                   z < 0 or z >= mask_cpu.shape[2]:
                    continue
                
                if visited[x, y, z] or not mask_cpu[x, y, z]:
                    continue
                
                visited[x, y, z] = True
                labeled_cpu[x, y, z] = current_label
                
                # Add 6-connected neighbors
                stack.extend([(x+1, y, z), (x-1, y, z),
                            (x, y+1, z), (x, y-1, z),
                            (x, y, z+1), (x, y, z-1)])
        
        return labeled_cpu.to(device), current_label
    
    def _calculate_radius(self, mask: torch.Tensor, center: Tuple[int, int, int]) -> float:
        """Calculate RMS radius of mass distribution"""
        indices = torch.where(mask)
        
        if len(indices[0]) == 0:
            return 1.0
        
        # Distance from center
        dx = indices[0].float() - center[0]
        dy = indices[1].float() - center[1]
        dz = indices[2].float() - center[2]
        r2 = dx**2 + dy**2 + dz**2
        
        return torch.sqrt(r2.mean()).item()
    
    def _calculate_core_density(self, M: torch.Tensor, center: Tuple[int, int, int], 
                                radius: float) -> float:
        """Calculate mass density at core"""
        # Extract small region around center
        x, y, z = center
        r = max(1, int(radius / 4))  # Core is inner 25%
        
        x_min, x_max = max(0, x-r), min(M.shape[0], x+r+1)
        y_min, y_max = max(0, y-r), min(M.shape[1], y+r+1)
        z_min, z_max = max(0, z-r), min(M.shape[2], z+r+1)
        
        core = M[x_min:x_max, y_min:y_max, z_min:z_max]
        volume = core.numel()
        
        return (core.sum() / volume).item() if volume > 0 else 0.0
    
    def _calculate_potential(self, M: torch.Tensor, center: Tuple[int, int, int],
                           radius: float) -> float:
        """Estimate gravitational potential depth"""
        # Potential ~ -M/r at surface
        # Deeper well = stronger gravity
        if radius < 1.0:
            radius = 1.0
        
        x, y, z = center
        r = int(radius)
        
        x_min, x_max = max(0, x-r), min(M.shape[0], x+r+1)
        y_min, y_max = max(0, y-r), min(M.shape[1], y+r+1)
        z_min, z_max = max(0, z-r), min(M.shape[2], z+r+1)
        
        region_mass = M[x_min:x_max, y_min:y_max, z_min:z_max].sum().item()
        
        return -region_mass / radius
    
    def _calculate_energy_density(self, E: torch.Tensor, I: torch.Tensor, 
                                 center: Tuple[int, int, int], radius: float) -> float:
        """Calculate total field energy density at core"""
        x, y, z = center
        r = max(1, int(radius / 4))
        
        x_min, x_max = max(0, x-r), min(E.shape[0], x+r+1)
        y_min, y_max = max(0, y-r), min(E.shape[1], y+r+1)
        z_min, z_max = max(0, z-r), min(E.shape[2], z+r+1)
        
        E_core = E[x_min:x_max, y_min:y_max, z_min:z_max]
        I_core = I[x_min:x_max, y_min:y_max, z_min:z_max]
        
        # Total energy: E + I fields
        volume = E_core.numel()
        return ((E_core.sum() + I_core.sum()) / volume).item() if volume > 0 else 0.0
    
    def _estimate_temperature(self, E: torch.Tensor, I: torch.Tensor,
                            center: Tuple[int, int, int], radius: float) -> float:
        """Estimate effective temperature from field fluctuations"""
        x, y, z = center
        r = max(1, int(radius / 4))
        
        x_min, x_max = max(0, x-r), min(E.shape[0], x+r+1)
        y_min, y_max = max(0, y-r), min(E.shape[1], y+r+1)
        z_min, z_max = max(0, z-r), min(E.shape[2], z+r+1)
        
        E_core = E[x_min:x_max, y_min:y_max, z_min:z_max]
        
        # Temperature ~ variance of energy field
        # High variance = high kinetic energy = high temp
        return E_core.std().item()
    
    def _classify_structure(self, mass: float, core_density: float, 
                          energy_density: float, temperature: float,
                          potential_depth: float, radius: float) -> str:
        """Classify the type of stellar structure"""
        
        # Black hole: extreme mass concentration, small radius
        schwarzschild = 2.0 * mass
        if radius < schwarzschild * 1.5 and mass > 2000.0:
            return "black_hole"
        
        # Star: high core density + high energy + fusion conditions
        if core_density > 100.0 and energy_density > 50.0 and temperature > 10.0:
            if mass > 5000.0:
                return "massive_star"
            elif mass > 2000.0:
                return "star"
            else:
                return "dwarf_star"
        
        # Proto-star: high mass, moderate density, not yet fusing
        if mass > 1000.0 and core_density > 50.0:
            return "proto_star"
        
        # Mass concentration: not yet stellar
        if mass > 500.0:
            return "mass_concentration"
        
        return "unknown"
    
    def _is_in_region(self, position: Tuple[int, int, int], mask: torch.Tensor) -> bool:
        """Check if position is within masked region"""
        x, y, z = position
        if (0 <= x < mask.shape[0] and 
            0 <= y < mask.shape[1] and 
            0 <= z < mask.shape[2]):
            return mask[x, y, z].item()
        return False
    
    def detect_fusion_events(self, E: torch.Tensor, M: torch.Tensor, 
                           step: int) -> List[FusionEvent]:
        """
        Detect potential fusion events in high-density regions
        
        Fusion signature:
        - High mass density (many particles close)
        - High energy (compression/heating)
        - Mass increase (particles combining)
        """
        events = []
        
        # Look for regions with extreme conditions
        for structure in self.structures:
            if structure.is_fusion_active():
                # This structure has fusion conditions
                # Check for mass increase (signature of particle creation)
                x, y, z = structure.position
                r = max(1, int(structure.radius / 4))
                
                x_min, x_max = max(0, x-r), min(M.shape[0], x+r+1)
                y_min, y_max = max(0, y-r), min(M.shape[1], y+r+1)
                z_min, z_max = max(0, z-r), min(M.shape[2], z+r+1)
                
                core_M = M[x_min:x_max, y_min:y_max, z_min:z_max]
                core_E = E[x_min:x_max, y_min:y_max, z_min:z_max]
                
                # Energy released in fusion (positive for exothermic)
                energy_released = np.sum(core_E) * 0.01  # Small fraction
                
                # Record potential fusion
                event = FusionEvent(
                    location=structure.position,
                    parent_masses=[structure.core_density],  # Simplified
                    product_mass=structure.total_mass,
                    energy_released=energy_released,
                    timestamp=step
                )
                events.append(event)
                self.fusion_events.append(event)
        
        return events
    
    def print_summary(self):
        """Print summary of detected structures"""
        print("\n" + "="*70)
        print("STELLAR STRUCTURES DETECTED")
        print("="*70)
        print(f"Total structures: {len(self.structures)}")
        
        if not self.structures:
            print("  (No large-scale structures detected yet)")
            return
        
        # Group by type
        by_type = {}
        for s in self.structures:
            by_type[s.structure_type] = by_type.get(s.structure_type, 0) + 1
        
        print("\nStructure Types:")
        for stype, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            print(f"  {stype:20s}: {count:3d}")
        
        # Details of most interesting structures
        print("\nNotable Structures:")
        for i, s in enumerate(sorted(self.structures, 
                                    key=lambda x: x.total_mass, 
                                    reverse=True)[:5]):
            print(f"\n  [{i+1}] {s.structure_type.upper()}")
            print(f"      Position: {s.position}")
            print(f"      Mass: {s.total_mass:.1f}")
            print(f"      Radius: {s.radius:.1f}")
            print(f"      Core density: {s.core_density:.2f}")
            print(f"      Temperature: {s.temperature:.2f}")
            print(f"      Potential: {s.potential_depth:.2f}")
            print(f"      Particles: {s.particle_count}")
            
            if s.is_fusion_active():
                print(f"      >>> FUSION ACTIVE <<<")
        
        # Fusion summary
        if self.fusion_events:
            print(f"\n{'='*70}")
            print(f"FUSION EVENTS DETECTED: {len(self.fusion_events)}")
            print(f"{'='*70}")
            print("  (Regions where new atoms are being created!)")
