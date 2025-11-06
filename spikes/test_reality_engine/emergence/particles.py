"""
Particle Emergence Detection

Particles should emerge as stable topological knots in the field.
No particle definitions - just persistent vortex structures.

Based on:
- Topological defect theory
- Vortex dynamics in field theory
- Memory-based stability detection

GPU-accelerated with PyTorch CUDA
"""

import torch
from typing import List, Dict, Tuple


class ParticleEmergence:
    """
    Identify particles as stable topological defects.
    
    Particles are NOT fundamental - they're emergent structures:
    - Stable vortices in the energy field
    - Persistent patterns in memory field
    - Topological knots that resist dissipation
    
    Mass = concentrated information density
    Charge = topological winding number
    Spin = angular momentum of vortex
    """
    
    def __init__(self, field):
        """
        Initialize particle detector.
        
        Args:
            field: DawnField instance to analyze
        """
        self.field = field
        self.stability_threshold = 1.0  # Memory persistence threshold
        self.vorticity_percentile = 95  # Top 5% vorticity regions
        
    def identify_particles(self) -> List[Dict]:
        """
        Find persistent excitations that behave like particles.
        
        Process:
        1. Compute vorticity (topological structure)
        2. Check memory (persistence/stability)
        3. Classify by topological properties
        4. Assign emergent properties (mass, charge, spin)
        
        Returns:
            List of detected particles with properties
        """
        # Compute vorticity field
        vorticity = self.compute_vorticity()
        
        # Find stable regions (high vorticity + high memory)
        stable_regions = self.find_stable_regions(vorticity)
        
        # Label connected components (move to CPU for scipy if available)
        try:
            from scipy.ndimage import label as scipy_label
            stable_cpu = stable_regions.cpu().numpy()
            labeled, num_particles = scipy_label(stable_cpu)
            labeled = torch.from_numpy(labeled).to(self.field.device)
        except ImportError:
            # Fallback: simple clustering
            labeled = stable_regions.int()
            num_particles = (stable_regions).sum().item()
        
        particles = []
        for particle_id in range(1, num_particles + 1):
            # Get region for this particle
            region = (labeled == particle_id)
            
            if region.sum() == 0:
                continue
            
            # Compute particle properties
            properties = self.compute_particle_properties(region, vorticity)
            
            # Add particle if sufficiently stable
            if properties and properties['stability'] > self.stability_threshold:
                properties['id'] = particle_id
                particles.append(properties)
        
        return particles
    
    def compute_vorticity(self) -> torch.Tensor:
        """
        Compute vorticity of energy field.
        
        Vorticity = curl magnitude = |∇ × E|
        
        For 3D scalar field, use gradient magnitude as proxy:
        vorticity ≈ |∇E|
        
        Returns:
            Vorticity field
        """
        # Compute gradients using torch.roll for periodic boundaries (fully vectorized)
        grad_x = (torch.roll(self.field.E, shifts=-1, dims=0) - torch.roll(self.field.E, shifts=1, dims=0)) / 2.0
        grad_y = (torch.roll(self.field.E, shifts=-1, dims=1) - torch.roll(self.field.E, shifts=1, dims=1)) / 2.0
        grad_z = (torch.roll(self.field.E, shifts=-1, dims=2) - torch.roll(self.field.E, shifts=1, dims=2)) / 2.0
        
        # Magnitude (fully vectorized)
        vorticity = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        return vorticity
    
    def find_stable_regions(self, vorticity: torch.Tensor) -> torch.Tensor:
        """
        Find regions that are both high vorticity AND high memory.
        
        High vorticity = topological structure
        High memory = persistence over time
        
        Args:
            vorticity: Vorticity field
            
        Returns:
            Boolean mask of stable regions
        """
        # High vorticity regions
        vorticity_threshold = torch.quantile(vorticity, self.vorticity_percentile / 100.0)
        high_vorticity = vorticity > vorticity_threshold
        
        # High memory regions (stability)
        high_memory = self.field.M > self.stability_threshold
        
        # Both conditions must be met
        stable = high_vorticity & high_memory
        
        return stable
    
    def compute_particle_properties(self, region: torch.Tensor, 
                                    vorticity: torch.Tensor) -> Dict:
        """
        Compute emergent properties of a particle.
        
        Args:
            region: Boolean mask of particle location
            vorticity: Vorticity field
            
        Returns:
            Dictionary of particle properties
        """
        # Get region coordinates
        coords = torch.nonzero(region, as_tuple=False)
        
        if len(coords) == 0:
            return {}
        
        # Position (center of mass) - move to CPU
        position = tuple(coords.float().mean(dim=0).cpu().tolist())
        
        # Mass (information density)
        mass = self.field.I[region].sum().item()
        
        # Stability (memory accumulation)
        stability = self.field.M[region].mean().item()
        
        # Energy
        energy = self.field.E[region].mean().item()
        
        # Topological charge (winding number) - simplified
        charge = self.compute_topological_charge(region, vorticity)
        
        # Spin (angular momentum) - simplified
        spin = self.compute_angular_momentum(coords, position)
        
        # Size (radius)
        size = self.compute_radius(coords, position)
        
        return {
            'position': position,
            'mass': mass,
            'energy': energy,
            'stability': stability,
            'charge': charge,
            'spin': spin,
            'size': size,
            'volume': len(coords)
        }
    
    def compute_center_of_mass(self, coords: torch.Tensor) -> Tuple[float, float, float]:
        """
        Compute center of mass of particle.
        
        Args:
            coords: Tensor of coordinates (N x 3)
            
        Returns:
            (x, y, z) center of mass
        """
        center = coords.float().mean(dim=0).cpu()
        return (float(center[0]), float(center[1]), float(center[2]))
    
    def compute_mass(self, region: torch.Tensor) -> float:
        """
        Compute mass of particle.
        
        Mass = integrated information density
        
        Args:
            region: Boolean mask of particle
            
        Returns:
            Particle mass
        """
        # Mass is concentrated information
        mass = self.field.I[region].sum().item()
        
        return float(mass)
    
    def compute_topological_charge(self, region: torch.Tensor, 
                                   vorticity: torch.Tensor) -> int:
        """
        Compute topological winding number (charge).
        
        This is the topological classification of the vortex.
        Integer winding = bosonic
        Half-integer winding = fermionic
        
        Args:
            region: Particle region
            vorticity: Vorticity field
            
        Returns:
            Topological charge (integer)
        """
        # Simplified: use vorticity integral around boundary
        # Full implementation would compute actual winding number
        
        total_vorticity = vorticity[region].sum().item()
        
        # Quantize to nearest integer
        charge = int(round(total_vorticity / (2 * 3.14159265359)))
        
        return charge
    
    def compute_angular_momentum(self, coords: torch.Tensor,
                                center: Tuple[float, float, float]) -> torch.Tensor:
        """
        Compute angular momentum (spin) of particle.
        
        L = r × p (classical)
        For field: L ≈ ∫ r × ∇E dV
        
        Args:
            coords: Coordinate tensor (N x 3)
            center: Center of mass
            
        Returns:
            Angular momentum vector (Lx, Ly, Lz)
        """
        # Relative positions
        rx = coords[:, 0].float() - center[0]
        ry = coords[:, 1].float() - center[1]
        rz = coords[:, 2].float() - center[2]
        
        # Compute field momentum (gradient) at particle locations
        # Extract gradients at these specific coordinates
        grad_x = torch.zeros(len(coords), device=self.field.device)
        grad_y = torch.zeros(len(coords), device=self.field.device)
        grad_z = torch.zeros(len(coords), device=self.field.device)
        
        for i, coord in enumerate(coords):
            x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
            if 1 <= x < self.field.E.shape[0]-1:
                grad_x[i] = (self.field.E[x+1, y, z] - self.field.E[x-1, y, z]) / 2.0
            if 1 <= y < self.field.E.shape[1]-1:
                grad_y[i] = (self.field.E[x, y+1, z] - self.field.E[x, y-1, z]) / 2.0
            if 1 <= z < self.field.E.shape[2]-1:
                grad_z[i] = (self.field.E[x, y, z+1] - self.field.E[x, y, z-1]) / 2.0
        
        # Angular momentum: L = r × p
        Lx = (ry * grad_z - rz * grad_y).sum()
        Ly = (rz * grad_x - rx * grad_z).sum()
        Lz = (rx * grad_y - ry * grad_x).sum()
        
        return torch.tensor([Lx.item(), Ly.item(), Lz.item()])
    
    def compute_radius(self, coords: torch.Tensor,
                      center: Tuple[float, float, float]) -> float:
        """
        Compute effective radius of particle.
        
        Args:
            coords: Coordinate tensor (N x 3)
            center: Center of mass
            
        Returns:
            RMS radius
        """
        # Distances from center
        dx = coords[:, 0].float() - center[0]
        dy = coords[:, 1].float() - center[1]
        dz = coords[:, 2].float() - center[2]
        
        distances = torch.sqrt(dx**2 + dy**2 + dz**2)
        
        # RMS radius
        radius = torch.sqrt((distances**2).mean()).item()
        
        return float(radius)
    
    def classify_particle_type(self, particle: Dict) -> str:
        """
        Classify particle by its topological properties.
        
        Args:
            particle: Particle property dictionary
            
        Returns:
            Particle type string
        """
        charge = particle['charge']
        spin_magnitude = torch.linalg.norm(torch.tensor(particle['spin'])).item()
        
        # Integer charge = boson
        # Half-integer spin = fermion (approximately)
        
        if abs(charge) == 0:
            return "scalar"
        elif abs(charge) == 1:
            if spin_magnitude < 0.5:
                return "fermion"  # Spin-0 fermion (unlikely)
            else:
                return "gauge_boson"
        elif abs(charge) == 2:
            return "composite_boson"
        else:
            return "exotic"
    
    def get_particle_statistics(self, particles: List[Dict]) -> Dict:
        """
        Get statistical summary of detected particles.
        
        Args:
            particles: List of particles
            
        Returns:
            Statistics dictionary
        """
        if not particles:
            return {
                'count': 0,
                'mean_mass': 0,
                'mean_energy': 0,
                'mean_stability': 0
            }
        
        masses = [p['mass'] for p in particles]
        energies = [p['energy'] for p in particles]
        stabilities = [p['stability'] for p in particles]
        sizes = [p['size'] for p in particles]
        charges = [p['charge'] for p in particles]
        
        return {
            'count': len(particles),
            'mean_mass': sum(masses) / len(masses),
            'mean_energy': sum(energies) / len(energies),
            'mean_stability': sum(stabilities) / len(stabilities),
            'mean_size': sum(sizes) / len(sizes),
            'total_charge': sum(charges),
            'charge_distribution': torch.bincount(torch.tensor([abs(p['charge']) for p in particles])).tolist()
        }
    
    def __repr__(self) -> str:
        """String representation."""
        particles = self.identify_particles()
        return f"ParticleEmergence(detected={len(particles)} particles)"
