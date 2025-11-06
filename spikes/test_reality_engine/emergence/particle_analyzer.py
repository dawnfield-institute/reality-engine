"""
Particle Analyzer - Detect and Classify Emergent Particles

Identifies stable structures in the field and classifies them by:
- Mass (memory field concentration)
- Charge (energy field circulation)  
- Spin (angular momentum in field)
- Stability (lifetime before decay)

Builds an emergent periodic table from pure field dynamics.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy import ndimage
from scipy.ndimage import label, center_of_mass


@dataclass
class Particle:
    """Represents a detected particle with quantum properties"""
    position: Tuple[int, int, int]
    mass: float  # Memory field integral
    charge: float  # Energy field circulation
    spin: float  # Angular momentum
    radius: float  # Characteristic size
    stability: float  # Lifetime/decay rate
    binding_energy: float  # How tightly bound
    quantum_numbers: Dict[str, float]  # Additional quantum properties
    
    @property
    def classification(self) -> str:
        """Classify particle type based on properties"""
        if self.mass < 0.01:
            return "photon" if abs(self.charge) < 0.001 else "neutrino"
        elif self.mass < 0.1:
            if abs(self.charge) > 0.5:
                return "electron" if self.charge < 0 else "positron"
            else:
                return "meson"
        elif self.mass < 1.0:
            if abs(self.spin - 0.5) < 0.1:
                return "fermion"
            else:
                return "boson"
        else:
            # Heavy particles
            if abs(self.charge) < 0.1:
                return "neutron"
            elif abs(self.charge - 1.0) < 0.1:
                return "proton"
            else:
                return "exotic"
    
    def __hash__(self):
        return hash((self.position, round(self.mass, 3), round(self.charge, 3)))
    
    def __repr__(self):
        return (f"Particle({self.classification}, pos={self.position}, "
                f"m={self.mass:.3f}, q={self.charge:+.3f}, "
                f"spin={self.spin:.2f}, stability={self.stability:.2f})")


class ParticleAnalyzer:
    """Analyzes field to detect and classify particles"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.particles = []
        self.periodic_table = {}
        
    def detect_particles(self, E, I, M, threshold=0.1, stability_threshold=0.01) -> List[Particle]:
        """
        Detect stable vortices/solitons in the field
        
        Args:
            E: Energy field
            I: Information field  
            M: Memory field
            threshold: Detection threshold
            
        Returns:
            List of detected particles
        """
        particles = []
        
        # Convert to numpy for scipy operations
        E_np = E.cpu().numpy() if torch.is_tensor(E) else E
        I_np = I.cpu().numpy() if torch.is_tensor(I) else I
        M_np = M.cpu().numpy() if torch.is_tensor(M) else M
        
        # Find local maxima in Memory field (stable structures)
        from scipy.ndimage import maximum_filter, generate_binary_structure
        structure = generate_binary_structure(3, 2)
        local_max = maximum_filter(M_np, footprint=structure) == M_np
        
        # Threshold to get significant peaks
        peaks = local_max & (M_np > threshold)
        labeled, num_features = label(peaks)
        
        print(f"  Found {num_features} candidate particle sites")
        
        # Analyze each detected feature
        for i in range(1, min(num_features + 1, 200)):  # Limit to 200 particles
            mask = labeled == i
            
            # Get position (center of mass)
            pos = center_of_mass(M_np * mask)
            pos = tuple(int(p) for p in pos)
            
            # Skip if at boundary
            if (pos[0] < 3 or pos[0] > M_np.shape[0]-3 or
                pos[1] < 3 or pos[1] > M_np.shape[1]-3 or
                pos[2] < 3 or pos[2] > M_np.shape[2]-3):
                continue
            
            # Extract local field region (7x7x7 around peak)
            x, y, z = pos
            E_local = E_np[x-3:x+4, y-3:y+4, z-3:z+4]
            I_local = I_np[x-3:x+4, y-3:y+4, z-3:z+4]
            M_local = M_np[x-3:x+4, y-3:y+4, z-3:z+4]
            
            # Verify we got the right shape (edge cases)
            if E_local.shape != (7, 7, 7) or I_local.shape != (7, 7, 7) or M_local.shape != (7, 7, 7):
                continue
            
            # Calculate particle properties
            particle = self._analyze_local_field(
                E_local, I_local, M_local, pos
            )
            
            if particle and particle.stability > stability_threshold:  # Use parameter
                particles.append(particle)
        
        self.particles = particles
        print(f"  Detected {len(particles)} stable particles")
        return particles
    
    def _analyze_local_field(self, E, I, M, position) -> Optional[Particle]:
        """Analyze local field configuration to extract particle properties"""
        
        # Mass = total memory in region
        mass = np.sum(M)
        
        if mass < 1e-6:
            return None
        
        # Charge = circulation of E field (curl)
        charge = self._calculate_circulation(E)
        
        # Spin = angular momentum of field
        spin = self._calculate_angular_momentum(E, I, M)
        
        # Radius = RMS distance from center weighted by M
        radius = self._calculate_radius(M)
        
        # Stability = how well localized (inverse of spread)
        stability = 1.0 / (1.0 + np.std(M) + 1e-6)
        
        # Binding energy = difference from sum of parts
        binding_energy = self._calculate_binding_energy(E, I, M)
        
        # Additional quantum numbers
        quantum_numbers = {
            'isospin': self._calculate_isospin(E, I),
            'baryon_number': self._calculate_baryon_number(M, mass),
            'lepton_number': self._calculate_lepton_number(M),
        }
        
        return Particle(
            position=position,
            mass=mass,
            charge=charge,
            spin=spin,
            radius=radius,
            stability=stability,
            binding_energy=binding_energy,
            quantum_numbers=quantum_numbers
        )
    
    def _calculate_circulation(self, E):
        """Calculate field circulation (curl · n integrated)"""
        # Approximate curl using finite differences
        dy_E = np.gradient(E, axis=1)
        dx_E = np.gradient(E, axis=0)
        curl_z = dx_E[:,:,3] - dy_E[:,:,3]  # z-component of curl
        return np.sum(curl_z)
    
    def _calculate_angular_momentum(self, E, I, M):
        """Calculate total angular momentum"""
        # Verify shape
        if E.shape != (7, 7, 7):
            return 0.0
            
        # Create coordinate grids
        x, y, z = np.meshgrid(range(7), range(7), range(7), indexing='ij')
        x, y, z = x - 3, y - 3, z - 3  # Center at origin
        
        # Angular momentum L = r × p, where p ~ E gradient
        # Use simple centered differences
        grad_x = (np.roll(E, -1, axis=0) - np.roll(E, 1, axis=0)) / 2.0
        grad_y = (np.roll(E, -1, axis=1) - np.roll(E, 1, axis=1)) / 2.0
        
        # Ensure correct shape
        if grad_x.shape != (7, 7, 7) or grad_y.shape != (7, 7, 7):
            return 0.0
        
        L_z = x * grad_y - y * grad_x  # z-component of L = r × p
        
        total_M = np.sum(M) + 1e-10
        return np.sum(L_z * M) / total_M
    
    def _calculate_radius(self, M):
        """Calculate RMS radius"""
        x, y, z = np.meshgrid(range(7), range(7), range(7), indexing='ij')
        x, y, z = x - 3, y - 3, z - 3
        r2 = x**2 + y**2 + z**2
        
        total_M = np.sum(M) + 1e-10
        return np.sqrt(np.sum(r2 * M) / total_M)
    
    def _calculate_binding_energy(self, E, I, M):
        """Calculate binding energy from field configuration"""
        # Binding = how much energy would be released if split
        center_val = E[3,3,3] + I[3,3,3] + M[3,3,3]
        surround_val = np.sum(E) + np.sum(I) + np.sum(M) - center_val
        return center_val - 0.5 * surround_val
    
    def _calculate_isospin(self, E, I):
        """Calculate isospin from E-I asymmetry"""
        return (np.sum(E) - np.sum(I)) / (np.sum(E) + np.sum(I) + 1e-10)
    
    def _calculate_lepton_number(self, M):
        """Lepton number from memory chirality"""
        # Check for left/right asymmetry
        left = np.sum(M[:3, :, :])
        right = np.sum(M[4:, :, :])
        return (left - right) / (left + right + 1e-10)
    
    def _calculate_baryon_number(self, M, mass):
        """Baryon number from mass quantization"""
        # Baryons have mass ~1 in natural units
        if mass > 0.8:
            return round(mass)
        return 0
    
    def build_periodic_table(self, particles: List[Particle]) -> Dict:
        """
        Build periodic table from detected particles
        
        Groups particles by similar properties
        """
        table = {}
        
        for particle in particles:
            key = particle.classification
            
            if key not in table:
                table[key] = {
                    'count': 0,
                    'avg_mass': 0.0,
                    'avg_charge': 0.0,
                    'avg_spin': 0.0,
                    'mass_range': [float('inf'), float('-inf')],
                    'charge_range': [float('inf'), float('-inf')],
                    'instances': []
                }
            
            entry = table[key]
            entry['count'] += 1
            entry['avg_mass'] += particle.mass
            entry['avg_charge'] += particle.charge
            entry['avg_spin'] += particle.spin
            entry['mass_range'][0] = min(entry['mass_range'][0], particle.mass)
            entry['mass_range'][1] = max(entry['mass_range'][1], particle.mass)
            entry['charge_range'][0] = min(entry['charge_range'][0], particle.charge)
            entry['charge_range'][1] = max(entry['charge_range'][1], particle.charge)
            entry['instances'].append(particle)
        
        # Normalize averages
        for key in table:
            n = table[key]['count']
            table[key]['avg_mass'] /= n
            table[key]['avg_charge'] /= n
            table[key]['avg_spin'] /= n
        
        self.periodic_table = table
        return table
    
    def find_composite_structures(self, particles: List[Particle]) -> List[Dict]:
        """Find atoms/molecules (composite structures)"""
        composites = []
        
        if len(particles) < 2:
            return composites
        
        # Calculate median separation for reference
        separations = []
        for i, p1 in enumerate(particles[:min(50, len(particles))]):  # Sample
            for j in range(i+1, min(i+6, len(particles))):
                p2 = particles[j]
                dist = np.linalg.norm(
                    np.array(p1.position) - np.array(p2.position)
                )
                separations.append(dist)
        
        if not separations:
            return composites
            
        median_sep = np.median(separations)
        
        # Look for bound pairs/groups
        for i, p1 in enumerate(particles):
            for j in range(i+1, len(particles)):
                p2 = particles[j]
                
                dist = np.linalg.norm(
                    np.array(p1.position) - np.array(p2.position)
                )
                
                # Particles are "bound" if closer than 0.5 * median separation
                # This adapts to the particle distribution
                binding_threshold = 0.5 * median_sep
                
                if dist < binding_threshold:
                    # Calculate binding type
                    charge_product = p1.charge * p2.charge
                    
                    if charge_product < -0.01:
                        bond_type = 'ionic'  # Opposite charges
                    elif abs(p1.charge) < 0.1 and abs(p2.charge) < 0.1:
                        bond_type = 'covalent'  # Both neutral - shared field
                    elif charge_product > 0:
                        bond_type = 'metallic'  # Same charge but still close
                    else:
                        bond_type = 'weak'
                    
                    composites.append({
                        'type': bond_type,
                        'particles': [p1, p2],
                        'binding_strength': (p1.mass * p2.mass) / (dist**2 + 0.1),
                        'total_mass': p1.mass + p2.mass,
                        'net_charge': p1.charge + p2.charge,
                        'separation': dist,
                        'binding_threshold': binding_threshold
                    })
        
        return composites
    
    def print_summary(self):
        """Print summary of detected particles"""
        print("\n" + "="*60)
        print("PARTICLE DETECTION SUMMARY")
        print("="*60)
        print(f"Total particles detected: {len(self.particles)}")
        print(f"Unique particle types: {len(self.periodic_table)}")
        
        if self.periodic_table:
            print("\nParticle Types:")
            for ptype, data in sorted(self.periodic_table.items(), 
                                     key=lambda x: x[1]['count'], reverse=True):
                print(f"  {ptype:12s}: {data['count']:3d} particles, "
                      f"mass={data['avg_mass']:7.3f}, "
                      f"charge={data['avg_charge']:+7.3f}, "
                      f"spin={data['avg_spin']:6.2f}")
        
        # Charge balance
        total_charge = sum(p.charge for p in self.particles)
        pos_charge = sum(p.charge for p in self.particles if p.charge > 0)
        neg_charge = sum(p.charge for p in self.particles if p.charge < 0)
        
        print(f"\nCharge Balance:")
        print(f"  Positive: {pos_charge:+.3f}")
        print(f"  Negative: {neg_charge:+.3f}")
        print(f"  Net:      {total_charge:+.3f}")
        print(f"  Balance:  {abs(total_charge)/(abs(pos_charge)+abs(neg_charge)+1e-10)*100:.1f}%")
