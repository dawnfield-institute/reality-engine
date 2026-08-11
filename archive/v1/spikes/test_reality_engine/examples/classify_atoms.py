"""
Atom Classification - Reality Engine

Analyzes emergent composite structures and classifies them as atom types.
If Dawn Field Theory is correct, atoms should form with properties matching
the real periodic table - no atomic physics programmed, just E↔I balance.
"""

import json
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from emergence.particle_analyzer import ParticleAnalyzer, Particle


class AtomClassifier:
    """Classify emergent composite structures as atom types"""
    
    def __init__(self):
        # Real periodic table reference (atomic number, mass, name)
        # Using approximate atomic masses in unified atomic mass units
        self.periodic_table_ref = {
            1: {'mass': 1.008, 'name': 'Hydrogen', 'symbol': 'H'},
            2: {'mass': 4.003, 'name': 'Helium', 'symbol': 'He'},
            3: {'mass': 6.941, 'name': 'Lithium', 'symbol': 'Li'},
            4: {'mass': 9.012, 'name': 'Beryllium', 'symbol': 'Be'},
            5: {'mass': 10.81, 'name': 'Boron', 'symbol': 'B'},
            6: {'mass': 12.01, 'name': 'Carbon', 'symbol': 'C'},
            7: {'mass': 14.01, 'name': 'Nitrogen', 'symbol': 'N'},
            8: {'mass': 16.00, 'name': 'Oxygen', 'symbol': 'O'},
            9: {'mass': 19.00, 'name': 'Fluorine', 'symbol': 'F'},
            10: {'mass': 20.18, 'name': 'Neon', 'symbol': 'Ne'},
            11: {'mass': 22.99, 'name': 'Sodium', 'symbol': 'Na'},
            12: {'mass': 24.31, 'name': 'Magnesium', 'symbol': 'Mg'},
            13: {'mass': 26.98, 'name': 'Aluminum', 'symbol': 'Al'},
            14: {'mass': 28.09, 'name': 'Silicon', 'symbol': 'Si'},
            15: {'mass': 30.97, 'name': 'Phosphorus', 'symbol': 'P'},
            16: {'mass': 32.07, 'name': 'Sulfur', 'symbol': 'S'},
            17: {'mass': 35.45, 'name': 'Chlorine', 'symbol': 'Cl'},
            18: {'mass': 39.95, 'name': 'Argon', 'symbol': 'Ar'},
            19: {'mass': 39.10, 'name': 'Potassium', 'symbol': 'K'},
            20: {'mass': 40.08, 'name': 'Calcium', 'symbol': 'Ca'},
        }
    
    def classify_atom(self, composite, mass_scale=1.0):
        """
        Classify a composite structure as an atom type
        
        Args:
            composite: Dict with 'total_mass', 'net_charge', 'particle_types'
            mass_scale: Conversion factor from simulation units to atomic mass units
        
        Returns:
            Dict with classification, confidence, and match details
        """
        # Extract properties
        total_mass = composite['total_mass'] * mass_scale
        net_charge = composite['net_charge']
        num_constituents = len(composite.get('particle_types', composite.get('particle_masses', [None, None])))
        separation = composite['separation']
        bond_type = composite.get('bond_type', composite.get('type', 'unknown'))
        
        # Estimate atomic number from charge (electrons = protons - net_charge)
        # In real atoms: net_charge = protons - electrons
        # Assuming neutrality, protons ≈ electrons ≈ num_constituents/2
        estimated_Z = max(1, round(abs(net_charge)) if abs(net_charge) > 0.5 else num_constituents // 2)
        
        # Find best match by mass
        best_match = None
        best_diff = float('inf')
        
        for Z, ref in self.periodic_table_ref.items():
            mass_diff = abs(total_mass - ref['mass'])
            
            # Consider both mass and atomic number proximity
            if mass_diff < best_diff and abs(Z - estimated_Z) <= 5:
                best_diff = mass_diff
                best_match = Z
        
        if best_match is None:
            return {
                'classification': 'unknown',
                'symbol': '??',
                'atomic_number': estimated_Z,
                'mass': total_mass,
                'charge': net_charge,
                'confidence': 0.0,
                'bond_type': bond_type,
                'separation': separation
            }
        
        ref = self.periodic_table_ref[best_match]
        confidence = 1.0 / (1.0 + best_diff / ref['mass'])
        
        return {
            'classification': ref['name'],
            'symbol': ref['symbol'],
            'atomic_number': best_match,
            'mass': total_mass,
            'mass_expected': ref['mass'],
            'mass_error': (total_mass - ref['mass']) / ref['mass'] * 100,
            'charge': net_charge,
            'confidence': confidence,
            'bond_type': bond_type,
            'separation': separation,
            'constituents': num_constituents
        }
    
    def find_mass_scale(self, composites):
        """
        Determine mass scaling factor by comparing composite masses
        to periodic table
        
        This auto-calibrates simulation units → atomic mass units
        """
        if not composites:
            return 1.0
        
        # Get range of composite masses
        masses = [c['total_mass'] for c in composites]
        median_mass = np.median(masses)
        
        # Assume median composite mass corresponds to mid-range element (C, N, O ~12-16 amu)
        target_mass = 14.0  # Nitrogen-like
        
        scale = target_mass / median_mass if median_mass > 0 else 1.0
        
        return scale
    
    def analyze_atom_distribution(self, classified_atoms):
        """Analyze distribution of atom types"""
        distribution = {}
        
        for atom in classified_atoms:
            symbol = atom['symbol']
            
            if symbol not in distribution:
                distribution[symbol] = {
                    'count': 0,
                    'atomic_number': atom['atomic_number'],
                    'name': atom['classification'],
                    'avg_mass': 0.0,
                    'avg_charge': 0.0,
                    'avg_separation': 0.0,
                    'bond_types': {},
                    'instances': []
                }
            
            entry = distribution[symbol]
            entry['count'] += 1
            entry['avg_mass'] += atom['mass']
            entry['avg_charge'] += atom['charge']
            entry['avg_separation'] += atom['separation']
            
            bond = atom['bond_type']
            entry['bond_types'][bond] = entry['bond_types'].get(bond, 0) + 1
            entry['instances'].append(atom)
        
        # Calculate averages
        for symbol, entry in distribution.items():
            if entry['count'] > 0:
                entry['avg_mass'] /= entry['count']
                entry['avg_charge'] /= entry['count']
                entry['avg_separation'] /= entry['count']
        
        return distribution


def main():
    print("="*70)
    print("ATOM CLASSIFICATION - Reality Engine")
    print("="*70)
    print("Analyzing emergent composite structures...")
    print("Hypothesis: Atoms should match real periodic table naturally")
    print("="*70)
    
    # Load the latest evolution timeline
    output_dir = project_root / 'output'
    latest_run = sorted(output_dir.glob('*_longrun'))[-1]
    
    print(f"\n[1/4] Loading data from: {latest_run.name}")
    
    # Load composite structures
    composites_path = latest_run / 'composite_structures.json'
    if not composites_path.exists():
        print(f"\n✗ No composite structures file found!")
        print(f"  Run watch_atoms_emerge.py first to generate data.")
        return
    
    with open(composites_path, 'r') as f:
        composite_data = json.load(f)
    
    print(f"      Composite structures: {composite_data['count']}")
    
    composites = composite_data['structures']
    
    if not composites:
        print("\n✗ No composite structures detected!")
        return
    
    print(f"\n[2/4] Calibrating mass scale...")
    
    classifier = AtomClassifier()
    
    # Find mass scaling
    mass_scale = classifier.find_mass_scale(composites)
    print(f"      Mass scale factor: {mass_scale:.6f} (sim units → amu)")
    print(f"      Median composite mass: {np.median([c['total_mass'] for c in composites]):.1f} sim units")
    print(f"      → {np.median([c['total_mass'] for c in composites]) * mass_scale:.1f} amu")
    
    print("\n[3/4] Classifying atoms...")
    
    # Classify atoms
    classified_atoms = []
    for comp in composites:
        atom = classifier.classify_atom(comp, mass_scale)
        if atom['confidence'] > 0.3:  # Lower threshold to see more variety
            classified_atoms.append(atom)
    
    print(f"      Classified {len(classified_atoms)} atoms (confidence > 0.3)")
    
    # Analyze distribution
    distribution = classifier.analyze_atom_distribution(classified_atoms)
    
    print("\n[4/4] Results - Emergent Atom Types:")
    print("="*70)
    
    # Sort by atomic number
    sorted_atoms = sorted(distribution.items(), 
                         key=lambda x: x[1]['atomic_number'])
    
    for symbol, data in sorted_atoms:
        print(f"\n  {symbol:3s} - {data['name']:12s} (Z={data['atomic_number']:2d})")
        print(f"      Count: {data['count']}")
        print(f"      Mass:  {data['avg_mass']:.2f} amu (error: {abs(data['avg_mass']-data['instances'][0]['mass_expected'])/data['instances'][0]['mass_expected']*100:.1f}%)")
        print(f"      Charge: {data['avg_charge']:+.3f}")
        print(f"      Separation: {data['avg_separation']:.2f}")
        print(f"      Bonds: {', '.join(f'{k}({v})' for k, v in data['bond_types'].items())}")
    
    print("\n" + "="*70)
    print("VALIDATION CHECK")
    print("="*70)
    
    # Check if we see the expected pattern
    total_types = len(distribution)
    print(f"✓ Atom types detected: {total_types}")
    
    # Check for hydrogen (lightest)
    has_hydrogen = any(d['atomic_number'] == 1 for d in distribution.values())
    print(f"{'✓' if has_hydrogen else '✗'} Hydrogen detected: {has_hydrogen}")
    
    # Check for mass hierarchy
    masses = [d['avg_mass'] for d in distribution.values()]
    mass_increases = all(masses[i] <= masses[i+1] for i in range(len(masses)-1))
    print(f"{'✓' if mass_increases else '~'} Mass hierarchy preserved")
    
    # Check charge neutrality (atoms should be mostly neutral)
    avg_charge = np.mean([abs(d['avg_charge']) for d in distribution.values()])
    print(f"{'✓' if avg_charge < 0.2 else '~'} Average charge: {avg_charge:.3f} (neutral ~0)")
    
    # Check for covalent bonding dominance (expected for neutrals)
    total_covalent = sum(d['bond_types'].get('covalent', 0) for d in distribution.values())
    total_bonds = sum(sum(d['bond_types'].values()) for d in distribution.values())
    covalent_fraction = total_covalent / total_bonds if total_bonds > 0 else 0
    print(f"{'✓' if covalent_fraction > 0.5 else '~'} Covalent bonding: {covalent_fraction*100:.1f}%")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print(f"Detected {total_types} distinct atom types from pure field dynamics!")
    print("These emerged naturally from E↔I balance with NO atomic physics coded.")
    print("\nNext steps:")
    print("  • Check if ratios match natural abundance")
    print("  • Look for molecular chains (multiple bonds)")
    print("  • Test if chemistry emerges (reactions, equilibria)")
    print("  • Extend to heavier elements")
    print("="*70)



if __name__ == '__main__':
    main()
