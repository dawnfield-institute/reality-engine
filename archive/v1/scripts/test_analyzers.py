"""
Test the new modular analyzer system.

Runs a simulation and uses multiple analyzers to detect:
- Gravitational forces
- Conservation laws
- Atomic structures
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'analyzers'))

from core.reality_engine import RealityEngine
from analyzers.laws.gravity_analyzer import GravityAnalyzer
from analyzers.laws.conservation_analyzer import ConservationAnalyzer
from analyzers.matter.atom_detector import AtomDetector
from analyzers.cosmic.star_detector import StarDetector
from analyzers.cosmic.quantum_detector import QuantumDetector
from analyzers.cosmic.galaxy_analyzer import GalaxyAnalyzer
from tools.emergence_observer import EmergenceObserver


def main():
    print("="*70)
    print("MODULAR ANALYZER DEMONSTRATION")
    print("="*70)
    print("Running simulation with multiple independent analyzers...")
    print()
    
    # Initialize reality engine
    print("Initializing Reality Engine (64×16 grid)...")
    engine = RealityEngine(size=(64, 16))
    engine.initialize()
    observer = EmergenceObserver()
    
    # Initialize analyzers with unit calibration
    print("Initializing Analyzers...")
    
    # Gravity analyzer (calibrated to atomic scale for demonstration)
    gravity = GravityAnalyzer(
        length_scale=1e-10,  # 1 Ångström per grid unit (atomic scale)
        mass_scale=1.67e-27, # ~proton mass per mass unit
        time_scale=1e-15,    # 1 femtosecond per time unit
        min_confidence=0.5
    )
    
    # Conservation analyzer (strict thresholds)
    conservation = ConservationAnalyzer(min_confidence=0.9)
    
    # Atom detector
    atoms = AtomDetector(min_confidence=0.6)
    
    # Star detector
    stars = StarDetector(min_confidence=0.7)
    
    # Quantum detector
    quantum = QuantumDetector(min_confidence=0.6)
    
    # Galaxy analyzer
    galaxies = GalaxyAnalyzer(min_confidence=0.65)
    
    analyzers = [gravity, conservation, atoms, stars, quantum, galaxies]
    
    print(f"  • Gravity Analyzer (atomic scale calibration)")
    print(f"  • Conservation Analyzer")
    print(f"  • Atom Detector")
    print(f"  • Star Detector")
    print(f"  • Quantum Detector")
    print(f"  • Galaxy Analyzer")
    print(f"  • Atom Detector")
    print()
    
    # Run simulation
    num_steps = 1000
    print(f"Running {num_steps} steps...")
    print()
    
    for step in range(num_steps):
        # Evolve reality
        state = engine.step()
        
        # Observe structures
        structures = observer.observe(engine.current_state)
        
        # Prepare state for analyzers
        analyzer_state = {
            'actual': engine.current_state.actual,
            'potential': engine.current_state.potential,
            'memory': engine.current_state.memory,
            'temperature': engine.current_state.temperature,
            'step': step,
            'time': engine.time_elapsed,
            'structures': structures
        }
        
        # Run all analyzers
        for analyzer in analyzers:
            detections = analyzer.update(analyzer_state)
            
            # Print significant detections
            for d in detections:
                if d.confidence > 0.8:
                    print(f"[{step:4d}] {analyzer.name}: {d}")
        
        # Progress indicator
        if (step + 1) % 100 == 0:
            print(f"\nProgress: {step + 1}/{num_steps} steps")
            print(f"  Structures: {len(structures)}")
            print()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    
    # Print summaries from each analyzer
    for analyzer in analyzers:
        analyzer.print_summary()
    
    # Detailed gravity analysis
    print("\n" + "="*70)
    print("GRAVITY FORCE COMPARISON")
    print("="*70)
    
    force_stats = gravity.get_force_statistics()
    G_evolution = gravity.get_G_evolution()
    
    if force_stats:
        print(f"\nForce Measurements:")
        print(f"  Total measurements: {force_stats['total_measurements']}")
        print(f"  Mean force: {force_stats['force']['mean']:.3e} (simulation units)")
        print(f"  Force range: {force_stats['force']['min']:.3e} to {force_stats['force']['max']:.3e}")
        print(f"  Mean distance: {force_stats['distance']['mean']:.2f} grid units")
    
    if G_evolution:
        print(f"\nGravitational Constant:")
        print(f"  Mean G (simulation): {G_evolution['mean_G']:.3e}")
        print(f"  Std G: {G_evolution['std_G']:.3e}")
        print(f"  Mean power law: n = {G_evolution['mean_n']:.3f} (Newton = 2.0)")
        
        # Convert to physical units
        G_physical = G_evolution['mean_G'] * (1e-10**3) / (1.67e-27 * 1e-15**2)
        G_real = 6.67430e-11
        ratio = G_physical / G_real
        
        print(f"\nPhysical Unit Conversion:")
        print(f"  G (physical): {G_physical:.3e} m³/kg·s²")
        print(f"  G (reality): {G_real:.3e} m³/kg·s²")
        print(f"  Ratio: {ratio:.3f}x")
        
        if 0.1 < ratio < 10:
            print(f"  [OK] Within an order of magnitude of reality!")
        elif ratio < 0.1:
            print(f"  [!] Weaker than reality by {1/ratio:.1f}x")
        else:
            print(f"  [!] Stronger than reality by {ratio:.1f}x")
    
    # Conservation summary
    print("\n" + "="*70)
    print("CONSERVATION LAWS SUMMARY")
    print("="*70)
    
    cons_summary = conservation.get_conservation_summary()
    for law, data in cons_summary.items():
        print(f"\n{law}:")
        print(f"  Mean value: {data['mean']:.3e}")
        print(f"  Variation: {data['relative_variation']*100:.2f}%")
        print(f"  Conserved: {'[YES]' if data.get('conserved', False) else '[NO]'}")
    
    # Atom/matter summary
    print("\n" + "="*70)
    print("MATTER STRUCTURE SUMMARY")
    print("="*70)
    
    mass_dist = atoms.get_mass_distribution()
    if mass_dist:
        print(f"\nMass Distribution:")
        print(f"  Total structures analyzed: {mass_dist['total_structures']}")
        print(f"  Distinct mass levels: {mass_dist['num_mass_levels']}")
        print(f"  Mean mass: {mass_dist['mean_mass']:.3f}")
        print(f"  Std mass: {mass_dist['std_mass']:.3f}")
        
        # Show mass peaks
        if mass_dist['num_mass_levels'] > 0:
            print(f"\nMass peaks (like periodic table):")
            for m, c in zip(mass_dist['masses'][:10], mass_dist['counts'][:10]):
                print(f"    {m:.2f}: {c} structures")
    
    # Star/cosmic summary
    print("\n" + "="*70)
    print("STELLAR OBJECTS SUMMARY")
    print("="*70)
    
    star_report = stars.get_report()
    star_detections = star_report.get('detections', [])
    stellar_objects = [d for d in star_detections if d['type'] == 'stellar_object']
    fusion_events = [d for d in star_detections if d['type'] == 'fusion_process']
    
    if stellar_objects:
        print(f"\nStars detected: {len(stellar_objects)}")
        types = {}
        for s in stellar_objects:
            stype = s['properties'].get('star_type', 'unknown')
            types[stype] = types.get(stype, 0) + 1
        print(f"  Star types: {types}")
    else:
        print("\nNo stellar objects detected above threshold.")
    
    if fusion_events:
        print(f"\nFusion events: {len(fusion_events)}")
        avg_efficiency = np.mean([f['properties']['efficiency'] for f in fusion_events])
        print(f"  Average fusion efficiency: {avg_efficiency:.3e}")
    
    hr_data = stars.get_hr_diagram_data()
    if hr_data.get('n_points', 0) > 0:
        print(f"\nH-R diagram data points: {hr_data['n_points']}")
    
    # Quantum phenomena summary
    print("\n" + "="*70)
    print("QUANTUM PHENOMENA SUMMARY")
    print("="*70)
    
    quantum_report = quantum.get_report()
    quantum_detections = quantum_report.get('detections', [])
    
    quantum_types = {}
    for d in quantum_detections:
        qtype = d['type']
        quantum_types[qtype] = quantum_types.get(qtype, 0) + 1
    
    if quantum_types:
        print(f"\nQuantum phenomena detected:")
        for qtype, count in quantum_types.items():
            print(f"  {qtype}: {count}")
    else:
        print("\nNo quantum phenomena detected above threshold.")
    
    # Galaxy/cosmic structure summary
    print("\n" + "="*70)
    print("GALACTIC STRUCTURES SUMMARY")
    print("="*70)
    
    galaxy_report = galaxies.get_report()
    galaxy_detections = galaxy_report.get('detections', [])
    
    galaxy_types = {}
    for d in galaxy_detections:
        gtype = d['type']
        galaxy_types[gtype] = galaxy_types.get(gtype, 0) + 1
    
    if galaxy_types:
        print(f"\nCosmic structures detected:")
        for gtype, count in galaxy_types.items():
            print(f"  {gtype}: {count}")
    else:
        print("\nNo galactic structures detected above threshold.")
    
    print("\n" + "="*70)
    print("ANALYSIS DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nThe modular analyzer system allows:")
    print("  • Independent analysis modules")
    print("  • Unit calibration for physical comparison")
    print("  • Easy addition of new analyzers")
    print("  • Pure observation (no interference)")
    print("  • Detection of: gravity, conservation, atoms, stars, quantum, galaxies")
    print()


if __name__ == "__main__":
    main()
