"""
Discover Physics - Extended simulation to discover emergent phenomena.

No assumptions - let the system reveal what physics emerges naturally.
Comprehensive tracking of structures, interactions, and laws.
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add parent to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine
from tools.law_quantifier import LawQuantifier
from tools.emergence_observer import EmergenceObserver

def discover_physics(steps: int = 5000, size: tuple = (128, 32), checkpoint_every: int = 500):
    """
    Run extended simulation and discover all emergent physics.
    
    Args:
        steps: Number of timesteps
        size: Grid size
        checkpoint_every: Save analysis every N steps
    """
    
    print("="*70)
    print("DISCOVERING EMERGENT PHYSICS")
    print("="*70)
    print(f"Configuration:")
    print(f"  Steps: {steps:,}")
    print(f"  Grid: {size[0]}×{size[1]} = {size[0]*size[1]:,} points")
    print(f"  Theory: B = ∇²(E-I) + λM∇²M - α||E-I||² - γ(E-I)")
    print(f"  Confluence: Ξ-balance through Möbius topology")
    print("="*70 + "\n")
    
    # Initialize
    engine = RealityEngine(size=size, device='cpu')
    engine.initialize('big_bang')
    
    quantifier = LawQuantifier()
    observer = EmergenceObserver()
    
    # Storage
    trajectory = []
    structure_history = []
    
    print("Running simulation...")
    print("-"*70)
    
    # Track structure lifetimes
    structure_lifetimes = {}
    
    for i in range(steps):
        # Step engine
        state = engine.step()
        
        # Observe structures every 10 steps
        if i % 10 == 0:
            field_state = engine.current_state
            structures = observer.observe(field_state)
            
            # Update lifetimes
            for s in structures:
                if s.id not in structure_lifetimes:
                    structure_lifetimes[s.id] = 0
                structure_lifetimes[s.id] += 1
                s.lifetime = structure_lifetimes[s.id]
            
            structure_history.append(structures)
            
            # Store trajectory data
            trajectory.append({
                'step': i,
                'A': field_state.actual.cpu().numpy() if hasattr(field_state.actual, 'cpu') else field_state.actual,
                'P': field_state.potential.cpu().numpy() if hasattr(field_state.potential, 'cpu') else field_state.potential,
                'M': field_state.memory.cpu().numpy() if hasattr(field_state.memory, 'cpu') else field_state.memory,
                'temperature': float(state.get('temperature', 0)),
                'entropy': float(state.get('entropy', 0)),
                'total_energy': float(state.get('disequilibrium', 0)),
                'structures': structures,
                'qpl_phase': float(state.get('qpl_phase', 0))
            })
        
        # Progress report
        if i % 100 == 0:
            n_structures = len(structures) if 'structures' in locals() else 0
            pac_quality = state.get('pac_quality', 0)
            gamma = state.get('gamma', 0)
            
            print(f"Step {i:5d}: {n_structures:3d} structures | "
                  f"PAC={pac_quality:.1%} | "
                  f"γ={gamma:.4f}")
        
        # Intermediate analysis
        if i > 0 and i % checkpoint_every == 0:
            print(f"\n{'='*70}")
            print(f"CHECKPOINT @ Step {i}")
            print(f"{'='*70}")
            
            # Discover structure classifications
            if len(structure_history) > 50:
                print("\nDiscovering structure types...")
                classifications = observer.discover_natural_classifications(structure_history[-100:])
                
                if classifications.get('n_classes', 0) > 0:
                    print(f"  Found {classifications['n_classes']} distinct types:\n")
                    
                    for class_id, info in sorted(classifications['discovered_types'].items(),
                                                 key=lambda x: x[1]['mean_mass']):
                        print(f"  [{class_id}] {info['name']} ({info['interaction']})")
                        print(f"      Mass={info['mean_mass']:.2f}, "
                              f"Coherence={info['mean_coherence']:.3f}, "
                              f"Lifetime={info['mean_lifetime']:.1f}")
                        print(f"      Count: {info['count']}")
            
            # Measure laws from recent trajectory
            if len(trajectory) > 100:
                print("\nMeasuring emergent laws...")
                recent_laws = quantifier.measure_conservation_laws(trajectory[-100:])
                
                for name, law in recent_laws.items():
                    if law.confidence > 0.7:
                        match_str = f" [→ {law.known_match}]" if law.known_match else ""
                        print(f"  • {law.name}: {law.equation}{match_str}")
                        print(f"    Confidence: {law.confidence:.1%}")
            
            print("="*70 + "\n")
    
    # Final comprehensive analysis
    print("\n" + "="*70)
    print("FINAL ANALYSIS")
    print("="*70 + "\n")
    
    results = analyze_final_results(trajectory, structure_history, quantifier, observer)
    
    # Save results
    output_dir = Path("output/physics_discovery")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"discovery_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return results

def analyze_final_results(trajectory, structure_history, quantifier, observer):
    """Comprehensive final analysis."""
    
    results = {
        'parameters': {
            'total_steps': len(trajectory) * 10,
            'grid_size': trajectory[0]['A'].shape if trajectory else None
        },
        'structures': {},
        'laws': {},
        'classifications': {},
        'summary': {}
    }
    
    # 1. Structure Analysis
    print("1. STRUCTURE CLASSIFICATION")
    print("-"*70)
    
    if len(structure_history) > 50:
        classifications = observer.discover_natural_classifications(structure_history)
        results['classifications'] = classifications
        
        if classifications.get('n_classes', 0) > 0:
            print(f"Discovered {classifications['n_classes']} distinct structure types:\n")
            
            for class_id, info in sorted(classifications['discovered_types'].items(),
                                        key=lambda x: x[1]['mean_mass']):
                print(f"  Type {class_id}: {info['name']}")
                print(f"    Properties: mass={info['mean_mass']:.2f}, "
                      f"coherence={info['mean_coherence']:.3f}")
                print(f"    Behavior: lifetime={info['mean_lifetime']:.1f}, "
                      f"neighbors={info['mean_neighbors']:.1f}")
                print(f"    Observed: {info['count']} instances")
                print()
    
    # 2. Conservation Laws
    print("\n2. CONSERVATION LAWS")
    print("-"*70)
    
    conservation_laws = quantifier.measure_conservation_laws(trajectory)
    results['laws']['conservation'] = {k: v.__dict__ for k, v in conservation_laws.items()}
    
    for name, law in conservation_laws.items():
        if law.confidence > 0.5:
            match_str = f" ✓ [{law.known_match}]" if law.known_match else ""
            print(f"  • {law.name}: {law.equation}{match_str}")
            print(f"    Confidence: {law.confidence:.1%}, Deviation: {law.deviation:.2%}")
    
    # 3. Force Laws
    print("\n3. FORCE LAWS")
    print("-"*70)
    
    force_laws = quantifier.measure_force_laws(trajectory)
    results['laws']['forces'] = {k: v.__dict__ if v else None for k, v in force_laws.items()}
    
    gravity_detected = False
    if force_laws:
        for name, law in force_laws.items():
            if law and law.confidence > 0.5:
                print(f"  • {law.equation}")
                print(f"    Parameters: {law.parameters}")
                print(f"    Confidence: {law.confidence:.1%}")
                
                if law.known_match == 'gravity':
                    gravity_detected = True
                    print(f"    ⭐ GRAVITY DETECTED!")
    
    if not gravity_detected:
        print("  No clear force laws detected")
        print("  (May need more interacting structures)")
    
    # 4. Thermodynamics
    print("\n4. THERMODYNAMIC LAWS")
    print("-"*70)
    
    thermo_laws = quantifier.measure_thermodynamic_laws(trajectory)
    results['laws']['thermodynamics'] = {k: v.__dict__ for k, v in thermo_laws.items()}
    
    for name, law in thermo_laws.items():
        if law.confidence > 0.5:
            match_str = f" ✓ [{law.known_match}]" if law.known_match else ""
            print(f"  • {law.name}: {law.equation}{match_str}")
            print(f"    Confidence: {law.confidence:.1%}")
    
    # 5. Summary Statistics
    print("\n5. SUMMARY")
    print("-"*70)
    
    total_structures = sum(len(s) for s in structure_history)
    unique_ids = len(set(s.id for structs in structure_history for s in structs))
    total_laws = sum(len(laws) for laws in results['laws'].values() if isinstance(laws, dict))
    
    results['summary'] = {
        'total_structures_observed': total_structures,
        'unique_structure_ids': unique_ids,
        'structure_types_discovered': classifications.get('n_classes', 0) if 'classifications' in locals() else 0,
        'laws_discovered': total_laws,
        'gravity_detected': gravity_detected,
        'pac_conserved': 'PAC' in conservation_laws or 'pac_functional' in conservation_laws
    }
    
    print(f"  Total structures observed: {total_structures:,}")
    print(f"  Unique structures: {unique_ids}")
    print(f"  Structure types: {results['summary']['structure_types_discovered']}")
    print(f"  Laws discovered: {total_laws}")
    print(f"  Gravity: {'✓ DETECTED' if gravity_detected else '✗ NOT FOUND'}")
    print(f"  PAC Conservation: {'✓ YES' if results['summary']['pac_conserved'] else '✗ NO'}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover emergent physics")
    parser.add_argument('--steps', type=int, default=5000, help='Number of steps')
    parser.add_argument('--width', type=int, default=128, help='Grid width')
    parser.add_argument('--height', type=int, default=32, help='Grid height')
    parser.add_argument('--checkpoint', type=int, default=500, help='Checkpoint interval')
    
    args = parser.parse_args()
    
    results = discover_physics(
        steps=args.steps,
        size=(args.width, args.height),
        checkpoint_every=args.checkpoint
    )
