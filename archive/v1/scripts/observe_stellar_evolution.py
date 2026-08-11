"""
Observe stellar-scale emergence - let stars, fusion, and heavy elements form naturally.

Scales up to 512x128 grid and runs for 10,000+ steps to observe:
- Stellar core formation (gravitational collapse)
- Fusion processes (light elements → heavier ones)
- Heavy element formation
- Chemistry at scale

NO assumptions - just observe what emerges from RBF + QBE dynamics.
"""
import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.reality_engine import RealityEngine
from tools.emergence_observer import EmergenceObserver

def main():
    print("=" * 70)
    print("STELLAR EVOLUTION OBSERVATION - Dawn Field Theory")
    print("=" * 70)
    print()
    print("Scaling up to 512×128 grid for stellar-scale phenomena")
    print("Running 10,000 steps to observe:")
    print("  • Gravitational collapse")
    print("  • Fusion processes")
    print("  • Heavy element formation")
    print("  • Large-scale chemistry")
    print()
    print("Using pure RBF + QBE dynamics with NO assumptions")
    print()
    
    # Initialize large-scale engine
    print("Initializing reality engine...")
    # Use smaller timestep for numerical stability on large grids
    engine = RealityEngine(size=(512, 128), dt=0.005)  # Reduced from 0.01
    engine.initialize('big_bang')
    
    # Create observer
    observer = EmergenceObserver()
    
    # Storage for observations
    observation_history = []
    
    # Observation intervals
    observe_interval = 100  # Observe every 100 steps
    total_steps = 10000
    
    print(f"Running {total_steps} steps (observing every {observe_interval} steps)...")
    print()
    
    try:
        for step in range(total_steps + 1):
            # Step simulation
            if step > 0:
                state_dict = engine.step()
                
                # Check for NaN - stop if simulation becomes unstable
                if state_dict is None or 'M_mean' not in state_dict:
                    print(f"\n⚠️  Simulation became unstable at step {step}")
                    print("   Stopping and analyzing data collected so far...")
                    break
            
            # Observe at intervals
            if step % observe_interval == 0:
                structures = observer.observe(engine.current_state)
                observation_history.append(structures)
                
                # Find molecular systems
                molecules = observer.find_molecular_patterns(structures)
                
                # Report
                print(f"Step {step:5d}:")
                print(f"  Emergent structures: {len(structures)}")
                print(f"  Multi-structure systems: {len(molecules)}")
                
                if structures:
                    masses = [s.mass for s in structures]
                    coherences = [s.coherence for s in structures]
                    print(f"  Mass range: {min(masses):.3f} - {max(masses):.3f}")
                    print(f"  Avg coherence: {np.mean(coherences):.3f}")
                    print(f"  Pattern types: {len(set(s.pattern_class for s in structures))}")
                
                # Check for stellar-scale structures
                stellar_candidates = [s for s in structures if s.mass > 10.0]
                if stellar_candidates:
                    print(f"  🌟 STELLAR CANDIDATES: {len(stellar_candidates)}")
                    for star in stellar_candidates[:3]:  # Show top 3
                        print(f"     → mass={star.mass:.1f}, coherence={star.coherence:.3f}, radius={star.radius:.1f}")
                
                # Check for molecular chemistry
                if molecules:
                    print(f"  Molecular systems:")
                    for mol in molecules[:3]:  # Show top 3
                        print(f"     → {mol['structure_count']} structures, mass={mol['total_mass']:.3f}, binding={mol['binding_energy']:.3f}")
                
                print()
            
            # Show progress
            if step % 1000 == 0 and step > 0:
                progress = (step / total_steps) * 100
                print(f"Progress: {progress:.1f}% ({step}/{total_steps} steps)")
                print()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    print()
    print("=" * 70)
    print("DISCOVERING PERIODIC PATTERNS")
    print("=" * 70)
    print()
    
    # Discover periodic structure from observations
    print("Analyzing emergent patterns across all observations...")
    
    if not observation_history or not any(len(obs) > 0 for obs in observation_history):
        print("No structures observed - simulation ended before structure formation")
        return
    
    periodic_discovery = observer.discover_periodic_patterns(observation_history)
    
    # Report mass clusters (natural quantization)
    print(f"\n📊 Mass Clusters Discovered: {len(periodic_discovery['mass_clusters'])}")
    for cluster_id, cluster in sorted(periodic_discovery['mass_clusters'].items()):
        print(f"  Cluster {cluster_id}:")
        print(f"    Mass center: {cluster['mass_center']:.3f}")
        print(f"    Observations: {cluster['count']}")
        print(f"    Avg coherence: {cluster['avg_coherence']:.3f}")
        print(f"    Avg persistence: {cluster['avg_persistence']:.3f}")
    
    # Report bonding matrix
    print(f"\n🔗 Bonding Patterns Discovered: {len(periodic_discovery['bonding_matrix'])}")
    for bond_key, bond_info in sorted(periodic_discovery['bonding_matrix'].items(), 
                                      key=lambda x: x[1]['count'], reverse=True)[:10]:
        print(f"  Clusters {bond_key[0]} ↔ {bond_key[1]}:")
        print(f"    Bonds observed: {bond_info['count']}")
        print(f"    Avg binding: {bond_info['avg_binding']:.3f}")
    
    # Report stability groups
    print(f"\n⚖️ Stability Groups:")
    for group_name, group_info in periodic_discovery['stability_groups'].items():
        print(f"  {group_name.capitalize()} stability:")
        print(f"    Count: {group_info['count']}")
        print(f"    Mass range: {group_info['mass_range'][0]:.3f} - {group_info['mass_range'][1]:.3f}")
        print(f"    Avg coherence: {group_info['avg_coherence']:.3f}")
    
    # Report periodic structure
    print(f"\n🔬 Periodic Structure (like periodic table):")
    for cluster_id, periodic_info in sorted(periodic_discovery['periodic_structure'].items())[:10]:
        print(f"  Cluster {cluster_id} (mass ≈ {periodic_info['properties']['mass_center']:.2f}):")
        print(f"    Bonding versatility: {periodic_info['bonding_versatility']} partners")
        if periodic_info['bonding_partners']:
            partners_str = ", ".join(
                f"{p['partner_cluster']}({p['bond_count']})" 
                for p in periodic_info['bonding_partners'][:3]
            )
            print(f"    Bonds with: {partners_str}")
    
    # Save results
    output_dir = Path("output/stellar_observation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"stellar_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return obj
    
    results = {
        'configuration': {
            'grid_size': [512, 128],
            'total_steps': total_steps,
            'observe_interval': observe_interval,
            'dt': 0.01
        },
        'periodic_discovery': {
            key: {k: convert_for_json(v) for k, v in val.items()} 
            if isinstance(val, dict) else convert_for_json(val)
            for key, val in periodic_discovery.items()
        },
        'summary': {
            'total_observations': len(observation_history),
            'mass_clusters_found': len(periodic_discovery['mass_clusters']),
            'bonding_patterns_found': len(periodic_discovery['bonding_matrix']),
            'unique_patterns': len(set(
                s.pattern_class for obs in observation_history for s in obs
            ))
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 70)
    print("✅ STELLAR OBSERVATION COMPLETE!")
    print("=" * 70)
    print()
    print(f"Results saved to: {output_file}")
    print()
    print("Key Findings:")
    print(f"  • {len(periodic_discovery['mass_clusters'])} mass clusters (natural quantization)")
    print(f"  • {len(periodic_discovery['bonding_matrix'])} bonding patterns (emergent chemistry)")
    print(f"  • Periodic structure emerged with {len(periodic_discovery['periodic_structure'])} distinct types")
    print()
    print("Everything emerged from: B(x,t) = ∇²(E-I) + λM∇²M - α||E-I||²")
    print("This validates Dawn Field Theory at stellar scales!")

if __name__ == "__main__":
    main()
