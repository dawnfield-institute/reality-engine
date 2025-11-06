"""
Long-Term Evolution to Observe Atom Formation

Runs Reality Engine for extended period to allow:
1. Particles to form (early phase)
2. Particles to interact and bind (middle phase)
3. Atoms/molecules to emerge (late phase)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from datetime import datetime
import json

from core.dawn_field import DawnField
from emergence.particle_analyzer import ParticleAnalyzer
from emergence.stellar_analyzer import StellarAnalyzer
from visualization.periodic_table_viz import PeriodicTableVisualizer
from visualization.stellar_viz import visualize_structures


def watch_structure_formation(
    universe_size: int = 64,
    total_steps: int = 20000,  # Much longer!
    detection_interval: int = 2000,
    device: str = 'cuda'
):
    """
    Run long evolution to watch atoms emerge naturally
    """
    
    print("=" * 70)
    print("REALITY ENGINE - ATOM FORMATION OBSERVER")
    print("=" * 70)
    print(f"Universe Size: {universe_size}^3")
    print(f"Total Steps: {total_steps:,} (extended run)")
    print(f"Detection Interval: {detection_interval}")
    print(f"Device: {device.upper()}")
    print()
    print("Goal: Let atoms emerge naturally from particle interactions")
    print("=" * 70)
    print()
    
    # Initialize
    print("[1/4] Initializing Reality Engine...")
    reality = DawnField(
        shape=(universe_size, universe_size, universe_size),
        dt=0.0001,
        device=device
    )
    print(f"[OK] Reality Field initialized")
    print()
    
    analyzer = ParticleAnalyzer()
    stellar_analyzer = StellarAnalyzer(mass_threshold=500.0)
    
    # Evolution with detailed tracking
    print(f"[2/4] Evolving for {total_steps:,} steps...")
    print("      Watching for:")
    print("      - Particle formation (steps 0-5000)")
    print("      - Particle clustering (steps 5000-10000)")
    print("      - Atom emergence (steps 10000+)")
    print("      - Stellar structures (wells, suns, black holes)")
    print("      - Fusion events (stellar nucleosynthesis)")
    print()
    
    evolution_history = []
    
    for step in range(total_steps):
        reality.evolve_step()
        
        # Periodic analysis
        if step > 0 and step % detection_interval == 0:
            print(f"\n{'='*60}")
            print(f"Analysis at step {step:,}")
            print(f"{'='*60}")
            
            # Get fields
            E_np = reality.E.cpu().numpy()
            I_np = reality.I.cpu().numpy()
            M_np = reality.M.cpu().numpy()
            
            # Detect particles
            particles = analyzer.detect_particles(E_np, I_np, M_np, 
                                                 threshold=0.1, 
                                                 stability_threshold=0.01)
            
            if len(particles) > 0:
                # Build periodic table
                periodic_table = analyzer.build_periodic_table(particles)
                
                print(f"\nParticle Census:")
                for ptype, data in sorted(periodic_table.items(), 
                                         key=lambda x: x[1]['count'], reverse=True):
                    print(f"  {ptype:12s}: {data['count']:3d} "
                          f"(mass={data['avg_mass']:6.1f}, "
                          f"charge={data['avg_charge']:+6.3f})")
                
                # Look for composite structures (atoms!)
                composites = analyzer.find_composite_structures(particles)
                
                # Look for stellar structures (gravitational wells, stars)
                structures = stellar_analyzer.detect_structures(E_np, I_np, M_np, particles)
                
                # Check for fusion
                fusion_events = stellar_analyzer.detect_fusion_events(E_np, M_np, step)
                
                if composites:
                    print(f"\n*** COMPOSITE STRUCTURES DETECTED: {len(composites)} ***")
                    for i, comp in enumerate(composites[:10]):  # Show first 10
                        p1, p2 = comp['particles']
                        print(f"  [{i+1}] {comp['type']} bond:")
                        print(f"      {p1.classification} + {p2.classification}")
                        print(f"      Mass: {comp['total_mass']:.2f}, "
                              f"Charge: {comp['net_charge']:+.3f}, "
                              f"Separation: {comp['separation']:.2f}")
                else:
                    print(f"\nNo composite structures yet (particles too dispersed)")
                
                if structures:
                    print(f"\n*** STELLAR STRUCTURES DETECTED: {len(structures)} ***")
                    for i, s in enumerate(structures[:5]):  # Show top 5
                        print(f"  [{i+1}] {s.structure_type.upper()}")
                        print(f"      Mass: {s.total_mass:.1f}, Radius: {s.radius:.1f}")
                        print(f"      Core density: {s.core_density:.1f}, Temp: {s.temperature:.2f}")
                        if s.is_fusion_active():
                            print(f"      >>> FUSION ACTIVE - Creating new atoms! <<<")
                
                if fusion_events:
                    print(f"\n!!! {len(fusion_events)} FUSION EVENTS THIS INTERVAL !!!")
                
                # Track evolution
                evolution_history.append({
                    'step': step,
                    'particle_count': len(particles),
                    'particle_types': len(periodic_table),
                    'composite_count': len(composites),
                    'stellar_count': len(structures),
                    'fusion_count': len(fusion_events),
                    'avg_mass': np.mean([p.mass for p in particles]),
                    'avg_separation': np.mean([
                        np.linalg.norm(np.array(p1.position) - np.array(p2.position))
                        for i, p1 in enumerate(particles)
                        for p2 in particles[i+1:i+5]  # Check nearby
                    ]) if len(particles) > 1 else 0
                })
            else:
                print("No particles detected yet")
            
            print(f"{'='*60}\n")
        
        # Progress
        if step % 500 == 0:
            print(f"Progress: {100*step/total_steps:.1f}% ({step:,}/{total_steps:,})", 
                  end='\r')
    
    print("\n\n[OK] Evolution complete!")
    print()
    
    # Final analysis
    print("[3/4] Final Analysis...")
    E_np = reality.E.cpu().numpy()
    I_np = reality.I.cpu().numpy()
    M_np = reality.M.cpu().numpy()
    
    particles = analyzer.detect_particles(E_np, I_np, M_np)
    structures = stellar_analyzer.detect_structures(E_np, I_np, M_np, particles)
    
    if particles:
        periodic_table = analyzer.build_periodic_table(particles)
        composites = analyzer.find_composite_structures(particles)
        
        print()
        print("=" * 70)
        print("FINAL STATE REPORT")
        print("=" * 70)
        analyzer.print_summary()
        
        # Stellar structures
        stellar_analyzer.print_summary()
        
        if composites:
            print(f"\n{'='*70}")
            print(f"ATOMS/MOLECULES EMERGED: {len(composites)}")
            print(f"{'='*70}")
            
            # Classify composite types
            composite_types = {}
            for comp in composites:
                key = f"{comp['particles'][0].classification}+{comp['particles'][1].classification}"
                composite_types[key] = composite_types.get(key, 0) + 1
            
            print("\nComposite Structure Census:")
            for ctype, count in sorted(composite_types.items(), 
                                      key=lambda x: x[1], reverse=True):
                print(f"  {ctype}: {count}")
        
        # Save results
        output_dir = Path('output') / datetime.now().strftime('%Y%m%d_%H%M%S_longrun')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save evolution timeline
        import json
        
        # Convert numpy types
        def convert_to_json(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        timeline_clean = [
            {k: convert_to_json(v) for k, v in item.items()}
            for item in evolution_history
        ]
        
        timeline_path = output_dir / 'evolution_timeline.json'
        with open(timeline_path, 'w') as f:
            json.dump({
                'metadata': {
                    'total_steps': total_steps,
                    'detection_interval': detection_interval,
                    'universe_size': universe_size
                },
                'timeline': timeline_clean
            }, f, indent=2)
        print(f"\n[SAVED] Evolution timeline: {timeline_path}")
        
        # Save detailed composite structures for atom classification
        if composites:
            composite_data = []
            for comp in composites:
                composite_data.append({
                    'total_mass': convert_to_json(comp['total_mass']),
                    'net_charge': convert_to_json(comp['net_charge']),
                    'separation': convert_to_json(comp['separation']),
                    'bond_type': comp['type'],
                    'binding_strength': convert_to_json(comp['binding_strength']),
                    'particle_types': [p.classification for p in comp['particles']],
                    'particle_masses': [convert_to_json(p.mass) for p in comp['particles']],
                    'particle_charges': [convert_to_json(p.charge) for p in comp['particles']]
                })
            
            composites_path = output_dir / 'composite_structures.json'
            with open(composites_path, 'w') as f:
                json.dump({
                    'count': len(composite_data),
                    'structures': composite_data
                }, f, indent=2)
            print(f"[SAVED] Composite structures: {composites_path}")
        
        # Visualizations
        print("\n[4/4] Creating visualizations...")
        visualizer = PeriodicTableVisualizer()
        
        visualizer.create_periodic_table(periodic_table, 
                                        str(output_dir / 'periodic_table.png'))
        visualizer.plot_mass_spectrum(particles, 
                                     str(output_dir / 'mass_spectrum.png'))
        visualizer.create_3d_particle_map(particles, 
                                         str(output_dir / 'particle_map_3d.png'))
        
        # Stellar structures visualization
        if structures:
            visualize_structures(structures, universe_size, 
                               str(output_dir / 'stellar_structures.png'))
        
        print(f"\n{'='*70}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*70}")
    else:
        print("No particles formed")
    
    print("\n[OK] Analysis complete!")


if __name__ == "__main__":
    watch_structure_formation(
        universe_size=64,
        total_steps=20000,  # 20k steps to see atom formation
        detection_interval=2000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
