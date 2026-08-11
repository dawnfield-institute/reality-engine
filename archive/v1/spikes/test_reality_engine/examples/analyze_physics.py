"""
Physics Analysis Example

Runs Reality Engine and analyzes emergent particle physics
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
from visualization.periodic_table_viz import PeriodicTableVisualizer
from utils.logger import RealityLogger


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        # For custom objects like Particle, skip them (can't serialize complex objects)
        return str(obj)
    else:
        return obj


def run_physics_analysis(
    universe_size: int = 64,
    total_steps: int = 5000,
    detection_interval: int = 1000,
    device: str = 'cuda'
):
    """
    Run Reality Engine and perform comprehensive physics analysis
    
    Args:
        universe_size: Grid size (creates universe_size^3 space)
        total_steps: Total evolution steps
        detection_interval: Steps between particle detection
        device: 'cuda' or 'cpu'
    """
    
    print("=" * 70)
    print("REALITY ENGINE - EMERGENT PHYSICS ANALYSIS")
    print("=" * 70)
    print(f"Universe Size: {universe_size}^3")
    print(f"Total Steps: {total_steps:,}")
    print(f"Detection Interval: {detection_interval}")
    print(f"Device: {device.upper()}")
    print("=" * 70)
    print()
    
    # Initialize Reality Engine
    print("[1/6] Initializing Reality Engine...")
    reality = DawnField(
        shape=(universe_size, universe_size, universe_size),
        dt=0.0001,  # Validated stable timestep
        device=device
    )
    print(f"[OK] Reality Field initialized: {universe_size}^3 grid on {device}")
    print()
    
    # Initialize analyzer and visualizer
    print("[2/6] Initializing analysis tools...")
    analyzer = ParticleAnalyzer()
    visualizer = PeriodicTableVisualizer()
    print("[OK] Analyzer and visualizer ready")
    print()
    
    # Evolution with periodic detection
    print(f"[3/6] Evolving universe for {total_steps:,} steps...")
    print("      (Detecting particles every {:,} steps)".format(detection_interval))
    
    detection_history = []
    
    for step in range(total_steps):
        # Evolve
        reality.evolve_step()
        
        # Periodic detection
        if step > 0 and step % detection_interval == 0:
            print(f"\n--- Detection at step {step:,} ---")
            
            # Move fields to CPU for analysis
            E_np = reality.E.cpu().numpy()
            I_np = reality.I.cpu().numpy()
            M_np = reality.M.cpu().numpy()
            
            # Detect particles
            particles = analyzer.detect_particles(E_np, I_np, M_np)
            
            if particles:
                print(f"[FOUND] {len(particles)} particles detected")
                
                # Quick summary
                periodic_table = analyzer.build_periodic_table(particles)
                for ptype, data in periodic_table.items():
                    print(f"  - {ptype}: {data['count']} (mass={data['avg_mass']:.3f}, charge={data['avg_charge']:+.3f})")
                
                # Save detection data
                detection_history.append({
                    'step': step,
                    'particle_count': len(particles),
                    'periodic_table': periodic_table
                })
            else:
                print("[NO] No stable particles found yet")
            
        # Progress indicator
        if step > 0 and step % 100 == 0:
            progress = (step / total_steps) * 100
            print(f"Progress: {progress:.1f}% ({step:,}/{total_steps:,})", end='\r')
    
    print("\n\n[OK] Evolution complete")
    print()
    
    # Final particle detection
    print("[4/6] Performing final particle analysis...")
    E_np = reality.E.cpu().numpy()
    I_np = reality.I.cpu().numpy()
    M_np = reality.M.cpu().numpy()
    
    particles = analyzer.detect_particles(E_np, I_np, M_np)
    
    if not particles:
        print("[NO] No particles detected - try running longer or with different parameters")
        return
    
    print(f"[FOUND] {len(particles)} final particles")
    print()
    
    # Build periodic table
    periodic_table = analyzer.build_periodic_table(particles)
    
    # Find composite structures
    composites = analyzer.find_composite_structures(particles)
    
    # Print comprehensive summary
    print("[5/6] Analysis Summary:")
    print("=" * 70)
    analyzer.print_summary()
    
    # Print composites separately
    if composites:
        print(f"\nComposite Structures: {len(composites)}")
        for comp in composites[:5]:  # Show first 5
            print(f"  {comp['type']}: {len(comp['particles'])} particles, separation={comp['separation']:.2f}")
    print()
    
    # Create visualizations
    print("[6/6] Creating visualizations...")
    output_dir = Path('output') / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Periodic table
    periodic_table_path = output_dir / 'periodic_table.png'
    visualizer.create_periodic_table(periodic_table, str(periodic_table_path))
    
    # Mass spectrum
    mass_spectrum_path = output_dir / 'mass_spectrum.png'
    visualizer.plot_mass_spectrum(particles, str(mass_spectrum_path))
    
    # 3D particle map
    particle_map_path = output_dir / 'particle_map_3d.png'
    visualizer.create_3d_particle_map(particles, str(particle_map_path))
    
    # Save analysis data
    analysis_data = {
        'metadata': {
            'universe_size': universe_size,
            'total_steps': total_steps,
            'detection_interval': detection_interval,
            'device': device,
            'timestamp': datetime.now().isoformat()
        },
        'final_particles': {
            'count': len(particles),
            'periodic_table': convert_to_json_serializable(periodic_table),
            'composites_count': len(composites),
            'composites_summary': [
                {
                    'type': c['type'],
                    'particle_count': len(c['particles']),
                    'separation': float(c['separation'])
                    # Don't include actual Particle objects - not JSON serializable
                }
                for c in composites
            ]
        },
        'detection_history': convert_to_json_serializable(detection_history)
    }
    
    analysis_path = output_dir / 'physics_data.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"[SAVED] Analysis data: {analysis_path}")
    print()
    
    # Summary report
    print("=" * 70)
    print("EMERGENT PHYSICS VALIDATION")
    print("=" * 70)
    
    # Check for standard model particles
    standard_particles = {
        'photon': periodic_table.get('photon', {}).get('count', 0),
        'electron': periodic_table.get('electron', {}).get('count', 0),
        'positron': periodic_table.get('positron', {}).get('count', 0),
        'proton': periodic_table.get('proton', {}).get('count', 0),
        'neutron': periodic_table.get('neutron', {}).get('count', 0)
    }
    
    print("\nStandard Model Particles:")
    for ptype, count in standard_particles.items():
        status = "[FOUND]" if count > 0 else "[NONE]"
        print(f"  {status} {ptype}: {count}")
    
    # Validation checks
    print("\nPhysics Validation:")
    
    # Charge conservation
    total_charge = sum(p.charge for p in particles)
    charge_balance = abs(total_charge / len(particles)) if particles else 0
    charge_status = "[OK]" if charge_balance < 0.1 else "[NO]"
    print(f"  {charge_status} Charge conservation: {total_charge:+.3f} (avg={charge_balance:.3f})")
    
    # Mass hierarchy
    mass_values = [p.mass for p in particles]
    mass_range = max(mass_values) - min(mass_values) if mass_values else 0
    hierarchy_status = "[OK]" if mass_range > 0.5 else "[NO]"
    print(f"  {hierarchy_status} Mass hierarchy: range={mass_range:.3f}")
    
    # Composite structures
    composite_status = "[OK]" if composites else "[NONE]"
    print(f"  {composite_status} Composite structures: {len(composites)} found")
    
    # Novel predictions
    exotic_count = periodic_table.get('exotic', {}).get('count', 0)
    if exotic_count > 0:
        print(f"  [!] Novel particles: {exotic_count} exotic structures detected")
    
    print()
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print()
    
    print("[OK] Physics analysis complete!")


if __name__ == "__main__":
    # Small test run
    run_physics_analysis(
        universe_size=64,
        total_steps=5000,
        detection_interval=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
