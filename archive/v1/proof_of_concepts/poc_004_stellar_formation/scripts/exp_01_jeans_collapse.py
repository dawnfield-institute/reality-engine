"""
Experiment 01: Jeans Collapse Test

Tests gravitational collapse via Jeans criterion in PAC-Lazy hierarchical field.

Jeans Wavelength: λ_J = c_s × sqrt(π / (G × ρ))
Collapse occurs when λ > λ_J (perturbation scale exceeds Jeans scale)

Author: Dawn Field Institute
Date: 2026-01-08
POC: POC-004 Stellar Formation
"""

import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from substrate.hierarchical_field import HierarchicalMobiusField, FieldCell
from substrate.constants import PHI, PHI_INV, XI, DELTA_FEIGENBAUM
from dynamics.feigenbaum_detector import FeigenbaumDetector, detect_period


def jeans_criterion(cell: FieldCell, G: float = 1.0, c_s: float = 0.5, min_mass: float = 1.0) -> bool:
    """
    Check if a cell satisfies the Jeans criterion for gravitational collapse.
    
    Jeans length: λ_J = c_s × sqrt(π / (G × ρ))
    Collapse occurs when region size > Jeans length
    
    In our units:
    - ρ (density) ~ memory_delta (accumulated mass)
    - c_s (sound speed) ~ sqrt(temperature)  
    - G ~ coupling strength
    
    Returns True if cell should collapse.
    """
    # Need significant mass accumulation first
    if cell.memory_delta < min_mass:
        return False
    
    # Already deeply herniated - stable
    if cell.herniation_depth >= 5:
        return False
    
    rho = abs(cell.memory_delta)
    c_s_local = c_s * np.sqrt(max(cell.temperature, 0.01))
    
    # Jeans length in cell units  
    lambda_J = c_s_local * np.sqrt(np.pi / (G * rho + 1e-10))
    
    # Cell size (normalized to 1)
    cell_size = 1.0
    
    # Collapse if cell size > Jeans length (very dense region)
    return cell_size > lambda_J


def run_experiment(
    resolution: tuple = (32, 32, 32),
    steps: int = 1000,
    dt: float = 0.01,
    n_seeds: int = 5,
    seed_amplitude: float = 5.0,
    G: float = 2.0,
    save_results: bool = True,
    verbose: bool = True
):
    """
    Run Jeans collapse experiment.
    
    Starts from premass phase and evolves until:
    1. Structures form (memory accumulation)
    2. Gravitational collapse occurs (Jeans instability)
    3. Proto-stellar objects emerge
    """
    print("=" * 70)
    print("EXP_01: JEANS COLLAPSE TEST")
    print("=" * 70)
    print()
    
    # Initialize hierarchical field
    print(f"1. Initializing field: {resolution}")
    field = HierarchicalMobiusField(
        base_resolution=resolution,
        max_levels=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    field.initialize(mode='premass')
    
    # Add initial density perturbations
    print(f"2. Adding {n_seeds} density seeds (amplitude={seed_amplitude})")
    nx, ny, nz = resolution
    
    for _ in range(n_seeds):
        sx = np.random.randint(2, nx-2)
        sy = np.random.randint(2, ny-2)
        sz = np.random.randint(2, nz-2)
        
        sigma = 4.0
        
        for pos, cell in field.root_cells.items():
            x, y, z = pos
            dist_sq = (x - sx)**2 + (y - sy)**2 + (z - sz)**2
            perturbation = seed_amplitude * np.exp(-dist_sq / (2 * sigma**2))
            
            cell.potential_delta += perturbation
            cell.memory_delta += perturbation * 0.5
    
    field._update_pac_total()
    
    # Initialize trackers
    bifurcation_detector = FeigenbaumDetector()
    energy_history = []
    structure_history = []
    herniation_history = []
    collapse_events = []
    collapsed_cells = {}  # Track cells: cell_id -> last herniation depth
    
    print(f"3. Running simulation for {steps} steps")
    print()
    
    start_time = time.time()
    
    for step in range(steps):
        metrics = field.step(dt=dt)
        
        energy_history.append(metrics['total_energy'])
        structure_history.append(metrics['structures'])
        herniation_history.append(metrics['herniations'])
        
        # Check for gravitational collapse
        # Allow continued herniation if mass keeps growing
        for cell in list(field.cache._hot.values()):
            last_depth = collapsed_cells.get(cell.cell_id, 0)
            
            # Can herniate again if mass criterion met and not maxed out
            if jeans_criterion(cell, G=G, min_mass=1.0):
                old_depth = cell.herniation_depth
                field.herniate(cell, strength=0.5)
                
                if cell.herniation_depth > old_depth:
                    collapsed_cells[cell.cell_id] = cell.herniation_depth
                    collapse_events.append({
                        'step': step,
                        'position': cell.position,
                        'mass': cell.memory_delta,
                        'herniation_depth': cell.herniation_depth,
                        'temperature': cell.temperature
                    })
                    cell.has_structure = True
                    field.active_structures.add(cell.cell_id)
        
        # Check for bifurcations
        if len(energy_history) > 50:
            period = detect_period(np.array(energy_history[-100:]))
            if period > 0 and step % 100 == 0:
                bifurcation_detector._period_history.append((step, period))
        
        # Periodic output
        if verbose and step % (steps // 10) == 0:
            print(f"  Step {step:5d}/{steps}: "
                  f"structures={metrics['structures']:3d}, "
                  f"herniations={metrics['herniations']:5d}, "
                  f"PAC={metrics['pac_conservation']:.2e}")
    
    elapsed = time.time() - start_time
    print()
    print(f"4. Complete in {elapsed:.1f}s ({steps/elapsed:.1f} steps/sec)")
    print()
    
    # Analysis - check actual field state
    summary = field.get_summary()
    
    # Get herniation depth distribution from actual cells
    depths = [c.herniation_depth for c in field.root_cells.values()]
    depth_dist = {}
    for d in range(-1, 6):
        count = sum(1 for x in depths if x == d)
        if count > 0:
            depth_dist[d] = count
    
    # Count proto-stars (cells with max herniation and high mass)
    actual_proto_stars = sum(1 for c in field.root_cells.values() 
                            if c.herniation_depth >= 3 and c.memory_delta > 1.0)
    
    # Check for stellar formation (deep herniations with significant mass)
    stellar_formation = actual_proto_stars > 0
    
    # Mass statistics
    masses = [c.memory_delta for c in field.root_cells.values()]
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'resolution': resolution,
            'steps': steps,
            'dt': dt,
            'n_seeds': n_seeds,
            'seed_amplitude': seed_amplitude,
            'G': G
        },
        'metrics': {
            'elapsed_seconds': elapsed,
            'steps_per_second': steps / elapsed,
            'total_cells': summary['total_cells'],
            'final_structures': structure_history[-1],
            'peak_structures': max(structure_history),
            'total_herniations': herniation_history[-1] if herniation_history else 0,
            'proto_stars': actual_proto_stars,
            'max_herniation_depth': max(depths),
            'stellar_formation': stellar_formation,
            'final_pac_conservation': summary['total_pac'],
            'mass_stats': {
                'min': float(min(masses)),
                'max': float(max(masses)),
                'mean': float(np.mean(masses)),
                'high_mass_cells': sum(1 for m in masses if m > 2.0)
            }
        },
        'herniation_distribution': depth_dist,
        'cache_stats': summary['cache'],
        'jeans_collapse_events': len(collapse_events)
    }
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Final structures: {results['metrics']['final_structures']}")
    print(f"  Herniations (depth>=1): {results['metrics']['total_herniations']}")
    print(f"  Proto-stellar objects (depth>=3, M>1): {actual_proto_stars}")
    print(f"  Max herniation depth: {results['metrics']['max_herniation_depth']}")
    print()
    print(f"  Herniation distribution:")
    for d, count in sorted(depth_dist.items()):
        label = 'premass' if d == -1 else f'depth {d}'
        star = ' [STAR]' if d >= 5 else ''
        print(f"    {label}: {count} cells{star}")
    print()
    print(f"  Mass statistics:")
    print(f"    Max mass: {results['metrics']['mass_stats']['max']:.2f}")
    print(f"    High mass cells (>2.0): {results['metrics']['mass_stats']['high_mass_cells']}")
    print()
    print(f"  Stellar formation: {'YES' if stellar_formation else 'NO'}")
    print()
    
    if save_results:
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = results_dir / f'exp_01_jeans_collapse_{timestamp}.json'
        
        # Convert tuples for JSON
        results_json = json.loads(json.dumps(results, default=str))
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"Results saved: {filepath}")
    
    return results


def run_scale_comparison():
    """Compare different scales to find where stellar formation occurs"""
    print()
    print("=" * 70)
    print("SCALE COMPARISON")
    print("=" * 70)
    
    scales = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
    results = []
    
    for resolution in scales:
        print(f"\n--- Testing {resolution} ---")
        result = run_experiment(
            resolution=resolution,
            steps=500,
            verbose=False,
            save_results=False
        )
        results.append({
            'resolution': resolution,
            'cells': np.prod(resolution),
            **result['metrics']
        })
    
    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Resolution':<15} {'Cells':>10} {'Proto-Stars':>12} {'Max Depth':>10} {'Stellar':>10}")
    print("-" * 60)
    for r in results:
        print(f"{str(r['resolution']):<15} {r['cells']:>10,} {r['proto_stars']:>12} "
              f"{r['max_herniation_depth']:>10} {'YES' if r['stellar_formation'] else 'NO':>10}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Jeans Collapse Experiment")
    parser.add_argument('--scale', type=int, default=32)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--compare', action='store_true')
    
    args = parser.parse_args()
    
    if args.compare:
        run_scale_comparison()
    else:
        run_experiment(
            resolution=(args.scale, args.scale, args.scale),
            steps=args.steps
        )
