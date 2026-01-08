"""
Experiment 01: Hierarchical Field Validation

Tests the HierarchicalMobiusField substrate for:
1. PAC conservation convergence
2. Structure formation from seeds
3. Cache tiering behavior
4. Möbius boundary enforcement

Author: Dawn Field Institute
Date: 2026-01-08
POC: POC-001 PAC-Lazy Field
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from substrate.hierarchical_field import HierarchicalMobiusField


def run_experiment(
    resolution: tuple = (16, 16, 16),
    steps: int = 500,
    dt: float = 0.01,
    save_results: bool = True
):
    """
    Validate PAC-Lazy hierarchical field substrate.
    """
    print("=" * 70)
    print("EXP_01: HIERARCHICAL FIELD VALIDATION")
    print("=" * 70)
    print()
    
    print(f"1. Initializing {resolution} field...")
    field = HierarchicalMobiusField(
        base_resolution=resolution,
        max_levels=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    field.initialize()
    
    print(f"2. Running {steps} evolution steps...")
    pac_history = []
    structure_history = []
    
    for step in range(steps):
        metrics = field.step(dt=dt)
        pac_history.append(metrics['pac_conservation'])
        structure_history.append(metrics['structures'])
        
        if step % (steps // 5) == 0:
            print(f"   Step {step:4d}: PAC_cons={metrics['pac_conservation']:.2e}, "
                  f"structures={metrics['structures']}")
    
    print()
    summary = field.get_summary()
    
    # Check convergence
    final_pac = pac_history[-1]
    converged = abs(final_pac) < 1e-10
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'resolution': resolution,
            'steps': steps,
            'dt': dt
        },
        'metrics': {
            'pac_converged': converged,
            'final_pac_conservation': final_pac,
            'min_pac_conservation': min(pac_history),
            'convergence_step': next((i for i, p in enumerate(pac_history) if abs(p) < 1e-12), -1),
            'final_structures': structure_history[-1],
            'total_cells': summary['total_cells']
        },
        'cache_stats': summary['cache']
    }
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  PAC Converged: {'YES' if converged else 'NO'}")
    print(f"  Final PAC Conservation: {final_pac:.2e}")
    print(f"  Convergence Step: {results['metrics']['convergence_step']}")
    print(f"  Final Structures: {results['metrics']['final_structures']}")
    print()
    
    if save_results:
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = results_dir / f'exp_01_hierarchical_validation_{timestamp}.json'
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved: {filepath}")
    
    return results


if __name__ == "__main__":
    run_experiment()
