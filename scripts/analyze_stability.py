"""
Stability Analysis for Atomic Structures

Analyzes why atoms disappear after ~50-100 steps and identifies
patterns that could lead to improved stability.
"""

import sys
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine
from tools.atomic_analyzer import AtomicAnalyzer
from tools.viz_utils import plot_field_statistics, quick_snapshot


def run_stability_analysis(
    steps: int = 500,
    size: Tuple[int, int] = (128, 32),
    save_dir: str = None
) -> Dict:
    """
    Run comprehensive stability analysis on atomic structures.
    
    Args:
        steps: Number of simulation steps
        size: Field dimensions
        save_dir: Directory to save results
        
    Returns:
        Analysis results dictionary
    """
    print(f"Starting stability analysis: {steps} steps on {size[0]}x{size[1]} field")
    
    # Initialize engine
    engine = RealityEngine(
        size=size,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Initialize with big_bang mode for hot dense start (more likely to form atoms)
    engine.initialize('big_bang')
    
    # Run simulation and collect history
    print("Running simulation...")
    history = []
    atom_counts = []
    
    # Use lower threshold to detect emergent structures from RBF dynamics
    analyzer = AtomicAnalyzer(min_stability=0.2)  # Lower for emergent structures
    
    for step_data in engine.evolve(steps):
        step = step_data['step']
        
        # Store full state periodically for analysis
        if step % 10 == 0:
            # Access fields from engine's current_state
            state = engine.current_state
            history.append({
                'A': state.actual.cpu().numpy(),
                'P': state.potential.cpu().numpy(),
                'M': state.memory.cpu().numpy(),
                'T': state.temperature.cpu().numpy(),
                'step': step
            })
            
            # Detect atoms (create state wrapper)
            atoms = analyzer.detect_atoms(state)
            atom_counts.append(len(atoms))
            
            if step % 50 == 0:
                M_stats = f"M: min={history[-1]['M'].min():.3f}, max={history[-1]['M'].max():.3f}, mean={history[-1]['M'].mean():.3f}"
                print(f"Step {step}: {len(atoms)} atoms detected | {M_stats}")
    
    print("\nAnalyzing atom lifecycles...")
    
    # Analyze atom lifecycles (with error handling for large datasets)
    try:
        lifecycle_data = analyzer.track_atom_lifecycle(history)
    except Exception as e:
        print(f"Warning: Lifecycle analysis failed: {e}")
        lifecycle_data = {
            'formation_events': [],
            'dissolution_events': [],
            'average_lifetime': 0,
            'max_lifetime': 0,
            'instability_causes': {
                'thermal_fluctuation': 0,
                'memory_decay': 0,
                'field_divergence': 0,
                'neighbor_collision': 0
            },
            'stability_correlations': {}
        }
    
    # Compile analysis results
    results = {
        'simulation': {
            'steps': steps,
            'size': size,
            'atom_counts': atom_counts,
            'max_atoms': max(atom_counts) if atom_counts else 0,
            'final_atoms': atom_counts[-1] if atom_counts else 0
        },
        'lifecycle': lifecycle_data,
        'stability_metrics': {
            'average_lifetime': lifecycle_data.get('average_lifetime', 0),
            'max_lifetime': lifecycle_data.get('max_lifetime', 0),
            'formation_rate': len(lifecycle_data['formation_events']) / steps,
            'dissolution_rate': len(lifecycle_data['dissolution_events']) / steps
        },
        'instability_analysis': {
            'primary_cause': max(lifecycle_data['instability_causes'], 
                                key=lifecycle_data['instability_causes'].get) if lifecycle_data['instability_causes'] else 'none',
            'cause_distribution': lifecycle_data['instability_causes'],
            'correlations': lifecycle_data.get('stability_correlations', {})
        }
    }
    
    # Identify critical instability windows
    results['critical_windows'] = identify_critical_windows(atom_counts)
    
    # Suggest parameter adjustments
    results['recommendations'] = generate_recommendations(results)
    
    # Save results
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = save_path / f"stability_analysis_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {json_path}")
        
        # Save visualization if available
        if hasattr(engine, 'history') and engine.history:
            try:
                viz_path = save_path / f"stability_viz_{timestamp}.png"
                plot_field_statistics(engine.history[-500:], save_path=str(viz_path))
                print(f"Visualization saved to: {viz_path}")
            except Exception as e:
                print(f"Could not save visualization: {e}")
    
    return results


def identify_critical_windows(atom_counts: List[int]) -> List[Dict]:
    """Identify time windows where atoms rapidly disappear."""
    critical_windows = []
    
    for i in range(1, len(atom_counts)):
        if atom_counts[i-1] > 0:
            loss_rate = (atom_counts[i-1] - atom_counts[i]) / atom_counts[i-1]
            if loss_rate > 0.2:  # Lost >20% of atoms
                critical_windows.append({
                    'step': i * 10,  # Accounting for sampling rate
                    'atoms_before': int(atom_counts[i-1]),
                    'atoms_after': int(atom_counts[i]),
                    'loss_rate': float(loss_rate)
                })
    
    return critical_windows


def generate_recommendations(results: Dict) -> Dict:
    """Generate parameter adjustment recommendations based on analysis."""
    recommendations = {}
    
    # Based on primary instability cause
    primary_cause = results['instability_analysis']['primary_cause']
    
    if primary_cause == 'thermal_fluctuation':
        recommendations['cooling_rate'] = {
            'current': 0.85,
            'suggested': 0.95,
            'reason': 'Reduce thermal disruption of atomic structures'
        }
        recommendations['sec_gamma'] = {
            'current': 0.01,
            'suggested': 0.005,
            'reason': 'Lower thermal coupling in SEC operator'
        }
    
    elif primary_cause == 'memory_decay':
        recommendations['memory_decay_rate'] = {
            'current': 0.001,
            'suggested': 0.0001,
            'reason': 'Slower memory decay to maintain structures'
        }
        recommendations['memory_growth_factor'] = {
            'current': 1.0,
            'suggested': 1.5,
            'reason': 'Stronger memory accumulation in stable regions'
        }
    
    elif primary_cause == 'field_divergence':
        recommendations['sec_beta'] = {
            'current': 0.5,
            'suggested': 0.7,
            'reason': 'Stronger spatial smoothing to maintain coherence'
        }
        recommendations['confluence_strength'] = {
            'current': 1.0,
            'suggested': 0.8,
            'reason': 'Gentler geometric evolution'
        }
    
    # Based on lifetime statistics
    avg_lifetime = results['stability_metrics'].get('average_lifetime', 0)
    if avg_lifetime > 0 and avg_lifetime < 50:
        recommendations['dt'] = {
            'current': 0.01,
            'suggested': 0.005,
            'reason': 'Smaller timestep for more stable evolution'
        }
    
    return recommendations


def print_analysis_summary(results: Dict):
    """Print human-readable analysis summary."""
    print("\n" + "="*60)
    print("STABILITY ANALYSIS SUMMARY")
    print("="*60)
    
    sim = results.get('simulation', {})
    print(f"\nSimulation Statistics:")
    print(f"  • Total steps: {sim.get('steps', 0)}")
    print(f"  • Max atoms observed: {sim.get('max_atoms', 0)}")
    print(f"  • Final atom count: {sim.get('final_atoms', 0)}")
    
    metrics = results.get('stability_metrics', {})
    print(f"\nLifetime Analysis:")
    print(f"  • Average lifetime: {metrics.get('average_lifetime', 0):.1f} steps")
    print(f"  • Maximum lifetime: {metrics.get('max_lifetime', 0)} steps")
    print(f"  • Formation rate: {metrics.get('formation_rate', 0):.3f} atoms/step")
    print(f"  • Dissolution rate: {metrics.get('dissolution_rate', 0):.3f} atoms/step")
    
    instability = results.get('instability_analysis', {})
    cause_dist = instability.get('cause_distribution', {})
    if any(cause_dist.values()):
        print(f"\nInstability Causes:")
        total_events = sum(cause_dist.values())
        for cause, count in cause_dist.items():
            percentage = (count / total_events) * 100 if total_events > 0 else 0
            print(f"  • {cause}: {count} events ({percentage:.1f}%)")
    
    critical = results.get('critical_windows', [])
    if critical:
        print(f"\nCritical Windows:")
        for window in critical[:3]:  # Show top 3
            print(f"  • Step {window['step']}: Lost {window['loss_rate']:.1%} of atoms")
    
    recs = results.get('recommendations', {})
    if recs:
        print(f"\nRecommendations:")
        for param, rec in list(recs.items())[:3]:  # Show top 3
            print(f"  • {param}: {rec['current']} → {rec['suggested']}")
            print(f"    Reason: {rec['reason']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run analysis with parameters tuned for atom formation
    results = run_stability_analysis(
        steps=2000,  # Longer evolution for structure formation
        size=(128, 32),
        save_dir="output/stability_analysis"
    )
    
    # Print summary
    print_analysis_summary(results)
    
    print("\n✅ Stability analysis complete!")
