"""
Compare Evolution Runs - Reality Engine

Compares multiple evolution runs to see differences in:
- Stellar structure types (stars vs black holes)
- Fusion activity (temperature evolution)
- Particle/atom formation patterns
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


def load_timeline(run_dir):
    """Load evolution timeline from run directory"""
    timeline_path = Path(run_dir) / 'evolution_timeline.json'
    with open(timeline_path, 'r') as f:
        return json.load(f)


def compare_runs(run_dirs, labels=None):
    """Compare multiple evolution runs"""
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(run_dirs))]
    
    # Load all timelines
    timelines = []
    for run_dir in run_dirs:
        try:
            timeline = load_timeline(run_dir)
            timelines.append(timeline)
        except FileNotFoundError:
            print(f"⚠ Could not find timeline in {run_dir}")
            return
    
    if not timelines:
        print("No valid timelines found!")
        return
    
    # Create comparison plots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Particle count evolution
    ax1 = fig.add_subplot(2, 3, 1)
    for timeline, label in zip(timelines, labels):
        steps = [t['step'] for t in timeline['timeline']]
        counts = [t['particle_count'] for t in timeline['timeline']]
        ax1.plot(steps, counts, marker='o', label=label, linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Particle Count')
    ax1.set_title('Particle Formation', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average particle mass
    ax2 = fig.add_subplot(2, 3, 2)
    for timeline, label in zip(timelines, labels):
        steps = [t['step'] for t in timeline['timeline']]
        masses = [t['avg_mass'] for t in timeline['timeline']]
        ax2.plot(steps, masses, marker='s', label=label, linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Average Mass')
    ax2.set_title('Mass Growth', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Composite structures
    ax3 = fig.add_subplot(2, 3, 3)
    for timeline, label in zip(timelines, labels):
        steps = [t['step'] for t in timeline['timeline']]
        composites = [t['composite_count'] for t in timeline['timeline']]
        ax3.plot(steps, composites, marker='^', label=label, linewidth=2)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Composite Count')
    ax3.set_title('Atom/Molecule Formation', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Stellar structures
    ax4 = fig.add_subplot(2, 3, 4)
    for timeline, label in zip(timelines, labels):
        steps = [t['step'] for t in timeline['timeline']]
        stellar = [t.get('stellar_count', 0) for t in timeline['timeline']]
        ax4.plot(steps, stellar, marker='*', label=label, linewidth=2, markersize=10)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Stellar Structure Count')
    ax4.set_title('Gravitational Wells', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Fusion events
    ax5 = fig.add_subplot(2, 3, 5)
    for timeline, label in zip(timelines, labels):
        steps = [t['step'] for t in timeline['timeline']]
        fusion = [t.get('fusion_count', 0) for t in timeline['timeline']]
        total_fusion = [sum(fusion[:i+1]) for i in range(len(fusion))]
        ax5.plot(steps, total_fusion, marker='D', label=label, linewidth=2)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Cumulative Fusion Events')
    ax5.set_title('Stellar Nucleosynthesis', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_title('Final State Comparison', fontweight='bold', fontsize=12)
    
    summary_text = ""
    for timeline, label in zip(timelines, labels):
        final = timeline['timeline'][-1]
        summary_text += f"{label}:\n"
        summary_text += f"  Particles: {final['particle_count']}\n"
        summary_text += f"  Composites: {final['composite_count']}\n"
        summary_text += f"  Stellar: {final.get('stellar_count', 0)}\n"
        summary_text += f"  Fusion: {sum(t.get('fusion_count', 0) for t in timeline['timeline'])}\n"
        summary_text += f"  Avg Mass: {final['avg_mass']:.1f}\n"
        summary_text += f"  Separation: {final.get('avg_separation', 0):.1f}\n"
        summary_text += f"\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save comparison
    output_path = Path('output') / 'run_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Comparison plot: {output_path}")
    
    plt.show()
    plt.close()
    
    # Print key differences
    print("\n" + "="*70)
    print("KEY DIFFERENCES")
    print("="*70)
    
    for i, (timeline, label) in enumerate(zip(timelines, labels)):
        total_fusion = sum(t.get('fusion_count', 0) for t in timeline['timeline'])
        final_stellar = timeline['timeline'][-1].get('stellar_count', 0)
        
        print(f"\n{label}:")
        if total_fusion > 0:
            print(f"  ⭐ FUSION ACTIVE: {total_fusion} events detected!")
        else:
            print(f"  ❄️  Cold collapse: No fusion detected")
        
        print(f"  Stellar structures: {final_stellar}")
        
        # Check for mass concentration pattern
        masses = [t['avg_mass'] for t in timeline['timeline']]
        mass_growth = masses[-1] / masses[0]
        print(f"  Mass growth: {mass_growth:.1f}x")


if __name__ == '__main__':
    output_dir = Path(__file__).parent.parent / 'output'
    
    # Find longrun directories
    longrun_dirs = sorted(output_dir.glob('*_longrun'))
    
    if len(longrun_dirs) < 1:
        print("Need at least one longrun directory to compare!")
        print("Run watch_atoms_emerge.py first")
        sys.exit(1)
    
    print("="*70)
    print("EVOLUTION RUN COMPARISON")
    print("="*70)
    
    if len(longrun_dirs) == 1:
        print(f"Only one run found: {longrun_dirs[0].name}")
        print("Run watch_atoms_emerge.py again to compare with new physics!")
    else:
        print(f"Found {len(longrun_dirs)} runs:")
        for d in longrun_dirs:
            print(f"  - {d.name}")
        
        # Compare last two runs
        compare_dirs = longrun_dirs[-2:]
        labels = [
            f"{d.name.split('_')[0]}_{d.name.split('_')[1]}" 
            for d in compare_dirs
        ]
        
        print(f"\nComparing: {labels[0]} vs {labels[1]}")
        compare_runs(compare_dirs, labels)
