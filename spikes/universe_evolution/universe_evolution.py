"""
Universe Evolution Analyzer

Long-run simulation to detect emergent structures:
- Gravity wells (density concentrations)
- Dark matter patterns (memory field clustering)
- Stellar formation (hot dense regions)
- Atomic structures (stable oscillating patterns)
- Molecular bonds (coupled oscillators)
"""
import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from scipy import ndimage

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine
from tools.atomic_analyzer import build_periodic_table
from emergence.structure_analyzer import StructureAnalyzer

def visualize_periodic_table(periodic_table: Dict, save_path: str = None):
    """Create a visual representation of the emergent periodic table."""
    import matplotlib.patches as patches
    
    if not periodic_table:
        print("No elements detected yet!")
        return None
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Sort by atomic number
    elements = sorted(periodic_table.items(), key=lambda x: x[0])
    
    # Create grid layout (simplified)
    max_z = max(periodic_table.keys())
    rows = (max_z - 1) // 10 + 1
    cols = 10
    
    for z, data in elements:
        # Position in grid
        row = (z - 1) // 10
        col = (z - 1) % 10
        
        x = col * 1.2
        y = rows - row - 1
        
        # Color based on occurrence frequency
        occurrences = data['occurrences']
        alpha = min(1.0, occurrences / 20.0)
        
        # Draw element box
        rect = patches.Rectangle((x, y), 1.0, 0.8, 
                                linewidth=2, 
                                edgecolor='black',
                                facecolor=(0, 0.5, 1, alpha))
        ax.add_patch(rect)
        
        # Add element symbol
        ax.text(x + 0.5, y + 0.5, data['element'], 
               ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Add atomic number
        ax.text(x + 0.1, y + 0.7, str(z), 
               ha='left', va='top', fontsize=8)
        
        # Add mass
        ax.text(x + 0.5, y + 0.2, f"{data['avg_mass']:.1f}", 
               ha='center', va='center', fontsize=8)
        
        # Add occurrence count
        ax.text(x + 0.9, y + 0.1, str(occurrences), 
               ha='right', va='bottom', fontsize=7, color='red', fontweight='bold')
    
    ax.set_xlim(-0.5, cols * 1.2)
    ax.set_ylim(-0.5, rows + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('Emergent Periodic Table from Reality Engine', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_text = 'Element Symbol (center) | Z (top-left) | Mass (below) | Count (bottom-right)'
    ax.text(cols * 1.2 / 2, -1, legend_text,
           ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Periodic table saved to: {save_path}")
    
    return fig

def run_universe_evolution(steps: int = 2000, size: Tuple[int, int] = (128, 32),
                          detect_interval: int = 50):
    """Run a long universe evolution and analyze emergent structures."""
    
    print("="*70)
    print("UNIVERSE EVOLUTION EXPERIMENT")
    print("="*70)
    print(f"Field size: {size[0]}×{size[1]}")
    print(f"Evolution steps: {steps}")
    print(f"Detection interval: every {detect_interval} steps")
    print("Looking for: Gravity Wells, Dark Matter, Stars, Atoms, Molecules")
    print("="*70 + "\n")
    
    # Initialize engine
    engine = RealityEngine(size=size, dt=0.1, device='cpu')
    engine.initialize(mode='big_bang')
    
    analyzer = StructureAnalyzer(engine)
    
    # Storage for detailed analysis
    atoms_history = []
    
    # Evolution loop
    print("[*] Starting universe evolution...\n")
    
    for step in range(steps):
        engine.step()
        
        # Analyze at intervals
        if step % detect_interval == 0:
            structures = analyzer.analyze_step(step)
            atoms_history.append(structures['atoms'])
            
            print(f"Step {step:4d}:")
            print(f"  Gravity wells: {len(structures['gravity_wells'])}")
            print(f"  Dark matter regions: {len(structures['dark_matter'])}")
            print(f"  Stellar candidates: {len(structures['stellar_regions'])}")
            print(f"  Atoms detected: {len(structures['atoms'])}")
            print(f"  Molecules formed: {len(structures['molecules'])}")
            
            if structures['atoms']:
                elements = [a.element for a in structures['atoms']]
                element_counts = {e: elements.count(e) for e in set(elements)}
                print(f"  Elements: {element_counts}")
            
            if structures['molecules']:
                formulas = [m['formula'] for m in structures['molecules']]
                molecule_counts = {f: formulas.count(f) for f in set(formulas)}
                print(f"  Molecules: {molecule_counts}")
            
            print()
    
    # Final visualization
    print("\n" + "="*70)
    print("FINAL UNIVERSE ANALYSIS")
    print("="*70 + "\n")
    
    # Build periodic table
    periodic_table = build_periodic_table(atoms_history)
    
    if periodic_table:
        print("📊 EMERGENT PERIODIC TABLE:")
        print("-"*70)
        for z in sorted(periodic_table.keys()):
            data = periodic_table[z]
            print(f"  {data['element']:3s} (Z={z:2d}): "
                  f"count={data['occurrences']:3d}, "
                  f"mass={data['avg_mass']:5.2f}, "
                  f"stability={data['avg_stability']:.3f}, "
                  f"quantum_n={data['quantum_states_observed']}")
        
        # Visualize periodic table
        viz_path = repo_root / 'examples' / 'emergent_periodic_table.png'
        visualize_periodic_table(periodic_table, save_path=viz_path)
    else:
        print("No stable atomic structures detected.")
    
    # Structure summary
    print("\n🌌 STRUCTURE EVOLUTION SUMMARY:")
    print("-"*70)
    print(f"  Max gravity wells: {max(analyzer.history['gravity_wells']) if analyzer.history['gravity_wells'] else 0}")
    print(f"  Max dark matter regions: {max(analyzer.history['dark_matter_regions']) if analyzer.history['dark_matter_regions'] else 0}")
    print(f"  Max stellar candidates: {max(analyzer.history['stellar_regions']) if analyzer.history['stellar_regions'] else 0}")
    print(f"  Max atoms: {max(analyzer.history['atoms']) if analyzer.history['atoms'] else 0}")
    print(f"  Max molecules: {max(analyzer.history['molecules']) if analyzer.history['molecules'] else 0}")
    
    # Create evolution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Final state fields
    state = engine.current_state
    
    im1 = axes[0, 0].imshow(state.M.cpu().numpy(), cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'Memory Field (Matter) - Step {steps}', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(state.T.cpu().numpy(), cmap='hot', aspect='auto')
    axes[0, 1].set_title('Temperature Field', fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    im3 = axes[0, 2].imshow(state.A.cpu().numpy(), cmap='RdBu_r', aspect='auto')
    axes[0, 2].set_title('Actual Field (Dynamics)', fontweight='bold')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # Structure evolution
    axes[1, 0].plot(analyzer.history['time'], analyzer.history['gravity_wells'], 
                   'o-', label='Gravity Wells', markersize=4)
    axes[1, 0].plot(analyzer.history['time'], analyzer.history['dark_matter_regions'], 
                   's-', label='Dark Matter', markersize=4)
    axes[1, 0].plot(analyzer.history['time'], analyzer.history['stellar_regions'], 
                   '^-', label='Stellar', markersize=4)
    axes[1, 0].set_xlabel('Step', fontweight='bold')
    axes[1, 0].set_ylabel('Count', fontweight='bold')
    axes[1, 0].set_title('Large Structure Evolution', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Atomic evolution
    axes[1, 1].plot(analyzer.history['time'], analyzer.history['atoms'], 
                   'o-', label='Atoms', color='green', markersize=4)
    axes[1, 1].plot(analyzer.history['time'], analyzer.history['molecules'], 
                   's-', label='Molecules', color='purple', markersize=4)
    axes[1, 1].set_xlabel('Step', fontweight='bold')
    axes[1, 1].set_ylabel('Count', fontweight='bold')
    axes[1, 1].set_title('Atomic/Molecular Evolution', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Periodic table bar chart
    if periodic_table:
        elements = sorted(periodic_table.keys())
        counts = [periodic_table[z]['occurrences'] for z in elements]
        labels = [periodic_table[z]['element'] for z in elements]
        
        axes[1, 2].bar(labels, counts, color='teal', edgecolor='black', linewidth=1.5)
        axes[1, 2].set_xlabel('Element', fontweight='bold')
        axes[1, 2].set_ylabel('Total Occurrences', fontweight='bold')
        axes[1, 2].set_title('Element Abundance', fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 2].text(0.5, 0.5, 'No atoms detected', 
                       ha='center', va='center', fontsize=14)
    
    fig.suptitle(f'Universe Evolution: {steps} Steps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    evolution_path = repo_root / 'examples' / f'universe_evolution_{steps}steps.png'
    plt.savefig(evolution_path, dpi=150, bbox_inches='tight')
    print(f"\n[+] Evolution visualization saved to: {evolution_path}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = repo_root / 'examples' / f'universe_results_{timestamp}.json'
    
    # Convert periodic table for JSON
    periodic_table_json = {}
    for z, data in periodic_table.items():
        periodic_table_json[str(z)] = data
    
    results = {
        'parameters': {
            'size': list(size),
            'steps': steps,
            'detect_interval': detect_interval
        },
        'periodic_table': periodic_table_json,
        'structure_history': analyzer.history,
        'summary': {
            'max_gravity_wells': max(analyzer.history['gravity_wells']) if analyzer.history['gravity_wells'] else 0,
            'max_dark_matter': max(analyzer.history['dark_matter_regions']) if analyzer.history['dark_matter_regions'] else 0,
            'max_stellar': max(analyzer.history['stellar_regions']) if analyzer.history['stellar_regions'] else 0,
            'max_atoms': max(analyzer.history['atoms']) if analyzer.history['atoms'] else 0,
            'max_molecules': max(analyzer.history['molecules']) if analyzer.history['molecules'] else 0,
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[+] Results saved to: {results_path}")
    
    print("\n" + "="*70)
    print("🌟 UNIVERSE EVOLUTION COMPLETE!")
    print("="*70)
    
    return results, periodic_table

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run universe evolution experiment')
    parser.add_argument('--steps', type=int, default=2000, help='Number of evolution steps')
    parser.add_argument('--size', type=int, default=128, help='Field size (u dimension)')
    parser.add_argument('--width', type=int, default=32, help='Field width (v dimension)')
    parser.add_argument('--interval', type=int, default=50, help='Detection interval')
    
    args = parser.parse_args()
    
    results, periodic_table = run_universe_evolution(
        steps=args.steps,
        size=(args.size, args.width),
        detect_interval=args.interval
    )
