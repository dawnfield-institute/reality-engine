"""
Visualize Emergent Stellar Structures

Shows gravitational wells, mass concentrations, and fusion regions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from emergence.stellar_analyzer import StellarAnalyzer


def visualize_structures(structures, universe_size=64, save_path=None):
    """
    Create 3D visualization of stellar structures
    """
    if not structures:
        print("No structures to visualize")
        return
    
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_title('Stellar Structures in Universe', fontsize=14, fontweight='bold')
    
    # Color map by structure type
    type_colors = {
        'black_hole': '#000000',
        'massive_star': '#FF0000',
        'star': '#FFA500',
        'dwarf_star': '#FFFF00',
        'proto_star': '#00FF00',
        'mass_concentration': '#0000FF',
        'unknown': '#888888'
    }
    
    # Plot each structure
    for s in structures:
        x, y, z = s.position
        color = type_colors.get(s.structure_type, '#888888')
        
        # Size proportional to mass
        size = min(1000, s.total_mass / 10)
        
        # Alpha based on density
        alpha = min(1.0, s.core_density / 200.0)
        
        ax1.scatter([x], [y], [z], c=[color], s=size, alpha=alpha, 
                   edgecolors='white', linewidths=0.5)
        
        # Draw radius sphere (simplified)
        if s.radius > 2:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            xs = s.radius * np.outer(np.cos(u), np.sin(v)) + x
            ys = s.radius * np.outer(np.sin(u), np.sin(v)) + y
            zs = s.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
            ax1.plot_surface(xs, ys, zs, color=color, alpha=0.1)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(0, universe_size)
    ax1.set_ylim(0, universe_size)
    ax1.set_zlim(0, universe_size)
    
    # Mass distribution
    ax2 = fig.add_subplot(222)
    ax2.set_title('Mass Distribution', fontsize=12, fontweight='bold')
    
    masses = [s.total_mass for s in structures]
    types = [s.structure_type for s in structures]
    
    # Group by type
    type_masses = {}
    for s in structures:
        if s.structure_type not in type_masses:
            type_masses[s.structure_type] = []
        type_masses[s.structure_type].append(s.total_mass)
    
    colors = [type_colors.get(t, '#888888') for t in type_masses.keys()]
    ax2.bar(range(len(type_masses)), 
           [sum(m) for m in type_masses.values()],
           color=colors)
    ax2.set_xticks(range(len(type_masses)))
    ax2.set_xticklabels([t.replace('_', '\n') for t in type_masses.keys()], 
                        rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Total Mass')
    ax2.grid(True, alpha=0.3)
    
    # Temperature vs Mass
    ax3 = fig.add_subplot(223)
    ax3.set_title('Temperature vs Mass (HR-like Diagram)', fontsize=12, fontweight='bold')
    
    for s in structures:
        color = type_colors.get(s.structure_type, '#888888')
        size = min(200, s.core_density)
        ax3.scatter(s.total_mass, s.temperature, c=color, s=size, alpha=0.6,
                   edgecolors='black', linewidths=0.5)
    
    ax3.set_xlabel('Mass')
    ax3.set_ylabel('Temperature')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Add fusion line
    if any(s.is_fusion_active() for s in structures):
        fusion_threshold_x = [min(masses), max(masses)]
        fusion_threshold_y = [10.0, 10.0]  # Temperature threshold
        ax3.plot(fusion_threshold_x, fusion_threshold_y, 'r--', 
                linewidth=2, label='Fusion Threshold', alpha=0.7)
        ax3.legend()
    
    # Statistics panel
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    ax4.set_title('Structure Statistics', fontsize=12, fontweight='bold')
    
    stats_text = f"Total Structures: {len(structures)}\n\n"
    
    # Count by type
    stats_text += "By Type:\n"
    for stype, masses in sorted(type_masses.items(), 
                               key=lambda x: sum(x[1]), reverse=True):
        count = len(masses)
        total_mass = sum(masses)
        avg_mass = total_mass / count
        stats_text += f"  {stype}: {count} (avg mass: {avg_mass:.0f})\n"
    
    stats_text += f"\n"
    
    # Fusion activity
    fusion_active = sum(1 for s in structures if s.is_fusion_active())
    stats_text += f"Fusion Active: {fusion_active}/{len(structures)}\n"
    
    if fusion_active > 0:
        stats_text += f"\n>>> {fusion_active} structures creating new atoms! <<<\n"
    
    # Mass stats
    stats_text += f"\nMass Statistics:\n"
    stats_text += f"  Total: {sum(masses):.0f}\n"
    stats_text += f"  Mean: {np.mean(masses):.0f}\n"
    stats_text += f"  Median: {np.median(masses):.0f}\n"
    stats_text += f"  Max: {max(masses):.0f}\n"
    
    # Density stats
    densities = [s.core_density for s in structures]
    stats_text += f"\nCore Density:\n"
    stats_text += f"  Mean: {np.mean(densities):.1f}\n"
    stats_text += f"  Max: {max(densities):.1f}\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Legend for structure types
    legend_y = 0.1
    for stype, color in type_colors.items():
        if stype in type_masses:
            ax4.scatter([], [], c=color, s=100, label=stype.replace('_', ' '))
    
    ax4.legend(loc='lower left', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Stellar structures: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    print("Stellar structure visualization")
    print("Use this with output from watch_atoms_emerge.py")
