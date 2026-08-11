"""
Periodic Table Visualization for Emergent Particles

Creates visual representations of discovered particles and their properties
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Dict, List
import json
from pathlib import Path


class PeriodicTableVisualizer:
    """Visualize emergent particle periodic table"""
    
    def __init__(self):
        self.colors = {
            'photon': '#FFD700',      # Gold
            'electron': '#4169E1',     # Royal Blue
            'positron': '#FF69B4',     # Hot Pink
            'neutrino': '#E6E6FA',     # Lavender
            'meson': '#90EE90',        # Light Green
            'fermion': '#87CEEB',      # Sky Blue
            'boson': '#DDA0DD',        # Plum
            'proton': '#FF6347',       # Tomato
            'neutron': '#A9A9A9',      # Dark Gray
            'exotic': '#FF00FF'        # Magenta
        }
    
    def create_periodic_table(self, periodic_table: Dict, save_path: str = None):
        """Create visual periodic table of particles"""
        
        if not periodic_table:
            print("No particles to visualize")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.4)
        
        # Main title
        fig.suptitle('Emergent Particle Periodic Table\nReality Engine Discovery', 
                     fontsize=16, fontweight='bold')
        
        # Create a subplot for each particle type
        row, col = 0, 0
        for particle_type, data in periodic_table.items():
            if row >= 4:
                break
            ax = fig.add_subplot(gs[row, col])
            self._draw_particle_card(ax, particle_type, data)
            
            col += 1
            if col >= 5:
                col = 0
                row += 1
        
        # Add legend
        self._add_legend(fig)
        
        # Add statistics panel
        self._add_statistics(fig, periodic_table)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SAVED] Periodic table: {save_path}")
        
        return fig
    
    def _draw_particle_card(self, ax, particle_type: str, data: Dict):
        """Draw individual particle card"""
        
        # Set up the card
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        
        # Background color based on particle type
        bg_color = self.colors.get(particle_type, '#FFFFFF')
        rect = patches.Rectangle((0, 0), 10, 10, 
                                linewidth=2, edgecolor='black', 
                                facecolor=bg_color, alpha=0.3)
        ax.add_patch(rect)
        
        # Particle name
        ax.text(5, 8.5, particle_type.upper(), 
                ha='center', fontsize=12, fontweight='bold')
        
        # Count
        ax.text(5, 7.5, f"Count: {data['count']}", 
                ha='center', fontsize=10)
        
        # Mass
        ax.text(5, 6.5, f"Mass: {data['avg_mass']:.3f}", 
                ha='center', fontsize=9)
        
        # Charge
        charge_str = f"Charge: {data['avg_charge']:+.3f}"
        ax.text(5, 5.5, charge_str, 
                ha='center', fontsize=9)
        
        # Spin
        ax.text(5, 4.5, f"Spin: {data['avg_spin']:.2f}", 
                ha='center', fontsize=9)
        
        # Mass range bar
        if data['mass_range'][1] > data['mass_range'][0]:
            range_width = min((data['mass_range'][1] - data['mass_range'][0]) * 5, 8)
            range_bar = patches.Rectangle((1, 2), range_width, 0.5,
                                         facecolor='gray', alpha=0.5)
            ax.add_patch(range_bar)
            ax.text(5, 1.5, 'Mass Range', ha='center', fontsize=8)
        
        # Visual spin indicator
        if abs(data['avg_spin']) > 0.1:
            arrow = patches.FancyArrowPatch((7, 8), (8.5, 8),
                                          mutation_scale=20,
                                          arrowstyle='->', 
                                          color='black', lw=2)
            ax.add_patch(arrow)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    
    def _add_legend(self, fig):
        """Add color legend"""
        legend_elements = [
            patches.Patch(facecolor=color, alpha=0.3, edgecolor='black', 
                        label=ptype.capitalize())
            for ptype, color in self.colors.items()
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=5, frameon=True, title='Particle Types',
                  fontsize=9)
    
    def _add_statistics(self, fig, periodic_table: Dict):
        """Add statistics panel"""
        stats_text = self._calculate_statistics(periodic_table)
        fig.text(0.02, 0.5, stats_text, transform=fig.transFigure,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
    
    def _calculate_statistics(self, periodic_table: Dict) -> str:
        """Calculate overall statistics"""
        total_particles = sum(d['count'] for d in periodic_table.values())
        particle_types = len(periodic_table)
        
        if total_particles > 0:
            avg_mass = sum(d['avg_mass'] * d['count'] 
                         for d in periodic_table.values()) / total_particles
            avg_charge = sum(d['avg_charge'] * d['count'] 
                           for d in periodic_table.values()) / total_particles
            
            # Find most common
            most_common = max(periodic_table.items(), 
                            key=lambda x: x[1]['count'])[0]
            
            stats = f"""STATISTICS
━━━━━━━━━━━
Total: {total_particles}
Types: {particle_types}
Most: {most_common}

Avg Mass: {avg_mass:.3f}
Avg Charge: {avg_charge:+.3f}

Charges:
 +: {sum(1 for d in periodic_table.values() if d['avg_charge'] > 0.1)}
 -: {sum(1 for d in periodic_table.values() if d['avg_charge'] < -0.1)}
 0: {sum(1 for d in periodic_table.values() if abs(d['avg_charge']) < 0.1)}
"""
        else:
            stats = "No particles\ndetected"
        
        return stats

    def plot_mass_spectrum(self, particles: List, save_path: str = None):
        """Plot mass spectrum of particles"""
        
        if not particles:
            print("No particles to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Mass distribution
        masses = [p.mass for p in particles]
        axes[0, 0].hist(masses, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Mass')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Mass Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Charge distribution
        charges = [p.charge for p in particles]
        axes[0, 1].hist(charges, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_xlabel('Charge')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Charge Distribution')
        axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mass vs Charge scatter
        axes[1, 0].scatter(masses, charges, alpha=0.6, s=20, c='purple')
        axes[1, 0].set_xlabel('Mass')
        axes[1, 0].set_ylabel('Charge')
        axes[1, 0].set_title('Mass-Charge Relationship')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Stability distribution
        stabilities = [p.stability for p in particles]
        axes[1, 1].hist(stabilities, bins=30, alpha=0.7, 
                       color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Stability')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Stability Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Particle Property Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SAVED] Mass spectrum: {save_path}")
        
        return fig

    def create_3d_particle_map(self, particles: List, save_path: str = None):
        """Create 3D visualization of particle positions"""
        
        if not particles:
            print("No particles to plot")
            return
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot particles by type
        particle_types = {}
        for p in particles:
            ptype = p.classification
            if ptype not in particle_types:
                particle_types[ptype] = []
            particle_types[ptype].append(p)
        
        for ptype, plist in particle_types.items():
            xs = [p.position[0] for p in plist]
            ys = [p.position[1] for p in plist]
            zs = [p.position[2] for p in plist]
            sizes = [50 * p.mass for p in plist]
            
            color = self.colors.get(ptype, '#000000')
            ax.scatter(xs, ys, zs, c=color, s=sizes, alpha=0.7,
                      edgecolors='black', linewidth=1, label=ptype)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Particle Distribution in Reality Field')
        ax.legend(loc='upper right', fontsize=9)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SAVED] 3D particle map: {save_path}")
        
        return fig
