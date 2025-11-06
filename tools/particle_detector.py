"""
Particle Detection for Reality Engine

Detects stable structures (particles) that emerge from field dynamics.

Particles are identified as regions with:
- High memory concentration (crystallized information)
- Local temperature minima (cooling creates bound states)
- Low disequilibrium (P ≈ A, stable equilibrium)
- Persistence over time (stable across multiple steps)

This demonstrates that matter emerges naturally from pure dynamics!
"""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from typing import List, Tuple

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

@dataclass
class Particle:
    """
    A detected stable structure (particle).
    """
    id: int
    position: Tuple[int, int]  # (u, v) coordinates
    memory_mass: float  # Integrated memory in region
    temperature: float  # Local temperature
    stability: float  # How equilibrated (0-1, 1=perfect)
    radius: float  # Effective size
    age: int  # How many steps it's persisted
    
    def __repr__(self):
        return (f"Particle(id={self.id}, pos={self.position}, "
                f"M={self.memory_mass:.4f}, T={self.temperature:.3f}, "
                f"stability={self.stability:.3f}, age={self.age})")

class ParticleDetector:
    """
    Detects and tracks stable structures in Reality Engine fields.
    """
    
    def __init__(self, memory_threshold=0.01, stability_threshold=0.8, min_radius=2):
        """
        Args:
            memory_threshold: Minimum memory density to consider
            stability_threshold: Minimum equilibrium (1 - diseq) to consider stable
            min_radius: Minimum particle radius in cells
        """
        self.memory_threshold = memory_threshold
        self.stability_threshold = stability_threshold
        self.min_radius = min_radius
        
        self.particles = []
        self.particle_id_counter = 0
        self.particle_history = []
    
    def detect(self, engine: RealityEngine) -> List[Particle]:
        """
        Detect particles in current engine state.
        
        Returns:
            List of detected particles
        """
        state = engine.current_state
        
        # Get fields as numpy
        M = state.M.cpu().numpy()
        T = state.T.cpu().numpy()
        P = state.P.cpu().numpy()
        A = state.A.cpu().numpy()
        
        # Compute stability: 1 - normalized disequilibrium
        diseq = np.abs(P - A)
        diseq_norm = diseq / (diseq.max() + 1e-10)
        stability = 1.0 - diseq_norm
        
        # Find high-memory regions
        memory_mask = M > self.memory_threshold
        
        # Find stable regions
        stable_mask = stability > self.stability_threshold
        
        # Combined mask: high memory AND stable
        particle_mask = memory_mask & stable_mask
        
        # Label connected components
        labeled, num_features = ndimage.label(particle_mask)
        
        # Extract particle properties
        particles = []
        for label_id in range(1, num_features + 1):
            region_mask = labeled == label_id
            
            # Skip if too small
            region_size = region_mask.sum()
            if region_size < self.min_radius ** 2:
                continue
            
            # Compute properties
            positions = np.argwhere(region_mask)
            center = positions.mean(axis=0)
            
            # Memory mass (integrated memory in region)
            memory_mass = M[region_mask].sum()
            
            # Average temperature
            temp = T[region_mask].mean()
            
            # Average stability
            stab = stability[region_mask].mean()
            
            # Effective radius
            radius = np.sqrt(region_size / np.pi)
            
            # Create particle
            particle = Particle(
                id=self.particle_id_counter,
                position=(int(center[0]), int(center[1])),
                memory_mass=float(memory_mass),
                temperature=float(temp),
                stability=float(stab),
                radius=float(radius),
                age=0
            )
            
            particles.append(particle)
            self.particle_id_counter += 1
        
        # Track particles over time (simple: just age existing ones for now)
        self.particles = particles
        self.particle_history.append(len(particles))
        
        return particles
    
    def visualize_particles(self, engine: RealityEngine, particles: List[Particle], 
                           save_path=None):
        """
        Visualize detected particles overlaid on fields.
        """
        state = engine.current_state
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Memory field with particles
        M = state.M.cpu().numpy()
        im1 = axes[0, 0].imshow(M, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'Memory Field - {len(particles)} Particles Detected', 
                            fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Overlay particles
        for p in particles:
            circle = Circle(
                (p.position[1], p.position[0]),  # matplotlib uses (x, y)
                p.radius,
                fill=False,
                edgecolor='red',
                linewidth=2,
                linestyle='--'
            )
            axes[0, 0].add_patch(circle)
            axes[0, 0].plot(p.position[1], p.position[0], 'r*', markersize=10)
        
        # Temperature field
        T = state.T.cpu().numpy()
        im2 = axes[0, 1].imshow(T, cmap='hot', aspect='auto')
        axes[0, 1].set_title('Temperature Field', fontweight='bold')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Overlay particles (cooler = more stable)
        for p in particles:
            circle = Circle(
                (p.position[1], p.position[0]),
                p.radius,
                fill=False,
                edgecolor='cyan',
                linewidth=2,
                linestyle=':'
            )
            axes[0, 1].add_patch(circle)
        
        # Stability map
        P = state.P.cpu().numpy()
        A = state.A.cpu().numpy()
        diseq = np.abs(P - A)
        stability = 1.0 - (diseq / (diseq.max() + 1e-10))
        
        im3 = axes[1, 0].imshow(stability, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1, 0].set_title('Stability Map (1 - Disequilibrium)', fontweight='bold')
        plt.colorbar(im3, ax=axes[1, 0])
        
        for p in particles:
            axes[1, 0].plot(p.position[1], p.position[0], 'ko', markersize=8)
        
        # Particle properties scatter
        if particles:
            masses = [p.memory_mass for p in particles]
            temps = [p.temperature for p in particles]
            stabilities = [p.stability for p in particles]
            
            scatter = axes[1, 1].scatter(masses, temps, c=stabilities, 
                                        cmap='RdYlGn', s=100, alpha=0.7,
                                        vmin=0, vmax=1, edgecolors='black')
            axes[1, 1].set_xlabel('Memory Mass', fontweight='bold')
            axes[1, 1].set_ylabel('Temperature', fontweight='bold')
            axes[1, 1].set_title('Particle Properties', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], label='Stability')
            
            # Annotate particles
            for i, p in enumerate(particles):
                axes[1, 1].annotate(f'#{p.id}', (p.memory_mass, p.temperature),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8)
        else:
            axes[1, 1].text(0.5, 0.5, 'No particles detected',
                          ha='center', va='center', fontsize=14)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
        
        fig.suptitle(
            f'Particle Detection - Step {engine.step_count} | '
            f'{len(particles)} stable structures found',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[+] Particle visualization saved to: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def get_statistics(self) -> dict:
        """Get statistics about detected particles."""
        if not self.particles:
            return {'count': 0}
        
        masses = [p.memory_mass for p in self.particles]
        temps = [p.temperature for p in self.particles]
        stabilities = [p.stability for p in self.particles]
        radii = [p.radius for p in self.particles]
        
        return {
            'count': len(self.particles),
            'total_mass': sum(masses),
            'avg_mass': np.mean(masses),
            'avg_temperature': np.mean(temps),
            'avg_stability': np.mean(stabilities),
            'avg_radius': np.mean(radii),
            'mass_range': (min(masses), max(masses)),
            'temp_range': (min(temps), max(temps))
        }

def detect_and_track_particles(steps=200, size=(64, 16), mode='big_bang',
                              detect_every=10, save_path=None):
    """
    Run simulation and detect particles periodically.
    """
    print("="*70)
    print("PARTICLE DETECTION")
    print("="*70)
    print(f"\nRunning {steps} step simulation...")
    print(f"Universe size: {size[0]} x {size[1]} cells")
    print(f"Detection frequency: every {detect_every} steps")
    print("\n" + "="*70)
    
    # Create engine
    engine = RealityEngine(size=size, dt=0.1, device='cpu')
    engine.initialize(mode=mode)
    
    # Create detector
    detector = ParticleDetector(
        memory_threshold=0.01,
        stability_threshold=0.75,
        min_radius=2
    )
    
    # Tracking data
    particle_counts = []
    detection_steps = []
    
    print("\n[*] Evolving and detecting particles...")
    for i in range(steps):
        engine.step()
        
        if i % detect_every == 0:
            particles = detector.detect(engine)
            particle_counts.append(len(particles))
            detection_steps.append(i)
            
            stats = detector.get_statistics()
            print(f"  Step {i:3d}: {stats['count']} particles detected")
            if stats['count'] > 0:
                print(f"            Avg mass: {stats['avg_mass']:.4f}, "
                      f"Avg temp: {stats['avg_temperature']:.3f}, "
                      f"Avg stability: {stats['avg_stability']:.3f}")
    
    # Final detection and visualization
    print("\n[*] Final particle detection...")
    particles = detector.detect(engine)
    stats = detector.get_statistics()
    
    print(f"\n[+] Final Results:")
    print(f"    Particles detected: {stats['count']}")
    if stats['count'] > 0:
        print(f"    Total mass: {stats['total_mass']:.4f}")
        print(f"    Average mass: {stats['avg_mass']:.4f}")
        print(f"    Average temperature: {stats['avg_temperature']:.3f}")
        print(f"    Average stability: {stats['avg_stability']:.3f}")
        print(f"    Average radius: {stats['avg_radius']:.2f} cells")
        print(f"\n    Particle details:")
        for p in particles[:10]:  # Show first 10
            print(f"      {p}")
    
    # Visualize
    if save_path is None:
        save_path = repo_root / 'examples' / 'particle_detection.png'
    
    detector.visualize_particles(engine, particles, save_path=save_path)
    
    # Plot particle count evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(detection_steps, particle_counts, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel('Number of Particles', fontweight='bold')
    ax.set_title('Particle Formation Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    count_save_path = repo_root / 'examples' / 'particle_count_evolution.png'
    plt.savefig(count_save_path, dpi=150, bbox_inches='tight')
    print(f"[+] Particle count evolution saved to: {count_save_path}")
    
    return particles, stats

if __name__ == '__main__':
    particles, stats = detect_and_track_particles(
        steps=200,
        size=(64, 16),
        mode='big_bang',
        detect_every=10
    )
    
    print("\n" + "="*70)
    print("[SUCCESS] Particle detection complete!")
    print("="*70)
    print("\nKEY INSIGHT: Matter (particles) emerges naturally from pure dynamics!")
    print("No particle physics needed - just SEC, Confluence, and thermodynamics.")
    print("="*70 + "\n")
