"""
Real-Time Field Visualization for Reality Engine

Visualize the four fundamental fields as they evolve:
- P (Potential): What could be
- A (Actual): What is
- M (Memory): Crystallized information (matter)
- T (Temperature): Heat distribution

Shows structure formation, heat diffusion, and matter crystallization in real-time!
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine

class FieldVisualizer:
    """
    Real-time visualizer for Reality Engine fields.
    """
    
    def __init__(self, engine: RealityEngine):
        self.engine = engine
        self.history = {
            'P': [], 'A': [], 'M': [], 'T': [],
            'T_mean': [], 'M_mean': [], 'step': []
        }
        
        # Create figure with 2x3 layout
        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        
        # Field plots (top row)
        self.ax_P = self.fig.add_subplot(gs[0, 0])
        self.ax_A = self.fig.add_subplot(gs[0, 1])
        self.ax_M = self.fig.add_subplot(gs[0, 2])
        
        # Temperature and time series (bottom row)
        self.ax_T = self.fig.add_subplot(gs[1, 0])
        self.ax_T_history = self.fig.add_subplot(gs[1, 1])
        self.ax_M_history = self.fig.add_subplot(gs[1, 2])
        
        # Create custom colormaps
        self.cmap_diverging = 'RdBu_r'  # For P, A (red=positive, blue=negative)
        self.cmap_sequential = 'viridis'  # For M (matter density)
        self.cmap_hot = 'hot'  # For T (temperature)
        
        # Initialize image plots
        self.im_P = None
        self.im_A = None
        self.im_M = None
        self.im_T = None
        
        # Line plots for history
        self.line_T = None
        self.line_M = None
        
        self.step_count = 0
    
    def update(self, frame=None):
        """
        Update visualization with current engine state.
        """
        # Evolve one step
        state = self.engine.step()
        self.step_count += 1
        
        # Get current fields
        P = self.engine.current_state.P.cpu().numpy()
        A = self.engine.current_state.A.cpu().numpy()
        M = self.engine.current_state.M.cpu().numpy()
        T = self.engine.current_state.T.cpu().numpy()
        
        # Record history
        self.history['P'].append(P.copy())
        self.history['A'].append(A.copy())
        self.history['M'].append(M.copy())
        self.history['T'].append(T.copy())
        self.history['T_mean'].append(T.mean())
        self.history['M_mean'].append(M.mean())
        self.history['step'].append(self.step_count)
        
        # Keep only recent history (last 200 steps)
        max_history = 200
        if len(self.history['step']) > max_history:
            for key in ['P', 'A', 'M', 'T', 'T_mean', 'M_mean', 'step']:
                self.history[key] = self.history[key][-max_history:]
        
        # Update field plots
        self._update_field_plot(self.ax_P, P, 'Potential Field (P)', self.cmap_diverging, 
                                self.im_P, symmetric=True)
        self._update_field_plot(self.ax_A, A, 'Actual Field (A)', self.cmap_diverging,
                                self.im_A, symmetric=True)
        self._update_field_plot(self.ax_M, M, 'Memory Field (M) - Matter', self.cmap_sequential,
                                self.im_M, symmetric=False)
        self._update_field_plot(self.ax_T, T, 'Temperature Field (T)', self.cmap_hot,
                                self.im_T, symmetric=False)
        
        # Update time series
        self._update_time_series()
        
        # Update title with current stats
        self.fig.suptitle(
            f'Reality Engine: Step {self.step_count} | '
            f'T_mean={T.mean():.3f} | M_mean={M.mean():.5f} | '
            f'Time={self.engine.time_elapsed:.2f}',
            fontsize=14, fontweight='bold'
        )
        
        return [self.im_P, self.im_A, self.im_M, self.im_T, self.line_T, self.line_M]
    
    def _update_field_plot(self, ax, field, title, cmap, im, symmetric=True):
        """Update a single field plot."""
        if symmetric:
            # Symmetric colormap around zero
            vmax = max(abs(field.min()), abs(field.max()))
            vmin = -vmax
        else:
            vmin = field.min()
            vmax = field.max()
        
        if im is None:
            # First time - create image
            im = ax.imshow(field, cmap=cmap, aspect='auto', 
                          interpolation='bilinear', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('v coordinate')
            ax.set_ylabel('u coordinate')
            
            # Store reference
            if 'Potential' in title:
                self.im_P = im
            elif 'Actual' in title:
                self.im_A = im
            elif 'Memory' in title:
                self.im_M = im
            elif 'Temperature' in title:
                self.im_T = im
        else:
            # Update existing image
            im.set_data(field)
            im.set_clim(vmin, vmax)
    
    def _update_time_series(self):
        """Update time series plots."""
        steps = self.history['step']
        
        # Temperature history
        self.ax_T_history.clear()
        self.ax_T_history.plot(steps, self.history['T_mean'], 
                               linewidth=2, color='red', label='T_mean')
        self.ax_T_history.set_xlabel('Step')
        self.ax_T_history.set_ylabel('Mean Temperature')
        self.ax_T_history.set_title('Temperature Evolution', fontsize=11, fontweight='bold')
        self.ax_T_history.grid(True, alpha=0.3)
        self.ax_T_history.legend()
        
        # Memory history
        self.ax_M_history.clear()
        self.ax_M_history.plot(steps, self.history['M_mean'],
                               linewidth=2, color='blue', label='M_mean')
        self.ax_M_history.set_xlabel('Step')
        self.ax_M_history.set_ylabel('Mean Memory')
        self.ax_M_history.set_title('Matter Crystallization', fontsize=11, fontweight='bold')
        self.ax_M_history.grid(True, alpha=0.3)
        self.ax_M_history.legend()
    
    def animate(self, steps=200, interval=50, save_path=None):
        """
        Create animation of field evolution.
        
        Args:
            steps: Number of evolution steps
            interval: Milliseconds between frames
            save_path: If provided, save animation to file
        """
        print(f"[*] Creating animation for {steps} steps...")
        print(f"    Interval: {interval}ms per frame")
        
        anim = FuncAnimation(
            self.fig, self.update, frames=steps,
            interval=interval, blit=False, repeat=False
        )
        
        if save_path:
            print(f"[*] Saving animation to: {save_path}")
            anim.save(save_path, writer='pillow', fps=20, dpi=100)
            print(f"[+] Animation saved!")
        else:
            plt.show()
        
        return anim
    
    def snapshot(self, save_path=None):
        """
        Take a single snapshot of current state.
        """
        self.update()
        
        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[+] Snapshot saved to: {save_path}")
        else:
            plt.show()

def visualize_evolution(steps=100, size=(64, 16), mode='big_bang', save_gif=False):
    """
    Visualize field evolution in real-time.
    
    Args:
        steps: Number of evolution steps
        size: Field dimensions (nu, nv)
        mode: Initialization mode
        save_gif: If True, save animation as GIF
    """
    print("="*70)
    print("FIELD VISUALIZATION")
    print("="*70)
    print(f"\nUniverse size: {size[0]} x {size[1]} cells")
    print(f"Evolution steps: {steps}")
    print(f"Initial mode: {mode}")
    print("\nFields to visualize:")
    print("  - Potential (P): What could be")
    print("  - Actual (A): What is")
    print("  - Memory (M): Crystallized information (matter)")
    print("  - Temperature (T): Heat distribution")
    print("\n" + "="*70)
    
    # Create engine
    engine = RealityEngine(size=size, dt=0.1, device='cpu')
    engine.initialize(mode=mode)
    
    # Create visualizer
    viz = FieldVisualizer(engine)
    
    # Animate or snapshot
    if save_gif:
        save_path = repo_root / 'examples' / f'field_evolution_{mode}.gif'
        viz.animate(steps=steps, interval=50, save_path=save_path)
    else:
        print("\n[*] Taking snapshots at key moments...")
        
        # Initial snapshot
        viz.snapshot(save_path=repo_root / 'examples' / f'fields_step_000.png')
        
        # Evolve to mid-point
        for _ in range(steps // 2):
            viz.update()
        viz.snapshot(save_path=repo_root / 'examples' / f'fields_step_{steps//2:03d}.png')
        
        # Evolve to end
        for _ in range(steps // 2):
            viz.update()
        viz.snapshot(save_path=repo_root / 'examples' / f'fields_step_{steps:03d}.png')
        
        print(f"\n[+] Snapshots saved!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Reality Engine field evolution')
    parser.add_argument('--steps', type=int, default=100, help='Number of evolution steps')
    parser.add_argument('--size', type=int, nargs=2, default=[64, 16], help='Field size (nu nv)')
    parser.add_argument('--mode', type=str, default='big_bang', 
                       choices=['big_bang', 'random', 'cold'],
                       help='Initialization mode')
    parser.add_argument('--gif', action='store_true', help='Save as animated GIF')
    
    args = parser.parse_args()
    
    visualize_evolution(
        steps=args.steps,
        size=tuple(args.size),
        mode=args.mode,
        save_gif=args.gif
    )
    
    print("\n" + "="*70)
    print("[SUCCESS] Visualization complete!")
    print("="*70 + "\n")
