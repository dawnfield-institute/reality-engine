"""
Quick field visualization utilities for Reality Engine.

Simple one-liners for common visualization tasks.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine

def quick_snapshot(engine: RealityEngine, title="Reality Engine State", save_path=None):
    """
    Quick 4-panel snapshot of current state.
    
    Usage:
        engine = RealityEngine(size=(64, 16))
        engine.initialize('big_bang')
        for _ in range(100):
            engine.step()
        quick_snapshot(engine, save_path='output.png')
    """
    state = engine.current_state
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # P field
    im1 = axes[0, 0].imshow(state.P.cpu().numpy(), cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title('Potential Field (P)', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # A field
    im2 = axes[0, 1].imshow(state.A.cpu().numpy(), cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('Actual Field (A)', fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # M field
    im3 = axes[1, 0].imshow(state.M.cpu().numpy(), cmap='viridis', aspect='auto')
    axes[1, 0].set_title('Memory Field (M) - Matter', fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # T field
    im4 = axes[1, 1].imshow(state.T.cpu().numpy(), cmap='hot', aspect='auto')
    axes[1, 1].set_title('Temperature Field (T)', fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1])
    
    fig.suptitle(
        f'{title}\nStep: {engine.step_count} | Time: {engine.time_elapsed:.2f} | '
        f'T_mean: {state.T.mean():.3f} | M_mean: {state.M.mean():.5f}',
        fontsize=13, fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Saved to: {save_path}")
    else:
        plt.show()
    
    return fig

def compare_states(engine: RealityEngine, steps_between=10, title="Field Evolution", save_path=None):
    """
    Compare fields at current state and after N steps.
    
    Shows before/after for all 4 fields.
    """
    # Capture initial state
    state_before = {
        'P': engine.current_state.P.cpu().numpy().copy(),
        'A': engine.current_state.A.cpu().numpy().copy(),
        'M': engine.current_state.M.cpu().numpy().copy(),
        'T': engine.current_state.T.cpu().numpy().copy(),
        'step': engine.step_count
    }
    
    # Evolve
    for _ in range(steps_between):
        engine.step()
    
    # Capture after state
    state_after = {
        'P': engine.current_state.P.cpu().numpy(),
        'A': engine.current_state.A.cpu().numpy(),
        'M': engine.current_state.M.cpu().numpy(),
        'T': engine.current_state.T.cpu().numpy(),
        'step': engine.step_count
    }
    
    # Plot comparison
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
    fields = [
        ('P', 'Potential', 'RdBu_r', True),
        ('A', 'Actual', 'RdBu_r', True),
        ('M', 'Memory', 'viridis', False),
        ('T', 'Temperature', 'hot', False)
    ]
    
    for i, (key, name, cmap, symmetric) in enumerate(fields):
        # Before
        data_before = state_before[key]
        if symmetric:
            vmax = max(abs(data_before.min()), abs(data_before.max()))
            vmin = -vmax
        else:
            vmin, vmax = data_before.min(), data_before.max()
        
        im1 = axes[i, 0].imshow(data_before, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'{name} - Step {state_before["step"]}', fontweight='bold')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # After
        data_after = state_after[key]
        if symmetric:
            vmax = max(abs(data_after.min()), abs(data_after.max()))
            vmin = -vmax
        else:
            vmin, vmax = data_after.min(), data_after.max()
        
        im2 = axes[i, 1].imshow(data_after, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'{name} - Step {state_after["step"]}', fontweight='bold')
        plt.colorbar(im2, ax=axes[i, 1])
    
    fig.suptitle(f'{title} ({steps_between} steps)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Saved comparison to: {save_path}")
    else:
        plt.show()
    
    return fig

def plot_field_statistics(engine: RealityEngine, save_path=None):
    """
    Plot statistical evolution from engine history.
    """
    if len(engine.history) < 2:
        print("[!] Need more history (run more steps)")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    steps = [s['step'] for s in engine.history]
    
    # Temperature
    T_mean = [s['T_mean'] for s in engine.history]
    T_std = [s['T_std'] for s in engine.history]
    axes[0, 0].plot(steps, T_mean, linewidth=2, color='red')
    axes[0, 0].fill_between(steps, 
                            [m-s for m,s in zip(T_mean, T_std)],
                            [m+s for m,s in zip(T_mean, T_std)],
                            alpha=0.3, color='red')
    axes[0, 0].set_title('Temperature Evolution', fontweight='bold')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('T')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Memory
    M_mean = [s['M_mean'] for s in engine.history]
    axes[0, 1].plot(steps, M_mean, linewidth=2, color='blue')
    axes[0, 1].set_title('Matter Crystallization', fontweight='bold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('M')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Actual field
    A_mean = [s['A_mean'] for s in engine.history]
    A_std = [s['A_std'] for s in engine.history]
    axes[0, 2].plot(steps, A_mean, linewidth=2, color='green')
    axes[0, 2].fill_between(steps,
                            [m-s for m,s in zip(A_mean, A_std)],
                            [m+s for m,s in zip(A_mean, A_std)],
                            alpha=0.3, color='green')
    axes[0, 2].set_title('Actual Field Evolution', fontweight='bold')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('A')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Disequilibrium
    diseq = [s['disequilibrium'] for s in engine.history]
    axes[1, 0].plot(steps, diseq, linewidth=2, color='purple')
    axes[1, 0].set_title('Disequilibrium', fontweight='bold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('|P - A|')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Heat generation
    total_heat = [s['total_heat'] for s in engine.history]
    axes[1, 1].plot(steps, total_heat, linewidth=2, color='orange')
    axes[1, 1].set_title('Cumulative Heat Generated', fontweight='bold')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Total Heat')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Collapse events
    collapses = [s['collapse_events'] for s in engine.history]
    axes[1, 2].plot(steps, collapses, linewidth=2, color='red')
    axes[1, 2].set_title('SEC Collapse Events', fontweight='bold')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Cumulative Collapses')
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.suptitle(f'Reality Engine Statistics (Steps: {len(steps)})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Saved statistics to: {save_path}")
    else:
        plt.show()
    
    return fig

if __name__ == '__main__':
    print("="*70)
    print("QUICK VISUALIZATION DEMO")
    print("="*70)
    
    # Create engine and evolve
    print("\n[*] Creating universe...")
    engine = RealityEngine(size=(64, 16), dt=0.1)
    engine.initialize('big_bang')
    
    # Evolve for a bit
    print("[*] Evolving 100 steps...")
    for i in range(100):
        engine.step()
        if i % 20 == 0:
            print(f"    Step {i}/100")
    
    # Snapshot
    print("\n[*] Creating snapshot...")
    quick_snapshot(engine, save_path=repo_root / 'examples' / 'quick_snapshot.png')
    
    # Statistics
    print("[*] Plotting statistics...")
    plot_field_statistics(engine, save_path=repo_root / 'examples' / 'field_statistics.png')
    
    # Comparison
    print("[*] Creating before/after comparison (evolving 25 more steps)...")
    compare_states(engine, steps_between=25, 
                  save_path=repo_root / 'examples' / 'field_comparison.png')
    
    print("\n" + "="*70)
    print("[SUCCESS] All visualizations created!")
    print("="*70 + "\n")
