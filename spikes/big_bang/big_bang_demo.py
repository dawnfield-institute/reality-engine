"""
Big Bang Demonstration

Shows the full cycle of universe evolution:
1. Hot dense start (maximum disequilibrium, pure entropy)
2. Rapid cooling (thermal radiation)
3. Matter formation (memory crystallization from structure)
4. Structure emergence (patterns form from SEC collapses)
5. Approach to equilibrium (heat death)

This demonstrates that time, matter, and structure all emerge
from pure thermodynamic principles - no magic needed!
"""
import sys
from pathlib import Path

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine
import matplotlib.pyplot as plt
import numpy as np

def big_bang_demo(steps=200, size=(64, 16)):
    """
    Run Big Bang simulation and visualize evolution.
    
    Args:
        steps: Number of evolution steps
        size: Field dimensions (nu, nv)
    """
    print("="*60)
    print("BIG BANG DEMONSTRATION")
    print("="*60)
    print(f"\nUniverse size: {size[0]} × {size[1]} cells")
    print(f"Evolution steps: {steps}")
    print("\nPhysics enabled:")
    print("  ✓ Thermodynamic SEC (energy minimization)")
    print("  ✓ Möbius Confluence (geometric time)")
    print("  ✓ Heat generation & diffusion")
    print("  ✓ Memory crystallization")
    print("  ✓ Time emergence from disequilibrium")
    print("\n" + "="*60)
    
    # Create engine
    engine = RealityEngine(size=size, dt=0.1, device='cpu')
    
    # Big Bang initialization (hot dense start)
    print("\n🌌 T=0: BIG BANG - Maximum disequilibrium, pure entropy")
    engine.initialize(mode='big_bang')
    
    # Track evolution
    temperatures = []
    memory_levels = []
    disequilibria = []
    structure_measures = []
    collapse_events = []
    
    print("\n⏳ Evolving universe...")
    print(f"{'Step':>5} {'Time':>8} {'T_mean':>8} {'M_mean':>8} {'Diseq':>8} {'Struct':>8} {'Events':>7}")
    print("-" * 60)
    
    for i, state in enumerate(engine.evolve(steps=steps)):
        T_mean = state['T_mean']
        M_mean = state['M_mean']
        diseq = state.get('time_metrics', {}).get('disequilibrium', state['disequilibrium'])
        
        # Structure measure: low variance = high structure
        A_var = state['A_std'] ** 2
        structure = 1.0 / (1.0 + A_var)
        
        temperatures.append(T_mean)
        memory_levels.append(M_mean)
        disequilibria.append(diseq)
        structure_measures.append(structure)
        collapse_events.append(state['collapse_events'])
        
        if i % (steps // 10) == 0:
            time_val = state['time']
            events = state['collapse_events']
            print(f"{i:5d} {time_val:8.2f} {T_mean:8.2f} {M_mean:8.5f} {diseq:8.5f} {structure:8.5f} {events:7d}")
    
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
    
    # Analysis
    T_initial, T_final = temperatures[0], temperatures[-1]
    M_initial, M_final = memory_levels[0], memory_levels[-1]
    D_initial, D_final = disequilibria[0], disequilibria[-1]
    
    cooling_factor = T_initial / T_final if T_final > 0 else float('inf')
    matter_formed = M_final - M_initial
    equilibration = (D_initial - D_final) / D_initial if D_initial > 0 else 0
    
    print(f"\nThermodynamics:")
    print(f"  Temperature: {T_initial:.2f} → {T_final:.2f} (cooled {cooling_factor:.1f}×)")
    print(f"  Memory (matter): {M_initial:.5f} → {M_final:.5f} (+{matter_formed:.5f})")
    print(f"  Disequilibrium: {D_initial:.5f} → {D_final:.5f} ({equilibration*100:.1f}% toward equilibrium)")
    print(f"  Total collapses: {collapse_events[-1]}")
    
    # Identify epochs
    print(f"\nEpochs detected:")
    
    # Hot epoch: T > T_initial / 2
    hot_epoch_end = next((i for i, T in enumerate(temperatures) if T < T_initial / 2), len(temperatures))
    print(f"  🔥 Hot Epoch: steps 0-{hot_epoch_end} (rapid cooling)")
    
    # Matter formation: when M grows fastest
    M_growth_rates = [memory_levels[i+1] - memory_levels[i] for i in range(len(memory_levels)-1)]
    max_growth_idx = np.argmax(M_growth_rates) if M_growth_rates else 0
    print(f"  ⚛️  Matter Formation Peak: step ~{max_growth_idx}")
    
    # Structure emergence: when variance decreases
    A_variances = [1.0 / s - 1 for s in structure_measures]
    min_var_idx = np.argmin(A_variances) if A_variances else len(A_variances) - 1
    print(f"  🌟 Structure Emergence: step ~{min_var_idx}")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    steps_range = range(len(temperatures))
    
    # Temperature evolution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps_range, temperatures, linewidth=2, color='red')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Temperature')
    ax1.set_title('🔥 Cosmic Cooling')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(0, hot_epoch_end, alpha=0.2, color='red', label='Hot Epoch')
    ax1.legend()
    
    # Memory (matter) formation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps_range, memory_levels, linewidth=2, color='blue')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Memory Field')
    ax2.set_title('⚛️ Matter Crystallization')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(max_growth_idx, color='green', linestyle='--', alpha=0.5, label='Peak Formation')
    ax2.legend()
    
    # Disequilibrium (approach to equilibrium)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps_range, disequilibria, linewidth=2, color='purple')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Disequilibrium')
    ax3.set_title('⚖️ Approach to Equilibrium')
    ax3.grid(True, alpha=0.3)
    
    # Structure formation
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(steps_range, structure_measures, linewidth=2, color='green')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Structure (1 / (1 + var(A)))')
    ax4.set_title('🌟 Pattern Emergence')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(min_var_idx, color='red', linestyle='--', alpha=0.5, label='Min Variance')
    ax4.legend()
    
    # Collapse events (cumulative)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(steps_range, collapse_events, linewidth=2, color='orange')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Cumulative Collapses')
    ax5.set_title('💥 SEC Collapse Events')
    ax5.grid(True, alpha=0.3)
    
    # Phase diagram: Temperature vs Disequilibrium
    ax6 = fig.add_subplot(gs[2, 1])
    scatter = ax6.scatter(disequilibria, temperatures, c=steps_range, cmap='viridis', s=20, alpha=0.6)
    ax6.set_xlabel('Disequilibrium')
    ax6.set_ylabel('Temperature')
    ax6.set_title('📊 Phase Diagram (color = time)')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Step')
    
    # Add Big Bang marker
    ax6.scatter([disequilibria[0]], [temperatures[0]], color='red', s=200, 
                marker='*', edgecolors='yellow', linewidths=2, label='Big Bang', zorder=10)
    ax6.legend()
    
    plt.suptitle('Big Bang → Structure Formation → Equilibrium', fontsize=16, fontweight='bold')
    
    save_path = repo_root / 'examples' / 'big_bang_evolution.png'
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {save_path}")
    
    return {
        'cooling_factor': cooling_factor,
        'matter_formed': matter_formed,
        'equilibration': equilibration,
        'total_collapses': collapse_events[-1]
    }

if __name__ == '__main__':
    results = big_bang_demo(steps=200, size=(64, 16))
    
    print("\n" + "="*60)
    print("🎉 Big Bang demonstration complete!")
    print("="*60)
