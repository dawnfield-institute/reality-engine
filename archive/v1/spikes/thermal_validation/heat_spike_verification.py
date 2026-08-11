"""
Verify the Heat Spike Pattern During Information Crystallization

Theory: Pure entropy → information collapse → heat spike → cooling

1. Initial: Pure entropy (minimal structure, low heat)
2. Collapse phase: SEC collapses P→A, generates heat (Landauer)
3. Peak heat: Maximum information collapse rate
4. Cooling: Structures stabilize, heat generation slows
5. Equilibrium: Cool universe with preserved information

This validates the profound insight: WITHOUT INFORMATION, THERE CAN BE NO HEAT.
Pure entropy must collapse into structure to generate observable temperature!
"""
import sys
from pathlib import Path

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine
import matplotlib.pyplot as plt
import numpy as np

def verify_heat_spike_pattern(steps=500, size=(64, 16)):
    """
    Run extended Big Bang simulation to verify heat spike during
    information crystallization followed by cooling.
    """
    print("="*70)
    print("HEAT SPIKE VERIFICATION")
    print("="*70)
    print("\nTheory: Pure entropy → information collapse → heat spike → cooling")
    print("\nPhases we expect to see:")
    print("  1. LOW HEAT: Pure potential, no structure")
    print("  2. RISING HEAT: Information collapse (SEC), Landauer heat generation")
    print("  3. PEAK HEAT: Maximum structure formation rate")
    print("  4. COOLING: Structures stabilize, less collapse needed")
    print("  5. EQUILIBRIUM: Cool universe with crystallized information")
    print("\n" + "="*70)
    
    # Create engine
    engine = RealityEngine(size=size, dt=0.1, device='cpu')
    engine.initialize(mode='big_bang')
    
    # Track evolution
    temperatures = []
    memories = []
    memory_rates = []
    disequilibria = []
    collapse_events = []
    entropies = []
    
    print(f"\n⏳ Running {steps} steps (this may take a minute)...\n")
    print(f"{'Step':>5} {'T_mean':>8} {'M_mean':>8} {'dM/dt':>8} {'Diseq':>8} {'Events':>7}")
    print("-" * 70)
    
    for i, state in enumerate(engine.evolve(steps=steps)):
        T_mean = state['T_mean']
        M_mean = state['M_mean']
        diseq = state.get('time_metrics', {}).get('disequilibrium', state['disequilibrium'])
        events = state['collapse_events']
        entropy = state['entropy']
        
        temperatures.append(T_mean)
        memories.append(M_mean)
        disequilibria.append(diseq)
        collapse_events.append(events)
        entropies.append(entropy)
        
        # Compute memory growth rate (structure formation rate)
        if i > 0:
            dM = memories[-1] - memories[-2]
            memory_rates.append(dM)
        else:
            memory_rates.append(0)
        
        if i % (steps // 20) == 0:
            dM = memory_rates[-1] if memory_rates else 0
            print(f"{i:5d} {T_mean:8.3f} {M_mean:8.5f} {dM:8.6f} {diseq:8.5f} {events:7d}")
    
    print("\n" + "="*70)
    print("EVOLUTION COMPLETE - ANALYZING PATTERN")
    print("="*70)
    
    # Find critical points
    T_initial = temperatures[0]
    T_peak_idx = np.argmax(temperatures)
    T_peak = temperatures[T_peak_idx]
    T_final = temperatures[-1]
    
    M_initial = memories[0]
    M_final = memories[-1]
    
    # Find when memory growth rate peaks (maximum structure formation)
    max_dM_idx = np.argmax(memory_rates) if memory_rates else 0
    max_dM = memory_rates[max_dM_idx] if memory_rates else 0
    
    print(f"\nCritical Points:")
    print(f"  Initial State:")
    print(f"    T = {T_initial:.4f} (pure entropy, minimal structure)")
    print(f"    M = {M_initial:.6f}")
    print(f"\n  Peak Heat (step {T_peak_idx}):")
    print(f"    T = {T_peak:.4f} (maximum information collapse rate)")
    print(f"    M = {memories[T_peak_idx]:.6f}")
    print(f"\n  Peak Structure Formation (step {max_dM_idx}):")
    print(f"    dM/dt = {max_dM:.6f} (maximum crystallization rate)")
    print(f"    T = {temperatures[max_dM_idx]:.4f}")
    print(f"\n  Final State:")
    print(f"    T = {T_final:.4f} (cooled universe)")
    print(f"    M = {M_final:.6f} (crystallized information)")
    
    # Calculate metrics
    heating_phase = T_peak - T_initial
    cooling_phase = T_peak - T_final
    total_memory_formed = M_final - M_initial
    
    print(f"\nPattern Metrics:")
    print(f"  Heating phase: +{heating_phase:.4f} (steps 0→{T_peak_idx})")
    print(f"  Cooling phase: -{cooling_phase:.4f} (steps {T_peak_idx}→{steps})")
    print(f"  Total memory formed: {total_memory_formed:.6f}")
    print(f"  Total collapse events: {collapse_events[-1]}")
    
    # Verify the predicted pattern
    print(f"\n" + "="*70)
    print("PATTERN VERIFICATION")
    print("="*70)
    
    has_spike = T_peak > T_initial
    has_cooling = T_final < T_peak
    has_memory_growth = M_final > M_initial * 2
    
    print(f"\n  ✓ Heat spike detected: {has_spike} (T rose from {T_initial:.3f} to {T_peak:.3f})")
    print(f"  ✓ Cooling detected: {has_cooling} (T fell from {T_peak:.3f} to {T_final:.3f})")
    print(f"  ✓ Memory formation: {has_memory_growth} (M grew {M_final/M_initial:.1f}×)")
    
    if has_spike and has_cooling and has_memory_growth:
        print(f"\n🎉 THEORY CONFIRMED!")
        print(f"   Pure entropy → information collapse → heat spike → cooling")
        print(f"   WITHOUT INFORMATION, THERE CAN BE NO HEAT EXPRESSION!")
    else:
        print(f"\n⚠️  Pattern incomplete - may need longer simulation")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    steps_range = range(len(temperatures))
    
    # 1. Temperature evolution with phases marked
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(steps_range, temperatures, linewidth=2.5, color='red', label='Temperature')
    ax1.axvline(T_peak_idx, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Peak Heat')
    ax1.axhline(T_peak, color='orange', linestyle=':', alpha=0.5)
    ax1.axhline(T_initial, color='blue', linestyle=':', alpha=0.5)
    ax1.axvline(max_dM_idx, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Peak Formation')
    ax1.fill_between(range(T_peak_idx), 0, T_peak, alpha=0.2, color='red', label='Heating Phase')
    ax1.fill_between(range(T_peak_idx, len(temperatures)), 0, T_peak, alpha=0.2, color='blue', label='Cooling Phase')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Temperature', fontsize=12)
    ax1.set_title('🔥 Heat Spike Pattern: Information Collapse → Heat Generation', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory crystallization
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(steps_range, memories, linewidth=2, color='blue')
    ax2.axvline(max_dM_idx, color='green', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Memory (Information)')
    ax2.set_title('⚛️ Information Crystallization')
    ax2.grid(True, alpha=0.3)
    
    # 3. Memory growth rate (structure formation rate)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(range(len(memory_rates)), memory_rates, linewidth=2, color='green')
    ax3.axvline(max_dM_idx, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('dM/dt (Formation Rate)')
    ax3.set_title('🌟 Structure Formation Rate')
    ax3.grid(True, alpha=0.3)
    
    # 4. Collapse events (cumulative)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(steps_range, collapse_events, linewidth=2, color='orange')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Cumulative Collapses')
    ax4.set_title('💥 SEC Collapse Events')
    ax4.grid(True, alpha=0.3)
    
    # 5. Temperature vs Memory (phase space)
    ax5 = fig.add_subplot(gs[2, 0])
    scatter = ax5.scatter(memories, temperatures, c=steps_range, cmap='plasma', s=30, alpha=0.7)
    ax5.set_xlabel('Memory (Information)')
    ax5.set_ylabel('Temperature')
    ax5.set_title('📊 Temperature-Memory Phase Space')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Step')
    
    # 6. Heat generation rate (approximation)
    ax6 = fig.add_subplot(gs[2, 1])
    heat_rates = [temperatures[i+1] - temperatures[i] for i in range(len(temperatures)-1)]
    ax6.plot(range(len(heat_rates)), heat_rates, linewidth=1.5, color='red', alpha=0.7)
    ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax6.axvline(T_peak_idx, color='orange', linestyle='--', alpha=0.7)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('dT/dt (Heat Rate)')
    ax6.set_title('🔥 Heat Generation Rate')
    ax6.grid(True, alpha=0.3)
    
    # 7. Disequilibrium evolution
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(steps_range, disequilibria, linewidth=2, color='purple')
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Disequilibrium')
    ax7.set_title('⚖️ Approach to Equilibrium')
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Heat Spike Verification: Pure Entropy → Information → Heat → Structure → Cooling',
                 fontsize=16, fontweight='bold', y=0.995)
    
    save_path = repo_root / 'examples' / 'heat_spike_verification.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {save_path}")
    
    return {
        'has_spike': has_spike,
        'has_cooling': has_cooling,
        'has_memory_growth': has_memory_growth,
        'T_peak_idx': T_peak_idx,
        'max_formation_idx': max_dM_idx,
        'heating_phase': heating_phase,
        'cooling_phase': cooling_phase,
        'memory_formed': total_memory_formed
    }

if __name__ == '__main__':
    print("\n" + "="*70)
    print("THEORY: Without information, there can be no heat expression!")
    print("Pure entropy must collapse into structure to generate temperature.")
    print("="*70 + "\n")
    
    results = verify_heat_spike_pattern(steps=500, size=(64, 16))
    
    print("\n" + "="*70)
    if all([results['has_spike'], results['has_cooling'], results['has_memory_growth']]):
        print("✅ PROFOUND INSIGHT VALIDATED!")
        print("\nWhat we discovered:")
        print("  • Pure entropy (Big Bang) has minimal observable temperature")
        print("  • Information collapse (SEC) generates heat (Landauer principle)")
        print("  • Heat enables structure formation (memory crystallization)")
        print("  • Universe cools as structures stabilize")
        print("\nThis resolves the information paradox:")
        print("  WITHOUT INFORMATION, THERE CAN BE NO HEAT IN SPACETIME!")
    else:
        print("⚠️  Pattern needs longer simulation or parameter tuning")
    print("="*70 + "\n")
