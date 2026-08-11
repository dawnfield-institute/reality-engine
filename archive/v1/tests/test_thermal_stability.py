"""
Test thermal stability over extended evolution.
"""
import sys
from pathlib import Path

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine
import matplotlib.pyplot as plt

def test_thermal_stability(steps=100):
    """Run engine for many steps and plot temperature evolution."""
    
    engine = RealityEngine(size=(32, 8), dt=0.1, device='cpu')
    engine.initialize(mode='random')
    
    temperatures = []
    heat_generated_list = []
    
    print(f"Running {steps} steps...")
    for i, state_dict in enumerate(engine.evolve(steps=steps)):
        T_mean = state_dict['T_mean']
        temperatures.append(T_mean)
        
        if i % 10 == 0:
            print(f"  Step {i:3d}: T_mean={T_mean:.2f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(temperatures)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Mean Temperature')
    ax1.set_title('Temperature Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Check if temperature stabilizes
    final_100 = temperatures[-100:] if len(temperatures) >= 100 else temperatures[-len(temperatures)//2:]
    mean_final = sum(final_100) / len(final_100)
    std_final = (sum((t - mean_final)**2 for t in final_100) / len(final_100))**0.5
    cv = std_final / mean_final if mean_final > 0 else float('inf')
    
    ax2.text(0.1, 0.7, f"Final Temperature Stats:", fontsize=12, weight='bold')
    ax2.text(0.1, 0.5, f"  Mean: {mean_final:.2f}", fontsize=11)
    ax2.text(0.1, 0.3, f"  Std Dev: {std_final:.2f}", fontsize=11)
    ax2.text(0.1, 0.1, f"  Coef. of Variation: {cv:.3f}", fontsize=11)
    ax2.axis('off')
    
    stability_verdict = "STABLE" if cv < 0.1 else "UNSTABLE"
    color = 'green' if cv < 0.1 else 'red'
    ax2.text(0.1, -0.1, f"Verdict: {stability_verdict}", fontsize=14, weight='bold', color=color)
    
    plt.tight_layout()
    plt.savefig(repo_root / 'tests' / 'thermal_stability.png', dpi=150)
    print(f"\nPlot saved to: {repo_root / 'tests' / 'thermal_stability.png'}")
    
    return cv < 0.1, mean_final, cv

if __name__ == '__main__':
    stable, T_final, cv = test_thermal_stability(steps=100)
    
    if stable:
        print(f"\n✅ STABLE: Temperature converged to {T_final:.2f} (CV={cv:.3f})")
    else:
        print(f"\n❌ UNSTABLE: Temperature not stable (CV={cv:.3f})")
