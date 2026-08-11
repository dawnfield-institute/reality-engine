"""
Test memory accumulation from structure formation.
"""
import sys
from pathlib import Path

# Add parent directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine
import matplotlib.pyplot as plt

def test_memory_accumulation(steps=50):
    """Run engine and plot memory field evolution."""
    
    engine = RealityEngine(size=(32, 8), dt=0.1, device='cpu')
    engine.initialize(mode='random')
    
    memory_means = []
    memory_maxes = []
    structure_measures = []
    
    print(f"Running {steps} steps to observe memory accumulation...")
    for i, state_dict in enumerate(engine.evolve(steps=steps)):
        M_mean = state_dict['M_mean']
        M_max = state_dict['M_max']
        
        # Compute structure measure (low variance in A = high structure)
        A_variance = state_dict['A_std'] ** 2
        structure = 1.0 / (1.0 + A_variance)  # High when A is stable
        
        memory_means.append(M_mean)
        memory_maxes.append(M_max)
        structure_measures.append(structure)
        
        if i % 10 == 0:
            print(f"  Step {i:3d}: M_mean={M_mean:.4f}, M_max={M_max:.4f}, structure={structure:.4f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    steps_range = range(len(memory_means))
    
    ax1.plot(steps_range, memory_means, label='Mean Memory', linewidth=2)
    ax1.plot(steps_range, memory_maxes, label='Max Memory', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Memory Field')
    ax1.set_title('Memory Accumulation Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps_range, structure_measures, label='Structure Measure', color='green', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Structure (1 / (1 + var(A)))')
    ax2.set_title('Field Structure Formation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(repo_root / 'tests' / 'memory_accumulation.png', dpi=150)
    print(f"\nPlot saved to: {repo_root / 'tests' / 'memory_accumulation.png'}")
    
    # Analysis
    final_M = memory_means[-1]
    M_growth = memory_means[-1] - memory_means[0]
    M_growth_rate = M_growth / steps
    
    print(f"\nMemory Analysis:")
    print(f"  Initial M_mean: {memory_means[0]:.6f}")
    print(f"  Final M_mean: {final_M:.6f}")
    print(f"  Total growth: {M_growth:.6f}")
    print(f"  Growth rate: {M_growth_rate:.6f} per step")
    
    if M_growth > 0.001:
        print(f"\n✅ Memory is accumulating (growth={M_growth:.6f})")
    else:
        print(f"\n❌ Memory not accumulating significantly (growth={M_growth:.6f})")
    
    return M_growth > 0.001

if __name__ == '__main__':
    success = test_memory_accumulation(steps=50)
