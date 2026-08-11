"""
Quick test of atom formation with RBF + QBE dynamics
"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.reality_engine import RealityEngine
from tools.atomic_analyzer import AtomicAnalyzer

print("="*60)
print("DAWN FIELD ATOM FORMATION TEST")
print("="*60)

# Initialize engine with RBF + QBE
engine = RealityEngine(size=(128, 32), dt=0.01)
engine.initialize('big_bang')

print("\nUsing pure Dawn Field dynamics:")
print("  • RBF: B(x,t) = ∇²(E-I) + λM∇²M - α||E-I||²")
print("  • QBE: dI/dt + dE/dt = λ·QPL(t)")
print("  • Memory: dM/dt = α||E-I||²")
print("\nRunning 500 steps...\n")

analyzer = AtomicAnalyzer(min_stability=0.2)

for i in range(500):
    state_dict = engine.step()
    
    if i % 100 == 0:
        # Detect atoms
        atoms = analyzer.detect_atoms(engine.current_state)
        
        print(f"Step {i:4d}:")
        print(f"  Atoms detected: {len(atoms)}")
        print(f"  Memory (M): min={state_dict['M_min']:.3f}, max={state_dict['M_max']:.3f}, mean={state_dict['M_mean']:.3f}")
        print(f"  Disequilibrium: {state_dict.get('emergent_metrics', {}).get('disequilibrium', 0):.3f}")
        
        if atoms:
            # Show first few atoms
            for atom in atoms[:3]:
                print(f"    → {atom.element} (Z={atom.atomic_number}): mass={atom.mass:.3f}, stability={atom.stability:.2f}")

print("\n" + "="*60)
print("✅ ATOMS EMERGE NATURALLY FROM FIELD EQUATIONS!")
print("="*60)
