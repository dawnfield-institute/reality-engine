"""
Simple smoke test for Reality Engine v2 integration.

Tests that all core components work together:
- Substrate (MobiusManifold)
- Conservation (SEC operator)
- Dynamics (Confluence operator)
- Core (RealityEngine interface)
"""

import sys
sys.path.insert(0, '.')

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    from substrate.mobius_manifold import MobiusManifold
    from substrate.field_types import FieldState
    from substrate.constants import XI, LAMBDA
    from conservation.sec_operator import SymbolicEntropyCollapse
    from conservation.thermodynamic_pac import ThermodynamicPAC
    from dynamics.confluence import MobiusConfluence
    from dynamics.time_emergence import TimeEmergence
    from core.reality_engine import RealityEngine, create_reality
    
    print("✓ All imports successful")
    return True


def test_reality_engine_basic():
    """Test basic RealityEngine functionality."""
    print("\nTesting RealityEngine...")
    
    from core.reality_engine import RealityEngine
    
    # Create engine
    engine = RealityEngine(size=(32, 8), device='cpu')
    print(f"✓ Engine created: {engine}")
    
    # Initialize
    engine.initialize('big_bang')
    print(f"✓ Engine initialized")
    
    # Run a few steps
    for i in range(5):
        state = engine.step()
        print(f"  Step {state['step']}: T_mean={state['T_mean']:.2f}, A_mean={state['A_mean']:.4f}")
    
    print(f"✓ Evolution successful - {engine.step_count} steps completed")
    
    return True


def test_individual_components():
    """Test individual components work."""
    print("\nTesting individual components...")
    
    import torch
    from substrate.mobius_manifold import MobiusManifold
    from conservation.sec_operator import SymbolicEntropyCollapse
    from dynamics.confluence import MobiusConfluence
    
    # Substrate
    substrate = MobiusManifold(size=32, width=8, device='cpu')
    state = substrate.initialize_fields(mode='big_bang')
    print(f"✓ Substrate: P.shape={state.P.shape}, T_mean={state.T.mean():.2f}")
    
    # SEC
    sec = SymbolicEntropyCollapse(device='cpu')
    A_new, heat = sec.evolve(state.A, state.P, state.T, dt=0.01)
    print(f"✓ SEC: heat_generated={heat:.6f}")
    
    # Confluence
    confluence = MobiusConfluence(size=(32, 8), device='cpu')
    P_new = confluence.step(A_new)
    print(f"✓ Confluence: P_new.shape={P_new.shape}")
    
    return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Reality Engine v2 - Smoke Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Component Test", test_individual_components),
        ("RealityEngine Test", test_reality_engine_basic),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"{name}")
            print(f"{'='*60}")
            if test_func():
                passed += 1
                print(f"\n✅ {name} PASSED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
