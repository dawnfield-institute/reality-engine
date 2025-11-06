"""
Test Thermodynamic Components - Simple Version

Tests basic functionality without unicode issues
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from substrate.mobius_manifold import MobiusManifold
from substrate.field_types import FieldState
from conservation.thermodynamic_pac import ThermodynamicPAC
from dynamics.time_emergence import TimeEmergence


def test_field_state_thermodynamic():
    """Test that FieldState has thermodynamic methods"""
    print("\n" + "="*70)
    print("TEST: FieldState Thermodynamic Methods")
    print("="*70)
    
    substrate = MobiusManifold(size=16, width=8)
    state = substrate.initialize_fields(mode='random')
    
    print(f"Potential sum: {state.potential.sum().item():.4f}")
    print(f"Actual sum: {state.actual.sum().item():.4f}")
    print(f"Memory sum: {state.memory.sum().item():.4f}")
    print(f"Temperature mean: {state.temperature.mean().item():.4f}")
    
    # Test thermodynamic methods
    print(f"\nThermodynamic quantities:")
    print(f"  Energy: {state.energy():.4f}")
    print(f"  Entropy: {state.entropy():.4f}")
    print(f"  Free energy: {state.free_energy():.4f}")
    print(f"  Disequilibrium: {state.disequilibrium():.4f}")
    print(f"  Thermal variance: {state.thermal_variance():.4f}")
    
    print("\n[PASS] FieldState has all thermodynamic methods!")
    return True


def test_pac_enforcement():
    """Test basic PAC enforcement"""
    print("\n" + "="*70)
    print("TEST: PAC Enforcement")
    print("="*70)
    
    substrate = MobiusManifold(size=16, width=8)
    state = substrate.initialize_fields(mode='random')
    thermo_pac = ThermodynamicPAC(tolerance=1e-12)
    
    # Introduce violation
    initial_pac = state.total_pac()
    state.potential *= 1.1
    violated_pac = state.total_pac()
    
    print(f"Initial PAC: {initial_pac:.6f}")
    print(f"After violation: {violated_pac:.6f}")
    print(f"Error: {abs(violated_pac - initial_pac):.6e}")
    
    # Enforce
    state, metrics = thermo_pac.enforce(state)
    final_pac = state.total_pac()
    
    print(f"\nAfter enforcement:")
    print(f"  Final PAC: {final_pac:.6f}")
    print(f"  PAC error: {metrics.pac_error:.6e}")
    print(f"  Landauer cost: {metrics.landauer_cost:.6e}")
    
    assert metrics.pac_error < 1e-10, f"PAC error too large: {metrics.pac_error}"
    
    print("\n[PASS] PAC enforcement works!")
    return True


def test_second_law():
    """Test that entropy increases"""
    print("\n" + "="*70)
    print("TEST: Second Law (Entropy Never Decreases)")
    print("="*70)
    
    substrate = MobiusManifold(size=16, width=8)
    state = substrate.initialize_fields(mode='random')
    thermo_pac = ThermodynamicPAC()
    
    initial_entropy = state.entropy()
    print(f"Initial entropy: {initial_entropy:.6f}")
    
    # Run 50 steps
    entropy_violations = 0
    for step in range(50):
        # Simple evolution
        state.actual += 0.01 * (state.potential - state.actual)
        state, metrics = thermo_pac.enforce(state)
        
        if metrics.entropy < initial_entropy:
            entropy_violations += 1
        initial_entropy = metrics.entropy
    
    print(f"Final entropy: {initial_entropy:.6f}")
    print(f"Violations: {entropy_violations}/50")
    
    assert entropy_violations == 0, "2nd law violated!"
    
    print("\n[PASS] 2nd law holds - entropy never decreased!")
    return True


def test_time_emergence():
    """Test basic time emergence"""
    print("\n" + "="*70)
    print("TEST: Time Emergence")
    print("="*70)
    
    substrate = MobiusManifold(size=16, width=8)
    state = substrate.initialize_fields(mode='random')
    time_engine = TimeEmergence()
    
    # Compute time rate
    time_rate, metrics = time_engine.compute_time_rate(state)
    
    print(f"Global time: {metrics.global_time:.4f}")
    print(f"Disequilibrium: {metrics.disequilibrium:.6f}")
    print(f"Interaction density: {metrics.interaction_density:.6f}")
    print(f"Time dilation factor: {metrics.time_dilation_factor:.6f}")
    print(f"c_effective: {metrics.c_effective:.2e}")
    
    assert time_rate is not None
    assert metrics.global_time >= 0
    
    print("\n[PASS] Time emergence computed!")
    return True


def test_big_bang_init():
    """Test Big Bang initialization"""
    print("\n" + "="*70)
    print("TEST: Big Bang Initialization")
    print("="*70)
    
    substrate = MobiusManifold(size=16, width=8)
    state = substrate.initialize_fields(mode='random')
    time_engine = TimeEmergence()
    
    # Big Bang
    state = time_engine.big_bang_initialization(state)
    
    print(f"After Big Bang:")
    print(f"  Disequilibrium: {state.disequilibrium():.6f}")
    print(f"  Matter: {state.matter():.6f}")
    print(f"  Temperature mean: {state.temperature.mean().item():.6f}")
    print(f"  Entropy: {state.entropy():.6f}")
    
    assert state.matter() == 0.0 or state.matter() < 0.1, "Matter should be near zero initially"
    assert state.temperature.mean().item() > 1.0, "Temperature should be high"
    
    print("\n[PASS] Big Bang initialization correct!")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("REALITY ENGINE V2 - THERMODYNAMIC TESTS")
    print("="*70)
    
    tests = [
        test_field_state_thermodynamic,
        test_pac_enforcement,
        test_second_law,
        test_time_emergence,
        test_big_bang_init
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n*** ALL TESTS PASSED! ***")
        print("\nThermodynamic-information duality validated!")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
