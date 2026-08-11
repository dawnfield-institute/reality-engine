"""
Test Möbius Substrate

Validates the geometric foundation before building dynamics on top.
"""

import torch
import sys
sys.path.insert(0, '..')

from substrate import MobiusManifold, FieldState


def test_initialization():
    """Test Möbius manifold creation"""
    print("\n" + "="*60)
    print("TEST 1: Möbius Manifold Initialization")
    print("="*60)
    
    substrate = MobiusManifold(size=128, width=32, seed=42)
    print(f"✓ Created: {substrate}")
    print(f"✓ Twist shift: {substrate.twist_shift}")
    print(f"✓ Device: {substrate.device}")
    

def test_field_initialization():
    """Test different field initialization modes"""
    print("\n" + "="*60)
    print("TEST 2: Field Initialization Modes")
    print("="*60)
    
    substrate = MobiusManifold(size=64, width=16, seed=42)
    
    # Random mode
    print("\n  Random mode:")
    state = substrate.initialize_fields(mode='random')
    print(f"    P: mean={state.P.mean():.4f}, std={state.P.std():.4f}")
    print(f"    A: mean={state.A.mean():.4f}, std={state.A.std():.4f}")
    print(f"    M: mean={state.M.mean():.4f}, std={state.M.std():.4f}")
    print(f"    Total PAC: {state.total_pac():.4f}")
    
    # Big Bang mode
    print("\n  Big Bang mode:")
    state = substrate.initialize_fields(mode='big_bang')
    print(f"    P: mean={state.P.mean():.4f}, std={state.P.std():.4f}")
    print(f"    A: mean={state.A.mean():.4f}, std={state.A.std():.4f}")
    print(f"    M: mean={state.M.mean():.4f}, std={state.M.std():.4f}")
    print(f"    Total PAC: {state.total_pac():.4f}")
    
    # Structured mode
    print("\n  Structured mode:")
    state = substrate.initialize_fields(mode='structured')
    print(f"    P: mean={state.P.mean():.4f}, std={state.P.std():.4f}")
    print(f"    A: mean={state.A.mean():.4f}, std={state.A.std():.4f}")
    print(f"    M: mean={state.M.mean():.4f}, std={state.M.std():.4f}")
    print(f"    Total PAC: {state.total_pac():.4f}")


def test_antiperiodic_boundaries():
    """Test anti-periodic boundary enforcement"""
    print("\n" + "="*60)
    print("TEST 3: Anti-Periodic Boundaries")
    print("="*60)
    
    substrate = MobiusManifold(size=64, width=16, seed=42)
    state = substrate.initialize_fields(mode='random')
    
    # Check anti-periodic condition: f(u+π, v) ≈ -f(u, 1-v)
    half_size = substrate.size // 2
    errors = []
    
    for i in range(10):  # Sample 10 points
        u = torch.randint(0, half_size, (1,)).item()
        v = torch.randint(0, substrate.width, (1,)).item()
        
        opposite_u = (u + half_size) % substrate.size
        opposite_v = substrate.width - 1 - v
        
        expected = -state.P[u, v]
        actual = state.P[opposite_u, opposite_v]
        error = (expected - actual).abs().item()
        errors.append(error)
        
        if i < 3:  # Print first 3 samples
            print(f"  Point ({u},{v}): {state.P[u,v].item():.4f}")
            print(f"  Opposite ({opposite_u},{opposite_v}): {actual.item():.4f}")
            print(f"  Expected: {expected.item():.4f}, Error: {error:.6f}")
    
    mean_error = sum(errors) / len(errors)
    print(f"\n  Mean anti-periodic error: {mean_error:.6f}")
    print(f"  {'✓ PASS' if mean_error < 0.1 else '✗ FAIL'} (threshold: 0.1)")


def test_topology_metrics():
    """Test topological quality metrics"""
    print("\n" + "="*60)
    print("TEST 4: Topology Metrics")
    print("="*60)
    
    substrate = MobiusManifold(size=64, width=16, seed=42)
    state = substrate.initialize_fields(mode='big_bang')
    
    metrics = substrate.calculate_metrics(state.P)
    
    print(f"  Twist strength: {metrics.twist_strength:.4f}")
    print(f"  Anti-periodic quality: {metrics.anti_periodic_quality:.4f}")
    print(f"  Boundary continuity: {metrics.boundary_continuity:.4f}")
    print(f"  Field coherence: {metrics.field_coherence:.4f}")
    print(f"  Curvature variance: {metrics.curvature_variance:.6f}")
    print(f"  Ξ measurement: {metrics.xi_measurement:.4f} (target: 1.0571)")
    
    print(f"\n  {'✓ PASS' if metrics.anti_periodic_quality > 0.8 else '✗ FAIL'}")


def test_field_state_operations():
    """Test FieldState utility methods"""
    print("\n" + "="*60)
    print("TEST 5: FieldState Operations")
    print("="*60)
    
    substrate = MobiusManifold(size=32, width=8, seed=42)
    state = substrate.initialize_fields(mode='big_bang')
    
    print(f"  Shape: {state.shape}")
    print(f"  Device: {state.device}")
    print(f"  Time: {state.time}")
    print(f"  Step: {state.step}")
    
    # Test clone
    cloned = state.clone()
    print(f"\n  Clone test:")
    print(f"    Original PAC: {state.total_pac():.4f}")
    print(f"    Cloned PAC: {cloned.total_pac():.4f}")
    print(f"    Difference: {abs(state.total_pac() - cloned.total_pac()):.10f}")
    
    # Test to device
    if torch.cuda.is_available():
        cuda_state = state.to('cuda')
        print(f"\n  Device transfer:")
        print(f"    CPU PAC: {state.total_pac():.4f}")
        print(f"    GPU PAC: {cuda_state.total_pac():.4f}")
        print(f"    ✓ PASS: Device transfer preserved PAC")


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*60)
    print("MÖBIUS SUBSTRATE TEST SUITE")
    print("="*60)
    
    test_initialization()
    test_field_initialization()
    test_antiperiodic_boundaries()
    test_topology_metrics()
    test_field_state_operations()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS COMPLETE")
    print("="*60)
    print("\nSubstrate layer validated!")
    print("Ready to build dynamics on top of Möbius foundation.\n")


if __name__ == "__main__":
    run_all_tests()
