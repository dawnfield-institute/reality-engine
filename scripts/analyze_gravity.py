"""
Analyze detected gravity and compare with classical gravity.

This script loads a physics discovery result and performs detailed
analysis on the recursive gravity that was detected, comparing it
to real-world gravitational behavior.
"""

import json
import numpy as np
import sys
from pathlib import Path

def analyze_gravity(discovery_file: str):
    """Analyze gravity from discovery results."""
    
    with open(discovery_file, 'r') as f:
        data = json.load(f)
    
    print("=" * 70)
    print("GRAVITY ANALYSIS")
    print("=" * 70)
    print()
    
    # Extract gravity data
    forces = data.get('laws', {}).get('forces', {})
    
    if 'recursive_gravity' not in forces and 'universal' not in forces:
        print("❌ No gravity detected in this simulation.")
        return
    
    # Get gravity law (could be under 'recursive_gravity' or 'universal')
    gravity = forces.get('recursive_gravity') or forces.get('universal')
    
    if not gravity:
        print("❌ No gravity data found.")
        return
    
    print("✅ GRAVITY DETECTED!")
    print()
    
    # Extract parameters
    params = gravity.get('parameters', {})
    G_eff = params.get('G_eff', 0)
    coalescence_rate = params.get('coalescence_rate', 0)
    power_law_exp = params.get('power_law_exponent', 0)
    info_corr = params.get('info_correlation', 0)
    initial_structures = params.get('initial_structures', 0)
    final_structures = params.get('final_structures', 0)
    confidence = gravity.get('confidence', 0)
    
    print("1. DETECTION METRICS")
    print("-" * 70)
    print(f"  Overall Confidence: {confidence*100:.1f}%")
    print(f"  Structure Coalescence: {coalescence_rate*100:.1f}%")
    print(f"    • {int(initial_structures)} → {final_structures:.1f} structures")
    print(f"    • Merged: {int(initial_structures - final_structures)} structures")
    print(f"  Info-density Correlation: {info_corr:.3f}")
    print(f"  Power-law Exponent: α = {power_law_exp:.3f}")
    print()
    
    # Compare with classical gravity
    print("2. COMPARISON WITH CLASSICAL GRAVITY")
    print("-" * 70)
    
    # Real gravity: G = 6.674×10⁻¹¹ m³/(kg·s²)
    # But our units are arbitrary simulation units
    print(f"  Effective G: {G_eff:.3f} (simulation units)")
    print(f"  Real G: 6.674e-11 m³/(kg·s²)")
    print()
    print("  Note: Direct comparison impossible - different unit systems!")
    print("  Instead, we analyze qualitative gravitational behavior...")
    print()
    
    # Analyze gravitational signatures
    print("3. GRAVITATIONAL SIGNATURES")
    print("-" * 70)
    
    # Signature 1: Structure coalescence
    print(f"  ✓ Mass Aggregation: {coalescence_rate*100:.1f}%")
    if coalescence_rate > 0.8:
        print(f"    → STRONG gravitational collapse (like galaxy formation)")
    elif coalescence_rate > 0.5:
        print(f"    → MODERATE gravitational attraction")
    else:
        print(f"    → WEAK gravitational effects")
    print()
    
    # Signature 2: Power-law scaling
    print(f"  ✓ Power-law Dynamics: N(t) ∝ t^{power_law_exp:.2f}")
    if -2.0 < power_law_exp < -1.0:
        print(f"    → Consistent with hierarchical gravitational collapse")
        print(f"    → Similar to dark matter halo formation (α ≈ -1.5)")
    elif -1.0 < power_law_exp < 0.0:
        print(f"    → Slower coalescence (weak gravity regime)")
    else:
        print(f"    → Unusual scaling (non-standard gravity?)")
    print()
    
    # Signature 3: Information-density correlation
    print(f"  ✓ Info-Density Correlation: {info_corr:.3f}")
    if info_corr > 0.3:
        print(f"    → High ρ_I regions drive strong mergers")
        print(f"    → Consistent with F ∝ ∇(ρ_I) theory")
    elif info_corr > 0.0:
        print(f"    → Weak info-gravity coupling")
    else:
        print(f"    → No clear info-density → gravity link")
    print()
    
    # Theoretical comparison
    print("4. GRAVITY MECHANISM COMPARISON")
    print("-" * 70)
    print("  Classical Gravity (Newton/Einstein):")
    print("    • Force: F = G·m₁·m₂/r²")
    print("    • Mechanism: Mass curves spacetime")
    print("    • Range: Infinite (1/r² falloff)")
    print("    • Attractive only")
    print()
    print("  Dawn Field Recursive Gravity:")
    print("    • Force: F = G·∇(ρ_I)")
    print("    • Mechanism: Information density gradients")
    print("    • Range: Emerges from field recursion")
    print("    • Driven by memory accumulation")
    print()
    
    # Key differences
    print("5. KEY DIFFERENCES")
    print("-" * 70)
    differences = [
        ("Origin", "Mass-based", "Information-based"),
        ("Mechanism", "Spacetime curvature", "Recursive field density"),
        ("Coupling", "Universal (all mass)", "Context-dependent (ρ_I)"),
        ("Scale dependence", "Scale-invariant 1/r²", "Power-law recursive"),
        ("Detection method", "Pairwise forces", "Collective coalescence")
    ]
    
    print(f"  {'Property':<20} {'Classical':<25} {'Dawn Field':<25}")
    print(f"  {'-'*20} {'-'*25} {'-'*25}")
    for prop, classical, dawn in differences:
        print(f"  {prop:<20} {classical:<25} {dawn:<25}")
    print()
    
    # Similarities
    print("6. SIMILARITIES WITH REAL GRAVITY")
    print("-" * 70)
    similarities = []
    
    if coalescence_rate > 0.7:
        similarities.append("✓ Strong attractive force causing structure aggregation")
    
    if -2.0 < power_law_exp < -1.0:
        similarities.append("✓ Hierarchical collapse (like cosmological structure formation)")
    
    if G_eff > 0:
        similarities.append("✓ Positive coupling constant (attractive)")
    
    if initial_structures > final_structures * 2:
        similarities.append("✓ Many-body gravitational collapse (galaxy clusters)")
    
    if similarities:
        for s in similarities:
            print(f"  {s}")
    else:
        print("  No strong similarities detected.")
    print()
    
    # Verdict
    print("7. VERDICT")
    print("-" * 70)
    
    if coalescence_rate > 0.8 and confidence > 0.5:
        verdict = "GRAVITY-LIKE BEHAVIOR CONFIRMED"
        explanation = (
            "The simulation exhibits strong gravitational behavior:\n"
            "  • Massive structure coalescence (many → few)\n"
            "  • Power-law collapse dynamics\n"
            "  • Attractive force dominates\n\n"
            "However, the MECHANISM differs from classical gravity:\n"
            "  • Emerges from recursive information density\n"
            "  • Not distance-based (1/r²)\n"
            "  • Context-dependent, not universal\n\n"
            "This is EMERGENT GRAVITY from information field dynamics,\n"
            "not fundamental gravitational force as in General Relativity."
        )
    elif coalescence_rate > 0.5:
        verdict = "WEAK GRAVITATIONAL EFFECTS"
        explanation = (
            "Some gravitational behavior present, but weaker than expected.\n"
            "The recursive information field shows attraction, but not\n"
            "as dominant as classical gravity in our universe."
        )
    else:
        verdict = "NON-GRAVITATIONAL DYNAMICS"
        explanation = (
            "Little evidence of gravity-like behavior.\n"
            "Structures may be interacting through other mechanisms."
        )
    
    print(f"  {verdict}")
    print()
    print(f"  {explanation}")
    print()
    
    # Physical interpretation
    print("8. PHYSICAL INTERPRETATION")
    print("-" * 70)
    print("  In Dawn Field Theory, gravity is NOT fundamental.")
    print("  Instead, it EMERGES from:")
    print()
    print("    1. Information accumulates in regions (high ρ_I)")
    print("    2. Information gradients create 'pull' toward high-ρ_I regions")
    print("    3. Recursive memory reinforces these gradients")
    print("    4. Structures coalesce → gravitational collapse")
    print()
    print("  This is analogous to:")
    print("    • Entropic gravity (Verlinde)")
    print("    • Thermodynamic gravity emergence")
    print("    • Holographic gravity from quantum entanglement")
    print()
    print("  The key insight: GRAVITY = INFORMATION GEOMETRY")
    print()
    
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Use most recent discovery file
        output_dir = Path("output/physics_discovery")
        if output_dir.exists():
            files = sorted(output_dir.glob("discovery_*.json"))
            if files:
                discovery_file = files[-1]
                print(f"Using most recent discovery: {discovery_file.name}\n")
            else:
                print("No discovery files found in output/physics_discovery/")
                sys.exit(1)
        else:
            print("Output directory not found!")
            sys.exit(1)
    else:
        discovery_file = sys.argv[1]
    
    analyze_gravity(str(discovery_file))
