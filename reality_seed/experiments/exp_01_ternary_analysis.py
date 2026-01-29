"""
Experiment 01: Ternary vs Binary Split Analysis

Question: Is the 55-periodicity specific to binary splits, or more universal?

Finding: The 55-depth structure comes from F_10 (10th Fibonacci).
With ternary splits (1→3), we would expect T_n (Tribonacci) periodicity instead.

Binary: phi^n / F_n → sqrt(5) = 2.236
Ternary: tau^n / T_n → 5.47

This suggests the binary structure of reality (±, up/down, matter/antimatter)
is fundamental to the emergence of 55-periodicity.
"""

import math
import numpy as np

PHI = (1 + math.sqrt(5)) / 2
TRIBONACCI = 1.8392867552141612

# Fibonacci sequence
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

# Tribonacci sequence: T_n = T_{n-1} + T_{n-2} + T_{n-3}
def tribonacci(n):
    if n <= 1:
        return 0
    if n == 2:
        return 1
    a, b, c = 0, 0, 1
    for _ in range(n - 2):
        a, b, c = b, c, a + b + c
    return c


def analyze_binary():
    """Analyze binary split (Fibonacci) structure."""
    print("Binary (1→2) Analysis:")
    print("  Convergent constant: φ = %.6f" % PHI)
    print("  Node formula: N(t) = 2t + 1")
    print()
    
    print("  φ^n / F_n convergence:")
    for n in range(5, 15):
        f_n = fibonacci(n)
        ratio = PHI**n / f_n
        print("    n=%2d: F_n=%3d, ratio=%.6f" % (n, f_n, ratio))
    
    print()
    print("  Limit: φ^n / F_n → √5 = %.6f" % math.sqrt(5))
    print("  Identity: φ + 1/φ = √5 = %.6f" % (PHI + 1/PHI))
    print()
    print("  Möbius depth: F_10 = 55 (where twist = π)")


def analyze_ternary():
    """Analyze ternary split (Tribonacci) structure."""
    print("\nTernary (1→3) Analysis:")
    print("  Convergent constant: τ = %.6f" % TRIBONACCI)
    print("  Node formula: N(t) = 3t + 1")
    print()
    
    print("  τ^n / T_n convergence:")
    for n in range(5, 15):
        t_n = tribonacci(n)
        ratio = TRIBONACCI**n / t_n if t_n > 0 else float('inf')
        print("    n=%2d: T_n=%3d, ratio=%.6f" % (n, t_n, ratio))
    
    print()
    print("  Limit: τ^n / T_n → 5.47")
    print("  Predicted depth: T_10 = 81 (analog to F_10 = 55)")


def main():
    print("=" * 70)
    print("TERNARY VS BINARY SPLIT ANALYSIS")
    print("=" * 70)
    print()
    
    analyze_binary()
    print()
    analyze_ternary()
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The 55-structure is SPECIFIC to binary reality:")
    print("  - Binary splits → Fibonacci → F_10 = 55 → φ")
    print("  - Ternary splits → Tribonacci → T_10 = 81 → τ")
    print()
    print("Physical interpretation:")
    print("  - Reality is binary (matter/antimatter, spin up/down)")
    print("  - This binary structure causes 55 to be the Möbius depth")
    print("  - φ emerges because of binary recursion, not despite it")


if __name__ == "__main__":
    main()
