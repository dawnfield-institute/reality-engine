"""Quick check of what the balance field values are"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from core.dawn_field import DawnField

# Initialize
print("Initializing...")
reality = DawnField(shape=(32, 32, 32), dt=0.0001, device='cuda')  # Much smaller dt

# Check initial balance field
B = reality.rbf_engine.compute_balance_field(reality.E, reality.I, reality.M)

print(f"\nInitial Balance Field stats:")
print(f"  min={B.min().item():.6f}")
print(f"  max={B.max().item():.6f}")
print(f"  mean={B.mean().item():.6f}")
print(f"  std={B.std().item():.6f}")

print(f"\nField ranges:")
print(f"  E: [{reality.E.min().item():.4f}, {reality.E.max().item():.4f}]")
print(f"  I: [{reality.I.min().item():.4f}, {reality.I.max().item():.4f}]")
print(f"  M: [{reality.M.min().item():.4f}, {reality.M.max().item():.4f}]")

# Take a few steps
print("\nTaking 10 steps...")
for i in range(10):
    reality.evolve_step()
    
    if i % 2 == 0:
        B = reality.rbf_engine.compute_balance_field(reality.E, reality.I, reality.M)
        print(f"Step {i}: B range=[{B.min().item():.4f}, {B.max().item():.4f}], E range=[{reality.E.min().item():.4f}, {reality.E.max().item():.4f}]")
