"""Quick diagnostic to check what stability values we're getting"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np

from core.dawn_field import DawnField
from emergence.particle_analyzer import ParticleAnalyzer

# Initialize
print("Initializing...")
reality = DawnField(shape=(32, 32, 32), dt=0.0001, device='cuda')  # Validated stable dt

# Evolve a bit
print("Evolving 2000 steps...")
for i in range(2000):
    reality.evolve_step()
    if i % 500 == 0:
        print(f"  Step {i}: E=[{reality.E.min().item():.4f}, {reality.E.max().item():.4f}], M=[{reality.M.min().item():.4f}, {reality.M.max().item():.4f}]")

# Get fields
E_np = reality.E.cpu().numpy()
I_np = reality.I.cpu().numpy()
M_np = reality.M.cpu().numpy()

print(f"\nField stats:")
print(f"  E: min={E_np.min():.4f}, max={E_np.max():.4f}, mean={E_np.mean():.4f}")
print(f"  I: min={I_np.min():.4f}, max={I_np.max():.4f}, mean={I_np.mean():.4f}")
print(f"  M: min={M_np.min():.4f}, max={M_np.max():.4f}, mean={M_np.mean():.4f}")

# Try detection with NO stability threshold
print("\nDetecting particles (no stability threshold)...")
analyzer = ParticleAnalyzer()

# Manually check a few local regions
from scipy.ndimage import maximum_filter
peaks = (M_np == maximum_filter(M_np, size=7)) & (M_np > 0.1)
positions = np.array(np.where(peaks)).T

print(f"Found {len(positions)} candidate sites")

if len(positions) > 0:
    # Check first 5 sites
    stabilities = []
    for pos in positions[:min(5, len(positions))]:
        x, y, z = pos
        if x < 3 or y < 3 or z < 3 or x >= 29 or y >= 29 or z >= 29:
            continue
        
        M_local = M_np[x-3:x+4, y-3:y+4, z-3:z+4]
        stability = 1.0 / (1.0 + np.std(M_local) + 1e-6)
        stabilities.append(stability)
        
        print(f"\nSite {pos}:")
        print(f"  M center: {M_np[x,y,z]:.4f}")
        print(f"  M local mean: {M_local.mean():.4f}")
        print(f"  M local std: {np.std(M_local):.4f}")
        print(f"  Stability: {stability:.6f}")
    
    if stabilities:
        print(f"\nStability stats:")
        print(f"  Min: {min(stabilities):.6f}")
        print(f"  Max: {max(stabilities):.6f}")
        print(f"  Mean: {np.mean(stabilities):.6f}")
        
        print(f"\nWith threshold=0.01, would detect: {sum(1 for s in stabilities if s > 0.01)}")
        print(f"With threshold=0.001, would detect: {sum(1 for s in stabilities if s > 0.001)}")
