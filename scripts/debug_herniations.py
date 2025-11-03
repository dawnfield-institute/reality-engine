"""Debug visualization - check if herniations create structure."""
import torch
from core.dawn_field import DawnField
import matplotlib.pyplot as plt

print("Testing herniation structure formation...")

# Create field
field = DawnField(shape=(64, 64, 64))

# Initialize with some variation (skip Big Bang for now)
print("\nInitializing field...")
field.E = torch.rand_like(field.E) * 0.5 + 0.25
field.I = torch.rand_like(field.I) * 0.5 + 0.25

print(f"Initial PAC: {(field.E.sum() + field.I.sum() + field.M.sum()).item():.2f}")
print(f"Memory sites: {(field.M > 0.01).sum().item()}")
print(f"Memory max: {field.M.max().item():.4f}")
print(f"Memory mean: {field.M.mean().item():.4f}")

# Evolve for 100 steps
print("\nEvolving 100 steps...")
for i in range(100):
    field.evolve_step()
    if (i+1) % 20 == 0:
        print(f"  Step {i+1}: Memory sites={(field.M > 0.01).sum().item()}, max={field.M.max().item():.4f}")

# Check memory accumulation
print(f"\nFinal PAC: {(field.E.sum() + field.I.sum() + field.M.sum()).item():.2f}")
print(f"Memory sites: {(field.M > 0.01).sum().item()}")
print(f"Memory max: {field.M.max().item():.4f}")
print(f"Memory mean: {field.M.mean().item():.4f}")

# Check for 1/r structure in energy
print("\nChecking energy structure...")
center_slice = field.E[32, 32, :].cpu()
print(f"Energy center slice: min={center_slice.min():.4f}, max={center_slice.max():.4f}, mean={center_slice.mean():.4f}")

# Plot memory field slice
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(field.M[32, :, :].cpu(), cmap='hot')
plt.colorbar()
plt.title('Memory Field (XY slice at Z=32)')

plt.subplot(132)
plt.imshow(field.E[32, :, :].cpu(), cmap='coolwarm')
plt.colorbar()
plt.title('Energy Field (XY slice at Z=32)')

plt.subplot(133)
plt.imshow(field.I[32, :, :].cpu(), cmap='viridis')
plt.colorbar()
plt.title('Info Field (XY slice at Z=32)')

plt.tight_layout()
plt.savefig('herniation_structure.png', dpi=150, bbox_inches='tight')
print("\nSaved visualization to herniation_structure.png")
print("\nLook for:")
print("  - Memory accumulation at specific sites (bright spots)")
print("  - Energy dispersion patterns (radial from herniations)")
print("  - Information crystallization (fractal structure)")
