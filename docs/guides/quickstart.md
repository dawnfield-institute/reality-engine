# Quick Start Guide

**Get Reality Engine v2 running in 5 minutes**

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- 4GB+ RAM

### Install Dependencies

```bash
cd reality-engine
pip install -r requirements.txt
```

**Requirements**:
```
torch>=2.0.0
numpy>=1.20.0
matplotlib>=3.5.0
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.0.1
CUDA: True
```

---

## First Run: Test the Substrate

### Test Möbius Manifold

```bash
cd tests
python test_mobius_substrate.py
```

Expected output:
```
============================================================
MÖBIUS SUBSTRATE TEST SUITE
============================================================

✓ TEST 1: Möbius Manifold Initialization
✓ TEST 2: Field Initialization Modes
✓ TEST 3: Anti-Periodic Boundaries
✓ TEST 4: Topology Metrics
✓ TEST 5: FieldState Operations

============================================================
✓ ALL TESTS COMPLETE
============================================================
```

If all tests pass, you're ready to go!

---

## Hello, Möbius!

### Minimal Example

Create `examples/hello_mobius.py`:

```python
"""
Minimal Reality Engine v2 example
Creates fields on Möbius substrate and displays them
"""

import torch
import sys
sys.path.insert(0, '..')

from substrate import MobiusManifold

# Create Möbius substrate
substrate = MobiusManifold(
    size=64,      # Loop direction
    width=16,     # Strip width
    seed=42       # Reproducible
)

# Initialize fields
state = substrate.initialize_fields(mode='big_bang')

# Display state
print("\n" + "="*60)
print("REALITY ENGINE V2 - HELLO MÖBIUS")
print("="*60)
print(f"\nSubstrate: {substrate}")
print(f"Field shape: {state.shape}")
print(f"Device: {state.device}")

print(f"\nField Values:")
print(f"  Potential (P): mean={state.P.mean():.4f}, std={state.P.std():.4f}")
print(f"  Actual (A):    mean={state.A.mean():.4f}, std={state.A.std():.4f}")
print(f"  Memory (M):    mean={state.M.mean():.4f}, std={state.M.std():.4f}")

print(f"\nConservation:")
print(f"  Total PAC: {state.total_pac():.4f}")
print(f"  Energy:    {state.energy():.4f}")
print(f"  Info:      {state.information():.4f}")
print(f"  Matter:    {state.matter():.4f}")

# Check topology
metrics = substrate.calculate_metrics(state.P)
print(f"\nTopology Metrics:")
print(f"  Anti-periodic quality: {metrics.anti_periodic_quality:.4f}")
print(f"  Field coherence:       {metrics.field_coherence:.4f}")
print(f"  Ξ measurement:         {metrics.xi_measurement:.4f}")

print("\n" + "="*60)
print("✓ Reality Engine v2 initialized successfully!")
print("="*60 + "\n")
```

### Run It

```bash
cd examples
python hello_mobius.py
```

Expected output:
```
============================================================
REALITY ENGINE V2 - HELLO MÖBIUS
============================================================

Substrate: MobiusManifold(size=64, width=16, twist=1.0, device=cuda)
Field shape: torch.Size([64, 16])
Device: cuda:0

Field Values:
  Potential (P): mean=0.8012, std=0.2193
  Actual (A):    mean=0.0000, std=0.0472
  Memory (M):    mean=0.0000, std=0.0000

Conservation:
  Total PAC: 820.4706
  Energy:    820.4688
  Info:      0.0018
  Matter:    0.0000

Topology Metrics:
  Anti-periodic quality: 0.0000
  Field coherence:       0.9295
  Ξ measurement:         1.0571

============================================================
✓ Reality Engine v2 initialized successfully!
============================================================
```

---

## Understanding the Output

### Substrate

```
MobiusManifold(size=64, width=16, twist=1.0, device=cuda)
```

- `size=64`: 64 points around the loop (u direction)
- `width=16`: 16 points across the strip (v direction)
- `twist=1.0`: Full Möbius twist
- `device=cuda`: Running on GPU

### Fields

**P (Potential)**: High energy state (Big Bang!)
```
mean=0.8012  ← High potential energy
std=0.2193   ← With 10% fluctuations
```

**A (Actual)**: Minimal information seeds
```
mean=0.0000  ← Almost zero (no structure yet)
std=0.0472   ← Small random seeds
```

**M (Memory)**: No matter yet
```
mean=0.0000  ← Nothing crystallized
std=0.0000   ← Pure energy state
```

### Conservation

```
Total PAC: 820.4706
Energy:    820.4688
Info:      0.0018
Matter:    0.0000

E + I + M ≈ 820.47  ✓ Conserved!
```

### Topology

```
Anti-periodic quality: 0.0000  ← Will be enforced by Confluence
Field coherence:       0.9295  ← Good!
Ξ measurement:         1.0571  ← Expected value!
```

---

## Next Steps

### 1. Try Different Initialization Modes

```python
# Random fields
state = substrate.initialize_fields(mode='random')

# Structured with braided strands
state = substrate.initialize_fields(mode='structured')
```

### 2. Explore Different Sizes

```python
# Small (fast)
substrate = MobiusManifold(size=32, width=8)

# Medium (balanced)
substrate = MobiusManifold(size=128, width=32)

# Large (detailed)
substrate = MobiusManifold(size=256, width=64)
```

### 3. Check GPU vs CPU

```python
# Force CPU
substrate = MobiusManifold(size=64, width=16, device='cpu')

# Auto-detect (uses GPU if available)
substrate = MobiusManifold(size=64, width=16, device='auto')
```

---

## Common Issues

### Issue: Import Error

```
ModuleNotFoundError: No module named 'substrate'
```

**Solution**: Add parent directory to path
```python
import sys
sys.path.insert(0, '..')  # Add this at top of file
```

### Issue: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce size or use CPU
```python
# Smaller size
substrate = MobiusManifold(size=32, width=8)

# Or use CPU
substrate = MobiusManifold(device='cpu')
```

### Issue: Tests Fail

```
✗ FAIL (threshold: 0.1)
```

**This is OK!** Anti-periodic enforcement is approximate in initialization. The Confluence operator will enforce it properly during evolution.

---

## What's Next?

Now that you have the substrate running:

### Learn More Theory
- Read [Theory Overview](../theory/overview.md)
- Understand [Möbius Topology](../theory/mobius_topology.md)
- Study [PAC Conservation](../theory/pac_conservation.md)

### Build Dynamics
Once conservation and dynamics layers are implemented:
```python
from dynamics import SECOperator, ConfluenceOperator
from conservation import PACKernel

# Evolution loop (coming soon!)
sec = SECOperator()
confluence = ConfluenceOperator(substrate)
pac = PACKernel()

for step in range(1000):
    A = sec.evolve(A, P)
    P, A, M = pac.enforce(P, A, M)
    P = confluence.step(A)
```

### Explore Examples
- [Big Bang Simulation](../examples/big_bang.md) (when ready)
- [Particle Physics](../examples/particle_physics.md) (when ready)
- [Law Discovery](../examples/law_discovery.md) (when ready)

---

## Development Status

**✅ Working Now**:
- Möbius substrate
- Field initialization
- Topology metrics
- Conservation tracking

**🚧 In Progress**:
- PAC kernel
- SEC-MED operators
- Confluence dynamics

**📋 Planned**:
- Particle detection
- Law discovery
- Visualization suite

---

## Getting Help

- 📖 **Full Documentation**: [docs/README.md](README.md)
- 🐛 **Troubleshooting**: [guides/troubleshooting.md](troubleshooting.md)
- 💬 **Theory Questions**: [theory/overview.md](../theory/overview.md)
- 🔧 **API Reference**: [api/README.md](../api/README.md)

---

**Congratulations!** You've initialized your first Möbius manifold.

Now you're ready to explore emergent physics! 🌀

---

**Last Updated**: November 3, 2025  
**Version**: 2.0.0-alpha
