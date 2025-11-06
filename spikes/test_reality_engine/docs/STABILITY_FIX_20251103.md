# Reality Engine Stability Fix - November 3, 2025

## Problem
Reality Engine fields were exploding to clamp values (±1,000,000) within 2000 steps, preventing particle detection.

## Root Causes Identified

### 1. **Timestep Too Large** (PRIMARY ISSUE)
- **Was**: `dt = 0.1` 
- **Now**: `dt = 0.0001`
- **Impact**: 1000x reduction in step size
- **Result**: Fields stay in range [0, 2] instead of saturating at clamps

### 2. **Double-Laplacian Bug**
```python
# BEFORE (WRONG):
B = recursive_balance_field()  # Contains ∇²(E-I) already
balance_gradient = laplacian(B)  # Taking Laplacian AGAIN!
energy_update = dt * balance_gradient  # ∇⁴ operator → exponential growth

# AFTER (CORRECT):
B = rbf_engine.compute_balance_field(E, I, M)  # ∇²(E-I) + λM∇²M - α||E-I||²
dE_dt = 0.1 * B  # Scale for stability
dE = dt * dE_dt  # Apply directly, no double differentiation
```

### 3. **Fracton PyTorch Compatibility**
```python
# fracton/field/rbf_engine.py - Fixed roll() syntax

# BEFORE (NumPy only):
self.np.roll(field, 1, axis=axis)

# AFTER (PyTorch compatible):
if self.is_torch:
    self.np.roll(field, 1, dims=axis)  # PyTorch uses 'dims'
else:
    self.np.roll(field, 1, axis=axis)  # NumPy uses 'axis'
```

### 4. **QBE Regulator Integration**
```python
# Fixed tuple unpacking from Fracton QBE regulator

# enforce_qbe_constraint() returns (dE_dt, dI_dt_qbe) tuple
_, dI_dt_qbe = self.qbe_regulator.enforce_qbe_constraint(
    dE_dt, -dE_dt, self.time
)
```

## Changes Made

### `reality-engine/core/dawn_field.py`
1. Removed double-Laplacian: `balance_gradient = self.laplacian(B)` ❌
2. Use RBF balance field directly with 0.1 scaling
3. Proper QBE regulator integration with tuple unpacking
4. Memory growth from collapse: `dM_dt = 0.001 * alpha * (E-I)²`

### `fracton/fracton/field/rbf_engine.py`
1. Added PyTorch `dims` vs NumPy `axis` compatibility in `compute_laplacian()`

### `reality-engine/examples/analyze_physics.py`
1. Changed `dt=0.1` → `dt=0.0001`

### `reality-engine/scripts/diagnose_stability.py`
1. Changed `dt=0.1` → `dt=0.0001`
2. Added field range monitoring

## Validation Results

### Before Fix:
```
Step 2000:
  E: [-1000000, 1000000] (SATURATED)
  M: [0, 25500012] (EXPLODED)
  Particles: 0 detected
```

### After Fix:
```
Step 2000:
  E: [0.1925, 0.5871] ✓
  I: [0.1659, 0.4001] ✓
  M: [0.0000, 1.0258] ✓
  Particles: 1 detected with stability=0.926 ✓
```

## Technical Details

### RBF Evolution (Corrected)
```
B(x,t) = ∇²(E-I) + λM∇²M - α||E-I||²
dE/dt = 0.1 · B          (scaled for stability)
dI/dt = λ·QPL(t) - dE/dt (QBE constraint)
dM/dt = 0.001·α·||E-I||² (collapse accumulation)
```

### Stability Analysis
- Balance field B ∈ [-2, 1] for typical initial conditions
- With dt=0.0001 and 0.1 scaling: max change per step = 0.1 × 0.0001 × 2 = 0.00002
- This prevents runaway growth while allowing structure formation

### Particle Detection Success
- After 2000 steps, detected stable vortex with:
  - Position: [10, 16, 24]
  - M_center: 1.0258
  - Stability: 0.926 (well above 0.01 threshold)
  - Localized: std(M_local) = 0.0804

## Next Steps
- ✅ Fields stable
- ✅ Particle detection working
- 🔄 Running full 5000-step analysis to build periodic table
- ⏭️ Validate if standard model particles emerge

## Key Insight
The timestep `dt=0.0001` was validated in MED experiments and documented in the codebase but was incorrectly changed to `dt=0.1` in the analysis example. Reverting to the proven value restored stability.
