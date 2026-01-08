# Thermal Noise Restoration - Reality Engine Stability Fix

**Date**: 2026-01-08 12:50
**Commit**: pending
**Type**: bugfix

## Summary
Diagnosed and fixed critical issue where RealityEngine was producing no structures or herniations. The root cause was that SEC.evolve() with thermal noise injection was removed during the RBF+QBE refactor, causing the system to over-stabilize and "freeze" into equilibrium.

## Changes

### Added
- Thermal fluctuations (Langevin dynamics) to step() function - step 5b
- Field normalization clamps to prevent runaway values - step 5c
  - E, I clamped to [-100, 100]
  - M clamped to [0, 100]

### Changed
- `core/reality_engine.py`: Added thermal noise injection using T_mean and randn
- `core/reality_engine.py`: Reduced noise_scale from 0.1 to 0.01 for stability
- `core/reality_engine.py`: Reduced memory accumulation rate with 0.01 multiplier
- `emergence/herniation_detector.py`: Normalized herniation potential using tanh() to prevent intensity explosion (was reaching 10^10)

### Fixed
- Missing SEC.evolve() call that provided thermal noise (removed in previous refactor)
- Herniation detector intensity explosion (now bounded 0-1 range)
- Field value explosion causing NaN/Inf errors
- Over-stabilization that prevented structure formation

## Details

### Root Cause Analysis
Git archaeology revealed commit 47b87ff had:
```python
A_new, heat_generated = self.sec.evolve(
    state.A, state.P, state.T, dt=self.dt, add_thermal_noise=True
)
```
This was completely removed in the RBF+QBE refactor, leaving no thermal noise injection.

### Solution
Added Langevin-style thermal fluctuations directly in step():
```python
thermal_amplitude = torch.sqrt(2.0 * T_current.mean() * self.dt + 1e-10)
thermal_noise_E = thermal_amplitude * torch.randn_like(E_new)
thermal_noise_I = thermal_amplitude * torch.randn_like(I_new)
```

### Validation Results
After fixes:
- 75 structures forming (M > 0.1) after 2000 steps
- 6,749 herniations detected and processed
- PAC conservation ~97%
- No NaN/Inf errors
- Quantum centers spawning at herniation sites

## Related
- SEC operator in `conservation/sec_operator.py` - still unused but available for future integration
- HerniationDetector now uses normalized inputs for stability
- AdaptiveParameters working correctly with restored dynamics
