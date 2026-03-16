# New Physics Operators: SpinStatistics, ChargeDynamics + Stability Fixes

## Changes

### New Operators
- **SpinStatisticsOperator** (`operators/spin_statistics.py`): Emergent Pauli exclusion from information cost of co-locating identical half-integer spin states. Spin emerges from curl of disequilibrium, charge from antisymmetric gradient. Enhanced diffusion for fermion-like states resists gravitational collapse.
- **ChargeDynamicsOperator** (`operators/charge_dynamics.py`): EM-like forces from charge field Q = dE/du - dI/dv. Poisson potential → gradient force → like charges repel, unlike attract. Force feeds into dE_dt (balance field) so actualization gate limits perturbation — prevents positive feedback.

### Stability Fixes
- **Normalization**: Fixed crystallisation overflow — M excess after crystallisation now gets Landauer-reinjected to E+I instead of being lost.
- **Pipeline ordering**: Documented that actualization must run BEFORE normalization (not after). The stable order is: RBF → QBE → Actualization → Memory → Gravity → [new ops] → Fusion → Confluence → Temperature → Noise → Normalization → Adaptive → TimeEmergence.

### Memory Operator
- Reverted emergent saturation attempt (soft cap was too permissive → NaN). Hard cap (field_scale/5.0) preserved. Emergent mass limits come from spin-statistics degeneracy pressure instead.

## Validation Results (10000 ticks, 128x64, seed 7)

| Metric | Without new ops | With new ops | Change |
|--------|----------------|-------------|--------|
| Mass peaks | 7 | 10-13 | +43-86% |
| Mass diversity (std) | 0.60 | 1.05 | +75% |
| Cap pileup (t=2000) | 66.6% | 5.2% | -92% |
| Bound fraction | 87% | 98% | +11% |
| 1/φ² spacing error | 22-60% | 7.7% | sharper |
| PAC drift | 8.5e-13 | 8.5e-13 | machine precision |

## Key Findings
1. Degeneracy pressure delays cap pileup by ~3x (from tick ~1000 to ~3000)
2. 12-13 mass peaks vs 7 — richer intermediate mass spectrum
3. 98.5% of structures are bound (vs 87%) — charge creates molecular binding
4. 1/φ² mass spacing sharpens to 7.7% error with new physics
5. Half-integer spin fraction stays at ~55% — emergent fermionic statistics
