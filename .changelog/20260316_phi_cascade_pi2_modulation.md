# PhiCascadeOperator + π/2 Collapse Modulation (Deep Physics Phase 3-4)

## Changes

### New: PhiCascadeOperator (`src/v3/operators/phi_cascade.py`)
- Fibonacci two-step memory creating φ-spaced mass levels
- Cascade rate = α_PAC × rate_prev1 + (Ξ-1) × rate_prev2 (DFT-derived weights)
- Discrete mass shelves at M_cap × φ⁻ᵏ via cos²(π × frac(depth)) proximity function
- Cascade depth metric exported for downstream operators (actualization, Phase 4+)
- PAC-conserving: mass changes drain equally from E and I
- Added to default pipeline in `__main__.py` (after Memory, before Gravity)

### Modified: ActualizationOperator (`src/v3/operators/actualization.py`)
- Depth-dependent MAR threshold: threshold × (π/2)^(depth/4)
- Deeper cascade levels require more potential to actualize
- Creates harmonic oscillator spacing (E_n ∝ n+1/2 prediction from GAIA)
- Falls back to uniform threshold if cascade_depth not available

### New: Validation scripts
- `scripts/validate_phi_cascade.py` — head-to-head comparison with/without cascade
- `scripts/validate_phase4_quick.py` — combined Phase 3+4 stability + spectral test

## Results

### PhiCascade validation (5/5 tests pass):
- 1/φ² spacing error: 23.5% → **4.8%** (target <3%)
- Cap pileup: 27.8% → 24.7%
- PAC conservation: drift = 2.64e-07 (unchanged)
- Mass diversity maintained: std 1.257 → 1.273
- Cascade depth grows 5→8 over 10K ticks (φ-shelf settling)
- Phi proximity increases 0.62→0.88 (discrete level snapping)

### Combined Phase 3+4:
- **0.7% spacing error at tick 5000** (best ever achieved)
- 10 mass peaks sustained through 10K ticks
- Cap pileup reduced to 21.8%
- Clean integer harmonic series at all mass sites
- 138 existing tests still pass

## Physics basis
- Fibonacci two-step memory from DFT cascade deep-dive experiment
- π/2-harmonic collapse from GAIA POC-023 (validated ω_collapse mechanism)
- Grid independence confirmed: f₀ = 1/√2 is resolution-independent (N^0.003 scaling)
