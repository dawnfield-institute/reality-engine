# Fracton Integration + Unified Force Operator (Deep Physics Phase 2 + 5)

## Changes

### Phase 2A: SECTrackingOperator (`src/v3/operators/sec_tracking.py`)
- Read-only observer importing fracton's `SECFieldEvolver`
- Computes SEC energy functional (coupling, smoothness, thermal)
- Field entropy S = -Σp·log(p) and entropy reduction rate
- Graceful degradation if fracton not installed
- Pipeline position: after NormalizationOperator

### Phase 2B: PACValidator audit (`src/v3/operators/normalization.py`)
- Fracton's `PACValidator(tolerance=1e-10, auto_correct=False)` as secondary audit
- Logs `pac_validator_residual` and violation count to metrics
- No auto-correction — existing PAC correction remains primary

### Phase 2C: 2D Projections Module (`src/v3/substrate/projections.py`)
- Adapted from fracton's 3D projections for 2D Möbius manifold
- `project_symmetric_2d`: amplitude mean over field-type axis → gravity potential
- `project_antisymmetric_2d`: phase-weighted mean → 2D EM vector field (Fu, Fv)
- `depth_2_projection_2d`: both projections from same pre-field
- 2D differential operators: `gradient_2d`, `divergence_2d`, `curl_2d`

### Phase 5: UnifiedForceOperator (`src/v3/operators/unified_force.py`)
- Derives BOTH gravity and EM from `torch.stack([E, I, M])` pre-field
- Symmetric projection → Poisson solve → mass redistribution (gravity)
- Antisymmetric projection → curl → dE_dt contribution (EM)
- Tracks emergent grav/EM energy ratio (converges to ~2.6)
- Available as alternative to separate Gravity + Charge operators

### Validation script
- `scripts/validate_unified_force.py` — side-by-side comparison

## Results

### Unified Force (5/5 pass):
- 1721 structures (vs 1580 separate)
- φ² spacing: **0.2% error** at tick 500-1000, 16.6% at tick 10000
- Cap pileup: 18.8% (vs 21.8%)
- PAC drift: 2.91e-06 (vs 2.64e-07)
- Emergent grav/EM ratio: ~2.6 (new prediction)
- 138 existing tests still pass

### Pipeline decision
Default pipeline keeps separate operators (better late-game φ² spacing: 6.4% vs 16.6%).
Unified operator available for experiments requiring single pre-field origin.

## Physics basis
- DFT experiment `gravity_from_maxwell_pac`: EM + gravity from same pre-field
- fracton's `depth_2_projection`: symmetric=gravity, antisymmetric=EM
- SEC energy functional: E(A|P,T) = α||A-P||² + β||∇A||² + γ∫T·|A|²
