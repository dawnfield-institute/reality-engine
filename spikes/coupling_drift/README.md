# Coupling Drift Spikes

Research experiments investigating and fixing coupling constant drift in the Reality Engine.

**Problem**: Emergent coupling constants (f, gamma, alpha, G, lambda) converge to DFT-predicted attractors at tick 1000 but drift badly by tick 10000. Root cause: gravity runaway clumping via local-global asymmetry.

**Result**: 5/13 (D+) -> 9/13 (C+) physics scorecard through 6 targeted interventions.

## Spikes

| Spike | What it tests | Key finding |
|-------|---------------|-------------|
| `sweep_interventions.py` | 10 config-level parameter combos | No config change alone fixes G_local (45-67% error) |
| `code_interventions.py` | 9 code-level gravity variants | sqrt Poisson clear winner (12.1% avg error) |
| `spectral_tiling.py` | 6 cascade-depth tiling strengths | Tiling 1.0x: drift reduced from +17% to +0.7% |
| `nonlocal_landauer.py` | 5 Landauer redistribution strategies | No effect — tiling prevents cap hits, Landauer=0 |
| `memory_generation.py` | 5 memory operator variants | Gradient seeding: G_local 2.0%, empty cells 37%->25% |
| `late_drift.py` | 6 gravity+memory combos for late-time fix | phi-scaled xi_mod + adaptive gradient: zero drift |

## What was implemented

### Gravity operator (`src/v3/operators/gravity.py`)
1. **xi_mod entropy-coherence modulation**: G_local = G_mass * xi_mod where xi_s = I^2/E^2
2. **Amplitude coupling**: Poisson source = sqrt(M) instead of M
3. **Cascade-depth spectral tiling**: filter(k) = (ln^2(2))^(Xi * n(k)) — local gravity strong, global suppressed
4. **phi-scaled xi_mod**: xi_s^(1/phi) flattens response, prevents late-time overshoot

### Memory operator (`src/v3/operators/memory.py`)
5. **Gradient boundary seeding**: mass_gen += gamma_local * |grad(E-I)|^2
6. **Adaptive suppression**: boundary term * 1/(1+M) — targets empty cells

## DFT theory sources
- **exp_28**: ln^2(2) = 0.4805 as round-trip deficit at Landauer fraction
- **exp_29**: Xi = gamma_EM + ln(phi) as global frame attractor, infodynamic gravity framework
- **exp_36**: Cosmological constant as SEC cost of tiling local PAC patches globally

## Running

Each spike is standalone. Run from the reality-engine root:

```bash
python spikes/coupling_drift/spectral_tiling.py    # ~5 min on GPU
python spikes/coupling_drift/late_drift.py          # ~15 min on GPU (10K ticks x 6 variants)
```

Full scorecard validation:
```bash
python scripts/physics_scorecard.py                 # ~3 min on GPU
```
