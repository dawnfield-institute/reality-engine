# Reality Engine v2 — Changelog

## 2026-02-22 — Ξ_L2 Convergence via RBF + Topology-Modulated Collapse

### Summary

**Ξ_L2 converges to 1.0581** (0.09% error from Ξ_ref = 1.0571).
Rock-solid from step 1000 onward across 10,000+ steps.

This was achieved through three interlocking mechanisms:

1. **RBF self-regulation** — PI control with Fibonacci harmonics and
   recursive memory dampening
2. **Symmetric collapse modulation** — topology-aware spectral asymmetry
   in the nonlinear collapse operator
3. **Low-k anti mode mixing** — cos(u) at φ⁻² ratio reduces diffusion
   penalty by 10× vs the pure Möbius mode sin(u)·sin(πv)

### New Mechanisms

#### RBF (Recursive Breathing Field)
- Formula: `B = ρ · gain · Φ / (1 + α|M|)`
- PI control: `gain = -tanh(ξ_gain · Δξ) - ki · ∫Δξ dt`
- Fibonacci harmonics: `Φ = cos(ω·t) + cos(ω·φ·t)`
- Recursive memory: `M = decay · M + (1-decay) · B`
- Parameters: ρ=5.0, α_rbf=5.0, ki_rbf=1.0, ω=0.2, decay=0.995

#### Symmetric Collapse Modulation
- Decompose collapse C(S) into symmetric/anti components via confluence
- Modulate: `collapse = (1+tanh(B))·C_sym + (1-tanh(B))·C_anti`
- Doubles spectral asymmetry vs one-sided modulation
- Preserves total θ (C_anti sums to zero on Möbius)

#### Low-K Anti Mode
- Original: `anti_mode = sin(u)·sin(πv)` — k²=1+π²≈10.87
- New: `anti_mode = (1-mix)·sin(u)·sin(πv) + mix·cos(u)` — effective k²≈1
- mix = φ⁻² ≈ 0.382 (topology-natural golden ratio)
- cos(u) is antiperiodic on Möbius with k²=1 (10× less diffusion)

### Key Discovery: Collapse Saturation at S≈1

At P_mean ≈ 0.5 (high initialization), the field converges to S ≈ 1
where the nonlinear collapse C(S) = S·exp(-βS) has dC/dS = 0.
The anti-component of collapse vanishes to **third order** (C_anti ∝ δ³),
making all reinforcement and modulation mechanisms ineffective.
Result: Ξ_L2 locks at ~1.0006 regardless of parameters.

**Solution**: Initialize at P_mean = 0.1 (SEC linear regime where
dC/dS ≈ 0.97). This keeps the collapse operator responsive and allows
RBF + collapse modulation to drive Ξ_L2 → 1.057.

### Failed Approaches (Documented)

| Attempt | Result | Why |
|---------|--------|-----|
| Direct RBF reinforcement (R = B·f_anti) | P_std explodes | Unbounded energy injection |
| Source-only steering | Ξ stuck at 1.000 | σ₀ scale too weak |
| Normalized reinforcement | Ξ peaks 1.008, decays | Transient only |
| Conservative transfer (R = B·(f_anti - ξ·f_sym)) | Ξ = 1.0007 | Diffusion wins at k²≈11 |
| Multiplicative PAC | Ξ drops to 0.57 | Loses the E_sym pump that maintains collapse saturation |
| One-sided collapse modulation | Ξ = 1.035 ceiling | C_anti ∝ δ³ at S≈1 |
| Symmetric collapse (no low-k mix) | Ξ = 1.035 | Anti mode k²≈11 limits equilibrium |

### Configuration Changes

**constants.py** (`SEC_DEFAULTS`):
- `ki_rbf`: 0.5 → 1.0
- `low_k_mix`: new, 0.382 (≈ φ⁻²)

**default.yaml**:
- `P_mean`: 0.5 → 0.1
- `A_mean`: 0.5 → 0.1
- `ki_rbf`: 0.5 → 1.0
- `low_k_mix`: new, 0.382
- `antiperiodic_amp`: 0.05 → 0.01

### Files Modified

- `src/dynamics/sec.py` — RBF, symmetric collapse modulation, low-k anti mode
- `src/dynamics/pac.py` — docstring (additive correction is intentional)
- `src/substrate/constants.py` — SEC_DEFAULTS
- `configs/default.yaml` — init parameters
- `src/engine.py` — RBF diagnostics (M_rbf, xi_integral)
- `tests/v2/test_sec.py` — 18 tests including RBF mechanism tests
- `experiments/exp_04_xi_emergence.py` — uses engine with Ξ_L2 metric
- `experiments/exp_05_full_integration.py` — updated config + Ξ_L2 threshold

### Test Results

- **54/54 v2 unit tests pass**
- **exp_01**: 4/4 substrate validation ✓
- **exp_02**: 5/5 confluence conservation ✓
- **exp_03**: 4/4 SEC structure formation ✓
- **exp_04**: 4/4 Ξ emergence (Ξ_L2=1.0582, mean±std = 1.0565 ± 0.0012) ✓
- **exp_05**: 5/5 full integration (Ξ_L2=1.0581, PAC residual < 5e-7, 361 steps/sec) ✓
