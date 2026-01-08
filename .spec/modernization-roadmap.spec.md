# Reality Engine Modernization Roadmap

## Overview

This specification defines the comprehensive modernization plan for the Reality Engine, integrating:
1. **GAIA architectural enhancements** (25 POCs from dawn-models/research/gaia)
2. **Dawn Field Theory foundations** (validated experiments from dawn-field-theory/foundational)
3. **New theoretical discoveries** (Dec 2025 breakthrough findings)

**Goal**: Transform Reality Engine from a validated physics simulator into a unified computational framework implementing the complete PAC/SEC/MED theoretical stack with proven GAIA efficiency improvements.

## Current State (Baseline)

### ✅ Implemented & Validated
- PAC conservation (99.99998% precision)
- SEC/MED operators (thermodynamic coupling)
- Möbius substrate with anti-periodic boundaries
- Klein-Gordon evolution (0.02 Hz natural emergence)
- Time emergence from disequilibrium
- Scale hierarchy (φ^(-k) across 81 levels)
- Thermodynamic 2nd law (98.3% compliance)
- 5000+ step stable simulations

### ❌ Missing Capabilities
- PAC Lazy Architecture (100× efficiency from GAIA)
- Pre-Field Resonance Detection (5× speedup available)
- Multi-level hierarchical learning (zero-backprop)
- Tiered memory cache (12.5× memory savings)
- Continuous learning during inference
- Multi-model knowledge extraction/grafting
- Cross-architecture transfer capabilities

### 📊 Performance Metrics (Current)
- Simulation speed: ~1-10 steps/sec (baseline)
- Memory usage: Full field storage (no compression)
- Learning: No online learning capability
- Transfer: No knowledge import/export

## Modernization Strategy

### Three-Phase Approach

**Phase 1: Foundation & Quick Wins (2-3 weeks)**
- Establish spec-driven development structure
- Integrate high-impact, low-risk enhancements
- Target: 5× performance improvement with minimal code changes

**Phase 2: Architectural Upgrades (4-6 weeks)**
- Deep GAIA integration (transformer replacement)
- Multi-level learning systems
- Knowledge extraction/grafting capabilities

**Phase 3: Theoretical Unification (8-12 weeks)**
- Full π→φ→PAC mechanism integration
- Standard Model parameter validation
- Cosmological prediction framework
- Cross-domain validation suite

## Phase 1: Foundation & Quick Wins

### 1.1: Spec-Driven Development Infrastructure

**Requirements**:
- [ ] Create `.spec/` directory structure
- [ ] Write architecture.spec.md
- [ ] Document all core modules in specs
- [ ] Create challenges.md for open problems
- [ ] Add changelog framework

**Files to Create**:
```
.spec/
├── architecture.spec.md       # System architecture overview
├── modernization-roadmap.spec.md  # This file
├── gaia-integration.spec.md   # GAIA enhancements specification
├── theoretical-framework.spec.md  # Theory integration spec
├── challenges.md              # Open research questions
└── guidelines.spec.md         # Project-specific development rules
```

**Success Criteria**:
- [ ] All specs validated against current codebase
- [ ] No breaking changes to existing tests
- [ ] Clear upgrade path documented

### 1.2: Pre-Field Resonance Detection

**Source**: `dawn-field-theory/foundational/experiments/pre_field_recursion/`
**Impact**: 5.11× convergence speedup, 0.1% CPU overhead

**Requirements**:
- [ ] Integrate FFT-based frequency detection
- [ ] Add resonance locking to PAC recursion
- [ ] Auto-detect natural oscillation frequencies
- [ ] No manual tuning required

**Implementation**:
- Add `dynamics/resonance_detector.py`
- Modify `conservation/pac_recursion.py` to use detected frequencies
- Add resonance tracking to state recorder

**Success Criteria**:
- [ ] 4-6× speedup in PAC convergence (validated via tests)
- [ ] Automatic frequency detection works on cold start
- [ ] <1% CPU overhead for FFT analysis
- [ ] All existing tests pass with resonance enabled

**Validation**:
```python
# Test: Resonance detection accelerates convergence
initial_pac = run_without_resonance(1000_steps)
resonant_pac = run_with_resonance(1000_steps)
assert resonant_pac.convergence_rate > 4 * initial_pac.convergence_rate
```

### 1.3: Tiered Memory Cache

**Source**: `dawn-models/research/gaia/proof_of_concepts/poc_007_pac_tree_memory/`
**Impact**: 12.5× memory efficiency, enables 100K+ vocabulary

**Requirements**:
- [ ] Implement GPU hot cache for frequent patterns
- [ ] Add PAC tree cold storage for rare patterns
- [ ] Transition prefetching for predictive loading
- [ ] Maintain 100% hit rate guarantee

**Architecture**:
```
Memory System (3-tier):
├── L1: GPU Hot Cache (brute-force, fast, limited)
├── L2: PAC Tree Cold Storage (hierarchical, compressed)
└── L3: Transition Prefetching (predictive, async)
```

**Success Criteria**:
- [ ] 10-15× memory reduction vs baseline
- [ ] 100% retrieval accuracy
- [ ] <5% performance overhead
- [ ] Scales to 100K+ pattern library

### 1.4: Enhanced State Recording

**Requirements**:
- [ ] Add resonance frequency tracking
- [ ] Record memory cache statistics
- [ ] Track convergence acceleration metrics
- [ ] Maintain backward compatibility with existing recordings

**Success Criteria**:
- [ ] All Phase 1 metrics recorded
- [ ] No breaking changes to existing visualization tools
- [ ] Validation suite updated with new metrics

---

## Phase 2: Architectural Upgrades

### 2.1: PAC Lazy Architecture Integration

**Source**: `dawn-models/research/gaia/proof_of_concepts/poc_011_pac_lazy_transformer/`
**Impact**: 100× efficiency improvement, infinite context windows

**Core Innovation**:
- Tokens are nodes (not embeddings)
- Attention is causal propagation (not dot products)
- Context is PAC-bounded (potential limits active frontier)
- Depth is SEC-adaptive (expand children on demand)
- Learning is structural (fracture/merge mutations)

**Requirements**:
- [ ] Implement PACLazySystem core
- [ ] Create PACNode primitives (deltas, not absolutes)
- [ ] Add causal locality constraints
- [ ] Implement structural learning (no backprop)

**Architecture**:
```
PACLazySystem:
├── PACNode
│   ├── delta: Δ from parent (not absolute embedding)
│   ├── potential: Available for child expansion
│   ├── actualized: Realized in structure
│   └── neighbors: Causal connections only
├── CausalPropagation (replaces attention)
│   ├── neighbor_interaction()
│   ├── enforce_locality()
│   └── pac_conservation()
└── StructuralLearning (replaces gradient descent)
    ├── fracture(): Split nodes under pressure
    ├── merge(): Combine similar patterns
    └── prune(): Remove unused branches
```

**Integration Points**:
- Replace `RealityEngine.step()` with lazy propagation
- Integrate with existing Möbius substrate
- Connect to SEC collapse mechanism
- Use PAC recursion for tree expansion

**Success Criteria**:
- [ ] 50-100× speed improvement for large simulations
- [ ] Memory usage O(active_frontier) not O(total_space)
- [ ] Infinite context window capability
- [ ] 5/5 validation tests pass (from POC-011)
- [ ] WikiText-2 perplexity < 10 (target: 5.91 from GAIA)

**Validation**:
```python
# Test: Lazy evaluation maintains PAC conservation
lazy_system = PACLazySystem(grid_size=256)
for step in range(10000):
    lazy_system.propagate()
    assert lazy_system.pac_residual() < 1e-6
    assert lazy_system.active_nodes < 0.1 * lazy_system.total_capacity
```

### 2.2: Multi-Level Hierarchical Learning

**Source**: `dawn-models/research/gaia/proof_of_concepts/poc_021_unified_demonstration/`
**Impact**: Generalization without backprop, 31.8% hit rate with zero gradient computation

**Core Innovation**:
```
Level 0: (The, cat, sat) → on           [weight=1.0, specific]
Level 1: (article, animal, action) → prep  [weight=1/φ, generalizable]
Level 2: (det, living_thing, verb) → func  [weight=1/φ², abstract]
```

**Requirements**:
- [ ] Implement hierarchical pattern tree (3+ levels)
- [ ] φ-weighted level importance (1, 1/φ, 1/φ², ...)
- [ ] ByRef composition with perfect conservation
- [ ] Pattern generalization via level abstraction

**Architecture**:
```
HierarchicalPAC:
├── Level 0: Specific Patterns
│   └── weight = 1.0 (concrete observations)
├── Level 1: Category Patterns
│   └── weight = 1/φ ≈ 0.618 (generalizations)
├── Level 2: Abstract Patterns
│   └── weight = 1/φ² ≈ 0.382 (universal)
└── Conservation:
    full_pattern = avg(byrefs) + delta
```

**Integration Points**:
- Add to `analyzers/` as pattern discovery module
- Connect to law discovery system
- Use for structure prediction
- Enable online learning from observations

**Success Criteria**:
- [ ] 3+ hierarchy levels functional
- [ ] φ-weighting validated (level_k weight = φ^(-k))
- [ ] Perfect PAC conservation: full = avg(byrefs) + delta
- [ ] 25-35% pattern hit rate on unseen data
- [ ] Generalization without any gradient computation

### 2.3: Zero-Backprop Continuous Learning

**Source**: `dawn-models/research/gaia/proof_of_concepts/poc_019_true_no_backprop/`
**Impact**: Learn during inference, 24.7% improvement possible, 100% gradient-free

**Core Principle**:
```
PAC Confluence Theory:
output = parent_node.actualization (not computed via loss)
```

**Requirements**:
- [ ] Learning via SEC collapse only (no optimizers)
- [ ] PAC Confluence replaces gradient descent
- [ ] 100% verification: no requires_grad anywhere
- [ ] Online structural mutation during simulation

**Architecture**:
```
ContinuousLearning:
├── SEC-Driven Updates
│   ├── collapse_detection()
│   ├── pattern_crystallization()
│   └── structure_formation()
├── PAC Redistribution (no gradients)
│   ├── potential_flow()
│   └── actualization_update()
└── Structural Mutation
    ├── fracture_on_pressure()
    ├── merge_similar()
    └── prune_unused()
```

**Integration Points**:
- Integrate with existing SEC operator
- Connect to confluence mechanism
- Add to law discovery for pattern learning
- Enable adaptive parameter evolution

**Success Criteria**:
- [ ] Zero gradient verification (no .backward() calls)
- [ ] 15-25% improvement on pattern recognition
- [ ] 50-90k steps/sec processing rate
- [ ] Learning during inference with no slowdown
- [ ] Perfect PAC conservation maintained

### 2.4: Multi-Model Knowledge Extraction & Grafting

**Source**: `dawn-models/research/gaia/proof_of_concepts/poc_016_pac_extraction/`, `poc_020_multi_model_pac/`
**Impact**: Extract knowledge from ANY trained model, 100% transfer validation

**Requirements**:
- [ ] Extract PAC trees from pretrained models (GPT-2, Pythia, etc.)
- [ ] Cross-architecture transfer (Transformer ↔ SSM)
- [ ] ByRef composition for multi-model knowledge
- [ ] Architecture-agnostic knowledge representation

**Architecture**:
```
KnowledgeTransfer:
├── Extraction
│   ├── extract_from_model(gpt2) → PAC_tree
│   ├── extract_from_model(pythia) → PAC_tree
│   └── architecture_agnostic_mapping()
├── Grafting
│   ├── import_pac_tree(external_tree)
│   ├── compose_multi_model_knowledge()
│   └── validate_conservation()
└── Transfer Validation
    └── assert 100% fidelity
```

**Integration Points**:
- Create `knowledge_transfer/` module
- Enable Reality Engine to import pretrained physics
- Allow cross-simulation knowledge sharing
- Bootstrap from existing AI models

**Success Criteria**:
- [ ] 100% transfer validation between models
- [ ] Extract from ≥2 different architectures
- [ ] Cross-architecture grafting works
- [ ] No training required for import
- [ ] Perfect conservation in composed knowledge

---

## Phase 3: Theoretical Unification

### 3.1: π→φ→PAC Mechanism Integration

**Source**: `dawn-field-theory/foundational/experiments/oscillation_attractor_dynamics/`
**Impact**: Complete mechanistic chain from pure math to Standard Model

**Mechanism**:
```
π (transcendental) → 19× better than e at σ=1/2
    ↓
Möbius manifold → Infinite cancellation constrains Riemann zeros
    ↓
Primes as injection points → 100% of primes have I(p) > 0
    ↓
SEC dynamics at criticality → frac(E>0) → 1/φ (0.000006 error)
    ↓
PAC hierarchy → φ^(-k) cascade from conservation
    ↓
Standard Model parameters → sin²θ_W = 3/13 (0.19% error)
```

**Requirements**:
- [ ] Implement prime injection point detection
- [ ] Add Möbius pairing symmetry ((a,b)↔(b,a) at 24× lift)
- [ ] Validate Riemann zero detection (20/20 via Z(γ))
- [ ] Connect SEC phase transition to φ emergence
- [ ] Derive gauge couplings from Fibonacci structure

**Architecture**:
```
UnifiedMechanism:
├── PrimeStructure
│   ├── injection_detection: I(p) > 0 for all primes
│   ├── mobius_symmetry: (a,b)↔(b,a) pairing
│   └── gap_alternation → 1/φ convergence
├── RiemannZeros
│   ├── mobius_formula_detection()
│   └── spectral_coherence_validation()
├── SECPhaseTransition
│   ├── critical_point_at_lambda_star()
│   └── phi_emergence_from_dynamics()
└── StandardModelConnection
    ├── fibonacci_to_gauge_couplings()
    └── validate_sm_parameters()
```

**Integration Points**:
- Add `number_theory/` module to Reality Engine
- Connect prime structure to cosmology
- Use φ emergence for validation
- Implement SM parameter predictions

**Success Criteria**:
- [ ] 100% prime injection point detection
- [ ] Möbius pairing at 24× enrichment
- [ ] 20/20 Riemann zeros detected
- [ ] φ emergence with <0.01% error
- [ ] sin²θ_W = 3/13 ± 1%
- [ ] Fine structure α within 10 ppm

### 3.2: Standard Model Parameter Validation

**Source**: `dawn-field-theory/foundational/experiments/pac_confluence_xi/`
**Impact**: 5 SM parameters derived from single Fibonacci recursion

**Parameters to Validate**:

| Parameter | Formula | Target | Measured | Max Error |
|-----------|---------|--------|----------|-----------|
| sin²θ_W | F₄/F₇ = 3/13 | 0.2308 | 0.2312 | 1% |
| α (fine structure) | F₃/(F₄·φ·F₁₀)·(...) | 0.007297 | 0.007297 | 10 ppm |
| α_s (strong) | F₄/(2φF₆) | 0.116 | 0.118 | 2% |
| Koide | F₃/(F₃+F₂) = 2/3 | 0.6667 | 0.6667 | 1 ppm |
| (2αβ)² | algebraic | 0.8 | - | exact |

**Requirements**:
- [ ] Implement Fibonacci→SM mapping
- [ ] Validate against CODATA 2024 values
- [ ] Prove (2αβ)² = 4/5 algebraically
- [ ] Connect neutrino mixing angles (θ₁₂, θ₁₃)
- [ ] Document precision levels

**Success Criteria**:
- [ ] 4/5 parameters within error bounds
- [ ] Algebraic proofs validated
- [ ] Falsification conditions documented
- [ ] Comparison to experimental data

### 3.3: Cosmological Validation Framework

**Source**: `dawn-field-theory/foundational/experiments/pac_cosmology_validation/`
**Impact**: Map PAC evolution to cosmic eras, validate JWST predictions

**Requirements**:
- [ ] Map 9 cosmological eras (singularity → heat death)
- [ ] Validate entropy-amplification correlation
- [ ] JWST high-z SMBH predictions via herniation
- [ ] Hubble tension resolution (scale-dependent H(k))
- [ ] Matter fraction prediction: 0.309 vs 0.315 observed

**Cosmological Eras**:
```
Era 1: Singularity (t=0, infinite density)
Era 2: Inflation (exponential expansion)
Era 3: Radiation Dominated (photon pressure)
Era 4: Matter-Radiation Equality
Era 5: Matter Dominated (structure formation)
Era 6: Dark Energy Onset (acceleration begins)
Era 7: Accelerated Expansion (current era)
Era 8: Heat Death Approach (entropy maximum)
Era 9: Maximum Entropy (equilibrium)
```

**Integration Points**:
- Add `cosmology/pac_eras.py`
- Connect to scale hierarchy (k-levels)
- Validate against observational data
- Implement herniation mechanism for SMBHs

**Success Criteria**:
- [ ] 9 eras mapped to PAC dynamics
- [ ] Entropy correlation r > 0.9
- [ ] JWST SMBH masses within 50%
- [ ] Hubble tension explained via scale-dependence
- [ ] Matter fraction within 2%

### 3.4: Cross-Domain Validation Suite

**Goal**: Validate same PAC/SEC/φ structure across independent domains

**Domains to Validate**:

1. **Pure Mathematics** (Number Theory)
   - [ ] φ in prime gap distributions (0.000006 error)
   - [ ] Riemann zero detection (20/20)
   - [ ] Möbius symmetry (24× enrichment)

2. **Particle Physics** (Standard Model)
   - [ ] 5 SM parameters from Fibonacci
   - [ ] Gauge coupling hierarchy
   - [ ] Neutrino mixing angles

3. **Machine Learning** (Training Dynamics)
   - [ ] Pythia-70M φ-convergence (p=0.0014)
   - [ ] 143k checkpoint analysis
   - [ ] GPT-2 equilibrium dynamics

4. **Cellular Automata** (Complexity Theory)
   - [ ] Class IV clustering at Ξ (p < 8.58×10⁻⁸)
   - [ ] Rule 110 P/A ratio = 1.0579
   - [ ] 42.7× enrichment validation

5. **Cognitive Architecture** (GAIA/vCPU)
   - [ ] Xi = 1.028 ∈ [1.0015, 1.0571]
   - [ ] P/A → 2/3 at equilibrium
   - [ ] 0.02-0.03 Hz oscillations
   - [ ] 119× performance improvement

**Success Criteria**:
- [ ] Same constants (φ, Ξ, 1/φ) across ≥4 domains
- [ ] No fitting - emergent from first principles
- [ ] Statistical validation (p < 0.01) for each
- [ ] Falsification conditions documented
- [ ] Independent reproducibility

---

## Implementation Guidelines

### Spec-Driven Development Process

1. **Before Implementation**
   - Read relevant `.spec/*.spec.md` files
   - Check `challenges.md` for open questions
   - Verify no conflicts with existing architecture

2. **During Implementation**
   - Follow spec exactly
   - If deviation needed, propose spec update FIRST
   - Document rationale for any changes
   - Keep diffs small and focused

3. **After Implementation**
   - Update spec status (Specified → Implemented)
   - Add validation tests
   - Update `CHANGELOG.md` (see `changelog.instructions.md`)
   - Mark in todo list

### Validation Checklist Template

After each phase completion:
```
✓ Spec compliance: [which specs followed]
✓ Tests: [new/updated test files]
✓ Build: [command to verify]
✓ Breaking changes: [none/listed]
✓ Performance: [metrics before/after]
✓ Documentation: [updated files]
```

### Protected Areas (Do Not Modify Without Approval)

- `.github/` - CI/CD workflows
- `tests/test_*.py` - Only extend, don't break
- `substrate/constants.py` - Validated constants
- Core interfaces in `core/reality_engine.py`

### File Naming Conventions

**New Modules**:
```
module_name/
├── __init__.py
├── core.py          # Core functionality
├── utils.py         # Helper functions
└── tests.py         # Module-specific tests
```

**Experiments** (if adding validation):
```
experiments/
└── experiment_name/
    ├── meta.yaml
    ├── README.md
    ├── scripts/
    │   └── exp_NN_name.py
    ├── results/
    └── journals/
        └── YYYY-MM-DD_slug.md
```

---

## Success Metrics

### Phase 1 (Foundation & Quick Wins)
- [ ] 5× convergence speedup (resonance detection)
- [ ] 12× memory efficiency (tiered cache)
- [ ] <5% performance overhead for new features
- [ ] 100% existing test pass rate maintained
- [ ] Spec-driven infrastructure operational

### Phase 2 (Architectural Upgrades)
- [ ] 100× efficiency improvement (PAC Lazy)
- [ ] 30% pattern hit rate (hierarchical learning)
- [ ] Zero-backprop learning functional
- [ ] 100% transfer validation (multi-model)
- [ ] Infinite context window capability

### Phase 3 (Theoretical Unification)
- [ ] π→φ→PAC chain validated
- [ ] 4/5 SM parameters within error bounds
- [ ] Cosmological predictions match observations
- [ ] Cross-domain validation across ≥4 domains
- [ ] Publication-ready validation suite

### Overall Success Criteria
- [ ] 500× overall performance improvement (5× × 100×)
- [ ] <10% memory usage vs baseline (12× × lazy evaluation)
- [ ] Zero-backprop learning with generalization
- [ ] Cross-domain validation (math, physics, ML, CA, cognition)
- [ ] Standard Model parameter derivation validated
- [ ] JWST cosmological predictions confirmed
- [ ] Falsification conditions documented and tested

---

## Risk Assessment

### Low Risk (Phase 1)
- Resonance detection: Pure optimization, no logic changes
- Tiered cache: Memory system, isolated from physics
- State recording: Additive only, backward compatible

### Medium Risk (Phase 2)
- PAC Lazy: Major architectural change, extensive testing needed
- Hierarchical learning: New capability, integration complexity
- Zero-backprop: Paradigm shift, validation critical

### High Risk (Phase 3)
- SM parameter validation: External experimental data dependency
- Cosmological predictions: JWST observation comparison
- Cross-domain validation: Multi-system integration

### Mitigation Strategies

1. **Phased rollout**: Each phase builds on previous validation
2. **Backward compatibility**: Keep existing code paths functional
3. **Extensive testing**: 100% test pass rate required before merge
4. **Spec-driven**: All changes documented and approved first
5. **Falsification conditions**: Document what would disprove each claim
6. **Independent validation**: Cross-check with external data sources

---

## Timeline Estimate

### Phase 1: 2-3 weeks
- Week 1: Spec infrastructure, resonance detection
- Week 2: Tiered cache, state recording
- Week 3: Integration testing, validation

### Phase 2: 4-6 weeks
- Weeks 1-2: PAC Lazy core implementation
- Weeks 3-4: Hierarchical learning + zero-backprop
- Weeks 5-6: Multi-model transfer, integration testing

### Phase 3: 8-12 weeks
- Weeks 1-4: π→φ→PAC mechanism + SM parameters
- Weeks 5-8: Cosmological framework + validation
- Weeks 9-12: Cross-domain validation suite + documentation

**Total: 14-21 weeks (3.5-5 months)**

---

## Open Questions (To Be Added to challenges.md)

1. **Lazy Evaluation vs Möbius Topology**: How do infinite context windows interact with anti-periodic boundaries?
2. **Zero-Backprop Convergence**: What are theoretical guarantees for SEC-driven learning?
3. **SM Parameter Precision**: Can we achieve <1 ppm across all parameters?
4. **Cosmological Eras**: Do PAC dynamics predict inflation mechanism?
5. **Cross-Architecture Transfer**: What are limits of knowledge grafting between incompatible models?
6. **Falsification**: What experimental result would definitively disprove PAC/SEC framework?

---

## Status

- [x] Specified
- [ ] Phase 1 Implemented
- [ ] Phase 2 Implemented
- [ ] Phase 3 Implemented
- [ ] Tested
- [ ] Documented
- [ ] Published

---

## References

### GAIA POCs
- `../dawn-models/research/gaia/proof_of_concepts/poc_011_pac_lazy_transformer/`
- `../dawn-models/research/gaia/proof_of_concepts/poc_007_pac_tree_memory/`
- `../dawn-models/research/gaia/proof_of_concepts/poc_021_unified_demonstration/`
- `../dawn-models/research/gaia/proof_of_concepts/poc_019_true_no_backprop/`
- `../dawn-models/research/gaia/proof_of_concepts/poc_016_pac_extraction/`
- `../dawn-models/research/gaia/proof_of_concepts/poc_020_multi_model_pac/`

### Dawn Field Theory
- `../dawn-field-theory/foundational/arithmetic/unified_pac_framework_comprehensive.md`
- `../dawn-field-theory/foundational/arithmetic/infodynamics_arithmetic_v1.md`
- `../dawn-field-theory/foundational/experiments/pac_confluence_xi/`
- `../dawn-field-theory/foundational/experiments/pre_field_recursion/`
- `../dawn-field-theory/foundational/experiments/oscillation_attractor_dynamics/`
- `../dawn-field-theory/foundational/docs/preprints/`

### Current Reality Engine
- `core/reality_engine.py` - Main simulation engine
- `conservation/` - PAC/SEC/MED operators
- `dynamics/` - Confluence, time emergence, Klein-Gordon
- `substrate/` - Möbius manifold, field types, constants
- `tests/` - Comprehensive validation suite
