# Reality Engine -- Claude Code Context

## Identity
Reality Engine is a computational physics framework where fundamental physics emerges from information dynamics. Three fields (E=energy, I=information, M=mass) plus local rules generate quantum mechanics, gravity, thermodynamics, and relativity without programming them. v3 architecture (2026-03-14), pre-alpha research software based on Dawn Field Theory principles.

## Architecture

```
reality-engine/
├── src/v3/                    # ACTIVE — v3 composable operator pipeline
│   ├── engine/                # Engine, FieldState, EventBus, Config
│   │   ├── engine.py          # Main simulation loop
│   │   ├── state.py           # Immutable FieldState (E, I, M, T, metrics)
│   │   ├── event_bus.py       # Pub/sub event system
│   │   └── config.py          # SimulationConfig (nu, nv, dt, field_scale, ...)
│   ├── operators/ (18)        # Physics operators (see below)
│   ├── analyzers/ (6)         # Conservation, Gravity, Atom, Star, Quantum, Galaxy
│   ├── emergence/ (3)         # Structure, Particle, Herniation detectors
│   ├── substrate/             # MobiusManifold, Constants, Projections
│   └── dashboard/             # FastAPI + WebSocket + Plotly.js
├── scripts/                   # Diagnostics and validation
│   ├── physics_scorecard.py   # 3-tier physics validation (13 metrics)
│   ├── diagnose_gravity.py    # Gravity operator diagnostics
│   └── ...                    # Various analysis scripts
├── spikes/                    # Research experiments by topic
│   ├── theory_integration/    # DFT theory vs simulator (13 spikes, 10/10 pass)
│   ├── coupling_drift/        # Gravity/memory optimization (6 spikes)
│   ├── atomic_emergence/      # Atom classification
│   ├── big_bang/              # Big bang evolution
│   └── ...
├── tests/v3/                  # 138 tests (pytest)
├── examples/                  # field_visualizer.py
├── docs/                      # Theory and guides
├── proof_of_concepts/         # v2 POCs (001-007, reference only)
└── [legacy dirs]              # core/, dynamics/, conservation/, etc. — v1/v2 reference
```

## v3 Operator Pipeline (18 operators)

Default order in Pipeline:
1. **RBF** — Recursive Balance Field: dE/dt from laplacian, coupling terms
2. **QBE** — Quantum Balance: dI/dt = -dE/dt (PAC conservation)
3. **Actualization** — MAR-gated integration, ln(phi) split, pi/2 harmonic modulation
4. **Memory** — Mass generation (bulk + gradient boundary seeding), de-actualization (PAC cycle completion), quantum pressure, diffusion
5. **PhiCascade** — Fibonacci two-step memory for phi-spaced mass levels
6. **Gravity** — Self-gravity via spectral Poisson solver, xi_mod, cascade-depth tiling filter
7. **SpinStatistics** — Emergent Pauli exclusion from information cost
8. **ChargeDynamics** — EM-like forces from charge field Q
9. **Fusion** — Nuclear fusion in dense, hot, gravity-compressed regions
10. **Confluence** — Mobius antiperiodic projection f(u+pi,1-v) = -f(u,v)
11. **Temperature** — Local T from |E-I| gradients
12. **ThermalNoise** — Langevin noise
13. **Normalization** — Soft-clamp E/I, cap M, Landauer reinjection
14. **SECTracking** — Read-only SEC energy functional + entropy tracking
15. **Adaptive** — Self-tuning damping and dt from energy growth
16. **TimeEmergence** — dt = dt_base / (1 + kappa*max|E-I|)

Also available: EulerIntegrator, UnifiedForce (combined gravity+EM).

## Physics Scorecard

`python scripts/physics_scorecard.py` — 3-tier validation (13 metrics):
- **Tier 1** (coupling attractors): f->gamma_EM, gamma->1/phi, alpha->ln(2), G->1/phi^2, lambda->1-ln(2)
- **Tier 2** (structural): phi^2 mass spacing, PAC conservation, spin 1/2, entropy reduction
- **Tier 3** (aspirational): fine structure 1/137, Koide Q=2/3, mu/e ratio
- **Current score**: 8/13 passing (GPA C), as of 2026-03-16 (de-actualization reduced coupling drift 24%)

## Key Physics

- **Gravity**: Spectral Poisson solver with cascade-depth tiling filter (DFT exp_36). Entropy-coherence modulation xi_mod. Amplitude coupling nabla^2 Phi = sqrt(M).
- **Mass generation**: Bulk (gamma_local * diseq^2) + boundary gradient seeding (gamma_local * |grad(diseq)|^2 / (1+M))
- **De-actualization**: dM_deact = -eta * M * (1 - gamma_local). Memory fades where disequilibrium resolves, completing PAC cycle: potential -> actualization -> memory -> potential
- **PAC conservation**: E + I + M = const enforced at machine precision (<1e-12)
- **DFT constants**: Xi = gamma_EM + ln(phi) = 1.05843, ln^2(2) = 0.4805, phi = golden ratio

## Conventions

- Physics must EMERGE, never be programmed — no hardcoded F=ma, E=mc^2, etc.
- PAC conservation enforced at machine precision (< 1e-12)
- Mobius manifold substrate with anti-periodic boundaries: f(x+pi) = -f(x)
- Tests: `pytest tests/v3/` from repo root (138 tests)
- Installation: `pip install -r requirements.txt`
- Run quick demo: `python examples/field_visualizer.py`
- Scorecard: `python scripts/physics_scorecard.py`

## Related Repos

- `fracton` — Infodynamics SDK (provides PAC/Mobius primitives)
- `dawn-field-theory` — theoretical foundation (exp_28, exp_29, exp_36 feed gravity)
- `dawn-models` — AI architectures using same DFT principles
- `kronos-vault` — knowledge vault (FDOs: proj-reality-engine, reality-engine-dynamics, coupling-drift-physics)

## Theory Integration (2026-03-17)

13 spikes in `spikes/theory_integration/` bridging DFT theory corpus to simulator. Final synthesis: **10/10 quantitative predictions confirmed**.

Key findings:
- **PAC (global) vs SEC (local)**: PAC conserves at 2.06e-14; SEC drives local dynamics. Coupling constants are averages of turbulent local landscape — trajectory shapes universal, absolute values not.
- **Init independence**: Late-time trajectories correlate >0.997 regardless of init. Info-dominated converges fastest (tick 2300), entropy slowest (13950).
- **Emergent RG flow**: Coupling drift = renormalization group flow, not a bug. Beta functions init-independent. Gravity running 1.3x at high-z (JWST match).
- **SEC duty cycle**: Log-time mapping + info fraction proxy gives r=+0.954 correlation with theory.
- **"YOU ARE HERE" tick**: 8450/20000 = 42% lifecycle, 6.03% avg error across 5 Tier 1 constants.

## Current State

- v3 architecture, 18 operators, 138 tests
- Physics scorecard: 8/13 (C) — de-actualization completes PAC cycle, coupling drift reduced 24%
- Theory integration: 10/10 DFT predictions confirmed (spikes 09-13)
- 6 analyzers operational, PAC conservation validated
- Not accepting code contributions yet

## Guardrails

- Do NOT hardcode physics laws — all physics must emerge from field dynamics
- Do NOT break PAC conservation invariants (< 1e-12 error)
- Do NOT modify substrate geometry without understanding Mobius topology
- Always run `pytest tests/v3/` after changes
- spikes/ are research experiments — treat as exploratory, not production code
- Legacy dirs (core/, dynamics/, conservation/, etc.) are v1/v2 reference — do not modify
