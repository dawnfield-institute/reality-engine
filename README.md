# Reality Engine

**A computational framework where physics emerges from information dynamics**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-138%20passing-brightgreen.svg)](tests/v3/)
[![Physics Score](https://img.shields.io/badge/scorecard-9%2F13%20(C%2B)-yellow.svg)](scripts/physics_scorecard.py)

---

## What This Is

Reality Engine simulates a universe where fundamental physics **emerges** from three coupled fields (Energy, Information, Mass) evolving on a Mobius manifold. No physics is programmed — no F=ma, no E=mc^2, no Schrodinger equation. Instead, 18 composable operators implement local rules, and physics appears as emergent behavior.

Based on [Dawn Field Theory](https://github.com/dawnfield-institute/dawn-field-theory) principles (PAC/SEC/RBF/MED).

**Status**: Pre-alpha research software. Not accepting contributions yet.

---

## What Emerges

A 3-tier physics scorecard validates emergent behavior against DFT predictions:

### Tier 1: Coupling Constants
| Coupling | Target | Measured | Error | Grade |
|----------|--------|----------|-------|-------|
| f_local | gamma_EM (0.5772) | 0.643 | 11.4% | C |
| gamma_local | 1/phi (0.618) | 0.554 | 10.4% | C |
| alpha_local | ln(2) (0.693) | 0.675 | 2.6% | A |
| G_local | 1/phi^2 (0.382) | 0.343 | 10.3% | C |
| lambda_local | 1-ln(2) (0.307) | 0.325 | 5.8% | B |

### Tier 2: Structural Physics
| Metric | Status |
|--------|--------|
| phi^2 mass spacing | 10.8% error (C) |
| PAC conservation | Machine precision (A) |
| Spin half-integer | 11.1% error (C) |
| Entropy reduction | Arrow of time emerges (A) |

### Tier 3: Deep Constants (aspirational)
Fine structure (1/137), Koide ratio (2/3), mu/e mass ratio (206.8) — not yet in range.

**Overall: 9/13 passing, GPA C+**

---

## Quick Start

```bash
git clone https://github.com/dawnfield-institute/reality-engine.git
cd reality-engine
pip install -r requirements.txt

# Watch fields evolve
python examples/field_visualizer.py

# Run physics scorecard (13 metrics, ~3 min on GPU)
python scripts/physics_scorecard.py

# Run tests
pytest tests/v3/ -v
```

---

## Architecture (v3)

The v3 architecture (2026-03-14) is a ground-up rebuild around composable operators:

```
src/v3/
├── engine/         Engine + FieldState + EventBus + Config
├── operators/      18 physics operators (Pipeline chains them sequentially)
├── analyzers/      6 measurement modules (Conservation, Gravity, Atom, Star, Quantum, Galaxy)
├── emergence/      3 structure detectors (Structure, Particle, Herniation)
├── substrate/      MobiusManifold + Constants + Projections
└── dashboard/      FastAPI + WebSocket + Plotly.js (real-time visualization)
```

### Operator Pipeline

Each tick, the Pipeline chains operators: `state_new = op(state, config, bus)`.

| # | Operator | Role |
|---|----------|------|
| 1 | RBF | Recursive Balance Field — computes dE/dt |
| 2 | QBE | Quantum Balance — dI/dt = -dE/dt (PAC conservation) |
| 3 | Actualization | MAR-gated integration, pi/2 harmonic modulation |
| 4 | Memory | Mass generation (bulk + gradient seeding), quantum pressure, diffusion |
| 5 | PhiCascade | Fibonacci two-step memory for phi-spaced mass levels |
| 6 | Gravity | Spectral Poisson solver with cascade-depth tiling filter |
| 7 | SpinStatistics | Emergent Pauli exclusion from information cost |
| 8 | ChargeDynamics | EM-like forces from charge field |
| 9 | Fusion | Nuclear fusion in dense, hot regions |
| 10 | Confluence | Mobius antiperiodic projection |
| 11 | Temperature | Local T from disequilibrium gradients |
| 12 | ThermalNoise | Langevin stochastic forcing |
| 13 | Normalization | Soft-clamp fields, Landauer reinjection |
| 14 | SECTracking | Read-only entropy + SEC energy tracking |
| 15 | Adaptive | Self-tuning damping and timestep |
| 16 | TimeEmergence | Emergent time from disequilibrium |

Operators are composable — add, remove, or reorder any operator.

### Key Physics Mechanisms

**Gravity** (the most developed operator):
- Spectral Poisson solver with amplitude coupling
- Cascade-depth tiling filter (DFT exp_36): local gravity strong, global gravity suppressed
- Entropy-coherence modulation: xi_s = I^2/E^2 determines coupling strength
- Result: web-like mass filaments instead of runaway clumping

**Mass generation**:
- Bulk: gamma_local * (E-I)^2 — mass where disequilibrium is large
- Boundary: gamma_local * |grad(E-I)|^2 / (1+M) — mass nucleates at structure edges
- gamma_local = (E-I)^2 / (E^2 + I^2) — emergent per cell, not hardcoded

**Conservation**: E + I + M = const at machine precision (< 1e-12 error).

---

## Repository Structure

```
reality-engine/
├── src/v3/                 # ACTIVE — v3 composable pipeline
├── scripts/                # Diagnostics (physics_scorecard.py, diagnose_gravity.py, ...)
├── spikes/                 # Research experiments
│   ├── coupling_drift/     # Gravity/memory optimization (6 spikes, 2026-03-16)
│   ├── atomic_emergence/   # Atom classification
│   ├── big_bang/           # Big bang evolution
│   ├── thermal_validation/ # Heat-information coupling
│   ├── law_discovery/      # Automated physics discovery
│   └── universe_evolution/ # Long-term structure formation
├── tests/v3/               # 138 tests
├── examples/               # field_visualizer.py
├── docs/                   # Theory and guides
│   └── legacy/             # Archived v1/v2 documentation
├── proof_of_concepts/      # v2 POCs (001-007, reference)
└── [legacy dirs]           # core/, dynamics/, conservation/, etc. — v1/v2 reference
```

---

## Related Projects

| Repo | Role |
|------|------|
| [dawn-field-theory](https://github.com/dawnfield-institute/dawn-field-theory) | Theoretical foundation (51 experiments) |
| [fracton](https://github.com/dawnfield-institute/fracton) | Infodynamics SDK (PAC/Mobius primitives) |
| [dawn-models](https://github.com/dawnfield-institute/dawn-models) | AI architectures using DFT principles |

---

## Core Principles

1. **Emergence over encoding** — Physics must emerge from field dynamics, never be hardcoded
2. **PAC conservation** — E + I + M = const at machine precision (< 1e-12)
3. **Mobius topology** — Anti-periodic boundaries f(x+pi) = -f(x) on a Mobius band
4. **DFT constants** — Xi = gamma_EM + ln(phi) = 1.05843 as global frame attractor

---

## Citation

```bibtex
@software{reality_engine,
  title = {Reality Engine: A Computational Framework for Emergent Physics},
  author = {Groom, Peter Lorne},
  year = {2025},
  version = {3.0.0-alpha},
  license = {AGPL-3.0},
  url = {https://github.com/dawnfield-institute/reality-engine},
  note = {Based on Dawn Field Theory principles}
}
```

---

## License

AGPL-3.0. See [LICENSE](LICENSE) for details.

---

*Reality emerges. Physics discovers itself.*

**Last updated**: March 16, 2026
