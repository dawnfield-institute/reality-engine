# Reality Engine -- Claude Code Context

## Identity
Reality Engine is a computational physics framework where fundamental physics emerges from information dynamics. Three fields (energy, potential, information) plus local rules generate quantum mechanics, gravity, thermodynamics, and relativity without programming them. Version 2.0.0-alpha, pre-alpha research software based on Dawn Field Theory principles.

## Architecture

```
reality-engine/
├── core/                   # Core field operators
│   ├── reality_engine.py   # RealityEngine main class
│   ├── dawn_field.py       # Dawn field implementation
│   ├── pac_rescaler.py     # PAC rescaling
│   └── adaptive_parameters.py
├── dynamics/               # Evolution operators
│   ├── confluence.py       # Confluence dynamics
│   ├── klein_gordon.py     # Klein-Gordon evolution
│   ├── time_emergence.py   # Time from disequilibrium
│   ├── feigenbaum_detector.py
│   └── resonance_detector.py
├── conservation/           # Conservation laws
│   ├── thermodynamic_pac.py # ThermodynamicPAC kernel
│   ├── pac_recursion.py    # PAC recursion
│   └── sec_operator.py     # SEC operator
├── emergence/              # Structure detection
│   ├── particle_analyzer.py
│   ├── structure_analyzer.py
│   ├── stellar_analyzer.py
│   └── herniation_detector.py
├── analyzers/              # 6 independent physics analyzers
│   ├── laws/               # GravityAnalyzer, ConservationAnalyzer
│   ├── matter/             # AtomDetector
│   └── cosmic/             # StarDetector, QuantumDetector, GalaxyAnalyzer
├── cosmology/              # Cosmological predictions
│   ├── pac_cosmology.py    # PAC-based cosmology
│   ├── jwst_predictions.py # JWST observables
│   └── entropic_time_dilation.py
├── experiments/            # Numbered validation experiments (exp_01..exp_05)
├── dashboard/              # Web dashboard (HTML + Python server)
├── spikes/                 # Research experiments by topic
├── tests/                  # Test suite
├── docs/                   # Theory and documentation
├── examples/               # Production-ready demos
└── scripts/                # Discovery and analysis scripts
```

## Key Features

**Core**: RealityEngine (main simulation loop), DawnField (3-field substrate)
**Dynamics**: SEC collapse, Klein-Gordon evolution, time emergence, confluence
**Conservation**: ThermodynamicPAC (error < 1e-12), Landauer principle enforcement
**Analyzers**: Gravity, Conservation, Atom, Star, Quantum, Galaxy (6 independent modules)
**Cosmology**: JWST predictions, entropic time dilation, herniation mechanism

## Conventions

- Physics must EMERGE, never be programmed -- no hardcoded F=ma, E=mc2, etc.
- PAC conservation enforced at machine precision (< 1e-12)
- Mobius manifold substrate with anti-periodic boundaries: f(x+pi) = -f(x)
- Tests: `pytest tests/` from repo root
- Installation: `pip install -r requirements.txt`
- Run quick demo: `python examples/field_visualizer.py`

## Related Repos

- `fracton` -- Infodynamics SDK (provides PAC/Mobius primitives imported here)
- `dawn-field-theory` -- theoretical foundation
- `dawn-models` -- AI architectures using same DFT principles
- `kronos-vault` -- knowledge vault

## Current State

- v2.0.0-alpha, pre-alpha research software
- Foundation complete: atoms, molecules, gravity wells detected
- 6 analyzers operational, PAC conservation validated
- Phase 2 (structure stabilization) in progress
- Not accepting code contributions yet

## Guardrails

- Do NOT hardcode physics laws -- all physics must emerge from field dynamics
- Do NOT break PAC conservation invariants (< 1e-12 error)
- Do NOT modify substrate geometry without understanding Mobius topology
- Always run `pytest tests/` after changes
- spikes/ are research experiments -- treat as exploratory, not production code
