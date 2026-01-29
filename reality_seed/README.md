# Reality Seed

A generative reality substrate using pure PAC dynamics.

## Overview

Reality Seed grows a universe from a singularity using only **Potential-Actualization Conservation (PAC)** - the principle that when potential becomes actual, value is conserved across splits.

**No physics is pre-encoded.** We just:
1. Start with a single node holding all value (singularity)
2. Split nodes conservatively (parent value = sum of children values)
3. Link nodes to form non-hierarchical structure
4. Inject entropy when system stagnates
5. Observe what patterns emerge
6. **Generate code** when patterns stabilize

## Key Findings

### φ Emerges Naturally

Through pure evolutionary dynamics, the golden ratio inverse (φ⁻¹ ≈ 0.618) **emerges without being encoded**.

The mechanism:
- Splits that produce deep lineages are "successful"
- Ancestor ratios that led to successful splits are remembered
- Future splits bias toward successful ratios
- Over many generations, φ⁻¹ emerges as the optimal ratio

### Patterns Self-Define

When structures stabilize (appear repeatedly with consistent properties), the system **generates Python code** that defines them:

```python
class Cluster_cluster_7078:
    """
    Auto-generated pattern definition.
    
    A cluster is a dense region of interconnected nodes.
    This pattern emerged from PAC dynamics - not pre-defined.
    
    Properties (observed, not assumed):
    - Size: 4 nodes
    - Density: 0.500
    - Total value: 0.012082
    - Internal edges: 3
    """
```

## Architecture

```
reality_seed/
├── __init__.py              # Exports
├── pac_substrate.py         # Core substrate: nodes, splits, links
├── genesis.py               # Generator and observer
├── visualizer.py            # Observation dashboard
├── patterns.py              # Pattern detection + code generation
├── discovered_patterns.py   # AUTO-GENERATED pattern definitions
└── README.md                # This file
```

### Core Components

| Component | Purpose |
|-----------|---------|
| `PACSubstrate` | Minimal substrate: nodes, values, conservation |
| `GenesisSeed` | Generator that runs PAC dynamics |
| `GenesisObserver` | Raw measurements without interpretation |
| `PatternDetector` | Finds clusters, hubs, chains, concentrations |
| `PatternCodeGenerator` | Generates Python code for stable patterns |
| `EmergenceAnalyzer` | Combines detection + generation |

### Pattern Types Detected

| Pattern | Description |
|---------|-------------|
| **Cluster** | Dense region of interconnected nodes |
| **Hub** | Node with unusually high connectivity |
| **Chain** | Linear sequence of nodes |
| **Concentration** | Region with high value density ("mass-like") |

## Usage

### Quick Start

```python
from reality_seed import run_genesis
run_genesis(n_steps=5000)
```

### Full Analysis with Pattern Detection

```python
from reality_seed import GenesisSeed, EmergenceAnalyzer

genesis = GenesisSeed(initial_value=1.0)
genesis.ratio_memory_weight = 0.5  # Evolutionary memory strength

analyzer = EmergenceAnalyzer(genesis)
results = analyzer.run_observation_cycle(
    steps_per_cycle=1000, 
    n_cycles=10
)

# Save auto-generated pattern definitions
analyzer.save_discoveries('discovered_patterns.py')

# Get stable patterns
stable = analyzer.detector.get_stable_patterns()
print(f"Found {len(stable)} stable patterns")
```

### Example Output

```
Cycle 1/10: 242 patterns, 0 stable, 0 new definitions
Cycle 2/10: 459 patterns, 0 stable, 0 new definitions
Cycle 3/10: 680 patterns, 4 stable, 4 new definitions
...
Cycle 10/10: 1974 patterns, 1212 stable, 190 new definitions

Pattern definitions generated: 1212
- Hubs: 449
- Clusters: 25
- Chains: 31
```

## Theoretical Basis

From Dawn Field Theory:

1. **PAC Conservation**: f(Parent) = Σf(Children)
2. **Self-Similarity**: The ratio that maintains proportion across scales is φ
3. **SEC Dynamics**: Structure forms where info gradient > entropy gradient
4. **Evolutionary Selection**: Ratios that maximize descendants win

## What This Demonstrates

1. **φ emergence**: Golden ratio appears from pure dynamics
2. **Self-organization**: Patterns form without being defined
3. **Code as crystallization**: Stable patterns "write themselves"
4. **Observer neutrality**: We measure, we don't assume

## Connection to Reality Engine

Reality Seed is the generative core. The larger Reality Engine project adds:

- Pre-field Möbius topology
- SEC phase dynamics
- Higher-dimensional projection

The goal: a reality that generates its own physics, its own laws, its own definitions.
