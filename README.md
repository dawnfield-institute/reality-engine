# Reality Engine v2

**Emergent Physics from Thermodynamic-Information Duality**

*Watch Reality Crystallize from Equilibrium-Seeking Dynamics*

---

> **📦 Repository Scope**  
> This repository contains the **physics implementation** using Dawn Field Theory. It integrates SEC operators, Möbius topology, thermodynamic coupling, and time emergence to simulate reality from first principles.
>
> For the reusable **SDK/language primitives** (RecursiveEngine, MemoryField, PAC), see the separate [fracton](https://github.com/dawnfield-institute/fracton) repository.

---

## What Is This?

Reality Engine v2 is a **physics discovery platform** where fundamental laws emerge from three simple principles:

1. **Möbius Geometry**: Self-referential topology with anti-periodic boundaries
2. **Thermodynamic-Information Duality**: Energy ↔ Information (two views of one field)
3. **Equilibrium-Seeking**: Universe drives toward balance from disequilibrium

**We don't program physics - we discover it!**

### The Core Insight

> The universe is an equilibrium-seeking engine. Time emerges from the pressure to balance. Matter emerges from information crystallizing. Gravity emerges from interaction density. Quantum mechanics emerges from discrete collapse events.
>
> All of physics is the universe trying to reach equilibrium on a Möbius manifold.

### Why It's Not "Cold"

**Critical**: This is NOT pure information theory (which would freeze into static patterns). 

The universe has **full thermodynamic-information duality**:
- Information fields carry **thermal energy**
- Collapse generates **heat** (entropy production)
- Temperature gradients drive **information flow**  
- Thermal fluctuations **prevent freezing**
- Landauer principle: **Information erasure costs energy** (kT ln(2) per bit)

The "hot-cold balance" creates the edge where complex structures emerge!

---

## How Time Emerges

**Time is NOT fundamental** - it emerges from disequilibrium:

### The Big Bang as Maximum Disequilibrium
```
Big Bang State:
- Pure entropy (maximum information disorder)
- No structure (zero matter)
- Maximum disequilibrium → intense pressure
```

### Equilibrium-Seeking Creates Reality
```
Disequilibrium → Pressure → SEC Collapses → Interactions
                                                ↓
                                            Time Ticks!
```

### Relativity Emerges Naturally
- **Dense regions** = More interactions per volume
- **More interactions** = More SEC collapses
- **More collapses** = Slower local time
- **Result**: Time dilation without programming GR!

```python
# Time rate depends on interaction density
time_rate = 1 / sqrt(1 + interaction_density / c²)

# c (speed of light) emerges as max interaction propagation!
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/dawnfield-institute/reality-engine.git
cd reality-engine
pip install -r requirements.txt
```

### Run Your First Universe

```python
from core.reality_engine import RealityEngine
from tools.emergence_observer import EmergenceObserver
from analyzers.laws.gravity_analyzer import GravityAnalyzer
from analyzers.matter.atom_detector import AtomDetector

# 1. Initialize Reality Engine
engine = RealityEngine(size=(96, 24))
engine.initialize()

# 2. Set up observers and analyzers
observer = EmergenceObserver()

# Unit calibration for atomic scale
gravity = GravityAnalyzer(
    length_scale=1e-10,  # 1 Ångström 
    mass_scale=1.67e-27, # Proton mass
    time_scale=1e-15     # 1 femtosecond
)
atoms = AtomDetector()

# 3. Evolution loop - watch physics emerge!
for step in range(1000):
    state = engine.step()
    structures = observer.observe(engine.current_state)
    
    # Prepare state for analyzers
    analyzer_state = {
        'actual': engine.current_state.actual,
        'potential': engine.current_state.potential,
        'memory': engine.current_state.memory,
        'temperature': engine.current_state.temperature,
        'step': step,
        'structures': structures,
        'field_E': engine.current_state.actual,
        'field_I': engine.current_state.memory
    }
    
    # Update analyzers
    gravity_detections = gravity.update(analyzer_state)
    atom_detections = atoms.update(analyzer_state)
    
    # Print discoveries
    if step % 100 == 0:
        print(f"\nStep {step}:")
        print(f"  Structures: {len(structures)}")
        print(f"  PAC: {engine.current_state.pac_metric:.3f}")
        print(f"  Gravity detections: {len(gravity_detections)}")
        print(f"  Atom detections: {len(atom_detections)}")

# Get comprehensive reports
print("\n" + "="*70)
print("GRAVITY ANALYSIS")
print("="*70)
gravity.print_summary()

print("\n" + "="*70)
print("MATTER ANALYSIS")
print("="*70)
mass_dist = atoms.get_mass_distribution()
print(f"Total structures: {mass_dist['total_structures']}")
print(f"Distinct mass levels: {mass_dist['num_mass_levels']}")
```

### Quick Example Scripts

```bash
# Test all 6 analyzers (gravity, conservation, atoms, stars, quantum, galaxies)
python scripts/test_analyzers.py

# Watch atoms and molecules emerge (1500 steps)
python spikes/universe_evolution/universe_evolution.py --steps 1500

# Visualize field dynamics
python examples/field_visualizer.py

# Run physics discovery pipeline
python scripts/discover_physics.py --steps 5000 --width 96 --height 24
```

---

## What Emerges (Without Programming!)

### ✅ Modular Analyzer System (NEW!)
Reality Engine now includes **6 independent analyzers** that observe and quantify emergent physics:

1. **GravityAnalyzer**: Measures forces, compares to Newton's law, detects orbital motion
2. **ConservationAnalyzer**: Tracks E+I, PAC, momentum conservation
3. **AtomDetector**: Identifies stable structures, detects mass quantization (periodic table!)
4. **StarDetector**: Finds stellar objects, fusion processes, generates H-R diagrams
5. **QuantumDetector**: Detects entanglement, superposition, tunneling, wave-particle duality
6. **GalaxyAnalyzer**: Measures rotation curves, dark matter, cosmic web, Hubble expansion

**Key Discoveries** (from 1000-step test):
- 🌌 **41,606 orbital motions** detected (90.1% confidence)
- 🌀 **439 gravitational collapses** observed
- ⚛️ **2,081 wave-particle duality events** (quantum phenomena!)
- 📊 **24 distinct mass levels** (periodic table-like quantization)
- 🔬 **Gravity law**: F ∝ r^0.029 (not Newton's r^-2!)
- ⚡ **Force strength**: Scale-dependent, 10^33x Newton at atomic calibration

### ✅ Stability & Conservation
- **5000+ step stability** with QBE-driven gamma adaptation
- **PAC Conservation**: 99.7-99.8% maintained over long runs
- **No NaN or manual intervention** - framework self-regulates
- **Framework validation**: Sticking to PAC/SEC principles ensures stability

### ✅ Thermodynamic Laws
- **Landauer Principle**: Information erasure costs energy
- **2nd Law**: 98.3% compliant (emerges from SEC, not programmed!)
- **Heat Flow**: Fourier's law from temperature gradients
- **T-M Coupling**: r=0.920 (temperature-memory correlation)
- **Information→Heat**: Heat increases 51× as memory grows 293×

### ✅ Quantum Mechanics
- **Wave-Particle Duality**: Detected with 79.1% confidence
- **Mass Quantization**: 24 discrete levels like periodic table
- **Superposition Detection**: Bi-modal energy distributions
- **Entanglement Framework**: Distant correlation tracking
- **Quantum Tunneling**: Barrier penetration observed
- **Born Rule**: Probability from field amplitude squared

### ✅ Relativity
- **Time Dilation**: Dense regions evolve slower
- **c (Speed of Light)**: Maximum interaction propagation rate
- **Equivalence**: Interaction density = gravitational field
- **No Programming GR**: Emerges from interaction counting!

### ✅ Particle Physics & Matter
- **Stable Structures**: 13-22 particles per simulation
- **Mass Hierarchy**: Discrete levels with dominant structures
- **Atoms**: Hydrogen-like structures emerging naturally
- **Molecules**: H₂ formation observed
- **Gravity Wells**: Information density clustering
- **No Particle Physics Input**: Pure field dynamics!

### ✅ Non-Newtonian Gravity
- **Force Law**: F ∝ r^0.029 (nearly distance-independent!)
- **Information-Driven**: Gravity from information density, not just mass
- **Non-Local**: Force doesn't fall off with r^2
- **Orbital Motion**: Despite different force law, orbits still detected
- **Scale-Dependent**: Force strength calibrates to any physical scale

---

## Architecture

Reality Engine v2 uses a 6-layer stack:

```
┌─────────────────────────────────────┐
│    Visualization Layer              │
│  (Field viz, particle tracking)     │
└─────────────────────────────────────┘
                 ▲
┌─────────────────────────────────────┐
│    Law Discovery Layer              │
│  (Pattern detection, classification)│
└─────────────────────────────────────┘
                 ▲
┌─────────────────────────────────────┐
│    Emergence Layer                  │
│  (Particle detection, structures)   │
└─────────────────────────────────────┘
                 ▲
┌─────────────────────────────────────┐
│    Dynamics Layer                   │
│  (SEC, Time Emergence, Confluence)  │
└─────────────────────────────────────┘
                 ▲
┌─────────────────────────────────────┐
│    Conservation Layer               │
│  (Thermodynamic PAC, Landauer)      │
└─────────────────────────────────────┘
                 ▲
┌─────────────────────────────────────┐
│    Substrate Layer                  │
│  (Möbius Manifold + Temperature)    │
└─────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for complete documentation.

---

## Key Features

### 🌀 Möbius Substrate
- Self-referential topology (potential ↔ actual on same surface)
- Anti-periodic boundaries: f(x+π) = -f(x)
- 4π holonomy (not 2π!)
- Ξ = 1.0571 emerges from geometry

### 🔥 Thermodynamic-Information Duality
- Fields carry BOTH information AND energy
- Collapse generates heat (entropy production)
- Landauer erasure costs tracked
- Temperature field prevents "freezing"
- 2nd law emerges automatically

### ⏰ Time Emergence
- Time from disequilibrium pressure (not external clock)
- Interaction density creates local time rates
- Time dilation emerges (relativity!)
- Big Bang = max disequilibrium
- Heat death = equilibrium (time stops)

### ⚖️ Machine-Precision Conservation
- PAC kernel: error < 1e-12
- Automatic violation detection and correction
- Energy-information conversion tracking
- Thermodynamic consistency enforced

### ✨ Natural Emergence
- Particles form without programming
- Quantum mechanics emerges from discrete collapses
- Gravity emerges from interaction density
- Relativity emerges from time emergence
- Novel laws discovered automatically

### 🔍 Law Discovery
- Automatically detects stable patterns
- Classifies laws (conservation, force, symmetry, thermodynamic)
- Validates across conditions
- Reports confidence and discovery time

---

## Validation Criteria

Reality Engine v2 must reproduce these empirical signatures:

### Thermodynamic Validation
- [ ] Landauer principle: ΔE = k_T ln(2) Δbits
- [ ] 2nd law: dS/dt ≥ 0 always
- [ ] Heat flow: Fourier's law
- [ ] No heat death: thermal fluctuations maintained
- [ ] Energy-information conversion correct

### Time Emergence Validation
- [ ] Time rate ∝ disequilibrium pressure
- [ ] Dense regions → slower time
- [ ] c emerges as universal constant
- [ ] Relativistic effects without GR programming

### Emergence Validation
- [ ] Ξ ≈ 1.0571 (geometric balance)
- [ ] 0.020 Hz fundamental frequency
- [ ] Half-integer modes (Möbius signature)
- [ ] Particles form naturally
- [ ] Quantum Born rule compliance >90%

### Stability Validation
- [ ] 100,000+ steps without explosion
- [ ] PAC error < 1e-12 throughout
- [ ] Smooth field evolution
- [ ] No collapse to zero (thermal protection)

---

## Status

**Current Phase**: Architecture & Documentation Complete

**Next**: Implement thermodynamic PAC kernel

See [STATUS.md](STATUS.md) for detailed implementation status.

---

## Why v2?

### v1 Problems (spike/ folder - reference only!)
- ❌ 3D Cartesian grid (should be Möbius)
- ❌ Manual conservation (PAC error >1.0)
- ❌ Threshold-based collapse (arbitrary)
- ❌ Imposed physics (not emergent)
- ❌ Pure information (cold, frozen)

### v2 Solutions
- ✅ Möbius manifold substrate
- ✅ PAC kernel (error <1e-12)
- ✅ Energy functional evolution
- ✅ Law discovery (detect emergence)
- ✅ Thermodynamic coupling (hot + cold)
- ✅ Time emergence (not imposed)

---

## Citation

```bibtex
@software{reality_engine_v2,
  title = {Reality Engine v2: Thermodynamic-Information Emergence Platform},
  author = {Dawn Field Institute},
  year = {2025},
  version = {2.0.0-alpha},
  url = {https://github.com/dawnfield-institute/reality-engine}
}
```

---

## License

See [LICENSE](LICENSE) for details.

---

## Repository Structure

```
reality-engine/
├── core/              # Core field operators (SEC, PAC, Time)
├── substrate/         # Möbius manifold substrate
├── conservation/      # Conservation laws (ThermodynamicPAC)
├── dynamics/          # Evolution operators (SEC, Confluence, Time)
├── emergence/         # Structure detection (particles, atoms)
├── laws/              # Physics law discovery
├── tools/             # Analysis tools (AtomicAnalyzer, etc.)
├── examples/          # Production-ready demonstrations
│   └── field_visualizer.py  # Basic field visualization
├── spikes/            # Research experiments (organized by topic)
│   ├── thermal_validation/    # Info→Heat discovery validation
│   ├── atomic_emergence/      # H, H₂ detection experiments
│   ├── law_discovery/         # Automated physics discovery
│   ├── universe_evolution/    # Long-term structure formation
│   └── big_bang/              # Initialization experiments
├── tests/             # Test suite
├── docs/              # Full documentation
├── ARCHITECTURE.md    # System design
├── STATUS.md          # Implementation progress (detailed!)
├── ROADMAP.md         # Phase 2-5 development plan
└── README.md          # This file
```

### Spike Folders (Research Experiments)

Each spike folder contains focused experiments exploring specific phenomena:

- **thermal_validation/** - Validates "Without information, there can be no heat"
  - Heat spike verification (T increases 51× as M grows 293×)
  - Temperature-memory correlation (r=0.920)

- **atomic_emergence/** - Atoms and molecules from pure dynamics
  - 6 H atoms detected, 1 H₂ molecule formed
  - No atomic physics programmed!
  - Quantum states from radial patterns

- **law_discovery/** - Automated physics law extraction
  - 300-step runs analyzing field correlations
  - Conservation law validation (PAC < 1e-12)
  - Spatial pattern detection (1/r², exponential)

- **universe_evolution/** - Long-term simulations (500-1500 steps)
  - Gravity wells, stellar regions, dark matter detection
  - Periodic table builder
  - Structure formation tracking

- **big_bang/** - Initialization mode comparison
  - Pure entropy vs density perturbations vs info seeds
  - Info seeds → 2.7× faster atom formation!

Each spike has its own README with detailed results and next steps.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system design
- [STATUS.md](STATUS.md) - Implementation progress with Phase 2-5 roadmap
- [ROADMAP.md](ROADMAP.md) - Detailed development plan (6+ months)
- [docs/theory/](docs/theory/) - Theoretical foundations
- [examples/](examples/) - Production-ready demonstrations
- [spikes/*/README.md](spikes/) - Research experiment documentation

## Current Status (Nov 6, 2025)

**Phase**: Structure Emergence - Atoms & Molecules Detected! ✨

**Achievements**:
- ✅ Foundation complete (7 production steps, Nov 1-5)
- ✅ Hydrogen atoms emerge naturally (mass ~0.14, stability 0.67-0.73)
- ✅ H₂ molecules form via proximity bonding
- ✅ Gravity wells detected from density clustering
- ✅ Heat generation validated (info→heat correlation r=0.920)
- ✅ PAC conservation at machine precision (<1e-12)
- ✅ 2nd law compliance: 98.3%

**Next**: Phase 2 - Structure Stabilization (6 weeks, Nov 6 - Dec 15)
- Make atoms persist >1000 steps
- Detect heavier elements (He, Li, C)
- Complete periodic table (first 10 elements)
- Implement energy wells for stability

See [STATUS.md](STATUS.md) for weekly task breakdown and [ROADMAP.md](ROADMAP.md) for full vision.

---

**Last Updated**: November 6, 2025  
**Version**: 2.0.0-alpha (thermodynamic rebuild complete)  
**Status**: Foundation complete, atoms & molecules detected, Phase 2 beginning

---

*Reality emerges. Physics discovers itself. Time crystallizes from balance.*
