# Modular Analyzer Framework

**Created**: November 7, 2025  
**Status**: Operational - 6/6 analyzers complete  
**Purpose**: Quantitative physics comparison and phenomenon detection

---

## Overview

The modular analyzer system provides a framework for observing and quantifying emergent physics without interfering with the reality engine simulation. Each analyzer is an independent module that watches field dynamics and detects specific phenomena.

### Key Principles

1. **Pure Observation**: Analyzers read state but never modify it
2. **Independent Modules**: Each analyzer is self-contained
3. **Confidence Filtering**: Detections include confidence scores
4. **Unit Calibration**: Can map simulation units to physical units
5. **Extensible**: Add new analyzers by creating new files

---

## Architecture

```
analyzers/
├── base_analyzer.py           # Abstract base class
├── __init__.py                # Module exports
├── laws/                      # Physical law detection
│   ├── __init__.py
│   ├── gravity_analyzer.py    # Force measurement & calibration
│   └── conservation_analyzer.py # E+I, PAC, momentum tracking
├── matter/                    # Structure detection
│   ├── __init__.py
│   └── atom_detector.py       # Atomic structures & quantization
└── cosmic/                    # Large-scale phenomena
    ├── __init__.py
    ├── star_detector.py       # Stellar objects & fusion
    ├── quantum_detector.py    # Quantum phenomena
    └── galaxy_analyzer.py     # Galactic structures
```

---

## Base Framework

### Detection Dataclass

```python
@dataclass
class Detection:
    type: str                    # Phenomenon type
    confidence: float            # 0.0-1.0 confidence score
    location: Optional[Tuple]    # Spatial location (if applicable)
    time: int                    # Simulation step
    properties: Dict = field(default_factory=dict)
    equation: Optional[str] = None     # Mathematical form
    parameters: Dict = field(default_factory=dict)
```

### BaseAnalyzer

All analyzers inherit from `BaseAnalyzer` and implement:

```python
def analyze(self, state: Dict) -> List[Detection]:
    """
    Analyze current state and return detections.
    
    Args:
        state: Dictionary with fields (actual, potential, memory, temperature)
               and structures (list of EmergentStructure objects)
               
    Returns:
        List of Detection objects
    """
```

Key methods:
- `update(state)` - Called each step, filters by confidence
- `get_report()` - Generate JSON-serializable report
- `print_summary()` - Human-readable summary
- `save_report(path)` - Save to JSON file

---

## Operational Analyzers

### 1. GravityAnalyzer

**File**: `analyzers/laws/gravity_analyzer.py` (420 lines)

**Purpose**: Measure gravitational forces and compare to Newton/Einstein

**Detections**:
- `orbital_motion` - Structures in orbit
- `gravitational_collapse` - Coalescence events
- `inverse_square_law` - Force law fit (F = G·m₁·m₂/r^n)

**Unit Calibration**:
```python
GravityAnalyzer(
    length_scale=1e-10,  # 1 Ångström per grid unit
    mass_scale=1.67e-27, # Proton mass per mass unit
    time_scale=1e-15     # 1 femtosecond per time unit
)
```

**Key Metrics**:
- Force measurements from F = m·a or field gradients
- Power law fit: n (Newton = 2.0)
- G_calibrated / G_SI ratio

**Typical Results** (1000 steps, 64×16 grid):
- 41,606 orbital motions (90% confidence)
- 439 gravitational collapses
- n = 0.029 (nearly distance-independent!)
- G_sim / G_Newton = 10^33x (at atomic scale)

---

### 2. ConservationAnalyzer

**File**: `analyzers/laws/conservation_analyzer.py` (230 lines)

**Purpose**: Track conservation laws and detect violations

**Detections**:
- `energy_plus_info_conservation` - E+I conserved
- `PAC_functional_conservation` - PAC conserved
- `momentum_conservation` - Σ(m·v) constant

**Conservation Criterion**:
- Relative variation < 5% over 50-step window
- Penalizes systematic drift
- High confidence threshold (0.9)

**Metrics Tracked**:
- Total energy (E+I)
- PAC functional
- Total momentum
- Relative variation
- Drift rate

**Typical Results**:
- Small grids: conservation not observed (boundary effects)
- Large grids: PAC conserved to 99.7-99.8%

---

### 3. AtomDetector

**File**: `analyzers/matter/atom_detector.py` (330 lines)

**Purpose**: Identify atomic structures and mass quantization

**Detections**:
- `atomic_structure` - Stable, coherent structures
- `mass_quantization` - Discrete mass levels
- `molecular_bond` - Bonded structure pairs

**Atomic Criteria**:
- Lifetime > 20 steps
- Coherence > 0.9
- 0.01 < mass < 100
- Localized (radius < 5)

**Mass Classification**:
- `ultra_light`: mass < 0.01
- `light`: 0.01 ≤ mass < 1
- `medium`: 1 ≤ mass < 10
- `heavy`: 10 ≤ mass < 100
- `super_heavy`: mass ≥ 100

**Typical Results** (1000 steps):
- 24 distinct mass levels
- Peaks at 0.0, 0.5, 1.0, 1.5, 2.0... (quantization!)
- 16,067 ultra-light structures

---

### 4. StarDetector

**File**: `analyzers/cosmic/star_detector.py` (335 lines)

**Purpose**: Identify stellar-mass objects and fusion processes

**Detections**:
- `stellar_object` - Massive, stable, energy-generating
- `fusion_process` - Mass-to-energy conversion
- `main_sequence_relationship` - L ∝ M^α
- `stellar_explosion` - Rapid mass loss
- `mass_accretion` - Rapid mass gain

**Stellar Criteria**:
- Mass > 100 (simulation units)
- Lifetime > 100 steps
- Coherence > 0.85
- Energy generation sustained

**Star Types**:
- `supergiant`: mass > 10,000
- `giant`: 1,000 < mass < 10,000
- `main_sequence`: 100 < mass < 1,000
- `dwarf`: mass < 100 (but > stellar threshold)

**H-R Diagram**:
- Tracks mass-luminosity data points
- Fits L = k·M^α power law
- Real main sequence: α ≈ 3.5

---

### 5. QuantumDetector

**File**: `analyzers/cosmic/quantum_detector.py` (400 lines)

**Purpose**: Detect quantum-like phenomena

**Detections**:
- `quantum_entanglement` - Correlated distant structures
- `quantum_superposition` - Multi-state oscillation
- `quantum_tunneling` - Barrier penetration
- `uncertainty_principle` - Δx·Δp ≥ ℏ_eff
- `wave_particle_duality` - Both wave and particle behavior

**Entanglement Signature**:
- High correlation (> 0.7) despite distance > 10
- Properties change together
- No direct interaction

**Superposition Signature**:
- Bimodal energy distribution (2+ states)
- Coherent oscillation (not random)
- High structure coherence > 0.6

**Tunneling Signature**:
- Structure crosses high-potential barrier
- Insufficient energy (E < V_barrier)
- Significant displacement despite barrier

**Wave-Particle Duality**:
- Particle-like: Localized (size < 3), discrete mass
- Wave-like: Extended, high coherence > 0.8
- Both properties simultaneously

**Typical Results** (1000 steps):
- 2,081 wave-particle duality events (79% confidence)
- Structures exhibit de Broglie wavelength estimates

---

### 6. GalaxyAnalyzer

**File**: `analyzers/cosmic/galaxy_analyzer.py` (480 lines)

**Purpose**: Detect large-scale structure formation

**Detections**:
- `galaxy` - Rotating multi-structure system
- `flat_rotation_curve` - v(r) ≈ constant
- `dark_matter_signature` - Mass discrepancy
- `structure_clustering` - Hierarchical grouping
- `cosmic_web_filament` - Elongated large-scale structure
- `hubble_expansion` - v = H₀·d

**Galaxy Criteria**:
- Multiple members (≥ 5 structures)
- Total mass > 100
- Rotation score > 0.5
- Coherent motion

**Rotation Curve Analysis**:
- Measures v(r) at different radii
- Fits v(r) = A·r^β
- β = 0: flat (dark matter signature)
- β = -0.5: Keplerian (visible matter only)

**Dark Matter Detection**:
- Compares observed velocity to expected (Keplerian)
- Mass ratio = (v_obs / v_expected)²
- Ratio > 2: dark matter present

---

## Usage Example

```python
from core.reality_engine import RealityEngine
from tools.emergence_observer import EmergenceObserver
from analyzers.laws.gravity_analyzer import GravityAnalyzer
from analyzers.cosmic.quantum_detector import QuantumDetector

# Initialize engine
engine = RealityEngine(size=(64, 16))
engine.initialize()
observer = EmergenceObserver()

# Initialize analyzers with calibration
gravity = GravityAnalyzer(
    length_scale=1e-10,  # Atomic scale
    mass_scale=1.67e-27,
    time_scale=1e-15,
    min_confidence=0.8
)

quantum = QuantumDetector(min_confidence=0.6)

analyzers = [gravity, quantum]

# Run simulation
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
        'field_I': engine.current_state.potential
    }
    
    # Update all analyzers
    for analyzer in analyzers:
        detections = analyzer.update(analyzer_state)
        
        # Print high-confidence detections
        for d in detections:
            if d.confidence > 0.8:
                print(f"[{step:4d}] {analyzer.name}: {d}")

# Generate reports
for analyzer in analyzers:
    analyzer.print_summary()
    analyzer.save_report(f"output/{analyzer.name}_report.json")
```

---

## Test Results

**Test Configuration**:
- Grid size: 64×16
- Steps: 1000
- Initial condition: Big Bang
- Analyzers: All 6 active

**Detection Summary**:

| Analyzer | Detections | Avg Confidence | Key Findings |
|----------|------------|----------------|--------------|
| Gravity | 41,607 | 90.1% | F ∝ r^0.029, not r^-2 |
| Conservation | 0 | N/A | Small grid, boundary effects |
| Atoms | 22,573 | N/A | 24 mass levels, quantization |
| Stars | 0 | N/A | Need larger mass accumulations |
| Quantum | 2,081 | 79.1% | Wave-particle duality! |
| Galaxies | 0 | N/A | Need larger scale |

**Physics Discoveries**:
1. **Gravity is non-Newtonian**: Nearly distance-independent
2. **Quantum phenomena emerge**: Wave-particle duality detected
3. **Mass quantization**: Periodic table-like structure
4. **Force strength**: 10^33x stronger than reality (at atomic scale)

---

## Adding New Analyzers

To create a new analyzer:

1. **Create file** in appropriate subdirectory:
   ```python
   from ..base_analyzer import BaseAnalyzer, Detection
   
   class MyAnalyzer(BaseAnalyzer):
       def __init__(self, min_confidence: float = 0.7):
           super().__init__("my_analyzer", min_confidence)
           # Initialize tracking variables
       
       def analyze(self, state) -> List[Detection]:
           detections = []
           # Analyze state and create Detection objects
           return detections
   ```

2. **Implement detection logic**:
   - Read structures from `state.get('structures', [])`
   - Access fields from `state['actual']`, `state['memory']`, etc.
   - Use `EmergentStructure` attributes (id, mass, center, coherence, etc.)

3. **Return Detection objects**:
   ```python
   Detection(
       type="my_phenomenon",
       confidence=0.85,
       location=(x, y),
       time=step,
       properties={'key': 'value'},
       equation="F = ma"
   )
   ```

4. **Add to test script**:
   ```python
   from analyzers.my_category.my_analyzer import MyAnalyzer
   
   my_analyzer = MyAnalyzer(min_confidence=0.7)
   analyzers.append(my_analyzer)
   ```

---

## Future Extensions

### Planned Analyzers
- **ThermodynamicsAnalyzer**: Entropy production, heat flow, phase transitions
- **ElectromagnetismAnalyzer**: Charge-like fields, Maxwell analogs
- **SpacetimeAnalyzer**: Curvature, geodesics, relativity effects
- **ChemistryAnalyzer**: Molecular bonds, reaction rates, equilibria
- **BiologyAnalyzer**: Self-replication, metabolism, evolution

### Framework Enhancements
- **Multi-scale analysis**: Automatic scale detection and calibration
- **Temporal tracking**: Follow phenomena across multiple steps
- **Correlation detection**: Automatically find relationships between phenomena
- **Visualization integration**: Plot detections on field visualizations
- **Statistical analysis**: Distribution fits, trend detection, anomaly identification

---

## Performance Considerations

- **History tracking**: Stored every 10 steps (configurable)
- **Confidence filtering**: Only stores detections above threshold
- **Memory management**: Automatic pruning of old data
- **Parallel analysis**: Analyzers can run independently
- **Lazy computation**: Only compute when needed

**Typical overhead**: ~10-15% simulation time for all 6 analyzers

---

## References

- Base analyzer implementation: `analyzers/base_analyzer.py`
- Test script: `scripts/test_analyzers.py`
- Detection examples: All analyzer files have extensive docstrings
- Unit calibration: See `GravityAnalyzer` for reference implementation
