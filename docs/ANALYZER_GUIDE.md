# Analyzer System Guide

**Reality Engine v2 - Modular Physics Detection Framework**

---

## Overview

The analyzer system provides a modular framework for observing and quantifying emergent physics without interfering with the simulation. Each analyzer is an independent module that:

1. **Observes** field states and emergent structures
2. **Detects** specific physical phenomena with confidence scores
3. **Reports** findings in a structured, analyzable format
4. **Calibrates** measurements to real physical units

**Key Principle**: Analyzers are **pure observers** - they never modify the engine state.

---

## Architecture

```
analyzers/
├── base_analyzer.py           # Abstract base class + Detection dataclass
├── __init__.py                # Exports BaseAnalyzer, Detection
├── laws/                      # Physical law detection
│   ├── gravity_analyzer.py    # Force measurement & calibration
│   └── conservation_analyzer.py # E+I, PAC, momentum tracking
├── matter/                    # Structure detection
│   └── atom_detector.py       # Atomic structures & mass quantization
└── cosmic/                    # Large-scale phenomena
    ├── star_detector.py       # Stellar objects & fusion
    ├── quantum_detector.py    # Quantum phenomena
    └── galaxy_analyzer.py     # Galactic structures
```

---

## Core Classes

### Detection Dataclass

Every detected phenomenon is represented as a `Detection`:

```python
@dataclass
class Detection:
    type: str                          # Phenomenon type (e.g., "orbital_motion")
    confidence: float                  # 0.0-1.0 confidence score
    location: Optional[Tuple[float, float]]  # Spatial location (if applicable)
    time: int                          # Simulation step
    properties: Dict[str, Any] = field(default_factory=dict)  # Additional data
    equation: Optional[str] = None     # Mathematical description
    parameters: Dict[str, float] = field(default_factory=dict)  # Fitted constants
```

**Example**:
```python
Detection(
    type="orbital_motion",
    confidence=0.92,
    location=(45.3, 18.7),
    time=156,
    properties={
        'structure_ids': [3, 7],
        'orbital_period': 12.3,
        'eccentricity': 0.15
    }
)
```

### BaseAnalyzer Class

All analyzers inherit from `BaseAnalyzer`:

```python
class BaseAnalyzer(ABC):
    def __init__(self, name: str, min_confidence: float = 0.5):
        """
        Args:
            name: Analyzer identifier
            min_confidence: Minimum confidence threshold for reporting
        """
        self.name = name
        self.min_confidence = min_confidence
        self.detections = []
        self.history = []
        self.step_count = 0
    
    @abstractmethod
    def analyze(self, state: Dict) -> List[Detection]:
        """Implement detection logic here"""
        pass
    
    def update(self, state: Dict) -> List[Detection]:
        """Called by engine each step"""
        detections = self.analyze(state)
        # Filters by confidence, stores history
        return [d for d in detections if d.confidence >= self.min_confidence]
    
    def get_report(self) -> Dict:
        """Generate JSON-serializable report"""
        pass
    
    def print_summary(self):
        """Human-readable summary"""
        pass
```

---

## Creating a New Analyzer

### Step 1: Inherit from BaseAnalyzer

```python
from analyzers.base_analyzer import BaseAnalyzer, Detection
from typing import List, Dict
import numpy as np

class MyPhenomenonDetector(BaseAnalyzer):
    def __init__(self, min_confidence: float = 0.6):
        super().__init__("my_phenomenon_detector", min_confidence)
        # Add your custom state tracking
        self.phenomenon_history = []
```

### Step 2: Implement analyze() Method

```python
    def analyze(self, state: Dict) -> List[Detection]:
        """
        Detect your phenomenon from current state.
        
        Args:
            state: Dictionary with keys:
                - 'actual': Energy/actuation field (numpy array)
                - 'potential': Potential field (numpy array)
                - 'memory': Information/memory field (numpy array)
                - 'temperature': Temperature field (numpy array)
                - 'step': Current simulation step (int)
                - 'structures': List of EmergentStructure objects
                - 'field_E': Alias for 'actual'
                - 'field_I': Alias for 'memory'
        
        Returns:
            List of Detection objects
        """
        detections = []
        current_step = state['step']
        structures = state.get('structures', [])
        
        # Your detection logic here
        for s in structures:
            if self._is_phenomenon(s):
                confidence = self._compute_confidence(s)
                
                detections.append(Detection(
                    type="my_phenomenon",
                    confidence=confidence,
                    location=s.center,
                    time=current_step,
                    properties={
                        'structure_id': s.id,
                        'mass': s.mass,
                        'your_metric': self._compute_metric(s)
                    }
                ))
        
        return detections
```

### Step 3: Add Helper Methods

```python
    def _is_phenomenon(self, structure) -> bool:
        """Check if structure exhibits your phenomenon"""
        return (structure.mass > 10 and 
                structure.coherence > 0.8 and
                structure.lifetime > 50)
    
    def _compute_confidence(self, structure) -> float:
        """Compute detection confidence"""
        # Example: based on how well properties match expectations
        score = 0.5
        if structure.mass > 20:
            score += 0.2
        if structure.coherence > 0.9:
            score += 0.3
        return min(1.0, score)
    
    def _compute_metric(self, structure) -> float:
        """Compute phenomenon-specific metric"""
        return structure.mass * structure.coherence
```

### Step 4: Use EmergentStructure Attributes

Available attributes (use dot notation, not dictionary access):
- `s.id` - Unique structure identifier
- `s.mass` - Total mass/information content
- `s.center` - (x, y) center position
- `s.radius` - Spatial extent
- `s.coherence` - How aligned E and I are (0-1)
- `s.persistence` - Stability over time (0-1)
- `s.frequency` - Dominant oscillation frequency
- `s.entropy` - Local entropy
- `s.neighbors` - List of nearby structure IDs
- `s.binding_energy` - Energy binding components
- `s.angular_momentum` - Rotational component
- `s.charge_like` - Field divergence analog
- `s.lifetime` - How many timesteps observed
- `s.velocity` - (vx, vy) movement velocity
- `s.acceleration` - (ax, ay) for force measurement

---

## Unit Calibration

To compare simulation results to real physics, use unit calibration:

```python
class MyForceAnalyzer(BaseAnalyzer):
    def __init__(self, length_scale: float, mass_scale: float, 
                 time_scale: float, min_confidence: float = 0.7):
        super().__init__("my_force_analyzer", min_confidence)
        
        # Physical scales
        self.length_scale = length_scale  # meters per grid unit
        self.mass_scale = mass_scale      # kg per mass unit
        self.time_scale = time_scale      # seconds per time unit
        
        # Derived scales
        self.velocity_scale = length_scale / time_scale  # m/s
        self.acceleration_scale = length_scale / (time_scale**2)  # m/s²
        self.force_scale = mass_scale * self.acceleration_scale  # Newtons
    
    def analyze(self, state: Dict) -> List[Detection]:
        detections = []
        structures = state.get('structures', [])
        
        for s in structures:
            # Measure force in simulation units
            F_sim = self._measure_force(s)
            
            # Convert to physical units
            F_physical = F_sim * self.force_scale
            
            # Compare to known physics
            F_expected = self._expected_force(s)  # From theory
            ratio = F_physical / F_expected
            
            detections.append(Detection(
                type="force_measurement",
                confidence=0.8,
                location=s.center,
                time=state['step'],
                properties={
                    'F_simulation': F_sim,
                    'F_physical': F_physical,
                    'F_expected': F_expected,
                    'ratio': ratio
                }
            ))
        
        return detections
```

**Example calibrations**:
```python
# Atomic scale
GravityAnalyzer(
    length_scale=1e-10,   # 1 Ångström per grid unit
    mass_scale=1.67e-27,  # Proton mass per mass unit
    time_scale=1e-15      # 1 femtosecond per time unit
)

# Stellar scale
GravityAnalyzer(
    length_scale=1e9,     # 1 million km per grid unit
    mass_scale=2e30,      # Solar mass per mass unit
    time_scale=1.0        # 1 second per time unit
)

# Galactic scale
GravityAnalyzer(
    length_scale=9.46e15, # 1 light-year per grid unit
    mass_scale=2e42,      # 1 million solar masses per mass unit
    time_scale=3.15e13    # 1 million years per time unit
)
```

---

## Confidence Scores

Guidelines for assigning confidence:

- **0.9-1.0**: Very high confidence, phenomenon clearly detected
  - Multiple independent criteria all satisfied
  - Strong statistical significance (p < 0.001)
  - Example: Perfect conservation law over 1000 steps
  
- **0.7-0.9**: High confidence, phenomenon likely present
  - Most criteria satisfied
  - Good statistical significance (p < 0.01)
  - Example: Gravity well with R² > 0.9
  
- **0.5-0.7**: Moderate confidence, phenomenon probable
  - Some criteria satisfied
  - Moderate statistical support
  - Example: Possible molecular bond
  
- **0.3-0.5**: Low confidence, phenomenon possible
  - Weak signals
  - Limited statistical support
  - Use for exploratory detection
  
- **< 0.3**: Very low confidence
  - Noise level, should not report

**Tip**: Set `min_confidence` based on how permissive you want detection to be. Conservation laws should use high thresholds (0.9), exploratory detectors can use lower (0.5-0.6).

---

## Example: Full Analyzer

```python
from analyzers.base_analyzer import BaseAnalyzer, Detection
from typing import List, Dict
import numpy as np

class WaveDetector(BaseAnalyzer):
    """Detects wave-like propagation patterns in fields."""
    
    def __init__(self, min_confidence: float = 0.65):
        super().__init__("wave_detector", min_confidence)
        self.field_history = []  # Store field snapshots
    
    def analyze(self, state: Dict) -> List[Detection]:
        detections = []
        current_step = state['step']
        field_E = state.get('field_E')
        
        if field_E is None:
            return detections
        
        # Store field for temporal analysis
        self.field_history.append(field_E.copy())
        if len(self.field_history) > 10:
            self.field_history.pop(0)
        
        # Need at least 3 frames for wave detection
        if len(self.field_history) < 3:
            return detections
        
        # Detect waves by Fourier analysis
        wave_detected, wavelength, frequency = self._detect_wave()
        
        if wave_detected:
            confidence = self._compute_wave_confidence(wavelength, frequency)
            
            detections.append(Detection(
                type="wave_propagation",
                confidence=confidence,
                location=None,  # Field-wide phenomenon
                time=current_step,
                equation="E(x,t) = A*sin(kx - ωt)",
                parameters={
                    'wavelength': wavelength,
                    'frequency': frequency,
                    'wave_speed': wavelength * frequency
                },
                properties={
                    'field': 'energy',
                    'mode': 'transverse' if self._is_transverse() else 'longitudinal'
                }
            ))
        
        return detections
    
    def _detect_wave(self):
        """Analyze field history for wave patterns"""
        if len(self.field_history) < 3:
            return False, 0, 0
        
        # 2D FFT of most recent field
        fft = np.fft.fft2(self.field_history[-1])
        power = np.abs(fft)**2
        
        # Find dominant spatial frequency
        h, w = power.shape
        ky, kx = np.unravel_index(np.argmax(power[1:, 1:]), (h-1, w-1))
        
        # Wavelength from spatial frequency
        wavelength = min(h, w) / max(kx, ky, 1)
        
        # Temporal frequency from phase progression
        phases = [np.angle(np.fft.fft2(f)) for f in self.field_history[-3:]]
        phase_diff = np.mean(phases[2] - phases[1])
        frequency = phase_diff / (2 * np.pi)
        
        # Wave detected if wavelength reasonable and frequency non-zero
        detected = (3 < wavelength < 20 and abs(frequency) > 0.01)
        
        return detected, wavelength, frequency
    
    def _compute_wave_confidence(self, wavelength: float, frequency: float) -> float:
        """Confidence based on wave properties"""
        confidence = 0.5
        
        # Well-defined wavelength
        if 5 < wavelength < 15:
            confidence += 0.2
        
        # Clear frequency
        if abs(frequency) > 0.05:
            confidence += 0.2
        
        # Consistency across history
        if len(self.field_history) >= 5:
            consistency = self._measure_consistency()
            confidence += 0.1 * consistency
        
        return min(1.0, confidence)
    
    def _is_transverse(self) -> bool:
        """Check if wave is transverse or longitudinal"""
        # Simplified: check if oscillation perpendicular to propagation
        # Real implementation would analyze polarization
        return True
    
    def _measure_consistency(self) -> float:
        """Measure how consistent wave pattern is over time"""
        if len(self.field_history) < 5:
            return 0.0
        
        # Compute correlation between successive frames
        correlations = []
        for i in range(len(self.field_history) - 1):
            corr = np.corrcoef(
                self.field_history[i].flatten(),
                self.field_history[i+1].flatten()
            )[0, 1]
            correlations.append(corr)
        
        return np.mean(correlations)
```

---

## Using Analyzers

### Basic Usage

```python
from core.reality_engine import RealityEngine
from tools.emergence_observer import EmergenceObserver
from analyzers.laws.gravity_analyzer import GravityAnalyzer

# Initialize
engine = RealityEngine(size=(96, 24))
engine.initialize()
observer = EmergenceObserver()
gravity = GravityAnalyzer(length_scale=1e-10, mass_scale=1.67e-27, time_scale=1e-15)

# Simulation loop
for step in range(1000):
    state = engine.step()
    structures = observer.observe(engine.current_state)
    
    # Prepare analyzer state
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
    
    # Get detections
    detections = gravity.update(analyzer_state)
    
    # Print high-confidence detections
    for d in detections:
        if d.confidence > 0.8:
            print(f"[{step}] {d.type}: {d.confidence:.1%}")

# Generate report
report = gravity.get_report()
print(f"Total detections: {report['total_detections']}")
print(f"Average confidence: {report['avg_confidence']:.1%}")

# Save to JSON
gravity.save_report("gravity_analysis.json")
```

### Multiple Analyzers

```python
from analyzers.laws.gravity_analyzer import GravityAnalyzer
from analyzers.laws.conservation_analyzer import ConservationAnalyzer
from analyzers.matter.atom_detector import AtomDetector

# Initialize all analyzers
analyzers = [
    GravityAnalyzer(length_scale=1e-10, mass_scale=1.67e-27, time_scale=1e-15),
    ConservationAnalyzer(min_confidence=0.9),
    AtomDetector(min_confidence=0.6)
]

# Simulation loop
for step in range(1000):
    state = engine.step()
    structures = observer.observe(engine.current_state)
    
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
    
    # Update all analyzers
    for analyzer in analyzers:
        detections = analyzer.update(analyzer_state)
        
        # Print interesting detections
        for d in detections:
            if d.confidence > 0.8:
                print(f"[{step}] {analyzer.name}: {d.type} ({d.confidence:.1%})")

# Generate comprehensive report
for analyzer in analyzers:
    print(f"\n{'='*70}")
    print(f"{analyzer.name.upper()} ANALYSIS")
    print('='*70)
    analyzer.print_summary()
```

---

## Best Practices

### 1. Keep Analyzers Independent
- Don't share state between analyzers
- Each analyzer should be self-contained
- Use history tracking within the analyzer

### 2. Memory Management
- Store history sparingly (every 10 steps, not every step)
- Limit history length (keep last 100-200 entries)
- Use rolling windows for statistics

### 3. Performance
- Avoid expensive operations every step
- Cache intermediate calculations
- Use numpy vectorization

### 4. Confidence Calibration
- Test your analyzer on known phenomena
- Adjust thresholds based on false positive/negative rates
- Document what confidence levels mean for your detector

### 5. Unit Calibration
- Always specify units in property descriptions
- Document the physical scale you're calibrating to
- Test at multiple scales if possible

---

## Troubleshooting

**Problem**: Analyzer finds nothing
- Lower `min_confidence` threshold
- Check if phenomenon exists in your simulation
- Add debug prints in `analyze()` method
- Verify `state` dictionary has expected keys

**Problem**: Too many false positives
- Increase `min_confidence` threshold
- Add more stringent detection criteria
- Use multiple independent checks

**Problem**: Confidence scores always the same
- Make confidence dynamic based on multiple factors
- Weight different criteria differently
- Use continuous metrics, not binary checks

**Problem**: Memory usage growing
- Limit history length (implement rolling window)
- Store only every Nth step
- Use summary statistics instead of raw data

---

## Next Steps

1. **Study existing analyzers**: See `analyzers/laws/gravity_analyzer.py` for a complete example
2. **Test your analyzer**: Create a test script like `scripts/test_analyzers.py`
3. **Calibrate**: Run simulations and adjust confidence thresholds
4. **Document**: Add docstrings and examples to your analyzer
5. **Contribute**: Share your analyzer with the community!

---

## Reference: Existing Analyzers

### GravityAnalyzer
- **File**: `analyzers/laws/gravity_analyzer.py`
- **Detects**: Orbital motion, gravitational collapse, force measurements
- **Key Feature**: Unit calibration to compare with Newton's G

### ConservationAnalyzer
- **File**: `analyzers/laws/conservation_analyzer.py`
- **Detects**: E+I conservation, PAC conservation, momentum conservation
- **Key Feature**: 50-step window with drift detection

### AtomDetector
- **File**: `analyzers/matter/atom_detector.py`
- **Detects**: Atomic structures, molecular bonds, mass quantization
- **Key Feature**: Mass histogram for periodic table signature

### StarDetector
- **File**: `analyzers/cosmic/star_detector.py`
- **Detects**: Stellar objects, fusion processes, stellar evolution
- **Key Feature**: H-R diagram data collection

### QuantumDetector
- **File**: `analyzers/cosmic/quantum_detector.py`
- **Detects**: Entanglement, superposition, tunneling, wave-particle duality
- **Key Feature**: Correlation tracking for entanglement

### GalaxyAnalyzer
- **File**: `analyzers/cosmic/galaxy_analyzer.py`
- **Detects**: Galaxies, rotation curves, dark matter, cosmic web
- **Key Feature**: Multi-scale clustering analysis
