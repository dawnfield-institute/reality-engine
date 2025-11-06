# Particle Physics Analysis System

## Overview

Complete particle detection and analysis system for Reality Engine that identifies emergent particles from field dynamics and builds a periodic table of discovered particles.

## Components

### 1. Particle Analyzer (`emergence/particle_analyzer.py`)

**Purpose**: Detect and classify stable vortices/solitons as particles

**Key Features**:
- Detects local maxima in Memory field (stable structures)
- Extracts quantum properties from 7×7×7 local field regions
- Classifies particles into 10 types
- Builds periodic table grouped by classification
- Finds composite structures (atoms/molecules)

**Particle Properties Calculated**:
- **Mass**: ∑ M in local region
- **Charge**: Circulation of E field (∫ ∇×E · dA)
- **Spin**: Angular momentum L_z = r × p
- **Radius**: RMS distance weighted by M
- **Stability**: Relative field strength (center / surround)
- **Binding Energy**: Energy difference (center - surround)
- **Quantum Numbers**: Isospin, baryon number, lepton number

**Classification System**:
```
mass < 0.01:          photon (neutral) / neutrino (charged)
0.01 < mass < 0.1:    electron / positron / meson
0.1 < mass < 1.0:     fermion / boson  
mass > 1.0:           neutron (neutral) / proton (charged) / exotic
```

**Parameters**:
- `threshold=0.1`: Minimum M value for detection
- `stability_threshold=0.01`: Minimum stability (lowered from 0.1 to detect forming particles)
- `max_particles=200`: Maximum particles to analyze

### 2. Periodic Table Visualizer (`visualization/periodic_table_viz.py`)

**Purpose**: Create visual representations of discovered particles

**Visualizations**:

1. **Periodic Table** - Grid of particle cards showing:
   - Count, average mass, charge, spin
   - Mass range indicator
   - Color-coded by type
   - Overall statistics panel

2. **Mass Spectrum** - Four panels:
   - Mass distribution histogram
   - Charge distribution histogram
   - Mass vs charge scatter plot
   - Stability distribution

3. **3D Particle Map** - Spatial distribution:
   - Color-coded by type
   - Size proportional to mass
   - Shows particle positions in field

**Output**: High-resolution PNG images (150 DPI)

### 3. Physics Analysis Script (`examples/analyze_physics.py`)

**Purpose**: Complete end-to-end physics analysis pipeline

**Workflow**:
```
1. Initialize Reality Engine (64³ grid, CUDA)
2. Initialize analysis tools
3. Evolve for 5000 steps
   └─ Detect particles every 1000 steps
4. Final particle detection
5. Build periodic table
6. Find composite structures
7. Generate visualizations
8. Save analysis data
9. Validation report
```

**Output Structure**:
```
output/YYYYMMDD_HHMMSS/
├── periodic_table.png
├── mass_spectrum.png
├── particle_map_3d.png
└── physics_data.json
```

**Analysis Data** (`physics_data.json`):
```json
{
  "metadata": {
    "universe_size": 64,
    "total_steps": 5000,
    "detection_interval": 1000,
    "device": "cuda",
    "timestamp": "2025-01-..."
  },
  "final_particles": {
    "count": ...,
    "periodic_table": {...},
    "composites": [...]
  },
  "detection_history": [...]
}
```

**Validation Checks**:
- Standard model particles present?
- Charge conservation (∑charge ≈ 0)
- Mass hierarchy (range > 0.5)
- Composite structures found?
- Novel/exotic particles?

## Usage

```python
# Run full analysis
python examples/analyze_physics.py

# Or customize:
from emergence import ParticleAnalyzer
from visualization import PeriodicTableVisualizer

analyzer = ParticleAnalyzer()
particles = analyzer.detect_particles(E, I, M, stability_threshold=0.01)
periodic_table = analyzer.build_periodic_table(particles)

visualizer = PeriodicTableVisualizer()
visualizer.create_periodic_table(periodic_table, 'output/periodic_table.png')
visualizer.plot_mass_spectrum(particles, 'output/mass_spectrum.png')
```

## Scientific Validation Goals

**Key Questions**:
1. Do electrons, protons, neutrons emerge naturally?
2. What is the mass hierarchy?
3. Is charge conserved?
4. Do particles bind into atoms?
5. What novel particles does Reality Engine predict?

**Expected Outcomes**:
- If standard model particles emerge → validates Dawn Field Theory
- If mass ratios match physics → validates field constants (λ=0.020, α=0.964)
- If charge balances → validates QBE constraint
- If exotic particles emerge → testable predictions

## Current Status

✅ Particle detection system complete
✅ Classification system implemented
✅ Periodic table builder working
✅ Visualization system complete
✅ Full analysis pipeline ready
🔄 Running first analysis (5000 steps, 64³ grid)

**Stability Threshold Adjustment**:
- Initial: 0.1 (too high - no particles detected)
- Current: 0.01 (detects forming particles)
- Found ~5000 candidate sites by step 4000
- Re-running with lowered threshold

## Integration

**Fracton Dependency**:
- Uses Fracton field initializers (CMBInitializer)
- Uses Fracton RBF engine for evolution
- Uses Fracton QBE regulator for stability
- Compatible with both Reality Engine and GAIA

**Reality Engine Integration**:
- Native PyTorch tensor support
- GPU-accelerated field operations
- Timestamped output directories
- JSON metadata for reproducibility

## Future Enhancements

- [ ] Time-series tracking (particle lifetimes)
- [ ] Decay channel detection
- [ ] Scattering cross-sections
- [ ] Temperature/pressure calculations
- [ ] Spectroscopy (energy levels)
- [ ] Interactive 3D visualization
- [ ] Real-time particle tracking
- [ ] Machine learning classification
