"""
Emergence Layer - Multi-scale structure detection and analysis.

Provides unified interface for detecting emergent structures from field dynamics:

**Atomic Scale:**
- AtomicAnalyzer: Detect atoms (H, He, Li, C, etc.) by mass, stability, quantum states
- Molecular detection: H₂, H₂O bonds via proximity coupling

**Macro Scale:**
- StructureAnalyzer: Gravity wells, dark matter, stellar regions, molecules
- ParticleAnalyzer: Stable field configurations, charge, spin, lifetime
- StellarAnalyzer: Mass concentrations, fusion regions, black holes

**Usage:**

```python
from emergence import StructureAnalyzer, AtomicAnalyzer

# Comprehensive analysis
analyzer = StructureAnalyzer(engine, min_atom_stability=0.65)
structures = analyzer.analyze_step(step=100)

print(f"Atoms: {len(structures['atoms'])}")
print(f"Molecules: {len(structures['molecules'])}")
print(f"Gravity wells: {len(structures['gravity_wells'])}")
print(f"Stellar regions: {len(structures['stellar_regions'])}")

# Atomic-only analysis
from tools.atomic_analyzer import AtomicAnalyzer, Atom, build_periodic_table

atomic = AtomicAnalyzer(min_stability=0.7)
atoms = atomic.detect_atoms(state)
periodic_table = build_periodic_table(atoms_history)
```

**No Physics Hardcoded:**
- Atoms emerge from stable M field oscillations
- Molecules form via proximity bonding
- Gravity wells appear from density clustering
- Dark matter emerges as high M, low A regions
- Stars form where heat + mass concentrate

Everything detected here is EMERGENT from SEC + PAC + Thermodynamics!
"""

# Core structure detection
from emergence.structure_analyzer import StructureAnalyzer

# Legacy analyzers (consider deprecating or integrating)
from emergence.particle_analyzer import ParticleAnalyzer
from emergence.stellar_analyzer import StellarAnalyzer

# Atomic analysis from tools (re-exported for convenience)
from tools.atomic_analyzer import AtomicAnalyzer, Atom, build_periodic_table

__all__ = [
    # Primary interface
    'StructureAnalyzer',
    
    # Atomic/molecular
    'AtomicAnalyzer',
    'Atom',
    'build_periodic_table',
    
    # Legacy (phase out or integrate)
    'ParticleAnalyzer',
    'StellarAnalyzer',
]
