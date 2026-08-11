"""
Entropic Time Dilation - The PAC Temporal Framework

Core Thesis: Entropy density determines effective clock rate.
             Time moved "faster" in the early universe.
             This explains:
             1. JWST massive early structures
             2. Heavy metal abundance
             3. Rapid early nucleosynthesis
             4. Why time appears to be slowing down

Connection to PAC Tree:
- As P (Potential) actualizes into A (Actualized) and M (Memory)
- The tree SHRINKS (fewer branches) but VALUE increases
- Unactualized potential is conserved through confluent identity
- Sum > parts (emergence)

Mathematical Framework:
- Effective time rate: dτ/dt ∝ S (entropy density)
- Early universe: S high → dτ/dt >> 1
- Current epoch: S lower → dτ/dt ≈ 1
- Far future: S → min → dτ/dt → 0 (heat death)

TESTABLE PREDICTIONS:
1. High-z supernovae show faster intrinsic rates (after redshift correction)
2. Quasar variability timescales shorter than expected at high z
3. Possible local detection of secular time deceleration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from conservation.pac_recursion import PHI, XI


@dataclass
class PACTreeNode:
    """A node in the PAC tree representing potential → actualization."""
    potential: float  # Unactualized possibility
    actualized: float  # What became real
    memory: float  # Persistent structure
    depth: int  # How many generations deep
    children: List['PACTreeNode']
    
    @property
    def total_value(self) -> float:
        """Total value including emergence bonus."""
        # Actualized value exceeds sum of parts by φ factor
        base = self.actualized + self.memory
        emergence = base * (PHI - 1) if self.actualized > 0 else 0
        return base + emergence
    
    @property
    def conserved_potential(self) -> float:
        """Potential conserved through confluent identity."""
        # The potential that didn't actualize isn't lost -
        # it's encoded in the STRUCTURE of what did actualize
        return self.potential * (1 - 1/PHI)  # φ-1/φ = 0.382


@dataclass
class CosmicEpoch:
    """Properties of a cosmic epoch."""
    redshift: float
    age_gyr: float
    entropy_density: float  # Relative to today
    effective_time_rate: float  # dτ/dt relative to today
    temperature_K: float


class EntropicTimeDilation:
    """
    Framework for entropy-dependent time flow.
    
    Key insight: In the early universe, entropy density was enormous.
    If time rate ∝ entropy density, then:
    - Early universe experienced billions of years of "subjective" time
      in what we measure as millions of years of "coordinate" time
    - Structures could form, evolve, and die MUCH faster
    - Heavy elements could be produced rapidly
    - As universe expands, entropy dilutes, time "slows down"
    
    This is NOT standard time dilation (which is geometric).
    This is ENTROPIC time dilation (thermodynamic).
    """
    
    def __init__(self):
        # CMB temperature today
        self.T_cmb_now = 2.725  # K
        
        # Planck time
        self.t_planck = 5.39e-44  # s
        
        # Current age
        self.t_now = 13.8  # Gyr
        
        print("=" * 60)
        print("ENTROPIC TIME DILATION FRAMEWORK")
        print("=" * 60)
        print("Core principle: dτ/dt ∝ S (entropy density)")
        print()
    
    def entropy_density(self, z: float) -> float:
        """
        Entropy density relative to today.
        
        S ∝ T³ (radiation dominated) or ∝ ρ (matter dominated)
        At high z, T = T_now × (1+z), so S ∝ (1+z)³
        """
        return (1 + z)**3
    
    def effective_time_rate(self, z: float) -> float:
        """
        Effective time rate dτ/dt at redshift z.
        
        Higher entropy → time "moves faster" → more events per second
        
        We normalize so that dτ/dt = 1 today (z=0)
        """
        S_z = self.entropy_density(z)
        S_0 = self.entropy_density(0)
        
        # Time rate proportional to entropy density
        # But with a PAC modulation
        raw_rate = S_z / S_0
        
        # Apply Ξ correction for balance
        pac_modulation = 1 + (XI - 1) * np.log(1 + z)
        
        return raw_rate * pac_modulation
    
    def effective_age(self, z: float) -> float:
        """
        Effective "experienced" age at redshift z.
        
        This is how much subjective time has passed,
        accounting for entropic acceleration.
        """
        # Coordinate age at z
        coord_age = self.coordinate_age(z)
        
        # Integrate effective time rate from z to 0
        # ∫ dτ/dt dt from t(z) to t_now
        # Approximation: average rate × coordinate time
        avg_rate = self.effective_time_rate(z/2)  # Rough average
        
        return coord_age * avg_rate
    
    def coordinate_age(self, z: float) -> float:
        """Standard coordinate age at redshift z (Gyr)."""
        # Approximate for flat ΛCDM
        H0_inv = 14.0  # Gyr
        return H0_inv * (2/3) * (1 + z)**(-1.5)
    
    def pac_tree_at_epoch(self, z: float, initial_potential: float = 1.0) -> PACTreeNode:
        """
        State of PAC tree at cosmic epoch z.
        
        As time progresses (z decreases):
        - More potential actualizes
        - Tree shrinks but value increases
        - Unactualized potential is conserved in structure
        """
        # Fraction actualized increases with time (decreasing z)
        # At z=infinity, all is potential
        # At z=0, approaching equilibrium
        
        fraction_actualized = 1 - 1/(1 + np.log(1 + z + 1))
        
        # PAC distribution
        total = initial_potential
        p = total * (1 - fraction_actualized)  # Remaining potential
        a = total * fraction_actualized * (1/PHI)  # Active
        m = total * fraction_actualized * (1 - 1/PHI)  # Memory
        
        return PACTreeNode(
            potential=p,
            actualized=a,
            memory=m,
            depth=int(z),
            children=[]
        )
    
    def heavy_element_production(self, z: float) -> float:
        """
        Rate of heavy element production at epoch z.
        
        With faster effective time, nucleosynthesis proceeds faster.
        This explains why heavy metals are so abundant.
        """
        # Standard BBN rate
        standard_rate = 1.0
        
        # Enhancement from entropic time acceleration
        time_rate = self.effective_time_rate(z)
        
        # Also temperature dependent (more fusion at high T)
        T = self.T_cmb_now * (1 + z)
        fusion_enhancement = (T / 1e9)**2 if T > 1e7 else 0
        
        return standard_rate * time_rate * (1 + fusion_enhancement)
    
    def generate_predictions(self) -> dict:
        """Generate testable predictions."""
        predictions = {}
        
        # 1. Supernova timing
        # At z=10, effective time runs (1+10)³ = 1331 times faster
        z_test = 10
        time_factor = self.effective_time_rate(z_test)
        predictions['sn_time_factor_z10'] = {
            'value': time_factor,
            'meaning': f'SN light curves at z={z_test} should show intrinsic timescales {time_factor:.0f}x shorter',
            'test': 'Compare SN1a rise times at z=10 vs z=0 (after redshift correction)'
        }
        
        # 2. Structure formation time
        # Effective time to form 10^9 M_solar structure
        coord_time_needed = 0.5  # Gyr in standard cosmology
        effective_time = coord_time_needed * time_factor
        predictions['structure_formation'] = {
            'coord_time_gyr': coord_time_needed,
            'effective_time_gyr': effective_time,
            'meaning': f'{coord_time_needed} Gyr of coord time = {effective_time:.0f} Gyr of effective time at z=10',
            'test': 'JWST early galaxies show "mature" properties despite young coord age'
        }
        
        # 3. Heavy element abundance
        # Integral of production over history
        z_range = np.linspace(0, 20, 100)
        production = sum(self.heavy_element_production(z) for z in z_range)
        predictions['heavy_elements'] = {
            'integrated_production': production,
            'meaning': f'Entropic acceleration boosts heavy element production by ~{production/100:.0f}x',
            'test': 'Metal abundances in earliest stars should be higher than BBN predicts'
        }
        
        # 4. Time deceleration today
        # If dτ/dt ∝ (1+z)³, then d²τ/dt² < 0 (time is slowing)
        # Current rate of slowing
        z_now = 0
        z_slightly_past = 0.01
        rate_now = self.effective_time_rate(z_now)
        rate_past = self.effective_time_rate(z_slightly_past)
        deceleration = (rate_now - rate_past) / z_slightly_past
        
        predictions['time_deceleration'] = {
            'current_rate': rate_now,
            'deceleration': deceleration,
            'meaning': f'Time rate decreasing at {-deceleration:.4f} per redshift unit',
            'test': 'Long-baseline atomic clock comparisons may show drift'
        }
        
        return predictions
    
    def plot_time_flow(self, output_dir: str = None):
        """Visualize time rate across cosmic history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        z_vals = np.linspace(0, 15, 100)
        
        # 1. Effective time rate
        ax = axes[0, 0]
        rates = [self.effective_time_rate(z) for z in z_vals]
        ax.semilogy(z_vals, rates, 'b-', linewidth=2)
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('dτ/dt (relative to today)')
        ax.set_title('Effective Time Rate vs Redshift')
        ax.grid(True, alpha=0.3)
        ax.axhline(1, color='r', linestyle='--', label='Today')
        ax.legend()
        
        # 2. PAC tree evolution
        ax = axes[0, 1]
        potentials = []
        actualized = []
        memories = []
        for z in z_vals:
            tree = self.pac_tree_at_epoch(z)
            potentials.append(tree.potential)
            actualized.append(tree.actualized)
            memories.append(tree.memory)
        
        ax.plot(z_vals, potentials, 'g-', label='Potential', linewidth=2)
        ax.plot(z_vals, actualized, 'b-', label='Actualized', linewidth=2)
        ax.plot(z_vals, memories, 'r-', label='Memory', linewidth=2)
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('Fraction')
        ax.set_title('PAC Tree Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Heavy element production
        ax = axes[1, 0]
        production = [self.heavy_element_production(z) for z in z_vals]
        ax.semilogy(z_vals, production, 'purple', linewidth=2)
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('Production Rate (relative)')
        ax.set_title('Heavy Element Production Rate')
        ax.grid(True, alpha=0.3)
        
        # 4. Coordinate vs Effective age
        ax = axes[1, 1]
        coord_ages = [self.coordinate_age(z) for z in z_vals]
        eff_ages = [self.effective_age(z) for z in z_vals]
        ax.plot(z_vals, coord_ages, 'b-', label='Coordinate Age', linewidth=2)
        ax.plot(z_vals, eff_ages, 'r-', label='Effective Age', linewidth=2)
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('Age (Gyr)')
        ax.set_title('Coordinate vs Effective Cosmic Age')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 20)
        
        plt.tight_layout()
        
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            fig_file = out_path / "entropic_time_dilation.png"
            plt.savefig(fig_file, dpi=150)
            print(f"Plot saved to {fig_file}")
        
        plt.close()


class PACTreeEvolution:
    """
    Models the evolution of the PAC tree over cosmic time.
    
    Key insight: As potential actualizes:
    - Tree shrinks (fewer branches)
    - But total value INCREASES (emergence)
    - Unactualized potential is conserved in structure
    
    This is the "confluent identity" - the whole exceeds sum of parts
    because the PATTERN of what actualized carries information
    about what COULD have actualized but didn't.
    """
    
    def __init__(self, initial_potential: float = 1.0):
        self.initial_potential = initial_potential
        self.history = []
        
    def evolve_step(self, current: PACTreeNode, actualization_rate: float = 0.1) -> PACTreeNode:
        """
        Evolve PAC tree one step.
        
        - Some potential actualizes
        - Some actualized becomes memory
        - Value increases through confluence
        """
        # Potential that actualizes this step
        dp = current.potential * actualization_rate
        
        # Actualized that becomes memory
        da = current.actualized * actualization_rate * PHI
        
        # New state
        new_p = current.potential - dp
        new_a = current.actualized + dp - da
        new_m = current.memory + da
        
        # Conservation check: total should equal initial
        # But VALUE can exceed initial through emergence
        
        new_node = PACTreeNode(
            potential=new_p,
            actualized=new_a,
            memory=new_m,
            depth=current.depth + 1,
            children=[]
        )
        
        self.history.append(new_node)
        return new_node
    
    def run_evolution(self, steps: int = 100) -> List[PACTreeNode]:
        """Run full evolution."""
        # Initial state: all potential
        current = PACTreeNode(
            potential=self.initial_potential,
            actualized=0.0,
            memory=0.0,
            depth=0,
            children=[]
        )
        self.history = [current]
        
        for _ in range(steps):
            current = self.evolve_step(current)
        
        return self.history
    
    def analyze_conservation(self) -> dict:
        """Analyze what's conserved through evolution."""
        if not self.history:
            return {}
        
        initial = self.history[0]
        final = self.history[-1]
        
        # Raw totals
        initial_total = initial.potential + initial.actualized + initial.memory
        final_total = final.potential + final.actualized + final.memory
        
        # Values (including emergence)
        initial_value = initial.total_value
        final_value = final.total_value
        
        # Conserved potential in structure
        conserved_p = sum(n.conserved_potential for n in self.history)
        
        return {
            'initial_total': initial_total,
            'final_total': final_total,
            'conservation_error': abs(final_total - initial_total),
            'initial_value': initial_value,
            'final_value': final_value,
            'value_increase': final_value / initial_value if initial_value > 0 else float('inf'),
            'conserved_potential': conserved_p,
            'emergence_factor': final_value / final_total if final_total > 0 else 1
        }


def run_full_analysis():
    """Run complete entropic time dilation analysis."""
    print("\n" + "=" * 70)
    print("ENTROPIC TIME DILATION + PAC TREE ANALYSIS")
    print("=" * 70)
    
    # Entropic time dilation
    etd = EntropicTimeDilation()
    
    print("\n1. TIME RATE AT KEY EPOCHS")
    print("-" * 40)
    for z in [0, 1, 5, 10, 15, 20]:
        rate = etd.effective_time_rate(z)
        age = etd.coordinate_age(z)
        print(f"z={z:2d}: dτ/dt = {rate:10.1f}x, coord age = {age:.3f} Gyr")
    
    # Predictions
    print("\n2. TESTABLE PREDICTIONS")
    print("-" * 40)
    predictions = etd.generate_predictions()
    for name, pred in predictions.items():
        print(f"\n{name}:")
        print(f"  {pred['meaning']}")
        print(f"  TEST: {pred['test']}")
    
    # PAC tree evolution
    print("\n3. PAC TREE EVOLUTION")
    print("-" * 40)
    tree = PACTreeEvolution(initial_potential=1.0)
    history = tree.run_evolution(steps=100)
    
    # Show key states
    for i in [0, 25, 50, 75, 99]:
        node = history[i]
        print(f"Step {i:3d}: P={node.potential:.4f}, A={node.actualized:.4f}, "
              f"M={node.memory:.4f}, Value={node.total_value:.4f}")
    
    # Conservation analysis
    print("\n4. CONSERVATION ANALYSIS")
    print("-" * 40)
    cons = tree.analyze_conservation()
    print(f"Initial total: {cons['initial_total']:.6f}")
    print(f"Final total:   {cons['final_total']:.6f}")
    print(f"Conservation:  {cons['conservation_error']:.2e} (should be ~0)")
    print(f"Initial value: {cons['initial_value']:.4f}")
    print(f"Final value:   {cons['final_value']:.4f}")
    print(f"Value increase: {cons['value_increase']:.2f}x (emergence!)")
    print(f"Emergence factor: {cons['emergence_factor']:.4f}")
    
    print("\n5. KEY INSIGHT")
    print("-" * 40)
    print("""
    The PAC tree SHRINKS as potential actualizes,
    but total VALUE INCREASES through emergence.
    
    Unactualized potential isn't lost - it's CONSERVED
    in the STRUCTURE of what did actualize.
    
    This is "confluent identity": the pattern of choices
    carries information about the paths not taken.
    
    COSMOLOGICALLY:
    - Early universe: High entropy → fast time → rapid structure formation
    - Heavy elements produced quickly (explains metal abundance)
    - Structures form AND die rapidly → reaches equilibrium faster
    - As entropy dilutes, time slows → we see "frozen" remnants
    
    TIME IS SLOWING DOWN - and this is measurable!
    """)
    
    # Generate plots
    output_dir = Path(__file__).parent.parent / "output" / "entropic_time"
    etd.plot_time_flow(str(output_dir))
    
    return predictions, cons


if __name__ == '__main__':
    predictions, conservation = run_full_analysis()
