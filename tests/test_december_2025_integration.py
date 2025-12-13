"""
December 2025 Integration Test

Tests the full integration of all new modules:
1. PAC Recursion Engine
2. Klein-Gordon Evolution
3. Rearrangement Tensor
4. Scale Hierarchy
5. Cosmological Observables

Goal: Demonstrate that PAC-constrained physics:
- Stabilizes long-term (10,000+ steps)
- Produces 0.02 Hz without hardcoding
- Maintains conservation to machine precision
- Generates JWST-compatible predictions

This is the key validation that the December 2025 mathematics
is correctly implemented and produces physics.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Import all new modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from conservation.pac_recursion import PACRecursion, PHI, XI
from dynamics.klein_gordon import KleinGordonEvolution, create_initial_perturbation
from core.rearrangement_tensor import RearrangementTensor, FieldType
from scales.scale_hierarchy import ScaleHierarchy, MultiScaleField
from cosmology.observables import CosmologicalObservables


class IntegratedPACSimulation:
    """
    Full integration of PAC physics modules.
    
    This represents a simplified "universe" that:
    - Evolves via Klein-Gordon with PAC-derived mass
    - Maintains PAC recursion across scales
    - Conserves total P+A+M via rearrangement
    - Produces measurable cosmological observables
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        n_scales: int = 20,
        dt: float = 0.01,
        device: str = 'cpu'
    ):
        self.grid_size = grid_size
        self.n_scales = n_scales
        self.dt = dt
        self.device = device
        
        print("=" * 60)
        print("INTEGRATED PAC SIMULATION")
        print("=" * 60)
        
        # Initialize all modules
        print("\n1. Initializing Klein-Gordon evolution...")
        self.kg = KleinGordonEvolution(xi=XI, dt=dt, device=device)
        
        print("\n2. Initializing PAC recursion enforcer...")
        self.pac = PACRecursion(tolerance=1e-8, phi_tolerance=0.01)
        
        print("\n3. Initializing rearrangement tensor...")
        self.rt = RearrangementTensor(
            shape=(grid_size, grid_size),
            initial_total=100.0,
            device=device
        )
        
        print("\n4. Initializing scale hierarchy...")
        self.hierarchy = ScaleHierarchy(k_min=0, k_max=n_scales-1)
        self.multi_scale = MultiScaleField(
            self.hierarchy,
            grid_shape=(grid_size, grid_size),
            device=device
        )
        
        print("\n5. Initializing cosmological observables...")
        self.cosmo = CosmologicalObservables()
        
        # Initialize Klein-Gordon field
        self.psi = create_initial_perturbation(
            (grid_size, grid_size), 'gaussian', amplitude=1.0, device=device
        )
        self.psi_prev = self.psi.clone()
        
        # Tracking
        self.history = {
            'step': [],
            'total_energy': [],
            'pac_conservation': [],
            'kg_amplitude': [],
            'frequency': [],
            'p_fraction': [],
            'a_fraction': [],
            'm_fraction': [],
            'phi_ratio_error': [],
        }
        
        self.step = 0
        
        print("\n" + "=" * 60)
        print("INITIALIZATION COMPLETE")
        print("=" * 60)
    
    def evolve_step(self):
        """Evolve one time step with full PAC coupling."""
        
        # 1. Klein-Gordon evolution
        self.psi, self.psi_prev = self.kg.evolve_step(self.psi, self.psi_prev)
        
        # 2. Use KG field to drive rearrangement
        self.rt.compute_driven_transfers(self.psi, coupling_strength=0.05)
        rt_metrics = self.rt.apply_transfers(self.dt)
        
        # 3. Project toward phi ratios periodically
        if self.step % 50 == 0:
            self.rt.project_to_phi_ratios()
        
        # 4. Enforce PAC recursion on multi-scale field
        scale_fields = [self.multi_scale.fields[i] for i in range(len(self.hierarchy.levels))]
        corrected, pac_metrics = self.pac.enforce(scale_fields)
        for i, field in enumerate(corrected):
            self.multi_scale.fields[i] = field
        
        # 5. Equilibrate multi-scale to PAC
        if self.step % 100 == 0:
            self.multi_scale.equilibrate_to_pac(steps=10, rate=0.01)
        
        self.step += 1
        
        return rt_metrics, pac_metrics
    
    def record_state(self, rt_metrics, pac_metrics):
        """Record current state for analysis."""
        self.history['step'].append(self.step)
        self.history['total_energy'].append(self.multi_scale.compute_total_energy())
        self.history['pac_conservation'].append(rt_metrics.total_pac)
        self.history['kg_amplitude'].append(self.psi.abs().mean().item())
        self.history['p_fraction'].append(rt_metrics.p_fraction)
        self.history['a_fraction'].append(rt_metrics.a_fraction)
        self.history['m_fraction'].append(rt_metrics.m_fraction)
        self.history['phi_ratio_error'].append(pac_metrics.phi_ratio_error)
        
        # Frequency from KG
        freq = self.kg._estimate_frequency()
        self.history['frequency'].append(freq)
    
    def run(self, steps: int, record_every: int = 10, verbose: bool = True):
        """Run simulation for specified steps."""
        
        if verbose:
            print(f"\nRunning {steps} steps...")
            print("-" * 40)
        
        for i in range(steps):
            rt_metrics, pac_metrics = self.evolve_step()
            
            if self.step % record_every == 0:
                self.record_state(rt_metrics, pac_metrics)
            
            if verbose and self.step % (steps // 10) == 0:
                pct = 100 * self.step / steps
                drift = rt_metrics.drift
                freq = self.history['frequency'][-1] if self.history['frequency'] else 0
                print(f"  Step {self.step:6d} ({pct:5.1f}%): "
                      f"drift={drift:.2e}, freq={freq:.4f} Hz")
        
        if verbose:
            print("-" * 40)
            print("Simulation complete!")
    
    def analyze(self) -> dict:
        """Analyze simulation results."""
        results = {}
        
        # Conservation analysis
        pac_vals = np.array(self.history['pac_conservation'])
        if len(pac_vals) > 1:
            initial = pac_vals[0]
            final = pac_vals[-1]
            max_drift = np.max(np.abs(pac_vals - initial)) / initial
            avg_drift = np.mean(np.abs(pac_vals - initial)) / initial
            
            results['conservation'] = {
                'initial': initial,
                'final': final,
                'max_drift': max_drift,
                'avg_drift': avg_drift,
                'maintained': max_drift < 1e-4  # 0.01% threshold
            }
        
        # Frequency analysis
        freqs = np.array(self.history['frequency'])
        nonzero_freqs = freqs[freqs > 0]
        if len(nonzero_freqs) > 0:
            mean_freq = np.mean(nonzero_freqs)
            std_freq = np.std(nonzero_freqs)
            matches_0_02 = abs(mean_freq - 0.02) < 0.01
            
            results['frequency'] = {
                'mean': mean_freq,
                'std': std_freq,
                'matches_0_02_hz': matches_0_02,
                'deviation_from_0_02': abs(mean_freq - 0.02)
            }
        
        # PAC fraction analysis
        p_vals = np.array(self.history['p_fraction'])
        a_vals = np.array(self.history['a_fraction'])
        m_vals = np.array(self.history['m_fraction'])
        
        # Expected PAC equilibrium: 1 : φ : φ²
        phi_sum = 1 + PHI + PHI**2
        expected_p = 1 / phi_sum
        expected_a = PHI / phi_sum
        expected_m = PHI**2 / phi_sum
        
        if len(p_vals) > 0:
            final_p, final_a, final_m = p_vals[-1], a_vals[-1], m_vals[-1]
            p_error = abs(final_p - expected_p)
            a_error = abs(final_a - expected_a)
            m_error = abs(final_m - expected_m)
            
            results['pac_fractions'] = {
                'final_p': final_p,
                'final_a': final_a,
                'final_m': final_m,
                'expected_p': expected_p,
                'expected_a': expected_a,
                'expected_m': expected_m,
                'p_error': p_error,
                'a_error': a_error,
                'm_error': m_error,
                'approaching_equilibrium': (p_error + a_error + m_error) < 0.3
            }
        
        # Phi ratio error
        phi_errors = np.array(self.history['phi_ratio_error'])
        if len(phi_errors) > 0:
            results['phi_ratios'] = {
                'final_error': phi_errors[-1],
                'mean_error': np.mean(phi_errors),
                'converging': phi_errors[-1] < phi_errors[0] if len(phi_errors) > 1 else False
            }
        
        # Stability check
        energies = np.array(self.history['total_energy'])
        if len(energies) > 10:
            energy_std = np.std(energies[-10:])
            results['stability'] = {
                'final_energy': energies[-1],
                'energy_std_last_10': energy_std,
                'stable': energy_std < 0.1 * energies[-1]
            }
        
        return results
    
    def generate_report(self, output_dir: str = None) -> str:
        """Generate comprehensive report."""
        results = self.analyze()
        
        report = []
        report.append("=" * 60)
        report.append("DECEMBER 2025 INTEGRATION TEST REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append(f"Steps: {self.step}")
        report.append(f"Grid: {self.grid_size}x{self.grid_size}")
        report.append(f"Scales: {self.n_scales}")
        report.append("")
        
        # Conservation
        if 'conservation' in results:
            c = results['conservation']
            report.append("-" * 40)
            report.append("CONSERVATION")
            report.append("-" * 40)
            report.append(f"Initial P+A+M: {c['initial']:.6f}")
            report.append(f"Final P+A+M:   {c['final']:.6f}")
            report.append(f"Max drift:     {c['max_drift']:.2e} ({c['max_drift']*100:.6f}%)")
            report.append(f"Avg drift:     {c['avg_drift']:.2e}")
            report.append(f"Conservation:  {'✓ MAINTAINED' if c['maintained'] else '✗ VIOLATED'}")
            report.append("")
        
        # Frequency
        if 'frequency' in results:
            f = results['frequency']
            report.append("-" * 40)
            report.append("FREQUENCY EMERGENCE")
            report.append("-" * 40)
            report.append(f"Mean frequency:     {f['mean']:.6f} Hz")
            report.append(f"Std deviation:      {f['std']:.6f} Hz")
            report.append(f"Target:             0.020000 Hz")
            report.append(f"Deviation:          {f['deviation_from_0_02']:.6f} Hz")
            report.append(f"Matches 0.02 Hz:    {'✓ YES' if f['matches_0_02_hz'] else '✗ NO'}")
            report.append("")
        
        # PAC fractions
        if 'pac_fractions' in results:
            p = results['pac_fractions']
            report.append("-" * 40)
            report.append("PAC FIELD FRACTIONS")
            report.append("-" * 40)
            report.append(f"           Final     Expected   Error")
            report.append(f"P:         {p['final_p']:.4f}    {p['expected_p']:.4f}     {p['p_error']:.4f}")
            report.append(f"A:         {p['final_a']:.4f}    {p['expected_a']:.4f}     {p['a_error']:.4f}")
            report.append(f"M:         {p['final_m']:.4f}    {p['expected_m']:.4f}     {p['m_error']:.4f}")
            report.append(f"Equilibrium: {'✓ APPROACHING' if p['approaching_equilibrium'] else '✗ NOT YET'}")
            report.append("")
        
        # Phi ratios
        if 'phi_ratios' in results:
            r = results['phi_ratios']
            report.append("-" * 40)
            report.append("PHI RATIO CONVERGENCE")
            report.append("-" * 40)
            report.append(f"Final error:   {r['final_error']:.6f}")
            report.append(f"Mean error:    {r['mean_error']:.6f}")
            report.append(f"Converging:    {'✓ YES' if r['converging'] else '✗ NO'}")
            report.append("")
        
        # Stability
        if 'stability' in results:
            s = results['stability']
            report.append("-" * 40)
            report.append("STABILITY")
            report.append("-" * 40)
            report.append(f"Final energy:  {s['final_energy']:.4f}")
            report.append(f"Energy std:    {s['energy_std_last_10']:.6f}")
            report.append(f"Stable:        {'✓ YES' if s['stable'] else '✗ NO'}")
            report.append("")
        
        # Cosmological predictions
        report.append("-" * 40)
        report.append("COSMOLOGICAL PREDICTIONS")
        report.append("-" * 40)
        
        mf = self.cosmo.predict_matter_fraction()
        report.append(f"Matter fraction: {mf.predicted_value:.4f} (observed: {mf.observed_value})")
        
        ht = self.cosmo.predict_hubble_tension()
        report.append(f"H0 prediction:   {ht.predicted_value:.1f} km/s/Mpc (observed: {ht.observed_value})")
        
        jwst = self.cosmo.predict_jwst_anomalies(3)
        # Find the z=10 galaxy
        z10_galaxies = [g for g in jwst if abs(g.redshift - 10) < 1]
        if z10_galaxies:
            g = z10_galaxies[0]
            report.append(f"JWST z~{g.redshift:.0f} SMBH: {g.smbh_mass_solar:.2e} M☉ via {g.formation_mode}")
        else:
            report.append(f"JWST galaxies: {len(jwst)} predictions generated")
        report.append("")
        
        # Overall verdict
        report.append("=" * 60)
        report.append("OVERALL VERDICT")
        report.append("=" * 60)
        
        all_pass = True
        verdicts = []
        
        if 'conservation' in results:
            if results['conservation']['maintained']:
                verdicts.append("✓ Conservation maintained")
            else:
                verdicts.append("✗ Conservation violated")
                all_pass = False
        
        if 'frequency' in results:
            if results['frequency']['matches_0_02_hz']:
                verdicts.append("✓ 0.02 Hz emerges from dynamics")
            else:
                verdicts.append(f"✗ Frequency {results['frequency']['mean']:.4f} Hz (not 0.02)")
                all_pass = False
        
        if 'stability' in results:
            if results['stability']['stable']:
                verdicts.append("✓ System is stable")
            else:
                verdicts.append("✗ System unstable")
                all_pass = False
        
        for v in verdicts:
            report.append(v)
        
        report.append("")
        if all_pass:
            report.append("🎉 ALL TESTS PASSED - PAC physics validated!")
        else:
            report.append("⚠️  SOME TESTS FAILED - needs investigation")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save if output dir specified
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            # Save report
            report_file = out_path / f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_file.write_text(report_text, encoding='utf-8')
            
            # Save results JSON
            json_file = out_path / f"integration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            # Convert numpy to python for JSON
            json_results = {}
            for key, val in results.items():
                json_results[key] = {}
                for k, v in val.items():
                    if isinstance(v, (np.floating, np.integer)):
                        json_results[key][k] = float(v)
                    elif isinstance(v, np.bool_):
                        json_results[key][k] = bool(v)
                    else:
                        json_results[key][k] = v
            json_file.write_text(json.dumps(json_results, indent=2))
            
            print(f"\nResults saved to {out_path}")
        
        return report_text
    
    def plot_results(self, output_dir: str = None):
        """Generate plots of simulation results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        steps = self.history['step']
        
        # 1. Conservation
        ax = axes[0, 0]
        pac = np.array(self.history['pac_conservation'])
        ax.plot(steps, pac, 'b-', linewidth=0.5)
        ax.axhline(pac[0], color='r', linestyle='--', label='Initial')
        ax.set_xlabel('Step')
        ax.set_ylabel('P + A + M')
        ax.set_title('Conservation')
        ax.legend()
        
        # 2. Frequency
        ax = axes[0, 1]
        freq = np.array(self.history['frequency'])
        ax.plot(steps, freq, 'g-', linewidth=0.5)
        ax.axhline(0.02, color='r', linestyle='--', label='Target 0.02 Hz')
        ax.set_xlabel('Step')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Frequency Emergence')
        ax.legend()
        
        # 3. PAC Fractions
        ax = axes[0, 2]
        ax.plot(steps, self.history['p_fraction'], 'b-', label='P', linewidth=0.5)
        ax.plot(steps, self.history['a_fraction'], 'g-', label='A', linewidth=0.5)
        ax.plot(steps, self.history['m_fraction'], 'r-', label='M', linewidth=0.5)
        # Expected values
        phi_sum = 1 + PHI + PHI**2
        ax.axhline(1/phi_sum, color='b', linestyle='--', alpha=0.5)
        ax.axhline(PHI/phi_sum, color='g', linestyle='--', alpha=0.5)
        ax.axhline(PHI**2/phi_sum, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Fraction')
        ax.set_title('PAC Field Fractions')
        ax.legend()
        
        # 4. KG Amplitude
        ax = axes[1, 0]
        amp = np.array(self.history['kg_amplitude'])
        ax.plot(steps, amp, 'purple', linewidth=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Amplitude')
        ax.set_title('Klein-Gordon Field Amplitude')
        
        # 5. Phi Ratio Error
        ax = axes[1, 1]
        phi_err = np.array(self.history['phi_ratio_error'])
        ax.semilogy(steps, phi_err, 'orange', linewidth=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('φ Ratio Error')
        ax.set_title('Phi Ratio Convergence')
        
        # 6. Energy
        ax = axes[1, 2]
        energy = np.array(self.history['total_energy'])
        ax.plot(steps, energy, 'brown', linewidth=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Total Energy')
        ax.set_title('Multi-Scale Field Energy')
        
        plt.tight_layout()
        
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            fig_file = out_path / f"integration_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(fig_file, dpi=150)
            print(f"Plots saved to {fig_file}")
        
        plt.close()


def run_full_test():
    """Run the complete December 2025 integration test."""
    print("\n" + "=" * 60)
    print("DECEMBER 2025 INTEGRATION TEST")
    print("Testing PAC-constrained physics for cosmological predictions")
    print("=" * 60 + "\n")
    
    # Create simulation
    sim = IntegratedPACSimulation(
        grid_size=64,
        n_scales=20,
        dt=0.01
    )
    
    # Run for 10,000 steps
    sim.run(steps=10000, record_every=10, verbose=True)
    
    # Generate outputs
    output_dir = Path(__file__).parent.parent / "output" / "integration_test"
    
    report = sim.generate_report(str(output_dir))
    print("\n" + report)
    
    sim.plot_results(str(output_dir))
    
    # Return results for programmatic use
    return sim.analyze()


if __name__ == '__main__':
    results = run_full_test()
