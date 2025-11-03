"""
Reality Engine - Main Orchestrator

Coordinates field evolution and emergence detection.
This is where reality unfolds from pure field dynamics.

The engine doesn't simulate physics - it grows it.
"""

import torch
from typing import Dict, Tuple
from core.dawn_field import DawnField
from core.big_bang import BigBangEvent
from emergence.quantum import QuantumEmergence
from emergence.particles import ParticleEmergence
from utils.metrics import EmergenceMetrics


class RealityEngine:
    """
    Main engine - pure field evolution creating all of physics.
    
    No imposed laws - only emergence from recursive balance.
    
    Process:
    1. Initialize primordial field (E=I=M in perfect balance)
    2. Big Bang (symmetry breaking via herniation)
    3. Field evolution (PAC dynamics)
    4. Emergence detection (recognize physics as it crystallizes)
    5. Validation (check against known physics)
    """
    
    def __init__(self, shape: Tuple[int, int, int] = (128, 128, 128)):
        """
        Initialize the Reality Engine.
        
        Args:
            shape: 3D spatial dimensions for field lattice
        """
        print(f"🌟 Initializing Reality Engine")
        print(f"   Field dimensions: {shape}")
        
        # Create primordial field
        self.field = DawnField(shape=shape)
        
        # Metrics tracker
        self.metrics = EmergenceMetrics()
        
        # Track emerged phenomena
        self.emerged_physics = {}
        
        # Evolution history
        self.history = {
            'energy': [],
            'information': [],
            'memory': [],
            'herniations': [],
            'particles': [],
            'quantum_emerged_step': None,
            'particles_emerged_step': None
        }
        
        print("   ✓ Reality Engine initialized")
        
    def big_bang(self, seed_perturbation: float = 1e-10, 
                 multipoint: bool = False, num_seeds: int = 5) -> Dict:
        """
        Initialize universe via symmetry breaking.
        
        Args:
            seed_perturbation: Magnitude of initial quantum fluctuation
            multipoint: Use multiple initial fluctuations instead of single center
            num_seeds: Number of seeds if multipoint=True
            
        Returns:
            Space-time characterization dictionary
        """
        print("\n" + "="*60)
        event = BigBangEvent(self.field)
        
        if multipoint:
            spacetime_info = event.trigger_multipoint(
                num_seeds=num_seeds,
                seed_perturbation=seed_perturbation
            )
        else:
            spacetime_info = event.trigger(seed_perturbation=seed_perturbation)
        
        print("="*60 + "\n")
        
        return spacetime_info
        
    def evolve(self, steps: int = 10000, check_interval: int = 100, 
              verbose: bool = True) -> Dict:
        """
        Let reality unfold from pure field dynamics.
        
        Args:
            steps: Number of evolution steps
            check_interval: How often to check for emergence
            verbose: Print progress updates
            
        Returns:
            Final report with all emerged phenomena
        """
        if verbose:
            print(f"\n⏳ Evolving universe for {steps} steps...")
            print(f"   Check interval: every {check_interval} steps\n")
        
        for step in range(steps):
            # Core field evolution (minimal stats, no GPU sync)
            self.field.evolve_step()
            
            # Record metrics periodically
            if step % check_interval == 0:
                self.record_state()
                self.check_emergence(step)
                
                if verbose and step % (check_interval * 10) == 0:
                    # Only get detailed stats when printing (forces GPU sync)
                    step_stats = self.field.get_statistics()
                    self.print_status(step, step_stats)
        
        # Final recording
        self.record_state()
        self.check_emergence(steps)
        
        if verbose:
            print("\n✅ Evolution complete!")
            self.print_final_summary()
        
        return self.generate_report()
        
    def record_state(self) -> None:
        """Record current field state to history."""
        # Use last herniation count (already computed in evolve_step)
        self.history['herniations'].append(self.field.herniation_history[-1] if self.field.herniation_history else 0)
        
        # Batch GPU→CPU transfers to minimize syncs
        stats = torch.stack([
            self.field.E.mean(),
            self.field.I.mean(),
            self.field.M.mean()
        ]).cpu()
        
        self.history['energy'].append(stats[0].item())
        self.history['information'].append(stats[1].item())
        self.history['memory'].append(stats[2].item())
        
        # Check conservation (uses more .item() calls but only every 500 steps)
        conservation = self.metrics.check_conservation(self.field)
        
    def check_emergence(self, step: int) -> None:
        """
        Check what's emerging - don't impose physics, recognize it.
        
        Args:
            step: Current evolution step
        """
        # Check for quantum behavior emergence
        if 'quantum' not in self.emerged_physics:
            qe = QuantumEmergence(self.field)
            if qe.has_quantum_behavior():
                self.emerged_physics['quantum'] = qe
                self.history['quantum_emerged_step'] = step
                print(f"\n   🔬 QUANTUM MECHANICS EMERGED at step {step}!")
                
                # Get quantum stats
                qstats = qe.measure_quantum_statistics()
                print(f"      Excitations: {qstats['num_excitations']}")
                print(f"      Born rule valid: {qstats['born_rule_valid']}")
                print(f"      Superposition detected: {qstats['superposition_detected']}")
        
        # Check for particle formation
        if 'particles' not in self.emerged_physics and step > 1000:
            pe = ParticleEmergence(self.field)
            particles = pe.identify_particles()
            
            if len(particles) > 0:
                self.emerged_physics['particles'] = pe
                self.history['particles_emerged_step'] = step
                print(f"\n   ⚛️  PARTICLES EMERGED at step {step}!")
                
                # Get particle stats
                pstats = pe.get_particle_statistics(particles)
                print(f"      Count: {pstats['count']}")
                print(f"      Mean mass: {pstats['mean_mass']:.4f}")
                print(f"      Mean stability: {pstats['mean_stability']:.4f}")
                
                self.history['particles'].append(len(particles))
    
    def print_status(self, step: int, stats: Dict) -> None:
        """
        Print current evolution status.
        
        Args:
            step: Current step
            stats: Step statistics
        """
        # Build base status line
        status = (f"   t={step:6d} | " +
                 f"E={stats['mean_energy']:8.4f} | " +
                 f"I={stats['mean_info']:8.4f} | " +
                 f"M={stats['mean_memory']:8.4f} | " +
                 f"H={stats['herniations']:3d}")
        
        # Add quantum stats if available
        if 'quantum_coherence' in stats:
            status += (f" | QC={stats['quantum_coherence']:6.4f} | " +
                      f"Ent={stats['entanglement']:6.4f} | " +
                      f"Born={stats['born_rule_compliance']:6.4f}")
        
        print(status)
    
    def print_final_summary(self) -> None:
        """Print final summary of emerged physics."""
        print("\n" + "="*60)
        print(" EMERGED PHYSICS SUMMARY")
        print("="*60)
        
        print(f"\n Phenomena that emerged:")
        if 'quantum' in self.emerged_physics:
            print(f"   ✓ Quantum Mechanics (step {self.history['quantum_emerged_step']})")
            qe = self.emerged_physics['quantum']
            qstats = qe.measure_quantum_statistics()
            print(f"      Excitations: {qstats['num_excitations']}")
            print(f"      Born rule: {'✓' if qstats['born_rule_valid'] else '✗'}")
            print(f"      Superposition: {'✓' if qstats['superposition_detected'] else '✗'}")
            print(f"      Coherence: {qstats.get('coherence', 0.0):.4f}")
            print(f"      Entanglement: {qstats.get('entanglement', 0.0):.4f}")
        else:
            print(f"   ✗ Quantum Mechanics (not yet emerged)")
            
        if 'particles' in self.emerged_physics:
            print(f"   ✓ Particles (step {self.history['particles_emerged_step']})")
            particles = self.emerged_physics['particles'].identify_particles()
            print(f"      Final count: {len(particles)}")
        else:
            print(f"   ✗ Particles (not yet emerged)")
        
        # Conservation check
        print(f"\n PAC Conservation:")
        conservation = self.metrics.check_conservation(self.field)
        print(f"   Total: {conservation['total']:.6f}")
        print(f"   Stability: {conservation['stability']:.6f}")
        print(f"   Conserved: {'✓' if conservation['conserved'] else '✗'}")
        print(f"   Correlation: {conservation['correlation']:.4f} (target: 0.964)")
        print(f"   Target met: {'✓' if conservation.get('target_met', False) else '✗'}")
        
        # Resonance check
        if len(self.metrics.conservation_history) >= 100:
            resonance = self.metrics.detect_resonance()
            print(f"\n Universal Resonance:")
            print(f"   Detected: {'✓' if resonance['detected'] else '✗'}")
            if resonance['detected']:
                print(f"   Frequency: {resonance['detected_freq']:.6f} Hz (target: 0.020 Hz)")
                print(f"   Error: {resonance['error']:.6f}")
        
        # Gravity check
        gravity = self.metrics.validate_gravity_emergence(self.field)
        print(f"\n Gravitational Force:")
        print(f"   Emerged: {'✓' if gravity.get('emerged', False) else '✗'}")
        if gravity.get('emerged', False):
            print(f"   Potential slope: {gravity['slope']:.4f} (expected: -1.0)")
            print(f"   Error: {gravity['error']:.4f}")
        
        print("\n" + "="*60 + "\n")
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive report of simulation.
        
        Returns:
            Dictionary with all results
        """
        # Get current particles if they exist
        particles = []
        if 'particles' in self.emerged_physics:
            particles = self.emerged_physics['particles'].identify_particles()
        
        # Get quantum stats if emerged
        quantum_stats = {}
        if 'quantum' in self.emerged_physics:
            quantum_stats = self.emerged_physics['quantum'].measure_quantum_statistics()
        
        report = {
            'emerged_phenomena': list(self.emerged_physics.keys()),
            'total_time': self.field.time,
            'total_steps': self.field.step_count,
            'final_state': {
                'energy': self.field.E.mean().item(),
                'information': self.field.I.mean().item(),
                'memory': self.field.M.mean().item()
            },
            'conservation': self.metrics.check_conservation(self.field),
            'resonance': self.metrics.detect_resonance() if len(self.metrics.conservation_history) >= 100 else {},
            'gravity': self.metrics.validate_gravity_emergence(self.field),
            'quantum': quantum_stats,
            'particles': {
                'count': len(particles),
                'emerged_step': self.history['particles_emerged_step'],
                'list': particles
            },
            'history': self.history,
            'validation': {
                'pac_conservation': self.metrics.check_conservation(self.field).get('target_met', False),
                'resonance_detected': self.metrics.detect_resonance().get('detected', False) if len(self.metrics.conservation_history) >= 100 else False,
                'quantum_emerged': 'quantum' in self.emerged_physics,
                'particles_emerged': 'particles' in self.emerged_physics,
                'gravity_emerged': self.metrics.validate_gravity_emergence(self.field).get('emerged', False)
            }
        }
        
        return report
    
    def save_state(self, filepath: str) -> None:
        """
        Save complete state to file.
        
        Args:
            filepath: Path to save file
        """
        import pickle
        
        state = {
            'field': self.field.get_state(),
            'history': self.history,
            'emerged_physics': list(self.emerged_physics.keys()),
            'metrics': self.metrics.generate_report()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"💾 State saved to {filepath}")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"RealityEngine(shape={self.field.E.shape}, " +
                f"time={self.field.time:.2f}, " +
                f"emerged={list(self.emerged_physics.keys())})")
