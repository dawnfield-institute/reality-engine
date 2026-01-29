"""
Experiment 04: Energy Field Coupling

Question: How does an oscillatory energy field affect dynamics?
         Does the 55-step period create resonance?

Method: Add sinusoidal energy with period 55, measure correlations
"""

import numpy as np
import math
from reality_seed.genesis import GenesisSeed


XI = 1 + math.pi / 55  # ≈ 1.0571


class EnergyFieldGenesisSeed(GenesisSeed):
    """
    GenesisSeed with oscillatory energy field coupling.
    
    Energy oscillates with period 55 steps, modulating:
    - Split threshold (higher energy = easier splits)
    - Contraction probability (low energy = merges)
    """
    
    def __init__(self, initial_value=1.0, energy_strength=0.5):
        super().__init__(initial_value)
        self.energy_strength = energy_strength
        self.base_split_threshold = 0.9  # Default
        self.contraction_count = 0
        self.current_step = 0  # Track time internally
        
    def get_energy(self, step):
        """Oscillatory energy with period 55."""
        return math.sin(2 * math.pi * step / 55)
    
    def step(self):
        # Get current energy
        energy = self.get_energy(self.current_step)
        self.current_step += 1
        
        # Modulate split threshold
        # High energy = lower threshold = easier splits
        self.split_threshold = self.base_split_threshold - 0.2 * energy * self.energy_strength
        
        # Normal step
        result = super().step()
        
        # Low energy promotes contractions (merges)
        if energy < -0.5:
            self._try_contraction(abs(energy) * self.energy_strength)
        
        return result
    
    def _try_contraction(self, probability):
        """Try to merge two nearby nodes."""
        nodes = list(self.substrate.nodes.values())
        if len(nodes) < 3:
            return
        
        if np.random.random() < probability * 0.1:
            # Pick two random nodes
            n1, n2 = np.random.choice(nodes, 2, replace=False)
            
            # Merge: transfer value and remove one
            n1.value += n2.value
            
            # Transfer neighbors
            for neighbor in list(n2.neighbors):
                if neighbor != n1:
                    n1.neighbors.add(neighbor)
                    neighbor.neighbors.discard(n2)
                    neighbor.neighbors.add(n1)
            
            # Remove n2
            if n2.id in self.substrate.nodes:
                del self.substrate.nodes[n2.id]
                self.contraction_count += 1


def run_experiment(n_steps=500, strengths=None):
    """Test energy coupling at various strengths."""
    if strengths is None:
        strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("=" * 70)
    print("ENERGY FIELD COUPLING")
    print("=" * 70)
    print()
    print("Energy field: E(t) = sin(2πt/55)")
    print("Period = 55 steps (matching Fibonacci emergence)")
    print()
    
    results = {}
    
    for strength in strengths:
        genesis = EnergyFieldGenesisSeed(energy_strength=strength)
        genesis.ratio_memory_weight = 0.5
        
        node_counts = []
        energies = []
        
        for step in range(n_steps):
            genesis.step()
            node_counts.append(len(genesis.substrate.nodes))
            energies.append(genesis.get_energy(step))
        
        final_nodes = len(genesis.substrate.nodes)
        
        # Correlate energy with node count changes
        if len(node_counts) > 1:
            deltas = np.diff(node_counts)
            corr = np.corrcoef(energies[1:], deltas)[0, 1]
        else:
            corr = 0
        
        results[strength] = {
            'final_nodes': final_nodes,
            'contractions': genesis.contraction_count,
            'energy_node_correlation': corr
        }
        
        print("Strength %.2f: %d nodes, %d contractions, corr=%.3f" % 
              (strength, final_nodes, genesis.contraction_count, corr))
    
    print()
    print("Analysis:")
    print("-" * 40)
    
    # Compare with/without energy
    baseline = results[0.0]['final_nodes']
    coupled = results[1.0]['final_nodes']
    reduction = (baseline - coupled) / baseline * 100
    
    print("  Baseline (no energy): %d nodes" % baseline)
    print("  Full coupling: %d nodes (%.1f%% reduction)" % (coupled, reduction))
    print("  Contractions at full: %d" % results[1.0]['contractions'])
    print()
    
    # Check phase at 55-step intervals
    print("Energy phase at t = 55n:")
    genesis = EnergyFieldGenesisSeed(energy_strength=1.0)
    for n in range(10):
        t = n * 55
        e = genesis.get_energy(t)
        print("  t=%3d: E=%.4f" % (t, e))
    
    return results


if __name__ == "__main__":
    run_experiment()
