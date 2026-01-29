"""
Experiment 02: Observer Density Effects

Implements explicit observer density (ρ_obs) parameter and tests
the theoretical prediction: τ_collapse = τ_0 / (ρ_obs × Ξ)

Higher observer density should:
- Increase collapse/observation frequency
- Create more entanglement (connections)
- Shift system toward classical behavior
"""

import numpy as np
import math
from reality_seed.genesis import GenesisSeed
from reality_seed.pac_substrate import PACSubstrate

XI = 1 + math.pi / 55  # ≈ 1.0571


class ObserverDensityGenesisSeed(GenesisSeed):
    """Genesis with explicit observer density parameter."""
    
    def __init__(self, initial_value=1.0, observer_density=1.0):
        super().__init__(initial_value)
        self.observer_density = observer_density  # ρ_obs
        self.collapse_count = 0
        self.entanglement_count = 0
        
    def step(self):
        """Modified step with observer density effect."""
        result = super().step()
        
        # Observer density affects collapse probability
        # Theory: τ_collapse = τ_0 / (ρ × Ξ)
        # Probability of collapse in one step: 1 - exp(-ρ × Ξ × dt)
        collapse_prob = 1 - np.exp(-self.observer_density * XI * 0.01)
        
        if np.random.random() < collapse_prob:
            self._observer_induced_collapse()
            self.collapse_count += 1
            
        return result
    
    def _observer_induced_collapse(self):
        """
        Observer forces a pair of nodes to become entangled.
        
        In the paper's terms: observation creates informational dependency
        that PAC must satisfy. Here, we model this as forced connection.
        """
        nodes = list(self.substrate.nodes.values())
        if len(nodes) >= 2:
            # Random pair becomes entangled
            n1, n2 = np.random.choice(nodes, 2, replace=False)
            if n2 not in n1.neighbors:
                n1.neighbors.add(n2)
                n2.neighbors.add(n1)
                self.entanglement_count += 1


def run_experiment(n_steps=200, densities=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """Test observer density effects."""
    print("=" * 70)
    print("OBSERVER DENSITY EXPERIMENT")
    print("=" * 70)
    print()
    print("Theory: Higher observer density → more collapses → more entanglement")
    print("        Collapse rate ∝ ρ_obs × Ξ where Ξ = %.4f" % XI)
    print()
    
    results = []
    
    for rho in densities:
        genesis = ObserverDensityGenesisSeed(
            initial_value=1.0, 
            observer_density=rho
        )
        genesis.ratio_memory_weight = 0.5
        
        for _ in range(n_steps):
            genesis.run(1)
        
        n_nodes = len(genesis.substrate.nodes)
        n_connections = sum(len(n.neighbors) for n in genesis.substrate.nodes.values()) // 2
        
        results.append({
            'rho': rho,
            'nodes': n_nodes,
            'collapses': genesis.collapse_count,
            'entanglements': genesis.entanglement_count,
            'connections': n_connections
        })
        
        print("ρ=%.2f: %3d nodes, %3d collapses, %3d entanglements" % 
              (rho, n_nodes, genesis.collapse_count, genesis.entanglement_count))
    
    print()
    
    # Verify scaling
    print("Collapse rate scaling:")
    for r in results:
        expected_rate = r['rho'] * XI * 0.01 * n_steps
        actual_rate = r['collapses']
        ratio = actual_rate / expected_rate if expected_rate > 0 else 0
        print("  ρ=%.2f: expected~%.1f, actual=%d, ratio=%.2f" % 
              (r['rho'], expected_rate, actual_rate, ratio))
    
    return results


if __name__ == "__main__":
    run_experiment()
