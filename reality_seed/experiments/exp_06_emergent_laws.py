"""
Experiment 06: Emergent Laws and Mass

PURELY OBSERVATIONAL - we impose nothing, we only watch.

Questions:
1. Do any consistent relationships emerge? (laws)
2. Do stable value concentrations form? (mass)
3. What ratios/frequencies recur without being imposed?

Method: Run vanilla simulation, record everything, find patterns.
"""

import numpy as np
import math
from collections import defaultdict, Counter
from reality_seed.genesis import GenesisSeed


PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
XI = 1 + math.pi / 55


class EmergentLawObserver:
    """
    Passive observer - records what happens, never intervenes.
    """
    
    def __init__(self):
        # Split ratio observations
        self.observed_ratios = []
        
        # Value distribution snapshots
        self.value_snapshots = []
        
        # Lifetime tracking (how long nodes survive)
        self.birth_times = {}
        self.death_times = {}
        
        # Relationship observations
        self.parent_child_ratios = []  # ratio of child/parent values
        self.sibling_ratios = []  # ratio between siblings at birth
        
        # "Mass" candidates - nodes that persist
        self.persistent_nodes = defaultdict(int)  # node_id -> survival_count
        
        # Recurring patterns
        self.value_histogram = []
        
    def observe_split(self, parent_value, child1_value, child2_value, step):
        """Record a split event without intervening."""
        if parent_value > 0:
            ratio = child1_value / parent_value
            self.observed_ratios.append(ratio)
            
            # Sibling ratio
            if child2_value > 0:
                sib_ratio = child1_value / child2_value
                self.sibling_ratios.append(sib_ratio)
            
            # Parent-child ratio
            self.parent_child_ratios.append(child1_value / parent_value)
            self.parent_child_ratios.append(child2_value / parent_value)
    
    def observe_state(self, nodes, step):
        """Snapshot current state."""
        values = [n.value for n in nodes]
        self.value_snapshots.append({
            'step': step,
            'values': values.copy(),
            'count': len(values),
            'total': sum(values),
            'max': max(values) if values else 0,
            'min': min(values) if values else 0,
        })
        
        # Track persistence
        for n in nodes:
            self.persistent_nodes[n.id] += 1
    
    def observe_birth(self, node_id, step):
        """Record when a node is born."""
        self.birth_times[node_id] = step
    
    def observe_death(self, node_id, step):
        """Record when a node dies (merged/removed)."""
        self.death_times[node_id] = step


def run_observation(n_steps=2000, snapshot_interval=10):
    """
    Pure observation run.
    
    We watch the simulation, record everything, look for patterns.
    """
    print("=" * 70)
    print("EMERGENT LAW OBSERVATION")
    print("=" * 70)
    print()
    print("Method: Run simulation, observe only, impose nothing")
    print("Looking for: emergent ratios, stable structures, recurring patterns")
    print()
    
    genesis = GenesisSeed(initial_value=1.0)
    genesis.ratio_memory_weight = 0.5  # Let it learn
    
    observer = EmergentLawObserver()
    
    # Track nodes for birth/death
    previous_nodes = set()
    
    print("Running %d steps..." % n_steps)
    
    for step in range(n_steps):
        # Snapshot before step
        nodes_before = set(genesis.substrate.nodes.keys())
        values_before = {nid: genesis.substrate.nodes[nid].value 
                        for nid in nodes_before}
        
        # Take step
        genesis.step()
        
        # Snapshot after
        nodes_after = set(genesis.substrate.nodes.keys())
        
        # Detect births
        new_nodes = nodes_after - nodes_before
        for nid in new_nodes:
            observer.observe_birth(nid, step)
        
        # Detect deaths
        dead_nodes = nodes_before - nodes_after
        for nid in dead_nodes:
            observer.observe_death(nid, step)
        
        # Detect splits (new nodes that sum to parent)
        if len(new_nodes) == 2 and len(dead_nodes) == 0:
            # Likely a split - find parent by value conservation
            new_values = [genesis.substrate.nodes[nid].value for nid in new_nodes]
            for parent_id, parent_val in values_before.items():
                if abs(sum(new_values) - parent_val) < 0.0001:
                    # Found the split
                    observer.observe_split(parent_val, new_values[0], new_values[1], step)
                    break
        
        # Periodic snapshots
        if step % snapshot_interval == 0:
            nodes = list(genesis.substrate.nodes.values())
            observer.observe_state(nodes, step)
        
        previous_nodes = nodes_after
    
    print("Observation complete.")
    print()
    
    # === ANALYSIS ===
    
    print("=" * 70)
    print("EMERGENT PATTERNS (not imposed, just observed)")
    print("=" * 70)
    print()
    
    # 1. Split ratio distribution
    print("1. SPLIT RATIO DISTRIBUTION")
    print("-" * 40)
    if observer.observed_ratios:
        ratios = np.array(observer.observed_ratios)
        print("   Observed %d splits" % len(ratios))
        print("   Mean ratio: %.4f" % np.mean(ratios))
        print("   Std ratio: %.4f" % np.std(ratios))
        print()
        
        # Check for clustering near special values
        near_phi = np.sum(np.abs(ratios - PHI_INV) < 0.05) / len(ratios) * 100
        near_half = np.sum(np.abs(ratios - 0.5) < 0.05) / len(ratios) * 100
        near_third = np.sum(np.abs(ratios - 0.333) < 0.05) / len(ratios) * 100
        
        print("   Clustering (±0.05):")
        print("     Near 1/φ (%.3f): %.1f%%" % (PHI_INV, near_phi))
        print("     Near 1/2 (0.500): %.1f%%" % near_half)
        print("     Near 1/3 (0.333): %.1f%%" % near_third)
        
        # Histogram
        print()
        print("   Ratio histogram:")
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(ratios, bins=bins)
        for i, count in enumerate(hist):
            bar = '█' * (count // 5)
            print("     %.1f-%.1f: %4d %s" % (bins[i], bins[i+1], count, bar))
    print()
    
    # 2. Sibling ratios (potential "mass" signature)
    print("2. SIBLING RATIOS (mass signature?)")
    print("-" * 40)
    if observer.sibling_ratios:
        sib = np.array(observer.sibling_ratios)
        # Normalize to always be >= 1
        sib = np.maximum(sib, 1/sib)
        
        print("   Observed %d sibling pairs" % len(sib))
        print("   Mean ratio: %.4f" % np.mean(sib))
        print("   Median ratio: %.4f" % np.median(sib))
        
        # Check for φ
        near_phi_sib = np.sum(np.abs(sib - PHI) < 0.1) / len(sib) * 100
        print("   Near φ (%.3f ± 0.1): %.1f%%" % (PHI, near_phi_sib))
        
        # Top recurring ratios
        rounded = np.round(sib, 2)
        counts = Counter(rounded)
        print()
        print("   Most common sibling ratios:")
        for ratio, count in counts.most_common(10):
            print("     %.2f: %d times" % (ratio, count))
    print()
    
    # 3. Node lifetimes (stability = mass?)
    print("3. NODE LIFETIMES (stability → mass?)")
    print("-" * 40)
    lifetimes = []
    for nid, birth in observer.birth_times.items():
        death = observer.death_times.get(nid, n_steps)
        lifetimes.append(death - birth)
    
    if lifetimes:
        lifetimes = np.array(lifetimes)
        print("   Mean lifetime: %.1f steps" % np.mean(lifetimes))
        print("   Max lifetime: %d steps" % np.max(lifetimes))
        print("   Survivors (lived full run): %d" % np.sum(lifetimes >= n_steps - 10))
        
        # Long-lived nodes are "massive"
        long_lived = np.sum(lifetimes > np.mean(lifetimes) * 2)
        print("   Long-lived (>2× mean): %d (%.1f%%)" % 
              (long_lived, 100 * long_lived / len(lifetimes)))
    print()
    
    # 4. Value concentrations (mass distribution)
    print("4. VALUE CONCENTRATIONS (mass distribution)")
    print("-" * 40)
    if observer.value_snapshots:
        final = observer.value_snapshots[-1]
        values = np.array(final['values'])
        
        print("   Final state: %d nodes" % len(values))
        print("   Total value: %.6f (conserved: %s)" % 
              (sum(values), "YES" if abs(sum(values) - 1.0) < 0.001 else "NO"))
        
        # Value distribution
        print()
        print("   Value distribution:")
        percentiles = [10, 25, 50, 75, 90, 99]
        for p in percentiles:
            print("     P%d: %.6f" % (p, np.percentile(values, p)))
        
        # "Heavy" nodes (potential mass)
        mean_val = np.mean(values)
        heavy = values[values > mean_val * 5]
        print()
        print("   'Heavy' nodes (>5× mean value): %d" % len(heavy))
        if len(heavy) > 0:
            print("   Heavy node values: %s" % 
                  ', '.join(["%.4f" % v for v in sorted(heavy, reverse=True)[:10]]))
    print()
    
    # 5. Emergent laws (consistent relationships)
    print("5. EMERGENT LAWS (consistent relationships)")
    print("-" * 40)
    
    # Check N(t) = 2t + 1 in early phase
    if len(observer.value_snapshots) > 10:
        early = observer.value_snapshots[:20]
        expected = [2 * s['step'] + 1 for s in early]
        actual = [s['count'] for s in early]
        matches = sum(1 for e, a in zip(expected, actual) if e == a)
        print("   N(t) = 2t + 1 match (first 20 snapshots): %d/20" % matches)
    
    # Check for recurring ratios that weren't imposed
    if observer.observed_ratios:
        # Most common ratio (binned)
        binned = np.round(np.array(observer.observed_ratios), 2)
        most_common_ratio = Counter(binned).most_common(1)[0]
        print("   Most common split ratio: %.2f (appeared %d times)" % most_common_ratio)
        
        # Check if it's near a "natural" constant
        r = most_common_ratio[0]
        if abs(r - PHI_INV) < 0.05:
            print("   → This is near 1/φ = %.3f (GOLDEN RATIO EMERGENCE)" % PHI_INV)
        elif abs(r - 0.5) < 0.05:
            print("   → This is near 1/2 (EQUAL SPLIT)")
        elif abs(r - 1/3) < 0.05:
            print("   → This is near 1/3")
    
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
    These patterns EMERGED from PAC dynamics alone.
    We did not impose φ, 55, or any specific structure.
    
    If ratios cluster near 1/φ, that's the system discovering golden ratio.
    If certain nodes persist, that's "mass" emerging from stability.
    If sibling ratios follow patterns, that's "law" emerging from dynamics.
    
    The question: which of these are fundamental vs. artifacts?
    """)
    
    return observer


if __name__ == "__main__":
    observer = run_observation(n_steps=2000)
