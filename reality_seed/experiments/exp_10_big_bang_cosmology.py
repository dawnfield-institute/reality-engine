"""
Experiment 10: Big Bang Cosmology Interpretation

REFRAME: The singularity is not mass - it's PURE ENTROPY.

Big Bang in PAC terms:
1. Singularity = 1.0 entropy (infinite potential, zero structure)
2. First split = first actualization = time begins
3. Entanglement (connections) = causal structure = spacetime
4. "Mass" = stable, low-entropy configurations that resist splitting
5. Gravity = tendency of actualized structure to re-entangle

The dispersion we see IS the expansion of the universe!
The question: does structure crystallize out of this expansion?
"""

import json
import numpy as np
from collections import defaultdict
from scipy import stats
import math
import glob
import os

PHI = (1 + math.sqrt(5)) / 2
XI = 1 + math.pi / 55


def load_latest_trace():
    traces = glob.glob("results/full_trace_*.json")
    if not traces:
        return None
    latest = max(traces, key=os.path.getctime)
    print("Loading: %s" % latest)
    with open(latest, 'r') as f:
        return json.load(f)


def analyze_entropy_to_mass_conversion(trace):
    """
    Reframe: Value starts as entropy, becomes mass through stabilization.
    
    High entropy = high value, unstable (will split)
    Low entropy = low value, stable (resists splitting)
    Mass = nodes that DON'T split (crystallized structure)
    """
    print("\n" + "=" * 70)
    print("ENTROPY → MASS CONVERSION")
    print("=" * 70)
    
    # Track which nodes split vs stayed stable
    split_parents = set()
    for s in trace['steps']:
        if s['split_info']:
            split_parents.add(s['split_info']['parent_id'])
    
    # Final state
    final = trace['steps'][-1]
    final_ids = {n['id'] for n in final['nodes']}
    
    # Nodes that survived without splitting = "mass" (crystallized entropy)
    survivors = final_ids - split_parents
    splitters = final_ids & split_parents
    
    print("\nFinal state: %d nodes" % len(final['nodes']))
    print("  Stable (never split, became 'mass'): %d (%.1f%%)" % 
          (len(survivors), 100 * len(survivors) / len(final['nodes'])))
    print("  Continued splitting: %d" % len(splitters))
    
    # Value distribution of stable vs unstable
    final_map = {n['id']: n['value'] for n in final['nodes']}
    
    survivor_values = [final_map[nid] for nid in survivors if nid in final_map]
    splitter_values = [final_map[nid] for nid in splitters if nid in final_map]
    
    if survivor_values:
        print("\n'Mass' (stable) nodes:")
        print("  Mean value: %.6f" % np.mean(survivor_values))
        print("  Total mass: %.4f" % sum(survivor_values))
    
    if splitter_values:
        print("\n'Entropy' (still splitting) nodes:")
        print("  Mean value: %.6f" % np.mean(splitter_values))
        print("  Total entropy: %.4f" % sum(splitter_values))
    
    # Entropy conversion rate
    if survivor_values and splitter_values:
        mass_fraction = sum(survivor_values)
        entropy_fraction = sum(splitter_values)
        print("\nEntropy → Mass conversion:")
        print("  Mass (crystallized): %.1f%%" % (100 * mass_fraction))
        print("  Entropy (active): %.1f%%" % (100 * entropy_fraction))


def analyze_time_emergence(trace):
    """
    Time = entanglement = dependency chains.
    
    First split creates first dependency → time begins.
    Depth of genealogy = "age" of that branch.
    More connections = more causal structure = denser time.
    """
    print("\n" + "=" * 70)
    print("TIME EMERGENCE (entanglement as causality)")
    print("=" * 70)
    
    # Build genealogy
    parent_of = {}
    for s in trace['steps']:
        if s['split_info']:
            info = s['split_info']
            for cid in info['child_ids']:
                parent_of[cid] = info['parent_id']
    
    # Calculate depth (causal chain length = "age")
    def get_depth(node_id):
        depth = 0
        current = node_id
        while current in parent_of:
            current = parent_of[current]
            depth += 1
        return depth
    
    final = trace['steps'][-1]
    depths = [(n['id'], get_depth(n['id']), n['value']) for n in final['nodes']]
    
    # Correlation: deeper nodes = older = more "time" experienced
    depth_vals = np.array([d for _, d, _ in depths])
    value_vals = np.array([v for _, _, v in depths])
    
    print("\nCausal depth (time experienced):")
    print("  Max depth: %d" % max(depth_vals))
    print("  Mean depth: %.2f" % np.mean(depth_vals))
    
    # Do deeper (older) nodes have different values?
    corr, p = stats.pearsonr(depth_vals, value_vals)
    print("\nDepth-Value correlation: r=%.4f (p=%.4f)" % (corr, p))
    
    if p < 0.05 and corr < 0:
        print("  → Older nodes have LESS entropy (more crystallized)")
    elif p < 0.05 and corr > 0:
        print("  → Older nodes have MORE entropy (still active)")
    
    # Connection density = causal richness
    connections = [len(n['neighbors']) for n in final['nodes']]
    depths_only = [get_depth(n['id']) for n in final['nodes']]
    
    corr_conn, p_conn = stats.pearsonr(depths_only, connections)
    print("\nDepth-Connection correlation: r=%.4f (p=%.4f)" % (corr_conn, p_conn))
    if p_conn < 0.05 and corr_conn > 0:
        print("  → Older regions have denser causal structure!")


def analyze_expansion(trace):
    """
    The dispersion we see IS cosmic expansion.
    
    Rate of node creation = expansion rate.
    Does it match Hubble-like behavior?
    """
    print("\n" + "=" * 70)
    print("COSMIC EXPANSION ANALYSIS")
    print("=" * 70)
    
    steps = [s['step'] for s in trace['steps']]
    node_counts = [s['node_count'] for s in trace['steps']]
    
    # Already found: N ~ t^0.632 ≈ t^(1/φ)
    # This is SUB-LINEAR expansion (decelerating)
    
    log_t = np.log(np.array(steps[1:]))
    log_n = np.log(np.array(node_counts[1:]))
    
    slope, intercept, r, _, _ = stats.linregress(log_t, log_n)
    
    print("\nExpansion law: N(t) ∝ t^%.4f" % slope)
    print("  r² = %.4f" % (r**2))
    print("  1/φ = %.4f" % (1/PHI))
    print("  Difference from 1/φ: %.4f" % abs(slope - 1/PHI))
    
    if abs(slope - 1/PHI) < 0.05:
        print("\n  → GOLDEN RATIO EXPANSION!")
        print("     Expansion rate is governed by φ")
    
    # Deceleration?
    early_growth = (node_counts[50] - node_counts[10]) / 40
    late_growth = (node_counts[-1] - node_counts[-51]) / 50
    
    print("\nExpansion rate evolution:")
    print("  Early (steps 10-50): %.2f nodes/step" % early_growth)
    print("  Late (last 50 steps): %.2f nodes/step" % late_growth)
    
    if early_growth > late_growth:
        print("  → DECELERATING expansion (like early universe)")
    else:
        print("  → ACCELERATING expansion (like dark energy era)")


def analyze_structure_crystallization(trace):
    """
    Does structure crystallize out of the entropic expansion?
    
    Look for:
    - Clusters forming
    - Stable configurations
    - "Atoms" (bound multi-node systems)
    """
    print("\n" + "=" * 70)
    print("STRUCTURE CRYSTALLIZATION")
    print("=" * 70)
    
    final = trace['steps'][-1]
    nodes = final['nodes']
    node_map = {n['id']: n for n in nodes}
    
    # Find connected components
    visited = set()
    clusters = []
    
    for n in nodes:
        if n['id'] in visited:
            continue
        
        # BFS
        cluster = []
        queue = [n['id']]
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            if nid in node_map:
                cluster.append(node_map[nid])
                for nb_id in node_map[nid]['neighbors']:
                    if nb_id not in visited:
                        queue.append(nb_id)
        
        if cluster:
            clusters.append(cluster)
    
    # Analyze clusters
    cluster_sizes = [len(c) for c in clusters]
    isolated = sum(1 for s in cluster_sizes if s == 1)
    bound = sum(1 for s in cluster_sizes if s > 1)
    
    print("\nCluster analysis (bound structures):")
    print("  Total clusters: %d" % len(clusters))
    print("  Isolated nodes: %d (%.1f%%)" % (isolated, 100 * isolated / len(nodes)))
    print("  Bound structures (size > 1): %d" % bound)
    
    if bound > 0:
        multi_clusters = [c for c in clusters if len(c) > 1]
        
        print("\nBound structures (proto-atoms):")
        for i, c in enumerate(sorted(multi_clusters, key=len, reverse=True)[:5]):
            total_mass = sum(n['value'] for n in c)
            mean_mass = np.mean([n['value'] for n in c])
            print("  Cluster %d: %d nodes, total=%.5f, mean=%.5f" % 
                  (i+1, len(c), total_mass, mean_mass))
    
    # Structure fraction
    bound_value = sum(sum(n['value'] for n in c) for c in clusters if len(c) > 1)
    total_value = sum(n['value'] for n in nodes)
    
    print("\nCrystallization progress:")
    print("  Bound structure fraction: %.1f%% of total value" % (100 * bound_value / total_value))


def analyze_entanglement_pressure(trace):
    """
    Entanglement creates "pressure" that drives actualization.
    
    High entropy → wants to resolve → splits
    Entanglement → creates dependency → stabilizes
    
    Balance between expansion pressure and entanglement binding.
    """
    print("\n" + "=" * 70)
    print("ENTANGLEMENT PRESSURE vs EXPANSION")
    print("=" * 70)
    
    # Track entropy (node count) vs structure (connections)
    steps = []
    node_counts = []
    connection_counts = []
    
    for s in trace['steps'][::10]:  # Sample
        total_connections = sum(len(n['neighbors']) for n in s['nodes']) // 2
        steps.append(s['step'])
        node_counts.append(s['node_count'])
        connection_counts.append(total_connections)
    
    steps = np.array(steps)
    nodes = np.array(node_counts)
    connections = np.array(connection_counts)
    
    # Ratio: connections per node = entanglement density
    density = connections / nodes
    
    print("\nEntanglement density evolution:")
    print("  Initial: %.4f connections/node" % density[0])
    print("  Final: %.4f connections/node" % density[-1])
    
    # Is density increasing or decreasing?
    slope, _, r, p, _ = stats.linregress(steps, density)
    print("  Trend: slope=%.6f (p=%.4f)" % (slope, p))
    
    if slope > 0 and p < 0.05:
        print("  → STRUCTURE WINNING: Entanglement growing faster than expansion")
    elif slope < 0 and p < 0.05:
        print("  → EXPANSION WINNING: Space expanding faster than structure forms")
    else:
        print("  → EQUILIBRIUM: Expansion and structure balanced")


def main():
    trace = load_latest_trace()
    if not trace:
        print("No trace found!")
        return
    
    print("\n" + "=" * 70)
    print("BIG BANG COSMOLOGY REINTERPRETATION")
    print("=" * 70)
    print("""
    REFRAME:
    - Singularity = pure entropy (potential), not mass
    - Splits = actualization (entropy → structure)
    - Connections = entanglement = time/causality
    - Stable nodes = crystallized mass
    - Dispersion = cosmic expansion
    """)
    
    analyze_entropy_to_mass_conversion(trace)
    analyze_time_emergence(trace)
    analyze_expansion(trace)
    analyze_structure_crystallization(trace)
    analyze_entanglement_pressure(trace)
    
    print("\n" + "=" * 70)
    print("COSMOLOGICAL SUMMARY")
    print("=" * 70)
    print("""
    What we're seeing:
    1. Initial singularity (entropy=1.0) actualizes via splitting
    2. Each split creates time (dependency = causality)
    3. Expansion follows t^(1/φ) - golden ratio cosmology!
    4. Some entropy crystallizes into stable "mass"
    5. Entanglement creates structure against expansion
    
    The "anti-gravity" we saw was actually CORRECT:
    - High entropy wants to disperse (expansion)
    - Gravity would emerge when structure creates feedback loops
    - We need entanglement-induced attraction for gravity
    """)


if __name__ == "__main__":
    main()
