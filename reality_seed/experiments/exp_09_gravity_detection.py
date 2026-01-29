"""
Experiment 09: Gravity Detection

Does gravity-like behavior emerge from PAC dynamics?

From recursive_gravity experiment, gravity signature is:
1. Feedback proportional to exp(-distance) or 1/r²
2. "Massive" nodes attract others
3. Clustering around high-value nodes
4. Memory of interactions (entropy accumulation)

We check: Do high-value nodes accumulate connections?
         Do nodes "orbit" or cluster around massive centers?
         Is there a distance-dependent attraction?
"""

import json
import numpy as np
from collections import defaultdict
from scipy import stats
import math
import glob
import os

PHI = (1 + math.sqrt(5)) / 2


def load_latest_trace():
    """Load most recent trace file."""
    traces = glob.glob("results/full_trace_*.json")
    if not traces:
        print("No trace files found!")
        return None
    latest = max(traces, key=os.path.getctime)
    print("Loading: %s" % latest)
    with open(latest, 'r') as f:
        return json.load(f)


def analyze_mass_attraction(trace):
    """Do high-value nodes attract more connections?"""
    print("\n" + "=" * 70)
    print("MASS-ATTRACTION ANALYSIS")
    print("=" * 70)
    
    final = trace['steps'][-1]
    nodes = final['nodes']
    
    # Build lookup
    node_map = {n['id']: n for n in nodes}
    
    # For each node: (value, degree)
    value_degree = []
    for n in nodes:
        degree = len(n['neighbors'])
        value_degree.append((n['value'], degree))
    
    values, degrees = zip(*value_degree)
    values = np.array(values)
    degrees = np.array(degrees)
    
    # Correlation: do massive nodes have more connections?
    corr, p = stats.pearsonr(values, degrees)
    
    print("\nValue-Degree correlation:")
    print("  Pearson r = %.4f (p = %.4f)" % (corr, p))
    
    if p < 0.05 and corr > 0:
        print("  → GRAVITY SIGNATURE: Massive nodes attract more connections!")
    elif p < 0.05 and corr < 0:
        print("  → ANTI-GRAVITY: Massive nodes repel!")
    else:
        print("  → No significant mass-degree relationship")
    
    # Binned analysis
    print("\nDegree by value quintile:")
    quintiles = np.percentile(values, [20, 40, 60, 80, 100])
    for i, q in enumerate(quintiles):
        if i == 0:
            mask = values <= q
            label = "Q1 (lowest)"
        else:
            mask = (values > quintiles[i-1]) & (values <= q)
            label = "Q%d" % (i+1)
        
        if np.sum(mask) > 0:
            mean_deg = np.mean(degrees[mask])
            mean_val = np.mean(values[mask])
            print("  %s: mean_value=%.5f, mean_degree=%.3f" % (label, mean_val, mean_deg))


def analyze_clustering_around_mass(trace):
    """Do nodes cluster around high-value centers?"""
    print("\n" + "=" * 70)
    print("CLUSTERING AROUND MASS CENTERS")
    print("=" * 70)
    
    final = trace['steps'][-1]
    nodes = final['nodes']
    node_map = {n['id']: n for n in nodes}
    
    # Find the "massive" nodes (top 10% by value)
    values = np.array([n['value'] for n in nodes])
    threshold = np.percentile(values, 90)
    massive = [n for n in nodes if n['value'] >= threshold]
    
    print("\n'Massive' nodes (top 10%% by value): %d nodes" % len(massive))
    print("  Total mass: %.4f (%.1f%% of total)" % 
          (sum(n['value'] for n in massive), 100 * sum(n['value'] for n in massive)))
    
    # For each massive node, count neighbors within 1-hop and 2-hop
    for m in sorted(massive, key=lambda x: x['value'], reverse=True)[:5]:
        neighbors_1 = set(m['neighbors'])
        neighbors_2 = set()
        for nb_id in m['neighbors']:
            if nb_id in node_map:
                neighbors_2.update(node_map[nb_id]['neighbors'])
        neighbors_2.discard(m['id'])
        neighbors_2 -= neighbors_1
        
        total_neighbor_value = sum(node_map[nid]['value'] for nid in neighbors_1 if nid in node_map)
        
        print("\n  Mass center %s: value=%.5f" % (m['id'][:8], m['value']))
        print("    1-hop neighbors: %d (total value: %.5f)" % (len(neighbors_1), total_neighbor_value))
        print("    2-hop neighbors: %d" % len(neighbors_2))


def analyze_neighbor_value_distribution(trace):
    """Do connected nodes have similar or different values?"""
    print("\n" + "=" * 70)
    print("NEIGHBOR VALUE RELATIONSHIPS")
    print("=" * 70)
    
    final = trace['steps'][-1]
    nodes = final['nodes']
    node_map = {n['id']: n for n in nodes}
    
    # Collect (node_value, neighbor_value) pairs
    pairs = []
    ratios = []
    
    for n in nodes:
        for nb_id in n['neighbors']:
            if nb_id in node_map:
                nb = node_map[nb_id]
                pairs.append((n['value'], nb['value']))
                if nb['value'] > 0:
                    ratios.append(n['value'] / nb['value'])
    
    if not pairs:
        print("No connected pairs found!")
        return
    
    v1, v2 = zip(*pairs)
    v1, v2 = np.array(v1), np.array(v2)
    
    # Correlation
    corr, p = stats.pearsonr(v1, v2)
    print("\nNeighbor value correlation: r=%.4f (p=%.4f)" % (corr, p))
    
    # Value ratio between neighbors
    ratios = np.array(ratios)
    ratios = np.minimum(ratios, 1/ratios)  # Normalize to [0, 1]
    
    print("\nNeighbor value ratios (smaller/larger):")
    print("  Mean: %.4f" % np.mean(ratios))
    print("  Median: %.4f" % np.median(ratios))
    print("  Std: %.4f" % np.std(ratios))
    
    # Near golden ratio?
    near_phi = np.sum(np.abs(ratios - 1/PHI) < 0.1) / len(ratios) * 100
    print("  Near 1/φ (±0.1): %.1f%%" % near_phi)
    
    # Do big nodes connect to big or small?
    big_threshold = np.percentile([n['value'] for n in nodes], 80)
    big_big = sum(1 for a, b in pairs if a > big_threshold and b > big_threshold)
    big_small = sum(1 for a, b in pairs if a > big_threshold and b <= big_threshold)
    
    print("\nBig node (top 20%%) connections:")
    print("  Big-Big: %d" % (big_big // 2))  # Divide by 2 for double counting
    print("  Big-Small: %d" % big_small)
    
    if big_big > big_small:
        print("  → CLUSTERING: Big attracts big!")
    elif big_small > big_big:
        print("  → HIERARCHY: Big attracts small (gravity-like)!")


def analyze_genealogical_gravity(trace):
    """Does value flow toward certain lineages (gravitational wells)?"""
    print("\n" + "=" * 70)
    print("GENEALOGICAL GRAVITY (value accumulation in lineages)")
    print("=" * 70)
    
    # Track parent-child relationships
    parent_of = {}
    children_of = defaultdict(list)
    birth_value = {}
    
    for s in trace['steps']:
        if s['split_info']:
            info = s['split_info']
            pid = info['parent_id']
            for i, cid in enumerate(info['child_ids']):
                parent_of[cid] = pid
                children_of[pid].append(cid)
                birth_value[cid] = info['child_values'][i]
    
    # Find lineages (descendants of root)
    def get_descendants(node_id, depth=0, max_depth=10):
        if depth > max_depth:
            return []
        desc = [node_id]
        for child in children_of.get(node_id, []):
            desc.extend(get_descendants(child, depth + 1, max_depth))
        return desc
    
    # Find roots
    all_children = set(parent_of.keys())
    all_parents = set(parent_of.values())
    roots = all_parents - all_children
    
    print("\nRoot lineages: %d" % len(roots))
    
    # Analyze each root's total descendant value
    final = trace['steps'][-1]
    final_values = {n['id']: n['value'] for n in final['nodes']}
    
    for root in list(roots)[:5]:
        descendants = get_descendants(root)
        surviving = [d for d in descendants if d in final_values]
        total_value = sum(final_values.get(d, 0) for d in surviving)
        
        print("\n  Lineage from %s:" % root[:8])
        print("    Total descendants: %d" % len(descendants))
        print("    Surviving: %d" % len(surviving))
        print("    Total value: %.4f" % total_value)


def analyze_temporal_gravity(trace):
    """Does 'mass' (high value) persist and grow over time?"""
    print("\n" + "=" * 70)
    print("TEMPORAL GRAVITY (mass persistence)")
    print("=" * 70)
    
    # Track value of top nodes over time
    value_history = defaultdict(list)
    
    for s in trace['steps']:
        for n in s['nodes']:
            value_history[n['id']].append((s['step'], n['value']))
    
    # Find nodes that existed for a long time
    long_lived = [(nid, len(hist)) for nid, hist in value_history.items() if len(hist) > 100]
    long_lived.sort(key=lambda x: x[1], reverse=True)
    
    print("\nLong-lived nodes (>100 steps): %d" % len(long_lived))
    
    # Did their value grow or shrink?
    growers = 0
    shrinkers = 0
    stable = 0
    
    for nid, _ in long_lived[:50]:
        hist = value_history[nid]
        first_val = hist[0][1]
        last_val = hist[-1][1]
        
        if last_val > first_val * 1.1:
            growers += 1
        elif last_val < first_val * 0.9:
            shrinkers += 1
        else:
            stable += 1
    
    print("\nValue evolution of top 50 long-lived nodes:")
    print("  Growers (>10%% gain): %d" % growers)
    print("  Shrinkers (>10%% loss): %d" % shrinkers)
    print("  Stable: %d" % stable)
    
    if growers > shrinkers:
        print("  → GRAVITY: Long-lived nodes accumulate mass!")
    elif shrinkers > growers:
        print("  → DISPERSION: Long-lived nodes lose mass!")
    else:
        print("  → EQUILIBRIUM: Value is conserved locally")


def main():
    trace = load_latest_trace()
    if not trace:
        return
    
    print("\n" + "=" * 70)
    print("GRAVITY DETECTION IN REALITY_SEED")
    print("=" * 70)
    print("\nLooking for emergent gravity signatures:")
    print("  - Mass attracts connections")
    print("  - Clustering around mass centers")
    print("  - Value-dependent neighbor relationships")
    print("  - Gravitational wells in lineages")
    print("  - Mass accumulation over time")
    
    analyze_mass_attraction(trace)
    analyze_clustering_around_mass(trace)
    analyze_neighbor_value_distribution(trace)
    analyze_genealogical_gravity(trace)
    analyze_temporal_gravity(trace)
    
    print("\n" + "=" * 70)
    print("GRAVITY DETECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
