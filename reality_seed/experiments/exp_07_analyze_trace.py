"""
Post-Analysis for Full Trace

Reads the JSON trace and analyzes:
1. Split ratio evolution
2. Node genealogy (parent-child trees)
3. Mass emergence (persistent value concentrations)
4. Law detection (recurring patterns)
"""

import json
import numpy as np
from collections import defaultdict, Counter
import math
import sys
import glob
import os

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
SQRT5 = math.sqrt(5)


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


def analyze_splits(trace):
    """Analyze all split events."""
    print("\n" + "=" * 70)
    print("SPLIT ANALYSIS")
    print("=" * 70)
    
    ratios = []
    ratio_evolution = []  # (step, ratio)
    
    for step_data in trace['steps']:
        if step_data['split_info']:
            info = step_data['split_info']
            r = info['ratio']
            ratios.append(r)
            ratio_evolution.append((step_data['step'], r))
    
    if not ratios:
        print("No splits detected!")
        return
    
    ratios = np.array(ratios)
    
    print("\nTotal splits: %d" % len(ratios))
    print("Mean ratio: %.4f (uniform would be 0.5)" % np.mean(ratios))
    print("Std ratio: %.4f" % np.std(ratios))
    
    # Check for φ emergence
    near_phi_inv = np.abs(ratios - PHI_INV) < 0.05
    near_phi = np.abs(ratios - (1 - PHI_INV)) < 0.05  # Other child gets 1-φ⁻¹
    phi_related = near_phi_inv | near_phi
    
    print("\nφ-related splits (within 0.05 of 1/φ or 1-1/φ):")
    print("  Count: %d / %d (%.1f%%)" % (np.sum(phi_related), len(ratios), 100*np.sum(phi_related)/len(ratios)))
    
    # Evolution over time - do ratios converge?
    if len(ratio_evolution) > 20:
        early = [r for s, r in ratio_evolution[:50]]
        late = [r for s, r in ratio_evolution[-50:]]
        
        print("\nRatio evolution:")
        print("  Early (first 50): mean=%.4f, std=%.4f" % (np.mean(early), np.std(early)))
        print("  Late (last 50): mean=%.4f, std=%.4f" % (np.mean(late), np.std(late)))
        
        # Check if converging to any special value
        late_mean = np.mean(late)
        specials = {
            '1/2': 0.5,
            '1/φ': PHI_INV,
            '1/3': 1/3,
            '2/3': 2/3,
            '1/√5': 1/SQRT5,
        }
        closest = min(specials.items(), key=lambda x: abs(x[1] - late_mean))
        print("  Closest special value: %s (%.4f), distance=%.4f" % 
              (closest[0], closest[1], abs(closest[1] - late_mean)))


def analyze_genealogy(trace):
    """Build and analyze parent-child relationships."""
    print("\n" + "=" * 70)
    print("GENEALOGY ANALYSIS")
    print("=" * 70)
    
    # Build family tree
    parent_of = {}  # child_id -> parent_id
    children_of = defaultdict(list)  # parent_id -> [child_ids]
    birth_step = {}  # node_id -> step
    birth_value = {}  # node_id -> initial value
    
    for step_data in trace['steps']:
        if step_data['split_info']:
            info = step_data['split_info']
            pid = info['parent_id']
            for i, cid in enumerate(info['child_ids']):
                parent_of[cid] = pid
                children_of[pid].append(cid)
                birth_step[cid] = step_data['step']
                birth_value[cid] = info['child_values'][i]
    
    # Find root(s) - nodes with no parent
    all_children = set(parent_of.keys())
    all_parents = set(parent_of.values())
    roots = all_parents - all_children
    
    print("\nFamily tree:")
    print("  Total nodes ever created: %d" % len(birth_step))
    print("  Root nodes: %d" % len(roots))
    
    # Calculate depth for each node
    def get_depth(node_id):
        depth = 0
        current = node_id
        while current in parent_of:
            current = parent_of[current]
            depth += 1
        return depth
    
    depths = [get_depth(nid) for nid in birth_step.keys()]
    if depths:
        print("  Max depth: %d" % max(depths))
        print("  Mean depth: %.2f" % np.mean(depths))
        
        # Depth distribution
        depth_counts = Counter(depths)
        print("\n  Depth distribution:")
        for d in sorted(depth_counts.keys())[:15]:
            bar = '█' * (depth_counts[d] // 3)
            print("    Depth %2d: %3d nodes %s" % (d, depth_counts[d], bar))


def analyze_mass(trace):
    """Look for persistent value concentrations (mass)."""
    print("\n" + "=" * 70)
    print("MASS ANALYSIS (persistent value concentrations)")
    print("=" * 70)
    
    # Track each node's value over time
    node_history = defaultdict(list)  # node_id -> [(step, value), ...]
    
    for step_data in trace['steps']:
        step = step_data['step']
        for node in step_data['nodes']:
            node_history[node['id']].append((step, node['value']))
    
    # Find long-lived nodes
    lifetimes = {nid: len(history) for nid, history in node_history.items()}
    
    print("\nNode lifetimes:")
    print("  Total unique nodes: %d" % len(lifetimes))
    print("  Mean lifetime: %.1f steps" % np.mean(list(lifetimes.values())))
    print("  Max lifetime: %d steps" % max(lifetimes.values()))
    
    # "Massive" nodes - long-lived with significant value
    n_steps = trace['metadata']['n_steps']
    survivors = [nid for nid, lt in lifetimes.items() if lt >= n_steps * 0.9]
    
    print("\nSurvivors (lived 90%%+ of simulation): %d" % len(survivors))
    
    if survivors:
        # Get their final values
        final_step = trace['steps'][-1]
        final_nodes = {n['id']: n['value'] for n in final_step['nodes']}
        
        survivor_values = [(nid, final_nodes.get(nid, 0)) for nid in survivors if nid in final_nodes]
        survivor_values.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop survivors by value (potential 'mass'):")
        for nid, val in survivor_values[:10]:
            lifetime = lifetimes[nid]
            print("    %s: value=%.6f, lifetime=%d steps" % (nid[:12], val, lifetime))
    
    # Value stability - nodes whose value doesn't change much
    stable_nodes = []
    for nid, history in node_history.items():
        if len(history) > 10:
            values = [v for _, v in history]
            cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else float('inf')
            if cv < 0.1:  # <10% coefficient of variation
                stable_nodes.append((nid, np.mean(values), cv))
    
    print("\nStable nodes (CV < 10%%): %d" % len(stable_nodes))
    if stable_nodes:
        stable_nodes.sort(key=lambda x: x[1], reverse=True)
        print("  Top stable by value:")
        for nid, val, cv in stable_nodes[:5]:
            print("    %s: value=%.6f, CV=%.3f" % (nid[:12], val, cv))


def analyze_laws(trace):
    """Look for recurring patterns that could be 'laws'."""
    print("\n" + "=" * 70)
    print("LAW DETECTION (recurring patterns)")
    print("=" * 70)
    
    # 1. N(t) relationship
    steps = [s['step'] for s in trace['steps']]
    counts = [s['node_count'] for s in trace['steps']]
    
    # Check N = 2t + 1 (early phase)
    early_matches = sum(1 for i, (s, c) in enumerate(zip(steps[:50], counts[:50])) 
                       if c == 2*s + 1)
    print("\nN(t) = 2t + 1 matches (first 50 steps): %d/50" % early_matches)
    
    # 2. Conservation law
    totals = [s['total_value'] for s in trace['steps']]
    conservation_error = max(abs(t - 1.0) for t in totals)
    print("Conservation law: max error = %.10f" % conservation_error)
    
    # 3. Split ratio patterns
    all_ratios = []
    for step_data in trace['steps']:
        if step_data['split_info']:
            all_ratios.append(step_data['split_info']['ratio'])
    
    if all_ratios:
        # Bin ratios and look for peaks
        hist, edges = np.histogram(all_ratios, bins=20, range=(0, 1))
        peak_bin = np.argmax(hist)
        peak_ratio = (edges[peak_bin] + edges[peak_bin + 1]) / 2
        
        print("\nSplit ratio distribution peak: %.2f (bin %d with %d splits)" % 
              (peak_ratio, peak_bin, hist[peak_bin]))
        
        # Check for ratio periodicity
        if len(all_ratios) > 55:
            # Autocorrelation at lag 55
            r = np.array(all_ratios)
            r_mean = r - np.mean(r)
            autocorr_55 = np.sum(r_mean[:-55] * r_mean[55:]) / (len(r) * np.var(r))
            print("Split ratio autocorrelation at lag 55: %.4f" % autocorr_55)
    
    # 4. Step-55 periodicity in node count
    if len(counts) > 110:
        c = np.array(counts)
        c_mean = c - np.mean(c)
        autocorr_55 = np.sum(c_mean[:-55] * c_mean[55:]) / (len(c) * np.var(c))
        print("Node count autocorrelation at lag 55: %.4f" % autocorr_55)


def main():
    trace = load_latest_trace()
    if not trace:
        return
    
    print("\nTrace metadata:")
    print("  Steps: %d" % trace['metadata']['n_steps'])
    print("  Timestamp: %s" % trace['metadata']['timestamp'])
    
    analyze_splits(trace)
    analyze_genealogy(trace)
    analyze_mass(trace)
    analyze_laws(trace)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
