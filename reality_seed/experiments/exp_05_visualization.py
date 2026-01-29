"""
Experiment 05: Visualization & Time Series

Question: What does the dynamics look like step-by-step?
         Can we identify phase transitions?

Method: Collect detailed time series for visualization
"""

import numpy as np
import math
from reality_seed.genesis import GenesisSeed


PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618


def run_experiment(n_steps=500, save_data=False):
    """Collect time series data for visualization."""
    print("=" * 70)
    print("TIME SERIES VISUALIZATION DATA")
    print("=" * 70)
    print()
    
    genesis = GenesisSeed(initial_value=1.0)
    genesis.ratio_memory_weight = 0.5
    
    # Data collectors
    time_series = {
        'step': [],
        'node_count': [],
        'total_value': [],
        'min_value': [],
        'max_value': [],
        'mean_value': [],
        'std_value': [],
        'splits': [],
        'merges': [],
        'cumulative_splits': [],
        'cumulative_merges': [],
    }
    
    cumulative_splits = 0
    cumulative_merges = 0
    last_node_count = 1
    
    for step in range(n_steps):
        # Take step
        genesis.step()
        
        # Collect data
        nodes = list(genesis.substrate.nodes.values())
        values = [n.value for n in nodes]
        current_count = len(nodes)
        
        # Track splits and merges
        delta = current_count - last_node_count
        splits = max(0, delta)  # Positive = splits
        merges = max(0, -delta)  # Negative = merges
        cumulative_splits += splits
        cumulative_merges += merges
        
        time_series['step'].append(step)
        time_series['node_count'].append(current_count)
        time_series['total_value'].append(sum(values))
        time_series['min_value'].append(min(values))
        time_series['max_value'].append(max(values))
        time_series['mean_value'].append(np.mean(values))
        time_series['std_value'].append(np.std(values))
        time_series['splits'].append(splits)
        time_series['merges'].append(merges)
        time_series['cumulative_splits'].append(cumulative_splits)
        time_series['cumulative_merges'].append(cumulative_merges)
        
        last_node_count = current_count
    
    # Analysis
    print("Time series collected: %d steps" % n_steps)
    print()
    
    # Find growth phase end
    node_counts = time_series['node_count']
    growth_phase_end = None
    for i in range(len(node_counts) - 55):
        if node_counts[i + 55] <= node_counts[i] * 1.01:  # <1% growth over 55 steps
            growth_phase_end = i
            break
    
    if growth_phase_end:
        print("Growth phase ends: step %d (/ 55 = %.2f)" % 
              (growth_phase_end, growth_phase_end / 55))
    else:
        print("Growth phase: ongoing through %d steps" % n_steps)
    
    print()
    print("Peak values:")
    print("  Max nodes: %d at step %d" % 
          (max(node_counts), node_counts.index(max(node_counts))))
    print("  Min mean value: %.6f at step %d" % 
          (min(time_series['mean_value']), 
           time_series['mean_value'].index(min(time_series['mean_value']))))
    
    print()
    print("Dynamics totals:")
    print("  Total splits: %d" % cumulative_splits)
    print("  Total merges: %d" % cumulative_merges)
    print("  Net creation: %d nodes" % (cumulative_splits - cumulative_merges))
    
    # Step-by-step for first 20 steps
    print()
    print("Step-by-step (first 20):")
    print("-" * 60)
    print("%4s  %5s  %8s  %8s  %10s" % 
          ("Step", "Nodes", "MeanVal", "Split?", "φ-distance"))
    
    genesis2 = GenesisSeed(initial_value=1.0)
    genesis2.ratio_memory_weight = 0.5
    
    for step in range(20):
        nodes = list(genesis2.substrate.nodes.values())
        values = [n.value for n in nodes]
        mean_val = np.mean(values)
        
        # Check for splits
        n_before = len(genesis2.substrate.nodes)
        genesis2.step()
        n_after = len(genesis2.substrate.nodes)
        
        split_marker = "→ %d" % n_after if n_after > n_before else ""
        
        # Distance from φ ratio
        if len(values) > 1:
            ratio = max(values) / min(values) if min(values) > 0 else float('inf')
            phi_dist = abs(ratio - PHI)
        else:
            phi_dist = float('inf')
        
        print("%4d  %5d  %8.5f  %8s  %10.4f" % 
              (step, n_before, mean_val, split_marker, phi_dist if phi_dist < 100 else float('inf')))
    
    if save_data:
        import json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/visualization_data_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(time_series, f, indent=2)
        print()
        print("Data saved to: %s" % filename)
    
    return time_series


def print_ascii_chart(data, label, width=60):
    """Simple ASCII line chart."""
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val > min_val else 1
    
    print()
    print("%s (min=%.1f, max=%.1f)" % (label, min_val, max_val))
    print("-" * width)
    
    # Sample to fit width
    step = max(1, len(data) // width)
    sampled = data[::step][:width]
    
    for val in sampled:
        normalized = int((val - min_val) / range_val * 40)
        print("█" * normalized)


if __name__ == "__main__":
    data = run_experiment(n_steps=500)
    print_ascii_chart(data['node_count'], "Node count over time")
