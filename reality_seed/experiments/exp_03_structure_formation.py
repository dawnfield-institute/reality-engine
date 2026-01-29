"""
Experiment 03: Long-Run Structure Formation

Question: What stable structures emerge at equilibrium?
         Do we see atoms, molecules, or other bound states?

Method: Run for many steps, analyze cluster distribution
"""

import numpy as np
import math
from collections import Counter
from reality_seed.genesis import GenesisSeed


def find_clusters(substrate):
    """Find connected components via BFS."""
    visited = set()
    clusters = []
    
    for node in substrate.nodes.values():
        if node.id in visited:
            continue
        
        cluster = []
        queue = [node]
        while queue:
            n = queue.pop(0)
            if n.id in visited:
                continue
            visited.add(n.id)
            cluster.append(n)
            for neighbor in n.neighbors:
                if neighbor.id not in visited:
                    queue.append(neighbor)
        
        clusters.append(cluster)
    
    return clusters


def analyze_cluster(cluster):
    """Analyze a single cluster's properties."""
    if not cluster:
        return None
    
    total_value = sum(n.value for n in cluster)
    mean_value = np.mean([n.value for n in cluster])
    internal_edges = sum(len(n.neighbors) for n in cluster) // 2
    
    return {
        'size': len(cluster),
        'total_value': total_value,
        'mean_value': mean_value,
        'internal_edges': internal_edges,
        'density': 2 * internal_edges / (len(cluster) * (len(cluster) - 1)) if len(cluster) > 1 else 0
    }


def run_experiment(n_steps=10000, link_probability=0.0):
    """Run long simulation and analyze emergent structures."""
    print("=" * 70)
    print("LONG-RUN STRUCTURE FORMATION")
    print("=" * 70)
    print()
    print("Running %d steps..." % n_steps)
    print()
    
    genesis = GenesisSeed(initial_value=1.0)
    genesis.ratio_memory_weight = 0.5
    
    for epoch in range(n_steps // 100):
        genesis.run(100)
    
    # Analyze final state
    nodes = list(genesis.substrate.nodes.values())
    values = [n.value for n in nodes]
    
    print("Final state:")
    print("  Total nodes: %d" % len(nodes))
    print("  Total value: %.6f (conserved: %s)" % 
          (sum(values), 'YES' if abs(sum(values) - 1.0) < 0.001 else 'NO'))
    print()
    
    # Connectivity
    n_neighbors = [len(n.neighbors) for n in nodes]
    print("Connectivity:")
    print("  Mean neighbors: %.2f" % np.mean(n_neighbors))
    print("  Max neighbors: %d" % max(n_neighbors) if n_neighbors else 0)
    print("  Nodes with neighbors: %d (%.1f%%)" % 
          (sum(1 for n in n_neighbors if n > 0), 
           100 * sum(1 for n in n_neighbors if n > 0) / len(nodes)))
    print()
    
    # Cluster analysis
    clusters = find_clusters(genesis.substrate)
    cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
    
    print("Cluster analysis:")
    print("  Total clusters: %d" % len(clusters))
    print("  Largest cluster: %d nodes" % cluster_sizes[0] if cluster_sizes else 0)
    print("  Top 10 sizes: %s" % cluster_sizes[:10])
    print()
    
    # Size distribution
    size_counts = Counter(cluster_sizes)
    print("Cluster size distribution:")
    for size in sorted(size_counts.keys())[:10]:
        count = size_counts[size]
        bar = '*' * min(50, count // 10)
        print("  size %2d: %4d clusters %s" % (size, count, bar))
    print()
    
    # Analyze largest clusters
    print("Detailed analysis of top 5 clusters:")
    for i, cluster in enumerate(sorted(clusters, key=len, reverse=True)[:5]):
        info = analyze_cluster(cluster)
        print("  Cluster %d: %d nodes, value=%.4f, density=%.2f" % 
              (i+1, info['size'], info['total_value'], info['density']))
    
    return clusters


if __name__ == "__main__":
    run_experiment()
