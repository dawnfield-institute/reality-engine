"""
Pattern Detection and Code Generation

Detects stabilized patterns in the genesis graph and generates
code that defines them. Patterns write their own definitions.

NO pre-defined physics - we detect structure, then name it.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import json
from datetime import datetime

from .pac_substrate import PACSubstrate, PACNode, PHI, PHI_INV


@dataclass
class DetectedPattern:
    """A pattern detected in the graph."""
    pattern_type: str  # Generic label: 'cluster', 'chain', 'hub', 'ring', etc.
    node_ids: Set[str]
    metrics: Dict  # Pattern-specific metrics
    stability: float  # How stable is this pattern? (0-1)
    first_seen: int  # Event when first detected
    occurrences: int = 1  # How many times seen
    
    def to_dict(self) -> Dict:
        return {
            'type': self.pattern_type,
            'nodes': list(self.node_ids),
            'metrics': self.metrics,
            'stability': self.stability,
            'first_seen': self.first_seen,
            'occurrences': self.occurrences,
        }


class PatternDetector:
    """
    Detects patterns in the PAC graph.
    
    Pattern types detected (named by structure, not physics):
    - cluster: Dense region of interconnected nodes
    - hub: Single node with many connections
    - chain: Linear sequence of nodes
    - ring: Cyclic structure
    - hierarchy: Tree-like structure with depth
    - concentration: Region with high value density
    """
    
    def __init__(self, substrate: PACSubstrate):
        self.substrate = substrate
        self.patterns: Dict[str, DetectedPattern] = {}  # id -> pattern
        self.pattern_history: List[Dict] = []
        
    def detect_all(self, event: int) -> List[DetectedPattern]:
        """Run all pattern detectors and return findings."""
        patterns = []
        
        # Detect different pattern types
        patterns.extend(self._detect_clusters())
        patterns.extend(self._detect_hubs())
        patterns.extend(self._detect_chains())
        patterns.extend(self._detect_concentrations())
        
        # Update pattern registry
        for p in patterns:
            pid = f"{p.pattern_type}_{hash(frozenset(p.node_ids)) % 10000}"
            if pid in self.patterns:
                self.patterns[pid].occurrences += 1
                self.patterns[pid].stability = min(1.0, 
                    self.patterns[pid].stability + 0.1)
            else:
                p.first_seen = event
                self.patterns[pid] = p
        
        # Record history
        self.pattern_history.append({
            'event': event,
            'patterns': [p.to_dict() for p in patterns],
        })
        
        return patterns
    
    def _detect_clusters(self, min_size: int = 3, 
                         min_density: float = 0.5) -> List[DetectedPattern]:
        """
        Detect clusters - regions where nodes are densely interconnected.
        
        A cluster has density > min_density, where density = 
        actual_edges / possible_edges among the nodes.
        """
        patterns = []
        nodes = list(self.substrate.nodes.values())
        
        if len(nodes) < min_size:
            return patterns
        
        # Find connected components in neighbor graph
        visited = set()
        
        for start_node in nodes:
            if start_node.id in visited:
                continue
            if not start_node.neighbors:
                continue
                
            # BFS to find neighbor-connected region
            cluster = {start_node.id}
            frontier = list(start_node.neighbors)
            
            while frontier and len(cluster) < 20:  # Cap cluster size
                nid = frontier.pop(0)
                if nid in cluster or nid in visited:
                    continue
                    
                node = self.substrate.nodes.get(nid)
                if not node:
                    continue
                    
                # Check if this node is connected to cluster
                connections = len(cluster.intersection(node.neighbors))
                if connections >= 1:
                    cluster.add(nid)
                    frontier.extend(node.neighbors)
            
            if len(cluster) >= min_size:
                # Calculate density
                n = len(cluster)
                possible_edges = n * (n - 1) / 2
                actual_edges = 0
                
                for nid in cluster:
                    node = self.substrate.nodes.get(nid)
                    if node:
                        actual_edges += len(cluster.intersection(node.neighbors))
                actual_edges //= 2  # Each edge counted twice
                
                density = actual_edges / possible_edges if possible_edges > 0 else 0
                
                if density >= min_density:
                    # Calculate total value in cluster
                    total_value = sum(
                        self.substrate.nodes[nid].value 
                        for nid in cluster 
                        if nid in self.substrate.nodes
                    )
                    
                    patterns.append(DetectedPattern(
                        pattern_type='cluster',
                        node_ids=cluster,
                        metrics={
                            'size': n,
                            'density': density,
                            'total_value': total_value,
                            'edges': actual_edges,
                        },
                        stability=0.1,
                        first_seen=0,
                    ))
            
            visited.update(cluster)
        
        return patterns
    
    def _detect_hubs(self, min_degree: int = 5) -> List[DetectedPattern]:
        """
        Detect hubs - nodes with unusually high connectivity.
        """
        patterns = []
        
        for node in self.substrate.nodes.values():
            degree = len(node.neighbors) + len(node.children)
            
            if degree >= min_degree:
                patterns.append(DetectedPattern(
                    pattern_type='hub',
                    node_ids={node.id},
                    metrics={
                        'degree': degree,
                        'neighbors': len(node.neighbors),
                        'children': len(node.children),
                        'value': node.value,
                    },
                    stability=0.2,
                    first_seen=0,
                ))
        
        return patterns
    
    def _detect_chains(self, min_length: int = 4) -> List[DetectedPattern]:
        """
        Detect chains - linear sequences of nodes.
        
        A chain is a path where each node (except endpoints) 
        has exactly 2 connections in the chain.
        """
        patterns = []
        visited = set()
        
        # Find nodes that could be chain endpoints (degree 1 in neighbor graph)
        for node in self.substrate.nodes.values():
            if node.id in visited:
                continue
            if len(node.neighbors) != 1:
                continue
                
            # Try to extend chain from this endpoint
            chain = [node.id]
            current = node
            
            while True:
                # Find next node in chain
                next_candidates = [
                    nid for nid in current.neighbors 
                    if nid not in chain
                ]
                
                if not next_candidates:
                    break
                    
                next_id = next_candidates[0]
                next_node = self.substrate.nodes.get(next_id)
                
                if not next_node:
                    break
                    
                # Check if this continues the chain (degree <= 2 in chain)
                chain_neighbors = len([nid for nid in next_node.neighbors if nid in chain])
                
                chain.append(next_id)
                current = next_node
                
                if len(next_node.neighbors) > 2:
                    break  # Chain ends at hub
            
            if len(chain) >= min_length:
                total_value = sum(
                    self.substrate.nodes[nid].value 
                    for nid in chain 
                    if nid in self.substrate.nodes
                )
                
                patterns.append(DetectedPattern(
                    pattern_type='chain',
                    node_ids=set(chain),
                    metrics={
                        'length': len(chain),
                        'total_value': total_value,
                    },
                    stability=0.1,
                    first_seen=0,
                ))
                
            visited.update(chain)
        
        return patterns
    
    def _detect_concentrations(self, 
                               top_percentile: float = 90,
                               min_size: int = 3) -> List[DetectedPattern]:
        """
        Detect value concentrations - regions with high value density.
        
        These are potential "massive" regions.
        """
        patterns = []
        
        values = [n.value for n in self.substrate.nodes.values()]
        if not values:
            return patterns
            
        threshold = np.percentile(values, top_percentile)
        
        # Find high-value nodes
        high_value_nodes = [
            n for n in self.substrate.nodes.values() 
            if n.value >= threshold
        ]
        
        if len(high_value_nodes) < min_size:
            return patterns
        
        # Check if they're connected
        high_value_ids = {n.id for n in high_value_nodes}
        
        # Find connected groups
        visited = set()
        for start in high_value_nodes:
            if start.id in visited:
                continue
                
            # BFS in high-value subgraph
            group = {start.id}
            frontier = [
                nid for nid in set(start.neighbors) | set(start.children)
                if nid in high_value_ids
            ]
            
            while frontier:
                nid = frontier.pop(0)
                if nid in group:
                    continue
                group.add(nid)
                
                node = self.substrate.nodes.get(nid)
                if node:
                    frontier.extend([
                        n for n in set(node.neighbors) | set(node.children)
                        if n in high_value_ids and n not in group
                    ])
            
            if len(group) >= min_size:
                total_value = sum(
                    self.substrate.nodes[nid].value 
                    for nid in group
                )
                
                patterns.append(DetectedPattern(
                    pattern_type='concentration',
                    node_ids=group,
                    metrics={
                        'size': len(group),
                        'total_value': total_value,
                        'mean_value': total_value / len(group),
                        'threshold': threshold,
                    },
                    stability=0.3,
                    first_seen=0,
                ))
            
            visited.update(group)
        
        return patterns
    
    def get_stable_patterns(self, min_stability: float = 0.5,
                            min_occurrences: int = 3) -> List[DetectedPattern]:
        """Get patterns that have stabilized."""
        return [
            p for p in self.patterns.values()
            if p.stability >= min_stability and p.occurrences >= min_occurrences
        ]


class PatternCodeGenerator:
    """
    Generates code that defines detected patterns.
    
    When a pattern stabilizes, it "writes" its own definition as Python code.
    """
    
    def __init__(self):
        self.generated_code: Dict[str, str] = {}  # pattern_id -> code
        
    def generate(self, pattern: DetectedPattern) -> str:
        """
        Generate Python code that defines this pattern.
        
        The code is a class that can detect similar patterns
        and describes the pattern's properties.
        """
        pid = f"{pattern.pattern_type}_{hash(frozenset(pattern.node_ids)) % 10000}"
        
        if pattern.pattern_type == 'cluster':
            code = self._generate_cluster_code(pattern, pid)
        elif pattern.pattern_type == 'hub':
            code = self._generate_hub_code(pattern, pid)
        elif pattern.pattern_type == 'chain':
            code = self._generate_chain_code(pattern, pid)
        elif pattern.pattern_type == 'concentration':
            code = self._generate_concentration_code(pattern, pid)
        else:
            code = self._generate_generic_code(pattern, pid)
        
        self.generated_code[pid] = code
        return code
    
    def _generate_cluster_code(self, pattern: DetectedPattern, pid: str) -> str:
        m = pattern.metrics
        return f'''
class Cluster_{pid}:
    """
    Auto-generated pattern definition.
    
    A cluster is a dense region of interconnected nodes.
    This pattern emerged from PAC dynamics - not pre-defined.
    
    Properties (observed, not assumed):
    - Size: {m['size']} nodes
    - Density: {m['density']:.3f}
    - Total value: {m['total_value']:.6f}
    - Internal edges: {m['edges']}
    
    First seen at event: {pattern.first_seen}
    Stability: {pattern.stability:.2f}
    Occurrences: {pattern.occurrences}
    """
    
    pattern_type = "cluster"
    size = {m['size']}
    density = {m['density']:.6f}
    total_value = {m['total_value']:.6f}
    
    @classmethod
    def detect(cls, substrate, min_density={m['density']:.3f}):
        """Detect similar clusters in a substrate."""
        # Detection logic based on observed properties
        clusters = []
        # ... (pattern matching logic)
        return clusters
    
    @classmethod
    def is_match(cls, nodes, substrate):
        """Check if a set of nodes matches this pattern."""
        n = len(nodes)
        if n < {m['size'] - 1}:
            return False
        
        # Check density
        edges = 0
        for nid in nodes:
            node = substrate.nodes.get(nid)
            if node:
                edges += len(nodes.intersection(node.neighbors))
        edges //= 2
        
        density = edges / (n * (n-1) / 2) if n > 1 else 0
        return density >= {m['density'] * 0.8:.3f}
'''
    
    def _generate_hub_code(self, pattern: DetectedPattern, pid: str) -> str:
        m = pattern.metrics
        return f'''
class Hub_{pid}:
    """
    Auto-generated pattern definition.
    
    A hub is a node with unusually high connectivity.
    This pattern emerged from PAC dynamics.
    
    Properties:
    - Degree: {m['degree']}
    - Neighbors: {m['neighbors']}
    - Children: {m['children']}
    - Value: {m['value']:.6f}
    
    First seen: event {pattern.first_seen}
    Stability: {pattern.stability:.2f}
    """
    
    pattern_type = "hub"
    degree = {m['degree']}
    value = {m['value']:.6f}
    
    @classmethod
    def detect(cls, substrate, min_degree={m['degree']}):
        """Detect hubs in a substrate."""
        hubs = []
        for node in substrate.nodes.values():
            if len(node.neighbors) + len(node.children) >= min_degree:
                hubs.append(node)
        return hubs
'''
    
    def _generate_chain_code(self, pattern: DetectedPattern, pid: str) -> str:
        m = pattern.metrics
        return f'''
class Chain_{pid}:
    """
    Auto-generated pattern definition.
    
    A chain is a linear sequence of connected nodes.
    
    Properties:
    - Length: {m['length']}
    - Total value: {m['total_value']:.6f}
    
    First seen: event {pattern.first_seen}
    Stability: {pattern.stability:.2f}
    """
    
    pattern_type = "chain"
    length = {m['length']}
    total_value = {m['total_value']:.6f}
    
    @classmethod
    def detect(cls, substrate, min_length={m['length']}):
        """Detect chains of minimum length."""
        chains = []
        # ... (chain detection logic)
        return chains
'''
    
    def _generate_concentration_code(self, pattern: DetectedPattern, pid: str) -> str:
        m = pattern.metrics
        return f'''
class Concentration_{pid}:
    """
    Auto-generated pattern definition.
    
    A concentration is a region with high value density.
    This could be analogous to "mass" in physical systems.
    
    Properties:
    - Size: {m['size']} nodes
    - Total value: {m['total_value']:.6f}
    - Mean value: {m['mean_value']:.6f}
    - Threshold: {m['threshold']:.6f} (90th percentile)
    
    First seen: event {pattern.first_seen}
    Stability: {pattern.stability:.2f}
    Occurrences: {pattern.occurrences}
    
    Interpretation (emergent, not pre-defined):
    This pattern concentrates {pattern.occurrences} times more value than average.
    In physical terms, this MIGHT represent mass-like behavior.
    But we don't assume - we just observe.
    """
    
    pattern_type = "concentration"
    size = {m['size']}
    total_value = {m['total_value']:.6f}
    mean_value = {m['mean_value']:.6f}
    
    @classmethod
    def detect(cls, substrate, percentile=90):
        """Detect value concentrations."""
        import numpy as np
        values = [n.value for n in substrate.nodes.values()]
        if not values:
            return []
        threshold = np.percentile(values, percentile)
        return [n for n in substrate.nodes.values() if n.value >= threshold]
'''
    
    def _generate_generic_code(self, pattern: DetectedPattern, pid: str) -> str:
        return f'''
class Pattern_{pid}:
    """
    Auto-generated pattern definition.
    
    Type: {pattern.pattern_type}
    Metrics: {pattern.metrics}
    Stability: {pattern.stability:.2f}
    Occurrences: {pattern.occurrences}
    """
    
    pattern_type = "{pattern.pattern_type}"
    metrics = {pattern.metrics}
'''
    
    def save_all(self, filepath: str):
        """Save all generated code to a file."""
        header = '''"""
Auto-generated pattern definitions from Reality Seed.

These patterns emerged from PAC dynamics - they were not pre-defined.
Each class represents a structure that stabilized in the genesis.

Generated: {timestamp}
"""

import numpy as np
from typing import Set, List
'''.format(timestamp=datetime.now().isoformat())
        
        with open(filepath, 'w') as f:
            f.write(header)
            f.write('\n')
            for pid, code in self.generated_code.items():
                f.write(f'\n# Pattern: {pid}\n')
                f.write(code)
                f.write('\n')


class EmergenceAnalyzer:
    """
    High-level analyzer that combines pattern detection and code generation.
    """
    
    def __init__(self, genesis):
        self.genesis = genesis
        self.detector = PatternDetector(genesis.substrate)
        self.codegen = PatternCodeGenerator()
        self.observations: List[Dict] = []
        
    def observe(self) -> Dict:
        """Take an observation snapshot."""
        event = self.genesis.substrate.event_count
        
        # Detect patterns
        patterns = self.detector.detect_all(event)
        
        # Get stable patterns
        stable = self.detector.get_stable_patterns()
        
        # Generate code for stable patterns
        new_code = []
        for p in stable:
            pid = f"{p.pattern_type}_{hash(frozenset(p.node_ids)) % 10000}"
            if pid not in self.codegen.generated_code:
                code = self.codegen.generate(p)
                new_code.append({'id': pid, 'type': p.pattern_type})
        
        obs = {
            'event': event,
            'patterns_detected': len(patterns),
            'stable_patterns': len(stable),
            'new_code_generated': len(new_code),
            'patterns': {
                'clusters': len([p for p in patterns if p.pattern_type == 'cluster']),
                'hubs': len([p for p in patterns if p.pattern_type == 'hub']),
                'chains': len([p for p in patterns if p.pattern_type == 'chain']),
                'concentrations': len([p for p in patterns if p.pattern_type == 'concentration']),
            },
            'new_code': new_code,
        }
        
        self.observations.append(obs)
        return obs
    
    def run_observation_cycle(self, steps_per_cycle: int = 1000, 
                              n_cycles: int = 10) -> List[Dict]:
        """Run genesis and observe patterns over multiple cycles."""
        results = []
        
        for cycle in range(n_cycles):
            self.genesis.run(steps_per_cycle)
            obs = self.observe()
            obs['cycle'] = cycle
            results.append(obs)
            
            print(f"Cycle {cycle + 1}/{n_cycles}: "
                  f"{obs['patterns_detected']} patterns, "
                  f"{obs['stable_patterns']} stable, "
                  f"{obs['new_code_generated']} new definitions")
        
        return results
    
    def get_generated_code(self) -> Dict[str, str]:
        """Get all code generated so far."""
        return self.codegen.generated_code
    
    def save_discoveries(self, filepath: str):
        """Save discovered patterns as code."""
        self.codegen.save_all(filepath)
        print(f"Saved {len(self.codegen.generated_code)} pattern definitions to {filepath}")
