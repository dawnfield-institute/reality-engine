"""
Genesis - The Generator

Runs PAC dynamics. Nothing more.
Doesn't know about physics, matter, forces, or structure.
Just: inject entropy, split conservatively, let things happen.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from .pac_substrate import PACSubstrate, PACNode, PHI, PHI_INV


class GenesisSeed:
    """
    The generator. Runs PAC events.
    
    Knows NOTHING about what might emerge.
    Just executes the rules:
    1. Create nodes
    2. Split nodes (conserving value)
    3. Link nodes
    4. That's it
    """
    
    def __init__(self, initial_value: float = 1.0, device: str = 'auto'):
        self.substrate = PACSubstrate(device=device)
        
        # Create the singularity - one node with all value
        self.root = self.substrate.create_node(value=initial_value)
        
        # Track split history for analysis (not for dynamics)
        self.split_ratios: List[float] = []
        
        # Evolutionary memory: map from node_id -> ratio that created it
        # Used to learn which ratios produce successful descendants
        self.birth_ratios: Dict[str, float] = {}
        
        # Weighted history of successful ratios
        self.ratio_fitness: List[tuple] = []  # (ratio, fitness_score)
        self.ratio_memory_weight = 0.4  # How much to pull toward successful ratios
        
    def step(self) -> Dict:
        """
        One genesis step.
        
        The simplest possible dynamics:
        1. Pick a node with value
        2. Split it with evolutionary bias
        3. Maybe link the children to nearby nodes
        4. If system stagnates, inject entropy (small random value)
        
        Returns event info.
        """
        # Find nodes with value to split
        candidates = [n for n in self.substrate.nodes.values() if n.value > 0.01]
        
        if not candidates:
            # System stagnated - inject entropy to keep it alive
            # This is the SEC "entropy pump" - structure can emerge from noise
            return self._inject_entropy()
        
        # Pick one - bias toward nodes with more value (they can sustain more splits)
        weights = np.array([n.value for n in candidates])
        weights = weights / weights.sum()
        parent = np.random.choice(candidates, p=weights)
        
        # Split ratio selection
        # Start random, then evolve toward ratios that produced successful descendants
        ratio = np.random.random()
        
        if self.ratio_fitness and np.random.random() < self.ratio_memory_weight:
            # Weighted selection from fitness history
            ratios, fitnesses = zip(*self.ratio_fitness[-500:])  # Recent memory
            total_fitness = sum(fitnesses)
            if total_fitness > 0:
                probs = [f / total_fitness for f in fitnesses]
                memory_ratio = np.random.choice(ratios, p=probs)
                # Blend with random (mutation preserves exploration)
                ratio = 0.8 * memory_ratio + 0.2 * ratio
        
        child1, child2 = self.substrate.split_node(parent.id, ratio=ratio)
        
        if child1 is None:
            return {'type': 'failed_split', 'node': parent.id}
        
        self.split_ratios.append(ratio)
        
        # Record birth ratios for evolutionary learning
        self.birth_ratios[child1.id] = ratio
        self.birth_ratios[child2.id] = 1 - ratio  # Complement
        
        # ========================================================================
        # EMERGENT ENTROPY REINJECTION (Xi mechanism)
        # ========================================================================
        # Every PAC event produces a "wake" of entropy that gets reinjected
        # This is NOT external - it's the byproduct of structure formation
        #
        # From oscillation_attractor_dynamics exp_24:
        #   Within-level (siblings): -0.028 (φ-split reduces coherence)
        #   Cross-level (network): +0.085 (interference amplifies)
        #   Net: Ξ - 1 = π/55 ≈ 0.0571 per event
        #
        # The "wake" emerges from cross-level interference and gets
        # distributed to connected nodes (the micro-interactions,
        # the employee relationships, the local perturbations)
        # ========================================================================
        
        emergence = self._compute_emergence_wake(child1, child2, ratio)
        
        # Credit ancestor lineage: when a node splits, credit all ancestors' ratios
        # This creates selection pressure for ratios that produce deep lineages
        ancestors = self.substrate.get_ancestors(parent)  # Now byref
        depth = 0
        for ancestor in ancestors:
            if ancestor.id in self.birth_ratios:
                ancestor_ratio = self.birth_ratios[ancestor.id]
                # Fitness decays with distance but deep lineage is rewarded
                fitness = 1.0 / (1.0 + depth * 0.3)
                self.ratio_fitness.append((ancestor_ratio, fitness))
                depth += 1
                if depth > 5:  # Only credit recent ancestry
                    break
        
        # Keep fitness bounded
        if len(self.ratio_fitness) > 2000:
            self.ratio_fitness = self.ratio_fitness[-1000:]
        
        # Maybe link children to other nearby nodes (creates non-hierarchical structure)
        linked1 = self._maybe_link_to_relatives(child1, max_distance=3, probability=0.15)
        linked2 = self._maybe_link_to_relatives(child2, max_distance=3, probability=0.15)
        
        # VALUE FLOW: After split, let value flow naturally through connections
        self._flow_value()
        
        return {
            'type': 'split',
            'parent': parent.id,
            'children': [child1.id, child2.id],
            'ratio': ratio,
            'event': self.substrate.event_count,
            'linked': linked1 or linked2,
            'emergence': emergence,
        }
    
    def _compute_emergence_wake(self, child1: PACNode, child2: PACNode, 
                                ratio: float) -> Dict:
        """
        Compute and distribute the emergent entropy wake from a PAC event.
        
        This is the Ξ mechanism:
        - Within-level: 2*sqrt(r*(1-r)) - 1 (negative for coherence loss)
        - Cross-level: +0.085 per connection (network amplifies)
        - Net: Ξ - 1 = π/55 ≈ 0.0571 per event
        
        The wake is distributed to connected nodes as value perturbations.
        
        IMPORTANT: The cross term is TOPOLOGICAL, not value-dependent.
        What matters is the existence of connections, not their magnitudes.
        This is the "employee interaction" effect - relationships form
        regardless of how much "value" each person carries.
        """
        # Within-level: sibling interference (the "product creation cost")
        # This is purely about the split ratio
        within = 2 * np.sqrt(ratio * (1 - ratio)) - 1  # ≈ -0.028 at φ
        
        # Cross-level: +0.085 per cross-level connection
        # This is TOPOLOGICAL - it's about network structure, not values
        # From oscillation_attractor_dynamics: cross = +0.085 per event at depth
        n_cross_connections = len(child1.neighbors) + len(child2.neighbors)
        # Subtract the sibling connection (that's within-level)
        n_cross_connections = max(0, n_cross_connections - 2)
        
        # Each cross connection contributes ~0.085/N where N is average connections
        # For a balanced tree, this gives net ≈ π/55
        cross_per_connection = 0.085
        cross = n_cross_connections * cross_per_connection
        
        # Net emergence per event
        net_emergence = within + cross
        
        # Distribute the wake to local neighborhood
        if net_emergence > 0:
            neighbors_to_energize = child1.neighbors | child2.neighbors
            
            if neighbors_to_energize:
                # Wake is scaled by the parent's value (the "energy" of the event)
                parent_value = child1.value + child2.value
                wake_amount = net_emergence * parent_value * 0.1  # 10% of theoretical
                wake_per_neighbor = wake_amount / len(neighbors_to_energize)
                for neighbor in neighbors_to_energize:
                    neighbor.value += wake_per_neighbor
        
        return {
            'within': within,
            'cross': cross,
            'net': net_emergence,
            'n_connections': n_cross_connections,
            'xi_distance': abs(net_emergence - (np.pi / 55)),
        }
    
    def _are_connected(self, node1: PACNode, node2: PACNode, max_depth: int = 3) -> bool:
        """Check if two nodes are connected within max_depth hops."""
        if node1 is node2:
            return False
        if node2 in node1.neighbors:
            return True
        
        # BFS to check connectivity (byref)
        visited = {node1}
        frontier = set(node1.neighbors)
        
        for _ in range(max_depth - 1):
            if node2 in frontier:
                return True
            next_frontier = set()
            for neighbor in frontier:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.update(neighbor.neighbors)
            frontier = next_frontier - visited
        
        return node2 in frontier
    
    def _flow_value(self, flow_rate: float = 0.02):
        """
        Value flows through connections naturally.
        BYREF: nodes iterate directly over their neighbors.
        Conservation maintained.
        """
        nodes = list(self.substrate.nodes.values())
        if len(nodes) < 2:
            return
            
        for _ in range(min(10, len(nodes) // 10)):
            node = np.random.choice(nodes)
            # Use node's built-in flow method (byref)
            node.flow_to_neighbors(flow_rate)
    
    def _maybe_link_to_relatives(self, node: PACNode, 
                                  max_distance: int = 3,
                                  probability: float = 0.1) -> bool:
        """
        With some probability, link to a relative.
        
        This creates non-tree structure - necessary for complex dynamics.
        Returns True if a link was made.
        """
        if np.random.random() > probability:
            return False
        
        # Find relatives (nodes sharing ancestors within distance) - BYREF
        ancestors = self.substrate.get_ancestors(node)
        
        relatives = set()
        for ancestor in list(ancestors)[:max_distance]:
            for child in ancestor.children:
                if child is not node:
                    relatives.add(child)
        
        if relatives:
            relative = np.random.choice(list(relatives))
            if relative not in node.neighbors:
                self.substrate.link_nodes(node, relative)
                return True
        
        return False
    
    def _consolidate_value(self) -> Dict:
        """
        Consolidate value when system stagnates.
        
        Instead of artificial injection, we allow low-value nodes to
        merge/donate their value to neighbors. This is the inverse of
        splitting - structure collapses back to concentrate value.
        
        This models the natural cycle:
        - Split: potential → actual (structure forms)
        - Consolidate: diffuse structure → concentrated potential
        
        Conservation is maintained - we're just redistributing.
        """
        self.substrate.event_count += 1
        
        # First: clean up zero-value nodes (already consolidated)
        nodes_to_remove = [n for n in self.substrate.nodes.values() if n.value <= 0]
        for node in nodes_to_remove:
            self.substrate.remove_node(node)
        
        nodes = list(self.substrate.nodes.values())
        
        # Find nodes with very low value (dust)
        dust_nodes = [n for n in nodes if 0.001 < n.value < 0.01]
        
        if not dust_nodes:
            # No dust to consolidate - try self-observation
            return self._merge_small_neighbors()
        
        # Try to consolidate dust to neighbors
        consolidated = []
        orphan_dust = []
        for dust in dust_nodes[:10]:  # Process up to 10 per step
            if dust.neighbors:
                # BYREF: pick a neighbor directly
                neighbor = np.random.choice(list(dust.neighbors))
                neighbor.value += dust.value
                dust.value = 0
                consolidated.append(dust.id)
                self.substrate.remove_node(dust)
            else:
                # This dust has no neighbors - mark for merging
                orphan_dust.append(dust)
        
        # If we have orphans but no consolidation, go to self-observation
        if not consolidated and orphan_dust:
            return self._merge_small_neighbors()
        
        return {
            'type': 'consolidation',
            'consolidated': consolidated,
            'event': self.substrate.event_count,
        }
    
    def _merge_small_neighbors(self) -> Dict:
        """
        System observes itself, creating new dependency structures.
        
        From "Observation as Dependency" paper:
        "Observer's Information ← depends on → Observed System State"
        
        Self-observation means:
        1. System creates information about its own structure
        2. This information DEPENDS ON current state
        3. Dependency forces new actualization
        4. This IS the mechanism for emergence from equilibrium
        
        CRITICAL INSIGHT: Observation is NON-LOCAL.
        Unlike value flow (which requires connections), observation
        can reach ANY part of the system. The observer "collapses"
        distant states into local information.
        
        Implementation: Merge small nodes regardless of connectivity.
        The merged node is NEW - it didn't exist before.
        Its existence depends on all absorbed nodes' prior states.
        This creates fresh dependency structure with new potential.
        """
        nodes = list(self.substrate.nodes.values())
        
        # Find nodes that can be merged (below threshold but with value)
        mergeable = [n for n in nodes if 0 < n.value < 0.01]
        
        if len(mergeable) < 2:
            # True equilibrium - nothing can merge
            return {'type': 'equilibrium', 'event': self.substrate.event_count}
        
        # Sort by value to find the largest as survivor
        mergeable.sort(key=lambda n: n.value, reverse=True)
        survivor = mergeable[0]
        
        # Absorb nodes until we have splittable value
        # Observation is NON-LOCAL - no connectivity required
        absorbed_ids = []
        for node in mergeable[1:]:
            if node.id not in self.substrate.nodes:
                continue  # Already absorbed
            
            # Absorb regardless of connectivity (observation reaches everywhere)
            survivor.absorb(node)
            absorbed_ids.append(node.id)
            if node.id in self.substrate.nodes:
                del self.substrate.nodes[node.id]
            
            # Stop when we have enough to split
            if survivor.value >= 0.01:
                return {
                    'type': 'self_observation',
                    'survivor': survivor.id,
                    'absorbed': absorbed_ids,
                    'new_value': survivor.value,
                    'event': self.substrate.event_count,
                    'insight': 'Non-local observation collapses distributed state',
                }
        
        # Partial progress (shouldn't reach here if we have enough nodes)
        if absorbed_ids:
            return {
                'type': 'partial_observation',
                'survivor': survivor.id if survivor else None,
                'absorbed': absorbed_ids,
                'value': survivor.value if survivor else 0,
                'event': self.substrate.event_count,
            }
        
        return {'type': 'equilibrium', 'event': self.substrate.event_count}
    
    def _inject_entropy(self, amount: float = 0.05) -> Dict:
        """
        DEPRECATED: Use _consolidate_value instead.
        
        Kept for fallback when consolidation also fails.
        This represents TRUE external energy input (cosmic background).
        """
        # First try consolidation
        result = self._consolidate_value()
        if result['type'] != 'equilibrium':
            return result
        
        # Only inject if system is truly at equilibrium
        self.substrate.event_count += 1
        
        # Minimal injection - cosmic background level
        nodes = list(self.substrate.nodes.values())
        n_to_energize = max(1, len(nodes) // 50)  # 2% of nodes
        
        energized = []
        for _ in range(n_to_energize):
            node = np.random.choice(nodes)
            injection = amount * 0.1 * np.random.random()  # Much smaller
            node.value += injection
            energized.append(node.id)
        
        return {
            'type': 'entropy_injection',
            'amount': amount,
            'nodes': energized,
            'event': self.substrate.event_count,
        }
    
    def run(self, n_steps: int) -> List[Dict]:
        """Run n steps of genesis."""
        events = []
        for _ in range(n_steps):
            event = self.step()
            events.append(event)
        return events
    
    def get_state(self) -> Dict:
        """Get current genesis state."""
        return {
            'substrate': self.substrate.get_stats(),
            'conservation': self.substrate.check_conservation(),
            'split_ratios': {
                'count': len(self.split_ratios),
                'mean': np.mean(self.split_ratios) if self.split_ratios else 0,
                'std': np.std(self.split_ratios) if self.split_ratios else 0,
            }
        }


class GenesisObserver:
    """
    The observer. Watches without defining what to look for.
    
    Provides generic measurements:
    - Graph structure (components, density, depth)
    - Value distribution
    - Event statistics
    
    Does NOT provide:
    - "Matter detection"
    - "Force measurement"  
    - "Physics analysis"
    
    Those are interpretations the HUMAN makes while watching.
    """
    
    def __init__(self, genesis: GenesisSeed):
        self.genesis = genesis
        self.substrate = genesis.substrate
        
        # History for tracking trends (observation only)
        self.observations: List[Dict] = []
    
    def observe(self) -> Dict:
        """
        Take a snapshot of current state.
        
        Returns raw measurements, not interpretations.
        """
        obs = {
            'event': self.substrate.event_count,
            'n_nodes': len(self.substrate.nodes),
            'total_value': self.substrate.total_value,
            'conservation': self.substrate.check_conservation(),
        }
        
        # Component structure
        components = self.substrate.get_all_components()
        obs['n_components'] = len(components)
        obs['component_sizes'] = sorted([len(c) for c in components], reverse=True)[:10]
        
        # Value distribution
        values = [n.value for n in self.substrate.nodes.values()]
        if values:
            obs['value_stats'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'max': float(np.max(values)),
                'min': float(np.min(values)),
                'gini': self._gini(values),
            }
        
        # Connectivity
        degrees = []
        for n in self.substrate.nodes.values():
            degree = len(n.parents) + len(n.children) + len(n.neighbors)
            degrees.append(degree)
        if degrees:
            obs['connectivity'] = {
                'mean_degree': float(np.mean(degrees)),
                'max_degree': max(degrees),
            }
        
        # Depth structure
        depths = []
        for nid in self.substrate.nodes:
            depths.append(len(self.substrate.get_ancestors(nid)))
        if depths:
            obs['depth'] = {
                'mean': float(np.mean(depths)),
                'max': max(depths),
            }
        
        # Split ratio convergence (are we approaching PHI_INV?)
        ratios = self.genesis.split_ratios
        if len(ratios) > 10:
            recent = ratios[-100:] if len(ratios) > 100 else ratios
            obs['split_convergence'] = {
                'mean_ratio': float(np.mean(recent)),
                'distance_from_phi_inv': float(abs(np.mean(recent) - PHI_INV)),
                'std': float(np.std(recent)),
            }
        
        self.observations.append(obs)
        return obs
    
    def _gini(self, values: List[float]) -> float:
        """Gini coefficient - measures inequality of value distribution."""
        if not values or sum(values) == 0:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n+1) * sorted_values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
    
    def get_component_graph(self, component_id: int = 0) -> Dict:
        """
        Get graph data for a specific component.
        
        Returns node positions (computed via layout) and edges.
        For visualization by external tools.
        """
        components = self.substrate.get_all_components()
        if component_id >= len(components):
            return {'error': 'component not found'}
        
        component = list(components[component_id])
        
        # Collect edges
        edges = []
        for nid in component:
            node = self.substrate.nodes[nid]
            for cid in node.children:
                if cid in component:
                    edges.append({'from': nid, 'to': cid, 'type': 'child'})
            for nbid in node.neighbors:
                if nbid in component and nid < nbid:  # Avoid duplicates
                    edges.append({'from': nid, 'to': nbid, 'type': 'neighbor'})
        
        # Node data
        nodes = []
        for nid in component:
            node = self.substrate.nodes[nid]
            nodes.append({
                'id': nid,
                'value': node.value,
                'born_at': node.born_at,
                'n_children': len(node.children),
                'n_neighbors': len(node.neighbors),
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'size': len(component),
        }
    
    def get_trends(self) -> Dict:
        """
        Get trends over observation history.
        
        How are things changing over time?
        """
        if len(self.observations) < 2:
            return {'error': 'need more observations'}
        
        events = [o['event'] for o in self.observations]
        n_nodes = [o['n_nodes'] for o in self.observations]
        n_components = [o['n_components'] for o in self.observations]
        
        trends = {
            'growth_rate': (n_nodes[-1] - n_nodes[0]) / (events[-1] - events[0] + 1),
            'component_trend': n_components[-1] - n_components[0],
        }
        
        # Value concentration trend
        ginis = [o.get('value_stats', {}).get('gini', 0) for o in self.observations]
        if len(ginis) > 1:
            trends['gini_trend'] = ginis[-1] - ginis[0]  # Positive = more concentrated
        
        # Depth trend
        depths = [o.get('depth', {}).get('mean', 0) for o in self.observations]
        if len(depths) > 1:
            trends['depth_trend'] = depths[-1] - depths[0]
        
        # Split ratio convergence
        phi_distances = [o.get('split_convergence', {}).get('distance_from_phi_inv', 1) 
                        for o in self.observations]
        if len(phi_distances) > 1 and all(d is not None for d in phi_distances):
            trends['phi_convergence'] = phi_distances[0] - phi_distances[-1]  # Positive = converging
        
        return trends
