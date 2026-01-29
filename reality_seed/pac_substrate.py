"""
PAC Substrate - The Only Ground Truth

Based on poc_011 PAC-Lazy architecture.

Rules (the ONLY rules):
1. Nodes hold deltas, not absolute values
2. Value is conserved on split: parent.value = sum(children.values)
3. Nodes only interact through explicit links
4. Observation = composition of deltas (temporary, not stored)

Everything else EMERGES. We don't define what can happen.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import deque
import uuid

# Dawn Field Constants (these are derived, not fitted)
PHI = 1.618033988749895
PHI_INV = 0.618033988749895
XI = 1.0571428571428572


@dataclass
class PACNode:
    """
    The atomic unit. A node IS its relationships.
    
    Connections are BY REFERENCE - the connection IS the channel
    through which value flows. No lookups, no indirection.
    
    Properties:
    - id: unique identifier
    - delta: local change (not absolute value)
    - value: conserved quantity for splits
    - links: direct references to connected nodes
    """
    id: str
    delta: torch.Tensor = None      # Local residual
    value: float = 0.0              # Conserved quantity
    
    # Relationships BY REFERENCE (the node IS these)
    # Using sets of node references for O(1) lookup
    parents: Set['PACNode'] = field(default_factory=set)
    children: Set['PACNode'] = field(default_factory=set)
    neighbors: Set['PACNode'] = field(default_factory=set)  # Non-hierarchical links
    
    # Event timing (not clock time - event index)
    born_at: int = 0
    last_active: int = 0
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, PACNode) and self.id == other.id
    
    def all_connections(self) -> Set['PACNode']:
        """All nodes this one is connected to."""
        return self.parents | self.children | self.neighbors
    
    def flow_to_neighbors(self, rate: float = 0.02) -> float:
        """
        Value flows through connections naturally.
        Returns amount flowed.
        """
        if not self.neighbors or self.value < 0.001:
            return 0.0
        
        # Flow from high to low through each connection
        total_flowed = 0.0
        for neighbor in self.neighbors:
            if neighbor.value < self.value:
                flow = self.value * rate / len(self.neighbors)
                self.value -= flow
                neighbor.value += flow
                total_flowed += flow
        return total_flowed
    
    def absorb(self, other: 'PACNode') -> bool:
        """
        Absorb another node's value and connections.
        Returns True if successful.
        """
        if other is self:
            return False
        
        # Take value
        self.value += other.value
        other.value = 0
        
        # Take connections (except self-references)
        for parent in other.parents:
            if parent is not self:
                self.parents.add(parent)
                parent.children.discard(other)
                parent.children.add(self)
        
        for child in other.children:
            if child is not self:
                self.children.add(child)
                child.parents.discard(other)
                child.parents.add(self)
        
        for neighbor in other.neighbors:
            if neighbor is not self:
                self.neighbors.add(neighbor)
                neighbor.neighbors.discard(other)
                neighbor.neighbors.add(self)
        
        # Clear other's connections
        other.parents.clear()
        other.children.clear()
        other.neighbors.clear()
        
        return True


class PACSubstrate:
    """
    The computational substrate. Just nodes and conservation.
    
    No physics. No fields. No forces.
    Just: things connect to things, values conserve.
    """
    
    def __init__(self, device: str = 'auto'):
        self.device = torch.device(
            'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu'
        )
        
        self.nodes: Dict[str, PACNode] = {}
        self.event_count: int = 0
        
        # Conservation tracking
        self.total_value: float = 0.0
        
        # Event history (for observation, not for dynamics)
        self.events: List[Dict] = []
        
    # ========== Node Operations ==========
    
    def create_node(self, value: float = 0.0, 
                   parent: Optional[PACNode] = None) -> PACNode:
        """
        Create a new node.
        
        If parent specified, this is a PAC split:
        - Parent's value decreases
        - Child's value = what parent gave
        - Conservation maintained
        """
        node_id = str(uuid.uuid4())[:8]
        
        node = PACNode(
            id=node_id,
            value=value,
            born_at=self.event_count,
            last_active=self.event_count,
        )
        
        if parent is not None:
            # PAC CONSERVATION: parent gives value to child
            if value > parent.value:
                value = parent.value  # Can't give more than you have
            
            parent.value -= value
            node.value = value
            
            # Link by reference
            parent.children.add(node)
            node.parents.add(parent)
            
            parent.last_active = self.event_count
        else:
            # Genesis node - adds to total
            self.total_value += value
        
        self.nodes[node_id] = node
        
        # Record event
        self.events.append({
            'type': 'create',
            'node': node_id,
            'parent': parent.id if parent else None,
            'value': value,
            'event': self.event_count,
        })
        self.event_count += 1
        
        return node
    
    def split_node(self, node_id: str, ratio: float = 0.5) -> Tuple[PACNode, PACNode]:
        """
        Split a node into two children.
        
        PAC CONSERVATION: parent.value = child1.value + child2.value
        
        Ratio determines the split (0.5 = equal, converges to PHI_INV naturally)
        """
        parent = self.nodes.get(node_id)
        if not parent or parent.value <= 0:
            return None, None
        
        # Split the value
        value1 = parent.value * ratio
        value2 = parent.value * (1 - ratio)
        
        # Create children (no parent in create - we link manually for split)
        child1 = self.create_node(value=0)
        child2 = self.create_node(value=0)
        
        # PAC transfer
        parent.value = 0  # Parent gives everything
        child1.value = value1
        child2.value = value2
        
        # Link by reference
        parent.children.add(child1)
        parent.children.add(child2)
        child1.parents.add(parent)
        child2.parents.add(parent)
        
        # Siblings are neighbors (by reference)
        child1.neighbors.add(child2)
        child2.neighbors.add(child1)
        
        # Record
        self.events.append({
            'type': 'split',
            'parent': node_id,
            'children': [child1.id, child2.id],
            'ratio': ratio,
            'values': [value1, value2],
            'event': self.event_count,
        })
        self.event_count += 1
        
        return child1, child2
    
    def link_nodes(self, node_a: PACNode, node_b: PACNode) -> bool:
        """Create a non-hierarchical link between nodes (by reference)."""
        if node_a is node_b:
            return False
        
        node_a.neighbors.add(node_b)
        node_b.neighbors.add(node_a)
        
        self.events.append({
            'type': 'link',
            'nodes': [node_a.id, node_b.id],
            'event': self.event_count,
        })
        self.event_count += 1
        
        return True
    
    def merge_nodes(self, node_a: PACNode, node_b: PACNode) -> Optional[PACNode]:
        """
        Merge two nodes into one using byref absorb.
        
        PAC CONSERVATION: survivor.value = a.value + b.value
        """
        if node_a is None or node_b is None or node_a is node_b:
            return None
        
        # Use node's own absorb method
        node_a.absorb(node_b)
        
        # Remove absorbed node from substrate
        if node_b.id in self.nodes:
            del self.nodes[node_b.id]
        
        self.events.append({
            'type': 'merge',
            'survivor': node_a.id,
            'absorbed': node_b.id,
            'value': node_a.value,
            'event': self.event_count,
        })
        self.event_count += 1
        
        return node_a
    
    def remove_node(self, node: PACNode) -> bool:
        """Remove a node from the substrate."""
        if node.id not in self.nodes:
            return False
        
        # Disconnect from all neighbors
        for neighbor in list(node.neighbors):
            neighbor.neighbors.discard(node)
        for parent in list(node.parents):
            parent.children.discard(node)
        for child in list(node.children):
            child.parents.discard(node)
        
        node.neighbors.clear()
        node.parents.clear()
        node.children.clear()
        
        del self.nodes[node.id]
        return True
    
    # ========== Conservation Check ==========
    
    def check_conservation(self) -> Dict:
        """
        Verify PAC conservation.
        
        Total value should equal initial injection.
        """
        current_total = sum(n.value for n in self.nodes.values())
        
        return {
            'initial_value': self.total_value,
            'current_value': current_total,
            'conserved': np.isclose(current_total, self.total_value, rtol=1e-10),
            'error': abs(current_total - self.total_value),
        }
    
    # ========== Graph Queries (for observation) ==========
    
    def get_descendants(self, node: PACNode) -> Set[PACNode]:
        """Get all descendants of a node (by reference)."""
        descendants = set()
        queue = deque([node])
        
        while queue:
            n = queue.popleft()
            for child in n.children:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)
        
        return descendants
    
    def get_ancestors(self, node: PACNode) -> Set[PACNode]:
        """Get all ancestors of a node (by reference)."""
        ancestors = set()
        queue = deque([node])
        
        while queue:
            n = queue.popleft()
            for parent in n.parents:
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
        
        return ancestors
    
    def get_connected_component(self, node: PACNode) -> Set[PACNode]:
        """Get all nodes reachable from this one (any link type)."""
        component = set()
        queue = deque([node])
        
        while queue:
            n = queue.popleft()
            if n in component:
                continue
            component.add(n)
            
            for linked in n.all_connections():
                if linked not in component:
                    queue.append(linked)
        
        return component
    
    def get_all_components(self) -> List[Set[PACNode]]:
        """Get all connected components (by reference)."""
        visited = set()
        components = []
        
        for node in self.nodes.values():
            if node not in visited:
                component = self.get_connected_component(node)
                components.append(component)
                visited.update(component)
        
        return components
    
    # ========== Statistics (for observation) ==========
    
    def get_stats(self) -> Dict:
        """Get substrate statistics."""
        components = self.get_all_components()
        
        depths = []
        for nid in self.nodes:
            depths.append(len(self.get_ancestors(nid)))
        
        return {
            'total_nodes': len(self.nodes),
            'total_events': self.event_count,
            'total_value': self.total_value,
            'current_value': sum(n.value for n in self.nodes.values()),
            'n_components': len(components),
            'largest_component': max(len(c) for c in components) if components else 0,
            'mean_depth': np.mean(depths) if depths else 0,
            'max_depth': max(depths) if depths else 0,
        }
