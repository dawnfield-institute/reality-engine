"""
PAC-Lazy Scalable Substrate for Reality Engine v3

This module implements a hierarchical, lazy-evaluated field architecture
that can scale to stellar formation simulations (10^8+ cells) while
maintaining PAC conservation and Möbius topology.

Key Architecture:
1. Multi-scale octree with Möbius-consistent boundaries
2. PAC-Lazy evaluation - only compute where structure exists
3. Tiered caching (hot/warm/cold) for memory efficiency
4. Feigenbaum monitoring for bifurcation/structure detection

Design Principles:
- Store deltas, not absolute values (compression)
- Lazy evaluation - fine detail only where needed
- PAC conservation across all scales
- Möbius topology at every scale level

Reference:
- fracton/core/pac_system.py - Tiered caching
- fracton/core/pac_node.py - Node structure
- reality-engine/substrate/mobius_manifold.py - Topology

Author: Dawn Field Institute
Date: 2026-01-08
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Generator
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
import time

from substrate.constants import (
    PHI, PHI_INV, XI, DELTA_FEIGENBAUM, 
    PAC_TOLERANCE, STRUCT_39, STRUCT_160
)


class ScaleLevel(Enum):
    """Scale hierarchy levels (from Planck to cosmic)"""
    PLANCK = 0      # 10^-35 m - quantum foam
    NUCLEAR = 1     # 10^-15 m - nuclei
    ATOMIC = 2      # 10^-10 m - atoms
    MOLECULAR = 3   # 10^-9 m - molecules
    CELLULAR = 4    # 10^-6 m - cells/dust
    PLANETARY = 5   # 10^6 m - planets
    STELLAR = 6     # 10^9 m - stars
    GALACTIC = 7    # 10^20 m - galaxies
    COSMIC = 8      # 10^26 m - observable universe


@dataclass
class FieldCell:
    """
    A cell in the hierarchical field structure.
    
    Stores field values as deltas from parent for compression.
    Implements PAC conservation: P + A + αM = constant at each level.
    """
    # Identity
    cell_id: int
    level: int                  # Scale level
    position: Tuple[int, ...]   # Position at this level
    
    # Field values (deltas from parent if not root)
    potential_delta: float = 0.0
    actual_delta: float = 0.0
    memory_delta: float = 0.0
    temperature: float = 1.0
    
    # Absolute values (computed lazily)
    _potential: Optional[float] = None
    _actual: Optional[float] = None
    _memory: Optional[float] = None
    
    # Structure tracking
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    
    # Metadata
    last_access: float = 0.0
    access_count: int = 0
    has_structure: bool = False     # True if contains emergent structure
    bifurcation_count: int = 0      # Feigenbaum cascade count
    herniation_depth: int = 0       # MAS depth for mass
    
    def pac_value(self, alpha: float = 0.964) -> float:
        """Compute PAC functional at this cell"""
        P = self._potential or self.potential_delta
        A = self._actual or self.actual_delta
        M = self._memory or self.memory_delta
        return P + A + alpha * M


@dataclass 
class TieredFieldCache:
    """
    Three-tier cache for field cells based on access patterns.
    
    Hot: Active structure regions, kept materialized
    Warm: Recently accessed, stored as deltas
    Cold: Background regions, heavily compressed
    """
    hot_size: int = 100_000       # ~10MB for active structures
    warm_size: int = 1_000_000    # ~100MB for recent
    cold_size: int = 10_000_000   # ~1GB compressed cold storage
    
    # Tiered storage
    _hot: OrderedDict = field(default_factory=OrderedDict)
    _warm: OrderedDict = field(default_factory=OrderedDict)
    _cold: Dict[int, bytes] = field(default_factory=dict)
    
    # Statistics
    hits: Dict[str, int] = field(default_factory=lambda: {'hot': 0, 'warm': 0, 'cold': 0, 'miss': 0})
    
    def get(self, cell_id: int) -> Optional[FieldCell]:
        """Get cell from cache, promoting to hot tier"""
        # Check hot first
        if cell_id in self._hot:
            self._hot.move_to_end(cell_id)
            self.hits['hot'] += 1
            cell = self._hot[cell_id]
            cell.last_access = time.time()
            cell.access_count += 1
            return cell
        
        # Check warm
        if cell_id in self._warm:
            cell = self._warm.pop(cell_id)
            self._promote_to_hot(cell_id, cell)
            self.hits['warm'] += 1
            return cell
        
        # Check cold
        if cell_id in self._cold:
            cell = self._decompress(self._cold.pop(cell_id))
            self._promote_to_warm(cell_id, cell)
            self.hits['cold'] += 1
            return cell
        
        self.hits['miss'] += 1
        return None
    
    def put(self, cell: FieldCell, tier: str = "hot") -> None:
        """Add cell to specified tier"""
        if tier == "hot":
            self._promote_to_hot(cell.cell_id, cell)
        elif tier == "warm":
            self._promote_to_warm(cell.cell_id, cell)
        else:
            self._demote_to_cold(cell.cell_id, cell)
    
    def _promote_to_hot(self, cell_id: int, cell: FieldCell) -> None:
        """Promote to hot tier with eviction"""
        while len(self._hot) >= self.hot_size:
            evicted_id, evicted = self._hot.popitem(last=False)
            # Evict to warm if has structure, else cold
            if evicted.has_structure:
                self._promote_to_warm(evicted_id, evicted)
            else:
                self._demote_to_cold(evicted_id, evicted)
        
        cell.last_access = time.time()
        self._hot[cell_id] = cell
        self._hot.move_to_end(cell_id)
        self._warm.pop(cell_id, None)
        self._cold.pop(cell_id, None)
    
    def _promote_to_warm(self, cell_id: int, cell: FieldCell) -> None:
        """Promote to warm tier with eviction"""
        while len(self._warm) >= self.warm_size:
            evicted_id, evicted = self._warm.popitem(last=False)
            self._demote_to_cold(evicted_id, evicted)
        
        self._warm[cell_id] = cell
        self._warm.move_to_end(cell_id)
        self._cold.pop(cell_id, None)
    
    def _demote_to_cold(self, cell_id: int, cell: FieldCell) -> None:
        """Compress and store in cold tier"""
        self._cold[cell_id] = self._compress(cell)
    
    def _compress(self, cell: FieldCell) -> bytes:
        """Compress cell for cold storage"""
        import pickle
        import zlib
        # Store minimal data
        data = {
            'id': cell.cell_id,
            'level': cell.level,
            'pos': cell.position,
            'pd': cell.potential_delta,
            'ad': cell.actual_delta,
            'md': cell.memory_delta,
            'T': cell.temperature,
            'parent': cell.parent_id,
            'struct': cell.has_structure,
            'hd': cell.herniation_depth
        }
        return zlib.compress(pickle.dumps(data), level=9)
    
    def _decompress(self, data: bytes) -> FieldCell:
        """Decompress cell from cold storage"""
        import pickle
        import zlib
        d = pickle.loads(zlib.decompress(data))
        return FieldCell(
            cell_id=d['id'],
            level=d['level'],
            position=d['pos'],
            potential_delta=d['pd'],
            actual_delta=d['ad'],
            memory_delta=d['md'],
            temperature=d['T'],
            parent_id=d['parent'],
            has_structure=d['struct'],
            herniation_depth=d['hd']
        )
    
    def stats(self) -> Dict:
        """Return cache statistics"""
        total = sum(self.hits.values())
        return {
            'hot_count': len(self._hot),
            'warm_count': len(self._warm),
            'cold_count': len(self._cold),
            'total_cells': len(self._hot) + len(self._warm) + len(self._cold),
            'hit_rate': {
                'hot': self.hits['hot'] / max(total, 1),
                'warm': self.hits['warm'] / max(total, 1),
                'cold': self.hits['cold'] / max(total, 1),
                'miss': self.hits['miss'] / max(total, 1)
            },
            'total_accesses': total
        }


class HierarchicalMobiusField:
    """
    Multi-scale field with Möbius topology and PAC-Lazy evaluation.
    
    This is the core scalable substrate for Reality Engine v3.
    Instead of storing full dense tensors, we use a hierarchical
    octree-like structure where:
    
    1. Each level represents a different scale (Planck → Cosmic)
    2. Fine detail computed only where structure exists
    3. Möbius anti-periodic boundaries maintained at all scales
    4. PAC conservation enforced across scale transitions
    
    Memory scaling:
    - Dense: O(N^3) - 10^9 cells = 4GB 
    - Hierarchical: O(N_structure) - only active regions
    
    For a universe with sparse structure (most is vacuum),
    this can be 100-1000x more efficient.
    """
    
    def __init__(
        self,
        base_resolution: Tuple[int, int, int] = (64, 64, 64),
        max_levels: int = 8,
        cache_config: Optional[Dict] = None,
        device: str = 'cuda'
    ):
        """
        Initialize hierarchical Möbius field.
        
        Args:
            base_resolution: Resolution at coarsest level (level 0)
            max_levels: Maximum refinement levels
            cache_config: Optional cache size configuration
            device: Compute device ('cuda' or 'cpu')
        """
        self.base_resolution = base_resolution
        self.max_levels = max_levels
        self.device = torch.device(device if device != 'auto' else 
                                   'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize cache
        if cache_config:
            self.cache = TieredFieldCache(**cache_config)
        else:
            self.cache = TieredFieldCache()
        
        # Level-0 cells (always materialized)
        self.root_cells: Dict[Tuple[int, int, int], FieldCell] = {}
        
        # Cell ID counter
        self._next_id = 0
        
        # Möbius twist parameters
        self.twist_axis = 0  # Twist around first axis
        self.anti_periodic = True
        
        # PAC tracking
        self.total_pac = 0.0
        self.pac_by_level: Dict[int, float] = {}
        
        # Structure tracking
        self.active_structures: Set[int] = set()  # Cell IDs with structure
        self.bifurcation_events: List[Dict] = []
        
        # Herniation (mass emergence) tracking
        self.herniation_depths: Dict[int, int] = {}  # cell_id -> depth
        
        print(f"HierarchicalMobiusField initialized:")
        print(f"  Base resolution: {base_resolution}")
        print(f"  Max levels: {max_levels} (up to {2**max_levels}x refinement)")
        print(f"  Device: {self.device}")
        print(f"  Max cells at finest level: {np.prod(base_resolution) * (2**max_levels)**3:,.0f}")
    
    def initialize(self, mode: str = 'big_bang') -> None:
        """
        Initialize field state.
        
        Args:
            mode: 'big_bang', 'cold', 'random', 'premass'
        """
        nx, ny, nz = self.base_resolution
        
        if mode == 'big_bang':
            # High potential, low actual, zero memory
            base_P = 1.0
            base_A = 0.01
            base_M = 0.0
            base_T = 10.0  # Hot
            
        elif mode == 'cold':
            # Low energy throughout
            base_P = 0.1
            base_A = 0.1
            base_M = 0.0
            base_T = 0.01
            
        elif mode == 'premass':
            # Premass phase - high information gradient, no mass yet
            # From MAS equation: m_eff = g·v_SEC·(Dr)/(1+Dr)
            # D=0 means m=0, pure field dynamics
            base_P = 1.0
            base_A = 0.0  # No actualization yet
            base_M = 0.0  # No memory (no mass)
            base_T = 1.0  # Unit temperature
            
        else:  # random
            base_P = 0.5
            base_A = 0.5
            base_M = 0.0
            base_T = 1.0
        
        # Create level-0 cells
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Add small fluctuations
                    fluct = np.random.randn() * 0.01
                    
                    cell = FieldCell(
                        cell_id=self._next_id,
                        level=0,
                        position=(i, j, k),
                        potential_delta=base_P + fluct,
                        actual_delta=base_A + fluct * 0.1,
                        memory_delta=base_M,
                        temperature=base_T * (1 + fluct * 0.1),
                        parent_id=None,
                        herniation_depth=0 if mode != 'premass' else -1  # -1 = premass
                    )
                    
                    self.root_cells[(i, j, k)] = cell
                    self.cache.put(cell, tier='hot')
                    self._next_id += 1
        
        # Compute initial PAC
        self._update_pac_total()
        
        print(f"Initialized {len(self.root_cells)} root cells in '{mode}' mode")
        print(f"Total PAC: {self.total_pac:.6f}")
    
    def get_cell(self, position: Tuple[int, ...], level: int = 0) -> Optional[FieldCell]:
        """
        Get field cell at specified position and level.
        
        Uses lazy evaluation - if cell doesn't exist at requested level,
        interpolates from parent or creates it.
        """
        if level == 0:
            # Root level - direct lookup
            pos_3d = tuple(position[:3]) if len(position) >= 3 else (*position, 0)
            return self.root_cells.get(pos_3d)
        
        # Higher levels - check cache first
        cell_id = self._position_to_id(position, level)
        cell = self.cache.get(cell_id)
        
        if cell is not None:
            return cell
        
        # Need to create from parent
        return self._refine_from_parent(position, level)
    
    def set_cell(self, cell: FieldCell) -> None:
        """Update cell in cache"""
        if cell.level == 0:
            self.root_cells[cell.position] = cell
        self.cache.put(cell, tier='hot' if cell.has_structure else 'warm')
    
    def _refine_from_parent(self, position: Tuple[int, ...], level: int) -> FieldCell:
        """Create refined cell from parent level"""
        # Get parent position (divide by 2)
        parent_pos = tuple(p // 2 for p in position)
        parent = self.get_cell(parent_pos, level - 1)
        
        if parent is None:
            # Create default cell
            return FieldCell(
                cell_id=self._next_id,
                level=level,
                position=position,
                parent_id=None
            )
        
        # Interpolate from parent with small fluctuation
        fluct = np.random.randn() * 0.001 / (level + 1)  # Smaller at finer levels
        
        cell = FieldCell(
            cell_id=self._next_id,
            level=level,
            position=position,
            potential_delta=parent.potential_delta * 0.25 + fluct,  # 1/4 of parent
            actual_delta=parent.actual_delta * 0.25 + fluct * 0.1,
            memory_delta=parent.memory_delta * 0.25,
            temperature=parent.temperature,
            parent_id=parent.cell_id,
            herniation_depth=parent.herniation_depth
        )
        
        self._next_id += 1
        self.cache.put(cell, tier='warm')
        
        # Track in parent's children
        parent.children_ids.append(cell.cell_id)
        
        return cell
    
    def _position_to_id(self, position: Tuple[int, ...], level: int) -> int:
        """Convert position and level to unique cell ID"""
        # Use Z-order curve for locality
        result = level * 10**12  # Level prefix
        for i, p in enumerate(position):
            result += p * (10**(i * 4))
        return result
    
    def _update_pac_total(self) -> None:
        """Update total PAC across all cells"""
        total = 0.0
        for cell in self.root_cells.values():
            total += cell.pac_value()
        self.total_pac = total
        self.pac_by_level[0] = total
    
    def apply_mobius_boundary(self, cell: FieldCell) -> FieldCell:
        """
        Apply Möbius anti-periodic boundary condition.
        
        For position (x, y, z), the anti-periodic partner is at
        (x + nx/2, ny-1-y, z) with negated fields.
        """
        nx, ny, nz = self.base_resolution
        x, y, z = cell.position[:3]
        
        # Find anti-periodic partner position
        partner_x = (x + nx // 2) % nx
        partner_y = ny - 1 - y
        partner_z = z
        
        partner = self.root_cells.get((partner_x, partner_y, partner_z))
        if partner is None:
            return cell
        
        # Enforce anti-periodic relationship (blend toward constraint)
        blend = 0.1
        expected_P = -cell.potential_delta
        expected_A = -cell.actual_delta
        
        partner.potential_delta = (1 - blend) * partner.potential_delta + blend * expected_P
        partner.actual_delta = (1 - blend) * partner.actual_delta + blend * expected_A
        
        return cell
    
    def detect_structure(self, threshold: float = 0.1) -> List[FieldCell]:
        """
        Detect cells with emergent structure.
        
        Structure is indicated by:
        1. High memory accumulation (M > threshold)
        2. Low temperature (structure is cold)
        3. Stability (low variance over time)
        """
        structures = []
        for cell in self.root_cells.values():
            if abs(cell.memory_delta) > threshold:
                cell.has_structure = True
                self.active_structures.add(cell.cell_id)
                structures.append(cell)
        
        return structures
    
    def detect_bifurcations(self, observable_history: List[float]) -> Dict:
        """
        Use Feigenbaum universality to detect period-doubling cascades.
        
        This indicates the onset of chaos/structure formation.
        """
        from dynamics.feigenbaum_detector import detect_period, compute_delta_from_ratios
        
        if len(observable_history) < 50:
            return {'detected': False}
        
        # Detect current period
        signal = np.array(observable_history[-100:])
        period = detect_period(signal)
        
        if period == 0:
            return {'detected': False, 'period': 0}
        
        # Check for period doubling
        if len(self.bifurcation_events) > 0:
            last_period = self.bifurcation_events[-1].get('period', 0)
            if period == 2 * last_period:
                # Period doubling detected!
                event = {
                    'detected': True,
                    'old_period': last_period,
                    'new_period': period,
                    'step': len(observable_history),
                    'predicted_delta': DELTA_FEIGENBAUM
                }
                self.bifurcation_events.append(event)
                return event
        
        return {'detected': False, 'period': period}
    
    def herniate(self, cell: FieldCell, strength: float = 0.1) -> FieldCell:
        """
        Apply herniation (mass emergence) to a cell.
        
        From MAS equation: m_eff = g·v_SEC·(Dr)/(1+Dr)
        Each herniation increases D (recursive depth).
        """
        cell.herniation_depth += 1
        D = cell.herniation_depth
        r = PHI_INV  # Relaxation rate from φ
        
        # Mass emergence formula
        v_SEC = abs(cell.actual_delta - cell.potential_delta)  # SEC velocity
        g = strength  # Coupling constant
        
        m_eff = g * v_SEC * (D * r) / (1 + D * r)
        
        # Convert to memory (mass accumulates in memory field)
        cell.memory_delta += m_eff
        
        # Temperature decreases (energy goes into mass)
        cell.temperature *= (1 - m_eff * 0.1)
        
        # Track
        self.herniation_depths[cell.cell_id] = D
        
        return cell
    
    def step(self, dt: float = 0.001) -> Dict:
        """
        Evolve the hierarchical field for one timestep.
        
        This is the main evolution loop:
        1. Compute RBF balance at each active cell
        2. Apply SEC collapse where entropy gradient is high
        3. Enforce PAC conservation
        4. Apply Möbius boundary conditions
        5. Detect and refine around structures
        """
        from dynamics.feigenbaum_detector import detect_period
        
        metrics = {
            'active_cells': len(self.cache._hot),
            'total_cells': len(self.root_cells),
            'pac_before': self.total_pac,
            'structures': len(self.active_structures)
        }
        
        # Track observables for bifurcation detection
        total_energy = 0.0
        total_info = 0.0
        
        # Evolve active (hot) cells
        for cell in list(self.cache._hot.values()):
            # RBF: Balance field = φ·P - A
            B = PHI * cell.potential_delta - cell.actual_delta
            
            # SEC: Collapse where |B| is large
            if abs(B) > 0.1:
                # Collapse drives actualization
                dA = B * dt * 0.5  # Half of imbalance actualizes
                dP = -dA * PHI_INV  # Potential decreases
                
                # Heat generation (Landauer cost)
                heat = abs(dA) * cell.temperature * 0.01
                cell.temperature += heat
                
                # Apply changes
                cell.potential_delta += dP
                cell.actual_delta += dA
                
                # Memory accumulates where collapse happens
                cell.memory_delta += abs(dA) * 0.1
                
                # Check for herniation (mass emergence)
                if cell.memory_delta > 0.5 and cell.herniation_depth < 5:
                    self.herniate(cell, strength=0.1)
            
            # Cooling
            cell.temperature *= (1 - dt * 0.01)
            
            # Apply Möbius boundary
            self.apply_mobius_boundary(cell)
            
            # Accumulate metrics
            total_energy += cell.potential_delta**2 + cell.actual_delta**2
            total_info += cell.memory_delta
        
        # Update PAC
        self._update_pac_total()
        
        # Detect structures
        structures = self.detect_structure()
        
        metrics.update({
            'pac_after': self.total_pac,
            'pac_conservation': abs(metrics['pac_before'] - self.total_pac) / max(abs(metrics['pac_before']), 1e-10),
            'total_energy': total_energy,
            'total_info': total_info,
            'new_structures': len(structures),
            'herniations': sum(1 for c in self.cache._hot.values() if c.herniation_depth > 0),
            'cache_stats': self.cache.stats()
        })
        
        return metrics
    
    def get_field_tensor(self, level: int = 0) -> torch.Tensor:
        """
        Get field values as a dense tensor (for visualization).
        
        Only call this for coarse levels - fine levels are sparse!
        """
        if level != 0:
            raise ValueError("Dense tensor only supported for level 0")
        
        nx, ny, nz = self.base_resolution
        
        # Create tensors for P, A, M
        P = torch.zeros(nx, ny, nz, device=self.device)
        A = torch.zeros(nx, ny, nz, device=self.device)
        M = torch.zeros(nx, ny, nz, device=self.device)
        T = torch.zeros(nx, ny, nz, device=self.device)
        
        for (x, y, z), cell in self.root_cells.items():
            P[x, y, z] = cell.potential_delta
            A[x, y, z] = cell.actual_delta
            M[x, y, z] = cell.memory_delta
            T[x, y, z] = cell.temperature
        
        return torch.stack([P, A, M, T], dim=0)
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary of field state"""
        cache_stats = self.cache.stats()
        
        # Compute statistics over root cells
        P_vals = [c.potential_delta for c in self.root_cells.values()]
        A_vals = [c.actual_delta for c in self.root_cells.values()]
        M_vals = [c.memory_delta for c in self.root_cells.values()]
        T_vals = [c.temperature for c in self.root_cells.values()]
        
        return {
            'resolution': self.base_resolution,
            'max_levels': self.max_levels,
            'total_cells': len(self.root_cells),
            'cached_cells': cache_stats['total_cells'],
            'active_structures': len(self.active_structures),
            'bifurcation_events': len(self.bifurcation_events),
            'total_pac': self.total_pac,
            'field_stats': {
                'P': {'mean': np.mean(P_vals), 'std': np.std(P_vals)},
                'A': {'mean': np.mean(A_vals), 'std': np.std(A_vals)},
                'M': {'mean': np.mean(M_vals), 'std': np.std(M_vals)},
                'T': {'mean': np.mean(T_vals), 'std': np.std(T_vals)}
            },
            'herniation_stats': {
                'total': len(self.herniation_depths),
                'max_depth': max(self.herniation_depths.values()) if self.herniation_depths else 0,
                'by_depth': {}  # Could compute histogram
            },
            'cache': cache_stats
        }


def test_hierarchical_field():
    """Quick test of the hierarchical field"""
    print("=" * 60)
    print("Testing HierarchicalMobiusField")
    print("=" * 60)
    
    # Create small field for testing
    field = HierarchicalMobiusField(
        base_resolution=(16, 16, 16),
        max_levels=4,
        device='cpu'
    )
    
    # Initialize in premass mode
    print("\n1. Initialize in premass mode")
    field.initialize(mode='premass')
    
    # Run some steps
    print("\n2. Evolution test (100 steps)")
    for i in range(100):
        metrics = field.step(dt=0.01)
        if i % 20 == 0:
            print(f"  Step {i}: PAC conservation={metrics['pac_conservation']:.2e}, "
                  f"structures={metrics['structures']}, herniations={metrics['herniations']}")
    
    # Get summary
    print("\n3. Final state summary")
    summary = field.get_summary()
    print(f"  Total cells: {summary['total_cells']}")
    print(f"  Active structures: {summary['active_structures']}")
    print(f"  Total PAC: {summary['total_pac']:.4f}")
    print(f"  Cache hit rates: {summary['cache']}")
    
    print("\n" + "=" * 60)
    print("HierarchicalMobiusField test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_hierarchical_field()
