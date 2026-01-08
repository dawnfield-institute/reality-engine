"""
Simple Field State Cache for Reality Engine
============================================

Lightweight LRU cache for field states, adapted from Kronos memory system.

Features:
- In-memory LRU cache for hot field states
- Automatic eviction when capacity reached
- Cache hit/miss statistics
- Memory usage tracking

Phase 1 implementation - simplified for immediate integration.
Phase 2 will add PAC tree compression and delta encoding.

Based on: fracton/storage/kronos_memory.py
"""

import torch
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class CacheStats:
    """Statistics for cache performance"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    stores: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_accesses(self) -> int:
        """Total cache accesses"""
        return self.hits + self.misses


class FieldStateCache:
    """
    Simple LRU cache for Reality Engine field states.

    Stores field states (P, A, M, T tensors) with automatic LRU eviction.
    Tracks cache statistics for performance monitoring.

    Example:
        >>> cache = FieldStateCache(max_size=1000)
        >>> cache.store(step=100, state=field_state)
        >>> cached_state = cache.retrieve(step=100)
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats.hit_rate:.2%}")
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize field state cache.

        Args:
            max_size: Maximum number of states to cache
        """
        self.max_size = max_size

        # LRU cache: OrderedDict maintains insertion order
        # Key: (step, state_id), Value: FieldState
        self._cache: OrderedDict[Tuple[int, str], 'FieldState'] = OrderedDict()

        # Statistics
        self.stats = CacheStats()

    def store(self, step: int, state: 'FieldState', state_id: str = "default") -> None:
        """
        Store a field state in cache.

        Args:
            step: Simulation step number
            state: FieldState to cache
            state_id: Identifier for this state (default: "default")
        """
        key = (step, state_id)

        # If already in cache, move to end (most recently used)
        if key in self._cache:
            self._cache.move_to_end(key)
            return

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest (first item)
            self.stats.evictions += 1

        # Store new state
        self._cache[key] = state
        self.stats.stores += 1

    def retrieve(self, step: int, state_id: str = "default") -> Optional['FieldState']:
        """
        Retrieve a field state from cache.

        Args:
            step: Simulation step number
            state_id: Identifier for state

        Returns:
            FieldState if found, None otherwise
        """
        key = (step, state_id)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return self._cache[key]
        else:
            self.stats.misses += 1
            return None

    def contains(self, step: int, state_id: str = "default") -> bool:
        """
        Check if state is in cache without affecting LRU order.

        Args:
            step: Simulation step number
            state_id: Identifier for state

        Returns:
            True if state is cached
        """
        return (step, state_id) in self._cache

    def clear(self) -> None:
        """Clear all cached states."""
        self._cache.clear()
        self.stats = CacheStats()

    def get_stats(self) -> Dict:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'utilization': len(self._cache) / self.max_size if self.max_size > 0 else 0.0,
            'hit_rate': self.stats.hit_rate,
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'evictions': self.stats.evictions,
            'stores': self.stats.stores,
            'total_accesses': self.stats.total_accesses,
        }

    def get_memory_usage(self) -> Dict:
        """
        Estimate memory usage of cached states.

        Returns:
            Dictionary with memory statistics (bytes)
        """
        if not self._cache:
            return {
                'total_bytes': 0,
                'bytes_per_state': 0,
                'num_states': 0
            }

        # Get a sample state to estimate size
        sample_state = next(iter(self._cache.values()))

        # Estimate bytes per state (rough)
        bytes_per_field = 0
        for field_name in ['potential', 'actual', 'memory', 'temperature']:
            if hasattr(sample_state, field_name):
                field_tensor = getattr(sample_state, field_name)
                if isinstance(field_tensor, torch.Tensor):
                    bytes_per_field += field_tensor.element_size() * field_tensor.nelement()

        total_bytes = bytes_per_field * len(self._cache)

        return {
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'bytes_per_state': bytes_per_field,
            'num_states': len(self._cache),
        }

    def __len__(self) -> int:
        """Return number of cached states."""
        return len(self._cache)

    def __repr__(self) -> str:
        """String representation of cache."""
        stats = self.get_stats()
        return (
            f"FieldStateCache(size={stats['size']}/{stats['max_size']}, "
            f"hit_rate={stats['hit_rate']:.2%}, "
            f"evictions={stats['evictions']})"
        )


# Quick test
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from substrate.field_types import FieldState

    print("=" * 60)
    print("FIELD STATE CACHE TEST")
    print("=" * 60)

    # Create cache
    cache = FieldStateCache(max_size=10)

    # Create some test states
    print("\nStoring 15 states (capacity=10)...")
    for i in range(15):
        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8),
            temperature=torch.rand(8, 8)
        )
        cache.store(step=i, state=state)

    print(f"Cache: {cache}")

    # Test retrieval
    print("\nRetrieving states...")
    print(f"  Step 0 (evicted): {cache.retrieve(0) is not None}")
    print(f"  Step 5 (cached): {cache.retrieve(5) is not None}")
    print(f"  Step 14 (cached): {cache.retrieve(14) is not None}")

    # Print statistics
    stats = cache.get_stats()
    print("\nCache Statistics:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Evictions: {stats['evictions']}")
    print(f"  Utilization: {stats['utilization']:.1%}")

    # Memory usage
    mem = cache.get_memory_usage()
    print("\nMemory Usage:")
    print(f"  Total: {mem['total_mb']:.2f} MB")
    print(f"  Per state: {mem['bytes_per_state'] / 1024:.2f} KB")
    print(f"  States cached: {mem['num_states']}")

    print("\n" + "=" * 60)
    print("CACHE TEST COMPLETE")
    print("=" * 60)
