"""
Tests for Phase 1.2: Simple Field State Cache

Validates that the cache:
1. Stores and retrieves field states correctly
2. Implements LRU eviction properly
3. Tracks statistics accurately
4. Reduces memory usage by avoiding redundant storage
5. Handles edge cases (empty cache, capacity=1, etc.)

Spec: .spec/modernization-roadmap.spec.md Phase 1.2
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import torch
from substrate.field_types import FieldState
from memory.simple_cache import FieldStateCache, CacheStats


class TestCacheBasics:
    """Basic cache functionality tests"""

    def test_cache_initialization(self):
        """Test cache initializes with correct defaults"""
        cache = FieldStateCache(max_size=100)

        assert cache.max_size == 100
        assert len(cache) == 0
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
        assert cache.stats.evictions == 0
        assert cache.stats.stores == 0

    def test_store_and_retrieve(self):
        """Test basic store and retrieve operations"""
        cache = FieldStateCache(max_size=10)

        # Create test state
        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        # Store state
        cache.store(step=0, state=state)
        assert len(cache) == 1
        assert cache.stats.stores == 1

        # Retrieve state
        retrieved = cache.retrieve(step=0)
        assert retrieved is not None
        assert torch.allclose(retrieved.potential, state.potential)
        assert cache.stats.hits == 1

        # Miss
        missing = cache.retrieve(step=999)
        assert missing is None
        assert cache.stats.misses == 1

    def test_contains_check(self):
        """Test contains() doesn't affect LRU order or stats"""
        cache = FieldStateCache(max_size=10)

        state = FieldState(
            potential=torch.rand(4, 4),
            actual=torch.rand(4, 4),
            memory=torch.rand(4, 4)
        )

        # Initially not in cache
        assert not cache.contains(step=0)

        # Store and check
        cache.store(step=0, state=state)
        assert cache.contains(step=0)

        # Contains should not affect hit/miss stats
        initial_hits = cache.stats.hits
        initial_misses = cache.stats.misses
        cache.contains(step=0)
        assert cache.stats.hits == initial_hits
        assert cache.stats.misses == initial_misses

    def test_lru_eviction(self):
        """Test LRU eviction when capacity reached"""
        cache = FieldStateCache(max_size=3)

        # Store 5 states (capacity=3)
        for i in range(5):
            state = FieldState(
                potential=torch.ones(4, 4) * i,
                actual=torch.ones(4, 4) * i,
                memory=torch.ones(4, 4) * i
            )
            cache.store(step=i, state=state)

        # Should have evicted 2 oldest states (0, 1)
        assert len(cache) == 3
        assert cache.stats.evictions == 2

        # Oldest states should be gone
        assert cache.retrieve(step=0) is None
        assert cache.retrieve(step=1) is None

        # Recent states should remain
        assert cache.retrieve(step=2) is not None
        assert cache.retrieve(step=3) is not None
        assert cache.retrieve(step=4) is not None

    def test_lru_reordering(self):
        """Test that retrieval updates LRU order"""
        cache = FieldStateCache(max_size=3)

        # Fill cache
        for i in range(3):
            state = FieldState(
                potential=torch.ones(4, 4) * i,
                actual=torch.ones(4, 4) * i,
                memory=torch.ones(4, 4) * i
            )
            cache.store(step=i, state=state)

        # Access step 0 (make it most recent)
        cache.retrieve(step=0)

        # Add new state (should evict step 1, not step 0)
        new_state = FieldState(
            potential=torch.ones(4, 4) * 99,
            actual=torch.ones(4, 4) * 99,
            memory=torch.ones(4, 4) * 99
        )
        cache.store(step=99, state=new_state)

        # Step 0 should still be cached (was accessed recently)
        assert cache.retrieve(step=0) is not None

        # Step 1 should be evicted (was least recently used)
        assert cache.retrieve(step=1) is None

    def test_duplicate_store(self):
        """Test storing same key twice updates LRU without creating duplicate"""
        cache = FieldStateCache(max_size=3)

        state1 = FieldState(
            potential=torch.ones(4, 4),
            actual=torch.ones(4, 4),
            memory=torch.ones(4, 4)
        )

        state2 = FieldState(
            potential=torch.ones(4, 4) * 2,
            actual=torch.ones(4, 4) * 2,
            memory=torch.ones(4, 4) * 2
        )

        # Store twice with same key
        cache.store(step=0, state=state1)
        cache.store(step=0, state=state2)

        # Should only have 1 entry
        assert len(cache) == 1

        # Should have first state (duplicate store doesn't replace)
        retrieved = cache.retrieve(step=0)
        assert torch.allclose(retrieved.potential, state1.potential)


class TestCacheStatistics:
    """Test cache statistics and metrics"""

    def test_hit_rate_calculation(self):
        """Test hit rate is calculated correctly"""
        cache = FieldStateCache(max_size=10)

        state = FieldState(
            potential=torch.rand(4, 4),
            actual=torch.rand(4, 4),
            memory=torch.rand(4, 4)
        )
        cache.store(step=0, state=state)

        # 3 hits, 2 misses = 60% hit rate
        cache.retrieve(step=0)  # hit
        cache.retrieve(step=0)  # hit
        cache.retrieve(step=0)  # hit
        cache.retrieve(step=1)  # miss
        cache.retrieve(step=2)  # miss

        stats = cache.get_stats()
        assert stats['hits'] == 3
        assert stats['misses'] == 2
        assert abs(stats['hit_rate'] - 0.6) < 1e-6

    def test_utilization_tracking(self):
        """Test cache utilization percentage"""
        cache = FieldStateCache(max_size=10)

        stats = cache.get_stats()
        assert stats['utilization'] == 0.0

        # Fill to 50%
        for i in range(5):
            state = FieldState(
                potential=torch.rand(4, 4),
                actual=torch.rand(4, 4),
                memory=torch.rand(4, 4)
            )
            cache.store(step=i, state=state)

        stats = cache.get_stats()
        assert stats['utilization'] == 0.5

        # Fill to 100%
        for i in range(5, 10):
            state = FieldState(
                potential=torch.rand(4, 4),
                actual=torch.rand(4, 4),
                memory=torch.rand(4, 4)
            )
            cache.store(step=i, state=state)

        stats = cache.get_stats()
        assert stats['utilization'] == 1.0

    def test_memory_usage_estimation(self):
        """Test memory usage estimation"""
        cache = FieldStateCache(max_size=10)

        # Empty cache
        mem = cache.get_memory_usage()
        assert mem['total_bytes'] == 0
        assert mem['num_states'] == 0

        # Add states with known size
        for i in range(5):
            state = FieldState(
                potential=torch.rand(8, 8, dtype=torch.float32),
                actual=torch.rand(8, 8, dtype=torch.float32),
                memory=torch.rand(8, 8, dtype=torch.float32),
                temperature=torch.rand(8, 8, dtype=torch.float32)
            )
            cache.store(step=i, state=state)

        mem = cache.get_memory_usage()

        # Each field: 8×8 × 4 bytes (float32) = 256 bytes
        # 4 fields × 256 bytes = 1024 bytes per state
        # 5 states × 1024 bytes = 5120 bytes total
        assert mem['num_states'] == 5
        assert mem['bytes_per_state'] == 1024
        assert mem['total_bytes'] == 5120
        assert abs(mem['total_mb'] - 5120 / (1024 * 1024)) < 1e-6

    def test_clear_cache(self):
        """Test clearing cache resets everything"""
        cache = FieldStateCache(max_size=10)

        # Fill cache
        for i in range(10):
            state = FieldState(
                potential=torch.rand(4, 4),
                actual=torch.rand(4, 4),
                memory=torch.rand(4, 4)
            )
            cache.store(step=i, state=state)

        # Access some entries
        cache.retrieve(step=0)
        cache.retrieve(step=99)

        # Clear
        cache.clear()

        # Everything should be reset
        assert len(cache) == 0
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
        assert cache.stats.evictions == 0
        assert cache.stats.stores == 0


class TestCacheEdgeCases:
    """Test edge cases and error handling"""

    def test_capacity_one(self):
        """Test cache with capacity=1"""
        cache = FieldStateCache(max_size=1)

        state1 = FieldState(
            potential=torch.ones(4, 4),
            actual=torch.ones(4, 4),
            memory=torch.ones(4, 4)
        )

        state2 = FieldState(
            potential=torch.ones(4, 4) * 2,
            actual=torch.ones(4, 4) * 2,
            memory=torch.ones(4, 4) * 2
        )

        cache.store(step=0, state=state1)
        assert len(cache) == 1

        cache.store(step=1, state=state2)
        assert len(cache) == 1
        assert cache.stats.evictions == 1

        # Only most recent should be cached
        assert cache.retrieve(step=0) is None
        assert cache.retrieve(step=1) is not None

    def test_multiple_state_ids(self):
        """Test storing multiple states at same step with different IDs"""
        cache = FieldStateCache(max_size=10)

        state_default = FieldState(
            potential=torch.ones(4, 4),
            actual=torch.ones(4, 4),
            memory=torch.ones(4, 4)
        )

        state_alt = FieldState(
            potential=torch.ones(4, 4) * 2,
            actual=torch.ones(4, 4) * 2,
            memory=torch.ones(4, 4) * 2
        )

        # Store at same step with different IDs
        cache.store(step=0, state=state_default, state_id="default")
        cache.store(step=0, state=state_alt, state_id="alternative")

        assert len(cache) == 2

        # Retrieve both
        retrieved_default = cache.retrieve(step=0, state_id="default")
        retrieved_alt = cache.retrieve(step=0, state_id="alternative")

        assert retrieved_default is not None
        assert retrieved_alt is not None
        assert torch.allclose(retrieved_default.potential, state_default.potential)
        assert torch.allclose(retrieved_alt.potential, state_alt.potential)

    def test_large_field_states(self):
        """Test cache with realistic large field sizes"""
        cache = FieldStateCache(max_size=5)

        # Realistic size: 256×256 grids
        for i in range(10):
            state = FieldState(
                potential=torch.rand(256, 256),
                actual=torch.rand(256, 256),
                memory=torch.rand(256, 256)
            )
            cache.store(step=i, state=state)

        # Should handle large states correctly
        assert len(cache) == 5
        assert cache.stats.evictions == 5

        # Memory usage should be substantial
        mem = cache.get_memory_usage()
        # 256×256 × 4 bytes × 4 fields = 1 MB per state
        # 5 states ≈ 5 MB
        assert mem['total_mb'] > 4.0  # At least 4 MB


class TestCacheIntegration:
    """Integration tests with realistic usage patterns"""

    def test_simulation_usage_pattern(self):
        """Test cache with realistic simulation access pattern"""
        cache = FieldStateCache(max_size=50)

        # Simulate 1000 steps, storing every 5th
        stored_steps = []
        for i in range(1000):
            if i % 5 == 0:
                state = FieldState(
                    potential=torch.rand(16, 16),
                    actual=torch.rand(16, 16),
                    memory=torch.rand(16, 16)
                )
                cache.store(step=i, state=state)
                stored_steps.append(i)

        # Should have cached last 50 snapshots (200 states total, capacity 50)
        assert len(cache) == 50

        # Recent snapshots should be accessible
        for i in range(750, 1000, 5):
            assert cache.retrieve(step=i) is not None

        # Old snapshots should be evicted
        for i in range(0, 500, 5):
            assert cache.retrieve(step=i) is None

    def test_memory_savings_validation(self):
        """
        Test memory savings compared to storing all states

        Target: 10-15× memory reduction (from roadmap)
        """
        # Scenario: 1000-step simulation, store every step
        # Without cache: Keep all 1000 states in memory
        # With cache (max_size=100): Keep only 100 recent states

        cache = FieldStateCache(max_size=100)

        # Simulate storing 1000 states
        for i in range(1000):
            state = FieldState(
                potential=torch.rand(16, 16),
                actual=torch.rand(16, 16),
                memory=torch.rand(16, 16)
            )
            cache.store(step=i, state=state)

        # Cache should contain exactly 100 states
        assert len(cache) == 100

        # Memory reduction: 1000 states → 100 states = 10× reduction
        theoretical_full_memory = 1000  # If we stored all
        actual_cached = len(cache)

        reduction_factor = theoretical_full_memory / actual_cached
        assert reduction_factor >= 10.0, f"Expected 10× reduction, got {reduction_factor:.1f}×"

        print(f"\nMemory reduction achieved: {reduction_factor:.1f}×")
        print(f"  Full storage would need: {theoretical_full_memory} states")
        print(f"  Cache stores: {actual_cached} states")


def test_phase1_2_integration():
    """
    Integration test for Phase 1.2: Simple Field State Cache

    Validates complete cache functionality with realistic usage.
    """
    print("\n" + "=" * 60)
    print("PHASE 1.2: SIMPLE CACHE - INTEGRATION TEST")
    print("=" * 60)

    cache = FieldStateCache(max_size=100)

    print("\nSimulating 500-step evolution with caching...")

    # Simulate evolution with periodic state saving
    for step in range(500):
        state = FieldState(
            potential=torch.rand(32, 32),
            actual=torch.rand(32, 32),
            memory=torch.rand(32, 32),
            step=step,
            time=step * 0.01
        )

        # Store every state
        cache.store(step=step, state=state)

        # Occasionally retrieve past states (simulate analysis)
        if step > 10 and step % 50 == 0:
            # Try to retrieve recent state
            recent = cache.retrieve(step=step - 5)
            if recent is not None:
                # Validate it's the right state
                assert recent.step == step - 5

    # Final statistics
    stats = cache.get_stats()
    mem = cache.get_memory_usage()

    print("\n" + "-" * 60)
    print("FINAL CACHE STATISTICS:")
    print(f"  States stored: {stats['stores']}")
    print(f"  States cached: {stats['size']}/{stats['max_size']}")
    print(f"  Cache utilization: {stats['utilization']:.1%}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Evictions: {stats['evictions']}")

    print(f"\nMEMORY USAGE:")
    print(f"  Total: {mem['total_mb']:.2f} MB")
    print(f"  Per state: {mem['bytes_per_state'] / 1024:.2f} KB")
    print(f"  States in memory: {mem['num_states']}")

    # Calculate memory reduction
    theoretical_full = stats['stores']
    actual_cached = stats['size']
    reduction = theoretical_full / actual_cached if actual_cached > 0 else 0

    print(f"\nMEMORY REDUCTION:")
    print(f"  Without cache: {theoretical_full} states would be in memory")
    print(f"  With cache: {actual_cached} states in memory")
    print(f"  Reduction factor: {reduction:.1f}× memory savings")

    print("\n" + "=" * 60)
    print("PHASE 1.2 INTEGRATION TEST COMPLETE")
    print("=" * 60)

    # Validation
    assert stats['stores'] == 500
    assert stats['size'] == 100  # Should be at capacity
    assert stats['evictions'] == 400  # 500 - 100 = 400 evictions
    assert reduction >= 5.0, f"Expected ≥5× reduction, got {reduction:.1f}×"


if __name__ == '__main__':
    # Run integration test directly
    test_phase1_2_integration()
