"""
Tests for Phase 1.3: Enhanced State Recording

Validates that state recording:
1. Integrates cache from Phase 1.2
2. Tracks resonance metrics from Phase 1.1
3. Records at configurable intervals
4. Detects convergence automatically
5. Exports history to disk
6. Provides analysis helpers

Spec: .spec/modernization-roadmap.spec.md Phase 1.3
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import torch
import json
from pathlib import Path

from substrate.field_types import FieldState
from conservation.pac_recursion import PACRecursion, PACMetrics
from dynamics.state_recorder import StateRecorder, RecordedState


class TestRecordedState:
    """Test RecordedState dataclass"""

    def test_from_pac_metrics(self):
        """Test creating RecordedState from PACMetrics"""
        metrics = PACMetrics(
            recursion_error=0.001,
            phi_ratio_error=0.0001,
            total_conserved=100.0,
            max_level_deviation=0.002,
            levels_in_tolerance=5,
            total_levels=5,
            resonance_frequency=0.03,
            resonance_confidence=0.85,
            resonance_locked=True
        )

        record = RecordedState.from_pac_metrics(
            step=100,
            time=1.0,
            metrics=metrics,
            has_cached_state=True
        )

        assert record.step == 100
        assert record.time == 1.0
        assert record.pac_error == 0.001
        assert record.phi_error == 0.0001
        assert record.resonance_frequency == 0.03
        assert record.resonance_confidence == 0.85
        assert record.resonance_locked is True
        assert record.has_cached_state is True

    def test_to_dict(self):
        """Test serialization to dictionary"""
        record = RecordedState(
            step=50,
            time=0.5,
            pac_error=0.001,
            phi_error=0.0001,
            conservation_total=100.0,
            resonance_frequency=0.03,
            resonance_confidence=0.8,
            resonance_locked=False
        )

        data = record.to_dict()

        assert isinstance(data, dict)
        assert data['step'] == 50
        assert data['time'] == 0.5
        assert 'resonance_frequency' in data
        assert 'resonance_confidence' in data


class TestStateRecorderBasics:
    """Basic state recorder functionality"""

    def test_initialization(self):
        """Test recorder initializes correctly"""
        recorder = StateRecorder(cache_size=100, record_interval=10)

        assert recorder.cache_size == 100
        assert recorder.record_interval == 10
        assert len(recorder) == 0
        assert recorder.convergence_step is None
        assert recorder.resonance_lock_step is None

    def test_should_record(self):
        """Test record interval logic"""
        recorder = StateRecorder(record_interval=10)

        assert recorder.should_record(0) is True
        assert recorder.should_record(10) is True
        assert recorder.should_record(20) is True
        assert recorder.should_record(5) is False
        assert recorder.should_record(15) is False

    def test_record_state(self):
        """Test recording a single state"""
        recorder = StateRecorder(cache_size=10, record_interval=1)

        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        metrics = PACMetrics(
            recursion_error=0.001,
            phi_ratio_error=0.0001,
            total_conserved=100.0,
            max_level_deviation=0.002,
            levels_in_tolerance=5,
            total_levels=5
        )

        recorder.record(step=0, state=state, metrics=metrics, time=0.0)

        assert len(recorder) == 1
        assert recorder.history[0].step == 0
        assert recorder.history[0].pac_error == 0.001

    def test_record_interval_filtering(self):
        """Test that only interval steps are recorded"""
        recorder = StateRecorder(cache_size=100, record_interval=10)

        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        metrics = PACMetrics(
            recursion_error=0.001,
            phi_ratio_error=0.0001,
            total_conserved=100.0,
            max_level_deviation=0.002,
            levels_in_tolerance=5,
            total_levels=5
        )

        # Record 100 steps, but interval=10 means only 11 should be recorded (0, 10, 20, ..., 100)
        for step in range(101):
            recorder.record(step=step, state=state, metrics=metrics)

        assert len(recorder) == 11

    def test_force_record(self):
        """Test forcing a record outside interval"""
        recorder = StateRecorder(cache_size=100, record_interval=10)

        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        metrics = PACMetrics(
            recursion_error=0.001,
            phi_ratio_error=0.0001,
            total_conserved=100.0,
            max_level_deviation=0.002,
            levels_in_tolerance=5,
            total_levels=5
        )

        # Force record at step 5 (not a multiple of 10)
        recorder.record(step=5, state=state, metrics=metrics, force=True)

        assert len(recorder) == 1
        assert recorder.history[0].step == 5


class TestCacheIntegration:
    """Test integration with FieldStateCache"""

    def test_cache_stores_states(self):
        """Test that states are cached"""
        recorder = StateRecorder(cache_size=10, record_interval=1)

        for i in range(20):
            state = FieldState(
                potential=torch.ones(8, 8) * i,
                actual=torch.ones(8, 8) * i,
                memory=torch.ones(8, 8) * i
            )

            metrics = PACMetrics(
                recursion_error=0.001,
                phi_ratio_error=0.0001,
                total_conserved=100.0,
                max_level_deviation=0.002,
                levels_in_tolerance=5,
                total_levels=5
            )

            recorder.record(step=i, state=state, metrics=metrics)

        # Cache should have last 10 states
        assert len(recorder.cache) == 10

        # Recent states should be retrievable
        state_19 = recorder.retrieve_state(step=19)
        assert state_19 is not None
        assert torch.allclose(state_19.potential, torch.ones(8, 8) * 19)

        # Old states should be evicted
        state_0 = recorder.retrieve_state(step=0)
        assert state_0 is None

    def test_cache_statistics(self):
        """Test that cache statistics are accessible"""
        recorder = StateRecorder(cache_size=100, record_interval=1)

        state = FieldState(
            potential=torch.rand(16, 16),
            actual=torch.rand(16, 16),
            memory=torch.rand(16, 16)
        )

        metrics = PACMetrics(
            recursion_error=0.001,
            phi_ratio_error=0.0001,
            total_conserved=100.0,
            max_level_deviation=0.002,
            levels_in_tolerance=5,
            total_levels=5
        )

        for i in range(50):
            recorder.record(step=i, state=state, metrics=metrics)

        stats = recorder.get_statistics()

        assert 'cache' in stats
        assert 'cache_memory_mb' in stats
        assert stats['cache']['size'] == 50


class TestResonanceTracking:
    """Test resonance metrics tracking from Phase 1.1"""

    def test_resonance_evolution(self):
        """Test resonance evolution tracking"""
        recorder = StateRecorder(cache_size=100, record_interval=10)

        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        # Simulate resonance being detected and locked over time
        for i in range(0, 100, 10):
            freq = 0.03 if i >= 50 else None
            conf = min(i / 100.0, 1.0)
            locked = i >= 80

            metrics = PACMetrics(
                recursion_error=0.001,
                phi_ratio_error=0.0001,
                total_conserved=100.0,
                max_level_deviation=0.002,
                levels_in_tolerance=5,
                total_levels=5,
                resonance_frequency=freq,
                resonance_confidence=conf,
                resonance_locked=locked
            )

            recorder.record(step=i, state=state, metrics=metrics)

        # Get resonance evolution
        resonance = recorder.get_resonance_evolution()

        assert resonance['lock_achieved'] is True
        assert resonance['lock_step'] == 80
        assert len(resonance['locked_steps']) == 2  # Steps 80, 90

    def test_resonance_lock_detection(self):
        """Test automatic resonance lock detection"""
        recorder = StateRecorder(cache_size=100, record_interval=1)

        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        # Record until lock
        for i in range(100):
            metrics = PACMetrics(
                recursion_error=0.001,
                phi_ratio_error=0.0001,
                total_conserved=100.0,
                max_level_deviation=0.002,
                levels_in_tolerance=5,
                total_levels=5,
                resonance_locked=(i >= 50)
            )

            recorder.record(step=i, state=state, metrics=metrics)

        # Should detect first lock
        assert recorder.resonance_lock_step == 50
        assert recorder.is_resonance_locked() is True


class TestConvergenceTracking:
    """Test convergence detection"""

    def test_convergence_detection(self):
        """Test automatic convergence detection"""
        recorder = StateRecorder(cache_size=100, record_interval=1)

        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        # Simulate convergence at step 100 (errors < 1e-6)
        for i in range(200):
            pac_err = 1e-7 if i >= 100 else 0.01
            phi_err = 1e-7 if i >= 100 else 0.01

            metrics = PACMetrics(
                recursion_error=pac_err,
                phi_ratio_error=phi_err,
                total_conserved=100.0,
                max_level_deviation=0.002,
                levels_in_tolerance=5,
                total_levels=5
            )

            recorder.record(step=i, state=state, metrics=metrics, time=i * 0.01)

        # Should detect first convergence
        assert recorder.convergence_step == 100
        assert recorder.convergence_time == 1.0
        assert recorder.is_converged() is True

    def test_convergence_evolution(self):
        """Test convergence evolution tracking"""
        recorder = StateRecorder(cache_size=100, record_interval=10)

        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        # Exponentially decreasing errors
        for i in range(0, 100, 10):
            pac_err = 0.1 * (0.9 ** (i / 10))
            phi_err = 0.1 * (0.9 ** (i / 10))

            metrics = PACMetrics(
                recursion_error=pac_err,
                phi_ratio_error=phi_err,
                total_conserved=100.0,
                max_level_deviation=0.002,
                levels_in_tolerance=5,
                total_levels=5
            )

            recorder.record(step=i, state=state, metrics=metrics)

        convergence = recorder.get_convergence_evolution()

        assert len(convergence['steps']) == 10
        assert len(convergence['pac_errors']) == 10
        # Errors should be decreasing
        assert convergence['pac_errors'][-1] < convergence['pac_errors'][0]


class TestHistoryManagement:
    """Test history retrieval and management"""

    def test_get_history_full(self):
        """Test getting full history"""
        recorder = StateRecorder(cache_size=100, record_interval=1)

        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        metrics = PACMetrics(
            recursion_error=0.001,
            phi_ratio_error=0.0001,
            total_conserved=100.0,
            max_level_deviation=0.002,
            levels_in_tolerance=5,
            total_levels=5
        )

        for i in range(50):
            recorder.record(step=i, state=state, metrics=metrics)

        history = recorder.get_history()

        assert len(history) == 50
        assert history[0].step == 0
        assert history[-1].step == 49

    def test_get_history_range(self):
        """Test getting history within range"""
        recorder = StateRecorder(cache_size=100, record_interval=1)

        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        metrics = PACMetrics(
            recursion_error=0.001,
            phi_ratio_error=0.0001,
            total_conserved=100.0,
            max_level_deviation=0.002,
            levels_in_tolerance=5,
            total_levels=5
        )

        for i in range(100):
            recorder.record(step=i, state=state, metrics=metrics)

        # Get steps 20-30
        history = recorder.get_history(start_step=20, end_step=30)

        assert len(history) == 11  # Inclusive
        assert history[0].step == 20
        assert history[-1].step == 30

    def test_clear_old_records(self):
        """Test clearing old records"""
        recorder = StateRecorder(cache_size=1000, record_interval=1)

        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        metrics = PACMetrics(
            recursion_error=0.001,
            phi_ratio_error=0.0001,
            total_conserved=100.0,
            max_level_deviation=0.002,
            levels_in_tolerance=5,
            total_levels=5
        )

        for i in range(500):
            recorder.record(step=i, state=state, metrics=metrics)

        # Clear, keeping last 100
        removed = recorder.clear_old_records(keep_last_n=100)

        assert removed == 400
        assert len(recorder) == 100
        assert recorder.history[0].step == 400


class TestExport:
    """Test export functionality"""

    def test_export_json(self, tmp_path):
        """Test JSON export"""
        recorder = StateRecorder(cache_size=100, record_interval=10)

        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        metrics = PACMetrics(
            recursion_error=0.001,
            phi_ratio_error=0.0001,
            total_conserved=100.0,
            max_level_deviation=0.002,
            levels_in_tolerance=5,
            total_levels=5,
            resonance_frequency=0.03,
            resonance_confidence=0.8
        )

        for i in range(0, 100, 10):
            recorder.record(step=i, state=state, metrics=metrics)

        # Export
        export_path = str(tmp_path / "test_export.json")
        result_path = recorder.export_to_disk(filepath=export_path, format='json')

        assert Path(result_path).exists()

        # Load and verify
        with open(result_path) as f:
            data = json.load(f)

        assert 'metadata' in data
        assert 'history' in data
        assert 'statistics' in data
        assert data['metadata']['total_records'] == 10

    def test_export_torch(self, tmp_path):
        """Test PyTorch export with cached states"""
        recorder = StateRecorder(cache_size=100, record_interval=1)

        state = FieldState(
            potential=torch.rand(8, 8),
            actual=torch.rand(8, 8),
            memory=torch.rand(8, 8)
        )

        metrics = PACMetrics(
            recursion_error=0.001,
            phi_ratio_error=0.0001,
            total_conserved=100.0,
            max_level_deviation=0.002,
            levels_in_tolerance=5,
            total_levels=5
        )

        for i in range(20):
            recorder.record(step=i, state=state, metrics=metrics)

        # Export
        export_path = str(tmp_path / "test_export.pt")
        result_path = recorder.export_to_disk(filepath=export_path, format='torch')

        assert Path(result_path).exists()

        # Load and verify
        data = torch.load(result_path)

        assert 'metadata' in data
        assert 'history' in data
        assert 'cached_states' in data
        assert len(data['cached_states']) > 0


def test_phase1_3_integration():
    """
    Integration test for Phase 1.3: Enhanced State Recording

    Validates complete workflow with resonance detection and caching.
    """
    print("\n" + "=" * 60)
    print("PHASE 1.3: ENHANCED STATE RECORDING - INTEGRATION TEST")
    print("=" * 60)

    # Create recorder with caching
    recorder = StateRecorder(cache_size=100, record_interval=10)

    # Create PAC enforcer with resonance
    enforcer = PACRecursion(enable_resonance=True, resonance_check_interval=5)

    # Simulate evolution
    print("\nSimulating 500 steps with recording...")
    fields = [torch.rand(16, 16) for _ in range(5)]

    for step in range(500):
        # Evolve
        fields, metrics = enforcer.enforce(fields)

        # Create field state
        state = FieldState(
            potential=fields[0],
            actual=fields[1],
            memory=fields[2],
            step=step,
            time=step * 0.01
        )

        # Record (automatic interval checking)
        recorder.record(step=step, state=state, metrics=metrics)

    # Print statistics
    print(f"\nRecorder: {recorder}")

    stats = recorder.get_statistics()
    print("\n" + "-" * 60)
    print("STATISTICS:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Record interval: {stats['record_interval']}")
    print(f"  Cache size: {stats['cache']['size']}/{stats['cache']['max_size']}")
    print(f"  Cache memory: {stats['cache_memory_mb']:.2f} MB")
    print(f"  Cache hit rate: {stats['cache']['hit_rate']:.2%}")

    print(f"\nCONVERGENCE:")
    print(f"  Converged: {stats['converged']}")
    if stats['convergence_step']:
        print(f"  Convergence step: {stats['convergence_step']}")
    print(f"  Latest PAC error: {stats['latest_pac_error']:.6e}")
    print(f"  Latest Phi error: {stats['latest_phi_error']:.6e}")

    print(f"\nRESONANCE:")
    print(f"  Lock achieved: {stats['resonance_locked']}")
    if stats['resonance_lock_step']:
        print(f"  Lock step: {stats['resonance_lock_step']}")
    print(f"  Latest frequency: {stats['latest_resonance_freq']:.6f}")
    print(f"  Latest confidence: {stats['latest_resonance_conf']:.3f}")

    # Resonance evolution
    resonance = recorder.get_resonance_evolution()
    print(f"\nRESONANCE EVOLUTION:")
    print(f"  Lock achieved: {resonance['lock_achieved']}")
    if resonance['lock_step']:
        print(f"  First lock at step: {resonance['lock_step']}")
    print(f"  Final frequency: {resonance['frequencies'][-1]:.6f}")
    print(f"  Final confidence: {resonance['confidences'][-1]:.3f}")

    # Convergence evolution
    convergence = recorder.get_convergence_evolution()
    print(f"\nCONVERGENCE EVOLUTION:")
    print(f"  Converged: {convergence['converged']}")
    print(f"  Final PAC error: {convergence['pac_errors'][-1]:.6e}")
    print(f"  Final Phi error: {convergence['phi_errors'][-1]:.6e}")

    print("\n" + "=" * 60)
    print("PHASE 1.3 INTEGRATION TEST COMPLETE")
    print("=" * 60)

    # Validation
    assert len(recorder) == 50  # 500 steps / 10 interval = 50 records
    assert stats['cache']['size'] <= 100  # Should not exceed cache size


if __name__ == '__main__':
    # Run integration test directly
    test_phase1_3_integration()
