"""
State Recorder for Reality Engine
==================================

Records field evolution history with integrated caching and resonance tracking.

Features:
- Automatic state recording at configurable intervals
- Integrated LRU cache for memory efficiency
- Resonance metrics tracking from PAC recursion
- Flexible recording modes (all, periodic, on-demand)
- Export to disk for long simulations
- Statistics and analysis helpers

Phase 1.3 implementation - combines cache (1.2) with resonance (1.1).

Integration:
    recorder = StateRecorder(cache_size=1000, record_interval=10)

    for step in range(10000):
        # ... evolution ...
        recorder.record(step=step, state=current_state, metrics=pac_metrics)

    # Analyze
    history = recorder.get_history()
    resonance_evolution = recorder.get_resonance_evolution()

Author: Dawn Field Institute
Date: 2026-01-02
Version: 1.0.0
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

# Import cache (no circular dependency)
from memory.simple_cache import FieldStateCache

# Forward declare types to avoid circular imports at module level
# Actual imports happen in methods that need them


@dataclass
class RecordedState:
    """
    A single recorded state with full metadata

    Combines field state reference (cached) with computed metrics
    and resonance information for comprehensive history tracking.
    """
    step: int
    time: float

    # PAC conservation metrics
    pac_error: float
    phi_error: float
    conservation_total: float

    # Resonance tracking (Phase 1.1)
    resonance_frequency: Optional[float] = None
    resonance_confidence: float = 0.0
    resonance_locked: bool = False

    # Convergence status
    pac_converged: bool = False
    phi_converged: bool = False

    # Field state cached separately (not serialized by default)
    has_cached_state: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_pac_metrics(cls, step: int, time: float, metrics: Any,
                         has_cached_state: bool = False) -> 'RecordedState':
        """
        Create RecordedState from PACMetrics

        Args:
            step: Simulation step
            time: Simulation time
            metrics: PACMetrics from PAC recursion
            has_cached_state: Whether field state is cached

        Returns:
            RecordedState instance
        """
        return cls(
            step=step,
            time=time,
            pac_error=metrics.recursion_error,
            phi_error=metrics.phi_ratio_error,
            conservation_total=metrics.total_conserved,
            resonance_frequency=metrics.resonance_frequency,
            resonance_confidence=metrics.resonance_confidence,
            resonance_locked=metrics.resonance_locked,
            pac_converged=(metrics.recursion_error < 1e-6),
            phi_converged=(metrics.phi_ratio_error < 1e-6),
            has_cached_state=has_cached_state
        )


class StateRecorder:
    """
    Records field evolution with integrated caching and resonance tracking

    Manages state history efficiently using LRU cache for field states
    while keeping lightweight metrics in memory. Provides analysis tools
    for convergence, resonance detection, and export capabilities.

    Example:
        >>> recorder = StateRecorder(cache_size=1000, record_interval=10)
        >>> for step in range(10000):
        ...     state, metrics = pac_recursion.enforce(fields)
        ...     recorder.record(step=step, state=state, metrics=metrics, time=step*dt)
        >>>
        >>> # Analysis
        >>> if recorder.is_converged():
        ...     print(f"Converged at step {recorder.convergence_step}")
        >>>
        >>> resonance = recorder.get_resonance_evolution()
        >>> print(f"Final resonance frequency: {resonance['frequencies'][-1]}")
    """

    def __init__(self,
                 cache_size: int = 1000,
                 record_interval: int = 1,
                 auto_export_threshold: Optional[int] = None):
        """
        Initialize state recorder

        Args:
            cache_size: Maximum number of field states to cache
            record_interval: Record every Nth step (1 = every step)
            auto_export_threshold: Automatically export to disk after N records
        """
        self.cache_size = cache_size
        self.record_interval = record_interval
        self.auto_export_threshold = auto_export_threshold

        # Integrated cache (Phase 1.2)
        self.cache = FieldStateCache(max_size=cache_size)

        # Lightweight history (always in memory)
        self.history: List[RecordedState] = []

        # Track convergence
        self.convergence_step: Optional[int] = None
        self.convergence_time: Optional[float] = None

        # Track resonance lock
        self.resonance_lock_step: Optional[int] = None
        self.resonance_lock_time: Optional[float] = None

        # Export tracking
        self.exports: List[str] = []

    def should_record(self, step: int) -> bool:
        """
        Determine if this step should be recorded

        Args:
            step: Current simulation step

        Returns:
            True if should record
        """
        return (step % self.record_interval == 0)

    def record(self,
               step: int,
               state: Any,  # FieldState
               metrics: Any,  # PACMetrics
               time: Optional[float] = None,
               force: bool = False) -> None:
        """
        Record current state and metrics

        Args:
            step: Simulation step number
            state: Current field state
            metrics: PAC conservation metrics
            time: Simulation time (default: step * 0.01)
            force: Force recording even if not at interval
        """
        # Check if should record
        if not force and not self.should_record(step):
            return

        # Default time
        if time is None:
            time = step * 0.01

        # Cache field state
        self.cache.store(step=step, state=state)

        # Create recorded state
        recorded = RecordedState.from_pac_metrics(
            step=step,
            time=time,
            metrics=metrics,
            has_cached_state=True
        )

        # Add to history
        self.history.append(recorded)

        # Track convergence (first time both converge)
        if (self.convergence_step is None and
            recorded.pac_converged and recorded.phi_converged):
            self.convergence_step = step
            self.convergence_time = time

        # Track resonance lock (first time locked)
        if (self.resonance_lock_step is None and
            recorded.resonance_locked):
            self.resonance_lock_step = step
            self.resonance_lock_time = time

        # Auto-export if threshold reached
        if (self.auto_export_threshold is not None and
            len(self.history) >= self.auto_export_threshold):
            self.export_to_disk(auto_export=True)

    def retrieve_state(self, step: int) -> Optional[Any]:  # Returns FieldState or None
        """
        Retrieve cached field state

        Args:
            step: Step number to retrieve

        Returns:
            FieldState if cached, None otherwise
        """
        return self.cache.retrieve(step=step)

    def get_history(self,
                   start_step: Optional[int] = None,
                   end_step: Optional[int] = None) -> List[RecordedState]:
        """
        Get recorded history within step range

        Args:
            start_step: First step to include (default: all)
            end_step: Last step to include (default: all)

        Returns:
            List of RecordedState objects
        """
        if start_step is None and end_step is None:
            return self.history

        filtered = []
        for record in self.history:
            if start_step is not None and record.step < start_step:
                continue
            if end_step is not None and record.step > end_step:
                break
            filtered.append(record)

        return filtered

    def get_resonance_evolution(self) -> Dict:
        """
        Get evolution of resonance detection over time

        Returns:
            Dictionary with resonance metrics evolution:
                - steps: List of steps
                - frequencies: Detected frequencies
                - confidences: Detection confidences
                - locked_steps: Steps where locked=True
        """
        steps = []
        frequencies = []
        confidences = []
        locked_steps = []

        for record in self.history:
            steps.append(record.step)
            frequencies.append(record.resonance_frequency or 0.0)
            confidences.append(record.resonance_confidence)
            if record.resonance_locked:
                locked_steps.append(record.step)

        return {
            'steps': steps,
            'frequencies': frequencies,
            'confidences': confidences,
            'locked_steps': locked_steps,
            'lock_achieved': len(locked_steps) > 0,
            'lock_step': locked_steps[0] if locked_steps else None
        }

    def get_convergence_evolution(self) -> Dict:
        """
        Get evolution of PAC convergence over time

        Returns:
            Dictionary with convergence metrics:
                - steps: List of steps
                - pac_errors: PAC recursion errors
                - phi_errors: Phi ratio errors
                - converged_step: First step where both converged
        """
        steps = []
        pac_errors = []
        phi_errors = []

        for record in self.history:
            steps.append(record.step)
            pac_errors.append(record.pac_error)
            phi_errors.append(record.phi_error)

        return {
            'steps': steps,
            'pac_errors': pac_errors,
            'phi_errors': phi_errors,
            'converged': self.convergence_step is not None,
            'convergence_step': self.convergence_step,
            'convergence_time': self.convergence_time
        }

    def is_converged(self, tolerance: float = 1e-6) -> bool:
        """
        Check if simulation has converged

        Args:
            tolerance: Error tolerance for convergence

        Returns:
            True if converged
        """
        if not self.history:
            return False

        latest = self.history[-1]
        return (latest.pac_error < tolerance and
                latest.phi_error < tolerance)

    def is_resonance_locked(self) -> bool:
        """
        Check if resonance is currently locked

        Returns:
            True if locked
        """
        if not self.history:
            return False

        return self.history[-1].resonance_locked

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics

        Returns:
            Dictionary with recorder statistics
        """
        cache_stats = self.cache.get_stats()
        cache_memory = self.cache.get_memory_usage()

        stats = {
            'total_records': len(self.history),
            'record_interval': self.record_interval,
            'converged': self.convergence_step is not None,
            'convergence_step': self.convergence_step,
            'resonance_locked': self.resonance_lock_step is not None,
            'resonance_lock_step': self.resonance_lock_step,
            'cache': cache_stats,
            'cache_memory_mb': cache_memory['total_mb'],
            'exports': len(self.exports)
        }

        if self.history:
            latest = self.history[-1]
            stats['latest_step'] = latest.step
            stats['latest_pac_error'] = latest.pac_error
            stats['latest_phi_error'] = latest.phi_error
            stats['latest_resonance_freq'] = latest.resonance_frequency
            stats['latest_resonance_conf'] = latest.resonance_confidence

        return stats

    def export_to_disk(self,
                      filepath: Optional[str] = None,
                      format: str = 'json',
                      auto_export: bool = False) -> str:
        """
        Export history to disk

        Args:
            filepath: Path to export file (default: auto-generated)
            format: Export format ('json' or 'torch')
            auto_export: Whether this is an automatic export

        Returns:
            Path to exported file
        """
        from datetime import datetime

        # Generate filename if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "auto_" if auto_export else ""
            filepath = f"exports/{prefix}history_{timestamp}.{format}"

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            # Export as JSON (lightweight, no field states)
            data = {
                'metadata': {
                    'total_records': len(self.history),
                    'cache_size': self.cache_size,
                    'record_interval': self.record_interval,
                    'converged': self.convergence_step is not None,
                    'convergence_step': self.convergence_step,
                    'resonance_locked': self.resonance_lock_step is not None,
                    'resonance_lock_step': self.resonance_lock_step
                },
                'history': [record.to_dict() for record in self.history],
                'statistics': self.get_statistics()
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == 'torch':
            # Export as PyTorch checkpoint (includes cached states)
            # Get all cached states
            cached_states = {}
            for record in self.history:
                if record.has_cached_state:
                    state = self.cache.retrieve(step=record.step)
                    if state is not None:
                        cached_states[record.step] = {
                            'potential': state.potential,
                            'actual': state.actual,
                            'memory': state.memory,
                            'temperature': state.temperature
                        }

            data = {
                'metadata': {
                    'total_records': len(self.history),
                    'cache_size': self.cache_size,
                    'record_interval': self.record_interval,
                    'converged': self.convergence_step is not None,
                    'convergence_step': self.convergence_step,
                    'resonance_locked': self.resonance_lock_step is not None,
                    'resonance_lock_step': self.resonance_lock_step
                },
                'history': [record.to_dict() for record in self.history],
                'cached_states': cached_states
            }

            torch.save(data, filepath)

        else:
            raise ValueError(f"Unknown format: {format}")

        self.exports.append(filepath)
        return filepath

    def clear_old_records(self, keep_last_n: int = 1000) -> int:
        """
        Clear old records to save memory

        Args:
            keep_last_n: Number of recent records to keep

        Returns:
            Number of records removed
        """
        if len(self.history) <= keep_last_n:
            return 0

        removed = len(self.history) - keep_last_n
        self.history = self.history[-keep_last_n:]

        return removed

    def __len__(self) -> int:
        """Return number of recorded states"""
        return len(self.history)

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"StateRecorder(records={len(self.history)}, "
            f"cache={len(self.cache)}/{self.cache_size}, "
            f"converged={self.convergence_step is not None}, "
            f"resonance_locked={self.resonance_lock_step is not None})"
        )


# Note: Test moved to tests/test_state_recorder.py to avoid import issues
