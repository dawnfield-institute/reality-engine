"""Engine — the tick loop that drives the simulation.

Holds a Pipeline, EventBus, and FieldState.
Each tick() pushes state through the pipeline and emits events.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from .event_bus import EventBus
from .state import FieldState
from .config import SimulationConfig


class Engine:
    """Reality Engine v3 — composable operator pipeline with event-driven observability.

    Usage::

        engine = Engine(config=SimulationConfig())
        engine.state = FieldState.big_bang(128, 32)
        engine.pipeline.add(RBFOperator())
        engine.pipeline.add(EulerIntegrator())

        for _ in range(1000):
            engine.tick()
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        pipeline: Optional[object] = None,
        bus: Optional[EventBus] = None,
        state: Optional[FieldState] = None,
    ) -> None:
        from src.v3.operators.protocol import Pipeline

        self.config = config or SimulationConfig()
        self.pipeline = pipeline or Pipeline()
        self.bus = bus or EventBus()
        self._state: Optional[FieldState] = state

    # -- state property ---------------------------------------------------

    @property
    def state(self) -> FieldState:
        if self._state is None:
            raise RuntimeError("Engine has no state. Set engine.state or call initialize().")
        return self._state

    @state.setter
    def state(self, value: FieldState) -> None:
        self._state = value

    @property
    def initialized(self) -> bool:
        return self._state is not None

    # -- lifecycle --------------------------------------------------------

    def initialize(self, mode: str = "big_bang", **kwargs) -> None:
        """Create initial field state.

        Args:
            mode: 'big_bang' or 'zeros'
        """
        device = self.config.get_device()
        if mode == "big_bang":
            self._state = FieldState.big_bang(
                self.config.nu, self.config.nv, device=device, **kwargs
            )
        elif mode == "zeros":
            self._state = FieldState.zeros(
                self.config.nu, self.config.nv, device=device
            )
        else:
            raise ValueError(f"Unknown init mode: {mode}")
        self.bus.emit("initialized", {"mode": mode, "shape": self._state.shape})

    def tick(self) -> FieldState:
        """Advance simulation by one tick.

        Returns:
            The new FieldState after all operators have run.
        """
        old = self.state
        new = self.pipeline(old, self.config, self.bus)

        # Advance bookkeeping
        new = new.replace(
            tick=old.tick + 1,
            dt=self.config.dt,
            time=old.time + self.config.dt,
        )
        self._state = new

        self.bus.emit("tick_complete", {
            "tick": new.tick,
            "time": new.time,
            "total_energy": new.total_energy,
            "pac_total": new.pac_total,
        })
        return new

    def run(self, n: int, callback: Optional[Callable[[FieldState], None]] = None) -> FieldState:
        """Run *n* ticks, optionally calling *callback* after each.

        Returns:
            Final FieldState.
        """
        for _ in range(n):
            state = self.tick()
            if callback is not None:
                callback(state)
        return self.state

    def run_until(
        self,
        predicate: Callable[[FieldState], bool],
        max_ticks: int = 100_000,
    ) -> FieldState:
        """Run until *predicate(state)* returns True or *max_ticks* reached."""
        for _ in range(max_ticks):
            state = self.tick()
            if predicate(state):
                return state
        return self.state
