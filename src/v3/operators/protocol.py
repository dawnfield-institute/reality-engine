"""Operator protocol and Pipeline — the composable building blocks.

Every physics operation is an Operator: a callable that transforms FieldState.
Pipeline chains operators sequentially.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Operator(Protocol):
    """Protocol for composable physics operators.

    Each operator receives the current state and returns a new state.
    Operators may emit events via the bus for observability.
    """

    @property
    def name(self) -> str:
        """Human-readable operator name."""
        ...

    def __call__(self, state: Any, config: Any, bus: Any = None) -> Any:
        """Apply this operator, returning a new FieldState."""
        ...


class Pipeline:
    """Sequential chain of operators.

    Executes each operator in order, threading the FieldState through.
    """

    def __init__(self, operators: Optional[List] = None) -> None:
        self._operators: List = list(operators or [])

    def add(self, op: Any) -> Pipeline:
        """Append an operator. Returns self for chaining."""
        self._operators.append(op)
        return self

    def remove(self, name: str) -> Pipeline:
        """Remove operator by name. Returns self for chaining."""
        self._operators = [op for op in self._operators if op.name != name]
        return self

    def __call__(self, state: Any, config: Any, bus: Any = None) -> Any:
        """Run all operators sequentially."""
        for op in self._operators:
            state = op(state, config, bus)
        return state

    def __len__(self) -> int:
        return len(self._operators)

    def __iter__(self):
        return iter(self._operators)

    @property
    def operator_names(self) -> List[str]:
        return [op.name for op in self._operators]
