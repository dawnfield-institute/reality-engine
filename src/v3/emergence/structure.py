"""StructureAnalyzer — tracks structure persistence across ticks."""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.v3.analyzers.base import Detection


class StructureAnalyzer:
    """Track which structures persist across ticks.

    Matches detections between ticks by proximity and kind.
    Structures that persist for multiple ticks are considered stable.
    """

    def __init__(self, match_radius: int = 3) -> None:
        self.match_radius = match_radius
        self._tracked: Dict[int, _TrackedStructure] = {}
        self._next_id: int = 0

    def update(self, detections: List[Detection], tick: int) -> List[Detection]:
        """Match new detections to tracked structures and update persistence.

        Returns detections annotated with 'structure_id' and 'persistence' (ticks alive).
        """
        matched_ids = set()

        for d in detections:
            best_id = self._find_match(d)
            if best_id is not None:
                self._tracked[best_id].last_tick = tick
                self._tracked[best_id].ticks_alive += 1
                d.properties["structure_id"] = best_id
                d.properties["persistence"] = self._tracked[best_id].ticks_alive
                matched_ids.add(best_id)
            else:
                sid = self._next_id
                self._next_id += 1
                self._tracked[sid] = _TrackedStructure(
                    kind=d.kind, position=d.position, last_tick=tick, ticks_alive=1
                )
                d.properties["structure_id"] = sid
                d.properties["persistence"] = 1

        # Prune structures not seen for 10+ ticks
        stale = [sid for sid, s in self._tracked.items() if tick - s.last_tick > 10]
        for sid in stale:
            del self._tracked[sid]

        return detections

    def _find_match(self, detection: Detection) -> int | None:
        u, v = detection.position
        best_id = None
        best_dist = float("inf")
        for sid, s in self._tracked.items():
            if s.kind != detection.kind:
                continue
            su, sv = s.position
            dist = abs(u - su) + abs(v - sv)
            if dist < self.match_radius and dist < best_dist:
                best_dist = dist
                best_id = sid
        return best_id

    @property
    def stable_count(self) -> int:
        """Number of structures alive for 5+ ticks."""
        return sum(1 for s in self._tracked.values() if s.ticks_alive >= 5)

    @property
    def tracked_count(self) -> int:
        return len(self._tracked)


class _TrackedStructure:
    __slots__ = ("kind", "position", "last_tick", "ticks_alive")

    def __init__(self, kind: str, position: Tuple[int, int], last_tick: int, ticks_alive: int):
        self.kind = kind
        self.position = position
        self.last_tick = last_tick
        self.ticks_alive = ticks_alive
