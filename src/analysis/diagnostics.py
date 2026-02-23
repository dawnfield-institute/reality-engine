"""
DiagnosticsMonitor — stability and conservation monitoring.

Collects per-step diagnostics, detects divergence, reports summary
statistics.  Designed for both live monitoring and post-run analysis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class DiagnosticsMonitor:
    """Accumulate and check per-step diagnostics."""

    def __init__(
        self,
        pac_threshold: float = 1e-8,
        xi_target: float = 1.0571,
        divergence_limit: float = 1e6,
    ) -> None:
        self.pac_threshold = pac_threshold
        self.xi_target = xi_target
        self.divergence_limit = divergence_limit
        self.records: list[dict[str, Any]] = []
        self._diverged = False

    # ------------------------------------------------------------------
    def record(self, diag: dict[str, Any]) -> None:
        """Append a diagnostics dict and run divergence checks."""
        self.records.append(diag)

        # Divergence check — any NaN or extremely large value
        for key in ("P_mean", "A_mean", "M_mean"):
            val = diag.get(key, 0.0)
            if math.isnan(val) or math.isinf(val) or abs(val) > self.divergence_limit:
                self._diverged = True

    @property
    def diverged(self) -> bool:
        return self._diverged

    @property
    def n_steps(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Return summary over all recorded steps."""
        if not self.records:
            return {}

        residuals = [r["residual"] for r in self.records]
        xis = [r.get("xi_spectral", float("nan")) for r in self.records]

        return {
            "n_steps": self.n_steps,
            "diverged": self._diverged,
            "pac_residual_max": max(residuals),
            "pac_residual_mean": sum(residuals) / len(residuals),
            "pac_residual_final": residuals[-1],
            "xi_final": xis[-1] if xis else float("nan"),
            "xi_mean_last_100": (
                sum(xis[-100:]) / len(xis[-100:]) if len(xis) >= 100 else float("nan")
            ),
            "xi_target": self.xi_target,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Dump full history to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.records, fh, indent=2)
