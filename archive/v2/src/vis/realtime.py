"""
Real-time visualisation of Reality Engine fields using pygame.

Layout:
    ┌──────────┬──────────┬──────────┐
    │  P field  │  A field  │  M field │
    ├──────────┴──────────┴──────────┤
    │  Ξ(t) trace  │  PAC residual   │
    ├──────────────┴─────────────────┤
    │  Status bar                     │
    └─────────────────────────────────┘

Falls back to a headless / matplotlib summary if pygame is unavailable.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Sequence

import torch
import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# ======================================================================
# Colour-map helpers (no matplotlib dependency for the live loop)
# ======================================================================

def _viridis_lut(n: int = 256) -> np.ndarray:
    """Approximate viridis colour map as (n, 3) uint8."""
    t = np.linspace(0.0, 1.0, n)
    r = np.clip(np.interp(t, [0, 0.25, 0.5, 0.75, 1.0], [68, 49, 33, 144, 253]), 0, 255)
    g = np.clip(np.interp(t, [0, 0.25, 0.5, 0.75, 1.0], [1, 104, 165, 206, 231]), 0, 255)
    b = np.clip(np.interp(t, [0, 0.25, 0.5, 0.75, 1.0], [84, 142, 132, 65, 37]), 0, 255)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


VIRIDIS = _viridis_lut()


def _field_to_surface(
    field: torch.Tensor,
    width: int,
    height: int,
) -> "pygame.Surface":
    """Convert a (n_u, n_v) GPU tensor to a scaled pygame Surface."""
    arr = field.detach().cpu().float().numpy()
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-12:
        idx = np.zeros_like(arr, dtype=np.int32)
    else:
        idx = ((arr - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.int32)

    rgb = VIRIDIS[idx]  # (n_u, n_v, 3)
    surf = pygame.surfarray.make_surface(rgb)
    return pygame.transform.scale(surf, (width, height))


# ======================================================================
# Renderer
# ======================================================================

class RealtimeRenderer:
    """Pygame-based real-time field visualiser."""

    PANEL_W = 320
    PANEL_H = 240
    TRACE_H = 140
    STATUS_H = 30
    BG = (20, 20, 30)
    TXT = (200, 210, 220)

    def __init__(self, fps: int = 30) -> None:
        if not HAS_PYGAME:
            raise RuntimeError(
                "pygame is not installed.  Install with: pip install pygame"
            )
        pygame.init()
        self.fps = fps
        total_w = self.PANEL_W * 3
        total_h = self.PANEL_H + self.TRACE_H + self.STATUS_H
        self.screen = pygame.display.set_mode((total_w, total_h))
        pygame.display.set_caption("Reality Engine v2")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 14)
        self._xi_history: list[float] = []
        self._pac_history: list[float] = []

    # ------------------------------------------------------------------
    def alive(self) -> bool:
        """Process events; return False if the user closed the window."""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return False
        return True

    # ------------------------------------------------------------------
    def render(self, state: Any, diag: dict[str, Any]) -> None:
        """Draw one frame from *state* (FieldState) and *diag* dict."""
        self.screen.fill(self.BG)

        # --- Field heatmaps (top row) ---
        panels = [
            (state.P, "P (potential)"),
            (state.A, "A (actualized)"),
            (state.M, "M (memory)"),
        ]
        for i, (field, label) in enumerate(panels):
            x = i * self.PANEL_W
            surf = _field_to_surface(field, self.PANEL_W, self.PANEL_H)
            self.screen.blit(surf, (x, 0))
            lbl = self.font.render(label, True, self.TXT)
            self.screen.blit(lbl, (x + 4, 2))

        # --- Time-series traces (middle row) ---
        self._xi_history.append(diag.get("xi_spectral", 0.0))
        self._pac_history.append(math.log10(max(diag.get("residual", 1e-15), 1e-15)))

        y_base = self.PANEL_H
        half_w = self.PANEL_W * 3 // 2

        # Ξ trace
        self._draw_trace(
            self._xi_history,
            0.9,
            1.2,
            0,
            y_base,
            half_w,
            self.TRACE_H,
            f"Xi = {diag.get('xi_spectral', 0):.4f}",
        )

        # PAC residual trace (log scale)
        self._draw_trace(
            self._pac_history,
            -15,
            0,
            half_w,
            y_base,
            half_w,
            self.TRACE_H,
            f"log10(PAC) = {self._pac_history[-1]:.1f}",
        )

        # --- Status bar ---
        y_status = y_base + self.TRACE_H
        t = diag.get("t", 0)
        xi = diag.get("xi_spectral", 0)
        res = diag.get("residual", 0)
        fps_val = self.clock.get_fps()
        status = (
            f"t={t:6d}  |  Xi={xi:.4f}  |  "
            f"PAC={res:.2e}  |  {fps_val:.0f} fps"
        )
        txt = self.font.render(status, True, self.TXT)
        self.screen.blit(txt, (8, y_status + 6))

        pygame.display.flip()
        self.clock.tick(self.fps)

    # ------------------------------------------------------------------
    def _draw_trace(
        self,
        data: list[float],
        y_min: float,
        y_max: float,
        x0: int,
        y0: int,
        w: int,
        h: int,
        label: str,
    ) -> None:
        """Draw a simple scrolling line chart."""
        # Background
        pygame.draw.rect(self.screen, (30, 30, 40), (x0, y0, w, h))

        n = len(data)
        if n < 2:
            return

        visible = data[-w:]  # one pixel per sample
        rng = y_max - y_min if y_max != y_min else 1.0
        points = []
        for i, val in enumerate(visible):
            px = x0 + i
            py = y0 + h - int((min(max(val, y_min), y_max) - y_min) / rng * h)
            points.append((px, py))

        if len(points) >= 2:
            pygame.draw.lines(self.screen, (0, 200, 120), False, points, 1)

        lbl = self.font.render(label, True, self.TXT)
        self.screen.blit(lbl, (x0 + 4, y0 + 2))

    # ------------------------------------------------------------------
    def close(self) -> None:
        pygame.quit()
