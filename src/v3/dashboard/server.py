"""Dashboard server — FastAPI + WebSocket for real-time simulation visualization.

Runs the Engine in a background thread, streams field state to connected clients
via WebSocket binary frames (Float64Array for efficient transfer).

Usage::

    python -m src.v3.dashboard.server
    # Opens http://localhost:8050
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.engine.state import FieldState
from src.v3.operators.protocol import Pipeline
from src.v3.operators.rbf import RBFOperator
from src.v3.operators.qbe import QBEOperator
from src.v3.operators.integrator import EulerIntegrator
from src.v3.operators.memory import MemoryOperator
from src.v3.operators.confluence import ConfluenceOperator
from src.v3.operators.temperature import TemperatureOperator
from src.v3.operators.thermal_noise import ThermalNoiseOperator
from src.v3.operators.normalization import NormalizationOperator
from src.v3.operators.adaptive import AdaptiveOperator
from src.v3.operators.time_emergence import TimeEmergenceOperator
from src.v3.analyzers import (
    ConservationAnalyzer, GravityAnalyzer, AtomDetector,
    StarDetector, QuantumDetector, GalaxyAnalyzer,
)
from src.v3.emergence import HerniationDetector, StructureAnalyzer


def build_default_pipeline() -> Pipeline:
    """Full physics pipeline with all operators."""
    return Pipeline([
        RBFOperator(),
        QBEOperator(),
        EulerIntegrator(),
        MemoryOperator(),
        ConfluenceOperator(),
        TemperatureOperator(),
        ThermalNoiseOperator(),
        NormalizationOperator(),
        AdaptiveOperator(),
        TimeEmergenceOperator(),
    ])


class SimulationRunner:
    """Runs the engine in a background thread and accumulates state for clients."""

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig(device=torch.device("cpu"))
        self.engine = Engine(
            config=self.config,
            pipeline=build_default_pipeline(),
        )
        self.engine.initialize("big_bang", temperature=2.0)

        # Analyzers
        self._analyzers = [
            ConservationAnalyzer(),
            GravityAnalyzer(mass_threshold=0.5, min_curvature=0.01),
            AtomDetector(mass_threshold=0.5, gradient_threshold=0.1),
            StarDetector(mass_threshold=2.0, temp_threshold=2.0),
            QuantumDetector(coherence_threshold=0.8),
            GalaxyAnalyzer(mass_threshold=0.2, min_region_fraction=0.02),
            HerniationDetector(threshold=0.5),
        ]
        self._structure_analyzer = StructureAnalyzer()

        # State for clients
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_snapshot: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def step_once(self) -> None:
        """Single tick (for step mode)."""
        self._tick_and_snapshot()

    @property
    def running(self) -> bool:
        return self._running

    def get_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_snapshot

    def _run_loop(self) -> None:
        while self._running:
            self._tick_and_snapshot()
            time.sleep(0.01)  # ~100 ticks/sec max

    def _tick_and_snapshot(self) -> None:
        state = self.engine.tick()

        # Run analyzers
        all_detections = []
        for analyzer in self._analyzers:
            dets = analyzer.analyze(state, self.engine.bus)
            all_detections.extend(dets)

        # Track structure persistence
        self._structure_analyzer.update(all_detections, state.tick)

        # Build snapshot
        snapshot = {
            "tick": state.tick,
            "time": state.time,
            "dt": state.dt,
            "total_energy": state.total_energy,
            "pac_total": state.pac_total,
            "mass_total": float(state.M.sum().item()),
            "temp_mean": float(state.T.mean().item()),
            "detections": [
                {"kind": d.kind, "position": d.position, "properties": d.properties}
                for d in all_detections
            ],
            "stable_structures": self._structure_analyzer.stable_count,
            "tracked_structures": self._structure_analyzer.tracked_count,
            "metrics": {k: v for k, v in state.metrics.items() if isinstance(v, (int, float, str))},
            # Field data as lists for JSON serialisation
            "fields": {
                "E": state.E.tolist(),
                "I": state.I.tolist(),
                "M": state.M.tolist(),
                "T": state.T.tolist(),
            },
        }

        with self._lock:
            self._latest_snapshot = snapshot


def create_app(config: Optional[SimulationConfig] = None) -> "FastAPI":
    """Create the FastAPI dashboard app."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. pip install fastapi uvicorn")

    app = FastAPI(title="Reality Engine v3 Dashboard")
    runner = SimulationRunner(config)

    frontend_dir = Path(__file__).parent / "frontend"

    # --- REST endpoints ---

    @app.get("/")
    async def index():
        index_path = frontend_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return HTMLResponse("<h1>Reality Engine v3</h1><p>Frontend not built yet.</p>")

    @app.get("/api/state")
    async def get_state():
        snap = runner.get_snapshot()
        if snap is None:
            return {"status": "no_data"}
        # Return without field data (too large for polling)
        return {k: v for k, v in snap.items() if k != "fields"}

    @app.get("/api/config")
    async def get_config():
        c = runner.config
        return {
            "nu": c.nu, "nv": c.nv, "dt": c.dt,
            "xi": c.xi, "phi": c.phi,
            "running": runner.running,
            "tick": runner.engine.state.tick if runner.engine.initialized else 0,
        }

    @app.post("/api/start")
    async def start():
        runner.start()
        return {"status": "running"}

    @app.post("/api/stop")
    async def stop():
        runner.stop()
        return {"status": "stopped"}

    @app.post("/api/step")
    async def step():
        runner.step_once()
        return {"status": "stepped", "tick": runner.engine.state.tick}

    @app.post("/api/reset")
    async def reset():
        runner.stop()
        runner.engine.initialize("big_bang", temperature=2.0)
        return {"status": "reset"}

    # --- WebSocket for real-time streaming ---

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                snap = runner.get_snapshot()
                if snap is not None:
                    await websocket.send_json(snap)
                await asyncio.sleep(0.1)  # 10 FPS to dashboard
        except WebSocketDisconnect:
            pass

    return app


# --- CLI entry point ---

if __name__ == "__main__":
    import uvicorn
    app = create_app(SimulationConfig(nu=64, nv=16, device=torch.device("cpu")))
    print("Reality Engine v3 Dashboard: http://localhost:8050")
    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="info")
