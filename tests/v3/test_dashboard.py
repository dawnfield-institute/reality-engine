"""Phase 5 tests — Dashboard server and SimulationRunner."""

import pytest
import torch

from src.v3.dashboard.server import SimulationRunner, build_default_pipeline, create_app
from src.v3.engine.config import SimulationConfig


CPU = torch.device("cpu")


class TestSimulationRunner:
    def test_creation(self):
        r = SimulationRunner(SimulationConfig(nu=16, nv=8, device=CPU))
        assert r.engine.initialized
        assert not r.running

    def test_step_once(self):
        r = SimulationRunner(SimulationConfig(nu=16, nv=8, device=CPU))
        r.step_once()
        assert r.engine.state.tick == 1

    def test_snapshot_after_step(self):
        r = SimulationRunner(SimulationConfig(nu=16, nv=8, device=CPU))
        r.step_once()
        snap = r.get_snapshot()
        assert snap is not None
        assert snap["tick"] == 1
        assert "fields" in snap
        assert "E" in snap["fields"]
        assert "detections" in snap

    def test_multiple_steps(self):
        r = SimulationRunner(SimulationConfig(nu=16, nv=8, device=CPU))
        for _ in range(10):
            r.step_once()
        snap = r.get_snapshot()
        assert snap["tick"] == 10

    def test_start_stop(self):
        r = SimulationRunner(SimulationConfig(nu=16, nv=8, device=CPU))
        r.start()
        assert r.running
        import time
        time.sleep(0.2)  # let it run a few ticks
        r.stop()
        assert not r.running
        assert r.engine.state.tick > 0

    def test_snapshot_has_detections_list(self):
        r = SimulationRunner(SimulationConfig(nu=16, nv=8, device=CPU))
        r.step_once()
        snap = r.get_snapshot()
        assert isinstance(snap["detections"], list)
        assert isinstance(snap["stable_structures"], int)

    def test_field_data_shape(self):
        r = SimulationRunner(SimulationConfig(nu=16, nv=8, device=CPU))
        r.step_once()
        snap = r.get_snapshot()
        E = snap["fields"]["E"]
        assert len(E) == 16
        assert len(E[0]) == 8


class TestBuildDefaultPipeline:
    def test_has_all_operators(self):
        p = build_default_pipeline()
        names = p.operator_names
        assert "rbf" in names
        assert "qbe" in names
        assert "euler" in names
        assert "memory" in names
        assert "confluence" in names
        assert "temperature" in names
        assert "thermal_noise" in names
        assert "normalization" in names
        assert "adaptive" in names
        assert "time_emergence" in names


class TestCreateApp:
    def test_app_creation(self):
        app = create_app(SimulationConfig(nu=16, nv=8, device=CPU))
        assert app.title == "Reality Engine v3 Dashboard"
        # Check routes exist
        routes = [r.path for r in app.routes]
        assert "/api/state" in routes
        assert "/api/config" in routes
        assert "/api/start" in routes
        assert "/ws" in routes
