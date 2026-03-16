"""Phase 3 tests — AdaptiveOperator and TimeEmergenceOperator."""

import pytest
import torch

from src.v3.operators.adaptive import AdaptiveOperator
from src.v3.operators.time_emergence import TimeEmergenceOperator
from src.v3.operators.rbf import RBFOperator
from src.v3.operators.qbe import QBEOperator
from src.v3.operators.integrator import EulerIntegrator
from src.v3.operators.memory import MemoryOperator
from src.v3.operators.confluence import ConfluenceOperator
from src.v3.operators.temperature import TemperatureOperator
from src.v3.operators.thermal_noise import ThermalNoiseOperator
from src.v3.operators.normalization import NormalizationOperator
from src.v3.operators.protocol import Pipeline
from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.engine import Engine
from src.v3.engine.event_bus import EventBus


CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# AdaptiveOperator
# ---------------------------------------------------------------------------

class TestAdaptiveOperator:
    def test_gamma_increases_on_energy_growth(self):
        op = AdaptiveOperator()
        cfg = SimulationConfig(nu=8, nv=4, gamma_damping=0.01, device=CPU)
        # First call: baseline
        s1 = FieldState.big_bang(8, 4, temperature=1.0, device=CPU)
        op(s1, cfg)
        # Second call: much higher energy
        E_big = torch.ones(8, 4, dtype=torch.float64) * 100
        s2 = FieldState(E=E_big, I=E_big, M=torch.zeros(8, 4, dtype=torch.float64),
                        T=torch.ones(8, 4, dtype=torch.float64))
        op(s2, cfg)
        assert cfg.gamma_damping > 0.01

    def test_gamma_stays_in_bounds(self):
        op = AdaptiveOperator(gamma_min=0.001, gamma_max=0.1)
        cfg = SimulationConfig(nu=8, nv=4, gamma_damping=0.001, device=CPU)
        # Drive energy down repeatedly
        for i in range(100):
            E = torch.ones(8, 4, dtype=torch.float64) * (0.01 / (i + 1))
            s = FieldState(E=E, I=E, M=torch.zeros(8, 4, dtype=torch.float64),
                           T=torch.ones(8, 4, dtype=torch.float64))
            op(s, cfg)
        assert cfg.gamma_damping >= op.gamma_min

    def test_disabled_passthrough(self):
        op = AdaptiveOperator()
        cfg = SimulationConfig(nu=8, nv=4, enable_adaptive=False, device=CPU)
        s = FieldState.big_bang(8, 4, device=CPU)
        s2 = op(s, cfg)
        assert "adaptive_gamma" not in s2.metrics

    def test_emits_event(self):
        bus = EventBus()
        events = []
        bus.subscribe("parameters_adapted", lambda d: events.append(d))
        op = AdaptiveOperator()
        cfg = SimulationConfig(nu=8, nv=4, device=CPU)
        s = FieldState.big_bang(8, 4, device=CPU)
        op(s, cfg, bus)
        op(s, cfg, bus)  # needs 2 calls (first sets baseline)
        assert len(events) == 2


# ---------------------------------------------------------------------------
# TimeEmergenceOperator
# ---------------------------------------------------------------------------

class TestTimeEmergenceOperator:
    def test_dt_decreases_with_disequilibrium(self):
        op = TimeEmergenceOperator(kappa=1.0, dt_base=0.001)
        cfg = SimulationConfig(nu=8, nv=4, dt=0.001, device=CPU)
        E = torch.ones(8, 4, dtype=torch.float64) * 10
        I = torch.zeros(8, 4, dtype=torch.float64)
        s = FieldState(E=E, I=I, M=torch.zeros(8, 4, dtype=torch.float64),
                       T=torch.ones(8, 4, dtype=torch.float64))
        op(s, cfg)
        # max |E-I| = 10, so dt = 0.001 / (1 + 1.0 * 10) = 0.001/11
        assert cfg.dt < 0.001

    def test_equilibrium_gives_base_dt(self):
        op = TimeEmergenceOperator(kappa=1.0, dt_base=0.001)
        cfg = SimulationConfig(nu=8, nv=4, dt=0.001, device=CPU)
        E = torch.ones(8, 4, dtype=torch.float64)
        s = FieldState(E=E, I=E.clone(), M=torch.zeros(8, 4, dtype=torch.float64),
                       T=torch.ones(8, 4, dtype=torch.float64))
        op(s, cfg)
        assert cfg.dt == pytest.approx(0.001, rel=0.01)

    def test_emits_event(self):
        bus = EventBus()
        events = []
        bus.subscribe("time_emerged", lambda d: events.append(d))
        op = TimeEmergenceOperator()
        cfg = SimulationConfig(nu=8, nv=4, device=CPU)
        s = FieldState.big_bang(8, 4, device=CPU)
        op(s, cfg, bus)
        assert len(events) == 1
        assert "dt" in events[0]

    def test_metrics_populated(self):
        op = TimeEmergenceOperator()
        cfg = SimulationConfig(nu=8, nv=4, device=CPU)
        s = FieldState.big_bang(8, 4, device=CPU)
        s2 = op(s, cfg)
        assert "emergent_dt" in s2.metrics
        assert "max_disequilibrium" in s2.metrics


# ---------------------------------------------------------------------------
# Full pipeline with adaptive + time emergence
# ---------------------------------------------------------------------------

class TestCompletePipeline:
    def _build_complete_pipeline(self):
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

    def test_500_ticks_stable(self):
        cfg = SimulationConfig(nu=16, nv=8, dt=0.0001, device=CPU)
        engine = Engine(config=cfg, pipeline=self._build_complete_pipeline())
        engine.initialize("big_bang", temperature=1.0)
        engine.run(500)
        s = engine.state
        assert not torch.isnan(s.E).any()
        assert not torch.isinf(s.E).any()
        assert s.tick == 500

    def test_pipeline_composability(self):
        """Can remove operators and still run."""
        pipe = self._build_complete_pipeline()
        pipe.remove("thermal_noise")
        pipe.remove("adaptive")
        cfg = SimulationConfig(nu=16, nv=8, dt=0.0001, device=CPU)
        engine = Engine(config=cfg, pipeline=pipe)
        engine.initialize("big_bang", temperature=1.0)
        engine.run(50)
        assert engine.state.tick == 50
