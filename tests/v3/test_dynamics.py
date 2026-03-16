"""Phase 2 tests â€” Memory, Confluence, Temperature, ThermalNoise, Normalization operators."""

import pytest
import torch

from src.v3.operators.memory import MemoryOperator
from src.v3.operators.confluence import ConfluenceOperator
from src.v3.operators.temperature import TemperatureOperator
from src.v3.operators.thermal_noise import ThermalNoiseOperator
from src.v3.operators.normalization import NormalizationOperator
from src.v3.operators.rbf import RBFOperator
from src.v3.operators.qbe import QBEOperator
from src.v3.operators.integrator import EulerIntegrator
from src.v3.operators.protocol import Pipeline
from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.engine import Engine
from src.v3.substrate.manifold import MobiusManifold


CPU = torch.device("cpu")


def _hot_state(nu=16, nv=8, temp=3.0):
    """Create a state with disequilibrium (mass should grow)."""
    E = torch.randn(nu, nv, dtype=torch.float64) * temp
    I = torch.randn(nu, nv, dtype=torch.float64) * temp
    M = torch.zeros(nu, nv, dtype=torch.float64)
    T = torch.full((nu, nv), temp, dtype=torch.float64)
    return FieldState(E=E, I=I, M=M, T=T)


# ---------------------------------------------------------------------------
# MemoryOperator
# ---------------------------------------------------------------------------

class TestMemoryOperator:
    def test_mass_grows_in_disequilibrium(self):
        op = MemoryOperator()
        s = _hot_state()
        cfg = SimulationConfig(nu=16, nv=8, dt=0.01, device=CPU)
        s2 = op(s, cfg)
        assert s2.M.sum().item() > 0

    def test_mass_stays_nonnegative(self):
        op = MemoryOperator()
        s = _hot_state()
        cfg = SimulationConfig(nu=16, nv=8, dt=0.01, device=CPU)
        s2 = op(s, cfg)
        assert (s2.M >= 0).all()

    def test_zero_disequilibrium_no_mass(self):
        op = MemoryOperator()
        E = torch.ones(8, 4, dtype=torch.float64)
        s = FieldState(E=E, I=E.clone(), M=torch.zeros(8, 4, dtype=torch.float64),
                       T=torch.ones(8, 4, dtype=torch.float64))
        cfg = SimulationConfig(nu=8, nv=4, dt=0.01, device=CPU)
        s2 = op(s, cfg)
        assert s2.M.sum().item() < 1e-6

    def test_emits_event(self):
        from src.v3.engine.event_bus import EventBus
        bus = EventBus()
        events = []
        bus.subscribe("memory_evolved", lambda d: events.append(d))
        op = MemoryOperator()
        s = _hot_state()
        cfg = SimulationConfig(nu=16, nv=8, dt=0.01, device=CPU)
        op(s, cfg, bus)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# ConfluenceOperator
# ---------------------------------------------------------------------------

class TestConfluenceOperator:
    def test_modifies_I_field(self):
        op = ConfluenceOperator()
        s = _hot_state()
        cfg = SimulationConfig(nu=16, nv=8, device=CPU)
        s2 = op(s, cfg)
        assert not torch.equal(s.I, s2.I)

    def test_disabled_passthrough(self):
        op = ConfluenceOperator()
        s = _hot_state()
        cfg = SimulationConfig(nu=16, nv=8, enable_confluence=False, device=CPU)
        s2 = op(s, cfg)
        assert torch.equal(s.I, s2.I)

    def test_preserves_antiperiodicity(self):
        """After confluence, I should be closer to antiperiodic."""
        op = ConfluenceOperator()
        s = _hot_state()
        cfg = SimulationConfig(nu=16, nv=8, confluence_weight=1.0, device=CPU)
        s2 = op(s, cfg)
        m = MobiusManifold(16, 8)
        error = m.validate_antiperiodicity(s2.I)
        assert error < 1e-10

    def test_confluence_every_n(self):
        op = ConfluenceOperator()
        s = _hot_state()
        cfg = SimulationConfig(nu=16, nv=8, confluence_every=3, device=CPU)
        # First two calls should be passthrough
        s2 = op(s, cfg)
        s3 = op(s, cfg)
        assert torch.equal(s.I, s2.I)  # tick 1: skip
        assert torch.equal(s.I, s3.I)  # tick 2: skip
        s4 = op(s, cfg)
        assert not torch.equal(s.I, s4.I)  # tick 3: apply


# ---------------------------------------------------------------------------
# TemperatureOperator
# ---------------------------------------------------------------------------

class TestTemperatureOperator:
    def test_temperature_from_disequilibrium(self):
        op = TemperatureOperator(blend=0.0)
        E = torch.ones(8, 4, dtype=torch.float64) * 5
        I = torch.zeros(8, 4, dtype=torch.float64)
        T = torch.ones(8, 4, dtype=torch.float64)
        s = FieldState(E=E, I=I, M=torch.zeros(8, 4, dtype=torch.float64), T=T)
        cfg = SimulationConfig(nu=8, nv=4, t_min=0.0, t_max=100.0, device=CPU)
        s2 = op(s, cfg)
        # |E-I| = 5 everywhere, blend=0 â†’ T should be 5
        assert s2.T.mean().item() == pytest.approx(5.0)

    def test_temperature_clamped(self):
        op = TemperatureOperator(blend=0.0)
        E = torch.ones(8, 4, dtype=torch.float64) * 100
        I = torch.zeros(8, 4, dtype=torch.float64)
        s = FieldState(E=E, I=I, M=torch.zeros(8, 4, dtype=torch.float64),
                       T=torch.ones(8, 4, dtype=torch.float64))
        cfg = SimulationConfig(nu=8, nv=4, t_max=10.0, device=CPU)
        s2 = op(s, cfg)
        assert s2.T.max().item() <= 10.0


# ---------------------------------------------------------------------------
# ThermalNoiseOperator
# ---------------------------------------------------------------------------

class TestThermalNoiseOperator:
    def test_adds_noise(self):
        torch.manual_seed(42)
        op = ThermalNoiseOperator()
        s = FieldState.zeros(16, 8)
        s = s.replace(T=torch.ones(16, 8, dtype=torch.float64))
        cfg = SimulationConfig(nu=16, nv=8, dt=0.01, noise_scale=0.1, device=CPU)
        s2 = op(s, cfg)
        assert not torch.equal(s.E, s2.E)

    def test_disabled_passthrough(self):
        op = ThermalNoiseOperator()
        s = FieldState.zeros(8, 4)
        s = s.replace(T=torch.ones(8, 4, dtype=torch.float64))
        cfg = SimulationConfig(nu=8, nv=4, enable_thermal_noise=False, device=CPU)
        s2 = op(s, cfg)
        assert torch.equal(s.E, s2.E)

    def test_noise_scales_with_temperature(self):
        """Higher temperature â†’ more noise variance."""
        torch.manual_seed(0)
        op = ThermalNoiseOperator()
        cfg = SimulationConfig(nu=32, nv=16, dt=0.01, noise_scale=1.0, device=CPU)

        s_cold = FieldState.zeros(32, 16).replace(T=torch.full((32, 16), 0.1, dtype=torch.float64))
        s_hot = FieldState.zeros(32, 16).replace(T=torch.full((32, 16), 10.0, dtype=torch.float64))

        torch.manual_seed(0)
        cold_out = op(s_cold, cfg)
        torch.manual_seed(0)
        hot_out = op(s_hot, cfg)

        cold_var = (cold_out.E - s_cold.E).var().item()
        hot_var = (hot_out.E - s_hot.E).var().item()
        assert hot_var > cold_var


# ---------------------------------------------------------------------------
# NormalizationOperator
# ---------------------------------------------------------------------------

class TestNormalizationOperator:
    def test_pac_conserving_normalization(self):
        """QBE cross-injection: tanh losses go to dual field. PAC conserved."""
        op = NormalizationOperator()
        E = torch.full((8, 4), 1000.0, dtype=torch.float64)
        I = torch.full((8, 4), 500.0, dtype=torch.float64)  # asymmetric
        M = torch.zeros(8, 4, dtype=torch.float64)
        s = FieldState(E=E, I=I, M=M, T=torch.zeros(8, 4, dtype=torch.float64))
        cfg = SimulationConfig(nu=8, nv=4, field_scale=50.0, device=CPU)
        s2 = op(s, cfg)
        # PAC conserved: E + I + M before = after
        pac_before = (s.E + s.I + s.M).sum().item()
        pac_after = (s2.E + s2.I + s2.M).sum().item()
        assert abs(pac_after - pac_before) < 1.0
        # QBE cross-injection: E got I's tanh loss, I got E's tanh loss
        # With asymmetric fields, E should be reduced (lost more to tanh,
        # got back less from I's smaller loss)
        assert s2.E.mean().item() < s.E.mean().item()

    def test_mass_nonnegative(self):
        op = NormalizationOperator()
        M = torch.full((8, 4), -5.0, dtype=torch.float64)
        s = FieldState(E=torch.zeros(8, 4, dtype=torch.float64),
                       I=torch.zeros(8, 4, dtype=torch.float64), M=M,
                       T=torch.zeros(8, 4, dtype=torch.float64))
        cfg = SimulationConfig(nu=8, nv=4, device=CPU)
        s2 = op(s, cfg)
        assert (s2.M >= 0).all()

    def test_disabled_passthrough(self):
        op = NormalizationOperator()
        E = torch.full((8, 4), 1000.0, dtype=torch.float64)
        s = FieldState(E=E, I=E.clone(), M=torch.zeros(8, 4, dtype=torch.float64),
                       T=torch.zeros(8, 4, dtype=torch.float64))
        cfg = SimulationConfig(nu=8, nv=4, enable_normalization=False, device=CPU)
        s2 = op(s, cfg)
        assert torch.equal(s.E, s2.E)


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def _build_full_pipeline(self):
        return Pipeline([
            RBFOperator(),
            QBEOperator(),
            EulerIntegrator(),
            MemoryOperator(),
            ConfluenceOperator(),
            TemperatureOperator(),
            ThermalNoiseOperator(),
            NormalizationOperator(),
        ])

    def test_full_pipeline_runs_100_ticks(self):
        cfg = SimulationConfig(nu=16, nv=8, dt=0.0001, device=CPU)
        engine = Engine(config=cfg, pipeline=self._build_full_pipeline())
        engine.initialize("big_bang", temperature=1.0)
        engine.run(100)
        s = engine.state
        assert not torch.isnan(s.E).any()
        assert not torch.isinf(s.E).any()
        assert s.tick == 100

    def test_mass_accumulates_over_time(self):
        cfg = SimulationConfig(nu=16, nv=8, dt=0.001, device=CPU)
        engine = Engine(config=cfg, pipeline=self._build_full_pipeline())
        engine.initialize("big_bang", temperature=2.0)
        engine.run(50)
        assert engine.state.M.sum().item() > 0

    def test_total_energy_bounded(self):
        cfg = SimulationConfig(nu=16, nv=8, dt=0.0001, device=CPU)
        engine = Engine(config=cfg, pipeline=self._build_full_pipeline())
        engine.initialize("big_bang", temperature=1.0)
        initial_energy = engine.state.total_energy
        engine.run(200)
        # With normalization + Landauer reinjection, energy is bounded
        # but can grow as mass cap returns energy to E+I fields
        assert engine.state.total_energy < initial_energy * 50
