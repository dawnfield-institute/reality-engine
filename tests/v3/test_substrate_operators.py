"""Phase 1 tests — MobiusManifold, RBFOperator, QBEOperator, EulerIntegrator, Pipeline integration."""

import math
import pytest
import torch

from src.v3.substrate.manifold import MobiusManifold
from src.v3.substrate.constants import XI, PHI, LAMBDA, ALPHA_PAC
from src.v3.operators.rbf import RBFOperator
from src.v3.operators.qbe import QBEOperator
from src.v3.operators.integrator import EulerIntegrator
from src.v3.operators.protocol import Pipeline
from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.engine.engine import Engine


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_xi_value(self):
        assert XI == pytest.approx(1 + math.pi / 55, rel=1e-10)

    def test_phi_value(self):
        assert PHI == pytest.approx((1 + math.sqrt(5)) / 2, rel=1e-10)

    def test_lambda(self):
        assert LAMBDA == 0.020

    def test_alpha_pac(self):
        assert ALPHA_PAC == 0.964


# ---------------------------------------------------------------------------
# MobiusManifold
# ---------------------------------------------------------------------------

class TestMobiusManifold:
    def test_creation(self):
        m = MobiusManifold(16, 8)
        assert m.nu == 16
        assert m.nv == 8

    def test_odd_nu_rejected(self):
        with pytest.raises(ValueError):
            MobiusManifold(15, 8)

    def test_twist_is_involution_up_to_sign(self):
        """Twisting twice should negate (since twist is shift+flip, and antiperiodic)."""
        m = MobiusManifold(16, 8)
        f = torch.randn(16, 8, dtype=torch.float64)
        # twist(twist(f)) should equal f (applying the map twice returns to original)
        f_tt = m.twist(m.twist(f))
        assert torch.allclose(f_tt, f, atol=1e-12)

    def test_antiperiodic_projection(self):
        """After projection, f + f_twisted should be ~0."""
        m = MobiusManifold(16, 8)
        f = torch.randn(16, 8, dtype=torch.float64)
        fp = m.project_antiperiodic(f)
        error = m.validate_antiperiodicity(fp)
        assert error < 1e-12

    def test_laplacian_of_constant_is_zero(self):
        m = MobiusManifold(16, 8)
        f = torch.ones(16, 8, dtype=torch.float64) * 5.0
        lap = m.laplacian(f)
        assert lap.abs().max().item() < 1e-10

    def test_laplacian_shape_preserved(self):
        m = MobiusManifold(32, 16)
        f = torch.randn(32, 16, dtype=torch.float64)
        assert m.laplacian(f).shape == (32, 16)

    def test_gradient_magnitude_of_constant_near_zero(self):
        m = MobiusManifold(16, 8)
        f = torch.ones(16, 8, dtype=torch.float64) * 3.0
        gm = m.gradient_magnitude(f)
        assert gm.max().item() < 1e-4  # small due to sqrt(eps)

    def test_coordinate_grids(self):
        m = MobiusManifold(16, 8)
        assert m.u_grid.shape == (16, 8)
        assert m.v_grid.shape == (16, 8)
        assert m.u_grid[0, 0].item() == pytest.approx(0.0)
        assert m.v_grid[0, 0].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RBFOperator
# ---------------------------------------------------------------------------

class TestRBFOperator:
    def test_produces_dE_dt(self):
        op = RBFOperator()
        s = FieldState.big_bang(16, 8, device=torch.device("cpu"))
        cfg = SimulationConfig(nu=16, nv=8, device=torch.device("cpu"))
        s2 = op(s, cfg)
        assert "dE_dt" in s2.metrics
        assert s2.metrics["dE_dt"].shape == (16, 8)

    def test_zero_fields_give_zero_balance(self):
        op = RBFOperator()
        s = FieldState.zeros(16, 8)
        cfg = SimulationConfig(nu=16, nv=8, device=torch.device("cpu"))
        s2 = op(s, cfg)
        assert s2.metrics["dE_dt"].abs().max().item() < 1e-10

    def test_emits_event(self):
        bus = EventBus()
        events = []
        bus.subscribe("rbf_computed", lambda d: events.append(d))
        op = RBFOperator()
        s = FieldState.big_bang(16, 8, device=torch.device("cpu"))
        cfg = SimulationConfig(nu=16, nv=8, device=torch.device("cpu"))
        op(s, cfg, bus)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# QBEOperator
# ---------------------------------------------------------------------------

class TestQBEOperator:
    def test_produces_dI_dt(self):
        rbf = RBFOperator()
        qbe = QBEOperator()
        s = FieldState.big_bang(16, 8, device=torch.device("cpu"))
        cfg = SimulationConfig(nu=16, nv=8, device=torch.device("cpu"))
        s = rbf(s, cfg)
        s = qbe(s, cfg)
        assert "dI_dt" in s.metrics

    def test_passthrough_without_rbf(self):
        qbe = QBEOperator()
        s = FieldState.zeros(16, 8)
        cfg = SimulationConfig(device=torch.device("cpu"))
        s2 = qbe(s, cfg)
        assert "dI_dt" not in s2.metrics  # no dE_dt → passthrough

    def test_approximate_balance(self):
        """dE_dt + dI_dt should be small (close to zero for balance)."""
        rbf = RBFOperator()
        qbe = QBEOperator()
        s = FieldState.big_bang(16, 8, device=torch.device("cpu"))
        cfg = SimulationConfig(nu=16, nv=8, device=torch.device("cpu"))
        s = rbf(s, cfg)
        s = qbe(s, cfg)
        total = s.metrics["dE_dt"] + s.metrics["dI_dt"]
        # Should be close to λ·QPL (small), not to dE_dt magnitude
        assert total.abs().mean().item() < 1.0


# ---------------------------------------------------------------------------
# EulerIntegrator
# ---------------------------------------------------------------------------

class TestEulerIntegrator:
    def test_applies_derivatives(self):
        euler = EulerIntegrator()
        s = FieldState.zeros(4, 4)
        dE = torch.ones(4, 4, dtype=torch.float64)
        dI = torch.ones(4, 4, dtype=torch.float64) * 2
        s = s.replace(metrics={"dE_dt": dE, "dI_dt": dI})
        cfg = SimulationConfig(dt=0.1, device=torch.device("cpu"))
        s2 = euler(s, cfg)
        assert s2.E.mean().item() == pytest.approx(0.1)
        assert s2.I.mean().item() == pytest.approx(0.2)

    def test_passthrough_without_derivatives(self):
        euler = EulerIntegrator()
        s = FieldState.zeros(4, 4)
        cfg = SimulationConfig(device=torch.device("cpu"))
        s2 = euler(s, cfg)
        assert torch.equal(s.E, s2.E)


# ---------------------------------------------------------------------------
# Pipeline integration: RBF → QBE → Euler
# ---------------------------------------------------------------------------

class TestCorePipeline:
    def test_full_pipeline_runs(self):
        pipe = Pipeline([RBFOperator(), QBEOperator(), EulerIntegrator()])
        s = FieldState.big_bang(16, 8, device=torch.device("cpu"))
        cfg = SimulationConfig(nu=16, nv=8, device=torch.device("cpu"))
        s2 = pipe(s, cfg)
        # Fields should have changed
        assert not torch.equal(s.E, s2.E)

    def test_energy_direction(self):
        """After one tick, total energy should not explode."""
        pipe = Pipeline([RBFOperator(), QBEOperator(), EulerIntegrator()])
        s = FieldState.big_bang(16, 8, device=torch.device("cpu"), temperature=1.0)
        cfg = SimulationConfig(nu=16, nv=8, dt=0.0001, device=torch.device("cpu"))
        s2 = pipe(s, cfg)
        # With small dt, energy shouldn't grow by more than 10%
        assert s2.total_energy < s.total_energy * 1.1

    def test_engine_multi_tick(self):
        """Run 100 ticks through the Engine — should not NaN or explode."""
        cfg = SimulationConfig(nu=16, nv=8, dt=0.0001, device=torch.device("cpu"))
        engine = Engine(
            config=cfg,
            pipeline=Pipeline([RBFOperator(), QBEOperator(), EulerIntegrator()]),
        )
        engine.initialize("big_bang", temperature=1.0)
        engine.run(100)
        assert not torch.isnan(engine.state.E).any()
        assert not torch.isinf(engine.state.E).any()
        assert engine.state.tick == 100
