"""Phase 0 tests — EventBus, FieldState, Operator protocol, Pipeline, Engine."""

import pytest
import torch

from src.v3.engine.event_bus import EventBus
from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.operators.protocol import Operator, Pipeline
from src.v3.engine.engine import Engine


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class TestEventBus:
    def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []
        bus.subscribe("tick", lambda d: received.append(d))
        bus.emit("tick", {"n": 1})
        assert len(received) == 1
        assert received[0]["n"] == 1
        assert received[0]["event"] == "tick"

    def test_multiple_subscribers(self):
        bus = EventBus()
        a, b = [], []
        bus.subscribe("x", lambda d: a.append(d))
        bus.subscribe("x", lambda d: b.append(d))
        bus.emit("x")
        assert len(a) == 1
        assert len(b) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        calls = []
        cb = lambda d: calls.append(d)
        bus.subscribe("x", cb)
        bus.unsubscribe("x", cb)
        bus.emit("x")
        assert len(calls) == 0

    def test_unsubscribe_nonexistent_is_safe(self):
        bus = EventBus()
        bus.unsubscribe("x", lambda d: None)  # should not raise

    def test_history_capped(self):
        bus = EventBus()
        bus._max_history = 10
        for i in range(20):
            bus.emit("tick", {"i": i})
        assert len(bus.history) == 10
        assert bus.history[0]["i"] == 10  # oldest kept

    def test_clear(self):
        bus = EventBus()
        bus.subscribe("x", lambda d: None)
        bus.emit("x")
        bus.clear()
        assert len(bus.history) == 0

    def test_emit_no_data(self):
        bus = EventBus()
        received = []
        bus.subscribe("ping", lambda d: received.append(d))
        bus.emit("ping")
        assert received[0] == {"event": "ping"}


# ---------------------------------------------------------------------------
# FieldState
# ---------------------------------------------------------------------------

class TestFieldState:
    def test_creation(self):
        s = FieldState.zeros(16, 8)
        assert s.shape == torch.Size([16, 8])
        assert s.tick == 0
        assert s.device == torch.device("cpu")

    def test_big_bang(self):
        s = FieldState.big_bang(16, 8, temperature=3.0)
        assert s.E.std() > 0  # not all zeros
        assert s.M.sum() == 0  # memory starts empty
        assert s.T.mean().item() == pytest.approx(3.0)

    def test_replace_returns_new(self):
        s = FieldState.zeros(8, 8)
        s2 = s.replace(tick=5, dt=0.01)
        assert s.tick == 0  # original unchanged
        assert s2.tick == 5
        assert s2.dt == 0.01

    def test_immutable(self):
        s = FieldState.zeros(8, 8)
        with pytest.raises(AttributeError):
            s.tick = 99

    def test_disequilibrium(self):
        E = torch.ones(4, 4)
        I = torch.zeros(4, 4)
        s = FieldState(E=E, I=I, M=torch.zeros(4, 4), T=torch.zeros(4, 4))
        assert s.disequilibrium.mean().item() == pytest.approx(1.0)

    def test_total_energy(self):
        E = torch.ones(4, 4) * 2
        I = torch.ones(4, 4) * 3
        M = torch.zeros(4, 4)
        T = torch.zeros(4, 4)
        s = FieldState(E=E, I=I, M=M, T=T)
        expected = (4 * 16) + (9 * 16)  # E²=4*16, I²=9*16
        assert s.total_energy == pytest.approx(expected)

    def test_pac_total(self):
        E = torch.ones(4, 4)
        I = torch.ones(4, 4) * 2
        M = torch.ones(4, 4) * 3
        T = torch.zeros(4, 4)
        s = FieldState(E=E, I=I, M=M, T=T)
        expected = 16 * (1 + 2 + 3)  # PAC = E + I + M (coefficient 1.0 on all)
        assert s.pac_total == pytest.approx(expected)


# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        c = SimulationConfig()
        assert c.nu == 128
        assert c.nv == 32
        assert c.dt == 0.001

    def test_device_override(self):
        c = SimulationConfig(device=torch.device("cpu"))
        assert c.get_device() == torch.device("cpu")


# ---------------------------------------------------------------------------
# Operator protocol & Pipeline
# ---------------------------------------------------------------------------

class _DoubleE:
    """Test operator that doubles E field."""
    @property
    def name(self):
        return "double_e"

    def __call__(self, state, config, bus=None):
        return state.replace(E=state.E * 2)


class _AddOne:
    """Test operator that adds 1 to I field."""
    @property
    def name(self):
        return "add_one"

    def __call__(self, state, config, bus=None):
        return state.replace(I=state.I + 1)


class TestPipeline:
    def test_conforms_to_protocol(self):
        assert isinstance(_DoubleE(), Operator)

    def test_single_operator(self):
        p = Pipeline([_DoubleE()])
        s = FieldState.zeros(4, 4)
        s = s.replace(E=torch.ones(4, 4))
        s2 = p(s, SimulationConfig())
        assert s2.E.mean().item() == pytest.approx(2.0)

    def test_chained_operators(self):
        p = Pipeline([_DoubleE(), _AddOne()])
        s = FieldState.zeros(4, 4)
        s = s.replace(E=torch.ones(4, 4))
        s2 = p(s, SimulationConfig())
        assert s2.E.mean().item() == pytest.approx(2.0)
        assert s2.I.mean().item() == pytest.approx(1.0)

    def test_add_remove(self):
        p = Pipeline()
        p.add(_DoubleE()).add(_AddOne())
        assert len(p) == 2
        assert p.operator_names == ["double_e", "add_one"]
        p.remove("double_e")
        assert len(p) == 1

    def test_empty_pipeline_is_identity(self):
        p = Pipeline()
        s = FieldState.zeros(4, 4)
        s2 = p(s, SimulationConfig())
        assert torch.equal(s.E, s2.E)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TestEngine:
    def test_initialize_big_bang(self):
        e = Engine(config=SimulationConfig(nu=16, nv=8, device=torch.device("cpu")))
        e.initialize("big_bang")
        assert e.initialized
        assert e.state.shape == torch.Size([16, 8])

    def test_initialize_zeros(self):
        e = Engine(config=SimulationConfig(nu=16, nv=8, device=torch.device("cpu")))
        e.initialize("zeros")
        assert e.state.E.sum().item() == 0.0

    def test_tick_advances_state(self):
        e = Engine(
            config=SimulationConfig(nu=8, nv=4, device=torch.device("cpu")),
            pipeline=Pipeline([_DoubleE()]),
        )
        e.initialize("big_bang")
        old_tick = e.state.tick
        e.tick()
        assert e.state.tick == old_tick + 1

    def test_tick_emits_event(self):
        bus = EventBus()
        events = []
        bus.subscribe("tick_complete", lambda d: events.append(d))
        e = Engine(
            config=SimulationConfig(nu=8, nv=4, device=torch.device("cpu")),
            bus=bus,
        )
        e.initialize("big_bang")
        e.tick()
        assert len(events) == 1
        assert events[0]["tick"] == 1

    def test_run_n_ticks(self):
        e = Engine(config=SimulationConfig(nu=8, nv=4, device=torch.device("cpu")))
        e.initialize("zeros")
        e.run(10)
        assert e.state.tick == 10

    def test_run_until(self):
        e = Engine(
            config=SimulationConfig(nu=8, nv=4, device=torch.device("cpu")),
            pipeline=Pipeline([_AddOne()]),
        )
        e.initialize("zeros")
        final = e.run_until(lambda s: s.I.mean().item() > 5)
        assert final.I.mean().item() > 5

    def test_uninitialized_raises(self):
        e = Engine()
        with pytest.raises(RuntimeError):
            e.tick()

    def test_time_accumulates(self):
        cfg = SimulationConfig(nu=8, nv=4, dt=0.01, device=torch.device("cpu"))
        e = Engine(config=cfg)
        e.initialize("zeros")
        e.run(10)
        assert e.state.time == pytest.approx(0.1)
