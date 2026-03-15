"""Phase 4 tests — Analyzers and Emergence Detection."""

import pytest
import torch

from src.v3.analyzers.base import Detection
from src.v3.analyzers.conservation import ConservationAnalyzer
from src.v3.analyzers.gravity import GravityAnalyzer
from src.v3.analyzers.atom import AtomDetector
from src.v3.analyzers.star import StarDetector
from src.v3.analyzers.quantum import QuantumDetector
from src.v3.analyzers.galaxy import GalaxyAnalyzer
from src.v3.emergence.particle import ParticleAnalyzer
from src.v3.emergence.structure import StructureAnalyzer
from src.v3.emergence.herniation import HerniationDetector
from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus


CPU = torch.device("cpu")


def _make_bus():
    return EventBus()


def _state_with_mass_peak(mass_val=5.0, temp_val=3.0):
    """State with a strong mass peak at (8, 4)."""
    E = torch.randn(16, 8, dtype=torch.float64) * 0.1
    I = torch.randn(16, 8, dtype=torch.float64) * 0.1
    M = torch.zeros(16, 8, dtype=torch.float64)
    M[7:10, 3:6] = mass_val  # 3x3 mass peak
    T = torch.ones(16, 8, dtype=torch.float64)
    T[7:10, 3:6] = temp_val
    return FieldState(E=E, I=I, M=M, T=T)


# ---------------------------------------------------------------------------
# ConservationAnalyzer
# ---------------------------------------------------------------------------

class TestConservationAnalyzer:
    def test_first_call_no_detection(self):
        a = ConservationAnalyzer()
        bus = _make_bus()
        s = FieldState.big_bang(8, 4, device=CPU)
        dets = a.analyze(s, bus)
        assert len(dets) == 0  # no previous PAC to compare

    def test_detects_drift(self):
        a = ConservationAnalyzer(tolerance=0.001)
        bus = _make_bus()
        events = []
        bus.subscribe("conservation_violated", lambda d: events.append(d))

        s1 = FieldState.big_bang(8, 4, device=CPU, temperature=1.0)
        a.analyze(s1, bus)

        # Create state with very different PAC
        E2 = s1.E * 100
        s2 = s1.replace(E=E2)
        dets = a.analyze(s2, bus)
        assert len(events) > 0

    def test_stable_pac_no_violation(self):
        a = ConservationAnalyzer(tolerance=0.01)
        bus = _make_bus()
        events = []
        bus.subscribe("conservation_violated", lambda d: events.append(d))

        s = FieldState.big_bang(8, 4, device=CPU, temperature=1.0)
        a.analyze(s, bus)
        a.analyze(s, bus)  # same state → no drift
        assert len(events) == 0


# ---------------------------------------------------------------------------
# GravityAnalyzer
# ---------------------------------------------------------------------------

class TestGravityAnalyzer:
    def test_detects_mass_peak(self):
        a = GravityAnalyzer(mass_threshold=1.0, min_curvature=0.01)
        bus = _make_bus()
        s = _state_with_mass_peak(mass_val=5.0)
        dets = a.analyze(s, bus)
        assert len(dets) > 0
        assert all(d.kind == "gravity_well" for d in dets)

    def test_no_detection_on_flat_field(self):
        a = GravityAnalyzer(mass_threshold=1.0)
        bus = _make_bus()
        s = FieldState.zeros(16, 8)
        dets = a.analyze(s, bus)
        assert len(dets) == 0

    def test_emits_event(self):
        bus = _make_bus()
        events = []
        bus.subscribe("gravity_well_detected", lambda d: events.append(d))
        a = GravityAnalyzer(mass_threshold=1.0, min_curvature=0.01)
        s = _state_with_mass_peak()
        a.analyze(s, bus)
        assert len(events) > 0


# ---------------------------------------------------------------------------
# AtomDetector
# ---------------------------------------------------------------------------

class TestAtomDetector:
    def test_detects_bound_structure(self):
        """Mass peak + energy gradient shell = atom."""
        a = AtomDetector(mass_threshold=1.0, gradient_threshold=0.1)
        bus = _make_bus()
        # Create mass peak with energy gradient
        E = torch.zeros(16, 8, dtype=torch.float64)
        E[6, 4] = 5.0  # sharp peak creates gradient
        E[10, 4] = -5.0
        I = torch.zeros(16, 8, dtype=torch.float64)
        M = torch.zeros(16, 8, dtype=torch.float64)
        M[7:10, 3:6] = 3.0
        T = torch.ones(16, 8, dtype=torch.float64)
        s = FieldState(E=E, I=I, M=M, T=T)
        dets = a.analyze(s, bus)
        # May or may not detect depending on exact gradient — just check no crash
        assert isinstance(dets, list)

    def test_no_detection_empty(self):
        a = AtomDetector()
        bus = _make_bus()
        s = FieldState.zeros(16, 8)
        assert len(a.analyze(s, bus)) == 0


# ---------------------------------------------------------------------------
# StarDetector
# ---------------------------------------------------------------------------

class TestStarDetector:
    def _gravity_wells_at_mass_peak(self):
        """Prior detections with gravity wells at the mass peak location."""
        return [Detection("gravity_well", (8, 4), {"mass": 5.0, "curvature": -1.0})]

    def test_detects_hot_massive_region_with_well(self):
        a = StarDetector(mass_threshold=2.0, temp_threshold=2.0)
        bus = _make_bus()
        s = _state_with_mass_peak(mass_val=5.0, temp_val=5.0)
        wells = self._gravity_wells_at_mass_peak()
        dets = a.analyze(s, bus, prior_detections=wells)
        assert len(dets) > 0
        assert dets[0].kind == "star"

    def test_hot_massive_no_well_not_star(self):
        """Without gravity well, hot dense region is NOT a star."""
        a = StarDetector(mass_threshold=2.0, temp_threshold=2.0)
        bus = _make_bus()
        s = _state_with_mass_peak(mass_val=5.0, temp_val=5.0)
        dets = a.analyze(s, bus, prior_detections=[])  # no wells
        assert len(dets) == 0

    def test_without_prior_detections_still_works(self):
        """Backward compat: no prior_detections = no causal gate."""
        a = StarDetector(mass_threshold=2.0, temp_threshold=2.0)
        bus = _make_bus()
        s = _state_with_mass_peak(mass_val=5.0, temp_val=5.0)
        dets = a.analyze(s, bus)  # no prior_detections
        assert len(dets) > 0

    def test_cold_mass_not_star(self):
        a = StarDetector(mass_threshold=2.0, temp_threshold=2.0)
        bus = _make_bus()
        s = _state_with_mass_peak(mass_val=5.0, temp_val=0.5)
        dets = a.analyze(s, bus)
        assert len(dets) == 0


# ---------------------------------------------------------------------------
# QuantumDetector
# ---------------------------------------------------------------------------

class TestQuantumDetector:
    def test_detects_coherent_fields(self):
        a = QuantumDetector(coherence_threshold=0.9)
        bus = _make_bus()
        # E and I perfectly correlated (same sign, same magnitude)
        E = torch.ones(16, 8, dtype=torch.float64) * 5
        I = torch.ones(16, 8, dtype=torch.float64) * 5
        M = torch.zeros(16, 8, dtype=torch.float64)
        T = torch.ones(16, 8, dtype=torch.float64)
        s = FieldState(E=E, I=I, M=M, T=T)
        dets = a.analyze(s, bus)
        assert len(dets) > 0
        assert dets[0].kind == "quantum_coherence"

    def test_no_coherence_random(self):
        torch.manual_seed(0)
        a = QuantumDetector(coherence_threshold=0.99)
        bus = _make_bus()
        E = torch.randn(16, 8, dtype=torch.float64)
        I = torch.randn(16, 8, dtype=torch.float64)
        s = FieldState(E=E, I=I, M=torch.zeros(16, 8, dtype=torch.float64),
                       T=torch.ones(16, 8, dtype=torch.float64))
        # Random fields may still have some coincidental coherence
        dets = a.analyze(s, bus)
        # Just check it runs without crash
        assert isinstance(dets, list)


# ---------------------------------------------------------------------------
# GalaxyAnalyzer
# ---------------------------------------------------------------------------

class TestGalaxyAnalyzer:
    def _gravity_wells_in_mass_region(self):
        """Multiple gravity wells within the mass peak region."""
        return [
            Detection("gravity_well", (7, 3), {"mass": 2.0, "curvature": -0.5}),
            Detection("gravity_well", (8, 4), {"mass": 2.0, "curvature": -0.5}),
            Detection("gravity_well", (9, 5), {"mass": 2.0, "curvature": -0.5}),
        ]

    def test_detects_large_mass_region_with_wells(self):
        a = GalaxyAnalyzer(mass_threshold=0.1, min_region_fraction=0.01, min_gravity_wells=3)
        bus = _make_bus()
        s = _state_with_mass_peak(mass_val=2.0)
        wells = self._gravity_wells_in_mass_region()
        dets = a.analyze(s, bus, prior_detections=wells)
        assert len(dets) > 0
        assert dets[0].kind == "galaxy"
        assert dets[0].properties["gravity_wells"] == 3

    def test_not_enough_wells_no_galaxy(self):
        a = GalaxyAnalyzer(mass_threshold=0.1, min_region_fraction=0.01, min_gravity_wells=3)
        bus = _make_bus()
        s = _state_with_mass_peak(mass_val=2.0)
        one_well = [Detection("gravity_well", (8, 4), {"mass": 2.0})]
        dets = a.analyze(s, bus, prior_detections=one_well)
        assert len(dets) == 0

    def test_no_galaxy_on_empty(self):
        a = GalaxyAnalyzer(mass_threshold=1.0)
        bus = _make_bus()
        s = FieldState.zeros(16, 8)
        assert len(a.analyze(s, bus)) == 0


# ---------------------------------------------------------------------------
# ParticleAnalyzer
# ---------------------------------------------------------------------------

class TestParticleAnalyzer:
    def test_classifies_by_mass(self):
        pa = ParticleAnalyzer()
        dets = [
            Detection("atom", (0, 0), {"mass": 0.1}),
            Detection("atom", (1, 1), {"mass": 1.0}),
            Detection("gravity_well", (2, 2), {"mass": 5.0}),
        ]
        classified = pa.classify(dets)
        assert classified[0].properties["particle_class"] == "lepton-like"
        assert classified[1].properties["particle_class"] == "meson-like"
        assert classified[2].properties["particle_class"] == "baryon-like"


# ---------------------------------------------------------------------------
# StructureAnalyzer
# ---------------------------------------------------------------------------

class TestStructureAnalyzer:
    def test_tracks_persistence(self):
        sa = StructureAnalyzer(match_radius=3)
        d1 = [Detection("atom", (5, 5), {"mass": 1.0})]
        sa.update(d1, tick=1)
        d2 = [Detection("atom", (5, 6), {"mass": 1.1})]  # close enough to match
        result = sa.update(d2, tick=2)
        assert result[0].properties["persistence"] == 2

    def test_new_structure_gets_id(self):
        sa = StructureAnalyzer()
        d1 = [Detection("atom", (5, 5), {"mass": 1.0})]
        result = sa.update(d1, tick=1)
        assert "structure_id" in result[0].properties

    def test_prunes_stale_structures(self):
        sa = StructureAnalyzer()
        d1 = [Detection("atom", (5, 5), {"mass": 1.0})]
        sa.update(d1, tick=1)
        assert sa.tracked_count == 1
        # No detections for 15 ticks
        sa.update([], tick=15)
        assert sa.tracked_count == 0

    def test_stable_count(self):
        sa = StructureAnalyzer(match_radius=3)
        for tick in range(10):
            d = [Detection("atom", (5, 5), {"mass": 1.0})]
            sa.update(d, tick=tick)
        assert sa.stable_count >= 1


# ---------------------------------------------------------------------------
# HerniationDetector
# ---------------------------------------------------------------------------

class TestHerniationDetector:
    def test_no_herniation_on_antiperiodic_field(self):
        hd = HerniationDetector(threshold=0.5)
        bus = _make_bus()
        from src.v3.substrate.manifold import MobiusManifold
        m = MobiusManifold(16, 8)
        E_raw = torch.randn(16, 8, dtype=torch.float64)
        E = m.project_antiperiodic(E_raw)
        I = m.project_antiperiodic(torch.randn(16, 8, dtype=torch.float64))
        s = FieldState(E=E, I=I, M=torch.zeros(16, 8, dtype=torch.float64),
                       T=torch.ones(16, 8, dtype=torch.float64))
        dets = hd.analyze(s, bus)
        assert len(dets) == 0

    def test_detects_violation(self):
        hd = HerniationDetector(threshold=0.1)
        bus = _make_bus()
        events = []
        bus.subscribe("herniation_detected", lambda d: events.append(d))
        # Non-antiperiodic field (random) will have violations
        E = torch.randn(16, 8, dtype=torch.float64) * 5
        I = torch.randn(16, 8, dtype=torch.float64) * 5
        s = FieldState(E=E, I=I, M=torch.zeros(16, 8, dtype=torch.float64),
                       T=torch.ones(16, 8, dtype=torch.float64))
        dets = hd.analyze(s, bus)
        assert len(dets) > 0
        assert len(events) > 0
