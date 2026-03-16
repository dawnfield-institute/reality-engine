"""CLI entry point for Reality Engine v3.

Usage:
    python -m src.v3                    # Run 1000 ticks, print progress
    python -m src.v3 --ticks 5000       # Custom tick count
    python -m src.v3 --dashboard        # Launch dashboard UI
"""

import argparse
import sys
import time

import torch

from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
from src.v3.operators.protocol import Pipeline
from src.v3.operators.rbf import RBFOperator
from src.v3.operators.qbe import QBEOperator
from src.v3.operators.integrator import EulerIntegrator
from src.v3.operators.actualization import ActualizationOperator
from src.v3.operators.memory import MemoryOperator
from src.v3.operators.confluence import ConfluenceOperator
from src.v3.operators.temperature import TemperatureOperator
from src.v3.operators.thermal_noise import ThermalNoiseOperator
from src.v3.operators.normalization import NormalizationOperator
from src.v3.operators.adaptive import AdaptiveOperator
from src.v3.operators.time_emergence import TimeEmergenceOperator
from src.v3.operators.gravity import GravitationalCollapseOperator
from src.v3.operators.spin_statistics import SpinStatisticsOperator
from src.v3.operators.charge_dynamics import ChargeDynamicsOperator
from src.v3.operators.phi_cascade import PhiCascadeOperator
from src.v3.operators.fusion import FusionOperator
from src.v3.operators.sec_tracking import SECTrackingOperator
from src.v3.analyzers import (
    ConservationAnalyzer, GravityAnalyzer, AtomDetector,
    StarDetector, QuantumDetector, GalaxyAnalyzer,
)
from src.v3.emergence import HerniationDetector, StructureAnalyzer


def build_pipeline():
    return Pipeline([
        RBFOperator(),
        QBEOperator(),
        ActualizationOperator(),  # replaces EulerIntegrator — MAR-gated integration
        MemoryOperator(),
        PhiCascadeOperator(),
        GravitationalCollapseOperator(),
        SpinStatisticsOperator(),
        ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(),
        TemperatureOperator(),
        ThermalNoiseOperator(),
        NormalizationOperator(),
        SECTrackingOperator(),
        AdaptiveOperator(),
        TimeEmergenceOperator(),
    ])


def run_cli(ticks: int, nu: int, nv: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Reality Engine v3 — {nu}x{nv} Mobius manifold on {device}")
    print(f"Running {ticks} ticks...\n")

    config = SimulationConfig(nu=nu, nv=nv, dt=0.001, device=device)
    engine = Engine(config=config, pipeline=build_pipeline())
    engine.initialize("big_bang", temperature=2.0)
    pac_initial = engine.state.pac_total

    analyzers = [
        ConservationAnalyzer(),
        GravityAnalyzer(mass_threshold=2.0, min_curvature=0.5),
        AtomDetector(mass_threshold=1.5, gradient_threshold=0.3),
        StarDetector(mass_threshold=2.5, temp_threshold=2.0),
        QuantumDetector(coherence_threshold=0.8),
        GalaxyAnalyzer(mass_threshold=1.0, min_region_fraction=0.05, min_gravity_wells=3),
        HerniationDetector(threshold=1.0),
    ]
    structure_tracker = StructureAnalyzer()

    t0 = time.perf_counter()
    report_every = max(1, ticks // 20)

    # Capture true counts from bus events (analyzers report top-N but emit true totals)
    true_counts = {}

    def _capture_count(event_name, key):
        def handler(data):
            true_counts[key] = data.get("count", data.get("atoms", 0))
            if "hydrogen" in data:
                true_counts["hydrogen"] = data["hydrogen"]
        engine.bus.subscribe(event_name, handler)

    _capture_count("gravity_well_detected", "gravity_well")
    _capture_count("star_detected", "star")
    _capture_count("atom_detected", "atom")
    _capture_count("herniation_detected", "herniation")

    # Track Landauer reinjection events
    landauer_total = [0.0]
    def _on_landauer(data):
        landauer_total[0] += data.get("energy_reinjected", 0.0)
    engine.bus.subscribe("landauer_reinjection", _on_landauer)

    for i in range(ticks):
        state = engine.tick()

        if (i + 1) % report_every == 0 or i == 0:
            true_counts.clear()

            # Run analyzers in causal chain order
            all_dets = []
            for a in analyzers:
                all_dets.extend(a.analyze(state, engine.bus, prior_detections=all_dets))
            structure_tracker.update(all_dets, state.tick)

            gw = true_counts.get("gravity_well", 0)
            st = true_counts.get("star", 0)
            at = true_counts.get("atom", 0)
            hy = true_counts.get("hydrogen", 0)
            he = true_counts.get("herniation", 0)

            # Spatial structure metrics
            n_cells = state.M.numel()
            void_pct = (state.M < 0.5).sum().item() / n_cells * 100
            dense_pct = (state.M > 2.0).sum().item() / n_cells * 100
            m_max = state.M.max().item()

            # PAC conservation tracking
            pac_now = state.pac_total
            pac_drift = pac_now - pac_initial

            elapsed = time.perf_counter() - t0
            tps = (i + 1) / elapsed

            reinj = state.metrics.get("landauer_reinjection", 0.0)
            reinj_str = f" reinj={reinj:>6.1f}" if reinj > 0.01 else ""

            print(
                f"  tick {state.tick:>6d} | "
                f"PAC={pac_now:>9.1f} d={pac_drift:>+8.1f} | "
                f"M={state.M.sum().item():>8.1f} Mmax={m_max:>5.2f} | "
                f"Z={state.Z.sum().item():>7.1f} | "
                f"T={state.T.mean().item():>5.2f} | "
                f"void={void_pct:>4.0f}% dense={dense_pct:>4.0f}% | "
                f"wells={gw:>3d} stars={st:>3d} atoms={at:>3d} H={hy:>3d} | "
                f"stable={structure_tracker.stable_count:>3d}{reinj_str} | "
                f"{tps:.0f} t/s"
            )

    elapsed = time.perf_counter() - t0
    pac_final = engine.state.pac_total
    print(f"\nDone. {ticks} ticks in {elapsed:.2f}s ({ticks/elapsed:.0f} ticks/sec)")
    print(f"Final: PAC={pac_final:.1f} (drift={pac_final - pac_initial:+.1f}), "
          f"mass={engine.state.M.sum().item():.2f}, "
          f"stable_structures={structure_tracker.stable_count}")
    print(f"Landauer reinjection total: {landauer_total[0]:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Reality Engine v3")
    parser.add_argument("--ticks", type=int, default=1000, help="Number of ticks to run")
    parser.add_argument("--nu", type=int, default=128, help="Grid circumference")
    parser.add_argument("--nv", type=int, default=32, help="Grid width")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard UI")
    args = parser.parse_args()

    if args.dashboard:
        import uvicorn
        from src.v3.dashboard.server import create_app
        config = SimulationConfig(nu=args.nu, nv=args.nv, device=torch.device("cpu"))
        app = create_app(config)
        print(f"Reality Engine v3 Dashboard: http://localhost:8050")
        uvicorn.run(app, host="0.0.0.0", port=8050, log_level="info")
    else:
        run_cli(args.ticks, args.nu, args.nv)


if __name__ == "__main__":
    main()
