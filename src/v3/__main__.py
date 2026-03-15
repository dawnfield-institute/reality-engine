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
from src.v3.operators.memory import MemoryOperator
from src.v3.operators.confluence import ConfluenceOperator
from src.v3.operators.temperature import TemperatureOperator
from src.v3.operators.thermal_noise import ThermalNoiseOperator
from src.v3.operators.normalization import NormalizationOperator
from src.v3.operators.adaptive import AdaptiveOperator
from src.v3.operators.time_emergence import TimeEmergenceOperator
from src.v3.operators.gravity import GravitationalCollapseOperator
from src.v3.operators.fusion import FusionOperator
from src.v3.analyzers import (
    ConservationAnalyzer, GravityAnalyzer, AtomDetector,
    StarDetector, QuantumDetector, GalaxyAnalyzer,
)
from src.v3.emergence import HerniationDetector, StructureAnalyzer


def build_pipeline():
    return Pipeline([
        RBFOperator(),
        QBEOperator(),
        EulerIntegrator(),
        MemoryOperator(),
        GravitationalCollapseOperator(),
        FusionOperator(),
        ConfluenceOperator(),
        TemperatureOperator(),
        ThermalNoiseOperator(),
        NormalizationOperator(),
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

    analyzers = [
        ConservationAnalyzer(),
        GravityAnalyzer(mass_threshold=0.3, min_curvature=0.01),
        AtomDetector(mass_threshold=0.5, gradient_threshold=0.1),
        StarDetector(mass_threshold=2.0, temp_threshold=2.0),
        QuantumDetector(coherence_threshold=0.8),
        GalaxyAnalyzer(mass_threshold=0.2, min_region_fraction=0.02),
        HerniationDetector(threshold=0.5),
    ]
    structure_tracker = StructureAnalyzer()

    t0 = time.perf_counter()
    report_every = max(1, ticks // 20)

    for i in range(ticks):
        state = engine.tick()

        if (i + 1) % report_every == 0 or i == 0:
            # Run analyzers in causal chain order
            all_dets = []
            for a in analyzers:
                all_dets.extend(a.analyze(state, engine.bus, prior_detections=all_dets))
            structure_tracker.update(all_dets, state.tick)

            # Detection summary
            det_counts = {}
            for d in all_dets:
                det_counts[d.kind] = det_counts.get(d.kind, 0) + 1
            det_str = ", ".join(f"{k}:{v}" for k, v in det_counts.items()) if det_counts else "none"

            elapsed = time.perf_counter() - t0
            tps = (i + 1) / elapsed

            print(
                f"  tick {state.tick:>6d} | "
                f"E={state.total_energy:>10.1f} | "
                f"M={state.M.sum().item():>8.2f} | "
                f"Z={state.Z.sum().item():>6.2f} | "
                f"T={state.T.mean().item():>5.2f} | "
                f"PAC={state.pac_total:>10.1f} | "
                f"stable={structure_tracker.stable_count} | "
                f"det=[{det_str}] | "
                f"{tps:.0f} t/s"
            )

    elapsed = time.perf_counter() - t0
    print(f"\nDone. {ticks} ticks in {elapsed:.2f}s ({ticks/elapsed:.0f} ticks/sec)")
    print(f"Final: energy={engine.state.total_energy:.1f}, mass={engine.state.M.sum().item():.2f}, "
          f"stable_structures={structure_tracker.stable_count}")


def main():
    parser = argparse.ArgumentParser(description="Reality Engine v3")
    parser.add_argument("--ticks", type=int, default=1000, help="Number of ticks to run")
    parser.add_argument("--nu", type=int, default=64, help="Grid circumference")
    parser.add_argument("--nv", type=int, default=16, help="Grid width")
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
