"""
Entry point: ``python -m src --config configs/default.yaml``

Supports both headless (no vis) and real-time (pygame) modes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import RealityEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="reality-engine-v2",
        description="Reality Engine v2 — PAC/SEC on a Möbius manifold",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override n_steps from config",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable real-time visualisation",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save diagnostics JSON to this path",
    )
    args = parser.parse_args()

    engine = RealityEngine.from_yaml(args.config)

    # Determine step count
    import yaml
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    n_steps = args.steps or cfg.get("run", {}).get("n_steps", 10000)
    log_every = cfg.get("run", {}).get("log_interval", 100)

    vis_enabled = cfg.get("run", {}).get("vis", True) and not args.no_vis

    if vis_enabled:
        try:
            from .vis.realtime import RealtimeRenderer
            fps = cfg.get("run", {}).get("vis_fps", 30)
            renderer = RealtimeRenderer(fps=fps)
        except RuntimeError:
            print("pygame not available — running headless")
            vis_enabled = False

    print(f"Reality Engine v2 — {n_steps} steps on "
          f"{engine.manifold.n_u}×{engine.manifold.n_v} Möbius manifold")
    print(f"Device: {engine.manifold.device}")
    print()

    for i in range(n_steps):
        diag = engine.step()

        if vis_enabled:
            if not renderer.alive():
                print("\nWindow closed — stopping.")
                break
            renderer.render(engine.state, diag)

        if i % log_every == 0:
            print(
                f"t={diag['t']:6d} | "
                f"PAC={diag['residual']:.2e} | "
                f"Ξ={diag['xi_spectral']:.4f} | "
                f"P={diag['P_mean']:.4f} | "
                f"A={diag['A_mean']:.4f}"
            )

        if engine.diagnostics.diverged:
            print(f"\n⚠ DIVERGENCE at step {diag['t']}.")
            break

    if vis_enabled:
        renderer.close()

    summary = engine.diagnostics.summary()
    print(f"\n{'=' * 60}")
    print(f"Steps: {summary.get('n_steps', 0)}")
    print(f"Diverged: {summary.get('diverged', False)}")
    print(f"PAC max residual: {summary.get('pac_residual_max', 0):.2e}")
    print(f"Ξ final: {summary.get('xi_final', 0):.4f}")
    print(f"Wall time: {summary.get('wall_seconds', 0):.1f}s")

    if args.save:
        engine.diagnostics.save(args.save)
        print(f"Diagnostics saved to {args.save}")


if __name__ == "__main__":
    main()
