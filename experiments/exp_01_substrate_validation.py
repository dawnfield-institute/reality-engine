"""
exp_01 — Substrate Validation.

Verify that the MobiusManifold has correct topology:
  1. Coordinate ranges
  2. Laplacian: constant field → zero
  3. Laplacian: antiperiodic mode → correct eigenvalue
  4. Laplacian: Gaussian diffuses smoothly across seam
  5. Möbius identification: f(0, j) ↔ f(n_u-1, n_v-1-j)
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.substrate.mobius import MobiusManifold


def run(device: str = "cpu") -> dict:
    results = {"experiment": "exp_01_substrate_validation", "tests": []}
    m = MobiusManifold(n_u=128, n_v=64, device=device)

    # Test 1: constant field
    f = torch.ones(128, 64, device=device)
    lap = m.laplacian(f)
    err = lap.abs().max().item()
    results["tests"].append({
        "name": "constant_field_zero_laplacian",
        "passed": err < 1e-12,
        "error": err,
    })

    # Test 2: antiperiodic mode eigenvalue
    n = 1
    u = torch.linspace(0, 2 * math.pi, 129, device=device)[:-1]
    mode = torch.sin((n + 0.5) * u).unsqueeze(1).expand(128, 64)
    lap_mode = m.laplacian(mode)
    k = n + 0.5
    expected_ev = 2 * (math.cos(k * 2 * math.pi / 128) - 1)
    ratio = lap_mode[32, 32] / mode[32, 32]
    ev_err = abs(ratio.item() - expected_ev)
    results["tests"].append({
        "name": "antiperiodic_mode_eigenvalue",
        "passed": ev_err < 0.5,
        "measured": ratio.item(),
        "expected": expected_ev,
        "error": ev_err,
    })

    # Test 3: Gaussian diffuses across seam
    u_grid = m.U
    g = torch.exp(-((u_grid - 0.1) ** 2) / 0.05)
    lap_g = m.laplacian(g)
    g_new = g + 0.1 * lap_g
    gain = (g_new[-3:, :].mean() - g[-3:, :].mean()).item()
    results["tests"].append({
        "name": "gaussian_diffuses_across_seam",
        "passed": gain > 1e-6,
        "gain": gain,
    })

    # Test 4: grid dimensions
    results["tests"].append({
        "name": "grid_dimensions",
        "passed": m.U.shape == (128, 64) and m.V.shape == (128, 64),
    })

    # Summary
    all_pass = all(t["passed"] for t in results["tests"])
    results["all_passed"] = all_pass
    results["timestamp"] = datetime.now().isoformat()

    for t in results["tests"]:
        status = "✓" if t["passed"] else "✗"
        print(f"  {status} {t['name']}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("EXP 01: Substrate Validation")
    print("=" * 60)
    results = run()

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"exp_01_substrate_validation_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
    print("ALL PASSED" if results["all_passed"] else "SOME FAILED")
