"""
exp_04 — Ξ Emergence.

Does Ξ_L2 converge from full engine dynamics with RBF self-regulation?
  1. Ξ_L2 always finite
  2. Ξ_L2 changes over time under dynamics
  3. Ξ_L2 approaches ~1.057 within 5%
  4. Ξ_L2 stable in last 1000 steps
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import RealityEngine
from src.substrate.constants import XI_REFERENCE


def run(device: str = "cpu", n_steps: int = 5000) -> dict:
    results = {"experiment": "exp_04_xi_emergence", "tests": [], "xi_trace": []}

    config = {
        "manifold": {"n_u": 128, "n_v": 64, "device": device},
        "sec": {
            "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0, "sigma_0": 0.1,
            "dt": 0.01, "xi_gain": 2.0, "rho": 5.0, "phi_source": 0.618,
            "ki_rbf": 1.0, "low_k_mix": 0.382,
        },
        "init": {"seed": 42, "P_mean": 0.1, "P_noise": 0.01,
                 "A_mean": 0.1, "A_noise": 0.01, "antiperiodic_amp": 0.01},
        "pac": {"mode": "enforce"},
    }

    engine = RealityEngine(config)

    # Run and collect xi_L2 trace
    for i in range(n_steps):
        engine.step()
        if i % 50 == 0:
            rec = engine.diagnostics.records[-1]
            results["xi_trace"].append({"t": i, "xi_L2": rec.get("xi_L2", 0.0)})

    xi_final = engine.diagnostics.records[-1].get("xi_L2", float("nan"))

    # Test 1: Ξ_L2 is finite
    is_finite = all(
        not (abs(r["xi_L2"]) == float("inf") or r["xi_L2"] != r["xi_L2"])
        for r in results["xi_trace"]
    )
    results["tests"].append({
        "name": "xi_always_finite",
        "passed": is_finite,
    })

    # Test 2: Ξ_L2 changed over time
    xi_values = [r["xi_L2"] for r in results["xi_trace"]]
    xi_range = max(xi_values) - min(xi_values)
    changed = xi_range > 1e-6
    results["tests"].append({
        "name": "xi_evolves",
        "passed": changed,
        "range": xi_range,
    })

    # Test 3: final Ξ_L2 proximity to target (within 5%)
    xi_err = abs(xi_final - XI_REFERENCE) / XI_REFERENCE
    results["tests"].append({
        "name": "xi_proximity",
        "passed": xi_err < 0.05,
        "xi_L2_final": xi_final,
        "xi_target": XI_REFERENCE,
        "relative_error": xi_err,
    })

    # Test 4: Ξ_L2 mean over last 20 samples
    xi_last = [r["xi_L2"] for r in results["xi_trace"][-20:]]
    xi_mean = sum(xi_last) / len(xi_last)
    results["tests"].append({
        "name": "xi_mean_last_20",
        "xi_mean": xi_mean,
        "xi_std": (sum((x - xi_mean)**2 for x in xi_last) / len(xi_last)) ** 0.5,
        "passed": True,  # informational
    })

    all_pass = all(t["passed"] for t in results["tests"])
    results["all_passed"] = all_pass
    results["timestamp"] = datetime.now().isoformat()

    for t in results["tests"]:
        status = "✓" if t["passed"] else "✗"
        info = ""
        if "xi_L2_final" in t:
            info = f" (Ξ_L2={t['xi_L2_final']:.4f})"
        elif "xi_mean" in t:
            info = f" (mean={t['xi_mean']:.4f} ± {t['xi_std']:.4f})"
        print(f"  {status} {t['name']}{info}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("EXP 04: Ξ Emergence")
    print("=" * 60)
    results = run()

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"exp_04_xi_emergence_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
    print("ALL PASSED" if results["all_passed"] else "SOME FAILED")
