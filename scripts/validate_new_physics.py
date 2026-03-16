#!/usr/bin/env python3
"""Validate new physics operators: SpinStatistics, ChargeDynamics, emergent mass limits.

Compares the FULL PIPELINE (with new operators) against the old pipeline (without)
to show that spin-statistics feedback, charge forces, and emergent mass caps
produce richer, more physically realistic matter.

Key questions:
1. Does degeneracy pressure prevent universal collapse to the mass cap?
2. Do we see more mass diversity (more peaks, wider range)?
3. Does charge create bound structures (molecular binding)?
4. Do half-integer spin states actually resist co-location?
5. Does the 1/φ² mass spacing persist or sharpen?
"""

import sys
import os
import time
import math
import json
from datetime import datetime

import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
from src.v3.engine.state import FieldState
from src.v3.operators.rbf import RBFOperator
from src.v3.operators.qbe import QBEOperator
from src.v3.operators.memory import MemoryOperator
from src.v3.operators.actualization import ActualizationOperator
from src.v3.operators.gravity import GravitationalCollapseOperator
from src.v3.operators.confluence import ConfluenceOperator
from src.v3.operators.temperature import TemperatureOperator
from src.v3.operators.thermal_noise import ThermalNoiseOperator
from src.v3.operators.normalization import NormalizationOperator
from src.v3.operators.adaptive import AdaptiveOperator
from src.v3.operators.time_emergence import TimeEmergenceOperator
from src.v3.operators.fusion import FusionOperator
from src.v3.operators.spin_statistics import SpinStatisticsOperator
from src.v3.operators.charge_dynamics import ChargeDynamicsOperator
from src.v3.operators.protocol import Pipeline


PHI = (1 + math.sqrt(5)) / 2
INV_PHI2 = 1.0 / (PHI * PHI)  # 0.3820


def build_pipeline(include_new_physics: bool = True) -> Pipeline:
    """Build the operator pipeline matching the stable analyze_matter_v2 ordering.

    Critical: Actualization runs BEFORE memory/gravity/normalization.
    Normalization must be the last field-modifying operator to keep bounds.
    """
    ops = [
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), GravitationalCollapseOperator(),
    ]
    if include_new_physics:
        ops.extend([SpinStatisticsOperator(), ChargeDynamicsOperator()])
    ops.extend([
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), AdaptiveOperator(), TimeEmergenceOperator(),
    ])
    return Pipeline(ops)


def find_local_maxima(M: torch.Tensor, min_mass: float = 0.1):
    """Find local maxima in the mass field."""
    M_np = M.cpu().numpy()
    import numpy as np
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(M_np, size=3)
    peaks = (M_np == local_max) & (M_np > min_mass)
    coords = np.argwhere(peaks)
    masses = M_np[peaks]
    return coords, masses


def find_mass_peaks(masses, n_bins=50):
    """Find peaks in mass histogram."""
    import numpy as np
    if len(masses) < 10:
        return [], 0.0
    hist, bin_edges = np.histogram(masses, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Find peaks: bins higher than both neighbors
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 3:
            peaks.append((bin_centers[i], hist[i]))
    if len(peaks) >= 2:
        spacings = [peaks[i+1][0] - peaks[i][0] for i in range(len(peaks)-1)]
        mean_spacing = sum(spacings) / len(spacings)
    else:
        mean_spacing = 0.0
    return peaks, mean_spacing


def analyze_snapshot(state, label, include_new=True):
    """Analyze a field state snapshot."""
    import numpy as np
    M = state.M
    E, I = state.E, state.I

    coords, masses = find_local_maxima(M, min_mass=0.1)
    peaks, spacing = find_mass_peaks(masses)

    # Mass distribution stats
    n_structures = len(masses)
    mass_mean = float(np.mean(masses)) if n_structures > 0 else 0
    mass_std = float(np.std(masses)) if n_structures > 0 else 0
    mass_max = float(np.max(masses)) if n_structures > 0 else 0
    mass_min = float(np.min(masses)) if n_structures > 0 else 0

    # Fraction at "cap" (> 90% of max mass)
    if n_structures > 0:
        cap_threshold = mass_max * 0.95
        at_cap = float(np.sum(masses > cap_threshold)) / n_structures
    else:
        at_cap = 0.0

    # Spin statistics
    diseq = (E - I).cpu().numpy()
    d_du = (np.roll(diseq, -1, axis=0) - np.roll(diseq, 1, axis=0)) / 2.0
    d_dv = (np.roll(diseq, -1, axis=1) - np.roll(diseq, 1, axis=1)) / 2.0
    S = d_du - d_dv

    # Half-integer fraction at structure sites
    if n_structures > 0:
        S_at_peaks = S[coords[:, 0], coords[:, 1]]
        f_half = np.cos(np.pi * S_at_peaks) ** 2
        half_int_frac = float(np.mean(f_half > 0.5))
    else:
        half_int_frac = 0.0

    # Charge at structure sites
    E_np = E.cpu().numpy()
    I_np = I.cpu().numpy()
    dE_du = (np.roll(E_np, -1, axis=0) - np.roll(E_np, 1, axis=0)) / 2.0
    dI_dv = (np.roll(I_np, -1, axis=1) - np.roll(I_np, 1, axis=1)) / 2.0
    Q = dE_du - dI_dv

    if n_structures > 0:
        Q_at_peaks = Q[coords[:, 0], coords[:, 1]]
        charge_balance = abs(float(np.mean(Q_at_peaks)))
        charge_std = float(np.std(Q_at_peaks))
    else:
        charge_balance = 0.0
        charge_std = 0.0

    # Binding energy proxy: M - |E-I| at structure sites
    if n_structures > 0:
        M_np = M.cpu().numpy()
        binding = M_np[coords[:, 0], coords[:, 1]] - abs(diseq[coords[:, 0], coords[:, 1]])
        n_bound = float(np.sum(binding > 0)) / n_structures
    else:
        n_bound = 0.0

    # PAC conservation
    pac = (E + I + M).sum().item()

    # Metrics from operators
    metrics = state.metrics

    result = {
        "label": label,
        "tick": state.tick,
        "n_structures": n_structures,
        "n_peaks": len(peaks),
        "peak_masses": [round(p[0], 3) for p in peaks],
        "peak_spacing": round(spacing, 4),
        "spacing_vs_inv_phi2": round(abs(spacing - INV_PHI2) / INV_PHI2 * 100, 1) if spacing > 0 else None,
        "mass_mean": round(mass_mean, 3),
        "mass_std": round(mass_std, 3),
        "mass_range": [round(mass_min, 3), round(mass_max, 3)],
        "fraction_at_cap": round(at_cap, 3),
        "half_integer_fraction": round(half_int_frac, 3),
        "charge_balance": round(charge_balance, 4),
        "charge_std": round(charge_std, 4),
        "fraction_bound": round(n_bound, 3),
        "pac_total": pac,
    }

    # Add operator metrics if available
    for key in ["degeneracy_pressure_mean", "charge_force_mean", "e_local_mean",
                 "spin_half_integer_fraction", "state_similarity_mean",
                 "f_local_mean", "alpha_local_mean", "G_local_mean", "gamma_local_mean"]:
        if key in metrics:
            result[key] = round(metrics[key], 4)

    return result


def print_comparison(old_data, new_data, tick):
    """Print side-by-side comparison."""
    print(f"\n{'='*78}")
    print(f"  Tick {tick} — Comparison")
    print(f"{'='*78}")
    print(f"  {'Metric':<30} | {'WITHOUT new physics':>20} | {'WITH new physics':>20}")
    print(f"  {'-'*30}-+-{'-'*20}-+-{'-'*20}")

    compare_keys = [
        ("n_structures", "Structures found"),
        ("n_peaks", "Mass peaks"),
        ("mass_mean", "Mass mean"),
        ("mass_std", "Mass std (diversity)"),
        ("mass_range", "Mass range"),
        ("fraction_at_cap", "Fraction at cap"),
        ("peak_spacing", "Peak spacing"),
        ("spacing_vs_inv_phi2", "Spacing err vs 1/φ² (%)"),
        ("half_integer_fraction", "Half-integer spin frac"),
        ("charge_balance", "Charge balance (→0)"),
        ("charge_std", "Charge spread"),
        ("fraction_bound", "Fraction bound"),
    ]

    for key, name in compare_keys:
        old_val = old_data.get(key, "N/A")
        new_val = new_data.get(key, "N/A")
        if isinstance(old_val, list):
            old_str = f"[{old_val[0]:.2f}, {old_val[1]:.2f}]"
            new_str = f"[{new_val[0]:.2f}, {new_val[1]:.2f}]"
        elif isinstance(old_val, float):
            old_str = f"{old_val:.4f}"
            new_str = f"{new_val:.4f}" if isinstance(new_val, float) else str(new_val)
        else:
            old_str = str(old_val)
            new_str = str(new_val)

        # Highlight improvements
        marker = ""
        if key == "fraction_at_cap" and isinstance(old_val, float) and isinstance(new_val, float):
            if new_val < old_val:
                marker = " <<"
        elif key == "n_peaks" and isinstance(old_val, int) and isinstance(new_val, int):
            if new_val > old_val:
                marker = " <<"
        elif key == "mass_std" and isinstance(old_val, float) and isinstance(new_val, float):
            if new_val > old_val:
                marker = " <<"

        print(f"  {name:<30} | {old_str:>20} | {new_str:>20}{marker}")

    # New physics metrics (only in new pipeline)
    if "degeneracy_pressure_mean" in new_data:
        print(f"\n  New physics metrics:")
        print(f"    Degeneracy pressure:  {new_data.get('degeneracy_pressure_mean', 0):.6f}")
        print(f"    Charge force:         {new_data.get('charge_force_mean', 0):.6f}")
        print(f"    State similarity:     {new_data.get('state_similarity_mean', 0):.4f}")


def main():
    print("=" * 78)
    print("  Reality Engine v3 — New Physics Validation")
    print("  SpinStatistics + ChargeDynamics + Emergent Mass Limits")
    print("=" * 78)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SimulationConfig(nu=128, nv=64, device=device)
    print(f"\n  Grid: {config.nu}x{config.nv}  Device: {device}")

    # Use same initial conditions for fair comparison
    torch.manual_seed(7)  # deterministic seed that produces stable runs
    state_template = FieldState.big_bang(config.nu, config.nv, device=device, temperature=3.0)

    snapshots = [500, 1000, 2000, 5000, 10000]

    results = {"old": [], "new": []}

    for label, include_new in [("WITHOUT", False), ("WITH", True)]:
        print(f"\n{'='*78}")
        print(f"  Running {label} new physics operators")
        print(f"{'='*78}")

        pipeline = build_pipeline(include_new_physics=include_new)
        engine = Engine(config=config, pipeline=pipeline)
        engine.state = state_template.replace()  # fresh copy

        t0 = time.time()
        snap_idx = 0

        for tick in range(1, max(snapshots) + 1):
            engine.tick()

            if tick == snapshots[snap_idx]:
                elapsed = time.time() - t0
                data = analyze_snapshot(engine.state, label, include_new)
                key = "new" if include_new else "old"
                results[key].append(data)

                print(f"\n  Tick {tick} ({elapsed:.0f}s):")
                print(f"    Structures: {data['n_structures']}")
                print(f"    Mass peaks: {data['n_peaks']} — {data['peak_masses']}")
                print(f"    Mass range: {data['mass_range']}")
                print(f"    At cap: {data['fraction_at_cap']*100:.1f}%")
                print(f"    Half-int spin: {data['half_integer_fraction']*100:.1f}%")
                print(f"    Bound: {data['fraction_bound']*100:.1f}%")
                print(f"    PAC: {data['pac_total']:.4f}")
                if "degeneracy_pressure_mean" in data:
                    print(f"    Degen pressure: {data['degeneracy_pressure_mean']:.6f}")
                    print(f"    Charge force: {data['charge_force_mean']:.6f}")

                snap_idx += 1
                if snap_idx >= len(snapshots):
                    break

    # Side-by-side comparison
    print("\n" + "=" * 78)
    print("  HEAD-TO-HEAD COMPARISON")
    print("=" * 78)

    for i, tick in enumerate(snapshots):
        if i < len(results["old"]) and i < len(results["new"]):
            print_comparison(results["old"][i], results["new"][i], tick)

    # Summary verdict
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)

    final_old = results["old"][-1] if results["old"] else {}
    final_new = results["new"][-1] if results["new"] else {}

    checks = []

    # 1. Less cap pileup?
    old_cap = final_old.get("fraction_at_cap", 1.0)
    new_cap = final_new.get("fraction_at_cap", 1.0)
    cap_improved = new_cap < old_cap
    checks.append(("Mass cap reduction", cap_improved,
                    f"{old_cap*100:.1f}% -> {new_cap*100:.1f}%"))

    # 2. More mass diversity?
    old_std = final_old.get("mass_std", 0)
    new_std = final_new.get("mass_std", 0)
    diversity_improved = new_std > old_std
    checks.append(("Mass diversity (std)", diversity_improved,
                    f"{old_std:.3f} -> {new_std:.3f}"))

    # 3. More mass peaks?
    old_peaks = final_old.get("n_peaks", 0)
    new_peaks = final_new.get("n_peaks", 0)
    peaks_improved = new_peaks >= old_peaks
    checks.append(("Mass peak count", peaks_improved,
                    f"{old_peaks} -> {new_peaks}"))

    # 4. PAC conserved?
    if results["new"]:
        pac_values = [r["pac_total"] for r in results["new"]]
        pac_drift = abs(pac_values[-1] - pac_values[0]) / (abs(pac_values[0]) + 1e-10)
        pac_ok = pac_drift < 1e-4
    else:
        pac_ok = False
        pac_drift = 1.0
    checks.append(("PAC conservation", pac_ok, f"drift = {pac_drift:.2e}"))

    # 5. Charge balance?
    charge_bal = final_new.get("charge_balance", 1.0)
    charge_ok = charge_bal < 0.5
    checks.append(("Charge balance", charge_ok, f"|mean Q| = {charge_bal:.4f}"))

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    n_pass = sum(1 for _, p, _ in checks if p)
    print(f"\n  Score: {n_pass}/{len(checks)}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"new_physics_validation_{ts}.json")

    # Convert for JSON
    save_data = {"old": results["old"], "new": results["new"], "checks": [
        {"name": n, "passed": p, "detail": d} for n, p, d in checks
    ]}
    with open(outpath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {outpath}")


if __name__ == "__main__":
    main()
