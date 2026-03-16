"""Physics Scorecard -- compare Reality Engine emergent metrics to known physics.

Runs a 10K-tick simulation and scores emergent coupling constants, structural
physics, and mass spectrum properties against DFT-predicted and experimentally
measured values.

Tiers:
  1. Emergent coupling attractors (direct metric -> DFT constant)
  2. Structural physics (mass peaks, PAC conservation, entropy)
  3. Deep DFT constants (fine structure, Koide, mass ratios -- aspirational)

All target values from PACSeries v0.2 and exp_43 results.
"""

import math
import os
import sys
import time

re_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if re_path not in sys.path:
    sys.path.insert(0, re_path)

import torch

from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
from src.v3.operators.protocol import Pipeline
from src.v3.operators.rbf import RBFOperator
from src.v3.operators.qbe import QBEOperator
from src.v3.operators.memory import MemoryOperator
from src.v3.operators.confluence import ConfluenceOperator
from src.v3.operators.temperature import TemperatureOperator
from src.v3.operators.thermal_noise import ThermalNoiseOperator
from src.v3.operators.normalization import NormalizationOperator
from src.v3.operators.adaptive import AdaptiveOperator
from src.v3.operators.time_emergence import TimeEmergenceOperator
from src.v3.operators.gravity import GravitationalCollapseOperator
from src.v3.operators.fusion import FusionOperator
from src.v3.operators.actualization import ActualizationOperator
from src.v3.operators.spin_statistics import SpinStatisticsOperator
from src.v3.operators.charge_dynamics import ChargeDynamicsOperator
from src.v3.operators.phi_cascade import PhiCascadeOperator
from src.v3.operators.sec_tracking import SECTrackingOperator

# ============================================================
# DFT Constants
# ============================================================
PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
LN2 = math.log(2)
GAMMA_EM = 0.5772156649015328  # Euler-Mascheroni
PHI_INV = 1.0 / PHI
PHI_INV2 = 1.0 / PHI ** 2

# Fibonacci numbers
F2, F3, F4, F5, F6, F7, F9, F10, F11, F12 = 1, 2, 3, 5, 8, 13, 34, 55, 89, 144

# Fine structure constant: F3/(F4*phi*F10) * (1 - F10/(4*pi*F7^2))
ALPHA_EM = (F3 / (F4 * PHI * F10)) * (1 - F10 / (4 * math.pi * F7**2))
# Weinberg angle
SIN2_THETA_W = F4 / F7  # 3/13
# Koide Q
KOIDE_Q = F3 / (F3 + F2)  # 2/3
# Muon/electron mass ratio
MU_E_RATIO = F4 * F6**2 * (1 + 1/F7)  # 206.769...
# Proton/electron mass ratio
P_E_RATIO = F4 * F9 * F12 / F6  # 1836


# ============================================================
# Target Table
# ============================================================
def build_targets():
    """Build the target table: list of dicts with scoring info."""
    targets = []

    # --- Tier 1: Emergent coupling attractors ---
    targets.append({
        "name": "f_local -> gamma_EM",
        "tier": 1,
        "metric_key": "f_local_mean",
        "target": GAMMA_EM,
        "target_label": "0.5772",
        "description": "actualization ratio attractor",
    })
    targets.append({
        "name": "gamma_local -> 1/phi",
        "tier": 1,
        "metric_key": "gamma_local_mean",
        "target": PHI_INV,
        "target_label": "0.6180",
        "description": "mass generation coefficient",
    })
    targets.append({
        "name": "alpha_local -> ln(2)",
        "tier": 1,
        "metric_key": "alpha_local_mean",
        "target": LN2,
        "target_label": "0.6931",
        "description": "RBF collapse attraction",
    })
    targets.append({
        "name": "G_local -> 1/phi^2",
        "tier": 1,
        "metric_key": "G_local_mean",
        "target": PHI_INV2,
        "target_label": "0.3820",
        "description": "gravitational coupling",
    })
    targets.append({
        "name": "lambda_local -> 1-ln(2)",
        "tier": 1,
        "metric_key": "lambda_local_mean",
        "target": 1 - LN2,
        "target_label": "0.3069",
        "description": "RBF memory coupling",
    })
    targets.append({
        "name": "f_deviation -> 0",
        "tier": 1,
        "metric_key": "f_local_deviation",
        "target": 0.0,
        "target_label": "0 (ln_phi)",
        "description": "distance from ln(phi)",
        "absolute": True,  # score by |measured| not |measured-target|/|target|
    })

    # --- Tier 2: Structural physics ---
    targets.append({
        "name": "phi^2 mass spacing",
        "tier": 2,
        "metric_key": "_phi2_spacing_err",
        "target": 0.0,
        "target_label": "0% error",
        "description": "mass peak spacing vs 1/phi^2",
        "is_error_pct": True,  # measured is already an error percentage
    })
    targets.append({
        "name": "PAC conservation",
        "tier": 2,
        "metric_key": "_pac_drift",
        "target": 0.0,
        "target_label": "0 drift",
        "description": "E+I+M conserved",
        "absolute": True,
    })
    targets.append({
        "name": "spin half-integer",
        "tier": 2,
        "metric_key": "spin_half_integer_fraction",
        "target": 0.5,
        "target_label": "0.50",
        "description": "fraction of half-integer spin sites",
    })
    targets.append({
        "name": "entropy reduction",
        "tier": 2,
        "metric_key": "entropy_reduction_rate",
        "target": None,  # just track, positive = good
        "target_label": "> 0",
        "description": "SEC entropy decrease",
        "positive_good": True,
    })

    # --- Tier 3: Deep DFT constants (aspirational) ---
    targets.append({
        "name": "alpha (fine structure)",
        "tier": 3,
        "metric_key": "_alpha_em",
        "target": ALPHA_EM,
        "target_label": f"1/{1/ALPHA_EM:.1f}",
        "description": "EM coupling from e_local x charge energy ratio",
    })
    targets.append({
        "name": "Koide Q",
        "tier": 3,
        "metric_key": "_koide_q",
        "target": KOIDE_Q,
        "target_label": "2/3",
        "description": "mass ratio formula for 3 lightest peaks",
    })
    targets.append({
        "name": "mu/e mass ratio",
        "tier": 3,
        "metric_key": "_mu_e_ratio",
        "target": MU_E_RATIO,
        "target_label": "206.8",
        "description": "2nd/1st mass peak ratio",
    })

    return targets


# ============================================================
# Mass Peak Analysis (from validate_unified_force.py)
# ============================================================
def find_mass_peaks(M, min_mass=0.1, n_bins=40):
    is_max = (
        (M > torch.roll(M, 1, 0)) &
        (M > torch.roll(M, -1, 0)) &
        (M > torch.roll(M, 1, 1)) &
        (M > torch.roll(M, -1, 1)) &
        (M > min_mass)
    )
    masses = M[is_max].cpu().tolist()
    if len(masses) < 10:
        return [], masses
    m_min, m_max = min(masses), max(masses)
    if m_max - m_min < 0.05:
        return [(m_min, len(masses))], masses
    bin_width = (m_max - m_min) / n_bins
    bins = [0] * n_bins
    for m in masses:
        idx = min(int((m - m_min) / bin_width), n_bins - 1)
        bins[idx] += 1
    peaks = []
    for i in range(1, n_bins - 1):
        if bins[i] > bins[i-1] and bins[i] > bins[i+1] and bins[i] >= 3:
            peak_mass = m_min + (i + 0.5) * bin_width
            peaks.append((peak_mass, bins[i]))
    return peaks, masses


def compute_derived(state, metrics, peaks):
    """Compute Tier 2-3 derived quantities, return as dict."""
    derived = {}

    # phi^2 spacing error
    peak_masses = sorted(p[0] for p in peaks)
    spacings = [peak_masses[i+1] - peak_masses[i]
                for i in range(len(peak_masses) - 1)] if len(peak_masses) > 1 else []
    mean_sp = sum(spacings) / len(spacings) if spacings else 0
    derived["_phi2_spacing_err"] = (
        abs(mean_sp - PHI_INV2) / PHI_INV2 * 100 if spacings else 999
    )

    # PAC drift (absolute value, as percentage of initial)
    derived["_pac_drift"] = abs(metrics.get("_pac_drift_raw", 0))

    # Tier 3: fine structure proxy
    em_energy = metrics.get("charge_force_mean", 0)
    total_energy = state.total_energy
    if total_energy > 0 and em_energy > 0:
        # e_local_mean is the emergent charge-dominance ratio
        # Scale it to match alpha's order of magnitude
        e_local = metrics.get("e_local_mean", 0)
        alpha_local = metrics.get("alpha_local_mean", 0)
        # Proxy: e_local * alpha_local * some geometric factor
        # This is exploratory -- we're looking for what combination matches
        derived["_alpha_em"] = e_local * alpha_local if e_local > 0 else None
    else:
        derived["_alpha_em"] = None

    # Tier 3: Koide Q from 3 lightest mass peaks
    if len(peak_masses) >= 3:
        m1, m2, m3 = peak_masses[0], peak_masses[1], peak_masses[2]
        sqrt_sum = math.sqrt(m1) + math.sqrt(m2) + math.sqrt(m3)
        if sqrt_sum > 0:
            derived["_koide_q"] = (m1 + m2 + m3) / sqrt_sum**2
        else:
            derived["_koide_q"] = None
    else:
        derived["_koide_q"] = None

    # Tier 3: mu/e mass ratio (2nd peak / 1st peak)
    if len(peak_masses) >= 2 and peak_masses[0] > 0:
        derived["_mu_e_ratio"] = peak_masses[1] / peak_masses[0]
    else:
        derived["_mu_e_ratio"] = None

    return derived


# ============================================================
# Scoring
# ============================================================
def grade(error_pct):
    """Letter grade from error percentage."""
    if error_pct < 1:
        return "A"
    elif error_pct < 5:
        return "B"
    elif error_pct < 15:
        return "C"
    elif error_pct < 30:
        return "D"
    else:
        return "F"


def score_target(target, measured):
    """Score a single target. Returns (error_pct, letter, pass_fail)."""
    if measured is None:
        return None, "N/A", False

    if target.get("positive_good"):
        # Just check sign
        ok = measured > 0
        return (0.0 if ok else 100.0), ("A" if ok else "F"), ok

    tgt = target["target"]
    if target.get("is_error_pct"):
        # Measured is already an error percentage
        error_pct = abs(measured)
    elif target.get("absolute"):
        # Score by absolute value of measured
        error_pct = abs(measured) * 100 if tgt == 0.0 else (
            abs(measured - tgt) / max(abs(tgt), 1e-12) * 100
        )
    else:
        if abs(tgt) < 1e-12:
            error_pct = abs(measured) * 100
        else:
            error_pct = abs(measured - tgt) / abs(tgt) * 100

    g = grade(error_pct)
    return error_pct, g, error_pct < 15


# ============================================================
# Tuning Suggestions
# ============================================================
TUNING_MAP = {
    "f_local -> gamma_EM": [
        ("gamma_damping", "RBF damping -> collapse rate"),
        ("actualization_threshold", "MAR gate -> actualization rate"),
        ("confluence_weight", "E-I mixing -> disequilibrium"),
    ],
    "gamma_local -> 1/phi": [
        ("mass_gen_coeff", "mass generation -> M/E balance"),
        ("actualization_threshold", "MAR gate -> mass generation rate"),
    ],
    "alpha_local -> ln(2)": [
        ("gamma_damping", "RBF damping -> collapse rate"),
        ("field_scale", "soft clamp -> field magnitude ceiling"),
    ],
    "G_local -> 1/phi^2": [
        ("mass_gen_coeff", "mass generation -> M/E ratio"),
        ("actualization_threshold", "MAR threshold -> settled fraction"),
        ("field_scale", "soft clamp -> mass ceiling"),
    ],
    "lambda_local -> 1-ln(2)": [
        ("gamma_damping", "coupled with alpha (alpha+lambda~1)"),
    ],
    "phi^2 mass spacing": [
        ("quantum_pressure_coeff", "pressure vs gravity -> peak separation"),
        ("mass_gen_coeff", "mass generation rate -> peak density"),
        ("field_scale", "mass cap -> peak crowding"),
    ],
}


def suggest_tuning(results, config):
    """Print tuning suggestions for poorly-scoring metrics."""
    suggestions = []
    for r in results:
        if r["grade"] in ("D", "F") and r["name"] in TUNING_MAP:
            direction = "too high" if r["measured"] is not None and r["target"] is not None and r["measured"] > r["target"] else "too low"
            params = TUNING_MAP[r["name"]]
            param_strs = []
            for pname, mechanism in params:
                current = getattr(config, pname, "?")
                param_strs.append(f"    {pname} = {current} ({mechanism})")
            suggestions.append({
                "name": r["name"],
                "direction": direction,
                "error": r["error_pct"],
                "params": param_strs,
            })
    return suggestions


# ============================================================
# Pipeline
# ============================================================
def build_pipeline():
    return Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
        GravitationalCollapseOperator(),
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), SECTrackingOperator(),
        AdaptiveOperator(), TimeEmergenceOperator(),
    ])


# ============================================================
# Main
# ============================================================
def run_scorecard(total_ticks=10000, device=None):
    checkpoints = [500, 1000, 2000, 5000, total_ticks]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True,
        actualization_threshold=0.05,
    )
    torch.manual_seed(42)
    pipeline = build_pipeline()
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)
    pac_initial = engine.state.pac_total

    targets = build_targets()
    all_checkpoint_results = []
    t0 = time.time()
    cp_idx = 0

    print(f"\n{'=' * 70}")
    print(f"  REALITY ENGINE PHYSICS SCORECARD")
    print(f"  Device: {device} | Grid: {config.nu}x{config.nv} | dt={config.dt}")
    print(f"{'=' * 70}")
    print(f"\n  Running {total_ticks} ticks...")

    for tick in range(1, total_ticks + 1):
        engine.tick()
        if cp_idx < len(checkpoints) and tick == checkpoints[cp_idx]:
            cp_idx += 1
            elapsed = time.time() - t0
            state = engine.state
            metrics = dict(state.metrics)
            metrics["_pac_drift_raw"] = state.pac_total - pac_initial

            peaks, masses = find_mass_peaks(state.M)

            derived = compute_derived(state, metrics, peaks)
            metrics.update(derived)

            # Score all targets
            results = []
            for t in targets:
                key = t["metric_key"]
                measured = metrics.get(key)
                if isinstance(measured, torch.Tensor):
                    measured = measured.mean().item()
                error_pct, g, passed = score_target(t, measured)
                results.append({
                    "name": t["name"],
                    "tier": t["tier"],
                    "target": t["target"],
                    "target_label": t["target_label"],
                    "measured": measured,
                    "error_pct": error_pct,
                    "grade": g,
                    "passed": passed,
                    "description": t["description"],
                })

            all_checkpoint_results.append({
                "tick": tick,
                "elapsed": elapsed,
                "n_peaks": len(peaks),
                "n_structures": len(masses),
                "results": results,
            })

            n_pass = sum(1 for r in results if r["passed"])
            n_scored = sum(1 for r in results if r["grade"] != "N/A")
            print(f"  Tick {tick:5d} ({elapsed:5.0f}s): "
                  f"{len(peaks)} peaks, {len(masses)} structs, "
                  f"score {n_pass}/{n_scored}")

    # ================================================================
    # Print final scorecard
    # ================================================================
    final = all_checkpoint_results[-1]
    results = final["results"]

    for tier in [1, 2, 3]:
        tier_results = [r for r in results if r["tier"] == tier]
        if not tier_results:
            continue
        tier_labels = {
            1: "Emergent Coupling Attractors",
            2: "Structural Physics",
            3: "Deep DFT Constants (aspirational)",
        }
        print(f"\n{'=' * 70}")
        print(f"  TIER {tier}: {tier_labels[tier]}")
        print(f"{'=' * 70}")
        print(f"  {'Metric':<25s} {'Target':>12s} {'Measured':>12s} {'Error':>8s} {'Grade':>6s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*8} {'-'*6}")
        for r in tier_results:
            tgt_str = r["target_label"]
            if r["measured"] is None:
                meas_str = "N/A"
                err_str = ""
            elif r.get("error_pct") is None:
                meas_str = f"{r['measured']:.6f}"
                err_str = ""
            else:
                meas_str = f"{r['measured']:.6f}" if abs(r['measured']) < 1000 else f"{r['measured']:.1f}"
                err_str = f"{r['error_pct']:.1f}%"
            print(f"  {r['name']:<25s} {tgt_str:>12s} {meas_str:>12s} {err_str:>8s}   [{r['grade']}]")

    # Overall
    scored = [r for r in results if r["grade"] != "N/A"]
    passing = [r for r in scored if r["passed"]]
    grades = [r["grade"] for r in scored]
    gpa_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
    gpa = sum(gpa_map.get(g, 0) for g in grades) / max(len(grades), 1)
    gpa_letter = "A" if gpa >= 3.5 else "B" if gpa >= 2.5 else "C" if gpa >= 1.5 else "D" if gpa >= 0.5 else "F"
    plus_minus = "+" if gpa % 1 >= 0.67 else "-" if gpa % 1 < 0.33 else ""

    best = min(scored, key=lambda r: r["error_pct"] if r["error_pct"] is not None else 999)
    worst = max(scored, key=lambda r: r["error_pct"] if r["error_pct"] is not None else -1)

    print(f"\n{'=' * 70}")
    print(f"  OVERALL: {len(passing)}/{len(scored)} passing (<=15% error), "
          f"GPA: {gpa:.1f} ({gpa_letter}{plus_minus})")
    print(f"  Best:  {best['name']} ({best['error_pct']:.1f}%)")
    print(f"  Worst: {worst['name']} ({worst['error_pct']:.1f}%)")
    print(f"{'=' * 70}")

    # Convergence table
    print(f"\n{'=' * 70}")
    print(f"  CONVERGENCE")
    print(f"{'=' * 70}")

    # Pick Tier 1 metrics to track over time
    tier1_names = [r["name"] for r in results if r["tier"] == 1][:5]
    header = f"  {'Tick':>6s}"
    for name in tier1_names:
        short = name.split(" -> ")[0] if " -> " in name else name[:10]
        header += f" | {short:>10s}"
    print(header)
    print(f"  {'-'*6}" + ("-+-" + "-"*10) * len(tier1_names))

    for cp in all_checkpoint_results:
        row = f"  {cp['tick']:6d}"
        for name in tier1_names:
            r = next((x for x in cp["results"] if x["name"] == name), None)
            if r and r["error_pct"] is not None:
                row += f" | {r['error_pct']:9.1f}%"
            else:
                row += f" | {'N/A':>10s}"
        print(row)

    # Tuning suggestions
    suggestions = suggest_tuning(results, config)
    if suggestions:
        print(f"\n{'=' * 70}")
        print(f"  TUNING SUGGESTIONS")
        print(f"{'=' * 70}")
        for s in suggestions:
            print(f"\n  {s['name']} ({s['direction']}, {s['error']:.1f}% error):")
            for p in s["params"]:
                print(p)

    print()
    return all_checkpoint_results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_scorecard(total_ticks=10000, device=device)


if __name__ == "__main__":
    main()
