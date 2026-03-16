"""Validate PhiCascadeOperator — compare mass spectra with/without cascade.

Measures:
1. Mass peak count and positions
2. Peak spacing vs phi^(-k) prediction
3. Cap pileup fraction
4. PAC conservation
5. Cascade-specific metrics (depth, proximity, cascade_dM)

Runs two 10K-tick simulations side by side (without/with PhiCascade).
"""

import math
import os
import sys
import time
from datetime import datetime

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

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1.0 / PHI
PHI_INV2 = PHI_INV ** 2


def build_pipeline(with_cascade=False):
    ops = [
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(),
    ]
    if with_cascade:
        ops.append(PhiCascadeOperator())
    ops.extend([
        GravitationalCollapseOperator(),
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), AdaptiveOperator(), TimeEmergenceOperator(),
    ])
    return Pipeline(ops)


def find_mass_peaks(M, min_mass=0.1, n_bins=40):
    """Find peaks in mass histogram from local maxima."""
    # Local maxima
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


def analyze_phi_spacing(peaks):
    """Check if peak spacing matches phi^(-k) intervals."""
    if len(peaks) < 3:
        return {}

    peak_masses = sorted(p[0] for p in peaks)
    spacings = [peak_masses[i+1] - peak_masses[i] for i in range(len(peak_masses)-1)]
    mean_sp = sum(spacings) / len(spacings)

    # Check ratios between consecutive spacings — should be ~phi or ~1
    ratios = [spacings[i+1] / (spacings[i] + 1e-10) for i in range(len(spacings)-1)]

    # Error vs 1/phi^2 spacing
    err_phi2 = abs(mean_sp - PHI_INV2) / PHI_INV2 * 100

    # Error vs 1/phi spacing
    err_phi1 = abs(mean_sp - PHI_INV) / PHI_INV * 100

    # Check if peak masses themselves are at phi^(-k) intervals from max
    m_max = max(peak_masses)
    phi_levels = [m_max * PHI ** (-k) for k in range(15) if m_max * PHI ** (-k) > 0.05]

    # For each peak, find nearest phi-level
    phi_errors = []
    for pm in peak_masses:
        nearest = min(phi_levels, key=lambda pl: abs(pl - pm))
        phi_errors.append(abs(pm - nearest) / nearest * 100)

    return {
        'n_peaks': len(peaks),
        'mean_spacing': mean_sp,
        'spacing_std': (sum((s - mean_sp)**2 for s in spacings) / len(spacings)) ** 0.5,
        'err_vs_phi2': err_phi2,
        'err_vs_phi1': err_phi1,
        'spacing_ratios': ratios,
        'phi_level_errors': phi_errors,
        'mean_phi_level_err': sum(phi_errors) / len(phi_errors) if phi_errors else 999,
    }


def run_simulation(label, with_cascade, device, total_ticks=10000):
    """Run a full simulation and collect metrics at checkpoints."""
    print(f"\n{'='*70}")
    print(f"  Running {label}")
    print(f"{'='*70}\n")

    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True,
        actualization_threshold=0.05,
    )

    torch.manual_seed(42)
    pipeline = build_pipeline(with_cascade=with_cascade)
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)
    pac_initial = engine.state.pac_total

    checkpoints = [500, 1000, 2000, 5000, 10000]
    results = []
    t0 = time.time()
    cp_idx = 0

    for tick in range(1, total_ticks + 1):
        engine.tick()

        if cp_idx < len(checkpoints) and tick == checkpoints[cp_idx]:
            cp_idx += 1
            elapsed = time.time() - t0
            state = engine.state

            peaks, masses = find_mass_peaks(state.M)
            M_cap = config.field_scale / 5.0
            n_at_cap = sum(1 for m in masses if m >= M_cap * 0.95)
            frac_cap = n_at_cap / max(len(masses), 1)

            spacing = analyze_phi_spacing(peaks)

            # Cascade-specific metrics
            cascade_metrics = {}
            if with_cascade:
                cascade_metrics = {
                    'cascade_depth_mean': state.metrics.get('cascade_depth_mean', 0),
                    'cascade_depth_std': state.metrics.get('cascade_depth_std', 0),
                    'phi_proximity_mean': state.metrics.get('phi_proximity_mean', 0),
                    'cascade_dM': state.metrics.get('cascade_dM', 0),
                    'cascade_rate_mean': state.metrics.get('cascade_rate_mean', 0),
                }

            result = {
                'tick': tick,
                'elapsed': elapsed,
                'n_structures': len(masses),
                'n_peaks': len(peaks),
                'peak_masses': [round(p[0], 3) for p in sorted(peaks)],
                'mass_mean': sum(masses) / max(len(masses), 1),
                'mass_std': (sum((m - sum(masses)/max(len(masses),1))**2 for m in masses) / max(len(masses),1)) ** 0.5 if masses else 0,
                'mass_range': [round(min(masses), 3), round(max(masses), 3)] if masses else [0, 0],
                'frac_at_cap': frac_cap,
                'pac_drift': state.pac_total - pac_initial,
                'spacing': spacing,
                'cascade_metrics': cascade_metrics,
            }
            results.append(result)

            peak_str = ", ".join(f"{p:.3f}" for p in result['peak_masses'][:8])
            if len(result['peak_masses']) > 8:
                peak_str += f" (+{len(result['peak_masses'])-8} more)"
            print(f"  Tick {tick:5d} ({elapsed:.0f}s): "
                  f"{len(masses)} structs, {len(peaks)} peaks [{peak_str}] "
                  f"cap={frac_cap:.1%} PAC_d={result['pac_drift']:.2e}")
            if cascade_metrics:
                print(f"    cascade: depth={cascade_metrics['cascade_depth_mean']:.2f} "
                      f"prox={cascade_metrics['phi_proximity_mean']:.3f} "
                      f"dM={cascade_metrics['cascade_dM']:.4f}")

    return results


def compare_and_verdict(without, with_cascade):
    """Compare final-tick results and produce verdict."""
    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD COMPARISON (tick 10000)")
    print(f"{'='*70}\n")

    w = without[-1]  # last checkpoint
    c = with_cascade[-1]

    rows = [
        ("Structures", w['n_structures'], c['n_structures']),
        ("Mass peaks", w['n_peaks'], c['n_peaks']),
        ("Mass mean", f"{w['mass_mean']:.3f}", f"{c['mass_mean']:.3f}"),
        ("Mass std (diversity)", f"{w['mass_std']:.3f}", f"{c['mass_std']:.3f}"),
        ("Mass range", str(w['mass_range']), str(c['mass_range'])),
        ("Fraction at cap", f"{w['frac_at_cap']:.1%}", f"{c['frac_at_cap']:.1%}"),
        ("Mean spacing", f"{w['spacing'].get('mean_spacing', 0):.4f}", f"{c['spacing'].get('mean_spacing', 0):.4f}"),
        ("Spacing err vs 1/phi^2", f"{w['spacing'].get('err_vs_phi2', 999):.1f}%", f"{c['spacing'].get('err_vs_phi2', 999):.1f}%"),
        ("Spacing err vs 1/phi", f"{w['spacing'].get('err_vs_phi1', 999):.1f}%", f"{c['spacing'].get('err_vs_phi1', 999):.1f}%"),
        ("Mean phi-level err", f"{w['spacing'].get('mean_phi_level_err', 999):.1f}%", f"{c['spacing'].get('mean_phi_level_err', 999):.1f}%"),
        ("PAC drift", f"{w['pac_drift']:.2e}", f"{c['pac_drift']:.2e}"),
    ]

    print(f"  {'Metric':<25s} | {'WITHOUT':>15s} | {'WITH CASCADE':>15s}")
    print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*15}")
    for name, v1, v2 in rows:
        marker = ""
        print(f"  {name:<25s} | {str(v1):>15s} | {str(v2):>15s} {marker}")

    if c.get('cascade_metrics'):
        cm = c['cascade_metrics']
        print(f"\n  Cascade metrics:")
        print(f"    Depth mean:     {cm['cascade_depth_mean']:.3f}")
        print(f"    Depth std:      {cm['cascade_depth_std']:.3f}")
        print(f"    Phi proximity:  {cm['phi_proximity_mean']:.3f}")
        print(f"    Cascade dM:     {cm['cascade_dM']:.6f}")
        print(f"    Cascade rate:   {cm['cascade_rate_mean']:.6f}")

    # Verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    score = 0
    total = 5

    # 1. More peaks?
    if c['n_peaks'] >= w['n_peaks']:
        print(f"  [PASS] Peak count: {w['n_peaks']} -> {c['n_peaks']}")
        score += 1
    else:
        print(f"  [FAIL] Peak count: {w['n_peaks']} -> {c['n_peaks']}")

    # 2. Better phi-spacing?
    w_sp_err = w['spacing'].get('err_vs_phi2', 999)
    c_sp_err = c['spacing'].get('err_vs_phi2', 999)
    if c_sp_err < w_sp_err:
        print(f"  [PASS] Phi^2 spacing error: {w_sp_err:.1f}% -> {c_sp_err:.1f}%")
        score += 1
    else:
        print(f"  [FAIL] Phi^2 spacing error: {w_sp_err:.1f}% -> {c_sp_err:.1f}%")

    # 3. Less cap pileup?
    if c['frac_at_cap'] <= w['frac_at_cap']:
        print(f"  [PASS] Cap pileup: {w['frac_at_cap']:.1%} -> {c['frac_at_cap']:.1%}")
        score += 1
    else:
        print(f"  [FAIL] Cap pileup: {w['frac_at_cap']:.1%} -> {c['frac_at_cap']:.1%}")

    # 4. PAC conservation
    if abs(c['pac_drift']) < 1e-6:
        print(f"  [PASS] PAC conservation: drift = {c['pac_drift']:.2e}")
        score += 1
    else:
        print(f"  [FAIL] PAC conservation: drift = {c['pac_drift']:.2e}")

    # 5. More mass diversity?
    if c['mass_std'] >= w['mass_std'] * 0.9:
        print(f"  [PASS] Mass diversity: {w['mass_std']:.3f} -> {c['mass_std']:.3f}")
        score += 1
    else:
        print(f"  [FAIL] Mass diversity: {w['mass_std']:.3f} -> {c['mass_std']:.3f}")

    print(f"\n  Score: {score}/{total}")
    return score, total


def main():
    print("=" * 70)
    print("  PhiCascadeOperator Validation")
    print("  Fibonacci two-step memory for phi-spaced mass levels")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    without = run_simulation("WITHOUT PhiCascade", with_cascade=False, device=device)
    with_cascade = run_simulation("WITH PhiCascade", with_cascade=True, device=device)

    score, total = compare_and_verdict(without, with_cascade)

    # Evolution comparison
    print(f"\n{'='*70}")
    print(f"  EVOLUTION COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Tick':>6s} | {'Peaks (w/o)':>11s} | {'Peaks (w/)':>10s} | {'Cap (w/o)':>9s} | {'Cap (w/)':>8s} | {'Phi2 err (w/o)':>14s} | {'Phi2 err (w/)':>13s}")
    print(f"  {'-'*6}-+-{'-'*11}-+-{'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*14}-+-{'-'*13}")
    for wo, wc in zip(without, with_cascade):
        w_err = wo['spacing'].get('err_vs_phi2', 999)
        c_err = wc['spacing'].get('err_vs_phi2', 999)
        print(f"  {wo['tick']:6d} | {wo['n_peaks']:11d} | {wc['n_peaks']:10d} | {wo['frac_at_cap']:8.1%} | {wc['frac_at_cap']:7.1%} | {w_err:13.1f}% | {c_err:12.1f}%")


if __name__ == "__main__":
    main()
