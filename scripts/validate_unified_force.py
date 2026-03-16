"""Validate UnifiedForceOperator — compare unified vs separate gravity + EM.

Runs two 10K-tick simulations:
1. SEPARATE: GravitationalCollapseOperator + ChargeDynamicsOperator (current)
2. UNIFIED: UnifiedForceOperator (replaces both)

Compares mass peaks, spacing, PAC conservation, and new coupling metrics.
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
from src.v3.operators.unified_force import UnifiedForceOperator

PHI = (1 + math.sqrt(5)) / 2
PHI_INV2 = 1.0 / PHI ** 2


def build_separate_pipeline():
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


def build_unified_pipeline():
    return Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
        UnifiedForceOperator(),  # replaces Gravity + Charge
        SpinStatisticsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), SECTrackingOperator(),
        AdaptiveOperator(), TimeEmergenceOperator(),
    ])


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


def run_sim(label, pipeline_fn, device, total_ticks=10000):
    print(f"\n{'='*70}")
    print(f"  Running {label}")
    print(f"{'='*70}\n")

    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True,
        actualization_threshold=0.05,
    )
    torch.manual_seed(42)
    pipeline = pipeline_fn()
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
            frac_cap = sum(1 for m in masses if m >= M_cap * 0.95) / max(len(masses), 1)

            peak_masses = sorted(p[0] for p in peaks)
            spacings = [peak_masses[i+1] - peak_masses[i] for i in range(len(peak_masses)-1)] if len(peak_masses) > 1 else []
            mean_sp = sum(spacings) / len(spacings) if spacings else 0
            err_phi2 = abs(mean_sp - PHI_INV2) / PHI_INV2 * 100 if spacings else 999

            result = {
                'tick': tick,
                'elapsed': elapsed,
                'n_structures': len(masses),
                'n_peaks': len(peaks),
                'peak_masses': [round(p, 3) for p in peak_masses[:8]],
                'mass_mean': sum(masses) / max(len(masses), 1),
                'mass_std': (sum((m - sum(masses)/max(len(masses),1))**2 for m in masses) / max(len(masses),1)) ** 0.5 if masses else 0,
                'frac_at_cap': frac_cap,
                'pac_drift': state.pac_total - pac_initial,
                'mean_spacing': mean_sp,
                'err_phi2': err_phi2,
                'grav_em_ratio': state.metrics.get('grav_em_ratio', 0),
                'field_entropy': state.metrics.get('field_entropy', 0),
            }
            results.append(result)

            peak_str = ", ".join(f"{p:.3f}" for p in result['peak_masses'][:6])
            print(f"  Tick {tick:5d} ({elapsed:.0f}s): "
                  f"{len(masses)} structs, {len(peaks)} peaks [{peak_str}] "
                  f"cap={frac_cap:.1%} phi2_err={err_phi2:.1f}% "
                  f"PAC_d={result['pac_drift']:.2e}")
            if result['grav_em_ratio']:
                print(f"    grav/em ratio: {result['grav_em_ratio']:.4f}, "
                      f"entropy: {result['field_entropy']:.2f}")

    return results


def compare(sep, uni):
    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD COMPARISON (tick 10000)")
    print(f"{'='*70}\n")

    s, u = sep[-1], uni[-1]

    rows = [
        ("Structures", s['n_structures'], u['n_structures']),
        ("Mass peaks", s['n_peaks'], u['n_peaks']),
        ("Mass mean", f"{s['mass_mean']:.3f}", f"{u['mass_mean']:.3f}"),
        ("Mass std", f"{s['mass_std']:.3f}", f"{u['mass_std']:.3f}"),
        ("Fraction at cap", f"{s['frac_at_cap']:.1%}", f"{u['frac_at_cap']:.1%}"),
        ("Mean spacing", f"{s['mean_spacing']:.4f}", f"{u['mean_spacing']:.4f}"),
        ("Spacing err vs 1/phi^2", f"{s['err_phi2']:.1f}%", f"{u['err_phi2']:.1f}%"),
        ("PAC drift", f"{s['pac_drift']:.2e}", f"{u['pac_drift']:.2e}"),
        ("Grav/EM ratio", f"{s['grav_em_ratio']:.4f}" if s['grav_em_ratio'] else "N/A", f"{u['grav_em_ratio']:.4f}" if u['grav_em_ratio'] else "N/A"),
        ("Field entropy", f"{s['field_entropy']:.2f}" if s['field_entropy'] else "N/A", f"{u['field_entropy']:.2f}" if u['field_entropy'] else "N/A"),
    ]

    print(f"  {'Metric':<25s} | {'SEPARATE':>15s} | {'UNIFIED':>15s}")
    print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*15}")
    for name, v1, v2 in rows:
        print(f"  {name:<25s} | {str(v1):>15s} | {str(v2):>15s}")

    # Verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    score = 0
    total = 5

    if u['n_peaks'] >= max(s['n_peaks'] - 2, 5):
        print(f"  [PASS] Peak count: {s['n_peaks']} -> {u['n_peaks']}")
        score += 1
    else:
        print(f"  [FAIL] Peak count: {s['n_peaks']} -> {u['n_peaks']}")

    if u['err_phi2'] < 50:
        print(f"  [PASS] Phi^2 spacing error: {u['err_phi2']:.1f}% (< 50%)")
        score += 1
    else:
        print(f"  [FAIL] Phi^2 spacing error: {u['err_phi2']:.1f}%")

    if abs(u['pac_drift']) < 1e-3:
        print(f"  [PASS] PAC conservation: drift = {u['pac_drift']:.2e}")
        score += 1
    else:
        print(f"  [FAIL] PAC conservation: drift = {u['pac_drift']:.2e}")

    if u['mass_std'] >= s['mass_std'] * 0.5:
        print(f"  [PASS] Mass diversity: {s['mass_std']:.3f} -> {u['mass_std']:.3f}")
        score += 1
    else:
        print(f"  [FAIL] Mass diversity: {s['mass_std']:.3f} -> {u['mass_std']:.3f}")

    if u['n_structures'] >= s['n_structures'] * 0.3:
        print(f"  [PASS] Structure count: {s['n_structures']} -> {u['n_structures']}")
        score += 1
    else:
        print(f"  [FAIL] Structure count: {s['n_structures']} -> {u['n_structures']}")

    print(f"\n  Score: {score}/{total}")

    # Evolution
    print(f"\n{'='*70}")
    print(f"  EVOLUTION")
    print(f"{'='*70}")
    print(f"  {'Tick':>6s} | {'Peaks(sep)':>10s} | {'Peaks(uni)':>10s} | {'Phi2(sep)':>9s} | {'Phi2(uni)':>9s} | {'G/EM ratio':>10s}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}")
    for si, ui in zip(sep, uni):
        r = f"{ui['grav_em_ratio']:.4f}" if ui['grav_em_ratio'] else "N/A"
        print(f"  {si['tick']:6d} | {si['n_peaks']:10d} | {ui['n_peaks']:10d} | {si['err_phi2']:8.1f}% | {ui['err_phi2']:8.1f}% | {r:>10s}")

    return score, total


def main():
    print("=" * 70)
    print("  UnifiedForceOperator Validation")
    print("  Single pre-field -> gravity + EM via symmetric/antisymmetric")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    sep = run_sim("SEPARATE (Gravity + Charge)", build_separate_pipeline, device)
    uni = run_sim("UNIFIED (UnifiedForceOperator)", build_unified_pipeline, device)
    compare(sep, uni)


if __name__ == "__main__":
    main()
