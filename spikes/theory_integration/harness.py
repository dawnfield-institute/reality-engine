"""Shared harness for theory integration spikes.

Provides standard pipeline construction, scoring, and reporting
so each spike can focus on its specific modification.
"""

import math
import os
import sys
import time

re_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
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

PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
LN2 = math.log(2)
GAMMA_EM = 0.5772156649015328
PHI_INV = 1.0 / PHI
PHI_INV2 = 1.0 / PHI ** 2

TARGETS = {
    "f_local": ("f_local_mean", GAMMA_EM),
    "gamma": ("gamma_local_mean", PHI_INV),
    "alpha": ("alpha_local_mean", LN2),
    "G_local": ("G_local_mean", PHI_INV2),
    "lambda": ("lambda_local_mean", 1 - LN2),
}


def default_pipeline():
    """Standard v3 pipeline."""
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


def default_config(device=None, **overrides):
    """Standard config with optional overrides."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = dict(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True,
        actualization_threshold=0.05,
        quantum_pressure_coeff=0.020,
        mass_gen_coeff=0.63,
        deactualization_rate=0.01,
    )
    kwargs.update(overrides)
    return SimulationConfig(**kwargs)


def tier1_errors(metrics):
    """Compute percentage errors for all Tier 1 constants."""
    errs = {}
    for name, (key, target) in TARGETS.items():
        val = metrics.get(key, None)
        if val is not None:
            errs[name] = abs(val - target) / abs(target) * 100
        else:
            errs[name] = 999
    return errs


def find_mass_peaks(M, min_mass=0.1, n_bins=40):
    """Find mass peaks for phi^2 spacing analysis."""
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


def phi2_spacing_err(peaks):
    """Percentage error in mean peak spacing vs 1/phi^2."""
    peak_masses = sorted(p[0] for p in peaks)
    spacings = [peak_masses[i+1] - peak_masses[i]
                for i in range(len(peak_masses) - 1)] if len(peak_masses) > 1 else []
    mean_sp = sum(spacings) / len(spacings) if spacings else 0
    return abs(mean_sp - PHI_INV2) / PHI_INV2 * 100 if spacings else 999


def run_and_score(label, pipeline, config, ticks=10000, snapshots=None):
    """Run simulation and return full scoring results.

    If snapshots is a list of tick numbers, also records metrics at those ticks.
    """
    torch.manual_seed(42)
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    snapshot_data = {}
    t0 = time.time()
    for tick in range(1, ticks + 1):
        engine.tick()
        if snapshots and tick in snapshots:
            m = dict(engine.state.metrics)
            snapshot_data[tick] = {
                "t1_errs": tier1_errors(m),
                "metrics": m,
            }

    elapsed = time.time() - t0
    state = engine.state
    metrics = dict(state.metrics)
    peaks, masses = find_mass_peaks(state.M)

    result = {
        "label": label,
        "t1_errs": tier1_errors(metrics),
        "avg_t1": sum(tier1_errors(metrics).values()) / len(TARGETS),
        "phi2_err": phi2_spacing_err(peaks),
        "n_peaks": len(peaks),
        "n_structs": len(masses),
        "entropy_rate": metrics.get("entropy_reduction_rate", 0),
        "entropy_cumul": metrics.get("entropy_reduction_cumulative", 0),
        "M_mean": state.M.mean().item(),
        "M_max": state.M.max().item(),
        "elapsed": elapsed,
        "metrics": metrics,
        "snapshots": snapshot_data,
    }
    return result


def print_result(r, show_t1=True):
    """Pretty-print a scoring result."""
    print(f"\n  {r['label']}")
    print(f"  {'='*60}")
    print(f"  avg_t1={r['avg_t1']:.1f}%  phi2={r['phi2_err']:.1f}%  "
          f"peaks={r['n_peaks']}  structs={r['n_structs']}")
    print(f"  M_mean={r['M_mean']:.3f}  M_max={r['M_max']:.2f}  "
          f"entropy_cumul={r['entropy_cumul']:.4f}")
    print(f"  [{r['elapsed']:.0f}s]")
    if show_t1:
        t1 = r["t1_errs"]
        print(f"  Tier 1: f={t1['f_local']:.1f}%  gamma={t1['gamma']:.1f}%  "
              f"alpha={t1['alpha']:.1f}%  G={t1['G_local']:.1f}%  "
              f"lambda={t1['lambda']:.1f}%")


def print_comparison(results):
    """Print side-by-side comparison table."""
    print(f"\n  {'Label':<30s} {'avg_t1':>7s} {'phi2%':>7s} {'entropy':>8s} "
          f"{'f':>6s} {'gamma':>6s} {'alpha':>6s} {'G':>6s} {'lam':>6s} "
          f"{'peaks':>5s}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*8} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*5}")
    for r in results:
        t1 = r["t1_errs"]
        print(f"  {r['label']:<30s} {r['avg_t1']:>6.1f}% {r['phi2_err']:>6.1f}% "
              f"{r['entropy_cumul']:>+8.3f} "
              f"{t1['f_local']:>5.1f}% {t1['gamma']:>5.1f}% {t1['alpha']:>5.1f}% "
              f"{t1['G_local']:>5.1f}% {t1['lambda']:>5.1f}% "
              f"{r['n_peaks']:>5d}")
