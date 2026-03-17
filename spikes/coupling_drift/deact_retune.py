"""Retune memory/gravity parameters WITH de-actualization active.

De-actualization (eta=0.01) improved Tier 1 coupling constants (8.1% -> 6.2%)
but regressed two Tier 2 metrics:
  - phi^2 mass spacing: 10.8% -> 40.6% (F)
  - entropy reduction: A -> F

The pre-deact parameters were tuned for a system where mass only grows.
With mass now dissolving in balanced regions, the peak distribution changes.
This sweep re-tunes quantum_pressure_coeff and mass_gen_coeff with deact active.

Hypothesis: Higher quantum pressure (>0.015) compensates for deact-induced
peak compression. Mass gen coefficient may need adjustment too.
"""

import math
import os
import sys
import time
from itertools import product

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

# Tier 1 targets
TARGETS = {
    "f_local": ("f_local_mean", GAMMA_EM),
    "gamma": ("gamma_local_mean", PHI_INV),
    "alpha": ("alpha_local_mean", LN2),
    "G_local": ("G_local_mean", PHI_INV2),
    "lambda": ("lambda_local_mean", 1 - LN2),
}


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


def phi2_spacing_err(peaks):
    peak_masses = sorted(p[0] for p in peaks)
    spacings = [peak_masses[i+1] - peak_masses[i]
                for i in range(len(peak_masses) - 1)] if len(peak_masses) > 1 else []
    mean_sp = sum(spacings) / len(spacings) if spacings else 0
    return abs(mean_sp - PHI_INV2) / PHI_INV2 * 100 if spacings else 999


def tier1_errors(metrics):
    errs = {}
    for name, (key, target) in TARGETS.items():
        val = metrics.get(key, None)
        if val is not None:
            errs[name] = abs(val - target) / abs(target) * 100
        else:
            errs[name] = 999
    return errs


def run_variant(label, qp_coeff, mg_coeff, eta, ticks=5000, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True,
        actualization_threshold=0.05,
        quantum_pressure_coeff=qp_coeff,
        mass_gen_coeff=mg_coeff,
        deactualization_rate=eta,
    )
    torch.manual_seed(42)
    pipeline = Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
        GravitationalCollapseOperator(),
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), SECTrackingOperator(),
        AdaptiveOperator(), TimeEmergenceOperator(),
    ])
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    t0 = time.time()
    for tick in range(1, ticks + 1):
        engine.tick()

    elapsed = time.time() - t0
    state = engine.state
    metrics = dict(state.metrics)

    peaks, masses = find_mass_peaks(state.M)
    phi2_err = phi2_spacing_err(peaks)
    t1_errs = tier1_errors(metrics)
    avg_t1 = sum(t1_errs.values()) / len(t1_errs)
    entropy_rate = metrics.get("entropy_reduction_rate", 0)

    return {
        "label": label,
        "qp_coeff": qp_coeff,
        "mg_coeff": mg_coeff,
        "eta": eta,
        "phi2_err": phi2_err,
        "entropy_rate": entropy_rate,
        "n_peaks": len(peaks),
        "n_structs": len(masses),
        "t1_errs": t1_errs,
        "avg_t1": avg_t1,
        "M_mean": state.M.mean().item(),
        "M_max": state.M.max().item(),
        "elapsed": elapsed,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticks = 5000

    print("=" * 90)
    print("  DE-ACTUALIZATION PARAMETER RETUNE")
    print(f"  Device: {device} | Grid: 128x64 | {ticks} ticks")
    print("=" * 90)

    # Sweep: quantum_pressure_coeff x mass_gen_coeff
    # Current defaults: qp=0.015, mg=0.63, eta=0.01
    qp_values = [0.010, 0.015, 0.020, 0.030, 0.050]
    mg_values = [0.40, 0.63, 0.80]
    eta = 0.01  # keep de-actualization rate fixed

    variants = []
    for qp, mg in product(qp_values, mg_values):
        label = f"qp={qp:.3f}_mg={mg:.2f}"
        variants.append((label, qp, mg, eta))

    results = []
    for i, (label, qp, mg, eta_val) in enumerate(variants):
        print(f"\n  [{i+1}/{len(variants)}] {label} ...", end="", flush=True)
        r = run_variant(label, qp, mg, eta_val, ticks=ticks, device=device)
        results.append(r)
        entropy_sign = "+" if r["entropy_rate"] > 0 else ""
        print(f"  phi2={r['phi2_err']:.1f}%  avg_t1={r['avg_t1']:.1f}%  "
              f"entropy={entropy_sign}{r['entropy_rate']:.6f}  "
              f"peaks={r['n_peaks']}  [{r['elapsed']:.0f}s]")

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"  RESULTS (sorted by composite score: avg_t1 + phi2_err/10)")
    print(f"{'=' * 90}")
    print(f"  {'Label':<22s} {'phi2%':>7s} {'avg_t1%':>8s} {'entropy':>10s} "
          f"{'f_local':>8s} {'gamma':>8s} {'alpha':>8s} {'G_local':>8s} {'lambda':>8s} "
          f"{'peaks':>5s} {'M_mean':>6s}")
    print(f"  {'-'*22} {'-'*7} {'-'*8} {'-'*10} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*5} {'-'*6}")

    # Sort by composite: we want low avg_t1 AND low phi2_err AND positive entropy
    def composite(r):
        entropy_penalty = 50 if r["entropy_rate"] <= 0 else 0
        return r["avg_t1"] + r["phi2_err"] / 10 + entropy_penalty

    results.sort(key=composite)

    for r in results:
        entropy_sign = "+" if r["entropy_rate"] > 0 else ""
        t1 = r["t1_errs"]
        print(f"  {r['label']:<22s} {r['phi2_err']:>6.1f}% {r['avg_t1']:>7.1f}% "
              f"{entropy_sign}{r['entropy_rate']:>9.6f} "
              f"{t1['f_local']:>7.1f}% {t1['gamma']:>7.1f}% {t1['alpha']:>7.1f}% "
              f"{t1['G_local']:>7.1f}% {t1['lambda']:>7.1f}% "
              f"{r['n_peaks']:>5d} {r['M_mean']:>6.2f}")

    # Best overall
    best = results[0]
    print(f"\n  BEST: {best['label']}  phi2={best['phi2_err']:.1f}%  "
          f"avg_t1={best['avg_t1']:.1f}%  entropy={'>' if best['entropy_rate'] > 0 else '<'}0")
    print(f"  Current defaults: qp=0.015, mg=0.63 -> phi2=40.6%, avg_t1=8.7%")


if __name__ == "__main__":
    main()
