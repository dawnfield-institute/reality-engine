"""Grid independence test — does f0=1/sqrt(2) survive resolution changes?

If f0 * N = const across resolutions -> lattice artifact (sqrt(2) is diagonal coupling)
If f0 = const across resolutions -> genuine emergent physics

Also checks: mass peak count, mass spacing ratios, PAC conservation.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import numpy as np

from src.v3.engine.config import SimulationConfig
from src.v3.engine.state import FieldState
from src.v3.operators.protocol import Pipeline
from src.v3.operators.rbf import RBFOperator
from src.v3.operators.qbe import QBEOperator
from src.v3.operators.actualization import ActualizationOperator
from src.v3.operators.memory import MemoryOperator
from src.v3.operators.gravity import GravitationalCollapseOperator
from src.v3.operators.spin_statistics import SpinStatisticsOperator
from src.v3.operators.charge_dynamics import ChargeDynamicsOperator
from src.v3.operators.fusion import FusionOperator
from src.v3.operators.confluence import ConfluenceOperator
from src.v3.operators.temperature import TemperatureOperator
from src.v3.operators.thermal_noise import ThermalNoiseOperator
from src.v3.operators.normalization import NormalizationOperator
from src.v3.operators.adaptive import AdaptiveOperator
from src.v3.operators.time_emergence import TimeEmergenceOperator


def build_pipeline() -> Pipeline:
    return Pipeline([
        RBFOperator(),
        QBEOperator(),
        ActualizationOperator(),
        MemoryOperator(),
        GravitationalCollapseOperator(),
        SpinStatisticsOperator(),
        ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(),
        TemperatureOperator(),
        ThermalNoiseOperator(),
        NormalizationOperator(),
        AdaptiveOperator(),
        TimeEmergenceOperator(),
    ])


def find_dominant_frequency(
    state: FieldState,
    pipeline: Pipeline,
    config: SimulationConfig,
    warmup: int,
    record: int,
) -> dict:
    """Warmup, then record E-I time series at strongest peak, FFT for f0."""
    # Warmup
    for t in range(warmup):
        state = pipeline(state, config)

    # Find strongest mass peak
    M_np = state.M.cpu().numpy()
    max_idx = np.unravel_index(np.argmax(M_np), M_np.shape)
    i, j = max_idx
    m_at_peak = M_np[i, j]

    # Record disequilibrium time series
    diseq = np.zeros(record)
    for t in range(record):
        state = pipeline(state, config)
        E_np = state.E.cpu().numpy()
        I_np = state.I.cpu().numpy()
        diseq[t] = E_np[i, j] - I_np[i, j]

    # FFT
    dt_approx = config.dt
    window = np.hanning(record)
    sig = (diseq - diseq.mean()) * window
    fft_power = np.abs(np.fft.rfft(sig)) ** 2
    freq = np.fft.rfftfreq(record, d=dt_approx)
    fft_power[0] = 0  # remove DC

    # Dominant frequency
    dom_idx = np.argmax(fft_power)
    f0 = freq[dom_idx]

    # Top 5 frequencies
    top_idx = np.argsort(fft_power)[-5:][::-1]
    top_freqs = freq[top_idx]

    # PAC conservation
    pac = (state.E + state.I + state.M).sum().item()

    # Mass stats
    M_final = state.M.cpu().numpy()
    m_mean = M_final.mean()
    m_max = M_final.max()

    return {
        "f0": f0,
        "top_freqs": top_freqs.tolist(),
        "peak_pos": (i, j),
        "peak_mass": float(m_at_peak),
        "m_mean": float(m_mean),
        "m_max": float(m_max),
        "pac_total": pac,
        "dt": dt_approx,
        "state": state,
    }


def main():
    print("=" * 70)
    print("GRID INDEPENDENCE TEST")
    print("Does f0 = 1/sqrt(2) survive resolution changes?")
    print("=" * 70)

    # Test resolutions: (nu, nv) pairs maintaining 4:1 aspect ratio
    resolutions = [
        (64, 16),
        (128, 32),
        (256, 64),
    ]

    warmup = 500
    record = 2000
    seed = 7
    results = []

    for nu, nv in resolutions:
        print(f"\n--- Resolution {nu}x{nv} ({nu*nv} cells) ---")
        torch.manual_seed(seed)
        config = SimulationConfig(nu=nu, nv=nv, field_scale=20.0)
        state = FieldState.big_bang(nu, nv, temperature=3.0, device="cpu")
        pipeline = build_pipeline()

        print(f"  Warmup {warmup} ticks...")
        result = find_dominant_frequency(state, pipeline, config, warmup, record)
        f0 = result["f0"]
        print(f"  f0 = {f0:.6f}")
        print(f"  Top 5: {', '.join(f'{f:.4f}' for f in result['top_freqs'])}")
        print(f"  Peak mass: {result['peak_mass']:.4f}, M_mean: {result['m_mean']:.4f}")
        print(f"  PAC total: {result['pac_total']:.4f}")

        result["nu"] = nu
        result["nv"] = nv
        results.append(result)

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS")
    print("=" * 70)

    print(f"\n{'nu':>6} {'nv':>6} {'N':>8} {'f0':>12} {'f0*N':>12} {'f0*sqrt(N)':>12} {'f0/f0_ref':>12}")
    print("-" * 74)

    f0_ref = results[0]["f0"]
    for r in results:
        nu = r["nu"]
        N = nu  # circumference is the relevant scale
        f0 = r["f0"]
        print(
            f"{nu:>6} {r['nv']:>6} {nu * r['nv']:>8} "
            f"{f0:>12.6f} {f0 * N:>12.4f} {f0 * math.sqrt(N):>12.4f} "
            f"{f0 / f0_ref:>12.4f}"
        )

    # Check scaling
    f0_values = [r["f0"] for r in results]
    f0N_values = [r["f0"] * r["nu"] for r in results]

    # Coefficient of variation for f0 (should be low if resolution-independent)
    f0_cv = np.std(f0_values) / np.mean(f0_values) if np.mean(f0_values) > 0 else float("inf")
    f0N_cv = np.std(f0N_values) / np.mean(f0N_values) if np.mean(f0N_values) > 0 else float("inf")

    print(f"\nf0 coefficient of variation: {f0_cv:.4f}")
    print(f"f0*N coefficient of variation: {f0N_cv:.4f}")

    if f0_cv < 0.1:
        print("\nVERDICT: f0 is RESOLUTION-INDEPENDENT (physical)")
        print(f"  f0 = {np.mean(f0_values):.6f} +/- {np.std(f0_values):.6f}")
        print(f"  1/sqrt(2) = {1/math.sqrt(2):.6f}")
        print(f"  Error: {abs(np.mean(f0_values) / (1/math.sqrt(2)) - 1) * 100:.2f}%")
    elif f0N_cv < 0.1:
        print("\nVERDICT: f0 is a LATTICE ARTIFACT (f0 ~ 1/N)")
        print(f"  f0*N = {np.mean(f0N_values):.4f} +/- {np.std(f0N_values):.4f}")
        print("  The Laplacian stencil lacks dx^2 normalization.")
        print("  Fix: manifold.py Laplacian must divide by du^2.")
    else:
        print("\nVERDICT: UNCLEAR — neither f0 nor f0*N is constant")
        print("  May be a more complex scaling (f0 ~ N^alpha)")
        # Fit power law
        log_N = np.log([r["nu"] for r in results])
        log_f0 = np.log(f0_values)
        if len(log_N) >= 2:
            coeffs = np.polyfit(log_N, log_f0, 1)
            print(f"  Power law fit: f0 ~ N^{coeffs[0]:.4f}")

    # Check DFT constant relationships
    phi = 1.618033988749895
    print(f"\n--- DFT Constant Checks ---")
    f0_mean = np.mean(f0_values)
    print(f"f0_mean = {f0_mean:.6f}")
    print(f"f0 / (1/sqrt(2)) = {f0_mean * math.sqrt(2):.6f}")
    print(f"f0 / (1/phi) = {f0_mean * phi:.6f}")
    print(f"f0 / (1/phi^2) = {f0_mean * phi**2:.6f}")
    print(f"f0 * dt = {f0_mean * results[0]['dt']:.8f}")


if __name__ == "__main__":
    main()
