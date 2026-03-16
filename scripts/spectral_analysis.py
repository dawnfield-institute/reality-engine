"""Spectral analysis of emergent structures in Reality Engine v3.

Finds resonant frequencies of stable mass peaks by:
1. Warmup 2000 ticks with full pipeline (including SpinStatistics + ChargeDynamics)
2. Identify stable mass peaks (local maxima)
3. Record E, I, M time series at each peak for 2000 ticks
4. FFT to find power spectrum and dominant frequencies
5. Analyze f vs M relationship (looking for E=hf=mc^2 behavior)
6. Check for harmonic structure (integer/half-integer frequency ratios)
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from src.v3.engine.engine import Engine
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
    """Full pipeline with correct ordering."""
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


def find_mass_peaks(M: torch.Tensor, min_mass: float = 0.5, min_distance: int = 5):
    """Find local maxima in the mass field."""
    M_np = M.cpu().numpy()
    peaks = []

    for i in range(min_distance, M_np.shape[0] - min_distance):
        for j in range(min_distance, M_np.shape[1] - min_distance):
            val = M_np[i, j]
            if val < min_mass:
                continue
            # Check if local maximum in a window
            window = M_np[
                i - min_distance : i + min_distance + 1,
                j - min_distance : j + min_distance + 1,
            ]
            if val >= window.max():
                peaks.append((i, j, val))

    # Sort by mass descending
    peaks.sort(key=lambda x: -x[2])
    return peaks


def spectral_analysis():
    """Run spectral analysis on emergent structures."""
    print("=" * 70)
    print("SPECTRAL ANALYSIS OF EMERGENT STRUCTURES")
    print("=" * 70)

    # Setup
    seed = 7
    torch.manual_seed(seed)
    nu, nv = 128, 32
    config = SimulationConfig(nu=nu, nv=nv, field_scale=20.0)
    state = FieldState.big_bang(nu, nv, temperature=3.0, device="cpu")
    pipeline = build_pipeline()

    warmup_ticks = 2000
    record_ticks = 2000

    # --- Phase 1: Warmup ---
    print(f"\nPhase 1: Warmup ({warmup_ticks} ticks)...")
    for t in range(warmup_ticks):
        state = pipeline(state, config)
        if (t + 1) % 500 == 0:
            M = state.M
            print(
                f"  tick {t+1}: M_mean={M.mean().item():.4f}, "
                f"M_max={M.max().item():.4f}, "
                f"M_sum={M.sum().item():.2f}"
            )

    # --- Phase 2: Find stable mass peaks ---
    print("\nPhase 2: Finding stable mass peaks...")
    peaks = find_mass_peaks(state.M, min_mass=0.3, min_distance=3)
    print(f"  Found {len(peaks)} peaks")

    if len(peaks) == 0:
        print("  No peaks found! Trying lower threshold...")
        peaks = find_mass_peaks(state.M, min_mass=0.1, min_distance=3)
        print(f"  Found {len(peaks)} peaks with lower threshold")

    if len(peaks) == 0:
        print("  Still no peaks. Dumping M field stats:")
        M = state.M
        print(f"    M range: [{M.min().item():.6f}, {M.max().item():.6f}]")
        print(f"    M mean: {M.mean().item():.6f}")
        print(f"    M nonzero: {(M > 0.01).sum().item()}")
        return

    # Limit to top 20 peaks
    peaks = peaks[:20]
    print(f"  Tracking top {len(peaks)} peaks:")
    for idx, (i, j, m) in enumerate(peaks):
        print(f"    Peak {idx}: ({i},{j}) M={m:.4f}")

    # --- Phase 3: Record time series ---
    print(f"\nPhase 3: Recording time series ({record_ticks} ticks)...")
    n_peaks = len(peaks)
    E_series = np.zeros((n_peaks, record_ticks))
    I_series = np.zeros((n_peaks, record_ticks))
    M_series = np.zeros((n_peaks, record_ticks))

    for t in range(record_ticks):
        state = pipeline(state, config)

        E_np = state.E.cpu().numpy()
        I_np = state.I.cpu().numpy()
        M_np = state.M.cpu().numpy()

        for p, (i, j, _) in enumerate(peaks):
            E_series[p, t] = E_np[i, j]
            I_series[p, t] = I_np[i, j]
            M_series[p, t] = M_np[i, j]

        if (t + 1) % 500 == 0:
            print(f"  tick {warmup_ticks + t + 1}")

    # --- Phase 4: FFT analysis ---
    print("\nPhase 4: Spectral analysis...")
    print("-" * 70)

    dt_approx = config.dt  # approximate, since dt is emergent
    freq = np.fft.rfftfreq(record_ticks, d=dt_approx)

    all_dominant_freqs = []
    all_masses = []

    for p, (i, j, m_init) in enumerate(peaks):
        # Detrend the signals (remove DC + linear trend)
        E_sig = E_series[p] - np.mean(E_series[p])
        I_sig = I_series[p] - np.mean(I_series[p])
        M_sig = M_series[p] - np.mean(M_series[p])

        # Apply Hann window to reduce spectral leakage
        window = np.hanning(record_ticks)
        E_sig = E_sig * window
        I_sig = I_sig * window
        M_sig = M_sig * window

        # FFT
        E_fft = np.abs(np.fft.rfft(E_sig)) ** 2
        I_fft = np.abs(np.fft.rfft(I_sig)) ** 2
        M_fft = np.abs(np.fft.rfft(M_sig)) ** 2

        # Combined power spectrum
        combined = E_fft + I_fft + M_fft

        # Find dominant frequencies (top 5, excluding DC)
        combined[0] = 0  # remove DC
        top_indices = np.argsort(combined)[-5:][::-1]
        top_freqs = freq[top_indices]
        top_powers = combined[top_indices]

        # Mean mass during recording
        m_mean = np.mean(M_series[p])
        all_masses.append(m_mean)

        # Dominant frequency (highest power)
        dom_freq = top_freqs[0]
        dom_power = top_powers[0]
        all_dominant_freqs.append(dom_freq)

        print(f"\nPeak {p} at ({i},{j}) | M_init={m_init:.4f} | M_mean={m_mean:.4f}")
        print(f"  Dominant frequency: f = {dom_freq:.6f} (power={dom_power:.2e})")
        print(f"  Top 5 frequencies: {', '.join(f'{f:.4f}' for f in top_freqs)}")

        # Check for harmonic structure
        if dom_freq > 0:
            ratios = top_freqs / dom_freq
            print(f"  Frequency ratios (to fundamental): {', '.join(f'{r:.3f}' for r in ratios)}")

    # --- Phase 5: f vs M analysis ---
    print("\n" + "=" * 70)
    print("f vs M RELATIONSHIP")
    print("=" * 70)

    all_masses = np.array(all_masses)
    all_dominant_freqs = np.array(all_dominant_freqs)

    # Filter out zero-frequency peaks
    valid = all_dominant_freqs > 0
    if valid.sum() < 2:
        print("Too few peaks with nonzero frequency for regression.")
    else:
        m_valid = all_masses[valid]
        f_valid = all_dominant_freqs[valid]

        # Linear fit: f = a * M + b
        coeffs = np.polyfit(m_valid, f_valid, 1)
        print(f"\nLinear fit: f = {coeffs[0]:.6f} * M + {coeffs[1]:.6f}")
        residuals = f_valid - np.polyval(coeffs, m_valid)
        r2 = 1.0 - np.sum(residuals**2) / np.sum((f_valid - f_valid.mean()) ** 2)
        print(f"R^2 = {r2:.4f}")

        # Power law fit: log(f) = a * log(M) + b => f = exp(b) * M^a
        if (m_valid > 0).all() and (f_valid > 0).all():
            log_coeffs = np.polyfit(np.log(m_valid), np.log(f_valid), 1)
            print(f"\nPower law fit: f = {np.exp(log_coeffs[1]):.6f} * M^{log_coeffs[0]:.4f}")
            log_residuals = np.log(f_valid) - np.polyval(log_coeffs, np.log(m_valid))
            log_r2 = 1.0 - np.sum(log_residuals**2) / np.sum(
                (np.log(f_valid) - np.log(f_valid).mean()) ** 2
            )
            print(f"R^2 (log-log) = {log_r2:.4f}")

        print(f"\nData points (M, f):")
        for m, f in sorted(zip(m_valid, f_valid)):
            ratio = f / m if m > 0 else float("inf")
            print(f"  M={m:.4f}  f={f:.6f}  f/M={ratio:.4f}")

    # --- Phase 6: Cross-peak frequency ratios ---
    print("\n" + "=" * 70)
    print("CROSS-PEAK FREQUENCY RATIOS")
    print("=" * 70)

    valid_freqs = all_dominant_freqs[valid]
    valid_masses_sorted = all_masses[valid]

    if len(valid_freqs) >= 2:
        # Sort by frequency
        sort_idx = np.argsort(valid_freqs)
        sorted_freqs = valid_freqs[sort_idx]
        f_base = sorted_freqs[0]

        print(f"\nBase frequency: {f_base:.6f}")
        print(f"Frequency ratios to base:")
        for idx in sort_idx:
            f = all_dominant_freqs[idx]
            m = all_masses[idx]
            ratio = f / f_base
            # Check proximity to simple fractions
            nearest_half = round(ratio * 2) / 2
            err = abs(ratio - nearest_half) / nearest_half * 100 if nearest_half > 0 else 0
            print(
                f"  f={f:.6f}  M={m:.4f}  ratio={ratio:.4f}  "
                f"nearest_half_int={nearest_half:.1f}  err={err:.1f}%"
            )

    # --- Phase 7: Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Peaks analyzed: {n_peaks}")
    print(f"Peaks with resonant frequency: {valid.sum()}")
    if valid.sum() > 0:
        print(f"Frequency range: [{all_dominant_freqs[valid].min():.6f}, {all_dominant_freqs[valid].max():.6f}]")
        print(f"Mass range: [{all_masses[valid].min():.4f}, {all_masses[valid].max():.4f}]")

        # Check for discrete spectrum
        if len(valid_freqs) >= 3:
            freq_diffs = np.diff(np.sort(valid_freqs))
            if freq_diffs.std() / freq_diffs.mean() < 0.3:
                print(
                    f"DISCRETE SPECTRUM detected: frequency spacing "
                    f"{freq_diffs.mean():.6f} +/- {freq_diffs.std():.6f}"
                )
            else:
                print("Frequency spectrum is NOT evenly spaced (complex harmonic structure)")


if __name__ == "__main__":
    spectral_analysis()
