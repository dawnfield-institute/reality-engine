"""Spectral analysis v2 — mass-diverse, disequilibrium-focused.

Key improvements over v1:
- Short warmup (500 ticks) to catch structures BEFORE cap pileup
- Tracks E-I disequilibrium oscillation (the "quantum" vibration), not just M
- Bins peaks by mass to find frequency vs mass relationship
- Multiple snapshot windows to see how spectrum evolves
- Analyzes both the universal mode AND mass-dependent modes
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


PHI = 1.618033988749895


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


def find_all_peaks(M: torch.Tensor, min_mass: float = 0.1, min_distance: int = 3):
    """Find ALL local maxima in the mass field, across all mass scales."""
    M_np = M.cpu().numpy()
    peaks = []
    for i in range(min_distance, M_np.shape[0] - min_distance):
        for j in range(min_distance, M_np.shape[1] - min_distance):
            val = M_np[i, j]
            if val < min_mass:
                continue
            window = M_np[
                i - min_distance : i + min_distance + 1,
                j - min_distance : j + min_distance + 1,
            ]
            if val >= window.max():
                peaks.append((i, j, val))
    peaks.sort(key=lambda x: -x[2])
    return peaks


def analyze_window(
    state: FieldState,
    pipeline: Pipeline,
    config: SimulationConfig,
    window_ticks: int,
    label: str,
):
    """Record and analyze one spectral window."""
    print(f"\n{'=' * 70}")
    print(f"SPECTRAL WINDOW: {label}")
    print(f"{'=' * 70}")

    # Find peaks at START of window
    peaks = find_all_peaks(state.M, min_mass=0.1, min_distance=3)
    if not peaks:
        print("  No mass peaks found!")
        return state

    # Mass distribution
    masses = np.array([p[2] for p in peaks])
    print(f"\nMass distribution: {len(peaks)} peaks")
    print(f"  Range: [{masses.min():.4f}, {masses.max():.4f}]")
    print(f"  Mean: {masses.mean():.4f}, Std: {masses.std():.4f}")

    # Bin by mass
    M_cap = config.field_scale / 5.0
    bins = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, M_cap - 0.1, M_cap + 0.1]
    bin_labels = []
    for k in range(len(bins) - 1):
        count = np.sum((masses >= bins[k]) & (masses < bins[k + 1]))
        if count > 0:
            bin_labels.append(f"  [{bins[k]:.1f}, {bins[k+1]:.1f}): {count} peaks")
    print("Mass bins:")
    for bl in bin_labels:
        print(bl)

    # Select representative peaks: up to 3 from each mass bin
    selected = []
    for k in range(len(bins) - 1):
        bin_peaks = [p for p in peaks if bins[k] <= p[2] < bins[k + 1]]
        selected.extend(bin_peaks[:3])

    if len(selected) > 30:
        selected = selected[:30]

    print(f"\nTracking {len(selected)} representative peaks across mass range")

    # Record time series
    n_peaks = len(selected)
    E_series = np.zeros((n_peaks, window_ticks))
    I_series = np.zeros((n_peaks, window_ticks))
    M_series = np.zeros((n_peaks, window_ticks))
    diseq_series = np.zeros((n_peaks, window_ticks))

    for t in range(window_ticks):
        state = pipeline(state, config)
        E_np = state.E.cpu().numpy()
        I_np = state.I.cpu().numpy()
        M_np = state.M.cpu().numpy()

        for p, (i, j, _) in enumerate(selected):
            E_series[p, t] = E_np[i, j]
            I_series[p, t] = I_np[i, j]
            M_series[p, t] = M_np[i, j]
            diseq_series[p, t] = E_np[i, j] - I_np[i, j]

        if (t + 1) % 250 == 0:
            print(f"  recorded tick {t + 1}/{window_ticks}")

    # FFT analysis
    dt_approx = config.dt
    freq = np.fft.rfftfreq(window_ticks, d=dt_approx)
    window_fn = np.hanning(window_ticks)

    print(f"\n--- Spectral Results ---")
    print(f"{'Peak':>4} {'(i,j)':>8} {'M_init':>8} {'M_mean':>8} "
          f"{'f_dom':>10} {'f_diseq':>10} {'Power':>10} {'Harmonics':>30}")
    print("-" * 100)

    results = []

    for p, (i, j, m_init) in enumerate(selected):
        m_mean = np.mean(M_series[p])

        # Disequilibrium spectrum (the "quantum oscillation")
        diseq_sig = diseq_series[p] - np.mean(diseq_series[p])
        diseq_sig *= window_fn
        diseq_fft = np.abs(np.fft.rfft(diseq_sig)) ** 2
        diseq_fft[0] = 0

        # E spectrum
        E_sig = E_series[p] - np.mean(E_series[p])
        E_sig *= window_fn
        E_fft = np.abs(np.fft.rfft(E_sig)) ** 2
        E_fft[0] = 0

        # Combined
        combined = E_fft + diseq_fft

        # Dominant frequencies
        top_idx = np.argsort(combined)[-5:][::-1]
        top_freqs = freq[top_idx]
        top_powers = combined[top_idx]

        # Dominant diseq frequency
        diseq_top_idx = np.argmax(diseq_fft)
        f_diseq = freq[diseq_top_idx]

        f_dom = top_freqs[0]
        power = top_powers[0]

        # Harmonic ratios
        if f_dom > 0:
            ratios = top_freqs[top_freqs > 0] / f_dom
            harm_str = ", ".join(f"{r:.2f}" for r in ratios[:5])
        else:
            harm_str = "n/a"

        print(f"{p:>4} ({i:>3},{j:>3}) {m_init:>8.4f} {m_mean:>8.4f} "
              f"{f_dom:>10.4f} {f_diseq:>10.4f} {power:>10.2e} {harm_str:>30}")

        results.append({
            "peak": (i, j),
            "m_init": m_init,
            "m_mean": m_mean,
            "f_dom": f_dom,
            "f_diseq": f_diseq,
            "power": power,
            "top_freqs": top_freqs,
        })

    # --- f vs M analysis ---
    print(f"\n--- Frequency vs Mass ---")
    masses_arr = np.array([r["m_mean"] for r in results])
    freqs_arr = np.array([r["f_dom"] for r in results])
    diseq_freqs = np.array([r["f_diseq"] for r in results])

    valid = freqs_arr > 0
    if valid.sum() >= 2:
        m_v = masses_arr[valid]
        f_v = freqs_arr[valid]
        fd_v = diseq_freqs[valid]

        # Check if all frequencies are the same (universal mode)
        f_unique = np.unique(np.round(f_v, 4))
        if len(f_unique) == 1:
            print(f"UNIVERSAL MODE: all peaks oscillate at f = {f_unique[0]:.6f}")
            print(f"  This is the system clock / pipeline frequency")
            print(f"  f/M ratios: {', '.join(f'{f_v[k]/m_v[k]:.4f}' for k in range(min(10, len(f_v))))}")
        else:
            print(f"DISCRETE SPECTRUM: {len(f_unique)} distinct frequencies found!")
            for fu in sorted(f_unique):
                mask = np.abs(f_v - fu) < 0.01
                m_at_f = m_v[mask]
                print(f"  f={fu:.6f}: {mask.sum()} peaks, M in [{m_at_f.min():.3f}, {m_at_f.max():.3f}]")

            # f vs M correlation
            if (m_v > 0).all() and (f_v > 0).all():
                corr = np.corrcoef(m_v, f_v)[0, 1]
                print(f"\nCorrelation(M, f) = {corr:.4f}")

                # Power law
                log_coeffs = np.polyfit(np.log(m_v), np.log(f_v), 1)
                print(f"Power law: f = {np.exp(log_coeffs[1]):.4f} * M^{log_coeffs[0]:.4f}")

        # Check diseq frequencies separately
        fd_unique = np.unique(np.round(fd_v, 4))
        if len(fd_unique) > 1:
            print(f"\nDisequilibrium frequencies: {len(fd_unique)} distinct modes")
            for fu in sorted(fd_unique)[:10]:
                mask = np.abs(fd_v - fu) < 0.01
                m_at_f = m_v[mask]
                if len(m_at_f) > 0:
                    print(f"  f_diseq={fu:.6f}: {mask.sum()} peaks, M_mean={m_at_f.mean():.4f}")

    # --- Check frequency against golden ratio ---
    print(f"\n--- Golden Ratio Check ---")
    if valid.sum() > 0:
        f_base = freqs_arr[valid][0]
        print(f"Fundamental frequency: {f_base:.6f}")
        print(f"f / phi = {f_base / PHI:.6f}")
        print(f"f * phi = {f_base * PHI:.6f}")
        print(f"f / phi^2 = {f_base / PHI**2:.6f}")
        print(f"2*pi*f = {2 * np.pi * f_base:.6f}")

        # Check if f relates to dt
        print(f"\ndt = {config.dt}")
        print(f"1/dt = {1.0/config.dt:.4f}")
        print(f"f * dt = {f_base * config.dt:.6f}")
        print(f"f / (1/dt) = {f_base * config.dt:.6f}")

    return state


def main():
    print("=" * 70)
    print("SPECTRAL ANALYSIS v2 — Mass-Diverse, Disequilibrium-Focused")
    print("=" * 70)

    seed = 7
    torch.manual_seed(seed)
    nu, nv = 128, 32
    config = SimulationConfig(nu=nu, nv=nv, field_scale=20.0)
    state = FieldState.big_bang(nu, nv, temperature=3.0, device="cpu")
    pipeline = build_pipeline()

    # Three spectral windows at different evolution stages:
    # Window 1: ticks 300-800 (early — diverse masses, before cap pileup)
    # Window 2: ticks 800-1300 (mid — some cap pileup starting)
    # Window 3: ticks 1300-1800 (late — more structures at cap)

    print("\nPhase 1: Initial warmup (300 ticks)...")
    for t in range(300):
        state = pipeline(state, config)
        if (t + 1) % 100 == 0:
            M = state.M
            print(f"  tick {t+1}: M_mean={M.mean().item():.4f}, M_max={M.max().item():.4f}")

    state = analyze_window(state, pipeline, config, 500, "EARLY (ticks 300-800)")

    print("\nContinuing evolution...")
    state = analyze_window(state, pipeline, config, 500, "MID (ticks 800-1300)")

    print("\nContinuing evolution...")
    state = analyze_window(state, pipeline, config, 500, "LATE (ticks 1300-1800)")

    # --- Global summary ---
    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)
    print("Key question: does the oscillation frequency depend on mass?")
    print("If f = const for all M -> system has a single clock (pipeline frequency)")
    print("If f ~ M -> emergent E=hf=mc^2")
    print("If f ~ 1/M -> emergent uncertainty (heavy structures oscillate slower)")
    print("If f ~ sqrt(M) -> emergent harmonic oscillator")
    print("If discrete f values -> emergent quantum levels")


if __name__ == "__main__":
    main()
