"""Spike 08: Resonance frequency detection in field oscillations.

Theory source: Pre-field resonance (foundational/docs/pre_field_recursion_resonance).
    Pre-field states exhibit natural resonance frequency ~0.03 cycles/iteration.
    Resonance-aware evolution gives 5.11x speedup in PAC convergence.

    Also: vCPU empirical validation shows oscillation frequency 0.025 Hz.
    Phase synchronization gives 119x GPU speedup — meaning the natural
    frequency is computationally significant.

Hypothesis: The simulator's fields oscillate at natural frequencies
    determined by the operator pipeline. If these frequencies match
    (or clash with) the natural resonance ~0.03/tick, the coupling
    constants will converge (or drift). Identifying the natural
    frequency could inform dt selection or operator ordering.

    Additionally: if the field oscillations show harmonic structure
    (integer frequency ratios), this validates the pi-harmonic
    phase locking hypothesis.

Implementation: Run 10K ticks, sample E, I, M field means and
    coupling constants every tick. Compute FFT to find dominant
    frequencies. Check for harmonic relationships.
"""

import math
import torch
import numpy as np
from harness import (
    default_pipeline, default_config, TARGETS,
)
from src.v3.engine.engine import Engine
import time


def find_peaks_1d(power, min_prominence=0.1):
    """Simple peak finding in 1D power spectrum."""
    peaks = []
    max_power = max(power) if len(power) > 0 else 0
    threshold = max_power * min_prominence
    for i in range(1, len(power) - 1):
        if power[i] > power[i-1] and power[i] > power[i+1] and power[i] > threshold:
            peaks.append(i)
    return peaks


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticks = 10000

    print("=" * 80)
    print("  SPIKE 08: Resonance Frequency Detection")
    print(f"  Device: {device} | Grid: 128x64 | {ticks} ticks")
    print(f"  Theory prediction: natural resonance ~0.03 cycles/tick")
    print("=" * 80)

    torch.manual_seed(42)
    config = default_config(device=device)
    pipeline = default_pipeline()
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    # Collect every tick
    field_series = {"E_mean": [], "I_mean": [], "M_mean": [],
                    "E_std": [], "I_std": [], "M_std": [],
                    "diseq_mean": []}
    constant_series = {name: [] for name in TARGETS}
    constant_series["xi_s"] = []

    t0 = time.time()
    for tick in range(1, ticks + 1):
        engine.tick()
        s = engine.state
        m = s.metrics

        field_series["E_mean"].append(s.E.mean().item())
        field_series["I_mean"].append(s.I.mean().item())
        field_series["M_mean"].append(s.M.mean().item())
        field_series["E_std"].append(s.E.std().item())
        field_series["I_std"].append(s.I.std().item())
        field_series["M_std"].append(s.M.std().item())
        field_series["diseq_mean"].append((s.E - s.I).abs().mean().item())

        for name, (key, _) in TARGETS.items():
            constant_series[name].append(m.get(key, 0))
        constant_series["xi_s"].append(m.get("xi_s_mean", 0))

    elapsed = time.time() - t0
    print(f"\n  Data collected: {ticks} samples in {elapsed:.0f}s")

    # FFT analysis
    print(f"\n  FIELD FREQUENCY ANALYSIS")
    print(f"  {'Signal':<15s} {'Freq_1':>10s} {'Per_1':>8s} {'Pwr_1':>8s} | "
          f"{'Freq_2':>10s} {'Per_2':>8s} | {'Freq_3':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*8} | {'-'*10} {'-'*8} | {'-'*10}")

    all_freqs = {}

    for label, data_dict in [("Fields", field_series), ("Constants", constant_series)]:
        for name, raw in data_dict.items():
            x = np.array(raw)
            x = x - x.mean()
            if np.std(x) < 1e-15:
                continue

            fft = np.fft.rfft(x)
            power = np.abs(fft[1:])
            freqs = np.fft.rfftfreq(len(x))[1:]

            peaks = find_peaks_1d(power)
            peak_data = [(freqs[p], 1.0/freqs[p] if freqs[p] > 0 else 0, power[p])
                         for p in peaks]
            peak_data.sort(key=lambda x: -x[2])  # sort by power

            all_freqs[name] = peak_data

            if len(peak_data) >= 3:
                f1, p1, pw1 = peak_data[0]
                f2, p2, _ = peak_data[1]
                f3, _, _ = peak_data[2]
                print(f"  {name:<15s} {f1:>10.6f} {p1:>8.1f} {pw1:>8.1f} | "
                      f"{f2:>10.6f} {p2:>8.1f} | {f3:>10.6f}")
            elif len(peak_data) >= 1:
                f1, p1, pw1 = peak_data[0]
                print(f"  {name:<15s} {f1:>10.6f} {p1:>8.1f} {pw1:>8.1f}")
            else:
                print(f"  {name:<15s} no peaks detected")

    # Harmonic analysis: check for integer ratios between dominant frequencies
    print(f"\n  HARMONIC RELATIONSHIPS")
    print(f"  Checking for integer/golden-ratio frequency ratios...")

    key_signals = ["E_mean", "I_mean", "M_mean", "f_local", "gamma", "G_local"]
    for i, sig_a in enumerate(key_signals):
        for sig_b in key_signals[i+1:]:
            if sig_a not in all_freqs or sig_b not in all_freqs:
                continue
            if not all_freqs[sig_a] or not all_freqs[sig_b]:
                continue
            f_a = all_freqs[sig_a][0][0]
            f_b = all_freqs[sig_b][0][0]
            if f_a < 1e-10 or f_b < 1e-10:
                continue
            ratio = max(f_a, f_b) / min(f_a, f_b)
            # Check if ratio is near an integer
            nearest_int = round(ratio)
            int_err = abs(ratio - nearest_int) / nearest_int * 100 if nearest_int > 0 else 999
            # Check if ratio is near phi
            phi = (1 + math.sqrt(5)) / 2
            phi_err = abs(ratio - phi) / phi * 100
            # Check if ratio is near pi/2
            pi2_err = abs(ratio - math.pi / 2) / (math.pi / 2) * 100

            marker = ""
            if int_err < 5:
                marker = f" *** {nearest_int}:1 ***"
            elif phi_err < 5:
                marker = " *** phi ***"
            elif pi2_err < 5:
                marker = " *** pi/2 ***"

            if marker or ratio < 5:
                print(f"  {sig_a}/{sig_b}: ratio={ratio:.4f}{marker}")

    # Check if natural frequency matches theory prediction
    print(f"\n  THEORY COMPARISON")
    print(f"  Predicted natural resonance: ~0.03 cycles/tick")
    for name in ["E_mean", "I_mean", "f_local", "gamma"]:
        if name in all_freqs and all_freqs[name]:
            f_dom = all_freqs[name][0][0]
            print(f"  {name}: dominant freq = {f_dom:.6f} cycles/tick "
                  f"(ratio to 0.03: {f_dom/0.03:.3f})")

    print(f"\n  DONE")


if __name__ == "__main__":
    main()
