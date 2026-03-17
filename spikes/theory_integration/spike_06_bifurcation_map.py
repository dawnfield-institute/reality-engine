"""Spike 06: Feigenbaum bifurcation map of simulator dynamics.

Theory source: Feigenbaum closed form (Paper 3), sec_threshold_detection.
    Period-doubling cascades converge at the accumulation point r_inf.
    The Feigenbaum constants have Fibonacci closed forms (delta to 13 sig figs).
    The simulator's operators iterate a discrete map each tick — this IS
    a one-dimensional map iteration in the space of coupling constants.

Hypothesis: The simulator's coupling constant time series may exhibit
    period-doubling behavior as parameters vary. If we can identify the
    bifurcation parameter, we can find the accumulation point where
    ALL constants converge simultaneously (the Feigenbaum edge of chaos).

    We sweep a control parameter (field_scale) and measure the late-time
    variance of coupling constants. At period-1 (stable fixed point),
    variance is zero. At period-2, variance is nonzero but periodic.
    At the accumulation point, variance is maximal but structured.
    Beyond: chaos.

Implementation: For each field_scale value, run 5000 ticks, then sample
    coupling constants every tick for another 2000 ticks. Compute the
    FFT of each constant's time series to detect periodicity.
"""

import math
import torch
import numpy as np
from harness import (
    default_pipeline, default_config, PHI, GAMMA_EM, LN2,
    PHI_INV, PHI_INV2, TARGETS,
)
from src.v3.engine.engine import Engine
import time


def collect_time_series(config, warmup=5000, sample=2000):
    """Run warmup ticks, then collect coupling constant time series."""
    torch.manual_seed(42)
    pipeline = default_pipeline()
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    # Warmup
    for _ in range(warmup):
        engine.tick()

    # Collect
    series = {name: [] for name in TARGETS}
    for _ in range(sample):
        engine.tick()
        m = engine.state.metrics
        for name, (key, _) in TARGETS.items():
            series[name].append(m.get(key, 0))

    return series


def analyze_periodicity(series, name):
    """Detect dominant period in time series using FFT."""
    x = np.array(series)
    x = x - x.mean()
    if np.std(x) < 1e-10:
        return {"period": 0, "power": 0, "variance": 0}

    fft = np.fft.rfft(x)
    power = np.abs(fft[1:])  # skip DC
    freqs = np.fft.rfftfreq(len(x))[1:]

    if len(power) == 0:
        return {"period": 0, "power": 0, "variance": float(np.var(x))}

    peak_idx = np.argmax(power)
    peak_freq = freqs[peak_idx]
    peak_period = 1.0 / peak_freq if peak_freq > 0 else 0

    return {
        "period": peak_period,
        "power": float(power[peak_idx]),
        "variance": float(np.var(x)),
        "mean": float(np.mean(series)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("  SPIKE 06: Feigenbaum Bifurcation Map")
    print(f"  Device: {device} | Grid: 128x64")
    print(f"  Warmup: 5000 ticks | Sample: 2000 ticks")
    print("=" * 80)

    # Sweep field_scale as bifurcation parameter
    # Default is 20.0. Sweep from 5 to 50.
    scale_values = [5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    print(f"\n  {'scale':>6s} | {'f_var':>8s} {'f_per':>6s} | "
          f"{'gam_var':>8s} {'gam_per':>6s} | "
          f"{'G_var':>8s} {'G_per':>6s} | "
          f"{'f_mean':>7s} {'gam_mean':>8s} {'G_mean':>7s}")
    print(f"  {'-'*6} | {'-'*8} {'-'*6} | {'-'*8} {'-'*6} | "
          f"{'-'*8} {'-'*6} | {'-'*7} {'-'*8} {'-'*7}")

    all_results = []
    for scale in scale_values:
        config = default_config(device=device, field_scale=scale)
        t0 = time.time()
        series = collect_time_series(config)
        elapsed = time.time() - t0

        results = {}
        for name in ["f_local", "gamma", "G_local"]:
            results[name] = analyze_periodicity(series[name], name)

        all_results.append((scale, results))

        f = results["f_local"]
        g = results["gamma"]
        G = results["G_local"]
        print(f"  {scale:>6.1f} | {f['variance']:>8.2e} {f['period']:>6.1f} | "
              f"{g['variance']:>8.2e} {g['period']:>6.1f} | "
              f"{G['variance']:>8.2e} {G['period']:>6.1f} | "
              f"{f.get('mean',0):>7.4f} {g.get('mean',0):>8.4f} {G.get('mean',0):>7.4f} "
              f"  [{elapsed:.0f}s]")

    # Summary: look for bifurcation signature
    print(f"\n  BIFURCATION ANALYSIS")
    print(f"  Looking for variance jumps (period-doubling onset)...")

    for name in ["f_local", "gamma", "G_local"]:
        variances = [r[name]["variance"] for _, r in all_results]
        max_var_idx = np.argmax(variances)
        max_scale = scale_values[max_var_idx]
        print(f"  {name}: max variance at scale={max_scale:.1f} "
              f"(var={variances[max_var_idx]:.2e})")

        # Detect jumps (ratio > 10x between adjacent scales)
        for i in range(1, len(variances)):
            if variances[i-1] > 0 and variances[i] / (variances[i-1] + 1e-20) > 10:
                print(f"    JUMP at scale {scale_values[i-1]:.1f} -> {scale_values[i]:.1f}: "
                      f"{variances[i-1]:.2e} -> {variances[i]:.2e}")

    # Check if any constant's time series shows exact period-2
    print(f"\n  PERIOD-2 CHECK (Feigenbaum signature)")
    for scale, results in all_results:
        for name in ["f_local", "gamma", "G_local"]:
            period = results[name]["period"]
            if 1.8 < period < 2.2:
                print(f"    scale={scale:.1f} {name}: period={period:.2f} *** PERIOD-2 ***")


if __name__ == "__main__":
    main()
