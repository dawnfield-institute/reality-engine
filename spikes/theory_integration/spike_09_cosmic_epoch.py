"""Spike 09: Cosmic Epoch — coupling constant running curves and "you are here."

The drift isn't a bug. It's emergent RG flow.

If coupling constants emerge from E/I/M field dynamics, they SHOULD change
as the universe evolves. A young universe (high disequilibrium, low mass)
has different effective couplings than an old thermalized one.

This spike:
1. Runs a long simulation (20K ticks) with fine-grained logging (every 50 ticks)
2. Plots the "running curve" of each Tier 1 constant vs tick (= cosmic epoch)
3. Finds the "NOW" tick — where combined error across ALL constants is minimized
4. Computes the drift rate (beta function analogue) at that tick
5. Checks whether constants drift coherently (correlated) or independently
6. Extracts the "future prediction" — what do the constants become later?

If the drift is physics:
    - Constants should drift COHERENTLY (driven by same underlying M growth)
    - The "NOW" tick should be well-defined (sharp minimum in combined error)
    - The drift rate should be small near NOW (we're near equilibrium)
    - Earlier ticks = high-redshift physics, later ticks = far future

If the drift is numerical:
    - Constants should drift independently
    - No clear "NOW" tick
    - Drift rate random
"""

import math
import os
import sys
import time

import numpy as np

re_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if re_path not in sys.path:
    sys.path.insert(0, re_path)

import torch

from harness import (
    default_pipeline, default_config, TARGETS,
    PHI, GAMMA_EM, LN2, PHI_INV, PHI_INV2,
    find_mass_peaks, phi2_spacing_err,
)
from src.v3.engine.engine import Engine


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticks = 20000
    sample_every = 50

    print("=" * 90)
    print("  SPIKE 09: Cosmic Epoch — Running Curves and 'You Are Here'")
    print(f"  Device: {device} | Grid: 128x64 | {ticks} ticks | sample every {sample_every}")
    print("=" * 90)

    torch.manual_seed(42)
    config = default_config(device=device)
    pipeline = default_pipeline()
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    # Collect time series
    target_vals = {name: target for name, (_, target) in TARGETS.items()}
    series = {name: [] for name in TARGETS}
    series["M_mean"] = []
    series["M_max"] = []
    series["xi_s"] = []
    series["entropy"] = []
    tick_stamps = []

    t0 = time.time()
    for tick in range(1, ticks + 1):
        engine.tick()
        if tick % sample_every == 0:
            m = engine.state.metrics
            for name, (key, _) in TARGETS.items():
                series[name].append(m.get(key, 0))
            series["M_mean"].append(engine.state.M.mean().item())
            series["M_max"].append(engine.state.M.max().item())
            series["xi_s"].append(m.get("xi_s_mean", 0))
            series["entropy"].append(m.get("entropy_reduction_cumulative", 0))
            tick_stamps.append(tick)

            if tick % 5000 == 0:
                elapsed = time.time() - t0
                errs = {name: abs(series[name][-1] - target_vals[name]) / abs(target_vals[name]) * 100
                        for name in TARGETS}
                avg = sum(errs.values()) / len(errs)
                print(f"  Tick {tick:>6d} ({elapsed:>5.0f}s): avg_err={avg:.1f}%  "
                      f"M_mean={series['M_mean'][-1]:.3f}")

    elapsed = time.time() - t0
    n_samples = len(tick_stamps)
    print(f"\n  Collection complete: {n_samples} samples in {elapsed:.0f}s")

    # Convert to numpy
    for k in series:
        series[k] = np.array(series[k])
    tick_stamps = np.array(tick_stamps)

    # ====================================================================
    # 1. RUNNING CURVES — error vs tick for each constant
    # ====================================================================
    print(f"\n  RUNNING CURVES (% error vs tick)")
    print(f"  {'Tick':>6s} {'f%':>7s} {'gam%':>7s} {'alp%':>7s} {'G%':>7s} "
          f"{'lam%':>7s} {'AVG%':>7s} {'M_mean':>7s}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    # Compute errors at every sample
    errors = {}
    for name in TARGETS:
        errors[name] = np.abs(series[name] - target_vals[name]) / np.abs(target_vals[name]) * 100
    avg_errors = np.mean([errors[name] for name in TARGETS], axis=0)

    # Print at checkpoints
    checkpoints = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000,
                   6000, 7000, 8000, 10000, 12000, 15000, 20000]
    for cp in checkpoints:
        idx = cp // sample_every - 1
        if idx < 0 or idx >= n_samples:
            continue
        print(f"  {cp:>6d} {errors['f_local'][idx]:>6.1f}% {errors['gamma'][idx]:>6.1f}% "
              f"{errors['alpha'][idx]:>6.1f}% {errors['G_local'][idx]:>6.1f}% "
              f"{errors['lambda'][idx]:>6.1f}% {avg_errors[idx]:>6.1f}% "
              f"{series['M_mean'][idx]:>7.3f}")

    # ====================================================================
    # 2. FIND "NOW" TICK — minimum combined error
    # ====================================================================
    now_idx = np.argmin(avg_errors)
    now_tick = tick_stamps[now_idx]
    now_err = avg_errors[now_idx]

    print(f"\n  'YOU ARE HERE' TICK: {now_tick}")
    print(f"  Combined error at NOW: {now_err:.2f}%")
    for name in TARGETS:
        val = series[name][now_idx]
        target = target_vals[name]
        err = errors[name][now_idx]
        print(f"    {name:<12s}: {val:.6f}  (target {target:.6f}, error {err:.2f}%)")
    print(f"    M_mean:       {series['M_mean'][now_idx]:.4f}")
    print(f"    M_max:        {series['M_max'][now_idx]:.4f}")

    # ====================================================================
    # 3. DRIFT RATE (beta function analogue) at NOW
    # ====================================================================
    print(f"\n  DRIFT RATES AT NOW (beta functions)")
    print(f"  Positive = constant increasing, negative = decreasing")

    window = 10  # samples for local derivative
    if now_idx > window and now_idx < n_samples - window:
        for name in TARGETS:
            # Local slope (per tick)
            y_before = series[name][now_idx - window]
            y_after = series[name][now_idx + window]
            dt = (tick_stamps[now_idx + window] - tick_stamps[now_idx - window])
            slope = (y_after - y_before) / dt
            # Fractional rate (per tick)
            frac_rate = slope / target_vals[name]
            print(f"    d({name})/dt = {slope:+.2e}  "
                  f"(fractional: {frac_rate:+.2e} per tick)")

    # ====================================================================
    # 4. COHERENCE CHECK — do constants drift together?
    # ====================================================================
    print(f"\n  COHERENCE CHECK (are constants driven by the same physics?)")

    # Compute pairwise correlations of the ERROR trajectories (not values)
    err_matrix = np.array([errors[name] for name in TARGETS])
    names = list(TARGETS.keys())

    # Correlation of error trajectories
    corr = np.corrcoef(err_matrix)
    print(f"  {'':>12s}", end="")
    for n in names:
        print(f" {n:>8s}", end="")
    print()
    for i, n1 in enumerate(names):
        print(f"  {n1:>12s}", end="")
        for j, n2 in enumerate(names):
            print(f" {corr[i,j]:>+8.3f}", end="")
        print()

    # Overall coherence: mean absolute pairwise correlation
    n = len(names)
    abs_corrs = [abs(corr[i,j]) for i in range(n) for j in range(i+1, n)]
    mean_coherence = np.mean(abs_corrs)
    print(f"\n  Mean absolute pairwise correlation: {mean_coherence:.3f}")
    if mean_coherence > 0.7:
        print(f"  *** HIGHLY COHERENT — drift is driven by common physics ***")
    elif mean_coherence > 0.4:
        print(f"  *** MODERATELY COHERENT — partially common driver ***")
    else:
        print(f"  *** LOW COHERENCE — drift may be independent/numerical ***")

    # ====================================================================
    # 5. EPOCH CHARACTERIZATION
    # ====================================================================
    print(f"\n  EPOCH CHARACTERIZATION")
    print(f"  {'Epoch':<20s} {'Ticks':>12s} {'avg_err%':>8s} {'M_mean':>7s} {'Description'}")
    print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*7} {'-'*30}")

    # Define epochs by M_mean
    m_vals = series["M_mean"]
    epochs = [
        ("Big Bang", 0, np.argmax(m_vals > 0.1)),
        ("Inflation", np.argmax(m_vals > 0.1), np.argmax(m_vals > 0.5)),
        ("Recombination", np.argmax(m_vals > 0.5), np.argmax(m_vals > 1.0)),
        ("Present epoch", max(0, now_idx - 20), min(n_samples-1, now_idx + 20)),
        ("Heat death", n_samples - 40, n_samples - 1),
    ]

    for label, start, end in epochs:
        if start >= end or start >= n_samples or end >= n_samples:
            continue
        tick_range = f"{tick_stamps[start]}-{tick_stamps[end]}"
        avg_err_epoch = np.mean(avg_errors[start:end+1])
        m_mean_epoch = np.mean(m_vals[start:end+1])

        desc = ""
        if label == "Present epoch":
            desc = f"<-- NOW (tick {now_tick})"
        elif avg_err_epoch < 10:
            desc = "Constants near attractors"
        elif avg_err_epoch > 20:
            desc = "Constants far from attractors"

        print(f"  {label:<20s} {tick_range:>12s} {avg_err_epoch:>7.1f}% "
              f"{m_mean_epoch:>7.3f} {desc}")

    # ====================================================================
    # 6. FUTURE PREDICTIONS
    # ====================================================================
    if now_idx < n_samples - 1:
        print(f"\n  FUTURE PREDICTIONS (beyond NOW tick {now_tick})")
        print(f"  If 'drift = physics', these are predictions for the far future:")
        last_idx = n_samples - 1
        for name in TARGETS:
            now_val = series[name][now_idx]
            future_val = series[name][last_idx]
            target = target_vals[name]
            change_pct = (future_val - now_val) / now_val * 100
            direction = "increasing" if future_val > now_val else "decreasing"
            print(f"    {name:<12s}: {now_val:.6f} -> {future_val:.6f} "
                  f"({change_pct:+.1f}%, {direction})")

    # ====================================================================
    # 7. OBSERVATIONAL COMPARISON
    # ====================================================================
    print(f"\n  OBSERVATIONAL COMPARISON")
    print(f"  Webb et al. quasar data: |da/a| ~ 10^-5 per ~10 Gyr lookback")

    # Compute fractional alpha change rate at NOW
    if now_idx > window and now_idx < n_samples - window:
        alpha_before = series["alpha"][now_idx - window]
        alpha_after = series["alpha"][now_idx + window]
        dt_ticks = tick_stamps[now_idx + window] - tick_stamps[now_idx - window]
        dalpha_alpha = abs(alpha_after - alpha_before) / series["alpha"][now_idx]

        print(f"  Simulator: |da/a| = {dalpha_alpha:.2e} per {dt_ticks} ticks")
        print(f"  If 1 tick ≈ T_universe/{ticks}, then per-Gyr rate can be calibrated")
        print(f"  Key question: what's the physical time scale of one tick?")

    print(f"\n  DONE")


if __name__ == "__main__":
    main()
