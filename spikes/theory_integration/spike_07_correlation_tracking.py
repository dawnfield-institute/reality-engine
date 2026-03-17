"""Spike 07: Inter-constant correlation tracking over time.

Theory source: M5 exp_11 (anti-correlation discovery).
    alpha and lambda: r = +1.000 (exact)
    f_local and G_local: r = +0.997
    Group 1 (gamma, alpha, lambda) vs Group 2 (f, G): r = -0.98

    PAC conservation FORCES this correlation structure. The question is:
    when does the correlation structure break down? If it's maintained
    throughout the run, the drift is structural (PAC trade-off). If it
    degrades, something is destabilizing the inter-constant coherence.

Hypothesis: Track all 5 coupling constants every 10 ticks for 10K ticks.
    Compute running correlations in windows of 500 ticks. Look for:
    1. When does the alpha-lambda correlation drop below 1.000?
    2. When does the Group 1 vs Group 2 anti-correlation weaken?
    3. Are there periodic modulations in the correlation structure?
    4. Does xi_s (entropy-coherence ratio) predict correlation strength?
"""

import math
import torch
import numpy as np
from harness import (
    default_pipeline, default_config, TARGETS,
    PHI, GAMMA_EM, LN2, PHI_INV, PHI_INV2,
)
from src.v3.engine.engine import Engine
import time


def running_correlation(x, y, window=500):
    """Compute running Pearson correlation."""
    n = len(x)
    corrs = []
    for i in range(window, n):
        xi = x[i-window:i]
        yi = y[i-window:i]
        if np.std(xi) < 1e-15 or np.std(yi) < 1e-15:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(xi, yi)[0, 1]))
    return corrs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticks = 10000
    sample_every = 10
    window = 500  # correlation window in samples (= 5000 ticks)

    print("=" * 80)
    print("  SPIKE 07: Inter-Constant Correlation Tracking")
    print(f"  Device: {device} | Grid: 128x64 | {ticks} ticks")
    print(f"  Sampling every {sample_every} ticks | Correlation window: {window} samples")
    print("=" * 80)

    torch.manual_seed(42)
    config = default_config(device=device)
    pipeline = default_pipeline()
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    # Collect time series
    series = {name: [] for name in TARGETS}
    series["xi_s"] = []
    series["xi_mod"] = []
    series["M_mean"] = []
    tick_stamps = []

    t0 = time.time()
    for tick in range(1, ticks + 1):
        engine.tick()
        if tick % sample_every == 0:
            m = engine.state.metrics
            for name, (key, _) in TARGETS.items():
                series[name].append(m.get(key, 0))
            series["xi_s"].append(m.get("xi_s_mean", 0))
            series["xi_mod"].append(m.get("xi_mod_mean", 0))
            series["M_mean"].append(engine.state.M.mean().item())
            tick_stamps.append(tick)

    elapsed = time.time() - t0
    print(f"\n  Data collected in {elapsed:.0f}s ({len(tick_stamps)} samples)")

    # Convert to numpy
    for k in series:
        series[k] = np.array(series[k])

    # Overall correlations
    print(f"\n  FULL-RUN CORRELATIONS (ticks 1-{ticks})")
    print(f"  {'Pair':<25s} {'r':>8s}")
    print(f"  {'-'*25} {'-'*8}")
    pairs = [
        ("alpha", "lambda"),
        ("f_local", "G_local"),
        ("gamma", "alpha"),
        ("gamma", "f_local"),
        ("f_local", "gamma"),
    ]
    for a, b in pairs:
        r = np.corrcoef(series[a], series[b])[0, 1]
        print(f"  {a} vs {b:<15s} {r:>+8.4f}")

    # Group correlations
    group1 = (series["gamma"] + series["alpha"] + series["lambda"]) / 3
    group2 = (series["f_local"] + series["G_local"]) / 2
    r_groups = np.corrcoef(group1, group2)[0, 1]
    print(f"\n  Group1 (gam,alp,lam) vs Group2 (f,G): r = {r_groups:+.4f}")

    # Error from targets over time
    print(f"\n  CONVERGENCE TRAJECTORY")
    print(f"  {'Tick':>6s} {'f%':>7s} {'gam%':>7s} {'alp%':>7s} {'G%':>7s} "
          f"{'lam%':>7s} {'avg%':>7s} {'M_mean':>7s} {'xi_s':>7s}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    target_vals = {name: target for name, (_, target) in TARGETS.items()}
    checkpoints = [500, 1000, 2000, 3000, 5000, 7000, 10000]
    for cp in checkpoints:
        idx = cp // sample_every - 1
        if idx >= len(tick_stamps):
            continue
        errs = {}
        for name in TARGETS:
            val = series[name][idx]
            target = target_vals[name]
            errs[name] = abs(val - target) / abs(target) * 100
        avg_err = sum(errs.values()) / len(errs)
        print(f"  {cp:>6d} {errs['f_local']:>6.1f}% {errs['gamma']:>6.1f}% "
              f"{errs['alpha']:>6.1f}% {errs['G_local']:>6.1f}% "
              f"{errs['lambda']:>6.1f}% {avg_err:>6.1f}% "
              f"{series['M_mean'][idx]:>7.3f} {series['xi_s'][idx]:>7.3f}")

    # Running correlations (if we have enough data)
    if len(tick_stamps) > window:
        print(f"\n  RUNNING CORRELATIONS (window={window} samples = {window*sample_every} ticks)")
        # alpha-lambda correlation over time
        al_corr = running_correlation(series["alpha"], series["lambda"], window)
        fg_corr = running_correlation(series["f_local"], series["G_local"], window)
        group_corr = running_correlation(group1, group2, window)

        # Report at checkpoints
        print(f"  {'Tick':>6s} {'alp-lam':>8s} {'f-G':>8s} {'Grp1-2':>8s}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
        for i in range(0, len(al_corr), max(1, len(al_corr) // 8)):
            tick = tick_stamps[i + window]
            print(f"  {tick:>6d} {al_corr[i]:>+8.4f} {fg_corr[i]:>+8.4f} "
                  f"{group_corr[i]:>+8.4f}")

        # Correlation stability: std of running correlation
        print(f"\n  CORRELATION STABILITY (std of running correlation)")
        print(f"  alpha-lambda:  mean={np.mean(al_corr):+.4f}  std={np.std(al_corr):.4f}")
        print(f"  f_local-G:     mean={np.mean(fg_corr):+.4f}  std={np.std(fg_corr):.4f}")
        print(f"  Group1-Group2: mean={np.mean(group_corr):+.4f}  std={np.std(group_corr):.4f}")

        # Does xi_s predict correlation strength?
        xi_s_late = series["xi_s"][window:]
        if len(xi_s_late) == len(group_corr):
            r_xi_corr = np.corrcoef(xi_s_late, np.abs(group_corr))[0, 1]
            print(f"\n  xi_s vs |Group correlation|: r = {r_xi_corr:+.4f}")
            if abs(r_xi_corr) > 0.5:
                print(f"    *** xi_s PREDICTS inter-group coherence ***")

    print(f"\n  DONE")


if __name__ == "__main__":
    main()
