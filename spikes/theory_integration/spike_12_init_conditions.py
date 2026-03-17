"""Spike 12: Initialization Conditions — Does the Big Bang Matter?

The v3 big_bang init sets E ~ I (equal amplitude random), giving gamma ~ 0.5
from tick 0. But DFT says the big bang is pure entropy: E >> I, M = 0.

Spike 11 showed a "bounce" in cascade depth — gamma starts high, crashes,
then recovers. Is this physics or initialization artifact?

This spike tests:
  A. Current init (E ~ I ~ temperature)         — "symmetric"
  B. DFT-correct init (E >> I)                   — "entropy-dominated"
  C. Inverted init (I >> E)                      — "information-dominated"
  D. Cold start (E ~ I ~ small)                  — "cold"
  E. Different temperatures on current init      — "hot" vs "cool"

For each, we track:
  - gamma_local evolution (duty cycle)
  - NOW tick location (minimum coupling error)
  - Post-NOW drift rates
  - Whether constants still converge to the same attractors

If the NOW tick and late-time behavior are invariant across inits,
the physics is real. If they depend on init, it's artifact.
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
)
from src.v3.engine.engine import Engine
from src.v3.engine.state import FieldState


def custom_init(engine, mode, temperature=3.0):
    """Initialize engine with custom field conditions."""
    config = engine.config
    nu, nv = config.nu, config.nv
    device = config.device
    dtype = torch.float64

    torch.manual_seed(42)

    if mode == "symmetric":
        # Current default: E ~ I
        E = torch.randn(nu, nv, dtype=dtype, device=device) * temperature
        I = torch.randn(nu, nv, dtype=dtype, device=device) * temperature
    elif mode == "entropy_dominated":
        # DFT big bang: E >> I (pure entropy, no structure)
        E = torch.randn(nu, nv, dtype=dtype, device=device) * temperature
        I = torch.randn(nu, nv, dtype=dtype, device=device) * temperature * 0.01
    elif mode == "strong_entropy":
        # Even more extreme: E >>> I
        E = torch.randn(nu, nv, dtype=dtype, device=device) * temperature
        I = torch.randn(nu, nv, dtype=dtype, device=device) * temperature * 0.001
    elif mode == "info_dominated":
        # Inverted: I >> E (should NOT produce good physics)
        E = torch.randn(nu, nv, dtype=dtype, device=device) * temperature * 0.01
        I = torch.randn(nu, nv, dtype=dtype, device=device) * temperature
    elif mode == "cold":
        # Low energy, equal E and I
        E = torch.randn(nu, nv, dtype=dtype, device=device) * 0.1
        I = torch.randn(nu, nv, dtype=dtype, device=device) * 0.1
    elif mode == "hot":
        # Very high temperature, equal E and I
        E = torch.randn(nu, nv, dtype=dtype, device=device) * 10.0
        I = torch.randn(nu, nv, dtype=dtype, device=device) * 10.0
    elif mode == "v2_style":
        # v2 big bang: high uniform E + noise, minimal I
        E = torch.ones(nu, nv, dtype=dtype, device=device) * temperature
        E += torch.randn(nu, nv, dtype=dtype, device=device) * temperature * 0.1
        I = torch.randn(nu, nv, dtype=dtype, device=device) * 0.05
    else:
        raise ValueError(f"Unknown mode: {mode}")

    M = torch.zeros(nu, nv, dtype=dtype, device=device)
    T = torch.full((nu, nv), temperature, dtype=dtype, device=device)
    Z = torch.zeros(nu, nv, dtype=dtype, device=device)

    engine._state = FieldState(E=E, I=I, M=M, T=T, Z=Z)
    engine.bus.emit("initialized", {"mode": mode, "shape": (nu, nv)})


def run_variant(label, mode, temperature=3.0, ticks=15000, sample_every=50):
    """Run one initialization variant and return time series."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = default_config(device=device)
    pipeline = default_pipeline()
    engine = Engine(config=config, pipeline=pipeline)

    # Custom init instead of engine.initialize()
    custom_init(engine, mode, temperature)

    target_vals = {name: target for name, (_, target) in TARGETS.items()}
    series = {name: [] for name in TARGETS}
    series["gamma_raw"] = []
    series["M_mean"] = []
    series["G_local"] = []
    series["EI_ratio"] = []
    tick_stamps = []

    # Record initial state
    E0 = engine.state.E
    I0 = engine.state.I
    gamma0 = (I0.pow(2) / (E0.pow(2) + I0.pow(2) + 1e-10)).mean().item()
    ei0 = E0.abs().mean().item() / (E0.abs().mean().item() + I0.abs().mean().item() + 1e-10)

    t0 = time.time()
    for tick in range(1, ticks + 1):
        engine.tick()
        if tick % sample_every == 0:
            m = engine.state.metrics
            for name, (key, _) in TARGETS.items():
                series[name].append(float(m.get(key, 0.0)))
            series["gamma_raw"].append(float(m.get("gamma_local_mean", 0.0)))
            series["M_mean"].append(engine.state.M.mean().item())
            series["G_local"].append(float(m.get("G_local_mean", 0.0)))

            E_abs = engine.state.E.abs().mean().item()
            I_abs = engine.state.I.abs().mean().item()
            series["EI_ratio"].append(E_abs / (E_abs + I_abs + 1e-10))

            tick_stamps.append(tick)

    elapsed = time.time() - t0

    # Convert to numpy
    for k in series:
        series[k] = np.array(series[k])
    tick_stamps = np.array(tick_stamps)
    n_samples = len(tick_stamps)

    # Find NOW tick
    errors = {}
    for name in TARGETS:
        s = series[name]
        t = target_vals[name]
        if s.ndim == 0 or len(s) == 0:
            errors[name] = np.array([999.0] * n_samples)
        else:
            errors[name] = np.abs(s - t) / (np.abs(t) + 1e-30) * 100
    err_stack = np.stack([errors[name][:n_samples] for name in TARGETS], axis=0)
    avg_errors = np.mean(err_stack, axis=0)
    now_idx = np.argmin(avg_errors)
    now_tick = tick_stamps[now_idx]
    now_err = avg_errors[now_idx]

    # Drift rate at NOW
    window = 10
    drift_rates = {}
    if now_idx > window and now_idx < n_samples - window:
        for name in TARGETS:
            y_before = series[name][now_idx - window]
            y_after = series[name][now_idx + window]
            dt = tick_stamps[now_idx + window] - tick_stamps[now_idx - window]
            drift_rates[name] = (y_after - y_before) / dt

    # Gamma trajectory checkpoints
    gamma_at = {}
    for cp in [50, 100, 500, 1000, 2000, 5000]:
        idx = cp // sample_every - 1
        if 0 <= idx < n_samples:
            gamma_at[cp] = series["gamma_raw"][idx]

    # Late-time values (last 1000 ticks)
    late_start = max(0, n_samples - 20)
    late_errs = {name: np.mean(errors[name][late_start:]) for name in TARGETS}
    late_avg = np.mean(list(late_errs.values()))

    return {
        "label": label,
        "mode": mode,
        "temperature": temperature,
        "elapsed": elapsed,
        "gamma0": gamma0,
        "ei0": ei0,
        "now_tick": now_tick,
        "now_err": now_err,
        "now_idx": now_idx,
        "drift_rates": drift_rates,
        "gamma_at": gamma_at,
        "gamma_now": series["gamma_raw"][now_idx],
        "G_now": series["G_local"][now_idx],
        "late_avg": late_avg,
        "late_errs": late_errs,
        "series": series,
        "tick_stamps": tick_stamps,
        "avg_errors": avg_errors,
        "errors": errors,
    }


def main():
    print("=" * 90)
    print("  SPIKE 12: Initialization Conditions -- Does the Big Bang Matter?")
    print("=" * 90)

    variants = [
        ("A: symmetric (current)",   "symmetric",        3.0),
        ("B: entropy-dominated",     "entropy_dominated", 3.0),
        ("C: strong entropy",        "strong_entropy",    3.0),
        ("D: info-dominated",        "info_dominated",    3.0),
        ("E: cold start",            "cold",              3.0),
        ("F: hot start",             "hot",               3.0),
        ("G: v2-style big bang",     "v2_style",          3.0),
    ]

    results = []
    for label, mode, temp in variants:
        print(f"\n  Running {label}...")
        r = run_variant(label, mode, temp)
        results.append(r)
        print(f"    gamma0={r['gamma0']:.4f}  E/total={r['ei0']:.4f}  "
              f"NOW={r['now_tick']}  err={r['now_err']:.2f}%  "
              f"late_avg={r['late_avg']:.1f}%  [{r['elapsed']:.0f}s]")

    # ====================================================================
    # COMPARISON TABLE
    # ====================================================================
    print(f"\n{'='*90}")
    print(f"  COMPARISON: How Init Affects Physics")
    print(f"{'='*90}")

    print(f"\n  {'Variant':<28s} {'gamma0':>7s} {'E/tot':>6s} {'NOW':>6s} "
          f"{'NOW%':>6s} {'gam@NOW':>8s} {'G@NOW':>7s} {'late%':>6s}")
    print(f"  {'-'*28} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*6}")

    for r in results:
        print(f"  {r['label']:<28s} {r['gamma0']:>7.4f} {r['ei0']:>6.3f} "
              f"{r['now_tick']:>6d} {r['now_err']:>5.1f}% "
              f"{r['gamma_now']:>8.4f} {r['G_now']:>7.4f} {r['late_avg']:>5.1f}%")

    # ====================================================================
    # GAMMA EVOLUTION COMPARISON
    # ====================================================================
    print(f"\n  GAMMA EVOLUTION (trajectory shape)")
    print(f"  {'Variant':<28s}", end="")
    checkpoints = [50, 100, 500, 1000, 2000, 5000]
    for cp in checkpoints:
        print(f" {f't={cp}':>8s}", end="")
    print()
    print(f"  {'-'*28}", end="")
    for _ in checkpoints:
        print(f" {'-'*8}", end="")
    print()

    for r in results:
        print(f"  {r['label']:<28s}", end="")
        for cp in checkpoints:
            val = r["gamma_at"].get(cp, float('nan'))
            print(f" {val:>8.4f}", end="")
        print()

    # ====================================================================
    # CONVERGENCE: Do they all reach the same attractors?
    # ====================================================================
    print(f"\n  ATTRACTOR CONVERGENCE (values at NOW tick)")
    print(f"  {'Variant':<28s} {'f':>8s} {'gamma':>8s} {'alpha':>8s} "
          f"{'G':>8s} {'lambda':>8s}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for r in results:
        idx = r["now_idx"]
        print(f"  {r['label']:<28s}", end="")
        for name in TARGETS:
            val = r["series"][name][idx]
            print(f" {val:>8.4f}", end="")
        print()

    print(f"  {'TARGET':<28s}", end="")
    for name, (_, target) in TARGETS.items():
        print(f" {target:>8.4f}", end="")
    print()

    # ====================================================================
    # DRIFT RATES AT NOW
    # ====================================================================
    print(f"\n  DRIFT RATES AT NOW (per tick)")
    print(f"  {'Variant':<28s} {'d(f)/dt':>10s} {'d(gam)/dt':>10s} "
          f"{'d(alp)/dt':>10s} {'d(G)/dt':>10s} {'d(lam)/dt':>10s}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in results:
        dr = r["drift_rates"]
        if dr:
            print(f"  {r['label']:<28s}", end="")
            for name in TARGETS:
                print(f" {dr.get(name, 0):>+10.2e}", end="")
            print()

    # ====================================================================
    # ERROR CURVES AT KEY EPOCHS
    # ====================================================================
    print(f"\n  AVG ERROR AT KEY EPOCHS")
    epochs = [500, 1000, 2000, 3000, 5000, 7000, 10000]
    print(f"  {'Variant':<28s}", end="")
    for ep in epochs:
        print(f" {'t='+str(ep):>8s}", end="")
    print()
    print(f"  {'-'*28}", end="")
    for _ in epochs:
        print(f" {'-'*8}", end="")
    print()

    for r in results:
        print(f"  {r['label']:<28s}", end="")
        for ep in epochs:
            idx = ep // 50 - 1
            if 0 <= idx < len(r["avg_errors"]):
                print(f" {r['avg_errors'][idx]:>7.1f}%", end="")
            else:
                print(f" {'N/A':>8s}", end="")
        print()

    # ====================================================================
    # THE KEY QUESTION: Is post-convergence behavior universal?
    # ====================================================================
    print(f"\n{'='*90}")
    print(f"  THE KEY QUESTION: Is late-time physics universal?")
    print(f"{'='*90}")

    # Compare gamma trajectories from tick 5000 onward across all variants
    # (after initial transient has died)
    ref = results[0]  # symmetric (current default)
    ref_late = ref["series"]["gamma_raw"][100:]  # tick 5000+

    print(f"\n  Gamma correlation with reference (symmetric) from tick 5000+:")
    for r in results:
        late = r["series"]["gamma_raw"][100:]  # tick 5000+
        if len(late) == len(ref_late) and len(late) > 2:
            corr = np.corrcoef(ref_late, late)[0, 1]
            rmse = np.sqrt(np.mean((ref_late - late) ** 2))
            print(f"    {r['label']:<28s}: corr={corr:+.4f}  rmse={rmse:.4f}")

    # Compare NOW tick errors
    now_errs = [r["now_err"] for r in results]
    now_ticks = [r["now_tick"] for r in results]
    print(f"\n  NOW tick range: {min(now_ticks)} - {max(now_ticks)} "
          f"(spread: {max(now_ticks)-min(now_ticks)} ticks)")
    print(f"  NOW error range: {min(now_errs):.2f}% - {max(now_errs):.2f}%")

    # Check if attractors are the same
    print(f"\n  Attractor value spread at NOW (std across variants):")
    for name in TARGETS:
        vals = [r["series"][name][r["now_idx"]] for r in results]
        print(f"    {name:<12s}: mean={np.mean(vals):.4f}  "
              f"std={np.std(vals):.4f}  "
              f"range={max(vals)-min(vals):.4f}")

    print(f"\n  DONE")


if __name__ == "__main__":
    main()
