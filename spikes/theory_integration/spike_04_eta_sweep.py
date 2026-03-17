"""Spike 04: De-actualization rate (eta) sweep.

Theory source: Unified Emergence Framework (internal/).
    Optimal recursive memory decay nu ~ 0.025 found across domains.
    Current simulator uses eta = 0.01 (from initial de-actualization work).

Hypothesis: The optimal eta may be higher than 0.01. The unified
    emergence framework found nu ~ 0.025 across multiple systems.
    With attractor-gated forgetting now active (forgetting = 0 at
    equilibrium), higher eta is safe — it only affects cells that
    are far from the attractor, where faster dissolution is desirable.

    Also test whether eta = 1/phi^3 ~ 0.0236 or eta = ln(2)/phi^4 ~ 0.101
    have any special significance (DFT constants as natural rates).

Sweep: eta from 0.005 to 0.10, including DFT-motivated values.
"""

import math
import torch
from harness import (
    default_pipeline, default_config, run_and_score,
    print_result, print_comparison, PHI, LN2,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticks = 10000

    print("=" * 80)
    print("  SPIKE 04: De-actualization Rate (eta) Sweep")
    print(f"  Device: {device} | Grid: 128x64 | {ticks} ticks")
    print(f"  Current default: eta = 0.01")
    print(f"  Theory suggestion: nu ~ 0.025")
    print("=" * 80)

    eta_values = [
        (0.005, "half_current"),
        (0.010, "current_default"),
        (0.020, "double"),
        (1.0 / PHI**3, "1/phi^3"),       # 0.02360...
        (0.025, "unified_emergence"),
        (0.030, "triple"),
        (LN2 / 10, "ln2/10"),            # 0.06931...
        (0.050, "5x"),
        (1.0 / PHI**4, "1/phi^4"),       # 0.01459...
    ]

    results = []
    for eta, label in sorted(eta_values, key=lambda x: x[0]):
        full_label = f"eta={eta:.4f} ({label})"
        print(f"\n  Running {full_label}...", flush=True)
        config = default_config(device=device, deactualization_rate=eta)
        r = run_and_score(full_label, default_pipeline(), config, ticks=ticks)
        results.append(r)
        print_result(r, show_t1=True)

    print("\n" + "=" * 80)
    print("  COMPARISON (sorted by avg_t1)")
    print("=" * 80)
    results.sort(key=lambda r: r["avg_t1"])
    print_comparison(results)

    best = results[0]
    print(f"\n  BEST: {best['label']}")
    print(f"  avg_t1={best['avg_t1']:.1f}%  phi2={best['phi2_err']:.1f}%  "
          f"entropy_cumul={best['entropy_cumul']:.4f}")


if __name__ == "__main__":
    main()
