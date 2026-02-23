"""
Final attack: beta_0 x rho x init_Pm grid search.

The ξ equilibrium depends on the balance between source, collapse,
diffusion, and the topology modulation. At different β₀ values, the
collapse responds differently to the field level, changing the ξ
equilibrium point.

Within the ±20% validated range: β₀ ∈ [0.8, 1.2].
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import RealityEngine
from src.substrate.constants import XI_REFERENCE


def run_beta_pm_grid():
    """Grid of beta_0 x init_Pm at fixed rho, ki."""
    print("=" * 70)
    print("  beta_0 x init_Pm grid (rho=5, ki=1)")
    print("=" * 70)

    betas = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    p_means = [0.05, 0.1, 0.15, 0.2, 0.3]

    print(f"  {'':8s}", end="")
    for pm in p_means:
        print(f"  Pm={pm:.2f}  ", end="")
    print()

    for beta in betas:
        print(f"  b={beta:.2f}  ", end="")
        for pm in p_means:
            sec_params = {
                "kappa": 0.1, "gamma": 1.0, "beta_0": beta,
                "sigma_0": 0.1, "dt": 0.01,
                "xi_gain": 2.0, "rho": 5.0, "phi_source": 0.618,
                "alpha_rbf": 5.0, "rbf_decay": 0.995,
                "ki_rbf": 1.0, "integral_clamp": 1.0, "rbf_omega": 0.2,
            }
            config = {
                "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
                "sec": sec_params,
                "init": {"seed": 42, "P_mean": pm, "P_noise": 0.01,
                         "A_mean": pm, "A_noise": 0.01,
                         "antiperiodic_amp": 0.01},
                "pac": {"mode": "enforce"},
            }
            engine = RealityEngine(config)

            for i in range(8000):
                diag = engine.step()
                if engine.diagnostics.diverged:
                    break

            xi = diag.get("xi_L2", float("nan"))
            err = abs(xi - XI_REFERENCE) / XI_REFERENCE * 100
            marker = "*" if err < 0.5 else "~" if err < 1.5 else " "
            print(f" {xi:.4f}{marker}", end="")
        print()


def find_best_and_trace():
    """Run the best config from the grid for 20k steps."""
    # Based on prior results, beta ≈ 0.9-1.1 and Pm=0.1 or 0.15
    # seems promising. Let me do a focused trace.
    configs = [
        ("b=0.95 Pm=0.1 rho=5", {"beta_0": 0.95, "rho": 5.0}, 0.1),
        ("b=1.0  Pm=0.1 rho=5", {"beta_0": 1.0, "rho": 5.0}, 0.1),
        ("b=1.05 Pm=0.1 rho=5", {"beta_0": 1.05, "rho": 5.0}, 0.1),
        ("b=1.1  Pm=0.1 rho=5", {"beta_0": 1.1, "rho": 5.0}, 0.1),
        ("b=1.0  Pm=0.15 rho=5", {"beta_0": 1.0, "rho": 5.0}, 0.15),
        ("b=1.0  Pm=0.1 rho=10", {"beta_0": 1.0, "rho": 10.0}, 0.1),
    ]
    
    print("\n" + "=" * 70)
    print("  FOCUSED CONFIGS — 15k steps")
    print("=" * 70)
    
    for name, overrides, pm in configs:
        sec_params = {
            "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
            "sigma_0": 0.1, "dt": 0.01,
            "xi_gain": 2.0, "rho": 5.0, "phi_source": 0.618,
            "alpha_rbf": 5.0, "rbf_decay": 0.995,
            "ki_rbf": 1.0, "integral_clamp": 1.0, "rbf_omega": 0.2,
        }
        sec_params.update(overrides)
        config = {
            "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
            "sec": sec_params,
            "init": {"seed": 42, "P_mean": pm, "P_noise": 0.01,
                     "A_mean": pm, "A_noise": 0.01,
                     "antiperiodic_amp": 0.01},
            "pac": {"mode": "enforce"},
        }
        engine = RealityEngine(config)

        for i in range(15000):
            diag = engine.step()
            if engine.diagnostics.diverged:
                break

        xi = diag.get("xi_L2", float("nan"))
        P_mean = diag.get("P_mean", float("nan"))
        P_std = diag.get("P_std", float("nan"))
        err = (xi - XI_REFERENCE) / XI_REFERENCE * 100
        marker = " <<<" if abs(err) < 0.5 else " <" if abs(err) < 2.0 else ""
        print(f"  {name:30s}  xi={xi:.6f} (err={err:+6.2f}%)  "
              f"Pm={P_mean:.4f}  Pstd={P_std:.4f}{marker}")


if __name__ == "__main__":
    run_beta_pm_grid()
    find_best_and_trace()
