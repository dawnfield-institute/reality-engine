"""
Final tuning: close the ξ gap from 1.030 → 1.057.

Two approaches:
  A. κ reduction (0.08-0.1, within ±20% spec): less anti diffusion
  B. Leaky integrator: reset integral on initial transient, only
     accumulate near the target → prevent saturation at -1.0
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import RealityEngine
from src.substrate.constants import XI_REFERENCE


def run_kappa_sweep():
    """Sweep κ at low P_mean to close the gap."""
    print("=" * 70)
    print("  κ SWEEP — closing the gap from 1.030 → 1.057")
    print("=" * 70)

    kappas = [0.08, 0.085, 0.09, 0.092, 0.094, 0.096, 0.098, 0.1]
    
    for kappa in kappas:
        sec_params = {
            "kappa": kappa, "gamma": 1.0, "beta_0": 1.0,
            "sigma_0": 0.1, "dt": 0.01,
            "xi_gain": 2.0, "rho": 5.0, "phi_source": 0.618,
            "alpha_rbf": 5.0, "rbf_decay": 0.995,
            "ki_rbf": 0.5, "integral_clamp": 1.0, "rbf_omega": 0.2,
        }
        config = {
            "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
            "sec": sec_params,
            "init": {"seed": 42, "P_mean": 0.1, "P_noise": 0.01,
                     "A_mean": 0.1, "A_noise": 0.01,
                     "antiperiodic_amp": 0.01},
            "pac": {"mode": "enforce"},
        }
        engine = RealityEngine(config)

        for i in range(10000):
            diag = engine.step()
            if engine.diagnostics.diverged:
                break

        xi = diag.get("xi_L2", float("nan"))
        P_mean = diag.get("P_mean", float("nan"))
        err = abs(xi - XI_REFERENCE) / XI_REFERENCE * 100
        cfl = kappa * 0.01 / (2*3.14159/128)**2
        marker = " <<<" if err < 0.5 else " <" if err < 1.5 else ""
        print(f"  κ={kappa:.3f}  ξ={xi:.6f} (err={err:5.2f}%)  "
              f"Pm={P_mean:.4f}  CFL={cfl:.3f}{marker}")


def run_kappa_rho_grid():
    """2D grid sweep of κ × ρ around the sweet spot."""
    print("\n" + "=" * 70)
    print("  κ × ρ GRID — finding the sweet spot")
    print("=" * 70)

    kappas = [0.085, 0.09, 0.095]
    rhos = [3.0, 5.0, 7.0, 10.0]
    
    print(f"  {'':8s}", end="")
    for rho in rhos:
        print(f"  ρ={rho:5.1f}  ", end="")
    print()
    
    for kappa in kappas:
        print(f"  κ={kappa:.3f}", end="")
        for rho in rhos:
            sec_params = {
                "kappa": kappa, "gamma": 1.0, "beta_0": 1.0,
                "sigma_0": 0.1, "dt": 0.01,
                "xi_gain": 2.0, "rho": rho, "phi_source": 0.618,
                "alpha_rbf": 5.0, "rbf_decay": 0.995,
                "ki_rbf": 0.5, "integral_clamp": 1.0, "rbf_omega": 0.2,
            }
            config = {
                "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
                "sec": sec_params,
                "init": {"seed": 42, "P_mean": 0.1, "P_noise": 0.01,
                         "A_mean": 0.1, "A_noise": 0.01,
                         "antiperiodic_amp": 0.01},
                "pac": {"mode": "enforce"},
            }
            engine = RealityEngine(config)

            for i in range(10000):
                diag = engine.step()
                if engine.diagnostics.diverged:
                    break

            xi = diag.get("xi_L2", float("nan"))
            err = (xi - XI_REFERENCE) / XI_REFERENCE * 100
            marker = "*" if abs(err) < 0.5 else "~" if abs(err) < 1.5 else " "
            print(f"  {xi:.4f}{marker}", end="")
        print()


def run_convergence_best():
    """Run convergence trace at the best κ,ρ found."""
    # Will update after finding best params from sweeps above
    print("\n" + "=" * 70)
    print("  CONVERGENCE TRACE at κ=0.09, ρ=7 (estimated best)")
    print("=" * 70)

    sec_params = {
        "kappa": 0.09, "gamma": 1.0, "beta_0": 1.0,
        "sigma_0": 0.1, "dt": 0.01,
        "xi_gain": 2.0, "rho": 7.0, "phi_source": 0.618,
        "alpha_rbf": 5.0, "rbf_decay": 0.995,
        "ki_rbf": 0.5, "integral_clamp": 1.0, "rbf_omega": 0.2,
    }
    config = {
        "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
        "sec": sec_params,
        "init": {"seed": 42, "P_mean": 0.1, "P_noise": 0.01,
                 "A_mean": 0.1, "A_noise": 0.01,
                 "antiperiodic_amp": 0.01},
        "pac": {"mode": "enforce"},
    }
    engine = RealityEngine(config)

    n_steps = 15000
    print(f"  {'step':>6s} | {'ξ_L2':>10s} | {'error%':>8s} | {'P_mean':>8s} | "
          f"{'P_std':>8s}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for i in range(n_steps):
        diag = engine.step()
        if i % 1000 == 0 or i == n_steps - 1:
            xi = diag.get("xi_L2", 0)
            err = (xi - XI_REFERENCE) / XI_REFERENCE * 100
            pm = diag.get("P_mean", 0)
            ps = diag.get("P_std", 0)
            print(f"  {diag['t']:6d} | {xi:10.6f} | {err:+8.3f}% | {pm:8.4f} | "
                  f"{ps:8.4f}")

    xi_final = diag.get("xi_L2", 0)
    err_final = abs(xi_final - XI_REFERENCE) / XI_REFERENCE
    print(f"\n  Final: ξ = {xi_final:.6f} (target = {XI_REFERENCE}, error = {err_final:.3%})")


if __name__ == "__main__":
    run_kappa_sweep()
    run_kappa_rho_grid()
    run_convergence_best()
