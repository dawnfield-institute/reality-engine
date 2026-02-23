"""
ki_rbf sweep — the integral is clamped at -1.0, so gain_i = ki * 1.0.
Higher ki → more B → more source steering + collapse modulation → higher ξ.

The key insight: the integral clamp creates a hard ceiling on gain_i.
By increasing ki, we get more gain per unit of clamped integral.
At high ki, ξ should overshoot then settle as integral unwinds.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import RealityEngine
from src.substrate.constants import XI_REFERENCE


def run_ki_sweep():
    """Sweep ki_rbf to find ξ=1.057."""
    print("=" * 70)
    print("  ki_rbf SWEEP — breaking through the ξ=1.032 ceiling")
    print(f"  Target: ξ_L2 = {XI_REFERENCE}")
    print("=" * 70)

    configs = [
        ("ki=0.5",  {"ki_rbf": 0.5}),
        ("ki=1.0",  {"ki_rbf": 1.0}),
        ("ki=1.5",  {"ki_rbf": 1.5}),
        ("ki=2.0",  {"ki_rbf": 2.0}),
        ("ki=3.0",  {"ki_rbf": 3.0}),
        ("ki=5.0",  {"ki_rbf": 5.0}),
    ]

    for name, overrides in configs:
        sec_params = {
            "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
            "sigma_0": 0.1, "dt": 0.01,
            "xi_gain": 2.0, "rho": 5.0, "phi_source": 0.618,
            "alpha_rbf": 5.0, "rbf_decay": 0.995,
            "integral_clamp": 1.0, "rbf_omega": 0.2,
        }
        sec_params.update(overrides)
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
        P_std = diag.get("P_std", float("nan"))
        M_rbf = diag.get("M_rbf", 0)
        xi_int = diag.get("xi_integral", 0)
        err = (xi - XI_REFERENCE) / XI_REFERENCE * 100
        marker = " <<<" if abs(err) < 0.5 else " <" if abs(err) < 2.0 else ""
        divtxt = "DIVERGED" if engine.diagnostics.diverged else ""
        print(f"  {name:8s}  ξ={xi:.6f} (err={err:+6.2f}%)  "
              f"Pm={P_mean:.4f}  Pstd={P_std:.4f}  "
              f"M={M_rbf:.3f}  ∫={xi_int:.3f}  {divtxt}{marker}")


def run_ki_rho_grid():
    """2D sweep ki × ρ to find sweet spot."""
    print("\n" + "=" * 70)
    print("  ki × ρ GRID")
    print("=" * 70)

    kis = [1.0, 2.0, 3.0, 5.0]
    rhos = [3.0, 5.0, 7.0, 10.0]
    
    print(f"  {'':8s}", end="")
    for rho in rhos:
        print(f"   ρ={rho:4.1f}  ", end="")
    print()
    
    for ki in kis:
        print(f"  ki={ki:.1f}  ", end="")
        for rho in rhos:
            sec_params = {
                "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
                "sigma_0": 0.1, "dt": 0.01,
                "xi_gain": 2.0, "rho": rho, "phi_source": 0.618,
                "alpha_rbf": 5.0, "rbf_decay": 0.995,
                "ki_rbf": ki, "integral_clamp": 1.0, "rbf_omega": 0.2,
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
            err = abs(xi - XI_REFERENCE) / XI_REFERENCE * 100
            marker = "*" if err < 0.5 else "~" if err < 2.0 else " "
            print(f"  {xi:.4f}{marker}", end="")
        print()


def run_convergence_best(ki, rho, n_steps=20000):
    """Run convergence trace at a specified ki and rho."""  
    print(f"\n{'='*70}")
    print(f"  CONVERGENCE TRACE — ki={ki}, ρ={rho}, low Pm")
    print(f"{'='*70}")

    sec_params = {
        "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
        "sigma_0": 0.1, "dt": 0.01,
        "xi_gain": 2.0, "rho": rho, "phi_source": 0.618,
        "alpha_rbf": 5.0, "rbf_decay": 0.995,
        "ki_rbf": ki, "integral_clamp": 1.0, "rbf_omega": 0.2,
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

    print(f"  {'step':>6s} | {'ξ_L2':>10s} | {'error%':>8s} | {'P_mean':>8s} | "
          f"{'P_std':>8s} | {'M_rbf':>8s} | {'∫ξ_err':>8s}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for i in range(n_steps):
        diag = engine.step()
        if i % 1000 == 0 or i == n_steps - 1:
            xi = diag.get("xi_L2", 0)
            err = (xi - XI_REFERENCE) / XI_REFERENCE * 100
            pm = diag.get("P_mean", 0)
            ps = diag.get("P_std", 0)
            mr = diag.get("M_rbf", 0)
            xi_int = diag.get("xi_integral", 0)
            print(f"  {diag['t']:6d} | {xi:10.6f} | {err:+8.3f}% | {pm:8.4f} | "
                  f"{ps:8.4f} | {mr:8.4f} | {xi_int:8.4f}")

    xi_final = diag.get("xi_L2", 0)
    err_final = abs(xi_final - XI_REFERENCE) / XI_REFERENCE
    print(f"\n  Final: ξ = {xi_final:.6f} (target = {XI_REFERENCE}, error = {err_final:.3%})")


if __name__ == "__main__":
    run_ki_sweep()
    run_ki_rho_grid()
    # Run convergence for the two most promising
    run_convergence_best(ki=3.0, rho=5.0)
    run_convergence_best(ki=5.0, rho=5.0)
