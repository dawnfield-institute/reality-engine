"""
Focused sweep: find parameters where ξ_L2 → 1.057.

Key finding: at low P_mean (near SEC equilibrium S_ss ≈ 0.12),
the collapse is responsive (dC/dS > 0) and ξ naturally approaches
the target.  High P_mean (≈1.0) causes nonlinear collapse saturation
that locks ξ to 1.0.

Strategy: use init_Pm=0.1 (field operates at P_mean≈0.035) and
sweep ρ to find convergence to ξ=1.057.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import RealityEngine
from src.substrate.constants import XI_REFERENCE


def run_low_pm_rho_sweep():
    """Sweep ρ at low P_mean to find ξ=1.057."""
    print("=" * 70)
    print("  LOW P_mean + ρ SWEEP → ξ=1.057")
    print(f"  Target: ξ_L2 = {XI_REFERENCE}")
    print("=" * 70)

    rhos = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    n_steps = 10_000

    for rho in rhos:
        sec_params = {
            "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
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

        last_xi = 0
        diverged = False
        for i in range(n_steps):
            diag = engine.step()
            if engine.diagnostics.diverged:
                diverged = True
                break
            if i == n_steps - 1:
                last_xi = diag.get("xi_L2", 0)

        xi = diag.get("xi_L2", float("nan"))
        P_mean = diag.get("P_mean", float("nan"))
        P_std = diag.get("P_std", float("nan"))
        M_rbf = diag.get("M_rbf", 0)
        xi_int = diag.get("xi_integral", 0)
        err = abs(xi - XI_REFERENCE) / XI_REFERENCE * 100

        status = f"DIVERGED step {diag['t']}" if diverged else ""
        marker = " <<<" if err < 1.0 else " <" if err < 3.0 else ""
        print(f"  ρ={rho:5.1f}  ξ={xi:.6f} (err={err:5.2f}%)  "
              f"Pm={P_mean:.4f}  Pstd={P_std:.3f}  "
              f"M={M_rbf:.3f}  ∫={xi_int:.3f}  {status}{marker}")


def run_low_pm_phi_sweep():
    """Sweep phi_source at low P_mean to find additional tuning."""
    print("\n" + "=" * 70)
    print("  phi_source SWEEP at low P_mean, ρ=5")
    print("=" * 70)

    phis = [0.3, 0.5, 0.618, 0.7, 0.8, 1.0, 1.2]
    n_steps = 10_000

    for phi in phis:
        sec_params = {
            "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
            "sigma_0": 0.1, "dt": 0.01,
            "xi_gain": 2.0, "rho": 5.0, "phi_source": phi,
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

        for i in range(n_steps):
            diag = engine.step()
            if engine.diagnostics.diverged:
                break

        xi = diag.get("xi_L2", float("nan"))
        P_mean = diag.get("P_mean", float("nan"))
        err = abs(xi - XI_REFERENCE) / XI_REFERENCE * 100
        marker = " <<<" if err < 1.0 else " <" if err < 3.0 else ""
        print(f"  φ={phi:.3f}  ξ={xi:.6f} (err={err:5.2f}%)  Pm={P_mean:.4f}{marker}")


def run_convergence_trace():
    """Trace ξ convergence at the best parameters found."""
    print("\n" + "=" * 70)
    print("  CONVERGENCE TRACE — best params (low Pm, ρ=5)")
    print("=" * 70)

    sec_params = {
        "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
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

    print(f"  {'step':>6s} | {'ξ_L2':>10s} | {'error%':>8s} | {'P_mean':>8s} | "
          f"{'P_std':>8s} | {'M_rbf':>8s} | {'∫ξ_err':>8s}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    n_steps = 20_000
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
    run_low_pm_rho_sweep()
    run_low_pm_phi_sweep()
    run_convergence_trace()
