"""
RBF Long Run — 50k steps with source steering.

Tests whether ξ_L2 continues climbing past 1.0 toward 1.057
when given enough time and stronger ρ.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import RealityEngine
from src.substrate.constants import XI_REFERENCE


CONFIGS = {
    "RBF_PI_rho5": {
        "rho": 5.0,
        "alpha_rbf": 5.0,
        "ki_rbf": 0.5,
        "rbf_omega": 0.2,
        "rbf_decay": 0.995,
        "integral_clamp": 1.0,
    },
    "RBF_PI_rho20": {
        "rho": 20.0,
        "alpha_rbf": 5.0,
        "ki_rbf": 0.5,
        "rbf_omega": 0.2,
        "rbf_decay": 0.995,
        "integral_clamp": 1.0,
    },
    "RBF_PI_rho50": {
        "rho": 50.0,
        "alpha_rbf": 5.0,
        "ki_rbf": 0.5,
        "rbf_omega": 0.2,
        "rbf_decay": 0.995,
        "integral_clamp": 1.0,
    },
    "RBF_PI_rho50_lowmem": {
        # Lower memory dampening → B stays stronger
        "rho": 50.0,
        "alpha_rbf": 1.0,
        "ki_rbf": 1.0,
        "rbf_omega": 0.2,
        "rbf_decay": 0.995,
        "integral_clamp": 2.0,
    },
}


def run_config(name: str, sec_overrides: dict, n_steps: int = 50_000) -> None:
    sec_params = {
        "kappa": 0.1,
        "gamma": 1.0,
        "beta_0": 1.0,
        "sigma_0": 0.1,
        "dt": 0.01,
        "xi_gain": 2.0,
        "phi_source": 0.618,
    }
    sec_params.update(sec_overrides)

    config = {
        "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
        "sec": sec_params,
        "init": {"seed": 42, "P_mean": 0.5, "P_noise": 0.01, "A_mean": 0.5, "A_noise": 0.01},
        "pac": {"mode": "enforce"},
    }

    engine = RealityEngine(config)

    print(f"\n{'='*72}")
    print(f"  {name} ({n_steps} steps)")
    print(f"  ρ={sec_overrides.get('rho', 1.0)}, α_rbf={sec_overrides.get('alpha_rbf', 5.0)}, "
          f"ki={sec_overrides.get('ki_rbf', 0.5)}")
    print(f"{'='*72}")
    print(f"  {'step':>6s} | {'ξ_L2':>8s} | {'P_mean':>8s} | {'P_std':>8s} | {'M_rbf':>8s} | {'∫ξ_err':>8s}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for i in range(n_steps):
        diag = engine.step()

        if i % 2500 == 0 or i == n_steps - 1:
            xi_L2 = diag.get("xi_L2", float("nan"))
            P_mean = diag.get("P_mean", float("nan"))
            P_std = diag.get("P_std", float("nan"))
            M_rbf = diag.get("M_rbf", 0)
            xi_int = diag.get("xi_integral", 0)
            print(f"  {diag['t']:6d} | {xi_L2:8.4f} | {P_mean:8.4f} | {P_std:8.4f} | {M_rbf:8.4f} | {xi_int:8.4f}")

        if engine.diagnostics.diverged:
            print(f"  ⚠ DIVERGED at step {diag['t']}")
            break

    xi_final = diag.get("xi_L2", float("nan"))
    xi_err = abs(xi_final - XI_REFERENCE) / XI_REFERENCE
    print(f"\n  Final ξ_L2 = {xi_final:.6f} (target = {XI_REFERENCE:.4f}, error = {xi_err:.2%})")


if __name__ == "__main__":
    for name, overrides in CONFIGS.items():
        run_config(name, overrides)
