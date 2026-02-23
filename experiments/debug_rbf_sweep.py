"""
RBF Parameter Sweep — diagnostic script.

Tests how ξ_L2 converges under different RBF configurations:
  1. BASELINE: old behavior (no RBF — α_rbf=0, ki_rbf=0, Φ disabled)
  2. RBF_ONLY: memory dampening only (ki_rbf=0)
  3. RBF_PI: memory dampening + integral (full RBF)
  4. RBF_PI_HIGH_RHO: full RBF with higher ρ (should be safe with dampening)

Each run: 10,000 steps, log ξ_L2 every 500 steps.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import RealityEngine
from src.substrate.constants import XI_REFERENCE


CONFIGS = {
    "BASELINE": {
        # Effectively old behavior: no RBF dampening, no integral, constant Phi
        "alpha_rbf": 0.0,
        "ki_rbf": 0.0,
        "rbf_omega": 0.0,  # cos(0) = 1 always → Phi = 1
        "rho": 1.0,
    },
    "RBF_ONLY": {
        # Memory dampening but no integral — proportional droop expected
        "alpha_rbf": 5.0,
        "ki_rbf": 0.0,
        "rbf_omega": 0.2,
        "rho": 2.0,
    },
    "RBF_PI": {
        # Full RBF: memory dampening + integral + Fibonacci
        "alpha_rbf": 5.0,
        "ki_rbf": 0.5,
        "rbf_omega": 0.2,
        "rho": 2.0,
    },
    "RBF_PI_HIGH_RHO": {
        # Higher ρ — should be stable with RBF dampening
        "alpha_rbf": 5.0,
        "ki_rbf": 0.5,
        "rbf_omega": 0.2,
        "rho": 5.0,
    },
    "RBF_PI_VERY_HIGH_RHO": {
        # Even higher ρ — stress test for RBF self-limitation
        "alpha_rbf": 5.0,
        "ki_rbf": 1.0,
        "rbf_omega": 0.2,
        "rho": 10.0,
    },
}


def run_config(name: str, sec_overrides: dict, n_steps: int = 10_000) -> None:
    sec_params = {
        "kappa": 0.1,
        "gamma": 1.0,
        "beta_0": 1.0,
        "sigma_0": 0.1,
        "dt": 0.01,
        "xi_gain": 2.0,
        "phi_source": 0.618,
        "rbf_decay": 0.995,
        "integral_clamp": 1.0,
    }
    sec_params.update(sec_overrides)

    config = {
        "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
        "sec": sec_params,
        "init": {"seed": 42, "P_mean": 0.5, "P_noise": 0.01, "A_mean": 0.5, "A_noise": 0.01},
        "pac": {"mode": "enforce"},
    }

    engine = RealityEngine(config)

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  ρ={sec_overrides.get('rho', 1.0)}, α_rbf={sec_overrides.get('alpha_rbf', 5.0)}, "
          f"ki={sec_overrides.get('ki_rbf', 0.5)}, ω={sec_overrides.get('rbf_omega', 0.2)}")
    print(f"{'='*70}")
    print(f"  {'step':>6s} | {'ξ_L2':>8s} | {'P_mean':>8s} | {'P_std':>8s} | {'M_rbf':>8s} | {'∫ξ_err':>8s}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    diverged = False
    for i in range(n_steps):
        diag = engine.step()

        if i % 500 == 0 or i == n_steps - 1:
            xi_L2 = diag.get("xi_L2", float("nan"))
            P_mean = diag.get("P_mean", float("nan"))
            P_std = diag.get("P_std", float("nan"))
            M_rbf = diag.get("M_rbf", 0)
            xi_int = diag.get("xi_integral", 0)
            print(f"  {diag['t']:6d} | {xi_L2:8.4f} | {P_mean:8.4f} | {P_std:8.4f} | {M_rbf:8.4f} | {xi_int:8.4f}")

        if engine.diagnostics.diverged:
            print(f"  ⚠ DIVERGED at step {diag['t']}")
            diverged = True
            break

    if not diverged:
        xi_final = diag.get("xi_L2", float("nan"))
        xi_err = abs(xi_final - XI_REFERENCE) / XI_REFERENCE
        print(f"\n  Final ξ_L2 = {xi_final:.6f} (target = {XI_REFERENCE:.4f}, error = {xi_err:.2%})")


if __name__ == "__main__":
    for name, overrides in CONFIGS.items():
        run_config(name, overrides)
