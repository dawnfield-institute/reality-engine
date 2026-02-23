"""
Pure SEC diagnostic — strips away engine (no PAC, no actualization, no memory).

Tests whether the SEC PDE alone can push ξ_L2 above 1.0 with
source steering, or if ξ ≈ 1.0 is inherent to the Möbius topology.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.substrate.mobius import MobiusManifold
from src.dynamics.sec import SECEvolver
from src.dynamics.confluence import ConfluenceOperator
from src.substrate.constants import XI_REFERENCE, PHI_INV


def run_pure_sec(
    label: str,
    rho: float = 1.0,
    n_steps: int = 20_000,
    initial_phi: float | None = None,
    **sec_kwargs,
):
    """Run pure SEC (no engine) and track ξ_L2."""
    manifold = MobiusManifold(n_u=128, n_v=64, device="cpu")
    confluence = ConfluenceOperator()

    params = {
        "kappa": 0.1,
        "gamma": 1.0,
        "beta_0": 1.0,
        "sigma_0": 0.1,
        "dt": 0.01,
        "xi_gain": 2.0,
        "rho": rho,
        "phi_source": initial_phi if initial_phi is not None else PHI_INV,
        "alpha_rbf": 5.0,
        "rbf_decay": 0.995,
        "ki_rbf": 0.5,
        "integral_clamp": 1.0,
        "rbf_omega": 0.2,
    }
    params.update(sec_kwargs)

    sec = SECEvolver(manifold, params, confluence=confluence)

    # Initial field: uniform + antiperiodic seed
    torch.manual_seed(42)
    S = torch.ones(128, 64) * 0.12  # near steady state
    u = manifold.U
    v = manifold.V
    S = S + 0.01 * torch.sin(u) * torch.sin(math.pi * v)

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  ρ={rho}, phi_source={params['phi_source']:.3f}")
    print(f"{'='*72}")
    print(f"  {'step':>6s} | {'ξ_L2':>8s} | {'S_mean':>8s} | {'S_std':>8s} | {'E_anti':>10s} | {'E_sym':>10s}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

    theta = 0.0
    for i in range(n_steps):
        # Measure ξ_L2
        f_sym, f_anti = confluence.decompose(S)
        E_sym = f_sym.pow(2).sum().item()
        E_anti = f_anti.pow(2).sum().item()
        xi_L2 = E_anti / max(E_sym, 1e-14)

        if i % 1000 == 0 or i == n_steps - 1:
            print(f"  {i:6d} | {xi_L2:8.4f} | {S.mean().item():8.4f} | {S.std().item():8.4f} | {E_anti:10.4f} | {E_sym:10.4f}")

        S, theta = sec.step(S, xi_L2, theta)

    xi_err = abs(xi_L2 - XI_REFERENCE) / XI_REFERENCE
    print(f"\n  Final ξ_L2 = {xi_L2:.6f} (target = {XI_REFERENCE}, error = {xi_err:.2%})")


if __name__ == "__main__":
    # 1. Pure SEC with NO steering (baseline)
    run_pure_sec(
        "PURE_SEC_NO_STEERING",
        rho=0.0,  # no RBF
    )

    # 2. Pure SEC with source steering ρ=5
    run_pure_sec(
        "PURE_SEC_STEERING_rho5",
        rho=5.0,
    )

    # 3. Pure SEC with VERY different phi_source (force asymmetry)
    run_pure_sec(
        "PURE_SEC_phi=1.5",
        rho=0.0,
        initial_phi=1.5,  # very high anti content
    )

    # 4. Pure SEC with phi_source = 0 (only uniform source)
    run_pure_sec(
        "PURE_SEC_phi=0",
        rho=0.0,
        initial_phi=0.0,  # pure uniform source
    )

    # 5. Pure SEC with phi_source tuned for target
    # ξ ∝ phi² * R_ratio, and ξ ≈ 1.0 at phi=0.618
    # For ξ = 1.057: phi = 0.618 * sqrt(1.057) = 0.635
    run_pure_sec(
        "PURE_SEC_phi=0.635",
        rho=0.0,
        initial_phi=0.635,
    )

    # 6. Pure SEC phi = phi itself (golden ratio)
    run_pure_sec(
        "PURE_SEC_phi=1.618",
        rho=0.0,
        initial_phi=1.618,
    )
