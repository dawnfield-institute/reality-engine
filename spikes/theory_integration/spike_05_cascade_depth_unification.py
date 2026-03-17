"""Spike 05: Cascade depth unification of coupling constants.

Theory source: PAC derivation chain (Papers 1-4), pac_confluence_xi.
    All Standard Model constants are projections of the PAC recursion
    at different Fibonacci depths. Fine structure alpha = F3/(F4*phi*F10).
    The 5 Tier 1 constants should be structurally locked, not computed
    independently.

    From M5 exp_11: alpha and lambda correlate at r = 1.000 (exact).
    f_local and G_local at r = 0.997. The two groups anti-correlate
    at r = -0.98. This IS structural locking, but the simulator doesn't
    enforce it.

Hypothesis: Instead of computing gamma_local and G_local independently
    in memory and gravity operators, compute a single "local cascade depth"
    n_local and derive both from it:
        gamma_local = phi^(-n_local)    [memory at depth n]
        G_local = phi^(-2*n_local)      [gravity at depth 2n]

    This forces G_local = gamma_local^2, which is the theoretical
    relationship (G -> 1/phi^2 when gamma -> 1/phi).

    The cascade depth n_local is computed from the field state:
        n_local = log_phi(sqrt(E^2 + I^2) / |E - I|)
    At equilibrium (gamma = 1/phi): n_local = 1, G = 1/phi^2. Correct.
    At high disequilibrium: n_local -> 0, G -> 1, gamma -> 1. Strong coupling.
    At perfect balance: n_local -> inf, G -> 0, gamma -> 0. No coupling.

Implementation: Modified gravity operator that uses gamma_local from
    memory (passed via metrics) squared, instead of its own M^2/(M^2+diseq^2).
"""

import math
import torch
from harness import (
    default_pipeline, default_config, run_and_score,
    print_result, print_comparison, PHI,
)
from src.v3.operators.gravity import GravitationalCollapseOperator
from src.v3.operators.memory import MemoryOperator

_EPS = 1e-12
_PHI = (1 + math.sqrt(5)) / 2
_LN_PHI = math.log(_PHI)


class CascadeDepthGravityOperator(GravitationalCollapseOperator):
    """Gravity operator that derives G_local from cascade depth.

    Instead of G_local = M^2 / (M^2 + diseq^2), uses:
        G_local = gamma_local^2 * xi_mod

    This locks G_local to gamma_local structurally, enforcing
    G -> 1/phi^2 when gamma -> 1/phi (the theoretical relationship).
    """

    def __init__(self, mode="gamma_squared", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        # Compute cascade depth from E, I fields
        diseq2 = (E - I).pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        if self.mode == "gamma_squared":
            # G_local = gamma_local^2 — structural locking
            G_mass = gamma_local.pow(2)
        elif self.mode == "cascade_depth":
            # Explicit cascade depth: n = -log_phi(gamma_local)
            # Then G = phi^(-2n) = gamma_local^2 (same result, different path)
            gamma_safe = torch.clamp(gamma_local, min=_EPS, max=1.0 - _EPS)
            n_local = -torch.log(gamma_safe) / _LN_PHI
            G_mass = _PHI ** (-2.0) * torch.ones_like(n_local)  # at equilibrium
            # Scale by deviation from equilibrium depth (n=1)
            depth_factor = torch.exp(-torch.abs(n_local - 1.0) * _LN_PHI)
            G_mass = gamma_local.pow(2) * depth_factor
        else:
            # Original formula
            M2 = M.pow(2)
            G_mass = M2 / (M2 + diseq2 + _EPS)

        # Entropy-coherence modulation (keep from parent)
        E2 = E.pow(2)
        I2 = I.pow(2)
        xi_s = I2 / (E2 + _EPS)
        xi_s_phi = xi_s.pow(1.0 / _PHI)
        xi_mod = torch.sqrt(xi_s_phi / (xi_s_phi + 1.0))

        G_local = G_mass * xi_mod

        # Poisson solve and flux (reuse parent's infrastructure)
        phi_pot = self._solve_poisson(torch.sqrt(M + _EPS))
        grad_phi_u = (torch.roll(phi_pot, -1, 0) - torch.roll(phi_pot, 1, 0)) / 2.0
        grad_phi_v = (torch.roll(phi_pot, -1, 1) - torch.roll(phi_pot, 1, 1)) / 2.0
        flux_u = M * grad_phi_u
        flux_v = M * grad_phi_v
        div_flux = (
            (torch.roll(flux_u, -1, 0) - torch.roll(flux_u, 1, 0)) / 2.0 +
            (torch.roll(flux_v, -1, 1) - torch.roll(flux_v, 1, 1)) / 2.0
        )
        dM_grav = G_local * div_flux * dt

        M_candidate = M + dM_grav
        M_new = torch.clamp(M_candidate, min=0.0)
        mass_created = M_new - M_candidate
        pac_leak = mass_created * 0.5
        E_new = E - pac_leak
        I_new = I - pac_leak

        G_mean = G_local.mean().item()
        metrics = dict(state.metrics)
        metrics["gravitational_potential_max"] = phi_pot.max().item()
        metrics["G_local_mean"] = G_mean
        metrics["G_local_std"] = G_local.std().item()
        metrics["xi_s_mean"] = xi_s.mean().item()
        metrics["xi_s_std"] = xi_s.std().item()
        metrics["xi_mod_mean"] = xi_mod.mean().item()
        metrics["cascade_G_mass_mean"] = G_mass.mean().item()

        if bus is not None:
            bus.emit("gravity_evolved", {
                "potential_max": phi_pot.max().item(),
                "G_local_mean": G_mean,
            })

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


def make_pipeline_with_gravity(grav_op):
    from src.v3.operators.protocol import Pipeline
    from src.v3.operators.rbf import RBFOperator
    from src.v3.operators.qbe import QBEOperator
    from src.v3.operators.actualization import ActualizationOperator
    from src.v3.operators.memory import MemoryOperator
    from src.v3.operators.phi_cascade import PhiCascadeOperator
    from src.v3.operators.spin_statistics import SpinStatisticsOperator
    from src.v3.operators.charge_dynamics import ChargeDynamicsOperator
    from src.v3.operators.fusion import FusionOperator
    from src.v3.operators.confluence import ConfluenceOperator
    from src.v3.operators.temperature import TemperatureOperator
    from src.v3.operators.thermal_noise import ThermalNoiseOperator
    from src.v3.operators.normalization import NormalizationOperator
    from src.v3.operators.sec_tracking import SECTrackingOperator
    from src.v3.operators.adaptive import AdaptiveOperator
    from src.v3.operators.time_emergence import TimeEmergenceOperator

    return Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
        grav_op,
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), SECTrackingOperator(),
        AdaptiveOperator(), TimeEmergenceOperator(),
    ])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = default_config(device=device)
    ticks = 10000

    print("=" * 80)
    print("  SPIKE 05: Cascade Depth Unification of Coupling Constants")
    print(f"  Device: {device} | Grid: 128x64 | {ticks} ticks")
    print(f"  Theory: G_local = gamma_local^2 (structural locking)")
    print(f"  Expected: G -> 1/phi^2 when gamma -> 1/phi")
    print("=" * 80)

    results = []

    # Baseline
    print("\n  Running baseline (original G formula)...", flush=True)
    r = run_and_score("baseline", default_pipeline(), config, ticks=ticks)
    results.append(r)
    print_result(r)

    # Variant A: G_local = gamma_local^2 * xi_mod
    print("\n  Running gamma_squared...", flush=True)
    p = make_pipeline_with_gravity(CascadeDepthGravityOperator(mode="gamma_squared"))
    r = run_and_score("G=gamma^2", p, config, ticks=ticks)
    results.append(r)
    print_result(r)

    # Variant B: Explicit cascade depth with equilibrium scaling
    print("\n  Running cascade_depth...", flush=True)
    p = make_pipeline_with_gravity(CascadeDepthGravityOperator(mode="cascade_depth"))
    r = run_and_score("G=cascade_depth", p, config, ticks=ticks)
    results.append(r)
    print_result(r)

    print("\n" + "=" * 80)
    print("  COMPARISON")
    print("=" * 80)
    print_comparison(results)

    best = min(results, key=lambda r: r["avg_t1"])
    print(f"\n  BEST: {best['label']}  avg_t1={best['avg_t1']:.1f}%")

    # Check if G_local = gamma_local^2 holds
    for r in results:
        gamma = r["metrics"].get("gamma_local_mean", 0)
        G = r["metrics"].get("G_local_mean", 0)
        print(f"  {r['label']}: gamma={gamma:.4f}  G={G:.4f}  "
              f"gamma^2={gamma**2:.4f}  ratio={G/(gamma**2+1e-12):.3f}")


if __name__ == "__main__":
    main()
