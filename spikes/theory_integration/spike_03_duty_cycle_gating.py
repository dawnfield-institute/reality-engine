"""Spike 03: SEC duty cycle gating on mass generation.

Theory source: SEC threshold detection, cellular_automata_pac_attractors.
    Equilibrium duty cycle = phi/(phi+1) = 0.618. This is the fraction of
    time the system spends in "actualization" (coherence > entropy).

    The Conditional Attractor Hypothesis: Xi ~ 1.0571 is the maximum
    sustainable asymmetry for closed recursive systems. Rule 110 (Class IV,
    edge of chaos) shows P/A = 1.0579 (0.07% from Xi).

Hypothesis: Mass should only generate when gamma_local exceeds the duty
    cycle threshold phi/(phi+1). Below this threshold, the cell is in
    the "potential" phase of the PAC cycle and shouldn't crystallize.

    This creates a hard phase gate: mass generation is ON only during
    the actualization phase of the local PAC cycle.

    Current behavior: mass generates everywhere gamma_local > 0 (no gate).
    This means mass forms even in entropy-dominated regions, then has to
    be dissolved by de-actualization. The duty cycle gate prevents this
    wasted work and should produce cleaner mass distributions.

Variants:
    A) Hard gate: mass_gen = 0 when gamma_local < phi/(phi+1)
    B) Soft gate: mass_gen *= sigmoid(k * (gamma_local - threshold))
    C) Xi gate: mass_gen *= clamp(gamma_local / Xi - 1, 0, 1) — onset at Xi
"""

import math
import torch
from harness import (
    default_pipeline, default_config, run_and_score,
    print_result, print_comparison,
)
from src.v3.operators.memory import MemoryOperator
from src.v3.substrate.manifold import MobiusManifold

_EPS = 1e-12
_PHI = (1 + math.sqrt(5)) / 2
_GAMMA_EM = 0.5772156649015328
_XI = _GAMMA_EM + math.log(_PHI)


class DutyCycleMemoryOperator(MemoryOperator):
    """Memory operator with SEC duty cycle gating."""

    def __init__(self, mode="hard"):
        super().__init__()
        self.mode = mode
        self._duty_threshold = _PHI / (_PHI + 1.0)  # 0.618

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        # Compute gamma_local to determine duty cycle phase
        E, I = state.E, state.I
        diseq2 = (E - I).pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        if self.mode == "hard":
            # Hard gate: mass generation only when gamma > threshold
            gate = (gamma_local > self._duty_threshold).float()
        elif self.mode == "soft":
            # Soft sigmoid gate (k=20 for moderately sharp transition)
            gate = torch.sigmoid(20.0 * (gamma_local - self._duty_threshold))
        elif self.mode == "xi":
            # Gate onset at Xi (broader, more permissive)
            gate = torch.clamp((gamma_local - 1.0) / (_XI - 1.0), min=0.0, max=1.0)
        else:
            gate = torch.ones_like(gamma_local)

        # Run normal memory operator
        result = super().__call__(state, config, bus)

        # Apply gate to mass change
        dM = result.M - state.M
        dM_gated = dM * gate
        M_new = state.M + dM_gated
        M_new = torch.clamp(M_new, min=0.0)

        # PAC correction
        dM_actual = M_new - state.M
        orig_dM_actual = result.M - state.M
        drain_diff = (dM_actual - orig_dM_actual) * 0.5
        E_new = result.E - drain_diff
        I_new = result.I - drain_diff

        metrics = dict(result.metrics)
        metrics["duty_gate_mean"] = gate.mean().item()
        metrics["duty_gate_active_frac"] = (gate > 0.5).float().mean().item()

        return result.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


def make_pipeline_with(memory_op):
    from src.v3.operators.protocol import Pipeline
    from src.v3.operators.rbf import RBFOperator
    from src.v3.operators.qbe import QBEOperator
    from src.v3.operators.actualization import ActualizationOperator
    from src.v3.operators.phi_cascade import PhiCascadeOperator
    from src.v3.operators.gravity import GravitationalCollapseOperator
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
        memory_op, PhiCascadeOperator(),
        GravitationalCollapseOperator(),
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
    print("  SPIKE 03: SEC Duty Cycle Gating on Mass Generation")
    print(f"  Device: {device} | Grid: 128x64 | {ticks} ticks")
    print(f"  Duty threshold: phi/(phi+1) = {_PHI / (_PHI + 1.0):.6f}")
    print("=" * 80)

    results = []

    # Baseline
    print("\n  Running baseline...", flush=True)
    r = run_and_score("baseline", default_pipeline(), config, ticks=ticks)
    results.append(r)
    print_result(r)

    for mode in ["hard", "soft", "xi"]:
        print(f"\n  Running duty_{mode}...", flush=True)
        p = make_pipeline_with(DutyCycleMemoryOperator(mode=mode))
        r = run_and_score(f"duty_{mode}", p, config, ticks=ticks)
        results.append(r)
        print_result(r)

    print("\n" + "=" * 80)
    print("  COMPARISON")
    print("=" * 80)
    print_comparison(results)

    best = min(results, key=lambda r: r["avg_t1"])
    print(f"\n  BEST: {best['label']}  avg_t1={best['avg_t1']:.1f}%")


if __name__ == "__main__":
    main()
