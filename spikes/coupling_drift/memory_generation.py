"""Spike: memory operator mass generation fixes.

Problem: 37% of cells are empty (M < 0.1) with no mechanism to recover.
The current mass generation formula has an extremely steep nonlinearity:

  mass_gen = gamma_local * diseq^2 = (E-I)^4 / (E^2 + I^2 + eps)

Cells with even moderate disequilibrium generate mass much faster than
cells with weak disequilibrium. Once gravity moves M away, the cell's
E+I is already depleted (drained during mass generation), so diseq is
low, and the cell stays empty forever.

Interventions:
  A. Baseline (diseq^4 / field^2)
  B. Softer nonlinearity: diseq^2 instead of diseq^4
  C. Gradient seeding: add |grad(diseq)|^2 term for boundary generation
  D. Both: softer + gradient
  E. Adaptive diffusion: D_local = D * sqrt(M_mean / (M + eps))
"""

import math
import os
import sys
import time
from typing import Optional

re_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))
if re_path not in sys.path:
    sys.path.insert(0, re_path)

import torch

from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus
from src.v3.operators.protocol import Pipeline
from src.v3.operators.rbf import RBFOperator
from src.v3.operators.qbe import QBEOperator
from src.v3.operators.memory import MemoryOperator
from src.v3.operators.confluence import ConfluenceOperator
from src.v3.operators.temperature import TemperatureOperator
from src.v3.operators.thermal_noise import ThermalNoiseOperator
from src.v3.operators.normalization import NormalizationOperator
from src.v3.operators.adaptive import AdaptiveOperator
from src.v3.operators.time_emergence import TimeEmergenceOperator
from src.v3.operators.gravity import GravitationalCollapseOperator
from src.v3.operators.fusion import FusionOperator
from src.v3.operators.actualization import ActualizationOperator
from src.v3.operators.spin_statistics import SpinStatisticsOperator
from src.v3.operators.charge_dynamics import ChargeDynamicsOperator
from src.v3.operators.phi_cascade import PhiCascadeOperator
from src.v3.operators.sec_tracking import SECTrackingOperator
from src.v3.substrate.manifold import MobiusManifold

PHI = (1 + math.sqrt(5)) / 2
LN2 = math.log(2)
_EPS = 1e-12

TARGETS = {
    "f_local":     0.5772,
    "gamma_local": 1 / PHI,
    "alpha_local": LN2,
    "G_local":     1 / PHI**2,
    "lambda_local": 1 - LN2,
}


class _BaseMemory:
    """Base memory operator. Subclasses override _compute_mass_gen."""

    def __init__(self):
        self._manifold = None

    @property
    def name(self):
        return "memory"

    def _get_manifold(self, state):
        nu, nv = state.shape
        if self._manifold is None or self._manifold.nu != nu or self._manifold.nv != nv:
            self._manifold = MobiusManifold(nu, nv, device=state.device)
        return self._manifold

    def _compute_mass_gen(self, E, I, M, config, manifold):
        """Override this."""
        raise NotImplementedError

    def _compute_diffusion(self, M, config, manifold):
        """Override for adaptive diffusion."""
        lap_M = manifold.laplacian(M)
        return config.mass_diffusion_coeff * lap_M

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        m = self._get_manifold(state)
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        mass_gen, gamma_local = self._compute_mass_gen(E, I, M, config, m)

        # Quantum pressure
        M_safe = M + 1e-6
        lap_M2 = m.laplacian(M * M)
        quantum_pressure = -config.quantum_pressure_coeff * lap_M2 / M_safe

        # Diffusion
        diffusion = self._compute_diffusion(M, config, m)

        dM_dt = mass_gen + quantum_pressure + diffusion
        M_candidate = M + dt * dM_dt
        M_new = torch.clamp(M_candidate, min=0.0)

        dM_actual = M_new - M
        net_pac_drain = dM_actual * 0.5
        E_new = E - net_pac_drain
        I_new = I - net_pac_drain

        gamma_mean = gamma_local.mean().item()
        metrics = dict(state.metrics)
        metrics["mass_generation_rate"] = mass_gen.mean().item()
        metrics["gamma_local_mean"] = gamma_mean
        metrics["gamma_local_std"] = gamma_local.std().item()

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


class BaselineMemory(_BaseMemory):
    """A: Original diseq^4 / field^2."""
    def _compute_mass_gen(self, E, I, M, config, manifold):
        diseq2 = (E - I).pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2
        mass_gen = gamma_local * diseq2

        M_cap = config.field_scale / 5.0
        sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
        mass_gen = mass_gen * sat_cap
        return mass_gen, gamma_local


class SofterMemory(_BaseMemory):
    """B: diseq^2 instead of diseq^4. gamma_local * |diseq| instead of gamma_local * diseq^2."""
    def _compute_mass_gen(self, E, I, M, config, manifold):
        diseq = E - I
        diseq2 = diseq.pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        # Softer: gamma_local * |diseq| = diseq^3 / field^2
        mass_gen = gamma_local * diseq.abs()

        M_cap = config.field_scale / 5.0
        sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
        mass_gen = mass_gen * sat_cap
        return mass_gen, gamma_local


class GradientMemory(_BaseMemory):
    """C: Original + gradient term for boundary mass generation."""
    def _compute_mass_gen(self, E, I, M, config, manifold):
        diseq = E - I
        diseq2 = diseq.pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        # Standard bulk term
        mass_gen_bulk = gamma_local * diseq2

        # Gradient term: |grad(E-I)|^2 drives mass at boundaries
        grad_u = (torch.roll(diseq, -1, 0) - torch.roll(diseq, 1, 0)) / 2.0
        grad_v = (torch.roll(diseq, -1, 1) - torch.roll(diseq, 1, 1)) / 2.0
        grad_diseq2 = grad_u.pow(2) + grad_v.pow(2)

        # Scale gradient term to be comparable to bulk at boundaries
        mass_gen_boundary = gamma_local * grad_diseq2

        mass_gen = mass_gen_bulk + mass_gen_boundary

        M_cap = config.field_scale / 5.0
        sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
        mass_gen = mass_gen * sat_cap
        return mass_gen, gamma_local


class SofterGradientMemory(_BaseMemory):
    """D: Softer nonlinearity + gradient term."""
    def _compute_mass_gen(self, E, I, M, config, manifold):
        diseq = E - I
        diseq2 = diseq.pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        # Softer bulk
        mass_gen_bulk = gamma_local * diseq.abs()

        # Gradient boundary term
        grad_u = (torch.roll(diseq, -1, 0) - torch.roll(diseq, 1, 0)) / 2.0
        grad_v = (torch.roll(diseq, -1, 1) - torch.roll(diseq, 1, 1)) / 2.0
        grad_diseq2 = grad_u.pow(2) + grad_v.pow(2)
        mass_gen_boundary = gamma_local * torch.sqrt(grad_diseq2 + _EPS)

        mass_gen = mass_gen_bulk + mass_gen_boundary

        M_cap = config.field_scale / 5.0
        sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
        mass_gen = mass_gen * sat_cap
        return mass_gen, gamma_local


class AdaptiveDiffMemory(_BaseMemory):
    """E: Original mass gen + adaptive diffusion (stronger in sparse regions)."""
    def _compute_mass_gen(self, E, I, M, config, manifold):
        diseq2 = (E - I).pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2
        mass_gen = gamma_local * diseq2

        M_cap = config.field_scale / 5.0
        sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
        mass_gen = mass_gen * sat_cap
        return mass_gen, gamma_local

    def _compute_diffusion(self, M, config, manifold):
        lap_M = manifold.laplacian(M)
        M_mean = M.mean()
        D_local = config.mass_diffusion_coeff * torch.sqrt(M_mean / (M + 0.01))
        return D_local * lap_M


# ---------------------------------------------------------------
# Harness
# ---------------------------------------------------------------

def build_pipeline(memory_op):
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


def pct_err(measured, target):
    if target == 0:
        return abs(measured) * 100
    return abs(measured - target) / abs(target) * 100


def grade(err):
    if err < 1:   return "A+"
    if err < 5:   return "A"
    if err < 10:  return "B"
    if err < 15:  return "C"
    if err < 30:  return "D"
    return "F"


def run_variant(name, memory_op, device, ticks=5000):
    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True, actualization_threshold=0.05,
    )
    torch.manual_seed(42)
    pipeline = build_pipeline(memory_op)
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    checkpoints = [1000, 2000, 5000]
    results = {}
    cp_idx = 0

    for tick in range(1, ticks + 1):
        engine.tick()
        if cp_idx < len(checkpoints) and tick == checkpoints[cp_idx]:
            cp_idx += 1
            met = engine.state.metrics
            M = engine.state.M
            M_cap = config.field_scale / 5.0

            results[tick] = {
                "f_local": met.get("f_local_mean", 0),
                "gamma_local": met.get("gamma_local_mean", 0),
                "alpha_local": met.get("alpha_local_mean", 0),
                "G_local": met.get("G_local_mean", 0),
                "lambda_local": met.get("lambda_local_mean", 0),
                "M_mean": M.mean().item(),
                "M_std": M.std().item(),
                "frac_cap": (M > M_cap * 0.9).float().mean().item(),
                "frac_empty": (M < 0.1).float().mean().item(),
                "mass_gen_rate": met.get("mass_generation_rate", 0),
            }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VARIANTS = {
        "A_baseline":       BaselineMemory(),
        "B_softer":         SofterMemory(),
        "C_gradient":       GradientMemory(),
        "D_softer+gradient": SofterGradientMemory(),
        "E_adaptive_diff":  AdaptiveDiffMemory(),
    }

    print(f"\n{'='*105}")
    print(f"  MEMORY OPERATOR MASS GENERATION SPIKE")
    print(f"  Device: {device} | Grid: 128x64 | 5000 ticks per variant")
    print(f"  (Uses current gravity with xi_mod + sqrt + tiling)")
    print(f"{'='*105}")

    all_results = {}
    for name, mem_op in VARIANTS.items():
        t0 = time.time()
        print(f"\n  [{name}] ...", end="", flush=True)
        try:
            results = run_variant(name, mem_op, device)
            elapsed = time.time() - t0
            print(f" {elapsed:.0f}s")
            all_results[name] = results
        except Exception as e:
            print(f" FAILED: {e}")
            import traceback; traceback.print_exc()

    # --- Coupling errors ---
    for tick_label, tick in [("TICK 1000", 1000), ("TICK 5000", 5000)]:
        print(f"\n{'='*105}")
        print(f"  TIER 1 COUPLING ERRORS AT {tick_label}")
        print(f"{'='*105}")
        header = f"  {'Variant':<22s}"
        for metric in TARGETS:
            header += f" | {metric:>12s}"
        header += " |  avg_err"
        print(header)
        print(f"  {'-'*22}" + ("-+-" + "-"*12) * len(TARGETS) + "-+---------")

        for name, results in all_results.items():
            r = results.get(tick, {})
            if not r:
                continue
            row = f"  {name:<22s}"
            errs = []
            for metric, target in TARGETS.items():
                measured = r.get(metric, 0)
                err = pct_err(measured, target)
                errs.append(err)
                g = grade(err)
                row += f" | {err:>7.1f}% {g:>2s}"
            avg = sum(errs) / len(errs)
            row += f" | {avg:>6.1f}%"
            print(row)

    # --- Mass distribution ---
    print(f"\n{'='*105}")
    print(f"  MASS DISTRIBUTION AT TICK 5000")
    print(f"{'='*105}")
    print(f"  {'Variant':<22s} | {'M_mean':>7s} {'M_std':>7s} {'%cap':>6s} {'%empty':>7s} | {'gen_rate':>10s}")
    print(f"  {'-'*22}-+-{'-'*7}-{'-'*7}-{'-'*6}-{'-'*7}-+-{'-'*10}")

    for name, results in all_results.items():
        r = results.get(5000, {})
        if not r:
            continue
        print(f"  {name:<22s}"
              f" | {r['M_mean']:7.3f} {r['M_std']:7.3f} {r['frac_cap']:5.1%} {r['frac_empty']:6.1%}"
              f" | {r['mass_gen_rate']:10.6f}")

    # --- Drift ranking ---
    print(f"\n{'='*105}")
    print(f"  DRIFT RANKING")
    print(f"{'='*105}")
    scores = {}
    scores_1k = {}
    for name, results in all_results.items():
        for tick, store in [(1000, scores_1k), (5000, scores)]:
            r = results.get(tick, {})
            if r:
                errs = [pct_err(r.get(m, 0), t) for m, t in TARGETS.items()]
                store[name] = sum(errs) / len(errs)

    for name in sorted(scores, key=scores.get):
        e1 = scores_1k.get(name, 0)
        e5 = scores[name]
        drift = e5 - e1
        g_5k = pct_err(all_results[name].get(5000, {}).get("G_local", 0), 1/PHI**2)
        empty = all_results[name].get(5000, {}).get("frac_empty", 0)
        print(f"  {name:<22s}  t1k={e1:5.1f}%  t5k={e5:5.1f}%  drift={drift:+5.1f}%"
              f"  G={g_5k:5.1f}%  empty={empty:4.1%}")

    best = min(scores, key=scores.get)
    print(f"\n  >>> Best overall: {best} ({scores[best]:.1f}% avg error at tick 5000)")
    print()


if __name__ == "__main__":
    main()
