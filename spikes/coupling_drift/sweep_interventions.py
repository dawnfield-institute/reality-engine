"""Spike: sweep interventions to fix late-time coupling drift.

Problem: coupling constants converge beautifully at tick 1000 (G_local 2.5%,
alpha 1.4%, f_local 3.7%) but drift badly by 10K (54%, 13%, 7%).
Root cause: mass bimodality -- 17% at cap, 60% nearly empty.

Interventions tested:
  A. Baseline (current params)
  B. Higher mass diffusion (spread mass faster)
  C. Higher quantum pressure (resist collapse)
  D. Both diffusion + pressure
  E. Lower field_scale (tighter cap -> more Landauer recycling)
  F. Best combo from above + gamma_damping tuning

Each run does 5000 ticks (drift visible by tick 2K) and reports key metrics.
"""

import math
import os
import sys
import time

re_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))
if re_path not in sys.path:
    sys.path.insert(0, re_path)

import torch

from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
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

PHI = (1 + math.sqrt(5)) / 2
LN2 = math.log(2)

# DFT targets
TARGETS = {
    "f_local":     0.5772,   # gamma_EM
    "gamma_local": 1 / PHI,  # 0.6180
    "alpha_local": LN2,      # 0.6931
    "G_local":     1 / PHI**2,  # 0.3820
    "lambda_local": 1 - LN2, # 0.3069
}

INTERVENTIONS = {
    "A_baseline": dict(),
    "B_diffusion_5x": dict(mass_diffusion_coeff=0.0025),
    "C_diffusion_20x": dict(mass_diffusion_coeff=0.01),
    "D_qpressure_3x": dict(quantum_pressure_coeff=0.045),
    "E_qpressure_10x": dict(quantum_pressure_coeff=0.15),
    "F_diff5x_qp3x": dict(mass_diffusion_coeff=0.0025, quantum_pressure_coeff=0.045),
    "G_diff20x_qp3x": dict(mass_diffusion_coeff=0.01, quantum_pressure_coeff=0.045),
    "H_lower_scale": dict(field_scale=10.0),
    "I_diff5x_scale10": dict(mass_diffusion_coeff=0.0025, field_scale=10.0),
    "J_diff20x_qp3x_scale10": dict(mass_diffusion_coeff=0.01, quantum_pressure_coeff=0.045, field_scale=10.0),
}


def build_pipeline():
    return Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
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


def run_intervention(name, overrides, device, ticks=5000):
    """Run a single intervention and return metrics at checkpoints."""
    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True,
        actualization_threshold=0.05,
        **overrides,
    )
    torch.manual_seed(42)
    pipeline = build_pipeline()
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    checkpoints = [1000, 2000, 5000]
    results = {}
    cp_idx = 0

    for tick in range(1, ticks + 1):
        engine.tick()
        if cp_idx < len(checkpoints) and tick == checkpoints[cp_idx]:
            cp_idx += 1
            m = engine.state.metrics
            M = engine.state.M
            M_cap = config.field_scale / 5.0
            frac_cap = (M > M_cap * 0.9).float().mean().item()
            frac_empty = (M < 0.1).float().mean().item()

            results[tick] = {
                "f_local": m.get("f_local_mean", 0),
                "gamma_local": m.get("gamma_local_mean", 0),
                "alpha_local": m.get("alpha_local_mean", 0),
                "G_local": m.get("G_local_mean", 0),
                "lambda_local": m.get("lambda_local_mean", 0),
                "M_mean": M.mean().item(),
                "M_std": M.std().item(),
                "frac_cap": frac_cap,
                "frac_empty": frac_empty,
                "xi_mod_mean": m.get("xi_mod_mean", 0),
            }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*100}")
    print(f"  COUPLING DRIFT INTERVENTION SWEEP")
    print(f"  Device: {device} | Grid: 128x64 | 5000 ticks per run")
    print(f"  {len(INTERVENTIONS)} interventions")
    print(f"{'='*100}")

    all_results = {}
    for name, overrides in INTERVENTIONS.items():
        t0 = time.time()
        desc = ", ".join(f"{k}={v}" for k, v in overrides.items()) if overrides else "defaults"
        print(f"\n  [{name}] ({desc}) ...", end="", flush=True)
        results = run_intervention(name, overrides, device)
        elapsed = time.time() - t0
        print(f" {elapsed:.0f}s")
        all_results[name] = results

    # --- Report: coupling errors at tick 1000 and 5000 ---
    print(f"\n{'='*100}")
    print(f"  TIER 1 COUPLING ERRORS AT TICK 1000")
    print(f"{'='*100}")
    header = f"  {'Intervention':<28s}"
    for metric in TARGETS:
        header += f" | {metric:>12s}"
    header += " |  avg_err"
    print(header)
    print(f"  {'-'*28}" + ("-+-" + "-"*12) * len(TARGETS) + "-+---------")

    scores_1k = {}
    for name, results in all_results.items():
        r = results.get(1000, {})
        row = f"  {name:<28s}"
        errs = []
        for metric, target in TARGETS.items():
            measured = r.get(metric, 0)
            err = pct_err(measured, target)
            errs.append(err)
            g = grade(err)
            row += f" | {err:>7.1f}% {g:>2s}"
        avg = sum(errs) / len(errs)
        scores_1k[name] = avg
        row += f" | {avg:>6.1f}%"
        print(row)

    print(f"\n{'='*100}")
    print(f"  TIER 1 COUPLING ERRORS AT TICK 5000")
    print(f"{'='*100}")
    print(header)
    print(f"  {'-'*28}" + ("-+-" + "-"*12) * len(TARGETS) + "-+---------")

    scores_5k = {}
    for name, results in all_results.items():
        r = results.get(5000, {})
        row = f"  {name:<28s}"
        errs = []
        for metric, target in TARGETS.items():
            measured = r.get(metric, 0)
            err = pct_err(measured, target)
            errs.append(err)
            g = grade(err)
            row += f" | {err:>7.1f}% {g:>2s}"
        avg = sum(errs) / len(errs)
        scores_5k[name] = avg
        row += f" | {avg:>6.1f}%"
        print(row)

    # --- Report: mass distribution health ---
    print(f"\n{'='*100}")
    print(f"  MASS DISTRIBUTION HEALTH")
    print(f"{'='*100}")
    print(f"  {'Intervention':<28s} | {'M_mean':>7s} {'M_std':>7s} | {'%cap':>6s} {'%empty':>6s} | {'%cap':>6s} {'%empty':>6s} | {'xi_mod':>7s}")
    print(f"  {'':28s} | {'--- tick 1000 ---':^16s} | {'--- tick 5000 ---':^14s} |")
    print(f"  {'-'*28}-+-{'-'*7}-{'-'*7}-+-{'-'*6}-{'-'*6}-+-{'-'*6}-{'-'*6}-+-{'-'*7}")

    for name, results in all_results.items():
        r1 = results.get(1000, {})
        r5 = results.get(5000, {})
        print(f"  {name:<28s}"
              f" | {r1.get('M_mean',0):7.3f} {r1.get('M_std',0):7.3f}"
              f" | {r1.get('frac_cap',0):5.1%} {r1.get('frac_empty',0):5.1%}"
              f" | {r5.get('frac_cap',0):5.1%} {r5.get('frac_empty',0):5.1%}"
              f" | {r5.get('xi_mod_mean',0):7.4f}")

    # --- Drift report: tick 1000 -> 5000 ---
    print(f"\n{'='*100}")
    print(f"  DRIFT: avg error tick 1000 -> 5000 (lower = more stable)")
    print(f"{'='*100}")
    ranked = sorted(all_results.keys(), key=lambda n: scores_5k.get(n, 999))
    for name in ranked:
        s1 = scores_1k.get(name, 0)
        s5 = scores_5k.get(name, 0)
        drift = s5 - s1
        bar = "#" * int(min(s5, 80) / 2)
        print(f"  {name:<28s}  t1k={s1:5.1f}%  t5k={s5:5.1f}%  drift={drift:+5.1f}%  {bar}")

    print(f"\n  Best at tick 5000: {ranked[0]}")
    print(f"  Least drift: {sorted(all_results.keys(), key=lambda n: scores_5k.get(n,999) - scores_1k.get(n,0))[0]}")
    print()


if __name__ == "__main__":
    main()
