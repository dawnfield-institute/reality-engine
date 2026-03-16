"""Spike: diagnosing and fixing late-time G_local overshoot.

Problem: G_local hits 1/phi^2 perfectly at tick 5000 (0.8% error) but then
overshoots to 13.6% by tick 10000. The attractor isn't stable — the system
passes through equilibrium instead of settling.

G_local = G_mass * xi_mod where:
  G_mass = M^2 / (M^2 + (E-I)^2 + eps)
  xi_mod = sqrt(xi_s / (xi_s + 1)) where xi_s = I^2 / E^2

Hypothesis: as the system thermalizes, E and I equilibrate (xi_s -> 1),
so xi_mod -> 0.707 is correct. But G_mass keeps evolving because mass
continues concentrating. The product G_mass * xi_mod drops below 1/phi^2.

Interventions:
  A: Baseline (current engine)
  B: Stronger xi_mod (remove sqrt: xi_mod = xi_s / (xi_s + 1))
     -> more gravity suppression early, might stabilize the crossing
  C: xi_mod with phi-scaled sigmoid: xi_mod = xi_s^(1/phi) / (xi_s^(1/phi) + 1)
     -> flatter around xi_s=1, less sensitive to late-time xi_s drift
  D: Gradient seeding decay: boundary term * exp(-M_mean/M_cap)
     -> less mass generation at late times when system is mass-rich
  E: Adaptive gradient: boundary term * (1 - M/(M+1))
     -> gradient seeding weakens in cells that already have mass
  F: Combined: phi-scaled xi_mod + adaptive gradient
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
LN_PHI = math.log(PHI)
GAMMA_EM = 0.5772156649015329
XI = GAMMA_EM + LN_PHI
LN2_SQ = LN2 ** 2
_EPS = 1e-12

TARGETS = {
    "f_local":     0.5772,
    "gamma_local": 1 / PHI,
    "alpha_local": LN2,
    "G_local":     1 / PHI**2,
    "lambda_local": 1 - LN2,
}


# ---------------------------------------------------------------
# Gravity variants
# ---------------------------------------------------------------

class GravityPhiXiMod(GravitationalCollapseOperator):
    """C: phi-scaled xi_mod sigmoid — flatter response around xi_s=1."""

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        M2 = M.pow(2)
        diseq2 = (E - I).pow(2)
        G_mass = M2 / (M2 + diseq2 + _EPS)

        E2 = E.pow(2)
        I2 = I.pow(2)
        xi_s = I2 / (E2 + _EPS)
        # phi-scaled: xi_s^(1/phi) flattens the response around xi_s=1
        xi_s_phi = xi_s.pow(1.0 / PHI)
        xi_mod = torch.sqrt(xi_s_phi / (xi_s_phi + 1.0))

        G_local = G_mass * xi_mod

        phi = self._solve_poisson(torch.sqrt(M + _EPS))
        grad_phi_u = (torch.roll(phi, -1, 0) - torch.roll(phi, 1, 0)) / 2.0
        grad_phi_v = (torch.roll(phi, -1, 1) - torch.roll(phi, 1, 1)) / 2.0
        flux_u = M * grad_phi_u
        flux_v = M * grad_phi_v
        div_flux = (
            (torch.roll(flux_u, -1, 0) - torch.roll(flux_u, 1, 0)) / 2.0 +
            (torch.roll(flux_v, -1, 1) - torch.roll(flux_v, 1, 1)) / 2.0
        )
        dM_grav = G_local * div_flux * dt

        M_candidate = M + dM_grav
        M_new = torch.clamp(M_candidate, min=0.0)
        mass_created = (M_new - M_candidate)
        pac_leak = mass_created * 0.5
        E_new = E - pac_leak
        I_new = I - pac_leak

        metrics = dict(state.metrics)
        metrics["gravitational_potential_max"] = phi.max().item()
        metrics["G_local_mean"] = G_local.mean().item()
        metrics["G_local_std"] = G_local.std().item()
        metrics["xi_s_mean"] = xi_s.mean().item()
        metrics["xi_s_std"] = xi_s.std().item()
        metrics["xi_mod_mean"] = xi_mod.mean().item()

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


# ---------------------------------------------------------------
# Memory variants
# ---------------------------------------------------------------

class _BaseMemory:
    """Base memory operator with gradient seeding."""

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
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        m = self._get_manifold(state)
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        mass_gen, gamma_local = self._compute_mass_gen(E, I, M, config, m)

        M_safe = M + 1e-6
        lap_M2 = m.laplacian(M * M)
        quantum_pressure = -config.quantum_pressure_coeff * lap_M2 / M_safe
        lap_M = m.laplacian(M)
        diffusion = config.mass_diffusion_coeff * lap_M

        dM_dt = mass_gen + quantum_pressure + diffusion
        M_candidate = M + dt * dM_dt
        M_new = torch.clamp(M_candidate, min=0.0)

        dM_actual = M_new - M
        net_pac_drain = dM_actual * 0.5
        E_new = E - net_pac_drain
        I_new = I - net_pac_drain

        metrics = dict(state.metrics)
        metrics["mass_generation_rate"] = mass_gen.mean().item()
        metrics["gamma_local_mean"] = gamma_local.mean().item()
        metrics["gamma_local_std"] = gamma_local.std().item()

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


class CurrentMemory(_BaseMemory):
    """A: Current production (bulk + gradient seeding)."""
    def _compute_mass_gen(self, E, I, M, config, manifold):
        diseq = E - I
        diseq2 = diseq.pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        mass_gen_bulk = gamma_local * diseq2

        grad_u = (torch.roll(diseq, -1, 0) - torch.roll(diseq, 1, 0)) / 2.0
        grad_v = (torch.roll(diseq, -1, 1) - torch.roll(diseq, 1, 1)) / 2.0
        grad_diseq2 = grad_u.pow(2) + grad_v.pow(2)
        mass_gen_boundary = gamma_local * grad_diseq2

        mass_gen = mass_gen_bulk + mass_gen_boundary

        M_cap = config.field_scale / 5.0
        sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
        degen = None  # skip degeneracy for spike
        mass_gen = mass_gen * sat_cap
        return mass_gen, gamma_local


class DecayGradientMemory(_BaseMemory):
    """D: Gradient seeding decays as system becomes mass-rich."""
    def _compute_mass_gen(self, E, I, M, config, manifold):
        diseq = E - I
        diseq2 = diseq.pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        mass_gen_bulk = gamma_local * diseq2

        grad_u = (torch.roll(diseq, -1, 0) - torch.roll(diseq, 1, 0)) / 2.0
        grad_v = (torch.roll(diseq, -1, 1) - torch.roll(diseq, 1, 1)) / 2.0
        grad_diseq2 = grad_u.pow(2) + grad_v.pow(2)

        # Gradient seeding decays exponentially as mean mass grows
        M_cap = config.field_scale / 5.0
        decay = torch.exp(-M.mean() / M_cap)
        mass_gen_boundary = gamma_local * grad_diseq2 * decay

        mass_gen = mass_gen_bulk + mass_gen_boundary

        sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
        mass_gen = mass_gen * sat_cap
        return mass_gen, gamma_local


class AdaptiveGradientMemory(_BaseMemory):
    """E: Gradient seeding weakens where mass already exists."""
    def _compute_mass_gen(self, E, I, M, config, manifold):
        diseq = E - I
        diseq2 = diseq.pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        mass_gen_bulk = gamma_local * diseq2

        grad_u = (torch.roll(diseq, -1, 0) - torch.roll(diseq, 1, 0)) / 2.0
        grad_v = (torch.roll(diseq, -1, 1) - torch.roll(diseq, 1, 1)) / 2.0
        grad_diseq2 = grad_u.pow(2) + grad_v.pow(2)

        # Gradient term suppressed per-cell where M is already high
        # 1/(1+M) -> 1 when M=0 (empty, full seeding), -> small when M large
        grad_suppression = 1.0 / (1.0 + M)
        mass_gen_boundary = gamma_local * grad_diseq2 * grad_suppression

        mass_gen = mass_gen_bulk + mass_gen_boundary

        M_cap = config.field_scale / 5.0
        sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
        mass_gen = mass_gen * sat_cap
        return mass_gen, gamma_local


class CombinedMemory(_BaseMemory):
    """F: Adaptive gradient + bulk uses phi-scaled gamma."""
    def _compute_mass_gen(self, E, I, M, config, manifold):
        diseq = E - I
        diseq2 = diseq.pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        mass_gen_bulk = gamma_local * diseq2

        grad_u = (torch.roll(diseq, -1, 0) - torch.roll(diseq, 1, 0)) / 2.0
        grad_v = (torch.roll(diseq, -1, 1) - torch.roll(diseq, 1, 1)) / 2.0
        grad_diseq2 = grad_u.pow(2) + grad_v.pow(2)

        # Adaptive gradient: suppressed where M exists
        grad_suppression = 1.0 / (1.0 + M)
        mass_gen_boundary = gamma_local * grad_diseq2 * grad_suppression

        mass_gen = mass_gen_bulk + mass_gen_boundary

        M_cap = config.field_scale / 5.0
        sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
        mass_gen = mass_gen * sat_cap
        return mass_gen, gamma_local


# ---------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------

def build_pipeline(gravity_op=None, memory_op=None):
    gravity = gravity_op or GravitationalCollapseOperator()
    memory = memory_op or MemoryOperator()
    return Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        memory, PhiCascadeOperator(),
        gravity,
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


def run_variant(name, gravity_op, memory_op, device, ticks=10000):
    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True, actualization_threshold=0.05,
    )
    torch.manual_seed(42)
    pipeline = build_pipeline(gravity_op, memory_op)
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    checkpoints = [1000, 2000, 5000, 7500, 10000]
    results = {}
    cp_idx = 0

    for tick in range(1, ticks + 1):
        engine.tick()
        if cp_idx < len(checkpoints) and tick == checkpoints[cp_idx]:
            cp_idx += 1
            met = engine.state.metrics
            M = engine.state.M
            E = engine.state.E
            I_field = engine.state.I

            results[tick] = {
                "f_local": met.get("f_local_mean", 0),
                "gamma_local": met.get("gamma_local_mean", 0),
                "alpha_local": met.get("alpha_local_mean", 0),
                "G_local": met.get("G_local_mean", 0),
                "lambda_local": met.get("lambda_local_mean", 0),
                "M_mean": M.mean().item(),
                "M_std": M.std().item(),
                "M_max": M.max().item(),
                "xi_s_mean": met.get("xi_s_mean", 0),
                "xi_mod_mean": met.get("xi_mod_mean", 0),
                "frac_empty": (M < 0.1).float().mean().item(),
                "gen_rate": met.get("mass_generation_rate", 0),
            }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test combinations of gravity and memory variants
    VARIANTS = {
        "A_current":          (None, None),  # current production
        "B_phiXiMod":         (GravityPhiXiMod(), None),  # phi-scaled xi_mod gravity only
        "C_decayGrad":        (None, DecayGradientMemory()),  # decay gradient memory only
        "D_adaptGrad":        (None, AdaptiveGradientMemory()),  # adaptive gradient memory only
        "E_phiXi+adaptGrad":  (GravityPhiXiMod(), AdaptiveGradientMemory()),  # combined
        "F_phiXi+decayGrad":  (GravityPhiXiMod(), DecayGradientMemory()),  # combined
    }

    print(f"\n{'='*115}")
    print(f"  LATE-TIME G_LOCAL OVERSHOOT SPIKE")
    print(f"  Device: {device} | Grid: 128x64 | 10000 ticks per variant")
    print(f"  Problem: G_local hits 1/phi^2 at t=5K (0.8%) but drifts to 13.6% by t=10K")
    print(f"{'='*115}")

    all_results = {}
    for name, (grav_op, mem_op) in VARIANTS.items():
        t0 = time.time()
        print(f"\n  [{name}] ...", end="", flush=True)
        results = run_variant(name, grav_op, mem_op, device)
        elapsed = time.time() - t0
        print(f" {elapsed:.0f}s")
        all_results[name] = results

    # --- G_local trajectory ---
    print(f"\n{'='*115}")
    print(f"  G_LOCAL TRAJECTORY (target = 1/phi^2 = {1/PHI**2:.4f})")
    print(f"{'='*115}")
    header = f"  {'Variant':<22s}"
    for tick in [1000, 2000, 5000, 7500, 10000]:
        header += f" | {'t'+str(tick//1000)+'K':>9s}"
    header += " | overshoot"
    print(header)
    print(f"  {'-'*22}" + ("-+-" + "-"*9) * 5 + "-+-----------")

    for name, results in all_results.items():
        row = f"  {name:<22s}"
        errs = []
        for tick in [1000, 2000, 5000, 7500, 10000]:
            r = results.get(tick, {})
            g = r.get("G_local", 0)
            err = pct_err(g, 1/PHI**2)
            errs.append(err)
            row += f" | {err:7.1f}% {grade(err)[0]}"
        # Overshoot = error at 10K - min error
        min_err = min(errs)
        overshoot = errs[-1] - min_err
        row += f" | {overshoot:+6.1f}%"
        print(row)

    # --- Full coupling errors at 5K and 10K ---
    for tick_label, tick in [("TICK 5000", 5000), ("TICK 10000", 10000)]:
        print(f"\n{'='*115}")
        print(f"  TIER 1 COUPLING ERRORS AT {tick_label}")
        print(f"{'='*115}")
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

    # --- Diagnostics ---
    print(f"\n{'='*115}")
    print(f"  DIAGNOSTICS AT TICK 10000")
    print(f"{'='*115}")
    print(f"  {'Variant':<22s} | {'xi_s':>6s} {'xi_mod':>7s} | {'M_mean':>7s} {'%empty':>7s} | {'gen_rate':>9s}")
    print(f"  {'-'*22}-+-{'-'*6}-{'-'*7}-+-{'-'*7}-{'-'*7}-+-{'-'*9}")

    for name, results in all_results.items():
        r = results.get(10000, {})
        if not r:
            continue
        print(f"  {name:<22s}"
              f" | {r.get('xi_s_mean',0):6.3f} {r.get('xi_mod_mean',0):7.4f}"
              f" | {r['M_mean']:7.3f} {r['frac_empty']:6.1%}"
              f" | {r['gen_rate']:9.6f}")

    # --- Drift ranking ---
    print(f"\n{'='*115}")
    print(f"  DRIFT RANKING (avg Tier 1 error at tick 10000)")
    print(f"{'='*115}")
    scores_5k = {}
    scores_10k = {}
    for name, results in all_results.items():
        for tick, store in [(5000, scores_5k), (10000, scores_10k)]:
            r = results.get(tick, {})
            if r:
                errs = [pct_err(r.get(m, 0), t) for m, t in TARGETS.items()]
                store[name] = sum(errs) / len(errs)

    for name in sorted(scores_10k, key=scores_10k.get):
        e5 = scores_5k.get(name, 0)
        e10 = scores_10k[name]
        drift = e10 - e5
        g5 = pct_err(all_results[name].get(5000, {}).get("G_local", 0), 1/PHI**2)
        g10 = pct_err(all_results[name].get(10000, {}).get("G_local", 0), 1/PHI**2)
        print(f"  {name:<22s}  t5k={e5:5.1f}%  t10k={e10:5.1f}%  drift={drift:+5.1f}%"
              f"  G(5k)={g5:5.1f}%  G(10k)={g10:5.1f}%")

    best = min(scores_10k, key=scores_10k.get)
    print(f"\n  >>> Best at 10K: {best} ({scores_10k[best]:.1f}% avg error)")
    print()


if __name__ == "__main__":
    main()
