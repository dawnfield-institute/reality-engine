"""Spike: code-level interventions for G_local drift.

Config sweeps showed G_local is the universal bottleneck -- no parameter
tuning fixes the 45-67% error at tick 5000. The root cause: as mass
bimodalizes (17% at cap, 50% empty), sparse cells contribute G~0,
dragging the mean far below 1/phi^2 = 0.382.

Code-level interventions tested:

1. Non-local Landauer: when normalization removes mass at cap, redistribute
   to low-density cells instead of local E+I. Creates mass recycling.

2. Sqrt Poisson source: solve nabla^2 Phi = sqrt(M) instead of M.
   Makes gravitational potential less sensitive to mass peaks, creating
   a smoother, more diffuse force field. Dense regions still attract
   but don't dominate as completely.

3. Density-contrast gravity: flux = (M - M_mean) * grad_Phi instead of
   M * grad_Phi. Overdense regions attract, underdense regions repel.
   Creates web-like structure (filaments between voids).

4. Adaptive diffusion: diffusion coefficient D_local = D * (M_mean/M)^p.
   Sparse regions get stronger diffusion (mass flows in from neighbors),
   dense regions get weaker diffusion (mass stays put, but also gravity
   is modulated by xi_mod to not pile up more).

Each is tested as a patched operator, no permanent code changes.
"""

import math
import os
import sys
import time
import copy
from typing import Optional

re_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))
if re_path not in sys.path:
    sys.path.insert(0, re_path)

import torch

from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.engine.state import FieldState
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


# ---------------------------------------------------------------
# Intervention 1: Non-local Landauer reinjection
# ---------------------------------------------------------------
class NonLocalNormalizationOperator:
    """Normalization with non-local Landauer: excess mass goes to sparse cells."""

    def __init__(self):
        self._initial_pac = None

    @property
    def name(self):
        return "normalization"

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        if not config.enable_normalization:
            return state

        s = config.field_scale
        M_cap = s / 5.0

        M_floored = torch.clamp(state.M, min=0.0)
        M_new = torch.clamp(M_floored, max=M_cap)
        dM_removed = M_floored - M_new  # >= 0

        # NON-LOCAL: redistribute removed mass to low-density cells
        total_removed = dM_removed.sum()
        if total_removed > 1e-10:
            # Weight by inverse mass: sparse cells get more
            inv_M = 1.0 / (M_new + 0.1)
            weights = inv_M / inv_M.sum()
            M_new = M_new + total_removed * weights

        # Energy accounting: the NET change in M must come from E+I
        dM_net = M_new - state.M  # per-cell net change
        reinjection_drain = dM_net * 0.5
        E_cur = state.E - reinjection_drain
        I_cur = state.I - reinjection_drain

        # Tanh safety clamp
        E_clamped = s * torch.tanh(E_cur / s)
        I_clamped = s * torch.tanh(I_cur / s)
        E_loss = E_cur - E_clamped
        I_loss = I_cur - I_clamped

        # Bounded QBE cross-injection
        E_inject = torch.clamp(I_loss, min=-s - E_clamped, max=s - E_clamped)
        I_inject = torch.clamp(E_loss, min=-s - I_clamped, max=s - I_clamped)
        E_new = E_clamped + E_inject
        I_new = I_clamped + I_inject

        E_remainder = I_loss - E_inject
        I_remainder = E_loss - I_inject
        crystallised = E_remainder + I_remainder
        M_new = M_new + crystallised
        M_new = torch.clamp(M_new, min=0.0)

        # Global PAC safety
        if self._initial_pac is None:
            self._initial_pac = (E_new + I_new + M_new).sum().item()
        current_pac = (E_new + I_new + M_new).sum().item()
        pac_residual = self._initial_pac - current_pac
        if abs(pac_residual) > 1e-8:
            correction = pac_residual / (2.0 * E_new.numel())
            E_new = E_new + correction
            I_new = I_new + correction

        metrics = dict(state.metrics)
        metrics["landauer_reinjection"] = total_removed.item()
        metrics["crystallisation"] = crystallised.sum().item()
        metrics["pac_correction"] = pac_residual
        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


# ---------------------------------------------------------------
# Intervention 2: Sqrt Poisson source
# ---------------------------------------------------------------
class SqrtGravityOperator(GravitationalCollapseOperator):
    """Gravity with nabla^2 Phi = sqrt(M) instead of M."""

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
        xi_mod = torch.sqrt(xi_s / (xi_s + 1.0))
        G_local = G_mass * xi_mod

        # KEY CHANGE: sqrt(M) as Poisson source
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
        mass_created = M_new - M_candidate
        pac_leak = mass_created * 0.5
        E_new = E - pac_leak
        I_new = I - pac_leak

        metrics = dict(state.metrics)
        metrics["G_local_mean"] = G_local.mean().item()
        metrics["G_local_std"] = G_local.std().item()
        metrics["xi_s_mean"] = xi_s.mean().item()
        metrics["xi_s_std"] = xi_s.std().item()
        metrics["xi_mod_mean"] = xi_mod.mean().item()
        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


# ---------------------------------------------------------------
# Intervention 3: Density-contrast flux
# ---------------------------------------------------------------
class DensityContrastGravityOperator(GravitationalCollapseOperator):
    """Gravity with flux = (M - M_mean) * grad_Phi."""

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
        xi_mod = torch.sqrt(xi_s / (xi_s + 1.0))
        G_local = G_mass * xi_mod

        phi = self._solve_poisson(M)
        grad_phi_u = (torch.roll(phi, -1, 0) - torch.roll(phi, 1, 0)) / 2.0
        grad_phi_v = (torch.roll(phi, -1, 1) - torch.roll(phi, 1, 1)) / 2.0

        # KEY CHANGE: density contrast flux
        delta_M = M - M.mean()
        flux_u = delta_M * grad_phi_u
        flux_v = delta_M * grad_phi_v

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

        metrics = dict(state.metrics)
        metrics["G_local_mean"] = G_local.mean().item()
        metrics["G_local_std"] = G_local.std().item()
        metrics["xi_s_mean"] = xi_s.mean().item()
        metrics["xi_s_std"] = xi_s.std().item()
        metrics["xi_mod_mean"] = xi_mod.mean().item()
        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


# ---------------------------------------------------------------
# Intervention 4: Adaptive diffusion in memory operator
# ---------------------------------------------------------------
class AdaptiveDiffusionMemoryOperator(MemoryOperator):
    """Memory with D_local = D * (M_mean / M)^0.5 -- sparse regions diffuse faster."""

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        m = self._get_manifold(state)
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        disequilibrium = E - I
        diseq2 = disequilibrium.pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        mass_gen = gamma_local * diseq2
        M_cap = config.field_scale / 5.0
        sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
        degen = state.metrics.get("degeneracy_level", None)
        sat_degen = (1.0 - degen) if degen is not None else 1.0
        saturation = sat_cap * sat_degen
        mass_gen = mass_gen * saturation

        M_safe = M + 1e-6
        lap_M2 = m.laplacian(M * M)
        quantum_pressure = -config.quantum_pressure_coeff * lap_M2 / M_safe

        # KEY CHANGE: adaptive diffusion
        lap_M = m.laplacian(M)
        M_mean = M.mean()
        D_local = config.mass_diffusion_coeff * torch.sqrt(M_mean / (M + 0.01))
        diffusion = D_local * lap_M

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


# ---------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------

def build_pipeline_with(gravity_cls=None, memory_cls=None, norm_cls=None):
    gravity = gravity_cls() if gravity_cls else GravitationalCollapseOperator()
    memory = memory_cls() if memory_cls else MemoryOperator()
    norm = norm_cls() if norm_cls else NormalizationOperator()
    return Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        memory, PhiCascadeOperator(),
        gravity,
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        norm, SECTrackingOperator(),
        AdaptiveOperator(), TimeEmergenceOperator(),
    ])


INTERVENTIONS = {
    "A_baseline": dict(pipeline_args={}),
    "B_nonlocal_landauer": dict(pipeline_args=dict(norm_cls=NonLocalNormalizationOperator)),
    "C_sqrt_poisson": dict(pipeline_args=dict(gravity_cls=SqrtGravityOperator)),
    "D_density_contrast": dict(pipeline_args=dict(gravity_cls=DensityContrastGravityOperator)),
    "E_adaptive_diffusion": dict(pipeline_args=dict(memory_cls=AdaptiveDiffusionMemoryOperator)),
    "F_sqrt_poisson+nonlocal": dict(pipeline_args=dict(
        gravity_cls=SqrtGravityOperator,
        norm_cls=NonLocalNormalizationOperator,
    )),
    "G_contrast+nonlocal": dict(pipeline_args=dict(
        gravity_cls=DensityContrastGravityOperator,
        norm_cls=NonLocalNormalizationOperator,
    )),
    "H_contrast+adaptdiff": dict(pipeline_args=dict(
        gravity_cls=DensityContrastGravityOperator,
        memory_cls=AdaptiveDiffusionMemoryOperator,
    )),
    "I_all_three": dict(pipeline_args=dict(
        gravity_cls=DensityContrastGravityOperator,
        memory_cls=AdaptiveDiffusionMemoryOperator,
        norm_cls=NonLocalNormalizationOperator,
    )),
}


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


def run_intervention(name, pipeline_args, device, ticks=5000):
    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True,
        actualization_threshold=0.05,
    )
    torch.manual_seed(42)
    pipeline = build_pipeline_with(**pipeline_args)
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
                "xi_mod_mean": met.get("xi_mod_mean", 0),
            }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*105}")
    print(f"  CODE-LEVEL INTERVENTION SWEEP")
    print(f"  Device: {device} | Grid: 128x64 | 5000 ticks per run")
    print(f"  {len(INTERVENTIONS)} interventions")
    print(f"{'='*105}")

    all_results = {}
    for name, spec in INTERVENTIONS.items():
        t0 = time.time()
        print(f"\n  [{name}] ...", end="", flush=True)
        try:
            results = run_intervention(name, spec["pipeline_args"], device)
            elapsed = time.time() - t0
            print(f" {elapsed:.0f}s")
            all_results[name] = results
        except Exception as e:
            print(f" FAILED: {e}")
            all_results[name] = {}

    # --- Report ---
    for tick_label, tick in [("TICK 1000", 1000), ("TICK 5000", 5000)]:
        print(f"\n{'='*105}")
        print(f"  TIER 1 COUPLING ERRORS AT {tick_label}")
        print(f"{'='*105}")
        header = f"  {'Intervention':<28s}"
        for metric in TARGETS:
            header += f" | {metric:>12s}"
        header += " |  avg_err"
        print(header)
        print(f"  {'-'*28}" + ("-+-" + "-"*12) * len(TARGETS) + "-+---------")

        for name, results in all_results.items():
            r = results.get(tick, {})
            if not r:
                print(f"  {name:<28s}  -- FAILED --")
                continue
            row = f"  {name:<28s}"
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
    print(f"  {'Intervention':<28s} | {'M_mean':>7s} {'M_std':>7s} {'%cap':>6s} {'%empty':>7s} | {'xi_mod':>7s}")
    print(f"  {'-'*28}-+-{'-'*7}-{'-'*7}-{'-'*6}-{'-'*7}-+-{'-'*7}")

    for name, results in all_results.items():
        r = results.get(5000, {})
        if not r:
            print(f"  {name:<28s}  -- FAILED --")
            continue
        print(f"  {name:<28s}"
              f" | {r['M_mean']:7.3f} {r['M_std']:7.3f} {r['frac_cap']:5.1%} {r['frac_empty']:6.1%}"
              f" | {r['xi_mod_mean']:7.4f}")

    # --- Drift ranking ---
    print(f"\n{'='*105}")
    print(f"  DRIFT RANKING (avg Tier 1 error at tick 5000, lower = better)")
    print(f"{'='*105}")

    scores = {}
    for name, results in all_results.items():
        r = results.get(5000, {})
        if not r:
            continue
        errs = [pct_err(r.get(m, 0), t) for m, t in TARGETS.items()]
        scores[name] = sum(errs) / len(errs)

    for name in sorted(scores, key=scores.get):
        r1 = all_results[name].get(1000, {})
        e1 = sum(pct_err(r1.get(m, 0), t) for m, t in TARGETS.items()) / len(TARGETS) if r1 else 0
        e5 = scores[name]
        drift = e5 - e1
        g_err = pct_err(all_results[name].get(5000, {}).get("G_local", 0), 1/PHI**2)
        bar = "#" * int(min(e5, 60) / 2)
        print(f"  {name:<28s}  t1k={e1:5.1f}%  t5k={e5:5.1f}%  drift={drift:+5.1f}%  G={g_err:5.1f}%  {bar}")

    best = min(scores, key=scores.get)
    print(f"\n  >>> Best overall: {best} ({scores[best]:.1f}% avg error at tick 5000)")
    print()


if __name__ == "__main__":
    main()
