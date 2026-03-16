"""Diagnose gravity behavior -- trace local vs global dynamics.

Runs simulation and dumps spatial statistics at checkpoints:
- Where is mass concentrating? (histogram + spatial stats)
- What does G_local look like in high-M vs low-M regions?
- Is the Poisson solver creating global attraction or local clumping?
- How much mass is at the cap? How much Landauer reinjection?
"""

import math
import os
import sys
import time

re_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
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


def analyze_state(state, config, tick):
    """Detailed spatial analysis of the field state."""
    E, I, M = state.E, state.I, state.M
    metrics = state.metrics

    M_cap = config.field_scale / 5.0
    nu, nv = M.shape

    # Mass distribution
    M_flat = M.flatten()
    M_mean = M.mean().item()
    M_max = M.max().item()
    M_min = M.min().item()
    M_std = M.std().item()

    # Fraction at cap
    near_cap = (M > M_cap * 0.9).float()
    frac_cap = near_cap.mean().item()

    # Fraction with significant mass (> 10% of cap)
    has_mass = (M > M_cap * 0.1).float()
    frac_mass = has_mass.mean().item()

    # G_local computation (same as gravity operator)
    M2 = M.pow(2)
    diseq2 = (E - I).pow(2)
    G_local = M2 / (M2 + diseq2 + 1e-12)

    # G_local in high-M vs low-M regions
    high_M_mask = M > M_mean
    low_M_mask = M <= M_mean
    G_high = G_local[high_M_mask].mean().item() if high_M_mask.any() else 0
    G_low = G_local[low_M_mask].mean().item() if low_M_mask.any() else 0

    # Disequilibrium in high-M vs low-M regions
    diseq = (E - I).abs()
    diseq_high = diseq[high_M_mask].mean().item() if high_M_mask.any() else 0
    diseq_low = diseq[low_M_mask].mean().item() if low_M_mask.any() else 0

    # Poisson potential analysis
    # Recompute the gravitational potential to analyze it
    ku = torch.arange(nu, device=M.device, dtype=torch.float64)
    kv = torch.arange(nv, device=M.device, dtype=torch.float64)
    ku_grid, kv_grid = torch.meshgrid(ku, kv, indexing='ij')
    eigenvalues = (
        2.0 * (torch.cos(2.0 * torch.pi * ku_grid / nu) - 1.0) +
        2.0 * (torch.cos(2.0 * torch.pi * kv_grid / nv) - 1.0)
    )
    eigenvalues[0, 0] = 1.0
    inv_lap = 1.0 / eigenvalues
    inv_lap[0, 0] = 0.0

    M_hat = torch.fft.fft2(M)
    phi_hat = M_hat * inv_lap
    phi = torch.fft.ifft2(phi_hat).real

    # Potential in high-M vs low-M regions
    phi_high = phi[high_M_mask].mean().item() if high_M_mask.any() else 0
    phi_low = phi[low_M_mask].mean().item() if low_M_mask.any() else 0

    # Gradient magnitude (force strength)
    grad_u = (torch.roll(phi, -1, 0) - torch.roll(phi, 1, 0)) / 2.0
    grad_v = (torch.roll(phi, -1, 1) - torch.roll(phi, 1, 1)) / 2.0
    grad_mag = (grad_u.pow(2) + grad_v.pow(2)).sqrt()
    grad_high = grad_mag[high_M_mask].mean().item() if high_M_mask.any() else 0
    grad_low = grad_mag[low_M_mask].mean().item() if low_M_mask.any() else 0

    # Flux analysis: where is mass actually moving?
    flux_u = M * grad_u
    flux_v = M * grad_v
    div_flux = (
        (torch.roll(flux_u, -1, 0) - torch.roll(flux_u, 1, 0)) / 2.0 +
        (torch.roll(flux_v, -1, 1) - torch.roll(flux_v, 1, 1)) / 2.0
    )
    # Positive div = mass accumulating, negative = mass leaving
    div_high = div_flux[high_M_mask].mean().item() if high_M_mask.any() else 0
    div_low = div_flux[low_M_mask].mean().item() if low_M_mask.any() else 0

    # Net gravitational change after G_local coupling
    dM_grav = G_local * div_flux * config.dt
    dM_grav_high = dM_grav[high_M_mask].mean().item() if high_M_mask.any() else 0
    dM_grav_low = dM_grav[low_M_mask].mean().item() if low_M_mask.any() else 0

    # Mass generation saturation
    sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
    sat_high = sat_cap[high_M_mask].mean().item() if high_M_mask.any() else 0
    sat_low = sat_cap[low_M_mask].mean().item() if low_M_mask.any() else 0

    # Key coupling constants
    alpha_local_mean = metrics.get("alpha_local_mean", 0)
    lambda_local_mean = metrics.get("lambda_local_mean", 0)
    f_local_mean = metrics.get("f_local_mean", 0)
    gamma_local_mean = metrics.get("gamma_local_mean", 0)
    G_local_mean = metrics.get("G_local_mean", 0)

    # Landauer reinjection
    landauer = metrics.get("landauer_reinjection", 0)
    crystallisation = metrics.get("crystallisation", 0)

    return {
        "tick": tick,
        # Mass stats
        "M_mean": M_mean, "M_max": M_max, "M_min": M_min, "M_std": M_std,
        "M_cap": M_cap, "frac_cap": frac_cap, "frac_mass": frac_mass,
        # Local vs global gravity
        "G_high_M": G_high, "G_low_M": G_low, "G_ratio": G_high / max(G_low, 1e-12),
        # Disequilibrium asymmetry
        "diseq_high": diseq_high, "diseq_low": diseq_low,
        # Potential asymmetry
        "phi_high": phi_high, "phi_low": phi_low,
        # Force asymmetry
        "grad_high": grad_high, "grad_low": grad_low,
        # Flux asymmetry (THE KEY: is mass flowing toward clumps?)
        "div_high": div_high, "div_low": div_low,
        "dM_grav_high": dM_grav_high, "dM_grav_low": dM_grav_low,
        # Saturation
        "sat_high": sat_high, "sat_low": sat_low,
        # Coupling constants
        "alpha": alpha_local_mean, "lambda": lambda_local_mean,
        "f_local": f_local_mean, "gamma": gamma_local_mean, "G_mean": G_local_mean,
        # Normalization feedback
        "landauer": landauer, "crystallisation": crystallisation,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True,
        actualization_threshold=0.05,
    )
    torch.manual_seed(42)
    pipeline = build_pipeline()
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    checkpoints = [100, 500, 1000, 2000, 5000, 10000]
    results = []
    t0 = time.time()
    cp_idx = 0

    print(f"\n{'=' * 90}")
    print(f"  GRAVITY DIAGNOSTIC -- Local vs Global Dynamics")
    print(f"  Device: {device} | Grid: {config.nu}x{config.nv} | M_cap: {config.field_scale/5:.1f}")
    print(f"{'=' * 90}")

    for tick in range(1, 10001):
        engine.tick()
        if cp_idx < len(checkpoints) and tick == checkpoints[cp_idx]:
            cp_idx += 1
            r = analyze_state(engine.state, config, tick)
            results.append(r)
            elapsed = time.time() - t0
            print(f"\n  --- Tick {tick} ({elapsed:.0f}s) ---")

    # Print comparative table
    print(f"\n{'=' * 90}")
    print(f"  MASS DISTRIBUTION")
    print(f"{'=' * 90}")
    print(f"  {'Tick':>6s} | {'M_mean':>8s} {'M_max':>8s} {'M_std':>8s} | {'%cap':>6s} {'%mass':>6s} | {'Landau':>8s}")
    print(f"  {'-'*6}-+-{'-'*8}-{'-'*8}-{'-'*8}-+-{'-'*6}-{'-'*6}-+-{'-'*8}")
    for r in results:
        print(f"  {r['tick']:6d} | {r['M_mean']:8.4f} {r['M_max']:8.4f} {r['M_std']:8.4f} "
              f"| {r['frac_cap']:5.1%} {r['frac_mass']:5.1%} | {r['landauer']:8.4f}")

    print(f"\n{'=' * 90}")
    print(f"  GRAVITY: LOCAL vs GLOBAL ASYMMETRY")
    print(f"{'=' * 90}")
    print(f"  {'Tick':>6s} | {'G(hi)':>7s} {'G(lo)':>7s} {'ratio':>7s} | "
          f"{'diseq_hi':>8s} {'diseq_lo':>8s} | {'dM_hi':>10s} {'dM_lo':>10s}")
    print(f"  {'-'*6}-+-{'-'*7}-{'-'*7}-{'-'*7}-+-{'-'*8}-{'-'*8}-+-{'-'*10}-{'-'*10}")
    for r in results:
        print(f"  {r['tick']:6d} | {r['G_high_M']:7.4f} {r['G_low_M']:7.4f} {r['G_ratio']:7.1f} | "
              f"{r['diseq_high']:8.4f} {r['diseq_low']:8.4f} | "
              f"{r['dM_grav_high']:10.2e} {r['dM_grav_low']:10.2e}")

    print(f"\n{'=' * 90}")
    print(f"  POTENTIAL & FLUX ASYMMETRY")
    print(f"{'=' * 90}")
    print(f"  {'Tick':>6s} | {'phi_hi':>8s} {'phi_lo':>8s} | "
          f"{'grad_hi':>8s} {'grad_lo':>8s} | {'div_hi':>10s} {'div_lo':>10s}")
    print(f"  {'-'*6}-+-{'-'*8}-{'-'*8}-+-{'-'*8}-{'-'*8}-+-{'-'*10}-{'-'*10}")
    for r in results:
        print(f"  {r['tick']:6d} | {r['phi_high']:8.4f} {r['phi_low']:8.4f} | "
              f"{r['grad_high']:8.4f} {r['grad_low']:8.4f} | "
              f"{r['div_high']:10.2e} {r['div_low']:10.2e}")

    print(f"\n{'=' * 90}")
    print(f"  SATURATION & FEEDBACK")
    print(f"{'=' * 90}")
    print(f"  {'Tick':>6s} | {'sat_hi':>7s} {'sat_lo':>7s} | "
          f"{'crystal':>8s} | {'alpha':>7s} {'lambda':>7s} {'G_mean':>7s} {'gamma':>7s}")
    print(f"  {'-'*6}-+-{'-'*7}-{'-'*7}-+-{'-'*8}-+-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")
    for r in results:
        print(f"  {r['tick']:6d} | {r['sat_high']:7.4f} {r['sat_low']:7.4f} | "
              f"{r['crystallisation']:8.4f} | "
              f"{r['alpha']:7.4f} {r['lambda']:7.4f} {r['G_mean']:7.4f} {r['gamma']:7.4f}")

    # Diagnosis
    print(f"\n{'=' * 90}")
    print(f"  DIAGNOSIS")
    print(f"{'=' * 90}")

    final = results[-1]
    early = results[2]  # tick 1000

    if final['G_ratio'] > 5:
        print(f"  [!] G_local ratio (high/low M) = {final['G_ratio']:.1f}x")
        print(f"      Gravity is {final['G_ratio']:.0f}x stronger in dense regions.")
        print(f"      This is positive feedback: mass attracts mass, G increases,")
        print(f"      which attracts more mass. No counter-mechanism.")

    if final['frac_cap'] > 0.1:
        print(f"  [!] {final['frac_cap']:.1%} of mass structures at cap")
        print(f"      Mass pileup at field_scale/5 = {final['M_cap']:.1f}")
        print(f"      Once at cap, G_local -> 1.0 (gravity maxed out)")
        print(f"      but mass can't grow further, creating dead zones.")

    if final['dM_grav_high'] > 0:
        print(f"  [!] Gravity ADDING mass to already-dense regions: {final['dM_grav_high']:.2e}")
        print(f"      This is runaway clumping (no web formation).")
    elif final['dM_grav_high'] < 0:
        print(f"  [OK] Gravity removing mass from dense regions: {final['dM_grav_high']:.2e}")

    if final['diseq_high'] < final['diseq_low'] * 0.5:
        print(f"  [!] Disequilibrium dead in dense regions: {final['diseq_high']:.4f} vs {final['diseq_low']:.4f}")
        print(f"      Dense regions are thermalized (E~I), meaning:")
        print(f"      - G_local -> 1 (gravity maxed)")
        print(f"      - gamma_local -> 0 (no new mass generation)")
        print(f"      - alpha_local -> 1 (collapse attraction maxed)")
        print(f"      All coupling constants lose their attractor dynamics.")

    if final['sat_high'] < 0.1:
        print(f"  [!] Saturation throttle fully engaged in dense regions: {final['sat_high']:.4f}")
        print(f"      Mass generation capped, but gravity still pulls mass in.")
        print(f"      Net effect: gravity piles up what already exists.")

    print()


if __name__ == "__main__":
    main()
