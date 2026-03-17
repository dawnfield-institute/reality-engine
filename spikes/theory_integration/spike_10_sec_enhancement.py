"""Spike 10: SEC Enhancement Factor — Simulator vs JWST Theory Prediction

Bridge between two DFT results:
  1. JWST paper: SEC enhancement epsilon = phi^(1 + (k_eq - k)/2) ~ 1.62x at high z
     Predicts duty cycle increases from ~60% (z~0) to ~81% (z>8)
  2. Spike 09: Simulator shows coupling constants converge then drift (RG flow)
     "NOW" tick at 8450/20000 = 42% of lifecycle

This spike:
1. Runs 20K tick simulation with fine-grained sampling
2. Computes "effective duty cycle" at each tick from gamma_local
   (gamma = I^2/(E^2+I^2) ~ information fraction ~ duty cycle proxy)
3. Maps simulator ticks to redshift z via the NOW tick anchor
4. Extracts the SEC enhancement factor: ratio of growth efficiency at each z vs z=0
5. Compares against the JWST prediction: epsilon(z) = phi^(1 + (k_eq - k(z))/2)
6. Checks whether G_local running matches the SMBH growth advantage at high z

If the simulator reproduces SEC enhancement from field dynamics alone,
it validates the JWST paper's framework independently.
"""

import math
import os
import sys
import time

import numpy as np

re_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if re_path not in sys.path:
    sys.path.insert(0, re_path)

import torch

from harness import (
    default_pipeline, default_config, TARGETS,
    PHI, GAMMA_EM, LN2, PHI_INV, PHI_INV2,
)
from src.v3.engine.engine import Engine


# DFT constants from JWST paper
XI = GAMMA_EM + math.log(PHI)  # 1.0584...
K_EQ = 2  # equilibrium cascade depth (z ~ 0)
PHI_DUTY_EQ = PHI / (PHI + 1)  # 0.618... duty cycle at equilibrium


def sec_enhancement_theory(k):
    """SEC enhancement factor from JWST paper.

    epsilon = phi^(1 + (k_eq - k)/2)
    At k=k_eq (z~0): epsilon = phi^1 = 1.618 (but normalized to 1.0 at equilibrium)
    At k=0 (high z): epsilon = phi^(1 + k_eq/2) = phi^2 = 2.618

    We normalize so epsilon(k_eq) = 1.0:
    epsilon_norm(k) = phi^((k_eq - k)/2)
    """
    return PHI ** ((K_EQ - k) / 2.0)


def sec_duty_cycle_theory(k):
    """SEC predicted duty cycle at cascade depth k.

    duty(k) = R(k) / (R(k) + 1)
    R(k) = phi^(1 + (k_eq - k)/2)
    """
    r = PHI ** (1 + (K_EQ - k) / 2.0)
    return r / (r + 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticks = 20000
    sample_every = 50

    print("=" * 90)
    print("  SPIKE 10: SEC Enhancement Factor -- Simulator vs JWST Theory")
    print(f"  Device: {device} | Grid: 128x64 | {ticks} ticks | sample every {sample_every}")
    print(f"  JWST theory: epsilon = phi^((k_eq - k)/2), duty(z=0) = {PHI_DUTY_EQ:.4f}")
    print(f"  Xi = {XI:.4f}")
    print("=" * 90)

    torch.manual_seed(42)
    config = default_config(device=device)
    pipeline = default_pipeline()
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    # Collect time series
    target_vals = {name: target for name, (_, target) in TARGETS.items()}
    series = {name: [] for name in TARGETS}
    series["gamma_raw"] = []       # raw gamma_local (duty cycle proxy)
    series["M_mean"] = []
    series["M_max"] = []
    series["diseq_mean"] = []      # |E - I| mean
    series["E_mean"] = []
    series["I_mean"] = []
    series["xi_s"] = []
    series["dM_gen"] = []          # mass generation rate proxy
    tick_stamps = []

    t0 = time.time()
    prev_M_mean = 0.0
    for tick in range(1, ticks + 1):
        engine.tick()
        if tick % sample_every == 0:
            m = engine.state.metrics
            for name, (key, _) in TARGETS.items():
                series[name].append(m.get(key, 0))
            series["gamma_raw"].append(m.get("gamma_local_mean", 0))
            series["M_mean"].append(engine.state.M.mean().item())
            series["M_max"].append(engine.state.M.max().item())
            series["E_mean"].append(engine.state.E.mean().item())
            series["I_mean"].append(engine.state.I.mean().item())
            series["diseq_mean"].append(m.get("disequilibrium_mean", 0))
            series["xi_s"].append(m.get("xi_s_mean", 0))

            # Mass generation rate: dM/dt proxy
            cur_M = engine.state.M.mean().item()
            series["dM_gen"].append((cur_M - prev_M_mean) / sample_every)
            prev_M_mean = cur_M

            tick_stamps.append(tick)

            if tick % 5000 == 0:
                elapsed = time.time() - t0
                errs = {name: abs(series[name][-1] - target_vals[name]) / abs(target_vals[name]) * 100
                        for name in TARGETS}
                avg = sum(errs.values()) / len(errs)
                print(f"  Tick {tick:>6d} ({elapsed:>5.0f}s): avg_err={avg:.1f}%  "
                      f"gamma={series['gamma_raw'][-1]:.4f}  "
                      f"G={series['G_local'][-1]:.4f}  "
                      f"M_mean={series['M_mean'][-1]:.3f}")

    elapsed = time.time() - t0
    n_samples = len(tick_stamps)
    print(f"\n  Collection complete: {n_samples} samples in {elapsed:.0f}s")

    # Convert to numpy
    for k in series:
        series[k] = np.array(series[k])
    tick_stamps = np.array(tick_stamps)

    # ====================================================================
    # 1. FIND "NOW" TICK (same as spike 09)
    # ====================================================================
    errors = {}
    for name in TARGETS:
        errors[name] = np.abs(series[name] - target_vals[name]) / np.abs(target_vals[name]) * 100
    avg_errors = np.mean([errors[name] for name in TARGETS], axis=0)
    now_idx = np.argmin(avg_errors)
    now_tick = tick_stamps[now_idx]
    now_err = avg_errors[now_idx]

    print(f"\n  'NOW' tick: {now_tick} (avg error {now_err:.2f}%)")
    print(f"  gamma at NOW: {series['gamma_raw'][now_idx]:.4f} "
          f"(theory equilibrium: {PHI_INV:.4f})")

    # ====================================================================
    # 2. MAP TICKS TO REDSHIFT
    # ====================================================================
    # Anchor: NOW tick = z=0 (present universe)
    # Early ticks = high redshift (young universe)
    # Convention: z proportional to (NOW - tick)/NOW for tick < NOW
    #             z < 0 (future) for tick > NOW
    #
    # Scale factor a(t) ~ tick / now_tick
    # Redshift z = (a_now / a) - 1 = (now_tick / tick) - 1

    print(f"\n  TICK-TO-REDSHIFT MAPPING (anchored at tick {now_tick} = z=0)")
    print(f"  {'Tick':>6s} {'z':>7s} {'Epoch':>20s}")

    z_checkpoints = [
        (50,   "Planck era"),
        (100,  "Inflation"),
        (250,  "Nucleosynthesis"),
        (500,  "Radiation era"),
        (1000, "Matter domination"),
        (2000, "Structure formation"),
        (4000, "Galaxy assembly"),
        (now_tick, "NOW (z=0)"),
    ]

    for cp_tick, label in z_checkpoints:
        if cp_tick <= now_tick:
            z = (now_tick / cp_tick) - 1.0
            print(f"  {cp_tick:>6d} {z:>7.1f} {label:>20s}")

    # Add some future ticks
    for future_tick in [12000, 15000, 20000]:
        if future_tick > now_tick:
            z = (now_tick / future_tick) - 1.0  # negative z = future
            print(f"  {future_tick:>6d} {z:>+7.2f} {'Far future':>20s}")

    # ====================================================================
    # 3. SIMULATOR DUTY CYCLE vs THEORY
    # ====================================================================
    print(f"\n  DUTY CYCLE COMPARISON: Simulator gamma vs SEC Theory")
    print(f"  gamma_local IS the simulator's duty cycle (I^2 / (E^2 + I^2))")
    print(f"  SEC theory: duty(k) = R(k) / (R(k)+1), R(k) = phi^(1+(k_eq-k)/2)")
    print()
    print(f"  {'Tick':>6s} {'z':>7s} {'gamma_sim':>10s} {'duty_theory':>12s} "
          f"{'ratio':>7s} {'match':>8s}")
    print(f"  {'-'*6} {'-'*7} {'-'*10} {'-'*12} {'-'*7} {'-'*8}")

    # Map tick to cascade depth k: k = k_eq * (tick / now_tick)
    # At NOW: k = k_eq = 2. At tick=0: k = 0. At 2*NOW: k = 2*k_eq = 4.
    sample_ticks = [100, 250, 500, 1000, 2000, 3000, 4000, 5000,
                    6000, 7000, 8000, now_tick, 10000, 12000, 15000, 20000]

    sim_duties = []
    theory_duties = []

    for st in sample_ticks:
        idx = st // sample_every - 1
        if idx < 0 or idx >= n_samples:
            continue
        z = (now_tick / st) - 1.0 if st <= now_tick else (now_tick / st) - 1.0
        k = K_EQ * (st / now_tick)  # cascade depth
        gamma_sim = series["gamma_raw"][idx]
        duty_th = sec_duty_cycle_theory(k)

        ratio = gamma_sim / duty_th if duty_th > 0 else 0
        match = "OK" if abs(ratio - 1.0) < 0.15 else ("HIGH" if ratio > 1 else "LOW")
        if st == now_tick:
            match = "<-- NOW"

        print(f"  {st:>6d} {z:>+7.2f} {gamma_sim:>10.4f} {duty_th:>12.4f} "
              f"{ratio:>7.3f} {match:>8s}")

        sim_duties.append(gamma_sim)
        theory_duties.append(duty_th)

    sim_duties = np.array(sim_duties)
    theory_duties = np.array(theory_duties)
    corr = np.corrcoef(sim_duties, theory_duties)[0, 1]
    rmse = np.sqrt(np.mean((sim_duties - theory_duties)**2))

    print(f"\n  Correlation (sim vs theory duty): {corr:+.4f}")
    print(f"  RMSE: {rmse:.4f}")

    # ====================================================================
    # 4. SEC ENHANCEMENT FACTOR EXTRACTION
    # ====================================================================
    print(f"\n  SEC ENHANCEMENT FACTOR")
    print(f"  Ratio of growth efficiency at each epoch vs NOW")
    print(f"  Theory: epsilon(k) = phi^((k_eq - k)/2)")
    print()
    print(f"  {'Tick':>6s} {'z':>7s} {'eps_sim':>8s} {'eps_theory':>10s} "
          f"{'ratio':>7s} {'G_local':>8s} {'dM/dt':>10s}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*10} {'-'*7} {'-'*8} {'-'*10}")

    # Enhancement from simulator: use G_local as gravity coupling strength
    # At high z (early ticks), if G_local is higher, gravitational collapse
    # is more efficient -> SMBH growth faster -> enhancement > 1
    G_now = series["G_local"][now_idx]
    gamma_now = series["gamma_raw"][now_idx]

    for st in sample_ticks:
        idx = st // sample_every - 1
        if idx < 0 or idx >= n_samples:
            continue
        z = (now_tick / st) - 1.0
        k = K_EQ * (st / now_tick)

        # Simulator enhancement: composite of gravity strength + duty cycle
        # Growth ~ G_local * gamma_local (gravity coupling * actualization fraction)
        G_local = series["G_local"][idx]
        gamma_local = series["gamma_raw"][idx]
        growth_proxy = G_local * gamma_local
        growth_now = G_now * gamma_now
        eps_sim = growth_proxy / growth_now if growth_now > 0 else 0

        eps_th = sec_enhancement_theory(k)

        ratio = eps_sim / eps_th if eps_th > 0 else 0
        dM = series["dM_gen"][idx]

        print(f"  {st:>6d} {z:>+7.2f} {eps_sim:>8.3f} {eps_th:>10.3f} "
              f"{ratio:>7.3f} {G_local:>8.4f} {dM:>+10.2e}")

    # ====================================================================
    # 5. GRAVITY RUNNING vs SMBH REQUIREMENTS
    # ====================================================================
    print(f"\n  GRAVITY RUNNING CURVE")
    print(f"  JWST paper: PAC needs lighter seeds (10^5.3 vs 10^6.8 solar masses)")
    print(f"  This is because gravity is MORE EFFICIENT at high z")
    print(f"  Simulator prediction: G_local at different epochs")
    print()

    # G_local at key cosmological epochs
    cosmological = [
        (100,  "z~{:.0f} (GLASS-z12 epoch)"),
        (500,  "z~{:.0f} (GN-z11 epoch)"),
        (1000, "z~{:.0f} (UHZ-1 epoch)"),
        (3000, "z~{:.0f} (JADES-dormant)"),
    ]

    print(f"  {'Epoch':>30s} {'G_local':>8s} {'G/G_now':>8s} {'Advantage':>10s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*10}")

    for ct, label_tmpl in cosmological:
        idx = ct // sample_every - 1
        if idx < 0 or idx >= n_samples:
            continue
        z = (now_tick / ct) - 1.0
        G_val = series["G_local"][idx]
        G_ratio = G_val / G_now
        label = label_tmpl.format(z)
        advantage = f"{G_ratio:.2f}x" if G_ratio > 1 else f"{G_ratio:.2f}x (weaker)"
        print(f"  {label:>30s} {G_val:>8.4f} {G_ratio:>8.3f} {advantage:>10s}")

    print(f"  {'NOW (z=0)':>30s} {G_now:>8.4f} {1.0:>8.3f} {'reference':>10s}")

    # ====================================================================
    # 6. COSMOLOGICAL PARAMETER CROSS-CHECK
    # ====================================================================
    print(f"\n  COSMOLOGICAL PARAMETER CROSS-CHECK")
    print(f"  From JWST paper: Omega values from Fibonacci/Xi arithmetic")
    print(f"  Can we extract analogous ratios from the simulator?")
    print()

    # At NOW tick, compute field energy fractions
    # E ~ entropy/radiation, I ~ information/dark matter, M ~ mass/baryons
    E_now = series["E_mean"][now_idx]
    I_now = series["I_mean"][now_idx]
    M_now = series["M_mean"][now_idx]
    total = abs(E_now) + abs(I_now) + M_now

    frac_E = abs(E_now) / total if total > 0 else 0
    frac_I = abs(I_now) / total if total > 0 else 0
    frac_M = M_now / total if total > 0 else 0

    print(f"  Field fractions at NOW (tick {now_tick}):")
    print(f"    |E| / (|E|+|I|+M) = {frac_E:.4f}  (cf. Omega_Lambda = 0.685)")
    print(f"    |I| / (|E|+|I|+M) = {frac_I:.4f}  (cf. Omega_c     = 0.265)")
    print(f"     M  / (|E|+|I|+M) = {frac_M:.4f}  (cf. Omega_b     = 0.049)")
    print()

    # DFT predictions from JWST paper
    omega_c_dft = 13 * XI**2 / 55    # F7 * Xi^2 / F10 = 0.2648
    omega_l_dft = 8 * XI**2 / 13     # F6 * Xi^2 / F7  = 0.6894
    omega_b_dft = 8 / (XI**2 * 144)  # F6 / (Xi^2 * F12) = 0.0496

    print(f"  DFT Fibonacci predictions:")
    print(f"    Omega_c = F7*Xi^2/F10 = {omega_c_dft:.4f} (Planck: 0.265)")
    print(f"    Omega_L = F6*Xi^2/F7  = {omega_l_dft:.4f} (Planck: 0.685)")
    print(f"    Omega_b = F6/(Xi^2*F12) = {omega_b_dft:.4f} (Planck: 0.049)")
    print()

    # Check if simulator fractions match DFT predictions
    print(f"  Simulator-to-DFT ratio at NOW:")
    if omega_l_dft > 0:
        print(f"    E_frac / Omega_L = {frac_E / omega_l_dft:.3f}")
    if omega_c_dft > 0:
        print(f"    I_frac / Omega_c = {frac_I / omega_c_dft:.3f}")
    if omega_b_dft > 0:
        print(f"    M_frac / Omega_b = {frac_M / omega_b_dft:.3f}")

    # ====================================================================
    # 7. SUMMARY VERDICT
    # ====================================================================
    print(f"\n  {'='*70}")
    print(f"  SUMMARY: SEC Enhancement Factor Validation")
    print(f"  {'='*70}")
    print(f"  Duty cycle correlation (sim vs theory): {corr:+.4f}")
    print(f"  Duty cycle RMSE:                        {rmse:.4f}")
    print(f"  NOW tick:                                {now_tick} / {ticks}")
    print(f"  Lifecycle fraction:                      {now_tick/ticks:.1%}")
    print(f"  Gamma at NOW:                            {series['gamma_raw'][now_idx]:.4f}")
    print(f"  Theory duty at NOW:                      {PHI_DUTY_EQ:.4f}")
    print(f"  G_local at NOW:                          {G_now:.4f} (target {PHI_INV2:.4f})")

    if corr > 0.7:
        print(f"\n  *** STRONG MATCH: Simulator duty cycle tracks SEC prediction ***")
    elif corr > 0.3:
        print(f"\n  *** MODERATE MATCH: Partial SEC signal in simulator ***")
    else:
        print(f"\n  *** WEAK MATCH: Duty cycle evolution differs from SEC ***")

    print(f"\n  DONE")


if __name__ == "__main__":
    main()
