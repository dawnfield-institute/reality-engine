"""Spike 11: Duty Cycle Probe — Alternative Mappings and Proxies

Spike 10 found the gravity running curve matches JWST predictions (~1.5x at high z),
but the duty cycle correlation was inverted (-0.61). Two hypotheses:

  H1: The tick-to-cascade-depth mapping k = k_eq * (tick/now_tick) is wrong.
      Alternative mappings: logarithmic, entropy-based, M-based.

  H2: gamma_local isn't the right duty cycle proxy.
      Alternatives: xi_s, actualization fraction, mass generation duty,
      E/I ratio, SEC energy functional.

This spike tests both systematically.
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

XI = GAMMA_EM + math.log(PHI)
K_EQ = 2
PHI_DUTY_EQ = PHI / (PHI + 1)


def sec_duty_theory(k):
    """SEC duty cycle at cascade depth k."""
    r = PHI ** (1 + (K_EQ - k) / 2.0)
    return r / (r + 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticks = 20000
    sample_every = 50

    print("=" * 90)
    print("  SPIKE 11: Duty Cycle Probes -- Alternative Mappings and Proxies")
    print(f"  Device: {device} | Grid: 128x64 | {ticks} ticks | sample every {sample_every}")
    print("=" * 90)

    torch.manual_seed(42)
    config = default_config(device=device)
    pipeline = default_pipeline()
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    # Collect comprehensive time series
    series = {}
    for name in TARGETS:
        series[name] = []
    extras = [
        "gamma_raw", "M_mean", "M_max", "E_mean", "I_mean", "E_abs_mean",
        "I_abs_mean", "diseq_mean", "xi_s", "entropy_cumul",
        "E_var", "I_var", "M_var",
        # Duty cycle proxy candidates
        "act_fraction",    # fraction of cells where actualization is active
        "mass_gen_frac",   # fraction of cells gaining mass
        "EI_ratio",        # |E| / (|E| + |I|)
        "coherence",       # I^2 / (E^2 + I^2) = gamma_local (same as gamma)
        "sec_energy",      # SEC functional if available
    ]
    for e in extras:
        series[e] = []
    tick_stamps = []

    t0 = time.time()
    for tick in range(1, ticks + 1):
        engine.tick()
        if tick % sample_every == 0:
            m = engine.state.metrics
            E = engine.state.E
            I = engine.state.I
            M = engine.state.M

            for name, (key, _) in TARGETS.items():
                series[name].append(m.get(key, 0))

            series["gamma_raw"].append(m.get("gamma_local_mean", 0))
            series["M_mean"].append(M.mean().item())
            series["M_max"].append(M.max().item())
            series["E_mean"].append(E.mean().item())
            series["I_mean"].append(I.mean().item())
            series["E_abs_mean"].append(E.abs().mean().item())
            series["I_abs_mean"].append(I.abs().mean().item())
            series["diseq_mean"].append(m.get("disequilibrium_mean", 0))
            series["xi_s"].append(m.get("xi_s_mean", 0))
            series["entropy_cumul"].append(m.get("entropy_reduction_cumulative", 0))

            # Field variances (spatial structure)
            series["E_var"].append(E.var().item())
            series["I_var"].append(I.var().item())
            series["M_var"].append(M.var().item())

            # --- Duty cycle proxy candidates ---

            # 1. Actualization fraction: cells where |E-I| > threshold
            diseq = (E - I).abs()
            act_thresh = config.actualization_threshold
            act_frac = (diseq > act_thresh).float().mean().item()
            series["act_fraction"].append(act_frac)

            # 2. Mass generation fraction: cells where M is increasing
            #    Proxy: cells where gamma_local * diseq^2 > eta * M
            E2 = E.pow(2)
            I2 = I.pow(2)
            gamma_field = I2 / (E2 + I2 + 1e-10)
            diseq2 = (E - I).pow(2)
            mass_gen = gamma_field * diseq2
            mass_decay = config.deactualization_rate * M * (1.0 - gamma_field)
            gen_frac = (mass_gen > mass_decay).float().mean().item()
            series["mass_gen_frac"].append(gen_frac)

            # 3. E/I energy ratio
            E_abs = E.abs().mean().item()
            I_abs = I.abs().mean().item()
            ei_ratio = E_abs / (E_abs + I_abs + 1e-10)
            series["EI_ratio"].append(ei_ratio)

            # 4. Coherence (= gamma_local, redundant but explicit)
            series["coherence"].append(gamma_field.mean().item())

            # 5. SEC energy functional
            series["sec_energy"].append(m.get("sec_energy", 0))

            tick_stamps.append(tick)

            if tick % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  Tick {tick:>6d} ({elapsed:>5.0f}s): "
                      f"gamma={series['gamma_raw'][-1]:.4f}  "
                      f"act_frac={act_frac:.3f}  "
                      f"gen_frac={gen_frac:.3f}  "
                      f"EI={ei_ratio:.4f}")

    elapsed = time.time() - t0
    n_samples = len(tick_stamps)
    print(f"\n  Collection complete: {n_samples} samples in {elapsed:.0f}s")

    # Convert to numpy
    for k in series:
        series[k] = np.array(series[k])
    tick_stamps = np.array(tick_stamps)

    # Find NOW tick
    errors = {}
    for name in TARGETS:
        target = dict(TARGETS)[name][1] if isinstance(dict(TARGETS)[name], tuple) else 0
    target_vals = {name: target for name, (_, target) in TARGETS.items()}
    for name in TARGETS:
        errors[name] = np.abs(series[name] - target_vals[name]) / np.abs(target_vals[name]) * 100
    avg_errors = np.mean([errors[name] for name in TARGETS], axis=0)
    now_idx = np.argmin(avg_errors)
    now_tick = tick_stamps[now_idx]
    print(f"  NOW tick: {now_tick} (avg error {avg_errors[now_idx]:.2f}%)")

    # ====================================================================
    # PART 1: ALTERNATIVE TICK-TO-K MAPPINGS
    # ====================================================================
    print(f"\n{'='*90}")
    print(f"  PART 1: ALTERNATIVE TICK-TO-CASCADE-DEPTH MAPPINGS")
    print(f"{'='*90}")
    print(f"  Testing which mapping makes gamma_sim track sec_duty_theory(k)")

    # Define mapping functions: tick -> k (cascade depth)
    M_arr = series["M_mean"]
    M_now = M_arr[now_idx]
    M_max_global = M_arr.max()
    entropy_arr = series["entropy_cumul"]
    ent_now = entropy_arr[now_idx]
    ent_min = entropy_arr.min()
    ent_max = entropy_arr.max()

    mappings = {
        "linear: k = k_eq * t/t_now": lambda idx: K_EQ * tick_stamps[idx] / now_tick,

        "log: k = k_eq * log(t)/log(t_now)": lambda idx: (
            K_EQ * np.log(tick_stamps[idx]) / np.log(now_tick)
            if tick_stamps[idx] > 0 else 0
        ),

        "sqrt: k = k_eq * sqrt(t/t_now)": lambda idx: K_EQ * np.sqrt(tick_stamps[idx] / now_tick),

        "M-based: k = k_eq * M/M_now": lambda idx: K_EQ * M_arr[idx] / M_now if M_now > 0 else 0,

        "entropy: k = k_eq * S/S_now": lambda idx: (
            K_EQ * entropy_arr[idx] / ent_now if ent_now != 0 else 0
        ),

        "inverted: k = k_eq * (2 - t/t_now)": lambda idx: K_EQ * (2.0 - tick_stamps[idx] / now_tick),

        "gamma-anchored: k from gamma": lambda idx: (
            # Invert the duty cycle formula: if duty = R/(R+1), R = phi^(1+(k_eq-k)/2)
            # Then k = k_eq + 2 - 2*log(duty/(1-duty))/log(phi)
            # This is the "what k would theory need to match gamma_sim?"
            K_EQ + 2 - 2 * np.log(series["gamma_raw"][idx] / (1 - series["gamma_raw"][idx] + 1e-10)) / np.log(PHI)
            if series["gamma_raw"][idx] > 0.01 and series["gamma_raw"][idx] < 0.99 else 0
        ),
    }

    print(f"\n  {'Mapping':<40s} {'Corr':>7s} {'RMSE':>7s} {'Verdict':>10s}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*10}")

    best_corr = -2
    best_mapping = ""

    for label, k_func in mappings.items():
        sim_vals = []
        theory_vals = []
        for idx in range(n_samples):
            k = k_func(idx)
            if not np.isfinite(k):
                continue
            duty_th = sec_duty_theory(k)
            gamma_sim = series["gamma_raw"][idx]
            sim_vals.append(gamma_sim)
            theory_vals.append(duty_th)

        sim_vals = np.array(sim_vals)
        theory_vals = np.array(theory_vals)

        if len(sim_vals) > 2:
            corr = np.corrcoef(sim_vals, theory_vals)[0, 1]
            rmse = np.sqrt(np.mean((sim_vals - theory_vals) ** 2))
        else:
            corr = 0
            rmse = 999

        verdict = ""
        if corr > 0.9:
            verdict = "STRONG"
        elif corr > 0.5:
            verdict = "moderate"
        elif corr > 0:
            verdict = "weak"
        else:
            verdict = "inverted"

        print(f"  {label:<40s} {corr:>+7.3f} {rmse:>7.4f} {verdict:>10s}")

        if corr > best_corr:
            best_corr = corr
            best_mapping = label

    print(f"\n  Best mapping: {best_mapping} (corr={best_corr:+.3f})")

    # ====================================================================
    # PART 2: ALTERNATIVE DUTY CYCLE PROXIES
    # ====================================================================
    print(f"\n{'='*90}")
    print(f"  PART 2: ALTERNATIVE DUTY CYCLE PROXIES")
    print(f"{'='*90}")
    print(f"  Testing which simulator observable best matches SEC duty cycle")
    print(f"  Using linear mapping k = k_eq * t/t_now (and best mapping)")

    # Compute theory duty cycle for linear mapping
    theory_linear = np.array([sec_duty_theory(K_EQ * tick_stamps[i] / now_tick)
                              for i in range(n_samples)])

    proxies = {
        "gamma_local (I^2/(E^2+I^2))": series["gamma_raw"],
        "1 - gamma_local": 1.0 - series["gamma_raw"],
        "act_fraction (|E-I| > thresh)": series["act_fraction"],
        "1 - act_fraction": 1.0 - series["act_fraction"],
        "mass_gen_fraction": series["mass_gen_frac"],
        "1 - mass_gen_fraction": 1.0 - series["mass_gen_frac"],
        "EI_ratio (|E|/(|E|+|I|))": series["EI_ratio"],
        "1 - EI_ratio": 1.0 - series["EI_ratio"],
        "xi_s (I^2/E^2)": series["xi_s"],
        "xi_s / (xi_s + 1)": series["xi_s"] / (series["xi_s"] + 1.0),
        "sqrt(gamma)": np.sqrt(np.clip(series["gamma_raw"], 0, 1)),
        "gamma^2": series["gamma_raw"] ** 2,
        "gamma^phi": np.power(np.clip(series["gamma_raw"], 1e-10, 1), PHI),
        "gamma^(1/phi)": np.power(np.clip(series["gamma_raw"], 1e-10, 1), 1.0/PHI),
        "M_mean / M_max_global": series["M_mean"] / (M_max_global + 1e-10),
        "G_local": series["G_local"],
        "1 - G_local": 1.0 - series["G_local"],
    }

    print(f"\n  {'Proxy':<35s} {'Corr':>7s} {'RMSE':>7s} {'@NOW':>7s} "
          f"{'theory':>7s} {'ratio':>7s}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    best_proxy_corr = -2
    best_proxy_name = ""

    for label, proxy_arr in proxies.items():
        if len(proxy_arr) != n_samples:
            continue
        valid = np.isfinite(proxy_arr)
        if valid.sum() < 10:
            continue

        corr = np.corrcoef(proxy_arr[valid], theory_linear[valid])[0, 1]
        rmse = np.sqrt(np.mean((proxy_arr[valid] - theory_linear[valid]) ** 2))
        val_now = proxy_arr[now_idx]
        th_now = theory_linear[now_idx]
        ratio = val_now / th_now if th_now > 0 else 0

        print(f"  {label:<35s} {corr:>+7.3f} {rmse:>7.4f} {val_now:>7.4f} "
              f"{th_now:>7.4f} {ratio:>7.3f}")

        if corr > best_proxy_corr:
            best_proxy_corr = corr
            best_proxy_name = label

    print(f"\n  Best proxy: {best_proxy_name} (corr={best_proxy_corr:+.3f})")

    # ====================================================================
    # PART 3: BEST COMBO — scan all proxy x mapping combinations
    # ====================================================================
    print(f"\n{'='*90}")
    print(f"  PART 3: BEST PROXY x MAPPING COMBINATION")
    print(f"{'='*90}")

    results = []

    for m_label, k_func in mappings.items():
        # Compute theory for this mapping
        theory_mapped = []
        valid_indices = []
        for idx in range(n_samples):
            k = k_func(idx)
            if np.isfinite(k):
                theory_mapped.append(sec_duty_theory(k))
                valid_indices.append(idx)
        theory_mapped = np.array(theory_mapped)
        valid_indices = np.array(valid_indices)

        if len(theory_mapped) < 10:
            continue

        for p_label, proxy_arr in proxies.items():
            proxy_vals = proxy_arr[valid_indices]
            valid = np.isfinite(proxy_vals)
            if valid.sum() < 10:
                continue
            corr = np.corrcoef(proxy_vals[valid], theory_mapped[valid])[0, 1]
            if np.isfinite(corr):
                results.append((corr, m_label, p_label))

    results.sort(key=lambda x: -x[0])

    print(f"\n  TOP 10 COMBINATIONS:")
    print(f"  {'Corr':>7s}  {'Mapping':<40s}  {'Proxy':<35s}")
    print(f"  {'-'*7}  {'-'*40}  {'-'*35}")
    for corr, ml, pl in results[:10]:
        print(f"  {corr:>+7.3f}  {ml:<40s}  {pl:<35s}")

    print(f"\n  BOTTOM 5 (most inverted):")
    for corr, ml, pl in results[-5:]:
        print(f"  {corr:>+7.3f}  {ml:<40s}  {pl:<35s}")

    # ====================================================================
    # PART 4: DETAILED VIEW OF BEST COMBINATION
    # ====================================================================
    if results:
        best_corr_combo, best_m, best_p = results[0]
        print(f"\n{'='*90}")
        print(f"  PART 4: DETAILED VIEW OF BEST COMBINATION")
        print(f"  Mapping: {best_m}")
        print(f"  Proxy:   {best_p}")
        print(f"  Correlation: {best_corr_combo:+.4f}")
        print(f"{'='*90}")

        k_func = mappings[best_m]
        proxy_arr = proxies[best_p]

        print(f"\n  {'Tick':>6s} {'k':>7s} {'proxy':>8s} {'theory':>8s} "
              f"{'diff':>8s} {'gamma':>8s} {'G_local':>8s}")
        print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        checkpoints = [100, 250, 500, 1000, 2000, 3000, 4000, 5000,
                       6000, 7000, 8000, int(now_tick), 10000, 12000, 15000, 20000]
        for cp in checkpoints:
            idx = cp // sample_every - 1
            if idx < 0 or idx >= n_samples:
                continue
            k = k_func(idx)
            if not np.isfinite(k):
                continue
            duty_th = sec_duty_theory(k)
            pval = proxy_arr[idx]
            diff = pval - duty_th
            marker = " <-- NOW" if cp == int(now_tick) else ""

            print(f"  {cp:>6d} {k:>7.3f} {pval:>8.4f} {duty_th:>8.4f} "
                  f"{diff:>+8.4f} {series['gamma_raw'][idx]:>8.4f} "
                  f"{series['G_local'][idx]:>8.4f}{marker}")

    # ====================================================================
    # PART 5: INVERSE QUESTION — what k(t) does the simulator imply?
    # ====================================================================
    print(f"\n{'='*90}")
    print(f"  PART 5: IMPLIED CASCADE DEPTH k(t) FROM SIMULATOR")
    print(f"  If gamma_local IS the duty cycle, what k does that imply?")
    print(f"{'='*90}")

    print(f"\n  {'Tick':>6s} {'gamma':>8s} {'k_implied':>10s} {'k_linear':>9s} "
          f"{'k_ratio':>8s} {'Interpretation':>20s}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*9} {'-'*8} {'-'*20}")

    for cp in [100, 250, 500, 1000, 2000, 3000, 5000, 8000, int(now_tick),
               10000, 12000, 15000, 20000]:
        idx = cp // sample_every - 1
        if idx < 0 or idx >= n_samples:
            continue
        g = series["gamma_raw"][idx]
        if g > 0.01 and g < 0.99:
            # Invert: duty = R/(R+1), R = duty/(1-duty)
            # R = phi^(1 + (k_eq - k)/2)
            # log(R) = (1 + (k_eq-k)/2) * log(phi)
            # k = k_eq + 2 - 2*log(R)/log(phi)
            R = g / (1.0 - g)
            k_implied = K_EQ + 2 - 2 * np.log(R) / np.log(PHI)
        else:
            k_implied = float('nan')

        k_linear = K_EQ * cp / now_tick

        if np.isfinite(k_implied) and k_linear > 0:
            k_ratio = k_implied / k_linear
        else:
            k_ratio = float('nan')

        # Interpretation
        if not np.isfinite(k_implied):
            interp = "out of range"
        elif k_implied < 0:
            interp = "pre-cascade"
        elif k_implied < 1:
            interp = "early cascade"
        elif abs(k_implied - K_EQ) < 0.3:
            interp = "near equilibrium"
        elif k_implied > K_EQ:
            interp = "post-equilibrium"
        else:
            interp = "mid-cascade"

        marker = " <-- NOW" if cp == int(now_tick) else ""
        print(f"  {cp:>6d} {g:>8.4f} {k_implied:>10.3f} {k_linear:>9.3f} "
              f"{k_ratio:>8.3f} {interp:>20s}{marker}")

    print(f"\n  If k_implied tracks k_linear, the linear mapping is correct.")
    print(f"  If k_implied follows a different curve, that curve IS the mapping.")

    print(f"\n  DONE")


if __name__ == "__main__":
    main()
