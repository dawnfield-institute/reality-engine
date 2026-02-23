"""
Diagnostic: understand WHY ξ is locked at 1.0 in the engine.

Key questions:
1. What is the spectral decomposition of f_anti by wavenumber?
2. How much does each SEC term (diffusion, source, collapse, modulation) 
   contribute to dE_anti/dt and dE_sym/dt per step?
3. What happens at different β₀ values (moving off the degenerate peak)?
4. What happens at dramatically higher ρ?

Root cause hypothesis: at β₀=1.0 and P_mean≈1.0, the collapse C(S)=S·exp(-S)
has dC/dS=0 at S=1. This means:
  - The collapse anti-component C_anti ∝ δ³ (CUBIC in anti amplitude)
  - Collapse modulation has nothing to modulate
  - Conservative transfer battles diffusion (κk²≈1.09 > available B)
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import math
from src.engine import RealityEngine
from src.substrate.constants import XI_REFERENCE


def run_beta_sweep():
    """Test different β₀ values to find where ξ responds."""
    print("=" * 70)
    print("  β₀ SWEEP — breaking the S=1/β₀ degeneracy")
    print("=" * 70)
    
    betas = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    
    for beta in betas:
        sec_params = {
            "kappa": 0.1, "gamma": 1.0, "beta_0": beta,
            "sigma_0": 0.1, "dt": 0.01,
            "xi_gain": 2.0, "rho": 5.0, "phi_source": 0.618,
            "alpha_rbf": 5.0, "rbf_decay": 0.995,
            "ki_rbf": 0.5, "integral_clamp": 1.0, "rbf_omega": 0.2,
        }
        config = {
            "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
            "sec": sec_params,
            "init": {"seed": 42, "P_mean": 0.5, "P_noise": 0.01,
                     "A_mean": 0.5, "A_noise": 0.01},
            "pac": {"mode": "enforce"},
        }
        engine = RealityEngine(config)
        
        for i in range(5000):
            diag = engine.step()
            if engine.diagnostics.diverged:
                break
        
        xi = diag.get("xi_L2", float("nan"))
        P_mean = diag.get("P_mean", float("nan"))
        P_std = diag.get("P_std", float("nan"))
        
        # Compute dC/dS at P_mean 
        dCdS = (1 - beta * P_mean) * math.exp(-beta * P_mean)
        peak = 1.0 / beta
        
        status = "DIVERGED" if engine.diagnostics.diverged else ""
        print(f"  β₀={beta:.1f}  peak={peak:.2f}  dC/dS={dCdS:+.3f}  "
              f"ξ_L2={xi:.4f}  P_mean={P_mean:.3f}  P_std={P_std:.2f}  {status}")


def run_extreme_rho():
    """Test with extremely high ρ to see if any ξ shift is possible."""
    print("\n" + "=" * 70)
    print("  EXTREME ρ SWEEP — can ξ be moved at all?")
    print("=" * 70)
    
    configs = [
        ("ρ=1",    {"rho": 1.0, "alpha_rbf": 5.0}),
        ("ρ=10",   {"rho": 10.0, "alpha_rbf": 5.0}),
        ("ρ=50",   {"rho": 50.0, "alpha_rbf": 1.0}),
        ("ρ=100",  {"rho": 100.0, "alpha_rbf": 0.5}),
        ("ρ=500",  {"rho": 500.0, "alpha_rbf": 0.1}),
    ]
    
    for name, overrides in configs:
        sec_params = {
            "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
            "sigma_0": 0.1, "dt": 0.01,
            "xi_gain": 2.0, "phi_source": 0.618,
            "rbf_decay": 0.995, "ki_rbf": 0.5,
            "integral_clamp": 1.0, "rbf_omega": 0.2,
        }
        sec_params.update(overrides)
        config = {
            "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
            "sec": sec_params,
            "init": {"seed": 42, "P_mean": 0.5, "P_noise": 0.01,
                     "A_mean": 0.5, "A_noise": 0.01},
            "pac": {"mode": "enforce"},
        }
        engine = RealityEngine(config)
        
        diverged = False
        for i in range(5000):
            diag = engine.step()
            if engine.diagnostics.diverged:
                diverged = True
                break
        
        xi = diag.get("xi_L2", float("nan"))
        P_mean = diag.get("P_mean", float("nan"))
        P_std = diag.get("P_std", float("nan"))
        M_rbf = diag.get("M_rbf", 0)
        
        status = f"DIVERGED step {diag['t']}" if diverged else ""
        print(f"  {name:8s}  ξ_L2={xi:.6f}  P_mean={P_mean:.3f}  "
              f"P_std={P_std:.2f}  M_rbf={M_rbf:.3f}  {status}")


def run_per_step_decomposition():
    """Decompose one engine step into contributions to dE_anti and dE_sym."""
    print("\n" + "=" * 70)
    print("  PER-STEP DECOMPOSITION at steady state")
    print("=" * 70)
    
    sec_params = {
        "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
        "sigma_0": 0.1, "dt": 0.01,
        "xi_gain": 2.0, "rho": 10.0, "phi_source": 0.618,
        "alpha_rbf": 5.0, "rbf_decay": 0.995,
        "ki_rbf": 1.0, "integral_clamp": 1.0, "rbf_omega": 0.2,
    }
    config = {
        "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
        "sec": sec_params,
        "init": {"seed": 42, "P_mean": 0.5, "P_noise": 0.01,
                 "A_mean": 0.5, "A_noise": 0.01},
        "pac": {"mode": "enforce"},
    }
    engine = RealityEngine(config)
    
    # Run to near-steady state
    for i in range(5000):
        diag = engine.step()
    
    xi_L2 = diag.get("xi_L2", 0)
    print(f"\n  At step 5000: ξ_L2 = {xi_L2:.6f}")
    
    # Now decompose one SEC step manually
    S = engine.state.P.clone()
    sec = engine.sec
    conf = engine.confluence
    
    # Current spectral state
    f_sym = (S + conf(S)) / 2.0
    f_anti = (S - conf(S)) / 2.0
    E_sym = f_sym.pow(2).sum().item()
    E_anti = f_anti.pow(2).sum().item()
    xi = E_anti / max(E_sym, 1e-14)
    print(f"  E_sym={E_sym:.2f}  E_anti={E_anti:.2f}  ξ={xi:.6f}")
    print(f"  P_mean={S.mean().item():.4f}  P_std={S.std().item():.4f}")
    
    # Diffusion contribution
    diffusion = sec.kappa * sec.manifold.laplacian(S)
    diff_anti = (diffusion - conf(diffusion)) / 2.0
    diff_sym = (diffusion + conf(diffusion)) / 2.0
    dE_anti_diff = 2 * (f_anti * diff_anti).sum().item()
    dE_sym_diff = 2 * (f_sym * diff_sym).sum().item()
    
    # Source contribution 
    xi_dev = (xi_L2 - XI_REFERENCE) / max(XI_REFERENCE, 1e-14)
    gain_p = -math.tanh(sec.xi_gain * xi_dev)
    gain_i = -sec.ki_rbf * sec._xi_integral
    gain = gain_p + gain_i
    Phi = 0.5 + 0.5 * sec._fibonacci_harmonic()
    B = sec.rho * gain * Phi / (1.0 + sec.alpha_rbf * abs(sec._M_rbf))
    mod = math.tanh(B)
    effective_phi = sec.phi_source + B
    theta_density = engine._theta / max(S.numel(), 1)
    source = sec.sigma_0 * (1.0 + effective_phi * sec._anti_mode) + theta_density
    src_anti = (source - conf(source)) / 2.0
    src_sym = (source + conf(source)) / 2.0
    dE_anti_src = 2 * (f_anti * src_anti).sum().item()
    dE_sym_src = 2 * (f_sym * src_sym).sum().item()
    
    # Collapse contribution (raw)
    collapse_raw = S * torch.exp(-sec.beta_0 * S)
    C_conf = conf(collapse_raw)
    C_anti_col = (collapse_raw - C_conf) / 2.0
    C_sym_col = (collapse_raw + C_conf) / 2.0
    
    # Unmodulated collapse contribution
    dE_anti_col_raw = -2 * sec.gamma * (f_anti * C_anti_col).sum().item()
    dE_sym_col_raw = -2 * sec.gamma * (f_sym * C_sym_col).sum().item()
    
    # Modulated: collapse = collapse_raw - mod * C_anti_col
    # Extra term: +mod * C_anti_col (subtracted from collapse → added to dS)
    dE_anti_mod = 2 * sec.gamma * mod * (f_anti * C_anti_col).sum().item()
    dE_sym_mod = 2 * sec.gamma * mod * (f_sym * C_anti_col).sum().item()
    
    print(f"\n  B = {B:.4f}, tanh(B) = {mod:.4f}")
    print(f"  gain_p={gain_p:.4f}, gain_i={gain_i:.4f}, Phi={Phi:.4f}, M_rbf={sec._M_rbf:.4f}")
    print(f"  ∫ξ_err = {sec._xi_integral:.4f}")
    
    print(f"\n  --- dE per time unit ---")
    print(f"  {'Term':25s} {'dE_anti':>12s} {'dE_sym':>12s} {'dξ effect':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    
    def xi_effect(dEa, dEs):
        """dξ/dt = (dEa*Es - Ea*dEs)/Es²"""
        return (dEa * E_sym - E_anti * dEs) / (E_sym * E_sym) if E_sym > 0 else 0
    
    items = [
        ("Diffusion",      dE_anti_diff, dE_sym_diff),
        ("Source",         dE_anti_src,  dE_sym_src),
        ("Collapse (raw)", dE_anti_col_raw, dE_sym_col_raw),
        ("Collapse mod",   dE_anti_mod, dE_sym_mod),
    ]
    
    total_anti = 0
    total_sym = 0
    for name, dEa, dEs in items:
        total_anti += dEa
        total_sym += dEs
        dxi = xi_effect(dEa, dEs)
        print(f"  {name:25s} {dEa:+12.4f} {dEs:+12.4f} {dxi:+12.6f}")
    
    dxi_total = xi_effect(total_anti, total_sym)
    print(f"  {'TOTAL':25s} {total_anti:+12.4f} {total_sym:+12.4f} {dxi_total:+12.6f}")
    
    # Collapse anti-component analysis
    print(f"\n  --- Collapse anti-component ---")
    print(f"  ||C_anti_col|| = {C_anti_col.pow(2).sum().sqrt().item():.6f}")
    print(f"  ||C_sym_col|| = {C_sym_col.pow(2).sum().sqrt().item():.6f}")
    print(f"  ||C_raw||     = {collapse_raw.pow(2).sum().sqrt().item():.6f}")
    print(f"  C_anti/C_raw ratio = {C_anti_col.pow(2).sum().item() / collapse_raw.pow(2).sum().clamp(min=1e-14).item():.6f}")
    
    # Wavenumber analysis of f_anti
    print(f"\n  --- f_anti spectral content ---")
    # Compute 2D FFT of f_anti to see k-distribution
    fft = torch.fft.fft2(f_anti)
    power = fft.abs().pow(2)
    nu, nv = f_anti.shape
    # k² = ku² + kv² for each mode
    ku = torch.fft.fftfreq(nu).unsqueeze(1) * nu  # integer wavenumbers
    kv = torch.fft.fftfreq(nv).unsqueeze(0) * nv
    k_sq = ku**2 + (kv * math.pi)**2  # scale: kv in physical units (πm for v∈[0,1])
    
    # Bin by k²
    bins = [(0, 2, "k²<2"), (2, 15, "2≤k²<15"), (15, 100, "15≤k²<100"), (100, 10000, "k²≥100")]
    total_power = power.sum().item()
    for lo, hi, label in bins:
        mask = (k_sq >= lo) & (k_sq < hi)
        frac = (power * mask).sum().item() / max(total_power, 1e-14)
        print(f"  {label:15s}: {frac*100:6.2f}% of E_anti")


def run_lower_pmean():
    """Test with lower P_mean where SEC collapse is more responsive."""
    print("\n" + "=" * 70)
    print("  LOWER P_mean — move to responsive collapse regime")
    print("=" * 70)
    
    p_means = [0.1, 0.2, 0.3, 0.5]
    
    for pm in p_means:
        sec_params = {
            "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
            "sigma_0": 0.1, "dt": 0.01,
            "xi_gain": 2.0, "rho": 5.0, "phi_source": 0.618,
            "alpha_rbf": 5.0, "rbf_decay": 0.995,
            "ki_rbf": 0.5, "integral_clamp": 1.0, "rbf_omega": 0.2,
        }
        config = {
            "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
            "sec": sec_params,
            "init": {"seed": 42, "P_mean": pm, "P_noise": 0.01,
                     "A_mean": pm, "A_noise": 0.01},
            "pac": {"mode": "enforce"},
        }
        engine = RealityEngine(config)
        
        for i in range(5000):
            diag = engine.step()
            if engine.diagnostics.diverged:
                break
        
        xi = diag.get("xi_L2", float("nan"))
        P_mean = diag.get("P_mean", float("nan"))
        P_std = diag.get("P_std", float("nan"))
        
        dCdS = (1 - P_mean) * math.exp(-P_mean)
        status = "DIVERGED" if engine.diagnostics.diverged else ""
        print(f"  init_Pm={pm:.1f}  final_Pm={P_mean:.3f}  dC/dS={dCdS:+.3f}  "
              f"ξ_L2={xi:.4f}  P_std={P_std:.2f}  {status}")


if __name__ == "__main__":
    run_per_step_decomposition()
    run_beta_sweep()
    run_extreme_rho()
    run_lower_pmean()
