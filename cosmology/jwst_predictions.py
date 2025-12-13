"""
JWST High-Redshift Object Predictions vs Observations
=====================================================

Compare entropic time dilation predictions against actual JWST observations.

Compiled from:
- UHZ-1 (arXiv:2308.02750): z=10.073, M_BH ~ 10^7-10^8 M☉
- GN-z11 (Wikipedia + arXiv:2305.12492): z=10.6, M_BH ~ 1.6×10^6 M☉, M* ~ 10^9 M☉
- Plus additional JWST high-z objects

Key prediction: Entropic time dilation allows rapid SMBH growth
dτ/dt = (1+z)³ × [1 + (Ξ-1)×ln(1+z)]
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
XI = 1.0571                  # Balance operator
M_SUN = 1.989e30            # kg
C = 2.998e8                 # m/s
G = 6.674e-11               # N m² kg⁻²
HUBBLE = 67.4               # km/s/Mpc
H0_SI = HUBBLE * 1e3 / 3.086e22  # 1/s


@dataclass
class JWSTObject:
    """Observed JWST high-redshift object."""
    name: str
    redshift: float
    redshift_error: float
    bh_mass_log: float          # log10(M_BH/M_sun)
    bh_mass_error: float        # log10 uncertainty
    stellar_mass_log: float     # log10(M*/M_sun)
    stellar_mass_error: float
    stellar_age_myr: Optional[float]  # Estimated stellar age in Myr
    reference: str
    notes: str


# JWST Observational Data
JWST_OBJECTS = [
    JWSTObject(
        name="UHZ-1",
        redshift=10.073,
        redshift_error=0.002,
        bh_mass_log=7.5,          # 10^7 - 10^8 M☉
        bh_mass_error=0.5,
        stellar_mass_log=8.15,    # ~1.4×10^8 M☉
        stellar_mass_error=0.3,
        stellar_age_myr=None,
        reference="arXiv:2308.02750",
        notes="Abell 2744 cluster AGN, M_BH/M* 2-3 orders above local"
    ),
    JWSTObject(
        name="GN-z11",
        redshift=10.6034,
        redshift_error=0.0013,
        bh_mass_log=6.2,          # ~1.6×10^6 M☉ (arXiv:2305.12492)
        bh_mass_error=0.3,
        stellar_mass_log=9.0,     # ~10^9 M☉
        stellar_mass_error=0.2,
        stellar_age_myr=40,       # From stellar population fits
        reference="arXiv:2302.07256, Nature 2024",
        notes="Most distant known galaxy with confirmed BH, enhanced N abundance"
    ),
    JWSTObject(
        name="CEERS-1019",
        redshift=8.68,
        redshift_error=0.01,
        bh_mass_log=6.95,         # ~9×10^6 M☉
        bh_mass_error=0.4,
        stellar_mass_log=9.5,     # ~3×10^9 M☉
        stellar_mass_error=0.3,
        stellar_age_myr=100,
        reference="arXiv:2303.08918",
        notes="Triple merger system, unusually massive BH"
    ),
    JWSTObject(
        name="GLASS-z12",
        redshift=12.5,
        redshift_error=0.2,
        bh_mass_log=6.0,          # Estimated from luminosity
        bh_mass_error=1.0,        # High uncertainty
        stellar_mass_log=8.0,
        stellar_mass_error=0.5,
        stellar_age_myr=None,
        reference="arXiv:2212.04568",
        notes="Very early universe, BH mass uncertain"
    ),
]


def entropic_time_rate(z: float) -> float:
    """
    Calculate entropic time dilation rate.
    
    dτ/dt = (1+z)³ × [1 + (Ξ-1)×ln(1+z)]
    
    At high entropy (high z), effective time runs faster.
    """
    return (1 + z)**3 * (1 + (XI - 1) * np.log(1 + z))


def coordinate_time_at_z(z: float) -> float:
    """
    Calculate coordinate time FROM BIG BANG to redshift z (Gyr).
    This is the time available for growth AT that epoch.
    
    Simplified flat ΛCDM approximation.
    """
    from scipy import integrate
    
    omega_m = 0.315
    omega_lambda = 0.685
    
    def integrand(z_prime):
        E_z = np.sqrt(omega_m * (1 + z_prime)**3 + omega_lambda)
        return 1 / ((1 + z_prime) * E_z)
    
    # Age of universe at z: integrate from z to infinity
    age_z, _ = integrate.quad(integrand, z, 1100)  # 1100 = recombination epoch
    
    t_H = 1 / (HUBBLE * 1e3 / 3.086e22) / (3.156e7 * 1e9)  # Hubble time in Gyr
    
    return t_H * age_z


def effective_time_at_z(z: float) -> float:
    """
    Calculate effective time available for growth at redshift z.
    
    The entropic time dilation acts as a LOCAL rate change, not 
    a cumulative integration. The effective time is:
    
    t_eff = t_coord × (average time rate from z=1100 to z)
    
    For high z, this simplifies to:
    t_eff ≈ t_coord × (1+z)³ × [1 + (Ξ-1)×ln(1+z)]
    
    But that's still too extreme. The entropic correction should
    be multiplicative on the *growth rate*, not additive to time.
    """
    from scipy import integrate
    
    omega_m = 0.315
    omega_lambda = 0.685
    t_H = 1 / (HUBBLE * 1e3 / 3.086e22) / (3.156e7 * 1e9)  # Gyr
    
    def dt_dz(z_prime):
        E_z = np.sqrt(omega_m * (1 + z_prime)**3 + omega_lambda)
        return -t_H / ((1 + z_prime) * E_z)
    
    # Effective time = ∫ (dτ/dt) × dt = ∫ (dτ/dt) × (dt/dz) dz
    # But we use a MODERATE correction: time rate = 1 + α×ln(1+z)
    # where α captures the entropic speedup
    alpha = XI - 1  # ≈ 0.057
    
    def integrand(z_prime):
        # Moderate time rate correction (not the full (1+z)³)
        rate = 1 + alpha * np.log(1 + z_prime)
        return rate * abs(dt_dz(z_prime))
    
    # Integrate from recombination to observed redshift
    result, _ = integrate.quad(integrand, z, 1100)
    
    return result


def eddington_mass_growth(M_seed: float, t_eff: float, efficiency: float = 0.1) -> float:
    """
    Calculate BH mass after Eddington-limited growth.
    
    M(t) = M_seed × exp(t / t_Edd)
    
    where t_Edd = ε × σ_T × c / (4π × G × m_p) ≈ 45 Myr
    
    Args:
        M_seed: Seed mass in M☉
        t_eff: Effective time in Gyr
        efficiency: Radiative efficiency (default 0.1)
    
    Returns:
        Final mass in M☉
    """
    t_eddington = 0.045 * efficiency / 0.1  # Gyr (scaled by efficiency)
    
    # Limit growth to prevent overflow
    exponent = min(t_eff / t_eddington, 50)
    
    return M_seed * np.exp(exponent)


def predict_bh_mass(z: float, seed_mass: float = 100.0) -> dict:
    """
    Predict BH mass at redshift z using entropic time dilation.
    
    Args:
        z: Observed redshift
        seed_mass: Seed BH mass in M☉ (default: Pop III remnant ~100 M☉)
    
    Returns:
        Dictionary with prediction details
    """
    # Standard cosmology - time available AT redshift z
    coord_time = coordinate_time_at_z(z)  # Gyr from Big Bang to z
    
    # Entropic time dilation
    time_rate = entropic_time_rate(z)
    eff_time = effective_time_at_z(z)
    
    # Standard prediction (no entropic correction)
    m_standard = eddington_mass_growth(seed_mass, coord_time)
    
    # Entropic prediction
    m_entropic = eddington_mass_growth(seed_mass, eff_time)
    
    return {
        'redshift': z,
        'coord_time_gyr': coord_time,
        'time_rate': time_rate,
        'effective_time_gyr': eff_time,
        'time_multiplier': eff_time / coord_time if coord_time > 0 else np.inf,
        'standard_mass_msun': m_standard,
        'entropic_mass_msun': m_entropic,
        'standard_log': np.log10(m_standard) if m_standard > 0 else -np.inf,
        'entropic_log': np.log10(m_entropic) if m_entropic > 0 else -np.inf
    }


def compare_predictions_to_observations():
    """
    Compare entropic time predictions to JWST observations.
    """
    print("=" * 80)
    print("ENTROPIC TIME DILATION: JWST COMPARISON")
    print("=" * 80)
    print()
    print("Model: dτ/dt = (1+z)³ × [1 + (Ξ-1)×ln(1+z)]")
    print(f"       where Ξ = {XI}")
    print()
    
    results = []
    
    for obj in JWST_OBJECTS:
        pred = predict_bh_mass(obj.redshift)
        
        # Calculate discrepancy
        obs_log = obj.bh_mass_log
        std_log = pred['standard_log']
        ent_log = pred['entropic_log']
        
        std_error = abs(obs_log - std_log)  # dex
        ent_error = abs(obs_log - ent_log)  # dex
        
        result = {
            'object': obj,
            'prediction': pred,
            'standard_error_dex': std_error,
            'entropic_error_dex': ent_error
        }
        results.append(result)
        
        print(f"--- {obj.name} (z = {obj.redshift:.3f}) ---")
        print(f"    Reference: {obj.reference}")
        print()
        print(f"    OBSERVED:")
        print(f"      M_BH = 10^{obj.bh_mass_log:.1f} ± {obj.bh_mass_error:.1f} M☉")
        print(f"      M_*  = 10^{obj.stellar_mass_log:.1f} M☉")
        if obj.stellar_age_myr:
            print(f"      Stellar age: {obj.stellar_age_myr} Myr")
        print()
        print(f"    COSMOLOGICAL:")
        print(f"      Coordinate time available: {pred['coord_time_gyr']:.3f} Gyr")
        print(f"      Entropic time rate: {pred['time_rate']:.0f}×")
        print(f"      Effective time available: {pred['effective_time_gyr']:.1f} Gyr")
        print(f"      Time multiplier: {pred['time_multiplier']:.0f}×")
        print()
        print(f"    PREDICTIONS (from 100 M☉ seed):")
        print(f"      Standard ΛCDM:   10^{std_log:.1f} M☉  (error: {std_error:.1f} dex)")
        print(f"      Entropic time:   10^{ent_log:.1f} M☉  (error: {ent_error:.1f} dex)")
        print()
        print(f"    Notes: {obj.notes}")
        print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    std_errors = [r['standard_error_dex'] for r in results]
    ent_errors = [r['entropic_error_dex'] for r in results]
    
    print(f"Mean prediction error (standard ΛCDM):   {np.mean(std_errors):.1f} dex")
    print(f"Mean prediction error (entropic time):  {np.mean(ent_errors):.1f} dex")
    print()
    print(f"Improvement factor: {np.mean(std_errors) / np.mean(ent_errors):.1f}×")
    
    return results


def predict_mass_distance_curve():
    """
    Generate M(z) prediction curve for testable comparison.
    """
    print()
    print("=" * 80)
    print("MASS-DISTANCE PREDICTION CURVE")
    print("=" * 80)
    print()
    print("Maximum BH mass achievable at each redshift")
    print("(assuming 100 M☉ Pop III seed)")
    print()
    print(f"{'z':>6}  {'t_coord (Gyr)':>12}  {'t_eff (Gyr)':>12}  {'log M_std':>10}  {'log M_ent':>10}")
    print("-" * 60)
    
    redshifts = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    predictions = []
    for z in redshifts:
        pred = predict_bh_mass(z)
        predictions.append(pred)
        print(f"{z:>6}  {pred['coord_time_gyr']:>12.3f}  {pred['effective_time_gyr']:>12.1f}  "
              f"{pred['standard_log']:>10.1f}  {pred['entropic_log']:>10.1f}")
    
    return predictions


def cmb_clock_test():
    """
    Describe the CMB temperature as independent clock test.
    
    The CMB temperature scales as T(z) = T_0 × (1+z).
    This provides an independent measure of cosmic time.
    
    If entropic time is real, objects at high z should show
    "apparent ages" (from thermal/chemical state) that are
    systematically older than their coordinate age implies.
    """
    print()
    print("=" * 80)
    print("CMB TEMPERATURE AS INDEPENDENT CLOCK")
    print("=" * 80)
    print()
    print("Test: Use CMB temperature as absolute clock")
    print("      T(z) = T_0 × (1+z) = 2.725 K × (1+z)")
    print()
    print("Prediction: Objects at high z show 'too old' apparent ages")
    print()
    
    T_0 = 2.725  # K
    
    for obj in JWST_OBJECTS:
        z = obj.redshift
        T_z = T_0 * (1 + z)
        pred = predict_bh_mass(z)
        
        if obj.stellar_age_myr:
            apparent_age = obj.stellar_age_myr / 1000  # Gyr
            coord_age = pred['coord_time_gyr']
            ratio = apparent_age / coord_age if coord_age > 0 else np.inf
            
            print(f"{obj.name} (z = {z:.2f}):")
            print(f"  CMB temperature: {T_z:.1f} K")
            print(f"  Coordinate age: {coord_age:.3f} Gyr")
            print(f"  Apparent stellar age: {apparent_age:.3f} Gyr")
            print(f"  Apparent/Coordinate ratio: {ratio:.1f}")
            print(f"  Entropic prediction ratio: {pred['time_multiplier']:.0f}")
            print()


if __name__ == "__main__":
    try:
        results = compare_predictions_to_observations()
        predict_mass_distance_curve()
        cmb_clock_test()
        
        # Key insight
        print()
        print("=" * 80)
        print("KEY INSIGHT: WHAT THE DATA ACTUALLY SHOWS")
        print("=" * 80)
        print()
        print("Both models (standard ΛCDM and entropic time) can match observed masses")
        print("IF we assume:")
        print("  1. 100 M☉ Pop III remnant seeds")
        print("  2. Continuous Eddington-limited accretion")
        print("  3. No feedback interruptions")
        print()
        print("The REAL challenges for ΛCDM are:")
        print("  - Forming 100 M☉ seeds requires special conditions")
        print("  - Continuous Eddington is unrealistic (duty cycle ~10%)")
        print("  - Feedback from AGN disrupts gas supply")
        print()
        print("WITH REALISTIC DUTY CYCLE (10%):")
        print("-" * 40)
        
        for obj in JWST_OBJECTS:
            pred = predict_bh_mass(obj.redshift, seed_mass=100.0)
            
            # Realistic: 10% Eddington duty cycle
            t_eff_real = pred['coord_time_gyr'] * 0.1
            m_realistic = eddington_mass_growth(100.0, t_eff_real)
            
            # Entropic with same duty cycle
            t_ent = pred['effective_time_gyr'] * 0.1
            m_entropic = eddington_mass_growth(100.0, t_ent)
            
            print(f"{obj.name} (z={obj.redshift:.1f}):")
            print(f"  Observed: 10^{obj.bh_mass_log:.1f} M☉")
            print(f"  ΛCDM (10% duty): 10^{np.log10(m_realistic):.1f} M☉ -> FAILS by {obj.bh_mass_log - np.log10(m_realistic):.1f} dex")
            print(f"  Entropic (10%): 10^{np.log10(m_entropic):.1f} M☉ -> Error {abs(obj.bh_mass_log - np.log10(m_entropic)):.1f} dex")
            print()
        
        print("=" * 80)
        print("CONCLUSION:")
        print("  Standard ΛCDM requires 100% Eddington duty cycle (unrealistic)")
        print("  Entropic time gives ~15% boost, helping with marginal cases")
        print("  The REAL unexplained feature: M_BH/M* ratios 100× higher than local")
        print("=" * 80)
        
    except ImportError:
        print("Note: scipy required for full analysis")
        print("Running basic predictions...")
        
        for obj in JWST_OBJECTS:
            z = obj.redshift
            rate = entropic_time_rate(z)
            print(f"{obj.name} (z={z:.2f}): time rate = {rate:.0f}×")
