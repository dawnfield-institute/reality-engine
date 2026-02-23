"""
Parameter sweep: find ρ that drives ξ_L2 → 1.057 in the full engine.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from src.engine import RealityEngine

for rho in [0.5, 1.0, 2.0, 5.0, 10.0]:
    config = {
        "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
        "sec": {
            "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
            "sigma_0": 0.1, "dt": 0.01, "xi_gain": 2.0,
            "rho": rho, "phi_source": 0.618,
        },
        "init": {"seed": 42, "P_mean": 0.5, "P_noise": 0.01,
                 "A_mean": 0.5, "A_noise": 0.01},
        "pac": {"mode": "enforce"},
    }
    engine = RealityEngine(config)
    print(f"\n{'='*60}")
    print(f"  rho = {rho}")
    print(f"{'='*60}")
    for i in range(5000):
        diag = engine.step()
        if i < 3 or i % 1000 == 0:
            print(f"  t={diag['t']:5d} | xi_L2={diag.get('xi_L2',0):.6f} | "
                  f"xi_spec={diag['xi_spectral']:.4f} | "
                  f"P_mean={diag['P_mean']:.4f} | A_mean={diag['A_mean']:.4f} | "
                  f"P_std={diag['P_std']:.6f}")
