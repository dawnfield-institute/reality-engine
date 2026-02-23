"""
low_k_mix sweep: find the mix ratio that gives xi=1.057.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import RealityEngine
from src.substrate.constants import XI_REFERENCE


def run_mix_sweep():
    print("=" * 70)
    print(f"  low_k_mix sweep (Pm=0.1, rho=5, ki=1) target={XI_REFERENCE}")
    print("=" * 70)

    mixes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for mix in mixes:
        sec_params = {
            "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
            "sigma_0": 0.1, "dt": 0.01,
            "xi_gain": 2.0, "rho": 5.0, "phi_source": 0.618,
            "alpha_rbf": 5.0, "rbf_decay": 0.995,
            "ki_rbf": 1.0, "integral_clamp": 1.0, "rbf_omega": 0.2,
            "low_k_mix": mix,
        }
        config = {
            "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
            "sec": sec_params,
            "init": {"seed": 42, "P_mean": 0.1, "P_noise": 0.01,
                     "A_mean": 0.1, "A_noise": 0.01,
                     "antiperiodic_amp": 0.01},
            "pac": {"mode": "enforce"},
        }
        engine = RealityEngine(config)
        for i in range(10000):
            diag = engine.step()
            if engine.diagnostics.diverged:
                break
        xi = diag.get("xi_L2", float("nan"))
        pm = diag.get("P_mean", float("nan"))
        ps = diag.get("P_std", float("nan"))
        err = (xi - XI_REFERENCE) / XI_REFERENCE * 100
        marker = " <<<" if abs(err) < 0.5 else " <" if abs(err) < 2.0 else ""
        print(f"  mix={mix:.1f}  xi={xi:.6f} (err={err:+6.2f}%)  Pm={pm:.4f}  Pstd={ps:.4f}{marker}")


def run_convergence(mix, rho, ki, pm, n_steps=20000):
    print(f"\n{'='*70}")
    print(f"  CONVERGENCE: mix={mix}, rho={rho}, ki={ki}, Pm={pm}")
    print(f"{'='*70}")
    sec_params = {
        "kappa": 0.1, "gamma": 1.0, "beta_0": 1.0,
        "sigma_0": 0.1, "dt": 0.01,
        "xi_gain": 2.0, "rho": rho, "phi_source": 0.618,
        "alpha_rbf": 5.0, "rbf_decay": 0.995,
        "ki_rbf": ki, "integral_clamp": 1.0, "rbf_omega": 0.2,
        "low_k_mix": mix,
    }
    config = {
        "manifold": {"n_u": 128, "n_v": 64, "device": "cpu"},
        "sec": sec_params,
        "init": {"seed": 42, "P_mean": pm, "P_noise": 0.01,
                 "A_mean": pm, "A_noise": 0.01,
                 "antiperiodic_amp": 0.01},
        "pac": {"mode": "enforce"},
    }
    engine = RealityEngine(config)
    print(f"  {'step':>6s} | {'xi_L2':>10s} | {'err%':>8s} | {'Pm':>8s} | {'Pstd':>8s} | {'M':>8s}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for i in range(n_steps):
        diag = engine.step()
        if i % 2000 == 0 or i == n_steps - 1:
            xi = diag.get("xi_L2", 0)
            err = (xi - XI_REFERENCE) / XI_REFERENCE * 100
            print(f"  {diag['t']:6d} | {xi:10.6f} | {err:+8.3f}% | {diag.get('P_mean',0):8.4f} | "
                  f"{diag.get('P_std',0):8.4f} | {diag.get('M_rbf',0):8.4f}")
    xi_final = diag.get("xi_L2", 0)
    err_final = abs(xi_final - XI_REFERENCE) / XI_REFERENCE
    print(f"\n  Final: xi = {xi_final:.6f} (target = {XI_REFERENCE}, error = {err_final:.3%})")


if __name__ == "__main__":
    run_mix_sweep()
    # Run convergence trace for the closest match
    run_convergence(mix=0.5, rho=5, ki=1, pm=0.1)
