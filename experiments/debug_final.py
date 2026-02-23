"""Final convergence trace at optimal parameters."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.engine import RealityEngine
from src.substrate.constants import XI_REFERENCE, PHI_INV

def run():
    # mix = PHI_INV^2 = 0.382 (golden ratio squared, topology-natural)
    mix = PHI_INV ** 2
    print(f"  mix = PHI_INV^2 = {mix:.6f}")
    print(f"  Target xi = {XI_REFERENCE}")
    sec = {"kappa":0.1,"gamma":1,"beta_0":1,"sigma_0":0.1,"dt":0.01,
           "xi_gain":2,"rho":5,"phi_source":0.618,"alpha_rbf":5,
           "rbf_decay":0.995,"ki_rbf":1,"integral_clamp":1,"rbf_omega":0.2,
           "low_k_mix":mix}
    cfg = {"manifold":{"n_u":128,"n_v":64,"device":"cpu"},
           "sec":sec,"init":{"seed":42,"P_mean":0.1,"P_noise":0.01,
           "A_mean":0.1,"A_noise":0.01,"antiperiodic_amp":0.01},
           "pac":{"mode":"enforce"}}
    eng = RealityEngine(cfg)
    n = 20000
    print(f"  {'step':>6s} | {'xi':>10s} | {'err%':>8s} | {'Pm':>8s} | {'Pstd':>8s}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for i in range(n):
        d = eng.step()
        if i % 2000 == 0 or i == n-1:
            xi = d.get("xi_L2",0)
            err = (xi-XI_REFERENCE)/XI_REFERENCE*100
            print(f"  {d['t']:6d} | {xi:10.6f} | {err:+8.3f}% | {d.get('P_mean',0):8.4f} | {d.get('P_std',0):8.4f}")
    xi = d.get("xi_L2",0)
    print(f"\n  Final xi = {xi:.6f} (target {XI_REFERENCE}, error {abs(xi-XI_REFERENCE)/XI_REFERENCE:.4%})")

    # Also try with standard Pm=0.5
    print(f"\n  --- Same params but Pm=0.5 (standard) ---")
    cfg2 = {"manifold":{"n_u":128,"n_v":64,"device":"cpu"},
           "sec":sec,"init":{"seed":42,"P_mean":0.5,"P_noise":0.01,
           "A_mean":0.5,"A_noise":0.01,"antiperiodic_amp":0.05},
           "pac":{"mode":"enforce"}}
    eng2 = RealityEngine(cfg2)
    for i in range(10000):
        d2 = eng2.step()
    xi2 = d2.get("xi_L2",0)
    err2 = (xi2-XI_REFERENCE)/XI_REFERENCE*100
    print(f"  At step 10000: xi={xi2:.6f} (err={err2:+.2f}%) Pm={d2.get('P_mean',0):.4f}")

if __name__ == "__main__":
    run()
