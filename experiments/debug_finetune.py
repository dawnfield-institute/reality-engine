"""Fine-tuning: narrow sweep around mix=0.4 to nail xi=1.0571."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.engine import RealityEngine
from src.substrate.constants import XI_REFERENCE

def run():
    mixes = [0.33, 0.35, 0.37, 0.39, 0.40, 0.41, 0.43, 0.45, 0.47]
    print(f"  low_k_mix fine-tune (target={XI_REFERENCE})")
    print("-" * 65)
    for mix in mixes:
        sec = {"kappa":0.1,"gamma":1,"beta_0":1,"sigma_0":0.1,"dt":0.01,
               "xi_gain":2,"rho":5,"phi_source":0.618,"alpha_rbf":5,
               "rbf_decay":0.995,"ki_rbf":1,"integral_clamp":1,"rbf_omega":0.2,
               "low_k_mix":mix}
        cfg = {"manifold":{"n_u":128,"n_v":64,"device":"cpu"},
               "sec":sec,"init":{"seed":42,"P_mean":0.1,"P_noise":0.01,
               "A_mean":0.1,"A_noise":0.01,"antiperiodic_amp":0.01},
               "pac":{"mode":"enforce"}}
        eng = RealityEngine(cfg)
        for i in range(12000):
            d = eng.step()
            if eng.diagnostics.diverged: break
        xi = d.get("xi_L2", float("nan"))
        err = (xi - XI_REFERENCE) / XI_REFERENCE * 100
        m = " <<<" if abs(err) < 0.1 else " <<" if abs(err) < 0.3 else " <" if abs(err) < 0.5 else ""
        print(f"  mix={mix:.2f}  xi={xi:.6f} (err={err:+6.3f}%){m}")

if __name__ == "__main__":
    run()
