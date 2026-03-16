"""
Matter Analysis v2 — Full-spectrum scan of emergent matter.

v1 only saw top-20 heaviest structures (all at mass cap). This version:
1. Scans the ENTIRE field for local maxima at ALL mass scales
2. Captures the early mass spectrum before everything collapses to cap
3. More frequent snapshots to catch transient structures
4. Tracks mass peak evolution over time
"""

import math
import os
import sys
import json
import time
from datetime import datetime
from collections import Counter, defaultdict

re_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if re_path not in sys.path:
    sys.path.insert(0, re_path)

import torch
import torch.nn.functional as F

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
from src.v3.substrate.manifold import MobiusManifold

PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)


def print_header(title, subtitle=None):
    print("\n" + "=" * 70)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 70 + "\n")


def find_all_local_maxima(M, min_mass=0.1):
    """Find ALL local maxima in the mass field — not just top-20.

    A local maximum is a cell greater than all 4 cardinal neighbors.
    Returns list of (u, v, mass) sorted by mass descending.
    """
    # Compare each cell with its 4 neighbors
    is_max = (
        (M > torch.roll(M, 1, 0)) &
        (M > torch.roll(M, -1, 0)) &
        (M > torch.roll(M, 1, 1)) &
        (M > torch.roll(M, -1, 1)) &
        (M > min_mass)
    )

    positions = torch.nonzero(is_max, as_tuple=False)
    if len(positions) == 0:
        return []

    masses = M[is_max]
    # Sort by mass descending
    sorted_idx = masses.argsort(descending=True)

    results = []
    for i in range(len(sorted_idx)):
        idx = sorted_idx[i]
        u, v = positions[idx][0].item(), positions[idx][1].item()
        results.append((u, v, masses[idx].item()))

    return results


def compute_structure_properties(state, u, v, radius=2):
    """Compute all physical properties at a structure site."""
    nu, nv = state.shape
    E, I, M, T, Z = state.E, state.I, state.M, state.T, state.Z

    props = {
        'position': (u, v),
        'mass': M[u, v].item(),
        'energy': E[u, v].item(),
        'information': I[u, v].item(),
        'temperature': T[u, v].item(),
        'metallicity': Z[u, v].item() if Z is not None else 0.0,
        'disequilibrium': abs(E[u, v].item() - I[u, v].item()),
    }

    # Actualization fraction
    E2 = E[u, v].item() ** 2
    I2 = I[u, v].item() ** 2
    props['f_local'] = E2 / (E2 + I2 + 1e-12)

    # Local neighborhood
    u_lo, u_hi = max(0, u-radius), min(nu, u+radius+1)
    v_lo, v_hi = max(0, v-radius), min(nv, v+radius+1)
    local_M = M[u_lo:u_hi, v_lo:v_hi]
    local_E = E[u_lo:u_hi, v_lo:v_hi]
    local_I = I[u_lo:u_hi, v_lo:v_hi]

    # Integrated mass (total mass in neighborhood — "atomic mass")
    props['integrated_mass'] = local_M.sum().item()

    # Binding energy: how much more mass at center than surroundings
    props['binding_energy'] = M[u, v].item() - local_M.mean().item()

    # Coherence: how aligned are E and I?
    EI = (local_E * local_I).sum().item()
    E_norm = (local_E ** 2).sum().sqrt().item()
    I_norm = (local_I ** 2).sum().sqrt().item()
    props['coherence'] = EI / (E_norm * I_norm + 1e-10)

    # Charge analog: E field curl
    if u_hi - u_lo >= 3 and v_hi - v_lo >= 3:
        dE_du = (local_E[2:, 1:-1] - local_E[:-2, 1:-1]) / 2.0
        dE_dv = (local_E[1:-1, 2:] - local_E[1:-1, :-2]) / 2.0
        props['charge'] = (dE_dv.mean() - dE_du.mean()).item()
    else:
        props['charge'] = 0.0

    # Spin analog: angular momentum
    spin = 0.0
    for du in range(-min(radius, u), min(radius+1, nu-u)):
        for dv in range(-min(radius, v), min(radius+1, nv-v)):
            if du == 0 and dv == 0:
                continue
            uu, vv = (u + du) % nu, (v + dv) % nv
            grad_u = (E[(uu+1)%nu, vv] - E[(uu-1)%nu, vv]) / 2.0
            grad_v = (E[uu, (vv+1)%nv] - E[uu, (vv-1)%nv]) / 2.0
            L = du * grad_v.item() - dv * grad_u.item()
            spin += L * M[uu, vv].item()
    props['spin'] = spin / (local_M.sum().item() + 1e-10)

    # Gravitational coupling at this cell
    M2 = M[u, v].item() ** 2
    diseq2 = (E[u, v].item() - I[u, v].item()) ** 2
    props['G_local'] = M2 / (M2 + diseq2 + 1e-12)

    # Mass generation rate at this cell
    props['gamma_local'] = diseq2 / (E2 + I2 + 1e-12)

    return props


def mass_histogram_peaks(masses, n_bins=30):
    """Find peaks in mass distribution."""
    if len(masses) < 10:
        return [], {}

    m_min, m_max = min(masses), max(masses)
    if m_max - m_min < 0.01:
        return [m_min], {}

    bin_width = (m_max - m_min) / n_bins
    bins = [0] * n_bins
    for m in masses:
        idx = min(int((m - m_min) / bin_width), n_bins - 1)
        bins[idx] += 1

    peaks = []
    for i in range(1, n_bins - 1):
        if bins[i] > bins[i-1] and bins[i] > bins[i+1] and bins[i] >= 3:
            peak_mass = m_min + (i + 0.5) * bin_width
            peaks.append((peak_mass, bins[i]))

    # Check spacing regularity
    info = {}
    if len(peaks) >= 3:
        peak_masses = [p[0] for p in peaks]
        spacings = [peak_masses[i+1] - peak_masses[i] for i in range(len(peak_masses)-1)]
        mean_sp = sum(spacings) / len(spacings)
        std_sp = (sum((s - mean_sp)**2 for s in spacings) / len(spacings)) ** 0.5
        info['mean_spacing'] = mean_sp
        info['spacing_std'] = std_sp
        info['regularity'] = 1 - std_sp / (mean_sp + 1e-10)

    return peaks, info


def main():
    print_header("Matter Analysis v2 — Full Spectrum",
                 "Scanning ALL mass scales, frequent snapshots")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True,
        actualization_threshold=0.05,
    )

    pipeline = Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), GravitationalCollapseOperator(), FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), AdaptiveOperator(), TimeEmergenceOperator(),
    ])

    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    pac_initial = engine.state.pac_total
    total_ticks = 10000

    # Frequent snapshots early (where mass spectrum is richest), sparser later
    snapshots = [100, 200, 500, 1000, 1500, 2000, 3000, 5000, 7000, 10000]

    all_snapshots = []
    mass_peak_evolution = []

    print(f"  Grid: {config.nu}x{config.nv}")
    print(f"  Snapshots at: {snapshots}")

    t0 = time.time()
    snap_idx = 0

    for tick in range(1, total_ticks + 1):
        engine.tick()

        if snap_idx < len(snapshots) and tick == snapshots[snap_idx]:
            snap_idx += 1
            elapsed = time.time() - t0
            state = engine.state

            print_header(f"Tick {tick}", f"({elapsed:.0f}s)")

            # Find ALL local maxima
            maxima = find_all_local_maxima(state.M, min_mass=0.1)
            masses = [m for _, _, m in maxima]

            print(f"  Local maxima found: {len(maxima)}")
            if masses:
                print(f"  Mass range: [{min(masses):.3f}, {max(masses):.3f}]")
                print(f"  Mass mean:  {sum(masses)/len(masses):.3f}")

            # Mass distribution with peaks
            peaks, peak_info = mass_histogram_peaks(masses)
            if peaks:
                print(f"\n  Mass peaks ({len(peaks)}):")
                for mass, count in sorted(peaks):
                    print(f"    M = {mass:.3f}  (count = {count})")
                if peak_info.get('regularity', 0) > 0.5:
                    print(f"    Spacing: {peak_info['mean_spacing']:.4f} +/- {peak_info['spacing_std']:.4f}")
                    print(f"    Regularity: {peak_info['regularity']:.3f}")
                    # Check if spacing relates to any DFT constant
                    sp = peak_info['mean_spacing']
                    for name, val in [('ln(phi)', LN_PHI), ('1/phi', 1/PHI), ('1/phi^2', 1/PHI**2)]:
                        err = abs(sp - val) / val * 100
                        if err < 30:
                            print(f"    Spacing ~ {name} = {val:.4f} (err = {err:.1f}%)")

            mass_peak_evolution.append({
                'tick': tick,
                'n_maxima': len(maxima),
                'peaks': [(m, c) for m, c in peaks],
                'peak_info': peak_info,
            })

            # Mass bins for the periodic table view
            mass_bins = [
                (0.1, 0.3, 'trace'),
                (0.3, 0.6, 'ultra-light'),
                (0.6, 1.0, 'light'),
                (1.0, 1.5, 'mid-light'),
                (1.5, 2.0, 'medium'),
                (2.0, 2.5, 'mid-heavy'),
                (2.5, 3.0, 'heavy'),
                (3.0, 3.5, 'very-heavy'),
                (3.5, 4.1, 'at-cap'),
            ]

            print(f"\n  Mass distribution:")
            print(f"  {'Group':12s} | {'Count':>5s} | {'% of all':>7s} | {'Mass range':>14s}")
            print("  " + "-" * 48)
            for lo, hi, name in mass_bins:
                count = sum(1 for m in masses if lo < m <= hi)
                if count > 0:
                    sub = [m for m in masses if lo < m <= hi]
                    pct = count / len(masses) * 100
                    print(f"  {name:12s} | {count:5d} | {pct:6.1f}% | [{min(sub):.3f}, {max(sub):.3f}]")

            # Sample detailed properties from a few structures at each mass scale
            if len(maxima) > 0:
                # Pick up to 5 from each mass bin
                print(f"\n  Sample structure properties:")
                print(f"  {'Mass':>6s} | {'Charge':>8s} | {'Spin':>8s} | {'T':>6s} | {'Z':>6s} | {'f':>6s} | {'G':>6s} | {'Coher':>6s} | {'Bind':>6s}")
                print("  " + "-" * 75)

                sampled = []
                for lo, hi, name in mass_bins:
                    bin_maxima = [(u, v, m) for u, v, m in maxima if lo < m <= hi]
                    # Sample up to 3 from this bin
                    for u, v, m in bin_maxima[:3]:
                        p = compute_structure_properties(state, u, v)
                        sampled.append(p)
                        print(f"  {p['mass']:6.3f} | {p['charge']:+8.4f} | {p['spin']:+8.4f} | {p['temperature']:6.3f} | {p['metallicity']:6.4f} | {p['f_local']:6.3f} | {p['G_local']:6.3f} | {p['coherence']:+6.3f} | {p['binding_energy']:+6.3f}")

                snap_data = {
                    'tick': tick,
                    'n_maxima': len(maxima),
                    'mass_distribution': {name: sum(1 for m in masses if lo < m <= hi) for lo, hi, name in mass_bins},
                    'samples': sampled[:30],
                }
                all_snapshots.append(snap_data)

            # Emergent couplings
            m = state.metrics
            print(f"\n  Couplings: f={m.get('f_local_mean',0):.4f} alpha={m.get('alpha_local_mean',0):.4f} G={m.get('G_local_mean',0):.4f} gamma={m.get('gamma_local_mean',0):.4f}")
            print(f"  PAC drift: {state.pac_total - pac_initial:.4e}")

    # ============================================================
    # Final synthesis: what does the periodic table look like?
    # ============================================================
    print_header("SYNTHESIS — Emergent Periodic Table")

    # Use the richest snapshot (most mass diversity, typically early)
    richest = max(all_snapshots, key=lambda s: len(s.get('samples', [])))
    print(f"  Richest snapshot: tick {richest['tick']} ({richest['n_maxima']} structures)")

    # Aggregate all samples across all snapshots
    all_samples = []
    for snap in all_snapshots:
        all_samples.extend(snap.get('samples', []))

    if all_samples:
        print(f"  Total sampled structures: {len(all_samples)}")

        # Check charge distribution
        charges = [s['charge'] for s in all_samples]
        spins = [s['spin'] for s in all_samples]

        print(f"\n  Charge distribution:")
        print(f"    Mean: {sum(charges)/len(charges):+.6f}")
        pos = sum(1 for c in charges if c > 0.05)
        neg = sum(1 for c in charges if c < -0.05)
        neutral = len(charges) - pos - neg
        print(f"    + : {pos}  - : {neg}  0 : {neutral}")

        # Check for charge quantization
        charge_abs = sorted(abs(c) for c in charges if abs(c) > 0.01)
        if len(charge_abs) > 5:
            # Minimum nonzero charge as "unit"
            q_unit = charge_abs[len(charge_abs)//10]  # 10th percentile
            if q_unit > 0.001:
                normalized = [c / q_unit for c in charge_abs]
                near_int = sum(1 for n in normalized if abs(n - round(n)) < 0.2)
                print(f"    Unit charge: {q_unit:.4f}")
                print(f"    Quantized (near integer multiples): {near_int/len(normalized)*100:.0f}%")

        print(f"\n  Spin distribution:")
        print(f"    Mean: {sum(spins)/len(spins):+.6f}")
        spin_abs = sorted(abs(s) for s in spins if abs(s) > 0.01)
        if len(spin_abs) > 5:
            s_unit = spin_abs[len(spin_abs)//10]
            if s_unit > 0.001:
                normalized = [s / s_unit for s in spin_abs]
                near_half = sum(1 for n in normalized if abs(n - round(n*2)/2) < 0.15)
                print(f"    Unit spin: {s_unit:.4f}")
                print(f"    Half-integer quantized: {near_half/len(normalized)*100:.0f}%")

    # Mass peak evolution
    print(f"\n  Mass peak evolution:")
    print(f"  {'Tick':>6s} | {'N peaks':>7s} | {'Peak masses'}")
    print("  " + "-" * 60)
    for entry in mass_peak_evolution:
        peak_str = ", ".join(f"{m:.3f}" for m, c in sorted(entry['peaks']))
        print(f"  {entry['tick']:6d} | {len(entry['peaks']):7d} | {peak_str[:50]}")

    # Save
    results = {
        'experiment': 'emergent_matter_v2',
        'timestamp': datetime.now().isoformat(),
        'mass_peak_evolution': mass_peak_evolution,
        'snapshots': [{k: v for k, v in s.items() if k != 'samples'} for s in all_snapshots],
        'all_samples': all_samples[:500],
    }
    out_dir = os.path.join(re_path, 'scripts', 'results')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"matter_v2_{ts}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
