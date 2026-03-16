"""
Deep Matter Analysis — What forms in a fully-emergent Reality Engine?

Runs the simulation with all emergent coupling constants (no hardcoded physics),
then exhaustively analyzes the matter that forms:

1. Mass distribution — is there quantization? discrete levels?
2. Structure census — how many of each type, stability over time
3. Local field properties at each structure — E/I ratio, temperature, metallicity
4. Spatial correlation — do structures cluster? What spacing?
5. Mass-charge-spin catalog — periodic table attempt

This is a computational microscope on emergent matter.
"""

import math
import os
import sys
import json
import time
from datetime import datetime
from collections import Counter, defaultdict

# Add reality-engine to path
re_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if re_path not in sys.path:
    sys.path.insert(0, re_path)

import torch

from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
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
from src.v3.analyzers.base import Detection
from src.v3.analyzers.gravity import GravityAnalyzer
from src.v3.analyzers.star import StarDetector
from src.v3.analyzers.atom import AtomDetector
from src.v3.analyzers.quantum import QuantumDetector
from src.v3.substrate.manifold import MobiusManifold


PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)


def print_header(title, subtitle=None):
    print("\n" + "=" * 70)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 70 + "\n")


def analyze_mass_distribution(M, threshold=0.5):
    """Analyze the mass field for quantization and structure."""
    M_flat = M.flatten()
    above = M_flat[M_flat > threshold]

    if len(above) == 0:
        return {'n_massive': 0}

    result = {
        'n_massive': len(above),
        'mass_mean': above.mean().item(),
        'mass_std': above.std().item(),
        'mass_min': above.min().item(),
        'mass_max': above.max().item(),
        'mass_median': above.median().item(),
    }

    # Histogram for quantization detection
    n_bins = min(50, max(10, len(above) // 20))
    hist = torch.histc(above, bins=n_bins, min=above.min().item(), max=above.max().item())
    bin_width = (above.max().item() - above.min().item()) / n_bins

    # Find peaks (bins higher than both neighbors)
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > len(above) * 0.02:
            peak_mass = above.min().item() + (i + 0.5) * bin_width
            peaks.append(peak_mass)

    result['n_peaks'] = len(peaks)
    result['peak_masses'] = peaks

    # Check for uniform spacing (mass quantization)
    if len(peaks) >= 3:
        spacings = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        mean_spacing = sum(spacings) / len(spacings)
        spacing_std = (sum((s - mean_spacing)**2 for s in spacings) / len(spacings)) ** 0.5
        result['mean_spacing'] = mean_spacing
        result['spacing_regularity'] = 1 - (spacing_std / (mean_spacing + 1e-10))
        result['quantized'] = result['spacing_regularity'] > 0.7
    else:
        result['quantized'] = False

    return result


def extract_structures(state, manifold, bus):
    """Run all analyzers and extract every detected structure with full properties."""
    detections = []

    # Run analyzers in causal order
    grav = GravityAnalyzer()
    grav_dets = grav.analyze(state, bus, [])
    detections.extend(grav_dets)

    star = StarDetector()
    star_dets = star.analyze(state, bus, grav_dets)
    detections.extend(star_dets)

    atom = AtomDetector(mass_threshold=0.5, gradient_threshold=0.2, metallicity_threshold=0.001)
    atom_dets = atom.analyze(state, bus, grav_dets + star_dets)
    detections.extend(atom_dets)

    quantum = QuantumDetector()
    quantum_dets = quantum.analyze(state, bus, detections)
    detections.extend(quantum_dets)

    return detections


def compute_local_properties(state, pos, manifold, radius=2):
    """Compute detailed local field properties at a given position."""
    u, v = pos
    nu, nv = state.shape
    E, I, M, T = state.E, state.I, state.M, state.T
    Z = state.Z

    # Extract local neighborhood
    props = {}
    props['mass'] = M[u, v].item()
    props['energy'] = E[u, v].item()
    props['information'] = I[u, v].item()
    props['temperature'] = T[u, v].item()
    props['metallicity'] = Z[u, v].item() if Z is not None else 0.0
    props['disequilibrium'] = abs(E[u, v].item() - I[u, v].item())

    # E/I ratio (actualization fraction at this point)
    E2 = E[u, v].item() ** 2
    I2 = I[u, v].item() ** 2
    props['f_local'] = E2 / (E2 + I2 + 1e-12)

    # Local neighborhood stats (clamped to grid bounds)
    u_lo, u_hi = max(0, u-radius), min(nu, u+radius+1)
    v_lo, v_hi = max(0, v-radius), min(nv, v+radius+1)
    local_M = M[u_lo:u_hi, v_lo:v_hi]
    local_E = E[u_lo:u_hi, v_lo:v_hi]
    local_I = I[u_lo:u_hi, v_lo:v_hi]

    props['local_mass_total'] = local_M.sum().item()
    props['local_mass_std'] = local_M.std().item()

    # "Charge" analog: E field circulation in neighborhood
    # curl(E) ~ dE/dv - dE/du in 2D
    if u_hi - u_lo >= 3 and v_hi - v_lo >= 3:
        dE_du = (local_E[2:, 1:-1] - local_E[:-2, 1:-1]) / 2.0
        dE_dv = (local_E[1:-1, 2:] - local_E[1:-1, :-2]) / 2.0
        # Treat as scalar curl for 2D
        curl = (dE_dv.mean() - dE_du.mean()).item()
        props['charge_analog'] = curl
    else:
        props['charge_analog'] = 0.0

    # "Spin" analog: angular momentum of E field around M peak
    # L = sum(r x v) where v ~ gradient of E
    spin = 0.0
    for du in range(-radius, radius+1):
        for dv in range(-radius, radius+1):
            uu = (u + du) % nu
            vv = (v + dv) % nv
            if du == 0 and dv == 0:
                continue
            # r cross grad_E (2D scalar)
            grad_u = (E[(uu+1)%nu, vv] - E[(uu-1)%nu, vv]) / 2.0
            grad_v = (E[uu, (vv+1)%nv] - E[uu, (vv-1)%nv]) / 2.0
            # L_z = du * grad_v - dv * grad_u
            L = du * grad_v.item() - dv * grad_u.item()
            weight = M[uu, vv].item()
            spin += L * weight
    total_mass = local_M.sum().item()
    props['spin_analog'] = spin / (total_mass + 1e-10)

    # Binding energy: peak M minus average surroundings
    props['binding_energy'] = M[u, v].item() - local_M.mean().item()

    # Coherence: |E*I| / (|E|*|I|)
    EI = (local_E * local_I).abs().sum().item()
    E_abs = local_E.abs().sum().item()
    I_abs = local_I.abs().sum().item()
    props['coherence'] = EI / (E_abs * I_abs + 1e-10)

    return props


def classify_structure(props):
    """Classify a structure based on its emergent properties."""
    mass = props['mass']
    charge = abs(props.get('charge_analog', 0))
    spin = abs(props.get('spin_analog', 0))
    Z = props.get('metallicity', 0)
    coherence = props.get('coherence', 0)
    binding = props.get('binding_energy', 0)

    # Mass-based primary classification
    if mass < 0.3:
        if charge < 0.01:
            return 'photon-like'
        else:
            return 'lepton-like'
    elif mass < 1.0:
        if Z > 0.01:
            return 'light-element'  # hydrogen/helium analog
        else:
            return 'meson-like'
    elif mass < 2.0:
        if Z > 0.1:
            return 'medium-element'  # carbon/oxygen analog
        elif coherence > 0.8:
            return 'baryon-like'
        else:
            return 'bound-state'
    elif mass < 3.5:
        if Z > 0.5:
            return 'heavy-element'  # iron-group analog
        else:
            return 'massive-baryon'
    else:
        if binding > 1.0:
            return 'stellar-core'
        else:
            return 'exotic'

    return 'unclassified'


def main():
    print_header("Deep Matter Analysis",
                 "What forms in a fully-emergent Reality Engine?")

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
    bus = EventBus()
    manifold = MobiusManifold(config.nu, config.nv, device=device)

    pac_initial = engine.state.pac_total
    total_ticks = 10000
    snapshot_every = 2000

    # Track evolution
    census_history = []
    all_structures = []

    print(f"  Grid: {config.nu}x{config.nv}")
    print(f"  Ticks: {total_ticks}")
    print(f"  All coupling constants: EMERGENT (no hardcoded physics)")
    print(f"  Device: {device}")

    t0 = time.time()

    for tick in range(1, total_ticks + 1):
        engine.tick()

        if tick % snapshot_every == 0 or tick == total_ticks:
            elapsed = time.time() - t0
            state = engine.state
            pac_drift = state.pac_total - pac_initial

            print_header(f"Snapshot at tick {tick}", f"({elapsed:.0f}s elapsed)")

            # --- Mass distribution analysis ---
            mass_dist = analyze_mass_distribution(state.M, threshold=0.3)
            print(f"  Mass field:")
            print(f"    Cells with M > 0.3: {mass_dist['n_massive']}")
            if mass_dist['n_massive'] > 0:
                print(f"    Mass range: [{mass_dist['mass_min']:.3f}, {mass_dist['mass_max']:.3f}]")
                print(f"    Mass mean:  {mass_dist['mass_mean']:.3f} +/- {mass_dist['mass_std']:.3f}")
                print(f"    Peaks detected: {mass_dist['n_peaks']}")
                if mass_dist['peak_masses']:
                    print(f"    Peak masses: {[f'{m:.3f}' for m in mass_dist['peak_masses']]}")
                if mass_dist.get('quantized'):
                    print(f"    MASS QUANTIZATION DETECTED (regularity={mass_dist['spacing_regularity']:.3f})")

            # --- Structure detection ---
            detections = extract_structures(state, manifold, bus)
            census = Counter(d.kind for d in detections)

            print(f"\n  Structure census:")
            for kind in ['gravity_well', 'star', 'atom', 'hydrogen', 'quantum_coherence']:
                count = census.get(kind, 0)
                if count > 0:
                    print(f"    {kind:25s}: {count}")
            if not census:
                print(f"    (no structures detected)")

            census_history.append({'tick': tick, 'census': dict(census)})

            # --- Deep analysis of each structure ---
            if detections:
                print(f"\n  Detailed structure catalog:")
                classified = defaultdict(list)

                for det in detections:
                    if det.kind in ('quantum_coherence',):
                        continue  # skip non-localized detections

                    props = compute_local_properties(state, det.position, manifold)
                    props.update(det.properties)
                    props['kind'] = det.kind
                    props['position'] = det.position
                    props['tick'] = tick
                    props['classification'] = classify_structure(props)

                    classified[props['classification']].append(props)
                    all_structures.append(props)

                print(f"\n  {'Classification':20s} | {'Count':>5s} | {'Mass range':>14s} | {'Charge':>8s} | {'Spin':>8s} | {'f_local':>8s}")
                print("  " + "-" * 75)
                for cls_name in sorted(classified.keys()):
                    items = classified[cls_name]
                    masses = [p['mass'] for p in items]
                    charges = [p.get('charge_analog', 0) for p in items]
                    spins = [p.get('spin_analog', 0) for p in items]
                    f_locals = [p.get('f_local', 0) for p in items]

                    m_lo, m_hi = min(masses), max(masses)
                    c_mean = sum(charges) / len(charges)
                    s_mean = sum(spins) / len(spins)
                    f_mean = sum(f_locals) / len(f_locals)
                    print(f"  {cls_name:20s} | {len(items):5d} | [{m_lo:5.2f}, {m_hi:5.2f}] | {c_mean:+8.4f} | {s_mean:+8.4f} | {f_mean:8.4f}")

            # --- Emergent coupling snapshot ---
            m = state.metrics
            print(f"\n  Emergent couplings:")
            for key in ['f_local_mean', 'alpha_local_mean', 'lambda_local_mean',
                        'G_local_mean', 'gamma_local_mean']:
                val = m.get(key, None)
                if val is not None:
                    print(f"    {key:25s} = {val:.6f}")

            print(f"\n  PAC drift: {pac_drift:.4e}")
            print(f"  Temperature: mean={state.T.mean().item():.4f} max={state.T.max().item():.4f}")

    # ============================================================
    # Final synthesis
    # ============================================================
    print_header("MATTER SYNTHESIS")

    if all_structures:
        # Mass histogram across all snapshots
        all_masses = [s['mass'] for s in all_structures if s.get('mass', 0) > 0.3]
        all_charges = [s.get('charge_analog', 0) for s in all_structures]
        all_spins = [s.get('spin_analog', 0) for s in all_structures]
        all_f = [s.get('f_local', 0) for s in all_structures]

        print(f"  Total structures cataloged: {len(all_structures)}")
        print(f"  Unique classifications: {len(set(s['classification'] for s in all_structures))}")

        # Classification census
        cls_census = Counter(s['classification'] for s in all_structures)
        print(f"\n  Classification frequency:")
        for cls_name, count in cls_census.most_common():
            pct = count / len(all_structures) * 100
            # Get mass stats for this class
            cls_masses = [s['mass'] for s in all_structures if s['classification'] == cls_name]
            m_mean = sum(cls_masses) / len(cls_masses)
            print(f"    {cls_name:20s}: {count:5d} ({pct:4.1f}%)  mass={m_mean:.3f}")

        # Mass quantization check across full catalog
        if all_masses:
            mass_t = torch.tensor(all_masses)
            final_dist = analyze_mass_distribution(
                mass_t.unsqueeze(0).unsqueeze(0).squeeze(), threshold=0.01)
            print(f"\n  Full catalog mass analysis:")
            print(f"    Masses: {len(all_masses)}")
            print(f"    Range: [{min(all_masses):.3f}, {max(all_masses):.3f}]")
            if final_dist.get('peak_masses'):
                print(f"    Mass peaks: {[f'{m:.3f}' for m in final_dist['peak_masses']]}")
            if final_dist.get('quantized'):
                print(f"    QUANTIZATION DETECTED!")
                print(f"    Spacing: {final_dist.get('mean_spacing', 0):.4f}")
                print(f"    Regularity: {final_dist.get('spacing_regularity', 0):.4f}")

        # Charge distribution
        if all_charges:
            print(f"\n  Charge analog distribution:")
            print(f"    Mean: {sum(all_charges)/len(all_charges):+.6f}")
            print(f"    Std:  {(sum((c - sum(all_charges)/len(all_charges))**2 for c in all_charges)/len(all_charges))**0.5:.6f}")
            pos_charge = sum(1 for c in all_charges if c > 0.01)
            neg_charge = sum(1 for c in all_charges if c < -0.01)
            neutral = len(all_charges) - pos_charge - neg_charge
            print(f"    Positive: {pos_charge}  Negative: {neg_charge}  Neutral: {neutral}")

        # Spin distribution
        if all_spins:
            print(f"\n  Spin analog distribution:")
            print(f"    Mean: {sum(all_spins)/len(all_spins):+.6f}")
            # Check for half-integer quantization
            spin_abs = [abs(s) for s in all_spins if abs(s) > 0.01]
            if spin_abs:
                # Normalize by the smallest nonzero spin
                min_spin = sorted(spin_abs)[len(spin_abs)//10] if len(spin_abs) > 10 else min(spin_abs)
                if min_spin > 0.001:
                    normalized = [s / min_spin for s in spin_abs]
                    # Check if normalized spins cluster near integers or half-integers
                    near_half = sum(1 for s in normalized if abs(s - round(s*2)/2) < 0.15)
                    pct_quantized = near_half / len(normalized) * 100
                    print(f"    Spin quantum: {pct_quantized:.0f}% near half-integer multiples")

        # f_local at structure sites
        if all_f:
            f_mean = sum(all_f) / len(all_f)
            print(f"\n  Actualization ratio at structure sites:")
            print(f"    f_local mean: {f_mean:.4f} (gamma_EM = {0.5772:.4f}, ln(phi) = {LN_PHI:.4f})")

        # Periodic table attempt: group by mass bins
        print(f"\n  --- EMERGENT PERIODIC TABLE ---")
        print(f"  (grouped by mass, showing mean properties)")
        print()

        mass_bins = [(0, 0.5, "ultra-light"), (0.5, 1.0, "light"),
                     (1.0, 1.5, "mid-light"), (1.5, 2.0, "medium"),
                     (2.0, 2.5, "mid-heavy"), (2.5, 3.0, "heavy"),
                     (3.0, 3.5, "very-heavy"), (3.5, 5.0, "super-heavy")]

        print(f"  {'Group':12s} | {'Count':>5s} | {'Mass':>8s} | {'Charge':>8s} | {'Spin':>8s} | {'T':>6s} | {'Z':>6s} | {'f':>6s} | {'Bind':>6s}")
        print("  " + "-" * 85)

        for lo, hi, name in mass_bins:
            group = [s for s in all_structures if lo < s['mass'] <= hi]
            if not group:
                continue
            m_mean = sum(s['mass'] for s in group) / len(group)
            c_mean = sum(s.get('charge_analog', 0) for s in group) / len(group)
            s_mean = sum(s.get('spin_analog', 0) for s in group) / len(group)
            t_mean = sum(s.get('temperature', 0) for s in group) / len(group)
            z_mean = sum(s.get('metallicity', 0) for s in group) / len(group)
            f_mean = sum(s.get('f_local', 0) for s in group) / len(group)
            b_mean = sum(s.get('binding_energy', 0) for s in group) / len(group)
            print(f"  {name:12s} | {len(group):5d} | {m_mean:8.3f} | {c_mean:+8.4f} | {s_mean:+8.4f} | {t_mean:6.3f} | {z_mean:6.4f} | {f_mean:6.3f} | {b_mean:+6.3f}")

    else:
        print("  No structures detected across all snapshots.")

    # Save results
    results = {
        'experiment': 'emergent_matter_analysis',
        'timestamp': datetime.now().isoformat(),
        'grid': f"{config.nu}x{config.nv}",
        'ticks': total_ticks,
        'census_history': census_history,
        'total_structures': len(all_structures),
        'classifications': dict(Counter(s['classification'] for s in all_structures)) if all_structures else {},
        'structures': all_structures[:200],  # cap for file size
    }

    out_dir = os.path.join(re_path, 'scripts', 'results')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"matter_analysis_{ts}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
