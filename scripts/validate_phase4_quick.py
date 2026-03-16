"""Quick Phase 4 validation — PhiCascade + pi/2 collapse modulation.

Runs 10K ticks with full pipeline (including PhiCascade + depth-modulated
actualization threshold) and checks stability + spectral properties.
"""

import math
import os
import sys
import time

re_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if re_path not in sys.path:
    sys.path.insert(0, re_path)

import torch
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
from src.v3.operators.spin_statistics import SpinStatisticsOperator
from src.v3.operators.charge_dynamics import ChargeDynamicsOperator
from src.v3.operators.phi_cascade import PhiCascadeOperator

PHI = (1 + math.sqrt(5)) / 2


def find_mass_peaks(M, min_mass=0.1, n_bins=40):
    is_max = (
        (M > torch.roll(M, 1, 0)) &
        (M > torch.roll(M, -1, 0)) &
        (M > torch.roll(M, 1, 1)) &
        (M > torch.roll(M, -1, 1)) &
        (M > min_mass)
    )
    masses = M[is_max].cpu().tolist()
    if len(masses) < 10:
        return [], masses
    m_min, m_max = min(masses), max(masses)
    if m_max - m_min < 0.05:
        return [(m_min, len(masses))], masses
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
    return peaks, masses


def main():
    print("=" * 70)
    print("  Phase 4 Validation — PhiCascade + pi/2 Collapse Modulation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True,
        actualization_threshold=0.05,
    )

    torch.manual_seed(42)
    pipeline = Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
        GravitationalCollapseOperator(),
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), AdaptiveOperator(), TimeEmergenceOperator(),
    ])

    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)
    pac_initial = engine.state.pac_total

    # Track oscillation at a few mass peak sites for spectral analysis
    checkpoints = [500, 1000, 2000, 5000, 10000]
    t0 = time.time()
    cp_idx = 0

    # Record E values at 4 fixed sites for FFT later
    record_sites = [(32, 16), (64, 32), (96, 48), (16, 8)]
    E_traces = {s: [] for s in record_sites}
    record_every = 5  # every 5 ticks

    total_ticks = 10000
    for tick in range(1, total_ticks + 1):
        engine.tick()

        if tick % record_every == 0:
            for s in record_sites:
                E_traces[s].append(engine.state.E[s[0], s[1]].item())

        if cp_idx < len(checkpoints) and tick == checkpoints[cp_idx]:
            cp_idx += 1
            elapsed = time.time() - t0
            state = engine.state
            peaks, masses = find_mass_peaks(state.M)
            M_cap = config.field_scale / 5.0
            n_at_cap = sum(1 for m in masses if m >= M_cap * 0.95)
            frac_cap = n_at_cap / max(len(masses), 1)

            # Phi spacing
            PHI_INV2 = 1.0 / PHI ** 2
            peak_masses = sorted(p[0] for p in peaks)
            spacings = [peak_masses[i+1] - peak_masses[i] for i in range(len(peak_masses)-1)] if len(peak_masses) > 1 else []
            mean_sp = sum(spacings) / len(spacings) if spacings else 0
            err_phi2 = abs(mean_sp - PHI_INV2) / PHI_INV2 * 100 if spacings else 999

            # Cascade metrics
            cd = state.metrics.get('cascade_depth_mean', 0)
            pp = state.metrics.get('phi_proximity_mean', 0)
            ac = state.metrics.get('actualization_count', 0)
            fm = state.metrics.get('f_local_mean', 0)

            peak_str = ", ".join(f"{p:.3f}" for p in peak_masses[:6])
            print(f"\n  Tick {tick:5d} ({elapsed:.0f}s):")
            print(f"    Structures: {len(masses)}, Peaks: {len(peaks)} [{peak_str}]")
            print(f"    Cap: {frac_cap:.1%}, PAC drift: {state.pac_total - pac_initial:.2e}")
            print(f"    Spacing: {mean_sp:.4f} (err vs 1/phi^2: {err_phi2:.1f}%)")
            print(f"    Cascade depth: {cd:.2f}, Phi proximity: {pp:.3f}")
            print(f"    Actualization: {ac} cells, f_local={fm:.4f}")

    # FFT analysis at recorded sites
    print(f"\n{'='*70}")
    print(f"  SPECTRAL ANALYSIS at fixed sites")
    print(f"{'='*70}")

    for site, trace in E_traces.items():
        if len(trace) < 100:
            continue
        signal = torch.tensor(trace, dtype=torch.float64)
        signal = signal - signal.mean()
        fft = torch.fft.rfft(signal)
        power = fft.abs().pow(2)
        freqs = torch.fft.rfftfreq(len(signal), d=record_every * config.dt)

        # Skip DC, find top 3 frequencies
        power[0] = 0
        top_idx = power.argsort(descending=True)[:5]
        print(f"\n  Site {site} (M={engine.state.M[site[0], site[1]].item():.3f}):")
        for i, idx in enumerate(top_idx):
            f = freqs[idx].item()
            p = power[idx].item()
            print(f"    f={f:.4f} (power={p:.2f})")

        # Check if frequency ratios are ~pi/2
        top_freqs = [freqs[idx].item() for idx in top_idx[:3] if freqs[idx].item() > 0]
        if len(top_freqs) >= 2:
            ratios = [top_freqs[i] / top_freqs[i+1] for i in range(len(top_freqs)-1) if top_freqs[i+1] > 0]
            for j, r in enumerate(ratios):
                pi2 = math.pi / 2
                err = abs(r - pi2) / pi2 * 100
                print(f"    Ratio f{j}/f{j+1} = {r:.4f} (pi/2={pi2:.4f}, err={err:.1f}%)")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
