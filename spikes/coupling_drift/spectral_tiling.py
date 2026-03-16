"""Spike: cascade-depth spectral tiling filter for gravity.

DFT theory (exp_36) says the cosmological constant IS the SEC cost of
tiling local PAC patches globally. Per cascade level, the residual
suppression is ln^2(2) = 0.4805 (from exp_28: round-trip deficit at
Landauer fraction). Over 2*183*Xi levels, this gives 10^-123 — the CC.

For the Reality Engine's Poisson solver, this means gravity should be
SCALE-DEPENDENT in the spectral domain:
  - High |k| (local scale): full gravity strength
  - Low |k| (global scale): suppressed by tiling cost

The spectral filter is:
  filter(k) = (ln^2(2))^(Xi * n(k))
where n(k) = log_phi(|k|_max / |k|) is the cascade depth at mode k.

This makes gravity local-dominant: strong attraction between neighbors
(web filaments), weak across the entire grid (prevents global clumping).

Tests several suppression strengths to find the sweet spot.
"""

import math
import os
import sys
import time

re_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))
if re_path not in sys.path:
    sys.path.insert(0, re_path)

import torch

from src.v3.engine.engine import Engine
from src.v3.engine.config import SimulationConfig
from src.v3.engine.state import FieldState
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
from src.v3.operators.spin_statistics import SpinStatisticsOperator
from src.v3.operators.charge_dynamics import ChargeDynamicsOperator
from src.v3.operators.phi_cascade import PhiCascadeOperator
from src.v3.operators.sec_tracking import SECTrackingOperator

PHI = (1 + math.sqrt(5)) / 2
LN2 = math.log(2)
LN_PHI = math.log(PHI)
GAMMA_EM = 0.5772156649015329
XI = GAMMA_EM + LN_PHI  # 1.05843 — global frame attractor
LN2_SQ = LN2 ** 2       # 0.4805 — tiling residual per cascade level
_EPS = 1e-12

TARGETS = {
    "f_local":     0.5772,
    "gamma_local": 1 / PHI,
    "alpha_local": LN2,
    "G_local":     1 / PHI**2,
    "lambda_local": 1 - LN2,
}


class TilingGravityOperator(GravitationalCollapseOperator):
    """Gravity with cascade-depth spectral tiling filter.

    Each spectral mode k has a cascade depth n(k) = log_phi(|k|_max / |k|).
    The tiling suppression at depth n is (ln^2(2))^(Xi * n * strength),
    where strength controls how aggressive the filtering is.
    """

    def __init__(self, strength: float = 1.0) -> None:
        super().__init__()
        self._strength = strength
        self._tiling_filter = None

    def _build_tiling_filter(self, nu: int, nv: int, device: torch.device) -> torch.Tensor:
        """Build the spectral tiling filter.

        filter(k) = (ln^2(2))^(Xi * n(k) * strength)
        n(k) = log_phi(|k|_max / max(|k|, 1))

        High |k| (local): filter -> 1 (full gravity)
        Low |k| (global): filter -> small (suppressed)
        """
        # Frequency grids (centered: negative freqs for k > N/2)
        ku = torch.arange(nu, device=device, dtype=torch.float64)
        kv = torch.arange(nv, device=device, dtype=torch.float64)
        # Map to centered frequencies: [0, 1, ..., N/2, -(N/2-1), ..., -1]
        ku = torch.where(ku > nu // 2, ku - nu, ku)
        kv = torch.where(kv > nv // 2, kv - nv, kv)
        ku_grid, kv_grid = torch.meshgrid(ku, kv, indexing='ij')

        # Wavenumber magnitude |k|
        k_mag = torch.sqrt(ku_grid**2 + kv_grid**2)
        k_max = torch.sqrt(torch.tensor(
            (nu // 2)**2 + (nv // 2)**2, dtype=torch.float64, device=device
        ))

        # Cascade depth: n(k) = log_phi(k_max / |k|)
        # At |k| = k_max: n = 0 (local, no suppression)
        # At |k| = 1: n = log_phi(k_max) ~ 8-9 levels
        k_safe = torch.clamp(k_mag, min=1.0)
        cascade_depth = torch.log(k_max / k_safe) / LN_PHI

        # Tiling suppression: (ln^2(2))^(Xi * n * strength)
        # ln^2(2) = 0.4805, so log of filter = Xi * n * strength * ln(0.4805)
        log_suppression = XI * cascade_depth * self._strength * math.log(LN2_SQ)
        tiling_filter = torch.exp(log_suppression)

        # DC component: zero (no mean potential)
        tiling_filter[0, 0] = 0.0

        return tiling_filter

    def _solve_poisson(self, source: torch.Tensor) -> torch.Tensor:
        """Spectral Poisson solver WITH tiling filter."""
        nu, nv = source.shape

        if not hasattr(self, '_inv_lap') or self._inv_lap.shape != (nu, nv):
            self._inv_lap = self._build_inv_laplacian(nu, nv, source.device)

        if self._tiling_filter is None or self._tiling_filter.shape != (nu, nv):
            self._tiling_filter = self._build_tiling_filter(nu, nv, source.device)

        # FFT -> divide by eigenvalues -> apply tiling filter -> IFFT
        source_hat = torch.fft.fft2(source)
        phi_hat = source_hat * self._inv_lap * self._tiling_filter
        phi = torch.fft.ifft2(phi_hat).real
        return phi


# ---------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------

def build_pipeline(gravity_op=None):
    gravity = gravity_op or GravitationalCollapseOperator()
    return Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
        gravity,
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), SECTrackingOperator(),
        AdaptiveOperator(), TimeEmergenceOperator(),
    ])


def pct_err(measured, target):
    if target == 0:
        return abs(measured) * 100
    return abs(measured - target) / abs(target) * 100


def grade(err):
    if err < 1:   return "A+"
    if err < 5:   return "A"
    if err < 10:  return "B"
    if err < 15:  return "C"
    if err < 30:  return "D"
    return "F"


def run_variant(name, gravity_op, device, ticks=5000):
    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True, actualization_threshold=0.05,
    )
    torch.manual_seed(42)
    pipeline = build_pipeline(gravity_op)
    engine = Engine(config=config, pipeline=pipeline)
    engine.initialize("big_bang", temperature=3.0)

    checkpoints = [1000, 2000, 5000]
    results = {}
    cp_idx = 0

    for tick in range(1, ticks + 1):
        engine.tick()
        if cp_idx < len(checkpoints) and tick == checkpoints[cp_idx]:
            cp_idx += 1
            met = engine.state.metrics
            M = engine.state.M
            M_cap = config.field_scale / 5.0

            results[tick] = {
                "f_local": met.get("f_local_mean", 0),
                "gamma_local": met.get("gamma_local_mean", 0),
                "alpha_local": met.get("alpha_local_mean", 0),
                "G_local": met.get("G_local_mean", 0),
                "lambda_local": met.get("lambda_local_mean", 0),
                "M_mean": M.mean().item(),
                "M_std": M.std().item(),
                "M_max": M.max().item(),
                "frac_cap": (M > M_cap * 0.9).float().mean().item(),
                "frac_empty": (M < 0.1).float().mean().item(),
                "xi_mod_mean": met.get("xi_mod_mean", 0),
            }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test several tiling strengths
    VARIANTS = {
        "A_no_tiling":       GravitationalCollapseOperator(),
        "B_tiling_0.25x":    TilingGravityOperator(strength=0.25),
        "C_tiling_0.50x":    TilingGravityOperator(strength=0.50),
        "D_tiling_0.75x":    TilingGravityOperator(strength=0.75),
        "E_tiling_1.00x":    TilingGravityOperator(strength=1.00),
        "F_tiling_1.50x":    TilingGravityOperator(strength=1.50),
    }

    print(f"\n{'='*105}")
    print(f"  CASCADE-DEPTH SPECTRAL TILING FILTER")
    print(f"  Device: {device} | Grid: 128x64 | 5000 ticks per variant")
    print(f"  Tiling residual per level: ln^2(2) = {LN2_SQ:.4f}")
    print(f"  Xi (global attractor): {XI:.5f}")
    print(f"  Max cascade depth on grid: log_phi(sqrt(64^2+32^2)) = {math.log(math.sqrt(64**2+32**2))/LN_PHI:.1f} levels")
    print(f"{'='*105}")

    # Show filter profile for one variant
    print(f"\n  Spectral filter profile (strength=1.0):")
    k_max = math.sqrt(64**2 + 32**2)
    for k in [1, 2, 4, 8, 16, 32, 64]:
        n = math.log(k_max / k) / LN_PHI
        filt = LN2_SQ ** (XI * n)
        print(f"    |k|={k:3d}  depth={n:4.1f}  filter={filt:.4f}  ({filt*100:.1f}%)")

    all_results = {}
    for name, grav_op in VARIANTS.items():
        t0 = time.time()
        print(f"\n  [{name}] ...", end="", flush=True)
        results = run_variant(name, grav_op, device)
        elapsed = time.time() - t0
        print(f" {elapsed:.0f}s")
        all_results[name] = results

    # --- Coupling errors ---
    for tick_label, tick in [("TICK 1000", 1000), ("TICK 5000", 5000)]:
        print(f"\n{'='*105}")
        print(f"  TIER 1 COUPLING ERRORS AT {tick_label}")
        print(f"{'='*105}")
        header = f"  {'Variant':<22s}"
        for metric in TARGETS:
            header += f" | {metric:>12s}"
        header += " |  avg_err"
        print(header)
        print(f"  {'-'*22}" + ("-+-" + "-"*12) * len(TARGETS) + "-+---------")

        for name, results in all_results.items():
            r = results.get(tick, {})
            if not r:
                continue
            row = f"  {name:<22s}"
            errs = []
            for metric, target in TARGETS.items():
                measured = r.get(metric, 0)
                err = pct_err(measured, target)
                errs.append(err)
                g = grade(err)
                row += f" | {err:>7.1f}% {g:>2s}"
            avg = sum(errs) / len(errs)
            row += f" | {avg:>6.1f}%"
            print(row)

    # --- Mass distribution ---
    print(f"\n{'='*105}")
    print(f"  MASS DISTRIBUTION AT TICK 5000")
    print(f"{'='*105}")
    print(f"  {'Variant':<22s} | {'M_mean':>7s} {'M_std':>7s} {'M_max':>7s} | {'%cap':>6s} {'%empty':>7s} | {'xi_mod':>7s}")
    print(f"  {'-'*22}-+-{'-'*7}-{'-'*7}-{'-'*7}-+-{'-'*6}-{'-'*7}-+-{'-'*7}")

    for name, results in all_results.items():
        r = results.get(5000, {})
        if not r:
            continue
        print(f"  {name:<22s}"
              f" | {r['M_mean']:7.3f} {r['M_std']:7.3f} {r['M_max']:7.3f}"
              f" | {r['frac_cap']:5.1%} {r['frac_empty']:6.1%}"
              f" | {r['xi_mod_mean']:7.4f}")

    # --- Drift ranking ---
    print(f"\n{'='*105}")
    print(f"  DRIFT RANKING (avg Tier 1 error at tick 5000)")
    print(f"{'='*105}")
    scores = {}
    scores_1k = {}
    for name, results in all_results.items():
        for tick, store in [(1000, scores_1k), (5000, scores)]:
            r = results.get(tick, {})
            if r:
                errs = [pct_err(r.get(m, 0), t) for m, t in TARGETS.items()]
                store[name] = sum(errs) / len(errs)

    for name in sorted(scores, key=scores.get):
        e1 = scores_1k.get(name, 0)
        e5 = scores[name]
        drift = e5 - e1
        g_1k = pct_err(all_results[name].get(1000, {}).get("G_local", 0), 1/PHI**2)
        g_5k = pct_err(all_results[name].get(5000, {}).get("G_local", 0), 1/PHI**2)
        print(f"  {name:<22s}  t1k={e1:5.1f}%  t5k={e5:5.1f}%  drift={drift:+5.1f}%"
              f"  G(1k)={g_1k:5.1f}%  G(5k)={g_5k:5.1f}%")

    best = min(scores, key=scores.get)
    print(f"\n  >>> Best overall: {best} ({scores[best]:.1f}% avg error at tick 5000)")
    print()


if __name__ == "__main__":
    main()
