"""Spike: non-local Landauer reinjection.

Problem: 10% of cells at mass cap, 37% nearly empty.  Dense cells hit
the cap, Landauer dumps energy back into E+I AT THE SAME CELL, the cell
re-generates mass, hits cap again -- a dead cycle.  Sparse cells can't
generate mass because gamma_local ~ 0 when E and I are both small.

Fix: when Landauer removes mass at the cap, redistribute the released
energy to LOW-DENSITY regions instead of locally.  This feeds sparse
cells with E+I energy, enabling mass generation there.  With the tiling
filter creating local-dominant gravity, these new mass seeds attract
neighbors -> filaments form -> web structure.

Physical analogy: supernovae enrich the interstellar medium, seeding
next-generation star formation far from the explosion site.

Variants:
  A. Baseline (local reinjection)
  B. Fully non-local (1/M weighted)
  C. 50/50 split (half local, half non-local)
  D. Diffusive (spread to neighbors of capped cells via Laplacian)
  E. Disequilibrium-seeded (reinject where |E-I| gradient is highest)
"""

import math
import os
import sys
import time
from typing import Optional

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
from src.v3.substrate.manifold import MobiusManifold

try:
    from fracton.physics import PACValidator as _FractonPACValidator
    _HAS_PAC_VALIDATOR = True
except ImportError:
    _HAS_PAC_VALIDATOR = False

PHI = (1 + math.sqrt(5)) / 2
LN2 = math.log(2)

TARGETS = {
    "f_local":     0.5772,
    "gamma_local": 1 / PHI,
    "alpha_local": LN2,
    "G_local":     1 / PHI**2,
    "lambda_local": 1 - LN2,
}


# ---------------------------------------------------------------
# Variant normalizers
# ---------------------------------------------------------------

class _BaseNonLocalNorm:
    """Base class with shared PAC machinery.  Subclasses override _distribute."""

    def __init__(self):
        self._initial_pac = None
        self._pac_validator = (
            _FractonPACValidator(tolerance=1e-10, auto_correct=False)
            if _HAS_PAC_VALIDATOR else None
        )

    @property
    def name(self):
        return "normalization"

    def _distribute(self, dM_removed, M_new, state, config):
        """Return (dE, dI) tensors for the Landauer reinjection."""
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        if not config.enable_normalization:
            return state

        s = config.field_scale
        M_cap = s / 5.0

        M_floored = torch.clamp(state.M, min=0.0)
        M_capped = torch.clamp(M_floored, max=M_cap)
        dM_removed = M_floored - M_capped  # >= 0

        # Subclass decides WHERE the energy goes
        dE, dI = self._distribute(dM_removed, M_capped, state, config)

        E_cur = state.E + dE
        I_cur = state.I + dI
        M_new = M_capped

        # --- Tanh safety clamp + QBE cross-injection (same as original) ---
        E_clamped = s * torch.tanh(E_cur / s)
        I_clamped = s * torch.tanh(I_cur / s)
        E_loss = E_cur - E_clamped
        I_loss = I_cur - I_clamped

        E_inject = torch.clamp(I_loss, min=-s - E_clamped, max=s - E_clamped)
        I_inject = torch.clamp(E_loss, min=-s - I_clamped, max=s - I_clamped)
        E_new = E_clamped + E_inject
        I_new = I_clamped + I_inject

        E_remainder = I_loss - E_inject
        I_remainder = E_loss - I_inject
        crystallised = E_remainder + I_remainder
        M_new = M_new + crystallised
        M_new = torch.clamp(M_new, min=0.0)

        # Global PAC safety
        if self._initial_pac is None:
            self._initial_pac = (E_new + I_new + M_new).sum().item()
        current_pac = (E_new + I_new + M_new).sum().item()
        pac_residual = self._initial_pac - current_pac
        if abs(pac_residual) > 1e-8:
            correction = pac_residual / (2.0 * E_new.numel())
            E_new = E_new + correction
            I_new = I_new + correction

        metrics = dict(state.metrics)
        metrics["landauer_reinjection"] = dM_removed.sum().item()
        metrics["crystallisation"] = crystallised.sum().item()
        metrics["pac_correction"] = pac_residual

        if self._pac_validator is not None:
            E_sum, I_sum, M_sum = E_new.sum().item(), I_new.sum().item(), M_new.sum().item()
            result = self._pac_validator.validate(E_sum + I_sum + M_sum, [E_sum, I_sum, M_sum])
            metrics["pac_validator_residual"] = result.residual
            metrics["pac_validator_violations"] = self._pac_validator.stats["violations"]

        if dM_removed.sum().item() > 0.01 and bus is not None:
            bus.emit("landauer_reinjection", {
                "energy_reinjected": dM_removed.sum().item(),
                "cells_affected": (dM_removed > 1e-6).sum().item(),
            })

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


class LocalNorm(_BaseNonLocalNorm):
    """A: Original local reinjection."""
    def _distribute(self, dM_removed, M_new, state, config):
        r = dM_removed * 0.5
        return r, r


class FullyNonLocalNorm(_BaseNonLocalNorm):
    """B: All removed mass -> sparse regions (1/M weighted)."""
    def _distribute(self, dM_removed, M_new, state, config):
        total = dM_removed.sum()
        if total < 1e-10:
            z = torch.zeros_like(dM_removed)
            return z, z
        inv_M = 1.0 / (M_new + 0.1)
        weights = inv_M / inv_M.sum()
        r = total * weights * 0.5
        return r, r


class HybridNorm(_BaseNonLocalNorm):
    """C: 50% local, 50% non-local."""
    def _distribute(self, dM_removed, M_new, state, config):
        local = dM_removed * 0.25  # half of 0.5
        total_nonlocal = dM_removed.sum() * 0.5  # other half
        if total_nonlocal < 1e-10:
            return local, local
        inv_M = 1.0 / (M_new + 0.1)
        weights = inv_M / inv_M.sum()
        spread = total_nonlocal * weights * 0.5
        return local + spread, local + spread


class DiffusiveNorm(_BaseNonLocalNorm):
    """D: Spread removed energy to neighbors via Laplacian smoothing."""
    def _distribute(self, dM_removed, M_new, state, config):
        # Start with local reinjection
        r = dM_removed * 0.5
        # Then diffuse: replace each cell with average of neighbors (3 iterations)
        for _ in range(3):
            r = (torch.roll(r, 1, 0) + torch.roll(r, -1, 0) +
                 torch.roll(r, 1, 1) + torch.roll(r, -1, 1)) / 4.0
        return r, r


class DisequilibriumNorm(_BaseNonLocalNorm):
    """E: Reinject where |grad(E-I)| is highest (active boundaries)."""
    def _distribute(self, dM_removed, M_new, state, config):
        total = dM_removed.sum()
        if total < 1e-10:
            z = torch.zeros_like(dM_removed)
            return z, z
        # Gradient magnitude of disequilibrium field
        diseq = state.E - state.I
        grad_u = (torch.roll(diseq, -1, 0) - torch.roll(diseq, 1, 0)) / 2.0
        grad_v = (torch.roll(diseq, -1, 1) - torch.roll(diseq, 1, 1)) / 2.0
        grad_mag = torch.sqrt(grad_u**2 + grad_v**2 + 1e-12)
        weights = grad_mag / grad_mag.sum()
        r = total * weights * 0.5
        return r, r


# ---------------------------------------------------------------
# Harness
# ---------------------------------------------------------------

def build_pipeline(norm_op):
    return Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
        GravitationalCollapseOperator(),
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        norm_op, SECTrackingOperator(),
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


def run_variant(name, norm_op, device, ticks=5000):
    config = SimulationConfig(
        nu=128, nv=64, dt=0.001, device=device,
        enable_actualization=True, actualization_threshold=0.05,
    )
    torch.manual_seed(42)
    pipeline = build_pipeline(norm_op)
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
                "frac_cap": (M > M_cap * 0.9).float().mean().item(),
                "frac_empty": (M < 0.1).float().mean().item(),
                "landauer": met.get("landauer_reinjection", 0),
            }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VARIANTS = {
        "A_local":          LocalNorm(),
        "B_nonlocal_full":  FullyNonLocalNorm(),
        "C_hybrid_50_50":   HybridNorm(),
        "D_diffusive":      DiffusiveNorm(),
        "E_disequilibrium": DisequilibriumNorm(),
    }

    print(f"\n{'='*105}")
    print(f"  NON-LOCAL LANDAUER REINJECTION SPIKE")
    print(f"  Device: {device} | Grid: 128x64 | 5000 ticks per variant")
    print(f"  (Uses current gravity operator with xi_mod + sqrt + tiling filter)")
    print(f"{'='*105}")

    all_results = {}
    for name, norm_op in VARIANTS.items():
        t0 = time.time()
        print(f"\n  [{name}] ...", end="", flush=True)
        try:
            results = run_variant(name, norm_op, device)
            elapsed = time.time() - t0
            print(f" {elapsed:.0f}s")
            all_results[name] = results
        except Exception as e:
            print(f" FAILED: {e}")
            import traceback; traceback.print_exc()

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
    print(f"  {'Variant':<22s} | {'M_mean':>7s} {'M_std':>7s} {'%cap':>6s} {'%empty':>7s} | {'Landauer':>10s}")
    print(f"  {'-'*22}-+-{'-'*7}-{'-'*7}-{'-'*6}-{'-'*7}-+-{'-'*10}")

    for name, results in all_results.items():
        r = results.get(5000, {})
        if not r:
            continue
        print(f"  {name:<22s}"
              f" | {r['M_mean']:7.3f} {r['M_std']:7.3f} {r['frac_cap']:5.1%} {r['frac_empty']:6.1%}"
              f" | {r['landauer']:10.4f}")

    # --- Drift ranking ---
    print(f"\n{'='*105}")
    print(f"  DRIFT RANKING")
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
        g_5k = pct_err(all_results[name].get(5000, {}).get("G_local", 0), 1/PHI**2)
        cap = all_results[name].get(5000, {}).get("frac_cap", 0)
        empty = all_results[name].get(5000, {}).get("frac_empty", 0)
        print(f"  {name:<22s}  t1k={e1:5.1f}%  t5k={e5:5.1f}%  drift={drift:+5.1f}%"
              f"  G={g_5k:5.1f}%  cap={cap:4.1%}  empty={empty:4.1%}")

    best = min(scores, key=scores.get)
    print(f"\n  >>> Best overall: {best} ({scores[best]:.1f}% avg error at tick 5000)")
    print()


if __name__ == "__main__":
    main()
