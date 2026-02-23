"""
exp_02 — Confluence Conservation.

Verify that the confluence operator:
  1. Preserves L² norm (energy)
  2. Has period 4
  3. Preserves PAC total under repeated cycling
  4. Antiperiodic projection is idempotent
  5. Orthogonal decomposition holds
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dynamics.confluence import ConfluenceOperator
from src.substrate.state import FieldState


def run(device: str = "cpu") -> dict:
    results = {"experiment": "exp_02_confluence_conservation", "tests": []}
    C = ConfluenceOperator()

    torch.manual_seed(42)
    f = torch.randn(128, 64, device=device)

    # Test 1: norm preservation
    norm_before = f.pow(2).sum().item()
    norm_after = C(f).pow(2).sum().item()
    err_norm = abs(norm_after - norm_before)
    results["tests"].append({
        "name": "norm_preservation",
        "passed": err_norm < 1e-10,
        "error": err_norm,
    })

    # Test 2: period 4
    g = f.clone()
    for _ in range(4):
        g = C(g)
    err_period = (g - f).abs().max().item()
    results["tests"].append({
        "name": "period_4",
        "passed": err_period < 1e-12,
        "error": err_period,
    })

    # Test 3: PAC conservation under 1000 confluence cycles
    P = torch.ones(128, 64, device=device) * 0.5 + torch.randn(128, 64, device=device) * 0.01
    A = torch.ones(128, 64, device=device) * 0.5 + torch.randn(128, 64, device=device) * 0.01
    M = torch.zeros(128, 64, device=device)
    pac_initial = (P + A + M).sum().item()

    for _ in range(1000):
        P_new = C(A)
        A = P.clone()
        P = P_new

    pac_final = (P + A + M).sum().item()
    pac_drift = abs(pac_final - pac_initial) / max(abs(pac_initial), 1e-14)
    results["tests"].append({
        "name": "pac_under_confluence_1000_steps",
        "passed": pac_drift < 1e-8,
        "residual": pac_drift,
    })

    # Test 4: idempotent antiperiodic projection
    f_anti = C.project_antiperiodic(f)
    f_anti2 = C.project_antiperiodic(f_anti)
    err_idem = (f_anti - f_anti2).abs().max().item()
    results["tests"].append({
        "name": "antiperiodic_projection_idempotent",
        "passed": err_idem < 1e-12,
        "error": err_idem,
    })

    # Test 5: orthogonal decomposition
    f_sym, f_anti = C.decompose(f)
    norm_total = f.pow(2).sum().item()
    norm_parts = f_sym.pow(2).sum().item() + f_anti.pow(2).sum().item()
    err_orth = abs(norm_total - norm_parts)
    results["tests"].append({
        "name": "orthogonal_decomposition",
        "passed": err_orth < 1e-3,  # float32 precision
        "error": err_orth,
    })

    # Summary
    all_pass = all(t["passed"] for t in results["tests"])
    results["all_passed"] = all_pass
    results["timestamp"] = datetime.now().isoformat()

    for t in results["tests"]:
        status = "✓" if t["passed"] else "✗"
        print(f"  {status} {t['name']}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("EXP 02: Confluence Conservation")
    print("=" * 60)
    results = run()

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"exp_02_confluence_conservation_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
    print("ALL PASSED" if results["all_passed"] else "SOME FAILED")
