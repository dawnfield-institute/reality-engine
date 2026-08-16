#!/usr/bin/env python3
"""POC-03 (v4) exp_01 — does local durability predict global conservation?

Both quantities come from the SAME runs, so they cannot differ by run conditions:

    local R       info_fraction recovery ratio under a ledger-preserving impulse
    global drift  |Q(end) - Q(start)| / |Q(start)| / elapsed time, on the UNPERTURBED
                  reference twin, with enforce_pac = False

Registered outcome is the Spearman rank correlation between them. See README.md.

    python proof_of_concepts/v4/poc_03_local_enforces_global/scripts/exp_01_local_global_correlation.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))

from harness import durability, one_at_a_time  # noqa: E402

BASE = {
    "nu": 32, "nv": 16, "dt": 1e-3, "enforce_pac": False, "noise_scale": 0.0,
    "quantum_pressure_coeff": 0.020,
    "deactualization_rate": 0.025,
    "confluence_weight": 0.3,
    "mass_diffusion_coeff": 0.0005,
    "gamma_damping": 0.01,
}

# mass_gen_coeff is NOT swept: it is declared in config.py and read by ZERO operators.
# memory.py computes gamma_local = diseq2/total_field2 directly, with no coefficient.
# Varying it produced bit-identical runs — verified: base, x0.5 and x2.0 gave the same
# drift to 9 decimals at both seeds. Swept parameters are audited as live before use.
FACTORS = {
    "quantum_pressure_coeff": [0.5, 2.0],
    "deactualization_rate":   [0.5, 2.0],
    "confluence_weight":      [0.5, 2.0],
    "mass_diffusion_coeff":   [0.5, 2.0],
    "gamma_damping":          [0.5, 2.0],
}


def _info_fraction(e) -> float:
    st = e.state
    i = st.I.abs().sum().item()
    return i / (st.E.abs().sum().item() + i + 1e-30)


def _ledger(e) -> float:
    st = e.state
    return (st.E + st.I + st.M).sum().item()


OBSERVABLES = {
    "info_fraction": _info_fraction,
    "ledger": _ledger,
    "max_disequilibrium": lambda e: e.state.metrics.get("max_disequilibrium", float("nan")),
    "balance_magnitude": lambda e: e.state.metrics.get("balance_magnitude", float("nan")),
}


def balance_impulse(frac: float):
    """Ledger-preserving: E += d, I -= d."""
    def perturb(eng) -> None:
        st = eng.state
        d = frac * st.E.abs().mean().item()
        eng.state = st.replace(E=st.E + d, I=st.I - d)
    return perturb


def spearman(xs, ys):
    """Rank correlation, with a manual fallback so scipy is not a hard dependency."""
    try:
        from scipy.stats import spearmanr
        r = spearmanr(xs, ys)
        return float(r.statistic), float(r.pvalue)
    except Exception:
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0.0] * len(v)
            for pos, i in enumerate(order):
                rk[i] = float(pos)
            return rk
        rx, ry = rank(xs), rank(ys)
        n = len(xs)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return (num / den if den else float("nan")), float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=int, default=200)
    ap.add_argument("--observe", type=int, default=400)
    ap.add_argument("--impulse", type=float, default=0.10)
    # Seed dominates: the first run showed drift varying 36x by seed against ~40% by
    # parameter. Pooling across few seeds produced a spurious rho of -0.43 that was
    # entirely between-cluster. Default raised, and rho is reported WITHIN seed.
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    rows, t0 = [], time.time()
    print(f"{'varied':>24} {'x':>5} {'seed':>5} {'local R':>10} {'global drift':>14} {'vigor':>10}")
    for cfg in one_at_a_time(BASE, FACTORS):
        varied, factor = cfg.pop("_varied"), cfg.pop("_factor")
        for s in range(args.seeds):
            res = durability(OBSERVABLES, balance_impulse(args.impulse),
                             seed=4000 + s, settle=args.settle,
                             observe=args.observe, **cfg)

            local_R = res["info_fraction"].R

            # Global drift from the REFERENCE twin only — the impulse never touches it.
            q = res["ledger"].reference_trajectory
            elapsed = (len(q) - 1) * cfg["dt"]
            global_drift = (abs(q[-1] - q[0]) / abs(q[0]) / elapsed) if q[0] and elapsed else float("nan")

            vigor = res["max_disequilibrium"].reference_trajectory
            vigor_mean = sum(vigor) / len(vigor) if vigor and vigor[0] == vigor[0] else float("nan")

            rows.append({"varied": varied, "factor": factor, "seed": 4000 + s,
                         "local_R": local_R, "global_drift": global_drift,
                         "vigor_max_diseq_mean": vigor_mean,
                         "pre_impulse_gap": res["info_fraction"].pre_impulse_gap})
            print(f"{varied:>24} {factor:>5.1f} {4000+s:>5} {local_R:>10.4f} "
                  f"{global_drift:>14.6e} {vigor_mean:>10.4f}")
        cfg["_varied"], cfg["_factor"] = varied, factor

    good = [r for r in rows if r["local_R"] == r["local_R"]
            and r["global_drift"] == r["global_drift"]]
    rho_pooled, p_pooled = spearman([r["local_R"] for r in good],
                                    [r["global_drift"] for r in good])

    # WITHIN-SEED is the registered quantity. Seed changes global drift by ~36x while
    # parameters change it by ~40%, so a pooled correlation measures which seed a point
    # came from, not whether local durability predicts global conservation. Reporting the
    # pooled value alone would have been Simpson's paradox.
    per_seed = {}
    for sd in sorted({r["seed"] for r in good}):
        sub = [r for r in good if r["seed"] == sd]
        if len(sub) >= 4:
            rr, _ = spearman([r["local_R"] for r in sub], [r["global_drift"] for r in sub])
            per_seed[sd] = rr
    rho = statistics.median(per_seed.values()) if per_seed else float("nan")
    p = float("nan")

    if rho > 0.6:
        verdict = f"SUPPORTED — local durability predicts global conservation (rho={rho:.3f})"
    elif rho > 0.3:
        verdict = f"WEAK — rho={rho:.3f}, in the registered weak band"
    else:
        verdict = (f"NO SUPPORT — kill sentence fired (rho={rho:.3f}). Local durability does "
                   f"not predict global conservation; the two are independent here.")

    # The confound, inspected but NOT registered as a control.
    rho_vg, _ = spearman([r["vigor_max_diseq_mean"] for r in good],
                         [r["global_drift"] for r in good])
    rho_vl, _ = spearman([r["vigor_max_diseq_mean"] for r in good],
                         [r["local_R"] for r in good])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).resolve().parents[1] / "results" / f"exp_01_local_global_correlation_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "poc_03_local_enforces_global / exp_01",
        "registered": "README.md (2026-08-11), pre-registered",
        "measured": "Spearman rho between local balance recovery ratio and global ledger "
                    "drift rate, from the same runs",
        "n": len(good), "rho_within_seed_median": rho, "rho_pooled": rho_pooled,
        "rho_per_seed": per_seed, "p_value": p, "verdict": verdict,
        "note": "within-seed rho is the registered quantity; pooled rho is reported only "
                "to show the between-cluster artifact it produces",
        "confound_vigor_vs_global_drift": rho_vg,
        "confound_vigor_vs_local_R": rho_vl,
        "rows": rows,
        "wall_seconds": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")

    print()
    print(f"  n = {len(good)}   WITHIN-SEED median rho = {rho:+.4f}")
    print(f"  per-seed rho: {[f'{v:+.3f}' for v in per_seed.values()]}")
    print(f"  pooled rho (artifact, not registered) = {rho_pooled:+.4f}")
    print(f"  confound check — vigor vs global drift: rho={rho_vg:.3f}; "
          f"vigor vs local R: rho={rho_vl:.3f}")
    print(f"  VERDICT: {verdict}")
    print(f"  wrote {out.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
