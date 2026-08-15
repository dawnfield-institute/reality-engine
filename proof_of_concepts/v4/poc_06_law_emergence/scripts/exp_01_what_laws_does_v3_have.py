#!/usr/bin/env python3
"""What laws does v3 actually have?

The project's stated purpose: "We don't program the laws. We discover what emerges."
(`docs/theory/law_emergence.md`.) The detectors that would answer that were designed, built
for v1, and left in `archive/v1/` — last run November 2025, never against v3.

This asks v3 the question. Nothing here is tuned toward an expected answer; the force-law
fitter reports whatever exponent it finds, including none.

Every result is labelled ENFORCED (an operator maintains it, so finding it proves nothing),
EMERGENT (nothing implements it and it holds anyway), or ABSENT. The v1 detector's single
"discovery" was QBE — hardcoded — reported at correlation exactly -1.0.

    python proof_of_concepts/v4/poc_06_law_emergence/scripts/exp_01_what_laws_does_v3_have.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from law_detector import (Law, conservation_scan, entropy_trend,  # noqa: E402
                          fit_force_law)
from src.v3.engine.config import SimulationConfig  # noqa: E402
from src.v3.engine.engine import Engine  # noqa: E402
from src.v3.engine.pipelines import build_canonical_pipeline  # noqa: E402


def candidate_quantities(s) -> dict[str, float]:
    """Quantities to watch. Deliberately wider than the ones anything maintains."""
    E, I, M = s.E, s.I, s.M
    P = s.P if s.P is not None else torch.zeros_like(M)
    f = lambda t: float(t.sum().item())  # noqa: E731
    return {
        "E+I+M": f(E + I + M),                       # ENFORCED — the control
        "E+I+M+P": f(E + I + M + P),                 # the full ledger with the Delta term
        "E": f(E), "I": f(I), "M": f(M), "P": f(P),
        "E^2+I^2": f(E.pow(2) + I.pow(2)),
        "E^3+I^3+M^3": f(E.pow(3) + I.pow(3) + M.pow(3)),
        "E*I": f(E * I),
        "|E-I|": f((E - I).abs()),
        "E-I": f(E - I),
        "M^2": f(M.pow(2)),
        "grad_E^2": f((torch.roll(E, -1, 0) - E).pow(2) + (torch.roll(E, -1, 1) - E).pow(2)),
        # A momentum-like object: mass times the local disequilibrium gradient, summed.
        # Nothing in the engine maintains this; the question is whether anything does.
        "M*grad(E-I)_u": f(M * (torch.roll(E - I, -1, 0) - torch.roll(E - I, 1, 0)) / 2),
        "M*grad(E-I)_v": f(M * (torch.roll(E - I, -1, 1) - torch.roll(E - I, 1, 1)) / 2),
    }


def find_peaks(M: np.ndarray, k: int = 6, min_frac: float = 2.0):
    """Local maxima above `min_frac` x mean — the engine's candidate 'particles'."""
    up = M > np.roll(M, 1, 0); dn = M > np.roll(M, -1, 0)
    lf = M > np.roll(M, 1, 1); rt = M > np.roll(M, -1, 1)
    peak = up & dn & lf & rt & (M > min_frac * M.mean())
    idx = np.argwhere(peak)
    if idx.size == 0:
        return np.empty((0, 2))
    order = np.argsort(-M[peak])
    return idx[order][:k].astype(float)


def track_peaks(frames, max_move: float = 3.0, min_len: int = 40):
    """Link peaks across frames by proximity. Returns tracks as (T, 2) arrays.

    `max_move` is deliberately generous. If nothing survives `min_len` frames, that is the
    result: the engine has no persistent objects to have a force law between.
    """
    tracks = []
    for p in frames[0]:
        tracks.append([p])
    for frame in frames[1:]:
        used = set()
        for tr in tracks:
            if tr[-1] is None:
                tr.append(None)
                continue
            last = tr[-1]
            best, bd = None, max_move
            for j, q in enumerate(frame):
                if j in used:
                    continue
                dv = np.abs(q - last)
                dv[0] = min(dv[0], frame.shape[0] if False else abs(dv[0]))
                dist = float(np.hypot(*dv))
                if dist < bd:
                    best, bd = j, dist
            if best is None:
                tr.append(None)
            else:
                used.add(best)
                tr.append(frame[best])
    out = []
    for tr in tracks:
        run = []
        for p in tr:
            if p is None:
                break
            run.append(p)
        if len(run) >= min_len:
            out.append(np.array(run))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=12000)
    ap.add_argument("--grid", type=int, nargs=2, default=[128, 128])
    ap.add_argument("--eta", type=float, default=0.025)
    args = ap.parse_args()

    torch.manual_seed(42)
    eng = Engine(config=SimulationConfig(nu=args.grid[0], nv=args.grid[1],
                                         noise_scale=0.0,
                                         deactualization_rate=args.eta),
                 pipeline=build_canonical_pipeline())
    eng.initialize("big_bang", temperature=3.0)

    history: dict[str, list[float]] = {}
    entropy: list[float] = []
    peak_frames = []
    dts = []

    print(f"  v3 canonical, {args.grid[0]}x{args.grid[1]}, {args.ticks} ticks, "
          f"eta={args.eta}\n  running...")
    for t in range(1, args.ticks + 1):
        eng.tick()
        s = eng.state
        for k, v in candidate_quantities(s).items():
            history.setdefault(k, []).append(v)
        # SECTrackingOperator exports "field_entropy"; "mass_entropy" is the M-field one.
        ent = s.metrics.get("field_entropy")
        if isinstance(ent, (int, float)):
            entropy.append(float(ent))
        ment = s.metrics.get("mass_entropy")
        if isinstance(ment, (int, float)):
            history.setdefault("mass_entropy", []).append(float(ment))
        dts.append(float(eng.config.dt))
        if t % 20 == 0:                      # sample peaks every 20 ticks
            peak_frames.append(find_peaks(s.M.detach().cpu().numpy().astype(float)))

    laws: list[Law] = conservation_scan(history, tol=1e-3)

    # --- force law between tracked mass concentrations ---
    tracks = track_peaks(peak_frames)
    dt_mean = float(np.mean(dts)) * 20.0
    if len(tracks) >= 2:
        a, b = tracks[0], tracks[1]
        n = min(len(a), len(b))
        pair = np.stack([a[:n], b[:n]], axis=1)
        from law_detector import two_body_tracks
        r, acc = two_body_tracks(pair, dt=dt_mean)
        expo, r2, npts = fit_force_law(r, np.abs(acc))
        if npts >= 12 and r2 == r2 and r2 > 0.5:
            laws.append(Law("force(r)", "force", "EMERGENT", expo,
                            f"F ~ r^{expo:.3f}, R2={r2:.3f}, {npts} points"))
        else:
            laws.append(Law("force(r)", "force", "ABSENT", float(r2 if r2 == r2 else 0),
                            f"{npts} usable points, no fit"))
    else:
        laws.append(Law("force(r)", "force", "ABSENT", float(len(tracks)),
                        f"only {len(tracks)} peaks survived 40 samples — "
                        "no persistent objects to have a force between"))

    # --- second law ---
    if len(entropy) > 10:
        slope, viol = entropy_trend(entropy)
        laws.append(Law("2nd law (dS/dt>0)", "thermodynamic",
                        "EMERGENT" if slope > 0 and viol < 0.05 else "ABSENT",
                        slope, f"violations {viol:.1%}"))
    else:
        laws.append(Law("2nd law (dS/dt>0)", "thermodynamic", "ABSENT", float("nan"),
                        "no entropy metric exported"))

    print(f"\n  {'':<2}{'law':<26} {'kind':<14} {'verdict':<9} {'statistic':>10}  detail")
    print("  " + "-" * 100)
    for l in sorted(laws, key=lambda x: {"EMERGENT": 0, "ENFORCED": 1, "ABSENT": 2}[x.verdict]):
        print(l)

    n_em = sum(1 for l in laws if l.verdict == "EMERGENT")
    n_en = sum(1 for l in laws if l.verdict == "ENFORCED")
    print(f"\n  {n_em} EMERGENT, {n_en} ENFORCED, {len(laws)-n_em-n_en} ABSENT")
    print(f"  peaks tracked: {len(tracks)}   mean dt: {np.mean(dts):.2e}")

    outdir = Path(__file__).resolve().parents[1] / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (outdir / f"exp_01_laws_{stamp}.json").write_text(json.dumps({
        "config": {"grid": args.grid, "ticks": args.ticks, "eta": args.eta},
        "laws": [{"name": l.name, "kind": l.kind, "verdict": l.verdict,
                  "statistic": l.statistic, "detail": l.detail} for l in laws],
        "n_tracks": len(tracks), "mean_dt": float(np.mean(dts)),
    }, indent=2), encoding="utf-8")
    print(f"  wrote results/exp_01_laws_{stamp}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
