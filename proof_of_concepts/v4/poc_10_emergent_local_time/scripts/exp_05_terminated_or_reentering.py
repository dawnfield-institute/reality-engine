#!/usr/bin/env python3
"""Is this thing finished, or is it mid-transfer? — identity as accumulated history.

Every other instrument here reports a STATE: percolation at t=400, a correlation at one
radius, a class fraction at one scale. A state cannot distinguish "this ended" from "this
moved somewhere I did not measure". That is not an edge case, it is the default failure of
snapshot measurement applied to a process that re-enters itself — and it is why a web
collapsing into clumps reads as decay when it may be the next level starting.

If a thing is the accumulation of its history, the measurement has to be over the TRAJECTORY.
And PAC gives the discriminator: potential is conserved through actualization, so structure
that vanishes at one scale must appear at another. The ledger has to balance across scale.

    TERMINATED   character stops moving AND the cross-scale ledger does not balance.
                 Structure is gone. This verdict must be reachable or the instrument is
                 useless, so it is calibrated against systems built to die (below).

    RE-ENTERING  character keeps moving and loss at one scale is matched by gain at another.
                 The apparent defeat is a transfer.

    SETTLING     character stops moving but the ledger balanced on the way. Ran its course
                 without losing anything — a completed stage, not a failure.

**Calibration is the point of this file.** Three controls with known answers run every time:

    frozen      positions fixed. Nothing can change. MUST read TERMINATED.
    dissolving  the field diffused toward uniform — real destruction of structure.
                MUST read TERMINATED.
    web         the substrate. Verdict is the measurement.

If a control comes back wrong the run says so and the web verdict is not to be trusted.

    python identity.py [--epochs 40 80 140 220 320 440] [--dims 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V4))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from particles import EXP11_TIME, ParticleConfig, ParticleEngine  # noqa: E402
from exp_04_scale_character import CLASSES, classify, smooth  # noqa: E402

OUT = HERE.parent / "results"
STRUCTURED = ("sheet", "filament", "node")      # anything collapsed on >= 1 axis


def character(rho, scales):
    """The (scale x class) character spectrum — one row per smoothing scale."""
    return np.array([[classify(rho, R).get(c, 0.0) for c in CLASSES] for R in scales])


def motion(traj):
    """How fast character is still changing, late vs early.

    `traj` is (epoch, scale, class). Returns (early_rate, late_rate). A process at a fixed
    point has late_rate ~ 0; one still re-entering does not.
    """
    steps = np.abs(np.diff(traj, axis=0)).sum(axis=(1, 2)) / traj.shape[1]
    if len(steps) < 2:
        return float("nan"), float("nan")
    h = max(1, len(steps) // 2)
    return float(steps[:h].mean()), float(steps[h:].mean())


def collapse_order(traj):
    """Mean collapse order at each (epoch, scale): void 0, sheet 1, filament 2, node 3.

    The structured fraction (sheet+filament+node) is nearly 1 - void everywhere, so it barely
    moves and its ledger is noise-dominated — measured, and it failed to separate a real web
    from a dissolving control. Collapse ORDER is the quantity that actually flows: matter
    moving from sheet to filament to node at one scale, while a larger scale takes up the
    sheet stage, IS the transfer, and this counts it.
    """
    return (traj * np.arange(len(CLASSES))).sum(axis=-1)     # (epoch, scale)


def ledger(traj, scales):
    """Is change across scales REDISTRIBUTION or NET LOSS? — the PAC test.

    For each epoch step take the change in collapse order at every scale. Split it into the
    part that cancels across scales (transfer) and the part that does not (net change):

        balance = 1 - |sum of changes| / sum of |changes|

    1.0 = every loss at one scale is matched by a gain at another; the budget only moved.
    0.0 = every scale moved the same way; the budget itself changed.

    Returns NaN when nothing moved at all. That is the honest value: a static system offers
    no evidence of transfer, and the earlier version scoring it 1.0 ("vacuously balanced") is
    exactly why the frozen control was misread as SETTLING.
    """
    C = collapse_order(traj)
    bal = []
    for a, b in zip(C[:-1], C[1:]):
        d = b - a
        tot = float(np.abs(d).sum())
        if tot < 1e-9:
            continue                              # no evidence, not perfect balance
        bal.append(1.0 - abs(float(d.sum())) / tot)
    return float(np.mean(bal)) if bal else float("nan")


def peak_migration(traj, scales):
    """Where the filament peak sits at each epoch — the organizing scale, over time."""
    fi = CLASSES.index("filament")
    return [float(scales[int(np.argmax(traj[t, :, fi]))]) for t in range(traj.shape[0])]


def migration_trend(peaks):
    """Rank correlation of organizing scale with epoch — DIRECTION, not just change.

    The dissolving control's peak bounces 2.4 -> 1.0 -> 6.0 -> 4.6 -> 2.4; that is argmax
    noise on a flattening profile, and treating any change as migration is what made it read
    RE-ENTERING. Systematic outward movement has a trend; noise does not.
    """
    n = len(peaks)
    if n < 3 or len(set(peaks)) == 1:
        return 0.0
    r = np.argsort(np.argsort(np.asarray(peaks, float)))
    t = np.arange(n, dtype=float)
    if r.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(t, r)[0, 1])


def exchange(traj):
    """Collapse order EXCHANGED between scales per step — the discriminating quantity.

    Three candidates were measured before this one was kept, and the two obvious ones failed:

      balance = 1 - |sum d| / sum|d|   web 0.125-0.273 vs dissolving 0.094-0.287. Overlap.
                It folds growth and decay together; both read as "net change".
      net drift (signed)               web +0.021-+0.064 vs dissolving -0.089-+0.021. Overlap.
      late/early exchange ratio        web 0.69-1.28 vs dissolving 0.76-2.38. Overlap.

    What separates is the MAGNITUDE of cross-scale exchange: dissolving moves every scale the
    same direction (little exchange), while a re-entering system moves scales AGAINST each
    other — small scales shedding collapse order as large scales take it up. That is the
    transfer, and this counts it directly.

        frozen 0.000  |  dissolving 0.08-0.15  |  web 0.28-0.34     (3 seeds each, no overlap)
    """
    return float(np.abs(np.diff(collapse_order(traj), axis=0)).sum(axis=1).mean())


def verdict(traj, scales, reference=None, still=0.004, ratio_min=2.0):
    """Classify a trajectory against a MEASURED reference, not a hard-coded threshold.

    `reference` is the exchange rate of the dissolving control, measured in the same run at
    the same n / box / scales. The verdict is a ratio against it, so it travels across
    configurations instead of encoding numbers fitted to one. Same discipline that caught the
    percolation artifact: compare to a system whose answer is known.
    """
    early, late = motion(traj)
    ex = exchange(traj)
    bal = ledger(traj, scales)
    peaks = peak_migration(traj, scales)
    trend = migration_trend(peaks)
    ratio = (ex / reference) if (reference and reference > 1e-9) else float("nan")
    drift = float(np.diff(collapse_order(traj), axis=0).sum(axis=1).mean())

    if ex < 1e-6 and not (late > still):
        v = "TERMINATED"                          # never moved at all — static
    elif np.isnan(ratio) or ratio < ratio_min:
        # Indistinguishable from a system being smoothed out of existence.
        v = "TERMINATED" if late > still else "SETTLING"
    elif drift < -1e-3:
        v = "TERMINATED"                          # active, but collapse order is draining
    else:
        v = "RE-ENTERING"
    return {"verdict": v, "motion_early": early, "motion_late": late,
            "exchange": ex, "exchange_vs_control": ratio, "net_drift": drift,
            "ledger_balance": bal, "peak_scale_by_epoch": peaks,
            "migration_trend": trend}


def evolve_web(a, seed):
    cfg = ParticleConfig(n=a.n, box=a.box, r0=a.r0, g=a.g, sec_balance=a.sec_balance,
                         dims=a.dims, seed=seed, entropy_init=0.1,
                         time_mode="potential", time_viscosity=0.3)
    return ParticleEngine(cfg, pipeline=EXP11_TIME)


def trajectory(kind, a, seed):
    """Character trajectory for one system. Controls have known answers."""
    eng = evolve_web(a, seed)
    frames, done = [], 0

    if kind == "frozen":
        # Positions never change. Whatever it is, it is finished. MUST read TERMINATED.
        rho = eng.density_field(a.res).cpu().numpy().astype(float)
        frames = [character(rho, a.scales) for _ in a.epochs]

    elif kind == "dissolving":
        # Genuine destruction: a web, then smoothed further at every epoch until uniform.
        for _ in range(a.epochs[0]):
            eng.tick()
        rho0 = eng.density_field(a.res).cpu().numpy().astype(float)
        for i, _ in enumerate(a.epochs):
            frames.append(character(smooth(rho0, 0.8 * (i + 1) ** 1.5), a.scales))

    else:                                          # "web" — the real system
        for ep in a.epochs:
            for _ in range(ep - done):
                eng.tick()
            done = ep
            rho = eng.density_field(a.res).cpu().numpy().astype(float)
            frames.append(character(rho, a.scales))

    return np.array(frames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=3, choices=(2, 3))
    ap.add_argument("-n", type=int, default=8000)
    ap.add_argument("--box", type=float, default=60.0)
    ap.add_argument("--r0", type=float, default=10.0)
    ap.add_argument("--g", type=float, default=1.5)
    ap.add_argument("--sec-balance", dest="sec_balance", type=float, default=0.65334)
    ap.add_argument("--epochs", type=int, nargs="+", default=[40, 90, 150, 230, 330, 450])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.6, 2.4, 3.4, 4.6, 6.0])
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    a = ap.parse_args()

    print(f"  {a.n} particles, {a.dims}D, epochs {a.epochs}, scales {a.scales}")
    print(f"\n  {'system':>11} {'seed':>5} | {'motion(early)':>13} {'motion(late)':>12} "
          f"{'ledger':>7} | {'peak scale by epoch':>26} | {'VERDICT':>12}")

    rows, ok = [], True
    expected = {"frozen": "TERMINATED", "dissolving": "TERMINATED", "web": None}

    # Measure the reference FIRST — the dissolving control's exchange rate, at the same n,
    # box and scales. Every verdict is a multiple of it, so nothing is hard-coded.
    trajs = {(k, sd): trajectory(k, a, sd) for k in expected for sd in a.seeds}
    reference = float(np.mean([exchange(trajs[("dissolving", sd)]) for sd in a.seeds]))
    print(f"  reference exchange (dissolving control) = {reference:.4f} — a system must "
          f"exceed 2x this to read RE-ENTERING")

    for kind in ("frozen", "dissolving", "web"):
        for seed in a.seeds:
            v = verdict(trajs[(kind, seed)], a.scales, reference=reference)
            rows.append({"kind": kind, "seed": seed, **v,
                         "trajectory": trajs[(kind, seed)].tolist()})
            flag = ""
            if expected[kind] and v["verdict"] != expected[kind]:
                flag = "  <-- CONTROL FAILED"; ok = False
            print(f"  {kind:>11} {seed:>5} | {v['exchange']:>8.4f} "
                  f"{v['exchange_vs_control']:>7.2f}x {v['net_drift']:>+9.4f} "
                  f"{v['motion_late']:>8.4f} | "
                  + " ".join(f"{p:>3.1f}" for p in v["peak_scale_by_epoch"])
                  + f" | {v['verdict']:>12}{flag}")

    # A calibration that can only ever say PASS guards nothing. The controls must land on
    # TERMINATED *and* the reference must have separated them from the web — an earlier
    # version reported PASS while every verdict was NaN-defaulted to TERMINATED.
    spread = [r["exchange_vs_control"] for r in rows if r["kind"] == "web"]
    separated = bool(spread) and min(spread) > 1.0
    ok = ok and separated
    print(f"\n  CALIBRATION: {'PASS' if ok else 'FAIL'} — "
          + ("controls read TERMINATED and the reference separates them from the web"
             if ok else ("a control misread" if not separated
                         else "web not separated from the dissolving control")
             + "; do NOT trust the web verdict"))

    web = [r for r in rows if r["kind"] == "web"]
    if ok and web:
        vs = {r["verdict"] for r in web}
        print(f"\n  THE WEB: {' / '.join(sorted(vs))}")
        print(f"    A snapshot at the last epoch would report a collapsed, clumped field.")
        print(f"    The trajectory reports whether that is an ending or a handover.")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    fi = CLASSES.index("filament")
    for ax, kind in zip(axes, ("frozen", "dissolving", "web")):
        r = next(x for x in rows if x["kind"] == kind)
        traj = np.array(r["trajectory"])
        for j, R in enumerate(a.scales):
            ax.plot(a.epochs, traj[:, j, fi], marker="o", ms=3,
                    label=f"R={R:g}", alpha=0.85)
        ax.set_title(f"{kind}  —  {r['verdict']}\nledger {r['ledger_balance']:.2f}, "
                     f"late motion {r['motion_late']:.4f}", fontsize=10)
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.set_ylim(0, 0.75)
    axes[0].set_ylabel("filament fraction")
    axes[2].legend(fontsize=8, ncol=2)
    fig.suptitle("ended, or moved? — structure character over history, per scale", fontsize=12)
    fig.tight_layout()
    p = OUT / f"identity_{a.dims}d_{stamp}.png"
    fig.savefig(p, dpi=115, bbox_inches="tight")
    plt.close(fig)
    for r in rows:
        r.pop("trajectory", None)
    (OUT / f"identity_{stamp}.json").write_text(
        json.dumps({"config": vars(a), "calibration_pass": ok, "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"\n  wrote {p.name}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
