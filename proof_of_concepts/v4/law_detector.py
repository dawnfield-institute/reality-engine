"""Law detection for v3 — what laws does this engine actually have?

`docs/theory/law_emergence.md` states the project's purpose plainly: "We don't program the
laws. We discover what emerges." The detection architecture is marked designed, the detectors
marked in-progress, and none of it reached v3 — `law_quantifier.py`, `spikes/law_discovery/`
and `analyzers/laws` all sit in `archive/v1/`, unrun since November 2025.

Its last verdict, on v1 at 1000 steps:

    conservation_laws: complementarity — dA/dt . dP/dt < 0, correlation -1.0
    thermodynamic_laws: {}   symmetries: {}   force_laws: {}
    "Discovered 1 laws: 1 match known physics, 0 are novel, 6 expected laws not found"

**That one law is QBE.** `dI/dt = -dE/dt` is hardcoded in `QBEOperator`, which is why the
correlation is exactly -1.0 rather than approximately. The detector rediscovered a law that
was programmed in and reported it as a discovery.

That failure mode is the thing this module is built to avoid. A law detector run on a system
whose laws are implemented will rediscover the implementation. So every result here is
labelled:

    ENFORCED     an operator explicitly maintains it — finding it proves nothing
    EMERGENT     nothing implements it, and it holds anyway — this is the interesting case
    ABSENT       expected and not found

Written fresh rather than ported: `archive/` is read-only lineage, and a one-for-one port of
era-1 code into a different substrate is a mistake this work has already made once.

Calibrate before trusting a null (`python law_detector.py`). A detector that finds nothing may
be a weak detector rather than a lawless engine, and four metrics in the neighbouring POC
turned out to be broken in exactly that way.

**KNOWN LIMITATION, measured — `fit_force_law` does not work in a many-body system.**

It recovers r^-2, r^-3 and r^-1 exactly (R2 = 1.0000) from a clean two-body orbit. Run against
4000 interacting particles whose force law is *known by construction* — POC-07's substrate,
built with F ~ exp(-r/r0)/r — it returns r^0.03 at **R2 = 0.0005**, over 14000 samples, twice,
at two different run lengths. It finds nothing where something certainly is.

The reason is structural rather than a bug: in a dense system each particle's acceleration is
the sum over many neighbours, so projecting onto the nearest neighbour's direction measures
mostly the other ones. A two-body fit needs two bodies.

What this costs: the "0 EMERGENT, 1 ENFORCED" verdict on the v3 field engine holds for the
**conservation** scan — that arm is calibrated, and it correctly finds mass conserved in the
particle substrate and nothing conserved in the field one. It does **not** hold for the force
arm. `force(r): ABSENT` on the field engine was reported as a finding and is really an
instrument limit, and the field engine's own reason for having no force law (no persistent
peaks at all) is established independently, not by this fitter.

A many-body force fitter needs a different method — direct summation against a candidate
kernel, or measuring the pair correlation function — and is not attempted here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field

import numpy as np

# Laws this engine IMPLEMENTS. Finding one of these is not a discovery — it is the detector
# reading back the source. Kept explicit rather than inferred, so the list can be argued with.
ENFORCED_BY_CONSTRUCTION = {
    "E+I+M": "NormalizationOperator applies an explicit global correction every tick "
             "(~99.8% of ticks) to hold this constant. It is enforced, not observed.",
    "dI = -dE": "QBEOperator sets dI_dt = -dE_dt directly. This is the law the v1 detector "
                "reported as its single discovery, at correlation exactly -1.0.",
}


@dataclass
class Law:
    name: str
    kind: str                      # conservation | force | symmetry | thermodynamic
    verdict: str                   # ENFORCED | EMERGENT | ABSENT
    statistic: float
    detail: str = ""
    params: dict = _field(default_factory=dict)

    def __str__(self) -> str:
        mark = {"EMERGENT": "*", "ENFORCED": "=", "ABSENT": " "}[self.verdict]
        return (f"  {mark} {self.name:<26} {self.kind:<14} {self.verdict:<9} "
                f"{self.statistic:>10.4g}  {self.detail}")


# ======================================================================================
# Conservation
# ======================================================================================

def conservation_scan(history: dict[str, list[float]], tol: float = 1e-3,
                      scales: dict[str, float] | None = None) -> list[Law]:
    """Which tracked quantities hold steady?

    Reports variation over the second half of the run — the second half, because the first
    is transient and a quantity still settling looks non-conserved. Conserved when the
    normalised variation is below `tol`.

    `scales` matters more than it looks. Dividing by the mean fails for any quantity whose
    mean is legitimately zero: total momentum in a symmetric two-body system is ~0 by
    construction, so std/|mean| is infinite and perfectly conserved momentum reads as
    "drifts". Pass the natural magnitude of the quantity instead — for momentum, the typical
    |m*v| of a single body. Without it, this detector would miss exactly the conservation
    laws that hold most exactly.
    """
    scales = scales or {}
    out = []
    for name, series in history.items():
        v = np.asarray(series, float)
        v = v[len(v) // 2:]
        if v.size < 4 or not np.isfinite(v).all():
            out.append(Law(name, "conservation", "ABSENT", float("nan"), "non-finite"))
            continue
        denom = abs(scales.get(name, v.mean()))
        cv = float(v.std() / denom) if denom > 1e-30 else float("inf")
        if cv < tol:
            enforced = name in ENFORCED_BY_CONSTRUCTION
            out.append(Law(
                name, "conservation", "ENFORCED" if enforced else "EMERGENT", cv,
                ENFORCED_BY_CONSTRUCTION.get(name, "conserved, nothing implements it")))
        else:
            out.append(Law(name, "conservation", "ABSENT", cv, "drifts"))
    return out


# ======================================================================================
# Force law
# ======================================================================================

def fit_force_law(sep: np.ndarray, acc: np.ndarray, min_points: int = 12):
    """Fit |a| = A r^n and return (n, r_squared, n_points).

    Reports whatever exponent appears. It is not told what to look for — the point of the
    exercise is that -2 is a possible answer rather than the expected one.
    """
    sep = np.asarray(sep, float)
    acc = np.asarray(acc, float)
    ok = np.isfinite(sep) & np.isfinite(acc) & (sep > 0) & (acc > 0)
    if ok.sum() < min_points:
        return float("nan"), float("nan"), int(ok.sum())
    x, y = np.log(sep[ok]), np.log(acc[ok])
    n, c = np.polyfit(x, y, 1)
    pred = n * x + c
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(n), float(r2), int(ok.sum())


def two_body_tracks(pos: np.ndarray, dt: float):
    """Separation and relative acceleration from a tracked pair.

    `pos` is (T, 2, D). Acceleration by centred second difference, projected onto the
    separation direction so a tangential drift does not read as attraction.
    """
    rel = pos[:, 1] - pos[:, 0]
    r = np.linalg.norm(rel, axis=-1)
    a = (pos[2:, 0] - 2 * pos[1:-1, 0] + pos[:-2, 0]) / dt ** 2
    rel_mid = rel[1:-1]
    unit = rel_mid / np.linalg.norm(rel_mid, axis=-1, keepdims=True)
    a_radial = (a * unit).sum(axis=-1)          # positive = accelerating toward the other
    return r[1:-1], a_radial


# ======================================================================================
# Symmetry
# ======================================================================================

def test_symmetry(evolve, transform, state0, steps: int = 100) -> float:
    """Does the dynamics commute with a transformation?

    Returns relative mismatch between evolve(transform(x)) and transform(evolve(x)).
    Near zero means the symmetry holds.
    """
    a = evolve(transform(state0), steps)
    b = transform(evolve(state0, steps))
    denom = np.abs(b).mean()
    return float(np.abs(a - b).mean() / denom) if denom > 0 else float("nan")


# ======================================================================================
# Thermodynamic
# ======================================================================================

def entropy_trend(series) -> tuple[float, float]:
    """Slope of entropy vs time, and the fraction of steps where it decreased.

    The second law is a statement about the fraction, not the slope: a monotone increase
    means violations ~0.
    """
    v = np.asarray(series, float)
    if v.size < 4 or not np.isfinite(v).all():
        return float("nan"), float("nan")
    t = np.arange(v.size, dtype=float)
    slope = float(np.polyfit(t, v, 1)[0])
    d = np.diff(v)
    return slope, float((d < 0).mean())


# ======================================================================================
# Calibration — systems whose laws are known
# ======================================================================================

def nbody_two(exponent: float = -2.0, steps: int = 4000, dt: float = 1e-3,
              G: float = 1.0, seed: int = 0):
    """Two bodies under F = G m1 m2 r^exponent. Leapfrog, so energy is well behaved.

    The calibration case. If `fit_force_law` cannot recover the exponent of a system built
    with a known one, a null on the engine means nothing.
    """
    rng = np.random.default_rng(seed)
    m = np.array([1.0, 1.0])
    pos = np.array([[-1.0, 0.0], [1.0, 0.0]])
    vel = np.array([[0.0, -0.35], [0.0, 0.35]])
    traj, energy, momentum = [], [], []

    def accel(p):
        d = p[1] - p[0]
        r = np.linalg.norm(d)
        u = d / r
        f = G * m[0] * m[1] * r ** exponent
        return np.stack([f * u / m[0], -f * u / m[1]])

    a = accel(pos)
    for _ in range(steps):
        vel = vel + 0.5 * dt * a
        pos = pos + dt * vel
        a = accel(pos)
        vel = vel + 0.5 * dt * a
        traj.append(pos.copy())
        r = np.linalg.norm(pos[1] - pos[0])
        # F_r = -dU/dr with F attractive, so U = +G m1 m2 r^(n+1)/(n+1). Checked at n = -2:
        # U = G/(-1) * 1/r = -G/r, which is the sign Newtonian gravity should have. The
        # first version had this negated and total energy read as non-conserved.
        pot = (G * m[0] * m[1] * math.log(r) if abs(exponent + 1) < 1e-12
               else G * m[0] * m[1] * r ** (exponent + 1) / (exponent + 1))
        energy.append(0.5 * (m[:, None] * vel ** 2).sum() + pot)
        momentum.append(float((m[:, None] * vel).sum(axis=0)[1]))
    # Momentum's mean is ~0 by construction here, so it needs an explicit scale.
    scales = {"momentum_y": float(np.abs(m[:, None] * vel).max())}
    return np.array(traj), {"total_energy": energy, "momentum_y": momentum}, scales


def pure_diffusion(steps: int = 2000, n: int = 64, seed: int = 0):
    """Diffusion on a ring: conserves the total, entropy rises, and there is NO force law.

    The null control. A detector that reports a force law here is finding one in noise.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    total, entropy, peak = [], [], []
    for _ in range(steps):
        u = u + 0.2 * (np.roll(u, 1) + np.roll(u, -1) - 2 * u)
        total.append(float(u.sum()))
        p = np.clip(u / u.sum(), 1e-30, None)
        entropy.append(float(-(p * np.log(p)).sum()))
        peak.append(float(u.max() - u.min()))
    # `peak` is the not-conserved control. Entropy is NOT used for that: it rises and then
    # plateaus at equilibrium, so over the second half it is genuinely constant and calling
    # it non-conserved would be wrong. The detector was right about that and the first
    # version of this test was not.
    return {"total": total, "entropy": entropy, "peak_spread": peak}


def selftest() -> bool:
    """Calibrate every detector against a system whose answer is known."""
    ok = True
    print("  law detector calibration")

    # 1. Force-law fitter must recover the exponent it was NOT told about.
    for want in (-2.0, -3.0, -1.0):
        traj, hist, _ = nbody_two(exponent=want)
        r, a = two_body_tracks(traj, dt=1e-3)
        n, r2, npts = fit_force_law(r, np.abs(a))
        good = abs(n - want) < 0.15 and r2 > 0.98
        ok &= good
        print(f"    force law  built r^{want:<5.1f} -> recovered r^{n:<7.3f} "
              f"R2 {r2:.4f}  {'OK' if good else 'FAIL'}")

    # 2. Conservation must find energy and momentum in the N-body run...
    traj, hist, scales = nbody_two(exponent=-2.0)
    laws = {l.name: l for l in conservation_scan(hist, tol=5e-3, scales=scales)}
    for nm in ("total_energy", "momentum_y"):
        good = laws[nm].verdict == "EMERGENT"
        ok &= good
        print(f"    conservation {nm:<14} {laws[nm].verdict:<9} CV {laws[nm].statistic:.2e}"
              f"  {'OK' if good else 'FAIL'}")

    # 3. ...and must NOT find one where there is none.
    d = pure_diffusion()
    dl = {l.name: l for l in conservation_scan(d, tol=1e-3)}
    good = dl["total"].verdict == "EMERGENT" and dl["peak_spread"].verdict == "ABSENT"
    ok &= good
    print(f"    diffusion   total {dl['total'].verdict}, "
          f"peak_spread {dl['peak_spread'].verdict}  {'OK' if good else 'FAIL'}")

    # 4. Second law in a system that has it.
    slope, viol = entropy_trend(d["entropy"])
    good = slope > 0 and viol < 0.02
    ok &= good
    print(f"    second law  slope {slope:+.3e}  violations {viol:.1%}  "
          f"{'OK' if good else 'FAIL'}")

    # 5. The enforced/emergent split must actually fire.
    tagged = conservation_scan({"E+I+M": [1.0] * 50}, tol=1e-3)[0]
    good = tagged.verdict == "ENFORCED"
    ok &= good
    print(f"    enforced tag  E+I+M -> {tagged.verdict}  {'OK' if good else 'FAIL'}")

    print(f"  overall: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
