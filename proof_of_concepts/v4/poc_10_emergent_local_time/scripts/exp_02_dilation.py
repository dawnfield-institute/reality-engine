#!/usr/bin/env python3
"""Is emergent local time gravitational dilation, or is it density wearing a new label?

`time_flow.py` reports corr(tau, delta) ~ -0.67 in "potential" mode. That number is close to
DEFINITIONAL: tau is built as 1/(1 + kappa*delta), so measuring that it falls with delta
measures the arithmetic, not the physics. This script is the test that can actually fail.

The claim being tested is that the clock rate behaves like a gravitational potential. So
compare tau against the POTENTIAL rather than the density:

    Phi_yukawa   solves  (grad^2 - 1/r0^2) Phi = 4 pi G rho
                 The potential of the force law the substrate ACTUALLY uses,
                 exp(-r/r0)/r. Local, range r0 — essentially a smoothed density, so a
                 strong correlation here is close to definitional too. Included as the
                 near-tautological control, not as evidence.

    Phi_newton   solves  grad^2 Phi = 4 pi G rho
                 Long-range. Every mass in the box contributes at every point. The substrate
                 NEVER computes this: tau is set from neighbour counts inside r0. So any
                 agreement is information tau was not given directly.

**The discriminator is the partial correlation** corr(tau, Phi_newton | delta): how much of
the long-range potential the clock field tracks AFTER local density is regressed out. Near
zero means emergent time is local density relabelled — a real and reportable null. Clearly
non-zero means the clock field carries non-local structure.

The second question is FORM, not just direction. GR's weak field gives
dtau/dt = sqrt(1 + 2 Phi) ~ 1 + Phi. So fit both, and also a plain quadratic, and report
which describes the measured relation best. A monotone relation with the wrong shape is a
different result from dilation.

    python dilation.py [--mode potential|rate] [--steps 180] [--dims 3]
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

from particles import CANONICAL_TIME, EXP11_TIME, ParticleConfig, ParticleEngine  # noqa: E402
from structure import web_metrics  # noqa: E402

from worldmodel import matched_res  # noqa: E402

OUT = HERE.parent / "results"


def potential(rho, box, screening=None, g=1.0):
    """Spectral Poisson solve on a periodic box.

    `screening = None` gives Newtonian (long-range); `screening = r0` gives Yukawa, the
    potential belonging to the substrate's own exp(-r/r0)/r force. The k = 0 mode is dropped
    for the Newtonian case (a periodic box has no absolute zero point) but kept for Yukawa,
    where the screening term makes it finite.
    """
    d = rho.ndim
    res = rho.shape[0]
    delta = rho / rho.mean() - 1.0
    dk = 2 * np.pi * res / box
    freqs = [np.fft.fftfreq(res) * dk for _ in range(d)]
    k2 = sum(K ** 2 for K in np.meshgrid(*freqs, indexing="ij"))
    denom = k2 + (0.0 if screening is None else 1.0 / screening ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_k = -4 * np.pi * g * np.fft.fftn(delta) / np.where(denom > 0, denom, np.inf)
    phi = np.fft.ifftn(phi_k).real
    return phi - phi.mean()


def _basis(c, kappa, nonlinear):
    """Design matrix for regressing out delta.

    A LINEAR basis is not good enough here and using one would fake the result. tau is built
    as 1/(1 + kappa*delta) — nonlinear — so a linear regression leaves f(delta) behind in the
    residual, and Phi is itself a smoothed delta, so the two residuals correlate for a purely
    arithmetic reason. The nonlinear basis includes tau's OWN functional form plus low-order
    polynomial terms, so what survives is genuinely not explainable by local density.
    """
    cols = [np.ones_like(c), c]
    if nonlinear:
        cols += [1.0 / (1.0 + kappa * np.clip(c, -0.99 / max(kappa, 1e-9), None)),
                 c ** 2, c ** 3, np.log1p(np.clip(c, -0.99, None))]
    return np.vstack(cols).T


def partial_corr(a, b, c, kappa=1.0, nonlinear=True):
    """corr(a, b) with c regressed out of both — the emergence discriminator."""
    A = _basis(c, kappa, nonlinear)
    def resid(x):
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        return x - A @ coef
    ra, rb = resid(a), resid(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def fit_forms(tau, phi):
    """Which functional form describes tau(Phi)? Report R^2 for each."""
    out = {}
    scale = np.abs(phi).max() or 1.0
    x = phi / scale                                     # dimensionless; absorbs G and units
    def r2(pred):
        ss = ((tau - pred) ** 2).sum()
        tt = ((tau - tau.mean()) ** 2).sum()
        return float(1 - ss / tt) if tt > 0 else float("nan")

    A = np.vstack([x, np.ones_like(x)]).T
    (a1, b1), *_ = np.linalg.lstsq(A, tau, rcond=None)
    out["linear_1_plus_phi"] = {"r2": r2(A @ [a1, b1]), "slope": float(a1)}

    arg = 1 + 2 * a1 * x
    if (arg > 0).all():
        sq = np.sqrt(arg)
        A2 = np.vstack([sq, np.ones_like(sq)]).T
        (a2, b2), *_ = np.linalg.lstsq(A2, tau, rcond=None)
        out["gr_weak_field_sqrt"] = {"r2": r2(A2 @ [a2, b2])}
    else:
        out["gr_weak_field_sqrt"] = {"r2": float("nan")}

    A3 = np.vstack([x ** 2, x, np.ones_like(x)]).T
    c3, *_ = np.linalg.lstsq(A3, tau, rcond=None)
    out["quadratic"] = {"r2": r2(A3 @ c3)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("potential", "rate"), default="potential")
    ap.add_argument("--dims", type=int, default=3, choices=(2, 3))
    ap.add_argument("-n", type=int, default=8000)
    ap.add_argument("--box", type=float, default=60.0)
    ap.add_argument("--r0", type=float, default=10.0)
    ap.add_argument("--g", type=float, default=1.5)
    ap.add_argument("--sec-balance", dest="sec_balance", type=float, default=0.65334)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 101])
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--viscosity", type=float, default=0.3)
    ap.add_argument("--time-coupling", dest="time_coupling", choices=("ball", "knn"),
                    default="ball", help="how tau's viscosity finds neighbours; knn bounds degree")
    ap.add_argument("--time-k", dest="time_k", type=int, default=6)
    ap.add_argument("--res", type=int, default=None)
    ap.add_argument("--convention", choices=("attractive", "exp11"), default="exp11")
    a = ap.parse_args()

    pipe = EXP11_TIME if a.convention == "exp11" else CANONICAL_TIME
    res = a.res or matched_res(a.n, a.dims)
    print(f"  {a.n} particles, {a.dims}D, {a.steps} steps, mode '{a.mode}', "
          f"grid {res}^{a.dims}, seeds {a.seeds}, coupling '{a.time_coupling}'"
          + (f" k={a.time_k}" if a.time_coupling == "knn" else ""))
    print(f"  {'seed':>5} {'c(t,d)':>8} {'c(t,Yuk)':>9} {'c(t,New)':>9} "
          f"{'lin-part':>9} {'NONLIN-PC':>10} {'perc':>6}")

    rows, keep = [], None
    for sd in a.seeds:
        cfg = ParticleConfig(n=a.n, box=a.box, r0=a.r0, g=a.g, sec_balance=a.sec_balance,
                             dims=a.dims, seed=sd, entropy_init=0.1, time_mode=a.mode,
                             time_kappa=a.kappa, time_viscosity=a.viscosity,
                             time_coupling=a.time_coupling, time_k=a.time_k)
        eng = ParticleEngine(cfg, pipeline=pipe)
        for _ in range(a.steps):
            eng.tick()

        rho = eng.density_field(res).cpu().numpy().astype(float)
        tau = eng.field_of(eng.state.tau, res).cpu().numpy()
        ok = np.isfinite(tau)
        if ok.sum() < 50:
            continue
        delta = rho / rho.mean() - 1.0
        phi_y = potential(rho, a.box, screening=a.r0, g=a.g)
        phi_n = potential(rho, a.box, screening=None, g=a.g)

        t, d, py, pn = tau[ok], delta[ok], phi_y[ok], phi_n[ok]
        c_d = float(np.corrcoef(t, d)[0, 1])
        c_y = float(np.corrcoef(t, py)[0, 1])
        c_n = float(np.corrcoef(t, pn)[0, 1])
        pc = partial_corr(t, pn, d, a.kappa, nonlinear=True)
        pc_lin = partial_corr(t, pn, d, a.kappa, nonlinear=False)
        web = web_metrics(rho)
        print(f"  {sd:>5} {c_d:>+8.3f} {c_y:>+9.3f} {c_n:>+9.3f} {pc_lin:>+9.3f} "
              f"{pc:>+10.3f} {web['percolation']:>6.3f}")
        rows.append({"seed": sd, "corr_tau_delta": c_d, "corr_tau_phi_yukawa": c_y,
                     "corr_tau_phi_newton": c_n, "partial_corr_newton_given_delta": pc,
                     "partial_corr_linear_basis": pc_lin,
                     "forms_vs_newton": fit_forms(t, pn),
                     "forms_vs_yukawa": fit_forms(t, py),
                     "percolation": float(web["percolation"])})
        if keep is None:
            keep = (t, d, py, pn, sd)

    if not rows:
        print("  no usable runs"); return 1

    arr = lambda k: np.array([r[k] for r in rows])
    pc = arr("partial_corr_newton_given_delta")
    print(f"\n  corr(tau, delta)            {arr('corr_tau_delta').mean():+.3f} "
          f"+/- {arr('corr_tau_delta').std():.3f}   <- near-definitional, tau IS f(delta)")
    print(f"  corr(tau, Phi_yukawa)       {arr('corr_tau_phi_yukawa').mean():+.3f} "
          f"+/- {arr('corr_tau_phi_yukawa').std():.3f}   <- control, r0-range = local")
    print(f"  corr(tau, Phi_newton)       {arr('corr_tau_phi_newton').mean():+.3f} "
          f"+/- {arr('corr_tau_phi_newton').std():.3f}")
    pcl = arr("partial_corr_linear_basis")
    print(f"  partial (LINEAR basis)      {pcl.mean():+.3f} +/- {pcl.std():.3f}   "
          f"<- CONFOUNDED: leaves f(delta) in the residual")
    print(f"  PARTIAL (NONLINEAR basis)   {pc.mean():+.3f} +/- {pc.std():.3f}   "
          f"<- THE TEST")
    print(f"\n  Reading: |partial| < 0.1 => emergent time is local density relabelled.")
    print(f"           clearly non-zero => the clock field carries non-local structure.")

    f_n = rows[0]["forms_vs_newton"]
    print(f"\n  functional form of tau(Phi_newton), seed {rows[0]['seed']}:")
    for k, v in f_n.items():
        print(f"     {k:<22} R2 {v['r2']:+.4f}")

    t, d, py, pn, sd = keep
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for axi, (x, lab) in zip(ax, [(d, "local overdensity  delta  (what tau is built from)"),
                                  (py, "Phi_yukawa  (r0-range — the substrate's own force)"),
                                  (pn, "Phi_newton  (long-range — never computed by tau)")]):
        axi.scatter(x, t, s=5, alpha=0.2, edgecolors="none")
        axi.set_xlabel(lab, fontsize=9); axi.set_ylabel("clock rate  tau")
        axi.grid(alpha=0.3)
        axi.set_title(f"corr {np.corrcoef(t, x)[0,1]:+.3f}", fontsize=10)
    fig.suptitle(f"is emergent time dilation, or density relabelled?  mode '{a.mode}', "
                 f"seed {sd} — partial corr (Newton | delta) = {pc.mean():+.3f}", fontsize=11)
    fig.tight_layout()
    p = OUT / f"dilation_{a.mode}_{a.dims}d_{stamp}.png"
    fig.savefig(p, dpi=115, bbox_inches="tight")
    plt.close(fig)
    (OUT / f"dilation_{a.mode}_{stamp}.json").write_text(
        json.dumps({"config": vars(a), "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n  wrote {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
