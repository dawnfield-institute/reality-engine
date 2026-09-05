# 2026-09-05 — the derived trigger selects the cores

Milestone R exp_28 (dawn-field-theory, sealed `bf833113`) asked whether a *derived* dissipation
channel — severance on exp_15's all-edges overstress barrier, exp_01's decoupling form, no amount —
lets this substrate hold structure once its speed cap and tuned drag are gone. It does not, and the
way it fails is the result: score 0/4, with T2 failing *with sign* at the one informative τ (the
severed set is less connected than a random subset of the baseline at the same count, 3/3 seeds,
−2.6 pooled σ) and T1 failing in the direction the registration named (what leaves carries less
energy than the mean, so the retained per-particle energy rises).

**Why.** `min_j |S_i − S_j| > τ` is true only at a local extremum of the entropy field. exp_15 ran it
on a noise field, whose extrema are random; here the SEC entropy grows where the density is high,
so its extrema are the collapse cores. The barrier is a core detector: `u_out < 0` in every run,
fired/retained KE ratio ≈ 0.5 at first firing, and random removal at the same count leaves *more*
percolation than the selected removal. The tuned drag D, calibrated to the same retained KE per
particle, holds as much or more in 3/3.

**n = 4000 at τ\* = 10, three seeds (2.0–2.4 min a run, 48–54% severed, 1827–2062 retained):** the
same picture at full size. Retained-set percolation over t ∈ [10, 15]: S 0.051 / 0.046 / 0.051,
R 0.055 / 0.046 / 0.052 — indistinguishable; B0 on its own 4000 dissolves to 0.035 / 0.038 / 0.028
(peak at the first collapse, as on the proxy); the matched drag D 0.024 / 0.026 / 0.030 on its own
set. Frame-matched: S vs a random subset of B0 at the same count 0.051 / 0.046 / 0.051 vs 0.076 /
0.037 / 0.058 — a null (−0.5 σ), where the proxy's τ = 20 was −2.6 σ. The selection effect is real
at the onset, when the first 6–10% to fire are the cores, and diluted once half is gone: the
barrier fires on everything after the neighbourhood degree collapses. Verdicts in the
dawn-field-theory scored JSON (`exp_28_dynamical_severance_20260905_232916.json`).

**What this closes.** `.spec/challenges.md` C4.1 option 3 — a derived sink lets the substrate hold.
What remains is option 2: the impulsive pressure that detonates the collapse (KE/|U| 0.07 → 10
between t = 3 and 5) is the term to derive from the SEC functional. A sink downstream of the
detonation removes the wrong energy from the wrong particles.

**What the instruments showed.** On the corrected force the baseline's detonation is a hundred times
weaker than on the self-propelling one (KE/|U| ≈ 11–18 in the window, not 960; dt recovers to
0.023, no floor ticks; a proxy run is 6 s, not 26); structure forms at the first collapse (percolation
0.18–0.37 at t = 3–4) and dissolves (0.06–0.10 by t = 10). Every closure residual in the grid is
below 6 × 10⁻⁷. Landauer erasure is ~1% of the pressure work (SP3 registered < 1%: missed by a
hair; unscored) and changes nothing. `memory_decay` orders the severed fraction as registered (SP2)
without changing the sign of anything.

**What the round caught before it could lie.** The design pass matched D on a ratio of signed
energies that crosses zero as the set unbinds; the smoke test returned ρ = −5.4 and the control had
silently become B0. Replaced with the retained kinetic energy per particle (positive-definite) and
one pre-declared refinement. The aggregator globbed its own grid file on the second pass and crashed
the D phase; excluded by name. Both are in the outcomes journal, and neither touched a threshold.

Pointers: the grids (`results/proxy/`, `results/full/`), the scored JSONs in dawn-field-theory
`milestone-r/results/`, the outcomes journal `milestone-r/journals/2026-09-05_exp28_outcomes.md`.
