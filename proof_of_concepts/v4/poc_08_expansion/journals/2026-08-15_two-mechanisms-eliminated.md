# 2026-08-15: Two mechanisms eliminated, and the one that survives needs no mechanism

POC-07 found the particle web is transient — percolation peaks then decays as structure
merges. Two explanations were proposed and both are now dead. What replaces them is the null
hypothesis that should have been on the table before either.

---

## The sequence

**exp_01a — Ω_Λ sweep. Measured the wrong thing.** Varying Ω_Λ at fixed H₀ = 0.02 made
expansion look strictly harmful, monotonically: static retained 42.5% of peak, DFT's 1/φ 9.5%,
over-expanding 6.3%. But `a` reached 25–45, so a fixed physical `r0 = 5` sat inside a physical
box of ~3000 and particles could not reach each other at all. `retained` being monotone in
`a_final` is the signature of dilution, not of expansion competing with collapse.

**exp_01b — H₀ swept three decades. A clean story on one seed.** Holding Ω_Λ = 1/φ and bringing
H₀ down so `a` only grows to ~1.2, an interior optimum appeared: H₀ = 0.001 retained **53.4%**
against static's 42.5%, and beat it on final percolation 0.093 to 0.068. Too little expansion
looked like static, too much was worse than static. Exactly the shape expansion-versus-collapse
should produce.

**exp_02 — five seeds killed it.**

| arm | peak | final (tail mean) | retained |
|---|---|---|---|
| static | 0.192 ± 0.041 | 0.082 ± 0.023 | 44.4% ± 13.4 |
| H₀ 0.0005 | 0.162 ± 0.025 | 0.070 ± 0.013 | 43.0% ± 5.2 |
| **H₀ 0.001** | 0.169 ± 0.027 | 0.096 ± 0.034 | 57.7% ± 20.9 |
| H₀ 0.002 | 0.140 ± 0.012 | 0.067 ± 0.010 | 47.8% ± 6.2 |

**0.76σ** on final percolation, **1.2σ** on retention. Static alone spans ±13 percentage points
across seeds — the scatter swamps every between-arm difference. The window was a
single-realization fluctuation with a good story attached.

**exp_03 — the alternative died, and so did its premise.**

The argument was that `damping = 0.99` and the speed cap bleed energy every tick with nothing
replacing it, so the system virializes regardless of the box. Measured:

| arm | final | retained | **KE end/start** |
|---|---|---|---|
| damping 0.99 (POC-07) | 0.071 ± 0.007 | 41.7% ± 12.9 | **4.43** |
| damping 0.999 | 0.079 ± 0.006 | 36.6% ± 8.5 | 3.40 |
| damping 1.0 (none) | 0.093 ± 0.022 | 55.5% ± 4.2 | 3.30 |
| damping 1.0, cap 20 | 0.067 ± 0.008 | 42.6% ± 8.5 | **242** |

**Kinetic energy grows 3–4×.** The system gains energy; gravitational infall injects far more
than the damping removes. The premise was stated twice as though established and one column
refuted it.

Removing damping does read better — 55.5% vs 41.7% — but that is **1.77σ on three seeds**, the
same size of effect that had just evaporated one experiment earlier. Not claimed.

Incidentally: the speed cap, not the damping factor, is where energy management actually
happens. Lifting it from 2 to 20 lets KE grow **242×**.

## What survives

**Hierarchical merging reduces connectivity by construction.** Structures form, then merge into
fewer, larger, more widely separated objects. Percolation falls because the *number of distinct
objects* falls — no special mechanism required, and none of expansion, dissipation or any other
force needs to be invoked.

The "web epoch" is then just the window where objects are numerous enough and close enough to
touch: too diffuse before it, too few and too far apart after. That is qualitatively what real
cosmology shows, and it explains POC-07's peak-then-decay without adding anything.

## Scope limit, stated before the results and unchanged by them

**This toy cannot test Ω_Λ = 1/φ.** In real cosmology Ω_Λ is meaningful because H₀, G and the
matter density are physically anchored. Here `g = 0.8`, `r0 = 5` and `h0` are numbers chosen by
hand, so Ω_Λ sets only the *shape* of a(t) and no real balance. The framework's prediction is
about a ratio this simulation has no units to express.

Making it testable means **deriving `g` and `r0`** rather than choosing them. That is a real
piece of work and not a parameter change, and until it is done, no expansion result here says
anything about the framework.

## What the round is worth

Two nulls and a refuted premise, which is a fair description of a day's work and not a bad one.

The expansion hypothesis was proposed here as the flagship — "a DFT prediction going in as a
parameter, and the simulation doing something it wasn't tuned to do." It was the wrong
experiment twice over, and neither failure is a statement about DFT. Both are statements about
this machine and about proposing mechanisms faster than they can be checked.

The replication is the part worth keeping. exp_01b's window had a clean mechanism, an interior
optimum, and a monotone story on either side of it. It was noise. The only reason that is known
is five seeds, and the only reason five seeds were run is the pattern from earlier in the day,
where four structure metrics and a detector arm failed in exactly this way.

**The single-seed result with the good story is the one to distrust most.**

## One caught before it could do damage

`pairwise()` originally divided a *comoving* displacement by a *physical* distance, yielding
unit vectors of length 1/a. That would have diluted gravity by an extra factor of `a` —
indistinguishable in the output from expansion doing its job, and fatal to the whole
experiment. It now returns `(r_physical, d_comoving, r_comoving)`, with the force magnitude
taking the physical radius and the direction the comoving one. Verified: unit vectors 1.0000 at
a = 4, `r_phys/r_com` exactly 4.0000, static path bit-identical to POC-07.
