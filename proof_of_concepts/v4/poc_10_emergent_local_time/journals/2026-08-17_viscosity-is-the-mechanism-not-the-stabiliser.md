# 2026-08-17: viscosity is the mechanism, not the stabiliser

Research log for POC-10. Written in the order things arrived, including the parts that were
wrong, because the pattern in the failures turned out to matter more than the results.

---

## The idea, as it was put to me

> "If a quantum collapse is a time framerate and Planck time is that, but time is local, it also
> has turbulence, viscosity and as it moves with mass and gravity — and it must be dispersed
> evenly enough for it to not collapse as well. It could flow through the filaments just like
> macro EM."

Four separable claims: a framerate set by collapse; locality; turbulence/viscosity with a
dispersion condition; and conduction along the web.

## The sign works out, and it is PAC-native

The naive reading has more collapse = faster time, which gives dilation the wrong way round.
But mass is *already-actualized* potential. Where potential has been spent there are fewer
collapse events still available, so mass-dense regions tick **slowly**. Dilation falls out of
the ledger rather than being inserted.

That also says which field to couple to: `state.P`, the unactualized potential buffer. In the
particle substrate its analogue is the local collapse budget, `1/(1 + κδ)`.

## What I built

`LocalTime`, ahead of the integrator so the clock rate is set from the current configuration
before anything moves. τ normalised to mean 1 — local time is a **redistribution** of a fixed
budget, never a global speed-up, which is what keeps the ledger auditable. Both candidate
sources implemented as a switch rather than arguing for one, because the corpus does not fix
the choice and they make *different topological predictions*.

## First result, and why it was nearly worthless

`corr(τ, δ) = −0.673` in `potential` mode, slow clocks 89% on filaments. A good-looking picture:
blue slow-clock regions sitting exactly on the mass, red fast clocks in the voids, flow
converging along filaments into nodes.

**It is near-definitional.** τ *is* `1/(1+κδ)`. I defined the clock rate as a decreasing function
of local density and then measured that it decreases with local density. Only −0.67 rather than
−1.0 because of viscosity smoothing and binning. Evidence that the arithmetic works.

What is *not* circular: the mode comparison. `rate` gives −0.089 and a 56/37 filament/boundary
split against `potential`'s 89/11, and `rate` is built from `|dδ/dt|` — a different quantity.

## The test that could fail

Compare τ against the **potential** rather than the density. `Φ_yukawa` (range r0) is the
substrate's own force law and is near-definitional too. `Φ_newton` is long-range: every mass in
the box contributes at every point, and the substrate never computes it.

First attempt: partial correlation +0.730, controlling for δ linearly. Strong.

**Wrong, twice over.** First suspicion was nonlinearity — τ is nonlinear in δ, and a linear
regression leaves `f(δ)` in the residual. Added τ's own functional form plus polynomial terms
to the basis: +0.746. Barely moved, so the nonlinearity was not it.

The real confound was **scale**. τ is built from neighbour counts within r0 = 10; the grid cell
is 3.0. Those are different quantities, so `f(δ_r0)` survives a regression on `δ_grid`, and Φ is
smooth, so the residuals correlate for a purely arithmetic reason.

`state.prev_delta` stores the exact input to τ. Regressing *that* out:

| ν | partial given grid δ | partial given τ's exact input |
|---|---|---|
| 0.0 | +0.630 | **−0.009 ± 0.04** |
| 0.3 | +0.746 | +0.240 |
| 0.6 | +0.817 | +0.322 |

**Zero at ν = 0.** With diffusion off, emergent time is exactly its own local input relabelled.
The +0.630 was entirely the scale confound.

And that null is the point. It means the test can return nothing — so the ν > 0 rows say
something. Diffusion over neighbours, iterated 180 ticks, carries clock information past r0.

**Viscosity is not the stabiliser I added it as. It is the mechanism by which time becomes a
field.** I named that channel before running the sweep — it is the only non-local path in the
operator — so this is a called shot rather than a story fitted afterwards.

## What does not support the picture

Fitting τ(Φ_newton): quadratic R² 0.908, GR weak-field √(1+2Φ) 0.802, linear 0.749. If this were
gravitational dilation the √ form should win. Monotone response to the potential, wrong shape.
Recorded sharp rather than softened.

## Conduction: three broken observables

"Flows through the filaments like macro EM" is a transport claim with a clean null. Freeze the
positions, raise τ at one source particle, iterate only the viscosity step, and ask whether at
**fixed Euclidean distance** the perturbation prefers dense paths.

1. **Amplitude — degree-confounded.** Relaxation shares a value among neighbours, so a dense
   filament divides the same signal over ~7× more particles. Amplitude falls with density
   whether or not filaments channel. It manufactured exactly the negative correlation it was
   meant to test for.
2. **Arrival time — also degree-confounded**, after I explicitly claimed it was not. Crossing a
   threshold still depends on amplitude.
3. **The graph was saturated.** At r0/box = 1/6 each particle neighbours ~800 of 6000; the graph
   diameter is ~2 hops; arrival fraction was **1.00 in every run**. There was no front to
   propagate. Both earlier "results" were noise on a saturated measurement.

That third one is a finding about the implementation: **I tied the clock coupling to the force
radius.** Local time only has propagation structure when its coupling is short-ranged against
the web. Separating them (`--diffuse-radius`) is what made a front exist at all.

Then the answer, with a uniform control that can return zero:

| form | web | uniform | difference |
|---|---|---|---|
| mean (random-walk Laplacian) | −0.346 | +0.093 | −4.83σ |
| conduction (unnormalised) | +0.217 | +0.179 | +0.30σ |

**Time does not conduct along the web.** It tracks how much matter is nearby, not how it is
connected. The control's +0.179 is the giveaway — density fluctuations do most of the work.

> **SUPERSEDED the same day by exp_06 — this conclusion is WRONG.** The distance-ball operator
> entangles density with connectivity by construction (~798 neighbours in the web against 116 in
> the control) and could not have detected conduction either way. On a bounded-degree k-NN graph,
> effective resistance separates the web from the control at **+9.14σ**, k-independent. Filaments
> ARE conduction paths — and it is a parallel-path effect, since the shortest-path arm shows
> nothing. See
> [`2026-08-17_filaments-are-conduction-paths-and-my-null-was-the-operator.md`](2026-08-17_filaments-are-conduction-paths-and-my-null-was-the-operator.md).
> Everything else in this journal stands.

But which Laplacian SEC viscosity *is* was never fixed, and the two forms predict opposite
signs. A distance-ball coupling cannot see connectivity by construction. Testing conduction
properly needs a bounded-degree neighbour graph so filaments are paths, not blobs.

## The reframe that made the null suspect

> "You are expecting a recursion of a process to act like the precursor of that process… gravity
> from a local perspective clumps and clusters. When it recurs on itself into a galactic or
> universal process, it's more the filaments. And then those filaments start clumping."

That makes the transport test malformed rather than negative: I probed **one** coupling scale and
read the answer as if it were *the* answer. A single-scale probe cannot distinguish "does not
conduct" from "conducts at a level I did not sample."

And we had already produced the alternation and I had mislabelled it. The 3D run goes uniform →
filaments at t≈180 → clumps by t=400, and I called the web "transient, then it collapses." Under
the re-entry reading that is not the web dying, it is the next level starting.

## Scale-resolved measurement

Hessian classification across a smoothing ladder. Character **is** scale-dependent: filaments
peak at 58% at 0.45 r0, nodes rise monotonically outward, voids dominate the small end. And
across epochs the filament peak migrates **outward** (0.45 → 0.64 r0) while small-scale voids
grow 6.3% → 25.3% → 32.7%. Small scales evacuate as structure re-forms at larger ones.

That is standard hierarchical structure formation seen from the DFT side — Zel'dovich collapse
runs sheet → filament → node, and larger scales collapse later, so at any instant different
scales sit at different points in the sequence.

## A retraction inside the round

τ's character is sheet-dominated where matter is filament-dominated, consistently. I wrote that
up as "the clock field sits one axis of collapse behind matter," which would have shipped as a
finding.

Control: apply τ's *formula* to the same ρ with no viscosity and no evolution. It reproduces the
inversion — 30.5% filament / 58.6% sheet against matter's 61.5 / 29.9. **The inversion is the
nonlinear transform.** What survives is |τ − algebra| = 0.136 against a 0.04–0.07 reference,
with τ sitting between matter and the pure algebra. Second-order, not a headline.

## The instrument the round actually produced

> "Something is the accumulation of its history. That is the definition of what it is… if we
> could figure out how to detect it, the risk of feeling defeated when it's actually a victory
> is less of a risk."

Every instrument here reports a **state**, and a state cannot distinguish "this ended" from "this
moved somewhere I am not measuring." That is not an edge case — it is the *default* failure of
snapshot measurement applied to a process that re-enters itself, and it explains why the errors
were one-directional: a snapshot can register "gone from here" but has no way to register
"arrived elsewhere."

PAC gives the discriminator. If potential is conserved through actualization, structure that
vanishes at one scale must appear at another.

Four candidate metrics, three failed on overlapping ranges: `1 − |Σd|/Σ|d|` (web 0.125–0.273 vs
dissolving 0.094–0.287), signed net drift (+0.021 vs +0.021 on one seed), late/early exchange
ratio (0.69–1.28 vs 0.76–2.38). What works is the **magnitude of cross-scale exchange of collapse
order** — a dissolving system moves every scale the same way, a re-entering one moves scales
against each other:

| system | exchange | vs control | verdict |
|---|---|---|---|
| frozen | 0.000 | 0.00× | TERMINATED ✓ |
| dissolving | 0.137 / 0.146 | 0.97× / 1.03× | TERMINATED ✓ |
| web at t=450 | 0.330 / 0.343 | **2.34× / 2.43×** | **RE-ENTERING** |

**The calibration once passed while broken** — an intermediate version NaN-defaulted every
verdict to TERMINATED, so both controls "passed" for reasons unrelated to what they guard. It
now also requires that the reference *separated* the web from the controls. A check that can only
ever say PASS guards nothing.

## The pattern worth keeping

Every measurement error today ran in the same direction: **understating structure.** Percolation
read low, transport read null, τ's structure read as artifact, the web read as decay. Never once
did I overclaim. An unbiased error process does not do that.

I think the cause is that **skepticism was standing in for rigour.** A null feels rigorous, so
the error mode that survived my own review was the one that looked careful.

There is a structural reason this project is exposed to it: the system and the instrument are
both built, so there is no nature to calibrate against, and constructions fail toward their
author's expectations. The one habit that held up all day was **measuring the reference in the
same run** instead of reasoning about thresholds. Every result that survived had a control that
could have killed it; every result that died lacked one.
