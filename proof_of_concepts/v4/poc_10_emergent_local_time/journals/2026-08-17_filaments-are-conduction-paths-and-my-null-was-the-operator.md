# 2026-08-17 (later): filaments ARE conduction paths — my null was the operator

Correction to exp_03, which reported that time does not conduct along the cosmic web. It does.
The null came from an operator that could not have detected conduction either way, and I
reported it as a physical result.

Supersedes the conduction section of
[`2026-08-17_viscosity-is-the-mechanism-not-the-stabiliser.md`](2026-08-17_viscosity-is-the-mechanism-not-the-stabiliser.md).
Everything else in that journal stands — this touches only the transport arm.

---

## What was claimed

> **Time does not conduct along the web.** Under the conduction form the web channels no better
> than a uniform control at the same n, box and r0: +0.217 vs +0.179, a +0.30σ difference. The
> clock field tracks how much matter is nearby, not how it is connected.

## Why that test could not work

exp_03 coupled particles by a **distance ball** of radius r0. A ball around a point in a dense
node contains far more particles than a ball on a filament or in a void — measured, ~798
neighbours in the web against 116 in the uniform control at the same `n` and `box`.

So "how much matter is nearby" and "how it is connected" are entangled **in the definition of
the operator.** There is no analysis that separates them afterwards. Worse, the relaxation
divides a signal among neighbours, so more neighbours means lower amplitude regardless of
topology — the ball manufactures an anti-channelling signal from geometry alone.

The null was structural. It said nothing about the physics, and I reported it as though it did.

## The fix: bounded degree

A **k-nearest-neighbour graph** gives every particle exactly k links whether it sits in a node,
on a filament, or alone in a void. Degree no longer tracks density:

| | mean degree | degree CV |
|---|---|---|
| web | 7.37 | 0.163 |
| uniform control | 7.14 | 0.170 |

Against exp_03's ball at 798 vs 116. Degree dilution is removed **by construction** rather than
corrected for, so whatever survives is topology.

## Three more broken observables, in this experiment alone

1. **Hop count.** On a k-NN graph, edge length scales with local spacing: in a dense filament
   the k nearest neighbours are physically close, so a hop covers little ground; in a void hops
   are long. Crossing a fixed Euclidean distance through dense material therefore takes *more
   hops* for purely geometric reasons. It gave −3.59σ "anti-channelling" — the same wrong answer
   as exp_03, arrived at by a completely different mechanism. Replaced with shortest **physical
   path length**.

2. **Explicit time-stepping.** With conductance ∝ 1/length, dense-region degrees become huge, so
   the stable step `ν/deg_max` collapses and nothing propagates. It returned **all NaN**, and the
   danger was that "nothing arrives" reads like a physical statement. It is a solver failure. The
   network is stiff *precisely because* it has the structure being measured.

3. **The length weighting itself.** Conductance ∝ 1/length hands dense regions high conductance
   automatically, because their edges are short. Same artifact as hop count, one level up.

Only the fourth formulation answers the question: **effective resistance** on a bounded-degree
graph with **topology-only edges**. A steady-state linear solve — no threshold, no step count,
no stiffness, nothing to tune.

## The result

Effective resistance from a source against local density, within fixed Euclidean shells,
topology-only edges, 5 seeds:

| k | web | uniform control | difference |
|---|---|---|---|
| 4 | +0.410 | −0.003 | +5.21σ |
| 6 | +0.422 | −0.022 | +6.22σ |
| 10 | +0.442 | −0.028 | **+9.14σ** |

**The effect size is k-independent** — 0.41 to 0.44 — with significance rising only because more
edges means less noise. That is what a real structural property looks like: the number should
not depend on how many links I chose to draw. Controls sit at zero throughout.

And it is **stronger** with length removed (+0.42) than with conductance ∝ 1/length (+0.297,
+2.57σ), so it is not the edge-length artifact either. The confound I was most worried about was
working *against* the signal.

**Dense particles are better connected than their Euclidean separation implies. Filaments are
conduction paths.**

## The shape of it, which I did not anticipate

The geodesic arm — shortest physical path length along the same graph — shows **nothing**:
+0.68σ, +0.02σ, −1.64σ across k, scattering around zero with no consistent sign.

Effective resistance and shortest path diverge in exactly one way: **redundancy.** Resistance
falls when there are many parallel routes between two points; shortest path does not care how
many routes exist, only how good the best one is.

So a filament does not conduct because it is a short wire. **It conducts because it is a thick
bundle of parallel ones.** That is a sharper statement than the original intuition, and it fits
the re-entry picture better than a single-route model would — redundancy is a property of the
structure at a scale, not of any one path through it.

## What this costs, and the pattern

This is the **fifth** measurement failure in one day running the same direction: understating
structure. Percolation read low (undersampled grid), transport read null (distance ball), τ's
character read as artifact (correct, but I nearly kept the wrong version), the forming web read
as decay (snapshot on a re-entering process), and now conduction read null (entangled operator).

Not one of them overstated structure. An unbiased error process does not do that.

The version I now believe: **skepticism was standing in for rigour.** A null feels rigorous, so
the error mode that survived my own review was the one that looked careful. And this one had
already been written into a merged PR as a settled result — which is exactly how a measurement
artifact becomes a fact about the world.

The habit that keeps catching it is unchanged: **an operator has to be able to answer the
question before its answer means anything.** Ask what the measurement would return if the
hypothesis were true, and check the apparatus can produce that number at all. exp_03 could not
have.

## Still open

- **Which Laplacian SEC viscosity is.** Mean (random-walk) and conduction (unnormalised) predict
  opposite signs, and the corpus never fixed the choice. exp_06 shows the conduction reading is
  the one with structure behind it, but that is evidence, not a derivation.
- **Whether the clock field itself conducts, or only the graph does.** exp_06 measures the
  network's conduction properties. Coupling `LocalTime`'s viscosity to a bounded-degree graph
  rather than a ball, and re-running the dilation and transport arms on it, is the follow-on.
