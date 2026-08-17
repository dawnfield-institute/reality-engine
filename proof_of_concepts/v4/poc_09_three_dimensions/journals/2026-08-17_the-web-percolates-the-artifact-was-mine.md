# The published web percolates. The artifact was mine.

**2026-08-17 — retraction of this POC's headline claim.**

Supersedes [`2026-08-16_the-published-web-does-not-percolate.md`](2026-08-16_the-published-web-does-not-percolate.md),
which stays in place as the record of what was believed. Corrections layer forward.

---

## What was claimed

> **THE PUBLISHED WEB DOES NOT PERCOLATE.** At exp_11's config, 32³ … exp_11 as written gives
> percolation 0.0068 at occupancy 0.1124, against a 3D white-noise control of 0.0025 at
> occupancy 0.1085. Rendered, it is scattered isolated cells distributed uniformly through the
> box — a random point process with high contrast.

That claim went on to become the empirical anchor for a milestone (`dawn-field-theory` M17),
which opened on the premise that four independent routes had triangulated on a percolation
floor of 0.007–0.019 and that the framework was therefore "maximally sub-critical."

## What is true

**exp_11's cosmic web percolates.** Same run, same physics, same code — read at different
binnings:

| res | particles/cell | percolation | `is_web` |
|---|---|---|---|
| 16 | 0.98 | 0.472 | True |
| 24 | 0.29 | 0.433 | True |
| **32 — exp_11's own binning** | 0.12 | **0.385** | **True** |
| 48 | 0.036 | 0.281 | False |
| **64 — where 0.0068 came from** | **0.015** | 0.062 | False |

And this POC's *own committed script*, run unmodified at exp_11's resolution, reports
**percolation 0.3443 ± 0.0628** across its seeds. The claim was never reproducible by the code
that shipped with it.

At matched sampling the 3D substrate gives percolation 0.406 ± 0.069 from exp_11's own
**uncorrelated lattice** start, with **5/5 seeds** passing the web gate.

## The mechanism

4000 particles binned onto a 64³ grid is **0.015 particles per cell**. At that sampling the
density field is empty by construction: the overdense set shatters into singletons, so *any*
web — real or deliberately synthetic — reads as disconnected. A control settles it
independently: a known-connected 3D web reads percolation 1.000 across occupancy 0.082–0.268,
covering this whole regime. The instrument was never the problem.

**The tell was in view the entire time.** Occupancy read 0.012–0.04 in every one of those runs
and was printed beside every percolation value — this POC added that reporting *specifically*
because percolation is meaningless without it. I never asked whether that occupancy was
physically attainable. In 3D the site-percolation threshold is 0.312; nothing at occupancy
0.015 can span, whatever it is.

## Three further defects in this POC, found while checking

1. **The script contradicts its own meta.** `scripts/exp_01_replicate_exp11.py` defaults to
   `res=64` — the exact resolution `meta.yaml` identifies as the failed first attempt
   ("read CV 15.4 against a target of 2.94"). Run today it still reports CV 15.418. The
   finding was documented and the code was never changed. Now defaults to 32, exp_11's own.

2. **"REPLICATION IS EXACT" is not reproducible.** meta.yaml records void 0.888 against 0.89
   and CV 2.948 against 2.94. The committed script gives void 0.939 and CV 7.105 at 32³ — the
   void gate passes, the CV gate fails. Whatever produced the recorded numbers is not what is
   in the repository, so the replication must be treated as UNVERIFIED rather than exact.

3. **Ξ ambiguity, exactly as CLAUDE.md warns.** The script uses Ξ_analytic/φ = 0.65414; the
   follow-on work used Ξ_discrete/φ = 0.65334. A 0.1% difference, too small to explain the
   above, but it means two rounds were not run at the same operating point and neither said so.

## What this costs

The POC-05 finding — contrast and connectivity are decoupled — was measured in 2D and is not
touched by this. But **the 3D extension of it was wrong**, and the "wall" it appeared to
establish never existed. Everything downstream that treated a low percolation number as a fact
about DFT rather than about my grid needs re-reading:

- POC-09's headline result and `meta.yaml` `result:` block — corrected here.
- The claim that the attractive variant "spans only 0.47 of the box" — measured at the same
  bad sampling, now **unverified**.
- M17's founding premise in `dawn-field-theory` — retracted separately in that repo.

## The rule this earns

**Match particles-per-cell (`n/res^d ≈ 1`) before reading any connectivity statistic, and
check every threshold-based measure against a known-connected control at the same occupancy.**

Encoded in `worldmodel.matched_res()` so the default is right without anyone remembering, and
the run banner now prints particles/cell on every invocation.

## The larger pattern

Every measurement error in this round ran in the same direction: understating structure. A
null reads as rigour, so the error mode that survived review was the one that looked careful.
That asymmetry is the thing to watch — not any single artifact.
