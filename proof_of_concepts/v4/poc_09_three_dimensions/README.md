# POC-09 (v4): Three dimensions

**Status**: completed · **Pillar**: SEC

> ## ⚠ RETRACTED AND CORRECTED — 2026-08-17
>
> **This POC's headline claim was wrong, and the error was in my measurement, not in exp_11.**
> exp_11's web **percolates**: 0.385 with `is_web=True` at exp_11's own 32³ binning, and
> **0.3443 ± 0.0628** from this POC's own committed script run unmodified at that resolution.
>
> The retracted 0.0068 came from binning 4000 particles onto a **64³** grid — 0.015 particles
> per cell — where the density field is empty by construction and *any* web, real or
> deliberately synthetic, reads as disconnected.
>
> Full account: [`journals/2026-08-17_the-web-percolates-the-artifact-was-mine.md`](journals/2026-08-17_the-web-percolates-the-artifact-was-mine.md).
> The original claim and reasoning are preserved in the 2026-08-16 journal — corrections layer
> forward.

## Why

POC-05 (contrast and connectivity are decoupled) and POC-07 (particles percolate far better
than the field engine) both rest on percolation measured in **2D**, and both engines had only
ever been run there. That is not a neutral choice for a connectivity measurement:

- In 2D, filaments that cross **must** intersect, voids must be fully enclosed, and sheets do
  not exist. exp_11 names the target topology as filaments, **sheets**, nodes and voids.
- The site-percolation threshold is **0.593 in 2D against 0.312 in 3D** — the same field
  percolates far more readily in three dimensions.

exp_31 Part A wants 3D independently: the cascade 1/r profile requires `d_spatial = 3`.

## Result — the hypothesis is CONFIRMED, after correcting my own error

**In 3D at matched sampling the web percolates**: 0.406 ± 0.069 across 5 seeds, every one
passing the web gate — and from exp_11's own *uncorrelated lattice* start, with nothing added.

### Binning alone moves the answer by 8×

One run, identical physics, read at different grids:

| res | particles/cell | percolation | `is_web` |
|---|---|---|---|
| 16 | 0.98 | 0.472 | True |
| 24 | 0.29 | 0.433 | True |
| **32 — exp_11's own** | 0.12 | **0.385** | **True** |
| 48 | 0.036 | 0.281 | False |
| **64 — the retracted claim** | **0.015** | 0.062 | False |

4000 particles on 64³ cannot fill it: the overdense set shatters into singletons. A
known-connected 3D control reads **1.000 across occupancy 0.082–0.268**, covering this regime,
so the instrument was sound and the sampling was not.

**The tell was printed on every line.** Occupancy read 0.012–0.04 in all those runs — and this
POC added occupancy reporting *because percolation is meaningless without it*. In 3D the site
threshold is 0.312; nothing at 0.015 could ever have spanned.

### The replication is UNVERIFIED, not exact

The previously recorded void 0.888 / CV 2.948 is **not reproducible by the committed script**:

| | void | density CV | gate |
|---|---|---|---|
| exp_11 reported | 0.89 | 2.94 | — |
| script at 32³ (today) | 0.939 | 7.105 | void PASS, **CV FAIL** |
| script at 64³ (old default) | 0.988 | 15.418 | **both fail** |

Whatever produced the recorded numbers is not what is in this repository. The script shipped
with `res=64` — the resolution this POC's *own notes* identify as the failed first attempt.
The artifact was documented and the code was never changed. Now defaults to 32.

### The span claim is unverified

"Largest component spans only 0.47 of the box, coexists with 231 other components" was measured
at the same bad sampling and has not been re-measured. Treat as withdrawn pending a rerun.

## Withdraws a POC-07 claim

POC-07 recorded a *"sign error caught by the physics, not by a test"* in exp_09 — `d[i,j] =
pos_i − pos_j` points away from `j`, so the force reads as repulsive. **The reference
reproduces exactly as written**, so the repulsive convention is what produced the corpus's
published numbers. The v4 substrate runs a different system, which is legitimate, but calling it
a correction to exp_09 was wrong.

Both are now available: `CANONICAL` (attractive) and `EXP11` (as written), plus
`SECUpdateRelative` — exp_11's SEC rule differs from exp_09's in having no threshold and no
decay.

## Two measurement artifacts

**Percolation is severely resolution-dependent** for sparse particle data. The same 3D state:

| binning | 24³ | 32³ | 48³ | 64³ |
|---|---|---|---|---|
| percolation | 0.489 | 0.425 | 0.102 | 0.027 |
| void | 0.899 | 0.938 | 0.974 | 0.988 |

An 18× swing from grid choice alone. exp_11 measures at 32³; the first replication attempt used
64³ and read CV 15.4 against a target of 2.94.

**POC-07's "15×" compared a sparse field to a dense one.** At matched cell count (128²) the
field engine has occupancy 0.257 against the particles' 0.084 and still percolates worse
(0.0117 vs 0.0563). The conclusion survives — the particles win from a disadvantaged position —
but the multiplier is ~5×.

`web_metrics` now reports **`occupancy` on every call**. A percolation number without its
occupancy cannot be compared to anything.

## Running it

```
python proof_of_concepts/v4/structure.py      # includes selftest_3d(); must PASS
python proof_of_concepts/v4/poc_09_three_dimensions/scripts/exp_01_replicate_exp11.py
```

3D selftest expectations are **derived, not copied**: ξ = 1−1/e and ξ = 2σ are per-axis and
carry across dimension; `coherent_fraction`'s baseline is the exact discretised mode count
`Π(2c+1)/n` (0.0199 on 48³, not `band³` = 0.0156); and 3D white noise must not percolate because
occupancy 0.11 sits below the 3D threshold of 0.312.

## Open

- Sweep `sec_balance` in **exp_10's own convention**, past 1.3 so the optimum is bracketed —
  see `poc_07/journals/2026-08-16_xi-is-not-the-optimum.md`.
- Whether any parameter regime between "scattered" and "collapsed" gives a space-filling
  network at all.
