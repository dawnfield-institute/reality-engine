# 2026-08-12: The engine does reduce entropy — in M, not E

## The failing metric

The scorecard reports `entropy reduction: target > 0, measured -0.533, 100% [F]`,
identically with and without PAC enforcement. Entropy is *increasing*. Since SEC —
`∂S/∂t = α∇I − β∇H` — is the engine's core dynamic, its own signature metric pointing the
wrong way is not a tuning miss.

## What the metric measures

`sec_tracking.field_entropy` is the Shannon entropy of the **E² spatial distribution**:

```python
p = E_sq / E_total
entropy = -(p * p.log()).sum()
```

Low entropy = energy concentrated. High = spread out. Reduction is read as structure
forming.

## The finding

**Energy diffuses. Mass concentrates.** Measured at 64×32, canonical pipeline, noise off:

| tick | H(E) | H(I) | H(M) |
|---|---|---|---|
| 1 | 6.9137 | 6.9405 | 6.1660 |
| 250 | 7.2184 | 7.2074 | **7.2841** ← peak |
| 1000 | 7.2726 | 7.2096 | 7.1732 |
| 3000 | 7.4219 | 7.3014 | 6.9552 |
| 10000 | 7.4658 | 7.2973 | **6.8229** |

`log(N) = 7.6246` is the maximum. H(E) climbs monotonically toward it — energy spreads,
as energy does. **H(M) falls monotonically from its peak** — mass concentrates, which is
structure forming.

Across three seeds, 10k ticks, sampled every 500:

| seed | peak | final | dH | monotone after peak |
|---|---|---|---|---|
| 42 | 7.3914 | 7.0644 | **−0.3271** | **True** |
| 43 | 7.3999 | 7.0329 | **−0.3670** | **True** |
| 44 | 7.3710 | 7.0026 | **−0.3684** | **True** |

Zero exceptions to monotonicity in any seed, and still falling at t = 10000 — the
concentration has not saturated.

## Why the metric never saw it

Two independent reasons, either sufficient:

1. **Wrong field.** Structure forms in M, under gravity. E is energy; it diffuses by
   construction. Measuring structure formation on E² measures the one field guaranteed
   not to show it.
2. **Degenerate reference.** `entropy_reduction_cumulative = initial − current`, with
   initial taken at t = 0 — where **M is all zeros**. H(M) is therefore spuriously low at
   the reference point, so genuine concentration still reads as an increase. H(M) rises
   from 6.17 to 7.28 while mass is being *generated*, then falls; anything referenced to
   t = 0 sees only the rise.

## What was added

`mass_entropy`, `mass_entropy_rate`, `mass_entropy_reduction_from_peak` in
`sec_tracking.py`. The rate is reference-free — negative means concentrating — and the
cumulative form is referenced to the **peak**, not to t = 0, for the reason above.

The existing `field_entropy` is left alone. Redefining a scorecard metric changes the
score, and that is a decision to take deliberately rather than as a side effect.

## What this does and does not settle

- **Settles:** the engine reduces entropy, continuously and monotonically, in the field
  where structure lives. The [F] was an artifact of which field was watched and from
  where.
- **Does not settle:** whether this is SEC or simply gravitational clumping. Mass
  concentrating under self-gravity is structure formation, but attributing it to SEC
  specifically requires isolating the SEC contribution from the gravity operator's.
- **Does not settle:** the scorecard's score. Nothing was rescored.

## Next

- Isolate SEC's contribution from gravity's — run with the gravity operator removed and
  see whether H(M) still falls.
- The concentration has not saturated by 10k ticks. Where does it stop, and does the
  stopping point relate to a DFT constant?
