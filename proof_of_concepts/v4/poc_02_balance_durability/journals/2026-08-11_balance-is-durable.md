# 2026-08-11: Balance is durable — and it is *local*, not global

## Result

**Kill sentence did not fire.** Balance restores, robustly, across the whole swept range.

18 runs: 9 parameter settings (base + 4 parameters × ×0.5/×2) × 2 seeds. Ledger-preserving
impulse. Twin sanity exact — pre-impulse gap `0.000e+00`.

| observable | median R | range | durable settings |
|---|---|---|---|
| **`info_fraction`** | **0.0281** | [0.0070, 0.1152] | **18/18 (100%)** |
| **`gamma_local_mean`** | **0.0500** | [0.0152, 0.6556] | **17/18 (94%)** |
| `xi_s_mean` | 0.3246 | [0.0000, 5.4119] | 10/18 (56%) |
| `balance_magnitude` | 0.7469 | [0.4950, 1.0530] | 1/18 (6%) |
| `alpha_local_mean` | 0.8540 | [0.3480, 1.5771] | 2/18 (11%) |
| `lambda_local_mean` | 0.8540 | [0.3480, 1.5771] | 2/18 (11%) |

`info_fraction` — the E/I balance, |I|/(|E|+|I|) — recovers **97% of any displacement**, in
every single parameter setting tested. That is a wide basin: ×0.5 to ×2 across four
different parameters and it never left the durable band.

`alpha` and `lambda` are identical to four decimals throughout, because they are one degree
of freedom (λ = 1 − α), not two observables.

## The finding is the contrast with POC-01

| | R | reading |
|---|---|---|
| **Global ledger** (POC-01) | **1.000** | neutral — displace it and it stays displaced, forever |
| **Local E/I balance** (POC-02) | **0.028** | durable — 97% of any displacement decays |

Same engine, same method, same bands, same twin-difference. **The attractor is local. The
global total has no basin at all.**

## This contradicts the stated expectation, and that matters

The framing under test was:

> *"When we're seeing something globally, that's when conservation is being enforced as an
> attractor, not with a locality."* (Peter, 2026-08-11)

The engine says the reverse: **global is neutral, local is the attractor.** Recorded as
measured. Three readings, none of which this experiment can decide between:

1. **The engine is missing the global mechanism.** Plausible — POC-01 showed nothing in the
   dynamics references an absolute ledger value.
2. **The framing needs revision** — globality is not itself an attractor but *emerges* from
   local balance being one.
3. **"Global" means something else.** The engine's global is the sum over one grid. In the
   parent/sibling reading, that grid is a *local patch* of something larger — so the
   engine's "global" is the framework's "local", and what was measured here as local
   balance is the attractor within the patch, while the patch total is free to drift
   precisely because it is one term in a parent's sum.

**Reading 3 would make both results consistent**, and it is the one the framework's own
structure suggests: *our global is someone else's local*. A child term that restores
internally while its total floats is exactly what the hierarchy predicts. But this
experiment did not test a hierarchy, and consistency is not evidence — recorded as an
interpretation to be tested, not as a result.

## Secondary

**One Tier-1 coupling is an attractor and the others are not.** `gamma_local_mean` (DFT
target 1/φ) is durable in 17/18 settings; `alpha` and `lambda` (targets ln 2, 1 − ln 2) are
not, and are unstable in 2. The scorecard treats all of them as coupling attractors. That is
now differentiated: γ behaves like one, α and λ do not.

**`xi_s_mean` is bimodal** — range [0.0000, 5.4119], durable in 56%. Some settings restore
it completely, others amplify. Worth its own experiment; it is the Ξ-adjacent quantity and
the least stable thing measured.

**Distance from DFT targets was recorded but not graded**, per the registration. It is not
a criterion here and no claim is made about it.

## Method notes

Two silent implementation bugs were found during wiring, both of which would have produced
confident wrong answers:

1. `info_fraction` is emitted by `sec_tracking` only on some ticks. Computed from the fields
   instead of read from metrics.
2. **`D₀` was measured immediately after the impulse, but metric-based observables lag one
   tick** — the metrics dict still held the previous tick's values, so `D₀ = 0` and
   `R = nan` for all five of them. Field-computed observables have no lag, so the bug was
   *invisible* for `info_fraction`, which reported a plausible R throughout. That is how it
   was missed. `D₀` is now taken after one tick.

The second is the same class as POC-01's Amendment 3: a quantity meaning different things
in two contexts, silently. Worth watching for.

## Next

- **Test reading 3.** A hierarchy — nested patches where a sub-grid's ledger is a term in a
  parent's — is the experiment that distinguishes "engine is missing the mechanism" from
  "global-is-someone's-local". That is a v4 engine capability, not just an experiment.
- **`xi_s_mean` bimodality.** Which settings amplify it, and why.
- **γ vs α/λ.** One coupling restores and the others do not. The scorecard treats them
  alike; it should not.
