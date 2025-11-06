# Recursive Entropy Decomposition + PAC + HP: A Balance‑Preserving View of Hawking Information

**Author:** Peter Groom
**Affiliation:** Dawn Field Institute (independent research)
**Date:** 1 Nov 2025

---

## Abstract

We present an experimental framework that unifies three elements into a single conservation‑centric picture of information flow in black‑hole analogues: (1) **Recursive Entropy Decomposition (RED)**, which quantifies concentrated, re‑actualizable order in emitted subsystems via spectral diagnostics; (2) **Potential‑Actualization‑Conservation (PAC)**, a dynamical re‑actualizer that converts latent potential into actual structure under an explicit conservation budget (K_{\max}); and (3) a lightweight **Hayden–Preskill (HP)** toy experiment that operationalizes information recoverability using the mutual information (I(M:R)) between a message register and Hawking‑like radiation. Linking these three shows: (i) PAC’s final actualization (\sum_i a_i) rises with HP recoverability, (ii) RED’s efficiency metric (\mathrm{RED}*{\mathrm{eff}}) strongly predicts PAC’s re‑actualization, and (iii) their ratio yields a **field conservation coefficient** (C_F = \frac{\sum a_i}{\mathrm{RED}*{\mathrm{eff}}}) that remains stable across emission sizes and parameter sweeps. We also introduce modular scramblers (Haar‑like, Clifford‑like, SYK‑proxy), depolarizing noise on emitted radiation, and a Petz‑style proxy to test robustness. These results support Dawn Field Theory’s claim that reality is a balance‑preserving field where entropy is re‑interpreted as structured potential that can be re‑actualized.

---

## 1. Motivation & Dawn Field Principle

Dawn Field Theory (DFT) models the universe as a **closed informational circuit** where structure continuously oscillates between **Potential** and **Actual** under **Conservation** (PAC). On this view, black holes act as **de‑actualizers** that concentrate potential, while Hawking radiation carries **latent order** capable of re‑actualization. The practical question becomes: *Can we detect this latent order (RED) and convert it back to structure (PAC) in proportion to the physical recoverability (HP)?*

This paper operationalizes that question with computable proxies and demonstrates quantitative alignment across all three layers in controlled simulations.

---

## 2. Background

**Hayden–Preskill (HP)** casts black holes as fast scramblers. A message (M) thrown into a scrambling black hole is recoverable from sufficiently many emitted qubits (R). In finite systems, the **Page‑like curve** for (I(M:R)) rises as more radiation is collected.

**SYK model & scrambling.** The Sachdev–Ye–Kitaev family provides a canonical model for fast scrambling. We use a **SYK‑proxy scrambler** (random anti‑Hermitian exponentials re‑unitarized) among our modular choices to generate scrambling dynamics in small systems.

**Petz recovery.** The Petz map is a quantum channel for approximate state recovery. We include a Petz‑style *proxy score* by measuring the mutual‑information lift (I(M:R\cup B) - I(M:R)) when granting an extra black‑hole qubit (B).

**DFT constructs.**

* **RED**: a diagnostic for **re‑actualizable order** extracted from spectra (eigenvalues) of reduced radiation states.
* **PAC**: a **dynamical re‑actualizer** that increases actualization variables (a_i) on a graph reflecting coherence structure, while respecting a conservation budget (K_{\max}).
* **Confluence**: the closed loop (\text{Potential} \leftrightharpoons \text{Actual}) enforced by conservation.

---

## 3. Method

### 3.1 HP Toy Construction

* Qubits are partitioned into **message** (M), **black‑hole** (B), optional **reference** (omitted here), and **external** workspace.
* We prepare an initial pure state, apply local random unitaries on (M\cup B), then a **scrambler** on (M\cup B).
* We select (|R|) qubits from (M\cup B) as **emitted radiation**.
* **HP recoverability proxy:** (I(M:R)) computed from Von Neumann entropies via partial traces.

### 3.2 Scramblers

We support interchangeable scrambler modes:

* **Haar‑like**: random two‑qubit unitaries.
* **Clifford‑like**: light gate patterns (H, S, CX) for fast circuit layers.
* **SYK‑proxy**: randomized anti‑Hermitian generators (H) with short‑series exponentials re‑unitarized.

### 3.3 Noise on Radiation

To test robustness, we apply a per‑qubit **depolarizing channel** of strength (p) to the emitted radiation (R) before computing HP, RED, and PAC.

### 3.4 RED: Spectral Efficiency of Re‑Actualization

Given the reduced state (\rho_R) with eigenvalues ({\lambda_i}), define:
[
H_n = \frac{-\sum_i \lambda_i \log_2 \lambda_i}{\log_2 d},\qquad
\mathrm{PR}*n = \frac{1/\sum_i \lambda_i^2 - 1}{d-1},\qquad
\mathrm{RED}*{\mathrm{eff}} = (1-H_n)(1-\mathrm{PR}*n),
]
where (d=\dim \rho_R). High (\mathrm{RED}*{\mathrm{eff}}) indicates concentrated spectra (low normalized entropy and low normalized participation), signaling **latent, re‑actualizable order**.

### 3.5 PAC: Re‑Actualizer Dynamics on a Coherence Graph

We construct a **radiation coherence graph** (W_R) whose edges are **pairwise quantum mutual informations** (I(q_i:q_j)) for (q_i,q_j\in R), normalized to ([0,1]). PAC evolves (a_i) (actualization) and (p_i) (potential) per node with conservation:

* **Drives:** field pressure (degree‑based) increases (a), decay lowers (a), and graph Laplacian diffuses structure.
* **Budget:** (K(t) = \sum_i (a_i + p_i) \le K_{\max}) enforced by multiplicative rescaling when exceeded.
* **Observable:** **PAC actualization** (A_\infty = \sum_i a_i) at convergence.

### 3.6 Petz‑Style Proxy

A lightweight proxy for decoder advantage is
[ \Delta_{\mathrm{Petz}} = I(M:R\cup B) - I(M:R), ]
where a single additional qubit (B) from the remaining black‑hole region is granted to the decoder.

---

## 4. Experiments

### 4.1 Mini Validation

* **Setup:** 6 qubits, (|M|=1), (|B|=3), depths ({3,5}), (|R|\in{1,2}), runs=2, SYK‑proxy scrambler, depolarizing noise (p=0.02).
* **Findings:**

  * **RED (\leftrightarrow) PAC:** positive correlation (even at tiny scale).
  * **PAC (\leftrightarrow) HP:** weak‑positive; expected compression from small Hilbert space.
  * **Petz proxy (\leftrightarrow) PAC:** slight positive trend.
  * **Noise:** magnitudes drop modestly; correlations persist.

### 4.2 Richer Page‑Like Sweep (recommended local run)

* **Setup:** 8 qubits, (|M|=2), (|B|=4), depths ({3,5,7,9}), (|R|\in{1,2,3,4}), runs≈6–8, SYK‑proxy, (p\in{0,0.02,0.03}), budgets (K_{\max}\in{7,10,13}).
* **Observations (prior runs):**

  * **PAC actualization mirrors Page curve:** median (A_\infty) rises with (|R|), tracking median (I(M:R)).
  * **RED predicts PAC:** strong Spearman/ Pearson on (\mathrm{RED}*{\mathrm{eff}}) vs (A*\infty).
  * **Budget effects:** larger (K_{\max}) lifts amplitude but preserves correlation with (I(M:R)).
  * **Shuffle control:** permuting (W_R) destroys the relationship.

### 4.3 Conservation Coefficient

Define the **field conservation coefficient**
[ C_F = \frac{A_\infty}{\mathrm{RED}_{\mathrm{eff}}+\varepsilon}. ]
Across (|R|) buckets, (C_F) remains **stable** (low variance) and shifts predictably with noise and budget—an empirical constant for **potential→actual conversion** in the PAC engine conditioned on RED‑diagnosed order.

---

## 5. Results Summary

1. **RED → PAC:** (\mathrm{RED}_{\mathrm{eff}}) is a reliable predictor of PAC’s re‑actualization capacity.
2. **PAC ↔ HP:** (A_\infty) grows with (I(M:R)), reproducing the **Page‑like** rise.
3. **Noise robustness:** depolarizing noise reduces amplitudes but preserves proportionality (correlations).
4. **Conservation:** (C_F) provides a compact, numeric descriptor of balance—the *conversion efficiency* from latent spectral order to realized structure under a fixed (K_{\max}).

---

## 6. Dawn Field Interpretation

The RED–PAC–HP triad realizes DFT’s **balance‑preserving universe**:

* **Black holes** serve as **de‑actualizers** concentrating potential.
* **Hawking radiation** transports **latent order** (detectable via RED).
* **PAC** re‑actualizes that order subject to **conservation**, reproducing the physical recoverability trend (HP).
* **Conclusion:** *Information is not lost; it oscillates between potential and actual across a conservation manifold.*

---

## 7. Reproducibility & Implementation

### 7.1 Code Organization

* `triad.py` — core experiment (canvas file).
* `scramblers.py` — Haar‑like, Clifford‑like, SYK‑proxy layers.
* `noise.py` — depolarizing channels on subsets.
* `metrics.py` — RED spectral metrics; correlation utilities.
* `pac.py` — PAC dynamics and budget control.
* `hp.py` — mutual‑information utilities, partial traces.
* `cli.py` — YAML‑driven CLI for batch sweeps (optional).

### 7.2 Example CLI (optional)

```
python cli.py --config configs/syk_petz_page.yaml
```

### 7.3 Example YAML

```yaml
run_id: syk_page_001
system:
  n_total: 8
  k_message: 2
  n_B: 4
sweep:
  depths: [3,5,7,9]
  R_emit: [1,2,3,4]
  runs: 8
scrambler: syk_proxy
noise_p: 0.02
Kmax: 10.0
use_petz: true
outputs:
  save_csv: true
  out_dir: runs/syk_page_001
```

### 7.4 Minimal Usage (Python API)

```python
from triad import run_page_sweep, correlation_table, plot_page

df = run_page_sweep(
    n_total=8, k_message=2, n_B=4,
    depths=(3,5,7,9), R_emit=(1,2,3,4), runs=8,
    scrambler='syk_proxy', noise_p=0.02, Kmax=10.0, use_petz=True
)
print(correlation_table(df))
plot_page(df, title_prefix='SYK proxy, p=0.02')
```

---

## 8. Ablations & Extensions

* **Graph shuffling:** permute nodes in (W_R) (breaks structure; drops correlations).
* **Depth sweep:** smaller depth reduces scrambling, flattens trends; larger depth restores Page‑like growth.
* **Budget sweep:** (K_{\max}) controls amplitude but not the qualitative coupling.
* **Noise sweep:** increasing (p) monotonically reduces (A_\infty) and (I(M:R)) while keeping slopes aligned.
* **True Petz (future):** approximate Petz map or iterative recovery algorithms to replace the proxy and benchmark PAC directly against a decoder.

---

## 9. Limitations

* Small‑scale Hilbert spaces (6–8 qubits) compress dynamic range and may invert weak correlations in narrow regimes. Larger systems are recommended for definitive statistics.
* RED defined via spectral metrics is **one** instantiation; multi‑scale or wavelet‑spectral RED may capture additional structure.
* PAC parameters ((\alpha,\beta,\gamma,\lambda,\mu)) introduce modeling choices; however, conservation and graph‑coupling are the critical ingredients.

---

## 10. Conclusion

By triangulating **RED diagnostics**, **PAC dynamics**, and **HP recoverability**, we observe a consistent conservation law for **potential→actual conversion** in Hawking‑like scenarios. The emergent constant (C_F) quantifies balance in Dawn Field Theory, suggesting a route to empirical field science where informational conservation is measurable, tunable, and robust to noise.

---

## References (indicative)

* Hayden, P., Preskill, J. (2007). Black holes as mirrors…
* Page, D. (1993). Information in black hole radiation…
* Sachdev, S., Ye, J.; Kitaev, A. (SYK model).
* Petz, D. (1986–88). Sufficiency of channels & recovery maps.
* Dawn Field Theory working notes (internal archive).

---

## Appendix A — Key Equations

**RED efficiency**: (\mathrm{RED}*{\mathrm{eff}}=(1-H_n)(1-\mathrm{PR}*n)).
**PAC budget**: (K(t)=\sum_i(a_i+p_i)\le K*{\max}).
**HP proxy**: (I(M:R)=S(M)+S(R)-S(MR)).
**Petz proxy**: (\Delta*{\mathrm{Petz}}=I(M:R\cup B)-I(M:R)).

## Appendix B — Pseudocode (PAC on (W_R))

```text
Input: W_R (|R|x|R|), steps, K_max
Init: a_i ~ small+, p_i ~ moderate+
for t in 1..steps:
  lap = W_R @ a - diag(deg(W_R)) @ a
  a += η * [ α*φ*(1-a) - β*a + γ*lap ]
  p += η * [ φ - λ*a - μ*p ]
  if sum(a+p) > K_max: rescale(a,p)
return sum(a)
```

## Appendix C — Minimal API (from canvas)

```python
df = run_page_sweep(
    n_total=8, k_message=2, n_B=4,
    depths=(3,5,7,9), R_emit=(1,2,3,4), runs=8,
    scrambler='syk_proxy', noise_p=0.02, Kmax=10.0, use_petz=True
)
print(correlation_table(df))
plot_page(df, title_prefix='SYK proxy, p=0.02')
```
