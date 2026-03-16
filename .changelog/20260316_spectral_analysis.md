# Spectral Analysis: Emergent Quantized Oscillation Spectrum

## Discovery

Fourier analysis of field evolution at mass peak sites reveals a **quantized harmonic spectrum** emerging purely from field dynamics.

## Method

- Full pipeline (14 operators including SpinStatistics + ChargeDynamics)
- Track E, I, M, and disequilibrium (E-I) time series at local mass maxima
- FFT with Hann window, multiple spectral windows at different evolution stages
- High-resolution run: 4000-tick recording window after 500-tick warmup

## Key Results

### 1. Fundamental frequency = 1/sqrt(2)
- Measured: f0 = 0.7104
- Theoretical: 1/sqrt(2) = 0.7071
- Error: **0.47%**
- Origin: geometric resonance of the 2D Mobius lattice (diagonal coupling distance = sqrt(2))

### 2. Complete n/2 harmonic series
Both integer (bosonic) and half-integer (fermionic) harmonics present:

| n | f_n | Type |
|---|-----|------|
| 1 | 0.355 | Sub-harmonic |
| 2 | 0.710 | Fundamental |
| 3 | 1.066 | 3/2 harmonic |
| 4 | 1.421 | 2nd harmonic |
| 5 | 1.776 | 5/2 harmonic |
| 6 | 2.131 | 3rd harmonic |

SpinStatisticsOperator creates genuine spin-statistics: both spin-0 (integer) and spin-1/2 (half-integer) oscillation modes coexist.

### 3. Inverse mass-frequency scaling
- Early evolution (diverse masses): f ~ M^(-1.74), correlation = -0.57
- Lighter structures oscillate faster — uncertainty-principle-like behavior
- Structures lock to discrete modes, not continuous frequencies

### 4. Discrete mode assignment
- Structures don't oscillate at arbitrary frequencies
- They lock to specific harmonic modes
- Light structures (M ~ 2.5) prefer higher modes (3rd harmonic and above)
- Heavy structures (M ~ 3.5-4.0) settle to ground mode

## Scripts

- `scripts/spectral_analysis.py` — initial analysis (v1)
- `scripts/spectral_analysis_v2.py` — mass-diverse analysis with three evolution windows

## Significance

This is emergent quantization — discrete energy levels arising from continuous field dynamics on a Mobius manifold. No quantum mechanics was programmed; the spectrum emerges from:
- RBF evolution (wave equation)
- Mobius topology (boundary conditions create standing waves)
- Spin statistics (selection rules suppress/enhance specific modes)
- Mass-frequency coupling (heavier structures = lower modes)

The 1/sqrt(2) fundamental ties the emergent "Planck frequency" to the lattice geometry, analogous to how the Debye frequency in condensed matter relates to lattice spacing.
