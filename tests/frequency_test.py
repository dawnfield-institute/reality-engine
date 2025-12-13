"""Quick frequency emergence test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from dynamics.klein_gordon import KleinGordonEvolution, create_initial_perturbation

print('=== FREQUENCY EMERGENCE TEST ===')
print('Running 5000 steps with proper FFT analysis...\n')

# Create evolution - smaller dt for higher frequency resolution
kg = KleinGordonEvolution(xi=1.0571, dt=0.5, damping=0.001)

# Initial perturbation
psi = create_initial_perturbation((64, 64), 'gaussian', amplitude=1.0)
psi_prev = psi.clone()

# Evolve and collect amplitude - track the FIELD value at center
amplitudes = []
center_vals = []
for i in range(5000):
    psi, psi_prev = kg.evolve_step(psi, psi_prev)
    amplitudes.append(psi.abs().mean().item())
    center_vals.append(psi[32, 32].item())  # Track actual field oscillation

# FFT analysis on CENTER values (actual oscillation)
signal = np.array(center_vals)
signal = signal - signal.mean()  # Remove DC

fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), 0.5)  # dt=0.5

# Find positive frequencies
pos_mask = freqs > 0
pos_freqs = freqs[pos_mask]
pos_power = np.abs(fft[pos_mask])

# Top 5 frequencies
top_idx = np.argsort(pos_power)[-5:][::-1]
print('Top 5 frequencies by power (field oscillation at center):')
for i, idx in enumerate(top_idx):
    print(f'  {i+1}. {pos_freqs[idx]:.6f} Hz (power: {pos_power[idx]:.2f})')

dominant_freq = pos_freqs[top_idx[0]]
print(f'\nDOMINANT FREQUENCY: {dominant_freq:.4f} Hz')
print(f'Theoretical f = m/(2π): {0.232413/(2*np.pi):.4f} Hz')
print(f'Target (from QBE legacy): 0.0200 Hz')

# Check relationship to mass
m_squared = 0.054016
m = np.sqrt(m_squared)
f_theory = m / (2 * np.pi)
print(f'\n=== PHYSICAL INTERPRETATION ===')
print(f'm² = (Ξ-1)/Ξ = {m_squared:.6f}')
print(f'm = {m:.6f}')
print(f'f = m/(2π) = {f_theory:.6f} Hz')
print(f'Dominant f = {dominant_freq:.6f} Hz')
print(f'Ratio f_dom/f_theory = {dominant_freq/f_theory:.4f}')
