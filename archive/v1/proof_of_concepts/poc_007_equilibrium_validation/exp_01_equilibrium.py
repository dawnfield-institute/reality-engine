"""
Dawn Field Theory Validation Test
Tests for emergent c², mass quantization, and Fibonacci patterns
"""

from core.reality_service import RealityEngineService, EngineConfig
import numpy as np

PHI = 1.618033988749895
XI = 1.057

print('=' * 60)
print('PHASE 3: Long-run equilibrium & mass quantization')
print('=' * 60)
print()

config = EngineConfig(size=(64, 32))
service = RealityEngineService(config=config)
service.initialize(mode='big_bang')

# Run to equilibrium
print('Running 5000 steps to reach equilibrium...')
E_vals, M_vals = [], []
for i in range(5000):
    service._step_engine()
    state = service.engine.current_state
    E_vals.append(state.actual.cpu().sum().item())
    M_vals.append(state.memory.cpu().sum().item())
    
    if (i+1) % 1000 == 0:
        # Check c^2 convergence
        E, M = np.array(E_vals), np.array(M_vals)
        coeffs = np.polyfit(M[-500:], E[-500:], 1)
        c2 = -coeffs[0]
        print(f'  Step {i+1}: c^2 = {c2:.4f}')

print()
print('FINAL c^2 ANALYSIS')
print('-' * 40)
E, M = np.array(E_vals), np.array(M_vals)
# Use last 1000 steps for stable measurement
coeffs = np.polyfit(M[-1000:], E[-1000:], 1)
c2_final = -coeffs[0]
print(f'  Equilibrium c^2 = {c2_final:.4f}')

# Check ratios to Dawn constants
print()
print('  Ratios to constants:')
print(f'    c^2 / phi^2 = {c2_final / (PHI**2):.4f}')
print(f'    c^2 / (2*phi) = {c2_final / (2*PHI):.4f}')
print(f'    c^2 / pi = {c2_final / np.pi:.4f}')
print(f'    c^2 / e = {c2_final / np.e:.4f}')
print(f'    c^2 / (phi + Xi) = {c2_final / (PHI + XI):.4f}')

print()
print('MASS QUANTIZATION ANALYSIS')
print('-' * 40)
M_field = service.engine.current_state.memory.cpu().numpy().flatten()
masses = M_field[M_field > 0.1]
print(f'  Total structures: {len(masses)}')
print(f'  Mass range: {masses.min():.3f} - {masses.max():.3f}')

# Find mass peaks
hist, bins = np.histogram(masses, bins=50)
peak_masses = []
for i in range(1, len(hist)-1):
    if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 20:
        peak_masses.append((bins[i] + bins[i+1])/2)

print(f'  Peaks found: {len(peak_masses)}')
if len(peak_masses) > 0:
    peak_masses = sorted(peak_masses)
    print(f'  Peak values: {[f"{p:.3f}" for p in peak_masses]}')
    
    if len(peak_masses) >= 2:
        print()
        print('  Peak ratios (checking for phi, Fibonacci):')
        for i in range(len(peak_masses)-1):
            ratio = peak_masses[i+1] / peak_masses[i]
            fib_match = ' <- phi!' if abs(ratio - PHI) < 0.1 else ''
            xi_match = ' <- Xi!' if abs(ratio - XI) < 0.05 else ''
            print(f'    {peak_masses[i+1]:.3f} / {peak_masses[i]:.3f} = {ratio:.4f}{fib_match}{xi_match}')
            
        # Check spacing for equal quantization
        spacings = np.diff(peak_masses)
        print()
        print(f'  Peak spacings: {[f"{s:.3f}" for s in spacings]}')
        print(f'  Spacing mean: {np.mean(spacings):.4f}')
        print(f'  Spacing std: {np.std(spacings):.4f}')
        
        # Is spacing related to phi?
        print(f'  Spacing / phi = {np.mean(spacings) / PHI:.4f}')
        print(f'  Spacing / Xi = {np.mean(spacings) / XI:.4f}')

# Check for Fibonacci-like mass ladder
print()
print('FIBONACCI LADDER CHECK')
print('-' * 40)
fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
# Find the mass quantum
if len(peak_masses) >= 1:
    quantum = peak_masses[0]
    print(f'  Base mass quantum: {quantum:.4f}')
    print(f'  Fibonacci prediction vs actual:')
    for f in fib[:6]:
        predicted = quantum * f
        actual_matches = masses[(masses > predicted*0.9) & (masses < predicted*1.1)]
        count = len(actual_matches)
        marker = ' <- FOUND!' if count > 10 else ''
        print(f'    n={f}: predicted M={predicted:.3f}, count={count}{marker}')

print()
print('=' * 60)
print('VALIDATION COMPLETE')
print('=' * 60)
