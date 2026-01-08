"""
Integrated field analysis for dark matter, mass quantization, and c² validation.
"""
from core.reality_service import RealityEngineService, EngineConfig
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

PHI = 1.618033988749895
XI = 1.057
c2_theory = np.pi * PHI / XI

print('='*60)
print('INTEGRATED FIELD ANALYSIS')
print('='*60)

config = EngineConfig(size=(50, 28))
service = RealityEngineService(config=config)
service.initialize(mode='big_bang')

# Collect time series
E_total, M_total = [], []
for i in range(8000):
    service._step_engine()
    state = service.engine.current_state
    E_total.append(state.actual.cpu().sum().item())
    M_total.append(state.memory.cpu().sum().item())
    if (i+1) % 2000 == 0:
        print(f'  Step {i+1}')

# Final state analysis
state = service.engine.current_state
# Convention from test: E = actual, I = potential
E = state.actual.cpu().numpy()
I = state.potential.cpu().numpy()
M = state.memory.cpu().numpy()

print(f'\nField stats:')
print(f'  E (actual): mean={E.mean():.3f}, std={E.std():.3f}')
print(f'  I (potential): mean={I.mean():.3f}, std={I.std():.3f}')
print(f'  M (memory): mean={M.mean():.3f}, std={M.std():.3f}')

# c² from time series (last 1000 steps)
E_arr, M_arr = np.array(E_total[-1000:]), np.array(M_total[-1000:])
coeffs = np.polyfit(M_arr, E_arr, 1)
c2_measured = -coeffs[0]
print(f'\nc² from time series:')
print(f'  c² = {c2_measured:.4f}')
print(f'  πφ/Ξ = {c2_theory:.4f}')
print(f'  Match: {100 * c2_measured / c2_theory:.1f}%')

# Correlations in final state
E_flat = E.flatten()
I_flat = I.flatten()
M_flat = M.flatten()

print(f'\nSpatial correlations:')
print(f'  E-M: {np.corrcoef(E_flat, M_flat)[0,1]:.4f}')
print(f'  I-M: {np.corrcoef(I_flat, M_flat)[0,1]:.4f}')
print(f'  E-I: {np.corrcoef(E_flat, I_flat)[0,1]:.4f}')

# RECURSIVE GRAVITY ANALYSIS (Dawn Field Theory)
# Dark matter is NOT a substance - it's emergent gravity from recursive memory fields
# See: recursive_gravity.py and entropy_information_polarity_field experiments
print('\n' + '-'*40)
print('RECURSIVE GRAVITY ANALYSIS (SEC/EIPF)')
print('-'*40)
print('Dark matter = Emergent gravity from recursive memory fields')

from scipy.ndimage import uniform_filter, laplace

Xi = 1.0571  # SEC-validated threshold

# 1. LOCALIZED COLLAPSED MASS (visible matter)
#    High M concentration = actual collapsed structure
M_threshold = np.percentile(M, 80)  # Top 20% = collapsed cores
visible_mass = M[M > M_threshold].sum()
visible_cells = (M > M_threshold).sum()

# 2. RECURSIVE MEMORY FIELD (what creates "dark matter" gravity effects)
#    = M-weighted entropy/information gradients creating gravitational curvature
#    From EIPF: Φ_E = -∇S + Γ·R (entropy gradient + recursive ancestry)
#    The SPREAD of M (not just low density) creates field effects

# Memory field diffusion = how much M has spread (creates extended gravity)
M_smooth = uniform_filter(M, size=5, mode='wrap')
memory_field_extent = M_smooth.sum()  # Total field influence

# Recursive memory = M in non-peak regions that creates gravitational effects
diffuse_memory = M[M <= M_threshold].sum()
diffuse_cells = (M <= M_threshold).sum()

# 3. GRAVITATIONAL CURVATURE from memory field (like tangle_strength in recursive_gravity)
#    ∇²M = Laplacian gives field curvature
laplacian_M = laplace(M, mode='wrap')
field_curvature = np.abs(laplacian_M).sum()

# 4. EFFECTIVE GRAVITY RATIO
#    This is what observers would measure as "dark/visible matter ratio"
#    = (Field memory effects) / (Localized mass)
#    In Dawn Field Theory: gravity from memory field >> gravity from localized mass
field_gravity_ratio = diffuse_memory / (visible_mass + 0.01)

print(f'\nCollapsed Mass (Visible):')
print(f'  Cells: {visible_cells} ({100*visible_cells/M.size:.1f}%)')
print(f'  Mass: {visible_mass:.2f}')

print(f'\nRecursive Memory Field (creates "dark matter" effects):')
print(f'  Diffuse cells: {diffuse_cells} ({100*diffuse_cells/M.size:.1f}%)')
print(f'  Diffuse mass: {diffuse_memory:.2f}')
print(f'  Field curvature (∇²M): {field_curvature:.2f}')

print(f'\nEffective Gravity Ratio (Diffuse/Visible):')
print(f'  Ratio: {field_gravity_ratio:.2f}')
print(f'  Cosmological observation: ~5:1')
print(f'  Match: {100 * field_gravity_ratio / 5:.1f}%')

# Mass clustering
print('\n' + '-'*40)
print('MASS CLUSTERS')
print('-'*40)
M_high = M > np.percentile(M, 80)
labeled, n = ndimage.label(M_high)
print(f'Found {n} clusters in top 20%')

masses = []
for i in range(1, n+1):
    masses.append(M[labeled == i].sum())
masses = sorted(masses, reverse=True)[:5]
print(f'Top 5 cluster masses: {[f"{m:.1f}" for m in masses]}')

# Check for Fibonacci ratios
if len(masses) >= 2:
    ratios = [masses[i]/masses[i+1] for i in range(len(masses)-1)]
    print(f'Mass ratios: {[f"{r:.2f}" for r in ratios]}')
    print(f'φ = {PHI:.3f}')

# Shell structure analysis
print('\n' + '-'*40)
print('SHELL STRUCTURE')
print('-'*40)

# Find largest mass concentration
max_pos = np.unravel_index(M.argmax(), M.shape)
cy, cx = max_pos
print(f'Highest mass at: ({cy}, {cx})')

H, W = M.shape
y, x = np.ogrid[:H, :W]
r = np.sqrt((y - cy)**2 + (x - cx)**2)
max_r = min(H, W) // 2

radii = np.arange(1, max_r)
profile = []
for r_val in radii:
    shell = (r >= r_val - 0.5) & (r < r_val + 0.5)
    if shell.sum() > 0:
        profile.append(M[shell].mean())
    else:
        profile.append(0)

profile = np.array(profile)
peaks, _ = find_peaks(profile, height=profile.mean() * 0.8, distance=2)

print(f'Shell peaks at radii: {radii[peaks]}')
if len(peaks) >= 2:
    ratios = radii[peaks] / radii[peaks[0]]
    print(f'Normalized: {np.round(ratios, 2)}')
    print(f'Hydrogen n²: [1, 4, 9, 16, ...]')
    print(f'Fibonacci: [1, 2, 3, 5, 8, ...]')

print('\nRadial profile:')
for i in range(min(len(radii), 10)):
    bar = '█' * int(profile[i] * 2)
    marker = ' ◄' if i in peaks else ''
    print(f'  r={radii[i]:2d}: {profile[i]:.2f} {bar}{marker}')

print('\n' + '='*60)
print('ANALYSIS COMPLETE')
print('='*60)
