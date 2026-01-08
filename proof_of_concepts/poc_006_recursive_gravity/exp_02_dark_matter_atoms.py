"""
Test for:
1. Atomic-like shell structures in mass distribution
2. Dark matter signatures (mass without E-field coupling)
3. Force law refinement with better statistics
"""
from core.reality_service import RealityEngineService, EngineConfig
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter

PHI = 1.618033988749895
XI = 1.057

print('='*60)
print('DARK MATTER & ATOMIC STRUCTURE ANALYSIS')
print('='*60)

# Use optimal grid at critical scale
config = EngineConfig(size=(50, 28))
service = RealityEngineService(config=config)
service.initialize(mode='big_bang')

# Run longer for better equilibration
print('\nRunning 8000 steps to deep equilibrium...')
for i in range(8000):
    service._step_engine()
    if (i+1) % 2000 == 0:
        print(f'  Step {i+1}')

state = service.engine.current_state
E = state.actual.cpu().numpy()
I = state.potential.cpu().numpy()
M = state.memory.cpu().numpy()

print(f'\nField statistics:')
print(f'  E: min={E.min():.3f}, max={E.max():.3f}, mean={E.mean():.3f}')
print(f'  I: min={I.min():.3f}, max={I.max():.3f}, mean={I.mean():.3f}')
print(f'  M: min={M.min():.3f}, max={M.max():.3f}, mean={M.mean():.3f}')

# ============================================================
# DARK MATTER ANALYSIS
# Dark matter = mass (M) that doesn't correlate with energy (E)
# ============================================================
print('\n' + '='*60)
print('DARK MATTER ANALYSIS')
print('='*60)

# Normalize fields
E_norm = (E - E.mean()) / (E.std() + 1e-6)
M_norm = (M - M.mean()) / (M.std() + 1e-6)

# Global E-M correlation
global_corr = np.corrcoef(E.flatten(), M.flatten())[0,1]
print(f'\nGlobal E-M correlation: {global_corr:.4f}')

# Per-cell "dark matter fraction" = M that can't be explained by E
# Dark matter where M is high but E is not correspondingly high
E_predicted_M = E_norm * M.std() + M.mean()  # If E perfectly predicted M
residual = M - E_predicted_M  # The unexplained mass

dark_matter_mask = (residual > residual.std()) & (M > M.mean())
dark_matter_fraction = dark_matter_mask.sum() / M.size

print(f'Dark matter fraction (high M, low E-coupling): {100*dark_matter_fraction:.1f}%')

# Dark matter distribution
if dark_matter_mask.sum() > 0:
    dark_M = M[dark_matter_mask]
    visible_M = M[~dark_matter_mask & (M > M.mean())]
    print(f'Dark matter total mass: {dark_M.sum():.2f}')
    print(f'Visible matter total mass: {visible_M.sum():.2f}')
    print(f'Dark/Visible ratio: {dark_M.sum() / (visible_M.sum() + 0.01):.2f}')
    
    # Compare to cosmological dark matter ratio (~5:1)
    print(f'(Cosmological dark/visible ratio is ~5:1)')

# ============================================================
# DARK MATTER HALO ANALYSIS
# Check if dark matter forms halos around visible structures
# ============================================================
print('\n' + '-'*40)
print('Dark Matter Halo Analysis')
print('-'*40)

# Find visible structure centers (high E AND high M)
visible_mask = (E > E.mean() + E.std()) & (M > M.mean())
labeled_visible, n_visible = ndimage.label(visible_mask)
print(f'Found {n_visible} visible structures')

if n_visible > 0:
    # For each visible structure, check for dark matter halo
    for i in range(1, min(n_visible + 1, 4)):  # Check first 3
        struct_mask = (labeled_visible == i)
        y_coords, x_coords = np.where(struct_mask)
        cy, cx = y_coords.mean(), x_coords.mean()
        
        # Check annular region around structure
        y_grid, x_grid = np.ogrid[:M.shape[0], :M.shape[1]]
        dist_from_center = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
        
        # Inner region (the visible structure)
        inner_mask = dist_from_center < 3
        # Halo region
        halo_mask = (dist_from_center >= 3) & (dist_from_center < 8)
        
        inner_M = M[inner_mask].mean() if inner_mask.sum() > 0 else 0
        inner_E = E[inner_mask].mean() if inner_mask.sum() > 0 else 0
        halo_M = M[halo_mask].mean() if halo_mask.sum() > 0 else 0
        halo_E = E[halo_mask].mean() if halo_mask.sum() > 0 else 0
        
        # Dark matter ratio in halo vs core
        core_ratio = inner_M / (inner_E + 0.01)
        halo_ratio = halo_M / (halo_E + 0.01)
        
        print(f'\nStructure {i} at ({cx:.1f}, {cy:.1f}):')
        print(f'  Core M/E ratio: {core_ratio:.3f}')
        print(f'  Halo M/E ratio: {halo_ratio:.3f}')
        if halo_ratio > core_ratio:
            print(f'  => Halo is DARKER (more M per E) - dark matter signature!')

# ============================================================
# ATOMIC SHELL STRUCTURE ANALYSIS
# ============================================================
print('\n' + '='*60)
print('ATOMIC SHELL STRUCTURE ANALYSIS')
print('='*60)

# Find the highest mass concentration
max_pos = np.unravel_index(M.argmax(), M.shape)
print(f'\nHighest mass at position: {max_pos}')

# Radial profile from mass center
y_grid, x_grid = np.ogrid[:M.shape[0], :M.shape[1]]
dist = np.sqrt((y_grid - max_pos[0])**2 + (x_grid - max_pos[1])**2)

# Bin by distance
max_r = min(M.shape) // 2
radii = np.arange(0, max_r, 0.5)
M_profile = []
E_profile = []

for r in radii:
    shell_mask = (dist >= r) & (dist < r + 0.5)
    if shell_mask.sum() > 0:
        M_profile.append(M[shell_mask].mean())
        E_profile.append(E[shell_mask].mean())
    else:
        M_profile.append(0)
        E_profile.append(0)

M_profile = np.array(M_profile)
E_profile = np.array(E_profile)

# Look for shell structure (local maxima in radial profile)
from scipy.signal import find_peaks

# Smooth slightly to reduce noise
M_smooth = gaussian_filter(M_profile, sigma=1)
peaks, properties = find_peaks(M_smooth, prominence=0.1)

print(f'\nRadial mass profile from center:')
print(f'  Found {len(peaks)} shell peaks at radii: {radii[peaks]}')

if len(peaks) >= 2:
    # Check if shell radii follow quantum pattern (n^2)
    shell_radii = radii[peaks]
    if shell_radii[0] > 0:
        normalized_radii = shell_radii / shell_radii[0]
        print(f'  Normalized shell radii: {normalized_radii}')
        
        # Compare to hydrogen: 1, 4, 9, 16... (n^2)
        n_squared = np.arange(1, len(normalized_radii)+1)**2
        
        # Compare to Fibonacci: 1, 2, 3, 5, 8...
        fib = [1, 2, 3, 5, 8, 13][:len(normalized_radii)]
        
        print(f'  Hydrogen n^2: {n_squared}')
        print(f'  Fibonacci: {fib}')

# ============================================================
# REFINED FORCE LAW WITH BETTER STATISTICS
# ============================================================
print('\n' + '='*60)
print('REFINED FORCE LAW ANALYSIS')
print('='*60)

# Use all cells above threshold, compute gradients
threshold = M.mean() + 0.5 * M.std()
labeled, n_clusters = ndimage.label(M > threshold)
print(f'\nFound {n_clusters} mass clusters')

# Get cluster properties
clusters = []
for i in range(1, n_clusters + 1):
    mask = (labeled == i)
    y_coords, x_coords = np.where(mask)
    cy, cx = y_coords.mean(), x_coords.mean()
    total_M = M[mask].sum()
    total_E = E[mask].sum()
    
    # Compute local E gradient (proxy for force)
    E_grad_y, E_grad_x = np.gradient(E)
    grad_mag = np.sqrt(E_grad_y**2 + E_grad_x**2)
    avg_grad = grad_mag[mask].mean()
    
    clusters.append({
        'pos': (cy, cx),
        'M': total_M,
        'E': total_E,
        'grad': avg_grad,
        'size': mask.sum()
    })

# Compute pairwise interactions
pairs = []
for i in range(len(clusters)):
    for j in range(i+1, len(clusters)):
        c1, c2 = clusters[i], clusters[j]
        
        dy = c1['pos'][0] - c2['pos'][0]
        dx = c1['pos'][1] - c2['pos'][1]
        r = np.sqrt(dy**2 + dx**2)
        
        if r < 2:  # Skip overlapping
            continue
        
        # Force proxy: average gradient between clusters
        force = (c1['grad'] + c2['grad']) / 2
        
        # Mass product
        m1m2 = c1['M'] * c2['M']
        
        pairs.append({'r': r, 'force': force, 'm1m2': m1m2})

if len(pairs) > 5:
    rs = np.array([p['r'] for p in pairs])
    forces = np.array([p['force'] for p in pairs])
    m1m2s = np.array([p['m1m2'] for p in pairs])
    
    # Fit F = k * M1*M2 / r^n
    # log(F/M1M2) = log(k) - n*log(r)
    F_norm = forces / (m1m2s + 0.001)
    
    valid = (F_norm > 0) & (rs > 0)
    if valid.sum() > 3:
        log_r = np.log(rs[valid])
        log_F = np.log(F_norm[valid])
        
        slope, intercept = np.polyfit(log_r, log_F, 1)
        r_squared = np.corrcoef(log_r, log_F)[0,1]**2
        
        print(f'\nForce law fit: F/(M1*M2) ~ r^{slope:.3f}')
        print(f'R^2: {r_squared:.4f}')
        print(f'\nInterpretation:')
        print(f'  slope = -2.0 would be 3D gravity')
        print(f'  slope = -1.0 would be 2D gravity')
        print(f'  slope = {slope:.2f} suggests {abs(slope)+1:.1f}D effective gravity')

# ============================================================
# CONSERVATION LAW CHECK
# ============================================================
print('\n' + '='*60)
print('CONSERVATION LAW CHECK')
print('='*60)

# Total E, I, M
total_E = E.sum()
total_I = I.sum()
total_M = M.sum()

# PAC predicts: total potential = sum of actualized
# E + I should relate to M through conservation

print(f'\nTotal E: {total_E:.4f}')
print(f'Total I: {total_I:.4f}')
print(f'Total M: {total_M:.4f}')
print(f'\nE + I: {total_E + total_I:.4f}')
print(f'E - I: {total_E - total_I:.4f}')
print(f'(E + I) / M: {(total_E + total_I) / total_M:.4f}')
print(f'(E - I) / M: {(total_E - total_I) / total_M:.4f}')

# Check if E*M is conserved (energy-mass equivalence)
print(f'\nE * M: {total_E * total_M:.4f}')
print(f'sqrt(E * M): {np.sqrt(total_E * total_M):.4f}')

# Check PAC ratio
print(f'\nE / (E + I): {total_E / (total_E + total_I):.4f}')
print(f'Compare to 1/phi: {1/PHI:.4f}')
print(f'Compare to phi-1: {PHI - 1:.4f}')

print('\n' + '='*60)
print('ANALYSIS COMPLETE')
print('='*60)
