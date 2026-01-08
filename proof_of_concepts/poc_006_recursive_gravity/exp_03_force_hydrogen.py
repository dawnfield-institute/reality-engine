"""Test force laws and hydrogen spectrum comparison."""
from core.reality_service import RealityEngineService, EngineConfig
import numpy as np
from scipy import ndimage

PHI = 1.618033988749895
XI = 1.057

print('='*60)
print('TEST 2: FORCE LAW + TEST 3: HYDROGEN COMPARISON')
print('='*60)

config = EngineConfig(size=(50, 28))
service = RealityEngineService(config=config)
service.initialize(mode='big_bang')

# Run to equilibrium
for i in range(5000):
    service._step_engine()

state = service.engine.current_state
E = state.actual.cpu().numpy()
M = state.memory.cpu().numpy()

print(f'M stats: min={M.min():.4f}, max={M.max():.4f}, mean={M.mean():.4f}')

# Find mass centers with lower threshold
threshold = M.mean() + 0.5*M.std()
heavy_cells = np.where(M > threshold)

print(f'Found {len(heavy_cells[0])} cells above threshold')

# Cluster nearby cells
labeled, num_features = ndimage.label(M > threshold)
print(f'Found {num_features} distinct mass clusters')

# Get cluster centers and masses
clusters = []
for i in range(1, num_features+1):
    mask = labeled == i
    y_coords, x_coords = np.where(mask)
    cy, cx = y_coords.mean(), x_coords.mean()
    total_mass = M[mask].sum()
    total_E = E[mask].sum()
    clusters.append((cy, cx, total_mass, total_E))

clusters = sorted(clusters, key=lambda x: -x[2])[:20]  # Top 20 by mass

print()
print('='*60)
print('FORCE LAW ANALYSIS')
print('='*60)

forces = []
for i in range(len(clusters)):
    for j in range(i+1, len(clusters)):
        c1, c2 = clusters[i], clusters[j]
        
        r = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        if r < 2: continue
        
        # Energy gradient between clusters
        dE = abs(c1[3] - c2[3])
        
        # Product of masses
        m1m2 = c1[2] * c2[2]
        
        forces.append((r, dE, m1m2))

if forces:
    rs = np.array([f[0] for f in forces])
    dEs = np.array([f[1] for f in forces])
    m1m2s = np.array([f[2] for f in forces])
    
    # Fit: dE = k * (m1*m2) / r^n
    # log(dE) = log(k) + log(m1m2) - n*log(r)
    
    # First normalize by mass product
    dE_norm = dEs / (m1m2s + 0.001)
    
    log_r = np.log(rs)
    log_dE_norm = np.log(dE_norm + 0.0001)
    
    slope, intercept = np.polyfit(log_r, log_dE_norm, 1)
    r_sq = np.corrcoef(log_r, log_dE_norm)[0,1]**2
    
    print(f'Force scaling: F/M1M2 ~ r^{slope:.3f}')
    print(f'R^2: {r_sq:.4f}')
    print(f'Expected for gravity: slope ~ -2')
    print(f'Expected for 2D: slope ~ -1')

print()
print('='*60)
print('HYDROGEN SPECTRUM COMPARISON')
print('='*60)

# Get all mass values
masses = [c[2] for c in clusters]
masses = sorted(masses)

print(f'Cluster masses ({len(masses)} clusters):')
for i, m in enumerate(masses[:10]):
    print(f'  Cluster {i+1}: {m:.4f}')

if len(masses) >= 4:
    base = masses[0]
    ratios = [m/base for m in masses[:8]]
    
    print()
    print('Mass ratios (vs smallest):')
    for i, r in enumerate(ratios):
        hydrogen_ratio = (i+1)**2  # 1, 4, 9, 16...
        fib_ratio = [1, 2, 3, 5, 8, 13, 21, 34][min(i, 7)]
        print(f'  m_{i+1}/m_1 = {r:.3f}  (hydrogen n^2: {hydrogen_ratio}, Fib: {fib_ratio})')
    
    # Check for n^2 pattern (hydrogen)
    n_vals = np.arange(1, len(ratios)+1)
    n_squared = n_vals**2
    corr_h2 = np.corrcoef(ratios, n_squared[:len(ratios)])[0,1]
    
    # Check for Fibonacci pattern
    fib = [1, 2, 3, 5, 8, 13, 21, 34][:len(ratios)]
    corr_fib = np.corrcoef(ratios, fib)[0,1]
    
    print()
    print(f'Correlation with n^2 (hydrogen): {corr_h2:.4f}')
    print(f'Correlation with Fibonacci: {corr_fib:.4f}')
    
    if corr_fib > corr_h2:
        print('=> Fibonacci pattern is STRONGER')
    else:
        print('=> Hydrogen n^2 pattern is STRONGER')

print()
print('='*60)
print('TEST 4: L_CRITICAL DERIVATION')
print('='*60)

# The critical length L ~ 39 where c^2 = pi*phi/Xi
# Let's see if this relates to fundamental constants

L_crit = 39.0
print(f'Critical length L = {L_crit}')
print()
print('Possible relationships:')
print(f'  L / (2*pi) = {L_crit / (2*np.pi):.4f}')
print(f'  L / (pi*phi) = {L_crit / (np.pi * PHI):.4f}')
print(f'  L / (phi^3) = {L_crit / (PHI**3):.4f}')
print(f'  L / (10*phi) = {L_crit / (10*PHI):.4f}')
print(f'  L * Xi / (10*pi) = {L_crit * XI / (10*np.pi):.4f}')
print()
print(f'  sqrt(L) = {np.sqrt(L_crit):.4f}')
print(f'  sqrt(L) / phi = {np.sqrt(L_crit) / PHI:.4f}')
print(f'  L^(1/3) = {L_crit**(1/3):.4f}')
print()

# Check if L relates to the optimal grid dimensions
print('Grid dimension analysis:')
print(f'  50 / phi = {50/PHI:.4f}')
print(f'  28 / phi = {28/PHI:.4f}')
print(f'  50/28 = {50/28:.4f} (compare to phi={PHI:.4f})')
print(f'  sqrt(50*28) = {np.sqrt(50*28):.4f}')
