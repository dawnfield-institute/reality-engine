"""
Scale Invariance Test
Tests if c^2 = pi*phi/Xi holds across grid sizes
"""

from core.reality_service import RealityEngineService, EngineConfig
import numpy as np

PHI = 1.618033988749895
XI = 1.057
PREDICTED_C2 = np.pi * PHI / XI  # 4.8091

print('=' * 60)
print('SCALE INVARIANCE TEST')
print('Testing if c^2 = pi*phi/Xi holds across grid sizes')
print(f'Predicted c^2 = {PREDICTED_C2:.4f}')
print('=' * 60)
print()

grid_sizes = [(32, 16), (48, 24), (64, 32), (96, 48)]
results = []

for size in grid_sizes:
    print(f'Testing grid {size[0]}x{size[1]}...')
    config = EngineConfig(size=size)
    service = RealityEngineService(config=config)
    service.initialize(mode='big_bang')
    
    E_vals, M_vals = [], []
    steps = 2000
    for i in range(steps):
        service._step_engine()
        state = service.engine.current_state
        E_vals.append(state.actual.cpu().sum().item())
        M_vals.append(state.memory.cpu().sum().item())
    
    E, M = np.array(E_vals), np.array(M_vals)
    coeffs = np.polyfit(M[-500:], E[-500:], 1)
    c2 = -coeffs[0]
    
    # Also measure dE/dM correlation
    dE, dM = np.diff(E), np.diff(M)
    r = np.corrcoef(dE, dM)[0,1]
    
    error_pct = 100 * abs(c2 - PREDICTED_C2) / PREDICTED_C2
    results.append((size, c2, r, error_pct))
    print(f'  c^2 = {c2:.4f}, r(dE,dM) = {r:.4f}, error = {error_pct:.1f}%')
    print()

print('=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'Predicted c^2 = pi*phi/Xi = {PREDICTED_C2:.4f}')
print()
print('Grid         c^2        Error    Correlation')
print('-' * 50)
for size, c2, r, err in results:
    marker = ' <- MATCH' if err < 10 else ''
    print(f'{size[0]}x{size[1]}       {c2:8.4f}   {err:6.1f}%   {r:8.4f}{marker}')

# Check mean and std
c2_values = [r[1] for r in results]
print()
print(f'Mean c^2: {np.mean(c2_values):.4f}')
print(f'Std c^2:  {np.std(c2_values):.4f}')
print(f'CV:       {100*np.std(c2_values)/np.mean(c2_values):.1f}%')
